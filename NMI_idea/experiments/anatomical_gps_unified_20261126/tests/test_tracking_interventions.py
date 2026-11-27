"""Tracking interventions are offline, model-consistent, and denominator honest."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from reader import dev_tracking_interventions as diag
from reader.baselines import Full_AGPS_Reader
from reader.dynamic_factor_graph import _score_associations
from reader.eval_reader import _run, _tape_truth
from reader.split_prior import DynamicScanSimulator, _ScanMotion, _SupervisedKernel, build_split_prior
from sim.anatomy_assets import LOAO_OPEN3D, TrainingAnatomyBank
from sim.physics_ultrasound_kernel import PhysicsUltrasoundKernel, ProbePose, UltrasoundObservation


def _fixture(image, echoes):
    image = np.array(image, dtype=float)
    # Retain nonzero shared covariance so the forced score must use the joint
    # observation model, not the sum of independent per-echo likelihoods.
    covariance = np.eye(image.size) * 1.5 + .15
    pred = SimpleNamespace(image=image, visible=np.linspace(.50, .75, len(image)))
    frame = SimpleNamespace(echoes=np.asarray(echoes, dtype=float).reshape(-1, 5), force=4.,
                            probe=ProbePose(np.zeros(3), np.eye(3), 0., .4))
    return frame, pred, covariance


def _force(groups, roles, frame, pred, covariance):
    supported = tuple(reason is None for reason in diag._unsupported_slots(groups, roles))
    known, admissible = diag._known_association(groups, supported, frame, roles, pred, covariance, 6., 3., construct=True)
    return known, supported, admissible


def _real(frame, roles, pred, covariance):
    return _score_associations(frame.echoes, roles, pred.image, covariance, pred.visible, 4.7, frame.force,
                               frame.probe.dwell_s, 6., 3.)


def test_forced_singleton_merge_and_null_use_exact_production_sensor_likelihood():
    roles = np.arange(3)
    groups = ((), (0, 1), (2,))
    frame, pred, covariance = _fixture([[0., 20.], [2., 20.], [12., 25.]],
                                      [[-60., 65., 6., -.15, .02], [1.5, 20.2, 8., -.18, .08],
                                       [12.1, 24.8, 6., -.38, .012]])
    known, supported, admissible = _force(groups, roles, frame, pred, covariance)
    real = next(a for a in _real(frame, roles, pred, covariance) if a.groups == groups)
    assert supported == admissible == (True, True, True)
    assert known.groups == groups
    assert_array_equal(known.echoes, [1, 2])
    assert_allclose(known.operator, real.operator)
    assert_allclose(known.residual, frame.echoes[[1, 2], :2].ravel() - known.operator @ pred.image.ravel())
    assert_allclose(known.residual, real.residual)
    assert known.score == pytest.approx(real.score)
    assert known.acoustic_score == pytest.approx(real.acoustic_score)


def test_connected_triple_and_empty_frame_and_search_gate_bypass():
    roles = np.arange(3)
    groups = ((0, 1, 2),)
    frame, pred, covariance = _fixture([[0., 20.], [2., 20.], [4., 20.]], [[2., 20., 9., -.2, .08]])
    known, supported, admissible = _force(groups, roles, frame, pred, covariance)
    real = next(a for a in _real(frame, roles, pred, covariance) if a.groups == groups)
    assert supported == admissible == (True,)
    assert known.groups == groups
    assert_allclose(known.operator, real.operator)
    assert known.score == pytest.approx(real.score)
    pred.image[-1, 0] = 40.
    known, supported, admissible = _force(groups, roles, frame, pred, covariance)
    assert supported == (True,) and admissible == (False,)
    assert known.groups == groups and np.count_nonzero(known.operator[0]) == 3
    assert np.isfinite(known.score) and not any(a.groups == groups for a in _real(frame, roles, pred, covariance))

    frame.echoes = np.empty((0, 5))
    known, supported, admissible = _force((), roles, frame, pred, covariance)
    assert known.groups == supported == admissible == ()
    assert known.operator.shape == (0, 6) and not known.echoes.size and not known.residual.size
    assert known.score == pytest.approx(np.log1p(-pred.visible).sum())
    assert diag._identity(np.array([], dtype=int), np.array([], dtype=int))['micro_f1'] is None
    assert diag._tracking(np.empty((0, 3)), 0)['complete_rmse_mm']['world'] is None


def test_unsupported_merge_keeps_all_rendered_roles_in_fixed_denominators():
    groups, roles = ((0,), (1, 5), ()), np.arange(2)
    frame, pred, covariance = _fixture([[0., 20.], [10., 20.]],
                                      [[.5, 20., 6., -.3, .01], [10., 20., 8., -.13, .04], [-6., 24., 6., 0., .01]])
    known, supported, admissible = _force(groups, roles, frame, pred, covariance)
    assert supported == admissible == (True, False, True)
    assert_array_equal(known.echoes, [0])
    assert_array_equal(known.operator, [[1., 0., 0., 0.], [0., 1., 0., 0.]])
    assert diag._unsupported_slots(groups, roles) == [None, {'missing_design_roles': [5], 'reused_roles': []}, None]
    assert diag._unsupported_slots(((0,), (0, 1)), roles) == [{'missing_design_roles': [], 'reused_roles': [0]}] * 2
    # The discarded echo supplies neither geometry nor NULL likelihood.
    dropped = SimpleNamespace(echoes=frame.echoes[[0, 2]], force=frame.force, probe=frame.probe)
    without, _, _ = _force(((0,), ()), roles, dropped, pred, covariance)
    assert without.score == pytest.approx(known.score)
    assert without.acoustic_score == pytest.approx(known.acoustic_score)
    identity = diag._identity(np.array([1, 34, 0]), np.array([1, diag._UNAVAILABLE, 0]))
    assert identity['rendered_echoes'] == 3 and identity['abstentions'] == 1
    assert identity['exact_mask_accuracy_on_all_rendered'] == pytest.approx(2 / 3)
    for role in ('MAIN_ARTERY', 'DFA'):
        assert identity['by_role'][role]['truth_memberships'] == 1
        assert identity['by_role'][role]['f1'] == 0.
        assert identity['by_role'][role]['f1_on_decisions'] is None
    assert identity['by_role']['NULL']['predicted_memberships'] == 1
    measured = diag._tracking(np.array([[3., 4., 0.], [np.nan] * 3, [np.nan] * 3]), 3)
    assert measured['rendered'] == 3 and measured['measured'] == 1
    assert measured['conditional_rmse_mm'] == {'world': 5., 'image_plane': 3., 'elevation': 4.}
    assert measured['complete_rmse_mm'] == dict.fromkeys(('world', 'image_plane', 'elevation'))
    truth = SimpleNamespace(observed=np.ones(3, dtype=bool), echo_masks=(np.array([1, 34, 0]),))
    cell = SimpleNamespace(residuals=np.zeros((3, 3)), identity=(np.array([1, 34, 0]),),
                           trace=[{'supported_exact_slots': [True, False, True]}])
    criteria = diag._tape_criteria(truth, cell)
    assert criteria['all_rendered_masks_match_reference']
    assert criteria['all_rendered_role_centers_strictly_below_6_mm']
    assert not criteria['joint_representable_masks_match_and_all_role_centers_below_6_mm']


def test_generated_pose_and_ghost_supervision_stay_off_public_input(monkeypatch):
    rendered, motion = [], []
    kernel_render, sample_motion = _SupervisedKernel.render_supervised, _ScanMotion.sample

    def render(self, truth, pose, force, step, roles):
        obs, targets, ghosts = kernel_render(self, truth, pose, force, step, roles)
        plain = PhysicsUltrasoundKernel(self.seed, lateral_half_width_mm=28.).render(truth, pose, force, step)
        assert obs.echoes.tobytes() == plain.echoes.tobytes()
        rendered.append((obs, pose, force, targets, ghosts))
        return obs, targets, ghosts

    def sample(self, t):
        values = sample_motion(self, t)
        motion.append(values)
        return values

    monkeypatch.setattr(_SupervisedKernel, 'render_supervised', render)
    monkeypatch.setattr(_ScanMotion, 'sample', sample)
    source = LOAO_OPEN3D.train_source_ids[0]
    bank = TrainingAnatomyBank.load()
    sim = DynamicScanSimulator(bank, LOAO_OPEN3D)
    episode = sim.episode(source, 42, frames=1)
    frame, = episode.frames
    obs, pose, force, targets, ghosts = rendered[0]
    motion_truth, tracker = motion[0]
    supervision = frame.supervision
    assert supervision.echo_targets == targets and supervision.echo_ghosts == ghosts
    assert len(ghosts) == len(targets) == len(obs.echoes) and all(type(g) is bool for g in ghosts)
    assert supervision.acquisition_h_mm == episode.h_start_mm
    assert supervision.true_q_rad.shape == (4,)
    assert_array_equal(supervision.true_q_rad, motion_truth.q)
    with pytest.raises(ValueError):
        supervision.true_q_rad.setflags(write=True)
    assert frame.q_hat.tobytes() == tracker.q_hat.tobytes()
    assert frame.Sigma_q.tobytes() == tracker.Sigma_q.tobytes()

    class Spy:
        def reset(self):
            self.calls = []

        def step(self, *args):
            self.calls.append(args)

    spy = Spy()
    _run(spy, episode)
    (inputs,) = spy.calls
    for actual, public in zip(inputs, (obs, pose, force, frame.q_hat, frame.Sigma_q, frame.dt), strict=True):
        assert actual is public or actual == public
    assert vars(obs).keys() == {'echoes'}
    prior = build_split_prior(bank, LOAO_OPEN3D, chart=sim._chart)
    expected, _ = _run(Full_AGPS_Reader(prior, sim.kinematics), episode)
    reader = diag._DiagnosticReader(prior, sim.kinematics)
    digest = diag._tape_hash(episode)
    truth = _tape_truth(episode)
    assert_array_equal(truth.centers[0], supervision.centers_mm)
    for cell in diag._CELLS:
        result = diag._replay(reader, episode, truth, cell, digest)
        assert result.tape_sha256 == digest
        actual, _ = reader._history[-1]
        if cell.endswith('_true_pose'):
            assert_array_equal(actual.q, supervision.true_q_rad)
            assert_array_equal(actual.pose_cov, np.zeros((4, 4)))
        else:
            assert_array_equal(actual.q, frame.q_hat)
            assert_array_equal(actual.pose_cov, frame.Sigma_q)
        if cell == 'real_tracker':
            assert_allclose(reader._latest.centers_mm, expected[0].centers_mm, atol=0., rtol=0.)
            assert_allclose(reader._latest.identity_probabilities, expected[0].identity_probabilities, atol=0., rtol=0.)
        if cell.startswith('forced_'):
            available = np.array(result.trace[0]['supported_exact_slots'], dtype=bool)
            assert_array_equal(result.identity[0][available], truth.echo_masks[0][available])
            assert np.all(result.identity[0][~available] == diag._UNAVAILABLE)
    original = reader._latest
    altered_supervision = replace(supervision, acquisition_h_mm=supervision.acquisition_h_mm + 1.,
                                  centers_mm=supervision.centers_mm + 50.)
    altered = replace(episode, frames=(replace(frame, supervision=altered_supervision),))
    altered_digest = diag._tape_hash(altered)
    assert altered_digest != digest
    diag._replay(reader, altered, _tape_truth(altered), 'forced_true_pose', altered_digest)
    assert_allclose(reader._latest.centers_mm, original.centers_mm, atol=0., rtol=0.)
    assert reader._latest.h_mm == original.h_mm

    empty_obs = UltrasoundObservation(np.frombuffer(b'', dtype=float).reshape(0, 5))
    empty_frame = replace(frame, obs=empty_obs, supervision=replace(supervision, echo_targets=(), echo_ghosts=()))
    empty_tape = replace(episode, frames=(empty_frame,))
    empty_truth, empty_digest = _tape_truth(empty_tape), diag._tape_hash(empty_tape)
    for cell in diag._CELLS:
        result = diag._replay(reader, empty_tape, empty_truth, cell, empty_digest)
        assert result.tape_sha256 == empty_digest
        assert len(result.trace) == 1 and result.trace[0]['admissible_exact_slots_any_seed'] == []
        assert result.identity[0].shape == (0,) and np.isnan(result.residuals).all()
        assert not any(diag._tape_criteria(empty_truth, result).values())
        if not cell.startswith('forced_'):
            assert result.trace[0]['real_search_funnel']['retained']['true_prefix']
    assert len(rendered) == len(motion) == 1  # four cells replayed one physical render


def test_development_cli_refuses_holdout_environment_and_existing_output(tmp_path, monkeypatch):
    out = tmp_path / 'existing.json'
    out.write_text('previous evidence')
    monkeypatch.setenv(diag.HOLDOUT_ENV, '0')
    with pytest.raises(SystemExit) as denied:
        diag.main(['--out', str(out)])
    assert denied.value.code == 2
    monkeypatch.delenv(diag.HOLDOUT_ENV)
    with pytest.raises(SystemExit) as denied:
        diag.main(['--out', str(out)])
    assert denied.value.code == 2
    assert out.read_text() == 'previous evidence'
