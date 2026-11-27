"""Step 3 numerical invariants and label-blind behavior on acoustic tapes."""
from dataclasses import replace
from itertools import product
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from reader.belief_ipe import PAIRS, BeliefIPE, gaussian_fourier
from reader.dynamic_factor_graph import DynamicFactorGraphReader, FrameEstimate, NULL_ROLE, joint_pullback_covariance
from reader.joint_association import joint_beam
from reader.split_prior import DynamicScanSimulator, build_split_prior
from reader.station_classifier import (SegmentRecord, _FrameTruth, conformal_rank_cutoff,
                                       fit_segment_certifier, make_segment_prediction)
from sim.anatomy_assets import FAMILY_HELD_OUT_ATLASES, LOAO_OPEN3D, AnatomySource, TrainingAnatomyBank
from sim.material_chart import RightLegMaterialChart
from sim.metric_graph_gp import MetricGraphPrior, VascularMetricGraph
from sim.physics_ultrasound_kernel import (PhysicsUltrasoundKernel, ProbePose, SpecularReflector,
                                           UltrasoundTruth, VesselTruth)
from sim.smplx_dynamic_atlas import NEUTRAL_BETA, SMPLXLegKinematics


@pytest.fixture(scope='module')
def kinematics():
    return SMPLXLegKinematics()


@pytest.fixture(scope='module')
def chart():
    return RightLegMaterialChart()


def _prior(chart, roles=(0, 1, 2), around_h=-635.):
    heights = around_h + np.arange(-20., 21., 5.)
    angles = np.array([1.65, 1.17, 2.14])
    centers = np.full((len(heights), 6, 3), np.nan)
    for i, r in enumerate(roles):
        centers[:, r] = chart.xi_to_world(np.column_stack((np.full(len(heights), angles[i]), heights,
                                                           np.full(len(heights), .32 + .10 * i))))
    support = np.isfinite(centers[..., 0])
    source = AnatomySource('test_fixture', heights, centers, support, support[1:] & support[:-1],
                           np.where(support, 'fixture_lumen', ''), {})
    graph = VascularMetricGraph.from_source(source, {}, domain='thigh_nysora')
    return MetricGraphPrior(graph, chart=chart)


def test_belief_ipe_exact_shared_gauge_and_pose_invariance(chart, kinematics):
    prior = _prior(chart)
    encoder = BeliefIPE(prior)
    rng = np.random.default_rng(18)
    mean = prior.sample_latent(rng)
    shifted = mean.copy()
    shifted[prior.layout.shared] += rng.normal(0., 100., size=prior.layout.shared.stop)
    h = -634.13  # Off-knot shared interpolation must also cancel exactly.
    factor = splu(prior.precision)
    ids, mu, cov = encoder.contrasts(h, mean, factor=factor)
    next_ids, next_mu, next_cov = encoder.contrasts(h, shifted, factor=factor)
    assert_array_equal(ids, next_ids)
    assert_allclose(next_mu, mu, atol=1e-12, rtol=0.)
    assert_allclose(next_cov, cov, atol=1e-12, rtol=0.)
    a = encoder.features(h, mean, factor=factor)
    b = encoder.features(h, shifted, factor=factor)
    assert_allclose(a, b, atol=1e-12, rtol=0.)
    world = prior.decode_knots_to_world(mean)
    posed, _, _ = kinematics.forward_pose_canonical(np.deg2rad([16., -8., 5., 28.]), world)
    assert np.max(np.abs(posed - world)) > 10.
    assert_allclose(a, encoder.features(h, mean, factor=factor), atol=1e-12, rtol=0.)


def test_belief_ipe_propagates_full_cross_covariance(chart):
    prior = _prior(chart)
    encoder = BeliefIPE(prior)
    rho = .81
    sigma = np.eye(prior.layout.size)
    i, j = prior.layout.relative.start, prior.layout.relative.start + 3
    sigma[i, j] = sigma[j, i] = rho
    h = prior.graph.h_mm[0]
    ids, mean, covariance = encoder.contrasts(h, prior.mean, covariance=sigma)
    for pair, actual in zip(PAIRS[ids], covariance, strict=True):
        a, b = [prior.center_query(h, role) for role in pair]
        contrast = (a.operator - b.operator).toarray()
        assert_allclose(actual, contrast @ sigma @ contrast.T, atol=1e-13, rtol=0.)
        if tuple(pair) == (0, 1):
            assert actual[0, 0] < (contrast @ contrast.T)[0, 0]
    base = gaussian_fourier(mean, covariance)
    enlarged = gaussian_fourier(mean, covariance + np.diag([.01, 4., .01]))
    assert np.all(np.linalg.norm(enlarged, axis=-1) <= np.linalg.norm(base, axis=-1) + 2e-15)
    assert np.any(np.linalg.norm(enlarged, axis=-1) < np.linalg.norm(base, axis=-1) - .01)


def test_kirchhoff_generalized_eigenfunctions(chart):
    prior = _prior(chart)
    ipe = BeliefIPE(prior)
    mass = prior.graph.lumped_mass_mm[:, None]
    psi = ipe.eigenfunctions
    assert psi.shape[1] == 8
    assert_allclose(psi.T @ (mass * psi), np.eye(8), atol=2e-11, rtol=0.)
    assert_allclose(prior.kirchhoff_laplacian() @ psi, mass * psi * ipe.eigenvalues, atol=2e-11, rtol=0.)


def test_anchored_assignment_recovers_full_joint_optimum_with_clutter():
    positions = np.array([[-8., 11.], [-1., 23.], [7., 27.], [31., 53.]])
    joint = np.pad(np.kron(np.ones((4, 4)), np.diag([900., 400.])) + np.eye(8), ((2, 0), (2, 0)))
    shift = np.array([24., -10.])
    masks = np.array([0, 1, 2, 4, 8], dtype=np.intp)
    for echoes in (positions[:3] + shift, np.vstack(([-80., 94.], positions[:3] + shift))):
        base = np.zeros((len(echoes), len(masks)))
        base[:, 0] = -8.
        errors = echoes[:, None, :] - positions
        expectations = []
        for path in product(range(len(masks)), repeat=len(echoes)):
            slots = np.flatnonzero(path)
            codes = np.asarray(path)[slots]
            if len(np.unique(codes)) < len(slots):
                continue
            idx = (2 * codes[:, None] + np.arange(2)).ravel()
            residual = errors[slots, codes - 1].ravel()
            c = joint[np.ix_(idx, idx)]
            score = base[np.arange(len(path)), path].sum()
            if len(slots):
                score -= (.5 * (residual @ np.linalg.solve(c, residual) + np.linalg.slogdet(c)[1])
                          + len(slots) * np.log(2. * np.pi))
            expectations.append((score, path))
        expected_score, expected_path = max(expectations)
        paths, scores = joint_beam(base, errors, joint, masks, 8)
        assert tuple(paths[0]) == expected_path
        assert_allclose(scores[0], expected_score, atol=1.e-10, rtol=0.)
        if len(echoes) == 4:
            assert expected_path[0] == 0


def test_joint_dynamic_pullback_psd_and_relative_pose_suppression(chart, kinematics):
    h = -650.
    material = np.array([[1.38, h, -.5], [1.48, h, -.4]])
    _, _, jq, _ = kinematics.pose_points_and_material_frame(NEUTRAL_BETA, np.deg2rad([14., 4., 2., 15.]),
                                                           chart, material)
    joint_jq = jq[:, [0, 2]]
    pose_cov = np.diag(np.deg2rad([2., 2., 3., 3.])**2)
    noise = np.repeat(np.diag([.8**2, 1.])[None], 2, axis=0)
    r = joint_pullback_covariance(noise, joint_jq, pose_cov)
    assert np.linalg.eigvalsh(r).min() > 0.
    assert_allclose(r[:2, 2:], joint_jq[0] @ pose_cov @ joint_jq[1].T, atol=1e-13, rtol=0.)
    contrast = np.column_stack((np.eye(2), -np.eye(2)))
    pose_error_contrast = contrast @ r @ contrast.T - noise.sum(axis=0)
    assert np.linalg.eigvalsh(pose_error_contrast).min() >= -1e-12
    assert np.trace(pose_error_contrast) < .1 * np.trace(r[:2, :2] - noise[0])
    assert joint_pullback_covariance(np.empty((0, 2, 2)), np.empty((0, 2, 4)), pose_cov).shape == (0, 0)


def test_pose_and_world_posterior_equal_dense_joint_conditioning():
    rng = np.random.default_rng(4)
    hx, jq, jz = (rng.normal(size=(6, width)) for width in (3, 4, 2))
    px, pq, pz = np.diag([2., 3., 5.]), np.diag([.02, .03, .05, .01]), np.diag([1., 2.])
    frame = SimpleNamespace(q=rng.normal(size=4), pose_cov=pq, probe=SimpleNamespace(rotation=np.eye(3)))
    prior = SimpleNamespace(mean=rng.normal(size=3), factor=splu(csc_matrix(np.linalg.inv(px))), ou_cov=pz)
    linearization = rng.normal(size=3)
    keep = [0, 2, 3, 5]
    t = np.eye(4)[:2]
    observed = np.asarray(keep)[np.any(t > 0., axis=0)]
    mu = rng.normal(size=(2, 3))
    sensor = .6 * np.eye(4)
    pred = SimpleNamespace(world=mu, H_world=hx, H=hx[keep], J_world_q=jq, J_q=jq[keep],
                           J_world_ou=jz, J_ou=jz[keep],
                           variance=sensor + jq[keep] @ pq @ jq[keep].T + jz[keep] @ pz @ jz[keep].T)
    assoc = SimpleNamespace(operator=t, residual=np.array([.7, -.8]))
    design = SimpleNamespace(cols=np.arange(3))
    actual = DynamicFactorGraphReader._condition_world(frame, design, prior, pred, assoc, linearization)

    delta = prior.mean - linearization
    world_cov = hx @ px @ hx.T + jq @ pq @ jq.T + jz @ pz @ jz.T
    innovation = world_cov[np.ix_(observed, observed)] + sensor[:2, :2]
    correction = np.linalg.solve(innovation, assoc.residual - hx[observed] @ delta)
    world_cross, pose_cross = world_cov[:, observed], pq @ jq[observed].T
    expected = ((mu.ravel() + hx @ delta + world_cross @ correction).reshape(2, 3),
                world_cov - world_cross @ np.linalg.solve(innovation, world_cross.T),
                frame.q + pose_cross @ correction,
                pq - pose_cross @ np.linalg.solve(innovation, pose_cross.T))
    for a, b in zip(actual, expected, strict=True):
        assert_allclose(a, b, atol=1.e-12, rtol=0.)


def test_neutral_skin_fast_equivalence_and_dynamic_contact_height(chart, kinematics):
    reader = DynamicFactorGraphReader(_prior(chart), kinematics)
    heights = np.linspace(-875., -480., 18)
    material = np.column_stack((np.linspace(.6, 2.5, len(heights)), heights, np.zeros(len(heights))))
    points = chart.tilde_xi_to_world(material)
    for angles in ([0., 0., 0., 0.], [25., 10., -8., 35.], [25., -10., 8., 35.], [13., 6., 4., 17.]):
        q = np.deg2rad(angles)
        transforms = reader._skin.transforms(q)
        expected = kinematics._deform_canonical(q, points, NEUTRAL_BETA, gradient=True)
        actual = reader._skin.deform(points, transforms)
        for a, e in zip(actual, expected, strict=True):
            assert_allclose(a, e, atol=1e-9, rtol=1e-11)
        contact = chart.xi_to_world(np.column_stack((material[:, 0], heights, np.zeros(len(heights)))))
        posed, _, _ = kinematics.forward_pose_canonical(q, contact)
        inverted = [reader._skin.invert_height(origin, transforms) for origin in posed]
        assert_allclose(inverted, heights, atol=.5, rtol=0.)


def _probe(target, tilt=0., *, dwell=0.):
    a = np.deg2rad(tilt)
    depth = np.array([np.sin(a), 0., -np.cos(a)])
    elevation = np.array([0., 1., 0.])
    lateral = np.cross(elevation, depth)
    return ProbePose(target - 27. * depth, np.column_stack((lateral, elevation, depth)), dwell_s=dwell)


def test_byte_identical_reader_under_truth_role_and_order_permutation(chart, kinematics):
    prior = _prior(chart, roles=(0, 1))
    h = -635.
    xyz = chart.tilde_xi_to_world(np.array([prior.center_query(h, r).nominal for r in (0, 1)]))
    probe = _probe(xyz.mean(axis=0), dwell=.4)
    vessels = [VesselTruth(xyz[0], 6.5, .20, .02, role=0, noise_key=120),
               VesselTruth(xyz[1], 7.0, .04, .13, role=1, noise_key=766)]
    permuted = [replace(vessels[1], role=0), replace(vessels[0], role=1)]
    kernel = PhysicsUltrasoundKernel(42)
    reader = [DynamicFactorGraphReader(prior, kinematics, max_branches=2) for _ in range(2)]
    for i in range(3):
        a, b = [kernel.render(UltrasoundTruth(v), probe, 4.5, i) for v in (vessels, permuted)]
        assert a.echoes.tobytes() == b.echoes.tobytes()
        f = [r.step(o, probe, 4.5, np.zeros(4), np.zeros((4, 4)), .1) for r, o in zip(reader, (a, b), strict=True)]
        assert f[0].role_probabilities.tobytes() == f[1].role_probabilities.tobytes()
        assert f[0].centers_mm.tobytes() == f[1].centers_mm.tobytes()
    for branch in reader[0]._branches:
        direct = splu(branch.precision)
        rhs = np.random.default_rng(12).standard_normal((prior.layout.size, 3))
        assert_allclose(branch.factor.solve(rhs), direct.solve(rhs), atol=2e-6, rtol=1e-7)
        if branch.precision is not prior.precision:  # A retained coarse seed may have no matches.
            assert_allclose(branch.mean, direct.solve(branch.information), atol=2e-6, rtol=1e-7)


def test_retroactive_ghost_rejection_after_rewindow_and_compression(chart, kinematics):
    prior = _prior(chart, roles=(0,))
    nominal = prior.center_query(-635., 0).nominal
    vessel_address = nominal + np.array([.62, 0., 0.])
    ghost, true = chart.tilde_xi_to_world(np.stack((nominal, vessel_address)))
    early = _probe(ghost)
    late = _probe(true, 45., dwell=.43)
    depth = early.rotation[:, 2]
    reflector = SpecularReflector(ghost - 12. * depth, -depth, ghost - 24. * depth, diam_mm=6., noise_key=77)
    vessel = VesselTruth(true, 6.2, .23, .01, role=0, noise_key=88)
    # A small fascia patch intercepts only the early lumen ray, not the ghost ray.
    point = .5 * (early.origin_mm + true)
    x, y = early.rotation[:, :2].T
    fascia = np.array([[point - 1.5 * (x + y), point + 1.5 * (x - y), point + 1.5 * y]])
    kernel = PhysicsUltrasoundKernel(seed=12)
    reader = DynamicFactorGraphReader(prior, kinematics)
    obs = kernel.render(UltrasoundTruth((vessel,), fascia_triangles_mm=fascia, reflectors=(reflector,)), early, 2.5, 0)
    assert len(obs.echoes) == 1
    assert len(kernel.render(UltrasoundTruth((vessel,), fascia_triangles_mm=fascia), early, 2.5, 0).echoes) == 0
    f = reader.step(obs, early, 2.5, np.zeros(4), np.zeros((4, 4)), .1)
    assert np.argmax(f.role_probabilities[0]) == 0
    for i in range(1, 9):
        obs = kernel.render(UltrasoundTruth((vessel,), reflectors=(reflector,)), late, 4.7, i)
        assert len(obs.echoes) == 1
        reader.step(obs, late, 4.7, np.zeros(4), np.zeros((4, 4)), .1)
    relabeled = reader.relabel_history()
    assert np.argmax(relabeled[0].role_probabilities[0]) == NULL_ROLE
    assert all(np.argmax(f.role_probabilities[0]) == 0 for f in relabeled[1:])


def test_imports_never_load_holdout():
    env = dict(os.environ)
    env.pop('AGPS_EVAL_HOLDOUT_ALLOWED', None)
    cmd = ('import sys; from reader import belief_ipe, split_prior, dynamic_factor_graph, '
           'station_classifier, baselines, eval_reader; assert "sim.holdout_eval_assets" not in sys.modules')
    subprocess.run((sys.executable, '-c', cmd), check=True, env=env, timeout=15)


def test_split_conformal_rank_bound_on_exchangeable_segments():
    scores = np.random.default_rng(19).permutation(np.linspace(.1, 1., 201))
    violations = []
    for i, score in enumerate(scores):
        rank, cutoff = conformal_rank_cutoff(np.delete(scores, i), alpha=.2)
        assert rank == 161
        violations.append(score > cutoff)
    assert sum(violations) == 40
    assert np.mean(violations) <= .2
    assert conformal_rank_cutoff([.1, .2, .3], alpha=.2) == (4, float('inf'))


def test_whole_segment_conformal_can_certify_and_checks_exact_merges():
    source = LOAO_OPEN3D.train_source_ids[0]
    exact = np.zeros((1, 64))
    exact[0, [0, 3]] = [.015, .985]
    estimate = FrameEstimate(np.array([[.4925, .4925, 0., 0., 0., 0., .015]]), np.zeros((6, 3)),
                             np.zeros((6, 2)), np.broadcast_to(np.eye(2), (6, 2, 2)),
                             np.zeros(5), -610., np.empty(0), np.zeros((6, 3, 3)), exact)
    prediction = make_segment_prediction([estimate])

    def record(index, dx, mask=3):
        centers = np.zeros((6, 2))
        centers[:2, 0] = dx
        return SegmentRecord(source, index, prediction, (_FrameTruth(np.array([mask]), centers),))

    records = [record(i, dx) for i, dx in enumerate(np.linspace(0., 2., 200))]
    certifier = fit_segment_certifier(records, LOAO_OPEN3D)
    audit = certifier.audit(records=[record(200, 1.), record(201, 1., mask=1), record(202, 9.)])
    assert audit['strata']['well_observed']['rank'] == 161
    assert audit['certified_segments'] == 3
    assert audit['false_certification']['failures'] == audit['miscoverage']['failures'] == 2
    diffuse = replace(estimate, identity_probabilities=np.full((1, 64), 1. / 64.))
    assert not certifier.verify(certifier.predict_set(make_segment_prediction([diffuse])))


def test_split_prior_and_scan_tape_isolation(chart, kinematics):
    bank = TrainingAnatomyBank.load()
    split = LOAO_OPEN3D

    class TrainOnly:
        native_graph = bank.native_graph

        def source(self, source_id):
            assert source_id in split.train_source_ids
            return bank.source(source_id)

    prior = build_split_prior(TrainOnly(), split, chart=chart)
    assert prior.provenance['train_source_ids'] == split.train_source_ids
    assert prior.provenance['extrapolated_dfa_heights_mm']
    assert prior.relative_std_scale.max() > 1.
    assert all(group <= set(split.train_source_ids) for group in (
        set(roles) for row in prior.provenance['direct_source_ids_by_row_role'] for roles in row))
    ct = build_split_prior(bank, FAMILY_HELD_OUT_ATLASES, chart=chart)
    assert ct.graph.h_mm.min() == -560.
    assert ct.graph.h_mm.max() == -480.
    assert not ct.station_windows['S5_POPLITEAL'].size
    for pooled in (prior, ct):
        windows = pooled.station_windows
        shared = np.intersect1d(windows['S2_SFJ_MICKEY'], windows['S3_ART_BIF'])
        assert shared.size and np.isin(shared, windows['S1_CFV_CFA']).all()
    sim = DynamicScanSimulator(bank, split, kinematics=kinematics)
    source = split.test_source_ids[0]
    a, b = (sim.episode(source, 42, benchmark='dynamic', frames=3) for _ in range(2))
    assert a.provenance['clinical_station_labels_available'] is False
    for f, g in zip(a.frames, b.frames, strict=True):
        assert f.obs.echoes.tobytes() == g.obs.echoes.tobytes()
        assert f.echo_targets == g.echo_targets
        assert f.q_hat.tobytes() == g.q_hat.tobytes()
        assert np.deg2rad(2.)**2 <= np.diag(f.Sigma_q).min()
    assert any(any(f.echo_targets) for f in a.frames)
    for benchmark, offset in (('dynamic', 0.), ('offset', 40.)):
        frames = sim.episode(source, 42, benchmark=benchmark, frames=8, offset_mm=offset).frames
        assert sum(4.3 <= f.f_contact <= 4.5 for f in frames) >= 2
        assert sum(f.probe_pose.dwell_s >= .35 for f in frames) >= 2

