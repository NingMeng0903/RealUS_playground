"""Bounded detached orchestration checks; no acceptance seeds or robot I/O."""
from dataclasses import replace

import numpy as np
import pytest

from peirastic.apps import contact_qp_experiment as experiment
from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.types import ContactObservation, ContactStatus
from peirastic.contact_qp.qp import QpResult
from peirastic.contact_qp.coverage import PathCoverage


def trace(path, alpha, omega):
    return dict(path_m=np.asarray(path), alpha=np.asarray(alpha), omega_measured=np.asarray(omega))


def test_coverage_unions_revisits_without_double_counting_or_counting_reverse_motion():
    coverage = PathCoverage(.06)
    for i, path in enumerate((0., .02, .01, .03, .025, .04)):
        assert coverage.observe(path, True, i*.005)
    assert coverage.valid_length_m == pytest.approx(.04)
    assert coverage.missing_length_m == pytest.approx(.02)
    assert coverage.maximum_path_m == .04
    assert coverage.intervals == [(0., .04)]


def test_coverage_revisit_can_fill_a_previous_acquisition_gap():
    coverage = PathCoverage(.06)
    for i, (path, valid) in enumerate(((0., True), (.01, True), (.02, False),
                                      (.03, True), (.04, True))):
        coverage.observe(path, valid, i*.005)
    assert coverage.valid_length_m == pytest.approx(.02)
    assert coverage.invalid_forward_travel_m == pytest.approx(.02)
    coverage.observe(.01, True, .025)  # reversal earns no new forward coverage
    coverage.observe(.035, True, .030)
    assert coverage.valid_length_m == pytest.approx(.04)
    assert coverage.invalid_forward_travel_m == pytest.approx(.02)


def test_coverage_does_not_bridge_timestamp_gap_or_invalid_measurement():
    coverage = PathCoverage(.06, max_gap_s=.02)
    coverage.observe(0., True, 0.)
    coverage.observe(.01, True, .01)
    coverage.observe(.03, True, .1)  # unknown .02 m interval cannot be inferred
    coverage.observe(.04, True, .11)
    assert not coverage.observe(.045, True, .115, measurement_valid=False)
    coverage.observe(.05, True, .12)
    coverage.observe(.06, True, .13)
    assert coverage.valid_length_m == pytest.approx(.03)
    assert coverage.missing_length_m == pytest.approx(.03)
    assert coverage.maximum_path_m == .06


def test_coverage_rejects_duplicate_and_reordered_samples_and_clips_path_extent():
    coverage = PathCoverage(.06)
    coverage.observe(-.01, True, 1.)
    coverage.observe(.01, True, 1.01)
    assert not coverage.observe(.05, True, 1.01)
    assert not coverage.observe(.04, True, 1.)
    assert coverage.maximum_path_m == .01
    coverage.observe(.08, True, 1.02)
    assert coverage.valid_length_m == .06
    assert coverage.missing_length_m == 0.


def test_matched_speed_replays_each_zero_alpha_dwell_once_then_resumes():
    replay = experiment.MatchedMotion(trace([0., 0., 0., .001, .002], [0., 0., 1., .5, 1.],
                                           [0., 0., 0., 0., 0.]), .005)
    assert replay.speed(0.) == 0.
    assert replay.speed(0.) == 0.
    assert replay.speed(0.) == 1.
    assert replay.speed(.0005) == 1.
    assert replay.speed(.001) == .5
    assert replay.speed(.002) == 1.


def test_matched_speed_does_not_skip_zero_dwell_when_actual_path_overtakes_record():
    replay = experiment.MatchedMotion(trace([0., .001, .001, .002], [1., 0., .5, 1.],
                                           [0., 0., 0., 0.]), .005)
    assert replay.speed(.0015) == 0.
    assert replay.speed(.0015) == .5


def test_matched_rocking_uses_measured_absolute_envelope_and_remaining_budget():
    replay = experiment.MatchedMotion(trace([0., .001, .002], [1., 1., 1.], [-.2, .1, -.3]), .005)
    assert replay.rotation_budget == pytest.approx(.003)
    assert replay.rocking(0., 0.) == .2
    assert replay.rocking(.0015, 0.) == .1
    assert replay.rocking(.002, .00275) == pytest.approx(.05)
    assert replay.rocking(.002, .004) == 0.


def make_observation(config, registration="expected_mapping"):
    return ContactObservation(0, "camera", 1., 1., np.ones(3), np.ones(3, dtype=bool),
                              registration, config.window_version,
                              calibration_version=config.calibration_version)


def test_observation_guard_requires_expected_mapping_window_and_calibration_revision():
    cfg = FeatureConfig(calibration_version="cal_v1")
    obs = make_observation(cfg)
    guard = lambda candidate: experiment.observation_matches(candidate, cfg,
                                                             registration_version="expected_mapping")
    assert guard(obs)
    assert not guard(None)
    assert not guard(replace(obs, registration_version="flipped_or_cropped_mapping"))
    assert not guard(replace(obs, window_version="old_windows"))
    assert not guard(replace(obs, calibration_version="old_calibration"))
    for changed in (replace(cfg, image_x_sign=-1), replace(cfg, near_depth=(.05, .22)),
                    replace(cfg, lateral_windows=((.05, .34), (.34, .66), (.66, .95)))):
        assert not experiment.observation_matches(obs, changed, registration_version="expected_mapping")


def test_actual_delay_reference_commit_and_abort_tail_are_separate(monkeypatch):
    """Four scripted admissions then a rejection, using healthy design seed 0."""
    events, prepares, commits, proposals = [], [], [], []
    plants = []
    original_plant, original_build = experiment.FiniteAreaPlant, experiment.build_contact_nominal
    pc, fc, qc = experiment.settings()
    # This test scripts admissions; acceleration-boundary solving is tested separately.
    monkeypatch.setattr(experiment, "settings", lambda: (pc, fc, replace(qc, max_acceleration=np.full(6, 1000.))))

    class ObservedPlant(original_plant):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            plants.append(self)

        def command(self, command):
            super().command(command)
            events.append(("sent", self.t, np.array(command, copy=True)))

    def observed_build(dt):
        law, cfg = original_build(dt)
        prepare, commit, abort = law.prepare, law.commit_applied, law.abort

        def prepared(**kwargs):
            prepares.append((kwargs["v_tcp_z_actual"], plants[0].velocity.copy()))
            events.append(("prepare", plants[0].t, None))
            return prepare(**kwargs)

        def committed(force, **kwargs):
            full = np.array(kwargs["final_full_twist"], copy=True)
            commits.append(full)
            events.append(("commit", plants[0].t, full))
            return commit(force, **kwargs)

        def aborted():
            events.append(("abort", plants[0].t, None))
            return abort()

        monkeypatch.setattr(law, "prepare", prepared)
        monkeypatch.setattr(law, "commit_applied", committed)
        monkeypatch.setattr(law, "abort", aborted)
        return law, cfg

    class ScriptedSolver:
        def __init__(self, config):
            self.config = config

        def solve(self, data, **kwargs):
            i = len(proposals)
            proposals.append(data)
            if i == 4:
                return QpResult(None, 0., np.zeros(2), data.mechanical, ContactStatus.SOLVER_FAILED,
                                {"reason": "test_rejected_proposal"})
            alpha = (0., .25, .5, .75)[i]
            command = data.path_twist*alpha
            command[[2, 4]] = np.clip(data.nominal_twist[[2, 4]]*.5,
                                      data.mechanical.lower[[2, 4]], data.mechanical.upper[[2, 4]])
            return QpResult(command, alpha, np.zeros(2), data.mechanical, ContactStatus.REPAIR)

    monkeypatch.setattr(experiment, "FiniteAreaPlant", ObservedPlant)
    monkeypatch.setattr(experiment, "build_contact_nominal", observed_build)
    monkeypatch.setattr(experiment, "ContactQp", ScriptedSolver)
    result, recorded = experiment.run_case("healthy", 0, "full")
    assert result["status"] == "aborted" and result["reason"] == "solver_failed"
    assert not result["valid"]
    assert len(proposals) == len(prepares) == 5 and len(commits) == 4
    np.testing.assert_allclose(recorded["reference_s"][:4], np.cumsum([0., .25, .5, .75])*.005)
    np.testing.assert_allclose(recorded["reference_s"][4:], .0075)
    assert result["reference_s"] == pytest.approx(.0075)
    assert np.all(recorded["alpha"][4:] == 0.)
    np.testing.assert_array_equal(recorded["committed"][:4], np.ones(4, dtype=bool))
    assert not recorded["committed"][4:].any()
    assert np.all(recorded["phase"][:4] == "scan")
    assert np.all(recorded["phase"][4:] == "stop_tail")
    np.testing.assert_array_equal(recorded["nominal_twist"][:4], [d.nominal_twist for d in proposals[:4]])
    np.testing.assert_array_equal(recorded["qp_twist"][:4], commits)
    np.testing.assert_array_equal(recorded["sent_twist"][:4], commits)
    np.testing.assert_array_equal(recorded["measured_twist"][:4], [p[1] for p in prepares[:4]])
    assert not recorded["sent_twist"][4:].any()
    assert np.any(np.abs(recorded["measured_twist"][4:]) > 1e-7)
    assert len(recorded["force_n"]) > 4  # held physical exposure survives logical abort
    assert np.ptp(recorded["path_m"][4:]) > 0.
    assert result["measured_forward_m"] != pytest.approx(result["reference_s"]*.02)
    for measured_scalar, actual in prepares:
        assert measured_scalar == actual[2]
    assert any(not np.allclose(d.nominal_twist, sent) for d, sent in zip(proposals, commits))
    assert any(not np.allclose(actual, sent) for (_, actual), sent in zip(prepares, commits))
    # Only a published candidate may commit nominal state/reference progress.
    for i, (kind, timestamp, value) in enumerate(events):
        if kind == "commit":
            assert events[i-1][0] == "sent" and events[i-1][1] == timestamp
            np.testing.assert_array_equal(events[i-1][2], value)


def test_shared_forearm_interval_starts_at_zero_speed_without_zero_candidate():
    ref = experiment.FiniteIntervalScanReference(.06, .02, .005)
    np.testing.assert_array_equal(ref.reference.sample(0).vel_ff, np.zeros(6))
    first = ref.sample(0.)
    assert 0 < first.vel_ff[1] < .001
    np.testing.assert_allclose(first.vel_ff[:3]*.005,
                               ref.reference.sample(.005).pose_d[:3]-first.pose_d[:3])
    np.testing.assert_array_equal(ref.sample(0).pose_d, first.pose_d)  # no clock mutation
    assert ref.duration_s == pytest.approx(3.4)


def test_shared_forearm_interval_end_is_bounded_and_both_position_and_ff_are_smooth():
    ref = experiment.FiniteIntervalScanReference(.06, .02, .005)
    times = np.r_[np.arange(0., ref.duration_s, .005), ref.duration_s]
    samples = [ref.sample(t) for t in times]
    velocities = np.array([r.vel_ff for r in samples])
    assert np.max(abs(np.diff(velocities[:,1]))) < .001
    assert np.max(velocities[:,1]) <= .02+1e-12
    np.testing.assert_array_equal(samples[-1].vel_ff, np.zeros(6))
    assert samples[-1].pose_d[1] == pytest.approx(.06)
    start = ref.duration_s-.001
    assert ref.candidate_duration(start) == pytest.approx(.001)
    assert start + .25*ref.candidate_duration(start) < ref.duration_s
    np.testing.assert_allclose(ref.sample(start).pose_d, ref.reference.sample(start).pose_d)
    np.testing.assert_allclose(ref.sample(start).vel_ff[:3]*.005,
                               samples[-1].pose_d[:3]-ref.sample(start).pose_d[:3])
    assert ref.sample(start).pose_d[1] < samples[-1].pose_d[1]


def test_matched_motion_does_not_replay_mechanical_stop_as_task_dwell():
    values = trace([0., .01, .02, .021], [1., .5, 0., 0.], [0., .2, .1, .05])
    values['phase'] = np.array(['scan', 'scan', 'stop_tail', 'stop_tail'])
    replay = experiment.MatchedMotion(values, .005)
    assert len(replay.path) == 2
    assert replay.speed(.021) == .5
    assert replay.rotation_budget == pytest.approx(.001)


def test_image_trace_metadata_is_explicit_and_age_uses_current_time():
    from peirastic.contact_qp.types import ContactObservation
    obs = ContactObservation(8, 'camera', 1., 1.15, [.5]*3, [True]*3, 'reg', 'win')
    early = experiment.image_trace_fields(obs,1.2)
    late = experiment.image_trace_fields(obs,1.4)
    assert early['image_frame_seq']==8
    assert early['image_effective_time_s']==1.
    assert early['image_received_time_s']==1.15
    assert late['image_age_s']-early['image_age_s']==pytest.approx(.2)
    assert early['quality_version']==late['quality_version']
    assert experiment.image_trace_fields(None,1.2)['image_frame_seq']==-1


def test_positive_fractional_publications_exhaust_only_machine_precision_clock_tail():
    ref=experiment.FiniteIntervalScanReference(.06,.02,.005)
    s=ref.duration_s-.001
    for _ in range(200):
        s=ref.commit_time(s,.41674)
    assert s==ref.duration_s
    near=np.nextafter(ref.duration_s,0.)
    assert ref.commit_time(near,0.)==near
    assert ref.commit_time(ref.duration_s-.001,0.)==ref.duration_s-.001
    # The exhaustion rule is not a macroscopic endpoint-time tolerance.
    s=ref.duration_s-1e-10
    assert ref.commit_time(s,.1)<ref.duration_s


@pytest.mark.parametrize('remaining', [1e-5,1e-6,1e-7])
def test_reference_geometry_roundoff_completes_without_giving_zero_alpha_clock(remaining):
    ref=experiment.FiniteIntervalScanReference(.06,.02,.005)
    s=ref.duration_s-remaining
    np.testing.assert_array_equal(ref.reference.sample(s).pose_d,
                                  ref.reference.sample(ref.duration_s).pose_d)
    assert ref.exhaustion_reason(s)=='reference_geometry_roundoff'
    assert ref.commit_time(s,0.)==s
    for elsewhere in (0.,.1,1.,ref.duration_s-.1):
        assert ref.exhaustion_reason(elsewhere)==''
        assert ref.commit_time(elsewhere,0.)==elsewhere
