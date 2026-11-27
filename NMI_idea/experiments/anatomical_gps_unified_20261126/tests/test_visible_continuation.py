"""Asset-free witnesses for visible-combination supervision and the Step 3R dev audit."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from reader.dev_visible_continuation import (
    RawStationWindows, _ALLOWED_SOURCES, _DEV_REPEATS, _DEV_SEED, _FIT_REPEATS, _FIT_SEED, _audit,
    _certificate_reason, _classification, _collect, _continuation, _funnel_flags, _pose_heights,
    _request_diagnostic, _reuse_fit, _shared_height_control, _tapes, label_frame,
)
from reader.station_classifier import SegmentCertifier, make_segment_prediction
from reader.visible_stations import STATIONS, VisibleStationReadout, frame_combo_scores, pose_features, visible_features
from sim.physics_ultrasound_kernel import ProbePose


S1, S2, S3, S4, S5 = STATIONS


def _posterior(*exact_masks):
    rows = np.zeros((len(exact_masks), 64))
    rows[np.arange(len(rows)), np.asarray(exact_masks, dtype=int)] = 1.
    return rows


def _supervision(h, targets=(), ghosts=None):
    return SimpleNamespace(acquisition_h_mm=h, echo_targets=targets,
                           echo_ghosts=tuple(False for _ in targets) if ghosts is None else ghosts,
                           centers_mm=np.zeros((6, 3)))


def _windows():
    masks = np.zeros((4, len(STATIONS)), dtype=bool)
    masks[0, :2] = True  # -530: S1/S2; -525: native unsupported gap
    masks[2, 0] = True  # -520: S1 only; -510: another nonstation row
    return RawStationWindows(np.array([-530., -525., -520., -510.]), masks)


def test_matching_uses_distinct_exact_singletons_in_one_frame():
    both = _posterior(1, 2)
    score, exact = frame_combo_scores(both)
    assert_array_equal(score, (1., 0., 0., 0., 1.))
    assert_array_equal(exact, (True, False, False, False, True))
    assert_array_equal(frame_combo_scores(_posterior(3, 0))[0], 0.)  # merged echo, not two singletons
    one_slot = _posterior(1, 0)
    one_slot[0, 1:3] = .5
    assert_array_equal(frame_combo_scores(one_slot)[0], 0.)  # two role candidates but only one slot
    weighted = both.copy()
    weighted[0, [0, 1, 2]] = [.2, .5, .3]
    weighted[1, [0, 1, 2]] = [.1, .8, .1]
    assert_allclose(frame_combo_scores(weighted)[0][0], np.sqrt(.3 * .8))
    assert_allclose(frame_combo_scores(weighted[::-1])[0], frame_combo_scores(weighted)[0])

    h = [-520., -520.]
    across = visible_features(h, (_posterior(1), _posterior(2)))
    together = visible_features(h, (_posterior(), both))
    empty = visible_features(h, (_posterior(), _posterior()))
    width = pose_features(h).shape[1]
    assert_array_equal(across, empty)
    assert_array_equal(across[:, :width], pose_features(h))
    assert across[0, width] == 0. and together[0, width] == 1.
    readout = VisibleStationReadout('visible', np.full((len(STATIONS), across.shape[1] + 1), 1.))
    assert not readout.predict(across).any()
    assert not readout.predict(empty).any()
    assert readout.predict(together)[0]


def test_labels_preserve_native_gaps_ignore_ghosts_and_do_not_merge_frames():
    windows = _windows()
    assert_array_equal(windows.opportunities, (True, True, False, False, False))
    assert windows.nearest_row(-527.5) == 0  # tie goes to lower physical height
    assert windows.nearest_row(-522.5) == 1  # never skip an unsupported native row
    assert windows.nearest_row(-515.) is None  # closest existing rows are both outside a half-bin
    assert windows.nearest_row(-530.1) is None  # no extrapolation
    empty = label_frame(_supervision(-530.), windows)
    assert empty.counts['echoes'] == 0
    assert empty.coverage[0] and not empty.visible.any()
    no_clean_pair = label_frame(_supervision(-530., ((0,), (1,), (1, 2), (), ()),
                                              (False, True, False, True, False)), windows)
    assert no_clean_pair.coverage[0] and not no_clean_pair.visible.any()
    assert no_clean_pair.counts == {'echoes': 5, 'clean_singletons': 1, 'true_null': 1, 'pure_ghost': 1,
                                    'mixed_ghost_lumen': 1, 'unresolved_merge': 1}
    off_window = label_frame(_supervision(-525., ((0,), (1,), (2,), (5,))), windows)
    assert not off_window.coverage.any() and not off_window.visible.any()
    just_second_role = label_frame(_supervision(-530., ((1,),)), windows)
    assert not np.any([no_clean_pair.visible, off_window.visible, just_second_role.visible])
    complete = label_frame(_supervision(-530., ((0,), (1,), (2,), (5,))), windows)
    assert_array_equal(complete.visible, (True, True, False, False, False))


def test_pose_control_reads_only_public_probe_and_tracker_inputs():
    class PoseOnlyFrame:
        probe_pose = SimpleNamespace(origin_mm=np.array([0., -522., 0.]))
        q_hat = np.array([.02, 0., 0., 0.])

        def __getattr__(self, field):
            raise AssertionError(f'pose comparator accessed {field}')

    def public_height(probe_pose, q_hat):
        return probe_pose.origin_mm[1] + 100. * q_hat[0]

    h = _pose_heights(public_height, [PoseOnlyFrame()] * 4)
    assert_array_equal(h, [-520.] * 4)
    assert_array_equal(pose_features(h), pose_features([-520.]))


def test_fitted_readouts_round_trip_with_rare_and_zero_positive_labels():
    heights = [[-530.] * 4] * 5 + [[-850.] * 4] * 5
    echoes = [_posterior(1, 2, 4)] + [_posterior(1, 2)] * 4 + [_posterior()] * 5
    y = np.zeros((len(heights), len(STATIONS)), dtype=bool)
    y[:5, 0] = True
    y[0, 1] = True  # retain a rare class alongside three absent classes
    features = {'pose_only': np.stack([pose_features(h) for h in heights]),
                'visible': np.stack([visible_features(h, [echo] * 4) for h, echo in zip(heights, echoes, strict=True)])}
    for mode, x in features.items():
        fitted = VisibleStationReadout.fit(mode, x, y)
        state = json.loads(json.dumps(fitted.state_dict(), allow_nan=False))
        restored = VisibleStationReadout.from_state_dict(state)
        assert_array_equal(restored.predict_proba(x), fitted.predict_proba(x))
        assert_array_equal(restored.predict(x), fitted.predict_proba(x) >= .5)
        assert restored.predict_proba(x).shape == y.shape
        assert np.isfinite(restored.parameters).all() and not restored.parameters.flags.writeable
        assert not restored.predict(x)[:, 2:].any()
        state['config']['threshold'] = .6
        with pytest.raises(ValueError, match='configuration'):
            VisibleStationReadout.from_state_dict(state)


def _estimate(posterior, *, world=None, scale=1., invalid=False):
    centers = np.zeros((6, 2))
    if invalid:
        centers[0] = np.nan
    return SimpleNamespace(identity_probabilities=posterior, role_probabilities=np.empty((len(posterior), 7)),
                           centers_mm=np.zeros((6, 3)) if world is None else world,
                           image_centers_mm=centers, image_covariance_mm2=np.tile(np.eye(2) * scale**2, (6, 1, 1)))


def test_funnel_earliest_complete_frame_and_strict_world_tolerance():
    frames = [SimpleNamespace(supervision=_supervision(h, t)) for h, t in
              [(-525., ((0,), (1,))), (-530., ((0,), (1,))), (-530., ((0,), (1,)))]]
    labels = [label_frame(frame.supervision, _windows()) for frame in frames]
    world = np.zeros((6, 3))
    world[0, 0] = 6.
    estimates = [_estimate(_posterior(1, 2)), _estimate(_posterior(2, 2), world=world), _estimate(_posterior(1, 2))]
    result = _request_diagnostic(S1, frames, labels, estimates)
    assert result['earliest_true_complete_frame_index'] == 1
    assert result['required_singleton_hits'] == 1 and not result['exact_identity_right']
    assert not result['world_tracking_under_6_mm'] and result['world_errors_mm']['MAIN_VEIN'] == 6.
    world[0, 0] = np.nan
    missing = _request_diagnostic(S1, frames, labels, estimates)
    assert missing['world_errors_mm']['MAIN_VEIN'] is None and not missing['world_tracking_under_6_mm']


def test_joint_funnel_does_not_intersect_hits_from_different_tapes():
    requests = [
        {'earliest_true_complete_frame_index': 0, 'exact_identity_right': False, 'world_tracking_under_6_mm': True},
        {'earliest_true_complete_frame_index': 1, 'exact_identity_right': True, 'world_tracking_under_6_mm': True},
        {'earliest_true_complete_frame_index': 1, 'exact_identity_right': True, 'world_tracking_under_6_mm': True},
        {'earliest_true_complete_frame_index': None, 'exact_identity_right': True, 'world_tracking_under_6_mm': True},
    ]
    certs = [
        dict(accepted=True, miscovered=False, false_certified=False, first_verifier_outcome='accepted'),
        dict(accepted=False, miscovered=False, false_certified=False, first_verifier_outcome='geometric_ellipse_too_large'),
        dict(accepted=True, miscovered=True, false_certified=False, first_verifier_outcome='accepted'),
        dict(accepted=True, miscovered=False, false_certified=False, first_verifier_outcome='accepted'),
    ]
    samples = [SimpleNamespace(log={'certificate': c, 'funnel_joint': _funnel_flags(r, c)})
               for r, c in zip(requests, certs, strict=True)]
    report = _continuation(samples)
    assert list(report['joint_counts'].values()) == [4, 3, 2, 2, 0]
    assert report['denominator_scheduled_tapes'] == 4
    raw = report['raw_whole_segment_certificate']
    assert raw['accepted'] == 3 and raw['violations_on_accepted'] == 1


def test_first_rejection_reasons_use_frozen_predictive_set_and_verifier():
    exact = _posterior(1)
    ambiguous = exact.copy()
    ambiguous[0, :2] = .5
    cases = [(1., _estimate(exact), 'accepted'),
             (1., _estimate(ambiguous), 'singleton_ambiguity_or_empty_factor'),
             (1., _estimate(exact, scale=9.), 'geometric_ellipse_too_large'),
             (1., _estimate(exact, invalid=True), 'geometric_ellipse_unavailable'),
             (1., _estimate(_posterior(0)), 'no_lumen'),
             (1., _estimate(_posterior()), 'no_lumen'),
             (np.inf, _estimate(exact), 'unbounded')]
    for cutoff, estimate, reason in cases:
        certifier = SegmentCertifier(cutoff=cutoff, tolerance_mm=8., alpha=.2, calibration_segments=1, rank=1)
        predicted = certifier.predict_set(make_segment_prediction([estimate]))
        assert _certificate_reason(predicted, certifier.tolerance_mm) == reason
        assert certifier.verify(predicted) == (reason == 'accepted')


def test_tape_schedule_and_paired_collection_report_unsupported_and_empty_cases():
    source = _ALLOWED_SOURCES[0]
    windows = {source: _windows()}
    population = [(source, (S1, S2))]
    fit = _tapes(population, _FIT_SEED, _FIT_REPEATS)
    dev = _tapes(population, _DEV_SEED, _DEV_REPEATS)
    assert len(fit) == 24 and len(dev) == 16
    assert len({t.seed for t in (*fit, *dev)}) == len(fit) + len(dev)
    assert _tapes(list(reversed(population)), _FIT_SEED, _FIT_REPEATS) == fit
    pose = ProbePose(np.array([0., -530., 0.]), np.eye(3))

    class Sim:
        def episode(self, source_id, seed, *, station_id, benchmark, frames):
            assert source_id == source and benchmark == 'station' and frames == 4
            roles = ((0,), (1,)) if station_id == S1 else ()
            frame = SimpleNamespace(probe_pose=pose, q_hat=np.zeros(4), Sigma_q=np.eye(4), dt=.18,
                                    f_contact=2.4, supervision=_supervision(-530., roles),
                                    obs=_posterior(1, 2) if roles else _posterior(), echo_targets=roles,
                                    centers_mm=np.zeros((6, 3)))
            return SimpleNamespace(h_start_mm=-530., frames=(frame,) * frames)

    reader = SimpleNamespace(_height=lambda p, q: p.origin_mm[1], reset=lambda: None,
                             step=lambda obs, p, force, q, cov, dt: _estimate(obs))
    samples = _collect(Sim(), reader, (dev[0], dev[-1]), windows, audit=True)
    for sample in samples:
        sample.log['probabilities'] = {'visible': dict.fromkeys(STATIONS, 1.),
                                       'pose_only': dict.fromkeys(STATIONS, 0.)}
    _audit(samples, SegmentCertifier(cutoff=1., tolerance_mm=8., alpha=.2, calibration_segments=1, rank=1))
    result = _classification(samples)
    assert result['empty_segments'] == 1
    assert result['class_counts'][S1]['C_without_V'] == 1
    assert result['class_counts'][S3]['scored_supported'] == 0
    assert result['visible']['by_station'][S3]['false_positives'] == 0
    assert result['visible']['by_station'][S3]['unscored_unsupported_predictions'] == 2
    assert result['visible']['evidence_status'] == 'insufficient_evidence'
    assert result['visible']['macro_f1_including_all_five_classes'] == pytest.approx((2 / 3) / 5)
    assert result['pose_only']['macro_f1_including_all_five_classes'] == 0.
    assert_array_equal(samples[0].features['pose_only'], samples[1].features['pose_only'])
    assert _shared_height_control(samples, windows)['tapes_at_realized_duplicate_request_heights'] == 2
    assert list(_continuation(samples)['joint_counts'].values()) == [2, 1, 1, 1, 1]


def test_confirmation_restores_development_weights_without_refitting():
    population = [(_ALLOWED_SOURCES[0], (S1,))]
    fit_tapes = _tapes(population, _FIT_SEED, _FIT_REPEATS)
    dev_tapes = _tapes(population, _DEV_SEED, _DEV_REPEATS)
    x = {'pose_only': pose_features([-530.]), 'visible': visible_features([-530.], [_posterior(1, 2)])}
    models = {mode: VisibleStationReadout(mode, np.full((len(STATIONS), f.shape[-1] + 1), .25))
              for mode, f in x.items()}
    report = {'schema': 'step3r-visible-continuation-dev-v1', 'cohort': 'dev',
              'provenance': {'split': 'LOAO_OPEN3D', 'train_source_ids': list(_ALLOWED_SOURCES),
                             'checkpoint_sha256': 'frozen'},
              'fit': {'tapes': [asdict(t) for t in fit_tapes]}, 'dev': {'tapes': [asdict(t) for t in dev_tapes]},
              'fitted_readouts': {mode: model.state_dict() for mode, model in models.items()}}
    fit_summary, restored = _reuse_fit(report, 'frozen', fit_tapes, dev_tapes)
    assert fit_summary is report['fit']
    for mode, f in x.items():
        assert_array_equal(models[mode].predict_proba(f), restored[mode].predict_proba(f))
    with pytest.raises(ValueError, match='same frozen'):
        _reuse_fit(report, 'different', fit_tapes, dev_tapes)
    report['dev']['tapes'].pop()
    with pytest.raises(ValueError, match='tape schedules'):
        _reuse_fit(report, 'frozen', fit_tapes, dev_tapes)
