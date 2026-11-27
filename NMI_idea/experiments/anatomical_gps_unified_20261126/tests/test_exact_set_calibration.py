"""Asset-free witnesses for exact echo training and segment set inversion."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import torch
from scipy.special import softmax

from reader.dynamic_factor_graph import DynamicFactorGraphReader, FrameEstimate, _IDENTITY_MARGINAL
from reader.station_classifier import (
    ECHO_IDENTITIES, STATION_IDS, SegmentCertifier, SegmentRecord, _FrameTruth,
    fit_segment_certifier, fit_station_scorer, load_station_checkpoint, make_segment_prediction,
    make_training_record, segment_nonconformity, uncalibrated_segment_certifier,
)
from sim.anatomy_assets import FAMILY_HELD_OUT_ATLASES, LOAO_OPEN3D
from sim.physics_ultrasound_kernel import ProbePose


_SPLIT = LOAO_OPEN3D
_SOURCE = _SPLIT.train_source_ids[0]
_STATION = np.array([.6, .4, 0., 0., 0.])
_IPE = np.array([.2, -.1])


def _estimate(exact, *, image=None, covariance=None):
    if image is None:
        image = np.zeros((6, 2))
    if covariance is None:
        covariance = np.tile(np.eye(2), (6, 1, 1))
    return FrameEstimate(exact @ _IDENTITY_MARGINAL, np.zeros((6, 3)), image, covariance,
                         _STATION, -520., _IPE, np.zeros((6, 3, 3)), exact)


@pytest.fixture(scope='module')
def fitted():
    echoes = np.array([[-25., 19., 5., .21, np.nan], [2., 33., 4., np.nan, .01],
                       [27., 39., 11., .11, .10]])
    graph = np.zeros((3, 64))
    graph[:, 2] = 1.  # All three true labels are absent from the retained beam.
    frame = SimpleNamespace(obs=echoes, echo_targets=((0,), (), (5, 0, 2)))
    fit = make_training_record(_SOURCE, 'fit', frame, _estimate(graph), STATION_IDS[0])
    shorter = SimpleNamespace(obs=echoes[:1], echo_targets=((0,),))
    extra = make_training_record(_SOURCE, 'fit', shorter, _estimate(graph[:1]))
    temp = make_training_record(_SOURCE, 'temp', frame, _estimate(graph), STATION_IDS[0])
    scorer = fit_station_scorer((fit, extra), (temp,), _SPLIT, epochs=6, seed=19)
    return scorer, fit, extra, temp, echoes, graph


def test_exact_training_keeps_single_null_and_triple_masks(fitted):
    scorer, fit, extra, _, echoes, graph = fitted
    labels = np.array([1, 0, 37])
    assert_array_equal(fit.identity_targets, labels)
    assert_array_equal(fit.graph_identities, graph)
    posterior, station = scorer.score_joint(echoes, graph, _IPE, -520., _STATION, 2.5, .4)
    assert posterior.shape == (3, 64)
    assert (posterior > 0.).all()
    assert_allclose(posterior.sum(axis=-1), 1., atol=2e-15, rtol=0.)
    assert_allclose(station.sum(), 1.)
    assert_array_equal(station[2:], 0.)

    # Compare to the fit-only reference at the same temperature. With an
    # identical beam row the reference cannot separate these anonymous echoes.
    labels_fit = np.concatenate((fit.identity_targets, extra.identity_targets))
    prior = (1 + np.bincount(labels_fit, minlength=64)) / (64 + len(labels_fit))
    backoff = (len(labels_fit) + 1) / (len(labels_fit) + 2)
    reference = softmax(np.log((1. - backoff) * graph + backoff * prior) / scorer.identity_temperature, axis=-1)
    observed = (np.arange(len(labels)), labels)
    assert np.log(posterior[observed]).mean() > np.log(reference[observed]).mean()
    assert not np.array_equal(posterior[0], posterior[2])

    # Output rows follow anonymous input rows; raw height/force/dwell do not
    # change identity or station inference when public features are fixed.
    order = [2, 0, 1]
    permuted, same_station = scorer.score_joint(echoes[order], graph[order], _IPE, -870., _STATION, 4.5, .0)
    assert_allclose(permuted, posterior[order], atol=0., rtol=1e-6)
    assert_array_equal(same_station, station)
    no_echo, _ = scorer.score_joint(echoes[:0], graph[:0], _IPE, -520., _STATION, 2.5, .4)
    assert no_echo.shape == (0, 64)
    with pytest.raises(ValueError, match='positive total'):
        scorer.score_joint(echoes, np.zeros_like(graph), _IPE, -520., _STATION, 2.5, .4)


def test_graph_readout_and_certificate_share_one_corrected_distribution(fitted):
    scorer, _, _, _, echoes, graph = fitted
    readout = SimpleNamespace(
        prior=SimpleNamespace(station_windows={STATION_IDS[0]: [-520.], STATION_IDS[1]: [-530.]}),
        scorer=None, _landmarks=np.array([-505., -530., -540., -850.]),
    )
    frame = SimpleNamespace(echoes=echoes, h=-520., force=2.5, probe=ProbePose(np.zeros(3), np.eye(3)))
    roles = np.arange(6)
    path = SimpleNamespace(masks=graph.argmax(axis=-1), centers=np.zeros((6, 3)),
                           world_cov=np.tile(np.eye(3), (6, 1, 1)), q=np.zeros(4), q_cov=np.eye(4),
                           ipe=_IPE, presence=np.zeros(14))
    estimate = DynamicFactorGraphReader._frame_estimate
    base = estimate(readout, frame, roles, [path], np.ones(1))
    assert_array_equal(base.identity_probabilities, graph)
    assert_array_equal(base.role_probabilities, graph @ _IDENTITY_MARGINAL)
    readout.scorer = scorer
    corrected = estimate(readout, frame, roles, [path], np.ones(1))
    expected, stations = scorer.score_joint(
        echoes, graph, base.ipe, frame.h, base.station_probabilities, frame.force, frame.probe.dwell_s,
        station_role_presence=base.station_role_presence,
        station_landmark_offsets_mm=base.station_landmark_offsets_mm,
    )
    assert_array_equal(corrected.identity_probabilities, expected)
    assert_array_equal(corrected.station_probabilities, stations)
    assert_array_equal(corrected.role_probabilities, expected @ _IDENTITY_MARGINAL)
    assert not np.array_equal(corrected.identity_probabilities, base.identity_probabilities)
    prediction = make_segment_prediction([corrected]).frames[0]
    expected_odds = np.log(expected.max(axis=-1, keepdims=True)) - np.log(expected)
    assert_allclose(prediction.identity_scores, expected_odds, atol=2e-14, rtol=0.)
    with pytest.raises(ValueError, match='exact contributor probabilities'):
        make_segment_prediction([replace(base, identity_probabilities=None)])


def _record(episode_id, prediction, masks, centers):
    truth = tuple(_FrameTruth(np.asarray(ids), values) for ids, values in zip(masks, centers, strict=True))
    return SegmentRecord(_SOURCE, episode_id, prediction, truth)


def _contains(predictive_set, truth):
    for frame, labels in zip(predictive_set.frames, truth, strict=True):
        roles = set()
        for row, mask in enumerate(labels.identities):
            if not frame.echo_candidates[row, mask]:
                return False
            roles.update(ECHO_IDENTITIES[mask])
        for role in roles:
            radii = frame.ellipse_radii_mm[role]
            if np.isinf(radii).all():
                continue
            local = (labels.centers_mm[role] - frame.centers_mm[role]) @ frame.ellipse_axes[role]
            if np.all(radii == 0.):
                if np.any(local != 0.):
                    return False
            elif np.linalg.norm(local / radii) > 1.:
                return False
    return True


def test_joint_set_inversion_ties_zero_beam_and_missing_geometry():
    first, second = np.zeros((1, 64)), np.zeros((1, 64))
    first[0, [0, 3, 37]] = [.10, .45, .45]
    second[0, [0, 4]] = [1., 1.e-30]
    missing = np.zeros((6, 2))
    missing[5] = np.nan
    cov = np.tile(np.diag([4., 1.]), (6, 1, 1))
    prediction = make_segment_prediction([_estimate(first, image=missing, covariance=cov), _estimate(second)])
    zeros = np.zeros((6, 2))
    far = zeros.copy()
    far[1, 0] = 6.  # 3 sigma on the second contributor of mask 3.
    very_far = zeros.copy()
    very_far[5] = [1.e9, -1.e9]  # No estimate for role 5: full plane, not measured geometry.
    examples = [(3, 0, zeros, zeros), (3, 0, far, zeros), (37, 0, very_far, zeros),
                (0, 0, zeros, zeros), (1, 0, zeros, zeros), (37, 4, very_far, far)]
    records = [_record(idx, prediction, ([a], [b]), (p, q)) for idx, (a, b, p, q) in enumerate(examples)]
    assert segment_nonconformity(records[0]) == segment_nonconformity(records[2]) == 0.
    assert segment_nonconformity(records[1]) == 3.
    assert np.isinf(segment_nonconformity(records[4]))
    tiny_mass_score = segment_nonconformity(records[5])
    assert tiny_mass_score > 9.

    previous = [np.zeros((1, 64), dtype=bool)] * 2
    for cutoff in (0., segment_nonconformity(records[3]), 3., 9., tiny_mass_score, np.inf):
        certifier = SegmentCertifier(cutoff, 8., .2, 0, None)
        predicted = certifier.predict_set(prediction)
        for last, frame in zip(previous, predicted.frames, strict=True):
            assert (frame.echo_candidates[last]).all()
        assert np.isinf(predicted.frames[0].ellipse_radii_mm[5]).all()
        previous = [frame.echo_candidates for frame in predicted.frames]
        expected = [segment_nonconformity(record) <= cutoff for record in records]
        assert [_contains(predicted, record.truth) for record in records] == expected
        assert_array_equal(~certifier.outcomes(records)[:, 0], expected)
    assert all(frame.all() for frame in previous)  # Infinite rank includes even zero-probability masks.

    tied = SegmentCertifier(0., 8., .2, 0, None).predict_set(prediction)
    assert_array_equal(np.flatnonzero(tied.frames[0].echo_candidates[0]), [3, 37])
    missing_only = np.zeros((1, 64))
    missing_only[0, 37] = 1.
    policy = SegmentCertifier(1., 8., .2, 0, None)
    one = make_segment_prediction([_estimate(missing_only, image=missing)])
    only_truth = _record('missing', one, ([37],), (very_far,))
    assert_array_equal(policy.outcomes([only_truth]), [[False, False, False]])


def test_large_geometry_rank_is_not_automatic_identity_saturation():
    exact = np.zeros((1, 64))
    exact[0, [1, 3]] = [1., 1.e-30]
    unit = np.tile(.25 * np.eye(2), (6, 1, 1))
    prediction = make_segment_prediction([_estimate(exact, covariance=unit)])

    def segment(index, dx, mask=1):
        centers = np.zeros((6, 2))
        centers[0, 0] = dx
        return _record(index, prediction, ([mask],), (centers,))

    cal = [segment(idx, dx) for idx, dx in enumerate([0., .5, 1., 4.5])]
    certifier = fit_segment_certifier(cal, _SPLIT)
    predicted = certifier.predict_set(prediction)
    assert certifier.cutoff == 9.
    assert_array_equal(np.flatnonzero(predicted.frames[0].echo_candidates[0]), [1])
    assert certifier.verify(predicted)
    assert_array_equal(certifier.outcomes([segment(10, 4.5), segment(11, 4.6)]),
                       [[False, False, True], [True, False, True]])

    # A true beam-zero identity at the selected rank is explicitly unrestricted.
    unbounded = fit_segment_certifier(cal[:3] + [segment(3, 0., mask=2)], _SPLIT)
    assert unbounded.predict_set(prediction).unbounded
    audit = unbounded.audit([segment(10, 4.5), segment(11, 9., mask=2)])
    assert audit['strata']['well_observed']['rank'] == 4
    assert audit['miscoverage']['rate'] == 0.
    assert audit['acceptance_rate'] == 0.
    assert audit['conditional_failure']['trials'] == 0
    assert audit['conditional_failure']['rate'] is audit['conditional_failure']['upper_bound'] is None
    assert 'not target capture' in audit['certification_scope']


def test_nominal_comparator_keeps_marginal_probability_rule():
    exact = np.zeros((1, 64))
    exact[0, [1, 3, 37]] = [.68, .21, .11]
    prediction = make_segment_prediction([_estimate(exact)])
    nominal = uncalibrated_segment_certifier(alpha=.2)
    assert_array_equal(np.flatnonzero(nominal.predict_set(prediction).frames[0].echo_candidates[0]), [1, 3])
    assert not nominal.verify(nominal.predict_set(prediction))
    assert nominal.audit(outcomes=np.empty((0, 3), dtype=bool))['rank_violation_bound'] is None


def test_checkpoint_roundtrip_version_authority_and_episode_isolation(fitted, tmp_path):
    scorer, fit, _, temp, echoes, graph = fitted
    with pytest.raises(ValueError, match='disjoint'):
        fit_station_scorer((fit,), (fit,), _SPLIT, epochs=1)
    with pytest.raises(ValueError, match='only split.train_source_ids'):
        fit_station_scorer((fit,), (replace(temp, source_id=_SPLIT.test_source_ids[0]),), _SPLIT, epochs=1)
    with pytest.raises(ValueError, match='each need exact echo and station supervision'):
        fit_station_scorer((replace(fit, station_target=-1),), (temp,), _SPLIT, epochs=1)
    with pytest.raises(ValueError, match='at least one supervised segment'):
        fit_segment_certifier([], _SPLIT, scorer=scorer)

    inputs = (echoes, graph, _IPE, -520., _STATION, 2.5, .4)
    exact, _ = scorer.score_joint(*inputs)
    prediction = make_segment_prediction([_estimate(exact)])
    truth = np.zeros((6, 2))
    ids = (exact.argmax(axis=-1),)
    for episode_id in (fit.episode_id, temp.episode_id):
        with pytest.raises(ValueError, match='independent of scorer fit and temperature'):
            fit_segment_certifier([_record(episode_id, prediction, ids, (truth,))], _SPLIT, scorer=scorer)
    segments = [_record(f'cal-{idx}', prediction, ids, (truth,)) for idx in range(4)]
    certifier = fit_segment_certifier(segments, _SPLIT, scorer=scorer)
    path = tmp_path / 'frozen-exact.pt'
    scorer.save(path, certifier=certifier, metadata={'purpose': 'roundtrip'})
    frozen, restored, metadata = load_station_checkpoint(path, _SPLIT)
    assert restored.state_dict() == certifier.state_dict()
    assert metadata == {'purpose': 'roundtrip'}
    assert type(frozen.identity_temperature) is float
    for old, new in zip(scorer.score_joint(*inputs), frozen.score_joint(*inputs), strict=True):
        assert_array_equal(old, new)
    with pytest.raises(ValueError, match='incompatible|source split'):
        load_station_checkpoint(path, FAMILY_HELD_OUT_ATLASES)
    payload = torch.load(path, weights_only=True)
    payload['version'] = 4
    old_path = tmp_path / 'schema-4.pt'
    torch.save(payload, old_path)
    with pytest.raises(ValueError, match='checkpoint version'):
        load_station_checkpoint(old_path, _SPLIT)
