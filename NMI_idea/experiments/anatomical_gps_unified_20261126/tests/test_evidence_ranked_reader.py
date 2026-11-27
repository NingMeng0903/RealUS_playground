"""Common-parent ranking through the real acoustic beam, factors and replay."""
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import CubicHermiteSpline
from scipy.sparse import csc_matrix, csr_matrix

from reader.baselines import Full_AGPS_Reader
from reader.dynamic_factor_graph import FrameEstimate, ROLE_COUNT, _Design, _Prediction
from reader.evidence_ranked_reader import EvidenceRankedReader
from sim.physics_ultrasound_kernel import ProbePose, UltrasoundObservation
from sim.smplx_dynamic_atlas import _HIP, _KNEE


_LOG_CLUTTER = np.log(.42 / (50. * 77.))
_LOG_HALF = np.log(.5)


class _OneLumenProjection:
    """One scalar normal prior and analytic image tangents; real reader updates."""

    def __init__(self, tangent, *, seed_offsets):
        self.tangent, self.seed_offsets = tangent, seed_offsets
        prior = SimpleNamespace(coordinate_mode='material', mean=np.zeros(1), station_windows={},
                                graph=SimpleNamespace(h_mm=np.array([-700., -600.]), junctions={}),
                                prior_precision_csc=lambda: csc_matrix([[1.]]),
                                knot_design_matrix=lambda: csr_matrix((0, 1)), nominal_knots=np.empty((0, 3)))
        joints = max(_HIP, _KNEE) + 1
        kinematics = SimpleNamespace(_neutral=SimpleNamespace(joints_m=np.zeros((joints, 3))),
                                     _parents=np.zeros(joints, dtype=int), _weights=np.zeros((0, joints)))
        super().__init__(prior, kinematics, max_branches=2, max_iterations=1)

    def _design(self, h):
        return _Design(np.array([0]), np.zeros((1, 3)), np.zeros((1, 3)), np.array([0]),
                       np.array([[[1.], [0.], [0.]]]), np.zeros((1, 3, 1)),
                       np.array([0], dtype=np.int32), np.array([0, 1], dtype=np.int32))

    def _predict_many(self, frame, design, branches):
        predictions = []
        for branch in branches:
            x, slope = self.tangent(branch.mean.item())
            world = np.array([[x, frame.h, 20.]])
            h_world = np.array([[slope], [0.], [0.]])
            predictions.append(_Prediction(world, world[:, [0, 2]], h_world, h_world[[0, 2]],
                                           np.zeros((3, 4)), np.zeros((2, 4)),
                                           np.zeros((3, 3 * ROLE_COUNT)), np.zeros((2, 3 * ROLE_COUNT)),
                                           np.eye(2), np.array([.5])))
        return predictions

    def _initial_modes(self, frame, design, parent):
        return [parent, *(replace(parent, mean=parent.mean + offset) for offset in self.seed_offsets)]


class _Baseline(_OneLumenProjection, Full_AGPS_Reader):
    pass


class _Candidate(_OneLumenProjection, EvidenceRankedReader):
    pass


def _step(reader, xs):
    echoes = np.full((len(xs), 5), np.nan)
    echoes[:, 0], echoes[:, 1] = xs, 20.
    probe = ProbePose(np.array([0., -650., 0.]), np.eye(3))
    return reader.step(UltrasoundObservation(echoes), probe, 2.5, np.zeros(4), np.zeros((4, 4)), .1)


def _match_evidence(adjusted, slope, *, rho=.85, clutter_count=0):
    # Integrate the powered 2D unit-noise likelihood over x ~ N(0,1).
    variance = 1. + rho * slope**2
    return (rho * (_LOG_HALF + clutter_count * _LOG_CLUTTER - np.log(2. * np.pi))
            - .5 * (np.log(variance) + rho * adjusted**2 / variance))


def _record_refinements(reader, monkeypatch):
    refine = reader._refine
    pools = []

    def record(frame, design, parents, proposals, rho):
        pools.append(tuple(proposals))
        return refine(frame, design, parents, proposals, rho)

    monkeypatch.setattr(reader, '_refine', record)
    return pools


def test_tangent_selected_by_common_parent_evidence_not_seed_predictive_score(monkeypatch):
    chart = CubicHermiteSpline([0., 5., 10.], [3., 1., 0.], [0., .2, 1.])
    derivative = chart.derivative()

    def tangent(z):
        return chart(z).item(), derivative(z).item()

    baseline = _Baseline(tangent, seed_offsets=(5., 10.))
    candidate = _Candidate(tangent, seed_offsets=(5., 10.))
    old_pools = _record_refinements(baseline, monkeypatch)
    new_pools = _record_refinements(candidate, monkeypatch)
    evaluate = candidate._local_factor
    calls = {}

    def record(parent, linearization, pred, design, association, observation, rho, solved):
        assert parent.mean.item() == 0.
        key = (linearization.item(), association.groups)
        assert key not in calls  # No repeated scoring to reconstruct finalists.
        result = evaluate(parent, linearization, pred, design, association, observation, rho, solved)
        calls[key] = association.score, result[2]
        return result

    monkeypatch.setattr(candidate, '_local_factor', record)
    old = _step(baseline, [0.])
    new = _step(candidate, [0.])
    match = ((0,),)
    assert calls.keys() == {(0., match), (5., match), (10., match), (0., ((),))}
    assert calls[10., match][0] > calls[5., match][0] > calls[0., match][0]
    assert_allclose([calls[s, match][1] for s in (0., 5., 10.)],
                    [_match_evidence(-3., 0.), _match_evidence(0., .2), _match_evidence(10., 1.)])
    # v6 keeps the point that explains the echo at z=10; transported back
    # to the parent it gives an adjusted residual of +10 and loses the path.
    assert [(p.masks, p.linearization.item()) for p in old_pools[0]] == [((1,), 10.), ((0,), 0.)]
    assert [(p.masks, p.linearization.item()) for p in new_pools[0]] == [((1,), 5.), ((0,), 0.)]
    assert [b.path.masks for b in baseline._branches] == [(0,)]
    assert [b.path.masks for b in candidate._branches] == [(1,)]
    assert_array_equal(old.identity_probabilities[0, :2], [1., 0.])
    assert_array_equal(new.identity_probabilities[0, :2], [0., 1.])


def test_integrated_ranking_recovers_path_before_refinement_cap(monkeypatch):
    chart = CubicHermiteSpline([0., 10.], [20., 0.], [0., 1.])
    derivative = chart.derivative()

    def tangent(z):
        return chart(z).item(), derivative(z).item()

    baseline = _Baseline(tangent, seed_offsets=(10.,))
    candidate = _Candidate(tangent, seed_offsets=(10.,))
    old_pools = _record_refinements(baseline, monkeypatch)
    new_pools = _record_refinements(candidate, monkeypatch)
    xs = [-10., 0., 1., 2., 3.]
    old = _step(baseline, xs)
    new = _step(candidate, xs)
    target, *decoys = (tuple(row) for row in np.eye(len(xs), dtype=int))
    null = (0,) * len(xs)
    assert [p.masks for p in old_pools[0]] == decoys  # v6's max_branches + 2.
    assert [p.masks for p in new_pools[0]] == [target, null, decoys[0]]
    best, _, crossing = new_pools[0]
    assert crossing.log_evidence < best.log_evidence - 6.
    assert_allclose(best.log_evidence, _match_evidence(0., 1., clutter_count=4))
    assert old.identity_probabilities[0, 1] == 0.
    assert new.identity_probabilities[0, 1] == 1.
    assert [b.path.masks for b in candidate._branches] == [target]


def _assert_estimates_equal(actual, expected):
    for field in fields(FrameEstimate):
        assert_allclose(getattr(actual, field.name), getattr(expected, field.name), rtol=0., atol=2.e-12,
                        err_msg=field.name)


def test_duplicate_seeds_have_no_mass_and_null_empty_preserve_both_ancestors(monkeypatch):
    reader = _Candidate(lambda z: (z, 1.), seed_offsets=(0., 0., 2.))
    pools = _record_refinements(reader, monkeypatch)
    first = _step(reader, [3.])
    match, null = pools[0]
    assert (match.masks, null.masks) == ((1,), (0,))
    assert match.linearization is null.linearization  # Exact seed tie: original parent is first.
    log_match = _match_evidence(3., 1.)
    log_null = .85 * _LOG_CLUTTER + _LOG_HALF
    expected_mass = np.exp(np.array([log_null, log_match]) - np.logaddexp(log_null, log_match))
    assert_allclose(first.identity_probabilities[0, :2], expected_mass, rtol=0., atol=2.e-12)

    prefix = [first]
    for xs, masks, evidence in [([20.], (0,), .25 * (_LOG_CLUTTER + _LOG_HALF)), ([], (), .25 * _LOG_HALF)]:
        previous = reader._branches
        prefix.append(_step(reader, xs))
        assert len(reader._branches) == len(previous) == 2
        for old, new in zip(previous, reader._branches, strict=True):
            assert new.path.previous is old.path
            assert new.path.masks == masks
            assert new.factor is old.factor and new.precision is old.precision and new.mean is old.mean
            assert_allclose([new.log_weight, new.path.log_evidence], [old.log_weight, evidence],
                            rtol=0., atol=2.e-12)
        for actual, expected in zip(reader.relabel_history(), prefix, strict=True):
            _assert_estimates_equal(actual, expected)
    assert_array_equal(prefix[1].identity_probabilities[0, :2], [1., 0.])
    assert prefix[2].identity_probabilities.shape == (0, 1 << ROLE_COUNT)

    reader.reset()
    assert reader.relabel_history() == ()
    _assert_estimates_equal(_step(reader, [3.]), first)
