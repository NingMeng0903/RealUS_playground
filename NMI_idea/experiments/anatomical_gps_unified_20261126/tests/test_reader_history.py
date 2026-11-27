"""Retained-path posterior checks on a one-coordinate acoustic graph."""
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csc_matrix, csr_matrix

from reader.dynamic_factor_graph import (DynamicFactorGraphReader, FrameEstimate, NULL_ROLE, ROLE_COUNT,
                                         _Association, _Design, _Prediction, _score_associations)
from sim.physics_ultrasound_kernel import ProbePose, UltrasoundObservation
from sim.smplx_dynamic_atlas import _HIP, _KNEE


class _LinearReader(DynamicFactorGraphReader):
    """Replace only chart projection with an affine model; keep the real filter."""

    def __init__(self, roles, *, noise=1., gain=1., prior_variance=1.):
        self.roles = np.asarray(roles, dtype=int)
        self.noise, self.gain = noise, gain
        self.seed_calls = []
        chart = SimpleNamespace(_axis=lambda h: np.zeros(2), nominal_reference_radius=lambda h: 20.)
        graph = SimpleNamespace(h_mm=np.array([-700., -600.]), junctions={}, vertices=())
        prior = SimpleNamespace(coordinate_mode='material', graph=graph, chart=chart, mean=np.zeros(1),
                                prior_precision_csc=lambda: csc_matrix([[1. / prior_variance]]),
                                knot_design_matrix=lambda: csr_matrix((0, 1)), nominal_knots=np.empty((0, 3)),
                                layout=SimpleNamespace(shared=slice(None)), station_windows={'S1_CFV_CFA': (-650.,)})
        joints = max(_HIP, _KNEE) + 1
        kinematics = SimpleNamespace(_neutral=SimpleNamespace(joints_m=np.zeros((joints, 3))),
                                     _parents=np.zeros(joints, dtype=int), _weights=np.zeros((0, joints)))
        super().__init__(prior, kinematics, max_branches=6, max_iterations=1)

    def _design(self, h):
        roles = self.roles if h >= -675. else self.roles[:0]
        n = len(roles)
        cols = np.arange(bool(n), dtype=np.intp)
        op = np.zeros((n, 3, len(cols)))
        op[:, 0] = self.gain
        return _Design(roles, np.zeros((n, 3)), np.zeros((n, 3)), cols, op, np.zeros_like(op),
                       cols.astype(np.int32), np.array([0, len(cols)], dtype=np.int32))

    def _predict_many(self, frame, design, branches):
        if not len(design.roles):
            return super()._predict_many(frame, design, branches)
        n = len(design.roles)
        h_world = design.operator.reshape(3 * n, 1)
        h_image = design.operator[:, [0, 2]].reshape(2 * n, 1)
        covariance = self.noise * np.eye(2 * n)
        base = np.column_stack((.4 * design.roles, np.full(n, frame.h), np.full(n, 20.)))
        result = []
        for branch in branches:
            world = base + (h_world @ branch.mean).reshape(n, 3)
            result.append(_Prediction(world, world[:, [0, 2]], h_world, h_image,
                                      np.zeros((3*n, 4)), np.zeros((2*n, 4)),
                                      np.zeros((3*n, 3*ROLE_COUNT)), np.zeros((2*n, 3*ROLE_COUNT)),
                                      covariance, np.full(n, .5)))
        return result

    def _initial_modes(self, frame, design, parent):
        seeds = super()._initial_modes(frame, design, parent)
        self.seed_calls.append((parent, seeds))
        return seeds


def _observation(xs):
    echoes = np.full((len(xs), 5), np.nan)
    echoes[:, 0] = xs
    echoes[:, 1:3] = [20., 6.]
    return UltrasoundObservation(echoes)


def _step(reader, xs, *, h=-650., force=2.5, dt=.1):
    probe = ProbePose(np.array([0., h, 0.]), np.eye(3))
    return reader.step(_observation(xs), probe, force, np.zeros(4), np.zeros((4, 4)), dt)


def _assert_estimate_equal(before, after):
    for field in fields(FrameEstimate):
        assert_allclose(getattr(before, field.name), getattr(after, field.name), rtol=0., atol=2.e-12,
                        err_msg=field.name)


def test_one_frame_does_not_replace_discounted_assignment_with_raw_null(monkeypatch):
    # The raw one-lumen score is slightly worse than NULL. On a novel view,
    # nondetection is not discounted, so the actual online ranking reverses.
    clutter_density = .42 / (50. * 77.)
    noise = np.exp(.05) / (2. * np.pi * clutter_density)
    raw = _score_associations(_observation([0.]).echoes, np.array([0]), np.array([[0., 20.]]),
                              noise * np.eye(2), np.array([.5]), 4.7, 2.5, 0., 6., 3.)
    assert raw[0].groups == ((),)
    assert_allclose(raw[1].score - raw[0].score, -.05, atol=1.e-12)

    reader = _LinearReader((0,), noise=noise, gain=0.)
    online = _step(reader, [0.])
    log_odds = -.05 * .85 + (.85 - 1.) * np.log(.5)
    p_match = 1. / (1. + np.exp(-log_odds))
    assert_allclose(online.identity_probabilities[0, [0, 1]], [1. - p_match, p_match], atol=1.e-12)
    assert np.argmax(online.role_probabilities[0]) == 0

    def fail_reassociation(*args, **kwargs):
        pytest.fail('history readout must not reassociate or apply another likelihood')

    monkeypatch.setattr(reader, '_associate', fail_reassociation)
    monkeypatch.setattr(reader, '_local_factor', fail_reassociation)
    replay, = reader.relabel_history()
    assert_array_equal(online.identity_probabilities, replay.identity_probabilities)
    assert_array_equal(online.role_probabilities, replay.role_probabilities)
    _assert_estimate_equal(online, replay)


_FIRST = ((((0, 1), ()), .5), (((), (0,)), .8), (((), ()), -2.2))
_SECOND = ((((0,),), 7.), (((),), 0.))


class _ScriptedReader(_LinearReader):
    def __init__(self, reverse):
        super().__init__((0, 1))
        self.reverse = reverse

    def _associate(self, frame, design, branch, pred, *, solved=None):
        count = len(frame.echoes)
        if not count:
            return super()._associate(frame, design, branch, pred, solved=solved)
        candidates = []
        for groups, acoustic in (_FIRST if count == 2 else _SECOND):
            slots = np.flatnonzero([bool(g) for g in groups])
            operator = np.zeros((2*len(slots), 2*len(design.roles)))
            for i, slot in enumerate(slots):
                group = np.asarray(groups[slot])
                operator[2*i, 2*group] = operator[2*i+1, 2*group + 1] = 1. / len(group)
            residual = frame.echoes[slots, :2].ravel() - operator @ pred.image.ravel()
            # Deliberately tied screening scores, to test that proposal/seed
            # traversal order does not become prior mass in the retained beam.
            candidates.append(_Association(0., groups, slots, operator, residual, acoustic))
        return tuple(reversed(candidates)) if self.reverse else tuple(candidates)


def _masks(spec):
    return tuple(sum(2**role for role in group) for group in spec[0])


def _integrate(spec, xs, mu, var, rho, *, novel):
    """Independent scalar Gaussian integral: one matched centroid at most."""
    groups, acoustic = spec
    present = sum(map(len, groups))
    score = rho * (acoustic + present*np.log(.5)) + (1. if novel else rho) * (2-present)*np.log(.5)
    if not present:
        return score, mu, var
    slot, group = next((slot, group) for slot, group in enumerate(groups) if group)
    r = 1. / len(group)  # independent unit noise on each contributing lumen
    dx = xs[slot] - .4*np.mean(group) - mu
    innovation = var + r/rho
    score -= .5 * (dx**2 / innovation + np.log1p(rho*var/r) + 2*rho*np.log(2*np.pi*r))
    return score, mu + var/innovation * dx, var * (r/rho) / innovation


def _normalize(log_mass):
    total = sum(np.exp(log_weight) for log_weight in log_mass.values())
    return {key: np.exp(log_weight) / total for key, log_weight in log_mass.items()}


def _check_exact(estimate, mass):
    exact = np.zeros_like(estimate.identity_probabilities)
    roles = np.zeros_like(estimate.role_probabilities)
    for masks, weight in mass.items():
        for i, mask in enumerate(masks):
            exact[i, mask] += weight
            members = [role for role in range(ROLE_COUNT) if mask & (2**role)] or [NULL_ROLE]
            roles[i, members] += weight / len(members)
    assert_allclose(estimate.identity_probabilities, exact, atol=2.e-12, rtol=0.)
    assert_allclose(estimate.role_probabilities, roles, atol=2.e-12, rtol=0.)


@pytest.mark.parametrize('reverse', [False, True])
def test_two_frame_paths_match_enumeration_with_null_and_merge(reverse):
    reader = _ScriptedReader(reverse)
    x1, x2 = [-.4, 1.05], [-2.75]
    first = _step(reader, x1)
    old_prefix, = reader.relabel_history()
    _assert_estimate_equal(first, old_prefix)
    stage1 = {_masks(spec): _integrate(spec, x1, 0., 1., .85, novel=True) for spec in _FIRST}
    prefix_mass = _normalize({mask: score for mask, (score, _, _) in stage1.items()})
    _check_exact(old_prefix, prefix_mass)
    assert np.argmax(old_prefix.identity_probabilities[0]) == 0

    latest = _step(reader, x2)
    first_revised, second_revised = reader.relabel_history()
    _assert_estimate_equal(latest, second_revised)
    terminal = _normalize({(mask, _masks(spec)): score + _integrate(spec, x2, mu, var, .25, novel=False)[0]
                           for mask, (score, mu, var) in stage1.items() for spec in _SECOND})
    actual = {(b.path.previous.masks, b.path.masks): np.exp(b.log_weight) for b in reader._branches}
    assert len(reader._branches) == len(terminal) == 6
    assert actual.keys() == terminal.keys()
    assert_allclose([actual[key] for key in terminal], list(terminal.values()), atol=2.e-12, rtol=0.)
    first_mass = {mask: sum(terminal[mask, _masks(spec)] for spec in _SECOND) for mask in stage1}
    second_mass = {_masks(spec): sum(terminal[mask, _masks(spec)] for mask in stage1) for spec in _SECOND}
    _check_exact(first_revised, first_mass)
    _check_exact(second_revised, second_mass)
    assert np.argmax(first_revised.identity_probabilities[0]) == 3
    _check_exact(old_prefix, prefix_mass)  # replay never mutates an earlier prefix

    # Geometry uses the same discrete weights and includes disagreement over
    # the conditional means, not the MAP branch or just the within-path variance.
    local = {_masks(spec): _integrate(spec, x1, 0., 1., 1., novel=True)[1:] for spec in _FIRST}
    mean = sum(first_mass[mask]*mu for mask, (mu, _) in local.items())
    variance = sum(first_mass[mask]*(var + (mu-mean)**2) for mask, (mu, var) in local.items())
    assert_allclose(first_revised.centers_mm[:2, 0], [mean, .4 + mean], rtol=0., atol=2.e-12)
    assert_allclose(first_revised.world_covariance_mm2[:2, 0, 0], variance, rtol=0., atol=2.e-12)

    # An empty view has the same missed-detection likelihood on each path.
    # It advances time without reassigning a single historical slot.
    empty = _step(reader, [])
    a, b, c = reader.relabel_history()
    _assert_estimate_equal(first_revised, a)
    _assert_estimate_equal(second_revised, b)
    _assert_estimate_equal(empty, c)
    assert empty.identity_probabilities.shape == (0, 1 << ROLE_COUNT)
    assert all(branch.path.masks == () for branch in reader._branches)


def test_first_usable_frame_seeds_after_unsupported_and_empty_opening():
    reader = _LinearReader((0,), noise=.7, prior_variance=400.)
    lag = np.ones(3 * ROLE_COUNT)
    reader._branches[0].ou_mean = lag
    reader._branches[0].ou_cov = .3 * reader._ou_initial_cov
    unsupported = _step(reader, [1.], h=-690., dt=.08)
    assert np.argmax(unsupported.role_probabilities[0]) == NULL_ROLE
    empty = _step(reader, [], dt=.05)
    assert empty.identity_probabilities.shape == (0, 1 << ROLE_COUNT)
    empty_path = reader._branches[0].path
    assert empty_path.masks == () and empty_path.previous.masks == (0,)
    assert_allclose(empty_path.log_evidence, np.log(.5), rtol=0., atol=1.e-12)
    assert reader.seed_calls == []

    found = _step(reader, [10. * np.pi], force=3.1, dt=.07)
    (parent, seeds), = reader.seed_calls
    assert seeds[0] is parent and len(seeds) == 6
    assert_array_equal(parent.mean, [0.])
    assert_allclose(seeds[1].mean, [10.*np.pi], atol=1.e-12, rtol=0.)
    assert parent.path is empty_path
    decay, _ = reader._ou.ou_transition(.20)
    assert_allclose(parent.ou_mean, decay * lag)
    assert_allclose(parent.ou_cov, (1. - .7 * decay**2) * reader._ou_initial_cov)
    assert np.argmax(found.role_probabilities[0]) == 0
    _step(reader, [10. * np.pi], force=3.1)
    assert len(reader.seed_calls) == 1
