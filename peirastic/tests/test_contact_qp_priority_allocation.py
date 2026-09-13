"""Priority optima survive CoP, hard limits, empty tanks and deadlines."""
import numpy as np
import pytest

from peirastic.contact_qp import qp as module
from peirastic.contact_qp import priority_allocation
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.types import ContactObservation, ContactStatus, ProbeGeometry, TwistConstraints


def config(policy='confidence_cop_v1', **kwargs):
    return QpConfig(allocation_policy=policy, max_acceleration=np.full(6, 1000.), **kwargs)


def datum(**kwargs):
    path = np.array([0., .02, 0., 0., 0., 0.])
    values = dict(geometry=ProbeGeometry.synthetic(), nominal_twist=path, path_twist=path,
        force_n=4., dt_s=.005, now_s=1., alpha_preferred=.75,
        visual_omega_target_rad_s=.2, loading_velocity_m_s=.001, cop_m=.025,
        observation=ContactObservation(1, 'camera', 1., 1., [.3]*3, [True]*3, 'reg', 'window'))
    values.update(kwargs)
    return QpInput(**values)


@pytest.mark.parametrize('cop', [-.025, .0, .025])
def test_cop_cannot_buy_progress_or_rocking_to_improve_normal_pairing(cop):
    result = ContactQp(config()).solve(datum(cop_m=cop), online=True)
    assert result.success, result.diagnostics
    assert result.alpha == pytest.approx(.75, abs=1e-8)
    assert result.qp_twist[4] == pytest.approx(.2, abs=1e-8)
    assert result.qp_twist[2] == pytest.approx(np.clip(.001+cop*.2, .00025, .00175), abs=1e-8)
    assert result.final_velocity_admissible(result.qp_twist)
    assert not result.diagnostics['visual_rows_active']
    assert result.diagnostics['priority_order'] == ('progress', 'visual_omega', 'normal_pairing')
    assert len(result.diagnostics['priority_stages']) == 3
    assert all(stage['duration_s'] >= 0. for stage in result.diagnostics['priority_stages'])
    assert all(attempt['max_inner_iterations'] == 10 for attempt in result.diagnostics['numeric_attempts'])


def test_angular_only_pairs_full_omega_with_zero_arm_and_needs_no_tissue_model():
    result = ContactQp(config('confidence_angular_v1')).solve(datum(cop_m=None))
    assert result.success and result.qp_twist[2] == pytest.approx(.001, abs=1e-8)
    assert result.qp_twist[4] == pytest.approx(.2, abs=1e-8)
    assert result.diagnostics['cop_m'] == 0.
    assert result.diagnostics['pairing_residual_m_s'] == pytest.approx(0., abs=1e-8)


def test_angular_limit_pairs_with_achieved_full_omega_not_target_or_delta():
    mechanics = TwistConstraints(np.eye(6)[[4]], [-.1], [.1], valid_until_s=1.01)
    result = ContactQp(config(enable_aperture=False)).solve(datum(mechanical=mechanics))
    assert result.success, result.diagnostics
    assert result.alpha == pytest.approx(.75, abs=1e-8)
    assert result.qp_twist[4] == pytest.approx(.1, abs=1e-8)
    assert result.qp_twist[2] == pytest.approx(.001+.025*.1, abs=1e-8)
    assert result.diagnostics['normal_pairing_target_m_s'] == pytest.approx(.0035, abs=1e-8)


def test_progress_is_established_before_angular_target():
    row = np.array([[0., 1., 0., 0., .1, 0.]])
    mechanics = TwistConstraints(row, [-np.inf], [.025], valid_until_s=1.01)
    result = ContactQp(config(enable_aperture=False)).solve(datum(mechanical=mechanics))
    assert result.success, result.diagnostics
    assert result.alpha == pytest.approx(.75, abs=1e-8)
    assert result.qp_twist[4] == pytest.approx(.1, abs=1e-8)


@pytest.mark.parametrize('available, expected', [(0., 0.), (.00025, .5)])
def test_full_final_energy_can_reduce_progress(available, expected):
    energy = PortEnergyConstraint([0., -5., 0., 0., 0., 0.], available, .005)
    result = ContactQp(config()).solve(datum(energy=energy))
    assert result.success, result.diagnostics
    assert result.alpha == pytest.approx(expected, abs=1e-8)
    assert result.qp_twist[4] == pytest.approx(.2, abs=1e-8)
    assert 'alpha_reduced_by_hard_constraints' in result.diagnostics['allocation_reason_codes']
    assert energy.admissible(result.qp_twist)
    assert result.energy_certificate is energy


def test_energy_epigraphs_are_enforced_with_uncertainty_and_damping():
    energy = PortEnergyConstraint([0., -5., 0., 0., 0., 0.], .0003, .005,
        wrench_error=[0., 1., 0., 0., 0., 0.], contact_map=np.eye(6)[[1]],
        damping=[1.], contact_speed_bound=[.02])
    result = ContactQp(config()).solve(datum(energy=energy))
    assert result.success, result.diagnostics
    assert result.alpha < .5
    assert energy.admissible(result.qp_twist)
    assert not result.diagnostics['exact_priority_fast_path']


def test_degenerate_active_energy_lock_either_certifies_or_defers_safely():
    # A rank-deficient active epigraph can expose a subnanometric infeasible
    # lock after the first solve. Never weaken the energy row to rescue it.
    energy = PortEnergyConstraint([0., -5., 0., 0., 0., 0.], .00025, .005,
        wrench_error=[0., 1., 0., 0., 0., 0.], contact_map=np.eye(6)[[1]],
        damping=[1.], contact_speed_bound=[.02])
    result = ContactQp(config()).solve(datum(energy=energy), online=True)
    if result.success:
        assert energy.admissible(result.qp_twist)
        assert result.alpha == pytest.approx(.05 / (.02*6.02), abs=1e-8)
    else:
        assert result.status == ContactStatus.DEFERRED and result.qp_twist is None
        assert result.diagnostics['reason'] == 'solver_attempts_exhausted'


def test_hard_mechanical_limit_reduces_progress_and_is_exported():
    mechanics = TwistConstraints(np.eye(6)[[1]], [0.], [.012], valid_until_s=1.01)
    result = ContactQp(config()).solve(datum(mechanical=mechanics))
    assert result.success and result.alpha == pytest.approx(.6, abs=1e-8)
    assert result.final_velocity_admissible(result.qp_twist)


def test_infeasible_problem_defers_without_online_lp(monkeypatch):
    forbidden = lambda *a, **k: pytest.fail('priority allocation invoked an unbounded LP')
    monkeypatch.setattr(module, '_linear_feasible', forbidden)
    monkeypatch.setattr(module, '_omega_interval', forbidden)
    mechanics = TwistConstraints(np.eye(6)[[1]], [-.02], [-.01], valid_until_s=1.01)
    result = ContactQp(config()).solve(datum(mechanical=mechanics), online=True, interval_diagnostics=True)
    assert result.status == ContactStatus.DEFERRED and result.qp_twist is None
    assert result.diagnostics['failed_priority_stage'] == 'progress'
    assert len(result.diagnostics['numeric_attempts']) == 2


def test_zero_path_fixes_unobservable_alpha():
    result = ContactQp(config()).solve(datum(path_twist=np.zeros(6), nominal_twist=np.zeros(6)))
    assert result.success and result.alpha == 0.
    assert result.qp_twist[4] == pytest.approx(.2, abs=1e-8)


@pytest.mark.parametrize('field', ['visual_omega_target_rad_s', 'loading_velocity_m_s', 'cop_m'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_targets_rejected(field, value):
    with pytest.raises(ValueError, match='finite'):
        datum(**{field: value})


@pytest.mark.parametrize('missing', ['visual_omega_target_rad_s', 'loading_velocity_m_s', 'cop_m'])
def test_missing_required_target_does_not_fall_back_to_legacy(missing):
    result = ContactQp(config()).solve(datum(**{missing: None}))
    assert result.status == ContactStatus.CERTIFICATE_INVALID and result.qp_twist is None


def test_cop_must_lie_on_transformed_physical_face():
    result = ContactQp(config()).solve(datum(cop_m=.026))
    assert result.status == ContactStatus.CERTIFICATE_INVALID


def test_one_absolute_deadline_fences_every_priority_stage(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    solver = ContactQp(config())
    original = solver._solve_numeric_bounded
    deadlines = []

    def wrapped(*args, **kwargs):
        deadlines.append(kwargs['deadline_s'])
        result = original(*args, **kwargs)
        clock[0] = 1.
        return result

    monkeypatch.setattr(solver, '_solve_numeric_bounded', wrapped)
    result = solver.solve(datum(), deadline_s=1., online=True)
    assert result.status == ContactStatus.DEFERRED and result.qp_twist is None
    assert deadlines == [1., 1.]
    assert result.diagnostics['reason'] == 'solver_deadline_exceeded'
    assert result.diagnostics['failed_priority_stage'] == 'visual_omega'


@pytest.mark.parametrize('policy', ['confidence_angular_v1', 'confidence_cop_v1'])
@pytest.mark.parametrize('zero_path', [False, True])
def test_exact_all_target_optimum_matches_bounded_numeric_path(monkeypatch, policy, zero_path):
    data = datum(cop_m=.002, energy=PortEnergyConstraint([0, 0, -4, 0, -.04, 0], .01, .005))
    if zero_path:
        from dataclasses import replace
        data = replace(data, path_twist=np.zeros(6), nominal_twist=np.zeros(6))
    exact = ContactQp(config(policy)).solve(data)
    assert exact.success and exact.diagnostics['exact_priority_fast_path']
    assert not exact.diagnostics['transparent']
    assert exact.diagnostics['solver_status'] == 'exact_priority_optimum'
    assert exact.diagnostics['iterations'] == 0
    assert exact.diagnostics['numeric_attempts'] == ()
    assert len(exact.diagnostics['priority_stages']) == 3
    assert all(stage['solver_status'] == 'exact_priority_optimum'
               for stage in exact.diagnostics['priority_stages'])
    monkeypatch.setattr(priority_allocation, '_exact_priority_candidate', lambda *a, **kw: None)
    numeric = ContactQp(config(policy)).solve(data)
    assert numeric.success and not numeric.diagnostics['transparent']
    assert exact.status == numeric.status
    assert not numeric.diagnostics['exact_priority_fast_path']
    assert len(numeric.diagnostics['numeric_attempts']) >= 3
    np.testing.assert_allclose(exact.qp_twist, numeric.qp_twist, atol=1e-8, rtol=0)
    assert exact.alpha == pytest.approx(numeric.alpha, abs=1e-8)
    assert exact.final_velocity_admissible(exact.qp_twist)


def test_fast_path_cannot_skip_expired_deadline_or_final_deadline_fence(monkeypatch):
    clock = iter([0., 1.])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(clock))
    result = ContactQp(config()).solve(datum(cop_m=0.), deadline_s=1., online=True)
    assert result.status == ContactStatus.DEFERRED and result.qp_twist is None
    assert result.diagnostics['exact_priority_fast_path']
    assert result.diagnostics['reason'] == 'solver_deadline_exceeded'
    monkeypatch.setattr(module.time, 'monotonic', lambda: 1.)
    monkeypatch.setattr(priority_allocation, '_exact_priority_candidate',
                        lambda *a, **kw: pytest.fail('expired input reached fast path'))
    result = ContactQp(config()).solve(datum(cop_m=0.), deadline_s=1., online=True)
    assert result.status == ContactStatus.DEFERRED and result.qp_twist is None


def test_fast_candidate_requires_stricter_solver_tolerance_for_hard_rows():
    qp = ContactQp(config())
    data = datum(cop_m=0.)
    scales = np.array([.002, .1, 1.])
    basis = module.motion_basis(data.path_twist)*scales
    candidate = np.array([.5, 2., .75])
    # Cartesian violation passes final feasibility tolerance but must not be
    # accepted by the stricter algebraic optimum admission shortcut.
    hard_rows = (np.eye(6)[[2]], np.array([-np.inf]), np.array([.001-5e-9]))
    assert priority_allocation._exact_priority_candidate(qp, data, basis, scales,
        np.eye(3), candidate-1., candidate+1., preferred=.75, arm=0., hard_rows=hard_rows) is None


def test_fast_candidate_uses_feasibility_tolerance_when_stricter_than_solver():
    qp = ContactQp(config(solver_tolerance=1e-6, feasibility_tolerance=1e-8))
    data = datum(cop_m=0.)
    scales = np.array([.002, .1, 1.])
    basis = module.motion_basis(data.path_twist)*scales
    candidate = np.array([.5, 2., .75])
    hard_rows = (np.eye(6)[[2]], np.array([-np.inf]), np.array([.001-5e-8]))
    assert priority_allocation._exact_priority_candidate(qp, data, basis, scales,
        np.eye(3), candidate-1., candidate+1., preferred=.75, arm=0., hard_rows=hard_rows) is None


def test_fast_candidate_rejects_even_tiny_negative_direct_energy_margin():
    qp = ContactQp(config())
    energy = PortEnergyConstraint([0, 0, -4, 0, 0, 0], (.004-5e-10)*.005, .005)
    data = datum(cop_m=0., energy=energy)
    scales = np.array([.002, .1, 1.])
    basis = module.motion_basis(data.path_twist)*scales
    candidate = np.array([.5, 2., .75])
    twist = basis @ candidate
    assert -qp.config.solver_tolerance < energy.margin_power_w(twist) < 0.
    assert energy.admissible(twist, tolerance_w=qp.config.solver_tolerance)
    hard_rows = (np.eye(6), -np.ones(6), np.ones(6))
    assert priority_allocation._exact_priority_candidate(qp, data, basis, scales,
        np.eye(3), candidate-1., candidate+1., preferred=.75, arm=0., hard_rows=hard_rows) is None
