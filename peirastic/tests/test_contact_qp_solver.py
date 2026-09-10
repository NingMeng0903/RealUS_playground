"""Portable numerical/geometry tests for the physical three-variable QP."""
from dataclasses import replace
import math
import time

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

from peirastic.contact_qp.geometry import aperture_rows, motion_basis, point_normal_row, window_rows
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput, _omega_interval
from peirastic.contact_qp.types import ContactObservation, ContactStatus, ProbeGeometry, TwistConstraints


B = np.array([.02, 0., 0., 0., 0., 0.])
FAST = QpConfig(max_acceleration=np.full(6, 1000.))


def observation(quality=(.8, .8, .8), valid=(True, True, True), *, now=1., seq=1):
    return ContactObservation(seq, 'synthetic', now, now, np.asarray(quality), np.asarray(valid),
                              'registration_v1', 'windows_v1')


def datum(quality=(.8, .8, .8), **kwargs):
    values = dict(geometry=ProbeGeometry.synthetic(), nominal_twist=B.copy(), path_twist=B,
                  force_n=4., dt_s=.005, now_s=1., observation=observation(quality), previous_twist=B)
    values.update(kwargs)
    return QpInput(**values)


def test_design_shadow_asymmetric_gamma_converges_at_inactive_slack_boundary():
    """Captured design seed 0 input; inner cap 32 stalled above the dual tolerance."""
    cfg = QpConfig(enable_consistency=True, angle_limit_rad=.35)
    nominal = np.array([0., .02, .00024980312713129, 0., .00360921739154128, 0.])
    previous = np.array([0., .00833474291454225, .00040727295264347, 0., .00672916977539819, 0.])
    mechanics = TwistConstraints(np.eye(6), previous-cfg.max_acceleration*.005,
                                previous+cfg.max_acceleration*.005, valid_until_s=1.01)
    data = datum((.11115809715140852, .21729224921338502, .41054465368162607),
                 nominal_twist=nominal, path_twist=np.array([0., .02, 0., 0., 0., 0.]),
                 previous_twist=previous, mechanical=mechanics, force_n=4.140757474388942,
                 measured_angle=.018792306803429915, gamma=[1., .3723238800370152])
    solver = ContactQp(cfg)
    result = solver.solve(data)
    assert result.success
    assert result.diagnostics["solver_status"] == "QPSolverOutput.PROXQP_SOLVED"
    assert solver._solver.results.info.pri_res <= cfg.solver_tolerance
    assert solver._solver.results.info.dua_res <= cfg.solver_tolerance
    assert result.hard_constraints.violation(result.qp_twist) <= cfg.feasibility_tolerance
    assert result.slack[0] > .001 and result.slack[1] == 0.


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_inner_iteration_budget_requires_positive_integer(value):
    with pytest.raises(ValueError):
        QpConfig(inner_iterations=value)


def test_composed_reference_shadow_input_uses_physical_normalization_without_dual_stall():
    """Design seed 0 v2: second equilibration stalls even with 1500 inner iterations."""
    cfg = QpConfig(enable_consistency=True, angle_limit_rad=.35)
    path = np.array([8.665174719444024e-7, .01999976342291000, 0., 0., 0., 0.])
    nominal = path.copy()
    nominal[[2, 4]] = [.0004149122753827067, .006008427719607738]
    previous = np.array([3.1723520361563426e-7, .008336007872514168,
                         .00058677817551403466, 0., .00923220671562546, 0.])
    mechanics = TwistConstraints(np.eye(6), previous-cfg.max_acceleration*.005,
                                previous+cfg.max_acceleration*.005, valid_until_s=1.01)
    data = datum((.11120365015822455, .42056692971253123, .4600684944505315),
                 nominal_twist=nominal, path_twist=path, previous_twist=previous,
                 mechanical=mechanics, force_n=4.133765922223868,
                 measured_angle=.025863094034819133, gamma=[1., .2887502167539988])
    solver = ContactQp(cfg)
    for _ in range(3):  # exercise initial and cached-workspace paths
        result = solver.solve(data)
        assert result.success
        assert result.diagnostics["solver_status"] == "QPSolverOutput.PROXQP_SOLVED"
        assert solver._solver.results.info.pri_res <= cfg.solver_tolerance
        assert solver._solver.results.info.dua_res <= cfg.solver_tolerance
        assert result.hard_constraints.violation(result.qp_twist) <= cfg.feasibility_tolerance
        assert result.diagnostics["iterations"] < 30


def test_solver_preconditioning_policy_is_explicit_and_validated():
    assert QpConfig().solver_preconditioning is False
    with pytest.raises(ValueError):
        QpConfig(solver_preconditioning="false")


def test_100000_ticks_exact_transparency_and_no_downstream_state_drift():
    solver = ContactQp()
    old_state, new_state = np.zeros(6), np.zeros(6)
    previous = B.copy()
    elapsed = time.perf_counter()
    max_error = 0.
    for i in range(100000):
        now = 1. + i * .005
        b = np.array([.02 + .0001 * math.sin(i * .001), 0., 0., 0., 0., 0.])
        nominal = b.copy()
        nominal[2] = .0002 * math.sin(i * .005)
        nominal[4] = .01 * math.sin(i * .003)
        result = solver.solve(QpInput(ProbeGeometry.synthetic(), nominal, b,
                                      4. + .2 * math.sin(i * .002), .005, now,
                                      observation((1., 0., 1.), now=now, seq=i),
                                      previous_twist=previous))
        assert result.success and result.diagnostics['transparent']
        assert result.alpha == 1. and np.array_equal(result.slack, np.zeros(2))
        error = float(np.max(np.abs(result.qp_twist - nominal)))
        max_error = max(max_error, error)
        if error != 0.:
            pytest.fail(f'QP changed inactive nominal at tick {i}: {error}')
        old_state += nominal * .005
        new_state += result.qp_twist * .005
        previous = nominal
    np.testing.assert_array_equal(old_state, new_state)
    assert solver._solver is None  # the exact optimum does not need numerical optimization
    print(dict(ticks=100000, max_absolute_twist_error=max_error,
               downstream_state_error=float(np.max(np.abs(old_state-new_state))),
               elapsed_s=time.perf_counter()-elapsed))


def test_image_left_right_mirror_and_center_never_repairs():
    cfg = replace(FAST, enable_aperture=False, enable_force_priority=False)
    solver = ContactQp(cfg)
    left = solver.solve(datum((.1, .8, .8)))
    right = solver.solve(datum((.8, .8, .1)))
    assert left.success and right.success
    assert left.qp_twist[4] > 0. and right.qp_twist[4] < 0.
    np.testing.assert_allclose(left.qp_twist[[0, 2]], right.qp_twist[[0, 2]], atol=1e-10)
    assert left.qp_twist[4] == pytest.approx(-right.qp_twist[4], abs=1e-10)
    center = solver.solve(datum((.8, 0., .8), observation=observation((.8, 0., .8), (True, False, True))))
    np.testing.assert_array_equal(center.qp_twist, B)
    assert center.alpha == 1. and center.diagnostics['image_valid']


def test_gamma_scales_only_deficit_keep_margin_rows_persist():
    solver = ContactQp(replace(FAST, enable_consistency=True))
    full = solver.solve(datum((.1, .8, .8), gamma=np.ones(2)))
    zero = solver.solve(datum((.1, .8, .8), gamma=np.zeros(2)))
    a = full.diagnostics['visual_requests_m_s']
    b = zero.diagnostics['visual_requests_m_s']
    assert a[0] > 0. and b[0] == 0.
    assert a[1] == b[1] < 0.
    assert full.diagnostics['acquisition_loss'] == zero.diagnostics['acquisition_loss']
    repaired = solver.solve(datum((.8, .8, .8), gamma=np.zeros(2)))
    assert repaired.diagnostics['visual_rows_active']
    assert np.all(repaired.diagnostics['visual_requests_m_s'] < 0.)
    without_consistency = ContactQp(FAST).solve(datum((.1, .8, .8), gamma=0.))
    np.testing.assert_array_equal(without_consistency.diagnostics['gamma_effective'], np.ones(2))


@pytest.mark.parametrize('kind', ['none', 'stale', 'invalid_required', 'future_receive'])
def test_invalid_image_disables_visual_rows_with_maximal_loss(kind):
    obs = observation((.1, .1, .1))
    if kind == 'none':
        obs = None
    elif kind == 'stale':
        obs = replace(obs, effective_time_s=.1, received_time_s=.1)
    elif kind == 'invalid_required':
        obs = replace(obs, valid=np.array([False, True, True]))
    else:
        obs = replace(obs, received_time_s=1.1)
    result = ContactQp(FAST).solve(datum(observation=obs))
    assert result.success and result.status == ContactStatus.IMAGE_UNAVAILABLE
    assert not result.diagnostics['visual_rows_active']
    assert result.diagnostics['acquisition_loss'] == 1.
    assert result.diagnostics['alpha_preferred'] == .25
    assert result.alpha == pytest.approx(.25, abs=1e-8)
    np.testing.assert_array_equal(result.slack, np.zeros(2))


def test_progress_preference_normalized_but_hard_alpha_can_be_zero():
    mech = TwistConstraints(np.array([[1., 0., 0., 0., 0., 0.]]), np.array([0.]), np.array([0.]), valid_until_s=2.)
    result = ContactQp(FAST).solve(datum((0., .8, 0.), mechanical=mech))
    assert result.success and result.alpha <= 1e-8
    assert result.diagnostics['alpha_preferred'] == .25
    no_loss = ContactQp(replace(FAST, enable_progress_loss=False)).solve(datum((0., .8, 0.)))
    assert no_loss.alpha == pytest.approx(1., abs=1e-8)
    assert no_loss.diagnostics['acquisition_loss'] == 1.


@pytest.mark.parametrize('force_n', [3., 5., -.01])
def test_reliable_force_priority_and_aperture_hold_over_entire_interval(force_n):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [.2, .3, -.1]).as_matrix()
    transform[:3, 3] = [.013, -.004, .007]
    geometry = ProbeGeometry.synthetic(T_tcp_face=transform)
    data = datum((.1, .8, .9), geometry=geometry, force_n=force_n)
    result = ContactQp(FAST).solve(data, interval_diagnostics=True)
    assert result.success
    for x in np.linspace(-.025, .025, 101):
        row = point_normal_row(geometry, x)
        row[[0, 1, 3, 5]] = 0.
        added = row @ (result.qp_twist - data.nominal_twist)
        assert abs(added) <= FAST.aperture_budget_m_s + 1e-8
        assert np.sign(force_n - 4.) * added <= 1e-8
    assert result.hard_constraints.violation(result.qp_twist) <= 1e-8
    assert result.diagnostics['omega_interval_complete'] is not None


def test_general_window_rows_include_offset_rotation_path_and_image_axis():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [.2, .5, .1]).as_matrix()
    transform[:3, 3] = [.012, .003, .015]
    geometry = ProbeGeometry.synthetic(T_tcp_face=transform, image_x_sign=-1)
    cfg = replace(FAST, enable_aperture=False, enable_force_priority=False)
    data = datum((.1, .8, .7), geometry=geometry)
    result = ContactQp(cfg).solve(data)
    assert result.success
    for j, w in enumerate((0, 2)):
        interval = cfg.lateral_windows[w]
        face_x = geometry.image_x_sign * geometry.half_length_m * (sum(interval) - 1.)
        point = transform[:3, 3] + face_x * transform[:3, 0]
        speed = transform[:3, 2] @ (result.qp_twist[:3] + np.cross(result.qp_twist[3:], point))
        request = result.diagnostics['visual_requests_m_s'][j]
        assert speed + result.slack[j] >= request - 1e-8
        assert speed == pytest.approx(window_rows(geometry, cfg.lateral_windows)[w] @ result.qp_twist)


def test_no_aperture_removes_budget_and_cost_but_retains_force_priority():
    cfg = replace(FAST, aperture_budget_m_s=0.)
    data = datum((.1, .8, .8), force_n=3.)
    full = ContactQp(cfg).solve(data)
    no_aperture = ContactQp(replace(cfg, enable_aperture=False, aperture_cost_weight=1e6)).solve(data)
    zero_cost = ContactQp(replace(cfg, enable_aperture=False, aperture_cost_weight=0.)).solve(data)
    assert full.success and no_aperture.success
    assert abs(full.qp_twist[2]) < 1e-8
    assert no_aperture.qp_twist[2] > .0001
    np.testing.assert_array_equal(no_aperture.qp_twist, zero_cost.qp_twist)
    assert not any('aperture' in label for label in no_aperture.hard_constraints.labels)
    assert sum('force_priority' in label for label in no_aperture.hard_constraints.labels) == 2


def test_mechanical_infeasibility_distinguished_from_task_conflict():
    row = np.array([[0., 0., 1., 0., 0., 0.]])
    impossible = TwistConstraints(row, np.array([.1]), np.array([.2]), valid_until_s=2.)
    mech_result = ContactQp(FAST).solve(datum(mechanical=impossible))
    assert not mech_result.success and mech_result.status == ContactStatus.MECHANICAL_INFEASIBLE
    press_required = TwistConstraints(row, np.array([.001]), np.array([math.inf]), valid_until_s=2.)
    for force in (4., 5.):
        result = ContactQp(FAST).solve(datum(mechanical=press_required, force_n=force))
        assert not result.success and result.status == ContactStatus.TASK_INFEASIBLE
        assert result.qp_twist is None
    admitted = ContactQp(replace(FAST, enable_aperture=False, enable_force_priority=False)).solve(datum(mechanical=press_required))
    assert admitted.success and admitted.qp_twist[2] >= .001 - 1e-8


def test_velocity_acceleration_angle_and_final_exported_progress_subspace():
    cfg = replace(FAST, angle_limit_rad=.35)
    result = ContactQp(cfg).solve(datum((.1, .8, .8), measured_angle=.35))
    assert result.success and result.qp_twist[4] <= 1e-8
    limits = result.hard_constraints
    assert not any('visual' in name for name in limits.labels)
    faster = result.qp_twist.copy()
    faster[0] += .001
    assert limits.violation(faster) > 1e-5
    off_subspace = result.qp_twist.copy()
    off_subspace[1] += .001
    assert limits.violation(off_subspace) > 1e-5
    assert math.isfinite(limits.valid_until_s) and limits.valid_until_s > 1.
    zero = ContactQp(FAST).solve(datum(nominal_twist=np.zeros(6), path_twist=np.zeros(6)))
    assert zero.success and zero.alpha == 0.


def test_certificate_expiry_and_unsupported_normal_recovery_fail_closed():
    a = np.array([[1., 0., 0., 0., 0., 0.]])
    for expiry in (math.inf, 1., .9):
        mech = TwistConstraints(a, np.array([-1.]), np.array([1.]), valid_until_s=expiry)
        result = ContactQp().solve(datum(mechanical=mech))
        assert result.status == ContactStatus.CERTIFICATE_INVALID and not result.success
    recovery = B.copy()
    recovery[1] = .001
    result = ContactQp().solve(datum(nominal_twist=recovery))
    assert result.status == ContactStatus.TASK_INFEASIBLE
    assert 'motion_basis' in result.diagnostics['reason']


def test_projected_omega_interval_matches_independent_linear_program():
    rng = np.random.default_rng(218)
    for _ in range(40):
        a = np.vstack((np.eye(3), rng.normal(size=(8, 3))))
        lo, hi = -np.ones(len(a)), np.ones(len(a))
        lo[3:] -= rng.random(len(a)-3)
        hi[3:] += rng.random(len(a)-3)
        interval = _omega_interval(a, lo, hi)
        expected = []
        for sign in (1., -1.):
            out = linprog(np.array([0., sign, 0.]), A_ub=np.vstack((a, -a)), b_ub=np.r_[hi, -lo],
                          bounds=[(None, None)] * 3, method='highs')
            assert out.success
            expected.append(out.x[1])
        np.testing.assert_allclose(interval, expected, atol=1e-8, rtol=1e-8)
    projected = ContactQp(FAST).solve(datum(force_n=5.), interval_diagnostics=True)
    intervals = projected.diagnostics
    assert intervals['omega_interval_complete'][0] >= intervals['omega_interval_mechanical'][0] - 1e-8
    assert intervals['omega_interval_complete'][1] <= intervals['omega_interval_mechanical'][1] + 1e-8


def test_immutable_validated_configuration_inputs_outputs():
    b = B.copy()
    data = datum(path_twist=b)
    b[0] = 999.
    assert data.path_twist[0] == .02
    with pytest.raises(ValueError):
        data.path_twist[0] = 999.
    with pytest.raises(ValueError):
        QpConfig(force_target_n=5.)
    with pytest.raises(ValueError):
        QpConfig(enable_aperture='yes')
    with pytest.raises(ValueError):
        datum(force_n=np.nan)
    with pytest.raises(ValueError):
        datum(gamma=[1., 2.])
    with pytest.raises(ValueError):
        datum(path_twist=np.ones(6))
    result = ContactQp().solve(data)
    with pytest.raises(TypeError):
        result.diagnostics['port_verified'] = True
    with pytest.raises(ValueError):
        result.hard_constraints.A[0, 0] = 123.
