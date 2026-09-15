from dataclasses import replace
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.contact_qp.geometry import (accepted_alpha_affine, affine_contact_motion,
    contact_cop_from_wrench, twist_tcp_to_face, wrench_tcp_to_face)
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.types import ContactStatus, ProbeGeometry, TwistConstraints
from peirastic.contact_qp.weighted_outer import (drop_unreachable_acceleration_rows,
    merge_exact_constraint_rows)


def config(**kwargs):
    return QpConfig(allocation_policy='delay_kf_cop_v1', max_acceleration=np.full(6, 1e6), **kwargs)


def datum(**kwargs):
    values = dict(geometry=ProbeGeometry.synthetic(), nominal_twist=np.zeros(6),
        path_twist=np.array([.01, 0, 0, 0, 0, 0]), force_n=4., dt_s=.005, now_s=1.,
        mechanical_normal_m_s=.001, mechanical_omega_rad_s=.02,
        path_feedforward_contact=np.array([.01, 0, 0, 0, 0, 0]),
        path_feedback_contact=np.array([.002, 0, 0, 0, 0, 0]), alpha_preferred=.7,
        cop_m=.015)
    values.update(kwargs)
    return QpInput(**values)


@pytest.mark.parametrize('visual', [False, True])
def test_feasible_mechanical_nominal_is_exact_with_inactive_visual(visual):
    d = datum(visual_task_valid=visual, visual_request_rad_s=0.)
    result = ContactQp(config()).solve(d)
    assert result.success, result.diagnostics
    np.testing.assert_array_equal(result.diagnostics['solution'], [.001, .02, .7, 0.])
    assert result.diagnostics['transparent']
    assert result.diagnostics['objective_normal'] == 0.
    assert result.diagnostics['objective_angular'] == 0.
    assert result.diagnostics['visual_rows_active'] is False
    assert result.energy_certificate is None
    assert not any('energy' in label or 'aperture' in label for label in result.hard_constraints.labels)


def test_nominal_satisfying_visual_is_not_added_a_second_time():
    result = ContactQp(config()).solve(datum(mechanical_omega_rad_s=.12,
        visual_task_valid=True, visual_request_rad_s=.1))
    assert result.success, result.diagnostics
    assert result.diagnostics['Omega_rad_s'] == .12
    assert result.diagnostics['sigma_I_rad_s'] == 0.
    assert result.diagnostics['U_m_s'] == .001


@pytest.mark.parametrize('sign', [-1, 1])
def test_joint_weighted_solution_matches_closed_form_including_opposed_torque(sign):
    mechanical = -.03*sign
    request = .12*sign
    cop = .018
    result = ContactQp(config()).solve(datum(mechanical_omega_rad_s=mechanical,
        visual_task_valid=True, visual_request_rad_s=request, cop_m=cop))
    assert result.success, result.diagnostics
    expected_omega = (mechanical+10.*request)/11.
    expected_normal = .001+cop*(expected_omega-mechanical)
    assert result.diagnostics['Omega_rad_s'] == pytest.approx(expected_omega, abs=2e-9)
    assert result.diagnostics['U_m_s'] == pytest.approx(expected_normal, abs=2e-10)
    assert result.diagnostics['sigma_I_rad_s'] == pytest.approx(abs(request)-sign*expected_omega, abs=2e-9)
    assert result.diagnostics['cop_pairing'] == 'coupled_normal_task_v1'
    assert result.diagnostics['cop_increment_residual_m_s'] == pytest.approx(0., abs=2e-12)
    assert result.diagnostics['objective_normal'] == pytest.approx(0., abs=1e-16)
    assert len(result.diagnostics['numeric_attempts']) <= 2


@pytest.mark.parametrize('force,cop', [(.799, .015), (4., .02501), (4., None), (-4., .015)])
def test_invalid_cop_disables_soft_term_instead_of_clamping(force, cop):
    result = ContactQp(config()).solve(datum(force_n=force, cop_m=cop,
        mechanical_omega_rad_s=0., visual_task_valid=True, visual_request_rad_s=.11))
    assert result.success, result.diagnostics
    assert result.diagnostics['cop_valid'] is False
    assert result.diagnostics['Omega_rad_s'] == pytest.approx(.1, abs=2e-9)
    assert result.diagnostics['U_m_s'] == pytest.approx(.001, abs=2e-10)


def test_visual_gate_scales_request_and_zero_gate_removes_row():
    d = datum(mechanical_omega_rad_s=0., visual_task_valid=True,
        visual_request_rad_s=.11, visual_gamma=.5, cop_m=None)
    r = ContactQp(config()).solve(d)
    assert r.success, r.diagnostics
    assert r.diagnostics['Omega_rad_s'] == pytest.approx(.05, abs=2e-9)
    r = ContactQp(config()).solve(replace(d, visual_gamma=0.))
    assert r.diagnostics['visual_rows_active'] is False
    assert r.diagnostics['Omega_rad_s'] == 0.


def test_affine_feedback_not_scaled_and_alpha_zero_keeps_contact_mechanics():
    d = datum(alpha_preferred=0.)
    r = ContactQp(config()).solve(d)
    assert r.success, r.diagnostics
    np.testing.assert_allclose(r.qp_twist, [.002, 0, .001, 0, .02, 0], atol=1e-15)
    assert r.alpha == 0.
    assert r.final_velocity_admissible(r.qp_twist, now_s=1.)
    assert not r.final_velocity_admissible(r.qp_twist, now_s=1.+config().certificate_horizon_s)


def test_noncoincident_rotated_face_uses_full_twist_and_affine_certificate():
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', [.4, -.3, .7]).as_matrix()
    t[:3, 3] = [.01, -.02, .08]
    d = datum(geometry=ProbeGeometry(.025, t), path_feedback_contact=[.002, .003, .8, .01, .8, 0.],
              path_feedforward_contact=[.01, .005, .9, .01, .9, .02])
    r = ContactQp(config()).solve(d)
    assert r.success, r.diagnostics
    expected_face = np.array([.009, .0065, .001, .017, .02, .014])
    np.testing.assert_allclose(twist_tcp_to_face(d.geometry)@r.qp_twist, expected_face, atol=1e-14)
    assert r.hard_constraints.violation(r.qp_twist) < 1e-12
    a, b = affine_contact_motion(d.geometry, d.path_feedback_contact, d.path_feedforward_contact)
    sent = a+b@[.001, .02, .6]
    predicted = a+b@[.001, .02, .5]
    assert accepted_alpha_affine(b, a, r.alpha, sent, predicted,
        config().subspace_tolerance) == pytest.approx(.5)
    assert r.hard_constraints.violation(a+b@[.001, .02, .8]) > .09


def test_tcp_limits_constrain_rotated_face_and_external_jerk_remains_hard():
    t = np.eye(4); t[:3, :3] = Rotation.from_euler('z', .5).as_matrix()
    d = datum(geometry=ProbeGeometry(.025, t), mechanical=TwistConstraints(
        np.eye(6)[[4]], [-.005], [.005], valid_until_s=1.05, labels=('published_jerk_y',)),
        visual_task_valid=True, visual_request_rad_s=.15)
    r = ContactQp(config()).solve(d)
    assert r.success, r.diagnostics
    assert abs(r.qp_twist[4]) <= .005+1e-8
    assert 'published_jerk_y' in r.diagnostics['active_hard_rows']
    assert r.diagnostics['sigma_I_rad_s'] > .14


def test_conflicting_mechanical_rows_are_never_softened_by_visual_slack():
    d = datum(mechanical=TwistConstraints(np.eye(6)[[0]], [.06], [.07], valid_until_s=1.1),
              visual_task_valid=True, visual_request_rad_s=.1)
    r = ContactQp(config()).solve(d)
    assert not r.success
    assert r.status == ContactStatus.MECHANICAL_INFEASIBLE
    assert r.qp_twist is None


def test_expired_solver_deadline_rejects_even_exact_nominal():
    r = ContactQp(config()).solve(datum(), deadline_s=time.monotonic()-1., online=True)
    assert r.status == ContactStatus.DEFERRED
    assert r.diagnostics['reason'] == 'solver_deadline_exceeded'


@pytest.mark.parametrize('x', [-.018, .018])
def test_point_load_cop_after_rotation_and_tcp_shift_with_tangential_force(x):
    t = np.eye(4); t[:3, :3] = Rotation.from_euler('xyz', [.5, -.3, .8]).as_matrix()
    t[:3, 3] = [.012, -.02, .06]
    geometry = ProbeGeometry(.025, t)
    force_face = np.array([1.2, -.4, -4.])
    wrench_face = np.r_[force_face, np.cross([x, 0, 0], force_face)]
    wrench_tcp = twist_tcp_to_face(geometry).T@wrench_face
    actual_face = wrench_tcp_to_face(wrench_tcp, geometry)
    cop, reason = contact_cop_from_wrench(actual_face)
    assert reason == 'valid'
    assert cop == pytest.approx(x, abs=1e-14)
    assert cop == pytest.approx(actual_face[4]/(-actual_face[2]))


def test_cop_gates_do_not_invent_centered_contact():
    assert contact_cop_from_wrench([0, 0, -.7, 0, .01, 0])[0] is None
    assert contact_cop_from_wrench([0, 0, 4., 0, .01, 0])[0] is None
    assert contact_cop_from_wrench([0, 0, -4., 0, .12, 0])[0] is None


def test_seek_slew_ignores_substep_rocking_gap_from_session006():
    # 006 RH_Per_L_DtP/002 first seek tick: leftover retract twist vs 0.64 ms
    # rocking gap. Default accel limits; identity face. Old path was empty.
    previous = np.array([-9.963132645691925e-05, -0.0009756655031079071,
                         0.0002986811879365003, 0.0010785198108304295,
                         0.0010566936261009361, 0.0011681979359987385])
    offset = np.array([0.00010350876992989112, 0.0004094106702881522, 0.,
                       0.0004957293284419115, 0., 0.00042729018761546617])
    common = dict(previous_twist=previous, path_feedback_contact=offset,
                  path_feedforward_contact=np.zeros(6), mechanical_normal_m_s=.001,
                  mechanical_omega_rad_s=0., force_n=-0.1882847475556741, cop_m=None,
                  visual_task_valid=True, visual_request_rad_s=0., alpha_preferred=.58)
    cfg = QpConfig(allocation_policy='delay_kf_cop_v1')
    tiny = ContactQp(cfg).solve(datum(dt_s=.0006385390006471425,
        acceleration_dt_s=.0006385390006471425, **common), online=True)
    assert not tiny.success
    assert tiny.diagnostics['reason'] == 'fixed_affine_component_outside_mechanical_interval'
    held = ContactQp(cfg).solve(datum(dt_s=.005, acceleration_dt_s=.0006385390006471425,
        **common), online=True)
    assert held.success, held.diagnostics
    assert held.diagnostics['command_slew_dt_s'] == pytest.approx(.005)
    assert held.qp_twist is not None


def test_exact_row_merge_preserves_intersection_and_does_not_merge_near_parallel():
    c=np.array([[1.,0.],[-1.,0.],[1.,0.],[0.,0.],[1.,1e-15]])
    lo=np.array([-2.,-3.,-.5,-1.,-4.]);hi=np.array([2.,1.,1.,1.,4.])
    merged,l,u=merge_exact_constraint_rows(c,lo,hi)
    assert len(merged)==2
    np.testing.assert_array_equal(merged[0],[1.,0.])
    assert l[0]==-.5 and u[0]==1.
    np.testing.assert_array_equal(merged[1],[1.,1e-15])
    for x in np.random.default_rng(8).normal(size=(100,2)):
        assert np.all((lo<=c@x)&(c@x<=hi))==np.all((l<=merged@x)&(merged@x<=u))
    merged,l,u=merge_exact_constraint_rows(np.zeros((1,2)),np.array([1.]),np.array([2.]))
    assert len(merged)==1 and l[0]==1.  # preserve contradictory constant row


@pytest.mark.parametrize('case',range(3))
def test_recorded_feasible_false_infeasibility_solves_with_exact_duplicate_merge(case):
    from pathlib import Path
    from peirastic.contact_qp.qp import _violation
    with np.load(Path(__file__).parent/'fixtures/contact_qp_weighted_exact_rows.npz') as saved:
        h,g,c,l,u=(saved[f'case_{case}_{key}'] for key in ('H','g','C','l','u'))
    numeric_c,numeric_l,numeric_u=merge_exact_constraint_rows(c,l,u)
    assert len(c)==16 and len(numeric_c)==4
    qp=ContactQp(config())
    x,_,solved,_=qp._solve_numeric_bounded(h,g,numeric_c,numeric_l,numeric_u,
        energy_active=False,deadline_s=None)
    assert solved,qp._numeric_attempts
    assert _violation(c,l,u,x)<=qp.config.feasibility_tolerance
    assert qp._numeric_attempts[-1]['dual_residual']<=qp.config.solver_tolerance
    assert abs(qp._numeric_attempts[-1]['duality_gap'])<=qp.config.solver_tolerance


def test_impossible_fixed_feedback_is_reported_before_numerical_retries():
    d=datum(previous_twist=[0.,.0085,0.,0.,0.,0.],
        path_feedback_contact=[0.,-.001,0.,0.,0.,0.],path_feedforward_contact=np.zeros(6),
        dt_s=.005,acceleration_dt_s=.005,cop_m=None)
    cfg=QpConfig(allocation_policy='delay_kf_cop_v1')
    result=ContactQp(cfg).solve(d,online=True)
    assert result.status==ContactStatus.MECHANICAL_INFEASIBLE
    assert result.diagnostics['reason']=='fixed_affine_component_outside_mechanical_interval'
    assert result.diagnostics['infeasible_fixed_rows'][0]['label']=='acceleration_1'
    assert not result.diagnostics['numeric_attempts']


def test_small_fixed_acceleration_overshoot_is_deferred_online():
    d=datum(previous_twist=np.zeros(6),
        path_feedback_contact=[0.,.0055,0.,0.,0.,0.],path_feedforward_contact=np.zeros(6),
        dt_s=.005,acceleration_dt_s=.005,cop_m=None,mechanical_omega_rad_s=0.)
    cfg=QpConfig(allocation_policy='delay_kf_cop_v1')
    online=ContactQp(cfg).solve(d,online=True)
    assert online.status==ContactStatus.DEFERRED
    assert online.diagnostics['reason']=='fixed_affine_acceleration_retry'
    assert online.diagnostics['infeasible_fixed_rows'][0]['label']=='acceleration_1'
    offline=ContactQp(cfg).solve(d,online=False)
    assert offline.status==ContactStatus.MECHANICAL_INFEASIBLE
    assert offline.diagnostics['reason']=='fixed_affine_component_outside_mechanical_interval'


def test_fixed_velocity_row_stays_mechanically_infeasible():
    d=datum(previous_twist=np.zeros(6),
        path_feedback_contact=[0.,.05,0.,0.,0.,0.],path_feedforward_contact=np.zeros(6),
        dt_s=.005,acceleration_dt_s=.005,cop_m=None,mechanical_omega_rad_s=0.)
    result=ContactQp(QpConfig(allocation_policy='delay_kf_cop_v1')).solve(d,online=True)
    assert result.status==ContactStatus.MECHANICAL_INFEASIBLE
    assert result.diagnostics['reason']=='fixed_affine_component_outside_mechanical_interval'
    labels={row['label'] for row in result.diagnostics['infeasible_fixed_rows']}
    assert 'velocity_1' in labels


def test_zero_row_roundoff_is_checked_in_physical_units():
    d=datum(previous_twist=[0.,.005+1e-10,0.,0.,0.,0.],
        path_feedback_contact=np.zeros(6),path_feedforward_contact=np.zeros(6),
        dt_s=.005,acceleration_dt_s=.005,cop_m=None,mechanical_omega_rad_s=0.)
    result=ContactQp(QpConfig(allocation_policy='delay_kf_cop_v1')).solve(d)
    assert result.success,result.diagnostics
    assert result.diagnostics['max_hard_violation']<1e-8


def _hardware_accel_cfg(**kwargs):
    return QpConfig(allocation_policy='delay_kf_cop_v1',
                    max_velocity=np.array([.22, .22, .015, .6, .28, .6]),
                    max_acceleration=np.array([1., 1., .8, 2., 2., 2.]), **kwargs)


def test_path_yaw_rate_drop_relaxes_only_conflicting_acceleration_online():
    # Previous tool-z spin is hotter than the current path feedforward band.
    # The accel box around that history misses alpha in [0,1]; U/Omega cannot help.
    previous = np.array([.002, 0., .001, 0., .02, .20])
    d = datum(previous_twist=previous, path_feedback_contact=np.zeros(6),
              path_feedforward_contact=[.002, 0., 0., 0., 0., .05],
              mechanical_normal_m_s=.001, mechanical_omega_rad_s=.02,
              alpha_preferred=1., dt_s=.005, acceleration_dt_s=.005, cop_m=None,
              visual_task_valid=True, visual_request_rad_s=0.)
    cfg = _hardware_accel_cfg()
    online = ContactQp(cfg).solve(d, online=True)
    assert online.success, online.diagnostics
    assert [row['label'] for row in online.diagnostics['relaxed_acceleration_rows']] == ['acceleration_5']
    assert online.qp_twist[5] == pytest.approx(.05, abs=1e-9)
    offline = ContactQp(cfg).solve(d, online=False)
    assert not offline.success
    assert offline.diagnostics['reason'] == 'affine_motion_subspace_conflict'
    assert 'relaxed_acceleration_rows' not in offline.diagnostics


def test_ultrapoc_002_terminal_yaw_rate_drop_solves_online():
    # Last rejected tick of Comparison Study/ultrapoc/002 (cid 6862, t_ref=46.22).
    d = datum(
        previous_twist=[0.01556837183087477, -0.0005806359386244092, -0.0006328892331416709,
                        -0.019702226075322567, 0.1243758077418761, -0.12076660688583725],
        path_feedback_contact=[-0.0007263466777532072, -0.0022363271384764076, 0.,
                               -0.002423104542857868, 0., -0.026907850570618404],
        path_feedforward_contact=[0.0021350028198943586, 0.00018967563846646632, -7.326753595774665e-05,
                                  -0.0035220861143232234, 0.03270419814435191, -0.018514868866950324],
        mechanical_normal_m_s=-0.0012966544122483257, mechanical_omega_rad_s=0.15828486879683273,
        dt_s=0.005, acceleration_dt_s=0.014715763012645766, force_n=4.236630222796778,
        cop_m=0.013525114450338454, alpha_preferred=1., visual_task_valid=True,
        visual_request_rad_s=0., visual_gamma=0.5267395544064435,
        measured_angle=-0.07145609948099549)
    cfg = _hardware_accel_cfg()
    rows = np.eye(6)
    lo = d.previous_twist - cfg.max_acceleration * max(d.acceleration_dt_s, d.dt_s)
    hi = d.previous_twist + cfg.max_acceleration * max(d.acceleration_dt_s, d.dt_s)
    labels = [f'acceleration_{i}' for i in range(6)]
    offset, basis = affine_contact_motion(d.geometry, d.path_feedback_contact, d.path_feedforward_contact)
    _, _, _, _, dropped = drop_unreachable_acceleration_rows(
        rows, lo, hi, labels, offset, basis, cfg)
    assert [row['label'] for row in dropped] == ['acceleration_5']
    online = ContactQp(cfg).solve(d, online=True)
    assert online.success, online.diagnostics
    assert [row['label'] for row in online.diagnostics['relaxed_acceleration_rows']] == ['acceleration_5']
    assert -0.04543 <= online.qp_twist[5] <= -0.02690
