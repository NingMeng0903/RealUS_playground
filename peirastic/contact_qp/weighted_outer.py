"""One weighted, four-variable, energy-free affine contact allocation.

Physical variables are [U (m/s), Omega (rad/s), alpha, sigma_I (rad/s)].
Only the numerical problem is scaled. CoP couples deviations from the
mechanical nominal; the image inequality acts on the final contact Omega.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time

import numpy as np

from .geometry import affine_contact_motion, twist_tcp_to_face
from .types import ContactStatus, TwistConstraints


def merge_exact_constraint_rows(matrix, lower, upper):
    """Merge only byte-for-byte equal rows after canonical sign selection.

    Parallel near-equality is deliberately not inferred. This is an exact
    algebraic rewrite of duplicate/negative rows, not a tolerance change.
    Original rows remain the final admission and exported-certificate checks.
    """
    rows={}
    for coefficients,lo,hi in zip(matrix,lower,upper):
        row=np.array(coefficients,copy=True)
        nonzero=np.flatnonzero(row)
        if not len(nonzero):
            if lo<=0.<=hi:continue
        elif row[nonzero[0]]<0.:
            row=-row;lo,hi=-hi,-lo
        row[row==0.]=0.  # canonical positive zero, without changing any value
        key=tuple(row)
        if key in rows:
            old_lo,old_hi=rows[key];rows[key]=(max(old_lo,lo),min(old_hi,hi))
        else:rows[key]=(lo,hi)
    return (np.asarray(list(rows),dtype=float).reshape(-1,matrix.shape[1]),
            np.asarray([bounds[0] for bounds in rows.values()]),
            np.asarray([bounds[1] for bounds in rows.values()]))


def _affine_box_range(offset, basis, axis, bounds):
    lo = hi = float(offset[axis])
    for coeff, (b0, b1) in zip(basis[axis], bounds):
        c = float(coeff)
        lo += min(c * b0, c * b1)
        hi += max(c * b0, c * b1)
    return lo, hi


def drop_unreachable_acceleration_rows(hard_a, hard_lo, hard_hi, labels, offset, basis, cfg):
    """Drop accel rows whose affine image of (U, Omega, alpha) misses the slew box.

    Fixed axes (zero basis row) stay; those still use the existing
    infeasible / small-overshoot retry policy. Path-coupled axes can leave
    the previous-command box when feedforward rate drops in one tick.
    """
    bounds = ((-cfg.max_velocity[2], cfg.max_velocity[2]),
              (-cfg.max_velocity[4], cfg.max_velocity[4]), (0., 1.))
    keep, dropped = [], []
    tol = float(cfg.feasibility_tolerance)
    for i, label in enumerate(labels):
        if not str(label).startswith('acceleration_'):
            keep.append(i)
            continue
        axis = int(str(label).rsplit('_', 1)[1])
        if np.linalg.norm(basis[axis]) <= 1e-14:
            keep.append(i)
            continue
        reach_lo, reach_hi = _affine_box_range(offset, basis, axis, bounds)
        box_lo, box_hi = float(hard_lo[i]), float(hard_hi[i])
        if reach_hi < box_lo - tol or reach_lo > box_hi + tol:
            dropped.append(dict(label=label, axis=axis, reachable_lo=reach_lo,
                reachable_hi=reach_hi, slew_lo=box_lo, slew_hi=box_hi))
            continue
        keep.append(i)
    if not dropped:
        return hard_a, hard_lo, hard_hi, labels, ()
    idx = np.asarray(keep, dtype=int)
    return hard_a[idx], hard_lo[idx], hard_hi[idx], [labels[i] for i in keep], tuple(dropped)


def solve_weighted_outer(solver, data, *, deadline_s=None, online=False):
    # Local import avoids a cycle with ContactQp's policy dispatch.
    from .qp import QpResult, _linear_feasible, _normalize_rows, _violation

    start = time.perf_counter()
    cfg, mech = solver.config, data.mechanical
    # Never slew on a window shorter than the command hold. Callers used to
    # pass rocking publication gaps (~0.6 ms) and empty the first seek QP.
    slew_dt = max(float(data.acceleration_dt_s), float(data.dt_s))
    diagnostics = dict(allocation_policy='delay_kf_cop_v1',
        quality_policy_version=cfg.quality_policy_version, solver_policy='bounded_retry_v1',
        solver_deadline_s=deadline_s, energy_enabled=False,
        cop_pairing='coupled_normal_task_v1', acoustic_derivative_certified=False,
        command_slew_dt_s=slew_dt, command_hold_s=data.dt_s)

    def failure(status, reason):
        from .numeric_record import encode, SCHEMA
        diagnostics.update(reason=reason, total_time_s=time.perf_counter()-start,
            numeric_attempts=solver._numeric_attempts,
            solver_backend_version=solver._numeric_backend_version,
            replay_input=dict(schema=SCHEMA, payload=encode(dict(qp_input=asdict(data),
                qp_config=asdict(cfg), numeric_problem=diagnostics.get('numeric_problem')))))
        if status == ContactStatus.DEFERRED:
            diagnostics.update(retryable=True, qp_input=asdict(data), qp_config=asdict(cfg))
        return QpResult(None, 0., np.zeros(2), TwistConstraints(valid_until_s=data.now_s),
                        status, diagnostics, created_time_s=data.now_s)

    if solver._deadline_expired(deadline_s):
        return failure(ContactStatus.DEFERRED, 'solver_deadline_exceeded')
    if data.energy is not None:
        return failure(ContactStatus.CERTIFICATE_INVALID, 'energy_not_supported_by_delay_kf_cop_v1')
    if (data.mechanical_normal_m_s is None or data.mechanical_omega_rad_s is None
            or data.path_feedforward_contact is None):
        return failure(ContactStatus.CERTIFICATE_INVALID, 'missing_affine_mechanical_inputs')
    if (data.dt_s > cfg.max_step_s or (len(mech.A) and
            (not math.isfinite(mech.valid_until_s) or data.now_s >= mech.valid_until_s))):
        return failure(ContactStatus.CERTIFICATE_INVALID, 'expired_or_unbounded_mechanical_certificate_or_step')

    offset, basis = affine_contact_motion(data.geometry, data.path_feedback_contact,
                                         data.path_feedforward_contact)
    transform = twist_tcp_to_face(data.geometry)
    zero_path = np.linalg.norm(basis[:, 2]) <= 1e-14
    alpha_des = 1. if data.alpha_preferred is None else float(data.alpha_preferred)
    nominal = np.array([data.mechanical_normal_m_s, data.mechanical_omega_rad_s, alpha_des, 0.])
    scale = np.array([cfg.normal_scale_m_s, cfg.angular_scale_rad_s, 1., cfg.angular_scale_rad_s])
    scaled_basis = np.column_stack((basis * scale[:3], np.zeros(6)))
    target = nominal / scale
    nominal_tcp = offset + basis @ nominal[:3]

    # All hard Cartesian rows are expressed at the original TCP; the angle
    # bound uses the contact-y angular row so offset/rotation cannot mislabel it.
    rows = [*np.eye(6), *np.eye(6), transform[4]]
    lower = [*(-cfg.max_velocity), *(data.previous_twist-cfg.max_acceleration*slew_dt),
             (-cfg.angle_limit_rad-data.measured_angle)/data.dt_s]
    upper = [*cfg.max_velocity, *(data.previous_twist+cfg.max_acceleration*slew_dt),
             (cfg.angle_limit_rad-data.measured_angle)/data.dt_s]
    labels = [*(f'velocity_{i}' for i in range(6)), *(f'acceleration_{i}' for i in range(6)),
              'angle_limit_contact_y']
    if len(mech.A):
        rows.extend(mech.A); lower.extend(mech.lower); upper.extend(mech.upper)
        labels.extend(mech.labels or tuple(f'mechanical_{i}' for i in range(len(mech.A))))
    hard_a, hard_lo, hard_hi = np.asarray(rows), np.asarray(lower), np.asarray(upper)
    if online:
        hard_a, hard_lo, hard_hi, labels, dropped = drop_unreachable_acceleration_rows(
            hard_a, hard_lo, hard_hi, labels, offset, basis, cfg)
        if dropped:
            diagnostics['relaxed_acceleration_rows'] = dropped
    projected=hard_a @ scaled_basis
    shifted_lo,shifted_hi=hard_lo-hard_a@offset,hard_hi-hard_a@offset
    fixed=~np.any(projected!=0.,axis=1)
    impossible=fixed & ((shifted_lo>cfg.solver_tolerance) | (shifted_hi < -cfg.solver_tolerance))
    if np.any(impossible):
        rows_info=[dict(label=labels[i],value=float(hard_a[i]@offset),
            lower=float(hard_lo[i]),upper=float(hard_hi[i])) for i in np.flatnonzero(impossible)]
        diagnostics['infeasible_fixed_rows']=rows_info
        allow=float(cfg.fixed_acceleration_overshoot)
        accel_only=all(str(row['label']).startswith('acceleration_') for row in rows_info)
        small=True
        for row in rows_info:
            half=0.5*(row['upper']-row['lower'])
            over=max(row['lower']-row['value'], row['value']-row['upper'])
            if half<=0 or over/half>allow:
                small=False
                break
        if online and accel_only and small and allow>0:
            return failure(ContactStatus.DEFERRED,'fixed_affine_acceleration_retry')
        return failure(ContactStatus.MECHANICAL_INFEASIBLE,'fixed_affine_component_outside_mechanical_interval')
    # Exact zero rows carry no decision variable. Check them in SI units above;
    # normalizing them by 1e-12 created billion-scale residuals and futile retries.
    diagnostics['fixed_rows_checked']=int(np.count_nonzero(fixed))
    constraints, lo, hi = _normalize_rows(projected[~fixed],shifted_lo[~fixed],shifted_hi[~fixed])
    constraints = np.concatenate((constraints, np.eye(4)[2:]))
    lo, hi = np.r_[lo, 0., 0.], np.r_[hi, 1., math.inf]

    request = 0. if data.visual_request_rad_s is None else float(data.visual_request_rad_s)
    image_valid = bool(data.visual_task_valid)
    visual_active = bool(cfg.enable_visual and image_valid and data.repair_execution_enabled
                         and data.visual_gamma > 0. and abs(request) > 0.)
    request_sign = float(np.sign(request)) if visual_active else 0.
    requested = float(data.visual_gamma*abs(request)) if visual_active else 0.
    if visual_active:
        # Dividing by omega_s leaves s*Omega/omega_s + sigma/omega_s >= q/omega_s.
        constraints = np.concatenate((constraints, [[0., request_sign, 0., 1.]]))
        lo, hi = np.r_[lo, requested/scale[1]], np.r_[hi, math.inf]

    cop_limit = min(cfg.cop_max_m, data.geometry.half_length_m)
    cop_valid = bool(data.cop_m is not None and data.force_n >= cfg.cop_min_force_n
                     and abs(data.cop_m) <= cop_limit)
    cop = float(data.cop_m) if cop_valid else 0.
    weights = np.array([0., cfg.angular_weight, cfg.progress_weight, cfg.slack_weight])
    # The factor two converts the published squared objective to 1/2 x'Hx+g'x.
    hessian = 2.*np.diag(weights)
    gradient = -2.*weights*target
    r_row = np.array([1., -cop*scale[1]/scale[0], 0., 0.]) if cop_valid else np.array([1., 0., 0., 0.])
    hessian += 2.*cfg.normal_weight*np.outer(r_row, r_row)
    gradient -= 2.*cfg.normal_weight*r_row*float(r_row@target)
    diagnostics.update(mechanical_normal_m_s=nominal[0], mechanical_omega_rad_s=nominal[1],
        mechanical_nominal_twist_tcp=nominal_tcp, affine_offset_tcp=offset,
        affine_basis_tcp=basis, alpha_preferred=alpha_des, alpha_target=alpha_des,
        zero_path=zero_path, image_valid=image_valid, visual_rows_active=visual_active,
        visual_request_rad_s=request, visual_request_gated_rad_s=request_sign*requested,
        visual_gamma=data.visual_gamma, cop_valid=cop_valid, cop_m=data.cop_m,
        cop_limit_m=cop_limit, repair_force_gate=data.visual_gamma)

    transparent = _violation(constraints, lo, hi, target) <= cfg.solver_tolerance
    solver_time = 0.
    if transparent:
        x = target.copy()
        diagnostics.update(transparent=True, iterations=0, solver_status='exact_nominal_optimum')
    else:
        numeric_c,numeric_lo,numeric_hi=merge_exact_constraint_rows(constraints,lo,hi)
        diagnostics.update(original_numeric_rows=len(constraints),merged_numeric_rows=len(numeric_c))
        numeric_start = time.perf_counter()
        try:
            x, status, solved, iterations = solver._solve_numeric_bounded(
                hessian, gradient, numeric_c, numeric_lo, numeric_hi, energy_active=False, deadline_s=deadline_s)
        except (ValueError, RuntimeError) as exc:
            return failure(ContactStatus.DEFERRED if online else ContactStatus.SOLVER_FAILED, str(exc))
        solver_time = time.perf_counter()-numeric_start
        diagnostics.update(transparent=False, solver_status=status, iterations=iterations,
            numeric_attempts=solver._numeric_attempts, solver_backend_version=solver._numeric_backend_version)
        if not solved or not np.isfinite(x).all() or _violation(constraints, lo, hi, x) > cfg.feasibility_tolerance:
            diagnostics['numeric_problem'] = dict(H=hessian, g=gradient, C=constraints, l=lo, u=hi)
            diagnostics['solver_numeric_problem'] = dict(H=hessian,g=gradient,C=numeric_c,l=numeric_lo,u=numeric_hi)
            if not online and not solver._deadline_expired(deadline_s):
                try:
                    if not _linear_feasible(hard_a, hard_lo, hard_hi):
                        return failure(ContactStatus.MECHANICAL_INFEASIBLE, 'mechanical_rows_infeasible')
                    if not _linear_feasible(constraints, lo, hi):
                        return failure(ContactStatus.TASK_INFEASIBLE, 'affine_motion_subspace_conflict')
                except RuntimeError as exc:
                    return failure(ContactStatus.SOLVER_FAILED, str(exc))
            return failure(ContactStatus.DEFERRED, solver._numeric_deferred_reason or 'solver_attempts_exhausted')

    solution = x*scale
    solution[2] = np.clip(solution[2], 0., 1.)
    # sigma's analytic minimizer avoids persisting numerical noise when no
    # visual row is present. It never modifies the commanded motion.
    solution[3] = max(0., requested-request_sign*solution[1]) if visual_active else 0.
    twist = offset+basis@solution[:3]
    residual = _violation(hard_a, hard_lo, hard_hi, twist)
    if residual > cfg.feasibility_tolerance:
        return failure(ContactStatus.SOLVER_FAILED, 'final_hard_constraint_residual')

    # Admit the affine subspace, not a linear subspace through zero. The final
    # inner velocity may slow progress but cannot exceed this proposal's alpha.
    exported_a, exported_lo, exported_hi = hard_a.copy(), hard_lo.copy(), hard_hi.copy()
    if not zero_path:
        progress_row = np.linalg.pinv(basis)[2]
        shift = float(progress_row@offset)
        exported_a = np.concatenate((exported_a, progress_row[None, :]))
        exported_lo = np.r_[exported_lo, shift]
        exported_hi = np.r_[exported_hi, shift+solution[2]]
        labels.append('progress_range')
    projector = np.eye(6)-basis@np.linalg.pinv(basis)
    exported_a = np.concatenate((exported_a, projector))
    exported_lo = np.r_[exported_lo, projector@offset-cfg.subspace_tolerance]
    exported_hi = np.r_[exported_hi, projector@offset+cfg.subspace_tolerance]
    labels.extend(f'affine_motion_subspace_{i}' for i in range(6))
    expiry = data.now_s+cfg.certificate_horizon_s
    if len(mech.A):
        expiry = min(expiry, mech.valid_until_s)
    exported = TwistConstraints(exported_a, exported_lo, exported_hi, 'tcp_tool', expiry,
                                mech.sequence, mech.stop_epoch, tuple(labels))
    if exported.violation(twist) > cfg.feasibility_tolerance:
        return failure(ContactStatus.SOLVER_FAILED, 'final_affine_certificate_residual')
    delta = solution[:2]-nominal[:2]
    cop_residual = float(delta[0]-cop*delta[1])
    active = np.minimum(abs(hard_a@twist-hard_lo), abs(hard_a@twist-hard_hi)) <= cfg.feasibility_tolerance
    diagnostics.update(solution=solution, U_m_s=solution[0], Omega_rad_s=solution[1],
        sigma_I_rad_s=solution[3], qp_delta_omega_y_rad_s=delta[1],
        cop_increment_residual_m_s=cop_residual if cop_valid else None,
        visual_shortfall_rad_s=solution[3],
        visual_residual_rad_s=request_sign*solution[1]+solution[3]-requested if visual_active else 0.,
        objective_normal=float(cfg.normal_weight*(r_row@(solution/scale-target))**2),
        objective_angular=float(cfg.angular_weight*(delta[1]/scale[1])**2),
        objective_visual=float(cfg.slack_weight*(solution[3]/scale[3])**2),
        objective_progress=float(cfg.progress_weight*(solution[2]-alpha_des)**2),
        active_hard_rows=tuple(label for label, on in zip(labels[:len(hard_a)], active) if on),
        max_hard_violation=residual, solver_time_s=solver_time, total_time_s=time.perf_counter()-start)
    if solver._deadline_expired(deadline_s):
        return failure(ContactStatus.DEFERRED, 'solver_deadline_exceeded')
    status = ContactStatus.IMAGE_UNAVAILABLE if not image_valid else (
        ContactStatus.NOMINAL if transparent else ContactStatus.REPAIR)
    # The legacy two-entry slot remains serializable; explicit rad/s slack is
    # versioned in diagnostics instead of pretending it is a window m/s slack.
    return QpResult(twist, float(solution[2]), np.zeros(2), exported, status, diagnostics,
                    created_time_s=data.now_s)
