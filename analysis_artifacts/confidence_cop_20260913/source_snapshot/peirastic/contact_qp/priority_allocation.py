"""Bounded lexicographic allocation over the shared hard contact rows.

Each scalar optimum becomes an equality before the next objective is added.
The rank-one PSD objectives deliberately have no regularizer or fusion weight:
neither loading nor a CoP request can purchase progress or angular error.
"""
from __future__ import annotations

import time

import numpy as np


def solve_priority(qp, data, scaled_basis, scales, cn, ln, un, *,
                   alpha_preferred, zero_path, deadline_s, diagnostics):
    # Import at call time because qp owns the public interfaces and hard rows.
    from .qp import _energy_rows, _violation

    preferred = 0. if zero_path else float(alpha_preferred)
    constraints = np.concatenate((cn, np.array([[0., 0., 1.]])))
    lo, hi = np.r_[ln, 0.], np.r_[un, preferred]
    count = 3
    if data.energy is not None:
        ea, el, eu, auxiliary = _energy_rows(data.energy, scaled_basis, count)
        constraints = np.pad(constraints, ((0, 0), (0, auxiliary)))
        constraints = np.concatenate((constraints, ea))
        lo, hi = np.r_[lo, el], np.r_[hi, eu]
        count += auxiliary
        diagnostics['energy_auxiliary_variables'] = auxiliary
    arm = float(data.cop_m) if qp.config.allocation_policy == 'confidence_cop_v1' else 0.
    objectives = (
        ('progress', np.array([0., 0., 1.]), preferred, 2),
        ('visual_omega', np.array([0., 1., 0.]), data.visual_omega_target_rad_s / scales[1], 1),
        ('normal_pairing', np.array([1., -arm * scales[1] / scales[0], 0.]),
         data.loading_velocity_m_s / scales[0], None),
    )
    stages, attempts, iterations = [], [], 0
    diagnostics.update(priority_order=('progress', 'visual_omega', 'normal_pairing'),
        cop_m=arm, visual_omega_target_rad_s=data.visual_omega_target_rad_s,
        loading_velocity_m_s=data.loading_velocity_m_s, priority_stages=stages,
        transparent=False, visual_rows_active=False)
    x = np.full(count, np.nan)
    fixed = np.zeros(count)
    free = list(range(count))
    for name, objective, target, lock_axis in objectives:
        started = time.perf_counter()
        objective = np.pad(objective, (0, count-3))
        # Substitute established optima rather than encoding duplicate active
        # inequalities. This keeps the later PSD problems well conditioned and
        # locks earlier decisions algebraically, with no objective trade-off.
        reduced_objective = objective[free]
        hessian = np.outer(reduced_objective, reduced_objective)
        gradient = (float(objective @ fixed)-float(target)) * reduced_objective
        reduced_c = constraints[:, free]
        offset = constraints @ fixed
        reduced_lo, reduced_hi = lo-offset, hi-offset
        qp._numeric_attempts = []
        constant = np.all(reduced_c == 0., axis=1)
        if (np.any(reduced_lo[constant] > qp.config.feasibility_tolerance)
                or np.any(reduced_hi[constant] < -qp.config.feasibility_tolerance)):
            qp._numeric_deferred_reason = 'priority_lock_infeasible'
            diagnostics['failed_priority_stage'] = name
            return None
        reduced_c = reduced_c[~constant]
        reduced_lo, reduced_hi = reduced_lo[~constant], reduced_hi[~constant]
        qp._numeric_attempts = []
        qp._numeric_deferred_reason = None
        reduced_x, status, solved, used_iterations = qp._solve_numeric_bounded(
            hessian, gradient, reduced_c, reduced_lo, reduced_hi,
            energy_active=data.energy is not None, deadline_s=deadline_s)
        x = fixed.copy()
        x[free] = reduced_x
        iterations += used_iterations
        attempts.extend(dict(attempt, priority_stage=name) for attempt in qp._numeric_attempts)
        qp._numeric_attempts = attempts.copy()
        accepted = (solved and np.isfinite(x).all()
                    and _violation(constraints, lo, hi, x) <= qp.config.feasibility_tolerance)
        stages.append(dict(stage=name, duration_s=time.perf_counter()-started,
                           solver_status=status, accepted=bool(accepted),
                           target=float(target), achieved=float(objective @ x) if accepted else None))
        diagnostics.update(iterations=iterations, numeric_attempts=attempts,
                           solver_backend_version=qp._numeric_backend_version,
                           solver_status=status)
        if not accepted:
            diagnostics['numeric_problem'] = dict(H=hessian, g=gradient, C=reduced_c,
                                                  l=reduced_lo, u=reduced_hi)
            diagnostics['failed_priority_stage'] = name
            return None
        if lock_axis is not None:
            value = float(x[lock_axis])
            if lock_axis == 2:
                value = float(np.clip(value, 0., preferred))
            fixed[lock_axis] = value
            free.remove(lock_axis)
    alpha = float(np.clip(x[2], 0., preferred))
    omega = float(x[1] * scales[1])
    normal = float(x[0] * scales[0])
    paired_target = data.loading_velocity_m_s + arm * omega
    reasons = []
    if zero_path:
        reasons.append('zero_path_alpha_zero')
    elif alpha < preferred - qp.config.feasibility_tolerance:
        reasons.append('alpha_reduced_by_hard_constraints')
    if abs(omega-data.visual_omega_target_rad_s) > qp.config.feasibility_tolerance:
        reasons.append('visual_omega_limited_by_hard_constraints_at_fixed_alpha')
    if abs(normal-paired_target) > qp.config.feasibility_tolerance:
        reasons.append('normal_pairing_limited_by_hard_constraints_at_fixed_alpha_omega')
    diagnostics.update(alpha_achieved=alpha, visual_omega_achieved_rad_s=omega,
        visual_omega_residual_rad_s=omega-data.visual_omega_target_rad_s,
        normal_pairing_target_m_s=paired_target, normal_achieved_m_s=normal,
        pairing_residual_m_s=normal-paired_target,
        allocation_reason_codes=tuple(reasons or ['all_priority_targets_achieved']))
    return x
