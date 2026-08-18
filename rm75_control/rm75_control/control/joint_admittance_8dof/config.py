"""YAML loader for the 8-DOF slack-QP inner loop (Escande WBC + rail extension)."""

from __future__ import annotations

import math

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.ik_types import SrDampingConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    CartesianTrackGains,
    JointIkConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import ArmAngleTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
    ManipulabilityTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.tasks.ird_adapter import IrdConfig
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import PsiRetargetConfig


def _mapping(value, *, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_unknown(section: dict, allowed: set[str], *, name: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {name} configuration keys: " + ", ".join(unknown))


def _finite_float(value, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return out


def _arr(value, default) -> np.ndarray:
    return np.asarray(value if value is not None else default, dtype=float)


def _finite_array(value, *, name: str, ndim: int | None = None) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {out.ndim}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _resolve_rail_mode(r: dict) -> tuple[RailMode, LockedStyle]:
    mode_str = str(r.get("mode", "coupled")).lower()
    raw_style = r.get("locked_style", "hold")
    if mode_str == "coupled":
        return RailMode.COUPLED, LockedStyle.HOLD
    if mode_str == "locked":
        style = LockedStyle(str(raw_style).lower()) if raw_style else LockedStyle.HOLD
        return RailMode.LOCKED, style
    raise ValueError(f"unknown rail.mode: {r.get('mode')!r}")


_RETIRED_QPIK = {
    "protected_task",
    "scalable_tasks",
    "task_profile",
    "compatibility",
    "reference_governor",
    "accepted_reference_governor",
    "governor",
    "psi_lift",
}
_RETIRED_SOLVER = {
    "max_rows",
    "max_constraint_rows",
    "max_p0_rows",
    "max_scalable_groups",
    "max_groups",
    "protected_tolerance",
    "regularization",
    "previous_velocity_weight",
    "scalable_weight",
    "posture_weight",
    "posture_regularization",
    "margin_weight",
    "margin_weight_gain",
    "psi_weight",
    "psi_k",
    "psi_lift_weight_scale",
    "psi_err_boost_rad",
    "psi_err_weight_scale",
    "comfort_k_g",
    "comfort_qdot_max",
    "row_scale_floor",
    "qp1",
    "qp2",
    "qp3",
    "retry",
    "regularization_retry",
    "fallback_qp",
    "p0_fallback",
    "health_to_alpha",
    "sigma_escape_enter",
    "sigma_escape_exit",
    "rail_escape_v_min_m_s",
    "rail_escape_v_max_m_s",
}
_LEFTOVER_28VAR = {
    "dexterity",
    "working_set",
    "whole_body",
    "health",
    "indices",
    "task_velocity_scales",
}


def _reject_retired_qpik(qpik: dict) -> None:
    solver = _mapping(qpik.get("solver"), name="qpik.solver")
    retired = sorted((set(qpik) & _RETIRED_QPIK) | (set(solver) & _RETIRED_SOLVER))
    if retired:
        raise ValueError(
            "retired multi-level QPIK configuration keys: " + ", ".join(retired)
        )
    leftover = sorted(set(qpik) & _LEFTOVER_28VAR)
    if leftover:
        raise ValueError(
            "retired 28-var QPIK keys (use inner.qp / inner.nullspace / "
            "inner.rail_extension): " + ", ".join(leftover)
        )
    if "solver" in qpik:
        raise ValueError("qpik.solver is retired; use inner.qp")


def _parse_collision(raw: dict, *, name: str) -> CollisionConfig:
    section = _mapping(raw, name=name)
    _reject_unknown(
        section,
        {"enabled", "d_safe", "d_activate", "gamma", "max_pairs"},
        name=name,
    )
    collision = CollisionConfig(
        enabled=bool(section.get("enabled", True)),
        d_safe=_finite_float(section.get("d_safe", 0.01), name=f"{name}.d_safe"),
        d_activate=_finite_float(
            section.get("d_activate", 0.04), name=f"{name}.d_activate"
        ),
        gamma=_finite_float(section.get("gamma", 5.0), name=f"{name}.gamma"),
        max_pairs=int(section.get("max_pairs", 8)),
    )
    if not 0.0 <= collision.d_safe < collision.d_activate:
        raise ValueError("collision distances must satisfy 0 <= d_safe < d_activate")
    if collision.gamma <= 0.0 or collision.max_pairs <= 0:
        raise ValueError("collision gamma/max_pairs must be positive")
    return collision


def _parse_qp(inner: dict, collision: CollisionConfig, euler_order: str) -> QpConfig:
    c = _mapping(inner.get("qp"), name="inner.qp")
    _reject_unknown(
        c,
        {
            "task_weight", "reg", "backend", "eps_abs", "max_iter", "max_iter_cap",
            "max_solve_ms", "fail_qdot_decay", "twist_sigma_floor", "warn_on_fail",
            "sr_damping", "task_weight_min_frac", "task_weight_lpf_tau_s",
            "aniso_task_damping",
            "use_mass_weighted_reg", "mass_reg_floor", "mass_weight_exempt_rail",
            "mass_reg_lpf_tau_s", "use_dyn_nullspace",
            "limit_damper_band_rad", "limit_damper_band_rail_m",
            "limit_damper_rail_reaction_s",
            "sigma_setbased", "branch_barrier", "joint_comfort",
            "smoothness_weight", "near_arm_margin_rad",
            "j_max_arm_rad_s3", "j_max_rail_m_s3",
            "use_cpp_kernel", "nullspace_vel_damp",
        },
        name="inner.qp",
    )
    backend = str(c.get("backend", "proxqp")).lower()
    if backend not in {"proxqp", "osqp", "scipy"}:
        raise ValueError(
            "inner.qp.backend must be 'proxqp', 'osqp', or 'scipy' "
            f"(got {backend!r})"
        )
    if backend == "scipy":
        # Slack QP has no scipy path; ProxQP falls back to OSQP at runtime.
        backend = "proxqp"
    sr = _mapping(c.get("sr_damping"), name="inner.qp.sr_damping")
    _reject_unknown(
        sr, {"lam0", "sigma_ref", "sigma_floor"}, name="inner.qp.sr_damping"
    )
    from rm75_control.control.joint_admittance_8dof.solver.branch_barrier import (
        BranchBarrierConfig,
    )
    from rm75_control.control.joint_admittance_8dof.solver.joint_comfort import (
        JointComfortConfig,
    )
    from rm75_control.control.joint_admittance_8dof.solver.sigma_setbased import (
        SigmaSetBasedConfig,
    )

    ss = _mapping(c.get("sigma_setbased"), name="inner.qp.sigma_setbased")
    _reject_unknown(
        ss,
        {
            "enabled", "activate", "safe", "exit", "gamma", "slack_weight",
            "grad_eps", "grad_period_ticks",
        },
        name="inner.qp.sigma_setbased",
    )
    bb = _mapping(c.get("branch_barrier"), name="inner.qp.branch_barrier")
    _reject_unknown(
        bb,
        {
            "enabled", "activate_rad", "box_activate_rad", "eps_rad", "gamma",
            "slack_weight", "target_eps_rad", "dwell_free_s", "dwell_ramp_s",
            "dwell_scale_max", "j4_limit_eps_rad", "j4_limit_activate_rad",
            "j1_overfold_abs_rad", "j1_overfold_activate_rad", "j1_overfold_eps_rad",
        },
        name="inner.qp.branch_barrier",
    )
    jc = _mapping(c.get("joint_comfort"), name="inner.qp.joint_comfort")
    _reject_unknown(
        jc,
        {
            "enabled", "m_comfort_deg", "activate_deg", "gamma", "slack_weight",
        },
        name="inner.qp.joint_comfort",
    )
    smooth_raw = c.get("smoothness_weight", 0.15)
    if isinstance(smooth_raw, (list, tuple, np.ndarray)):
        smoothness_weight = _finite_array(
            smooth_raw,
            name="inner.qp.smoothness_weight",
            ndim=1,
        )
        if smoothness_weight.size != 8:
            raise ValueError(
                "inner.qp.smoothness_weight must be scalar or length 8"
            )
    else:
        smoothness_weight = _finite_float(
            smooth_raw, name="inner.qp.smoothness_weight"
        )
    return QpConfig(
        task_weight=_arr(c.get("task_weight"), [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
        reg=_arr(
            c.get("reg"),
            [1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 5.0e-3, 5.0e-3, 5.0e-3],
        ),
        backend=backend,
        use_cpp_kernel=bool(c.get("use_cpp_kernel", True)),
        eps_abs=_finite_float(c.get("eps_abs", 1.0e-6), name="inner.qp.eps_abs"),
        max_iter=int(c.get("max_iter", 400)),
        max_iter_cap=int(c.get("max_iter_cap", 400)),
        euler_order=euler_order,
        collision=collision,
        sr_damping=SrDampingConfig(
            lam0=_finite_float(sr.get("lam0", 0.05), name="sr_damping.lam0"),
            sigma_ref=_finite_float(
                sr.get("sigma_ref", 0.08), name="sr_damping.sigma_ref"
            ),
            sigma_floor=_finite_float(
                sr.get("sigma_floor", 1e-6), name="sr_damping.sigma_floor"
            ),
        ),
        task_weight_min_frac=_finite_float(
            c.get("task_weight_min_frac", 0.05), name="inner.qp.task_weight_min_frac"
        ),
        task_weight_lpf_tau_s=_finite_float(
            c.get("task_weight_lpf_tau_s", 0.25),
            name="inner.qp.task_weight_lpf_tau_s",
        ),
        aniso_task_damping=bool(c.get("aniso_task_damping", True)),
        use_mass_weighted_reg=bool(c.get("use_mass_weighted_reg", True)),
        mass_reg_floor=_finite_float(
            c.get("mass_reg_floor", 0.05), name="inner.qp.mass_reg_floor"
        ),
        mass_weight_exempt_rail=bool(c.get("mass_weight_exempt_rail", True)),
        mass_reg_lpf_tau_s=_finite_float(
            c.get("mass_reg_lpf_tau_s", 0.2), name="inner.qp.mass_reg_lpf_tau_s"
        ),
        use_dyn_nullspace=bool(c.get("use_dyn_nullspace", False)),
        limit_damper_band_rad=_finite_float(
            c.get("limit_damper_band_rad", 0.15),
            name="inner.qp.limit_damper_band_rad",
        ),
        limit_damper_band_rail_m=_finite_float(
            c.get("limit_damper_band_rail_m", 0.01),
            name="inner.qp.limit_damper_band_rail_m",
        ),
        limit_damper_rail_reaction_s=_finite_float(
            c.get("limit_damper_rail_reaction_s", 0.15),
            name="inner.qp.limit_damper_rail_reaction_s",
        ),
        warn_on_fail=bool(c.get("warn_on_fail", False)),
        fail_qdot_decay=_finite_float(
            c.get("fail_qdot_decay", 0.85), name="inner.qp.fail_qdot_decay"
        ),
        max_solve_ms=_finite_float(
            c.get("max_solve_ms", 5.0), name="inner.qp.max_solve_ms"
        ),
        twist_sigma_floor=_finite_float(
            c.get("twist_sigma_floor", 0.02), name="inner.qp.twist_sigma_floor"
        ),
        sigma_setbased=SigmaSetBasedConfig(
            enabled=bool(ss.get("enabled", True)),
            activate=_finite_float(
                ss.get("activate", 0.14), name="sigma_setbased.activate"
            ),
            safe=_finite_float(ss.get("safe", 0.06), name="sigma_setbased.safe"),
            exit=_finite_float(ss.get("exit", 0.18), name="sigma_setbased.exit"),
            gamma=_finite_float(ss.get("gamma", 8.0), name="sigma_setbased.gamma"),
            slack_weight=_finite_float(
                ss.get("slack_weight", 200.0), name="sigma_setbased.slack_weight"
            ),
            grad_eps=_finite_float(
                ss.get("grad_eps", 1.0e-4), name="sigma_setbased.grad_eps"
            ),
            grad_period_ticks=max(1, int(ss.get("grad_period_ticks", 10))),
        ),
        branch_barrier=BranchBarrierConfig(
            enabled=bool(bb.get("enabled", True)),
            activate_rad=_finite_float(
                bb.get("activate_rad", 0.52), name="branch_barrier.activate_rad"
            ),
            box_activate_rad=_finite_float(
                bb.get("box_activate_rad", 0.87),
                name="branch_barrier.box_activate_rad",
            ),
            eps_rad=_finite_float(
                bb.get("eps_rad", 0.35), name="branch_barrier.eps_rad"
            ),
            j4_limit_eps_rad=_finite_float(
                bb.get("j4_limit_eps_rad", 5.0 * math.pi / 180.0),
                name="branch_barrier.j4_limit_eps_rad",
            ),
            j4_limit_activate_rad=_finite_float(
                bb.get("j4_limit_activate_rad", 25.0 * math.pi / 180.0),
                name="branch_barrier.j4_limit_activate_rad",
            ),
            j1_overfold_abs_rad=_finite_float(
                bb.get("j1_overfold_abs_rad", 140.0 * math.pi / 180.0),
                name="branch_barrier.j1_overfold_abs_rad",
            ),
            j1_overfold_activate_rad=_finite_float(
                bb.get("j1_overfold_activate_rad", 25.0 * math.pi / 180.0),
                name="branch_barrier.j1_overfold_activate_rad",
            ),
            j1_overfold_eps_rad=_finite_float(
                bb.get("j1_overfold_eps_rad", 0.0),
                name="branch_barrier.j1_overfold_eps_rad",
            ),
            gamma=_finite_float(bb.get("gamma", 6.0), name="branch_barrier.gamma"),
            slack_weight=_finite_float(
                bb.get("slack_weight", 80.0), name="branch_barrier.slack_weight"
            ),
            target_eps_rad=_finite_float(
                bb.get("target_eps_rad", 1.0e-3),
                name="branch_barrier.target_eps_rad",
            ),
            dwell_free_s=_finite_float(
                bb.get("dwell_free_s", 0.3), name="branch_barrier.dwell_free_s"
            ),
            dwell_ramp_s=_finite_float(
                bb.get("dwell_ramp_s", 1.0), name="branch_barrier.dwell_ramp_s"
            ),
            dwell_scale_max=_finite_float(
                bb.get("dwell_scale_max", 5.0),
                name="branch_barrier.dwell_scale_max",
            ),
        ),
        joint_comfort=JointComfortConfig(
            enabled=bool(jc.get("enabled", True)),
            m_comfort_rad=math.radians(
                _finite_float(
                    jc.get("m_comfort_deg", 15.0),
                    name="joint_comfort.m_comfort_deg",
                )
            ),
            activate_rad=math.radians(
                _finite_float(
                    jc.get("activate_deg", 25.0),
                    name="joint_comfort.activate_deg",
                )
            ),
            gamma=_finite_float(jc.get("gamma", 6.0), name="joint_comfort.gamma"),
            slack_weight=_finite_float(
                jc.get("slack_weight", 80.0), name="joint_comfort.slack_weight"
            ),
        ),
        near_arm_margin_rad=_finite_float(
            c.get("near_arm_margin_rad", 0.08),
            name="inner.qp.near_arm_margin_rad",
        ),
        smoothness_weight=smoothness_weight,
        j_max_arm_rad_s3=_finite_float(
            c.get("j_max_arm_rad_s3", 300.0), name="inner.qp.j_max_arm_rad_s3"
        ),
        j_max_rail_m_s3=_finite_float(
            c.get("j_max_rail_m_s3", 3.0), name="inner.qp.j_max_rail_m_s3"
        ),
        nullspace_vel_damp=_finite_float(
            c.get("nullspace_vel_damp", 0.0),
            name="inner.qp.nullspace_vel_damp",
        ),
    )


def _parse_nullspace(inner: dict) -> tuple[NullspaceTaskConfig, ManipulabilityTaskConfig]:
    n = _mapping(inner.get("nullspace"), name="inner.nullspace")
    _reject_unknown(
        n,
        {
            "k_center", "k_limit", "activation", "weights", "q_nominal_deg",
            "manipulability", "engage_s",
        },
        name="inner.nullspace",
    )
    q_nominal_deg = n.get("q_nominal_deg")
    m = _mapping(n.get("manipulability"), name="inner.nullspace.manipulability")
    _reject_unknown(
        m, {"k_mu", "eps_rad", "sigma_fade_ref", "grad_period_ticks"},
        name="inner.nullspace.manipulability",
    )
    nullspace = NullspaceTaskConfig(
        k_center=_finite_float(n.get("k_center", 1.0), name="nullspace.k_center"),
        k_limit=_finite_float(n.get("k_limit", 2.0), name="nullspace.k_limit"),
        activation=_finite_float(
            n.get("activation", 0.85), name="nullspace.activation"
        ),
        weights=(
            _finite_array(n["weights"], name="nullspace.weights", ndim=1)
            if n.get("weights") is not None
            else None
        ),
        q_nominal_rad=(
            np.radians(_finite_array(q_nominal_deg, name="nullspace.q_nominal_deg", ndim=1))
            if q_nominal_deg is not None
            else None
        ),
        engage_s=_finite_float(
            n.get("engage_s", 0.35), name="nullspace.engage_s"
        ),
    )
    manipulability = ManipulabilityTaskConfig(
        k_mu=_finite_float(m.get("k_mu", 0.8), name="manipulability.k_mu"),
        eps_rad=_finite_float(m.get("eps_rad", 1e-4), name="manipulability.eps_rad"),
        sigma_fade_ref=_finite_float(
            m.get("sigma_fade_ref", 0.12), name="manipulability.sigma_fade_ref"
        ),
        grad_period_ticks=max(1, int(m.get("grad_period_ticks", 10))),
    )
    return nullspace, manipulability


def _parse_arm_angle(inner: dict) -> ArmAngleTaskConfig:
    a = _mapping(inner.get("arm_angle"), name="inner.arm_angle")
    _reject_unknown(
        a,
        {
            "enabled", "k_psi", "psi_ref_deg", "fd_eps_rad", "safe_denom_eps",
            "obs_decay_gain", "obs_smooth_floor", "max_qdot_frac",
            "psi_home_deg", "max_psi_swing_deg", "engage_s",
        },
        name="inner.arm_angle",
    )
    psi_ref_deg = a.get("psi_ref_deg")
    psi_home_deg = a.get("psi_home_deg")
    return ArmAngleTaskConfig(
        enabled=bool(a.get("enabled", False)),
        k_psi=_finite_float(a.get("k_psi", 1.0), name="arm_angle.k_psi"),
        obs_smooth_floor=_finite_float(
            a.get("obs_smooth_floor", 0.3), name="arm_angle.obs_smooth_floor"
        ),
        psi_ref_rad=(
            math.radians(_finite_float(psi_ref_deg, name="arm_angle.psi_ref_deg"))
            if psi_ref_deg is not None
            else None
        ),
        psi_home_rad=(
            math.radians(_finite_float(psi_home_deg, name="arm_angle.psi_home_deg"))
            if psi_home_deg is not None
            else None
        ),
        engage_s=_finite_float(a.get("engage_s", 0.35), name="arm_angle.engage_s"),
    )


def _parse_psi_retarget(inner: dict) -> PsiRetargetConfig:
    p = _mapping(inner.get("psi_retarget"), name="inner.psi_retarget")
    _reject_unknown(
        p,
        {
            "enabled", "n_y", "n_d", "n_psi", "w_sigma", "w_wrist",
            "margin_floor_deg", "z_replan_m", "psi_rate_deg_s", "rail_margin_m",
            "wrist_min_deg", "d_center_rate_m_s", "d_band_m",
            "d_slew_psi_err_deg", "psi_cmd_lead_deg",
            "psi_replan_period_s", "psi_search_half_span_deg", "psi_search_n",
            "psi_wrist_ok_deg", "psi_envelope_deg",
            "psi_attr_deg", "d_attr_m", "psi_return_dwell_s",
            "require_design_family",
            # Retired keys accepted so older yaml still loads.
            "d_replan_m", "d_replan_period_s", "d_pref_rate_m_s",
            "evals_per_tick", "psi_step_deg", "psi_lpf_tau_s",
            "rail_step_m", "d_pref_lpf_tau_s",
        },
        name="inner.psi_retarget",
    )
    env = p.get("psi_envelope_deg", [40.0, 110.0])
    if isinstance(env, (list, tuple)) and len(env) == 2:
        env_lo, env_hi = float(env[0]), float(env[1])
    else:
        env_lo, env_hi = 40.0, 110.0
    return PsiRetargetConfig(
        enabled=bool(p.get("enabled", True)),
        n_y=int(p.get("n_y", 9)),
        n_d=int(p.get("n_d", 8)),
        n_psi=int(p.get("n_psi", 9)),
        w_sigma=_finite_float(p.get("w_sigma", 0.5), name="psi_retarget.w_sigma"),
        w_wrist=_finite_float(p.get("w_wrist", 0.5), name="psi_retarget.w_wrist"),
        margin_floor_rad=math.radians(
            _finite_float(
                p.get("margin_floor_deg", 15.0), name="psi_retarget.margin_floor_deg"
            )
        ),
        z_replan_m=_finite_float(
            p.get("z_replan_m", 0.0), name="psi_retarget.z_replan_m"
        ),
        psi_rate_rad_s=math.radians(
            _finite_float(
                p.get("psi_rate_deg_s", 25.0), name="psi_retarget.psi_rate_deg_s"
            )
        ),
        d_center_rate_m_s=_finite_float(
            p.get("d_center_rate_m_s", 0.02), name="psi_retarget.d_center_rate_m_s"
        ),
        d_band_m=_finite_float(
            p.get("d_band_m", 0.08), name="psi_retarget.d_band_m"
        ),
        d_slew_psi_err_rad=math.radians(
            _finite_float(
                p.get("d_slew_psi_err_deg", 40.0),
                name="psi_retarget.d_slew_psi_err_deg",
            )
        ),
        psi_cmd_lead_rad=math.radians(
            _finite_float(
                p.get("psi_cmd_lead_deg", 18.0),
                name="psi_retarget.psi_cmd_lead_deg",
            )
        ),
        psi_attr_rad=math.radians(
            _finite_float(p.get("psi_attr_deg", 68.0), name="psi_retarget.psi_attr_deg")
        ),
        d_attr_m=_finite_float(
            p.get("d_attr_m", -0.185), name="psi_retarget.d_attr_m"
        ),
        psi_return_dwell_s=_finite_float(
            p.get("psi_return_dwell_s", 1.0), name="psi_retarget.psi_return_dwell_s"
        ),
        require_design_family=bool(p.get("require_design_family", False)),
        psi_replan_period_s=_finite_float(
            p.get("psi_replan_period_s", 0.1),
            name="psi_retarget.psi_replan_period_s",
        ),
        psi_search_half_span_rad=math.radians(
            _finite_float(
                p.get("psi_search_half_span_deg", 45.0),
                name="psi_retarget.psi_search_half_span_deg",
            )
        ),
        psi_search_n=int(p.get("psi_search_n", 9)),
        psi_wrist_ok_rad=math.radians(
            _finite_float(
                p.get("psi_wrist_ok_deg", 40.0),
                name="psi_retarget.psi_wrist_ok_deg",
            )
        ),
        psi_envelope_lo_rad=math.radians(env_lo),
        psi_envelope_hi_rad=math.radians(env_hi),
        rail_margin_m=_finite_float(
            p.get("rail_margin_m", 0.02), name="psi_retarget.rail_margin_m"
        ),
        wrist_min_rad=math.radians(
            _finite_float(
                p.get("wrist_min_deg", 30.0), name="psi_retarget.wrist_min_deg"
            )
        ),
    )


def _parse_ird(inner: dict) -> IrdConfig:
    r = _mapping(inner.get("ird"), name="inner.ird")
    _reject_unknown(
        r,
        {
            "enabled", "checkpoint", "robot_spec", "device", "allow_stale",
            "goodness_period_ticks",
        },
        name="inner.ird",
    )
    defaults = IrdConfig()
    return IrdConfig(
        enabled=bool(r.get("enabled", False)),
        checkpoint=str(r.get("checkpoint", defaults.checkpoint)),
        robot_spec=str(r.get("robot_spec", defaults.robot_spec)),
        device=str(r.get("device", "cpu")),
        allow_stale=bool(r.get("allow_stale", True)),
        goodness_period_ticks=int(r.get("goodness_period_ticks", 10)),
    )


def _parse_rail_extension(inner: dict) -> RailExtensionConfig:
    r = _mapping(inner.get("rail_extension"), name="inner.rail_extension")
    _reject_unknown(
        r,
        {
            "enabled", "k_ext", "k_ff", "v_ff_thr_m_s", "v_ff_span_m_s",
            "e0_m", "e1_m", "w_max", "v_max_m_s", "limit_margin_m",
            "pin_margin_m", "escape_leave_m",
            "k_sigma_boost", "k_esc", "w_sigma_floor",
            "k_pose", "pose_e0_m", "pose_e1_m", "pose_w_max",
            "sigma_guard_enter", "sigma_guard_exit", "v_guard_max_m_s",
            "v_lpf_tau_s", "v_lpf_tau_escape_s",
            "sigma_escape_enter", "sigma_escape_exit",
            "margin_escape_enter", "margin_escape_exit", "sigma_drop_rate",
            "escape_enter_dwell_s",
            "k_escape_boost", "escape_grad_floor",
            "k_margin_boost", "w_ext_cap",
            "soft_min_m", "soft_max_m", "v_reach_cap_m_s",
            "v_reach_idle_cap_m_s", "d_band_m",
            "healthy_sigma_mute",
            "v_reach_total_max_m_s",
            "d_star_err0_m", "d_star_err1_m", "d_star_w_mult", "d_star_reg_mult",
            "press_v_force_min_m_s", "press_dz_max_m", "press_y_err_m",
            "press_stall_s", "d_star_nudge_m", "open_travel_min_m",
            "escape_sign_policy",
        },
        name="inner.rail_extension",
    )
    return RailExtensionConfig(
        enabled=bool(r.get("enabled", True)),
        k_ext=_finite_float(r.get("k_ext", 1.0), name="rail_extension.k_ext"),
        k_ff=_finite_float(r.get("k_ff", 1.0), name="rail_extension.k_ff"),
        v_ff_thr_m_s=_finite_float(
            r.get("v_ff_thr_m_s", 0.01), name="rail_extension.v_ff_thr_m_s"
        ),
        v_ff_span_m_s=_finite_float(
            r.get("v_ff_span_m_s", 0.03), name="rail_extension.v_ff_span_m_s"
        ),
        e0_m=_finite_float(r.get("e0_m", 0.05), name="rail_extension.e0_m"),
        e1_m=_finite_float(r.get("e1_m", 0.15), name="rail_extension.e1_m"),
        w_max=_finite_float(r.get("w_max", 1.5), name="rail_extension.w_max"),
        v_max_m_s=_finite_float(
            r.get("v_max_m_s", 0.08), name="rail_extension.v_max_m_s"
        ),
        limit_margin_m=_finite_float(
            r.get("limit_margin_m", 0.15), name="rail_extension.limit_margin_m"
        ),
        pin_margin_m=_finite_float(
            r.get("pin_margin_m", 0.008), name="rail_extension.pin_margin_m"
        ),
        escape_leave_m=_finite_float(
            r.get("escape_leave_m", 0.04), name="rail_extension.escape_leave_m"
        ),
        k_sigma_boost=_finite_float(
            r.get("k_sigma_boost", 2.0), name="rail_extension.k_sigma_boost"
        ),
        k_esc=_finite_float(r.get("k_esc", 0.5), name="rail_extension.k_esc"),
        w_sigma_floor=_finite_float(
            r.get("w_sigma_floor", 1.0), name="rail_extension.w_sigma_floor"
        ),
        k_pose=_finite_float(r.get("k_pose", 2.0), name="rail_extension.k_pose"),
        pose_e0_m=_finite_float(
            r.get("pose_e0_m", 0.005), name="rail_extension.pose_e0_m"
        ),
        pose_e1_m=_finite_float(
            r.get("pose_e1_m", 0.04), name="rail_extension.pose_e1_m"
        ),
        pose_w_max=_finite_float(
            r.get("pose_w_max", 4.0), name="rail_extension.pose_w_max"
        ),
        sigma_guard_enter=_finite_float(
            r.get("sigma_guard_enter", 0.45), name="rail_extension.sigma_guard_enter"
        ),
        sigma_guard_exit=_finite_float(
            r.get("sigma_guard_exit", 0.70), name="rail_extension.sigma_guard_exit"
        ),
        v_guard_max_m_s=_finite_float(
            r.get("v_guard_max_m_s", 0.04), name="rail_extension.v_guard_max_m_s"
        ),
        v_lpf_tau_s=_finite_float(
            r.get("v_lpf_tau_s", 0.05), name="rail_extension.v_lpf_tau_s"
        ),
        v_lpf_tau_escape_s=_finite_float(
            r.get("v_lpf_tau_escape_s", 0.04),
            name="rail_extension.v_lpf_tau_escape_s",
        ),
        sigma_escape_enter=_finite_float(
            r.get("sigma_escape_enter", 0.55),
            name="rail_extension.sigma_escape_enter",
        ),
        sigma_escape_exit=_finite_float(
            r.get("sigma_escape_exit", 0.80),
            name="rail_extension.sigma_escape_exit",
        ),
        margin_escape_enter=_finite_float(
            r.get("margin_escape_enter", 0.12),
            name="rail_extension.margin_escape_enter",
        ),
        margin_escape_exit=_finite_float(
            r.get("margin_escape_exit", 0.25),
            name="rail_extension.margin_escape_exit",
        ),
        sigma_drop_rate=_finite_float(
            r.get("sigma_drop_rate", 0.0), name="rail_extension.sigma_drop_rate"
        ),
        escape_enter_dwell_s=_finite_float(
            r.get("escape_enter_dwell_s", 0.05),
            name="rail_extension.escape_enter_dwell_s",
        ),
        k_escape_boost=_finite_float(
            r.get("k_escape_boost", 1.2), name="rail_extension.k_escape_boost"
        ),
        escape_grad_floor=_finite_float(
            r.get("escape_grad_floor", 0.0), name="rail_extension.escape_grad_floor"
        ),
        k_margin_boost=_finite_float(
            r.get("k_margin_boost", 4.0), name="rail_extension.k_margin_boost"
        ),
        w_ext_cap=_finite_float(
            r.get("w_ext_cap", 24.0), name="rail_extension.w_ext_cap"
        ),
        soft_min_m=_finite_float(
            r.get("soft_min_m", 0.025), name="rail_extension.soft_min_m"
        ),
        soft_max_m=_finite_float(
            r.get("soft_max_m", 0.78), name="rail_extension.soft_max_m"
        ),
        v_reach_cap_m_s=_finite_float(
            r.get("v_reach_cap_m_s", 0.05), name="rail_extension.v_reach_cap_m_s"
        ),
        v_reach_idle_cap_m_s=_finite_float(
            r.get("v_reach_idle_cap_m_s", 0.010),
            name="rail_extension.v_reach_idle_cap_m_s",
        ),
        healthy_sigma_mute=_finite_float(
            r.get("healthy_sigma_mute", 0.08),
            name="rail_extension.healthy_sigma_mute",
        ),
        v_reach_total_max_m_s=(
            None
            if r.get("v_reach_total_max_m_s") is None
            else _finite_float(
                r["v_reach_total_max_m_s"],
                name="rail_extension.v_reach_total_max_m_s",
            )
        ),
        d_band_m=_finite_float(
            r.get("d_band_m", 0.005), name="rail_extension.d_band_m"
        ),
        d_star_err0_m=_finite_float(
            r.get("d_star_err0_m", 0.01), name="rail_extension.d_star_err0_m"
        ),
        d_star_err1_m=_finite_float(
            r.get("d_star_err1_m", 0.04), name="rail_extension.d_star_err1_m"
        ),
        d_star_w_mult=_finite_float(
            r.get("d_star_w_mult", 6.0), name="rail_extension.d_star_w_mult"
        ),
        d_star_reg_mult=_finite_float(
            r.get("d_star_reg_mult", 20.0), name="rail_extension.d_star_reg_mult"
        ),
        press_v_force_min_m_s=_finite_float(
            r.get("press_v_force_min_m_s", 0.02),
            name="rail_extension.press_v_force_min_m_s",
        ),
        press_dz_max_m=_finite_float(
            r.get("press_dz_max_m", 0.002), name="rail_extension.press_dz_max_m"
        ),
        press_y_err_m=_finite_float(
            r.get("press_y_err_m", 0.005), name="rail_extension.press_y_err_m"
        ),
        press_stall_s=_finite_float(
            r.get("press_stall_s", 0.5), name="rail_extension.press_stall_s"
        ),
        d_star_nudge_m=_finite_float(
            r.get("d_star_nudge_m", 0.01), name="rail_extension.d_star_nudge_m"
        ),
        open_travel_min_m=_finite_float(
            r.get("open_travel_min_m", 0.01),
            name="rail_extension.open_travel_min_m",
        ),
        escape_sign_policy=str(r.get("escape_sign_policy", "minus")).strip().lower(),
    )


def _parse_rail(rail_raw: dict, hw_lw: dict) -> RailLockConfig:
    _reject_unknown(
        rail_raw,
        {
            "mode", "locked_style", "q_ref_m", "lock_gain", "lock_reg_scale",
            "lock_vel_eps_m_s", "lock_hard_pin", "v_max_m_s", "travel_m",
            "soft_min_m", "soft_max_m", "hard_min_m", "hard_max_m",
        },
        name="rail",
    )
    rail_mode, locked_style = _resolve_rail_mode(rail_raw)
    soft_min = _finite_float(
        rail_raw.get("soft_min_m", hw_lw.get("soft_min_m", 0.015)),
        name="rail.soft_min_m",
    )
    soft_max = _finite_float(
        rail_raw.get("soft_max_m", hw_lw.get("soft_max_m", 0.77)),
        name="rail.soft_max_m",
    )
    hard_min = _finite_float(
        rail_raw.get("hard_min_m", hw_lw.get("hard_min_m", 0.005)),
        name="rail.hard_min_m",
    )
    hard_max = _finite_float(
        rail_raw.get("hard_max_m", hw_lw.get("hard_max_m", 0.78)),
        name="rail.hard_max_m",
    )
    travel = _finite_float(rail_raw.get("travel_m", 0.80), name="rail.travel_m")
    if not 0.0 <= hard_min <= soft_min < soft_max <= hard_max <= travel:
        raise ValueError(
            "rail limits must satisfy "
            "0 <= hard_min <= soft_min < soft_max <= hard_max <= travel"
        )
    if rail_raw and hw_lw and ("soft_min_m" in hw_lw or "soft_max_m" in hw_lw):
        hw_min = _finite_float(hw_lw.get("soft_min_m", soft_min), name="hw rail soft_min")
        hw_max = _finite_float(hw_lw.get("soft_max_m", soft_max), name="hw rail soft_max")
        if abs(hw_min - soft_min) > 1.0e-6 or abs(hw_max - soft_max) > 1.0e-6:
            raise ValueError("rail soft-limit mismatch between QPIK and hardware")
    if rail_raw and hw_lw and ("hard_min_m" in hw_lw or "hard_max_m" in hw_lw):
        hw_hmin = _finite_float(hw_lw.get("hard_min_m", hard_min), name="hw rail hard_min")
        hw_hmax = _finite_float(hw_lw.get("hard_max_m", hard_max), name="hw rail hard_max")
        if abs(hw_hmin - hard_min) > 1.0e-6 or abs(hw_hmax - hard_max) > 1.0e-6:
            raise ValueError("rail hard-limit mismatch between QPIK and hardware")
    return RailLockConfig(
        mode=rail_mode,
        locked_style=locked_style,
        q_ref_m=(
            None
            if rail_raw.get("q_ref_m") is None
            else _finite_float(rail_raw["q_ref_m"], name="rail.q_ref_m")
        ),
        lock_gain=_finite_float(rail_raw.get("lock_gain", 200.0), name="rail.lock_gain"),
        lock_reg_scale=_finite_float(
            rail_raw.get("lock_reg_scale", 100.0), name="rail.lock_reg_scale"
        ),
        lock_vel_eps_m_s=_finite_float(
            rail_raw.get("lock_vel_eps_m_s", 0.0), name="rail.lock_vel_eps_m_s"
        ),
        lock_hard_pin=bool(rail_raw.get("lock_hard_pin", True)),
        v_max_m_s=(
            None
            if rail_raw.get("v_max_m_s") is None
            else _finite_float(rail_raw["v_max_m_s"], name="rail.v_max_m_s")
        ),
        travel_m=travel,
        soft_min_m=soft_min,
        soft_max_m=soft_max,
        hard_min_m=hard_min,
        hard_max_m=hard_max,
    )


def _parse_cartesian_track(raw: dict) -> CartesianTrackGains:
    section = _mapping(raw.get("cartesian_track"), name="cartesian_track")
    _reject_unknown(
        section,
        {"k_task_lin", "k_task_rot", "max_pos_err_m", "max_rot_err_rad"},
        name="cartesian_track",
    )
    defaults = CartesianTrackGains()
    gains = CartesianTrackGains(
        k_task_lin=_finite_float(
            section.get("k_task_lin", defaults.k_task_lin),
            name="cartesian_track.k_task_lin",
        ),
        k_task_rot=_finite_float(
            section.get("k_task_rot", defaults.k_task_rot),
            name="cartesian_track.k_task_rot",
        ),
        max_pos_err_m=_finite_float(
            section.get("max_pos_err_m", defaults.max_pos_err_m),
            name="cartesian_track.max_pos_err_m",
        ),
        max_rot_err_rad=_finite_float(
            section.get("max_rot_err_rad", defaults.max_rot_err_rad),
            name="cartesian_track.max_rot_err_rad",
        ),
    )
    if gains.k_task_lin < 0.0 or gains.k_task_rot < 0.0:
        raise ValueError("cartesian_track gains must be non-negative")
    if gains.max_pos_err_m <= 0.0 or gains.max_rot_err_rad <= 0.0:
        raise ValueError("cartesian_track error limits must be positive")
    return gains


def build_joint_ik_config(raw: dict) -> JointIkConfig:
    """Build JointIkConfig from inner.qp + qpik.hard_limits."""

    if not isinstance(raw, dict):
        raise ValueError("controller config root must be a mapping")
    timing = _mapping(raw.get("timing"), name="timing")
    inner = _mapping(raw.get("inner"), name="inner")
    qpik = _mapping(raw.get("qpik"), name="qpik")
    _reject_retired_qpik(qpik)
    _reject_unknown(qpik, {"hard_limits"}, name="qpik")

    hard = _mapping(qpik.get("hard_limits"), name="qpik.hard_limits")
    _reject_unknown(
        hard,
        {
            "v_scale", "a_max_arm_rad_s2", "a_max_rail_m_s2",
            "position_margin_deg", "position_margin_rail_mm",
            "command_lead_arm_deg", "command_lead_rail_mm",
            "velocity_damper", "collision", "rail",
        },
        name="qpik.hard_limits",
    )
    retired_inner = sorted(
        set(inner)
        & {
            "a_max_rail_escape_m_s2",
            "rail_escape_v_min_m_s",
            "rail_escape_v_max_m_s",
            "sigma_escape_enter",
            "sigma_escape_exit",
        }
    )
    if retired_inner:
        raise ValueError(
            "retired QPIK configuration keys in inner: " + ", ".join(retired_inner)
        )
    _reject_unknown(
        inner,
        {
            "control_frame", "euler_order", "sync_tcp_from_robot",
            "v_scale", "a_max_arm", "a_max_arm_rad_s2", "a_max_rail_m_s2",
            "position_margin_deg", "position_margin_rail_mm",
            "resync_err_deg", "resync_err_rail_mm",
            "qp", "collision", "nullspace", "arm_angle", "rail_extension", "rail",
            "psi_retarget", "ird",
            "nullspace_d_null", "nullspace_d_null_adaptive", "nullspace_max_qdot_frac",
            "post_qp_step_clamp",
            "qmeas_filter", "qmeas_lowpass_hz",
        },
        name="inner",
    )

    euler_order = str(
        _mapping(raw.get("frames"), name="frames").get(
            "euler_order", inner.get("euler_order", "xyz")
        )
    )
    collision = _parse_collision(
        hard.get("collision", inner.get("collision")),
        name="collision",
    )
    qp = _parse_qp(inner, collision, euler_order)
    damper = _mapping(hard.get("velocity_damper"), name="qpik.hard_limits.velocity_damper")
    if damper:
        _reject_unknown(
            damper, {"arm_band_rad", "rail_band_m", "rail_reaction_s"},
            name="qpik.hard_limits.velocity_damper",
        )
        if "arm_band_rad" in damper:
            qp.limit_damper_band_rad = _finite_float(
                damper["arm_band_rad"], name="velocity_damper.arm_band_rad"
            )
        if "rail_band_m" in damper:
            qp.limit_damper_band_rail_m = _finite_float(
                damper["rail_band_m"], name="velocity_damper.rail_band_m"
            )
        if "rail_reaction_s" in damper:
            qp.limit_damper_rail_reaction_s = _finite_float(
                damper["rail_reaction_s"], name="velocity_damper.rail_reaction_s"
            )
    if qp.limit_damper_band_rad < 0.0 or qp.limit_damper_band_rail_m < 0.0:
        raise ValueError("velocity damper bands must be non-negative")
    if qp.limit_damper_rail_reaction_s < 0.0:
        raise ValueError("velocity_damper.rail_reaction_s must be non-negative")

    nullspace, manipulability = _parse_nullspace(inner)
    arm_angle = _parse_arm_angle(inner)
    psi_retarget = _parse_psi_retarget(inner)
    ird = _parse_ird(inner)
    rail_extension = _parse_rail_extension(inner)
    cartesian_track = _parse_cartesian_track(raw)

    hw_lw = _mapping(
        _mapping(raw.get("hw"), name="hw").get("lw100"), name="hw.lw100"
    )
    rail_raw = _mapping(
        hard.get("rail", inner.get("rail")), name="rail"
    )
    rail = _parse_rail(rail_raw, hw_lw)
    if "hard_min_m" in rail_raw or "hard_max_m" in rail_raw:
        band = float(qp.limit_damper_band_rail_m)
        lo_gap = float(rail.soft_min_m) - float(rail.hard_min_m)
        hi_gap = float(rail.hard_max_m) - float(rail.soft_max_m)
        if abs(lo_gap - band) > 1.0e-6 or abs(hi_gap - band) > 1.0e-6:
            raise ValueError(
                "rail damper band must equal the hard–soft gap "
                f"(band={band:.6f}, lo_gap={lo_gap:.6f}, hi_gap={hi_gap:.6f})"
            )
    rail_extension.soft_min_m = float(rail.hard_min_m)
    rail_extension.soft_max_m = float(rail.hard_max_m)
    policy = str(rail_extension.escape_sign_policy).strip().lower()
    if policy not in {"minus", "plus", "-", "+", "neg", "negative", "pos", "positive"}:
        raise ValueError(
            "rail_extension.escape_sign_policy must be 'minus' or 'plus', "
            f"got {rail_extension.escape_sign_policy!r}"
        )
    rail_extension.escape_sign_policy = "minus" if policy in {
        "minus", "-", "neg", "negative"
    } else "plus"

    def hard_value(name: str, legacy_name: str, default):
        return hard.get(name, inner.get(legacy_name, default))

    cfg = JointIkConfig(
        dt=_finite_float(timing.get("dt_ms", 5.0), name="timing.dt_ms") / 1000.0,
        feedback_timeout_s=_finite_float(
            timing.get("feedback_timeout_ms", 50.0),
            name="timing.feedback_timeout_ms",
        )
        / 1000.0,
        control_cpu=(
            int(timing["control_cpu"])
            if timing.get("control_cpu") is not None
            else None
        ),
        disable_cstates=bool(timing.get("disable_cstates", True)),
        qp_use_cpp_kernel=bool(timing.get("qp_use_cpp_kernel", True)),
        control_frame=str(inner.get("control_frame", "tool")),
        euler_order=euler_order,
        qp=qp,
        nullspace=nullspace,
        manipulability=manipulability,
        arm_angle=arm_angle,
        psi_retarget=psi_retarget,
        ird=ird,
        collision=collision,
        rail=rail,
        rail_extension=rail_extension,
        cartesian_track=cartesian_track,
        v_scale=_finite_float(hard_value("v_scale", "v_scale", 0.5), name="v_scale"),
        a_max_arm_rad_s2=_finite_float(
            hard_value("a_max_arm_rad_s2", "a_max_arm", 20.0), name="a_max_arm_rad_s2"
        ),
        a_max_rail_m_s2=_finite_float(
            hard_value("a_max_rail_m_s2", "a_max_rail_m_s2", 0.60),
            name="a_max_rail_m_s2",
        ),
        position_margin_rad=math.radians(
            _finite_float(
                hard_value("position_margin_deg", "position_margin_deg", 0.3),
                name="position_margin_deg",
            )
        ),
        position_margin_rail_m=_finite_float(
            hard_value("position_margin_rail_mm", "position_margin_rail_mm", 0.0),
            name="position_margin_rail_mm",
        )
        / 1000.0,
        resync_err_rad=math.radians(
            _finite_float(
                hard_value("command_lead_arm_deg", "resync_err_deg", 6.0),
                name="command_lead_arm_deg",
            )
        ),
        resync_err_rail_m=_finite_float(
            hard_value("command_lead_rail_mm", "resync_err_rail_mm", 20.0),
            name="command_lead_rail_mm",
        )
        / 1000.0,
        nullspace_d_null=_finite_float(
            inner.get("nullspace_d_null", 0.5), name="inner.nullspace_d_null"
        ),
        nullspace_d_null_adaptive=_finite_float(
            inner.get("nullspace_d_null_adaptive", 1.0),
            name="inner.nullspace_d_null_adaptive",
        ),
        nullspace_max_qdot_frac=_finite_float(
            inner.get("nullspace_max_qdot_frac", 0.2),
            name="inner.nullspace_max_qdot_frac",
        ),
        post_qp_step_clamp=bool(inner.get("post_qp_step_clamp", False)),
        qmeas_filter=str(inner.get("qmeas_filter", "lowpass") or "lowpass"),
        qmeas_lowpass_hz=_finite_float(
            inner.get("qmeas_lowpass_hz", 25.0), name="inner.qmeas_lowpass_hz"
        ),
    )
    assert_design_attractor_consistent(cfg)
    return cfg


def assert_design_attractor_consistent(cfg: JointIkConfig, kin=None) -> None:
    """Refuse a yaml whose two nullspace attractors point at different families."""
    qn = getattr(cfg.nullspace, "q_nominal_rad", None)
    if qn is None:
        return
    qn = np.asarray(qn, dtype=float).reshape(-1)
    if qn.size != 8:
        raise ValueError(
            f"nullspace.q_nominal_deg must be length 8, got {qn.size}"
        )
    from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
    from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
        d_from_q,
        fold_psi_to_positive,
    )
    from rm75_control.kinematics.srs_ik import Q_LOWER, Q_UPPER, psi_from_q

    psi_attr = float(cfg.psi_retarget.psi_attr_rad)
    d_attr = float(cfg.psi_retarget.d_attr_m)
    lo = float(cfg.psi_retarget.psi_envelope_lo_rad)
    hi = float(cfg.psi_retarget.psi_envelope_hi_rad)
    if psi_attr < lo - 1.0e-9 or psi_attr > hi + 1.0e-9:
        raise ValueError(
            "psi_retarget.psi_attr_deg must lie inside psi_envelope_deg "
            f"({math.degrees(psi_attr):.2f} not in "
            f"[{math.degrees(lo):.2f}, {math.degrees(hi):.2f}])"
        )
    psi_q = fold_psi_to_positive(psi_from_q(qn))
    psi_err = abs(psi_q - fold_psi_to_positive(psi_attr))
    if psi_err > math.radians(1.0) + 1.0e-9:
        raise ValueError(
            "q_nominal ψ disagrees with psi_attr: "
            f"ψ(q_nominal)={math.degrees(psi_q):.2f}° "
            f"psi_attr={math.degrees(psi_attr):.2f}° "
            f"(|Δ|={math.degrees(psi_err):.2f}° > 1°)"
        )
    if kin is None:
        kin = RobotKinematics()
    d_q = d_from_q(kin, qn)
    if abs(d_q - d_attr) > 0.005 + 1.0e-9:
        raise ValueError(
            "q_nominal d disagrees with d_attr: "
            f"d(q_nominal)={d_q:.4f} m d_attr={d_attr:.4f} m "
            f"(|Δ|={abs(d_q - d_attr) * 1000.0:.1f} mm > 5 mm)"
        )
    q_arm = qn[1:]
    margin = float(np.min(np.minimum(q_arm - Q_LOWER, Q_UPPER - q_arm)))
    need = float(cfg.qp.joint_comfort.activate_rad)
    if margin + 1.0e-9 < need:
        raise ValueError(
            "q_nominal worst-joint margin is inside the comfort wall: "
            f"margin={math.degrees(margin):.2f}° "
            f"joint_comfort.activate={math.degrees(need):.2f}°"
        )


__all__ = ["assert_design_attractor_consistent", "build_joint_ik_config"]
