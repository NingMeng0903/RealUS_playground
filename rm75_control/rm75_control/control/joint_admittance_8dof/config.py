"""YAML loader for the fixed single-shot RM75 Cartesian QPIK controller."""

from __future__ import annotations

import math

import numpy as np

from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.single_qpik import (
    SingleQpikConfig,
)
from rm75_control.control.joint_admittance_8dof.health_monitor import HealthThresholds


def _mapping(value, *, name: str) -> dict:
    """Return a mapping config section, rejecting ambiguous YAML values."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_unknown(section: dict, allowed: set[str], *, name: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {name} configuration keys: " + ", ".join(unknown))


def _finite_array(value, *, name: str, ndim: int | None = None) -> np.ndarray:
    """Convert a numeric YAML vector/matrix and reject non-finite entries."""

    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {out.ndim}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _parse_health_thresholds(raw: dict) -> HealthThresholds:
    """Read health hysteresis thresholds while accepting compact nested YAML."""

    section = _mapping(raw.get("health"), name="qpik.health")
    arm = _mapping(section.get("arm"), name="qpik.health.arm")
    joint = _mapping(
        section.get("joint_margin", section.get("joint")),
        name="qpik.health.joint_margin",
    )
    wrist = _mapping(
        section.get("wrist_margin", section.get("wrist")),
        name="qpik.health.wrist_margin",
    )
    _reject_unknown(
        section,
        {
            "arm", "joint", "joint_margin", "wrist", "wrist_margin",
            "arm_warn", "arm_danger", "arm_exit", "joint_danger_deg",
            "joint_warn_deg", "joint_exit_deg", "wrist_danger_deg",
            "wrist_warn_deg", "wrist_exit_deg", "settling_s",
            "settling_time_s", "task_velocity_scales",
        },
        name="qpik.health",
    )
    _reject_unknown(
        arm, {"warn", "warn_rho", "danger", "danger_rho", "exit", "exit_rho"},
        name="qpik.health.arm",
    )
    margin_keys = {"danger_deg", "enter_deg", "warn_deg", "warn_margin_deg", "exit_deg"}
    _reject_unknown(joint, margin_keys, name="qpik.health.joint_margin")
    _reject_unknown(wrist, margin_keys, name="qpik.health.wrist_margin")

    def pick(section_value, *keys, default=None):
        for key in keys:
            if key in section_value:
                return section_value[key]
        return default

    return HealthThresholds(
        arm_warn=pick(arm, "warn", "warn_rho", default=section.get("arm_warn", 0.08)),
        arm_danger=pick(
            arm, "danger", "danger_rho", default=section.get("arm_danger", 0.04)
        ),
        arm_exit=pick(arm, "exit", "exit_rho", default=section.get("arm_exit", 0.10)),
        joint_danger_deg=pick(
            joint,
            "danger_deg",
            "enter_deg",
            default=section.get("joint_danger_deg", 15.0),
        ),
        joint_warn_deg=pick(
            joint,
            "warn_deg",
            "warn_margin_deg",
            default=section.get("joint_warn_deg", 20.0),
        ),
        joint_exit_deg=pick(
            joint,
            "exit_deg",
            default=section.get("joint_exit_deg", 25.0),
        ),
        wrist_danger_deg=pick(
            wrist,
            "danger_deg",
            "enter_deg",
            default=section.get("wrist_danger_deg", 20.0),
        ),
        wrist_warn_deg=pick(
            wrist,
            "warn_deg",
            "warn_margin_deg",
            default=section.get("wrist_warn_deg", 25.0),
        ),
        wrist_exit_deg=pick(
            wrist,
            "exit_deg",
            default=section.get("wrist_exit_deg", 30.0),
        ),
        settling_s=section.get("settling_s", section.get("settling_time_s", 0.20)),
    )


def _parse_generic_qpik(raw: dict) -> GenericQpikRuntimeConfig:
    """Parse the fixed single-shot solver and whole-body policy."""

    section = _mapping(raw.get("qpik"), name="qpik")
    solver = _mapping(section.get("solver"), name="qpik.solver")
    backend = str(solver.get("backend", "proxqp")).lower()
    if backend not in {"proxqp", "scipy"}:
        raise ValueError(
            "qpik.solver.backend must be explicit 'proxqp' or 'scipy' "
            f"(got {backend!r})"
        )

    retired = sorted(
        (
            set(section)
            & {
                "protected_task",
                "scalable_tasks",
                "task_profile",
                "compatibility",
                "reference_governor",
                "accepted_reference_governor",
                "governor",
                "psi_lift",
            }
        )
        | (
            set(solver)
            & {
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
        )
    )
    if retired:
        raise ValueError(
            "retired multi-level QPIK configuration keys: " + ", ".join(retired)
        )

    _reject_unknown(
        section,
        {
            "solver", "dexterity", "working_set", "whole_body", "health",
            "indices", "hard_limits", "task_velocity_scales",
        },
        name="qpik",
    )
    _reject_unknown(
        solver,
        {
            "backend", "max_iter", "max_iter_in", "max_solve_ms",
            "feasibility_tolerance", "equality_tolerance", "protected_limits",
            "task_scales", "protected_weight", "beta_weight", "recovery_weight",
            "recovery_linear_weight", "alpha_weight", "preference_weight",
            "smoothness_weight", "rail_smoothness_weight",
            "ridge_weight", "authority_quadratic", "authority_rise_per_s",
            "anchor_decay_tau_s", "anchor_projection_sweeps", "warm_start",
            "scipy_ftol",
        },
        name="qpik.solver",
    )

    qcfg = SingleQpikConfig(
        backend=backend,
        max_iter=int(solver.get("max_iter", 20)),
        max_iter_in=int(solver.get("max_iter_in", 10)),
        max_solve_ms=float(solver.get("max_solve_ms", 3.0)),
        feasibility_tolerance=float(solver.get("feasibility_tolerance", 1.0e-5)),
        equality_tolerance=float(solver.get("equality_tolerance", 1.0e-5)),
        protected_limits=_finite_array(
            solver.get("protected_limits", [0.010, 0.050, 0.050, 0.050]),
            name="qpik.solver.protected_limits",
            ndim=1,
        ),
        task_scales=_finite_array(
            solver.get("task_scales", [0.10, 0.50, 0.50, 0.50, 0.10, 0.10]),
            name="qpik.solver.task_scales",
            ndim=1,
        ),
        protected_weight=float(solver.get("protected_weight", 1.0e5)),
        beta_weight=float(solver.get("beta_weight", 1.0e4)),
        recovery_weight=float(solver.get("recovery_weight", 1.0e3)),
        recovery_linear_weight=float(solver.get("recovery_linear_weight", 1.0e3)),
        alpha_weight=float(solver.get("alpha_weight", 1.0e2)),
        preference_weight=float(solver.get("preference_weight", 10.0)),
        smoothness_weight=float(solver.get("smoothness_weight", 1.0)),
        rail_smoothness_weight=float(solver.get("rail_smoothness_weight", 5.0)),
        ridge_weight=float(solver.get("ridge_weight", 1.0e-4)),
        authority_quadratic=float(solver.get("authority_quadratic", 0.05)),
        authority_rise_per_s=float(solver.get("authority_rise_per_s", 2.0)),
        anchor_decay_tau_s=float(solver.get("anchor_decay_tau_s", 0.08)),
        anchor_projection_sweeps=int(solver.get("anchor_projection_sweeps", 64)),
        warm_start=bool(solver.get("warm_start", True)),
        scipy_ftol=float(solver.get("scipy_ftol", 1.0e-9)),
    )
    dexterity = _mapping(section.get("dexterity"), name="qpik.dexterity")
    _reject_unknown(
        dexterity, {"d_safe", "d_activate", "gamma", "k_d"},
        name="qpik.dexterity",
    )
    dexterity_d_safe = float(dexterity.get("d_safe", 0.04))
    dexterity_gamma = float(dexterity.get("gamma", 5.0))
    dexterity_d_activate = float(
        dexterity.get("d_activate", max(2.0 * dexterity_d_safe, dexterity_d_safe + 0.02))
    )
    dexterity_k_d = float(dexterity.get("k_d", 0.15))
    working = _mapping(section.get("working_set"), name="qpik.working_set")
    _reject_unknown(
        working, {"arm_margin_rad", "rail_margin_m", "gamma"},
        name="qpik.working_set",
    )
    working_arm_margin_rad = float(working.get("arm_margin_rad", 0.30))
    working_rail_margin_m = float(working.get("rail_margin_m", 0.02))
    working_gamma = float(working.get("gamma", 8.0))
    health = _parse_health_thresholds(section)
    task_scales = section.get(
        "task_velocity_scales",
        _mapping(section.get("health"), name="qpik.health").get(
            "task_velocity_scales", [0.10, 0.10, 0.10, 0.50, 0.50, 0.50]
        ),
    )
    task_scales_arr = _finite_array(
        task_scales, name="qpik.task_velocity_scales", ndim=1
    ).reshape(-1)
    if task_scales_arr.size != 6 or np.any(task_scales_arr <= 0.0):
        raise ValueError("qpik.task_velocity_scales must contain six positive values")

    indices = _mapping(section.get("indices"), name="qpik.indices")
    _reject_unknown(indices, {"rail", "wrist"}, name="qpik.indices")
    rail_indices = tuple(int(i) for i in indices.get("rail", (0,)))
    wrist_indices = tuple(int(i) for i in indices.get("wrist", (5, 6, 7)))
    whole_body = _mapping(section.get("whole_body"), name="qpik.whole_body")
    _reject_unknown(
        whole_body,
        {
            "arm_nominal_k", "arm_nominal_qdot_max", "rail_macro", "risk",
            "feedback_lpf_tau_s", "feedback_accel_max_m_s2",
        },
        name="qpik.whole_body",
    )
    rail_macro = _mapping(
        whole_body.get("rail_macro"), name="qpik.whole_body.rail_macro"
    )
    risk = _mapping(whole_body.get("risk"), name="qpik.whole_body.risk")
    _reject_unknown(
        rail_macro,
        {"tau_s", "v_max_m_s", "a_max_m_s2", "jerk_max_m_s3", "center_k", "center_v_max_m_s"},
        name="qpik.whole_body.rail_macro",
    )
    _reject_unknown(
        risk,
        {
            "collision_k_d", "attack_s", "release_s", "exit_dwell_s",
            "gradient_period_ticks", "gradient_lpf_tau_s", "wrist_danger_deg",
            "wrist_warn_deg", "wrist_exit_deg",
        },
        name="qpik.whole_body.risk",
    )
    return GenericQpikRuntimeConfig(
        solver=qcfg,
        health=health,
        rail_indices=rail_indices,
        wrist_indices=wrist_indices,
        task_velocity_scales=task_scales_arr,
        dexterity_d_safe=dexterity_d_safe,
        dexterity_gamma=dexterity_gamma,
        dexterity_d_activate=dexterity_d_activate,
        dexterity_k_d=dexterity_k_d,
        collision_k_d=float(risk.get("collision_k_d", 0.10)),
        working_arm_margin_rad=working_arm_margin_rad,
        working_rail_margin_m=working_rail_margin_m,
        working_gamma=working_gamma,
        arm_nominal_k=float(whole_body.get("arm_nominal_k", 0.25)),
        arm_nominal_qdot_max=float(whole_body.get("arm_nominal_qdot_max", 0.30)),
        risk_attack_s=float(risk.get("attack_s", 0.05)),
        risk_release_s=float(risk.get("release_s", 0.40)),
        risk_exit_dwell_s=float(risk.get("exit_dwell_s", 0.20)),
        gradient_period_ticks=int(risk.get("gradient_period_ticks", 10)),
        gradient_lpf_tau_s=float(risk.get("gradient_lpf_tau_s", 0.10)),
        wrist_danger_deg=float(risk.get("wrist_danger_deg", 10.0)),
        wrist_warn_deg=float(risk.get("wrist_warn_deg", 20.0)),
        wrist_exit_deg=float(risk.get("wrist_exit_deg", 25.0)),
        rail_macro_tau_s=float(rail_macro.get("tau_s", 0.15)),
        rail_macro_v_max_m_s=float(rail_macro.get("v_max_m_s", 0.12)),
        rail_macro_a_max_m_s2=float(rail_macro.get("a_max_m_s2", 0.30)),
        rail_macro_jerk_max_m_s3=float(rail_macro.get("jerk_max_m_s3", 2.0)),
        rail_center_k=float(rail_macro.get("center_k", 0.04)),
        rail_center_v_max_m_s=float(rail_macro.get("center_v_max_m_s", 0.025)),
        feedback_lpf_tau_s=float(whole_body.get("feedback_lpf_tau_s", 0.05)),
        feedback_accel_max_m_s2=float(
            whole_body.get("feedback_accel_max_m_s2", 0.30)
        ),
    )


def _finite_float(value, *, name: str) -> float:
    """Convert a config scalar and reject NaN/Inf at the loader boundary.

    Config values eventually become QP bounds and velocity envelopes.  Letting
    a non-finite value through here usually produces a much less actionable
    solver failure several layers later, so all safety-critical scalars use
    this small fail-fast conversion helper.
    """

    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return out


def _resolve_rail_mode(r: dict) -> tuple[RailMode, LockedStyle]:
    """Read (mode, locked_style) from yaml.

    Schema::
        rail:
          mode: coupled | locked
          locked_style: hold | rail_only | tcp_fixed   # only if mode=locked
    """
    mode_str = str(r.get("mode", "coupled")).lower()
    raw_style = r.get("locked_style", "hold")
    if mode_str == "coupled":
        return RailMode.COUPLED, LockedStyle.HOLD
    if mode_str == "locked":
        style = LockedStyle(str(raw_style).lower()) if raw_style else LockedStyle.HOLD
        return RailMode.LOCKED, style
    raise ValueError(f"unknown qpik.hard_limits.rail.mode: {r.get('mode')!r}")


def build_joint_ik_config(raw: dict) -> JointIkConfig:
    """Build the production controller config from responsibility blocks.

    The preferred schema is ``qpik.solver``, ``qpik.hard_limits``, task
    profiles, health and planner/guide sections.  A small read-only fallback
    for dbb's ``inner.collision`` and ``inner.rail`` is retained so older
    application YAML can be inspected without reviving the retired weighted
    QP, null-space or rail-extension APIs.
    """

    if not isinstance(raw, dict):
        raise ValueError("controller config root must be a mapping")
    timing = _mapping(raw.get("timing"), name="timing")
    inner = _mapping(raw.get("inner"), name="inner")
    qpik = _mapping(raw.get("qpik"), name="qpik")
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
    legacy_qp = _mapping(inner.get("qp"), name="inner.qp")
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
            "retired QPIK configuration keys in inner: "
            + ", ".join(retired_inner)
        )
    euler_order = str(
        _mapping(raw.get("frames"), name="frames").get(
            "euler_order", inner.get("euler_order", "xyz")
        )
    )

    collision_raw = _mapping(
        hard.get("collision", inner.get("collision")),
        name="qpik.hard_limits.collision",
    )
    _reject_unknown(
        collision_raw,
        {"enabled", "d_safe", "d_activate", "gamma", "max_pairs"},
        name="qpik.hard_limits.collision",
    )
    collision = CollisionConfig(
        enabled=bool(collision_raw.get("enabled", True)),
        d_safe=_finite_float(collision_raw.get("d_safe", 0.01), name="collision.d_safe"),
        d_activate=_finite_float(
            collision_raw.get("d_activate", 0.04), name="collision.d_activate"
        ),
        gamma=_finite_float(collision_raw.get("gamma", 5.0), name="collision.gamma"),
        max_pairs=int(collision_raw.get("max_pairs", 8)),
    )
    if not 0.0 <= collision.d_safe < collision.d_activate:
        raise ValueError("collision distances must satisfy 0 <= d_safe < d_activate")
    if collision.gamma <= 0.0 or collision.max_pairs <= 0:
        raise ValueError("collision gamma/max_pairs must be positive")

    velocity_damper = _mapping(
        hard.get("velocity_damper"), name="qpik.hard_limits.velocity_damper"
    )
    _reject_unknown(
        velocity_damper, {"arm_band_rad", "rail_band_m"},
        name="qpik.hard_limits.velocity_damper",
    )
    damper_arm = _finite_float(
        velocity_damper.get(
            "arm_band_rad", legacy_qp.get("limit_damper_band_rad", 0.15)
        ),
        name="velocity_damper.arm_band_rad",
    )
    damper_rail = _finite_float(
        velocity_damper.get(
            "rail_band_m", legacy_qp.get("limit_damper_band_rail_m", 0.05)
        ),
        name="velocity_damper.rail_band_m",
    )
    if damper_arm < 0.0 or damper_rail < 0.0:
        raise ValueError("velocity damper bands must be non-negative")

    rail_raw = _mapping(
        hard.get("rail", inner.get("rail")), name="qpik.hard_limits.rail"
    )
    _reject_unknown(
        rail_raw,
        {
            "mode", "locked_style", "q_ref_m", "lock_vel_eps_m_s",
            "v_max_m_s", "travel_m", "soft_min_m", "soft_max_m",
        },
        name="qpik.hard_limits.rail",
    )
    rail_mode, locked_style = _resolve_rail_mode(rail_raw)
    hw_lw = _mapping(
        _mapping(raw.get("hw"), name="hw").get("lw100"), name="hw.lw100"
    )
    soft_min = _finite_float(
        rail_raw.get("soft_min_m", hw_lw.get("soft_min_m", 0.01)),
        name="rail.soft_min_m",
    )
    soft_max = _finite_float(
        rail_raw.get("soft_max_m", hw_lw.get("soft_max_m", 0.78)),
        name="rail.soft_max_m",
    )
    travel = _finite_float(rail_raw.get("travel_m", 0.80), name="rail.travel_m")
    if not 0.0 <= soft_min < soft_max <= travel:
        raise ValueError("rail limits must satisfy 0 <= soft_min < soft_max <= travel")
    if rail_raw and hw_lw and ("soft_min_m" in hw_lw or "soft_max_m" in hw_lw):
        hw_min = _finite_float(hw_lw.get("soft_min_m", soft_min), name="hw rail soft_min")
        hw_max = _finite_float(hw_lw.get("soft_max_m", soft_max), name="hw rail soft_max")
        if abs(hw_min - soft_min) > 1.0e-6 or abs(hw_max - soft_max) > 1.0e-6:
            raise ValueError("rail soft-limit mismatch between QPIK and hardware")
    rail = RailLockConfig(
        mode=rail_mode,
        locked_style=locked_style,
        q_ref_m=(
            None
            if rail_raw.get("q_ref_m") is None
            else _finite_float(rail_raw["q_ref_m"], name="rail.q_ref_m")
        ),
        lock_vel_eps_m_s=_finite_float(
            rail_raw.get("lock_vel_eps_m_s", 0.0), name="rail.lock_vel_eps_m_s"
        ),
        v_max_m_s=(
            None
            if rail_raw.get("v_max_m_s") is None
            else _finite_float(rail_raw["v_max_m_s"], name="rail.v_max_m_s")
        ),
        travel_m=travel,
        soft_min_m=soft_min,
        soft_max_m=soft_max,
    )

    def hard_value(name: str, legacy_name: str, default):
        return hard.get(name, inner.get(legacy_name, default))

    return JointIkConfig(
        dt=_finite_float(timing.get("dt_ms", 5.0), name="timing.dt_ms") / 1000.0,
        feedback_timeout_s=_finite_float(
            timing.get("feedback_timeout_ms", 50.0),
            name="timing.feedback_timeout_ms",
        )
        / 1000.0,
        control_frame=str(inner.get("control_frame", "tool")),
        euler_order=euler_order,
        generic_qpik=_parse_generic_qpik(raw),
        collision=collision,
        limit_damper_band_rad=damper_arm,
        limit_damper_band_rail_m=damper_rail,
        rail=rail,
        v_scale=_finite_float(hard_value("v_scale", "v_scale", 0.5), name="v_scale"),
        a_max_arm_rad_s2=_finite_float(
            hard_value("a_max_arm_rad_s2", "a_max_arm", 20.0), name="a_max_arm_rad_s2"
        ),
        a_max_rail_m_s2=_finite_float(
            hard_value("a_max_rail_m_s2", "a_max_rail_m_s2", 0.30), name="a_max_rail_m_s2"
        ),
        position_margin_rad=math.radians(
            _finite_float(
                hard_value("position_margin_deg", "position_margin_deg", 1.0),
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
    )


__all__ = ["build_joint_ik_config"]
