"""YAML -> JointIkConfig loader for the joint-space inner loop.

Keeps the inner-loop tuning (QP weights, CBF, nullspace/arm-angle, safety
limits) in one config section so bring-up is a matter of editing yaml, not
code.  The outer admittance loop is configured via admittance_common keys and built via AdmittanceConfig.from_dict.
"""

from __future__ import annotations

import math

import numpy as np

from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig
from rm75_control.control.joint_admittance_8dof.ik_types import SrDampingConfig
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import ArmAngleTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import ManipulabilityTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import NullspaceTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import RailExtensionConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.utils.safety import (
    RAIL_ESCAPE_ACCEL_M_S2,
)


def _arr(v, default):
    return np.asarray(v if v is not None else default, dtype=float)


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


def _shared_float(
    primary: dict,
    primary_key: str,
    alias: dict,
    alias_key: str,
    *,
    default: float,
    name: str,
    compare=lambda x: x,
) -> float:
    """Resolve a value shared by the QP and rail-extension sections.

    ``inner.qp`` keys are retained as compatibility aliases for older YAMLs;
    ``inner.rail_extension`` is the canonical section.  If both spellings are
    present they must agree, otherwise startup fails instead of silently
    running the two controllers with different envelopes.
    """

    has_primary = primary_key in primary
    has_alias = alias_key in alias
    p = _finite_float(primary[primary_key], name=f"inner.qp.{primary_key}") if has_primary else None
    a = (
        _finite_float(alias[alias_key], name=f"inner.rail_extension.{alias_key}")
        if has_alias
        else None
    )
    if p is not None and a is not None and not math.isclose(
        float(compare(p)), float(compare(a)), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"{name} mismatch: inner.qp.{primary_key}={p:.12g} vs "
            f"inner.rail_extension.{alias_key}={a:.12g}"
        )
    if a is not None:
        return a
    if p is not None:
        return p
    return _finite_float(default, name=name)


def _validate_escape_thresholds(
    sigma_escape_enter: float,
    sigma_limit_escape_enter: float,
    sigma_escape_exit: float,
) -> None:
    """Validate the one-way singularity escape hysteresis ordering."""

    if not (
        0.0 < sigma_escape_enter
        <= sigma_limit_escape_enter
        <= sigma_escape_exit
    ):
        raise ValueError(
            "invalid singularity escape thresholds: expected "
            "0 < sigma_escape_enter <= sigma_limit_escape_enter <= "
            f"sigma_escape_exit, got {sigma_escape_enter:.12g}, "
            f"{sigma_limit_escape_enter:.12g}, {sigma_escape_exit:.12g}"
        )


def _validate_escape_velocity(v_min: float, v_max: float) -> None:
    """Validate the non-negative rail escape speed envelope."""

    if not (0.0 <= v_min <= v_max):
        raise ValueError(
            "invalid rail escape velocity envelope: expected "
            f"0 <= v_min <= v_max, got {v_min:.12g}, {v_max:.12g}"
        )


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
    raise ValueError(f"unknown inner.rail.mode: {r.get('mode')!r}")


def build_joint_ik_config(raw: dict) -> JointIkConfig:
    timing = raw.get("timing", {})
    dt = float(timing.get("dt_ms", 5.0)) / 1000.0

    inner = raw.get("inner", {})
    euler_order = str(raw.get("frames", {}).get("euler_order", inner.get("euler_order", "xyz")))

    c = inner.get("qp", {})
    # ``rail_extension`` is the canonical source for the runtime rail escape
    # envelope.  The QP spellings below remain accepted as compatibility
    # aliases, but both sections are resolved once so the derived objects
    # cannot disagree.
    re_cfg = inner.get("rail_extension", {})
    sigma_escape_enter = _finite_float(
        c.get("sigma_escape_enter", 0.10), name="inner.qp.sigma_escape_enter"
    )
    sigma_limit_escape_enter = _finite_float(
        c.get("sigma_limit_escape_enter", 0.12),
        name="inner.qp.sigma_limit_escape_enter",
    )
    sigma_escape_exit = _finite_float(
        c.get("sigma_escape_exit", 0.12), name="inner.qp.sigma_escape_exit"
    )
    _validate_escape_thresholds(
        sigma_escape_enter,
        sigma_limit_escape_enter,
        sigma_escape_exit,
    )
    escape_v_min = _shared_float(
        c,
        "rail_escape_v_min_m_s",
        re_cfg,
        "escape_v_min_m_s",
        default=0.010,
        name="rail escape v_min_m_s",
    )
    escape_v_max = _shared_float(
        c,
        "rail_escape_v_max_m_s",
        re_cfg,
        "escape_v_max_m_s",
        default=0.020,
        name="rail escape v_max_m_s",
    )
    _validate_escape_velocity(escape_v_min, escape_v_max)
    # This is the one rail escape slew shared by QP and SafetyLimiter.  Keep
    # it in the inner section so startup rejects NaN/Inf/negative overrides
    # before any controller object is constructed.
    a_max_rail_escape = _finite_float(
        inner.get("a_max_rail_escape_m_s2", RAIL_ESCAPE_ACCEL_M_S2),
        name="inner.a_max_rail_escape_m_s2",
    )
    if a_max_rail_escape < 0.0:
        raise ValueError(
            "inner.a_max_rail_escape_m_s2 must be non-negative, "
            f"got {a_max_rail_escape:.12g}"
        )
    if "rail_escape_accel_m_s2" in c:
        qp_escape = _finite_float(
            c["rail_escape_accel_m_s2"],
            name="inner.qp.rail_escape_accel_m_s2",
        )
        if qp_escape < 0.0:
            raise ValueError(
                "inner.qp.rail_escape_accel_m_s2 must be non-negative, "
                f"got {qp_escape:.12g}"
            )
        if not math.isclose(
            qp_escape, a_max_rail_escape, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "rail escape acceleration mismatch: inner.a_max_rail_escape_m_s2="
                f"{a_max_rail_escape:.12g} vs inner.qp.rail_escape_accel_m_s2="
                f"{qp_escape:.12g}"
            )

    # Healthy relocation aliases resolve to the wider preferred-task budget;
    # deep escape is tightened independently by the final QP boundary.
    rail_weight_hard_max = _shared_float(
        c,
        "rail_task_weight_hard_max",
        re_cfg,
        "weight_hard_max",
        default=4.5,
        name="rail task hard weight",
    )
    if rail_weight_hard_max < 0.0:
        raise ValueError(
            "rail task hard weight must be non-negative, "
            f"got {rail_weight_hard_max:.12g}"
        )
    # Older configs occasionally used a larger scheduling cap.  Keep those
    # inputs loadable, but normalize them to the current healthy hard boundary;
    # duplicate aliases were already compared before this clipping step.
    rail_weight_hard_max = min(rail_weight_hard_max, 4.5)
    rail_weight_max_frac = _shared_float(
        c,
        "rail_task_weight_max_frac",
        re_cfg,
        "task_weight_max_frac",
        default=0.80,
        name="rail task weight fraction",
    )
    if not 0.0 <= rail_weight_max_frac <= 1.0:
        raise ValueError(
            "rail task weight fraction must be within [0, 1], "
            f"got {rail_weight_max_frac:.12g}"
        )
    # As with the absolute cap, a legacy wider fraction remains compatible but
    # cannot widen the healthy hierarchy.  Deep escape is tightened in QP.
    rail_weight_max_frac = min(rail_weight_max_frac, 0.80)
    limit_escape_activation = _finite_float(
        c.get("limit_escape_activation", 0.80),
        name="inner.qp.limit_escape_activation",
    )
    if not 0.0 <= limit_escape_activation <= 1.0:
        raise ValueError(
            "inner.qp.limit_escape_activation must be within [0, 1], "
            f"got {limit_escape_activation:.12g}"
        )
    reg = c.get("reg", None)
    if isinstance(reg, (list, tuple)):
        reg_arr = _arr(reg, [1e-2] * 8)
    elif reg is None:
        reg_arr = None  # let QpConfig defaults through
    else:
        reg_arr = np.full(8, float(reg))

    coll = inner.get("collision", {})
    collision = CollisionConfig(
        enabled=bool(coll.get("enabled", True)),
        d_safe=float(coll.get("d_safe", 0.03)),
        d_activate=float(coll.get("d_activate", 0.08)),
        gamma=float(coll.get("gamma", 5.0)),
        max_pairs=int(coll.get("max_pairs", 8)),
    )

    sr = c.get("sr_damping", {})
    sr_damping = SrDampingConfig(
        lam0=float(sr.get("lam0", 0.05)),
        sigma_ref=float(sr.get("sigma_ref", 0.08)),
        sigma_floor=float(sr.get("sigma_floor", 1e-6)),
    )

    qp_kwargs: dict = dict(
        task_weight=_arr(c.get("task_weight"), [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]),
        backend=str(c.get("backend", "proxqp")),
        eps_abs=float(c.get("eps_abs", 1e-6)),
        max_iter=int(c.get("max_iter", 200)),
        euler_order=euler_order,
        collision=collision,
        sr_damping=sr_damping,
        use_dyn_nullspace=bool(c.get("use_dyn_nullspace", False)),
        limit_damper_band_rad=float(c.get("limit_damper_band_rad", 0.15)),
        limit_damper_band_rail_m=float(c.get("limit_damper_band_rail_m", 0.05)),
        warn_on_fail=bool(c.get("warn_on_fail", True)),
        mass_reg_floor=float(c.get("mass_reg_floor", 0.05)),
        mass_weight_exempt_rail=bool(c.get("mass_weight_exempt_rail", True)),
        mass_reg_lpf_tau_s=float(c.get("mass_reg_lpf_tau_s", 0.2)),
        task_weight_min_frac=float(c.get("task_weight_min_frac", 0.05)),
        task_weight_lpf_tau_s=float(c.get("task_weight_lpf_tau_s", 0.25)),
        max_iter_cap=int(c.get("max_iter_cap", 400)),
        fail_qdot_decay=float(c.get("fail_qdot_decay", 0.85)),
        max_solve_ms=float(c.get("max_solve_ms", 8.0)),
        twist_sigma_floor=float(c.get("twist_sigma_floor", 0.08)),
        # Continuous full-twist brake: the canonical Stage-1 LPF is 80 ms.
        twist_scale_lpf_tau_s=float(c.get("twist_scale_lpf_tau_s", 0.08)),
        # Kept for old YAMLs; explicit absolute thresholds below are the
        # runtime hysteresis contract (enter=.10, exit=.12).
        sigma_escape_ref_scale=float(c.get("sigma_escape_ref_scale", 1.25)),
        sigma_escape_enter=sigma_escape_enter,
        sigma_escape_exit=sigma_escape_exit,
        sigma_limit_escape_enter=sigma_limit_escape_enter,
        limit_escape_activation=limit_escape_activation,
        rail_task_weight_hard_max=rail_weight_hard_max,
        rail_task_weight_max_frac=rail_weight_max_frac,
        rail_escape_v_min_m_s=escape_v_min,
        rail_escape_v_max_m_s=escape_v_max,
        rail_escape_accel_m_s2=a_max_rail_escape,
    )
    if reg_arr is not None:
        qp_kwargs["reg"] = reg_arr
    if "use_mass_weighted_reg" in c:
        qp_kwargs["use_mass_weighted_reg"] = bool(c["use_mass_weighted_reg"])
    qp = QpConfig(**qp_kwargs)

    n = inner.get("nullspace", {})
    q_nominal_deg = n.get("q_nominal_deg")
    nullspace = NullspaceTaskConfig(
        k_center=float(n.get("k_center", 0.5)),
        k_limit=float(n.get("k_limit", 2.0)),
        activation=float(n.get("activation", 0.85)),
        weights=(np.asarray(n["weights"], dtype=float) if n.get("weights") is not None else None),
        q_nominal_rad=(
            np.radians(np.asarray(q_nominal_deg, dtype=float)) if q_nominal_deg is not None else None
        ),
    )

    m = n.get("manipulability", {})
    manipulability = ManipulabilityTaskConfig(
        k_mu=float(m.get("k_mu", 0.8)),
        eps_rad=float(m.get("eps_rad", 1e-4)),
        sigma_fade_ref=float(m.get("sigma_fade_ref", 0.12)),
    )

    a = inner.get("arm_angle", {})
    psi_ref_deg = a.get("psi_ref_deg")
    psi_home_deg = a.get("psi_home_deg")
    psi_hard_lower_deg = a.get("psi_hard_lower_deg")
    psi_hard_upper_deg = a.get("psi_hard_upper_deg")
    arm_angle = ArmAngleTaskConfig(
        enabled=bool(a.get("enabled", False)),
        k_psi=float(a.get("k_psi", 1.0)),
        psi_ref_rad=(math.radians(float(psi_ref_deg)) if psi_ref_deg is not None else None),
        psi_home_rad=(math.radians(float(psi_home_deg)) if psi_home_deg is not None else None),
        max_psi_swing_rad=math.radians(float(a.get("max_psi_swing_deg", 150.0))),
        psi_hard_lower_rad=(
            math.radians(float(psi_hard_lower_deg)) if psi_hard_lower_deg is not None else None
        ),
        psi_hard_upper_rad=(
            math.radians(float(psi_hard_upper_deg)) if psi_hard_upper_deg is not None else None
        ),
    )

    margin_deg = float(inner.get("position_margin_deg", 1.0))
    resync_deg = float(inner.get("resync_err_deg", 6.0))
    resync_rail_mm = float(inner.get("resync_err_rail_mm", 20.0))

    a_max_arm = float(inner.get("a_max_arm", 20.0))
    a_max_rail = float(inner.get("a_max_rail_m_s2", 0.5))

    r = inner.get("rail", {})
    rail_mode, locked_style = _resolve_rail_mode(r)
    hw_lw = (raw.get("hw", {}) or {}).get("lw100", {}) or {}
    # One canonical usable rail band.  Old hardware-side fields remain a
    # compatibility mirror, but conflicting values are a startup error: a QP
    # and a bridge must never believe in different endpoints.
    inner_has_soft = "soft_min_m" in r or "soft_max_m" in r
    hw_has_soft = "soft_min_m" in hw_lw or "soft_max_m" in hw_lw
    soft_min = _finite_float(
        r.get("soft_min_m", hw_lw.get("soft_min_m", 0.01)),
        name="rail soft_min_m",
    )
    soft_max = _finite_float(
        r.get("soft_max_m", hw_lw.get("soft_max_m", 0.78)),
        name="rail soft_max_m",
    )
    if inner_has_soft and hw_has_soft:
        hw_soft_min = _finite_float(
            hw_lw.get("soft_min_m", soft_min), name="hw.lw100.soft_min_m"
        )
        hw_soft_max = _finite_float(
            hw_lw.get("soft_max_m", soft_max), name="hw.lw100.soft_max_m"
        )
        if (
            abs(soft_min - hw_soft_min) > 1.0e-6
            or abs(soft_max - hw_soft_max) > 1.0e-6
        ):
            raise ValueError(
                "rail soft-limit mismatch: inner.rail "
                f"[{soft_min:.6f}, {soft_max:.6f}] vs hw.lw100 "
                f"[{hw_soft_min:.6f}, {hw_soft_max:.6f}]"
            )
    travel_m = _finite_float(r.get("travel_m", 0.80), name="rail travel_m")
    if not (
        np.isfinite(soft_min)
        and np.isfinite(soft_max)
        and 0.0 <= soft_min < soft_max <= travel_m
    ):
        raise ValueError(
            "invalid rail soft limits: expected 0 <= soft_min < soft_max "
            f"<= travel_m ({travel_m:.6f}), got "
            f"[{soft_min:.6f}, {soft_max:.6f}]"
        )
    # Keep the QP's public canonical-band fields synchronized with the same
    # inner-rail/hardware precedence used by SafetyLimits and RailExtension.
    qp.rail_soft_min_m = soft_min
    qp.rail_soft_max_m = soft_max
    escape_max_travel = _finite_float(
        re_cfg.get("escape_max_travel_m", 0.080),
        name="inner.rail_extension.escape_max_travel_m",
    )
    rail_span = soft_max - soft_min
    if not (escape_max_travel >= 0.0 and escape_max_travel <= rail_span):
        raise ValueError(
            "invalid rail escape_max_travel_m: expected "
            f"0 <= escape_max_travel_m <= soft_max_m-soft_min_m "
            f"({rail_span:.12g}), got {escape_max_travel:.12g}"
        )
    rail_extension = RailExtensionConfig(
        enabled=bool(re_cfg.get("enabled", True)),
        k_ext=float(re_cfg.get("k_ext", 2.0)),
        k_ff=float(re_cfg.get("k_ff", 1.0)),
        v_ff_thr_m_s=float(re_cfg.get("v_ff_thr_m_s", 0.005)),
        v_ff_span_m_s=float(re_cfg.get("v_ff_span_m_s", 0.015)),
        e0_m=float(re_cfg.get("e0_m", 0.02)),
        e1_m=float(re_cfg.get("e1_m", 0.08)),
        w_max=float(re_cfg.get("w_max", 2.0)),
        soft_min_m=soft_min,
        soft_max_m=soft_max,
        weight_hard_max=rail_weight_hard_max,
        task_weight_max_frac=rail_weight_max_frac,
        v_max_m_s=float(re_cfg.get("v_max_m_s", 0.08)),
        escape_v_min_m_s=escape_v_min,
        escape_v_max_m_s=escape_v_max,
        escape_max_travel_m=escape_max_travel,
        limit_margin_m=float(re_cfg.get("limit_margin_m", 0.08)),
        k_sigma_boost=float(re_cfg.get("k_sigma_boost", 2.0)),
        k_esc=float(re_cfg.get("k_esc", 0.5)),
        w_sigma_floor=float(re_cfg.get("w_sigma_floor", 1.0)),
        k_pose=float(re_cfg.get("k_pose", 2.0)),
        pose_e0_m=float(re_cfg.get("pose_e0_m", 0.005)),
        pose_e1_m=float(re_cfg.get("pose_e1_m", 0.04)),
        pose_w_max=float(re_cfg.get("pose_w_max", 4.0)),
        sigma_guard_enter=float(re_cfg.get("sigma_guard_enter", 0.45)),
        sigma_guard_exit=float(re_cfg.get("sigma_guard_exit", 0.70)),
        v_guard_max_m_s=float(re_cfg.get("v_guard_max_m_s", 0.04)),
        v_lpf_tau_s=float(re_cfg.get("v_lpf_tau_s", 0.12)),
    )
    # Direct QpIkController callers retain defensive defaults, while the
    # production JointIk path uses the rail-extension values as the canonical
    # hierarchy/escape envelope.
    qp.rail_task_weight_hard_max = float(rail_extension.weight_hard_max)
    qp.rail_task_weight_max_frac = float(rail_extension.task_weight_max_frac)
    qp.rail_escape_v_min_m_s = float(rail_extension.escape_v_min_m_s)
    qp.rail_escape_v_max_m_s = float(rail_extension.escape_v_max_m_s)

    rail = RailLockConfig(
        mode=rail_mode,
        locked_style=locked_style,
        q_ref_m=(float(r["q_ref_m"]) if r.get("q_ref_m") is not None else None),
        lock_gain=float(r.get("lock_gain", 200.0)),
        lock_reg_scale=float(r.get("lock_reg_scale", 100.0)),
        lock_vel_eps_m_s=float(r.get("lock_vel_eps_m_s", 0.0)),
        lock_hard_pin=bool(r.get("lock_hard_pin", True)),
        v_max_m_s=(float(r["v_max_m_s"]) if r.get("v_max_m_s") is not None else None),
        travel_m=travel_m,
        soft_min_m=soft_min,
        soft_max_m=soft_max,
    )

    return JointIkConfig(
        dt=dt,
        control_frame=str(inner.get("control_frame", "tool")),
        euler_order=euler_order,
        qp=qp,
        nullspace=nullspace,
        manipulability=manipulability,
        arm_angle=arm_angle,
        rail=rail,
        rail_extension=rail_extension,
        v_scale=float(inner.get("v_scale", 0.5)),
        a_max_arm_rad_s2=a_max_arm,
        a_max_rail_m_s2=a_max_rail,
        a_max_rail_escape_m_s2=a_max_rail_escape,
        position_margin_rad=math.radians(margin_deg),
        position_margin_rail_m=float(inner.get("position_margin_rail_mm", 0.0)) / 1000.0,
        resync_err_rad=math.radians(resync_deg),
        resync_err_rail_m=resync_rail_mm / 1000.0,
        nullspace_d_null=float(inner.get("nullspace_d_null", 0.0)),
        nullspace_d_null_adaptive=float(inner.get("nullspace_d_null_adaptive", 1.0)),
        nullspace_max_qdot_frac=float(inner.get("nullspace_max_qdot_frac", 0.2)),
        # Stage-1 keeps recovery neutral by default; callers may opt into a
        # stronger posture pull explicitly, but there is no implicit 3x step.
        centering_recovery_gain=float(inner.get("centering_recovery_gain", 1.0)),
        centering_recovery_max_qdot_frac=float(
            inner.get("centering_recovery_max_qdot_frac", 0.2)
        ),
        centering_recovery_tol=float(inner.get("centering_recovery_tol", 0.12)),
    )
