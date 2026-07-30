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


def _arr(v, default):
    return np.asarray(v if v is not None else default, dtype=float)


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
        sigma_escape_ref_scale=float(c.get("sigma_escape_ref_scale", 2.0)),
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
    re_cfg = inner.get("rail_extension", {})
    rail_extension = RailExtensionConfig(
        enabled=bool(re_cfg.get("enabled", True)),
        k_ext=float(re_cfg.get("k_ext", 2.0)),
        k_ff=float(re_cfg.get("k_ff", 1.0)),
        v_ff_thr_m_s=float(re_cfg.get("v_ff_thr_m_s", 0.005)),
        v_ff_span_m_s=float(re_cfg.get("v_ff_span_m_s", 0.015)),
        e0_m=float(re_cfg.get("e0_m", 0.02)),
        e1_m=float(re_cfg.get("e1_m", 0.08)),
        w_max=float(re_cfg.get("w_max", 2.0)),
        v_max_m_s=float(re_cfg.get("v_max_m_s", 0.08)),
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

    rail = RailLockConfig(
        mode=rail_mode,
        locked_style=locked_style,
        q_ref_m=(float(r["q_ref_m"]) if r.get("q_ref_m") is not None else None),
        lock_gain=float(r.get("lock_gain", 200.0)),
        lock_reg_scale=float(r.get("lock_reg_scale", 100.0)),
        lock_vel_eps_m_s=float(r.get("lock_vel_eps_m_s", 0.0)),
        lock_hard_pin=bool(r.get("lock_hard_pin", True)),
        v_max_m_s=(float(r["v_max_m_s"]) if r.get("v_max_m_s") is not None else None),
        travel_m=float(r.get("travel_m", 0.80)),
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
        position_margin_rad=math.radians(margin_deg),
        position_margin_rail_m=float(inner.get("position_margin_rail_mm", 0.0)) / 1000.0,
        resync_err_rad=math.radians(resync_deg),
        resync_err_rail_m=resync_rail_mm / 1000.0,
        nullspace_d_null=float(inner.get("nullspace_d_null", 0.0)),
        nullspace_d_null_adaptive=float(inner.get("nullspace_d_null_adaptive", 1.0)),
        nullspace_max_qdot_frac=float(inner.get("nullspace_max_qdot_frac", 0.2)),
    )
