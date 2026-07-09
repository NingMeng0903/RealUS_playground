"""YAML -> JointIkConfig loader for the joint-space inner loop.

Keeps the inner-loop tuning (QP weights, CBF, nullspace/arm-angle, safety
limits) in one config section so bring-up is a matter of editing yaml, not
code.  The outer admittance loop is configured via admittance_common keys and built via AdmittanceConfig.from_dict.
"""

from __future__ import annotations

import math

import numpy as np

from rm75_control.control.joint_admittance.loop import JointIkConfig
from rm75_control.control.joint_admittance.ik_types import SrDampingConfig
from rm75_control.control.joint_admittance.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance.collision_model import CollisionConfig
from rm75_control.control.joint_admittance.tasks.arm_angle import ArmAngleTaskConfig
from rm75_control.control.joint_admittance.tasks.manipulability_task import ManipulabilityTaskConfig
from rm75_control.control.joint_admittance.tasks.nullspace_task import NullspaceTaskConfig


def _arr(v, default):
    return np.asarray(v if v is not None else default, dtype=float)


def build_joint_ik_config(raw: dict) -> JointIkConfig:
    timing = raw.get("timing", {})
    dt = float(timing.get("dt_ms", 5.0)) / 1000.0

    inner = raw.get("inner", {})
    euler_order = str(raw.get("frames", {}).get("euler_order", inner.get("euler_order", "xyz")))

    c = inner.get("qp", {})
    reg = c.get("reg", 1e-2)
    if isinstance(reg, (list, tuple)):
        reg_arr = _arr(reg, [1e-2] * 7)
    else:
        reg_arr = np.full(7, float(reg))

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

    qp = QpConfig(
        task_weight=_arr(c.get("task_weight"), [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]),
        reg=reg_arr,
        backend=str(c.get("backend", "proxqp")),
        eps_abs=float(c.get("eps_abs", 1e-6)),
        max_iter=int(c.get("max_iter", 200)),
        euler_order=euler_order,
        collision=collision,
        sr_damping=sr_damping,
        task_weight_min_frac=float(c.get("task_weight_min_frac", 0.01)),
        use_mass_weighted_reg=bool(c.get("use_mass_weighted_reg", False)),
        mass_reg_floor=float(c.get("mass_reg_floor", 0.05)),
        use_dyn_nullspace=bool(c.get("use_dyn_nullspace", False)),
        limit_damper_band_rad=float(c.get("limit_damper_band_rad", 0.15)),
    )

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
    arm_angle = ArmAngleTaskConfig(
        enabled=bool(a.get("enabled", False)),
        k_psi=float(a.get("k_psi", 1.0)),
        psi_ref_rad=(math.radians(float(psi_ref_deg)) if psi_ref_deg is not None else None),
    )

    margin_deg = float(inner.get("position_margin_deg", 1.0))
    resync_deg = float(inner.get("resync_err_deg", 6.0))

    return JointIkConfig(
        dt=dt,
        control_frame=str(inner.get("control_frame", "tool")),
        euler_order=euler_order,
        qp=qp,
        nullspace=nullspace,
        manipulability=manipulability,
        arm_angle=arm_angle,
        v_scale=float(inner.get("v_scale", 0.5)),
        a_max=float(inner.get("a_max", 20.0)),
        position_margin_rad=math.radians(margin_deg),
        resync_err_rad=math.radians(resync_deg),
        nullspace_d_null=float(inner.get("nullspace_d_null", 0.0)),
        nullspace_d_null_adaptive=float(inner.get("nullspace_d_null_adaptive", 1.0)),
        nullspace_max_qdot_frac=float(inner.get("nullspace_max_qdot_frac", 0.2)),
    )
