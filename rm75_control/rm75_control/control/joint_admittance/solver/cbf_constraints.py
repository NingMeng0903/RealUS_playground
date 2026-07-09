"""Control Barrier Function rows for self-collision avoidance (Faverjon / Khazoom)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from rm75_control.control.joint_admittance.collision_model import (
    CollisionConfig,
    CollisionModel,
    CollisionPairInfo,
)
from rm75_control.control.joint_admittance.model import RobotKinematics


@dataclass
class CbfRows:
    jacobian: np.ndarray   # (n_active, nv)
    lower: np.ndarray      # (n_active,)  J_col qdot >= lower


def _frame_linear_jacobians(
    model: pin.Model,
    data: pin.Data,
    geom_model: pin.GeometryModel,
) -> dict[int, np.ndarray]:
    pin.computeJointJacobians(model, data)
    pin.updateFramePlacements(model, data)
    out: dict[int, np.ndarray] = {}
    for go in geom_model.geometryObjects:
        fid = int(go.parentFrame)
        if fid not in out:
            J6 = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
            out[fid] = np.asarray(J6[:3, :], dtype=float)
    return out


def collision_jacobian(
    frame_jacs: dict[int, np.ndarray],
    geom_model: pin.GeometryModel,
    pair: CollisionPairInfo,
) -> np.ndarray:
    go_a = geom_model.geometryObjects[pair.geom_a]
    go_b = geom_model.geometryObjects[pair.geom_b]
    J_a = frame_jacs[int(go_a.parentFrame)]
    J_b = frame_jacs[int(go_b.parentFrame)]
    return pair.normal @ (J_a - J_b)


def build_cbf_rows(
    collision: CollisionModel,
    kin: RobotKinematics,
    q_rad: np.ndarray,
    cfg: CollisionConfig,
) -> CbfRows:
    """Build active CBF inequality rows J_col qdot >= v_safe."""
    if not cfg.enabled:
        return CbfRows(jacobian=np.zeros((0, kin.nv)), lower=np.zeros(0))

    collision.update(q_rad)
    pairs = collision.active_pairs(cfg.d_activate)[: cfg.max_pairs]
    if not pairs:
        return CbfRows(jacobian=np.zeros((0, kin.nv)), lower=np.zeros(0))

    kin_data = collision._kin_data  # noqa: SLF001
    frame_jacs = _frame_linear_jacobians(collision.model, kin_data, collision.geom_model)
    rows = []
    lowers = []
    for pair in pairs:
        J_col = collision_jacobian(frame_jacs, collision.geom_model, pair)
        v_safe = -cfg.gamma * (pair.distance - cfg.d_safe)
        rows.append(J_col)
        lowers.append(v_safe)

    return CbfRows(
        jacobian=np.vstack(rows),
        lower=np.asarray(lowers, dtype=float),
    )
