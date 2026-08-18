"""Control Barrier Function rows for self-collision avoidance (Faverjon / Khazoom)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from rm75_control.control.joint_admittance_8dof.collision_model import (
    CollisionConfig,
    CollisionModel,
    CollisionPairInfo,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


@dataclass
class CbfRows:
    jacobian: np.ndarray   # (n_rows, nv) — packed active or fixed slot layout
    lower: np.ndarray      # (n_rows,)  J_col qdot >= lower
    slot_index: np.ndarray | None = None  # (n_rows,) QP row offset within CBF block
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameJacobian:
    """Base-aligned frame Jacobian together with its point of application."""

    jacobian: np.ndarray  # (6, nv), [v_origin; omega]
    origin: np.ndarray  # (3,), expressed in the base frame


@dataclass
class CbfSlotTracker:
    """Sticky pair→row slot assignment with enter/exit hysteresis.

    Keeps the same ProxQP inequality row for a given (geom_a, geom_b) across
    ticks so warm-start multipliers do not thrash when distance rank order
    changes.  A pair leaves its slot only after ``distance > d_activate + hyst``.
    """

    max_pairs: int
    hyst_m: float = 0.01
    _keys: list[tuple[int, int] | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._keys:
            self._keys = [None] * int(self.max_pairs)

    def update(
        self,
        pairs: list[CollisionPairInfo],
        d_activate: float,
    ) -> list[CollisionPairInfo | None]:
        """Return length-``max_pairs`` list of pair-or-None per sticky slot."""
        d_keep = float(d_activate) + float(self.hyst_m)
        by_key = {(int(p.geom_a), int(p.geom_b)): p for p in pairs}

        # Drop slots that left the keep band.
        for i, key in enumerate(self._keys):
            if key is None:
                continue
            p = by_key.get(key)
            if p is None or float(p.distance) > d_keep:
                self._keys[i] = None

        occupied = {k for k in self._keys if k is not None}

        # Prefer currently active pairs (distance <= d_activate) for free slots.
        candidates = sorted(
            (p for p in pairs if float(p.distance) <= float(d_activate)),
            key=lambda p: float(p.distance),
        )
        for p in candidates:
            key = (int(p.geom_a), int(p.geom_b))
            if key in occupied:
                continue
            try:
                free = self._keys.index(None)
            except ValueError:
                break
            self._keys[free] = key
            occupied.add(key)

        out: list[CollisionPairInfo | None] = []
        for key in self._keys:
            if key is None:
                out.append(None)
            else:
                out.append(by_key.get(key))  # may be None if momentarily missing
        return out


def _frame_linear_jacobians(
    model: pin.Model,
    data: pin.Data,
    geom_model: pin.GeometryModel,
    *,
    kinematics_ready: bool = False,
) -> dict[int, FrameJacobian]:
    if not kinematics_ready:
        pin.computeJointJacobians(model, data)
        pin.updateFramePlacements(model, data)
    out: dict[int, FrameJacobian] = {}
    for go in geom_model.geometryObjects:
        fid = int(go.parentFrame)
        if fid not in out:
            J6 = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
            out[fid] = FrameJacobian(
                jacobian=np.asarray(J6, dtype=float).copy(),
                origin=np.asarray(data.oMf[fid].translation, dtype=float).copy(),
            )
    return out


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float
    )


def _point_linear_jacobian(
    frame_jac: FrameJacobian | np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    """Linear Jacobian at a collision witness point, not the frame origin."""

    if isinstance(frame_jac, FrameJacobian):
        J6 = np.asarray(frame_jac.jacobian, dtype=float)
        r = np.asarray(point, dtype=float).reshape(3) - frame_jac.origin
        # v_point = v_origin + omega x r = v_origin - skew(r) omega.
        return J6[:3, :] - _skew(r) @ J6[3:, :]
    J = np.asarray(frame_jac, dtype=float)
    if J.ndim != 2 or J.shape[0] < 3:
        raise ValueError("frame Jacobian must have at least three rows")
    return J[:3, :]


def cbf_v_safe(
    distance: float,
    cfg: CollisionConfig,
) -> float:
    """Closing-speed floor: J_col qdot >= v_safe.  Leave (positive) is free."""
    return float(-float(cfg.gamma) * (float(distance) - float(cfg.d_safe)))


def collision_jacobian(
    frame_jacs: dict[int, FrameJacobian | np.ndarray],
    geom_model: pin.GeometryModel,
    pair: CollisionPairInfo,
) -> np.ndarray:
    go_a = geom_model.geometryObjects[pair.geom_a]
    go_b = geom_model.geometryObjects[pair.geom_b]
    J_a = _point_linear_jacobian(
        frame_jacs[int(go_a.parentFrame)], pair.point_a
    )
    J_b = _point_linear_jacobian(
        frame_jacs[int(go_b.parentFrame)], pair.point_b
    )
    return pair.normal @ (J_a - J_b)


def build_cbf_rows(
    collision: CollisionModel,
    kin: RobotKinematics,
    q_rad: np.ndarray,
    cfg: CollisionConfig,
    *,
    tracker: CbfSlotTracker | None = None,
    kinematics_ready: bool = False,
) -> CbfRows:
    """Build CBF inequality rows J_col qdot >= v_safe with optional sticky slots."""
    nv = kin.nv
    if not cfg.enabled:
        return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))

    # Caller has computed J(q_meas) immediately before CBF rows and
    # explicitly proves that fact with ``kinematics_ready``.  Direct callers
    # default to CollisionModel's self-contained kinematics path.
    snapshot_ready = bool(kinematics_ready)
    band = float(cfg.d_activate) + (float(tracker.hyst_m) if tracker else 0.0)
    collision.update(
        q_rad,
        kinematic_data=kin.data if snapshot_ready else None,
        kinematics_ready=snapshot_ready,
        distance_threshold=band,
    )
    jacobian_data = kin.data if snapshot_ready else collision._kin_data  # noqa: SLF001
    raw_pairs = collision.active_pairs(band)

    if tracker is not None:
        slotted = tracker.update(raw_pairs, cfg.d_activate)
        frame_jacs = _frame_linear_jacobians(
            collision.model,
            jacobian_data,
            collision.geom_model,
            kinematics_ready=snapshot_ready,
        )
        rows = []
        lowers = []
        slots = []
        names = []
        for i, pair in enumerate(slotted):
            if pair is None:
                continue
            J_col = collision_jacobian(frame_jacs, collision.geom_model, pair)
            v_safe = cbf_v_safe(pair.distance, cfg)
            rows.append(J_col)
            lowers.append(v_safe)
            slots.append(i)
            names.append(f"self_collision:{pair.name_a}:{pair.name_b}")
        if not rows:
            return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))
        return CbfRows(
            jacobian=np.vstack(rows),
            lower=np.asarray(lowers, dtype=float),
            slot_index=np.asarray(slots, dtype=int),
            names=tuple(names),
        )

    pairs = raw_pairs[: cfg.max_pairs]
    if not pairs:
        return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))

    frame_jacs = _frame_linear_jacobians(
        collision.model,
        jacobian_data,
        collision.geom_model,
        kinematics_ready=snapshot_ready,
    )
    rows = []
    lowers = []
    names = []
    for pair in pairs:
        J_col = collision_jacobian(frame_jacs, collision.geom_model, pair)
        v_safe = cbf_v_safe(pair.distance, cfg)
        rows.append(J_col)
        lowers.append(v_safe)
        names.append(f"self_collision:{pair.name_a}:{pair.name_b}")

    return CbfRows(
        jacobian=np.vstack(rows),
        lower=np.asarray(lowers, dtype=float),
        names=tuple(names),
    )
