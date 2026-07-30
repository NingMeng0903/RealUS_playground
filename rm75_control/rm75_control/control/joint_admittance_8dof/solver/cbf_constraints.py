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
    *,
    tracker: CbfSlotTracker | None = None,
) -> CbfRows:
    """Build CBF inequality rows J_col qdot >= v_safe with optional sticky slots."""
    nv = kin.nv
    if not cfg.enabled:
        return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))

    collision.update(q_rad)
    raw_pairs = collision.active_pairs(cfg.d_activate + (tracker.hyst_m if tracker else 0.0))

    if tracker is not None:
        slotted = tracker.update(raw_pairs, cfg.d_activate)
        kin_data = collision._kin_data  # noqa: SLF001
        frame_jacs = _frame_linear_jacobians(collision.model, kin_data, collision.geom_model)
        rows = []
        lowers = []
        slots = []
        for i, pair in enumerate(slotted):
            if pair is None:
                continue
            J_col = collision_jacobian(frame_jacs, collision.geom_model, pair)
            v_safe = -cfg.gamma * (pair.distance - cfg.d_safe)
            rows.append(J_col)
            lowers.append(v_safe)
            slots.append(i)
        if not rows:
            return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))
        return CbfRows(
            jacobian=np.vstack(rows),
            lower=np.asarray(lowers, dtype=float),
            slot_index=np.asarray(slots, dtype=int),
        )

    pairs = raw_pairs[: cfg.max_pairs]
    if not pairs:
        return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))

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
