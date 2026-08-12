"""Measured-state construction of the common hard P0 velocity constraints."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import (
    CollisionConfig,
    CollisionModel,
)
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    LinearConstraintSet,
    RobotState,
)
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import (
    CbfSlotTracker,
    build_cbf_rows,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


class P0SafetyBuilder:
    """Build one named P0 set from a single immutable measured snapshot."""

    def __init__(
        self,
        kin,
        limits: SafetyLimits,
        *,
        collision_config: CollisionConfig | None = None,
        collision: CollisionModel | None = None,
        damper_band: float | np.ndarray = 0.15,
    ) -> None:
        self.kin = kin
        self.limits = limits
        self.velocity_box = VelocityBoxConstraints(
            limits, damper_band_rad=damper_band
        )
        self.collision_config = collision_config or CollisionConfig(enabled=False)
        self.collision = collision
        if self.collision_config.enabled and self.collision is None:
            self.collision = CollisionModel(
                kin.model,
                collision_urdf=self.collision_config.collision_urdf,
                pair_config=self.collision_config.pair_config,
            )
        self._slots = CbfSlotTracker(
            max_pairs=max(1, int(self.collision_config.max_pairs))
        )

    def set_collision_enabled(self, enabled: bool) -> None:
        requested = bool(enabled)
        if requested and self.collision is None:
            self.collision = CollisionModel(
                self.kin.model,
                collision_urdf=self.collision_config.collision_urdf,
                pair_config=self.collision_config.pair_config,
            )
        self.collision_config.enabled = requested

    def build(
        self,
        state: RobotState,
        *,
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        application_rows: Sequence[HardConstraintRow] = (),
        measured_kinematics_ready: bool = False,
    ) -> LinearConstraintSet:
        if state.n_joints != int(self.kin.nv):
            raise ValueError(
                f"state has {state.n_joints} joints, expected {self.kin.nv}"
            )
        lo_box, hi_box = self.velocity_box.bounds(
            state.q_meas,
            state.dt,
            state.qdot_applied_prev,
            q_meas=state.q_meas,
            q_cmd=state.q_cmd,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
        )

        matrices = [np.eye(state.n_joints)]
        lower = [lo_box]
        upper = [hi_box]
        names = [f"joint_velocity_box:{index}" for index in range(state.n_joints)]

        # Collision rows occupy sticky fixed slots even while inactive.  This
        # preserves row identity for the two persistent ProxQP instances and
        # their dual warm starts; active pairs never get compacted into a
        # different row on the next tick.
        n_collision_slots = max(1, int(self.collision_config.max_pairs))
        collision_C = np.zeros((n_collision_slots, state.n_joints), dtype=float)
        collision_lo = np.full(n_collision_slots, -np.inf, dtype=float)
        collision_names = [
            f"self_collision:slot:{slot}" for slot in range(n_collision_slots)
        ]
        if self.collision is not None and self.collision_config.enabled:
            cbf = build_cbf_rows(
                self.collision,
                self.kin,
                state.q_meas,
                self.collision_config,
                tracker=self._slots,
                kinematics_ready=measured_kinematics_ready,
            )
            for active_index in range(cbf.jacobian.shape[0]):
                slot = (
                    int(cbf.slot_index[active_index])
                    if cbf.slot_index is not None
                    else active_index
                )
                if not 0 <= slot < n_collision_slots:
                    continue
                collision_C[slot] = cbf.jacobian[active_index]
                collision_lo[slot] = cbf.lower[active_index]
                if cbf.names and active_index < len(cbf.names):
                    collision_names[slot] = str(cbf.names[active_index])
        matrices.append(collision_C)
        lower.append(collision_lo)
        upper.append(np.full(n_collision_slots, np.inf))
        names.extend(collision_names)

        for row in application_rows:
            if row.dimension != state.n_joints:
                raise ValueError(
                    f"constraint {row.name!r} has dimension {row.dimension}, "
                    f"expected {state.n_joints}"
                )
            matrices.append(row.a.reshape(1, -1))
            lower.append(np.array([-np.inf if row.lower is None else row.lower]))
            upper.append(np.array([np.inf if row.upper is None else row.upper]))
            names.append(row.name)

        return LinearConstraintSet(
            np.vstack(matrices),
            np.concatenate(lower),
            np.concatenate(upper),
            names,
        )


__all__ = ["P0SafetyBuilder"]
