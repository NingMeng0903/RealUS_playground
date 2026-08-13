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
    _frame_linear_jacobians,
    collision_jacobian,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


class CollisionHardCapacityExceeded(RuntimeError):
    """More absolute-hard collision events exist than the fixed QP can hold."""

    def __init__(self, active: int, capacity: int) -> None:
        super().__init__(
            "absolute self-collision pairs exceed fixed hard capacity: "
            f"{int(active)} > {int(capacity)}"
        )
        self.active = int(active)
        self.capacity = int(capacity)


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
        self._warning_slots = CbfSlotTracker(max_pairs=4)
        self.last_collision_warning_C = np.zeros((4, int(kin.nv)))
        self.last_collision_warning_lower = np.full(4, -np.inf)
        self.last_collision_warning_names = tuple(
            f"self_collision_warning:slot:{index}" for index in range(4)
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

        # Absolute collision rows are hard. Warning-band rows are exported to
        # the main QP as recoverable constraints with four independent slacks.
        n_collision_slots = max(1, int(self.collision_config.max_pairs))
        collision_C = np.zeros((n_collision_slots, state.n_joints), dtype=float)
        collision_lo = np.full(n_collision_slots, -np.inf, dtype=float)
        collision_names = [
            f"self_collision:slot:{slot}" for slot in range(n_collision_slots)
        ]
        warning_C = np.zeros((4, state.n_joints), dtype=float)
        warning_lower = np.full(4, -np.inf, dtype=float)
        warning_names = [f"self_collision_warning:slot:{slot}" for slot in range(4)]
        if self.collision is not None and self.collision_config.enabled:
            snapshot_ready = bool(measured_kinematics_ready)
            self.collision.update(
                state.q_meas,
                kinematic_data=self.kin.data if snapshot_ready else None,
                kinematics_ready=snapshot_ready,
            )
            keep_band = max(self._slots.hyst_m, self._warning_slots.hyst_m)
            pairs = self.collision.active_pairs(
                float(self.collision_config.d_activate) + keep_band
            )
            hard_pairs = [
                pair
                for pair in pairs
                if float(pair.distance) <= float(self.collision_config.d_safe)
            ]
            if len(hard_pairs) > n_collision_slots:
                raise CollisionHardCapacityExceeded(
                    len(hard_pairs), n_collision_slots
                )
            warning_pairs = [
                pair
                for pair in pairs
                if float(pair.distance) > float(self.collision_config.d_safe)
            ]
            hard_slotted = self._slots.update(
                hard_pairs, float(self.collision_config.d_safe)
            )
            warning_slotted = self._warning_slots.update(
                warning_pairs, float(self.collision_config.d_activate)
            )
            jacobian_data = (
                self.kin.data if snapshot_ready else self.collision._kin_data
            )
            frame_jacs = _frame_linear_jacobians(
                self.collision.model,
                jacobian_data,
                self.collision.geom_model,
                kinematics_ready=snapshot_ready,
            )
            for slot, pair in enumerate(hard_slotted):
                if pair is None:
                    continue
                collision_C[slot] = collision_jacobian(
                    frame_jacs, self.collision.geom_model, pair
                )
                collision_lo[slot] = -float(self.collision_config.gamma) * (
                    float(pair.distance) - float(self.collision_config.d_safe)
                )
                collision_names[slot] = (
                    f"self_collision:{pair.name_a}:{pair.name_b}"
                )
            for slot, pair in enumerate(warning_slotted):
                if pair is None:
                    continue
                warning_C[slot] = collision_jacobian(
                    frame_jacs, self.collision.geom_model, pair
                )
                warning_lower[slot] = -float(self.collision_config.gamma) * (
                    float(pair.distance) - float(self.collision_config.d_safe)
                )
                warning_names[slot] = (
                    f"self_collision_warning:{pair.name_a}:{pair.name_b}"
                )
        self.last_collision_warning_C = warning_C
        self.last_collision_warning_lower = warning_lower
        self.last_collision_warning_names = tuple(warning_names)
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


__all__ = ["CollisionHardCapacityExceeded", "P0SafetyBuilder"]
