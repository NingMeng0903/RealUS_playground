"""Gravity compensation modes for PEIRASTIC-compatible Genesis Franka simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from projects.genesis_ue_sync.sim_platform.control.motion import MotionInterface


def apply_genesis_material_gravity_compensation(motion: "MotionInterface", value: float) -> None:
    """Enable Genesis rigid-body gravity compensation via ``material.gravity_compensation``.

    When ``value`` is 1.0, the simulator injects joint torques so gravity effects are partially or fully
    cancelled depending on Genesis internals—similar to a ``Gravity Compensation`` assist mode, not
    libfranka ``Model::gravity``.
    """
    motion.set_gravity_compensation(float(value))


GravityTorqueFn = Callable[["MotionInterface", np.ndarray], np.ndarray]


def compute_gravity_torque_placeholder(_motion: "MotionInterface", joint_positions: np.ndarray) -> np.ndarray:
    """Reserved hook for explicit :math:`\\tau_g(q)` using Pinocchio/Drake + Panda URDF.

    Returns zero torque by default; replace with inverse dynamics gravity vector when hardware-accurate
    simulation is required with ``material.gravity_compensation`` set to 0.
    """
    return np.zeros_like(np.asarray(joint_positions, dtype=np.float32).reshape(-1), dtype=np.float32)
