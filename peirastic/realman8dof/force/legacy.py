"""Adapter around the current AdmittanceController (A path + gated B layers)."""

from __future__ import annotations

import numpy as np

from peirastic.realman8dof.force.protocol import ForceOutput


class LegacyForceLaw:
    """Wrap ``AdmittanceController.compute_velocity_command``.

    A and B share this object. B layers stay behind yaml
    ``bidirectional_flow.mode`` / ``surface_force_modulation``.
    """

    def __init__(self, controller) -> None:
        self.controller = controller

    def reset(self, *, pose: np.ndarray, f_ext: np.ndarray) -> None:
        del f_ext
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        seed = np.zeros(6, dtype=float)
        if hasattr(self.controller, "begin_hybrid_episode"):
            self.controller.begin_hybrid_episode(seed)

    def update(
        self,
        *,
        dt_s: float,
        pose: np.ndarray,
        f_ext: np.ndarray,
        f_des: np.ndarray,
        path_twist: np.ndarray,
        contact: bool | None = None,
    ) -> ForceOutput:
        pose_d = np.asarray(pose, dtype=float).reshape(6).copy()
        cmd = self.controller.compute_velocity_command(
            pose,
            pose_d,
            np.asarray(path_twist, dtype=float).reshape(6),
            np.asarray(f_ext, dtype=float).reshape(6),
            np.asarray(f_des, dtype=float).reshape(6),
            dt_actual=float(dt_s),
            in_contact=contact,
        )
        v = np.asarray(cmd, dtype=float).reshape(6)
        v_force_z = float(getattr(self.controller, "v_force_z", v[2]))
        v_force = np.zeros(6, dtype=float)
        v_force[2] = v_force_z
        return ForceOutput(
            v_force=v_force,
            v_force_z=v_force_z,
            contact_active=bool(getattr(self.controller, "contact_present", False)),
            f_des_z=float(getattr(self.controller, "f_des_z_eff", f_des[2])),
            telemetry={"v_cmd": v},
        )
