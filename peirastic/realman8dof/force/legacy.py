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
        f_ext_raw: np.ndarray | None = None,
        dt_actual: float | None = None,
        sensor_age_s: float | None = None,
        feedback_age_s: float | None = None,
        v_tcp_z_actual: float | None = None,
    ) -> ForceOutput:
        pose_d = np.asarray(pose, dtype=float).reshape(6).copy()
        dt_use = float(dt_actual) if dt_actual is not None else float(dt_s)
        cmd = self.controller.compute_velocity_command(
            pose,
            pose_d,
            np.asarray(path_twist, dtype=float).reshape(6),
            np.asarray(f_ext, dtype=float).reshape(6),
            np.asarray(f_des, dtype=float).reshape(6),
            dt_actual=dt_use,
            in_contact=contact,
            f_ext_raw=f_ext_raw,
            sensor_age_s=sensor_age_s,
            feedback_age_s=feedback_age_s,
            v_tcp_z_actual=v_tcp_z_actual,
        )
        v = np.asarray(cmd, dtype=float).reshape(6)
        # R1: emit the clamped command.  v_force_z is the pre-clamp admittance
        # state and bypasses barrier / slew / shield / corridor.
        if hasattr(self.controller, "v_force_cmd_z"):
            v_force_z = float(self.controller.v_force_cmd_z)
        else:
            v_force_z = float(v[2])
        v_force = np.zeros(6, dtype=float)
        v_force[2] = v_force_z
        return ForceOutput(
            v_force=v_force,
            v_force_z=v_force_z,
            contact_active=bool(getattr(self.controller, "contact_present", False)),
            f_des_z=float(getattr(self.controller, "f_des_z_eff", f_des[2])),
            telemetry={"v_cmd": v},
        )
