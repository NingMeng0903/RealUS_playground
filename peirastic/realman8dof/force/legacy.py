"""Adapter around the current AdmittanceController (A path + gated B layers)."""

from __future__ import annotations

import numpy as np

from peirastic.realman8dof.force.protocol import ForceOutput
from peirastic.realman8dof.force.nominal_transaction import (
    NominalCommandTransaction,
    finite_twist,
)


class LegacyForceLaw:
    """Wrap ``AdmittanceController.compute_velocity_command``.

    A and B share this object. Optional flow and safety layers remain selected
    by their corresponding YAML modes.
    """

    def __init__(self, controller) -> None:
        self.controller = controller
        self._transaction = NominalCommandTransaction(controller)

    def reset(self, *, pose: np.ndarray, f_ext: np.ndarray) -> None:
        del f_ext
        self._transaction.reset()
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        seed = np.zeros(6, dtype=float)
        if hasattr(self.controller, "begin_hybrid_episode"):
            self.controller.begin_hybrid_episode(seed)

    def prepare(self, *, measurement_id: int | None=None,control_step_id=None,
                source_sample_id=None,source_t_s=None,**kwargs) -> ForceOutput:
        """Consume one fresh sample and return a speculative clamped output."""
        for name in ("path_twist", "f_ext_raw"):
            if kwargs.get(name) is not None:
                finite_twist(kwargs[name], name)
        return self._transaction.prepare(
            measurement_id, lambda: self.update(**kwargs),
            **{name: kwargs[name] for name in ("pose", "f_des", "f_ext", "dt_s")},
            dt_actual=kwargs.get("dt_actual"),
            control_step_id=control_step_id,source_sample_id=source_sample_id,source_t_s=source_t_s,
            measurement_fresh=kwargs.get('measurement_fresh',True),
        )

    def commit_applied(
        self, final_force_twist: np.ndarray, *,
        final_full_twist: np.ndarray | None = None,
        accepted_normal_z: float | None = None,
    ) -> None:
        """Commit the final accepted tool-frame force contribution.

        ``final_full_twist`` also carries the final motion contribution for the
        controller's command-delay history. This records software acceptance;
        it makes no assertion about measured motion or transport atomicity.
        """
        force = finite_twist(final_force_twist, "accepted force twist")
        self._transaction.commit_applied(
            force[2] if accepted_normal_z is None else accepted_normal_z,
            final_full_twist=final_full_twist,
        )

    def abort(self) -> None:
        self._transaction.abort()

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
        measurement_fresh: bool = True,
        source_dt_s: float | None = None,
        velocity_measurement_fresh: bool | None = None,
        velocity_dt_s: float | None = None,
        **_kwargs,
    ) -> ForceOutput:
        if self._transaction.pending is not None:
            raise RuntimeError("cannot update while a nominal proposal is pending")
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
            measurement_fresh=measurement_fresh,
            source_dt_s=source_dt_s,
            velocity_measurement_fresh=velocity_measurement_fresh,
            velocity_dt_s=velocity_dt_s,
        )
        v = np.asarray(cmd, dtype=float).reshape(6)
        # R1: emit the clamped command.  v_force_z is the pre-clamp admittance
        # state and bypasses barrier / slew / shield.
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
