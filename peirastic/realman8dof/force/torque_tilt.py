"""Tool-y torque balance for probe contact, in the compensated TCP frame.

The signed wrench has the same convention as LegacyForceLaw and FCE:
the admittance drive is desired minus measured wrench. Keep the TCP contact
moment, including the pressure-center moment; projecting it parallel to force
would remove the very signal that this surface-conforming task needs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from peirastic.realman8dof.force.fce import kikuuwe_step
from peirastic.realman8dof.force.protocol import ForceOutput
from peirastic.realman8dof.force.tff import SELECTION_TOOL_Z_FORCE


@dataclass(frozen=True)
class TorqueTiltConfig:
    enabled: bool = True
    axis: int = 4
    mass: float = 0.04
    damping: float = 0.30
    coulomb_nm: float = 0.025
    vmax_rad_s: float = 0.20
    a_max: float = 5.0
    contact_only: bool = True
    contact_n: float = 0.8

    def __post_init__(self) -> None:
        if self.axis != 4:
            raise ValueError("torque_tilt.axis must be 4 (tool omega_y)")
        for name in ("mass", "damping", "vmax_rad_s", "a_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"torque_tilt.{name} must be finite and positive")
        for name in ("coulomb_nm", "contact_n"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"torque_tilt.{name} must be finite and nonnegative")

    @classmethod
    def from_dict(cls, raw: dict | None) -> "TorqueTiltConfig":
        root = raw if isinstance(raw, dict) else {}
        hm = root.get("hybrid_motion", root)
        hm = hm if isinstance(hm, dict) else {}
        block = hm.get("torque_tilt") or {}
        pc = hm.get("physical_contact") or {}
        return cls(
            enabled=bool(block.get("enabled", True)),
            axis=int(block.get("axis", 4)),
            mass=float(block.get("mass", 0.04)),
            damping=float(block.get("damping", 0.30)),
            coulomb_nm=float(block.get("coulomb_nm", 0.025)),
            vmax_rad_s=float(block.get("vmax_rad_s", 0.20)),
            a_max=float(block.get("a_max", 5.0)),
            contact_only=bool(block.get("contact_only", True)),
            contact_n=float(block.get("contact_n", pc.get("enter_n", 0.8))),
        )


def apply_tilt_selection(
    selection: np.ndarray | None, cfg: TorqueTiltConfig
) -> np.ndarray:
    """Open tool-y on the default mask; respect an explicit task selection."""
    if selection is not None:
        return np.asarray(selection, dtype=float).reshape(6).copy()
    out = np.asarray(SELECTION_TOOL_Z_FORCE, dtype=float).copy()
    if cfg.enabled:
        out[cfg.axis] = 0.0
    return out


class TorqueTilt:
    """Mass-damper/Coulomb admittance without an angle or force-error gate.

    Only angular velocity is a control state. The integrated command is
    telemetry, never an angle budget, spring reference, or position limit.
    No online moment-bias adaptation is used: it could learn away a real
    contact imbalance. A fresh calibration defines zero moment.
    """

    def __init__(self, cfg: TorqueTiltConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else TorqueTiltConfig()
        self.reset()

    def reset(self) -> None:
        self._w = 0.0
        self.tau_y = 0.0
        self.tau_error_y = 0.0
        self.omega_y = 0.0
        self.theta_tilt = 0.0
        self.stuck = True
        self.engaged = False

    def telemetry(self) -> dict:
        return {
            "tau_y": float(self.tau_y),
            "tau_error_y": float(self.tau_error_y),
            "omega_y": float(self.omega_y),
            # Integral of requested rate, NOT a measured Euler angle.
            "theta_tilt": float(self.theta_tilt),
            "stuck": bool(self.stuck),
            "tilt_engaged": bool(self.engaged),
        }

    def update(
        self,
        f_ext: np.ndarray,
        f_des: np.ndarray,
        *,
        dt_s: float,
        contact: bool | None = None,
    ) -> float:
        cfg = self.cfg
        wrench = np.asarray(f_ext, dtype=float).reshape(6)
        desired = np.asarray(f_des, dtype=float).reshape(6)
        dt = float(dt_s)
        if not np.isfinite(wrench).all() or not np.isfinite(desired).all():
            raise ValueError("torque_tilt requires a finite compensated TCP wrench")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("torque_tilt requires a finite positive dt_s")
        self.tau_y = float(wrench[cfg.axis])
        self.tau_error_y = float(desired[cfg.axis] - wrench[cfg.axis])
        normal_sign = 1.0 if desired[2] >= 0.0 else -1.0
        in_contact = (
            bool(contact)
            if contact is not None
            else normal_sign * float(wrench[2]) >= cfg.contact_n
        )
        self.engaged = bool(cfg.enabled and (in_contact or not cfg.contact_only))
        target = 0.0
        if self.engaged:
            vel, _ = kikuuwe_step(
                np.array([self._w]),
                np.array([self.tau_error_y]),
                mass=cfg.mass,
                damping=cfg.damping,
                coulomb=cfg.coulomb_nm,
                dt=dt,
                vmax=cfg.vmax_rad_s,
            )
            target = float(vel[0])
        # Slew toward the current target, never accelerate away from it due
        # to a stale jerk state. Joint acceleration/jerk limits remain in QPIK.
        # Feed the limited output back into the admittance (no hidden windup).
        self._w += float(np.clip(target - self._w, -cfg.a_max * dt, cfg.a_max * dt))
        self._w = float(np.clip(self._w, -cfg.vmax_rad_s, cfg.vmax_rad_s))
        self.omega_y = self._w
        self.theta_tilt += self._w * dt
        self.stuck = abs(self._w) <= 1e-12
        return self._w


class LegacyForceWithTilt:
    """Preserve the legacy normal-force path and add one TCP torque axis."""

    def __init__(self, z_law, tilt: TorqueTilt) -> None:
        self.z_law = z_law
        self.tilt = tilt
        self.controller = getattr(z_law, "controller", None)

    def reset(self, *, pose: np.ndarray, f_ext: np.ndarray) -> None:
        self.z_law.reset(pose=pose, f_ext=f_ext)
        self.tilt.reset()

    def update(self, **kwargs) -> ForceOutput:
        zout = self.z_law.update(**kwargs)
        contact = kwargs.get("contact")
        if contact is None and self.controller is not None:
            contact = bool(getattr(self.controller, "contact_present", False))
        dt_s = kwargs.get("dt_actual")
        if dt_s is None:
            dt_s = kwargs.get("dt_s", 0.005)
        wy = self.tilt.update(
            kwargs["f_ext"], kwargs["f_des"], dt_s=float(dt_s), contact=contact
        )
        velocity = np.asarray(zout.v_force, dtype=float).reshape(6).copy()
        velocity[self.tilt.cfg.axis] = wy
        telemetry = dict(zout.telemetry or {})
        telemetry.update(self.tilt.telemetry())
        return ForceOutput(
            v_force=velocity,
            v_force_z=float(zout.v_force_z),
            contact_active=bool(zout.contact_active or self.tilt.engaged),
            f_des_z=float(zout.f_des_z),
            telemetry=telemetry,
        )
