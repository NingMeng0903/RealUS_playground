"""Tool-y torque balance for probe contact, in the compensated TCP frame.

The signed wrench has the same convention as LegacyForceLaw and FCE:
the admittance drive is desired minus measured wrench. Keep the TCP contact
moment, including the pressure-center moment; projecting it parallel to force
would remove the very signal that this surface-conforming task needs.

Wrap-around scan needs a large TCP angle. The contact tube provides contact
geometry telemetry. A slowly changing center of pressure alone cannot tell
surface curvature from residual torque, so its legacy stall latch is opt-in.
Contact, QPIK feasibility and the configured angle/rate limits still apply.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from peirastic.realman8dof.force.fce import kikuuwe_step
from peirastic.realman8dof.force.protocol import ForceOutput
from peirastic.realman8dof.force.tff import SELECTION_TOOL_Z_FORCE

_THETA_MAX_RAD = math.radians(150.0)
_ALIGN_MIN = 0.0


def estimate_contact_cop(
    wrench: np.ndarray,
    *,
    f_min: float,
) -> tuple[float, float, float, bool]:
    """TCP-face center of pressure from a compensated wrench.

    On the tool-Z=0 face, ``τ = r × F`` gives ``Mx = y Fz`` and
    ``My = −x Fz``, so ``x = −My/Fz`` and ``y = Mx/Fz``. The allowed
    contact set is a tube around the tool-Z line through the TCP: the
    face disk plus a rim neighborhood for oblique edge contact.
    """
    w = np.asarray(wrench, dtype=float).reshape(6)
    fz = float(w[2])
    floor = float(f_min)
    if not math.isfinite(fz) or not math.isfinite(floor) or abs(fz) < max(floor, 1e-9):
        return float("nan"), float("nan"), float("nan"), False
    cop_x = -float(w[4]) / fz
    cop_y = float(w[3]) / fz
    if not (math.isfinite(cop_x) and math.isfinite(cop_y)):
        return float("nan"), float("nan"), float("nan"), False
    return cop_x, cop_y, math.hypot(cop_x, cop_y), True


@dataclass(frozen=True)
class TorqueTiltConfig:
    enabled: bool = True
    axis: int = 4
    mass: float = 0.065
    damping: float = 0.28
    coulomb_nm: float = 0.025
    vmax_rad_s: float = 0.22
    a_max: float = 4.5
    contact_only: bool = True
    contact_n: float = 0.8
    theta_max_rad: float = _THETA_MAX_RAD
    align_min: float = _ALIGN_MIN
    slack_freeze: float = 0.05
    r_face_m: float = 0.012
    r_margin_m: float = 0.008
    cop_stall_m: float = 0.006
    # Disabled by default: curved/soft contact need not reduce CoP by 1 mm
    # within 350 ms. Stopping rotation can prevent the latch from recovering.
    cop_stall_s: float = 0.0

    def __post_init__(self) -> None:
        if self.axis != 4:
            raise ValueError("torque_tilt.axis must be 4 (tool omega_y)")
        for name in ("mass", "damping", "vmax_rad_s", "a_max", "r_face_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"torque_tilt.{name} must be finite and positive")
        for name in ("coulomb_nm", "contact_n", "slack_freeze", "r_margin_m", "cop_stall_m", "cop_stall_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"torque_tilt.{name} must be finite and nonnegative")
        tmax = float(self.theta_max_rad)
        if not math.isfinite(tmax) or tmax <= 0.0:
            raise ValueError("torque_tilt.theta_max_rad must be finite and positive")
        align = float(self.align_min)
        if not math.isfinite(align) or align < 0.0 or align > 1.0:
            raise ValueError("torque_tilt.align_min must be in [0, 1]")

    @property
    def r_tube_m(self) -> float:
        return float(self.r_face_m) + float(self.r_margin_m)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "TorqueTiltConfig":
        root = raw if isinstance(raw, dict) else {}
        hm = root.get("hybrid_motion", root)
        hm = hm if isinstance(hm, dict) else {}
        block = hm.get("torque_tilt") or {}
        if "theta_max_rad" in block:
            theta_max = float(block["theta_max_rad"])
        else:
            theta_max = math.radians(float(block.get("theta_max_deg", 150.0)))
        if "align_min" in block:
            align_min = float(block["align_min"])
        else:
            twist_deg = float(block.get("twist_align_deg", 0.0))
            align_min = 0.0 if twist_deg <= 0.0 else math.cos(math.radians(twist_deg))
        return cls(
            enabled=bool(block.get("enabled", True)),
            axis=int(block.get("axis", 4)),
            mass=float(block.get("mass", 0.065)),
            damping=float(block.get("damping", 0.28)),
            coulomb_nm=float(block.get("coulomb_nm", 0.025)),
            vmax_rad_s=float(block.get("vmax_rad_s", 0.22)),
            a_max=float(block.get("a_max", 4.5)),
            contact_only=bool(block.get("contact_only", True)),
            contact_n=float(block.get("contact_n", pc_enter(hm))),
            theta_max_rad=theta_max,
            align_min=align_min,
            slack_freeze=float(block.get("slack_freeze", 0.05)),
            r_face_m=float(block.get("r_face_m", 0.012)),
            r_margin_m=float(block.get("r_margin_m", 0.008)),
            cop_stall_m=float(block.get("cop_stall_m", 0.006)),
            cop_stall_s=float(block.get("cop_stall_s", 0.0)),
        )


def pc_enter(hm: dict) -> float:
    pc = hm.get("physical_contact") or {}
    return float(pc.get("enter_n", 0.8)) if isinstance(pc, dict) else 0.8


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


def rotation_from_pose(pose: np.ndarray, *, euler_order: str = "xyz") -> np.ndarray:
    return Rsc.from_euler(
        str(euler_order),
        np.asarray(pose, dtype=float).reshape(6)[3:6],
        degrees=False,
    ).as_matrix()


def tool_world_align(pose: np.ndarray, *, euler_order: str = "xyz") -> float:
    """Absolute |tool_z · world_z|. 1 = tool Z vertical."""
    tool_z = rotation_from_pose(pose, euler_order=euler_order)[:, 2]
    return float(abs(tool_z[2]))


class TorqueTilt:
    """Mass-damper/Coulomb admittance with contact geometry telemetry.

    ``theta_tilt`` is the integral of commanded ωy since the last contact
    rising edge. A large last-resort cap bounds accumulated rotation. CoP
    can remain off-center during valid surface following; it is not a
    default reason to latch rotation off.
    """

    def __init__(self, cfg: TorqueTiltConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else TorqueTiltConfig()
        self.reset()

    def reset(self) -> None:
        self._w = 0.0
        self._was_contact = False
        self.tau_y = 0.0
        self.tau_error_y = 0.0
        self.omega_y = 0.0
        self.theta_tilt = 0.0
        self.stuck = True
        self.engaged = False
        self.tilt_frozen = False
        self.tilt_capped = False
        self.tilt_stalled = False
        self.tilt_stop_reason = "no_contact"
        self.align_z = 1.0
        self.n_world = np.array([0.0, 0.0, 1.0], dtype=float)
        self._have_normal = False
        self.cop_x = float("nan")
        self.cop_y = float("nan")
        self.cop_r = float("nan")
        self.cop_valid = False
        self.on_face = False
        self.on_tube = False
        self._cop_x_watch = float("nan")
        self._cop_stall_t = 0.0

    def telemetry(self) -> dict:
        return {
            "tau_y": float(self.tau_y),
            "tau_error_y": float(self.tau_error_y),
            "omega_y": float(self.omega_y),
            "theta_tilt": float(self.theta_tilt),
            "stuck": bool(self.stuck),
            "tilt_engaged": bool(self.engaged),
            "tilt_frozen": bool(self.tilt_frozen),
            "tilt_capped": bool(self.tilt_capped),
            "tilt_stalled": bool(self.tilt_stalled),
            "tilt_stop_reason": str(self.tilt_stop_reason),
            "tilt_deadband_nm": float(self.cfg.coulomb_nm),
            "align_z": float(self.align_z),
            "cop_x": float(self.cop_x),
            "cop_y": float(self.cop_y),
            "cop_r": float(self.cop_r),
            "cop_valid": bool(self.cop_valid),
            "on_face": bool(self.on_face),
            "on_tube": bool(self.on_tube),
        }

    @property
    def needs_normal_retract(self) -> bool:
        return bool(self.tilt_frozen or self.tilt_stalled or (self.tilt_capped and not self.on_tube))

    def _update_cop(self, wrench: np.ndarray) -> None:
        cfg = self.cfg
        self.cop_x, self.cop_y, self.cop_r, self.cop_valid = estimate_contact_cop(
            wrench, f_min=max(float(cfg.contact_n), 1e-6)
        )
        tube = cfg.r_tube_m
        self.on_face = bool(self.cop_valid and self.cop_r <= cfg.r_face_m)
        self.on_tube = bool(self.cop_valid and self.cop_r <= tube)

    def _update_stall(self, dt: float) -> None:
        cfg = self.cfg
        stall_s = float(cfg.cop_stall_s)
        if stall_s <= 0.0 or not self.engaged or not self.cop_valid:
            self._cop_stall_t = 0.0
            self.tilt_stalled = False
            if self.cop_valid:
                self._cop_x_watch = float(self.cop_x)
            return
        cop_x = float(self.cop_x)
        improved = (
            abs(cop_x) <= float(cfg.cop_stall_m)
            or (
                math.isfinite(self._cop_x_watch)
                and (abs(cop_x) < abs(float(self._cop_x_watch)) - 0.001
                     or cop_x * float(self._cop_x_watch) < 0.0)
            )
        )
        if self.tilt_stalled:
            if improved:
                self.tilt_stalled = False
                self._cop_stall_t = 0.0
                self._cop_x_watch = cop_x
            return
        # Leftover looks like a CoP that does not walk in as the tool tilts.
        # Real wrap keeps |CoP| inside the stall deadband or shrinking.
        if abs(cop_x) <= float(cfg.cop_stall_m) or abs(self._w) <= 0.02:
            self._cop_stall_t = 0.0
            self._cop_x_watch = cop_x
            return
        if not math.isfinite(self._cop_x_watch):
            self._cop_x_watch = cop_x
            self._cop_stall_t = 0.0
            return
        if abs(cop_x) < abs(float(self._cop_x_watch)) - 0.001:
            self._cop_x_watch = cop_x
            self._cop_stall_t = 0.0
            return
        self._cop_stall_t += dt
        self.tilt_stalled = bool(self._cop_stall_t > stall_s)

    def update(
        self,
        f_ext: np.ndarray,
        f_des: np.ndarray,
        *,
        dt_s: float,
        contact: bool | None = None,
        pose: np.ndarray | None = None,
        slack_norm: float | None = None,
        euler_order: str = "xyz",
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
        if in_contact and not self._was_contact:
            self.theta_tilt = 0.0
            self._cop_x_watch = float("nan")
            self._cop_stall_t = 0.0
            self.tilt_stalled = False
        self._was_contact = bool(in_contact)
        self.engaged = bool(cfg.enabled and (in_contact or not cfg.contact_only))
        self._update_cop(wrench)

        self.align_z = 1.0
        if pose is not None:
            rot = rotation_from_pose(pose, euler_order=euler_order)
            self.align_z = float(abs(rot[2, 2]))
            if self.engaged and self.on_tube:
                outward = -rot[:, 2]
                nrm = float(np.linalg.norm(outward))
                if nrm > 1e-9:
                    self.n_world = outward / nrm
                    self._have_normal = True
        if not self._have_normal:
            self.n_world = np.array([0.0, 0.0, 1.0], dtype=float)

        slack = float(slack_norm) if slack_norm is not None else 0.0
        attitude_freeze = bool(
            cfg.align_min > 0.0
            and pose is not None
            and self.align_z < cfg.align_min
        )
        slack_freeze = bool(
            slack_norm is not None and math.isfinite(slack) and slack > cfg.slack_freeze
        )
        self.tilt_frozen = bool(self.engaged and (attitude_freeze or slack_freeze))
        at_pos = self.theta_tilt >= cfg.theta_max_rad
        at_neg = self.theta_tilt <= -cfg.theta_max_rad
        self.tilt_capped = bool(self.engaged and (at_pos or at_neg))
        self._update_stall(dt)

        target = 0.0
        self.tilt_stop_reason = ""
        if not cfg.enabled:
            self.tilt_stop_reason = "disabled"
        elif not self.engaged:
            self.tilt_stop_reason = "no_contact"
        elif attitude_freeze:
            self.tilt_stop_reason = "attitude_limit"
        elif slack_freeze:
            self.tilt_stop_reason = "qp_slack"
        elif self.tilt_stalled:
            self.tilt_stop_reason = "cop_stall"
        if self.engaged and not self.tilt_frozen and not self.tilt_stalled:
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
            if at_pos and target > 0.0:
                target = 0.0
                self.tilt_stop_reason = "angle_limit"
            if at_neg and target < 0.0:
                target = 0.0
                self.tilt_stop_reason = "angle_limit"
            if not self.tilt_stop_reason and abs(self.tau_error_y) <= cfg.coulomb_nm:
                self.tilt_stop_reason = "torque_deadband"
        self._w += float(np.clip(target - self._w, -cfg.a_max * dt, cfg.a_max * dt))
        self._w = float(np.clip(self._w, -cfg.vmax_rad_s, cfg.vmax_rad_s))
        self.omega_y = self._w
        self.theta_tilt += self._w * dt
        self.theta_tilt = float(
            np.clip(self.theta_tilt, -cfg.theta_max_rad, cfg.theta_max_rad)
        )
        self.stuck = abs(self._w) <= 1e-12
        return self._w


def remap_force_along_world_normal(
    v_force: np.ndarray,
    *,
    rotation: np.ndarray,
    n_world: np.ndarray,
    v_force_z: float,
) -> np.ndarray:
    """Apply the signed tool-Z speed along the latched contact/world normal."""
    out = np.asarray(v_force, dtype=float).reshape(6).copy()
    n = np.asarray(n_world, dtype=float).reshape(3)
    nrm = float(np.linalg.norm(n))
    if nrm <= 1e-9:
        n = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n = n / nrm
    r_mat = np.asarray(rotation, dtype=float).reshape(3, 3)
    n_press_world = -n
    n_press_tool = r_mat.T @ n_press_world
    pn = float(np.linalg.norm(n_press_tool))
    if pn <= 1e-9:
        return out
    out[:3] = float(v_force_z) * (n_press_tool / pn)
    return out


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
        euler = str(kwargs.get("euler_order") or "xyz")
        wy = self.tilt.update(
            kwargs["f_ext"],
            kwargs["f_des"],
            dt_s=float(dt_s),
            contact=contact,
            pose=kwargs.get("pose"),
            slack_norm=kwargs.get("slack_norm"),
            euler_order=euler,
        )
        velocity = np.asarray(zout.v_force, dtype=float).reshape(6).copy()
        velocity[self.tilt.cfg.axis] = wy
        if self.tilt.needs_normal_retract and kwargs.get("pose") is not None:
            rotation = rotation_from_pose(kwargs["pose"], euler_order=euler)
            velocity = remap_force_along_world_normal(
                velocity,
                rotation=rotation,
                n_world=self.tilt.n_world,
                v_force_z=float(zout.v_force_z),
            )
        telemetry = dict(zout.telemetry or {})
        telemetry.update(self.tilt.telemetry())
        return ForceOutput(
            v_force=velocity,
            v_force_z=float(zout.v_force_z),
            contact_active=bool(zout.contact_active or self.tilt.engaged),
            f_des_z=float(zout.f_des_z),
            telemetry=telemetry,
        )
