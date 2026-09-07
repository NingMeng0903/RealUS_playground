"""Constrained tool-ωy tilt: torque-guided surface conforming, not hover.

Hover drops M ⊥ F (point-force couple). Finite-area CoP is τ_y ≈ x Fz and
must be kept. Z stays on LegacyForceLaw. No torque integrator / DOB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from peirastic.realman8dof.force.fce import kikuuwe_step
from peirastic.realman8dof.force.protocol import ForceOutput
from peirastic.realman8dof.force.tff import SELECTION_TOOL_Z_FORCE

_C1_FRAC = 0.20


def _smoothstep(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = (float(x) - float(lo)) / (float(hi) - float(lo))
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _smooth_deadband(err: float, deadband: float, width: float) -> float:
    if width <= 0.0:
        if abs(err) <= deadband:
            return 0.0
        return err - float(np.copysign(deadband, err))
    z = max(abs(err) - deadband, 0.0)
    if z <= 0.0:
        return 0.0
    if z < width:
        mag = (z * z) / (2.0 * width)
    else:
        mag = z - 0.5 * width
    return float(np.copysign(mag, err))


def _slew_scalar(
    prev: float,
    target: float,
    *,
    a_max: float,
    j_max: float,
    dt: float,
    acc: float,
) -> tuple[float, float]:
    dt_eff = max(float(dt), 1e-9)
    a_lim = max(float(a_max), 0.0)
    j_lim = max(float(j_max), 0.0)
    if a_lim <= 1e-12:
        return float(target), 0.0
    dv = float(target) - float(prev)
    if j_lim <= 1e-12:
        step = a_lim * dt_eff
        if abs(dv) <= step:
            return float(target), 0.0
        out = float(prev) + float(np.copysign(step, dv))
        return out, (out - float(prev)) / dt_eff
    want = dv / dt_eff
    da = want - float(acc)
    j_step = j_lim * dt_eff
    if abs(da) > j_step:
        da = float(np.copysign(j_step, da))
    a_new = float(acc) + da
    if abs(a_new) > a_lim:
        a_new = float(np.copysign(a_lim, a_new))
    out = float(prev) + a_new * dt_eff
    if target >= prev:
        out = min(out, float(target))
    else:
        out = max(out, float(target))
    return out, (out - float(prev)) / dt_eff


@dataclass
class TorqueTiltConfig:
    enabled: bool = True
    axis: int = 4
    mass: float = 0.04
    damping: float = 0.30
    coulomb_nm: float = 0.025
    vmax_rad_s: float = 0.20
    a_max: float = 5.0
    j_max: float = 80.0
    theta_max_rad: float = 0.52
    contact_only: bool = True
    contact_n: float = 0.8
    engage_err_n: float = 0.25
    engage_fade_n: float = 2.0
    sign: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict | None) -> "TorqueTiltConfig":
        root = raw if isinstance(raw, dict) else {}
        hm = root.get("hybrid_motion", root)
        if not isinstance(hm, dict):
            hm = {}
        block = hm.get("torque_tilt", {})
        if not isinstance(block, dict):
            block = {}
        pc = hm.get("physical_contact", {})
        if not isinstance(pc, dict):
            pc = {}
        enter = float(pc.get("enter_n", cls.contact_n))
        return cls(
            enabled=bool(block.get("enabled", True)),
            axis=int(block.get("axis", 4)),
            mass=float(block.get("mass", 0.04)),
            damping=float(block.get("damping", 0.30)),
            coulomb_nm=float(block.get("coulomb_nm", 0.025)),
            vmax_rad_s=float(block.get("vmax_rad_s", 0.20)),
            a_max=float(block.get("a_max", 5.0)),
            j_max=float(block.get("j_max", 80.0)),
            theta_max_rad=float(block.get("theta_max_rad", 0.52)),
            contact_only=bool(block.get("contact_only", True)),
            contact_n=float(block.get("contact_n", enter)),
            engage_err_n=float(block.get("engage_err_n", 0.25)),
            engage_fade_n=float(block.get("engage_fade_n", 2.0)),
            sign=float(block.get("sign", 1.0)),
        )


def apply_tilt_selection(
    selection: np.ndarray | None,
    cfg: TorqueTiltConfig,
) -> np.ndarray:
    """Open the tilt axis on the default Z-only mask. Explicit S is unchanged."""

    if selection is not None:
        return np.asarray(selection, dtype=float).reshape(6)
    s = np.asarray(SELECTION_TOOL_Z_FORCE, dtype=float).copy()
    if cfg.enabled:
        axis = int(np.clip(cfg.axis, 0, 5))
        s[axis] = 0.0
    return s


class TorqueTilt:
    """Single-axis Kikuuwe admittance on τ_axis. Keeps the CoP moment."""

    def __init__(self, cfg: TorqueTiltConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else TorqueTiltConfig()
        self.reset()

    def reset(self) -> None:
        self._w = 0.0
        self._w_out = 0.0
        self._acc = 0.0
        self._theta = 0.0
        self._was_contact = False
        self._engaged = False
        self.tau_y = 0.0
        self.omega_y = 0.0
        self.theta_tilt = 0.0
        self.stuck = True
        self.engaged = False

    def telemetry(self) -> dict:
        return {
            "tau_y": float(self.tau_y),
            "omega_y": float(self.omega_y),
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
        axis = int(np.clip(cfg.axis, 0, 5))
        wrench = np.asarray(f_ext, dtype=float).reshape(6)
        des = np.asarray(f_des, dtype=float).reshape(6)
        tau = float(wrench[axis])
        self.tau_y = tau
        fz = float(wrench[2])
        f_star = float(des[2])
        in_contact = (
            bool(contact)
            if contact is not None
            else abs(fz) >= max(float(cfg.contact_n), 0.0)
        )
        if cfg.contact_only and not in_contact:
            self._was_contact = False
            return self._hold(dt_s, engaged=False)
        if not self._was_contact:
            # Fresh tool-ω budget from this contact, not world pitch.
            self._theta = 0.0
            self._w = 0.0
            self._w_out = 0.0
            self._acc = 0.0
        self._was_contact = True
        e_f = abs(fz - f_star)
        band = float(cfg.engage_err_n)
        fade = max(float(cfg.engage_fade_n), 0.0)
        if band <= 0.0:
            gain = 1.0
        elif e_f <= band:
            gain = 1.0
        elif fade <= 0.0:
            gain = 0.0
        else:
            gain = 1.0 - _smoothstep(e_f, band, band + fade)
        self._engaged = True
        self.engaged = True
        drive = -float(cfg.sign) * tau
        width = max(float(cfg.coulomb_nm) * _C1_FRAC, 0.0)
        psi = gain * _smooth_deadband(drive, 0.0, width)
        v, stuck = kikuuwe_step(
            np.array([self._w], dtype=float),
            np.array([psi], dtype=float),
            mass=float(cfg.mass),
            damping=float(cfg.damping),
            coulomb=float(cfg.coulomb_nm),
            dt=float(dt_s),
            vmax=float(cfg.vmax_rad_s),
        )
        w_tgt = float(v[0])
        lim = max(float(cfg.theta_max_rad), 0.0)
        if lim > 0.0 and abs(self._theta) >= lim - 1e-12 and w_tgt * self._theta > 0.0:
            w_tgt = 0.0
        w_out, self._acc = _slew_scalar(
            self._w_out,
            w_tgt,
            a_max=float(cfg.a_max),
            j_max=float(cfg.j_max),
            dt=float(dt_s),
            acc=self._acc,
        )
        self._w = w_out
        self._w_out = w_out
        self._theta += w_out * max(float(dt_s), 0.0)
        if lim > 0.0:
            self._theta = float(np.clip(self._theta, -lim, lim))
        self.omega_y = w_out
        self.theta_tilt = float(self._theta)
        self.stuck = bool(stuck) and abs(w_out) <= 1e-12
        return w_out

    def _hold(self, dt_s: float, *, engaged: bool) -> float:
        self._engaged = bool(engaged)
        self.engaged = bool(engaged)
        w_out, self._acc = _slew_scalar(
            self._w_out,
            0.0,
            a_max=float(self.cfg.a_max),
            j_max=float(self.cfg.j_max),
            dt=float(dt_s),
            acc=self._acc,
        )
        self._w = w_out
        self._w_out = w_out
        self.omega_y = w_out
        self.theta_tilt = float(self._theta)
        self.stuck = abs(w_out) <= 1e-12
        return w_out


class LegacyForceWithTilt:
    """Legacy tool-Z admittance plus one torque-tilt axis."""

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
            kwargs["f_ext"],
            kwargs["f_des"],
            dt_s=float(dt_s),
            contact=contact,
        )
        v = np.asarray(zout.v_force, dtype=float).reshape(6).copy()
        v[int(np.clip(self.tilt.cfg.axis, 0, 5))] = float(wy)
        tel = dict(zout.telemetry or {})
        tel.update(self.tilt.telemetry())
        return ForceOutput(
            v_force=v,
            v_force_z=float(zout.v_force_z),
            contact_active=bool(zout.contact_active or self.tilt.engaged),
            f_des_z=float(zout.f_des_z),
            telemetry=tel,
        )
