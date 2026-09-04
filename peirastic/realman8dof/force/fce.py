"""fce787a9 implicit-Euler force law, one integrator per TFF force axis.

    M · v̇ + D · v = e_f ,   e_f = F* − F̂

F̂ is the 6D residual after a slow wrench-bias leak (full-direction
compensation of leftover gravity/payload). A C1 ball deadzone on force
and on torque (Kikuuwe implicit friction; dVRK clutch) kills sensor
walk. Inside both balls, v is clutched to 0. No Lee/BEFM, no xd_gain.
"""

from __future__ import annotations

import math

import numpy as np

from peirastic.realman8dof.force.config import apply_force_payload, load_force_raw
from peirastic.realman8dof.force.protocol import ForceOutput
from peirastic.realman8dof.force.tff import SELECTION_TOOL_Z_FORCE

# Hover / all-force clutch. Tiny C1 balls kill sensor noise only. Plant-ID
# leftover (~0.3 N / 0.02 Nm) is averaged into a 6D bias during a short
# freeze — not a fat deadzone. Leak uses a larger ball so a slow gravity
# change after a hand rotate can be eaten once v is still. D_rot stays
# light: 0.1 Nm → ~0.45 rad/s.
_HOVER_F_DEAD_N = 0.12
_HOVER_F_WIDTH_N = 0.08
_HOVER_T_DEAD_NM = 0.020
_HOVER_T_WIDTH_NM = 0.015
_HOVER_LEAK_F_N = 0.75
_HOVER_LEAK_T_NM = 0.10
_HOVER_SETTLE_S = 0.25
_HOVER_M_LIN = 1.0
_HOVER_M_ROT = 0.04
_HOVER_D_LIN = 18.0
_HOVER_D_ROT = 0.22
_HOVER_VMAX_LIN = 0.08
_HOVER_VMAX_ROT = 0.45
_HOVER_BIAS_TAU_S = 1.2
_HOVER_STILL_V_M_S = 0.008
_HOVER_STILL_W_RAD_S = 0.05

_ROT_INERTIA_FRAC = 0.04
_ROT_DEADBAND_FRAC = 0.05


def _smooth_deadband(f_err: float, deadband: float, width: float) -> float:
    if width <= 0.0:
        if abs(f_err) <= deadband:
            return 0.0
        return f_err - math.copysign(deadband, f_err)
    af = abs(f_err)
    if af <= deadband:
        return 0.0
    if af >= deadband + width:
        return f_err - math.copysign(deadband + 0.5 * width, f_err)
    t = (af - deadband) / width
    gain = t * t * (3.0 - 2.0 * t)
    return math.copysign(gain * (af - deadband), f_err)


def _radial_c1(vec: np.ndarray, radius: float, width: float) -> np.ndarray:
    """Keep direction; C1-shrink the magnitude (force or torque ball)."""

    v = np.asarray(vec, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros_like(v)
    mag = _smooth_deadband(n, float(radius), float(width))
    if mag <= 0.0:
        return np.zeros_like(v)
    return v * (mag / n)


class FceAdmittanceLaw:
    """Per-axis implicit Euler from fce787a9. TFF picks which axes run."""

    def __init__(
        self,
        dt: float,
        *,
        force_axes: np.ndarray,
        mass: np.ndarray,
        damping: np.ndarray,
        deadband: np.ndarray,
        deadband_width: np.ndarray,
        max_velocity: np.ndarray,
        force_dead_n: float = 0.0,
        force_width_n: float = 0.0,
        torque_dead_nm: float = 0.0,
        torque_width_nm: float = 0.0,
        bias_tau_s: float = 0.0,
        leak_force_n: float = 0.0,
        leak_torque_nm: float = 0.0,
        settle_s: float = 0.0,
    ) -> None:
        self.dt = float(dt)
        self.force_axes = np.clip(np.asarray(force_axes, dtype=float).reshape(6), 0.0, 1.0)
        self.mass = np.maximum(np.asarray(mass, dtype=float).reshape(6), 1e-3)
        self.damping = np.maximum(np.asarray(damping, dtype=float).reshape(6), 0.0)
        self.deadband = np.maximum(np.asarray(deadband, dtype=float).reshape(6), 0.0)
        self.deadband_width = np.maximum(
            np.asarray(deadband_width, dtype=float).reshape(6), 0.0
        )
        self.max_velocity = np.maximum(
            np.asarray(max_velocity, dtype=float).reshape(6), 0.0
        )
        self.force_dead_n = max(float(force_dead_n), 0.0)
        self.force_width_n = max(float(force_width_n), 0.0)
        self.torque_dead_nm = max(float(torque_dead_nm), 0.0)
        self.torque_width_nm = max(float(torque_width_nm), 0.0)
        self.bias_tau_s = max(float(bias_tau_s), 0.0)
        self.leak_force_n = max(float(leak_force_n), self.force_dead_n)
        self.leak_torque_nm = max(float(leak_torque_nm), self.torque_dead_nm)
        self.settle_s = max(float(settle_s), 0.0)
        self._v = np.zeros(6, dtype=float)
        self._bias = np.zeros(6, dtype=float)
        self._bias_acc = np.zeros(6, dtype=float)
        self._bias_n = 0
        self._settle_age = 0.0
        self._bias_seeded = False

    @classmethod
    def from_payload(cls, dt: float, payload: dict | None = None) -> "FceAdmittanceLaw":
        pay = dict(payload or {})
        raw = apply_force_payload(load_force_raw(), pay)
        hm = dict(raw.get("hybrid_motion") or {})
        if pay.get("force_axes") is not None:
            force_axes = np.asarray(pay["force_axes"], dtype=float).reshape(6)
        elif pay.get("selection") is not None:
            force_axes = 1.0 - np.asarray(pay["selection"], dtype=float).reshape(6)
        else:
            force_axes = 1.0 - np.asarray(SELECTION_TOOL_Z_FORCE, dtype=float)
        if float(np.min(force_axes)) > 0.5:
            return cls(
                dt,
                force_axes=force_axes,
                mass=np.array(
                    [_HOVER_M_LIN, _HOVER_M_LIN, _HOVER_M_LIN, _HOVER_M_ROT, _HOVER_M_ROT, _HOVER_M_ROT]
                ),
                damping=np.array(
                    [_HOVER_D_LIN, _HOVER_D_LIN, _HOVER_D_LIN, _HOVER_D_ROT, _HOVER_D_ROT, _HOVER_D_ROT]
                ),
                deadband=np.zeros(6),
                deadband_width=np.zeros(6),
                max_velocity=np.array(
                    [
                        _HOVER_VMAX_LIN,
                        _HOVER_VMAX_LIN,
                        _HOVER_VMAX_LIN,
                        _HOVER_VMAX_ROT,
                        _HOVER_VMAX_ROT,
                        _HOVER_VMAX_ROT,
                    ]
                ),
                force_dead_n=_HOVER_F_DEAD_N,
                force_width_n=_HOVER_F_WIDTH_N,
                torque_dead_nm=_HOVER_T_DEAD_NM,
                torque_width_nm=_HOVER_T_WIDTH_NM,
                bias_tau_s=_HOVER_BIAS_TAU_S,
                leak_force_n=_HOVER_LEAK_F_N,
                leak_torque_nm=_HOVER_LEAK_T_NM,
                settle_s=_HOVER_SETTLE_S,
            )
        mass_z = float(hm.get("admittance_mass_z", 1.0))
        damp_z = float(hm.get("admittance_damping_z", 25.0))
        dead_n = float(hm.get("deadband_n", 0.08))
        width_n = float(hm.get("deadband_width_n", 0.10))
        mass = np.full(6, mass_z, dtype=float)
        mass[3:6] = mass_z * _ROT_INERTIA_FRAC
        damp = np.full(6, damp_z, dtype=float)
        damp[3:6] = damp_z * _ROT_INERTIA_FRAC
        dead = np.full(6, dead_n, dtype=float)
        dead[3:6] = dead_n * _ROT_DEADBAND_FRAC
        width = np.full(6, width_n, dtype=float)
        width[3:6] = width_n * _ROT_DEADBAND_FRAC
        max_vel = np.asarray(
            hm.get("max_velocity", [0.22, 0.22, 0.08, 0.6, 0.6, 0.6]),
            dtype=float,
        ).reshape(6)
        return cls(
            dt,
            force_axes=force_axes,
            mass=mass,
            damping=damp,
            deadband=dead,
            deadband_width=width,
            max_velocity=max_vel,
        )

    def reset(self, *, pose: np.ndarray, f_ext: np.ndarray) -> None:
        del pose, f_ext
        self._v[:] = 0.0
        self._bias[:] = 0.0
        self._bias_acc[:] = 0.0
        self._bias_n = 0
        self._settle_age = 0.0
        self._bias_seeded = False

    def _leftover(self, wrench: np.ndarray) -> bool:
        return float(np.linalg.norm(wrench[:3])) <= self.leak_force_n + 1e-9 and float(
            np.linalg.norm(wrench[3:6])
        ) <= self.leak_torque_nm + 1e-9

    def _held_output(self, f_des: np.ndarray) -> ForceOutput:
        return ForceOutput(
            v_force=np.zeros(6, dtype=float),
            v_force_z=0.0,
            contact_active=True,
            f_des_z=float(f_des[2]),
        )

    def _try_settle(self, residual: np.ndarray, f_des: np.ndarray, dt_eff: float):
        if self._bias_seeded or self.settle_s <= 1e-9:
            if not self._bias_seeded:
                self._bias_seeded = True
            return None
        self._settle_age += dt_eff
        if self._leftover(residual):
            self._bias_acc += residual
            self._bias_n += 1
            self._bias = self._bias_acc / float(self._bias_n)
            if self._settle_age < self.settle_s:
                return self._held_output(f_des)
            self._bias_seeded = True
            return None
        if self._bias_n > 0:
            self._bias = self._bias_acc / float(self._bias_n)
        self._bias_seeded = True
        return None

    def _compensated(self, f_ext: np.ndarray, f_des: np.ndarray) -> np.ndarray:
        return f_ext - self._bias

    def _leak_bias(
        self,
        f_ext: np.ndarray,
        f_des: np.ndarray,
        f_comp: np.ndarray,
        dt_eff: float,
    ) -> None:
        if self.bias_tau_s <= 1e-6:
            return
        still = float(np.linalg.norm(self._v[:3])) <= _HOVER_STILL_V_M_S and float(
            np.linalg.norm(self._v[3:6])
        ) <= _HOVER_STILL_W_RAD_S
        if not still or not self._leftover(f_comp - f_des):
            return
        blend = min(1.0, dt_eff / self.bias_tau_s)
        self._bias += blend * ((f_ext - f_des) - self._bias)

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
        del pose, path_twist, contact, f_ext_raw
        del sensor_age_s, feedback_age_s, v_tcp_z_actual
        dt_eff = float(dt_actual) if dt_actual is not None else float(dt_s)
        if not math.isfinite(dt_eff) or dt_eff <= 0.0:
            dt_eff = self.dt
        f_ext = np.asarray(f_ext, dtype=float).reshape(6)
        f_des = np.asarray(f_des, dtype=float).reshape(6)
        held = self._try_settle(f_ext - f_des, f_des, dt_eff)
        if held is not None:
            return held
        f_comp = self._compensated(f_ext, f_des)
        drive = np.zeros(6, dtype=float)
        if self.force_dead_n > 0.0 or self.torque_dead_nm > 0.0:
            wrench_err = f_comp - f_des
            f_hat = _radial_c1(wrench_err[:3], self.force_dead_n, self.force_width_n)
            t_hat = _radial_c1(wrench_err[3:6], self.torque_dead_nm, self.torque_width_nm)
            clutched = float(np.linalg.norm(f_hat)) <= 1e-12 and float(
                np.linalg.norm(t_hat)
            ) <= 1e-12
            drive[:3] = -f_hat
            drive[3:6] = -t_hat
            if clutched:
                self._v[:] = 0.0
                self._leak_bias(f_ext, f_des, f_comp, dt_eff)
                return ForceOutput(
                    v_force=np.zeros(6, dtype=float),
                    v_force_z=0.0,
                    contact_active=True,
                    f_des_z=float(f_des[2]),
                )
        else:
            for i in range(6):
                drive[i] = _smooth_deadband(
                    float(f_des[i] - f_comp[i]),
                    float(self.deadband[i]),
                    float(self.deadband_width[i]),
                )
        self._leak_bias(f_ext, f_des, f_comp, dt_eff)
        v = np.zeros(6, dtype=float)
        for i in range(6):
            if self.force_axes[i] <= 0.5:
                self._v[i] = 0.0
                continue
            denom = self.mass[i] / dt_eff + self.damping[i]
            vel = ((self.mass[i] / dt_eff) * self._v[i] + drive[i]) / max(denom, 1e-6)
            cap = float(self.max_velocity[i])
            if cap > 0.0:
                vel = float(np.clip(vel, -cap, cap))
            self._v[i] = vel
            v[i] = vel
        return ForceOutput(
            v_force=v,
            v_force_z=float(v[2]),
            contact_active=True,
            f_des_z=float(f_des[2]),
        )


def use_fce_law(payload: dict | None) -> bool:
    law = str((payload or {}).get("law") or "").lower()
    return law in ("fce", "fce787a9")
