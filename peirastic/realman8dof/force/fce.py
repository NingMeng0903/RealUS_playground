"""fce787a9 implicit-Euler force law, one integrator per TFF force axis.

    M · v̇ + D · v = e_f ,   e_f = F* − F̂

Hover (all six force axes) adds: Poinsot r̂ + ω about the grasp, Kikuuwe
stick, stick-gated bias leak, τ-matched (M, D), accel/jerk limits, and a
Landi residual gate that raises (M, D) together. No Lee/BEFM, no xd_gain.
"""

from __future__ import annotations

import math

import numpy as np

from peirastic.realman8dof.force.config import apply_force_payload, load_force_raw
from peirastic.realman8dof.force.protocol import ForceOutput
from peirastic.realman8dof.force.tff import SELECTION_TOOL_Z_FORCE

# Hover: Kikuuwe Coulomb radii sit just above plant-ID leftover
# (~0.22 N RMS / 0.015 Nm). C¹ width is 0.2·F_c. Leak is stick-gated
# (Nadeau), not a velocity still-gate. τ = M/D = I/D_ω ≈ 60 ms.
_HOVER_F_COULOMB_N = 0.32
_HOVER_T_COULOMB_NM = 0.025
_HOVER_F_C1_FRAC = 0.20
_HOVER_T_C1_FRAC = 0.20
_HOVER_LEAK_F_N = 0.75
_HOVER_LEAK_T_NM = 0.10
_HOVER_SETTLE_S = 0.25
_HOVER_STILL_S = 0.30
_HOVER_RHAT_F_ON = 0.80
_HOVER_RHAT_F_OFF = 0.40
_HOVER_RHAT_LPF_S = 0.080
_HOVER_TOOL_R_M = 0.040
_HOVER_TOOL_Z_MIN = -0.020
_HOVER_TOOL_Z_MAX = 0.160
_HOVER_M_LIN = 1.5
_HOVER_M_ROT = 0.04
_HOVER_D_LIN = 25.0
_HOVER_D_ROT = 0.65
_HOVER_VMAX_LIN = 0.08
_HOVER_VMAX_ROT = 0.45
_HOVER_ALIN_M_S2 = 1.20
_HOVER_AROT_RAD_S2 = 5.0
_HOVER_JLIN_M_S3 = 40.0
_HOVER_JROT_RAD_S3 = 80.0
_HOVER_BIAS_TAU_S = 0.80
_HOVER_LANDI_EPS_N = 0.80
_HOVER_LANDI_HOLD_S = 0.080
_HOVER_LANDI_RELEASE_S = 0.70
_HOVER_LANDI_SCALE = 2.0
_HOVER_DECOUPLE_F_MIN = 0.25
_HOVER_DECOUPLE_F_FULL = 0.80

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


def _smoothstep(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = (float(x) - float(lo)) / (float(hi) - float(lo))
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _moment_not_from_force(
    force: np.ndarray,
    moment: np.ndarray,
    *,
    f_min: float,
    f_full: float,
) -> np.ndarray:
    """Keep the couple along F; drop M that a point force can explain (M ⊥ F)."""

    f = np.asarray(force, dtype=float).reshape(3)
    m = np.asarray(moment, dtype=float).reshape(3)
    fn = float(np.linalg.norm(f))
    if fn <= float(f_min) or fn < 1e-9:
        return m
    u = f / fn
    m_couple = u * float(np.dot(m, u))
    if fn >= float(f_full):
        return m_couple
    t = (fn - float(f_min)) / max(float(f_full) - float(f_min), 1e-9)
    gain = t * t * (3.0 - 2.0 * t)
    return (1.0 - gain) * m + gain * m_couple


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


def poinsot_axis(force: np.ndarray, moment: np.ndarray) -> tuple[np.ndarray, float]:
    """Closest point r0 and pitch h of the wrench axis (Bicchi / Poinsot)."""

    f = np.asarray(force, dtype=float).reshape(3)
    m = np.asarray(moment, dtype=float).reshape(3)
    fn2 = float(np.dot(f, f))
    if fn2 < 1e-12:
        return np.zeros(3, dtype=float), 0.0
    r0 = np.cross(f, m) / fn2
    h = float(np.dot(f, m)) / fn2
    return r0, h


def project_tool_cylinder(
    r: np.ndarray,
    *,
    radius: float = _HOVER_TOOL_R_M,
    z_min: float = _HOVER_TOOL_Z_MIN,
    z_max: float = _HOVER_TOOL_Z_MAX,
) -> np.ndarray:
    """Clamp a point onto the tool bounding cylinder (TCP z along the probe)."""

    out = np.asarray(r, dtype=float).reshape(3).copy()
    rho = float(math.hypot(out[0], out[1]))
    rad = max(float(radius), 0.0)
    if rho > rad and rho > 1e-12:
        scale = rad / rho
        out[0] *= scale
        out[1] *= scale
    out[2] = min(max(out[2], float(z_min)), float(z_max))
    return out


def kikuuwe_step(
    v: np.ndarray,
    drive: np.ndarray,
    *,
    mass: float,
    damping: float,
    coulomb: float,
    dt: float,
    vmax: float,
) -> tuple[np.ndarray, bool]:
    """Implicit-Euler mass-damper with Kikuuwe Coulomb stick (norm ball)."""

    v0 = np.asarray(v, dtype=float).reshape(-1)
    h = np.asarray(drive, dtype=float).reshape(-1)
    dt_eff = max(float(dt), 1e-9)
    m = max(float(mass), 1e-6)
    d = max(float(damping), 0.0)
    denom = m / dt_eff + d
    v_star = ((m / dt_eff) * v0 + h) / max(denom, 1e-9)
    thresh = max(float(coulomb), 0.0) / max(denom, 1e-9)
    n = float(np.linalg.norm(v_star))
    if n <= thresh + 1e-15:
        out = np.zeros_like(v_star)
        stuck = True
    else:
        out = v_star - (thresh / n) * v_star
        stuck = False
    cap = float(vmax)
    if cap > 0.0:
        speed = float(np.linalg.norm(out))
        if speed > cap:
            out = out * (cap / speed)
    return out, stuck


def _slew_vector(
    prev: np.ndarray,
    target: np.ndarray,
    *,
    a_max: float,
    j_max: float,
    dt: float,
    acc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accel/jerk limit a 3-vector command (Kikuuwe 2024)."""

    p = np.asarray(prev, dtype=float).reshape(3)
    t = np.asarray(target, dtype=float).reshape(3)
    a = np.asarray(acc, dtype=float).reshape(3)
    dt_eff = max(float(dt), 1e-9)
    a_lim = max(float(a_max), 0.0)
    j_lim = max(float(j_max), 0.0)
    if a_lim <= 1e-12:
        return t.copy(), np.zeros(3, dtype=float)
    dv = t - p
    if j_lim <= 1e-12:
        step = a_lim * dt_eff
        n = float(np.linalg.norm(dv))
        if n <= step:
            return t.copy(), np.zeros(3, dtype=float)
        out = p + dv * (step / n)
        return out, (out - p) / dt_eff
    want_acc = dv / dt_eff
    da = want_acc - a
    j_step = j_lim * dt_eff
    jn = float(np.linalg.norm(da))
    if jn > j_step:
        da = da * (j_step / jn)
    a_new = a + da
    an = float(np.linalg.norm(a_new))
    if an > a_lim:
        a_new = a_new * (a_lim / an)
    out = p + a_new * dt_eff
    return out, a_new


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
        decouple_f_min: float = 0.0,
        decouple_f_full: float = 0.0,
        hover: bool = False,
        force_coulomb_n: float = 0.0,
        torque_coulomb_nm: float = 0.0,
        still_s: float = 0.0,
        rhat_f_on: float = 0.0,
        rhat_f_off: float = 0.0,
        rhat_lpf_s: float = 0.0,
        a_lin: float = 0.0,
        a_rot: float = 0.0,
        j_lin: float = 0.0,
        j_rot: float = 0.0,
        landi_eps_n: float = 0.0,
        landi_hold_s: float = 0.0,
        landi_release_s: float = 0.0,
        landi_scale: float = 1.0,
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
        self.decouple_f_min = max(float(decouple_f_min), 0.0)
        self.decouple_f_full = max(float(decouple_f_full), self.decouple_f_min)
        self.hover = bool(hover)
        self.force_coulomb_n = max(float(force_coulomb_n), 0.0)
        self.torque_coulomb_nm = max(float(torque_coulomb_nm), 0.0)
        self.still_s = max(float(still_s), 0.0)
        self.rhat_f_on = max(float(rhat_f_on), 0.0)
        self.rhat_f_off = max(float(rhat_f_off), 0.0)
        if self.rhat_f_on + 1e-12 < self.rhat_f_off:
            self.rhat_f_on = self.rhat_f_off
        self.rhat_lpf_s = max(float(rhat_lpf_s), 0.0)
        self.a_lin = max(float(a_lin), 0.0)
        self.a_rot = max(float(a_rot), 0.0)
        self.j_lin = max(float(j_lin), 0.0)
        self.j_rot = max(float(j_rot), 0.0)
        self.landi_eps_n = max(float(landi_eps_n), 0.0)
        self.landi_hold_s = max(float(landi_hold_s), 0.0)
        self.landi_release_s = max(float(landi_release_s), 0.0)
        self.landi_scale = max(float(landi_scale), 1.0)
        self._v = np.zeros(6, dtype=float)
        self._v_tcp = np.zeros(6, dtype=float)
        self._acc_lin = np.zeros(3, dtype=float)
        self._acc_rot = np.zeros(3, dtype=float)
        self._bias = np.zeros(6, dtype=float)
        self._bias_acc = np.zeros(6, dtype=float)
        self._bias_n = 0
        self._settle_age = 0.0
        self._bias_seeded = False
        self._r_hat = np.zeros(3, dtype=float)
        self._r_hold = np.zeros(3, dtype=float)
        self._rhat_latched = False
        self._stuck_lin = True
        self._stuck_rot = True
        self._stuck_age = 0.0
        self._landi_age = 0.0
        self._landi_gain = 1.0

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
                force_dead_n=0.0,
                force_width_n=_HOVER_F_COULOMB_N * _HOVER_F_C1_FRAC,
                torque_dead_nm=0.0,
                torque_width_nm=_HOVER_T_COULOMB_NM * _HOVER_T_C1_FRAC,
                bias_tau_s=_HOVER_BIAS_TAU_S,
                leak_force_n=_HOVER_LEAK_F_N,
                leak_torque_nm=_HOVER_LEAK_T_NM,
                settle_s=_HOVER_SETTLE_S,
                decouple_f_min=_HOVER_DECOUPLE_F_MIN,
                decouple_f_full=_HOVER_DECOUPLE_F_FULL,
                hover=True,
                force_coulomb_n=_HOVER_F_COULOMB_N,
                torque_coulomb_nm=_HOVER_T_COULOMB_NM,
                still_s=_HOVER_STILL_S,
                rhat_f_on=_HOVER_RHAT_F_ON,
                rhat_f_off=_HOVER_RHAT_F_OFF,
                rhat_lpf_s=_HOVER_RHAT_LPF_S,
                a_lin=_HOVER_ALIN_M_S2,
                a_rot=_HOVER_AROT_RAD_S2,
                j_lin=_HOVER_JLIN_M_S3,
                j_rot=_HOVER_JROT_RAD_S3,
                landi_eps_n=_HOVER_LANDI_EPS_N,
                landi_hold_s=_HOVER_LANDI_HOLD_S,
                landi_release_s=_HOVER_LANDI_RELEASE_S,
                landi_scale=_HOVER_LANDI_SCALE,
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
        self._v_tcp[:] = 0.0
        self._acc_lin[:] = 0.0
        self._acc_rot[:] = 0.0
        self._bias[:] = 0.0
        self._bias_acc[:] = 0.0
        self._bias_n = 0
        self._settle_age = 0.0
        self._bias_seeded = False
        self._r_hat[:] = 0.0
        self._r_hold[:] = 0.0
        self._rhat_latched = False
        self._stuck_lin = True
        self._stuck_rot = True
        self._stuck_age = 0.0
        self._landi_age = 0.0
        self._landi_gain = 1.0

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
            telemetry={"r_hat": self._r_hat.copy(), "stuck": True},
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
        *,
        stuck: bool,
    ) -> None:
        if self.bias_tau_s <= 1e-6:
            return
        if stuck:
            self._stuck_age += dt_eff
        else:
            self._stuck_age = 0.0
            return
        if self._stuck_age + 1e-12 < self.still_s:
            return
        leftover = f_comp - f_des
        if not self._leftover(leftover):
            return
        thin_f = max(self.force_coulomb_n, self.force_dead_n, 1e-6)
        thin_t = max(self.torque_coulomb_nm, self.torque_dead_nm, 1e-6)
        if float(np.linalg.norm(leftover[:3])) > thin_f + 1e-9:
            return
        if float(np.linalg.norm(leftover[3:6])) > thin_t + 1e-9:
            return
        blend = min(1.0, dt_eff / self.bias_tau_s)
        self._bias += blend * ((f_ext - f_des) - self._bias)

    def _update_rhat(self, force: np.ndarray, moment: np.ndarray, dt_eff: float) -> np.ndarray:
        fn = float(np.linalg.norm(force))
        if fn >= self.rhat_f_on:
            self._rhat_latched = True
        elif fn <= self.rhat_f_off:
            self._rhat_latched = False
        if self._rhat_latched and fn > 1e-9:
            r0, _h = poinsot_axis(force, moment)
            r_inst = project_tool_cylinder(r0)
            self._r_hold = r_inst
            alpha = _smoothstep(fn, self.rhat_f_off, self.rhat_f_on)
            r_blend = alpha * r_inst + (1.0 - alpha) * self._r_hat
        else:
            r_blend = self._r_hold if self._rhat_latched else self._r_hat
        if self.rhat_lpf_s > 1e-9:
            blend = min(1.0, dt_eff / self.rhat_lpf_s)
            self._r_hat = (1.0 - blend) * self._r_hat + blend * r_blend
        else:
            self._r_hat = np.asarray(r_blend, dtype=float).reshape(3)
        return self._r_hat

    def _update_landi(self, wrench: np.ndarray, v_rhat: np.ndarray, dt_eff: float) -> float:
        if self.landi_eps_n <= 1e-9 or self.landi_scale <= 1.0 + 1e-12:
            self._landi_gain = 1.0
            return 1.0
        a = (np.asarray(v_rhat, dtype=float).reshape(6) - self._v) / max(dt_eff, 1e-9)
        pred = self.mass * a + self.damping * v_rhat
        # M a + D v = −F̂ when the law holds, so ψ = |F + Ma + Dv|.
        psi = float(np.linalg.norm(wrench[:3] + pred[:3]))
        if psi > self.landi_eps_n:
            self._landi_age += dt_eff
        else:
            self._landi_age = 0.0
        target = self.landi_scale if self._landi_age >= self.landi_hold_s else 1.0
        tau = 0.05 if target > self._landi_gain else max(self.landi_release_s, 0.05)
        blend = min(1.0, dt_eff / tau)
        self._landi_gain += blend * (target - self._landi_gain)
        return float(self._landi_gain)

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
        if not np.isfinite(f_ext).all() or not np.isfinite(f_des).all():
            self._v[:] = 0.0
            self._v_tcp[:] = 0.0
            return self._held_output(f_des if np.isfinite(f_des).all() else np.zeros(6))
        held = self._try_settle(f_ext - f_des, f_des, dt_eff)
        if held is not None:
            return held
        f_comp = self._compensated(f_ext, f_des)
        wrench_err = f_comp - f_des
        if self.hover:
            return self._update_hover(f_ext, f_des, f_comp, wrench_err, dt_eff)
        drive = np.zeros(6, dtype=float)
        if self.force_dead_n > 0.0 or self.torque_dead_nm > 0.0:
            if self.decouple_f_full > 1e-9:
                wrench_err[3:6] = _moment_not_from_force(
                    wrench_err[:3],
                    wrench_err[3:6],
                    f_min=self.decouple_f_min,
                    f_full=self.decouple_f_full,
                )
            f_hat = _radial_c1(wrench_err[:3], self.force_dead_n, self.force_width_n)
            t_hat = _radial_c1(wrench_err[3:6], self.torque_dead_nm, self.torque_width_nm)
            clutched = float(np.linalg.norm(f_hat)) <= 1e-12 and float(
                np.linalg.norm(t_hat)
            ) <= 1e-12
            drive[:3] = -f_hat
            drive[3:6] = -t_hat
            if clutched:
                self._v[:] = 0.0
                self._leak_bias(f_ext, f_des, f_comp, dt_eff, stuck=True)
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
        self._leak_bias(f_ext, f_des, f_comp, dt_eff, stuck=False)
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

    def _update_hover(
        self,
        f_ext: np.ndarray,
        f_des: np.ndarray,
        f_comp: np.ndarray,
        wrench_err: np.ndarray,
        dt_eff: float,
    ) -> ForceOutput:
        force = wrench_err[:3].copy()
        moment = wrench_err[3:6].copy()
        r_hat = self._update_rhat(force, moment, dt_eff)
        moment_r = moment - np.cross(r_hat, force)
        if self.decouple_f_full > 1e-9:
            moment_r = _moment_not_from_force(
                force,
                moment_r,
                f_min=self.decouple_f_min,
                f_full=self.decouple_f_full,
            )
        f_hat = _radial_c1(force, 0.0, self.force_width_n)
        t_hat = _radial_c1(moment_r, 0.0, self.torque_width_nm)
        drive_f = -f_hat
        drive_t = -t_hat
        gain = float(self._landi_gain)
        m_lin = float(self.mass[0]) * gain
        d_lin = float(self.damping[0]) * gain
        m_rot = float(self.mass[3]) * gain
        d_rot = float(self.damping[3]) * gain
        v_lin, stuck_f = kikuuwe_step(
            self._v[:3],
            drive_f,
            mass=m_lin,
            damping=d_lin,
            coulomb=self.force_coulomb_n,
            dt=dt_eff,
            vmax=float(self.max_velocity[0]),
        )
        v_rot, stuck_t = kikuuwe_step(
            self._v[3:6],
            drive_t,
            mass=m_rot,
            damping=d_rot,
            coulomb=self.torque_coulomb_nm,
            dt=dt_eff,
            vmax=float(self.max_velocity[3]),
        )
        self._stuck_lin = bool(stuck_f)
        self._stuck_rot = bool(stuck_t)
        v_rhat = np.zeros(6, dtype=float)
        v_rhat[:3] = v_lin
        v_rhat[3:6] = v_rot
        if not (stuck_f and stuck_t):
            self._update_landi(wrench_err, v_rhat, dt_eff)
        else:
            self._landi_age = 0.0
        v_tcp = v_rhat.copy()
        v_tcp[:3] = v_lin + np.cross(v_rot, -r_hat)
        if self.a_lin > 1e-12:
            v_tcp[:3], self._acc_lin = _slew_vector(
                self._v_tcp[:3],
                v_tcp[:3],
                a_max=self.a_lin,
                j_max=self.j_lin,
                dt=dt_eff,
                acc=self._acc_lin,
            )
        if self.a_rot > 1e-12:
            v_tcp[3:6], self._acc_rot = _slew_vector(
                self._v_tcp[3:6],
                v_tcp[3:6],
                a_max=self.a_rot,
                j_max=self.j_rot,
                dt=dt_eff,
                acc=self._acc_rot,
            )
        self._v = v_rhat
        self._v_tcp = v_tcp
        stuck = bool(stuck_f and stuck_t)
        if stuck:
            self._v[:] = 0.0
            self._v_tcp[:] = 0.0
            self._acc_lin[:] = 0.0
            self._acc_rot[:] = 0.0
            v_tcp[:] = 0.0
        self._leak_bias(f_ext, f_des, f_comp, dt_eff, stuck=stuck)
        return ForceOutput(
            v_force=v_tcp.copy(),
            v_force_z=float(v_tcp[2]),
            contact_active=True,
            f_des_z=float(f_des[2]),
            telemetry={
                "r_hat": r_hat.copy(),
                "stuck": stuck,
                "landi_gain": float(self._landi_gain),
            },
        )


def use_fce_law(payload: dict | None) -> bool:
    law = str((payload or {}).get("law") or "").lower()
    return law in ("fce", "fce787a9")
