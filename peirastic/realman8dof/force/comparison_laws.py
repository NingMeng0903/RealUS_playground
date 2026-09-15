"""Comparison outer laws: 1D AC, 2D AC (z + ωy), and TAFAC-z. No bounce stack.

TAFAC follows Li et al., Control Engineering Practice 168:106674 (2026),
Eq. (14) and the discrete relative-velocity form of Eq. (15). The HTML
index writes the last state as Δx[k-1]; the continuous law is delayed
relative velocity, so this implementation uses Δẋ. Do not claim a
byte-identical PDF reproduction until the PDF is checked.

τ is the force-loop period dt, not the ultrasound image delay.
"""
from __future__ import annotations

import numpy as np

from peirastic.realman8dof.force.protocol import ForceOutput

DEFAULT_MASS = 1.0
DEFAULT_DAMPING = 40.0
DEFAULT_VMAX_M_S = 0.012
DEFAULT_INERTIA_YY = 0.051
# 75% of the ICRA tilt plant: ω_ss = e / b and ω_max both drop.
MOMENT_TRACK_SCALE = 0.75
DEFAULT_DAMPING_YY = 0.22 / MOMENT_TRACK_SCALE
DEFAULT_VMAX_OMEGA_Y = 0.28 * MOMENT_TRACK_SCALE
DEFAULT_COULOMB_YY = 0.02
DEFAULT_CONTACT_ENTER_N = 0.8
DEFAULT_CONTACT_CONFIRM_S = 0.05
COMPARISON_OUTER_LAWS = ("admittance_1d", "ac2d", "tafac")


class ComparisonContactLatch:
    """Compression-positive contact flag for wait_for_contact.

    Comparison laws skip the compiled hybrid controller. The gate still
    needs outer.controller.contact_present; this matches s_scan's
    set_force_control(contact_enter_n=0.8, enter_confirm_s=0.05).
    """

    def __init__(
        self,
        *,
        enter_n: float = DEFAULT_CONTACT_ENTER_N,
        confirm_s: float = DEFAULT_CONTACT_CONFIRM_S,
    ) -> None:
        self.enter_n = _finite(enter_n, "contact_enter_n")
        self.confirm_s = _finite(confirm_s, "enter_confirm_s", default=0.0)
        if self.enter_n <= 0.0 or self.confirm_s < 0.0:
            raise ValueError("contact enter_n must be positive and confirm_s nonnegative")
        self.contact_present = False
        self.v_force_cmd_z = 0.0
        self._hold_s = 0.0

    def reset(self) -> None:
        self.contact_present = False
        self.v_force_cmd_z = 0.0
        self._hold_s = 0.0

    def observe(self, fz: float, dt_s: float) -> bool:
        if float(fz) >= self.enter_n:
            self._hold_s += max(float(dt_s), 0.0)
            if self._hold_s + 1e-12 >= self.confirm_s:
                self.contact_present = True
        else:
            self._hold_s = 0.0
            self.contact_present = False
        return bool(self.contact_present)


def _finite(value, name, default=None):
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        value = default
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _tool_z(path):
    return float(np.asarray(path, dtype=float).reshape(6)[2])


class ComparisonAdmittance1D:
    """m v̇ + b v = F_d − F_z on tool-z. Path z is discarded."""

    def __init__(
        self,
        *,
        mass: float = DEFAULT_MASS,
        damping: float = DEFAULT_DAMPING,
        vmax_m_s: float = DEFAULT_VMAX_M_S,
        dt: float = 0.005,
    ) -> None:
        self.mass = _finite(mass, "mass")
        self.damping = _finite(damping, "damping")
        if self.mass <= 0.0 or self.damping < 0.0:
            raise ValueError("mass must be positive and damping nonnegative")
        self.vmax_m_s = max(_finite(vmax_m_s, "vmax_m_s"), 1.0e-6)
        self.dt = _finite(dt, "dt")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self.v = 0.0
        self.last = {}
        self.controller = ComparisonContactLatch()

    def reset(self, *, pose: np.ndarray, f_ext: np.ndarray) -> None:
        del pose, f_ext
        self.v = 0.0
        self.last = {}
        self.controller.reset()

    def update(
        self,
        *,
        dt_s: float,
        pose: np.ndarray,
        f_ext: np.ndarray,
        f_des: np.ndarray,
        path_twist: np.ndarray,
        path_twist_raw: np.ndarray | None = None,
        contact: bool | None = None,
        **_kwargs,
    ) -> ForceOutput:
        del pose
        dt_use = _finite(dt_s if _kwargs.get("dt_actual") is None else _kwargs.get("dt_actual"), "dt")
        dt_use = max(dt_use, 1.0e-4)
        fz = float(np.asarray(f_ext, dtype=float).reshape(6)[2])
        fd = float(np.asarray(f_des, dtype=float).reshape(6)[2])
        e_f = fd - fz
        self.controller.observe(fz, dt_use)
        # Implicit Euler: (m/dt + b) v = m/dt v_prev + e_f
        denom = self.mass / dt_use + self.damping
        v = (self.mass / dt_use * self.v + e_f) / max(denom, 1.0e-9)
        v = float(np.clip(v, -self.vmax_m_s, self.vmax_m_s))
        self.v = v
        self.controller.v_force_cmd_z = v
        raw = path_twist if path_twist_raw is None else path_twist_raw
        v_in = _tool_z(raw)
        v_force = np.zeros(6, dtype=float)
        v_force[2] = v
        self.last = dict(
            law="admittance_1d",
            e_f=e_f,
            fz=fz,
            fd=fd,
            v_z=v,
            v_in=v_in,
            a_in=0.0,
            delta_v=v,
            my=float(np.asarray(f_ext, dtype=float).reshape(6)[4]),
            contact=None if contact is None else bool(contact),
            contact_present=bool(self.controller.contact_present),
            dt_s=dt_use,
        )
        return ForceOutput(
            v_force=v_force,
            v_force_z=v,
            contact_active=bool(contact) if contact is not None else bool(self.controller.contact_present),
            f_des_z=fd,
            telemetry=dict(self.last),
        )


class ComparisonAdmittance2D(ComparisonAdmittance1D):
    """Same z-law as 1D AC, plus My → ωy. Owns UltraPoC's two force axes."""

    def __init__(
        self,
        *,
        mass: float = DEFAULT_MASS,
        damping: float = DEFAULT_DAMPING,
        vmax_m_s: float = DEFAULT_VMAX_M_S,
        inertia_yy: float = DEFAULT_INERTIA_YY,
        damping_yy: float = DEFAULT_DAMPING_YY,
        vmax_omega_y: float = DEFAULT_VMAX_OMEGA_Y,
        coulomb_yy: float = DEFAULT_COULOMB_YY,
        dt: float = 0.005,
    ) -> None:
        super().__init__(mass=mass, damping=damping, vmax_m_s=vmax_m_s, dt=dt)
        self.inertia_yy = _finite(inertia_yy, "inertia_yy")
        self.damping_yy = _finite(damping_yy, "damping_yy")
        if self.inertia_yy <= 0.0 or self.damping_yy < 0.0:
            raise ValueError("inertia_yy must be positive and damping_yy nonnegative")
        self.vmax_omega_y = max(_finite(vmax_omega_y, "vmax_omega_y"), 1.0e-6)
        self.coulomb_yy = max(_finite(coulomb_yy, "coulomb_yy", default=0.0), 0.0)
        self.omega_y = 0.0

    def reset(self, *, pose: np.ndarray, f_ext: np.ndarray) -> None:
        super().reset(pose=pose, f_ext=f_ext)
        self.omega_y = 0.0

    def update(
        self,
        *,
        dt_s: float,
        pose: np.ndarray,
        f_ext: np.ndarray,
        f_des: np.ndarray,
        path_twist: np.ndarray,
        path_twist_raw: np.ndarray | None = None,
        contact: bool | None = None,
        **_kwargs,
    ) -> ForceOutput:
        out = super().update(
            dt_s=dt_s,
            pose=pose,
            f_ext=f_ext,
            f_des=f_des,
            path_twist=path_twist,
            path_twist_raw=path_twist_raw,
            contact=contact,
            **_kwargs,
        )
        dt_use = float(self.last["dt_s"])
        wrench = np.asarray(f_ext, dtype=float).reshape(6)
        desired = np.asarray(f_des, dtype=float).reshape(6)
        my = float(wrench[4])
        md = float(desired[4]) if desired.size >= 5 else 0.0
        e_raw = md - my
        if abs(e_raw) <= self.coulomb_yy:
            e_m = 0.0
        else:
            e_m = e_raw - float(np.copysign(self.coulomb_yy, e_raw))
        engaged = bool(out.contact_active)
        if engaged:
            denom = self.inertia_yy / dt_use + self.damping_yy
            omega = (self.inertia_yy / dt_use * self.omega_y + e_m) / max(denom, 1.0e-9)
            omega = float(np.clip(omega, -self.vmax_omega_y, self.vmax_omega_y))
        else:
            omega = 0.0
        self.omega_y = omega
        out.v_force[4] = omega
        self.last.update(
            law="ac2d",
            e_m=e_m,
            e_m_raw=e_raw,
            my=my,
            md=md,
            omega_y=omega,
            coulomb_yy=self.coulomb_yy,
            tilt_engaged=engaged,
        )
        out.telemetry = dict(self.last)
        return out


class TafacNormalLaw:
    """TAFAC on tool-z: v_z = v_in + Δv, with trajectory feedforward."""

    def __init__(
        self,
        *,
        mass: float = DEFAULT_MASS,
        damping: float = DEFAULT_DAMPING,
        vmax_m_s: float = DEFAULT_VMAX_M_S,
        dt: float = 0.005,
    ) -> None:
        self.mass = _finite(mass, "mass")
        self.damping = _finite(damping, "damping")
        if self.mass <= 0.0 or self.damping < 0.0:
            raise ValueError("mass must be positive and damping nonnegative")
        self.vmax_m_s = max(_finite(vmax_m_s, "vmax_m_s"), 1.0e-6)
        self.dt = _finite(dt, "dt")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        self._delta_v = 0.0
        self._e_f_prev = 0.0
        self._v_in_prev = None
        self.last = {}
        self.controller = ComparisonContactLatch()

    def reset(self, *, pose: np.ndarray, f_ext: np.ndarray) -> None:
        del pose, f_ext
        self._delta_v = 0.0
        self._e_f_prev = 0.0
        self._v_in_prev = None
        self.last = {}
        self.controller.reset()

    def update(
        self,
        *,
        dt_s: float,
        pose: np.ndarray,
        f_ext: np.ndarray,
        f_des: np.ndarray,
        path_twist: np.ndarray,
        path_twist_raw: np.ndarray | None = None,
        contact: bool | None = None,
        **_kwargs,
    ) -> ForceOutput:
        del pose
        dt_use = _finite(dt_s if _kwargs.get("dt_actual") is None else _kwargs.get("dt_actual"), "dt")
        dt_use = max(dt_use, 1.0e-4)
        raw = path_twist if path_twist_raw is None else path_twist_raw
        v_in = _tool_z(raw)
        if self._v_in_prev is None:
            a_in = 0.0
        else:
            a_in = (v_in - self._v_in_prev) / dt_use
        fz = float(np.asarray(f_ext, dtype=float).reshape(6)[2])
        fd = float(np.asarray(f_des, dtype=float).reshape(6)[2])
        e_f = fd - fz
        self.controller.observe(fz, dt_use)
        # Eq. (14) delayed relative velocity; τ = Ts = dt.
        # Δa[k] = (1/m)[Δf[k-1] − b v_in[k] − (m − b Ts) a_in[k] − b Δv[k-1]]
        delta_a = (1.0 / self.mass) * (
            self._e_f_prev
            - self.damping * v_in
            - (self.mass - self.damping * dt_use) * a_in
            - self.damping * self._delta_v
        )
        delta_v = self._delta_v + dt_use * delta_a
        v_z = v_in + delta_v
        v_z = float(np.clip(v_z, -self.vmax_m_s, self.vmax_m_s))
        delta_v = v_z - v_in
        self._delta_v = delta_v
        self.controller.v_force_cmd_z = v_z
        self._e_f_prev = e_f
        self._v_in_prev = v_in
        v_force = np.zeros(6, dtype=float)
        v_force[2] = v_z
        self.last = dict(
            law="tafac",
            e_f=e_f,
            fz=fz,
            fd=fd,
            v_z=v_z,
            v_in=v_in,
            a_in=a_in,
            delta_v=delta_v,
            delta_a=delta_a,
            tau_s=dt_use,
            my=float(np.asarray(f_ext, dtype=float).reshape(6)[4]),
            contact=None if contact is None else bool(contact),
            contact_present=bool(self.controller.contact_present),
            dt_s=dt_use,
        )
        return ForceOutput(
            v_force=v_force,
            v_force_z=v_z,
            contact_active=bool(contact) if contact is not None else bool(self.controller.contact_present),
            f_des_z=fd,
            telemetry=dict(self.last),
        )


def comparison_law_from_payload(dt: float, payload: dict | None):
    pay = dict(payload or {})
    name = str(pay.get("law") or "").lower()
    mass = pay.get("admittance_mass", DEFAULT_MASS)
    damping = pay.get("admittance_damping", DEFAULT_DAMPING)
    vmax = pay.get("max_vz_tool_m_s", DEFAULT_VMAX_M_S)
    enter_n = pay.get("contact_enter_n", DEFAULT_CONTACT_ENTER_N)
    confirm_s = pay.get("enter_confirm_s", DEFAULT_CONTACT_CONFIRM_S)
    if name == "admittance_1d":
        law = ComparisonAdmittance1D(mass=mass, damping=damping, vmax_m_s=vmax, dt=dt)
    elif name == "ac2d":
        law = ComparisonAdmittance2D(
            mass=mass,
            damping=damping,
            vmax_m_s=vmax,
            inertia_yy=pay.get("admittance_inertia_yy", DEFAULT_INERTIA_YY),
            damping_yy=pay.get("admittance_damping_yy", DEFAULT_DAMPING_YY),
            vmax_omega_y=pay.get("max_omega_y_rad_s", DEFAULT_VMAX_OMEGA_Y),
            coulomb_yy=pay.get("admittance_coulomb_yy", DEFAULT_COULOMB_YY),
            dt=dt,
        )
    elif name == "tafac":
        law = TafacNormalLaw(mass=mass, damping=damping, vmax_m_s=vmax, dt=dt)
    else:
        raise ValueError(f"unsupported comparison law {name!r}")
    law.controller = ComparisonContactLatch(enter_n=enter_n, confirm_s=confirm_s)
    return law


def wrap_comparison_phase(phase, payload):
    """Record AC/TAFAC internals (and optional unused US features) to jsonl."""

    pay = dict(payload or {})
    path = pay.get("comparison_log_path")
    if path is None:
        return phase
    from peirastic.realman8dof.modes.contact_recording import (
        ContactRecordSink,
        FeatureReceiver,
        clock_metadata,
    )

    sink = ContactRecordSink(path)
    features = None
    endpoint = pay.get("feature_endpoint")
    if endpoint:
        features = FeatureReceiver(endpoint)
    sink.emit(
        "study_start",
        law=str(pay.get("law")),
        mass=pay.get("admittance_mass", DEFAULT_MASS),
        damping=pay.get("admittance_damping", DEFAULT_DAMPING),
        inertia_yy=pay.get("admittance_inertia_yy"),
        damping_yy=pay.get("admittance_damping_yy"),
        coulomb_yy=pay.get("admittance_coulomb_yy"),
        **clock_metadata(),
    )
    baseline = phase.outer
    original_sample = baseline.sample
    control_id = 0
    previous_tick, previous_exit = phase.on_tick, phase.on_exit

    def sample(t_s, current_pose, f_ext, **kwargs):
        nonlocal control_id
        twist = original_sample(t_s, current_pose, f_ext, **kwargs)
        control_id += 1
        law = getattr(baseline, "force_law", None)
        state = dict(getattr(law, "last", None) or {})
        observation = None if features is None else features.observation
        feature_fields = None if observation is None else observation.to_dict()
        sink.emit(
            "control_sample",
            control_id=control_id,
            reference_s=t_s,
            pose_tcp_base=current_pose,
            control_wrench_tool=f_ext,
            feature=feature_fields,
            feature_error=None if features is None else features.error,
            **state,
        )
        return twist

    baseline.sample = sample

    def on_tick(t_ref, step, q_meas):
        if previous_tick is not None:
            previous_tick(t_ref, step, q_meas)

    def on_exit():
        try:
            sink.emit("stop", control_id=control_id, reason="phase_exit_or_replaced")
            if previous_exit is not None:
                previous_exit()
        finally:
            if features is not None:
                features.close(wait=False)
            sink.close(wait=False)

    phase.on_tick, phase.on_exit = on_tick, on_exit
    return phase

