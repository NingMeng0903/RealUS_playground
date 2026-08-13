"""One-dimensional bidirectional energy-flow adaptation.

This module is an engineering adaptation of the proxy/real-port structure
described by Lee et al. (2024).  It is deliberately *not* a torque theorem:
the normal axis is a speed-level interface and the energy account only
credits damping which is explicitly identified as nominal.  Unidentified
physical friction, Dimeas damping, and actuator losses are never credited.

The implementation keeps the two important safety properties of the
structure useful to the RM75 controller:

* the proxy may be bidirectional, while the real auxiliary path is one-sided
  and can only add retract velocity;
* an energy gate is applied to positive (press) velocity only.  Retract
  velocity passes through when the gate is closed, and stale or unverified
  feedback closes the press gate.

``BidirectionalFlowController.update`` is intentionally small and scalar so
it can be used by simulation tests as well as the 200 Hz controller.  A
``step`` alias is provided for callers that use the usual discrete-controller
terminology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


@dataclass
class BidirectionalFlowConfig:
    """Configuration for the scalar normal-axis flow adapter.

    Upper-case gain/tank names are retained because they match the notation
    used in the design note (``K_d``, ``T_0``).  Lower-case aliases are
    accepted for YAML and Python callers as well.  ``Ki`` is intentionally
    zero by default; enabling integral mismatch feedback is an explicit
    tuning choice rather than an accidental source of energy.
    """

    # ``off`` preserves the legacy controller, ``observe`` computes all
    # states/telemetry but returns the unmodulated proxy speed, and ``active``
    # enables the retract-through/press gate.
    mode: str = "off"
    sign_verified: bool = False
    feedback_delay_verified: bool = False
    require_sign_verification: bool = True
    require_delay_verification: bool = True
    normal_sign: float = 1.0

    # Proxy/real-port mismatch feedback.
    # Defaults follow the speed-level design note: Dtrack is also the
    # mismatch damping and Kp gives a 0.10 s mismatch time constant.
    Kd: float | None = None
    Kp: float | None = None
    Ki: float = 0.0
    lambda_gain: float = 0.25
    Dtrack: float = 20.0
    track_correction_max_m_s: float = 0.020
    # Lee Sec. V-C: alpha must be zero in free space.  Below this |F| the
    # modulation is held off and the tank charges from proxy damping instead.
    # 0 restores the pure power test (paper-faithful, noise-sensitive).
    free_space_force_n: float = 0.5
    M_p: float = 1.0
    D_p: float = 0.0
    m_p: float | None = None
    d_p: float | None = None

    # Optional lower-case spelling used by configuration loaders.
    kd: float | None = None
    kp: float | None = None
    ki: float | None = None
    lambda_: float | None = None
    d_track: float | None = None

    # Positive press path and one-sided real auxiliary path.
    gamma_active: float = 1.0
    aux_tau_s: float = 0.05
    aux_max_retract_m_s: float = 0.05
    press_epsilon_m_s: float = 1.0e-6
    # Independent one-sided real auxiliary mass/impedance.  x_safe follows
    # the real port only while the gate is open; when alpha rises it freezes
    # and the implicit M_a/D_a update can only generate retract velocity.
    M_a: float = 0.01
    D_a: float = 0.20
    K_a: float = 5.0
    B_a: float = 0.0
    u_retract_n: float = 0.0
    # Deprecated speed-form spelling; converted to force with D_a.
    u_retract_m_s: float | None = None
    u_retract: float | None = None
    m_a: float | None = None
    d_a: float | None = None
    k_a: float | None = None

    # Energy tank.  The small values are intentional: this is the scalar
    # speed-level tank used by the normal axis, not a robot-wide battery.
    T0: float = 0.001
    Tmax: float = 0.004
    Tmin: float = 0.0001
    t0: float | None = None
    t_max: float | None = None
    t_min: float | None = None
    nominal_damping: float = 0.0
    # Constant unmodelled active-power allowance (watts).  Keep zero by
    # default; active effort*press is accounted separately below.
    mu_power_w: float = 0.0
    mu: float | None = None
    active_press_debit_n: float = 1.0
    positive_switching_cost_j: float = 0.0
    switch_epsilon_m_s: float = 1.0e-6

    # Lee-style modulation smoothing.  A rise (closing the gate) is quick;
    # reopening is deliberately slower to avoid press chatter.
    alpha_attack_s: float = 0.02
    alpha_release_s: float = 0.15

    # A missing actual velocity or an old feedback sample is fail-closed.
    max_feedback_age_s: float = 0.02
    feedback_timeout_s: float | None = None

    # Optional labels/telemetry metadata.
    engineering_adaptation_label: str = (
        "engineering adaptation; not a torque theorem"
    )

    def __post_init__(self) -> None:
        if self.kd is not None:
            self.Kd = float(self.kd)
        if self.kp is not None:
            self.Kp = float(self.kp)
        if self.ki is not None:
            self.Ki = float(self.ki)
        if self.lambda_ is not None:
            self.lambda_gain = float(self.lambda_)
        if self.d_track is not None:
            self.Dtrack = float(self.d_track)
        if self.m_p is not None:
            self.M_p = float(self.m_p)
        if self.d_p is not None:
            self.D_p = float(self.d_p)
        if self.m_a is not None:
            self.M_a = float(self.m_a)
        if self.d_a is not None:
            self.D_a = float(self.d_a)
        if self.k_a is not None:
            self.K_a = float(self.k_a)
        if self.u_retract is not None:
            self.u_retract_n = float(self.u_retract)
        if self.u_retract_m_s is not None:
            self.u_retract_n = float(self.u_retract_m_s) * max(
                float(self.D_a), 0.0
            )
        if self.mu is not None:
            # Backward-compatible ``mu`` spelling now denotes watts, not a
            # multiplier on active press effort.
            self.mu_power_w = float(self.mu)
        if self.t0 is not None:
            self.T0 = float(self.t0)
        if self.t_max is not None:
            self.Tmax = float(self.t_max)
        if self.t_min is not None:
            self.Tmin = float(self.t_min)

        mode = str(self.mode).strip().lower().replace("-", "_")
        if mode in {"disabled", "none", "legacy", "0"}:
            mode = "off"
        elif mode in {"monitor", "logging", "1"}:
            mode = "observe"
        elif mode in {"enabled", "on", "2"}:
            mode = "active"
        if mode not in {"off", "observe", "active"}:
            raise ValueError(
                "bidirectional flow mode must be one of off/observe/active"
            )
        self.mode = mode

        if self.Kd is None:
            self.Kd = self.Dtrack
        if self.Kp is None:
            self.Kp = self.Dtrack / 0.10
        self.Kd = max(_finite(self.Kd), 0.0)
        self.Kp = max(_finite(self.Kp), 0.0)
        self.Ki = max(_finite(self.Ki), 0.0)
        self.lambda_gain = max(_finite(self.lambda_gain), 0.0)
        self.Dtrack = max(_finite(self.Dtrack), 1.0e-9)
        self.M_p = max(_finite(self.M_p, 1.0), 1.0e-6)
        self.D_p = max(_finite(self.D_p, 0.0), 0.0)
        self.gamma_active = float(np.clip(_finite(self.gamma_active, 1.0), 0.0, 1.0))
        self.aux_tau_s = max(_finite(self.aux_tau_s, 0.05), 0.0)
        self.aux_max_retract_m_s = max(_finite(self.aux_max_retract_m_s, 0.05), 0.0)
        self.press_epsilon_m_s = max(_finite(self.press_epsilon_m_s, 1e-6), 0.0)
        self.M_a = max(_finite(self.M_a, 0.01), 1.0e-6)
        self.D_a = max(_finite(self.D_a, 0.20), 0.0)
        self.K_a = max(_finite(self.K_a, 5.0), 0.0)
        self.B_a = max(_finite(self.B_a, 0.0), 0.0)
        # Signed auxiliary effort in the press-positive frame; resulting
        # velocity is still clamped non-positive below.
        self.u_retract_n = _finite(self.u_retract_n, 0.0)
        self.switch_epsilon_m_s = max(_finite(self.switch_epsilon_m_s, 1e-6), 0.0)

        # Enforce the stated tank ordering even when a hand-edited YAML file
        # contains a malformed value.  T0 is clamped into the usable range.
        self.Tmin = max(_finite(self.Tmin, 0.0001), 0.0)
        self.Tmax = max(_finite(self.Tmax, 0.004), self.Tmin)
        self.T0 = float(np.clip(_finite(self.T0, 0.001), self.Tmin, self.Tmax))
        self.nominal_damping = max(_finite(self.nominal_damping), 0.0)
        self.mu_power_w = max(_finite(self.mu_power_w, 0.0), 0.0)
        self.active_press_debit_n = max(_finite(self.active_press_debit_n, 1.0), 0.0)
        self.positive_switching_cost_j = max(
            _finite(self.positive_switching_cost_j, 0.00005), 0.0
        )
        self.alpha_attack_s = max(_finite(self.alpha_attack_s, 0.02), 0.0)
        self.alpha_release_s = max(_finite(self.alpha_release_s, 0.15), 0.0)
        self.max_feedback_age_s = max(_finite(self.max_feedback_age_s, 0.02), 0.0)
        if self.feedback_timeout_s is not None:
            self.max_feedback_age_s = max(
                _finite(self.feedback_timeout_s, self.max_feedback_age_s), 0.0
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "BidirectionalFlowConfig":
        """Read the flow section from either a controller or root mapping."""

        if raw is None:
            return cls()
        root = dict(raw)
        c = root.get("hybrid_motion", root.get("controller", root))
        if not isinstance(c, Mapping):
            c = root
        section: Mapping[str, Any] = {}
        for name in (
            "bidirectional_flow",
            "bidirectional",
            "energy_flow",
            "normal_axis_flow",
            "befm",
        ):
            value = c.get(name, root.get(name))
            if isinstance(value, Mapping):
                section = value
                break
        if not section:
            section = c

        def value(*names: str, default: Any = None) -> Any:
            return _first(section, *names, default=_first(c, *names, default=default))

        mode = value("mode", "bidirectional_flow_mode", default="off")
        dtrack_value = _finite(value("Dtrack", "d_track", default=20.0), 20.0)
        kd_value = value("Kd", "kd", "mismatch_damping", default=None)
        kp_value = value("Kp", "kp", "mismatch_stiffness", default=None)
        sign_verified = value(
            "sign_verified",
            "normal_sign_verified",
            "force_sign_verified",
            "sign_verification",
            default=False,
        )
        delay_verified = value(
            "feedback_delay_verified",
            "delay_verified",
            "velocity_delay_verified",
            default=False,
        )
        # A mapping is a convenient explicit verification record.  Requiring
        # both fields prevents ``sign_verification: {configured: true}`` from
        # accidentally enabling an active press path.
        if isinstance(sign_verified, Mapping):
            sign_verified = bool(
                sign_verified.get("verified", sign_verified.get("ok", False))
                and sign_verified.get("positive_is_press", True)
            )
        return cls(
            mode=str(mode),
            sign_verified=bool(sign_verified),
            feedback_delay_verified=bool(delay_verified),
            require_sign_verification=bool(
                value("require_sign_verification", default=True)
            ),
            require_delay_verification=bool(
                value("require_delay_verification", default=True)
            ),
            normal_sign=_finite(value("normal_sign", "press_sign", default=1.0), 1.0),
            Kd=(None if kd_value is None else _finite(kd_value, dtrack_value)),
            Kp=(None if kp_value is None else _finite(kp_value, dtrack_value / 0.10)),
            Ki=_finite(value("Ki", "ki", "mismatch_integral", default=0.0), 0.0),
            lambda_gain=_finite(
                value("lambda_gain", "lambda", "lambda_", default=0.25), 0.25
            ),
            Dtrack=dtrack_value,
            M_p=_finite(value("M_p", "m_p", "proxy_mass", default=1.0), 1.0),
            D_p=_finite(value("D_p", "d_p", "proxy_damping", default=0.0), 0.0),
            track_correction_max_m_s=_finite(
                value("track_correction_max_m_s", "v_track_max_m_s", default=0.020),
                0.020,
            ),
            free_space_force_n=_finite(
                value("free_space_force_n", "air_force_n", default=0.5), 0.5
            ),
            gamma_active=_finite(value("gamma_active", "gamma", default=1.0), 1.0),
            aux_tau_s=_finite(value("aux_tau_s", "auxiliary_tau_s", default=0.05), 0.05),
            aux_max_retract_m_s=_finite(
                value("aux_max_retract_m_s", "v_aux_max_retract_m_s", default=0.05),
                0.05,
            ),
            M_a=_finite(value("M_a", "m_a", "aux_mass", default=0.01), 0.01),
            D_a=_finite(value("D_a", "d_a", "aux_damping", default=0.20), 0.20),
            K_a=_finite(value("K_a", "k_a", "aux_stiffness", default=5.0), 5.0),
            B_a=_finite(value("B_a", "b_a", "aux_velocity_damping", default=0.0), 0.0),
            u_retract_n=_finite(
                value("u_retract_n", "retract_effort_n", default=0.0), 0.0
            ),
            u_retract_m_s=(
                None
                if value("u_retract_m_s", "retract_through_m_s", default=None)
                is None
                else _finite(
                    value("u_retract_m_s", "retract_through_m_s", default=0.0),
                    0.0,
                )
            ),
            T0=_finite(value("T0", "t0", "tank_t0", default=0.001), 0.001),
            Tmax=_finite(value("Tmax", "t_max", "tank_tmax", default=0.004), 0.004),
            Tmin=_finite(value("Tmin", "t_min", "tank_tmin", default=0.0001), 0.0001),
            nominal_damping=_finite(
                value("nominal_damping", "D0", "d0", default=0.0), 0.0
            ),
            mu_power_w=_finite(
                value("mu_power_w", "mu", "tank_mu", default=0.0), 0.0
            ),
            active_press_debit_n=_finite(
                value("active_press_debit_n", "press_debit_n", default=1.0), 1.0
            ),
            positive_switching_cost_j=_finite(
                value(
                    "positive_switching_cost_j",
                    "switching_cost_j",
                    "switch_cost_j",
                    default=0.0,
                ),
                0.0,
            ),
            alpha_attack_s=_finite(value("alpha_attack_s", "attack_s", default=0.02), 0.02),
            alpha_release_s=_finite(
                value("alpha_release_s", "release_s", default=0.15), 0.15
            ),
            max_feedback_age_s=_finite(
                value("max_feedback_age_s", "feedback_timeout_s", "stale_after_s", default=0.02),
                0.02,
            ),
            engineering_adaptation_label=str(
                value(
                    "engineering_adaptation_label",
                    default="engineering adaptation; not a torque theorem",
                )
            ),
        )


@dataclass
class BidirectionalFlowTelemetry:
    """Snapshot returned by the most recent update.

    Scalar fields intentionally have stable names suitable for CSV logging.
    The controller mirrors these onto itself for existing loggers that use
    ``getattr(controller, name)``.
    """

    xp: float = 0.0
    vp: float = 0.0
    xa: float = 0.0
    va: float = 0.0
    fc: float = 0.0
    v_track: float = 0.0
    v_aux: float = 0.0
    retract_through: float = 0.0
    press: float = 0.0
    command: float = 0.0
    alpha: float = 1.0
    alpha_raw: float = 1.0
    alpha_case: str = "init"
    tank_energy: float = 0.001
    tank_power_credit: float = 0.0
    tank_power_debit: float = 0.0
    psi: float = 0.0
    tank_switch_cost: float = 0.0
    Pe: float = 0.0
    Pc: float = 0.0
    P_phys: float = 0.0
    P_mismatch: float = 0.0
    energy_phys_j: float = 0.0
    energy_mismatch_j: float = 0.0
    Sn: float = 0.001
    Sr_hat: float = 0.001
    alpha_delta_energy_j: float = 0.0
    modulation_debit_j: float = 0.0
    feedback_age_s: float = float("nan")
    feedback_stale: bool = True
    sign_verified: bool = False
    sign_fault: bool = False
    mode: str = "off"
    active: bool = False
    blocked_reason: str = ""
    feedback_delay_verified: bool = False
    gamma_effective: float = 0.0
    engineering_adaptation: str = "engineering adaptation; not a torque theorem"

    @property
    def velocity_mismatch(self) -> float:
        return float(self.vp - self.va)

    @property
    def position_mismatch(self) -> float:
        return float(self.xp - self.xa)


class BidirectionalFlowController:
    """Stateful scalar proxy/real-port controller.

    ``vp_cmd`` is the legacy force-admittance speed (positive means press).
    The real port can be supplied as a measured velocity and position.  If a
    position is unavailable, the last position is integrated from ``va``;
    missing velocity feedback is nevertheless considered stale and therefore
    cannot open the active press gate.
    """

    # Enough proxy-velocity history to reach back one staleness budget even at
    # the fastest tick rate this loop runs at.
    _VP_HISTORY_MAX = 64

    ENGINEERING_ADAPTATION = "engineering adaptation; not a torque theorem"

    def __init__(
        self,
        dt: float,
        config: BidirectionalFlowConfig | None = None,
    ) -> None:
        self.dt = max(_finite(dt, 0.005), 1.0e-6)
        self.cfg = config or BidirectionalFlowConfig()
        self.reset()

    def reset(self, *, x_actual: float | None = None) -> None:
        x0 = _finite(x_actual, 0.0) if x_actual is not None else 0.0
        self.xp = x0
        self.xa = x0
        self.va = 0.0
        self.vp = 0.0
        self.fc = 0.0
        self.v_track = 0.0
        self.v_aux = 0.0
        self.aux_anchor = x0
        self.x_aux = x0
        self.x_safe = x0
        self.retract_through = 0.0
        self.press = 0.0
        self.command = 0.0
        self.alpha = 0.0
        self.alpha_raw = 0.0
        self.alpha_case = "init"
        self.tank_energy = float(self.cfg.T0)
        self.tank_power_credit = 0.0
        self.tank_power_debit = 0.0
        self.psi = 0.0
        self.tank_switch_cost = 0.0
        self.Pe = 0.0
        self.Pc = 0.0
        self.P_phys = 0.0
        self.P_mismatch = 0.0
        self.energy_phys_j = 0.0
        self.energy_mismatch_j = 0.0
        self.Sn = self.tank_energy
        self.Sr_hat = self.tank_energy
        self.alpha_delta_energy_j = 0.0
        self.modulation_debit_j = 0.0
        self.feedback_age_s = float("nan")
        self.feedback_stale = True
        self._vp_history: list[float] = []
        self.mismatch_velocity_aligned = 0.0
        self.alpha_would_gate_m_s = 0.0
        self.sign_verified = bool(self.cfg.sign_verified)
        self.sign_fault = bool(
            self.cfg.require_sign_verification and not self.cfg.sign_verified
        )
        self.feedback_delay_verified = bool(self.cfg.feedback_delay_verified)
        self.active = False
        self.blocked_reason = ""
        self.gamma_effective = 0.0
        self.integral_position_error = 0.0
        self.proxy_mass_now = float(self.cfg.M_p)
        self.proxy_damping_now = float(self.cfg.D_p)
        self.nominal_damping_now = float(self.cfg.nominal_damping)
        self._prev_press_request = 0.0
        self._accounted_press_m_s = 0.0
        self._active_effort_budget_n = 0.0
        self._accounting_dt_s = self.dt
        self._initialized = False
        self.last_dt_actual = self.dt
        self.telemetry = BidirectionalFlowTelemetry(
            tank_energy=self.tank_energy,
            mode=self.cfg.mode,
            sign_verified=self.sign_verified,
            engineering_adaptation=self.ENGINEERING_ADAPTATION,
        )
        self._mirror_telemetry()

    def begin_episode(
        self,
        v_actual: float,
        *,
        tank_energy: float | None = None,
        energy_phys_j: float | None = None,
        energy_mismatch_j: float | None = None,
    ) -> None:
        """Clear episode transients without adding energy to the tank."""

        previous_tank = (
            float(self.tank_energy) if tank_energy is None else float(tank_energy)
        )
        previous_phys = (
            float(self.energy_phys_j)
            if energy_phys_j is None
            else float(energy_phys_j)
        )
        previous_mismatch = (
            float(self.energy_mismatch_j)
            if energy_mismatch_j is None
            else float(energy_mismatch_j)
        )
        if not np.isfinite(previous_tank):
            raise ValueError("tank energy must be finite at episode entry")
        if previous_tank < float(self.cfg.Tmin) - 1.0e-12:
            raise RuntimeError("tank energy is below Tmin at episode entry")
        seed = _finite(v_actual, 0.0)
        self.reset(x_actual=0.0)
        # A changed upper bound may remove available energy; no phase boundary
        # is allowed to raise the stored energy toward Tmin or T0.
        self.tank_energy = min(previous_tank, float(self.cfg.Tmax))
        self.energy_phys_j = previous_phys
        self.energy_mismatch_j = previous_mismatch
        self.xp = 0.0
        self.xa = 0.0
        self.aux_anchor = 0.0
        self.x_aux = 0.0
        self.x_safe = 0.0
        self.va = seed
        self.vp = seed
        self.v_track = seed
        self.command = seed
        self.v_aux = 0.0
        self.retract_through = min(seed, 0.0)
        self.press = max(seed, 0.0)
        self._prev_press_request = self.press
        # Re-arm conservatively. Stale feedback and the normal tank gate still
        # decide whether positive press is allowed on the first live tick.
        self.alpha = 1.0
        self.alpha_raw = 1.0
        self.alpha_case = "episode_entry"
        self.gamma_effective = 0.0
        self.Sn = self.tank_energy
        self.Sr_hat = self.tank_energy
        self.telemetry = BidirectionalFlowTelemetry(
            tank_energy=self.tank_energy,
            energy_phys_j=self.energy_phys_j,
            energy_mismatch_j=self.energy_mismatch_j,
            mode=self.cfg.mode,
            sign_verified=bool(self.cfg.sign_verified),
            feedback_stale=True,
            alpha=1.0,
            alpha_raw=1.0,
            alpha_case="episode_entry",
            engineering_adaptation=self.ENGINEERING_ADAPTATION,
        )
        self._mirror_telemetry()

    @property
    def mode(self) -> str:
        return self.cfg.mode

    @property
    def active_enabled(self) -> bool:
        return self.cfg.mode == "active" and (
            (bool(self.cfg.sign_verified) or not self.cfg.require_sign_verification)
            and (
                bool(self.cfg.feedback_delay_verified)
                or not self.cfg.require_delay_verification
            )
        )

    def _feedback_is_stale(
        self,
        *,
        v_actual: float | None,
        feedback_age_s: float | None,
        feedback_fresh: bool | float | None,
    ) -> bool:
        try:
            velocity_finite = v_actual is not None and np.isfinite(float(v_actual))
        except (TypeError, ValueError):
            velocity_finite = False
        if not velocity_finite:
            return True
        if feedback_fresh is not None:
            try:
                fresh = (
                    bool(feedback_fresh)
                    if isinstance(feedback_fresh, (bool, np.bool_))
                    else float(feedback_fresh) > 0.5
                )
            except (TypeError, ValueError):
                fresh = False
            if not fresh:
                return True
        if feedback_age_s is None:
            return False
        age = _finite(feedback_age_s, float("inf"))
        return (not np.isfinite(age)) or age > self.cfg.max_feedback_age_s

    def _vp_delayed(self, age_s: float, dt: float) -> float:
        """Proxy velocity resampled back to when ``va`` was measured."""
        hist = self._vp_history
        if not hist:
            return float(self.vp)
        age = _finite(age_s, 0.0)
        if not np.isfinite(age) or age <= 0.0 or dt <= 0.0:
            return float(hist[-1])
        # Cap at the staleness budget: beyond it the sample is rejected as
        # stale anyway, and reaching further back would fabricate a match.
        age = min(float(age), float(self.cfg.max_feedback_age_s))
        steps = int(round(age / dt))
        if steps <= 0:
            return float(hist[-1])
        idx = max(0, len(hist) - 1 - steps)
        return float(hist[idx])

    def _lee_alpha_raw(
        self,
        *,
        Pe: float,
        Pc: float,
        dt: float,
        stale: bool,
    ) -> tuple[float, str]:
        """Return the exact Lee ``P_e/P_c`` gate cases.

        Here ``Pe = (vp-va) F_g`` is the press-positive power at the real port
        and ``Pc = (vp-va) Fc`` is the mismatch-controller power.  The
        positive-power cases are intentionally asymmetric:

        ``0 < lambda Pc < Pe`` -> ``alpha=1``;
        ``0 < Pe < lambda Pc`` -> ``alpha=Pe/(lambda Pc)``;
        ``Pe <= 0`` or ``Pc <= 0`` -> ``alpha=0``.

        Tank-low and stale feedback are hard fail-closed overrides and return
        exactly ``alpha=1``.  ``Pc``/``Pe`` are *not* replaced by damping or
        an arbitrary press debit in this branch.
        """

        if stale:
            return 1.0, "stale"
        if self.tank_energy <= self.cfg.Tmin + 1.0e-12:
            return 1.0, "tank_low"
        if Pe <= 0.0 or Pc <= 0.0:
            return 0.0, "nonpositive"
        lam_pc = max(self.cfg.lambda_gain * Pc, 0.0)
        if lam_pc <= 0.0:
            return 0.0, "nonpositive"
        if lam_pc < Pe:
            return 1.0, "Pe"
        return float(np.clip(Pe / lam_pc, 0.0, 1.0)), "Pc"

    def _smooth_alpha(self, target: float, dt: float, *, hard: bool) -> float:
        target = float(np.clip(target, 0.0, 1.0))
        if hard:
            self.alpha = 1.0
            return self.alpha
        tau = self.cfg.alpha_attack_s if target > self.alpha else self.cfg.alpha_release_s
        if tau <= 1.0e-12:
            self.alpha = target
        else:
            self.alpha += float(np.clip(dt / tau, 0.0, 1.0)) * (target - self.alpha)
        self.alpha = float(np.clip(self.alpha, 0.0, 1.0))
        return self.alpha

    def _mirror_telemetry(self) -> None:
        t = self.telemetry
        t.xp = float(self.xp)
        t.vp = float(self.vp)
        t.xa = float(self.xa)
        t.va = float(self.va)
        t.fc = float(self.fc)
        t.v_track = float(self.v_track)
        t.v_aux = float(self.v_aux)
        t.retract_through = float(self.retract_through)
        t.press = float(self.press)
        t.command = float(self.command)
        t.alpha = float(self.alpha)
        t.alpha_raw = float(self.alpha_raw)
        t.alpha_case = str(self.alpha_case)
        t.tank_energy = float(self.tank_energy)
        t.tank_power_credit = float(self.tank_power_credit)
        t.tank_power_debit = float(self.tank_power_debit)
        t.psi = float(self.psi)
        t.tank_switch_cost = float(self.tank_switch_cost)
        t.Pe = float(self.Pe)
        t.Pc = float(self.Pc)
        t.P_phys = float(self.P_phys)
        t.P_mismatch = float(self.P_mismatch)
        t.energy_phys_j = float(self.energy_phys_j)
        t.energy_mismatch_j = float(self.energy_mismatch_j)
        t.Sn = float(self.Sn)
        t.Sr_hat = float(self.Sr_hat)
        t.alpha_delta_energy_j = float(self.alpha_delta_energy_j)
        t.modulation_debit_j = float(self.modulation_debit_j)
        t.feedback_age_s = float(self.feedback_age_s)
        t.feedback_stale = bool(self.feedback_stale)
        t.sign_verified = bool(self.sign_verified)
        t.sign_fault = bool(self.sign_fault)
        t.feedback_delay_verified = bool(self.feedback_delay_verified)
        t.mode = self.cfg.mode
        t.active = bool(self.active)
        t.blocked_reason = str(self.blocked_reason)
        t.gamma_effective = float(getattr(self, "gamma_effective", 0.0))
        t.engineering_adaptation = self.ENGINEERING_ADAPTATION

        # Upper-case and descriptive aliases are useful for existing loggers
        # and make the real-port/mismatch telemetry self-documenting.
        self.Fc = float(self.fc)
        self.Kp_error = float(self.cfg.Kp * (self.xp - self.xa))
        self.real_port_position = float(self.xa)
        self.real_port_velocity = float(self.va)
        self.mismatch_position = float(self.xp - self.xa)
        self.mismatch_velocity = float(self.vp - self.va)
        self.e = float(self.mismatch_position)
        self.edot = float(self.mismatch_velocity)
        self.x_aux = float(self.aux_anchor)
        self.x_safe = float(self.x_safe)
        self.alpha_gate = float(self.alpha)
        self.T = float(self.tank_energy)
        self.psi_tank = float(self.psi)
        self.tank_T = float(self.tank_energy)
        self.feedback_fresh = not bool(self.feedback_stale)
        self.retract_through_velocity = float(self.retract_through)
        self.press_velocity = float(self.press)
        self.v_cmd = float(self.command)
        self.Pphys = float(self.P_phys)
        self.Pmismatch = float(self.P_mismatch)
        self.E_phys = float(self.energy_phys_j)
        self.E_mismatch = float(self.energy_mismatch_j)
        self.cumulative_energy_phys_j = float(self.energy_phys_j)
        self.cumulative_energy_mismatch_j = float(self.energy_mismatch_j)

    def update(
        self,
        vp_cmd: float = 0.0,
        x_actual: float | None = None,
        v_actual: float | None = None,
        force: float = 0.0,
        dt_actual: float | None = None,
        *,
        feedback_age_s: float | None = None,
        feedback_fresh: bool | float | None = None,
        feedback_freshness: bool | float | None = None,
        nominal_damping: float | None = None,
        proxy_mass: float | None = None,
        proxy_damping: float | None = None,
        active_effort_n: float | None = None,
        **kwargs: Any,
    ) -> float:
        """Advance one scalar flow tick and return the normal command.

        Keyword aliases (``v_proxy``, ``v_p``, ``xa``, ``va``, ``sensor_age_s``)
        are accepted to ease integration with older loop code.  Unknown
        keywords are ignored intentionally; the controller is often called
        from a telemetry-rich loop with additional fields.
        """

        vp_cmd = _finite(
            kwargs.pop("v_proxy", kwargs.pop("v_p", kwargs.pop("vp", vp_cmd))),
            0.0,
        )
        proxy_position_input = kwargs.pop(
            "xp", kwargs.pop("proxy_position", None)
        )
        if x_actual is None:
            x_actual = kwargs.pop("xa", kwargs.pop("actual_position", None))
        if v_actual is None:
            v_actual = kwargs.pop("va", kwargs.pop("actual_velocity", None))
        if feedback_age_s is None:
            feedback_age_s = kwargs.pop("sensor_age_s", kwargs.pop("feedback_age", None))
        if feedback_freshness is not None and feedback_fresh is None:
            feedback_fresh = feedback_freshness
        if feedback_fresh is None:
            feedback_fresh = kwargs.pop("fresh", kwargs.pop("is_fresh", None))
        if force == 0.0:
            force = kwargs.pop(
                "F_g",
                kwargs.pop(
                    "f_g",
                    kwargs.pop("generalized_force", kwargs.pop("f_ext", force)),
                ),
            )
        force = _finite(force, 0.0)
        if dt_actual is None:
            dt_actual = kwargs.pop("dt", kwargs.pop("dt_s", None))
        if proxy_mass is None:
            proxy_mass = kwargs.pop("Mp", kwargs.pop("m_p", None))
        if proxy_damping is None:
            proxy_damping = kwargs.pop("Dp", kwargs.pop("d_p", None))
        if active_effort_n is None:
            active_effort_n = kwargs.pop(
                "active_effort", kwargs.pop("F_active", None)
            )

        dt = self.dt if dt_actual is None else _finite(dt_actual, self.dt)
        dt = float(np.clip(dt, 1.0e-6, 0.25))
        self.last_dt_actual = dt
        self.feedback_age_s = (
            float("nan") if feedback_age_s is None else _finite(feedback_age_s, float("inf"))
        )
        stale = self._feedback_is_stale(
            v_actual=v_actual,
            feedback_age_s=feedback_age_s,
            feedback_fresh=feedback_fresh,
        )
        self.feedback_stale = bool(stale)
        va = _finite(v_actual, self.va)
        if x_actual is None or not np.isfinite(float(x_actual)):
            xa = self.xa + va * dt
        else:
            xa = _finite(x_actual, self.xa)
        self.xa = xa
        self.va = va

        if not self._initialized:
            self.xp = (
                self.xa
                if proxy_position_input is None
                else _finite(proxy_position_input, self.xa)
            )
            self.aux_anchor = self.xa
            self.x_safe = self.xa
            self._initialized = True
        elif proxy_position_input is not None:
            self.xp = _finite(proxy_position_input, self.xp)

        # Force/mismatch feedback.  ``xp`` is the current proxy position; the
        # resulting ``vp`` is then integrated with wall-clock dt.  Solving the
        # one-step equation in closed form makes the -lambda*alpha*Fc update
        # genuinely implicit rather than an explicit force kick.
        dx = self.xp - self.xa
        self.integral_position_error += dx * dt
        # The proxy coupling uses the *previous* gate value.  This is the
        # causal one-tick form of the implicit Lee update; using an
        # unconditional lambda would inject a press correction while the
        # current gate is closed.
        alpha_prev = float(self.alpha)
        gain = self.cfg.lambda_gain * alpha_prev
        mp = max(
            _finite(
                self.cfg.M_p if proxy_mass is None else proxy_mass,
                self.cfg.M_p,
            ),
            1.0e-6,
        )
        dp = max(
            _finite(
                self.cfg.D_p if proxy_damping is None else proxy_damping,
                self.cfg.D_p,
            ),
            0.0,
        )
        A = mp / dt + dp
        self.proxy_mass_now = mp
        self.proxy_damping_now = dp
        # Reconstruct the nominal implicit-Euler RHS.  In particular, alpha=0
        # is exactly vp_cmd; only the gated mismatch coupling contributes when
        # alpha_prev is nonzero.
        denom = A + gain * self.cfg.Kd
        self.vp = (
            A * vp_cmd
            + gain
            * (
                self.cfg.Kd * self.va
                - self.cfg.Kp * dx
                - self.cfg.Ki * self.integral_position_error
            )
        ) / max(denom, 1.0e-9)
        self.fc = (
            self.cfg.Kd * (self.vp - self.va)
            + self.cfg.Kp * dx
            + self.cfg.Ki * self.integral_position_error
        )
        self.xp += self.vp * dt

        correction = float(
            np.clip(self.fc / self.cfg.Dtrack, -self.cfg.track_correction_max_m_s, self.cfg.track_correction_max_m_s)
        )
        self.v_track = self.vp + correction
        self.retract_through = min(self.v_track, 0.0)
        self.press = max(self.v_track, 0.0)

        # Independent one-sided real auxiliary.  While the press gate is
        # effectively open, x_safe follows the measured real port.  Once the
        # gate closes it freezes, and the implicit mass/damping update can only
        # produce retract velocity from K_a(x_safe-xa)-D_a*va+u_retract.
        if alpha_prev <= 1.0e-6:
            self.x_safe = self.xa
        aux_force = (
            self.cfg.K_a * (self.x_safe - self.xa)
            - self.cfg.B_a * self.va
            + self.cfg.u_retract_n
        )
        aux_den = self.cfg.M_a / dt + self.cfg.D_a
        self.v_aux = (
            (self.cfg.M_a / dt) * self.v_aux + aux_force
        ) / max(aux_den, 1.0e-9)
        self.v_aux = float(
            np.clip(self.v_aux, -self.cfg.aux_max_retract_m_s, 0.0)
        )
        self.aux_anchor += self.v_aux * dt

        nominal_d = (
            self.cfg.nominal_damping
            if nominal_damping is None
            else max(_finite(nominal_damping), 0.0)
        )
        self.nominal_damping_now = float(nominal_d)
        # Press-positive generalized force/port powers.  These are kept
        # separate from the tank's known nominal-damping credit below.
        #
        # ``edot`` is Lee's e_nr_dot and must compare the two ports at the same
        # instant.  ``va`` arrives one CANFD round trip late (15-20 ms here,
        # against a 20 ms staleness budget) while ``vp`` is current, so the raw
        # difference is dominated by transport lag rather than by energy
        # generation — alpha would then be measuring the link, not the contact.
        self._vp_history.append(float(self.vp))
        if len(self._vp_history) > self._VP_HISTORY_MAX:
            del self._vp_history[: -self._VP_HISTORY_MAX]
        vp_aligned = self._vp_delayed(self.feedback_age_s, dt)
        edot = vp_aligned - self.va
        self.mismatch_velocity_aligned = float(edot)
        self.P_phys = force * self.va
        self.P_mismatch = force * edot
        self.Pe = self.P_mismatch
        self.Pc = edot * self.fc
        self.energy_phys_j += self.P_phys * dt
        self.energy_mismatch_j += self.P_mismatch * dt

        # A discrete positive switch cost is optional bookkeeping; it must
        # not replace the per-tick alpha-flow debit required by the tank.
        switch_cost = 0.0
        if (
            self.press > self.cfg.press_epsilon_m_s
            and self._prev_press_request <= self.cfg.switch_epsilon_m_s
        ):
            switch_cost = self.cfg.positive_switching_cost_j
        raw_alpha, alpha_case = self._lee_alpha_raw(
            Pe=self.Pe,
            Pc=self.Pc,
            dt=dt,
            stale=stale,
        )
        # Lee Sec. V-C: "when the robot is moving in free space, alpha should
        # always be zero because there is no energy generation in the nominal
        # system."  Structurally Pe=0 without contact, but the paper's own
        # free-space run (Fig. 13) still saw alpha lifted by F/T noise at 4 kHz
        # with collocated sensing; this link is slower and delayed, so make it
        # explicit rather than hoping the power test stays clean.
        free_n = max(float(getattr(self.cfg, "free_space_force_n", 0.0)), 0.0)
        in_free_space = free_n > 0.0 and abs(float(force)) < free_n
        if in_free_space and not stale:
            raw_alpha, alpha_case = 0.0, "free_space"
        self.alpha_raw = float(raw_alpha)
        self.alpha_case = alpha_case
        # A drained tank must not force alpha=1 in free space: there is no
        # energy generation to gate, and alpha=1 there is exactly the
        # performance loss the paper warns about.  Stale feedback still is a
        # hard gate — an unknown port is not a safe port.
        hard_gate = stale or (
            self.tank_energy <= self.cfg.Tmin + 1.0e-12 and not in_free_space
        )
        alpha_before_smoothing = float(self.alpha)
        self._smooth_alpha(raw_alpha, dt, hard=hard_gate)

        sign_ok = bool(self.cfg.sign_verified) or not self.cfg.require_sign_verification
        self.sign_verified = bool(sign_ok)
        self.sign_fault = bool(self.cfg.require_sign_verification and not sign_ok)
        delay_ok = bool(self.cfg.feedback_delay_verified) or not self.cfg.require_delay_verification
        self.blocked_reason = ""
        if self.cfg.mode == "active" and not sign_ok:
            self.blocked_reason = "sign_unverified"
        elif self.cfg.mode == "active" and not delay_ok:
            self.blocked_reason = "feedback_delay_unverified"
        elif self.cfg.mode == "active" and stale:
            self.blocked_reason = "feedback_stale"

        active_effort_budget = max(
            _finite(active_effort_n, self.cfg.active_press_debit_n)
            if active_effort_n is not None
            else self.cfg.active_press_debit_n,
            0.0,
        )
        # Ki is disabled in the first release.  If a later configuration opts
        # in, its positive mismatch effort is an active term and must buy tank
        # energy instead of appearing as free proxy work.
        active_effort_budget += max(
            self.cfg.Ki * self.integral_position_error,
            0.0,
        )
        self._active_effort_budget_n = float(active_effort_budget)
        self._accounting_dt_s = float(dt)
        # Compute conservative storage/credit terms before selecting the
        # positive command so gamma is budget-limited, not retroactively
        # clipped after an overdraw.
        # Lee's S_n and Ŝ_r are the *same* scaled inertia (M_n = λM̂), so the
        # α-interpolated storage S = (1-α)S_n + αŜ_r is a single physical
        # quantity.  Using M_p=1.0 for one and M_a=0.05 for the other made
        # Ŝ_r - S_n a 20x scale artefact, so every α rise booked a positive
        # modulation debit and drained the tank one way.
        self.Sn = float(0.5 * self.proxy_mass_now * self.vp * self.vp)
        self.Sr_hat = float(
            max(
                0.5 * self.proxy_mass_now * self.va * self.va
                + 0.5
                * self.cfg.K_a
                * (self.x_safe - self.xa)
                * (self.x_safe - self.xa),
                0.0,
            )
        )
        credit_j = (
            (1.0 - self.alpha) * self.nominal_damping_now * self.vp * self.vp
            + self.alpha * self.cfg.D_a * self.va * self.va
        ) * dt
        if in_free_space:
            # Free-space motion is pure proxy damping dissipation, which is a
            # credit term in Lee eq. (32).  Without it the tank only ever sits
            # or falls and arrives at contact already empty.
            air_d = max(float(self.proxy_damping_now), float(self.cfg.D_p), 0.0)
            credit_j += air_d * self.vp * self.vp * dt
        delta_alpha = self.alpha - alpha_before_smoothing
        self.alpha_delta_energy_j = delta_alpha * (self.Sr_hat - self.Sn)
        self.modulation_debit_j = max(self.alpha_delta_energy_j, 0.0)
        fixed_debit_j = (
            (1.0 - self.alpha) * self.cfg.mu_power_w * dt
            + self.modulation_debit_j
            + (
                switch_cost
                if self.cfg.mode == "active"
                and sign_ok
                and delay_ok
                and not stale
                else 0.0
            )
        )
        # Shadow of what the gate would remove if it were driving.  In observe
        # this is the only way to judge alpha before handing it the command:
        # it must sit at zero in free space and rise only at force peaks.
        self.alpha_would_gate_m_s = float(
            max(self.press, 0.0)
            * (1.0 - (1.0 - self.alpha) * self.cfg.gamma_active)
        )
        if self.cfg.mode == "active" and sign_ok and delay_ok and not stale:
            # Retract-through is never alpha-gated.  Only the positive branch
            # is modulated by the tank and active gain.
            self.active = True
            requested_gain = (1.0 - self.alpha) * self.cfg.gamma_active
            if self.tank_energy <= self.cfg.Tmin + 1.0e-12:
                requested_gain = 0.0
            # Pre-limit positive velocity by the energy available this tick;
            # tank clipping below is then only a numerical guard, not a way
            # to hide an already-overdrawn command.
            available_j = max(
                self.tank_energy
                - self.cfg.Tmin
                + credit_j
                - fixed_debit_j,
                0.0,
            )
            cost_per_speed_j = active_effort_budget * dt
            if self.press > self.cfg.press_epsilon_m_s and cost_per_speed_j > 0.0:
                budget_press = available_j / cost_per_speed_j
                requested_gain = min(
                    requested_gain,
                    float(np.clip(budget_press / self.press, 0.0, 1.0)),
                )
            self.gamma_effective = float(np.clip(requested_gain, 0.0, 1.0))
            gated_press = self.gamma_effective * self.press
            self.command = self.v_aux + self.retract_through + gated_press
        elif self.cfg.mode == "active" and sign_ok and delay_ok:
            # Stale feedback/tank-low remains active for retract-through only.
            self.active = True
            self.gamma_effective = 0.0
            self.command = self.v_aux + self.retract_through
        elif self.cfg.mode == "active":
            # An unverified sign must never fall back to the positive legacy
            # command.  Fail closed to the one-sided retract path.
            self.active = True
            self.gamma_effective = 0.0
            self.command = self.v_aux + self.retract_through
        else:
            self.active = False
            # Observe computes the full state but must not alter the legacy
            # command.  Off does the same and keeps its telemetry harmless.
            self.command = vp_cmd

        # Tank bookkeeping is done after the command is known.  ``Sn`` and
        # ``Sr_hat`` are storage estimates, not aliases for tank fill:
        # proxy kinetic storage versus conservative real-port auxiliary
        # storage.  Only identified damping channels are credited.
        sent_press = max(self.command, 0.0) if self.active else 0.0
        self._accounted_press_m_s = float(sent_press)
        effort = active_effort_budget
        active_power_w = effort * sent_press
        active_press_debit_j = active_power_w * dt
        # fixed_debit_j already contains the conservative constant-power,
        # storage-interpolation, and optional switching terms computed before
        # command selection.
        debit_j = fixed_debit_j + active_press_debit_j
        self.tank_switch_cost = switch_cost if sent_press > 0.0 else 0.0
        self.tank_power_credit = credit_j / dt if dt > 0.0 else 0.0
        self.tank_power_debit = debit_j / dt if dt > 0.0 else 0.0
        self.psi = self.tank_power_credit - self.tank_power_debit
        self.tank_energy = float(
            np.clip(self.tank_energy + credit_j - debit_j, self.cfg.Tmin, self.cfg.Tmax)
        )
        self._prev_press_request = self.press
        self._mirror_telemetry()
        return float(self.command)

    def settle_applied_press(self, applied_press_m_s: float) -> float:
        """Charge any positive speed added after the flow controller.

        The outer force-axis slew can retain more press than ``command`` when
        the flow gate closes.  That extra real command must buy energy from the
        same tank.  Retraction and reductions never receive a refund.
        """

        requested = max(_finite(applied_press_m_s, 0.0), 0.0)
        if not self.active:
            return requested
        extra = max(requested - float(self._accounted_press_m_s), 0.0)
        effort = max(float(self._active_effort_budget_n), 0.0)
        dt = max(float(self._accounting_dt_s), 1.0e-9)
        if extra <= 0.0 or effort <= 0.0:
            return requested
        available = max(float(self.tank_energy) - float(self.cfg.Tmin), 0.0)
        allowed_extra = min(extra, available / (effort * dt))
        debit = allowed_extra * effort * dt
        self.tank_energy = max(float(self.cfg.Tmin), float(self.tank_energy) - debit)
        self.tank_power_debit += debit / dt
        self.psi = self.tank_power_credit - self.tank_power_debit
        self._accounted_press_m_s += allowed_extra
        self._mirror_telemetry()
        return min(requested, float(self._accounted_press_m_s))

    # Common aliases used by small simulation harnesses.
    step = update
    compute = update
    compute_velocity = update
    compute_velocity_command = update


# Descriptive aliases used by a few standalone simulation harnesses.
BidirectionalFlowCore = BidirectionalFlowController
BidirectionalEnergyFlowController = BidirectionalFlowController


__all__ = [
    "BidirectionalFlowConfig",
    "BidirectionalFlowController",
    "BidirectionalFlowCore",
    "BidirectionalEnergyFlowController",
    "BidirectionalFlowTelemetry",
]
