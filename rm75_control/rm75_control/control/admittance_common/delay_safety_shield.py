"""Delay-aware normal-port safety/energy shield.

Nominal feel is ``u_nom`` (low-M/D admittance + optional CDYOB).  This
module only certifies a predicted force upper bound and a measured-port
energy lower bound.  It does not claim whole-robot passivity, zero
overshoot, or a theorem until the plant set, ``K_ub``, and the terminal
set have been validated on hardware.

Modes
-----
observe
    Run the same force certificate as ``force``; send ``u_nom``.
force
    Enforce ``F_ub <= F_max`` on the backup-to-terminal rollout.
passive
    force + ``E_lb >= eps`` with ``rho = 0``.
ospf
    force + ``E_lb >= eps`` with ``rho > 0`` (output-strict, normal port).

The first command is ``u(λ) = u_b + λ (u_nom − u_b)``; the remaining
``N_b − 1`` steps use the backup law.  A certified stop table covers
pure backup from the current state, so the candidate bound is
``D_ub(ξ, u(λ)) = Δx_1^ub(ξ, u(λ)) + D_b^ub(ξ_1)``, not
``max(model tube, D_b^ub(ξ))``.  Receding-horizon recursive feasibility
follows from shifting that backup tail, provided the next state stays
in the prediction tube.

Force indent uses the press-positive error ``ē_{x,+}``, not signed
position error.  Lookup state ``a0`` is ``[a_actual]_+``.  The stop
table also covers ``[u_prev]_+``, ``[a_cmd]_+``, and ``[q_front]_+``;
``q_remain`` alone is not a proven abstraction of the delay-line
permutation.  Finite-horizon ``ē_v(N)`` is not ``ē_v(∞)``.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field


def _clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def inf_minus_fv(
    f_lo: float,
    f_hi: float,
    v_lo: float,
    v_hi: float,
) -> float:
    """Lower bound of ``p = -F v`` on a rectangle (F is compression ≥ 0)."""
    f0 = max(0.0, float(f_lo))
    f1 = max(f0, float(f_hi))
    corners = (
        f0 * float(v_lo),
        f0 * float(v_hi),
        f1 * float(v_lo),
        f1 * float(v_hi),
    )
    return -max(corners)


def measured_power_lb(
    f_csv: float,
    v_csv: float,
    bar_f: float,
    bar_v: float,
) -> float:
    """Conservative ``p = F_e v = -F_csv v_csv`` with sensor bounds."""
    f_hat = -float(f_csv)
    v_hat = float(v_csv)
    return (
        f_hat * v_hat
        - abs(f_hat) * max(float(bar_v), 0.0)
        - abs(v_hat) * max(float(bar_f), 0.0)
        - max(float(bar_f), 0.0) * max(float(bar_v), 0.0)
    )


def default_velocity_error_ub(
    horizon_steps: int,
    ev0_m_s: float = 0.003,
    slope_m_s: float = 0.0004,
    ev_cap_m_s: float = 0.008,
) -> list[float]:
    n = max(int(horizon_steps), 1)
    return [
        float(min(ev0_m_s + slope_m_s * i, ev_cap_m_s))
        for i in range(1, n + 1)
    ]


def default_position_error_ub(
    velocity_error_ub_m_s: list[float],
    dt_s: float,
) -> list[float]:
    acc = 0.0
    out: list[float] = []
    ts = max(float(dt_s), 0.0)
    for ev in velocity_error_ub_m_s:
        acc += ts * max(float(ev), 0.0)
        out.append(acc)
    return out


@dataclass
class StopDxBin:
    """One monotonic covering sample of remaining press-positive indent.

    ``a0_m_s2`` is ``[a_actual]_+``.  ``q_remain_m`` is the delay-line
    remaining press ``dt * Σ [u_j]_+``, not ``max(queue)``.  ``u_prev``,
    ``a_cmd``, and ``q_front`` are the jerk-backup / next-applied
    command.  Missing fields parse as 0 and fail-close on a nonzero
    query.  Same ``(v,a,q_remain)`` with a different queue order is
    still not proven equivalent.
    """

    v0_m_s: float
    a0_m_s2: float = 0.0
    q_press_m_s: float = 0.0
    q_remain_m: float = 0.0
    u_prev_m_s: float = 0.0
    a_cmd_m_s2: float = 0.0
    q_front_m_s: float = 0.0
    dx_ub_m: float = 0.0
    n_b: int = 0


def _parse_stop_dx_bins(raw: object) -> list[StopDxBin]:
    if not isinstance(raw, list):
        return []
    out: list[StopDxBin] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q_press = max(float(item.get("q_press_m_s", 0.0)), 0.0)
        if "q_remain_m" in item:
            q_remain = max(float(item.get("q_remain_m", 0.0)), 0.0)
        else:
            # Old tables stored max(queue) in q_press_m_s (m/s).  Do not
            # treat that velocity as remaining meters.
            q_remain = 0.0
        out.append(
            StopDxBin(
                v0_m_s=abs(float(item.get("v0_m_s", 0.0))),
                a0_m_s2=max(float(item.get("a0_m_s2", 0.0)), 0.0),
                q_press_m_s=q_press,
                u_prev_m_s=max(float(item.get("u_prev_m_s", 0.0)), 0.0),
                a_cmd_m_s2=max(float(item.get("a_cmd_m_s2", 0.0)), 0.0),
                q_front_m_s=max(float(item.get("q_front_m_s", 0.0)), 0.0),
                q_remain_m=q_remain,
                dx_ub_m=max(float(item.get("dx_ub_m", 0.0)), 0.0),
                n_b=max(int(item.get("n_b", 0)), 0),
            )
        )
    return out


@dataclass
class SafetyShieldConfig:
    enabled: bool = True
    # observe | force | passive | ospf
    mode: str = "observe"
    t0_s: float = 0.005
    tp_s: float = 0.060
    horizon_steps: int = 40
    k_ub_n_m: float = 8000.0
    r_f_n_s: float = 8.0
    r_f_window_steps: int = 20
    f_release_n: float = 0.70
    v_hold_m_s: float = 0.015
    a_hold_m_s2: float = 0.15
    u_retract_m_s: float = 0.040
    a_max_m_s2: float = 1.20
    j_max_m_s3: float = 40.0
    queue_clear_m_s: float = 0.015
    e0_j: float = 0.004
    eps_j: float = 0.0005
    rho: float = 0.0
    bar_f_n: float = 0.15
    bar_v_m_s: float = 0.004
    l_p_w_s: float = 40.0
    e_x_m: float = 0.0004
    e_f_n: float = 0.20
    solver_budget_us: float = 2000.0
    fail_safe_on_solver_timeout: bool | None = None
    lambda_tol: float = 0.01
    recovery_hold_s: float = 0.050
    # False until a phantom-validated release model exists.  F_ub never
    # decreases, so requiring F_ub <= F_release at mid-contact empties T.
    require_contact_free_terminal: bool = False
    enforce_terminal: bool = True
    velocity_error_ub_m_s: list[float] = field(default_factory=list)
    position_error_ub_m: list[float] = field(default_factory=list)
    # Press-positive indent error.  Signed position_error_ub_m is not this.
    position_error_ub_plus_m: list[float] = field(default_factory=list)
    acceleration_error_ub_m_s2: list[float] = field(default_factory=list)
    # ē_v(∞).  None = unknown; a finite table's last entry is not this.
    velocity_error_persistent_m_s: float | None = None
    # Initial gap g_0 (m).  Not extra contact force.  Need g_0 >= D_T^ub.
    x_detach_m: float = 0.0
    # Empty / certified=false: plant-ID only.  First contact still uses T_stop.
    stop_dx_certified: bool = False
    stop_dx_source: str = ""
    stop_dx_bins: list[StopDxBin] = field(default_factory=list)
    # Writing velocity_error_persistent_m_s=0 is not a proof.  These
    # stay false until a hold property / infinite-horizon e_x / stable
    # closed-loop argument and a measured energy-sign check exist.
    terminal_invariance_proven: bool = False
    energy_sign_verified: bool = False
    # Runtime certificate domain.  Undeclared pose/payload or an empty
    # stop table makes domain_ok false; that is the certificate meaning.
    max_feedback_age_s: float = 0.015
    v_domain_m_s: float = 0.0
    a_domain_m_s2: float = 0.0
    u_domain_m_s: float = 0.0
    pose_domain_declared: bool = False
    payload_domain_declared: bool = False
    pose_min: list[float] = field(default_factory=list)
    pose_max: list[float] = field(default_factory=list)
    payload_min_kg: float | None = None
    payload_max_kg: float | None = None
    payload_kg: float | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "SafetyShieldConfig":
        root = raw if isinstance(raw, dict) else {}
        controller = root.get("hybrid_motion", root.get("controller", root))
        if not isinstance(controller, dict):
            controller = root if isinstance(root, dict) else {}
        section = controller.get("safety_shield", root.get("safety_shield", {}))
        if not isinstance(section, dict):
            section = {}
        plant = section.get("plant", {})
        if not isinstance(plant, dict):
            plant = {}
        ev = section.get("velocity_error_ub_m_s", plant.get("velocity_error_ub_m_s", []))
        ex = section.get("position_error_ub_m", plant.get("position_error_ub_m", []))
        ex_plus = section.get(
            "position_error_ub_plus_m", plant.get("position_error_ub_plus_m", [])
        )
        if not isinstance(ev, list):
            ev = []
        if not isinstance(ex, list):
            ex = []
        if not isinstance(ex_plus, list):
            ex_plus = []
        ea = section.get(
            "acceleration_error_ub_m_s2",
            plant.get("acceleration_error_ub_m_s2", []),
        )
        if isinstance(ea, (int, float)):
            ea = [float(ea)]
        if not isinstance(ea, list):
            ea = []
        stop_sec = section.get("stop_dx_ub", {})
        if not isinstance(stop_sec, dict):
            stop_sec = {}
        pose_min = section.get("pose_min", [])
        pose_max = section.get("pose_max", [])
        if not isinstance(pose_min, list):
            pose_min = []
        if not isinstance(pose_max, list):
            pose_max = []
        timeout_default = None
        if "fail_safe_on_solver_timeout" in section:
            timeout_default = bool(section.get("fail_safe_on_solver_timeout"))
        return cls(
            enabled=bool(section.get("enabled", True)),
            mode=str(section.get("mode", "observe")).strip().lower(),
            t0_s=float(plant.get("t0_s", section.get("t0_s", 0.005))),
            tp_s=float(plant.get("tp_s", section.get("tp_s", 0.060))),
            horizon_steps=int(
                plant.get("horizon_steps", section.get("horizon_steps", 40))
            ),
            k_ub_n_m=float(section.get("k_ub_n_m", 8000.0)),
            r_f_n_s=float(section.get("r_f_n_s", 8.0)),
            r_f_window_steps=int(section.get("r_f_window_steps", 20)),
            f_release_n=float(section.get("f_release_n", 0.70)),
            v_hold_m_s=float(section.get("v_hold_m_s", 0.015)),
            a_hold_m_s2=float(section.get("a_hold_m_s2", 0.15)),
            u_retract_m_s=float(section.get("u_retract_m_s", 0.040)),
            a_max_m_s2=float(section.get("a_max_m_s2", 1.20)),
            j_max_m_s3=float(section.get("j_max_m_s3", 40.0)),
            queue_clear_m_s=float(section.get("queue_clear_m_s", 0.015)),
            e0_j=float(section.get("e0_j", 0.004)),
            eps_j=float(section.get("eps_j", 0.0005)),
            rho=float(section.get("rho", 0.0)),
            bar_f_n=float(section.get("bar_f_n", 0.15)),
            bar_v_m_s=float(section.get("bar_v_m_s", 0.004)),
            l_p_w_s=float(section.get("l_p_w_s", 40.0)),
            e_x_m=float(section.get("e_x_m", 0.0004)),
            e_f_n=float(section.get("e_f_n", 0.20)),
            solver_budget_us=float(section.get("solver_budget_us", 2000.0)),
            fail_safe_on_solver_timeout=timeout_default,
            lambda_tol=float(section.get("lambda_tol", 0.01)),
            recovery_hold_s=float(section.get("recovery_hold_s", 0.050)),
            require_contact_free_terminal=bool(
                section.get("require_contact_free_terminal", False)
            ),
            enforce_terminal=bool(section.get("enforce_terminal", True)),
            velocity_error_ub_m_s=[float(x) for x in ev],
            position_error_ub_m=[float(x) for x in ex],
            position_error_ub_plus_m=[float(x) for x in ex_plus],
            acceleration_error_ub_m_s2=[float(x) for x in ea],
            velocity_error_persistent_m_s=(
                None
                if section.get(
                    "velocity_error_persistent_m_s",
                    plant.get("velocity_error_persistent_m_s", None),
                )
                is None
                else float(
                    section.get(
                        "velocity_error_persistent_m_s",
                        plant.get("velocity_error_persistent_m_s"),
                    )
                )
            ),
            x_detach_m=max(float(section.get("x_detach_m", 0.0)), 0.0),
            stop_dx_certified=bool(stop_sec.get("certified", False)),
            stop_dx_source=str(stop_sec.get("source", "")),
            stop_dx_bins=_parse_stop_dx_bins(stop_sec.get("bins", [])),
            terminal_invariance_proven=bool(
                section.get("terminal_invariance_proven", False)
            ),
            energy_sign_verified=bool(section.get("energy_sign_verified", False)),
            max_feedback_age_s=float(section.get("max_feedback_age_s", 0.015)),
            v_domain_m_s=float(section.get("v_domain_m_s", 0.0)),
            a_domain_m_s2=float(section.get("a_domain_m_s2", 0.0)),
            u_domain_m_s=float(section.get("u_domain_m_s", 0.0)),
            pose_domain_declared=bool(section.get("pose_domain_declared", False)),
            payload_domain_declared=bool(
                section.get("payload_domain_declared", False)
            ),
            pose_min=[float(x) for x in pose_min],
            pose_max=[float(x) for x in pose_max],
            payload_min_kg=(
                None
                if section.get("payload_min_kg") is None
                else float(section.get("payload_min_kg"))
            ),
            payload_max_kg=(
                None
                if section.get("payload_max_kg") is None
                else float(section.get("payload_max_kg"))
            ),
            payload_kg=(
                None
                if section.get("payload_kg") is None
                else float(section.get("payload_kg"))
            ),
        )

    def normalized_mode(self) -> str:
        mode = str(self.mode).strip().lower()
        if mode == "energy":
            return "passive"
        if mode not in ("observe", "force", "passive", "ospf", "off"):
            return "observe"
        return mode

    def energy_constrained(self) -> bool:
        return self.normalized_mode() in ("passive", "ospf")

    def force_constrained(self) -> bool:
        return self.normalized_mode() in ("force", "passive", "ospf")

    def diagnoses_force(self) -> bool:
        """Observe uses the same force certificate as force; it just does not apply."""
        return self.normalized_mode() in ("observe", "force", "passive", "ospf")

    def applies_command(self) -> bool:
        return self.normalized_mode() in ("force", "passive", "ospf")

    def should_enforce_terminal(self) -> bool:
        # Without a validated release model, force/observe are empirical
        # peak guards.  Terminal membership is required only for energy modes
        # or an explicit contact-free terminal.
        if self.require_contact_free_terminal:
            return True
        return self.energy_constrained()

    def rho_used(self) -> float:
        if self.normalized_mode() == "ospf":
            return max(float(self.rho), 0.0)
        return 0.0

    def fail_safe_timeout(self) -> bool:
        if self.fail_safe_on_solver_timeout is not None:
            return bool(self.fail_safe_on_solver_timeout)
        return self.applies_command()

    def enforcement_blockers(self) -> list[str]:
        """Why force/passive/ospf must refuse to start.

        Observe/off have no blockers.  Writing
        ``velocity_error_persistent_m_s: 0`` is not a terminal proof.
        """
        mode = self.normalized_mode()
        if mode in ("observe", "off"):
            return []
        reasons: list[str] = []
        if mode in ("force", "passive", "ospf"):
            if not self.stop_dx_certified:
                reasons.append("stop_dx_ub.certified is false")
            if not self.stop_dx_bins:
                reasons.append("stop_dx_ub bins empty")
            if not self.pose_domain_declared:
                reasons.append("pose domain not declared")
            elif len(self.pose_min) < 6 or len(self.pose_max) < 6:
                reasons.append("pose_min/max missing")
            elif any(
                not math.isfinite(float(value))
                for value in (*self.pose_min[:6], *self.pose_max[:6])
            ) or any(
                float(lo) > float(hi)
                for lo, hi in zip(self.pose_min[:6], self.pose_max[:6])
            ):
                reasons.append("pose_min/max invalid")
            if not self.payload_domain_declared:
                reasons.append("payload domain not declared")
            elif (
                self.payload_min_kg is None
                or self.payload_max_kg is None
                or self.payload_kg is None
                or not math.isfinite(float(self.payload_min_kg))
                or not math.isfinite(float(self.payload_max_kg))
                or not math.isfinite(float(self.payload_kg))
                or float(self.payload_min_kg) > float(self.payload_max_kg)
                or float(self.payload_kg) < float(self.payload_min_kg)
                or float(self.payload_kg) > float(self.payload_max_kg)
            ):
                reasons.append("payload_kg outside declared range")
        if mode in ("passive", "ospf"):
            if not self.terminal_invariance_proven:
                reasons.append(
                    "terminal_invariance_proven is false "
                    "(ē_v(∞)=0 must come from a hold / infinite-horizon "
                    "or closed-loop argument, not a written zero)"
                )
            if not self.energy_sign_verified:
                reasons.append("energy_sign_verified is false")
        return reasons


class ShieldCertificationError(RuntimeError):
    """force / passive / ospf requested without a completed certificate."""


@dataclass
class SafetyShieldResult:
    u_nom: float
    u_b: float
    u_shield_hyp: float
    u_sent: float
    lambda_star: float
    lambda_obs: float
    shield_applied: bool
    shield_feasible: bool
    solver_timeout: bool
    f_ub_n: float
    e_lb_j: float
    w_lb_j: float
    rho_v2_w: float
    n_stop: int
    tube_violation: bool
    solver_us: float
    dx_pipe_ub_m: float
    in_terminal: bool
    infeasible_reason: str = ""
    f_constraint_margin_n: float = 0.0
    energy_margin_j: float = 0.0
    terminal_ok: bool = False
    recovery_latched: bool = False
    domain_ok: bool = True
    aj_ok: bool = True
    uncertified_brake: bool = False


@dataclass
class _PlantState:
    v: float
    delay: deque[float]
    u_prev: float
    u_prev2: float
    a_plus: float = 0.0


class DelaySafetyShield:
    """Online backup-to-terminal filter on the press-positive normal axis."""

    def __init__(
        self,
        cfg: SafetyShieldConfig,
        dt_s: float,
        *,
        require_certificate: bool = False,
    ) -> None:
        self.cfg = cfg
        self.dt_s = max(float(dt_s), 1e-4)
        self._mode_frozen = cfg.normalized_mode()
        self.require_certificate = bool(require_certificate)
        self.reset()

    def enforcement_blockers(self) -> list[str]:
        reasons = list(self.cfg.enforcement_blockers())
        mode = self.cfg.normalized_mode()
        if mode in ("force", "passive", "ospf"):
            if self.cfg.stop_dx_certified and self.cfg.stop_dx_bins:
                dx = self.lookup_stop_dx(0.0, 0.0, 0.0)
                if not math.isfinite(dx):
                    reasons.append("stop table does not cover the rest query")
        if mode in ("passive", "ospf") and not self.terminal_set_invariant():
            reasons.append("terminal_set_invariant is false")
        return reasons

    def assert_enforcement_ready(self) -> None:
        reasons = self.enforcement_blockers()
        if reasons:
            raise ShieldCertificationError(
                f"refuse {self.cfg.normalized_mode()}: " + "; ".join(reasons)
            )

    def reset(self) -> None:
        cfg = self.cfg
        delay_n = self._delay_steps()
        self._delay: deque[float] = deque([0.0] * delay_n, maxlen=max(delay_n, 1))
        self._v_plant = 0.0
        self._a_plus = 0.0
        self._u_prev = 0.0
        self._u_prev2 = 0.0
        self.energy_lb_j = float(cfg.e0_j)
        self._recovery_latched = False
        self._recovery_ok_s = 0.0
        self._uncertified_brake_latched = False
        self.last = SafetyShieldResult(
            u_nom=0.0,
            u_b=0.0,
            u_shield_hyp=0.0,
            u_sent=0.0,
            lambda_star=1.0,
            lambda_obs=1.0,
            shield_applied=False,
            shield_feasible=True,
            solver_timeout=False,
            f_ub_n=0.0,
            e_lb_j=float(cfg.e0_j),
            w_lb_j=0.0,
            rho_v2_w=0.0,
            n_stop=0,
            tube_violation=False,
            solver_us=0.0,
            dx_pipe_ub_m=0.0,
            in_terminal=True,
        )

    def _delay_steps(self) -> int:
        return max(int(round(float(self.cfg.t0_s) / self.dt_s)), 0)

    def _alpha(self) -> float:
        tp = max(float(self.cfg.tp_s), 1e-4)
        return math.exp(-self.dt_s / tp)

    def _error_v(self, step_index: int) -> float:
        table = self.cfg.velocity_error_ub_m_s
        if table:
            idx = min(max(int(step_index) - 1, 0), len(table) - 1)
            return max(float(table[idx]), 0.0)
        defaults = default_velocity_error_ub(max(int(self.cfg.horizon_steps), 1))
        idx = min(max(int(step_index) - 1, 0), len(defaults) - 1)
        return defaults[idx]

    def _error_v_persistent(self) -> float:
        """Declared ``ē_v(∞)``.  A finite table's last entry is not this.

        ``None`` (unknown) or any positive value makes ``D_T^ub = ∞``.
        """
        pers = self.cfg.velocity_error_persistent_m_s
        if pers is None:
            return float("inf")
        return max(float(pers), 0.0)

    def _error_v_infinite(self, step_index: int) -> float:
        """Finite-table ``ē_v(k)``; do not hold the last entry past ``N``."""
        table = self.cfg.velocity_error_ub_m_s
        idx = int(step_index) - 1
        if table and 0 <= idx < len(table):
            return max(float(table[idx]), 0.0)
        return self._error_v_persistent()

    def _error_x(self, step_index: int) -> float:
        table = self.cfg.position_error_ub_m
        if table:
            idx = min(max(int(step_index) - 1, 0), len(table) - 1)
            return max(float(table[idx]), 0.0)
        ev = self.cfg.velocity_error_ub_m_s or default_velocity_error_ub(
            max(int(self.cfg.horizon_steps), 1)
        )
        px = default_position_error_ub(ev, self.dt_s)
        idx = min(max(int(step_index) - 1, 0), len(px) - 1)
        return px[idx]

    def _error_x_plus(self, step_index: int) -> float:
        """Press-positive indent error ``ē_{x,+}(i)``.

        Signed ``position_error_ub_m`` can cancel and is not used.  If no
        plus table is loaded, ``Σ dt ē_v`` bounds both ``|Σ dt (v−v̂)|``
        and ``[Σ dt [v]_+ − Σ dt [v̂]_+]_+``.
        """
        table = self.cfg.position_error_ub_plus_m
        if table:
            idx = min(max(int(step_index) - 1, 0), len(table) - 1)
            return max(float(table[idx]), 0.0)
        ev = self.cfg.velocity_error_ub_m_s or default_velocity_error_ub(
            max(int(self.cfg.horizon_steps), 1)
        )
        px = default_position_error_ub(ev, self.dt_s)
        idx = min(max(int(step_index) - 1, 0), len(px) - 1)
        return px[idx]

    def _error_a(self, step_index: int) -> float:
        """Identified ``ē_a(i)``.  NaN if the bound has not been fitted."""
        table = self.cfg.acceleration_error_ub_m_s2
        if not table:
            return float("nan")
        idx = min(max(int(step_index) - 1, 0), len(table) - 1)
        return max(float(table[idx]), 0.0)

    def _worst_successor(
        self,
        v0: float,
        v_hat: float,
        a_hat_plus: float,
        ev: float,
        ea: float,
    ) -> tuple[float, float]:
        """Worst covering query ``(v_{1,q}, a_{1,q})`` in the one-step tube.

        ``v_{1,q} = max{|v̂−ē_v|, |v̂+ē_v|}``.  ``a_{1,q}`` is the
        press-positive acceleration upper bound: the velocity-tube implied
        value, never bare nominal ``[Δv̂/T_s]_+``.  An identified ``ē_a``
        is added when present.
        """
        v_q = max(abs(float(v_hat) - float(ev)), abs(float(v_hat) + float(ev)))
        a_tube = 0.0
        if self.dt_s > 0.0:
            a_tube = max((float(v_hat) + float(ev) - float(v0)) / self.dt_s, 0.0)
        if math.isfinite(float(ea)):
            a_q = max(max(float(a_hat_plus), 0.0) + max(float(ea), 0.0), a_tube)
        else:
            a_q = a_tube
        return v_q, a_q

    def queue_press(self) -> float:
        if not self._delay:
            return 0.0
        return max(0.0, max(float(x) for x in self._delay))

    def queue_remain_m(self, delay: deque[float] | None = None) -> float:
        """Weighted remaining press of the known delay line, ``dt Σ [u]_+``."""
        q = self._delay if delay is None else delay
        if not q:
            return 0.0
        return self.dt_s * sum(max(float(u), 0.0) for u in q)

    def _stop_query_extras(
        self, plant: _PlantState | None = None
    ) -> tuple[float, float, float]:
        """``([u_prev]+, [a_cmd]+, [q_front]+)`` for the jerk-backup table."""
        u_prev = float(plant.u_prev) if plant is not None else float(self._u_prev)
        u_prev2 = float(plant.u_prev2) if plant is not None else float(self._u_prev2)
        delay = plant.delay if plant is not None else self._delay
        a_cmd = (u_prev - u_prev2) / self.dt_s if self.dt_s > 0.0 else 0.0
        q_front = float(delay[0]) if delay else 0.0
        return max(u_prev, 0.0), max(a_cmd, 0.0), max(q_front, 0.0)

    def lookup_stop_dx(
        self,
        v0: float,
        a0: float,
        q0: float,
        u_prev: float = 0.0,
        a_cmd: float = 0.0,
        q_front: float = 0.0,
    ) -> float:
        """Monotonic covering of remaining press-positive indent.

        Query is ``(|v0|,[a0]+,q_remain,[u_prev]+,[a_cmd]+,[q_front]+)``.
        ``a0`` is press-direction actual acceleration.  Missing table
        fields parse as 0 and fail-close on a nonzero extra.  Returns
        NaN if the table is empty, +inf if uncovered.
        """
        bins = self.cfg.stop_dx_bins
        if not bins:
            return float("nan")
        v = abs(float(v0))
        a = max(float(a0), 0.0)
        q = max(float(q0), 0.0)
        up = max(float(u_prev), 0.0)
        ac = max(float(a_cmd), 0.0)
        qf = max(float(q_front), 0.0)
        covering = [
            float(b.dx_ub_m)
            for b in bins
            if float(b.v0_m_s) + 1e-12 >= v
            and float(b.a0_m_s2) + 1e-12 >= a
            and float(b.q_remain_m) + 1e-12 >= q
            and float(b.u_prev_m_s) + 1e-12 >= up
            and float(b.a_cmd_m_s2) + 1e-12 >= ac
            and float(b.q_front_m_s) + 1e-12 >= qf
        ]
        if not covering:
            return float("inf")
        return min(covering)

    def max_safe_approach_m_s(self, *, room_n: float, a0: float, q0: float) -> float | None:
        """Largest tabulated ``v0`` with ``K_ub D_ub <= room``.  None = no table."""
        if not self.cfg.stop_dx_certified or not self.cfg.stop_dx_bins:
            return None
        k_ub = max(float(self.cfg.k_ub_n_m), 1.0)
        if float(room_n) <= 0.0:
            return 0.0
        u_prev, a_cmd, q_front = self._stop_query_extras()
        best = 0.0
        any_ok = False
        for b in self.cfg.stop_dx_bins:
            dx = self.lookup_stop_dx(
                float(b.v0_m_s),
                a0,
                q0,
                u_prev,
                a_cmd,
                q_front,
            )
            if not math.isfinite(dx):
                continue
            if k_ub * dx <= float(room_n) + 1e-12:
                best = max(best, abs(float(b.v0_m_s)))
                any_ok = True
        return best if any_ok else 0.0

    def _limit_increment(self, u_cmd: float, u_prev: float, u_prev2: float) -> float:
        cfg = self.cfg
        dt = self.dt_s
        a_max = max(float(cfg.a_max_m_s2), 0.0)
        j_max = max(float(cfg.j_max_m_s3), 0.0)
        desired = float(u_cmd)
        if a_max > 0.0:
            desired = _clip(desired, u_prev - a_max * dt, u_prev + a_max * dt)
        if j_max > 0.0 and dt > 0.0:
            acc = (desired - u_prev) / dt
            last_acc = (u_prev - u_prev2) / dt
            acc = _clip(acc, last_acc - j_max * dt, last_acc + j_max * dt)
            desired = u_prev + acc * dt
        return float(desired)

    def backup_command(
        self,
        u_prev: float,
        u_prev2: float,
        *,
        released: bool = False,
        v_pred: float = 0.0,
    ) -> float:
        pressing = (not released) and float(v_pred) > 0.0
        target = -abs(float(self.cfg.u_retract_m_s)) if pressing else 0.0
        return self._limit_increment(target, u_prev, u_prev2)

    def terminal_hold_command(self, u_prev: float, u_prev2: float) -> float:
        return self._limit_increment(0.0, u_prev, u_prev2)

    def _copy_plant(self) -> _PlantState:
        return _PlantState(
            v=float(self._v_plant),
            delay=deque(self._delay, maxlen=self._delay.maxlen),
            u_prev=float(self._u_prev),
            u_prev2=float(self._u_prev2),
            a_plus=float(self._a_plus),
        )

    def _step_plant(self, plant: _PlantState, u_cmd: float) -> float:
        delay_n = self._delay_steps()
        v_old = float(plant.v)
        if delay_n <= 0:
            u_app = float(u_cmd)
        else:
            if len(plant.delay) >= delay_n:
                u_app = float(plant.delay[0])
                plant.delay.append(float(u_cmd))
            else:
                plant.delay.append(float(u_cmd))
                u_app = 0.0
        alpha = self._alpha()
        plant.v = alpha * plant.v + (1.0 - alpha) * u_app
        plant.a_plus = (
            max((float(plant.v) - v_old) / self.dt_s, 0.0) if self.dt_s > 0.0 else 0.0
        )
        plant.u_prev2 = plant.u_prev
        plant.u_prev = float(u_cmd)
        return float(plant.v)

    def _advance_energy(
        self,
        energy: float,
        f_lo: float,
        f_hi: float,
        v_lo: float,
        v_hi: float,
        rho: float,
    ) -> tuple[float, float, float]:
        p_lb = inf_minus_fv(f_lo, f_hi, v_lo, v_hi)
        w_lb = self.dt_s * p_lb - 0.5 * max(float(self.cfg.l_p_w_s), 0.0) * self.dt_s**2
        rho_v2 = max(float(rho), 0.0) * self.dt_s * max(v_lo * v_lo, v_hi * v_hi)
        return energy + w_lb - rho_v2, w_lb, rho_v2

    def _in_terminal(
        self,
        *,
        f_ub: float,
        v_abs_ub: float,
        u_cmd: float,
        u_prev: float,
        delay: deque[float],
        energy: float,
        require_energy: bool,
        f_term: float | None = None,
    ) -> bool:
        cfg = self.cfg
        f_lim = float(cfg.f_release_n) if f_term is None else float(f_term)
        if cfg.require_contact_free_terminal and f_ub > f_lim + 1e-9:
            return False
        if v_abs_ub > float(cfg.v_hold_m_s) + 1e-9:
            return False
        acc = abs(u_cmd - u_prev) / self.dt_s if self.dt_s > 0.0 else 0.0
        if acc > float(cfg.a_hold_m_s2) + 1e-9:
            return False
        if delay and max(abs(x) for x in delay) > float(cfg.queue_clear_m_s) + 1e-9:
            return False
        if require_energy and energy + 1e-12 < float(cfg.eps_j):
            return False
        return True

    def _terminal_box_vertices(self) -> list[_PlantState]:
        """Vertices / press corners of the box ``T``.

        ``T`` allows ``|v|≤v_hold``, ``|q_i|≤q_clear``, ``|a|≤a_hold``.
        The origin is not the worst initial state.  Queue permutation
        inside ``T`` is sampled at all-press, front, and back, not
        enumerated (that would be ``3^{N_d}``).
        """
        n = max(self._delay_steps(), 1)
        vh = max(float(self.cfg.v_hold_m_s), 0.0)
        qc = max(float(self.cfg.queue_clear_m_s), 0.0)
        ah = max(float(self.cfg.a_hold_m_s2), 0.0)
        da = ah * self.dt_s

        def make(
            v: float,
            delay_vals: list[float],
            u_prev: float,
            u_prev2: float,
            a_plus: float,
        ) -> _PlantState:
            u2 = _clip(u_prev2, -qc, qc)
            return _PlantState(
                v=float(v),
                delay=deque(delay_vals, maxlen=n),
                u_prev=float(u_prev),
                u_prev2=float(u2),
                a_plus=float(a_plus),
            )

        zeros = [0.0] * n
        plus = [qc] * n
        minus = [-qc] * n
        front = [qc] + [0.0] * (n - 1)
        back = [0.0] * (n - 1) + [qc]
        corners = (
            (zeros, 0.0, 0.0, 0.0),
            (plus, qc, qc - da, ah),
            (plus, qc, qc, 0.0),
            (front, qc, qc - da, ah),
            (back, qc, qc - da, ah),
            (minus, -qc, -qc + da, 0.0),
            (zeros, qc, qc - da, ah),
        )
        out: list[_PlantState] = []
        for v in (vh, 0.0, -vh):
            for delay, up, up2, ap in corners:
                out.append(make(v, delay, up, up2, ap))
        return out

    def _indent_from_plant(self, plant0: _PlantState) -> float:
        """``Σ T_s [v_k^{ub}]_+`` under ``u_T`` from one initial state."""
        n = max(self._delay_steps(), 1)
        plant = _PlantState(
            v=float(plant0.v),
            delay=deque(plant0.delay, maxlen=n),
            u_prev=float(plant0.u_prev),
            u_prev2=float(plant0.u_prev2),
            a_plus=float(plant0.a_plus),
        )
        table_n = len(self.cfg.velocity_error_ub_m_s) if self.cfg.velocity_error_ub_m_s else 0
        n_max = max(int(self.cfg.horizon_steps), table_n, n, 8) + 200
        d_t = 0.0
        for k in range(n_max):
            u_t = self.terminal_hold_command(plant.u_prev, plant.u_prev2)
            v = self._step_plant(plant, u_t)
            ev = self._error_v_infinite(k + 1)
            if not math.isfinite(ev):
                return float("inf")
            d_t += self.dt_s * max(0.0, v + ev)
            if not math.isfinite(d_t):
                return float("inf")
        ev_inf = self._error_v_persistent()
        if max(0.0, float(plant.v) + ev_inf) > 1e-9:
            return float("inf")
        return d_t

    def terminal_indent_ub(self) -> float:
        """``D_T^{ub} = sup_{ξ∈T} Σ T_s [v_k]_+`` on the sampled box.

        Returns ``+∞`` unless ``ē_v(∞)`` is declared to be 0.  A last
        finite-horizon ``ē_v(N)`` is not that declaration.
        """
        ev_inf = self._error_v_persistent()
        if not math.isfinite(ev_inf) or ev_inf > 1e-15:
            return float("inf")
        worst = 0.0
        for plant in self._terminal_box_vertices():
            d_t = self._indent_from_plant(plant)
            if not math.isfinite(d_t):
                return float("inf")
            worst = max(worst, d_t)
        return worst

    def terminal_set_invariant(self, *, require_energy: bool | None = None) -> bool:
        """``g_0 >= D_T^{ub}`` over the box ``T``, not just the origin.

        ``x_detach_m`` is ``g_0``.  ``ē_v(∞)`` must be declared 0;
        a finite table ending at 0 is not enough.  Energy is not a
        substitute for the gap test.
        """
        del require_energy
        d_t = self.terminal_indent_ub()
        if not math.isfinite(d_t):
            return False
        return max(float(self.cfg.x_detach_m), 0.0) + 1e-12 >= d_t

    def _rollout(
        self,
        u0: float,
        *,
        f0: float,
        energy0: float,
        enforce_force: bool,
        enforce_energy: bool,
        rho: float,
        f_max: float,
    ) -> tuple[bool, float, float, int, bool, float, str]:
        """Return (feasible, F_ub, E_lb, n_stop, reached_T, dx_pipe, reason).

        Force indent is the every-tick max-hold
        ``Δx_ub(i) = max{Δx_ub(i−1), Δx̂^+(i) + ē_{x,+}(i)}``.
        When a certified stop table exists, that tube is replaced by
        ``Δx_1^ub(ξ, u0) + D_b^ub(ξ_1)`` rather than
        ``max(model, D_b^ub(ξ))``.
        """
        cfg = self.cfg
        plant = self._copy_plant()
        v_start = float(plant.v)
        f_ub = max(float(f0), 0.0) + max(float(cfg.e_f_n), 0.0)
        energy = float(energy0)
        dx_hat = 0.0
        dx_ub = 0.0
        dx_table = float("nan")
        rf_acc = 0.0
        n_stop = 0
        stopped = False
        horizon = max(int(cfg.horizon_steps), 1)
        k_ub = max(float(cfg.k_ub_n_m), 0.0)
        released = False
        use_lookup = bool(cfg.stop_dx_certified)
        if enforce_force and f_ub > float(f_max) + 1e-9:
            return False, f_ub, energy, 0, False, 0.0, "force"
        if use_lookup and not cfg.stop_dx_bins and enforce_force:
            return False, f_ub, energy, 0, False, 0.0, "force"
        for i in range(horizon):
            if i == 0:
                u_cmd = float(u0)
            else:
                u_cmd = self.backup_command(
                    plant.u_prev,
                    plant.u_prev2,
                    released=released,
                    v_pred=plant.v,
                )
            v = self._step_plant(plant, u_cmd)
            ev = self._error_v(i + 1)
            ex_plus = self._error_x_plus(i + 1)
            v_hi = v + ev
            v_lo = v - ev
            r_f = float(cfg.r_f_n_s) if i < int(cfg.r_f_window_steps) else 0.0
            dx_hat += self.dt_s * max(0.0, v)
            dx_ub = max(dx_ub, dx_hat + max(ex_plus, 0.0))
            if use_lookup and i == 0:
                v_q, a_q = self._worst_successor(
                    v_start,
                    v,
                    plant.a_plus,
                    ev,
                    self._error_a(1),
                )
                u_p, a_c, q_f = self._stop_query_extras(plant)
                tail = self.lookup_stop_dx(
                    v_q,
                    a_q,
                    self.queue_remain_m(plant.delay),
                    u_p,
                    a_c,
                    q_f,
                )
                if (math.isnan(tail) or math.isinf(tail)) and enforce_force:
                    return False, f_ub, energy, n_stop, False, dx_ub, "force"
                if math.isfinite(tail):
                    dx_table = dx_ub + float(tail)
            if use_lookup and math.isfinite(dx_table):
                dx_use = float(dx_table)
            else:
                dx_use = dx_ub
            rf_acc += self.dt_s * max(r_f, 0.0)
            f_ub = (
                max(float(f0), 0.0)
                + max(float(cfg.e_f_n), 0.0)
                + k_ub * dx_use
                + rf_acc
            )
            energy, _, _ = self._advance_energy(
                energy, 0.0, f_ub, v_lo, v_hi, rho
            )
            if enforce_force and f_ub > float(f_max) + 1e-9:
                return False, f_ub, energy, n_stop, False, dx_use, "force"
            if enforce_energy and energy + 1e-12 < float(cfg.eps_j):
                return False, f_ub, energy, n_stop, False, dx_use, "energy"
            if (not stopped) and v_hi <= 0.0:
                stopped = True
                n_stop = i + 1
            if f_ub <= float(cfg.f_release_n):
                released = True
        if not stopped:
            n_stop = horizon
        reached = self._in_terminal(
            f_ub=f_ub,
            v_abs_ub=max(abs(v_lo), abs(v_hi)),
            u_cmd=plant.u_prev,
            u_prev=plant.u_prev2,
            delay=plant.delay,
            energy=energy,
            require_energy=enforce_energy,
        )
        if enforce_force and cfg.should_enforce_terminal() and not reached:
            return False, f_ub, energy, n_stop, False, dx_ub, "terminal"
        dx_out = dx_table if (use_lookup and math.isfinite(dx_table)) else dx_ub
        return True, f_ub, energy, n_stop, reached, dx_out, ""

    def pipeline_penetration_ub(
        self,
        f_csv: float | None = None,
        v_actual: float | None = None,
        a_actual: float | None = None,
    ) -> float:
        """Remaining indentation if the backup law starts now.

        Uses the same ``u_b`` plant as the certificate.  Measured speed
        and ``[a_actual]_+`` correct ``ξ``; the known delay queue is kept.
        A certified table is ``D_b^ub(ξ)`` (backup from now).  Without it
        the bound is the ``ē_{x,+}`` max-hold of the backup rollout.
        """
        if v_actual is None or not math.isfinite(float(v_actual)):
            return 0.0
        self._sync_plant_from_measurement(v_actual, a_actual)
        if self.cfg.stop_dx_certified and self.cfg.stop_dx_bins:
            u_p, a_c, q_f = self._stop_query_extras()
            lookup = self.lookup_stop_dx(
                self._v_plant,
                self._a_plus,
                self.queue_remain_m(),
                u_p,
                a_c,
                q_f,
            )
            if math.isfinite(lookup):
                dx = max(float(lookup), 0.0)
                if dx > 1e-12:
                    dx += max(float(self.cfg.e_x_m), 0.0)
                return dx
            if math.isinf(lookup):
                return float("inf")
        u_b = self.backup_command(
            self._u_prev,
            self._u_prev2,
            released=False,
            v_pred=self._v_plant,
        )
        _ok, _f, _e, _n, _t, dx, _reason = self._rollout(
            u_b,
            f0=0.0 if f_csv is None else max(float(f_csv), 0.0),
            energy0=self.energy_lb_j,
            enforce_force=False,
            enforce_energy=False,
            rho=0.0,
            f_max=1e9,
        )
        dx = max(float(dx), 0.0)
        if dx > 1e-12:
            dx += max(float(self.cfg.e_x_m), 0.0)
        return dx

    def shift_tail_feasible(
        self,
        u0: float,
        *,
        f0: float,
        energy0: float,
        enforce_force: bool,
        enforce_energy: bool,
        rho: float,
        f_max: float,
    ) -> bool:
        """If ``(u0, u_b, …)`` is feasible, the shifted tail stays feasible."""
        ok0, *_rest = self._rollout(
            u0,
            f0=f0,
            energy0=energy0,
            enforce_force=enforce_force,
            enforce_energy=enforce_energy,
            rho=rho,
            f_max=f_max,
        )
        if not ok0:
            return False
        saved = (
            float(self._v_plant),
            deque(self._delay, maxlen=self._delay.maxlen),
            float(self._u_prev),
            float(self._u_prev2),
            float(self._a_plus),
        )
        plant = self._copy_plant()
        u_lim = self._limit_increment(float(u0), plant.u_prev, plant.u_prev2)
        v = self._step_plant(plant, u_lim)
        ev = self._error_v(1)
        v_hi = v + ev
        v_press_ub = max(0.0, v_hi)
        f_next = (
            max(float(f0), 0.0)
            + max(float(self.cfg.e_f_n), 0.0)
            + self.dt_s * max(float(self.cfg.k_ub_n_m), 0.0) * v_press_ub
        )
        e_next, _, _ = self._advance_energy(
            float(energy0), 0.0, f_next, v - ev, v_hi, float(rho)
        )
        self._v_plant = plant.v
        self._delay = plant.delay
        self._u_prev = plant.u_prev
        self._u_prev2 = plant.u_prev2
        self._a_plus = plant.a_plus
        u_b = self.backup_command(
            self._u_prev,
            self._u_prev2,
            released=False,
            v_pred=self._v_plant,
        )
        try:
            ok1, *_ = self._rollout(
                u_b,
                f0=f_next,
                energy0=e_next,
                enforce_force=enforce_force,
                enforce_energy=enforce_energy,
                rho=rho,
                f_max=f_max,
            )
        finally:
            self._v_plant, self._delay, self._u_prev, self._u_prev2, self._a_plus = saved
        return bool(ok1)

    def _sync_plant_from_measurement(
        self,
        v_actual: float | None,
        a_actual: float | None = None,
    ) -> None:
        if v_actual is not None and math.isfinite(float(v_actual)):
            self._v_plant = float(v_actual)
        if a_actual is not None and math.isfinite(float(a_actual)):
            self._a_plus = max(float(a_actual), 0.0)

    def _commit_sent(self, u_sent: float, *, keep_measured_state: bool = False) -> None:
        v_save = float(self._v_plant)
        a_save = float(self._a_plus)
        plant = self._copy_plant()
        self._step_plant(plant, float(u_sent))
        self._v_plant = plant.v
        self._delay = plant.delay
        self._u_prev = plant.u_prev
        self._u_prev2 = plant.u_prev2
        self._a_plus = plant.a_plus
        if keep_measured_state:
            self._v_plant = v_save
            self._a_plus = a_save

    def update_measured_energy(self, f_csv: float, v_csv: float) -> tuple[float, float]:
        """Advance the certified tank from measurements.  Never clamp in observe."""
        cfg = self.cfg
        p_lb = measured_power_lb(f_csv, v_csv, cfg.bar_f_n, cfg.bar_v_m_s)
        w_lb = self.dt_s * p_lb - 0.5 * max(float(cfg.l_p_w_s), 0.0) * self.dt_s**2
        v_abs = abs(float(v_csv)) + max(float(cfg.bar_v_m_s), 0.0)
        rho_v2 = cfg.rho_used() * self.dt_s * v_abs * v_abs
        self.energy_lb_j = float(self.energy_lb_j + w_lb - rho_v2)
        return w_lb, rho_v2

    def _recovery_hold_ok(
        self,
        *,
        f_csv: float,
        f_max: float,
        v_meas: float,
        a_actual: float | None,
    ) -> bool:
        """True only when force, measured motion, delay queue, and ξ are in hold."""
        cfg = self.cfg
        u_clear = max(float(cfg.queue_clear_m_s), 0.0)
        v_hold = max(float(cfg.v_hold_m_s), 0.0)
        a_hold = max(float(cfg.a_hold_m_s2), 0.0)
        if max(float(f_csv), 0.0) > float(f_max) + 1e-9:
            return False
        if abs(float(v_meas)) > v_hold + 1e-9:
            return False
        if abs(float(self._v_plant)) > v_hold + 1e-9:
            return False
        pending = [float(self._u_prev), float(self._u_prev2), *self._delay]
        if pending and max(abs(u) for u in pending) > u_clear + 1e-9:
            return False
        if self.dt_s > 0.0:
            a_cmd = abs(float(self._u_prev) - float(self._u_prev2)) / self.dt_s
            if a_cmd > a_hold + 1e-9:
                return False
        if (
            a_actual is not None
            and math.isfinite(float(a_actual))
            and abs(float(a_actual)) > a_hold + 1e-9
        ):
            return False
        return True

    @staticmethod
    def _mode_applies(mode: str) -> bool:
        return str(mode).strip().lower() in ("force", "passive", "ospf")

    def _apply_this_tick(self) -> bool:
        """Frozen construct-time mode only.  A live mode change refuses."""
        return self._mode_applies(self._mode_frozen)

    def _mode_mutated(self) -> bool:
        return self.cfg.normalized_mode() != self._mode_frozen

    def _v_domain_m_s(self) -> float:
        if float(self.cfg.v_domain_m_s) > 0.0:
            return float(self.cfg.v_domain_m_s)
        if self.cfg.stop_dx_bins:
            return max(float(b.v0_m_s) for b in self.cfg.stop_dx_bins)
        return float("nan")

    def _a_domain_m_s2(self) -> float:
        if float(self.cfg.a_domain_m_s2) > 0.0:
            return float(self.cfg.a_domain_m_s2)
        declared = max(float(self.cfg.a_max_m_s2), 0.0)
        if self.cfg.stop_dx_bins:
            return max(declared, max(float(b.a0_m_s2) for b in self.cfg.stop_dx_bins))
        return declared if declared > 0.0 else float("nan")

    def _u_domain_m_s(self) -> float:
        if float(self.cfg.u_domain_m_s) > 0.0:
            return float(self.cfg.u_domain_m_s)
        if self.cfg.stop_dx_bins:
            return max(float(b.u_prev_m_s) for b in self.cfg.stop_dx_bins)
        return float("nan")

    def lookup_covers_state(
        self,
        v0: float | None = None,
        a0: float | None = None,
    ) -> bool:
        """True only when the 6-D backup table covers the measured state."""
        v = self._v_plant if v0 is None else float(v0)
        a = self._a_plus if a0 is None else float(a0)
        if not math.isfinite(v) or not math.isfinite(a):
            return False
        q = self.queue_remain_m()
        up, ac, qf = self._stop_query_extras()
        dx = self.lookup_stop_dx(v, a, q, up, ac, qf)
        return math.isfinite(dx)

    def evaluate_domain(
        self,
        *,
        v_actual: float | None,
        a_actual: float | None,
        feedback_age_s: float | None,
        pose_in_domain: bool,
        payload_in_domain: bool,
    ) -> tuple[bool, list[str]]:
        """Certificate-domain membership.  Missing measurements fail closed."""
        reasons: list[str] = []
        if not pose_in_domain:
            reasons.append("pose")
        if not payload_in_domain:
            reasons.append("payload")
        if v_actual is None or not math.isfinite(float(v_actual)):
            reasons.append("v_actual")
        else:
            v_lim = self._v_domain_m_s()
            if not math.isfinite(v_lim):
                reasons.append("v_domain")
            elif abs(float(v_actual)) > v_lim + 1e-12:
                reasons.append("v")
        if a_actual is None or not math.isfinite(float(a_actual)):
            reasons.append("a_actual")
        else:
            a_lim = self._a_domain_m_s2()
            if not math.isfinite(a_lim):
                reasons.append("a_domain")
            elif abs(float(a_actual)) > a_lim + 1e-12:
                reasons.append("a")
        pending = [float(self._u_prev), float(self._u_prev2), *self._delay]
        if any(not math.isfinite(u) for u in pending):
            reasons.append("queue")
        else:
            u_lim = self._u_domain_m_s()
            if not math.isfinite(u_lim):
                reasons.append("queue_domain")
            elif pending and max(abs(u) for u in pending) > u_lim + 1e-12:
                reasons.append("queue")
        if feedback_age_s is None or not math.isfinite(float(feedback_age_s)):
            reasons.append("feedback_age")
        elif float(feedback_age_s) > float(self.cfg.max_feedback_age_s) + 1e-12:
            reasons.append("feedback_age")
        v_q = (
            float(v_actual)
            if v_actual is not None and math.isfinite(float(v_actual))
            else None
        )
        a_q = (
            float(a_actual)
            if a_actual is not None and math.isfinite(float(a_actual))
            else None
        )
        if not self.lookup_covers_state(v_q, a_q):
            reasons.append("lookup")
        return (not reasons), reasons

    def update(
        self,
        u_nom: float,
        *,
        f_csv: float,
        v_actual: float | None,
        f_max_n: float,
        in_domain: bool | None = None,
        a_actual: float | None = None,
        feedback_age_s: float | None = None,
        pose_in_domain: bool = False,
        payload_in_domain: bool = False,
    ) -> SafetyShieldResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        u_nom_f = float(u_nom)
        predicted = float(self._v_plant)
        tube_violation = False
        if v_actual is not None and math.isfinite(float(v_actual)):
            if abs(float(v_actual) - predicted) > self._error_v(1) + 1e-9:
                tube_violation = True
        self._sync_plant_from_measurement(v_actual, a_actual)
        v_meas = (
            float(v_actual)
            if v_actual is not None and math.isfinite(float(v_actual))
            else float(self._v_plant)
        )
        w_lb, rho_v2 = self.update_measured_energy(float(f_csv), v_meas)

        u_b = self.backup_command(
            self._u_prev,
            self._u_prev2,
            released=False,
            v_pred=self._v_plant,
        )
        enforce_force = cfg.diagnoses_force()
        enforce_energy = cfg.energy_constrained()
        rho = cfg.rho_used()
        f_max = max(float(f_max_n), float(cfg.f_release_n))
        f_margin0 = float(f_max) - (max(float(f_csv), 0.0) + max(float(cfg.e_f_n), 0.0))
        energy_margin0 = float(self.energy_lb_j) - float(cfg.eps_j)
        runtime_ok, domain_reasons = self.evaluate_domain(
            v_actual=v_actual,
            a_actual=a_actual,
            feedback_age_s=feedback_age_s,
            pose_in_domain=bool(pose_in_domain),
            payload_in_domain=bool(payload_in_domain),
        )
        if in_domain is False:
            runtime_ok = False
            if "caller" not in domain_reasons:
                domain_reasons = ["caller", *domain_reasons]
        domain_ok = bool(runtime_ok)
        lookup_ok = self.lookup_covers_state(v_actual, a_actual)
        apply = self._apply_this_tick()

        def _aj_send(intent: float) -> tuple[float, bool]:
            sent = self._limit_increment(intent, self._u_prev, self._u_prev2)
            return sent, abs(sent - intent) <= 1e-9 or abs(
                sent - self._limit_increment(sent, self._u_prev, self._u_prev2)
            ) <= 1e-9

        def _refuse(reason: str, *, brake: bool) -> SafetyShieldResult:
            self._recovery_latched = True
            self._recovery_ok_s = 0.0
            if brake:
                self._uncertified_brake_latched = True
            intent = 0.0 if brake else u_b
            u_sent, aj_ok = _aj_send(intent)
            self._commit_sent(u_sent)
            self.last = SafetyShieldResult(
                u_nom=u_nom_f,
                u_b=u_b,
                u_shield_hyp=u_b,
                u_sent=u_sent,
                lambda_star=0.0,
                lambda_obs=float("nan"),
                shield_applied=True,
                shield_feasible=False,
                solver_timeout=False,
                f_ub_n=max(float(f_csv), 0.0),
                e_lb_j=float(self.energy_lb_j),
                w_lb_j=w_lb,
                rho_v2_w=rho_v2,
                n_stop=0,
                tube_violation=tube_violation,
                solver_us=1e6 * (time.perf_counter() - t0),
                dx_pipe_ub_m=0.0,
                in_terminal=False,
                infeasible_reason=reason,
                f_constraint_margin_n=f_margin0,
                energy_margin_j=energy_margin0,
                terminal_ok=False,
                recovery_latched=True,
                domain_ok=domain_ok,
                aj_ok=aj_ok,
                uncertified_brake=bool(self._uncertified_brake_latched),
            )
            return self.last

        if self._mode_mutated():
            return _refuse(
                f"mode_changed:{self._mode_frozen}->{self.cfg.normalized_mode()}",
                brake=True,
            )
        if apply and self._uncertified_brake_latched:
            return _refuse("uncertified_brake", brake=True)
        if apply:
            blockers = self.enforcement_blockers()
            if blockers:
                return _refuse(
                    "uncertified:" + ",".join(blockers),
                    brake=not lookup_ok,
                )
        if (not cfg.enabled) or cfg.normalized_mode() == "off":
            u_sent = u_nom_f
            self._commit_sent(u_sent)
            self.last = SafetyShieldResult(
                u_nom=u_nom_f,
                u_b=u_b,
                u_shield_hyp=u_nom_f,
                u_sent=u_sent,
                lambda_star=1.0,
                lambda_obs=1.0,
                shield_applied=False,
                shield_feasible=True,
                solver_timeout=False,
                f_ub_n=max(float(f_csv), 0.0),
                e_lb_j=float(self.energy_lb_j),
                w_lb_j=w_lb,
                rho_v2_w=rho_v2,
                n_stop=0,
                tube_violation=tube_violation,
                solver_us=1e6 * (time.perf_counter() - t0),
                dx_pipe_ub_m=0.0,
                in_terminal=False,
                infeasible_reason="",
                f_constraint_margin_n=f_margin0,
                energy_margin_j=energy_margin0,
                terminal_ok=False,
                recovery_latched=bool(self._recovery_latched),
                domain_ok=domain_ok,
                aj_ok=True,
                uncertified_brake=False,
            )
            return self.last

        if apply and tube_violation:
            return _refuse(
                "tube",
                brake=not lookup_ok,
            )
        if apply and not domain_ok:
            return _refuse(
                "domain:" + ",".join(domain_reasons),
                brake=True,
            )

        def evaluate(lam: float) -> tuple[bool, float, float, int, bool, float, float, str]:
            u0 = u_b + float(lam) * (u_nom_f - u_b)
            u0 = self._limit_increment(u0, self._u_prev, self._u_prev2)
            ok, f_ub, e_lb, n_stop, reached, dx, reason = self._rollout(
                u0,
                f0=max(float(f_csv), 0.0),
                energy0=self.energy_lb_j,
                enforce_force=enforce_force,
                enforce_energy=enforce_energy,
                rho=rho,
                f_max=f_max,
            )
            return ok, u0, f_ub, n_stop, reached, dx, e_lb, reason

        timeout = False
        budget_s = max(float(cfg.solver_budget_us), 1.0) * 1e-6
        lo, hi = 0.0, 1.0
        best_ok = False
        best_lam = float("nan")
        best_u0 = u_b
        best_f = max(float(f_csv), 0.0)
        best_e = float(self.energy_lb_j)
        best_n = 0
        best_t = False
        best_dx = 0.0
        fail_reason = ""

        ok1, u1, f1, n1, t1, dx1, e1, r1 = evaluate(1.0)
        if time.perf_counter() - t0 > budget_s:
            timeout = True
            fail_reason = "timeout"
        if ok1:
            best_ok, best_lam, best_u0 = True, 1.0, u1
            best_f, best_e, best_n, best_t, best_dx = f1, e1, n1, t1, dx1
        elif not timeout:
            fail_reason = r1 or "force"
            ok0, u0c, f0c, n0c, t0c, dx0c, e0c, r0 = evaluate(0.0)
            if time.perf_counter() - t0 > budget_s:
                timeout = True
                fail_reason = "timeout"
            if ok0:
                best_ok, best_lam, best_u0 = True, 0.0, u0c
                best_f, best_e, best_n, best_t, best_dx = f0c, e0c, n0c, t0c, dx0c
                fail_reason = ""
                while hi - lo > max(float(cfg.lambda_tol), 1e-4):
                    if time.perf_counter() - t0 > budget_s:
                        timeout = True
                        break
                    mid = 0.5 * (lo + hi)
                    okm, um, fm, nm, tm, dxm, em, _rm = evaluate(mid)
                    if okm:
                        lo = mid
                        best_ok, best_lam, best_u0 = True, mid, um
                        best_f, best_e, best_n, best_t, best_dx = fm, em, nm, tm, dxm
                    else:
                        hi = mid
            else:
                fail_reason = r0 or fail_reason
                best_f, best_e, best_n, best_t, best_dx = f0c, e0c, n0c, t0c, dx0c

        feasible = bool(best_ok) and (not timeout)
        if timeout and not best_ok:
            feasible = False
            fail_reason = "timeout"

        lambda_obs = best_lam if best_ok else float("nan")
        u_hyp = best_u0 if best_ok else u_b
        hold_ok = self._recovery_hold_ok(
            f_csv=float(f_csv),
            f_max=f_max,
            v_meas=v_meas,
            a_actual=a_actual,
        )
        if apply and ((not feasible) or timeout):
            self._recovery_latched = True
            self._recovery_ok_s = 0.0
        if self._recovery_latched and apply:
            if hold_ok:
                self._recovery_ok_s += self.dt_s
            else:
                self._recovery_ok_s = 0.0
            if self._recovery_ok_s + 1e-12 >= max(float(cfg.recovery_hold_s), 0.0) and feasible:
                self._recovery_latched = False
                self._recovery_ok_s = 0.0

        if apply:
            if self._recovery_latched:
                u_sent = self._limit_increment(u_b, self._u_prev, self._u_prev2)
                applied = True
            elif feasible:
                u_sent = u_hyp
                applied = abs(lambda_obs - 1.0) > max(float(cfg.lambda_tol), 1e-4)
            else:
                u_sent = self._limit_increment(u_b, self._u_prev, self._u_prev2)
                applied = True
        else:
            u_sent = u_nom_f
            applied = False

        self._commit_sent(u_sent)
        self.last = SafetyShieldResult(
            u_nom=u_nom_f,
            u_b=u_b,
            u_shield_hyp=float(u_hyp),
            u_sent=float(u_sent),
            lambda_star=float(lambda_obs) if apply and best_ok and not self._recovery_latched else 1.0,
            lambda_obs=float(lambda_obs),
            shield_applied=bool(applied),
            shield_feasible=bool(best_ok) and not timeout,
            solver_timeout=bool(timeout),
            f_ub_n=float(best_f),
            e_lb_j=float(self.energy_lb_j),
            w_lb_j=w_lb,
            rho_v2_w=rho_v2,
            n_stop=int(best_n),
            tube_violation=tube_violation,
            solver_us=1e6 * (time.perf_counter() - t0),
            dx_pipe_ub_m=float(best_dx),
            in_terminal=bool(best_t),
            infeasible_reason="" if (best_ok and not timeout) else fail_reason,
            f_constraint_margin_n=float(f_max) - float(best_f),
            energy_margin_j=float(best_e) - float(cfg.eps_j),
            terminal_ok=bool(best_t),
            recovery_latched=bool(self._recovery_latched),
            domain_ok=domain_ok,
            aj_ok=True,
            uncertified_brake=bool(self._uncertified_brake_latched),
        )
        if not best_ok:
            self.last.shield_feasible = False
        return self.last
