"""Single rail-owner mixer: d* PI, task share, and soft-escape guard.

``u_task`` is the TCP-motion share assigned to the rail.  ``d = y_tcp − y_rail``
is J4's geometry: if the rail does not carry that Y share, ``e_d`` grows and
J4 walks (run 152537).  Posture ``u_post`` sits on the same command.  Only an
explicitly active soft ``u_escape`` is un-cancellable.  Hard wall / collision /
over-force stay downstream of the 2 Hz LPF.

``V_d_proxy = 0.5 * kp_mid * e_d²`` is a configuration-error storage proxy.
``kp_mid`` has units s⁻¹; this is not stiffness and not joules.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

EPS_ENTER = 2.0e-4
EPS_EXIT = 1.0e-4


def clip(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        return 0.5 * (float(lo) + float(hi))
    return float(min(hi, max(lo, x)))


def project(x: float, lo: float, hi: float) -> float:
    return clip(float(x), float(lo), float(hi))


def j4_index(nv: int) -> int:
    """J4 in an 8-DoF vector is q[4]; in a 7-axis arm vector it is q[3]."""
    n = int(nv)
    if n >= 8:
        return 4
    if n == 7:
        return 3
    return min(4, max(0, n - 1))


def update_escape_dir(
    *,
    explicit_active: bool,
    u_escape_raw: float,
    prev_dir: int,
    eps_enter: float = EPS_ENTER,
    eps_exit: float = EPS_EXIT,
) -> int:
    """Latch escape-guard direction from explicit safety state + hysteresis.

    Direction is never inferred from instantaneous speed alone.  It locks only
    when the safety module is explicitly active and |u_escape| exceeds
    ``eps_enter``, and it is held until explicit deactivation (or, while still
    active, until |u| drops below ``eps_exit``).
    """
    if not explicit_active:
        return 0
    u = float(u_escape_raw)
    mag = abs(u)
    d = int(prev_dir)
    if d == 0:
        if mag > float(eps_enter):
            return 1 if u > 0.0 else -1
        return 0
    if mag < float(eps_exit):
        return 0
    return d


def wall_velocity_bounds(
    u_max: float,
    leave_sign: float,
) -> tuple[float, float]:
    cap = abs(float(u_max))
    lo, hi = -cap, cap
    s = float(leave_sign)
    if s > 0.0:
        hi = min(hi, 0.0)
    elif s < 0.0:
        lo = max(lo, 0.0)
    return lo, hi


def allocate_rail_shares(
    *,
    u_task_raw: float,
    u_post_raw: float,
    u_escape_raw: float,
    escape_dir: int,
    u_lo: float,
    u_hi: float,
) -> dict[str, float]:
    """Static mix.  Only the latched escape direction is a hard guard."""
    u_esc_f = project(float(u_escape_raw), float(u_lo), float(u_hi))
    d = int(escape_dir)
    if d > 0:
        t_lo, t_hi = 0.0, float(u_hi) - u_esc_f
    elif d < 0:
        t_lo, t_hi = float(u_lo) - u_esc_f, 0.0
    else:
        t_lo, t_hi = float(u_lo), float(u_hi)
    u_task_f = project(float(u_task_raw), t_lo, t_hi)
    u_base = u_esc_f + u_task_f
    if d > 0:
        p_lo, p_hi = u_esc_f - u_base, float(u_hi) - u_base
    elif d < 0:
        p_lo, p_hi = float(u_lo) - u_base, u_esc_f - u_base
    else:
        p_lo, p_hi = float(u_lo) - u_base, float(u_hi) - u_base
    u_post_f = project(float(u_post_raw), p_lo, p_hi)
    u_feas = u_base + u_post_f
    return {
        "u_escape_feasible": u_esc_f,
        "u_task_feasible": u_task_f,
        "u_base": u_base,
        "u_post_feasible": u_post_f,
        "u_feasible": u_feas,
    }


@dataclass
class RailMixTelemetry:
    u_task_raw: float = 0.0
    u_task_feasible: float = 0.0
    u_pi_raw: float = 0.0
    u_mid_cmd: float = 0.0
    u_post_raw: float = 0.0
    u_post_feasible: float = 0.0
    u_mid_applied: float = 0.0
    d_star_dot_cmd: float = 0.0
    d_star_ref: float = 0.0
    u_escape_raw: float = 0.0
    u_escape_feasible: float = 0.0
    escape_active: float = 0.0
    escape_dir: float = 0.0
    u_base: float = 0.0
    u_feasible: float = 0.0
    e_d: float = 0.0
    V_d_proxy: float = 0.0
    xi: float = 0.0


class DStarRef:
    """Stateful d* used by the rail owner.  Init from d_live, never snap."""

    def __init__(self) -> None:
        self.ref: float | None = None
        self.dot: float = 0.0

    def reset(self) -> None:
        self.ref = None
        self.dot = 0.0

    def init_from_live(self, d_live: float) -> None:
        self.ref = float(d_live)
        self.dot = 0.0

    def follow_target(
        self, d_target: float, d_live: float | None = None
    ) -> tuple[float, float]:
        """Ride the live target.  Used when this mixer is not commanding."""
        if np.isfinite(d_target):
            self.ref = float(d_target)
        elif d_live is not None and np.isfinite(d_live):
            self.ref = float(d_live)
        elif self.ref is None:
            self.ref = 0.0
        self.dot = 0.0
        return float(self.ref), 0.0

    def step(
        self,
        d_star_target: float,
        dt: float,
        *,
        rate_m_s: float,
        hold: bool,
        d_live: float | None = None,
    ) -> tuple[float, float]:
        if self.ref is None:
            seed = float(d_live) if d_live is not None and np.isfinite(d_live) else float(
                d_star_target
            )
            self.init_from_live(seed)
            return float(self.ref), 0.0
        if hold or dt <= 1.0e-12:
            self.dot = 0.0
            return float(self.ref), 0.0
        target = float(d_star_target)
        if not np.isfinite(target):
            self.dot = 0.0
            return float(self.ref), 0.0
        lim = abs(float(rate_m_s)) * float(dt)
        err = target - float(self.ref)
        if lim <= 1.0e-15:
            delta = 0.0
        else:
            delta = lim * math.tanh(err / lim)
        self.ref = float(self.ref) + delta
        self.dot = delta / float(dt)
        return float(self.ref), float(self.dot)


class RailCommandMixer:
    """PI + hard clip + static mix with anti-windup vs u_pi_raw.

    Unsaturated and unprojected: ``u_mid_applied == u_pi_raw``, so Kaw is
    strictly zero.  PI saturation / share projection / ordinary velocity
    bounds still back-calculate.  ``wall_pi_frozen`` freezes ``ξ`` (no
    integrate, no back-calc).  ``quiescent`` is telemetry only.
    """

    def __init__(
        self,
        *,
        kp: float = 1.2,
        ki: float = 0.80,
        u_mid_max: float = 0.12,
        kaw: float = 8.0,
        d_center_rate: float = 0.02,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.u_mid_max = float(u_mid_max)
        self.kaw = float(kaw)
        self.d_center_rate = float(d_center_rate)
        self.xi = 0.0
        self.escape_dir = 0
        self.d_star = DStarRef()
        self.last = RailMixTelemetry()
        self.wall_pi_frozen = False

    def reset(self, d_live: float | None = None) -> None:
        self.xi = 0.0
        self.escape_dir = 0
        self.d_star.reset()
        if d_live is not None and np.isfinite(d_live):
            self.d_star.init_from_live(float(d_live))
        self.last = RailMixTelemetry(
            d_star_ref=float(self.d_star.ref) if self.d_star.ref is not None else 0.0
        )
        self.wall_pi_frozen = False

    def track_applied(
        self,
        *,
        d_live: float,
        d_star_target: float,
        applied_rail_vel: float,
        dt: float,
    ) -> RailMixTelemetry:
        """Keep ``d*_ref`` on the live target while another path owns the rail.

        Integrator and mix terms stay as they are so a later ``step`` does
        not inherit a second state jump.  ``u_feasible`` records the
        velocity that was actually written.
        """
        del dt
        d_ref, d_dot = self.d_star.follow_target(
            float(d_star_target), d_live=float(d_live)
        )
        e_d = float(d_live) - float(d_ref)
        prev = self.last
        tel = RailMixTelemetry(
            d_star_ref=float(d_ref),
            d_star_dot_cmd=float(d_dot),
            e_d=float(e_d),
            V_d_proxy=0.5 * self.kp * e_d * e_d,
            xi=float(self.xi),
            u_pi_raw=float(prev.u_pi_raw),
            u_mid_cmd=float(prev.u_mid_cmd),
            u_post_raw=float(prev.u_post_raw),
            u_post_feasible=float(prev.u_post_feasible),
            u_mid_applied=float(prev.u_mid_applied),
            u_task_raw=float(prev.u_task_raw),
            u_task_feasible=float(prev.u_task_feasible),
            u_escape_raw=float(prev.u_escape_raw),
            u_escape_feasible=float(prev.u_escape_feasible),
            u_feasible=float(applied_rail_vel),
            u_base=float(applied_rail_vel),
        )
        self.last = tel
        return tel

    def step(
        self,
        *,
        d_live: float,
        d_star_target: float,
        u_task_raw: float,
        u_escape_raw: float,
        escape_explicit: bool,
        dt: float,
        u_max: float,
        leave_sign: float = 0.0,
        hold_d_star: bool = False,
        quiescent: bool = False,
        secondary_alpha: float = 1.0,
        posture_hold: bool = False,
        in_wall: bool = False,
    ) -> RailMixTelemetry:
        if float(leave_sign) * float(u_task_raw) > 1.0e-4:
            u_lo, u_hi = 0.0, 0.0
        else:
            u_lo, u_hi = wall_velocity_bounds(u_max, leave_sign)
        self.escape_dir = update_escape_dir(
            explicit_active=bool(escape_explicit),
            u_escape_raw=float(u_escape_raw),
            prev_dir=int(self.escape_dir),
        )
        guard_dir = int(self.escape_dir) if escape_explicit else 0
        hold = bool(hold_d_star)
        d_ref, d_dot = self.d_star.step(
            float(d_star_target),
            float(dt),
            rate_m_s=self.d_center_rate,
            hold=hold,
            d_live=float(d_live),
        )
        e_d = float(d_live) - float(d_ref)
        V = 0.5 * self.kp * e_d * e_d

        if bool(in_wall):
            self.wall_pi_frozen = True
        else:
            self.wall_pi_frozen = False

        tel = RailMixTelemetry(
            u_task_raw=float(u_task_raw),
            u_escape_raw=float(u_escape_raw),
            escape_active=1.0 if escape_explicit else 0.0,
            escape_dir=float(guard_dir),
            d_star_dot_cmd=float(d_dot),
            d_star_ref=float(d_ref),
            e_d=float(e_d),
            V_d_proxy=float(V),
        )

        from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import (
            soft_saturate,
        )

        alpha = 0.0 if bool(posture_hold) else float(np.clip(secondary_alpha, 0.0, 1.0))
        del quiescent
        u_pi_raw = self.kp * e_d + self.xi
        u_mid_cmd = soft_saturate(u_pi_raw, self.u_mid_max)
        u_post_raw = alpha * (u_mid_cmd - float(d_dot))
        tel.xi = float(self.xi)
        tel.u_pi_raw = float(u_pi_raw)
        tel.u_mid_cmd = float(u_mid_cmd)
        tel.u_post_raw = float(u_post_raw)

        shares = allocate_rail_shares(
            u_task_raw=float(u_task_raw),
            u_post_raw=float(u_post_raw),
            u_escape_raw=float(u_escape_raw),
            escape_dir=guard_dir,
            u_lo=u_lo,
            u_hi=u_hi,
        )
        u_post_f = float(shares["u_post_feasible"])
        u_mid_applied = u_post_f + float(d_dot)
        if (not self.wall_pi_frozen) and dt > 0.0:
            if alpha < 1.0e-6:
                self.xi = -self.kp * e_d
            else:
                self.xi += (
                    self.ki * e_d + self.kaw * (u_mid_applied - u_pi_raw)
                ) * float(dt)
                self.xi = (1.0 - alpha) * (-self.kp * e_d) + alpha * self.xi
            tel.xi = float(self.xi)

        tel.u_task_feasible = float(shares["u_task_feasible"])
        tel.u_escape_feasible = float(shares["u_escape_feasible"])
        tel.u_base = float(shares["u_base"])
        tel.u_post_feasible = u_post_f
        tel.u_feasible = float(shares["u_feasible"])
        tel.u_mid_applied = float(u_mid_applied)
        self.last = tel
        return tel


def project_lpf_into_wall(v: float, leave_sign: float) -> float:
    """Zero (or keep leave-direction) LPF / v_r_ref state that points into a wall."""
    s = float(leave_sign)
    x = float(v)
    if s > 0.0 and x > 0.0:
        return 0.0
    if s < 0.0 and x < 0.0:
        return 0.0
    return x


def press_escape_allowed_from_flags(
    *,
    demanding: bool,
    has_travel: bool,
    press_stalled: bool,
    j4_blocked: bool,
    arm_starved: bool,
    policy_leave: bool,
) -> bool:
    """Final press-escape grant.  Inputs are already-evaluated booleans."""
    if not bool(demanding) or not bool(has_travel):
        return False
    return bool(
        press_stalled
        or (j4_blocked and not policy_leave)
        or (policy_leave and arm_starved)
    )


def soft_rail_travel(
    urdf_lo: float,
    urdf_hi: float,
    soft_min: float,
    soft_max: float,
) -> tuple[float, float]:
    lo = max(float(urdf_lo), float(soft_min))
    hi = min(float(urdf_hi), float(soft_max))
    if not (lo < hi):
        return float(urdf_lo), float(urdf_hi)
    return lo, hi


def leave_margin_m(escape_leave: float, pin_margin: float) -> float:
    return max(float(escape_leave), float(pin_margin))


def policy_escape_sign(
    policy: str,
    y: float,
    lo: float,
    hi: float,
    *,
    latched_sign: float = 0.0,
) -> float:
    raw = str(policy).strip().lower()
    if raw in ("minus", "-", "neg", "negative"):
        return -1.0
    if raw in ("plus", "+", "pos", "positive"):
        return 1.0
    if abs(float(latched_sign)) > 1.0e-9:
        return 1.0 if float(latched_sign) > 0.0 else -1.0
    if not np.isfinite(float(y)):
        return 0.0
    plus_room = float(hi) - float(y)
    minus_room = float(y) - float(lo)
    if plus_room > minus_room + 1.0e-9:
        return 1.0
    if minus_room > plus_room + 1.0e-9:
        return -1.0
    return 0.0


def in_leave_band(
    y: float,
    lo: float,
    hi: float,
    leave: float,
    sign: float,
) -> bool:
    if float(sign) > 0.0:
        return bool(float(y) >= float(hi) - float(leave))
    if float(sign) < 0.0:
        return bool(float(y) <= float(lo) + float(leave))
    return False


def q_star_srs_valid(
    q_star: np.ndarray | None,
    *,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    ik_ok: bool = True,
    halfplane_ok: bool = True,
) -> bool:
    if not ik_ok or not halfplane_ok or q_star is None:
        return False
    q = np.asarray(q_star, dtype=float).reshape(-1)
    lo = np.asarray(q_lo, dtype=float).reshape(-1)
    hi = np.asarray(q_hi, dtype=float).reshape(-1)
    if q.size == 0 or q.size != lo.size or q.size != hi.size:
        return False
    if not np.all(np.isfinite(q)):
        return False
    return bool(np.all(q >= lo - 1.0e-9) and np.all(q <= hi + 1.0e-9))


__all__ = (
    "DStarRef",
    "EPS_ENTER",
    "EPS_EXIT",
    "RailCommandMixer",
    "RailMixTelemetry",
    "allocate_rail_shares",
    "clip",
    "in_leave_band",
    "j4_index",
    "leave_margin_m",
    "policy_escape_sign",
    "press_escape_allowed_from_flags",
    "project",
    "project_lpf_into_wall",
    "q_star_srs_valid",
    "soft_rail_travel",
    "update_escape_dir",
    "wall_velocity_bounds",
)
