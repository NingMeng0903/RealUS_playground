"""Per-tick inequality constraints for the WBC QP inner loop.

Joint velocity box (velocity / position look-ahead / acceleration) plus optional
CBF self-collision rows stacked into ProxQP's l <= C x <= u form.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


def collapse_interval(
    lo: np.ndarray,
    hi: np.ndarray,
    qdot_prev: np.ndarray | None = None,
    a_max: np.ndarray | None = None,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse an empty velocity interval to a singleton feasible brake.

    When ``lo > hi``, set both bounds to one executable velocity: keep 0 if it
    lies strictly between the conflicting bounds, otherwise take the closest
    side that prefers braking toward the limit (matching command-lead
    behaviour).  Never raises.
    """
    lo = np.asarray(lo, dtype=float).copy()
    hi = np.asarray(hi, dtype=float).copy()
    crossed = lo > hi
    if not np.any(crossed):
        return lo, hi

    # hi < 0 < lo: the empty box straddles standstill — stop.
    keep_zero = crossed & (hi < 0.0) & (lo > 0.0)
    if qdot_prev is None:
        pick_lo = np.abs(lo) <= np.abs(hi)
        collapsed = np.where(pick_lo, lo, hi)
    else:
        prev = np.asarray(qdot_prev, dtype=float)
        # Moving positive: collapse onto lo (strongest brake of further +motion).
        # Moving negative: collapse onto hi.  Same rule as command_lead.
        collapsed = np.where(prev >= 0.0, lo, hi)
    collapsed = np.where(keep_zero, 0.0, collapsed)
    if (
        qdot_prev is not None
        and a_max is not None
        and dt is not None
        and float(dt) > 0.0
    ):
        prev = np.asarray(qdot_prev, dtype=float)
        a_step = np.asarray(a_max, dtype=float) * float(dt)
        collapsed = np.clip(collapsed, prev - a_step, prev + a_step)
    lo = np.where(crossed, collapsed, lo)
    hi = np.where(crossed, collapsed, hi)
    return lo, hi


def stopping_velocity(distance: np.ndarray, acceleration: np.ndarray, reaction_s: float) -> np.ndarray:
    """Maximum speed toward a limit while retaining delayed braking viability."""

    d = np.maximum(np.asarray(distance, dtype=float), 0.0)
    a = np.maximum(np.asarray(acceleration, dtype=float), 1.0e-9)
    reaction = np.maximum(np.asarray(reaction_s, dtype=float), 0.0)
    return np.sqrt(np.square(a * reaction) + 2.0 * a * d) - a * reaction


def wall_cap(
    x: float,
    *,
    lo: float,
    hi: float,
    a_max: float,
    reaction_s: float,
) -> tuple[float, float]:
    """One-sided speed limits toward each wall.  Never produces a restoring push."""

    v_out_lo = float(stopping_velocity(float(x) - float(lo), float(a_max), float(reaction_s)))
    v_out_hi = float(stopping_velocity(float(hi) - float(x), float(a_max), float(reaction_s)))
    return -v_out_lo, +v_out_hi


class VelocityBoxConstraints:
    def __init__(
        self,
        limits: SafetyLimits,
        *,
        damper_band_rad: float | np.ndarray = 0.15,
        rail_reaction_s: float = 0.06,
    ) -> None:
        self.lim = limits
        # Faverjon/Tournassoud velocity-damper influence zone before each
        # (margin-backed) joint limit; see bounds() below.  Scalar or per-joint
        # vector — units are per joint (rad for revolute, m for the prismatic
        # rail), so a scalar rad band must NOT be applied to the rail.
        self.damper_band_rad = np.asarray(damper_band_rad, dtype=float)
        # Extra look-ahead on the rail stopping envelope.  0 falls back to dt.
        self.rail_reaction_s = max(float(rail_reaction_s), 0.0)

    def bounds(
        self,
        q: np.ndarray,
        dt: float,
        qdot_prev: np.ndarray | None = None,
        *,
        q_meas: np.ndarray | None = None,
        q_cmd: np.ndarray | None = None,
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        qdot_prev2: np.ndarray | None = None,
        j_max: np.ndarray | None = None,
        box_dt: float | None = None,
        box_h1: float | None = None,
        box_h2: float | None = None,
        rail_lead_exempt: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        lim = self.lim
        q = np.asarray(q, dtype=float)
        # ``dt`` is the nominal period the command is integrated with (the
        # CANFD stream assumes a fixed one).  ``box_h1`` / ``box_h2`` are the
        # two most recent wall periods; rate limits describe physical motion
        # so they belong on wall time.  ``box_dt`` remains a one-period
        # fallback for older callers.
        if box_h1 is not None:
            a_dt = float(box_h1)
        else:
            a_dt = float(dt if box_dt is None else box_dt)
        h2 = float(box_h2) if box_h2 is not None else float("nan")

        lo = -lim.v_max.copy()
        hi = lim.v_max.copy()

        m = lim.position_margin
        q_cmd_arr = None
        if q_cmd is not None:
            q_cmd_arr = np.asarray(q_cmd, dtype=float)
            if q_cmd_arr.shape != q.shape or not np.all(np.isfinite(q_cmd_arr)):
                raise ValueError("q_cmd must be finite and match q")
        # Rail damper / stop envelope use the state closer to the wall.
        # Command lead or servo overshoot of a few millimetres otherwise
        # eats a 10 mm band before qdot can fall.
        q_rail_hi = float(q[0])
        q_rail_lo = float(q[0])
        if q_cmd_arr is not None:
            q_rail_hi = max(q_rail_hi, float(q_cmd_arr[0]))
            q_rail_lo = min(q_rail_lo, float(q_cmd_arr[0]))

        # Faverjon & Tournassoud (1987) velocity damper toward each joint
        # limit: the allowed speed TOWARD a limit ramps linearly to zero over
        # the last ``damper_band_rad`` before the (margin-backed) limit, while
        # motion AWAY stays unconstrained.  This replaces the old binary
        # "|u| > 0.95 -> zero bound" rule, which flipped the box between
        # +-v_max and 0 in a single tick and chattered against the soft
        # centering / arm-angle tasks whenever the nullspace parked a joint on
        # the threshold.  The ramp is continuous in q and always keeps 0
        # inside the box.  The damper never restricts motion AWAY from a
        # limit, so it can never block a margin recovery.
        band = np.broadcast_to(self.damper_band_rad, q.shape)
        if np.any(band > 1e-9):
            b = np.maximum(band, 1e-9)
            d_hi = np.clip(((lim.q_upper - m) - q) / b, 0.0, 1.0)
            d_lo = np.clip((q - (lim.q_lower + m)) / b, 0.0, 1.0)
            # Joints with band <= 0 keep the full velocity box.
            d_hi = np.where(band > 1e-9, d_hi, 1.0)
            d_lo = np.where(band > 1e-9, d_lo, 1.0)
            hi = np.minimum(hi, lim.v_max * d_hi)
            lo = np.maximum(lo, -lim.v_max * d_lo)
            # Rail linear taper uses the leading state so a few millimetres
            # of command lead / servo overshoot cannot skip the cone.
            m0 = float(np.broadcast_to(np.asarray(m, dtype=float).reshape(-1), q.shape)[0])
            b0 = float(np.broadcast_to(band, q.shape)[0])
            if b0 > 1e-9:
                d_hi[0] = float(
                    np.clip((float(lim.q_upper[0]) - m0 - q_rail_hi) / b0, 0.0, 1.0)
                )
                d_lo[0] = float(
                    np.clip((q_rail_lo - float(lim.q_lower[0]) - m0) / b0, 0.0, 1.0)
                )
                hi[0] = min(float(hi[0]), float(lim.v_max[0]) * float(d_hi[0]))
                lo[0] = max(float(lo[0]), -float(lim.v_max[0]) * float(d_lo[0]))

        m = np.broadcast_to(np.asarray(m, dtype=float), q.shape)
        a_max = None if lim.a_max is None else np.asarray(lim.a_max, dtype=float).copy()
        # Stopping envelope toward HARD travel (5/780).  30/755 is only the
        # Faverjon inner edge (full-speed start of the 25 mm band), not a
        # zero-velocity wall.  At v=0.12, a=0.60, τ=0.06 the envelope is
        # ~19 mm and stays inside that band.
        if a_max is not None and float(self.rail_reaction_s) > 0.0:
            m0 = float(m[0])
            hard_lo = float(lim.q_lower[0]) + m0
            hard_hi = float(lim.q_upper[0]) - m0
            if hard_hi > hard_lo:
                lo_cap, hi_cap = wall_cap(
                    float(q[0]),
                    lo=hard_lo,
                    hi=hard_hi,
                    a_max=float(a_max[0]),
                    reaction_s=float(self.rail_reaction_s),
                )
                lo_hi, hi_hi = wall_cap(
                    q_rail_hi,
                    lo=hard_lo,
                    hi=hard_hi,
                    a_max=float(a_max[0]),
                    reaction_s=float(self.rail_reaction_s),
                )
                lo_lo, hi_lo = wall_cap(
                    q_rail_lo,
                    lo=hard_lo,
                    hi=hard_hi,
                    a_max=float(a_max[0]),
                    reaction_s=float(self.rail_reaction_s),
                )
                hi[0] = min(float(hi[0]), hi_cap, hi_hi, hi_lo)
                lo[0] = max(float(lo[0]), lo_cap, lo_hi, lo_lo)

        p_lo = (lim.q_lower + m - q) / dt
        p_hi = (lim.q_upper - m - q) / dt
        # Rail hard box: past 5/780, one-tick look-ahead would require
        # returning by Δq/dt in a single period (reverse kick / chatter).
        # Kill into-wall only; leave stays open.
        rail_lo = float(lim.q_lower[0] + m[0])
        rail_hi = float(lim.q_upper[0] - m[0])
        if q[0] < rail_lo:
            p_lo[0] = min(float(p_lo[0]), 0.0)
        if q[0] > rail_hi:
            p_hi[0] = max(float(p_hi[0]), 0.0)
        lo = np.maximum(lo, p_lo)
        hi = np.minimum(hi, p_hi)
        lo, hi = collapse_interval(
            lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt
        )

        if a_max is not None and qdot_prev is not None:
            qdot_prev = np.asarray(qdot_prev, dtype=float)
            a = a_max * a_dt
            lo = np.maximum(lo, qdot_prev - a)
            hi = np.minimum(hi, qdot_prev + a)
            lo, hi = collapse_interval(
                lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt
            )

        # Third order.  Velocity and acceleration boxes still permit the
        # acceleration to flip sign every tick.  Bounding |a_k - a_{k-1}|
        # on unequal samples is
        #   qdot in qdot_prev + (h1/h2)(qdot_prev - qdot_prev2) +- j_max*h1^2
        # The equal-period form 2*qdot_prev - qdot_prev2 is recovered when
        # h1 == h2.  If h2 is unavailable (first tick / reset) the centre
        # stays at qdot_prev so only the acceleration box decides.
        if (
            j_max is not None
            and qdot_prev is not None
            and qdot_prev2 is not None
            and float(dt) > 0.0
        ):
            qdot_prev2 = np.asarray(qdot_prev2, dtype=float)
            if np.isfinite(h2) and h2 > 1.0e-9:
                centre = qdot_prev + (a_dt / h2) * (qdot_prev - qdot_prev2)
            else:
                centre = np.asarray(qdot_prev, dtype=float)
            span = np.asarray(j_max, dtype=float) * a_dt * a_dt
            lo = np.maximum(lo, centre - span)
            hi = np.minimum(hi, centre + span)
            lo, hi = collapse_interval(
                lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt
            )

        # Command lead is an anti-windup envelope, not a physical joint limit.
        # Start braking before |q_cmd-q_meas| reaches ``resync_err``.  If stale
        # tracking has already left too little distance for the acceleration
        # box, request the strongest acceleration-feasible braking velocity
        # instead of manufacturing an empty interval and stopping the robot.
        # ``resync_err`` is arm radians for joints 1..7 and metres for rail 0.
        if q_meas is not None:
            re = np.broadcast_to(
                np.asarray(resync_err, dtype=float), q.shape
            ).astype(float)
            active = re > 0.0
            if np.any(active):
                q_meas = np.asarray(q_meas, dtype=float)
                # Safety geometry is evaluated at measured q.  Command lead
                # is the one exception: compare the independently integrated
                # command state against the same measured snapshot.
                q_for_lead = q if q_cmd is None else np.asarray(q_cmd, dtype=float)
                lead = q_for_lead - q_meas
                # COUPLED rail velocity is authoritative; the 20 mm command
                # integrator lag is not a tracking error and must not freeze
                # the rail box.
                if rail_lead_exempt:
                    lead[0] = 0.0
                if a_max is None:
                    band = np.maximum(re * 0.5, 1.0e-6)
                    toward_hi = lim.v_max * np.clip((re - lead) / band, 0.0, 1.0)
                    toward_lo = -lim.v_max * np.clip((re + lead) / band, 0.0, 1.0)
                else:
                    reaction = np.full(q.shape, float(dt), dtype=float)
                    if float(self.rail_reaction_s) > 0.0:
                        reaction[0] = float(self.rail_reaction_s)
                    toward_hi = stopping_velocity(re - lead, a_max, reaction)
                    toward_lo = -stopping_velocity(re + lead, a_max, reaction)

                candidate_hi = np.minimum(hi, toward_hi)
                candidate_lo = np.maximum(lo, toward_lo)
                crossed = candidate_lo > candidate_hi
                # A positive lead must brake positive motion; a negative lead
                # must brake negative motion.  Collapse only the offending
                # side to the closest acceleration-feasible velocity.
                candidate_hi = np.where(
                    crossed & (lead >= 0.0), candidate_lo, candidate_hi
                )
                candidate_lo = np.where(
                    crossed & (lead < 0.0), candidate_hi, candidate_lo
                )
                hi = np.where(active, candidate_hi, hi)
                lo = np.where(active, candidate_lo, lo)
                lo, hi = collapse_interval(
                    lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt
                )

        if rail_vel_pin_m_s is not None:
            v = float(rail_vel_pin_m_s)
            if not np.isfinite(v):
                raise ValueError("rail_vel_pin_m_s must be finite")
            # Plan ownership is subordinate to the already assembled safety
            # box; it may pin the closest executable velocity, never replace
            # velocity/position/acceleration/command-lead bounds.
            v_safe = float(np.clip(v, lo[0], hi[0]))
            lo[0] = v_safe
            hi[0] = v_safe
        elif rail_locked:
            eps = max(float(rail_lock_vel_eps_m_s), 0.0)
            previous = 0.0 if qdot_prev is None else float(qdot_prev[0])
            rail_acceleration = (
                float(a_max[0]) if a_max is not None else float("inf")
            )
            if abs(previous) <= eps and float(lo[0]) <= 0.0 <= float(hi[0]):
                target = 0.0
            elif np.isfinite(rail_acceleration):
                target = np.sign(previous) * max(
                    abs(previous) - rail_acceleration * dt, 0.0
                )
                target = float(np.clip(target, lo[0], hi[0]))
            else:
                target = float(np.clip(0.0, lo[0], hi[0]))
            lo[0] = target
            hi[0] = target

        return lo, hi


def build_wbc_inequalities(
    nv: int,
    n_task_slack: int,
    lo_box: np.ndarray,
    hi_box: np.ndarray,
    cbf: CbfRows,
    max_cbf_rows: int,
    *,
    n_pref_slack: int = 0,
    max_pref_rows: int = 0,
    pref_jacobian: np.ndarray | None = None,
    pref_slack_col: np.ndarray | None = None,
    pref_lower: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack qdot box + CBF + optional preference inequalities + pref-slack >= 0.

    Decision vector: ``x = [qdot(nv); w_task(n_task_slack); s_pref(n_pref_slack)]``.
    Inactive CBF / pref slots are ``l=-inf, u=+inf``.
    """
    n_in = nv + max_cbf_rows + max_pref_rows + n_pref_slack
    n_var = nv + n_task_slack + n_pref_slack
    C = np.zeros((n_in, n_var), dtype=float)
    C[:nv, :nv] = np.eye(nv)
    l = np.full(n_in, -np.inf, dtype=float)
    u = np.full(n_in, np.inf, dtype=float)
    l[:nv] = lo_box
    u[:nv] = hi_box

    n_active = cbf.jacobian.shape[0]
    if cbf.slot_index is not None and cbf.slot_index.size == n_active:
        for k in range(n_active):
            i = int(cbf.slot_index[k])
            if i < 0 or i >= max_cbf_rows:
                continue
            C[nv + i, :nv] = cbf.jacobian[k]
            l[nv + i] = cbf.lower[k]
    else:
        for i in range(min(n_active, max_cbf_rows)):
            C[nv + i, :nv] = cbf.jacobian[i]
            l[nv + i] = cbf.lower[i]

    pref_base = nv + max_cbf_rows
    if (
        max_pref_rows > 0
        and pref_jacobian is not None
        and pref_lower is not None
        and pref_slack_col is not None
    ):
        n_pref = min(int(pref_jacobian.shape[0]), max_pref_rows)
        for k in range(n_pref):
            C[pref_base + k, :nv] = pref_jacobian[k]
            s_idx = int(pref_slack_col[k])
            if 0 <= s_idx < n_pref_slack:
                C[pref_base + k, nv + n_task_slack + s_idx] = 1.0
            l[pref_base + k] = float(pref_lower[k])

    # Pref slacks are one-sided: s >= 0.
    slack_base = pref_base + max_pref_rows
    for k in range(n_pref_slack):
        C[slack_base + k, nv + n_task_slack + k] = 1.0
        l[slack_base + k] = 0.0
    return C, l, u


__all__ = [
    "VelocityBoxConstraints",
    "build_wbc_inequalities",
    "collapse_interval",
    "stopping_velocity",
]
