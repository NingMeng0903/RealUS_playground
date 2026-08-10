"""Per-tick inequality constraints for the WBC QP inner loop.

Joint velocity box (velocity / position look-ahead / acceleration) plus optional
CBF self-collision rows stacked into ProxQP's l <= C x <= u form.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows
from rm75_control.control.joint_admittance_8dof.utils.safety import (
    RAIL_ESCAPE_ACCEL_M_S2,
    SafetyLimits,
)


def _collapse_to(
    lo: np.ndarray,
    hi: np.ndarray,
    keep_lo: np.ndarray,
    keep_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve an empty ``[lo, hi]`` by projecting onto ``[keep_lo, keep_hi]``.

    ``lo > hi`` means the stage just applied is infeasible against the
    higher-priority interval ``[keep_lo, keep_hi]``.  The requested value is
    whichever endpoint caused the crossing (``lo`` when the new stage pushed
    the floor up, ``hi`` when it pushed the ceiling down); clamping it back
    into the priority interval keeps the intent of the lower-priority stage
    while guaranteeing the returned box is always executable.

    Rows that are already feasible are returned untouched.
    """
    crossed = lo > hi
    if not np.any(crossed):
        return lo, hi
    # ``lo`` rose above ``hi``: the new stage wants at least ``lo``.
    # Otherwise ``hi`` fell below ``lo``: the new stage wants at most ``hi``.
    want = np.where(lo > keep_hi, lo, hi)
    # ``+ 0.0`` normalises -0.0 so a collapsed box compares as lo == hi.
    pinned = np.clip(want, keep_lo, keep_hi) + 0.0
    return np.where(crossed, pinned, lo), np.where(crossed, pinned, hi)


class VelocityBoxConstraints:
    def __init__(
        self,
        limits: SafetyLimits,
        *,
        damper_band_rad: float | np.ndarray = 0.15,
    ) -> None:
        self.lim = limits
        # Faverjon/Tournassoud velocity-damper influence zone before each
        # (margin-backed) joint limit; see bounds() below.  Scalar or per-joint
        # vector — units are per joint (rad for revolute, m for the prismatic
        # rail), so a scalar rad band must NOT be applied to the rail.
        self.damper_band_rad = np.asarray(damper_band_rad, dtype=float)

    def bounds(
        self,
        q: np.ndarray,
        dt: float,
        qdot_prev: np.ndarray | None = None,
        *,
        q_meas: np.ndarray | None = None,
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        rail_escape_active: bool = False,
        rail_escape_sign: float = 0.0,
        rail_escape_stop: bool = False,
        rail_escape_accel_m_s2: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lim = self.lim
        q = np.asarray(q, dtype=float)

        # Staged/prioritised clamp: v_max + position margin are hard safety
        # bounds and are always honoured.  a_max and the resync anti-windup
        # bound are secondary - each is applied only if it doesn't render the
        # box infeasible against the *previous* (higher-priority) stage; a
        # single combined "crossed -> discard everything" check would let a
        # transient accel/resync conflict silently drop the resync bound
        # (or worse, both) for the rest of the move, which is exactly what
        # let the command lead run away unbounded instead of saturating.
        #
        # When a stage IS infeasible against the previous one, the conflict is
        # resolved by PROJECTING onto the higher-priority interval (see
        # ``_collapse_to``), never by averaging the two.  An unclamped midpoint
        # silently inverts the documented priority: at the rail's 0 m end stop
        # it produced lo == hi == +0.925 m/s against v_max = 0.16 m/s (a forced
        # 6x over-speed the servo answers with Er-01), and an inbound rail
        # arriving at the stop was pinned at a negative velocity - i.e. forced
        # to keep driving INTO the stop - because a_max could not decelerate
        # inside the damper band.
        lo = -lim.v_max.copy()
        hi = lim.v_max.copy()

        m = lim.position_margin

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

        # v_max + damper is the *executable envelope*: every later stage is
        # projected back into it, so the returned box can never ask for a
        # velocity the joint cannot run or one that points into a hard stop.
        v_lo, v_hi = lo, hi

        # The hardware bridge uses a 0.8 m/s² slew while an escape episode is
        # signed (or braking at its travel stop).  Apply that override to the
        # rail only, and only while the episode is explicitly active.  This is
        # deliberately computed before the acceleration box so the resulting
        # interval remains intersected with the hard velocity/position box.
        escape_slew_active = bool(rail_escape_active) and (
            abs(float(rail_escape_sign)) >= 0.5 or bool(rail_escape_stop)
        )
        a_max = None if lim.a_max is None else np.asarray(lim.a_max, dtype=float).copy()
        if escape_slew_active:
            accel = (
                RAIL_ESCAPE_ACCEL_M_S2
                if rail_escape_accel_m_s2 is None
                else float(rail_escape_accel_m_s2)
            )
            if np.isfinite(accel) and accel >= 0.0 and q.shape[0] > 0:
                if a_max is None:
                    # Keep the ordinary arm joints unbounded when the
                    # caller disabled their acceleration stage.
                    a_max = np.full(q.shape, np.inf, dtype=float)
                a_max[0] = accel

        if a_max is not None and qdot_prev is not None:
            qdot_prev = np.asarray(qdot_prev, dtype=float)
            a = a_max * dt
            # a_max is secondary: honour it whenever it intersects the
            # envelope, drop it when it does not.  A rail decelerating into an
            # end stop cannot brake inside the damper band at a_max_rail
            # (0.3 m/s^2 needs ~17 mm from 0.1 m/s, band is 20 mm), so the
            # intersection goes empty exactly there — and "keep the envelope"
            # is the only safe answer.
            a_lo = np.maximum(v_lo, qdot_prev - a)
            a_hi = np.minimum(v_hi, qdot_prev + a)
            lo, hi = _collapse_to(a_lo, a_hi, v_lo, v_hi)

        # Position look-ahead, applied last so a margin recovery ramps up
        # under a_max instead of stepping to v_max.  Inside the margin the
        # push-back ``p_lo`` is margin/dt, routinely tens of times v_max: it
        # is a direction, not an achievable speed, so it is clamped into the
        # box the earlier stages left.
        p_lo = (lim.q_lower + m - q) / dt
        p_hi = (lim.q_upper - m - q) / dt
        lo, hi = _collapse_to(
            np.maximum(lo, p_lo), np.minimum(hi, p_hi), lo, hi
        )

        # Vectorised command-lead damper: resync_err is either scalar (legacy;
        # arm-only, radians) or an nv-vector with per-joint bounds — arm rad
        # for joints 1..7 and metres for joint 0 (rail).  Using a scalar rad
        # bound for the prismatic joint was a silent unit bug: 0.10 rad =
        # 100 mm of lead allowed on the rail, and the QP would happily plan
        # multiple centimetres ahead of the encoder before anti-windup engaged.
        if q_meas is not None:
            re = np.broadcast_to(
                np.asarray(resync_err, dtype=float), q.shape
            ).astype(float)
            active = re > 0.0
            if np.any(active):
                q_meas = np.asarray(q_meas, dtype=float)
                lead = q - q_meas
                band = np.maximum(re * 0.5, 1e-6)
                d_hi = np.clip((re - lead) / band, 0.0, 1.0)
                d_lo = np.clip((re + lead) / band, 0.0, 1.0)
                hi_new = np.where(hi > 0.0, hi * d_hi, hi)
                lo_new = np.where(lo < 0.0, lo * d_lo, lo)
                hi = np.where(active, hi_new, hi)
                lo = np.where(active, lo_new, lo)
                # Scaling a one-sided box toward 0 can push a bound past the
                # other one; keep the interval non-empty.
                lo, hi = _collapse_to(lo, hi, v_lo, v_hi)

        if rail_vel_pin_m_s is not None:
            v = float(rail_vel_pin_m_s)
            lo[0] = v
            hi[0] = v
        elif rail_locked:
            eps = max(float(rail_lock_vel_eps_m_s), 0.0)
            lo[0] = -eps
            hi[0] = eps

        return lo, hi


def build_wbc_inequalities(
    nv: int,
    n_slack: int,
    lo_box: np.ndarray,
    hi_box: np.ndarray,
    cbf: CbfRows,
    max_cbf_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack [I_nv, 0; J_cbf, 0] with box + CBF lower bounds.

    Returns C (n_in, nv+n_slack), l, u for l <= C x <= u.
    Inactive CBF slots are l=-inf, u=+inf.
    """
    n_in = nv + max_cbf_rows
    n_var = nv + n_slack
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
    return C, l, u
