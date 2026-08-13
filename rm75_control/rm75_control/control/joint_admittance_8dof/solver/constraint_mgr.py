"""Per-tick inequality constraints for the WBC QP inner loop.

Joint velocity box (velocity / position look-ahead / acceleration) plus optional
CBF self-collision rows stacked into ProxQP's l <= C x <= u form.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


class VelocityBoxInfeasible(RuntimeError):
    """The physical per-joint velocity intervals have an empty intersection."""

    def __init__(self, stage: str, indices: np.ndarray, lo: np.ndarray, hi: np.ndarray):
        joints = ",".join(str(int(index)) for index in np.asarray(indices).reshape(-1))
        super().__init__(
            f"velocity viability conflict at {stage}; joints=[{joints}], "
            f"lo={np.asarray(lo)[indices].tolist()}, hi={np.asarray(hi)[indices].tolist()}"
        )
        self.stage = str(stage)
        self.indices = tuple(int(index) for index in np.asarray(indices).reshape(-1))


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


def _validate_interval(
    lo: np.ndarray,
    hi: np.ndarray,
    stage: str,
    *,
    qdot_prev: np.ndarray | None = None,
    a_max: np.ndarray | None = None,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Never raises: empty intersections collapse to a feasible brake."""
    del stage
    return collapse_interval(lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt)


def stopping_velocity(distance: np.ndarray, acceleration: np.ndarray, reaction_s: float) -> np.ndarray:
    """Maximum speed toward a limit while retaining delayed braking viability."""

    d = np.maximum(np.asarray(distance, dtype=float), 0.0)
    a = np.maximum(np.asarray(acceleration, dtype=float), 1.0e-9)
    reaction = max(float(reaction_s), 0.0)
    return np.sqrt(np.square(a * reaction) + 2.0 * a * d) - a * reaction


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
        q_cmd: np.ndarray | None = None,
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        qdot_prev2: np.ndarray | None = None,
        j_max: np.ndarray | None = None,
        box_dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lim = self.lim
        q = np.asarray(q, dtype=float)
        # ``dt`` is the nominal period the command is integrated with (the
        # CANFD stream assumes a fixed one).  ``box_dt`` is the wall period the
        # motion is actually played out over; the rate limits describe physical
        # motion, so they belong on wall time.  Defaults to ``dt``.
        a_dt = float(dt if box_dt is None else box_dt)

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

        m = np.broadcast_to(np.asarray(m, dtype=float), q.shape)
        a_max = None if lim.a_max is None else np.asarray(lim.a_max, dtype=float).copy()
        if a_max is not None:
            # Enter the braking envelope before acceleration and one-tick
            # position constraints can conflict.  Only speed toward a limit is
            # reduced; motion away remains available.
            d_upper = (lim.q_upper - m) - q
            d_lower = q - (lim.q_lower + m)
            hi = np.minimum(hi, stopping_velocity(d_upper, a_max, dt))
            lo = np.maximum(lo, -stopping_velocity(d_lower, a_max, dt))
            lo, hi = collapse_interval(
                lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt
            )

        p_lo = (lim.q_lower + m - q) / dt
        p_hi = (lim.q_upper - m - q) / dt
        lo = np.maximum(lo, p_lo)
        hi = np.minimum(hi, p_hi)
        lo, hi = collapse_interval(
            lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt
        )
        if q_cmd is not None:
            q_cmd_arr = np.asarray(q_cmd, dtype=float)
            if q_cmd_arr.shape != q.shape or not np.all(np.isfinite(q_cmd_arr)):
                raise ValueError("q_cmd must be finite and match q")
            cmd_lo = (lim.q_lower + m - q_cmd_arr) / dt
            cmd_hi = (lim.q_upper - m - q_cmd_arr) / dt
            lo = np.maximum(lo, cmd_lo)
            hi = np.minimum(hi, cmd_hi)
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
        # acceleration to flip sign every tick, which is what the commanded
        # joint trajectory actually did (sign reversals on ~50% of samples,
        # jerk RMS 250-570 rad/s^3) even though the reference twist was
        # smooth.  Bounding |a_k - a_{k-1}| is a plain velocity box:
        #   qdot in 2*qdot_prev - qdot_prev2 +- j_max*dt^2
        if (
            j_max is not None
            and qdot_prev is not None
            and qdot_prev2 is not None
            and float(dt) > 0.0
        ):
            qdot_prev2 = np.asarray(qdot_prev2, dtype=float)
            centre = 2.0 * qdot_prev - qdot_prev2
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
                if a_max is None:
                    band = np.maximum(re * 0.5, 1.0e-6)
                    toward_hi = lim.v_max * np.clip((re - lead) / band, 0.0, 1.0)
                    toward_lo = -lim.v_max * np.clip((re + lead) / band, 0.0, 1.0)
                else:
                    toward_hi = stopping_velocity(re - lead, a_max, dt)
                    toward_lo = -stopping_velocity(re + lead, a_max, dt)

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
    "VelocityBoxInfeasible",
    "build_wbc_inequalities",
    "collapse_interval",
    "stopping_velocity",
]
