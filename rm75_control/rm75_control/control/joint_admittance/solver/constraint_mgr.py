"""Per-tick inequality constraints for the WBC QP inner loop.

Joint velocity box (velocity / position look-ahead / acceleration) plus optional
CBF self-collision rows stacked into ProxQP's l <= C x <= u form.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.solver.cbf_constraints import CbfRows
from rm75_control.control.joint_admittance.utils.safety import SafetyLimits


class VelocityBoxConstraints:
    def __init__(self, limits: SafetyLimits, *, damper_band_rad: float = 0.15) -> None:
        self.lim = limits
        # Faverjon/Tournassoud velocity-damper influence zone before each
        # (margin-backed) joint limit; see bounds() below.
        self.damper_band_rad = float(damper_band_rad)

    def bounds(
        self,
        q: np.ndarray,
        dt: float,
        qdot_prev: np.ndarray | None = None,
        *,
        q_meas: np.ndarray | None = None,
        resync_err: float = 0.0,
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
        # inside the box.  Applied BEFORE the position look-ahead stage so a
        # margin-overshoot recovery (position stage collapsing the box onto a
        # push-back velocity) keeps priority over the damper.
        band = self.damper_band_rad
        if band > 1e-9:
            d_hi = np.clip(((lim.q_upper - m) - q) / band, 0.0, 1.0)
            d_lo = np.clip((q - (lim.q_lower + m)) / band, 0.0, 1.0)
            hi = np.minimum(hi, lim.v_max * d_hi)
            lo = np.maximum(lo, -lim.v_max * d_lo)

        p_lo = (lim.q_lower + m - q) / dt
        p_hi = (lim.q_upper - m - q) / dt
        lo = np.maximum(lo, p_lo)
        hi = np.minimum(hi, p_hi)
        crossed = lo > hi
        if np.any(crossed):
            mid = 0.5 * (lo + hi)
            lo = np.where(crossed, mid, lo)
            hi = np.where(crossed, mid, hi)

        if lim.a_max is not None and qdot_prev is not None:
            qdot_prev = np.asarray(qdot_prev, dtype=float)
            a = lim.a_max * dt
            a_lo = np.maximum(lo, qdot_prev - a)
            a_hi = np.minimum(hi, qdot_prev + a)
            ok = a_lo <= a_hi
            lo = np.where(ok, a_lo, lo)
            hi = np.where(ok, a_hi, hi)

        if resync_err > 0.0 and q_meas is not None:
            # Faverjon-style command-lead damper (replaces the old hard
            # r_lo/r_hi intersection that collapsed lo==hi when lead exceeded
            # resync_err — a C0 break in the feasible set that produced
            # one-tick velocity steps and limit-cycle jitter at joint extremes).
            # As q_cmd's lead on q_meas approaches ±resync_err, the allowed
            # velocity in the direction that *increases* |lead| ramps smoothly
            # to 0; motion that reduces lead stays unconstrained.  The box
            # stays lo<=hi because we only shrink bounds from the outside.
            q_meas = np.asarray(q_meas, dtype=float)
            lead = q - q_meas
            band = max(resync_err * 0.5, 1e-6)
            d_hi = np.clip((resync_err - lead) / band, 0.0, 1.0)
            d_lo = np.clip((resync_err + lead) / band, 0.0, 1.0)
            hi = np.where(hi > 0.0, hi * d_hi, hi)
            lo = np.where(lo < 0.0, lo * d_lo, lo)

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
    for i in range(n_active):
        C[nv + i, :nv] = cbf.jacobian[i]
        l[nv + i] = cbf.lower[i]
    return C, l, u
