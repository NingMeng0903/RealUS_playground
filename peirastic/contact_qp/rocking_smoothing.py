"""Final published angular history, distinct from outer/nominal integrators."""
import math
import numpy as np

from .types import TwistConstraints

# Same numbering as the inner QPIK envelope: drop jerk, then acceleration,
# then the rate row. 4 leaves only the existing six-axis mechanical limits.
ROCKING_TIER_JERK = 1
ROCKING_TIER_ACCELERATION = 2
ROCKING_TIER_RATE = 3
ROCKING_TIER_MECHANICAL = 4
OUTER_ROCKING_RELAX_REASONS = frozenset((
    'solver_attempts_exhausted',
    'mechanical_rows_infeasible',
    'fixed_affine_component_outside_mechanical_interval',
    'final_hard_constraint_residual',
))


def rocking_interval(bounds, tier):
    """1 jerk, 2 acceleration, 3 rate, 4 mechanical-only, 0 disabled."""
    if tier in (0, ROCKING_TIER_MECHANICAL):
        return -float('inf'), float('inf')
    pairs = np.asarray(bounds, dtype=float).reshape(3, 2)[:4 - int(tier)]
    return float(pairs[:, 0].max()), float(pairs[:, 1].min())


def published_rocking_rows(face_axis_row, bounds, *, now_s, horizon_s, tier):
    """Hard tool-Y row at ``tier``, or empty constraints when the interval is open."""
    if int(tier) in (0, ROCKING_TIER_MECHANICAL):
        return TwistConstraints(), ROCKING_TIER_MECHANICAL
    low, high = rocking_interval(bounds, tier)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        return None, int(tier)
    return TwistConstraints(
        np.asarray(face_axis_row, dtype=float).reshape(1, 6),
        [low], [high], valid_until_s=float(now_s) + float(horizon_s),
        labels=('published_contact_rocking_speed_acc_jerk',),
    ), int(tier)


def first_admissible_rocking_tier(bounds, start=ROCKING_TIER_JERK):
    for tier in range(int(start), ROCKING_TIER_MECHANICAL + 1):
        low, high = rocking_interval(bounds, tier)
        if low <= high:
            return tier
    return ROCKING_TIER_MECHANICAL


class RockingSmoothing:
    policy = 'final_tool_y_v1'

    def __init__(self, velocity_limit, acceleration_limit, inertia, damping):
        values = (velocity_limit, acceleration_limit, inertia, damping)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError('positive existing rocking dynamics required')
        self.velocity_limit = velocity_limit
        self.acceleration_limit = acceleration_limit
        self.tau_s = inertia/damping
        self.jerk_limit = acceleration_limit/self.tau_s
        self.time_s = None
        self.omega_base = np.zeros(3)
        self.acceleration_base = np.zeros(3)

    def seed(self, omega_base, now_s):
        omega = np.asarray(omega_base, dtype=float).reshape(3)
        if not np.isfinite(omega).all() or not math.isfinite(now_s):
            raise ValueError('invalid final rocking seed')
        self.omega_base = omega.copy()
        self.acceleration_base = np.zeros(3)
        self.time_s = float(now_s)

    def preview(self, rotation_base_tcp, now_s, default_dt_s):
        rotation = np.asarray(rotation_base_tcp, dtype=float).reshape(3, 3)
        elapsed = default_dt_s if self.time_s is None else now_s-self.time_s
        dt = max(float(elapsed), float(default_dt_s))
        if not np.isfinite(rotation).all() or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('invalid final rocking frame/time')
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('invalid final rocking frame/time')
        axis = rotation[:, 1].copy()
        previous = float(axis @ self.omega_base)
        acceleration = float(axis @ self.acceleration_base)
        a = self.acceleration_limit
        # Keep an empty acceleration/jerk intersection empty. A preceding
        # mechanical override can exceed a_max; clipping here would falsely
        # label an abrupt return to a_max as jerk-constrained tier 1.
        # Floor the slew window at the control hold. A 1 ms proposal gap
        # otherwise collapses J*dt^2 by ~50x versus the real 7 ms cycle.
        jerk_acc = [acceleration-self.jerk_limit*dt,
                    acceleration+self.jerk_limit*dt]
        bounds = np.array([-self.velocity_limit, self.velocity_limit,
                           previous-a*dt, previous+a*dt,
                           previous+jerk_acc[0]*dt, previous+jerk_acc[1]*dt])
        return axis, bounds, dict(previous_rad_s=previous, previous_acceleration_rad_s2=acceleration,
                                 elapsed_s=float(elapsed), bound_dt_s=float(dt),
                                 jerk_limit_rad_s3=self.jerk_limit)

    def commit(self, final_tool, rotation_base_tcp, now_s):
        omega = np.asarray(rotation_base_tcp) @ np.asarray(final_tool)[3:]
        if not np.isfinite(omega).all() or not math.isfinite(now_s):
            raise ValueError('invalid final rocking publication')
        if self.time_s is not None:
            dt = now_s-self.time_s
            if dt <= 0: raise ValueError('final rocking publication time reversed')
            self.acceleration_base = (omega-self.omega_base)/dt
        self.omega_base = omega.copy()
        self.time_s = float(now_s)

    def publication_audit(self, final_tool, rotation_base_tcp, now_s, default_dt_s,
                          selected_tier, tolerance):
        """Measure the final command against its actual publication interval.

        Solver bounds precede transport. A timing-induced jerk miss is exposed
        here, never silently reported as a bounded actual-publication jerk and
        never used to undo a completed device send.
        """
        _, bounds, history = self.preview(rotation_base_tcp, now_s, default_dt_s)
        value = float(np.asarray(final_tool)[4])
        dt = history['elapsed_s']
        acceleration = (value-history['previous_rad_s'])/dt
        jerk = (acceleration-history['previous_acceleration_rad_s2'])/dt
        tier = selected_tier
        while tier < 4:
            pairs = np.asarray(bounds).reshape(3, 2)[:4-tier]
            lower, upper = float(np.max(pairs[:, 0])), float(np.min(pairs[:, 1]))
            if lower <= upper and lower-tolerance <= value <= upper+tolerance: break
            tier += 1
        return dict(actual_interval_s=dt, actual_acceleration_rad_s2=acceleration,
                    actual_jerk_rad_s3=jerk, actual_policy_tier=tier,
                    timing_limited=tier>selected_tier)

    @staticmethod
    def review(final_tool, facts, tolerance):
        """Audit the tier accepted by the mechanical solver; never clip a send."""
        tier = int(facts.get('rocking_policy_tier', 0))
        if tier == 4: return True
        if tier not in (1, 2, 3): return False
        lower, upper = facts.get('rocking_lower_rad_s', float('nan')), facts.get('rocking_upper_rad_s', float('nan'))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:return False
        value = float(np.asarray(final_tool)[4])
        return math.isfinite(value) and lower-tolerance <= value <= upper+tolerance
