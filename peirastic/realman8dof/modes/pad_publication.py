"""Pad-only angular bounds valid throughout a command's remaining lifetime.

The scan smoother is deliberately unchanged. The interval includes the turning
points of the quadratic jerk bounds, not just its two endpoints. A stopping
distance bound reserves enough velocity headroom to bring acceleration to zero.
"""
import math

import numpy as np


def dispatch_rocking_bounds(history, rotation, earliest_s, latest_s, *, minimum_period_s=0.):
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    if not np.isfinite(rotation).all():
        raise ValueError('invalid pad dispatch rotation')
    axis = rotation[:, 1]
    d0, d1 = earliest_s-history.time_s, latest_s-history.time_s
    if not 0 < d0 <= d1 or not math.isfinite(d1):
        raise ValueError('invalid pad dispatch window')
    v = float(axis @ history.omega_base)
    a = float(axis @ history.acceleration_base)
    j = history.jerk_limit
    # Reserve half the available jerk for braking; the remaining half absorbs
    # dispatch jitter. Limit acceleration so a future minimum-length window
    # can still admit braking at that reserved jerk. This is a tighter bound,
    # never a relaxation of the configured acceleration or jerk limits.
    braking_j = .5*j
    acceleration_limit = history.acceleration_limit
    if minimum_period_s > 0.:
        t = float(minimum_period_s)
        # Minimize (J*(t+w)^2 - Jb*t^2)/w over ALL future widths w>0,
        # rather than depending on a fragile fixed-duration send window.
        acceleration_limit = min(acceleration_limit,
            2*j*t*(1+math.sqrt(1-braking_j/j)))
    # The lower parabola is concave and the upper one convex.
    lower_times = [d0, d1, float(np.clip(a/(2*j), d0, d1))]
    upper_times = [d0, d1, float(np.clip(-a/(2*j), d0, d1))]
    lower = max(v+a*d-j*d*d for d in lower_times)
    upper = min(v+a*d+j*d*d for d in upper_times)
    # v_new + a_new^2/(2J) <= V for positive acceleration (symmetric
    # for negative acceleration). Largest |a_new| occurs at earliest send.
    def headroom(margin):
        margin = max(0., margin)
        return 2*margin/(math.sqrt(1+2*margin/(braking_j*d0*d0))+1)
    viable_low = v-headroom(history.velocity_limit+v)
    viable_high = v+headroom(history.velocity_limit-v)
    bounds = np.array([
        max(-history.velocity_limit, viable_low),
        min(history.velocity_limit, viable_high),
        v-acceleration_limit*d0, v+acceleration_limit*d0,
        lower, upper,
    ])
    return axis.copy(), bounds
