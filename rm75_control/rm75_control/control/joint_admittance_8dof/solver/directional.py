"""One task progress variable and a nominal rail/arm compatibility preview.

The preview is kinematic, not a prediction of hardware response. The current
task still uses measured rail motion; its future command has its own arm
compensation witness. Both HQP levels retain this witness and its constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DirectionalTask:
    base_size: int
    nv: int
    J: np.ndarray
    J_task: np.ndarray
    requested: np.ndarray
    rail_contribution: np.ndarray
    rail_velocity: float
    lo_box: np.ndarray
    hi_box: np.ndarray
    q: np.ndarray
    q_lower: np.ndarray
    q_upper: np.ndarray
    a_max: np.ndarray
    j_max: np.ndarray | None
    previous: np.ndarray
    dt: float
    horizon: float
    cbf_jac: np.ndarray
    cbf_lower: np.ndarray
    preview: bool = True

    @property
    def alpha(self):
        return self.base_size

    @property
    def arm(self):
        return slice(self.base_size + 1, self.base_size + self.nv)

    @property
    def arm_hold(self):
        return slice(self.base_size + self.nv, self.base_size + 2 * self.nv - 1)

    @property
    def size(self):
        return self.base_size + 2 * self.nv - 1

    def inequalities(self, C, lo, hi, *, progress=None):
        rows = [np.pad(C, ((0, 0), (0, self.size - self.base_size)))]
        lows, highs = [np.asarray(lo)], [np.asarray(hi)]

        def add(row, lower, upper):
            rows.append(np.atleast_2d(row))
            lows.append(np.atleast_1d(lower))
            highs.append(np.atleast_1d(upper))

        row = np.zeros(self.size)
        row[self.alpha] = 1.0
        add(row, 0.0 if progress is None else progress,
            1.0 if progress is None else progress)
        # No independent Cartesian slack can purchase a change of direction.
        pins = np.zeros((6, self.size))
        pins[:, self.nv:self.nv + 6] = np.eye(6)
        add(pins, np.zeros(6), np.zeros(6))
        m = self.nv - 1
        box = np.zeros((m, self.size))
        box[:, self.arm] = np.eye(m)
        if not self.preview:
            add(box, np.zeros(m), np.zeros(m))
        else:
            h = max(float(self.horizon), float(self.dt), 1e-6)
            lower = np.maximum(self.lo_box[1:], (self.q_lower[1:] - self.q[1:]) / h)
            upper = np.minimum(self.hi_box[1:], (self.q_upper[1:] - self.q[1:]) / h)
            add(box, lower, upper)
            # The same steady compensation must also be reachable on the
            # next ARM tick. Otherwise greedily reaching alpha=1 with a large
            # acceleration can make the very next tick infeasible under jerk.
            dt = max(float(self.dt), 1e-6)
            for transition in sorted({dt, h}):
                accel = box.copy()
                accel[:, 1:self.nv] = -np.eye(m)
                add(accel, -self.a_max[1:] * transition, self.a_max[1:] * transition)
                if self.j_max is not None:
                    jerk = box.copy()
                    jerk[:, 1:self.nv] = -(1.0 + transition / dt) * np.eye(m)
                    center = -transition / dt * self.previous[1:]
                    radius = self.j_max[1:] * transition * 0.5 * (transition + dt)
                    add(jerk, center - radius, center + radius)
            if self.cbf_jac.size:
                collision = np.zeros((len(self.cbf_lower), self.size))
                collision[:, 0] = self.cbf_jac[:, 0]
                collision[:, self.arm] = self.cbf_jac[:, 1:]
                add(collision, self.cbf_lower, np.full(len(self.cbf_lower), np.inf))
        # A low-rate rail can retain its present motion until the next write.
        # Reserve a next-ARM-tick witness for that held contribution as well
        # as the candidate above. Otherwise QP1 can buy progress by requiring
        # a rail change before the worker has actually refreshed it.
        hold_box = np.zeros((m, self.size))
        hold_box[:, self.arm_hold] = np.eye(m)
        if not self.preview:
            add(hold_box, np.zeros(m), np.zeros(m))
        else:
            dt = max(float(self.dt), 1e-6)
            add(hold_box, self.lo_box[1:], self.hi_box[1:])
            accel = hold_box.copy()
            accel[:, 1:self.nv] = -np.eye(m)
            add(accel, -self.a_max[1:] * dt, self.a_max[1:] * dt)
            if self.j_max is not None:
                jerk = hold_box.copy()
                jerk[:, 1:self.nv] = -2.0 * np.eye(m)
                add(jerk, -self.previous[1:] - self.j_max[1:] * dt * dt,
                    -self.previous[1:] + self.j_max[1:] * dt * dt)
            if self.cbf_jac.size:
                collision = np.zeros((len(self.cbf_lower), self.size))
                collision[:, self.arm_hold] = self.cbf_jac[:, 1:]
                add(collision, self.cbf_lower - self.cbf_jac[:, 0] * self.rail_velocity,
                    np.full(len(self.cbf_lower), np.inf))
        return np.vstack(rows), np.concatenate(lows), np.concatenate(highs)

    def equations(self, *, locked_velocity=None, progress=None):
        n = 18 if self.preview else 6
        A, b = np.zeros((n, self.size)), np.zeros(n)
        A[:6, :self.nv] = self.J_task
        if locked_velocity is None:
            A[:6, self.alpha] = -self.requested
            b[:6] = -self.rail_contribution
        else:
            b[:6] = locked_velocity
        if self.preview:
            A[6:12, 0] = self.J[:, 0]
            A[6:12, self.arm] = self.J[:, 1:]
            A[6:12, self.alpha] = -self.requested
            A[12:, self.arm_hold] = self.J[:, 1:]
            A[12:, self.alpha] = -self.requested
            b[12:] = -self.rail_contribution
        return A, b

    def first_cost(self):
        H, g = np.zeros((self.size, self.size)), np.zeros(self.size)
        H[self.alpha, self.alpha] = 1.0
        g[self.alpha] = -1.0
        return H, g

    def secondary_cost(self, H, g):
        extra = self.size - self.base_size
        return np.pad(H, ((0, extra), (0, extra))), np.pad(g, (0, extra))
