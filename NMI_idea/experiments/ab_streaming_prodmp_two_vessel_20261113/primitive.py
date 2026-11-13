"""Fit and recondition the lateral ProDMP from visual-servo demonstrations."""

from dataclasses import dataclass

import numpy as np

from NMI_idea.experiments.ab_leg_bifurcation_20261112.learned_prodmp import (
    LAMBDA,
    _FallbackIntegralBasis,
)


SAMPLE_DT_S = 0.02
FRAME_STRIDE = 5
BLOCK_STEPS = 50
BLOCKS = 4
TOTAL_STEPS = BLOCKS * BLOCK_STEPS
MAX_LATERAL_SPEED_MM_S = 18.0
MAX_LATERAL_ACCEL_MM_S2 = 100.0
LATERAL_BOUNDS_MM = (-30.0, 0.0)


@dataclass(frozen=True)
class Plan:
    step: int
    x: np.ndarray
    v: np.ndarray

    @property
    def goal(self):
        return float(self.x[-1])


class LateralPrimitive:
    """Integrated DMP basis with initial state and exact terminal constraints."""

    def __init__(self):
        basis = _FallbackIntegralBasis()
        self.grids = {}
        for steps in range(FRAME_STRIDE, BLOCK_STEPS + 1, FRAME_STRIDE):
            phase = np.linspace(0.0, 1.0, steps + 1)
            values, slopes = basis.evaluate(phase)
            self.grids[steps] = phase, values, slopes
        _, values, slopes = self.grids[BLOCK_STEPS]
        endpoint = np.vstack((values[-1], slopes[-1]))
        self.endpoint_inverse = np.linalg.pinv(endpoint)
        self.null = np.linalg.svd(endpoint, full_matrices=True)[2][2:].T

    @staticmethod
    def homogeneous(phase, start, velocity, goal, duration):
        delta = start - goal
        speed = duration * velocity
        decay = np.exp(-LAMBDA * phase)
        pos = goal + decay * ((1.0 + LAMBDA * phase) * delta + phase * speed)
        slope = decay * (speed - LAMBDA * phase * (LAMBDA * delta + speed))
        return pos, slope

    def fit(self, x, v):
        """Fit endpoint-preserving shape to observed positions and velocities."""
        phase, basis, slopes = self.grids[BLOCK_STEPS]
        duration = BLOCK_STEPS * SAMPLE_DT_S
        x0, v0, goal, vg = x[0], v[0], x[-1], v[-1]
        base, base_slope = self.homogeneous(phase, x0, v0, goal, duration)
        weights = self.endpoint_inverse @ [goal - base[-1], duration * vg - base_slope[-1]]
        design = np.vstack((basis, 0.35 * slopes))
        target = np.r_[x - base, 0.35 * (duration * v - base_slope)]
        shape, *_ = np.linalg.lstsq(design @ self.null, target - design @ weights, rcond=None)
        return self.null @ shape

    def condition(self, step, steps, x, v, goal, goal_v, shape):
        phase, basis, slopes = self.grids[steps]
        duration = steps * SAMPLE_DT_S
        pos, slope = self.homogeneous(phase, x, v, goal, duration)
        weights = self.endpoint_inverse @ [goal - pos[-1], duration * goal_v - slope[-1]]
        # Scale the remaining shape with squared duration to bound its acceleration.
        weights += (steps / BLOCK_STEPS) ** 2 * shape
        return Plan(step, pos + basis @ weights, (slope + slopes @ weights) / duration)


def admissible(plan):
    return bool(
        np.min(plan.x) >= LATERAL_BOUNDS_MM[0]
        and np.max(plan.x) <= LATERAL_BOUNDS_MM[1]
        and np.max(np.abs(plan.v)) <= MAX_LATERAL_SPEED_MM_S
        and np.max(np.abs(np.diff(plan.v))) <= MAX_LATERAL_ACCEL_MM_S2 * SAMPLE_DT_S
    )


def revise(primitive, step, block_end, x, v, goal, goal_v, shape, previous, nominal_change):
    """Change the goal from the current state, preserving a feasible suffix."""
    remaining = block_end - step
    if previous is not None and abs(previous.goal - goal) < 1.0e-6:
        return previous
    safe_goal = previous.goal if previous is not None else x + nominal_change
    for fraction in (1.0, 0.5, 0.25, 0.0):
        if fraction == 0.0 and previous is not None:
            return previous
        next_goal = safe_goal + fraction * (goal - safe_goal)
        plan = primitive.condition(step, remaining, x, v, next_goal, goal_v, shape)
        if admissible(plan):
            return plan
    raise RuntimeError("no admissible lateral reference at a new block boundary")
