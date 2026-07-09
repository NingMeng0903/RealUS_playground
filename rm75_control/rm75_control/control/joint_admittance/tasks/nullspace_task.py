"""Nullspace secondary task: joint centering + limit avoidance (Liegeois 1977).

Produces a desired joint velocity `qdot0` that the CLIK/QP core projects into the
nullspace of the primary Cartesian task, so it never perturbs TCP tracking.  It
uses the redundancy of the 7-DOF arm to (a) pull joints toward the middle of
their range and (b) repel them harder as they approach a limit.

The cost being descended is the classic Liegeois manipulability/limit criterion
    H(q) = 1/2 * sum_i w_i * ((q_i - q_mid_i) / half_range_i)^2
    qdot0 = -k * dH/dq
plus a smooth activation term that grows near the limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance.model import RobotKinematics


@dataclass
class NullspaceTaskConfig:
    k_center: float = 1.0        # centering velocity gain (rad/s per normalized unit)
    k_limit: float = 2.0         # extra repulsion gain near a limit
    activation: float = 0.8      # |u| beyond which limit repulsion ramps in (u in [-1,1])
    weights: np.ndarray | None = None   # optional per-joint weighting (len 7)
    # Centering target (rad). Defaults to the midpoint of each joint's position
    # limits, which for a symmetric elbow limit (e.g. J4 +-135deg) is 0deg - a
    # dead-straight arm. Set this to a natural "elbow bent" posture instead
    # (e.g. J4 ~ 90deg) so the redundant DOF doesn't fight the primary task by
    # trying to snap the elbow straight; see JointCenteringTask.__call__.
    q_nominal_rad: np.ndarray | None = None


class JointCenteringTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s)."""

    def __init__(
        self,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        cfg: NullspaceTaskConfig | None = None,
    ) -> None:
        self.q_lower = np.asarray(q_lower, dtype=float)
        self.q_upper = np.asarray(q_upper, dtype=float)
        self.cfg = cfg or NullspaceTaskConfig()
        # Geometric mid/half-range: ALWAYS from the true limits, used only for the
        # limit-repulsion term below - do not confuse with the centering target.
        self.q_mid = 0.5 * (self.q_lower + self.q_upper)
        self.half = 0.5 * (self.q_upper - self.q_lower)
        # guard against zero-range joints
        self.half = np.where(self.half > 1e-9, self.half, 1.0)
        # Centering target: nominal "comfortable" posture if given, else the
        # geometric mid (which, on a symmetric elbow limit, is a straight arm).
        self.q_target = (
            self.q_mid.copy()
            if self.cfg.q_nominal_rad is None
            else np.asarray(self.cfg.q_nominal_rad, dtype=float)
        )
        self.w = (
            np.ones_like(self.q_mid)
            if self.cfg.weights is None
            else np.asarray(self.cfg.weights, dtype=float)
        )

    @classmethod
    def from_kinematics(
        cls, kin: RobotKinematics, cfg: NullspaceTaskConfig | None = None
    ) -> "JointCenteringTask":
        return cls(kin.q_lower, kin.q_upper, cfg)

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        q = np.asarray(q_rad, dtype=float)

        # gradient-descent centering toward q_target (nominal posture, or geometric mid)
        u_target = (q - self.q_target) / self.half
        qdot0 = -cfg.k_center * self.w * u_target

        # smooth limit repulsion beyond activation band - always relative to the
        # TRUE joint range, independent of the centering target above.
        if cfg.k_limit > 0.0 and cfg.activation < 1.0:
            u_limit = (q - self.q_mid) / self.half     # normalized position in [-1, 1]
            span = max(1.0 - cfg.activation, 1e-6)
            over = np.clip((np.abs(u_limit) - cfg.activation) / span, 0.0, 1.0)
            qdot0 = qdot0 - cfg.k_limit * np.sign(u_limit) * (over * over)
        return qdot0
