"""Per-joint linear 3-state KF fusing q and SDK qdot. Rail locked → derivatives 0."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointKF:
    x: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = field(default_factory=lambda: np.diag([1e-6, 1e-3, 1.0]))
    q_pos: float = 1e-9
    q_vel: float = 1e-6
    q_acc: float = 5.0
    r_pos: float = 4e-8
    r_vel: float = 4e-6

    def predict(self, dt: float) -> None:
        T = float(dt)
        F = np.array([[1.0, T, 0.5 * T * T], [0.0, 1.0, T], [0.0, 0.0, 1.0]])
        q = np.diag([self.q_pos, self.q_vel, self.q_acc])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q

    def update(self, q: float, qdot: float | None) -> None:
        if qdot is None or not np.isfinite(qdot):
            H = np.array([[1.0, 0.0, 0.0]])
            z = np.array([float(q)])
            R = np.array([[self.r_pos]])
        else:
            H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            z = np.array([float(q), float(qdot)])
            R = np.diag([self.r_pos, self.r_vel])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P


@dataclass
class ArmJointObserver:
    n_arm: int = 7
    filters: list[JointKF] = field(default_factory=list)
    rail_locked: bool = True
    last_t: float | None = None

    def __post_init__(self) -> None:
        if not self.filters:
            self.filters = [JointKF() for _ in range(self.n_arm)]

    def step(
        self,
        t_s: float,
        q_arm: np.ndarray,
        qdot_sdk: np.ndarray | None,
        *,
        rail_q: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return q8, qd8, qdd8, qdd_var8 (rail derivatives 0 when locked)."""
        q_arm = np.asarray(q_arm, dtype=float).reshape(self.n_arm)
        dt = 0.005 if self.last_t is None else max(1e-4, float(t_s) - float(self.last_t))
        self.last_t = float(t_s)
        qd_sdk = None if qdot_sdk is None else np.asarray(qdot_sdk, dtype=float).reshape(-1)
        q = np.zeros(self.n_arm)
        qd = np.zeros(self.n_arm)
        qdd = np.zeros(self.n_arm)
        var = np.zeros(self.n_arm)
        for i, kf in enumerate(self.filters):
            kf.predict(dt)
            vmeas = None
            if qd_sdk is not None and i < qd_sdk.size and np.isfinite(qd_sdk[i]):
                vmeas = float(qd_sdk[i])
            kf.update(float(q_arm[i]), vmeas)
            q[i], qd[i], qdd[i] = kf.x
            var[i] = float(kf.P[2, 2])
        q8 = np.concatenate([[float(rail_q)], q])
        if self.rail_locked:
            qd8 = np.concatenate([[0.0], qd])
            qdd8 = np.concatenate([[0.0], qdd])
            var8 = np.concatenate([[0.0], var])
        else:
            qd8 = np.concatenate([[np.nan], qd])
            qdd8 = np.concatenate([[np.nan], qdd])
            var8 = np.concatenate([[np.inf], var])
        return q8, qd8, qdd8, var8


class DelayRing:
    """History of (t, q, qd, qdd) for delayed-state estimates ``x̂_{k-d|k}``."""

    def __init__(self, maxlen: int = 64) -> None:
        self.t: list[float] = []
        self.q: list[np.ndarray] = []
        self.qd: list[np.ndarray] = []
        self.qdd: list[np.ndarray] = []
        self.maxlen = int(maxlen)

    def push(self, t: float, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> None:
        self.t.append(float(t))
        self.q.append(np.asarray(q, dtype=float).copy())
        self.qd.append(np.asarray(qd, dtype=float).copy())
        self.qdd.append(np.asarray(qdd, dtype=float).copy())
        if len(self.t) > self.maxlen:
            self.t.pop(0)
            self.q.pop(0)
            self.qd.pop(0)
            self.qdd.pop(0)

    def at(self, t_query: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.t:
            z = np.zeros(8)
            return z, z, z
        t = np.asarray(self.t)
        i = int(np.searchsorted(t, t_query, side="right") - 1)
        i = int(np.clip(i, 0, len(self.t) - 1))
        if i + 1 < len(self.t) and t[i + 1] > t[i]:
            a = (t_query - t[i]) / (t[i + 1] - t[i])
            a = float(np.clip(a, 0.0, 1.0))
            q = (1 - a) * self.q[i] + a * self.q[i + 1]
            qd = (1 - a) * self.qd[i] + a * self.qd[i + 1]
            qdd = (1 - a) * self.qdd[i] + a * self.qdd[i + 1]
            return q, qd, qdd
        return self.q[i], self.qd[i], self.qdd[i]
