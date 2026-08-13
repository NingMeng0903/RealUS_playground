"""Amortized online SRS retarget of arm-angle ψ and preferred rail extension.

The analytic SRS family is a 1-parameter (ψ) family at a *fixed* rail; elbow
J4 is a function of shoulder–wrist distance (i.e. the rail), not of ψ.  This
module therefore:

* hill-climbs ψ inside the *currently connected* feasible interval (never
  jumps an infeasible gap on the ψ-circle);
* hill-climbs the rail coordinate that maximises J4 margin + σ_min, then
  writes ``d_pref = y_tcp − y_rail★``.

Both searches share a 2-eval/tick budget so the 200 Hz QP stays inside its
ProxQP wall-clock.  Outputs are rate-limited and first-order filtered before
they reach :class:`ArmAngleTask` / :class:`RailExtensionTask`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import (
    RAIL_INDEX,
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.pose_ik import (
    PlannerGoalWeights,
    goal_score,
)
from rm75_control.kinematics.srs_ik import (
    Q_LOWER,
    Q_UPPER,
    branch_from_q,
    flange_tcp_from_kin,
    psi_from_q,
    shoulder_y_from_q_rail,
    srs_ik,
)


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _lpf(prev: float, target: float, dt_s: float, tau_s: float) -> float:
    if tau_s <= 1e-9 or dt_s <= 0.0:
        return float(target)
    alpha = float(dt_s) / (tau_s + float(dt_s))
    return (1.0 - alpha) * float(prev) + alpha * float(target)


@dataclass
class PsiRetargetConfig:
    enabled: bool = True
    evals_per_tick: int = 2
    psi_step_rad: float = 5.0 * np.pi / 180.0
    psi_rate_rad_s: float = 20.0 * np.pi / 180.0
    psi_lpf_tau_s: float = 0.30
    rail_step_m: float = 0.05
    d_pref_rate_m_s: float = 0.02
    d_pref_lpf_tau_s: float = 0.40
    weights: PlannerGoalWeights = field(
        default_factory=lambda: PlannerGoalWeights(
            w_home=0.05,
            w_sigma_floor=100.0,
            w_wrist=1.0,
            w_elbow=0.5,
        )
    )


class _SrsEval:
    """Cached flange TCP + one srs_ik + Jacobian/σ evaluation."""

    def __init__(self, kin: RobotKinematics) -> None:
        self.kin = kin
        self._R, self._t = flange_tcp_from_kin(kin)

    def evaluate(
        self,
        pose: np.ndarray,
        psi: float,
        branch: int,
        y_rail: float,
    ) -> tuple[np.ndarray, float, float] | None:
        q_arm = srs_ik(
            pose,
            float(psi),
            int(branch),
            y_rail=shoulder_y_from_q_rail(float(y_rail)),
            R_flange_tcp=self._R,
            t_flange_tcp=self._t,
        )
        if q_arm is None:
            return None
        q_full = full_q_from_arm(q_arm, rail_m=float(y_rail))
        sigma = float(self.kin.singular_values(self.kin.jacobian(q_full)).min())
        return q_arm, q_full, sigma


class PostureRetarget:
    """Shared-budget ψ + d_pref hill-climb; call ``step`` once per control tick."""

    def __init__(
        self,
        kin: RobotKinematics,
        cfg: PsiRetargetConfig | None = None,
        *,
        euler_order: str = "xyz",
    ) -> None:
        self.kin = kin
        self.cfg = cfg or PsiRetargetConfig()
        self.euler_order = str(euler_order)
        self._eval = _SrsEval(kin)
        self._psi_cmd: float | None = None
        self._psi_best: float | None = None
        self._d_pref_cmd: float | None = None
        self._d_pref_best: float | None = None
        self._rail_best: float | None = None
        self._psi_slot: int = 0
        self._rail_slot: int = 0
        self.last_psi_score: float = float("nan")
        self.last_dpref_score: float = float("nan")
        self.last_elbow_margin_rad: float = float("nan")
        self.last_wrist_open_rad: float = float("nan")

    def reset(self, q_rad: np.ndarray) -> None:
        q = np.asarray(q_rad, dtype=float)
        psi = float(psi_from_q(q))
        self._psi_cmd = psi
        self._psi_best = psi
        y_tcp = float(self.kin.fk_placement(q).translation[1])
        d_pref = y_tcp - float(q[RAIL_INDEX])
        self._d_pref_cmd = d_pref
        self._d_pref_best = d_pref
        self._rail_best = float(q[RAIL_INDEX])
        self._psi_slot = 0
        self._rail_slot = 0
        self.last_psi_score = float("nan")
        self.last_dpref_score = float("nan")
        self._update_margins(q)

    def _update_margins(self, q: np.ndarray) -> None:
        q_arm = np.asarray(q, dtype=float).reshape(-1)
        if q_arm.size == 8:
            q_arm = q_arm[1:]
        q4 = float(q_arm[3])
        q6 = float(q_arm[5])
        self.last_elbow_margin_rad = float(
            min(q4 - float(Q_LOWER[3]), float(Q_UPPER[3]) - q4)
        )
        self.last_wrist_open_rad = float(abs(q6))

    def step(
        self,
        q_rad: np.ndarray,
        dt_s: float,
        *,
        rail_lo: float,
        rail_hi: float,
    ) -> tuple[float, float]:
        """Return ``(psi_ref_rad, d_pref_m)`` after at most ``evals_per_tick`` IK solves."""
        q = np.asarray(q_rad, dtype=float)
        if self._psi_cmd is None or self._d_pref_cmd is None:
            self.reset(q)
        pose = self.kin.fk_pose(q)
        branch = int(branch_from_q(q))
        psi_meas = float(psi_from_q(q))
        y_rail = float(q[RAIL_INDEX])
        y_tcp = float(pose[1])
        budget = max(1, int(self.cfg.evals_per_tick))
        # Split: first eval ψ neighbour, remaining evals rail neighbours.
        n_psi = 1 if budget == 1 else max(1, budget - 1)
        n_rail = max(0, budget - n_psi)
        self._climb_psi(pose, branch, y_rail, psi_meas, n_psi)
        self._climb_rail(pose, branch, y_tcp, y_rail, rail_lo, rail_hi, n_rail)
        dt = max(float(dt_s), 0.0)
        psi_out = self._rate_limit_psi(dt)
        d_out = self._rate_limit_dpref(dt)
        self._update_margins(q)
        return psi_out, d_out

    def _climb_psi(
        self,
        pose: np.ndarray,
        branch: int,
        y_rail: float,
        psi_meas: float,
        n_eval: int,
    ) -> None:
        if n_eval <= 0:
            return
        step = float(self.cfg.psi_step_rad)
        center = float(self._psi_best if self._psi_best is not None else psi_meas)
        # Neighbours on the connected component: ±1 step, then ±2, cycling.
        offsets = [0.0, step, -step]
        scored: list[tuple[float, float]] = []
        used = 0
        start = self._psi_slot
        for k in range(len(offsets)):
            if used >= n_eval:
                break
            off = offsets[(start + k) % len(offsets)]
            psi = center + off
            pack = self._eval.evaluate(pose, psi, branch, y_rail)
            used += 1
            if pack is None:
                continue
            q_arm, q_full, sigma = pack
            psi_home = float(self._psi_cmd if self._psi_cmd is not None else psi_meas)
            score = goal_score(
                q_arm, q_full, float(psi), psi_home, sigma, self.kin, self.cfg.weights
            )
            scored.append((score, float(psi)))
        self._psi_slot = (start + used) % len(offsets)
        if not scored:
            return
        score, psi_star = max(scored, key=lambda t: t[0])
        self.last_psi_score = float(score)
        self._psi_best = float(psi_star)

    def _climb_rail(
        self,
        pose: np.ndarray,
        branch: int,
        y_tcp: float,
        y_rail: float,
        rail_lo: float,
        rail_hi: float,
        n_eval: int,
    ) -> None:
        if n_eval <= 0:
            return
        step = float(self.cfg.rail_step_m)
        center = float(self._rail_best if self._rail_best is not None else y_rail)
        psi = float(self._psi_best if self._psi_best is not None else 0.0)
        offsets = [0.0, step, -step]
        scored: list[tuple[float, float]] = []
        used = 0
        start = self._rail_slot
        for k in range(len(offsets)):
            if used >= n_eval:
                break
            y = float(np.clip(center + offsets[(start + k) % len(offsets)], rail_lo, rail_hi))
            pack = self._eval.evaluate(pose, psi, branch, y)
            used += 1
            if pack is None:
                continue
            q_arm, _q_full, sigma = pack
            q4 = float(q_arm[3])
            elbow_margin = min(q4 - float(Q_LOWER[3]), float(Q_UPPER[3]) - q4)
            score = float(elbow_margin) + 0.5 * float(sigma)
            scored.append((score, y))
        self._rail_slot = (start + used) % len(offsets)
        if not scored:
            return
        score, y_star = max(scored, key=lambda t: t[0])
        self.last_dpref_score = float(score)
        self._rail_best = float(y_star)
        self._d_pref_best = float(y_tcp) - float(y_star)

    def _rate_limit_psi(self, dt_s: float) -> float:
        target = float(self._psi_best if self._psi_best is not None else 0.0)
        cur = float(self._psi_cmd if self._psi_cmd is not None else target)
        err = _wrap_pi(target - cur)
        max_step = float(self.cfg.psi_rate_rad_s) * dt_s
        if abs(err) > max_step > 0.0:
            err = float(np.clip(err, -max_step, max_step))
        stepped = cur + err
        filtered = _lpf(cur, stepped, dt_s, float(self.cfg.psi_lpf_tau_s))
        self._psi_cmd = float(filtered)
        return float(self._psi_cmd)

    def _rate_limit_dpref(self, dt_s: float) -> float:
        target = float(self._d_pref_best if self._d_pref_best is not None else 0.0)
        cur = float(self._d_pref_cmd if self._d_pref_cmd is not None else target)
        max_step = float(self.cfg.d_pref_rate_m_s) * dt_s
        delta = target - cur
        if abs(delta) > max_step > 0.0:
            delta = float(np.clip(delta, -max_step, max_step))
        stepped = cur + delta
        filtered = _lpf(cur, stepped, dt_s, float(self.cfg.d_pref_lpf_tau_s))
        self._d_pref_cmd = float(filtered)
        return float(self._d_pref_cmd)


__all__ = [
    "PostureRetarget",
    "PsiRetargetConfig",
]
