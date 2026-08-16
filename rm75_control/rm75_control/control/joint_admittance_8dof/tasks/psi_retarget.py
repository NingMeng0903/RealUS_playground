"""One-shot min-max (d*, ψ*) planner for a known scan stroke.

Online hill-climb of instantaneous elbow margin is a double-well: both rail
ends score high and the interior (rail facing the TCP) scores low, so a
greedy climber parks the carriage on a stop.  For a periodic scan the
literature answer (Pin–Culioli minimax / Vahrenkamp ORM_tr) is to pick the
offset that maximises the *worst* joint margin over the whole stroke, then
hold it.

Call :meth:`PostureRetarget.plan_stroke` once when the scan starts.  After
that :meth:`step` only slews ψ toward ψ* with a single rate limit (no LPF)
and holds d* constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import (
    RAIL_INDEX,
    RobotKinematics,
    full_q_from_arm,
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


class StrokeInfeasibleError(RuntimeError):
    """Raised when no (d, ψ) covers the requested stroke inside rail travel."""


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _arm7(q_arm: np.ndarray) -> np.ndarray:
    q = np.asarray(q_arm, dtype=float).reshape(-1)
    return q[1:] if q.size == 8 else q


def joint_margin_frac(q_arm: np.ndarray) -> float:
    """Normalised per-joint slack in (0, 1]; return the worst joint."""
    q = _arm7(q_arm)
    half = 0.5 * (Q_UPPER - Q_LOWER)
    half = np.maximum(half, 1.0e-6)
    lo = (q - Q_LOWER) / half
    hi = (Q_UPPER - q) / half
    return float(np.min(np.minimum(lo, hi)))


def wrist_band_frac(
    q6: float,
    *,
    peak_rad: float = 45.0 * np.pi / 180.0,
) -> float:
    """1 at |q6|≈45°, 0 at a straight wrist and at the J6 stop."""
    a = abs(float(q6))
    q6_max = max(abs(float(Q_LOWER[5])), abs(float(Q_UPPER[5])), 1.0e-6)
    peak = min(max(float(peak_rad), 1.0e-6), q6_max)
    if a <= peak:
        return a / peak
    return max(0.0, 1.0 - (a - peak) / (q6_max - peak))


def arm_respects_floor(q_arm: np.ndarray, floor_rad: float) -> bool:
    """True iff every arm joint is at least ``floor_rad`` from a stop."""
    if float(floor_rad) <= 0.0:
        return True
    q = _arm7(q_arm)
    margin = np.minimum(q - Q_LOWER, Q_UPPER - q)
    return bool(np.all(margin >= float(floor_rad) - 1.0e-9))


def stroke_score(
    q_arm: np.ndarray,
    sigma: float,
    *,
    w_sigma: float,
    w_wrist: float,
) -> float:
    """One-shot cell score: worst-joint margin + σ + J6 band around 45°.

    ``|q6|/q6_max`` rewarded opening the wrist all the way to ±128° and
    parked J2 on a stop.  The band peaks at the yaml attractor (45°).
    """
    q = _arm7(q_arm)
    return (
        joint_margin_frac(q)
        + float(w_sigma) * float(sigma)
        + float(w_wrist) * wrist_band_frac(float(q[5]))
    )


@dataclass
class PsiRetargetConfig:
    enabled: bool = True
    n_y: int = 9
    n_d: int = 8
    n_psi: int = 9
    w_sigma: float = 0.5
    # Same scale as w_sigma.  Scores a 45° wrist band, not |q6|/q6_max.
    w_wrist: float = 0.5
    # Reject a (d, ψ) cell if any arm joint is closer than this to a stop.
    margin_floor_rad: float = 15.0 * np.pi / 180.0
    # Kept for yaml compatibility.  step() never replans; 0 is the contract.
    z_replan_m: float = 0.0
    # Used only when ψ* changes (new scan segment).  No LPF on top.
    psi_rate_rad_s: float = 20.0 * np.pi / 180.0
    # Soft travel used by the planner (must cover the whole stroke).
    rail_margin_m: float = 0.02
    # Reject a cell whose wrist sits on the branch-barrier floor (~15°).
    wrist_min_rad: float = 25.0 * np.pi / 180.0


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
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
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
    """Stroke min-max planner; ``step`` holds (d*, ψ*) after ``plan_stroke``."""

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
        self._psi_star: float | None = None
        self._d_star: float | None = None
        self._planned: bool = False
        self._z_plan: float = float("nan")
        self._y_center_m: float = float("nan")
        self._amplitude_m: float = float("nan")
        self._rail_lo: float = float("nan")
        self._rail_hi: float = float("nan")
        self.last_psi_score: float = float("nan")
        self.last_dpref_score: float = float("nan")
        self.last_minmax_margin: float = float("nan")
        self.last_elbow_margin_rad: float = float("nan")
        self.last_wrist_open_rad: float = float("nan")
        self.d_star_m: float = float("nan")
        self.psi_star_rad: float = float("nan")
        self._ird = None

    @property
    def planned(self) -> bool:
        return bool(self._planned)

    def reset(self, q_rad: np.ndarray) -> None:
        q = np.asarray(q_rad, dtype=float)
        psi = float(psi_from_q(q))
        self._psi_cmd = psi
        self._psi_star = psi
        y_tcp = float(self.kin.fk_placement(q).translation[1])
        d_pref = y_tcp - float(q[RAIL_INDEX])
        self._d_star = d_pref
        self._planned = False
        self._z_plan = float("nan")
        self.d_star_m = float(d_pref)
        self.psi_star_rad = float(psi)
        self.last_psi_score = float("nan")
        self.last_dpref_score = float("nan")
        self.last_minmax_margin = float("nan")
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

    def plan_stroke(
        self,
        q_rad: np.ndarray,
        *,
        y_center_m: float,
        amplitude_m: float,
        rail_lo: float,
        rail_hi: float,
    ) -> tuple[float, float]:
        """Grid-search ``(d*, ψ*)`` over the scan stroke.  Raises if empty."""
        q = np.asarray(q_rad, dtype=float)
        pose0 = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        branch = int(branch_from_q(q))
        amp = abs(float(amplitude_m))
        y_c = float(y_center_m)
        y_lo = y_c - amp
        y_hi = y_c + amp
        margin = max(float(self.cfg.rail_margin_m), 0.0)
        rail_lo_s = float(rail_lo) + margin
        rail_hi_s = float(rail_hi) - margin
        # y - d ∈ [rail_lo_s, rail_hi_s] for every y in the stroke.
        d_min = y_hi - rail_hi_s
        d_max = y_lo - rail_lo_s
        if d_min > d_max + 1.0e-9:
            raise StrokeInfeasibleError(
                f"scan stroke [{y_lo:.3f}, {y_hi:.3f}] m does not fit rail "
                f"[{rail_lo_s:.3f}, {rail_hi_s:.3f}] m; reduce amplitude"
            )
        n_y = max(int(self.cfg.n_y), 3)
        n_d = max(int(self.cfg.n_d), 3)
        n_psi = max(int(self.cfg.n_psi), 3)
        y_samples = np.linspace(y_lo, y_hi, n_y)
        d_grid = np.linspace(d_min, d_max, n_d)
        d_samples = d_grid
        if self._ird is not None and getattr(self._ird, "available", False):
            T_ird0 = self._ird.tcp_ird_from_q(self.kin, q)
            d_ird = self._ird.query_d_star(
                T_ird0,
                y_tcp0_m=float(pose0[1]),
                y_samples_m=y_samples,
                d_samples_m=d_grid,
                rail_lo=rail_lo_s,
                rail_hi=rail_hi_s,
            )
            if d_ird is not None and d_min - 1.0e-9 <= d_ird <= d_max + 1.0e-9:
                rails = y_samples - float(d_ird)
                if np.all(rails >= rail_lo_s - 1.0e-9) and np.all(
                    rails <= rail_hi_s + 1.0e-9
                ):
                    d_samples = np.array([float(d_ird)], dtype=float)
        psi0 = float(psi_from_q(q))
        psi_samples = psi0 + np.linspace(-np.pi, np.pi, n_psi, endpoint=False)
        w_sigma = float(self.cfg.w_sigma)
        w_wrist = float(self.cfg.w_wrist)
        floor = float(self.cfg.margin_floor_rad)

        def _search(d_list: np.ndarray) -> tuple[bool, float, float, float]:
            best_s = -np.inf
            best_dv = float(self._d_star if self._d_star is not None else 0.0)
            best_pv = psi0
            found = False
            for d in d_list:
                for psi in psi_samples:
                    worst = np.inf
                    feasible = True
                    last_q: np.ndarray | None = None
                    for y in y_samples:
                        y_rail = float(y) - float(d)
                        if y_rail < rail_lo_s - 1.0e-9 or y_rail > rail_hi_s + 1.0e-9:
                            feasible = False
                            break
                        pose = pose0.copy()
                        pose[1] = float(y)
                        pack = self._eval.evaluate(pose, float(psi), branch, y_rail)
                        if pack is None:
                            feasible = False
                            break
                        q_arm, q_full, sigma = pack
                        last_q = q_full
                        if not arm_respects_floor(q_arm, floor):
                            feasible = False
                            break
                        if abs(float(q_arm[5])) < float(self.cfg.wrist_min_rad) - 1.0e-9:
                            feasible = False
                            break
                        score_y = stroke_score(
                            q_arm, sigma, w_sigma=w_sigma, w_wrist=w_wrist
                        )
                        if score_y < worst:
                            worst = score_y
                    if not feasible or not np.isfinite(worst):
                        continue
                    found = True
                    if worst > best_s:
                        best_s = float(worst)
                        best_dv = float(d)
                        best_pv = float(psi)
                        if last_q is not None:
                            self._update_margins(last_q)
            return found, best_s, best_dv, best_pv

        any_feasible, best_score, best_d, best_psi = _search(d_samples)
        if not any_feasible and d_samples.size == 1 and d_grid.size > 1:
            any_feasible, best_score, best_d, best_psi = _search(d_grid)
        if not any_feasible:
            raise StrokeInfeasibleError(
                "no feasible (d, ψ) covers the scan stroke; reduce amplitude "
                "or choose a less extended start pose"
            )
        # The ψ grid is psi0 ± π, so best_psi can land outside (-π, π].  The
        # rate limiter already takes the short way round, but an unwrapped
        # value made psi_star_deg / |ψ − ψ_ref| unreadable in the CSV.
        best_psi = _wrap_pi(best_psi)
        self._d_star = float(best_d)
        self._psi_star = float(best_psi)
        self._planned = True
        self._z_plan = float(pose0[2])
        self._y_center_m = y_c
        self._amplitude_m = amp
        self._rail_lo = float(rail_lo)
        self._rail_hi = float(rail_hi)
        self.d_star_m = float(best_d)
        self.psi_star_rad = float(best_psi)
        self.last_minmax_margin = float(best_score)
        self.last_dpref_score = float(best_score)
        self.last_psi_score = float(best_score)
        if self._psi_cmd is None:
            self._psi_cmd = float(best_psi)
        return float(best_d), float(best_psi)

    def step(
        self,
        q_rad: np.ndarray,
        dt_s: float,
        *,
        rail_lo: float,
        rail_hi: float,
        q_nominal: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Hold a planned (d*, ψ*), or recapture d* and slew ψ toward q*."""
        del rail_lo, rail_hi
        q = np.asarray(q_rad, dtype=float)
        if self._psi_cmd is None or self._d_star is None:
            self.reset(q)
        dt = max(float(dt_s), 0.0)
        if not self._planned:
            y_tcp = float(self.kin.fk_placement(q).translation[1])
            self._d_star = y_tcp - float(q[RAIL_INDEX])
            self.d_star_m = float(self._d_star)
            if q_nominal is not None:
                self._psi_star = float(psi_from_q(np.asarray(q_nominal, dtype=float)))
            psi_out = self._rate_limit_psi(dt)
            self._update_margins(q)
            return float(psi_out), float(self._d_star)
        psi_out = self._rate_limit_psi(dt)
        self._update_margins(q)
        return float(psi_out), float(self._d_star)

    def nudge_d_star(
        self,
        delta_m: float,
        *,
        y_des_m: float,
        rail_lo: float,
        rail_hi: float,
    ) -> float:
        """Shift d* so rail_ff = y_des − d* stays inside the soft travel."""
        if self._d_star is None:
            return float("nan")
        y_des = float(y_des_m)
        lo = float(rail_lo)
        hi = float(rail_hi)
        d_lo = y_des - hi
        d_hi = y_des - lo
        if d_lo > d_hi:
            d_lo, d_hi = d_hi, d_lo
        d_new = float(np.clip(float(self._d_star) + float(delta_m), d_lo, d_hi))
        self._d_star = d_new
        self.d_star_m = d_new
        return d_new

    def _rate_limit_psi(self, dt_s: float) -> float:
        target = float(self._psi_star if self._psi_star is not None else 0.0)
        cur = float(self._psi_cmd if self._psi_cmd is not None else target)
        err = _wrap_pi(target - cur)
        max_step = float(self.cfg.psi_rate_rad_s) * dt_s
        if max_step > 0.0 and abs(err) > max_step:
            err = float(np.clip(err, -max_step, max_step))
        self._psi_cmd = float(cur + err)
        return float(self._psi_cmd)


__all__ = [
    "PostureRetarget",
    "PsiRetargetConfig",
    "StrokeInfeasibleError",
    "arm_respects_floor",
    "joint_margin_frac",
    "stroke_score",
    "wrist_band_frac",
]
