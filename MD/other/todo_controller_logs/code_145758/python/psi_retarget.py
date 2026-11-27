"""One-shot min-max (d*, ψ*) planner for a known scan stroke.

Online hill-climb of instantaneous elbow margin is a double-well: both rail
ends score high and the interior (rail facing the TCP) scores low, so a
greedy climber parks the carriage on a stop.  For a periodic scan the
literature answer (Pin–Culioli minimax / Vahrenkamp ORM_tr) is to pick the
offset that maximises the *worst* joint margin over the whole stroke, then
hold it.

Call :meth:`PostureRetarget.plan_stroke` once when the scan starts.  After
that :meth:`step` only slews ψ toward ψ* with a single rate limit (no LPF)
and holds the planned d* constant.

Unplanned ``step`` homes ``(d*, ψ*, q*)`` on one progress ``s``.  ``T``
is the slower of the existing ψ and d rates; ``q*`` is ``srs_ik`` at the
current TCP (same branch), not the yaml photo at t=0.  Hunt ``d*`` /
``ψ*`` while moving; freeze ``hold_setpoint`` only when the command and
TCP are both quiet (or slack is high).  Local ψ search takes over only
while the wrist is collapsed and the elbow is still open (SEW is
undefined near the J4 floor).
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


def nearest_planar_psi(psi_rad: float) -> float:
    """Quantize swivel to the nearer SEW plane ``{0, ±π}``.

    The taught home (J1≈0, J6≈90°) sits at ψ=π; yaml ``q_nominal``
    (J6=45°) sits at ψ=0.  Those are opposite elbow orbits.  Snap once
    at reset so swivel returns to the start family, not the other plane.
    """
    a = _wrap_pi(float(psi_rad))
    if abs(a) <= 0.5 * np.pi:
        return 0.0
    # ±π are the same SEW plane; keep +π so CSV ψ* reads 180°.
    return float(np.pi)


def fold_psi_to_positive(psi_rad: float) -> float:
    """Map ψ into ``[0, π]`` so the one-sided envelope is well-defined.

    ``−π`` and ``+π`` are the same SEW plane; the negative half-plane is
    folded across 0 so the attractor never asks the arm to cross ψ = 0.
    """
    a = abs(_wrap_pi(float(psi_rad)))
    return min(a, float(np.pi))


def clamp_psi_to_envelope(
    psi_rad: float,
    lo_rad: float,
    hi_rad: float,
) -> float:
    """Fold onto the positive family, then clamp to ``[lo, hi] ⊂ (0, π)``."""
    lo = max(float(lo_rad), 1.0e-6)
    hi = min(float(hi_rad), float(np.pi) - 1.0e-6)
    if lo > hi:
        lo, hi = hi, lo
    return float(np.clip(fold_psi_to_positive(psi_rad), lo, hi))


def psi_err_avoiding_zero(cur_rad: float, target_rad: float) -> float:
    """Signed ψ error that never takes the short path through 0."""
    cur = _wrap_pi(float(cur_rad))
    target = _wrap_pi(float(target_rad))
    err = _wrap_pi(target - cur)
    nxt = cur + err
    if cur * nxt < 0.0 and abs(cur) < 0.5 * np.pi and abs(target) < 0.5 * np.pi:
        if err > 0.0:
            err -= 2.0 * np.pi
        else:
            err += 2.0 * np.pi
    return float(err)


# Half-width of the first ``plan_stroke`` search around the taught plane.
# Opposite-family search uses the same width only when this band is empty.
_PLAN_FAMILY_HALF_SPAN_RAD = 40.0 * np.pi / 180.0


def _arm7(q_arm: np.ndarray) -> np.ndarray:
    q = np.asarray(q_arm, dtype=float).reshape(-1)
    return q[1:] if q.size == 8 else q


def d_from_q(kin: RobotKinematics, q_rad: np.ndarray) -> float:
    """Arm Y-reach ``d = y_tcp − q0``.  Invariant to the rail coordinate."""
    q = np.asarray(q_rad, dtype=float).reshape(-1)
    if q.size == 7:
        q = np.concatenate([[0.0], q])
    if q.size != 8:
        raise ValueError(f"q must be length 7 or 8, got {q.size}")
    return float(kin.fk_placement(q).translation[1]) - float(q[RAIL_INDEX])


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
    peak_rad: float = 60.0 * np.pi / 180.0,
) -> float:
    """1 at |q6|≈45°, 0 at a straight wrist and at the J6 stop."""
    a = abs(float(q6))
    q6_max = max(abs(float(Q_LOWER[5])), abs(float(Q_UPPER[5])), 1.0e-6)
    peak = min(max(float(peak_rad), 1.0e-6), q6_max)
    if a <= peak:
        return a / peak
    return max(0.0, 1.0 - (a - peak) / (q6_max - peak))


def design_family_ok(
    q_meas: np.ndarray,
    q_nominal: np.ndarray,
    *,
    psi_tol_rad: float = 45.0 * np.pi / 180.0,
) -> bool:
    """True if measured q is the same SEW family as the design attractor."""
    qm = np.asarray(q_meas, dtype=float).reshape(-1)
    qn = np.asarray(q_nominal, dtype=float).reshape(-1)
    if qm.size == 7:
        qm = np.concatenate([[0.0], qm])
    if qn.size == 7:
        qn = np.concatenate([[0.0], qn])
    if qm.size != 8 or qn.size != 8:
        return False
    psi_m = fold_psi_to_positive(psi_from_q(qm))
    psi_n = fold_psi_to_positive(psi_from_q(qn))
    if abs(psi_m - psi_n) > float(psi_tol_rad):
        return False
    if int(branch_from_q(qm)) != int(branch_from_q(qn)):
        return False
    if abs(float(qn[1])) > 1.0e-3 and abs(float(qm[1])) > 1.0e-3:
        if float(qm[1]) * float(qn[1]) < 0.0:
            return False
    return True


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
    # Used only when ψ* changes (new scan segment).  No LPF on top.
    psi_rate_rad_s: float = 25.0 * np.pi / 180.0
    # Unplanned d* is a band around the design split, not a chasing point.
    d_center_rate_m_s: float = 0.02
    # Do not let ψ_cmd run more than this ahead of live ψ.
    psi_cmd_lead_rad: float = 18.0 * np.pi / 180.0
    # Design family (side-lying).  Unplanned homotopy and plan_stroke.
    psi_attr_rad: float = 68.0 * np.pi / 180.0
    d_attr_m: float = -0.185
    # Runtime elbow band.  Open rail travel must not pick J4≈135°.
    elbow_center_rad: float = 95.0 * np.pi / 180.0
    elbow_lo_rad: float = 70.0 * np.pi / 180.0
    elbow_hi_rad: float = 115.0 * np.pi / 180.0
    elbow_hi_illegal_rad: float = 130.0 * np.pi / 180.0
    psi_return_dwell_s: float = 1.0
    require_design_family: bool = False
    # Local ψ search (unplanned).  9 srs_ik × 0.09 ms ≈ 0.8 ms at 10 Hz.
    psi_replan_period_s: float = 0.1
    psi_search_half_span_rad: float = 45.0 * np.pi / 180.0
    psi_search_n: int = 9
    psi_wrist_ok_rad: float = 40.0 * np.pi / 180.0
    psi_envelope_lo_rad: float = 40.0 * np.pi / 180.0
    psi_envelope_hi_rad: float = 110.0 * np.pi / 180.0
    # Soft travel used by the planner (must cover the whole stroke).
    rail_margin_m: float = 0.02
    # Reject a cell whose wrist sits on the branch-barrier floor (~20°).
    wrist_min_rad: float = 30.0 * np.pi / 180.0


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
        self._d_center_target: float | None = None
        self._s: float = 0.0
        self._d0: float = float("nan")
        self._psi0: float = float("nan")
        self._branch: int = 0
        self.q_star_rad: np.ndarray | None = None
        self.homotopy_s: float = 0.0
        self._search_age_s: float = 0.0
        self.last_psi_search_count: int = 0
        self.last_search_j6_rad: float = float("nan")
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
        self.last_psi_family_degraded: bool = False
        self._healthy_dwell_s: float = 0.0
        self._held_prev: bool = False
        self._ird = None

    @property
    def planned(self) -> bool:
        return bool(self._planned)

    def reset(self, q_rad: np.ndarray) -> None:
        q = np.asarray(q_rad, dtype=float)
        # ±π are the same SEW plane.  Stay on the positive half so the
        # command slews 180°→70°, never −180°→−290° through ψ = 0.
        psi = fold_psi_to_positive(float(psi_from_q(q)))
        psi_star = clamp_psi_to_envelope(
            float(self.cfg.psi_attr_rad),
            self.cfg.psi_envelope_lo_rad,
            self.cfg.psi_envelope_hi_rad,
        )
        self._psi_cmd = psi
        self._psi_star = psi_star
        # Start at the live split.  q* is the live configuration — not the
        # yaml photo — so J1 is not pinned to −90° while d* is still here.
        d_live = d_from_q(self.kin, q)
        self._d_star = d_live
        self._d_center_target = float(self.cfg.d_attr_m)
        self._s = 0.0
        self._d0 = float(d_live)
        self._psi0 = float(psi)
        self._branch = int(branch_from_q(q))
        self.q_star_rad = np.asarray(q, dtype=float).reshape(-1).copy()
        self.homotopy_s = 0.0
        self._search_age_s = 0.0
        self._healthy_dwell_s = 0.0
        self.last_psi_search_count = 0
        self.last_search_j6_rad = float("nan")
        self._planned = False
        self._z_plan = float("nan")
        self._held_prev = False
        self.d_star_m = float(self._d_star)
        self.psi_star_rad = float(psi_star)

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
        """Grid-search ``(d*, ψ*)`` over the scan stroke.  Raises if empty.

        Search the taught SEW family first.  The opposite plane is used only
        when that family has no feasible cell (singularity / travel).
        """
        q = np.asarray(q_rad, dtype=float)
        self.last_psi_family_degraded = False
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
        # Unplanned home (psi_attr) must not steal the stroke family.
        if self._planned and self._psi_star is not None:
            psi_family = float(self._psi_star)
        else:
            psi_family = nearest_planar_psi(psi0)
        half = float(_PLAN_FAMILY_HALF_SPAN_RAD)
        family_samples = psi_family + np.linspace(-half, half, n_psi)
        opposite = _wrap_pi(psi_family + np.pi)
        opposite_samples = opposite + np.linspace(-half, half, n_psi)
        w_sigma = float(self.cfg.w_sigma)
        w_wrist = float(self.cfg.w_wrist)
        floor = float(self.cfg.margin_floor_rad)

        def _search(
            d_list: np.ndarray, psi_list: np.ndarray
        ) -> tuple[bool, float, float, float]:
            best_s = -np.inf
            best_dv = float(self._d_star if self._d_star is not None else 0.0)
            best_pv = psi_family
            found = False
            for d in d_list:
                for psi in psi_list:
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

        def _search_d(psi_list: np.ndarray) -> tuple[bool, float, float, float]:
            found, score, d_v, p_v = _search(d_samples, psi_list)
            if not found and d_samples.size == 1 and d_grid.size > 1:
                found, score, d_v, p_v = _search(d_grid, psi_list)
            return found, score, d_v, p_v

        any_feasible, best_score, best_d, best_psi = _search_d(family_samples)
        degraded = False
        if not any_feasible:
            degraded = True
            any_feasible, best_score, best_d, best_psi = _search_d(opposite_samples)
        if not any_feasible:
            raise StrokeInfeasibleError(
                "no feasible (d, ψ) covers the scan stroke; reduce amplitude "
                "or choose a less extended start pose"
            )
        # Family grids are already near 0 or ±π; wrap so CSV ψ* stays readable.
        best_psi = _wrap_pi(best_psi)
        self.last_psi_family_degraded = bool(degraded)
        self._d_star = float(best_d)
        self._d_center_target = float(best_d)
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
        hold_setpoint: bool = False,
    ) -> tuple[float, float]:
        """Slew (d*, ψ*, q*) on one s; planned strokes only slew ψ."""
        del q_nominal
        q = np.asarray(q_rad, dtype=float)
        if self._psi_cmd is None or self._d_star is None:
            self.reset(q)
        dt = max(float(dt_s), 0.0)
        live_psi = fold_psi_to_positive(float(psi_from_q(q)))
        if self._planned:
            psi_out = self._rate_limit_psi(dt, live_psi=live_psi)
            self._update_margins(q)
            return float(psi_out), float(self._d_star)
        if self._held_prev and not hold_setpoint:
            if self._d_star is not None and np.isfinite(float(self._d_star)):
                self._d0 = float(self._d_star)
            if self._psi_cmd is not None and np.isfinite(float(self._psi_cmd)):
                self._psi0 = float(self._psi_cmd)
            self._s = 0.0
            self.homotopy_s = 0.0
        self._held_prev = bool(hold_setpoint)
        if hold_setpoint:
            psi_out = self._rate_limit_psi(dt, live_psi=live_psi)
            self._update_margins(q)
            return float(psi_out), float(self._d_star)
        self._maybe_retarget_psi(
            q,
            dt_s=dt,
            rail_lo=float(rail_lo),
            rail_hi=float(rail_hi),
        )
        self._advance_homotopy(
            q,
            dt,
            rail_lo=float(rail_lo),
            rail_hi=float(rail_hi),
            live_psi=live_psi,
        )
        self._update_margins(q)
        return float(self._psi_cmd), float(self._d_star)

    def _advance_homotopy(
        self,
        q: np.ndarray,
        dt_s: float,
        *,
        rail_lo: float,
        rail_hi: float,
        live_psi: float,
    ) -> None:
        psi_goal = fold_psi_to_positive(
            float(self._psi_star if self._psi_star is not None else self._psi0)
        )
        pose = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        d_goal = self._select_d_for_elbow(
            q,
            pose=pose,
            psi=psi_goal,
            rail_lo=float(rail_lo),
            rail_hi=float(rail_hi),
        )
        if d_goal is None or not np.isfinite(float(d_goal)):
            self._rate_limit_psi(float(dt_s), live_psi=live_psi)
            return
        d0 = float(self._d0) if np.isfinite(self._d0) else float(self._d_star)
        psi0 = float(self._psi0) if np.isfinite(self._psi0) else float(self._psi_cmd)
        T = self._homotopy_T(d0, float(d_goal), psi0, psi_goal)
        s_try = min(1.0, float(self._s) + float(dt_s) / T)
        d_try = float(d0 + s_try * (float(d_goal) - d0))
        y_tcp = float(pose[1])
        d_try = self._clip_d_to_travel(
            d_try,
            y_tcp=y_tcp,
            rail_lo=float(rail_lo),
            rail_hi=float(rail_hi),
            d_live=y_tcp - float(q[RAIL_INDEX]),
        )
        if d_try is None:
            self._rate_limit_psi(float(dt_s), live_psi=live_psi)
            return
        d_step = max(float(self.cfg.d_center_rate_m_s), 0.0) * max(float(dt_s), 0.0)
        d_prev = (
            float(self._d_star)
            if self._d_star is not None and np.isfinite(float(self._d_star))
            else float(d_try)
        )
        d_try = max(d_prev - d_step, min(d_prev + d_step, float(d_try)))
        psi_s = fold_psi_to_positive(
            float(psi0) + s_try * psi_err_avoiding_zero(psi0, psi_goal)
        )
        pack = self._eval_at_split(pose, float(psi_s), float(d_try))
        if pack is None or not self._q_star_acceptable(pack[0], q, rail_lo, rail_hi):
            self._rate_limit_psi(float(dt_s), live_psi=live_psi)
            return
        self._s = float(s_try)
        self.homotopy_s = float(s_try)
        self._d_star = float(d_try)
        self.d_star_m = float(d_try)
        q_arm, q_full, _sigma = pack
        self.q_star_rad = np.asarray(q_full, dtype=float).copy()
        self._update_margins(q_full)
        del q_arm
        self._rate_limit_psi(float(dt_s), live_psi=live_psi)

    def _homotopy_T(
        self,
        d0: float,
        d_goal: float,
        psi0: float,
        psi_goal: float,
    ) -> float:
        d_rate = max(float(self.cfg.d_center_rate_m_s), 1.0e-9)
        psi_rate = max(float(self.cfg.psi_rate_rad_s), 1.0e-9)
        t_d = abs(float(d_goal) - float(d0)) / d_rate
        t_psi = abs(psi_err_avoiding_zero(float(psi0), float(psi_goal))) / psi_rate
        return max(t_d, t_psi, 1.0e-6)

    def _j4_in_design_band(self, j4_rad: float, *, loose: bool = False) -> bool:
        lo = float(self.cfg.elbow_lo_rad)
        hi = float(self.cfg.elbow_hi_rad)
        if loose:
            lo -= np.deg2rad(5.0)
            hi += np.deg2rad(7.0)
        return bool(lo - 1.0e-9 <= float(j4_rad) <= hi + 1.0e-9)

    def _j4_illegal_at_stop(self, j4_rad: float, *, has_travel: bool) -> bool:
        if not has_travel:
            return False
        return bool(abs(float(j4_rad)) >= float(self.cfg.elbow_hi_illegal_rad) - 1.0e-9)

    def _rail_window(
        self, y_tcp: float, rail_lo: float, rail_hi: float
    ) -> tuple[float, float] | None:
        margin = max(float(self.cfg.rail_margin_m), 0.0)
        y_lo = float(rail_lo) + margin
        y_hi = float(rail_hi) - margin
        if y_lo > y_hi + 1.0e-12:
            return None
        d_lo = float(y_tcp) - y_hi
        d_hi = float(y_tcp) - y_lo
        if d_lo > d_hi + 1.0e-12:
            return None
        return float(d_lo), float(d_hi)

    def _clip_d_to_travel(
        self,
        d: float,
        *,
        y_tcp: float,
        rail_lo: float,
        rail_hi: float,
        d_live: float | None,
    ) -> float | None:
        window = self._rail_window(float(y_tcp), float(rail_lo), float(rail_hi))
        if window is None:
            if d_live is not None and np.isfinite(float(d_live)):
                return float(d_live)
            return None
        return float(np.clip(float(d), window[0], window[1]))

    def _eval_at_split(
        self,
        pose: np.ndarray,
        psi: float,
        d: float,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        y_rail = float(pose[1]) - float(d)
        return self._eval.evaluate(pose, float(psi), int(self._branch), y_rail)

    def _q_star_acceptable(
        self,
        q_arm: np.ndarray,
        q_live: np.ndarray,
        rail_lo: float,
        rail_hi: float,
    ) -> bool:
        j4 = float(np.asarray(q_arm, dtype=float).reshape(-1)[3])
        window = self._rail_window(
            float(self.kin.fk_placement(q_live).translation[1]),
            float(rail_lo),
            float(rail_hi),
        )
        has_travel = window is not None and (window[1] - window[0]) > 0.01
        if self._j4_illegal_at_stop(j4, has_travel=has_travel):
            return False
        return True

    def _select_d_for_elbow(
        self,
        q: np.ndarray,
        *,
        pose: np.ndarray,
        psi: float,
        rail_lo: float,
        rail_hi: float,
    ) -> float | None:
        """Split at ``psi`` whose IK J4 stays in the design band, near d_attr."""
        y_tcp = float(pose[1])
        window = self._rail_window(y_tcp, float(rail_lo), float(rail_hi))
        if window is None:
            return None
        d_lo, d_hi = window
        d_pref = (
            float(self._d_center_target)
            if self._d_center_target is not None
            else float(self.cfg.d_attr_m)
        )
        has_travel = (d_hi - d_lo) > 0.01
        samples = list(np.linspace(d_lo, d_hi, 11))
        for extra in (d_pref, float(self._d_star), float(self._d0)):
            if extra is None or not np.isfinite(float(extra)):
                continue
            if d_lo - 1.0e-9 <= float(extra) <= d_hi + 1.0e-9:
                samples.append(float(extra))
        samples = [float(x) for x in np.unique(np.asarray(samples, dtype=float))]
        # Prefer the yaml family (J1 < 0).  Do not freeze s on a live/IK
        # sign mismatch — that locked d* while ψ already folded J1.
        sign_pref = -1.0
        j4_c = float(self.cfg.elbow_center_rad)
        best_d: float | None = None
        best_cost = float("inf")
        fallback_d: float | None = None
        fallback_cost = float("inf")
        for d in samples:
            pack = self._eval_at_split(pose, float(psi), float(d))
            if pack is None:
                continue
            q_arm = pack[0]
            j4 = float(q_arm[3])
            j1 = float(q_arm[0])
            if self._j4_illegal_at_stop(j4, has_travel=has_travel):
                continue
            sign_pen = 0.0
            if abs(j1) > np.deg2rad(10.0) and j1 * sign_pref < 0.0:
                sign_pen = 10.0
            cost = abs(float(d) - d_pref) + 0.15 * abs(j4 - j4_c) + sign_pen
            if cost < fallback_cost:
                fallback_cost = float(cost)
                fallback_d = float(d)
            if not self._j4_in_design_band(j4, loose=False):
                continue
            if cost < best_cost:
                best_cost = float(cost)
                best_d = float(d)
        if best_d is not None:
            return float(best_d)
        return fallback_d

    def _maybe_retarget_psi(
        self,
        q: np.ndarray,
        *,
        dt_s: float,
        rail_lo: float,
        rail_hi: float,
    ) -> None:
        dt = max(float(dt_s), 0.0)
        self._search_age_s += dt
        period = max(float(self.cfg.psi_replan_period_s), 0.0)
        due = self._search_age_s + 1.0e-12 >= period
        q_arm = np.asarray(q, dtype=float).reshape(-1)
        if q_arm.size == 8:
            q_arm = q_arm[1:]
        j4 = abs(float(q_arm[3]))
        j6 = abs(float(q_arm[5]))
        attr = clamp_psi_to_envelope(
            float(self.cfg.psi_attr_rad),
            self.cfg.psi_envelope_lo_rad,
            self.cfg.psi_envelope_hi_rad,
        )
        # SEW is undefined near a straight elbow; searching ψ there flipped
        # the family on 035411 (J4 through 0, ψ 39°→−141°).
        if j4 < float(self.cfg.psi_envelope_lo_rad):
            return
        wrist_bad = j6 < float(self.cfg.psi_wrist_ok_rad)
        if wrist_bad:
            self._healthy_dwell_s = 0.0
            if not due:
                return
            self._search_age_s = 0.0
            found = self.search_psi_at_pose(q, rail_lo=rail_lo, rail_hi=rail_hi)
            self.last_psi_search_count += 1
            if found is None:
                return
            self._psi_star = float(found)
            self.psi_star_rad = float(found)
            return
        self._healthy_dwell_s += dt
        if due:
            self._search_age_s = 0.0
        dwell = max(float(self.cfg.psi_return_dwell_s), 0.0)
        if self._healthy_dwell_s + 1.0e-12 >= dwell:
            self._psi_star = float(attr)
            self.psi_star_rad = float(attr)

    def _psi_infeasible_at(
        self,
        q_rad: np.ndarray,
        psi: float,
        *,
        rail_lo: float,
        rail_hi: float,
    ) -> bool:
        q = np.asarray(q_rad, dtype=float)
        pose = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        d_c = (
            float(self._d_star)
            if self._d_star is not None
            else d_from_q(self.kin, q)
        )
        y_rail = float(pose[1]) - d_c
        margin = max(float(self.cfg.rail_margin_m), 0.0)
        if y_rail < float(rail_lo) + margin or y_rail > float(rail_hi) - margin:
            return True
        pack = self._eval.evaluate(
            pose, float(psi), int(branch_from_q(q)), y_rail
        )
        return pack is None

    def search_psi_at_pose(
        self,
        q_rad: np.ndarray,
        *,
        rail_lo: float,
        rail_hi: float,
    ) -> float | None:
        """Best ψ in the local envelope window at the current TCP, or None.

        Score is wrist openness plus joint margin.  Samples stay inside
        ``[psi_envelope_lo, psi_envelope_hi]`` so the family never crosses 0.
        """
        q = np.asarray(q_rad, dtype=float)
        pose = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        branch = int(branch_from_q(q))
        d_c = (
            float(self._d_star)
            if self._d_star is not None
            else d_from_q(self.kin, q)
        )
        y_rail = float(pose[1]) - d_c
        margin = max(float(self.cfg.rail_margin_m), 0.0)
        if y_rail < float(rail_lo) + margin or y_rail > float(rail_hi) - margin:
            return None
        lo = float(self.cfg.psi_envelope_lo_rad)
        hi = float(self.cfg.psi_envelope_hi_rad)
        center = (
            float(self._psi_star)
            if self._psi_star is not None
            else clamp_psi_to_envelope(float(psi_from_q(q)), lo, hi)
        )
        center = clamp_psi_to_envelope(center, lo, hi)
        half = max(float(self.cfg.psi_search_half_span_rad), 0.0)
        n = max(int(self.cfg.psi_search_n), 3)
        raw = np.linspace(center - half, center + half, n)
        local = np.unique(
            np.array([clamp_psi_to_envelope(p, lo, hi) for p in raw], dtype=float)
        )
        best_psi, best_j6 = self._score_psi_samples(
            local, pose=pose, branch=branch, y_rail=y_rail
        )
        wrist_ok = float(self.cfg.psi_wrist_ok_rad)
        if best_psi is None or not np.isfinite(best_j6) or best_j6 < wrist_ok:
            full = np.linspace(lo, hi, n)
            best_full, j6_full = self._score_psi_samples(
                full, pose=pose, branch=branch, y_rail=y_rail
            )
            if best_full is not None and (
                best_psi is None or j6_full > best_j6 + 1.0e-9
            ):
                best_psi, best_j6 = best_full, j6_full
        self.last_search_j6_rad = best_j6
        return best_psi

    def _score_psi_samples(
        self,
        samples: np.ndarray,
        *,
        pose: np.ndarray,
        branch: int,
        y_rail: float,
    ) -> tuple[float | None, float]:
        best_s = -np.inf
        best_psi: float | None = None
        best_j6 = float("nan")
        for psi in samples:
            pack = self._eval.evaluate(pose, float(psi), branch, y_rail)
            if pack is None:
                continue
            q_arm, q_full, _sigma = pack
            j6 = abs(float(q_arm[5]))
            if j6 < float(self.cfg.wrist_min_rad) - 1.0e-9:
                continue
            marg = float(np.min(np.minimum(q_arm - Q_LOWER, Q_UPPER - q_arm)))
            score = min(j6 / (60.0 * np.pi / 180.0), 1.0) + 0.8 * min(
                marg / (30.0 * np.pi / 180.0), 1.0
            )
            if score > best_s + 1.0e-9:
                best_s = float(score)
                best_psi = float(psi)
                best_j6 = float(j6)
                self._update_margins(q_full)
                self.last_dpref_score = float(score)
                self.last_psi_score = float(score)
        return best_psi, best_j6

    def nudge_d_star(
        self,
        delta_m: float,
        *,
        y_des_m: float,
        rail_lo: float,
        rail_hi: float,
        dt_s: float = 0.005,
    ) -> float:
        """Shift d* so rail_ff = y_des − d* stays inside the soft travel.

        The clip is a bound, not a step.  ``d_center_rate_m_s`` then slews.
        """
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
        self._d_center_target = d_new
        return self._rate_limit_d(float(dt_s))

    def _rate_limit_d(
        self,
        dt_s: float,
        *,
        y_tcp: float | None = None,
        rail_lo: float | None = None,
        rail_hi: float | None = None,
        d_live: float | None = None,
    ) -> float:
        if self._d_star is None:
            return float("nan")
        target = (
            float(self._d_center_target)
            if self._d_center_target is not None
            else float(self._d_star)
        )
        cur = float(self._d_star)
        err = target - cur
        max_step = max(float(self.cfg.d_center_rate_m_s), 0.0) * max(float(dt_s), 0.0)
        if max_step > 0.0 and abs(err) > max_step:
            err = float(np.clip(err, -max_step, max_step))
        new_d = float(cur + err)
        if (
            y_tcp is not None
            and rail_lo is not None
            and rail_hi is not None
            and np.isfinite(float(y_tcp))
        ):
            margin = max(float(self.cfg.rail_margin_m), 0.0)
            y_lo = float(rail_lo) + margin
            y_hi = float(rail_hi) - margin
            if y_lo > y_hi + 1.0e-12:
                if d_live is not None and np.isfinite(float(d_live)):
                    self._d_star = float(d_live)
                self.d_star_m = float(self._d_star)
                return float(self._d_star)
            d_lo = float(y_tcp) - y_hi
            d_hi = float(y_tcp) - y_lo
            if d_lo > d_hi + 1.0e-12:
                if d_live is not None and np.isfinite(float(d_live)):
                    self._d_star = float(d_live)
                self.d_star_m = float(self._d_star)
                return float(self._d_star)
            new_d = float(np.clip(new_d, d_lo, d_hi))
        self._d_star = new_d
        self.d_star_m = float(self._d_star)
        return float(self._d_star)

    def _rate_limit_psi(
        self, dt_s: float, live_psi: float | None = None
    ) -> float:
        target = fold_psi_to_positive(
            float(self._psi_star if self._psi_star is not None else 0.0)
        )
        cur = fold_psi_to_positive(
            float(self._psi_cmd if self._psi_cmd is not None else target)
        )
        err = psi_err_avoiding_zero(cur, target)
        max_step = float(self.cfg.psi_rate_rad_s) * dt_s
        if max_step > 0.0 and abs(err) > max_step:
            err = float(np.clip(err, -max_step, max_step))
        nxt = float(cur + err)
        # Never publish a command that sits on the wrong side of 0.
        if cur * nxt < 0.0 and abs(cur) > 1.0e-6:
            nxt = float(np.sign(cur) * 1.0e-6)
        nxt = fold_psi_to_positive(nxt)
        lead = max(float(self.cfg.psi_cmd_lead_rad), 0.0)
        if (
            lead > 0.0
            and live_psi is not None
            and np.isfinite(float(live_psi))
        ):
            live = fold_psi_to_positive(float(live_psi))
            lead_nxt = abs(psi_err_avoiding_zero(live, nxt))
            lead_cur = abs(psi_err_avoiding_zero(live, cur))
            if lead_nxt > lead + 1.0e-12 and lead_nxt > lead_cur + 1.0e-12:
                nxt = cur
        self._psi_cmd = nxt
        return float(self._psi_cmd)


__all__ = [
    "PostureRetarget",
    "PsiRetargetConfig",
    "StrokeInfeasibleError",
    "arm_respects_floor",
    "clamp_psi_to_envelope",
    "d_from_q",
    "design_family_ok",
    "fold_psi_to_positive",
    "joint_margin_frac",
    "nearest_planar_psi",
    "psi_err_avoiding_zero",
    "stroke_score",
    "wrist_band_frac",
]
