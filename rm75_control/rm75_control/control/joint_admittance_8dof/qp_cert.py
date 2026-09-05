"""QP accept / in-box brake helpers shared by the Python loop and tests."""

from __future__ import annotations

import math

import numpy as np

QP_STATUS_NOT_RUN = 0
QP_STATUS_SOLVED = 1
QP_STATUS_MAX_ITER = 2
QP_STATUS_FAILED = 3
QP_STATUS_PRIMAL_INFEASIBLE = 4
QP_STATUS_DUAL_INFEASIBLE = 5
QP_STATUS_CLOSEST_PRIMAL = 6
QP_STATUS_P0_CONFLICT = 7
QP_STATUS_NAMES = (
    "not_run",
    "solved",
    "max_iter",
    "failed",
    "primal_infeasible",
    "dual_infeasible",
    "closest_primal",
    "p0_conflict",
)


def qp_status_name(code) -> str:
    try:
        i = int(code)
    except (TypeError, ValueError):
        text = str(code).lower()
        if "p0_conflict" in text:
            return "p0_conflict"
        if "primal_infeasible" in text or "primal-infeasible" in text:
            return "primal_infeasible"
        if "dual_infeasible" in text or "dual-infeasible" in text:
            return "dual_infeasible"
        if "closest" in text:
            return "closest_primal"
        if "max_iter" in text:
            return "max_iter"
        if "solved" in text:
            return "solved"
        if "not_run" in text:
            return "not_run"
        return "failed"
    if 0 <= i < len(QP_STATUS_NAMES):
        return QP_STATUS_NAMES[i]
    return "failed"


def qp_status_code_from_prox(status) -> int:
    text = str(status)
    upper = text.upper()
    if "CLOSEST_PRIMAL" in upper or "SOLVED_CLOSEST" in upper:
        return QP_STATUS_CLOSEST_PRIMAL
    if "PRIMAL_INFEASIBLE" in upper or "PRIMAL-INFEASIBLE" in upper:
        return QP_STATUS_PRIMAL_INFEASIBLE
    if "DUAL_INFEASIBLE" in upper or "DUAL-INFEASIBLE" in upper:
        return QP_STATUS_DUAL_INFEASIBLE
    if "P0_CONFLICT" in upper:
        return QP_STATUS_P0_CONFLICT
    if "PROXQP_SOLVED" in text or text == "solved":
        return QP_STATUS_SOLVED
    if "MAX_ITER" in text or "max_iter" in text.lower():
        return QP_STATUS_MAX_ITER
    if text in ("not_run", ""):
        return QP_STATUS_NOT_RUN
    return QP_STATUS_FAILED


def qp_status_publishable(name: str, *, certified: bool) -> bool:
    """Only SOLVED, or certified MAX_ITER, may leave the solver."""

    key = qp_status_name(name)
    if key == "solved":
        return True
    if key == "max_iter":
        return bool(certified)
    return False


def inbox_brake(qdot_prev, lo, hi, a_max, h1: float) -> np.ndarray:
    prev = np.asarray(qdot_prev, dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    a_max = np.asarray(a_max, dtype=float).reshape(-1)
    h = max(float(h1), 0.0)
    out = np.zeros_like(prev)
    n = int(prev.size)
    for i in range(n):
        step = max(float(a_max[i]) * h, 0.0) if i < a_max.size else 0.0
        brake = float(prev[i])
        if prev[i] > 0.0:
            brake = max(0.0, float(prev[i]) - step)
        elif prev[i] < 0.0:
            brake = min(0.0, float(prev[i]) + step)
        loi = float(lo[i]) if i < lo.size else brake
        hii = float(hi[i]) if i < hi.size else brake
        if np.isfinite(loi) and np.isfinite(hii) and loi <= hii:
            out[i] = min(hii, max(loi, brake))
        elif np.isfinite(loi) and np.isfinite(hii):
            out[i] = 0.5 * (loi + hii)
        else:
            out[i] = brake
    return out


def measure_qdot_box(qdot, lo, hi) -> tuple[float, bool, bool, bool]:
    qdot = np.asarray(qdot, dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    excess_max = 0.0
    degenerate = False
    infeasible = False
    substantial = False
    n = min(qdot.size, lo.size, hi.size)
    for i in range(n):
        if not (np.isfinite(lo[i]) and np.isfinite(hi[i])):
            continue
        w = float(hi[i] - lo[i])
        if w < -1.0e-12:
            infeasible = True
            degenerate = True
        elif w <= 1.0e-9:
            degenerate = True
        excess = 0.0
        if qdot[i] < lo[i]:
            excess = float(lo[i] - qdot[i])
        if qdot[i] > hi[i]:
            excess = max(excess, float(qdot[i] - hi[i]))
        excess_max = max(excess_max, excess)
        if excess > 1.0e-6 and (
            i == 0 or w <= 1.0e-9 or excess > 0.10 * w
        ):
            substantial = True
    return excess_max, degenerate, infeasible, substantial


def raised_cosine_alpha(
    slack: float,
    slack_exit: float,
    slack_enter: float,
    sigma: float,
    sigma_ref: float,
) -> float:
    span = max(float(slack_enter) - float(slack_exit), 1.0e-9)
    x = min(1.0, max(0.0, (float(slack) - float(slack_exit)) / span))
    a_slack = 0.5 * (1.0 + math.cos(math.pi * x))
    a_sigma = min(1.0, max(0.0, float(sigma) / max(float(sigma_ref), 1.0e-9)))
    return float(a_slack * a_sigma)


def dual_cancel_frac(u_task: float, u_post: float, active: float = 0.002) -> float:
    if abs(u_task) <= active or abs(u_post) <= active:
        return 0.0
    if u_task * u_post >= 0.0:
        return 0.0
    den = abs(u_task) + abs(u_post)
    if den <= 1.0e-12:
        return 0.0
    return 1.0 - abs(u_task + u_post) / den
