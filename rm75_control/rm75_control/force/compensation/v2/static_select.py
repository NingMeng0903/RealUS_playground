"""Conditional D-optimal static pose selection for m,h given bias nuisances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.force.compensation.v2.regressor_v2 import static_design


def gravity_dirs_seed() -> np.ndarray:
    axes = np.vstack([np.eye(3), -np.eye(3)])
    diag = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                v = np.array([sx, sy, sz], dtype=float)
                diag.append(v / np.linalg.norm(v))
    return np.vstack([axes, np.asarray(diag)])


def _A_from_g(g_L: np.ndarray, *, include_drift: bool = False) -> np.ndarray:
    return static_design(g_L, include_drift=include_drift, t_s=0.0)


def conditional_info(A: np.ndarray, Q: np.ndarray, n_payload: int = 4) -> np.ndarray:
    """I_{p|n} for columns ``[m,h | bias...]``."""
    Ap = A[:, :n_payload]
    An = A[:, n_payload:]
    QA = Q @ A
    # Use slices on QA
    Ip = Ap.T @ Q @ Ap
    if An.size == 0:
        return Ip
    In = An.T @ Q @ An
    Ipn = Ap.T @ Q @ An
    try:
        corr = Ipn @ np.linalg.pinv(In, rcond=1e-10) @ Ipn.T
    except np.linalg.LinAlgError:
        corr = np.zeros_like(Ip)
    return Ip - corr


def greedy_d_optimal(
    g_candidates: np.ndarray,
    *,
    n_select: int,
    sigma: np.ndarray | None = None,
    lam: float = 1e-6,
) -> np.ndarray:
    G = np.asarray(g_candidates, dtype=float).reshape(-1, 3)
    n = G.shape[0]
    if sigma is None:
        Q = np.eye(6)
    else:
        S = np.asarray(sigma, dtype=float)
        if S.ndim == 1:
            Q = np.diag(1.0 / np.maximum(S**2, 1e-12))
        else:
            Q = np.linalg.pinv(S, rcond=1e-10)
    selected: list[int] = []
    remaining = set(range(n))
    I = np.zeros((4, 4))
    for _ in range(min(int(n_select), n)):
        best_i, best_score = None, -np.inf
        for i in remaining:
            A = _A_from_g(G[i])
            dI = conditional_info(A, Q)
            trial = I + dI
            sign, logdet = np.linalg.slogdet(trial + lam * np.eye(4))
            score = float(logdet) if sign > 0 else -np.inf
            evals = np.linalg.eigvalsh(trial + lam * np.eye(4))
            score = score + 1e-3 * float(evals.min())
            if score > best_score:
                best_score, best_i = score, i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
        I = I + conditional_info(_A_from_g(G[best_i]), Q)
    return np.asarray(selected, dtype=int)


@dataclass
class StaticPoseSet:
    train_g: np.ndarray
    holdout_g: np.ndarray
    cable_yaw_pairs: list[tuple[np.ndarray, np.ndarray]]
    rank_m0: int
    cond_m0: float


def stacked_A(g_list: np.ndarray, *, include_drift: bool = False) -> np.ndarray:
    rows = [_A_from_g(g, include_drift=include_drift) for g in np.asarray(g_list).reshape(-1, 3)]
    return np.vstack(rows)


def rank_cond(A: np.ndarray, *, n_cols: int | None = None) -> tuple[int, float]:
    M = A if n_cols is None else A[:, :n_cols]
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-12]
    rank = int(s.size)
    cond = float(s[0] / s[-1]) if s.size else float("inf")
    return rank, cond


def build_default_set(
    *,
    n_train: int = 14,
    n_holdout: int = 4,
    g_scale: float = 9.80665,
) -> StaticPoseSet:
    dirs = gravity_dirs_seed()
    extra = []
    rng = np.random.default_rng(7)
    for _ in range(80):
        v = rng.normal(size=3)
        extra.append(v / np.linalg.norm(v))
    cand = np.vstack([dirs, np.asarray(extra)])
    idx = greedy_d_optimal(cand, n_select=n_train + n_holdout)
    chosen = cand[idx]
    train = chosen[:n_train] * g_scale
    hold = chosen[n_train:] * g_scale
    A = stacked_A(train)
    rank, cond = rank_cond(A, n_cols=10)
    yaw_pairs = []
    for g in train[:3]:
        yaw_pairs.append((g.copy(), g.copy()))
    return StaticPoseSet(train, hold, yaw_pairs, rank, cond)
