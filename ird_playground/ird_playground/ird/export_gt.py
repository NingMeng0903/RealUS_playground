"""Build dense stratified IRD training samples from a CapabilityMap.

Layers (default): 35% reachable interior / 40% boundary band / 25% exterior.
Labels: reachable, m_gt (margin, not a strict SDF), q and optional factor channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class IrdGtConfig:
    n_interior: int = 700_000
    n_boundary: int = 800_000
    n_exterior: int = 500_000
    # legacy aliases used by older configs
    n_positive: int = 700_000
    n_negative: int = 500_000
    seed: int = 0
    comfort_from: str = "auto"
    bbox_margin_m: float = 0.20
    max_orients_per_voxel: int = 24
    hard_negative_frac: float = 0.45
    hard_negative_radius_m: float = 0.06
    sigma_p_m: float = 0.03
    sigma_r_deg: float = 10.0
    boundary_d_lo: float = 0.02
    boundary_d_hi: float = 0.08
    m_pos_scale: float = 2.0
    m_neg_scale: float = 2.0
    w_manip: float = 0.5
    w_d: float = 0.5
    k_candidates: int = 4
    n_dof: int = 7


def _batch_feat_from_p_u(ps: np.ndarray, us: np.ndarray) -> np.ndarray:
    """ps (N,3), us (N,3) → features (N,9), vectorized frame build."""
    n = int(ps.shape[0])
    z = np.asarray(us, dtype=np.float64)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    a = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    flip = np.abs(z[:, 0]) >= 0.9
    a[flip] = np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    y = np.cross(z, x)
    # R columns = x,y,z; T_base_tcp; ΔT = T^{-1} = [R^T | -R^T t]
    # features = t_delta(3) + R_delta[:,0](3) + R_delta[:,1](3)
    # R_delta = R^T, t_delta = -R^T @ p
    Rt = np.stack([x, y, z], axis=1)  # (N,3,3) rows = x,y,z → actually want columns
    # stack as columns: R[..., :, 0] = x
    R = np.stack([x, y, z], axis=2)  # (N,3,3)
    Rt = np.transpose(R, (0, 2, 1))  # R^T
    t_delta = -np.einsum("nij,nj->ni", Rt, ps.astype(np.float64))
    r6 = np.concatenate([Rt[:, :, 0], Rt[:, :, 1]], axis=1)
    return np.concatenate([t_delta, r6], axis=1).astype(np.float32)


def _precompute_orient_table(cm, orients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_ids, orient_ids) flattened over all reachable (voxel, orient) pairs."""
    from ird_playground.ird.capability_io import unpack_bits_5dof

    n_orient = orients.shape[0]
    if cm.roll is None:
        bits = unpack_bits_5dof(np.asarray(cm.bitmask), n_orient)
    else:
        bits = np.any(cm.bitmask, axis=-1)
    rows, oids = np.nonzero(bits)
    return rows.astype(np.int64), oids.astype(np.int64)


def _sample_from_table(
    rng: np.random.Generator,
    table_rows: np.ndarray,
    table_oids: np.ndarray,
    pool_idx: np.ndarray,
    orients: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample n (row, u) from precomputed indices into the orient table."""
    if pool_idx.size == 0:
        rows = np.zeros(n, dtype=np.int64)
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return rows, v
    pick = rng.integers(0, pool_idx.shape[0], size=n)
    ti = pool_idx[pick]
    return table_rows[ti], orients[table_oids[ti]]


def export_ird_gt_from_capability_map(
    cm,
    cfg: IrdGtConfig | None = None,
    *,
    batch_size: int = 16384,
) -> dict[str, np.ndarray]:
    """Dense stratified GT with margin labels."""
    cfg = cfg or IrdGtConfig()
    rng = np.random.default_rng(cfg.seed)

    if cfg.n_interior or cfg.n_boundary or cfg.n_exterior:
        n_int = int(cfg.n_interior)
        n_bnd = int(cfg.n_boundary)
        n_ext = int(cfg.n_exterior)
    else:
        n_tot = int(cfg.n_positive + cfg.n_negative)
        n_int = int(round(0.35 * n_tot))
        n_bnd = int(round(0.40 * n_tot))
        n_ext = max(0, n_tot - n_int - n_bnd)

    orients = np.asarray(cm.orientations.vectors, dtype=np.float64)
    voxel_xyz = cm.grid.center_of(cm.voxel_ids)
    d_vals = np.asarray(cm.d_value, dtype=np.float64)
    M = int(cm.voxel_ids.shape[0])
    d_max = float(max(d_vals.max(), 1e-6))

    interior_rows = np.flatnonzero(d_vals >= cfg.boundary_d_hi)
    boundary_rows = np.flatnonzero((d_vals >= cfg.boundary_d_lo) & (d_vals < cfg.boundary_d_hi))
    if interior_rows.size == 0:
        interior_rows = np.arange(M)
    if boundary_rows.size == 0:
        boundary_rows = interior_rows

    print(f"[gt] unpacking orientation table for {M} voxels…", flush=True)
    table_rows, table_oids = _precompute_orient_table(cm, orients)
    print(f"[gt] table size={table_rows.shape[0]:,}", flush=True)

    is_int = np.zeros(M, dtype=bool)
    is_bnd = np.zeros(M, dtype=bool)
    is_int[interior_rows] = True
    is_bnd[boundary_rows] = True
    pool_int = np.flatnonzero(is_int[table_rows])
    pool_bnd = np.flatnonzero(is_bnd[table_rows])
    if pool_int.size == 0:
        pool_int = np.arange(table_rows.shape[0])
    if pool_bnd.size == 0:
        pool_bnd = pool_int
    print(f"[gt] pool interior={pool_int.size:,} boundary={pool_bnd.size:,}", flush=True)

    d_n_all = np.clip(d_vals / d_max, 0.0, 1.0).astype(np.float32)
    if cm.mu_mean is not None:
        mu = np.asarray(cm.mu_mean, dtype=np.float64)
        qm_all = np.clip(mu / (np.abs(mu) + 1.0), 0.0, 1.0).astype(np.float32)
        bad = ~np.isfinite(mu)
        qm_all[bad] = d_n_all[bad]
    else:
        qm_all = d_n_all.copy()
    qj_all = d_n_all.copy()
    qs_all = d_n_all.copy()
    qn_all = d_n_all.copy()
    q_all = np.clip(cfg.w_manip * qm_all + cfg.w_d * d_n_all, 0.0, 1.0).astype(np.float32)

    chunks: list[dict[str, np.ndarray]] = []

    def _emit(
        ps: np.ndarray,
        us: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        rows: np.ndarray | None,
        *,
        zero_q: bool = False,
    ) -> None:
        feats = _batch_feat_from_p_u(ps, us)
        if zero_q or rows is None:
            q = np.zeros(ps.shape[0], dtype=np.float32)
            qm = qj = qs = qn = q
        else:
            q = q_all[rows]
            qm = qm_all[rows]
            qj = qj_all[rows]
            qs = qs_all[rows]
            qn = qn_all[rows]
            q = np.where(y >= 0.5, q, 0.0).astype(np.float32)
            qm = np.where(y >= 0.5, qm, 0.0).astype(np.float32)
            qj = np.where(y >= 0.5, qj, 0.0).astype(np.float32)
            qs = np.where(y >= 0.5, qs, 0.0).astype(np.float32)
            qn = np.where(y >= 0.5, qn, 0.0).astype(np.float32)
        chunks.append(
            {
                "features": feats,
                "reachable": y.astype(np.float32),
                "m_gt": m.astype(np.float32),
                "q": q,
                "q_manip": qm,
                "q_joint": qj,
                "q_selfcol": qs,
                "q_nullspace": qn,
            }
        )

    def _layer_reachable(n_total: int, pool: np.ndarray, m_fn, label: str) -> None:
        for s in range(0, n_total, batch_size):
            n = min(batch_size, n_total - s)
            rows, us = _sample_from_table(rng, table_rows, table_oids, pool, orients, n)
            m = m_fn(rows)
            _emit(voxel_xyz[rows], us, np.ones(n, dtype=np.float32), m, rows)
            if (s // batch_size) % 5 == 0:
                print(f"[gt] {label} {s + n:,}/{n_total:,}", flush=True)

    print(f"[gt] interior {n_int:,}", flush=True)
    _layer_reachable(
        n_int,
        pool_int,
        lambda rows: cfg.m_pos_scale * (0.5 + 0.5 * (d_vals[rows] / d_max)),
        "interior",
    )

    n_bnd_vox = n_bnd // 2
    n_bnd_jit = n_bnd - n_bnd_vox
    print(f"[gt] boundary voxels {n_bnd_vox:,}", flush=True)
    _layer_reachable(
        n_bnd_vox,
        pool_bnd,
        lambda rows: cfg.m_pos_scale * ((d_vals[rows] / d_max) - 0.5) * 0.5,
        "boundary",
    )

    print(f"[gt] boundary jitter {n_bnd_jit:,}", flush=True)
    for s in range(0, n_bnd_jit, batch_size):
        n = min(batch_size, n_bnd_jit - s)
        rows, us = _sample_from_table(rng, table_rows, table_oids, pool_int, orients, n)
        jitter = rng.normal(scale=cfg.sigma_p_m, size=(n, 3))
        ps = voxel_xyz[rows] + jitter
        dist = np.linalg.norm(jitter, axis=1) / max(cfg.sigma_p_m, 1e-6)
        y = (dist < 1.0).astype(np.float32)
        m = np.where(
            y >= 0.5,
            cfg.m_pos_scale * 0.15 * (1.0 - np.minimum(dist, 2.0) / 2.0),
            -cfg.m_neg_scale * np.minimum(dist, 2.0) / 2.0,
        )
        _emit(ps, us, y, m, rows)
        if (s // batch_size) % 10 == 0:
            print(f"[gt] jitter {s + n:,}/{n_bnd_jit:,}", flush=True)

    # --- Exterior ---
    mins = voxel_xyz.min(axis=0) - cfg.bbox_margin_m
    maxs = voxel_xyz.max(axis=0) + cfg.bbox_margin_m
    n_hard = int(round(n_ext * float(cfg.hard_negative_frac)))
    n_unif = max(0, n_ext - n_hard)
    n_cent = min(voxel_xyz.shape[0], 80_000)
    cent_idx = rng.choice(voxel_xyz.shape[0], size=n_cent, replace=False)
    centers = voxel_xyz[cent_idx]

    t_unif = rng.uniform(mins, maxs, size=(n_unif, 3)) if n_unif else np.zeros((0, 3))
    pick = rng.integers(0, centers.shape[0], size=n_hard) if n_hard else np.zeros(0, dtype=int)
    t_hard = (
        centers[pick] + rng.normal(scale=cfg.hard_negative_radius_m, size=(n_hard, 3))
        if n_hard
        else np.zeros((0, 3))
    )
    t_neg = np.concatenate([t_unif, t_hard], axis=0) if n_ext > 0 else np.zeros((0, 3))
    v = rng.normal(size=(t_neg.shape[0], 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12

    print(f"[gt] exterior {t_neg.shape[0]:,}", flush=True)
    for s in range(0, t_neg.shape[0], batch_size):
        n = min(batch_size, t_neg.shape[0] - s)
        pts = t_neg[s : s + n]
        sub = centers[rng.choice(centers.shape[0], size=min(2048, centers.shape[0]), replace=False)]
        dmin = np.full(n, np.inf, dtype=np.float64)
        for c0 in range(0, sub.shape[0], 512):
            cblk = sub[c0 : c0 + 512]
            d2 = ((pts[:, None, :] - cblk[None, :, :]) ** 2).sum(axis=2)
            dmin = np.minimum(dmin, np.sqrt(d2.min(axis=1)))
        m = -cfg.m_neg_scale * np.minimum(dmin / max(cfg.hard_negative_radius_m, 1e-6), 2.0) / 2.0
        _emit(pts, v[s : s + n], np.zeros(n, dtype=np.float32), m, None, zero_q=True)
        if (s // batch_size) % 10 == 0:
            print(f"[gt] exterior {s + n:,}/{t_neg.shape[0]:,}", flush=True)

    if not chunks:
        raise RuntimeError("no IRD samples extracted from capability map")

    def _cat(key: str) -> np.ndarray:
        return np.concatenate([c[key] for c in chunks], axis=0)

    features = _cat("features")
    y = _cat("reachable")
    m_arr = _cat("m_gt")
    q_arr = _cat("q")
    perm = rng.permutation(features.shape[0])

    aabb_lo = (voxel_xyz.min(axis=0) - cfg.bbox_margin_m).astype(np.float32)
    aabb_hi = (voxel_xyz.max(axis=0) + cfg.bbox_margin_m).astype(np.float32)

    n = features.shape[0]
    k = int(cfg.k_candidates)
    q_best = np.zeros((n, cfg.n_dof), dtype=np.float32)
    q_candidates = np.zeros((n, k, cfg.n_dof), dtype=np.float32)

    print(f"[gt] stacking N={n:,}", flush=True)
    return {
        "features": features[perm],
        "reachable": y[perm],
        "p_reach": y[perm],
        "m_gt": m_arr[perm],
        "q": q_arr[perm],
        "q_comfort": q_arr[perm],
        "q_manip": _cat("q_manip")[perm],
        "q_joint": _cat("q_joint")[perm],
        "q_selfcol": _cat("q_selfcol")[perm],
        "q_nullspace": _cat("q_nullspace")[perm],
        "q_best": q_best[perm],
        "q_candidates": q_candidates[perm],
        "d": (y * q_arr)[perm],
        "aabb_lo": aabb_lo,
        "aabb_hi": aabb_hi,
        "sigma_p_m": np.array([cfg.sigma_p_m], dtype=np.float32),
        "sigma_r_deg": np.array([cfg.sigma_r_deg], dtype=np.float32),
    }


def save_ird_gt(path: str | Path, arrays: dict[str, np.ndarray], meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    if meta is not None:
        path.with_suffix(".yaml").write_text(
            yaml.safe_dump(meta, sort_keys=False), encoding="utf-8"
        )
    return path


def load_ird_gt(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


def make_synthetic_ird_gt(
    n: int = 4096,
    *,
    seed: int = 0,
    reach_radius: float = 0.6,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    r6 = np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (n, 1))
    features = np.concatenate([t, r6], axis=1)
    dist = np.linalg.norm(t, axis=1)
    y = (dist < reach_radius).astype(np.float32)
    m_gt = ((reach_radius - dist) / max(reach_radius, 1e-6) * 2.0).astype(np.float32)
    q = np.clip(1.0 - dist / (reach_radius + 1e-6), 0.0, 1.0).astype(np.float32) * y
    aabb_lo = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    aabb_hi = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return {
        "features": features,
        "reachable": y,
        "p_reach": y,
        "m_gt": m_gt,
        "q": q,
        "q_comfort": q,
        "q_manip": q,
        "q_joint": q,
        "q_selfcol": q,
        "q_nullspace": q,
        "q_best": np.zeros((n, 7), dtype=np.float32),
        "q_candidates": np.zeros((n, 4, 7), dtype=np.float32),
        "d": y * q,
        "aabb_lo": aabb_lo,
        "aabb_hi": aabb_hi,
        "sigma_p_m": np.array([0.03], dtype=np.float32),
        "sigma_r_deg": np.array([10.0], dtype=np.float32),
    }
