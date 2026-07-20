"""Build IRD GT from CapabilityMap — sign-consistent, bitmask-exact, 6-D (t,u).

Contract:
  y=1 ⇒ m_gt > 0 ;  y=0 ⇒ m_gt < 0
  features = [ΔT_translation(3), tool_axis(3)]
  aabb from features[:, :3] (ΔT frame)
  reachable labels from bitmask (jitter re-queries)
  m_gt = per-orient signed EDT / σ_p , clipped
  q = capability comfort (D, μ) on positives only
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import binary_dilation, distance_transform_edt


@dataclass
class IrdGtConfig:
    n_interior: int = 700_000
    n_boundary: int = 800_000
    n_exterior: int = 500_000
    n_positive: int = 700_000
    n_negative: int = 500_000
    seed: int = 0
    comfort_from: str = "auto"
    bbox_margin_m: float = 0.20
    max_orients_per_voxel: int = 28
    hard_negative_frac: float = 0.45
    hard_negative_radius_m: float = 0.06
    sigma_p_m: float = 0.03
    sigma_r_deg: float = 10.0
    m_clip: float = 3.0
    m_eps: float = 0.05
    edt_dilate: int = 2
    w_manip: float = 0.5
    w_d: float = 0.5
    k_candidates: int = 4
    n_dof: int = 7
    aabb_pad_frac: float = 0.05
    aabb_pad_min_m: float = 0.02
    n_jitter: int = 200_000


def features_from_p_u(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(N,6): t=−R(u)ᵀp , tool axis u (TCP +Z)."""
    p = np.asarray(p, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p[None, :]
        u = u[None, :]
    z = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    a = np.tile(np.array([1.0, 0.0, 0.0]), (z.shape[0], 1))
    a[np.abs(z[:, 0]) >= 0.9] = np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    y = np.cross(z, x)
    # R columns = [x,y,z] ⇒ Rᵀp via einsum on rows
    Rt = np.stack([x, y, z], axis=1)  # (N,3,3) rows of Rᵀ
    t = -np.einsum("nij,nj->ni", Rt, p)
    out = np.concatenate([t, z], axis=1).astype(np.float32)
    return out[0] if single else out


def _edt_signed(occ: np.ndarray, step: float) -> np.ndarray:
    inside = distance_transform_edt(occ, sampling=step)
    outside = distance_transform_edt(~occ, sampling=step)
    return (inside - outside).astype(np.float32)


def _enforce_sign(m: np.ndarray, y: np.ndarray, eps: float, m_clip: float) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64)
    m = np.clip(m, -m_clip, m_clip)
    m = np.where((y >= 0.5) & (m <= 0.0), eps, m)
    m = np.where((y < 0.5) & (m >= 0.0), -eps, m)
    return np.clip(m, -m_clip, m_clip).astype(np.float32)


def export_ird_gt_from_capability_map(
    cm,
    cfg: IrdGtConfig | None = None,
    *,
    batch_size: int = 65536,
) -> dict[str, np.ndarray]:
    cfg = cfg or IrdGtConfig()
    rng = np.random.default_rng(cfg.seed)
    from ird_playground.ird.capability_io import unpack_bits_5dof

    n_int, n_bnd, n_ext = int(cfg.n_interior), int(cfg.n_boundary), int(cfg.n_exterior)
    orients = np.asarray(cm.orientations.vectors, dtype=np.float64)
    n_orient = int(orients.shape[0])
    voxel_xyz = cm.grid.center_of(cm.voxel_ids)
    d_vals = np.asarray(cm.d_value, dtype=np.float64)
    M = int(cm.voxel_ids.shape[0])
    shape = tuple(int(s) for s in cm.grid.shape)
    nx, ny, nz = shape
    step = float(cm.grid.step_m)
    origin = np.asarray(cm.grid.origin_m, dtype=np.float64)

    print(f"[gt] unpack bitmask M={M:,} n_orient={n_orient}", flush=True)
    bits = (
        unpack_bits_5dof(np.asarray(cm.bitmask), n_orient)
        if cm.roll is None
        else np.any(cm.bitmask, axis=-1)
    )
    pos_rows, pos_oids = np.nonzero(bits)
    neg_rows, neg_oids = np.nonzero(~bits)
    print(f"[gt] pos_pairs={pos_rows.size:,} neg_on_map={neg_rows.size:,}", flush=True)

    # linear index → sparse row
    lin = (
        cm.voxel_ids[:, 0].astype(np.int64) * (ny * nz)
        + cm.voxel_ids[:, 1].astype(np.int64) * nz
        + cm.voxel_ids[:, 2].astype(np.int64)
    )
    row_of = -np.ones(nx * ny * nz, dtype=np.int32)
    row_of[lin] = np.arange(M, dtype=np.int32)

    # Precompute per-orient signed EDT → float16 memmap (avoids RAM explosion)
    edt_dir = Path(__file__).resolve().parents[2] / "data" / "ird"
    edt_dir.mkdir(parents=True, exist_ok=True)
    edt_path = edt_dir / f"_edt_cache_{os.getpid()}.dat"
    print(f"[gt] precompute {n_orient} orient EDTs → {edt_path}", flush=True)
    edt_mm = np.memmap(
        edt_path, mode="w+", dtype=np.float16, shape=(n_orient, nx, ny, nz)
    )
    t_edt0 = time.time()
    for oid in range(n_orient):
        occ = np.zeros(shape, dtype=bool)
        hit = bits[:, oid]
        if hit.any():
            ids = cm.voxel_ids[hit]
            occ[ids[:, 0], ids[:, 1], ids[:, 2]] = True
            # slight dilation: per-orient occupancy is sparse; raw EDT collapses to ~1 voxel
            if cfg.edt_dilate > 0:
                occ = binary_dilation(occ, iterations=int(cfg.edt_dilate))
            edt_mm[oid] = _edt_signed(occ, step).astype(np.float16)
        else:
            # everywhere outside: negative far margin in meters
            edt_mm[oid] = np.float16(-cfg.m_clip * cfg.sigma_p_m)
        if (oid + 1) % 20 == 0 or oid + 1 == n_orient:
            print(
                f"[gt] EDT {oid+1}/{n_orient}  elapsed={time.time()-t_edt0:.0f}s",
                flush=True,
            )
    edt_mm.flush()

    def margins_for(oids: np.ndarray, ijk: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized margin lookup from precomputed per-orient EDT."""
        oids = np.asarray(oids, dtype=np.int32)
        ijk = np.asarray(ijk, dtype=np.int32)
        y = np.asarray(y, dtype=np.float32)
        out = np.empty(oids.shape[0], dtype=np.float32)
        order = np.argsort(oids, kind="mergesort")
        o_sorted = oids[order]
        ijk_s = ijk[order]
        inb = (
            (ijk_s[:, 0] >= 0)
            & (ijk_s[:, 0] < nx)
            & (ijk_s[:, 1] >= 0)
            & (ijk_s[:, 1] < ny)
            & (ijk_s[:, 2] >= 0)
            & (ijk_s[:, 2] < nz)
        )
        start = 0
        inv_sig = 1.0 / max(cfg.sigma_p_m, 1e-6)
        while start < o_sorted.size:
            o = int(o_sorted[start])
            end = start + 1
            while end < o_sorted.size and int(o_sorted[end]) == o:
                end += 1
            sd = edt_mm[o]  # float16 view
            sl = slice(start, end)
            ii = ijk_s[sl]
            ok = inb[sl]
            vals = np.full(end - start, -cfg.m_clip, dtype=np.float32)
            if ok.any():
                ii_ok = ii[ok]
                vals[ok] = sd[ii_ok[:, 0], ii_ok[:, 1], ii_ok[:, 2]].astype(np.float32) * inv_sig
            out[order[sl]] = vals
            start = end
        return _enforce_sign(out, y, cfg.m_eps, cfg.m_clip)

    # comfort
    d_max = float(max(d_vals.max(), 1e-6))
    d_n = np.clip(d_vals / d_max, 0.0, 1.0).astype(np.float32)
    if cm.mu_mean is not None:
        mu = np.asarray(cm.mu_mean, dtype=np.float64)
        q_manip = np.clip(mu / (np.abs(mu) + 1.0), 0.0, 1.0).astype(np.float32)
        q_manip[~np.isfinite(mu)] = d_n[~np.isfinite(mu)]
    else:
        q_manip = d_n.copy()
    q_cap = np.clip(cfg.w_manip * q_manip + cfg.w_d * d_n, 0.0, 1.0).astype(np.float32)

    d_lo = float(np.percentile(d_vals, 35))
    d_hi = float(np.percentile(d_vals, 70))
    is_int = d_vals >= d_hi
    is_bnd = (d_vals >= d_lo) & (d_vals < d_hi)
    if not is_int.any():
        is_int[:] = True
    if not is_bnd.any():
        is_bnd = is_int.copy()

    pool_int = np.flatnonzero(is_int[pos_rows])
    pool_bnd = np.flatnonzero(is_bnd[pos_rows])
    if pool_int.size == 0:
        pool_int = np.arange(pos_rows.size)
    if pool_bnd.size == 0:
        pool_bnd = pool_int

    chunks_f, chunks_y, chunks_m, chunks_q, chunks_qm = [], [], [], [], []

    def flush(ps, us, y, m, rows_or_none):
        feat = features_from_p_u(ps, us)
        q = np.zeros(len(y), dtype=np.float32)
        qm = np.zeros(len(y), dtype=np.float32)
        if rows_or_none is not None:
            pos = y >= 0.5
            q[pos] = q_cap[rows_or_none[pos]]
            qm[pos] = q_manip[rows_or_none[pos]]
        chunks_f.append(feat)
        chunks_y.append(y.astype(np.float32))
        chunks_m.append(m.astype(np.float32))
        chunks_q.append(q)
        chunks_qm.append(qm)

    # Interior positives
    print(f"[gt] interior {n_int:,}", flush=True)
    for s in range(0, n_int, batch_size):
        n = min(batch_size, n_int - s)
        ti = pool_int[rng.integers(0, pool_int.size, size=n)]
        rows, oids = pos_rows[ti], pos_oids[ti]
        ijk = cm.voxel_ids[rows]
        y = np.ones(n, dtype=np.float32)
        m = margins_for(oids, ijk, y)
        flush(voxel_xyz[rows], orients[oids], y, m, rows)
        if (s // batch_size) % 4 == 0:
            print(f"[gt] interior {s+n:,}/{n_int:,}", flush=True)

    # Boundary: true face pairs (pos cell + unreachable 6-neighbour, same orient)
    print("[gt] boundary face pairs…", flush=True)
    neigh = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.int32)
    n_cand = min(300_000, pool_bnd.size)
    ti = pool_bnd[rng.choice(pool_bnd.size, size=n_cand, replace=False)]
    rows_c, oids_c = pos_rows[ti], pos_oids[ti]
    ijk0 = cm.voxel_ids[rows_c].astype(np.int32)
    assigned = np.zeros(n_cand, dtype=bool)
    bnd_r = np.empty(n_cand, dtype=np.int32)
    bnd_o = np.empty(n_cand, dtype=np.int32)
    bnd_ijk_neg = np.empty((n_cand, 3), dtype=np.int32)
    for dlt in neigh:
        j = ijk0 + dlt
        out_of = (
            (j[:, 0] < 0) | (j[:, 0] >= nx)
            | (j[:, 1] < 0) | (j[:, 1] >= ny)
            | (j[:, 2] < 0) | (j[:, 2] >= nz)
        )
        keys = (
            np.clip(j[:, 0], 0, nx - 1).astype(np.int64) * (ny * nz)
            + np.clip(j[:, 1], 0, ny - 1).astype(np.int64) * nz
            + np.clip(j[:, 2], 0, nz - 1).astype(np.int64)
        )
        r2 = row_of[keys]
        reachable_nb = np.zeros(n_cand, dtype=bool)
        ok = (~out_of) & (r2 >= 0)
        if ok.any():
            reachable_nb[ok] = bits[r2[ok], oids_c[ok]]
        fail = (~assigned) & (out_of | (~reachable_nb))
        if fail.any():
            bnd_r[fail] = rows_c[fail]
            bnd_o[fail] = oids_c[fail]
            bnd_ijk_neg[fail] = j[fail]
            assigned[fail] = True
        if assigned.all():
            break
    keep = assigned
    bnd_r = bnd_r[keep]
    bnd_o = bnd_o[keep]
    bnd_ijk_neg = bnd_ijk_neg[keep]
    print(f"[gt] face pairs kept={bnd_r.size:,}", flush=True)

    n_bp = n_bnd // 2
    n_bn = n_bnd - n_bp
    if bnd_r.size == 0:
        raise RuntimeError("no boundary face pairs found")

    pick = rng.integers(0, bnd_r.size, size=n_bp)
    rows, oids = bnd_r[pick], bnd_o[pick]
    y = np.ones(n_bp, dtype=np.float32)
    m = margins_for(oids, cm.voxel_ids[rows], y)
    flush(voxel_xyz[rows], orients[oids], y, m, rows)

    pick = rng.integers(0, bnd_r.size, size=n_bn)
    oids = bnd_o[pick]
    ijk = bnd_ijk_neg[pick]
    y = np.zeros(n_bn, dtype=np.float32)
    m = margins_for(oids, ijk, y)
    ps = origin + step * (ijk.astype(np.float64) + 0.5)
    flush(ps, orients[oids], y, m, None)
    print(f"[gt] boundary done pos={n_bp:,} neg={n_bn:,}", flush=True)

    # Jitter: re-query bitmask
    n_jit = int(cfg.n_jitter)
    print(f"[gt] jitter re-query {n_jit:,}", flush=True)
    for s in range(0, n_jit, batch_size):
        n = min(batch_size, n_jit - s)
        ti = pool_int[rng.integers(0, pool_int.size, size=n)]
        rows0, oids = pos_rows[ti], pos_oids[ti]
        ps = voxel_xyz[rows0] + rng.normal(scale=cfg.sigma_p_m, size=(n, 3))
        ijk = np.floor((ps - origin) / step).astype(np.int32)
        y = np.zeros(n, dtype=np.float32)
        rows_out = np.full(n, -1, dtype=np.int32)
        inb = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < nx)
            & (ijk[:, 1] >= 0) & (ijk[:, 1] < ny)
            & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
        )
        keys = (
            np.clip(ijk[:, 0], 0, nx - 1).astype(np.int64) * (ny * nz)
            + np.clip(ijk[:, 1], 0, ny - 1).astype(np.int64) * nz
            + np.clip(ijk[:, 2], 0, nz - 1).astype(np.int64)
        )
        r = np.full(n, -1, dtype=np.int32)
        r[inb] = row_of[keys[inb]]
        hit = np.zeros(n, dtype=bool)
        ok = r >= 0
        if ok.any():
            hit[ok] = bits[r[ok], oids[ok]]
        y[hit] = 1.0
        rows_out[hit] = r[hit]
        m = margins_for(oids, ijk, y)
        flush(ps, orients[oids], y, m, np.where(rows_out >= 0, rows_out, 0))
        if (s // batch_size) % 4 == 0:
            print(f"[gt] jitter {s+n:,}/{n_jit:,}", flush=True)

    # Exterior: true bitmask negatives + off-map
    n_hard = int(round(n_ext * cfg.hard_negative_frac))
    n_unif = max(0, n_ext - n_hard)
    print(f"[gt] exterior hard={n_hard:,} offmap={n_unif:,}", flush=True)
    if neg_rows.size and n_hard:
        pick = rng.integers(0, neg_rows.size, size=n_hard)
        rows, oids = neg_rows[pick], neg_oids[pick]
        y = np.zeros(n_hard, dtype=np.float32)
        m = margins_for(oids, cm.voxel_ids[rows], y)
        flush(voxel_xyz[rows], orients[oids], y, m, None)

    if n_unif:
        mins = voxel_xyz.min(0) - 0.2
        maxs = voxel_xyz.max(0) + 0.2
        for s in range(0, n_unif, batch_size):
            n = min(batch_size, n_unif - s)
            ps = rng.uniform(mins, maxs, size=(n, 3))
            oids = rng.integers(0, n_orient, size=n).astype(np.int32)
            ijk = np.floor((ps - origin) / step).astype(np.int32)
            y = np.zeros(n, dtype=np.float32)
            rows_out = np.full(n, -1, dtype=np.int32)
            inb = (
                (ijk[:, 0] >= 0) & (ijk[:, 0] < nx)
                & (ijk[:, 1] >= 0) & (ijk[:, 1] < ny)
                & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
            )
            keys = (
                np.clip(ijk[:, 0], 0, nx - 1).astype(np.int64) * (ny * nz)
                + np.clip(ijk[:, 1], 0, ny - 1).astype(np.int64) * nz
                + np.clip(ijk[:, 2], 0, nz - 1).astype(np.int64)
            )
            r = np.full(n, -1, dtype=np.int32)
            r[inb] = row_of[keys[inb]]
            hit = np.zeros(n, dtype=bool)
            ok = r >= 0
            if ok.any():
                hit[ok] = bits[r[ok], oids[ok]]
            y[hit] = 1.0
            rows_out[hit] = r[hit]
            m = margins_for(oids, ijk, y)
            off = ~inb
            if off.any():
                m[off] = _enforce_sign(
                    np.full(int(off.sum()), -cfg.m_clip),
                    np.zeros(int(off.sum())),
                    cfg.m_eps,
                    cfg.m_clip,
                )
            flush(ps, orients[oids], y, m, np.where(rows_out >= 0, rows_out, 0))

    features = np.concatenate(chunks_f, axis=0)
    y = np.concatenate(chunks_y, axis=0)
    m_arr = np.concatenate(chunks_m, axis=0)
    q_arr = np.concatenate(chunks_q, axis=0)
    qm_arr = np.concatenate(chunks_qm, axis=0)
    # force q=0 on negatives (jitter/offmap may have polluted)
    q_arr = np.where(y >= 0.5, q_arr, 0.0).astype(np.float32)
    qm_arr = np.where(y >= 0.5, qm_arr, 0.0).astype(np.float32)
    m_arr = _enforce_sign(m_arr, y, cfg.m_eps, cfg.m_clip)

    max_abs = np.max(np.abs(features[:, :3]), axis=0)
    scale = np.maximum(max_abs * 1.05, 0.1).astype(np.float32)
    aabb_lo, aabb_hi = -scale, scale.copy()

    bad = ((y >= 0.5) & (m_arr <= 0.0)) | ((y < 0.5) & (m_arr >= 0.0))
    if bad.any():
        raise RuntimeError(f"sign conflict {bad.mean():.4%} n={bad.sum()}")
    outside = np.any((features[:, :3] < aabb_lo) | (features[:, :3] > aabb_hi), axis=1)
    if outside.mean() > 1e-4:
        raise RuntimeError(f"outside AABB {outside.mean():.4%}")

    perm = rng.permutation(features.shape[0])
    n = int(features.shape[0])
    print(
        f"[gt] N={n:,} reach={float(y.mean()):.3f} "
        f"m+|[{m_arr[y>=0.5].min():.2f},{m_arr[y>=0.5].max():.2f}] "
        f"m-|[{m_arr[y<0.5].min():.2f},{m_arr[y<0.5].max():.2f}]",
        flush=True,
    )
    out = {
        "features": features[perm],
        "reachable": y[perm],
        "p_reach": y[perm],
        "m_gt": m_arr[perm],
        "q": q_arr[perm],
        "q_comfort": q_arr[perm],
        "q_capability": q_arr[perm],
        "q_manip": qm_arr[perm],
        "q_joint": q_arr[perm],
        "q_selfcol": q_arr[perm],
        "q_nullspace": q_arr[perm],
        "q_best": np.zeros((n, cfg.n_dof), dtype=np.float32),
        "q_candidates": np.zeros((n, cfg.k_candidates, cfg.n_dof), dtype=np.float32),
        "d": (y * q_arr)[perm],
        "aabb_lo": aabb_lo,
        "aabb_hi": aabb_hi,
        "sigma_p_m": np.array([cfg.sigma_p_m], dtype=np.float32),
        "sigma_r_deg": np.array([cfg.sigma_r_deg], dtype=np.float32),
        "feature_dim": np.array([6], dtype=np.int32),
    }
    try:
        del edt_mm
        edt_path.unlink(missing_ok=True)
    except OSError:
        pass
    return out


def save_ird_gt(path: str | Path, arrays: dict[str, np.ndarray], meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    if meta is not None:
        path.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return path


def load_ird_gt(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


def assert_gt_contract(arrays: dict[str, np.ndarray]) -> None:
    x, y, m = arrays["features"], arrays["reachable"], arrays["m_gt"]
    q = arrays["q"]
    lo, hi = arrays["aabb_lo"], arrays["aabb_hi"]
    assert x.shape[1] == 6
    assert np.isfinite(x).all() and np.isfinite(m).all() and np.isfinite(q).all()
    bad = ((y > 0.5) & (m <= 0.0)) | ((y < 0.5) & (m >= 0.0))
    assert float(bad.mean()) < 1e-5, f"sign conflict {bad.mean()}"
    outside = np.any((x[:, :3] < lo) | (x[:, :3] > hi), axis=1)
    assert float(outside.mean()) < 1e-4, f"outside AABB {outside.mean()}"


def make_synthetic_ird_gt(n: int = 4096, *, seed: int = 0, reach_radius: float = 0.6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(-1.0, 1.0, size=(n, 3))
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    features = features_from_p_u(p, u)
    dist = np.linalg.norm(p, axis=1)
    y = (dist < reach_radius).astype(np.float32)
    m = np.clip((reach_radius - dist) / reach_radius * 3.0, -3.0, 3.0)
    m = _enforce_sign(m, y, 0.05, 3.0)
    q = (np.clip(1.0 - dist / (reach_radius + 1e-6), 0, 1) * y).astype(np.float32)
    scale = np.maximum(np.max(np.abs(features[:, :3]), axis=0) * 1.05, 0.1).astype(np.float32)
    return {
        "features": features,
        "reachable": y,
        "p_reach": y,
        "m_gt": m,
        "q": q,
        "q_comfort": q,
        "q_capability": q,
        "q_manip": q,
        "q_joint": q,
        "q_selfcol": q,
        "q_nullspace": q,
        "q_best": np.zeros((n, 7), dtype=np.float32),
        "q_candidates": np.zeros((n, 4, 7), dtype=np.float32),
        "d": y * q,
        "aabb_lo": -scale,
        "aabb_hi": scale,
        "sigma_p_m": np.array([0.03], dtype=np.float32),
        "sigma_r_deg": np.array([10.0], dtype=np.float32),
        "feature_dim": np.array([6], dtype=np.int32),
    }
