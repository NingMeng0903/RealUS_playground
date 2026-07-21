"""Build IRD GT v6 — stable-support boundary; MC-hit ≠ unreachable.

Contract:
  features = [p_base,tcp(3), u_base(3)]  natural 5-DoF
  exact MC hit → positive
  trusted face pair: C+ >= min_positive_support AND C- == 0
    (non-overlapping half-neighborhoods; never soft_neg <= tau)
  near-miss / unstable faces → not exported (unknown)
  margin: continuous face-pair interpolation only on trusted faces
  jitter: face-normal ±delta from same trusted faces (pos/neg half)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

LAYER_INTERIOR = 0
LAYER_BND_POS = 1
LAYER_BND_NEG = 2
LAYER_JITTER_POS = 3
LAYER_JITTER_NEG = 4
LAYER_EXTERIOR = 5

# backward-compat aliases
LAYER_JITTER = LAYER_JITTER_POS


@dataclass
class IrdGtConfig:
    n_interior: int = 300_000
    n_boundary: int = 800_000
    n_exterior: int = 400_000
    n_positive: int = 700_000
    n_negative: int = 500_000
    seed: int = 0
    comfort_from: str = "auto"
    bbox_margin_m: float = 0.20
    max_orients_per_voxel: int = 28
    hard_negative_frac: float = 0.50
    hard_negative_radius_m: float = 0.06
    sigma_p_m: float = 0.03
    sigma_r_deg: float = 10.0
    m_clip: float = 3.0
    m_eps: float = 0.05
    w_manip: float = 0.5
    w_d: float = 0.5
    k_candidates: int = 4
    n_dof: int = 7
    aabb_pad_frac: float = 0.05
    aabb_pad_min_m: float = 0.02
    n_jitter: int = 400_000
    # soft / exterior thresholds (NOT used for boundary trust)
    orient_knn: int = 7
    soft_tau: float = 0.05
    unknown_soft_max: float = 0.25
    trusted_neg_soft_max: float = 1e-6
    # v6: stable-support boundary (C+ / C-)
    min_positive_support: int = 3
    min_trusted_face_pairs: int = 5000


def features_from_p_u(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(N,6): natural 5-DoF — TCP position in base + tool axis in base."""
    p = np.asarray(p, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p[None, :]
        u = u[None, :]
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    out = np.concatenate([p, u], axis=1).astype(np.float32)
    return out[0] if single else out


def _enforce_sign(m: np.ndarray, y: np.ndarray, eps: float, m_clip: float) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64)
    m = np.clip(m, -m_clip, m_clip)
    m = np.where((y >= 0.5) & (m <= 0.0), eps, m)
    m = np.where((y < 0.5) & (m >= 0.0), -eps, m)
    return np.clip(m, -m_clip, m_clip).astype(np.float32)


def _orient_knn(orients: np.ndarray, k: int) -> np.ndarray:
    dots = orients @ orients.T
    return np.argsort(-dots, axis=1)[:, :k].astype(np.int32)


def _tangents_for_dlt(dlt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit tangent axes orthogonal to face normal ``dlt`` (int voxel step)."""
    n = np.asarray(dlt, dtype=np.int32).reshape(3)
    if abs(int(n[0])) == 1:
        t1, t2 = np.array([0, 1, 0], np.int32), np.array([0, 0, 1], np.int32)
    elif abs(int(n[1])) == 1:
        t1, t2 = np.array([1, 0, 0], np.int32), np.array([0, 0, 1], np.int32)
    else:
        t1, t2 = np.array([1, 0, 0], np.int32), np.array([0, 1, 0], np.int32)
    return t1, t2


def _half_neighborhood(dlt: np.ndarray, *, positive_side: bool) -> np.ndarray:
    """Non-overlapping half-neighborhood for support counting.

    Positive side: current + interior (-dlt) + 4 tangents.
    Negative side: current + exterior (+dlt) + 4 tangents.
    """
    t1, t2 = _tangents_for_dlt(dlt)
    interior = -dlt if positive_side else dlt
    return np.stack(
        [
            np.array([0, 0, 0], np.int32),
            interior.astype(np.int32),
            t1,
            -t1,
            t2,
            -t2,
        ],
        axis=0,
    )


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
    print(
        f"[gt] MC hits={pos_rows.size:,} ({100.0 * pos_rows.size / max(bits.size, 1):.3f}% of "
        f"{bits.size:,} sparse bins) — bit=0 is NOT verified unreachable",
        flush=True,
    )

    lin = (
        cm.voxel_ids[:, 0].astype(np.int64) * (ny * nz)
        + cm.voxel_ids[:, 1].astype(np.int64) * nz
        + cm.voxel_ids[:, 2].astype(np.int64)
    )
    row_of = -np.ones(nx * ny * nz, dtype=np.int32)
    row_of[lin] = np.arange(M, dtype=np.int32)

    knn = _orient_knn(orients, int(cfg.orient_knn))
    spat = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.int32,
    )

    def _lookup_rows(ijk: np.ndarray) -> np.ndarray:
        ijk = np.asarray(ijk, dtype=np.int32)
        n = ijk.shape[0]
        inb = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < nx)
            & (ijk[:, 1] >= 0) & (ijk[:, 1] < ny)
            & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
        )
        rows = np.full(n, -1, dtype=np.int32)
        if inb.any():
            keys = (
                ijk[inb, 0].astype(np.int64) * (ny * nz)
                + ijk[inb, 1].astype(np.int64) * nz
                + ijk[inb, 2].astype(np.int64)
            )
            rows[inb] = row_of[keys]
        return rows

    def local_orient_hit_count(
        ijk: np.ndarray,
        oids: np.ndarray,
        spatial_offsets: np.ndarray,
    ) -> np.ndarray:
        """Count local MC hits over spatial_offsets × orient-KNN (integer)."""
        ijk = np.asarray(ijk, dtype=np.int32)
        oids = np.asarray(oids, dtype=np.int32)
        n = oids.shape[0]
        o_nb = knn[oids]
        count = np.zeros(n, dtype=np.int32)
        for dlt in spatial_offsets:
            rows = _lookup_rows(ijk + dlt)
            ok = rows >= 0
            if ok.any():
                count[ok] += bits[rows[ok][:, None], o_nb[ok]].sum(axis=1).astype(np.int32)
        return count

    def soft_at(ijk: np.ndarray, oids: np.ndarray) -> np.ndarray:
        """Local MC-hit fraction (7-spatial × K-orient) — exterior diagnostic only."""
        ijk = np.asarray(ijk, dtype=np.int32)
        oids = np.asarray(oids, dtype=np.int32)
        n = oids.shape[0]
        o_nb = knn[oids]
        acc = np.zeros(n, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)
        for dlt in spat:
            rows = _lookup_rows(ijk + dlt)
            ok = rows >= 0
            if not ok.any():
                continue
            acc[ok] += bits[rows[ok][:, None], o_nb[ok]].mean(axis=1)
            cnt[ok] += 1.0
        return (acc / np.maximum(cnt, 1.0)).astype(np.float32)

    def soft_at_batched(ijk: np.ndarray, oids: np.ndarray) -> np.ndarray:
        return soft_at(ijk, oids)

    d_max = float(max(d_vals.max(), 1e-6))
    d_n = np.clip(d_vals / d_max, 0.0, 1.0).astype(np.float32)
    if cm.mu_mean is not None:
        mu = np.asarray(cm.mu_mean, dtype=np.float64)
        q_manip = np.clip(mu / (np.abs(mu) + 1.0), 0.0, 1.0).astype(np.float32)
        q_manip[~np.isfinite(mu)] = d_n[~np.isfinite(mu)]
    else:
        q_manip = d_n.copy()
    q_cap = np.clip(cfg.w_manip * q_manip + cfg.w_d * d_n, 0.0, 1.0).astype(np.float32)

    d_hi = float(np.percentile(d_vals, 70))
    d_lo = float(np.percentile(d_vals, 35))
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

    # Trusted negatives: voxels with D≈0 (no MC hit at all) — still sparse-map cells
    # or off-map. On-map zeros with soft≈0 for a random orient.
    zero_d_rows = np.flatnonzero(d_vals <= 1e-8)
    print(f"[gt] zero-D voxels (trusted exterior candidates)={zero_d_rows.size:,}", flush=True)

    chunks: dict[str, list] = {k: [] for k in ("f", "y", "ys", "cw", "m", "q", "qm", "mw", "layer", "vid", "oid")}

    def flush(ps, us, y, y_soft, cw, m, mw, layer, rows_or_none, oids):
        feat = features_from_p_u(ps, us)
        n = len(y)
        q = np.zeros(n, dtype=np.float32)
        qm = np.zeros(n, dtype=np.float32)
        vid = np.full(n, -1, dtype=np.int32)
        if rows_or_none is not None:
            pos = np.asarray(y) >= 0.5
            rows_or_none = np.asarray(rows_or_none, dtype=np.int32)
            ok = pos & (rows_or_none >= 0)
            if ok.any():
                q[ok] = q_cap[rows_or_none[ok]]
                qm[ok] = q_manip[rows_or_none[ok]]
            vid[:] = rows_or_none
        layer_arr = np.asarray(layer, dtype=np.int32)
        if layer_arr.ndim == 0:
            layer_arr = np.full(n, int(layer_arr), dtype=np.int32)
        chunks["f"].append(feat)
        chunks["y"].append(np.asarray(y, dtype=np.float32))
        chunks["ys"].append(np.asarray(y_soft, dtype=np.float32))
        chunks["cw"].append(np.asarray(cw, dtype=np.float32))
        chunks["m"].append(np.asarray(m, dtype=np.float32))
        chunks["q"].append(q)
        chunks["qm"].append(qm)
        chunks["mw"].append(np.asarray(mw, dtype=np.float32))
        chunks["layer"].append(layer_arr)
        chunks["vid"].append(vid)
        chunks["oid"].append(np.asarray(oids, dtype=np.int32))

    # --- Interior: exact MC hits only (trusted positives) ---
    print(f"[gt] interior exact-hits {n_int:,}", flush=True)
    for s in range(0, n_int, batch_size):
        n = min(batch_size, n_int - s)
        ti = pool_int[rng.integers(0, pool_int.size, size=n)]
        rows, oids = pos_rows[ti], pos_oids[ti]
        y = np.ones(n, dtype=np.float32)
        ys = np.ones(n, dtype=np.float32)
        cw = np.ones(n, dtype=np.float32)
        m = np.full(n, cfg.m_eps, dtype=np.float32)
        mw = np.zeros(n, dtype=np.float32)
        flush(voxel_xyz[rows], orients[oids], y, ys, cw, m, mw, LAYER_INTERIOR, rows, oids)

    # --- Boundary face pairs with stable-support filter (v6) ---
    print("[gt] boundary face pairs…", flush=True)
    neigh = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.int32)
    n_cand = min(400_000, pool_bnd.size)
    ti = pool_bnd[rng.choice(pool_bnd.size, size=n_cand, replace=False)]
    rows_c, oids_c = pos_rows[ti], pos_oids[ti]
    ijk0 = cm.voxel_ids[rows_c].astype(np.int32)
    assigned = np.zeros(n_cand, dtype=bool)
    bnd_r = np.empty(n_cand, dtype=np.int32)
    bnd_o = np.empty(n_cand, dtype=np.int32)
    bnd_ijk_neg = np.empty((n_cand, 3), dtype=np.int32)
    bnd_dlt = np.empty((n_cand, 3), dtype=np.int32)
    for dlt in neigh:
        j = ijk0 + dlt
        out_of = (
            (j[:, 0] < 0) | (j[:, 0] >= nx) | (j[:, 1] < 0) | (j[:, 1] >= ny) | (j[:, 2] < 0) | (j[:, 2] >= nz)
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
            bnd_dlt[fail] = dlt
            assigned[fail] = True
        if assigned.all():
            break
    keep = assigned
    bnd_r, bnd_o, bnd_ijk_neg, bnd_dlt = (
        bnd_r[keep],
        bnd_o[keep],
        bnd_ijk_neg[keep],
        bnd_dlt[keep],
    )
    print(f"[gt] face pairs kept={bnd_r.size:,}", flush=True)
    if bnd_r.size == 0:
        raise RuntimeError("no boundary face pairs found")

    # v6: C+ / C- support on non-overlapping half-neighborhoods
    print("[gt] stable-support filter (C+, C-)…", flush=True)
    ijk_pos = cm.voxel_ids[bnd_r].astype(np.int32)
    support_pos = np.zeros(bnd_r.size, dtype=np.int32)
    support_neg = np.zeros(bnd_r.size, dtype=np.int32)
    for dlt in neigh:
        mask = np.all(bnd_dlt == dlt, axis=1)
        if not mask.any():
            continue
        pos_off = _half_neighborhood(dlt, positive_side=True)
        neg_off = _half_neighborhood(dlt, positive_side=False)
        support_pos[mask] = local_orient_hit_count(ijk_pos[mask], bnd_o[mask], pos_off)
        support_neg[mask] = local_orient_hit_count(bnd_ijk_neg[mask], bnd_o[mask], neg_off)

    cmin = int(cfg.min_positive_support)
    trusted = (support_pos >= cmin) & (support_neg == 0)
    if trusted.sum() < int(cfg.min_trusted_face_pairs) and cmin > 2:
        print(
            f"[gt] C+>={cmin} & C-=0 → {trusted.sum():,} pairs; "
            f"relaxing min_positive_support to 2",
            flush=True,
        )
        cmin = 2
        trusted = (support_pos >= cmin) & (support_neg == 0)

    trusted_idx = np.flatnonzero(trusted)
    qs = [0.0, 0.1, 0.5, 0.9, 1.0]
    print(
        f"[gt] support_pos quantiles={np.quantile(support_pos, qs).astype(int).tolist()} "
        f"support_neg quantiles={np.quantile(support_neg, qs).astype(int).tolist()}",
        flush=True,
    )
    print(
        f"[gt] trusted faces={trusted_idx.size:,}/{bnd_r.size:,} "
        f"(C+>={cmin} & C-=0); rejected={bnd_r.size - trusted_idx.size:,}",
        flush=True,
    )
    if trusted_idx.size < int(cfg.min_trusted_face_pairs):
        raise RuntimeError(
            f"Not enough trusted boundary pairs ({trusted_idx.size} < {cfg.min_trusted_face_pairs}). "
            "Increase MC coverage or lower min_positive_support explicitly — "
            "do NOT fall back to all face pairs."
        )

    # Cap n_bnd / n_jitter to available trusted diversity (no fake continuity)
    n_trusted = int(trusted_idx.size)
    n_bnd_eff = min(n_bnd, max(n_trusted * 2, n_trusted))
    if n_bnd_eff < n_bnd:
        print(f"[gt] capping boundary samples {n_bnd:,} → {n_bnd_eff:,}", flush=True)
    n_bnd = n_bnd_eff

    print(f"[gt] boundary interpolate {n_bnd:,} (trusted face pairs only)", flush=True)
    for s in range(0, n_bnd, batch_size):
        n = min(batch_size, n_bnd - s)
        pick = trusted_idx[rng.integers(0, trusted_idx.size, size=n)]
        rows = bnd_r[pick]
        oids = bnd_o[pick]
        ijk_neg = bnd_ijk_neg[pick]
        p_pos = voxel_xyz[rows]
        p_neg = origin + step * (ijk_neg.astype(np.float64) + 0.5)
        alpha = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
        ps = (1.0 - alpha[:, None]) * p_pos + alpha[:, None] * p_neg
        m = ((0.5 - alpha) * step / max(cfg.sigma_p_m, 1e-6)).astype(np.float32)
        m = np.clip(m, -cfg.m_clip, cfg.m_clip)
        y = (alpha < 0.5).astype(np.float32)
        m = _enforce_sign(m, y, cfg.m_eps, cfg.m_clip)
        ys = y.copy()
        cw = np.ones(n, dtype=np.float32)
        mw = np.ones(n, dtype=np.float32)
        layer = np.where(y >= 0.5, LAYER_BND_POS, LAYER_BND_NEG).astype(np.int32)
        rows_q = np.where(y >= 0.5, rows, -1)
        flush(ps, orients[oids], y, ys, cw, m, mw, layer, rows_q, oids)

    # --- Jitter from face normal (pos/neg half-half), NOT isotropic MC-noise ---
    n_jit = int(cfg.n_jitter)
    n_jit = min(n_jit, max(n_trusted * 2, n_trusted))
    n_jp = n_jit // 2
    n_jn = n_jit - n_jp
    print(f"[gt] face-normal jitter pos={n_jp:,} neg={n_jn:,}", flush=True)

    def face_jitter(n_samples: int, positive: bool):
        pick = trusted_idx[rng.integers(0, trusted_idx.size, size=n_samples)]
        rows = bnd_r[pick]
        oids = bnd_o[pick]
        ijk_neg = bnd_ijk_neg[pick]
        p_plus = voxel_xyz[rows]
        p_minus = origin + step * (ijk_neg.astype(np.float64) + 0.5)
        p_face = 0.5 * (p_plus + p_minus)
        nrm = p_minus - p_plus
        nn = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        n_hat = nrm / nn
        delta = rng.uniform(0.05 * step, 0.45 * step, size=n_samples)
        # tangent noise
        a = np.where(np.abs(n_hat[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        t1 = np.cross(a, n_hat)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
        t2 = np.cross(n_hat, t1)
        rad = rng.uniform(0.0, 0.35 * step, size=n_samples)
        ang = rng.uniform(0.0, 2 * np.pi, size=n_samples)
        tang = (rad * np.cos(ang))[:, None] * t1 + (rad * np.sin(ang))[:, None] * t2
        if positive:
            ps = p_face - delta[:, None] * n_hat + tang
            y = np.ones(n_samples, dtype=np.float32)
            layer = LAYER_JITTER_POS
            m = (delta / max(cfg.sigma_p_m, 1e-6)).astype(np.float32)
            m = np.clip(m, cfg.m_eps, cfg.m_clip)
            rows_out = rows
        else:
            ps = p_face + delta[:, None] * n_hat + tang
            y = np.zeros(n_samples, dtype=np.float32)
            layer = LAYER_JITTER_NEG
            m = (-delta / max(cfg.sigma_p_m, 1e-6)).astype(np.float32)
            m = np.clip(m, -cfg.m_clip, -cfg.m_eps)
            rows_out = np.full(n_samples, -1, dtype=np.int32)
        ys = y.copy()
        cw = np.ones(n_samples, dtype=np.float32)
        mw = np.ones(n_samples, dtype=np.float32)
        flush(ps, orients[oids], y, ys, cw, m, mw, layer, rows_out, oids)

    for s in range(0, n_jp, batch_size):
        face_jitter(min(batch_size, n_jp - s), True)
    for s in range(0, n_jn, batch_size):
        face_jitter(min(batch_size, n_jn - s), False)

    # --- Exterior trusted negatives: soft≈0 on-map bit=0 + off-map ---
    # Saved voxels almost never have D=0 (they exist because some orient hit).
    n_hard = int(round(n_ext * cfg.hard_negative_frac))
    n_unif = max(0, n_ext - n_hard)
    print(f"[gt] trusted exterior soft0={n_hard:,} offmap={n_unif:,}", flush=True)

    if n_hard:
        # Rejection-sample (row, oid) with exact bit=0 and local soft≈0 (far from MC hits)
        got = 0
        attempts = 0
        max_attempts = max(40, (n_hard // batch_size) * 80)
        thr = float(cfg.trusted_neg_soft_max)
        while got < n_hard and attempts < max_attempts:
            attempts += 1
            n = min(batch_size * 4, max(batch_size, (n_hard - got) * 4))
            rows = rng.integers(0, M, size=n).astype(np.int32)
            oids = rng.integers(0, n_orient, size=n).astype(np.int32)
            hit = bits[rows, oids]
            soft = soft_at_batched(cm.voxel_ids[rows], oids)
            keep = (~hit) & (soft <= thr)
            if not keep.any() and attempts > max_attempts // 4:
                thr = float(cfg.soft_tau)  # relax once if too strict
                keep = (~hit) & (soft <= thr)
            if not keep.any():
                continue
            take = min(int(keep.sum()), n_hard - got)
            sel = np.flatnonzero(keep)[:take]
            rows, oids, soft = rows[sel], oids[sel], soft[sel]
            n = len(rows)
            y = np.zeros(n, dtype=np.float32)
            ys = soft
            cw = np.ones(n, dtype=np.float32)
            m = np.full(n, -cfg.m_eps, dtype=np.float32)
            mw = np.zeros(n, dtype=np.float32)
            flush(voxel_xyz[rows], orients[oids], y, ys, cw, m, mw, LAYER_EXTERIOR, None, oids)
            got += n
        print(f"[gt] soft0 exterior accepted={got:,} attempts={attempts} thr={thr}", flush=True)

    if n_unif:
        mins = voxel_xyz.min(0) - float(cfg.bbox_margin_m)
        maxs = voxel_xyz.max(0) + float(cfg.bbox_margin_m)
        got = 0
        attempts = 0
        max_attempts = max(20, (n_unif // batch_size) * 40)
        while got < n_unif and attempts < max_attempts:
            attempts += 1
            n = min(batch_size, n_unif - got)
            ps = rng.uniform(mins, maxs, size=(n, 3))
            oids = rng.integers(0, n_orient, size=n).astype(np.int32)
            ijk = np.floor((ps - origin) / step).astype(np.int32)
            inb = (
                (ijk[:, 0] >= 0) & (ijk[:, 0] < nx)
                & (ijk[:, 1] >= 0) & (ijk[:, 1] < ny)
                & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
            )
            soft = np.zeros(n, dtype=np.float32)
            if inb.any():
                soft[inb] = soft_at_batched(ijk[inb], oids[inb])
            # keep clearly off-map OR soft≈0; never exact hits
            hit = np.zeros(n, dtype=bool)
            keys = (
                np.clip(ijk[:, 0], 0, nx - 1).astype(np.int64) * (ny * nz)
                + np.clip(ijk[:, 1], 0, ny - 1).astype(np.int64) * nz
                + np.clip(ijk[:, 2], 0, nz - 1).astype(np.int64)
            )
            r = np.full(n, -1, dtype=np.int32)
            r[inb] = row_of[keys[inb]]
            ok = r >= 0
            if ok.any():
                hit[ok] = bits[r[ok], oids[ok]]
            keep = ((~inb) | (soft <= cfg.soft_tau)) & (~hit)
            if not keep.any():
                continue
            n = int(keep.sum())
            y = np.zeros(n, dtype=np.float32)
            ys = soft[keep]
            cw = np.ones(n, dtype=np.float32)
            m = np.full(n, -cfg.m_eps, dtype=np.float32)
            mw = np.zeros(n, dtype=np.float32)
            flush(ps[keep], orients[oids[keep]], y, ys, cw, m, mw, LAYER_EXTERIOR, None, oids[keep])
            got += n
        print(f"[gt] offmap/soft0 exterior accepted={got:,} attempts={attempts}", flush=True)

    features = np.concatenate(chunks["f"], axis=0)
    y = np.concatenate(chunks["y"], axis=0)
    y_soft = np.concatenate(chunks["ys"], axis=0)
    cw = np.concatenate(chunks["cw"], axis=0)
    m_arr = np.concatenate(chunks["m"], axis=0)
    q_arr = np.concatenate(chunks["q"], axis=0)
    qm_arr = np.concatenate(chunks["qm"], axis=0)
    mw_arr = np.concatenate(chunks["mw"], axis=0)
    layer = np.concatenate(chunks["layer"], axis=0)
    vid = np.concatenate(chunks["vid"], axis=0)
    oid = np.concatenate(chunks["oid"], axis=0)

    q_arr = np.where(y >= 0.5, q_arr, 0.0).astype(np.float32)
    qm_arr = np.where(y >= 0.5, qm_arr, 0.0).astype(np.float32)
    mw_pos = mw_arr > 0
    if mw_pos.any():
        m_arr[mw_pos] = _enforce_sign(m_arr[mw_pos], y[mw_pos], cfg.m_eps, cfg.m_clip)

    max_abs = np.max(np.abs(features[:, :3]), axis=0)
    scale = np.maximum(max_abs * 1.05, 0.1).astype(np.float32)
    aabb_lo, aabb_hi = -scale, scale.copy()

    # sign only where margin supervised
    bad = mw_pos & (((y >= 0.5) & (m_arr <= 0.0)) | ((y < 0.5) & (m_arr >= 0.0)))
    if bad.any():
        raise RuntimeError(f"sign conflict on margin_weight>0: {bad.mean():.4%} n={bad.sum()}")
    outside = np.any((features[:, :3] < aabb_lo) | (features[:, :3] > aabb_hi), axis=1)
    if outside.mean() > 1e-4:
        raise RuntimeError(f"outside AABB {outside.mean():.4%}")

    ijk_feat = np.floor((features[:, :3] - origin) / step).astype(np.int32)
    block = (
        (np.clip(ijk_feat[:, 0], 0, nx - 1) // 8).astype(np.int64) * 1_000_000
        + (np.clip(ijk_feat[:, 1], 0, ny - 1) // 8).astype(np.int64) * 1_000
        + (np.clip(ijk_feat[:, 2], 0, nz - 1) // 8).astype(np.int64)
        + oid.astype(np.int64) * 10_000_000_000
    )

    perm = rng.permutation(features.shape[0])
    n = int(features.shape[0])
    supervised = cw > 0
    print(
        f"[gt] N={n:,} reach={float(y.mean()):.3f} supervised={float(supervised.mean()):.3f} "
        f"sup_pos={float(y[supervised].mean()) if supervised.any() else 0:.3f} "
        f"layers={dict(zip(*np.unique(layer, return_counts=True)))}",
        flush=True,
    )
    return {
        "features": features[perm],
        "reachable": y[perm],
        "p_reach": y[perm],
        "y_soft": y_soft[perm],
        "cls_weight": cw[perm],
        "m_gt": m_arr[perm],
        "margin_weight": mw_arr[perm],
        "layer_id": layer[perm],
        "voxel_id": vid[perm],
        "orient_id": oid[perm],
        "block_id": block[perm],
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
        "feature_kind": np.array([1], dtype=np.int32),
        "label_kind": np.array([3], dtype=np.int32),  # 3 = stable-support v6
    }


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
    if x.shape[1] not in (6, 8, 9):
        raise AssertionError(f"expected feature dim 6, 8, or 9, got {x.shape[1]}")
    if "feature_dim" in arrays:
        declared = int(np.asarray(arrays["feature_dim"]).reshape(-1)[0])
        if declared != int(x.shape[1]):
            raise AssertionError(
                f"feature_dim metadata ({declared}) != features.shape[1] ({x.shape[1]})"
            )
    assert np.isfinite(x).all() and np.isfinite(m).all() and np.isfinite(q).all()
    cw = arrays.get("cls_weight")
    mw = arrays.get("margin_weight")
    if mw is not None:
        mask = mw > 0
        if mask.any():
            bad = ((y[mask] > 0.5) & (m[mask] <= 0.0)) | ((y[mask] < 0.5) & (m[mask] >= 0.0))
            assert float(bad.mean()) < 1e-5, f"sign conflict {bad.mean()}"
    if cw is not None:
        assert float((cw >= 0).mean()) == 1.0
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
    mw = (np.abs(dist - reach_radius) < 0.15).astype(np.float32)
    # unknown band
    unknown = (np.abs(dist - reach_radius) >= 0.15) & (np.abs(dist - reach_radius) < 0.25)
    cw = (~unknown).astype(np.float32)
    q = (np.clip(1.0 - dist / (reach_radius + 1e-6), 0, 1) * y).astype(np.float32)
    layer = np.full(n, LAYER_INTERIOR, dtype=np.int32)
    layer = np.where(mw > 0, np.where(y >= 0.5, LAYER_BND_POS, LAYER_BND_NEG), layer)
    scale = np.maximum(np.max(np.abs(features[:, :3]), axis=0) * 1.05, 0.1).astype(np.float32)
    return {
        "features": features,
        "reachable": y,
        "p_reach": y,
        "y_soft": y,
        "cls_weight": cw,
        "m_gt": m,
        "margin_weight": mw,
        "layer_id": layer,
        "voxel_id": np.arange(n, dtype=np.int32),
        "orient_id": np.zeros(n, dtype=np.int32),
        "block_id": (np.floor(p[:, 0] * 4).astype(np.int64) * 1000 + np.floor(p[:, 1] * 4).astype(np.int64)),
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
        "feature_kind": np.array([1], dtype=np.int32),
        "label_kind": np.array([2], dtype=np.int32),
    }
