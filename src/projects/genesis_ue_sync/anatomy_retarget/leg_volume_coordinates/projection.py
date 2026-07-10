"""Project anatomy vessel centerlines into canonical SMPL leg volume coordinates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .atlas import LegVolumeAtlas, VesselSkinProjection
from .atlas import _piecewise_station, _piecewise_station_unclipped, _theta_for_points, query_atlas_coordinates
from .io import read_centerline_obj, write_centerline_obj


def _side_for_label(label: str) -> str | None:
    if label.startswith("L_"):
        return "left"
    if label.startswith("R_"):
        return "right"
    return None


def _resample_polyline_max_step(points: np.ndarray, *, max_step: float = 0.0035) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(seg))
    if total <= 1.0e-8:
        return pts[:1].copy()
    count = max(int(np.ceil(total / float(max_step))) + 1, pts.shape[0])
    src_s = np.concatenate([[0.0], np.cumsum(seg)])
    dst_s = np.linspace(0.0, total, count, dtype=np.float32)
    out = np.zeros((count, 3), dtype=np.float32)
    for dim in range(3):
        out[:, dim] = np.interp(dst_s, src_s, pts[:, dim]).astype(np.float32)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def _smooth_resampled_polyline(points: np.ndarray, *, passes: int = 4, alpha: float = 0.45) -> np.ndarray:
    """Smooth a scan-prior curve without moving its topology endpoints."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] < 4:
        return pts.copy()
    out = pts.copy()
    blend = float(np.clip(alpha, 0.0, 1.0))
    for _ in range(max(1, int(passes))):
        nxt = out.copy()
        nxt[1:-1] = (1.0 - blend) * out[1:-1] + blend * 0.5 * (out[:-2] + out[2:])
        nxt[0] = pts[0]
        nxt[-1] = pts[-1]
        out = nxt
    return out.astype(np.float32)


def _atlas_station_bounds(atlas: LegVolumeAtlas) -> tuple[float, float]:
    meta = atlas.metadata or {}
    lo = float(meta.get("proximal_station", 0.0))
    hi = float(meta.get("distal_station", 1.0))
    return lo, hi


def _clip_polyline_to_atlas_chart(atlas: LegVolumeAtlas, points: np.ndarray) -> tuple[np.ndarray, bool, bool]:
    """Clip anatomy samples to the cylindrical leg chart instead of clamping foot points."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] < 2:
        return pts.copy(), True, True
    lo, hi = _atlas_station_bounds(atlas)
    station = _piecewise_station_unclipped(pts, atlas.hip, atlas.knee, atlas.ankle)
    inside = (station >= lo) & (station <= hi)
    clipped: list[np.ndarray] = []

    def append_unique(point: np.ndarray) -> None:
        p = np.asarray(point, dtype=np.float32).reshape(3)
        if not clipped or float(np.linalg.norm(clipped[-1] - p)) > 1.0e-6:
            clipped.append(p)

    def crossing_point(i: int, level: float) -> np.ndarray | None:
        denom = float(station[i + 1] - station[i])
        if abs(denom) < 1.0e-8:
            return None
        t = (float(level) - float(station[i])) / denom
        if t < -1.0e-6 or t > 1.0 + 1.0e-6:
            return None
        t = float(np.clip(t, 0.0, 1.0))
        return ((1.0 - t) * pts[i] + t * pts[i + 1]).astype(np.float32)

    if inside[0]:
        append_unique(pts[0])
    for i in range(pts.shape[0] - 1):
        levels: list[float] = []
        a = float(station[i])
        b = float(station[i + 1])
        if (a < lo <= b) or (b < lo <= a):
            levels.append(lo)
        if (a <= hi < b) or (b <= hi < a):
            levels.append(hi)
        levels.sort(key=lambda level: abs(float(level) - a))
        for level in levels:
            hit = crossing_point(i, level)
            if hit is not None:
                append_unique(hit)
        if inside[i + 1]:
            append_unique(pts[i + 1])
    if len(clipped) < 2:
        nearest = int(np.argmin(np.minimum(np.abs(station - lo), np.abs(station - hi))))
        return pts[max(0, nearest - 1) : min(pts.shape[0], nearest + 2)].copy(), bool(inside[0]), bool(inside[-1])
    return np.stack(clipped, axis=0).astype(np.float32), bool(inside[0]), bool(inside[-1])


def _continuous_skin_projection(
    atlas: LegVolumeAtlas,
    points: np.ndarray,
    *,
    xi_hint: np.ndarray | None = None,
    h_band: float = 0.045,
    continuity_weight: float = 2.5,
    max_candidates: int = 192,
    fixed_points: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a polyline to continuous points on nearby SMPL skin triangles."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if xi_hint is not None:
        xi_arr = np.asarray(xi_hint, dtype=np.float32).reshape(-1, 3)
        if xi_arr.shape[0] != pts.shape[0]:
            raise ValueError("xi_hint must have the same length as points.")
        theta = np.mod(xi_arr[:, 0], 2.0 * np.pi).astype(np.float32)
        h = np.clip(xi_arr[:, 1], 0.0, 1.0).astype(np.float32)
    else:
        h, _core = _piecewise_station(pts, atlas.hip, atlas.knee, atlas.ankle)
        theta = _theta_for_points(pts, atlas, side=atlas.side, hip=atlas.hip, knee=atlas.knee, ankle=atlas.ankle, pelvis=atlas.pelvis)
    fixed = fixed_points or {}
    all_candidates: list[list[tuple[np.ndarray, np.ndarray, float]]] = []
    for i, (p, hh, tt) in enumerate(zip(pts, h, theta, strict=True)):
        if i in fixed:
            xi_fixed, p_fixed = fixed[i]
            cand = [(np.asarray(p_fixed, dtype=np.float32).reshape(3), np.asarray(xi_fixed, dtype=np.float32).reshape(3), 0.0)]
            all_candidates.append(cand)
            continue
        cand = _candidate_triangle_projection(atlas, p, float(hh), float(tt), h_band=float(h_band))
        cand = sorted(cand, key=lambda item: float(item[2]))[: max(4, int(max_candidates))]
        all_candidates.append(cand)
    if not all_candidates:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    chosen = _solve_candidate_path(all_candidates, continuity_weight=float(continuity_weight))
    projected = [all_candidates[i][choice][0] for i, choice in enumerate(chosen)]
    xis = [all_candidates[i][choice][1] for i, choice in enumerate(chosen)]
    p_skin = np.stack(projected, axis=0).astype(np.float32) if projected else np.zeros((0, 3), dtype=np.float32)
    xi_skin = np.stack(xis, axis=0).astype(np.float32) if xis else np.zeros((0, 3), dtype=np.float32)
    return xi_skin.astype(np.float32), p_skin


def _solve_candidate_path(
    candidates: list[list[tuple[np.ndarray, np.ndarray, float]]],
    *,
    continuity_weight: float,
) -> list[int]:
    """Dynamic-program the smoothest candidate sequence for one projected line."""
    costs: list[np.ndarray] = []
    back: list[np.ndarray] = []
    first = np.asarray([c[2] for c in candidates[0]], dtype=np.float64)
    costs.append(first)
    back.append(np.full(first.shape, -1, dtype=np.int32))
    for i in range(1, len(candidates)):
        prev_pts = np.stack([c[0] for c in candidates[i - 1]], axis=0).astype(np.float64)
        cur_pts = np.stack([c[0] for c in candidates[i]], axis=0).astype(np.float64)
        prev_xi = np.stack([c[1] for c in candidates[i - 1]], axis=0).astype(np.float64)
        cur_xi = np.stack([c[1] for c in candidates[i]], axis=0).astype(np.float64)
        data = np.asarray([c[2] for c in candidates[i]], dtype=np.float64)
        trans = np.linalg.norm(prev_pts[:, None, :] - cur_pts[None, :, :], axis=2) * float(continuity_weight)
        dtheta = np.abs((prev_xi[:, None, 0] - cur_xi[None, :, 0] + np.pi) % (2.0 * np.pi) - np.pi)
        dh = np.abs(prev_xi[:, None, 1] - cur_xi[None, :, 1])
        trans = trans + 0.035 * dtheta + 0.16 * dh
        total = costs[-1][:, None] + trans + data[None, :]
        best_prev = np.argmin(total, axis=0).astype(np.int32)
        best_cost = total[best_prev, np.arange(total.shape[1])]
        costs.append(best_cost)
        back.append(best_prev)
    chosen = [int(np.argmin(costs[-1]))]
    for i in range(len(candidates) - 1, 0, -1):
        chosen.append(int(back[i][chosen[-1]]))
    return list(reversed(chosen))


_ATLAS_AXIS_CHART_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _skin_axis_chart(atlas: LegVolumeAtlas) -> tuple[np.ndarray, np.ndarray]:
    """Axis-chart (h, theta) on the current skin surface for legacy vessel projection."""
    key = (int(id(atlas)), int(np.asarray(atlas.skin_vertices).shape[0]))
    cached = _ATLAS_AXIS_CHART_CACHE.get(key)
    if cached is not None:
        return cached
    skin = np.asarray(atlas.skin_vertices, dtype=np.float32)
    h, _core = _piecewise_station(skin, atlas.hip, atlas.knee, atlas.ankle)
    theta = _theta_for_points(
        skin,
        atlas,
        side=atlas.side,
        hip=atlas.hip,
        knee=atlas.knee,
        ankle=atlas.ankle,
        pelvis=atlas.pelvis,
    )
    cached = (h.astype(np.float32), theta.astype(np.float32))
    _ATLAS_AXIS_CHART_CACHE[key] = cached
    return cached


def _candidate_triangle_projection(
    atlas: LegVolumeAtlas,
    point: np.ndarray,
    h_value: float,
    theta_value: float,
    *,
    h_band: float,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Return continuous triangle-surface candidates as (p_skin, xi, cost)."""
    p = np.asarray(point, dtype=np.float32).reshape(3)
    verts = np.asarray(atlas.skin_vertices, dtype=np.float32)
    faces = np.asarray(atlas.skin_faces, dtype=np.int32)
    skin_h, skin_theta = _skin_axis_chart(atlas)
    mask = (np.min(skin_h[faces], axis=1) <= float(h_value) + float(h_band)) & (
        np.max(skin_h[faces], axis=1) >= float(h_value) - float(h_band)
    )
    face_idx = np.flatnonzero(mask)
    if face_idx.size < 12:
        face_h = np.mean(skin_h[faces], axis=1)
        face_idx = np.argsort(np.abs(face_h - float(h_value)))[: min(128, faces.shape[0])]
    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    target_feature = _theta_h_feature(float(theta_value), float(h_value))
    for fi in face_idx.tolist():
        tri = faces[int(fi)]
        feature_tri = np.stack(
            [
                _theta_h_feature(float(skin_theta[int(vi)]), float(skin_h[int(vi)]))
                for vi in tri.tolist()
            ],
            axis=0,
        )
        _feature_q, bary = _closest_point_on_triangle(target_feature, feature_tri[0], feature_tri[1], feature_tri[2])
        q = np.sum(verts[tri] * bary.reshape(3, 1), axis=0)
        xi = _interpolate_face_xi(atlas, tri, bary)
        normal = _interpolate_face_normal(atlas, tri, bary)
        inward = -normal
        v = p - q
        along = float(v @ inward)
        line_dist = float(np.linalg.norm(v - along * inward))
        behind_penalty = max(0.0, -along) * 0.25
        theta_penalty = 0.14 * _angular_distance(float(xi[0]), float(theta_value))
        h_penalty = 0.22 * abs(float(xi[1]) - float(h_value))
        score = line_dist + behind_penalty + theta_penalty + h_penalty
        candidates.append((q.astype(np.float32), xi.astype(np.float32), float(score)))
    return candidates


def _theta_h_feature(theta: float, h_value: float) -> np.ndarray:
    return np.asarray([0.18 * np.cos(theta), 0.18 * np.sin(theta), 0.65 * float(h_value)], dtype=np.float32)


def _angular_distance(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi))


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on triangle with barycentric coordinates."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(ab @ ap)
    d2 = float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    bp = p - b
    d3 = float(ab @ bp)
    d4 = float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, 1.0e-8)
        return (a + v * ab).astype(np.float32), np.asarray([1.0 - v, v, 0.0], dtype=np.float32)
    cp = p - c
    d5 = float(ab @ cp)
    d6 = float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, 1.0e-8)
        return (a + w * ac).astype(np.float32), np.asarray([1.0 - w, 0.0, w], dtype=np.float32)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-8)
        return (b + w * (c - b)).astype(np.float32), np.asarray([0.0, 1.0 - w, w], dtype=np.float32)
    denom = max(va + vb + vc, 1.0e-8)
    v = vb / denom
    w = vc / denom
    u = 1.0 - v - w
    return (u * a + v * b + w * c).astype(np.float32), np.asarray([u, v, w], dtype=np.float32)


def _interpolate_face_xi(atlas: LegVolumeAtlas, tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    weights = np.asarray(bary, dtype=np.float32).reshape(3)
    skin_h, skin_theta = _skin_axis_chart(atlas)
    theta = skin_theta[tri]
    h = skin_h[tri]
    sin_t = float(np.sum(np.sin(theta) * weights))
    cos_t = float(np.sum(np.cos(theta) * weights))
    th = float(np.mod(np.arctan2(sin_t, cos_t), 2.0 * np.pi))
    hh = float(np.sum(h * weights))
    return np.asarray([th, hh, 0.0], dtype=np.float32)


def _interpolate_face_normal(atlas: LegVolumeAtlas, tri: np.ndarray, bary: np.ndarray) -> np.ndarray:
    n = np.sum(np.asarray(atlas.skin_normals, dtype=np.float32)[tri] * np.asarray(bary, dtype=np.float32).reshape(3, 1), axis=0)
    n = n / max(float(np.linalg.norm(n)), 1.0e-8)
    return n.astype(np.float32)


def _atlas_has_harmonic_volume(atlas: LegVolumeAtlas) -> bool:
    return int(np.asarray(atlas.harmonic_vertices).size) > 0 and int(np.asarray(atlas.harmonic_tets).size) > 0


_SURFACE_REFINER_CACHE: dict[int, object] = {}


def _surface_refiner_for_atlas(atlas: LegVolumeAtlas) -> object:
    key = int(id(atlas))
    cached = _SURFACE_REFINER_CACHE.get(key)
    if cached is not None:
        return cached
    from .surface_refine import SurfaceAtlasRefiner

    refiner = SurfaceAtlasRefiner.from_atlas(atlas)
    refiner.candidate_k = max(int(refiner.candidate_k), 256)
    _SURFACE_REFINER_CACHE[key] = refiner
    return refiner


def _skin_points_from_xi(
    atlas: LegVolumeAtlas,
    xi: np.ndarray,
    *,
    reference_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map chart coordinates (theta, h, d=0) to continuous skin triangle positions."""
    coords = np.asarray(xi, dtype=np.float32).reshape(-1, 3)
    theta = np.mod(coords[:, 0], 2.0 * np.pi).astype(np.float32)
    h = np.clip(coords[:, 1], 0.0, 1.0).astype(np.float32)
    xi_skin = np.stack([theta, h, np.zeros(theta.shape[0], dtype=np.float32)], axis=1).astype(np.float32)
    refs = None if reference_points is None else np.asarray(reference_points, dtype=np.float32).reshape(-1, 3)
    p_skin = _surface_refiner_for_atlas(atlas).xi_to_p(xi_skin, reference_points=refs)
    return p_skin.astype(np.float32), xi_skin


def _harmonic_skin_projection(
    atlas: LegVolumeAtlas,
    points: np.ndarray,
    *,
    fixed_points: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project vessel samples to d=0 by querying the baked harmonic volume field."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    _xi_vol, _p_vol = query_atlas_coordinates(atlas, pts)
    p_skin, xi_skin = _skin_points_from_xi(atlas, _xi_vol, reference_points=pts)
    fixed = fixed_points or {}
    for idx, (xi_fixed, p_fixed) in fixed.items():
        i = int(idx)
        if 0 <= i < p_skin.shape[0]:
            xi_skin[i] = np.asarray(xi_fixed, dtype=np.float32).reshape(3)
            p_skin[i] = np.asarray(p_fixed, dtype=np.float32).reshape(3)
    return xi_skin.astype(np.float32), p_skin.astype(np.float32)


def _harmonic_project_single_point(atlas: LegVolumeAtlas, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xi, p = _harmonic_skin_projection(atlas, np.asarray(point, dtype=np.float32).reshape(1, 3))
    return xi.reshape(3).astype(np.float32), p.reshape(3).astype(np.float32)


def _project_single_point(atlas: LegVolumeAtlas, point: np.ndarray, *, h_band: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    if _atlas_has_harmonic_volume(atlas):
        return _harmonic_project_single_point(atlas, point)
    p = np.asarray(point, dtype=np.float32).reshape(1, 3)
    h, _core = _piecewise_station(p, atlas.hip, atlas.knee, atlas.ankle)
    theta = _theta_for_points(p, atlas, side=atlas.side, hip=atlas.hip, knee=atlas.knee, ankle=atlas.ankle, pelvis=atlas.pelvis)
    candidates = _candidate_triangle_projection(atlas, p.reshape(3), float(h[0]), float(theta[0]), h_band=float(h_band))
    best = int(np.argmin(np.asarray([c[2] for c in candidates], dtype=np.float32)))
    q, xi, _score = candidates[best]
    return xi.astype(np.float32), q.astype(np.float32)


def _manual_junction_groups() -> list[tuple[tuple[str, int], ...]]:
    groups: list[tuple[tuple[str, int], ...]] = []
    for prefix in ("L", "R"):
        groups.extend(
            [
                ((f"{prefix}_COM_FEM_V", -1), (f"{prefix}_SUPFEMV", 0), (f"{prefix}_SAPH_V", 0), (f"{prefix}_DEEP_FEM_V", 0)),
                ((f"{prefix}_SUPFEMV", -1), (f"{prefix}_POPV", 0)),
                ((f"{prefix}_POPV", -1), (f"{prefix}_POST_TIB_V", 0), (f"{prefix}_PERONEAL_V", 0)),
            ]
        )
    return groups


def _auto_junction_groups(
    original_lines: dict[str, np.ndarray],
    *,
    tolerance: float = 0.012,
) -> list[tuple[tuple[str, int], ...]]:
    """Cluster same-side segment endpoints that already meet in anatomy space."""
    endpoints: list[tuple[str, int, np.ndarray]] = []
    for label, line in original_lines.items():
        side = _side_for_label(label)
        if side is None:
            continue
        arr = np.asarray(line, dtype=np.float32).reshape(-1, 3)
        if arr.shape[0] < 2:
            continue
        endpoints.append((label, 0, arr[0]))
        endpoints.append((label, -1, arr[-1]))
    used = np.zeros(len(endpoints), dtype=bool)
    groups: list[tuple[tuple[str, int], ...]] = []
    for i, (label_i, idx_i, point_i) in enumerate(endpoints):
        if used[i]:
            continue
        side_i = _side_for_label(label_i)
        group: list[tuple[str, int]] = [(label_i, idx_i)]
        used[i] = True
        for j in range(i + 1, len(endpoints)):
            if used[j]:
                continue
            label_j, idx_j, point_j = endpoints[j]
            if _side_for_label(label_j) != side_i:
                continue
            if float(np.linalg.norm(point_j - point_i)) <= float(tolerance):
                group.append((label_j, idx_j))
                used[j] = True
        if len(group) >= 2:
            groups.append(tuple(group))
    return groups


def _junction_groups(original_lines: dict[str, np.ndarray]) -> list[tuple[tuple[str, int], ...]]:
    groups = _manual_junction_groups()
    groups.extend(_auto_junction_groups(original_lines))
    return groups


def _projected_junction_constraints(
    original_lines: dict[str, np.ndarray],
    atlases: dict[str, LegVolumeAtlas],
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    constraints: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for group in _junction_groups(original_lines):
        available = [(label, idx) for label, idx in group if label in original_lines]
        if len(available) < 2:
            continue
        side = _side_for_label(available[0][0])
        if side is None or side not in atlases:
            continue
        original_pts = []
        resolved: list[tuple[str, int]] = []
        for label, idx in available:
            arr = np.asarray(original_lines[label], dtype=np.float32)
            pos = idx if idx >= 0 else arr.shape[0] + idx
            if 0 <= pos < arr.shape[0]:
                original_pts.append(arr[pos])
                resolved.append((label, pos))
        if len(original_pts) < 2:
            continue
        shared_original = np.median(np.stack(original_pts, axis=0), axis=0)
        shared_xi, shared_skin = _project_single_point(atlases[side], shared_original)
        for label, pos in resolved:
            constraints.setdefault(label, {})[int(pos)] = (shared_xi, shared_skin)
    return constraints


def _remap_original_index_to_resampled(
    idx: int,
    *,
    original_count: int,
    resampled_count: int,
) -> int:
    if original_count <= 1:
        return 0
    pos = int(idx) if int(idx) >= 0 else int(original_count) + int(idx)
    pos = int(np.clip(pos, 0, original_count - 1))
    if resampled_count <= 1:
        return 0
    return int(round(pos * (resampled_count - 1) / (original_count - 1)))


def _remap_fixed_points_for_resampled_line(
    fixed_points: dict[int, tuple[np.ndarray, np.ndarray]] | None,
    *,
    original_count: int,
    resampled_count: int,
    retain_start: bool = True,
    retain_end: bool = True,
) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    if not fixed_points:
        return None
    remapped: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, value in fixed_points.items():
        pos = _remap_original_index_to_resampled(
            int(idx),
            original_count=int(original_count),
            resampled_count=int(resampled_count),
        )
        if int(idx) == 0 and not retain_start:
            continue
        if int(idx) == int(original_count) - 1 and not retain_end:
            continue
        remapped[int(pos)] = value
    return remapped or None


def _remove_projection_spikes(
    original: np.ndarray,
    projected: np.ndarray,
    xi: np.ndarray,
    *,
    spike_distance: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop isolated surface projection spikes while preserving endpoints."""
    src = np.asarray(original, dtype=np.float32).reshape(-1, 3)
    proj = np.asarray(projected, dtype=np.float32).reshape(-1, 3)
    xis = np.asarray(xi, dtype=np.float32).reshape(-1, 3)
    if proj.shape[0] < 3:
        return src, proj, xis
    keep = np.ones(proj.shape[0], dtype=bool)
    for i in range(1, proj.shape[0] - 1):
        if not keep[i - 1]:
            continue
        d0 = float(np.linalg.norm(proj[i] - proj[i - 1]))
        d1 = float(np.linalg.norm(proj[i + 1] - proj[i]))
        bridge = float(np.linalg.norm(proj[i + 1] - proj[i - 1]))
        if d0 > float(spike_distance) and d1 > float(spike_distance) and bridge < 0.7 * max(d0, d1):
            keep[i] = False
    return src[keep], proj[keep], xis[keep]


def _smooth_projected_surface_line(
    atlas: LegVolumeAtlas,
    projected: np.ndarray,
    xi: np.ndarray,
    *,
    fixed_points: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    passes: int = 5,
    alpha: float = 0.42,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a d=0 curve and reproject every moved sample onto skin triangles."""
    proj = np.asarray(projected, dtype=np.float32).reshape(-1, 3)
    xis = np.asarray(xi, dtype=np.float32).reshape(-1, 3)
    if proj.shape[0] < 4:
        return proj.copy(), xis.copy()
    fixed = fixed_points or {}
    fixed_idx = set(int(k) for k in fixed.keys())
    blend = float(np.clip(alpha, 0.0, 1.0))
    out = proj.copy()
    out_xi = xis.copy()
    for _ in range(max(1, int(passes))):
        smooth = out.copy()
        smooth[1:-1] = (1.0 - blend) * out[1:-1] + blend * 0.5 * (out[:-2] + out[2:])
        theta = np.unwrap(out_xi[:, 0].astype(np.float64))
        h = out_xi[:, 1].astype(np.float64)
        theta_s = theta.copy()
        h_s = h.copy()
        theta_s[1:-1] = (1.0 - blend) * theta[1:-1] + blend * 0.5 * (theta[:-2] + theta[2:])
        h_s[1:-1] = (1.0 - blend) * h[1:-1] + blend * 0.5 * (h[:-2] + h[2:])
        for idx in fixed_idx:
            if 0 <= idx < smooth.shape[0]:
                xi_fixed, p_fixed = fixed[idx]
                smooth[idx] = np.asarray(p_fixed, dtype=np.float32).reshape(3)
                theta_s[idx] = float(np.asarray(xi_fixed, dtype=np.float32).reshape(3)[0])
                h_s[idx] = float(np.asarray(xi_fixed, dtype=np.float32).reshape(3)[1])
        next_proj = out.copy()
        next_xi = out_xi.copy()
        for i in range(out.shape[0]):
            if i in fixed_idx:
                continue
            candidates = _candidate_triangle_projection(
                atlas,
                smooth[i],
                float(np.clip(h_s[i], 0.0, 1.0)),
                float(np.mod(theta_s[i], 2.0 * np.pi)),
                h_band=0.05,
            )
            best = int(np.argmin(np.asarray([c[2] for c in candidates], dtype=np.float32)))
            next_proj[i], next_xi[i], _score = candidates[best]
        out = next_proj.astype(np.float32)
        out_xi = next_xi.astype(np.float32)
    return out.astype(np.float32), out_xi.astype(np.float32)


def _smooth_projected_xi_line(
    atlas: LegVolumeAtlas,
    projected: np.ndarray,
    xi: np.ndarray,
    *,
    fixed_points: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    passes: int = 3,
    alpha: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth (theta, h) along a vessel line and re-lookup skin positions from the chart."""
    xis = np.asarray(xi, dtype=np.float32).reshape(-1, 3)
    if xis.shape[0] < 4:
        return np.asarray(projected, dtype=np.float32).reshape(-1, 3).copy(), xis.copy()
    fixed = fixed_points or {}
    fixed_idx = {int(k) for k in fixed.keys()}
    blend = float(np.clip(alpha, 0.0, 1.0))
    out_xi = xis.copy()
    for _ in range(max(1, int(passes))):
        theta = np.unwrap(out_xi[:, 0].astype(np.float64))
        h = out_xi[:, 1].astype(np.float64)
        theta_s = theta.copy()
        h_s = h.copy()
        theta_s[1:-1] = (1.0 - blend) * theta[1:-1] + blend * 0.5 * (theta[:-2] + theta[2:])
        h_s[1:-1] = (1.0 - blend) * h[1:-1] + blend * 0.5 * (h[:-2] + h[2:])
        for idx in fixed_idx:
            if 0 <= idx < out_xi.shape[0]:
                xi_fixed, _p_fixed = fixed[idx]
                xi_fixed = np.asarray(xi_fixed, dtype=np.float32).reshape(3)
                theta_s[idx] = float(xi_fixed[0])
                h_s[idx] = float(xi_fixed[1])
        out_xi[:, 0] = np.mod(theta_s, 2.0 * np.pi).astype(np.float32)
        out_xi[:, 1] = np.clip(h_s, 0.0, 1.0).astype(np.float32)
        out_xi[:, 2] = 0.0
    p_skin, out_xi = _skin_points_from_xi(atlas, out_xi, reference_points=projected)
    for idx in fixed_idx:
        if 0 <= idx < out_xi.shape[0]:
            xi_fixed, p_fixed = fixed[idx]
            out_xi[idx] = np.asarray(xi_fixed, dtype=np.float32).reshape(3)
            p_skin[idx] = np.asarray(p_fixed, dtype=np.float32).reshape(3)
    return p_skin.astype(np.float32), out_xi.astype(np.float32)


def _pin_projected_junctions(
    projected_lines: dict[str, np.ndarray],
    xi_lines: dict[str, np.ndarray],
    original_lines: dict[str, np.ndarray],
    atlases: dict[str, LegVolumeAtlas],
) -> None:
    """Make topological vessel junctions share one skin point after projection."""
    for group in _junction_groups(original_lines):
        available = [(label, idx) for label, idx in group if label in projected_lines and label in original_lines]
        if len(available) < 2:
            continue
        side = _side_for_label(available[0][0])
        if side is None or side not in atlases:
            continue
        original_pts = []
        for label, idx in available:
            arr = np.asarray(original_lines[label], dtype=np.float32)
            pos = idx if idx >= 0 else arr.shape[0] + idx
            original_pts.append(arr[pos])
        shared_original = np.median(np.stack(original_pts, axis=0), axis=0)
        shared_xi, shared_skin = _project_single_point(atlases[side], shared_original)
        for label, idx in available:
            original_count = int(np.asarray(original_lines[label], dtype=np.float32).shape[0])
            resampled_count = int(np.asarray(projected_lines[label], dtype=np.float32).shape[0])
            pos = _remap_original_index_to_resampled(
                int(idx),
                original_count=original_count,
                resampled_count=resampled_count,
            )
            projected_lines[label][pos] = shared_skin
            xi_lines[label][pos] = shared_xi


def remap_vessel_projection_to_skin(
    source_npz: Path | str,
    atlases: dict[str, LegVolumeAtlas],
    *,
    output_obj: Path | str | None = None,
    output_npz: Path | str | None = None,
) -> tuple[VesselSkinProjection, dict[str, np.ndarray]]:
    """Remap already-baked vessel surface coordinates to the current atlas skin.

    This is the stable path when the vessel topology/coordinates were baked on a
    previous atlas and only the skin surface was smoothed or subdivided.
    """
    payload = np.load(Path(source_npz), allow_pickle=True)
    labels_in = np.asarray(payload["labels"], dtype=object)
    xi_in = np.asarray(payload["xi_skin"], dtype=np.float32).reshape(-1, 3)
    original_in = np.asarray(payload["original_points"], dtype=np.float32).reshape(-1, 3)
    ref_in = np.asarray(payload["projected_points"], dtype=np.float32).reshape(-1, 3)
    side_in = np.asarray(payload["side"], dtype=object) if "side" in payload.files else np.asarray([_side_for_label(str(v)) for v in labels_in])

    from .surface_refine import SurfaceAtlasRefiner

    refiners: dict[str, SurfaceAtlasRefiner] = {}
    for side, atlas in atlases.items():
        skin_h, skin_theta = _skin_axis_chart(atlas)
        refiners[side] = SurfaceAtlasRefiner(
            vertices=np.asarray(atlas.skin_vertices, dtype=np.float32),
            faces=np.asarray(atlas.skin_faces, dtype=np.int32),
            theta=np.asarray(skin_theta, dtype=np.float32),
            h=np.asarray(skin_h, dtype=np.float32),
            candidate_k=256,
        )

    projected = np.zeros_like(original_in, dtype=np.float32)
    projected_lines: dict[str, np.ndarray] = {}
    xi_lines: dict[str, np.ndarray] = {}
    for label_obj in labels_in:
        label = str(label_obj)
        if label in projected_lines:
            continue
        idx = np.flatnonzero(labels_in == label_obj)
        if idx.size == 0:
            continue
        side = str(side_in[idx[0]]) if side_in.size else str(_side_for_label(label))
        if side not in refiners:
            continue
        pts = refiners[side].xi_to_p(xi_in[idx], reference_points=ref_in[idx]).astype(np.float32)
        projected[idx] = pts
        projected_lines[label] = pts
        xi_lines[label] = xi_in[idx].astype(np.float32)

    projection = VesselSkinProjection(
        labels=labels_in.astype(object),
        original_points=original_in.astype(np.float32),
        projected_points=projected.astype(np.float32),
        xi_skin=xi_in.astype(np.float32),
        side=side_in.astype(object),
    )
    if output_obj is not None:
        write_centerline_obj(output_obj, projected_lines, comment="Baked vessel coordinates remapped to current atlas skin d=0")
    if output_npz is not None:
        out = Path(output_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            labels=projection.labels,
            side=projection.side,
            original_points=projection.original_points,
            projected_points=projection.projected_points,
            xi_skin=projection.xi_skin,
        )
    return projection, projected_lines


def project_vessel_centerlines_to_skin(
    centerline_obj: Path | str,
    atlases: dict[str, LegVolumeAtlas],
    *,
    output_obj: Path | str | None = None,
    output_npz: Path | str | None = None,
) -> tuple[VesselSkinProjection, dict[str, np.ndarray]]:
    """Project T-pose vessel centerlines to the SMPL skin surface d=0."""
    centerlines = read_centerline_obj(centerline_obj)
    labels: list[str] = []
    sides: list[str] = []
    original: list[np.ndarray] = []
    projected: list[np.ndarray] = []
    xis: list[np.ndarray] = []
    projected_lines: dict[str, np.ndarray] = {}
    xi_lines: dict[str, np.ndarray] = {}
    sampled_lines: dict[str, np.ndarray] = {}
    junction_constraints = _projected_junction_constraints(centerlines, atlases)
    for label, line in centerlines.items():
        side = _side_for_label(label)
        if side is None or side not in atlases:
            continue
        raw_pts = np.asarray(line, dtype=np.float32).reshape(-1, 3)
        pts = _resample_polyline_max_step(raw_pts)
        pts, retain_start, retain_end = _clip_polyline_to_atlas_chart(atlases[side], pts)
        pts = _smooth_resampled_polyline(pts)
        fixed = _remap_fixed_points_for_resampled_line(
            junction_constraints.get(label),
            original_count=raw_pts.shape[0],
            resampled_count=pts.shape[0],
            retain_start=retain_start,
            retain_end=retain_end,
        )
        if _atlas_has_harmonic_volume(atlases[side]):
            xi_skin, p_skin = _continuous_skin_projection(
                atlases[side],
                pts,
                fixed_points=fixed,
                max_candidates=64,
            )
        else:
            xi_skin, p_skin = _continuous_skin_projection(atlases[side], pts, fixed_points=fixed)
            p_skin, xi_skin = _smooth_projected_surface_line(atlases[side], p_skin, xi_skin, fixed_points=fixed)
        pts, p_skin, xi_skin = _remove_projection_spikes(pts, p_skin, xi_skin)
        sampled_lines[label] = pts.astype(np.float32)
        projected_lines[label] = p_skin.astype(np.float32)
        xi_lines[label] = xi_skin.astype(np.float32)
    _pin_projected_junctions(projected_lines, xi_lines, centerlines, atlases)
    for label, pts in sampled_lines.items():
        side = _side_for_label(label)
        if side is None or label not in projected_lines:
            continue
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        labels.extend([label] * arr.shape[0])
        sides.extend([side] * arr.shape[0])
        original.append(arr)
        projected.append(projected_lines[label].astype(np.float32))
        xis.append(xi_lines[label].astype(np.float32))
    if original:
        projection = VesselSkinProjection(
            labels=np.asarray(labels, dtype=object),
            original_points=np.vstack(original).astype(np.float32),
            projected_points=np.vstack(projected).astype(np.float32),
            xi_skin=np.vstack(xis).astype(np.float32),
            side=np.asarray(sides, dtype=object),
        )
    else:
        projection = VesselSkinProjection(
            labels=np.asarray([], dtype=object),
            original_points=np.zeros((0, 3), dtype=np.float32),
            projected_points=np.zeros((0, 3), dtype=np.float32),
            xi_skin=np.zeros((0, 3), dtype=np.float32),
            side=np.asarray([], dtype=object),
        )
    if output_obj is not None:
        write_centerline_obj(output_obj, projected_lines, comment="Vessel centerlines projected to SMPL leg skin d=0")
    if output_npz is not None:
        out = Path(output_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            labels=projection.labels,
            side=projection.side,
            original_points=projection.original_points,
            projected_points=projection.projected_points,
            xi_skin=projection.xi_skin,
        )
    return projection, projected_lines
