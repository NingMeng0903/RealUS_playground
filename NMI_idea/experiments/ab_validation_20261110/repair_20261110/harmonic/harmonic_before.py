"""Harmonic leg volume fields: surface Laplace-Beltrami and 3D FEM Dirichlet solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LegSide = str


@dataclass(frozen=True)
class MeshQualityOptions:
    area_epsilon: float = 1.0e-14
    edge_epsilon: float = 1.0e-12
    max_aspect_ratio: float = 120.0
    cotangent_abs_max: float = 50.0
    mass_diagonal_epsilon: float = 1.0e-12
    regularization: float = 1.0e-8


@dataclass(frozen=True)
class LaplacianBuildResult:
    laplacian: object
    mass: object
    mass_diagonal: np.ndarray
    face_areas: np.ndarray
    face_normals: np.ndarray
    valid_faces: np.ndarray


@dataclass(frozen=True)
class HarmonicVolumeMesh:
    vertices: np.ndarray
    tets: np.ndarray
    skin_vertex_indices: np.ndarray
    medial_vertex_indices: np.ndarray


@dataclass(frozen=True)
class LegHarmonicFields:
    skin_h: np.ndarray
    skin_theta: np.ndarray
    skin_d: np.ndarray
    vol_h: np.ndarray
    vol_theta: np.ndarray
    vol_d: np.ndarray
    volume_mesh: HarmonicVolumeMesh
    medial_curve_h: np.ndarray
    medial_curve_points: np.ndarray
    metadata: dict[str, object]


def _cotangent(a: np.ndarray, b: np.ndarray, max_abs: float) -> float:
    cross_norm = float(np.linalg.norm(np.cross(a, b)))
    if cross_norm <= 1.0e-20:
        return 0.0
    raw = float(np.dot(a, b) / cross_norm)
    return float(np.clip(raw, -float(max_abs), float(max_abs)))


def build_cotangent_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    options: MeshQualityOptions | None = None,
) -> LaplacianBuildResult:
    from scipy import sparse

    opts = options or MeshQualityOptions()
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    n_vertices = int(verts.shape[0])
    n_faces = int(tris.shape[0])
    face_areas = np.zeros((n_faces,), dtype=np.float64)
    face_normals = np.zeros((n_faces, 3), dtype=np.float64)
    valid_faces = np.zeros((n_faces,), dtype=bool)
    mass_diag = np.zeros((n_vertices,), dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_weight(i: int, j: int, value: float) -> None:
        if not np.isfinite(value):
            return
        rows.extend((i, j))
        cols.extend((j, i))
        data.extend((value, value))

    for face_idx, (i, j, k) in enumerate(tris):
        p0, p1, p2 = verts[i], verts[j], verts[k]
        e01, e12, e20 = p1 - p0, p2 - p1, p0 - p2
        edge_lengths = np.asarray([np.linalg.norm(e01), np.linalg.norm(e12), np.linalg.norm(e20)], dtype=np.float64)
        normal_raw = np.cross(e01, p2 - p0)
        area = 0.5 * float(np.linalg.norm(normal_raw))
        face_areas[face_idx] = area
        if area > 0.0:
            face_normals[face_idx] = normal_raw / max(2.0 * area, 1.0e-20)
        if not np.isfinite(area) or area <= float(opts.area_epsilon) or np.any(edge_lengths <= float(opts.edge_epsilon)):
            continue
        valid_faces[face_idx] = True
        mass_diag[[i, j, k]] += area / 3.0
        cot_i = _cotangent(p1 - p0, p2 - p0, float(opts.cotangent_abs_max))
        cot_j = _cotangent(p2 - p1, p0 - p1, float(opts.cotangent_abs_max))
        cot_k = _cotangent(p0 - p2, p1 - p2, float(opts.cotangent_abs_max))
        add_weight(j, k, 0.5 * cot_i)
        add_weight(k, i, 0.5 * cot_j)
        add_weight(i, j, 0.5 * cot_k)

    weights = sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    weights.sum_duplicates()
    degree = np.asarray(weights.sum(axis=1)).reshape(-1)
    laplacian = sparse.diags(degree, format="csr") - weights
    if float(opts.regularization) > 0.0:
        laplacian = laplacian + sparse.eye(n_vertices, format="csr") * float(opts.regularization)
    mass_diag = np.maximum(mass_diag, float(opts.mass_diagonal_epsilon))
    mass = sparse.diags(mass_diag, format="csr")
    return LaplacianBuildResult(
        laplacian=laplacian,
        mass=mass,
        mass_diagonal=mass_diag,
        face_areas=face_areas,
        face_normals=face_normals,
        valid_faces=valid_faces,
    )


def _dirichlet_fixed_unique(
    fixed_indices: np.ndarray,
    fixed_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_map: dict[int, float] = {}
    for idx, val in zip(np.asarray(fixed_indices, dtype=np.int64).reshape(-1), np.asarray(fixed_values, dtype=np.float64).reshape(-1), strict=True):
        fixed_map[int(idx)] = float(val)
    fixed = np.asarray(list(fixed_map.keys()), dtype=np.int64)
    values = np.asarray([fixed_map[int(i)] for i in fixed], dtype=np.float64)
    return fixed, values


class _DirichletSolver:
    """Reuse sparse LU on the interior block for repeated Dirichlet solves."""

    def __init__(self, laplacian: object, *, vertex_count: int) -> None:
        from scipy import sparse

        self._laplacian = laplacian
        self._vertex_count = int(vertex_count)
        self._factors: dict[tuple[int, ...], object] = {}

    def solve(
        self,
        *,
        fixed_indices: np.ndarray,
        fixed_values: np.ndarray,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> np.ndarray:
        from scipy import sparse
        from scipy.sparse import linalg as spla

        fixed, values = _dirichlet_fixed_unique(fixed_indices, fixed_values)
        u = np.zeros((self._vertex_count,), dtype=np.float64)
        u[fixed] = values
        interior_mask = np.ones((self._vertex_count,), dtype=bool)
        interior_mask[fixed] = False
        interior = np.nonzero(interior_mask)[0]
        if interior.size:
            key = tuple(int(v) for v in np.sort(fixed).tolist())
            if key not in self._factors:
                block = self._laplacian[interior][:, interior]
                if sparse.issparse(block):
                    block = block.tocsc()
                self._factors[key] = spla.splu(block)
            rhs = -self._laplacian[interior][:, fixed] @ u[fixed]
            u[interior] = self._factors[key].solve(rhs)
        u = np.nan_to_num(u, nan=0.0, posinf=1.0, neginf=0.0)
        if clip_min is not None or clip_max is not None:
            lo = -np.inf if clip_min is None else float(clip_min)
            hi = np.inf if clip_max is None else float(clip_max)
            u = np.clip(u, lo, hi)
        return u.astype(np.float64)


def solve_dirichlet_values(
    laplacian: object,
    *,
    fixed_indices: np.ndarray,
    fixed_values: np.ndarray,
    vertex_count: int,
    clip_min: float | None = None,
    clip_max: float | None = None,
    solver: _DirichletSolver | None = None,
) -> np.ndarray:
    if solver is not None:
        return solver.solve(
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    from scipy.sparse import linalg as spla

    fixed, values = _dirichlet_fixed_unique(fixed_indices, fixed_values)
    u = np.zeros((int(vertex_count),), dtype=np.float64)
    u[fixed] = values
    interior_mask = np.ones((int(vertex_count),), dtype=bool)
    interior_mask[fixed] = False
    interior = np.nonzero(interior_mask)[0]
    if interior.size:
        rhs = -laplacian[interior][:, fixed] @ u[fixed]
        u[interior] = spla.spsolve(laplacian[interior][:, interior], rhs)
    u = np.nan_to_num(u, nan=0.0, posinf=1.0, neginf=0.0)
    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else float(clip_min)
        hi = np.inf if clip_max is None else float(clip_max)
        u = np.clip(u, lo, hi)
    return u.astype(np.float64)


def compute_face_gradients(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalar: np.ndarray,
    face_areas: np.ndarray,
    face_normals: np.ndarray,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    values = np.asarray(scalar, dtype=np.float64).reshape(-1)
    gradients = np.zeros((tris.shape[0], 3), dtype=np.float64)
    for face_idx, (i, j, k) in enumerate(tris):
        area = float(face_areas[face_idx])
        if area <= 1.0e-20:
            continue
        p0, p1, p2 = verts[[i, j, k]]
        n = face_normals[face_idx]
        edge_i, edge_j, edge_k = p2 - p1, p0 - p2, p1 - p0
        gradients[face_idx] = (
            values[i] * np.cross(n, edge_i) + values[j] * np.cross(n, edge_j) + values[k] * np.cross(n, edge_k)
        ) / (2.0 * area)
    return gradients


def area_weighted_vertex_vectors(
    faces: np.ndarray,
    face_vectors: np.ndarray,
    face_areas: np.ndarray,
    *,
    vertex_count: int,
    normalize: bool = False,
) -> np.ndarray:
    tris = np.asarray(faces, dtype=np.int64)
    vectors = np.asarray(face_vectors, dtype=np.float64)
    areas = np.asarray(face_areas, dtype=np.float64).reshape(-1)
    accum = np.zeros((int(vertex_count), 3), dtype=np.float64)
    for face_idx, tri in enumerate(tris):
        weight = float(areas[face_idx])
        if weight <= 0.0:
            continue
        accum[tri] += vectors[face_idx].reshape(1, 3) * weight
    if normalize:
        norms = np.linalg.norm(accum, axis=1, keepdims=True)
        valid = norms[:, 0] > 1.0e-12
        accum[valid] /= norms[valid]
    return accum.astype(np.float64)


def _assemble_tet_laplacian(vertices: np.ndarray, tets: np.ndarray) -> object:
    from scipy import sparse

    verts = np.asarray(vertices, dtype=np.float64)
    elems = np.asarray(tets, dtype=np.int64)
    n = int(verts.shape[0])
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for tet in elems:
        idx = np.asarray(tet, dtype=np.int64).reshape(4)
        coords = verts[idx]
        mat = np.column_stack([np.ones(4, dtype=np.float64), coords])
        det_a = float(np.linalg.det(mat))
        if abs(det_a) <= 1.0e-18:
            continue
        inv_a = np.linalg.inv(mat)
        grads = inv_a[:, 1:4]
        vol = abs(det_a) / 6.0
        stiff = vol * (grads @ grads.T)
        for i in range(4):
            for j in range(4):
                val = float(stiff[i, j])
                if abs(val) <= 1.0e-18:
                    continue
                rows.extend((int(idx[i]), int(idx[j])))
                cols.extend((int(idx[j]), int(idx[i])))
                data.extend((val, val))

    mat = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    mat.sum_duplicates()
    return (mat + sparse.eye(n, format="csr") * 1.0e-8).tocsr()


def _filter_tets(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    elems = np.asarray(tets, dtype=np.int64)
    keep: list[np.ndarray] = []
    for tet in elems:
        p = verts[tet]
        vol = float(np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0])))
        if abs(vol) <= 1.0e-18:
            continue
        edges = [
            np.linalg.norm(p[1] - p[0]),
            np.linalg.norm(p[2] - p[0]),
            np.linalg.norm(p[3] - p[0]),
            np.linalg.norm(p[2] - p[1]),
            np.linalg.norm(p[3] - p[1]),
            np.linalg.norm(p[3] - p[2]),
        ]
        if max(edges) / max(min(edges), 1.0e-8) > 25.0:
            continue
        keep.append(tet.astype(np.int32))
    if not keep:
        return np.zeros((0, 4), dtype=np.int32)
    return np.stack(keep, axis=0).astype(np.int32)


def _local_frame(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    pelvis: np.ndarray,
    station: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from .atlas import _axis_point_and_tangent

    axis, tangent = _axis_point_and_tangent(hip, knee, ankle, float(station))
    medial = np.asarray(pelvis, dtype=np.float64).reshape(3) - np.asarray(hip, dtype=np.float64).reshape(3)
    e1 = medial - float(medial @ tangent) * tangent
    if float(np.linalg.norm(e1)) < 1.0e-8:
        e1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-8)
    e2 = np.cross(tangent, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1.0e-8)
    return axis.astype(np.float64), tangent.astype(np.float64), e1, e2


def _skin_radius_lookup(
    skin_vertices: np.ndarray,
    skin_station: np.ndarray,
    skin_theta: np.ndarray,
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .atlas import _piecewise_station

    skin = np.asarray(skin_vertices, dtype=np.float64)
    station = np.asarray(skin_station, dtype=np.float64)
    theta = np.mod(np.asarray(skin_theta, dtype=np.float64), 2.0 * np.pi)
    _s, axis_pts = _piecewise_station(skin.astype(np.float32), hip, knee, ankle)
    del _s
    radial = np.linalg.norm(skin - axis_pts, axis=1)
    station_bins = np.linspace(float(station.min()), float(station.max()), 28)
    theta_bins = np.linspace(0.0, 2.0 * np.pi, 33)
    max_radius = np.zeros((station_bins.size - 1, theta_bins.size - 1), dtype=np.float64)
    for si in range(station_bins.size - 1):
        s_mask = (station >= station_bins[si]) & (station < station_bins[si + 1])
        for ti in range(theta_bins.size - 1):
            t_mask = (theta >= theta_bins[ti]) & (theta < theta_bins[ti + 1])
            mask = s_mask & t_mask
            if np.any(mask):
                max_radius[si, ti] = float(np.quantile(radial[mask], 0.96))
    return station_bins, theta_bins, max_radius


def _lookup_skin_radius(
    station: float,
    theta: float,
    station_bins: np.ndarray,
    theta_bins: np.ndarray,
    max_radius: np.ndarray,
) -> float:
    si = int(np.clip(np.searchsorted(station_bins, station, side="right") - 1, 0, max_radius.shape[0] - 1))
    ti = int(np.clip(np.searchsorted(theta_bins, np.mod(theta, 2.0 * np.pi), side="right") - 1, 0, max_radius.shape[1] - 1))
    val = float(max_radius[si, ti])
    if val <= 1.0e-8:
        return float(np.max(max_radius))
    return val


def _dedupe_points(points: np.ndarray, *, tol: float = 1.0e-5) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return pts
    keep: list[np.ndarray] = []
    for p in pts:
        if not any(float(np.linalg.norm(p - q)) < float(tol) for q in keep):
            keep.append(p)
    if not keep:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(keep, axis=0).astype(np.float64)


def _collect_skin_section_points(
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    skin_station: np.ndarray,
    h_value: float,
    *,
    tol: float = 0.003,
) -> np.ndarray:
    verts = np.asarray(skin_vertices, dtype=np.float64)
    faces = np.asarray(skin_faces, dtype=np.int64)
    h_field = np.asarray(skin_station, dtype=np.float64)
    h_target = float(h_value)
    pts_list: list[np.ndarray] = []
    for tri in faces:
        for a, b in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
            ha = float(h_field[a] - h_target)
            hb = float(h_field[b] - h_target)
            if abs(ha) <= float(tol):
                pts_list.append(verts[a])
            if abs(hb) <= float(tol):
                pts_list.append(verts[b])
            if ha * hb < 0.0:
                denom = ha - hb
                if abs(denom) > 1.0e-10:
                    t = ha / denom
                    if -1.0e-6 <= t <= 1.0 + 1.0e-6:
                        t = float(np.clip(t, 0.0, 1.0))
                        pts_list.append((1.0 - t) * verts[a] + t * verts[b])
    if not pts_list:
        return np.zeros((0, 3), dtype=np.float64)
    return _dedupe_points(np.stack(pts_list, axis=0), tol=max(float(tol), 1.0e-5))


def _project_points_to_plane(
    points: np.ndarray,
    origin: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    rel = np.asarray(points, dtype=np.float64).reshape(-1, 3) - np.asarray(origin, dtype=np.float64).reshape(1, 3)
    t = np.asarray(tangent, dtype=np.float64).reshape(3)
    rel = rel - (rel @ t)[:, None] * t.reshape(1, 3)
    e1v = np.asarray(e1, dtype=np.float64).reshape(3)
    e2v = np.asarray(e2, dtype=np.float64).reshape(3)
    return np.stack([rel @ e1v, rel @ e2v], axis=1).astype(np.float64)


def _medial_uv_from_boundary(uv: np.ndarray, *, grid_size: int = 128) -> tuple[float, float]:
    from matplotlib.path import Path as MplPath
    from scipy.ndimage import distance_transform_edt

    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0, 0.0
    center = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(ang)]
    mn = ordered.min(axis=0)
    mx = ordered.max(axis=0)
    span = np.maximum(mx - mn, 1.0e-4)
    pad = np.maximum(span * 0.08, 0.002)
    origin = mn - pad
    extent = span + 2.0 * pad
    res = int(grid_size)
    scale = float(res / max(float(np.max(extent)), 1.0e-6))
    yy, xx = np.mgrid[0:res, 0:res]
    sample_uv = np.stack(
        [
            origin[0] + (xx.ravel() + 0.5) / scale,
            origin[1] + (yy.ravel() + 0.5) / scale,
        ],
        axis=1,
    )
    inside = MplPath(ordered).contains_points(sample_uv)
    grid = inside.reshape(res, res)
    if not np.any(grid):
        return float(center[0]), float(center[1])
    dist = distance_transform_edt(grid)
    iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
    u = float(origin[0] + (float(ix) + 0.5) / scale)
    v = float(origin[1] + (float(iy) + 0.5) / scale)
    return u, v


def _point_to_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a_v = np.asarray(a, dtype=np.float64).reshape(3)
    b_v = np.asarray(b, dtype=np.float64).reshape(3)
    ab = b_v - a_v
    denom = float(ab @ ab)
    if denom <= 1.0e-16:
        return np.linalg.norm(pts - a_v.reshape(1, 3), axis=1)
    t = np.clip(((pts - a_v.reshape(1, 3)) @ ab) / denom, 0.0, 1.0)
    closest = a_v.reshape(1, 3) + t[:, None] * ab.reshape(1, 3)
    return np.linalg.norm(pts - closest, axis=1)


def _point_to_polyline_distance(points: np.ndarray, curve_points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    curve = np.asarray(curve_points, dtype=np.float64).reshape(-1, 3)
    if curve.shape[0] == 0:
        return np.full((pts.shape[0],), np.inf, dtype=np.float64)
    if curve.shape[0] == 1:
        return np.linalg.norm(pts - curve.reshape(1, 3), axis=1)
    dist = np.full((pts.shape[0],), np.inf, dtype=np.float64)
    for i in range(curve.shape[0] - 1):
        dist = np.minimum(dist, _point_to_segment_distance(pts, curve[i], curve[i + 1]))
    return dist


def _distance_to_skin_vertices(points: np.ndarray, skin_vertices: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    skin = np.asarray(skin_vertices, dtype=np.float64).reshape(-1, 3)
    try:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(skin).query(pts, k=1)
        return np.asarray(dist, dtype=np.float64).reshape(-1)
    except Exception:
        dist = np.linalg.norm(skin[:, None, :] - pts[None, :, :], axis=2)
        return np.min(dist, axis=0)


def compute_volume_d_distance_ratio(
    volume_vertices: np.ndarray,
    skin_vertices: np.ndarray,
    medial_curve_points: np.ndarray,
) -> np.ndarray:
    """Normalized distance from skin toward shrink medial core: d=0 skin, d=1 core."""
    verts = np.asarray(volume_vertices, dtype=np.float64).reshape(-1, 3)
    d_skin = _distance_to_skin_vertices(verts, skin_vertices)
    d_medial = _point_to_polyline_distance(verts, medial_curve_points)
    return np.clip(d_skin / np.maximum(d_skin + d_medial, 1.0e-8), 0.0, 1.0).astype(np.float64)


def _point_to_segment_distance_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(point, dtype=np.float64).reshape(2)
    a_v = np.asarray(a, dtype=np.float64).reshape(2)
    b_v = np.asarray(b, dtype=np.float64).reshape(2)
    ab = b_v - a_v
    denom = float(ab @ ab)
    if denom <= 1.0e-16:
        return float(np.linalg.norm(p - a_v))
    t = float(np.clip(((p - a_v) @ ab) / denom, 0.0, 1.0))
    closest = a_v + t * ab
    return float(np.linalg.norm(p - closest))


def _dist_to_polygon_boundary_2d(point: np.ndarray, polygon_uv: np.ndarray) -> float:
    poly = np.asarray(polygon_uv, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 2:
        return 0.0
    dist = np.inf
    for i in range(poly.shape[0]):
        j = (i + 1) % poly.shape[0]
        dist = min(dist, _point_to_segment_distance_2d(point, poly[i], poly[j]))
    return float(dist)


def boundary_uv_from_section_segments(
    segments: np.ndarray,
    core: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    """Order iso-h skin section segments into one closed 2D polygon in (u,v)."""
    segs = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    if segs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    core_v = np.asarray(core, dtype=np.float64).reshape(3)
    e1_v = np.asarray(e1, dtype=np.float64).reshape(3)
    e2_v = np.asarray(e2, dtype=np.float64).reshape(3)
    flat = segs.reshape(-1, 3)
    rel = flat - core_v.reshape(1, 3)
    uv = np.stack([rel @ e1_v, rel @ e2_v], axis=1)
    keep: list[np.ndarray] = []
    for p in uv:
        if not any(float(np.linalg.norm(p - q)) < 1.0e-5 for q in keep):
            keep.append(p)
    if len(keep) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    ordered = np.stack(keep, axis=0)
    center = np.mean(ordered, axis=0)
    ang = np.arctan2(ordered[:, 1] - center[1], ordered[:, 0] - center[0])
    return ordered[np.argsort(ang)].astype(np.float64)


def d_value_slice_uv(point_uv: np.ndarray, boundary_uv: np.ndarray) -> float:
    """Slice d from medial origin (0,0) to skin polygon boundary."""
    from matplotlib.path import Path as MplPath

    uv = np.asarray(point_uv, dtype=np.float64).reshape(2)
    poly = np.asarray(boundary_uv, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 3 or not MplPath(poly).contains_point(uv):
        return float("nan")
    r_core = float(np.linalg.norm(uv))
    r_skin = _dist_to_polygon_boundary_2d(uv, poly)
    return float(r_skin / max(r_skin + r_core, 1.0e-10))


def _dist_to_polygon_boundary_2d_batch(points: np.ndarray, polygon_uv: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(polygon_uv, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 2:
        return np.zeros((pts.shape[0],), dtype=np.float64)
    dist = np.full((pts.shape[0],), np.inf, dtype=np.float64)
    for i in range(poly.shape[0]):
        j = (i + 1) % poly.shape[0]
        a_v = poly[i]
        b_v = poly[j]
        ab = b_v - a_v
        denom = float(ab @ ab)
        if denom <= 1.0e-16:
            seg_dist = np.linalg.norm(pts - a_v.reshape(1, 2), axis=1)
        else:
            t = np.clip(((pts - a_v.reshape(1, 2)) @ ab) / denom, 0.0, 1.0)
            closest = a_v.reshape(1, 2) + t[:, None] * ab.reshape(1, 2)
            seg_dist = np.linalg.norm(pts - closest, axis=1)
        dist = np.minimum(dist, seg_dist)
    return dist.astype(np.float64)


def build_cross_section_d_grid_slice(
    boundary_uv: np.ndarray,
    *,
    radius: float,
    grid_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth slice d grid: analytic distance ratio inside the skin polygon only."""
    from matplotlib.path import Path as MplPath

    poly = np.asarray(boundary_uv, dtype=np.float64).reshape(-1, 2)
    r = max(float(radius), 1.0e-3)
    axis_vals = np.linspace(-r, r, int(grid_size), dtype=np.float64)
    gu, gv = np.meshgrid(axis_vals, axis_vals, indexing="xy")
    gd = np.full(gu.shape, np.nan, dtype=np.float64)
    if poly.shape[0] < 3:
        return gu.astype(np.float32), gv.astype(np.float32), gd.astype(np.float32)
    path = MplPath(poly)
    flat_uv = np.stack([gu.ravel(), gv.ravel()], axis=1)
    inside = path.contains_points(flat_uv)
    if np.any(inside):
        uv_in = flat_uv[inside]
        r_core = np.linalg.norm(uv_in, axis=1)
        r_skin = _dist_to_polygon_boundary_2d_batch(uv_in, poly)
        vals = r_skin / np.maximum(r_skin + r_core, 1.0e-10)
        out = np.full((flat_uv.shape[0],), np.nan, dtype=np.float64)
        out[inside] = vals
        gd = out.reshape(gu.shape)
    return gu.astype(np.float32), gv.astype(np.float32), gd.astype(np.float32)


def trace_streamlines_slice_uv(
    boundary_uv: np.ndarray,
    *,
    n_rays: int = 16,
    step_size: float = 0.00035,
    start_radius: float = 0.0015,
    max_steps: int = 240,
) -> list[np.ndarray]:
    """Integrate -grad d in the slice plane from medial core toward skin."""
    from matplotlib.path import Path as MplPath

    poly = np.asarray(boundary_uv, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 3:
        return []
    path = MplPath(poly)

    def grad_at(uv: np.ndarray) -> np.ndarray:
        eps = max(float(step_size) * 0.75, 5.0e-4)
        g = np.zeros(2, dtype=np.float64)
        for axis in range(2):
            step = np.zeros(2, dtype=np.float64)
            step[axis] = eps
            vp = d_value_slice_uv(uv + step, poly)
            vm = d_value_slice_uv(uv - step, poly)
            if np.isfinite(vp) and np.isfinite(vm):
                g[axis] = (vp - vm) / (2.0 * eps)
        return g

    lines: list[np.ndarray] = []
    for theta0 in np.linspace(0.0, 2.0 * np.pi, int(n_rays), endpoint=False):
        direction = np.asarray([np.cos(float(theta0)), np.sin(float(theta0))], dtype=np.float64)
        pos = direction * float(start_radius)
        if not path.contains_point(pos):
            continue
        pts: list[np.ndarray] = [pos.copy()]
        for _ in range(int(max_steps)):
            d0 = d_value_slice_uv(pos, poly)
            if not np.isfinite(d0) or float(d0) <= 0.015:
                break
            grad = grad_at(pos)
            gn = float(np.linalg.norm(grad))
            if gn <= 1.0e-10:
                step_dir = direction
            else:
                step_dir = -grad / gn
            trial = pos + float(step_size) * step_dir
            if not path.contains_point(trial):
                # Binary search to boundary so lines stop at skin, not outside.
                lo, hi = 0.0, 1.0
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    probe = pos + mid * float(step_size) * step_dir
                    if path.contains_point(probe):
                        lo = mid
                    else:
                        hi = mid
                pos = pos + lo * float(step_size) * step_dir
                pts.append(pos.copy())
                break
            pos = trial
            pts.append(pos.copy())
        if len(pts) >= 2:
            lines.append(np.stack(pts, axis=0).astype(np.float32))
    return lines


def compute_shrink_medial_curve(
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    skin_station: np.ndarray,
    *,
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    pelvis: np.ndarray,
    stations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Medial core from 2D cross-section shrink (max-inscribed circle via distance transform)."""
    h_out = np.asarray(stations, dtype=np.float32).reshape(-1)
    medial_pts = np.zeros((h_out.shape[0], 3), dtype=np.float32)
    for i, s in enumerate(h_out.tolist()):
        axis, tangent, e1, e2 = _local_frame(hip, knee, ankle, pelvis, float(s))
        section = _collect_skin_section_points(skin_vertices, skin_faces, skin_station, float(s))
        if section.shape[0] < 3:
            medial_pts[i] = axis.astype(np.float32)
            continue
        uv = _project_points_to_plane(section, axis, e1, e2, tangent)
        u, v = _medial_uv_from_boundary(uv)
        medial_pts[i] = (axis + u * e1 + v * e2).astype(np.float32)
    return h_out, medial_pts


def medial_point_at_station(medial_curve_h: np.ndarray, medial_curve_points: np.ndarray, station: float) -> np.ndarray:
    h = np.asarray(medial_curve_h, dtype=np.float64).reshape(-1)
    pts = np.asarray(medial_curve_points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("medial curve is empty")
    if pts.shape[0] == 1:
        return pts[0].astype(np.float32)
    s = float(station)
    if s <= float(h[0]):
        return pts[0].astype(np.float32)
    if s >= float(h[-1]):
        return pts[-1].astype(np.float32)
    j = int(np.searchsorted(h, s))
    t = (s - float(h[j - 1])) / max(float(h[j] - h[j - 1]), 1.0e-8)
    return ((1.0 - t) * pts[j - 1] + t * pts[j]).astype(np.float32)


def build_volume_delaunay_mesh(
    skin_vertices: np.ndarray,
    skin_station: np.ndarray,
    skin_theta: np.ndarray,
    *,
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    pelvis: np.ndarray,
    proximal_station: float,
    distal_station: float,
    interior_station_count: int = 24,
    interior_theta_count: int = 18,
    interior_radial_count: int = 5,
    medial_curve_h: np.ndarray | None = None,
    medial_curve_points: np.ndarray | None = None,
) -> HarmonicVolumeMesh:
    from scipy.spatial import Delaunay

    skin = np.asarray(skin_vertices, dtype=np.float64)
    n_skin = int(skin.shape[0])
    station_bins, theta_bins, max_radius = _skin_radius_lookup(
        skin, skin_station, skin_theta, hip, knee, ankle
    )
    interior_pts: list[np.ndarray] = []
    medial_vertex_indices: list[int] = []
    stations = np.linspace(float(proximal_station) + 0.03, float(distal_station) - 0.03, int(interior_station_count))
    thetas = np.linspace(0.0, 2.0 * np.pi, int(interior_theta_count), endpoint=False)
    radial_fracs = np.linspace(0.10, 0.92, int(interior_radial_count))
    for s in stations:
        if medial_curve_h is not None and medial_curve_points is not None and int(np.asarray(medial_curve_h).size) > 0:
            center = medial_point_at_station(medial_curve_h, medial_curve_points, float(s))
        else:
            center, _tangent, _e1, _e2 = _local_frame(hip, knee, ankle, pelvis, float(s))
            center = center.astype(np.float32)
        _axis, tangent, e1, e2 = _local_frame(hip, knee, ankle, pelvis, float(s))
        del _axis
        medial_vertex_indices.append(n_skin + len(interior_pts))
        interior_pts.append(np.asarray(center, dtype=np.float64).reshape(3))
        for theta in thetas:
            skin_r = _lookup_skin_radius(float(s), float(theta), station_bins, theta_bins, max_radius)
            rel = np.cos(float(theta)) * e1 + np.sin(float(theta)) * e2
            for frac in radial_fracs:
                interior_pts.append(np.asarray(center, dtype=np.float64).reshape(3) + float(frac) * float(skin_r) * rel)
    if interior_pts:
        interior = np.stack(interior_pts, axis=0).astype(np.float64)
    else:
        interior = np.zeros((0, 3), dtype=np.float64)
    all_vertices = np.vstack([skin, interior]).astype(np.float64)
    delaunay = Delaunay(all_vertices)
    tets = _filter_tets(all_vertices, delaunay.simplices.astype(np.int32))
    return HarmonicVolumeMesh(
        vertices=all_vertices.astype(np.float32),
        tets=tets,
        skin_vertex_indices=np.arange(n_skin, dtype=np.int32),
        medial_vertex_indices=np.asarray(medial_vertex_indices, dtype=np.int32),
    )


def _surface_theta_from_medial(
    skin_vertices: np.ndarray,
    *,
    side: LegSide,
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    pelvis: np.ndarray,
) -> np.ndarray:
    from .atlas import _piecewise_station, _side_sign

    pts = np.asarray(skin_vertices, dtype=np.float64)
    station, axis_pts = _piecewise_station(pts.astype(np.float32), hip, knee, ankle)
    medial = np.asarray(pelvis, dtype=np.float64).reshape(3) - np.asarray(hip, dtype=np.float64).reshape(3)
    theta = np.zeros(pts.shape[0], dtype=np.float64)
    for i, s in enumerate(station.tolist()):
        _axis, tangent = _local_frame(hip, knee, ankle, pelvis, float(s))[0:2]
        del _axis
        e1 = medial - float(medial @ tangent) * tangent
        if float(np.linalg.norm(e1)) < 1.0e-8:
            e1 = np.asarray([-_side_sign(side), 0.0, 0.0], dtype=np.float64)
        e1 = e1 / max(float(np.linalg.norm(e1)), 1.0e-8)
        e2 = np.cross(tangent, e1)
        e2 = e2 / max(float(np.linalg.norm(e2)), 1.0e-8)
        rel = pts[i] - axis_pts[i]
        rel = rel - float(rel @ tangent) * tangent
        ang = float(np.arctan2(float(rel @ e2), float(rel @ e1)))
        if ang < 0.0:
            ang += 2.0 * np.pi
        theta[i] = ang
    return theta.astype(np.float64)


def _volume_radial_fraction(
    points: np.ndarray,
    *,
    side: LegSide,
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    pelvis: np.ndarray,
    station_bins: np.ndarray,
    theta_bins: np.ndarray,
    max_radius: np.ndarray,
) -> np.ndarray:
    from .atlas import _piecewise_station

    pts = np.asarray(points, dtype=np.float64)
    station, axis_pts = _piecewise_station(pts.astype(np.float32), hip, knee, ankle)
    theta = _surface_theta_from_medial(
        pts,
        side=side,
        hip=hip,
        knee=knee,
        ankle=ankle,
        pelvis=pelvis,
    )
    radial = np.linalg.norm(pts - axis_pts, axis=1)
    skin_r = np.array(
        [_lookup_skin_radius(float(s), float(t), station_bins, theta_bins, max_radius) for s, t in zip(station, theta, strict=True)],
        dtype=np.float64,
    )
    return np.clip(radial / np.maximum(skin_r, 1.0e-8), 0.0, 1.25)


def solve_leg_harmonic_fields(
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    skin_station: np.ndarray,
    *,
    side: LegSide,
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    pelvis: np.ndarray,
    proximal_station: float,
    distal_station: float,
    proximal_band: float = 0.04,
    distal_band: float = 0.04,
    inner_core_radius_frac: float = 0.14,
    interior_station_count: int = 24,
    interior_theta_count: int = 18,
    interior_radial_count: int = 5,
    medial_station_count: int = 48,
) -> LegHarmonicFields:
    skin = np.asarray(skin_vertices, dtype=np.float64)
    faces = np.asarray(skin_faces, dtype=np.int64)
    station = np.asarray(skin_station, dtype=np.float64)
    lap = build_cotangent_laplacian(skin, faces)

    prox = np.where(station <= float(proximal_station) + float(proximal_band))[0]
    dist = np.where(station >= float(distal_station) - float(distal_band))[0]
    if prox.size < 3:
        prox = np.argsort(station)[: max(3, skin.shape[0] // 32)]
    if dist.size < 3:
        dist = np.argsort(station)[-max(3, skin.shape[0] // 32) :]
    skin_h = solve_dirichlet_values(
        lap.laplacian,
        fixed_indices=np.concatenate([prox, dist]),
        fixed_values=np.concatenate([np.zeros(prox.size), np.ones(dist.size)]),
        vertex_count=int(skin.shape[0]),
        clip_min=0.0,
        clip_max=1.0,
    )

    theta_seed = _surface_theta_from_medial(
        skin,
        side=side,
        hip=hip,
        knee=knee,
        ankle=ankle,
        pelvis=pelvis,
    )
    skin_theta = np.mod(theta_seed, 2.0 * np.pi).astype(np.float64)

    medial_stations = np.linspace(float(proximal_station) + 0.03, float(distal_station) - 0.03, int(medial_station_count))
    medial_curve_h, medial_curve_points = compute_shrink_medial_curve(
        skin,
        faces,
        station,
        hip=hip,
        knee=knee,
        ankle=ankle,
        pelvis=pelvis,
        stations=medial_stations,
    )

    volume_mesh = build_volume_delaunay_mesh(
        skin.astype(np.float32),
        station.astype(np.float32),
        skin_theta.astype(np.float32),
        hip=hip,
        knee=knee,
        ankle=ankle,
        pelvis=pelvis,
        proximal_station=float(proximal_station),
        distal_station=float(distal_station),
        interior_station_count=int(interior_station_count),
        interior_theta_count=int(interior_theta_count),
        interior_radial_count=int(interior_radial_count),
        medial_curve_h=medial_curve_h,
        medial_curve_points=medial_curve_points,
    )
    vol_lap = _assemble_tet_laplacian(volume_mesh.vertices, volume_mesh.tets)
    n_vol = int(volume_mesh.vertices.shape[0])
    skin_idx = np.asarray(volume_mesh.skin_vertex_indices, dtype=np.int64)

    vol_solver = _DirichletSolver(vol_lap, vertex_count=n_vol)
    vol_d = compute_volume_d_distance_ratio(
        volume_mesh.vertices,
        skin,
        medial_curve_points,
    ).astype(np.float32)
    vol_h = vol_solver.solve(
        fixed_indices=skin_idx,
        fixed_values=skin_h,
        clip_min=0.0,
        clip_max=1.0,
    )
    vol_theta = vol_solver.solve(
        fixed_indices=skin_idx,
        fixed_values=skin_theta,
    )
    vol_theta = np.mod(vol_theta, 2.0 * np.pi)

    metadata = {
        "surface_solver": "cotan_laplace_beltrami_dirichlet",
        "volume_solver": "tet_linear_fem_dirichlet_h_theta",
        "volume_d_method": "normalized_skin_to_shrink_medial_distance",
        "core_method": "cross_section_shrink_distance_transform",
        "volume_vertex_count": int(n_vol),
        "volume_tet_count": int(volume_mesh.tets.shape[0]),
        "proximal_anchor_count": int(prox.size),
        "distal_anchor_count": int(dist.size),
        "medial_curve_count": int(medial_curve_h.shape[0]),
        "medial_anchor_count": int(volume_mesh.medial_vertex_indices.size),
    }
    return LegHarmonicFields(
        skin_h=skin_h.astype(np.float32),
        skin_theta=skin_theta.astype(np.float32),
        skin_d=np.zeros(skin.shape[0], dtype=np.float32),
        vol_h=vol_h.astype(np.float32),
        vol_theta=vol_theta.astype(np.float32),
        vol_d=vol_d.astype(np.float32),
        volume_mesh=volume_mesh,
        medial_curve_h=medial_curve_h.astype(np.float32),
        medial_curve_points=medial_curve_points.astype(np.float32),
        metadata=metadata,
    )


def sample_volume_xi_points(
    fields: LegHarmonicFields,
    *,
    d_levels: tuple[float, ...],
    h_tolerance: float = 0.04,
    max_points_per_level: int = 4000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    verts = np.asarray(fields.volume_mesh.vertices, dtype=np.float32)
    h = np.asarray(fields.vol_h, dtype=np.float32)
    theta = np.asarray(fields.vol_theta, dtype=np.float32)
    d = np.asarray(fields.vol_d, dtype=np.float32)
    points: list[np.ndarray] = []
    xi: list[np.ndarray] = []
    for level in d_levels:
        target = float(level)
        band = np.abs(d - target) <= max(0.035, 0.5 * (1.0 / max(len(d_levels), 1)))
        idx = np.flatnonzero(band)
        if idx.size == 0:
            idx = np.argsort(np.abs(d - target))[:128]
        if idx.size > int(max_points_per_level):
            idx = rng.choice(idx, size=int(max_points_per_level), replace=False)
        pts = verts[idx]
        xi_level = np.stack([theta[idx], h[idx], d[idx]], axis=1).astype(np.float32)
        points.append(pts)
        xi.append(xi_level)
    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.vstack(points).astype(np.float32), np.vstack(xi).astype(np.float32)


_VOLUME_INTERP_CACHE: dict[tuple[int | str, ...], tuple[object, object, object, object]] = {}


def _volume_interp_cache_key(fields: LegHarmonicFields) -> tuple[int | str, ...]:
    atlas_id = fields.metadata.get("atlas_id")
    if atlas_id is not None:
        side = str(fields.metadata.get("side", ""))
        return ("atlas", side, int(atlas_id))
    return (
        int(fields.volume_mesh.vertices.ctypes.data),
        int(fields.vol_h.ctypes.data),
        int(fields.vol_theta.ctypes.data),
    )


def _skin_vertices_from_fields(fields: LegHarmonicFields) -> np.ndarray:
    verts = np.asarray(fields.volume_mesh.vertices, dtype=np.float64)
    idx = np.asarray(fields.volume_mesh.skin_vertex_indices, dtype=np.int64)
    if idx.size:
        return verts[idx]
    return verts


def _volume_interpolators(fields: LegHarmonicFields) -> tuple[object, object, object, object]:
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import cKDTree

    key = _volume_interp_cache_key(fields)
    cached = _VOLUME_INTERP_CACHE.get(key)
    if cached is not None:
        return cached
    verts = np.asarray(fields.volume_mesh.vertices, dtype=np.float64)
    theta = np.asarray(fields.vol_theta, dtype=np.float64)
    cached = (
        LinearNDInterpolator(verts, np.asarray(fields.vol_h, dtype=np.float64)),
        LinearNDInterpolator(verts, np.cos(theta)),
        LinearNDInterpolator(verts, np.sin(theta)),
        cKDTree(verts),
    )
    _VOLUME_INTERP_CACHE[key] = cached
    return cached


def interpolate_volume_field(
    fields: LegHarmonicFields,
    points: np.ndarray,
    *,
    _interps: tuple[object, object, object, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    h_interp, cos_interp, sin_interp, nearest_tree = _interps or _volume_interpolators(fields)
    h_out = np.asarray(h_interp(pts), dtype=np.float64)
    cos_out = np.asarray(cos_interp(pts), dtype=np.float64)
    sin_out = np.asarray(sin_interp(pts), dtype=np.float64)
    invalid = ~(np.isfinite(h_out) & np.isfinite(cos_out) & np.isfinite(sin_out))
    if np.any(invalid):
        _dist, idx = nearest_tree.query(pts[invalid], k=1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        theta = np.asarray(fields.vol_theta, dtype=np.float64)
        h_out[invalid] = np.asarray(fields.vol_h, dtype=np.float64)[idx]
        cos_out[invalid] = np.cos(theta[idx])
        sin_out[invalid] = np.sin(theta[idx])
    t_out = np.mod(np.arctan2(sin_out, cos_out), 2.0 * np.pi)
    d_out = compute_volume_d_distance_ratio(pts, _skin_vertices_from_fields(fields), fields.medial_curve_points)
    h_out = np.clip(np.nan_to_num(h_out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    t_out = np.nan_to_num(t_out, nan=0.0)
    d_out = np.clip(np.nan_to_num(d_out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return h_out.astype(np.float32), t_out.astype(np.float32), d_out.astype(np.float32)


def grad_d_at(
    fields: LegHarmonicFields,
    point: np.ndarray,
    *,
    eps: float = 0.0025,
) -> np.ndarray:
    """Central-difference gradient of the analytic distance-ratio d field."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    skin = _skin_vertices_from_fields(fields)
    medial = fields.medial_curve_points
    grad = np.zeros(3, dtype=np.float64)
    d0 = float(compute_volume_d_distance_ratio(p.reshape(1, 3), skin, medial)[0])
    for axis in range(3):
        step = np.zeros(3, dtype=np.float64)
        step[axis] = float(eps)
        dp = float(compute_volume_d_distance_ratio((p + step).reshape(1, 3), skin, medial)[0])
        dm = float(compute_volume_d_distance_ratio((p - step).reshape(1, 3), skin, medial)[0])
        grad[axis] = (dp - dm) / (2.0 * float(eps))
    if not np.isfinite(d0):
        return np.zeros(3, dtype=np.float32)
    return grad.astype(np.float32)


def _angular_distance_scalar(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi))


def _padded_skin_aabb(skin_vertices: np.ndarray, *, pad_frac: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    skin = np.asarray(skin_vertices, dtype=np.float64).reshape(-1, 3)
    mn = skin.min(axis=0)
    mx = skin.max(axis=0)
    span = np.maximum(mx - mn, 1.0e-4)
    pad = span * float(pad_frac)
    return (mn - pad).astype(np.float64), (mx + pad).astype(np.float64)


def _point_inside_aabb(point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def trace_streamline_core_to_skin(
    fields: LegHarmonicFields,
    atlas: object,
    *,
    theta0: float,
    h_axis: float,
    n_steps: int = 48,
    step_size: float = 0.006,
) -> np.ndarray:
    """Trace a fixed-(theta, h) streamline from shrink medial core (d~1) toward skin (d~0)."""
    from .atlas import _axis_point_and_tangent, _piecewise_station

    _, tangent = _axis_point_and_tangent(atlas.hip, atlas.knee, atlas.ankle, float(h_axis))
    tangent = np.asarray(tangent, dtype=np.float64).reshape(3)

    if hasattr(atlas, "core_h") and hasattr(atlas, "core_points") and np.asarray(atlas.core_points).size:
        core = medial_point_at_station(np.asarray(atlas.core_h, dtype=np.float32), np.asarray(atlas.core_points, dtype=np.float32), float(h_axis))
    else:
        core, _tangent = _axis_point_and_tangent(atlas.hip, atlas.knee, atlas.ankle, float(h_axis))
    core = np.asarray(core, dtype=np.float64).reshape(3)
    skin = np.asarray(atlas.skin_vertices, dtype=np.float32)
    lo, hi = _padded_skin_aabb(skin, pad_frac=0.10)
    skin_station, _ = _piecewise_station(skin, atlas.hip, atlas.knee, atlas.ankle)
    skin_theta = np.mod(np.asarray(atlas.skin_theta, dtype=np.float32), 2.0 * np.pi)
    delta = (skin_theta - float(theta0) + np.pi) % (2.0 * np.pi) - np.pi
    score = np.abs(delta) + 2.5 * np.abs(skin_station - float(h_axis))
    anchor = skin[int(np.argmin(score))].astype(np.float64)

    interps = _volume_interpolators(fields)
    pos = core.copy()
    path: list[np.ndarray] = [pos.copy()]
    prev_d = 1.0
    stall = 0
    for _ in range(max(8, int(n_steps))):
        if not _point_inside_aabb(pos, lo, hi):
            break
        _h, _t, d = interpolate_volume_field(fields, pos.reshape(1, 3), _interps=interps)
        d_val = float(d[0])
        if d_val <= 0.03 or float(np.linalg.norm(pos - anchor)) <= 0.014:
            path.append(anchor.copy())
            break
        if d_val > prev_d + 0.02:
            break
        if abs(d_val - prev_d) < 1.0e-4:
            stall += 1
            if stall >= 5:
                break
        else:
            stall = 0
        prev_d = d_val
        grad = grad_d_at(fields, pos).astype(np.float64)
        grad = grad - float(grad @ tangent) * tangent
        gn = float(np.linalg.norm(grad))
        if gn <= 1.0e-10:
            radial = anchor - core
            radial = radial - float(radial @ tangent) * tangent
            gn = float(np.linalg.norm(radial))
            if gn <= 1.0e-10:
                break
            direction = radial / gn
        else:
            direction = -grad / gn
        pos = pos + float(step_size) * direction
        if not _point_inside_aabb(pos, lo, hi):
            break
        path.append(pos.copy())
    return np.stack(path, axis=0).astype(np.float32)


def build_cross_section_d_grid(
    fields: LegHarmonicFields,
    atlas: object,
    *,
    h_axis: float,
    core: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    radius: float,
    grid_size: int = 96,
    boundary_uv: np.ndarray | None = None,
    section_segments: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample slice d on a cross-section plane (analytic 2D distance ratio inside skin polygon)."""
    del fields, h_axis
    if boundary_uv is None:
        if section_segments is None or np.asarray(section_segments).size == 0:
            r = max(float(radius), 1.0e-3)
            axis_vals = np.linspace(-r, r, int(grid_size), dtype=np.float64)
            gu, gv = np.meshgrid(axis_vals, axis_vals, indexing="xy")
            return gu.astype(np.float32), gv.astype(np.float32), np.full(gu.shape, np.nan, dtype=np.float32)
        boundary_uv = boundary_uv_from_section_segments(section_segments, core, e1, e2)
    return build_cross_section_d_grid_slice(boundary_uv, radius=float(radius), grid_size=int(grid_size))
