"""Serpentine coverage with positions and normals fitted to the measured top."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from peirastic.DEMO.phathom_scanning.detect import (
    _project_xy,
    tool_rpy_into_surface,
)
from peirastic.DEMO.phathom_scanning.surface import LocalSurface
from peirastic.DEMO.phathom_scanning.orientation import rotation_minimizing_frames, frame_diagnostics

PROBE_WIDTH_M = 0.05
OVERLAP = 0.20
EDGE_EXTRA_M = 0.010
STANDOFF_M = 0.040
LIFT_M = 0.010
SAMPLE_DS_M = 0.008
MIN_ROWS = 3
MAX_STRIDE_M = 0.040
MIN_ROW_SPAN_M = 0.020
_GEOM_EPS_M = 1.0e-9


def _unit(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    return a / max(n, 1e-12)


def plane_axes(normal: np.ndarray, toward_robot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n = surface outward, right = +u, far = +v (away from the mount)."""

    n = _unit(normal)
    toward = _project_xy(toward_robot, n)
    left = _project_xy(np.cross(toward, n), n)
    right = -left
    far = -toward
    return n, right, far


def _uv(pts: np.ndarray, origin: np.ndarray, right: np.ndarray, far: np.ndarray) -> np.ndarray:
    rel = np.asarray(pts, dtype=np.float64).reshape(-1, 3) - origin
    return np.column_stack([rel @ right, rel @ far])


def surface_edge_axes(cloud, normal, toward_robot):
    """Use the measured outline's minimum-area rectangle, not a camera axis.

    u follows the long edge. v crosses the short edge from the near side to
    the far side. The robot direction only resolves signs (and square ties).
    Unlike PCA, this does not weight a densely sampled half of the surface
    more heavily than a sparsely sampled half.
    """
    n, robot_right, robot_far = plane_axes(normal, toward_robot)
    origin = np.median(cloud, axis=0)
    uv = _uv(cloud, origin, robot_right, robot_far)
    try:
        border = uv[ConvexHull(uv).vertices]
    except QhullError as exc:
        raise ValueError("phantom top has no rectangular outline") from exc
    best = None
    for delta in np.roll(border, -1, axis=0) - border:
        length = np.linalg.norm(delta)
        if length < 1e-9:
            continue
        a = delta / length
        b = np.array([-a[1], a[0]])
        spans = np.ptp(np.column_stack([uv @ a, uv @ b]), axis=0)
        area = float(np.prod(spans))
        if best is None or area < best[0] - 1e-12:
            best = (area, a, b, spans)
    if best is None:
        raise ValueError("phantom outline is degenerate")
    _, a, b, spans = best
    if max(spans) <= 1.02 * min(spans):
        # A square has no identifiable long edge: retain a stable lateral cue.
        axis = a if abs(a[0]) >= abs(b[0]) else b
    else:
        axis = a if spans[0] >= spans[1] else b
    along = axis[0] * robot_right + axis[1] * robot_far
    across = np.cross(n, along)
    near_sign = float(np.dot(across, robot_far))
    if near_sign < -1e-10 or (abs(near_sign) <= 1e-10 and np.dot(along, robot_far) < 0):
        along, across = -along, -across
    aligned = _uv(cloud, origin, along, across)
    lo, hi = aligned.min(axis=0), aligned.max(axis=0)
    corners_uv = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    corners = origin + corners_uv[:, :1] * along + corners_uv[:, 1:] * across
    return n, along, across, (hi - lo), corners


def _dedupe(poses: np.ndarray, *, min_ds: float = 0.003) -> np.ndarray:
    keep = [poses[0]]
    for row in poses[1:]:
        if float(np.linalg.norm(row[:3] - keep[-1][:3])) >= min_ds:
            keep.append(row)
    if not np.array_equal(keep[-1], poses[-1]):
        if len(keep) > 1:
            keep[-1] = poses[-1]
        else:
            keep.append(poses[-1])
    return np.asarray(keep, dtype=np.float64)


def _convex_hull_equations(uv: np.ndarray) -> np.ndarray:
    """Return normalized ``a*u + b*v + c <= 0`` hull equations."""

    points = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3:
        raise ValueError("phantom top needs at least three non-collinear points")
    try:
        hull = ConvexHull(points)
    except QhullError as exc:
        raise ValueError("phantom top has no 2-D convex support") from exc
    equations = np.asarray(hull.equations, dtype=np.float64)
    if equations.shape[0] < 3 or not np.isfinite(equations).all():
        raise ValueError("phantom top convex support is invalid")
    return equations


def _eroded_v_limits(
    equations: np.ndarray,
    *,
    half_u: float,
    half_v: float,
) -> tuple[float, float]:
    """Get the v interval after an axis-aligned probe footprint inset.

    ``ConvexHull.equations`` describes the original support as
    ``a*u + b*v + c <= 0``.  A rectangle centered at ``(u, v)`` remains in
    that support when the right hand side is reduced by
    ``half_u*abs(a) + half_v*abs(b)``.  This is the exact erosion for a convex
    polygon and avoids replacing a rotated rectangle by its robot-facing box.
    """

    A, rhs = _eroded_halfplanes(
        equations,
        half_u=half_u,
        half_v=half_v,
    )
    # The extrema of v occur at vertices of the *intersection* of all
    # shifted half-planes.  Evaluating each edge at u=0 is only valid for an
    # axis-aligned rectangle and clips rotated plates unnecessarily.
    # This is only a 2-D polygon: intersect its edges directly, avoiding a
    # second solver/thread pool alongside the live robot controller.
    vertices = []
    for i in range(len(A)):
        for j in range(i):
            pair = A[[i, j]]
            if abs(float(np.linalg.det(pair))) < 1e-12:
                continue
            point = np.linalg.solve(pair, rhs[[i, j]])
            if np.all(A @ point <= rhs + 1e-9):
                vertices.append(point)
    if not vertices:
        raise ValueError("phantom top is too small for the probe inset")
    vs = np.asarray(vertices)[:, 1]
    lo, hi = float(vs.min()), float(vs.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        raise ValueError("phantom top is too small for the probe inset")
    return lo, hi


def _eroded_halfplanes(
    equations: np.ndarray,
    *,
    half_u: float,
    half_v: float,
) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(equations[:, :2], dtype=np.float64)
    rhs = (
        -np.asarray(equations[:, 2], dtype=np.float64)
        - half_u * np.abs(A[:, 0])
        - half_v * np.abs(A[:, 1])
    )
    return A, rhs


def _cross_section(
    equations: np.ndarray,
    v: float,
    *,
    half_u: float,
    half_v: float,
) -> tuple[float, float] | None:
    """Return the supported u interval at ``v`` after probe erosion."""

    A, rhs = _eroded_halfplanes(
        equations,
        half_u=half_u,
        half_v=half_v,
    )
    lo = -np.inf
    hi = np.inf
    for (a, b), limit in zip(A, rhs):
        if abs(float(a)) <= _GEOM_EPS_M:
            if float(b) * float(v) > float(limit) + 1.0e-8:
                return None
            continue
        bound = (float(limit) - float(b) * float(v)) / float(a)
        if a > 0.0:
            hi = min(hi, bound)
        else:
            lo = max(lo, bound)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo - 1.0e-8:
        return None
    return float(lo), float(hi)


def _sample_segment(lo: float, hi: float, ds: float) -> np.ndarray:
    """Sample a closed interval without stepping past its endpoint."""

    span = max(0.0, float(hi) - float(lo))
    if span <= 1.0e-12:
        return np.array([0.5 * (float(lo) + float(hi))], dtype=np.float64)
    n = max(1, int(np.ceil(span / max(float(ds), 1.0e-6))))
    # linspace includes the exact endpoint, unlike arange(lo, hi + ds, ds).
    return np.linspace(float(lo), float(hi), n + 1, dtype=np.float64)


def _max_area_scan_rectangle(equations: np.ndarray) -> tuple[float, float, float, float]:
    """Largest axis-aligned rectangle inside the already inset convex top.

    Every long scan row uses the same endpoints. Independent cross-sections
    at the hull's short-edge extremes produce tiny first/last rows and long
    diagonal connectors, even on an almost rectangular measured surface.
    Maximize log(half-width) + log(half-height) subject to linear support
    inequalities; this is a convex optimization with only four variables.
    Scaling to a unit-sized polygon keeps SLSQP tolerances meaningful in m.
    """
    a, bound = _eroded_halfplanes(equations, half_u=0.0, half_v=0.0)
    v0, v1 = _eroded_v_limits(equations, half_u=0.0, half_v=0.0)
    v_center = 0.5 * (v0 + v1)
    section = _cross_section(equations, v_center, half_u=0.0, half_v=0.0)
    if section is None or section[1] - section[0] <= _GEOM_EPS_M:
        raise ValueError("phantom top has no scan rectangle after probe inset")
    center = np.array([0.5 * sum(section), v_center])
    scale = max(section[1] - section[0], v1 - v0)
    support = np.column_stack([a, np.abs(a)])
    rhs = (bound - a @ center) / scale
    radius = 0.25 * float(np.min(rhs / np.abs(a).sum(axis=1)))
    if radius <= 0:
        raise ValueError("phantom top has no scan rectangle after probe inset")

    def objective(x):
        return -np.log(x[2]) - np.log(x[3])

    def gradient(x):
        return np.array([0.0, 0.0, -1.0 / x[2], -1.0 / x[3]])

    result = minimize(
        objective, np.array([0.0, 0.0, radius, radius]), jac=gradient,
        method="SLSQP", bounds=[(None, None), (None, None), (1e-8, None), (1e-8, None)],
        constraints={"type": "ineq", "fun": lambda x: rhs - support @ x,
                     "jac": lambda x: -support},
        options={"ftol": 1e-11, "maxiter": 100},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"could not fit rectangular scan area: {result.message}")
    if float(np.max(support @ result.x - rhs)) > 1e-8:
        raise ValueError("rectangular scan area exceeds measured probe support")
    midpoint = center + scale * result.x[:2]
    # Move slightly inside the optimizer's active boundaries.
    half = scale * result.x[2:] - 1e-7
    if half[0] * 2 < MIN_ROW_SPAN_M or half[1] <= 0:
        raise ValueError("phantom top has no useful rectangle after probe inset")
    return midpoint[0] - half[0], midpoint[0] + half[0], midpoint[1] - half[1], midpoint[1] + half[1]


def _rectangle_corners(bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Return the four corners of a UV rectangle in counter-clockwise order."""

    u0, u1, v0, v1 = (float(value) for value in bounds)
    return np.asarray(
        [[u0, v0], [u1, v0], [u1, v1], [u0, v1]],
        dtype=np.float64,
    )


def _fixed_rectangle(
    equations: np.ndarray,
    center_uv: np.ndarray,
    dimensions: np.ndarray,
) -> tuple[float, float, float, float]:
    """Fit the requested centered rectangle without changing its dimensions.

    ``equations`` already includes the per-edge probe footprint and edge
    margin erosion.  Checking all four corners is sufficient because the
    measured support is convex.  A failure is intentionally explicit: a
    requested scan must never be silently reduced or moved to make it fit.
    """

    center = np.asarray(center_uv, dtype=np.float64).reshape(-1)
    dims = np.asarray(dimensions, dtype=np.float64).reshape(-1)
    if center.shape != (2,) or dims.shape != (2,):
        raise ValueError("fixed scan rectangle requires two-dimensional center and dimensions")
    half = 0.5 * dims
    bounds = (
        float(center[0] - half[0]),
        float(center[0] + half[0]),
        float(center[1] - half[1]),
        float(center[1] + half[1]),
    )
    corners = _rectangle_corners(bounds)
    residual = np.asarray(equations[:, :2]) @ corners.T + np.asarray(equations[:, 2])[:, None]
    worst = float(np.max(residual))
    if not np.isfinite(worst) or worst > 2.0e-8:
        raise ValueError(
            "requested scan dimensions cannot fit the measured phantom after "
            f"probe footprint and edge margin (support residual={worst:.3g} m)"
        )
    return bounds


def _arc_resample(points: np.ndarray, step_m: float) -> np.ndarray:
    """Resample a dense polyline at uniform arc-length spacing."""

    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Lissajous path needs at least two finite samples")
    step = float(step_m)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Lissajous arc-length step must be positive")
    segment = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 1.0e-12:
        return values[[0, -1]].copy()
    targets = np.arange(0.0, total, step, dtype=np.float64)
    if targets.size == 0 or total - float(targets[-1]) > 1.0e-12:
        targets = np.concatenate((targets, [total]))
    else:
        targets[-1] = total
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(values) - 2)
    span = cumulative[indices + 1] - cumulative[indices]
    fraction = np.divide(
        targets - cumulative[indices],
        span,
        out=np.zeros_like(targets),
        where=span > 1.0e-12,
    )
    sampled = values[indices] + fraction[:, None] * (values[indices + 1] - values[indices])
    sampled[0] = values[0]
    sampled[-1] = values[-1]
    return sampled


def _lissajous_queries(
    bounds: tuple[float, float, float, float],
    step_m: float,
) -> np.ndarray:
    """Generate a closed, double-lobed Lissajous path in the scan rectangle."""

    u0, u1, v0, v1 = (float(value) for value in bounds)
    length = u1 - u0
    width = v1 - v0
    if length <= 0.0 or width <= 0.0:
        raise ValueError("Lissajous scan rectangle must have positive dimensions")
    center = np.array([0.5 * (u0 + u1), 0.5 * (v0 + v1)], dtype=np.float64)
    half_length = 0.5 * length
    half_width = 0.5 * width
    # The dense parameterization makes the later arc-length interpolation
    # independent of the aspect ratio and keeps the final projected spacing
    # below the requested waypoint spacing on gently curved surfaces.
    estimated_length = 2.0 * np.pi * np.hypot(half_length, half_width * 2.0)
    dense_step = max(float(step_m) * 0.1, 1.0e-5)
    n_dense = max(2048, int(np.ceil(estimated_length / dense_step)))
    theta = np.linspace(-0.5 * np.pi, 1.5 * np.pi, n_dense + 1, dtype=np.float64)
    dense = np.column_stack(
        (
            center[0] + half_length * np.sin(theta),
            center[1] + half_width * np.sin(2.0 * theta),
        )
    )
    # A half-step target gives margin for the small height change introduced
    # when UV waypoints are projected onto a curved measured surface.
    return _arc_resample(dense, max(float(step_m) * 0.5, 1.0e-5))


@dataclass
class PhantomSPlan:
    poses: np.ndarray
    standoff: np.ndarray
    lift: np.ndarray
    normal: np.ndarray
    right: np.ndarray
    far: np.ndarray
    n_rows: int
    stride_m: float
    length_m: float
    u_span_m: float
    v_span_m: float
    normals: np.ndarray | None = None
    surface_diagnostics: dict | None = None
    orientation_diagnostics: dict | None = None
    outline_dimensions_m: np.ndarray | None = None
    outline_corners: np.ndarray | None = None
    pattern: str = "raster"
    scan_center: np.ndarray | None = None
    requested_dimensions: np.ndarray | None = None


def plan_s_scan(
    top_pts: np.ndarray,
    normal: np.ndarray,
    toward_robot: np.ndarray,
    *,
    probe_width_m: float = PROBE_WIDTH_M,
    overlap: float = OVERLAP,
    edge_extra_m: float = EDGE_EXTRA_M,
    standoff_m: float = STANDOFF_M,
    lift_m: float = LIFT_M,
    sample_ds_m: float = SAMPLE_DS_M,
    yaw_axis: np.ndarray | None = None,
    initial_rpy: np.ndarray | None = None,
    fit_radius_m: float = 0.030,
    scan_length_m: float | None = None,
    scan_width_m: float | None = None,
    pattern: str = "raster",
) -> PhantomSPlan:
    cloud = np.asarray(top_pts, dtype=np.float64).reshape(-1, 3)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if len(cloud) < 8:
        raise ValueError(f"phantom top too small ({len(cloud)})")
    if (scan_length_m is None) != (scan_width_m is None):
        raise ValueError("scan_length_m and scan_width_m must be provided together")
    pattern_name = str(pattern).strip().lower()
    if pattern_name not in {"raster", "lissajous"}:
        raise ValueError("pattern must be 'raster' or 'lissajous'")
    requested_dimensions = None
    if scan_length_m is not None:
        requested_dimensions = np.asarray(
            [scan_length_m, scan_width_m], dtype=np.float64
        )
        if (
            requested_dimensions.shape != (2,)
            or not np.isfinite(requested_dimensions).all()
            or np.any(requested_dimensions <= 0.0)
        ):
            raise ValueError("scan_length_m and scan_width_m must be finite and positive")
    width, edge, overlap_f = float(probe_width_m), float(edge_extra_m), float(overlap)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("probe_width_m must be positive")
    if not np.isfinite(edge) or edge < 0:
        raise ValueError("edge_extra_m must be non-negative")
    if not np.isfinite(overlap_f) or not 0 <= overlap_f < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if not np.isfinite(sample_ds_m) or sample_ds_m <= 0:
        raise ValueError("sample_ds_m must be positive")
    n, along, across, dimensions, corners = surface_edge_axes(cloud, normal, toward_robot)
    origin = np.median(cloud, axis=0)
    uv = _uv(cloud, origin, along, across)
    hull = _convex_hull_equations(uv)
    # ``surface_edge_axes`` deliberately returns the measured outline corners
    # rather than a density-weighted cloud center.  Use their geometric mean
    # for every explicitly requested pattern area, so raster and Lissajous
    # plans remain coincident after a rigid transform of the phantom.
    scan_center = np.mean(np.asarray(corners, dtype=np.float64), axis=0)
    center_uv = _uv(scan_center[None], origin, along, across)[0]
    surface = LocalSurface(cloud, origin, along, across, n, fit_radius_m=fit_radius_m)
    if initial_rpy is None:
        hint = -along if yaw_axis is None else np.asarray(yaw_axis, dtype=float)
        initial_rpy = tool_rpy_into_surface(n, hint)
    initial_rpy = np.asarray(initial_rpy, dtype=float).reshape(3)
    if not np.isfinite(initial_rpy).all():
        raise ValueError("initial_rpy must be finite")
    initial_rotation = Rotation.from_euler("xyz", initial_rpy).as_matrix()
    ds = max(0.004, float(sample_ds_m))
    stride = min(width * (1 - overlap_f), MAX_STRIDE_M)

    # Keep the probe's real X/Y directions independent of the scan long edge.
    # The 50-mm square is an explicit footprint approximation. Its projection
    # on each hull normal must fit even when the probe is yawed relative to the
    # object, and during the rotation interpolation between waypoints.
    edge_world = hull[:, :1] * along + hull[:, 1:2] * across

    def footprint(rotations):
        dots = np.einsum("hj,pjk->hpk", edge_world, rotations[:, :, :2])
        support = 0.5 * width * np.max(np.abs(dots).sum(axis=2), axis=1)
        interpolation_guard = 0.0
        if len(rotations) > 1:
            r = Rotation.from_matrix(rotations)
            angle = float(np.max((r[:-1].inv() * r[1:]).magnitude()))
            # Bound the sin/cos interpolation bulge between endpoint supports.
            interpolation_guard = width * angle * angle / 8.0
        return support + edge + interpolation_guard

    def build(clearance):
        equations = hull.copy()
        equations[:, 2] += clearance
        if requested_dimensions is None:
            bounds = _max_area_scan_rectangle(equations)
            region_method = "maximum_area_aligned_inscribed_rectangle"
        else:
            try:
                bounds = _fixed_rectangle(equations, center_uv, requested_dimensions)
            except ValueError as exc:
                raise ValueError(
                    f"requested scan dimensions {requested_dimensions.tolist()} m "
                    "are unavailable in the measured support; dimensions were not reduced or shifted; "
                    f"{exc}"
                ) from exc
            region_method = "fixed_centered_rectangle"
        u0, u1, v0, v1 = bounds
        if pattern_name == "lissajous":
            queries = _lissajous_queries(bounds, ds)
            # Lissajous paths have no row spacing.  The zero values retain the
            # legacy fields while making the pattern unambiguous to callers.
            n_rows = 0
            stride_out = 0.0
            sections = []
        else:
            # Translation into the measured frame can perturb an exact 80/40
            # mm ratio above 2 by a few ulps. Keep row count repeatable.
            n_rows = max(MIN_ROWS, int(np.ceil((v1 - v0) / stride - 1.0e-12)) + 1)
            vs = np.linspace(v0, v1, n_rows)
            sections = [(u0, u1)] * n_rows
            queries = []
            for i, (v, section) in enumerate(zip(vs, sections)):
                lo, hi = section
                us = _sample_segment(lo, hi, ds)
                if i % 2:
                    us = us[::-1]
                queries.extend((float(u), float(v)) for u in us)
                if i + 1 < n_rows:
                    side = 1 if i % 2 == 0 else 0
                    u_end, u_next = section[side], sections[i + 1][side]
                    v_next = vs[i + 1]
                    distance = float(np.hypot(u_next - u_end, v_next - v))
                    count = max(1, int(np.ceil(distance / ds)))
                    queries.extend(
                        (float(u_end + f * (u_next - u_end)), float(v + f * (v_next - v)))
                        for f in np.linspace(0, 1, count + 1)[1:]
                    )
            stride_out = float(vs[1] - vs[0])
        samples = []
        for query in queries:
            point, local_normal = surface.project(np.asarray(query))
            samples.append(np.concatenate([point, local_normal]))
        samples = _dedupe(np.asarray(samples))
        normals = samples[:, 3:6]
        rotations = rotation_minimizing_frames(normals, initial_rotation)
        rpy = np.unwrap(Rotation.from_matrix(rotations).as_euler("xyz"), axis=0)
        poses = np.column_stack([samples[:, :3], rpy])
        metadata = dict(
            n_rows=n_rows,
            stride_m=stride_out,
            u_span_m=float(u1 - u0),
            v_span_m=float(v1 - v0),
            region_method=region_method,
        )
        return poses, normals, rotations, metadata

    # Surface tilt changes the footprint slightly. Rebuild the inset using a
    # monotonically increasing support bound; do not silently rotate the tool
    # to make it fit. Typical near-flat surfaces settle after one refinement.
    clearance = footprint(initial_rotation[None])
    for _ in range(6):
        poses, normals, rotations, metadata = build(clearance)
        required = footprint(rotations)
        if np.all(required <= clearance + 1e-9):
            break
        clearance = np.maximum(clearance, required) + 1e-5
    else:
        raise ValueError("probe footprint did not settle on the curved surface")
    standoff = poses[0].copy()
    standoff[:3] += float(standoff_m) * normals[0]
    lift = poses[-1].copy()
    lift[:3] += float(lift_m) * normals[-1]
    diagnostics = surface.diagnostics
    diagnostics["axis_method"] = "minimum_area_rectangle_long_edge"
    diagnostics["scan_region_method"] = metadata.pop("region_method")
    diagnostics["footprint_model"] = "tool_xy_square_projected_on_surface_outline"
    diagnostics["footprint_support_m"] = clearance.tolist()
    diagnostics["pattern"] = pattern_name
    if requested_dimensions is not None:
        diagnostics["requested_dimensions_m"] = requested_dimensions.tolist()
    return PhantomSPlan(
        poses=poses, standoff=standoff, lift=lift, normal=n, right=along, far=across,
        length_m=float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()),
        normals=normals, surface_diagnostics=diagnostics,
        orientation_diagnostics=frame_diagnostics(rotations, initial_rotation),
        outline_dimensions_m=dimensions,
        outline_corners=corners,
        pattern=pattern_name,
        scan_center=scan_center,
        requested_dimensions=(
            None if requested_dimensions is None else requested_dimensions.copy()
        ),
        **metadata,
    )
