"""Local measured-surface projection for phantom scan waypoints.

The scanner works in a local surface frame.  ``right`` and ``far`` define
the two in-plane coordinates and ``normal`` is the outward reference normal.
The point cloud is only used as measured support: a query must be inside a
Delaunay triangle.  Large Delaunay triangles are rejected against robust
global and local two-dimensional spacing statistics, with a hard 30 mm cap,
so an empty patch cannot silently become an interpolated surface.

This module is deliberately offline and geometry-only.  It does not read a
camera, create a worker pool, or know anything about robot orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree


# The absolute cap is intentional.  It prevents a very sparse cloud from
# allowing a several-centimetre hole to be filled merely because its global
# median spacing was also large.  Normal Orbbec phantom samples are around
# 3--4 mm, while the cap still permits a sparse but enclosing local sample.
MAX_TRIANGLE_EDGE_M = 0.030
MIN_TRIANGLE_EDGE_M = 0.008
MAX_FIT_RADIUS_M = 0.050
MIN_FIT_POINTS = 6
_EPS = 1.0e-12


def _vector(value: Any, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.size != 3 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    length = float(np.linalg.norm(out))
    if length <= _EPS:
        raise ValueError(f"{name} must be non-zero")
    return out / length


def _point(value: Any, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.size != 3 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return out


def _local_frame(
    right: Any,
    far: Any,
    normal: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make an orthonormal copy without changing either in-plane direction."""

    n = _vector(normal, "normal")
    r0 = np.asarray(right, dtype=np.float64).reshape(-1)
    f0 = np.asarray(far, dtype=np.float64).reshape(-1)
    if r0.size != 3 or not np.isfinite(r0).all():
        raise ValueError("right must be a finite 3-vector")
    if f0.size != 3 or not np.isfinite(f0).all():
        raise ValueError("far must be a finite 3-vector")

    # Small calibration non-orthogonality should not contaminate the height
    # fit.  Preserve the supplied right/far directions as much as possible.
    r = r0 - n * float(np.dot(r0, n))
    r_norm = float(np.linalg.norm(r))
    if r_norm <= _EPS:
        raise ValueError("right must span the local surface plane")
    r /= r_norm
    f = f0 - n * float(np.dot(f0, n)) - r * float(np.dot(f0, r))
    f_norm = float(np.linalg.norm(f))
    if f_norm <= _EPS:
        raise ValueError("far must span the local surface plane")
    f /= f_norm
    # Keep the handedness supplied by the caller.  A left-handed (right, far,
    # normal) basis is a valid coordinate convention; changing far here would
    # mirror every requested v coordinate.  ``project`` fixes only the sign
    # of the returned geometric normal after taking the tangent cross product.
    return r, f, n


def _finite_cloud(cloud: Any) -> np.ndarray:
    pts = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < MIN_FIT_POINTS:
        raise ValueError(
            f"point-cloud support unavailable: need at least {MIN_FIT_POINTS} finite points"
        )
    return pts


def _deduplicate_uv(uv: np.ndarray, height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse exactly repeated projected samples before Qhull and fitting."""

    unique_uv, inverse = np.unique(np.asarray(uv, dtype=np.float64), axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(height, dtype=np.float64))
    counts = np.bincount(inverse)
    return unique_uv, sums / np.maximum(counts, 1)


def _typical_spacing(tree: cKDTree, uv: np.ndarray) -> float:
    if uv.shape[0] < 2:
        return float("nan")
    distances, _ = tree.query(uv, k=2)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    nearest = nearest[np.isfinite(nearest) & (nearest > 1.0e-9)]
    if nearest.size == 0:
        return float("nan")
    # Median is stable when the cloud contains a few edge gaps or isolated
    # returns.  The triangle cap below remains the final sparsity guard.
    return float(np.median(nearest))


def _triangle_edges(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    tri = points[np.asarray(simplices, dtype=np.int64)]
    e01 = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
    e12 = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
    e20 = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
    return np.column_stack((e01, e12, e20))


def _robust_upper_limit(values: np.ndarray, *, floor: float) -> tuple[float, float]:
    """Return a robust high quantile and a data-derived upper allowance."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return float(floor), float(floor)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    q99 = float(np.percentile(finite, 99.0))
    # The allowance follows the observed local spread.  It is deliberately
    # not a second fixed multiplier of the global nearest-neighbour median.
    return q99, max(float(floor), q99 + max(3.0 * mad, 0.0))


@dataclass(frozen=True)
class _FitResult:
    height_m: float
    grad_u: float
    grad_v: float
    residual_rms_m: float
    condition: float
    spread_ratio: float
    point_count: int


class LocalSurface:
    """Project local ``(right, far)`` coordinates onto measured cloud support.

    ``origin`` is the reference point for the local frame.  A cloud point is
    represented as ``origin + u*right + v*far + h*normal``.  ``project``
    accepts one ``(u, v)`` pair in metres and returns the fitted world point
    and an outward unit normal.  The fitted normal follows the local surface
    gradient, rather than assuming world vertical.
    """

    def __init__(
        self,
        cloud: Any,
        origin: Any,
        right: Any,
        far: Any,
        normal: Any,
        *,
        fit_radius_m: float = 0.03,
    ) -> None:
        self.origin = _point(origin, "origin")
        self.right, self.far, self.normal = _local_frame(right, far, normal)
        radius = float(fit_radius_m)
        if not np.isfinite(radius) or radius <= 0.0 or radius > MAX_FIT_RADIUS_M:
            raise ValueError(
                f"point-cloud support unavailable: fit_radius_m must be in (0, {MAX_FIT_RADIUS_M:.3f}] m"
            )
        self.fit_radius_m = radius

        points = _finite_cloud(cloud)
        rel = points - self.origin
        uv_all = np.column_stack((rel @ self.right, rel @ self.far))
        height_all = rel @ self.normal
        uv, height = _deduplicate_uv(uv_all, height_all)
        if uv.shape[0] < 3:
            raise ValueError("point-cloud support unavailable: fewer than three distinct in-plane samples")

        self._uv = uv
        self._height = height
        self._tree = cKDTree(uv)
        self._sampling_spacing_m = _typical_spacing(self._tree, uv)
        if not np.isfinite(self._sampling_spacing_m):
            raise ValueError("point-cloud support unavailable: sampling spacing is undefined")

        # A factor of three is permissive for diagonal edges in sparse regular
        # sampling, but the absolute 30 mm cap prevents an 80 mm hole from
        # being bridged.  The 8 mm floor avoids rejecting mildly irregular
        # dense returns because of one unusually short nearest-neighbour pair.
        self._max_triangle_edge_m = min(
            MAX_TRIANGLE_EDGE_M,
            max(MIN_TRIANGLE_EDGE_M, 3.0 * self._sampling_spacing_m),
        )

        try:
            self._triangulation = Delaunay(uv)
        except QhullError as exc:
            raise ValueError("point-cloud support unavailable: no 2-D triangulation") from exc
        simplices = np.asarray(self._triangulation.simplices, dtype=np.int64)
        edge_lengths = _triangle_edges(uv, simplices)
        areas = 0.5 * np.abs(
            (uv[simplices[:, 1], 0] - uv[simplices[:, 0], 0])
            * (uv[simplices[:, 2], 1] - uv[simplices[:, 0], 1])
            - (uv[simplices[:, 1], 1] - uv[simplices[:, 0], 1])
            * (uv[simplices[:, 2], 0] - uv[simplices[:, 0], 0])
        )
        min_area = max(1.0e-14, 1.0e-5 * self._sampling_spacing_m**2)
        self._triangle_max_edges = np.max(edge_lengths, axis=1)
        self._triangle_edge_lengths = edge_lengths
        self._valid_triangle = (
            self._triangle_max_edges <= self._max_triangle_edge_m + 1.0e-12
        ) & (areas >= min_area)

        self._diagnostics: dict[str, Any] = {
            "n_points_input": int(points.shape[0]),
            "n_points_unique_uv": int(uv.shape[0]),
            "sampling_spacing_m": float(self._sampling_spacing_m),
            "max_triangle_edge_m": float(self._max_triangle_edge_m),
            "max_triangle_edge_limit_m": float(MAX_TRIANGLE_EDGE_M),
            "fit_radius_m": float(radius),
            "n_triangles": int(simplices.shape[0]),
            "n_valid_triangles": int(np.count_nonzero(self._valid_triangle)),
            "valid_triangle_fraction": float(
                np.count_nonzero(self._valid_triangle) / max(1, simplices.shape[0])
            ),
            "last_query_uv": None,
            "last_triangle_max_edge_m": None,
            "last_triangle_support_mode": None,
            "last_local_sampling_spacing_m": None,
            "last_local_triangle_q99_m": None,
            "last_local_triangle_limit_m": None,
            "last_local_angular_bins": None,
            "last_support_count": None,
            "last_fit_rms_m": None,
            "last_fit_condition": None,
            "last_fit_spread_ratio": None,
        }

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-compatible geometry and most-recent query diagnostics."""

        # Every value is a Python scalar/list, but return a copy so callers
        # cannot mutate the internal state used for subsequent error reports.
        out = dict(self._diagnostics)
        if isinstance(out.get("last_query_uv"), list):
            out["last_query_uv"] = list(out["last_query_uv"])
        return out

    def _local_triangle_limit(
        self,
        query: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[float, float, float, int]:
        """Estimate a local, anisotropy-tolerant triangle span limit.

        The global median is a useful baseline, but it is too brittle when a
        single projected return is missing or when the sampling is elongated
        in one in-plane direction.  Delaunay edge lengths touching the local
        neighbourhood provide a two-dimensional statistic; one local nearest
        neighbour spacing is retained as a conservative uncertainty allowance.
        The caller still applies the hard 30 mm cap and fit-quality checks.
        """

        local_uv = self._uv[indices]
        if local_uv.shape[0] < 2:
            return (
                float(self._sampling_spacing_m),
                float("nan"),
                float(self._max_triangle_edge_m),
                0,
            )
        local_tree = cKDTree(local_uv)
        distances, _ = local_tree.query(local_uv, k=2)
        local_nn = np.asarray(distances[:, 1], dtype=np.float64)
        local_nn = local_nn[np.isfinite(local_nn) & (local_nn > 1.0e-9)]
        local_spacing = (
            float(np.median(local_nn)) if local_nn.size else float(self._sampling_spacing_m)
        )

        simplex_points = self._uv[self._triangulation.simplices]
        local_triangle = np.any(
            np.linalg.norm(simplex_points - query[None, None, :], axis=2)
            <= self.fit_radius_m + 1.0e-12,
            axis=1,
        )
        local_edges = self._triangle_edge_lengths[local_triangle].reshape(-1)
        q99, robust_limit = _robust_upper_limit(
            local_edges,
            floor=self._max_triangle_edge_m,
        )
        # Measure radial coverage in angular sectors as a second, independent
        # 2-D statistic.  This lets a single missing return be bridged when
        # nearby samples surround the query, while a one-sided cloud cannot
        # authorize an otherwise over-long triangle.
        delta = local_uv - query[None, :]
        radial = np.linalg.norm(delta, axis=1)
        sectors = np.floor(
            np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2.0 * np.pi)
            / (2.0 * np.pi / 12.0)
        ).astype(np.int64)
        sector_min = np.full(12, np.nan, dtype=np.float64)
        for sector in range(12):
            values = radial[sectors == sector]
            if values.size:
                sector_min[sector] = float(values.min())
        angular_bins = int(np.count_nonzero(np.isfinite(sector_min)))
        radial_gap = (
            float(np.max(sector_min)) if angular_bins == sector_min.size else 0.0
        )

        # For anisotropic sampling, a valid diagonal can be one local spacing
        # longer than the robust high quantile of individual Delaunay edges.
        # Surrounding angular coverage provides another data-derived allowance
        # for an isolated Delaunay edge caused by a dropped return.
        local_limit = min(
            MAX_TRIANGLE_EDGE_M,
            max(
                self._max_triangle_edge_m,
                robust_limit,
                q99 + local_spacing,
                radial_gap + local_spacing,
            ),
        )
        return local_spacing, q99, local_limit, angular_bins

    def _support_error(
        self,
        query: np.ndarray,
        reason: str,
        *,
        triangle_edge: float | None = None,
        support_count: int | None = None,
        fit_rms: float | None = None,
        fit_condition: float | None = None,
        support_mode: str | None = None,
        local_stats: tuple[float, float, float, int] | None = None,
    ) -> ValueError:
        self._diagnostics["last_query_uv"] = [float(query[0]), float(query[1])]
        self._diagnostics["last_triangle_max_edge_m"] = (
            None if triangle_edge is None else float(triangle_edge)
        )
        self._diagnostics["last_triangle_support_mode"] = support_mode
        if local_stats is None:
            self._diagnostics["last_local_sampling_spacing_m"] = None
            self._diagnostics["last_local_triangle_q99_m"] = None
            self._diagnostics["last_local_triangle_limit_m"] = None
            self._diagnostics["last_local_angular_bins"] = None
        else:
            self._diagnostics["last_local_sampling_spacing_m"] = float(local_stats[0])
            self._diagnostics["last_local_triangle_q99_m"] = float(local_stats[1])
            self._diagnostics["last_local_triangle_limit_m"] = float(local_stats[2])
            self._diagnostics["last_local_angular_bins"] = int(local_stats[3])
        self._diagnostics["last_support_count"] = (
            None if support_count is None else int(support_count)
        )
        self._diagnostics["last_fit_rms_m"] = None if fit_rms is None else float(fit_rms)
        self._diagnostics["last_fit_condition"] = (
            None if fit_condition is None else float(fit_condition)
        )
        return ValueError(
            "point-cloud support unavailable for "
            f"uv=({query[0]:+.4f},{query[1]:+.4f}) m: {reason}"
        )

    def _fit_local(self, query: np.ndarray, indices: np.ndarray) -> _FitResult:
        delta = self._uv[indices] - query
        radius = self.fit_radius_m
        scale = max(radius, self._sampling_spacing_m)
        x = delta[:, 0] / scale
        y = delta[:, 1] / scale
        design = np.column_stack((
            np.ones(indices.size, dtype=np.float64),
            x,
            y,
            x * x,
            x * y,
            y * y,
        ))
        height = self._height[indices]
        distance = np.linalg.norm(delta, axis=1)
        base_weight = np.exp(-0.5 * (distance / max(0.45 * radius, _EPS)) ** 2)
        base_weight = np.maximum(base_weight, 1.0e-4)

        weighted_design = design * np.sqrt(base_weight)[:, None]
        weighted_height = height * np.sqrt(base_weight)
        try:
            condition = float(np.linalg.cond(weighted_design))
            coef = np.linalg.lstsq(weighted_design, weighted_height, rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise ValueError("fit quality is numerically singular") from exc
        if not np.isfinite(coef).all() or not np.isfinite(condition):
            raise ValueError("fit quality is non-finite")

        # A short robust reweighting pass keeps isolated depth outliers from
        # changing the local quadratic while retaining the requested smooth
        # weighted fit for normal samples.
        for _ in range(2):
            residual = height - design @ coef
            center = float(np.median(residual))
            scale_res = max(1.4826 * float(np.median(np.abs(residual - center))), 1.0e-6)
            huber = np.minimum(1.0, 2.5 * scale_res / np.maximum(np.abs(residual), 1.0e-12))
            weights = base_weight * huber
            wdesign = design * np.sqrt(weights)[:, None]
            wheight = height * np.sqrt(weights)
            try:
                condition = float(np.linalg.cond(wdesign))
                coef = np.linalg.lstsq(wdesign, wheight, rcond=None)[0]
            except (np.linalg.LinAlgError, ValueError) as exc:
                raise ValueError("fit quality is numerically singular") from exc
            if not np.isfinite(coef).all() or not np.isfinite(condition):
                raise ValueError("fit quality is non-finite")

        delta_centered = delta - np.average(delta, axis=0, weights=base_weight)
        covariance = (
            delta_centered.T @ (delta_centered * base_weight[:, None])
        ) / max(float(base_weight.sum()), _EPS)
        eigenvalues = np.linalg.eigvalsh(covariance)
        if not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= _EPS:
            spread_ratio = 0.0
        else:
            spread_ratio = float(np.sqrt(max(0.0, eigenvalues[0]) / eigenvalues[-1]))
        residual = height - design @ coef
        residual_rms = float(
            np.sqrt(np.sum(base_weight * residual * residual) / max(float(base_weight.sum()), _EPS))
        )
        return _FitResult(
            height_m=float(coef[0]),
            grad_u=float(coef[1] / scale),
            grad_v=float(coef[2] / scale),
            residual_rms_m=residual_rms,
            condition=condition,
            spread_ratio=spread_ratio,
            point_count=int(indices.size),
        )

    def project(self, uv: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return the measured surface point and outward normal at ``uv``.

        The query is accepted only in a Delaunay triangle whose measured
        edges are supported by the global or locally confirmed spacing limit.
        Thus an interior missing patch and a query outside the convex hull
        both fail closed.
        """

        query = np.asarray(uv, dtype=np.float64).reshape(-1)
        if query.size != 2 or not np.isfinite(query).all():
            raise ValueError("point-cloud support unavailable for invalid uv location")
        query = query.astype(np.float64, copy=False)
        self._diagnostics["last_query_uv"] = [float(query[0]), float(query[1])]

        simplex_id = int(self._triangulation.find_simplex(query, tol=1.0e-10))
        if simplex_id < 0:
            raise self._support_error(query, "query lies outside the measured convex support")
        triangle_edge = float(self._triangle_max_edges[simplex_id])
        indices = np.asarray(self._tree.query_ball_point(query, self.fit_radius_m), dtype=np.int64)
        local_stats = None
        if indices.size >= 2:
            local_stats = self._local_triangle_limit(query, indices)
            self._diagnostics["last_local_sampling_spacing_m"] = float(local_stats[0])
            self._diagnostics["last_local_triangle_q99_m"] = float(local_stats[1])
            self._diagnostics["last_local_triangle_limit_m"] = float(local_stats[2])
            self._diagnostics["last_local_angular_bins"] = int(local_stats[3])
        if indices.size < MIN_FIT_POINTS:
            raise self._support_error(
                query,
                f"fit quality has only {indices.size} local points "
                f"within {self.fit_radius_m:.4f} m; need {MIN_FIT_POINTS} "
                f"(typical spacing={self._sampling_spacing_m:.4f} m)",
                triangle_edge=triangle_edge,
                support_count=int(indices.size),
                local_stats=local_stats,
            )
        try:
            fit = self._fit_local(query, indices)
        except ValueError as exc:
            raise self._support_error(
                query,
                f"fit quality failed ({exc})",
                triangle_edge=triangle_edge,
                support_count=int(indices.size),
                local_stats=local_stats,
            ) from exc

        self._diagnostics["last_triangle_max_edge_m"] = triangle_edge
        self._diagnostics["last_support_count"] = fit.point_count
        self._diagnostics["last_fit_rms_m"] = fit.residual_rms_m
        self._diagnostics["last_fit_condition"] = fit.condition
        self._diagnostics["last_fit_spread_ratio"] = fit.spread_ratio
        # Normalized coordinates make this threshold independent of metres vs
        # millimetres, while the explicit RMS threshold prevents a bad local
        # patch or mixed-depth returns from becoming a waypoint.
        if fit.condition > 1.0e6 or fit.spread_ratio < 0.08:
            raise self._support_error(
                query,
                f"fit quality is poorly conditioned "
                f"(condition={fit.condition:.3g}, spread_ratio={fit.spread_ratio:.3f})",
                triangle_edge=triangle_edge,
                support_count=fit.point_count,
                fit_rms=fit.residual_rms_m,
                fit_condition=fit.condition,
                local_stats=local_stats,
            )
        max_rms = max(0.004, 0.15 * self.fit_radius_m)
        if fit.residual_rms_m > max_rms:
            raise self._support_error(
                query,
                f"fit quality residual={fit.residual_rms_m:.4f} m exceeds "
                f"{max_rms:.4f} m",
                triangle_edge=triangle_edge,
                support_count=fit.point_count,
                fit_rms=fit.residual_rms_m,
                fit_condition=fit.condition,
                local_stats=local_stats,
            )

        # Ordinary triangles use the global spacing-derived limit.  A triangle
        # just beyond that baseline may still be a valid local interpolation
        # when the two-dimensional neighbourhood has a compatible edge scale
        # and the quadratic fit above is well conditioned.  The local limit is
        # data-derived and remains bounded by the hard 30 mm cap.
        if bool(self._valid_triangle[simplex_id]):
            support_mode = "global"
        else:
            local_limit = float(local_stats[2]) if local_stats is not None else float("nan")
            angular_bins = int(local_stats[3]) if local_stats is not None else 0
            if (
                not np.isfinite(local_limit)
                or angular_bins < 8
                or triangle_edge > local_limit + 1.0e-12
            ):
                local_q99 = float(local_stats[1]) if local_stats is not None else float("nan")
                raise self._support_error(
                    query,
                    "sampling gap is too large for interpolation "
                    f"(triangle span={triangle_edge:.4f} m, "
                    f"typical spacing={self._sampling_spacing_m:.4f} m, "
                    f"global limit={self._max_triangle_edge_m:.4f} m, "
                    f"local edge q99={local_q99:.4f} m, "
                    f"local limit={local_limit:.4f} m, "
                    f"angular bins={angular_bins}/12)",
                    triangle_edge=triangle_edge,
                    support_count=fit.point_count,
                    fit_rms=fit.residual_rms_m,
                    fit_condition=fit.condition,
                    support_mode="rejected",
                    local_stats=local_stats,
                )
            support_mode = "local"
        self._diagnostics["last_triangle_support_mode"] = support_mode

        point = self.origin + query[0] * self.right + query[1] * self.far + fit.height_m * self.normal
        tangent_u = self.right + fit.grad_u * self.normal
        tangent_v = self.far + fit.grad_v * self.normal
        outward = np.cross(tangent_u, tangent_v)
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm <= _EPS or not np.isfinite(outward).all():
            raise self._support_error(
                query,
                "fit quality produced an invalid surface normal",
                triangle_edge=triangle_edge,
                support_count=fit.point_count,
                fit_rms=fit.residual_rms_m,
                fit_condition=fit.condition,
                support_mode=support_mode,
                local_stats=local_stats,
            )
        outward /= outward_norm
        if float(np.dot(outward, self.normal)) < 0.0:
            outward = -outward
        return np.asarray(point, dtype=np.float64), np.asarray(outward, dtype=np.float64)


__all__ = ["LocalSurface", "MAX_TRIANGLE_EDGE_M", "MAX_FIT_RADIUS_M"]
