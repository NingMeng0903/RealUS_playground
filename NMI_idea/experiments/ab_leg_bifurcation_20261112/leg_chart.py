"""Bounded, geometry-only chart for the repaired right-leg tetra atlas.

The asset used here is the sealed shaft ROI atlas produced by the November
2026 repair.  This module intentionally has no simulator, tissue, probe, or
belief-model dependencies.  A chart point is a nominal ``(x, y, z)`` in mm:

* ``x`` is signed centerline arclength from the repaired-atlas knee
  projection (positive with increasing atlas ``h`` / distal direction),
* ``y`` is the projection of the visible anterior skin arc onto a fixed
  ipsilateral lateral axis (positive toward increasing world x), and
* ``z`` is nominal inward depth.  The ``z`` coordinate is converted to the
  real atlas ``d`` coordinate using the local skin-to-inner-shell distance.

The final world point is always obtained by barycentric interpolation of an
actual atlas tetrahedron in ``(theta, h, d)`` label space.  There is no
nearest-vertex snap and no outside extrapolation: a query for which a
containing labelled tetrahedron cannot be found raises ``ChartDomainError``.

The module is deliberately self-contained so that it can be copied into an
experiment directory without modifying the production anatomy package.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TAU = float(2.0 * np.pi)
MM_PER_M = 1000.0

DEFAULT_ASSET_DIR = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/"
    "ab_validation_20261110/repair_20261110/harmonic/shaft_roi_v2"
)
DEFAULT_ATLAS_PATH = DEFAULT_ASSET_DIR / "atlas_right.npz"
SUPPORTED_POSES = (0, 45, 75)


class ChartDomainError(ValueError):
    """Raised when a chart query is outside the checked atlas patch."""


def _angle_delta(values: np.ndarray | float, reference: np.ndarray | float) -> np.ndarray:
    """Return the signed periodic delta ``values-reference`` in radians."""

    return (np.asarray(values) - np.asarray(reference) + np.pi) % TAU - np.pi


def _normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Normalize a vector, optionally using a fallback for a zero vector."""

    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        if fallback is None:
            raise ChartDomainError("cannot normalize a near-zero geometry vector")
        value = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(value))
        if norm <= 1.0e-12:
            raise ChartDomainError("geometry fallback is also near zero")
    return value / norm


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert numpy values to values suitable for metadata/JSON output."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class _TetLocation:
    """Barycentric location of one label-space query."""

    tet_index: int
    barycentric: np.ndarray


class LegChart:
    """Bounded chart patch backed by the repaired right-leg tetra atlas.

    Parameters
    ----------
    pose_deg:
        One of the repaired pose samples ``0``, ``45`` or ``75`` degrees.
    atlas_path:
        Path to ``atlas_right.npz``.  The sibling ``right_pose_XX.npz`` is
        loaded from the same directory.
    x_bounds_mm, y_bounds_mm, z_bounds_mm:
        Nominal chart bounds.  The defaults are the checked patch requested by
        the experiment: x ±42 mm, y ±36 mm and z 0..26 mm.

    Notes
    -----
    ``world`` and ``world_from_xi`` return mm, while the source atlas stores
    canonical and posed vertices in metres.  ``xi`` returns radians for
    theta, unitless h and unitless d.
    """

    def __init__(
        self,
        pose_deg: int = 0,
        *,
        atlas_path: Path | str = DEFAULT_ATLAS_PATH,
        x_bounds_mm: tuple[float, float] = (-42.0, 42.0),
        y_bounds_mm: tuple[float, float] = (-36.0, 36.0),
        z_bounds_mm: tuple[float, float] = (0.0, 26.0),
        tet_candidate_k: int = 256,
        tet_batch_size: int = 2048,
    ) -> None:
        self.pose_deg = int(pose_deg)
        if self.pose_deg not in SUPPORTED_POSES:
            raise ValueError(f"pose_deg must be one of {SUPPORTED_POSES}, got {pose_deg!r}")
        self.atlas_path = Path(atlas_path).expanduser().resolve()
        self.asset_dir = self.atlas_path.parent
        self.pose_path = self.asset_dir / f"right_pose_{self.pose_deg:02d}.npz"
        if not self.atlas_path.is_file():
            raise FileNotFoundError(self.atlas_path)
        if not self.pose_path.is_file():
            raise FileNotFoundError(self.pose_path)
        self.x_bounds_mm = self._validate_bounds("x", x_bounds_mm)
        self.y_bounds_mm = self._validate_bounds("y", y_bounds_mm)
        self.z_bounds_mm = self._validate_bounds("z", z_bounds_mm)
        if self.z_bounds_mm[0] < 0.0:
            raise ValueError("z chart coordinate must be non-negative inward depth")
        self._tet_candidate_k = max(8, int(tet_candidate_k))
        self._tet_batch_size = max(128, int(tet_batch_size))

        with np.load(self.atlas_path, allow_pickle=False) as payload:
            self._load_atlas_payload(payload)
        with np.load(self.pose_path, allow_pickle=False) as payload:
            if "vertices" not in payload.files:
                raise ValueError(f"pose asset has no vertices array: {self.pose_path}")
            self.pose_vertices_m = np.asarray(payload["vertices"], dtype=np.float64).reshape(-1, 3)
            self.pose_core_points_m = (
                np.asarray(payload["core_points"], dtype=np.float64).reshape(-1, 3)
                if "core_points" in payload.files
                else self.core_points_m.copy()
            )
        if self.pose_vertices_m.shape != self.harmonic_vertices_m.shape:
            raise ValueError(
                f"pose vertices shape {self.pose_vertices_m.shape} does not match atlas "
                f"{self.harmonic_vertices_m.shape}"
            )

        self._prepare_centerline()
        self._prepare_skin_chart()
        self._prepare_tet_lookup()
        self.metadata = self._make_metadata()

    @staticmethod
    def _validate_bounds(name: str, values: Iterable[float]) -> tuple[float, float]:
        bounds = tuple(float(v) for v in values)
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError(f"{name}_bounds_mm must be an increasing finite pair")
        return bounds

    def _load_atlas_payload(self, payload: Any) -> None:
        required = (
            "skin_vertices",
            "skin_faces",
            "skin_theta",
            "skin_h",
            "skin_d",
            "core_points",
            "core_h",
            "knee",
            "harmonic_vertices",
            "harmonic_tets",
            "harmonic_theta",
            "harmonic_h",
            "harmonic_d",
        )
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise ValueError(f"atlas is missing arrays: {missing}")
        self.skin_vertices_m = np.asarray(payload["skin_vertices"], dtype=np.float64).reshape(-1, 3)
        self.skin_faces = np.asarray(payload["skin_faces"], dtype=np.int64).reshape(-1, 3)
        self.skin_theta = np.mod(np.asarray(payload["skin_theta"], dtype=np.float64).reshape(-1), TAU)
        self.skin_h = np.asarray(payload["skin_h"], dtype=np.float64).reshape(-1)
        self.skin_d = np.asarray(payload["skin_d"], dtype=np.float64).reshape(-1)
        self.core_points_m = np.asarray(payload["core_points"], dtype=np.float64).reshape(-1, 3)
        self.core_h = np.asarray(payload["core_h"], dtype=np.float64).reshape(-1)
        self.knee_m = np.asarray(payload["knee"], dtype=np.float64).reshape(3)
        self.harmonic_vertices_m = np.asarray(payload["harmonic_vertices"], dtype=np.float64).reshape(-1, 3)
        self.harmonic_tets = np.asarray(payload["harmonic_tets"], dtype=np.int64).reshape(-1, 4)
        self.harmonic_theta = np.mod(np.asarray(payload["harmonic_theta"], dtype=np.float64).reshape(-1), TAU)
        self.harmonic_h = np.asarray(payload["harmonic_h"], dtype=np.float64).reshape(-1)
        self.harmonic_d = np.asarray(payload["harmonic_d"], dtype=np.float64).reshape(-1)
        if "skin_normals" in payload.files:
            self.skin_normals = np.asarray(payload["skin_normals"], dtype=np.float64).reshape(-1, 3)
        else:
            self.skin_normals = np.zeros_like(self.skin_vertices_m)
        if "metadata_json" in payload.files:
            try:
                self.atlas_metadata = json.loads(str(payload["metadata_json"].item()))
            except Exception:
                self.atlas_metadata = {}
        else:
            self.atlas_metadata = {}
        if self.skin_vertices_m.shape[0] != self.skin_h.size or self.skin_h.size != self.skin_theta.size:
            raise ValueError("skin geometry and labels have inconsistent lengths")
        if self.harmonic_vertices_m.shape[0] != self.harmonic_h.size:
            raise ValueError("harmonic geometry and h labels have inconsistent lengths")
        if self.harmonic_h.size != self.harmonic_theta.size or self.harmonic_h.size != self.harmonic_d.size:
            raise ValueError("harmonic labels have inconsistent lengths")
        if self.harmonic_tets.size and int(self.harmonic_tets.max()) >= self.harmonic_vertices_m.shape[0]:
            raise ValueError("harmonic tetrahedron index exceeds vertex count")

    def _prepare_centerline(self) -> None:
        if self.core_points_m.shape[0] < 3 or self.core_points_m.shape[0] != self.core_h.size:
            raise ValueError("core centerline requires at least three points with matching h labels")
        seg = np.linalg.norm(np.diff(self.core_points_m, axis=0), axis=1)
        if np.any(seg <= 1.0e-10):
            raise ValueError("core centerline contains a duplicate station")
        self.core_s_m = np.concatenate(([0.0], np.cumsum(seg)))
        if np.any(np.diff(self.core_h) <= 0.0):
            raise ValueError("core h labels must be strictly increasing")
        best_dist = float("inf")
        best_i = 0
        best_t = 0.0
        for index, (a, b) in enumerate(zip(self.core_points_m[:-1], self.core_points_m[1:], strict=True)):
            ab = b - a
            t = float(np.clip(((self.knee_m - a) @ ab) / float(ab @ ab), 0.0, 1.0))
            q = a + t * ab
            distance = float(np.linalg.norm(self.knee_m - q))
            if distance < best_dist:
                best_dist, best_i, best_t = distance, index, t
        self.knee_projection_m = self.core_points_m[best_i] + best_t * (
            self.core_points_m[best_i + 1] - self.core_points_m[best_i]
        )
        self.knee_center_s_m = float(self.core_s_m[best_i] + best_t * seg[best_i])
        self.h_center = float(np.interp(self.knee_center_s_m, self.core_s_m, self.core_h))
        self.knee_center_offset_mm = float(best_dist * MM_PER_M)
        s_lo = self.knee_center_s_m + self.x_bounds_mm[0] / MM_PER_M
        s_hi = self.knee_center_s_m + self.x_bounds_mm[1] / MM_PER_M
        if s_lo < self.core_s_m[0] or s_hi > self.core_s_m[-1]:
            raise ChartDomainError(
                "requested x patch is outside the repaired centerline; "
                f"allowable x is [{-self.knee_center_s_m * MM_PER_M:.3f}, "
                f"{(self.core_s_m[-1] - self.knee_center_s_m) * MM_PER_M:.3f}] mm"
            )
        self.h_bounds = (
            float(np.interp(s_lo, self.core_s_m, self.core_h)),
            float(np.interp(s_hi, self.core_s_m, self.core_h)),
        )
        if self.h_bounds[0] < float(self.core_h[0]) or self.h_bounds[1] > float(self.core_h[-1]):
            raise ChartDomainError("x patch extends outside the atlas h domain")
        # The tangent is positive with increasing h / distal direction.
        tangent = self._centerline_tangent_at_s(self.knee_center_s_m)
        lateral_seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        lateral_seed -= float(lateral_seed @ tangent) * tangent
        self.lateral_axis = _normalize(lateral_seed, np.asarray([1.0, 0.0, 0.0]))
        if float(self.lateral_axis @ np.asarray([1.0, 0.0, 0.0])) < 0.0:
            self.lateral_axis *= -1.0
        depth_seed = np.cross(tangent, self.lateral_axis)
        depth_seed = _normalize(depth_seed, np.asarray([0.0, 0.0, 1.0]))
        if float(depth_seed @ np.asarray([0.0, 0.0, 1.0])) < 0.0:
            depth_seed *= -1.0
        self.depth_reference_axis = depth_seed
        self.longitudinal_axis_at_knee = tangent

    def _centerline_tangent_at_s(self, s_value: float) -> np.ndarray:
        index = int(np.searchsorted(self.core_s_m, float(s_value), side="right") - 1)
        index = int(np.clip(index, 0, self.core_points_m.shape[0] - 2))
        return _normalize(self.core_points_m[index + 1] - self.core_points_m[index])

    def _prepare_skin_chart(self) -> None:
        self.skin_h_grid = np.unique(self.skin_h)
        self.skin_theta_grid = np.unique(np.mod(self.skin_theta, TAU))
        if self.skin_h_grid.size < 3 or self.skin_theta_grid.size < 8:
            raise ValueError("skin atlas grid is too small for a bounded chart")
        expected = int(self.skin_h_grid.size * self.skin_theta_grid.size)
        if expected != self.skin_vertices_m.shape[0]:
            raise ValueError(
                f"skin atlas is not a complete h/theta grid ({expected} expected, "
                f"{self.skin_vertices_m.shape[0]} vertices)"
            )
        # Reindex by labels rather than trusting file ordering.
        self.skin_grid_vertices_m = np.zeros((self.skin_h_grid.size, self.skin_theta_grid.size, 3), dtype=np.float64)
        for row, (h_value, theta_value, point) in enumerate(
            zip(self.skin_h, self.skin_theta, self.skin_vertices_m, strict=True)
        ):
            hi = int(np.argmin(np.abs(self.skin_h_grid - h_value)))
            ti = int(np.argmin(np.abs(_angle_delta(self.skin_theta_grid, theta_value))))
            if np.linalg.norm(self.skin_grid_vertices_m[hi, ti]) > 0.0:
                # A valid mesh vertex can itself be the zero vector only for a
                # degenerate asset, which this repaired atlas does not have.
                raise ValueError("duplicate skin grid labels")
            self.skin_grid_vertices_m[hi, ti] = point
        self.skin_grid_theta = self.skin_theta_grid.copy()
        # Choose the anterior (+depth-reference) arc at the h station nearest
        # the knee.  This is intentionally away from theta seam 0/2pi.
        h_index = int(np.argmin(np.abs(self.skin_h_grid - self.h_center)))
        ring = self.skin_grid_vertices_m[h_index]
        ring_center = ring.mean(axis=0)
        radial = ring - ring_center[None, :]
        center_theta_idx = int(np.argmax(radial @ self.depth_reference_axis))
        self.theta_reference = float(self.skin_grid_theta[center_theta_idx])
        self.theta_halfspan = self._choose_theta_halfspan()
        self.theta_window = (
            float(self.theta_reference - self.theta_halfspan),
            float(self.theta_reference + self.theta_halfspan),
        )
        if self.theta_window[0] <= 0.0 or self.theta_window[1] >= TAU:
            raise ChartDomainError(
                "the requested lateral patch would cross the atlas theta seam; "
                f"theta window={self.theta_window}"
            )
        self._skin_branch_indices = []
        self._skin_branch_u = []
        self._skin_branch_y_mm = []
        self.skin_ring_centers_m = np.mean(self.skin_grid_vertices_m, axis=1)
        for hi in range(self.skin_h_grid.size):
            u = self.theta_reference + _angle_delta(self.skin_theta_grid, self.theta_reference)
            order = np.argsort(u)
            u_sorted = u[order]
            mask = (u_sorted >= self.theta_window[0]) & (u_sorted <= self.theta_window[1])
            idx = order[mask]
            u_sorted = u_sorted[mask]
            if idx.size < 6:
                raise ChartDomainError("front skin branch has too few theta stations")
            q = self.skin_grid_vertices_m[hi, idx] - self.skin_ring_centers_m[hi]
            y_mm = (q @ self.lateral_axis) * MM_PER_M
            if np.any(np.diff(y_mm) >= -1.0e-6):
                raise ChartDomainError(
                    "visible skin branch is not monotonic in lateral chart y at "
                    f"h={self.skin_h_grid[hi]:.6f}"
                )
            # Stations outside the requested longitudinal patch are retained
            # for interpolation bookkeeping but are not required to cover the
            # chart's lateral bounds (the shaft narrows near the ankle).
            in_requested_h = (
                float(self.skin_h_grid[hi]) >= self.h_bounds[0] - 1.0e-8
                and float(self.skin_h_grid[hi]) <= self.h_bounds[1] + 1.0e-8
            )
            if in_requested_h and (y_mm[0] < self.y_bounds_mm[1] or y_mm[-1] > self.y_bounds_mm[0]):
                raise ChartDomainError(
                    "requested lateral bounds exceed the actual visible skin arc at "
                    f"h={self.skin_h_grid[hi]:.6f}: [{y_mm[-1]:.3f},{y_mm[0]:.3f}] mm"
                )
            self._skin_branch_indices.append(idx.astype(np.int64))
            self._skin_branch_u.append(u_sorted.astype(np.float64))
            self._skin_branch_y_mm.append(y_mm.astype(np.float64))
        # The selected h patch must be covered by the skin grid without h
        # extrapolation.  A small margin avoids relying on a station at the
        # exact x bound for the audit.
        if self.h_bounds[0] < self.skin_h_grid[0] or self.h_bounds[1] > self.skin_h_grid[-1]:
            raise ChartDomainError("x patch is outside the attached skin h grid")

    def _choose_theta_halfspan(self) -> float:
        """Select the smallest non-seam branch covering the requested y range."""

        candidates = (1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40)
        for halfspan in candidates:
            low = self.theta_reference - halfspan
            high = self.theta_reference + halfspan
            if low <= 0.02 or high >= TAU - 0.02:
                continue
            covered = True
            for h_value in np.linspace(self.h_bounds[0], self.h_bounds[1], 9):
                hi = int(np.searchsorted(self.skin_h_grid, h_value, side="right") - 1)
                hi = int(np.clip(hi, 0, self.skin_h_grid.size - 2))
                weight = float((h_value - self.skin_h_grid[hi]) / (self.skin_h_grid[hi + 1] - self.skin_h_grid[hi]))
                ranges = []
                for station in (hi, hi + 1):
                    u = self.theta_reference + _angle_delta(self.skin_theta_grid, self.theta_reference)
                    mask = (u >= self.theta_reference - halfspan) & (u <= self.theta_reference + halfspan)
                    q = self.skin_grid_vertices_m[station, mask] - self.skin_grid_vertices_m[station].mean(axis=0)
                    y_values = (q @ self.lateral_axis) * MM_PER_M
                    ranges.append((float(y_values.min()), float(y_values.max())))
                y_min = (1.0 - weight) * ranges[0][0] + weight * ranges[1][0]
                y_max = (1.0 - weight) * ranges[0][1] + weight * ranges[1][1]
                if y_min > self.y_bounds_mm[0] + 1.0e-6 or y_max < self.y_bounds_mm[1] - 1.0e-6:
                    covered = False
                    break
            if covered:
                return float(halfspan)
        raise ChartDomainError(
            "requested y patch is wider than the non-seam anterior skin arc; "
            f"actual theta reference={self.theta_reference:.6f}"
        )

    def _prepare_tet_lookup(self) -> None:
        if self.harmonic_tets.shape[0] == 0:
            raise ValueError("atlas contains no harmonic tetrahedra")
        labels = np.stack([self.harmonic_theta, self.harmonic_h, self.harmonic_d], axis=1)
        self._tet_labels = labels[self.harmonic_tets]
        theta = self._tet_labels[:, :, 0]
        self._tet_features = np.stack(
            [np.mean(np.cos(theta), axis=1), np.mean(np.sin(theta), axis=1), np.mean(self._tet_labels[:, :, 1], axis=1), np.mean(self._tet_labels[:, :, 2], axis=1)],
            axis=1,
        )
        try:
            from scipy.spatial import cKDTree  # type: ignore

            self._tet_tree = cKDTree(self._tet_features)
        except Exception as exc:  # pragma: no cover - scipy is in the genesis env
            raise RuntimeError("LegChart requires scipy.spatial.cKDTree") from exc

    def _make_metadata(self) -> dict[str, Any]:
        asset_meta = {
            "atlas_right.npz": {"path": str(self.atlas_path), "sha256": _sha256(self.atlas_path)},
            self.pose_path.name: {"path": str(self.pose_path), "sha256": _sha256(self.pose_path)},
        }
        return _jsonable(
            {
                "module": "ab_leg_bifurcation_20261112.leg_chart",
                "asset_paths_and_hashes": asset_meta,
                "pose_deg": self.pose_deg,
                "supported_pose_degrees": list(SUPPORTED_POSES),
                "source_asset_metadata": self.atlas_metadata,
                "units": {"chart": "mm nominal", "atlas_and_world_arrays": "m", "theta": "radians"},
                "chart_definition": {
                    "x_mm": list(self.x_bounds_mm),
                    "y_mm": list(self.y_bounds_mm),
                    "z_mm": list(self.z_bounds_mm),
                    "x_positive": "increasing atlas h / distal centerline arclength",
                    "y_positive": "projection onto fixed lateral axis with positive world-x component",
                    "z_positive": "inward from visible anterior skin toward atlas d=1 inner shell",
                    "skin": "z=0 exactly maps to atlas d=0",
                    "interior": "d=z / local Euclidean skin-to-d=1 distance; no d extrapolation",
                    "theta_window_radians": list(self.theta_window),
                    "theta_reference_radians": float(self.theta_reference),
                    "h_window": list(self.h_bounds),
                    "h_center": float(self.h_center),
                    "knee_projection_offset_mm": float(self.knee_center_offset_mm),
                    "centerline_s_window_mm": [
                        self.x_bounds_mm[0],
                        self.x_bounds_mm[1],
                    ],
                },
                "atlas_labels": {
                    "coordinate": "(theta, h, d)",
                    "theta_seam": "0/2pi avoided; all chart theta queries are interior to selected window",
                    "h_atlas_range": [float(np.min(self.harmonic_h)), float(np.max(self.harmonic_h))],
                    "d_atlas_range": [float(np.min(self.harmonic_d)), float(np.max(self.harmonic_d))],
                    "skin_d_unique": [float(np.min(self.skin_d)), float(np.max(self.skin_d))],
                    "interpolation": "piecewise-linear barycentric interpolation in attached atlas tets",
                    "outside_policy": "raise ChartDomainError; no nearest snap or silent extrapolation",
                },
                "geometry_assumptions": {
                    "right_leg": True,
                    "visible_arc": "anterior +depth-reference arc selected from true skin grid",
                    "lateral_axis_world_m": self.lateral_axis.tolist(),
                    "depth_reference_axis_world_m": self.depth_reference_axis.tolist(),
                    "longitudinal_axis_at_knee_world_m": self.longitudinal_axis_at_knee.tolist(),
                    "shaft_roi_only": True,
                    "no_tissue_or_probe_model": True,
                },
            }
        )

    @property
    def h_domain(self) -> tuple[float, float]:
        """Actual h interval covered by the requested x bounds."""

        return self.h_bounds

    @property
    def theta_domain(self) -> tuple[float, float]:
        """Unwrapped theta interval used by the selected visible skin arc."""

        return self.theta_window

    def _h_from_x(self, x_mm: np.ndarray) -> np.ndarray:
        s = self.knee_center_s_m + np.asarray(x_mm, dtype=np.float64) / MM_PER_M
        return np.interp(s, self.core_s_m, self.core_h)

    def _theta_for_h_y(self, h_values: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
        """Invert the actual visible skin branch for batched h/y queries."""

        h_arr = np.asarray(h_values, dtype=np.float64).reshape(-1)
        y_arr = np.asarray(y_mm, dtype=np.float64).reshape(-1)
        if h_arr.shape != y_arr.shape:
            raise ValueError("h_values and y_mm must have matching shapes")
        if np.any(h_arr < self.h_bounds[0] - 1.0e-8) or np.any(h_arr > self.h_bounds[1] + 1.0e-8):
            raise ChartDomainError("query h is outside the checked x-to-h interval")
        if np.any(y_arr < self.y_bounds_mm[0] - 1.0e-8) or np.any(y_arr > self.y_bounds_mm[1] + 1.0e-8):
            raise ChartDomainError("query y is outside the checked lateral interval")
        out = np.empty_like(h_arr)
        hi = np.searchsorted(self.skin_h_grid, h_arr, side="right") - 1
        hi = np.clip(hi, 0, self.skin_h_grid.size - 2).astype(np.int64)
        for station in np.unique(hi):
            rows = np.flatnonzero(hi == station)
            qy = y_arr[rows]
            u0 = self._skin_branch_u[int(station)]
            y0 = self._skin_branch_y_mm[int(station)]
            if np.any(qy < y0[-1] - 1.0e-7) or np.any(qy > y0[0] + 1.0e-7):
                raise ChartDomainError(
                    "query y is outside the actual visible skin arc at h station "
                    f"{self.skin_h_grid[int(station)]:.6f}: "
                    f"[{y0[-1]:.3f},{y0[0]:.3f}] mm"
                )
            theta0 = np.interp(qy, y0[::-1], u0[::-1])
            next_station = int(station) + 1
            u1 = self._skin_branch_u[next_station]
            y1 = self._skin_branch_y_mm[next_station]
            if np.any(qy < y1[-1] - 1.0e-7) or np.any(qy > y1[0] + 1.0e-7):
                raise ChartDomainError(
                    "query y is outside the actual visible skin arc at h station "
                    f"{self.skin_h_grid[next_station]:.6f}: "
                    f"[{y1[-1]:.3f},{y1[0]:.3f}] mm"
                )
            theta1 = np.interp(qy, y1[::-1], u1[::-1])
            weight = (h_arr[rows] - self.skin_h_grid[int(station)]) / (
                self.skin_h_grid[next_station] - self.skin_h_grid[int(station)]
            )
            out[rows] = (1.0 - weight) * theta0 + weight * theta1
        return out

    def _chart_to_xi(self, points_mm: np.ndarray, *, need_depth: bool = True) -> np.ndarray:
        points = np.asarray(points_mm, dtype=np.float64)
        flat = points.reshape(-1, 3)
        if not np.all(np.isfinite(flat)):
            raise ChartDomainError("chart query contains non-finite values")
        if np.any(flat[:, 0] < self.x_bounds_mm[0] - 1.0e-8) or np.any(flat[:, 0] > self.x_bounds_mm[1] + 1.0e-8):
            raise ChartDomainError(f"x query outside chart bounds {self.x_bounds_mm} mm")
        if np.any(flat[:, 1] < self.y_bounds_mm[0] - 1.0e-8) or np.any(flat[:, 1] > self.y_bounds_mm[1] + 1.0e-8):
            raise ChartDomainError(f"y query outside chart bounds {self.y_bounds_mm} mm")
        if np.any(flat[:, 2] < self.z_bounds_mm[0] - 1.0e-8) or np.any(flat[:, 2] > self.z_bounds_mm[1] + 1.0e-8):
            raise ChartDomainError(f"z query outside chart bounds {self.z_bounds_mm} mm")
        h = self._h_from_x(flat[:, 0])
        theta = self._theta_for_h_y(h, flat[:, 1])
        d = np.zeros(flat.shape[0], dtype=np.float64)
        if need_depth and np.any(flat[:, 2] > 0.0):
            surface_xi = np.stack([theta, h, np.zeros_like(h)], axis=1)
            inner_xi = np.stack([theta, h, np.ones_like(h)], axis=1)
            surface_m = self._map_xi_to_world_m(surface_xi)
            inner_m = self._map_xi_to_world_m(inner_xi)
            thickness_mm = np.linalg.norm(inner_m - surface_m, axis=1) * MM_PER_M
            if np.any(~np.isfinite(thickness_mm)) or np.any(thickness_mm <= 1.0e-5):
                raise ChartDomainError("atlas skin-to-inner-shell distance is invalid in chart patch")
            positive = flat[:, 2] > 0.0
            d[positive] = flat[positive, 2] / thickness_mm[positive]
            if np.any(d > 1.0 + 2.0e-6):
                bad = int(np.flatnonzero(d > 1.0 + 2.0e-6)[0])
                raise ChartDomainError(
                    "nominal depth exceeds actual attached atlas depth at query "
                    f"{flat[bad].tolist()}: requested d={d[bad]:.6f}"
                )
            d = np.clip(d, 0.0, 1.0)
        return np.stack([theta, h, d], axis=1).reshape(points.shape)

    def xi(self, points_mm: np.ndarray) -> np.ndarray:
        """Return actual atlas labels ``(theta radians, h, d)`` for chart points."""

        return self._chart_to_xi(points_mm, need_depth=True)

    def world(self, points_mm: np.ndarray) -> np.ndarray:
        """Map chart points in nominal mm to posed world mm.

        The input may have any leading batch shape ending in three coordinates.
        All queries are strict: no coordinate outside the bounded chart or
        outside a containing labelled tetrahedron is silently accepted.
        """

        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError("points_mm must have shape (..., 3)")
        xi = self._chart_to_xi(points, need_depth=True)
        return self._map_xi_to_world_m(xi.reshape(-1, 3)).reshape(points.shape) * MM_PER_M

    def world_from_xi(self, xi: np.ndarray) -> np.ndarray:
        """Map real atlas labels ``(theta, h, d)`` to posed world mm.

        This lower-level method is useful for audits and for callers that have
        already obtained labels from the attached atlas.  It performs the same
        strict containing-tet check as :meth:`world`.
        """

        labels = np.asarray(xi, dtype=np.float64)
        if labels.ndim == 0 or labels.shape[-1] != 3:
            raise ValueError("xi must have shape (..., 3)")
        flat = labels.reshape(-1, 3)
        if not np.all(np.isfinite(flat)):
            raise ChartDomainError("xi query contains non-finite values")
        if np.any(flat[:, 1] < self.h_bounds[0] - 1.0e-8) or np.any(flat[:, 1] > self.h_bounds[1] + 1.0e-8):
            raise ChartDomainError("xi h query outside the bounded chart h interval")
        if np.any(flat[:, 2] < -1.0e-8) or np.any(flat[:, 2] > 1.0 + 1.0e-8):
            raise ChartDomainError("xi d query outside attached atlas")
        theta_mod = np.mod(flat[:, 0], TAU)
        theta_unwrapped = self.theta_reference + _angle_delta(theta_mod, self.theta_reference)
        if np.any(theta_unwrapped < self.theta_window[0] - 1.0e-8) or np.any(theta_unwrapped > self.theta_window[1] + 1.0e-8):
            raise ChartDomainError("xi theta query outside the bounded non-seam chart interval")
        out = self._map_xi_to_world_m(np.stack([theta_mod, flat[:, 1], flat[:, 2]], axis=1))
        return out.reshape(labels.shape) * MM_PER_M

    def frame(self, surface_xy_mm: np.ndarray) -> np.ndarray:
        """Return posed world basis columns ``(longitudinal, lateral, inward)``.

        ``surface_xy_mm`` has shape ``(..., 2)`` and is evaluated at z=0.  The
        first two columns are finite-difference tangents of the actual mapped
        skin surface, with the lateral column oriented so increasing chart y
        has positive dot product.  The third column points toward increasing
        nominal depth and is orthonormalized against the tangents.
        """

        xy = np.asarray(surface_xy_mm, dtype=np.float64)
        if xy.ndim == 0 or xy.shape[-1] != 2:
            raise ValueError("surface_xy_mm must have shape (..., 2)")
        flat = xy.reshape(-1, 2)
        base = np.concatenate([flat, np.zeros((flat.shape[0], 1), dtype=np.float64)], axis=1)
        eps_long = min(0.5, 0.25 * (self.x_bounds_mm[1] - self.x_bounds_mm[0]))
        eps_lat = min(0.5, 0.25 * (self.y_bounds_mm[1] - self.y_bounds_mm[0]))
        long_plus = base.copy()
        long_minus = base.copy()
        long_plus[:, 0] = np.minimum(long_plus[:, 0] + eps_long, self.x_bounds_mm[1])
        long_minus[:, 0] = np.maximum(long_minus[:, 0] - eps_long, self.x_bounds_mm[0])
        lat_plus = base.copy()
        lat_minus = base.copy()
        lat_plus[:, 1] = np.minimum(lat_plus[:, 1] + eps_lat, self.y_bounds_mm[1])
        lat_minus[:, 1] = np.maximum(lat_minus[:, 1] - eps_lat, self.y_bounds_mm[0])
        p0 = self.world(base)
        pp = self.world(long_plus)
        pm = self.world(long_minus)
        qp = self.world(lat_plus)
        qm = self.world(lat_minus)
        inward_query = base.copy()
        inward_query[:, 2] = min(1.0, max(0.5, self.z_bounds_mm[1] * 0.04))
        pin = self.world(inward_query)
        e_long = pp - pm
        e_lat_raw = qp - qm
        e_in_raw = pin - p0
        e_long = e_long / np.maximum(np.linalg.norm(e_long, axis=1, keepdims=True), 1.0e-12)
        e_lat = e_lat_raw - np.sum(e_lat_raw * e_long, axis=1, keepdims=True) * e_long
        e_lat = e_lat / np.maximum(np.linalg.norm(e_lat, axis=1, keepdims=True), 1.0e-12)
        # Keep the sign tied to increasing chart y even after Gram-Schmidt.
        flip_lat = np.sum(e_lat * e_lat_raw, axis=1) < 0.0
        e_lat[flip_lat] *= -1.0
        e_in = e_in_raw - np.sum(e_in_raw * e_long, axis=1, keepdims=True) * e_long
        e_in = e_in - np.sum(e_in * e_lat, axis=1, keepdims=True) * e_lat
        e_in = e_in / np.maximum(np.linalg.norm(e_in, axis=1, keepdims=True), 1.0e-12)
        flip_in = np.sum(e_in * e_in_raw, axis=1) < 0.0
        e_in[flip_in] *= -1.0
        basis = np.stack([e_long, e_lat, e_in], axis=-1)
        return basis.reshape(xy.shape[:-1] + (3, 3))

    def depth_report(self, surface_xy_mm: np.ndarray, depths_mm: Iterable[float] = (12.0, 18.0, 24.0)) -> dict[str, Any]:
        """Report actual physical distances for selected nominal depths."""

        xy = np.asarray(surface_xy_mm, dtype=np.float64).reshape(-1, 2)
        rows: dict[str, Any] = {}
        surface = self.world(np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1))
        for depth in depths_mm:
            z = float(depth)
            points = np.concatenate([xy, np.full((xy.shape[0], 1), z)], axis=1)
            world = self.world(points)
            physical = np.linalg.norm(world - surface, axis=1)
            inward = self.frame(xy)[:, :, 2]
            signed = np.sum((world - surface) * inward, axis=1)
            rows[str(z)] = {
                "nominal_depth_mm": z,
                "physical_distance_mm_min": float(np.min(physical)),
                "physical_distance_mm_median": float(np.median(physical)),
                "physical_distance_mm_max": float(np.max(physical)),
                "inward_signed_projection_mm_min": float(np.min(signed)),
            }
        return rows

    def _map_xi_to_world_m(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.float64).reshape(-1, 3)
        if labels.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        query_features = np.stack(
            [np.cos(labels[:, 0]), np.sin(labels[:, 0]), labels[:, 1], labels[:, 2]], axis=1
        )
        candidate_k = min(self._tet_candidate_k, self.harmonic_tets.shape[0])
        _dist, candidate_ids = self._tet_tree.query(query_features, k=candidate_k)
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        if candidate_k == 1:
            candidate_ids = candidate_ids[:, None]
        out = np.empty((labels.shape[0], 3), dtype=np.float64)
        unresolved: list[tuple[int, np.ndarray]] = []
        for start in range(0, labels.shape[0], self._tet_batch_size):
            stop = min(start + self._tet_batch_size, labels.shape[0])
            q = labels[start:stop]
            candidates = candidate_ids[start:stop]
            tet_labels = self._tet_labels[candidates].copy()
            # Unwrap each candidate tetra's theta values around the query theta.
            tet_labels[:, :, :, 0] = q[:, None, None, 0] + _angle_delta(
                tet_labels[:, :, :, 0], q[:, None, None, 0]
            )
            a = tet_labels[:, :, 0, :]
            mat = np.transpose(tet_labels[:, :, 1:, :] - a[:, :, None, :], (0, 1, 3, 2))
            rhs = (q[:, None, :] - a)[..., None]
            try:
                local = np.linalg.solve(mat, rhs)[..., 0]
            except np.linalg.LinAlgError as exc:
                raise ChartDomainError("label-space tetrahedron is singular in checked atlas") from exc
            bary = np.concatenate([1.0 - np.sum(local, axis=2, keepdims=True), local], axis=2)
            valid = np.all(bary >= -2.0e-6, axis=2) & np.all(bary <= 1.0 + 2.0e-6, axis=2)
            score = np.where(valid, np.min(bary, axis=2), -np.inf)
            chosen = np.argmax(score, axis=1)
            best = score[np.arange(score.shape[0]), chosen]
            good = np.isfinite(best)
            if np.any(good):
                rows = np.flatnonzero(good)
                for row in rows:
                    weights = np.asarray(bary[row, chosen[row]], dtype=np.float64)
                    weights = np.clip(weights, 0.0, 1.0)
                    weights /= float(np.sum(weights))
                    tet = self.harmonic_tets[int(candidates[row, chosen[row]])]
                    out[start + row] = weights @ self.pose_vertices_m[tet]
            if np.any(~good):
                unresolved.extend((start + int(row), q[row].copy()) for row in np.flatnonzero(~good))
        if unresolved:
            # A nearest-centroid candidate miss is a lookup failure, not a
            # reason to snap.  Retry once with a larger exact candidate set so
            # numerical boundary ties do not create false negatives.
            if candidate_k < min(4096, self.harmonic_tets.shape[0]):
                old_k = self._tet_candidate_k
                self._tet_candidate_k = min(max(4096, 2 * old_k), self.harmonic_tets.shape[0])
                try:
                    retry_labels = np.stack([item[1] for item in unresolved], axis=0)
                    retry_points = self._map_xi_to_world_m(retry_labels)
                finally:
                    self._tet_candidate_k = old_k
                for (row, _label), point in zip(unresolved, retry_points, strict=True):
                    out[row] = point
                return out
            row, label = unresolved[0]
            raise ChartDomainError(
                "no containing attached atlas tetrahedron for xi="
                f"{label.tolist()} (query row {row}); refusing outside snap/extrapolation"
            )
        return out


__all__ = ["ChartDomainError", "LegChart", "DEFAULT_ASSET_DIR", "DEFAULT_ATLAS_PATH", "SUPPORTED_POSES"]

