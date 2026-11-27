"""Read an existing leg atlas without reconstructing its harmonic solution.

The internal reference is the *original_points* vessel chain in the production
export.  ``projected_points`` are used only as a nominal skin route.  Material
interpolation uses the existing volume's periodic theta / h / radial layers.
The knee/shape transform is a controlled, invertible geometric surrogate, not
SMPL skinning, a tissue simulation, or a newly solved harmonic field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


DEFAULT_ROOT = Path("/media/camp/EXT_DRIVE/Among_US")
CHAIN_LABELS = ("R_COM_FEM_V", "R_SUPFEMV", "R_POPV", "R_POST_TIB_V")
TAU = 2.0 * np.pi


def _points(value: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    a = np.asarray(value, dtype=float)
    if a.ndim == 0 or a.shape[-1] != 3 or not np.all(np.isfinite(a)):
        raise ValueError("Expected finite coordinates with final dimension 3")
    return a.reshape(-1, 3), a.shape


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-15)


@dataclass
class AtlasLeg:
    """One real exported atlas plus a declared geometric pose/shape surrogate.

    ``centerline_xi`` and ``centerline_material_ids`` are canonical and never
    depend on pose or body shape.  Material inverse queries are numerical
    approximations; inspect ``audit()`` rather than assuming exact FEM inversion.
    All point methods accept either (3,) or (...,3) arrays.
    """

    atlas_path: Path
    vessel_path: Path
    skin_vertices: np.ndarray
    skin_faces: np.ndarray
    skin_normals: np.ndarray
    skin_xi: np.ndarray
    volume_points: np.ndarray
    volume_xi: np.ndarray
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray
    centerline_points: np.ndarray
    skin_reference_points: np.ndarray
    centerline_labels: np.ndarray
    centerline_source_indices: np.ndarray
    source_metadata: dict[str, Any]
    centerline_xi: np.ndarray = field(init=False)
    centerline_material_ids: np.ndarray = field(init=False)
    centerline_progress: np.ndarray = field(init=False)

    @classmethod
    def load(cls, root: str | Path = DEFAULT_ROOT) -> "AtlasLeg":
        base = Path(root) / "dataset/processed/anatomy_retarget/leg_volume_coordinates/production"
        atlas_path = base / "atlas/atlas_right.npz"
        vessel_path = base / "vessels/vessel_material_coordinates.npz"
        for path in (atlas_path, vessel_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required preexisting anatomy asset is missing: {path}")
        with np.load(atlas_path, allow_pickle=False) as a:
            atlas = {k: np.asarray(a[k], dtype=float) for k in (
                "skin_vertices", "skin_normals", "volume_points", "volume_xi",
                "hip", "knee", "ankle")}
            atlas["skin_xi"] = np.column_stack((a["skin_theta"], a["skin_h"], a["skin_d"]))
            atlas["skin_faces"] = np.asarray(a["skin_faces"], dtype=np.int32)
            metadata = json.loads(a["metadata_json"].item())
        # This trusted local export uses object arrays for its anatomical labels.
        with np.load(vessel_path, allow_pickle=True) as a:
            labels = np.asarray(a["labels"], dtype=str)
            originals = np.asarray(a["original_points"], dtype=float)
            skin_refs = np.asarray(a["projected_points"], dtype=float)
        indices: list[int] = []
        for label in CHAIN_LABELS:
            group = np.flatnonzero(labels == label)
            if len(group) < 2:
                raise ValueError(f"Missing ordered vessel segment {label} in {vessel_path}")
            if not np.all(np.diff(group) == 1):
                raise ValueError(f"Vessel group {label} is not contiguous in source export")
            if indices:
                gap = np.linalg.norm(originals[indices[-1]] - originals[group[0]])
                if gap > 1e-5:
                    raise ValueError(f"Vessel chain junction for {label} is disconnected ({gap:g} m)")
                group = group[1:]
            indices.extend(group.tolist())
        idx = np.asarray(indices, dtype=int)
        ds = np.linalg.norm(np.diff(originals[idx], axis=0), axis=1)
        if np.any(ds <= 1e-8) or np.any(ds > 0.01):
            raise ValueError("Internal vessel export has duplicate or discontinuous samples")
        return cls(atlas_path=atlas_path, vessel_path=vessel_path, **atlas,
                   centerline_points=originals[idx], skin_reference_points=skin_refs[idx],
                   centerline_labels=labels[idx], centerline_source_indices=idx,
                   source_metadata=metadata)

    def __post_init__(self) -> None:
        self._world_tree = cKDTree(self.volume_points)
        self._skin_tree = cKDTree(self.skin_vertices)
        self._theta = np.unique(self.volume_xi[:, 0])
        self._h = np.unique(self.volume_xi[:, 1])
        nt, nh = len(self._theta), len(self._h)
        nr, remainder = divmod(len(self.volume_points), nt * nh)
        if remainder or nr < 2:
            raise ValueError("Atlas does not have the supported layered h/radial/theta layout")
        self._grid = self.volume_points.reshape(nh, nr, nt, 3)
        xi_grid = self.volume_xi.reshape(nh, nr, nt, 3)
        if not (np.allclose(xi_grid[..., 0], self._theta[None, None, :])
                and np.allclose(xi_grid[..., 1], self._h[:, None, None])):
            raise ValueError("Atlas volume ordering differs from its layered metadata")
        self._d_grid = xi_grid[..., 2]
        if not np.all(np.diff(self._d_grid, axis=1) < 0):
            raise ValueError("Atlas radial d samples must decrease monotonically toward skin")
        self.centerline_xi = self.material_at_point(self.centerline_points)
        self._raw_chain_count = len(self.centerline_points)
        in_chart = (self.centerline_xi[:, 1] >= self._h[0]) & (self.centerline_xi[:, 1] <= self._h[-1])
        supported = np.flatnonzero(in_chart)
        if len(supported) < 10 or not np.all(np.diff(supported) == 1):
            raise ValueError("No continuous usable vessel chain inside sampled atlas h range")
        self._trimmed_source_indices = self.centerline_source_indices[~in_chart].copy()
        # Preserve original coordinates and source order; exclude only the two
        # unsupported chart-end sections, never replace them with skin points.
        for name in ("centerline_points", "skin_reference_points", "centerline_labels",
                     "centerline_source_indices", "centerline_xi"):
            setattr(self, name, getattr(self, name)[supported])
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.centerline_points, axis=0), axis=1))]
        self.centerline_progress = lengths / lengths[-1]
        self.centerline_length_m = float(lengths[-1])
        self.centerline_material_ids = self.centerline_source_indices.copy()
        self._centerline_residual = np.linalg.norm(
            self.point_at_material(self.centerline_xi) - self.centerline_points, axis=1)
        # A poor coordinate recovery must not silently become internal truth.
        if np.max(self._centerline_residual) > 0.005:
            bad = np.argsort(self._centerline_residual)[-5:]
            raise ValueError("Internal centerline inverse exceeds 5 mm; inspect atlas/source alignment: "
                             f"indices={bad.tolist()} errors={self._centerline_residual[bad].tolist()} "
                             f"xi={self.centerline_xi[bad].tolist()}")
        self.deformation_model = "simplified_finite_knee_arc_not_SMPL_or_recomputed_harmonics"

    def point_at_material(self, xi: np.ndarray) -> np.ndarray:
        """Interpolate existing atlas samples, periodic in theta.

        h in [0,1] is allowed; h outside sampled [.05,.97] is explicitly linearly
        extrapolated from the end layers. d must be in [0,1]. Skin (d=0) here is
        the volume grid's outer layer, not an exact projection onto skin faces.
        """
        q, shape = _points(xi)
        if np.any((q[:, 1:] < -1e-9) | (q[:, 1:] > 1 + 1e-9)):
            raise ValueError("Material h and d must lie in [0,1]")
        theta = q[:, 0] % TAU
        tt = np.r_[self._theta, TAU]
        ti = np.clip(np.searchsorted(tt, theta, side="right") - 1, 0, len(self._theta) - 1)
        tj = (ti + 1) % len(self._theta)
        wt = ((theta - tt[ti]) / (tt[ti + 1] - tt[ti]))[:, None]
        hi = np.clip(np.searchsorted(self._h, q[:, 1], side="right") - 1, 0, len(self._h) - 2)
        wh = ((q[:, 1] - self._h[hi]) / (self._h[hi + 1] - self._h[hi]))[:, None]

        def radial_at_h(hindex: np.ndarray) -> np.ndarray:
            ps = ((1 - wt[..., None]) * self._grid[hindex, :, ti]
                  + wt[..., None] * self._grid[hindex, :, tj])[:, ::-1]
            ds = ((1 - wt) * self._d_grid[hindex, :, ti]
                  + wt * self._d_grid[hindex, :, tj])[:, ::-1]
            ri = np.clip(np.sum(ds <= q[:, 2, None], axis=1) - 1, 0, ds.shape[1] - 2)
            rows = np.arange(len(q))
            wr = (q[:, 2] - ds[rows, ri]) / (ds[rows, ri + 1] - ds[rows, ri])
            return ps[rows, ri] * (1 - wr[:, None]) + ps[rows, ri + 1] * wr[:, None]

        # Interpolate each station at the same material depth *before* h.
        # Extrapolating raw d layers instead could reverse their ordering near
        # the chart ends; this keeps the source depth parameterization intact.
        out = (1 - wh) * radial_at_h(hi) + wh * radial_at_h(hi + 1)
        return out.reshape(shape)

    def material_at_point(self, point: np.ndarray, *, refine: bool = True) -> np.ndarray:
        """Approximate canonical inverse; nearest volume seed then local solve.

        This is not a tetrahedral containment guarantee. Outside the represented
        volume it returns the closest bounded local fit; ``roundtrip_errors``
        exposes that error so callers can reject unsupported points.
        """
        p, shape = _points(point)
        _, indices = self._world_tree.query(p)
        result = self.volume_xi[indices].copy()
        if refine:
            for i, target in enumerate(p):
                fit = least_squares(
                    lambda q: self.point_at_material(q) - target,
                    np.clip(result[i], [-TAU, 1e-10, 1e-10], [2 * TAU, 1 - 1e-10, 1 - 1e-10]),
                    bounds=([-TAU, 0, 0], [2 * TAU, 1, 1]),
                    x_scale=[1.0, 0.2, 0.5], max_nfev=35,
                    ftol=1e-10, xtol=1e-10, gtol=1e-10,
                )
                result[i] = fit.x
        result[:, 0] %= TAU
        return result.reshape(shape)

    @staticmethod
    def _shape(shape: tuple[float, float]) -> tuple[float, float]:
        if len(shape) != 2:
            raise ValueError("shape is (width_scale, length_scale)")
        width, length = map(float, shape)
        if not np.isfinite(width + length) or min(width, length) <= 0:
            raise ValueError("Body shape scales must be positive and finite")
        return width, length

    def _bend_parameters(self, knee_deg: float, shape: tuple[float, float]):
        width, length = self._shape(shape)
        if not np.isfinite(knee_deg) or not 0 <= knee_deg <= 75:
            raise ValueError("This surrogate supports knee flexion from 0 to 75 degrees")
        scale = np.array([width, length, width])
        knee = self.hip + (self.knee - self.hip) * scale
        bend_length = 0.22 * length
        return scale, knee, bend_length, np.deg2rad(knee_deg)

    def deform_points(self, points: np.ndarray, knee_deg: float = 0,
                      shape: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """Apply width/length scaling and a finite arc bend about atlas knee.

        A 22 cm canonical region bends continuously; the distal part rotates
        rigidly after the arc. This smooth tube geometry is deliberately labeled
        a simplified articulation. No learned anatomy or tissue model is implied.
        """
        p, out_shape = _points(points)
        scale, knee, length, angle = self._bend_parameters(knee_deg, shape)
        p = self.hip + (p - self.hip) * scale
        if angle < 1e-12:
            return p.reshape(out_shape)
        k = angle / length
        l = knee[1] - p[:, 1]
        t = np.clip(l + length / 2, 0, length)
        a = k * t
        tail = np.maximum(l - length / 2, 0)
        before = np.minimum(l + length / 2, 0)
        cy = knee[1] + length / 2 - np.sin(a) / k - tail * np.cos(angle) - before
        cz = knee[2] + (1 - np.cos(a)) / k + tail * np.sin(angle)
        zoff = p[:, 2] - knee[2]
        # Beyond the curvature center the tube folds over and no unique inverse
        # exists. Reject such unsupported shape/point combinations explicitly.
        inside = (l >= -length / 2) & (l <= length / 2)
        if np.any(k * zoff[inside] >= 0.95):
            raise ValueError("Shape/point combination folds the simplified knee bend")
        p[:, 1] = cy + zoff * np.sin(a)
        p[:, 2] = cz + zoff * np.cos(a)
        return p.reshape(out_shape)

    def undeform_points(self, points: np.ndarray, knee_deg: float = 0,
                        shape: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        """Analytic inverse of the declared finite-arc deformation."""
        p, out_shape = _points(points)
        p = p.copy()
        scale, knee, length, angle = self._bend_parameters(knee_deg, shape)
        if angle >= 1e-12:
            k = angle / length
            base_y = knee[1] + length / 2
            center_z = knee[2] + 1 / k
            a = np.arctan2(base_y - p[:, 1], center_z - p[:, 2])
            middle = (a >= 0) & (a <= angle)
            proximal = a < 0
            distal = a > angle
            l = np.empty(len(p))
            zoff = np.empty(len(p))
            l[proximal] = knee[1] - p[proximal, 1]
            zoff[proximal] = p[proximal, 2] - knee[2]
            l[middle] = a[middle] / k - length / 2
            zoff[middle] = 1 / k - np.hypot(base_y - p[middle, 1], center_z - p[middle, 2])
            end_y = base_y - np.sin(angle) / k
            end_z = knee[2] + (1 - np.cos(angle)) / k
            dy, dz = p[distal, 1] - end_y, p[distal, 2] - end_z
            l[distal] = length / 2 - dy * np.cos(angle) + dz * np.sin(angle)
            zoff[distal] = dy * np.sin(angle) + dz * np.cos(angle)
            p[:, 1] = knee[1] - l
            p[:, 2] = knee[2] + zoff
        return (self.hip + (p - self.hip) / scale).reshape(out_shape)

    def material_to_world(self, xi: np.ndarray, knee_deg: float = 0,
                          shape: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        return self.deform_points(self.point_at_material(xi), knee_deg, shape)

    def world_to_material(self, point: np.ndarray, knee_deg: float = 0,
                          shape: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        return self.material_at_point(self.undeform_points(point, knee_deg, shape))

    def _curve(self, values: np.ndarray, progress: np.ndarray | float) -> np.ndarray:
        s = np.asarray(progress, dtype=float)
        if not np.all(np.isfinite(s)) or np.any((s < 0) | (s > 1)):
            raise ValueError("Material progress must lie in [0,1]")
        return np.stack([np.interp(s, self.centerline_progress, values[:, j]) for j in range(3)], axis=-1)

    def reference_points(self, progress: np.ndarray | float) -> np.ndarray:
        """Original canonical internal curve, sampled by canonical arclength."""
        return self._curve(self.centerline_points, progress)

    canonical_curve = reference_points
    canonical_nominal = reference_points

    def nominal_curve(self, progress: np.ndarray | float, knee_deg: float = 0,
                      shape: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        return self.deform_points(self.reference_points(progress), knee_deg, shape)

    def material_curve(self, progress: np.ndarray | float) -> np.ndarray:
        """Continuous canonical material curve with theta unwrapped for interpolation."""
        xi = self.centerline_xi.copy()
        xi[:, 0] = np.unwrap(xi[:, 0])
        out = self._curve(xi, progress)
        out[..., 0] %= TAU
        return out

    def skin_reference_curve(self, progress: np.ndarray | float, knee_deg: float = 0,
                             shape: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        return self.deform_points(self._curve(self.skin_reference_points, progress), knee_deg, shape)

    def _deform_normals(self, points: np.ndarray, normals: np.ndarray, knee_deg: float,
                        shape: tuple[float, float]) -> np.ndarray:
        # Covectors transform with J^-T. Numerical Jacobians also handle end
        # regions and anisotropic scaling consistently with deform_points.
        p, out_shape = _points(points)
        n = np.asarray(normals).reshape(-1, 3)
        eps = 1e-6
        jac = np.stack([(self.deform_points(p + np.eye(3)[j] * eps, knee_deg, shape)
                         - self.deform_points(p - np.eye(3)[j] * eps, knee_deg, shape)) / (2 * eps)
                        for j in range(3)], axis=-1)
        return _unit(np.linalg.solve(np.swapaxes(jac, -1, -2), n[..., None])[..., 0]).reshape(out_shape)

    def skin_point_and_normal(self, material_progress: np.ndarray | float,
                              knee_deg: float = 0, shape: tuple[float, float] = (1.0, 1.0)):
        """Nominal projected skin route and interpolated outward atlas normals.

        This route is a fixed observation-independent prior, never internal truth.
        Skin normals use the eight closest actual atlas vertices.
        """
        canonical = self._curve(self.skin_reference_points, material_progress)
        p, out_shape = _points(canonical)
        dist, idx = self._skin_tree.query(p, k=8)
        weights = 1 / np.maximum(dist, 1e-8) ** 2
        normal = _unit(np.sum(weights[..., None] * self.skin_normals[idx], axis=1))
        return (self.deform_points(p, knee_deg, shape).reshape(out_shape),
                self._deform_normals(p, normal, knee_deg, shape).reshape(out_shape))

    def contact_query(self, world_point: np.ndarray, knee_deg: float = 0,
                      shape: tuple[float, float] = (1.0, 1.0)) -> dict[str, np.ndarray]:
        """Nearest-skin-vertex contact proxy, not mesh collision or force physics."""
        canonical = self.undeform_points(world_point, knee_deg, shape)
        p, out_shape = _points(canonical)
        _, idx = self._skin_tree.query(p)
        skin = self.deform_points(self.skin_vertices[idx], knee_deg, shape)
        normals = self._deform_normals(self.skin_vertices[idx], self.skin_normals[idx], knee_deg, shape)
        distance = np.sum((np.asarray(world_point).reshape(-1, 3) - skin) * normals, axis=1)
        return {"point": skin.reshape(out_shape), "normal": normals.reshape(out_shape),
                "signed_distance_m": distance.reshape(out_shape[:-1]), "skin_vertex_index": idx.reshape(out_shape[:-1])}

    def roundtrip_errors(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        return np.linalg.norm(self.point_at_material(self.material_at_point(p)) - p, axis=-1)

    def audit(self) -> dict[str, Any]:
        """Provenance and approximation errors, suitable for saved run metadata."""
        return {
            "atlas_path": str(self.atlas_path), "vessel_path": str(self.vessel_path),
            "internal_source": "original_points", "skin_reference_source": "projected_points",
            "chain_labels": list(CHAIN_LABELS), "chain_points": len(self.centerline_points),
            "raw_chain_points": self._raw_chain_count,
            "trimmed_unsupported_endpoint_source_indices": self._trimmed_source_indices.tolist(),
            "chain_length_m": self.centerline_length_m,
            "deformation_model": self.deformation_model,
            "harmonic_field": "loaded_existing_volume_xi_not_recomputed",
            "material_interpolation": "periodic_theta_bilinear_h_theta_monotone_radial_d",
            "material_inverse": "nearest_seed_bounded_local_least_squares_approximation",
            "sampled_h_range": [float(self._h[0]), float(self._h[-1])],
            "centerline_extrapolated_h_count": int(np.sum((self.centerline_xi[:, 1] < self._h[0]) |
                                                           (self.centerline_xi[:, 1] > self._h[-1]))),
            "centerline_roundtrip_max_m": float(np.max(self._centerline_residual)),
            "centerline_roundtrip_rms_m": float(np.sqrt(np.mean(self._centerline_residual ** 2))),
            "contact_model": "nearest_atlas_skin_vertex_signed_normal_distance_proxy",
            "source_metadata": self.source_metadata,
        }
