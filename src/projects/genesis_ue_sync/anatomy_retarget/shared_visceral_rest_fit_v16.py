"""A small, offline shared rest-field experiment for visceral/bone contact.

This module deliberately changes one thing: the compiled target rest vertices.
It does not edit the authored sparse weights, the 235 controller bind, or the
runtime pose evaluator.  A single compact Wendland field is fitted from the
two-way signed samples of a closed organ/bone pair and then evaluated once on
all non-bone material points.  Consequently an artery, vein, nerve,
connective-tissue mesh, and an ``UNCUT`` mesh in the field support receive the
same spatial displacement as any nearby organ.

The result is an experiment.  The collision and edge measurements below are
sampled gates; exact triangle-triangle intersection and a full anatomical
review still have to be performed by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json

import numpy as np


DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("Diaphragm", "L2"),
    ("Liver", "Rib_10R"),
)


def _as_points(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [N, 3] array")
    return result


def _mesh_info(asset: Any, name: str) -> tuple[int, int, np.ndarray]:
    names = [str(x) for x in np.asarray(asset.source_mesh_names).tolist()]
    if name not in names:
        raise KeyError(f"unknown source mesh: {name}")
    index = names.index(name)
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[index]
    faces = np.asarray(asset.faces, dtype=np.int64)
    own = faces[
        (faces[:, 0] >= start)
        & (faces[:, 0] < stop)
        & (faces[:, 1] >= start)
        & (faces[:, 1] < stop)
        & (faces[:, 2] >= start)
        & (faces[:, 2] < stop)
    ] - int(start)
    if len(own) == 0:
        raise ValueError(f"source mesh {name} has no local faces")
    return int(start), int(stop), own.astype(np.int32, copy=False)


def _surface_samples(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    points = _as_points(vertices, "surface vertices")
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(triangles) == 0 or np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("surface faces are empty or out of range")
    centroids = points[triangles].mean(axis=1)
    return np.vstack((points, centroids))


def _assert_closed_oriented(faces: np.ndarray, name: str) -> None:
    """Fail closed before using winding numbers as a volume predicate."""

    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(triangles) == 0:
        raise ValueError(f"{name} has no faces")
    directed = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
    )
    _edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    signs = np.where(directed[:, 0] < directed[:, 1], 1.0, -1.0)
    balance = np.bincount(inverse, weights=signs, minlength=len(counts))
    if np.any(counts != 2) or np.any(np.abs(balance) > 0.0):
        raise ValueError(f"{name} must be a closed consistently oriented surface")


def _unit_vectors(vectors: np.ndarray, *, eps: float = 1.0e-10) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    result = np.zeros_like(vectors)
    valid = norms > eps
    result[valid] = vectors[valid] / norms[valid, None]
    return result, norms


def _query_closed_surface(
    query: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return inside mask, depth, closest surface points, and winding."""

    import igl

    points = _as_points(query, "closed-surface query")
    vertices = _as_points(surface_vertices, "closed-surface vertices")
    faces = np.asarray(surface_faces, dtype=np.int32).reshape(-1, 3)
    winding = np.asarray(igl.winding_number(vertices, faces, points), dtype=np.float64).reshape(-1)
    squared, _face, closest = igl.point_mesh_squared_distance(points, vertices, faces)
    depth = np.sqrt(np.maximum(np.asarray(squared, dtype=np.float64), 0.0))
    inside = np.abs(winding) >= 0.5
    if not np.isfinite(depth).all() or not np.isfinite(closest).all():
        raise ValueError("closed-surface query returned non-finite data")
    return inside, depth, np.asarray(closest, dtype=np.float64), winding


@dataclass
class PairConstraintsV16:
    """Two-way displacement observations for one closed organ/bone pair."""

    points: np.ndarray
    displacements: np.ndarray
    labels: np.ndarray
    summary: dict[str, Any]

    def __post_init__(self) -> None:
        self.points = _as_points(self.points, "constraint points")
        self.displacements = _as_points(self.displacements, "constraint displacements")
        self.labels = np.asarray(self.labels).astype(str).reshape(-1)
        if self.displacements.shape != self.points.shape or len(self.labels) != len(self.points):
            raise ValueError("constraint arrays must have matching [N, 3]/[N] shapes")


@dataclass
class SharedWendlandFieldV16:
    """Compact C2 Wendland RBF field with vector-valued coefficients."""

    centers: np.ndarray
    coefficients: np.ndarray
    support_radius_m: float
    regularization: float = 1.0e-6

    def __post_init__(self) -> None:
        self.centers = _as_points(self.centers, "RBF centers").copy()
        self.coefficients = _as_points(self.coefficients, "RBF coefficients").copy()
        if self.coefficients.shape != self.centers.shape:
            raise ValueError("RBF centers and coefficients must have matching shapes")
        self.support_radius_m = float(self.support_radius_m)
        self.regularization = float(self.regularization)
        if not np.isfinite(self.support_radius_m) or not 0.04 <= self.support_radius_m <= 0.06:
            raise ValueError("support_radius_m must be in [0.04, 0.06]")
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and non-negative")
        self.centers.setflags(write=False)
        self.coefficients.setflags(write=False)

    @staticmethod
    def _kernel(distance: np.ndarray, support_radius_m: float) -> np.ndarray:
        u = np.asarray(distance, dtype=np.float64) / float(support_radius_m)
        active = u < 1.0
        x = np.clip(1.0 - u, 0.0, 1.0)
        return np.where(active, x**4 * (4.0 * u + 1.0), 0.0)

    def evaluate(self, points: np.ndarray, *, chunk_size: int = 65536) -> np.ndarray:
        """Evaluate the fixed field without any mesh or joint lookup."""

        query = _as_points(points, "field query")
        result = np.zeros_like(query)
        if not len(query) or not len(self.centers):
            return result
        # scipy's tree keeps the compact support evaluation bounded when the
        # full 394k material points are evaluated at once.
        from scipy.spatial import cKDTree

        tree = cKDTree(self.centers)
        for start in range(0, len(query), max(1, int(chunk_size))):
            stop = min(len(query), start + max(1, int(chunk_size)))
            for row, ids in enumerate(tree.query_ball_point(query[start:stop], self.support_radius_m)):
                if not ids:
                    continue
                local = np.asarray(ids, dtype=np.int64)
                distance = np.linalg.norm(self.centers[local] - query[start + row], axis=1)
                weights = self._kernel(distance, self.support_radius_m)
                result[start + row] = weights @ self.coefficients[local]
        return result

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            centers=self.centers,
            coefficients=self.coefficients,
            support_radius_m=np.asarray(self.support_radius_m),
            regularization=np.asarray(self.regularization),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "SharedWendlandFieldV16":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["centers"],
                data["coefficients"],
                float(data["support_radius_m"]),
                float(data["regularization"]),
            )


def _cluster_constraints(
    points: np.ndarray,
    displacements: np.ndarray,
    labels: np.ndarray,
    *,
    cell_m: float,
    max_centers: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate nearby observations, then cap them by deterministic FPS."""

    points = _as_points(points, "raw constraint points")
    displacements = _as_points(displacements, "raw constraint displacements")
    labels = np.asarray(labels).astype(str).reshape(-1)
    if len(points) != len(displacements) or len(points) != len(labels):
        raise ValueError("raw constraint arrays must have matching lengths")
    if not len(points):
        return points, displacements, labels
    cell = float(cell_m)
    if not np.isfinite(cell) or cell <= 0:
        raise ValueError("constraint clustering cell must be positive")
    keys = np.floor(points / cell).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(keys_sorted[1:] != keys_sorted[:-1], axis=1))]
    stops = np.r_[starts[1:], len(order)]
    out_p = np.empty((len(starts), 3), dtype=np.float64)
    out_d = np.empty((len(starts), 3), dtype=np.float64)
    out_l = np.empty(len(starts), dtype="U32")
    for row, (lo, hi) in enumerate(zip(starts.tolist(), stops.tolist())):
        selected = order[lo:hi]
        out_p[row] = points[selected].mean(axis=0)
        out_d[row] = displacements[selected].mean(axis=0)
        unique, counts = np.unique(labels[selected], return_counts=True)
        out_l[row] = str(unique[np.argmax(counts)])
    if len(out_p) <= int(max_centers):
        return out_p, out_d, out_l
    # Start with the largest requested displacement so the two pair regions
    # are retained even when one pair has many more samples than the other.
    selected = [int(np.argmax(np.linalg.norm(out_d, axis=1)))]
    distance = np.linalg.norm(out_p - out_p[selected[0]], axis=1)
    for _ in range(1, int(max_centers)):
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.linalg.norm(out_p - out_p[index], axis=1))
        distance[np.asarray(selected, dtype=np.int64)] = -1.0
    selected = np.asarray(selected, dtype=np.int64)
    return out_p[selected], out_d[selected], out_l[selected]


def fit_wendland_field_v16(
    points: np.ndarray,
    displacements: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    support_radius_m: float = 0.05,
    cluster_cell_m: float | None = None,
    max_centers: int = 96,
    regularization: float = 0.1,
    max_increment_m: float = 0.003,
) -> tuple[SharedWendlandFieldV16, dict[str, Any]]:
    """Fit one shared compact RBF field from displacement observations."""

    points = _as_points(points, "constraint points")
    displacements = _as_points(displacements, "constraint displacements")
    if points.shape != displacements.shape:
        raise ValueError("constraint point and displacement shapes differ")
    labels = np.zeros(len(points), dtype="U1") if labels is None else np.asarray(labels).astype(str)
    labels = labels.reshape(-1)
    if len(labels) != len(points):
        raise ValueError("constraint labels length differs")
    radius = float(support_radius_m)
    if not 0.04 <= radius <= 0.06:
        raise ValueError("support_radius_m must be in [0.04, 0.06]")
    increment = float(max_increment_m)
    if not 0 < increment <= 0.003:
        raise ValueError("max_increment_m must be in (0, 0.003]")
    if not len(points):
        field = SharedWendlandFieldV16(
            np.empty((0, 3)), np.empty((0, 3)), radius, regularization
        )
        return field, {"raw_constraints": 0, "clustered_constraints": 0, "max_field_mm": 0.0}
    cell = radius / 4.0 if cluster_cell_m is None else float(cluster_cell_m)
    centers, targets, center_labels = _cluster_constraints(
        points, displacements, labels, cell_m=cell, max_centers=max_centers
    )
    del center_labels  # labels are retained in the caller's constraint summary.
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    kernel = SharedWendlandFieldV16._kernel(pairwise, radius)
    matrix = kernel + float(regularization) * np.eye(len(centers), dtype=np.float64)
    try:
        coefficients = np.linalg.solve(matrix, targets)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(matrix, targets, rcond=1.0e-10)[0]
    field = SharedWendlandFieldV16(centers, coefficients, radius, regularization)
    at_centers = field.evaluate(centers)
    magnitude = np.linalg.norm(at_centers, axis=1)
    peak = float(magnitude.max()) if len(magnitude) else 0.0
    if peak > increment:
        scale = increment / peak
        field = SharedWendlandFieldV16(centers, coefficients * scale, radius, regularization)
    final_peak = float(np.linalg.norm(field.evaluate(centers), axis=1).max()) if len(centers) else 0.0
    return field, {
        "raw_constraints": int(len(points)),
        "clustered_constraints": int(len(centers)),
        "support_radius_mm": radius * 1000.0,
        "cluster_cell_mm": cell * 1000.0,
        "max_field_at_centers_mm": final_peak * 1000.0,
        "target_displacement_max_mm": float(np.linalg.norm(targets, axis=1).max() * 1000.0),
        "max_increment_mm": increment * 1000.0,
    }


def generate_pair_constraints_v16(
    rest_vertices: np.ndarray,
    asset: Any,
    organ_name: str,
    bone_name: str,
    *,
    clearance_m: float = 0.0005,
    contact_band_m: float = 0.0001,
    max_increment_m: float = 0.003,
) -> PairConstraintsV16:
    """Build the requested two-way signed samples for one closed pair.

    The first direction moves organ samples away from the bone.  The second
    direction moves the closest organ boundary point along the interior ray
    from that boundary point to a bone sample, passing the sample by the
    requested clearance.  Both observations are then pooled with all other
    pairs before the one shared field is fitted.
    """

    rest = _as_points(rest_vertices, "target rest")
    organ_start, organ_stop, organ_faces = _mesh_info(asset, organ_name)
    bone_start, bone_stop, bone_faces = _mesh_info(asset, bone_name)
    _assert_closed_oriented(organ_faces, organ_name)
    _assert_closed_oriented(bone_faces, bone_name)
    organ = rest[organ_start:organ_stop]
    bone = rest[bone_start:bone_stop]
    organ_samples = _surface_samples(organ, organ_faces)
    bone_samples = _surface_samples(bone, bone_faces)
    organ_inside, organ_depth, organ_closest_bone, _ = _query_closed_surface(
        organ_samples, bone, bone_faces
    )
    bone_inside, bone_depth, bone_closest_organ, _ = _query_closed_surface(
        bone_samples, organ, organ_faces
    )
    max_delta = float(max_increment_m)
    if not 0 < max_delta <= 0.003:
        raise ValueError("max_increment_m must be in (0, 0.003]")
    contact = float(contact_band_m)
    clearance = float(clearance_m)
    if contact < 0 or clearance < 0:
        raise ValueError("contact band and clearance must be non-negative")

    # Organ points inside bone: q -> q + away_from_bone * (depth + clearance).
    # ``organ_samples - closest_bone`` points from the bone surface into the
    # bone.  The correction must use its opposite so the organ boundary moves
    # out of the bone.  Keeping this explicit avoids silently reversing the
    # signed containment direction when changing the sampling code.
    first = organ_inside & (organ_depth > contact)
    away, away_norm = _unit_vectors(organ_closest_bone - organ_samples)
    first &= away_norm > 1.0e-10
    first_points = organ_samples[first]
    first_disp = away[first] * np.minimum(organ_depth[first] + clearance, max_delta)[:, None]

    # Bone points inside organ: closest organ surface -> along the interior ray
    # until it is just beyond the bone sample.  This is a boundary observation,
    # so its point is the closest organ surface, not the bone sample itself.
    second = bone_inside & (bone_depth > contact)
    inward, inward_norm = _unit_vectors(bone_samples - bone_closest_organ)
    second &= inward_norm > 1.0e-10
    second_points = bone_closest_organ[second]
    second_disp = inward[second] * np.minimum(bone_depth[second] + clearance, max_delta)[:, None]

    points = np.vstack((first_points, second_points))
    displacements = np.vstack((first_disp, second_disp))
    labels = np.concatenate(
        (
            np.full(len(first_points), "organ_to_bone", dtype="U20"),
            np.full(len(second_points), "bone_to_organ", dtype="U20"),
        )
    )
    summary = {
        "organ": str(organ_name),
        "bone": str(bone_name),
        "organ_surface_samples": int(len(organ_samples)),
        "bone_surface_samples": int(len(bone_samples)),
        "organ_inside_bone_count": int(first.sum()),
        "bone_inside_organ_count": int(second.sum()),
        "organ_inside_bone_max_depth_mm": float(organ_depth[organ_inside].max() * 1000.0)
        if np.any(organ_inside)
        else 0.0,
        "bone_inside_organ_max_depth_mm": float(bone_depth[bone_inside].max() * 1000.0)
        if np.any(bone_inside)
        else 0.0,
        "constraint_count": int(len(points)),
        "clearance_mm": clearance * 1000.0,
        "contact_band_mm": contact * 1000.0,
    }
    return PairConstraintsV16(points, displacements, labels, summary)


def _bone_mask(asset: Any, count: int) -> np.ndarray:
    mask = np.zeros(count, dtype=bool)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = np.asarray(asset.source_tissues).astype(str)
    for (start, stop), tissue in zip(ranges.tolist(), tissues.tolist()):
        if tissue.strip().lower() == "bone":
            mask[int(start) : int(stop)] = True
    return mask


def apply_shared_field_v16(
    rest_vertices: np.ndarray,
    asset: Any,
    field: SharedWendlandFieldV16,
    *,
    path_lengths_m: np.ndarray | None = None,
    max_total_m: float = 0.010,
    max_increment_m: float = 0.003,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one field to all non-bone vertices with per-step/total caps."""

    rest = _as_points(rest_vertices, "target rest")
    bone = _bone_mask(asset, len(rest))
    delta = np.zeros_like(rest)
    nonbone = ~bone
    delta[nonbone] = field.evaluate(rest[nonbone])
    norms = np.linalg.norm(delta, axis=1)
    step = float(max_increment_m)
    total = float(max_total_m)
    if not 0 < step <= 0.003 or not 0 < total <= 0.010:
        raise ValueError("field caps must satisfy step <= 3 mm and total <= 10 mm")
    active = norms > step
    delta[active] *= (step / norms[active])[:, None]
    path = np.zeros(len(rest), dtype=np.float64) if path_lengths_m is None else np.asarray(path_lengths_m, dtype=np.float64).copy()
    if path.shape != (len(rest),) or not np.isfinite(path).all():
        raise ValueError("path_lengths_m must have one finite value per vertex")
    proposed = path + np.linalg.norm(delta, axis=1)
    over = proposed > total
    delta[over] *= ((total - path[over]).clip(min=0.0) / np.maximum(np.linalg.norm(delta[over], axis=1), 1.0e-12))[:, None]
    delta[bone] = 0.0
    new = rest + delta
    new[bone] = rest[bone]
    new_path = path + np.linalg.norm(delta, axis=1)
    if np.max(np.linalg.norm(delta[nonbone], axis=1), initial=0.0) > step + 1.0e-10:
        raise AssertionError("per-increment displacement cap was violated")
    if np.max(new_path, initial=0.0) > total + 1.0e-10:
        raise AssertionError("total displacement cap was violated")
    return new, delta, new_path


def pair_collision_metrics_v16(
    vertices: np.ndarray,
    asset: Any,
    pairs: Sequence[tuple[str, str]] = DEFAULT_PAIRS,
    *,
    contact_band_m: float = 0.0001,
) -> dict[str, Any]:
    """Measure two-way closed-surface containment for the selected pairs."""

    rest = _as_points(vertices, "collision vertices")
    result: dict[str, Any] = {"pairs": {}, "triangle_intersections_tested": False}
    for organ_name, bone_name in pairs:
        os, oe, of = _mesh_info(asset, organ_name)
        bs, be, bf = _mesh_info(asset, bone_name)
        _assert_closed_oriented(of, organ_name)
        _assert_closed_oriented(bf, bone_name)
        organ, bone = rest[os:oe], rest[bs:be]
        organ_samples = _surface_samples(organ, of)
        bone_samples = _surface_samples(bone, bf)
        org_inside, org_depth, _org_closest, _ = _query_closed_surface(organ_samples, bone, bf)
        bone_inside, bone_depth, _bone_closest, _ = _query_closed_surface(bone_samples, organ, of)
        org_valid = org_inside & (org_depth > contact_band_m)
        bone_valid = bone_inside & (bone_depth > contact_band_m)
        row = {
            "organ": str(organ_name),
            "bone": str(bone_name),
            "organ_inside_bone_count": int(org_valid.sum()),
            "bone_inside_organ_count": int(bone_valid.sum()),
            "organ_inside_bone_max_depth_mm": float(org_depth[org_valid].max() * 1000.0)
            if np.any(org_valid)
            else 0.0,
            "bone_inside_organ_max_depth_mm": float(bone_depth[bone_valid].max() * 1000.0)
            if np.any(bone_valid)
            else 0.0,
            "organ_inside_bone_p95_depth_mm": float(np.quantile(org_depth[org_valid], 0.95) * 1000.0)
            if np.any(org_valid)
            else 0.0,
            "bone_inside_organ_p95_depth_mm": float(np.quantile(bone_depth[bone_valid], 0.95) * 1000.0)
            if np.any(bone_valid)
            else 0.0,
            "closed_surface_signed_sampling": True,
            "contact_band_mm": contact_band_m * 1000.0,
        }
        result["pairs"][f"{organ_name}__{bone_name}"] = row
    return result


def _edge_metrics(before: np.ndarray, after: np.ndarray, asset: Any, bone: np.ndarray) -> dict[str, Any]:
    from .soft_constraints import unique_mesh_edges

    faces = np.asarray(asset.faces, dtype=np.int64)
    edges = unique_mesh_edges(faces)
    valid = (~bone[edges].any(axis=1))
    edges = edges[valid]
    old = np.linalg.norm(before[edges[:, 1]] - before[edges[:, 0]], axis=1)
    new = np.linalg.norm(after[edges[:, 1]] - after[edges[:, 0]], axis=1)
    valid = old > 1.0e-10
    old, new = old[valid], new[valid]
    if not len(old):
        return {"edge_count": 0, "ratio_min": 1.0, "ratio_max": 1.0, "symmetric_max_pct": 0.0}
    ratio = new / old
    symmetric = np.maximum(ratio, 1.0 / np.maximum(ratio, 1.0e-12)) - 1.0
    return {
        "edge_count": int(len(ratio)),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "ratio_p01": float(np.quantile(ratio, 0.01)),
        "ratio_p99": float(np.quantile(ratio, 0.99)),
        "symmetric_max_pct": float(symmetric.max() * 100.0),
        "symmetric_p99_pct": float(np.quantile(symmetric, 0.99) * 100.0),
        "symmetric_over_5pct": int(np.count_nonzero(symmetric > 0.05)),
        "symmetric_over_10pct": int(np.count_nonzero(symmetric > 0.10)),
    }


def skin_outside_metrics_v16(vertices: np.ndarray, asset: Any, skin_vertices: np.ndarray, skin_faces: np.ndarray) -> dict[str, Any]:
    """Return positive signed distance outside the supplied closed skin."""

    import igl

    points = _as_points(vertices, "skin query vertices")
    skin = _as_points(skin_vertices, "skin vertices")
    faces = np.asarray(skin_faces, dtype=np.int32).reshape(-1, 3)
    winding = np.asarray(igl.winding_number(skin, faces, points), dtype=np.float64).reshape(-1)
    squared, _face, _closest = igl.point_mesh_squared_distance(points, skin, faces)
    distance = np.sqrt(np.maximum(np.asarray(squared, dtype=np.float64), 0.0))
    signed = np.where(np.abs(winding) >= 0.5, -distance, distance)
    bone = _bone_mask(asset, len(points))
    query_ids = np.flatnonzero(~bone)
    outside = np.maximum(signed[query_ids], 0.0)
    result: dict[str, Any] = {
        "nonbone_vertex_count": int(len(query_ids)),
        "max_outside_mm": float(outside.max(initial=0.0) * 1000.0),
        "p99_outside_mm": float(np.quantile(outside, 0.99) * 1000.0) if len(outside) else 0.0,
        "outside_over_1mm": int(np.count_nonzero(outside > 0.001)),
    }
    names = [str(x) for x in np.asarray(asset.source_mesh_names).tolist()]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    selected = {"Diaphragm", "Liver", "Heart", "Artery", "Vein", "UNCUT_Digestive_Tract"}
    for name, (start, stop) in zip(names, ranges.tolist()):
        if name not in selected and "Nerve" not in name:
            continue
        chunk = np.maximum(signed[int(start) : int(stop)], 0.0)
        result.setdefault("meshes", {})[name] = {
            "max_outside_mm": float(chunk.max(initial=0.0) * 1000.0),
            "outside_over_1mm": int(np.count_nonzero(chunk > 0.001)),
        }
    return result


def _round_measurements(before: np.ndarray, after: np.ndarray, asset: Any, pairs: Sequence[tuple[str, str]], skin_vertices: np.ndarray, skin_faces: np.ndarray) -> dict[str, Any]:
    bone = _bone_mask(asset, len(before))
    displacement = after - before
    return {
        "pair_collision": pair_collision_metrics_v16(after, asset, pairs),
        "skin_outside": skin_outside_metrics_v16(after, asset, skin_vertices, skin_faces),
        "edge_strain": _edge_metrics(before, after, asset, bone),
        "max_nonbone_step_mm": float(np.linalg.norm(displacement[~bone], axis=1).max(initial=0.0) * 1000.0),
        "bone_vertices_bit_exact": bool(np.array_equal(before[bone], after[bone])),
    }


def _receiver_counts(asset: Any, delta: np.ndarray) -> dict[str, int]:
    """Count nonzero receivers by authored tissue class for audit output."""

    active = np.linalg.norm(np.asarray(delta, dtype=np.float64), axis=1) > 1.0e-12
    counts: dict[str, int] = {}
    for (start, stop), tissue in zip(
        np.asarray(asset.source_vertex_ranges, dtype=np.int64).tolist(),
        np.asarray(asset.source_tissues).astype(str).tolist(),
    ):
        key = str(tissue)
        counts[key] = counts.get(key, 0) + int(active[int(start) : int(stop)].sum())
    return counts


def _collision_signature(metrics: Mapping[str, Any]) -> tuple[dict[str, tuple[float, float]], float, float]:
    """Extract per-direction maxima, total energy, and global maximum."""

    directions: dict[str, tuple[float, float]] = {}
    total = 0.0
    maximum = 0.0
    for key, row in metrics["pairs"].items():
        organ = float(row["organ_inside_bone_max_depth_mm"])
        bone = float(row["bone_inside_organ_max_depth_mm"])
        directions[str(key)] = (organ, bone)
        total += organ + bone
        maximum = max(maximum, organ, bone)
    return directions, total, maximum


def _collision_nonincreasing(before: Mapping[str, Any], after: Mapping[str, Any], *, tolerance_mm: float = 1.0e-6) -> tuple[bool, str]:
    """Reject a step when pairwise two-way maxima or total energy rise.

    The two directional depths are retained in the report.  A strict
    per-direction gate would reject a useful outward move whenever the
    opposite sampled surface gains a sub-millimetre numerical contact; the
    acceptance gate therefore uses the pair's two-way maximum and its total
    energy, which is the requested conservative line search criterion.
    """

    before_directions, before_total, before_max = _collision_signature(before)
    after_directions, after_total, after_max = _collision_signature(after)
    for key, old in before_directions.items():
        new = after_directions.get(key)
        if new is None:
            return False, f"missing pair in candidate: {key}"
        old_max = max(old)
        new_max = max(new)
        if new_max > old_max + tolerance_mm:
            return False, f"two-way maximum regressed: {key} {old_max:.6f} -> {new_max:.6f} mm"
    if after_max > before_max + tolerance_mm:
        return False, f"global maximum regressed: {before_max:.6f} -> {after_max:.6f} mm"
    if after_total > before_total + tolerance_mm:
        return False, f"total penetration energy regressed: {before_total:.6f} -> {after_total:.6f} mm"
    return True, "accepted"


def bake_target_rest_v16(base: Any, target_rest: np.ndarray, field_metadata: Mapping[str, Any]) -> Any:
    """Return a V14 compiled object with the fixed field baked into rest."""

    target = _as_points(target_rest, "baked target rest")
    if target.shape != np.asarray(base.target_rest).shape:
        raise ValueError("baked target rest shape differs from compiled subject")
    provenance = dict(base.provenance)
    provenance["rest_field_application_count"] = int(provenance.get("rest_field_application_count", 0)) + 1
    provenance["v16_shared_rest_field"] = dict(field_metadata)
    return replace(base, target_rest=target, provenance=provenance)


def run_shared_rest_fit_v16(
    base: Any,
    *,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    pairs: Sequence[tuple[str, str]] = DEFAULT_PAIRS,
    rounds: int = 3,
    support_radius_m: float = 0.05,
    max_increment_m: float = 0.003,
    max_total_m: float = 0.010,
    regularization: float = 0.1,
    field_refiner: Any = None,
) -> dict[str, Any]:
    """Fit up to three T-pose rounds and return the baked candidate/report."""

    if not 1 <= int(rounds) <= 3:
        raise ValueError("rounds must be between 1 and 3")
    current = _as_points(base.target_rest, "compiled target rest").copy()
    initial = current.copy()
    path = np.zeros(len(current), dtype=np.float64)
    asset = base.source_asset
    pairs = tuple((str(a), str(b)) for a, b in pairs)
    before_metrics = {
        "pair_collision": pair_collision_metrics_v16(current, asset, pairs),
        "skin_outside": skin_outside_metrics_v16(current, asset, skin_vertices, skin_faces),
        "edge_strain": _edge_metrics(initial, initial, asset, _bone_mask(asset, len(initial))),
    }
    round_rows: list[dict[str, Any]] = []
    final_field: SharedWendlandFieldV16 | None = None
    for round_index in range(1, int(rounds) + 1):
        constraints = [
            generate_pair_constraints_v16(
                current,
                asset,
                organ_name,
                bone_name,
                max_increment_m=max_increment_m,
            )
            for organ_name, bone_name in pairs
        ]
        if not any(len(x.points) for x in constraints):
            break
        points = np.vstack([x.points for x in constraints if len(x.points)])
        displacements = np.vstack([x.displacements for x in constraints if len(x.points)])
        labels = np.concatenate([x.labels for x in constraints if len(x.points)])
        field, fit_meta = fit_wendland_field_v16(
            points,
            displacements,
            labels,
            support_radius_m=support_radius_m,
            regularization=regularization,
            max_increment_m=max_increment_m,
        )
        guard_check = None
        guard_meta: dict[str, Any] = {}
        if field_refiner is not None:
            field, guard_meta, guard_check = field_refiner(
                initial, current, asset, skin_vertices, skin_faces,
                points, displacements, field)
        current_collision = pair_collision_metrics_v16(current, asset, pairs)
        trial_scale = 1.0
        trial_field = field
        trial_updated = current.copy()
        trial_delta = np.zeros_like(current)
        trial_path = path.copy()
        trial_collision = current_collision
        accepted = False
        rejection_reason = "line search exhausted"
        # A smooth RBF can still overshoot at a sampled boundary.  Backtrack
        # the entire shared field, preserving its one-field/all-material
        # semantics, whenever a pair's two-way maximum or aggregate energy
        # regresses.  No rejected step contributes to cumulative displacement.
        for _attempt in range(9):
            trial_field = SharedWendlandFieldV16(
                field.centers,
                field.coefficients * trial_scale,
                field.support_radius_m,
                field.regularization,
            )
            candidate, candidate_delta, candidate_path = apply_shared_field_v16(
                current,
                asset,
                trial_field,
                path_lengths_m=path,
                max_total_m=max_total_m,
                max_increment_m=max_increment_m,
            )
            candidate_collision = pair_collision_metrics_v16(candidate, asset, pairs)
            ok, reason = _collision_nonincreasing(current_collision, candidate_collision)
            guard_result: dict[str, Any] = {}
            if guard_check is not None:
                guard_ok, guard_result = guard_check(candidate)
                if not guard_ok:
                    ok = False
                    reason = str(guard_result.get('reason', 'neighbour material guard rejected step'))
                elif np.linalg.norm(candidate_delta, axis=1).max(initial=0.0) < 1e-10:
                    ok = False
                    reason = 'no feasible nonzero material step'
            rejection_reason = reason
            if ok:
                trial_updated = candidate
                trial_delta = candidate_delta
                trial_path = candidate_path
                trial_collision = candidate_collision
                accepted = True
                break
            trial_scale *= 0.5
        updated, delta, path = trial_updated, trial_delta, trial_path
        field_meta = {
            "round": round_index,
            **fit_meta,
            "pair_summaries": [x.summary for x in constraints],
            "nonbone_vertices_receiving_field": int(np.count_nonzero(np.linalg.norm(delta, axis=1) > 0)),
            "receiver_counts_by_tissue": _receiver_counts(asset, delta),
            "max_applied_step_mm": float(np.linalg.norm(delta, axis=1).max(initial=0.0) * 1000.0),
            "max_cumulative_path_mm": float(path.max(initial=0.0) * 1000.0),
            "shared_field": True,
            "bone_unchanged": True,
            "line_search_scale": float(trial_scale if accepted else 0.0),
            "accepted": bool(accepted),
            "signed_energy_before_mm": float(_collision_signature(current_collision)[1]),
            "signed_energy_after_mm": float(_collision_signature(trial_collision)[1]),
            "rejection_reason": None if accepted else rejection_reason,
            "neighbor_guard": guard_meta,
            "neighbor_guard_check": guard_result,
        }
        if accepted:
            final_field = trial_field
        row = _round_measurements(current, updated, asset, pairs, skin_vertices, skin_faces)
        row.update({"round": round_index, "field": field_meta})
        round_rows.append(row)
        current = updated
        if not accepted:
            break
    if final_field is None:
        final_field = SharedWendlandFieldV16(np.empty((0, 3)), np.empty((0, 3)), support_radius_m)
    field_metadata = {
        "method": "two_way_closed_surface_signed_samples_wendland_c2_v16",
        "pairs": [list(x) for x in pairs],
        "support_radius_mm": support_radius_m * 1000.0,
        "max_increment_mm": max_increment_m * 1000.0,
        "max_total_mm": max_total_m * 1000.0,
        "fit_pose_data_used": False,
        "same_field_for_all_nonbone": True,
        "bone_geometry_changed": False,
        "weights_bind_topology_changed": False,
        "rounds_completed": len(round_rows),
        "neighbor_guard_enabled": field_refiner is not None,
        "serialization_scope": "last_increment_only; use total baked target_rest displacement for replay",
    }
    baked = bake_target_rest_v16(base, current, field_metadata)
    report = {
        "artifact_kind": "SharedVisceralRestFitV16",
        "publishable": False,
        "anatomical_passed": False,
        "fit_domain": "tpose_only_closed_pair_fit",
        "pairs": [list(x) for x in pairs],
        "before": before_metrics,
        "rounds": round_rows,
        "field": field_metadata,
        "bone_vertices_bit_exact": bool(np.array_equal(initial[_bone_mask(asset, len(initial))], current[_bone_mask(asset, len(initial))])),
        "max_cumulative_displacement_mm": float(np.linalg.norm(current - initial, axis=1).max(initial=0.0) * 1000.0),
        "max_cumulative_path_mm": float(path.max(initial=0.0) * 1000.0),
        "exact_triangle_intersection_tested": False,
        "note": "sampled exploratory candidate; inspect Genesis renders and regression poses before any acceptance decision",
    }
    return {"compiled": baked, "field": final_field, "report": report}


__all__ = [
    "DEFAULT_PAIRS",
    "PairConstraintsV16",
    "SharedWendlandFieldV16",
    "apply_shared_field_v16",
    "bake_target_rest_v16",
    "fit_wendland_field_v16",
    "generate_pair_constraints_v16",
    "pair_collision_metrics_v16",
    "run_shared_rest_fit_v16",
    "skin_outside_metrics_v16",
]
