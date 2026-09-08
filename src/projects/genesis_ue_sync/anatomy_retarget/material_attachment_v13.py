"""Topology-fixed surface attachments for material retargeting.

This module provides the small, pose-time part of the material retargeting
pipeline.  A compile pass attaches every authored soft/vessel point to one or
more nearby bone-mesh components.  The runtime pass evaluates those fixed
triangles and applies the relative surface motion to the already posed
material point::

    out = source_posed_point + (target_attached - source_attached)

The offset is stored in a complete triangle-local orthonormal frame.  The
normal component is intentionally retained; this keeps an authored point on
the inside/outside side of a surface under a rigid pose instead of silently
projecting it onto the triangle.

Compilation may use :mod:`igl` when present, but has a deterministic scipy /
NumPy exact-triangle fallback.  Runtime code does not perform a spatial
search, nearest-point query, or solve.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "material_attachment_v13"
MAX_COMPONENTS = 4
DEFAULT_EPSILON_M = 0.010
_AREA_TOLERANCE = 1.0e-14


def _digest_array(value: np.ndarray) -> str:
    """Return a stable digest including dtype, shape, and bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _as_vertices(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite coordinates")
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one point")
    return np.ascontiguousarray(array)


def _as_faces(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{name} must have shape [F, 3], got {raw.shape}")
    if np.issubdtype(raw.dtype, np.floating):
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            raise ValueError(f"{name} must contain finite integer indices")
    try:
        array = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer indices") from exc
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one triangle")
    return np.ascontiguousarray(array)


def _check_face_indices(faces: np.ndarray, vertex_count: int, *, name: str) -> None:
    if np.any(faces < 0) or np.any(faces >= int(vertex_count)):
        raise ValueError(f"{name} contains an out-of-range vertex index")


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def _validate_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    name: str,
) -> dict[str, Any]:
    _check_face_indices(faces, len(vertices), name=name)
    areas = _triangle_areas(vertices, faces)
    degenerate = areas <= _AREA_TOLERANCE
    report = {
        "face_count": int(len(faces)),
        "degenerate_face_count": int(np.count_nonzero(degenerate)),
        "minimum_triangle_area_m2": float(np.min(areas)),
        "maximum_triangle_area_m2": float(np.max(areas)),
    }
    if np.any(degenerate):
        bad = np.flatnonzero(degenerate)[:8].tolist()
        raise ValueError(
            f"{name} contains {int(np.count_nonzero(degenerate))} degenerate triangle(s); "
            f"indices={bad}, minimum_area_m2={float(np.min(areas)):.3e}"
        )
    return report


def _normalise_component_ids(
    value: Any,
    count: int,
    *,
    name: str,
    default: str = "0",
) -> np.ndarray:
    if value is None:
        return np.full(int(count), str(default), dtype="<U1")
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) != int(count):
        raise ValueError(f"{name} must have shape [{count}], got {raw.shape}")
    labels: list[str] = []
    for item in raw.tolist():
        if isinstance(item, (float, np.floating)) and not np.isfinite(float(item)):
            raise ValueError(f"{name} contains a non-finite component id")
        labels.append(str(item.item() if isinstance(item, np.generic) else item))
    if any(label == "" for label in labels):
        raise ValueError(f"{name} contains an empty component id")
    width = max(1, max(len(label) for label in labels))
    return np.asarray(labels, dtype=f"<U{width}")


def _component_groups(labels: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
    # Sorting makes the fixed map deterministic even when source faces were
    # supplied in a different component order.
    unique = sorted({str(item) for item in labels.tolist()})
    groups = {label: np.flatnonzero(labels == label).astype(np.int64) for label in unique}
    return unique, groups


def _component_centroids(
    vertices: np.ndarray,
    faces: np.ndarray,
    groups: Mapping[str, np.ndarray],
) -> np.ndarray:
    values = []
    for label in sorted(groups):
        tri = vertices[faces[groups[label]]]
        values.append(np.mean(tri, axis=(0, 1)))
    return np.asarray(values, dtype=np.float64).reshape(-1, 3)


def _component_candidates(
    points: np.ndarray,
    labels: list[str],
    centroids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest component labels by centroid, with deterministic ties."""

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centroids)
        distances, indices = tree.query(points, k=min(int(k), len(labels)))
    except Exception:
        diff = points[:, None, :] - centroids[None, :, :]
        all_d2 = np.einsum("pci,pci->pc", diff, diff)
        count = min(int(k), len(labels))
        indices = np.argsort(all_d2, axis=1, kind="stable")[:, :count]
        distances = np.sqrt(np.take_along_axis(all_d2, indices, axis=1))
    if int(k) == 1:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1, 1)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1, 1)
    else:
        indices = np.asarray(indices, dtype=np.int64).reshape(len(points), -1)
        distances = np.asarray(distances, dtype=np.float64).reshape(len(points), -1)
    return np.asarray([[labels[int(index)] for index in row] for row in indices], dtype=str), distances**2


def _barycentric_from_points(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute barycentrics for points on non-degenerate triangles."""

    a = triangles[:, 0]
    v0 = triangles[:, 1] - a
    v1 = triangles[:, 2] - a
    v2 = points - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denominator = d00 * d11 - d01 * d01
    if np.any(denominator <= _AREA_TOLERANCE):
        raise ValueError("cannot compute barycentrics on a degenerate triangle")
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    result = np.stack((1.0 - v - w, v, w), axis=1)
    # Closest points may be on an edge or vertex.  Small negative roundoff is
    # clipped, while a true out-of-range result indicates a fallback bug.
    result[np.abs(result) < 1.0e-12] = 0.0
    result = np.clip(result, 0.0, 1.0)
    result /= np.maximum(np.sum(result, axis=1, keepdims=True), 1.0e-15)
    return result


def _closest_point_triangle(point: np.ndarray, triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest point and barycentrics for one non-degenerate 3D triangle."""

    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(ab @ ap)
    d2 = float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    bp = point - b
    d3 = float(ab @ bp)
    d4 = float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, 1.0e-30)
        return a + v * ab, np.asarray((1.0 - v, v, 0.0), dtype=np.float64)
    cp = point - c
    d5 = float(ab @ cp)
    d6 = float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, 1.0e-30)
        return a + w * ac, np.asarray((1.0 - w, 0.0, w), dtype=np.float64)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-30)
        return b + w * (c - b), np.asarray((0.0, 1.0 - w, w), dtype=np.float64)
    denominator = max(va + vb + vc, 1.0e-30)
    v = vb / denominator
    w = vc / denominator
    return a + ab * v + ac * w, np.asarray((1.0 - v - w, v, w), dtype=np.float64)


def _closest_points_fallback(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    closest = np.empty_like(points)
    barycentric = np.empty((len(points), 3), dtype=np.float64)
    face_local = np.empty(len(points), dtype=np.int64)
    squared = np.empty(len(points), dtype=np.float64)
    triangles = vertices[faces]
    for row, point in enumerate(points):
        best_d2 = float("inf")
        best_face = 0
        best_point = triangles[0, 0].copy()
        best_bary = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        for index, triangle in enumerate(triangles):
            candidate, candidate_bary = _closest_point_triangle(point, triangle)
            distance = float(np.sum(np.square(point - candidate)))
            if distance < best_d2:
                best_d2 = distance
                best_face = index
                best_point = candidate
                best_bary = candidate_bary
        closest[row] = best_point
        barycentric[row] = best_bary
        face_local[row] = best_face
        squared[row] = best_d2
    return squared, face_local, closest, barycentric


def _closest_points_exact(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact point-to-triangle distance, preferring libigl when available."""

    try:
        import igl  # type: ignore

        squared, face_local, closest = igl.point_mesh_squared_distance(points, vertices, faces)
        squared = np.asarray(squared, dtype=np.float64).reshape(-1)
        face_local = np.asarray(face_local, dtype=np.int64).reshape(-1)
        closest = np.asarray(closest, dtype=np.float64).reshape(-1, 3)
        if (
            len(squared) != len(points)
            or len(face_local) != len(points)
            or len(closest) != len(points)
            or np.any(face_local < 0)
            or np.any(face_local >= len(faces))
            or not np.all(np.isfinite(squared))
            or not np.all(np.isfinite(closest))
        ):
            raise ValueError("libigl returned an invalid point-mesh distance result")
        barycentric = _barycentric_from_points(closest, vertices[faces[face_local]])
        return squared, face_local, closest, barycentric
    except Exception:
        return _closest_points_fallback(points, vertices, faces)


def _closest_by_component(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    groups: Mapping[str, np.ndarray],
    candidate_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find exact nearest triangles for each fixed candidate component."""

    count, width = candidate_labels.shape
    squared = np.empty((count, width), dtype=np.float64)
    face_indices = np.empty((count, width), dtype=np.int64)
    closest = np.empty((count, width, 3), dtype=np.float64)
    barycentric = np.empty((count, width, 3), dtype=np.float64)
    for label in sorted({str(item) for item in candidate_labels.reshape(-1).tolist()}):
        rows, slots = np.where(candidate_labels == label)
        if len(rows) == 0:
            continue
        component_faces = np.asarray(groups[label], dtype=np.int64)
        local_squared, local_faces, local_closest, local_bary = _closest_points_exact(
            points[rows], vertices, faces[component_faces]
        )
        squared[rows, slots] = local_squared
        face_indices[rows, slots] = component_faces[local_faces]
        closest[rows, slots] = local_closest
        barycentric[rows, slots] = local_bary
    return squared, face_indices, closest, barycentric


def _triangle_frames(triangles: np.ndarray, *, name: str) -> np.ndarray:
    """Build [e1, e2, normal] as frame columns for each triangle."""

    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    edge1_norm = np.linalg.norm(edge1, axis=1)
    normal_raw = np.cross(edge1, edge2)
    normal_norm = np.linalg.norm(normal_raw, axis=1)
    bad = (edge1_norm <= _AREA_TOLERANCE) | (normal_norm <= 2.0 * _AREA_TOLERANCE)
    if np.any(bad):
        raise ValueError(f"{name} contains degenerate runtime attachment triangle(s)")
    e1 = edge1 / edge1_norm[:, None]
    normal = normal_raw / normal_norm[:, None]
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    return np.stack((e1, e2, normal), axis=2)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MaterialAttachmentMapV13:
    """Compiled fixed-triangle material attachment map.

    Arrays use shape ``[point, candidate, ...]``.  ``candidate`` is at most
    four and is selected once from rest-pose component centroids.  Runtime
    evaluation only indexes ``source_faces``/``target_faces`` and evaluates
    the stored barycentrics and local offsets.
    """

    source_faces: np.ndarray
    target_faces: np.ndarray
    source_face_indices: np.ndarray
    target_face_indices: np.ndarray
    source_barycentric: np.ndarray
    target_barycentric: np.ndarray
    source_offset_local: np.ndarray
    target_offset_local: np.ndarray
    component_weights: np.ndarray
    component_ids: np.ndarray
    source_distances_m: np.ndarray
    source_points_rest: np.ndarray
    source_surface_digest: str
    target_surface_digest: str
    source_points_digest: str
    epsilon_m: float
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        arrays = (
            "source_faces",
            "target_faces",
            "source_face_indices",
            "target_face_indices",
            "source_barycentric",
            "target_barycentric",
            "source_offset_local",
            "target_offset_local",
            "component_weights",
            "component_ids",
            "source_distances_m",
            "source_points_rest",
        )
        for name in arrays:
            value = np.asarray(getattr(self, name))
            if name != "component_ids" and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "report", dict(self.report))
        self._validate_internal()

    @property
    def point_count(self) -> int:
        return int(self.source_points_rest.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.source_face_indices.shape[1])

    @property
    def source_rest_vertices_digest(self) -> str:
        return str(self.source_surface_digest)

    @property
    def target_rest_vertices_digest(self) -> str:
        return str(self.target_surface_digest)

    def _validate_internal(self) -> None:
        if self.source_faces.ndim != 2 or self.source_faces.shape[1] != 3:
            raise ValueError("source_faces must have shape [F, 3]")
        if self.target_faces.ndim != 2 or self.target_faces.shape[1] != 3:
            raise ValueError("target_faces must have shape [F, 3]")
        if self.source_points_rest.ndim != 2 or self.source_points_rest.shape[1] != 3:
            raise ValueError("source_points_rest must have shape [P, 3]")
        count = len(self.source_points_rest)
        face_shape = (count, self.candidate_count)
        if self.source_face_indices.shape != face_shape or self.target_face_indices.shape != face_shape:
            raise ValueError("source/target face-index shape mismatch")
        if self.component_weights.shape != face_shape or self.source_distances_m.shape != face_shape:
            raise ValueError("component weight/distance shape mismatch")
        if self.component_ids.shape != face_shape:
            raise ValueError("component_ids shape mismatch")
        for name in (
            "source_barycentric",
            "target_barycentric",
            "source_offset_local",
            "target_offset_local",
        ):
            value = getattr(self, name)
            if value.shape != (count, self.candidate_count, 3):
                raise ValueError(f"{name} must have shape [P, K, 3]")
        if np.any(self.source_face_indices < 0) or np.any(self.source_face_indices >= len(self.source_faces)):
            raise ValueError("source face attachment index out of range")
        if np.any(self.target_face_indices < 0) or np.any(self.target_face_indices >= len(self.target_faces)):
            raise ValueError("target face attachment index out of range")
        if np.any(self.component_weights < 0.0) or np.any(self.source_distances_m < 0.0):
            raise ValueError("component weights and distances must be non-negative")
        if not np.allclose(np.sum(self.component_weights, axis=1), 1.0, atol=2.0e-6):
            raise ValueError("component weights must sum to one")
        if not np.isfinite(float(self.epsilon_m)) or float(self.epsilon_m) <= 0.0:
            raise ValueError("epsilon_m must be positive and finite")
        if len(self.component_ids) and self.component_ids.dtype.kind not in "OUS":
            raise ValueError("component_ids must be string-like")

    def _runtime_attached_points(
        self,
        source_surface_vertices_posed: np.ndarray,
        target_surface_vertices_posed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_vertices = _as_vertices(source_surface_vertices_posed, name="source_surface_vertices_posed")
        target_vertices = _as_vertices(target_surface_vertices_posed, name="target_surface_vertices_posed")
        if np.max(self.source_faces, initial=-1) >= len(source_vertices):
            raise ValueError("source posed surface has fewer vertices than its compiled topology")
        if np.max(self.target_faces, initial=-1) >= len(target_vertices):
            raise ValueError("target posed surface has fewer vertices than its compiled topology")
        source_triangles = source_vertices[self.source_faces[self.source_face_indices]]
        target_triangles = target_vertices[self.target_faces[self.target_face_indices]]
        source_frames = _triangle_frames(source_triangles.reshape(-1, 3, 3), name="source")
        target_frames = _triangle_frames(target_triangles.reshape(-1, 3, 3), name="target")
        source_frames = source_frames.reshape(self.point_count, self.candidate_count, 3, 3)
        target_frames = target_frames.reshape(self.point_count, self.candidate_count, 3, 3)
        source_surface = np.einsum(
            "pkj,pkji->pki", self.source_barycentric, source_triangles
        )
        target_surface = np.einsum(
            "pkj,pkji->pki", self.target_barycentric, target_triangles
        )
        source_attached = source_surface + np.einsum(
            "pkij,pkj->pki", source_frames, self.source_offset_local
        )
        target_attached = target_surface + np.einsum(
            "pkij,pkj->pki", target_frames, self.target_offset_local
        )
        if not np.all(np.isfinite(source_attached)) or not np.all(np.isfinite(target_attached)):
            raise ValueError("runtime attachment reconstruction produced non-finite coordinates")
        return source_attached, target_attached

    def transport(
        self,
        source_points_posed: np.ndarray,
        source_surface_vertices_posed: np.ndarray,
        target_surface_vertices_posed: np.ndarray,
        *,
        return_audit: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Transport posed soft/vessel points through fixed surface attachments."""

        points = _as_vertices(source_points_posed, name="source_points_posed")
        if points.shape[0] != self.point_count:
            raise ValueError(
                f"source_points_posed has {len(points)} points; map expects {self.point_count}"
            )
        # This short circuit gives an exact identity for an identical posed
        # surface and identical topology.  It also avoids needless floating
        # point subtraction for the common source==target identity check.
        source_surface = _as_vertices(source_surface_vertices_posed, name="source_surface_vertices_posed")
        target_surface = _as_vertices(target_surface_vertices_posed, name="target_surface_vertices_posed")
        identity = bool(
            source_surface.shape == target_surface.shape
            and np.array_equal(source_surface, target_surface)
            and np.array_equal(self.source_faces, self.target_faces)
            and np.array_equal(self.source_face_indices, self.target_face_indices)
            and np.array_equal(self.source_barycentric, self.target_barycentric)
            and np.array_equal(self.source_offset_local, self.target_offset_local)
        )
        if identity:
            result = np.array(source_points_posed, copy=True)
            audit = {
                "identity_short_circuit": True,
                "rest_identity": False,
                "finite": True,
                "containment_status": "not_evaluated",
                "containment_verified": False,
                "candidate_count": self.candidate_count,
            }
            return (result, audit) if return_audit else result
        source_attached, target_attached = self._runtime_attached_points(source_surface, target_surface)
        displacement = target_attached - source_attached
        # Both rest reconstructions represent the same authored point.  The
        # two independent frame evaluations can differ by a few ulps, so
        # collapse only numerical zero; real pose motion remains untouched.
        residual_norm = np.linalg.norm(displacement, axis=2)
        residual_identity = bool(np.max(residual_norm, initial=0.0) <= 1.0e-12)
        if residual_identity:
            displacement = np.zeros_like(displacement)
        result = points + np.sum(self.component_weights[..., None] * displacement, axis=1)
        if not np.all(np.isfinite(result)):
            raise ValueError("material transport produced non-finite coordinates")
        audit = {
            "identity_short_circuit": False,
            "rest_identity": residual_identity,
            "finite": True,
            "containment_status": "not_evaluated",
            "containment_verified": False,
            "candidate_count": self.candidate_count,
            "max_source_attachment_motion_m": float(np.max(np.linalg.norm(source_attached, axis=2))),
            "max_target_attachment_motion_m": float(np.max(np.linalg.norm(target_attached, axis=2))),
            "max_transport_displacement_m": float(np.max(np.linalg.norm(result - points, axis=1))) if len(result) else 0.0,
        }
        return (result, audit) if return_audit else result

    def transport_points(self, *args: Any, **kwargs: Any) -> Any:
        """Alias kept for callers that use the noun form."""

        return self.transport(*args, **kwargs)

    def save(self, path: Path | str) -> Path:
        """Save the map using a pickle-free compressed NPZ payload."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        report_json = json.dumps(dict(self.report), sort_keys=True, separators=(",", ":"))
        np.savez_compressed(
            out,
            schema_version=np.asarray([SCHEMA_VERSION]),
            source_faces=self.source_faces.astype(np.int32),
            target_faces=self.target_faces.astype(np.int32),
            source_face_indices=self.source_face_indices.astype(np.int32),
            target_face_indices=self.target_face_indices.astype(np.int32),
            source_barycentric=self.source_barycentric.astype(np.float32),
            target_barycentric=self.target_barycentric.astype(np.float32),
            source_offset_local=self.source_offset_local.astype(np.float32),
            target_offset_local=self.target_offset_local.astype(np.float32),
            component_weights=self.component_weights.astype(np.float32),
            component_ids=np.asarray(self.component_ids, dtype=str),
            source_distances_m=self.source_distances_m.astype(np.float32),
            # Keep authored rest points in float64 so their content digest
            # remains valid after a round trip.  Other runtime coefficients
            # are compact float32 payloads and are checked by shape/invariants
            # on load.
            source_points_rest=self.source_points_rest.astype(np.float64),
            source_surface_digest=np.asarray([self.source_surface_digest]),
            target_surface_digest=np.asarray([self.target_surface_digest]),
            source_points_digest=np.asarray([self.source_points_digest]),
            epsilon_m=np.asarray([self.epsilon_m], dtype=np.float64),
            report_json=np.asarray([report_json]),
        )
        return out

    save_npz = save

    @classmethod
    def load(cls, path: Path | str) -> "MaterialAttachmentMapV13":
        """Load and validate a pickle-free attachment map."""

        with np.load(Path(path), allow_pickle=False) as data:
            required = {
                "schema_version",
                "source_faces",
                "target_faces",
                "source_face_indices",
                "target_face_indices",
                "source_barycentric",
                "target_barycentric",
                "source_offset_local",
                "target_offset_local",
                "component_weights",
                "component_ids",
                "source_distances_m",
                "source_points_rest",
                "source_surface_digest",
                "target_surface_digest",
                "source_points_digest",
                "epsilon_m",
                "report_json",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"attachment map is missing fields: {missing}")
            schema = str(np.asarray(data["schema_version"]).reshape(-1)[0])
            if schema != SCHEMA_VERSION:
                raise ValueError(f"unsupported attachment map schema {schema!r}")
            try:
                report = json.loads(str(np.asarray(data["report_json"]).reshape(-1)[0]))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("attachment map report_json is invalid") from exc
            result = cls(
                source_faces=np.asarray(data["source_faces"], dtype=np.int64),
                target_faces=np.asarray(data["target_faces"], dtype=np.int64),
                source_face_indices=np.asarray(data["source_face_indices"], dtype=np.int64),
                target_face_indices=np.asarray(data["target_face_indices"], dtype=np.int64),
                source_barycentric=np.asarray(data["source_barycentric"], dtype=np.float64),
                target_barycentric=np.asarray(data["target_barycentric"], dtype=np.float64),
                source_offset_local=np.asarray(data["source_offset_local"], dtype=np.float64),
                target_offset_local=np.asarray(data["target_offset_local"], dtype=np.float64),
                component_weights=np.asarray(data["component_weights"], dtype=np.float64),
                component_ids=np.asarray(data["component_ids"]).astype(str),
                source_distances_m=np.asarray(data["source_distances_m"], dtype=np.float64),
                source_points_rest=np.asarray(data["source_points_rest"], dtype=np.float64),
                source_surface_digest=str(np.asarray(data["source_surface_digest"]).reshape(-1)[0]),
                target_surface_digest=str(np.asarray(data["target_surface_digest"]).reshape(-1)[0]),
                source_points_digest=str(np.asarray(data["source_points_digest"]).reshape(-1)[0]),
                epsilon_m=float(np.asarray(data["epsilon_m"]).reshape(-1)[0]),
                report=report,
            )
        if result.source_points_digest != _digest_array(result.source_points_rest):
            raise ValueError("attachment map source_points_rest digest mismatch")
        return result

    load_npz = load


def compile_material_attachment_map_v13(
    source_surface_vertices: np.ndarray,
    source_surface_faces: np.ndarray,
    target_surface_vertices: np.ndarray,
    target_surface_faces: np.ndarray,
    source_points_rest: np.ndarray,
    *,
    source_face_component_ids: np.ndarray | None = None,
    target_face_component_ids: np.ndarray | None = None,
    component_map: Mapping[Any, Any] | None = None,
    max_components: int = MAX_COMPONENTS,
    epsilon_m: float = DEFAULT_EPSILON_M,
    return_report: bool = False,
) -> MaterialAttachmentMapV13 | tuple[MaterialAttachmentMapV13, dict[str, Any]]:
    """Compile fixed triangle attachments for soft/vessel rest points.

    ``source_face_component_ids`` labels bone-mesh components.  Up to four
    nearest components by centroid are retained per point; each retained
    component then receives an exact nearest triangle.  Target labels are
    matched by value unless ``component_map`` provides an explicit mapping.
    When target labels are omitted, target faces are treated as one component
    (or inherit source labels when face topologies have equal length).
    """

    source_vertices = _as_vertices(source_surface_vertices, name="source_surface_vertices")
    target_vertices = _as_vertices(target_surface_vertices, name="target_surface_vertices")
    source_faces_array = _as_faces(source_surface_faces, name="source_surface_faces")
    target_faces_array = _as_faces(target_surface_faces, name="target_surface_faces")
    source_points = _as_vertices(source_points_rest, name="source_points_rest")
    source_report = _validate_triangles(source_vertices, source_faces_array, name="source_surface_faces")
    target_report = _validate_triangles(target_vertices, target_faces_array, name="target_surface_faces")
    if not np.isfinite(float(epsilon_m)) or float(epsilon_m) <= 0.0:
        raise ValueError("epsilon_m must be positive and finite")
    requested_k = int(max_components)
    if requested_k <= 0:
        raise ValueError("max_components must be positive")
    source_labels = _normalise_component_ids(
        source_face_component_ids,
        len(source_faces_array),
        name="source_face_component_ids",
    )
    if target_face_component_ids is None:
        if source_face_component_ids is not None and len(target_faces_array) == len(source_faces_array):
            target_labels = _normalise_component_ids(
                source_labels,
                len(target_faces_array),
                name="target_face_component_ids",
            )
        else:
            target_labels = _normalise_component_ids(
                None,
                len(target_faces_array),
                name="target_face_component_ids",
            )
    else:
        target_labels = _normalise_component_ids(
            target_face_component_ids,
            len(target_faces_array),
            name="target_face_component_ids",
        )
    source_component_names, source_groups = _component_groups(source_labels)
    target_component_names, target_groups = _component_groups(target_labels)
    if component_map is None:
        mapping = {}
        if target_face_component_ids is None and len(target_component_names) == 1:
            mapping = {label: target_component_names[0] for label in source_component_names}
        else:
            for label in source_component_names:
                if label not in target_groups:
                    raise ValueError(
                        f"target surface has no component {label!r}; provide component_map"
                    )
                mapping[label] = label
    else:
        mapping = {str(key.item() if isinstance(key, np.generic) else key): str(value.item() if isinstance(value, np.generic) else value) for key, value in component_map.items()}
        missing = [label for label in source_component_names if label not in mapping]
        if missing:
            raise ValueError(f"component_map is missing source components: {missing}")
        unknown = sorted({mapping[label] for label in source_component_names} - set(target_groups))
        if unknown:
            raise ValueError(f"component_map targets unknown target components: {unknown}")
    centroids = _component_centroids(source_vertices, source_faces_array, source_groups)
    candidate_count = min(MAX_COMPONENTS, requested_k, len(source_component_names))
    candidate_ids, _centroid_distance_sq = _component_candidates(
        source_points, source_component_names, centroids, candidate_count
    )
    source_d2, source_face_indices, source_closest, source_bary = _closest_by_component(
        source_points,
        source_vertices,
        source_faces_array,
        source_groups,
        candidate_ids,
    )
    target_candidate_ids = np.asarray(
        [[mapping[str(label)] for label in row] for row in candidate_ids], dtype=str
    )
    target_d2, target_face_indices, target_closest, target_bary = _closest_by_component(
        source_closest.reshape(-1, 3),
        target_vertices,
        target_faces_array,
        target_groups,
        target_candidate_ids.reshape(-1, 1),
    )
    target_d2 = target_d2.reshape(len(source_points), candidate_count)
    target_face_indices = target_face_indices.reshape(len(source_points), candidate_count)
    target_closest = target_closest.reshape(len(source_points), candidate_count, 3)
    target_bary = target_bary.reshape(len(source_points), candidate_count, 3)
    # ``_closest_by_component`` above receives one query per candidate, so
    # source_closest is flattened in row-major [point, candidate] order.
    source_triangles = source_vertices[source_faces_array[source_face_indices]]
    target_triangles = target_vertices[target_faces_array[target_face_indices]]
    source_frames = _triangle_frames(source_triangles.reshape(-1, 3, 3), name="source rest")
    target_frames = _triangle_frames(target_triangles.reshape(-1, 3, 3), name="target rest")
    source_frames = source_frames.reshape(len(source_points), candidate_count, 3, 3)
    target_frames = target_frames.reshape(len(source_points), candidate_count, 3, 3)
    source_offset = source_points[:, None, :] - source_closest
    target_offset = source_points[:, None, :] - target_closest
    source_offset_local = np.einsum("pkji,pkj->pki", source_frames, source_offset)
    target_offset_local = np.einsum("pkji,pkj->pki", target_frames, target_offset)
    # The source distance controls blend locality.  It remains finite on the
    # surface because epsilon is explicit rather than an accidental divide by
    # zero.  The exact candidate triangle remains fixed after this step.
    weights = 1.0 / np.maximum(source_d2 + float(epsilon_m) ** 2, 1.0e-30)
    weights /= np.sum(weights, axis=1, keepdims=True)
    if not np.all(np.isfinite(weights)):
        raise ValueError("component weights are non-finite")
    source_rest_reconstructed = source_closest + np.einsum(
        "pkij,pkj->pki", source_frames, source_offset_local
    )
    target_rest_reconstructed = target_closest + np.einsum(
        "pkij,pkj->pki", target_frames, target_offset_local
    )
    source_reconstruction_error = np.linalg.norm(source_rest_reconstructed - source_points[:, None, :], axis=2)
    target_reconstruction_error = np.linalg.norm(target_rest_reconstructed - source_points[:, None, :], axis=2)
    report: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "compile_backend": "libigl_or_exact_triangle_fallback",
        "candidate_selection": "nearest_component_centroids",
        "candidate_count": int(candidate_count),
        "requested_max_components": int(requested_k),
        "epsilon_m": float(epsilon_m),
        "source_component_count": int(len(source_component_names)),
        "target_component_count": int(len(target_component_names)),
        "source_face_count": int(len(source_faces_array)),
        "target_face_count": int(len(target_faces_array)),
        "source_degenerate_face_count": int(source_report["degenerate_face_count"]),
        "target_degenerate_face_count": int(target_report["degenerate_face_count"]),
        "source_minimum_triangle_area_m2": float(source_report["minimum_triangle_area_m2"]),
        "target_minimum_triangle_area_m2": float(target_report["minimum_triangle_area_m2"]),
        "source_nearest_distance_max_m": float(np.max(np.sqrt(source_d2))),
        "source_nearest_distance_rms_m": float(np.sqrt(np.mean(source_d2))),
        "source_rest_reconstruction_max_m": float(np.max(source_reconstruction_error)),
        "target_rest_reconstruction_max_m": float(np.max(target_reconstruction_error)),
        # A surface attachment does not by itself prove volume containment.
        "containment_status": "not_evaluated",
        "containment_verified": False,
        "source_topology_digest": _digest_array(source_faces_array),
        "target_topology_digest": _digest_array(target_faces_array),
    }
    result = MaterialAttachmentMapV13(
        source_faces=source_faces_array,
        target_faces=target_faces_array,
        source_face_indices=source_face_indices,
        target_face_indices=target_face_indices,
        source_barycentric=source_bary,
        target_barycentric=target_bary,
        source_offset_local=source_offset_local,
        target_offset_local=target_offset_local,
        component_weights=weights,
        component_ids=candidate_ids,
        source_distances_m=np.sqrt(source_d2),
        source_points_rest=source_points,
        source_surface_digest=_digest_array(source_vertices),
        target_surface_digest=_digest_array(target_vertices),
        source_points_digest=_digest_array(source_points),
        epsilon_m=float(epsilon_m),
        report=report,
    )
    return (result, dict(report)) if return_report else result


def transport_material_attachments_v13(
    attachment_map: MaterialAttachmentMapV13,
    source_points_posed: np.ndarray,
    source_surface_vertices_posed: np.ndarray,
    target_surface_vertices_posed: np.ndarray,
    *,
    return_audit: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Functional wrapper around :meth:`MaterialAttachmentMapV13.transport`."""

    if not isinstance(attachment_map, MaterialAttachmentMapV13):
        raise TypeError("attachment_map must be a MaterialAttachmentMapV13")
    return attachment_map.transport(
        source_points_posed,
        source_surface_vertices_posed,
        target_surface_vertices_posed,
        return_audit=return_audit,
    )


# Short aliases make the experimental module easy to call from existing
# anatomy-retarget scripts while retaining one explicit v13 API name.
compile_attachment_map_v13 = compile_material_attachment_map_v13
transport_material_attachment_v13 = transport_material_attachments_v13
MaterialAttachmentV13 = MaterialAttachmentMapV13


__all__ = [
    "DEFAULT_EPSILON_M",
    "MAX_COMPONENTS",
    "SCHEMA_VERSION",
    "MaterialAttachmentMapV13",
    "MaterialAttachmentV13",
    "compile_material_attachment_map_v13",
    "compile_attachment_map_v13",
    "transport_material_attachments_v13",
    "transport_material_attachment_v13",
]
