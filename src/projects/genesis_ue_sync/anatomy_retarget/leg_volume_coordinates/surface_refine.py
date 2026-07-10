"""Surface precision refinement for canonical leg charts.

This module is geometry-first: on the skin surface the baked SMPL mesh
defines a piecewise-linear map between chart coordinates ``(theta, h)``
and canonical 3D points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atlas import LegVolumeAtlas

TAU = float(2.0 * np.pi)


def wrap_angle_delta(pred: np.ndarray | float, target: np.ndarray | float) -> np.ndarray:
    """Smallest signed angular difference ``pred - target`` in radians."""
    return ((np.asarray(pred) - np.asarray(target) + np.pi) % TAU - np.pi).astype(np.float32)


def _barycentric_2d(point: np.ndarray, tri: np.ndarray) -> np.ndarray | None:
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    v2 = point - a
    den = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(den) <= 1.0e-12:
        return None
    v = float((v2[0] * v1[1] - v1[0] * v2[1]) / den)
    w = float((v0[0] * v2[1] - v2[0] * v0[1]) / den)
    u = 1.0 - v - w
    return np.asarray([u, v, w], dtype=np.float32)


def _closest_point_segment_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    ab = b - a
    den = float(ab @ ab)
    if den <= 1.0e-12:
        return a, 0.0
    t = float(np.clip(((point - a) @ ab) / den, 0.0, 1.0))
    return a + t * ab, t


def _closest_barycentric_2d(point: np.ndarray, tri: np.ndarray) -> np.ndarray:
    bary = _barycentric_2d(point, tri)
    if bary is not None and float(np.min(bary)) >= 0.0:
        return bary
    best_bary = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    best_dist = float("inf")
    for i, j in ((0, 1), (1, 2), (2, 0)):
        closest, t = _closest_point_segment_2d(point, tri[i], tri[j])
        dist = float(np.sum(np.square(point - closest)))
        if dist < best_dist:
            b = np.zeros(3, dtype=np.float32)
            b[i] = 1.0 - t
            b[j] = t
            best_bary = b
            best_dist = dist
    return best_bary


def _closest_point_triangle_3d(point: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return closest point and barycentric coordinates on a 3D triangle."""
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(ab @ ap)
    d2 = float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    bp = point - b
    d3 = float(ab @ bp)
    d4 = float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.asarray([1.0 - v, v, 0.0], dtype=np.float32)

    cp = point - c
    d5 = float(ab @ cp)
    d6 = float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.asarray([1.0 - w, 0.0, w], dtype=np.float32)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.asarray([0.0, 1.0 - w, w], dtype=np.float32)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w, np.asarray([1.0 - v - w, v, w], dtype=np.float32)


@dataclass
class SurfaceAtlasRefiner:
    """Piecewise-linear high-precision skin map for ``d=0`` queries."""

    vertices: np.ndarray
    faces: np.ndarray
    theta: np.ndarray
    h: np.ndarray
    vertex_snap_tol: float = 1.0e-7
    candidate_k: int = 64
    _face_tree: object | None = field(default=None, init=False, repr=False)
    _face_features: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32), init=False, repr=False)

    def __post_init__(self) -> None:
        face_theta = self.theta[self.faces]
        theta_feature = np.stack([np.mean(np.cos(face_theta), axis=1), np.mean(np.sin(face_theta), axis=1)], axis=1)
        self._face_features = np.concatenate([theta_feature, np.mean(self.h[self.faces], axis=1, keepdims=True)], axis=1).astype(np.float32)
        try:
            from scipy.spatial import cKDTree

            self._face_tree = cKDTree(self._face_features)
        except Exception:
            self._face_tree = None

    @classmethod
    def from_atlas(cls, atlas: LegVolumeAtlas) -> "SurfaceAtlasRefiner":
        return cls(
            vertices=np.asarray(atlas.skin_vertices, dtype=np.float32).reshape(-1, 3),
            faces=np.asarray(atlas.skin_faces, dtype=np.int32).reshape(-1, 3),
            theta=np.mod(np.asarray(atlas.skin_theta, dtype=np.float32).reshape(-1), TAU),
            h=np.asarray(atlas.skin_h, dtype=np.float32).reshape(-1),
        )

    def xi_to_p(self, xi_radians: np.ndarray, reference_points: np.ndarray | None = None) -> np.ndarray:
        xi = np.asarray(xi_radians, dtype=np.float32).reshape(-1, 3)
        refs = None if reference_points is None else np.asarray(reference_points, dtype=np.float32).reshape(-1, 3)
        if refs is not None and refs.shape[0] != xi.shape[0]:
            raise ValueError("reference_points must have the same length as xi_radians.")
        out = np.empty((xi.shape[0], 3), dtype=np.float32)
        for row, sample in enumerate(xi):
            ref = None if refs is None else refs[row]
            out[row] = self._surface_point_for_theta_h(float(sample[0]), float(sample[1]), reference_point=ref)
        return out

    def p_to_xi(self, points_can: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_can, dtype=np.float32).reshape(-1, 3)
        xi = np.empty((points.shape[0], 3), dtype=np.float32)
        dist = np.empty((points.shape[0],), dtype=np.float32)
        tri_vertices = self.vertices[self.faces]
        for row, point in enumerate(points):
            best_dist = float("inf")
            best_face = self.faces[0]
            best_bary = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            for face, tri in zip(self.faces, tri_vertices, strict=True):
                closest, bary = _closest_point_triangle_3d(point, tri)
                d2 = float(np.sum(np.square(point - closest)))
                if d2 < best_dist:
                    best_dist = d2
                    best_face = face
                    best_bary = bary
            theta_tri = self._unwrapped_face_theta(best_face)
            theta = float(best_bary @ theta_tri)
            h = float(best_bary @ self.h[best_face])
            xi[row] = np.asarray([theta % TAU, np.clip(h, 0.0, 1.0), 0.0], dtype=np.float32)
            dist[row] = float(np.sqrt(max(best_dist, 0.0)))
        return xi, dist

    def _surface_point_for_theta_h(self, theta: float, h: float, *, reference_point: np.ndarray | None = None) -> np.ndarray:
        if reference_point is not None:
            return self._surface_point_for_theta_h_with_reference(theta, h, np.asarray(reference_point, dtype=np.float32))
        theta_delta = wrap_angle_delta(self.theta, theta)
        vertex_score = np.square(theta_delta / TAU) + np.square(self.h - float(h))
        nearest_vertex = int(np.argmin(vertex_score))
        if float(vertex_score[nearest_vertex]) <= float(self.vertex_snap_tol):
            return self.vertices[nearest_vertex].copy()
        best_face = self.faces[0]
        best_bary = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        best_score = float("inf")
        found_inside = False
        for face in self._candidate_faces(theta, h):
            theta_tri = self._unwrapped_face_theta(face)
            theta_q = float(theta_tri[0] + wrap_angle_delta(theta, theta_tri[0]))
            tri_uv = np.stack([theta_tri, self.h[face]], axis=1).astype(np.float32)
            point_uv = np.asarray([theta_q, h], dtype=np.float32)
            bary = _barycentric_2d(point_uv, tri_uv)
            if bary is not None and float(np.min(bary)) >= -1.0e-5:
                score = -float(np.min(bary))
                if (not found_inside) or score < best_score:
                    best_face = face
                    best_bary = bary
                    best_score = score
                    found_inside = True
            elif not found_inside:
                bary = _closest_barycentric_2d(point_uv, tri_uv)
                closest_uv = bary @ tri_uv
                score = float(np.sum(np.square(point_uv - closest_uv)))
                if score < best_score:
                    best_face = face
                    best_bary = bary
                    best_score = score
        return (best_bary @ self.vertices[best_face]).astype(np.float32)

    def _surface_point_for_theta_h_with_reference(self, theta: float, h: float, reference_point: np.ndarray) -> np.ndarray:
        """Resolve overlapping UV candidates by closest 3D point to a coarse reference."""
        theta_delta = wrap_angle_delta(self.theta, theta)
        vertex_score = np.square(theta_delta / TAU) + np.square(self.h - float(h))
        snap = np.flatnonzero(vertex_score <= float(self.vertex_snap_tol))
        if snap.size:
            ref = np.asarray(reference_point, dtype=np.float32).reshape(3)
            best = snap[int(np.argmin(np.sum(np.square(self.vertices[snap] - ref), axis=1)))]
            return self.vertices[int(best)].copy()
        candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        best_uv_score = float("inf")
        query_h = float(h)
        for face in self._candidate_faces(theta, h):
            theta_tri = self._unwrapped_face_theta(face)
            theta_q = float(theta_tri[0] + wrap_angle_delta(theta, theta_tri[0]))
            tri_uv = np.stack([theta_tri, self.h[face]], axis=1).astype(np.float32)
            point_uv = np.asarray([theta_q, query_h], dtype=np.float32)
            bary = _barycentric_2d(point_uv, tri_uv)
            if bary is not None and float(np.min(bary)) >= -1.0e-5:
                uv_score = 0.0
            else:
                bary = _closest_barycentric_2d(point_uv, tri_uv)
                closest_uv = bary @ tri_uv
                uv_score = float(np.sum(np.square(point_uv - closest_uv)))
            point = (bary @ self.vertices[face]).astype(np.float32)
            ref_score = float(np.sum(np.square(point - reference_point)))
            candidates.append((uv_score, ref_score, point, face))
            best_uv_score = min(best_uv_score, uv_score)
        uv_tol = max(best_uv_score + 1.0e-10, 1.0e-8)
        near_uv = [item for item in candidates if item[0] <= uv_tol] or candidates
        return min(near_uv, key=lambda item: item[1])[2].astype(np.float32)

    def _unwrapped_face_theta(self, face: np.ndarray) -> np.ndarray:
        raw = self.theta[face].astype(np.float32)
        base = float(raw[0])
        return np.asarray([base + float(wrap_angle_delta(v, base)) for v in raw], dtype=np.float32)

    def _candidate_faces(self, theta: float, h: float) -> np.ndarray:
        if self.faces.shape[0] <= int(self.candidate_k):
            return self.faces
        query = np.asarray([[np.cos(theta), np.sin(theta), float(h)]], dtype=np.float32)
        k = min(int(self.candidate_k), int(self.faces.shape[0]))
        if self._face_tree is not None:
            _dist, idx = self._face_tree.query(query, k=k)
            return self.faces[np.asarray(idx, dtype=np.int64).reshape(-1)]
        dist = np.sum(np.square(self._face_features - query), axis=1)
        return self.faces[np.argsort(dist)[:k]]
