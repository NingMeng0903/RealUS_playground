"""Volume precision refinement for canonical leg charts.

This module uses the baked harmonic tetrahedral volume as a deterministic
precision layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atlas import LegVolumeAtlas
from .harmonic import HarmonicVolumeMesh, LegHarmonicFields, interpolate_volume_field
from .surface_refine import TAU, wrap_angle_delta


def _barycentric_tet(point: np.ndarray, tet: np.ndarray) -> np.ndarray | None:
    a = tet[0].astype(np.float64)
    mat = np.stack([tet[1] - a, tet[2] - a, tet[3] - a], axis=1).astype(np.float64)
    rhs = np.asarray(point, dtype=np.float64).reshape(3) - a
    try:
        local = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return None
    return np.asarray([1.0 - local.sum(), local[0], local[1], local[2]], dtype=np.float64)


def _clip_barycentric(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(4)
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return w / total


@dataclass
class VolumeTetRefiner:
    """High-precision volume map backed by the harmonic tetrahedral atlas."""

    vertices: np.ndarray
    tets: np.ndarray
    theta: np.ndarray
    h: np.ndarray
    d: np.ndarray
    fields: LegHarmonicFields
    candidate_k: int = 256
    newton_steps: int = 0
    finite_diff_eps: float = 1.0e-4

    @classmethod
    def from_atlas(
        cls,
        atlas: LegVolumeAtlas,
        *,
        candidate_k: int = 256,
        newton_steps: int = 0,
        finite_diff_eps: float = 1.0e-4,
    ) -> "VolumeTetRefiner | None":
        if atlas.harmonic_vertices.size == 0 or atlas.harmonic_tets.size == 0:
            return None
        mesh = HarmonicVolumeMesh(
            vertices=np.asarray(atlas.harmonic_vertices, dtype=np.float32).reshape(-1, 3),
            tets=np.asarray(atlas.harmonic_tets, dtype=np.int32).reshape(-1, 4),
            skin_vertex_indices=np.arange(int(atlas.skin_vertices.shape[0]), dtype=np.int32),
            medial_vertex_indices=np.zeros(0, dtype=np.int32),
        )
        fields = LegHarmonicFields(
            skin_h=np.asarray(atlas.skin_h, dtype=np.float32),
            skin_theta=np.asarray(atlas.skin_theta, dtype=np.float32),
            skin_d=np.asarray(atlas.skin_d, dtype=np.float32),
            vol_h=np.asarray(atlas.harmonic_h, dtype=np.float32),
            vol_theta=np.asarray(atlas.harmonic_theta, dtype=np.float32),
            vol_d=np.asarray(atlas.harmonic_d, dtype=np.float32),
            volume_mesh=mesh,
            medial_curve_h=np.asarray(atlas.core_h, dtype=np.float32),
            medial_curve_points=np.asarray(atlas.core_points, dtype=np.float32),
            metadata={},
        )
        return cls(
            vertices=mesh.vertices,
            tets=mesh.tets,
            theta=np.mod(np.asarray(atlas.harmonic_theta, dtype=np.float32).reshape(-1), TAU),
            h=np.asarray(atlas.harmonic_h, dtype=np.float32).reshape(-1),
            d=np.asarray(atlas.harmonic_d, dtype=np.float32).reshape(-1),
            fields=fields,
            candidate_k=int(candidate_k),
            newton_steps=int(newton_steps),
            finite_diff_eps=float(finite_diff_eps),
        )

    def p_to_xi(self, points_can: np.ndarray) -> np.ndarray:
        points = np.asarray(points_can, dtype=np.float32).reshape(-1, 3)
        h, theta, d = interpolate_volume_field(self.fields, points)
        return np.stack([theta, h, d], axis=1).astype(np.float32)

    def xi_to_p(self, xi_radians: np.ndarray, reference_points: np.ndarray | None = None) -> np.ndarray:
        xi = np.asarray(xi_radians, dtype=np.float32).reshape(-1, 3)
        refs = None if reference_points is None else np.asarray(reference_points, dtype=np.float32).reshape(-1, 3)
        if refs is not None and refs.shape[0] != xi.shape[0]:
            raise ValueError("reference_points must have the same length as xi_radians.")
        initial = refs.copy() if refs is not None else self._initial_points_from_xi(xi, None)
        out = np.empty_like(initial)
        for idx, (target, start) in enumerate(zip(xi, initial, strict=True)):
            ref = None if refs is None else refs[idx]
            out[idx] = self._newton_refine(target, start, reference_point=ref)
        return out.astype(np.float32)

    def _initial_points_from_xi(self, xi: np.ndarray, refs: np.ndarray | None) -> np.ndarray:
        centroids = self._xi_centroids()
        query = self._xi_feature(xi)
        try:
            from scipy.spatial import cKDTree

            _dist, candidates = cKDTree(centroids).query(query, k=min(int(self.candidate_k), self.tets.shape[0]))
            candidates = np.asarray(candidates, dtype=np.int64).reshape(xi.shape[0], -1)
        except Exception:
            dist = np.linalg.norm(centroids[:, None, :] - query[None, :, :], axis=2).T
            candidates = np.argsort(dist, axis=1)[:, : min(int(self.candidate_k), self.tets.shape[0])]
        if refs is not None:
            p_candidates = self._p_candidates(refs)
            candidates = np.asarray(
                [np.unique(np.concatenate([a, b])).astype(np.int64) for a, b in zip(candidates, p_candidates, strict=True)],
                dtype=object,
            )

        points = np.empty((xi.shape[0], 3), dtype=np.float32)
        for row, target in enumerate(xi):
            ref = None if refs is None else refs[row]
            row_candidates = np.asarray(candidates[row], dtype=np.int64).reshape(-1)
            best_point = ref.copy() if ref is not None else self.vertices[self.tets[row_candidates[0]]].mean(axis=0)
            best_score = float("inf")
            for tet_idx in row_candidates:
                tet = self.tets[int(tet_idx)]
                xi_tet = self._unwrapped_tet_xi(tet, float(target[0]))
                bary = _barycentric_tet(np.asarray([target[0], target[1], target[2]], dtype=np.float32), xi_tet)
                if bary is None:
                    continue
                inside_penalty = float(np.sum(np.square(np.minimum(bary, 0.0))))
                weights = bary if inside_penalty <= 1.0e-10 else _clip_barycentric(bary)
                point = (weights @ self.vertices[tet]).astype(np.float32)
                ref_score = 0.0 if ref is None else float(np.sum(np.square(point - ref)))
                score = inside_penalty * 1.0e6 + ref_score
                if score < best_score:
                    best_score = score
                    best_point = point
            points[row] = best_point
        return points

    def _p_candidates(self, refs: np.ndarray) -> np.ndarray:
        centroids = self.vertices[self.tets].mean(axis=1).astype(np.float32)
        try:
            from scipy.spatial import cKDTree

            _dist, candidates = cKDTree(centroids).query(
                np.asarray(refs, dtype=np.float32).reshape(-1, 3),
                k=min(int(self.candidate_k), self.tets.shape[0]),
            )
            return np.asarray(candidates, dtype=np.int64).reshape(refs.shape[0], -1)
        except Exception:
            dist = np.linalg.norm(centroids[:, None, :] - refs[None, :, :], axis=2).T
            return np.argsort(dist, axis=1)[:, : min(int(self.candidate_k), self.tets.shape[0])]

    def _newton_refine(self, xi_target: np.ndarray, start: np.ndarray, *, reference_point: np.ndarray | None = None) -> np.ndarray:
        p = np.asarray(start, dtype=np.float64).reshape(3).copy()
        ref = None if reference_point is None else np.asarray(reference_point, dtype=np.float64).reshape(3)
        target = np.asarray(xi_target, dtype=np.float64).reshape(3)
        for _ in range(max(0, int(self.newton_steps))):
            current = self.p_to_xi(p.reshape(1, 3))[0].astype(np.float64)
            residual = np.asarray(
                [
                    float(wrap_angle_delta(current[0], target[0])),
                    current[1] - target[1],
                    current[2] - target[2],
                ],
                dtype=np.float64,
            )
            if float(np.linalg.norm(residual)) <= 1.0e-8:
                break
            jac = self._finite_difference_jacobian(p)
            if ref is not None:
                pos_weight = 0.5
                lhs = np.vstack([jac, np.eye(3, dtype=np.float64) * pos_weight])
                rhs = np.concatenate([residual, (p - ref) * pos_weight], axis=0)
            else:
                lhs = jac
                rhs = residual
            try:
                step, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            except np.linalg.LinAlgError:
                break
            step = np.clip(step, -0.01, 0.01)
            best_p = p
            best_norm = float(np.linalg.norm(residual))
            for scale in (1.0, 0.5, 0.25, 0.1):
                trial = p - scale * step
                trial_xi = self.p_to_xi(trial.reshape(1, 3))[0].astype(np.float64)
                trial_res = np.asarray(
                    [
                        float(wrap_angle_delta(trial_xi[0], target[0])),
                        trial_xi[1] - target[1],
                        trial_xi[2] - target[2],
                    ],
                    dtype=np.float64,
                )
                norm = float(np.linalg.norm(trial_res))
                if ref is not None:
                    norm += 0.5 * float(np.linalg.norm(trial - ref))
                if np.isfinite(norm) and norm < best_norm:
                    best_norm = norm
                    best_p = trial
                    break
            if np.allclose(best_p, p):
                break
            p = best_p
        return p.astype(np.float32)

    def _finite_difference_jacobian(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64).reshape(3)
        eps = float(self.finite_diff_eps)
        jac = np.zeros((3, 3), dtype=np.float64)
        for axis in range(3):
            delta = np.zeros(3, dtype=np.float64)
            delta[axis] = eps
            plus = self.p_to_xi((p + delta).reshape(1, 3))[0].astype(np.float64)
            minus = self.p_to_xi((p - delta).reshape(1, 3))[0].astype(np.float64)
            jac[:, axis] = np.asarray(
                [
                    float(wrap_angle_delta(plus[0], minus[0])) / (2.0 * eps),
                    (plus[1] - minus[1]) / (2.0 * eps),
                    (plus[2] - minus[2]) / (2.0 * eps),
                ],
                dtype=np.float64,
            )
        return jac

    def _unwrapped_tet_xi(self, tet: np.ndarray, query_theta: float) -> np.ndarray:
        theta = float(query_theta) + wrap_angle_delta(self.theta[tet], float(query_theta))
        return np.stack([theta, self.h[tet], self.d[tet]], axis=1).astype(np.float32)

    def _xi_feature(self, xi: np.ndarray) -> np.ndarray:
        arr = np.asarray(xi, dtype=np.float32).reshape(-1, 3)
        return np.stack([np.cos(arr[:, 0]), np.sin(arr[:, 0]), arr[:, 1], arr[:, 2]], axis=1).astype(np.float32)

    def _xi_centroids(self) -> np.ndarray:
        theta = np.mod(self.theta[self.tets], TAU)
        return np.stack(
            [
                np.mean(np.cos(theta), axis=1),
                np.mean(np.sin(theta), axis=1),
                np.mean(self.h[self.tets], axis=1),
                np.mean(self.d[self.tets], axis=1),
            ],
            axis=1,
        ).astype(np.float32)
