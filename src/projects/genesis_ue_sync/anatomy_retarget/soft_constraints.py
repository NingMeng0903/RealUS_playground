"""Local rigidity and volume constraints for retargeted organ surfaces."""

from __future__ import annotations

from typing import Any

import numpy as np


def unique_mesh_edges(faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    edges = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]),
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def signed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    return float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                points[triangles[:, 0]],
                np.cross(points[triangles[:, 1]], points[triangles[:, 2]]),
            )
        )
        / 6.0
    )


def limit_edge_strain(
    rest_vertices: np.ndarray,
    target_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    minimum_ratio: float = 0.75,
    maximum_ratio: float = 1.25,
    iterations: int = 20,
    fixed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Project a tube surface onto bounded authored edge lengths.

    This is a position-based graph constraint, not a surface projection.  It
    preserves branch topology and makes the runtime maximum-edge gate effective
    for isolated bad authored-weight transitions that a global q99 can hide.
    """
    rest = np.asarray(rest_vertices, dtype=np.float64).reshape(-1, 3)
    current = np.asarray(target_vertices, dtype=np.float64).reshape(-1, 3).copy()
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if rest.shape != current.shape:
        raise ValueError("rest and target tube vertices must have matching shape")
    if not len(rest) or not len(triangles):
        return current, {
            "iterations": 0,
            "edge_ratio_q99": 1.0,
            "edge_ratio_max": 1.0,
        }
    lower = float(minimum_ratio)
    upper = float(maximum_ratio)
    if not 0.0 < lower <= 1.0 <= upper:
        raise ValueError("edge strain ratios must straddle one")
    edges = unique_mesh_edges(triangles)
    fixed = (
        np.zeros(len(current), dtype=bool)
        if fixed_mask is None
        else np.asarray(fixed_mask, dtype=bool).reshape(-1)
    )
    if fixed.shape != (len(current),):
        raise ValueError("fixed_mask must have one value per vertex")
    authored = np.linalg.norm(
        rest[edges[:, 1]] - rest[edges[:, 0]],
        axis=1,
    )
    valid = authored > 1.0e-10
    edges = edges[valid]
    authored = authored[valid]
    for _iteration in range(max(0, int(iterations))):
        delta = current[edges[:, 1]] - current[edges[:, 0]]
        lengths = np.linalg.norm(delta, axis=1)
        desired = np.clip(lengths, lower * authored, upper * authored)
        correction = (
            0.5
            * ((lengths - desired) / np.maximum(lengths, 1.0e-12))[:, None]
            * delta
        )
        accumulated = np.zeros_like(current)
        counts = np.zeros(len(current), dtype=np.float64)
        np.add.at(accumulated, edges[:, 0], correction)
        np.add.at(accumulated, edges[:, 1], -correction)
        np.add.at(counts, edges[:, 0], 1.0)
        np.add.at(counts, edges[:, 1], 1.0)
        active = counts > 0.0
        active &= ~fixed
        current[active] += (
            0.9 * accumulated[active] / counts[active, None]
        )
    final = np.linalg.norm(
        current[edges[:, 1]] - current[edges[:, 0]],
        axis=1,
    )
    ratios = final / authored
    return current, {
        "iterations": max(0, int(iterations)),
        "edge_ratio_q99": float(np.quantile(ratios, 0.99)),
        "edge_ratio_max": float(np.max(ratios)),
    }


def surface_barrier_refine(
    vertices: np.ndarray,
    faces: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    *,
    strain_reference_vertices: np.ndarray | None = None,
    clearance_m: float = 2.5e-4,
    iterations: int = 4,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Resolve outside tube vertices with a smooth surface-barrier solve."""
    import igl

    source = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    strain_reference = np.asarray(
        source
        if strain_reference_vertices is None
        else strain_reference_vertices,
        dtype=np.float64,
    ).reshape(-1, 3)
    if strain_reference.shape != source.shape:
        raise ValueError("surface barrier strain reference must match vertices")
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    current = source.copy()
    if not len(source) or not len(triangles):
        return current, {
            "outside_before": 0,
            "outside_after": 0,
            "max_outside_before_m": 0.0,
            "max_outside_after_m": 0.0,
        }
    mesh_edges = unique_mesh_edges(triangles)

    def signed_distance(
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        signed, _face, closest, _normal = igl.signed_distance(
            points,
            np.asarray(surface_vertices, dtype=np.float64),
            np.asarray(surface_faces, dtype=np.int32),
        )
        return (
            np.asarray(signed, dtype=np.float64),
            np.asarray(closest, dtype=np.float64),
        )

    def coherent_barrier_target(
        points: np.ndarray,
        signed: np.ndarray,
        closest: np.ndarray,
        active: np.ndarray,
        clearance: float,
    ) -> np.ndarray:
        """Move each outside patch coherently toward its attached interior."""
        target = points.copy()
        active_neighbors: list[list[int]] = [[] for _ in range(len(points))]
        boundary_neighbors: list[list[int]] = [[] for _ in range(len(points))]
        for a, b in mesh_edges.tolist():
            if active[a] and active[b]:
                active_neighbors[a].append(b)
                active_neighbors[b].append(a)
            elif active[a]:
                boundary_neighbors[a].append(b)
            elif active[b]:
                boundary_neighbors[b].append(a)
        visited = np.zeros(len(points), dtype=bool)
        for seed in np.flatnonzero(active).tolist():
            if visited[seed]:
                continue
            stack = [seed]
            visited[seed] = True
            component: list[int] = []
            boundary: set[int] = set()
            while stack:
                vertex = stack.pop()
                component.append(vertex)
                boundary.update(boundary_neighbors[vertex])
                for neighbor in active_neighbors[vertex]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            component_index = np.asarray(component, dtype=np.int64)
            if boundary:
                boundary_index = np.fromiter(boundary, dtype=np.int64)
                direction = np.mean(points[boundary_index], axis=0) - np.mean(
                    points[component_index], axis=0
                )
            else:
                direction = np.mean(
                    closest[component_index] - points[component_index],
                    axis=0,
                )
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-12:
                direction = np.mean(
                    closest[component_index] - points[component_index],
                    axis=0,
                )
                norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-12:
                continue
            distance = float(np.max(signed[component_index])) + clearance
            target[component_index] += direction * (distance / norm)
        return target

    initial_signed, _initial_closest = signed_distance(current)
    for _iteration in range(max(0, int(iterations))):
        signed, closest = signed_distance(current)
        active = signed > 0.0
        if not np.any(active):
            break
        smooth_clearance = max(float(clearance_m), 5.0e-3)
        target = coherent_barrier_target(
            current,
            signed,
            closest,
            active,
            smooth_clearance,
        )
        weights = np.full(len(current), 0.01, dtype=np.float64)
        weights[active] = 100.0
        current, _arap_report = arap_volume_refine(
            strain_reference,
            target,
            triangles,
            target_weight=weights,
            iterations=8,
            volume_weight=0.0,
        )
    current, _strain_report = limit_edge_strain(
        strain_reference,
        current,
        triangles,
        minimum_ratio=0.85,
        maximum_ratio=1.08,
        iterations=100,
    )
    # Alternate coherent patch movement with a free edge projection.  The
    # millimetre buffer prevents the projection from merely oscillating across
    # the body boundary while still allowing neighbouring vertices to absorb
    # the displacement without a seam.
    for _iteration in range(6):
        signed, closest = signed_distance(current)
        active = signed > 0.0
        if np.any(active):
            target = coherent_barrier_target(
                current,
                signed,
                closest,
                active,
                max(float(clearance_m), 5.0e-3),
            )
            current[active] = target[active]
        current, _strain_report = limit_edge_strain(
            strain_reference,
            current,
            triangles,
            minimum_ratio=0.85,
            maximum_ratio=1.08,
            iterations=300,
        )
    signed, closest = signed_distance(current)
    active = signed > 0.0
    if np.any(active):
        target = coherent_barrier_target(
            current,
            signed,
            closest,
            active,
            float(clearance_m),
        )
        current[active] = target[active]
    final_signed, _final_closest = signed_distance(current)
    return current, {
        "outside_before": int(np.count_nonzero(initial_signed > 0.0)),
        "outside_after": int(np.count_nonzero(final_signed > 0.0)),
        "max_outside_before_m": float(
            max(0.0, float(np.max(initial_signed)))
        ),
        "max_outside_after_m": float(max(0.0, float(np.max(final_signed)))),
    }


def _vertex_rotations(
    rest: np.ndarray,
    current: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    covariance = np.zeros((len(rest), 3, 3), dtype=np.float64)
    a = edges[:, 0]
    b = edges[:, 1]
    p = rest[a] - rest[b]
    q = current[a] - current[b]
    outer = q[:, :, None] * p[:, None, :]
    np.add.at(covariance, a, outer)
    np.add.at(covariance, b, outer)
    u, _singular, vt = np.linalg.svd(covariance)
    rotations = u @ vt
    reflected = np.linalg.det(rotations) < 0.0
    if np.any(reflected):
        u[reflected, :, -1] *= -1.0
        rotations[reflected] = u[reflected] @ vt[reflected]
    return rotations


def arap_volume_refine(
    rest_vertices: np.ndarray,
    target_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_weight: float | np.ndarray = 8.0,
    iterations: int = 4,
    volume_weight: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine a harmonic organ target with local-rigidity and volume priors."""
    rest = np.asarray(rest_vertices, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_vertices, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if rest.shape != target.shape:
        raise ValueError("rest and target organ vertices must have matching shape")
    if not len(rest) or not len(triangles):
        return target.copy(), {
            "iterations": 0,
            "edge_ratio_p99": 1.0,
            "volume_ratio": 1.0,
        }
    edges = unique_mesh_edges(triangles)
    from scipy import sparse
    from scipy.sparse.linalg import factorized

    row = np.concatenate((edges[:, 0], edges[:, 1], edges[:, 0], edges[:, 1]))
    col = np.concatenate((edges[:, 0], edges[:, 1], edges[:, 1], edges[:, 0]))
    data = np.concatenate(
        (
            np.ones(len(edges)),
            np.ones(len(edges)),
            -np.ones(len(edges)),
            -np.ones(len(edges)),
        )
    )
    laplacian = sparse.coo_matrix((data, (row, col)), shape=(len(rest), len(rest))).tocsr()
    raw_weight = np.asarray(target_weight, dtype=np.float64)
    if raw_weight.ndim == 0:
        weights = np.full(
            len(rest),
            max(float(raw_weight), 1.0e-6),
            dtype=np.float64,
        )
    else:
        weights = raw_weight.reshape(-1)
        if weights.shape != (len(rest),):
            raise ValueError("target_weight array must have one value per vertex")
        weights = np.maximum(weights, 1.0e-6)
    solve = factorized(
        (laplacian + sparse.diags(weights)).tocsc()
    )
    current = target.copy()
    rest_volume = abs(signed_mesh_volume(rest, triangles))
    for _ in range(max(0, int(iterations))):
        rotations = _vertex_rotations(rest, current, edges)
        rhs = np.zeros_like(current)
        a = edges[:, 0]
        b = edges[:, 1]
        desired = np.einsum(
            "eij,ej->ei",
            0.5 * (rotations[a] + rotations[b]),
            rest[a] - rest[b],
        )
        np.add.at(rhs, a, desired)
        np.add.at(rhs, b, -desired)
        rhs += weights[:, None] * target
        current = np.stack([solve(rhs[:, axis]) for axis in range(3)], axis=1)
        current_volume = abs(signed_mesh_volume(current, triangles))
        if rest_volume > 1.0e-12 and current_volume > 1.0e-12 and volume_weight > 0.0:
            correction = (rest_volume / current_volume) ** (1.0 / 3.0)
            blended_scale = 1.0 + float(np.clip(volume_weight, 0.0, 1.0)) * (
                correction - 1.0
            )
            center = np.mean(current, axis=0)
            current = center + blended_scale * (current - center)

    before = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    after = np.linalg.norm(current[edges[:, 1]] - current[edges[:, 0]], axis=1)
    valid = before > 1.0e-10
    ratios = after[valid] / before[valid]
    final_volume = abs(signed_mesh_volume(current, triangles))
    return current, {
        "iterations": max(0, int(iterations)),
        "edge_ratio_p99": float(np.quantile(ratios, 0.99)) if len(ratios) else 1.0,
        "edge_ratio_max": float(np.max(ratios)) if len(ratios) else 1.0,
        "volume_ratio": final_volume / max(rest_volume, 1.0e-12),
        "target_rms_m": float(
            np.sqrt(np.mean(np.sum((current - target) ** 2, axis=1)))
        ),
    }


def regularize_asset_soft_materials(
    asset: Any,
    *,
    reference_vertices: np.ndarray,
    surface_vertices: np.ndarray | None = None,
    surface_faces: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply post-articulation material constraints per authored soft mesh."""
    from .material_fit import cranial_material_mask

    reference = np.asarray(reference_vertices, dtype=np.float64)
    current = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    if reference.shape != current.shape:
        raise ValueError("soft material reference must match asset vertices")
    faces = np.asarray(asset.faces, dtype=np.int64)
    cranial = cranial_material_mask(asset)
    reports: dict[str, Any] = {}
    for mesh_name, tissue, (start, stop) in zip(
        asset.source_mesh_names or [],
        asset.source_tissues or [],
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
    ):
        start_i, stop_i = int(start), int(stop)
        tissue_name = str(tissue).lower()
        if tissue_name in {"bone", "skin_boundary"} or np.any(
            cranial[start_i:stop_i]
        ):
            continue
        local_faces = faces[
            np.all((faces >= start_i) & (faces < stop_i), axis=1)
        ] - start_i
        if not len(local_faces):
            continue
        is_tube = tissue_name in {"vessel", "nerve"}
        is_organ = tissue_name in {"organ", "heart"}
        refined, arap_report = arap_volume_refine(
            reference[start_i:stop_i],
            current[start_i:stop_i],
            local_faces,
            target_weight=0.01 if is_tube else (0.001 if is_organ else 0.01),
            iterations=15 if is_tube else (30 if is_organ else 15),
            volume_weight=0.0 if not is_organ else 0.1,
        )
        refined, strain_report = limit_edge_strain(
            reference[start_i:stop_i],
            refined,
            local_faces,
            minimum_ratio=0.85 if is_tube else 0.75,
            maximum_ratio=1.25 if is_tube else 1.12,
            iterations=100 if is_tube else 300,
        )
        barrier_report = None
        # Skin distance is diagnostic/attenuation input for tubes. Never
        # project vessel or nerve vertices to the surface: this destroys thin
        # branch topology at exactly the difficult knee/foot/shoulder sites.
        current[start_i:stop_i] = refined
        reports[str(mesh_name)] = {
            "arap": arap_report,
            "bounded_edges": strain_report,
            "surface_barrier": barrier_report,
        }
    return current, reports
