"""QP guard for a shared V16 visceral rest-field increment.

The proposed field already supplies the spatial basis.  This module only
refits its vector coefficients under local linear guards, so the caller can
keep one field for every non-bone material and still run its ordinary baked
rest replay.  It deliberately has no mesh rebinding or runtime dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .shared_visceral_rest_fit_v16 import SharedWendlandFieldV16


_MM = 1.0e-3
_SKIN_TOLERANCE_M = 5.0e-5
_CONTACT_MM = 3.5


def _points(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite [N, 3]")
    return array


def _mesh_info(asset: Any, name: str) -> tuple[int, int, np.ndarray]:
    names = [str(x) for x in np.asarray(asset.source_mesh_names).tolist()]
    if name not in names:
        raise KeyError(name)
    index = names.index(name)
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[index]
    faces = np.asarray(asset.faces, dtype=np.int64)
    local = faces[
        (faces[:, 0] >= start) & (faces[:, 0] < stop)
        & (faces[:, 1] >= start) & (faces[:, 1] < stop)
        & (faces[:, 2] >= start) & (faces[:, 2] < stop)
    ] - int(start)
    return int(start), int(stop), np.asarray(local, dtype=np.int32)


def _closed_oriented(faces: np.ndarray) -> bool:
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return False
    directed = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
    )
    _edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    signs = np.where(directed[:, 0] < directed[:, 1], 1.0, -1.0)
    balance = np.bincount(inverse, weights=signs, minlength=len(counts))
    return bool(np.all(counts == 2) and np.all(np.abs(balance) == 0.0))


def _bone_mask(asset: Any, count: int) -> np.ndarray:
    result = np.zeros(count, dtype=bool)
    for (start, stop), tissue in zip(
        np.asarray(asset.source_vertex_ranges, dtype=np.int64).tolist(),
        np.asarray(asset.source_tissues).astype(str).tolist(),
    ):
        if str(tissue).strip().lower() == "bone":
            result[int(start) : int(stop)] = True
    return result


def _wendland_matrix(points: np.ndarray, centers: np.ndarray, radius: float):
    """Sparse compact Wendland C2 matrix, one row per query point."""

    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    query = _points(points, "field query")
    centers = _points(centers, "field centers")
    if not len(query) or not len(centers):
        return coo_matrix((len(query), len(centers)), dtype=np.float64).tocsr()
    tree = cKDTree(centers)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row, neighbours in enumerate(tree.query_ball_point(query, float(radius))):
        if not neighbours:
            continue
        ids = np.asarray(neighbours, dtype=np.int64)
        u = np.linalg.norm(centers[ids] - query[row], axis=1) / float(radius)
        active = u < 1.0
        ids, u = ids[active], u[active]
        if not len(ids):
            continue
        weight = (1.0 - u) ** 4 * (4.0 * u + 1.0)
        rows.extend([row] * len(ids))
        cols.extend(ids.tolist())
        values.extend(weight.tolist())
    return coo_matrix((values, (rows, cols)), shape=(len(query), len(centers))).tocsr()


def _signed_surface(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray):
    """Return negative-inside signed distance, closest points, and gradient."""

    import igl

    query = _points(points, "signed query")
    surface = _points(vertices, "signed surface")
    triangles = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    winding = np.asarray(igl.winding_number(surface, triangles, query), dtype=np.float64).reshape(-1)
    squared, face, closest = igl.point_mesh_squared_distance(query, surface, triangles)
    distance = np.sqrt(np.maximum(np.asarray(squared, dtype=np.float64), 0.0))
    inside = np.abs(winding) >= 0.5
    signed = np.where(inside, -distance, distance)
    closest = np.asarray(closest, dtype=np.float64)
    direction = closest - query
    norm = np.linalg.norm(direction, axis=1)
    # For points outside a closed surface the signed-distance gradient points
    # away from the closest point.  For points inside it points toward the
    # closest point, exactly as required by the half-space linearization.
    direction[~inside] *= -1.0
    valid = norm > 1.0e-10
    direction[valid] /= norm[valid, None]
    # A point exactly on a face has no closest-point displacement vector.  Use
    # the closed mesh's oriented face normal, calibrated once by the
    # centroid-shifted signed volume, so a zero-distance ligament/soft vertex
    # still gets a valid outward half-space gradient instead of being silently
    # dropped.
    zero = ~valid
    if np.any(zero):
        face_ids = np.asarray(face, dtype=np.int64).reshape(-1)
        face_ids = np.clip(face_ids, 0, len(triangles) - 1)
        tri = surface[triangles[face_ids]]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal_norm = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(normal_norm, 1.0e-12)[:, None]
        # The mesh is already checked for closed, consistently oriented
        # triangles.  Calibrate that orientation once from the signed volume;
        # flipping each face independently by a radial centroid heuristic is
        # wrong for concave bones where a face normal need not point away from
        # the global centroid.
        centroid = surface.mean(axis=0)
        shifted = surface - centroid
        tri_shifted = shifted[triangles]
        signed_volume6 = float(
            np.einsum(
                "ij,ij->i",
                tri_shifted[:, 0],
                np.cross(tri_shifted[:, 1], tri_shifted[:, 2]),
            ).sum()
        )
        if signed_volume6 < 0.0:
            normals *= -1.0
        direction[zero] = normals[zero]
    return signed, np.asarray(closest, dtype=np.float64), direction


def _active_edges(asset: Any, rest: np.ndarray, bone: np.ndarray, active: np.ndarray):
    from .soft_constraints import unique_mesh_edges

    edges = unique_mesh_edges(np.asarray(asset.faces, dtype=np.int64))
    keep = ~bone[edges].any(axis=1)
    keep &= active[edges].any(axis=1)
    edges = edges[keep]
    old = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    valid = old > 1.0e-10
    return edges[valid], old[valid]


def _zero_field(proposed: SharedWendlandFieldV16) -> SharedWendlandFieldV16:
    return SharedWendlandFieldV16(
        proposed.centers,
        np.zeros_like(proposed.coefficients),
        proposed.support_radius_m,
        proposed.regularization,
    )


def fit_guarded_field(
    initial: np.ndarray,
    current: np.ndarray,
    asset: Any,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
    points: np.ndarray,
    displacements: np.ndarray,
    proposed_field: SharedWendlandFieldV16,
) -> tuple[SharedWendlandFieldV16, dict[str, Any], Callable[[np.ndarray], tuple[bool, dict[str, Any]]]]:
    """Refit one proposed Wendland basis under bone/skin/edge guards.

    The returned callable evaluates the candidate with the actual closed
    surface signed distances.  It is intentionally independent of the QP's
    linearization and is suitable for the caller's final line search.
    """

    initial = _points(initial, "initial rest")
    current = _points(current, "current rest")
    if initial.shape != current.shape:
        raise ValueError("initial/current rest shapes differ")
    skin = _points(skin_vertices, "skin vertices")
    skin_triangles_raw = np.asarray(skin_faces, dtype=np.int32)
    if skin_triangles_raw.ndim != 2 or skin_triangles_raw.shape[1] != 3 or not len(skin_triangles_raw):
        raise ValueError("skin faces must be a non-empty [M, 3] array")
    skin_triangles = skin_triangles_raw.reshape(-1, 3)
    if np.any(skin_triangles < 0) or np.any(skin_triangles >= len(skin)):
        raise ValueError("skin faces contain an out-of-range vertex")
    if not _closed_oriented(skin_triangles):
        raise ValueError("skin surface must be closed with consistently oriented faces")
    observations = _points(points, "constraint points")
    targets = _points(displacements, "constraint displacements")
    if observations.shape != targets.shape:
        raise ValueError("constraint points/displacements shapes differ")
    centers = np.asarray(proposed_field.centers, dtype=np.float64)
    radius = float(proposed_field.support_radius_m)
    k = len(centers)
    n = len(initial)
    bone = _bone_mask(asset, n)
    nonbone = ~bone
    if not np.array_equal(initial[bone], current[bone]):
        raise ValueError("bone vertices must be unchanged before guarded fit")

    # Only rows inside the proposed compact support can receive a nonzero
    # increment.  All other non-bone vertices receive an exact zero row.
    nonbone_ids = np.flatnonzero(nonbone)
    W_all = _wendland_matrix(current[nonbone_ids], centers, radius)
    active_local = np.asarray(W_all.getnnz(axis=1)).reshape(-1) > 0
    active_ids = nonbone_ids[active_local]
    W_active = W_all[active_local].tocsr()
    active = np.zeros(n, dtype=bool)
    active[active_ids] = True

    # Stores actual SDF guard records for the returned checker.
    bone_guards: list[dict[str, Any]] = []
    skipped_bones: list[str] = []
    guard_rows: list[Any] = []
    guard_lower: list[np.ndarray] = []
    guard_upper: list[np.ndarray] = []
    guard_names: list[str] = []
    guard_groups: dict[str, list[tuple[Any, np.ndarray, np.ndarray]]] = {
        "bone": [], "skin": [], "edge": [], "step": []
    }
    for name, tissue in zip(
        np.asarray(asset.source_mesh_names).astype(str).tolist(),
        np.asarray(asset.source_tissues).astype(str).tolist(),
    ):
        if str(tissue).strip().lower() != "bone":
            continue
        start, stop, faces = _mesh_info(asset, str(name))
        if not _closed_oriented(faces):
            skipped_bones.append(str(name))
            continue
        bone_vertices = current[start:stop]
        low = bone_vertices.min(axis=0) - _CONTACT_MM * _MM
        high = bone_vertices.max(axis=0) + _CONTACT_MM * _MM
        nearby = active_ids[
            np.all(current[active_ids] >= low, axis=1)
            & np.all(current[active_ids] <= high, axis=1)
        ]
        if not len(nearby):
            continue
        local_rows = np.searchsorted(active_ids, nearby)
        signed_current, closest_current, gradient = _signed_surface(
            current[nearby], bone_vertices, faces
        )
        signed_initial, _closest_initial, _gradient_initial = _signed_surface(
            initial[nearby], bone_vertices, faces
        )
        near_surface = (signed_current <= _CONTACT_MM * _MM) | (signed_current < 0.0)
        if not np.any(near_surface):
            continue
        nearby = nearby[near_surface]
        local_rows = local_rows[near_surface]
        signed_current = signed_current[near_surface]
        signed_initial = signed_initial[near_surface]
        gradient = gradient[near_surface]
        valid = np.linalg.norm(gradient, axis=1) > 0.0
        nearby = nearby[valid]
        local_rows = local_rows[valid]
        signed_current = signed_current[valid]
        signed_initial = signed_initial[valid]
        gradient = gradient[valid]
        if not len(nearby):
            continue
        # A later fit must not undo an already improved clearance.  Use the
        # better of initial/current signed distances, capped at the requested
        # 0.5 mm target, before applying the small numerical margin.
        floor = (
            np.minimum(np.maximum(signed_initial, signed_current), 0.5 * _MM)
            - 0.02 * _MM
        )
        rhs = floor - signed_current
        D = W_active[local_rows]
        from scipy.sparse import hstack

        row_matrix = hstack(
            [D.multiply(gradient[:, 0, None]), D.multiply(gradient[:, 1, None]), D.multiply(gradient[:, 2, None])],
            format="csr",
        )
        guard_rows.append(row_matrix)
        guard_lower.append(rhs)
        guard_upper.append(np.full(len(rhs), np.inf, dtype=np.float64))
        guard_groups["bone"].append((row_matrix, rhs, np.full(len(rhs), np.inf, dtype=np.float64)))
        guard_names.extend([f"bone:{name}"] * len(rhs))
        bone_guards.append(
            dict(
                name=str(name), ids=nearby.copy(), vertices=bone_vertices.copy(), faces=faces.copy(), floor=floor.copy()
            )
        )

    # Skin guard uses the same active material rows and the same negative-
    # inside convention.  It is an upper half-space because moving outward
    # increases signed distance.
    skin_guards: dict[str, Any] | None = None
    if len(active_ids):
        skin_signed_current, skin_closest_current, skin_gradient = _signed_surface(
            current[active_ids], skin, skin_triangles
        )
        skin_signed_initial, _skin_closest_initial, _ = _signed_surface(
            initial[active_ids], skin, skin_triangles
        )
        skin_distance = np.abs(skin_signed_current)
        # Keep all active points already outside the skin in the guard set,
        # even when they are farther than the local 3.5 mm band; otherwise an
        # existing extrusion could be made worse by a nearby field.
        selected = (skin_distance <= _CONTACT_MM * _MM) | (skin_signed_current > 0.0)
        selected &= np.linalg.norm(skin_gradient, axis=1) > 0.0
        if np.any(selected):
            ids = active_ids[selected]
            local_rows = np.flatnonzero(selected)
            signed_current = skin_signed_current[selected]
            signed_initial = skin_signed_initial[selected]
            gradient = skin_gradient[selected]
            # Keep the better skin containment achieved by an earlier round:
            # the lower (more interior) initial/current distance determines
            # the ceiling, with a -1 mm containment cap.
            floor = (
                np.maximum(np.minimum(signed_initial, signed_current), -1.0 * _MM)
                + 0.02 * _MM
            )
            rhs = floor - signed_current
            D = W_active[local_rows]
            from scipy.sparse import hstack

            row_matrix = hstack(
                [D.multiply(gradient[:, 0, None]), D.multiply(gradient[:, 1, None]), D.multiply(gradient[:, 2, None])],
                format="csr",
            )
            guard_rows.append(row_matrix)
            guard_lower.append(np.full(len(rhs), -np.inf, dtype=np.float64))
            guard_upper.append(rhs)
            guard_groups["skin"].append((row_matrix, np.full(len(rhs), -np.inf, dtype=np.float64), rhs))
            guard_names.extend(["skin"] * len(rhs))
            skin_guards = dict(ids=ids.copy(), floor=floor.copy())

    # Rest-edge cumulative deformation guard.  A pre-existing violation gets
    # no remaining budget; this keeps the QP feasible and records the issue
    # for check_callable instead of silently granting more strain.
    from scipy.sparse import hstack, vstack

    edges, edge_lengths = _active_edges(asset, initial, bone, active)
    preexisting_edge_violations = 0
    if len(edges):
        # An edge can have only one active endpoint.  Index the full non-bone
        # matrix so an inactive endpoint contributes an exact zero row rather
        # than an invalid ``searchsorted(active_ids)`` lookup.
        rows_i = np.searchsorted(nonbone_ids, edges[:, 0])
        rows_j = np.searchsorted(nonbone_ids, edges[:, 1])
        D = W_all[rows_i] - W_all[rows_j]
        existing = (current[edges[:, 1]] - initial[edges[:, 1]]) - (current[edges[:, 0]] - initial[edges[:, 0]])
        bound = 0.15 * edge_lengths / np.sqrt(3.0)
        # D = W_i - W_j, while existing is old_delta_j - old_delta_i.  The
        # candidate cumulative edge displacement is existing - D*C, hence
        # existing-bound <= D*C <= existing+bound.  This permits a guarded
        # field to unload a prior deformation instead of freezing its sign.
        preexisting_edge_violations = int(np.count_nonzero(np.any(np.abs(existing) > bound[:, None], axis=1)))
        edge_rows = []
        edge_lower = []
        edge_upper = []
        for axis in range(3):
            component = D
            blocks = [component if j == axis else component.multiply(0.0) for j in range(3)]
            edge_rows.append(hstack(blocks, format="csr"))
            edge_lower.append(existing[:, axis] - bound)
            edge_upper.append(existing[:, axis] + bound)
        guard_rows.extend(edge_rows)
        guard_lower.extend(edge_lower)
        guard_upper.extend(edge_upper)
        for row_matrix, row_lower, row_upper in zip(edge_rows, edge_lower, edge_upper):
            guard_groups["edge"].append((row_matrix, row_lower, row_upper))

    # Every active non-bone point is bounded per axis by 3/sqrt(3) mm.
    step_axis = 0.003 / np.sqrt(3.0)
    if len(active_ids):
        for axis in range(3):
            blocks = [W_active if j == axis else W_active.multiply(0.0) for j in range(3)]
            guard_rows.append(hstack(blocks, format="csr"))
            guard_lower.append(np.full(len(active_ids), -step_axis, dtype=np.float64))
            guard_upper.append(np.full(len(active_ids), step_axis, dtype=np.float64))
            guard_groups["step"].append((guard_rows[-1], guard_lower[-1], guard_upper[-1]))

    # If no basis support exists, produce a valid zero candidate and checker.
    meta: dict[str, Any] = {
        "method": "osqp_guarded_wendland_v16",
        "support_radius_mm": radius * 1000.0,
        "center_count": int(k),
        "active_nonbone_vertices": int(len(active_ids)),
        "guarded_bone_rows": int(sum(len(x) for x in guard_lower[: len(bone_guards)])),
        "guarded_skin_rows": int(0 if skin_guards is None else len(skin_guards["ids"])),
        "guarded_edge_count": int(len(edges)),
        "preexisting_edge_violations": int(preexisting_edge_violations),
        "skipped_open_bones": int(len(skipped_bones)),
        "skin_closed_oriented": True,
        "triangle_crossings_tested": False,
        "solver_status": "not_run",
        "solver_failed": False,
        "zero_field": False,
    }

    def check_callable(candidate_vertices: np.ndarray) -> tuple[bool, dict[str, Any]]:
        candidate = _points(candidate_vertices, "candidate vertices")
        if candidate.shape != initial.shape:
            return False, {"reason": "candidate shape differs"}
        if not np.array_equal(candidate[bone], initial[bone]):
            return False, {"reason": "bone vertices changed"}
        inactive = nonbone & ~active
        if np.any(inactive) and not np.array_equal(candidate[inactive], current[inactive]):
            return False, {
                "reason": "inactive nonbone vertices changed",
                "count": int(np.count_nonzero(np.any(candidate[inactive] != current[inactive], axis=1))),
            }
        delta = candidate - current
        if len(active_ids):
            step_max = float(np.max(np.abs(delta[nonbone])))
            if step_max > step_axis + _SKIN_TOLERANCE_M:
                return False, {"reason": "per-axis step cap exceeded", "max_step_mm": step_max * 1000.0}
        if len(edges):
            cumulative = (candidate[edges[:, 1]] - initial[edges[:, 1]]) - (candidate[edges[:, 0]] - initial[edges[:, 0]])
            limits = 0.15 * edge_lengths[:, None] / np.sqrt(3.0)
            edge_tolerance = 1.0e-9 + 1.0e-4 * edge_lengths[:, None]
            if preexisting_edge_violations == 0 and np.any(np.abs(cumulative) > limits + edge_tolerance):
                return False, {"reason": "cumulative rest edge guard exceeded"}
            if preexisting_edge_violations:
                # The prior rest already violates the requested bound; the
                # candidate may not add to the violation, but it cannot be
                # declared clean by this guard.
                if np.any(np.abs(cumulative) > limits + edge_tolerance):
                    return False, {"reason": "preexisting cumulative rest edge violation", "count": preexisting_edge_violations}
        for record in bone_guards:
            signed, _closest, _gradient = _signed_surface(
                candidate[record["ids"]], record["vertices"], record["faces"]
            )
            margin = signed - record["floor"]
            if np.any(margin < -_SKIN_TOLERANCE_M):
                return False, {
                    "reason": f"bone signed guard failed: {record['name']}",
                    "minimum_margin_mm": float(margin.min() * 1000.0),
                }
        if skin_guards is not None:
            signed, _closest, _gradient = _signed_surface(
                candidate[skin_guards["ids"]], skin, skin_triangles
            )
            margin = skin_guards["floor"] - signed
            if np.any(margin < -_SKIN_TOLERANCE_M):
                return False, {
                    "reason": "skin signed guard failed",
                    "minimum_margin_mm": float(margin.min() * 1000.0),
                }
        return True, {
            "reason": "guards passed",
            "active_vertices": int(len(active_ids)),
            "guarded_bone_meshes": int(len(bone_guards)),
            "guarded_edges": int(len(edges)),
        }

    if not k or not len(active_ids) or not len(observations):
        meta.update(solver_status="zero_support", zero_field=True)
        return _zero_field(proposed_field), meta, check_callable

    # Objective: ||Phi(points) C - displacement||^2 + ridge ||C||^2.
    # The guard blocks below are kept separately so the active-set solver can
    # begin with all geometric guards and a small, uniform sample of the much
    # larger edge/step blocks.  Every iteration still evaluates the complete
    # raw constraint matrix; omitted rows are never silently treated as pass.
    W_points = _wendland_matrix(observations, centers, radius).tocsr()
    target = targets
    from scipy.sparse import block_diag, csr_matrix, eye

    Q = (W_points.T @ W_points).tocsr() + 1.0e-6 * eye(k, format="csr")
    P = block_diag((Q, Q, Q), format="csc")
    q = -np.concatenate(
        (W_points.T @ target[:, 0], W_points.T @ target[:, 1], W_points.T @ target[:, 2])
    )
    if not guard_rows:
        meta.update(solver_status="no_constraints", zero_field=True)
        return _zero_field(proposed_field), meta, check_callable

    group_order = ("bone", "skin", "edge", "step")

    def _assemble_group(
        blocks: list[tuple[Any, np.ndarray, np.ndarray]],
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        if not blocks:
            return csr_matrix((0, 3 * k), dtype=np.float64), np.empty(0), np.empty(0)
        matrices = [matrix.tocsr() for matrix, _lower, _upper in blocks]
        return (
            vstack(matrices, format="csr"),
            np.concatenate([np.asarray(lower, dtype=np.float64) for _matrix, lower, _upper in blocks]),
            np.concatenate([np.asarray(upper, dtype=np.float64) for _matrix, _lower, upper in blocks]),
        )

    # Assemble a single full matrix in a stable group order.  This gives both
    # the solver and the raw residual audit an unambiguous row mapping.
    full_groups: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {
        kind: _assemble_group(guard_groups[kind]) for kind in group_order
    }
    full_parts: list[Any] = []
    full_lower_parts: list[np.ndarray] = []
    full_upper_parts: list[np.ndarray] = []
    group_slices: dict[str, slice] = {}
    full_offset = 0
    for kind in group_order:
        matrix, lower_part, upper_part = full_groups[kind]
        rows = int(matrix.shape[0])
        group_slices[kind] = slice(full_offset, full_offset + rows)
        full_offset += rows
        if rows:
            full_parts.append(matrix)
            full_lower_parts.append(lower_part)
            full_upper_parts.append(upper_part)
    if not full_parts:
        meta.update(solver_status="no_constraints", zero_field=True)
        return _zero_field(proposed_field), meta, check_callable
    A_full = vstack(full_parts, format="csr")
    lower_full = np.concatenate(full_lower_parts)
    upper_full = np.concatenate(full_upper_parts)
    full_rows = int(A_full.shape[0])
    full_row_norm = np.sqrt(np.asarray(A_full.multiply(A_full).sum(axis=1)).reshape(-1))
    full_zero_rows = full_row_norm <= 1.0e-15
    impossible_zero = full_zero_rows & (
        (np.isfinite(lower_full) & (lower_full > 1.0e-12))
        | (np.isfinite(upper_full) & (upper_full < -1.0e-12))
    )
    meta.update(
        guarded_bone_rows=int(sum(len(block[1]) for block in guard_groups["bone"])),
        guarded_skin_rows=int(sum(len(block[1]) for block in guard_groups["skin"])),
        guarded_edge_rows=int(sum(len(block[1]) for block in guard_groups["edge"])),
        guarded_step_rows=int(sum(len(block[1]) for block in guard_groups["step"])),
        full_constraint_rows=full_rows,
        full_zero_constraint_rows=int(full_zero_rows.sum()),
        constraint_row_norm_min=float(full_row_norm[~full_zero_rows].min())
        if np.any(~full_zero_rows)
        else 0.0,
        constraint_row_norm_max=float(full_row_norm[~full_zero_rows].max())
        if np.any(~full_zero_rows)
        else 0.0,
        zero_constraint_rows=int(full_zero_rows.sum()),
    )
    if np.any(impossible_zero):
        meta.update(
            solver_failed=True,
            zero_field=True,
            solver_status="infeasible_zero_row",
            full_primal_constraints_satisfied=False,
            full_unsatisfied_constraint_rows=int(impossible_zero.sum()),
        )
        return _zero_field(proposed_field), meta, check_callable

    # Seed every bone/skin guard, then uniformly sample at most 512 rows from
    # each edge/step block.  Violated rows are added from the complete matrix
    # after each solve, with a bounded batch to keep OSQP practical on the
    # 100k-row anatomy case.
    def _seed_mask(size: int, limit: int = 512) -> np.ndarray:
        mask = np.zeros(size, dtype=bool)
        if size:
            count = min(int(size), int(limit))
            mask[np.linspace(0, size - 1, count, dtype=np.int64)] = True
        return mask

    active_by_group: dict[str, np.ndarray] = {}
    for kind in group_order:
        group_matrix, _group_lower, _group_upper = full_groups[kind]
        rows = int(group_matrix.shape[0])
        group_zero = full_zero_rows[group_slices[kind]]
        if kind in ("bone", "skin"):
            active_by_group[kind] = ~group_zero
        else:
            active_by_group[kind] = _seed_mask(rows)
            active_by_group[kind] &= ~group_zero

    def _selected_mask() -> np.ndarray:
        pieces = [active_by_group[kind] for kind in group_order if len(active_by_group[kind])]
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=bool)

    linear_tolerance = 1.0e-9
    max_active_iterations = 8
    add_per_iteration = 512
    last_x: np.ndarray | None = None
    last_status = "not_run"
    last_status_val = 0
    last_iterations = 0
    last_objective: float | None = None
    accepted = False
    constraints_added = 0
    active_iterations = 0
    last_full_ax = np.zeros(full_rows, dtype=np.float64)
    last_lower_violation = np.zeros(full_rows, dtype=np.float64)
    last_upper_violation = np.zeros(full_rows, dtype=np.float64)
    last_unselected_violations = 0

    try:
        import osqp

        for active_iterations in range(1, max_active_iterations + 1):
            selected = _selected_mask()
            selected_indices = np.flatnonzero(selected)
            if len(selected_indices):
                A_selected_raw = A_full[selected_indices].tocsr()
                lower_selected = lower_full[selected_indices]
                upper_selected = upper_full[selected_indices]
                selected_norm = np.sqrt(
                    np.asarray(A_selected_raw.multiply(A_selected_raw).sum(axis=1)).reshape(-1)
                )
                selected_keep = selected_norm > 1.0e-15
                if np.any(
                    (~selected_keep)
                    & (
                        (np.isfinite(lower_selected) & (lower_selected > 1.0e-12))
                        | (np.isfinite(upper_selected) & (upper_selected < -1.0e-12))
                    )
                ):
                    meta.update(
                        solver_failed=True,
                        zero_field=True,
                        solver_status="infeasible_selected_zero_row",
                    )
                    break
                row_scale = np.ones_like(selected_norm)
                row_scale[selected_keep] = 1.0 / selected_norm[selected_keep]
                A_selected = A_selected_raw[selected_keep].multiply(
                    row_scale[selected_keep, None]
                ).tocsc()
                lower_scaled = lower_selected[selected_keep] * row_scale[selected_keep]
                upper_scaled = upper_selected[selected_keep] * row_scale[selected_keep]
            else:
                A_selected = csr_matrix((0, 3 * k), dtype=np.float64).tocsc()
                lower_scaled = np.empty(0, dtype=np.float64)
                upper_scaled = np.empty(0, dtype=np.float64)

            # An empty selected set is possible only for a degenerate field;
            # the unconstrained ridge minimizer is then safe to evaluate below.
            solver = osqp.OSQP()
            solver.setup(
                P=P,
                q=q,
                A=A_selected,
                l=lower_scaled,
                u=upper_scaled,
                verbose=False,
                # Keep the scaled ADMM residual below the raw metre-level
                # audit threshold.  The latter is 1e-9 m; using a tenth of it
                # here avoids accepting a boundary row that is still outside
                # the complete unscaled matrix after multiplication by its
                # row norm.
                eps_abs=1.0e-10,
                eps_rel=1.0e-9,
                max_iter=100000,
                check_termination=25,
                polishing=True,
            )
            result = solver.solve()
            last_status = str(result.info.status)
            last_status_val = int(result.info.status_val)
            last_iterations = int(result.info.iter)
            objective_value = result.info.obj_val
            last_objective = float(objective_value) if np.isfinite(objective_value) else None
            if result.x is None or not np.isfinite(result.x).all():
                meta.update(
                    solver_failed=True,
                    zero_field=True,
                    solver_status=last_status,
                    solver_iterations=last_iterations,
                )
                break
            last_x = np.asarray(result.x, dtype=np.float64)

            # Audit every full row in original metres before deciding whether
            # the active set is complete.  Infinities denote one-sided bounds.
            last_full_ax = np.asarray(A_full @ last_x, dtype=np.float64).reshape(-1)
            last_lower_violation = np.where(
                np.isfinite(lower_full), np.maximum(lower_full - last_full_ax, 0.0), 0.0
            )
            last_upper_violation = np.where(
                np.isfinite(upper_full), np.maximum(last_full_ax - upper_full, 0.0), 0.0
            )
            full_violation = np.maximum(last_lower_violation, last_upper_violation)
            violating = full_violation > linear_tolerance
            last_unselected_violations = int(np.count_nonzero(violating & ~selected))
            if not np.any(violating):
                accepted = True
                break
            if not np.any(violating & ~selected):
                # All violating rows are already active.  Retrying with a
                # relaxed tolerance would hide a real primal failure, so fail
                # closed and return a zero field.
                break
            add_indices = np.flatnonzero(violating & ~selected)
            if len(add_indices) > add_per_iteration:
                order = np.argsort(full_violation[add_indices])[::-1][:add_per_iteration]
                add_indices = add_indices[order]
            for index in add_indices.tolist():
                for kind in group_order:
                    group_slice = group_slices[kind]
                    if group_slice.start <= index < group_slice.stop:
                        active_by_group[kind][index - group_slice.start] = True
                        break
            constraints_added += int(len(add_indices))

        # A finite x with a completely feasible full primal solution is
        # acceptable even when OSQP reports a non-optimal status (for example,
        # maximum iterations).  It is labelled explicitly for auditability.
        full_violation = np.maximum(last_lower_violation, last_upper_violation)
        full_primal_ok = bool(np.all(full_violation <= linear_tolerance))
        feasible_status = last_status_val not in (3, 4, 5, 6, 9)
        if last_x is not None and full_primal_ok and feasible_status:
            accepted = True
        meta.update(
            solver_status=(
                "solved"
                if accepted and last_status_val == 1
                else "feasible_nonoptimal"
                if accepted
                else last_status
            ),
            solver_original_status=last_status,
            solver_status_val=int(last_status_val),
            solver_iterations=int(last_iterations),
            constraint_rows=int(np.count_nonzero(_selected_mask())),
            active_constraint_rows=int(np.count_nonzero(_selected_mask())),
            active_set_iterations=int(active_iterations),
            constraints_added=int(constraints_added),
            objective_value=last_objective,
            full_max_lower_violation_m=float(last_lower_violation.max(initial=0.0)),
            full_max_upper_violation_m=float(last_upper_violation.max(initial=0.0)),
            max_raw_constraint_violation_m=float(full_violation.max(initial=0.0)),
            full_primal_constraints_satisfied=bool(full_primal_ok),
            full_unsatisfied_constraint_rows=int(np.count_nonzero(~(full_violation <= linear_tolerance))),
            unselected_linear_violations=int(last_unselected_violations),
            linear_primal_tolerance_m=float(linear_tolerance),
        )
        if not accepted or last_x is None:
            meta.update(solver_failed=True, zero_field=True)
            return _zero_field(proposed_field), meta, check_callable
        coefficients = last_x.reshape(3, k).T
        field = SharedWendlandFieldV16(
            centers, coefficients, radius, proposed_field.regularization
        )
        meta["coefficient_max_mm"] = float(
            np.linalg.norm(coefficients, axis=1).max(initial=0.0) * 1000.0
        )
        meta["solver_failed"] = False
        meta["zero_field"] = False
        return field, meta, check_callable
    except Exception as exc:  # solver failure must be a safe zero field
        meta.update(
            solver_failed=True,
            zero_field=True,
            solver_status=f"error:{type(exc).__name__}",
            solver_error=str(exc),
        )
        return _zero_field(proposed_field), meta, check_callable


__all__ = ["fit_guarded_field"]
