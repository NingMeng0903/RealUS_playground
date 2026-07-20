"""TetGen/FEM harmonic subject-beta deformation for internal anatomy."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset


_BETA_SOLVER_VERSION = "neutral_beta_volume_v2_surface_only_handles_jacobian05"
_MIN_JACOBIAN_RATIO = 0.05


def _digest_array(digest: Any, label: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _shape_cage_digest(neutral_v: np.ndarray, neutral_f: np.ndarray) -> str:
    digest = hashlib.sha256(_BETA_SOLVER_VERSION.encode("utf-8"))
    _digest_array(digest, "neutral_surface_vertices", np.asarray(neutral_v, dtype=np.float32))
    _digest_array(digest, "neutral_surface_faces", np.asarray(neutral_f, dtype=np.int32))
    return digest.hexdigest()


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(v.split("/", 1)[0]) - 1 for v in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _triangle_barycentric(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = triangles[:, 0]
    v0, v1, v2 = triangles[:, 1] - a, triangles[:, 2] - a, points - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = np.maximum(d00 * d11 - d01 * d01, 1.0e-16)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    out = np.clip(np.stack((1.0 - v - w, v, w), axis=1), 0.0, 1.0)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1.0e-12)


def _attach_smplx_boundary_map(
    cage: dict[str, np.ndarray],
    *,
    neutral_v: np.ndarray,
    neutral_f: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map every cage-boundary node onto the SMPL-X surface for beta BC values."""
    import igl

    if "source_triangles" in cage and "source_bary" in cage:
        return cage
    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    boundary = np.asarray(cage["boundary"], dtype=np.int64)
    _sq, face_index, closest = igl.point_mesh_squared_distance(nodes[boundary], neutral_v, neutral_f)
    source_triangles = neutral_f[np.asarray(face_index, dtype=np.int64)]
    source_bary = _triangle_barycentric(np.asarray(closest, dtype=np.float64), neutral_v[source_triangles])
    cage = dict(cage)
    cage["source_triangles"] = source_triangles.astype(np.int32)
    cage["source_bary"] = source_bary.astype(np.float32)
    return cage


def _build_cage(neutral_v: np.ndarray, neutral_f: np.ndarray, *, cache_path: Path) -> dict[str, np.ndarray]:
    signature = _shape_cage_digest(neutral_v, neutral_f)
    if cache_path.is_file():
        data = np.load(cache_path)
        cached = str(np.asarray(data.get("signature", "")).reshape(-1)[0])
        if cached == signature:
            return {key: np.asarray(data[key]) for key in data.files}

    import igl
    import pymeshfix
    import tetgen
    import trimesh

    mesh = trimesh.Trimesh(neutral_v, neutral_f, process=False)
    body = max(mesh.split(only_watertight=False), key=lambda item: len(item.faces))
    fixer = pymeshfix.MeshFix(np.asarray(body.vertices), np.asarray(body.faces))
    fixer.repair(joincomp=True, remove_smallest_components=False)
    repaired_v = np.asarray(fixer.points, dtype=np.float64)
    repaired_f = np.asarray(fixer.faces, dtype=np.int32)
    repaired = trimesh.Trimesh(repaired_v, repaired_f, process=False)
    if not repaired.is_watertight:
        raise RuntimeError("neutral SMPL-X cage repair did not produce a watertight body")

    generator = tetgen.TetGen(repaired_v, repaired_f)
    nodes, elements, _attributes, _markers = generator.tetrahedralize(
        order=1,
        mindihedral=5.0,
        minratio=2.0,
        maxvolume=5.0e-4,
        quiet=True,
    )
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int32)
    boundary = np.unique(np.asarray(generator.trifaces, dtype=np.int32).reshape(-1))
    _sq, face_index, closest = igl.point_mesh_squared_distance(nodes[boundary], neutral_v, neutral_f)
    source_triangles = neutral_f[np.asarray(face_index, dtype=np.int64)]
    source_bary = _triangle_barycentric(closest, neutral_v[source_triangles])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        nodes=nodes.astype(np.float32),
        elements=elements.astype(np.int32),
        boundary=boundary.astype(np.int32),
        source_triangles=source_triangles.astype(np.int32),
        source_bary=source_bary.astype(np.float32),
        signature=np.asarray([signature]),
        solver_version=np.asarray([_BETA_SOLVER_VERSION]),
    )
    return {
        "nodes": nodes,
        "elements": elements,
        "boundary": boundary,
        "source_triangles": source_triangles,
        "source_bary": source_bary,
        "signature": np.asarray([signature]),
        "solver_version": np.asarray([_BETA_SOLVER_VERSION]),
    }


def _tet_stiffness(nodes: np.ndarray, elements: np.ndarray):
    from scipy.sparse import coo_matrix

    tet = np.asarray(elements, dtype=np.int64)
    xyz = np.asarray(nodes[tet], dtype=np.float64)
    system = np.concatenate([np.ones((len(tet), 4, 1), dtype=np.float64), xyz], axis=2)
    determinants = np.linalg.det(xyz[:, 1:] - xyz[:, :1])
    volume = np.abs(determinants) / 6.0
    if np.any(volume <= 1.0e-18):
        raise RuntimeError("degenerate tetrahedron in neutral volume cage")
    gradients = np.linalg.inv(system)[:, 1:, :]
    local = volume[:, None, None] * np.einsum("tji,tjk->tik", gradients, gradients)
    row_idx = np.repeat(tet, 4, axis=1).reshape(-1)
    col_idx = np.tile(tet, (1, 4)).reshape(-1)
    values = local.reshape(-1)
    return coo_matrix((values, (row_idx, col_idx)), shape=(len(nodes), len(nodes))).tocsr()


def _solve_interior_harmonic(
    stiffness,
    interior: np.ndarray,
    boundary: np.ndarray,
    boundary_values: np.ndarray,
) -> np.ndarray:
    """Solve ``Kii x = -Kib boundary`` for one or more RHS columns."""
    from scipy.sparse.linalg import splu

    if interior.size == 0:
        return np.zeros((0, boundary_values.shape[-1]), dtype=np.float64)
    kii = stiffness[interior][:, interior].tocsc()
    kib = stiffness[interior][:, boundary]
    rhs = -(kib @ np.asarray(boundary_values, dtype=np.float64).reshape(len(boundary), -1))
    return np.asarray(splu(kii).solve(np.asarray(rhs, dtype=np.float64)), dtype=np.float64)


def _normalize_internal_handles(
    cage: dict[str, np.ndarray],
    internal_handles: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Resolve optional point/node handles to interior cage-node constraints."""
    if not internal_handles:
        return {
            "node_indices": np.zeros(0, dtype=np.int64),
            "displacements": np.zeros((0, 3), dtype=np.float64),
            "mode_codes": np.zeros(0, dtype=np.int8),
            "weights": np.zeros(0, dtype=np.float64),
        }
    if bool(internal_handles.get("_normalized", False)):
        return {
            "node_indices": np.asarray(internal_handles["node_indices"], dtype=np.int64),
            "displacements": np.asarray(internal_handles["displacements"], dtype=np.float64),
            "mode_codes": np.asarray(internal_handles["mode_codes"], dtype=np.int8),
            "weights": np.asarray(internal_handles["weights"], dtype=np.float64),
        }

    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    boundary = np.asarray(cage["boundary"], dtype=np.int64)
    raw_points = internal_handles.get("source_points", internal_handles.get("points"))
    raw_indices = internal_handles.get("node_indices")
    if (raw_points is None) == (raw_indices is None):
        raise ValueError("internal handles require exactly one of node_indices or source_points")

    points: np.ndarray | None = None
    if raw_indices is not None:
        node_indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
    else:
        points = np.asarray(raw_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("internal handle source_points must have shape [H,3]")
        _unused, outside_count, _outside = _sample_field(
            points, cage=cage, field=np.zeros_like(nodes)
        )
        if outside_count:
            raise ValueError(f"{outside_count} internal handle source points lie outside the cage")
        interior = np.setdiff1d(
            np.arange(len(nodes), dtype=np.int64), boundary, assume_unique=False
        )
        if not len(interior):
            raise ValueError("volume cage has no interior nodes for internal handles")
        from scipy.spatial import cKDTree

        candidate_count = min(max(1, len(points)), len(interior))
        _distance, candidates = cKDTree(nodes[interior]).query(
            points,
            k=candidate_count,
        )
        candidates = np.asarray(candidates, dtype=np.int64).reshape(
            len(points), candidate_count
        )
        selected: list[int] = []
        used: set[int] = set()
        for row in candidates:
            available = next(
                (int(candidate) for candidate in row.tolist() if int(candidate) not in used),
                None,
            )
            if available is None:
                raise ValueError("internal handles cannot be assigned to unique cage nodes")
            selected.append(available)
            used.add(available)
        node_indices = interior[np.asarray(selected, dtype=np.int64)]

    raw_displacements = internal_handles.get(
        "displacements", internal_handles.get("target_displacements")
    )
    if raw_displacements is None and points is not None and "target_points" in internal_handles:
        raw_displacements = np.asarray(internal_handles["target_points"], dtype=np.float64) - points
    if raw_displacements is None:
        raise ValueError("internal handles require displacements or target_points")
    displacements = np.asarray(raw_displacements, dtype=np.float64)
    if displacements.shape != (len(node_indices), 3):
        raise ValueError(
            f"internal handle displacements must have shape ({len(node_indices)}, 3)"
        )
    if np.any(node_indices < 0) or np.any(node_indices >= len(nodes)):
        raise ValueError("internal handle node index is outside the cage node range")
    if np.intersect1d(node_indices, boundary).size:
        raise ValueError("internal handles must not modify outer cage boundary nodes")
    if len(np.unique(node_indices)) != len(node_indices):
        raise ValueError("multiple internal handles resolve to the same cage node")
    if np.any(~np.isfinite(displacements)):
        raise ValueError("internal handle displacements must be finite")

    raw_modes = internal_handles.get("modes", internal_handles.get("mode"))
    raw_weights = internal_handles.get("weights", internal_handles.get("soft_weights"))
    if raw_modes is None:
        raw_modes = "soft" if raw_weights is not None else "dirichlet"
    modes = np.asarray(raw_modes)
    if modes.ndim == 0:
        modes = np.repeat(str(modes.item()), len(node_indices))
    modes = modes.astype(str).reshape(-1)
    if len(modes) != len(node_indices):
        raise ValueError("internal handle modes must be scalar or have one value per handle")
    mode_codes = np.empty(len(modes), dtype=np.int8)
    for index, mode in enumerate(modes):
        normalized_mode = mode.strip().lower()
        if normalized_mode in {"dirichlet", "hard"}:
            mode_codes[index] = 0
        elif normalized_mode in {"soft", "penalty"}:
            mode_codes[index] = 1
        else:
            raise ValueError(f"unsupported internal handle mode: {mode!r}")

    if raw_weights is None:
        weights = np.ones(len(node_indices), dtype=np.float64)
    else:
        weights = np.asarray(raw_weights, dtype=np.float64)
        if weights.ndim == 0:
            weights = np.full(len(node_indices), float(weights), dtype=np.float64)
        weights = weights.reshape(-1)
    if len(weights) != len(node_indices):
        raise ValueError("internal handle weights must be scalar or have one value per handle")
    if np.any((mode_codes == 1) & ((~np.isfinite(weights)) | (weights <= 0.0))):
        raise ValueError("soft internal handle weights must be finite and positive")
    weights[mode_codes == 0] = np.inf
    return {
        "node_indices": node_indices,
        "displacements": displacements,
        "mode_codes": mode_codes,
        "weights": weights,
    }


def _internal_handle_report(handles: dict[str, np.ndarray]) -> dict[str, Any]:
    modes = np.asarray(handles["mode_codes"], dtype=np.int8)
    hard = int(np.count_nonzero(modes == 0))
    soft = int(np.count_nonzero(modes == 1))
    digest = hashlib.sha256(_BETA_SOLVER_VERSION.encode("utf-8"))
    for key, dtype in (
        ("node_indices", np.int64),
        ("displacements", np.float64),
        ("mode_codes", np.int8),
        ("weights", np.float64),
    ):
        _digest_array(digest, f"internal_handle_{key}", np.asarray(handles[key], dtype=dtype))
    displacement = np.asarray(handles["displacements"], dtype=np.float64)
    maximum_displacement = (
        float(np.max(np.linalg.norm(displacement, axis=1))) if len(displacement) else 0.0
    )
    if hard and soft:
        mode = "harmonic_with_mixed_internal_handles"
    elif hard:
        mode = "harmonic_with_internal_dirichlet_handles"
    elif soft:
        mode = "harmonic_with_internal_soft_handles"
    else:
        mode = "harmonic_only"
    return {
        "volume_field_mode": mode,
        "harmonic_only": not bool(hard or soft),
        "internal_handle_count": hard + soft,
        "internal_dirichlet_handle_count": hard,
        "internal_soft_handle_count": soft,
        "internal_handle_digest": digest.hexdigest()[:24],
        "internal_handle_max_displacement_m": maximum_displacement,
    }


def _field_jacobian_diagnostics(
    nodes: np.ndarray,
    elements: np.ndarray,
    field: np.ndarray,
) -> dict[str, float | int]:
    tet = np.asarray(elements, dtype=np.int64)
    before = np.asarray(nodes, dtype=np.float64)[tet]
    after = (np.asarray(nodes, dtype=np.float64) + np.asarray(field, dtype=np.float64))[tet]
    det0 = np.linalg.det(before[:, 1:] - before[:, :1])
    det1 = np.linalg.det(after[:, 1:] - after[:, :1])
    if np.any(~np.isfinite(det0)) or np.any(np.abs(det0) <= 1.0e-18):
        raise RuntimeError("degenerate tetrahedron in neutral volume cage")
    ratio = det1 / det0
    inverted = int(np.count_nonzero((~np.isfinite(ratio)) | (ratio <= 0.0)))
    if inverted:
        raise RuntimeError(f"subject beta harmonic field flips {inverted} tetrahedra")
    minimum = float(np.min(ratio))
    if minimum < _MIN_JACOBIAN_RATIO:
        raise RuntimeError(
            "subject beta harmonic field is near-degenerate: "
            f"min Jacobian ratio {minimum:.6f}"
        )
    return {"minimum_jacobian_ratio": minimum, "inverted_tetrahedra": inverted}


def _jacobian_barrier_project(
    nodes: np.ndarray,
    elements: np.ndarray,
    deformed_nodes: np.ndarray,
    *,
    constrained_nodes: np.ndarray,
    minimum_ratio: float = _MIN_JACOBIAN_RATIO,
    iterations: int = 100,
) -> np.ndarray:
    """Project free cage nodes away from inverted/near-flat tetrahedra."""
    rest = np.asarray(nodes, dtype=np.float64)
    tet = np.asarray(elements, dtype=np.int64)
    current = np.asarray(deformed_nodes, dtype=np.float64).copy()
    constrained = np.zeros(len(rest), dtype=bool)
    constrained[np.asarray(constrained_nodes, dtype=np.int64)] = True
    rest_tet = rest[tet]
    det0 = np.linalg.det(rest_tet[:, 1:] - rest_tet[:, :1])
    orientation = np.sign(det0)
    minimum_volume = float(minimum_ratio) * np.abs(det0)
    target_volume = 1.1 * minimum_volume
    for _iteration in range(max(1, int(iterations))):
        points = current[tet]
        e1 = points[:, 1] - points[:, 0]
        e2 = points[:, 2] - points[:, 0]
        e3 = points[:, 3] - points[:, 0]
        det = np.einsum("ij,ij->i", e1, np.cross(e2, e3))
        oriented = orientation * det
        bad = np.flatnonzero(
            (~np.isfinite(oriented)) | (oriented < minimum_volume)
        )
        if not len(bad):
            return current
        p = points[bad]
        g1 = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])
        g2 = np.cross(p[:, 3] - p[:, 0], p[:, 1] - p[:, 0])
        g3 = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        gradients = np.stack((-(g1 + g2 + g3), g1, g2, g3), axis=1)
        gradients *= orientation[bad, None, None]
        free = ~constrained[tet[bad]]
        gradients *= free[..., None]
        denominator = np.sum(gradients * gradients, axis=(1, 2))
        if np.any(denominator <= 1.0e-24):
            raise RuntimeError(
                "subject beta boundary itself violates the Jacobian barrier"
            )
        deficit = target_volume[bad] - oriented[bad]
        correction = (
            deficit[:, None, None]
            * gradients
            / denominator[:, None, None]
        )
        accumulated = np.zeros_like(current)
        counts = np.zeros(len(current), dtype=np.float64)
        for corner in range(4):
            vertices = tet[bad, corner]
            np.add.at(accumulated, vertices, correction[:, corner])
            np.add.at(counts, vertices, free[:, corner].astype(np.float64))
        active = (counts > 0.0) & ~constrained
        current[active] += accumulated[active] / counts[active, None]
    raise RuntimeError("subject beta Jacobian barrier did not converge")


def _solve_harmonic_field(
    cage: dict[str, np.ndarray],
    *,
    surface_displacement: np.ndarray,
    internal_handles: dict[str, Any] | None = None,
) -> np.ndarray:
    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    elements = np.asarray(cage["elements"], dtype=np.int32)
    boundary = np.asarray(cage["boundary"], dtype=np.int64)
    triangles = np.asarray(cage["source_triangles"], dtype=np.int64)
    bary = np.asarray(cage["source_bary"], dtype=np.float64)
    boundary_values = np.sum(surface_displacement[triangles] * bary[:, :, None], axis=1)
    handles = _normalize_internal_handles(cage, internal_handles)
    handle_nodes = np.asarray(handles["node_indices"], dtype=np.int64)
    handle_values = np.asarray(handles["displacements"], dtype=np.float64)
    handle_modes = np.asarray(handles["mode_codes"], dtype=np.int8)
    hard = handle_modes == 0
    constrained = np.concatenate((boundary, handle_nodes[hard]))
    interior = np.setdiff1d(
        np.arange(len(nodes), dtype=np.int64), constrained, assume_unique=False
    )

    def linear_step(
        current_nodes: np.ndarray,
        step_boundary: np.ndarray,
        step_handles: np.ndarray,
    ) -> np.ndarray:
        from scipy.sparse import diags
        from scipy.sparse.linalg import splu

        constrained_values = np.concatenate(
            (step_boundary, step_handles[hard]),
            axis=0,
        )
        step = np.zeros((len(nodes), 3), dtype=np.float64)
        step[constrained] = constrained_values
        if not interior.size:
            return step
        stiffness = _tet_stiffness(current_nodes, elements)
        system = stiffness[interior][:, interior].tocsc()
        rhs = -(stiffness[interior][:, constrained] @ constrained_values)
        soft_nodes = handle_nodes[handle_modes == 1]
        if len(soft_nodes):
            local_index = np.full(len(nodes), -1, dtype=np.int64)
            local_index[interior] = np.arange(len(interior), dtype=np.int64)
            soft_local = local_index[soft_nodes]
            if np.any(soft_local < 0):
                raise RuntimeError("soft internal handle conflicts with a hard constraint")
            soft_weights = np.asarray(handles["weights"], dtype=np.float64)[handle_modes == 1]
            penalty = np.zeros(len(interior), dtype=np.float64)
            penalty[soft_local] = soft_weights
            system = system + diags(penalty, format="csc")
            rhs[soft_local] += (
                soft_weights[:, None] * step_handles[handle_modes == 1]
            )
        step[interior] = np.asarray(
            splu(system).solve(np.asarray(rhs, dtype=np.float64)), dtype=np.float64
        )
        return step

    field = linear_step(nodes, boundary_values, handle_values)
    try:
        _field_jacobian_diagnostics(nodes, elements, field)
        return field
    except RuntimeError:
        try:
            projected = _jacobian_barrier_project(
                nodes,
                elements,
                nodes + field,
                constrained_nodes=constrained,
            )
            projected_field = projected - nodes
            _field_jacobian_diagnostics(nodes, elements, projected_field)
            return projected_field
        except RuntimeError:
            pass

    # Large but legal beta changes can make a one-shot harmonic interpolation
    # fold a coarse interior tet even though the SMPL-X boundary itself remains
    # valid.  Continue on the deformed cage with adaptive rollback, preserving
    # the exact final boundary instead of scaling the requested beta.
    current = nodes.copy()
    remaining_boundary = boundary_values.copy()
    tet = np.asarray(elements, dtype=np.int64)
    base_tet = nodes[tet]
    base_det = np.linalg.det(base_tet[:, 1:] - base_tet[:, :1])
    for _iteration in range(64):
        current_handle_displacement = current[handle_nodes] - nodes[handle_nodes]
        remaining_handles = handle_values - current_handle_displacement
        boundary_remaining = (
            float(np.max(np.linalg.norm(remaining_boundary, axis=1)))
            if len(remaining_boundary)
            else 0.0
        )
        handle_remaining = (
            float(np.max(np.linalg.norm(remaining_handles, axis=1)))
            if len(remaining_handles)
            else 0.0
        )
        if max(boundary_remaining, handle_remaining) <= 1.0e-7:
            break
        fraction = 1.0
        while fraction >= 1.0 / 1024.0:
            step = linear_step(
                current,
                remaining_boundary * fraction,
                remaining_handles * fraction,
            )
            trial = current + step
            trial_tet = trial[tet]
            current_tet = current[tet]
            trial_det = np.linalg.det(trial_tet[:, 1:] - trial_tet[:, :1])
            current_det = np.linalg.det(
                current_tet[:, 1:] - current_tet[:, :1]
            )
            base_ratio = trial_det / base_det
            step_ratio = trial_det / current_det
            valid = (
                np.isfinite(base_ratio)
                & np.isfinite(step_ratio)
                & (base_ratio >= _MIN_JACOBIAN_RATIO)
                & (step_ratio >= _MIN_JACOBIAN_RATIO)
            )
            if np.all(valid):
                current = trial
                remaining_boundary *= 1.0 - fraction
                break
            fraction *= 0.5
        else:
            raise RuntimeError(
                "subject beta continuation cannot avoid tetrahedron inversion "
                "or minimum Jacobian-ratio violation"
            )
    else:
        raise RuntimeError("subject beta continuation did not reach the target boundary")
    field = current - nodes
    _field_jacobian_diagnostics(nodes, elements, field)
    return field


def _solve_harmonic_beta_basis(
    cage: dict[str, np.ndarray], surface_basis: np.ndarray
) -> np.ndarray:
    """Solve all SMPL-X beta directions with one sparse factorization."""
    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    elements = np.asarray(cage["elements"], dtype=np.int32)
    boundary = np.asarray(cage["boundary"], dtype=np.int64)
    triangles = np.asarray(cage["source_triangles"], dtype=np.int64)
    bary = np.asarray(cage["source_bary"], dtype=np.float64)
    shapedirs = np.asarray(surface_basis, dtype=np.float64)
    if shapedirs.ndim != 3 or shapedirs.shape[1] != 3:
        raise ValueError(f"SMPL-X shapedirs must be [V,3,B], got {shapedirs.shape}")
    boundary_basis = np.sum(
        shapedirs[triangles] * bary[:, :, None, None], axis=1
    )
    interior = np.setdiff1d(
        np.arange(len(nodes), dtype=np.int64), boundary, assume_unique=False
    )
    basis = np.zeros((len(nodes), 3, shapedirs.shape[2]), dtype=np.float64)
    basis[boundary] = boundary_basis
    if interior.size:
        stiffness = _tet_stiffness(nodes, elements)
        solved = _solve_interior_harmonic(
            stiffness,
            interior,
            boundary,
            boundary_basis.reshape(len(boundary), -1),
        )
        basis[interior] = solved.reshape(len(interior), 3, shapedirs.shape[2])
    return basis.astype(np.float32)


def _beta_volume_field(
    *,
    root: Path,
    cage: dict[str, np.ndarray],
    betas: np.ndarray,
    neutral_vertices: np.ndarray | None = None,
    neutral_faces: np.ndarray | None = None,
    subject_vertices: np.ndarray | None = None,
    subject_faces: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, str]:
    """Load/build the linear volume basis and combine it on CUDA when available."""
    weights_path = root / "smpl_canonical_weights.npz"
    weights = np.load(weights_path)
    if "shapedirs" not in weights.files:
        raise KeyError(f"{weights_path} does not contain SMPL-X shapedirs")
    shapedirs = np.asarray(weights["shapedirs"], dtype=np.float32)
    count = min(int(shapedirs.shape[2]), int(np.asarray(betas).size), 10)
    digest = _beta_basis_digest(
        cage,
        shapedirs,
        count,
        neutral_vertices=neutral_vertices,
        neutral_faces=neutral_faces,
        subject_vertices=subject_vertices,
        subject_faces=subject_faces,
    )
    shared_cache = root.parent / "volume_beta_basis_v2"
    shared_cache.mkdir(parents=True, exist_ok=True)
    cache_path = shared_cache / f"{digest}.npz"
    cache_hit = False
    basis: np.ndarray
    if cache_path.is_file():
        cached = np.load(cache_path)
        candidate = np.asarray(cached["field_basis"], dtype=np.float32)
        cached_digest = str(np.asarray(cached.get("digest", "")).reshape(-1)[0])
        if candidate.shape == (len(cage["nodes"]), 3, count) and cached_digest == digest:
            basis = candidate
            cache_hit = True
        else:
            basis = _solve_harmonic_beta_basis(cage, shapedirs[:, :, :count])
    else:
        basis = _solve_harmonic_beta_basis(cage, shapedirs[:, :, :count])
    if not cache_hit:
        np.savez_compressed(
            cache_path,
            field_basis=basis,
            digest=np.asarray([digest]),
            solver_version=np.asarray([_BETA_SOLVER_VERSION]),
        )

    beta = np.asarray(betas, dtype=np.float32).reshape(-1)[:count]
    backend = "numpy"
    try:
        import torch

        if torch.cuda.is_available():
            with torch.inference_mode():
                field_t = torch.tensordot(
                    torch.as_tensor(basis, device="cuda"),
                    torch.as_tensor(beta, device="cuda"),
                    dims=([2], [0]),
                )
                field = field_t.cpu().numpy().astype(np.float64)
            backend = "cuda"
        else:
            field = np.tensordot(basis, beta, axes=(2, 0)).astype(np.float64)
    except Exception:
        field = np.tensordot(basis, beta, axes=(2, 0)).astype(np.float64)
    return field, cache_hit, backend


def _beta_basis_digest(
    cage: dict[str, np.ndarray],
    shapedirs: np.ndarray,
    count: int,
    *,
    neutral_vertices: np.ndarray | None = None,
    neutral_faces: np.ndarray | None = None,
    subject_vertices: np.ndarray | None = None,
    subject_faces: np.ndarray | None = None,
    internal_handles: dict[str, Any] | None = None,
) -> str:
    """Digest every input that changes the neutral-to-beta volume solve."""
    digest = hashlib.sha256(_BETA_SOLVER_VERSION.encode("utf-8"))
    for key, dtype in (
        ("nodes", np.float32),
        ("elements", np.int32),
        ("boundary", np.int32),
        ("source_triangles", np.int32),
        ("source_bary", np.float32),
    ):
        if key not in cage:
            raise KeyError(f"volume cage is missing digest input {key!r}")
        _digest_array(digest, f"cage_{key}", np.asarray(cage[key], dtype=dtype))
    _digest_array(
        digest,
        "surface_shape_basis",
        np.asarray(shapedirs[:, :, :count], dtype=np.float32),
    )
    surfaces = (
        ("neutral_surface_vertices", neutral_vertices, np.float32),
        ("neutral_surface_faces", neutral_faces, np.int32),
        ("subject_surface_vertices", subject_vertices, np.float32),
        ("subject_surface_faces", subject_faces, np.int32),
    )
    for label, value, dtype in surfaces:
        if value is None:
            digest.update(f"{label}=none".encode("utf-8"))
        else:
            _digest_array(digest, label, np.asarray(value, dtype=dtype))
    handles = _normalize_internal_handles(cage, internal_handles)
    _digest_array(
        digest,
        "handle_node_indices",
        np.asarray(handles["node_indices"], dtype=np.int64),
    )
    _digest_array(
        digest,
        "handle_displacements",
        np.asarray(handles["displacements"], dtype=np.float64),
    )
    _digest_array(
        digest,
        "handle_mode_codes",
        np.asarray(handles["mode_codes"], dtype=np.int8),
    )
    _digest_array(
        digest,
        "handle_weights",
        np.asarray(handles["weights"], dtype=np.float64),
    )
    return digest.hexdigest()[:24]


def _tet_barycentric(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    tet = np.asarray(tetrahedra, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    if len(tet) != len(pts):
        raise ValueError("points and tetrahedra must have the same length")
    system = np.concatenate(
        [np.ones((len(tet), 4, 1), dtype=np.float64), tet], axis=2
    )
    rhs = np.concatenate(
        [np.ones((len(pts), 1), dtype=np.float64), pts], axis=1
    )
    return np.einsum("pj,pjk->pk", rhs, np.linalg.inv(system))


def _tet_boundary_faces(elements: np.ndarray) -> np.ndarray:
    tet = np.asarray(elements, dtype=np.int64)
    faces = np.concatenate(
        (tet[:, [0, 1, 2]], tet[:, [0, 1, 3]], tet[:, [0, 2, 3]], tet[:, [1, 2, 3]]), axis=0
    )
    canonical = np.sort(faces, axis=1)
    _unique, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse] == 1]


def _outside_cage_max_distance(
    points: np.ndarray,
    *,
    cage: dict[str, np.ndarray],
) -> float:
    """Measure an invalid outside query without inventing a displacement."""
    import igl

    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    boundary_faces = _tet_boundary_faces(np.asarray(cage["elements"], dtype=np.int64))
    squared, _face_index, _closest = igl.point_mesh_squared_distance(
        np.asarray(points, dtype=np.float64), nodes, boundary_faces
    )
    return float(np.sqrt(np.max(squared))) if len(squared) else 0.0


def _sample_field(
    points: np.ndarray,
    *,
    cage: dict[str, np.ndarray],
    field: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    import igl

    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    elements = np.asarray(cage["elements"], dtype=np.int64)
    tree = igl.AABB()
    tree.init(nodes, elements)
    element_index = np.asarray(
        igl.in_element(nodes, elements, np.asarray(points, dtype=np.float64), tree), dtype=np.int64
    )
    outside = element_index < 0
    displacement = np.zeros_like(points, dtype=np.float64)
    inside = ~outside
    if np.any(inside):
        selected = elements[element_index[inside]]
        bary = _tet_barycentric(points[inside], nodes[selected])
        displacement[inside] = np.sum(field[selected] * bary[:, :, None], axis=1)
    if np.any(outside):
        # Constant-normal extension of the harmonic boundary field. This is a
        # field sample, not a point projection: anatomy query positions remain
        # untouched and receive the barycentrically interpolated displacement
        # of their closest cage-boundary triangle. It covers small authored
        # excursions (teeth, cranial contents, superficial nerves) without SDF
        # pushing or silently freezing soft tissue outside the tetrahedra.
        boundary_faces = _tet_boundary_faces(elements)
        _squared, face_index, closest = igl.point_mesh_squared_distance(
            np.asarray(points, dtype=np.float64)[outside],
            nodes,
            boundary_faces,
        )
        selected_faces = boundary_faces[np.asarray(face_index, dtype=np.int64)]
        triangles = nodes[selected_faces]
        closest = np.asarray(closest, dtype=np.float64)
        v0 = triangles[:, 1] - triangles[:, 0]
        v1 = triangles[:, 2] - triangles[:, 0]
        v2 = closest - triangles[:, 0]
        d00 = np.sum(v0 * v0, axis=1)
        d01 = np.sum(v0 * v1, axis=1)
        d11 = np.sum(v1 * v1, axis=1)
        d20 = np.sum(v2 * v0, axis=1)
        d21 = np.sum(v2 * v1, axis=1)
        denominator = np.maximum(d00 * d11 - d01 * d01, 1.0e-16)
        weight1 = (d11 * d20 - d01 * d21) / denominator
        weight2 = (d00 * d21 - d01 * d20) / denominator
        bary = np.stack((1.0 - weight1 - weight2, weight1, weight2), axis=1)
        bary = np.clip(bary, 0.0, 1.0)
        bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1.0e-12)
        displacement[outside] = np.sum(
            np.asarray(field, dtype=np.float64)[selected_faces] * bary[:, :, None],
            axis=1,
        )
    return displacement, int(np.count_nonzero(outside)), outside


def _raise_subject_outside_query_error(
    asset: AnatomyRiggedAsset,
    *,
    points: np.ndarray,
    outside_mask: np.ndarray,
    protected: np.ndarray,
    cage: dict[str, np.ndarray],
) -> None:
    """Reject both soft and protected outliers with mesh/tissue diagnostics."""
    outside = np.asarray(outside_mask, dtype=bool)
    if not np.any(outside):
        return
    soft_outside = outside & ~protected
    protected_outside = outside & protected
    by_mesh: dict[str, dict[str, int]] = {}
    by_tissue: dict[str, int] = {}
    if asset.source_vertex_ranges is not None:
        ranges = asset.source_vertex_ranges
        if asset.source_mesh_names is not None:
            for name, (start, stop) in zip(asset.source_mesh_names, ranges):
                selected = outside[int(start) : int(stop)]
                if not np.any(selected):
                    continue
                selected_protected = protected[int(start) : int(stop)]
                by_mesh[str(name)] = {
                    "soft": int(np.count_nonzero(selected & ~selected_protected)),
                    "protected": int(np.count_nonzero(selected & selected_protected)),
                }
        if asset.source_tissues is not None:
            for tissue, (start, stop) in zip(asset.source_tissues, ranges):
                count = int(np.count_nonzero(outside[int(start) : int(stop)]))
                if count:
                    by_tissue[str(tissue)] = by_tissue.get(str(tissue), 0) + count
    maximum_distance = _outside_cage_max_distance(points[outside], cage=cage)
    raise RuntimeError(
        "subject beta volume cage excludes "
        f"{int(np.count_nonzero(outside))} anatomy vertices "
        f"(soft={int(np.count_nonzero(soft_outside))}, "
        f"protected={int(np.count_nonzero(protected_outside))}, "
        f"max distance={maximum_distance * 1000.0:.2f} mm): "
        f"meshes={dict(list(by_mesh.items())[:20])}, "
        f"tissues={dict(list(by_tissue.items())[:20])}"
    )


def apply_subject_beta_shape(
    asset: AnatomyRiggedAsset,
    *,
    canonical_dir: Path | str,
    config: dict[str, Any] | None = None,
    internal_handles: dict[str, Any] | None = None,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Apply a subject-beta harmonic volume field to anatomy and source rig."""
    root = Path(canonical_dir)
    neutral_v, faces = _load_obj(root / "smpl_canonical_tpose_neutral.obj")
    subject_v, subject_faces = _load_obj(root / "smpl_canonical_tpose.obj")
    if neutral_v.shape != subject_v.shape or not np.array_equal(faces, subject_faces):
        raise ValueError("neutral and subject SMPL-X surfaces must share exact topology")
    from .material_fit import bone_material_mask, cranial_material_mask, fit_articulated_rest
    from .source_skin_volume import _build_source_cage

    points = np.asarray(asset.vertices_rest, dtype=np.float64)
    protected = bone_material_mask(asset) | cranial_material_mask(asset)
    cage = _build_source_cage(
        neutral_v,
        faces,
        root / "neutral_volume_cage_v7_surface_only.npz",
    )
    cage = _attach_smplx_boundary_map(cage, neutral_v=neutral_v, neutral_f=faces)
    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    skeleton = json.loads(
        (root / "smpl_canonical_skeleton.json").read_text(encoding="utf-8")
    )
    _unused, _outside_count, preflight_outside = _sample_field(
        points, cage=cage, field=np.zeros_like(nodes)
    )
    extrapolation_max_m = (
        _outside_cage_max_distance(points[preflight_outside], cage=cage)
        if np.any(preflight_outside)
        else 0.0
    )
    maximum_extrapolation_m = float(
        (config or {}).get("maximum_harmonic_extrapolation_m", 0.075)
    )
    if extrapolation_max_m > maximum_extrapolation_m:
        _raise_subject_outside_query_error(
            asset,
            points=points,
            outside_mask=preflight_outside,
            protected=protected,
            cage=cage,
        )
    if internal_handles is None and config is not None:
        configured_handles = config.get("volume_internal_handles")
        if configured_handles is not None:
            if not isinstance(configured_handles, dict):
                raise ValueError("config volume_internal_handles must be a mapping")
            internal_handles = configured_handles
    if internal_handles is None:
        from .bone_handles import build_internal_joint_handles

        handle_weight = float(
            (config or {}).get("volume_internal_handle_weight", 25.0)
        )
        generated = build_internal_joint_handles(
            [str(name) for name in skeleton["joint_names"]],
            np.asarray(skeleton["rest_joints_neutral"], dtype=np.float64),
            np.asarray(skeleton["rest_joints_subject"], dtype=np.float64),
            weight=handle_weight,
        )
        internal_handles = {
            **generated.cache_payload(),
            "modes": "soft",
        }
    resolved_handles = _normalize_internal_handles(cage, internal_handles)
    normalized_handles: dict[str, Any] = {
        **resolved_handles,
        "_normalized": True,
    }
    handle_report = _internal_handle_report(resolved_handles)
    manifest_path = root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    beta_values = np.asarray(manifest.get("betas", []), dtype=np.float32).reshape(-1)
    basis_cache_hit = False
    basis_backend = "exact_surface_solve"
    displacement_cache_hit = False
    basis_digest = ""
    if beta_values.size and (root / "smpl_canonical_weights.npz").is_file():
        weights = np.load(root / "smpl_canonical_weights.npz")
        shapedirs = np.asarray(weights["shapedirs"], dtype=np.float64)
        count = min(shapedirs.shape[2], beta_values.size, 10)
        basis_digest = _beta_basis_digest(
            cage,
            shapedirs.astype(np.float32),
            count,
            neutral_vertices=neutral_v,
            neutral_faces=faces,
            subject_vertices=subject_v,
            subject_faces=subject_faces,
            internal_handles=normalized_handles,
        )
        predicted_surface_delta = np.tensordot(
            shapedirs[:, :, :count], beta_values[:count], axes=(2, 0)
        )
        surface_basis_error = float(
            np.max(np.linalg.norm((subject_v - neutral_v) - predicted_surface_delta, axis=1))
        )
        if surface_basis_error <= 1.0e-5 and not handle_report["internal_handle_count"]:
            field, basis_cache_hit, basis_backend = _beta_volume_field(
                root=root,
                cage=cage,
                betas=beta_values,
                neutral_vertices=neutral_v,
                neutral_faces=faces,
                subject_vertices=subject_v,
                subject_faces=subject_faces,
            )
        else:
            field = _solve_harmonic_field(
                cage,
                surface_displacement=subject_v - neutral_v,
                internal_handles=normalized_handles,
            )
            basis_backend = (
                "exact_surface_solve_internal_handles"
                if handle_report["internal_handle_count"]
                else "exact_surface_solve_basis_mismatch"
            )
    else:
        empty_basis = np.zeros((len(neutral_v), 3, 0), dtype=np.float32)
        basis_digest = _beta_basis_digest(
            cage,
            empty_basis,
            0,
            neutral_vertices=neutral_v,
            neutral_faces=faces,
            subject_vertices=subject_v,
            subject_faces=subject_faces,
            internal_handles=normalized_handles,
        )
        field = _solve_harmonic_field(
            cage,
            surface_displacement=subject_v - neutral_v,
            internal_handles=normalized_handles,
        )
        surface_basis_error = 0.0

    jacobian_report = _field_jacobian_diagnostics(
        nodes, np.asarray(cage["elements"], dtype=np.int64), field
    )
    minimum_jacobian_ratio = float(jacobian_report["minimum_jacobian_ratio"])
    boundary = np.asarray(cage["boundary"], dtype=np.int64)
    boundary_norm = np.linalg.norm(field[boundary], axis=1)
    boundary_rms = float(np.sqrt(np.mean(boundary_norm * boundary_norm)))
    boundary_max = float(np.max(boundary_norm))
    # Soft tissue (vessels/nerves/organs) follows one global harmonic volume
    # field from the neutral→subject surface.  Bones and cranial material stay
    # at authored positions here and are fitted by articulated material_fit.
    point_delta, outside_points, outside_mask = _sample_field(points, cage=cage, field=field)
    if not np.array_equal(outside_mask, preflight_outside):
        raise RuntimeError("harmonic query classification changed during beta solve")
    displacement_cache_hit = False
    harmonic_reference = points + point_delta
    shaped_vertices = harmonic_reference.copy()
    shaped_vertices[protected] = points[protected]

    harmonic_head = harmonic_mid = harmonic_tail = None
    harmonic_handle_outside = 0
    if asset.source_bone_names is not None and asset.target_bind_global is not None:
        base_head = np.asarray(
            asset.target_bone_head if asset.target_bone_head is not None else asset.source_bone_head,
            dtype=np.float64,
        )
        base_tail = np.asarray(
            asset.target_bone_tail if asset.target_bone_tail is not None else asset.source_bone_tail,
            dtype=np.float64,
        )
        base_mid = 0.5 * (base_head + base_tail)
        sampled_handles = []
        for handle_points in (base_head, base_mid, base_tail):
            handle_delta, outside_count, _outside = _sample_field(
                handle_points, cage=cage, field=field
            )
            sampled_handles.append(handle_points + handle_delta)
            harmonic_handle_outside += int(outside_count)
        harmonic_head, harmonic_mid, harmonic_tail = sampled_handles
    organ_constraint_reports: dict[str, Any] = {}
    if (
        asset.source_vertex_ranges is not None
        and asset.source_tissues is not None
        and asset.source_mesh_names
    ):
        from .soft_constraints import arap_volume_refine, unique_mesh_edges

        iterations = int((config or {}).get("organ_arap_iterations", 2))
        target_weight = float((config or {}).get("organ_arap_target_weight", 8.0))
        configured_volume_weight = float(
            (config or {}).get("organ_volume_weight", 0.25)
        )
        all_faces = np.asarray(asset.faces, dtype=np.int64)
        for mesh_name, (start, stop), tissue in zip(
            asset.source_mesh_names,
            np.asarray(asset.source_vertex_ranges, dtype=np.int64),
            asset.source_tissues,
        ):
            if str(tissue).lower() not in {"organ", "heart"}:
                continue
            local_faces = all_faces[
                np.all((all_faces >= int(start)) & (all_faces < int(stop)), axis=1)
            ] - int(start)
            if not len(local_faces):
                raise RuntimeError(
                    f"organ constraint mesh {mesh_name!r} has no local triangles"
                )
            edges = np.concatenate(
                (
                    local_faces[:, (0, 1)],
                    local_faces[:, (1, 2)],
                    local_faces[:, (2, 0)],
                ),
                axis=0,
            )
            edges.sort(axis=1)
            _unique, edge_count = np.unique(edges, axis=0, return_counts=True)
            watertight = bool(np.all(edge_count == 2))
            fitted, constraint_report = arap_volume_refine(
                points[int(start) : int(stop)],
                shaped_vertices[int(start) : int(stop)],
                local_faces,
                target_weight=target_weight,
                iterations=iterations,
                volume_weight=configured_volume_weight if watertight else 0.0,
            )
            shaped_vertices[int(start) : int(stop)] = fitted
            organ_constraint_reports[str(mesh_name)] = {
                **constraint_report,
                "watertight": watertight,
                "topology_edge_count": int(len(unique_mesh_edges(local_faces))),
            }
    meta = dict(asset.metadata or {})
    meta["shape_deformation"] = "tetgen_fem_harmonic_v7_soft_plus_articulated_bones"
    interim = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": shaped_vertices.astype(np.float32),
            "harmonic_reference_vertices": harmonic_reference.astype(np.float32),
            "harmonic_bone_head": None if harmonic_head is None else harmonic_head.astype(np.float32),
            "harmonic_bone_mid": None if harmonic_mid is None else harmonic_mid.astype(np.float32),
            "harmonic_bone_tail": None if harmonic_tail is None else harmonic_tail.astype(np.float32),
            "rest_joints": np.asarray(skeleton["rest_joints_subject"], dtype=np.float32),
            "inverse_bind": np.asarray(skeleton["inverse_bind"], dtype=np.float32),
            "metadata": meta,
        }
    )
    if bool(dict(config or {}).get("fast_publish", False)):
        from .anatomy_lbs import with_source_driver_coupling

        result = with_source_driver_coupling(interim)
        return result, {
            "backend": "fast_publish_subject_harmonic",
            "volume_solver_version": _BETA_SOLVER_VERSION,
            "skipped_material_rest_fit": True,
            "skipped_soft_follow": True,
            "reason": "preserve source LBS and joint links for live preview",
            "soft_rms_m": float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            np.square(np.asarray(shaped_vertices) - np.asarray(points)),
                            axis=1,
                        )
                    )
                )
            ),
            "soft_max_m": float(
                np.max(
                    np.linalg.norm(
                        np.asarray(shaped_vertices) - np.asarray(points),
                        axis=1,
                    )
                )
            ),
        }
    result, articulated_report = fit_articulated_rest(
        interim,
        canonical_dir=root,
        config=config,
        subject=True,
        stage="subject_beta",
        preserve_source_binding=True,
    )
    from .soft_constraints import (
        arap_volume_refine,
        limit_edge_strain,
        surface_barrier_refine,
    )

    barrier_vertices = np.asarray(result.vertices_rest, dtype=np.float64).copy()
    barrier_reports: dict[str, Any] = {}
    strain_reports: dict[str, Any] = {}
    all_faces = np.asarray(result.faces, dtype=np.int64)
    # Subject regularization is a neutral->beta constraint.  Using the authored
    # pre-registration mesh here made this stage independently satisfy a source
    # edge bound while still allowing severe neutral-to-subject graph changes.
    strain_reference = np.asarray(points, dtype=np.float64)
    for mesh_name, (start, stop), tissue in zip(
        result.source_mesh_names,
        np.asarray(result.source_vertex_ranges, dtype=np.int64),
        result.source_tissues,
    ):
        tissue_name = str(tissue)
        if tissue_name in {"vessel", "nerve", "connective_tissue"}:
            # Thin anatomy is finalized once by bake_station_soft_follow below.
            # In particular, it is never projected with an SDF barrier.
            continue
        if tissue_name not in {"vessel", "nerve"}:
            if tissue_name not in {
                "organ",
                "heart",
                "connective_tissue",
            }:
                continue
            start_i, stop_i = int(start), int(stop)
            if np.any(protected[start_i:stop_i]):
                continue
            is_organ = tissue_name in {"organ", "heart"}
            local_faces = all_faces[
                np.all(
                    (all_faces >= start_i) & (all_faces < stop_i),
                    axis=1,
                )
            ] - start_i
            arap_fitted, arap_report = arap_volume_refine(
                strain_reference[start_i:stop_i],
                barrier_vertices[start_i:stop_i],
                local_faces,
                target_weight=0.001 if is_organ else 0.01,
                iterations=30 if is_organ else 15,
                volume_weight=0.1 if is_organ else 0.0,
            )
            refined, strain_report = limit_edge_strain(
                strain_reference[start_i:stop_i],
                arap_fitted,
                local_faces,
                minimum_ratio=0.75,
                maximum_ratio=1.12,
                iterations=300,
            )
            if is_organ:
                refined, barrier_report = surface_barrier_refine(
                    refined,
                    local_faces,
                    subject_v,
                    subject_faces,
                    strain_reference_vertices=strain_reference[
                        start_i:stop_i
                    ],
                )
                barrier_reports[str(mesh_name)] = barrier_report
            barrier_vertices[start_i:stop_i] = refined
            strain_reports[str(mesh_name)] = {
                "arap": arap_report,
                "bounded_edges": strain_report,
            }
            continue
        start_i, stop_i = int(start), int(stop)
        local_faces = all_faces[
            np.all((all_faces >= start_i) & (all_faces < stop_i), axis=1)
        ] - start_i
        arap_fitted, arap_report = arap_volume_refine(
            strain_reference[start_i:stop_i],
            barrier_vertices[start_i:stop_i],
            local_faces,
            target_weight=0.01,
            iterations=15,
            volume_weight=0.0,
        )
        prestrained, strain_report = limit_edge_strain(
            strain_reference[start_i:stop_i],
            arap_fitted,
            local_faces,
            minimum_ratio=0.85,
            maximum_ratio=1.08,
            iterations=100,
        )
        strain_reports[str(mesh_name)] = {
            "arap": arap_report,
            "bounded_edges": strain_report,
        }
        refined, barrier_report = surface_barrier_refine(
            prestrained,
            local_faces,
            subject_v,
            subject_faces,
            strain_reference_vertices=strain_reference[start_i:stop_i],
        )
        barrier_vertices[start_i:stop_i] = refined
        barrier_reports[str(mesh_name)] = barrier_report
    result = type(result)(
        **{
            **result.__dict__,
            "vertices_rest": barrier_vertices.astype(np.float32),
        }
    )
    from .soft_follow import bake_station_soft_follow

    result, soft_follow_report = bake_station_soft_follow(
        result,
        skin_vertices=subject_v,
        skin_faces=subject_faces,
    )
    soft_mask = ~protected
    constrained_delta = shaped_vertices - points
    soft_norm = (
        np.linalg.norm(constrained_delta[soft_mask], axis=1)
        if np.any(soft_mask)
        else np.zeros(1)
    )
    soft_rms = float(np.sqrt(np.mean(soft_norm * soft_norm)))
    soft_max = float(np.max(soft_norm))
    if np.any(~np.isfinite(soft_norm)):
        raise RuntimeError("subject beta volume field produced non-finite soft displacement")
    if np.any(soft_mask) and boundary_max > 1.0e-7 and soft_max < 1.0e-7:
        raise RuntimeError(
            "subject beta volume field moved the boundary but produced no measurable soft displacement"
        )
    return result, {
        "backend": "tetgen_fem_harmonic_v7_soft_plus_articulated_bones",
        "volume_solver_version": _BETA_SOLVER_VERSION,
        "beta_basis_cache_hit": bool(basis_cache_hit),
        "beta_displacement_cache_hit": bool(displacement_cache_hit),
        "beta_basis_digest": basis_digest,
        "volume_cache_digest": basis_digest,
        "soft_beta_transport": "global_harmonic_volume_field",
        "harmonic_handle_outside_count": int(harmonic_handle_outside),
        "harmonic_extrapolated_vertices": int(outside_points),
        "harmonic_extrapolation_max_m": float(extrapolation_max_m),
        "soft_follow": soft_follow_report,
        "beta_basis_combine_backend": str(basis_backend),
        "surface_basis_error_m": float(surface_basis_error),
        "tetra_vertices": int(len(cage["nodes"])),
        "tetrahedra": int(len(cage["elements"])),
        "mean_displacement_m": float(np.mean(soft_norm)),
        "soft_displacement_rms_m": soft_rms,
        "max_displacement_m": soft_max,
        "soft_displacement_max_m": soft_max,
        "boundary_displacement_rms_m": boundary_rms,
        "boundary_displacement_max_m": boundary_max,
        "outside_query_count": int(outside_points),
        "outside_soft_material_count": 0,
        "outside_protected_material_count": 0,
        "outside_query_by_tissue": {},
        "max_outside_cage_distance_m": 0.0,
        "minimum_jacobian_ratio": minimum_jacobian_ratio,
        "inverted_tetrahedra": int(jacobian_report["inverted_tetrahedra"]),
        "protected_material_vertices": int(np.count_nonzero(protected)),
        "organ_constraints": organ_constraint_reports,
        "soft_edge_strain_regularizer": strain_reports,
        "surface_barrier_regularizer": barrier_reports,
        **handle_report,
        "articulated_rest_fit": articulated_report,
    }


def apply_material_bounded_soft_volume(
    asset: Any,
    *,
    canonical_dir: Path | str,
    samples_per_bone_mesh: int = 24,
) -> tuple[Any, dict[str, Any]]:
    """Re-solve soft anatomy between fitted bone surfaces and target skin.

    ``harmonic_reference_vertices`` is the all-harmonic first-stage result.
    Final material-fit bone surfaces provide interior Dirichlet displacements;
    the fitted SMPL-X skin has zero displacement on the outer boundary.  Soft
    anatomy is sampled exactly once from this field.  There is no LBS residual,
    station override, closest-surface push or SDF projection in this step.
    """
    from scipy.spatial import cKDTree
    from .source_skin_volume import _build_source_cage

    if asset.harmonic_reference_vertices is None:
        raise ValueError("material-bounded volume requires harmonic_reference_vertices")
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("material-bounded volume requires mesh ranges/tissues")
    reference = np.asarray(asset.harmonic_reference_vertices, dtype=np.float64)
    final = np.asarray(asset.vertices_rest, dtype=np.float64)
    if reference.shape != final.shape:
        raise ValueError("harmonic reference must match final anatomy topology")

    # Blender binding defines the local elastic target: transform each
    # reference point by its controlling bones, then blend by the authored
    # weights.  This target is not applied directly; it is mixed with the
    # multi-boundary volume solution below, so influence decays smoothly toward
    # the fixed skin instead of shearing at weight seams.
    weighted_bone_delta: np.ndarray | None = None
    if (
        asset.harmonic_bone_head is not None
        and asset.harmonic_bone_tail is not None
        and asset.target_bone_head is not None
        and asset.target_bone_tail is not None
        and asset.driver_indices is not None
        and asset.driver_weights is not None
    ):
        from .material_fit import _rotation_between

        reference_head = np.asarray(asset.harmonic_bone_head, dtype=np.float64)
        reference_tail = np.asarray(asset.harmonic_bone_tail, dtype=np.float64)
        fitted_head = np.asarray(asset.target_bone_head, dtype=np.float64)
        fitted_tail = np.asarray(asset.target_bone_tail, dtype=np.float64)
        rotations = np.stack(
            [
                _rotation_between(
                    reference_tail[index] - reference_head[index],
                    fitted_tail[index] - fitted_head[index],
                )
                for index in range(len(reference_head))
            ],
            axis=0,
        )
        translations = fitted_head - np.einsum(
            "bij,bj->bi", rotations, reference_head
        )
        drivers = np.asarray(asset.driver_indices, dtype=np.int64)
        weights = np.asarray(asset.driver_weights, dtype=np.float64)
        transformed = np.einsum(
            "nkij,nj->nki", rotations[drivers], reference
        ) + translations[drivers]
        weighted_target = np.sum(transformed * weights[:, :, None], axis=1)
        weighted_bone_delta = weighted_target - reference
        weighted_norm = np.linalg.norm(weighted_bone_delta, axis=1)
        maximum_weighted_delta = 0.03
        scale = np.minimum(
            1.0, maximum_weighted_delta / np.maximum(weighted_norm, 1.0e-12)
        )
        weighted_bone_delta *= scale[:, None]

    root = Path(canonical_dir)
    subject_v, subject_faces = _load_obj(root / "smpl_canonical_tpose.obj")
    cage = _build_source_cage(
        subject_v,
        subject_faces,
        # This is an ambient solve domain, not the skin boundary itself.  The
        # first (Skin_Glass) harmonic stage can legitimately place source
        # material a few centimetres beyond a non-watertight SMPL-X shell;
        # retain it in the solve and constrain the actual SMPL-X skin below.
        root / "material_bounded_subject_cage_v3_margin10.npz",
        dilation_iterations=10,
    )
    cage = _attach_smplx_boundary_map(
        cage, neutral_v=subject_v, neutral_f=subject_faces
    )
    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    boundary = np.asarray(cage["boundary"], dtype=np.int64)
    interior = np.setdiff1d(
        np.arange(len(nodes), dtype=np.int64), boundary, assume_unique=False
    )
    if not len(interior):
        raise RuntimeError("material-bounded volume cage has no interior nodes")

    handle_points: list[np.ndarray] = []
    handle_displacements: list[np.ndarray] = []
    bone_mesh_count = 0
    requested_samples = max(4, int(samples_per_bone_mesh))
    for (start, stop), tissue in zip(
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        asset.source_tissues,
    ):
        if str(tissue).lower() != "bone" or int(stop) <= int(start):
            continue
        start_i, stop_i = int(start), int(stop)
        count = min(requested_samples, stop_i - start_i)
        local = np.unique(
            np.linspace(0, stop_i - start_i - 1, count, dtype=np.int64)
        )
        indices = start_i + local
        displacement = final[indices] - reference[indices]
        active = np.linalg.norm(displacement, axis=1) > 1.0e-7
        if np.any(active):
            handle_points.append(reference[indices][active])
            handle_displacements.append(displacement[active])
            bone_mesh_count += 1
    if not handle_points:
        return asset, {
            "backend": "material_bounded_harmonic_v1",
            "skipped": True,
            "reason": "no fitted bone residual",
        }
    raw_points = np.concatenate(handle_points, axis=0)
    raw_displacements = np.concatenate(handle_displacements, axis=0)
    _zero, handle_outside_count, handle_outside = _sample_field(
        raw_points, cage=cage, field=np.zeros_like(nodes)
    )
    raw_points = raw_points[~handle_outside]
    raw_displacements = raw_displacements[~handle_outside]
    if not len(raw_points):
        raise RuntimeError("all material bone handles lie outside the subject cage")

    # Aggregate bone-surface samples which resolve to the same tet node.
    _distance, nearest = cKDTree(nodes[interior]).query(raw_points, k=1)
    nearest = interior[np.asarray(nearest, dtype=np.int64)]
    unique_nodes, inverse = np.unique(nearest, return_inverse=True)
    displacement_sum = np.zeros((len(unique_nodes), 3), dtype=np.float64)
    displacement_count = np.zeros(len(unique_nodes), dtype=np.float64)
    np.add.at(displacement_sum, inverse, raw_displacements)
    np.add.at(displacement_count, inverse, 1.0)
    handle_delta = displacement_sum / displacement_count[:, None]
    # The target skin is an explicit zero-displacement Dirichlet shell inside
    # the ambient cage.  It replaces the old post-hoc SDF projection: soft
    # material is affected only by this one multi-boundary harmonic solve.
    skin_nearest = cKDTree(nodes[interior]).query(subject_v, k=1)[1]
    skin_nodes = np.unique(interior[np.asarray(skin_nearest, dtype=np.int64)])
    skin_nodes = skin_nodes[~np.isin(skin_nodes, unique_nodes)]
    constrained_nodes = np.concatenate((skin_nodes, unique_nodes))
    constrained_delta = np.concatenate(
        (np.zeros((len(skin_nodes), 3), dtype=np.float64), handle_delta), axis=0
    )
    normalized_handles = {
        "_normalized": True,
        "node_indices": constrained_nodes,
        "displacements": constrained_delta,
        "mode_codes": np.concatenate(
            (
                np.zeros(len(skin_nodes), dtype=np.int8),
                np.ones(len(unique_nodes), dtype=np.int8),
            )
        ),
        "weights": np.concatenate(
            (
                np.full(len(skin_nodes), np.inf, dtype=np.float64),
                np.full(len(unique_nodes), 0.05, dtype=np.float64),
            )
        ),
    }
    field = _solve_harmonic_field(
        cage,
        surface_displacement=np.zeros_like(subject_v, dtype=np.float64),
        internal_handles=normalized_handles,
    )
    point_delta, outside_count, outside = _sample_field(
        reference, cage=cage, field=field
    )
    soft = np.zeros(len(final), dtype=bool)
    for (start, stop), tissue in zip(
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        asset.source_tissues,
    ):
        tissue_name = str(tissue).lower()
        if tissue_name != "bone":
            soft[int(start) : int(stop)] = True
    from .material_fit import cranial_material_mask

    raw_cranial = cranial_material_mask(asset)
    cranial = np.zeros(len(final), dtype=bool)
    for start, stop in np.asarray(asset.source_vertex_ranges, dtype=np.int64):
        start_i, stop_i = int(start), int(stop)
        if float(np.mean(raw_cranial[start_i:stop_i])) >= 0.90:
            cranial[start_i:stop_i] = True
    soft &= ~cranial
    soft_outside = outside & soft
    soft_outside_max_m = (
        _outside_cage_max_distance(reference[soft_outside], cage=cage)
        if np.any(soft_outside)
        else 0.0
    )
    if weighted_bone_delta is not None:
        inside_soft = soft & ~outside
        point_delta[inside_soft] = (
            0.5 * point_delta[inside_soft]
            + 0.5 * weighted_bone_delta[inside_soft]
        )
        # Hand/foot openings are outside only because the SMPL-X shell is not
        # watertight.  Continue with the Blender-weight target there rather
        # than freezing the distal vessels or projecting them to the cage.
        point_delta[soft_outside] = weighted_bone_delta[soft_outside]
    else:
        point_delta[soft_outside] = 0.0
    output = final.copy()
    output[soft] = reference[soft] + point_delta[soft]
    displacement = np.linalg.norm(output[soft] - reference[soft], axis=1)
    metadata = dict(asset.metadata or {})
    metadata.update(
        {
            "soft_volume_backend": "material_bounded_harmonic_v1",
            "soft_surface_sdf": "disabled",
            "soft_bone_residual_follow": "inner_dirichlet_volume_field",
        }
    )
    result = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": output.astype(np.float32),
            "soft_follow_driver_indices": None,
            "soft_follow_driver_weights": None,
            "soft_follow_stations": None,
            "soft_follow_strength": None,
            "soft_component_ids": None,
            "source_mesh_follow_modes": None,
            "metadata": metadata,
        }
    )
    return result, {
        "backend": "material_bounded_harmonic_v1",
        "outer_boundary": "subject_smplx_zero_dirichlet",
        "skin_dirichlet_node_count": int(len(skin_nodes)),
        "inner_boundary": "material_fit_bone_surface_soft_springs",
        "bone_spring_weight": 0.05,
        "bone_mesh_count": int(bone_mesh_count),
        "raw_handle_count": int(len(raw_points)),
        "handle_node_count": int(len(unique_nodes)),
        "outside_handle_count": int(handle_outside_count),
        "outside_soft_count": int(np.count_nonzero(soft_outside)),
        "outside_soft_max_m": float(soft_outside_max_m),
        "outside_soft_transport": "blender_weight_elastic_target" if np.any(soft_outside) else "none",
        "blender_weight_elastic_target": bool(weighted_bone_delta is not None),
        "soft_vertex_count": int(np.count_nonzero(soft)),
        "soft_residual_mean_m": float(np.mean(displacement)),
        "soft_residual_max_m": float(np.max(displacement)),
        "sdf_projection": False,
    }
