"""Offline Skin_Glass -> neutral SMPL-X volumetric registration.

The Blender skin is used only as a material boundary.  Internal anatomy is
transported by one continuous harmonic volume field; no anatomy vertex is
individually projected or clamped to the SMPL-X surface.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset
from .material_fit import bone_material_mask, cranial_material_mask
from .shape_volume import (
    _load_obj,
    _outside_cage_max_distance,
    _sample_field,
    _tet_stiffness,
)
from .soft_constraints import arap_volume_refine


_CAGE_VERSION = "source_skin_volume_v8_semantic_surface_map"
_SEMANTIC_MAP_VERSION = "skin_glass_smplx55_fixed_map_v1"
_MIN_JACOBIAN_RATIO = 0.05
_MAX_FINAL_SURFACE_RMS_M = 0.03
_MAX_FINAL_SURFACE_DISTANCE_M = 0.10
_MAX_BOUNDARY_DISPLACEMENT_M = 0.50
_MIN_REGISTRATION_PROGRESS_M = 1.0e-7


def _transport_sampled_material(
    query: np.ndarray,
    delta: np.ndarray,
    *,
    protected: np.ndarray,
    outside: np.ndarray,
) -> np.ndarray:
    """Apply a sampled volume field while keeping rigid material untouched.

    A sampled field is not a diagnostic: every non-rigid material point must
    actually receive its displacement.  Keeping this small invariant in a
    separately tested helper prevents the v5.8 regression where the expensive
    harmonic solve completed successfully and its result was then discarded.
    """
    points = np.asarray(query, dtype=np.float64)
    displacement = np.asarray(delta, dtype=np.float64)
    rigid = np.asarray(protected, dtype=bool).reshape(-1)
    outside_mask = np.asarray(outside, dtype=bool).reshape(-1)
    if points.shape != displacement.shape or points.shape[0] != len(rigid):
        raise ValueError("material points, displacement and masks must have matching lengths")
    if outside_mask.shape != rigid.shape:
        raise ValueError("outside and protected masks must have matching lengths")
    if np.any(outside_mask):
        soft_outside = outside_mask & ~rigid
        protected_outside = outside_mask & rigid
        raise ValueError(
            "volume cage excludes "
            f"{int(np.count_nonzero(outside_mask))} material vertices "
            f"(soft={int(np.count_nonzero(soft_outside))}, "
            f"protected={int(np.count_nonzero(protected_outside))})"
        )
    mapped = points + displacement
    mapped[rigid] = points[rigid]
    return mapped


def _signature(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Digest the solver and the complete authored source-surface topology."""
    digest = hashlib.sha256(_CAGE_VERSION.encode("utf-8"))
    digest.update(b"surface_vertices_float32")
    digest.update(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
    digest.update(b"surface_faces_int32")
    digest.update(np.ascontiguousarray(faces, dtype=np.int32).tobytes())
    digest.update(b"fixed_voxel_margin=1")
    return digest.hexdigest()


def _voxel_union(vertices: np.ndarray, faces: np.ndarray):
    """Repair the authored skin into a closed domain with one-voxel margin."""
    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, process=True)
    longest = float(np.max(mesh.extents))
    if not np.isfinite(longest) or longest <= 0.0:
        raise RuntimeError("Skin_Glass has invalid dimensions")
    pitch = longest / 180.0
    from scipy import ndimage
    import trimesh

    grid = mesh.voxelized(pitch).fill()
    # Skin_Glass contains small topological tunnels around facial openings.
    # A one-voxel closing removes those tunnels without changing the exterior
    # envelope.  Padding prevents scipy's closing from shrinking extremities
    # that touch the voxel-grid boundary.
    base = np.asarray(grid.matrix, dtype=bool)
    transform = np.asarray(grid.transform, dtype=np.float64).copy()
    padding = 3
    lower = np.full(3, -padding, dtype=np.int64)
    upper = np.asarray(base.shape, dtype=np.int64) + padding
    occupancy = np.zeros(tuple((upper - lower).tolist()), dtype=bool)
    shift = -lower
    occupancy[
        shift[0] : shift[0] + base.shape[0],
        shift[1] : shift[1] + base.shape[1],
        shift[2] : shift[2] + base.shape[2],
    ] = base
    # One voxel of material margin keeps points that lie exactly on a sampled
    # boundary inside the tetrahedral domain after marching-cubes rounding.
    occupancy = ndimage.binary_dilation(occupancy, iterations=1)
    occupancy = ndimage.binary_closing(occupancy, iterations=1)
    occupancy = ndimage.binary_fill_holes(occupancy)
    transform[:3, 3] += transform[:3, :3] @ lower.astype(np.float64)
    closed = trimesh.voxel.VoxelGrid(occupancy, transform=transform)
    surface = closed.marching_cubes
    surface.apply_transform(transform)
    surface.remove_unreferenced_vertices()
    surface.fix_normals()
    surface = trimesh.Trimesh(surface.vertices, surface.faces, process=True)
    if not surface.is_watertight or not surface.is_volume:
        raise RuntimeError("Skin_Glass voxel union is not a closed volume")
    return surface, pitch


def _barycentric_surface_map(
    points: np.ndarray,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    mapped_vertices: np.ndarray,
) -> np.ndarray:
    """Sample a fixed source-topology surface map at arbitrary nearby points."""
    import igl

    _squared, face_index, closest = igl.point_mesh_squared_distance(
        np.asarray(points, dtype=np.float64),
        np.asarray(source_vertices, dtype=np.float64),
        np.asarray(source_faces, dtype=np.int32),
    )
    triangles = np.asarray(source_vertices, dtype=np.float64)[
        np.asarray(source_faces, dtype=np.int64)[face_index]
    ]
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    q = np.asarray(closest, dtype=np.float64) - triangles[:, 0]
    aa = np.einsum("ij,ij->i", a, a)
    ab = np.einsum("ij,ij->i", a, b)
    bb = np.einsum("ij,ij->i", b, b)
    qa = np.einsum("ij,ij->i", q, a)
    qb = np.einsum("ij,ij->i", q, b)
    denominator = aa * bb - ab * ab
    denominator = np.where(np.abs(denominator) > 1.0e-16, denominator, 1.0)
    w1 = (bb * qa - ab * qb) / denominator
    w2 = (aa * qb - ab * qa) / denominator
    bary = np.column_stack((1.0 - w1 - w2, w1, w2))
    bary = np.clip(bary, 0.0, 1.0)
    bary /= np.maximum(np.sum(bary, axis=1, keepdims=True), 1.0e-12)
    return np.sum(
        np.asarray(mapped_vertices, dtype=np.float64)[
            np.asarray(source_faces, dtype=np.int64)[face_index]
        ]
        * bary[:, :, None],
        axis=1,
    )


def _semantic_map_digest(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_weights: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    target_weights: np.ndarray,
) -> str:
    digest = hashlib.sha256(_SEMANTIC_MAP_VERSION.encode("utf-8"))
    for value, dtype in (
        (source_vertices, np.float32),
        (source_faces, np.int32),
        (source_weights, np.float32),
        (target_vertices, np.float32),
        (target_faces, np.int32),
        (target_weights, np.float32),
    ):
        digest.update(np.ascontiguousarray(value, dtype=dtype).tobytes())
    return digest.hexdigest()


def _fixed_semantic_skin_map(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_weights: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    target_weights: np.ndarray,
    *,
    joint_names: list[str],
    cache_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map Skin_Glass to SMPL-X once using joint-material coordinates.

    Spatial nearest points alone confuse facing surfaces at the axilla, groin,
    fingers and neck.  Candidate target triangles are therefore selected in a
    joint-weight + position feature space, with an explicit left/right barrier.
    The selected triangle and point are frozen and cached; later harmonic solves
    never repeat ICP or change correspondence.
    """
    from scipy.spatial import cKDTree
    import trimesh

    source = np.asarray(source_vertices, dtype=np.float64)
    source_w = np.asarray(source_weights, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    faces = np.asarray(target_faces, dtype=np.int64)
    target_w = np.asarray(target_weights, dtype=np.float64)
    if source_w.shape != (len(source), len(joint_names)):
        raise ValueError("Skin_Glass semantic weights do not match source vertices/joints")
    if target_w.shape != (len(target), len(joint_names)):
        raise ValueError("SMPL-X semantic weights do not match target vertices/joints")
    digest = _semantic_map_digest(
        source, source_faces, source_w, target, faces, target_w
    )
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["digest"].item()) == digest:
                mapped = np.asarray(cached["mapped_vertices"], dtype=np.float64)
                if mapped.shape == source.shape:
                    return mapped, {
                        "backend": _SEMANTIC_MAP_VERSION,
                        "cache_hit": True,
                        "digest": digest,
                        "unique_target_faces": int(cached["unique_target_faces"].item()),
                        "side_mismatch_count": int(cached["side_mismatch_count"].item()),
                        "semantic_rms": float(cached["semantic_rms"].item()),
                    }

    triangle_vertices = target[faces]
    face_centers = np.mean(triangle_vertices, axis=1)
    face_weights = np.mean(target_w[faces], axis=1)
    # 12 cm spatial and 0.4 joint-weight feature scales keep candidates local
    # while making two spatially close but anatomically different surfaces far.
    tree_features = np.concatenate(
        (face_centers / 0.12, face_weights / 0.40), axis=1
    )
    source_features = np.concatenate((source / 0.12, source_w / 0.40), axis=1)
    tree = cKDTree(tree_features)
    _distance, candidates = tree.query(source_features, k=48, workers=-1)
    candidates = np.asarray(candidates, dtype=np.int64).reshape(len(source), -1)
    candidate_triangles = triangle_vertices[candidates.reshape(-1)]
    repeated_source = np.repeat(source, candidates.shape[1], axis=0)
    closest = trimesh.triangles.closest_point(
        candidate_triangles, repeated_source
    ).reshape(len(source), candidates.shape[1], 3)
    candidate_weights = face_weights[candidates]
    spatial_cost = np.sum((closest - source[:, None, :]) ** 2, axis=2)
    semantic_cost = np.sum(
        (candidate_weights - source_w[:, None, :]) ** 2, axis=2
    )
    cost = spatial_cost + (0.22**2) * semantic_cost

    left_ids = np.asarray(
        [index for index, name in enumerate(joint_names) if str(name).startswith("left_")],
        dtype=np.int64,
    )
    right_ids = np.asarray(
        [index for index, name in enumerate(joint_names) if str(name).startswith("right_")],
        dtype=np.int64,
    )
    source_side = np.sign(
        np.sum(source_w[:, left_ids], axis=1)
        - np.sum(source_w[:, right_ids], axis=1)
    )
    candidate_side = np.sign(
        np.sum(candidate_weights[:, :, left_ids], axis=2)
        - np.sum(candidate_weights[:, :, right_ids], axis=2)
    )
    source_side_strength = np.abs(
        np.sum(source_w[:, left_ids], axis=1)
        - np.sum(source_w[:, right_ids], axis=1)
    )
    wrong_side = (
        (source_side_strength[:, None] > 0.20)
        & (candidate_side != 0.0)
        & (candidate_side != source_side[:, None])
    )
    cost[wrong_side] = np.inf
    selected_slot = np.argmin(cost, axis=1)
    selected_face = candidates[np.arange(len(source)), selected_slot]
    mapped = closest[np.arange(len(source)), selected_slot]
    selected_weights = face_weights[selected_face]
    selected_side = np.sign(
        np.sum(selected_weights[:, left_ids], axis=1)
        - np.sum(selected_weights[:, right_ids], axis=1)
    )
    side_mismatch = int(
        np.count_nonzero(
            (source_side_strength > 0.20)
            & (selected_side != 0.0)
            & (selected_side != source_side)
        )
    )
    semantic_rms = float(
        np.sqrt(np.mean(np.sum((selected_weights - source_w) ** 2, axis=1)))
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        digest=np.asarray(digest),
        mapped_vertices=mapped.astype(np.float32),
        selected_faces=selected_face.astype(np.int32),
        unique_target_faces=np.asarray(len(np.unique(selected_face)), dtype=np.int32),
        side_mismatch_count=np.asarray(side_mismatch, dtype=np.int32),
        semantic_rms=np.asarray(semantic_rms, dtype=np.float64),
    )
    return mapped, {
        "backend": _SEMANTIC_MAP_VERSION,
        "cache_hit": False,
        "digest": digest,
        "unique_target_faces": int(len(np.unique(selected_face))),
        "side_mismatch_count": side_mismatch,
        "semantic_rms": semantic_rms,
    }


def _build_source_cage(
    vertices: np.ndarray,
    faces: np.ndarray,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    signature = _signature(vertices, faces)
    if cache_path.is_file():
        data = np.load(cache_path)
        cached = str(np.asarray(data.get("signature", "")).reshape(-1)[0])
        if cached == signature:
            return {key: np.asarray(data[key]) for key in data.files}

    import tetgen

    surface, pitch = _voxel_union(vertices, faces)
    generator = tetgen.TetGen(
        np.asarray(surface.vertices, dtype=np.float64),
        np.asarray(surface.faces, dtype=np.int32),
    )
    meshing_backend = "tetgen_quality"
    try:
        nodes, elements, _attributes, _markers = generator.tetrahedralize(
            order=1,
            mindihedral=5.0,
            minratio=2.0,
            maxvolume=float(np.max(surface.extents) ** 3 / 4000.0),
            quiet=True,
        )
    except RuntimeError:
        # TetGen's quality refinement can fail in split_subface on the valid,
        # high-genus voxel-union skin.  PLC tetrahedralization without Steiner
        # refinement is deterministic for this surface and still gives a
        # conforming piecewise-linear volume field.  Degenerate Delaunay cells
        # are removed explicitly below.
        generator = tetgen.TetGen(
            np.asarray(surface.vertices, dtype=np.float64),
            np.asarray(surface.faces, dtype=np.int32),
        )
        nodes, elements, _attributes, _markers = generator.tetrahedralize(
            order=1, quality=False, quiet=True
        )
        meshing_backend = "tetgen_plc_no_refinement"
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int32)
    tet = nodes[elements]
    determinant = np.linalg.det(tet[:, 1:] - tet[:, :1])
    valid = np.abs(determinant) > 1.0e-16
    elements = elements[valid]
    if not len(elements):
        raise RuntimeError("source volume cage contains no non-degenerate tetrahedra")
    boundary = np.unique(np.asarray(generator.trifaces, dtype=np.int32).reshape(-1))
    boundary_faces = np.asarray(generator.trifaces, dtype=np.int32).reshape(-1, 3)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        nodes=nodes.astype(np.float32),
        elements=elements,
        boundary=boundary.astype(np.int32),
        boundary_faces=boundary_faces.astype(np.int32),
        signature=np.asarray([signature]),
        voxel_pitch=np.asarray([pitch], dtype=np.float32),
        meshing_backend=np.asarray([meshing_backend]),
        removed_degenerate_tetrahedra=np.asarray([np.count_nonzero(~valid)], dtype=np.int32),
    )
    return {
        "nodes": nodes,
        "elements": elements,
        "boundary": boundary,
        "boundary_faces": boundary_faces,
        "signature": np.asarray([signature]),
        "voxel_pitch": np.asarray([pitch], dtype=np.float32),
        "meshing_backend": np.asarray([meshing_backend]),
        "removed_degenerate_tetrahedra": np.asarray([np.count_nonzero(~valid)], dtype=np.int32),
    }


def _topology_preserving_cage_registration(
    nodes: np.ndarray,
    elements: np.ndarray,
    boundary: np.ndarray,
    boundary_faces: np.ndarray,
    target: np.ndarray,
    target_faces: np.ndarray,
    *,
    fixed_target: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit the closed cage and reject no-op or low-quality registrations."""
    import igl
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import factorized

    original = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    boundary = np.asarray(boundary, dtype=np.int64)
    source = original[boundary]
    fixed = (
        None
        if fixed_target is None
        else np.asarray(fixed_target, dtype=np.float64).reshape(len(source), 3)
    )
    local_index = np.full(len(original), -1, dtype=np.int64)
    local_index[boundary] = np.arange(len(boundary), dtype=np.int64)
    faces = local_index[np.asarray(boundary_faces, dtype=np.int64)]
    if np.any(faces < 0):
        raise RuntimeError("cage boundary faces reference a non-boundary node")
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(source), len(source))
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    laplacian = eye(len(source), format="csr") - adjacency.multiply(
        (1.0 / np.maximum(degree, 1.0))[:, None]
    )
    smoothness = (laplacian.T @ laplacian).tocsr()
    # Establish a safe coarse fit before relaxing the differential-coordinate
    # regularizer.  Jumping directly to the lower weight collapses filled face
    # openings; retaining 1e6 forever leaves a 2--3 cm boundary residual.
    weight_schedule = (
        (1000000.0, 10),
        (300000.0, 8),
        (100000.0, 10),
        (30000.0, 12),
        (10000.0, 12),
    )
    differential = smoothness @ source
    registered = source.copy()
    base_tet = original[elements]
    base_det = np.linalg.det(base_tet[:, 1:] - base_tet[:, :1])
    if np.any(~np.isfinite(base_det)) or np.any(np.abs(base_det) <= 1.0e-18):
        raise RuntimeError("source volume cage contains a degenerate tetrahedron")
    if fixed is None:
        initial_squared, _face_index, _closest = igl.point_mesh_squared_distance(
            source, target, target_faces
        )
    else:
        initial_squared = np.sum((source - fixed) ** 2, axis=1)
    initial_rms = float(np.sqrt(np.mean(initial_squared)))
    initial_max = float(np.sqrt(np.max(initial_squared)))
    accepted_iterations = 0
    minimum_ratio = 1.0
    locked = np.zeros(len(boundary), dtype=bool)
    stage_iterations: list[int] = []
    for weight, iteration_count in weight_schedule:
        solve = factorized((eye(len(source), format="csc") + weight * smoothness).tocsc())
        accepted_in_stage = 0
        locked[:] = False
        for _iteration in range(iteration_count):
            if fixed is None:
                squared, _face_index, closest = igl.point_mesh_squared_distance(
                    registered, target, target_faces
                )
            else:
                squared = np.sum((registered - fixed) ** 2, axis=1)
                closest = fixed
            rhs = np.asarray(closest) + weight * differential
            proposal = np.column_stack([solve(rhs[:, axis]) for axis in range(3)])
            proposal[locked] = registered[locked]
            accepted = False
            for _barrier_iteration in range(12):
                proposal_field = _harmonic_step(
                    original, elements, boundary, proposal - source
                )
                trial = original + proposal_field
                trial_tet = trial[elements]
                ratio = np.linalg.det(trial_tet[:, 1:] - trial_tet[:, :1]) / base_det
                # Zero flips is insufficient for thin vessels: a nearly flat
                # tet creates an arbitrarily high-gradient interior field.
                bad = np.flatnonzero(
                    (~np.isfinite(ratio)) | (ratio <= 0.0) | (ratio < _MIN_JACOBIAN_RATIO)
                )
                if not len(bad):
                    accepted = True
                    break
                bad_boundary = local_index[np.unique(elements[bad])]
                bad_boundary = bad_boundary[bad_boundary >= 0]
                newly_locked = bad_boundary[~locked[bad_boundary]]
                if not len(newly_locked):
                    break
                locked[newly_locked] = True
                proposal[locked] = registered[locked]
            if not accepted:
                break
            registered = proposal
            accepted_iterations += 1
            accepted_in_stage += 1
            minimum_ratio = min(minimum_ratio, float(np.min(ratio)))
        stage_iterations.append(int(accepted_in_stage))
    if fixed is None:
        squared, _face_index, _closest = igl.point_mesh_squared_distance(
            registered, target, target_faces
        )
    else:
        squared = np.sum((registered - fixed) ** 2, axis=1)
    final_rms = float(np.sqrt(np.mean(squared)))
    final_max = float(np.sqrt(np.max(squared)))
    boundary_norm = np.linalg.norm(registered - source, axis=1)
    boundary_rms = float(np.sqrt(np.mean(boundary_norm * boundary_norm)))
    boundary_max = float(np.max(boundary_norm))
    progress = initial_rms - final_rms
    diagnostics = np.asarray(
        (
            initial_rms,
            initial_max,
            final_rms,
            final_max,
            boundary_rms,
            boundary_max,
            progress,
            minimum_ratio,
        ),
        dtype=np.float64,
    )
    if accepted_iterations == 0:
        if initial_rms <= _MIN_REGISTRATION_PROGRESS_M:
            raise RuntimeError("surface registration made no measurable progress")
        raise RuntimeError("surface registration rejected all proposals")
    if np.any(~np.isfinite(diagnostics)):
        raise RuntimeError("surface registration produced non-finite diagnostics")
    if progress < _MIN_REGISTRATION_PROGRESS_M or boundary_max < _MIN_REGISTRATION_PROGRESS_M:
        raise RuntimeError(
            "surface registration made no measurable progress "
            f"(initial RMS={initial_rms:.6f} m, final RMS={final_rms:.6f} m, "
            f"boundary max={boundary_max:.6f} m)"
        )
    if final_rms > _MAX_FINAL_SURFACE_RMS_M or final_max > _MAX_FINAL_SURFACE_DISTANCE_M:
        raise RuntimeError(
            "surface registration residual exceeds production limits "
            f"(RMS={final_rms:.6f}/{_MAX_FINAL_SURFACE_RMS_M:.6f} m, "
            f"max={final_max:.6f}/{_MAX_FINAL_SURFACE_DISTANCE_M:.6f} m)"
        )
    if boundary_max > _MAX_BOUNDARY_DISPLACEMENT_M:
        raise RuntimeError(
            "surface registration boundary displacement exceeds production limit "
            f"({boundary_max:.6f}/{_MAX_BOUNDARY_DISPLACEMENT_M:.6f} m)"
        )
    return registered, {
        "initial_surface_rms_m": initial_rms,
        "initial_surface_max_m": initial_max,
        "final_surface_rms_m": final_rms,
        "final_surface_max_m": final_max,
        "surface_rms_progress_m": progress,
        "boundary_displacement_rms_m": boundary_rms,
        "boundary_displacement_max_m": boundary_max,
        "accepted_surface_iterations": int(accepted_iterations),
        "accepted_surface_iterations_by_stage": stage_iterations,
        "surface_regularization_weights": [float(value[0]) for value in weight_schedule],
        "minimum_surface_jacobian_ratio": float(minimum_ratio),
        "locked_surface_vertices": int(np.count_nonzero(locked)),
        "fixed_semantic_correspondence": bool(fixed is not None),
    }


def _harmonic_step(
    nodes: np.ndarray,
    elements: np.ndarray,
    boundary: np.ndarray,
    boundary_values: np.ndarray,
) -> np.ndarray:
    from scipy.sparse.linalg import spsolve

    field = np.zeros_like(nodes, dtype=np.float64)
    field[boundary] = boundary_values
    interior = np.setdiff1d(np.arange(len(nodes)), boundary)
    stiffness = _tet_stiffness(nodes, elements)
    if len(interior):
        kii = stiffness[interior][:, interior]
        kib = stiffness[interior][:, boundary]
        for axis in range(3):
            field[interior, axis] = spsolve(kii, -(kib @ field[boundary, axis]))
    return field


def _incremental_harmonic_field(
    nodes: np.ndarray,
    elements: np.ndarray,
    boundary: np.ndarray,
    boundary_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Reach the full boundary displacement without ever flipping a tetrahedron."""
    original = np.asarray(nodes, dtype=np.float64)
    current = original.copy()
    remaining = np.asarray(boundary_values, dtype=np.float64).copy()
    accepted = 0
    minimum_fraction = 1.0
    minimum_jacobian_ratio = float("inf")
    minimum_step_jacobian_ratio = float("inf")
    base_tet = original[np.asarray(elements, dtype=np.int64)]
    base_det = np.linalg.det(base_tet[:, 1:] - base_tet[:, :1])
    if np.any(~np.isfinite(base_det)) or np.any(np.abs(base_det) <= 1.0e-18):
        raise RuntimeError("source volume cage contains a degenerate tetrahedron")
    for _iteration in range(64):
        if float(np.max(np.linalg.norm(remaining, axis=1))) <= 1.0e-7:
            break
        fraction = 1.0
        while fraction >= 1.0 / 1024.0:
            step_boundary = remaining * fraction
            step = _harmonic_step(current, elements, boundary, step_boundary)
            trial = current + step
            trial_tet = trial[np.asarray(elements, dtype=np.int64)]
            trial_det = np.linalg.det(trial_tet[:, 1:] - trial_tet[:, :1])
            ratios = trial_det / base_det
            current_tet = current[np.asarray(elements, dtype=np.int64)]
            current_det = np.linalg.det(current_tet[:, 1:] - current_tet[:, :1])
            step_ratios = trial_det / current_det
            positive = (
                np.isfinite(ratios)
                & np.isfinite(step_ratios)
                & (ratios > 0.0)
                & (step_ratios > 0.0)
            )
            if np.all(
                positive
                & (ratios >= _MIN_JACOBIAN_RATIO)
                & (step_ratios >= _MIN_JACOBIAN_RATIO)
            ):
                current = trial
                remaining -= step_boundary
                accepted += 1
                minimum_fraction = min(minimum_fraction, fraction)
                minimum_jacobian_ratio = min(minimum_jacobian_ratio, float(np.min(ratios)))
                minimum_step_jacobian_ratio = min(
                    minimum_step_jacobian_ratio, float(np.min(step_ratios))
                )
                break
            fraction *= 0.5
        else:
            raise RuntimeError(
                "harmonic volume registration cannot avoid tetrahedron inversion "
                "or minimum Jacobian-ratio violation"
            )
    else:
        raise RuntimeError("harmonic volume registration did not converge to the target boundary")
    final_tet = current[np.asarray(elements, dtype=np.int64)]
    final_det = np.linalg.det(final_tet[:, 1:] - final_tet[:, :1])
    final_ratio = final_det / base_det
    inverted = int(np.count_nonzero((~np.isfinite(final_ratio)) | (final_ratio <= 0.0)))
    if inverted:
        raise RuntimeError(f"harmonic volume registration inverted {inverted} tetrahedra")
    if np.any(final_ratio < _MIN_JACOBIAN_RATIO):
        raise RuntimeError(
            "harmonic volume registration violates the minimum Jacobian ratio "
            f"({float(np.min(final_ratio)):.6f} < {_MIN_JACOBIAN_RATIO:.6f})"
        )
    if not np.isfinite(minimum_jacobian_ratio):
        minimum_jacobian_ratio = 1.0
    if not np.isfinite(minimum_step_jacobian_ratio):
        minimum_step_jacobian_ratio = 1.0
    return current - original, {
        "incremental_steps": int(accepted),
        "minimum_step_fraction": float(minimum_fraction),
        "minimum_jacobian_ratio": float(minimum_jacobian_ratio),
        "minimum_incremental_step_jacobian_ratio": float(minimum_step_jacobian_ratio),
        "inverted_tetrahedra": inverted,
    }


def _nearest_skeleton_segment(
    points: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    *,
    batch_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    children = np.flatnonzero(np.asarray(parents, dtype=np.int64) >= 0)
    starts = joints[np.asarray(parents, dtype=np.int64)[children]]
    vectors = joints[children] - starts
    length2 = np.einsum("ij,ij->i", vectors, vectors)
    valid = length2 > 1.0e-10
    children = children[valid]
    starts = starts[valid]
    vectors = vectors[valid]
    length2 = length2[valid]
    assignment = np.empty(len(points), dtype=np.int32)
    centers = np.empty_like(points, dtype=np.float64)
    for begin in range(0, len(points), int(batch_size)):
        end = min(len(points), begin + int(batch_size))
        query = np.asarray(points[begin:end], dtype=np.float64)
        relative = query[:, None, :] - starts[None, :, :]
        parameter = np.clip(
            np.einsum("nsi,si->ns", relative, vectors) / length2[None, :],
            0.0,
            1.0,
        )
        projected = starts[None, :, :] + parameter[:, :, None] * vectors[None, :, :]
        distance2 = np.sum((query[:, None, :] - projected) ** 2, axis=2)
        selected = np.argmin(distance2, axis=1)
        rows = np.arange(len(query))
        assignment[begin:end] = selected.astype(np.int32)
        centers[begin:end] = projected[rows, selected]
    return assignment, centers, children


def _smooth_mesh_displacement(
    desired: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 30,
) -> np.ndarray:
    from scipy.sparse import coo_matrix, diags

    triangles = np.asarray(faces, dtype=np.int64)
    if not len(triangles):
        return np.asarray(desired, dtype=np.float64)
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(desired), len(desired)),
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    average = diags(1.0 / np.maximum(degree, 1.0)) @ adjacency
    target = np.asarray(desired, dtype=np.float64)
    output = target.copy()
    for _iteration in range(int(iterations)):
        output = 0.1 * target + 0.9 * (average @ output)
    return output


def _section_residual_regularizer(
    asset: AnatomyRiggedAsset,
    source_vertices: np.ndarray,
    mapped_vertices: np.ndarray,
    registered_skin: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Propagate residual skin mismatch through vessel/nerve cross-sections."""
    import igl

    joints = np.asarray(asset.rest_joints, dtype=np.float64)
    parents = np.asarray(asset.parents, dtype=np.int64)
    skin_segment, skin_center, children = _nearest_skeleton_segment(
        registered_skin,
        joints,
        parents,
    )
    _squared, _face_index, closest = igl.point_mesh_squared_distance(
        registered_skin,
        target_vertices,
        target_faces,
    )
    source_radius = np.linalg.norm(registered_skin - skin_center, axis=1)
    target_radius = np.linalg.norm(np.asarray(closest) - skin_center, axis=1)
    ratios = np.minimum(
        1.0,
        target_radius / np.maximum(source_radius, 1.0e-5),
    )
    scales = np.ones(len(children), dtype=np.float64)
    for segment in np.unique(skin_segment):
        local = ratios[skin_segment == segment]
        if len(local) >= 4:
            scales[int(segment)] = float(np.quantile(local, 0.02))

    child_to_segment = {
        int(child): index for index, child in enumerate(children.tolist())
    }
    joint_names = list(asset.joint_names)
    for index, child in enumerate(children.tolist()):
        name = joint_names[int(child)]
        mirror_name = (
            f"right_{name[5:]}"
            if name.startswith("left_")
            else (f"left_{name[6:]}" if name.startswith("right_") else None)
        )
        if mirror_name is None or mirror_name not in joint_names:
            continue
        mirror_child = joint_names.index(mirror_name)
        if mirror_child not in child_to_segment:
            continue
        mirror_index = child_to_segment[mirror_child]
        shared = min(float(scales[index]), float(scales[mirror_index]))
        scales[index] = shared
        scales[mirror_index] = shared
    for _iteration in range(3):
        previous = scales.copy()
        for index, child in enumerate(children.tolist()):
            neighbours = [index]
            parent = int(parents[child])
            if parent in child_to_segment:
                neighbours.append(child_to_segment[parent])
            neighbours.extend(
                child_to_segment[int(other)]
                for other in children
                if int(parents[int(other)]) == int(child)
            )
            scales[index] = min(
                previous[index],
                float(np.mean(previous[neighbours])),
            )

    output = np.asarray(mapped_vertices, dtype=np.float64).copy()
    assignment, centers, _children = _nearest_skeleton_segment(
        output,
        joints,
        parents,
    )
    eligible = np.zeros(len(output), dtype=bool)
    for (start, stop), tissue in zip(
        asset.source_vertex_ranges,
        asset.source_tissues,
    ):
        if str(tissue) != "bone":
            eligible[int(start) : int(stop)] = True
    eligible &= ~cranial_material_mask(asset)
    local_scale = scales[assignment]
    desired = (local_scale[:, None] - 1.0) * (output - centers)
    desired[~eligible] = 0.0
    all_faces = np.asarray(asset.faces, dtype=np.int64)
    arap_reports: dict[str, Any] = {}
    for mesh_name, start_stop, tissue in zip(
        asset.source_mesh_names,
        asset.source_vertex_ranges,
        asset.source_tissues,
    ):
        if str(tissue) == "bone":
            continue
        start, stop = (int(value) for value in start_stop)
        local_faces = all_faces[
            np.all((all_faces >= start) & (all_faces < stop), axis=1)
        ] - start
        output[start:stop] += _smooth_mesh_displacement(
            desired[start:stop],
            local_faces,
        )
        refined, arap_report = arap_volume_refine(
            np.asarray(source_vertices, dtype=np.float64)[start:stop],
            output[start:stop],
            local_faces,
            target_weight=8.0,
            iterations=2,
            volume_weight=0.25 if str(tissue) in {"organ", "heart"} else 0.0,
        )
        output[start:stop] = refined
        arap_reports[str(mesh_name)] = arap_report
    displacement = np.linalg.norm(output - mapped_vertices, axis=1)
    return output, {
        "minimum_section_scale": (
            float(np.min(local_scale[eligible])) if np.any(eligible) else 1.0
        ),
        "mean_displacement_m": (
            float(np.mean(displacement[eligible])) if np.any(eligible) else 0.0
        ),
        "max_displacement_m": (
            float(np.max(displacement[eligible])) if np.any(eligible) else 0.0
        ),
        "regularized_vertex_count": int(np.count_nonzero(eligible)),
        "mesh_arap": arap_reports,
    }


def _raise_outside_query_error(
    asset: AnatomyRiggedAsset,
    *,
    query: np.ndarray,
    outside_mask: np.ndarray,
    protected: np.ndarray,
    cage: dict[str, np.ndarray],
    context: str,
) -> None:
    """Raise a detailed fail-fast error for every outside material query."""
    outside = np.asarray(outside_mask, dtype=bool)
    if not np.any(outside):
        return
    soft_outside = outside & ~protected
    protected_outside = outside & protected
    by_mesh: dict[str, dict[str, int]] = {}
    if asset.source_vertex_ranges is not None and asset.source_mesh_names is not None:
        for name, (start, stop) in zip(asset.source_mesh_names, asset.source_vertex_ranges):
            mesh_outside = outside[int(start) : int(stop)]
            if not np.any(mesh_outside):
                continue
            mesh_protected = protected[int(start) : int(stop)]
            by_mesh[str(name)] = {
                "soft": int(np.count_nonzero(mesh_outside & ~mesh_protected)),
                "protected": int(np.count_nonzero(mesh_outside & mesh_protected)),
            }
    maximum_distance = _outside_cage_max_distance(query[outside], cage=cage)
    raise RuntimeError(
        f"{context} excludes {int(np.count_nonzero(outside))} anatomy vertices "
        f"(soft={int(np.count_nonzero(soft_outside))}, "
        f"protected={int(np.count_nonzero(protected_outside))}, "
        f"max distance={maximum_distance * 1000.0:.2f} mm): "
        f"{dict(list(by_mesh.items())[:20])}"
    )


def apply_source_skin_volume_registration(
    asset: AnatomyRiggedAsset, *, canonical_dir: Path | str
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    if asset.source_skin_vertices is None or asset.source_skin_faces is None:
        raise RuntimeError("source template lacks Skin_Glass; force source template rebake")
    if asset.source_skin_lbs_weights is None:
        raise RuntimeError(
            "source template lacks Skin_Glass semantic weights; use --force-source-rebake"
        )

    root = Path(canonical_dir)
    target_vertices, target_faces = _load_obj(root / "smpl_canonical_tpose_neutral.obj")
    skin_vertices = np.asarray(asset.source_skin_vertices, dtype=np.float64)
    skin_faces = np.asarray(asset.source_skin_faces, dtype=np.int32)
    skin_weights = np.asarray(asset.source_skin_lbs_weights, dtype=np.float64)
    target_weight_data = np.load(root / "smpl_canonical_weights.npz", allow_pickle=True)
    target_weights = np.asarray(target_weight_data["lbs_weights"], dtype=np.float64)
    target_joint_names = [str(value) for value in target_weight_data["joint_names"].tolist()]
    if target_joint_names != list(asset.joint_names):
        raise ValueError("Skin_Glass and SMPL-X joint semantic order does not match")
    query = np.asarray(asset.vertices_rest, dtype=np.float64)
    protected = bone_material_mask(asset) | cranial_material_mask(asset)
    cage = _build_source_cage(
        skin_vertices,
        skin_faces,
        root / "source_skin_volume_cage_v7_surface_only.npz",
    )
    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    elements = np.asarray(cage["elements"], dtype=np.int32)
    boundary = np.asarray(cage["boundary"], dtype=np.int64)
    _preflight_delta, _outside_count, preflight_outside = _sample_field(
        query, cage=cage, field=np.zeros_like(nodes)
    )
    _raise_outside_query_error(
        asset,
        query=query,
        outside_mask=preflight_outside,
        protected=protected,
        cage=cage,
        context="source Skin_Glass domain",
    )
    mapped_skin, semantic_map_report = _fixed_semantic_skin_map(
        skin_vertices,
        skin_faces,
        skin_weights,
        target_vertices,
        target_faces,
        target_weights,
        joint_names=list(asset.joint_names),
        cache_path=root / "source_skin_semantic_map_v1.npz",
    )
    fixed_boundary = _barycentric_surface_map(
        nodes[boundary], skin_vertices, skin_faces, mapped_skin
    )
    registered_boundary, surface_report = _topology_preserving_cage_registration(
        nodes,
        elements,
        boundary,
        np.asarray(cage["boundary_faces"], dtype=np.int32),
        target_vertices,
        target_faces,
        fixed_target=fixed_boundary,
    )
    boundary_values = registered_boundary - nodes[boundary]
    field, deformation_report = _incremental_harmonic_field(
        nodes, elements, boundary, boundary_values
    )
    delta, outside_count, outside_mask = _sample_field(query, cage=cage, field=field)
    _raise_outside_query_error(
        asset,
        query=query,
        outside_mask=outside_mask,
        protected=protected,
        cage=cage,
        context="registered source Skin_Glass domain",
    )
    mapped = _transport_sampled_material(
        query,
        delta,
        protected=protected,
        outside=outside_mask,
    )
    skin_delta, _skin_outside_count, skin_outside = _sample_field(
        skin_vertices,
        cage=cage,
        field=field,
    )
    if np.any(skin_outside):
        raise RuntimeError(
            "registered source skin cannot be sampled from its own volume cage"
        )
    section_report = {
        "disabled": True,
        "reason": "radial_section_shrink_forbidden_for_thin_anatomy",
    }
    from .soft_constraints import (
        arap_volume_refine,
        limit_edge_strain,
    )

    barrier_reports: dict[str, Any] = {}
    strain_reports: dict[str, Any] = {}
    all_faces = np.asarray(asset.faces, dtype=np.int64)
    for mesh_name, (start, stop), tissue in zip(
        asset.source_mesh_names,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        asset.source_tissues,
    ):
        if str(tissue) not in {"vessel", "nerve"}:
            continue
        start_i, stop_i = int(start), int(stop)
        local_faces = all_faces[
            np.all((all_faces >= start_i) & (all_faces < stop_i), axis=1)
        ] - start_i
        arap_fitted, arap_report = arap_volume_refine(
            query[start_i:stop_i],
            mapped[start_i:stop_i],
            local_faces,
            target_weight=0.01,
            iterations=15,
            volume_weight=0.0,
        )
        prestrained, strain_report = limit_edge_strain(
            query[start_i:stop_i],
            arap_fitted,
            local_faces,
            minimum_ratio=0.85,
            maximum_ratio=1.25,
            iterations=60,
        )
        strain_reports[str(mesh_name)] = {
            "arap": arap_report,
            "bounded_edges": strain_report,
        }
        mapped[start_i:stop_i] = prestrained
    soft_norm = np.linalg.norm(mapped[~protected] - query[~protected], axis=1)
    if soft_norm.size:
        soft_rms = float(np.sqrt(np.mean(soft_norm * soft_norm)))
        soft_max = float(np.max(soft_norm))
    else:
        soft_rms = 0.0
        soft_max = 0.0
    if not np.isfinite(soft_rms) or not np.isfinite(soft_max):
        raise RuntimeError("source skin registration produced non-finite soft displacement")
    if (
        soft_norm.size
        and float(surface_report["boundary_displacement_max_m"]) > _MIN_REGISTRATION_PROGRESS_M
        and soft_max < _MIN_REGISTRATION_PROGRESS_M
    ):
        raise RuntimeError(
            "source skin registration moved the boundary but produced no measurable soft displacement"
        )
    source_volume = np.linalg.det(nodes[elements][:, 1:] - nodes[elements][:, :1])
    target_nodes = nodes + field
    target_volume = np.linalg.det(target_nodes[elements][:, 1:] - target_nodes[elements][:, :1])
    inverted = int(np.count_nonzero(source_volume * target_volume <= 0.0))
    if inverted:
        raise RuntimeError(f"source skin harmonic field inverted {inverted} tetrahedra")
    minimum_jacobian_ratio = float(np.min(target_volume / source_volume))
    if minimum_jacobian_ratio < _MIN_JACOBIAN_RATIO:
        raise RuntimeError(
            f"source skin harmonic field is near-degenerate: min Jacobian ratio {minimum_jacobian_ratio:.6f}"
        )
    metadata = dict(asset.metadata or {})
    metadata["source_skin_volume_registration"] = "topology_preserving_harmonic_v7"
    result = type(asset)(
        **{**asset.__dict__, "vertices_rest": mapped.astype(np.float32), "metadata": metadata}
    )
    return result, {
        "backend": "topology_preserving_harmonic_v7",
        "cage_nodes": int(len(nodes)),
        "cage_tetrahedra": int(len(elements)),
        "cage_voxel_pitch_m": float(np.asarray(cage["voxel_pitch"]).reshape(-1)[0]),
        "cage_meshing_backend": str(np.asarray(cage["meshing_backend"]).reshape(-1)[0]),
        "removed_degenerate_tetrahedra": int(
            np.asarray(cage["removed_degenerate_tetrahedra"]).reshape(-1)[0]
        ),
        "outside_query_count": int(outside_count),
        "outside_protected_material_count": 0,
        "outside_soft_material_count": 0,
        "diagnostic_inverted_tetrahedra": 0,
        "minimum_jacobian_ratio": minimum_jacobian_ratio,
        "soft_displacement_rms_m": soft_rms,
        "soft_displacement_max_m": soft_max,
        "protected_material_vertices": int(np.count_nonzero(protected)),
        "anatomy_transport": "soft_volume_field_applied_rigid_material_preserved",
        "section_residual_regularizer": section_report,
        "soft_edge_strain_regularizer": strain_reports,
        "surface_barrier_regularizer": barrier_reports,
        "surface_correspondence": semantic_map_report,
        **deformation_report,
        **surface_report,
    }
