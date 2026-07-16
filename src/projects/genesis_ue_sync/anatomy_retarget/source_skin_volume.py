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
from .shape_volume import _load_obj, _sample_field, _tet_stiffness


_CAGE_VERSION = "source_skin_volume_v5_7_closed_domain"


def _signature(vertices: np.ndarray, faces: np.ndarray, enclosure_points: np.ndarray | None = None) -> str:
    digest = hashlib.sha256(_CAGE_VERSION.encode("utf-8"))
    digest.update(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(faces, dtype=np.int32).tobytes())
    if enclosure_points is not None:
        digest.update(np.ascontiguousarray(enclosure_points, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _voxel_union(
    vertices: np.ndarray, faces: np.ndarray, enclosure_points: np.ndarray | None = None
):
    """Turn visual seams/multiple shells into one watertight material domain."""
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
    point_indices = np.zeros((0, 3), dtype=np.int64)
    if enclosure_points is not None and len(enclosure_points):
        inverse = np.linalg.inv(transform)
        homo = np.concatenate(
            (np.asarray(enclosure_points, dtype=np.float64), np.ones((len(enclosure_points), 1))),
            axis=1,
        )
        point_indices = np.rint((homo @ inverse.T)[:, :3]).astype(np.int64)
        lower = np.minimum(lower, np.min(point_indices, axis=0) - padding)
        upper = np.maximum(upper, np.max(point_indices, axis=0) + padding + 1)
    occupancy = np.zeros(tuple((upper - lower).tolist()), dtype=bool)
    shift = -lower
    occupancy[
        shift[0] : shift[0] + base.shape[0],
        shift[1] : shift[1] + base.shape[1],
        shift[2] : shift[2] + base.shape[2],
    ] = base
    if len(point_indices):
        occupied_base = np.argwhere(base)
        from scipy.spatial import cKDTree

        inside_bounds = np.all((point_indices >= 0) & (point_indices < np.asarray(base.shape)), axis=1)
        already_inside = np.zeros(len(point_indices), dtype=bool)
        already_inside[inside_bounds] = base[tuple(point_indices[inside_bounds].T)]
        outside = point_indices[~already_inside]
        if len(outside):
            _distance, nearest = cKDTree(occupied_base).query(outside, k=1)
            anchors = occupied_base[np.asarray(nearest, dtype=np.int64)]
            for start, stop in zip(anchors, outside):
                count = int(np.max(np.abs(stop - start))) + 1
                line = np.rint(np.linspace(start, stop, count)).astype(np.int64) + shift
                occupancy[tuple(line.T)] = True
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


def _build_source_cage(
    vertices: np.ndarray,
    faces: np.ndarray,
    cache_path: Path,
    enclosure_points: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    signature = _signature(vertices, faces, enclosure_points)
    if cache_path.is_file():
        data = np.load(cache_path)
        cached = str(np.asarray(data.get("signature", "")).reshape(-1)[0])
        if cached == signature:
            return {key: np.asarray(data[key]) for key in data.files}

    import tetgen

    surface, pitch = _voxel_union(vertices, faces, enclosure_points)
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
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit the closed cage while retaining the last zero-inversion state."""
    import igl
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import factorized

    original = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    boundary = np.asarray(boundary, dtype=np.int64)
    source = original[boundary]
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
    weight_schedule = ((1000000.0, 10), (600000.0, 8), (300000.0, 8))
    differential = smoothness @ source
    registered = source.copy()
    base_tet = original[elements]
    base_det = np.linalg.det(base_tet[:, 1:] - base_tet[:, :1])
    initial_rms = 0.0
    accepted_iterations = 0
    minimum_ratio = 1.0
    locked = np.zeros(len(boundary), dtype=bool)
    stage_iterations: list[int] = []
    for weight, iteration_count in weight_schedule:
        solve = factorized((eye(len(source), format="csc") + weight * smoothness).tocsc())
        accepted_in_stage = 0
        locked[:] = False
        for _iteration in range(iteration_count):
            squared, _face_index, closest = igl.point_mesh_squared_distance(
                registered, target, target_faces
            )
            if accepted_iterations == 0:
                initial_rms = float(np.sqrt(np.mean(squared)))
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
                bad = np.flatnonzero(ratio <= 1.0e-5)
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
    squared, _face_index, _closest = igl.point_mesh_squared_distance(
        registered, target, target_faces
    )
    return registered, {
        "initial_surface_rms_m": initial_rms,
        "final_surface_rms_m": float(np.sqrt(np.mean(squared))),
        "final_surface_max_m": float(np.sqrt(np.max(squared))),
        "accepted_surface_iterations": int(accepted_iterations),
        "accepted_surface_iterations_by_stage": stage_iterations,
        "surface_regularization_weights": [float(value[0]) for value in weight_schedule],
        "minimum_surface_jacobian_ratio": float(minimum_ratio),
        "locked_surface_vertices": int(np.count_nonzero(locked)),
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
    base_tet = original[np.asarray(elements, dtype=np.int64)]
    base_det = np.linalg.det(base_tet[:, 1:] - base_tet[:, :1])
    if np.any(np.abs(base_det) <= 1.0e-18):
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
            if np.all(ratios > 1.0e-5):
                current = trial
                remaining -= step_boundary
                accepted += 1
                minimum_fraction = min(minimum_fraction, fraction)
                minimum_jacobian_ratio = min(minimum_jacobian_ratio, float(np.min(ratios)))
                break
            fraction *= 0.5
        else:
            raise RuntimeError("harmonic volume registration cannot avoid tetrahedron inversion")
    else:
        raise RuntimeError("harmonic volume registration did not converge to the target boundary")
    return current - original, {
        "incremental_steps": int(accepted),
        "minimum_step_fraction": float(minimum_fraction),
        "minimum_jacobian_ratio": float(minimum_jacobian_ratio),
        "inverted_tetrahedra": 0,
    }


def _nearest_skeleton_segment(
    points: np.ndarray, joints: np.ndarray, parents: np.ndarray, *, batch_size: int = 50000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    children = np.flatnonzero(np.asarray(parents, dtype=np.int64) >= 0)
    starts = joints[parents[children]]
    vectors = joints[children] - starts
    length2 = np.einsum("ij,ij->i", vectors, vectors)
    valid = length2 > 1.0e-10
    children, starts, vectors, length2 = children[valid], starts[valid], vectors[valid], length2[valid]
    assignment = np.empty(len(points), dtype=np.int32)
    centers = np.empty_like(points, dtype=np.float64)
    for begin in range(0, len(points), int(batch_size)):
        end = min(len(points), begin + int(batch_size))
        query = np.asarray(points[begin:end], dtype=np.float64)
        rel = query[:, None, :] - starts[None, :, :]
        parameter = np.clip(
            np.einsum("nsi,si->ns", rel, vectors) / length2[None, :], 0.0, 1.0
        )
        projected = starts[None, :, :] + parameter[:, :, None] * vectors[None, :, :]
        distance2 = np.sum((query[:, None, :] - projected) ** 2, axis=2)
        selected = np.argmin(distance2, axis=1)
        rows = np.arange(len(query))
        assignment[begin:end] = selected.astype(np.int32)
        centers[begin:end] = projected[rows, selected]
    return assignment, centers, children


def _smooth_material_displacement(
    desired: np.ndarray, faces: np.ndarray, *, iterations: int = 30
) -> np.ndarray:
    from scipy.sparse import coo_matrix, diags

    triangles = np.asarray(faces, dtype=np.int64)
    if not len(triangles):
        return np.asarray(desired, dtype=np.float64)
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
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
    for _ in range(int(iterations)):
        output = 0.1 * target + 0.9 * (average @ output)
    return output


def _skin_material_scale_correction(
    asset: AnatomyRiggedAsset,
    mapped_vertices: np.ndarray,
    registered_skin: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Propagate residual skin-fit scale through complete material sections.

    The correction is estimated exclusively from the source-skin boundary, not
    from anatomy penetration.  A single scale is shared by every soft point in
    a skeleton cross-section, preserving vessel topology and relative depth.
    """
    import igl

    joints = np.asarray(asset.rest_joints, dtype=np.float64)
    parents = np.asarray(asset.parents, dtype=np.int64)
    skin_segment, skin_center, children = _nearest_skeleton_segment(
        registered_skin, joints, parents
    )
    _squared, _face_index, closest = igl.point_mesh_squared_distance(
        registered_skin, target_vertices, target_faces
    )
    source_radius = np.linalg.norm(registered_skin - skin_center, axis=1)
    target_radius = np.linalg.norm(np.asarray(closest) - skin_center, axis=1)
    ratios = np.clip(target_radius / np.maximum(source_radius, 1.0e-5), 0.7, 1.0)
    scales = np.ones(len(children), dtype=np.float64)
    for segment in np.unique(skin_segment):
        local = ratios[skin_segment == segment]
        if len(local) >= 4:
            scales[segment] = min(1.0, float(np.quantile(local, 0.02)))

    joint_names = list(asset.joint_names)
    child_to_segment = {int(child): idx for idx, child in enumerate(children.tolist())}
    for idx, child in enumerate(children.tolist()):
        name = joint_names[int(child)]
        mirror_name = (
            ("right_" + name[5:]) if name.startswith("left_") else ("left_" + name[6:]) if name.startswith("right_") else None
        )
        if mirror_name is None:
            continue
        mirror_child = next(
            (int(other) for other in children.tolist() if joint_names[int(other)] == mirror_name),
            None,
        )
        if mirror_child is None:
            continue
        mirror_idx = child_to_segment[mirror_child]
        shared = min(float(scales[idx]), float(scales[mirror_idx]))
        scales[idx] = shared
        scales[mirror_idx] = shared

    child_to_segment = {int(child): idx for idx, child in enumerate(children.tolist())}
    for _ in range(3):
        previous = scales.copy()
        for idx, child in enumerate(children.tolist()):
            neighbours = [idx]
            parent = int(parents[child])
            if parent in child_to_segment:
                neighbours.append(child_to_segment[parent])
            neighbours.extend(
                child_to_segment[int(other)]
                for other in children
                if int(parents[int(other)]) == int(child)
            )
            scales[idx] = min(previous[idx], float(np.mean(previous[neighbours])))

    output = np.asarray(mapped_vertices, dtype=np.float64).copy()
    assignment, centers, _children = _nearest_skeleton_segment(output, joints, parents)
    soft = np.zeros(len(output), dtype=bool)
    for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
        if str(tissue) != "bone":
            soft[int(start) : int(stop)] = True
    local_scale = scales[assignment]
    desired = (local_scale[:, None] - 1.0) * (output - centers)
    all_faces = np.asarray(asset.faces, dtype=np.int64)
    for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
        if str(tissue) == "bone":
            continue
        start, stop = int(start), int(stop)
        local_faces = all_faces[
            (all_faces[:, 0] >= start)
            & (all_faces[:, 0] < stop)
            & (all_faces[:, 1] >= start)
            & (all_faces[:, 1] < stop)
            & (all_faces[:, 2] >= start)
            & (all_faces[:, 2] < stop)
        ] - start
        output[start:stop] += _smooth_material_displacement(
            desired[start:stop], local_faces
        )
    displacement = np.linalg.norm(output - mapped_vertices, axis=1)
    return output, {
        "minimum_section_scale": float(np.min(local_scale[soft])),
        "mean_soft_displacement_m": float(np.mean(displacement[soft])),
        "max_soft_displacement_m": float(np.max(displacement[soft])),
    }


def apply_source_skin_volume_registration(
    asset: AnatomyRiggedAsset, *, canonical_dir: Path | str
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    if asset.source_skin_vertices is None or asset.source_skin_faces is None:
        raise RuntimeError("source template lacks Skin_Glass; force source template rebake")

    root = Path(canonical_dir)
    target_vertices, target_faces = _load_obj(root / "smpl_canonical_tpose_neutral.obj")
    skin_vertices = np.asarray(asset.source_skin_vertices, dtype=np.float64)
    skin_faces = np.asarray(asset.source_skin_faces, dtype=np.int32)
    query = np.asarray(asset.vertices_rest, dtype=np.float64)
    protected = bone_material_mask(asset) | cranial_material_mask(asset)
    cage = _build_source_cage(
        skin_vertices,
        skin_faces,
        root / "source_skin_volume_cage_v5_7.npz",
        enclosure_points=query[~protected],
    )
    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    elements = np.asarray(cage["elements"], dtype=np.int32)
    boundary = np.asarray(cage["boundary"], dtype=np.int64)
    registered_boundary, surface_report = _topology_preserving_cage_registration(
        nodes,
        elements,
        boundary,
        np.asarray(cage["boundary_faces"], dtype=np.int32),
        target_vertices,
        target_faces,
    )
    boundary_values = registered_boundary - nodes[boundary]
    field, deformation_report = _incremental_harmonic_field(
        nodes, elements, boundary, boundary_values
    )
    delta, outside_count, outside_mask = _sample_field(query, cage=cage, field=field)
    soft_outside = outside_mask & ~protected
    if np.any(soft_outside):
        names: dict[str, int] = {}
        for name, (start, stop) in zip(asset.source_mesh_names, asset.source_vertex_ranges):
            count = int(np.count_nonzero(soft_outside[start:stop]))
            if count:
                names[str(name)] = count
        raise RuntimeError(
            f"source Skin_Glass domain excludes {np.count_nonzero(soft_outside)} soft anatomy vertices: "
            f"{dict(list(names.items())[:12])}"
        )
    mapped = query + delta
    mapped[protected] = query[protected]
    source_volume = np.linalg.det(nodes[elements][:, 1:] - nodes[elements][:, :1])
    target_nodes = nodes + field
    target_volume = np.linalg.det(target_nodes[elements][:, 1:] - target_nodes[elements][:, :1])
    inverted = int(np.count_nonzero(source_volume * target_volume <= 0.0))
    if inverted:
        raise RuntimeError(f"source skin harmonic field inverted {inverted} tetrahedra")
    metadata = dict(asset.metadata or {})
    metadata["source_skin_volume_registration"] = "topology_preserving_harmonic_v5_7"
    result = type(asset)(
        **{**asset.__dict__, "vertices_rest": mapped.astype(np.float32), "metadata": metadata}
    )
    return result, {
        "backend": "topology_preserving_harmonic_v5_7",
        "cage_nodes": int(len(nodes)),
        "cage_tetrahedra": int(len(elements)),
        "cage_voxel_pitch_m": float(np.asarray(cage["voxel_pitch"]).reshape(-1)[0]),
        "cage_meshing_backend": str(np.asarray(cage["meshing_backend"]).reshape(-1)[0]),
        "removed_degenerate_tetrahedra": int(
            np.asarray(cage["removed_degenerate_tetrahedra"]).reshape(-1)[0]
        ),
        "outside_query_count": int(outside_count),
        "outside_protected_material_count": int(np.count_nonzero(outside_mask & protected)),
        "outside_soft_material_count": 0,
        "diagnostic_inverted_tetrahedra": 0,
        "protected_material_vertices": int(np.count_nonzero(protected)),
        **deformation_report,
        **surface_report,
    }
