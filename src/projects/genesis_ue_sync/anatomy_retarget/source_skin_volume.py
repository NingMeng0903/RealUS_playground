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
from .shape_volume import _load_obj, _preserve_rigid_bone_components, _sample_field, _tet_stiffness
from .source_rebind import rebind_source_rig


_CAGE_VERSION = "source_skin_volume_v5_4"


def _signature(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256(_CAGE_VERSION.encode("utf-8"))
    digest.update(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(faces, dtype=np.int32).tobytes())
    return digest.hexdigest()


def _voxel_union(vertices: np.ndarray, faces: np.ndarray):
    """Turn visual seams/multiple shells into one watertight material domain."""
    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, process=True)
    longest = float(np.max(mesh.extents))
    if not np.isfinite(longest) or longest <= 0.0:
        raise RuntimeError("Skin_Glass has invalid dimensions")
    pitch = longest / 180.0
    grid = mesh.voxelized(pitch).fill()
    surface = grid.marching_cubes
    surface.apply_transform(grid.transform)
    surface.remove_unreferenced_vertices()
    surface.fix_normals()
    surface = trimesh.Trimesh(surface.vertices, surface.faces, process=True)
    if not surface.is_watertight or not surface.is_volume:
        raise RuntimeError("Skin_Glass voxel union is not a closed volume")
    return surface, pitch


def _build_source_cage(
    vertices: np.ndarray, faces: np.ndarray, cache_path: Path
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
    nodes, elements, _attributes, _markers = generator.tetrahedralize(
        order=1,
        mindihedral=5.0,
        minratio=2.0,
        maxvolume=float(np.max(surface.extents) ** 3 / 4000.0),
        quiet=True,
    )
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int32)
    boundary = np.unique(np.asarray(generator.trifaces, dtype=np.int32).reshape(-1))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        nodes=nodes.astype(np.float32),
        elements=elements,
        boundary=boundary.astype(np.int32),
        signature=np.asarray([signature]),
        voxel_pitch=np.asarray([pitch], dtype=np.float32),
    )
    return {
        "nodes": nodes,
        "elements": elements,
        "boundary": boundary,
        "signature": np.asarray([signature]),
        "voxel_pitch": np.asarray([pitch], dtype=np.float32),
    }


def _screened_surface_registration(
    source: np.ndarray,
    faces: np.ndarray,
    target: np.ndarray,
    target_faces: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Closest-surface fit with differential-coordinate regularisation."""
    import igl
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import factorized

    source = np.asarray(source, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
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
    weight = 8.0
    solve = factorized((eye(len(source), format="csc") + weight * smoothness).tocsc())
    differential = smoothness @ source
    registered = source.copy()
    initial_rms = 0.0
    for iteration in range(5):
        squared, _face_index, closest = igl.point_mesh_squared_distance(
            registered, target, target_faces
        )
        if iteration == 0:
            initial_rms = float(np.sqrt(np.mean(squared)))
        rhs = np.asarray(closest) + weight * differential
        registered = np.column_stack([solve(rhs[:, axis]) for axis in range(3)])
    squared, _face_index, _closest = igl.point_mesh_squared_distance(
        registered, target, target_faces
    )
    return registered, {
        "initial_surface_rms_m": initial_rms,
        "final_surface_rms_m": float(np.sqrt(np.mean(squared))),
        "final_surface_max_m": float(np.sqrt(np.max(squared))),
    }


def _barycentric_displacement(
    points: np.ndarray,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
) -> np.ndarray:
    import igl

    _squared, face_index, closest = igl.point_mesh_squared_distance(
        points, source_vertices, source_faces
    )
    triangles = source_vertices[source_faces[face_index]]
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    q = np.asarray(closest) - triangles[:, 0]
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
    displacement = target_vertices - source_vertices
    return np.einsum("ni,nij->nj", bary, displacement[source_faces[face_index]])


def _harmonic_field(
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
    registered_skin, surface_report = _screened_surface_registration(
        skin_vertices, skin_faces, target_vertices, target_faces
    )
    cage = _build_source_cage(
        skin_vertices, skin_faces, root / "source_skin_volume_cage_v5_4.npz"
    )
    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    elements = np.asarray(cage["elements"], dtype=np.int32)
    boundary = np.asarray(cage["boundary"], dtype=np.int64)
    boundary_values = _barycentric_displacement(
        nodes[boundary], skin_vertices, skin_faces, registered_skin
    )
    field = _harmonic_field(nodes, elements, boundary, boundary_values)
    query = np.asarray(asset.vertices_rest, dtype=np.float64)
    delta, outside_count, outside_mask = _sample_field(query, cage=cage, field=field)
    if outside_count:
        names: dict[str, int] = {}
        for name, (start, stop) in zip(asset.source_mesh_names, asset.source_vertex_ranges):
            count = int(np.count_nonzero(outside_mask[start:stop]))
            if count:
                names[str(name)] = count
        raise RuntimeError(
            f"source Skin_Glass domain excludes {outside_count} anatomy vertices: "
            f"{dict(list(names.items())[:12])}"
        )
    mapped = query + delta
    mapped, rigid_count = _preserve_rigid_bone_components(
        asset, source_vertices=query, field_vertices=mapped
    )
    mapped, material_report = _skin_material_scale_correction(
        asset, mapped, registered_skin, target_vertices, target_faces
    )
    rebound, rebind_report = rebind_source_rig(
        asset, source_vertices=query, target_vertices=mapped, stage="source_skin_to_neutral"
    )
    source_volume = np.linalg.det(nodes[elements][:, 1:] - nodes[elements][:, :1])
    target_nodes = nodes + field
    target_volume = np.linalg.det(target_nodes[elements][:, 1:] - target_nodes[elements][:, :1])
    inverted = int(np.count_nonzero(source_volume * target_volume <= 0.0))
    metadata = dict(rebound.metadata or {})
    metadata["source_skin_volume_registration"] = "screened_icp_harmonic_v5_4"
    result = type(rebound)(
        **{**rebound.__dict__, "vertices_rest": mapped.astype(np.float32), "metadata": metadata}
    )
    return result, {
        "backend": "screened_icp_harmonic_v5_4",
        "cage_nodes": int(len(nodes)),
        "cage_tetrahedra": int(len(elements)),
        "cage_voxel_pitch_m": float(np.asarray(cage["voxel_pitch"]).reshape(-1)[0]),
        "outside_query_count": 0,
        "diagnostic_inverted_tetrahedra": inverted,
        "rigid_bone_components": int(rigid_count),
        "skin_material_scale": material_report,
        "source_rig_rebind": rebind_report,
        **surface_report,
    }
