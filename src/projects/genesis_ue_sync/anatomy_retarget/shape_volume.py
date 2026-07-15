"""TetGen/FEM harmonic subject-beta deformation for internal anatomy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .anatomy_roles import is_cranial_shell_mesh, is_foot_toe_mesh

from .rigged_asset import AnatomyRiggedAsset
from .source_rebind import rebind_source_rig


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


def _build_cage(neutral_v: np.ndarray, neutral_f: np.ndarray, *, cache_path: Path) -> dict[str, np.ndarray]:
    if cache_path.is_file():
        data = np.load(cache_path)
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
    )
    return {
        "nodes": nodes,
        "elements": elements,
        "boundary": boundary,
        "source_triangles": source_triangles,
        "source_bary": source_bary,
    }


def _tet_stiffness(nodes: np.ndarray, elements: np.ndarray):
    from scipy.sparse import coo_matrix

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for tet in np.asarray(elements, dtype=np.int64):
        xyz = nodes[tet]
        system = np.column_stack((np.ones(4), xyz))
        determinant = float(np.linalg.det(xyz[1:] - xyz[0]))
        volume = abs(determinant) / 6.0
        if volume <= 1.0e-18:
            raise RuntimeError("degenerate tetrahedron in neutral volume cage")
        gradients = np.linalg.inv(system)[1:, :]
        local = volume * (gradients.T @ gradients)
        for i in range(4):
            for j in range(4):
                rows.append(int(tet[i]))
                cols.append(int(tet[j]))
                values.append(float(local[i, j]))
    return coo_matrix((values, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()


def _solve_harmonic_field(
    cage: dict[str, np.ndarray],
    *,
    surface_displacement: np.ndarray,
) -> np.ndarray:
    from scipy.sparse.linalg import spsolve

    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    elements = np.asarray(cage["elements"], dtype=np.int32)
    boundary = np.asarray(cage["boundary"], dtype=np.int64)
    triangles = np.asarray(cage["source_triangles"], dtype=np.int64)
    bary = np.asarray(cage["source_bary"], dtype=np.float64)
    boundary_values = np.sum(surface_displacement[triangles] * bary[:, :, None], axis=1)
    interior = np.setdiff1d(np.arange(len(nodes), dtype=np.int64), boundary, assume_unique=False)
    field = np.zeros((len(nodes), 3), dtype=np.float64)
    field[boundary] = boundary_values
    if interior.size:
        stiffness = _tet_stiffness(nodes, elements)
        Kii = stiffness[interior][:, interior]
        Kib = stiffness[interior][:, boundary]
        for axis in range(3):
            field[interior, axis] = spsolve(Kii, -(Kib @ boundary_values[:, axis]))

    before = nodes[elements]
    after = (nodes + field)[elements]
    det0 = np.linalg.det(before[:, 1:] - before[:, :1])
    det1 = np.linalg.det(after[:, 1:] - after[:, :1])
    if np.any(det0 * det1 <= 0.0):
        raise RuntimeError("subject beta harmonic field flips one or more tetrahedra")
    return field


def _tet_barycentric(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    output = np.empty((len(points), 4), dtype=np.float64)
    for idx, (point, tet) in enumerate(zip(points, tetrahedra)):
        system = np.column_stack((np.ones(4), tet))
        output[idx] = np.asarray([1.0, *point], dtype=np.float64) @ np.linalg.inv(system)
    return output


def _tet_boundary_faces(elements: np.ndarray) -> np.ndarray:
    tet = np.asarray(elements, dtype=np.int64)
    faces = np.concatenate(
        (tet[:, [0, 1, 2]], tet[:, [0, 1, 3]], tet[:, [0, 2, 3]], tet[:, [1, 2, 3]]), axis=0
    )
    canonical = np.sort(faces, axis=1)
    _unique, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse] == 1]


def _extend_from_cage_boundary(
    points: np.ndarray,
    *,
    cage: dict[str, np.ndarray],
    field: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Continuous boundary extension for points excluded only by surface repair."""
    import igl

    nodes = np.asarray(cage["nodes"], dtype=np.float64)
    boundary_faces = _tet_boundary_faces(np.asarray(cage["elements"], dtype=np.int64))
    squared, face_index, closest = igl.point_mesh_squared_distance(
        np.asarray(points, dtype=np.float64), nodes, boundary_faces
    )
    selected = boundary_faces[np.asarray(face_index, dtype=np.int64)]
    bary = _triangle_barycentric(np.asarray(closest, dtype=np.float64), nodes[selected])
    displacement = np.sum(field[selected] * bary[:, :, None], axis=1)
    return displacement, float(np.sqrt(np.max(squared))) if len(squared) else 0.0


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
    element_index = np.asarray(igl.in_element(nodes, elements, np.asarray(points, dtype=np.float64), tree), dtype=np.int64)
    outside = element_index < 0
    displacement = np.zeros_like(points, dtype=np.float64)
    inside = ~outside
    if np.any(inside):
        selected = elements[element_index[inside]]
        bary = _tet_barycentric(points[inside], nodes[selected])
        displacement[inside] = np.sum(field[selected] * bary[:, :, None], axis=1)
    return displacement, int(np.count_nonzero(outside)), outside


def _warp_source_frames(
    asset: AnatomyRiggedAsset,
    *,
    cage: dict[str, np.ndarray],
    field: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if asset.source_rest_global is None:
        return None, None, 0
    source = np.asarray(asset.source_rest_global, dtype=np.float64)
    epsilon = 0.01
    origins = source[:, :3, 3]
    probes = [origins]
    for axis in range(3):
        probes.append(origins + epsilon * source[:, :3, axis])
    query = np.concatenate(probes, axis=0)
    delta, outside, outside_mask = _sample_field(query, cage=cage, field=field)
    if outside:
        from scipy.spatial import cKDTree

        nodes = np.asarray(cage["nodes"], dtype=np.float64)
        tree = cKDTree(nodes)
        _distance, nearest = tree.query(query, k=1)
        delta[outside_mask] = field[np.asarray(nearest, dtype=np.int64)[outside_mask]]
    warped = query + delta
    count = len(source)
    result = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    result[:, :3, 3] = warped[:count]
    for bi in range(count):
        axes = np.stack(
            [warped[(axis + 1) * count + bi] - warped[bi] for axis in range(3)], axis=1
        )
        U, _S, Vt = np.linalg.svd(axes)
        R = U @ Vt
        if np.linalg.det(R) < 0.0:
            U[:, -1] *= -1.0
            R = U @ Vt
        result[bi, :3, :3] = R
    return result.astype(np.float32), np.linalg.inv(result).astype(np.float32), outside


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best rigid map for an anatomical bone component after beta field warp."""
    src_center, dst_center = source.mean(axis=0), target.mean(axis=0)
    u, _s, vt = np.linalg.svd((source - src_center).T @ (target - dst_center))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1] *= -1.0
        rot = vt.T @ u.T
    return (source - src_center) @ rot.T + dst_center


def _cranial_shell_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Head-local anisotropic beta response for a rigid cranial shell.

    The harmonic field supplies the subject-specific target samples.  We keep
    its principal stretches (rather than shrinking the skull with a generic
    containment pass), while retaining a single rigid orientation.  This is a
    material fit and contains no position/direction constants.
    """
    src_center, dst_center = source.mean(axis=0), target.mean(axis=0)
    u, _s, vt = np.linalg.svd((source - src_center).T @ (target - dst_center))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1] *= -1.0
        rot = vt.T @ u.T
    _evals, axes = np.linalg.eigh(np.cov((source - src_center).T))
    source_local = (source - src_center) @ axes
    target_local = (target - dst_center) @ (rot @ axes)
    source_std = np.sqrt(np.mean(source_local * source_local, axis=0))
    target_std = np.sqrt(np.mean(target_local * target_local, axis=0))
    scale = np.divide(target_std, source_std, out=np.ones(3), where=source_std > 1.0e-8)
    return (source_local * scale) @ (rot @ axes).T + dst_center


def _preserve_rigid_bone_components(
    asset: AnatomyRiggedAsset,
    *,
    source_vertices: np.ndarray,
    field_vertices: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Do not let a harmonic soft-tissue field bend individual bone meshes."""
    output = np.asarray(field_vertices, dtype=np.float64).copy()
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return output, 0
    rigid = 0
    for mesh_name, (start, stop), tissue in zip(
        asset.source_mesh_names, np.asarray(asset.source_vertex_ranges, dtype=np.int64), asset.source_tissues
    ):
        if str(tissue) != "bone" or int(stop - start) < 3:
            continue
        src = np.asarray(source_vertices[start:stop], dtype=np.float64)
        dst = np.asarray(field_vertices[start:stop], dtype=np.float64)
        # Toe phalanges deliberately keep the authored foot hierarchy.  They
        # have no corresponding SMPL-X joints and should not be squeezed by a
        # body-shape field.  The skull keeps the field's anisotropic head scale
        # so it remains the outer shell around the brain.
        if is_foot_toe_mesh(str(mesh_name)):
            output[start:stop] = src
        elif is_cranial_shell_mesh(str(mesh_name)):
            output[start:stop] = _cranial_shell_fit(src, dst)
        else:
            output[start:stop] = _rigid_fit(src, dst)
        rigid += 1
    return output, rigid


def apply_subject_beta_shape(
    asset: AnatomyRiggedAsset,
    *,
    canonical_dir: Path | str,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Apply a subject-beta harmonic volume field to anatomy and source rig."""
    root = Path(canonical_dir)
    neutral_v, faces = _load_obj(root / "smpl_canonical_tpose_neutral.obj")
    subject_v, subject_faces = _load_obj(root / "smpl_canonical_tpose.obj")
    if neutral_v.shape != subject_v.shape or not np.array_equal(faces, subject_faces):
        raise ValueError("neutral and subject SMPL-X surfaces must share exact topology")
    cage = _build_cage(neutral_v, faces, cache_path=root / "neutral_volume_cage_v2.npz")
    field = _solve_harmonic_field(cage, surface_displacement=subject_v - neutral_v)
    points = np.asarray(asset.vertices_rest, dtype=np.float64)
    point_delta, outside_points, outside_mask = _sample_field(points, cage=cage, field=field)
    outside_by_tissue: dict[str, int] = {}
    if outside_points:
        extension, max_extension_m = _extend_from_cage_boundary(
            points[outside_mask], cage=cage, field=field
        )
        if asset.source_vertex_ranges is not None and asset.source_tissues is not None:
            for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
                outside_by_tissue[str(tissue)] = outside_by_tissue.get(str(tissue), 0) + int(
                    np.count_nonzero(outside_mask[start:stop])
                )
        # Outside points receive an extrapolated *beta displacement* from the
        # cage boundary.  This does not move them toward skin; subsequent
        # skeleton-section fitting handles rest containment as a whole field.
        point_delta[outside_mask] = extension
    else:
        max_extension_m = 0.0
    field_vertices = points + point_delta
    # The cage is a soft-tissue shape field.  Projecting it independently onto
    # bone vertices bends skull/rib/limb meshes and breaks their source rig.
    # Replace each bone mesh with its best rigid response to the same field.
    shaped_vertices, rigid_bone_components = _preserve_rigid_bone_components(
        asset, source_vertices=points, field_vertices=field_vertices
    )
    rebound, rebind_report = rebind_source_rig(
        asset, source_vertices=points, target_vertices=shaped_vertices, stage="subject_beta_volume"
    )

    skeleton = json.loads((root / "smpl_canonical_skeleton.json").read_text(encoding="utf-8"))
    meta = dict(rebound.metadata or {})
    meta["shape_deformation"] = "tetgen_fem_harmonic_v5_soft_tissue"
    result = type(asset)(
        **{
            **rebound.__dict__,
            "vertices_rest": shaped_vertices.astype(np.float32),
            "rest_joints": np.asarray(skeleton["rest_joints_subject"], dtype=np.float32),
            "inverse_bind": np.asarray(skeleton["inverse_bind"], dtype=np.float32),
            "metadata": meta,
        }
    )
    norm = np.linalg.norm(point_delta, axis=1)
    return result, {
        "backend": "tetgen_fem_harmonic_v5_soft_tissue",
        "tetra_vertices": int(len(cage["nodes"])),
        "tetrahedra": int(len(cage["elements"])),
        "mean_displacement_m": float(np.mean(norm)),
        "max_displacement_m": float(np.max(norm)),
        "outside_query_count": int(outside_points),
        "outside_query_by_tissue": outside_by_tissue,
        "max_cage_boundary_extension_m": float(max_extension_m),
        "rigid_bone_components": int(rigid_bone_components),
        "source_rig_rebind": rebind_report,
    }
