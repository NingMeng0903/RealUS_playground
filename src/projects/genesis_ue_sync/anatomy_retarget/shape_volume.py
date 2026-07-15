"""TetGen/FEM harmonic subject-beta deformation for internal anatomy."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset
from .source_rebind import rebind_source_rig
from .pose_adapter import smplx_shape_hash


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


def _solve_harmonic_field(
    cage: dict[str, np.ndarray],
    *,
    surface_displacement: np.ndarray,
) -> np.ndarray:
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
        for axis in range(3):
            field[interior, axis] = _solve_interior_harmonic(
                stiffness,
                interior,
                boundary,
                boundary_values[:, axis : axis + 1],
            ).reshape(-1)

    before = nodes[elements]
    after = (nodes + field)[elements]
    det0 = np.linalg.det(before[:, 1:] - before[:, :1])
    det1 = np.linalg.det(after[:, 1:] - after[:, :1])
    if np.any(det0 * det1 <= 0.0):
        raise RuntimeError("subject beta harmonic field flips one or more tetrahedra")
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
) -> tuple[np.ndarray, bool, str]:
    """Load/build the linear volume basis and combine it on CUDA when available."""
    weights_path = root / "smpl_canonical_weights.npz"
    weights = np.load(weights_path)
    if "shapedirs" not in weights.files:
        raise KeyError(f"{weights_path} does not contain SMPL-X shapedirs")
    shapedirs = np.asarray(weights["shapedirs"], dtype=np.float32)
    count = min(int(shapedirs.shape[2]), int(np.asarray(betas).size), 10)
    digest = hashlib.sha256()
    digest.update(np.asarray(cage["nodes"], dtype=np.float32).tobytes())
    digest.update(np.asarray(cage["elements"], dtype=np.int32).tobytes())
    digest.update(shapedirs[:, :, :count].tobytes())
    shared_cache = root.parent / "volume_beta_basis_v1"
    shared_cache.mkdir(parents=True, exist_ok=True)
    cache_path = shared_cache / f"{digest.hexdigest()[:24]}.npz"
    cache_hit = False
    basis: np.ndarray
    if cache_path.is_file():
        cached = np.load(cache_path)
        candidate = np.asarray(cached["field_basis"], dtype=np.float32)
        if candidate.shape == (len(cage["nodes"]), 3, count):
            basis = candidate
            cache_hit = True
        else:
            basis = _solve_harmonic_beta_basis(cage, shapedirs[:, :, :count])
    else:
        basis = _solve_harmonic_beta_basis(cage, shapedirs[:, :, :count])
    if not cache_hit:
        np.savez_compressed(cache_path, field_basis=basis)

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


def _beta_basis_digest(cage: dict[str, np.ndarray], shapedirs: np.ndarray, count: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(cage["nodes"], dtype=np.float32).tobytes())
    digest.update(np.asarray(cage["elements"], dtype=np.int32).tobytes())
    digest.update(np.asarray(shapedirs[:, :, :count], dtype=np.float32).tobytes())
    return digest.hexdigest()[:24]


def _load_or_sample_point_delta(
    *,
    root: Path,
    cage: dict[str, np.ndarray],
    field: np.ndarray,
    points: np.ndarray,
    betas: np.ndarray,
    basis_digest: str,
    gender: str,
) -> tuple[np.ndarray, int, np.ndarray, bool]:
    shape_key = smplx_shape_hash(betas, gender=gender)
    cache_dir = root.parent / "volume_beta_displacement_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{basis_digest}_{shape_key[:16]}.npz"
    if cache_path.is_file():
        cached = np.load(cache_path)
        point_delta = np.asarray(cached["point_delta"], dtype=np.float64)
        outside_mask = np.asarray(cached["outside_mask"], dtype=bool)
        if point_delta.shape == points.shape:
            max_norm = float(np.max(np.linalg.norm(point_delta, axis=1)))
            if max_norm <= 0.25:
                return point_delta, int(np.count_nonzero(outside_mask)), outside_mask, True
    point_delta, outside_points, outside_mask = _sample_field(points, cage=cage, field=field)
    np.savez_compressed(
        cache_path,
        point_delta=point_delta.astype(np.float32),
        outside_mask=outside_mask.astype(np.uint8),
    )
    return point_delta, outside_points, outside_mask, False


def _tet_barycentric(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Return tet barycentric weights for each (point, tet) pair.

    Each tet matrix stacks vertex rows ``[1, x, y, z]``; weights satisfy
    ``[1, px, py, pz] @ inv(system)`` (not ``inv(system) @ rhs``).
    """
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


def _shared_anisotropic_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one positive, axis-aligned material transform for a mesh group.

    A single transform preserves the relative layout of all meshes in the
    articulated region.  The axes and scales come entirely from the source
    group and the harmonic target samples; there are no anatomical offsets.
    """
    src_center = np.asarray(source, dtype=np.float64).mean(axis=0)
    dst_center = np.asarray(target, dtype=np.float64).mean(axis=0)
    centered = np.asarray(source, dtype=np.float64) - src_center
    _values, axes = np.linalg.eigh(np.cov(centered.T))
    src_local = centered @ axes
    dst_centered = np.asarray(target, dtype=np.float64) - dst_center
    u, _s, vt = np.linalg.svd(centered.T @ dst_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    target_axes = rotation @ axes
    dst_local = dst_centered @ target_axes
    src_extent = np.sqrt(np.mean(src_local * src_local, axis=0))
    dst_extent = np.sqrt(np.mean(dst_local * dst_local, axis=0))
    scales = np.divide(dst_extent, src_extent, out=np.ones(3), where=src_extent > 1.0e-8)
    linear = axes @ np.diag(scales) @ target_axes.T
    translation = dst_center - src_center @ linear
    return linear, translation


def _dominant_source_bone(asset: AnatomyRiggedAsset, start: int, stop: int) -> int | None:
    if asset.driver_indices is None or asset.driver_weights is None or asset.source_bone_names is None:
        return None
    indices = np.asarray(asset.driver_indices[start:stop], dtype=np.int64).reshape(-1)
    weights = np.asarray(asset.driver_weights[start:stop], dtype=np.float64).reshape(-1)
    mass = np.bincount(indices, weights=weights, minlength=len(asset.source_bone_names))
    return int(np.argmax(mass)) if mass.size and float(mass.max()) > 0.0 else None


def _has_ancestor(bone: int, ancestor: int, parents: np.ndarray) -> bool:
    current = int(bone)
    visited = 0
    while current >= 0 and visited <= len(parents):
        if current == int(ancestor):
            return True
        current = int(parents[current])
        visited += 1
    return False


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
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    types = list(asset.source_bone_driver_types or [])
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    dominant = [_dominant_source_bone(asset, int(start), int(stop)) for start, stop in ranges]
    processed: set[int] = set()
    groups: list[list[int]] = []

    # Each foot uses one rest-space material transform.  Individual toe bones
    # are never fitted separately, so they cannot drift away from the foot.
    for side in ("left", "right"):
        members = [
            idx for idx, (bone, tissue) in enumerate(zip(dominant, asset.source_tissues))
            if bone is not None
            and str(tissue) == "bone"
            and bone < len(types)
            and str(types[bone]) == f"foot_chain_{side}"
        ]
        if members:
            groups.append(members)

    for side in ("left", "right"):
        members = [
            idx for idx, (bone, tissue) in enumerate(zip(dominant, asset.source_tissues))
            if bone is not None
            and str(tissue) == "bone"
            and bone < len(types)
            and str(types[bone]) == f"hand_chain_{side}"
        ]
        if members:
            groups.append(members)

    # Skull, brain and jaw share the source Head_Bone material frame.  Their
    # source enclosure relationship therefore survives beta adaptation.
    head_index = (
        asset.source_bone_names.index("Head_Bone")
        if asset.source_bone_names is not None and "Head_Bone" in asset.source_bone_names
        else None
    )
    if head_index is not None:
        members = [
            idx for idx, (bone, tissue) in enumerate(zip(dominant, asset.source_tissues))
            if bone is not None
            and str(tissue) in {"bone", "nerve", "organ"}
            and _has_ancestor(bone, head_index, parents)
        ]
        if members:
            groups.append(members)

    for members in groups:
        indices = np.concatenate(
            [np.arange(int(ranges[idx, 0]), int(ranges[idx, 1]), dtype=np.int64) for idx in members]
        )
        linear, translation = _shared_anisotropic_fit(
            np.asarray(source_vertices[indices], dtype=np.float64),
            np.asarray(field_vertices[indices], dtype=np.float64),
        )
        output[indices] = np.asarray(source_vertices[indices], dtype=np.float64) @ linear + translation
        processed.update(members)

    rigid = len(groups)
    for mesh_idx, ((start, stop), tissue) in enumerate(zip(ranges, asset.source_tissues)):
        if str(tissue) != "bone" or int(stop - start) < 3:
            continue
        if mesh_idx in processed:
            continue
        src = np.asarray(source_vertices[start:stop], dtype=np.float64)
        dst = np.asarray(field_vertices[start:stop], dtype=np.float64)
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
    manifest_path = root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    beta_values = np.asarray(manifest.get("betas", []), dtype=np.float32).reshape(-1)
    basis_cache_hit = False
    basis_backend = "exact_surface_solve"
    displacement_cache_hit = False
    basis_digest = ""
    gender = str(manifest.get("gender", "neutral"))
    if beta_values.size and (root / "smpl_canonical_weights.npz").is_file():
        weights = np.load(root / "smpl_canonical_weights.npz")
        shapedirs = np.asarray(weights["shapedirs"], dtype=np.float64)
        count = min(shapedirs.shape[2], beta_values.size, 10)
        basis_digest = _beta_basis_digest(cage, shapedirs.astype(np.float32), count)
        predicted_surface_delta = np.tensordot(
            shapedirs[:, :, :count], beta_values[:count], axes=(2, 0)
        )
        surface_basis_error = float(
            np.max(np.linalg.norm((subject_v - neutral_v) - predicted_surface_delta, axis=1))
        )
        if surface_basis_error <= 1.0e-5:
            field, basis_cache_hit, basis_backend = _beta_volume_field(
                root=root, cage=cage, betas=beta_values
            )
        else:
            field = _solve_harmonic_field(cage, surface_displacement=subject_v - neutral_v)
            basis_backend = "exact_surface_solve_basis_mismatch"
    else:
        field = _solve_harmonic_field(cage, surface_displacement=subject_v - neutral_v)
        surface_basis_error = 0.0

    before = np.asarray(cage["nodes"], dtype=np.float64)[np.asarray(cage["elements"], dtype=np.int64)]
    after = (np.asarray(cage["nodes"], dtype=np.float64) + field)[np.asarray(cage["elements"], dtype=np.int64)]
    det0 = np.linalg.det(before[:, 1:] - before[:, :1])
    det1 = np.linalg.det(after[:, 1:] - after[:, :1])
    if np.any(det0 * det1 <= 0.0):
        raise RuntimeError("cached subject beta harmonic basis flips one or more tetrahedra")
    points = np.asarray(asset.vertices_rest, dtype=np.float64)
    if basis_digest and beta_values.size:
        point_delta, outside_points, outside_mask, displacement_cache_hit = _load_or_sample_point_delta(
            root=root,
            cage=cage,
            field=field,
            points=points,
            betas=beta_values,
            basis_digest=basis_digest,
            gender=gender,
        )
    else:
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
        "beta_basis_cache_hit": bool(basis_cache_hit),
        "beta_displacement_cache_hit": bool(displacement_cache_hit),
        "beta_basis_combine_backend": str(basis_backend),
        "surface_basis_error_m": float(surface_basis_error),
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
