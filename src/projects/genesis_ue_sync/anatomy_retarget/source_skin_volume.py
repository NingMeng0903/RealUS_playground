"""Offline Skin_Glass -> subject SMPL-X volumetric registration.

The Blender skin is used only as a material boundary.  Internal anatomy is
transported by one continuous harmonic volume field; no anatomy vertex is
individually projected or clamped to the SMPL-X surface.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset
from .anatomy_lbs import with_source_driver_coupling
from .material_fit import (
    _fit_source_frames,
    cranial_material_mask,
    rigid_head_attachment_mask,
)
from .shape_volume import _load_obj, _outside_cage_max_distance, _sample_field, _tet_stiffness


_CAGE_VERSION = "source_skin_volume_v22_shared_bind_field"
# A Stage-1 field is a diffeomorphic initialisation, not a nearly-flat cage.
_MIN_JACOBIAN_RATIO = 0.05
_MAX_FINAL_SURFACE_RMS_M = 0.03
_MAX_FINAL_SURFACE_DISTANCE_M = 0.10
_MAX_BOUNDARY_DISPLACEMENT_M = 0.50
_MIN_REGISTRATION_PROGRESS_M = 1.0e-7
_FINE_SHELL_HOMOTOPY = (0.10, 0.25, 0.45, 0.65, 0.82, 1.00)
# The semantic guide is an orientation-preserving initialisation, not the
# subject-shape solve itself.  Slender/extreme betas can need a smaller first
# homotopy step even though the later shell registration remains full-strength.
_SEMANTIC_PREALIGN_CANDIDATES = (
    0.375,
    0.3125,
    0.30,
    0.25,
    0.20,
    0.15,
    0.125,
    0.10,
    0.075,
)
_SEMANTIC_PREALIGN_MIN_PROBE_STRETCH = 0.80
_SEMANTIC_PREALIGN_MAX_PROBE_STRETCH = 1.22
_SOFT_VOLUME_TISSUES_V811 = frozenset(
    {"vessel", "nerve", "organ", "heart", "connective_tissue"}
)


def _volume_transport_digest_v811(
    asset: AnatomyRiggedAsset,
    transport_domain: np.ndarray,
    protected_domain: np.ndarray,
) -> str:
    """Digest both the moved soft domain and frozen hard protection domain."""

    digest = hashlib.sha256(b"source-skin-volume-registration-v811\0")
    arrays = (
        np.asarray(transport_domain, dtype=np.uint8),
        np.asarray(protected_domain, dtype=np.uint8),
        np.asarray(asset.vertices_rest, dtype=np.float32)[transport_domain],
        np.asarray(asset.vertices_rest, dtype=np.float32)[protected_domain],
        np.asarray(asset.faces, dtype=np.int32),
        np.asarray(asset.driver_indices, dtype=np.int32),
        np.asarray(asset.driver_weights, dtype=np.float32),
        np.asarray(asset.rest_joints, dtype=np.float32),
        np.asarray(asset.inverse_bind, dtype=np.float32),
    )
    for array in arrays:
        packed = np.ascontiguousarray(array)
        digest.update(np.asarray(packed.shape, dtype=np.int64).tobytes())
        digest.update(packed.dtype.str.encode("ascii"))
        digest.update(packed.tobytes())
    return digest.hexdigest()


def soft_volume_material_mask_v811(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Return tissue-eligible vertices before hard-compound protection.

    The harmonic field is a useful topology-preserving transport for thin
    anatomy, but it is not a replacement for the authored source rig.  In
    particular it must never drift arbitrary unclassified meshes.  The
    craniocerebral protection domain is resolved separately from the source
    bone hierarchy because the brain is intentionally labelled ``organ``.
    """

    count = int(len(np.asarray(asset.vertices_rest)))
    ranges = asset.source_vertex_ranges
    tissues = asset.source_tissues
    if ranges is None or tissues is None:
        raise ValueError(
            "V8.11 source-skin volume transport requires mesh tissue metadata"
        )
    spans = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
    if len(spans) != len(tissues):
        raise ValueError("source mesh ranges and tissues do not match")
    mask = np.zeros(count, dtype=bool)
    for (start, stop), tissue in zip(spans, tissues, strict=True):
        lo = int(start)
        hi = int(stop)
        if lo < 0 or hi < lo or hi > count:
            raise ValueError("source mesh range is outside anatomy vertices")
        if str(tissue).strip().lower() in _SOFT_VOLUME_TISSUES_V811:
            mask[lo:hi] = True
    return mask


def rigid_hard_protection_mask_v811(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Return bone plus the rigid skull/brain/upper-teeth protection domain.

    Tissue labels alone cannot distinguish a movable organ from the brain.
    The latter is part of the rigid head compound when it is wholly attached
    to the Blender head subtree.  This exact mask is shared by the volume
    registration and beta-basis bake so neither path can quietly move it.
    """

    count = int(len(np.asarray(asset.vertices_rest)))
    ranges = getattr(asset, "source_vertex_ranges", None)
    tissues = getattr(asset, "source_tissues", None)
    if ranges is None or tissues is None:
        raise ValueError("V8.11 hard protection requires mesh tissue metadata")
    spans = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
    if len(spans) != len(tissues):
        raise ValueError("source mesh ranges and tissues do not match")
    bone = np.zeros(count, dtype=bool)
    for (start, stop), tissue in zip(spans, tissues, strict=True):
        lo = int(start)
        hi = int(stop)
        if lo < 0 or hi < lo or hi > count:
            raise ValueError("source mesh range is outside anatomy vertices")
        if str(tissue).strip().lower() == "bone":
            bone[lo:hi] = True

    # Tiny structural tests sometimes carry only tissue metadata.  A real
    # anatomy asset always has this hierarchy, and without it no head subtree
    # can be inferred safely.
    if any(
        getattr(asset, field, None) is None
        for field in (
            "source_bone_names",
            "source_bone_parents",
            "driver_indices",
            "driver_weights",
            "source_mesh_names",
        )
    ):
        return bone
    cranial = np.asarray(cranial_material_mask(asset), dtype=bool)
    rigid_head = np.asarray(rigid_head_attachment_mask(asset), dtype=bool)
    if cranial.shape != (count,) or rigid_head.shape != (count,):
        raise ValueError("V8.11 cranial protection masks do not match topology")
    return bone | (cranial & rigid_head)


def soft_volume_transport_mask_v811(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Return the only vertices a V8.11 source-skin field may move."""

    soft = soft_volume_material_mask_v811(asset)
    hard = rigid_hard_protection_mask_v811(asset)
    if soft.shape != hard.shape:
        raise ValueError("V8.11 soft and hard transport masks do not match")
    return soft & ~hard


def _semantic_rest_prealign(
    asset: AnatomyRiggedAsset,
    *,
    legacy_frames: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate the v35 semantic rest alignment without changing rig authority."""
    if (
        asset.source_skin_vertices is None
        or asset.source_skin_lbs_weights is None
        or asset.driver_indices is None
        or asset.driver_weights is None
        or asset.source_bone_smplx_a is None
    ):
        raise RuntimeError(
            "semantic Stage-1 prealign requires Skin_Glass and the complete source rig"
        )
    _target_global, _target_local, source_delta = _fit_source_frames(
        asset,
        preserve_same_semantic_offset=not legacy_frames,
    )
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    source_indices = np.asarray(asset.driver_indices, dtype=np.int64)
    source_weights = np.asarray(asset.driver_weights, dtype=np.float64)
    blended = np.sum(
        source_weights[..., None, None] * source_delta[source_indices], axis=1
    )
    anatomy = np.einsum(
        "nij,nj->ni",
        blended,
        np.concatenate((vertices, np.ones((len(vertices), 1))), axis=1),
    )[:, :3]

    semantic = np.asarray(asset.source_bone_smplx_a, dtype=np.int64)
    joint_count = len(asset.joint_names)
    skin_weights = np.asarray(asset.source_skin_lbs_weights, dtype=np.float64)
    if skin_weights.shape != (len(asset.source_skin_vertices), joint_count):
        raise ValueError("Skin_Glass weights must match the SMPL-X joint order")
    mass = np.bincount(
        source_indices.reshape(-1),
        weights=source_weights.reshape(-1),
        minlength=len(semantic),
    )
    semantic_delta = np.tile(np.eye(4, dtype=np.float64), (joint_count, 1, 1))
    mapped_joint_count = 0
    for joint in range(joint_count):
        candidates = np.flatnonzero(semantic == joint)
        if not len(candidates):
            continue
        semantic_delta[joint] = source_delta[candidates[np.argmax(mass[candidates])]]
        mapped_joint_count += 1
    skin = np.asarray(asset.source_skin_vertices, dtype=np.float64)
    skin_blended = np.einsum("nj,jkl->nkl", skin_weights, semantic_delta)
    skin_aligned = np.einsum(
        "nij,nj->ni",
        skin_blended,
        np.concatenate((skin, np.ones((len(skin), 1))), axis=1),
    )[:, :3]
    anatomy_displacement = np.linalg.norm(anatomy - vertices, axis=1)
    skin_displacement = np.linalg.norm(skin_aligned - skin, axis=1)
    return anatomy, skin_aligned, source_delta, {
        "backend": "v35_source_rig_semantic_rest_lbs_v1",
        "preserves_source_weights": True,
        "preserves_source_hierarchy": True,
        "mapped_semantic_joint_count": int(mapped_joint_count),
        "anatomy_displacement_rms_m": float(
            np.sqrt(np.mean(anatomy_displacement**2))
        ),
        "anatomy_displacement_max_m": float(np.max(anatomy_displacement)),
        "skin_displacement_rms_m": float(np.sqrt(np.mean(skin_displacement**2))),
        "skin_displacement_max_m": float(np.max(skin_displacement)),
    }


def _rebind_source_rig_from_volume_field(
    asset: AnatomyRiggedAsset,
    *,
    cage: dict[str, np.ndarray],
    field: np.ndarray,
    semantic_prealign_delta: np.ndarray | None = None,
    semantic_prealign_blend: float = 0.0,
    from_target_binding: bool = False,
    stage: str = "stage1_harmonic_volume",
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Map source bind frames through the same field as anatomy vertices."""
    source_global = np.asarray(
        asset.target_bind_global if from_target_binding else asset.source_bind_global,
        dtype=np.float64,
    )
    source_head = np.asarray(
        asset.target_bone_head if from_target_binding else asset.source_bone_head,
        dtype=np.float64,
    )
    source_tail = np.asarray(
        asset.target_bone_tail if from_target_binding else asset.source_bone_tail,
        dtype=np.float64,
    )
    count = len(source_global)
    epsilon = 1.0e-3
    origins = source_global[:, :3, 3]
    axes = source_global[:, :3, :3]
    probes = np.concatenate(
        (
            origins,
            (origins[:, None, :] + epsilon * np.transpose(axes, (0, 2, 1))).reshape(-1, 3),
            source_head,
            source_tail,
        ),
        axis=0,
    )
    if semantic_prealign_delta is not None:
        prealign_delta = np.asarray(semantic_prealign_delta, dtype=np.float64)
        if prealign_delta.shape != source_global.shape:
            raise ValueError("semantic prealign delta must match the source rig")
        probe_bones = np.concatenate(
            (
                np.arange(count, dtype=np.int64),
                np.repeat(np.arange(count, dtype=np.int64), 3),
                np.arange(count, dtype=np.int64),
                np.arange(count, dtype=np.int64),
            )
        )
        homogeneous = np.concatenate(
            (probes, np.ones((len(probes), 1), dtype=np.float64)), axis=1
        )
        prealigned = np.einsum(
            "nij,nj->ni", prealign_delta[probe_bones], homogeneous
        )[:, :3]
        blend = float(semantic_prealign_blend)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("semantic prealign blend must be in [0, 1]")
        probes = probes + blend * (prealigned - probes)
    probe_delta, _outside_count, outside = _sample_field(
        probes, cage=cage, field=field
    )
    if np.any(outside):
        raise RuntimeError(
            f"source bind probes outside harmonic cage: {int(np.count_nonzero(outside))}"
        )
    mapped = probes + probe_delta
    mapped_origin = mapped[:count]
    mapped_axes = mapped[count : count + 3 * count].reshape(count, 3, 3)
    mapped_head = mapped[count + 3 * count : count + 4 * count]
    mapped_tail = mapped[count + 4 * count :]
    target_global = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
    target_global[:, :3, 3] = mapped_origin
    minimum_stretch = float("inf")
    maximum_stretch = 0.0
    minimum_stretch_bone = -1
    maximum_stretch_bone = -1
    for bone in range(count):
        deformed_basis = (
            mapped_axes[bone] - mapped_origin[bone][None, :]
        ).T / epsilon
        u, singular, vt = np.linalg.svd(deformed_basis)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        target_global[bone, :3, :3] = rotation
        local_minimum = float(np.min(singular))
        local_maximum = float(np.max(singular))
        if local_minimum < minimum_stretch:
            minimum_stretch = local_minimum
            minimum_stretch_bone = bone
        if local_maximum > maximum_stretch:
            maximum_stretch = local_maximum
            maximum_stretch_bone = bone

    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    target_local = target_global.copy()
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            target_local[bone] = (
                np.linalg.inv(target_global[int(parent)]) @ target_global[bone]
            )
    metadata = dict(asset.metadata or {})
    report = {
        "stage": str(stage),
        "backend": "shared_volume_field_bind_probes_v1",
        "probe_count": int(len(probes)),
        "minimum_probe_stretch": minimum_stretch,
        "maximum_probe_stretch": maximum_stretch,
        "minimum_probe_stretch_bone": (
            str(asset.source_bone_names[minimum_stretch_bone])
            if minimum_stretch_bone >= 0
            else None
        ),
        "maximum_probe_stretch_bone": (
            str(asset.source_bone_names[maximum_stretch_bone])
            if maximum_stretch_bone >= 0
            else None
        ),
        "semantic_prealign_blend": float(semantic_prealign_blend),
    }
    history = list(metadata.get("source_rig_rebind") or [])
    history.append(report)
    metadata["source_rig_rebind"] = history
    result = type(asset)(
        **{
            **asset.__dict__,
            "target_rest_global": target_global.astype(np.float32),
            "target_rest_local": target_local.astype(np.float32),
            "target_inverse_bind": np.linalg.inv(target_global).astype(np.float32),
            "target_bone_head": mapped_head.astype(np.float32),
            "target_bone_tail": mapped_tail.astype(np.float32),
            "metadata": metadata,
        }
    )
    return result, report


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


def _voxel_union(
    vertices: np.ndarray, faces: np.ndarray, *, dilation_iterations: int = 1
):
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
    if int(dilation_iterations) > 0:
        occupancy = ndimage.binary_dilation(
            occupancy, iterations=int(dilation_iterations)
        )
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


def _semantic_skin_surface_correspondence(
    source_vertices: np.ndarray,
    guide_vertices: np.ndarray,
    source_weights: np.ndarray,
    target_vertices: np.ndarray,
    target_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match skin branches on the target surface by SMPL-X weight semantics."""
    from scipy.spatial import cKDTree

    source = np.asarray(source_vertices, dtype=np.float64)
    guide = np.asarray(guide_vertices, dtype=np.float64)
    source_lbs = np.asarray(source_weights, dtype=np.float64)
    target = np.asarray(target_vertices, dtype=np.float64)
    target_lbs = np.asarray(target_weights, dtype=np.float64)
    if source.shape != guide.shape or source_lbs.shape[0] != len(source):
        raise ValueError("source skin guide and weights must match source vertices")
    if target_lbs.shape[0] != len(target) or source_lbs.shape[1] != target_lbs.shape[1]:
        raise ValueError("source and target skin weights must share the joint order")
    source_label = np.argmax(source_lbs, axis=1)
    target_label = np.argmax(target_lbs, axis=1)
    mapped = np.empty_like(source)
    fallback = 0
    matched_counts: dict[str, int] = {}
    global_tree = cKDTree(target)
    for joint in np.unique(source_label).tolist():
        rows = np.flatnonzero(source_label == int(joint))
        candidates = np.flatnonzero(target_label == int(joint))
        if not len(candidates):
            _distance, selected = global_tree.query(guide[rows], k=1)
            mapped[rows] = target[np.asarray(selected, dtype=np.int64)]
            fallback += int(len(rows))
            continue
        _distance, local = cKDTree(target[candidates]).query(guide[rows], k=1)
        mapped[rows] = target[candidates[np.asarray(local, dtype=np.int64)]]
        matched_counts[str(int(joint))] = int(len(rows))

    guide_error = np.linalg.norm(mapped - guide, axis=1)
    return mapped, {
        "backend": "dominant_smplx_weight_constrained_surface_v1",
        "source_vertex_count": int(len(source)),
        "target_vertex_count": int(len(target)),
        "matched_joint_count": int(len(matched_counts)),
        "fallback_vertex_count": int(fallback),
        "guide_to_target_rms_m": float(np.sqrt(np.mean(guide_error**2))),
        "guide_to_target_max_m": float(np.max(guide_error)),
        "vertices_by_joint": matched_counts,
    }


def _outer_body_component(
    vertices: np.ndarray,
    faces: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Keep the sole external SMPL-X body component, never eyes or teeth.

    The canonical SMPL-X OBJ contains the body plus two closed eyeball
    components.  They are valid render geometry but invalid outer-volume
    boundary candidates: a nearest-point or signed-distance query can jump to
    an eyeball while registering the face.  The Skin_Glass field has exactly
    one outer boundary, so select its largest connected component and reindex
    the associated LBS weights with it.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    verts = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    skin_weights = np.asarray(weights, dtype=np.float64)
    if skin_weights.shape[0] != len(verts):
        raise ValueError("SMPL-X weights do not match the canonical OBJ vertices")
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
    )
    graph = coo_matrix(
        (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
        shape=(len(verts), len(verts)),
    )
    component_count, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, minlength=component_count)
    body_label = int(np.argmax(counts))
    keep = np.flatnonzero(labels == body_label)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep), dtype=np.int64)
    selected = np.all(labels[triangles] == body_label, axis=1)
    body_faces = remap[triangles[selected]]
    return verts[keep], body_faces.astype(np.int32), skin_weights[keep], {
        "input_components": int(component_count), "input_vertices": int(len(verts)),
        "input_faces": int(len(triangles)), "body_vertices": int(len(keep)),
        "body_faces": int(len(body_faces)),
    }


def _closed_solver_shell(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    resolution: int = 360,
    inward_margin_m: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a closed, strictly in-subject shell for Stage-1 registration.

    SMPL-X body render meshes contain a small facial boundary loop.  The
    harmonic boundary is still the authored body mesh; only inside/outside
    diagnostics use this voxel-closed shell.  This avoids any SDF feedback or
    geometry repair of anatomy itself.
    """
    import trimesh
    from scipy import ndimage
    from .containment import signed_distance

    body = trimesh.Trimesh(vertices, faces, process=True)
    pitch = float(np.max(body.extents) / max(int(resolution), 32))
    grid = body.voxelized(pitch).fill()
    occupancy = ndimage.binary_erosion(np.asarray(grid.matrix, dtype=bool), iterations=1)
    shell = trimesh.voxel.VoxelGrid(occupancy, transform=grid.transform).marching_cubes
    shell.apply_transform(grid.transform)
    shell.remove_unreferenced_vertices()
    shell.fix_normals()
    shell_vertices = np.asarray(shell.vertices, dtype=np.float64)
    signed, closest, normals = signed_distance(shell_vertices, vertices, faces)
    margin = float(inward_margin_m)
    needs_inset = signed > -margin
    shell_vertices[needs_inset] = (
        closest[needs_inset] - margin * normals[needs_inset]
    )
    verified, _closest, _normals = signed_distance(shell_vertices, vertices, faces)
    if float(np.max(verified)) > 1.0e-6:
        raise RuntimeError("closed subject solver shell is not fully inside SMPL-X")
    return (
        shell_vertices,
        np.asarray(shell.faces, dtype=np.int32),
        float(pitch),
    )


def _coarse_closed_solver_shell(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the robust, low-resolution closed shell used for initialization.

    The first correspondence cannot jump directly from the source Skin_Glass
    cage to the five-millimetre inset shell: that is a valid final target but
    can violate the cage Jacobian barrier.  This closed shell is only an
    initialization domain; the final boundary is refined to the strictly
    in-subject shell below.
    """
    return _closed_solver_shell(vertices, faces, resolution=180, inward_margin_m=0.0)


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a tiny deterministic OBJ for Stage-1 inspection."""
    with path.open("w", encoding="utf-8") as stream:
        for point in np.asarray(vertices, dtype=np.float64):
            stream.write(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
        for triangle in np.asarray(faces, dtype=np.int64):
            stream.write(
                f"f {int(triangle[0]) + 1} {int(triangle[1]) + 1} {int(triangle[2]) + 1}\n"
            )


def _export_stage1_debug(
    directory: Path,
    *,
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    shell_vertices: np.ndarray,
    shell_faces: np.ndarray,
    source_skin: np.ndarray,
    source_skin_faces: np.ndarray,
    stage1_skin: np.ndarray,
    stage1_skin_faces: np.ndarray,
    stage1_cage: np.ndarray,
    cage_faces: np.ndarray,
    stage1_anatomy: np.ndarray,
    report: dict[str, Any],
) -> None:
    """Export the actual pre-bone-fit field, not merely semantic point targets."""
    directory.mkdir(parents=True, exist_ok=True)
    _write_obj(directory / "target_body.obj", body_vertices, body_faces)
    _write_obj(directory / "target_solver_shell.obj", shell_vertices, shell_faces)
    _write_obj(directory / "source_skin_glass.obj", source_skin, source_skin_faces)
    _write_obj(directory / "stage1_skin_glass.obj", stage1_skin, stage1_skin_faces)
    _write_obj(directory / "stage1_cage_boundary.obj", stage1_cage, cage_faces)
    np.savez_compressed(
        directory / "stage1_anatomy.npz",
        vertices=np.asarray(stage1_anatomy, dtype=np.float32),
    )
    (directory / "stage1_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Matplotlib is intentionally diagnostic-only.  The opaque target body and
    # the deformed Skin_Glass are both rendered as triangle meshes so a point
    # cloud cannot hide a cross-face/foldover error.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15, 5))
    views = ((20, -90, "front"), (12, 0, "side"), (72, -35, "top"))
    all_points = np.vstack((body_vertices, stage1_skin))
    lower = np.min(all_points, axis=0)
    upper = np.max(all_points, axis=0)
    center = 0.5 * (lower + upper)
    radius = float(np.max(upper - lower) * 0.55)
    for index, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.plot_trisurf(
            body_vertices[:, 0], body_vertices[:, 1], body_vertices[:, 2],
            triangles=body_faces, color="#f0a060", alpha=0.22,
            linewidth=0.0, shade=False,
        )
        axis.plot_trisurf(
            stage1_skin[:, 0], stage1_skin[:, 1], stage1_skin[:, 2],
            triangles=stage1_skin_faces, color="#1565c0", alpha=0.36,
            linewidth=0.0, shade=False,
        )
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title)
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(directory / "stage1_skin_overlay.png", dpi=180)
    plt.close(figure)


def _build_source_cage(
    vertices: np.ndarray,
    faces: np.ndarray,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    # The material domain is defined exclusively by the authored Skin_Glass.
    # Keeping the fixed one-voxel margin inside this function prevents callers
    # from enlarging the domain to accommodate anatomy query points.
    signature = _signature(vertices, faces)
    if cache_path.is_file():
        data = np.load(cache_path)
        cached = str(np.asarray(data.get("signature", "")).reshape(-1)[0])
        if cached == signature:
            return {key: np.asarray(data[key]) for key in data.files}

    import tetgen

    surface, pitch = _voxel_union(vertices, faces, dilation_iterations=1)
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
    initial_boundary: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit the closed cage and reject no-op or low-quality registrations."""
    import igl
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import factorized, splu

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
        (
            (
                (1000000.0, 10),
                (300000.0, 8),
                (100000.0, 10),
                (30000.0, 12),
                (10000.0, 8),
                (3000.0, 8),
                (1000.0, 10),
                (300.0, 12),
                (100.0, 15),
                (30.0, 20),
            )
            if fixed is not None
            else (
                (1000000.0, 10),
                (300000.0, 8),
                (100000.0, 10),
                (30000.0, 12),
                (10000.0, 12),
                (3000.0, 10),
                (1000.0, 10),
                (300.0, 12),
            )
        )
    )
    differential = smoothness @ source
    registered = (
        source.copy()
        if initial_boundary is None
        else np.asarray(initial_boundary, dtype=np.float64).reshape(len(source), 3).copy()
    )
    base_tet = original[elements]
    base_det = np.linalg.det(base_tet[:, 1:] - base_tet[:, :1])
    if np.any(~np.isfinite(base_det)) or np.any(np.abs(base_det) <= 1.0e-18):
        raise RuntimeError("source volume cage contains a degenerate tetrahedron")
    if initial_boundary is not None:
        initial_field = _harmonic_step(original, elements, boundary, registered - source)
        initial_ratio = np.linalg.det(
            (original + initial_field)[elements][:, 1:]
            - (original + initial_field)[elements][:, :1]
        ) / base_det
        if np.any(~np.isfinite(initial_ratio) | (initial_ratio < _MIN_JACOBIAN_RATIO)):
            raise RuntimeError("initial Stage-1 boundary is not Jacobian-safe")
    interior = np.setdiff1d(np.arange(len(original), dtype=np.int64), boundary)
    stiffness = _tet_stiffness(original, elements)
    harmonic_solver = None if not len(interior) else splu(
        stiffness[interior][:, interior].tocsc()
    )

    def solve_original_harmonic(boundary_values: np.ndarray) -> np.ndarray:
        return _harmonic_step(
            original,
            elements,
            boundary,
            boundary_values,
            interior=interior,
            stiffness=stiffness,
            solver=harmonic_solver,
        )
    # Surface progress is measured against the actual SMPL-X shell even when
    # a semantic correspondence is present.  The latter is a separate
    # Dirichlet objective and cannot be compared numerically to point-to-shell
    # residuals (doing so falsely rejected a genuine 36 -> 31 mm improvement).
    initial_squared, _face_index, _closest = igl.point_mesh_squared_distance(
        source, target, target_faces
    )
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
            # A constrained proposal can be valid after a short continuous
            # step even when its full update flattens a thin tetrahedron.  The
            # former lock-only behaviour rejected the entire registration at
            # the first such node.  Backtracking changes only the boundary
            # step size; the applied field remains one harmonic solve.
            step_fraction = 1.0
            for _barrier_iteration in range(16):
                trial_boundary = registered + step_fraction * (proposal - registered)
                proposal_field = solve_original_harmonic(trial_boundary - source)
                trial = original + proposal_field
                trial_tet = trial[elements]
                ratio = np.linalg.det(trial_tet[:, 1:] - trial_tet[:, :1]) / base_det
                # Zero flips is insufficient for thin vessels: a nearly flat
                # tet creates an arbitrarily high-gradient interior field.
                bad = np.flatnonzero(
                    (~np.isfinite(ratio)) | (ratio <= 0.0) | (ratio < _MIN_JACOBIAN_RATIO)
                )
                if not len(bad):
                    proposal = trial_boundary
                    accepted = True
                    break
                # Prefer a smaller globally continuous update before pinning
                # vertices.  Locking first makes a local thin tet freeze an
                # entire limb's correspondence on the first iteration.
                if step_fraction > 1.0 / 2048.0:
                    step_fraction *= 0.5
                    continue
                bad_boundary = local_index[np.unique(elements[bad])]
                bad_boundary = bad_boundary[bad_boundary >= 0]
                newly_locked = bad_boundary[~locked[bad_boundary]]
                if len(newly_locked):
                    locked[newly_locked] = True
                    proposal[locked] = registered[locked]
                    step_fraction = 1.0
                    continue
                break
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
        fixed_final_rms = 0.0
        fixed_final_max = 0.0
    else:
        fixed_squared = np.sum((registered - fixed) ** 2, axis=1)
        fixed_final_rms = float(np.sqrt(np.mean(fixed_squared)))
        fixed_final_max = float(np.sqrt(np.max(fixed_squared)))
        squared, _face_index, _closest = igl.point_mesh_squared_distance(
            registered, target, target_faces
        )
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
    maximum_surface_rms = 0.04 if fixed is not None else _MAX_FINAL_SURFACE_RMS_M
    maximum_surface_distance = (
        0.12 if fixed is not None else _MAX_FINAL_SURFACE_DISTANCE_M
    )
    if final_rms > maximum_surface_rms or final_max > maximum_surface_distance:
        raise RuntimeError(
            "surface registration residual exceeds production limits "
            f"(RMS={final_rms:.6f}/{maximum_surface_rms:.6f} m, "
            f"max={final_max:.6f}/{maximum_surface_distance:.6f} m)"
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
        "fixed_correspondence_rms_m": fixed_final_rms,
        "fixed_correspondence_max_m": fixed_final_max,
    }


def _harmonic_step(
    nodes: np.ndarray,
    elements: np.ndarray,
    boundary: np.ndarray,
    boundary_values: np.ndarray,
    *,
    interior: np.ndarray | None = None,
    stiffness: Any | None = None,
    solver: Any | None = None,
) -> np.ndarray:
    from scipy.sparse.linalg import spsolve

    field = np.zeros_like(nodes, dtype=np.float64)
    field[boundary] = boundary_values
    if interior is None:
        interior = np.setdiff1d(np.arange(len(nodes)), boundary)
    else:
        interior = np.asarray(interior, dtype=np.int64)
    if stiffness is None:
        stiffness = _tet_stiffness(nodes, elements)
    if len(interior):
        kii = stiffness[interior][:, interior]
        kib = stiffness[interior][:, boundary]
        right_hand_side = np.asarray(
            -(kib @ field[boundary]), dtype=np.float64
        )
        if solver is not None:
            field[interior] = solver.solve(right_hand_side)
        else:
            for axis in range(3):
                field[interior, axis] = spsolve(kii, right_hand_side[:, axis])
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


def _jacobian_safe_harmonic_boundary_field(
    nodes: np.ndarray,
    elements: np.ndarray,
    boundary: np.ndarray,
    boundary_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Solve a Dirichlet field while freezing only locally unsafe boundary nodes.

    The inverse Skin_Glass counterpart differs from the SMPL-X shell mostly at
    authored openings and very thin layers.  Globally damping that displacement
    destroys the semantic map everywhere.  Instead, identify the boundary
    degrees of freedom incident to a bad tetrahedron, keep those few nodes at
    the common-topology reference position, and re-solve the same harmonic
    problem.  No anatomy point is projected or repaired here.
    """
    from scipy.spatial import cKDTree

    reference = np.asarray(nodes, dtype=np.float64)
    tetrahedra = np.asarray(elements, dtype=np.int64)
    outer = np.asarray(boundary, dtype=np.int64)
    values = np.asarray(boundary_values, dtype=np.float64).copy()
    if values.shape != (len(outer), 3):
        raise ValueError("boundary displacement must have shape [boundary, 3]")
    reference_tet = reference[tetrahedra]
    reference_det = np.linalg.det(reference_tet[:, 1:] - reference_tet[:, :1])
    if np.any(~np.isfinite(reference_det)) or np.any(np.abs(reference_det) <= 1.0e-18):
        raise RuntimeError("common-topology volume cage contains a degenerate tetrahedron")

    boundary_tree = cKDTree(reference[outer])
    _nearest_distance, _nearest_index = boundary_tree.query(reference[outer], k=2)
    shell_spacing = float(np.median(_nearest_distance[:, 1]))
    original_values = values.copy()
    attenuation = np.ones(len(outer), dtype=np.float64)
    initial_bad = 0
    final_ratio = np.ones(len(tetrahedra), dtype=np.float64)
    for iteration in range(32):
        field = _harmonic_step(reference, tetrahedra, outer, values)
        deformed_tet = (reference + field)[tetrahedra]
        final_ratio = (
            np.linalg.det(deformed_tet[:, 1:] - deformed_tet[:, :1])
            / reference_det
        )
        bad = np.flatnonzero(
            (~np.isfinite(final_ratio)) | (final_ratio < _MIN_JACOBIAN_RATIO)
        )
        if iteration == 0:
            initial_bad = int(len(bad))
        if not len(bad):
            return field, {
                "backend": "local_jacobian_safe_inverse_harmonic",
                "iterations": int(iteration + 1),
                "initial_unsafe_tetrahedra": initial_bad,
                "attenuated_boundary_nodes": int(
                    np.count_nonzero(attenuation < 1.0 - 1.0e-8)
                ),
                "minimum_boundary_attenuation": float(np.min(attenuation)),
                "minimum_jacobian_ratio": float(np.min(final_ratio)),
                "inverted_tetrahedra": 0,
            }

        # Reduce a smooth shell patch around each unsafe tet.  A hard zero at
        # just the incident node creates a boundary crease and merely moves the
        # inversion to its neighbour.  The nearest 96 shell nodes cover only a
        # few voxel rings (roughly 1--3 cm at the canonical resolution).
        centroids = np.mean(reference[tetrahedra[bad]], axis=1)
        neighbour_count = min(96, len(outer))
        distances, neighbours = boundary_tree.query(centroids, k=neighbour_count)
        distances = np.atleast_2d(np.asarray(distances, dtype=np.float64))
        neighbours = np.atleast_2d(np.asarray(neighbours, dtype=np.int64))
        patch_multiplier = np.ones(len(outer), dtype=np.float64)
        for local_distance, local_neighbour in zip(distances, neighbours):
            relative = local_distance - float(local_distance[0])
            sigma = max(2.0 * shell_spacing, float(local_distance[-1]) / 2.5)
            weight = np.exp(-0.5 * (relative / max(sigma, 1.0e-9)) ** 2)
            multiplier = 1.0 - 0.45 * weight
            np.minimum.at(patch_multiplier, local_neighbour, multiplier)
        changed = patch_multiplier < 1.0 - 1.0e-8
        if not np.any(changed):
            break
        attenuation[changed] *= patch_multiplier[changed]
        values = original_values * attenuation[:, None]

    remaining = int(
        np.count_nonzero((~np.isfinite(final_ratio)) | (final_ratio < _MIN_JACOBIAN_RATIO))
    )
    raise RuntimeError(
        "local boundary locking cannot produce a Jacobian-safe inverse harmonic "
        f"field ({remaining} unsafe tetrahedra remain, "
        f"{int(np.count_nonzero(attenuation < 1.0 - 1.0e-8))} boundary nodes attenuated)"
    )


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
    eligible = soft_volume_transport_mask_v811(asset)
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
        if str(tissue).strip().lower() not in _SOFT_VOLUME_TISSUES_V811:
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
    output[~eligible] = np.asarray(mapped_vertices, dtype=np.float64)[~eligible]
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
    asset: AnatomyRiggedAsset,
    *,
    canonical_dir: Path | str,
    debug_stage1_dir: Path | str | None = None,
    boundary_reference: Path | str | None = None,
    v35_semantic_prealign_shared_bind: bool = False,
    legacy_weighted_semantic_prealign: bool = False,
    preserve_protected_material: bool = False,
    rebind_source_rig: bool = True,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    if v35_semantic_prealign_shared_bind and legacy_weighted_semantic_prealign:
        raise ValueError("continuous and legacy semantic prealign are mutually exclusive")
    if asset.source_skin_vertices is None or asset.source_skin_faces is None:
        raise RuntimeError("source template lacks Skin_Glass; force source template rebake")
    root = Path(canonical_dir)
    target_vertices_full, target_faces_full = _load_obj(
        root / "smpl_canonical_tpose.obj"
    )
    target_weight_data = np.load(root / "smpl_canonical_weights.npz", allow_pickle=True)
    target_weights_full = np.asarray(target_weight_data["lbs_weights"], dtype=np.float64)
    target_joint_names = [str(value) for value in target_weight_data["joint_names"].tolist()]
    if target_joint_names != list(asset.joint_names):
        raise ValueError("Skin_Glass and SMPL-X joint semantic order does not match")
    subject_rest_joints = np.asarray(target_weight_data["rest_joints"], dtype=np.float32)
    subject_parents = np.asarray(target_weight_data["parents"], dtype=np.int32)
    subject_inverse_bind = np.asarray(target_weight_data["inverse_bind"], dtype=np.float32)
    if (
        subject_rest_joints.shape != np.asarray(asset.rest_joints).shape
        or subject_parents.shape != np.asarray(asset.parents).shape
        or subject_inverse_bind.shape != np.asarray(asset.inverse_bind).shape
    ):
        raise ValueError("subject SMPL-X skeleton shapes differ from the source template")
    joint_shift = np.linalg.norm(
        subject_rest_joints.astype(np.float64)
        - np.asarray(asset.rest_joints, dtype=np.float64),
        axis=1,
    )
    # Geometry, driver frames and the canonical body must all use this beta's
    # skeleton.  Keeping neutral joints after mapping vertices to subject beta
    # makes pose rotations use stale centers even though neutral LBS is exact.
    subject_asset = type(asset)(
        **{
            **asset.__dict__,
            "rest_joints": subject_rest_joints,
            "parents": subject_parents,
            "inverse_bind": subject_inverse_bind,
        }
    )
    source_vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    source_skin_vertices = np.asarray(asset.source_skin_vertices, dtype=np.float64)
    skin_faces = np.asarray(asset.source_skin_faces, dtype=np.int32)
    prealign_delta: np.ndarray | None = None
    prealign_blend = 0.0
    continuous_prealign_field: np.ndarray | None = None
    continuous_prealign_cage: dict[str, np.ndarray] | None = None
    continuous_prealign_stages: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
    if v35_semantic_prealign_shared_bind:
        _prealigned_vertices, prealigned_skin, _prealign_delta, prealign_report = (
            _semantic_rest_prealign(asset)
        )
        semantic_skin_guide = prealigned_skin
        semantic_skin_full_target = None
        prealign_report["bind_probes_share_prealign"] = True
        prealign_report["transport"] = "continuous_skin_boundary_harmonic"
        query = source_vertices
        skin_vertices = source_skin_vertices
    elif legacy_weighted_semantic_prealign:
        prealigned_vertices, prealigned_skin, prealign_delta, prealign_report = (
            _semantic_rest_prealign(asset, legacy_frames=True)
        )
        # This is the generic v62/e03 correspondence: the complete authored
        # source-weight field prealigns anatomy and Skin_Glass by the same
        # fixed fraction before the subject harmonic solve.  It is retained as
        # a hand-geometry reference; runtime coupling is rebuilt separately.
        prealign_blend = 0.25
        query = source_vertices + prealign_blend * (
            prealigned_vertices - source_vertices
        )
        skin_vertices = source_skin_vertices + prealign_blend * (
            prealigned_skin - source_skin_vertices
        )
        semantic_skin_guide = None
        semantic_skin_full_target = None
        prealign_report["backend"] = "source_rig_semantic_rest_lbs_v1"
        prealign_report["driver_skeleton"] = "source_neutral"
        prealign_report["blend"] = prealign_blend
        prealign_report["transport"] = "weighted_prealign_then_harmonic"
    else:
        query = source_vertices
        skin_vertices = source_skin_vertices
        semantic_skin_guide = None
        semantic_skin_full_target = None
        prealign_report = None
    target_vertices, target_faces, target_weights, body_report = _outer_body_component(
        target_vertices_full, target_faces_full, target_weights_full
    )
    if semantic_skin_guide is not None:
        semantic_skin_full_target, semantic_surface_report = (
            _semantic_skin_surface_correspondence(
                source_skin_vertices,
                semantic_skin_guide,
                np.asarray(asset.source_skin_lbs_weights, dtype=np.float64),
                target_vertices,
                target_weights,
            )
        )
        prealign_report["semantic_surface_correspondence"] = semantic_surface_report
    coarse_shell_vertices, coarse_shell_faces, coarse_shell_pitch = _coarse_closed_solver_shell(
        target_vertices, target_faces
    )
    shell_vertices, shell_faces, shell_pitch = _closed_solver_shell(
        target_vertices, target_faces
    )
    protected = rigid_hard_protection_mask_v811(asset)
    soft_volume_tissue_domain = soft_volume_material_mask_v811(asset)
    soft_volume_domain = soft_volume_transport_mask_v811(asset)
    # Skin_Glass is the only authored closed source domain that contains all
    # anatomy vertices.  Do not replace it with an expanded SMPL shell: that
    # loses hands/feet and invalidates harmonic sampling.  The old ARAP skin
    # map is intentionally not used here, because its output can drift away
    # from the target surface despite preserving source-triangle orientation.
    cached_reference: dict[str, np.ndarray] | None = None
    if boundary_reference is not None and prealign_report is not None:
        raise ValueError(
            "a legacy Stage-1 boundary reference cannot be reused after semantic rest prealign"
        )
    if boundary_reference is not None:
        reference_path = Path(boundary_reference).expanduser().resolve()
        with np.load(reference_path, allow_pickle=False) as reference_data:
            required = {
                "source_skin_vertices", "source_skin_faces", "nodes", "elements",
                "boundary_indices", "boundary_faces", "registered_boundary", "cage_signature",
            }
            missing = required - set(reference_data.files)
            if missing:
                raise ValueError(f"Stage-1 boundary reference lacks cage data: {sorted(missing)}")
            reference_skin = np.asarray(reference_data["source_skin_vertices"], dtype=np.float64)
            reference_faces = np.asarray(reference_data["source_skin_faces"], dtype=np.int32)
            skin_delta = np.max(np.abs(skin_vertices - reference_skin))
            if not np.array_equal(skin_faces, reference_faces) or skin_delta > 1.0e-5:
                raise ValueError(
                    "Stage-1 boundary reference Skin_Glass differs beyond stable-cage tolerance"
                )
            cached_reference = {key: np.asarray(reference_data[key]) for key in reference_data.files}
        cage = {
            "nodes": np.asarray(cached_reference["nodes"], dtype=np.float64),
            "elements": np.asarray(cached_reference["elements"], dtype=np.int32),
            "boundary": np.asarray(cached_reference["boundary_indices"], dtype=np.int32),
            "boundary_faces": np.asarray(cached_reference["boundary_faces"], dtype=np.int32),
            "signature": np.asarray(cached_reference["cage_signature"]),
            "voxel_pitch": np.asarray(cached_reference.get("voxel_pitch", [0.0])),
            "meshing_backend": np.asarray(cached_reference.get("meshing_backend", ["reference"])),
            "removed_degenerate_tetrahedra": np.asarray(
                cached_reference.get("removed_degenerate_tetrahedra", [0])
            ),
        }
    else:
        cage = _build_source_cage(
            skin_vertices,
            skin_faces,
            root / "source_skin_volume_cage_v18_subject_shell_full_domain.npz",
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
    if semantic_skin_full_target is not None:
        continuous_prealign_cage = {**cage, "nodes": nodes.copy()}
        boundary_source = _barycentric_surface_map(
            nodes[boundary], source_skin_vertices, skin_faces, source_skin_vertices
        )
        boundary_local_index = np.full(len(nodes), -1, dtype=np.int64)
        boundary_local_index[boundary] = np.arange(len(boundary), dtype=np.int64)
        boundary_faces_local = boundary_local_index[
            np.asarray(cage["boundary_faces"], dtype=np.int64)
        ]
        if np.any(boundary_faces_local < 0):
            raise RuntimeError("cage boundary faces reference non-boundary nodes")
        selection_attempts: list[dict[str, Any]] = []
        raw_boundary_displacement = np.zeros((len(boundary), 3), dtype=np.float64)
        smooth_boundary_displacement = raw_boundary_displacement.copy()
        continuous_report = {}
        for candidate_blend in _SEMANTIC_PREALIGN_CANDIDATES:
            semantic_skin_target = source_skin_vertices + float(candidate_blend) * (
                semantic_skin_full_target - source_skin_vertices
            )
            boundary_target = _barycentric_surface_map(
                nodes[boundary], source_skin_vertices, skin_faces, semantic_skin_target
            )
            candidate_raw = boundary_target - boundary_source
            candidate_smooth = _smooth_mesh_displacement(
                candidate_raw,
                boundary_faces_local,
                iterations=12,
            )
            try:
                candidate_field, candidate_report = (
                    _jacobian_safe_harmonic_boundary_field(
                        nodes,
                        elements,
                        boundary,
                        candidate_smooth,
                    )
                )
            except RuntimeError as exc:
                selection_attempts.append(
                    {
                        "blend": float(candidate_blend),
                        "accepted": False,
                        "reason": str(exc),
                    }
                )
                continue
            _probe_asset, probe_report = _rebind_source_rig_from_volume_field(
                asset,
                cage=continuous_prealign_cage,
                field=candidate_field,
                stage="stage1_semantic_prealign_probe_gate",
            )
            probe_safe = bool(
                float(probe_report["minimum_probe_stretch"])
                >= _SEMANTIC_PREALIGN_MIN_PROBE_STRETCH
                and float(probe_report["maximum_probe_stretch"])
                <= _SEMANTIC_PREALIGN_MAX_PROBE_STRETCH
            )
            selection_attempts.append(
                {
                    "blend": float(candidate_blend),
                    "accepted": probe_safe,
                    "minimum_probe_stretch": float(
                        probe_report["minimum_probe_stretch"]
                    ),
                    "maximum_probe_stretch": float(
                        probe_report["maximum_probe_stretch"]
                    ),
                }
            )
            if not probe_safe:
                continue
            prealign_blend = float(candidate_blend)
            continuous_prealign_field = candidate_field
            continuous_report = candidate_report
            raw_boundary_displacement = candidate_raw
            smooth_boundary_displacement = candidate_smooth
            break
        if continuous_prealign_field is None:
            raise RuntimeError(
                "no continuous semantic prealign candidate satisfies both "
                "Jacobian and bind-probe stretch gates: "
                f"{json.dumps(selection_attempts, sort_keys=True)}"
            )
        prealign_report["blend"] = prealign_blend
        continuous_report["selection_attempts"] = selection_attempts
        continuous_report["accepted_semantic_fraction"] = 1.0
        continuous_report["requested_semantic_blend"] = float(
            _SEMANTIC_PREALIGN_CANDIDATES[0]
        )
        continuous_report["effective_semantic_blend"] = prealign_blend
        continuous_report["probe_stretch_limits"] = {
            "minimum": _SEMANTIC_PREALIGN_MIN_PROBE_STRETCH,
            "maximum": _SEMANTIC_PREALIGN_MAX_PROBE_STRETCH,
        }
        continuous_report["raw_boundary_displacement_rms_m"] = float(
            np.sqrt(np.mean(np.sum(raw_boundary_displacement**2, axis=1)))
        )
        continuous_report["smoothed_boundary_displacement_rms_m"] = float(
            np.sqrt(np.mean(np.sum(smooth_boundary_displacement**2, axis=1)))
        )
        query_delta, _query_outside_count, query_outside = _sample_field(
            query,
            cage=continuous_prealign_cage,
            field=continuous_prealign_field,
        )
        skin_delta, _skin_outside_count, skin_outside = _sample_field(
            skin_vertices,
            cage=continuous_prealign_cage,
            field=continuous_prealign_field,
        )
        if np.any(query_outside) or np.any(skin_outside):
            raise RuntimeError("continuous semantic prealign lost source-cage queries")
        query = query + query_delta
        skin_vertices = skin_vertices + skin_delta
        # Composition must not inherit the first field's nearly-flat elements.
        # Remeshing the already-mapped closed Skin_Glass resets FEM quality;
        # anatomy topology, source weights and material coordinates are
        # untouched and will sample the next field from this new offline cage.
        cage = _build_source_cage(
            skin_vertices,
            skin_faces,
            root / "source_skin_volume_cage_v24_continuous_semantic_remesh.npz",
        )
        nodes = np.asarray(cage["nodes"], dtype=np.float64)
        elements = np.asarray(cage["elements"], dtype=np.int32)
        boundary = np.asarray(cage["boundary"], dtype=np.int64)
        _remesh_delta, _remesh_outside_count, remesh_outside = _sample_field(
            query,
            cage=cage,
            field=np.zeros_like(nodes),
        )
        if np.any(remesh_outside):
            raise RuntimeError(
                "continuous semantic remesh excludes mapped anatomy vertices: "
                f"{int(np.count_nonzero(remesh_outside))}"
            )
        continuous_prealign_stages.append(
            (continuous_prealign_cage, np.asarray(continuous_prealign_field))
        )
        prealign_report["continuous_remesh"] = {
            "backend": "mapped_skin_glass_closed_volume_remesh_v1",
            "cage_nodes": int(len(nodes)),
            "cage_tetrahedra": int(len(elements)),
            "outside_anatomy_vertices": 0,
        }
        prealign_report["continuous_volume_field"] = continuous_report
        prealign_report["anatomy_lbs_applied"] = False
        prealign_report["per_bone_probe_transform_applied"] = False
    reference_report: dict[str, Any] | None = None
    if boundary_reference is None:
        registered_boundary, coarse_surface_report = _topology_preserving_cage_registration(
            nodes,
            elements,
            boundary,
            np.asarray(cage["boundary_faces"], dtype=np.int32),
            coarse_shell_vertices,
            coarse_shell_faces,
        )
    else:
        reference_path = Path(boundary_reference).expanduser().resolve()
        with np.load(reference_path, allow_pickle=False) as cached:
            signature = str(np.asarray(cached["cage_signature"]).reshape(-1)[0])
            expected_signature = str(np.asarray(cage["signature"]).reshape(-1)[0])
            registered_boundary = np.asarray(cached["registered_boundary"], dtype=np.float64)
        if signature != expected_signature:
            raise ValueError("Stage-1 boundary reference cage signature does not match")
        if registered_boundary.shape != (len(boundary), 3):
            raise ValueError("Stage-1 boundary reference shape does not match cage boundary")
        reference_field = _harmonic_step(
            nodes, elements, boundary, registered_boundary - nodes[boundary]
        )
        reference_tetrahedra = nodes[elements]
        reference_ratio = np.linalg.det(
            (nodes + reference_field)[elements][:, 1:]
            - (nodes + reference_field)[elements][:, :1]
        ) / np.linalg.det(reference_tetrahedra[:, 1:] - reference_tetrahedra[:, :1])
        minimum_ratio = float(np.min(reference_ratio))
        if np.any(~np.isfinite(reference_ratio)) or minimum_ratio < _MIN_JACOBIAN_RATIO:
            raise RuntimeError("Stage-1 boundary reference is not Jacobian-safe on this cage")
        coarse_surface_report = {
            "backend": "accepted_same_cage_boundary_reference",
            "accepted_surface_iterations": 0,
            "minimum_surface_jacobian_ratio": minimum_ratio,
        }
        reference_report = {
            "path": str(reference_path),
            "cage_signature": signature,
            "minimum_jacobian_ratio": minimum_ratio,
        }
    phase_reports: list[dict[str, Any]] = []
    achieved_fraction = 0.0
    if boundary_reference is None:
        # Refine on the *same cage* towards a strictly in-subject target.  This
        # is a boundary-only homotopy, followed by one harmonic volume solve;
        # it never projects or clamps anatomy vertices with an SDF.
        import igl

        coarse_boundary = registered_boundary.copy()
        _squared, _face_index, fine_correspondence = (
            igl.point_mesh_squared_distance(
                registered_boundary, shell_vertices, shell_faces
            )
        )
        for fraction in _FINE_SHELL_HOMOTOPY:
            phase_target = coarse_boundary + float(fraction) * (
                np.asarray(fine_correspondence, dtype=np.float64)
                - coarse_boundary
            )
            try:
                registered_boundary, phase_report = (
                    _topology_preserving_cage_registration(
                        nodes,
                        elements,
                        boundary,
                        np.asarray(cage["boundary_faces"], dtype=np.int32),
                        shell_vertices,
                        shell_faces,
                        fixed_target=phase_target,
                        initial_boundary=registered_boundary,
                    )
                )
            except RuntimeError as exc:
                phase_reports.append(
                    {
                        "fine_shell_fraction": float(fraction),
                        "accepted": False,
                        "reason": str(exc),
                    }
                )
                break
            phase_report["fine_shell_fraction"] = float(fraction)
            phase_report["accepted"] = True
            phase_reports.append(phase_report)
            achieved_fraction = float(fraction)
    else:
        phase_reports.append({
            "accepted": True,
            "skipped": True,
            "reason": "accepted_same_cage_reference_is_the_complete_stage1_boundary",
        })
    surface_report = {
        "coarse_initialization": coarse_surface_report,
        "same_cage_boundary_reference": reference_report,
        "fine_inward_homotopy": phase_reports,
        "final_fine_shell_fraction": achieved_fraction,
    }
    boundary_delta = registered_boundary - nodes[boundary]
    # Preserve the actual Stage-1 boundary inspection even if the subsequent
    # volume solve correctly rejects an inversion.  This is intentionally
    # before any field sampling: it answers whether the skin correspondence,
    # rather than a later bone/soft-tissue step, is the failing condition.
    if debug_stage1_dir is not None:
        debug_local_index = np.full(len(nodes), -1, dtype=np.int64)
        debug_local_index[boundary] = np.arange(len(boundary), dtype=np.int64)
        debug_cage_faces = debug_local_index[
            np.asarray(cage["boundary_faces"], dtype=np.int64)
        ]
        if np.any(debug_cage_faces < 0):
            raise RuntimeError("boundary faces reference a non-boundary cage node")
        _export_stage1_debug(
            Path(debug_stage1_dir),
            body_vertices=target_vertices,
            body_faces=target_faces,
            shell_vertices=shell_vertices,
            shell_faces=shell_faces,
            source_skin=skin_vertices,
            source_skin_faces=skin_faces,
            stage1_skin=skin_vertices,
            stage1_skin_faces=target_faces,
            stage1_cage=registered_boundary,
            cage_faces=debug_cage_faces,
            stage1_anatomy=query,
            report={
                "state": "boundary_registered_before_volume_solve",
                "surface_correspondence": {
                    "backend": "closed_subject_shell_jacobian_safe",
                    "target_surface": "smpl_canonical_tpose.obj",
                },
                "cage_registration": surface_report,
            },
        )
    if prealign_report is not None:
        # The semantic-prealigned boundary was already accepted against the
        # complete source cage at the Jacobian threshold above.  Re-solving it
        # incrementally only allocates several full cage fields and is not a
        # different deformation; use the exact one-shot Dirichlet solution.
        field1 = _harmonic_step(nodes, elements, boundary, boundary_delta)
        prealign_tet = (nodes + field1)[elements]
        source_tet = nodes[elements]
        ratios = np.linalg.det(prealign_tet[:, 1:] - prealign_tet[:, :1]) / np.linalg.det(
            source_tet[:, 1:] - source_tet[:, :1]
        )
        minimum_ratio = float(np.min(ratios))
        if (
            np.any(~np.isfinite(ratios))
            or np.any(ratios <= 0.0)
            or minimum_ratio < _MIN_JACOBIAN_RATIO
        ):
            raise RuntimeError("semantic-prealigned Stage-1 field is not Jacobian-safe")
        harmonic_report = {
            "backend": "single_jacobian_verified_dirichlet",
            "incremental_steps": 1,
            "minimum_step_fraction": 1.0,
            "minimum_jacobian_ratio": minimum_ratio,
            "minimum_incremental_step_jacobian_ratio": minimum_ratio,
            "inverted_tetrahedra": 0,
        }
    else:
        field1, harmonic_report = _incremental_harmonic_field(
            nodes, elements, boundary, boundary_delta
        )
    deformation_report = {
        "backend": "stage1_outer_skin_dirichlet",
        **harmonic_report,
    }
    # This is Anatomy Transfer Stage 1: one outer-skin Dirichlet solve.  Every
    # anatomy vertex, including bones and cranium, receives this initial field.
    # Bone hardening and the later skin+bone multi-boundary solve are Stage 2;
    # neither belongs in this function or in a Stage-1 publication.
    deformed = np.asarray(nodes, dtype=np.float64) + field1
    local_index = np.full(len(nodes), -1, dtype=np.int64)
    local_index[boundary] = np.arange(len(boundary), dtype=np.int64)
    local_boundary_faces = local_index[
        np.asarray(cage["boundary_faces"], dtype=np.int64)
    ]
    if np.any(local_boundary_faces < 0):
        raise RuntimeError("boundary faces reference a non-boundary cage node")
    outer_boundary_report = {
        "backend": "stage1_subject_surface_dirichlet_harmonic_v2",
        "anatomy_vertices_projected": 0,
        "coarse_surface_rms_m": float(coarse_surface_report.get("final_surface_rms_m", 0.0)),
        "coarse_surface_max_m": float(coarse_surface_report.get("final_surface_max_m", 0.0)),
        "semantic_harmonic": deformation_report,
    }
    delta, outside_count, outside_mask = _sample_field(query, cage=cage, field=field1)
    _raise_outside_query_error(
        asset,
        query=query,
        outside_mask=outside_mask,
        protected=protected,
        cage=cage,
        context="registered source Skin_Glass domain",
    )
    mapped = query + delta
    if preserve_protected_material:
        # V8.11 keeps every non-transport material in its fitted rest frame.
        # The tissue-eligible set is vessels, nerves, organs, heart and
        # connective tissue, minus the rigid craniocerebral compound.  Skin
        # Glass and unclassified meshes are therefore never moved either.
        mapped[~soft_volume_domain] = source_vertices[~soft_volume_domain]
        if not np.array_equal(mapped[protected], source_vertices[protected]):
            raise RuntimeError(
                "V8.11 source-skin volume changed the rigid hard protection domain"
            )
    skin_delta, _skin_outside_count, skin_outside = _sample_field(
        skin_vertices, cage=cage, field=field1,
    )
    if np.any(skin_outside):
        raise RuntimeError(
            "registered source skin cannot be sampled from its own volume cage"
        )
    section_report = {
        "disabled": True,
        "reason": "radial_section_shrink_forbidden_for_thin_anatomy",
    }
    barrier_reports: dict[str, Any] = {}
    strain_reports: dict[str, Any] = {
        "disabled": True,
        "reason": "one_shot_multiboundary_harmonic_transport",
    }
    movable = soft_volume_domain if preserve_protected_material else ~protected
    soft_norm = np.linalg.norm(mapped[movable] - query[movable], axis=1)
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
        and float(np.max(np.linalg.norm(boundary_delta, axis=1))) > _MIN_REGISTRATION_PROGRESS_M
        and soft_max < _MIN_REGISTRATION_PROGRESS_M
    ):
        raise RuntimeError(
            "source skin registration moved the boundary but produced no measurable soft displacement"
        )
    source_volume = np.linalg.det(nodes[elements][:, 1:] - nodes[elements][:, :1])
    target_nodes = nodes + field1
    target_volume = np.linalg.det(target_nodes[elements][:, 1:] - target_nodes[elements][:, :1])
    inverted = int(np.count_nonzero(source_volume * target_volume <= 0.0))
    if inverted:
        raise RuntimeError(f"source skin harmonic field inverted {inverted} tetrahedra")
    minimum_jacobian_ratio = float(np.min(target_volume / source_volume))
    if minimum_jacobian_ratio < _MIN_JACOBIAN_RATIO:
        raise RuntimeError(
            f"source skin harmonic field is near-degenerate: min Jacobian ratio {minimum_jacobian_ratio:.6f}"
        )
    soft_gate = soft_volume_domain
    # Full-resolution signed distance against a 105k-face solver shell is a
    # diagnostic, not part of the field solve.  It can exceed practical host
    # memory for the 395k-vertex asset.  The publish path therefore defers
    # containment to run_audit_capture_pose, which evaluates the exact saved
    # runtime asset in rest and requested capture poses and is the release
    # authority.  The field's domain, finite values and Jacobian remain hard
    # requirements here.
    deferred_containment = True
    shell_outside = np.zeros(len(mapped), dtype=bool)
    gated_outside = np.zeros(len(mapped), dtype=bool)
    gated_fraction = 0.0
    gated_max = 0.0
    stage1_report = {
        "target_body_component": body_report,
        "solver_shell": {
            "vertices": int(len(shell_vertices)),
            "faces": int(len(shell_faces)),
            "voxel_pitch_m": float(shell_pitch),
            "coarse_voxel_pitch_m": float(coarse_shell_pitch),
        },
        "actual_stage1_field": {
            "all_material_vertices": int(len(mapped)),
            "solver_shell_outside_count": int(np.count_nonzero(shell_outside)),
            "solver_shell_outside_soft_count": int(np.count_nonzero(shell_outside & soft_gate)),
            "subject_outside_soft_count": int(np.count_nonzero(gated_outside)),
            "subject_outside_soft_fraction": gated_fraction,
            "subject_outside_soft_max_m": gated_max,
            "skin_sample_outside_source_cage": int(np.count_nonzero(skin_outside)),
            "containment_deferred_to_capture_audit": bool(deferred_containment),
        },
    }
    if debug_stage1_dir is not None:
        _export_stage1_debug(
            Path(debug_stage1_dir),
            body_vertices=target_vertices,
            body_faces=target_faces,
            shell_vertices=shell_vertices,
            shell_faces=shell_faces,
            source_skin=skin_vertices,
            source_skin_faces=skin_faces,
            stage1_skin=target_vertices,
            stage1_skin_faces=target_faces,
            stage1_cage=deformed[boundary],
            cage_faces=local_boundary_faces,
            stage1_anatomy=mapped,
            report={
                "surface_correspondence": {
                    "backend": "closed_subject_shell_jacobian_safe",
                    "target_surface": "smpl_canonical_tpose.obj",
                },
                "cage_registration": surface_report,
                "harmonic": deformation_report,
                **stage1_report,
            },
        )
    # The accepted same-cage reference deliberately precedes bounded regional
    # hand/oral Stage-1 fitting.  It still rejects a broken cage field, while
    # the final YA gate checks the declared 5% / 25 mm contract after that
    # local Stage-1 pass.
    intermediate_fraction_limit = 0.10 if boundary_reference is not None else 0.01
    intermediate_max_limit = 0.050 if boundary_reference is not None else 0.010
    if not deferred_containment and (gated_fraction > intermediate_fraction_limit or gated_max > intermediate_max_limit):
        raise RuntimeError(
            "Stage-1 soft containment gate failed against subject SMPL-X: "
            f"outside_fraction={gated_fraction:.4f}/{intermediate_fraction_limit:.4f}, "
            f"max={gated_max * 1000.0:.2f}/{intermediate_max_limit * 1000.0:.2f} mm"
        )

    metadata = dict(asset.metadata or {})
    metadata.update(
        {
            "source_skin_volume_registration": "stage1_subject_surface_dirichlet_harmonic_v3",
            "stage1_preserves_blender_source_binding": True,
            "stage1_fine_shell_homotopy": list(_FINE_SHELL_HOMOTOPY),
            "stage1_same_cage_boundary_reference": reference_report,
            "stage1_semantic_rest_prealign": prealign_report,
            "stage1_capture_audit_required": bool(deferred_containment),
            "stage1_subject_driver_skeleton": {
                "source": "smpl_canonical_weights.npz",
                "joint_shift_rms_m": float(np.sqrt(np.mean(joint_shift**2))),
                "joint_shift_max_m": float(np.max(joint_shift)),
            },
        }
    )
    result = type(subject_asset)(
        **{
            **subject_asset.__dict__,
            "vertices_rest": mapped.astype(np.float32),
            "harmonic_reference_vertices": mapped.astype(np.float32),
            "metadata": metadata,
        }
    )
    if not rebind_source_rig:
        source_rig_report = {
            "backend": "protected_rigid_source_bind_v811",
            "rebound": False,
            "reason": "bones and cranial compounds are excluded from soft transport",
        }
    elif continuous_prealign_field is not None and continuous_prealign_cage is not None:
        continuous_rebind_reports: list[dict[str, Any]] = []
        for stage_index, (stage_cage, stage_field) in enumerate(
            continuous_prealign_stages
        ):
            result, stage_rebind_report = _rebind_source_rig_from_volume_field(
                result,
                cage=stage_cage,
                field=stage_field,
                from_target_binding=stage_index > 0,
                stage=(
                    "stage1_continuous_semantic_prealign"
                    if stage_index == 0
                    else f"stage1_continuous_semantic_stage_{stage_index}"
                ),
            )
            continuous_rebind_reports.append(stage_rebind_report)
        result, outer_rebind_report = _rebind_source_rig_from_volume_field(
            result,
            cage=cage,
            field=field1,
            from_target_binding=True,
            stage="stage1_subject_outer_harmonic",
        )
        source_rig_report = {
            "backend": "composed_continuous_volume_bind_probes_v1",
            "continuous_semantic_prealign": continuous_rebind_reports,
            "subject_outer_harmonic": outer_rebind_report,
        }
    else:
        result, source_rig_report = _rebind_source_rig_from_volume_field(
            result,
            cage=cage,
            field=field1,
            semantic_prealign_delta=prealign_delta,
            semantic_prealign_blend=prealign_blend,
        )
    result = with_source_driver_coupling(result)
    report = {
        "schema_version": 1,
        "artifact_kind": "SourceSkinVolumeRegistrationV811",
        "backend": "stage1_subject_surface_dirichlet_harmonic_v3",
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
        "soft_volume_material_vertices": int(
            np.count_nonzero(soft_volume_tissue_domain)
        ),
        "soft_volume_transport_vertices": int(np.count_nonzero(soft_volume_domain)),
        "rigid_hard_protection_preserved": bool(preserve_protected_material),
        "soft_volume_tissues": sorted(_SOFT_VOLUME_TISSUES_V811),
        "anatomy_transport": (
            "soft_material_only_volume_field_v811"
            if preserve_protected_material
            else "all_material_volume_field_applied_before_bone_fit"
        ),
        "protected_material_preserved": bool(preserve_protected_material),
        "nonsoft_material_preserved": bool(preserve_protected_material),
        "semantic_rest_prealign": prealign_report,
        "source_rig_rebind": source_rig_report,
        "section_residual_regularizer": section_report,
        "soft_edge_strain_regularizer": strain_reports,
        "surface_barrier_regularizer": barrier_reports,
        "surface_correspondence": {
            "backend": "closed_subject_shell_jacobian_safe",
            "target_vertices": int(len(target_vertices)),
            "target_surface": "smpl_canonical_tpose.obj",
        },
        "outer_dirichlet_boundary": outer_boundary_report,
        **stage1_report,
        **surface_report,
    }
    report["content_digest"] = _volume_transport_digest_v811(
        result, soft_volume_domain, protected
    )
    return result, report
