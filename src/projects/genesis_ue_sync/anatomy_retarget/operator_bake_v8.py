"""Selective V8 operator assembly from the fitted product and V71 authority.

This module is an explicit migration boundary.  It may read the rejected V7
candidate to recover its beta response bases, but no V7 pose-time hinge,
patella oracle, tube-frame patch, shrink profile, or self-reported verdict is
copied into the resulting schema-v8 operator.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .anatomy_lbs import with_source_driver_coupling
from .articular_fit_v8 import calibrate_coupled_joint_roll_glide_v8
from .containment import load_body_surface
from .fk_policy_v8 import (
    SELECTIVE_AUTHORITY_FK_POLICY_V4,
    build_selective_fk_metadata_v4,
)
from .mechanism_v8 import (
    V71ParentLocalFKV8,
    build_ba9_head_selection_v8,
    topology_digest_v8,
)
from .head_compound_v8 import (
    fit_head_compound_v1,
    frozen_smplx_head_target_center_v1,
)
from .rigged_asset import AnatomyRiggedAsset
from .reference_fit_v8 import compose_unified_reference_template_v8
from .tube_frames_v8 import (
    bake_tube_coupling_v8,
    tube_coupling_pack_to_runtime_fields_v8,
)
from .tube_pose_corrective_v8 import (
    bake_tube_pose_corrective_v1,
    tube_pose_corrective_pack_to_runtime_fields_v1,
)
from .source_skin_volume import (
    apply_source_skin_volume_registration,
    soft_volume_transport_mask_v811,
)
from .v7_artifacts import SourceOperatorV7, rigged_asset_digest
from .v8_artifacts import SourceOperatorV8
from .vessel_route_v8 import bake_vessel_route_v8
from .version_v8 import (
    SOURCE_OPERATOR_ALGORITHM_VERSION,
    SOURCE_OPERATOR_CORRECTION_VERSION,
    SOURCE_OPERATOR_ORACLE_VERSION,
)


_OBSOLETE_RUNTIME_KEYS = frozenset(
    {
        "source_leg_hinge_solve_v1",
        "source_knee_hinge_splines_v7",
        "source_tibia_glide_splines_v7",
        "source_patella_splines_v7",
        "source_patella_v71_response_v8",
        "source_patella_response_v7",
        "source_ankle_roll_glide_v8",
        "patella_oracle",
        "patella_oracle_v7",
        "patella_oracle_digest",
        "source_local_fk_bones_v3",
        "source_leg_compound_roots_v1",
    }
)
_OFFLINE_METADATA_KEYS = frozenset({"mapping", "semantic_manifest"})

_V71_AUTHORITY_FIELDS = (
    "driver_indices",
    "driver_weights",
    "source_influence_offsets",
    "source_influence_group_indices",
    "source_influence_values",
    "source_group_names",
    "source_group_mesh_indices",
    "source_group_local_indices",
    "source_group_bone_indices",
    "source_bone_names",
    "source_bone_parents",
    "source_rest_global",
    "source_rest_local",
    "source_inverse_bind",
    "source_bone_head",
    "source_bone_tail",
    "source_bone_roll",
    "source_bone_use_connect",
    "source_bone_inherit_scale",
    "source_bone_smplx_a",
    "source_bone_smplx_b",
    "source_bone_blend",
    "source_bone_driver_types",
    "source_bone_frame_joints",
    "source_bone_corrective_driver",
    "source_bone_corrective_gain",
    "source_bone_corrective_axis",
)


def sanitize_v8_runtime_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Detach metadata and remove every known post-V71 pose-time patch."""

    def clean(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): clean(child)
                for key, child in item.items()
                if str(key).strip().lower()
                not in (_OBSOLETE_RUNTIME_KEYS | _OFFLINE_METADATA_KEYS)
            }
        if isinstance(item, list):
            return [clean(child) for child in item]
        if isinstance(item, tuple):
            return tuple(clean(child) for child in item)
        if isinstance(item, np.generic):
            return item.item()
        return copy.deepcopy(item)

    result = clean(value)
    result.update(
        {
            "source_fk_policy_v4": SELECTIVE_AUTHORITY_FK_POLICY_V4,
            "source_full_local_fk_v2": False,
            "source_joint_local_fk_v1": False,
            "source_connected_local_fk_v3": False,
            "disable_soft_follow": True,
            "v8_v71_parent_local_authority": True,
            "v8_obsolete_pose_patches_removed": True,
        }
    )
    return result


def _same_topology(first: AnatomyRiggedAsset, second: AnatomyRiggedAsset) -> bool:
    return bool(
        len(first.vertices_rest) == len(second.vertices_rest)
        and np.array_equal(first.faces, second.faces)
        and first.source_mesh_names == second.source_mesh_names
        and np.array_equal(first.source_vertex_ranges, second.source_vertex_ranges)
    )


def _has_anatomical_leg_guide_v810(asset: AnatomyRiggedAsset) -> bool:
    """Recognize the persisted bilateral hip-to-forefoot guide stations."""

    guide = getattr(asset, "source_driver_rest_joints", None)
    names = tuple(str(name) for name in (asset.joint_names or ()))
    if guide is None or len(names) != 55:
        return False
    values = np.asarray(guide, dtype=np.float64)
    if values.shape != (55, 3) or not np.all(np.isfinite(values)):
        return False
    required = (
        ("left_hip", "left_knee", "left_ankle", "left_foot"),
        ("right_hip", "right_knee", "right_ankle", "right_foot"),
    )
    try:
        for chain in required:
            ids = [names.index(name) for name in chain]
            segments = np.linalg.norm(np.diff(values[ids], axis=0), axis=1)
            if np.any(segments <= 1.0e-5):
                return False
    except ValueError:
        return False
    return True


def merge_v71_authority_v8(
    fitted_product: AnatomyRiggedAsset,
    v71_source: AnatomyRiggedAsset,
) -> AnatomyRiggedAsset:
    """Keep fitted product geometry while restoring V71 hierarchy and weights."""
    fitted_product.validate()
    v71_source.validate()
    if not _same_topology(fitted_product, v71_source):
        raise ValueError(
            "V8 direct migration requires identical product/V71 topology; "
            "a reviewed barycentric transfer map is required otherwise"
        )
    v71_indices = np.asarray(v71_source.driver_indices)
    v71_weights = np.asarray(v71_source.driver_weights)
    if v71_indices.shape != (len(v71_source.vertices_rest), 14):
        raise ValueError("V71 authority must contain the original 14 Armature slots")
    if v71_weights.shape != v71_indices.shape:
        raise ValueError("V71 driver indices/weights shape mismatch")
    replacements = {
        name: copy.deepcopy(getattr(v71_source, name))
        for name in _V71_AUTHORITY_FIELDS
    }
    runtime_metadata = sanitize_v8_runtime_metadata(v71_source.metadata or {})
    product_metadata = dict(fitted_product.metadata or {})
    for key in (
        "hidden_mesh_names_v1",
        "hidden_mesh_names_v2",
        "hidden_face_ids_v2",
        "oral_visibility_policy_v2",
        "v8_unified_reference_fit",
        "v8_reference_beta_origin",
        "v8_reference_field_authority",
        "v8_nonshrunk_bone_authority",
        "v8_foot_compound_authority",
        # A continuous product may carry a previously baked soft-only volume
        # field.  Keep its provenance while replacing all source-rig authority
        # from V71 below; the actual reuse path independently checks the
        # canonical rest skeleton and soft vertex domain.
        "source_skin_volume_registration",
        "stage1_preserves_blender_source_binding",
        "stage1_subject_driver_skeleton",
        "stage1_capture_audit_required",
        "shape_hash",
        # The guide stations, rather than arbitrary Blender child offsets,
        # anchor the hip/knee/ankle/forefoot chain at runtime.
        "source_anatomical_guide_fk_v810",
    ):
        if key in product_metadata:
            runtime_metadata[key] = copy.deepcopy(product_metadata[key])
    replacements.update(
        {
            "runtime_driver_indices_compressed": None,
            "runtime_driver_weights_compressed": None,
            "source_driver_coupling": None,
            "soft_follow_driver_indices": None,
            "soft_follow_driver_weights": None,
            "soft_follow_stations": None,
            "soft_follow_strength": None,
            "soft_component_ids": None,
            "source_mesh_follow_modes": None,
            "pose_cache_vertices": None,
            "pose_cache_hash": "",
            "metadata": runtime_metadata,
        }
    )
    # The target bind and anatomical driver pivots are beta-fit product data.
    # Authored source local matrices above remain immutable V71 provenance.
    merged = replace(fitted_product, **replacements)
    # The many-segment cervical chain remains parent-local, while the three
    # independently driven anchors recover their SMPL-X global virtual
    # joints.  This prevents accumulated neck translations from separating
    # C1/C2, skull, jaw, and teeth without weakening FK elsewhere.
    merged_metadata = build_selective_fk_metadata_v4(
        merged,
        merged.metadata,
        extra_direct_bone_names=("Spine_C7", "Head_Bone", "Jaw_Bone_base"),
    )
    if not _has_anatomical_leg_guide_v810(merged):
        raise ValueError(
            "V8.11 selective migration requires complete bilateral "
            "anatomical hip/knee/ankle/foot guide stations"
        )
    merged_metadata["source_anatomical_guide_fk_v810"] = True
    merged = replace(merged, metadata=merged_metadata)
    merged = with_source_driver_coupling(merged)
    merged.validate()
    V71ParentLocalFKV8(
        bone_names=tuple(merged.source_bone_names or ()),
        parents=merged.source_bone_parents,
        rest_local=merged.source_rest_local,
        bone_head=merged.source_bone_head,
        bone_tail=merged.source_bone_tail,
        bone_roll=merged.source_bone_roll,
        bone_use_connect=merged.source_bone_use_connect,
        bone_inherit_scale=merged.source_bone_inherit_scale,
        driver_types=tuple(merged.source_bone_driver_types or ()),
        driver_coupling=merged.source_driver_coupling,
        driver_indices=merged.driver_indices,
        driver_weights=merged.driver_weights,
        metadata=merged.metadata or {},
    )
    return merged


def _split_spatial_domain(
    vertices: np.ndarray,
    ids: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically interleave a surface domain into disjoint probes."""
    selected = np.unique(np.asarray(ids, dtype=np.int64).reshape(-1))
    if len(selected) < 8:
        raise ValueError("a V8 fit/validation domain needs at least eight vertices")
    xyz = np.asarray(vertices, dtype=np.float64)[selected]
    center = np.mean(xyz, axis=0)
    _u, _singular, axes = np.linalg.svd(xyz - center, full_matrices=False)
    projected = (xyz - center) @ axes.T
    order = np.lexsort((projected[:, 2], projected[:, 1], projected[:, 0]))
    fit = np.sort(selected[order[0::2]])
    validation = np.sort(selected[order[1::2]])
    if not len(fit) or not len(validation) or np.intersect1d(fit, validation).size:
        raise AssertionError("failed to construct disjoint V8 material probes")
    return fit.astype(np.int32), validation.astype(np.int32)


def _mesh_ids(asset: AnatomyRiggedAsset, name: str) -> np.ndarray:
    try:
        index = list(asset.source_mesh_names).index(name)
    except ValueError as exc:
        raise ValueError(f"required source mesh {name!r} is missing") from exc
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[index]
    return np.arange(int(start), int(stop), dtype=np.int32)


def _near_interface(
    vertices: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    fraction: float = 0.30,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    first_ids = np.asarray(first, dtype=np.int64).reshape(-1)
    second_ids = np.asarray(second, dtype=np.int64).reshape(-1)
    distance, _nearest = cKDTree(
        np.asarray(vertices, dtype=np.float64)[second_ids]
    ).query(np.asarray(vertices, dtype=np.float64)[first_ids], k=1)
    count = max(8, int(np.ceil(float(fraction) * len(first_ids))))
    return np.sort(first_ids[np.argsort(distance)[:count]]).astype(np.int32)


def build_frozen_domains_v8(
    asset: AnatomyRiggedAsset,
    legacy_joint_domains: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Freeze joint, head, and rib domains once against product topology."""
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    result: dict[str, np.ndarray] = {}
    for name, ids in legacy_joint_domains.items():
        selected = np.asarray(ids, dtype=np.int32).reshape(-1)
        if len(selected) >= 8:
            fit, validation = _split_spatial_domain(vertices, selected)
            result[f"{name}.fit"] = fit
            result[f"{name}.validation"] = validation
        else:
            result[str(name)] = selected

    for side in ("L", "R"):
        humerus = _mesh_ids(asset, f"Humerus_{side}")
        ulna = _mesh_ids(asset, f"Ulna_{side}")
        radius = _mesh_ids(asset, f"Radius_{side}")
        forearm = np.concatenate((ulna, radius))
        interface = {
            "humerus": _near_interface(vertices, humerus, forearm),
            "ulna": _near_interface(vertices, ulna, humerus),
            "radius": _near_interface(vertices, radius, humerus),
        }
        label = "left" if side == "L" else "right"
        for part, ids in interface.items():
            fit, validation = _split_spatial_domain(vertices, ids)
            result[f"elbow/{label}/{part}.fit"] = fit
            result[f"elbow/{label}/{part}.validation"] = validation

        tibia = _mesh_ids(asset, f"Tibia_{side}")
        fibula = _mesh_ids(asset, f"Fibula_{side}")
        talus = _mesh_ids(asset, f"Talus_{side}")
        ankle_interface = {
            "tibia": _near_interface(
                vertices, tibia, talus, fraction=0.15
            ),
            "fibula": _near_interface(
                vertices, fibula, talus, fraction=0.20
            ),
            "talus": _near_interface(
                vertices,
                talus,
                np.concatenate((tibia, fibula)),
                fraction=0.30,
            ),
        }
        for part, ids in ankle_interface.items():
            fit, validation = _split_spatial_domain(vertices, ids)
            result[f"ankle/{label}/{part}.fit"] = fit
            result[f"ankle/{label}/{part}.validation"] = validation

    for number in range(1, 13):
        for side in ("L", "R"):
            name = f"Rib_{number}{side}"
            result[f"rib/{number}/{side.lower()}"] = _mesh_ids(asset, name)

    topology = topology_digest_v8(asset.vertices_rest, asset.faces)
    head = build_ba9_head_selection_v8(asset, topology_digest=topology)
    for name, mask in (
        ("head/cranial", head.cranial_mask),
        ("head/jaw", head.jaw_mask),
        ("head/rigid_attachment", head.rigid_attachment_mask),
        ("head/hyoid_rest", head.hyoid_rest_mask),
    ):
        ids = np.flatnonzero(mask).astype(np.int32)
        if len(ids):
            result[name] = ids
    return result


def _soft_volume_beta_basis_v811(
    *,
    template: AnatomyRiggedAsset,
    source_basis: Any,
    source_skin_volume_dir: Any | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Freeze a soft-only beta displacement basis for matrix-only L1 use.

    A canonical volume directory may carry explicitly sampled unit-beta
    displacements.  When it does not, the already offline-baked V7 beta basis
    is the only admissible fallback: it is restricted to the V8.11 soft domain
    and never causes a volume solve during materialization.  Both paths make
    the selected bytes part of the operator identity.
    """

    base = np.asarray(source_basis, dtype=np.float32)
    expected = (10, len(template.vertices_rest), 3)
    if base.shape != expected or not np.all(np.isfinite(base)):
        raise ValueError(
            "source beta_vertex_basis must be finite [10,vertex_count,3]"
        )
    soft = soft_volume_transport_mask_v811(template)
    result = np.zeros_like(base)
    result[:, soft] = base[:, soft]
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "SourceSkinVolumeBetaBasisV1",
        "reference_beta_count": 1,
        "unit_beta_probe_count": 10,
        "soft_vertex_count": int(np.count_nonzero(soft)),
        "runtime_operation": "tensordot_only",
        "runtime_volume_query": False,
        "runtime_kdtree": False,
        "runtime_graph_solve": False,
        "source": "v7_baked_beta_basis_restricted_to_v811_soft_domain",
        "available": False,
        "passed": False,
        "reason": "explicit reference-plus-unit-beta volume probes are absent",
    }
    if source_skin_volume_dir is not None:
        path = Path(source_skin_volume_dir) / "source_skin_volume_beta_basis_v1.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                ids = np.asarray(data["vertex_ids"], dtype=np.int64).reshape(-1)
                basis_key = next(
                    (
                        name
                        for name in (
                            "displacement_basis_m",
                            "beta_displacement_basis_m",
                            "beta_vertex_basis",
                        )
                        if name in data.files
                    ),
                    None,
                )
                if basis_key is None:
                    raise ValueError(
                        f"{path} lacks displacement_basis_m for the V8.11 beta basis"
                    )
                sampled = np.asarray(data[basis_key], dtype=np.float32)
            if sampled.shape == (len(ids), 3, 10):
                sampled = np.moveaxis(sampled, -1, 0)
            if sampled.shape != (10, len(ids), 3):
                raise ValueError(
                    "source_skin_volume_beta_basis_v1 displacement basis must be "
                    "[10,V,3] or [V,3,10]"
                )
            if (
                not len(ids)
                or np.any(ids < 0)
                or np.any(ids >= len(soft))
                or len(np.unique(ids)) != len(ids)
                or not np.all(soft[ids])
                or not np.array_equal(
                    np.sort(ids), np.flatnonzero(soft).astype(np.int64)
                )
                or not np.all(np.isfinite(sampled))
            ):
                raise ValueError(
                    "source_skin_volume_beta_basis_v1 must cover every V8.11 "
                    "soft-tissue vertex id exactly once"
                )
            result[:, ids] = sampled
            report.update(
                {
                    "source": "source_skin_volume_beta_basis_v1.npz",
                    "sampled_vertex_count": int(len(ids)),
                    "available": True,
                    "passed": True,
                    "reason": None,
                    "sampled_vertex_ids_sha256": hashlib.sha256(
                        np.ascontiguousarray(ids.astype("<i4")).tobytes()
                    ).hexdigest(),
                }
            )
    digest = hashlib.sha256(b"source-skin-volume-beta-basis-v1\0")
    digest.update(np.ascontiguousarray(result).tobytes())
    report["content_digest"] = digest.hexdigest()
    return result, report


def _prebaked_soft_volume_reference_v811(
    template: AnatomyRiggedAsset,
    *,
    source_skin_volume_dir: Any,
) -> dict[str, Any] | None:
    """Authenticate a prior continuous soft-volume field for V8.11 reuse.

    Some known-good reference products already contain the offline harmonic
    transport that kept vessels and nerves within the body.  Re-running that
    field against their *source* Skin_Glass is incorrect: their soft vertices
    are now in the target frame while Skin_Glass remains source-frame input.
    Reuse is deliberately narrow: only an exact canonical skeleton and an
    unchanged soft transport domain are accepted; hard tissue is never reused
    as a geometry or bind authority.
    """

    metadata = dict(template.metadata or {})
    backend = str(metadata.get("source_skin_volume_registration", ""))
    if backend != "stage1_subject_surface_dirichlet_harmonic_v3":
        return None
    reference = getattr(template, "harmonic_reference_vertices", None)
    if reference is None:
        return None
    vertices = np.asarray(template.vertices_rest, dtype=np.float32)
    reference_vertices = np.asarray(reference, dtype=np.float32)
    if (
        reference_vertices.shape != vertices.shape
        or not np.all(np.isfinite(reference_vertices))
        or not np.all(np.isfinite(vertices))
    ):
        return None
    root = Path(source_skin_volume_dir)
    weights_path = root / "smpl_canonical_weights.npz"
    if not weights_path.is_file():
        raise ValueError(
            "V8.11 soft-volume reference reuse requires "
            "smpl_canonical_weights.npz"
        )
    try:
        with np.load(weights_path, allow_pickle=False) as data:
            canonical_joints = np.asarray(data["rest_joints"], dtype=np.float32)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(
            "V8.11 soft-volume reference reuse requires canonical rest_joints"
        ) from exc
    subject_joints = np.asarray(template.rest_joints, dtype=np.float32)
    if canonical_joints.shape != subject_joints.shape or not np.array_equal(
        canonical_joints, subject_joints
    ):
        return None
    soft = soft_volume_transport_mask_v811(template)
    protected = ~soft
    if not np.any(soft):
        return None
    digest = hashlib.sha256(b"prebaked-soft-volume-reference-v811\0")
    for values in (
        np.asarray(soft, dtype=np.uint8),
        np.ascontiguousarray(reference_vertices[soft]),
        np.ascontiguousarray(canonical_joints),
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    return {
        "schema_version": 1,
        "artifact_kind": "SourceSkinVolumeRegistrationV811",
        "backend": "prebaked_continuous_soft_volume_reference_v811",
        "available": True,
        "passed": False,
        "reason": "exact rest and capture containment audit is required after subject materialization",
        "capture_audit_required": True,
        "canonical_rest_joint_match": True,
        "source_harmonic_backend": backend,
        "final_soft_vertices_differ_from_stage1_reference": bool(
            not np.array_equal(reference_vertices[soft], vertices[soft])
        ),
        "soft_volume_transport_vertices": int(np.count_nonzero(soft)),
        "protected_material_vertices": int(np.count_nonzero(protected)),
        "topology_preserved": True,
        "source_weights_preserved": True,
        "protected_rigid_material": True,
        "runtime_volume_query": False,
        "content_digest": digest.hexdigest(),
    }


def _constraint_surface_v811(
    vessel_skin_vertices: Any | None,
    vessel_skin_faces: Any | None,
    *,
    source_skin_volume_dir: Any | None,
) -> tuple[Any | None, Any | None]:
    """Choose the one frozen shell used by head and tube constraints."""

    if vessel_skin_vertices is not None:
        return vessel_skin_vertices, vessel_skin_faces
    if source_skin_volume_dir is None:
        return None, None
    canonical_surface = Path(source_skin_volume_dir) / "smpl_canonical_tpose.obj"
    return load_body_surface(canonical_surface)


def _canonical_head_target_center_v811(
    template: AnatomyRiggedAsset,
    *,
    source_skin_volume_dir: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the exact canonical SMPL-X head envelope used for V8.11 fitting."""

    root = Path(source_skin_volume_dir)
    surface, _faces = load_body_surface(root / "smpl_canonical_tpose.obj")
    weights_path = root / "smpl_canonical_weights.npz"
    try:
        with np.load(weights_path, allow_pickle=False) as data:
            weights = np.asarray(data["lbs_weights"], dtype=np.float32)
            canonical_names = np.asarray(data["joint_names"]).reshape(-1)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(
            "V8.11 canonical head envelope requires smpl_canonical_weights.npz "
            "with lbs_weights and joint_names"
        ) from exc
    template_names = tuple(str(name) for name in template.joint_names)
    resolved_names = tuple(
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in canonical_names.tolist()
    )
    if template_names != resolved_names:
        raise ValueError(
            "V8.11 canonical head envelope joint order does not match the template"
        )
    center, report = frozen_smplx_head_target_center_v1(
        surface,
        lbs_weights=weights,
        joint_names=canonical_names,
    )
    return center, {
        **report,
        "canonical_joint_order_matched": True,
    }


def build_selective_source_operator_v8(
    *,
    v7_operator: SourceOperatorV7,
    v71_source: AnatomyRiggedAsset,
    reference_manifest: Mapping[str, Any],
    runtime_coefficients: Mapping[str, np.ndarray],
    fitted_product: AnatomyRiggedAsset | None = None,
    continuous_product: AnatomyRiggedAsset | None = None,
    foot_product: AnatomyRiggedAsset | None = None,
    reference_betas: Any | None = None,
    vessel_skin_vertices: Any | None = None,
    vessel_skin_faces: Any | None = None,
    source_skin_volume_dir: Any | None = None,
    tube_corrective_pose_axis_angle_samples: Any | None = None,
    tube_corrective_local_displacement_samples_m: Any | None = None,
    tube_corrective_vertex_ids: Any | None = None,
    tube_corrective_driver_joint_ids: Any | None = None,
    algorithm_version: str = SOURCE_OPERATOR_ALGORITHM_VERSION,
    oracle_version: str = SOURCE_OPERATOR_ORACLE_VERSION,
    correction_version: str = SOURCE_OPERATOR_CORRECTION_VERSION,
) -> SourceOperatorV8:
    """Assemble L0 without copying V7 pose-time mechanisms or verdicts."""
    unified_report: dict[str, Any] | None = None
    unified_coefficients: dict[str, np.ndarray] = {}
    vessel_route_report: dict[str, Any] = {
        "available": False,
        "passed": False,
        "reason": "canonical subject skin was not supplied",
    }
    volume_registration_report: dict[str, Any] = {
        "available": False,
        "passed": False,
        "reason": "canonical source-skin volume directory was not supplied",
    }
    reused_prebaked_soft_volume = False
    head_compound_report: dict[str, Any] = {
        "available": False,
        "passed": False,
        "reason": "canonical SMPL-X head surface was not supplied",
    }
    corrective_values = (
        tube_corrective_pose_axis_angle_samples,
        tube_corrective_local_displacement_samples_m,
        tube_corrective_vertex_ids,
        tube_corrective_driver_joint_ids,
    )
    if any(value is not None for value in corrective_values) and not all(
        value is not None for value in corrective_values
    ):
        raise ValueError(
            "tube pose corrective requires pose samples, local displacements, "
            "vertex_ids, and driver_joint_ids together"
        )
    tube_pose_corrective_report: dict[str, Any] = {
        "available": False,
        "passed": False,
        "reason": "offline tube corrective poses were not supplied",
    }
    if (vessel_skin_vertices is None) != (vessel_skin_faces is None):
        raise ValueError(
            "vessel_skin_vertices and vessel_skin_faces must be supplied together"
        )
    if any(
        value is not None
        for value in (
            fitted_product,
            continuous_product,
            foot_product,
            reference_betas,
        )
    ):
        if (
            fitted_product is None
            or
            continuous_product is None
            or foot_product is None
            or reference_betas is None
        ):
            raise ValueError(
                "fitted_product, continuous_product, foot_product, and "
                "reference_betas must be supplied together"
            )
        composed, unified_coefficients, unified_report = (
            compose_unified_reference_template_v8(
                fitted_product=fitted_product,
                continuous_product=continuous_product,
                foot_product=foot_product,
                reference_betas=reference_betas,
            )
        )
        template = merge_v71_authority_v8(composed, v71_source)
    else:
        template = merge_v71_authority_v8(v7_operator.template_asset, v71_source)
    if source_skin_volume_dir is not None:
        original_faces = np.asarray(template.faces).copy()
        original_indices = np.asarray(template.driver_indices).copy()
        original_weights = np.asarray(template.driver_weights).copy()
        volume_registration_report = _prebaked_soft_volume_reference_v811(
            template,
            source_skin_volume_dir=source_skin_volume_dir,
        )
        reused_prebaked_soft_volume = volume_registration_report is not None
        if volume_registration_report is None:
            template, volume_registration_report = apply_source_skin_volume_registration(
                template,
                canonical_dir=source_skin_volume_dir,
                preserve_protected_material=True,
                rebind_source_rig=False,
            )
            if (
                not np.array_equal(template.faces, original_faces)
                or not np.array_equal(template.driver_indices, original_indices)
                or not np.array_equal(template.driver_weights, original_weights)
            ):
                raise RuntimeError(
                    "V8.11 soft volume registration changed topology or source weights"
                )
            volume_registration_report = {
                **volume_registration_report,
                "available": True,
                # The harmonic map has completed here, but it has not yet been
                # checked against the exact saved subject in rest and capture
                # poses.  Do not turn that missing geometry evidence into a green
                # publish contract merely because topology and weights survived.
                "passed": False,
                "capture_audit_required": True,
                "reason": (
                    "exact rest and capture containment audit is required after "
                    "subject materialization"
                ),
                "topology_preserved": True,
                "source_weights_preserved": True,
                "protected_rigid_material": True,
            }
    # The required canonical volume directory already owns the frozen SMPL-X
    # shell.  Reuse it for both head fitting and the vessel/nerve route when a
    # separate audit surface was not supplied; otherwise a nominal V8.11 bake
    # would silently skip the mandatory route stage.
    vessel_skin_vertices, vessel_skin_faces = _constraint_surface_v811(
        vessel_skin_vertices,
        vessel_skin_faces,
        source_skin_volume_dir=source_skin_volume_dir,
    )
    head_surface_vertices = vessel_skin_vertices
    head_surface_faces = vessel_skin_faces
    if reused_prebaked_soft_volume:
        # The historical continuous field is being used to recover soft-tube
        # containment.  Do not silently refit its already reviewed head while
        # investigating legs/feet; head fit remains an explicit release gate.
        head_compound_report = {
            "available": False,
            "passed": False,
            "reason": "head compound fit deferred while validating reused soft-volume reference",
            "deferred": True,
        }
    elif head_surface_vertices is not None and head_surface_faces is not None:
        if source_skin_volume_dir is None:
            raise ValueError(
                "V8.11 head compound fitting requires the canonical source-skin "
                "volume directory"
            )
        head_target_center, head_target_report = _canonical_head_target_center_v811(
            template,
            source_skin_volume_dir=source_skin_volume_dir,
        )
        template, head_compound_report = fit_head_compound_v1(
            template,
            surface_vertices=np.asarray(head_surface_vertices, dtype=np.float64),
            surface_faces=np.asarray(head_surface_faces, dtype=np.int32),
            target_center_m=head_target_center,
        )
        head_compound_report = {
            **head_compound_report,
            "target_center_evidence": head_target_report,
            "available": True,
            "passed": True,
        }
    # Route the already volume-wrapped tubes.  Reconstructing their rest
    # vertices from immutable Blender bind coordinates here would overwrite
    # the V8.11 source_skin_volume result before the route has constrained it.
    # The original V71 14-slot indices and weights remain immutable either way.
    if vessel_skin_vertices is not None:
        template, vessel_route_report = bake_vessel_route_v8(
            template,
            skin_vertices=np.asarray(vessel_skin_vertices, dtype=np.float64),
            skin_faces=np.asarray(vessel_skin_faces, dtype=np.int32),
            reconstruct_source_weighted=False,
        )
    domains = build_frozen_domains_v8(template, v7_operator.fixed_material_domains)
    coupled_joint_domain_keys = {
        f"ankle/{side}/{part}.{partition}"
        for side in ("left", "right")
        for part in ("tibia", "fibula", "talus")
        for partition in ("fit", "validation")
    }
    coupled_joint_domain_keys.update(
        {
            f"{side}/{part}.{partition}"
            for side in ("left", "right")
            for part in (
                "femoral_condyle_medial",
                "femoral_condyle_lateral",
                "tibial_plateau_medial",
                "tibial_plateau_lateral",
            )
            for partition in ("fit", "validation")
        }
    )
    if coupled_joint_domain_keys.issubset(domains):
        template, coupled_joint_report = calibrate_coupled_joint_roll_glide_v8(
            template,
            domains=domains,
        )
    else:
        coupled_joint_report = {
            "available": False,
            "reason": "operator lacks bilateral frozen knee/ankle domains",
        }
    # Volume/head/route changes may replace fitted bind matrices.  Persist one
    # coupling made from the final L0 bind, never a pre-route intermediate.
    template = with_source_driver_coupling(template)
    final_runtime_coefficients = {
        str(name): np.asarray(value).copy()
        for name, value in runtime_coefficients.items()
        if not str(name).startswith(
            ("tube_coupling_v8.", "tube_pose_corrective_v1.")
        )
    }
    has_tube_material = any(
        str(tissue).strip().lower() in {"vessel", "nerve"}
        for tissue in (template.source_tissues or ())
    )
    if has_tube_material:
        tube_pack, tube_coupling_report = bake_tube_coupling_v8(template)
        final_runtime_coefficients.update(
            tube_coupling_pack_to_runtime_fields_v8(tube_pack)
        )
        if all(value is not None for value in corrective_values):
            corrective_ids = np.asarray(
                tube_corrective_vertex_ids, dtype=np.int64
            ).reshape(-1)
            tube_ids = np.asarray(tube_pack.vertex_ids, dtype=np.int64)
            if not np.all(np.isin(corrective_ids, tube_ids)):
                raise ValueError(
                    "tube pose corrective vertex_ids must belong to the final "
                    "vessel/nerve tube domain"
                )
            corrective_pack, tube_pose_corrective_report = (
                bake_tube_pose_corrective_v1(
                    corrective_ids,
                    tube_corrective_pose_axis_angle_samples,
                    tube_corrective_local_displacement_samples_m,
                    tube_corrective_driver_joint_ids,
                )
            )
            final_runtime_coefficients.update(
                tube_pose_corrective_pack_to_runtime_fields_v1(corrective_pack)
            )
            tube_pose_corrective_report = {
                **tube_pose_corrective_report,
                "tube_domain_digest": tube_pack.domain_digest,
                "tube_weight_digest": tube_pack.weight_digest,
                "vertex_ids_subset_of_tube_domain": True,
            }
    else:
        tube_coupling_report = {
            "available": False,
            "passed": False,
            "reason": "final L0 template contains no vessel or nerve material",
        }
        if all(value is not None for value in corrective_values):
            raise ValueError(
                "tube pose corrective samples were supplied but final L0 has no "
                "vessel or nerve material"
            )
    soft_beta_basis, soft_beta_basis_report = _soft_volume_beta_basis_v811(
        template=template,
        source_basis=v7_operator.beta_vertex_basis,
        source_skin_volume_dir=source_skin_volume_dir,
    )
    mechanism = {
        "v71.parents": np.asarray(template.source_bone_parents, dtype=np.int32),
        "v71.rest_local": np.asarray(template.source_rest_local, dtype=np.float32),
        "v71.bone_head": np.asarray(template.source_bone_head, dtype=np.float32),
        "v71.bone_tail": np.asarray(template.source_bone_tail, dtype=np.float32),
        "v71.bone_roll": np.asarray(template.source_bone_roll, dtype=np.float32),
        "v71.bone_use_connect": np.asarray(
            template.source_bone_use_connect, dtype=np.uint8
        ),
        "v71.bone_inherit_scale": np.asarray(
            template.source_bone_inherit_scale, dtype=np.uint8
        ),
        **unified_coefficients,
    }
    operator = SourceOperatorV8(
        template_asset=template,
        beta_vertex_basis=soft_beta_basis,
        beta_rest_joint_basis=np.asarray(
            v7_operator.beta_rest_joint_basis, dtype=np.float32
        ),
        beta_bind_twist_basis=np.asarray(
            v7_operator.beta_bind_twist_basis, dtype=np.float32
        ),
        internal_handle_basis=np.asarray(
            v7_operator.internal_handle_basis, dtype=np.float32
        ),
        fixed_material_domains=domains,
        mechanism_coefficients=mechanism,
        contact_envelopes={
            name: np.asarray(value).copy()
            for name, value in v7_operator.contact_envelopes.items()
        },
        runtime_coefficients=final_runtime_coefficients,
        reference_manifest=reference_manifest,
        algorithm_version=algorithm_version,
        oracle_version=oracle_version,
        correction_version=correction_version,
        provenance={
            "source_asset_digest": rigged_asset_digest(template),
            "migration_basis": "V7 beta bases only",
            "v7_template_digest": rigged_asset_digest(v7_operator.template_asset),
            "v71_source_digest": rigged_asset_digest(v71_source),
            "original_armature_slots": 14,
            "source_fk_policy_v4": SELECTIVE_AUTHORITY_FK_POLICY_V4,
            "source_full_local_fk_v2": False,
            "unified_reference_fit": unified_report is not None,
            "source_skin_volume_beta_basis_v1": soft_beta_basis_report,
        },
        correction_report={
            "passed": False,
            "publishable": False,
            "head_baseline": "ba9 product topology/compound semantics",
            "kinematic_baseline": "V71 parent-local bind and original weights",
            "obsolete_pose_path_count_removed": len(_OBSOLETE_RUNTIME_KEYS),
            "obsolete_pose_paths_absent": True,
            "tongue": "missing; release blocker",
            "unified_reference_fit": unified_report,
            "coupled_knee_ankle_roll_glide": coupled_joint_report,
            "vessel_route_v8": vessel_route_report,
            "source_skin_volume_v811": volume_registration_report,
            "source_skin_volume_beta_basis_v1": soft_beta_basis_report,
            "head_compound_fit_v1": head_compound_report,
            "tube_coupling_final_rest_v810": tube_coupling_report,
            "tube_pose_corrective_v1": tube_pose_corrective_report,
        },
        quality_report={
            "publishable": False,
            "reason": "independent V8 matrix, vessel, oral, and performance gates pending",
        },
    )
    operator.validate()
    return operator


__all__ = [
    "build_frozen_domains_v8",
    "build_selective_source_operator_v8",
    "merge_v71_authority_v8",
    "sanitize_v8_runtime_metadata",
]
