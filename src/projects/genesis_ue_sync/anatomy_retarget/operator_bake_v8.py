"""Selective V8 operator assembly from the fitted product and V71 authority.

This module is an explicit migration boundary.  It may read the rejected V7
candidate to recover its beta response bases, but no V7 pose-time hinge,
patella oracle, tube-frame patch, shrink profile, or self-reported verdict is
copied into the resulting schema-v8 operator.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .anatomy_lbs import with_source_driver_coupling
from .articular_fit_v8 import calibrate_coupled_joint_roll_glide_v8
from .functional_joint_v8 import (
    build_functional_joint_frames_v8,
    build_pelvis_harmonic_cage_v8,
)
from .mechanism_v8 import (
    V71ParentLocalFKV8,
    build_ba9_head_selection_v8,
    topology_digest_v8,
)
from .rigged_asset import AnatomyRiggedAsset
from .reference_fit_v8 import compose_unified_reference_template_v8
from .tube_frames_v8 import (
    bake_tube_coupling_v8,
    tube_coupling_pack_to_runtime_fields_v8,
)
from .v7_artifacts import SourceOperatorV7, rigged_asset_digest
from .v8_artifacts import SourceOperatorV8
from .vessel_route_v8 import bake_vessel_route_v8


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
            "source_full_local_fk_v2": True,
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
    merged_metadata = dict(merged.metadata or {})
    bone_names = list(merged.source_bone_names or ())
    merged_metadata["source_direct_driver_bones_v1"] = [
        bone_names.index(name)
        for name in ("Spine_C7", "Head_Bone", "Jaw_Bone_base")
    ]
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


def _mesh_ids_matching(
    asset: AnatomyRiggedAsset,
    *,
    suffix: str,
    tokens: tuple[str, ...],
) -> np.ndarray:
    selected = [
        _mesh_ids(asset, str(name))
        for name in (asset.source_mesh_names or ())
        if str(name).endswith(f"_{suffix}")
        and any(token in str(name).lower() for token in tokens)
    ]
    if not selected:
        raise ValueError(
            f"required {suffix} mesh group with tokens {tokens!r} is missing"
        )
    return np.unique(np.concatenate(selected)).astype(np.int32)


def build_frozen_domains_v8(
    asset: AnatomyRiggedAsset,
    legacy_joint_domains: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Freeze joint, head, and rib domains once against product topology."""
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    result: dict[str, np.ndarray] = {}
    for name, ids in legacy_joint_domains.items():
        selected = np.asarray(ids, dtype=np.int32).reshape(-1)
        domain_name = str(name)
        if domain_name.endswith((".fit", ".validation")):
            # V8 operators already carry immutable, disjoint material
            # partitions.  Re-baking a candidate from one of those operators
            # must preserve the IDs instead of recursively creating
            # ``.fit.fit`` domains and changing the validation oracle.
            result[domain_name] = selected.copy()
        elif len(selected) >= 8:
            fit, validation = _split_spatial_domain(vertices, selected)
            result[f"{domain_name}.fit"] = fit
            result[f"{domain_name}.validation"] = validation
        else:
            result[domain_name] = selected

    for side in ("L", "R"):
        humerus = _mesh_ids(asset, f"Humerus_{side}")
        scapula = _mesh_ids(asset, f"Scapula_{side}")
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

        shoulder_interface = {
            "humerus": _near_interface(
                vertices, humerus, scapula, fraction=0.20
            ),
            "scapula": _near_interface(
                vertices, scapula, humerus, fraction=0.12
            ),
        }
        for part, ids in shoulder_interface.items():
            fit, validation = _split_spatial_domain(vertices, ids)
            result[f"shoulder/{label}/{part}.fit"] = fit
            result[f"shoulder/{label}/{part}.validation"] = validation
        humeral_head = _near_interface(
            vertices, humerus, scapula, fraction=0.12
        )
        fit, validation = _split_spatial_domain(vertices, humeral_head)
        result[f"shoulder/{label}/humeral_head.fit"] = fit
        result[f"shoulder/{label}/humeral_head.validation"] = validation

        carpals = _mesh_ids_matching(
            asset,
            suffix=side,
            tokens=(
                "scaphoid",
                "lunate",
                "triquetr",
                "pisiform",
                "trapezium",
                "trapezoid",
                "capitate",
                "hamate",
            ),
        )
        wrist_interface = {
            "radius": _near_interface(
                vertices, radius, carpals, fraction=0.16
            ),
            "ulna": _near_interface(vertices, ulna, carpals, fraction=0.16),
            "carpals": _near_interface(
                vertices,
                carpals,
                np.concatenate((radius, ulna)),
                fraction=0.28,
            ),
        }
        for part, ids in wrist_interface.items():
            fit, validation = _split_spatial_domain(vertices, ids)
            result[f"wrist/{label}/{part}.fit"] = fit
            result[f"wrist/{label}/{part}.validation"] = validation

        for digit, ordinal in enumerate(
            ("1st", "2nd", "3rd", "4th", "5th"), start=1
        ):
            metacarpal = _mesh_ids(asset, f"_{ordinal}_Metacarpal_{side}")
            proximal = _mesh_ids(
                asset, f"_{ordinal}_Proximal_Phalanges_Hand_{side}"
            )
            middle_name = f"_{ordinal}_Intermediate_Phalanges_Hand_{side}"
            middle = (
                _mesh_ids(asset, middle_name)
                if middle_name in (asset.source_mesh_names or ())
                else np.zeros(0, dtype=np.int32)
            )
            distal = _mesh_ids(
                asset, f"_{ordinal}_Distal_Phalanges_Hand_{side}"
            )
            hand_interfaces = {
                "metacarpal_mcp": _near_interface(
                    vertices, metacarpal, proximal, fraction=0.20
                ),
                "proximal_mcp": _near_interface(
                    vertices, proximal, metacarpal, fraction=0.20
                ),
            }
            if digit == 1:
                hand_interfaces.update(
                    {
                        "carpals_cmc": _near_interface(
                            vertices, carpals, metacarpal, fraction=0.12
                        ),
                        "metacarpal_cmc": _near_interface(
                            vertices, metacarpal, carpals, fraction=0.20
                        ),
                    }
                )
            if len(middle):
                hand_interfaces.update(
                    {
                        "proximal_pip": _near_interface(
                            vertices, proximal, middle, fraction=0.20
                        ),
                        "middle_pip": _near_interface(
                            vertices, middle, proximal, fraction=0.22
                        ),
                        "middle_dip": _near_interface(
                            vertices, middle, distal, fraction=0.22
                        ),
                        "distal_dip": _near_interface(
                            vertices, distal, middle, fraction=0.22
                        ),
                    }
                )
            else:
                hand_interfaces.update(
                    {
                        "proximal_ip": _near_interface(
                            vertices, proximal, distal, fraction=0.22
                        ),
                        "distal_ip": _near_interface(
                            vertices, distal, proximal, fraction=0.22
                        ),
                    }
                )
            for part, ids in hand_interfaces.items():
                fit, validation = _split_spatial_domain(vertices, ids)
                result[f"hand/{label}/digit{digit}/{part}.fit"] = fit
                result[f"hand/{label}/digit{digit}/{part}.validation"] = validation

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
    algorithm_version: str = "selective-v8.1",
    oracle_version: str = "contact-independent-v8.1",
    correction_version: str = "ba9-head-v8.1",
) -> SourceOperatorV8:
    """Assemble L0 without copying V7 pose-time mechanisms or verdicts."""
    unified_report: dict[str, Any] | None = None
    unified_coefficients: dict[str, np.ndarray] = {}
    vessel_route_report: dict[str, Any] = {
        "available": False,
        "passed": False,
        "reason": "canonical subject skin was not supplied",
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
    # The source-weighted reconstruction must use the restored original V71
    # 14-slot Armature field.  The route changes vessel rest coordinates only;
    # final whole-bone geometry, bind matrices, and runtime FK stay untouched.
    if vessel_skin_vertices is not None:
        template, vessel_route_report = bake_vessel_route_v8(
            template,
            skin_vertices=np.asarray(vessel_skin_vertices, dtype=np.float64),
            skin_faces=np.asarray(vessel_skin_faces, dtype=np.int32),
        )
    domains = build_frozen_domains_v8(template, v7_operator.fixed_material_domains)
    functional_frames = build_functional_joint_frames_v8(
        template,
        domains=domains,
    )
    pelvis_cage_fields, pelvis_cage_report = build_pelvis_harmonic_cage_v8(
        template,
        domains=domains,
    )
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
        template = with_source_driver_coupling(template)
    else:
        coupled_joint_report = {
            "available": False,
            "reason": "operator lacks bilateral frozen knee/ankle domains",
        }
    final_runtime_coefficients = {
        str(name): np.asarray(value).copy()
        for name, value in runtime_coefficients.items()
        if not str(name).startswith("tube_coupling_v8.")
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
    else:
        tube_coupling_report = {
            "available": False,
            "passed": False,
            "reason": "final L0 template contains no vessel or nerve material",
        }
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
        **functional_frames.coefficient_fields(),
        **pelvis_cage_fields,
        **unified_coefficients,
    }
    operator = SourceOperatorV8(
        template_asset=template,
        beta_vertex_basis=np.asarray(v7_operator.beta_vertex_basis, dtype=np.float32),
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
            "source_full_local_fk_v2": True,
            "unified_reference_fit": unified_report is not None,
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
            "functional_joint_frames_v8": dict(functional_frames.report),
            "pelvis_harmonic_cage_v8": pelvis_cage_report,
            "vessel_route_v8": vessel_route_report,
            "tube_coupling_final_rest_v810": tube_coupling_report,
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
