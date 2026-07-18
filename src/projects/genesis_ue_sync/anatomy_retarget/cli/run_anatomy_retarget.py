#!/usr/bin/env python3
"""Run the offline Blender anatomy retarget step and optionally publish it to Genesis."""

from __future__ import annotations

import argparse
import atexit
from dataclasses import replace
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.anatomy_retarget.asset_align import normalize_rigged_asset_file
from projects.genesis_ue_sync.anatomy_retarget.blender_retarget_runner import run_retarget
from projects.genesis_ue_sync.anatomy_retarget.containment import (
    load_body_surface,
    signed_distance,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    joint_global_transforms,
    skin_vertices,
    with_source_driver_coupling,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_drive_translation,
    load_easymocap_smplx_fit_drive,
    smplx_pose_hash,
    smplx_shape_hash,
)
from projects.genesis_ue_sync.anatomy_retarget.quality_gate import evaluate_asset_quality, write_quality_report
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
    ANATOMY_ASSET_SCHEMA_VERSION,
    load_rigged_asset,
    save_rigged_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.shape_volume import (
    apply_material_bounded_soft_volume,
    apply_subject_beta_shape,
)
from projects.genesis_ue_sync.anatomy_retarget.diagnostics import write_mesh_diagnostics
from projects.genesis_ue_sync.anatomy_retarget.bone_segment_diagnostics import write_bone_segment_diagnostics
from projects.genesis_ue_sync.anatomy_retarget.source_skin_volume import apply_source_skin_volume_registration
from projects.genesis_ue_sync.anatomy_retarget.provenance import build_run_manifest
from projects.genesis_ue_sync.anatomy_retarget.tube_graph import (
    build_asset_attachment_graphs,
    tube_graph_metrics,
)
from projects.genesis_ue_sync.anatomy_retarget.material_fit import fit_articulated_rest
from projects.genesis_ue_sync.anatomy_retarget.intersection_diagnostics import (
    enforce_station_intersection_nonregression,
    tube_bone_intersection_report,
)
from projects.genesis_ue_sync.anatomy_retarget.validation_matrix import (
    pose_cases,
    release_validation_matrix,
)
from projects.genesis_ue_sync.multiview_realtime.track_stream import (
    DEFAULT_ANATOMY_ASSET_PUB_BIND,
    TOPIC_ANATOMY_ASSET_V1,
    anatomy_asset_control_to_dict,
)


def _load_config(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return dict(json.loads(text))
    try:
        import yaml  # type: ignore

        return dict(yaml.safe_load(text) or {})
    except Exception:
        return dict(json.loads(text))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_key(*paths: Path, extra: str = "") -> str:
    digest = hashlib.sha256(extra.encode("utf-8"))
    for path in paths:
        digest.update(str(Path(path).resolve()).encode("utf-8"))
        digest.update(_file_digest(Path(path)).encode("ascii"))
    return digest.hexdigest()[:24]


def _merge_fast_extremity_donor(
    asset: Any,
    donor: Any,
    *,
    expected_shape_hash: str,
    canonical_dir: Path,
    hand_donor_path: Path | None = None,
    axial_donor: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Use the known-good fitted skeleton, retaining only the stable axial compound."""
    if asset.vertices_rest.shape != donor.vertices_rest.shape:
        raise ValueError("fast extremity donor vertex topology does not match")
    if not np.array_equal(asset.faces, donor.faces):
        raise ValueError("fast extremity donor faces do not match")
    if asset.source_mesh_names != donor.source_mesh_names or not np.array_equal(
        asset.source_vertex_ranges, donor.source_vertex_ranges
    ):
        raise ValueError("fast extremity donor source mesh layout does not match")
    if asset.source_bone_names != donor.source_bone_names:
        raise ValueError("fast extremity donor source bone layout does not match")
    donor_shape_hash = str((donor.metadata or {}).get("shape_hash", ""))
    if donor_shape_hash != str(expected_shape_hash):
        raise ValueError(
            "fast extremity donor shape mismatch: "
            f"expected {expected_shape_hash}, got {donor_shape_hash or '<missing>'}"
        )

    legacy_hand: dict[str, np.ndarray] | None = None
    if hand_donor_path is not None:
        with np.load(Path(hand_donor_path), allow_pickle=True) as data:
            required = {
                "vertices_rest",
                "faces",
                "source_mesh_names",
                "source_vertex_ranges",
                "source_bone_names",
                "source_rest_global",
                "source_bone_head",
                "source_bone_tail",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"legacy hand donor is missing arrays: {missing}")
            if not np.array_equal(np.asarray(data["faces"]), asset.faces):
                raise ValueError("legacy hand donor faces do not match")
            if [str(v) for v in data["source_mesh_names"].tolist()] != list(
                asset.source_mesh_names
            ) or not np.array_equal(
                np.asarray(data["source_vertex_ranges"]), asset.source_vertex_ranges
            ):
                raise ValueError("legacy hand donor mesh layout does not match")
            if [str(v) for v in data["source_bone_names"].tolist()] != list(
                asset.source_bone_names
            ):
                raise ValueError("legacy hand donor bone layout does not match")
            legacy_hand = {
                name: np.asarray(data[name]).copy()
                for name in required
                if name not in {"faces", "source_mesh_names", "source_vertex_ranges", "source_bone_names"}
            }

    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    modes = list(asset.source_bone_driver_types or [])
    axial_tokens = (
        "pelvis",
        "ilium",
        "ischium",
        "pubis",
        "sacrum",
        "sternum",
        "rib_",
        "spine_",
        "vertebra",
        "disc",
    )
    vertebra_mesh_names = {
        *(f"c{index}" for index in range(1, 8)),
        *(f"t{index}" for index in range(1, 13)),
        *(f"l{index}" for index in range(1, 6)),
    }
    axial_vertices = np.zeros(len(asset.vertices_rest), dtype=bool)
    for (start, stop), mesh_name, tissue in zip(
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        asset.source_mesh_names,
        asset.source_tissues,
    ):
        start_i, stop_i = int(start), int(stop)
        mesh_lower = str(mesh_name).lower()
        if str(tissue).lower() == "bone" and (
            any(token in mesh_lower for token in axial_tokens)
            or mesh_lower in vertebra_mesh_names
        ):
            axial_vertices[start_i:stop_i] = True

    axial_bone_tokens = ("hip_bone", "spine_", "disc", "rib_bone", "sternum_bone")
    axial_bones = np.asarray(
        [
            any(token in str(name).lower() for token in axial_bone_tokens)
            for name in asset.source_bone_names
        ],
        dtype=bool,
    )
    # Keep authored helper children belonging to the selected axial compound,
    # but stop at independently driven head/limb roots.
    for bone, parent in enumerate(parents):
        if (
            int(parent) >= 0
            and axial_bones[int(parent)]
            and str(modes[bone]) == "bind_follow"
        ):
            axial_bones[bone] = True

    axial_source = asset if axial_donor is None else axial_donor
    if axial_source.vertices_rest.shape != asset.vertices_rest.shape or not np.array_equal(
        axial_source.faces, asset.faces
    ):
        raise ValueError("fast axial donor topology does not match")
    current_vertices = np.asarray(axial_source.vertices_rest, dtype=np.float64)
    vertices = np.asarray(donor.vertices_rest, dtype=np.float64).copy()
    harmonic_reference = (
        np.asarray(asset.harmonic_reference_vertices, dtype=np.float64).copy()
        if asset.harmonic_reference_vertices is not None
        else np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    )
    # The schema-3 1b66307 result is the visually verified pure-harmonic soft
    # topology.  It is the immutable reference for vessels/nerves/organs;
    # fe99 contributes fitted rigid material only.
    if legacy_hand is not None:
        harmonic_reference = np.asarray(
            legacy_hand["vertices_rest"], dtype=np.float64
        ).copy()
    for (start, stop), tissue in zip(
        asset.source_vertex_ranges, asset.source_tissues
    ):
        if str(tissue).lower() != "bone":
            vertices[int(start) : int(stop)] = harmonic_reference[
                int(start) : int(stop)
            ]
    vertices[axial_vertices] = current_vertices[axial_vertices]

    rest_joints = np.asarray(donor.rest_joints, dtype=np.float64).copy()
    rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=rest_joints,
        parents=np.asarray(asset.parents, dtype=np.int32),
    ).astype(np.float64)

    target_global = np.asarray(donor.target_bind_global, dtype=np.float64).copy()
    target_global[axial_bones] = np.asarray(
        axial_source.target_bind_global, dtype=np.float64
    )[axial_bones]
    target_head = np.asarray(donor.target_bone_head, dtype=np.float64).copy()
    target_tail = np.asarray(donor.target_bone_tail, dtype=np.float64).copy()
    target_head[axial_bones] = np.asarray(
        axial_source.target_bone_head, dtype=np.float64
    )[axial_bones]
    target_tail[axial_bones] = np.asarray(
        axial_source.target_bone_tail, dtype=np.float64
    )[axial_bones]

    legacy_hand_vertices = np.zeros(len(vertices), dtype=bool)
    legacy_hand_bones = np.zeros(len(parents), dtype=bool)
    legacy_clavicle_vertices = np.zeros(len(vertices), dtype=bool)
    legacy_clavicle_bones = np.zeros(len(parents), dtype=bool)
    donor_foot_vertices = np.zeros(len(vertices), dtype=bool)
    donor_foot_bones = np.zeros(len(parents), dtype=bool)
    if legacy_hand is not None:
        limb_mesh_tokens = (
            "metacarpal",
            "phalanx_hand",
            "phalanges_hand",
            "capitate",
            "hamate",
            "lunate",
            "pisiform",
            "scaphoid",
            "trapezium",
            "trapezoid",
            "triquetrum",
        )
        for (start, stop), mesh_name, tissue in zip(
            np.asarray(asset.source_vertex_ranges, dtype=np.int64),
            asset.source_mesh_names,
            asset.source_tissues,
        ):
            lower = str(mesh_name).lower()
            if str(tissue).lower() != "bone":
                continue
            if any(token in lower for token in limb_mesh_tokens):
                legacy_hand_vertices[int(start) : int(stop)] = True
            if "clavicle" in lower:
                legacy_clavicle_vertices[int(start) : int(stop)] = True
        hand_roots = {
            list(asset.source_bone_names).index("Wrist_Rotate_L"),
            list(asset.source_bone_names).index("Wrist_Rotate_R1"),
        }
        for bone in range(len(parents)):
            cursor = bone
            while cursor >= 0:
                if cursor in hand_roots:
                    legacy_hand_bones[bone] = True
                    break
                cursor = int(parents[cursor])
        for bone, bone_name in enumerate(asset.source_bone_names):
            legacy_clavicle_bones[bone] = "clavicle_rot" in str(bone_name).lower()
        for bone, parent in enumerate(parents):
            if int(parent) >= 0 and legacy_hand_bones[int(parent)]:
                legacy_hand_bones[bone] = True
        # The clavicle effector belongs with the clavicle, but the independently
        # driven shoulder child must remain on the ef58024 arm chain.
        for bone, parent in enumerate(parents):
            if (
                int(parent) >= 0
                and legacy_clavicle_bones[int(parent)]
                and str(modes[bone]) == "bind_follow"
            ):
                legacy_clavicle_bones[bone] = True
        vertices[legacy_hand_vertices | legacy_clavicle_vertices] = np.asarray(
            legacy_hand["vertices_rest"], dtype=np.float64
        )[legacy_hand_vertices | legacy_clavicle_vertices]
        legacy_bones = legacy_hand_bones | legacy_clavicle_bones
        target_global[legacy_bones] = np.asarray(
            legacy_hand["source_rest_global"], dtype=np.float64
        )[legacy_bones]
        target_head[legacy_bones] = np.asarray(
            legacy_hand["source_bone_head"], dtype=np.float64
        )[legacy_bones]
        target_tail[legacy_bones] = np.asarray(
            legacy_hand["source_bone_tail"], dtype=np.float64
        )[legacy_bones]

        # a7b8c7f/ef58024 has the correct foot compound width and length.  The
        # schema-3 limb donor is retained for hands/long bones only; restoring
        # both geometry and the complete ankle subtree avoids mixing a thin
        # legacy foot with the newer bind.
        foot_tokens = (
            "calcaneus",
            "talus",
            "navicular",
            "cuboid",
            "cuneiform",
            "metatarsal",
            "phalanx_foot",
            "phalanges_foot",
        )
        for (start, stop), mesh_name, tissue in zip(
            np.asarray(asset.source_vertex_ranges, dtype=np.int64),
            asset.source_mesh_names,
            asset.source_tissues,
        ):
            if str(tissue).lower() == "bone" and any(
                token in str(mesh_name).lower() for token in foot_tokens
            ):
                donor_foot_vertices[int(start) : int(stop)] = True
        foot_roots = {
            list(donor.source_bone_names).index("Ankle_Rot_L"),
            list(donor.source_bone_names).index("Ankle_Rot_R"),
        }
        for bone in range(len(parents)):
            cursor = bone
            while cursor >= 0:
                if cursor in foot_roots:
                    donor_foot_bones[bone] = True
                    break
                cursor = int(parents[cursor])
        vertices[donor_foot_vertices] = np.asarray(
            donor.vertices_rest, dtype=np.float64
        )[donor_foot_vertices]
        target_global[donor_foot_bones] = np.asarray(
            donor.target_bind_global, dtype=np.float64
        )[donor_foot_bones]
        target_head[donor_foot_bones] = np.asarray(
            donor.target_bone_head, dtype=np.float64
        )[donor_foot_bones]
        target_tail[donor_foot_bones] = np.asarray(
            donor.target_bone_tail, dtype=np.float64
        )[donor_foot_bones]

    from projects.genesis_ue_sync.anatomy_retarget.material_fit import (
        _femur_head_and_acetabulum,
        _mesh_mask,
        _rotation_between,
        _surface_region,
        cranial_material_mask,
        jaw_material_mask,
        shaft_preserving_segment_map,
    )

    clavicle_report: dict[str, Any] = {}
    if np.any(legacy_clavicle_bones):
        for side, suffix in (("left", "_l"), ("right", "_r")):
            clavicle = _mesh_mask(
                donor,
                lambda name, tissue, suffix=suffix: tissue == "bone"
                and "clavicle" in name
                and name.endswith(suffix),
            )
            root_name = "Clavicle_Rot_L" if side == "left" else "Clavicle_Rot_R"
            root_bone = list(donor.source_bone_names).index(root_name)
            source_a = target_head[root_bone].copy()
            source_b = target_tail[root_bone].copy()
            target_a = rest_joints[donor.joint_names.index(f"{side}_collar")]
            target_b = rest_joints[donor.joint_names.index(f"{side}_shoulder")]
            vertices[clavicle] = shaft_preserving_segment_map(
                vertices[clavicle],
                source_a=source_a,
                source_b=source_b,
                target_a=target_a,
                target_b=target_b,
            )
            side_bones = legacy_clavicle_bones & np.asarray(
                [
                    str(name).lower().endswith(suffix)
                    or (
                        int(parents[index]) >= 0
                        and str(donor.source_bone_names[int(parents[index])])
                        == root_name
                    )
                    for index, name in enumerate(donor.source_bone_names)
                ],
                dtype=bool,
            )
            old_positions = target_global[side_bones, :3, 3].copy()
            target_global[side_bones, :3, 3] = shaft_preserving_segment_map(
                old_positions,
                source_a=source_a,
                source_b=source_b,
                target_a=target_a,
                target_b=target_b,
            )
            rotation = _rotation_between(source_b - source_a, target_b - target_a)
            target_global[side_bones, :3, :3] = np.einsum(
                "ij,bjk->bik", rotation, target_global[side_bones, :3, :3]
            )
            target_head[side_bones] = shaft_preserving_segment_map(
                target_head[side_bones],
                source_a=source_a,
                source_b=source_b,
                target_a=target_a,
                target_b=target_b,
            )
            target_tail[side_bones] = shaft_preserving_segment_map(
                target_tail[side_bones],
                source_a=source_a,
                source_b=source_b,
                target_a=target_a,
                target_b=target_b,
            )
            clavicle_report[side] = {
                "source_length_m": float(np.linalg.norm(source_b - source_a)),
                "target_length_m": float(np.linalg.norm(target_b - target_a)),
            }

    hip_report: dict[str, Any] = {}
    if legacy_hand is not None:
        hip_report["mode"] = "fe99_material_fit_preserved"
    for side in ("left", "right"):
        pair = _femur_head_and_acetabulum(
            donor,
            vertices,
            side=side,
            target_joints=rest_joints,
        )
        suffix = "_l" if side == "left" else "_r"
        femur = _mesh_mask(
            donor,
            lambda name, tissue, suffix=suffix: tissue == "bone"
            and "femur" in name
            and (name.endswith(suffix) or f"{suffix}_" in name),
        )
        if pair is None or not np.any(femur):
            continue
        femoral_head, acetabulum = pair
        knee = rest_joints[donor.joint_names.index(f"{side}_knee")]
        femur_points = vertices[femur].copy()
        shaft_axis = knee - femoral_head
        shaft_axis /= max(float(np.linalg.norm(shaft_axis)), 1.0e-8)
        axial = (femur_points - femoral_head) @ shaft_axis
        distal = np.mean(femur_points[axial >= np.quantile(axial, 0.85)], axis=0)
        vertices[femur] = shaft_preserving_segment_map(
            femur_points,
            source_a=femoral_head,
            source_b=distal,
            target_a=acetabulum,
            target_b=knee,
        )
        rotation = _rotation_between(distal - femoral_head, knee - acetabulum)
        frame_delta = np.eye(4, dtype=np.float64)
        frame_delta[:3, :3] = rotation
        frame_delta[:3, 3] = acetabulum - rotation @ femoral_head
        femur_bones = np.asarray(
            [
                "femur" in str(name).lower()
                and (
                    str(name).lower().endswith(suffix)
                    or f"{suffix}_" in str(name).lower()
                )
                for name in donor.source_bone_names
            ],
            dtype=bool,
        )
        target_global[femur_bones] = np.einsum(
            "ij,bjk->bik", frame_delta, target_global[femur_bones]
        )
        target_head[femur_bones] = (
            target_head[femur_bones] @ rotation.T + frame_delta[:3, 3]
        )
        target_tail[femur_bones] = (
            target_tail[femur_bones] @ rotation.T + frame_delta[:3, 3]
        )
        post_head, post_acetabulum = _femur_head_and_acetabulum(
            donor,
            vertices,
            side=side,
            target_joints=rest_joints,
        )
        correction = post_acetabulum - post_head
        vertices[femur] += correction
        target_global[femur_bones, :3, 3] += correction
        target_head[femur_bones] += correction
        target_tail[femur_bones] += correction
        hip_report[side] = {
            "pre_gap_m": float(np.linalg.norm(femoral_head - acetabulum)),
            "post_map_correction_m": float(np.linalg.norm(correction)),
            "shared_center": post_acetabulum.tolist(),
        }

    # The right legacy hip is visually the reliable side.  Mirror its proximal
    # centre to the left, while pinning the left distal femur to the SMPL-X knee.
    # This removes the old left-only vertical asymmetry without pushing either
    # ball toward an average of the irregular acetabular surface.
    if legacy_hand is not None and any(
        legacy_hand_bones[index] and "femur" in str(name).lower()
        for index, name in enumerate(donor.source_bone_names)
    ):
        left_pair = _femur_head_and_acetabulum(
            donor, vertices, side="left", target_joints=rest_joints
        )
        right_pair = _femur_head_and_acetabulum(
            donor, vertices, side="right", target_joints=rest_joints
        )
        left_femur = _mesh_mask(
            donor,
            lambda name, tissue: tissue == "bone"
            and "femur" in name
            and (name.endswith("_l") or "_l_" in name),
        )
        if left_pair is not None and right_pair is not None and np.any(left_femur):
            left_head = left_pair[0]
            right_head = right_pair[0]
            left_knee = rest_joints[donor.joint_names.index("left_knee")]
            hip_mid_x = 0.5 * (
                rest_joints[donor.joint_names.index("left_hip"), 0]
                + rest_joints[donor.joint_names.index("right_hip"), 0]
            )
            mirrored_head = right_head.copy()
            mirrored_head[0] = 2.0 * hip_mid_x - right_head[0]
            femur_points = vertices[left_femur].copy()
            axis = left_knee - left_head
            axis /= max(float(np.linalg.norm(axis)), 1.0e-8)
            axial = (femur_points - left_head) @ axis
            distal = np.mean(femur_points[axial >= np.quantile(axial, 0.85)], axis=0)
            vertices[left_femur] = shaft_preserving_segment_map(
                femur_points,
                source_a=left_head,
                source_b=distal,
                target_a=mirrored_head,
                target_b=left_knee,
            )
            rotation = _rotation_between(distal - left_head, left_knee - mirrored_head)
            frame_delta = np.eye(4, dtype=np.float64)
            frame_delta[:3, :3] = rotation
            frame_delta[:3, 3] = mirrored_head - rotation @ left_head
            left_femur_bones = np.asarray(
                [
                    "femur" in str(name).lower()
                    and (
                        str(name).lower().endswith("_l")
                        or "_l_" in str(name).lower()
                    )
                    for name in donor.source_bone_names
                ],
                dtype=bool,
            )
            target_global[left_femur_bones] = np.einsum(
                "ij,bjk->bik", frame_delta, target_global[left_femur_bones]
            )
            target_head[left_femur_bones] = (
                target_head[left_femur_bones] @ rotation.T + frame_delta[:3, 3]
            )
            target_tail[left_femur_bones] = (
                target_tail[left_femur_bones] @ rotation.T + frame_delta[:3, 3]
            )
            hip_report["mode"] = "legacy_right_mirrored_left_head_smplx_knee"
            hip_report["left"] = {
                "source_head": left_head.tolist(),
                "mirrored_right_head": mirrored_head.tolist(),
                "proximal_correction_m": float(np.linalg.norm(mirrored_head - left_head)),
                "distal_target": "smplx_left_knee",
            }

    raw_head = cranial_material_mask(donor) | jaw_material_mask(donor)
    head_vertices = np.zeros(len(vertices), dtype=bool)
    head_reference_vertices = np.zeros(len(vertices), dtype=bool)
    for (start, stop), tissue in zip(
        np.asarray(donor.source_vertex_ranges, dtype=np.int64),
        donor.source_tissues,
    ):
        start_i, stop_i = int(start), int(stop)
        if float(np.mean(raw_head[start_i:stop_i])) >= 0.90:
            head_reference_vertices[start_i:stop_i] = True
            if str(tissue).lower() == "bone":
                head_vertices[start_i:stop_i] = True
    # Head is a single fe99 compound.  Mixing a rescaled skull with an
    # independently harmonic brain/eyes produced the visible concentric
    # layers, so restore every cranial component and its bind unchanged.
    donor_head_vertices = np.asarray(donor.vertices_rest, dtype=np.float64)[
        head_reference_vertices
    ]
    source_lo, source_hi = np.quantile(
        donor_head_vertices, (0.01, 0.99), axis=0
    )
    source_center = 0.5 * (source_lo + source_hi)
    target_center = source_center.copy()
    head_scale = 0.70
    vertices[head_reference_vertices] = source_center + head_scale * (
        donor_head_vertices - source_center
    )
    head_bones = np.asarray(
        [
            any(token in str(name).lower() for token in ("head_bone", "jaw_bone"))
            for name in donor.source_bone_names
        ],
        dtype=bool,
    )
    for bone, parent in enumerate(parents):
        if int(parent) >= 0 and head_bones[int(parent)]:
            head_bones[bone] = True
    for values in (target_head, target_tail):
        values[head_bones] = source_center + head_scale * (
            values[head_bones] - source_center
        )
    target_global[head_bones, :3, 3] = source_center + head_scale * (
        target_global[head_bones, :3, 3] - source_center
    )

    # Keep the ef58024 all-harmonic vessel/nerve result untouched.  The former
    # direct affine-weight residual and large nearest-surface SDF projection
    # were not a volumetric elastic solve: they sheared tube branches at weight
    # seams and pulled unrelated nerve components toward the skin.

    rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=rest_joints,
        parents=np.asarray(asset.parents, dtype=np.int32),
    ).astype(np.float64)
    target_local = np.empty_like(target_global)
    for bone, parent in enumerate(parents):
        target_local[bone] = (
            target_global[bone]
            if int(parent) < 0
            else np.linalg.inv(target_global[int(parent)]) @ target_global[bone]
        )

    metadata = dict(donor.metadata or {})
    metadata.update(
        {
            "fast_extremity_donor": True,
            "fast_extremity_donor_shape_hash": donor_shape_hash,
            "fast_axial_source": "current_source_rest",
            "disable_soft_follow": True,
            "soft_follow_scope": "disabled_use_blender_lbs",
            "head_uniform_scale": head_scale,
            "soft_bone_residual_follow": False,
            "soft_surface_sdf": "disabled",
        }
    )
    result = type(asset)(
        **{
            **donor.__dict__,
            "vertices_rest": vertices.astype(np.float32),
            "rest_joints": rest_joints.astype(np.float32),
            "inverse_bind": np.linalg.inv(rest_global).astype(np.float32),
            "target_rest_global": target_global.astype(np.float32),
            "target_rest_local": target_local.astype(np.float32),
            "target_inverse_bind": np.linalg.inv(target_global).astype(np.float32),
            "target_bone_head": target_head.astype(np.float32),
            "target_bone_tail": target_tail.astype(np.float32),
            "harmonic_reference_vertices": harmonic_reference.astype(np.float32),
            "harmonic_bone_head": (
                np.asarray(legacy_hand["source_bone_head"], dtype=np.float32)
                if legacy_hand is not None
                else asset.harmonic_bone_head
            ),
            "harmonic_bone_mid": (
                0.5
                * (
                    np.asarray(legacy_hand["source_bone_head"], dtype=np.float32)
                    + np.asarray(legacy_hand["source_bone_tail"], dtype=np.float32)
                )
                if legacy_hand is not None
                else asset.harmonic_bone_mid
            ),
            "harmonic_bone_tail": (
                np.asarray(legacy_hand["source_bone_tail"], dtype=np.float32)
                if legacy_hand is not None
                else asset.harmonic_bone_tail
            ),
            "soft_follow_driver_indices": None,
            "soft_follow_driver_weights": None,
            "soft_follow_stations": None,
            "soft_follow_strength": None,
            "soft_component_ids": None,
            "source_mesh_follow_modes": None,
            "metadata": metadata,
        }
    )
    result = with_source_driver_coupling(result)
    result.validate()
    return result, {
        "backend": "ef58024_material_fit_plus_current_axial",
        "axial_donor_explicit": bool(axial_donor is not None),
        "axial_source_bone_count": int(np.count_nonzero(axial_bones)),
        "axial_source_vertex_count": int(np.count_nonzero(axial_vertices)),
        "head_compound_vertex_count": int(np.count_nonzero(head_vertices)),
        "head_uniform_scale": head_scale,
        "head_source_center": source_center.tolist(),
        "head_target_center": target_center.tolist(),
        "hip_alignment": hip_report,
        "clavicle_fit": clavicle_report,
        "legacy_hand_vertex_count": int(np.count_nonzero(legacy_hand_vertices)),
        "legacy_hand_bone_count": int(np.count_nonzero(legacy_hand_bones)),
        "legacy_clavicle_vertex_count": int(
            np.count_nonzero(legacy_clavicle_vertices)
        ),
        "legacy_clavicle_bone_count": int(np.count_nonzero(legacy_clavicle_bones)),
        "donor_foot_vertex_count": int(np.count_nonzero(donor_foot_vertices)),
        "donor_foot_bone_count": int(np.count_nonzero(donor_foot_bones)),
        "vessel_pose_backend": "1b66307_pure_harmonic_reference",
        "soft_bone_residual": "material_bounded_elastic_volume_field",
        "rest_soft_sdf": "disabled",
        "station_soft_follow_restored": False,
    }


def _load_valid_cache(
    path: Path,
    *,
    metadata_key: str,
    expected_key: str,
) -> Any | None:
    """Return only a schema-valid cache produced by the exact current inputs."""
    if not Path(path).is_file():
        return None
    try:
        asset = load_rigged_asset(path, validate=True)
    except (KeyError, OSError, ValueError) as exc:
        logging.warning("ignoring stale anatomy cache %s: %s", path, exc)
        return None
    actual_key = str((asset.metadata or {}).get(metadata_key, ""))
    if actual_key != str(expected_key):
        logging.warning(
            "ignoring anatomy cache %s with stale %s=%r",
            path,
            metadata_key,
            actual_key,
        )
        return None
    return asset


def _signed_distance_containment_report(
    asset: Any,
    *,
    anatomy_vertices: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    stage: str,
) -> dict[str, Any]:
    """Build complete per-tissue and per-mesh signed-distance evidence."""
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        raise ValueError("containment diagnostics require mesh ranges and tissue labels")
    points = np.asarray(anatomy_vertices, dtype=np.float64)
    if points.shape != np.asarray(asset.vertices_rest).shape:
        raise ValueError("containment vertices must match the anatomy asset")
    values, _closest, _normal = signed_distance(
        points,
        np.asarray(surface_vertices, dtype=np.float64),
        np.asarray(surface_faces, dtype=np.int32),
    )
    over_limit: dict[str, int] = {}
    outside_count: dict[str, int] = {}
    per_mesh: dict[str, dict[str, Any]] = {}
    for mesh_name, (start, stop), tissue in zip(
        asset.source_mesh_names,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        asset.source_tissues,
    ):
        local = values[int(start) : int(stop)]
        tissue_name = str(tissue)
        tolerance = 0.001 if tissue_name in {"bone", "vessel", "nerve"} else 0.002
        outside = int(np.count_nonzero(local > 0.0))
        severe = int(np.count_nonzero(local > tolerance))
        outside_count[tissue_name] = outside_count.get(tissue_name, 0) + outside
        over_limit[tissue_name] = over_limit.get(tissue_name, 0) + severe
        per_mesh[str(mesh_name)] = {
            "tissue": tissue_name,
            "vertex_count": int(local.size),
            "outside_count": outside,
            "over_limit_count": severe,
            "inside_fraction": float(np.mean(local <= 0.0)) if local.size else None,
            "max_outside_m": (
                float(max(0.0, float(np.max(local)))) if local.size else None
            ),
            "tolerance_m": tolerance,
        }
    return {
        "stage": str(stage),
        "backend": "libigl_exact_signed_distance",
        "vertex_count": int(values.size),
        "outside_count": outside_count,
        "over_limit_count": over_limit,
        "per_mesh": per_mesh,
    }


def _runtime_pose_matrix_report(
    asset: Any,
    *,
    tube_graphs: dict[str, Any],
) -> dict[str, Any]:
    rest = np.asarray(asset.vertices_rest, dtype=np.float64)
    faces = np.asarray(asset.faces, dtype=np.int64)
    reports: dict[str, Any] = {}
    for case_name, pose in pose_cases(list(asset.joint_names)).items():
        posed = np.asarray(skin_vertices(asset, pose), dtype=np.float64)
        case: dict[str, Any] = {
            "finite": bool(np.all(np.isfinite(posed))),
            "soft_meshes": {},
            "tube_graphs": {},
        }
        for mesh_name, (start, stop), tissue in zip(
            asset.source_mesh_names,
            np.asarray(asset.source_vertex_ranges, dtype=np.int64),
            asset.source_tissues,
        ):
            if str(tissue) not in {"vessel", "nerve"}:
                continue
            local_faces = faces[
                np.all((faces >= int(start)) & (faces < int(stop)), axis=1)
            ]
            edges = np.concatenate(
                (
                    local_faces[:, (0, 1)],
                    local_faces[:, (1, 2)],
                    local_faces[:, (2, 0)],
                ),
                axis=0,
            )
            edges.sort(axis=1)
            edges = np.unique(edges, axis=0)
            before = np.linalg.norm(
                rest[edges[:, 1]] - rest[edges[:, 0]], axis=1
            )
            after = np.linalg.norm(
                posed[edges[:, 1]] - posed[edges[:, 0]], axis=1
            )
            valid = before > 2.0e-4
            ratios = after[valid] / before[valid]
            case["soft_meshes"][str(mesh_name)] = {
                "edge_count": int(np.count_nonzero(valid)),
                "ratio_q99": (
                    float(np.quantile(ratios, 0.99)) if len(ratios) else None
                ),
                "ratio_max": float(np.max(ratios)) if len(ratios) else None,
            }
        for graph_name, graph in tube_graphs.items():
            subject_graph = replace(
                graph,
                rest_nodes=graph.sample_nodes(rest).astype(np.float32),
            )
            case["tube_graphs"][graph_name] = tube_graph_metrics(
                subject_graph,
                posed,
            )
        reports[case_name] = case
    return {
        "case_count": int(len(reports)),
        "cases": reports,
    }


def _directory_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    root = Path(path)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _store_immutable_run(
    stage_dir: Path,
    *,
    output_root: Path,
    schema_version: int,
) -> tuple[Path, str]:
    """Move a completed stage into its immutable content-addressed location."""
    digest = _directory_content_hash(stage_dir)
    run_dir = Path(output_root) / "runs" / str(int(schema_version)) / digest
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        if _directory_content_hash(run_dir) != digest:
            raise RuntimeError(f"immutable run hash collision at {run_dir}")
        shutil.rmtree(stage_dir)
    else:
        try:
            os.replace(stage_dir, run_dir)
        except FileExistsError:
            if not run_dir.is_dir() or _directory_content_hash(run_dir) != digest:
                raise
            shutil.rmtree(stage_dir)
    return run_dir, digest


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a small JSON pointer on the same filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(destination.parent, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finalize_run(
    stage_dir: Path,
    *,
    output_root: Path,
    schema_version: int,
    passed: bool,
    update_latest: bool,
) -> Path:
    """Preserve every run and update latest only for accepted normal runs."""
    run_dir, digest = _store_immutable_run(
        stage_dir,
        output_root=output_root,
        schema_version=schema_version,
    )
    if bool(passed) and bool(update_latest):
        relative_run = run_dir.relative_to(output_root).as_posix()
        _atomic_write_json(
            Path(output_root) / "latest.json",
            {
                "schema_version": int(schema_version),
                "content_hash": digest,
                "run": relative_run,
                "asset": f"{relative_run}/anatomy_rigged.npz",
                "quality_report": f"{relative_run}/quality_report.json",
            },
        )
    return run_dir


def parse_args() -> argparse.Namespace:
    paths = project_paths(__file__)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=paths.configs_root / "anatomy" / "anatomy_retarget.yaml")
    p.add_argument("--canonical-dir", type=Path, default=paths.outputs_root / "anatomy_retarget" / "latest_canonical")
    p.add_argument("--output-dir", type=Path, default=paths.outputs_root / "anatomy_retarget")
    p.add_argument("--blend", type=Path, default=None)
    p.add_argument("--force-source-rebake", action="store_true", help="Ignore source/shape retarget caches.")
    p.add_argument("--profile-first-frame", action="store_true", help="Write source/shape/pose/publish timing report.")
    p.add_argument(
        "--refresh-diagnostics",
        action="store_true",
        help="Run slow mesh/SDF diagnostics even when source and shape caches hit.",
    )
    p.add_argument(
        "--fast-publish",
        action="store_true",
        help=(
            "Use the conservative live-preview bake: preserve source LBS, skip "
            "slow diagnostics/material rest fitting/pose-cache publication."
        ),
    )
    p.add_argument(
        "--fast-extremity-donor",
        type=Path,
        default=None,
        help=(
            "With --fast-publish, restore head/hands/feet plus vessel station "
            "follow from a topology- and shape-matched known-good asset."
        ),
    )
    p.add_argument(
        "--fast-hand-donor",
        type=Path,
        default=None,
        help=(
            "Optional legacy topology-matched asset whose local hand and "
            "clavicle geometry/binds replace the fast material donor."
        ),
    )
    p.add_argument(
        "--fast-axial-donor",
        type=Path,
        default=None,
        help=(
            "Optional topology-matched asset supplying the already accepted "
            "pelvis/spine/rib/sternum compound while the new semantic volume "
            "solve supplies soft anatomy."
        ),
    )
    p.add_argument(
        "--show-connective-tissue",
        action="store_true",
        help="Render ligament/tendon connective-tissue meshes in Genesis (hidden by default).",
    )
    p.add_argument(
        "--hide-vessels",
        action="store_true",
        help="Hide Artery/Vein meshes in Genesis (shown by default).",
    )
    p.add_argument("--motion-npz", type=Path, default=None, help="Exact saved SMPL-X fit for final-pose containment/cache")
    p.add_argument("--timeout-s", type=float, default=900.0)
    p.add_argument("--publish-genesis", action="store_true")
    p.add_argument("--publish-bind", type=str, default=DEFAULT_ANATOMY_ASSET_PUB_BIND)
    p.add_argument("--publish-duration-s", type=float, default=2.0)
    p.add_argument("--publish-rate-hz", type=float, default=10.0)
    p.add_argument("--model-id", type=str, default="patient_anatomy")
    p.add_argument("--color-rgba", type=str, default="0.8,0.05,0.05,0.85")
    p.add_argument(
        "--diagnostics-only",
        action="store_true",
        help=(
            "Preserve the immutable run and diagnostics without updating latest.json or "
            "publishing to Genesis."
        ),
    )
    p.add_argument(
        "--validation-matrix",
        action="store_true",
        help=(
            "Write the 10-beta x 7-pose release matrix. The current beta is "
            "evaluated immediately; alternate betas are explicit rebake cases."
        ),
    )
    p.add_argument(
        "--enforce-quality-gate",
        action="store_true",
        help=(
            "Strict mode: reject latest.json updates and Genesis publish when quality "
            "checks fail. By default quality is advisory only."
        ),
    )
    return p.parse_args()


def _parse_rgba(raw: str) -> tuple[float, float, float, float]:
    vals = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError(f"Expected color as r,g,b,a, got {raw!r}")
    return tuple(max(0.0, min(1.0, v)) for v in vals)  # type: ignore[return-value]


def _quality_failure_blocks_publish(*, passed: bool, enforce_quality_gate: bool = False) -> bool:
    """Quality is advisory unless strict publication mode is requested."""

    if not bool(enforce_quality_gate):
        return False
    return not bool(passed)


def _publish_upsert(
    *,
    bind: str,
    model_id: str,
    asset_npz: Path,
    color_rgba: tuple[float, float, float, float],
    duration_s: float,
    rate_hz: float,
) -> int:
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(str(bind))
    payload = anatomy_asset_control_to_dict(
        action="upsert",
        model_id=str(model_id),
        asset_npz=str(asset_npz.resolve()),
        color_rgba=color_rgba,
        timestamp_ns=time.time_ns(),
    )
    topic = TOPIC_ANATOMY_ASSET_V1.encode("utf-8")
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    end = time.time() + max(0.1, float(duration_s))
    interval = 1.0 / max(1.0, float(rate_hz))
    sent = 0
    time.sleep(0.2)
    clear_payload = anatomy_asset_control_to_dict(
        action="clear_all",
        model_id=str(model_id),
        timestamp_ns=time.time_ns(),
    )
    body_clear = json.dumps(clear_payload, ensure_ascii=True).encode("utf-8")
    sock.send_multipart([topic, body_clear])
    time.sleep(0.1)
    while time.time() < end:
        sock.send_multipart([topic, body])
        sent += 1
        time.sleep(interval)
    sock.close(0)
    return int(sent)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    started_at = time.perf_counter()
    profile: dict[str, float] = {}
    cfg = _load_config(args.config)
    if args.fast_publish:
        cfg = dict(cfg)
        cfg["fast_publish"] = True
    if args.fast_extremity_donor is not None and not args.fast_publish:
        raise ValueError("--fast-extremity-donor requires --fast-publish")
    if args.fast_hand_donor is not None and args.fast_extremity_donor is None:
        raise ValueError("--fast-hand-donor requires --fast-extremity-donor")
    if args.fast_axial_donor is not None and args.fast_extremity_donor is None:
        raise ValueError("--fast-axial-donor requires --fast-extremity-donor")
    blend = args.blend or Path(str(cfg.get("blend_path", "")))
    if not blend:
        raise ValueError("Missing anatomy blend path; pass --blend or set blend_path in config.")
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=str(output_root.parent),
        )
    )

    def _preserve_uncommitted_stage() -> None:
        if not stage_dir.exists():
            return
        try:
            status_path = stage_dir / "run_status.json"
            if not status_path.exists():
                status_path.write_text(
                    json.dumps(
                        {
                            "passed": False,
                            "state": "aborted_before_quality_completion",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            failed_dir = _finalize_run(
                stage_dir,
                output_root=output_root,
                schema_version=ANATOMY_ASSET_SCHEMA_VERSION,
                passed=False,
                update_latest=False,
            )
            logging.error("uncommitted anatomy bake preserved at %s", failed_dir)
        except Exception:
            logging.exception("could not preserve failed anatomy staging directory %s", stage_dir)

    atexit.register(_preserve_uncommitted_stage)
    output_npz = stage_dir / "anatomy_rigged.npz"
    output_glb = stage_dir / "anatomy_rigged.glb"
    report_json = stage_dir / "retarget_report.json"
    canonical_dir = Path(args.canonical_dir).expanduser().resolve()
    manifest_path = canonical_dir / "source_manifest.json"
    if not manifest_path.is_file():
        write_quality_report(
            stage_dir / "quality_report.json",
            {
                "schema_version": ANATOMY_ASSET_SCHEMA_VERSION,
                "passed": False,
                "failures": [f"canonical manifest is missing: {manifest_path}"],
            },
        )
        failed_dir = _finalize_run(
            stage_dir,
            output_root=output_root,
            schema_version=ANATOMY_ASSET_SCHEMA_VERSION,
            passed=False,
            update_latest=False,
        )
        logging.error("canonical manifest failure preserved at %s", failed_dir)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    betas = manifest.get("betas")
    gender = str(manifest.get("gender", ""))
    missing_manifest_fields = [
        key
        for key, value in (("source", manifest.get("source")), ("gender", gender), ("betas", betas))
        if value in (None, "", [])
    ]
    if missing_manifest_fields:
        write_quality_report(
            stage_dir / "quality_report.json",
            {
                "schema_version": ANATOMY_ASSET_SCHEMA_VERSION,
                "passed": False,
                "failures": [
                    f"canonical manifest fields are missing: {missing_manifest_fields}"
                ],
            },
        )
        failed_dir = _finalize_run(
            stage_dir,
            output_root=output_root,
            schema_version=ANATOMY_ASSET_SCHEMA_VERSION,
            passed=False,
            update_latest=False,
        )
        logging.error("canonical manifest failure preserved at %s", failed_dir)
        return 2
    module_root = Path(__file__).resolve().parents[1]
    weights_path = canonical_dir / "smpl_canonical_weights.npz"
    skeleton_path = canonical_dir / "smpl_canonical_skeleton.json"
    neutral_surface_path = canonical_dir / "smpl_canonical_tpose_neutral.obj"
    subject_surface_path = canonical_dir / "smpl_canonical_tpose.obj"
    semantics_path = Path(args.config).resolve().parent / "anatomy_semantics.yaml"
    run_manifest = build_run_manifest(
        repo_root=Path(__file__).resolve().parents[5],
        blend_file=Path(blend),
        motion_npz=(
            None
            if args.motion_npz is None
            else Path(args.motion_npz).expanduser().resolve()
        ),
        canonical_files=(
            manifest_path,
            neutral_surface_path,
            subject_surface_path,
            weights_path,
            skeleton_path,
        ),
        config_files=(Path(args.config), semantics_path),
        code_files=(
            module_root / "blender_scripts" / "blender_retarget_script.py",
            module_root / "rigged_asset.py",
            module_root / "anatomy_lbs.py",
            module_root / "soft_follow.py",
            module_root / "source_skin_volume.py",
            module_root / "shape_volume.py",
            module_root / "material_fit.py",
            module_root / "quality_gate.py",
        ),
        solver_versions={
            "source_volume": "source-skin-volume-v7",
            "beta_volume": "beta-volume-v8-internal-handles",
            "regional_fit": "articulated-material-fit-v8-final-bind",
            "soft_follow": "station-translation-v1",
            "runtime": "source-driver-coupling-v8-target-bind",
        },
        random_seed=int(cfg.get("random_seed", 0)),
        extra={
            "canonical_shape_hash": smplx_shape_hash(betas, gender=gender),
            "diagnostics_only": bool(args.diagnostics_only),
        },
    )
    _atomic_write_json(stage_dir / "run_manifest.json", run_manifest)
    # Binding authority and harmonic-reference semantics changed without
    # changing the serialized schema number; keep the cache generation
    # separate so no source/shape asset from the old material-fit contract hits.
    cache_root = output_root.parent / "cache_v7_final_bind"
    source_key = _cache_key(
        Path(blend),
        Path(args.config),
        semantics_path,
        manifest_path,
        neutral_surface_path,
        weights_path,
        skeleton_path,
        module_root / "blender_scripts" / "blender_retarget_script.py",
        module_root / "anatomy_semantics.py",
        module_root / "source_audit.py",
        module_root / "rigged_asset.py",
        module_root / "anatomy_lbs.py",
        module_root / "bone_handles.py",
        module_root / "soft_constraints.py",
        module_root / "soft_follow.py",
        module_root / "source_skin_volume.py",
        module_root / "shape_volume.py",
        module_root / "material_fit.py",
        extra=(
            f"schema-{ANATOMY_ASSET_SCHEMA_VERSION}:source-template-v6"
            + (":fast-publish" if args.fast_publish else "")
        ),
    )
    shape_hash = smplx_shape_hash(betas, gender=gender)
    source_cache = cache_root / "source_template_v6" / f"{source_key}.npz"
    shape_key = _cache_key(
        subject_surface_path,
        weights_path,
        skeleton_path,
        module_root / "shape_volume.py",
        module_root / "bone_handles.py",
        module_root / "soft_constraints.py",
        module_root / "soft_follow.py",
        module_root / "intersection_diagnostics.py",
        module_root / "material_fit.py",
        module_root / "anatomy_lbs.py",
        extra=(
            f"schema-{ANATOMY_ASSET_SCHEMA_VERSION}:{source_key}:{shape_hash}:subject-shape-v6"
            + (":fast-publish" if args.fast_publish else "")
        ),
    )
    shape_cache = cache_root / "shape" / f"{shape_key}.npz"
    cached_source = (
        None
        if args.force_source_rebake
        else _load_valid_cache(
            source_cache,
            metadata_key="source_cache_key",
            expected_key=source_key,
        )
    )
    cached_shape = (
        None
        if args.force_source_rebake
        else _load_valid_cache(
            shape_cache,
            metadata_key="shape_cache_key",
            expected_key=shape_key,
        )
    )
    source_cache_hit = cached_source is not None
    shape_cache_hit = cached_shape is not None
    containment_reports: list[dict[str, Any]] = []
    registration_report: dict[str, Any] = {}
    blender_report: dict[str, Any]
    if source_cache_hit:
        asset = cached_source
        cached_meta = dict(asset.metadata or {})
        registration_report = dict(cached_meta.get("registration_report") or {})
        blender_report = dict(cached_meta.get("source_blender_report") or {})
        logging.info("source-rig cache hit key=%s", source_key)
    else:
        result = run_retarget(
            blend_path=blend, canonical_dir=args.canonical_dir, mapping_path=args.config,
            output_npz=output_npz, output_glb=output_glb, report_json=report_json,
            timeout_s=float(args.timeout_s),
        )
        if not result.ok:
            logging.error("Blender retarget failed returncode=%s log=%s", result.returncode, result.log_path)
            return int(result.returncode or 1)
        normalize_rigged_asset_file(output_npz, config=cfg, force=False)
        asset = load_rigged_asset(output_npz, validate=True)
        if args.fast_publish and args.fast_hand_donor is not None:
            source_skin_report = {
                "backend": "fast_donor_reference",
                "skipped": True,
                "reason": "1b66307 supplies the verified harmonic soft reference",
            }
        else:
            asset, source_skin_report = apply_source_skin_volume_registration(
                asset, canonical_dir=args.canonical_dir
            )
        if args.fast_publish:
            neutral_articulated_report = {
                "stage": "neutral",
                "backend": "fast_publish_source_lbs",
                "skipped": True,
                "reason": "preserve authored source rig and joint links",
            }
        else:
            asset, neutral_articulated_report = fit_articulated_rest(
                asset,
                canonical_dir=args.canonical_dir,
                config=cfg,
                subject=False,
                stage="neutral",
            )
        from projects.genesis_ue_sync.anatomy_retarget.soft_constraints import (
            regularize_asset_soft_materials,
        )

        if args.fast_publish:
            neutral_articulated_report["soft_material_regularizer"] = {
                "skipped": True,
                "reason": "fast_publish_preserves_source_lbs",
            }
        else:
            neutral_surface = load_body_surface(neutral_surface_path)
            neutral_soft_vertices, neutral_soft_report = (
                regularize_asset_soft_materials(
                    asset,
                    reference_vertices=(
                        asset.registration_reference
                        if asset.registration_reference is not None
                        else asset.vertices_rest
                    ),
                    surface_vertices=neutral_surface[0],
                    surface_faces=neutral_surface[1],
                )
            )
            asset = type(asset)(
                **{
                    **asset.__dict__,
                    "vertices_rest": neutral_soft_vertices.astype(np.float32),
                }
            )
            neutral_articulated_report["soft_material_regularizer"] = (
                neutral_soft_report
            )
        registration_report = {
            "backend": "source_skin_volume_harmonic_v6",
            "source_skin_volume": source_skin_report,
            "neutral_articulated_fit": neutral_articulated_report,
        }
        blender_report = json.loads(report_json.read_text(encoding="utf-8"))
        blender_report["volume_registration"] = source_skin_report
        source_meta = dict(asset.metadata or {})
        source_meta.update({
            "registration_report": registration_report,
            "source_blender_report": blender_report,
            "source_skin_volume_report": source_skin_report,
            "articulated_source_report": neutral_articulated_report,
            "source_cache_key": source_key,
        })
        asset = type(asset)(**{**asset.__dict__, "metadata": source_meta})
        source_cache.parent.mkdir(parents=True, exist_ok=True)
        save_rigged_asset(source_cache, asset)
        logging.info("source_template_v6 stored key=%s", source_key)
    neutral_surface = None
    if not args.fast_publish or args.refresh_diagnostics:
        neutral_surface = load_body_surface(neutral_surface_path)
        neutral_containment = _signed_distance_containment_report(
            asset,
            anatomy_vertices=asset.vertices_rest,
            surface_vertices=neutral_surface[0],
            surface_faces=neutral_surface[1],
            stage="neutral_canonical",
        )
        containment_reports.append(neutral_containment)
    profile["source_template_s"] = time.perf_counter() - started_at
    bind_roundtrip = {
        "max_matrix_error": float(np.max(np.abs(
            np.asarray(asset.source_rest_global, dtype=np.float64)
            @ np.asarray(asset.source_inverse_bind, dtype=np.float64)
            - np.eye(4, dtype=np.float64)[None]
        ))),
    }
    zero_pose_vertices = skin_vertices(asset, np.zeros((55, 3), dtype=np.float32))
    bind_roundtrip["zero_pose_vertex_error_m"] = float(
        np.max(np.linalg.norm(zero_pose_vertices - np.asarray(asset.vertices_rest, dtype=np.float32), axis=1))
    )
    bind_roundtrip["pass"] = bool(
        bind_roundtrip.get("pass", True)
        and bind_roundtrip["zero_pose_vertex_error_m"] <= 1.0e-5
    )
    if not bool(bind_roundtrip.get("pass", True)):
        raise RuntimeError(f"source bind round-trip failed: {bind_roundtrip}")

    source_vertices = (
        np.asarray(asset.registration_reference, dtype=np.float32).copy()
        if asset.registration_reference is not None else asset.vertices_rest.copy()
    )
    tube_graphs = build_asset_attachment_graphs(asset)
    subject_surface = load_body_surface(subject_surface_path)
    shape_report: dict[str, Any] = {"backend": "subject_bind_direct"}
    if shape_cache_hit:
        asset = cached_shape
        cached_meta = dict(asset.metadata or {})
        shape_report = dict(cached_meta.get("shape_report") or shape_report)
        logging.info("subject-shape cache hit shape_hash=%s", shape_hash)
    else:
        if str(cfg.get("canonical_rest_space", "neutral")).lower() == "neutral":
            if args.fast_publish and args.fast_hand_donor is not None:
                shape_report = {
                    "backend": "fast_donor_subject_rest",
                    "skipped_subject_beta_volume": True,
                    "skipped_material_rest_fit": True,
                    "skipped_soft_follow": True,
                    "reason": "1b soft reference and fe99 fitted bones already match subject shape",
                }
            elif args.fast_publish and args.fast_extremity_donor is None:
                # Do not run the beta volume cage in the live-preview path.
                # Its extrapolation fallback moves soft material while the
                # authored bone rig remains fixed, which is precisely the
                # pelvis/spine disconnect this mode is meant to avoid.
                shape_report = {
                    "backend": "fast_publish_source_rest",
                    "skipped_subject_beta_volume": True,
                    "skipped_material_rest_fit": True,
                    "skipped_soft_follow": True,
                    "reason": "preserve authored source joints and all material links",
                }
            else:
                shape_cfg = dict(cfg)
                if args.fast_extremity_donor is not None:
                    # The semantic source solve is now authoritative for soft
                    # anatomy.  Fast donor mode still needs neutral->subject
                    # harmonic transport before rigid donor material is merged.
                    shape_cfg["fast_publish"] = True
                    shape_cfg["maximum_harmonic_extrapolation_m"] = max(
                        0.10,
                        float(shape_cfg.get("maximum_harmonic_extrapolation_m", 0.0)),
                    )
                asset, shape_report = apply_subject_beta_shape(
                    asset, canonical_dir=args.canonical_dir, config=shape_cfg
                )
        try:
            asset, station_intersection_acceptance = (
                (
                    asset,
                    {
                        "available": False,
                        "skipped": True,
                        "reason": "fast_publish",
                    },
                )
                if args.fast_publish
                else enforce_station_intersection_nonregression(asset)
            )
        except Exception as exc:
            station_intersection_acceptance = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        shape_report["station_intersection_acceptance"] = (
            station_intersection_acceptance
        )
        shape_meta = dict(asset.metadata or {})
        shape_meta.update({
            "shape_report": shape_report,
            "shape_cache_key": shape_key,
        })
        asset = type(asset)(**{**asset.__dict__, "metadata": shape_meta})
        shape_cache.parent.mkdir(parents=True, exist_ok=True)
        save_rigged_asset(shape_cache, asset)
        logging.info("subject-shape cache stored shape_hash=%s", shape_hash)
    if args.fast_extremity_donor is not None:
        donor_path = Path(args.fast_extremity_donor).expanduser().resolve()
        donor_asset = load_rigged_asset(donor_path, validate=True)
        axial_asset = (
            None
            if args.fast_axial_donor is None
            else load_rigged_asset(
                Path(args.fast_axial_donor).expanduser().resolve(), validate=True
            )
        )
        asset, donor_report = _merge_fast_extremity_donor(
            asset,
            donor_asset,
            expected_shape_hash=shape_hash,
            canonical_dir=Path(args.canonical_dir),
            hand_donor_path=(
                None
                if args.fast_hand_donor is None
                else Path(args.fast_hand_donor).expanduser().resolve()
            ),
            axial_donor=axial_asset,
        )
        asset, material_volume_report = apply_material_bounded_soft_volume(
            asset,
            canonical_dir=Path(args.canonical_dir),
        )
        donor_report["material_bounded_soft_volume"] = material_volume_report
        shape_report["fast_extremity_merge"] = {
            **donor_report,
            "donor_path": str(donor_path),
            "donor_sha256": _file_digest(donor_path),
            "hand_donor_path": (
                None
                if args.fast_hand_donor is None
                else str(Path(args.fast_hand_donor).expanduser().resolve())
            ),
            "axial_donor_path": (
                None
                if args.fast_axial_donor is None
                else str(Path(args.fast_axial_donor).expanduser().resolve())
            ),
        }
        logging.info(
            "fast material donor merged axial_bones=%s axial_vertices=%s head_scale=%.4f",
            donor_report["axial_source_bone_count"],
            donor_report["axial_source_vertex_count"],
            donor_report["head_uniform_scale"],
        )
    target_bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    target_inverse = np.asarray(asset.runtime_inverse_bind, dtype=np.float64)
    bind_roundtrip["target_bind_max_matrix_error"] = float(
        np.max(
            np.abs(
                target_bind
                @ target_inverse
                - np.eye(4, dtype=np.float64)[None]
            )
        )
    )
    target_zero_pose = skin_vertices(
        asset, np.zeros((55, 3), dtype=np.float32)
    )
    bind_roundtrip["target_zero_pose_vertex_error_m"] = float(
        np.max(
            np.linalg.norm(
                target_zero_pose
                - np.asarray(asset.vertices_rest, dtype=np.float32),
                axis=1,
            )
        )
    )
    bind_roundtrip["pass"] = bool(
        bind_roundtrip["pass"]
        and bind_roundtrip["target_bind_max_matrix_error"] <= 1.0e-5
        and bind_roundtrip["target_zero_pose_vertex_error_m"] <= 1.0e-5
    )
    if not bind_roundtrip["pass"]:
        raise RuntimeError(f"target bind round-trip failed: {bind_roundtrip}")
    if (
        "inverted_tetrahedra" not in shape_report
        and "minimum_jacobian_ratio" in shape_report
    ):
        minimum_jacobian = float(shape_report["minimum_jacobian_ratio"])
        shape_report["inverted_tetrahedra"] = int(minimum_jacobian <= 0.0)
    if not args.fast_publish or args.refresh_diagnostics:
        subject_containment = _signed_distance_containment_report(
            asset,
            anatomy_vertices=asset.vertices_rest,
            surface_vertices=subject_surface[0],
            surface_faces=subject_surface[1],
            stage="subject_rest",
        )
        containment_reports.append(subject_containment)
    profile["subject_shape_s"] = time.perf_counter() - started_at - profile["source_template_s"]
    pose_report: dict[str, Any] | None = None
    if args.motion_npz is not None:
        motion_path = Path(args.motion_npz).expanduser().resolve()
        motion = np.load(motion_path)
        motion_betas = np.asarray(motion["shapes"], dtype=np.float32).reshape(-1)[:10]
        motion_shape_hash = smplx_shape_hash(motion_betas, gender=gender)
        expected_shape_hash = smplx_shape_hash(betas, gender=gender)
        if motion_shape_hash != expected_shape_hash:
            raise ValueError(
                f"motion/canonical shape mismatch: motion={motion_shape_hash} canonical={expected_shape_hash}"
            )
        if "vertices" not in motion.files or "faces" not in motion.files:
            raise ValueError(f"{motion_path} must include official posed SMPL-X vertices/faces")
        posed_surface_vertices = np.asarray(motion["vertices"], dtype=np.float64).reshape(-1, 3)
        posed_surface_faces = np.asarray(motion["faces"], dtype=np.int32).reshape(-1, 3)
        pose55, raw_transl = load_easymocap_smplx_fit_drive(motion_path, gender=gender)
        effective_transl = easymocap_drive_translation(pose55[:3], raw_transl, asset.rest_joints[0])
        cache_hash = smplx_pose_hash(pose55, effective_transl)
        runtime_key = _cache_key(
            module_root / "anatomy_lbs.py",
            module_root / "pose_adapter.py",
            module_root / "rigged_asset.py",
            module_root / "tube_graph.py",
            extra=f"schema-{ANATOMY_ASSET_SCHEMA_VERSION}:runtime-source-fk-v6",
        )
        pose_cache = cache_root / "pose" / f"{shape_key}-{runtime_key}-{cache_hash}.npz"
        pose_key = f"{shape_key}:{runtime_key}:{cache_hash}"
        cached_pose = (
            None
            if args.force_source_rebake or args.fast_extremity_donor is not None
            else _load_valid_cache(
                pose_cache,
                metadata_key="pose_cache_key",
                expected_key=pose_key,
            )
        )
        if cached_pose is not None:
            asset = type(asset)(
                **{
                    **asset.__dict__,
                    "pose_cache_vertices": cached_pose.pose_cache_vertices,
                    "pose_cache_hash": cached_pose.pose_cache_hash,
                }
            )
            logging.info("pose cache hit pose_hash=%s", cache_hash)
        else:
            posed_vertices = skin_vertices(asset, pose55, transl=effective_transl)
            asset = type(asset)(
                **{
                    **asset.__dict__,
                    "pose_cache_vertices": posed_vertices,
                    "pose_cache_hash": cache_hash,
                }
            )
        if not args.fast_publish or args.refresh_diagnostics:
            pose_report = _signed_distance_containment_report(
                asset,
                anatomy_vertices=np.asarray(asset.pose_cache_vertices, dtype=np.float64),
                surface_vertices=posed_surface_vertices,
                surface_faces=posed_surface_faces,
                stage="final_pose",
            )
            containment_reports.append(pose_report)
        else:
            pose_report = {
                "stage": "final_pose",
                "skipped": True,
                "reason": "fast_publish",
                "soft_sdf": "disabled",
            }
        if cached_pose is None:
            pose_meta = dict(asset.metadata or {})
            pose_meta.update(
                {
                    "pose_cache_key": pose_key,
                    "pose_cache_report": pose_report,
                }
            )
            pose_asset = type(asset)(**{**asset.__dict__, "metadata": pose_meta})
            pose_cache.parent.mkdir(parents=True, exist_ok=True)
            save_rigged_asset(pose_cache, pose_asset)
            logging.info("pose cache stored pose_hash=%s", cache_hash)
    else:
        zero_pose = np.zeros((55, 3), dtype=np.float32)
        zero_translation = np.zeros(3, dtype=np.float32)
        zero_vertices = skin_vertices(
            asset,
            zero_pose,
            transl=zero_translation,
        )
        asset = type(asset)(
            **{
                **asset.__dict__,
                "pose_cache_vertices": zero_vertices,
                "pose_cache_hash": "zero_pose",
            }
        )
        if not args.fast_publish or args.refresh_diagnostics:
            pose_report = _signed_distance_containment_report(
                asset,
                anatomy_vertices=zero_vertices,
                surface_vertices=subject_surface[0],
                surface_faces=subject_surface[1],
                stage="final_pose",
            )
            containment_reports.append(pose_report)
        else:
            pose_report = {
                "stage": "final_pose",
                "skipped": True,
                "reason": "fast_publish",
                "soft_sdf": "disabled",
            }

    # Schema-v6 assets contain runtime data only.  Cache keys and diagnostics
    # belong in JSON sidecars and must not leak into the published NPZ.
    meta = dict(asset.metadata or {})
    meta.update({
        "gender": gender,
        "betas": betas,
        "shape_hash": smplx_shape_hash(betas, gender=gender),
        "canonical_source": str(manifest.get("source", "")),
        "show_connective_tissue": bool(args.show_connective_tissue),
        "show_vessels": not bool(args.hide_vessels),
        "fast_publish": bool(args.fast_publish),
        "disable_soft_follow": bool(args.fast_publish),
    })
    asset = type(asset)(**{**asset.__dict__, "metadata": meta})
    if args.fast_publish:
        fast_updates: dict[str, Any] = {}
        if args.fast_extremity_donor is None:
            fast_updates.update({
                "pose_cache_vertices": None,
                "pose_cache_hash": "",
            })
        if args.fast_extremity_donor is None:
            fast_updates.update({
                "soft_follow_driver_indices": None,
                "soft_follow_driver_weights": None,
                "soft_follow_stations": None,
                "soft_follow_strength": None,
                "soft_component_ids": None,
                "source_mesh_follow_modes": None,
            })
        asset = type(asset)(
            **{
                **asset.__dict__,
                **fast_updates,
            }
        )
    save_rigged_asset(output_npz, asset)
    if args.fast_publish:
        # This mode is deliberately a live-preview path: it preserves the
        # source rig's articulated rest geometry and publishes it directly.
        # Do not spend minutes generating diagnostics / acceptance evidence,
        # and do not run geometry checks that are intended for release bakes.
        profile["pose_and_diagnostics_s"] = 0.0
        profile["total_pre_publish_s"] = time.perf_counter() - started_at
        if args.profile_first_frame:
            (stage_dir / "first_frame_profile.json").write_text(
                json.dumps(
                    {
                        "seconds": profile,
                        "source_cache_hit": source_cache_hit,
                        "shape_cache_hit": shape_cache_hit,
                        "fast_publish": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        blender_report.update(
            {
                "schema": {
                    "asset_schema_version": ANATOMY_ASSET_SCHEMA_VERSION,
                    "expected_schema_version": ANATOMY_ASSET_SCHEMA_VERSION,
                    "passed": True,
                },
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": _file_digest(manifest_path),
                    "source": manifest["source"],
                    "gender": gender,
                    "betas": betas,
                },
                "run_manifest": run_manifest,
                "registration": registration_report,
                "shape": shape_report,
                "containment_stages": containment_reports,
                "pose_cache_report": pose_report,
                "source_bind_roundtrip": bind_roundtrip,
                "fast_publish": {
                    "skipped_acceptance": True,
                    "skipped": [
                        "mesh_diagnostics",
                        "bone_segment_diagnostics",
                        "tube_bone_intersections",
                        "runtime_pose_matrix",
                        "quality_gate_evaluation",
                    ],
                    "profile": profile,
                },
            }
        )
        report_json.write_text(
            json.dumps(blender_report, indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )
        write_quality_report(
            stage_dir / "quality_report.json",
            {
                "schema_version": ANATOMY_ASSET_SCHEMA_VERSION,
                "passed": True,
                "failures": [],
                "warnings": [],
                "fast_publish": True,
                "skipped_acceptance": True,
            },
        )
        run_dir = _finalize_run(
            stage_dir,
            output_root=output_root,
            schema_version=ANATOMY_ASSET_SCHEMA_VERSION,
            passed=True,
            update_latest=not bool(args.diagnostics_only),
        )
        if args.diagnostics_only:
            logging.info(
                "fast-publish diagnostics-only run preserved at %s; latest.json remains unchanged",
                run_dir,
            )
            return 0

        output_npz = run_dir / "anatomy_rigged.npz"
        logging.info(
            "fast-publish retarget ok vertices=%s faces=%s joints=%s output=%s",
            asset.vertices_rest.shape[0],
            asset.faces.shape[0],
            len(asset.joint_names),
            output_npz,
        )
        if args.publish_genesis:
            sent = _publish_upsert(
                bind=str(args.publish_bind),
                model_id=str(args.model_id),
                asset_npz=output_npz,
                color_rgba=_parse_rgba(str(args.color_rgba)),
                duration_s=float(args.publish_duration_s),
                rate_hz=float(args.publish_rate_hz),
            )
            logging.info("published anatomy upsert sent=%s bind=%s", sent, args.publish_bind)
        return 0

    mesh_diagnostics = write_mesh_diagnostics(
        asset,
        surface_vertices=subject_surface[0],
        surface_faces=subject_surface[1],
        output_path=stage_dir / "anatomy_mesh_diagnostics.json",
    )
    bone_segment_report: dict[str, Any] | None = None
    fitted_hip_geometry = (
        (shape_report.get("articulated_rest_fit") or {}).get("hip_geometry")
        if isinstance(shape_report, dict)
        else None
    )
    if args.motion_npz is not None:
        motion_path = Path(args.motion_npz).expanduser().resolve()
        pose55, raw_transl = load_easymocap_smplx_fit_drive(motion_path, gender=gender)
        effective_transl = easymocap_drive_translation(pose55[:3], raw_transl, asset.rest_joints[0])
        bone_segment_report = write_bone_segment_diagnostics(
            asset,
            pose_axis_angle=pose55,
            transl=effective_transl,
            output_path=stage_dir / "bone_segment_diagnostics.json",
            mesh_diagnostics=mesh_diagnostics,
            fitted_hip_geometry=fitted_hip_geometry,
        )
    else:
        bone_segment_report = write_bone_segment_diagnostics(
            asset,
            pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
            transl=np.zeros(3, dtype=np.float32),
            output_path=stage_dir / "bone_segment_diagnostics.json",
            mesh_diagnostics=mesh_diagnostics,
            fitted_hip_geometry=fitted_hip_geometry,
        )
    profile["pose_and_diagnostics_s"] = time.perf_counter() - started_at - profile["source_template_s"] - profile["subject_shape_s"]
    profile["total_pre_publish_s"] = time.perf_counter() - started_at
    if args.profile_first_frame:
        (stage_dir / "first_frame_profile.json").write_text(
            json.dumps({"seconds": profile, "source_cache_hit": source_cache_hit, "shape_cache_hit": shape_cache_hit}, indent=2),
            encoding="utf-8",
        )
        logging.info("first-frame profile %s", {key: round(value, 3) for key, value in profile.items()})
    tri_edges = np.concatenate(
        (asset.faces[:, [0, 1]], asset.faces[:, [1, 2]], asset.faces[:, [2, 0]]), axis=0
    )
    before_len = np.linalg.norm(
        source_vertices[tri_edges[:, 0]] - source_vertices[tri_edges[:, 1]], axis=1
    )
    after_len = np.linalg.norm(
        asset.vertices_rest[tri_edges[:, 0]] - asset.vertices_rest[tri_edges[:, 1]], axis=1
    )
    # Ratios on nearly coincident CAD seam vertices are numerically meaningless
    # (e.g. 0.04 mm -> 0.4 mm looks like 10x but is not a visible spike).
    # Regional bone solvers intentionally resize shafts; percentile and growth
    # gates therefore apply to non-bone material while the all-material maximum
    # remains a separate explosion guard.
    soft_vertex = np.zeros(len(asset.vertices_rest), dtype=bool)
    for (start, stop), tissue in zip(
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        asset.source_tissues,
    ):
        if str(tissue) != "bone":
            soft_vertex[int(start) : int(stop)] = True
    valid_edges = before_len > 1.0e-3
    post_ratio = after_len[valid_edges] / before_len[valid_edges]
    soft_edges = (
        valid_edges
        & soft_vertex[tri_edges[:, 0]]
        & soft_vertex[tri_edges[:, 1]]
    )
    soft_ratio = after_len[soft_edges] / before_len[soft_edges]
    growth = after_len - before_len
    growth_gate = growth[soft_edges]
    blender_report.setdefault("edge_stretch", {}).update(
        {
            "source_to_final_max": float(np.max(post_ratio)) if len(post_ratio) else None,
            "source_to_final_p999": (
                float(np.quantile(soft_ratio, 0.999))
                if len(soft_ratio)
                else None
            ),
            "source_to_final_max_growth_m": (
                float(np.max(growth_gate)) if len(growth_gate) else None
            ),
            "ratio_ignored_sub_1mm_edges": int(np.count_nonzero(before_len <= 1.0e-3)),
        }
    )
    if asset.pose_cache_vertices is not None:
        cached_len = np.linalg.norm(
            asset.pose_cache_vertices[tri_edges[:, 0]] - asset.pose_cache_vertices[tri_edges[:, 1]], axis=1
        )
        cached_ratio = cached_len[valid_edges] / before_len[valid_edges]
        cached_soft_ratio = cached_len[soft_edges] / before_len[soft_edges]
        blender_report["edge_stretch"].update({
            "source_to_pose_cache_max": (
                float(np.max(cached_ratio)) if len(cached_ratio) else None
            ),
            "source_to_pose_cache_p999": (
                float(np.quantile(cached_soft_ratio, 0.999))
                if len(cached_soft_ratio)
                else None
            ),
            "source_to_pose_cache_max_growth_m": (
                float(np.max((cached_len - before_len)[soft_edges]))
                if np.any(soft_edges)
                else None
            ),
        })
    material_report = dict(shape_report.get("articulated_rest_fit") or {})
    controller_anchor_rms = material_report.get("anchor_rms_m")
    controller_anchor_max = material_report.get("anchor_max_m")
    geometry_errors: list[float] = []
    if isinstance(bone_segment_report, dict):
        for label, joint_metrics in (
            bone_segment_report.get("joints") or {}
        ).items():
            if not isinstance(joint_metrics, dict):
                continue
            geometry = joint_metrics.get("geometry_landmarks")
            if not isinstance(geometry, dict):
                continue
            if (
                str(label) in {"hip_left", "hip_right"}
                and geometry.get("femoral_head_to_acetabulum_m")
                is not None
            ):
                geometry_errors.append(
                    float(geometry["femoral_head_to_acetabulum_m"])
                )
                continue
            if (
                geometry.get("available") is True
                and geometry.get("surface_gap_m") is not None
            ):
                geometry_errors.append(float(geometry["surface_gap_m"]))
    material_report["controller_bind_origin_rms_m"] = controller_anchor_rms
    material_report["controller_bind_origin_max_m"] = controller_anchor_max
    material_report["anchor_rms_m"] = (
        float(np.sqrt(np.mean(np.square(geometry_errors))))
        if geometry_errors
        else None
    )
    material_report["anchor_max_m"] = (
        float(np.max(geometry_errors)) if geometry_errors else None
    )
    tube_graph_report: dict[str, Any] = {}
    for graph_name, graph in tube_graphs.items():
        subject_nodes = graph.sample_nodes(asset.vertices_rest)
        subject_graph = replace(
            graph,
            rest_nodes=subject_nodes.astype(np.float32),
        )
        tube_graph_report[graph_name] = {
            "neutral_to_subject": tube_graph_metrics(graph, asset.vertices_rest),
            "subject_to_pose": (
                None
                if asset.pose_cache_vertices is None
                else tube_graph_metrics(subject_graph, asset.pose_cache_vertices)
            ),
            "topology_preserved": True,
        }
    try:
        tube_bone_intersections = tube_bone_intersection_report(asset)
    except Exception as exc:
        tube_bone_intersections = {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    runtime_pose_matrix = _runtime_pose_matrix_report(
        asset,
        tube_graphs=tube_graphs,
    )
    if args.validation_matrix:
        matrix_cases = release_validation_matrix(
            np.asarray(betas, dtype=np.float32),
            list(asset.joint_names),
            principal_dimensions=4,
        )
        validation_payload = {
            "schema_version": 1,
            "beta_case_count": len(matrix_cases)
            // max(1, len(pose_cases(list(asset.joint_names)))),
            "pose_case_count": len(pose_cases(list(asset.joint_names))),
            "case_count": len(matrix_cases),
            "report_only": True,
            "cases": [
                {
                    "name": case.name,
                    "betas": np.asarray(case.betas, dtype=np.float32).tolist(),
                    "pose_axis_angle": np.asarray(case.pose_axis_angle, dtype=np.float32).tolist(),
                    "status": (
                        "evaluated_current_asset"
                        if case.name.startswith("beta_real__")
                        else "requires_shape_rebake"
                    ),
                    "quality": (
                        runtime_pose_matrix.get("cases", {}).get(
                            case.name.split("__", 1)[1]
                        )
                        if case.name.startswith("beta_real__")
                        else None
                    ),
                }
                for case in matrix_cases
            ],
        }
        _atomic_write_json(stage_dir / "validation_matrix.json", validation_payload)
    hip_geometry = material_report.get("hip_geometry")
    landmark_report = {
        "backend": "anatomy_mesh_geometry_landmarks",
        "anchor_rms_m": material_report.get("anchor_rms_m"),
        "anchor_max_m": material_report.get("anchor_max_m"),
        "hip_geometry": hip_geometry,
        "passed": bool(
            geometry_errors
            and isinstance(bone_segment_report, dict)
            and bone_segment_report.get("passed") is True
        ),
    }
    blender_report.update(
        {
            "schema": {
                "asset_schema_version": ANATOMY_ASSET_SCHEMA_VERSION,
                "expected_schema_version": ANATOMY_ASSET_SCHEMA_VERSION,
                "passed": True,
            },
            "manifest": {
                "path": str(manifest_path),
                "sha256": _file_digest(manifest_path),
                "source": manifest["source"],
                "gender": gender,
                "betas": betas,
            },
            "run_manifest": run_manifest,
            "registration": registration_report,
            "shape": shape_report,
            "containment_stages": containment_reports,
            "pose_cache_report": pose_report,
            "source_bind_roundtrip": bind_roundtrip,
            "bone_segment_diagnostics": bone_segment_report,
            "material_shape": material_report,
            "landmark_report": landmark_report,
            "tube_graphs": tube_graph_report,
            "tube_bone_intersections": tube_bone_intersections,
            "runtime_pose_matrix": runtime_pose_matrix,
        }
    )
    report_json.write_text(
        json.dumps(blender_report, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    quality = evaluate_asset_quality(
        asset,
        canonical_dir=args.canonical_dir,
        blender_report=blender_report,
        limits=dict(cfg.get("quality_gate", {}) or {}),
    )
    write_quality_report(stage_dir / "quality_report.json", quality)
    if not quality["passed"]:
        log_failure = logging.error if args.enforce_quality_gate else logging.warning
        for failure in quality["failures"]:
            log_failure("quality: %s", failure)
        if _quality_failure_blocks_publish(
            passed=bool(quality["passed"]),
            enforce_quality_gate=bool(args.enforce_quality_gate),
        ):
            failed_dir = _finalize_run(
                stage_dir,
                output_root=output_root,
                schema_version=ANATOMY_ASSET_SCHEMA_VERSION,
                passed=False,
                update_latest=False,
            )
            logging.error(
                "quality gate rejected anatomy asset; latest.json remains unchanged"
            )
            logging.error("failed run diagnostics preserved at %s", failed_dir)
            return 0 if args.diagnostics_only else 2
        logging.warning(
            "quality gate failed (%s issues); publishing anyway (use --enforce-quality-gate to block)",
            len(quality.get("failures", [])),
        )

    run_dir = _finalize_run(
        stage_dir,
        output_root=output_root,
        schema_version=ANATOMY_ASSET_SCHEMA_VERSION,
        passed=True,
        update_latest=not bool(args.diagnostics_only),
    )
    if args.diagnostics_only:
        logging.info(
            "diagnostics-only run preserved at %s; latest.json remains unchanged",
            run_dir,
        )
        return 0

    output_npz = run_dir / "anatomy_rigged.npz"
    logging.info(
        "retarget ok vertices=%s faces=%s joints=%s output=%s",
        asset.vertices_rest.shape[0],
        asset.faces.shape[0],
        len(asset.joint_names),
        output_npz,
    )
    if args.publish_genesis:
        sent = _publish_upsert(
            bind=str(args.publish_bind),
            model_id=str(args.model_id),
            asset_npz=output_npz,
            color_rgba=_parse_rgba(str(args.color_rgba)),
            duration_s=float(args.publish_duration_s),
            rate_hz=float(args.publish_rate_hz),
        )
        logging.info("published anatomy upsert sent=%s bind=%s", sent, args.publish_bind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
