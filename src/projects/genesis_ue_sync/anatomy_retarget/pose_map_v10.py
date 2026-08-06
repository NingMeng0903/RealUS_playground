"""Joint-anchored FK pose map (V10) — SKEL-style parent-local composition.

Replaces the V6/V7 right-multiply ``G' = G_src @ inv(B_src) @ B_tgt`` which
cancels ``B_tgt`` and rotates every bone about the *source* Blender pivot.
V10 rebuilds FK on the *target* bind so the child origin stays at the target
anatomical joint by construction:

    L_src_pose[i] = inv(G_src[p]) @ G_src[i]
    D[i]          = inv(L_src_rest[i]) @ L_src_pose[i]
    L_tgt_pose[i] = L_tgt_rest[i] @ SE3(Q[i], s_seg[i] * t_res[i])
    G_tgt[i]      = G_tgt[p] @ L_tgt_pose[i]

Hand/foot terminal subtrees keep the copy-142 contract via per-subtree rigid
rebase from the wrist/ankle root after the root itself is joint-anchored.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from .anatomical_calibration_v1 import AnatomicalCalibrationV1
from .anatomy_lbs import source_bone_posed_global
from .chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    _global_to_local,
    _weighted_rest_correction,
)
from .pose_map_v1 import PoseMapV1, build_pose_map_v1
from .whole_chain_rest_fit_v1 import _descendants


POSE_MAP_V10_COMPOSITION = "joint_anchored_fk_v10"
HAND_ROOTS = ("Wrist_Rotate_L", "Wrist_Rotate_R1")
FOOT_ROOTS = ("Ankle_Rot_L", "Ankle_Rot_R")


def _se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _frozen_terminal_members(
    bone_names: Sequence[str],
    parents: np.ndarray,
) -> dict[str, set[int]]:
    """Map terminal root name -> descendant indices *excluding* the root."""

    names = [str(name) for name in bone_names]
    parent_ids = np.asarray(parents, dtype=np.int64)
    members: dict[str, set[int]] = {}
    for root_name in (*HAND_ROOTS, *FOOT_ROOTS):
        if root_name not in names:
            raise ValueError(f"terminal root missing: {root_name}")
        root = names.index(root_name)
        descendants = _descendants(parent_ids, root)
        members[root_name] = {int(i) for i in descendants if int(i) != root}
    return members


def apply_pose_map_global_v10(
    pose_map: PoseMapV1,
    *,
    source_asset: Any,
    pose_axis_angle: Any,
    segment_scales: np.ndarray | Mapping[str, float] | None = None,
) -> np.ndarray:
    """Joint-anchored FK: parent-local pose delta applied on the target bind.

    ``segment_scales`` optionally scales residual joint translations (tibia
    glide, patella RBF, …) after N3 segment similarity.  Default is 1.0.
    """

    pose_map.validate()
    names = [str(name) for name in pose_map.bone_names.tolist()]
    parents = np.asarray(pose_map.bone_parents, dtype=np.int64)
    source_bind_local = np.asarray(pose_map.source_bind_local, dtype=np.float64)
    target_bind_local = np.asarray(pose_map.target_bind_local, dtype=np.float64)
    source_global = np.asarray(
        source_bone_posed_global(source_asset, pose_axis_angle), dtype=np.float64
    )
    source_posed_local = _global_to_local(source_global, parents)

    if segment_scales is None:
        scales = np.ones(len(names), dtype=np.float64)
    elif isinstance(segment_scales, Mapping):
        scales = np.ones(len(names), dtype=np.float64)
        for name, value in segment_scales.items():
            scales[names.index(str(name))] = float(value)
    else:
        scales = np.asarray(segment_scales, dtype=np.float64).reshape(len(names))

    frozen = _frozen_terminal_members(names, parents)
    # Include wrist/ankle roots: LBS hand/foot verts carry weight on the root,
    # so freezing only finger/toe children leaves the palm/tarsals pulled by a
    # joint-anchored wrist (~37 mm left-hand translation on pose_213328).
    frozen_all: set[int] = set()
    frozen_roots: set[int] = set()
    for root_name, members in frozen.items():
        root = names.index(root_name)
        frozen_roots.add(root)
        frozen_all.add(root)
        frozen_all |= members

    target_global = np.empty((len(names), 4, 4), dtype=np.float64)
    for bone, parent in enumerate(parents.tolist()):
        if bone in frozen_all:
            # Filled in the identity terminal pass below.
            target_global[bone] = np.eye(4, dtype=np.float64)
            continue
        delta = np.linalg.inv(source_bind_local[bone]) @ source_posed_local[bone]
        rotation = delta[:3, :3]
        translation = scales[bone] * delta[:3, 3]
        local_pose = target_bind_local[bone] @ _se3(rotation, translation)
        if parent < 0:
            target_global[bone] = local_pose
        else:
            target_global[bone] = target_global[int(parent)] @ local_pose

    # Hybrid terminal policy: keep copy-142 hand/foot (including wrist/ankle
    # roots) at the absolute 142 posed globals — same as V7 right-multiply
    # when B_tgt == B_src on terminals.
    for bone in sorted(frozen_all):
        target_global[bone] = source_global[bone]

    return target_global


def pose_whole_chain_vertices_v10(
    value: ChainRestFitSubjectV1,
    pose_map: PoseMapV1,
    *,
    source_asset: Any,
    pose_axis_angle: Any,
    segment_scales: np.ndarray | Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose the whole-chain subject with joint-anchored FK."""

    rest = np.asarray(value.vertices_final, dtype=np.float64).copy()
    tissue = np.asarray(source_asset.source_tissues)
    ranges = np.asarray(source_asset.source_vertex_ranges, dtype=np.int64)
    bone_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for label, (start, stop) in zip(tissue.tolist(), ranges.tolist())
            if str(label).strip().lower() == "bone"
        ]
    )
    tube_ids = np.concatenate(
        [
            np.arange(int(start), int(stop), dtype=np.int64)
            for label, (start, stop) in zip(tissue.tolist(), ranges.tolist())
            if str(label).strip().lower() in {"vessel", "nerve"}
        ]
    )
    tube_mask = np.zeros(len(rest), dtype=bool)
    tube_mask[tube_ids] = True

    posed_global = apply_pose_map_global_v10(
        pose_map,
        source_asset=source_asset,
        pose_axis_angle=pose_axis_angle,
        segment_scales=segment_scales,
    )
    transforms = posed_global @ pose_map.target_inverse_bind
    posed_all = _weighted_rest_correction(
        rest,
        source_asset.driver_indices,
        source_asset.driver_weights,
        transforms,
    )
    # Soft tissue / organs that were not rest-moved keep the 142 posed path
    # (same contract as pose_map_v1.pose_whole_chain_vertices).
    source_global = source_bone_posed_global(source_asset, pose_axis_angle)
    source_transforms = source_global @ np.linalg.inv(pose_map.source_bind_global)
    source_posed = _weighted_rest_correction(
        value.vertices_prefit,
        source_asset.driver_indices,
        source_asset.driver_weights,
        source_transforms,
    )
    posed = source_posed
    posed[bone_ids] = posed_all[bone_ids]
    posed[tube_mask] = posed_all[tube_mask]
    return posed.astype(np.float32), posed_global


def build_pose_map_v10(
    value: ChainRestFitSubjectV1,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    oracle_path: Any,
    source_operator_digest: str,
) -> PoseMapV1:
    """Reuse PoseMapV1 bind tables; runtime composition is joint-anchored."""

    return build_pose_map_v1(
        value,
        asset=asset,
        calibration=calibration,
        oracle_path=oracle_path,
        source_operator_digest=source_operator_digest,
    )


def check_pose_map_v10(
    pose_map: PoseMapV1,
    value: ChainRestFitSubjectV1,
    *,
    source_asset: Any,
) -> dict[str, Any]:
    """T-pose must be bit-identical to the target rest (construction identity)."""

    started = time.perf_counter()
    pose_map.validate()
    zero = np.zeros((55, 3), dtype=np.float32)
    zero_vertices, zero_global = pose_whole_chain_vertices_v10(
        value,
        pose_map,
        source_asset=source_asset,
        pose_axis_angle=zero,
    )
    vertex_error = np.linalg.norm(
        zero_vertices[value.moved_vertex_ids]
        - np.asarray(value.vertices_final)[value.moved_vertex_ids],
        axis=1,
    )
    matrix_error = float(np.max(np.abs(zero_global - pose_map.target_bind_global)))
    # Stronger than V1: T-pose globals must match target bind exactly.
    matrix_ok = matrix_error <= 2.0e-6
    vertex_rms = float(np.sqrt(np.mean(vertex_error**2))) if len(vertex_error) else 0.0
    vertex_max = float(np.max(vertex_error)) if len(vertex_error) else 0.0
    vertex_ok = vertex_rms <= 1.0e-6 and vertex_max <= 1.0e-5
    passed = bool(matrix_ok and vertex_ok)
    return {
        "schema_version": 10,
        "artifact_kind": "PoseMapCheckV10",
        "passed": passed,
        "pose_composition": POSE_MAP_V10_COMPOSITION,
        "parent_local_mapping_only": True,
        "source_pose_authority": "frozen_142_source_bone_posed_global",
        "terminal_policy": "identity_142_hand_foot",
        "zero_pose_matrix_max_abs": matrix_error,
        "zero_pose_vertex_rms_m": vertex_rms,
        "zero_pose_vertex_max_m": vertex_max,
        "zero_pose_bit_identical_target_bind": matrix_ok,
        "pose_time_search": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "publishable": False,
    }


__all__ = [
    "FOOT_ROOTS",
    "HAND_ROOTS",
    "POSE_MAP_V10_COMPOSITION",
    "apply_pose_map_global_v10",
    "build_pose_map_v10",
    "check_pose_map_v10",
    "pose_whole_chain_vertices_v10",
]
