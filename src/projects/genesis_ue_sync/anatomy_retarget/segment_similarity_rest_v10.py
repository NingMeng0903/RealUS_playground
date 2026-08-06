"""Per-segment isotropic similarity rest fit (SKEL s(J_B) style).

After joint-anchored FK is correct, bone lengths may still exceed the
subject anatomical segment (femur mesh ~405 mm vs hip–knee ~373 mm).  This
module applies one similarity transform per main-chain segment:

    s_seg   = |A_child_subj - A_parent_subj| / |A_child_tmpl - A_parent_tmpl|
    R_align = minimal rotation aligning template segment axis to subject
    S_seg   = T(A_parent_subj) · R_align · s_seg · T(-A_parent_tmpl)

Per-beta anatomical pivots are *not* raw SMPL-X joints.  They are migrated
from the frozen Node1 calibration:

    A_subj[j] = F_station_subj[j] @ station_from_anatomical[j]

Bones, controller binds, and vessel/nerve verts share the same segment
transforms.  Cross-segment tubes blend similarities with the frozen 14-slot
weights.  Topology and weights are never rewritten.  Radial/centroid shrink
is forbidden (mechanism_v8 blacklist).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from .anatomical_calibration_v1 import AnatomicalCalibrationV1, JOINT_SPECS
from .anatomy_lbs import source_bone_driver_frames
from .chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    _global_to_local,
    _weighted_rest_correction,
)
from .pose_map_v10 import FOOT_ROOTS, HAND_ROOTS
from .whole_chain_rest_fit_v1 import _descendants


SEGMENT_SIMILARITY_KIND = "SegmentSimilarityRestV10"
SEGMENT_SIMILARITY_SCHEMA = 1

# Parent -> child anatomical joint pairs that define each scaled segment.
SEGMENT_SPECS = (
    ("left_hip", "left_knee", "left_femur"),
    ("left_knee", "left_ankle", "left_shank"),
    ("right_hip", "right_knee", "right_femur"),
    ("right_knee", "right_ankle", "right_shank"),
    ("left_shoulder", "left_elbow", "left_humerus"),
    ("left_elbow", "left_wrist", "left_forearm"),
    ("right_shoulder", "right_elbow", "right_humerus"),
    ("right_elbow", "right_wrist", "right_forearm"),
)

# Controllers whose bind origin rides with each segment.
# Distal hinge bones (Knee_Rotate / Elbow_Rot) stay on the *proximal* segment so
# femur/humerus and the hinge share one similarity and the child segment starts
# from that same joint center (SKEL seating).
SEGMENT_PROXIMAL_CONTROLLER = {
    "left_femur": "Femur_Rot_L",
    "right_femur": "Femur_Rot_R",
    "left_shank": "Tibia_Bone_L",
    "right_shank": "Tibia_Bone_R",
    "left_humerus": "Shoulder_Rotate_L",
    "right_humerus": "Shoulder_Rotate_R",
    "left_forearm": "Forearm_Bone_L",
    "right_forearm": "Forearm_Bone_R",
}
SEGMENT_DISTAL_CONTROLLER = {
    "left_femur": "Knee_Rotate_L",
    "right_femur": "Knee_Rotate_R",
    "left_shank": "Ankle_Rot_L",
    "right_shank": "Ankle_Rot_R",
    "left_humerus": "Elbow_Rot_L",
    "right_humerus": "Elbow_Rot_R",
    "left_forearm": "Wrist_Rotate_L",
    "right_forearm": "Wrist_Rotate_R1",
}
# Extra controllers glued to the proximal segment (patella follows femur).
SEGMENT_EXTRA_CONTROLLERS = {
    "left_femur": ("Patella_Rotate_L",),
    "right_femur": ("Patella_Rotate_R",),
}


def _rotation_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector a onto unit vector b."""

    u = np.asarray(a, dtype=np.float64).reshape(3)
    v = np.asarray(b, dtype=np.float64).reshape(3)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1.0e-12 or nv < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    u = u / nu
    v = v / nv
    cross = np.cross(u, v)
    cos = float(np.clip(np.dot(u, v), -1.0, 1.0))
    sin = float(np.linalg.norm(cross))
    if sin < 1.0e-12:
        if cos > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180 deg: pick any orthogonal axis.
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis, u))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - u * float(np.dot(axis, u))
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sin
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + k * sin + (k @ k) * (1.0 - cos)


def _similarity_matrix(
    *,
    parent_src: np.ndarray,
    child_src: np.ndarray,
    parent_tgt: np.ndarray,
    child_tgt: np.ndarray,
    source_length_m: float | None = None,
) -> tuple[np.ndarray, float]:
    """Map a source segment onto a target segment (isotropic similarity).

    ``source_length_m`` overrides ``|child_src-parent_src|`` when the authored
    bone mesh is longer than the bind span (femur mesh ~405 mm vs bind ~373 mm).
    """

    axis_src = child_src - parent_src
    axis_tgt = child_tgt - parent_tgt
    len_src = float(source_length_m) if source_length_m is not None else float(
        np.linalg.norm(axis_src)
    )
    len_tgt = float(np.linalg.norm(axis_tgt))
    if len_src < 1.0e-6:
        raise ValueError("source segment length is degenerate")
    scale = len_tgt / len_src
    rotation = _rotation_align(axis_src, axis_tgt)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = parent_tgt - scale * (rotation @ parent_src)
    return matrix, scale


def _mesh_axial_span_m(
    vertices: np.ndarray,
    asset: Any,
    *,
    controller_index: int,
    axis_origin: np.ndarray,
    axis_target: np.ndarray,
) -> float:
    names_ctrl = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = [str(t).strip().lower() for t in asset.source_tissues]
    ids = []
    for controller, (start, stop), tissue in zip(
        names_ctrl.tolist(), ranges.tolist(), tissues
    ):
        if int(controller) == int(controller_index) and tissue == "bone":
            ids.append(np.arange(int(start), int(stop), dtype=np.int64))
    if not ids:
        return float(np.linalg.norm(axis_target - axis_origin))
    pts = np.asarray(vertices, dtype=np.float64)[np.concatenate(ids)]
    origin = np.asarray(axis_origin, dtype=np.float64).reshape(3)
    axis = np.asarray(axis_target, dtype=np.float64) - origin
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-9:
        return 0.0
    axis = axis / norm
    # Distal extent from the parent origin only — do not add proximal overhang.
    proj = (pts - origin) @ axis
    distal = float(np.max(proj))
    if distal < 1.0e-6:
        return float(norm)
    return distal


def subject_anatomical_pivots_v10(
    asset: Any,
    calibration: AnatomicalCalibrationV1,
) -> np.ndarray:
    """Migrate Node1 anatomical pivots to a beta-shaped subject (metres).

    ``A_subj[j] = F_station_subj[j] @ station_from_anatomical[j]``.
    Never snaps to raw SMPL-X joints.
    """

    zero = np.zeros((55, 3), dtype=np.float32)
    station_all = np.asarray(source_bone_driver_frames(asset, zero), dtype=np.float64)
    controllers = np.asarray(calibration.controller_indices, dtype=np.int32)
    station = station_all[controllers]
    migrate = np.asarray(calibration.station_from_anatomical, dtype=np.float64)
    return station @ migrate


def _segment_controller_sets(
    asset: Any,
) -> dict[str, set[int]]:
    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    groups: dict[str, set[int]] = {}
    for _parent_j, _child_j, segment in SEGMENT_SPECS:
        proximal = names.index(SEGMENT_PROXIMAL_CONTROLLER[segment])
        distal = names.index(SEGMENT_DISTAL_CONTROLLER[segment])
        if segment.endswith("_femur") or segment.endswith("_humerus"):
            # Proximal segment owns the hinge bone; exclude hinge children.
            members = _descendants(parents, proximal)
            hinge_kids = _descendants(parents, distal) - {distal}
            groups[segment] = {int(i) for i in members - hinge_kids}
            for extra in SEGMENT_EXTRA_CONTROLLERS.get(segment, ()):
                if extra in names:
                    groups[segment].add(names.index(extra))
        else:
            # Distal segment: from its proximal bone through the distal joint,
            # excluding terminal hand/foot children of wrist/ankle.
            members = _descendants(parents, proximal)
            distal_kids = _descendants(parents, distal) - {distal}
            groups[segment] = {int(i) for i in members - distal_kids}
            groups[segment].add(distal)
            # Patella is parented under Tibia in the Blender rig but must ride
            # with the femur/knee hinge, never the shank similarity.
            for extra_seg, extras in SEGMENT_EXTRA_CONTROLLERS.items():
                if not segment.startswith(extra_seg.split("_")[0]):
                    continue
                for extra in extras:
                    if extra in names:
                        groups[segment].discard(names.index(extra))
    # Force extras onto their proximal segment (overwrite any shank claim).
    for segment, extras in SEGMENT_EXTRA_CONTROLLERS.items():
        for extra in extras:
            if extra in names and segment in groups:
                for other, members in groups.items():
                    if other != segment:
                        members.discard(names.index(extra))
                groups[segment].add(names.index(extra))
    return groups


def _controller_to_segment(groups: Mapping[str, set[int]], n_bones: int) -> np.ndarray:
    assign = np.full(n_bones, -1, dtype=np.int32)
    order = list(groups.keys())
    for index, segment in enumerate(order):
        for bone in groups[segment]:
            assign[int(bone)] = index
    return assign


@dataclass(frozen=True)
class SegmentSimilarityRestV10:
    subject_label: str
    betas: np.ndarray
    anatomical_pivots_subj: np.ndarray
    anatomical_pivots_tmpl: np.ndarray
    segment_names: tuple[str, ...]
    segment_scales: np.ndarray
    segment_matrices: np.ndarray
    controller_segment: np.ndarray
    vertices_final: np.ndarray
    B_final: np.ndarray
    C_bone: np.ndarray
    target_local_bind: np.ndarray
    inverse_bind: np.ndarray
    build_report: Mapping[str, Any]

    def validate(self) -> None:
        n = len(self.segment_names)
        if self.segment_scales.shape != (n,):
            raise ValueError("segment_scales shape mismatch")
        if self.segment_matrices.shape != (n, 4, 4):
            raise ValueError("segment_matrices shape mismatch")
        if self.anatomical_pivots_subj.shape != (len(JOINT_SPECS), 4, 4):
            raise ValueError("subject anatomical pivots must cover 12 joints")


def build_segment_similarity_rest_v10(
    value: ChainRestFitSubjectV1,
    *,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    scale_clip: tuple[float, float] = (0.85, 1.15),
) -> SegmentSimilarityRestV10:
    """Apply per-segment isotropic similarity on top of an existing rest fit.

    Starts from ``value.vertices_final`` / ``B_final`` (typically V7) and
    returns a new rest with shorter/longer segments seated on per-beta
    anatomical pivots.  Topology and LBS weights are untouched.
    """

    started = time.perf_counter()
    names = list(asset.source_bone_names or ())
    parents = np.asarray(value.bone_parents, dtype=np.int64)
    joint_index = {spec.name: i for i, spec in enumerate(JOINT_SPECS)}

    a_tmpl = np.asarray(calibration.anatomical_rest_global, dtype=np.float64)
    a_subj = subject_anatomical_pivots_v10(asset, calibration)

    # Guard: never collapse onto raw SMPL-X stations (hip offset was 57-61 mm).
    station = np.asarray(calibration.station_rest_global, dtype=np.float64)
    station_subj_all = np.asarray(
        source_bone_driver_frames(asset, np.zeros((55, 3), dtype=np.float32)),
        dtype=np.float64,
    )
    controllers = np.asarray(calibration.controller_indices, dtype=np.int32)
    station_subj = station_subj_all[controllers]
    hip_offsets = []
    for side in ("left_hip", "right_hip"):
        j = joint_index[side]
        hip_offsets.append(
            float(np.linalg.norm(a_subj[j, :3, 3] - station_subj[j, :3, 3]))
        )
    if min(hip_offsets) < 0.020:
        raise ValueError(
            "subject anatomical hip snapped too close to SMPL-X station "
            f"(offsets_m={hip_offsets}); refusing raw-joint snap"
        )

    groups = _segment_controller_sets(asset)
    segment_names = tuple(seg for *_unused, seg in SEGMENT_SPECS)
    assign = _controller_to_segment(groups, len(names))

    # One similarity per segment: scale by mesh axial span so overlong
    # condyles shrink, mapping parent_bind → A_parent and mesh-distal → A_child.
    # Distal hinge controllers are then snapped to A_child so FK seating stays
    # exact even though the bind joint was proximal to the mesh tip.
    matrices = np.tile(np.eye(4, dtype=np.float64), (len(segment_names), 1, 1))
    scales = np.ones(len(segment_names), dtype=np.float64)
    reports: dict[str, Any] = {}
    lo, hi = scale_clip
    b_src = np.asarray(value.B_final, dtype=np.float64)
    rest = np.asarray(value.vertices_final, dtype=np.float64)
    hinge_snap: dict[str, int] = {}
    for index, (parent_name, child_name, segment) in enumerate(SEGMENT_SPECS):
        p = joint_index[parent_name]
        c = joint_index[child_name]
        proximal = names.index(SEGMENT_PROXIMAL_CONTROLLER[segment])
        child_ctrl = names.index(SEGMENT_DISTAL_CONTROLLER[segment])
        if segment.endswith("_shank"):
            hinge = names.index(
                "Knee_Rotate_L" if segment.startswith("left") else "Knee_Rotate_R"
            )
            parent_src = b_src[hinge, :3, 3]
            mesh_controller = proximal
        elif segment.endswith("_forearm"):
            hinge = names.index(
                "Elbow_Rot_L" if segment.startswith("left") else "Elbow_Rot_R"
            )
            parent_src = b_src[hinge, :3, 3]
            mesh_controller = proximal
        else:
            parent_src = b_src[proximal, :3, 3]
            mesh_controller = proximal
            hinge_snap[segment] = child_ctrl
        child_bind = b_src[child_ctrl, :3, 3]
        parent_tgt = a_subj[p, :3, 3]
        child_tgt = a_subj[c, :3, 3]
        bind_len = float(np.linalg.norm(child_bind - parent_src))
        mesh_span = _mesh_axial_span_m(
            rest,
            asset,
            controller_index=mesh_controller,
            axis_origin=parent_src,
            axis_target=child_bind,
        )
        axis = child_bind - parent_src
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1.0e-9:
            raise ValueError(f"degenerate bind axis for {segment}")
        # Source child = mesh distal tip along the bind axis (not the hinge).
        child_src = parent_src + (axis / axis_norm) * float(mesh_span)
        matrix, scale = _similarity_matrix(
            parent_src=parent_src,
            child_src=child_src,
            parent_tgt=parent_tgt,
            child_tgt=child_tgt,
        )
        if not lo <= scale <= hi:
            raise ValueError(
                f"segment scale out of bounds for {segment}: {scale:.4f} not in [{lo},{hi}]"
            )
        matrices[index] = matrix
        scales[index] = scale
        reports[segment] = {
            "parent_joint": parent_name,
            "child_joint": child_name,
            "mesh_axial_span_m": float(mesh_span),
            "bind_length_m": float(bind_len),
            "subject_anatomical_length_m": float(np.linalg.norm(child_tgt - parent_tgt)),
            "template_anatomical_length_m": float(
                np.linalg.norm(a_tmpl[c, :3, 3] - a_tmpl[p, :3, 3])
            ),
            "scale": float(scale),
            "n_controllers": int(len(groups[segment])),
        }

    bone_S = np.tile(np.eye(4, dtype=np.float64), (len(names), 1, 1))
    for bone, seg_i in enumerate(assign.tolist()):
        if seg_i >= 0:
            bone_S[bone] = matrices[seg_i]

    posed = _weighted_rest_correction(
        rest,
        asset.driver_indices,
        asset.driver_weights,
        bone_S,
    )

    b_final = bone_S @ b_src
    for bone, seg_i in enumerate(assign.tolist()):
        if seg_i < 0:
            b_final[bone] = b_src[bone]
    # Snap proximal-segment hinge origins (Knee/Elbow) + patella to A_child.
    for segment, hinge_i in hinge_snap.items():
        seg_i = segment_names.index(segment)
        parent_name, child_name, _seg = SEGMENT_SPECS[seg_i]
        child_tgt = a_subj[joint_index[child_name], :3, 3]
        b_final[hinge_i, :3, 3] = child_tgt
        for extra in SEGMENT_EXTRA_CONTROLLERS.get(segment, ()):
            if extra in names:
                b_final[names.index(extra), :3, 3] = child_tgt
        reports[segment]["hinge_snapped_to_anatomical"] = True

    # Carry copy-142 hand/foot binds with the moved wrist/ankle root so T-pose
    # joint-anchored rebase stays identity on terminal B_tgt.
    terminal_transport: dict[str, Any] = {}
    for root_name in (*HAND_ROOTS, *FOOT_ROOTS):
        root = names.index(root_name)
        transport = b_final[root] @ np.linalg.inv(b_src[root])
        members = sorted(_descendants(parents, root) - {root})
        if np.allclose(transport, np.eye(4), atol=1.0e-12):
            terminal_transport[root_name] = {
                "applied": False,
                "n_descendants": len(members),
            }
            continue
        for bone in members:
            b_final[bone] = transport @ b_src[bone]
        # Rigidly carry terminal mesh verts that are dominated by those bones.
        controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
        ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
        tissues = [str(t).strip().lower() for t in asset.source_tissues]
        R = transport[:3, :3]
        t = transport[:3, 3]
        n_verts = 0
        for controller, (start, stop), tissue in zip(
            controllers.tolist(), ranges.tolist(), tissues
        ):
            if tissue != "bone" or int(controller) not in set(members) | {root}:
                continue
            # Root bone meshes (wrist/ankle caps) follow the root bind too.
            pts = posed[int(start) : int(stop)]
            posed[int(start) : int(stop)] = pts @ R.T + t
            n_verts += int(stop) - int(start)
        terminal_transport[root_name] = {
            "applied": True,
            "n_descendants": len(members),
            "n_vertices": n_verts,
            "translation_m": t.tolist(),
        }

    target_local = _global_to_local(b_final, parents)
    inverse = np.linalg.inv(b_final)
    c_bone = b_final @ np.linalg.inv(np.asarray(value.B_prefit, dtype=np.float64))

    result = SegmentSimilarityRestV10(
        subject_label=str(value.subject_label),
        betas=np.asarray(value.betas, dtype=np.float64).copy(),
        anatomical_pivots_subj=a_subj,
        anatomical_pivots_tmpl=a_tmpl,
        segment_names=segment_names,
        segment_scales=scales,
        segment_matrices=matrices,
        controller_segment=assign,
        vertices_final=posed.astype(np.float32),
        B_final=b_final.astype(np.float64),
        C_bone=c_bone.astype(np.float64),
        target_local_bind=target_local.astype(np.float64),
        inverse_bind=inverse.astype(np.float64),
        build_report={
            "method": "segment_isotropic_similarity_v10",
            "scale_policy": "mesh_axial_span_to_subject_anatomical",
            "segments": reports,
            "terminal_transport": terminal_transport,
            "hip_station_offset_m": {
                "left": hip_offsets[0],
                "right": hip_offsets[1],
            },
            "topology_preserved": True,
            "weights_preserved": True,
            "banned_shrink": True,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )
    result.validate()
    return result


def apply_segment_similarity_to_subject_v10(
    value: ChainRestFitSubjectV1,
    similarity: SegmentSimilarityRestV10,
) -> ChainRestFitSubjectV1:
    """Return a new ChainRestFitSubjectV1 with similarity-applied rest/bind."""

    report = dict(value.build_report)
    report["segment_similarity_v10"] = dict(similarity.build_report)
    report["pose_composition_authority"] = "joint_anchored_fk_v10"
    return replace(
        value,
        vertices_final=np.asarray(similarity.vertices_final, dtype=np.float32),
        B_final=np.asarray(similarity.B_final, dtype=np.float64),
        C_bone=np.asarray(similarity.C_bone, dtype=np.float64),
        target_local_bind=np.asarray(similarity.target_local_bind, dtype=np.float64),
        inverse_bind=np.asarray(similarity.inverse_bind, dtype=np.float64),
        build_report=report,
    )


def controller_segment_scales_v10(
    similarity: SegmentSimilarityRestV10,
) -> np.ndarray:
    """Per-controller scale factors for residual joint-translation scaling."""

    scales = np.ones(len(similarity.controller_segment), dtype=np.float64)
    for bone, seg_i in enumerate(similarity.controller_segment.tolist()):
        if seg_i >= 0:
            scales[bone] = float(similarity.segment_scales[seg_i])
    return scales


__all__ = [
    "SEGMENT_SIMILARITY_KIND",
    "SEGMENT_SIMILARITY_SCHEMA",
    "SEGMENT_SPECS",
    "SegmentSimilarityRestV10",
    "apply_segment_similarity_to_subject_v10",
    "build_segment_similarity_rest_v10",
    "controller_segment_scales_v10",
    "subject_anatomical_pivots_v10",
]
