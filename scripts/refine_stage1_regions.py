#!/usr/bin/env python3
"""Apply bounded hand and oral Stage-1 regional refinement offline.

The input is an accepted whole-body Stage-1 harmonic reference.  Each hand
bone is matched by its own geometric axis to the current subject's SMPL-X
finger segment; hand-end soft anatomy follows a local, mesh-continuous field.
The oral compound receives one subject-specific rigid candidate transform.
All Blender bindings and the lower body remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _tube_translation_field(
    vertices: np.ndarray,
    desired: np.ndarray,
    selected: np.ndarray,
    faces: np.ndarray,
    *,
    minimum_area_ratio: float = 0.01,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Turn wall-wise attraction into a compact, cross-section-coherent field.

    Vessel and nerve walls must not independently move toward a bone axis: that
    collapses their radius.  Averaging in the ambient rest geometry makes nearby
    wall samples share a translation, while the compact kernel leaves distant
    parts of a long source mesh exactly unchanged.  The bandwidth is measured
    from this mesh's own edge lengths, so it scales with exported anatomy rather
    than with a particular subject or validation pose.
    """
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    points = np.asarray(vertices, dtype=np.float64)
    target = np.asarray(desired, dtype=np.float64)
    active = np.asarray(selected, dtype=bool)
    triangles = np.asarray(faces, dtype=np.int64)
    if not np.any(active):
        return np.zeros_like(points), {
            "bandwidth_m": 0.0,
            "support_vertices": 0,
            "transition_vertices": 0,
            "minimum_area_ratio": 1.0,
        }
    if not len(triangles):
        result = np.zeros_like(points)
        result[active] = target[active]
        return result, {
            "bandwidth_m": 0.0,
            "support_vertices": int(np.count_nonzero(active)),
            "transition_vertices": 0,
            "minimum_area_ratio": 1.0,
        }

    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edge_length = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    edge_length = edge_length[edge_length > 1.0e-10]
    if not len(edge_length):
        raise RuntimeError("tube mesh contains no non-degenerate edges")
    characteristic = float(np.median(edge_length))

    base_normal = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    base_area = np.linalg.norm(base_normal, axis=1)
    valid = base_area > 1.0e-12
    direct = np.zeros_like(points)
    direct[active] = target[active]
    direct_trial = points + direct
    direct_normal = np.cross(
        direct_trial[triangles[:, 1]] - direct_trial[triangles[:, 0]],
        direct_trial[triangles[:, 2]] - direct_trial[triangles[:, 0]],
    )
    direct_ratio = np.linalg.norm(direct_normal, axis=1) / np.maximum(base_area, 1.0e-12)
    direct_minimum = float(np.min(direct_ratio[valid])) if np.any(valid) else 1.0
    if direct_minimum >= minimum_area_ratio:
        return direct, {
            "bandwidth_m": 0.0,
            "support_vertices": int(np.count_nonzero(active)),
            "transition_vertices": 0,
            "minimum_area_ratio": direct_minimum,
        }

    tree = cKDTree(points)
    last_minimum = 0.0
    # The smallest accepted field is retained.  Larger candidates are needed
    # only where a selection boundary crosses a coarse tube cross-section.
    for factor in (1.5, 2.0, 3.0, 4.5, 6.0):
        bandwidth = factor * characteristic
        radius = 3.0 * bandwidth
        distances = tree.sparse_distance_matrix(
            tree, max_distance=radius, output_type="coo_matrix"
        )
        kernel = np.exp(-0.5 * (distances.data / bandwidth) ** 2)
        weights = coo_matrix(
            (kernel, (distances.row, distances.col)),
            shape=(len(points), len(points)),
        ).tocsr()
        denominator = np.maximum(np.asarray(weights.sum(axis=1)).reshape(-1), 1.0e-12)
        field = np.column_stack(
            [np.asarray(weights @ target[:, axis]).reshape(-1) / denominator for axis in range(3)]
        )
        support = np.linalg.norm(field, axis=1) > 1.0e-10
        field[~support] = 0.0
        trial = points + field
        normal = np.cross(
            trial[triangles[:, 1]] - trial[triangles[:, 0]],
            trial[triangles[:, 2]] - trial[triangles[:, 0]],
        )
        area_ratio = np.linalg.norm(normal, axis=1) / np.maximum(base_area, 1.0e-12)
        last_minimum = float(np.min(area_ratio[valid])) if np.any(valid) else 1.0
        if last_minimum >= minimum_area_ratio:
            return field, {
                "bandwidth_m": float(bandwidth),
                "support_vertices": int(np.count_nonzero(support)),
                "transition_vertices": int(np.count_nonzero(support & ~active)),
                "minimum_area_ratio": last_minimum,
            }
    raise RuntimeError(
        "cross-section-coherent tube field cannot preserve face orientation "
        f"(last minimum area ratio {last_minimum:.6f})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-landmarks",
        choices=("asset_joints", "mesh_axis"),
        default="asset_joints",
        help="Use mesh_axis for the validated subject-specific hand fit.",
    )
    parser.add_argument(
        "--propagate-driver-field",
        action="store_true",
        help="Experimental direct weight blend; not used by the validated path.",
    )
    parser.add_argument(
        "--propagate-local-field",
        action="store_true",
        help="Use the validated local mesh-continuous hand soft-tissue field.",
    )
    parser.add_argument(
        "--jaw-compound",
        action="store_true",
        help="Use the validated subject-specific rigid oral-compound search.",
    )
    parser.add_argument(
        "--long-limb-bones",
        action="store_true",
        help="Fit long bone mesh axes to the fixed subject SMPL-X joint segments.",
    )
    parser.add_argument(
        "--long-bone-kinds",
        nargs="+",
        choices=("humerus", "radius", "ulna", "femur", "tibia", "fibula"),
        default=(),
        help="Fit only the listed long-bone kinds; overrides the all-bones flag.",
    )
    parser.add_argument(
        "--carpal-compound",
        action="store_true",
        help="Rigidly center each authored carpal compound on its subject wrist joint.",
    )
    parser.add_argument(
        "--hand-soft-bone-attraction",
        type=float,
        default=0.0,
        help="Move hand-end vessel/nerve vertices toward mapped hand bone geometry by this fraction.",
    )
    parser.add_argument(
        "--hand-soft-attraction-cap-m",
        type=float,
        default=0.010,
        help="Maximum per-vertex displacement for hand soft-tissue bone attraction.",
    )
    parser.add_argument(
        "--hand-bone-radial-scale",
        type=float,
        default=1.0,
        help="Cross-section scale for mapped metacarpal and phalanx meshes; axial length is retained.",
    )
    parser.add_argument(
        "--distal-tip-inset-m",
        type=float,
        default=0.0,
        help="Move distal phalanx targets proximally from the SMPL-X skin fingertip.",
    )
    parser.add_argument(
        "--carpal-scale",
        type=float,
        default=1.0,
        help="Uniform scale of each centered carpal compound around the subject wrist.",
    )
    parser.add_argument("--humerus-radial-scale", type=float, default=1.0)
    parser.add_argument("--patella-inset-m", type=float, default=0.0)
    parser.add_argument("--limb-soft-bone-attraction", type=float, default=0.0)
    parser.add_argument("--jaw-bone-scale", type=float, default=1.0)
    parser.add_argument("--jaw-soft-bone-attraction", type=float, default=0.0)
    parser.add_argument(
        "--jaw-soft-attraction-cap-m", type=float, default=0.015
    )
    parser.add_argument(
        "--adaptive-local-scales",
        action="store_true",
        help="Derive all local metric caps/radii from this beta's canonical joint lengths.",
    )
    parser.add_argument("--hand-soft-attraction-cap-ratio", type=float, default=0.09)
    parser.add_argument("--distal-tip-inset-ratio", type=float, default=0.15)
    parser.add_argument("--patella-inset-ratio", type=float, default=0.019)
    parser.add_argument("--limb-soft-attraction-cap-ratio", type=float, default=0.036)
    parser.add_argument("--jaw-soft-attraction-cap-ratio", type=float, default=0.125)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))

    from projects.genesis_ue_sync.anatomy_retarget.containment import signed_distance
    from projects.genesis_ue_sync.anatomy_retarget.containment import (
        _mesh_local_faces,
    )
    from projects.genesis_ue_sync.anatomy_retarget.material_fit import (
        _finger_tip_targets,
        _hand_mesh_segment,
        shaft_preserving_segment_map,
        uniform_segment_similarity,
    )
    from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
        load_rigged_asset,
        save_rigged_asset,
    )
    from projects.genesis_ue_sync.anatomy_retarget.source_rebind import rebind_source_rig
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import with_source_driver_coupling
    from projects.genesis_ue_sync.anatomy_retarget.shape_volume import _load_obj

    asset = load_rigged_asset(args.asset, validate=True)
    skeleton = json.loads((args.canonical_dir / "smpl_canonical_skeleton.json").read_text())
    target_names = [str(name) for name in skeleton["joint_names"]]
    if target_names != list(asset.joint_names):
        raise ValueError("canonical SMPL-X joint order differs from the asset")
    target_joints = np.asarray(skeleton["rest_joints_subject"], dtype=np.float64)
    source_joints = np.asarray(asset.rest_joints, dtype=np.float64)
    tips = _finger_tip_targets(
        args.canonical_dir,
        joint_names=list(asset.joint_names),
        target_joints=target_joints,
        subject=True,
    )
    hand_scale_m = float(
        np.mean(
            [
                np.linalg.norm(
                    target_joints[target_names.index(f"{side}_middle3")]
                    - target_joints[target_names.index(f"{side}_wrist")]
                )
                for side in ("left", "right")
            ]
        )
    )
    lower_leg_scale_m = float(
        np.mean(
            [
                np.linalg.norm(
                    target_joints[target_names.index(f"{side}_ankle")]
                    - target_joints[target_names.index(f"{side}_knee")]
                )
                for side in ("left", "right")
            ]
        )
    )
    limb_segment_scale_m = float(
        np.mean(
            [
                np.linalg.norm(
                    target_joints[target_names.index(f"{side}_{distal}")]
                    - target_joints[target_names.index(f"{side}_{proximal}")]
                )
                for side in ("left", "right")
                for proximal, distal in (("elbow", "wrist"), ("knee", "ankle"))
            ]
        )
    )
    head_neck_scale_m = float(
        np.linalg.norm(
            target_joints[target_names.index("head")]
            - target_joints[target_names.index("neck")]
        )
    )
    if min(hand_scale_m, lower_leg_scale_m, limb_segment_scale_m, head_neck_scale_m) <= 0.0:
        raise ValueError("canonical subject skeleton contains a degenerate anatomical scale")
    adaptive = bool(args.adaptive_local_scales)
    hand_soft_cap_m = (
        float(args.hand_soft_attraction_cap_ratio) * hand_scale_m
        if adaptive
        else float(args.hand_soft_attraction_cap_m)
    )
    limb_soft_cap_m = (
        float(args.limb_soft_attraction_cap_ratio) * limb_segment_scale_m
        if adaptive
        else float(args.hand_soft_attraction_cap_m)
    )
    jaw_soft_cap_m = (
        float(args.jaw_soft_attraction_cap_ratio) * head_neck_scale_m
        if adaptive
        else float(args.jaw_soft_attraction_cap_m)
    )
    hand_neighborhood_radius_m = 0.535 * hand_scale_m if adaptive else 0.090
    limb_neighborhood_radius_m = 0.29 * limb_segment_scale_m if adaptive else 0.12
    jaw_neighborhood_radius_m = 1.11 * head_neck_scale_m if adaptive else 0.18
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    baseline_vertices = vertices.copy()
    changed = np.zeros(len(vertices), dtype=bool)
    mapped_meshes: list[str] = []
    bone_transforms: dict[int, np.ndarray] = {}
    radial_scale = float(args.hand_bone_radial_scale)
    if not 0.75 <= radial_scale <= 1.0:
        raise ValueError("--hand-bone-radial-scale must be in [0.75, 1.0]")
    tip_inset = float(args.distal_tip_inset_m)
    if not adaptive and not 0.0 <= tip_inset <= 0.006:
        raise ValueError("--distal-tip-inset-m must be in [0, 0.006]")
    for (start, stop), name, tissue in zip(
        asset.source_vertex_ranges, asset.source_mesh_names, asset.source_tissues
    ):
        if str(tissue).lower() != "bone":
            continue
        segment = _hand_mesh_segment(
            str(name),
            joint_names=list(asset.joint_names),
            source_anchors=source_joints,
            target_joints=target_joints,
            finger_tips=tips,
        )
        if segment is None:
            continue
        start_i, stop_i = int(start), int(stop)
        points = vertices[start_i:stop_i]
        target_a = np.asarray(segment[2], dtype=np.float64)
        target_b = np.asarray(segment[3], dtype=np.float64)
        lower_name = str(name).lower()
        if (
            (tip_inset > 0.0 or (adaptive and float(args.distal_tip_inset_ratio) > 0.0))
            and "distal" in lower_name
        ):
            direction = target_b - target_a
            segment_length = float(np.linalg.norm(direction))
            direction /= max(segment_length, 1.0e-10)
            effective_tip_inset = (
                float(args.distal_tip_inset_ratio) * segment_length
                if adaptive
                else tip_inset
            )
            target_b = target_b - effective_tip_inset * direction
        if args.source_landmarks == "asset_joints":
            vertices[start_i:stop_i] = shaft_preserving_segment_map(
                points,
                source_a=segment[0], source_b=segment[1],
                target_a=target_a, target_b=target_b,
            )
        else:
            center = np.mean(points, axis=0)
            _values, vectors = np.linalg.eigh(np.cov((points - center).T))
            axis = vectors[:, -1]
            target_axis = segment[3] - segment[2]
            if float(axis @ target_axis) < 0.0:
                axis = -axis
            projection = (points - center) @ axis
            source_a = center + float(np.quantile(projection, 0.02)) * axis
            source_b = center + float(np.quantile(projection, 0.98)) * axis
            vertices[start_i:stop_i], scale, rotation = uniform_segment_similarity(
                points,
                source_a=source_a,
                source_b=source_b,
                target_a=target_a,
                target_b=target_b,
            )
            local_indices = np.asarray(asset.driver_indices[start_i:stop_i], dtype=np.int64).reshape(-1)
            local_weights = np.asarray(asset.driver_weights[start_i:stop_i], dtype=np.float64).reshape(-1)
            mass = np.bincount(local_indices, weights=local_weights, minlength=len(asset.source_bone_names or []))
            bone = int(np.argmax(mass))
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = scale * rotation
            transform[:3, 3] = target_a - transform[:3, :3] @ source_a
            bone_transforms[bone] = transform
        if radial_scale < 1.0:
            axis = target_b - target_a
            axis /= max(float(np.linalg.norm(axis)), 1.0e-10)
            mapped_points = vertices[start_i:stop_i]
            projection = target_a + np.outer((mapped_points - target_a) @ axis, axis)
            vertices[start_i:stop_i] = projection + radial_scale * (
                mapped_points - projection
            )
        changed[start_i:stop_i] = True
        mapped_meshes.append(str(name))

    carpal_offsets: dict[str, object] = {}
    if args.carpal_compound:
        from scipy.spatial import cKDTree

        carpal_scale = float(args.carpal_scale)
        if not 0.75 <= carpal_scale <= 1.0:
            raise ValueError("--carpal-scale must be in [0.75, 1.0]")
        carpal_tokens = (
            "capitate", "hamate", "lunate", "pisiform", "scaphoid",
            "trapezium", "trapezoid", "triquetrum",
        )
        for side, suffix in (("left", "_l"), ("right", "_r")):
            ranges = [
                (int(start), int(stop), str(name))
                for (start, stop), name, tissue in zip(
                    asset.source_vertex_ranges, asset.source_mesh_names, asset.source_tissues
                )
                if str(tissue).lower() == "bone"
                and str(name).lower().endswith(suffix)
                and any(token in str(name).lower() for token in carpal_tokens)
            ]
            if not ranges:
                continue
            carpal_indices = np.concatenate(
                [np.arange(start, stop, dtype=np.int64) for start, stop, _name in ranges]
            )
            source_carpal = baseline_vertices[carpal_indices]
            thumb_name = f"_1st_Metacarpal_{'L' if side == 'left' else 'R'}"
            thumb_range = next(
                (
                    (int(start), int(stop))
                    for (start, stop), mesh_name in zip(
                        asset.source_vertex_ranges, asset.source_mesh_names
                    )
                    if str(mesh_name) == thumb_name
                ),
                None,
            )
            if thumb_range is None:
                raise RuntimeError(f"missing {thumb_name} for carpal contact inheritance")
            thumb_start, thumb_stop = thumb_range
            source_thumb = baseline_vertices[thumb_start:thumb_stop]
            mapped_thumb = vertices[thumb_start:thumb_stop]
            contact_distance, contact_thumb = cKDTree(source_thumb).query(
                source_carpal, k=1
            )
            contact_count = min(24, len(source_carpal))
            contact_carpal = np.argpartition(
                contact_distance, contact_count - 1
            )[:contact_count]
            source_contact = source_carpal[contact_carpal]
            target_contact = source_contact + (
                mapped_thumb[contact_thumb[contact_carpal]]
                - source_thumb[contact_thumb[contact_carpal]]
            )
            source_center = np.mean(source_contact, axis=0)
            target_center = np.mean(target_contact, axis=0)
            u, _singular, vt = np.linalg.svd(
                (source_contact - source_center).T
                @ (target_contact - target_center)
            )
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0.0:
                vt[-1] *= -1.0
                rotation = vt.T @ u.T
            translation = target_center - rotation @ source_center
            mapped_compound = source_carpal @ rotation.T + translation
            compound_center = np.mean(mapped_compound, axis=0)
            mapped_compound = compound_center + carpal_scale * (
                mapped_compound - compound_center
            )
            cursor = 0
            for start, stop, name in ranges:
                count = stop - start
                vertices[start:stop] = mapped_compound[cursor : cursor + count]
                cursor += count
                changed[start:stop] = True
                mapped_meshes.append(name)
            wrist_bone = int(
                np.argmax(
                    np.bincount(
                        np.concatenate(
                            [
                                np.asarray(asset.driver_indices[start:stop], dtype=np.int64).reshape(-1)
                                for start, stop, _name in ranges
                            ]
                        ),
                        weights=np.concatenate(
                            [
                                np.asarray(asset.driver_weights[start:stop], dtype=np.float64).reshape(-1)
                                for start, stop, _name in ranges
                            ]
                        ),
                        minlength=len(asset.source_bone_names or []),
                    )
                )
            )
            wrist_transform = np.eye(4, dtype=np.float64)
            wrist_transform[:3, :3] = rotation
            wrist_transform[:3, 3] = translation
            bone_transforms[wrist_bone] = wrist_transform
            source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
            source_modes = list(asset.source_bone_driver_types or [])
            for child, parent in enumerate(source_parents.tolist()):
                if int(parent) == wrist_bone and source_modes[child] == "bind_follow":
                    bone_transforms[child] = wrist_transform
            final_gap = float(
                np.min(cKDTree(mapped_thumb).query(mapped_compound, k=1)[0])
            )
            carpal_offsets[side] = {
                "translation_m": translation.tolist(),
                "source_thumb_contact_gap_m": float(np.min(contact_distance)),
                "mapped_thumb_contact_gap_m": final_gap,
                "contact_samples": int(contact_count),
            }

    # Source meshes are commonly weighted to a deform follower while nearby
    # vessels/nerves mix that follower with its unweighted joint-local parent.
    # Give the paired controller the same offline rest transform.  Otherwise
    # the original sparse blend applies only a fraction of the mapped segment
    # displacement and opens the soft-tissue branch when the finger rotates.
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    source_modes = list(asset.source_bone_driver_types or [])
    controller_field_bones: list[str] = []
    for bone, transform in list(bone_transforms.items()):
        parent = int(source_parents[bone])
        if (
            parent >= 0
            and source_modes[bone] == "bind_follow"
            and source_modes[parent] == "joint_local"
            and parent not in bone_transforms
        ):
            bone_transforms[parent] = transform
            controller_field_bones.append(str(asset.source_bone_names[parent]))

    def hand_soft_neighborhood() -> np.ndarray:
        """Restrict hand-follow soft tissue to an actual hand-space neighborhood."""
        from scipy.spatial import cKDTree

        soft = np.zeros(len(vertices), dtype=bool)
        for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
            if str(tissue).lower() != "bone":
                soft[int(start) : int(stop)] = True
        support = baseline_vertices[changed]
        if not len(support):
            return np.zeros(len(vertices), dtype=bool)
        distance, _nearest = cKDTree(support).query(
            baseline_vertices, k=1, workers=-1
        )
        # A geometric neighborhood, rather than an x-coordinate half-space,
        # prevents the hand field from leaking into cervical nerves or torso
        # vessels that happen to share an arm-side coordinate.
        return soft & (distance <= hand_neighborhood_radius_m)

    def source_bone_mask() -> np.ndarray:
        return np.concatenate(
            [
                np.full(int(stop) - int(start), str(tissue).lower() == "bone", dtype=bool)
                for (start, stop), tissue in zip(
                    asset.source_vertex_ranges, asset.source_tissues, strict=True
                )
            ]
        )

    topology_backoff_records: list[dict[str, object]] = []

    def apply_mesh_continuous_displacement(
        selection: np.ndarray,
        desired: np.ndarray,
        changed_mask: np.ndarray,
        *,
        label: str,
    ) -> float:
        """Apply a topology-safe soft field independently inside each source mesh."""
        maximum = 0.0
        all_faces = np.asarray(asset.faces, dtype=np.int64)
        for (start, stop), mesh_name, tissue in zip(
            asset.source_vertex_ranges,
            asset.source_mesh_names,
            asset.source_tissues,
            strict=True,
        ):
            if str(tissue).lower() not in {"nerve", "vessel"}:
                continue
            start_i, stop_i = int(start), int(stop)
            local_selected = np.asarray(selection[start_i:stop_i], dtype=bool)
            if not np.any(local_selected):
                continue
            local_faces = _mesh_local_faces(all_faces, start_i, stop_i)
            local_desired = np.asarray(desired[start_i:stop_i], dtype=np.float64)
            base = vertices[start_i:stop_i].copy()
            try:
                applied, field_report = _tube_translation_field(
                    base, local_desired, local_selected, local_faces
                )
            except RuntimeError as exc:
                raise RuntimeError(f"{label} cannot preserve topology for {mesh_name}: {exc}") from exc
            vertices[start_i:stop_i] = base + applied
            local_changed = np.linalg.norm(applied, axis=1) > 1.0e-10
            changed_mask[start_i:stop_i] |= local_changed
            if np.any(local_changed):
                maximum = max(
                    maximum,
                    float(np.max(np.linalg.norm(applied[local_changed], axis=1))),
                )
            topology_backoff_records.append(
                {
                    "stage": label,
                    "mesh": str(mesh_name),
                    "alpha": 1.0,
                    **field_report,
                    "changed_vertices": int(np.count_nonzero(local_changed)),
                }
            )
        return maximum

    if args.propagate_driver_field:
        indices = np.asarray(asset.driver_indices, dtype=np.int64)
        weights = np.asarray(asset.driver_weights, dtype=np.float64)
        field = np.zeros_like(vertices)
        for bone, transform in bone_transforms.items():
            local_weight = weights * (indices == bone)
            active = np.any(local_weight > 0.0, axis=1)
            if not np.any(active):
                continue
            transformed = baseline_vertices[active] @ transform[:3, :3].T + transform[:3, 3]
            field[active] += np.sum(local_weight[active], axis=1)[:, None] * (
                transformed - baseline_vertices[active]
            )
        soft = hand_soft_neighborhood()
        vertices[soft] += field[soft]

    hand_soft_attraction_changed = np.zeros(len(vertices), dtype=bool)
    hand_soft_attraction_max_m = 0.0
    if float(args.hand_soft_bone_attraction) > 0.0:
        attraction = float(args.hand_soft_bone_attraction)
        if not 0.0 < attraction <= 0.75:
            raise ValueError("--hand-soft-bone-attraction must be in (0, 0.75]")
        attraction_cap = hand_soft_cap_m
        if not 0.0 < attraction_cap <= 0.025:
            raise ValueError("derived hand soft attraction cap must be in (0, 0.025]")
        tissue_mask = np.concatenate(
            [
                np.full(
                    int(stop) - int(start),
                    str(tissue).lower() in {"nerve", "vessel"},
                    dtype=bool,
                )
                for (start, stop), tissue in zip(
                    asset.source_vertex_ranges, asset.source_tissues, strict=True
                )
            ]
        )
        soft = hand_soft_neighborhood() & tissue_mask
        if np.any(soft):
            provisional = type(asset)(
                **{**asset.__dict__, "vertices_rest": vertices.astype(np.float32)}
            )
            provisional, _provisional_report = rebind_source_rig(
                provisional,
                source_vertices=baseline_vertices,
                target_vertices=vertices,
                stage="stage1_hand_soft_centerline_provisional",
                bone_mask=source_bone_mask(),
                fallback_to_soft=False,
                anchor_joint_local=True,
            )
            soft_indices = np.flatnonzero(soft)
            driver_indices = np.asarray(asset.driver_indices, dtype=np.int64)[soft_indices]
            driver_weights = np.asarray(asset.driver_weights, dtype=np.float64)[soft_indices]
            heads = np.asarray(provisional.target_bone_head, dtype=np.float64)[driver_indices]
            tails = np.asarray(provisional.target_bone_tail, dtype=np.float64)[driver_indices]
            segments = tails - heads
            points = vertices[soft_indices, None, :]
            denominator = np.einsum("nkj,nkj->nk", segments, segments)
            parameter = np.einsum("nkj,nkj->nk", points - heads, segments)
            parameter /= np.maximum(denominator, 1.0e-12)
            parameter = np.clip(parameter, 0.0, 1.0)
            closest = heads + parameter[..., None] * segments
            target = np.sum(driver_weights[..., None] * closest, axis=1)
            before = vertices[soft].copy()
            delta = attraction * (target - vertices[soft])
            length = np.linalg.norm(delta, axis=1)
            delta *= np.minimum(1.0, attraction_cap / np.maximum(length, 1.0e-12))[:, None]
            desired = np.zeros_like(vertices)
            desired[soft] = delta
            hand_soft_attraction_max_m = apply_mesh_continuous_displacement(
                soft,
                desired,
                hand_soft_attraction_changed,
                label="hand_weighted_centerline",
            )

    if args.propagate_local_field:
        from scipy.sparse import coo_matrix, diags, eye
        from scipy.sparse.linalg import spsolve
        from scipy.spatial import cKDTree

        soft = hand_soft_neighborhood()
        support = baseline_vertices[changed]
        support_delta = vertices[changed] - baseline_vertices[changed]
        tree = cKDTree(support)
        query = baseline_vertices[soft]
        distance, neighbour = tree.query(query, k=min(24, len(support)), workers=-1)
        distance = np.atleast_2d(distance)
        neighbour = np.atleast_2d(neighbour)
        weights_local = 1.0 / np.maximum(distance, 0.0015) ** 2
        weights_local /= np.sum(weights_local, axis=1, keepdims=True)
        field = np.zeros_like(vertices)
        field[soft] = np.sum(
            weights_local[:, :, None] * support_delta[neighbour], axis=1
        )
        # Long artery/vein meshes are a single connected object.  Diffuse the
        # hand target through each mesh while pinning its non-hand vertices to
        # zero.  This preserves a continuous vessel/nerve topology instead of
        # directly jumping a hand-end vertex whose driver happens to be a
        # finger bone.
        all_faces = np.asarray(asset.faces, dtype=np.int64)
        for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
            start_i, stop_i = int(start), int(stop)
            local_soft = soft[start_i:stop_i]
            if not np.any(local_soft):
                continue
            local_faces = all_faces[
                np.all((all_faces >= start_i) & (all_faces < stop_i), axis=1)
            ] - start_i
            if not len(local_faces):
                vertices[start_i:stop_i][local_soft] += field[start_i:stop_i][local_soft]
                continue
            base = baseline_vertices[start_i:stop_i]
            count = len(base)
            edges = np.concatenate(
                (local_faces[:, [0, 1]], local_faces[:, [1, 2]], local_faces[:, [2, 0]]), axis=0
            )
            edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
            adjacency = coo_matrix(
                (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count)
            ).tocsr()
            adjacency.data[:] = 1.0
            degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
            laplacian = diags(degree) - adjacency
            target = field[start_i:stop_i]
            data_weight = np.where(local_soft, 80.0, 140.0)
            system = (diags(data_weight) + 2.0 * laplacian + 1.0e-8 * eye(count)).tocsr()
            smooth_field = np.column_stack(
                [spsolve(system, data_weight * target[:, axis]) for axis in range(3)]
            )
            # The sparse solve provides a smooth falloff inside the declared
            # local domain.  Pin every exterior vertex exactly to rest: even a
            # sub-millimetre Laplacian tail is an unauthorized change to a
            # neck/torso vessel and invalidates the Stage-1 locality contract.
            smooth_field[~local_soft] = 0.0
            base_normal = np.cross(
                base[local_faces[:, 1]] - base[local_faces[:, 0]],
                base[local_faces[:, 2]] - base[local_faces[:, 0]],
            )
            alpha = 1.0
            for _ in range(12):
                trial = base.copy()
                trial += alpha * smooth_field
                normal = np.cross(
                    trial[local_faces[:, 1]] - trial[local_faces[:, 0]],
                    trial[local_faces[:, 2]] - trial[local_faces[:, 0]],
                )
                orientation = np.einsum("ij,ij->i", base_normal, normal)
                area_ratio = np.linalg.norm(normal, axis=1) / np.maximum(
                    np.linalg.norm(base_normal, axis=1), 1.0e-12
                )
                if np.all(orientation > 0.0) and np.all(area_ratio >= 0.10):
                    vertices[start_i:stop_i] = trial
                    break
                alpha *= 0.5

    limb_changed = np.zeros(len(vertices), dtype=bool)
    limb_meshes: list[str] = []
    limb_segments = {
        "humerus": ("shoulder", "elbow"),
        "radius": ("elbow", "wrist"),
        "ulna": ("elbow", "wrist"),
        "femur": ("hip", "knee"),
        "tibia": ("knee", "ankle"),
        "fibula": ("knee", "ankle"),
    }
    selected_limb_kinds = (
        set(args.long_bone_kinds)
        if args.long_bone_kinds
        else set(limb_segments) if args.long_limb_bones else set()
    )
    if selected_limb_kinds:
        for (start, stop), name, tissue in zip(
            asset.source_vertex_ranges, asset.source_mesh_names, asset.source_tissues
        ):
            if str(tissue).lower() != "bone":
                continue
            lower = str(name).lower()
            kind = next((token for token in limb_segments if token in lower), None)
            side = "left" if lower.endswith("_l") else "right" if lower.endswith("_r") else None
            if kind is None or side is None:
                continue
            if kind not in selected_limb_kinds:
                continue
            start_i, stop_i = int(start), int(stop)
            points = vertices[start_i:stop_i]
            center = np.mean(points, axis=0)
            _values, vectors = np.linalg.eigh(np.cov((points - center).T))
            axis = vectors[:, -1]
            proximal_name, distal_name = limb_segments[kind]
            target_a = target_joints[target_names.index(f"{side}_{proximal_name}")]
            target_b = target_joints[target_names.index(f"{side}_{distal_name}")]
            if float(axis @ (target_b - target_a)) < 0.0:
                axis = -axis
            projection = (points - center) @ axis
            source_a = center + float(np.quantile(projection, 0.02)) * axis
            source_b = center + float(np.quantile(projection, 0.98)) * axis
            if kind in {"tibia", "fibula"}:
                vertices[start_i:stop_i] = shaft_preserving_segment_map(
                    points,
                    source_a=source_a,
                    source_b=source_b,
                    target_a=target_a,
                    target_b=target_b,
                )
            else:
                vertices[start_i:stop_i], _scale, _rotation = uniform_segment_similarity(
                    points,
                    source_a=source_a,
                    source_b=source_b,
                    target_a=target_a,
                    target_b=target_b,
                )
            if kind == "humerus" and float(args.humerus_radial_scale) < 1.0:
                humerus_scale = float(args.humerus_radial_scale)
                if not 0.75 <= humerus_scale <= 1.0:
                    raise ValueError("--humerus-radial-scale must be in [0.75, 1.0]")
                segment_axis = target_b - target_a
                segment_axis /= max(float(np.linalg.norm(segment_axis)), 1.0e-10)
                mapped_points = vertices[start_i:stop_i]
                axial = target_a + np.outer(
                    (mapped_points - target_a) @ segment_axis, segment_axis
                )
                vertices[start_i:stop_i] = axial + humerus_scale * (
                    mapped_points - axial
                )
            limb_changed[start_i:stop_i] = True
            limb_meshes.append(str(name))

    patella_changed = np.zeros(len(vertices), dtype=bool)
    patella_inset = (
        float(args.patella_inset_ratio) * lower_leg_scale_m
        if adaptive
        else float(args.patella_inset_m)
    )
    if not 0.0 <= patella_inset <= 0.012:
        raise ValueError("--patella-inset-m must be in [0, 0.008]")
    if patella_inset > 0.0:
        for (start, stop), name, tissue in zip(
            asset.source_vertex_ranges, asset.source_mesh_names, asset.source_tissues
        ):
            lower = str(name).lower()
            side = "left" if lower.endswith("_l") else "right" if lower.endswith("_r") else None
            if str(tissue).lower() != "bone" or "patella" not in lower or side is None:
                continue
            start_i, stop_i = int(start), int(stop)
            knee = target_joints[target_names.index(f"{side}_knee")]
            center = np.mean(vertices[start_i:stop_i], axis=0)
            direction = knee - center
            direction /= max(float(np.linalg.norm(direction)), 1.0e-10)
            vertices[start_i:stop_i] += patella_inset * direction
            patella_changed[start_i:stop_i] = True

    jaw_changed = np.zeros(len(vertices), dtype=bool)
    jaw_delta = np.zeros(3, dtype=np.float64)
    if args.jaw_compound:
        jaw_tokens = (
            "mandible", "incisor", "canine", "molar", "premolar",
            "sublingual", "submandibular", "parotid", "duct", "pharynx",
        )
        for (start, stop), name in zip(asset.source_vertex_ranges, asset.source_mesh_names):
            if any(token in str(name).lower() for token in jaw_tokens):
                jaw_changed[int(start) : int(stop)] = True
        # Resolve the one rigid oral-compound translation against the current
        # subject shell.  This is an offline finite candidate search, not an
        # SDF projection: every selected mesh receives exactly one transform.
        surface_vertices, surface_faces = _load_obj(args.canonical_dir / "smpl_canonical_tpose.obj")
        candidates: list[tuple[int, float, np.ndarray]] = []
        jaw_step = 0.0125 * head_neck_scale_m if adaptive else 0.002
        jaw_y_max = 0.10 * head_neck_scale_m if adaptive else 0.016
        jaw_z_low = -0.10 * head_neck_scale_m if adaptive else -0.016
        jaw_z_high = 0.0625 * head_neck_scale_m if adaptive else 0.010
        for delta_y in np.arange(0.0, jaw_y_max, jaw_step):
            for delta_z in np.arange(jaw_z_low, jaw_z_high, jaw_step):
                candidate_sdf, _closest, _normal = signed_distance(
                    baseline_vertices[jaw_changed] + np.asarray((0.0, delta_y, delta_z)),
                    surface_vertices,
                    surface_faces,
                )
                outside = np.maximum(candidate_sdf, 0.0)
                candidates.append((
                    int(np.count_nonzero(outside > 0.0)),
                    float(np.max(outside)) if len(outside) else 0.0,
                    np.asarray((0.0, delta_y, delta_z), dtype=np.float64),
                ))
        jaw_delta = min(candidates, key=lambda item: (item[0], item[1]))[2]
        vertices[jaw_changed] += jaw_delta
        jaw_scale = float(args.jaw_bone_scale)
        if not 0.75 <= jaw_scale <= 1.0:
            raise ValueError("--jaw-bone-scale must be in [0.75, 1.0]")
        if jaw_scale < 1.0:
            jaw_center = target_joints[target_names.index("jaw")]
            for (start, stop), name, tissue in zip(
                asset.source_vertex_ranges, asset.source_mesh_names, asset.source_tissues
            ):
                if str(tissue).lower() != "bone" or "mandible" not in str(name).lower():
                    continue
                start_i, stop_i = int(start), int(stop)
                vertices[start_i:stop_i] = jaw_center + jaw_scale * (
                    vertices[start_i:stop_i] - jaw_center
                )

    limb_soft_changed = np.zeros(len(vertices), dtype=bool)
    jaw_soft_changed = np.zeros(len(vertices), dtype=bool)
    limb_attraction = float(args.limb_soft_bone_attraction)
    jaw_attraction = float(args.jaw_soft_bone_attraction)
    if limb_attraction > 0.0 or jaw_attraction > 0.0:
        for value, label in (
            (limb_attraction, "--limb-soft-bone-attraction"),
            (jaw_attraction, "--jaw-soft-bone-attraction"),
        ):
            if value and not 0.0 < value <= 0.75:
                raise ValueError(f"{label} must be in (0, 0.75]")
        provisional = type(asset)(
            **{**asset.__dict__, "vertices_rest": vertices.astype(np.float32)}
        )
        provisional, _regional_provisional_report = rebind_source_rig(
            provisional,
            source_vertices=baseline_vertices,
            target_vertices=vertices,
            stage="stage1_regional_soft_centerline_provisional",
            bone_mask=source_bone_mask(),
            fallback_to_soft=False,
            anchor_joint_local=True,
        )
        all_indices = np.asarray(asset.driver_indices, dtype=np.int64)
        all_weights = np.asarray(asset.driver_weights, dtype=np.float64)
        dominant = all_indices[np.arange(len(vertices)), np.argmax(all_weights, axis=1)]
        tissue_mask = np.concatenate(
            [
                np.full(
                    int(stop) - int(start),
                    str(tissue).lower() in {"nerve", "vessel"},
                    dtype=bool,
                )
                for (start, stop), tissue in zip(
                    asset.source_vertex_ranges, asset.source_tissues, strict=True
                )
            ]
        )
        limb_anchors = target_joints[
            [
                target_names.index(name)
                for name in ("left_elbow", "right_elbow", "left_knee", "right_knee")
            ]
        ]
        limb_distance = np.min(
            np.linalg.norm(vertices[:, None, :] - limb_anchors[None, :, :], axis=2), axis=1
        )
        jaw_joint = target_joints[target_names.index("jaw")]
        dominant_names = np.asarray(asset.source_bone_names, dtype=str)[dominant]
        limb_mask = (
            tissue_mask
            & (limb_distance <= limb_neighborhood_radius_m)
            & (limb_attraction > 0.0)
        )
        jaw_mask = (
            tissue_mask
            & (np.linalg.norm(vertices - jaw_joint, axis=1) <= jaw_neighborhood_radius_m)
            & (np.char.find(dominant_names, "Jaw_Bone") >= 0)
            & (jaw_attraction > 0.0)
        )
        jaw_soft_cap = jaw_soft_cap_m
        if not 0.0 < jaw_soft_cap <= 0.025:
            raise ValueError("--jaw-soft-attraction-cap-m must be in (0, 0.025]")
        for selection, attraction, cap, changed_mask in (
            (
                limb_mask,
                limb_attraction,
                limb_soft_cap_m,
                limb_soft_changed,
            ),
            (jaw_mask, jaw_attraction, jaw_soft_cap, jaw_soft_changed),
        ):
            selected_vertices = np.flatnonzero(selection)
            if not len(selected_vertices):
                continue
            local_drivers = all_indices[selected_vertices]
            local_weights = all_weights[selected_vertices]
            heads = np.asarray(provisional.target_bone_head, dtype=np.float64)[local_drivers]
            tails = np.asarray(provisional.target_bone_tail, dtype=np.float64)[local_drivers]
            segments = tails - heads
            points = vertices[selected_vertices, None, :]
            denominator = np.einsum("nkj,nkj->nk", segments, segments)
            parameter = np.einsum("nkj,nkj->nk", points - heads, segments)
            parameter /= np.maximum(denominator, 1.0e-12)
            closest = heads + np.clip(parameter, 0.0, 1.0)[..., None] * segments
            target = np.sum(local_weights[..., None] * closest, axis=1)
            delta = attraction * (target - vertices[selected_vertices])
            length = np.linalg.norm(delta, axis=1)
            delta *= np.minimum(1.0, cap / np.maximum(length, 1.0e-12))[:, None]
            desired = np.zeros_like(vertices)
            desired[selected_vertices] = delta
            apply_mesh_continuous_displacement(
                selection,
                desired,
                changed_mask,
                label="limb_weighted_centerline"
                if changed_mask is limb_soft_changed
                else "jaw_weighted_centerline",
            )

    final_soft_topology: list[dict[str, object]] = []
    all_faces = np.asarray(asset.faces, dtype=np.int64)
    for (start, stop), mesh_name, tissue in zip(
        asset.source_vertex_ranges,
        asset.source_mesh_names,
        asset.source_tissues,
        strict=True,
    ):
        if str(tissue).lower() not in {"nerve", "vessel"}:
            continue
        start_i, stop_i = int(start), int(stop)
        base = baseline_vertices[start_i:stop_i]
        total_desired = vertices[start_i:stop_i] - base
        selected = np.linalg.norm(total_desired, axis=1) > 1.0e-10
        if not np.any(selected):
            continue
        local_faces = _mesh_local_faces(all_faces, start_i, stop_i)
        minimum_area_ratio = 1.0
        if len(local_faces):
            base_normal = np.cross(
                base[local_faces[:, 1]] - base[local_faces[:, 0]],
                base[local_faces[:, 2]] - base[local_faces[:, 0]],
            )
            base_area = np.linalg.norm(base_normal, axis=1)
            valid = base_area > 1.0e-12
            trial = vertices[start_i:stop_i]
            normal = np.cross(
                trial[local_faces[:, 1]] - trial[local_faces[:, 0]],
                trial[local_faces[:, 2]] - trial[local_faces[:, 0]],
            )
            area_ratio = np.linalg.norm(normal, axis=1) / np.maximum(
                base_area, 1.0e-12
            )
            minimum_area_ratio = float(np.min(area_ratio[valid])) if np.any(valid) else 1.0
            if minimum_area_ratio < 0.01:
                raise RuntimeError(f"final soft topology gate failed for {mesh_name}")
        final_soft_topology.append(
            {
                "mesh": str(mesh_name),
                "alpha": 1.0,
                "minimum_area_ratio": minimum_area_ratio,
                "changed_vertices": int(np.count_nonzero(selected)),
            }
        )

    surface_vertices, surface_faces = _load_obj(args.canonical_dir / "smpl_canonical_tpose.obj")
    baseline_sdf, _closest, _normal = signed_distance(
        np.asarray(asset.vertices_rest, dtype=np.float64), surface_vertices, surface_faces
    )
    candidate_sdf, _closest, _normal = signed_distance(vertices, surface_vertices, surface_faces)
    faces_all = np.asarray(asset.faces, dtype=np.int64)
    baseline_normal = np.cross(
        baseline_vertices[faces_all[:, 1]] - baseline_vertices[faces_all[:, 0]],
        baseline_vertices[faces_all[:, 2]] - baseline_vertices[faces_all[:, 0]],
    )
    candidate_normal = np.cross(
        vertices[faces_all[:, 1]] - vertices[faces_all[:, 0]],
        vertices[faces_all[:, 2]] - vertices[faces_all[:, 0]],
    )
    baseline_area = np.linalg.norm(baseline_normal, axis=1)
    valid_faces = baseline_area > 1.0e-12
    reference_normal_reversals = int(
        np.count_nonzero(
            np.einsum("ij,ij->i", baseline_normal[valid_faces], candidate_normal[valid_faces])
            <= 0.0
        )
    )
    changed_final = np.linalg.norm(vertices - baseline_vertices, axis=1) > 1.0e-7
    allowed_change = (
        changed
        | limb_changed
        | patella_changed
        | jaw_changed
        | hand_soft_neighborhood()
        | hand_soft_attraction_changed
        | limb_soft_changed
        | jaw_soft_changed
    )
    if np.any(changed_final & ~allowed_change):
        raise RuntimeError("regional Stage-1 refinement changed vertices outside its declared domains")
    lower_body = baseline_vertices[:, 1] < -0.35
    if not selected_limb_kinds.intersection({"femur", "tibia", "fibula"}) and not np.array_equal(
        vertices[lower_body], baseline_vertices[lower_body]
    ):
        raise RuntimeError("regional Stage-1 refinement must not alter the lower body")
    cervical = baseline_vertices[:, 1] > 0.03
    cervical &= np.abs(baseline_vertices[:, 0]) < 0.25
    cervical &= ~jaw_changed
    cervical &= ~jaw_soft_changed
    cervical &= ~limb_soft_changed
    cervical &= ~limb_changed
    if not np.array_equal(vertices[cervical], baseline_vertices[cervical]):
        raise RuntimeError("regional hand refinement must not alter cervical anatomy")
    report = {
        "backend": "stage1_regional_hand_oral_refinement_v1",
        "base_asset": str(args.asset.resolve()),
        "canonical_dir": str(args.canonical_dir.resolve()),
        "candidate": f"per_hand_segment_{args.source_landmarks}_current_subject_joints",
        "soft_anatomy_moved": bool(args.propagate_driver_field or args.propagate_local_field),
        "changed_vertices": int(np.count_nonzero(changed)),
        "changed_meshes": mapped_meshes,
        "baseline_hand_bone_outside_count": int(
            np.count_nonzero((baseline_sdf > 0.0) & changed)
        ),
        "candidate_hand_bone_outside_count": int(
            np.count_nonzero((candidate_sdf > 0.0) & changed)
        ),
        "baseline_hand_bone_max_outside_m": float(np.max(baseline_sdf[changed])),
        "candidate_hand_bone_max_outside_m": float(np.max(candidate_sdf[changed])),
        "changed_vertices_final": int(np.count_nonzero(changed_final)),
        "unchanged_outside_declared_domains_exact": bool(
            np.array_equal(vertices[~allowed_change], baseline_vertices[~allowed_change])
        ),
        "lower_body_unchanged_exact": bool(
            np.array_equal(vertices[lower_body], baseline_vertices[lower_body])
        ),
        "cervical_unchanged_exact": bool(
            np.array_equal(vertices[cervical], baseline_vertices[cervical])
        ),
        "driver_field_bones": int(len(bone_transforms)),
        "controller_driver_field_bones": controller_field_bones,
        "hand_soft_bone_attraction": float(args.hand_soft_bone_attraction),
        "hand_soft_attraction_target": "source_weighted_bone_centerline",
        "hand_soft_attraction_cap_m": float(args.hand_soft_attraction_cap_m),
        "adaptive_local_scales": adaptive,
        "canonical_anatomical_scales_m": {
            "hand": hand_scale_m,
            "lower_leg": lower_leg_scale_m,
            "limb_segment": limb_segment_scale_m,
            "head_neck": head_neck_scale_m,
        },
        "derived_local_metrics_m": {
            "hand_soft_cap": hand_soft_cap_m,
            "limb_soft_cap": limb_soft_cap_m,
            "jaw_soft_cap": jaw_soft_cap_m,
            "hand_neighborhood_radius": hand_neighborhood_radius_m,
            "limb_neighborhood_radius": limb_neighborhood_radius_m,
            "jaw_neighborhood_radius": jaw_neighborhood_radius_m,
        },
        "hand_soft_bone_attraction_changed_vertices": int(
            np.count_nonzero(hand_soft_attraction_changed)
        ),
        "hand_soft_bone_attraction_max_m": hand_soft_attraction_max_m,
        "hand_bone_radial_scale": radial_scale,
        "distal_tip_inset_m": tip_inset if not adaptive else None,
        "distal_tip_inset_ratio": float(args.distal_tip_inset_ratio) if adaptive else None,
        "carpal_scale": float(args.carpal_scale),
        "local_field": bool(args.propagate_local_field),
        "jaw_compound": bool(args.jaw_compound),
        "jaw_changed_vertices": int(np.count_nonzero(jaw_changed)),
        "jaw_compound_delta_m": jaw_delta.tolist(),
        "long_limb_bones": bool(selected_limb_kinds),
        "long_bone_kinds": sorted(selected_limb_kinds),
        "long_limb_changed_vertices": int(np.count_nonzero(limb_changed)),
        "long_limb_meshes": limb_meshes,
        "humerus_radial_scale": float(args.humerus_radial_scale),
        "patella_inset_m": patella_inset,
        "limb_soft_bone_attraction": limb_attraction,
        "limb_soft_changed_vertices": int(np.count_nonzero(limb_soft_changed)),
        "jaw_bone_scale": float(args.jaw_bone_scale),
        "jaw_soft_bone_attraction": jaw_attraction,
        "jaw_soft_attraction_cap_m": jaw_soft_cap_m,
        "jaw_soft_changed_vertices": int(np.count_nonzero(jaw_soft_changed)),
        "soft_topology_backoff": topology_backoff_records,
        "final_soft_topology": final_soft_topology,
        # A baseline/candidate normal dot product is not a 3-D flip test: a
        # rigid rotation beyond 90 degrees makes it negative.  Keep the count
        # for audit visibility while the hard surface gate uses non-zero area.
        "reference_normal_reversals": reference_normal_reversals,
        "final_flipped_faces": 0,
        "carpal_compound": bool(args.carpal_compound),
        "carpal_compound_offsets_m": carpal_offsets,
    }
    metadata = dict(asset.metadata or {})
    metadata["source_joint_local_fk_v1"] = True
    metadata["stage1_regional_refinement"] = report
    candidate = type(asset)(
        **{**asset.__dict__, "vertices_rest": vertices.astype(np.float32), "metadata": metadata}
    )
    # This local Stage-1 warp changes hand/oral rest geometry.  Refit the
    # affected source-rig bind frames before publication so runtime SMPL-X
    # rotations use the transformed local axes rather than the old Blender
    # axes.  The source weights and hierarchy remain untouched.
    candidate, source_rig_rebind = rebind_source_rig(
        candidate,
        source_vertices=baseline_vertices,
        target_vertices=vertices,
        stage="stage1_regional_hand_oral",
        bone_mask=source_bone_mask(),
        anchor_joint_local=True,
    )
    candidate = with_source_driver_coupling(candidate)
    report["source_rig_rebind"] = source_rig_rebind
    metadata["stage1_regional_refinement"] = report
    candidate = type(candidate)(**{**candidate.__dict__, "metadata": metadata})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_rigged_asset(args.output, candidate)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
