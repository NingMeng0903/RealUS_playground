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
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))

    from projects.genesis_ue_sync.anatomy_retarget.containment import signed_distance
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
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    baseline_vertices = vertices.copy()
    changed = np.zeros(len(vertices), dtype=bool)
    mapped_meshes: list[str] = []
    bone_transforms: dict[int, np.ndarray] = {}
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
        if args.source_landmarks == "asset_joints":
            vertices[start_i:stop_i] = shaft_preserving_segment_map(
                points,
                source_a=segment[0], source_b=segment[1],
                target_a=segment[2], target_b=segment[3],
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
                target_a=segment[2],
                target_b=segment[3],
            )
            local_indices = np.asarray(asset.driver_indices[start_i:stop_i], dtype=np.int64).reshape(-1)
            local_weights = np.asarray(asset.driver_weights[start_i:stop_i], dtype=np.float64).reshape(-1)
            mass = np.bincount(local_indices, weights=local_weights, minlength=len(asset.source_bone_names or []))
            bone = int(np.argmax(mass))
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = scale * rotation
            transform[:3, 3] = segment[2] - transform[:3, :3] @ source_a
            bone_transforms[bone] = transform
        changed[start_i:stop_i] = True
        mapped_meshes.append(str(name))

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
        return soft & (distance <= 0.090)

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
        for delta_y in np.arange(0.0, 0.016, 0.002):
            for delta_z in np.arange(-0.016, 0.010, 0.002):
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

    surface_vertices, surface_faces = _load_obj(args.canonical_dir / "smpl_canonical_tpose.obj")
    baseline_sdf, _closest, _normal = signed_distance(
        np.asarray(asset.vertices_rest, dtype=np.float64), surface_vertices, surface_faces
    )
    candidate_sdf, _closest, _normal = signed_distance(vertices, surface_vertices, surface_faces)
    changed_final = np.linalg.norm(vertices - baseline_vertices, axis=1) > 1.0e-7
    allowed_change = changed | jaw_changed | hand_soft_neighborhood()
    if np.any(changed_final & ~allowed_change):
        raise RuntimeError("regional Stage-1 refinement changed vertices outside its declared domains")
    lower_body = baseline_vertices[:, 1] < -0.35
    if not np.array_equal(vertices[lower_body], baseline_vertices[lower_body]):
        raise RuntimeError("regional Stage-1 refinement must not alter the lower body")
    cervical = baseline_vertices[:, 1] > 0.03
    cervical &= np.abs(baseline_vertices[:, 0]) < 0.25
    cervical &= ~jaw_changed
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
        "local_field": bool(args.propagate_local_field),
        "jaw_compound": bool(args.jaw_compound),
        "jaw_changed_vertices": int(np.count_nonzero(jaw_changed)),
        "jaw_compound_delta_m": jaw_delta.tolist(),
    }
    metadata = dict(asset.metadata or {})
    metadata["stage1_regional_refinement"] = report
    candidate = type(asset)(
        **{**asset.__dict__, "vertices_rest": vertices.astype(np.float32), "metadata": metadata}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_rigged_asset(args.output, candidate)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
