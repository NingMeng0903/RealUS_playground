from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from common.types import MotionSequenceManifest

# Blender FBX export uses this scale so root motion (meters) matches Unreal world units (cm).
EXPECTED_FBX_GLOBAL_SCALE_FOR_UE = 100.0


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)


def _resolve_manifest_path(manifest_path: Path, stored_path: str | None) -> Path:
    if stored_path is None:
        raise FileNotFoundError(f"Manifest entry is missing a path: {manifest_path}")
    candidate = Path(stored_path)
    return candidate if candidate.is_absolute() else (manifest_path.parent / candidate).resolve()


def _payload_global_and_body_pose(payload: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    if "global_orient" in payload.files and "body_pose" in payload.files:
        return payload["global_orient"], payload["body_pose"]
    if "smplx_global_orient" in payload.files and "smplx_body_pose" in payload.files:
        return payload["smplx_global_orient"], payload["smplx_body_pose"]
    raise KeyError(
        "Expected global_orient+body_pose or smplx_global_orient+smplx_body_pose in NPZ payload."
    )


def _payload_betas(payload: np.lib.npyio.NpzFile) -> np.ndarray:
    if "betas" in payload.files:
        return np.asarray(payload["betas"], dtype=np.float32)
    if "smplx_betas" in payload.files:
        return np.asarray(payload["smplx_betas"], dtype=np.float32)
    raise KeyError("Expected betas or smplx_betas in NPZ payload.")


def _resolve_faces_npy(manifest_path: Path, manifest: MotionSequenceManifest) -> Path:
    if manifest.faces_path:
        candidate = _resolve_manifest_path(manifest_path, manifest.faces_path)
        if candidate.is_file():
            return candidate
    for ancestor in manifest_path.resolve().parents:
        world_faces = (
            ancestor / "world" / manifest.backend_name / manifest.sequence_id / "smpl_faces.npy"
        )
        if world_faces.is_file():
            return world_faces
        for name in ("smpl_faces.npy", "smplx_faces.npy"):
            sidecar = ancestor / name
            if sidecar.is_file():
                return sidecar
    raise FileNotFoundError(
        f"Could not find smpl_faces.npy for manifest {manifest_path} (faces_path={manifest.faces_path!r})."
    )


def _rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotmat, dtype=np.float32).reshape(3, 3)
    trace = float(np.trace(rot))
    cos_theta = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    if np.pi - theta < 1e-5:
        diag = np.clip((np.diag(rot) + 1.0) * 0.5, 0.0, None)
        axis = np.sqrt(diag)
        if axis[0] > 1e-5:
            axis[1] = np.copysign(axis[1], rot[0, 1] + rot[1, 0])
            axis[2] = np.copysign(axis[2], rot[0, 2] + rot[2, 0])
        elif axis[1] > 1e-5:
            axis[2] = np.copysign(axis[2], rot[1, 2] + rot[2, 1])
        axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
        return (axis * theta).astype(np.float32)
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(theta))
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    return (axis * theta).astype(np.float32)


def _axis_angle_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    vec = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z = vec / theta
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float32,
    )


def _ingest_global_orient_rotmat(raw: np.ndarray) -> np.ndarray:
    """Accept axis-angle (3,) or rotation matrix (3,3) / (9,)."""
    g = np.asarray(raw, dtype=np.float32).reshape(-1)
    if g.size == 3:
        return _axis_angle_to_rotmat(g)
    if g.size == 9:
        return g.reshape(3, 3).astype(np.float32)
    raise ValueError(f"global_orient must be 3 (aa) or 9 (rotmat), got size {g.size}")


def _ingest_body_pose_rotmats(raw: np.ndarray) -> np.ndarray:
    """Return [23,3,3] body pose rotation matrices (SMPL convention, excludes root)."""
    b = np.asarray(raw, dtype=np.float32)
    if b.ndim == 3 and b.shape[-2:] == (3, 3):
        mats = b.reshape(-1, 3, 3)
        if mats.shape[0] == 23:
            return mats.astype(np.float32)
        if mats.shape[0] == 21:
            pad = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
            return np.concatenate([mats.astype(np.float32), pad], axis=0)
        raise ValueError(f"body_pose rotmat stack must be 21 or 23, got {mats.shape[0]}")
    flat = b.reshape(-1)
    if flat.size == 23 * 9:
        return flat.reshape(23, 3, 3).astype(np.float32)
    if flat.size == 21 * 9:
        mats21 = flat.reshape(21, 3, 3).astype(np.float32)
        pad = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
        return np.concatenate([mats21, pad], axis=0)
    if flat.size % 3 != 0:
        raise ValueError(f"body_pose axis-angle length must be multiple of 3, got {flat.size}")
    n = flat.size // 3
    vecs = flat.reshape(n, 3)
    mats = np.stack([_axis_angle_to_rotmat(vecs[i]) for i in range(n)], axis=0)
    if mats.shape[0] == 23:
        return mats
    if mats.shape[0] == 21:
        pad = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
        return np.concatenate([mats, pad], axis=0)
    raise ValueError(
        f"body_pose axis-angle must describe 21 or 23 joints (got {n} joints from {flat.size} params)"
    )


def _rotmat_to_quaternion(rotmat: np.ndarray) -> np.ndarray:
    m = np.asarray(rotmat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-8)) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-8)) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-8)) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float32)
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    return quat


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for vertex in np.asarray(vertices, dtype=np.float32):
            handle.write(f"v {float(vertex[0]):.8f} {float(vertex[1]):.8f} {float(vertex[2]):.8f}\n")
        for face in np.asarray(faces, dtype=np.int32):
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def export_smpl_motion(
    manifest_path: Path,
    output_dir: Path,
    export_obj_sequence: bool = False,
    include_vertices: bool = False,
) -> tuple[Path, Path]:
    manifest = MotionSequenceManifest.load(manifest_path)
    if not manifest.frames:
        raise RuntimeError(f"No frames found in manifest: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    faces_path = _resolve_faces_npy(manifest_path, manifest)
    faces = np.load(faces_path).astype(np.int32)
    np.save(output_dir / "smpl_faces.npy", faces)

    joint_rotmats = []
    joint_quats = []
    joint_axis_angles = []
    root_translation = []
    frame_indices = []
    timestamps = []
    frame_stems = []
    world_vertices = []
    bed_plane_normals = []
    bed_plane_offsets = []
    support_plane_shifts = []
    world_grounded = []

    obj_dir = output_dir / "obj_sequence"
    if export_obj_sequence:
        obj_dir.mkdir(parents=True, exist_ok=True)

    for frame in manifest.frames:
        payload_path = _resolve_manifest_path(manifest_path, frame.world_smpl_path)
        payload = np.load(payload_path)
        if "transl_world" not in payload:
            raise KeyError(f"{payload_path} is missing transl_world.")

        g_raw, b_raw = _payload_global_and_body_pose(payload)
        root_rot = _ingest_global_orient_rotmat(g_raw).reshape(1, 3, 3)
        body_pose = _ingest_body_pose_rotmats(b_raw)
        all_rotmats = np.concatenate([root_rot, body_pose], axis=0).astype(np.float32)
        all_quats = np.stack([_rotmat_to_quaternion(rot) for rot in all_rotmats], axis=0).astype(np.float32)
        all_axis_angles = np.stack([_rotmat_to_axis_angle(rot) for rot in all_rotmats], axis=0).astype(np.float32)

        joint_rotmats.append(all_rotmats)
        joint_quats.append(all_quats)
        joint_axis_angles.append(all_axis_angles)
        root_translation.append(np.asarray(payload["transl_world"], dtype=np.float32).reshape(3))
        frame_indices.append(int(frame.frame_idx))
        timestamps.append(float(frame.timestamp_s))
        frame_stems.append(frame.frame_stem)
        world_grounded.append(bool(payload["world_grounded"]) if "world_grounded" in payload else False)

        if "bed_plane_normal" in payload:
            bed_plane_normals.append(np.asarray(payload["bed_plane_normal"], dtype=np.float32).reshape(3))
        if "bed_plane_offset" in payload:
            bed_plane_offsets.append(float(np.asarray(payload["bed_plane_offset"]).reshape(())))
        if "support_plane_shift" in payload:
            support_plane_shifts.append(float(np.asarray(payload["support_plane_shift"]).reshape(())))

        if include_vertices or export_obj_sequence:
            vertices = np.asarray(payload["vertices_world"], dtype=np.float32)
            if include_vertices:
                world_vertices.append(vertices)
            if export_obj_sequence:
                _write_obj(obj_dir / f"{frame.frame_stem}_{frame.person_id}.obj", vertices, faces)

    first_frame_path = _resolve_manifest_path(manifest_path, manifest.frames[0].world_smpl_path)
    betas_arr = _payload_betas(np.load(first_frame_path))

    motion_npz_path = output_dir / "smpl_motion_bundle.npz"
    npz_payload = {
        "joint_rotmats": np.stack(joint_rotmats, axis=0).astype(np.float32),
        "joint_quaternions_wxyz": np.stack(joint_quats, axis=0).astype(np.float32),
        "joint_axis_angles": np.stack(joint_axis_angles, axis=0).astype(np.float32),
        "root_translation_world": np.stack(root_translation, axis=0).astype(np.float32),
        "betas": betas_arr,
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "timestamps_s": np.asarray(timestamps, dtype=np.float32),
        "frame_stems": np.asarray(frame_stems),
        "joint_names": np.asarray(SMPL_JOINT_NAMES),
        "joint_parents": SMPL_PARENTS.astype(np.int32),
        "world_grounded": np.asarray(world_grounded, dtype=bool),
    }
    if bed_plane_normals:
        npz_payload["bed_plane_normals"] = np.stack(bed_plane_normals, axis=0).astype(np.float32)
    if bed_plane_offsets:
        npz_payload["bed_plane_offsets"] = np.asarray(bed_plane_offsets, dtype=np.float32)
    if support_plane_shifts:
        npz_payload["support_plane_shifts"] = np.asarray(support_plane_shifts, dtype=np.float32)
    if include_vertices:
        npz_payload["vertices_world"] = np.stack(world_vertices, axis=0).astype(np.float32)
    np.savez(motion_npz_path, **npz_payload)

    metadata = {
        "source_manifest": str(manifest_path),
        "source_video": manifest.source_video,
        "sequence_id": manifest.sequence_id,
        "backend_name": manifest.backend_name,
        "representation": manifest.representation,
        "fps": float(manifest.fps),
        "frame_count": len(manifest.frames),
        "joint_count": len(SMPL_JOINT_NAMES),
        "joint_names": SMPL_JOINT_NAMES,
        "joint_parents": SMPL_PARENTS.tolist(),
        "rotation_conventions": {
            "joint_rotmats": "[T,24,3,3]",
            "joint_quaternions_wxyz": "[T,24,4]",
            "joint_axis_angles": "[T,24,3]",
        },
        "translation_convention": "root_translation_world is in the same world coordinate frame as the input manifest.",
        "obj_sequence_dir": str(obj_dir) if export_obj_sequence else None,
        "faces_path": str(output_dir / "smpl_faces.npy"),
        "notes": [
            "This bundle keeps SMPL joint ordering and is intended for retargeting into Blender/FBX/BVH pipelines.",
            "BVH/FBX export is not performed here; this script writes a stable intermediate package for downstream DCC conversion.",
            "OBJ sequence uses the same topology for all frames and is suitable for Alembic-style mesh cache conversion.",
            "If the manifest used lightweight endpoint refit: vertices_world / joints3d_world / transl_world were rigidly updated, "
            "but global_orient and body_pose in each NPZ usually remain the original GVHMR (camera-space) parameters. "
            "For mesh-accurate Blender motion, prefer --export-obj-sequence or --include-vertices; armature-from-params may not match overlay.",
        ],
    }
    metadata_path = output_dir / "smpl_motion_bundle.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return motion_npz_path, metadata_path


def _numpy_global_root_z_floor_align(
    raw_root: np.ndarray,
    world_offset: tuple[float, float, float],
) -> np.ndarray:
    """Torch-free stand-in: subtract sequence-wide 2nd percentile of root z, then add world offset (not per-frame mesh floor)."""
    out = np.asarray(raw_root[:, :3], dtype=np.float32).copy()
    p = float(np.percentile(out[:, 2].astype(np.float64), 2.0))
    out[:, 2] -= np.float32(p)
    out += np.asarray(world_offset, dtype=np.float32).reshape(1, 3)
    return out


def _scene_fit_world_offset(sequence, scene_spec_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    from common.project import project_paths
    from projects.genesis_ue_sync.sim_platform.embodiments.smpl_capsule_runtime import prepare_smpl_capsule_runtime_asset
    from projects.genesis_ue_sync.sim_platform.human_refit.placement_resolver import (
        genesis_mesh_world_offset_m_from_placement,
        try_load_human_scene_placement_for_scene,
    )
    from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec
    from projects.genesis_ue_sync.sim_platform.scenes.human_bed_fit import fit_human_sequence_to_bed

    resolved_scene_spec = Path(scene_spec_path).expanduser().resolve()
    scene_spec = load_sync_scene_spec(resolved_scene_spec)
    repo = project_paths(__file__).root
    loaded = try_load_human_scene_placement_for_scene(scene_spec, repo_root=repo)
    if loaded is not None:
        ox, oy, oz = genesis_mesh_world_offset_m_from_placement(loaded)
        off = np.asarray([ox, oy, oz], dtype=np.float32).reshape(3)
        return (
            off,
            {
                "scene_fit_spec_path": str(resolved_scene_spec),
                "scene_fit_source": "human_scene_placement_json",
                "scene_fit_placement_revision": str(loaded.scene_fit_revision),
            },
        )
    capsule_asset = prepare_smpl_capsule_runtime_asset(
        sequence,
        cache_dir=repo / "outputs" / "genesis_capsule_urdf_cache",
        device="cpu",
        force_rewrite=False,
    )
    placement = fit_human_sequence_to_bed(
        sequence,
        scene_spec=scene_spec,
        proxy_geometry=capsule_asset.proxy_geometry,
        device="cpu",
        sample_count=1,
    )
    off = np.asarray(placement.world_offset, dtype=np.float32).reshape(3)
    off = off + np.asarray((0.0, 0.0, float(scene_spec.human.display_vertical_offset_m)), dtype=np.float32)
    return (
        off,
        {
            "scene_fit_spec_path": str(resolved_scene_spec),
            "scene_fit_support_plane_z_m": float(placement.support_plane_z),
            "scene_fit_support_contact_ratio": float(placement.support_contact_ratio),
            "scene_fit_lower_shell_snap_dz_m": float(placement.lower_shell_snap_dz_m),
            "scene_fit_sample_indices": [int(i) for i in placement.sample_indices],
            "scene_fit_placement_revision": 3,
            "scene_human_display_vertical_sink_m": float(scene_spec.human.display_vertical_sink_m),
            "scene_human_display_vertical_offset_m": float(scene_spec.human.display_vertical_offset_m),
            "scene_human_display_pitch_forward_deg": float(scene_spec.human.display_pitch_forward_deg),
        },
    )


def _ue_world_from_genesis_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32).copy()
    arr[..., 1] *= -1.0
    return arr


def export_smpl_motion_sequence(
    sequence_path: Path,
    output_dir: Path,
    export_obj_sequence: bool = False,
    include_vertices: bool = False,
    *,
    world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    align_floor: bool = False,
    max_frames: int | None = None,
    scene_spec_path: Path | None = None,
    output_world: str = "genesis",
) -> tuple[Path, Path]:
    from projects.genesis_ue_sync.sim_platform.datasets import HumanMotionSequence

    sequence = HumanMotionSequence.load(sequence_path)
    if max_frames is not None and int(max_frames) > 0:
        cap = min(sequence.frame_count, int(max_frames))
        if cap < int(max_frames):
            print(
                f"export_smpl_motion: --max-frames {max_frames} clamped to sequence length {cap}",
                file=sys.stderr,
            )
        cam_i = sequence.cam_int[:cap] if sequence.cam_int is not None else None
        cam_e = sequence.cam_ext[:cap] if sequence.cam_ext is not None else None
        names = list(sequence.image_names[:cap]) if sequence.image_names else []
        sequence = dataclasses.replace(
            sequence,
            poses=np.asarray(sequence.poses[:cap], dtype=np.float32),
            trans=np.asarray(sequence.trans[:cap], dtype=np.float32),
            image_names=names,
            cam_int=cam_i,
            cam_ext=cam_e,
        )
    if sequence.model_type.lower() != "smpl":
        raise ValueError(f"Only SMPL sequences are supported for direct bundle export, got {sequence.model_type}.")
    if sequence.poses.shape[1] < 72:
        raise ValueError(f"Expected at least 72 pose parameters for SMPL, got shape {sequence.poses.shape}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pose_axis_angles = np.asarray(sequence.poses[:, :72], dtype=np.float32).reshape(sequence.frame_count, 24, 3)
    joint_rotmats = np.stack(
        [[_axis_angle_to_rotmat(pose_axis_angles[frame_idx, joint_idx]) for joint_idx in range(24)] for frame_idx in range(sequence.frame_count)],
        axis=0,
    ).astype(np.float32)
    joint_quats = np.stack(
        [[_rotmat_to_quaternion(joint_rotmats[frame_idx, joint_idx]) for joint_idx in range(24)] for frame_idx in range(sequence.frame_count)],
        axis=0,
    ).astype(np.float32)

    frame_indices = np.arange(sequence.frame_count, dtype=np.int32)
    timestamps = frame_indices.astype(np.float32) / max(float(sequence.fps), 1e-6)
    frame_stems = [sequence.image_names[idx] if idx < len(sequence.image_names) else f"frame_{idx:05d}" for idx in range(sequence.frame_count)]

    raw_root = np.asarray(sequence.trans[:, :3], dtype=np.float32)
    use_genesis_match = bool(align_floor) or tuple(world_offset) != (0.0, 0.0, 0.0)
    genesis_align_floor_mode: str | None = None
    metadata_extra: dict[str, object] = {}
    if scene_spec_path is not None:
        applied_world_offset, metadata_extra = _scene_fit_world_offset(sequence, scene_spec_path)
        root_world = raw_root + applied_world_offset.reshape(1, 3)
        genesis_align_floor_mode = "scene_spec_bed_fit"
        genesis_root_mode = "scene_spec_bed_fit"
    elif use_genesis_match:
        if align_floor:
            try:
                from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
                    compute_genesis_matched_root_translation,
                )

                root_world = compute_genesis_matched_root_translation(
                    sequence, world_offset=world_offset, align_floor=True
                ).astype(np.float32)
                genesis_align_floor_mode = "mesh_percentile_torch"
                genesis_root_mode = "torch_floor"
            except (ModuleNotFoundError, ImportError) as exc:
                miss = (getattr(exc, "name", None) or "").lower()
                if miss in {"torch", "smplx"} or "torch" in str(exc).lower():
                    root_world = _numpy_global_root_z_floor_align(raw_root, world_offset)
                    genesis_align_floor_mode = "root_z_global_p2_numpy_fallback"
                    genesis_root_mode = "numpy_root_z_p2_fallback"
                    print(
                        "export_smpl_motion: align_floor uses numpy root-z fallback "
                        "(torch+smplx or AMONGUS_MOTION_PYTHON recommended for Genesis mesh floor match)",
                        file=sys.stderr,
                    )
                else:
                    raise
        else:
            off = np.asarray(world_offset, dtype=np.float32).reshape(1, 3)
            root_world = raw_root + off
            genesis_root_mode = "numpy_offset_only"
    else:
        root_world = raw_root
        genesis_root_mode = "raw"
    if scene_spec_path is None:
        applied_world_offset = np.asarray(world_offset, dtype=np.float32).reshape(3)

    genesis_canonical_offset_m = np.asarray(applied_world_offset, dtype=np.float32).reshape(3).copy()
    mesh_world_offset = tuple(float(x) for x in genesis_canonical_offset_m.tolist())

    if str(output_world).strip().lower() == "ue":
        root_world = _ue_world_from_genesis_points(root_world)
        applied_world_offset = _ue_world_from_genesis_points(genesis_canonical_offset_m.reshape(1, 3)).reshape(3)

    npz_payload = {
        "joint_rotmats": joint_rotmats,
        "joint_quaternions_wxyz": joint_quats,
        "joint_axis_angles": pose_axis_angles.astype(np.float32),
        "root_translation_world": root_world,
        "betas": np.asarray(sequence.betas, dtype=np.float32),
        "frame_indices": frame_indices,
        "timestamps_s": timestamps,
        "frame_stems": np.asarray(frame_stems),
        "joint_names": np.asarray(SMPL_JOINT_NAMES),
        "joint_parents": SMPL_PARENTS.astype(np.int32),
        "world_grounded": np.zeros(sequence.frame_count, dtype=bool),
    }

    if include_vertices or export_obj_sequence:
        from projects.genesis_ue_sync.sim_platform.datasets import build_trimesh_sequence

        meshes = build_trimesh_sequence(
            sequence,
            world_offset=mesh_world_offset,
            align_floor=False if scene_spec_path is not None else align_floor,
        )
        if str(output_world).strip().lower() == "ue":
            for mesh in meshes:
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                vertices[:, 1] *= -1.0
                mesh.vertices = vertices
        faces = np.asarray(meshes[0].faces, dtype=np.int32)
        np.save(output_dir / "smpl_faces.npy", faces)
        if include_vertices:
            npz_payload["vertices_world"] = np.stack([np.asarray(mesh.vertices, dtype=np.float32) for mesh in meshes], axis=0)
        if export_obj_sequence:
            obj_dir = output_dir / "obj_sequence"
            obj_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx, mesh in enumerate(meshes):
                _write_obj(obj_dir / f"{frame_stems[frame_idx]}.obj", np.asarray(mesh.vertices, dtype=np.float32), faces)

    motion_npz_path = output_dir / "smpl_motion_bundle.npz"
    np.savez(motion_npz_path, **npz_payload)

    metadata = {
        "source_sequence_npz": str(sequence_path),
        "source_dataset": sequence.source_dataset,
        "sequence_name": sequence.sequence_name,
        "frame_count": sequence.frame_count,
        "fps": float(sequence.fps),
        "model_type": sequence.model_type,
        "joint_count": len(SMPL_JOINT_NAMES),
        "joint_names": SMPL_JOINT_NAMES,
        "joint_parents": SMPL_PARENTS.tolist(),
        "faces_path": str(output_dir / "smpl_faces.npy") if (output_dir / "smpl_faces.npy").is_file() else None,
        "obj_sequence_dir": str(output_dir / "obj_sequence") if export_obj_sequence else None,
        "motion_bundle_format": 2,
        "genesis_world_offset_m": [float(x) for x in genesis_canonical_offset_m.tolist()],
        "offset_in_output_world_m": [float(x) for x in applied_world_offset.tolist()],
        "genesis_align_floor": bool(align_floor) if scene_spec_path is None else False,
        "genesis_align_floor_mode": genesis_align_floor_mode,
        "output_world_convention": str(output_world).strip().lower(),
        "expected_fbx_global_scale": float(EXPECTED_FBX_GLOBAL_SCALE_FOR_UE),
        "notes": [
            "This bundle was exported directly from HumanMotionSequence and preserves SMPL joint axis-angle values.",
            "It is intended for Blender/FBX animation export and UE animation import.",
            "When genesis_align_floor_mode is mesh_percentile_torch, root_translation_world matches build_trimesh_sequence.",
            "When genesis_align_floor_mode is root_z_global_p2_numpy_fallback, vertical placement is approximate (no torch).",
            "When genesis_align_floor_mode is scene_spec_bed_fit, root_translation_world matches the shared scene-spec bed fitting logic.",
        ],
    }
    metadata.update(metadata_extra)
    metadata_path = output_dir / "smpl_motion_bundle.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        stale_sidecar = output_dir / "fbx_export_sidecar.json"
        if stale_sidecar.is_file():
            stale_sidecar.unlink()
    except Exception:
        pass
    return motion_npz_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export manifest + per-frame SMPL NPZ files into a DCC-friendly motion bundle.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sequence-npz", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-obj-sequence", action="store_true")
    parser.add_argument("--include-vertices", action="store_true")
    parser.add_argument("--world-offset", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--align-floor", action="store_true", help="Match Genesis build_trimesh_sequence floor percentile + offset.")
    parser.add_argument(
        "--scene-spec",
        type=Path,
        default=None,
        help="Use shared scene-spec bed fitting to derive root_translation_world instead of plain world-offset.",
    )
    parser.add_argument(
        "--output-world",
        type=str,
        default="genesis",
        choices=["genesis", "ue"],
        help="Output root_translation_world in Genesis canonical world or UE-render world.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Trim the sequence to this many frames before export (matches scene motion.frame_count / UE render length).",
    )
    args = parser.parse_args()
    if (args.manifest is None) == (args.sequence_npz is None):
        parser.error("Provide exactly one of --manifest or --sequence-npz.")
    return args


def main() -> None:
    args = parse_args()
    if args.sequence_npz is not None:
        wo = tuple(args.world_offset) if args.world_offset is not None else (0.0, 0.0, 0.0)
        motion_npz_path, metadata_path = export_smpl_motion_sequence(
            args.sequence_npz.resolve(),
            args.output_dir.resolve(),
            export_obj_sequence=args.export_obj_sequence,
            include_vertices=args.include_vertices,
            world_offset=wo,
            align_floor=bool(args.align_floor),
            max_frames=args.max_frames,
            scene_spec_path=None if args.scene_spec is None else args.scene_spec.resolve(),
            output_world=str(args.output_world),
        )
    else:
        motion_npz_path, metadata_path = export_smpl_motion(
            args.manifest.resolve(),
            args.output_dir.resolve(),
            export_obj_sequence=args.export_obj_sequence,
            include_vertices=args.include_vertices,
        )
    print(motion_npz_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
