"""Host-side official BEDLAM retarget FBX generation (retargeting.uproject).

Must run outside the BE_IBL editor session: launching a second UnrealEditor for
retargeting while BE_IBL is handling prepare_render_pipeline causes conflicts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS.parent, *_THIS.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridge.adapters.ue import ue_world_point_from_genesis_m
from common.project import project_paths
from projects.genesis_ue_sync.config.toolchain import (
    discover_blender_executable,
    discover_python_command,
    discover_unreal_editor_executable,
)
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    HumanMotionSequence,
    npz_shape_reference_for_retarget_cache,
)
from projects.genesis_ue_sync.sim_platform.scenes import resolve_scene_spec_with_augmentation
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import SyncSceneSpec

PROJECT_PATHS = project_paths(__file__)
RETARGET_REPO = PROJECT_PATHS.bedlam_retarget_root
RETARGET_PROJECT = PROJECT_PATHS.bedlam_retarget_project_file
UE_ANIMATION_ASSET_SESSION_SCRIPT = _THIS.parent / "ue_animation_asset_session.py"
SMPLX_ADDON_REPO = PROJECT_PATHS.smplx_blender_addon_root
SMPLX_ADDON_REQUIRED_BLEND = PROJECT_PATHS.smplx_blender_required_blend
# Very short AMASS->FBX clips often produce FBX files UE imports as having no animation takes.
_MIN_OFFICIAL_NPZ_FRAMES = 24
EXPECTED_SCENE_FIT_PLACEMENT_REVISION = 3


def _python_bin() -> str:
    for candidate in ("python3", "python"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Cannot find python3/python in PATH.")


def _motion_export_python_bin() -> str:
    raw = os.environ.get("AMONGUS_MOTION_PYTHON", "").strip()
    if raw:
        cand = Path(raw)
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand.resolve())
        resolved = shutil.which(raw)
        if resolved:
            return resolved
    for candidate in (
        Path("/media/camp/EXT_DRIVE/envs/genesis/bin/python"),
        Path.home() / ".conda" / "envs" / "genesis" / "bin" / "python",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return _python_bin()


def _motion_export_python_cmd() -> list[str]:
    raw = os.environ.get("AMONGUS_MOTION_PYTHON", "").strip()
    if raw:
        return [_motion_export_python_bin()]
    python_cmd = discover_python_command()
    if python_cmd[:3] == ["conda", "run", "-n"] or (len(python_cmd) >= 4 and python_cmd[1:4] == ["run", "-n", "genesis"]):
        probe = subprocess.run(
            [*python_cmd, "-c", "import torch, smplx"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return python_cmd
    return [_motion_export_python_bin()]


def _resolve_existing_repo_path(raw_path: str) -> Path | None:
    candidate = str(raw_path or "").strip()
    if not candidate:
        return None
    p = Path(candidate)
    if not p.is_file():
        alt = REPO_ROOT / candidate
        if alt.is_file():
            p = alt
    return p.resolve() if p.is_file() else None


def _resolve_motion_input_path(motion_payload: dict) -> tuple[str, str]:
    sequence_npz_path = str(motion_payload.get("sequence_npz_path", "") or "")
    source_path = str(motion_payload.get("source_path", "") or "")
    mesh_manifest_path = str(motion_payload.get("mesh_manifest_path", "") or "")

    for raw in (sequence_npz_path, source_path):
        resolved = _resolve_existing_repo_path(raw)
        if resolved is not None:
            return "sequence_npz", str(resolved)

    resolved_manifest = _resolve_existing_repo_path(mesh_manifest_path)
    if resolved_manifest is not None:
        return "manifest", str(resolved_manifest)

    raise RuntimeError("No valid motion input exists for bundle export.")


def _build_export_smpl_command(
    export_script: Path,
    motion_payload: dict,
    bundle_dir: Path,
    *,
    scene_spec_path: Path,
) -> list[str]:
    motion_cmd = _motion_export_python_cmd()
    command = [*motion_cmd, str(export_script)]
    kind, resolved_input = _resolve_motion_input_path(motion_payload)
    if kind == "sequence_npz":
        command.extend(["--sequence-npz", resolved_input])
    else:
        command.extend(["--manifest", resolved_input])
    command.extend(["--output-dir", str(bundle_dir)])
    wo = motion_payload.get("human_world_offset_m")
    if isinstance(wo, (list, tuple)) and len(wo) == 3:
        command.extend(["--world-offset", str(float(wo[0])), str(float(wo[1])), str(float(wo[2]))])
    if bool(motion_payload.get("human_align_floor", False)):
        command.append("--align-floor")
    cap = int(motion_payload.get("frame_count", 0) or 0)
    if cap > 0:
        command.extend(["--max-frames", str(cap)])
    command.extend(["--scene-spec", str(scene_spec_path.expanduser().resolve())])
    command.extend(["--output-world", "ue"])
    return command


def _motion_payload_from_scene_spec(scene_spec: SyncSceneSpec) -> dict:
    human_anchor_m = scene_spec.resolved_human_anchor()
    wo = ue_world_point_from_genesis_m(np.asarray(human_anchor_m, dtype=np.float64)).tolist()
    seq = scene_spec.motion.resolved_sequence_npz_path
    man = scene_spec.motion.resolved_mesh_manifest_path
    src = scene_spec.motion.resolved_source_path
    return {
        "source_id": scene_spec.motion.source_id,
        "source_path": str(src) if src is not None else "",
        "sequence_npz_path": str(seq) if seq is not None else "",
        "mesh_manifest_path": str(man) if man is not None else "",
        "fps": float(scene_spec.motion.fps),
        "frame_count": int(scene_spec.motion.frame_count),
        "start_frame": int(scene_spec.motion.start_frame),
        "frame_step": int(scene_spec.motion.frame_step),
        "human_world_offset_m": [float(x) for x in wo],
        "human_align_floor": bool(scene_spec.human.align_floor),
    }


def _bundle_scene_fit_usable(meta: dict, scene_spec_path: Path) -> bool:
    if int(meta.get("motion_bundle_format", 0)) < 2:
        return False
    sp = str(meta.get("scene_fit_spec_path", "") or "").strip()
    if not sp:
        return False
    try:
        if Path(sp).resolve() != scene_spec_path.expanduser().resolve():
            return False
    except OSError:
        return False
    return _scene_fit_revision_matches(meta)


def _scene_fit_revision_matches(meta: dict) -> bool:
    """Path A (placement JSON) writes a hash string; Path B (fallback fit) writes an int matching EXPECTED_SCENE_FIT_PLACEMENT_REVISION."""
    if str(meta.get("scene_fit_source", "")).strip() == "human_scene_placement_json":
        return bool(str(meta.get("scene_fit_placement_revision", "")).strip())
    raw = meta.get("scene_fit_placement_revision", 0)
    try:
        return int(raw) == int(EXPECTED_SCENE_FIT_PLACEMENT_REVISION)
    except (TypeError, ValueError):
        return False


def _ensure_smpl_motion_bundle(
    *,
    bundle_dir: Path,
    motion_payload: dict,
    scene_spec_path: Path,
    force_rebuild: bool,
) -> tuple[Path, dict]:
    bundle_npz = bundle_dir / "smpl_motion_bundle.npz"
    bundle_json_path = bundle_dir / "smpl_motion_bundle.json"
    if (
        bundle_npz.is_file()
        and bundle_json_path.is_file()
        and not force_rebuild
    ):
        try:
            meta = json.loads(bundle_json_path.read_text(encoding="utf-8"))
            if _bundle_scene_fit_usable(meta, scene_spec_path):
                return bundle_npz, meta
        except Exception:
            pass
    bundle_dir.mkdir(parents=True, exist_ok=True)
    export_script = REPO_ROOT / "src" / "projects" / "genesis_ue_sync" / "motion_export" / "export_smpl_motion.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = _build_export_smpl_command(export_script, motion_payload, bundle_dir, scene_spec_path=scene_spec_path)
    print(f"[official_retarget_fbx_host] smpl bundle export: {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    meta = json.loads(bundle_json_path.read_text(encoding="utf-8"))
    if int(meta.get("motion_bundle_format", 0)) < 2:
        raise RuntimeError(f"Motion bundle export produced motion_bundle_format<2: {bundle_json_path}")
    if not _bundle_scene_fit_usable(meta, scene_spec_path):
        raise RuntimeError(f"Motion bundle missing scene_fit_spec_path or wrong revision: {bundle_json_path}")
    return bundle_npz, meta


def _amass_trans_from_bundle_ue_root(trans_ue: np.ndarray) -> np.ndarray:
    out = np.asarray(trans_ue, dtype=np.float32).copy()
    out[:, 1] *= -1.0
    return out


def _convert_smpl_bundle_to_official_npz(
    *,
    bundle_npz_path: Path,
    bundle_meta: dict,
    source_sequence_npz: Path,
    output_npz_path: Path,
    target_frame_count: int,
    target_fps: float,
) -> Path:
    with np.load(bundle_npz_path, allow_pickle=True) as data:
        joint_axis_angles = np.asarray(data["joint_axis_angles"], dtype=np.float32)
        trans_ue = np.asarray(data["root_translation_world"], dtype=np.float32)[:, :3]
        betas = np.asarray(data["betas"], dtype=np.float32)
    full_frame_count = int(joint_axis_angles.shape[0])
    bundle_fps = float(bundle_meta.get("fps", 30.0))
    if target_frame_count > 0:
        cap_requested = int(target_frame_count)
        if target_fps > 1e-6 and bundle_fps > 1e-6:
            cap_requested = int(math.ceil(float(target_frame_count) * bundle_fps / target_fps))
        cap = min(full_frame_count, cap_requested)
    else:
        cap = full_frame_count
    joint_axis_angles = joint_axis_angles[:cap]
    trans_ue = trans_ue[:cap]
    poses72 = joint_axis_angles.reshape(cap, 72)
    poses = np.zeros((cap, 165), dtype=np.float32)
    poses[:, :72] = poses72
    trans = _amass_trans_from_bundle_ue_root(trans_ue)
    if 0 < cap < _MIN_OFFICIAL_NPZ_FRAMES:
        pad_n = _MIN_OFFICIAL_NPZ_FRAMES - cap
        poses = np.concatenate([poses, np.tile(poses[-1:], (pad_n, 1))], axis=0)
        trans = np.concatenate([trans, np.tile(trans[-1:], (pad_n, 1))], axis=0)
        cap = int(poses.shape[0])
        print(
            f"[official_retarget_fbx_host] padded bundle clip {cap - pad_n}->{cap} frames for FBX/UE import",
            flush=True,
        )
    seq_gender = HumanMotionSequence.load(source_sequence_npz).gender
    gender = str(seq_gender).strip().lower() or "neutral"
    if gender not in {"female", "male", "neutral"}:
        gender = "neutral"
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        poses=poses,
        trans=trans,
        betas=betas,
        gender=np.asarray(gender),
        mocap_frame_rate=np.asarray(bundle_fps, dtype=np.float32),
    )
    print(
        f"[official_retarget_fbx_host] official npz from scene-fit bundle frames={cap} "
        f"full_frames={full_frame_count} bundle_fps={bundle_fps:.3f} -> {output_npz_path}",
        flush=True,
    )
    return output_npz_path


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "motion"


def _official_retarget_asset_name(target_dir: str, source_name: str) -> str:
    target_name = target_dir.rstrip("/").rsplit("/", maxsplit=1)[-1]
    return f"{target_name}+{source_name}_Anim"


def _execute_python_script_arg(script_path: Path, *args: str) -> str:
    joined = " ".join([str(script_path), *[str(arg) for arg in args]])
    return f"-ExecutePythonScript={joined}"


def _write_retarget_paths_json(unreal_cmd: str, blender_bin: str) -> None:
    payload = {
        "PYTHON_SCRIPT_DIR": str(PROJECT_PATHS.bedlam_retarget_python_root),
        "UNREAL_APP_PATH": str(unreal_cmd),
        "UNREAL_PROJECT_PATH": str(RETARGET_PROJECT),
        "BLENDER_APP_PATH": str(blender_bin),
    }
    (RETARGET_REPO / "paths.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_official_blender_prereqs() -> None:
    if not SMPLX_ADDON_REPO.exists():
        raise RuntimeError(
            "Missing ref_code_library/smplx_blender_addon. Clone the public addon repo or provide the official addon ZIP."
        )
    if not SMPLX_ADDON_REQUIRED_BLEND.is_file():
        raise RuntimeError(
            "Missing official SMPL-X Blender addon model bundle: "
            f"{SMPLX_ADDON_REQUIRED_BLEND}. "
            "Download the official smplx_blender_addon ZIP from https://smpl-x.is.tue.mpg.de."
        )


def _convert_sequence_to_official_npz(
    sequence_npz_path: Path,
    output_npz_path: Path,
    *,
    target_frame_count: int,
    target_fps: float,
) -> Path:
    sequence = HumanMotionSequence.load(sequence_npz_path)
    full_frame_count = int(sequence.frame_count)
    if target_frame_count > 0:
        cap_requested = int(target_frame_count)
        if target_fps > 1e-6 and float(sequence.fps) > 1e-6:
            cap_requested = int(math.ceil(float(target_frame_count) * float(sequence.fps) / target_fps))
        cap = min(sequence.frame_count, cap_requested)
        sequence = HumanMotionSequence(
            source_dataset=sequence.source_dataset,
            sequence_name=sequence.sequence_name,
            source_path=sequence.source_path,
            model_type=sequence.model_type,
            gender=sequence.gender,
            fps=sequence.fps,
            poses=np.asarray(sequence.poses[:cap], dtype=np.float32),
            trans=np.asarray(sequence.trans[:cap], dtype=np.float32),
            betas=np.asarray(sequence.betas, dtype=np.float32),
            image_names=list(sequence.image_names[:cap]) if sequence.image_names else [],
            cam_int=None if sequence.cam_int is None else np.asarray(sequence.cam_int[:cap], dtype=np.float32),
            cam_ext=None if sequence.cam_ext is None else np.asarray(sequence.cam_ext[:cap], dtype=np.float32),
            metadata=dict(sequence.metadata),
        )
    gender = str(sequence.gender).strip().lower() or "neutral"
    if gender not in {"female", "male", "neutral"}:
        gender = "neutral"
    poses = np.asarray(sequence.poses, dtype=np.float32)
    if poses.ndim != 2:
        raise RuntimeError(f"Expected 2D poses array, got shape {poses.shape}")
    if poses.shape[1] < 165:
        padded = np.zeros((poses.shape[0], 165), dtype=np.float32)
        padded[:, : poses.shape[1]] = poses
        poses = padded
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        poses=poses,
        trans=np.asarray(sequence.trans[:, :3], dtype=np.float32),
        betas=np.asarray(sequence.betas, dtype=np.float32),
        gender=np.asarray(gender),
        mocap_frame_rate=np.asarray(float(sequence.fps), dtype=np.float32),
    )
    print(
        f"[official_retarget_fbx_host] npz frames={int(sequence.frame_count)} "
        f"full_frames={full_frame_count} fps={float(sequence.fps):.3f} -> {output_npz_path}",
        flush=True,
    )
    return output_npz_path


def _retarget_signature(
    *,
    scene_spec_path: Path,
    scene_spec: SyncSceneSpec,
    official_motion_input: str,
) -> dict:
    motion = scene_spec.motion
    seq_path = motion.resolved_sequence_npz_path
    if seq_path is None or not seq_path.is_file():
        raise RuntimeError("Official retarget requires scene.motion.sequence_npz_path pointing to an existing file.")
    shape_ref = npz_shape_reference_for_retarget_cache(seq_path)
    avatar = scene_spec.ue_avatar
    skel = str(avatar.skeletal_mesh_path)
    target_dir = skel.rsplit("/", maxsplit=1)[0]
    return {
        "scene_spec_path": str(scene_spec_path.expanduser().resolve()),
        "sequence_npz_path": str(seq_path.resolve()),
        "body_mode": str(avatar.body_mode),
        "body_name": str(avatar.body_name),
        "skeletal_mesh_path": skel,
        "target_dir": target_dir,
        "texture_body": avatar.texture_body,
        "texture_clothing": avatar.texture_clothing,
        "texture_clothing_overlay": avatar.texture_clothing_overlay,
        "fbx_global_scale": float(avatar.fbx_global_scale),
        "fps": float(motion.fps),
        "frame_count": int(motion.frame_count),
        "human_anchor_m": [float(v) for v in scene_spec.resolved_human_anchor()],
        "human_align_floor": bool(scene_spec.human.align_floor),
        "human_display_vertical_offset_m": float(scene_spec.human.display_vertical_offset_m),
        "official_motion_input": str(official_motion_input),
        "shape_reference": shape_ref,
    }


def _run_checked(cmd: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"[official_retarget_fbx_host] {printable}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd or REPO_ROOT), env=os.environ.copy())


def ensure_official_retarget_fbx_cached(
    *,
    scene_spec_path: Path,
    augmentation_spec_path: Path | None,
    output_root: Path,
    force_rebuild: bool,
    unreal_cmd: str | None = None,
) -> Path | None:
    """Build official retarget FBX under output_root/cache/official_retarget/ when needed.

    Returns None if body_mode is not official_retargeted_overlay (no official retarget step).
    """
    scene_spec, _ = resolve_scene_spec_with_augmentation(
        scene_spec_path.expanduser().resolve(),
        None
        if augmentation_spec_path is None
        else augmentation_spec_path.expanduser().resolve(),
    )
    if not bool(getattr(scene_spec.render, "ue_spawn_human", True)):
        return None
    avatar = scene_spec.ue_avatar
    if str(avatar.body_mode) != "official_retargeted_overlay":
        return None

    motion_source_id = str(scene_spec.motion.source_id)
    if not motion_source_id:
        raise RuntimeError("scene.motion.source_id is required for official_retargeted_overlay.")

    official_root = output_root.expanduser().resolve() / "cache" / "official_retarget" / _sanitize_name(motion_source_id)
    source_npz_path = official_root / "source_npz" / f"{_sanitize_name(motion_source_id)}.npz"
    pool_root = official_root / "pool"
    pool_dir = f"/Game/BedlamRetarget/OfficialSync/{_sanitize_name(motion_source_id)}"
    skel = str(avatar.skeletal_mesh_path)
    target_dir = skel.rsplit("/", maxsplit=1)[0]
    sidecar_path = official_root / "retarget_sidecar.json"
    resolved_scene = scene_spec_path.expanduser().resolve()

    motion_payload = _motion_payload_from_scene_spec(scene_spec)
    bundle_dir = output_root.expanduser().resolve() / "cache" / "smpl_bundle" / _sanitize_name(motion_source_id)
    _, bundle_meta = _ensure_smpl_motion_bundle(
        bundle_dir=bundle_dir,
        motion_payload=motion_payload,
        scene_spec_path=resolved_scene,
        force_rebuild=force_rebuild,
    )
    official_motion_input = "scene_fit_bundle"

    sig = _retarget_signature(
        scene_spec_path=resolved_scene,
        scene_spec=scene_spec,
        official_motion_input=official_motion_input,
    )

    if sidecar_path.is_file() and not force_rebuild:
        try:
            cached = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if dict(cached.get("signature", {})) == sig:
                candidate = Path(str(cached.get("retarget_fbx_path", ""))).expanduser()
                if candidate.is_file():
                    print(
                        f"[official_retarget_fbx_host] cache hit: skip retarget Unreal launch -> {candidate}",
                        flush=True,
                    )
                    return candidate.resolve()
            else:
                print(
                    f"[official_retarget_fbx_host] cache stale: signature mismatch {sidecar_path}",
                    flush=True,
                )
        except Exception:
            pass

    print(
        f"[official_retarget_fbx_host] cache build: launching retarget Unreal for {motion_source_id}",
        flush=True,
    )
    _validate_official_blender_prereqs()
    ue = unreal_cmd or discover_unreal_editor_executable(PROJECT_PATHS)
    blender_bin = discover_blender_executable(PROJECT_PATHS)
    _write_retarget_paths_json(unreal_cmd=ue, blender_bin=blender_bin)

    motion = scene_spec.motion
    seq_path = motion.resolved_sequence_npz_path
    if seq_path is None or not seq_path.is_file():
        raise RuntimeError("Official retarget requires a valid sequence NPZ path.")

    req_fc = int(motion.frame_count)
    bake_fc = 0 if req_fc <= 0 else max(req_fc, _MIN_OFFICIAL_NPZ_FRAMES)
    if bake_fc != req_fc and req_fc > 0:
        print(
            f"[official_retarget_fbx_host] motion frame_count={req_fc} expanded to {bake_fc} for robust FBX/UE import",
            flush=True,
        )

    bundle_npz_path = bundle_dir / "smpl_motion_bundle.npz"
    converted_npz = _convert_smpl_bundle_to_official_npz(
        bundle_npz_path=bundle_npz_path,
        bundle_meta=bundle_meta,
        source_sequence_npz=seq_path,
        output_npz_path=source_npz_path,
        target_frame_count=bake_fc,
        target_fps=float(motion.fps),
    )
    _run_checked(
        [
            *discover_python_command(),
            "make_fbx_files.py",
            "--input_dir",
            str(converted_npz.parent),
            "--output_dir",
            str(pool_root),
            "--processes",
            "1",
            "--target_fps",
            str(int(round(float(motion.fps)))),
            "--anim_format",
            "AMASS",
        ],
        cwd=RETARGET_REPO,
    )
    animation_fbx_files = sorted((pool_root / "animations").glob("*.fbx"))
    if len(animation_fbx_files) != 1:
        raise RuntimeError(f"Expected exactly one official animation FBX, found {len(animation_fbx_files)}")
    source_animation_fbx = animation_fbx_files[0].resolve()
    source_name = source_animation_fbx.stem
    source_dest_dir = f"{pool_dir}/animations/{source_name}"
    retarget_out_dir = f"{pool_dir}/retargeting/official_single"
    retarget_asset_name = _official_retarget_asset_name(target_dir, source_name)
    retarget_fbx = (official_root / "retargeted_fbx" / f"{retarget_asset_name}.fbx").resolve()
    animation_session_cfg = official_root / "animation_asset_session.json"
    official_root.mkdir(parents=True, exist_ok=True)
    animation_session_cfg.write_text(
        json.dumps(
            {
                "input_fbx_path": str(source_animation_fbx),
                "import_destination_path": source_dest_dir,
                "import_destination_name": source_name,
                "source_asset_root": pool_dir,
                "source_asset_name": source_name,
                "target_asset_dir": target_dir,
                "output_asset_dir": retarget_out_dir,
                "output_fbx_path": str(retarget_fbx),
                "export_preview_mesh": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _run_checked(
        [
            ue,
            str(RETARGET_PROJECT),
            "-stdout",
            "-FullStdOutLogOutput",
            "-unattended",
            "-nop4",
            _execute_python_script_arg(UE_ANIMATION_ASSET_SESSION_SCRIPT, str(animation_session_cfg.resolve())),
        ],
    )
    if not retarget_fbx.is_file():
        raise RuntimeError(f"Retargeted FBX was not exported: {retarget_fbx}")
    sidecar_path.write_text(
        json.dumps(
            {
                "signature": sig,
                "retarget_fbx_path": str(retarget_fbx),
                "bundle_meta_path": str(bundle_dir / "smpl_motion_bundle.json"),
                "bundle_scene_fit": {
                    "scene_fit_spec_path": bundle_meta.get("scene_fit_spec_path"),
                    "scene_fit_placement_revision": bundle_meta.get("scene_fit_placement_revision"),
                    "scene_fit_support_plane_z_m": bundle_meta.get("scene_fit_support_plane_z_m"),
                    "scene_human_display_vertical_offset_m": bundle_meta.get(
                        "scene_human_display_vertical_offset_m"
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[official_retarget_fbx_host] wrote {retarget_fbx}", flush=True)
    return retarget_fbx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-spec", type=Path, required=True)
    p.add_argument("--augmentation-spec", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--unreal-editor", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_official_retarget_fbx_cached(
        scene_spec_path=args.scene_spec,
        augmentation_spec_path=args.augmentation_spec,
        output_root=args.output_root,
        force_rebuild=bool(args.force_rebuild),
        unreal_cmd=args.unreal_editor,
    )


if __name__ == "__main__":
    main()
