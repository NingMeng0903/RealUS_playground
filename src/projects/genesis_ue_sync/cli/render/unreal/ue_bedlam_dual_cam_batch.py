from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import unreal

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = next(parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
for candidate in (SCRIPT_DIR, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.project import project_paths
from projects.genesis_ue_sync.config.toolchain import discover_blender_executable, discover_python_command
from projects.genesis_ue_sync.integrations.ue import (
    EditorCommandResult,
    EditorSessionPaths,
    EditorSessionStatus,
    import_editor_python_module,
)
import ue_common_scene_loader as scene_loader

PROJECT_PATHS = project_paths(__file__)
sequence_tools = import_editor_python_module("create_level_sequences_csv", PROJECT_PATHS)
render_queue_tools = import_editor_python_module("create_movie_render_queue", PROJECT_PATHS)
PIPELINE_EXECUTOR = None
PIPELINE_RUNTIME_FLAGS = {
    "quit_editor_on_finish": False,
    "session_paths": None,
    "request_id": None,
    "sequence_names": [],
}

# Must match scene_fit_placement_revision written by motion_export/export_smpl_motion.py when using --scene-spec.
EXPECTED_SCENE_FIT_PLACEMENT_REVISION = 3
# Keep in sync with scripts/visualization/blender/import_smpl_motion_bundle.py FBX_EXPORT_PROFILE.
FBX_EXPORT_PROFILE = "gs_no_apply_unit_v1"
BEDLAM_TEXTURE_BODY_ROOT = "/Engine/PS/Meshcapade/SMPLX/Textures"
BEDLAM_TEXTURE_CLOTHING_OVERLAY_ROOT = "/Engine/PS/Bedlam/Clothing/MaterialsSMPLX/Textures"
BEDLAM_CLOTHING_MATERIAL_ROOT = "/Engine/PS/Bedlam/Clothing/Materials"


def _preferred_fbx_animation_length_mode():
    lt = getattr(unreal, "FBXAnimationLengthImportType", None)
    if lt is None:
        return None
    for name in ("FBXALIT_ANIMATED_KEY", "FBXALIT_EXPORTED_TIME"):
        mode = getattr(lt, name, None)
        if mode is not None:
            return mode
    return None


def _shape_reference_from_sequence_npz(sequence_npz_path: Path) -> dict:
    def _first_string(value, default: str = "") -> str:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return str(arr.reshape(-1)[0])

    try:
        with np.load(sequence_npz_path, allow_pickle=True) as payload:
            betas = np.asarray(payload["betas"], dtype=np.float32).reshape(-1) if "betas" in payload else np.asarray([], dtype=np.float32)
            gender_raw = payload["gender"] if "gender" in payload else np.asarray(["neutral"])
            model_type = _first_string(payload["model_type"], "") if "model_type" in payload else ""
            source_dataset = _first_string(payload["source_dataset"], "") if "source_dataset" in payload else ""
            gender = _first_string(gender_raw, "neutral")
    except Exception as exc:
        return {"error": repr(exc)}
    return {
        "source_dataset": source_dataset,
        "model_type": model_type,
        "gender": gender,
        "betas_dim": int(betas.size),
        "betas": [float(v) for v in betas.tolist()],
    }


def _apply_fbx_anim_length_import(options: unreal.FbxImportUI) -> None:
    if not bool(getattr(options, "import_animations", False)):
        return
    asd = getattr(options, "anim_sequence_import_data", None)
    if asd is None:
        return
    mode = _preferred_fbx_animation_length_mode()
    if mode is None:
        return
    if not hasattr(asd, "animation_length"):
        return
    try:
        asd.set_editor_property("animation_length", mode)
    except Exception:
        try:
            asd.animation_length = mode
        except Exception:
            return
    unreal.log(f"UE_PIPELINE: FBX anim import animation_length={mode}")


def _safe_asset_name(asset) -> str:
    try:
        if asset is not None and hasattr(asset, "get_name"):
            return str(asset.get_name())
    except Exception:
        pass
    return str(asset)


def _strip_unreal_asset_literal(value: str) -> str:
    raw = str(value or "").strip()
    if "'" in raw and raw.endswith("'"):
        parts = raw.split("'", maxsplit=2)
        if len(parts) >= 2:
            return parts[1].strip()
    return raw


def _asset_leaf_name(value: str) -> str:
    raw = _strip_unreal_asset_literal(value).rstrip("/")
    leaf = raw.rsplit("/", maxsplit=1)[-1]
    if "." in leaf:
        leaf = leaf.rsplit(".", maxsplit=1)[-1]
    return leaf


def _asset_reference_candidates(root: str, name: str | None, *, class_names: tuple[str, ...] = ()) -> list[str]:
    raw = _strip_unreal_asset_literal(str(name or ""))
    if not raw:
        return []
    candidates: list[str] = []
    if raw.startswith("/"):
        candidates.extend([raw])
        leaf = _asset_leaf_name(raw)
        if "." not in raw.rsplit("/", maxsplit=1)[-1]:
            candidates.append(f"{raw}.{leaf}")
    else:
        base = str(root).rstrip("/")
        candidates.extend([f"{base}/{raw}", f"{base}/{raw}.{raw}"])
    expanded: list[str] = []
    for candidate in candidates:
        expanded.append(candidate)
        for class_name in class_names:
            expanded.append(f"{class_name}'{candidate}'")
    seen: set[str] = set()
    return [item for item in expanded if item and not (item in seen or seen.add(item))]


def _asset_is_class(asset, class_names: tuple[str, ...]) -> bool:
    if not class_names:
        return True
    try:
        class_name = str(asset.get_class().get_name())
    except Exception:
        return False
    return class_name in set(class_names)


def _load_asset_from_candidates(candidates: list[str], *, class_names: tuple[str, ...] = ()):
    for candidate in candidates:
        asset_path = _strip_unreal_asset_literal(candidate)
        for loader_arg in (candidate, asset_path):
            try:
                asset = unreal.load_asset(loader_arg)
            except Exception:
                asset = None
            if asset is not None and _asset_is_class(asset, class_names):
                return asset, asset_path
            try:
                asset = unreal.EditorAssetLibrary.load_asset(loader_arg)
            except Exception:
                asset = None
            if asset is not None and _asset_is_class(asset, class_names):
                return asset, asset_path
    return None, None


def _search_asset_by_leaf(roots: tuple[str, ...], name: str | None, *, class_names: tuple[str, ...] = ()):
    leaf = _asset_leaf_name(str(name or ""))
    if not leaf:
        return None, None, []
    tried_roots: list[str] = []
    leaf_lower = leaf.lower()
    for root in roots:
        root = str(root or "").rstrip("/")
        if not root:
            continue
        tried_roots.append(root)
        try:
            listed = unreal.EditorAssetLibrary.list_assets(root, recursive=True, include_folder=False)
        except Exception as exc:
            unreal.log_warning(f"UE_PIPELINE: asset search failed root={root}: {exc!r}")
            continue
        matches = []
        for asset_path in listed:
            asset_leaf = _asset_leaf_name(str(asset_path)).lower()
            if asset_leaf == leaf_lower or leaf_lower in asset_leaf:
                matches.append(str(asset_path))
        for asset_path in sorted(matches, key=len):
            asset, resolved = _load_asset_from_candidates([asset_path], class_names=class_names)
            if asset is not None:
                return asset, resolved, tried_roots
    return None, None, tried_roots


def _resolve_ue_asset(
    root: str,
    name: str | None,
    *,
    class_names: tuple[str, ...] = (),
    search_roots: tuple[str, ...] = (),
):
    candidates = _asset_reference_candidates(root, name, class_names=class_names)
    asset, resolved = _load_asset_from_candidates(candidates, class_names=class_names)
    if asset is not None:
        return asset, resolved, {"candidates": candidates, "search_roots": [], "found_by": "candidate"}
    asset, resolved, tried_roots = _search_asset_by_leaf(search_roots or (root,), name, class_names=class_names)
    return asset, resolved, {"candidates": candidates, "search_roots": tried_roots, "found_by": "search" if asset is not None else None}


def _configure_anim_sequence_root_motion(asset, *, enable_root_motion: bool = True) -> None:
    if asset is None:
        return
    changed = False
    rm = bool(enable_root_motion)
    for prop, value in (("enable_root_motion", rm), ("force_root_lock", rm)):
        try:
            cur = asset.get_editor_property(prop)
        except Exception:
            continue
        if bool(cur) == bool(value):
            continue
        try:
            asset.set_editor_property(prop, value)
            changed = True
            unreal.log(f"UE_PIPELINE: animation asset {_safe_asset_name(asset)} set {prop}={value}")
        except Exception as exc:
            unreal.log_warning(f"UE_PIPELINE: failed to set {prop} on {_safe_asset_name(asset)}: {exc!r}")
    if changed:
        try:
            unreal.EditorAssetLibrary.save_loaded_asset(asset)
        except Exception as exc:
            unreal.log_warning(f"UE_PIPELINE: failed to save animation asset {_safe_asset_name(asset)}: {exc!r}")


def _find_level_actor(predicate):
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in actor_subsystem.get_all_level_actors():
        if predicate(actor):
            return actor
    return None


def _require_actor(name: str, predicate):
    actor = _find_level_actor(predicate)
    if actor is None:
        raise RuntimeError(f"Required actor not found: {name}")
    return actor


def _set_component_anim_tick_always(component) -> None:
    tick_opt = getattr(unreal, "VisibilityBasedAnimTickOption", None)
    if tick_opt is None:
        return
    desired = None
    for name in ("ALWAYS_TICK_POSE_AND_REFRESH_BONES", "AlwaysTickPoseAndRefreshBones"):
        desired = getattr(tick_opt, name, None)
        if desired is not None:
            break
    if desired is None:
        return
    setter = getattr(component, "set_visibility_based_anim_tick_option", None)
    if setter is not None:
        setter(desired)
        return
    component.set_editor_property("visibility_based_anim_tick_option", desired)


def _python_bin() -> str:
    for candidate in ("python3", "python"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError("Cannot find python3/python in PATH.")


def _motion_export_python_bin() -> str:
    """Interpreter for SMPL export (needs torch + smplx). Override with AMONGUS_MOTION_PYTHON."""
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


def _session_paths(session_dir: str | None) -> EditorSessionPaths | None:
    if not session_dir:
        return None
    return EditorSessionPaths(Path(session_dir).expanduser().resolve())


def _write_session_status(session_dir: str | None, *, state: str, ready: bool, detail: str = "") -> None:
    paths = _session_paths(session_dir)
    if paths is None:
        return
    EditorSessionStatus(
        state=state,
        project_path=str(PROJECT_PATHS.bedlam_unreal_project_file),
        ready=ready,
        detail=detail,
        process_pid=os.getpid(),
    ).save(paths)


def _write_command_result(*, request_id: str | None, success: bool, detail: str, payload: dict | None = None) -> None:
    paths = _session_paths(PIPELINE_RUNTIME_FLAGS.get("session_paths"))
    if paths is None or not request_id:
        return
    EditorCommandResult(
        request_id=request_id,
        success=success,
        detail=detail,
        payload=dict(payload or {}),
    ).write(paths)


def _resolve_motion_input_path(motion_payload: dict) -> tuple[str, str]:
    sequence_npz_path = str(motion_payload.get("sequence_npz_path", "") or "")
    mesh_manifest_path = str(motion_payload.get("mesh_manifest_path", "") or "")
    if sequence_npz_path:
        p = Path(sequence_npz_path)
        if not p.is_file():
            alt = REPO_ROOT / sequence_npz_path
            if alt.is_file():
                p = alt
        if p.is_file():
            return "sequence_npz", str(p.resolve())
    if mesh_manifest_path:
        p = Path(mesh_manifest_path)
        if not p.is_file():
            alt = REPO_ROOT / mesh_manifest_path
            if alt.is_file():
                p = alt
        if p.is_file():
            return "manifest", str(p.resolve())
    raise RuntimeError("No valid motion input exists for bundle export.")


def _blender_bin() -> str:
    return discover_blender_executable(PROJECT_PATHS)


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "motion"


def _asset_object_path(package_path: str) -> str:
    asset_name = package_path.rsplit("/", maxsplit=1)[-1]
    return f"{package_path}.{asset_name}"


def _resolve_optional_local_path(path_value: str) -> Path | None:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    resolved = (REPO_ROOT / candidate).resolve()
    return resolved if resolved.exists() else None


def _python_export_command(
    export_script: Path,
    motion_payload: dict,
    bundle_dir: Path,
    *,
    scene_spec_path: str | None = None,
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
    if scene_spec_path:
        command.extend(["--scene-spec", str(Path(scene_spec_path).expanduser().resolve())])
    command.extend(["--output-world", "ue"])
    return command


def _skeleton_from_mesh(skeletal_mesh) -> object:
    skeleton = None
    if hasattr(skeletal_mesh, "skeleton"):
        skeleton = skeletal_mesh.skeleton
    if skeleton is None and hasattr(skeletal_mesh, "get_editor_property"):
        skeleton = skeletal_mesh.get_editor_property("skeleton")
    if skeleton is None:
        raise RuntimeError(f"Cannot resolve skeleton from skeletal mesh: {skeletal_mesh}")
    return skeleton


@dataclass
class RenderConfig:
    output_root: str
    scene_spec_path: str
    augmentation_spec_path: str | None
    render_now: bool | None
    force_rebuild_motion: bool
    quit_editor_on_finish: bool
    session_dir: str | None = None
    request_id: str | None = None


@dataclass
class HumanAnimationAssets:
    motion_source_id: str
    source_sequence_npz: str
    source_mesh_manifest: str
    bundle_dir: str
    bundle_npz: str
    bundle_meta: str
    animation_fbx: str
    imported_animation_path: str
    skeletal_mesh_path: str
    texture_body: Optional[str]
    texture_clothing: Optional[str]
    texture_clothing_overlay: Optional[str]


class UeBedlamRenderPipeline:
    def __init__(self, config: RenderConfig) -> None:
        self.config = config
        self.scene_payload: dict | None = None
        self.queue: unreal.MoviePipelineQueue | None = None
        self.animation_assets: HumanAnimationAssets | None = None
        self.sequence_names: list[str] = []
        self.material_application: dict[str, object] = {}

    @property
    def output_root(self) -> Path:
        return Path(self.config.output_root)

    def prepare_scene(self) -> None:
        unreal.log("UE_PIPELINE: prepare_scene")
        self.scene_payload = scene_loader.apply_scene_to_current_level(
            self.config.scene_spec_path,
            self.config.augmentation_spec_path,
        )

    def prepare_robot_assets(self) -> None:
        if self.scene_payload is None:
            raise RuntimeError("Scene must be prepared before robot assets.")

    def _require_scene_payload(self) -> dict:
        if self.scene_payload is None:
            raise RuntimeError("Scene has not been prepared.")
        return self.scene_payload

    def _verify_motion_payload(self) -> dict:
        payload = self._require_scene_payload()["motion_payload"]
        if not payload["source_id"]:
            raise RuntimeError("scene.motion.source_id is required.")
        sequence_npz_path = Path(payload["sequence_npz_path"]) if payload["sequence_npz_path"] else None
        manifest_path = Path(payload["mesh_manifest_path"]) if payload["mesh_manifest_path"] else None
        if (sequence_npz_path is None or not sequence_npz_path.is_file()) and (manifest_path is None or not manifest_path.is_file()):
            raise RuntimeError("Either scene.motion.sequence_npz_path or scene.motion.mesh_manifest_path must exist.")
        frame_count = int(payload["frame_count"])
        render_frame_count = int(self._require_scene_payload()["render_payload"]["ue_frame_count"])
        if frame_count > 0 and render_frame_count > 0 and frame_count != render_frame_count:
            raise RuntimeError(
                f"Motion frame_count ({frame_count}) does not match render ue_frame_count ({render_frame_count})."
            )
        return payload

    def _bundle_output_dir(self, motion_source_id: str) -> Path:
        return self.output_root / "cache" / "smpl_bundle" / _sanitize_name(motion_source_id)

    def _smpl_bundle_cache_valid(self, bundle_meta: Path, motion_payload: dict) -> bool:
        try:
            meta = json.loads(bundle_meta.read_text(encoding="utf-8"))
        except Exception:
            return False
        if int(meta.get("motion_bundle_format", 0)) < 2:
            return False
        scene_fit_spec = str(meta.get("scene_fit_spec_path", "") or "").strip()
        if scene_fit_spec:
            try:
                same = Path(scene_fit_spec).resolve() == Path(self.config.scene_spec_path).expanduser().resolve()
            except Exception:
                return False
            if not same:
                return False
            if str(meta.get("scene_fit_source", "")).strip() == "human_scene_placement_json":
                return bool(str(meta.get("scene_fit_placement_revision", "")).strip())
            try:
                return int(meta.get("scene_fit_placement_revision", 0)) == int(EXPECTED_SCENE_FIT_PLACEMENT_REVISION)
            except (TypeError, ValueError):
                return False
        wo = motion_payload.get("human_world_offset_m")
        if not isinstance(wo, (list, tuple)) or len(wo) != 3:
            return False
        gw = meta.get("genesis_world_offset_m")
        if not isinstance(gw, list) or len(gw) != 3:
            return False
        for i in range(3):
            if abs(float(gw[i]) - float(wo[i])) > 1e-5:
                return False
        af = bool(motion_payload.get("human_align_floor", False))
        if bool(meta.get("genesis_align_floor", False)) != af:
            return False
        return True

    def _animation_fbx_path(self, motion_source_id: str) -> Path:
        return self.output_root / "cache" / "fbx" / f"{_sanitize_name(motion_source_id)}.fbx"

    def _official_retarget_search_roots(self) -> list[Path]:
        try:
            out = self.output_root.resolve()
        except OSError:
            out = self.output_root.expanduser()
        candidates = (out, out.parent, out.parent.parent)
        seen: set[str] = set()
        uniq: list[Path] = []
        for r in candidates:
            try:
                key = str(r.resolve())
            except OSError:
                key = str(r)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        return uniq

    def _official_retarget_signature(self, motion_payload: dict, ue_avatar: dict) -> dict:
        seq_path = Path(str(motion_payload.get("sequence_npz_path", "") or "")).expanduser()
        try:
            seq_path = seq_path.resolve()
        except OSError:
            pass
        skel = str(ue_avatar.get("skeletal_mesh_path", ""))
        target_dir = skel.rsplit("/", maxsplit=1)[0]
        return {
            "scene_spec_path": str(Path(self.config.scene_spec_path).expanduser().resolve()),
            "sequence_npz_path": str(seq_path),
            "body_mode": str(ue_avatar.get("body_mode", "")),
            "body_name": str(ue_avatar.get("body_name", "")),
            "skeletal_mesh_path": skel,
            "target_dir": target_dir,
            "texture_body": ue_avatar.get("texture_body"),
            "texture_clothing": ue_avatar.get("texture_clothing"),
            "texture_clothing_overlay": ue_avatar.get("texture_clothing_overlay"),
            "fbx_global_scale": float(ue_avatar.get("fbx_global_scale", 100.0)),
            "fps": float(motion_payload.get("fps", 0.0) or 0.0),
            "frame_count": int(motion_payload.get("frame_count", 0) or 0),
            "human_anchor_m": [float(v) for v in self._require_scene_payload().get("human_anchor_m", (0.0, 0.0, 0.0))],
            "human_align_floor": bool(motion_payload.get("human_align_floor", False)),
            "human_display_vertical_offset_m": float(
                (self._require_scene_payload().get("human_payload") or {}).get("display_vertical_offset_m", 0.0)
            ),
            "official_motion_input": "scene_fit_bundle",
            "shape_reference": _shape_reference_from_sequence_npz(seq_path),
        }

    def _resolve_official_retarget_fbx_path(self, motion_payload: dict, ue_avatar: dict) -> Path:
        motion_source_id = str(motion_payload["source_id"])
        explicit = _resolve_optional_local_path(str(ue_avatar.get("fallback_animation_path", "") or ""))
        if explicit is not None and explicit.is_file():
            return explicit
        expected_sig = self._official_retarget_signature(motion_payload, ue_avatar)
        for base in self._official_retarget_search_roots():
            sidecar = base / "cache" / "official_retarget" / _sanitize_name(motion_source_id) / "retarget_sidecar.json"
            if not sidecar.is_file():
                continue
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                if dict(data.get("signature", {})) != expected_sig:
                    unreal.log_warning(f"UE_PIPELINE: skip stale official retarget sidecar {sidecar}")
                    continue
                candidate = Path(str(data.get("retarget_fbx_path", ""))).expanduser()
                if candidate.is_file():
                    unreal.log(f"UE_PIPELINE: official retarget FBX resolved via {sidecar}")
                    return candidate.resolve()
            except Exception as exc:
                unreal.log_warning(f"UE_PIPELINE: failed to read official retarget sidecar {sidecar}: {exc!r}")
        raise RuntimeError(
            "Official retarget FBX not found: set scene.ue_avatar.fallback_animation_path, or run "
            "official_retarget_fbx_host so cache exists under <output_root>/cache/official_retarget/... "
            "(parent output directories are also searched for augmentation batch layouts)."
        )

    def _run_subprocess(
        self,
        command: list[str],
        *,
        env: Optional[dict[str, str]] = None,
        cwd: Path | None = None,
    ) -> None:
        unreal.log(f"UE_PIPELINE subprocess: {' '.join(command)}")
        effective_env = dict(os.environ)
        if env is not None:
            effective_env.update(env)
        src_root = str(REPO_ROOT / "src")
        existing_pythonpath = str(effective_env.get("PYTHONPATH", "") or "")
        pythonpath_parts = [part for part in existing_pythonpath.split(os.pathsep) if part]
        if src_root not in pythonpath_parts:
            effective_env["PYTHONPATH"] = (
                src_root if not existing_pythonpath else src_root + os.pathsep + existing_pythonpath
            )
        subprocess.run(command, check=True, cwd=str(cwd or REPO_ROOT), env=effective_env)

    def _export_motion_bundle(self, motion_payload: dict) -> tuple[Path, Path, Path]:
        motion_source_id = str(motion_payload["source_id"])
        bundle_dir = self._bundle_output_dir(motion_source_id)
        bundle_npz = bundle_dir / "smpl_motion_bundle.npz"
        bundle_meta = bundle_dir / "smpl_motion_bundle.json"
        cache_ok = (
            bundle_npz.is_file()
            and bundle_meta.is_file()
            and not self.config.force_rebuild_motion
            and self._smpl_bundle_cache_valid(bundle_meta, motion_payload)
        )
        if cache_ok:
            return bundle_dir, bundle_npz, bundle_meta
        bundle_dir.mkdir(parents=True, exist_ok=True)
        export_script = REPO_ROOT / "src" / "projects" / "genesis_ue_sync" / "motion_export" / "export_smpl_motion.py"
        unreal.log(
            f"UE_PIPELINE motion_export_python={' '.join(_motion_export_python_cmd())} "
            f"(set AMONGUS_MOTION_PYTHON if you need to override)"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self._run_subprocess(
            [
                *_python_export_command(
                    export_script,
                    motion_payload,
                    bundle_dir,
                    scene_spec_path=self.config.scene_spec_path,
                )
            ],
            env=env,
        )
        try:
            meta_after = json.loads(bundle_meta.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Motion bundle export did not write readable metadata: {bundle_meta}") from exc
        if int(meta_after.get("motion_bundle_format", 0)) < 2:
            raise RuntimeError(
                f"Motion export produced stale metadata (motion_bundle_format<2). "
                f"Use repo Among_US with export_smpl_motion.py v2 and PYTHONPATH=<repo>/src. File: {bundle_meta}"
            )
        return bundle_dir, bundle_npz, bundle_meta

    def _export_animation_fbx(self, bundle_dir: Path, motion_source_id: str) -> Path:
        animation_fbx = self._animation_fbx_path(motion_source_id)
        bundle_npz = bundle_dir / "smpl_motion_bundle.npz"
        ue_avatar = self._require_scene_payload().get("ue_avatar_payload", {})
        fbx_scale = float(ue_avatar.get("fbx_global_scale", 100.0))

        sidecar = bundle_dir / "fbx_export_sidecar.json"
        sidecar_matches = False
        if sidecar.is_file():
            try:
                sc = json.loads(sidecar.read_text(encoding="utf-8"))
                sidecar_matches = (
                    abs(float(sc.get("fbx_global_scale", -1.0)) - fbx_scale) < 1e-3
                    and str(sc.get("export_profile", "")) == FBX_EXPORT_PROFILE
                )
            except Exception:
                sidecar_matches = False

        npz_newer = (
            bundle_npz.is_file()
            and animation_fbx.is_file()
            and bundle_npz.stat().st_mtime > animation_fbx.stat().st_mtime
        )
        need_blender = (
            self.config.force_rebuild_motion
            or not animation_fbx.is_file()
            or npz_newer
            or not sidecar_matches
        )
        if not need_blender:
            return animation_fbx
        animation_fbx.parent.mkdir(parents=True, exist_ok=True)
        blender_script = (
            REPO_ROOT
            / "src"
            / "projects"
            / "bedlam_blender_sync"
            / "blender"
            / "import_smpl_motion_bundle.py"
        )
        self._run_subprocess(
            [
                _blender_bin(),
                "--background",
                "--python",
                str(blender_script),
                "--",
                "--bundle-dir",
                str(bundle_dir),
                "--export-fbx",
                str(animation_fbx),
                "--clear-existing",
                "--fbx-global-scale",
                str(fbx_scale),
            ]
        )
        sidecar.write_text(
            json.dumps({"fbx_global_scale": fbx_scale, "export_profile": FBX_EXPORT_PROFILE}, indent=2),
            encoding="utf-8",
        )
        return animation_fbx

    def _prepare_animation_from_genesis_blender_fbx(self, motion_payload: dict, ue_avatar: dict) -> HumanAnimationAssets:
        """Genesis scene-fit bundle + Blender FBX only (single BE_IBL editor, no retargeting.uproject)."""
        motion_source_id = str(motion_payload["source_id"])
        bundle_dir, bundle_npz, bundle_meta = self._export_motion_bundle(motion_payload)
        animation_fbx = self._export_animation_fbx(bundle_dir, motion_source_id)
        imported_animation_path = self._import_animation_asset(
            animation_fbx,
            motion_payload["source_id"],
            ue_avatar["skeletal_mesh_path"],
            ue_avatar["imported_fbx_root"],
        )
        return HumanAnimationAssets(
            motion_source_id=str(motion_payload["source_id"]),
            source_sequence_npz=str(motion_payload["sequence_npz_path"]),
            source_mesh_manifest=str(motion_payload["mesh_manifest_path"]),
            bundle_dir=str(bundle_dir),
            bundle_npz=str(bundle_npz),
            bundle_meta=str(bundle_meta),
            animation_fbx=str(animation_fbx),
            imported_animation_path=str(imported_animation_path),
            skeletal_mesh_path=str(ue_avatar["skeletal_mesh_path"]),
            texture_body=ue_avatar["texture_body"],
            texture_clothing=ue_avatar["texture_clothing"],
            texture_clothing_overlay=ue_avatar["texture_clothing_overlay"],
        )

    def _import_animation_asset(
        self,
        animation_fbx: Path,
        motion_source_id: str,
        skeletal_mesh_path: str,
        imported_fbx_root: str,
        *,
        fbx_import_root_motion: bool = True,
        sequence_enable_root_motion: bool = True,
    ) -> str:
        skeletal_mesh = unreal.load_asset(skeletal_mesh_path)
        if skeletal_mesh is None:
            raise RuntimeError(f"Cannot load BEDLAM skeletal mesh: {skeletal_mesh_path}")
        skeleton = _skeleton_from_mesh(skeletal_mesh)
        destination_path = f"{imported_fbx_root}/{_sanitize_name(motion_source_id)}"
        unreal.EditorAssetLibrary.make_directory(destination_path)
        destination_name = f"{_sanitize_name(motion_source_id)}_Anim"
        expected_asset = f"{destination_path}/{destination_name}"
        if unreal.EditorAssetLibrary.does_asset_exist(expected_asset) and self.config.force_rebuild_motion:
            unreal.EditorAssetLibrary.delete_asset(expected_asset)

        options = unreal.FbxImportUI()
        options.import_mesh = False
        options.import_as_skeletal = True
        options.import_animations = True
        options.mesh_type_to_import = unreal.FBXImportType.FBXIT_ANIMATION
        options.skeleton = skeleton
        _apply_fbx_anim_length_import(options)
        try:
            asd = getattr(options, "anim_sequence_import_data", None)
            if asd is not None:
                want_rm = bool(fbx_import_root_motion)
                for prop in ("import_root_motion", "b_import_root_motion"):
                    if hasattr(asd, prop):
                        try:
                            asd.set_editor_property(prop, want_rm)
                        except Exception:
                            try:
                                setattr(asd, prop, want_rm)
                            except Exception:
                                pass
                        unreal.log(f"UE_PIPELINE: FBX anim import set {prop}={want_rm}")
                        break
        except Exception as exc:
            unreal.log_warning(f"UE_PIPELINE: FBX root motion import flags skipped: {exc!r}")

        task = unreal.AssetImportTask()
        task.filename = str(animation_fbx)
        task.destination_path = destination_path
        task.destination_name = destination_name
        task.automated = True
        task.save = True
        task.replace_existing = True
        task.options = options
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        imported_paths = [str(item) for item in task.get_editor_property("imported_object_paths")]
        for imported_path in imported_paths:
            asset = unreal.load_asset(imported_path)
            if asset is not None and asset.get_class().get_name() == "AnimSequence":
                _configure_anim_sequence_root_motion(asset, enable_root_motion=bool(sequence_enable_root_motion))
                return imported_path
        if unreal.EditorAssetLibrary.does_asset_exist(expected_asset):
            _configure_anim_sequence_root_motion(
                unreal.load_asset(expected_asset),
                enable_root_motion=bool(sequence_enable_root_motion),
            )
            return expected_asset
        raise RuntimeError(f"Failed to import animation asset from FBX: {animation_fbx}")

    def _prepare_animation_from_official_fbx(self, motion_payload: dict, ue_avatar: dict) -> HumanAnimationAssets:
        animation_fbx = self._resolve_official_retarget_fbx_path(motion_payload, ue_avatar)
        imported_animation_path = self._import_animation_asset(
            animation_fbx,
            motion_payload["source_id"],
            ue_avatar["skeletal_mesh_path"],
            ue_avatar["imported_fbx_root"],
            fbx_import_root_motion=False,
            sequence_enable_root_motion=False,
        )
        return HumanAnimationAssets(
            motion_source_id=str(motion_payload["source_id"]),
            source_sequence_npz=str(motion_payload["sequence_npz_path"]),
            source_mesh_manifest=str(motion_payload["mesh_manifest_path"]),
            bundle_dir="",
            bundle_npz="",
            bundle_meta="",
            animation_fbx=str(animation_fbx),
            imported_animation_path=str(imported_animation_path),
            skeletal_mesh_path=str(ue_avatar["skeletal_mesh_path"]),
            texture_body=ue_avatar["texture_body"],
            texture_clothing=ue_avatar["texture_clothing"],
            texture_clothing_overlay=ue_avatar["texture_clothing_overlay"],
        )

    def prepare_human_animation(self) -> None:
        unreal.log("UE_PIPELINE: prepare_human_animation")
        payload = self._require_scene_payload()
        motion_payload = self._verify_motion_payload()
        ue_avatar = payload["ue_avatar_payload"]
        body_mode = ue_avatar["body_mode"]
        if body_mode == "official_retargeted_overlay":
            self.animation_assets = self._prepare_animation_from_official_fbx(motion_payload, ue_avatar)
            self._prepare_human_pose_cache(motion_payload)
            return
        if body_mode != "retargeted_overlay":
            raise RuntimeError(f"Unsupported scene ue_avatar.body_mode: {body_mode}")
        self.animation_assets = self._prepare_animation_from_genesis_blender_fbx(motion_payload, ue_avatar)
        self._prepare_human_pose_cache(motion_payload)

    def _prepare_human_pose_cache(self, motion_payload: dict) -> None:
        if self.animation_assets is None:
            return
        if not scene_loader._amongus_truthy_env("AMONGUS_UE_DRIVE_HUMAN_BONES", default=True):
            return
        meta_path = Path(self.config.output_root) / "human_smpl_realtime_pose.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": "smpl_body_axis_angle_realtime",
                    "skeletal_mesh_path": str(self.animation_assets.skeletal_mesh_path),
                    "fps": float(motion_payload.get("fps", 30.0) or 30.0),
                    "frame_count": int(motion_payload.get("frame_count", 0) or 0),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        unreal.log(f"UE_PIPELINE: wrote SMPL realtime human metadata {meta_path}")

    def _clear_queue(self) -> unreal.MoviePipelineQueue:
        subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        queue = subsystem.get_queue()
        for job in list(queue.get_jobs()):
            queue.delete_job(job)
        return queue

    def _add_png_job(
        self,
        queue: unreal.MoviePipelineQueue,
        *,
        level_sequence_path: str,
        map_path: str,
        image_size: tuple[int, int],
    ) -> None:
        seq_name = level_sequence_path.rsplit("/", maxsplit=1)[-1]
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.set_editor_property("job_name", seq_name)
        job.set_editor_property("sequence", unreal.SoftObjectPath(_asset_object_path(level_sequence_path)))
        job.set_editor_property("map", unreal.SoftObjectPath(map_path))
        job.set_editor_property("author", "Cursor")

        config = job.get_configuration()
        jpg_setting = config.find_setting_by_class(unreal.MoviePipelineImageSequenceOutput_JPG)
        if jpg_setting is not None:
            config.remove_setting(jpg_setting)
        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)

        output_png_root = Path(self.config.output_root) / "png"
        (output_png_root / seq_name).mkdir(parents=True, exist_ok=True)
        output_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        output_setting.output_directory = unreal.DirectoryPath(str(output_png_root))
        output_setting.file_name_format = "{sequence_name}/{sequence_name}_{frame_number}"
        output_setting.output_resolution = unreal.IntPoint(image_size[0], image_size[1])
        output_setting.zero_pad_frame_numbers = 4
        output_setting.output_frame_step = int(self._require_scene_payload()["render_payload"]["ue_frame_step"])
        if hasattr(output_setting, "use_custom_playback_range"):
            output_setting.use_custom_playback_range = True
            output_setting.custom_start_frame = 0
            output_setting.custom_end_frame = max(int(self._require_scene_payload()["render_payload"]["ue_frame_count"]), 0)

        aa_setting = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
        aa_setting.spatial_sample_count = 1
        aa_setting.temporal_sample_count = 1
        aa_setting.override_anti_aliasing = True
        aa_setting.anti_aliasing_method = unreal.AntiAliasingMethod.AAM_NONE
        aa_setting.render_warm_up_frames = False
        aa_setting.engine_warm_up_count = 0
        config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
        go_class = getattr(unreal, "MoviePipelineGameOverrideSetting", None)
        lod_zero_prop_set: str | None = None
        hlod_prop_set: str | None = None
        if go_class is not None:
            game_override = config.find_or_add_setting_by_class(go_class)
            for prop, val in (("b_use_lod_zero", True), ("bUseLODZero", True)):
                try:
                    game_override.set_editor_property(prop, val)
                    lod_zero_prop_set = prop
                    break
                except Exception:
                    continue
            for prop, val in (("b_disable_hlod", True), ("bDisableHLODs", True)):
                try:
                    game_override.set_editor_property(prop, val)
                    hlod_prop_set = prop
                    break
                except Exception:
                    continue
            unreal.log(
                f"UE_SYNC_DIAG M11 MRQ MoviePipelineGameOverrideSetting "
                f"class_ok=True lod_zero_prop={lod_zero_prop_set!r} hlod_prop={hlod_prop_set!r} job={seq_name}"
            )
        else:
            unreal.log_warning(
                f"UE_SYNC_DIAG M11 MRQ MoviePipelineGameOverrideSetting missing on this engine build job={seq_name}"
            )

    def _body_material(self):
        if self.animation_assets is None or self.animation_assets.texture_body is None:
            return None
        material_asset_path = f"{sequence_tools.material_body_root}/MI_{self.animation_assets.texture_body}"
        material, _, _ = _resolve_ue_asset(
            sequence_tools.material_body_root,
            f"MI_{self.animation_assets.texture_body}",
            class_names=("MaterialInstanceConstant",),
            search_roots=(sequence_tools.material_body_root, "/Engine/PS/Meshcapade/SMPLX"),
        )
        if material is None:
            material = unreal.EditorAssetLibrary.load_asset(f"MaterialInstanceConstant'{material_asset_path}'")
        if material is None:
            raise RuntimeError(f"Cannot load BEDLAM body material: {material_asset_path}")
        return material

    def _clothing_material(self):
        if self.animation_assets is None:
            return None
        texture_clothing = self.animation_assets.texture_clothing or self.animation_assets.texture_clothing_overlay
        if texture_clothing is None:
            return None
        outfit_name = str(texture_clothing).split("_texture_")[0]
        material_asset_path = f"{BEDLAM_CLOTHING_MATERIAL_ROOT}/{outfit_name}/MI_{texture_clothing}"
        material, resolved, info = _resolve_ue_asset(
            f"{BEDLAM_CLOTHING_MATERIAL_ROOT}/{outfit_name}",
            f"MI_{texture_clothing}",
            class_names=("MaterialInstanceConstant",),
            search_roots=(
                f"{BEDLAM_CLOTHING_MATERIAL_ROOT}/{outfit_name}",
                BEDLAM_CLOTHING_MATERIAL_ROOT,
                "/Engine/PS/Bedlam/Clothing",
                "/Engine/PS/Bedlam",
            ),
        )
        if material is None:
            unreal.log_warning(
                "UE_PIPELINE: cannot load clothing material "
                f"{material_asset_path} search={json.dumps(info, ensure_ascii=True)}"
            )
        else:
            unreal.log(f"UE_PIPELINE: clothing material resolved {resolved}")
        return material

    def _texture_asset(self, root: str, name: str | None):
        if not name:
            return None, None, {"candidates": [], "search_roots": [], "found_by": None}
        asset, resolved, info = _resolve_ue_asset(
            root,
            name,
            class_names=("Texture2D",),
            search_roots=(
                root,
                "/Engine/PS/Bedlam/Clothing/MaterialsSMPLX",
                "/Engine/PS/Bedlam/Clothing",
                "/Engine/PS/Bedlam",
                "/Engine/PS/Meshcapade/SMPLX/Textures",
            ),
        )
        if asset is None:
            unreal.log_warning(
                "UE_PIPELINE: cannot load texture "
                f"{root}/{name} search={json.dumps(info, ensure_ascii=True)}"
            )
        else:
            unreal.log(f"UE_PIPELINE: texture resolved {resolved}")
        return asset, resolved, info

    def _dynamic_material_instance(self, component, material_index: int, source_material):
        for method_name, args in (
            ("create_dynamic_material_instance", (material_index, source_material)),
            ("create_and_set_material_instance_dynamic", (material_index,)),
        ):
            method = getattr(component, method_name, None)
            if method is None:
                continue
            try:
                instance = method(*args)
            except Exception:
                continue
            if instance is not None:
                return instance
        return None

    def _apply_overlay_textures(self, component) -> dict[str, object]:
        report: dict[str, object] = {
            "requested": False,
            "body_texture_loaded": False,
            "overlay_texture_loaded": False,
            "body_texture_path": None,
            "overlay_texture_path": None,
            "parameters_set": 0,
            "search": {},
        }
        if self.animation_assets is None or self.animation_assets.texture_clothing_overlay is None:
            return report
        report["requested"] = True
        body_texture, body_path, body_info = self._texture_asset(BEDLAM_TEXTURE_BODY_ROOT, self.animation_assets.texture_body)
        overlay_texture, overlay_path, overlay_info = self._texture_asset(
            BEDLAM_TEXTURE_CLOTHING_OVERLAY_ROOT,
            self.animation_assets.texture_clothing_overlay,
        )
        report["body_texture_loaded"] = body_texture is not None
        report["overlay_texture_loaded"] = overlay_texture is not None
        report["body_texture_path"] = body_path
        report["overlay_texture_path"] = overlay_path
        report["search"] = {"body": body_info, "overlay": overlay_info}
        if body_texture is None or overlay_texture is None:
            return report
        material_slots = component.get_num_materials() if hasattr(component, "get_num_materials") else 1
        for material_index in range(max(int(material_slots), 1)):
            source_material = component.get_material(material_index)
            if source_material is None:
                continue
            dynamic_material = self._dynamic_material_instance(component, material_index, source_material)
            if dynamic_material is None:
                continue
            for param_name, texture in (
                ("bodytexture", body_texture),
                ("BodyTexture", body_texture),
                ("clothingtextureoverlay", overlay_texture),
                ("ClothingTextureOverlay", overlay_texture),
            ):
                try:
                    dynamic_material.set_texture_parameter_value(param_name, texture)
                    report["parameters_set"] = int(report["parameters_set"]) + 1
                except Exception:
                    continue
        return report

    def _resolve_smpl_bundle_meta_path(self) -> Path | None:
        if self.animation_assets is None:
            return None
        candidates: list[Path] = []
        raw_meta = str(self.animation_assets.bundle_meta or "").strip()
        if raw_meta:
            candidates.append(Path(raw_meta).expanduser())
        raw_dir = str(self.animation_assets.bundle_dir or "").strip()
        if raw_dir:
            candidates.append(Path(raw_dir).expanduser() / "smpl_motion_bundle.json")
        tried: list[str] = []
        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError as exc:
                tried.append(f"{path} (resolve_err={exc!r})")
                continue
            tried.append(str(resolved))
            if resolved.is_file():
                return resolved
        unreal.log_warning(
            "UE_PIPELINE: smpl_motion_bundle.json not resolved; tried candidates: "
            + ("; ".join(tried) if tried else "(none — bundle_meta and bundle_dir empty)")
        )
        return None

    def _motion_bundle_uses_world_root(self) -> bool:
        if self.animation_assets is None:
            return False
        # Only the Blender SMPL-bundle FBX export bakes scene YAML offsets into root motion.
        # Official IK-retarget FBX uses AMASS/BEDLAM root translation; spawning at origin misaligns vs Genesis.
        try:
            anim_fbx = Path(str(self.animation_assets.animation_fbx)).expanduser().resolve()
            blender_fbx = self._animation_fbx_path(str(self.animation_assets.motion_source_id)).resolve()
        except OSError:
            return False
        if anim_fbx != blender_fbx:
            unreal.log(
                "UE_PIPELINE: human spawn uses human_anchor (animation_fbx is not Blender bundle export): "
                f"{anim_fbx}"
            )
            return False
        path = self._resolve_smpl_bundle_meta_path()
        if path is None:
            return False
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            unreal.log_warning(f"UE_PIPELINE: failed to read motion bundle meta {path}: {exc!r}")
            return False
        ok = int(meta.get("motion_bundle_format") or 0) >= 2
        if not ok:
            unreal.log_warning(
                f"UE_PIPELINE: motion_bundle_format={meta.get('motion_bundle_format')!r} in {path}; "
                "human spawn uses anchor."
            )
        return ok

    def _apply_visible_actor_appearance(self, component) -> None:
        material_slots = component.get_num_materials() if hasattr(component, "get_num_materials") else 1
        report: dict[str, object] = {
            "material_slots": int(material_slots),
            "texture_body": None if self.animation_assets is None else self.animation_assets.texture_body,
            "texture_clothing": None if self.animation_assets is None else self.animation_assets.texture_clothing,
            "texture_clothing_overlay": None if self.animation_assets is None else self.animation_assets.texture_clothing_overlay,
            "body_material_loaded": False,
            "clothing_material_loaded": False,
            "clothing_material_applied": False,
            "clothing_single_slot_fallback": False,
            "clothing_slots_applied": [],
        }
        body_material = self._body_material()
        if body_material is not None:
            report["body_material_loaded"] = True
            for material_index in range(max(int(material_slots), 1)):
                component.set_material(material_index, body_material)
        clothing_material = self._clothing_material()
        if clothing_material is not None and int(material_slots) > 1:
            report["clothing_material_loaded"] = True
            report["clothing_material_applied"] = True
            for material_index in range(1, int(material_slots)):
                component.set_material(material_index, clothing_material)
                report["clothing_slots_applied"].append(int(material_index))
        elif clothing_material is not None:
            report["clothing_material_loaded"] = True
            unreal.log_warning(
                "UE_PIPELINE: clothing material loaded but not applied because skeletal mesh has one material slot; "
                "using body material plus Texture2D overlay only"
            )
        report["overlay"] = self._apply_overlay_textures(component)
        report["appearance_ok"] = bool(
            report["clothing_material_applied"]
            or int((report.get("overlay") or {}).get("parameters_set", 0)) > 0
            or not (self.animation_assets and (self.animation_assets.texture_clothing or self.animation_assets.texture_clothing_overlay))
        )
        self.material_application = report
        unreal.log(f"UE_PIPELINE: material_application {json.dumps(report, ensure_ascii=True)}")

    def _ensure_visible_human_actor(self):
        if self.animation_assets is None:
            raise RuntimeError("Human animation assets are not prepared.")
        skeletal_mesh = unreal.load_asset(self.animation_assets.skeletal_mesh_path)
        if skeletal_mesh is None:
            raise RuntimeError(f"Cannot load skeletal mesh: {self.animation_assets.skeletal_mesh_path}")
        human_anchor_cm = [float(v) for v in self._require_scene_payload().get("human_anchor_cm", (0.0, 0.0, 0.0))]
        use_world_baked_root_motion = self._motion_bundle_uses_world_root()
        actor_spawn_cm = [0.0, 0.0, 0.0] if use_world_baked_root_motion else list(human_anchor_cm)
        actor_label = "GEN_visible_human"
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        drive_bones = scene_loader._amongus_truthy_env("AMONGUS_UE_DRIVE_HUMAN_BONES", default=True)

        actor = _find_level_actor(lambda item: str(item.get_actor_label()) == actor_label)
        if drive_bones:
            # Always (re)spawn as SkeletalMeshActor so PIE reliably duplicates the mesh component.
            if actor is not None:
                try:
                    cname = str(actor.get_class().get_name())
                except Exception:
                    cname = ""
                if "SkeletalMeshActor" not in cname:
                    try:
                        actor_subsystem.destroy_actor(actor)
                    except Exception:
                        pass
                    actor = None
            if actor is None:
                actor = actor_subsystem.spawn_actor_from_class(
                    unreal.SkeletalMeshActor, unreal.Vector(*actor_spawn_cm)
                )
            actor.set_actor_label(actor_label)
            actor.set_folder_path(f"{scene_loader.GENERATED_SCENE_FOLDER}/human")
            scene_loader._ensure_actor_movable(actor)
            actor.set_actor_location(unreal.Vector(*actor_spawn_cm), False, False)
            skeletal_component = getattr(actor, "skeletal_mesh_component", None)
            if skeletal_component is None:
                try:
                    skeletal_component = actor.get_component_by_class(unreal.SkeletalMeshComponent)
                except Exception:
                    skeletal_component = None
            if skeletal_component is not None:
                try:
                    skeletal_component.set_skeletal_mesh_asset(skeletal_mesh)
                except Exception:
                    try:
                        skeletal_component.set_skeletal_mesh(skeletal_mesh, True)
                    except Exception as exc:
                        raise RuntimeError(f"UE_PIPELINE: set_skeletal_mesh failed: {exc!r}") from exc
            av = self._require_scene_payload().get("ue_avatar_payload") or {}
            scale = float(av.get("fbx_global_scale", 100.0))
            relative_scale = scale / 100.0
            if skeletal_component is not None and hasattr(skeletal_component, "set_relative_scale3d"):
                try:
                    skeletal_component.set_relative_scale3d(
                        unreal.Vector(relative_scale, relative_scale, relative_scale)
                    )
                except Exception:
                    pass
            component = scene_loader._attach_poseable_human_component(
                actor=actor,
                skeletal_mesh=skeletal_mesh,
                relative_scale=relative_scale,
                hide_underlying_skeletal=True,
            )
            if component is None:
                raise RuntimeError("UE_PIPELINE: failed to attach PoseableMeshComponent for visible human.")
            scene_loader._VISIBLE_HUMAN_SKELETAL_MESH_PATH = str(self.animation_assets.skeletal_mesh_path)
            scene_loader._VISIBLE_HUMAN_RELATIVE_SCALE = float(relative_scale)
        else:
            if actor is not None:
                try:
                    cname = str(actor.get_class().get_name())
                except Exception:
                    cname = ""
                if "SkeletalMeshActor" not in cname:
                    try:
                        actor_subsystem.destroy_actor(actor)
                    except Exception:
                        pass
                    actor = None
            if actor is None:
                actor = actor_subsystem.spawn_actor_from_class(
                    unreal.SkeletalMeshActor,
                    unreal.Vector(*actor_spawn_cm),
                )
            actor.set_actor_label(actor_label)
            actor.set_folder_path(f"{scene_loader.GENERATED_SCENE_FOLDER}/human")
            actor.set_actor_location(unreal.Vector(*actor_spawn_cm), False, False)
            component = actor.skeletal_mesh_component
            component.set_skeletal_mesh(skeletal_mesh)

        self._apply_visible_actor_appearance(component)
        try:
            _set_component_anim_tick_always(component)
        except Exception as exc:
            unreal.log_warning(f"UE_PIPELINE: human anim tick option failed: {exc!r}")
        try:
            temp_actor_hidden = getattr(actor, "set_actor_hidden_in_game", None)
            if temp_actor_hidden is not None:
                temp_actor_hidden(False)
            component.set_visibility(True, True)
            if hasattr(component, "set_hidden_in_game"):
                component.set_hidden_in_game(False)
            if hasattr(component, "set_render_in_main_pass"):
                component.set_render_in_main_pass(True)
        except Exception as exc:
            unreal.log_warning(f"UE_PIPELINE: human visibility flags failed: {exc!r}")
        return actor, component, human_anchor_cm, actor_spawn_cm, use_world_baked_root_motion

    def _add_animation_track_to_binding(self, binding, animation) -> None:
        add_track = getattr(binding, "add_track", None)
        if add_track is not None:
            anim_track = add_track(unreal.MovieSceneSkeletalAnimationTrack)
        else:
            anim_track = unreal.MovieSceneBindingExtensions.add_track(binding, unreal.MovieSceneSkeletalAnimationTrack)
        add_section = getattr(anim_track, "add_section", None)
        anim_section = add_section() if add_section is not None else unreal.MovieSceneTrackExtensions.add_section(anim_track)
        anim_section.params.animation = animation
        anim_section.set_range(0, int(self._require_scene_payload()["render_payload"]["ue_frame_count"]))
        try:
            completion = getattr(unreal, "MovieSceneCompletionMode", None)
            if completion is not None and hasattr(completion, "KEEP_STATE"):
                opts = anim_section.get_editor_property("eval_options")
                opts.set_editor_property("completion_mode", completion.KEEP_STATE)
                anim_section.set_editor_property("eval_options", opts)
        except Exception as exc:
            unreal.log_warning(f"UE_PIPELINE: could not set anim section completion_mode: {exc}")

    def _bind_human_actor_to_sequence(self, level_sequence, actor, animation) -> bool:
        try:
            seq_ext = getattr(unreal, "MovieSceneSequenceExtensions", None)
            if seq_ext is not None and hasattr(seq_ext, "add_possessable"):
                binding = seq_ext.add_possessable(level_sequence, actor)
            else:
                binding = level_sequence.add_possessable(actor)
            self._add_animation_track_to_binding(binding, animation)
            return True
        except Exception as exc:
            unreal.log_warning(f"UE_PIPELINE: possessable human binding failed, fallback to spawnable: {exc!r}")
            return False

    def _ensure_visible_human_appearance_only(self) -> None:
        """Ensure GEN_visible_human exists with the correct skeletal mesh and Bedlam materials.

        Used when AMONGUS_UE_DRIVE_HUMAN_BONES=1 drives bones at runtime: we still need
        materials applied so the actor is not rendered with the default checker fallback.
        """
        if self.animation_assets is None:
            return
        try:
            actor, component, _, _, _ = self._ensure_visible_human_actor()
        except Exception as exc:
            unreal.log_warning(f"UE_PIPELINE: visible human appearance prep failed: {exc!r}")
            return
        animation_mode = getattr(unreal, "AnimationMode", None)
        custom_mode = (
            getattr(animation_mode, "ANIMATION_CUSTOM_MODE", None)
            if animation_mode is not None
            else None
        )
        if custom_mode is None and animation_mode is not None:
            custom_mode = getattr(animation_mode, "ANIMATION_SINGLE_NODE", None)
        if custom_mode is not None:
            try:
                component.set_animation_mode(custom_mode)
            except Exception:
                pass
        try:
            if hasattr(component, "stop"):
                component.stop()
        except Exception:
            pass
        try:
            if hasattr(component, "set_animation"):
                component.set_animation(None)
        except Exception:
            pass
        unreal.log(
            f"UE_PIPELINE: visible human appearance ensured "
            f"actor={actor.get_actor_label()} materials={self.material_application}"
        )

    def _add_visible_skeletal_animation(self, level_sequence_path: str) -> None:
        if scene_loader._amongus_truthy_env("AMONGUS_UE_DRIVE_HUMAN_BONES", default=True):
            self._ensure_visible_human_appearance_only()
            unreal.log(
                "UE_PIPELINE: skip LevelSequence human anim track; AMONGUS_UE_DRIVE_HUMAN_BONES enables Genesis bone sync"
            )
            return
        if self.animation_assets is None:
            raise RuntimeError("Human animation assets are not prepared.")
        level_sequence = unreal.load_asset(_asset_object_path(level_sequence_path))
        if level_sequence is None:
            raise RuntimeError(f"Cannot load level sequence: {level_sequence_path}")
        animation = unreal.load_asset(self.animation_assets.imported_animation_path)
        if animation is None:
            raise RuntimeError("Cannot load imported animation asset.")
        actor, component, human_anchor_cm, actor_spawn_cm, use_world_baked_root_motion = self._ensure_visible_human_actor()
        unreal.log(
            f"UE_PIPELINE: human spawn use_world_baked_root_motion={use_world_baked_root_motion} "
            f"actor_spawn_cm={actor_spawn_cm} meta={self._resolve_smpl_bundle_meta_path()}"
        )

        if not self._bind_human_actor_to_sequence(level_sequence, actor, animation):
            temp_actor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
                unreal.SkeletalMeshActor,
                unreal.Vector(*actor_spawn_cm),
            )
            temp_actor.set_actor_label(f"{Path(level_sequence_path).name}_Human")
            temp_component = temp_actor.skeletal_mesh_component
            temp_component.set_skeletal_mesh(component.get_skeletal_mesh_asset() if hasattr(component, "get_skeletal_mesh_asset") else component.skeletal_mesh)
            self._apply_visible_actor_appearance(temp_component)
            binding = level_sequence.add_spawnable_from_instance(temp_actor)
            unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(temp_actor)
            self._add_animation_track_to_binding(binding, animation)
        unreal.EditorAssetLibrary.save_asset(level_sequence_path)

    def build_sequences(self) -> None:
        unreal.log("UE_PIPELINE: build_sequences")
        payload = self._require_scene_payload()
        spawn_human = bool(payload["render_payload"].get("ue_spawn_human", True))
        if spawn_human and self.animation_assets is None:
            raise RuntimeError("Human animation assets must be prepared before building sequences.")
        if not spawn_human and self.animation_assets is not None:
            unreal.log_warning("UE_PIPELINE: ue_spawn_human=false but animation assets are set; human tracks skipped.")

        camera_actor = _require_actor(
            "CineCameraActor",
            lambda actor: actor.get_class().get_name() in {"BE_CineCameraActor_Blueprint_C", "CineCameraActor"},
        )
        ground_truth_logger_actor = _find_level_actor(lambda actor: actor.get_class().get_name() == "BE_GroundTruthLogger_C")
        camera_target_actor = _require_actor("BE_CameraTarget", lambda actor: actor.get_actor_label() == "BE_CameraTarget")
        camera_operator_actor = _require_actor(
            "BE_CameraOperator",
            lambda actor: actor.get_class().get_name() == "BE_CameraOperator_C",
        )

        sequence_specs = []
        motion_id = (
            str(self.animation_assets.motion_source_id)
            if self.animation_assets is not None
            else str(payload["motion_payload"]["source_id"])
        )
        sequence_prefix = _sanitize_name(motion_id)
        frame_count = int(payload["render_payload"]["ue_frame_count"])
        for camera_payload in payload["camera_payloads"]:
            sequence_specs.append(
                (
                    f"{sequence_prefix}_{camera_payload['name']}",
                    sequence_tools.ActorPose(
                        camera_payload["x"],
                        camera_payload["y"],
                        camera_payload["z"],
                        camera_payload["yaw"],
                        camera_payload["pitch"],
                        camera_payload["roll"],
                    ),
                    camera_payload["fov"],
                    camera_payload["res"],
                )
            )
        root = self.output_root
        (root / "png").mkdir(parents=True, exist_ok=True)
        for sequence_name, _, _, _ in sequence_specs:
            (root / "png" / sequence_name).mkdir(parents=True, exist_ok=True)

        for sequence_name, camera_pose, camera_hfov, _ in sequence_specs:
            unreal.log(f"UE_PIPELINE: creating {sequence_name}")
            ok = sequence_tools.add_level_sequence(
                sequence_name,
                camera_actor,
                camera_pose,
                ground_truth_logger_actor,
                camera_target_actor,
                camera_operator_actor,
                [],
                frame_count,
                payload["ue_hdri_name"],
                camera_hfov=camera_hfov,
                camera_movement_type="Default",
                camera_animations=None,
                cameraroot_yaw=None,
                cameraroot_location=None,
                time_of_day=None,
                sunsky_actor=None,
            )
            if not ok:
                raise RuntimeError(f"Failed to create sequence: {sequence_name}")
            if spawn_human:
                self._add_visible_skeletal_animation(f"/Game/Bedlam/LevelSequences/{sequence_name}")
            sequence_tools.cleanup_mask_layers()

        self.sequence_names = [name for name, _, _, _ in sequence_specs]
        self.queue = self._clear_queue()
        for sequence_name, _, _, image_res in sequence_specs:
            self._add_png_job(
                self.queue,
                level_sequence_path=f"/Game/Bedlam/LevelSequences/{sequence_name}",
                map_path=payload["ue_map"],
                image_size=image_res,
            )

        render_queue_tools.save_movie_render_queue(
            self.queue,
            0,
            render_queue_tools.movie_render_queue_root,
            render_queue_tools.movie_render_queue_template,
        )
        unreal.log(f"UE_PIPELINE: queue saved for {len(sequence_specs)} sequences")

    def _write_meta(self) -> None:
        payload = self._require_scene_payload()
        av = payload["ue_avatar_payload"]
        if self.animation_assets is not None:
            motion_meta = {
                "source_id": self.animation_assets.motion_source_id,
                "source_sequence_npz": self.animation_assets.source_sequence_npz,
                "source_mesh_manifest": self.animation_assets.source_mesh_manifest,
                "bundle_dir": self.animation_assets.bundle_dir,
                "bundle_npz": self.animation_assets.bundle_npz,
                "bundle_meta": self.animation_assets.bundle_meta,
                "animation_fbx": self.animation_assets.animation_fbx,
                "imported_animation_path": self.animation_assets.imported_animation_path,
                "frame_count": int(payload["motion_payload"]["frame_count"]),
                "fps": float(payload["motion_payload"]["fps"]),
                "frame_step": int(payload["motion_payload"]["frame_step"]),
            }
            ue_av = {
                "body_mode": av.get("body_mode"),
                "body_name": av.get("body_name"),
                "skeletal_mesh_path": self.animation_assets.skeletal_mesh_path,
                "texture_body": self.animation_assets.texture_body,
                "texture_clothing": self.animation_assets.texture_clothing,
                "texture_clothing_overlay": self.animation_assets.texture_clothing_overlay,
            }
        else:
            motion_meta = {
                "source_id": str(payload["motion_payload"]["source_id"]),
                "frame_count": int(payload["motion_payload"]["frame_count"]),
                "fps": float(payload["motion_payload"]["fps"]),
                "frame_step": int(payload["motion_payload"]["frame_step"]),
                "human_animation_skipped": True,
            }
            ue_av = {
                "body_mode": av.get("body_mode"),
                "body_name": av.get("body_name"),
                "skeletal_mesh_path": str(av["skeletal_mesh_path"]),
                "texture_body": av.get("texture_body"),
                "texture_clothing": av.get("texture_clothing"),
                "texture_clothing_overlay": av.get("texture_clothing_overlay"),
            }
        meta = {
            "scene_spec": self.config.scene_spec_path,
            "augmentation_spec": self.config.augmentation_spec_path,
            "scene_name": payload["scene_name"],
            "camera_names": [item["name"] for item in payload["camera_payloads"]],
            "human_anchor_cm": list(payload["human_anchor_cm"]),
            "human": payload.get("human_payload"),
            "robot_visual_debug": payload.get("robot_visual_debug"),
            "motion": motion_meta,
            "ue_avatar": ue_av,
            "ue_avatar_source": payload.get("ue_avatar_source"),
            "material_application": self.material_application,
            "render": payload["render_payload"],
            "augmentation": payload.get("augmentation_payload"),
            "sequence_names": list(self.sequence_names),
        }
        (self.output_root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def build_queue(self) -> None:
        unreal.log("UE_PIPELINE: build_queue")
        if self.queue is None:
            raise RuntimeError("Queue has not been prepared.")
        self._write_meta()

    def prepare(self) -> list[str]:
        unreal.log("UE_PIPELINE: prepare_start")
        self.prepare_scene()
        self.prepare_robot_assets()
        if bool(self._require_scene_payload()["render_payload"].get("ue_spawn_human", True)):
            self.prepare_human_animation()
        else:
            self.animation_assets = None
            unreal.log("UE_PIPELINE: ue_spawn_human=false; skip human motion import and skeletal tracks")
        self.build_sequences()
        self.build_queue()
        unreal.log("UE_PIPELINE: prepare_done")
        return list(self.sequence_names)

    def render(self) -> None:
        if self.queue is None:
            raise RuntimeError("Queue has not been prepared.")
        unreal.EditorPythonScripting.set_keep_python_script_alive(True)
        global PIPELINE_EXECUTOR
        PIPELINE_EXECUTOR = unreal.MoviePipelinePIEExecutor()
        PIPELINE_EXECUTOR.on_executor_finished_delegate.add_callable_unique(_on_queue_finished)
        subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        unreal.log(f"UE_PIPELINE: starting jobs={len(self.queue.get_jobs())}")
        subsystem.render_queue_with_executor_instance(PIPELINE_EXECUTOR)


def _on_queue_finished(executor, success):
    global PIPELINE_EXECUTOR
    quit_on_finish = bool(PIPELINE_RUNTIME_FLAGS.get("quit_editor_on_finish"))
    unreal.log(
        "UE_PIPELINE: queue finished "
        f"success={success} quit_editor_on_finish={quit_on_finish} "
        f"request_id={PIPELINE_RUNTIME_FLAGS.get('request_id')} "
        f"session_paths={PIPELINE_RUNTIME_FLAGS.get('session_paths')}"
    )
    _write_session_status(
        PIPELINE_RUNTIME_FLAGS.get("session_paths"),
        state="idle" if success else "error",
        ready=bool(success),
        detail="render_queue_finished",
    )
    _write_command_result(
        request_id=PIPELINE_RUNTIME_FLAGS.get("request_id"),
        success=bool(success),
        detail="render_queue_finished" if success else "render_queue_failed",
        payload={"sequence_names": list(PIPELINE_RUNTIME_FLAGS.get("sequence_names", []))},
    )
    PIPELINE_EXECUTOR = None
    if quit_on_finish:
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)
        unreal.SystemLibrary.execute_console_command(None, "QUIT_EDITOR")
    else:
        unreal.EditorPythonScripting.set_keep_python_script_alive(True)


def _parse_args(argv: list[str]) -> RenderConfig:
    parser = argparse.ArgumentParser(description="Single-entry UE pipeline for scene load, motion import, sequence build, and MRQ render.")
    parser.add_argument("--output-root", default=str(PROJECT_PATHS.outputs_root / "ue_render_session"))
    parser.add_argument("--scene-spec", default=str(PROJECT_PATHS.default_scene_spec_path))
    parser.add_argument("--augmentation-spec", default=None)
    parser.add_argument("--render-now", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-rebuild-motion", action="store_true")
    parser.add_argument("--quit-editor-on-finish", action="store_true")
    parser.add_argument("--keep-editor-open", action="store_true")
    parser.add_argument("--session-dir", type=str, default=None)
    parser.add_argument("--request-id", type=str, default=None)
    args = parser.parse_args(argv)
    render_now = True if args.render_now else None
    if args.prepare_only:
        render_now = False
    quit_editor_on_finish = bool(args.quit_editor_on_finish)
    if args.keep_editor_open:
        quit_editor_on_finish = False
    return RenderConfig(
        output_root=str(Path(args.output_root)),
        scene_spec_path=str(args.scene_spec),
        augmentation_spec_path=None if args.augmentation_spec is None else str(args.augmentation_spec),
        render_now=render_now,
        force_rebuild_motion=bool(args.force_rebuild_motion),
        quit_editor_on_finish=quit_editor_on_finish,
        session_dir=args.session_dir,
        request_id=args.request_id,
    )


def run_pipeline(config: RenderConfig) -> list[str]:
    PIPELINE_RUNTIME_FLAGS["quit_editor_on_finish"] = bool(config.quit_editor_on_finish)
    PIPELINE_RUNTIME_FLAGS["session_paths"] = config.session_dir
    PIPELINE_RUNTIME_FLAGS["request_id"] = config.request_id
    PIPELINE_RUNTIME_FLAGS["sequence_names"] = []
    _write_session_status(config.session_dir, state="loading", ready=False, detail="pipeline_start")
    unreal.log(
        f"UE_PIPELINE: main output_root={config.output_root} scene_spec={config.scene_spec_path} "
        f"augmentation_spec={config.augmentation_spec_path} "
        f"render_now={config.render_now} force_rebuild_motion={config.force_rebuild_motion} "
        f"quit_editor_on_finish={config.quit_editor_on_finish}"
    )
    pipeline = UeBedlamRenderPipeline(config)
    sequence_names = pipeline.prepare()
    PIPELINE_RUNTIME_FLAGS["sequence_names"] = list(sequence_names)
    payload = pipeline._require_scene_payload()
    render_now = config.render_now
    if render_now is None:
        render_now = bool(payload["render_payload"]["ue_render_now"])
    if render_now:
        _write_session_status(config.session_dir, state="rendering", ready=False, detail="render_queue_started")
        pipeline.render()
    else:
        _write_session_status(config.session_dir, state="ready", ready=True, detail="pipeline_prepared")
        _write_command_result(
            request_id=config.request_id,
            success=True,
            detail="pipeline_prepared",
            payload={"sequence_names": list(sequence_names)},
        )
        unreal.log("UE_PIPELINE: no render; editor remains open unless quit requested")
        if config.quit_editor_on_finish:
            unreal.SystemLibrary.execute_console_command(None, "QUIT_EDITOR")
    return sequence_names


def main() -> None:
    config = _parse_args(sys.argv[1:])
    run_pipeline(config)


if __name__ == "__main__":
    main()
