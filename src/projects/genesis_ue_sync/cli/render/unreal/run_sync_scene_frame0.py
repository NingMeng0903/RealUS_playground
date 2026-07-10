#!/usr/bin/env python3
"""Render UE frame0 PNGs for every camera in a SyncSceneSpec through a reusable editor session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.config.toolchain import discover_unreal_editor_executable
from projects.genesis_ue_sync.integrations.ue import (
    EditorCommand,
    EditorSessionPaths,
    amongus_tool_env_for_ue_editor,
    ensure_editor_session,
    wait_for_command_result,
)
from projects.genesis_ue_sync.cli.render.unreal.official_retarget_fbx_host import ensure_official_retarget_fbx_cached
from projects.genesis_ue_sync.sim_platform.scenes import (
    SceneAugmentationSpec,
    load_scene_augmentation_spec,
    merge_scene_augmentation_specs,
    scene_augmentation_to_dict,
)

PROJECT_PATHS = project_paths(__file__)
UE_WATCHER_SCRIPT = (
    REPO_ROOT
    / "src"
    / "projects"
    / "genesis_ue_sync"
    / "cli"
    / "render"
    / "unreal"
    / "ue_editor_session_watch.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-spec", type=Path, default=PROJECT_PATHS.default_scene_spec_path)
    parser.add_argument("--augmentation-spec", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_PATHS.tmp_root / "ue_frame0")
    parser.add_argument(
        "--retarget-cache-root",
        type=Path,
        default=PROJECT_PATHS.outputs_root / "ue_retarget_cache",
        help="Shared cache root for official retarget FBX builds. Reuse this to avoid launching the retarget project per output folder.",
    )
    parser.add_argument("--session-dir", type=Path, default=PROJECT_PATHS.outputs_root / "ue_sessions" / "be_ibl")
    parser.add_argument("--force-rebuild-motion", action="store_true")
    parser.add_argument(
        "--clear-pending-commands",
        action="store_true",
        help="Remove stale JSON files in session commands/ (fixes a stuck queue after a failed UE command).",
    )
    parser.add_argument("--unreal-editor", type=str, default=None)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    return parser.parse_args()


def _frame0_augmentation(base_path: Path | None, output_root: Path) -> Path:
    base = load_scene_augmentation_spec(base_path) if base_path is not None else None
    overlay = SceneAugmentationSpec(
        name="ue_frame0_override",
        render_override={
            "ue_render_now": True,
            "ue_frame_count": 1,
            "ue_frame_step": 1,
            "frame_limit": 1,
        },
        motion_override={
            "frame_count": 1,
            "start_frame": 0,
            "frame_step": 1,
        },
    )
    merged = merge_scene_augmentation_specs(base, overlay)
    generated_root = output_root / "_generated"
    generated_root.mkdir(parents=True, exist_ok=True)
    target = generated_root / "frame0_augmentation.json"
    target.write_text(json.dumps(scene_augmentation_to_dict(merged), indent=2), encoding="utf-8")
    return target


def _set_generated_fallback_animation(augmentation_path: Path, retarget_fbx: Path) -> None:
    payload = json.loads(augmentation_path.read_text(encoding="utf-8"))
    character_visual = dict(payload.get("character_visual", {}))
    character_visual["body_mode"] = "official_retargeted_overlay"
    character_visual["fallback_animation_path"] = str(retarget_fbx)
    payload["character_visual"] = character_visual
    augmentation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    unreal_cmd = args.unreal_editor or discover_unreal_editor_executable(PROJECT_PATHS)
    output_root = args.output_root.expanduser().resolve()
    augmentation_path = _frame0_augmentation(
        None if args.augmentation_spec is None else args.augmentation_spec.expanduser().resolve(),
        output_root,
    )
    scene_spec_resolved = args.scene_spec.expanduser().resolve()
    retarget_fbx = ensure_official_retarget_fbx_cached(
        scene_spec_path=scene_spec_resolved,
        augmentation_spec_path=augmentation_path,
        output_root=args.retarget_cache_root.expanduser().resolve(),
        force_rebuild=bool(args.force_rebuild_motion),
        unreal_cmd=unreal_cmd,
    )
    _set_generated_fallback_animation(augmentation_path, retarget_fbx)
    session_paths = EditorSessionPaths(args.session_dir.expanduser().resolve())
    if bool(args.clear_pending_commands):
        session_paths.ensure()
        for pending in session_paths.commands_dir.glob("*.json"):
            pending.unlink(missing_ok=True)
    ensure_editor_session(
        session_paths,
        unreal_cmd=unreal_cmd,
        project_path=PROJECT_PATHS.bedlam_unreal_project_file,
        watcher_script=UE_WATCHER_SCRIPT,
        log_path=session_paths.root / "editor_session.log",
        timeout_s=float(args.timeout_s),
    )
    command = EditorCommand(
        command_type="prepare_render_pipeline",
        payload={
            "output_root": str(output_root),
            "scene_spec_path": str(scene_spec_resolved),
            "augmentation_spec_path": str(augmentation_path),
            "render_now": True,
            "force_rebuild_motion": bool(args.force_rebuild_motion),
            "quit_editor_on_finish": False,
            "tool_env": amongus_tool_env_for_ue_editor(),
        },
    )
    command.write(session_paths)
    result = wait_for_command_result(session_paths, command.request_id, timeout_s=max(float(args.timeout_s), 3600.0))
    if not result.success:
        raise RuntimeError(f"UE frame0 session command failed: {result.detail}")
    print(f"UE frame0 rendered to {output_root}")
    print(f"  session: {session_paths.root}")
    print(f"  request_id: {command.request_id}")
    print(f"  generated_augmentation: {augmentation_path}")
    print(f"  sequences: {result.payload.get('sequence_names', [])}")


if __name__ == "__main__":
    main()
