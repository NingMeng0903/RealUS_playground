#!/usr/bin/env python3
"""Reuse or launch a UE editor session, then refresh the current sync scene render pipeline."""

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
from projects.genesis_ue_sync.cli.render.unreal.official_retarget_fbx_host import ensure_official_retarget_fbx_cached
from projects.genesis_ue_sync.integrations.ue import (
    EditorCommand,
    EditorSessionPaths,
    amongus_tool_env_for_ue_editor,
    ensure_editor_session,
    wait_for_command_result,
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
    parser.add_argument("--output-root", type=Path, default=PROJECT_PATHS.outputs_root / "ue_render_session")
    parser.add_argument("--session-dir", type=Path, default=PROJECT_PATHS.outputs_root / "ue_sessions" / "be_ibl")
    parser.add_argument(
        "--retarget-cache-root",
        type=Path,
        default=PROJECT_PATHS.outputs_root / "ue_retarget_cache",
        help="Shared cache root for official retarget FBX builds.",
    )
    parser.add_argument("--render-now", dest="render_now", action="store_true")
    parser.add_argument("--prepare-only", dest="render_now", action="store_false")
    parser.add_argument(
        "--apply-only",
        action="store_true",
        help=(
            "Only update static scene actors in the persistent editor level; skip retarget, animation import, "
            "Sequencer, and MRQ. Use --prepare-only when you need the human actor/animation loaded."
        ),
    )
    parser.add_argument(
        "--watcher-only",
        action="store_true",
        help=(
            "Launch UE editor with the session watcher attached but do not enqueue any apply/render command. "
            "Use this when scene construction is delegated to run_scene_init_zmq_ue_bridge (Genesis-as-source)."
        ),
    )
    parser.add_argument("--force-rebuild-motion", action="store_true")
    parser.add_argument(
        "--clear-pending-commands",
        action="store_true",
        help="Remove stale command JSON files and clear an old session error before dispatching.",
    )
    parser.add_argument("--unreal-editor", type=str, default=None)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.set_defaults(render_now=True)
    return parser.parse_args()


def _augmentation_with_fallback(base_path: Path | None, output_root: Path, retarget_fbx: Path) -> Path:
    payload = _augmentation_payload(base_path)
    character_visual = dict(payload.get("character_visual", {}))
    character_visual["body_mode"] = "official_retargeted_overlay"
    character_visual["fallback_animation_path"] = str(retarget_fbx)
    payload["character_visual"] = character_visual
    return _write_generated_augmentation(payload, output_root, "session_augmentation.json")


def _augmentation_payload(base_path: Path | None) -> dict:
    if base_path is not None:
        raw = base_path.read_text(encoding="utf-8")
        try:
            import yaml
        except ImportError:
            yaml = None
        payload = yaml.safe_load(raw) if yaml is not None else json.loads(raw)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected mapping augmentation: {base_path}")
    else:
        payload = {"name": "ue_session_generated"}
    return dict(payload)


def _write_generated_augmentation(payload: dict, output_root: Path, filename: str) -> Path:
    generated = output_root / "_generated" / filename
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return generated


def _augmentation_json_copy(base_path: Path | None, output_root: Path) -> Path | None:
    if base_path is None:
        return None
    return _write_generated_augmentation(_augmentation_payload(base_path), output_root, "session_apply_augmentation.json")


def main() -> None:
    args = parse_args()
    unreal_cmd = args.unreal_editor or discover_unreal_editor_executable(PROJECT_PATHS)
    output_root = args.output_root.expanduser().resolve()
    aug = None if args.augmentation_spec is None else args.augmentation_spec.expanduser().resolve()
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

    if bool(args.watcher_only):
        print(
            f"UE editor session ready (watcher-only) at {session_paths.root}; "
            "drive scene construction via run_scene_init_zmq_ue_bridge."
        )
        return

    scene_spec_path = args.scene_spec.expanduser().resolve()
    command_type = "apply_scene_to_level" if bool(args.apply_only) else "prepare_render_pipeline"
    command_aug = _augmentation_json_copy(aug, output_root) if bool(args.apply_only) else aug
    if not bool(args.apply_only):
        retarget_fbx = ensure_official_retarget_fbx_cached(
            scene_spec_path=scene_spec_path,
            augmentation_spec_path=aug,
            output_root=args.retarget_cache_root.expanduser().resolve(),
            force_rebuild=bool(args.force_rebuild_motion),
            unreal_cmd=unreal_cmd,
        )
        command_aug = _augmentation_with_fallback(aug, output_root, retarget_fbx)

    command = EditorCommand(
        command_type=command_type,
        payload={
            "output_root": str(output_root),
            "scene_spec_path": str(scene_spec_path),
            "augmentation_spec_path": None if command_aug is None else str(command_aug),
            "render_now": bool(args.render_now),
            "force_rebuild_motion": bool(args.force_rebuild_motion),
            "quit_editor_on_finish": False,
            "preserve_visible_human": bool(args.apply_only),
            "tool_env": amongus_tool_env_for_ue_editor(),
        },
    )
    command.write(session_paths)
    result = wait_for_command_result(session_paths, command.request_id, timeout_s=max(float(args.timeout_s), 3600.0))
    if not result.success:
        raise RuntimeError(f"UE scene session command failed: {result.detail}")
    print(
        f"UE scene refreshed via session {session_paths.root} "
        f"(request_id={command.request_id}, sequences={result.payload.get('sequence_names', [])})"
    )


if __name__ == "__main__":
    main()
