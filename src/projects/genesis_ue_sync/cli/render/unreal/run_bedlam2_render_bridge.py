#!/usr/bin/env python3
"""Generate official BEDLAM2 be_seq.csv and create GeometryCache LevelSequences in a persistent UE session."""

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
from projects.genesis_ue_sync.bedlam_render import write_bedlam_be_seq_csv
from projects.genesis_ue_sync.config.toolchain import discover_unreal_editor_executable
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
    parser.add_argument("--output-root", type=Path, default=PROJECT_PATHS.outputs_root / "bedlam2_render_bridge")
    parser.add_argument("--session-dir", type=Path, default=PROJECT_PATHS.outputs_root / "ue_sessions" / "be_ibl")
    parser.add_argument("--sequence-prefix", type=str, default=None)
    parser.add_argument(
        "--bedlam-body-name",
        type=str,
        default=None,
        help="Official BEDLAM GeometryCache body asset name to use as the dressed motion source.",
    )
    parser.add_argument(
        "--clothing-mode",
        choices=("geometry", "overlay", "auto"),
        default="geometry",
        help="BEDLAM dressed rendering mode. geometry loads *_clo GeometryCache; overlay uses BE_ClothingOverlayActor texture overlay.",
    )
    parser.add_argument("--camera-movement-type", type=str, default="Default")
    parser.add_argument(
        "--action",
        choices=("create", "open", "play"),
        default="open",
        help="After generating BEDLAM LevelSequences, optionally open/play the first sequence in Sequencer.",
    )
    parser.add_argument("--unreal-editor", type=str, default=None)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--clear-pending-commands", action="store_true")
    parser.add_argument(
        "--no-apply-scene",
        action="store_true",
        help="Do not refresh the persistent SyncScene actors before generating BEDLAM LevelSequences.",
    )
    parser.add_argument(
        "--preserve-preview-human",
        action="store_true",
        help="Keep the static GEN_visible_human preview actor when creating BEDLAM GeometryCache sequences.",
    )
    return parser.parse_args()


def _read_augmentation_payload(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        yaml = None
    payload = yaml.safe_load(raw) if yaml is not None else json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping augmentation spec: {path}")
    return dict(payload)


def _augmentation_json_copy(path: Path | None, output_root: Path) -> Path | None:
    if path is None:
        return None
    generated = output_root / "_generated" / "bedlam2_render_augmentation.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(json.dumps(_read_augmentation_payload(path), indent=2), encoding="utf-8")
    return generated


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    scene_spec_path = args.scene_spec.expanduser().resolve()
    augmentation_spec_path = None if args.augmentation_spec is None else args.augmentation_spec.expanduser().resolve()
    bridge_result = write_bedlam_be_seq_csv(
        scene_spec_path=scene_spec_path,
        augmentation_spec_path=augmentation_spec_path,
        output_dir=output_root,
        sequence_prefix=args.sequence_prefix,
        clothing_mode=str(args.clothing_mode),
        bedlam_body_name=args.bedlam_body_name,
    )
    ue_augmentation_spec_path = _augmentation_json_copy(augmentation_spec_path, output_root)

    unreal_cmd = args.unreal_editor or discover_unreal_editor_executable(PROJECT_PATHS)
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
        command_type="create_bedlam_level_sequences",
        payload={
            "scene_spec_path": str(scene_spec_path),
            "augmentation_spec_path": None if ue_augmentation_spec_path is None else str(ue_augmentation_spec_path),
            "csv_path": str(bridge_result.csv_path),
            "camera_movement_type": str(args.camera_movement_type),
            "sequence_names": list(bridge_result.sequence_names),
            "apply_scene": not bool(args.no_apply_scene),
            "action": str(args.action),
            "preserve_visible_human": bool(args.preserve_preview_human),
            "tool_env": amongus_tool_env_for_ue_editor(),
        },
    )
    command.write(session_paths)
    result = wait_for_command_result(session_paths, command.request_id, timeout_s=max(float(args.timeout_s), 3600.0))
    if not result.success:
        raise RuntimeError(f"BEDLAM2 render bridge command failed: {result.detail}")
    print(f"BEDLAM2 be_seq: {bridge_result.csv_path}")
    print(f"BEDLAM2 meta: {bridge_result.meta_path}")
    print(f"UE session: {session_paths.root}")
    print(f"request_id: {command.request_id}")
    print(f"sequences: {result.payload.get('sequence_names', bridge_result.sequence_names)}")
    if "open" in result.payload:
        print(f"opened: {result.payload['open']}")


if __name__ == "__main__":
    main()
