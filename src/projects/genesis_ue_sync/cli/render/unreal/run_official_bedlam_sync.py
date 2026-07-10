from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.config.toolchain import (
    discover_python_command,
    discover_unreal_editor_executable,
)
from projects.genesis_ue_sync.integrations.ue import (
    EditorCommand,
    EditorSessionPaths,
    amongus_tool_env_for_ue_editor,
    ensure_editor_session,
    wait_for_command_result,
    wait_for_editor_ready,
)
from projects.genesis_ue_sync.cli.render.unreal.official_retarget_fbx_host import ensure_official_retarget_fbx_cached
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_payload

PROJECT_PATHS = project_paths(__file__)

BE_IBL_PROJECT = PROJECT_PATHS.bedlam_unreal_project_file
UE_BATCH_SCRIPT = REPO_ROOT / "src" / "projects" / "genesis_ue_sync" / "cli" / "render" / "unreal" / "ue_bedlam_dual_cam_batch.py"
COMPOSE_VIDEO_SCRIPT = REPO_ROOT / "src" / "projects" / "genesis_ue_sync" / "cli" / "render" / "media" / "compose_bedlam_dual_cam_video.py"
UE_WATCHER_SCRIPT = REPO_ROOT / "src" / "projects" / "genesis_ue_sync" / "cli" / "render" / "unreal" / "ue_editor_session_watch.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official BEDLAM retargeting, then render the synced UE multi-view video."
    )
    parser.add_argument("--scene-spec", type=Path, required=True)
    parser.add_argument("--augmentation-spec", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--render-now", dest="render_now", action="store_true")
    parser.add_argument("--prepare-only", dest="render_now", action="store_false")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--keep-editor-open", action="store_true", help="Compatibility flag; the render editor stays open by default.")
    parser.add_argument("--close-editor-on-finish", action="store_true", help="Close BE_IBL after rendering finishes.")
    parser.add_argument(
        "--ue-session-dir",
        type=Path,
        default=PROJECT_PATHS.outputs_root / "ue_sessions" / "be_ibl",
        help="Reuse or launch a persistent BE_IBL session.",
    )
    parser.add_argument("--one-shot-editor", action="store_true", help="Launch BE_IBL directly instead of using the persistent session.")
    parser.set_defaults(render_now=True, force_rebuild=False)
    return parser.parse_args()


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or "motion"


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"[run] {printable}")
    subprocess.run(command, check=True, cwd=str(cwd) if cwd is not None else str(REPO_ROOT), env=env)

def _discover_python_cmd() -> list[str]:
    return discover_python_command()


def _discover_unreal_cmd() -> str:
    return discover_unreal_editor_executable(PROJECT_PATHS)


def _build_ue_python_arg(script_path: Path, *args: str) -> str:
    joined = " ".join([str(script_path), *[str(arg) for arg in args]])
    return f"-ExecutePythonScript={joined}"


def _dispatch_to_editor_session(
    *,
    session_dir: Path,
    output_root: Path,
    scene_spec_path: Path,
    augmentation_spec_path: Path | None,
    render_now: bool,
    force_rebuild_motion: bool,
    keep_editor_open: bool,
) -> None:
    session_paths = EditorSessionPaths(session_dir.resolve())
    wait_for_editor_ready(session_paths, expected_project_path=BE_IBL_PROJECT)
    command = EditorCommand(
        command_type="prepare_render_pipeline",
        payload={
            "output_root": str(output_root),
            "scene_spec_path": str(scene_spec_path),
            "augmentation_spec_path": None if augmentation_spec_path is None else str(augmentation_spec_path),
            "render_now": bool(render_now),
            "force_rebuild_motion": bool(force_rebuild_motion),
            "quit_editor_on_finish": not keep_editor_open,
            "tool_env": amongus_tool_env_for_ue_editor(),
        },
    )
    command.write(session_paths)
    result = wait_for_command_result(session_paths, command.request_id, timeout_s=3600.0)
    if not result.success:
        raise RuntimeError(f"Unreal Editor session command failed: {result.detail}")


def main() -> None:
    args = parse_args()
    unreal_cmd = _discover_unreal_cmd()

    scene_payload = load_sync_scene_payload(args.scene_spec)
    motion_payload = dict(scene_payload["motion"])
    source_id = str(motion_payload["source_id"])
    work_root = args.output_root.resolve()
    official_root = work_root / "official_retarget"
    render_root = work_root / "ue_render"

    scene_override = json.loads(json.dumps(scene_payload))
    scene_override.setdefault("ue_avatar", {})
    scene_override["ue_avatar"]["body_mode"] = "official_retargeted_overlay"
    scene_override["ue_avatar"]["imported_fbx_root"] = "/Game/Bedlam/Generated/OfficialRetargetedMotion"
    scene_override_path = official_root / "scene_official_retarget.json"
    scene_override_path.parent.mkdir(parents=True, exist_ok=True)
    scene_override_path.write_text(json.dumps(scene_override, indent=2), encoding="utf-8")
    retarget_fbx = ensure_official_retarget_fbx_cached(
        scene_spec_path=scene_override_path,
        augmentation_spec_path=None if args.augmentation_spec is None else args.augmentation_spec.expanduser().resolve(),
        output_root=work_root,
        force_rebuild=bool(args.force_rebuild),
        unreal_cmd=unreal_cmd,
    )
    if retarget_fbx is None:
        raise RuntimeError("Official retarget cache was not created for the official scene override.")
    scene_override["ue_avatar"]["fallback_animation_path"] = str(retarget_fbx)
    scene_override_path.write_text(json.dumps(scene_override, indent=2), encoding="utf-8")

    ue_batch_args = [
        "--scene-spec",
        str(scene_override_path),
        "--output-root",
        str(render_root),
    ]
    if args.augmentation_spec is not None:
        ue_batch_args.extend(["--augmentation-spec", str(args.augmentation_spec.expanduser().resolve())])
    if args.force_rebuild:
        ue_batch_args.append("--force-rebuild-motion")
    if args.render_now:
        ue_batch_args.append("--render-now")
    else:
        ue_batch_args.append("--prepare-only")
    close_editor_on_finish = bool(args.close_editor_on_finish) and not bool(args.keep_editor_open)
    if close_editor_on_finish:
        ue_batch_args.append("--quit-editor-on-finish")
    else:
        ue_batch_args.append("--keep-editor-open")
    if not args.one_shot_editor:
        session_paths = EditorSessionPaths(args.ue_session_dir.expanduser().resolve())
        ensure_editor_session(
            session_paths,
            unreal_cmd=unreal_cmd,
            project_path=BE_IBL_PROJECT,
            watcher_script=UE_WATCHER_SCRIPT,
            log_path=session_paths.root / "editor_session.log",
            timeout_s=300.0,
        )
        _dispatch_to_editor_session(
            session_dir=session_paths.root,
            output_root=render_root,
            scene_spec_path=scene_override_path,
            augmentation_spec_path=None if args.augmentation_spec is None else args.augmentation_spec.expanduser().resolve(),
            render_now=bool(args.render_now),
            force_rebuild_motion=bool(args.force_rebuild),
            keep_editor_open=not close_editor_on_finish,
        )
    else:
        render_cmd = [
            unreal_cmd,
            str(BE_IBL_PROJECT),
            "-stdout",
            "-FullStdOutLogOutput",
            "-unattended",
            "-nop4",
            _build_ue_python_arg(UE_BATCH_SCRIPT, *ue_batch_args),
        ]
        _run(render_cmd)

    sequence_root = _sanitize_name(source_id)
    camera_names = [str(item.get("name", "")).strip() for item in scene_override.get("cameras", [])]
    camera_names = [name for name in camera_names if name]
    if not camera_names:
        camera_names = ["cam_left", "cam_right"]
    seq_names = [f"{sequence_root}_{camera_name}" for camera_name in camera_names]
    compose_fps = int(round(float(scene_override.get("render", {}).get("fps", 8.0))))
    compose_env = {**os.environ, "AMONGUS_COMPOSE_FPS": str(max(compose_fps, 1))}
    _run(
        [*_discover_python_cmd(), str(COMPOSE_VIDEO_SCRIPT), str(render_root), *seq_names],
        cwd=REPO_ROOT,
        env=compose_env,
    )
    print(f"[done] multiview_strip={render_root / 'multiview_strip.mp4'}")


if __name__ == "__main__":
    main()
