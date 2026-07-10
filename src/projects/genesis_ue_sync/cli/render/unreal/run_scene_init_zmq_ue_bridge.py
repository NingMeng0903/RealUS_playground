"""Subscribe SceneInitMessageV1 over ZMQ and apply it inside the running UE editor session.

The UE editor watcher already understands the ``apply_scene_to_level`` and
``prepare_render_pipeline`` commands. This bridge sits between Genesis (which
publishes scene_init_v1 on a ZMQ PUB) and the UE session: on each new payload
(deduped by SHA256) it persists the scene_spec / augmentation_spec JSON into
``$SESSION_DIR/incoming/`` and enqueues a single editor command pointing at
those files. UE then builds the scene from scratch, including the Genesis-driven
AmongUs capture rig.

Two scene-apply modes are supported:

- ``apply`` (default): lightweight; static URDF spawn + canonical-driven joints.
  No Bedlam textured avatar, no animation playback.
- ``prepare``: first sends ``apply_scene_to_level`` so cameras/robot/static
  preview actors appear immediately, then invokes ``ensure_official_retarget_fbx_cached``
  on the host **once per new scene_init payload**, injects the resulting path into
  the augmentation spec as ``character_visual.fallback_animation_path``, and sends
  ``prepare_render_pipeline`` (with ``render_now=False``) so UE imports the FBX,
  spawns the textured ``GEN_visible_human``, and queues the AnimSequence.

Run alongside ``run_canonical_zmq_ue_bridge`` (per-frame state) once you have
started UE with ``run_ue_scene_session --watcher-only``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in _THIS.parents if parent.name == "src")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.integrations.controller_bus.stream_schemas import TOPIC_SCENE_INIT_V1
from projects.genesis_ue_sync.integrations.ue.session import (
    EditorCommand,
    EditorCommandResult,
    EditorSessionPaths,
    amongus_tool_env_for_ue_editor,
    clear_session_editor_error,
    session_editor_error_blocking,
    wait_for_command_result,
)
from projects.genesis_ue_sync.sim_platform.state.scene_init import (
    SceneInitMessageV1,
    scene_init_message_from_dict,
    write_scene_init_specs_to_session_dir,
)

PROJECT_PATHS = project_paths(__file__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--connect", type=str, default="tcp://127.0.0.1:5588", help="ZMQ SUB connect endpoint.")
    p.add_argument("--session-dir", type=Path, required=True, help="UE editor session directory.")
    p.add_argument("--topic", type=str, default=TOPIC_SCENE_INIT_V1)
    p.add_argument("--recv-timeout-ms", type=int, default=500)
    p.add_argument("--apply-timeout-s", type=float, default=1800.0, help="Max seconds to wait for the UE apply command result.")
    p.add_argument("--exit-after-first", action="store_true", help="Apply once then exit (vs stay subscribed for hot reapply).")
    p.add_argument(
        "--scene-apply-mode",
        type=str,
        choices=("apply", "prepare"),
        default="apply",
        help="apply = lightweight apply_scene_to_level (default); prepare = bake Bedlam FBX once and run prepare_render_pipeline (textured human + animation).",
    )
    p.add_argument(
        "--bake-output-root",
        type=Path,
        default=None,
        help="Cache directory for retargeted FBX (default: <repo>/outputs/ue_retarget_cache).",
    )
    p.add_argument(
        "--force-rebuild-bake",
        action="store_true",
        help="Force re-baking the retargeted FBX even if a cached copy exists.",
    )
    p.add_argument(
        "--clear-session-error",
        action="store_true",
        help="Clear stale editor_status error before applying (retry after a failed apply without restarting UE).",
    )
    return p.parse_args()


def _bake_visible_human_fbx(
    *,
    scene_spec_path: Path,
    augmentation_spec_path: Path | None,
    output_root: Path,
    force_rebuild: bool,
) -> Path | None:
    """Run the official Bedlam retarget bake on the host and return the FBX path (or None if N/A)."""
    from projects.genesis_ue_sync.cli.render.unreal.official_retarget_fbx_host import (
        ensure_official_retarget_fbx_cached,
    )

    return ensure_official_retarget_fbx_cached(
        scene_spec_path=scene_spec_path,
        augmentation_spec_path=augmentation_spec_path,
        output_root=output_root,
        force_rebuild=bool(force_rebuild),
    )


def _augmentation_with_baked_fbx(
    *,
    aug_spec_path: Path | None,
    baked_fbx: Path,
    session_root: Path,
) -> Path:
    """Materialise an augmentation spec JSON that points character_visual.fallback_animation_path at the baked FBX."""
    payload: dict
    if aug_spec_path is not None and aug_spec_path.is_file():
        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None  # type: ignore
        raw = aug_spec_path.read_text(encoding="utf-8")
        try:
            payload = yaml.safe_load(raw) if yaml is not None else json.loads(raw)
        except Exception:
            payload = json.loads(raw)
        if not isinstance(payload, dict):
            payload = {"name": "scene_init_session"}
    else:
        payload = {"name": "scene_init_session"}
    character_visual = dict(payload.get("character_visual") or {})
    character_visual["body_mode"] = "official_retargeted_overlay"
    character_visual["fallback_animation_path"] = str(baked_fbx)
    payload["character_visual"] = character_visual
    out_dir = session_root / "incoming"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "augmentation_with_baked_fbx.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _enqueue_apply(
    session_paths: EditorSessionPaths,
    message: SceneInitMessageV1,
    *,
    scene_apply_mode: str,
    bake_output_root: Path,
    force_rebuild_bake: bool,
    apply_timeout_s: float,
) -> tuple[EditorCommand, Path, Path | None]:
    scene_path, aug_path = write_scene_init_specs_to_session_dir(message, session_paths.root)

    if scene_apply_mode == "prepare":
        static_command = EditorCommand(
            command_type="apply_scene_to_level",
            payload={
                "scene_spec_path": str(scene_path),
                "augmentation_spec_path": None if aug_path is None else str(aug_path),
                "preserve_visible_human": False,
                "tool_env": amongus_tool_env_for_ue_editor(),
            },
        )
        static_command.write(session_paths)
        static_result = wait_for_command_result(
            session_paths,
            static_command.request_id,
            timeout_s=float(apply_timeout_s),
        )
        if not static_result.success:
            raise RuntimeError(f"UE static scene apply failed before prepare: {static_result.detail}")
        logging.info("static scene applied before retarget bake request_id=%s", static_command.request_id)

        baked_fbx = _bake_visible_human_fbx(
            scene_spec_path=scene_path,
            augmentation_spec_path=aug_path,
            output_root=bake_output_root,
            force_rebuild=force_rebuild_bake,
        )
        if baked_fbx is None:
            logging.warning(
                "scene_apply_mode=prepare requested but bake returned None (avatar body_mode != official_retargeted_overlay?); falling back to apply mode."
            )
            scene_apply_mode = "apply"
        else:
            aug_path = _augmentation_with_baked_fbx(
                aug_spec_path=aug_path,
                baked_fbx=baked_fbx,
                session_root=session_paths.root,
            )
            logging.info("baked retarget FBX -> %s", baked_fbx)

    if scene_apply_mode == "prepare":
        output_root = session_paths.root / "ue_render_session"
        output_root.mkdir(parents=True, exist_ok=True)
        command = EditorCommand(
            command_type="prepare_render_pipeline",
            payload={
                "output_root": str(output_root),
                "scene_spec_path": str(scene_path),
                "augmentation_spec_path": None if aug_path is None else str(aug_path),
                "render_now": False,
                "force_rebuild_motion": bool(force_rebuild_bake),
                "quit_editor_on_finish": False,
                "preserve_visible_human": False,
                "tool_env": amongus_tool_env_for_ue_editor(),
            },
        )
    else:
        command = EditorCommand(
            command_type="apply_scene_to_level",
            payload={
                "scene_spec_path": str(scene_path),
                "augmentation_spec_path": None if aug_path is None else str(aug_path),
                "preserve_visible_human": False,
                "tool_env": amongus_tool_env_for_ue_editor(),
            },
        )
    command.write(session_paths)
    return command, scene_path, aug_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        import zmq
    except ImportError as exc:
        logging.error("pyzmq required: %s", exc)
        return 2

    session_paths = EditorSessionPaths(args.session_dir.expanduser().resolve())
    session_paths.ensure()
    if args.clear_session_error and clear_session_editor_error(session_paths):
        logging.info("cleared stale session editor error; apply retry is allowed")
    topic_bytes = str(args.topic).encode("utf-8")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, int(max(args.recv_timeout_ms, 1)))
    sock.connect(str(args.connect))
    sock.setsockopt(zmq.SUBSCRIBE, topic_bytes)
    logging.info(
        "scene_init bridge connect=%s topic=%s session_dir=%s",
        args.connect,
        args.topic,
        session_paths.root,
    )

    bake_output_root = (
        args.bake_output_root.expanduser().resolve()
        if args.bake_output_root is not None
        else (PROJECT_PATHS.outputs_root / "ue_retarget_cache")
    )

    last_applied_hash = ""
    while True:
        try:
            parts = sock.recv_multipart()
        except zmq.Again:
            continue
        except zmq.ZMQError as exc:
            logging.warning("zmq error (retrying): %s", exc)
            time.sleep(0.5)
            continue

        if len(parts) < 2 or parts[0] != topic_bytes:
            continue
        try:
            payload = json.loads(parts[-1].decode("utf-8"))
        except json.JSONDecodeError as exc:
            logging.warning("skip bad scene_init json: %s", exc)
            continue

        try:
            message = scene_init_message_from_dict(payload)
        except Exception as exc:
            logging.warning("skip invalid scene_init message: %s", exc)
            continue

        if not message.payload_hash_sha256:
            logging.warning("scene_init payload missing hash; ignoring.")
            continue
        if message.payload_hash_sha256 == last_applied_hash:
            continue

        blocking = session_editor_error_blocking(session_paths)
        if blocking is not None:
            logging.error("%s", blocking)
            last_applied_hash = message.payload_hash_sha256
            if args.exit_after_first:
                return 1
            continue

        logging.info(
            "new scene_init received hash=%s session_id=%s mode=%s -> enqueue %s",
            message.payload_hash_sha256[:12],
            message.session_id or "(none)",
            args.scene_apply_mode,
            "prepare_render_pipeline" if args.scene_apply_mode == "prepare" else "apply_scene_to_level",
        )
        command, scene_path, aug_path = _enqueue_apply(
            session_paths,
            message,
            scene_apply_mode=str(args.scene_apply_mode),
            bake_output_root=bake_output_root,
            force_rebuild_bake=bool(args.force_rebuild_bake),
            apply_timeout_s=float(args.apply_timeout_s),
        )
        try:
            result = wait_for_command_result(
                session_paths,
                command.request_id,
                timeout_s=float(args.apply_timeout_s),
            )
        except TimeoutError as exc:
            logging.error("UE apply timed out (request_id=%s): %s", command.request_id, exc)
            last_applied_hash = message.payload_hash_sha256
            if args.exit_after_first:
                return 1
            continue
        if not result.success:
            logging.error(
                "UE apply failed (request_id=%s): %s — check %s/editor_session.log and editor_status.json; "
                "fix UE error then restart watcher or clear pending commands.",
                command.request_id,
                result.detail,
                session_paths.root,
            )
            last_applied_hash = message.payload_hash_sha256
            if args.exit_after_first:
                return 1
            continue

        payload_obj = result.payload or {}
        rig_summary = payload_obj.get("amongus_capture_rig") or {}
        articulation = payload_obj.get("urdf_articulation") or {}
        sequence_names = payload_obj.get("sequence_names") or []
        logging.info(
            "scene applied: scene_spec=%s aug_spec=%s articulation=%s capture_rig_installed=%s sequence_names=%s",
            scene_path.name,
            None if aug_path is None else aug_path.name,
            articulation,
            rig_summary.get("installed"),
            sequence_names if sequence_names else "(none)",
        )
        last_applied_hash = message.payload_hash_sha256
        if args.exit_after_first:
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
