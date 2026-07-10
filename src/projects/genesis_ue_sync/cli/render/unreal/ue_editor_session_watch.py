from __future__ import annotations

import json
import os
import runpy
import socket
import sys
import threading
import time
from pathlib import Path

import unreal

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SRC_ROOT, SCRIPT_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.project import project_paths
from projects.genesis_ue_sync.integrations.ue import EditorCommandResult, EditorSessionPaths, EditorSessionStatus
import ue_common_scene_loader as scene_loader
from ue_bedlam_dual_cam_batch import RenderConfig, run_pipeline

PROJECT_PATHS = project_paths(__file__)
SESSION_WATCHER = None

CANONICAL_UDP_BIND_ENV = "AMONGUS_UE_CANONICAL_UDP_BIND"
CANONICAL_UDP_DEFAULT_HOST = "127.0.0.1"
CANONICAL_UDP_DEFAULT_PORT = 5601
CANONICAL_UDP_RECV_BUFFER = 1 << 16  # 64 KiB
CANONICAL_UDP_RESULT_EVERY_ENV = "AMONGUS_UE_CANONICAL_RESULT_EVERY"
CANONICAL_UDP_RESULT_DEFAULT_EVERY = 100


def _current_level_path() -> str:
    try:
        world = unreal.EditorLevelLibrary.get_editor_world()
        if world is None:
            return ""
        return str(world.get_path_name())
    except Exception:
        return ""


def _write_ready(paths: EditorSessionPaths, *, detail: str) -> None:
    EditorSessionStatus(
        state="ready",
        project_path=str(PROJECT_PATHS.bedlam_unreal_project_file),
        ready=True,
        level_path=_current_level_path(),
        detail=detail,
        process_pid=os.getpid(),
    ).save(paths)


def _apply_tool_env(tool_env: object) -> None:
    if not isinstance(tool_env, dict):
        return
    for key, value in tool_env.items():
        if value is None:
            continue
        os.environ[str(key)] = str(value)


def _apply_realtime_editor_cvars() -> dict[str, object]:
    disable = str(os.environ.get("AMONGUS_UE_DISABLE_BACKGROUND_THROTTLE", "1") or "1").strip().lower()
    if disable in {"0", "false", "no", "off"}:
        return {"applied": False, "reason": "disabled_by_env"}
    max_fps = str(os.environ.get("AMONGUS_UE_MAX_FPS", "120") or "120").strip() or "120"
    commands = ["Slate.bAllowThrottling 0", "r.VSync 0", f"t.MaxFPS {max_fps}"]
    applied: list[str] = []
    errors: list[str] = []
    for cmd in commands:
        try:
            unreal.SystemLibrary.execute_console_command(None, cmd)
            applied.append(cmd)
        except Exception as exc:
            errors.append(f"{cmd}: {exc!r}")
    return {"applied": bool(applied), "commands": applied, "errors": errors}


_CANONICAL_TICK_FILENAME_PREFIX = "canonical_tick__"


def _next_command_file(paths: EditorSessionPaths) -> Path | None:
    """Return the next command file to process, draining ``apply_canonical_scene_tick`` backlog.

    Canonical ticks are pure overwrites (latest joint/root state wins), so when Genesis publishes
    at 100 Hz and UE drains slower, we delete every queued canonical tick except the **newest** to
    keep the visual sync within one frame instead of building a multi-second backlog. Other command
    types (scene apply, render pipeline, etc.) are processed in order.
    """
    canonical_paths: list[Path] = []
    other_paths: list[Path] = []
    for path in sorted(paths.commands_dir.glob("*.json")):
        if path.name.startswith(_CANONICAL_TICK_FILENAME_PREFIX):
            canonical_paths.append(path)
        else:
            other_paths.append(path)
    if other_paths:
        return other_paths[0]
    if not canonical_paths:
        return None
    newest = canonical_paths[-1]
    for stale in canonical_paths[:-1]:
        try:
            stale.unlink(missing_ok=True)
        except OSError:
            continue
    return newest


def _render_config_from_command(payload: dict, session_dir: Path, request_id: str) -> RenderConfig:
    return RenderConfig(
        output_root=str(payload["output_root"]),
        scene_spec_path=str(payload["scene_spec_path"]),
        augmentation_spec_path=payload.get("augmentation_spec_path"),
        render_now=payload.get("render_now"),
        force_rebuild_motion=bool(payload.get("force_rebuild_motion", False)),
        quit_editor_on_finish=bool(payload.get("quit_editor_on_finish", False)),
        session_dir=str(session_dir),
        request_id=request_id,
    )


def _run_bedlam_level_sequence_csv(csv_path: str | Path, camera_movement_type: str = "Default") -> dict:
    script_path = PROJECT_PATHS.bedlam_engine_python_root / "create_level_sequences_csv.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing BEDLAM create_level_sequences_csv.py: {script_path}")
    old_argv = list(sys.argv)
    old_path = list(sys.path)
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    sys.argv = [str(script_path), str(Path(csv_path).expanduser().resolve()), str(camera_movement_type)]
    try:
        try:
            runpy.run_path(str(script_path), run_name="__main__")
        except SystemExit as exc:
            code = 0 if exc.code is None else int(exc.code)
            if code != 0:
                raise RuntimeError(f"BEDLAM LevelSequence generation failed with exit code {code}") from exc
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
    return {"csv_path": str(Path(csv_path).expanduser().resolve()), "camera_movement_type": str(camera_movement_type)}


def _level_sequence_asset_path(sequence_name: str) -> str:
    name = str(sequence_name).strip()
    if name.startswith("/"):
        return name
    return f"/Game/Bedlam/LevelSequences/{name}.{name}"


def _open_bedlam_level_sequence(sequence_names: list[str], *, play: bool = False) -> dict:
    if not sequence_names:
        raise RuntimeError("No BEDLAM LevelSequence name was provided.")
    asset_path = _level_sequence_asset_path(str(sequence_names[0]))
    sequence = unreal.load_asset(asset_path)
    if sequence is None:
        raise RuntimeError(f"Cannot load BEDLAM LevelSequence: {asset_path}")
    opened = False
    sequencer = getattr(unreal, "LevelSequenceEditorBlueprintLibrary", None)
    if sequencer is not None:
        open_fn = getattr(sequencer, "open_level_sequence", None)
        if open_fn is not None:
            open_fn(sequence)
            opened = True
        set_frame_fn = getattr(sequencer, "set_current_time", None)
        if set_frame_fn is not None:
            try:
                set_frame_fn(0)
            except Exception:
                pass
        if bool(play):
            play_fn = getattr(sequencer, "play", None)
            if play_fn is not None:
                try:
                    play_fn()
                except Exception:
                    pass
    if not opened:
        unreal.EditorAssetLibrary.sync_browser_to_objects([asset_path])
    return {"opened_sequence": asset_path, "played": bool(play), "opened_in_sequencer": bool(opened)}


def _clear_visible_preview_human() -> dict:
    destroyed = 0
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in list(actor_subsystem.get_all_level_actors()):
        try:
            label = str(actor.get_actor_label())
        except Exception:
            continue
        if label == "GEN_visible_human" or label.endswith("_Human"):
            actor_subsystem.destroy_actor(actor)
            destroyed += 1
    return {"destroyed_preview_humans": destroyed}


class EditorSessionWatcher:
    def __init__(self, session_dir: Path, *, poll_interval_s: float = 0.0) -> None:
        self.paths = EditorSessionPaths(session_dir.resolve())
        self.paths.ensure()
        # Default to 0 so file commands are picked up every Slate post-tick (canonical
        # ticks no longer use the file path; they arrive over UDP, see _udp_*).
        self.poll_interval_s = max(float(poll_interval_s), 0.0)
        self._callback_handle = None
        self._last_poll_time = 0.0
        self._busy = False
        self._active_render_request_id: str | None = None
        self._active_render_command_path: Path | None = None

        self._canonical_udp_sock: socket.socket | None = None
        self._canonical_udp_thread: threading.Thread | None = None
        self._canonical_udp_running = False
        self._canonical_lock = threading.Lock()
        self._latest_canonical_payload: dict | None = None
        self._latest_canonical_request_id: str | None = None
        self._latest_canonical_seq: int = 0
        self._canonical_apply_count: int = 0
        self._canonical_drop_count: int = 0
        try:
            self._canonical_diag_every = max(
                int(os.environ.get(CANONICAL_UDP_RESULT_EVERY_ENV, CANONICAL_UDP_RESULT_DEFAULT_EVERY)),
                0,
            )
        except ValueError:
            self._canonical_diag_every = CANONICAL_UDP_RESULT_DEFAULT_EVERY

    def start(self) -> None:
        unreal.EditorPythonScripting.set_keep_python_script_alive(True)
        register = getattr(unreal, "register_slate_post_tick_callback", None)
        if register is None:
            raise RuntimeError("Unreal Python does not expose register_slate_post_tick_callback on this build.")
        self._start_canonical_udp_listener()
        try:
            scene_loader.apply_level_editor_viewport_camera_speed_scale()
        except Exception as exc:
            unreal.log_warning(f"UE_SCENE: viewport camera speed scale skipped: {exc!r}")
        realtime = _apply_realtime_editor_cvars()
        if realtime.get("errors"):
            unreal.log_warning(f"UE_SCENE: realtime editor CVars partial failure: {realtime}")
        else:
            unreal.log(f"UE_SCENE: realtime editor CVars {realtime}")
        self._callback_handle = register(self._tick)
        _write_ready(self.paths, detail="watcher_started")
        unreal.log(f"UE session watcher started: {self.paths.root}")

    def stop(self) -> None:
        unregister = getattr(unreal, "unregister_slate_post_tick_callback", None)
        if unregister is not None and self._callback_handle is not None:
            unregister(self._callback_handle)
        self._callback_handle = None
        self._stop_canonical_udp_listener()
        unreal.EditorPythonScripting.set_keep_python_script_alive(False)
        EditorSessionStatus(
            state="stopped",
            project_path=str(PROJECT_PATHS.bedlam_unreal_project_file),
            ready=False,
            level_path=_current_level_path(),
            detail="watcher_stopped",
            process_pid=os.getpid(),
        ).save(self.paths)
        unreal.log("UE session watcher stopped.")

    def _resolve_canonical_udp_bind(self) -> tuple[str, int]:
        raw = str(os.environ.get(CANONICAL_UDP_BIND_ENV, "") or "").strip()
        if not raw:
            return CANONICAL_UDP_DEFAULT_HOST, CANONICAL_UDP_DEFAULT_PORT
        if "://" in raw:
            raw = raw.split("://", 1)[1]
        if ":" not in raw:
            return raw or CANONICAL_UDP_DEFAULT_HOST, CANONICAL_UDP_DEFAULT_PORT
        host, port_str = raw.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            port = CANONICAL_UDP_DEFAULT_PORT
        return host or CANONICAL_UDP_DEFAULT_HOST, port

    def _start_canonical_udp_listener(self) -> None:
        host, port = self._resolve_canonical_udp_bind()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Large kernel recv buffer so 100 Hz bursts do not drop while UE thread is busy.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        try:
            sock.bind((host, port))
        except OSError as exc:
            unreal.log_warning(
                f"UE_SCENE: canonical UDP bind failed on {host}:{port} ({exc!r}); "
                f"falling back to file-only canonical ticks."
            )
            try:
                sock.close()
            except OSError:
                pass
            return
        sock.settimeout(0.5)
        self._canonical_udp_sock = sock
        self._canonical_udp_running = True
        thread = threading.Thread(
            target=self._canonical_udp_loop,
            name="amongus-canonical-udp",
            daemon=True,
        )
        self._canonical_udp_thread = thread
        thread.start()
        unreal.log(
            f"UE_SCENE: canonical UDP listener bound on {host}:{port} (recv_every_tick=on)"
        )

    def _stop_canonical_udp_listener(self) -> None:
        self._canonical_udp_running = False
        sock = self._canonical_udp_sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        thread = self._canonical_udp_thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._canonical_udp_sock = None
        self._canonical_udp_thread = None

    def _canonical_udp_loop(self) -> None:
        sock = self._canonical_udp_sock
        if sock is None:
            return
        while self._canonical_udp_running:
            try:
                data, _addr = sock.recvfrom(CANONICAL_UDP_RECV_BUFFER)
            except (socket.timeout, BlockingIOError):
                continue
            except OSError:
                break
            if not data:
                continue
            try:
                packet = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            request_id = str(packet.get("request_id") or "")
            payload = packet.get("payload") if isinstance(packet, dict) else None
            if not isinstance(payload, dict):
                continue
            with self._canonical_lock:
                if self._latest_canonical_payload is not None:
                    self._canonical_drop_count += 1
                self._latest_canonical_payload = payload
                self._latest_canonical_request_id = request_id
                self._latest_canonical_seq += 1

    def _consume_latest_canonical_payload(self) -> tuple[dict | None, str | None, int]:
        with self._canonical_lock:
            payload = self._latest_canonical_payload
            request_id = self._latest_canonical_request_id
            seq = self._latest_canonical_seq
            self._latest_canonical_payload = None
            self._latest_canonical_request_id = None
        return payload, request_id, seq

    def _apply_udp_canonical(self, payload: dict, request_id: str | None) -> None:
        canonical_state = dict(payload.get("canonical_state") or {})
        try:
            detail = scene_loader.apply_canonical_scene_tick(canonical_state)
        except Exception as exc:
            unreal.log_warning(f"UE_SCENE: udp canonical tick failed: {exc!r}")
            return
        self._canonical_apply_count += 1
        if (
            self._canonical_diag_every > 0
            and request_id
            and (self._canonical_apply_count % self._canonical_diag_every == 0)
        ):
            try:
                detail.setdefault("canonical_udp_apply_count", int(self._canonical_apply_count))
                detail.setdefault("canonical_udp_drop_count", int(self._canonical_drop_count))
                EditorCommandResult(
                    request_id=str(request_id),
                    success=True,
                    detail="canonical_tick_applied",
                    payload=detail,
                ).write(self.paths)
            except Exception as exc:
                unreal.log_warning(f"UE_SCENE: udp canonical diag write failed: {exc!r}")

    def _tick(self, delta_seconds: float) -> None:
        now = time.monotonic()
        if self._busy:
            return
        if self._active_render_request_id is not None:
            self._last_poll_time = now
            self._finalize_active_render_if_done()
            return
        # 1) Hot path: apply newest UDP canonical state every UE frame, no file IO.
        canonical_payload, canonical_request_id, _seq = self._consume_latest_canonical_payload()
        if canonical_payload is not None:
            self._busy = True
            try:
                self._apply_udp_canonical(canonical_payload, canonical_request_id)
            finally:
                self._busy = False
            self._last_poll_time = now
            return
        # 2) Cold path: throttled file-based commands (scene apply, render pipeline, etc.).
        if now - self._last_poll_time < max(self.poll_interval_s, 0.0):
            return
        self._last_poll_time = now
        command_path = _next_command_file(self.paths)
        if command_path is None:
            return
        self._busy = True
        try:
            self._consume(command_path)
        finally:
            self._busy = False

    def _finalize_active_render_if_done(self) -> None:
        request_id = self._active_render_request_id
        if request_id is None:
            return
        result = EditorCommandResult.load(self.paths, request_id)
        if result is None:
            return
        command_path = self._active_render_command_path
        if command_path is not None:
            try:
                command_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._active_render_request_id = None
        self._active_render_command_path = None
        if result.success:
            unreal.EditorPythonScripting.set_keep_python_script_alive(True)
            _write_ready(self.paths, detail="render_complete")

    def _consume(self, command_path: Path) -> None:
        raw = command_path.read_text(encoding="utf-8").strip()
        if not raw:
            raise ValueError(f"UE session command file is empty: {command_path}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"UE session command file is not valid JSON: {command_path}") from exc
        request_id = str(payload["request_id"])
        command_type = str(payload["command_type"])
        os.environ["AMONGUS_UE_COMMAND_REQUEST_ID"] = request_id
        try:
            if command_type == "stop_session_watcher":
                command_path.unlink(missing_ok=True)
                EditorCommandResult(request_id=request_id, success=True, detail="watcher_stopping").write(self.paths)
                self.stop()
                return
            if command_type == "apply_scene_to_level":
                payload_data = dict(payload.get("payload", {}))
                _apply_tool_env(payload_data.get("tool_env"))
                result_payload = scene_loader.apply_scene_to_current_level(
                    payload_data["scene_spec_path"],
                    payload_data.get("augmentation_spec_path"),
                    preserve_visible_human=bool(payload_data.get("preserve_visible_human", True)),
                )
                EditorCommandResult(
                    request_id=request_id,
                    success=True,
                    detail="scene_applied",
                    payload=result_payload,
                ).write(self.paths)
                command_path.unlink(missing_ok=True)
                _write_ready(self.paths, detail="scene_applied")
                return
            if command_type == "create_bedlam_level_sequences":
                payload_data = dict(payload.get("payload", {}))
                _apply_tool_env(payload_data.get("tool_env"))
                if bool(payload_data.get("apply_scene", True)):
                    scene_loader.apply_scene_to_current_level(
                        payload_data["scene_spec_path"],
                        payload_data.get("augmentation_spec_path"),
                        preserve_visible_human=bool(payload_data.get("preserve_visible_human", False)),
                    )
                clear_report = _clear_visible_preview_human()
                result_payload = _run_bedlam_level_sequence_csv(
                    payload_data["csv_path"],
                    payload_data.get("camera_movement_type", "Default"),
                )
                result_payload["clear_preview_human"] = clear_report
                sequence_names = list(payload_data.get("sequence_names", []))
                result_payload["sequence_names"] = sequence_names
                if str(payload_data.get("action", "create")) in {"open", "play"}:
                    result_payload["open"] = _open_bedlam_level_sequence(
                        sequence_names,
                        play=str(payload_data.get("action", "create")) == "play",
                    )
                EditorCommandResult(
                    request_id=request_id,
                    success=True,
                    detail="bedlam_level_sequences_created",
                    payload=result_payload,
                ).write(self.paths)
                command_path.unlink(missing_ok=True)
                _write_ready(self.paths, detail="bedlam_level_sequences_created")
                return
            if command_type == "apply_canonical_scene_tick":
                payload_data = dict(payload.get("payload", {}))
                _apply_tool_env(payload_data.get("tool_env"))
                canonical_state = dict(payload_data.get("canonical_state") or {})
                detail = scene_loader.apply_canonical_scene_tick(canonical_state)
                EditorCommandResult(
                    request_id=request_id,
                    success=True,
                    detail="canonical_tick_applied",
                    payload=detail,
                ).write(self.paths)
                command_path.unlink(missing_ok=True)
                _write_ready(self.paths, detail="canonical_tick_applied")
                return
            if command_type == "update_urdf_robot_joints":
                import ue_urdf_visual_loader as urdf_art

                payload_data = dict(payload.get("payload", {}))
                robot_id = str(payload_data["robot_id"])
                joints = [float(v) for v in payload_data["joint_positions"]]
                detail = urdf_art.apply_articulated_robot_joints(robot_id, joints)
                EditorCommandResult(
                    request_id=request_id,
                    success=True,
                    detail="urdf_joints_updated",
                    payload=detail,
                ).write(self.paths)
                command_path.unlink(missing_ok=True)
                _write_ready(self.paths, detail="urdf_joints_updated")
                return
            if command_type != "prepare_render_pipeline":
                raise RuntimeError(f"Unsupported command_type: {command_type}")
            pipeline_payload = dict(payload.get("payload", {}))
            _apply_tool_env(pipeline_payload.get("tool_env"))
            EditorSessionStatus(
                state="busy",
                project_path=str(PROJECT_PATHS.bedlam_unreal_project_file),
                ready=False,
                level_path=_current_level_path(),
                detail=f"processing:{request_id}",
                process_pid=os.getpid(),
            ).save(self.paths)
            config = _render_config_from_command(pipeline_payload, self.paths.root, request_id)
            sequence_names = run_pipeline(config)
            if bool(config.render_now):
                self._active_render_request_id = request_id
                self._active_render_command_path = command_path
                return
            else:
                EditorCommandResult(
                    request_id=request_id,
                    success=True,
                    detail="command_completed",
                    payload={"sequence_names": sequence_names},
                ).write(self.paths)
            command_path.unlink(missing_ok=True)
            _write_ready(self.paths, detail="watcher_idle")
        except Exception as exc:
            import traceback

            unreal.log_error(
                f"UE session command failed request_id={request_id}: {exc!r}\n{traceback.format_exc()}"
            )
            EditorCommandResult(
                request_id=request_id,
                success=False,
                detail=repr(exc),
            ).write(self.paths)
            EditorSessionStatus(
                state="error",
                project_path=str(PROJECT_PATHS.bedlam_unreal_project_file),
                ready=False,
                level_path=_current_level_path(),
                detail=repr(exc),
                process_pid=os.getpid(),
            ).save(self.paths)
            try:
                command_path.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> None:
    global SESSION_WATCHER
    if len(sys.argv) < 2:
        raise SystemExit("Usage: ue_editor_session_watch.py <session_dir> [poll_interval_s]")
    session_dir = Path(sys.argv[1]).expanduser().resolve()
    poll_interval_s = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.0
    SESSION_WATCHER = EditorSessionWatcher(session_dir, poll_interval_s=poll_interval_s)
    SESSION_WATCHER.start()


if __name__ == "__main__":
    main()
