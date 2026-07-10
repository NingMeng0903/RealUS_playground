from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def amongus_tool_env_for_ue_editor() -> dict[str, str]:
    """Snapshot AMONGUS_* from the host for injection into the long-lived UE Editor Python process."""
    return {str(k): str(v) for k, v in os.environ.items() if str(k).startswith("AMONGUS_") and str(v).strip()}


@dataclass(frozen=True)
class EditorSessionPaths:
    root: Path

    @property
    def status_file(self) -> Path:
        return self.root / "editor_status.json"

    @property
    def commands_dir(self) -> Path:
        return self.root / "commands"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.commands_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EditorSessionStatus:
    state: str
    project_path: str
    ready: bool
    level_path: str = ""
    detail: str = ""
    process_pid: int = 0
    updated_at_ns: int = field(default_factory=time.time_ns)

    def save(self, paths: EditorSessionPaths) -> Path:
        paths.ensure()
        payload = asdict(self)
        paths.status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return paths.status_file

    @classmethod
    def load(cls, paths: EditorSessionPaths) -> "EditorSessionStatus | None":
        if not paths.status_file.is_file():
            return None
        payload = json.loads(paths.status_file.read_text(encoding="utf-8"))
        return cls(**payload)


@dataclass
class EditorCommand:
    command_type: str
    payload: dict[str, Any]
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at_ns: int = field(default_factory=time.time_ns)

    def write(self, paths: EditorSessionPaths) -> Path:
        paths.ensure()
        command_path = paths.commands_dir / f"{self.created_at_ns}_{self.request_id}.json"
        command_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return command_path


@dataclass
class EditorCommandResult:
    request_id: str
    success: bool
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    updated_at_ns: int = field(default_factory=time.time_ns)

    def write(self, paths: EditorSessionPaths) -> Path:
        paths.ensure()
        result_path = paths.results_dir / f"{self.request_id}.json"
        result_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return result_path

    @classmethod
    def load(cls, paths: EditorSessionPaths, request_id: str) -> "EditorCommandResult | None":
        result_path = paths.results_dir / f"{request_id}.json"
        if not result_path.is_file():
            return None
        raw = result_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return cls(**payload)


def _process_is_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def editor_session_is_ready(
    paths: EditorSessionPaths,
    *,
    expected_project_path: str | Path | None = None,
) -> EditorSessionStatus | None:
    status = EditorSessionStatus.load(paths)
    if status is None or not status.ready:
        return None
    if str(getattr(status, "state", "") or "") == "error":
        return None
    if expected_project_path is not None and str(Path(status.project_path).resolve()) != str(Path(expected_project_path).resolve()):
        return None
    if status.process_pid and not _process_is_alive(status.process_pid):
        return None
    return status


def clear_session_editor_error(paths: EditorSessionPaths) -> bool:
    """Clear a stale ``state=error`` on editor_status so scene apply can retry without restarting UE."""
    status = EditorSessionStatus.load(paths)
    if status is None or str(getattr(status, "state", "") or "") != "error":
        return False
    if status.process_pid and not _process_is_alive(status.process_pid):
        return False
    status.state = "ready"
    status.ready = True
    status.detail = "session_error_cleared"
    status.save(paths)
    for pending in paths.results_dir.glob("*.json"):
        pending.unlink(missing_ok=True)
    return True


def session_editor_error_blocking(
    paths: EditorSessionPaths,
    *,
    expected_project_path: str | Path | None = None,
) -> str | None:
    """If a live editor PID still reports session error, return a user-facing message; else None."""
    status = EditorSessionStatus.load(paths)
    if status is None:
        return None
    if str(getattr(status, "state", "") or "") != "error":
        return None
    if not status.process_pid or not _process_is_alive(status.process_pid):
        return None
    if expected_project_path is not None and str(Path(status.project_path).resolve()) != str(Path(expected_project_path).resolve()):
        return None
    return (
        "Unreal editor is still running but the last session command failed (see editor_status.json and "
        f"{paths.root / 'editor_session.log'}). Close BE_IBL or use --clear-pending-commands and retry."
    )


def _build_execute_python_arg(script_path: Path, *args: str) -> str:
    joined = " ".join([str(script_path), *[str(arg) for arg in args]])
    return f"-ExecutePythonScript={joined}"


def _sanitize_env_for_ue_editor(base: dict[str, str] | None = None) -> dict[str, str]:
    """Strip ROS/conda pollution from LD_LIBRARY_PATH before spawning UnrealEditor.

    UE Vulkan RHI prepends its bundled loader to LD_LIBRARY_PATH. If /opt/ros/humble
    precedes the NVIDIA ICD, vpCreateInstance fails with VK_ERROR_INCOMPATIBLE_DRIVER.
    """
    env = dict(base if base is not None else os.environ)
    ld = str(env.get("LD_LIBRARY_PATH", "") or "")
    cleaned: list[str] = []
    for part in ld.split(":"):
        if not part:
            continue
        if "/opt/ros/" in part or part.startswith("/opt/ros"):
            continue
        cleaned.append(part)
    env["LD_LIBRARY_PATH"] = ":".join(cleaned)
    # .bashrc may set /etc/vulkan/... which does not exist on Ubuntu; force the system ICD.
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    return env


def _default_editor_extra_args() -> list[str]:
    args: list[str] = []
    raw_extra = str(os.environ.get("AMONGUS_UE_EDITOR_EXTRA_ARGS", "") or "").strip()
    if raw_extra:
        args.extend(raw_extra.split())
    disable_throttle = str(os.environ.get("AMONGUS_UE_DISABLE_BACKGROUND_THROTTLE", "1")).strip().lower()
    max_fps = str(os.environ.get("AMONGUS_UE_MAX_FPS", "120")).strip() or "120"
    if disable_throttle not in {"0", "false", "no", "off"}:
        exec_cmds = [
            "Slate.bAllowThrottling 0",
            "r.VSync 0",
            f"t.MaxFPS {max_fps}",
        ]
        args.append(f"-ExecCmds={';'.join(exec_cmds)}")
    return args


def launch_editor_session(
    *,
    unreal_cmd: str,
    project_path: str | Path,
    watcher_script: str | Path,
    session_dir: str | Path,
    extra_args: list[str] | None = None,
    log_path: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[Any]:
    cmd = [
        str(unreal_cmd),
        str(Path(project_path).resolve()),
        "-stdout",
        "-FullStdOutLogOutput",
        "-nop4",
        _build_execute_python_arg(Path(watcher_script).resolve(), str(Path(session_dir).resolve())),
    ]
    if extra_args:
        cmd.extend([str(item) for item in extra_args])
    else:
        cmd.extend(_default_editor_extra_args())
    target_log = Path(log_path).resolve() if log_path is not None else Path(session_dir).resolve() / "editor_session.log"
    target_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = target_log.open("ab")
    proc_env = _sanitize_env_for_ue_editor(env)
    return subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=proc_env,
        start_new_session=True,
    )


def ensure_editor_session(
    paths: EditorSessionPaths,
    *,
    unreal_cmd: str,
    project_path: str | Path,
    watcher_script: str | Path,
    extra_args: list[str] | None = None,
    log_path: str | Path | None = None,
    timeout_s: float = 300.0,
    poll_interval_s: float = 1.0,
) -> EditorSessionStatus:
    blocking = session_editor_error_blocking(paths, expected_project_path=project_path)
    if blocking is not None:
        raise RuntimeError(blocking)
    ready = editor_session_is_ready(paths, expected_project_path=project_path)
    if ready is not None:
        return ready
    launch_editor_session(
        unreal_cmd=unreal_cmd,
        project_path=project_path,
        watcher_script=watcher_script,
        session_dir=paths.root,
        extra_args=extra_args,
        log_path=log_path,
    )
    return wait_for_editor_ready(
        paths,
        expected_project_path=project_path,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )


def wait_for_editor_ready(
    paths: EditorSessionPaths,
    *,
    expected_project_path: str | Path | None = None,
    timeout_s: float = 300.0,
    poll_interval_s: float = 1.0,
) -> EditorSessionStatus:
    deadline = time.time() + timeout_s
    while True:
        status = editor_session_is_ready(paths, expected_project_path=expected_project_path)
        if status is not None:
            return status
        if time.time() >= deadline:
            expected = str(Path(expected_project_path).resolve()) if expected_project_path is not None else None
            project_hint = f" for project {expected}" if expected is not None else ""
            raise TimeoutError(
                "Timed out waiting for Unreal Editor ready status"
                f"{project_hint}. Start the target .uproject and run `src/projects/genesis_ue_sync/cli/render/unreal/ue_editor_session_watch.py` "
                "or `src/projects/genesis_ue_sync/cli/render/unreal/ue_editor_session_ready.py` inside the editor."
            )
        time.sleep(poll_interval_s)


def wait_for_command_result(
    paths: EditorSessionPaths,
    request_id: str,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 1.0,
) -> EditorCommandResult:
    deadline = time.time() + timeout_s
    while True:
        result = EditorCommandResult.load(paths, request_id)
        if result is not None:
            return result
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for Unreal Editor command result: {request_id}")
        time.sleep(poll_interval_s)


def enqueue_update_urdf_robot_joints(
    paths: EditorSessionPaths,
    robot_id: str,
    joint_positions: list[float],
) -> EditorCommand:
    """Queue joint-angle sync for a robot registered via AMONGUS_REGISTER_URDF_ARTICULATION_ID (radians)."""
    cmd = EditorCommand(
        command_type="update_urdf_robot_joints",
        payload={"robot_id": str(robot_id), "joint_positions": [float(v) for v in joint_positions]},
    )
    cmd.write(paths)
    return cmd


_CANONICAL_TICK_CMD_PREFIX = "canonical_tick__"
CANONICAL_UDP_DEFAULT_HOST = "127.0.0.1"
CANONICAL_UDP_DEFAULT_PORT = 5601


def enqueue_apply_canonical_scene_tick(paths: EditorSessionPaths, canonical_state: dict[str, Any]) -> EditorCommand:
    """File-based fallback path for ``apply_canonical_scene_tick`` (legacy / diagnostic).

    Coalesces the on-disk queue: at most one pending ``apply_canonical_scene_tick`` file at a time
    so the UE editor watcher always processes the **newest** state when forced through the slow
    JSON-file IPC. Production hot path goes through :class:`CanonicalSceneTickUdpSender` instead
    (UE-side UDP listener bound on ``AMONGUS_UE_CANONICAL_UDP_BIND``).
    """
    paths.ensure()
    pending = list(paths.commands_dir.glob(f"{_CANONICAL_TICK_CMD_PREFIX}*.json"))
    for stale in pending:
        try:
            stale.unlink(missing_ok=True)
        except OSError:
            continue
    cmd = EditorCommand(
        command_type="apply_canonical_scene_tick",
        payload={"canonical_state": dict(canonical_state)},
    )
    paths.ensure()
    command_path = paths.commands_dir / (
        f"{_CANONICAL_TICK_CMD_PREFIX}{cmd.created_at_ns}_{cmd.request_id}.json"
    )
    command_path.write_text(json.dumps(asdict(cmd), indent=2), encoding="utf-8")
    return cmd


def parse_canonical_udp_endpoint(raw: str | None) -> tuple[str, int]:
    """Parse ``host:port`` (or ``udp://host:port``) into ``(host, port)`` with safe defaults."""
    text = str(raw or "").strip()
    if not text:
        return CANONICAL_UDP_DEFAULT_HOST, CANONICAL_UDP_DEFAULT_PORT
    if "://" in text:
        text = text.split("://", 1)[1]
    if ":" not in text:
        return text or CANONICAL_UDP_DEFAULT_HOST, CANONICAL_UDP_DEFAULT_PORT
    host, port_str = text.rsplit(":", 1)
    try:
        port = int(port_str)
    except ValueError:
        port = CANONICAL_UDP_DEFAULT_PORT
    return host or CANONICAL_UDP_DEFAULT_HOST, port


class CanonicalSceneTickUdpSender:
    """Lock-free UDP sender for ``apply_canonical_scene_tick`` packets to the UE watcher.

    UE watcher binds an in-process UDP socket and consumes the newest packet on every Slate
    post-tick (~60 Hz on a healthy editor). Canonical state is overwrite-only (latest joint /
    root / SMPL pose wins), so packet loss between consecutive ticks is harmless.
    """

    def __init__(self, host: str = CANONICAL_UDP_DEFAULT_HOST, port: int = CANONICAL_UDP_DEFAULT_PORT) -> None:
        self.host = str(host)
        self.port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        except OSError:
            pass
        self._sock.setblocking(False)
        self._send_count = 0
        self._dropped_too_large = 0

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    @property
    def send_count(self) -> int:
        return int(self._send_count)

    @property
    def dropped_too_large(self) -> int:
        return int(self._dropped_too_large)

    def send_canonical_state(self, canonical_state: dict[str, Any]) -> str:
        request_id = str(uuid.uuid4())
        packet = {
            "request_id": request_id,
            "command_type": "apply_canonical_scene_tick",
            "payload": {"canonical_state": dict(canonical_state)},
            "created_at_ns": time.time_ns(),
        }
        body = json.dumps(packet).encode("utf-8")
        if len(body) > (CANONICAL_UDP_MAX_PACKET_BYTES):
            self._dropped_too_large += 1
            return request_id
        try:
            self._sock.sendto(body, (self.host, self.port))
            self._send_count += 1
        except (BlockingIOError, InterruptedError):
            self._dropped_too_large += 1
        except OSError:
            self._dropped_too_large += 1
        return request_id


CANONICAL_UDP_MAX_PACKET_BYTES = 60 * 1024  # stay below typical 64 KiB UDP datagram cap
