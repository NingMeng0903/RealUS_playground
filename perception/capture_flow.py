"""Shared Window-8 SMPL-X capture: fit, preview PNGs, publish to Genesis."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, IO

DEFAULT_CAMERA_CONNECT = "tcp://127.0.0.1:17356"
DEFAULT_CONFIG_REL = Path("configs/tracking/realus_dwpose_easymocap.yaml")
_GENESIS_PY = Path("/media/camp/EXT_DRIVE/envs/genesis/bin/python")
_LOCK_NAME = ".capture.lock"
_COOLDOWN_S = 3.0


def repo_root() -> Path:
    return Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def smplx_output_root(repo: Path | None = None) -> Path:
    root = repo if repo is not None else repo_root()
    return Path(os.environ.get("REALUS_SMPLX_OUTPUT_ROOT", root / "smplx_outputs"))


def capture_cmd(
    repo: Path | None = None,
    *,
    run_name: str,
    camera_connect: str = DEFAULT_CAMERA_CONNECT,
    config: Path | None = None,
    extra_argv: list[str] | None = None,
) -> list[str]:
    root = repo if repo is not None else repo_root()
    py = str(_GENESIS_PY) if _GENESIS_PY.is_file() else os.environ.get("PY", sys.executable)
    cfg = Path(config) if config is not None else root / DEFAULT_CONFIG_REL
    cmd = [
        py,
        str(root / "perception/apps/run_smplx_capture.py"),
        "--config",
        str(cfg),
        "--connect",
        str(camera_connect),
        "--output-root",
        str(smplx_output_root(root)),
        "--run-name",
        str(run_name),
        "--write-debug-images",
        "--publish-genesis",
        "--publish-kind",
        "smplx_mesh",
    ]
    if extra_argv:
        cmd.extend(str(a) for a in extra_argv)
    return cmd


def capture_env(repo: Path | None = None) -> dict[str, str]:
    root = repo if repo is not None else repo_root()
    env = dict(os.environ)
    src = str((root / "src").resolve())
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}:{env['PYTHONPATH']}"
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Gamepad runs under rm75 without env.sh. ~/.local CPU onnxruntime must
    # not shadow genesis TensorRT/CUDA (same contract as env.sh).
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _tail_error(text: str) -> str:
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.startswith(("ValueError:", "ERROR ", "Error:", "RuntimeError:")):
            return s[:240]
    return ""


def is_capture_progress_line(line: str) -> bool:
    s = line.strip()
    return bool(
        s.startswith(("INFO ", "ERROR ", "WARNING ", "-> [", "ValueError"))
    )


@dataclass
class CaptureResult:
    run_name: str
    moment_dir: Path
    log_path: Path
    returncode: int = 1
    ok: bool = False
    quality_rejection: dict[str, object] | None = None
    error: str = ""


@dataclass
class CaptureStart:
    started: bool
    run_name: str = ""
    reason: str = ""
    moment_dir: Path | None = None


_job_lock = threading.Lock()
_job_running = False
_job_last_s = 0.0


def _lock_path(repo: Path) -> Path:
    return smplx_output_root(repo) / _LOCK_NAME


def _try_flock(path: Path) -> IO[str] | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def _quality_rejection(moment_dir: Path) -> dict[str, object] | None:
    summary_path = moment_dir / "moment.json"
    npz = moment_dir / "smplx_result.npz"
    if not summary_path.is_file() or not npz.is_file():
        return None
    # Burst diagnostics can be >100 MB; do not parse them on the pad thread.
    if summary_path.stat().st_size > 2_000_000:
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if bool(summary.get("fit_ok", True)):
        return None
    return dict(summary.get("final_quality") or {})


def run_smplx_capture(
    *,
    run_name: str,
    repo: Path | None = None,
    camera_connect: str = DEFAULT_CAMERA_CONNECT,
    config: Path | None = None,
    extra_argv: list[str] | None = None,
    timeout_s: float = 600.0,
    on_log: Callable[[str], None] | None = None,
) -> CaptureResult:
    root = repo if repo is not None else repo_root()
    out_root = smplx_output_root(root)
    moment_dir = out_root / run_name / "moment_0000"
    log_path = out_root / run_name / "capture_gui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = capture_cmd(
        root,
        run_name=run_name,
        camera_connect=camera_connect,
        config=config,
        extra_argv=extra_argv,
    )
    result = CaptureResult(run_name=run_name, moment_dir=moment_dir, log_path=log_path)
    chunks: list[str] = []
    proc: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"cmd: {' '.join(cmd)}\n\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                env=capture_env(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            deadline = time.monotonic() + float(timeout_s)
            for line in proc.stdout:
                log.write(line)
                chunks.append(line)
                if on_log is not None:
                    on_log(line.rstrip("\n"))
                if time.monotonic() > deadline:
                    proc.kill()
                    result.error = f"capture timed out after {timeout_s:.0f}s"
                    break
            rc = proc.wait(timeout=8.0)
        if not result.error:
            result.returncode = int(rc)
            result.ok = rc == 0 and (moment_dir / "smplx_result.npz").is_file()
            if not result.ok:
                result.quality_rejection = _quality_rejection(moment_dir)
                if result.quality_rejection is None and rc != 0:
                    result.error = _tail_error("".join(chunks)) or f"capture exit {rc}"
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            proc.kill()
        result.error = result.error or str(exc)
        log_path.write_text(
            log_path.read_text(encoding="utf-8") + f"\ncapture failed: {exc}\n"
            if log_path.is_file()
            else f"capture failed: {exc}\n",
            encoding="utf-8",
        )
    return result


def try_start_smplx_capture(
    *,
    label: str = "manual",
    repo: Path | None = None,
    camera_connect: str = DEFAULT_CAMERA_CONNECT,
    extra_argv: list[str] | None = None,
    cooldown_s: float = _COOLDOWN_S,
    on_done: Callable[[CaptureResult], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> CaptureStart:
    """Start one capture in a daemon thread. No-op if busy or in cooldown."""

    _ = label
    global _job_running, _job_last_s
    now = time.monotonic()
    root = repo if repo is not None else repo_root()
    with _job_lock:
        if _job_running:
            return CaptureStart(started=False, reason="busy")
        if now - _job_last_s < float(cooldown_s):
            return CaptureStart(started=False, reason="cooldown")
        lock_fh = _try_flock(_lock_path(root))
        if lock_fh is None:
            return CaptureStart(started=False, reason="busy")
        _job_running = True
        _job_last_s = now
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        moment_dir = smplx_output_root(root) / run_name / "moment_0000"

    def _worker() -> None:
        global _job_running
        try:
            result = run_smplx_capture(
                run_name=run_name,
                repo=root,
                camera_connect=camera_connect,
                extra_argv=extra_argv,
                on_log=on_log,
            )
        finally:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_fh.close()
            with _job_lock:
                _job_running = False
        if on_done is not None:
            on_done(result)

    threading.Thread(target=_worker, name="realus-smplx-capture", daemon=True).start()
    return CaptureStart(
        started=True,
        run_name=run_name,
        moment_dir=moment_dir,
    )
