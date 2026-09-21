"""Pad analysis session: hybrid jsonl plus the existing ICRA raw ultrasound H5.

Ordinary ``--hybrid-log`` stays control-only. ``--analysis`` starts one
``record.py --raw-only`` child per L3 so JPEG / TCP / force share a folder
with the pad jsonl. The pad process never decodes images.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


DEFAULT_RECORD_SCRIPT = Path("/media/camp/EXT_DRIVE/ICRA_YM/script/record.py")
DEFAULT_US_ENDPOINT = "tcp://127.0.0.1:17359"


def analysis_record_script() -> Path:
    env = os.environ.get("ICRA_SCRIPT_DIR", "").strip()
    if env:
        candidate = Path(env) / "record.py"
        if candidate.is_file():
            return candidate
    if DEFAULT_RECORD_SCRIPT.is_file():
        return DEFAULT_RECORD_SCRIPT
    raise FileNotFoundError(
        "pad analysis needs ICRA record.py; set ICRA_SCRIPT_DIR or install "
        f"{DEFAULT_RECORD_SCRIPT}"
    )


def allocate_pad_analysis_dir(repo, *, now=None) -> Path:
    """New directory under apps/logs/pad_hybrid/<stamp>[/_NNN]."""

    root = Path(repo) / "apps" / "logs" / "pad_hybrid"
    stamp = time.strftime("%Y%m%d_%H%M%S") if now is None else time.strftime(
        "%Y%m%d_%H%M%S", time.localtime(now)
    )
    root.mkdir(parents=True, exist_ok=True)
    for name in (stamp, *(f"{stamp}_{index:03d}" for index in range(1, 10000))):
        path = root / name
        try:
            path.mkdir()
            return path.resolve()
        except FileExistsError:
            continue
    raise ValueError(f"could not allocate pad analysis directory under {root}")


def write_analysis_session(directory, payload) -> Path:
    path = Path(directory) / "session.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(text)
    return path


class PadAnalysisRecorder:
    """One raw-only ICRA recorder child. Does not crop or delete the H5."""

    def __init__(
        self,
        *,
        python=None,
        repo=None,
        shm_prefix="",
        state_shm="rm75_state",
        force_shm=None,
        us_endpoint=DEFAULT_US_ENDPOINT,
        record_script=None,
    ):
        self.python = Path(python or os.environ.get("ICRA_RECORD_PYTHON") or sys.executable)
        self.repo = Path(repo)
        self.shm_prefix = str(shm_prefix or "")
        self.state_shm = str(state_shm)
        self.force_shm = force_shm
        self.us_endpoint = str(us_endpoint)
        self.record_script = Path(record_script) if record_script is not None else analysis_record_script()
        self.directory = None
        self.path = None
        self.proc = None
        self.log = None

    def start(self, directory, *, extra=None, timeout_s=8.0):
        if self.proc is not None:
            raise RuntimeError("pad analysis recorder already running")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        raw = directory / "raw.h5"
        log_path = directory / "recorder.log"
        command = [
            str(self.python), str(self.record_script),
            "--raw-only",
            "--output-file", str(raw),
            "--output-dir", str(directory),
            "--repo", str(self.repo),
            "--state-shm", self.state_shm,
            "--shm-prefix", self.shm_prefix,
            "--us-endpoint", self.us_endpoint,
            "--stream-timeout", "2.0",
            "--max-source-age", "0.5",
        ]
        if self.force_shm:
            command += ["--force-shm", str(self.force_shm)]
        self.log = log_path.open("w")
        try:
            self.proc = subprocess.Popen(
                command, cwd=str(self.record_script.parent),
                stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        except BaseException:
            self.log.close()
            self.log = None
            raise
        self.directory = directory
        self.path = raw
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            code = self.proc.poll()
            if code is not None:
                self.stop()
                raise RuntimeError(f"ultrasound recorder exited ({code}); see {log_path}")
            if raw.exists() and raw.stat().st_size > 0:
                payload = {
                    "schema": "pad_hybrid_analysis_v1",
                    "directory": str(directory),
                    "pad_jsonl": str(directory / "pad.jsonl"),
                    "raw_h5": str(raw),
                    "recorder_log": str(log_path),
                    "us_endpoint": self.us_endpoint,
                }
                if extra:
                    payload.update(extra)
                write_analysis_session(directory, payload)
                return raw
            time.sleep(0.05)
        self.stop()
        raise RuntimeError(f"ultrasound recorder did not create {raw}; see {log_path}")

    def close_if_inactive(self, hybrid_active):
        """Retire the raw session after an asynchronous controller exit/fault."""
        if hybrid_active or self.proc is None:
            return None
        path = self.path
        self.stop()
        return path

    def stop(self):
        proc = self.proc
        self.proc = None
        if proc is None:
            if self.log is not None:
                self.log.close()
                self.log = None
            self.path = self.directory = None
            return
        try:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
        finally:
            if self.log is not None:
                self.log.close()
                self.log = None
            self.directory = None
            self.path = None
