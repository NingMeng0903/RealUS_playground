from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def append_debug_log(*, location: str, message: str, data: dict[str, Any], run_id: str, hypothesis_id: str) -> None:
    raw = str(os.environ.get("AMONGUS_AGENT_TRACE_JSONL", "") or "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    payload = {
        "sessionId": "amongus_agent_trace",
        "runId": str(run_id),
        "hypothesisId": str(hypothesis_id),
        "location": str(location),
        "message": str(message),
        "data": dict(data),
        "timestamp": int(time.time() * 1000),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def append_cursor_debug_log(*, location: str, message: str, data: dict[str, Any], run_id: str, hypothesis_id: str) -> None:
    if str(os.environ.get("AMONGUS_CURSOR_DEBUG_TRACKING", "") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    payload = {
        "sessionId": "62415c",
        "runId": str(run_id),
        "hypothesisId": str(hypothesis_id),
        "location": str(location),
        "message": str(message),
        "data": dict(data),
        "timestamp": int(time.time() * 1000),
    }
    try:
        path = Path("/home/camp/.cursor/debug-logs/debug-62415c.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
