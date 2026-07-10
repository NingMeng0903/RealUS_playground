"""Optional logging: Genesis canonical human root vs UE actor location (cm)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_DEBUG_NDJSON_LOG = _REPO_ROOT / ".cursor" / "debug-2f5e72.log"
_DEFAULT_DEBUG_SESSION_ID = "2f5e72"


def amongus_debug_ndjson_log_path() -> str:
    raw = str(os.environ.get("AMONGUS_DEBUG_NDJSON_LOG", "") or "").strip()
    return raw if raw else str(_DEFAULT_DEBUG_NDJSON_LOG)


def amongus_debug_session_id() -> str:
    raw = str(os.environ.get("AMONGUS_DEBUG_SESSION_ID", "") or "").strip()
    return raw if raw else _DEFAULT_DEBUG_SESSION_ID


def amongus_debug_ndjson_enabled() -> bool:
    v = str(os.environ.get("AMONGUS_DEBUG_NDJSON", "") or "").strip().lower()
    if v == "":
        return False
    return v not in ("0", "false", "off", "no")


def amongus_debug_ndjson_every(*, default: int = 20) -> int:
    raw = str(os.environ.get("AMONGUS_DEBUG_NDJSON_EVERY", "") or "").strip()
    if not raw:
        return max(int(default), 1)
    try:
        return max(int(raw), 1)
    except ValueError:
        return max(int(default), 1)


def amongus_debug_ndjson_target_paths() -> list[str]:
    """Primary log path plus optional mirror under AMONGUS_SESSION_DIR (for bridge + UE session bundles)."""
    primary = amongus_debug_ndjson_log_path()
    out = [primary]
    sr = str(os.environ.get("AMONGUS_SESSION_DIR", "") or "").strip()
    if sr:
        mp = str(Path(sr) / "amongus_human_debug.ndjson")
        if mp != primary:
            out.append(mp)
    return out


def _append_ndjson_line_all_targets(line: str) -> None:
    for path in amongus_debug_ndjson_target_paths():
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except Exception:
            pass


def agent_debug_ndjson(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: Mapping[str, Any],
    run_id: str = "full_diag",
) -> None:
    if not amongus_debug_ndjson_enabled():
        return
    # region agent log
    try:
        sid = amongus_debug_session_id()
        payload: dict[str, Any] = {
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": {"runId": run_id, **dict(data)},
            "timestamp": int(time.time() * 1000),
        }
        if sid:
            payload["sessionId"] = sid
        _append_ndjson_line_all_targets(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # endregion agent log


def human_align_diag_enabled() -> bool:
    v = str(os.environ.get("AMONGUS_HUMAN_ALIGN_DIAG", "") or "").strip().lower()
    return v not in ("", "0", "false", "off", "no")


def human_align_diag_every(default: int = 200) -> int:
    raw = str(os.environ.get("AMONGUS_HUMAN_ALIGN_DIAG_EVERY", "") or "").strip()
    if not raw:
        return max(int(default), 1)
    try:
        return max(int(raw), 1)
    except ValueError:
        return max(int(default), 1)


def human_root_genesis_vs_ue_line(root_m: Sequence[float], *, step: int | None = None) -> str:
    from bridge.adapters.ue import ue_world_point_from_genesis_m

    g = np.asarray(root_m, dtype=np.float64).reshape(3)
    u = ue_world_point_from_genesis_m(g)
    cm = u * 100.0
    step_s = f" step={step}" if step is not None else ""
    return (
        f"[amongus_human_align]{step_s} root_genesis_m=({g[0]:.5f},{g[1]:.5f},{g[2]:.5f}) "
        f"ue_actor_xyz_cm=({cm[0]:.5f},{cm[1]:.5f},{cm[2]:.5f})"
    )
