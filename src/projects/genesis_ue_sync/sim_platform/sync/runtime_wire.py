"""Optional Genesis runtime observers for canonical state ZMQ / JSONL (host-side)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from projects.genesis_ue_sync.sim_platform.timebase import ClockService

_CANONICAL_ZMQ_DISABLE = frozenset({"0", "off", "false", "disable", "disabled", "none"})


def resolve_canonical_zmq_bind(*, default: str | None = None) -> str:
    """Resolve PUB bind endpoint from env, with optional default when unset."""
    raw = os.environ.get("AMONGUS_GENESIS_CANONICAL_ZMQ_BIND")
    if raw is not None:
        value = str(raw).strip()
        if value.lower() in _CANONICAL_ZMQ_DISABLE:
            return ""
        if value:
            return value
    if default is not None:
        value = str(default).strip()
        if value and value.lower() not in _CANONICAL_ZMQ_DISABLE:
            return value
    return ""


def resolve_canonical_state_jsonl_path() -> str:
    """Resolve optional JSONL audit path from env or session dir."""
    log_path = str(os.environ.get("AMONGUS_GENESIS_CANONICAL_STATE_JSONL", "") or "").strip()
    if log_path:
        return log_path
    session = str(os.environ.get("AMONGUS_SESSION_DIR", "") or os.environ.get("SESSION_DIR", "") or "").strip()
    if session:
        return str(Path(session).expanduser().resolve() / "genesis_canonical.jsonl")
    return ""


def attach_optional_canonical_observers(
    runtime: Any,
    *,
    default_zmq_bind: str | None = None,
) -> list[Any]:
    """Register tick observers when env/default is set; return handles to close (ZMQ publishers)."""
    handles: list[Any] = []
    bind = resolve_canonical_zmq_bind(default=default_zmq_bind)
    if bind:
        from projects.genesis_ue_sync.sim_platform.sync.zmq_pub import make_runtime_tick_publisher

        pub, cb = make_runtime_tick_publisher(zmq_endpoint=bind)
        runtime.register_sim_tick_observer(cb)
        handles.append(pub)

    log_path = resolve_canonical_state_jsonl_path()
    if log_path:
        from projects.genesis_ue_sync.sim_platform.state.canonical import snapshot_canonical_scene_state_v1
        from projects.genesis_ue_sync.sim_platform.sync.state_log import append_canonical_state_jsonl

        counter = {"n": 0}
        target = Path(log_path)

        def _log_cb(rt: Any) -> None:
            counter["n"] += 1
            clock = ClockService.from_runtime(rt).snapshot(int(counter["n"]))
            snap = snapshot_canonical_scene_state_v1(
                rt,
                sim_step_index=int(counter["n"]),
                wall_time_ns=clock.wall_time_ns,
                sim_time_ns=clock.sim_time_ns,
                source_time_ns=clock.source_time_ns,
                clock_domain=clock.clock_domain,
            )
            append_canonical_state_jsonl(target, snap)

        runtime.register_sim_tick_observer(_log_cb)

    return handles
