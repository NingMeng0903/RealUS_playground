"""Optional ZMQ PUB for canonical scene snapshots (host-side Genesis loop)."""

from __future__ import annotations

import json
from typing import Any, Callable

from projects.genesis_ue_sync.sim_platform.state.canonical import (
    CanonicalSceneStateV1,
    canonical_scene_state_to_dict,
    snapshot_canonical_scene_state_v1,
)
from projects.genesis_ue_sync.sim_platform.timebase import ClockService


class GenesisZmqStatePublisher:
    """Bind PUB socket; topic prefix separate from JSON payload."""

    def __init__(self, *, endpoint: str, topic: bytes = b"amongus_canonical_v1") -> None:
        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.bind(endpoint)
        self._topic = topic

    def publish_state(self, state: CanonicalSceneStateV1 | dict[str, Any]) -> None:
        payload = canonical_scene_state_to_dict(state) if isinstance(state, CanonicalSceneStateV1) else dict(state)
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self._sock.send_multipart([self._topic, body])

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass


def make_runtime_tick_publisher(
    *,
    zmq_endpoint: str,
    robot_names: list[str] | None = None,
) -> tuple[GenesisZmqStatePublisher, Callable[[Any], None]]:
    """Return publisher and a callback suitable for GenesisPlatformRuntime.register_sim_tick_observer."""
    pub = GenesisZmqStatePublisher(endpoint=zmq_endpoint)

    step_counter = {"n": 0}

    publish_errors = {"n": 0}

    def _cb(runtime: Any) -> None:
        step_counter["n"] += 1
        idx = int(step_counter["n"])
        try:
            clock = ClockService.from_runtime(runtime).snapshot(idx)
            snap = snapshot_canonical_scene_state_v1(
                runtime,
                robot_names=robot_names,
                sim_step_index=idx,
                wall_time_ns=clock.wall_time_ns,
                sim_time_ns=clock.sim_time_ns,
                source_time_ns=clock.source_time_ns,
                clock_domain=clock.clock_domain,
            )
            pub.publish_state(snap)
        except Exception as exc:
            publish_errors["n"] += 1
            if publish_errors["n"] <= 3 or publish_errors["n"] % 500 == 0:
                import logging

                logging.warning("canonical ZMQ publish failed (%s): %s", publish_errors["n"], exc)

    return pub, _cb
