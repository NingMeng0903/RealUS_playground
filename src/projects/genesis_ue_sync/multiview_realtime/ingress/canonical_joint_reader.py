"""Non-blocking ZMQ reader for latest Franka joint positions from Genesis canonical state."""

from __future__ import annotations

import json
from typing import Any

from projects.genesis_ue_sync.tracking.robot_kinematic_mask.config import RobotKinematicMaskConfig


class CanonicalJointReader:
    """Subscribe to amongus_canonical_v1 and keep the newest robot joint vector."""

    def __init__(self, config: RobotKinematicMaskConfig) -> None:
        self._config = config
        self._ctx = None
        self._sock = None
        self._latest_joints: list[float] | None = None
        self._connected = False

    @property
    def joint_positions(self) -> list[float] | None:
        return None if self._latest_joints is None else list(self._latest_joints)

    def connect(self) -> None:
        if self._connected:
            return
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq is required for canonical joint reader.") from exc
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, 0)
        self._sock.connect(str(self._config.canonical_connect))
        self._sock.setsockopt(zmq.SUBSCRIBE, str(self._config.canonical_topic).encode("utf-8"))
        self._connected = True

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
            self._sock = None
        self._connected = False

    def _parse_joint_positions(self, payload: dict[str, Any]) -> list[float] | None:
        robots = dict(payload.get("robot_entities") or {})
        entity = robots.get(str(self._config.robot_entity_name))
        if not isinstance(entity, dict):
            return None
        raw = entity.get("joint_positions")
        if not isinstance(raw, list) or not raw:
            return None
        return [float(v) for v in raw]

    def poll(self) -> bool:
        """Drain all pending messages; return True if at least one joint update was parsed."""
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq is required for canonical joint reader.") from exc
        updated = False
        while True:
            try:
                parts = self._sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception:
                break
            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[-1].decode("utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            joints = self._parse_joint_positions(payload)
            if joints is not None:
                self._latest_joints = joints
                updated = True
        return updated
