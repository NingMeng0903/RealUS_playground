from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EntityCommandOp = Literal["add", "update", "upsert", "move", "delete", "remove", "rename"]


@dataclass
class SceneEntityState:
    entity_id: str
    entity_type: str
    model_id: str = ""
    pose: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    time_ns: int = 0
    ttl_ns: int = 0


@dataclass
class SceneEntityCommand:
    op: EntityCommandOp
    entity_id: str
    entity_type: str = ""
    model_id: str = ""
    pose: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    new_entity_id: str = ""
    time_ns: int = 0
    ttl_ns: int = 0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SceneEntityCommand":
        payload = dict(data.get("payload", {}))
        if "ttl_ns" in data and "ttl_ns" not in payload:
            payload["ttl_ns"] = int(data.get("ttl_ns") or 0)
        if "time_ns" in data and "time_ns" not in payload:
            payload["time_ns"] = int(data.get("time_ns") or 0)
        return cls(
            op=str(data["op"]).strip().lower(),  # type: ignore[arg-type]
            entity_id=str(data["entity_id"]).strip(),
            entity_type=str(data.get("entity_type", "")).strip(),
            model_id=str(data.get("model_id", "")).strip(),
            pose=dict(data.get("pose", {})),
            payload=payload,
            new_entity_id=str(data.get("new_entity_id", "")).strip(),
            time_ns=int(data.get("time_ns") or payload.get("time_ns") or 0),
            ttl_ns=int(data.get("ttl_ns") or payload.get("ttl_ns") or 0),
        )


class SceneEntityRegistry:
    """Idempotent in-memory scene graph for add/update/move/delete/rename commands."""

    def __init__(self) -> None:
        self._entities: dict[str, SceneEntityState] = {}

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "entity_id": value.entity_id,
                "entity_type": value.entity_type,
                "model_id": value.model_id,
                "pose": dict(value.pose),
                "payload": dict(value.payload),
            }
            for key, value in sorted(self._entities.items())
        }

    def apply(self, command: SceneEntityCommand | dict[str, Any]) -> dict[str, Any]:
        cmd = command if isinstance(command, SceneEntityCommand) else SceneEntityCommand.from_mapping(command)
        if not cmd.entity_id:
            raise ValueError("entity_id must be non-empty.")
        if cmd.op in {"add", "update", "upsert"}:
            current = self._entities.get(cmd.entity_id)
            entity_type = cmd.entity_type or (current.entity_type if current is not None else "")
            model_id = cmd.model_id or (current.model_id if current is not None else "")
            pose = dict(current.pose) if current is not None else {}
            pose.update(cmd.pose)
            payload = dict(current.payload) if current is not None else {}
            payload.update(cmd.payload)
            if cmd.time_ns:
                payload["time_ns"] = int(cmd.time_ns)
            if cmd.ttl_ns:
                payload["ttl_ns"] = int(cmd.ttl_ns)
            self._entities[cmd.entity_id] = SceneEntityState(
                entity_id=cmd.entity_id,
                entity_type=entity_type,
                model_id=model_id,
                pose=pose,
                payload=payload,
            )
            return {"op": cmd.op, "entity_id": cmd.entity_id, "changed": True}
        if cmd.op == "move":
            current = self._entities.get(cmd.entity_id)
            if current is None:
                raise KeyError(f"Cannot move unknown entity {cmd.entity_id!r}.")
            current.pose.update(cmd.pose)
            return {"op": cmd.op, "entity_id": cmd.entity_id, "changed": True}
        if cmd.op in {"delete", "remove"}:
            existed = cmd.entity_id in self._entities
            self._entities.pop(cmd.entity_id, None)
            return {"op": cmd.op, "entity_id": cmd.entity_id, "changed": existed}
        if cmd.op == "rename":
            if not cmd.new_entity_id:
                raise ValueError("rename requires new_entity_id.")
            current = self._entities.pop(cmd.entity_id, None)
            if current is None:
                raise KeyError(f"Cannot rename unknown entity {cmd.entity_id!r}.")
            current.entity_id = cmd.new_entity_id
            self._entities[cmd.new_entity_id] = current
            return {"op": cmd.op, "entity_id": cmd.entity_id, "new_entity_id": cmd.new_entity_id, "changed": True}
        raise ValueError(f"Unsupported scene entity command op: {cmd.op!r}")
