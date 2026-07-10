"""Versioned scene-init message used to stand up a UE scene from Genesis-side spec.

Genesis is the source of truth: a publisher sends one ``SceneInitMessageV1`` payload
over ZMQ which describes the support surface, robot URDF + base pose, cameras, human
anchor / motion, and the AmongUs capture rig defaults. UE side persists the payload
to disk and feeds it to ``ue_common_scene_loader.apply_scene_to_current_level`` so
the editor stays empty until Genesis tells it what to build.

This module never imports ``genesis`` or ``unreal`` so it is safe on both sides.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_TOPIC = "amongus_scene_init_v1"


@dataclass
class SceneInitMessageV1:
    schema_version: int = SCHEMA_VERSION
    payload_hash_sha256: str = ""
    session_id: str = ""
    scene_spec: dict[str, Any] = field(default_factory=dict)
    augmentation_spec: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return json.loads(text)
    try:
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Cannot locate repository root (expected /src and /configs).")


def _make_paths_repo_relative(payload: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Strip absolute paths to repo-relative form so UE side resolves under its own repo root."""
    out = dict(payload)

    def _rel(value: str) -> str:
        try:
            p = Path(str(value))
        except Exception:
            return value
        if not p.is_absolute():
            return str(value)
        try:
            rel = p.resolve().relative_to(repo_root.resolve())
            return str(rel)
        except Exception:
            return str(value)

    motion = dict(out.get("motion") or {})
    for key in ("source_path", "sequence_npz_path", "mesh_manifest_path"):
        if motion.get(key):
            motion[key] = _rel(motion[key])
    if motion:
        out["motion"] = motion

    robot = dict(out.get("robot") or {})
    if robot.get("urdf_path"):
        robot["urdf_path"] = _rel(robot["urdf_path"])
    if robot:
        out["robot"] = robot
    robots = out.get("robots")
    if isinstance(robots, list):
        normalized = []
        for item in robots:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            robot_item = dict(item)
            if robot_item.get("urdf_path"):
                robot_item["urdf_path"] = _rel(robot_item["urdf_path"])
            normalized.append(robot_item)
        out["robots"] = normalized

    return out


def build_scene_init_message(
    scene_spec_path: str | Path,
    *,
    augmentation_spec_path: str | Path | None = None,
    session_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    robot_model: str = "",
) -> SceneInitMessageV1:
    """Read scene yaml (and optional augmentation), normalise paths, return a SceneInitMessageV1."""
    repo_root = _resolve_repo_root()
    from projects.genesis_ue_sync.sim_platform.scenes.scene_spec_resolve import effective_robot_model_id

    model_id = effective_robot_model_id(cli_model=robot_model)
    if model_id:
        from projects.genesis_ue_sync.sim_platform.scenes.scene_spec_resolve import resolve_scene_spec_payload

        spec_payload = resolve_scene_spec_payload(
            scene_spec_path,
            robot_model=model_id,
            for_ue_spawn=True,
            repo_root=repo_root,
        )
    else:
        spec_payload = _read_yaml_or_json(Path(scene_spec_path).expanduser().resolve())
    spec_payload = _make_paths_repo_relative(spec_payload, repo_root)
    if not isinstance(spec_payload, dict):
        raise TypeError(f"Scene spec must be a mapping: {scene_spec_path}")

    aug_payload: dict[str, Any] | None = None
    if augmentation_spec_path is not None:
        aug_payload = _read_yaml_or_json(Path(augmentation_spec_path).expanduser().resolve())
        if not isinstance(aug_payload, dict):
            raise TypeError(f"Augmentation spec must be a mapping: {augmentation_spec_path}")

    sid = str(session_id or os.environ.get("AMONGUS_SESSION_ID", "") or "").strip()
    extras = dict(extra_metadata or {})
    extras.setdefault("source", "genesis")
    extras.setdefault("scene_spec_basename", str(Path(scene_spec_path).name))
    model_id = effective_robot_model_id(cli_model=robot_model)
    if model_id:
        extras.setdefault("robot_model_id", model_id)

    canonical = json.dumps(
        {"scene_spec": spec_payload, "augmentation_spec": aug_payload, "session_id": sid},
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()

    return SceneInitMessageV1(
        schema_version=SCHEMA_VERSION,
        payload_hash_sha256=digest,
        session_id=sid,
        scene_spec=spec_payload,
        augmentation_spec=aug_payload,
        extras=extras,
    )


def scene_init_message_to_dict(message: SceneInitMessageV1) -> dict[str, Any]:
    return {
        "schema_version": int(message.schema_version),
        "payload_hash_sha256": str(message.payload_hash_sha256),
        "session_id": str(message.session_id),
        "scene_spec": dict(message.scene_spec or {}),
        "augmentation_spec": (None if message.augmentation_spec is None else dict(message.augmentation_spec)),
        "extras": dict(message.extras or {}),
    }


def scene_init_message_from_dict(payload: dict[str, Any]) -> SceneInitMessageV1:
    if not isinstance(payload, dict):
        raise TypeError(f"scene_init payload must be a mapping, got {type(payload).__name__}")
    aug = payload.get("augmentation_spec")
    return SceneInitMessageV1(
        schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        payload_hash_sha256=str(payload.get("payload_hash_sha256", "")),
        session_id=str(payload.get("session_id", "")),
        scene_spec=dict(payload.get("scene_spec") or {}),
        augmentation_spec=None if aug in (None, {}) else dict(aug),
        extras=dict(payload.get("extras") or {}),
    )


def write_scene_init_specs_to_session_dir(
    message: SceneInitMessageV1,
    session_dir: str | Path,
) -> tuple[Path, Path | None]:
    """Persist scene_spec (and optional augmentation_spec) JSON files inside session_dir/incoming/.

    Returns the absolute paths so UE side can pass them to existing apply_scene_to_level command.
    """
    session_root = Path(session_dir).expanduser().resolve()
    incoming = session_root / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    digest = message.payload_hash_sha256 or "unhashed"

    scene_path = incoming / f"scene_spec_{digest[:12]}.json"
    scene_path.write_text(
        json.dumps(message.scene_spec, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    aug_path: Path | None = None
    if message.augmentation_spec is not None:
        aug_path = incoming / f"augmentation_spec_{digest[:12]}.json"
        aug_path.write_text(
            json.dumps(message.augmentation_spec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return scene_path, aug_path
