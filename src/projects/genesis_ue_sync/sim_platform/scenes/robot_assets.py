from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.project import project_paths

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class RobotAssetSpec:
    model_id: str
    urdf_path: str
    mesh_root: str = "meshes/dae"
    visual_mesh_format: str = "fbx"
    default_joint_positions: list[float] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    controllers: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def robot_asset_root(repo_root: Path | None = None) -> Path:
    root = repo_root if repo_root is not None else project_paths(__file__).root
    return Path(root) / "assets" / "robots"


def robot_asset_dir(model_id: str, *, repo_root: Path | None = None) -> Path:
    return robot_asset_root(repo_root) / str(model_id).strip()


def robot_asset_manifest_path(model_id: str, *, repo_root: Path | None = None) -> Path:
    root = repo_root if repo_root is not None else project_paths(__file__).root
    primary = robot_asset_root(repo_root) / str(model_id).strip() / "robot.yaml"
    if primary.is_file():
        return primary
    fallback = Path(root) / "rm75_control" / "rm75_control" / "assets" / "robots" / str(model_id).strip() / "robot.yaml"
    if fallback.is_file():
        return fallback
    return primary


def _resolve_robot_urdf_path(payload: dict[str, Any], asset_dir: Path) -> str:
    raw = payload.get("urdf_path") or payload.get("urdf_genesis") or payload.get("urdf_kinematics")
    if not raw:
        raise KeyError("Robot asset manifest must define urdf_path, urdf_genesis, or urdf_kinematics.")
    urdf = Path(str(raw))
    if urdf.is_absolute():
        return str(urdf)
    candidate = (asset_dir / urdf).resolve()
    if candidate.is_file():
        return str(candidate)
    repo_root = asset_dir
    while repo_root.name and repo_root.parent != repo_root:
        if (repo_root / "assets" / "robots").is_dir() or (repo_root / "rm75_control").is_dir():
            break
        repo_root = repo_root.parent
    repo_candidate = (repo_root / urdf).resolve()
    if repo_candidate.is_file():
        return str(repo_candidate)
    return str(candidate)


def _read_mapping(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Robot asset manifest is empty: {path}")
    if yaml is not None:
        data = yaml.safe_load(raw)
    else:
        json_path = path.with_suffix(".json")
        if json_path.is_file():
            data = json.loads(json_path.read_text(encoding="utf-8"))
        else:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"PyYAML is unavailable in this Python runtime and {path} is YAML. "
                    f"Add {json_path.name} next to the manifest (Unreal Editor Python requires it)."
                ) from exc
    if not isinstance(data, dict):
        raise TypeError(f"Robot asset manifest must be a mapping: {path}")
    return data


def load_robot_asset_spec(model_id: str, *, repo_root: Path | None = None) -> RobotAssetSpec:
    mid = str(model_id).strip()
    if not mid:
        raise ValueError("model_id must be non-empty.")
    manifest = robot_asset_manifest_path(mid, repo_root=repo_root)
    payload = _read_mapping(manifest)
    urdf_path = str(
        payload.get("urdf_path")
        or payload.get("urdf_genesis")
        or payload.get("urdf_kinematics")
        or ""
    ).strip()
    if not urdf_path:
        raise ValueError(f"Robot asset manifest missing urdf_path/urdf_genesis: {manifest}")
    return RobotAssetSpec(
        model_id=str(payload.get("model_id") or mid),
        urdf_path=urdf_path,
        mesh_root=str(payload.get("mesh_root", "meshes/dae")),
        visual_mesh_format=str(payload.get("visual_mesh_format", "fbx")).strip().lower() or "fbx",
        default_joint_positions=[float(v) for v in payload.get("default_joint_positions", [])],
        defaults=dict(payload.get("defaults", {})),
        capabilities=dict(payload.get("capabilities", {})),
        controllers=dict(payload.get("controllers", {})),
        metadata=dict(payload.get("metadata", {})),
    )


def resolve_ue_visual_asset_root(robot_spec: Any, *, repo_root: Path | None = None) -> str:
    """UE Content root for robot link meshes; never silently default to Panda."""

    root = str(getattr(robot_spec, "ue_visual_asset_root", "") or "").strip()
    if root.startswith("/Game/"):
        return root
    model_id = str(getattr(robot_spec, "model_id", "") or "").strip()
    if model_id:
        try:
            merged = resolve_robot_model_payload(
                {"model_id": model_id, "name": getattr(robot_spec, "name", model_id)},
                repo_root=repo_root,
            )
            root = str(merged.get("ue_visual_asset_root", "") or "").strip()
            if root.startswith("/Game/"):
                return root
        except Exception:
            pass
    raise ValueError(
        f"Robot {getattr(robot_spec, 'name', '?')!r} (model_id={model_id!r}) has no valid "
        "ue_visual_asset_root. Set it in assets/robots/<model_id>/robot.yaml or scene robot block."
    )


def resolve_robot_model_payload(
    robot_payload: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Merge `assets/robots/<model_id>/robot.yaml` defaults under a scene robot instance."""

    payload = dict(robot_payload)
    model_id = str(payload.get("model_id") or "").strip()
    if not model_id:
        return payload

    asset = load_robot_asset_spec(model_id, repo_root=repo_root)
    asset_dir = robot_asset_dir(asset.model_id, repo_root=repo_root)
    if not asset_dir.is_dir():
        asset_dir = robot_asset_manifest_path(asset.model_id, repo_root=repo_root).parent
    merged: dict[str, Any] = dict(asset.defaults)
    merged.setdefault("name", payload.get("instance_id") or payload.get("name") or asset.model_id)
    merged.setdefault("model_id", asset.model_id)
    merged.setdefault("instance_id", payload.get("instance_id") or merged["name"])
    mesh_root_path = Path(asset.mesh_root)
    if mesh_root_path.is_absolute():
        merged.setdefault("mesh_root", str(mesh_root_path.resolve()))
    else:
        merged.setdefault("mesh_root", str((asset_dir / asset.mesh_root).resolve()))
    merged.setdefault("visual_mesh_format", asset.visual_mesh_format)
    merged.setdefault("joint_positions", list(asset.default_joint_positions))
    merged.setdefault("capabilities", dict(asset.capabilities))
    merged.setdefault("controllers", dict(asset.controllers))
    merged.setdefault("asset_metadata", dict(asset.metadata))
    if "urdf_path" not in merged:
        merged["urdf_path"] = _resolve_robot_urdf_path({"urdf_path": asset.urdf_path}, asset_dir)
    merged.update(payload)
    return merged
