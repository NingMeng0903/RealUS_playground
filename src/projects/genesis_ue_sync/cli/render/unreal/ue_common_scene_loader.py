from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import unreal

from bridge.adapters.ue import (
    ue_camera_payload_from_spec,
    ue_rotation_matrix_from_quat_xyzw,
    ue_world_rotation_from_genesis,
    ue_world_point_from_genesis_m,
    ue_world_quat_xyzw_from_genesis,
)
from bridge.adapters.urdf import root_transform_from_pose
from bridge.core.rotation import lookat_frame, quaternion_xyzw_to_matrix, ue_rotator_deg_from_lookat, ue_rotator_deg_from_matrix
from common.project import project_paths
from bridge.core.scene_capture_image_correction import derive_scene_capture_image_correction_from_spec
from projects.genesis_ue_sync.sim_platform.scenes import resolve_scene_spec_with_augmentation
from projects.genesis_ue_sync.sim_platform.scenes.robot_assets import resolve_ue_visual_asset_root
from projects.genesis_ue_sync.sim_platform.scenes.robot_probe_urdf import resolved_robot_urdf_for_robot_spec
from projects.genesis_ue_sync.sim_platform.human_refit.placement_json import (
    load_human_ue_calibration_dict,
    read_human_scene_placement_mesh_offset_m,
    resolve_human_scene_placement_json_path,
    resolve_human_ue_calibration_json_path,
)

REPO_ROOT = project_paths(__file__).root
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PROJECT_PATHS = project_paths(__file__)


def _amongus_truthy_env(name: str, *, default: bool = True) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if raw == "":
        return default
    return raw not in ("0", "false", "no", "off")


GENERATED_SCENE_FOLDER = "GeneratedScene"
GENERATED_SCENE_LABEL_PREFIX = "GEN_"
_LEVEL_EDITOR_CAMERA_SPEED_SCALED = False
# No global robot visual default — each model_id must declare ue_visual_asset_root in robot.yaml.
ROBOT_VISUAL_CACHE_ROOT = PROJECT_PATHS.ue_generated_cache_root / "robot_visual_obj"
ROBOT_VISUAL_FBX_CACHE_ROOT = PROJECT_PATHS.ue_generated_cache_root / "robot_visual_fbx"
_FBX_BATCH_CONVERTED_DIRS: set[tuple[str, str, str]] = set()
DEFAULT_STATIC_MATERIAL_PATH = "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"
# Meters in source mesh (DAE/OBJ) -> UE centimeters when importer treats file units as cm.
MESH_SOURCE_TO_UE_SCALE = 100.0
# AMONGUS_UE_ROBOT_VISUAL_BASIS_RPY_DEG: optional extra local rotation on each URDF visual mesh (roll pitch yaw, degrees).
# Default 0 0 0. Set only after audit_robot_visual_mesh.py / DAE-OBJ-UE checks show a constant mesh-vs-link-frame offset.
# This is separate from bridge ue_world_rotation_from_genesis (world pose), which already reconciles Genesis vs UE frames.
# Unreal Editor Python has no stock URDF articulation importer: see ue_urdf_visual_loader.py for joint-angle
# reuse after spawn (env AMONGUS_REGISTER_URDF_ARTICULATION_ID + EditorCommand update_urdf_robot_joints).

GENESIS_UE_AVATAR_SELECTION_ENV = "AMONGUS_GENESIS_UE_AVATAR_SELECTION_JSON"
DEFAULT_GENESIS_UE_AVATAR_SELECTION_PATH = REPO_ROOT / "outputs" / "genesis_viz" / "last_ue_avatar_selection.json"
USE_LAST_GENESIS_UE_AVATAR_SELECTION_ENV = "AMONGUS_USE_LAST_GENESIS_UE_AVATAR_SELECTION"

Matrix4 = list[list[float]]
Matrix3 = list[list[float]]


def _ensure_repo_on_path() -> None:
    src_path = REPO_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def load_scene_spec(scene_spec_path: str | Path, augmentation_spec_path: str | Path | None = None):
    _ensure_repo_on_path()
    scene_spec, augmentation_summary = resolve_scene_spec_with_augmentation(
        Path(scene_spec_path),
        augmentation_spec_path,
    )
    if augmentation_summary is not None:
        unreal.log(
            "UE_SCENE: applied augmentation "
            f"name={augmentation_summary.get('name', '')} "
            f"appearance_mode={((augmentation_summary.get('appearance') or {}).get('mode', 'inherit'))}"
        )
    return scene_spec, augmentation_summary


def _read_json_object(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _genesis_ue_avatar_selection_path() -> tuple[Path | None, str]:
    raw = (os.environ.get(GENESIS_UE_AVATAR_SELECTION_ENV) or "").strip()
    if raw:
        return Path(raw).expanduser(), "env"
    if _env_truthy(USE_LAST_GENESIS_UE_AVATAR_SELECTION_ENV) and DEFAULT_GENESIS_UE_AVATAR_SELECTION_PATH.is_file():
        return DEFAULT_GENESIS_UE_AVATAR_SELECTION_PATH, "last_genesis_viz"
    return None, "scene_spec"


def _apply_genesis_ue_avatar_selection(scene_spec, scene_spec_path: str | Path) -> dict:
    path, source = _genesis_ue_avatar_selection_path()
    if path is None:
        return {"source": "scene_spec", "selection_path": None, "applied": False}
    try:
        selection = _read_json_object(path)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        unreal.log_warning(f"UE_SCENE: skip genesis ue_avatar selection ({path}): {exc}")
        return {"source": "scene_spec", "selection_path": str(path), "applied": False, "reason": repr(exc)}
    if source != "env":
        selection_scene = str(selection.get("scene_spec", "") or "").strip()
        try:
            same_scene = bool(selection_scene) and Path(selection_scene).expanduser().resolve() == Path(scene_spec_path).expanduser().resolve()
        except OSError:
            same_scene = False
        if not same_scene:
            unreal.log_warning(f"UE_SCENE: skip stale genesis ue_avatar selection source={path}")
            return {"source": "scene_spec", "selection_path": str(path), "applied": False, "reason": "scene_spec_mismatch"}
    avatar = selection.get("ue_avatar") if isinstance(selection.get("ue_avatar"), dict) else selection
    if not isinstance(avatar, dict) or not avatar:
        unreal.log_warning(f"UE_SCENE: genesis ue_avatar selection has no usable mapping: {path}")
        return {"source": "scene_spec", "selection_path": str(path), "applied": False, "reason": "empty_avatar"}
    base = scene_spec.ue_avatar
    merged = dataclasses.replace(
        base,
        body_mode=str(avatar.get("body_mode", base.body_mode)),
        body_name=str(avatar.get("body_name", base.body_name)),
        texture_body=avatar.get("texture_body", base.texture_body),
        texture_clothing=avatar.get("texture_clothing", base.texture_clothing),
        texture_clothing_overlay=avatar.get("texture_clothing_overlay", base.texture_clothing_overlay),
        skeletal_mesh_path=str(avatar.get("skeletal_mesh_path", base.skeletal_mesh_path)),
        animation_asset_root=str(avatar.get("animation_asset_root", base.animation_asset_root)),
        imported_fbx_root=str(avatar.get("imported_fbx_root", base.imported_fbx_root)),
        fallback_animation_path=str(avatar.get("fallback_animation_path", base.fallback_animation_path)),
        hidden_material_path=str(avatar.get("hidden_material_path", base.hidden_material_path)),
        fbx_global_scale=float(avatar.get("fbx_global_scale", base.fbx_global_scale)),
    )
    scene_spec.bindings.character_visual = merged
    unreal.log(
        "UE_SCENE: applied genesis ue_avatar selection "
        f"skeletal_mesh_path={merged.skeletal_mesh_path} body_name={merged.body_name} source={path}"
    )
    return {
        "source": source,
        "selection_path": str(path),
        "applied": True,
        "selected_ue_avatar": dataclasses.asdict(merged),
    }


def m_to_cm(values: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    return tuple(float(item) * 100.0 for item in ue_world_point_from_genesis_m(values).tolist())


def _identity3() -> Matrix3:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _identity4() -> Matrix4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul3(a: Matrix3, b: Matrix3) -> Matrix3:
    return [[sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)] for row in range(3)]


def _matmul4(a: Matrix4, b: Matrix4) -> Matrix4:
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)] for row in range(4)]


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        float(a[1] * b[2] - a[2] * b[1]),
        float(a[2] * b[0] - a[0] * b[2]),
        float(a[0] * b[1] - a[1] * b[0]),
    )


def _normalize3(
    vec: tuple[float, float, float],
    *,
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    norm = math.sqrt(float(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]))
    if norm < 1e-8:
        return fallback
    inv = 1.0 / norm
    return (float(vec[0] * inv), float(vec[1] * inv), float(vec[2] * inv))


def _rot_x(angle: float) -> Matrix3:
    c, s = math.cos(angle), math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rot_y(angle: float) -> Matrix3:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(angle: float) -> Matrix3:
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def rpy_matrix(rpy: tuple[float, float, float]) -> Matrix3:
    roll, pitch, yaw = rpy
    return _matmul3(_matmul3(_rot_z(yaw), _rot_y(pitch)), _rot_x(roll))


def axis_angle_matrix(axis: tuple[float, float, float], angle: float) -> Matrix3:
    x, y, z = (float(axis[0]), float(axis[1]), float(axis[2]))
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-8:
        return _identity3()
    x /= norm
    y /= norm
    z /= norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def make_transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> Matrix4:
    transform = _identity4()
    rotation = rpy_matrix(rpy)
    for row in range(3):
        for col in range(3):
            transform[row][col] = rotation[row][col]
        transform[row][3] = float(xyz[row])
    return transform


def apply_transform(parent: Matrix4, local: Matrix4) -> Matrix4:
    return _matmul4(parent, local)


def _bfs_link_names(root_link: str, joints: list[dict]) -> list[str]:
    parent_to_children: dict[str, list[str]] = {}
    for joint in joints:
        parent_to_children.setdefault(joint["parent"], []).append(joint["child"])
    for key in parent_to_children:
        parent_to_children[key] = sorted(parent_to_children[key])
    ordered: list[str] = []
    queue = [root_link]
    seen: set[str] = set()
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        ordered.append(cur)
        queue.extend(parent_to_children.get(cur, []))
    return ordered


def _rotation3_from_transform(transform: Matrix4) -> Matrix3:
    return [row[:3] for row in transform[:3]]


def _set_rotation3(transform: Matrix4, rotation: Matrix3) -> None:
    for row in range(3):
        for col in range(3):
            transform[row][col] = rotation[row][col]


def _rotation3_from_quat_xyzw(x: float, y: float, z: float, w: float) -> Matrix3:
    """Unit quaternion (x, y, z, w) to column-vector rotation matrix."""
    return ue_rotation_matrix_from_quat_xyzw((x, y, z, w)).tolist()


def _parse_robot_visual_basis_rpy_deg_env() -> tuple[float, float, float]:
    raw = str(os.environ.get("AMONGUS_UE_ROBOT_VISUAL_BASIS_RPY_DEG", "0 0 0")).strip()
    parts = raw.replace(",", " ").split()
    if len(parts) != 3:
        unreal.log_warning(
            "UE_SCENE: invalid AMONGUS_UE_ROBOT_VISUAL_BASIS_RPY_DEG; expected 3 numbers, "
            f"got {raw!r}. Falling back to '0 0 0'."
        )
        return (0.0, 0.0, 0.0)
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        unreal.log_warning(
            "UE_SCENE: non-numeric AMONGUS_UE_ROBOT_VISUAL_BASIS_RPY_DEG="
            f"{raw!r}. Falling back to '0 0 0'."
        )
        return (0.0, 0.0, 0.0)


def _robot_visual_basis_correction_matrix() -> Matrix3:
    roll_deg, pitch_deg, yaw_deg = _parse_robot_visual_basis_rpy_deg_env()
    return rpy_matrix(
        (
            math.radians(float(roll_deg)),
            math.radians(float(pitch_deg)),
            math.radians(float(yaw_deg)),
        )
    )


def _make_root_transform_m(
    base_pos_m: tuple[float, float, float],
    base_quat_xyzw: tuple[float, float, float, float] | None,
) -> Matrix4:
    return root_transform_from_pose(base_pos_m, base_quat_xyzw)


def matrix_to_rotator_deg(matrix: Matrix3) -> tuple[float, float, float]:
    return ue_rotator_deg_from_matrix(ue_world_rotation_from_genesis(np.asarray(matrix, dtype=np.float64)))


def lookat_to_rotator_deg(
    pos: tuple[float, float, float],
    lookat: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float]:
    return ue_rotator_deg_from_lookat(pos, lookat, up)


def load_level(map_path: str) -> None:
    subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    subsystem.load_level(map_path)


_LAST_PIE_WORLD: object | None = None


def _query_pie_world():
    """Return the live PIE/game world if PIE is running, else ``None``.

    Caches the last seen PIE world reference so transient ``None`` returns from
    ``UnrealEditorSubsystem.get_game_world()`` (which can race during PIE frame
    boundaries) do not falsely demote canonical writes back to the editor world.
    The cache is dropped only when the cached world stops returning actors.
    """
    global _LAST_PIE_WORLD
    pie = None
    try:
        sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        if sub is not None and hasattr(sub, "get_game_world"):
            pie = sub.get_game_world()
    except Exception:
        pie = None
    if pie is not None:
        _LAST_PIE_WORLD = pie
        return pie
    if _LAST_PIE_WORLD is None:
        return None
    try:
        actors = unreal.GameplayStatics.get_all_actors_of_class(
            _LAST_PIE_WORLD, unreal.Actor
        )
    except Exception:
        actors = None
    if actors is None:
        _LAST_PIE_WORLD = None
        return None
    return _LAST_PIE_WORLD


def _active_world():
    """Editor world while not in PIE; PIE world while Play-in-Editor is running (race-tolerant)."""
    pie = _query_pie_world()
    if pie is not None:
        return pie
    return unreal.EditorLevelLibrary.get_editor_world()


def world_kind_for_canonical_tick() -> str:
    """Return ``pie`` when PIE is active (canonical writes must target game world), else ``editor``."""
    return "pie" if _query_pie_world() is not None else "editor"


def world_diagnostic_for_canonical_tick() -> dict[str, Any]:
    pie_world = _query_pie_world()
    diag: dict[str, Any] = {"world_kind": "pie" if pie_world is not None else "editor"}
    if pie_world is None:
        diag["note"] = "canonical writes are targeting editor world; press Play before expecting PIE viewport motion"
    try:
        diag["active_actor_count"] = len(_iter_world_actors())
    except Exception:
        pass
    return diag


def _iter_world_actors():
    world = _active_world()
    try:
        return list(unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor))
    except Exception:
        return []


def find_world_actor_by_label(label: str):
    for actor in _iter_world_actors():
        try:
            if str(actor.get_actor_label()) == str(label):
                return actor
        except Exception:
            continue
    return None


def find_editor_level_actor_by_label(label: str):
    """Resolve ``label`` on the editor persistent level (PIE duplicates are not in this list)."""
    for actor in _all_level_actors():
        try:
            if str(actor.get_actor_label()) == str(label):
                return actor
        except Exception:
            continue
    return None


def _mirror_poseable_relative_transform_to_editor_template(actor, comp) -> bool:
    """Copy PoseableMesh relative translation from ``comp`` to the editor-level actor with the same label.

    PIE instances are discarded when simulation stops; writing the corrected component transform onto the
    level template makes the next Play inherit the pelvis/bind fix without re-measuring.
    """
    if actor is None or comp is None:
        return False
    if not _amongus_truthy_env("AMONGUS_UE_PELVIS_MIRROR_TO_EDITOR_TEMPLATE", default=True):
        return False
    try:
        lab = str(actor.get_actor_label())
    except Exception:
        return False
    ed_actor = find_editor_level_actor_by_label(lab)
    if ed_actor is None:
        return False
    try:
        ed_comp = ed_actor.get_component_by_class(unreal.PoseableMeshComponent)
    except Exception:
        ed_comp = None
    if ed_comp is None:
        return False
    if ed_comp == comp:
        return False
    rel_cm = _scene_component_relative_translation_cm(comp)
    if not isinstance(rel_cm, list) or len(rel_cm) < 3:
        return False
    try:
        srl = getattr(ed_comp, "set_relative_location", None)
        if callable(srl):
            srl(unreal.Vector(float(rel_cm[0]), float(rel_cm[1]), float(rel_cm[2])), False)
        else:
            ed_comp.set_editor_property(
                "relative_location",
                unreal.Vector(float(rel_cm[0]), float(rel_cm[1]), float(rel_cm[2])),
            )
    except Exception as exc:
        unreal.log_warning(f"UE_SCENE: mirror poseable relative_location to editor failed: {exc!r}")
        return False
    rf = getattr(ed_comp, "refresh_bone_transforms", None)
    if callable(rf):
        try:
            rf()
        except Exception:
            pass
    unreal.log("UE_SCENE: mirrored PoseableMesh relative_location to editor GEN_visible_human template")
    return True


def resolve_active_world_actor(editor_actor):
    """Map a pre-PIE editor actor to its duplicate in the PIE world when Play is running."""
    if editor_actor is None:
        return None
    pie_world = _query_pie_world()
    if pie_world is None:
        return editor_actor
    try:
        label = str(editor_actor.get_actor_label())
    except Exception:
        return editor_actor
    try:
        for candidate in unreal.GameplayStatics.get_all_actors_of_class(pie_world, unreal.Actor):
            try:
                if str(candidate.get_actor_label()) == label:
                    return candidate
            except Exception:
                continue
    except Exception:
        pass
    return editor_actor


def _all_level_actors() -> list:
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()


def clear_sync_actors(*, preserve_labels: set[str] | None = None) -> None:
    preserve = preserve_labels or set()
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in _all_level_actors():
        folder = str(actor.get_folder_path())
        label = actor.get_actor_label()
        if str(label) in preserve:
            continue
        if folder.startswith(GENERATED_SCENE_FOLDER) or label.startswith(GENERATED_SCENE_LABEL_PREFIX):
            actor_subsystem.destroy_actor(actor)


def _spawn_static_mesh_actor(
    *,
    label: str,
    mesh_path: str,
    location_cm: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
    scale_xyz: tuple[float, float, float],
    folder: str,
    material_path: str | None = None,
    color_rgba: tuple[float, float, float, float] | None = None,
    rotation_matrix: Matrix3 | None = None,
) -> object:
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = actor_subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0.0, 0.0, 0.0))
    actor.set_actor_label(label)
    actor.set_folder_path(folder)
    mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
    component = actor.static_mesh_component
    component.set_static_mesh(mesh)
    if rotation_matrix is not None:
        loc_vec = unreal.Vector(float(location_cm[0]), float(location_cm[1]), float(location_cm[2]))
        scale_vec = unreal.Vector(float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2]))
        rot_deg = matrix_to_rotator_deg(rotation_matrix)
        actor.set_actor_location(loc_vec, False, False)
        actor.set_actor_rotation(unreal.Rotator(rot_deg[0], rot_deg[1], rot_deg[2]), False)
        actor.set_actor_scale3d(scale_vec)
    else:
        actor.set_actor_location(unreal.Vector(float(location_cm[0]), float(location_cm[1]), float(location_cm[2])), False, False)
        actor.set_actor_rotation(unreal.Rotator(rotation_deg[0], rotation_deg[1], rotation_deg[2]), False)
        actor.set_actor_scale3d(unreal.Vector(float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])))
    if material_path is not None:
        material = unreal.EditorAssetLibrary.load_asset(material_path)
        if material is not None:
            material_slots = component.get_num_materials() if hasattr(component, "get_num_materials") else 1
            for material_index in range(max(int(material_slots), 1)):
                component.set_material(material_index, material)
    if color_rgba is not None and hasattr(component, "set_vector_parameter_value_on_materials"):
        color = unreal.Vector(float(color_rgba[0]), float(color_rgba[1]), float(color_rgba[2]))
        for parameter_name in ("Color", "BaseColor"):
            component.set_vector_parameter_value_on_materials(parameter_name, color)
    mobility = getattr(unreal, "ComponentMobility", None)
    if mobility is not None:
        mov = getattr(mobility, "MOVABLE", None)
        if mov is not None:
            for target in (component, getattr(actor, "root_component", None)):
                if target is not None and hasattr(target, "set_mobility"):
                    try:
                        target.set_mobility(mov)
                    except Exception:
                        continue
    _ensure_actor_movable(actor)
    return actor


def _actor_transform_payload(actor, *, include_bounds: bool = False) -> dict[str, object]:
    loc = actor.get_actor_location()
    rot = actor.get_actor_rotation()
    scale = actor.get_actor_scale3d()
    payload = {
        "label": str(actor.get_actor_label()),
        "class_name": str(actor.get_class().get_name()),
        "location_cm": [float(loc.x), float(loc.y), float(loc.z)],
        "rotation_deg": [float(rot.roll), float(rot.pitch), float(rot.yaw)],
        "scale": [float(scale.x), float(scale.y), float(scale.z)],
    }
    if include_bounds:
        try:
            origin, extent = actor.get_actor_bounds(False)
            payload["world_bounds"] = {
                "origin_cm": [float(origin.x), float(origin.y), float(origin.z)],
                "extent_cm": [float(extent.x), float(extent.y), float(extent.z)],
            }
        except Exception as exc:
            payload["world_bounds_error"] = repr(exc)
        try:
            component = actor.static_mesh_component
            mesh = component.static_mesh if component is not None else None
            if mesh is not None:
                bounds = mesh.get_bounds()
                origin = bounds.origin
                extent = bounds.box_extent
                payload["static_mesh_local_bounds"] = {
                    "origin_cm": [float(origin.x), float(origin.y), float(origin.z)],
                    "extent_cm": [float(extent.x), float(extent.y), float(extent.z)],
                }
        except Exception as exc:
            payload["static_mesh_bounds_error"] = repr(exc)
    return payload


def _fix_imported_static_mesh_lod_for_mrq(asset_path: str | None) -> None:
    if not asset_path:
        return
    mesh = unreal.EditorAssetLibrary.load_asset(asset_path)
    if mesh is None:
        unreal.log_warning(f"UE_SCENE: lod_fix skip load_failed asset={asset_path}")
        return
    try:
        subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
        sizes = subsystem.get_lod_screen_sizes(mesh)
        before = [float(x) for x in sizes]
        new_sizes = [1.0 if s <= 1e-6 else s for s in before]
        changed = any(abs(a - b) > 1e-9 for a, b in zip(before, new_sizes))
        if changed:
            unreal.log(f"UE_SCENE: patched lod_screen_sizes asset={asset_path} before={before} after={new_sizes}")
            subsystem.set_lod_screen_sizes(mesh, new_sizes)
            unreal.EditorAssetLibrary.save_asset(asset_path)
    except Exception as exc:
        unreal.log_warning(f"UE_SCENE: lod_fix exception asset={asset_path} err={exc!r}")


def _import_static_mesh_asset(mesh_path: Path, *, asset_root: str, asset_name: str) -> str | None:
    asset_path = f"{asset_root}/{asset_name}.{asset_name}"
    force_reimport = (
        _env_truthy("AMONGUS_REIMPORT_ROBOT_MESH_ASSETS")
        or _env_truthy("AMONGUS_REIMPORT_PANDA_MESH_ASSETS")
        or _env_truthy("AMONGUS_REBUILD_ROBOT_FBX_CACHE")
        or _env_truthy("AMONGUS_REBUILD_PANDA_FBX_CACHE")
        or _env_truthy("AMONGUS_REBUILD_ROBOT_OBJ_CACHE")
        or _env_truthy("AMONGUS_REBUILD_PANDA_OBJ_CACHE")
    )
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path) and not force_reimport:
        return asset_path
    task = unreal.AssetImportTask()
    task.filename = str(mesh_path)
    task.destination_path = asset_root
    task.destination_name = asset_name
    task.automated = True
    task.save = True
    task.replace_existing = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    imported = [str(item) for item in task.get_editor_property("imported_object_paths")]
    result = imported[0] if imported else (asset_path if unreal.EditorAssetLibrary.does_asset_exist(asset_path) else None)
    _fix_imported_static_mesh_lod_for_mrq(result)
    return result


def _python_bin() -> str:
    candidate = shutil.which("python3") or shutil.which("python")
    if candidate is None:
        raise RuntimeError("Cannot find python3/python in PATH for COLLADA conversion.")
    return candidate


def _resolve_urdf_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    raw = str(mesh_filename).strip()
    if raw.startswith("package://"):
        rest = raw[len("package://") :]
        idx = rest.find("/")
        package_name = rest[:idx] if idx >= 0 else ""
        raw = rest[idx + 1 :] if idx >= 0 else rest
        for root in (urdf_path.parent, urdf_path.parent.parent, urdf_path.parent.parent.parent):
            if not root:
                continue
            if package_name and root.name != package_name and not (root / "package.xml").is_file():
                continue
            candidate = (root / raw).resolve()
            if candidate.exists():
                return candidate
    return (urdf_path.parent / raw).resolve()


def _robot_mesh_import_preference() -> str:
    return os.environ.get("AMONGUS_ROBOT_MESH_SOURCE", os.environ.get("AMONGUS_PANDA_MESH_SOURCE", "fbx")).strip().lower()


def _panda_mesh_import_preference() -> str:
    return _robot_mesh_import_preference()


def _resolve_ue_robot_mesh_source(scene_visual_mesh_format: str | None) -> str:
    """UE articulation visual import path: ``fbx``, ``dae`` (native COLLADA then OBJ), or ``obj`` only.

    ``robot.yaml`` / scene ``visual_mesh_format`` is the source of truth; env is used when unset.
    """
    fmt = str(scene_visual_mesh_format or "").strip().lower()
    if fmt in {"dae", "collada"}:
        return "dae"
    if fmt == "obj":
        return "obj"
    if fmt == "fbx":
        return "fbx"
    pref = (_robot_mesh_import_preference() or "fbx").strip().lower()
    return "obj" if pref == "obj" else "fbx"


def _fbx_axis_forward() -> str:
    return str(os.environ.get("AMONGUS_FBX_AXIS_FORWARD", "X")).strip() or "X"


def _fbx_axis_up() -> str:
    return str(os.environ.get("AMONGUS_FBX_AXIS_UP", "Z")).strip() or "Z"


def _fbx_global_scale() -> float:
    raw = str(os.environ.get("AMONGUS_FBX_GLOBAL_SCALE", "1.0")).strip() or "1.0"
    try:
        return float(raw)
    except ValueError:
        unreal.log_warning(f"UE_SCENE: invalid AMONGUS_FBX_GLOBAL_SCALE={raw!r}; using 1.0")
        return 1.0


def _mirror_y_for_unreal_fbx_import() -> bool:
    raw = str(os.environ.get("AMONGUS_MIRROR_Y_FOR_UNREAL_FBX_IMPORT", "0")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _mirror_y_for_unreal_obj_import() -> bool:
    raw = str(os.environ.get("AMONGUS_MIRROR_Y_FOR_UNREAL_OBJ_IMPORT", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _mesh_source_to_ue_scale(mesh_source: str) -> float:
    # UE's FBX importer converts the DAE meter units to centimeters, so actor scale stays at 1.
    if str(mesh_source).strip().lower() == "fbx":
        return 1.0
    return float(MESH_SOURCE_TO_UE_SCALE)


def _safe_asset_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value)).strip("_") or "asset"


def _robot_visual_cache_key(
    *,
    mesh_source: str,
    visual_basis_rpy_deg: tuple[float, float, float],
    visual_mesh_scale: float,
) -> str:
    payload = {
        "mesh_source": str(mesh_source),
        "mirror_y_for_unreal_obj_import": str(mesh_source).lower() == "obj" and _mirror_y_for_unreal_obj_import(),
        "obj_converter_revision": 2,
        "fbx_converter_revision": 6,
        "fbx_axis_forward": _fbx_axis_forward() if str(mesh_source).lower() == "fbx" else None,
        "fbx_axis_up": _fbx_axis_up() if str(mesh_source).lower() == "fbx" else None,
        "fbx_global_scale": round(float(_fbx_global_scale()), 6) if str(mesh_source).lower() == "fbx" else None,
        "mirror_y_for_unreal_fbx_import": str(mesh_source).lower() == "fbx" and _mirror_y_for_unreal_fbx_import(),
        "visual_basis_rpy_deg": [round(float(v), 6) for v in visual_basis_rpy_deg],
        "visual_mesh_scale": round(float(visual_mesh_scale), 6),
        "mesh_source_to_ue_scale": float(_mesh_source_to_ue_scale(mesh_source)),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{_safe_asset_token(str(mesh_source).lower())}_{digest}"


def _import_robot_link_visual_mesh(
    urdf_path: Path,
    mesh_filename: str,
    *,
    visual_asset_root: str,
    asset_name: str,
    visual_mesh_scale: float,
    visual_basis_rpy_deg: tuple[float, float, float],
    mesh_source: str,
    model_id: str = "",
) -> str | None:
    pref = str(mesh_source or "fbx").strip().lower()
    cache_key = _robot_visual_cache_key(
        mesh_source=pref,
        visual_basis_rpy_deg=visual_basis_rpy_deg,
        visual_mesh_scale=visual_mesh_scale,
    )
    keyed_asset_name = f"{_safe_asset_token(asset_name)}_{cache_key}"
    dae_path = _resolve_urdf_mesh_path(urdf_path, mesh_filename)
    if pref == "fbx":
        fbx_path = _ensure_visual_fbx(urdf_path, mesh_filename, cache_key=cache_key, model_id=model_id)
        if fbx_path is not None:
            return _import_static_mesh_asset(fbx_path, asset_root=visual_asset_root, asset_name=keyed_asset_name)
        unreal.log_warning(
            f"UE robot mesh: FBX path unavailable for {mesh_filename}; "
            "falling back to COLLADA/OBJ import."
        )
    if pref == "dae" and dae_path.is_file() and dae_path.suffix.lower() == ".dae":
        ap = _import_static_mesh_asset(dae_path, asset_root=visual_asset_root, asset_name=keyed_asset_name)
        if ap is not None and unreal.EditorAssetLibrary.load_asset(ap) is not None:
            return ap
        unreal.log_warning(f"UE robot mesh: direct COLLADA import failed for {dae_path}, using OBJ fallback.")
    elif pref == "dae":
        unreal.log_warning(f"UE robot mesh: missing COLLADA file {dae_path}, trying OBJ pipeline.")
    obj_path = _ensure_visual_obj(urdf_path, mesh_filename, cache_key=cache_key)
    if obj_path is None:
        return None
    return _import_static_mesh_asset(obj_path, asset_root=visual_asset_root, asset_name=keyed_asset_name)


def _ensure_visual_fbx(
    urdf_path: Path,
    mesh_filename: str,
    *,
    cache_key: str,
    model_id: str = "",
) -> Path | None:
    source_path = _resolve_urdf_mesh_path(urdf_path, mesh_filename)
    if not source_path.is_file():
        return None
    mid = _safe_asset_token(model_id or "unknown_model")
    output_dir = ROBOT_VISUAL_FBX_CACHE_ROOT / mid / urdf_path.stem / _safe_asset_token(cache_key)
    output_path = output_dir / f"{source_path.stem}_{_safe_asset_token(cache_key)}.fbx"
    force_rebuild = _env_truthy("AMONGUS_REBUILD_ROBOT_FBX_CACHE") or _env_truthy("AMONGUS_REBUILD_PANDA_FBX_CACHE")
    if output_path.is_file() and not force_rebuild:
        return output_path
    converter_script = (
        REPO_ROOT
        / "src"
        / "projects"
        / "genesis_ue_sync"
        / "cli"
        / "render"
        / "media"
        / "convert_collada_to_fbx.py"
    )
    if source_path.suffix.lower() == ".dae":
        force_token = str(os.environ.get("AMONGUS_UE_COMMAND_REQUEST_ID", "")) if force_rebuild else ""
        batch_key = (str(source_path.parent.resolve()), str(output_dir.resolve()), force_token)
        if batch_key not in _FBX_BATCH_CONVERTED_DIRS:
            try:
                subprocess.run(
                    [
                        _python_bin(),
                        str(converter_script),
                        str(source_path.parent),
                        str(output_dir),
                        f"--global-scale={float(_fbx_global_scale())}",
                        f"--axis-forward={_fbx_axis_forward()}",
                        f"--axis-up={_fbx_axis_up()}",
                        *(["--mirror-y-for-unreal"] if _mirror_y_for_unreal_fbx_import() else []),
                    ],
                    check=True,
                    cwd=str(REPO_ROOT),
                )
            except (subprocess.CalledProcessError, OSError) as exc:
                unreal.log_warning(
                    "UE robot mesh: DAE->FBX batch conversion failed "
                    f"({exc!r}). Set AMONGUS_BLENDER_BIN and retry, or use robot.visual_mesh_format: dae in scene."
                )
                return None
            for dae_path in sorted(source_path.parent.glob("*.dae")):
                raw_fbx = output_dir / f"{dae_path.stem}.fbx"
                keyed_fbx = output_dir / f"{dae_path.stem}_{_safe_asset_token(cache_key)}.fbx"
                if raw_fbx.is_file() and raw_fbx != keyed_fbx:
                    shutil.copyfile(raw_fbx, keyed_fbx)
            _FBX_BATCH_CONVERTED_DIRS.add(batch_key)
        return output_path if output_path.is_file() else None
    try:
        subprocess.run(
            [
                _python_bin(),
                str(converter_script),
                str(source_path),
                str(output_path),
                f"--global-scale={float(_fbx_global_scale())}",
                f"--axis-forward={_fbx_axis_forward()}",
                f"--axis-up={_fbx_axis_up()}",
                *(["--mirror-y-for-unreal"] if _mirror_y_for_unreal_fbx_import() else []),
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        unreal.log_warning(
            f"UE robot mesh: DAE->FBX conversion failed for {source_path} ({exc!r}). "
            "Set AMONGUS_BLENDER_BIN."
        )
        return None
    return output_path if output_path.is_file() else None


def _ensure_visual_obj(urdf_path: Path, mesh_filename: str, *, cache_key: str) -> Path | None:
    source_path = _resolve_urdf_mesh_path(urdf_path, mesh_filename)
    if not source_path.is_file():
        return None
    output_dir = ROBOT_VISUAL_CACHE_ROOT / urdf_path.stem / _safe_asset_token(cache_key)
    converter_output_path = output_dir / f"{source_path.stem}.obj"
    output_path = output_dir / f"{source_path.stem}_{_safe_asset_token(cache_key)}.obj"
    force_rebuild = bool(
        os.environ.get("AMONGUS_REBUILD_ROBOT_OBJ_CACHE", os.environ.get("AMONGUS_REBUILD_PANDA_OBJ_CACHE", "")).strip()
    )
    if output_path.is_file() and not force_rebuild:
        return output_path
    converter_script = (
        REPO_ROOT
        / "src"
        / "projects"
        / "genesis_ue_sync"
        / "cli"
        / "render"
        / "media"
        / "convert_collada_to_obj.py"
    )
    cmd = [
        _python_bin(),
        str(converter_script),
        str(source_path),
        str(output_dir),
    ]
    if _mirror_y_for_unreal_obj_import():
        cmd.append("--mirror-y-for-unreal-obj-import")
    subprocess.run(
        cmd,
        check=True,
        cwd=str(REPO_ROOT),
    )
    if converter_output_path.is_file() and converter_output_path != output_path:
        shutil.copyfile(converter_output_path, output_path)
    return output_path if output_path.is_file() else None


def spawn_visual_mesh(
    *,
    label: str,
    mesh_asset_path: str,
    pos_m: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
    folder: str = GENERATED_SCENE_FOLDER,
    color_rgba: tuple[float, float, float, float] | None = None,
    mesh_scale: float = 1.0,
    rot_mat: Matrix3 | None = None,
) -> object:
    s = float(mesh_scale)
    return _spawn_static_mesh_actor(
        label=label,
        mesh_path=mesh_asset_path,
        location_cm=m_to_cm(pos_m),
        rotation_deg=rotation_deg,
        scale_xyz=(s, s, s),
        folder=folder,
        material_path=None,
        color_rgba=None,
        rotation_matrix=rot_mat,
    )


def spawn_box(
    *,
    label: str,
    pos_m: tuple[float, float, float],
    size_m: tuple[float, float, float],
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    folder: str = GENERATED_SCENE_FOLDER,
    color_rgba: tuple[float, float, float, float] | None = None,
) -> object:
    return _spawn_static_mesh_actor(
        label=label,
        mesh_path="/Engine/BasicShapes/Cube.Cube",
        location_cm=m_to_cm(pos_m),
        rotation_deg=rotation_deg,
        scale_xyz=(float(size_m[0]), float(size_m[1]), float(size_m[2])),
        folder=folder,
        material_path=DEFAULT_STATIC_MATERIAL_PATH if color_rgba is not None else None,
        color_rgba=color_rgba,
    )


def spawn_marker(
    *,
    label: str,
    pos_m: tuple[float, float, float],
    scale_m: float = 0.05,
    folder: str = GENERATED_SCENE_FOLDER,
    color_rgba: tuple[float, float, float, float] | None = None,
) -> object:
    return _spawn_static_mesh_actor(
        label=label,
        mesh_path="/Engine/BasicShapes/Sphere.Sphere",
        location_cm=m_to_cm(pos_m),
        rotation_deg=(0.0, 0.0, 0.0),
        scale_xyz=(scale_m, scale_m, scale_m),
        folder=folder,
        material_path=DEFAULT_STATIC_MATERIAL_PATH if color_rgba is not None else None,
        color_rgba=color_rgba,
    )


def spawn_cylinder(
    *,
    label: str,
    pos_m: tuple[float, float, float],
    radius_m: float,
    length_m: float,
    rotation_deg: tuple[float, float, float],
    folder: str = GENERATED_SCENE_FOLDER,
    color_rgba: tuple[float, float, float, float] | None = None,
) -> object:
    return _spawn_static_mesh_actor(
        label=label,
        mesh_path="/Engine/BasicShapes/Cylinder.Cylinder",
        location_cm=m_to_cm(pos_m),
        rotation_deg=rotation_deg,
        scale_xyz=(radius_m * 2.0, radius_m * 2.0, max(length_m * 0.5, 0.001)),
        folder=folder,
        material_path=DEFAULT_STATIC_MATERIAL_PATH if color_rgba is not None else None,
        color_rgba=color_rgba,
    )


def _parse_origin(node) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if node is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = tuple(float(item) for item in node.attrib.get("xyz", "0 0 0").split())
    rpy = tuple(float(item) for item in node.attrib.get("rpy", "0 0 0").split())
    return xyz, rpy


def _forward_kinematics_link_world(
    *,
    urdf_path: Path,
    base_pos_m: tuple[float, float, float],
    joint_positions: list[float],
    base_quat_xyzw: tuple[float, float, float, float] | None = None,
) -> tuple[str, dict[str, dict], list[dict], dict[str, Matrix4], list[str]]:
    root_link, links, joints = _parse_urdf(urdf_path)
    link_world: dict[str, Matrix4] = {root_link: _make_root_transform_m(base_pos_m, base_quat_xyzw)}
    actuated_values = iter(joint_positions)
    pending = list(joints)
    while pending:
        progressed = False
        next_pending: list[dict] = []
        for joint in pending:
            parent = joint["parent"]
            child = joint["child"]
            if parent not in link_world:
                next_pending.append(joint)
                continue
            local = make_transform(joint["xyz"], joint["rpy"])
            if joint["type"] in ("revolute", "continuous"):
                local_rot = _matmul3(
                    _rotation3_from_transform(local),
                    axis_angle_matrix(joint["axis"], float(next(actuated_values, 0.0))),
                )
                _set_rotation3(local, local_rot)
            elif joint["type"] == "prismatic":
                q = float(next(actuated_values, 0.0))
                axis = joint["axis"]
                norm = math.sqrt(float(axis[0]) ** 2 + float(axis[1]) ** 2 + float(axis[2]) ** 2)
                if norm > 1e-8:
                    ax, ay, az = float(axis[0]) / norm, float(axis[1]) / norm, float(axis[2]) / norm
                else:
                    ax, ay, az = 0.0, 0.0, 1.0
                local = apply_transform(local, make_transform((ax * q, ay * q, az * q), (0.0, 0.0, 0.0)))
            link_world[child] = apply_transform(link_world[parent], local)
            progressed = True
        if not progressed:
            break
        pending = next_pending
    if pending:
        raise RuntimeError(
            "URDF joint chain could not be resolved (cycle or missing parent). "
            f"Pending joints: {[j.get('name') for j in pending]}"
        )
    ordered_links = _bfs_link_names(root_link, joints)
    for orphan in sorted(name for name in link_world if name not in set(ordered_links)):
        ordered_links.append(orphan)
    return root_link, links, joints, link_world, ordered_links


def compute_robot_visual_mesh_entries(
    *,
    urdf_path: Path,
    base_pos_m: tuple[float, float, float],
    joint_positions: list[float],
    base_quat_xyzw: tuple[float, float, float, float] | None = None,
    use_collision_geometry: bool = False,
    use_visual_mesh: bool = True,
    allow_collision_fallback: bool = False,
) -> list[dict[str, Any]]:
    """Genesis-frame mesh poses for URDF links that spawn as StaticMesh (for joint-angle-driven updates)."""
    _root_link, links, joints, link_world, ordered_links = _forward_kinematics_link_world(
        urdf_path=urdf_path,
        base_pos_m=base_pos_m,
        joint_positions=joint_positions,
        base_quat_xyzw=base_quat_xyzw,
    )
    visual_basis_correction = _robot_visual_basis_correction_matrix()
    entries: list[dict[str, Any]] = []
    for link_name in ordered_links:
        world = link_world.get(link_name)
        if world is None:
            continue
        link_info = links.get(link_name, {})
        visual_mesh = link_info.get("visual_mesh")
        if not use_collision_geometry and use_visual_mesh and visual_mesh:
            visual_local = make_transform(
                link_info.get("visual_origin_xyz", (0.0, 0.0, 0.0)),
                link_info.get("visual_origin_rpy", (0.0, 0.0, 0.0)),
            )
            _set_rotation3(
                visual_local,
                _matmul3(_rotation3_from_transform(visual_local), visual_basis_correction),
            )
            visual_world = apply_transform(world, visual_local)
            entries.append(
                {
                    "link_name": str(link_name),
                    "visual_mesh": str(visual_mesh),
                    "pos_m": (
                        float(visual_world[0][3]),
                        float(visual_world[1][3]),
                        float(visual_world[2][3]),
                    ),
                    "rot3": _rotation3_from_transform(visual_world),
                }
            )
        elif not use_collision_geometry and use_visual_mesh:
            for box_idx, box in enumerate(link_info.get("visual_boxes") or []):
                visual_local = make_transform(
                    tuple(float(v) for v in box["xyz"]),
                    tuple(float(v) for v in box["rpy"]),
                )
                visual_world = apply_transform(world, visual_local)
                entries.append(
                    {
                        "link_name": f"{link_name}__box{box_idx}",
                        "visual_mesh": None,
                        "pos_m": (
                            float(visual_world[0][3]),
                            float(visual_world[1][3]),
                            float(visual_world[2][3]),
                        ),
                        "rot3": _rotation3_from_transform(visual_world),
                    }
                )
        elif not use_collision_geometry and use_visual_mesh and visual_mesh is None and not allow_collision_fallback:
            continue
    return entries


def _ensure_actor_movable(actor) -> None:
    """Force root + every primitive component on this actor to MOVABLE so set_actor_* succeeds."""
    mobility_enum = getattr(unreal, "ComponentMobility", None)
    movable = getattr(mobility_enum, "MOVABLE", None) if mobility_enum is not None else None
    if movable is None:
        return
    root = None
    try:
        root = actor.root_component
    except Exception:
        try:
            root = actor.get_editor_property("root_component")
        except Exception:
            root = None
    if root is not None and hasattr(root, "set_mobility"):
        try:
            root.set_mobility(movable)
        except Exception:
            pass
    try:
        primitives = actor.get_components_by_class(unreal.SceneComponent)
    except Exception:
        primitives = []
    for comp in primitives:
        try:
            if hasattr(comp, "set_mobility"):
                comp.set_mobility(movable)
        except Exception:
            continue


def apply_static_mesh_actor_pose_genesis(actor, *, pos_m: tuple[float, float, float], rot3: Matrix3) -> None:
    """Apply Genesis-frame pose to a UE StaticMeshActor with teleport semantics so editor world updates immediately.

    Preserves the actor's existing scale. URDF visual boxes are spawned with non-1 scales
    (UE Cube is 1 m); resetting scale to (1,1,1) was inflating frame/rail boxes into huge cubes
    when the canonical bridge updated joint poses.
    """
    loc_cm = m_to_cm(pos_m)
    loc_vec = unreal.Vector(float(loc_cm[0]), float(loc_cm[1]), float(loc_cm[2]))
    rot_deg = matrix_to_rotator_deg(rot3)
    rotator = unreal.Rotator(float(rot_deg[0]), float(rot_deg[1]), float(rot_deg[2]))
    _ensure_actor_movable(actor)
    scale_vec = unreal.Vector(1.0, 1.0, 1.0)
    try:
        cur_scale = actor.get_actor_scale3d()
        scale_vec = unreal.Vector(float(cur_scale.x), float(cur_scale.y), float(cur_scale.z))
    except Exception:
        try:
            cur_scale = actor.get_actor_scale()
            scale_vec = unreal.Vector(float(cur_scale.x), float(cur_scale.y), float(cur_scale.z))
        except Exception:
            pass
    set_transform = getattr(actor, "set_actor_transform", None)
    if set_transform is not None:
        try:
            transform = unreal.Transform(loc_vec, rotator, scale_vec)
            set_transform(transform, False, True)
            return
        except Exception:
            pass
    # Fall back to split set_location / set_rotation with teleport=True (scale untouched).
    try:
        actor.set_actor_location(loc_vec, False, True)
    except Exception:
        actor.set_actor_location(loc_vec, False, False)
    try:
        actor.set_actor_rotation(rotator, True)
    except Exception:
        actor.set_actor_rotation(rotator, False)


def _parse_urdf(urdf_path: Path) -> tuple[str, dict[str, dict], list[dict]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links: dict[str, dict] = {}
    child_links: set[str] = set()
    joints: list[dict] = []

    for link in root.findall("link"):
        visual_boxes: list[dict[str, object]] = []
        visual_mesh = None
        visual_xyz, visual_rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        for visual in link.findall("visual"):
            v_xyz, v_rpy = _parse_origin(visual.find("origin") if visual is not None else None)
            visual_geometry = visual.find("geometry") if visual is not None else None
            if visual_geometry is None:
                continue
            mesh_el = visual_geometry.find("mesh")
            box_el = visual_geometry.find("box")
            if mesh_el is not None and visual_mesh is None:
                visual_mesh = str(mesh_el.attrib.get("filename"))
                visual_xyz, visual_rpy = v_xyz, v_rpy
            elif box_el is not None:
                color_rgba = None
                material = visual.find("material")
                if material is not None:
                    color_el = material.find("color")
                    if color_el is not None:
                        rgba_raw = str(color_el.attrib.get("rgba", "")).strip()
                        if rgba_raw:
                            color_rgba = tuple(float(item) for item in rgba_raw.split())
                visual_boxes.append(
                    {
                        "xyz": v_xyz,
                        "rpy": v_rpy,
                        "size": tuple(float(item) for item in box_el.attrib["size"].split()),
                        "color_rgba": color_rgba,
                    }
                )
        collision = link.find("collision")
        collision_xyz, collision_rpy = _parse_origin(collision.find("origin") if collision is not None else None)
        geometry = collision.find("geometry") if collision is not None else None
        geometry_payload: dict[str, object] | None = None
        if geometry is not None:
            if geometry.find("cylinder") is not None:
                cylinder = geometry.find("cylinder")
                geometry_payload = {
                    "type": "cylinder",
                    "radius": float(cylinder.attrib["radius"]),
                    "length": float(cylinder.attrib["length"]),
                }
            elif geometry.find("box") is not None:
                box = geometry.find("box")
                geometry_payload = {
                    "type": "box",
                    "size": tuple(float(item) for item in box.attrib["size"].split()),
                }
        links[link.attrib["name"]] = {
            "visual_origin_xyz": visual_xyz,
            "visual_origin_rpy": visual_rpy,
            "visual_mesh": visual_mesh,
            "visual_boxes": visual_boxes,
            "collision_origin_xyz": collision_xyz,
            "collision_origin_rpy": collision_rpy,
            "geometry": geometry_payload,
        }

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        xyz, rpy = _parse_origin(joint.find("origin"))
        axis = tuple(float(item) for item in joint.find("axis").attrib.get("xyz", "0 0 1").split()) if joint.find("axis") is not None else (0.0, 0.0, 1.0)
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type", "fixed"),
                "parent": parent.attrib["link"],
                "child": child.attrib["link"],
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
            }
        )
        child_links.add(child.attrib["link"])

    roots = [name for name in links if name not in child_links]
    if len(roots) != 1:
        raise RuntimeError(f"URDF must have exactly one root link; found {roots!r}")
    root_link = roots[0]
    return root_link, links, joints


def spawn_robot_from_urdf(
    *,
    urdf_path: Path,
    base_pos_m: tuple[float, float, float],
    joint_positions: list[float],
    base_quat_xyzw: tuple[float, float, float, float] | None = None,
    use_collision_geometry: bool = False,
    use_visual_mesh: bool = True,
    allow_collision_fallback: bool = False,
    visual_asset_root: str,
    model_id: str = "",
    visual_mesh_scale: float = 1.0,
    visual_mesh_format: str | None = None,
    folder: str = f"{GENERATED_SCENE_FOLDER}/robot",
    color_rgba: tuple[float, float, float, float] | None = None,
    articulation_actor_sink: dict[str, object] | None = None,
) -> dict[str, object]:
    # FK matches projects.genesis_ue_sync.urdf.kinematics + Genesis get_link_pose (validate_robot_urdf_consistency.py).
    _root_link, links, joints, link_world, ordered_links = _forward_kinematics_link_world(
        urdf_path=urdf_path,
        base_pos_m=base_pos_m,
        joint_positions=joint_positions,
        base_quat_xyzw=base_quat_xyzw,
    )
    visual_basis_rpy_deg = _parse_robot_visual_basis_rpy_deg_env()
    visual_basis_correction = _robot_visual_basis_correction_matrix()
    mesh_source = _resolve_ue_robot_mesh_source(visual_mesh_format)
    cache_key = _robot_visual_cache_key(
        mesh_source=mesh_source,
        visual_basis_rpy_deg=visual_basis_rpy_deg,
        visual_mesh_scale=visual_mesh_scale,
    )
    diagnostics = _env_truthy("AMONGUS_UE_ROBOT_SPAWN_DIAGNOSTICS")
    robot_summary: dict[str, object] = {
        "urdf_path": str(urdf_path),
        "base_pos_m": [float(v) for v in base_pos_m],
        "base_quat_xyzw": None if base_quat_xyzw is None else [float(v) for v in base_quat_xyzw],
        "joint_positions": [float(v) for v in joint_positions],
        "mesh_source": mesh_source,
        "visual_basis_rpy_deg": [float(v) for v in visual_basis_rpy_deg],
        "visual_mesh_scale": float(visual_mesh_scale),
        "mesh_source_to_ue_scale": float(_mesh_source_to_ue_scale(mesh_source)),
        "cache_key": cache_key,
        "fbx_axis_forward": _fbx_axis_forward() if mesh_source == "fbx" else None,
        "fbx_axis_up": _fbx_axis_up() if mesh_source == "fbx" else None,
        "fbx_global_scale": float(_fbx_global_scale()) if mesh_source == "fbx" else None,
        "mirror_y_for_unreal_fbx_import": _mirror_y_for_unreal_fbx_import() if mesh_source == "fbx" else None,
        "articulation_registered": articulation_actor_sink is not None,
        "visual_mesh_asset_paths": [],
        "diagnostics_enabled": diagnostics,
    }
    links_detail: list[dict[str, object]] = []

    robot_visual_asset_paths: list[str] = []
    skipped_visual_links: list[str] = []

    for link_name in ordered_links:
        world = link_world.get(link_name)
        if world is None:
            continue
        link_info = links.get(link_name, {})
        visual_mesh = link_info.get("visual_mesh")
        visual_boxes = list(link_info.get("visual_boxes") or [])
        geometry = link_info.get("geometry")
        if not use_collision_geometry and use_visual_mesh and visual_mesh:
            visual_local = make_transform(
                link_info.get("visual_origin_xyz", (0.0, 0.0, 0.0)),
                link_info.get("visual_origin_rpy", (0.0, 0.0, 0.0)),
            )
            _set_rotation3(
                visual_local,
                _matmul3(_rotation3_from_transform(visual_local), visual_basis_correction),
            )
            visual_world = apply_transform(world, visual_local)
            mesh_asset_path = _import_robot_link_visual_mesh(
                urdf_path,
                str(visual_mesh),
                visual_asset_root=visual_asset_root,
                asset_name=Path(str(visual_mesh)).stem,
                visual_mesh_scale=float(visual_mesh_scale),
                visual_basis_rpy_deg=visual_basis_rpy_deg,
                mesh_source=str(mesh_source),
                model_id=str(model_id),
            )
            if mesh_asset_path is not None:
                compensated_mesh_scale = float(visual_mesh_scale) * _mesh_source_to_ue_scale(mesh_source)
                robot_visual_asset_paths.append(mesh_asset_path)
                actor = spawn_visual_mesh(
                    label=f"{GENERATED_SCENE_LABEL_PREFIX}{link_name}",
                    mesh_asset_path=mesh_asset_path,
                    pos_m=(
                        float(visual_world[0][3]),
                        float(visual_world[1][3]),
                        float(visual_world[2][3]),
                    ),
                    rotation_deg=matrix_to_rotator_deg(_rotation3_from_transform(visual_world)),
                    folder=folder,
                    color_rgba=color_rgba,
                    mesh_scale=compensated_mesh_scale,
                    rot_mat=_rotation3_from_transform(visual_world),
                )
                if articulation_actor_sink is not None:
                    articulation_actor_sink[str(link_name)] = actor
                if diagnostics:
                    links_detail.append(
                        {
                            "link_name": str(link_name),
                            "visual_mesh": str(visual_mesh),
                            "mesh_asset_path": str(mesh_asset_path),
                            "expected_genesis_pos_m": [
                                float(visual_world[0][3]),
                                float(visual_world[1][3]),
                                float(visual_world[2][3]),
                            ],
                            "expected_ue_location_cm": list(
                                m_to_cm((float(visual_world[0][3]), float(visual_world[1][3]), float(visual_world[2][3])))
                            ),
                            "expected_ue_rotation_deg": list(matrix_to_rotator_deg(_rotation3_from_transform(visual_world))),
                            "actor": _actor_transform_payload(actor, include_bounds=True),
                        }
                    )
                continue
            if not allow_collision_fallback:
                raise RuntimeError(f"Cannot load robot visual mesh for link '{link_name}': {visual_mesh}")
        elif not use_collision_geometry and use_visual_mesh and visual_boxes:
            for box_idx, box in enumerate(visual_boxes):
                visual_local = make_transform(
                    tuple(float(v) for v in box["xyz"]),
                    tuple(float(v) for v in box["rpy"]),
                )
                visual_world = apply_transform(world, visual_local)
                box_color = box.get("color_rgba") or color_rgba
                actor = spawn_box(
                    label=f"{GENERATED_SCENE_LABEL_PREFIX}{link_name}_box{box_idx}",
                    pos_m=(
                        float(visual_world[0][3]),
                        float(visual_world[1][3]),
                        float(visual_world[2][3]),
                    ),
                    size_m=tuple(float(item) for item in box["size"]),
                    rotation_deg=matrix_to_rotator_deg(_rotation3_from_transform(visual_world)),
                    folder=folder,
                    color_rgba=(
                        tuple(float(v) for v in box_color)
                        if box_color is not None
                        else color_rgba
                    ),
                )
                if articulation_actor_sink is not None:
                    articulation_actor_sink[f"{link_name}__box{box_idx}"] = actor
            continue
        elif not use_collision_geometry and use_visual_mesh and visual_mesh is None and not allow_collision_fallback:
            if geometry is None:
                skipped_visual_links.append(str(link_name))
                continue
            raise RuntimeError(f"URDF link '{link_name}' does not define a robot visual mesh.")
        if not use_collision_geometry and not allow_collision_fallback and use_visual_mesh:
            raise RuntimeError(f"Cannot fall back to collision geometry for link '{link_name}'.")
        collision_local = make_transform(link_info.get("collision_origin_xyz", (0.0, 0.0, 0.0)), link_info.get("collision_origin_rpy", (0.0, 0.0, 0.0)))
        collision_world = apply_transform(world, collision_local)
        pos_m = (float(collision_world[0][3]), float(collision_world[1][3]), float(collision_world[2][3]))
        rot_deg = matrix_to_rotator_deg(_rotation3_from_transform(collision_world))
        if geometry is None:
            spawn_marker(label=f"{GENERATED_SCENE_LABEL_PREFIX}{link_name}", pos_m=pos_m, scale_m=0.04, folder=folder, color_rgba=color_rgba)
        elif geometry["type"] == "cylinder":
            spawn_cylinder(
                label=f"{GENERATED_SCENE_LABEL_PREFIX}{link_name}",
                pos_m=pos_m,
                radius_m=float(geometry["radius"]),
                length_m=float(geometry["length"]),
                rotation_deg=rot_deg,
                folder=folder,
                color_rgba=color_rgba,
            )
        elif geometry["type"] == "box":
            spawn_box(
                label=f"{GENERATED_SCENE_LABEL_PREFIX}{link_name}",
                pos_m=pos_m,
                size_m=tuple(float(item) for item in geometry["size"]),
                rotation_deg=rot_deg,
                folder=folder,
                color_rgba=color_rgba,
            )

    robot_summary["visual_mesh_asset_paths"] = list(robot_visual_asset_paths)
    robot_summary["skipped_visual_links"] = skipped_visual_links
    if diagnostics:
        robot_summary["links"] = links_detail
    unreal.log(
        f"UE_SCENE: spawn_robot_from_urdf visual_mesh_entries={len(robot_visual_asset_paths)} "
        f"ordered_links={len(ordered_links)} skipped_visual_links={len(skipped_visual_links)} diagnostics={diagnostics}"
    )
    return robot_summary


def _preload_human_ue_calibration(scene_spec) -> dict[str, Any] | None:
    global _HUMAN_UE_CALIBRATION, _CALIB_BONE_PRESET, _HUMAN_UE_DRIVE_HUMAN_BONES, _SMPL_ROOT_ALIGN_BONE_OVERRIDE

    _HUMAN_UE_CALIBRATION = None
    _CALIB_BONE_PRESET = ""
    _HUMAN_UE_DRIVE_HUMAN_BONES = None
    _SMPL_ROOT_ALIGN_BONE_OVERRIDE = str(os.environ.get("AMONGUS_UE_SMPL_ROOT_ALIGN_BONE", "") or "").strip()

    path = resolve_human_ue_calibration_json_path(scene_spec, repo_root=REPO_ROOT)
    data = load_human_ue_calibration_dict(path)
    if not data:
        return None
    _HUMAN_UE_CALIBRATION = data
    uv_raw = data.get("ue_visible_human")
    uv = uv_raw if isinstance(uv_raw, dict) else {}
    _CALIB_BONE_PRESET = str(uv.get("bone_preset") or "").strip()
    bone_ov = str(uv.get("smpl_root_alignment_bone_name") or "").strip()
    if bone_ov:
        _SMPL_ROOT_ALIGN_BONE_OVERRIDE = bone_ov
    if "drive_human_bones" in uv:
        _HUMAN_UE_DRIVE_HUMAN_BONES = bool(uv.get("drive_human_bones"))
    bsov = uv.get("bone_control_space_override_int")
    if bsov is not None:
        try:
            os.environ["AMONGUS_UE_SMPL_BONE_CONTROL_SPACE_INT"] = str(int(bsov))
        except (TypeError, ValueError):
            pass
    unreal.log(f"UE_SCENE: human_ue_calibration path={path} preset={_CALIB_BONE_PRESET!r}")
    return data


def resolve_human_anchor(scene_spec) -> tuple[float, float, float]:
    return scene_spec.resolved_human_anchor()


def _motion_human_offsets_from_scene(scene_spec) -> tuple[tuple[float, float, float], list[float], bool, Path | None]:
    placement_path = resolve_human_scene_placement_json_path(scene_spec, repo_root=REPO_ROOT)
    if placement_path is not None:
        parsed = read_human_scene_placement_mesh_offset_m(placement_path)
        if parsed is not None:
            anchor_m, mesh_off_m, align_floor = parsed
            wo_ue = ue_world_point_from_genesis_m(np.asarray(mesh_off_m, dtype=np.float64)).tolist()
            return anchor_m, [float(x) for x in wo_ue], align_floor, placement_path
    anchor_m = resolve_human_anchor(scene_spec)
    wo_ue = ue_world_point_from_genesis_m(np.asarray(anchor_m, dtype=np.float64)).tolist()
    return anchor_m, [float(x) for x in wo_ue], bool(scene_spec.human.align_floor), None


SMPL_BODY_BONE_NAMES_DEFAULT = (
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
)

# SMPL body joint order (23 joints, root excluded). Multiple naming conventions for Bedlam / SMPL-X assets.
SMPL_BODY_BONE_PRESETS: dict[str, tuple[str, ...]] = {
    "smpl_pascal": SMPL_BODY_BONE_NAMES_DEFAULT,
    "smplx_snake_case": (
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
    ),
    "smplx_official": (
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
    ),
    "bedlam_m_avg": (
        "m_avg_L_Hip",
        "m_avg_R_Hip",
        "m_avg_Spine1",
        "m_avg_L_Knee",
        "m_avg_R_Knee",
        "m_avg_Spine2",
        "m_avg_L_Ankle",
        "m_avg_R_Ankle",
        "m_avg_Spine3",
        "m_avg_L_Foot",
        "m_avg_R_Foot",
        "m_avg_Neck",
        "m_avg_L_Collar",
        "m_avg_R_Collar",
        "m_avg_Head",
        "m_avg_L_Shoulder",
        "m_avg_R_Shoulder",
        "m_avg_L_Elbow",
        "m_avg_R_Elbow",
        "m_avg_L_Wrist",
        "m_avg_R_Wrist",
        "m_avg_L_Hand",
        "m_avg_R_Hand",
    ),
}

_VISIBLE_HUMAN_SMPL_BONE_NAMES: tuple[str, ...] | None = None
_VISIBLE_HUMAN_BONE_PRESET_NAME: str = ""
_VISIBLE_HUMAN_SKELETAL_MESH_PATH: str = ""
_VISIBLE_HUMAN_RELATIVE_SCALE: float = 1.0
_HUMAN_UE_CALIBRATION: dict[str, Any] | None = None
_SMPL_ROOT_ALIGN_BONE_OVERRIDE: str = ""
_CALIB_BONE_PRESET: str = ""
_HUMAN_UE_DRIVE_HUMAN_BONES: bool | None = None

# Cache session bone JSON path + mtime so canonical ticks do not re-read every frame.
_VHB_SESSION_CACHE: tuple[str, int, bool] = ("", 0, False)

_PELVIS_DEFERRED_ALIGN_COMPLETED: bool = False
_PELVIS_DEFERRED_ALIGN_ATTEMPTS: int = 0
_PELVIS_SOCKET_WORLD_ALIGN_DONE: bool = False
# UE game world for the last pelvis align attempt; PIE creates a new world each Play, so "done" must not persist.
_PELVIS_ALIGN_TRACKED_WORLD: object | None = None


def _reset_pelvis_align_state_if_pie_world_changed() -> bool:
    """Clear one-shot pelvis flags when PIE restarts or the active game world object changes.

    Socket / deferred bind fixes modify the live PoseableMesh on the duplicated PIE actor; stopping
    PIE destroys that instance while editor templates stay unmodified. Without this reset, globals
    stay \"done\" and the second Play skips fixes (human looks tall / offset again).
    """
    global _PELVIS_DEFERRED_ALIGN_COMPLETED, _PELVIS_DEFERRED_ALIGN_ATTEMPTS
    global _PELVIS_SOCKET_WORLD_ALIGN_DONE, _PELVIS_ALIGN_TRACKED_WORLD
    pie = _query_pie_world()
    if pie is None:
        _PELVIS_ALIGN_TRACKED_WORLD = None
        return False
    if _PELVIS_ALIGN_TRACKED_WORLD is None or _PELVIS_ALIGN_TRACKED_WORLD is not pie:
        _PELVIS_DEFERRED_ALIGN_COMPLETED = False
        _PELVIS_DEFERRED_ALIGN_ATTEMPTS = 0
        _PELVIS_SOCKET_WORLD_ALIGN_DONE = False
        _PELVIS_ALIGN_TRACKED_WORLD = pie
        return True
    return False


def _bone_name_ci_map(names: list[str]) -> dict[str, str]:
    return {str(n).lower(): str(n) for n in names}


def collect_bone_names_from_skeletal_mesh_asset(mesh) -> list[str]:
    out: list[str] = []
    if mesh is None:
        return out
    try:
        skel = getattr(mesh, "skeleton", None)
        if skel is None:
            return out
        ref = getattr(skel, "ref_skeleton", None)
        if ref is None and hasattr(skel, "get_reference_skeleton"):
            try:
                ref = skel.get_reference_skeleton()
            except Exception:
                ref = None
        if ref is not None:
            nfunc = getattr(ref, "get_num_bones", None) or getattr(ref, "get_num", None)
            n = int(nfunc()) if callable(nfunc) else 0
            gname = getattr(ref, "get_bone_name", None) or getattr(ref, "get_ref_bone_name", None)
            for i in range(max(n, 0)):
                if callable(gname):
                    try:
                        out.append(str(gname(i)))
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def _pick_smpl_bone_preset(bone_names: list[str]) -> tuple[str, tuple[str, ...]]:
    snake = SMPL_BODY_BONE_PRESETS["smplx_snake_case"]
    if not bone_names:
        return "smplx_snake_case", snake

    bones_exact = set(bone_names)
    lower_set = {str(n).lower() for n in bone_names}

    def _preset_hits(preset: tuple[str, ...]) -> int:
        hits = 0
        for b in preset:
            if b in bones_exact:
                hits += 1
            elif b.lower() in lower_set:
                hits += 1
        return hits

    best_name = "smplx_snake_case"
    best_hits = -1
    best_tuple: tuple[str, ...] = snake
    for name, preset in SMPL_BODY_BONE_PRESETS.items():
        hits = _preset_hits(preset)
        if hits > best_hits:
            best_hits, best_name, best_tuple = hits, name, preset
    if best_hits <= 0:
        return "smplx_snake_case", snake
    return best_name, best_tuple


def _collect_skeletal_mesh_bone_names(component) -> list[str]:
    try:
        mesh = (
            component.get_skeletal_mesh_asset()
            if hasattr(component, "get_skeletal_mesh_asset")
            else getattr(component, "skeletal_mesh", None)
        )
        asset_bones = collect_bone_names_from_skeletal_mesh_asset(mesh)
        if asset_bones:
            return asset_bones
    except Exception:
        pass
    out: list[str] = []
    try:
        if hasattr(component, "get_bone_names"):
            raw = component.get_bone_names()
            if raw is not None:
                return [str(x) for x in raw]
    except Exception:
        pass
    try:
        nfunc = getattr(component, "get_num_bones", None)
        gname = getattr(component, "get_bone_name", None)
        if callable(nfunc) and callable(gname):
            n = int(nfunc())
            names: list[str] = []
            for i in range(max(n, 0)):
                try:
                    names.append(str(gname(i)))
                except Exception:
                    pass
            if names:
                return names
    except Exception:
        pass
    try:
        mesh = component.get_skeletal_mesh_asset() if hasattr(component, "get_skeletal_mesh_asset") else getattr(component, "skeletal_mesh", None)
        if mesh is None:
            return out
        skel = getattr(mesh, "skeleton", None)
        if skel is None:
            return out
        ref = getattr(skel, "ref_skeleton", None)
        if ref is None and hasattr(skel, "get_reference_skeleton"):
            try:
                ref = skel.get_reference_skeleton()
            except Exception:
                ref = None
        if ref is not None:
            nfunc = getattr(ref, "get_num_bones", None) or getattr(ref, "get_num", None)
            n = int(nfunc()) if callable(nfunc) else 0
            gname = getattr(ref, "get_bone_name", None) or getattr(ref, "get_ref_bone_name", None)
            for i in range(max(n, 0)):
                if callable(gname):
                    try:
                        out.append(str(gname(i)))
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def _smpl_body_bone_names_effective() -> tuple[str, ...]:
    raw = str(os.environ.get("AMONGUS_UE_SMPL_BODY_BONE_NAMES", "") or "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) == len(SMPL_BODY_BONE_NAMES_DEFAULT):
            return tuple(parts)
        unreal.log_warning(
            f"AMONGUS_UE_SMPL_BODY_BONE_NAMES has {len(parts)} entries, need {len(SMPL_BODY_BONE_NAMES_DEFAULT)} — use autodetect."
        )
    if _VISIBLE_HUMAN_SMPL_BONE_NAMES is not None:
        return _VISIBLE_HUMAN_SMPL_BONE_NAMES
    return SMPL_BODY_BONE_NAMES_DEFAULT


def _visible_human_bone_dump_path() -> Path | None:
    session_root = str(os.environ.get("AMONGUS_SESSION_DIR", "") or "").strip()
    if not session_root:
        return None
    return Path(session_root) / "visible_human_bones.json"


def _restore_visible_human_bone_mapping_from_session() -> bool:
    global _VISIBLE_HUMAN_SMPL_BONE_NAMES, _VISIBLE_HUMAN_BONE_PRESET_NAME
    global _VHB_SESSION_CACHE

    path = _visible_human_bone_dump_path()
    if path is None or not path.is_file():
        return False
    try:
        path_resolved = str(path.resolve())
        mtime_ns = int(path.stat().st_mtime_ns)
    except OSError:
        return False
    c_path, c_mtime, c_ok = _VHB_SESSION_CACHE
    if c_ok and path_resolved == c_path and mtime_ns == c_mtime:
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    mapping = payload.get("smpl_body_mapping")
    preset = str(payload.get("preset") or "").strip()
    if isinstance(mapping, list) and len(mapping) == len(SMPL_BODY_BONE_NAMES_DEFAULT):
        _VISIBLE_HUMAN_SMPL_BONE_NAMES = tuple(str(x) for x in mapping)
        _VISIBLE_HUMAN_BONE_PRESET_NAME = preset or "session_visible_human_bones"
        _VHB_SESSION_CACHE = (path_resolved, mtime_ns, True)
        unreal.log(
            "UE_SCENE: SMPL bone names from session visible_human_bones.json "
            f"preset={_VISIBLE_HUMAN_BONE_PRESET_NAME!r}"
        )
        return True
    if preset in SMPL_BODY_BONE_PRESETS:
        _VISIBLE_HUMAN_SMPL_BONE_NAMES = SMPL_BODY_BONE_PRESETS[preset]
        _VISIBLE_HUMAN_BONE_PRESET_NAME = preset
        _VHB_SESSION_CACHE = (path_resolved, mtime_ns, True)
        unreal.log(
            "UE_SCENE: SMPL bone preset from session visible_human_bones.json "
            f"preset={preset!r}"
        )
        return True
    return False


def _ensure_visible_human_bone_mapping_for_component(component) -> bool:
    global _VISIBLE_HUMAN_SMPL_BONE_NAMES, _VISIBLE_HUMAN_BONE_PRESET_NAME

    if _restore_visible_human_bone_mapping_from_session():
        return True
    if _VISIBLE_HUMAN_SMPL_BONE_NAMES is not None:
        return False
    mesh_bones = _collect_skeletal_mesh_bone_names(component)
    if not mesh_bones:
        return False
    preset_name, preset_tuple = _pick_smpl_bone_preset(mesh_bones)
    _VISIBLE_HUMAN_BONE_PRESET_NAME = preset_name
    _VISIBLE_HUMAN_SMPL_BONE_NAMES = preset_tuple
    return True


SMPL_BODY_PARENT_INDICES: tuple[int, ...] = (
    -1,  # left_hip
    -1,  # right_hip
    -1,  # spine1
    0,  # left_knee
    1,  # right_knee
    2,  # spine2
    3,  # left_ankle
    4,  # right_ankle
    5,  # spine3
    6,  # left_foot
    7,  # right_foot
    8,  # neck
    8,  # left_collar
    8,  # right_collar
    11,  # head
    12,  # left_shoulder
    13,  # right_shoulder
    15,  # left_elbow
    16,  # right_elbow
    17,  # left_wrist
    18,  # right_wrist
    19,  # left_hand
    20,  # right_hand
)


def _rotvec_to_matrix(rotvec_xyz: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = (float(rotvec_xyz[0]), float(rotvec_xyz[1]), float(rotvec_xyz[2]))
    angle = float(np.linalg.norm([rx, ry, rz]))
    if angle <= 1.0e-9:
        return np.eye(3, dtype=np.float64)
    ax, ay, az = rx / angle, ry / angle, rz / angle
    half = 0.5 * angle
    s = float(np.sin(half))
    qx, qy, qz, qw = ax * s, ay * s, az * s, float(np.cos(half))
    return np.asarray(ue_rotation_matrix_from_quat_xyzw([qx, qy, qz, qw]), dtype=np.float64)


def _smpl_body_component_space_rotators(
    body_pose_floats: list[float],
    smpl_root_rot_genesis: np.ndarray | None = None,
) -> list[object]:
    """Body-joint rotators in UE component space.

    When ``smpl_root_rot_genesis`` is None, root orientation is assumed to live on the **Actor**
    (legacy: parent==-1 joints use local-only FK). When set, SMPL global_orient is multiplied into
    FK (``root @ local`` for root-level joints) and the Actor should stay at identity rotation.
    """
    root_mat = (
        np.asarray(smpl_root_rot_genesis, dtype=np.float64).reshape(3, 3)
        if smpl_root_rot_genesis is not None
        else np.eye(3, dtype=np.float64)
    )
    triples = [
        body_pose_floats[i : i + 3]
        for i in range(0, len(body_pose_floats), 3)
        if i + 3 <= len(body_pose_floats)
    ]
    n = min(len(triples), len(SMPL_BODY_PARENT_INDICES))
    global_mats: list[np.ndarray] = []
    out: list[object] = []
    use_root_in_fk = smpl_root_rot_genesis is not None
    for i in range(n):
        local_mat = _rotvec_to_matrix(tuple(float(v) for v in triples[i]))
        parent = int(SMPL_BODY_PARENT_INDICES[i])
        if parent < 0:
            global_mat = root_mat @ local_mat if use_root_in_fk else local_mat
        else:
            global_mat = global_mats[parent] @ local_mat
        global_mats.append(global_mat)
        ue_mat = ue_world_rotation_from_genesis(global_mat)
        roll, pitch, yaw = ue_rotator_deg_from_matrix(np.asarray(ue_mat, dtype=np.float64))
        out.append(unreal.Rotator(float(roll), float(pitch), float(yaw)))
    return out


def _coerce_bone_space(space: object) -> object:
    bone_spaces = getattr(unreal, "BoneSpaces", None)
    if bone_spaces is not None:
        for attr in ("COMPONENT_SPACE", "ComponentSpace"):
            value = getattr(bone_spaces, attr, None)
            if value is not None:
                return value
    return space


def _bone_name_arg(mesh_name: str) -> Any:
    ctor = getattr(unreal, "Name", None)
    if not callable(ctor):
        return str(mesh_name)
    try:
        return ctor(str(mesh_name))
    except Exception:
        return str(mesh_name)


def _set_bone_rotation(component, mesh_name: str, rotator: object, bone_space: object) -> None:
    set_fn = getattr(component, "set_bone_rotation_by_name", None)
    coerced_space = _coerce_bone_space(bone_space)
    errors: list[str] = []
    if set_fn is not None:
        for name_arg in (_bone_name_arg(mesh_name), str(mesh_name)):
            try:
                set_fn(name_arg, rotator, coerced_space)
                return
            except Exception as exc:
                errors.append(f"set_bone_rotation_by_name:{type(exc).__name__}:{repr(exc)[:120]}")
    transform_fn = getattr(component, "set_bone_transform_by_name", None)
    if transform_fn is not None:
        try:
            transform = unreal.Transform(
                unreal.Vector(0.0, 0.0, 0.0),
                rotator,
                unreal.Vector(1.0, 1.0, 1.0),
            )
        except Exception:
            transform = None
        if transform is not None:
            for name_arg in (_bone_name_arg(mesh_name), str(mesh_name)):
                try:
                    transform_fn(name_arg, transform, coerced_space)
                    return
                except Exception as exc:
                    errors.append(f"set_bone_transform_by_name:{type(exc).__name__}:{repr(exc)[:120]}")
    raise AttributeError(";".join(errors) or "no_supported_bone_rotation_api")


def _component_space_bone_space() -> object:
    bone_spaces = getattr(unreal, "BoneSpaces", None)
    if bone_spaces is not None:
        for attr in ("COMPONENT_SPACE", "ComponentSpace"):
            value = getattr(bone_spaces, attr, None)
            if value is not None:
                return value
    return _coerce_bone_space(1)


_SMPL_ROOT_ALIGNMENT_NAME_CANDIDATES: tuple[str, ...] = (
    "m_avg_Pelvis",
    "f_avg_Pelvis",
    "pelvis",
    "Pelvis",
    "PELVIS",
    "hips",
    "Hips",
    "mixamorig:Hips",
    "root",
    "Root",
    "m_avg_ROOT",
    "SMPLX-neutral",
)


def _resolve_smpl_root_alignment_bone_name(mesh_bones: list[str]) -> str | None:
    """Skeleton bone for SMPL pelvis / global root in bind pose (must match Genesis joint 0 target)."""
    if not mesh_bones:
        return None
    override = str(_SMPL_ROOT_ALIGN_BONE_OVERRIDE or "").strip()
    if override:
        ci_map = _bone_name_ci_map(mesh_bones)
        hit = ci_map.get(override.lower())
        if hit is not None:
            return hit
    exact = set(mesh_bones)
    lower_map = {str(b).lower(): str(b) for b in mesh_bones}
    for cand in _SMPL_ROOT_ALIGNMENT_NAME_CANDIDATES:
        if cand in exact:
            return cand
        mapped = lower_map.get(str(cand).lower())
        if mapped is not None:
            return mapped
    for b in mesh_bones:
        bl = str(b).lower()
        if "pelvis" in bl and "twist" not in bl:
            return str(b)
    b0 = str(mesh_bones[0])
    b0l = b0.lower()
    for hint in ("pelvis", "hip"):
        if hint in b0l:
            return b0
    return None


def _poseable_mesh_bone_names_for_alignment(component) -> list[str]:
    mesh_bones: list[str] = []
    try:
        mesh = (
            component.get_skeletal_mesh_asset()
            if hasattr(component, "get_skeletal_mesh_asset")
            else getattr(component, "skeletal_mesh", None)
        )
        mesh_bones = collect_bone_names_from_skeletal_mesh_asset(mesh)
    except Exception:
        mesh_bones = []
    if not mesh_bones:
        try:
            mesh_bones = _collect_skeletal_mesh_bone_names(component)
        except Exception:
            mesh_bones = []
    return mesh_bones


def _vector_to_cm_list(value) -> list[float] | None:
    if value is None:
        return None
    for getter in ("get_translation", "get_location"):
        fn = getattr(value, getter, None)
        if callable(fn):
            try:
                got = fn()
                return [float(got.x), float(got.y), float(got.z)]
            except Exception:
                pass
    for attr in ("translation", "location"):
        got = getattr(value, attr, None)
        if got is not None:
            try:
                return [float(got.x), float(got.y), float(got.z)]
            except Exception:
                pass
    try:
        return [float(value.x), float(value.y), float(value.z)]
    except Exception:
        return None


def _scene_component_relative_translation_cm(component) -> list[float] | None:
    """Read SceneComponent relative location; UE Python does not always expose get_relative_location()."""
    if component is None:
        return None
    fn = getattr(component, "get_relative_location", None)
    if callable(fn):
        try:
            v = fn()
            return [float(v.x), float(v.y), float(v.z)]
        except Exception:
            pass
    if hasattr(component, "get_editor_property"):
        for prop in ("relative_location", "RelativeLocation"):
            try:
                v = component.get_editor_property(prop)
                if v is not None:
                    return [float(v.x), float(v.y), float(v.z)]
            except Exception:
                continue
    rel = getattr(component, "relative_location", None)
    if rel is not None:
        try:
            return [float(rel.x), float(rel.y), float(rel.z)]
        except Exception:
            pass
    gt = getattr(component, "get_relative_transform", None)
    if callable(gt):
        try:
            tr = gt()
            for getter in ("get_translation", "translation"):
                g = getattr(tr, getter, None)
                if callable(g):
                    v = g()
                else:
                    v = g if getter == "translation" else None
                if v is not None:
                    try:
                        return [float(v.x), float(v.y), float(v.z)]
                    except Exception:
                        continue
        except Exception:
            pass
    return None


def _ref_pose_bone_component_cm(component, bone_name: str) -> list[float] | None:
    try:
        mesh = (
            component.get_skeletal_mesh_asset()
            if hasattr(component, "get_skeletal_mesh_asset")
            else getattr(component, "skeletal_mesh", None)
        )
    except Exception:
        mesh = None
    if mesh is None:
        return None
    try:
        skel = getattr(mesh, "skeleton", None)
        ref = getattr(skel, "ref_skeleton", None) if skel is not None else None
        if ref is None and skel is not None and hasattr(skel, "get_reference_skeleton"):
            ref = skel.get_reference_skeleton()
    except Exception:
        ref = None
    if ref is None:
        return None
    try:
        idx = int(component.get_bone_index(str(bone_name)))
    except Exception:
        idx = -1
    if idx < 0:
        return None
    for getter in ("get_ref_bone_pose", "get_bone_pose", "get_raw_ref_bone_pose"):
        fn = getattr(ref, getter, None)
        if not callable(fn):
            continue
        try:
            pose = fn(idx)
            vec = _vector_to_cm_list(pose)
            if vec is not None:
                return vec
        except Exception:
            pass
    return None


def _transform_to_mat4_unreal(t) -> np.ndarray | None:
    if t is None:
        return None
    try:
        tr = t.get_translation()
        trans = np.asarray([float(tr.x), float(tr.y), float(tr.z)], dtype=np.float64).reshape(3)
        quat = t.get_rotation()
        q = [float(quat.x), float(quat.y), float(quat.z), float(quat.w)]
    except Exception:
        return None
    try:
        R = np.asarray(quaternion_xyzw_to_matrix(q), dtype=np.float64).reshape(3, 3)
    except Exception:
        return None
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = trans
    return M


def _reference_skeleton_for_poseable(component) -> tuple[object, object] | None:
    try:
        mesh = (
            component.get_skeletal_mesh_asset()
            if hasattr(component, "get_skeletal_mesh_asset")
            else getattr(component, "skeletal_mesh", None)
        )
    except Exception:
        mesh = None
    if mesh is None:
        return None
    try:
        skel = getattr(mesh, "skeleton", None)
        if skel is None:
            return None
        ref = getattr(skel, "ref_skeleton", None)
        if ref is None and hasattr(skel, "get_reference_skeleton"):
            ref = skel.get_reference_skeleton()
    except Exception:
        ref = None
    if ref is None:
        return None
    return ref, mesh


def _ref_bone_indices_chain_root_first(ref: object, leaf_idx: int) -> list[int] | None:
    if leaf_idx < 0:
        return None
    parent_fn = getattr(ref, "get_parent_index", None)
    if not callable(parent_fn):
        parent_fn = getattr(ref, "get_raw_parent_index", None)
    if not callable(parent_fn):
        return None
    chain: list[int] = []
    cur = int(leaf_idx)
    for _ in range(512):
        chain.append(cur)
        try:
            p = int(parent_fn(cur))
        except Exception:
            break
        if p < 0:
            break
        cur = p
    if not chain:
        return None
    chain.reverse()
    return chain


def _accumulated_ref_pose_bone_translation_cm(component, bone_mesh_name: str) -> list[float] | None:
    """Mesh-root accumulated translation of ``bone_mesh_name`` from reference pose (UE cm).

    ``get_bone_location_by_name`` often returns zeros on a freshly built PoseableMeshComponent; the
    reference skeleton chain ``parent -> child`` ref transforms are deterministic for bind alignment.
    """
    got = _reference_skeleton_for_poseable(component)
    if got is None:
        return None
    ref, mesh = got
    try:
        idx = int(component.get_bone_index(str(bone_mesh_name)))
    except Exception:
        idx = -1
    if idx < 0:
        return None
    chain = _ref_bone_indices_chain_root_first(ref, idx)
    if not chain:
        return None
    skel = getattr(mesh, "skeleton", None)

    def _local_mat_for_bone(bidx: int) -> np.ndarray | None:
        pose_fn = getattr(ref, "get_ref_bone_pose", None)
        if callable(pose_fn):
            try:
                pose = pose_fn(int(bidx))
                L = _transform_to_mat4_unreal(pose)
                if L is not None:
                    return L
            except Exception:
                pass
        if skel is not None:
            for alt in ("get_ref_bone_pose", "get_reference_bone_pose", "get_ref_bone_transform"):
                afn = getattr(skel, alt, None)
                if not callable(afn):
                    continue
                try:
                    pose = afn(int(bidx))
                    L = _transform_to_mat4_unreal(pose)
                    if L is not None:
                        return L
                except Exception:
                    continue
        return None

    M = np.eye(4, dtype=np.float64)
    for b in chain:
        L = _local_mat_for_bone(int(b))
        if L is None:
            return None
        M = M @ L
    p = M[:3, 3]
    return [float(p[0]), float(p[1]), float(p[2])]


def _write_visible_human_pelvis_align_report(payload: dict[str, Any]) -> None:
    session_root = str(os.environ.get("AMONGUS_SESSION_DIR", "") or "").strip()
    if not session_root:
        return
    try:
        merge = dict(payload)
        if _SMPL_ROOT_ALIGN_BONE_OVERRIDE:
            merge["smpl_root_alignment_bone_override"] = str(_SMPL_ROOT_ALIGN_BONE_OVERRIDE)
        Path(session_root).mkdir(parents=True, exist_ok=True)
        (Path(session_root) / "visible_human_pelvis_align.json").write_text(
            json.dumps(merge, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _align_poseable_mesh_smpl_root_to_parent_origin(component) -> dict[str, Any]:
    """Shift mesh so the SMPL root (pelvis) bone sits on the component parent origin in ref pose.

    Genesis ``root_translation_world_m`` targets the pelvis; the actor transform must pivot there.
    This uses the skeletal asset bind/ref pose only (no manual per-mesh numeric calibration).
    """
    out: dict[str, Any] = {"applied": False}
    if component is None:
        _write_visible_human_pelvis_align_report(out)
        return out
    mesh_bones = _poseable_mesh_bone_names_for_alignment(component)
    ci_map = _bone_name_ci_map(mesh_bones)
    bone_pick = _resolve_smpl_root_alignment_bone_name(mesh_bones)
    if not bone_pick:
        unreal.log_warning(
            "UE_SCENE: SMPL root alignment skipped — no pelvis/root bone name matched on skeleton. "
            "SMPL trans may not match actor pivot.",
        )
        out["reason"] = "no_root_bone_name"
        _write_visible_human_pelvis_align_report({**out, "mesh_bone_count": len(mesh_bones)})
        return {**out, "mesh_bone_count": len(mesh_bones)}

    mesh_name = str(ci_map.get(str(bone_pick).lower(), str(bone_pick)))

    reset_pose = getattr(component, "reset_all_bone_transforms", None)
    if callable(reset_pose):
        try:
            reset_pose()
        except Exception:
            pass

    if hasattr(component, "set_relative_location"):
        try:
            component.set_relative_location(unreal.Vector(0.0, 0.0, 0.0), False, False)
        except Exception:
            pass

    refresh = getattr(component, "refresh_bone_transforms", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass

    get_loc = getattr(component, "get_bone_location_by_name", None)
    if not callable(get_loc):
        ro = {**out, "reason": "no_get_bone_location", "bone": bone_pick}
        _write_visible_human_pelvis_align_report(ro)
        return ro
    bone_space = _component_space_bone_space()
    try:
        bone_index = component.get_bone_index(mesh_name)
        if bone_index is None or int(bone_index) < 0:
            ro = {**out, "reason": "bone_not_in_mesh", "bone": mesh_name}
            _write_visible_human_pelvis_align_report(ro)
            return ro
    except Exception:
        ro = {**out, "reason": "bone_index_error", "bone": mesh_name}
        _write_visible_human_pelvis_align_report(ro)
        return ro
    align_source = "component_bone_location"
    bx = by = bz = 0.0
    try:
        loc = get_loc(_bone_name_arg(mesh_name), bone_space)
        bx, by, bz = float(loc.x), float(loc.y), float(loc.z)
    except Exception as exc:
        acc_cm = _accumulated_ref_pose_bone_translation_cm(component, str(mesh_name))
        if acc_cm is not None and max(abs(float(v)) for v in acc_cm) > 1.0e-6:
            bx, by, bz = (float(acc_cm[i]) for i in range(3))
            align_source = "ref_skeleton_chain_accum"
            out["accumulated_ref_fallback_after_exc"] = True
        else:
            ref_vec = _ref_pose_bone_component_cm(component, str(mesh_name))
            if ref_vec is not None:
                bx, by, bz = (float(ref_vec[i]) for i in range(3))
                align_source = "ref_skeleton_pose"
            else:
                ro = {**out, "reason": repr(exc), "bone": mesh_name}
                _write_visible_human_pelvis_align_report(ro)
                return ro
    if abs(bx) <= 1.0e-6 and abs(by) <= 1.0e-6 and abs(bz) <= 1.0e-6:
        acc_cm = _accumulated_ref_pose_bone_translation_cm(component, str(mesh_name))
        if acc_cm is not None and max(abs(float(v)) for v in acc_cm) > 1.0e-6:
            bx, by, bz = (float(acc_cm[i]) for i in range(3))
            align_source = "ref_skeleton_chain_accum"
        else:
            ref_vec = _ref_pose_bone_component_cm(component, str(mesh_name))
            if ref_vec is not None and max(abs(float(v)) for v in ref_vec) > 1.0e-6:
                bx, by, bz = (float(ref_vec[i]) for i in range(3))
                align_source = "ref_skeleton_pose"

    align_bone_used = str(mesh_name)
    bone_primary = str(bone_pick)
    sx, sy, sz = -bx, -by, -bz
    candidate_bone_component_cm: dict[str, list[float] | None] = {}
    for dbg_name in ("SMPLX-neutral", "root", "pelvis", "left_ankle", "right_ankle", "left_foot", "right_foot"):
        nm = ci_map.get(str(dbg_name).lower())
        if not nm:
            candidate_bone_component_cm[dbg_name] = None
            continue
        try:
            loc_dbg = get_loc(_bone_name_arg(nm), bone_space)
            candidate_bone_component_cm[dbg_name] = [float(loc_dbg.x), float(loc_dbg.y), float(loc_dbg.z)]
        except Exception:
            candidate_bone_component_cm[dbg_name] = None
    component_relative_location_cm = _scene_component_relative_translation_cm(component)
    component_bounds: dict[str, list[float] | None] = {"origin_cm": None, "extent_cm": None}
    try:
        bounds = component.bounds
        bo = getattr(bounds, "origin", None)
        be = getattr(bounds, "box_extent", None)
        component_bounds = {
            "origin_cm": [float(bo.x), float(bo.y), float(bo.z)] if bo is not None else None,
            "extent_cm": [float(be.x), float(be.y), float(be.z)] if be is not None else None,
        }
    except Exception:
        pass
    if hasattr(component, "set_relative_location"):
        try:
            component.set_relative_location(unreal.Vector(sx, sy, sz), False, False)
        except Exception as exc:
            ro = {**out, "reason": repr(exc), "bone": align_bone_used}
            _write_visible_human_pelvis_align_report(ro)
            return ro

    out["applied"] = True
    out["bone"] = align_bone_used
    out["bone_primary_resolved"] = bone_primary
    out["align_source"] = align_source
    out["relative_shift_cm"] = [sx, sy, sz]
    out["bind_pelvis_component_cm"] = [bx, by, bz]
    out["candidate_bone_component_cm"] = candidate_bone_component_cm
    out["component_relative_location_before_cm"] = component_relative_location_cm
    out["component_bounds_before_align_cm"] = component_bounds
    # region agent log
    if _amongus_truthy_env("AMONGUS_DEBUG_NDJSON", default=False):
        try:
            _ue_debug_ndjson(
                hypothesis_id="UE_PELVIS_ALIGN",
                location="ue_common_scene_loader.py:_align_poseable_mesh_smpl_root_to_parent_origin",
                message="bind-pose pelvis shift and bone candidates (cm component space)",
                data={
                    "bone_primary_resolved": bone_primary,
                    "bone_used_for_shift": align_bone_used,
                    "align_source": align_source,
                    "relative_shift_cm": [sx, sy, sz],
                    "bind_pelvis_component_cm": [bx, by, bz],
                    "candidate_bone_component_cm": candidate_bone_component_cm,
                    "mesh_bone_count": len(mesh_bones),
                },
            )
        except Exception:
            pass
    # endregion agent log
    if abs(bx) > 1.0e-4 or abs(by) > 1.0e-4 or abs(bz) > 1.0e-4:
        unreal.log(
            f"UE_SCENE: PoseableMesh SMPL root bind-pose align bone={align_bone_used!r} "
            f"relative_shift_cm=[{sx:.4f}, {sy:.4f}, {sz:.4f}]",
        )
    else:
        unreal.log(
            f"UE_SCENE: PoseableMesh SMPL bind align bone={align_bone_used!r} "
            f"and fallbacks at component origin (no mesh shift)."
        )
    _write_visible_human_pelvis_align_report(out)
    return out


def _apply_pelvis_smpl_global_orient(component, smpl_root_rot_genesis: np.ndarray) -> tuple[bool, str]:
    """Drive skeleton pelvis/root bone with SMPL global_orient so pelvis-weighted verts rotate (not only Actor)."""
    mesh_bones = _poseable_mesh_bone_names_for_alignment(component)
    bone_pick = _resolve_smpl_root_alignment_bone_name(mesh_bones)
    if not bone_pick:
        return False, ""
    ci_map = _bone_name_ci_map(mesh_bones)
    mesh_name = ci_map.get(str(bone_pick).lower(), str(bone_pick))
    try:
        bone_index = component.get_bone_index(mesh_name)
        if bone_index is None or int(bone_index) < 0:
            return False, str(mesh_name)
    except Exception:
        return False, str(mesh_name)
    root_g = np.asarray(smpl_root_rot_genesis, dtype=np.float64).reshape(3, 3)
    ue_mat = ue_world_rotation_from_genesis(root_g)
    roll, pitch, yaw = ue_rotator_deg_from_matrix(np.asarray(ue_mat, dtype=np.float64))
    bone_space = _component_space_bone_space()
    try:
        _set_bone_rotation(
            component,
            mesh_name,
            unreal.Rotator(float(roll), float(pitch), float(yaw)),
            bone_space,
        )
    except Exception:
        return False, str(mesh_name)
    return True, str(mesh_name)


def _apply_smpl_body_pose_to_component(
    component,
    body_pose_floats: list[float],
    smpl_root_rot_genesis: np.ndarray | None = None,
) -> tuple[int, list[str]]:
    mesh_bones: list[str] = []
    try:
        mesh = (
            component.get_skeletal_mesh_asset()
            if hasattr(component, "get_skeletal_mesh_asset")
            else getattr(component, "skeletal_mesh", None)
        )
        mesh_bones = collect_bone_names_from_skeletal_mesh_asset(mesh)
    except Exception:
        mesh_bones = []
    if not mesh_bones:
        mesh_bones = _collect_skeletal_mesh_bone_names(component)
    _ensure_visible_human_bone_mapping_for_component(component)
    bone_names = _smpl_body_bone_names_effective()
    ci_map = _bone_name_ci_map(mesh_bones)

    if smpl_root_rot_genesis is not None:
        _apply_pelvis_smpl_global_orient(component, smpl_root_rot_genesis)

    rotators = _smpl_body_component_space_rotators(body_pose_floats, smpl_root_rot_genesis=smpl_root_rot_genesis)
    bone_space = _component_space_bone_space()
    applied = 0
    missing: list[str] = []
    for joint_index, name in enumerate(bone_names):
        if joint_index >= len(rotators):
            break
        mesh_name = ci_map.get(str(name).lower(), str(name))
        try:
            bone_index = component.get_bone_index(mesh_name)
        except Exception:
            bone_index = -1
        if bone_index is None or int(bone_index) < 0:
            missing.append(str(name))
            continue
        try:
            _set_bone_rotation(component, mesh_name, rotators[joint_index], bone_space)
            applied += 1
        except Exception as exc:
            missing.append(f"{name}:{type(exc).__name__}:{repr(exc)[:160]}")
    if applied > 0:
        mark_dirty = getattr(component, "mark_render_transform_dirty", None)
        if callable(mark_dirty):
            try:
                mark_dirty()
            except Exception:
                pass
    return applied, missing


# region agent log
def _amongus_debug_log_path_ue() -> str:
    raw = str(os.environ.get("AMONGUS_DEBUG_NDJSON_LOG", "") or "").strip()
    if raw:
        return raw
    return str(PROJECT_PATHS.root / ".cursor" / "debug-2f5e72.log")


def _ue_debug_ndjson_target_paths() -> list[str]:
    primary = _amongus_debug_log_path_ue()
    out: list[str] = [primary]
    sr = str(os.environ.get("AMONGUS_SESSION_DIR", "") or "").strip()
    if sr:
        mp = str(Path(sr) / "amongus_human_debug.ndjson")
        if mp and mp != primary:
            out.append(mp)
    return out


def _amongus_debug_session_id_ue() -> str:
    return str(os.environ.get("AMONGUS_DEBUG_SESSION_ID", "") or "").strip() or "2f5e72"


def _amongus_debug_ndjson_every_ue(*, default: int = 25) -> int:
    raw = str(os.environ.get("AMONGUS_DEBUG_NDJSON_EVERY", "") or "").strip()
    if not raw:
        return max(int(default), 1)
    try:
        return max(int(raw), 1)
    except ValueError:
        return max(int(default), 1)


def _ue_debug_ndjson(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    if not _amongus_truthy_env("AMONGUS_DEBUG_NDJSON", default=False):
        return
    try:
        payload: dict[str, Any] = {
            "sessionId": _amongus_debug_session_id_ue(),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": dict(data),
            "timestamp": int(time.time() * 1000),
        }
        line = json.dumps(payload, ensure_ascii=True) + "\n"
        for p in _ue_debug_ndjson_target_paths():
            try:
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a", encoding="utf-8") as handle:
                    handle.write(line)
            except Exception:
                pass
    except Exception:
        pass


def _ue_vec3_cm(v) -> list[float] | None:
    try:
        return [float(v.x), float(v.y), float(v.z)]
    except Exception:
        return None


def _ue_poseable_world_diag(actor, comp) -> dict[str, Any]:
    diag: dict[str, Any] = {}
    try:
        diag["actor_loc_cm"] = _ue_vec3_cm(actor.get_actor_location())
    except Exception:
        diag["actor_loc_cm"] = None
    try:
        r = actor.get_actor_rotation()
        diag["actor_rpy_deg"] = [float(r.roll), float(r.pitch), float(r.yaw)]
    except Exception:
        pass
    if comp is None:
        return diag
    try:
        diag["poseable_world_loc_cm"] = _ue_vec3_cm(comp.get_world_location())
        rel = _scene_component_relative_translation_cm(comp)
        if rel is not None:
            diag["poseable_rel_loc_cm"] = rel
    except Exception:
        pass
    try:
        b = comp.bounds
        diag["bounds_origin_cm"] = _ue_vec3_cm(getattr(b, "origin", None))
        diag["bounds_extent_cm"] = _ue_vec3_cm(getattr(b, "box_extent", None))
    except Exception:
        pass
    sk: dict[str, Any] = {}
    for key in (
        "pelvis",
        "Pelvis",
        "left_ankle",
        "right_ankle",
        "left_foot",
        "right_foot",
        "neck",
        "head",
        "SMPLX-neutral",
        "root",
    ):
        try:
            fn = getattr(comp, "get_socket_location", None)
            sk[key] = _ue_vec3_cm(fn(_bone_name_arg(key))) if callable(fn) else None
        except Exception:
            sk[key] = None
    diag["socket_world_cm"] = sk
    return diag


def _ue_poseable_alignment_summary(actor, comp) -> dict[str, Any]:
    diag = _ue_poseable_world_diag(actor, comp)
    out: dict[str, Any] = {}
    for key in ("actor_loc_cm", "poseable_world_loc_cm", "poseable_rel_loc_cm", "bounds_origin_cm", "bounds_extent_cm"):
        if key in diag:
            out[key] = diag[key]
    origin = diag.get("bounds_origin_cm")
    extent = diag.get("bounds_extent_cm")
    if isinstance(origin, list) and isinstance(extent, list) and len(origin) >= 3 and len(extent) >= 3:
        out["bounds_bottom_z_cm"] = float(origin[2]) - float(extent[2])
    sockets = diag.get("socket_world_cm")
    if isinstance(sockets, dict):
        actor_loc = diag.get("actor_loc_cm")
        picked: dict[str, Any] = {}
        for key in ("pelvis", "root", "SMPLX-neutral", "left_foot", "right_foot"):
            loc = sockets.get(key)
            if loc is not None:
                picked[key] = loc
        if picked:
            out["socket_world_cm"] = picked
        if isinstance(actor_loc, list) and len(actor_loc) >= 3:
            offsets: dict[str, list[float]] = {}
            for key, loc in picked.items():
                if isinstance(loc, list) and len(loc) >= 3:
                    offsets[key] = [float(loc[i]) - float(actor_loc[i]) for i in range(3)]
            if offsets:
                out["socket_minus_actor_cm"] = offsets
    return out


def _apply_pelvis_socket_world_align_once(actor, comp, alignment_summary: dict[str, Any]) -> dict[str, Any] | None:
    """Use live pelvis socket vs actor (world cm) to shift the poseable mesh once per PIE/world session.

    This matches what diagnostics already report as ``socket_minus_actor_cm``: after bones + root drive,
    moving the component by ``-(pelvis_world - actor_world)`` brings the pelvis socket onto the actor
    pivot, correcting SMPLX-neutral/root vs anatomical pelvis hierarchy without per-asset constants.
    Applies full world-space XYZ (not Z-only).
    """
    global _PELVIS_SOCKET_WORLD_ALIGN_DONE
    if actor is None or comp is None:
        return None
    if _PELVIS_SOCKET_WORLD_ALIGN_DONE:
        return None
    if not _amongus_truthy_env("AMONGUS_UE_PELVIS_SOCKET_WORLD_ALIGN", default=True):
        _PELVIS_SOCKET_WORLD_ALIGN_DONE = True
        return None
    sm = alignment_summary.get("socket_minus_actor_cm") or {}
    raw = sm.get("pelvis")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    dx, dy, dz = float(raw[0]), float(raw[1]), float(raw[2])
    mag = max(abs(dx), abs(dy), abs(dz))
    if mag < 5.0:
        _PELVIS_SOCKET_WORLD_ALIGN_DONE = True
        return None
    if mag > 280.0:
        unreal.log_warning(f"UE_SCENE: pelvis socket world-align skipped (delta too large: {mag:.1f} cm)")
        _PELVIS_SOCKET_WORLD_ALIGN_DONE = True
        return None
    try:
        add_off = getattr(comp, "add_world_offset", None)
        if callable(add_off):
            add_off(unreal.Vector(-dx, -dy, -dz), False)
        else:
            wloc = comp.get_world_location()
            comp.set_world_location(
                unreal.Vector(float(wloc.x) - dx, float(wloc.y) - dy, float(wloc.z) - dz),
                False,
                False,
            )
    except Exception as exc:
        unreal.log_warning(f"UE_SCENE: pelvis socket world-align failed: {exc!r}")
        return None
    rf = getattr(comp, "refresh_bone_transforms", None)
    if callable(rf):
        try:
            rf()
        except Exception:
            pass
    mirrored = _mirror_poseable_relative_transform_to_editor_template(actor, comp)
    _PELVIS_SOCKET_WORLD_ALIGN_DONE = True
    payload = {
        "applied": True,
        "via": "socket_minus_actor_world_once",
        "world_offset_cm_applied": [-dx, -dy, -dz],
        "socket_minus_actor_cm_before": [dx, dy, dz],
        "editor_template_mirrored": bool(mirrored),
    }
    try:
        prev: dict[str, Any] = {}
        session_root = str(os.environ.get("AMONGUS_SESSION_DIR", "") or "").strip()
        if session_root:
            p = Path(session_root) / "visible_human_pelvis_align.json"
            if p.is_file():
                try:
                    prev = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    prev = {}
        _write_visible_human_pelvis_align_report({**prev, **payload})
    except Exception:
        pass
    unreal.log(
        "UE_SCENE: pelvis socket world-align once "
        f"offset_cm=[{-dx:.4f}, {-dy:.4f}, {-dz:.4f}] (from socket_minus_actor)"
    )
    return payload


def _deferred_pelvis_bind_align_if_needed(comp) -> dict[str, Any] | None:
    """Spawn-time align often sees zero bone locations; retry in ref pose on early canonical ticks.

    PoseableMesh bone queries may not be valid until after the component has ticked at least once.
    """
    global _PELVIS_DEFERRED_ALIGN_COMPLETED, _PELVIS_DEFERRED_ALIGN_ATTEMPTS
    if _PELVIS_DEFERRED_ALIGN_COMPLETED:
        return None
    if not _amongus_truthy_env("AMONGUS_UE_DEFER_PELVIS_BIND_ALIGN", default=True):
        _PELVIS_DEFERRED_ALIGN_COMPLETED = True
        return None
    max_att = 48
    raw_max = str(os.environ.get("AMONGUS_UE_PELVIS_DEFER_ALIGN_MAX_ATTEMPTS", "") or "").strip()
    if raw_max:
        try:
            max_att = max(1, min(256, int(raw_max)))
        except ValueError:
            pass
    if _PELVIS_DEFERRED_ALIGN_ATTEMPTS >= max_att:
        _PELVIS_DEFERRED_ALIGN_COMPLETED = True
        unreal.log_warning(
            "UE_SCENE: deferred pelvis bind-align stopped after "
            f"{max_att} attempts (bone location still unavailable or near zero)."
        )
        return None
    _PELVIS_DEFERRED_ALIGN_ATTEMPTS += 1

    reset_pose = getattr(comp, "reset_all_bone_transforms", None)
    if callable(reset_pose):
        try:
            reset_pose()
        except Exception:
            pass
    refresh = getattr(comp, "refresh_bone_transforms", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass

    mesh_bones = _poseable_mesh_bone_names_for_alignment(comp)
    bone_pick = _resolve_smpl_root_alignment_bone_name(mesh_bones)
    if not bone_pick:
        return None
    ci_map = _bone_name_ci_map(mesh_bones)
    mesh_name = str(ci_map.get(str(bone_pick).lower(), str(bone_pick)))
    get_loc = getattr(comp, "get_bone_location_by_name", None)
    bone_space = _component_space_bone_space()
    bx = by = bz = 0.0
    align_source = "deferred_unknown"
    if callable(get_loc):
        try:
            loc = get_loc(_bone_name_arg(mesh_name), bone_space)
            bx, by, bz = float(loc.x), float(loc.y), float(loc.z)
            align_source = "deferred_component_bone_location"
        except Exception:
            align_source = "deferred_component_exc"
    if max(abs(bx), abs(by), abs(bz)) < 1.0e-5:
        acc = _accumulated_ref_pose_bone_translation_cm(comp, mesh_name)
        if acc is not None and max(abs(float(v)) for v in acc) > 1.0e-5:
            bx, by, bz = float(acc[0]), float(acc[1]), float(acc[2])
            align_source = "deferred_ref_skeleton_chain_accum"

    mag = max(abs(bx), abs(by), abs(bz))
    if mag < 1.0e-5:
        return None

    try:
        cur_l = _scene_component_relative_translation_cm(comp)
        if cur_l is None:
            nx, ny, nz = -float(bx), -float(by), -float(bz)
        else:
            nx, ny, nz = float(cur_l[0]) - bx, float(cur_l[1]) - by, float(cur_l[2]) - bz
        comp.set_relative_location(unreal.Vector(nx, ny, nz), False, False)
    except Exception as exc:
        unreal.log_warning(f"UE_SCENE: deferred pelvis bind-align set_relative_location failed: {exc!r}")
        return None

    refresh2 = getattr(comp, "refresh_bone_transforms", None)
    if callable(refresh2):
        try:
            refresh2()
        except Exception:
            pass

    own_fn = getattr(comp, "get_owner", None)
    owner_actor = own_fn() if callable(own_fn) else None
    mirrored_tpl = _mirror_poseable_relative_transform_to_editor_template(owner_actor, comp)

    _PELVIS_DEFERRED_ALIGN_COMPLETED = True
    payload: dict[str, Any] = {
        "applied": True,
        "bone": mesh_name,
        "align_source": align_source,
        "attempt": int(_PELVIS_DEFERRED_ALIGN_ATTEMPTS),
        "adjustment_cm": [-float(bx), -float(by), -float(bz)],
        "bind_pelvis_component_cm_measured": [float(bx), float(by), float(bz)],
        "deferred_on_canonical_tick": True,
        "editor_template_mirrored": bool(mirrored_tpl),
    }
    _write_visible_human_pelvis_align_report(payload)
    unreal.log(
        f"UE_SCENE: deferred pelvis bind-align attempt={_PELVIS_DEFERRED_ALIGN_ATTEMPTS} "
        f"bone={mesh_name!r} source={align_source} adjustment_cm=[{-bx:.4f}, {-by:.4f}, {-bz:.4f}]"
    )
    return payload


def _apply_human_canonical_tick(canonical_state: dict[str, Any]) -> dict[str, Any]:
    human_raw = canonical_state.get("human")
    human = human_raw if isinstance(human_raw, dict) else {}
    if not human:
        return {"updated": False, "reason": "empty_human_payload"}

    pelvis_pie_world_reset = _reset_pelvis_align_state_if_pie_world_changed()

    label = str(os.environ.get("AMONGUS_UE_HUMAN_ACTOR_LABEL", VISIBLE_HUMAN_LABEL) or "").strip()
    actor = find_world_actor_by_label(label)
    if actor is None:
        return {"updated": False, "reason": "actor_not_found", "label": label}

    detail: dict[str, Any] = {"updated": True, "label": label, "world_kind": world_kind_for_canonical_tick()}
    if pelvis_pie_world_reset:
        detail["pelvis_align_pie_world_reset"] = True

    quat = human.get("root_quat_xyzw_genesis")
    root_mat_g: np.ndarray | None = None
    if isinstance(quat, (list, tuple)) and len(quat) >= 4:
        root_mat_g = quaternion_xyzw_to_matrix([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])

    drive_bones = (
        _amongus_truthy_env("AMONGUS_UE_DRIVE_HUMAN_BONES", default=True)
        if _HUMAN_UE_DRIVE_HUMAN_BONES is None
        else bool(_HUMAN_UE_DRIVE_HUMAN_BONES)
    )
    body_pose_raw = human.get("smpl_body_pose_axis_angle")
    fold_root_into_skeleton = bool(
        _amongus_truthy_env("AMONGUS_UE_FOLD_ROOT_INTO_SKELETON", default=True)
        and drive_bones
        and root_mat_g is not None
        and isinstance(body_pose_raw, list)
        and len(body_pose_raw) > 0
    )
    detail["root_rotation_drive"] = "skeleton_pelvis_fk" if fold_root_into_skeleton else "actor"

    root_m = human.get("root_translation_world_m")
    if isinstance(root_m, (list, tuple)) and len(root_m) >= 3:
        loc_cm = tuple(
            float(v) * 100.0
            for v in ue_world_point_from_genesis_m(
                [float(root_m[0]), float(root_m[1]), float(root_m[2])],
            ).tolist()
        )
        actor.set_actor_location(
            unreal.Vector(float(loc_cm[0]), float(loc_cm[1]), float(loc_cm[2])), False, True
        )
        detail["location_cm"] = [float(loc_cm[0]), float(loc_cm[1]), float(loc_cm[2])]

    if fold_root_into_skeleton:
        actor.set_actor_rotation(unreal.Rotator(0.0, 0.0, 0.0), True)
        detail["human_root_on"] = "skeleton_pelvis_fk"
        ue_rm = ue_world_rotation_from_genesis(root_mat_g.reshape(3, 3))
        rdeg = ue_rotator_deg_from_matrix(np.asarray(ue_rm, dtype=np.float64))
        detail["rotation_deg"] = [float(rdeg[0]), float(rdeg[1]), float(rdeg[2])]
    elif isinstance(quat, (list, tuple)) and len(quat) >= 4:
        uq = ue_world_quat_xyzw_from_genesis([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])
        if uq is not None:
            rot_mat = ue_rotation_matrix_from_quat_xyzw(uq.tolist())
            rot_deg = ue_rotator_deg_from_matrix(np.asarray(rot_mat, dtype=np.float64))
            actor.set_actor_rotation(
                unreal.Rotator(float(rot_deg[0]), float(rot_deg[1]), float(rot_deg[2])),
                True,
            )
            detail["rotation_deg"] = [float(rot_deg[0]), float(rot_deg[1]), float(rot_deg[2])]
            detail["human_root_on"] = "actor"

    if drive_bones:
        if not isinstance(body_pose_raw, list) or not body_pose_raw:
            detail["body_pose_warning"] = "missing_smpl_body_pose_axis_angle"
        else:
            comp = None
            try:
                comp = actor.get_component_by_class(unreal.PoseableMeshComponent)
            except Exception:
                comp = None
            if comp is None:
                comp = _ensure_pie_human_poseable(actor)
                if comp is not None:
                    detail["pie_poseable_attached"] = True
            if comp is None:
                detail["body_pose_warning"] = "no_poseable_mesh_component"
            else:
                if _ensure_visible_human_bone_mapping_for_component(comp):
                    detail["smpl_bone_preset_recovered"] = True
                defer_plv = _deferred_pelvis_bind_align_if_needed(comp)
                if defer_plv:
                    detail["pelvis_deferred_bind_align"] = defer_plv
                root_for_pose = root_mat_g if fold_root_into_skeleton else None
                applied, missing = _apply_smpl_body_pose_to_component(comp, body_pose_raw, smpl_root_rot_genesis=root_for_pose)
                detail["human_pose_source"] = "smpl_body_pose_axis_angle"
                detail["motion_frame_index"] = int(human.get("motion_frame_index", 0) or 0)
                detail["body_pose_bones_applied"] = int(applied)
                if _VISIBLE_HUMAN_BONE_PRESET_NAME:
                    detail["smpl_bone_preset"] = _VISIBLE_HUMAN_BONE_PRESET_NAME
                if missing:
                    detail["body_pose_missing_bones"] = list(missing[:8])
                detail["poseable_alignment"] = _ue_poseable_alignment_summary(actor, comp)
                sock_align = _apply_pelvis_socket_world_align_once(
                    actor, comp, detail["poseable_alignment"]
                )
                if sock_align:
                    detail["pelvis_socket_world_align"] = sock_align
                    detail["poseable_alignment"] = _ue_poseable_alignment_summary(actor, comp)
    else:
        detail["body_pose_skipped"] = "AMONGUS_UE_DRIVE_HUMAN_BONES_disabled"

    # region agent log
    if _amongus_truthy_env("AMONGUS_DEBUG_NDJSON", default=False):
        try:
            _tn = int(getattr(_apply_human_canonical_tick, "_ue_tick_dbg_n", 0)) + 1
            setattr(_apply_human_canonical_tick, "_ue_tick_dbg_n", _tn)
            _tev = _amongus_debug_ndjson_every_ue(default=22)
            if _tn <= 12 or _tn % _tev == 0:
                pcomp = None
                try:
                    pcomp = actor.get_component_by_class(unreal.PoseableMeshComponent)
                except Exception:
                    pcomp = None
                thin = {k: detail.get(k) for k in (
                    "root_rotation_drive",
                    "human_root_on",
                    "location_cm",
                    "rotation_deg",
                    "motion_frame_index",
                    "body_pose_bones_applied",
                    "body_pose_warning",
                    "smpl_bone_preset",
                    "body_pose_missing_bones",
                    "pie_poseable_attached",
                ) if k in detail}
                root_raw = human.get("root_translation_world_m")
                _ue_debug_ndjson(
                    hypothesis_id="UE_HUMAN_TICK",
                    location="ue_common_scene_loader.py:_apply_human_canonical_tick",
                    message="UE actor + poseable diagnostics after canonical human tick",
                    data={
                        "tick_n": int(_tn),
                        "canonical_sim_step": canonical_state.get("sim_step_index"),
                        "canonical_frame_index": canonical_state.get("frame_index"),
                        "payload_root_translation_m": (
                            [float(root_raw[i]) for i in range(3)]
                            if isinstance(root_raw, (list, tuple)) and len(root_raw) >= 3
                            else None
                        ),
                        "fold_root_into_skeleton": bool(fold_root_into_skeleton),
                        "detail_subset": thin,
                        **_ue_poseable_world_diag(actor, pcomp),
                    },
                )
        except Exception:
            pass
    # endregion agent log

    return detail


def _iter_amongus_capture_components():
    for actor in _iter_world_actors():
        try:
            components = actor.get_components_by_class(unreal.ActorComponent)
        except Exception:
            continue
        for component in components:
            try:
                class_name = str(component.get_class().get_name())
            except Exception:
                class_name = type(component).__name__
            if "AmongUsTcpCaptureComponent" in class_name:
                yield component


def _apply_camera_clock_canonical_tick(canonical_state: dict[str, Any]) -> dict[str, Any]:
    frame_raw = canonical_state.get("frame_index", canonical_state.get("sim_step_index"))
    sim_raw = canonical_state.get("sim_time_ns")
    if frame_raw is None or sim_raw is None:
        return {"updated": 0, "reason": "missing_frame_or_sim_time"}

    extras = canonical_state.get("extras")
    session_id = ""
    if isinstance(extras, dict):
        session_id = str(extras.get("session_id") or "")

    detail: dict[str, Any] = {"updated": 0, "errors": []}
    for component in _iter_amongus_capture_components():
        try:
            if session_id and hasattr(component, "set_session_id"):
                component.set_session_id(session_id)
            if hasattr(component, "set_external_sim_clock"):
                component.set_external_sim_clock(int(frame_raw), int(sim_raw))
                detail["updated"] += 1
            else:
                detail["errors"].append("component_missing_set_external_sim_clock")
        except Exception as exc:
            detail["errors"].append(repr(exc))
    return detail


def _apply_dynamic_objects_canonical_tick(canonical_state: dict[str, Any]) -> dict[str, Any]:
    """Keep canonical dynamic-object diagnostics lightweight.

    Rendering/updating high-bandwidth objects belongs in a dedicated adapter; this path must stay
    cheap because it runs on every canonical tick.
    """
    objects = canonical_state.get("objects") or {}
    if not isinstance(objects, dict):
        return {"count": 0, "active": 0, "expired": 0, "reason": "objects_not_mapping"}
    now_ns = int(canonical_state.get("sim_time_ns") or canonical_state.get("wall_time_ns") or 0)
    active = 0
    expired = 0
    skipped = 0
    kinds: dict[str, int] = {}
    for _entity_id, raw in objects.items():
        if not isinstance(raw, dict):
            skipped += 1
            continue
        payload = dict(raw.get("payload") or {})
        expires_at_ns = int(payload.get("expires_at_ns") or 0)
        ttl_ns = int(payload.get("ttl_ns") or 0)
        time_ns = int(payload.get("time_ns") or 0)
        if expires_at_ns <= 0 and ttl_ns > 0 and time_ns > 0:
            expires_at_ns = time_ns + ttl_ns
        if expires_at_ns > 0 and now_ns > 0 and now_ns > expires_at_ns:
            expired += 1
            continue
        kind = str(raw.get("entity_type") or payload.get("entity_type") or "").strip().lower()
        kinds[kind or "unknown"] = int(kinds.get(kind or "unknown", 0)) + 1
        active += 1
    return {"count": len(objects), "active": active, "expired": expired, "skipped": skipped, "kinds": kinds}


def apply_canonical_scene_tick(canonical_state: dict[str, Any]) -> dict[str, Any]:
    """Incremental UE sync from CanonicalSceneStateV1-shaped dict (Genesis canonical joints -> UE articulation)."""
    import ue_urdf_visual_loader as urdf_art

    detail: dict[str, Any] = {"robot_updates": [], "world": world_diagnostic_for_canonical_tick()}
    detail["camera_clock"] = _apply_camera_clock_canonical_tick(canonical_state)
    robots = canonical_state.get("robot_entities") or {}
    articulation_id = str(os.environ.get("AMONGUS_REGISTER_URDF_ARTICULATION_ID", "") or "").strip()
    detail["registered_robot_ids"] = urdf_art.registered_robot_ids()
    if articulation_id and isinstance(robots, dict) and len(robots) > 1:
        detail["robot_id_warning"] = (
            "AMONGUS_REGISTER_URDF_ARTICULATION_ID overrides all canonical robot names; "
            "unset it for multi-robot routing."
        )
    if not isinstance(robots, dict):
        return {**detail, "warning": "robot_entities_not_dict"}
    for name, entity in robots.items():
        if not isinstance(entity, dict) or "joint_positions" not in entity:
            continue
        q = [float(v) for v in entity["joint_positions"]]
        rid = articulation_id or str(name)
        try:
            applied = urdf_art.apply_articulated_robot_joints(rid, q)
            merged = dict(applied) if isinstance(applied, dict) else {"raw": applied}
            detail["robot_updates"].append({"robot_id": rid, "dof": len(q), "result": merged})
        except Exception as exc:
            detail["robot_updates"].append({"robot_id": rid, "dof": len(q), "error": repr(exc)})
    detail["human_keys"] = list((canonical_state.get("human") or {}).keys()) if isinstance(canonical_state.get("human"), dict) else []
    detail["object_keys"] = list((canonical_state.get("objects") or {}).keys()) if isinstance(canonical_state.get("objects"), dict) else []
    detail["dynamic_objects"] = _apply_dynamic_objects_canonical_tick(canonical_state)
    detail["human"] = _apply_human_canonical_tick(canonical_state)
    _request_editor_viewport_redraw()
    return detail


VISIBLE_HUMAN_LABEL = "GEN_visible_human"
VISIBLE_HUMAN_FOLDER = f"{GENERATED_SCENE_FOLDER}/human"


def _attach_poseable_human_component(
    *,
    actor,
    skeletal_mesh,
    relative_scale: float,
    hide_underlying_skeletal: bool,
):
    """Attach a PoseableMeshComponent to ``actor`` (idempotent). Works in both editor and PIE worlds.

    After assigning mesh and scale, applies bind-pose alignment so the SMPL root (pelvis) bone
    coincides with the actor's root transform pivot — required for correct ``root_translation`` /
    ``root_quat`` application without per-asset numeric calibration.

    Returns the (existing or newly created) ``PoseableMeshComponent`` or ``None`` if attach failed.
    Tries multiple UE Python entry points (``add_component_by_class`` then UClass factory),
    attaches to the actor's RootComponent, registers as an instance component so PIE has a chance
    to inherit it, then registers and configures the mesh / scale / visibility.
    """
    if actor is None:
        return None
    poseable_cls = getattr(unreal, "PoseableMeshComponent", None)
    if poseable_cls is None:
        unreal.log_warning("UE_SCENE: unreal.PoseableMeshComponent not available in this UE build.")
        return None

    component = None
    try:
        component = actor.get_component_by_class(poseable_cls)
    except Exception:
        component = None

    last_exc: Exception | None = None
    if component is None:
        if hasattr(actor, "add_component_by_class"):
            try:
                cand = actor.add_component_by_class(
                    poseable_cls, False, unreal.Transform(), False
                )
                if cand is not None:
                    component = cand
            except Exception as exc:
                last_exc = exc
                unreal.log_warning(
                    f"UE_SCENE: PoseableMeshComponent add_component_by_class failed: {exc!r}"
                )
    if component is None:
        try:
            cand = poseable_cls(actor)
            if cand is not None:
                component = cand
        except Exception as exc:
            last_exc = exc
            unreal.log_warning(
                f"UE_SCENE: PoseableMeshComponent uclass factory failed: {exc!r}"
            )
    if component is None:
        unreal.log_warning(
            f"UE_SCENE: PoseableMeshComponent attach failed (last_error={last_exc!r}); "
            f"falling back to SkeletalMeshComponent.set_bone_transform_by_name path."
        )
        return None

    root = None
    try:
        root = actor.root_component
    except Exception:
        try:
            root = actor.get_editor_property("root_component")
        except Exception:
            root = None
    if root is not None and hasattr(component, "attach_to_component"):
        try:
            keep_world = getattr(unreal, "AttachmentRule", None)
            keep_relative = getattr(keep_world, "KEEP_RELATIVE", None) if keep_world is not None else None
            snap = getattr(keep_world, "SNAP_TO_TARGET", None) if keep_world is not None else None
            location_rule = snap if snap is not None else (keep_relative if keep_relative is not None else 0)
            component.attach_to_component(
                root,
                "",
                location_rule,
                location_rule,
                location_rule,
                False,
            )
        except Exception:
            pass

    add_instance = getattr(actor, "add_instance_component", None)
    if add_instance is not None:
        try:
            add_instance(component)
        except Exception:
            pass

    if hasattr(component, "register_component"):
        try:
            component.register_component()
        except Exception:
            pass

    if skeletal_mesh is not None:
        ok = False
        for setter in ("set_skeletal_mesh", "set_skeletal_mesh_asset"):
            fn = getattr(component, setter, None)
            if fn is None:
                continue
            try:
                fn(skeletal_mesh)
                ok = True
                break
            except Exception:
                continue
        if not ok:
            unreal.log_warning("UE_SCENE: failed to assign skeletal mesh to PoseableMeshComponent.")

    mobility_enum = getattr(unreal, "ComponentMobility", None)
    movable = getattr(mobility_enum, "MOVABLE", None) if mobility_enum is not None else None
    if movable is not None and hasattr(component, "set_mobility"):
        try:
            component.set_mobility(movable)
        except Exception:
            pass

    if hasattr(component, "set_relative_rotation"):
        try:
            component.set_relative_rotation(unreal.Rotator(0.0, 0.0, 0.0), False)
        except Exception:
            pass
    if hasattr(component, "set_relative_scale3d"):
        try:
            component.set_relative_scale3d(
                unreal.Vector(float(relative_scale), float(relative_scale), float(relative_scale))
            )
        except Exception:
            pass

    align_info = _align_poseable_mesh_smpl_root_to_parent_origin(component)

    # region agent log
    if _amongus_truthy_env("AMONGUS_DEBUG_NDJSON", default=False):
        try:
            _ue_debug_ndjson(
                hypothesis_id="UE_ATTACH_POSEABLE",
                location="ue_common_scene_loader.py:_attach_poseable_human_component",
                message="PoseableMesh attached; post-bind-align summary",
                data={
                    "relative_scale": float(relative_scale),
                    "hide_underlying_skeletal": bool(hide_underlying_skeletal),
                    "align_info_summary": {k: align_info.get(k) for k in (
                        "applied",
                        "bone",
                        "bone_primary_resolved",
                        "relative_shift_cm",
                        "bind_pelvis_component_cm",
                        "reason",
                    ) if k in align_info},
                    "mesh_asset_set": skeletal_mesh is not None,
                },
            )
        except Exception:
            pass
    # endregion agent log

    if hide_underlying_skeletal:
        skeletal_component = getattr(actor, "skeletal_mesh_component", None)
        if skeletal_component is None:
            try:
                skeletal_component = actor.get_component_by_class(unreal.SkeletalMeshComponent)
            except Exception:
                skeletal_component = None
        if (
            skeletal_component is not None
            and skeletal_component is not component
            and align_info.get("applied")
            and isinstance(align_info.get("relative_shift_cm"), list)
            and len(align_info["relative_shift_cm"]) == 3
        ):
            sx, sy, sz = (float(align_info["relative_shift_cm"][i]) for i in range(3))
            if hasattr(skeletal_component, "set_relative_location"):
                try:
                    skeletal_component.set_relative_location(unreal.Vector(sx, sy, sz), False, False)
                except Exception:
                    pass

    for setter, args in (
        ("set_visibility", (True, True)),
        ("set_hidden_in_game", (False, True)),
    ):
        fn = getattr(component, setter, None)
        if fn is None:
            continue
        try:
            fn(*args)
        except Exception:
            continue

    if hide_underlying_skeletal:
        skeletal_component = getattr(actor, "skeletal_mesh_component", None)
        if skeletal_component is None:
            try:
                skeletal_component = actor.get_component_by_class(unreal.SkeletalMeshComponent)
            except Exception:
                skeletal_component = None
        if skeletal_component is not None and skeletal_component is not component:
            for setter, args in (
                ("set_visibility", (False, True)),
                ("set_hidden_in_game", (True, True)),
            ):
                fn = getattr(skeletal_component, setter, None)
                if fn is None:
                    continue
                try:
                    fn(*args)
                except Exception:
                    continue
    return component


def _ensure_pie_human_poseable(actor):
    """PIE-side fallback: re-attach PoseableMeshComponent if PIE duplicate dropped the editor-side one."""
    if actor is None:
        return None
    skeletal_mesh = None
    if _VISIBLE_HUMAN_SKELETAL_MESH_PATH:
        try:
            skeletal_mesh = unreal.load_asset(_VISIBLE_HUMAN_SKELETAL_MESH_PATH)
        except Exception:
            skeletal_mesh = None
    return _attach_poseable_human_component(
        actor=actor,
        skeletal_mesh=skeletal_mesh,
        relative_scale=float(_VISIBLE_HUMAN_RELATIVE_SCALE or 1.0),
        hide_underlying_skeletal=True,
    )


def spawn_genesis_driven_visible_human(
    *,
    scene_spec,
    anchor_m: tuple[float, float, float],
) -> dict[str, Any]:
    """Spawn GEN_visible_human for Genesis-driven root + official AnimSequence frame sync.

    Always spawns a native ``SkeletalMeshActor`` so PIE reliably duplicates the mesh component.
    """
    global _VISIBLE_HUMAN_SMPL_BONE_NAMES, _VISIBLE_HUMAN_BONE_PRESET_NAME
    global _VISIBLE_HUMAN_SKELETAL_MESH_PATH, _VISIBLE_HUMAN_RELATIVE_SCALE
    global _PELVIS_DEFERRED_ALIGN_COMPLETED, _PELVIS_DEFERRED_ALIGN_ATTEMPTS
    global _PELVIS_SOCKET_WORLD_ALIGN_DONE, _PELVIS_ALIGN_TRACKED_WORLD

    _PELVIS_DEFERRED_ALIGN_COMPLETED = False
    _PELVIS_DEFERRED_ALIGN_ATTEMPTS = 0
    _PELVIS_SOCKET_WORLD_ALIGN_DONE = False
    _PELVIS_ALIGN_TRACKED_WORLD = None

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    uv_cal: dict[str, Any] = {}
    if _HUMAN_UE_CALIBRATION:
        raw_uv = _HUMAN_UE_CALIBRATION.get("ue_visible_human")
        if isinstance(raw_uv, dict):
            uv_cal = raw_uv
    skeletal_mesh_path = str(getattr(scene_spec.ue_avatar, "skeletal_mesh_path", "") or "").strip()
    sk_cal = str(uv_cal.get("skeletal_mesh_path") or "").strip()
    if sk_cal:
        skeletal_mesh_path = sk_cal
    skeletal_mesh = unreal.load_asset(skeletal_mesh_path) if skeletal_mesh_path else None
    spawn_loc_cm = m_to_cm(anchor_m)
    drive_bones = (
        _HUMAN_UE_DRIVE_HUMAN_BONES
        if _HUMAN_UE_DRIVE_HUMAN_BONES is not None
        else _amongus_truthy_env("AMONGUS_UE_DRIVE_HUMAN_BONES", default=True)
    )
    relative_scale = float(getattr(scene_spec.ue_avatar, "fbx_global_scale", 100.0) or 100.0) / 100.0
    if uv_cal.get("relative_scale") is not None:
        try:
            relative_scale = float(uv_cal["relative_scale"])
        except (TypeError, ValueError):
            pass

    _VISIBLE_HUMAN_SKELETAL_MESH_PATH = skeletal_mesh_path
    _VISIBLE_HUMAN_RELATIVE_SCALE = float(relative_scale)

    actor = actor_subsystem.spawn_actor_from_class(
        unreal.SkeletalMeshActor,
        unreal.Vector(float(spawn_loc_cm[0]), float(spawn_loc_cm[1]), float(spawn_loc_cm[2])),
    )
    actor.set_actor_label(VISIBLE_HUMAN_LABEL)
    actor.set_folder_path(VISIBLE_HUMAN_FOLDER)

    skeletal_component = getattr(actor, "skeletal_mesh_component", None)
    if skeletal_component is None:
        try:
            skeletal_component = actor.get_component_by_class(unreal.SkeletalMeshComponent)
        except Exception:
            skeletal_component = None
    if skeletal_component is None:
        return {
            "spawned": True,
            "label": VISIBLE_HUMAN_LABEL,
            "skeletal_mesh_path": skeletal_mesh_path,
            "warning": "no_skeletal_mesh_component_on_actor",
        }

    if skeletal_mesh is not None:
        try:
            skeletal_component.set_skeletal_mesh_asset(skeletal_mesh)
        except Exception:
            try:
                skeletal_component.set_skeletal_mesh(skeletal_mesh, True)
            except Exception:
                pass

    animation_mode = getattr(unreal, "AnimationMode", None)
    custom_mode = getattr(animation_mode, "ANIMATION_CUSTOM_MODE", None) if animation_mode is not None else None
    if custom_mode is None and animation_mode is not None:
        custom_mode = getattr(animation_mode, "ANIMATION_SINGLE_NODE", None)
    if custom_mode is not None:
        try:
            skeletal_component.set_animation_mode(custom_mode)
        except Exception:
            pass
    try:
        if hasattr(skeletal_component, "stop"):
            skeletal_component.stop()
    except Exception:
        pass
    try:
        if hasattr(skeletal_component, "set_animation"):
            skeletal_component.set_animation(None)
    except Exception:
        pass
    if hasattr(skeletal_component, "set_relative_scale3d"):
        try:
            skeletal_component.set_relative_scale3d(
                unreal.Vector(relative_scale, relative_scale, relative_scale)
            )
        except Exception:
            pass

    _ensure_actor_movable(actor)

    component = skeletal_component
    human_component_kind = "skeletal_mesh_actor"
    if drive_bones:
        poseable_component = _attach_poseable_human_component(
            actor=actor,
            skeletal_mesh=skeletal_mesh,
            relative_scale=relative_scale,
            hide_underlying_skeletal=True,
        )
        if poseable_component is not None:
            component = poseable_component
            human_component_kind = "poseable_mesh_smpl_realtime"
        else:
            human_component_kind = "skeletal_mesh_actor_no_poseable"

    bone_list = collect_bone_names_from_skeletal_mesh_asset(skeletal_mesh)
    if not bone_list:
        bone_list = _collect_skeletal_mesh_bone_names(component)

    preset_name, preset_tuple = _pick_smpl_bone_preset(bone_list)
    hits = sum(1 for b in preset_tuple if b in set(bone_list) or b.lower() in {str(x).lower() for x in bone_list})
    forced = str(_CALIB_BONE_PRESET or "").strip()
    if forced and forced in SMPL_BODY_BONE_PRESETS:
        preset_name, preset_tuple = forced, SMPL_BODY_BONE_PRESETS[forced]
        hits = sum(
            1 for b in preset_tuple if b in set(bone_list) or b.lower() in {str(x).lower() for x in bone_list}
        )
    _VISIBLE_HUMAN_BONE_PRESET_NAME = preset_name
    _VISIBLE_HUMAN_SMPL_BONE_NAMES = preset_tuple
    unreal.log(
        f"UE_SCENE: visible_human SMPL bone preset={preset_name} mesh_bones={len(bone_list)} preset_hits={hits}/{len(preset_tuple)} "
        f"component={human_component_kind}"
    )

    session_root = str(os.environ.get("AMONGUS_SESSION_DIR", "") or "").strip()
    if session_root:
        try:
            dump_path = Path(session_root) / "visible_human_bones.json"
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(
                json.dumps(
                    {
                        "preset": preset_name,
                        "mesh_bone_name_count": len(bone_list),
                        "bone_names_mesh": bone_list[:2048],
                        "smpl_body_mapping": list(preset_tuple),
                        "human_component": human_component_kind,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            unreal.log_warning(f"UE_SCENE: visible_human bone dump failed: {exc!r}")

    # region agent log
    if _amongus_truthy_env("AMONGUS_DEBUG_NDJSON", default=False):
        try:
            _ue_debug_ndjson(
                hypothesis_id="UE_SPAWN_VISIBLE_HUMAN",
                location="ue_common_scene_loader.py:spawn_genesis_driven_visible_human",
                message="spawn anchor vs SMPL bone preset at genesis-driven human create",
                data={
                    "anchor_m": [float(anchor_m[i]) for i in range(3)],
                    "spawn_location_cm": [float(v) for v in spawn_loc_cm],
                    "drive_bones": bool(drive_bones),
                    "relative_scale": float(relative_scale),
                    "skeletal_mesh_path": skeletal_mesh_path,
                    "smpl_bone_preset": preset_name,
                    "mesh_bone_name_count": int(len(bone_list)),
                    "human_component": human_component_kind,
                },
            )
        except Exception:
            pass
    # endregion agent log

    return {
        "spawned": True,
        "label": VISIBLE_HUMAN_LABEL,
        "skeletal_mesh_path": skeletal_mesh_path,
        "skeletal_mesh_loaded": skeletal_mesh is not None,
        "spawn_location_cm": [float(v) for v in spawn_loc_cm],
        "animation_mode": "custom" if custom_mode is not None else "default",
        "smpl_bone_preset": preset_name,
        "mesh_bone_name_count": len(bone_list),
        "preset_hit_count": hits,
        "bone_dump_path": str(Path(session_root) / "visible_human_bones.json") if session_root else "",
        "human_component": human_component_kind,
    }


def apply_level_editor_viewport_camera_speed_scale() -> dict[str, object]:
    """Multiply ``LevelEditorViewportSettings.camera_speed_scalar`` (WASD / flight camera).

    Controlled by ``AMONGUS_UE_EDITOR_CAMERA_SPEED_SCALE`` (default ``0.5``). Set to ``1`` to skip.
    """
    global _LEVEL_EDITOR_CAMERA_SPEED_SCALED
    if _LEVEL_EDITOR_CAMERA_SPEED_SCALED:
        return {"applied": False, "reason": "already_applied"}
    raw = str(os.environ.get("AMONGUS_UE_EDITOR_CAMERA_SPEED_SCALE", "0.5") or "").strip()
    try:
        mult = float(raw)
    except ValueError:
        return {"applied": False, "reason": "bad_env"}
    if abs(mult - 1.0) < 1e-6:
        return {"applied": False, "reason": "unity"}
    cls = getattr(unreal, "LevelEditorViewportSettings", None)
    if cls is None:
        unreal.log_warning("UE_SCENE: unreal.LevelEditorViewportSettings not available; camera speed unchanged.")
        return {"applied": False, "reason": "no_level_editor_viewport_settings"}
    try:
        settings = unreal.get_default_object(cls)
        cur = 1.0
        if hasattr(settings, "get_editor_property"):
            try:
                cur = float(settings.get_editor_property("camera_speed_scalar"))
            except Exception:
                cur = 1.0
        new_v = max(1.0e-4, cur * mult)
        settings.set_editor_property("camera_speed_scalar", new_v)
        unreal.log(
            f"UE_SCENE: LevelEditorViewportSettings.camera_speed_scalar {cur:.6g} -> {new_v:.6g} (x{mult:g})"
        )
        _LEVEL_EDITOR_CAMERA_SPEED_SCALED = True
        return {"applied": True, "before": cur, "after": new_v, "multiplier": mult}
    except Exception as exc:
        unreal.log_warning(f"UE_SCENE: camera_speed_scalar update failed: {exc!r}")
        return {"applied": False, "error": repr(exc)}


def _request_editor_viewport_redraw() -> None:
    """Force editor viewport redraw so canonical actor pose changes show up in non-PIE level view.

    No-op while PIE is running; PIE drives its own viewport ticks and editor-side redraw
    requests would otherwise spam ``Slate.RefreshViewport: editor is in play mode`` errors.
    """
    if _query_pie_world() is not None:
        return
    try:
        subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    except Exception:
        subsystem = None
    if subsystem is not None and hasattr(subsystem, "redraw_all_viewports"):
        try:
            subsystem.redraw_all_viewports()
            return
        except Exception:
            pass
    legacy = getattr(unreal, "EditorLevelLibrary", None)
    if legacy is not None:
        for name in ("redraw_all_viewports", "set_actor_selection_state"):
            fn = getattr(legacy, name, None)
            if fn is None:
                continue
            try:
                fn()
                return
            except Exception:
                continue
    try:
        unreal.SystemLibrary.execute_console_command(unreal.EditorLevelLibrary.get_editor_world(), "Slate.RefreshViewport")
    except Exception:
        pass


def camera_actor_pose_payload(camera_spec) -> dict:
    return ue_camera_payload_from_spec(camera_spec)


def _hide_actor_in_game_and_editor(actor) -> bool:
    """Hide a level actor during PIE and in the editor viewport."""
    changed = False
    for method_name in ("set_actor_hidden_in_game", "set_is_temporarily_hidden_in_editor"):
        method = getattr(actor, method_name, None)
        if method is None:
            continue
        try:
            method(True)
            changed = True
        except Exception:
            continue
    for comp in list(getattr(actor, "get_components_by_class", lambda *_a, **_k: [])() or []):
        for method_name in ("set_hidden_in_game", "set_visibility"):
            method = getattr(comp, method_name, None)
            if method is None:
                continue
            try:
                if method_name == "set_visibility":
                    method(False, True)
                else:
                    method(True)
                changed = True
            except Exception:
                continue
    return changed


def hide_bedlam_freeview_camera_rig(*, log: bool = True) -> dict[str, Any]:
    """Hide IBLMap BEDLAM free-view helpers (BE_CameraTarget sphere, BE_CameraOperator).

    Controlled by ``AMONGUS_UE_HIDE_BEDLAM_CAMERA_RIG`` (default on). Does not affect
    AmongUs SceneCapture2D rigs under GeneratedScene/amongus_capture.
    """
    if not _amongus_truthy_env("AMONGUS_UE_HIDE_BEDLAM_CAMERA_RIG", default=True):
        return {"hidden": [], "skipped": True, "reason": "env_disabled"}

    hidden: list[str] = []
    for actor in _all_level_actors():
        label = str(actor.get_actor_label() or "")
        class_name = ""
        try:
            class_name = str(actor.get_class().get_name() or "")
        except Exception:
            pass
        is_target = "BE_CameraTarget" in label
        is_operator = class_name == "BE_CameraOperator_C"
        if not (is_target or is_operator):
            continue
        if _hide_actor_in_game_and_editor(actor):
            hidden.append(label or class_name)
    summary = {"hidden": hidden, "skipped": False}
    if log and hidden:
        unreal.log(f"UE_SCENE: hid BEDLAM free-view camera rig actors: {hidden}")
    return summary


AMONGUS_CAPTURE_RIG_LABEL = "GEN_AmongUsCaptureRig"
AMONGUS_CAPTURE_FOLDER = f"{GENERATED_SCENE_FOLDER}/amongus_capture"
AMONGUS_CAPTURE_HOST_DEFAULT = "127.0.0.1"
AMONGUS_CAPTURE_PORT_DEFAULT = 17355
AMONGUS_CAPTURE_JPEG_QUALITY_DEFAULT = 85


def _amongus_capture_component_class():
    """Return the AmongUs TCP capture UClass when its plugin is loaded, else None."""
    try:
        cls = getattr(unreal, "AmongUsTcpCaptureComponent", None)
    except Exception:
        cls = None
    return cls


def _set_bool_array_property(target_object, candidates: list[str], values: list[bool]) -> str | None:
    """Set a TArray<bool> editor property; UE Python can silently mis-bind plain Python lists."""
    bools = [bool(v) for v in values]
    for prop in candidates:
        try:
            target_object.set_editor_property(prop, bools)
            return prop
        except Exception:
            pass
        try:
            import unreal as _unreal

            arr = _unreal.Array(bool)
            for item in bools:
                arr.append(bool(item))
            target_object.set_editor_property(prop, arr)
            return prop
        except Exception:
            continue
    return None


def _set_property_if_available(target_object, candidates: list[str], value) -> str | None:
    """Set the first available editor property on ``target_object``; return the property name used."""
    for prop in candidates:
        try:
            target_object.set_editor_property(prop, value)
        except Exception:
            continue
        return prop
    return None


def _spawn_scene_capture_for_amongus(
    *,
    camera_spec,
    folder: str,
) -> tuple[object, object, object]:
    """Spawn a SceneCapture2D actor, ensure a transient TextureRenderTarget2D, return (actor, capture_component, render_target)."""
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = actor_subsystem.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0.0, 0.0, 0.0))
    actor.set_actor_label(f"{GENERATED_SCENE_LABEL_PREFIX}cap_{camera_spec.name}")
    actor.set_folder_path(folder)

    payload = ue_camera_payload_from_spec(camera_spec)
    actor.set_actor_location(unreal.Vector(float(payload["x"]), float(payload["y"]), float(payload["z"])), False, False)
    actor.set_actor_rotation(unreal.Rotator(float(payload["roll"]), float(payload["pitch"]), float(payload["yaw"])), False)
    right_g, true_up_g, _ = lookat_frame(
        camera_spec.pos,
        camera_spec.lookat,
        camera_spec.up,
        roll_deg=float(getattr(camera_spec, "roll_deg", 0.0) or 0.0),
    )
    expected_right = ue_world_point_from_genesis_m(right_g)
    expected_up = ue_world_point_from_genesis_m(true_up_g)
    try:
        right_vec = actor.get_actor_right_vector()
        up_vec = actor.get_actor_up_vector()
        actor_right = np.asarray([float(right_vec.x), float(right_vec.y), float(right_vec.z)], dtype=np.float64)
        actor_up = np.asarray([float(up_vec.x), float(up_vec.y), float(up_vec.z)], dtype=np.float64)
        right_dot = float(np.dot(expected_right, actor_right))
        up_dot = float(np.dot(expected_up, actor_up))
    except Exception:
        right_dot = 1.0
        up_dot = 1.0
    if right_dot < -0.95 and up_dot < -0.95:
        payload["roll"] = float(payload["roll"]) + 180.0
        actor.set_actor_rotation(unreal.Rotator(float(payload["roll"]), float(payload["pitch"]), float(payload["yaw"])), False)
    # #region agent log
    if str(getattr(camera_spec, "name", "")) == "cam_top":
        try:
            import json as _json
            import time as _time

            _log_path = "/home/camp/.cursor/debug-logs/debug-05706c.log"
            _payload = {
                "sessionId": "05706c",
                "runId": "pre-fix",
                "hypothesisId": "D",
                "location": "ue_common_scene_loader.py:_spawn_scene_capture_for_amongus",
                "message": "cam_top SceneCapture actor spawned in UE",
                "data": {
                    "payload_rotator_deg": [float(payload["roll"]), float(payload["pitch"]), float(payload["yaw"])],
                    "payload_location_cm": [float(payload["x"]), float(payload["y"]), float(payload["z"])],
                    "basis_dot_right_up": [float(right_dot), float(up_dot)],
                    "actor_label": str(actor.get_actor_label()),
                },
                "timestamp": int(_time.time() * 1000),
            }
            with open(_log_path, "a", encoding="utf-8") as _fh:
                _fh.write(_json.dumps(_payload, ensure_ascii=True) + "\n")
        except Exception:
            pass
    # #endregion

    capture = getattr(actor, "capture_component2d", None)
    if capture is None:
        capture = actor.get_component_by_class(unreal.SceneCaptureComponent2D)
    if capture is None:
        raise RuntimeError(f"SceneCapture2D actor {actor.get_actor_label()} missing SceneCaptureComponent2D")

    render_target = unreal.TextureRenderTarget2D()
    width = int(camera_spec.res[0])
    height = int(camera_spec.res[1])
    if hasattr(render_target, "init_auto_format"):
        render_target.init_auto_format(width, height)
    else:
        render_target.set_editor_property("size_x", width)
        render_target.set_editor_property("size_y", height)
    if hasattr(render_target, "update_resource"):
        try:
            render_target.update_resource()
        except Exception:
            pass

    capture.set_editor_property("texture_target", render_target)
    capture.set_editor_property("fov_angle", float(camera_spec.fov))
    capture.set_editor_property("capture_every_frame", True)
    capture.set_editor_property("capture_on_movement", True)

    return actor, capture, render_target


def _instantiate_actor_component(component_class, outer_actor):
    """Create + register an actor component on ``outer_actor`` using whatever API UE 5.3 Python exposes."""
    last_exc: Exception | None = None
    if hasattr(outer_actor, "add_component_by_class"):
        try:
            comp = outer_actor.add_component_by_class(component_class, False, unreal.Transform(), False)
            if comp is not None:
                return comp, "add_component_by_class"
        except Exception as exc:
            last_exc = exc
    try:
        comp = component_class(outer_actor)
        if hasattr(comp, "register_component"):
            try:
                comp.register_component()
            except Exception:
                pass
        return comp, "uclass_factory"
    except Exception as exc:
        last_exc = exc
    raise RuntimeError(f"Cannot instantiate {component_class}: {last_exc!r}")


def spawn_amongus_capture_rig(
    cameras,
    *,
    host: str = AMONGUS_CAPTURE_HOST_DEFAULT,
    port: int = AMONGUS_CAPTURE_PORT_DEFAULT,
    jpeg_quality: int = AMONGUS_CAPTURE_JPEG_QUALITY_DEFAULT,
    folder: str = AMONGUS_CAPTURE_FOLDER,
) -> dict:
    """Auto-spawn an Actor + AmongUsTcpCaptureComponent driving one SceneCapture2D per camera spec.

    No-op (returns ``installed=False``) when the AmongUsRealtimeCapture plugin is not loaded.
    The rig actor is a concrete TargetPoint (has a root component) so editor-mode component
    creation works; the AmongUs component itself ticks at runtime/PIE and connects to mux.
    """
    cls = _amongus_capture_component_class()
    if cls is None:
        return {
            "installed": False,
            "reason": "AmongUsTcpCaptureComponent not exposed - plugin not loaded.",
            "hint": "Run install_amongus_capture_plugin.py and relaunch UE.",
        }
    if not cameras:
        return {"installed": False, "reason": "no_cameras_in_scene_spec"}

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    rig_actor_class = getattr(unreal, "TargetPoint", None) or unreal.Actor
    rig_actor = actor_subsystem.spawn_actor_from_class(rig_actor_class, unreal.Vector(0.0, 0.0, 0.0))
    rig_actor.set_actor_label(AMONGUS_CAPTURE_RIG_LABEL)
    rig_actor.set_folder_path(folder)

    capture_components: list[object] = []
    capture_actors: list[object] = []
    capture_render_targets: list[object] = []
    camera_names: list[str] = []
    camera_flip_u: list[bool] = []
    camera_flip_v: list[bool] = []
    image_corrections: dict[str, dict[str, Any]] = {}
    for camera_spec in cameras:
        actor, comp, rt = _spawn_scene_capture_for_amongus(camera_spec=camera_spec, folder=folder)
        correction = derive_scene_capture_image_correction_from_spec(camera_spec)
        capture_actors.append(actor)
        capture_components.append(comp)
        capture_render_targets.append(rt)
        camera_names.append(str(camera_spec.name))
        camera_flip_u.append(bool(correction.flip_u))
        camera_flip_v.append(bool(correction.flip_v))
        image_corrections[str(camera_spec.name)] = {
            **correction.as_dict(),
            "reason": str(correction.reason),
        }
        # #region agent log
        if str(camera_spec.name) == "cam_top":
            try:
                import json as _json
                import time as _time

                _log_path = "/home/camp/.cursor/debug-logs/debug-05706c.log"
                _payload = {
                    "sessionId": "05706c",
                    "runId": "pre-fix",
                    "hypothesisId": "B",
                    "location": "ue_common_scene_loader.py:spawn_amongus_capture_rig",
                    "message": "cam_top capture rig flip arrays",
                    "data": {
                        "flip_u": bool(correction.flip_u),
                        "flip_v": bool(correction.flip_v),
                        "reason": str(correction.reason),
                        "camera_flip_u_tail": camera_flip_u[-3:],
                        "camera_flip_v_tail": camera_flip_v[-3:],
                    },
                    "timestamp": int(_time.time() * 1000),
                }
                with open(_log_path, "a", encoding="utf-8") as _fh:
                    _fh.write(_json.dumps(_payload, ensure_ascii=True) + "\n")
            except Exception:
                pass
        # #endregion

    try:
        component, install_path = _instantiate_actor_component(cls, rig_actor)
    except Exception as exc:
        actor_subsystem.destroy_actor(rig_actor)
        for actor in capture_actors:
            actor_subsystem.destroy_actor(actor)
        return {"installed": False, "reason": f"add_component_failed: {exc!r}"}

    used_props: dict[str, str | None] = {}
    used_props["host"] = _set_property_if_available(component, ["tcp_host", "TcpHost"], str(host))
    used_props["port"] = _set_property_if_available(component, ["tcp_port", "TcpPort"], int(port))
    used_props["scene_captures"] = _set_property_if_available(component, ["scene_captures", "SceneCaptures"], capture_components)
    used_props["camera_names"] = _set_property_if_available(component, ["camera_names", "CameraNames"], camera_names)
    used_props["camera_flip_u"] = _set_bool_array_property(component, ["camera_flip_u", "CameraFlipU"], camera_flip_u)
    used_props["camera_flip_v"] = _set_bool_array_property(component, ["camera_flip_v", "CameraFlipV"], camera_flip_v)
    used_props["jpeg_quality"] = _set_property_if_available(component, ["jpeg_quality", "JpegQuality"], int(jpeg_quality))
    used_props["auto_connect"] = _set_property_if_available(
        component,
        ["b_auto_connect", "auto_connect", "bAutoConnect"],
        True,
    )

    return {
        "installed": True,
        "install_path": install_path,
        "rig_actor_label": rig_actor.get_actor_label(),
        "rig_actor_class": rig_actor_class.__name__,
        "capture_actor_labels": [actor.get_actor_label() for actor in capture_actors],
        "camera_names": camera_names,
        "scene_capture_image_corrections": image_corrections,
        "tcp_host": str(host),
        "tcp_port": int(port),
        "jpeg_quality": int(jpeg_quality),
        "render_target_count": len(capture_render_targets),
        "applied_property_names": used_props,
    }


def apply_scene_to_current_level(
    scene_spec_path: str | Path,
    augmentation_spec_path: str | Path | None = None,
    preserve_visible_human: bool = False,
) -> dict:
    scene_spec, augmentation_summary = load_scene_spec(scene_spec_path, augmentation_spec_path)
    _preload_human_ue_calibration(scene_spec)
    avatar_source = _apply_genesis_ue_avatar_selection(scene_spec, scene_spec_path)
    level_binding = scene_spec.scene_level_binding
    load_level(level_binding.map_path)
    preserve_labels = {"GEN_visible_human"} if preserve_visible_human else set()
    clear_sync_actors(preserve_labels=preserve_labels)

    if scene_spec.support_surface is not None and scene_spec.support_surface.spawn_in_ue:
        spawn_box(
            label=f"{GENERATED_SCENE_LABEL_PREFIX}{scene_spec.support_surface.name}",
            pos_m=tuple(scene_spec.support_surface.pos),
            size_m=tuple(scene_spec.support_surface.size),
            folder=f"{GENERATED_SCENE_FOLDER}/support_surface",
            color_rgba=tuple(float(v) for v in scene_spec.support_surface.color),
        )
    robot_visual_debug: list[dict[str, Any]] = []
    articulation_summaries: list[dict[str, Any]] = []
    robot_specs = scene_spec.iter_robot_specs() if hasattr(scene_spec, "iter_robot_specs") else [scene_spec.robot]
    articulation_override = str(os.environ.get("AMONGUS_REGISTER_URDF_ARTICULATION_ID", "") or "").strip()
    if articulation_override and len(robot_specs) > 1:
        unreal.log_warning(
            "UE_SCENE: ignoring AMONGUS_REGISTER_URDF_ARTICULATION_ID for multi-robot scene; "
            "canonical robot names route to matching scene robot names."
        )
    if scene_spec.render.ue_spawn_robot:
        for robot_spec in robot_specs:
            articulation_id = articulation_override or str(robot_spec.name).strip()
            articulation_sink: dict[str, object] | None = dict() if articulation_id else None
            folder = f"{GENERATED_SCENE_FOLDER}/robot/{articulation_id or robot_spec.name}"
            spawn_urdf_path = resolved_robot_urdf_for_robot_spec(
                robot_spec,
                enable_collision=bool(robot_spec.use_collision_geometry),
                repo_root=REPO_ROOT,
            )
            visual_debug = spawn_robot_from_urdf(
                urdf_path=spawn_urdf_path,
                base_pos_m=tuple(robot_spec.base_pos),
                base_quat_xyzw=tuple(float(v) for v in robot_spec.base_quat_xyzw)
                if robot_spec.base_quat_xyzw is not None
                else None,
                joint_positions=[float(item) for item in robot_spec.joint_positions],
                use_collision_geometry=bool(robot_spec.use_collision_geometry),
                use_visual_mesh=bool(robot_spec.use_visual_mesh),
                allow_collision_fallback=bool(robot_spec.allow_collision_fallback),
                visual_asset_root=resolve_ue_visual_asset_root(robot_spec, repo_root=REPO_ROOT),
                visual_mesh_scale=float(robot_spec.visual_mesh_scale),
                visual_mesh_format=str(getattr(robot_spec, "visual_mesh_format", "") or "") or None,
                folder=folder,
                color_rgba=tuple(float(item) for item in robot_spec.color),
                articulation_actor_sink=articulation_sink,
                model_id=str(getattr(robot_spec, "model_id", "") or ""),
            )
            visual_debug["robot_id"] = articulation_id
            visual_debug["model_id"] = str(getattr(robot_spec, "model_id", "") or "")
            visual_debug["instance_id"] = str(getattr(robot_spec, "instance_id", "") or "")
            robot_visual_debug.append(visual_debug)
            if not articulation_id or articulation_sink is None:
                continue
            import ue_urdf_visual_loader as urdf_art

            urdf_art.register_articulated_robot(
                articulation_id,
                actors_by_link=articulation_sink,
                urdf_path=spawn_urdf_path,
                base_pos_m=tuple(robot_spec.base_pos),
                base_quat_xyzw=tuple(float(v) for v in robot_spec.base_quat_xyzw)
                if robot_spec.base_quat_xyzw is not None
                else None,
                use_collision_geometry=bool(robot_spec.use_collision_geometry),
                use_visual_mesh=bool(robot_spec.use_visual_mesh),
                allow_collision_fallback=bool(robot_spec.allow_collision_fallback),
                visual_asset_root=resolve_ue_visual_asset_root(robot_spec, repo_root=REPO_ROOT),
                visual_mesh_scale=float(robot_spec.visual_mesh_scale),
                folder=folder,
                color_rgba=tuple(float(item) for item in robot_spec.color),
            )
            unreal.log(f"UE_SCENE: registered URDF articulation robot_id={articulation_id} links={len(articulation_sink)}")
            articulation_summaries.append({"robot_id": articulation_id, "registered_link_count": len(articulation_sink)})

    human_anchor_m, human_world_offset_ue_m, human_align_floor, placement_json_path = _motion_human_offsets_from_scene(scene_spec)
    visible_human_summary: dict[str, Any] | None = None
    if scene_spec.render.ue_spawn_human:
        if _amongus_truthy_env("AMONGUS_UE_SPAWN_HUMAN_ANCHOR_MARKER", default=True):
            spawn_marker(
                label=f"{GENERATED_SCENE_LABEL_PREFIX}human_anchor",
                pos_m=human_anchor_m,
                scale_m=0.06,
                folder=f"{GENERATED_SCENE_FOLDER}/human",
            )
        visible_human_summary = spawn_genesis_driven_visible_human(
            scene_spec=scene_spec,
            anchor_m=human_anchor_m,
        )

    payload = {
        "scene_name": str(scene_spec.name),
        "ue_map": str(level_binding.map_path),
        "ue_hdri_name": str(level_binding.hdri_name),
        "support_surface": None
        if scene_spec.support_surface is None
        else {
            "name": str(scene_spec.support_surface.name),
            "semantic_role": str(scene_spec.support_surface.semantic_role),
            "pos_m": [float(x) for x in scene_spec.support_surface.pos],
            "size_m": [float(x) for x in scene_spec.support_surface.size],
        },
        "robot_visual_debug": robot_visual_debug[0] if len(robot_visual_debug) == 1 else robot_visual_debug,
        "robot_visual_debug_all": robot_visual_debug,
        "urdf_articulation": articulation_summaries[0] if len(articulation_summaries) == 1 else None,
        "urdf_articulations": articulation_summaries,
        "human_anchor_m": human_anchor_m,
        "human_anchor_cm": m_to_cm(human_anchor_m),
        "human_payload": {
            "align_floor": bool(human_align_floor),
            "display_vertical_sink_m": float(scene_spec.human.display_vertical_sink_m),
            "display_vertical_offset_m": float(scene_spec.human.display_vertical_offset_m),
            "display_pitch_forward_deg": float(scene_spec.human.display_pitch_forward_deg),
        },
        "camera_payloads": [camera_actor_pose_payload(camera_spec) for camera_spec in scene_spec.cameras],
        "motion_payload": {
            "source_id": scene_spec.motion.source_id,
            "sequence_npz_path": str(scene_spec.motion.resolved_sequence_npz_path) if scene_spec.motion.resolved_sequence_npz_path else "",
            "mesh_manifest_path": str(scene_spec.motion.resolved_mesh_manifest_path) if scene_spec.motion.resolved_mesh_manifest_path else "",
            "fps": float(scene_spec.motion.fps),
            "frame_count": int(scene_spec.motion.frame_count),
            "start_frame": int(scene_spec.motion.start_frame),
            "frame_step": int(scene_spec.motion.frame_step),
            "human_world_offset_m": human_world_offset_ue_m,
            "human_align_floor": bool(human_align_floor),
            "human_scene_placement_json": str(placement_json_path) if placement_json_path else "",
            "human_ue_calibration_json": str(
                resolve_human_ue_calibration_json_path(scene_spec, repo_root=REPO_ROOT) or ""
            ),
        },
        "render_payload": {
            "fps": float(scene_spec.render.fps),
            "frame_limit": int(scene_spec.render.frame_limit),
            "ue_frame_count": int(scene_spec.render.ue_frame_count),
            "ue_frame_step": int(scene_spec.render.ue_frame_step),
            "ue_render_now": bool(scene_spec.render.ue_render_now),
            "ue_spawn_robot": bool(scene_spec.render.ue_spawn_robot),
            "ue_spawn_human": bool(scene_spec.render.ue_spawn_human),
        },
        "ue_avatar_payload": {
            "body_mode": scene_spec.ue_avatar.body_mode,
            "body_name": scene_spec.ue_avatar.body_name,
            "texture_body": scene_spec.ue_avatar.texture_body,
            "texture_clothing": scene_spec.ue_avatar.texture_clothing,
            "texture_clothing_overlay": scene_spec.ue_avatar.texture_clothing_overlay,
            "skeletal_mesh_path": scene_spec.ue_avatar.skeletal_mesh_path,
            "animation_asset_root": scene_spec.ue_avatar.animation_asset_root,
            "imported_fbx_root": scene_spec.ue_avatar.imported_fbx_root,
            "fallback_animation_path": scene_spec.ue_avatar.fallback_animation_path,
            "hidden_material_path": scene_spec.ue_avatar.hidden_material_path,
            "fbx_global_scale": float(scene_spec.ue_avatar.fbx_global_scale),
        },
        "ue_avatar_source": avatar_source,
        "augmentation_payload": augmentation_summary,
    }

    target_actor = next((actor for actor in _all_level_actors() if actor.get_actor_label() == "BE_CameraTarget"), None)
    if target_actor is not None and payload["camera_payloads"]:
        target_actor.set_actor_location(unreal.Vector(*payload["camera_payloads"][0]["lookat_cm"]), False, False)
    payload["bedlam_camera_rig_hidden"] = hide_bedlam_freeview_camera_rig(log=True)

    payload["visible_human"] = visible_human_summary
    capture_host = str(os.environ.get("AMONGUS_CAPTURE_TCP_HOST", AMONGUS_CAPTURE_HOST_DEFAULT)).strip() or AMONGUS_CAPTURE_HOST_DEFAULT
    try:
        capture_port = int(os.environ.get("AMONGUS_CAPTURE_TCP_PORT", AMONGUS_CAPTURE_PORT_DEFAULT))
    except (TypeError, ValueError):
        capture_port = AMONGUS_CAPTURE_PORT_DEFAULT
    try:
        capture_jpeg_quality = int(os.environ.get("AMONGUS_CAPTURE_JPEG_QUALITY", AMONGUS_CAPTURE_JPEG_QUALITY_DEFAULT))
    except (TypeError, ValueError):
        capture_jpeg_quality = AMONGUS_CAPTURE_JPEG_QUALITY_DEFAULT
    payload["amongus_capture_rig"] = spawn_amongus_capture_rig(
        scene_spec.cameras,
        host=capture_host,
        port=capture_port,
        jpeg_quality=capture_jpeg_quality,
    )

    return payload
