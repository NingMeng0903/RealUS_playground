from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import (
    SceneCharacterVisualBindingSpec,
    SceneMotionSpec,
    SceneRenderSpec,
    SyncSceneSpec,
)

try:
    import yaml
except ImportError:
    yaml = None


def _read_mapping(path: Path) -> dict[str, Any]:
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Augmentation spec is empty: {path}")
    if yaml is not None:
        try:
            payload = yaml.safe_load(raw_text)
        except Exception:
            payload = json.loads(raw_text)
    else:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Augmentation spec is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping payload in augmentation spec: {path}")
    return payload


def _optional_string_list(payload: Any) -> list[str | None]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise TypeError(f"Expected list value, got {type(payload).__name__}")
    out: list[str | None] = []
    for item in payload:
        out.append(None if item is None else str(item))
    return out


@dataclass
class SceneCharacterVisualOverrideSpec:
    body_mode: str | None = None
    body_name: str | None = None
    texture_body: str | None = None
    texture_clothing: str | None = None
    texture_clothing_overlay: str | None = None
    skeletal_mesh_path: str | None = None
    animation_asset_root: str | None = None
    imported_fbx_root: str | None = None
    fallback_animation_path: str | None = None
    hidden_material_path: str | None = None
    fbx_global_scale: float | None = None


@dataclass
class SceneAppearanceAugmentationSpec:
    mode: str = "inherit"
    seed: int | None = None
    candidate_body_names: list[str] = field(default_factory=list)
    candidate_skeletal_mesh_paths: list[str] = field(default_factory=list)
    candidate_texture_body: list[str] = field(default_factory=list)
    candidate_texture_clothing: list[str | None] = field(default_factory=list)
    candidate_texture_clothing_overlay: list[str | None] = field(default_factory=list)


@dataclass
class SceneAugmentationSpec:
    name: str = "default_augmentation"
    seed: int | None = None
    task: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    appearance: SceneAppearanceAugmentationSpec = field(default_factory=SceneAppearanceAugmentationSpec)
    character_visual: SceneCharacterVisualOverrideSpec = field(default_factory=SceneCharacterVisualOverrideSpec)
    render_override: dict[str, Any] = field(default_factory=dict)
    motion_override: dict[str, Any] = field(default_factory=dict)


def scene_augmentation_to_dict(spec: SceneAugmentationSpec) -> dict[str, Any]:
    return dataclasses.asdict(spec)


def merge_scene_augmentation_specs(
    base: SceneAugmentationSpec | None,
    overlay: SceneAugmentationSpec,
) -> SceneAugmentationSpec:
    if base is None:
        return overlay
    appearance = SceneAppearanceAugmentationSpec(
        mode=overlay.appearance.mode if overlay.appearance.mode != "inherit" else base.appearance.mode,
        seed=overlay.appearance.seed if overlay.appearance.seed is not None else base.appearance.seed,
        candidate_body_names=overlay.appearance.candidate_body_names or list(base.appearance.candidate_body_names),
        candidate_skeletal_mesh_paths=overlay.appearance.candidate_skeletal_mesh_paths
        or list(base.appearance.candidate_skeletal_mesh_paths),
        candidate_texture_body=overlay.appearance.candidate_texture_body or list(base.appearance.candidate_texture_body),
        candidate_texture_clothing=overlay.appearance.candidate_texture_clothing
        or list(base.appearance.candidate_texture_clothing),
        candidate_texture_clothing_overlay=overlay.appearance.candidate_texture_clothing_overlay
        or list(base.appearance.candidate_texture_clothing_overlay),
    )
    character_visual = SceneCharacterVisualOverrideSpec(
        body_mode=overlay.character_visual.body_mode or base.character_visual.body_mode,
        body_name=overlay.character_visual.body_name or base.character_visual.body_name,
        texture_body=overlay.character_visual.texture_body or base.character_visual.texture_body,
        texture_clothing=(
            overlay.character_visual.texture_clothing
            if overlay.character_visual.texture_clothing is not None
            else base.character_visual.texture_clothing
        ),
        texture_clothing_overlay=(
            overlay.character_visual.texture_clothing_overlay
            if overlay.character_visual.texture_clothing_overlay is not None
            else base.character_visual.texture_clothing_overlay
        ),
        skeletal_mesh_path=overlay.character_visual.skeletal_mesh_path or base.character_visual.skeletal_mesh_path,
        animation_asset_root=overlay.character_visual.animation_asset_root or base.character_visual.animation_asset_root,
        imported_fbx_root=overlay.character_visual.imported_fbx_root or base.character_visual.imported_fbx_root,
        fallback_animation_path=overlay.character_visual.fallback_animation_path or base.character_visual.fallback_animation_path,
        hidden_material_path=overlay.character_visual.hidden_material_path or base.character_visual.hidden_material_path,
        fbx_global_scale=(
            overlay.character_visual.fbx_global_scale
            if overlay.character_visual.fbx_global_scale is not None
            else base.character_visual.fbx_global_scale
        ),
    )
    return SceneAugmentationSpec(
        name=overlay.name or base.name,
        seed=overlay.seed if overlay.seed is not None else base.seed,
        task={**base.task, **overlay.task},
        metadata={**base.metadata, **overlay.metadata},
        appearance=appearance,
        character_visual=character_visual,
        render_override={**base.render_override, **overlay.render_override},
        motion_override={**base.motion_override, **overlay.motion_override},
    )


def load_scene_augmentation_spec(path: str | Path) -> SceneAugmentationSpec:
    payload = _read_mapping(project_paths(__file__).resolve_from_root(path))
    appearance_payload = dict(payload.get("appearance", {}))
    character_payload = dict(payload.get("character_visual", {}))
    return SceneAugmentationSpec(
        name=str(payload.get("name", "default_augmentation")),
        seed=None if payload.get("seed") is None else int(payload.get("seed")),
        task=dict(payload.get("task", {})),
        metadata=dict(payload.get("metadata", {})),
        appearance=SceneAppearanceAugmentationSpec(
            mode=str(appearance_payload.get("mode", "inherit")),
            seed=None if appearance_payload.get("seed") is None else int(appearance_payload.get("seed")),
            candidate_body_names=[str(item) for item in appearance_payload.get("candidate_body_names", [])],
            candidate_skeletal_mesh_paths=[str(item) for item in appearance_payload.get("candidate_skeletal_mesh_paths", [])],
            candidate_texture_body=[str(item) for item in appearance_payload.get("candidate_texture_body", [])],
            candidate_texture_clothing=_optional_string_list(appearance_payload.get("candidate_texture_clothing")),
            candidate_texture_clothing_overlay=_optional_string_list(
                appearance_payload.get("candidate_texture_clothing_overlay")
            ),
        ),
        character_visual=SceneCharacterVisualOverrideSpec(
            body_mode=character_payload.get("body_mode"),
            body_name=character_payload.get("body_name"),
            texture_body=character_payload.get("texture_body"),
            texture_clothing=character_payload.get("texture_clothing"),
            texture_clothing_overlay=character_payload.get("texture_clothing_overlay"),
            skeletal_mesh_path=character_payload.get("skeletal_mesh_path"),
            animation_asset_root=character_payload.get("animation_asset_root"),
            imported_fbx_root=character_payload.get("imported_fbx_root"),
            fallback_animation_path=character_payload.get("fallback_animation_path"),
            hidden_material_path=character_payload.get("hidden_material_path"),
            fbx_global_scale=(
                None if character_payload.get("fbx_global_scale") is None else float(character_payload.get("fbx_global_scale"))
            ),
        ),
        render_override=dict(payload.get("render_override", {})),
        motion_override=dict(payload.get("motion_override", {})),
    )


def _pick_or_keep(rng: random.Random, candidates: list[str | None], current: str | None) -> str | None:
    if not candidates:
        return current
    return candidates[int(rng.randrange(len(candidates)))]


def _apply_character_override(
    base: SceneCharacterVisualBindingSpec,
    override: SceneCharacterVisualOverrideSpec,
) -> SceneCharacterVisualBindingSpec:
    return dataclasses.replace(
        base,
        body_mode=base.body_mode if override.body_mode is None else str(override.body_mode),
        body_name=base.body_name if override.body_name is None else str(override.body_name),
        texture_body=base.texture_body if override.texture_body is None else override.texture_body,
        texture_clothing=base.texture_clothing if override.texture_clothing is None else override.texture_clothing,
        texture_clothing_overlay=(
            base.texture_clothing_overlay
            if override.texture_clothing_overlay is None
            else override.texture_clothing_overlay
        ),
        skeletal_mesh_path=base.skeletal_mesh_path if override.skeletal_mesh_path is None else str(override.skeletal_mesh_path),
        animation_asset_root=(
            base.animation_asset_root if override.animation_asset_root is None else str(override.animation_asset_root)
        ),
        imported_fbx_root=base.imported_fbx_root if override.imported_fbx_root is None else str(override.imported_fbx_root),
        fallback_animation_path=(
            base.fallback_animation_path
            if override.fallback_animation_path is None
            else str(override.fallback_animation_path)
        ),
        hidden_material_path=(
            base.hidden_material_path if override.hidden_material_path is None else str(override.hidden_material_path)
        ),
        fbx_global_scale=base.fbx_global_scale if override.fbx_global_scale is None else float(override.fbx_global_scale),
    )


def _apply_appearance_mode(
    base: SceneCharacterVisualBindingSpec,
    appearance: SceneAppearanceAugmentationSpec,
    *,
    inherited_seed: int | None,
) -> tuple[SceneCharacterVisualBindingSpec, dict[str, Any]]:
    mode = str(appearance.mode or "inherit").strip().lower()
    chosen_seed = appearance.seed if appearance.seed is not None else inherited_seed
    rng = random.Random(chosen_seed if chosen_seed is not None else 0)
    if mode in {"", "inherit"}:
        return base, {"mode": "inherit", "seed": chosen_seed}
    if mode == "nude":
        updated = dataclasses.replace(base, texture_clothing=None, texture_clothing_overlay=None)
        return updated, {"mode": "nude", "seed": chosen_seed}
    if mode == "dressed":
        updated = dataclasses.replace(
            base,
            texture_clothing=_pick_or_keep(rng, appearance.candidate_texture_clothing, base.texture_clothing),
            texture_clothing_overlay=_pick_or_keep(
                rng,
                appearance.candidate_texture_clothing_overlay,
                base.texture_clothing_overlay,
            ),
        )
        return updated, {"mode": "dressed", "seed": chosen_seed}
    if mode == "randomized":
        updated = dataclasses.replace(
            base,
            body_name=str(_pick_or_keep(rng, list(appearance.candidate_body_names), base.body_name) or base.body_name),
            skeletal_mesh_path=str(
                _pick_or_keep(rng, list(appearance.candidate_skeletal_mesh_paths), base.skeletal_mesh_path)
                or base.skeletal_mesh_path
            ),
            texture_body=_pick_or_keep(rng, list(appearance.candidate_texture_body), base.texture_body),
            texture_clothing=_pick_or_keep(rng, appearance.candidate_texture_clothing, base.texture_clothing),
            texture_clothing_overlay=_pick_or_keep(
                rng,
                appearance.candidate_texture_clothing_overlay,
                base.texture_clothing_overlay,
            ),
        )
        return updated, {"mode": "randomized", "seed": chosen_seed}
    raise ValueError(f"Unsupported augmentation appearance mode: {appearance.mode}")


def _apply_render_override(base: SceneRenderSpec, override: dict[str, Any]) -> SceneRenderSpec:
    if not override:
        return base
    return dataclasses.replace(
        base,
        fps=float(override.get("fps", base.fps)),
        frame_limit=int(override.get("frame_limit", base.frame_limit)),
        genesis_backend=str(override.get("genesis_backend", base.genesis_backend)),
        ue_frame_count=int(override.get("ue_frame_count", base.ue_frame_count)),
        ue_frame_step=int(override.get("ue_frame_step", base.ue_frame_step)),
        ue_render_now=bool(override.get("ue_render_now", base.ue_render_now)),
        ue_spawn_robot=bool(override.get("ue_spawn_robot", base.ue_spawn_robot)),
        ue_spawn_human=bool(override.get("ue_spawn_human", base.ue_spawn_human)),
    )


def _apply_motion_override(base: SceneMotionSpec, override: dict[str, Any]) -> SceneMotionSpec:
    if not override:
        return base
    return dataclasses.replace(
        base,
        source_id=str(override.get("source_id", base.source_id)),
        source_path=str(override.get("source_path", base.source_path)),
        sequence_npz_path=str(override.get("sequence_npz_path", base.sequence_npz_path)),
        mesh_manifest_path=str(override.get("mesh_manifest_path", base.mesh_manifest_path)),
        fps=float(override.get("fps", base.fps)),
        frame_count=int(override.get("frame_count", base.frame_count)),
        start_frame=int(override.get("start_frame", base.start_frame)),
        frame_step=int(override.get("frame_step", base.frame_step)),
    )


def apply_augmentation_to_scene_spec(
    scene_spec: SyncSceneSpec,
    augmentation_spec: SceneAugmentationSpec,
) -> tuple[SyncSceneSpec, dict[str, Any]]:
    avatar = _apply_character_override(scene_spec.ue_avatar, augmentation_spec.character_visual)
    avatar, appearance_summary = _apply_appearance_mode(
        avatar,
        augmentation_spec.appearance,
        inherited_seed=augmentation_spec.seed,
    )
    metadata = dict(scene_spec.metadata)
    augmentation_summary = {
        "name": augmentation_spec.name,
        "seed": augmentation_spec.seed,
        "task": dict(augmentation_spec.task),
        "metadata": dict(augmentation_spec.metadata),
        "appearance": appearance_summary,
        "selected_ue_avatar": dataclasses.asdict(avatar),
    }
    metadata["augmentation"] = augmentation_summary
    updated = dataclasses.replace(
        scene_spec,
        motion=_apply_motion_override(scene_spec.motion, augmentation_spec.motion_override),
        render=_apply_render_override(scene_spec.render, augmentation_spec.render_override),
        bindings=dataclasses.replace(scene_spec.bindings, character_visual=avatar),
        metadata=metadata,
    )
    return updated, augmentation_summary


def resolve_scene_spec_with_augmentation(
    scene_spec_path: str | Path,
    augmentation_spec_path: str | Path | None = None,
) -> tuple[SyncSceneSpec, dict[str, Any] | None]:
    from projects.genesis_ue_sync.sim_platform.scenes.common_scene import load_sync_scene_spec

    scene_spec = load_sync_scene_spec(project_paths(__file__).resolve_from_root(scene_spec_path))
    if augmentation_spec_path is None:
        return scene_spec, None
    augmentation = load_scene_augmentation_spec(augmentation_spec_path)
    return apply_augmentation_to_scene_spec(scene_spec, augmentation)
