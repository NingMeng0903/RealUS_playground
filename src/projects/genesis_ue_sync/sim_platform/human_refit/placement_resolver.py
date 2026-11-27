"""Resolve HumanScenePlacement JSON paths shared by Genesis tools and UE session payloads."""

from __future__ import annotations

import os
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from common.project import project_paths

from projects.genesis_ue_sync.sim_platform.scenes.common_scene import SyncSceneSpec
from projects.genesis_ue_sync.sim_platform.scenes.human_scene_placement import HumanScenePlacement

PLACEMENT_JSON_ENV = "AMONGUS_HUMAN_SCENE_PLACEMENT_JSON"
PLACEMENT_SIDECAR_NAME = "human_scene_placement.json"


def placement_cache_key(scene_spec: SyncSceneSpec, sequence_path: Path) -> str:
    """Placement belongs to a specific scene, body reference, and motion clip."""
    path = sequence_path.expanduser().resolve()
    stat = path.stat()
    body_ref = scene_spec.motion.resolved_sequence_npz_path
    ref_stat = body_ref.stat() if body_ref is not None and body_ref.is_file() else None
    payload = {
        "version": 2,
        "sequence": [str(path), stat.st_size, stat.st_mtime_ns],
        "body_ref": [str(body_ref), ref_stat.st_size, ref_stat.st_mtime_ns] if ref_stat else None,
        "human": asdict(scene_spec.human),
        "support": asdict(scene_spec.support_surface) if scene_spec.support_surface else None,
        "avatar": asdict(scene_spec.ue_avatar),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def resolve_human_scene_placement_path(scene_spec: SyncSceneSpec, *, repo_root: Path | None = None) -> Path | None:
    repo_root = repo_root or project_paths(__file__).root
    raw = str(os.environ.get(PLACEMENT_JSON_ENV, "") or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_file() else None

    meta_rel = scene_spec.metadata.get("human_scene_placement")
    if meta_rel:
        candidate = Path(str(meta_rel))
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if candidate.is_file():
            return candidate

    seq_path = scene_spec.motion.resolved_sequence_npz_path
    if seq_path is not None:
        sibling = Path(seq_path).expanduser().resolve().parent / PLACEMENT_SIDECAR_NAME
        if sibling.is_file():
            return sibling

    return None


def try_load_human_scene_placement_for_scene(
    scene_spec: SyncSceneSpec,
    *,
    repo_root: Path | None = None,
) -> HumanScenePlacement | None:
    path = resolve_human_scene_placement_path(scene_spec, repo_root=repo_root)
    if path is None:
        return None
    return HumanScenePlacement.load(path)


def genesis_mesh_world_offset_m_from_placement(placement: HumanScenePlacement) -> tuple[float, float, float]:
    ox, oy, oz = placement.world_offset_m
    extra_z = float(placement.display_vertical_sink_m + placement.display_vertical_offset_m)
    return float(ox), float(oy), float(oz + extra_z)


def compute_and_save_human_scene_placement(
    scene_spec: SyncSceneSpec,
    sequence_npz_path: Path,
    *,
    output_path: Path | None = None,
    device: str | None = "cpu",
    human_center_mode: str = "bed_center",
    root_projection_bed_center: bool = True,
    fit_samples: int = 7,
) -> HumanScenePlacement:
    """Run bed-fit placement (Genesis canonical) and write HumanScenePlacement JSON."""
    from projects.genesis_ue_sync.sim_platform.datasets import HumanMotionSequence
    from projects.genesis_ue_sync.sim_platform.scenes.human_scene_placement import compute_human_scene_placement

    sequence_npz_path = sequence_npz_path.expanduser().resolve()
    sequence = HumanMotionSequence.load(sequence_npz_path)
    placement = compute_human_scene_placement(
        sequence,
        scene_spec=scene_spec,
        sequence_npz_path=str(sequence_npz_path),
        device=device,
        human_center_mode=str(human_center_mode),
        root_projection_bed_center=bool(root_projection_bed_center),
        fit_samples=int(fit_samples),
    )
    out_path = output_path or (sequence_npz_path.parent / PLACEMENT_SIDECAR_NAME)
    placement.save(out_path)
    return placement


def placement_sidecar_for_npz(npz_path: Path) -> Path:
    stem = Path(npz_path).expanduser().resolve().stem
    return Path(npz_path).expanduser().resolve().with_name(f"{stem}_human_scene_placement.json")


def preferred_placement_output_path(
    scene_spec: SyncSceneSpec,
    amass_npz_path: Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Local scene-keyed cache; never write sidecars into shared AMASS assets."""
    repo_root = repo_root or project_paths(__file__).root
    key = placement_cache_key(scene_spec, amass_npz_path)
    return repo_root / "outputs/realus_sync/placement_cache" / key / PLACEMENT_SIDECAR_NAME


def resolve_or_compute_human_scene_placement(
    scene_spec: SyncSceneSpec,
    sequence_npz_path: Path,
    *,
    repo_root: Path | None = None,
    device: str | None = "cpu",
) -> tuple[HumanScenePlacement, Path]:
    """Load placement JSON if present; otherwise compute bed-fit and save alongside sequence NPZ."""
    loaded = try_load_human_scene_placement_for_scene(scene_spec, repo_root=repo_root)
    seq_path = sequence_npz_path.expanduser().resolve()
    if loaded is not None:
        path = resolve_human_scene_placement_path(scene_spec, repo_root=repo_root or project_paths(__file__).root)
        assert path is not None
        return loaded, path
    placement = compute_and_save_human_scene_placement(scene_spec, seq_path, device=device)
    return placement, seq_path.parent / PLACEMENT_SIDECAR_NAME


def resolve_or_compute_placement_for_amass(
    scene_spec: SyncSceneSpec,
    sequence: Any,
    *,
    amass_npz_path: Path,
    repo_root: Path | None = None,
    proxy_geometry: Any = None,
    placement_sample_frames: int = 11,
    device: str | None = "cpu",
    force_recompute: bool = False,
    human_center_mode: str = "bed_center",
    root_projection_bed_center: bool = True,
    fit_samples: int | None = None,
) -> tuple[HumanScenePlacement, Path]:
    """GT bed-fit only: load sidecar next to AMASS NPZ or compute and save."""
    from projects.genesis_ue_sync.sim_platform.scenes.human_scene_placement import compute_human_scene_placement

    repo_root = repo_root or project_paths(__file__).root
    npz_path = Path(amass_npz_path).expanduser().resolve()
    out_path = preferred_placement_output_path(scene_spec, npz_path, repo_root=repo_root)
    key = placement_cache_key(scene_spec, npz_path)

    if not force_recompute:
        env_raw = str(os.environ.get(PLACEMENT_JSON_ENV, "") or "").strip()
        if env_raw:
            candidate = Path(env_raw).expanduser()
            if candidate.is_file():
                placement = HumanScenePlacement.load(candidate)
                if placement.metadata.get("placement_cache_key") == key:
                    return placement, candidate
        if out_path.is_file():
            placement = HumanScenePlacement.load(out_path)
            if placement.metadata.get("placement_cache_key") == key:
                return placement, out_path

    placement = compute_human_scene_placement(
        sequence,
        scene_spec=scene_spec,
        sequence_npz_path=str(npz_path),
        device=device,
        placement_sample_frames=int(placement_sample_frames),
        proxy_geometry=proxy_geometry,
        human_center_mode=str(human_center_mode),
        root_projection_bed_center=bool(root_projection_bed_center),
        fit_samples=int(fit_samples) if fit_samples is not None else None,
    )
    placement = replace(placement, metadata={**placement.metadata, "placement_cache_key": key})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    placement.save(out_path)
    return placement, out_path
