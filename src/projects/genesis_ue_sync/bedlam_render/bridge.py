from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from bridge.adapters.ue import ue_camera_payload_from_spec, ue_world_point_from_genesis_m
from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.human_refit.placement_json import (
    read_human_scene_placement_mesh_offset_m,
    resolve_human_scene_placement_json_path,
)
from projects.genesis_ue_sync.sim_platform.scenes import resolve_scene_spec_with_augmentation


@dataclass(frozen=True)
class BedlamSequenceBridgeResult:
    csv_path: Path
    meta_path: Path
    sequence_names: list[str]


def _sanitize_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_") or "item"


def _body_comment(*, texture_body: str | None, texture_clothing: str | None, texture_clothing_overlay: str | None) -> str:
    items: list[str] = []
    if texture_body:
        items.append(f"texture_body={texture_body}")
    if texture_clothing:
        items.append(f"texture_clothing={texture_clothing}")
    if texture_clothing_overlay:
        items.append(f"texture_clothing_overlay={texture_clothing_overlay}")
    return ";".join(items)


def _ue_body_location_cm(scene_spec) -> tuple[float, float, float]:
    repo_root = project_paths(__file__).root
    placement_path = resolve_human_scene_placement_json_path(scene_spec, repo_root=repo_root)
    if placement_path is not None:
        parsed = read_human_scene_placement_mesh_offset_m(placement_path)
        if parsed is not None:
            _anchor_m, mesh_off_m, _align_floor = parsed
            ue_m = ue_world_point_from_genesis_m(mesh_off_m)
            return tuple(float(v) * 100.0 for v in ue_m.tolist())
    anchor_m = list(float(v) for v in scene_spec.resolved_human_anchor())
    anchor_m[2] += float(scene_spec.human.display_vertical_offset_m)
    ue_m = ue_world_point_from_genesis_m(anchor_m)
    return tuple(float(v) * 100.0 for v in ue_m.tolist())


def write_bedlam_be_seq_csv(
    *,
    scene_spec_path: Path,
    augmentation_spec_path: Path | None,
    output_dir: Path,
    sequence_prefix: str | None = None,
    clothing_mode: str = "geometry",
    bedlam_body_name: str | None = None,
) -> BedlamSequenceBridgeResult:
    scene_spec, augmentation_summary = resolve_scene_spec_with_augmentation(scene_spec_path, augmentation_spec_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "be_seq.csv"
    meta_path = output_dir / "be_seq_meta.json"

    prefix = _sanitize_token(sequence_prefix or scene_spec.name)
    frame_count = int(scene_spec.render.ue_frame_count or scene_spec.render.frame_limit or scene_spec.motion.frame_count or 1)
    frame_count = max(frame_count, 1)
    avatar = scene_spec.ue_avatar
    body_name = str(bedlam_body_name or avatar.body_name)
    body_x, body_y, body_z = _ue_body_location_cm(scene_spec)
    mode = str(clothing_mode).strip().lower()
    if mode not in {"geometry", "overlay", "auto"}:
        raise ValueError(f"Unsupported BEDLAM clothing_mode: {clothing_mode}")
    texture_clothing = avatar.texture_clothing
    texture_clothing_overlay = avatar.texture_clothing_overlay
    if mode == "geometry" and texture_clothing:
        # Official create_level_sequences_csv.py chooses overlay first when present.
        # For actual BEDLAM clothes geometry, leave overlay empty and keep texture_clothing.
        texture_clothing_overlay = None
    elif mode == "overlay":
        texture_clothing = None
    body_comment = _body_comment(
        texture_body=avatar.texture_body,
        texture_clothing=texture_clothing,
        texture_clothing_overlay=texture_clothing_overlay,
    )

    sequence_names: list[str] = []
    rows: list[dict[str, object]] = []
    for camera_index, camera_spec in enumerate(scene_spec.cameras):
        camera = ue_camera_payload_from_spec(camera_spec)
        sequence_name = f"{prefix}_{_sanitize_token(camera_spec.name)}_{camera_index:06d}"
        sequence_names.append(sequence_name)
        group_comment_items = [
            f"sequence_name={sequence_name}",
            f"frames={frame_count}",
            f"hdri={scene_spec.scene_level_binding.hdri_name}",
            f"camera_hfov={float(camera_spec.fov)}",
        ]
        rows.append(
            {
                "Type": "Group",
                "Index": "",
                "Body": "",
                "X": f"{float(camera['x']):.8f}",
                "Y": f"{float(camera['y']):.8f}",
                "Z": f"{float(camera['z']):.8f}",
                "Yaw": f"{float(camera['yaw']):.8f}",
                "Pitch": f"{float(camera['pitch']):.8f}",
                "Roll": f"{float(camera['roll']):.8f}",
                "Comment": ";".join(group_comment_items),
            }
        )
        rows.append(
            {
                "Type": "Body",
                "Index": "0",
                "Body": body_name,
                "X": f"{body_x:.8f}",
                "Y": f"{body_y:.8f}",
                "Z": f"{body_z:.8f}",
                "Yaw": "0.00000000",
                "Pitch": f"{float(scene_spec.human.display_pitch_forward_deg):.8f}",
                "Roll": "0.00000000",
                "Comment": body_comment,
            }
        )
    rows.append(
        {
            "Type": "Comment",
            "Index": "",
            "Body": "",
            "X": "",
            "Y": "",
            "Z": "",
            "Yaw": "",
            "Pitch": "",
            "Roll": "",
            "Comment": "sentinel row for BEDLAM create_level_sequences_csv.py",
        }
    )

    fieldnames = ["Type", "Index", "Body", "X", "Y", "Z", "Yaw", "Pitch", "Roll", "Comment"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    meta_path.write_text(
        json.dumps(
            {
                "scene_spec": str(scene_spec_path),
                "augmentation_spec": None if augmentation_spec_path is None else str(augmentation_spec_path),
                "augmentation": augmentation_summary,
                "sequence_names": sequence_names,
                "body_name": body_name,
                "geometry_cache_motion_source": body_name,
                "scene_motion_source_id": scene_spec.motion.source_id,
                "scene_motion_sequence_npz_path": str(scene_spec.motion.resolved_sequence_npz_path)
                if scene_spec.motion.resolved_sequence_npz_path
                else "",
                "body_location_cm": [body_x, body_y, body_z],
                "body_comment": body_comment,
                "clothing_mode": mode,
                "frame_count": frame_count,
                "note": (
                    "Generated for official BEDLAM2 create_level_sequences_csv.py GeometryCache render flow. "
                    "In this mode, animation comes from geometry_cache_motion_source, not scene_motion_source_id."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return BedlamSequenceBridgeResult(csv_path=csv_path, meta_path=meta_path, sequence_names=sequence_names)
