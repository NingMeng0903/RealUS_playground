# Copyright Among_US contributors — dumps MRQ-relevant camera pose from LevelSequence assets.
"""Run inside Unreal Editor (ExecutePythonScript). Reads static MovieScene3DTransform defaults
for the CineCamera possessable on each named LevelSequence, converts UE cm -> meters and
Rotator -> quaternion (x,y,z,w), writes JSON compatible with ue_camera_calibration_io.

Environment:
  AMONGUS_UE_SEQ_DUMP_OUT   (required) output JSON path
  AMONGUS_UE_SEQ_PREFIX     sequence name prefix, e.g. CMU_114_114_11_poses (matches *_cam_left)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import unreal

LS_ROOT = "/Game/Bedlam/LevelSequences/"

SUFFIX_TO_ID = {
    "_cam_left": "cam_left",
    "_cam_right": "cam_right",
    "_cam_top": "cam_top",
}


def _channel_scalar_default(channel) -> float:
    for name in ("get_default", "get_default_value"):
        fn = getattr(channel, name, None)
        if callable(fn):
            try:
                v = fn()
            except Exception:
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    keys = channel.get_keys()
    if keys:
        return float(keys[0].get_value())
    raise RuntimeError(f"Cannot read default for channel {channel}")


def _read_transform_rpy_cm(binding) -> tuple[float, float, float, float, float, float]:
    for track in binding.get_tracks():
        cls = track.get_class().get_name()
        if "MovieScene3DTransformTrack" not in cls:
            continue
        for section in track.get_sections():
            chans = section.get_all_channels()
            if len(chans) < 6:
                continue
            x = _channel_scalar_default(chans[0])
            y = _channel_scalar_default(chans[1])
            z = _channel_scalar_default(chans[2])
            roll = _channel_scalar_default(chans[3])
            pitch = _channel_scalar_default(chans[4])
            yaw = _channel_scalar_default(chans[5])
            return (x, y, z, roll, pitch, yaw)
    raise RuntimeError(f"No MovieScene3DTransformTrack with defaults on binding {binding.get_name()}")


def _find_cine_camera_binding(level_sequence):
    last_err: str | None = None
    for binding in level_sequence.get_possessables():
        name = binding.get_name()
        if "CameraComponent" in name and "Cine" not in name:
            continue
        if "CineCamera" not in name and name != "BE_CineCameraActor_Blueprint":
            continue
        try:
            _read_transform_rpy_cm(binding)
            return binding
        except RuntimeError as exc:
            last_err = str(exc)
            continue
    raise RuntimeError(f"No CineCamera binding with transform defaults (last: {last_err})")


def _focal_length_default(level_sequence: unreal.LevelSequence) -> float | None:
    for binding in level_sequence.get_possessables():
        if binding.get_name() != "CameraComponent":
            continue
        for track in binding.get_tracks():
            if "MovieSceneFloatTrack" not in track.get_class().get_name():
                continue
            prop = ""
            try:
                prop = str(track.get_property_name_and_path())
            except Exception:
                pass
            if "CurrentFocalLength" not in prop and "FocalLength" not in prop:
                continue
            for section in track.get_sections():
                chans = section.get_all_channels()
                if not chans:
                    continue
                return float(_channel_scalar_default(chans[0]))
    return None


def _dump_one_sequence(asset_path: str) -> dict:
    ls = unreal.load_asset(asset_path)
    if ls is None:
        raise RuntimeError(f"Cannot load {asset_path}")
    binding = _find_cine_camera_binding(ls)
    x_cm, y_cm, z_cm, roll, pitch, yaw = _read_transform_rpy_cm(binding)
    rot = unreal.Rotator(roll=float(roll), pitch=float(pitch), yaw=float(yaw))
    q = rot.quaternion()
    focal = _focal_length_default(ls)
    base = Path(asset_path).name
    cam_id = None
    for suf, cid in SUFFIX_TO_ID.items():
        if base.endswith(suf):
            cam_id = cid
            break
    if cam_id is None:
        cam_id = base
    return {
        "id": cam_id,
        "level_sequence": asset_path,
        "location_m": [x_cm / 100.0, y_cm / 100.0, z_cm / 100.0],
        "quaternion_xyzw": [float(q.x), float(q.y), float(q.z), float(q.w)],
        "rotator_roll_pitch_yaw_deg": [float(roll), float(pitch), float(yaw)],
        "location_cm": [float(x_cm), float(y_cm), float(z_cm)],
        "current_focal_length_mm": focal,
    }


def main() -> None:
    out_raw = os.environ.get("AMONGUS_UE_SEQ_DUMP_OUT", "").strip()
    prefix = os.environ.get("AMONGUS_UE_SEQ_PREFIX", "").strip()
    if not out_raw:
        unreal.log_error("ue_dump_sequencer_camera_poses: set AMONGUS_UE_SEQ_DUMP_OUT")
        sys.exit(2)
    if not prefix:
        unreal.log_error("ue_dump_sequencer_camera_poses: set AMONGUS_UE_SEQ_PREFIX")
        sys.exit(2)
    out_path = Path(out_raw).expanduser()

    try:
        asset_paths = unreal.EditorAssetLibrary.list_assets(LS_ROOT, recursive=False, include_folder=False)
    except Exception as exc:
        unreal.log_error(f"ue_dump_sequencer_camera_poses: list_assets failed: {exc}")
        sys.exit(3)

    selected: list[str] = []
    for ap in asset_paths:
        if not isinstance(ap, str):
            continue
        short = ap.rsplit("/", maxsplit=1)[-1]
        if not short.startswith(prefix):
            continue
        if not any(short.endswith(suf) for suf in SUFFIX_TO_ID):
            continue
        if unreal.EditorAssetLibrary.does_asset_exist(ap):
            selected.append(ap)
    selected.sort()

    if len(selected) != 3:
        unreal.log_warning(
            f"ue_dump_sequencer_camera_poses: expected 3 sequences for prefix {prefix!r}, got {len(selected)}: {selected}"
        )

    rows: list[dict] = []
    for ap in selected:
        try:
            rows.append(_dump_one_sequence(ap))
        except Exception as exc:
            unreal.log_error(f"ue_dump_sequencer_camera_poses: failed {ap}: {exc}")
            sys.exit(4)

    doc = {
        "metadata": {
            "source": "ue_dump_sequencer_camera_poses",
            "level_sequences_root": LS_ROOT,
            "sequence_prefix": prefix,
        },
        "cameras": [
            {
                "id": r["id"],
                "location_m": r["location_m"],
                "quaternion": r["quaternion_xyzw"],
                "quat_order": "xyzw",
                "width": None,
                "height": None,
                "fov_deg": None,
                "debug": {
                    "level_sequence": r["level_sequence"],
                    "rotator_roll_pitch_yaw_deg": r["rotator_roll_pitch_yaw_deg"],
                    "location_cm": r["location_cm"],
                    "current_focal_length_mm": r["current_focal_length_mm"],
                },
            }
            for r in rows
        ],
        "raw_rows": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    unreal.log(f"ue_dump_sequencer_camera_poses: wrote {out_path} ({len(rows)} cameras)")


if __name__ == "__main__":
    main()
