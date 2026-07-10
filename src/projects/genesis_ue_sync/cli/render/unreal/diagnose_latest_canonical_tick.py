from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _latest_result(session_dir: Path) -> Path | None:
    results_dir = session_dir / "results"
    if not results_dir.is_dir():
        return None
    candidates: list[Path] = []
    for path in results_dir.glob("*.json"):
        data = _load_json(path)
        if data is None:
            continue
        if str(data.get("detail", "")) == "canonical_tick_applied":
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def _visible_human_bones(session_dir: Path) -> dict[str, Any]:
    data = _load_json(session_dir / "visible_human_bones.json")
    return data if data is not None else {}


def _visible_human_pelvis_align(session_dir: Path) -> dict[str, Any]:
    data = _load_json(session_dir / "visible_human_pelvis_align.json")
    return data if data is not None else {}


def _print_json(label: str, payload: Any) -> None:
    print(f"{label}: {json.dumps(payload, indent=2, ensure_ascii=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    args = parser.parse_args()

    session_dir = args.session_dir.expanduser().resolve()
    latest = _latest_result(session_dir)
    print(f"SESSION_DIR={session_dir}")
    print(f"FILE={latest or ''}")
    if latest is None:
        raise SystemExit(1)

    data = _load_json(latest) or {}
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
    human = payload.get("human") if isinstance(payload.get("human"), dict) else {}
    robots = payload.get("robot_updates") if isinstance(payload.get("robot_updates"), list) else []
    visible_bones = _visible_human_bones(session_dir)
    pelvis_align = _visible_human_pelvis_align(session_dir)

    robot_summary: list[dict[str, Any]] = []
    for item in robots:
        if not isinstance(item, dict):
            continue
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        robot_summary.append(
            {
                "robot_id": item.get("robot_id"),
                "dof": item.get("dof"),
                "updated_links": result.get("updated_links"),
                "world_kind": result.get("world_kind"),
                "link0_label": result.get("link0_label"),
                "link0_actor_loc_cm": result.get("link0_actor_loc_cm"),
                "link_tip_label": result.get("link_tip_label"),
                "link_tip_actor_loc_cm": result.get("link_tip_actor_loc_cm"),
                "link_tip_delta_cm": result.get("link_tip_delta_cm"),
                "error": item.get("error"),
            }
        )

    human_summary = {
        "updated": human.get("updated"),
        "world_kind": human.get("world_kind"),
        "location_cm": human.get("location_cm"),
        "rotation_deg": human.get("rotation_deg"),
        "root_rotation_drive": human.get("root_rotation_drive"),
        "human_root_on": human.get("human_root_on"),
        "motion_frame_index": human.get("motion_frame_index"),
        "body_pose_bones_applied": human.get("body_pose_bones_applied"),
        "smpl_bone_preset": human.get("smpl_bone_preset"),
        "body_pose_missing_bones": human.get("body_pose_missing_bones"),
        "body_pose_warning": human.get("body_pose_warning"),
        "visible_human_bones_preset": visible_bones.get("preset"),
        "visible_human_bones_component": visible_bones.get("human_component"),
        "pelvis_align_applied": pelvis_align.get("applied"),
        "pelvis_align_bone": pelvis_align.get("bone"),
        "pelvis_align_relative_shift_cm": pelvis_align.get("relative_shift_cm"),
        "pelvis_align_reason": pelvis_align.get("reason"),
    }

    _print_json("ROBOT", robot_summary)
    _print_json("HUMAN", human_summary)


if __name__ == "__main__":
    main()
