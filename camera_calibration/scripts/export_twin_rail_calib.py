#!/usr/bin/env python3
"""Write ``slider_rail.calibrated.yaml`` from Stage 2 ``robot_world.yaml``.

The overlay only replaces ``world_calib.base_pos_m`` / ``base_quat_wxyz``
(``base_link`` world pose at ``rail_y = 0``). Viewer-only ``arm_mount`` and
``slider.top_to_rail_bottom_mm`` stay in the hand-authored spec.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from multicam_calib.io.results import load_robot_world, robot_world_path  # noqa: E402

DEFAULT_OUT = (
    ROOT.parent
    / "rm75_control"
    / "rm75_control"
    / "control"
    / "joint_admittance_8dof"
    / "config"
    / "slider_rail.calibrated.yaml"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot-world", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    payload = load_robot_world(args.robot_world or robot_world_path())
    if not payload:
        print("robot_world.yaml not found — run Stage 2 robot (and preferably corners) first.")
        return 1
    pos = payload.get("base_pos_m")
    quat = payload.get("base_quat_wxyz")
    if not pos or not quat:
        print("robot_world.yaml is missing base_pos_m / base_quat_wxyz.")
        return 2
    doc = {
        "slider_rail": {
            "world_calib": {
                "base_pos_m": [float(v) for v in pos],
                "base_quat_wxyz": [float(v) for v in quat],
            }
        }
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  base_pos_m     {doc['slider_rail']['world_calib']['base_pos_m']}")
    print(f"  base_quat_wxyz {doc['slider_rail']['world_calib']['base_quat_wxyz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
