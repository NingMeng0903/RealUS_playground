#!/usr/bin/env python3
"""Compute rail_base spawn pose from slider_rail.yaml (base_link @ rail_y=0 calibration)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("rm75_control/rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml"),
    )
    args = parser.parse_args()

    from rm75_control.control.joint_admittance_8dof.param_model.generator import compute_layout, load_spec
    from rm75_control.control.joint_admittance_8dof.param_model.placement import entity_pose_from_calib, resolve_world_calib

    spec = load_spec(args.spec.resolve())
    layout = compute_layout(spec)
    calib = resolve_world_calib(spec, layout)
    pose = entity_pose_from_calib(calib)
    out = {
        "base_pos": [float(v) for v in pose["pos"]],
        "base_quat_wxyz": [float(v) for v in pose["quat_wxyz"]],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
