"""Validate a DMP-to-arm-base task manifold mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ird_playground.traj.dmp_task import load_dmp_task_spec, load_task_tcp_poses


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args(argv)
    spec = load_dmp_task_spec(args.config)
    phase, T = load_task_tcp_poses(spec)
    out = {
        "trajectory_npz": str(spec.trajectory_npz),
        "samples": int(len(phase)),
        "phase_min": float(phase.min()),
        "phase_max": float(phase.max()),
        "arm_base_position_min_m": T[:, :3, 3].min(axis=0).tolist(),
        "arm_base_position_max_m": T[:, :3, 3].max(axis=0).tolist(),
        "tool_axis_convention": "tcp_plus_z",
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
