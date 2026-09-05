#!/usr/bin/env python3
"""Offline first-fault ablation.  Never treat brake qdot as a QP solution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rm75_control.control.joint_admittance_8dof.solver.fault_snapshot import (
    ablate_residual_qp1,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("snapshot", type=Path)
    p.add_argument(
        "--drop",
        default="",
        help="comma list: preview,hold,cbf,inset,j4_design",
    )
    args = p.parse_args()
    snap = json.loads(args.snapshot.read_text())
    drop = tuple(x.strip() for x in str(args.drop).split(",") if x.strip())
    print(json.dumps(ablate_residual_qp1(snap, drop=drop), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
