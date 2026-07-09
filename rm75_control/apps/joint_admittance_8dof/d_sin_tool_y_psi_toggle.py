#!/usr/bin/env python3
"""Hybrid force-position hold @ D with swivel psi toggling (window C or standalone).

Move to pose D, then force-position hold (no Y sin scan). Swivel alternates
center -> left -> right -> left ... with quintic ramps.

  source env.sh
  # 1) Manually move arm to LEFT side configuration
  # 2) Run (window C; window A must be hot-wait):
  python apps/joint_admittance_8dof/d_sin_tool_y_psi_toggle.py \\
      --config configs/joint_admittance_8dof.yaml --enable-force --desired-z 1.0

Defaults: --scan-duration 300 --psi-toggle-period 10 --hybrid-hold-at-d
Requires force compensation calibration (see MD/command.md section 1).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _inject_defaults(argv: list[str]) -> list[str]:
    out = list(argv)
    pairs = (
        ("--scan-duration", "300"),
        ("--psi-toggle-period", "10"),
        ("--hybrid-hold-at-d",),
    )
    for item in pairs:
        if len(item) == 2:
            flag, val = item
            if flag not in out:
                out.extend([flag, val])
        else:
            flag = item[0]
            if flag not in out:
                out.append(flag)
    return out


def main() -> int:
    sys.argv = [sys.argv[0]] + _inject_defaults(sys.argv[1:])
    target = Path(__file__).resolve().parent / "d_sin_tool_y.py"
    spec = importlib.util.spec_from_file_location("_d_sin_tool_y", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return int(mod.main())


if __name__ == "__main__":
    raise SystemExit(main())
