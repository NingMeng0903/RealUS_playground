#!/usr/bin/env python3
"""Build collision-filtered capability map (avoids Robotic_Arm package init)."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "rm75_control"
sys.path.insert(0, str(REPO))
pkg = types.ModuleType("rm75_control")
pkg.__path__ = [str(PKG)]
sys.modules["rm75_control"] = pkg

from rm75_control.tools.reachability.build.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:] or [
        "--config",
        str(REPO / "configs/reachability/rm75_6f_3cm_15deg_coll.yaml"),
    ]
    raise SystemExit(main(argv))
