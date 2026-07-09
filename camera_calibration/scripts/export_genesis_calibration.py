#!/usr/bin/env python3
"""Merge calibration_results/*.yaml into a single Genesis-ready bundle."""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path

if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

sys.path = [p for p in sys.path if not p.startswith(site.getusersitepackages())]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from multicam_calib.io.genesis_export import genesis_bundle_path, save_genesis_bundle  # noqa: E402


def main() -> int:
    out = save_genesis_bundle()
    print(f"Wrote {out}")
    meta = __import__("yaml").safe_load(out.read_text())
    bed = meta["bed"]
    print(
        f"  bed: {bed['size_m'][0]:.3f} x {bed['size_m'][1]:.3f} m, "
        f"z={bed['height_m']:.3f} m, rot={bed['rotation_deg']:.1f} deg"
    )
    print(f"  cameras: {', '.join(meta['cameras'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
