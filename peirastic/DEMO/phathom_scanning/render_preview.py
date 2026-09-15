#!/usr/bin/env python3
"""Render the dual-view phantom plan preview from a saved detect instant.

    python -m peirastic.DEMO.phathom_scanning.render_preview /path/to/run
    python -m peirastic.DEMO.phathom_scanning.render_preview /path/to/run --snapshot 2

Needs plan.json plus cloud_and_plan.npz (or capture.npz / snapshots/NNN_capture.npz).
Does not attach to the robot or camera.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from peirastic.DEMO.phathom_scanning.preview import load_preview_bundle, write_plan_preview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="run or detect-instant folder")
    parser.add_argument("--snapshot", type=int, default=None,
                        help="1-based index under snapshots/; default uses cloud_and_plan.npz")
    parser.add_argument("--output", type=Path, default=None,
                        help="preview PNG path; default <directory>/preview.png")
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    xyz, rgb, hit, plan, probe_width = load_preview_bundle(directory, snapshot=args.snapshot)
    output = args.output or directory / "preview.png"
    written = write_plan_preview(
        output, xyz=xyz, rgb=rgb, hit=hit, plan=plan, probe_width_m=probe_width,
    )
    print(f"[OK] preview {written}  pdf={written.with_suffix('.pdf')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
