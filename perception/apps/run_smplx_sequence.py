#!/usr/bin/env python3
"""Unified RealUS continuous or recorded-RGB DWPose/EasyMocap SMPL-X entrypoint.

Without ``--recording-dir`` this preserves the live interval sampler from
``run_delayed_smplx_sequence``.  With ``--recording-dir`` it decodes every
four-camera group in the benchmark recorder's ``manifest.jsonl`` order and
fits each saved JPEG group without connecting to camera ingress.

Examples::

  $PY perception/apps/run_smplx_sequence.py \\
      --config configs/tracking/realus_ue_dwpose_easymocap.yaml \\
      --recording-dir outputs/ue_benchmark/session_01 \\
      --publish-genesis

  $PY perception/apps/run_smplx_sequence.py \\
      --config configs/tracking/realus_ue_dwpose_easymocap.yaml \\
      --duration-s 30 --interval-s 0.5
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    original = sys.argv
    sys.argv = ["run_delayed_smplx_sequence", *original[1:]]
    try:
        runpy.run_module(
            "projects.genesis_ue_sync.multiview_realtime.cli.run_delayed_smplx_sequence",
            run_name="__main__",
        )
    finally:
        sys.argv = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
