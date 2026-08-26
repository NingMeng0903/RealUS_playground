#!/usr/bin/env python3
"""Headless Stage 3: open the Orbbec, build one colored cloud, write orbbec_rgbd.yaml."""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path

if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

_user_site = site.getusersitepackages()
sys.path = [p for p in sys.path if not p.startswith(_user_site)]

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from multicam_calib.calib.orbbec_rgbd import (  # noqa: E402
    DEFAULT_STAGE3_NOTES,
    OrbbecCheckReport,
    point_cloud_stats,
    save_orbbec_check,
)
from multicam_calib.devices.orbbec import OrbbecRGBDSession, diagnose_orbbec_usb  # noqa: E402
from multicam_calib.io.config import load_orbbec  # noqa: E402


def main() -> int:
    print(diagnose_orbbec_usb())
    cfg = load_orbbec()
    session = OrbbecRGBDSession(cfg)
    try:
        params = session.open()
        print(f"backend={session.backend} serial={params.serial!r} model={params.model}")
        if session.backend == "v4l2":
            frame = session.read(timeout_ms=4000)
            print(
                f"V4L2 RGB {frame.color_size[0]}x{frame.color_size[1]} — "
                "no depth on first-gen Gemini via pyorbbecsdk2"
            )
            return 0
        frame = session.read(timeout_ms=4000)
        xyz, _rgb, src = session.build_cloud(
            frame,
            min_m=cfg.min_depth_m,
            max_m=cfg.max_depth_m,
            min_valid=cfg.min_valid_points,
            min_valid_frac=cfg.min_valid_frac,
        )
        cloud = point_cloud_stats(
            xyz,
            min_m=cfg.min_depth_m,
            max_m=cfg.max_depth_m,
            min_valid=cfg.min_valid_points,
            min_valid_frac=cfg.min_valid_frac,
        )
        cloud.detail = src
        report = OrbbecCheckReport(
            serial=params.serial,
            model=params.model,
            color=params.color,
            depth=params.depth,
            T_color_depth=params.T_color_depth,
            cloud=cloud,
            align_mode=cfg.align,
            color_size=frame.color_size,
            depth_size=frame.depth_size,
            notes=list(DEFAULT_STAGE3_NOTES),
        )
        path = save_orbbec_check(report)
    finally:
        session.close()
    print(f"{'PASS' if cloud.ok else 'FAIL'}: {cloud.detail}")
    print(f"wrote {path}")
    return 0 if cloud.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
