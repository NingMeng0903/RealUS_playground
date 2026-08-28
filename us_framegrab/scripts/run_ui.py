#!/usr/bin/env python3
"""Launch the ultrasound HDMI crop UI (or a headless ZMQ publisher).

Isolate this process from ``~/.local`` so PyQt5/cv2 come from the camera_calib
env. Create ``QApplication`` before importing anything that pulls in OpenCV.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import site
import sys
from pathlib import Path


if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

_user_site = site.getusersitepackages()
sys.path = [p for p in sys.path if not p.startswith(_user_site)]

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "src"))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config (default: us_framegrab/configs/config.yaml)",
    )
    p.add_argument("--headless", action="store_true", help="Publish only; no PyQt window")
    p.add_argument("--pub-bind", type=str, default="", help="Override ZMQ bind")
    p.add_argument("--no-preview-topic", action="store_true", help="Do not publish preview topic")
    p.add_argument(
        "--no-auto-crop-on-startup",
        action="store_true",
        help="Skip the one-shot brightness auto-crop on the first frame",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from us_framegrab.config import load_config

    cfg = load_config(args.config)
    if args.pub_bind:
        cfg.pub_bind = str(args.pub_bind)
    if args.no_preview_topic:
        cfg.preview_topic = ""
    auto = False if args.no_auto_crop_on_startup else cfg.auto_crop_on_startup

    if args.headless:
        from us_framegrab.runtime import run_headless

        return run_headless(cfg, auto_crop_on_startup=auto)

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)  # noqa: F841 — must live for process lifetime
    from us_framegrab.ui.app import run_ui

    return run_ui(cfg, auto_crop_on_startup=auto)


if __name__ == "__main__":
    raise SystemExit(main())
