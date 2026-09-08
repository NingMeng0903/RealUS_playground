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

# ``run_ui.py`` is also used directly from a shell, so the controller package
# is not necessarily installed in the camera environment.  Add the sibling
# repository checkout before importing the observer bootstrap.  Keep this
# helper dependency-free: it must run before PyQt, OpenCV, ffmpeg bindings, or
# any other camera worker can be imported.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RM75_ROOT = _REPO_ROOT / "rm75_control"


def _prepare_observer_process() -> None:
    if (_RM75_ROOT / "rm75_control").is_dir():
        rm75_root = str(_RM75_ROOT)
        if rm75_root not in sys.path:
            sys.path.insert(0, rm75_root)
    try:
        from rm75_control.control.admittance_common.observer_runtime import (
            prepare_observer_process,
        )
    except (ImportError, OSError) as exc:
        logging.getLogger(__name__).warning(
            "observer resource setup unavailable: %s", exc
        )
        return
    prepare_observer_process()
    logging.getLogger(__name__).info(
        "observer resources prepared: one numeric thread, background priority"
    )


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _prepare_observer_process()
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
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
