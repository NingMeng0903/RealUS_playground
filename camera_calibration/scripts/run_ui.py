#!/usr/bin/env python3
"""Launch the multi-camera calibration UI.

We isolate this process from ``~/.local/lib/python3.10/site-packages`` (which
holds a legacy PyQt5 install and a NumPy-1.x-linked cv2 shared by the whole
system) so imports come only from the ``camera_calib`` conda env.
"""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path


if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

# Purge any user-site path from sys.path that Python already prepended.
_user_site = site.getusersitepackages()
sys.path = [p for p in sys.path if not p.startswith(_user_site)]

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# opencv-python bundles its own Qt platform plugins. If cv2 is imported before
# PyQt5 initialises, Qt tries to load xcb from cv2/qt/plugins and the UI
# crashes. Create QApplication first, then import anything that pulls in cv2.
from PyQt5.QtWidgets import QApplication  # noqa: E402

_qt_app = QApplication(sys.argv)  # noqa: F841 — must live for process lifetime

from multicam_calib.ui.app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
