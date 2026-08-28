"""The UI entrypoint must create QApplication before any cv2 import."""

from __future__ import annotations

import unittest
from pathlib import Path

_RUN_UI = Path(__file__).resolve().parents[1] / "scripts" / "run_ui.py"


class TestQtBeforeCv2(unittest.TestCase):
    def test_run_ui_imports_qapplication_before_ui_or_runtime(self) -> None:
        source = _RUN_UI.read_text(encoding="utf-8")
        self.assertNotIn("import cv2", source)
        self.assertNotIn("from cv2", source)
        qapp_idx = source.find("from PyQt5.QtWidgets import QApplication")
        run_ui_idx = source.find("from us_framegrab.ui.app import run_ui")
        self.assertGreater(qapp_idx, 0)
        self.assertGreater(run_ui_idx, qapp_idx)


if __name__ == "__main__":
    unittest.main()
