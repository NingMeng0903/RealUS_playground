"""Wire the frame-grab session to the PyQt window. QApplication must already exist."""

from __future__ import annotations

from PyQt5.QtWidgets import QApplication

from us_framegrab.config import FrameGrabConfig
from us_framegrab.runtime import FrameGrabSession
from us_framegrab.ui.main_window import MainWindow


def run_ui(cfg: FrameGrabConfig, *, auto_crop_on_startup: bool | None = None) -> int:
    flag = cfg.auto_crop_on_startup if auto_crop_on_startup is None else bool(auto_crop_on_startup)
    session = FrameGrabSession(cfg, auto_crop_on_startup=flag)
    session.start()
    win = MainWindow(session)
    win.show()
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication must be created before importing this module.")
    try:
        return int(app.exec_())
    finally:
        session.stop()
