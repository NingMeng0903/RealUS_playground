"""Application entrypoint — discovers cameras, loads configs, and spins up the UI."""
from __future__ import annotations

import sys
from typing import Sequence

from PyQt5.QtWidgets import QApplication, QMessageBox

from multicam_calib.board.apriltag_board import build_board_geometry
from multicam_calib.devices.discovery import open_all, resolve_roster
from multicam_calib.io.config import load_app, load_board
from multicam_calib.ui.main_window import MainWindow


def main(argv: Sequence[str] | None = None) -> int:
    app = QApplication.instance()
    if app is None:
        app = QApplication(list(argv or sys.argv))

    board_cfg = load_board()
    board_geom = build_board_geometry(board_cfg)
    app_cfg = load_app()

    resolved = resolve_roster(mutate_config=True)
    online = [r for r in resolved if r.online]
    if not online:
        QMessageBox.critical(None, "No cameras", "No cameras detected. Plug in and try again.")
        return 1

    try:
        devices = open_all(resolved, width=app_cfg.stream.width, height=app_cfg.stream.height, fps=app_cfg.stream.fps)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(None, "Failed to open cameras", str(exc))
        return 2

    aliases = list(devices.keys())
    win = MainWindow(aliases=aliases, devices=devices, board_geom=board_geom, app_cfg=app_cfg)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
