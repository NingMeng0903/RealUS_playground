"""Application entrypoint — discovers cameras, loads configs, and spins up the UI."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from typing import Sequence

from PyQt5.QtWidgets import QApplication, QMessageBox

from multicam_calib.board.apriltag_board import build_board_geometry
from multicam_calib.devices.discovery import open_all, resolve_roster
from multicam_calib.io.config import load_app, load_board, load_camera_roster
from multicam_calib.ui.main_window import MainWindow


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zmq-connect",
        type=str,
        default="",
        help="Use shared RealSense publisher for Stage 1/2 (tcp://127.0.0.1:17356). "
        "Run run_realsense_camera_publisher.py first. Stage 0 still needs local USB.",
    )
    return p.parse_args(list(argv or sys.argv[1:]))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    app = QApplication.instance()
    if app is None:
        app = QApplication(list(argv or sys.argv))

    board_cfg = load_board()
    board_geom = build_board_geometry(board_cfg)
    app_cfg = load_app()
    zmq_connect = str(args.zmq_connect or app_cfg.preview.zmq_connect or "").strip()
    zmq_mode = app_cfg.preview.source.strip().lower() == "zmq" or bool(zmq_connect)
    if zmq_connect:
        app_cfg = replace(
            app_cfg,
            preview=replace(app_cfg.preview, source="zmq", zmq_connect=zmq_connect),
        )

    resolved = resolve_roster(mutate_config=not zmq_mode)
    online = [r for r in resolved if r.online]
    if not zmq_mode and not online:
        QMessageBox.critical(None, "No cameras", "No cameras detected. Plug in and try again.")
        return 1

    devices = {}
    zmq_hub = None
    if zmq_mode:
        if not zmq_connect:
            QMessageBox.critical(None, "ZMQ preview", "--zmq-connect is required for zmq preview mode.")
            return 1
        from multicam_calib.ingress.zmq_streams import ZmqMulticamHub

        roster = load_camera_roster()
        aliases = [e.alias for e in roster]
        if not aliases:
            aliases = [r.entry.alias for r in resolved if r.entry.alias]
        if not aliases:
            aliases = [f"cam{i}" for i in range(1, 5)]
        zmq_hub = ZmqMulticamHub(
            connect=zmq_connect,
            aliases=aliases,
            preview_topic=app_cfg.preview.zmq_preview_topic,
            capture_topic=app_cfg.preview.zmq_capture_topic,
        )
        zmq_hub.start()
    else:
        try:
            devices = open_all(
                resolved,
                width=app_cfg.stream.width,
                height=app_cfg.stream.height,
                fps=app_cfg.stream.fps,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(None, "Failed to open cameras", str(exc))
            return 2

    aliases = list(devices.keys()) if devices else list(zmq_hub.aliases if zmq_hub else [])
    win = MainWindow(
        aliases=aliases,
        devices=devices,
        board_geom=board_geom,
        app_cfg=app_cfg,
        zmq_hub=zmq_hub,
    )
    win.show()
    code = app.exec_()
    if zmq_hub is not None:
        zmq_hub.stop()
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
