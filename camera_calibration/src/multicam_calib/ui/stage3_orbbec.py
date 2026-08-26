"""Stage 3 UI: Orbbec factory undistort + D2C / point-cloud check."""
from __future__ import annotations

import traceback

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from multicam_calib.calib.orbbec_rgbd import (
    DEFAULT_STAGE3_NOTES,
    OrbbecCheckReport,
    UndistortMaps,
    build_undistort_maps,
    point_cloud_stats,
    preview_mosaic,
    save_orbbec_check,
)
from multicam_calib.devices.orbbec import (
    OrbbecRGBDSession,
    diagnose_orbbec_usb,
    discover_orbbec,
)
from multicam_calib.io.config import OrbbecConfig


class _CheckWorker(QThread):
    finished_ok = pyqtSignal(object, str)  # report, path
    finished_err = pyqtSignal(str)

    def __init__(self, session: OrbbecRGBDSession, cfg: OrbbecConfig) -> None:
        super().__init__()
        self.session = session
        self.cfg = cfg

    def run(self) -> None:  # noqa: D401
        try:
            frame = self.session.read(timeout_ms=3000)
            params = self.session.params
            if params is None:
                raise RuntimeError("Orbbec factory parameters missing")
            xyz, _rgb, src = self.session.build_cloud(
                frame,
                min_m=self.cfg.min_depth_m,
                max_m=self.cfg.max_depth_m,
                min_valid=self.cfg.min_valid_points,
                min_valid_frac=self.cfg.min_valid_frac,
            )
            cloud = point_cloud_stats(
                xyz,
                min_m=self.cfg.min_depth_m,
                max_m=self.cfg.max_depth_m,
                min_valid=self.cfg.min_valid_points,
                min_valid_frac=self.cfg.min_valid_frac,
            )
            cloud.detail = f"{src}"
            report = OrbbecCheckReport(
                serial=params.serial,
                model=params.model,
                color=params.color,
                depth=params.depth,
                T_color_depth=params.T_color_depth,
                cloud=cloud,
                align_mode=self.cfg.align,
                color_size=frame.color_size,
                depth_size=frame.depth_size,
                notes=list(DEFAULT_STAGE3_NOTES),
            )
            path = save_orbbec_check(report)
        except Exception:  # noqa: BLE001
            self.finished_err.emit(traceback.format_exc())
        else:
            self.finished_ok.emit(report, str(path))


class Stage3OrbbecPanel(QWidget):
    def __init__(
        self,
        *,
        cfg: OrbbecConfig,
        session: OrbbecRGBDSession | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._session = session or OrbbecRGBDSession(cfg)
        self._maps: UndistortMaps | None = None
        self._worker: _CheckWorker | None = None
        self._active = False
        self._latest_views: dict[str, np.ndarray] = {}

        root = QVBoxLayout(self)

        row = QHBoxLayout()
        self._btn_open = QPushButton("Open Orbbec")
        self._btn_close = QPushButton("Close")
        self._btn_check = QPushButton("Run check & save")
        self._btn_close.setEnabled(False)
        self._btn_check.setEnabled(False)
        row.addWidget(self._btn_open)
        row.addWidget(self._btn_close)
        row.addWidget(self._btn_check)
        row.addSpacing(16)
        row.addWidget(QLabel("Overlay:"))
        self._mode_cb = QComboBox()
        self._mode_cb.addItem("Raw", "raw_d2c")
        self._mode_cb.addItem("RGB undistort", "undistort_rgb_only")
        self._mode_cb.addItem("Both", "undistort_both")
        row.addWidget(self._mode_cb)
        row.addStretch(1)
        root.addLayout(row)

        grid = QHBoxLayout()
        self._lab_color = _preview_label("RGB raw")
        self._lab_undist = _preview_label("RGB undistorted")
        self._lab_depth = _preview_label("Depth (colormap)")
        self._lab_overlay = _preview_label("Overlay")
        col_a = QVBoxLayout()
        col_a.addWidget(self._lab_color, 1)
        col_a.addWidget(self._lab_depth, 1)
        col_b = QVBoxLayout()
        col_b.addWidget(self._lab_undist, 1)
        col_b.addWidget(self._lab_overlay, 1)
        grid.addLayout(col_a, 1)
        grid.addLayout(col_b, 1)
        root.addLayout(grid, 5)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        root.addWidget(self._log, 0)
        self._status = QLabel("")
        root.addWidget(self._status)

        self._btn_open.clicked.connect(self._on_open)
        self._btn_close.clicked.connect(self._on_close)
        self._btn_check.clicked.connect(self._on_check)

        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._refresh)
        self._log_sdk_state()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if active and self._session.is_open and not self._timer.isActive():
            self._timer.start()
        if not active:
            self._timer.stop()

    def shutdown(self) -> None:
        self._timer.stop()

    def _log_sdk_state(self) -> None:
        found = list(discover_orbbec())
        if found:
            self._status.setText(f"{found[0].serial}  ready")
            return
        diag = diagnose_orbbec_usb()
        if "DENIED" in diag:
            self._status.setText("USB permission denied")
        elif "PID" in diag:
            self._status.setText("Orbbec USB present")
        else:
            self._status.setText("No Orbbec USB")

    def _on_open(self) -> None:
        self._btn_open.setEnabled(False)
        try:
            params = self._session.params if self._session.is_open else self._session.open()
            if params is None:
                params = self._session.open()
            self._maps = build_undistort_maps(params.color)
        except Exception as exc:  # noqa: BLE001
            self._session.close()
            self._btn_open.setEnabled(True)
            self._log.append(traceback.format_exc())
            QMessageBox.critical(self, "Open Orbbec", str(exc))
            return
        K = params.color.K
        self._log.append(
            f"{params.model}  {params.serial}  {self._session.backend}  "
            f"fx={K[0, 0]:.1f}  depth={'yes' if self._session.has_depth else 'no'}"
        )
        self._btn_close.setEnabled(True)
        self._btn_check.setEnabled(self._session.has_depth)
        if self._active:
            self._timer.start()
        self._status.setText(
            "streaming"
            if self._session.has_depth
            else "RGB only (no depth)"
        )

    def _on_close(self) -> None:
        self._timer.stop()
        self._session.close()
        self._maps = None
        self._btn_open.setEnabled(True)
        self._btn_close.setEnabled(False)
        self._btn_check.setEnabled(False)
        self._log.append("closed")
        self._status.setText("closed")

    def _on_check(self) -> None:
        if not self._session.is_open:
            return
        self._timer.stop()
        self._btn_check.setEnabled(False)
        self._log.append("Capturing one RGB-D frame and building a point cloud …")
        self._worker = _CheckWorker(self._session, self._cfg)
        self._worker.finished_ok.connect(self._on_check_ok)
        self._worker.finished_err.connect(self._on_check_err)
        self._worker.start()

    def _on_check_ok(self, report: OrbbecCheckReport, path: str) -> None:
        self._btn_check.setEnabled(True)
        if self._active and self._session.is_open:
            self._timer.start()
        cloud = report.cloud
        flag = "PASS" if cloud.ok else "FAIL"
        self._log.append(
            f"{flag}: {cloud.detail}  z∈[{cloud.z_min_m:.2f},{cloud.z_max_m:.2f}] m  "
            f"median={cloud.z_median_m:.2f} m\nSaved {path}"
        )
        self._status.setText(f"Check {flag} — {path}")
        if not cloud.ok:
            QMessageBox.warning(self, "Point cloud check", cloud.detail)

    def _on_check_err(self, err: str) -> None:
        self._btn_check.setEnabled(True)
        if self._active and self._session.is_open:
            self._timer.start()
        self._log.append("Check FAILED:\n" + err)

    def _refresh(self) -> None:
        if not self._session.is_open or self._maps is None:
            return
        try:
            frame = self._session.read(timeout_ms=200)
        except Exception:  # noqa: BLE001
            return
        mode = str(self._mode_cb.currentData() or "raw_d2c")
        views = preview_mosaic(
            frame.color_bgr,
            frame.depth_m,
            self._maps,
            mode=mode,  # type: ignore[arg-type]
            min_depth_m=self._cfg.min_depth_m,
            max_depth_m=self._cfg.max_depth_m,
            overlay_alpha=self._cfg.overlay_alpha,
        )
        self._latest_views = views
        _set_preview(self._lab_color, views["color"])
        _set_preview(self._lab_undist, views["color_undistorted"])
        _set_preview(self._lab_depth, views["depth"])
        _set_preview(self._lab_overlay, views["overlay"])
        self._status.setText(
            f"{frame.color_size[0]}x{frame.color_size[1]}  mode={mode}  "
            f"depth finite={int(np.isfinite(frame.depth_m).sum())}"
        )


def _preview_label(title: str) -> QLabel:
    lab = QLabel(title)
    lab.setAlignment(Qt.AlignCenter)
    lab.setStyleSheet("background: #000; color: #aaa;")
    lab.setMinimumSize(160, 120)
    lab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lab.setToolTip(title)
    return lab


def _set_preview(label: QLabel, bgr: np.ndarray) -> None:
    h, w = bgr.shape[:2]
    lw = max(2, int(label.width()) - 8)
    lh = max(2, int(label.height()) - 8)
    scale = min(lw / float(w), lh / float(h))
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))
    thumb = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    label.setPixmap(QPixmap.fromImage(qimg.copy()))
