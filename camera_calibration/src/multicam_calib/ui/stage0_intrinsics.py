"""Stage 0 UI panel: per-camera chessboard intrinsics.

Each camera is calibrated independently. The user picks a camera from a
dropdown, sees a live preview, presses "Capture" whenever the chessboard is
detected. After enough captures ("Solve" enabled at N>=8) the result is
persisted to `calibration_results/intrinsics.yaml`.
"""
from __future__ import annotations

import traceback
from collections.abc import Callable

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from multicam_calib.calib.intrinsics import ChessboardCaptures, ChessboardConfig, persist_intrinsics
from multicam_calib.recording.sync import CameraStreamThread


class _SolveWorker(QThread):
    finished_ok = pyqtSignal(object, float)  # (Intrinsics, rms)
    finished_err = pyqtSignal(str)

    def __init__(self, captures: ChessboardCaptures) -> None:
        super().__init__()
        self.captures = captures

    def run(self) -> None:  # noqa: D401
        try:
            intr, rms = self.captures.solve()
        except Exception:
            self.finished_err.emit(traceback.format_exc())
        else:
            self.finished_ok.emit(intr, rms)


class Stage0Panel(QWidget):
    def __init__(
        self,
        *,
        aliases: list[str],
        streams: dict[str, CameraStreamThread],
        on_intrinsics_saved: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._streams = streams
        self._on_intrinsics_saved = on_intrinsics_saved
        self._captures: dict[str, ChessboardCaptures] = {}
        self._cfg = ChessboardConfig(cols=11, rows=8, square_size_m=0.015)
        self._worker: _SolveWorker | None = None

        root = QVBoxLayout(self)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Camera:"))
        self._alias_cb = QComboBox()
        self._alias_cb.addItems(aliases)
        cfg_row.addWidget(self._alias_cb)
        cfg_row.addSpacing(20)

        cfg_row.addWidget(QLabel("Inner corners cols:"))
        self._cols_sp = QSpinBox(); self._cols_sp.setRange(2, 40); self._cols_sp.setValue(self._cfg.cols)
        cfg_row.addWidget(self._cols_sp)
        cfg_row.addWidget(QLabel("rows:"))
        self._rows_sp = QSpinBox(); self._rows_sp.setRange(2, 40); self._rows_sp.setValue(self._cfg.rows)
        cfg_row.addWidget(self._rows_sp)
        cfg_row.addWidget(QLabel("square (m):"))
        self._sq_sp = QDoubleSpinBox(); self._sq_sp.setRange(0.001, 1.0); self._sq_sp.setDecimals(4)
        self._sq_sp.setSingleStep(0.001); self._sq_sp.setValue(self._cfg.square_size_m)
        cfg_row.addWidget(self._sq_sp)
        cfg_row.addStretch(1)
        root.addLayout(cfg_row)

        # Live preview area for the selected camera.
        self._preview = QLabel(); self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setStyleSheet("background: #000;")
        self._preview.setMinimumHeight(400)
        root.addWidget(self._preview, 1)

        controls = QHBoxLayout()
        self._btn_capture = QPushButton("Capture chessboard")
        self._btn_capture.setShortcut("Space")
        self._btn_reset = QPushButton("Reset this camera")
        self._btn_solve = QPushButton("Solve & save")
        controls.addWidget(self._btn_capture)
        controls.addWidget(self._btn_reset)
        controls.addStretch(1)
        controls.addWidget(self._btn_solve)
        root.addLayout(controls)

        self._log = QTextEdit(); self._log.setReadOnly(True)
        root.addWidget(self._log, 1)
        self._status = QLabel("")
        root.addWidget(self._status)

        self._btn_capture.clicked.connect(self._on_capture)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_solve.clicked.connect(self._on_solve)
        self._alias_cb.currentTextChanged.connect(lambda _: self._update_status())

        self._timer = QTimer(self); self._timer.setInterval(80)
        self._timer.timeout.connect(self._refresh_preview); self._timer.start()
        self._update_status()

    def set_active(self, active: bool) -> None:
        """Pause the chessboard-detection preview timer while this tab isn't visible.

        ``findChessboardCornersSB`` is expensive and prints noisy OpenCV
        warnings ("Matrix is singular") whenever the live frame doesn't show a
        real chessboard — which is always true while another tab (e.g. the
        AprilTag board on Stage 1/2) is active. Only run it when the user can
        actually see this tab.
        """
        if active:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def _current_captures(self) -> ChessboardCaptures:
        alias = self._alias_cb.currentText()
        # Reconfigure if the user tweaked spinboxes.
        cfg = ChessboardConfig(cols=int(self._cols_sp.value()), rows=int(self._rows_sp.value()), square_size_m=float(self._sq_sp.value()))
        cap = self._captures.get(alias)
        if cap is None or (cap.cfg != cfg):
            self._captures[alias] = ChessboardCaptures(cfg=cfg)
        return self._captures[alias]

    def _refresh_preview(self) -> None:
        alias = self._alias_cb.currentText()
        stream = self._streams.get(alias)
        if stream is None:
            return
        frame = stream.latest()
        if frame is None:
            return
        img = frame.image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cap = self._current_captures()
        found, corners = cv2.findChessboardCornersSB(
            gray, (cap.cfg.cols, cap.cfg.rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        overlay_color = (0, 255, 0) if found else (0, 165, 255)
        if found:
            cv2.drawChessboardCorners(img, (cap.cfg.cols, cap.cfg.rows), corners, found)
        # Downscale for the label.
        h, w = img.shape[:2]
        target_w = min(960, w)
        target_h = int(round(h * target_w / w))
        thumb = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self._preview.setPixmap(QPixmap.fromImage(qimg.copy()))

    def _update_status(self) -> None:
        alias = self._alias_cb.currentText()
        cap = self._captures.get(alias)
        n = cap.num_captures() if cap else 0
        self._status.setText(f"{alias}: {n} chessboard captures")

    def _on_capture(self) -> None:
        alias = self._alias_cb.currentText()
        stream = self._streams.get(alias)
        if stream is None:
            return
        frame = stream.latest()
        if frame is None:
            self._log.append(f"{alias}: no frame available.")
            return
        cap = self._current_captures()
        try:
            ok = cap.try_add(frame.image)
        except Exception as exc:  # noqa: BLE001
            self._log.append(f"{alias}: capture error — {exc}")
            return
        if ok:
            self._log.append(f"{alias}: captured chessboard #{cap.num_captures()}")
        else:
            self._log.append(f"{alias}: chessboard not detected in this frame.")
        self._update_status()

    def _on_reset(self) -> None:
        alias = self._alias_cb.currentText()
        self._captures.pop(alias, None)
        self._log.append(f"{alias}: reset captures.")
        self._update_status()

    def _on_solve(self) -> None:
        alias = self._alias_cb.currentText()
        cap = self._captures.get(alias)
        if cap is None or cap.num_captures() < 8:
            QMessageBox.warning(self, "Not enough captures", "Need at least 8 chessboard captures.")
            return
        self._btn_solve.setEnabled(False)
        self._log.append(f"{alias}: solving intrinsics from {cap.num_captures()} captures ...")
        self._worker = _SolveWorker(cap)
        self._worker.finished_ok.connect(lambda intr, rms, a=alias: self._on_solve_ok(a, intr, rms))
        self._worker.finished_err.connect(self._on_solve_err)
        self._worker.start()

    def _on_solve_ok(self, alias, intr, rms) -> None:
        self._btn_solve.setEnabled(True)
        persist_intrinsics(alias, intr)
        if self._on_intrinsics_saved is not None:
            self._on_intrinsics_saved()
        K = intr.K
        self._log.append(
            f"{alias}: solve OK — RMS reprojection {rms:.4f} px. "
            f"fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}. "
            "Saved to intrinsics.yaml."
        )

    def _on_solve_err(self, err: str) -> None:
        self._btn_solve.setEnabled(True)
        self._log.append("Solve FAILED:\n" + err)
