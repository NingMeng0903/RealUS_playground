"""Stage 4: Orbbec RGB chessboard intrinsics (distortion for hand-eye PnP)."""
from __future__ import annotations

import traceback
from collections.abc import Callable

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from multicam_calib.calib.intrinsics import ChessboardCaptures, ChessboardConfig, persist_intrinsics
from multicam_calib.calib.orbbec_handeye import FACTORY_ORBBEC_COLOR_FX, orbbec_fx_compare_text
from multicam_calib.calib.orbbec_rgbd import build_undistort_maps, remap_like
from multicam_calib.devices.orbbec import OrbbecRGBDSession
from multicam_calib.io.config import OrbbecConfig
from multicam_calib.io.results import Intrinsics


class _SolveWorker(QThread):
    finished_ok = pyqtSignal(object, float)
    finished_err = pyqtSignal(str)

    def __init__(self, captures: ChessboardCaptures) -> None:
        super().__init__()
        self.captures = captures

    def run(self) -> None:  # noqa: D401
        try:
            intr, rms = self.captures.solve()
        except Exception:  # noqa: BLE001
            self.finished_err.emit(traceback.format_exc())
        else:
            self.finished_ok.emit(intr, rms)


class Stage4OrbbecIntrinsicsPanel(QWidget):
    def __init__(
        self,
        *,
        cfg: OrbbecConfig,
        session: OrbbecRGBDSession,
        on_intrinsics_saved: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._session = session
        self._on_intrinsics_saved = on_intrinsics_saved
        self._captures = ChessboardCaptures(cfg=ChessboardConfig(cols=11, rows=8, square_size_m=0.015))
        self._worker: _SolveWorker | None = None
        self._active = False
        self._solved: Intrinsics | None = None
        self._maps = None
        self._maps_key = None

        root = QVBoxLayout(self)

        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Inner corners cols:"))
        self._cols_sp = QSpinBox()
        self._cols_sp.setRange(2, 40)
        self._cols_sp.setValue(11)
        cfg_row.addWidget(self._cols_sp)
        cfg_row.addWidget(QLabel("rows:"))
        self._rows_sp = QSpinBox()
        self._rows_sp.setRange(2, 40)
        self._rows_sp.setValue(8)
        cfg_row.addWidget(self._rows_sp)
        cfg_row.addWidget(QLabel("square (m):"))
        self._sq_sp = QDoubleSpinBox()
        self._sq_sp.setRange(0.001, 1.0)
        self._sq_sp.setDecimals(4)
        self._sq_sp.setSingleStep(0.001)
        self._sq_sp.setValue(0.015)
        cfg_row.addWidget(self._sq_sp)
        cfg_row.addStretch(1)
        root.addLayout(cfg_row)

        previews = QHBoxLayout()
        self._lab_raw = QLabel("RGB + chessboard")
        self._lab_undist = QLabel("Undistort preview")
        for lab in (self._lab_raw, self._lab_undist):
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet("background: #000; color: #aaa;")
            lab.setMinimumSize(240, 240)
            lab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        previews.addWidget(self._lab_raw, 1)
        previews.addWidget(self._lab_undist, 1)
        root.addLayout(previews, 5)

        btns = QHBoxLayout()
        self._btn_open = QPushButton("Open Orbbec")
        self._btn_capture = QPushButton("Capture chessboard")
        self._btn_reset = QPushButton("Reset")
        self._btn_solve = QPushButton("Solve & save")
        self._btn_capture.setEnabled(False)
        self._btn_solve.setEnabled(False)
        btns.addWidget(self._btn_open)
        btns.addWidget(self._btn_capture)
        btns.addWidget(self._btn_reset)
        btns.addStretch(1)
        btns.addWidget(self._btn_solve)
        root.addLayout(btns)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        root.addWidget(self._log, 0)
        self._status = QLabel("")
        root.addWidget(self._status)

        self._btn_open.clicked.connect(self._on_open)
        self._btn_capture.clicked.connect(self._on_capture)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_solve.clicked.connect(self._on_solve)

        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._refresh)
        self._sync_open_buttons()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if active and self._session.is_open:
            self._timer.start()
            self._sync_open_buttons()
        else:
            self._timer.stop()

    def shutdown(self) -> None:
        self._timer.stop()

    def _board_cfg(self) -> ChessboardConfig:
        return ChessboardConfig(
            cols=int(self._cols_sp.value()),
            rows=int(self._rows_sp.value()),
            square_size_m=float(self._sq_sp.value()),
        )

    def _sync_open_buttons(self) -> None:
        opened = self._session.is_open
        self._btn_capture.setEnabled(opened)
        self._btn_open.setEnabled(not opened)
        if opened and self._session.params is not None:
            color = self._session.params.color
            factory_fx = (
                float(color.K[0, 0]) if str(color.source) == "factory" else FACTORY_ORBBEC_COLOR_FX
            )
            self._status.setText(
                f"{color.source} fx={color.K[0, 0]:.1f} fy={color.K[1, 1]:.1f}  "
                f"{orbbec_fx_compare_text(factory_fx=factory_fx)}  "
                f"captures={self._captures.num_captures()}"
            )

    def _on_open(self) -> None:
        try:
            if not self._session.is_open:
                params = self._session.open()
                self._log.append(
                    f"Opened {params.model} serial={params.serial or '(default)'}  "
                    f"backend={self._session.backend}  "
                    f"{params.color.image_size[0]}x{params.color.image_size[1]}  "
                    f"K source={params.color.source} fx={params.color.K[0, 0]:.1f}"
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Open Orbbec", str(exc))
            self._log.append(traceback.format_exc())
            return
        self._sync_open_buttons()
        if self._active:
            self._timer.start()

    def _refresh(self) -> None:
        if not self._session.is_open:
            return
        try:
            frame = self._session.read(timeout_ms=200)
        except Exception:  # noqa: BLE001
            return
        img = frame.color_bgr
        cfg = self._board_cfg()
        h, w = img.shape[:2]
        tw = min(960, w)
        if tw < w:
            thumb = cv2.resize(img, (tw, max(1, int(round(h * tw / w)))), interpolation=cv2.INTER_AREA)
        else:
            thumb = img.copy()
        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, (cfg.cols, cfg.rows))
        if found:
            cv2.drawChessboardCorners(thumb, (cfg.cols, cfg.rows), corners, found)
        _set_pix(self._lab_raw, thumb)
        model = None
        if self._solved is not None:
            from multicam_calib.calib.orbbec_rgbd import PinholeModel

            model = PinholeModel(self._solved.K, self._solved.dist, self._solved.image_size, "chessboard")
        elif self._session.params is not None:
            model = self._session.params.color
        if model is not None:
            key = (model.source, tuple(model.image_size), float(model.K[0, 0]), float(model.K[1, 1]))
            if self._maps is None or self._maps_key != key:
                self._maps = build_undistort_maps(model)
                self._maps_key = key
            _set_pix(self._lab_undist, remap_like(frame.color_bgr, self._maps))
        factory_fx = FACTORY_ORBBEC_COLOR_FX
        if self._session.params is not None and str(self._session.params.color.source) == "factory":
            factory_fx = float(self._session.params.color.K[0, 0])
        self._status.setText(
            f"preview only  capture={w}x{h}  "
            f"captures={self._captures.num_captures()}  chessboard={'yes' if found else 'no'}  "
            f"{orbbec_fx_compare_text(factory_fx=factory_fx)}"
        )

    def _on_capture(self) -> None:
        if not self._session.is_open:
            return
        try:
            frame = self._session.read(timeout_ms=800)
        except Exception as exc:  # noqa: BLE001
            self._log.append(f"capture failed: {exc}")
            return
        cfg = self._board_cfg()
        if self._captures.cfg != cfg:
            self._captures = ChessboardCaptures(cfg=cfg)
        try:
            ok = self._captures.try_add(frame.color_bgr)
        except Exception as exc:  # noqa: BLE001
            self._log.append(str(exc))
            return
        if ok:
            h, w = frame.color_bgr.shape[:2]
            self._log.append(f"captured #{self._captures.num_captures()}  {w}x{h}")
        else:
            self._log.append("chessboard not detected")
        self._btn_solve.setEnabled(self._captures.num_captures() >= 8)

    def _on_reset(self) -> None:
        self._captures = ChessboardCaptures(cfg=self._board_cfg())
        self._solved = None
        self._btn_solve.setEnabled(False)
        self._log.append("reset captures")

    def _on_solve(self) -> None:
        if self._captures.num_captures() < 8:
            QMessageBox.warning(self, "Not enough", "Need at least 8 chessboard captures.")
            return
        self._btn_solve.setEnabled(False)
        self._worker = _SolveWorker(self._captures)
        self._worker.finished_ok.connect(self._on_solve_ok)
        self._worker.finished_err.connect(self._on_solve_err)
        self._worker.start()

    def _on_solve_ok(self, intr: Intrinsics, rms: float) -> None:
        self._btn_solve.setEnabled(True)
        persist_intrinsics("orbbec", intr)
        self._solved = intr
        if self._on_intrinsics_saved is not None:
            self._on_intrinsics_saved()
        K = intr.K
        self._log.append(
            f"saved alias=orbbec  RMS={rms:.4f} px  "
            f"image_size={intr.image_size[0]}x{intr.image_size[1]}  "
            f"fx={K[0, 0]:.2f} fy={K[1, 1]:.2f}  → intrinsics.yaml"
        )

    def _on_solve_err(self, err: str) -> None:
        self._btn_solve.setEnabled(True)
        self._log.append("Solve FAILED:\n" + err)


def _set_pix(label: QLabel, bgr: np.ndarray) -> None:
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
