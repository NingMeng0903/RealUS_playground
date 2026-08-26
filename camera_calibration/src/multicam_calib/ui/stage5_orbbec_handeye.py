"""Stage 5: Orbbec color frame vs URDF link_7 (eye-in-hand)."""
from __future__ import annotations

import traceback

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from multicam_calib.board.apriltag_board import BoardGeometry
from multicam_calib.board.detector import AprilTagDetector, scale_detections
from multicam_calib.calib.orbbec_handeye import (
    load_orbbec_color_intrinsics,
    load_orbbec_handeye_captures,
    orbbec_handeye_captures_last_path,
    orbbec_handeye_captures_path,
    save_orbbec_handeye,
    save_orbbec_handeye_captures,
    solve_orbbec_handeye,
)
from multicam_calib.devices.orbbec import OrbbecRGBDSession
from multicam_calib.ingress.robot_state import RobotStateReader
from multicam_calib.io.config import OrbbecConfig, RobotConfig


class _SolveWorker(QThread):
    finished_ok = pyqtSignal(object, str)
    finished_err = pyqtSignal(str)

    def __init__(self, captures, board_geom, intrinsics, robot_cfg) -> None:
        super().__init__()
        self.captures = captures
        self.board_geom = board_geom
        self.intrinsics = intrinsics
        self.robot_cfg = robot_cfg

    def run(self) -> None:  # noqa: D401
        try:
            result = solve_orbbec_handeye(
                self.captures,
                board_geom=self.board_geom,
                intrinsics=self.intrinsics,
                robot_cfg=self.robot_cfg,
            )
            path = save_orbbec_handeye(result)
        except Exception:  # noqa: BLE001
            self.finished_err.emit(traceback.format_exc())
        else:
            self.finished_ok.emit(result, str(path))


class Stage5OrbbecHandeyePanel(QWidget):
    def __init__(
        self,
        *,
        cfg: OrbbecConfig,
        session: OrbbecRGBDSession,
        board_geom: BoardGeometry,
        detector: AprilTagDetector,
        robot_cfg: RobotConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._session = session
        self._board_geom = board_geom
        self._detector = detector
        self._robot_cfg = robot_cfg
        self._robot = RobotStateReader(robot_cfg.shm.name)
        self._captures: list[dict] = []
        self._worker: _SolveWorker | None = None
        self._active = False

        root = QVBoxLayout(self)

        body = QHBoxLayout()
        self._preview = QLabel("Orbbec RGB")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setStyleSheet("background: #000; color: #aaa;")
        self._preview.setMinimumSize(320, 240)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body.addWidget(self._preview, 4)
        side = QVBoxLayout()
        self._list = QListWidget()
        side.addWidget(self._list, 1)
        body.addLayout(side, 1)
        root.addLayout(body, 5)

        btns = QHBoxLayout()
        self._btn_open = QPushButton("640 + depth")
        self._btn_1080 = QPushButton("1080p RGB (no depth)")
        self._btn_capture = QPushButton("Capture")
        self._btn_delete = QPushButton("Delete selected")
        self._btn_clear = QPushButton("Clear")
        self._btn_load_last = QPushButton("Load last")
        self._btn_run = QPushButton("Solve T_link7_cam")
        self._btn_capture.setEnabled(False)
        self._btn_run.setEnabled(False)
        btns.addWidget(self._btn_open)
        btns.addWidget(self._btn_1080)
        btns.addWidget(self._btn_capture)
        btns.addWidget(self._btn_delete)
        btns.addWidget(self._btn_clear)
        btns.addWidget(self._btn_load_last)
        btns.addStretch(1)
        btns.addWidget(self._btn_run)
        root.addLayout(btns)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        root.addWidget(self._log, 0)
        self._status = QLabel("")
        root.addWidget(self._status)

        self._btn_open.clicked.connect(self._on_open_640)
        self._btn_1080.clicked.connect(self._on_open_1080)
        self._btn_capture.clicked.connect(self._on_capture)
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_load_last.clicked.connect(self._on_load_last)
        self._btn_run.clicked.connect(self._on_run)

        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._refresh)
        n_resume = self._load_from_path(orbbec_handeye_captures_path(), persist=False)
        if n_resume:
            self._log.append(
                f"resumed {n_resume} capture(s) from {orbbec_handeye_captures_path()}"
            )

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if active and self._session.is_open:
            self._btn_capture.setEnabled(True)
            self._timer.start()
        else:
            self._timer.stop()

    def shutdown(self) -> None:
        self._timer.stop()
        try:
            self._robot.close()
        except Exception:  # noqa: BLE001
            pass

    def _on_open_640(self) -> None:
        self._reopen(self._session.open, label="640 + depth")

    def _on_open_1080(self) -> None:
        self._reopen(lambda: self._session.open_rgb_only(1920, 1080), label="1080p RGB (no depth)")

    def _reopen(self, opener, *, label: str) -> None:
        self._timer.stop()
        try:
            params = opener()
        except Exception as exc:  # noqa: BLE001
            self._log.append(traceback.format_exc())
            try:
                params = self._session.open()
            except Exception:  # noqa: BLE001
                QMessageBox.critical(self, "Open Orbbec", str(exc))
                return
            QMessageBox.warning(
                self,
                "Open Orbbec",
                f"{exc}\n\nRestored 640 + depth. Stage 5 should stay on 640.",
            )
            label = "640 + depth (restored)"
        w, h = int(params.color.image_size[0]), int(params.color.image_size[1])
        sizes = {tuple(c.get("image_size") or ()) for c in self._captures}
        if self._captures and sizes != {(w, h)}:
            self._set_captures([])
            self._log.append(f"cleared captures (stream is now {w}x{h})")
        depth = "depth=off" if not self._session.has_depth else "depth=on"
        self._log.append(
            f"{label}: {params.model}  {w}x{h}  backend={self._session.backend}  {depth}  "
            f"K={params.color.source} fx={params.color.K[0, 0]:.1f}"
        )
        self._btn_capture.setEnabled(True)
        if self._active:
            self._timer.start()

    def _intrinsics(self):
        factory = None
        image_size = None
        if self._session.params is not None:
            factory = self._session.params.color.as_intrinsics()
            image_size = self._session.params.color.image_size
        return load_orbbec_color_intrinsics(factory=factory, image_size=image_size)

    def _refresh(self) -> None:
        if not self._session.is_open:
            return
        try:
            frame = self._session.read(timeout_ms=200)
        except Exception:  # noqa: BLE001
            return
        img = frame.color_bgr.copy()
        h, w = img.shape[:2]
        tw = min(960, w)
        if tw < w:
            thumb = cv2.resize(img, (tw, max(1, int(round(h * tw / w)))), interpolation=cv2.INTER_AREA)
            dets = scale_detections(
                self._detector.detect(thumb),
                from_wh=(thumb.shape[1], thumb.shape[0]),
                to_wh=(w, h),
            )
        else:
            dets = self._detector.detect(img)
        for d in dets:
            pts = d.corners.astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        _set_pix(self._preview, img)
        shm_ok, age = self._robot.is_fresh(self._robot_cfg.shm.max_age_s)
        src = "factory"
        try:
            src = str(self._intrinsics().source)
        except Exception:  # noqa: BLE001
            pass
        age_s = float("nan") if age is None else float(age)
        self._status.setText(
            f"preview only  capture={w}x{h}  "
            f"depth={'on' if self._session.has_depth else 'off'}  "
            f"tags={len(dets)}  samples={len(self._captures)}  "
            f"K={src}  SHM={'ok' if shm_ok else 'stale'} age={age_s:.2f}s"
        )

    def _on_capture(self) -> None:
        if not self._session.is_open:
            return
        rc = self._robot_cfg
        snap, still = self._robot.wait_still(
            window_s=rc.stillness.window_s,
            trans_m=rc.stillness.trans_m,
            rot_deg=rc.stillness.rot_deg,
            rail_m=rc.stillness.rail_m,
        )
        shm_ok, age = self._robot.is_fresh(rc.shm.max_age_s)
        if snap is None or not shm_ok or not still.ok:
            self._log.append(
                f"rejected: SHM={'ok' if shm_ok else f'stale {age:.2f}s'}  "
                f"{still.message or self._robot.last_error or 'no robot'}"
            )
            return
        try:
            frame = self._session.read(timeout_ms=800)
        except Exception as exc:  # noqa: BLE001
            self._log.append(str(exc))
            return
        h, w = frame.color_bgr.shape[:2]
        dets = self._detector.detect(frame.color_bgr)
        det_map = AprilTagDetector.detections_to_dict(dets)
        if len(det_map) < 8:
            self._log.append(f"rejected: only {len(det_map)} tags (need ≥8 on the large board)")
            return
        cap = {
            "detections": {int(k): v.tolist() for k, v in det_map.items()},
            "T_railbase_tcp": snap.T_railbase_tcp().tolist(),
            "rail_m": float(snap.rail_m),
            "q_deg": [float(v) for v in snap.q_deg.tolist()],
            "n_tags": len(det_map),
            "image_size": [w, h],
        }
        self._captures.append(cap)
        self._refresh_list()
        path = self._persist()
        self._log.append(
            f"captured #{len(self._captures)}  {w}x{h}  tags={len(det_map)}  saved {path.name}"
        )

    def _on_delete(self) -> None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._captures):
            return
        self._captures.pop(row)
        self._refresh_list()
        self._persist()
        self._log.append(f"deleted sample {row + 1}  ({len(self._captures)} left)")

    def _on_clear(self) -> None:
        self._set_captures([])
        self._log.append("cleared")

    def _refresh_list(self) -> None:
        self._list.clear()
        for i, cap in enumerate(self._captures):
            n = int(cap.get("n_tags") or len(cap.get("detections") or {}))
            rail = float(cap.get("rail_m") or 0.0)
            self._list.addItem(f"#{i + 1}  tags={n}  rail={rail:.3f} m")
        self._btn_run.setEnabled(len(self._captures) >= 6)

    def _set_captures(self, captures: list[dict]) -> None:
        self._captures = list(captures)
        self._refresh_list()
        self._persist()

    def _persist(self, *, also_last: bool = False):
        try:
            return save_orbbec_handeye_captures(self._captures, also_last=also_last)
        except Exception as exc:  # noqa: BLE001
            self._log.append(f"save captures FAILED: {exc}")
            return orbbec_handeye_captures_path()

    def _load_from_path(self, path, *, persist: bool) -> int:
        caps = load_orbbec_handeye_captures(path)
        if persist:
            self._set_captures(caps)
        else:
            self._captures = list(caps)
            self._refresh_list()
        return len(caps)

    def _on_load_last(self) -> None:
        last = orbbec_handeye_captures_last_path()
        working = orbbec_handeye_captures_path()
        src = last if last.is_file() else working
        if not src.is_file():
            QMessageBox.information(self, "No last data", "No saved Stage 5 captures yet.")
            return
        n_disk = len(load_orbbec_handeye_captures(src))
        if not n_disk:
            QMessageBox.information(self, "No last data", f"{src} has no usable views.")
            return
        if self._captures:
            ans = QMessageBox.question(
                self,
                "Load last",
                f"Replace {len(self._captures)} in-memory view(s) with {n_disk} from\n{src} ?",
            )
            if ans != QMessageBox.Yes:
                return
        n = self._load_from_path(src, persist=True)
        kind = "last/" if src == last else "working"
        self._log.append(f"loaded {n} view(s) from {kind}  {src}")

    def _on_run(self) -> None:
        if len(self._captures) < 6:
            QMessageBox.warning(self, "Not enough", "Need at least 6 captures with wrist rotation.")
            return
        sizes = {tuple(c.get("image_size") or ()) for c in self._captures}
        if len(sizes) > 1:
            QMessageBox.warning(self, "Mixed sizes", "Captures mix 640 and 1080. Clear and recapture at one size.")
            return
        try:
            intr = self._intrinsics()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Intrinsics", str(exc))
            return
        caps = []
        for c in self._captures:
            caps.append(
                {
                    **c,
                    "detections": {int(k): np.asarray(v, dtype=np.float64) for k, v in c["detections"].items()},
                }
            )
        self._btn_run.setEnabled(False)
        self._log.append(f"solving from {len(caps)} views …")
        self._persist()
        self._worker = _SolveWorker(caps, self._board_geom, intr, self._robot_cfg)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.finished_err.connect(self._on_err)
        self._worker.start()

    def _on_ok(self, result, path: str) -> None:
        self._btn_run.setEnabled(True)
        t = result.T_link7_cam[:3, 3]
        dq = getattr(result, "joint_zero_offsets_deg", None) or []
        if dq and any(abs(float(v)) > 1e-6 for v in dq[:6]):
            dq_txt = ", ".join(f"j{i + 1}={v:.2f}°" for i, v in enumerate(dq[:6]))
        else:
            dq_txt = "OFF"
        pv = getattr(result, "per_view_ba_rmse_px", None) or []
        pv_txt = ""
        if pv:
            pv_txt = (
                f"per-view RMSE px  min={min(pv):.2f} med={float(np.median(pv)):.2f} "
                f"max={max(pv):.2f}\n"
            )
        shm_mm = getattr(result, "shm_vs_fk_mm", None)
        shm_deg = getattr(result, "shm_vs_fk_deg", None)
        span = getattr(result, "gripper_rot_span_deg", None)
        extra = ""
        if shm_mm is not None:
            extra += f"SHM pose vs q+Δq FK  {shm_mm:.2f} mm / {shm_deg:.2f}°\n"
        if span is not None:
            extra += f"gripper rotation span  {span:.1f}°\n"
        last = self._persist(also_last=True)
        self._log.append(
            f"T_link7_cam t=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m  "
            f"BA RMSE={result.ba_rmse_px:.3f} px  init={result.init_rmse_px:.3f} px\n"
            f"FK offsets {dq_txt}\n"
            f"{pv_txt}{extra}"
            f"saved {path}\n"
            f"captures {last}  last/{orbbec_handeye_captures_last_path().name}"
        )

    def _on_err(self, err: str) -> None:
        self._btn_run.setEnabled(True)
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
