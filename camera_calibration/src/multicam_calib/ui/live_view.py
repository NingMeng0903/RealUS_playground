"""Live per-camera preview grid with AprilTag overlays.

Widget layout: N cameras arranged in a roughly square grid; each cell shows
the latest frame downscaled to a manageable width, with detected tags outlined
and a per-camera tag-count badge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, sqrt
from typing import Callable, Protocol

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from multicam_calib.board.detector import AprilTagDetector, TagDetection, draw_detections
from multicam_calib.io.config import SyncConfig
from multicam_calib.recording.sync import CameraStreamThread, MultiCamSnapshot, snapshot_best_effort


class StreamLike(Protocol):
    def latest(self): ...
    def last_error(self) -> BaseException | None: ...


@dataclass
class ViewCache:
    """Latest detections + image per alias, refreshed at UI cadence."""

    image_bgr: np.ndarray | None = None
    detections: list[TagDetection] = field(default_factory=list)
    frame_index: int = -1
    error: str | None = None


class LiveViewGrid(QWidget):
    """Grid of camera previews with tag overlays and per-cell status."""

    snapshot_ready = pyqtSignal(dict)  # emits {alias: (image_bgr, detections, ts_ns)}

    def __init__(
        self,
        aliases: list[str],
        streams: dict[str, StreamLike],
        detector: AprilTagDetector,
        *,
        capture_detector: AprilTagDetector | None = None,
        preview_detector: AprilTagDetector | None = None,
        sync_cfg: SyncConfig | None = None,
        cell_width: int = 480,
        refresh_hz: int = 15,
        min_tags_for_ba: int | None = None,
        snapshot_fn: Callable[[], MultiCamSnapshot | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._aliases = list(aliases)
        self._streams = streams
        self._detector = detector
        self._capture_detector = capture_detector or detector
        self._preview_detector = preview_detector or detector
        self._sync_cfg = sync_cfg or SyncConfig()
        self._cell_width = int(cell_width)
        self._min_tags_for_ba = min_tags_for_ba
        self._snapshot_fn = snapshot_fn
        self._active = True
        self._labels: dict[str, QLabel] = {}
        self._status_labels: dict[str, QLabel] = {}
        self._caches: dict[str, ViewCache] = {a: ViewCache() for a in self._aliases}
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(max(30, int(1000 / max(1, refresh_hz))))
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def set_active(self, active: bool) -> None:
        """Pause preview timer when tab is hidden (saves CPU, reduces lag)."""
        self._active = bool(active)
        if self._active:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def _build_ui(self) -> None:
        n = len(self._aliases)
        cols = max(1, int(ceil(sqrt(n))))
        grid = QGridLayout(self)
        grid.setSpacing(4)
        for i, alias in enumerate(self._aliases):
            r, c = i // cols, i % cols
            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(2, 2, 2, 2)
            v.setSpacing(2)
            status = QLabel(f"{alias}: --")
            status.setStyleSheet("color: white; background: #222; padding: 2px 6px;")
            pic = QLabel()
            pic.setAlignment(Qt.AlignCenter)
            pic.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            pic.setStyleSheet("background: #000;")
            pic.setMinimumSize(self._cell_width, int(self._cell_width * 9 / 16))
            v.addWidget(status)
            v.addWidget(pic, 1)
            grid.addWidget(container, r, c)
            self._labels[alias] = pic
            self._status_labels[alias] = status

    def _preview_detect_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run AprilTag on a downscaled image for fast live preview."""
        h, w = image_bgr.shape[:2]
        target_w = min(self._cell_width * 2, w)
        if target_w < w:
            target_h = max(1, int(round(h * target_w / w)))
            return cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return image_bgr

    def _refresh(self) -> None:
        if not self._active:
            return
        for alias in self._aliases:
            stream = self._streams.get(alias)
            if stream is None:
                self._status_labels[alias].setText(f"{alias}: (offline)")
                continue
            frame = stream.latest()
            if frame is None:
                err = stream.last_error()
                text = f"{alias}: (no frame)"
                if err is not None:
                    text += f" — last error: {type(err).__name__}"
                self._status_labels[alias].setText(text)
                continue
            cache = self._caches[alias]
            if frame.frame_index == cache.frame_index:
                self._render_cell(alias, cache)
                continue
            cache.frame_index = int(frame.frame_index or 0)
            cache.image_bgr = frame.image
            try:
                detect_img = self._preview_detect_image(frame.image)
                cache.detections = self._preview_detector.detect(detect_img)
                cache.error = None
            except Exception as exc:  # noqa: BLE001
                cache.detections = []
                cache.error = f"{type(exc).__name__}: {exc}"
            self._render_cell(alias, cache)

    def _render_cell(self, alias: str, cache: ViewCache) -> None:
        img = cache.image_bgr
        if img is None:
            return
        overlay = draw_detections(img, cache.detections)
        h, w = overlay.shape[:2]
        target_w = self._cell_width
        target_h = int(round(h * target_w / w))
        thumb = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self._labels[alias].setPixmap(QPixmap.fromImage(qimg.copy()))
        n = len(cache.detections)
        thr = self._min_tags_for_ba
        if thr is not None:
            if n >= thr:
                status = f"{alias}: {n} tags  [in BA]"
                style = "color: #9f9; background: #1a3a1a; padding: 2px 6px;"
            else:
                status = f"{alias}: {n} tags  [skipped]"
                style = "color: #f99; background: #3a1a1a; padding: 2px 6px;"
        else:
            status = f"{alias}: {n} tags"
            style = "color: white; background: #222; padding: 2px 6px;"
        if cache.error:
            status += f"   [!] {cache.error}"
        self._status_labels[alias].setText(status)
        self._status_labels[alias].setStyleSheet(style)

    def snapshot_now(self) -> dict[str, tuple[np.ndarray, list[TagDetection], int]] | None:
        """Return a copy of the best-synced (image, detections, sync_ts_ns) per alias."""
        if self._snapshot_fn is not None:
            snap = self._snapshot_fn()
        else:
            snap = snapshot_best_effort(
                self._streams,  # type: ignore[arg-type]
                attempts=self._sync_cfg.capture_poll_attempts,
                poll_interval_ms=self._sync_cfg.capture_poll_interval_ms,
                use_device_timestamp=self._sync_cfg.use_device_timestamp,
            )
        if snap is None:
            return None
        result: dict[str, tuple[np.ndarray, list[TagDetection], int]] = {}
        for alias in self._aliases:
            frame = snap.frames.get(alias)
            if frame is None:
                return None
            img = frame.image
            dets = self._capture_detector.detect(img)
            if self._sync_cfg.use_device_timestamp and frame.device_timestamp_ns is not None:
                ts_ns = int(frame.device_timestamp_ns)
            else:
                ts_ns = int(frame.timestamp_ns)
            result[alias] = (img.copy(), dets, ts_ns)
        return result

    def stop(self) -> None:
        self._timer.stop()
