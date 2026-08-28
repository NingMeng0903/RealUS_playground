"""Live HDMI preview with a draggable crop rectangle."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from us_framegrab.config import clamp_cbox

Handle = Literal["move", "n", "s", "e", "w", "nw", "ne", "sw", "se", ""]


class CropPreview(QWidget):
    crop_changed = pyqtSignal(int, int, int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)
        self._frame: np.ndarray | None = None
        self._rgb: np.ndarray | None = None
        self._cbox = [0, 100, 0, 100]
        self._dest = QRect()
        self._drag: Handle = ""
        self._drag_origin = QPoint()
        self._box_at_press = [0, 0, 0, 0]
        self._overlay = True

    def set_frame(self, frame: np.ndarray | None) -> None:
        self._frame = None if frame is None else np.ascontiguousarray(frame)
        self._rgb = None
        self.update()

    @property
    def is_dragging(self) -> bool:
        return bool(self._drag)

    def set_overlay_enabled(self, enabled: bool) -> None:
        self._overlay = bool(enabled)
        self.update()

    def set_cbox(self, cbox: list[int]) -> None:
        self._cbox = [int(v) for v in cbox]
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._frame is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for HDMI…")
            return
        rgb = self._ensure_rgb(self._frame)
        h, w = rgb.shape[:2]
        self._dest = self._letterbox(w, h)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        painter.drawImage(self._dest, qimg)
        if not self._overlay:
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        x0, x1, y0, y1 = self._cbox
        rect = QRect(
            self._img_to_widget(x0, y0),
            self._img_to_widget(x1, y1),
        ).normalized()
        painter.setPen(QPen(QColor(40, 230, 80, 150), 2))
        painter.setBrush(QColor(40, 230, 80, 70))
        painter.drawRect(rect)
        painter.setBrush(QColor(40, 230, 80, 180))
        for hx, hy in self._handle_centers(rect):
            painter.drawRect(hx - 4, hy - 4, 8, 8)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self._frame is None or not self._overlay:
            return
        handle = self._hit(event.pos())
        if not handle:
            return
        self._drag = handle
        self._drag_origin = event.pos()
        self._box_at_press = list(self._cbox)
        self._update_cursor(handle)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag:
            self._apply_drag(event.pos())
            return
        if not self._overlay:
            self._update_cursor("")
            return
        self._update_cursor(self._hit(event.pos()) if self._frame is not None else "")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag = ""
            self._update_cursor(self._hit(event.pos()) if self._frame is not None else "")

    def _apply_drag(self, pos: QPoint) -> None:
        if self._frame is None or self._dest.width() <= 0 or self._dest.height() <= 0:
            return
        h, w = self._frame.shape[:2]
        ox, oy = self._widget_to_img(self._drag_origin.x(), self._drag_origin.y())
        nx, ny = self._widget_to_img(pos.x(), pos.y())
        dx, dy = int(round(nx - ox)), int(round(ny - oy))
        x0, x1, y0, y1 = self._box_at_press
        mode = self._drag
        if mode == "move":
            x0, x1 = x0 + dx, x1 + dx
            y0, y1 = y0 + dy, y1 + dy
        if "w" in mode:
            x0 = x0 + dx
        if "e" in mode:
            x1 = x1 + dx
        if "n" in mode:
            y0 = y0 + dy
        if "s" in mode:
            y1 = y1 + dy
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        box = clamp_cbox([x0, x1, y0, y1], w, h)
        if box != self._cbox:
            self._cbox = box
            self.crop_changed.emit(box[0], box[1], box[2], box[3])
            self.update()

    def _hit(self, pos: QPoint) -> Handle:
        if self._dest.width() <= 0:
            return ""
        x0, x1, y0, y1 = self._cbox
        rect = QRect(self._img_to_widget(x0, y0), self._img_to_widget(x1, y1)).normalized()
        handles = {
            "nw": QPoint(rect.left(), rect.top()),
            "ne": QPoint(rect.right(), rect.top()),
            "sw": QPoint(rect.left(), rect.bottom()),
            "se": QPoint(rect.right(), rect.bottom()),
            "n": QPoint(rect.center().x(), rect.top()),
            "s": QPoint(rect.center().x(), rect.bottom()),
            "w": QPoint(rect.left(), rect.center().y()),
            "e": QPoint(rect.right(), rect.center().y()),
        }
        for name, center in handles.items():
            if abs(pos.x() - center.x()) <= 10 and abs(pos.y() - center.y()) <= 10:
                return name  # type: ignore[return-value]
        if rect.contains(pos):
            return "move"
        return ""

    def _update_cursor(self, handle: Handle) -> None:
        cursors = {
            "move": Qt.SizeAllCursor,
            "n": Qt.SizeVerCursor,
            "s": Qt.SizeVerCursor,
            "e": Qt.SizeHorCursor,
            "w": Qt.SizeHorCursor,
            "nw": Qt.SizeFDiagCursor,
            "se": Qt.SizeFDiagCursor,
            "ne": Qt.SizeBDiagCursor,
            "sw": Qt.SizeBDiagCursor,
            "": Qt.ArrowCursor,
        }
        self.setCursor(cursors.get(handle, Qt.ArrowCursor))

    def _letterbox(self, iw: int, ih: int) -> QRect:
        wr, hr = max(self.width(), 1), max(self.height(), 1)
        scale = min(wr / float(iw), hr / float(ih))
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        return QRect((wr - dw) // 2, (hr - dh) // 2, dw, dh)

    def _img_to_widget(self, x: int, y: int) -> QPoint:
        if self._frame is None or self._dest.width() <= 0:
            return QPoint(0, 0)
        h, w = self._frame.shape[:2]
        px = self._dest.x() + int(round(x * self._dest.width() / float(w)))
        py = self._dest.y() + int(round(y * self._dest.height() / float(h)))
        return QPoint(px, py)

    def _widget_to_img(self, x: int, y: int) -> tuple[float, float]:
        if self._frame is None or self._dest.width() <= 0:
            return 0.0, 0.0
        h, w = self._frame.shape[:2]
        ix = (x - self._dest.x()) * w / float(self._dest.width())
        iy = (y - self._dest.y()) * h / float(self._dest.height())
        return ix, iy

    def _handle_centers(self, rect: QRect) -> list[tuple[int, int]]:
        c = rect.center()
        return [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
            (c.x(), rect.top()),
            (c.x(), rect.bottom()),
            (rect.left(), c.y()),
            (rect.right(), c.y()),
        ]

    def _ensure_rgb(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            if self._rgb is None or self._rgb.shape[:2] != frame.shape[:2]:
                self._rgb = np.empty((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            self._rgb[:] = frame[..., None]
            return self._rgb
        self._rgb = np.ascontiguousarray(frame[:, :, ::-1])
        return self._rgb
