"""Sidebar wallpaper pane, theme picker, and locked 9:16 crop editor."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PyQt5.QtCore import QPoint, QRect, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QIcon,
    QImage,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from us_framegrab.wallpaper import (
    DEFAULT_ID,
    IMAGE_SUFFIXES,
    WallpaperStore,
    cover_source_box,
    crop_bgr,
    largest_portrait_box,
    list_image_paths,
    resize_portrait_box,
)

PANEL_GRAY = QColor(208, 208, 208)  # #d0d0d0
Handle = Literal["move", "n", "s", "e", "w", "nw", "ne", "sw", "se", ""]


def bgr_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        rgb = np.ascontiguousarray(np.stack([image, image, image], axis=-1))
    else:
        rgb = np.ascontiguousarray(image[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format_RGB888).copy()


def load_preview_pixmap(path: Path, max_size: QSize) -> QPixmap:
    pix = QPixmap(str(path))
    if pix.isNull():
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return QPixmap()
        pix = QPixmap.fromImage(bgr_to_qimage(bgr))
    return pix.scaled(max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def gray_thumb_pixmap(width: int = 90, height: int = 160) -> QPixmap:
    pix = QPixmap(width, height)
    pix.fill(QColor(176, 176, 176))
    return pix


class WallpaperPane(QWidget):
    """Bottom of the right strip: 9:16 wallpaper with a top fade into panel gray."""

    choose_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self._image: QImage | None = None
        self._theme_id = DEFAULT_ID
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 8)
        lay.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn = QPushButton("Wallpaper")
        self.btn.setFixedWidth(110)
        self.btn.setToolTip("Choose a sidebar wallpaper, or crop a new 9:16 theme")
        self.btn.clicked.connect(self.choose_clicked.emit)
        row.addWidget(self.btn)
        lay.addLayout(row)

    def set_theme(self, store: WallpaperStore, theme_id: str) -> None:
        self._theme_id = theme_id
        image = store.load_bgr(theme_id)
        self._image = None if image is None else bgr_to_qimage(image)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        area = self.rect()
        painter.fillRect(area, PANEL_GRAY)
        if self._image is not None and not self._image.isNull():
            sx, sy, sw, sh = cover_source_box(
                self._image.width(),
                self._image.height(),
                max(area.width(), 1),
                max(area.height(), 1),
                align="bottom",
            )
            dest = QRectF(area)
            dest.adjust(-2, -2, 2, 2)
            painter.drawImage(dest, self._image, QRectF(sx, sy, sw, sh))
        fade_h = max(80, int(self.height() * 0.58))
        grad = QLinearGradient(0, 0, 0, fade_h)
        r, g, b = PANEL_GRAY.red(), PANEL_GRAY.green(), PANEL_GRAY.blue()
        grad.setColorAt(0.00, QColor(r, g, b, 255))
        grad.setColorAt(0.18, QColor(r, g, b, 220))
        grad.setColorAt(0.38, QColor(r, g, b, 150))
        grad.setColorAt(0.58, QColor(r, g, b, 80))
        grad.setColorAt(0.78, QColor(r, g, b, 28))
        grad.setColorAt(1.00, QColor(r, g, b, 0))
        painter.fillRect(QRectF(0, 0, self.width(), fade_h), grad)


class PortraitCropView(QWidget):
    """Still-image view with a movable / resizable locked 9:16 rectangle."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self._bgr: np.ndarray | None = None
        self._qimg: QImage | None = None
        self._box = [0, 18, 0, 32]
        self._dest = QRect()
        self._drag: Handle = ""
        self._drag_origin = QPoint()
        self._box_at_press = [0, 0, 0, 0]

    def set_image(self, image: np.ndarray) -> None:
        self._bgr = np.ascontiguousarray(image)
        self._qimg = bgr_to_qimage(self._bgr)
        h, w = self._bgr.shape[:2]
        self._box = largest_portrait_box(w, h)
        self.update()

    def cropped_bgr(self) -> np.ndarray | None:
        if self._bgr is None:
            return None
        return crop_bgr(self._bgr, self._box)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        if self._qimg is None or self._bgr is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "Load an image")
            return
        h, w = self._bgr.shape[:2]
        self._dest = self._letterbox(w, h)
        painter.drawImage(self._dest, self._qimg)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))
        x0, x1, y0, y1 = self._box
        rect = QRect(self._img_to_widget(x0, y0), self._img_to_widget(x1, y1)).normalized()
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.drawImage(rect, self._qimg, self._img_rect())
        painter.setPen(QPen(QColor(40, 230, 80, 220), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        painter.setBrush(QColor(40, 230, 80, 200))
        for hx, hy in (
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
            (rect.center().x(), rect.top()),
            (rect.center().x(), rect.bottom()),
            (rect.left(), rect.center().y()),
            (rect.right(), rect.center().y()),
        ):
            painter.drawRect(hx - 4, hy - 4, 8, 8)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self._bgr is None:
            return
        handle = self._hit(event.pos())
        if not handle:
            return
        self._drag = handle
        self._drag_origin = event.pos()
        self._box_at_press = list(self._box)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag:
            self._apply_drag(event.pos())
            return
        self._update_cursor(self._hit(event.pos()) if self._bgr is not None else "")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag = ""
            self._update_cursor(self._hit(event.pos()) if self._bgr is not None else "")

    def _apply_drag(self, pos: QPoint) -> None:
        if self._bgr is None or self._dest.width() <= 0:
            return
        h, w = self._bgr.shape[:2]
        ox, oy = self._widget_to_img(self._drag_origin.x(), self._drag_origin.y())
        nx, ny = self._widget_to_img(pos.x(), pos.y())
        box = resize_portrait_box(
            self._box_at_press,
            self._drag,
            int(round(nx - ox)),
            int(round(ny - oy)),
            w,
            h,
        )
        if box != self._box:
            self._box = box
            self.update()

    def _hit(self, pos: QPoint) -> Handle:
        if self._dest.width() <= 0:
            return ""
        x0, x1, y0, y1 = self._box
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
        if self._bgr is None or self._dest.width() <= 0:
            return QPoint(0, 0)
        h, w = self._bgr.shape[:2]
        px = self._dest.x() + int(round(x * self._dest.width() / float(w)))
        py = self._dest.y() + int(round(y * self._dest.height() / float(h)))
        return QPoint(px, py)

    def _widget_to_img(self, x: int, y: int) -> tuple[float, float]:
        if self._bgr is None or self._dest.width() <= 0:
            return 0.0, 0.0
        h, w = self._bgr.shape[:2]
        return (x - self._dest.x()) * w / float(self._dest.width()), (
            y - self._dest.y()
        ) * h / float(self._dest.height())

    def _img_rect(self) -> QRect:
        x0, x1, y0, y1 = self._box
        return QRect(x0, y0, x1 - x0, y1 - y0)


class ImageBrowseDialog(QDialog):
    """Folder browser with image thumbnails and a large preview."""

    def __init__(self, start_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load image")
        self.resize(980, 640)
        self.selected: Path | None = None
        self._dir = Path(start_dir).expanduser()
        if not self._dir.is_dir():
            self._dir = Path.home()
        self._pending: list[tuple[QListWidgetItem, Path]] = []
        self._timer = QTimer(self)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._load_one_thumb)

        self.path_edit = QLineEdit()
        self.btn_up = QPushButton("Up")
        self.btn_folder = QPushButton("Folder…")
        self.btn_up.clicked.connect(self._go_up)
        self.btn_folder.clicked.connect(self._pick_folder)
        self.path_edit.returnPressed.connect(self._go_typed)

        nav = QHBoxLayout()
        nav.addWidget(self.btn_up)
        nav.addWidget(self.path_edit, 1)
        nav.addWidget(self.btn_folder)

        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setIconSize(QSize(120, 120))
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setSpacing(8)
        self.grid.setWordWrap(True)
        self.grid.itemSelectionChanged.connect(self._show_preview)
        self.grid.itemDoubleClicked.connect(self._activate)

        self.preview = QLabel("Select an image")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumWidth(280)
        self.preview.setStyleSheet("background: #2a2a2a; color: #ddd;")
        self.preview.setWordWrap(True)

        split = QHBoxLayout()
        split.addWidget(self.grid, 3)
        split.addWidget(self.preview, 2)

        buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._open_selected)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(nav)
        lay.addLayout(split, 1)
        lay.addWidget(buttons)
        self._scan()

    def _scan(self) -> None:
        self._timer.stop()
        self._pending = []
        self.grid.clear()
        self.path_edit.setText(str(self._dir))
        self.preview.setText("Select an image")
        self.preview.setPixmap(QPixmap())
        if not self._dir.is_dir():
            return
        folders = sorted(
            [path for path in self._dir.iterdir() if path.is_dir() and not path.name.startswith(".")],
            key=lambda path: path.name.lower(),
        )
        folder_icon = QPixmap(120, 120)
        folder_icon.fill(QColor(70, 90, 120))
        for folder in folders:
            item = QListWidgetItem(folder.name)
            item.setIcon(QIcon(folder_icon))
            item.setData(Qt.UserRole, str(folder))
            item.setData(Qt.UserRole + 1, "dir")
            item.setSizeHint(QSize(132, 148))
            self.grid.addItem(item)
        for path in list_image_paths(self._dir):
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path))
            item.setData(Qt.UserRole + 1, "image")
            item.setSizeHint(QSize(132, 148))
            self.grid.addItem(item)
            self._pending.append((item, path))
        if self._pending:
            self._timer.start()

    def _load_one_thumb(self) -> None:
        if not self._pending:
            self._timer.stop()
            return
        item, path = self._pending.pop(0)
        pix = load_preview_pixmap(path, QSize(120, 120))
        if not pix.isNull():
            item.setIcon(QIcon(pix))

    def _current_path(self) -> Path | None:
        item = self.grid.currentItem()
        if item is None:
            return None
        raw = item.data(Qt.UserRole)
        return Path(str(raw)) if raw else None

    def _show_preview(self) -> None:
        path = self._current_path()
        item = self.grid.currentItem()
        if path is None or item is None:
            return
        kind = str(item.data(Qt.UserRole + 1) or "")
        if kind == "dir":
            self.preview.setPixmap(QPixmap())
            self.preview.setText(path.name)
            return
        pix = load_preview_pixmap(path, self.preview.size() if self.preview.width() > 40 else QSize(360, 480))
        if pix.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText(path.name)
            return
        self.preview.setPixmap(pix)

    def _activate(self, _item=None) -> None:
        path = self._current_path()
        item = self.grid.currentItem()
        if path is None or item is None:
            return
        if str(item.data(Qt.UserRole + 1) or "") == "dir":
            self._dir = path
            self._scan()
            return
        self.selected = path
        self.accept()

    def _open_selected(self) -> None:
        path = self._current_path()
        item = self.grid.currentItem()
        if path is None or item is None:
            return
        if str(item.data(Qt.UserRole + 1) or "") == "dir":
            self._dir = path
            self._scan()
            return
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            return
        self.selected = path
        self.accept()

    def _go_up(self) -> None:
        parent = self._dir.parent
        if parent != self._dir:
            self._dir = parent
            self._scan()

    def _go_typed(self) -> None:
        typed = Path(self.path_edit.text().strip()).expanduser()
        if typed.is_dir():
            self._dir = typed
            self._scan()

    def _pick_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Folder", str(self._dir))
        if chosen:
            self._dir = Path(chosen)
            self._scan()


class WallpaperCropDialog(QDialog):
    def __init__(self, image: np.ndarray, default_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crop wallpaper (vertical 16:9)")
        self.resize(900, 680)
        self.view = PortraitCropView()
        self.view.set_image(image)
        self.name = QLineEdit(default_name)
        hint = QLabel("Drag the green box to choose a vertical 16:9 crop. Corners resize.")
        form = QFormLayout()
        form.addRow("Name", self.name)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addWidget(self.view, 1)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def cropped_with_name(self) -> tuple[str, np.ndarray] | None:
        cropped = self.view.cropped_bgr()
        if cropped is None or cropped.size == 0:
            return None
        label = str(self.name.text()).strip() or "wallpaper"
        return label, cropped


class WallpaperPickerDialog(QDialog):
    def __init__(
        self,
        store: WallpaperStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Wallpaper")
        self.resize(620, 460)
        self.store = store
        self.selected_id = store.active_id

        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(90, 160))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setMovement(QListWidget.Static)
        self.list.setSpacing(10)
        self.list.itemSelectionChanged.connect(self._sync_buttons)
        self.list.itemDoubleClicked.connect(self._use_selected)

        self.btn_file = QPushButton("New from file…")
        self.btn_delete = QPushButton("Delete")
        self.btn_use = QPushButton("Use")
        self.btn_file.clicked.connect(self._from_file)
        self.btn_delete.clicked.connect(self._delete)
        self.btn_use.clicked.connect(self._use_selected)

        actions = QHBoxLayout()
        actions.addWidget(self.btn_file)
        actions.addStretch(1)
        actions.addWidget(self.btn_delete)
        actions.addWidget(self.btn_use)

        lay = QVBoxLayout(self)
        lay.addWidget(self.list, 1)
        lay.addLayout(actions)
        self._reload(self.selected_id)

    def _reload(self, select_id: str) -> None:
        self.list.clear()
        current = None
        for theme in self.store.list_all():
            item = QListWidgetItem(theme.name)
            item.setData(Qt.UserRole, theme.id)
            item.setSizeHint(QSize(110, 196))
            if theme.locked:
                item.setIcon(QIcon(gray_thumb_pixmap()))
            else:
                thumb = self.store.load_thumb_bgr(theme.id)
                if thumb is not None:
                    item.setIcon(QIcon(QPixmap.fromImage(bgr_to_qimage(thumb))))
                else:
                    item.setIcon(QIcon(gray_thumb_pixmap()))
            self.list.addItem(item)
            if theme.id == select_id:
                current = item
        if current is not None:
            self.list.setCurrentItem(current)
        self._sync_buttons()

    def _current_id(self) -> str:
        item = self.list.currentItem()
        if item is None:
            return DEFAULT_ID
        return str(item.data(Qt.UserRole) or DEFAULT_ID)

    def _sync_buttons(self) -> None:
        self.btn_delete.setEnabled(self._current_id() != DEFAULT_ID)

    def _use_selected(self, _item=None) -> None:
        self.selected_id = self.store.set_active(self._current_id())
        self.accept()

    def _from_file(self) -> None:
        start = Path(self.store.last_browse).expanduser() if self.store.last_browse else Path.home() / "Pictures"
        dlg = ImageBrowseDialog(start, self)
        if dlg.exec_() != QDialog.Accepted or dlg.selected is None:
            return
        path = dlg.selected
        self.store.set_last_browse(path.parent)
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Wallpaper", "Could not read that image.")
            return
        self._crop_and_add(image, path.stem)

    def _crop_and_add(self, image: np.ndarray, default_name: str) -> None:
        dlg = WallpaperCropDialog(image, default_name, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        named = dlg.cropped_with_name()
        if named is None:
            return
        name, cropped = named
        theme = self.store.add_theme(name, cropped)
        self.selected_id = self.store.set_active(theme.id)
        self.accept()

    def _delete(self) -> None:
        theme_id = self._current_id()
        if theme_id == DEFAULT_ID:
            return
        theme = self.store.get(theme_id)
        label = theme.name if theme is not None else theme_id
        if QMessageBox.question(self, "Delete wallpaper", f"Delete “{label}”?") != QMessageBox.Yes:
            return
        self.store.delete_theme(theme_id)
        self._reload(self.store.active_id)
