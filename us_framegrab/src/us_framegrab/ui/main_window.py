"""Main window: live crop preview + rqt-equivalent controls."""

from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from us_framegrab.crop import apply_crop
from us_framegrab.presets import list_presets
from us_framegrab.runtime import FrameGrabSession
from us_framegrab.ui.controls import ControlsPanel
from us_framegrab.ui.preview import CropPreview
from us_framegrab.ui.wallpaper import WallpaperPickerDialog
from us_framegrab.wallpaper import WallpaperStore, default_wallpaper_dir


class MainWindow(QMainWindow):
    def __init__(self, session: FrameGrabSession, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("US frame grabber")
        self._session = session
        self._syncing = False
        self._last_size = (0, 0)
        self._stream_view = False
        self._wallpapers = WallpaperStore(default_wallpaper_dir(session.cfg.path))

        header = QLabel(str(session.cfg.pub_bind))
        header.setStyleSheet("padding: 4px; background: #333; color: white;")

        self.preview = CropPreview()
        self.controls = ControlsPanel()

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(6)
        body.addWidget(self.preview, 1)
        body.addWidget(self.controls)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        lay.addWidget(header)
        lay.addLayout(body, 1)
        self.setCentralWidget(central)
        self.resize(1280, 720)

        snap = session.snapshot()
        self.controls.set_machines(
            [(p.id, p.name) for p in list_presets()],
            str(session.cfg.machine),
        )
        self.controls.set_ranges(snap.image_size[0], snap.image_size[1])
        self.controls.set_cbox(snap.cbox)
        self.controls.set_flags(
            hflip=snap.hflip,
            color=snap.color,
            auto_cropping=snap.auto_cropping,
            jpeg_quality=snap.jpeg_quality,
        )
        self.preview.set_cbox(snap.cbox)
        self.controls.set_wallpaper(self._wallpapers, self._wallpapers.active_id)

        self.preview.crop_changed.connect(self._on_preview_crop)
        self.controls.crop_changed.connect(self._on_controls_crop)
        self.controls.hflip_changed.connect(session.set_hflip)
        self.controls.color_changed.connect(session.set_color)
        self.controls.auto_cropping_changed.connect(session.set_auto_cropping)
        self.controls.jpeg_quality_changed.connect(session.set_jpeg_quality)
        self.controls.auto_crop_clicked.connect(self._on_auto_crop)
        self.controls.preview_toggled.connect(self._on_preview_toggled)
        self.controls.save_clicked.connect(self._on_save)
        self.controls.prev_device_clicked.connect(lambda: session.switch_device(-1))
        self.controls.next_device_clicked.connect(lambda: session.switch_device(1))
        self.controls.refresh_devices_clicked.connect(self._on_refresh)
        self.controls.machine_changed.connect(self._on_machine)
        self.controls.wallpaper_clicked.connect(self._on_wallpaper)

        self._timer = QTimer(self)
        self._timer.setInterval(66)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _on_preview_crop(self, x0: int, x1: int, y0: int, y1: int) -> None:
        self._session.set_cbox([x0, x1, y0, y1])
        self._syncing = True
        self.controls.set_cbox([x0, x1, y0, y1])
        self._syncing = False

    def _on_controls_crop(self, x0: int, x1: int, y0: int, y1: int) -> None:
        if self._syncing:
            return
        self._session.set_cbox([x0, x1, y0, y1])
        self.preview.set_cbox([x0, x1, y0, y1])

    def _on_preview_toggled(self, show_stream: bool) -> None:
        self._stream_view = bool(show_stream)
        self.preview.set_overlay_enabled(not self._stream_view)

    def _on_machine(self, preset_id: str) -> None:
        if not self._session.apply_machine_preset(preset_id):
            return
        snap = self._session.snapshot()
        self._syncing = True
        self.controls.set_ranges(snap.image_size[0], snap.image_size[1])
        self.controls.set_cbox(snap.cbox)
        self.controls.set_flags(
            hflip=snap.hflip,
            color=snap.color,
            auto_cropping=snap.auto_cropping,
            jpeg_quality=snap.jpeg_quality,
        )
        self.preview.set_cbox(snap.cbox)
        self._syncing = False

    def _on_wallpaper(self) -> None:
        dlg = WallpaperPickerDialog(self._wallpapers, parent=self)
        dlg.exec_()
        self.controls.set_wallpaper(self._wallpapers, self._wallpapers.active_id)

    def _on_auto_crop(self) -> None:
        if not self._session.auto_crop_once():
            QMessageBox.warning(self, "Auto-crop", "Could not detect an ultrasound region.")

    def _on_save(self) -> None:
        path = self._session.save_settings()
        QMessageBox.information(self, "Saved", f"Wrote {path}")

    def _on_refresh(self) -> None:
        found = self._session.refresh_devices()
        QMessageBox.information(self, "Devices", f"Open indices: {found or '(fixed device)'}")

    def _tick(self) -> None:
        frame = self._session.latest_full()
        snap = self._session.snapshot()
        if self._stream_view and frame is not None:
            self.preview.set_overlay_enabled(False)
            self.preview.set_frame(
                apply_crop(frame, snap.cbox, color=snap.color, hflip=snap.hflip)
            )
        else:
            self.preview.set_overlay_enabled(True)
            self.preview.set_frame(frame)
        if snap.image_size != self._last_size:
            self.controls.set_ranges(snap.image_size[0], snap.image_size[1])
            self._last_size = snap.image_size
        dragging = self.preview.is_dragging or self.controls.sliders_down()
        if not self._syncing and not dragging:
            self._syncing = True
            self.controls.set_cbox(snap.cbox)
            self.preview.set_cbox(snap.cbox)
            self._syncing = False
        err = f"  [{snap.last_error}]" if snap.last_error else ""
        view = "stream" if self._stream_view else "full+box"
        self.controls.status.setText(
            f"view={view}  crop={snap.cbox[0]},{snap.cbox[1]},{snap.cbox[2]},{snap.cbox[3]}\n"
            f"pub={snap.publish_hz:.1f} Hz  frame={snap.frame_index}{err}"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().closeEvent(event)
