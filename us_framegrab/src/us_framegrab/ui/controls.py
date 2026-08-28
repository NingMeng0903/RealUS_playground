"""rqt-equivalent crop / device / stream controls."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from us_framegrab.ui.wallpaper import PANEL_GRAY, WallpaperPane
from us_framegrab.wallpaper import WallpaperStore


class _SliderSpin(QWidget):
    valueChanged = pyqtSignal(int)

    def __init__(self, lo: int, hi: int, value: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(value)
        self.spin = QSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setValue(value)
        self.spin.setFixedWidth(72)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.spin.valueChanged.connect(self.valueChanged.emit)

    def set_range(self, lo: int, hi: int) -> None:
        self.slider.setRange(lo, hi)
        self.spin.setRange(lo, hi)

    def set_value(self, value: int) -> None:
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.slider.setValue(int(value))
        self.spin.setValue(int(value))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)

    def value(self) -> int:
        return int(self.spin.value())


class ControlsPanel(QWidget):
    crop_changed = pyqtSignal(int, int, int, int)
    hflip_changed = pyqtSignal(bool)
    color_changed = pyqtSignal(bool)
    auto_cropping_changed = pyqtSignal(bool)
    jpeg_quality_changed = pyqtSignal(int)
    auto_crop_clicked = pyqtSignal()
    preview_toggled = pyqtSignal(bool)
    save_clicked = pyqtSignal()
    prev_device_clicked = pyqtSignal()
    next_device_clicked = pyqtSignal()
    refresh_devices_clicked = pyqtSignal()
    machine_changed = pyqtSignal(str)
    wallpaper_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(340)
        self.setObjectName("controlsPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"#controlsPanel {{ background: {PANEL_GRAY.name()}; }}")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        top = QWidget()
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(8, 8, 8, 6)

        machine_box = QGroupBox("Machine")
        machine_l = QVBoxLayout(machine_box)
        self.machine = QComboBox()
        machine_l.addWidget(self.machine)
        self.machine.activated[int].connect(self._emit_machine)
        top_l.addWidget(machine_box)

        crop_box = QGroupBox("Crop [x0, x1, y0, y1]")
        form = QFormLayout(crop_box)
        self.x0 = _SliderSpin(0, 1920, 0)
        self.x1 = _SliderSpin(0, 1920, 100)
        self.y0 = _SliderSpin(0, 1200, 0)
        self.y1 = _SliderSpin(0, 1200, 100)
        form.addRow("x0", self.x0)
        form.addRow("x1", self.x1)
        form.addRow("y0", self.y0)
        form.addRow("y1", self.y1)
        for w in (self.x0, self.x1, self.y0, self.y1):
            w.valueChanged.connect(self._emit_crop)
        top_l.addWidget(crop_box)

        flags = QGroupBox("Frame")
        flags_l = QVBoxLayout(flags)
        self.hflip = QCheckBox("Horizontal flip")
        self.color = QCheckBox("Color (BGR)")
        self.auto_cropping = QCheckBox("Auto-crop continuously")
        flags_l.addWidget(self.hflip)
        flags_l.addWidget(self.color)
        flags_l.addWidget(self.auto_cropping)
        self.hflip.toggled.connect(self.hflip_changed.emit)
        self.color.toggled.connect(self.color_changed.emit)
        self.auto_cropping.toggled.connect(self.auto_cropping_changed.emit)
        top_l.addWidget(flags)

        stream = QGroupBox("Stream")
        stream_l = QFormLayout(stream)
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(80)
        self.quality.valueChanged.connect(self.jpeg_quality_changed.emit)
        stream_l.addRow("JPEG quality", self.quality)
        top_l.addWidget(stream)

        btns = QVBoxLayout()
        self.btn_auto = QPushButton("Auto-crop now")
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.setCheckable(True)
        self.btn_preview.setToolTip("Toggle streamed crop vs full HDMI with crop box")
        self.btn_save = QPushButton("Save settings")
        row = QHBoxLayout()
        self.btn_prev = QPushButton("Prev video")
        self.btn_next = QPushButton("Next video")
        self.btn_refresh = QPushButton("Refresh devices")
        row.addWidget(self.btn_prev)
        row.addWidget(self.btn_next)
        btns.addWidget(self.btn_auto)
        btns.addWidget(self.btn_preview)
        btns.addLayout(row)
        btns.addWidget(self.btn_refresh)
        btns.addWidget(self.btn_save)
        top_l.addLayout(btns)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #222; font-size: 12px;")
        top_l.addWidget(self.status)
        root.addWidget(top)

        self.wallpaper = WallpaperPane()
        self.wallpaper.choose_clicked.connect(self.wallpaper_clicked.emit)
        root.addWidget(self.wallpaper, 1)

        self.btn_auto.clicked.connect(self.auto_crop_clicked.emit)
        self.btn_preview.toggled.connect(self._on_preview_toggled)
        self.btn_save.clicked.connect(self.save_clicked.emit)
        self.btn_prev.clicked.connect(self.prev_device_clicked.emit)
        self.btn_next.clicked.connect(self.next_device_clicked.emit)
        self.btn_refresh.clicked.connect(self.refresh_devices_clicked.emit)

    def set_ranges(self, width: int, height: int) -> None:
        self.x0.set_range(0, int(width))
        self.x1.set_range(0, int(width))
        self.y0.set_range(0, int(height))
        self.y1.set_range(0, int(height))

    def set_cbox(self, cbox: list[int]) -> None:
        self.x0.set_value(cbox[0])
        self.x1.set_value(cbox[1])
        self.y0.set_value(cbox[2])
        self.y1.set_value(cbox[3])

    def set_flags(self, *, hflip: bool, color: bool, auto_cropping: bool, jpeg_quality: int) -> None:
        self.hflip.blockSignals(True)
        self.color.blockSignals(True)
        self.auto_cropping.blockSignals(True)
        self.quality.blockSignals(True)
        self.hflip.setChecked(hflip)
        self.color.setChecked(color)
        self.auto_cropping.setChecked(auto_cropping)
        self.quality.setValue(int(jpeg_quality))
        self.hflip.blockSignals(False)
        self.color.blockSignals(False)
        self.auto_cropping.blockSignals(False)
        self.quality.blockSignals(False)

    def _on_preview_toggled(self, checked: bool) -> None:
        self.btn_preview.setText("Preview (stream)" if checked else "Preview")
        self.preview_toggled.emit(bool(checked))

    def set_machines(self, names: list[tuple[str, str]], current_id: str) -> None:
        self.machine.blockSignals(True)
        self.machine.clear()
        current = 0
        for index, (preset_id, name) in enumerate(names):
            self.machine.addItem(name, preset_id)
            if preset_id == current_id:
                current = index
        self.machine.setCurrentIndex(current)
        self.machine.blockSignals(False)

    def set_wallpaper(self, store: WallpaperStore, theme_id: str) -> None:
        self.wallpaper.set_theme(store, theme_id)

    def sliders_down(self) -> bool:
        return any(w.slider.isSliderDown() for w in (self.x0, self.x1, self.y0, self.y1))

    def _emit_machine(self, _index: int = 0) -> None:
        preset_id = self.machine.currentData()
        if preset_id:
            self.machine_changed.emit(str(preset_id))

    def _emit_crop(self, _value: int = 0) -> None:
        self.crop_changed.emit(self.x0.value(), self.x1.value(), self.y0.value(), self.y1.value())
