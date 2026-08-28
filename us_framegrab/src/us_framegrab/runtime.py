"""Capture thread: read HDMI, crop, publish ZMQ, expose latest frame to the UI."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from us_framegrab.capture import HdmiCapture
from us_framegrab.config import FrameGrabConfig, clamp_cbox
from us_framegrab.crop import apply_crop, get_cropping_param, to_gray
from us_framegrab.presets import get_preset
from us_framegrab.zmq_pub import UsImagePublisher

log = logging.getLogger("us_framegrab.runtime")


@dataclass
class SessionSnapshot:
    cbox: list[int]
    hflip: bool
    color: bool
    auto_cropping: bool
    jpeg_quality: int
    device_label: str
    frame_index: int
    publish_hz: float
    last_error: str
    image_size: tuple[int, int]


@dataclass
class FrameGrabSession:
    cfg: FrameGrabConfig
    auto_crop_on_startup: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _capture: HdmiCapture | None = field(default=None, init=False)
    _publisher: UsImagePublisher | None = field(default=None, init=False)
    _latest_full: np.ndarray | None = field(default=None, init=False)
    _cbox: list[int] = field(default_factory=list, init=False)
    _auto_cropping: bool = field(default=False, init=False)
    _startup_done: bool = field(default=False, init=False)
    _frame_index: int = field(default=0, init=False)
    _pub_count: int = field(default=0, init=False)
    _pub_stat_t: float = field(default=0.0, init=False)
    _publish_hz: float = field(default=0.0, init=False)
    _last_error: str = field(default="", init=False)
    _image_size: tuple[int, int] = field(default=(1920, 1080), init=False)

    def __post_init__(self) -> None:
        self._cbox = clamp_cbox(
            list(self.cfg.final_cbox),
            self.cfg.frame_width,
            self.cfg.frame_height,
        )
        self._image_size = (int(self.cfg.frame_width), int(self.cfg.frame_height))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._capture = HdmiCapture(self.cfg)
        self._publisher = UsImagePublisher(self.cfg)
        try:
            self._publisher.bind()
        except Exception as exc:
            self._last_error = f"ZMQ bind failed: {exc}"
            log.error("%s", self._last_error)
            raise
        try:
            self._capture.open()
        except Exception as exc:
            self._last_error = f"capture open failed: {exc}"
            log.error("%s", self._last_error)
        self._thread = threading.Thread(target=self._loop, name="us-framegrab", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._capture is not None:
            self._capture.close()
            self._capture = None
        if self._publisher is not None:
            self._publisher.close()
            self._publisher = None

    def latest_full(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_full is None:
                return None
            return self._latest_full.copy()

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            device = self._capture.device_label() if self._capture is not None else ""
            return SessionSnapshot(
                cbox=list(self._cbox),
                hflip=bool(self.cfg.hflip),
                color=bool(self.cfg.color),
                auto_cropping=bool(self._auto_cropping),
                jpeg_quality=int(self.cfg.compressed_quality),
                device_label=device,
                frame_index=int(self._frame_index),
                publish_hz=float(self._publish_hz),
                last_error=str(self._last_error),
                image_size=self._image_size,
            )

    def set_cbox(self, cbox: list[int]) -> None:
        with self._lock:
            w, h = self._image_size
            self._cbox = clamp_cbox(list(cbox), w, h)
            self.cfg.final_cbox = list(self._cbox)

    def set_hflip(self, value: bool) -> None:
        self.cfg.hflip = bool(value)

    def set_color(self, value: bool) -> None:
        self.cfg.color = bool(value)

    def set_jpeg_quality(self, value: int) -> None:
        self.cfg.compressed_quality = int(min(100, max(1, value)))

    def set_auto_cropping(self, value: bool) -> None:
        self._auto_cropping = bool(value)

    def apply_machine_preset(self, preset_id: str) -> bool:
        preset = get_preset(preset_id)
        if preset is None:
            return False
        with self._lock:
            self.cfg.machine = preset.id
            self.cfg.init_cbox = list(preset.init_cbox)
            self.cfg.final_cbox = list(preset.final_cbox)
            self.cfg.frame_width = int(preset.frame_width)
            self.cfg.frame_height = int(preset.frame_height)
            self.cfg.compressed_quality = int(preset.jpeg_quality)
            self.cfg.hflip = bool(preset.hflip)
            self.cfg.color = bool(preset.color)
            self._image_size = (int(preset.frame_width), int(preset.frame_height))
            self._cbox = clamp_cbox(
                list(preset.final_cbox),
                int(preset.frame_width),
                int(preset.frame_height),
            )
        log.info(
            "loaded machine preset %s init=%s final=%s %sx%s",
            preset.name,
            preset.init_cbox,
            preset.final_cbox,
            preset.frame_width,
            preset.frame_height,
        )
        return True

    def auto_crop_once(self) -> bool:
        frame = self.latest_full()
        if frame is None:
            return False
        from us_framegrab.crop import detect_sector_extrema

        ext = detect_sector_extrema(to_gray(frame), self.cfg.init_cbox)
        if ext is None:
            log.warning("auto-crop failed")
            return False
        cbox = [int(v) for v in ext["aabb"]]
        self.set_cbox(cbox)
        log.info(
            "auto-crop extrema L=%s R=%s T=%s B=%s bandL=%s bandR=%s aabb=%s",
            ext["left"],
            ext["right"],
            ext["top"],
            ext["bottom"],
            ext.get("band_left"),
            ext.get("band_right"),
            cbox,
        )
        return True

    def save_settings(self) -> str:
        with self._lock:
            self.cfg.final_cbox = list(self._cbox)
            dest = self.cfg.save()
            frame = self._latest_full
            overlay_src = None if frame is None else frame.copy()
            cbox = list(self._cbox)
        if overlay_src is not None:
            import cv2

            vis = overlay_src if overlay_src.ndim == 3 else cv2.cvtColor(overlay_src, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (cbox[0], cbox[2]), (cbox[1], cbox[3]), (90, 220, 50), 2)
            preview_path = dest.with_suffix(".png")
            cv2.imwrite(str(preview_path), vis)
        log.info("saved settings to %s", dest)
        return str(dest)

    def refresh_devices(self) -> list[int]:
        if self._capture is None:
            return []
        return self._capture.refresh_indices()

    def switch_device(self, step: int) -> str:
        if self._capture is None:
            return ""
        return self._capture.switch_index(step)

    def _loop(self) -> None:
        period = 1.0 / max(1.0, float(self.cfg.publish_rate))
        self._pub_stat_t = time.monotonic()
        while not self._stop.is_set():
            t0 = time.monotonic()
            cap = self._capture
            if cap is None:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(min(0.05, period))
                continue
            height, width = frame.shape[:2]
            with self._lock:
                self._latest_full = frame
                self._image_size = (int(width), int(height))
                self._cbox = clamp_cbox(self._cbox, width, height)
                do_startup = (not self._startup_done) and bool(self.auto_crop_on_startup)
                continuous = bool(self._auto_cropping)
                cbox = list(self._cbox)
                color = bool(self.cfg.color)
                hflip = bool(self.cfg.hflip)
            if do_startup:
                ok_c, found = get_cropping_param(to_gray(frame), self.cfg.init_cbox)
                if ok_c and found is not None:
                    self.set_cbox(found)
                    cbox = list(found)
                    log.info("startup auto-crop: x0=%s x1=%s y0=%s y1=%s", *found)
                else:
                    log.warning("startup auto-crop failed, using final_cbox")
                self._startup_done = True
            if continuous:
                ok_c, found = get_cropping_param(to_gray(frame), self.cfg.init_cbox)
                if ok_c and found is not None:
                    self.set_cbox(found)
                else:
                    log.warning("continuous auto-crop failed")
                # Match ROS node: preview-only while auto_cropping is on.
                self._sleep_remainder(t0, period)
                continue
            cropped = apply_crop(frame, cbox, color=color, hflip=hflip)
            pub = self._publisher
            if pub is not None:
                pub.send(cropped, self._frame_index)
            self._frame_index += 1
            self._note_publish()
            self._sleep_remainder(t0, period)

    def _note_publish(self) -> None:
        self._pub_count += 1
        now = time.monotonic()
        elapsed = now - self._pub_stat_t
        if elapsed >= 2.0:
            hz = self._pub_count / elapsed
            self._publish_hz = hz
            log.info("publish rate: %.1f Hz", hz)
            self._pub_count = 0
            self._pub_stat_t = now

    @staticmethod
    def _sleep_remainder(t0: float, period: float) -> None:
        remain = period - (time.monotonic() - t0)
        if remain > 0:
            time.sleep(remain)


def run_headless(cfg: FrameGrabConfig, *, auto_crop_on_startup: bool | None = None) -> int:
    flag = cfg.auto_crop_on_startup if auto_crop_on_startup is None else bool(auto_crop_on_startup)
    session = FrameGrabSession(cfg, auto_crop_on_startup=flag)
    session.start()
    log.info("headless publisher on %s  ctrl-c to stop", cfg.pub_bind)
    try:
        while True:
            time.sleep(0.5)
            snap = session.snapshot()
            if snap.last_error:
                log.warning("%s", snap.last_error)
    except KeyboardInterrupt:
        log.info("stopped")
        return 0
    finally:
        session.stop()
    return 0
