"""HDMI capture-card I/O: ffmpeg V4L2 pipe or OpenCV V4L2 fallback."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from typing import Any

import numpy as np

from us_framegrab.config import FrameGrabConfig

log = logging.getLogger("us_framegrab.capture")


def list_video_indices(max_num: int = 20) -> list[int]:
    import cv2

    found: list[int] = []
    for index in range(int(max_num)):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            found.append(index)
            cap.release()
    return found


class HdmiCapture:
    """One owner of the HDMI V4L device (ffmpeg process or OpenCV cap)."""

    def __init__(self, cfg: FrameGrabConfig) -> None:
        self._cfg = cfg
        self._backend = str(cfg.capture_backend).strip().lower() or "ffmpeg"
        self._width = int(cfg.frame_width)
        self._height = int(cfg.frame_height)
        self._fps = max(1.0, float(cfg.publish_rate))
        self._fixed_path = str(cfg.video_device_path).strip()
        self._index = int(cfg.video_index)
        self._indices: list[int] | None = None
        self._index_seq = 0
        self._ffmpeg: subprocess.Popen | None = None
        self._cap: Any = None
        self._frame_bytes = self._width * self._height
        self._active_backend = self._backend

    @property
    def use_fixed_device(self) -> bool:
        return bool(self._fixed_path)

    def device_label(self) -> str:
        if self.use_fixed_device:
            return self._fixed_path
        return f"/dev/video{self._index}"

    def open(self) -> None:
        self.close()
        device = self._device_arg()
        if self._backend == "ffmpeg":
            try:
                self._start_ffmpeg(device)
                self._active_backend = "ffmpeg"
                return
            except Exception as exc:
                log.warning("ffmpeg open failed (%s); falling back to OpenCV", exc)
        self._open_opencv(device)
        self._active_backend = "opencv"

    def close(self) -> None:
        self._stop_ffmpeg()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._active_backend == "ffmpeg":
            return self._read_ffmpeg()
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def refresh_indices(self) -> list[int]:
        if self.use_fixed_device:
            return []
        self._indices = list_video_indices()
        self._index_seq = 0
        if self._index in self._indices:
            self._index_seq = self._indices.index(self._index)
        log.info("video indices: %s", self._indices)
        return list(self._indices)

    def switch_index(self, step: int) -> str:
        if self.use_fixed_device:
            return self.device_label()
        if self._indices is None:
            self.refresh_indices()
        if not self._indices:
            return self.device_label()
        self._index_seq = (self._index_seq + int(step)) % len(self._indices)
        self._index = int(self._indices[self._index_seq])
        self._cfg.video_index = self._index
        self.open()
        log.info("switched to video index %s", self._index)
        return self.device_label()

    def _device_arg(self) -> str | int:
        if self.use_fixed_device:
            return self._fixed_path
        if self._backend == "ffmpeg":
            return f"/dev/video{self._index}"
        return int(self._index)

    def _start_ffmpeg(self, device: str | int) -> None:
        self._stop_ffmpeg()
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "v4l2",
            "-input_format",
            "yuyv422",
            "-video_size",
            f"{self._width}x{self._height}",
            "-framerate",
            str(int(self._fps)),
            "-i",
            str(device),
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
        self._ffmpeg = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        log.info(
            "ffmpeg capture %s %sx%s@%.1fHz",
            device,
            self._width,
            self._height,
            self._fps,
        )

    def _stop_ffmpeg(self) -> None:
        proc = self._ffmpeg
        self._ffmpeg = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

    def _read_ffmpeg(self) -> tuple[bool, np.ndarray | None]:
        if self._ffmpeg is None or self._ffmpeg.stdout is None:
            return False, None
        raw = self._ffmpeg.stdout.read(self._frame_bytes)
        if len(raw) != self._frame_bytes:
            log.warning("ffmpeg frame read failed, restarting capture")
            try:
                self._start_ffmpeg(self._device_arg())
            except Exception as exc:
                log.warning("ffmpeg restart failed: %s", exc)
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self._height, self._width))
        return True, frame

    def _open_opencv(self, device: str | int) -> None:
        import cv2

        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(device)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, int(self._fps))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            log.info(
                "OpenCV capture %s reported_fps=%.1f",
                device,
                float(cap.get(cv2.CAP_PROP_FPS)),
            )
        else:
            log.error("failed to open video device %s", device)
        self._cap = cap
