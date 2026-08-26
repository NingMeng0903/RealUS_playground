#!/usr/bin/env python3
"""Orbbec SDK v1.1.4 helper (CPython 3.9). Speaks length-prefixed pickle over a Unix socket.

Started by the 3.10 UI. Do not import multicam_calib here.
"""
from __future__ import annotations

import os
import pickle
import socket
import struct
import sys
import threading
import time
from pathlib import Path

_HDR = struct.Struct("!Q")


def send_msg(sock: socket.socket, obj: object) -> None:
    payload = pickle.dumps(obj, protocol=4)
    sock.sendall(_HDR.pack(len(payload)) + payload)


def recv_msg(sock: socket.socket) -> object:
    raw = _readexact(sock, _HDR.size)
    (n,) = _HDR.unpack(raw)
    if n > 64 * 1024 * 1024:
        raise RuntimeError(f"bridge message too large: {n}")
    return pickle.loads(_readexact(sock, n))


def _readexact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("bridge socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _prepare_sdk(sdk_root: Path) -> None:
    py = sdk_root / "python3.9"
    sys.path.insert(0, str(py / "lib" / "python_lib"))
    samples = py / "Samples"
    if samples.is_dir():
        os.chdir(samples)


def _as_str(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace").strip("\x00")
    return str(val or "")


def _pinhole_from_dict(intr: dict, dist: dict | None, image_size: tuple) -> dict:
    fx = float(intr.get("fx", 0.0) or 0.0)
    fy = float(intr.get("fy", fx) or fx)
    cx = float(intr.get("cx", image_size[0] / 2.0) or image_size[0] / 2.0)
    cy = float(intr.get("cy", image_size[1] / 2.0) or image_size[1] / 2.0)
    d = dist or {}
    coeffs = [
        float(d.get("k1", 0.0) or 0.0),
        float(d.get("k2", 0.0) or 0.0),
        float(d.get("p1", 0.0) or 0.0),
        float(d.get("p2", 0.0) or 0.0),
        float(d.get("k3", 0.0) or 0.0),
    ]
    sw, sh = int(image_size[0]), int(image_size[1])
    if sw > 0 and abs(cx - 0.5 * sw) > 0.18 * sw:
        for cand_w, cand_h in ((640, 480), (1280, 720), (1280, 960), (1920, 1080), (2592, 1944)):
            if abs(cx - 0.5 * cand_w) <= 0.12 * cand_w:
                sx, sy = sw / float(cand_w), sh / float(cand_h)
                fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
                break
    return {
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "dist": coeffs,
        "image_size": [sw, sh],
        "source": "factory",
    }


def _T_from_transform(tr: dict | None):
    if not isinstance(tr, dict):
        return None
    rot = tr.get("rot")
    trans = tr.get("trans")
    if rot is None or trans is None:
        return None
    R = [float(x) for x in rot]
    t = [float(x) for x in trans]
    if len(R) != 9 or len(t) != 3:
        return None
    n = (t[0] ** 2 + t[1] ** 2 + t[2] ** 2) ** 0.5
    if n > 0.5:
        t = [x * 1e-3 for x in t]
    return [
        [R[0], R[1], R[2], t[0]],
        [R[3], R[4], R[5], t[1]],
        [R[6], R[7], R[8], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _decode_color(frame, cv2, np):
    w, h = int(frame.width()), int(frame.height())
    data = frame.data()
    fmt = frame.format()
    from ObTypes import (
        OB_PY_FORMAT_I420,
        OB_PY_FORMAT_MJPG,
        OB_PY_FORMAT_RGB888,
        OB_PY_FORMAT_UYVY,
        OB_PY_FORMAT_YUYV,
    )

    if fmt == OB_PY_FORMAT_MJPG:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    if fmt == OB_PY_FORMAT_RGB888:
        return cv2.cvtColor(np.resize(data, (h, w, 3)), cv2.COLOR_RGB2BGR)
    if fmt == OB_PY_FORMAT_YUYV:
        return cv2.cvtColor(np.resize(data, (h, w, 2)), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OB_PY_FORMAT_UYVY:
        return cv2.cvtColor(np.resize(data, (h, w, 2)), cv2.COLOR_YUV2BGR_UYVY)
    if fmt == OB_PY_FORMAT_I420:
        return cv2.cvtColor(data.reshape((h * 3 // 2, w)), cv2.COLOR_YUV2BGR_I420)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    return cv2.cvtColor(np.resize(data, (h, w, 3)), cv2.COLOR_RGB2BGR)


def _decode_depth_m(frame, np):
    w, h = int(frame.width()), int(frame.height())
    data = np.asanyarray(frame.data())
    if data.size == w * h * 2:
        packed = np.resize(data, (h, w, 2)).astype(np.uint16)
        raw = packed[:, :, 0] + packed[:, :, 1] * 256
    else:
        raw = np.asarray(data, dtype=np.uint16).reshape(h, w)
    scale = float(frame.getValueScale())
    return (raw.astype(np.float32) * scale) * 1e-3


class _Session:
    def __init__(self) -> None:
        self.pipe = None
        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        self._thread = None
        self.params = None
        self.align = ""
        self._depth_flip_h = True

    def open(self, req: dict) -> dict:
        import cv2  # noqa: F401
        import numpy as np  # noqa: F401
        from Error import ObException
        from ObTypes import (
            OB_PY_ALIGN_D2C_HW_MODE,
            OB_PY_ALIGN_D2C_SW_MODE,
            OB_PY_ALIGN_DISABLE,
            OB_PY_SENSOR_COLOR,
            OB_PY_SENSOR_DEPTH,
            OB_PY_STREAM_VIDEO,
        )
        import Pipeline

        self.close()
        pipe = Pipeline.Pipeline(None, None)
        config = Pipeline.Config()
        color_wh = (int(req.get("color_width") or 640), int(req.get("color_height") or 480))
        depth_wh = (int(req.get("depth_width") or 640), int(req.get("depth_height") or 400))
        color_fps = int(req.get("fps") or 30)
        depth_fps = int(req.get("depth_fps") or req.get("fps") or 15)

        color_prof = _pick_video(pipe, OB_PY_SENSOR_COLOR, OB_PY_STREAM_VIDEO, color_wh, color_fps, kind="color")
        depth_prof = _pick_video(pipe, OB_PY_SENSOR_DEPTH, OB_PY_STREAM_VIDEO, depth_wh, depth_fps, kind="depth")
        config.enableStream(color_prof)
        config.enableStream(depth_prof)

        align_name = str(req.get("align") or "d2c_sw").strip().lower()
        align_enum = {
            "d2c_hw": OB_PY_ALIGN_D2C_HW_MODE,
            "hw": OB_PY_ALIGN_D2C_HW_MODE,
            "hardware": OB_PY_ALIGN_D2C_HW_MODE,
            "d2c_sw": OB_PY_ALIGN_D2C_SW_MODE,
            "sw": OB_PY_ALIGN_D2C_SW_MODE,
            "software": OB_PY_ALIGN_D2C_SW_MODE,
            "off": OB_PY_ALIGN_DISABLE,
            "disable": OB_PY_ALIGN_DISABLE,
            "none": OB_PY_ALIGN_DISABLE,
        }.get(align_name, OB_PY_ALIGN_D2C_SW_MODE)
        used = align_name
        try:
            config.setAlignMode(align_enum)
        except ObException:
            try:
                config.setAlignMode(OB_PY_ALIGN_DISABLE)
                used = "off"
            except ObException:
                used = "unsupported"

        pipe.start(config, None)
        self.pipe = pipe
        self.align = used
        self._depth_flip_h = bool(req.get("depth_flip_h", True))

        serial, model = "", "Orbbec"
        try:
            info = pipe.getDevice().getDeviceInfo()
            serial = _as_str(info.serialNumber())
            model = _as_str(info.name()) or model
        except Exception:
            pass

        cw = int(color_prof.width())
        ch = int(color_prof.height())
        dw = int(depth_prof.width())
        dh = int(depth_prof.height())
        color_model = {
            "K": [[0.7 * cw, 0.0, cw / 2.0], [0.0, 0.7 * cw, ch / 2.0], [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_size": [cw, ch],
            "source": "factory",
        }
        depth_model = None
        t_cd = None
        try:
            cam = pipe.getCameraParam()
            rgb = cam.get("rgbIntrinsic") if hasattr(cam, "get") else None
            dep = cam.get("depthIntrinsic") if hasattr(cam, "get") else None
            rgb_d = cam.get("rgbDistortion") if hasattr(cam, "get") else None
            dep_d = cam.get("depthDistortion") if hasattr(cam, "get") else None
            if isinstance(rgb, dict):
                color_model = _pinhole_from_dict(rgb, rgb_d if isinstance(rgb_d, dict) else None, (cw, ch))
            if isinstance(dep, dict):
                depth_model = _pinhole_from_dict(dep, dep_d if isinstance(dep_d, dict) else None, (dw, dh))
            t_cd = _T_from_transform(cam.get("transform") if hasattr(cam, "get") else None)
        except Exception:
            pass

        self.params = {
            "serial": serial,
            "model": model,
            "align": used,
            "color": color_model,
            "depth": depth_model,
            "T_color_depth": t_cd,
        }
        self._stop.clear()
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        return {"ok": True, "params": self.params}

    def _grab_loop(self) -> None:
        import cv2
        import numpy as np

        while not self._stop.is_set() and self.pipe is not None:
            try:
                frames = self.pipe.waitForFrames(100)
            except Exception:
                time.sleep(0.02)
                continue
            if frames is None:
                continue
            color_f = frames.colorFrame()
            depth_f = frames.depthFrame()
            if color_f is None:
                continue
            try:
                color = _decode_color(color_f, cv2, np)
            except Exception:
                continue
            if color is None:
                continue
            depth = None
            if depth_f is not None:
                try:
                    depth = _decode_depth_m(depth_f, np)
                    if depth is not None and self._depth_flip_h:
                        depth = np.ascontiguousarray(np.fliplr(depth))
                except Exception:
                    depth = None
            if depth is None:
                with self._lock:
                    prev = self._latest
                depth = (
                    prev["depth"]
                    if prev is not None
                    else np.zeros(color.shape[:2], dtype=np.float32)
                )
            with self._lock:
                self._latest = {
                    "color": color,
                    "depth": depth,
                    "t_ns": time.monotonic_ns(),
                }

    def read(self, timeout_ms: int) -> dict:
        deadline = time.monotonic() + max(0.05, float(timeout_ms) / 1000.0)
        while time.monotonic() < deadline:
            with self._lock:
                latest = self._latest
            if latest is not None:
                return {"ok": True, **latest}
            time.sleep(0.005)
        return {"ok": False, "error": "wait_for_frames timed out"}

    def close(self) -> None:
        self._stop.set()
        th = self._thread
        self._thread = None
        if th is not None:
            th.join(timeout=1.5)
        pipe = self.pipe
        self.pipe = None
        self._latest = None
        if pipe is not None:
            try:
                pipe.stop()
            except Exception:
                pass


def _pick_video(pipe, sensor, stream_enum, wh, fps, kind: str):
    profiles = pipe.getStreamProfileList(sensor)
    count = int(profiles.count()) if hasattr(profiles, "count") else 16
    best = None
    best_score = -1e9
    for i in range(count):
        try:
            video = profiles.getProfile(i).toConcreteStreamProfile(stream_enum)
        except Exception:
            break
        w, h, f = int(video.width()), int(video.height()), int(video.fps())
        score = 0.0
        want_w, want_h = int(wh[0] or 0), int(wh[1] or 0)
        if want_w:
            score += 120.0 if w == want_w else -40.0 * abs(w - want_w) / float(want_w)
        if want_h:
            score += 120.0 if h == want_h else -40.0 * abs(h - want_h) / float(want_h)
        if kind == "depth" and want_w in (0, 640) and (w, h) == (640, 400):
            score += 20
        if fps and f == fps:
            score += 30
        elif kind == "depth" and f == 15:
            score += 18
        elif f == 30:
            score += 8
        elif f == 60:
            score -= 15
        if kind == "color" and int(video.format()) == 5:
            score += 25
        if score > best_score:
            best, best_score = video, score
    if best is None:
        raise RuntimeError(f"no {kind} video profile")
    return best


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: orbbec_v1_bridge.py SOCK_PATH SDK_ROOT", file=sys.stderr)
        return 2
    sock_path = Path(sys.argv[1])
    sdk_root = Path(sys.argv[2])
    _prepare_sdk(sdk_root)
    if sock_path.exists():
        sock_path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(1)
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    conn, _addr = srv.accept()
    session = _Session()
    try:
        while True:
            req = recv_msg(conn)
            if not isinstance(req, dict):
                send_msg(conn, {"ok": False, "error": "bad request"})
                continue
            op = str(req.get("op") or "")
            if op == "open":
                try:
                    send_msg(conn, session.open(req))
                except Exception as exc:
                    session.close()
                    send_msg(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            elif op == "read":
                send_msg(conn, session.read(int(req.get("timeout_ms") or 200)))
            elif op == "close":
                session.close()
                send_msg(conn, {"ok": True})
                break
            else:
                send_msg(conn, {"ok": False, "error": f"unknown op {op!r}"})
    finally:
        session.close()
        try:
            conn.close()
        except Exception:
            pass
        srv.close()
        try:
            sock_path.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
