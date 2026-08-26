"""Orbbec RGB-D session for Stage 3–5 (pyorbbecsdk2, with V4L2 RGB fallback).

This is **not** part of the 4× RealSense roster. Stage 0–2 keep opening only
those cameras.

``pyorbbecsdk2`` matches Gemini 2 / 330-class modules. First-gen Gemini
(Astra 3D Camera F, USB PID ``0614`` + ``0511``) is opened through a
CPython 3.9 helper that loads bundled Orbbec SDK v1.1.4 (depth + D2C).
If that helper is missing, RGB falls back to V4L2.

The import name stays ``pyorbbecsdk``.
"""
from __future__ import annotations

import os
import pickle
import select
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from multicam_calib.calib.orbbec_rgbd import (
    PinholeModel,
    align_factory_pinhole_to_stream,
    pinhole_from_orbbec_intrinsic,
    point_cloud_stats,
    se3_from_orbbec_extrinsic,
    unproject_aligned_depth,
    warp_depth_to_color,
)
from multicam_calib.devices.base import DiscoveredCamera
from multicam_calib.io.config import OrbbecConfig

# First-gen Gemini / Astra 3D Camera(F): depth 0614 + RGB UVC 0511.
_FIRST_GEN_PIDS = frozenset({"0511", "0614"})
ob = None  # type: ignore[assignment]
_IMPORT_ERROR: Exception | None = None
_OB_TRIED = False


def _orbbec_usb_pids() -> set[str]:
    out: set[str] = set()
    sys_usb = Path("/sys/bus/usb/devices")
    if not sys_usb.is_dir():
        return out
    for dev in sys_usb.iterdir():
        vid_p = dev / "idVendor"
        pid_p = dev / "idProduct"
        try:
            if not vid_p.is_file() or vid_p.read_text().strip().lower() != "2bc5":
                continue
            if pid_p.is_file():
                out.add(pid_p.read_text().strip().lower())
        except OSError:
            continue
    return out


def _ensure_ob() -> Any:
    """Load pyorbbecsdk2 only for modules it actually matches.

    First-gen Gemini is visible on USB but v2 cannot read its serial; importing
    the package still enumerates and prints ``Failed to query USB device serial
    number``. Skip that import when PID 0614/0511 is on the bus.
    """
    global ob, _IMPORT_ERROR, _OB_TRIED
    if _OB_TRIED:
        return ob
    _OB_TRIED = True
    if _orbbec_usb_pids() & _FIRST_GEN_PIDS:
        return None
    try:
        import pyorbbecsdk as _ob  # noqa: WPS433

        ob = _ob
    except Exception as exc:  # noqa: BLE001
        _IMPORT_ERROR = exc
        ob = None
    return ob


def sdk_available() -> bool:
    return _ensure_ob() is not None


def sdk_import_error() -> str:
    _ensure_ob()
    if _orbbec_usb_pids() & _FIRST_GEN_PIDS:
        return ""
    if _IMPORT_ERROR is None:
        return ""
    return (
        "pyorbbecsdk is not importable in this environment: "
        f"{_IMPORT_ERROR!r}. In the camera_calib env run:\n"
        "  pip install --upgrade pyorbbecsdk2\n"
        "The tmp/OrbbecSDK_Python_v1.1.4 tree is Python 3.7–3.9 only."
    )


def _require_ob() -> None:
    if _ensure_ob() is None:
        raise RuntimeError(sdk_import_error() or "pyorbbecsdk2 is not used for this first-gen Gemini")


def _prefer_v4l2() -> bool:
    return bool(_orbbec_usb_pids() & _FIRST_GEN_PIDS) and bool(list_orbbec_v4l_nodes())


def _sdk_no_match(exc: BaseException) -> bool:
    msg = str(exc)
    needles = (
        "No device found",
        "CAMERA_DISCONNECTED",
        "No matched usb device",
        "not found",
        "deviceCount",
    )
    return any(n in msg for n in needles)


def _sdk_mismatch_hint(pids: Iterable[str]) -> str:
    norm = {str(p).strip().lower() for p in pids}
    if norm & _FIRST_GEN_PIDS:
        return (
            "udev 已通过（USB write=ok）。这是第一代 Gemini / Astra 3D Camera(F) "
            "(PID 0614 depth + 0511 RGB)。pyorbbecsdk2 不匹配这台机，"
            "所以 Pipeline 报 No device found——不是权限。"
            "RGB 走 V4L2；深度/D2C 需要 OrbbecSDK Python v1.1.4 (CPython 3.7–3.9)。"
        )
    return (
        "USB 权限正常，但 pyorbbecsdk2 枚举到 0 台设备。"
        "确认没有别的进程占用，或换与模块匹配的 SDK 版本。"
    )


def _usb_ancestor(path: Path, vid: str = "2bc5") -> Path | None:
    cur = path.resolve()
    for _ in range(16):
        vid_p = cur / "idVendor"
        if vid_p.is_file():
            try:
                if vid_p.read_text().strip().lower() == vid.lower():
                    return cur
            except OSError:
                return None
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return None


def _wait_orbbec_v4l_nodes(wait_s: float) -> list[dict[str, str]]:
    deadline = time.monotonic() + max(0.0, float(wait_s))
    nodes = list_orbbec_v4l_nodes()
    while not nodes and time.monotonic() < deadline:
        time.sleep(0.1)
        nodes = list_orbbec_v4l_nodes()
    return nodes


def list_orbbec_v4l_nodes() -> list[dict[str, str]]:
    """V4L nodes whose USB parent is VID 2bc5 (RGB UVC on first-gen Gemini)."""
    root = Path("/sys/class/video4linux")
    if not root.is_dir():
        return []
    out: list[dict[str, str]] = []
    for vdev in sorted(root.iterdir()):
        usb = _usb_ancestor(vdev)
        if usb is None:
            continue
        node = Path("/dev") / vdev.name
        if not node.exists():
            continue
        serial = ""
        sp = usb / "serial"
        if sp.is_file():
            try:
                serial = sp.read_text().strip()
            except OSError:
                serial = ""
        pid = ""
        pp = usb / "idProduct"
        if pp.is_file():
            try:
                pid = pp.read_text().strip()
            except OSError:
                pid = ""
        name = ""
        npth = vdev / "name"
        if npth.is_file():
            try:
                name = npth.read_text().strip()
            except OSError:
                name = ""
        index = ""
        ip = vdev / "index"
        if ip.is_file():
            try:
                index = ip.read_text().strip()
            except OSError:
                index = ""
        out.append(
            {
                "node": str(node),
                "serial": serial,
                "pid": pid,
                "name": name,
                "index": index,
            }
        )
    return out


def pinhole_guess_v4l(width: int, height: int) -> PinholeModel:
    """Placeholder K until Stage 4 chessboard. ~70° HFOV, zero distortion."""
    w, h = int(width), int(height)
    fx = 0.70 * float(max(w, 1))
    k = np.array(
        [[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros(5, dtype=np.float64)
    return PinholeModel(K=k, dist=dist, image_size=(w, h), source="v4l2_guess")


def diagnose_orbbec_usb() -> str:
    """Human-readable USB / udev / SDK-mismatch / V4L status (no SDK required)."""
    nodes: list[tuple[str, str, bool]] = []
    sys_usb = Path("/sys/bus/usb/devices")
    if sys_usb.is_dir():
        for dev in sys_usb.iterdir():
            vid_p = dev / "idVendor"
            pid_p = dev / "idProduct"
            if not vid_p.is_file() or vid_p.read_text().strip().lower() != "2bc5":
                continue
            pid = pid_p.read_text().strip() if pid_p.is_file() else "?"
            busdev = dev / "dev"
            node = ""
            writable = False
            if busdev.is_file():
                major_minor = busdev.read_text().strip().replace(":", " ")
                try:
                    maj, mino = (int(x) for x in major_minor.split())
                    for cand in Path("/dev/bus/usb").glob("*/*"):
                        st = cand.stat()
                        if os.major(st.st_rdev) == maj and os.minor(st.st_rdev) == mino:
                            node = str(cand)
                            writable = os.access(cand, os.W_OK)
                            break
                except (ValueError, OSError):
                    pass
            nodes.append((pid, node or str(dev), writable))
    if not nodes:
        return "lsusb/sysfs 里没有 VID 2bc5 的奥比中光。检查线材和 USB3 口。"
    blocked = [n for n in nodes if n[1].startswith("/dev/") and not n[2]]
    lines = [f"PID {pid}  {node}  write={'ok' if wr else 'DENIED'}" for pid, node, wr in nodes]
    if blocked:
        lines.append(
            "SDK 打不开是 USB 权限，不是没插上。装 udev 后拔插相机：\n"
            "  cd camera_calibration && bash scripts/install_orbbec_udev.sh"
        )
        return "\n".join(lines)
    lines.append(_sdk_mismatch_hint(pid for pid, _node, _wr in nodes))
    v4l = list_orbbec_v4l_nodes()
    if v4l:
        desc = ", ".join(
            f"{n['node']} idx={n['index'] or '?'} {n['serial'] or '?'}" for n in v4l
        )
        lines.append(f"V4L RGB: {desc}")
        lines.append("点 Open：RGB 走 V4L2（Stage 4/5 可用）。深度面板会是空的。")
    return "\n".join(lines)


def _raise_if_no_device(exc: Exception) -> None:
    if not _sdk_no_match(exc):
        raise
    hint = diagnose_orbbec_usb()
    raise RuntimeError(f"{exc}\n{hint}") from exc


def _enum(mod: Any, *names: str) -> Any:
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AttributeError(f"none of {names} on pyorbbecsdk")


def _call(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if not hasattr(obj, n):
            continue
        val = getattr(obj, n)
        return val() if callable(val) else val
    return default


@dataclass
class OrbbecRGBDFrame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    timestamp_ns: int
    color_size: tuple[int, int]
    depth_size: tuple[int, int]


@dataclass
class OrbbecFactoryParams:
    serial: str
    model: str
    color: PinholeModel
    depth: PinholeModel | None = None
    T_color_depth: np.ndarray | None = None


@dataclass
class OrbbecRGBDSession:
    """Open one Orbbec, stream color+depth with D2C, build a colored cloud.

    ``backend`` is ``sdk`` (pyorbbecsdk2 RGB-D) or ``v4l2`` (RGB only).
    """

    cfg: OrbbecConfig
    _pipeline: Any = field(default=None, init=False, repr=False)
    _align_filter: Any = field(default=None, init=False, repr=False)
    _pc_filter: Any = field(default=None, init=False, repr=False)
    _cap: Any = field(default=None, init=False, repr=False)
    _v1: Any = field(default=None, init=False, repr=False)
    _backend: str = field(default="", init=False)
    _params: OrbbecFactoryParams | None = field(default=None, init=False)
    _opened: bool = field(default=False, init=False)
    _emitter: dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def params(self) -> OrbbecFactoryParams | None:
        return self._params

    @property
    def is_open(self) -> bool:
        return bool(self._opened)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def has_depth(self) -> bool:
        return self._backend in ("sdk", "v1")

    @property
    def emitter(self) -> dict[str, Any]:
        return dict(self._emitter)

    def open_rgb_only(self, width: int = 1920, height: int = 1080) -> OrbbecFactoryParams:
        """Close depth/v1 and open UVC MJPEG RGB (Stage 5 does not need depth)."""
        self.close()
        time.sleep(0.8)
        return self._open_v4l2(width=int(width), height=int(height), wait_s=8.0)

    def open(self) -> OrbbecFactoryParams:
        self.close()
        errors: list[str] = []
        try:
            return self._open_v1_bridge()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"v1: {exc}")
        if _prefer_v4l2():
            try:
                return self._open_v4l2()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"v4l2: {exc}")
        if _ensure_ob() is not None:
            try:
                return self._open_sdk()
            except Exception as exc:  # noqa: BLE001
                if not _sdk_no_match(exc):
                    raise
                errors.append(f"sdk: {exc}")
        try:
            return self._open_v4l2()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"v4l2: {exc}")
        raise RuntimeError("Orbbec open failed.\n" + "\n".join(errors) + "\n" + diagnose_orbbec_usb())

    def _open_sdk(self) -> OrbbecFactoryParams:
        _require_ob()
        device = _select_device(self.cfg.serial)
        info = _device_info(device)
        try:
            pipeline = ob.Pipeline(device) if device is not None else ob.Pipeline()
        except Exception as exc:  # noqa: BLE001
            _raise_if_no_device(exc)
        config = ob.Config()
        color_profile = _pick_video_profile(
            pipeline,
            _enum(ob, "OBSensorType").COLOR_SENSOR,
            self.cfg.color_width,
            self.cfg.color_height,
            self.cfg.fps,
        )
        depth_profile = _pick_video_profile(
            pipeline,
            _enum(ob, "OBSensorType").DEPTH_SENSOR,
            self.cfg.depth_width,
            self.cfg.depth_height,
            self.cfg.fps,
        )
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        align_name = str(self.cfg.align).strip().lower()
        align_enum = _enum(ob, "OBAlignMode")
        use_align_filter = False
        if align_name in ("d2c_hw", "hw", "hardware"):
            config.set_align_mode(align_enum.HW_MODE)
        else:
            # Software D2C: AlignFilter warps depth onto the color grid.
            # SW_MODE on Config is optional and not supported on every module.
            if hasattr(align_enum, "DISABLE"):
                try:
                    config.set_align_mode(align_enum.DISABLE)
                except Exception:  # noqa: BLE001
                    pass
            use_align_filter = True
        agg = getattr(ob, "OBFrameAggregateOutputMode", None)
        if agg is not None and hasattr(config, "set_frame_aggregate_output_mode"):
            config.set_frame_aggregate_output_mode(agg.FULL_FRAME_REQUIRE)
        if hasattr(pipeline, "enable_frame_sync"):
            try:
                pipeline.enable_frame_sync()
            except Exception:  # noqa: BLE001
                pass
        pipeline.start(config)
        self._pipeline = pipeline
        self._align_filter = None
        if use_align_filter:
            try:
                self._align_filter = ob.AlignFilter(
                    align_to_stream=_enum(ob, "OBStreamType").COLOR_STREAM
                )
            except Exception:  # noqa: BLE001
                self._align_filter = None
        try:
            self._pc_filter = ob.PointCloudFilter()
            fmt = _enum(ob, "OBFormat")
            self._pc_filter.set_create_point_format(fmt.RGB_POINT)
            if hasattr(self._pc_filter, "set_frame_align_state"):
                self._pc_filter.set_frame_align_state(True)
        except Exception:  # noqa: BLE001
            self._pc_filter = None
        color_size = (
            int(_call(color_profile, "get_width", default=self.cfg.color_width)),
            int(_call(color_profile, "get_height", default=self.cfg.color_height)),
        )
        depth_size = (
            int(_call(depth_profile, "get_width", default=self.cfg.depth_width or color_size[0])),
            int(_call(depth_profile, "get_height", default=self.cfg.depth_height or color_size[1])),
        )
        color_model = pinhole_from_orbbec_intrinsic(
            color_profile.get_intrinsic(),
            color_profile.get_distortion(),
            color_size,
        )
        depth_model = None
        try:
            depth_model = pinhole_from_orbbec_intrinsic(
                depth_profile.get_intrinsic(),
                depth_profile.get_distortion(),
                depth_size,
            )
        except Exception:  # noqa: BLE001
            depth_model = None
        t_cd = None
        try:
            t_cd = se3_from_orbbec_extrinsic(depth_profile.get_extrinsic_to(color_profile))
        except Exception:  # noqa: BLE001
            t_cd = None
        self._params = OrbbecFactoryParams(
            serial=info["serial"],
            model=info["model"],
            color=color_model,
            depth=depth_model,
            T_color_depth=t_cd,
        )
        self._backend = "sdk"
        self._opened = True
        return self._params

    def _open_v1_bridge(self) -> OrbbecFactoryParams:
        self._v1 = _V1Bridge.start(self.cfg)
        try:
            reply = self._v1.request(
                {
                    "op": "open",
                    "align": self.cfg.align,
                    "color_width": self.cfg.color_width,
                    "color_height": self.cfg.color_height,
                    "depth_width": self.cfg.depth_width,
                    "depth_height": self.cfg.depth_height,
                    "fps": self.cfg.fps,
                    "depth_fps": self.cfg.depth_fps,
                    "depth_flip_h": self.cfg.depth_flip_h,
                },
                timeout=20.0,
            )
        except Exception:
            self._v1.close()
            self._v1 = None
            raise
        if not reply.get("ok"):
            self._v1.close()
            self._v1 = None
            raise RuntimeError(str(reply.get("error") or "v1 open failed"))
        raw = reply["params"]
        params = _params_from_v1(raw)
        self._emitter = {"laser": raw.get("laser"), "ldp": raw.get("ldp")}
        want = (int(self.cfg.color_width), int(self.cfg.color_height))
        got = (int(params.color.image_size[0]), int(params.color.image_size[1]))
        if want[0] >= 1280 and (abs(got[0] - want[0]) > 16 or abs(got[1] - want[1]) > 16):
            self.close()
            raise RuntimeError(f"v1 color {got[0]}x{got[1]} != requested {want[0]}x{want[1]}")
        self._params = params
        self._backend = "v1"
        self._opened = True
        return params

    def _open_v4l2(
        self,
        width: int | None = None,
        height: int | None = None,
        wait_s: float = 0.0,
    ) -> OrbbecFactoryParams:
        req_w = int(width if width is not None else self.cfg.color_width)
        req_h = int(height if height is not None else self.cfg.color_height)
        wanted = str(self.cfg.serial or "").strip()
        nodes = _wait_orbbec_v4l_nodes(wait_s)
        if not nodes:
            raise RuntimeError("no Orbbec V4L node (VID 2bc5) under /dev/video*")
        if wanted:
            matched = [n for n in nodes if n["serial"] == wanted]
            if matched:
                nodes = matched
        nodes = sorted(nodes, key=lambda n: (int(n["index"] or 99), n["node"]))
        last_err = "no capture node accepted a frame"
        cap = None
        chosen: dict[str, str] | None = None
        for item in nodes:
            probe = cv2.VideoCapture(item["node"], cv2.CAP_V4L2)
            if not probe.isOpened():
                last_err = f"{item['node']} did not open"
                continue
            probe.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if req_w >= 1280:
                probe.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            if req_w > 0:
                probe.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
            if req_h > 0:
                probe.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
            if int(self.cfg.fps) > 0:
                probe.set(cv2.CAP_PROP_FPS, int(self.cfg.fps))
            ok = False
            frame = None
            for _ in range(8):
                ok, frame = probe.read()
                if ok and frame is not None and getattr(frame, "size", 0):
                    break
                ok = False
            if not ok:
                probe.release()
                last_err = f"{item['node']} opened but produced no frame"
                continue
            cap = probe
            chosen = item
            break
        if cap is None or chosen is None:
            raise RuntimeError(last_err)
        w = int(frame.shape[1] if frame is not None else cap.get(cv2.CAP_PROP_FRAME_WIDTH) or req_w or 640)
        h = int(frame.shape[0] if frame is not None else cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or req_h or 480)
        want_w, want_h = req_w, req_h
        if want_w >= 1280 and (abs(w - want_w) > 16 or abs(h - want_h) > 16):
            cap.release()
            raise RuntimeError(f"{chosen['node']} opened {w}x{h}, wanted {want_w}x{want_h} MJPG")
        model = pinhole_guess_v4l(w, h)
        try:
            from multicam_calib.calib.orbbec_handeye import load_orbbec_color_intrinsics

            saved = load_orbbec_color_intrinsics(factory=model.as_intrinsics(), image_size=(w, h))
            model = PinholeModel(
                K=np.asarray(saved.K, dtype=np.float64),
                dist=np.asarray(saved.dist, dtype=np.float64),
                image_size=(int(saved.image_size[0]), int(saved.image_size[1])),
                source=str(getattr(saved, "source", None) or "chessboard"),
            )
        except Exception:  # noqa: BLE001
            pass
        serial = chosen["serial"] or wanted
        name = chosen["name"] or "Orbbec V4L2"
        self._cap = cap
        self._params = OrbbecFactoryParams(
            serial=serial,
            model=name,
            color=model,
            depth=None,
            T_color_depth=None,
        )
        self._backend = "v4l2"
        self._opened = True
        return self._params

    def close(self) -> None:
        pipe = self._pipeline
        cap = self._cap
        v1 = self._v1
        self._pipeline = None
        self._align_filter = None
        self._pc_filter = None
        self._cap = None
        self._v1 = None
        self._backend = ""
        self._opened = False
        self._emitter = {}
        if v1 is not None:
            try:
                v1.close()
            except Exception:  # noqa: BLE001
                pass
        if pipe is not None:
            try:
                pipe.stop()
            except Exception:  # noqa: BLE001
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass

    def read(self, timeout_ms: int = 2000) -> OrbbecRGBDFrame:
        if self._backend == "v1":
            return self._read_v1(timeout_ms)
        if self._backend == "v4l2":
            return self._read_v4l2(timeout_ms)
        if self._pipeline is None:
            raise RuntimeError("Orbbec session is not open")
        frames = self._pipeline.wait_for_frames(int(timeout_ms))
        if frames is None:
            raise TimeoutError("Orbbec wait_for_frames timed out")
        if self._align_filter is not None:
            try:
                aligned = self._align_filter.process(frames)
                if aligned is not None:
                    frames = aligned
            except Exception:  # noqa: BLE001
                pass
        color_f = frames.get_color_frame()
        depth_f = frames.get_depth_frame()
        if color_f is None or depth_f is None:
            raise RuntimeError("Orbbec frameset missing color or depth")
        color = frame_to_bgr(color_f)
        depth_m = self._depth_on_color_grid(color, depth_frame_to_meters(depth_f))
        return OrbbecRGBDFrame(
            color_bgr=color,
            depth_m=depth_m,
            timestamp_ns=time.monotonic_ns(),
            color_size=(color.shape[1], color.shape[0]),
            depth_size=(depth_m.shape[1], depth_m.shape[0]),
        )

    def _read_v1(self, timeout_ms: int = 2000) -> OrbbecRGBDFrame:
        if self._v1 is None:
            raise RuntimeError("Orbbec v1 session is not open")
        reply = self._v1.request({"op": "read", "timeout_ms": int(timeout_ms)}, timeout=max(1.0, timeout_ms / 1000.0 + 0.5))
        if not reply.get("ok"):
            raise TimeoutError(str(reply.get("error") or "v1 read failed"))
        color = np.ascontiguousarray(reply["color"])
        depth_m = self._depth_on_color_grid(color, np.ascontiguousarray(reply["depth"], dtype=np.float32))
        return OrbbecRGBDFrame(
            color_bgr=color,
            depth_m=depth_m,
            timestamp_ns=int(reply.get("t_ns") or time.monotonic_ns()),
            color_size=(color.shape[1], color.shape[0]),
            depth_size=(depth_m.shape[1], depth_m.shape[0]),
        )

    def _depth_on_color_grid(self, color: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
        if depth_m.shape[:2] == color.shape[:2]:
            return depth_m
        params = self._params
        if params is not None and params.depth is not None:
            depth_k = np.asarray(params.depth.K, dtype=np.float64).copy()
            if self.cfg.depth_flip_h:
                depth_k[0, 2] = float(depth_m.shape[1] - 1) - depth_k[0, 2]
            return warp_depth_to_color(
                depth_m,
                depth_k,
                (color.shape[1], color.shape[0]),
                params.color.K,
                params.T_color_depth,
            )
        return cv2.resize(depth_m, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)

    def _read_v4l2(self, timeout_ms: int = 2000) -> OrbbecRGBDFrame:
        if self._cap is None:
            raise RuntimeError("Orbbec V4L2 session is not open")
        deadline = time.monotonic() + max(0.05, float(timeout_ms) / 1000.0)
        color = None
        while time.monotonic() < deadline:
            # Drop queued frames so the preview is the latest, not 3–4 frames late.
            for _ in range(4):
                if not self._cap.grab():
                    break
            ok, frame = self._cap.retrieve()
            if ok and frame is not None and getattr(frame, "size", 0):
                color = frame
                break
        if color is None:
            raise TimeoutError("Orbbec V4L2 read timed out")
        h, w = color.shape[:2]
        depth_m = np.zeros((h, w), dtype=np.float32)
        return OrbbecRGBDFrame(
            color_bgr=color,
            depth_m=depth_m,
            timestamp_ns=time.monotonic_ns(),
            color_size=(w, h),
            depth_size=(w, h),
        )

    def build_cloud(
        self,
        frame: OrbbecRGBDFrame,
        *,
        min_m: float,
        max_m: float,
        min_valid: int,
        min_valid_frac: float,
        stride: int = 2,
    ) -> tuple[np.ndarray, np.ndarray | None, str]:
        """Return (xyz, rgb, source). ``source`` is ``sdk`` or ``unproject``."""
        xyz, rgb, source = self._try_sdk_cloud(frame)
        if xyz is None or xyz.size == 0:
            assert self._params is not None
            xyz, rgb = unproject_aligned_depth(
                frame.depth_m,
                self._params.color.K,
                color_bgr=frame.color_bgr,
                stride=int(max(stride, 1)),
                min_m=min_m,
                max_m=max_m,
            )
            source = "unproject"
        stats = point_cloud_stats(
            xyz, min_m=min_m, max_m=max_m, min_valid=min_valid, min_valid_frac=min_valid_frac
        )
        return xyz, rgb, f"{source}; {stats.detail}"

    def _try_sdk_cloud(
        self, frame: OrbbecRGBDFrame
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        if self._pipeline is None or self._pc_filter is None:
            return None, None, "sdk"
        try:
            frames = self._pipeline.wait_for_frames(200)
            if frames is None:
                return None, None, "sdk"
            if self._align_filter is not None:
                aligned = self._align_filter.process(frames)
                if aligned is not None:
                    frames = aligned
            pc_frame = self._pc_filter.process(frames)
            if pc_frame is None:
                return None, None, "sdk"
            if hasattr(self._pc_filter, "calculate"):
                arr = np.asarray(self._pc_filter.calculate(pc_frame))
            else:
                arr = np.asarray(pc_frame.get_data())
            arr = np.reshape(arr, (-1, arr.shape[-1] if arr.ndim > 1 else 3))
            if arr.shape[1] >= 6:
                xyz = arr[:, :3].astype(np.float32)
                rgb = arr[:, 3:6].astype(np.uint8)
            else:
                xyz = arr[:, :3].astype(np.float32)
                rgb = None
            # Orbbec often returns millimeters.
            if xyz.size and float(np.nanmedian(np.abs(xyz[:, 2]))) > 20.0:
                xyz = xyz * 1e-3
            return xyz, rgb, "sdk"
        except Exception:  # noqa: BLE001
            return None, None, "sdk"


def frame_to_bgr(color_frame: Any) -> np.ndarray:
    width = int(_call(color_frame, "get_width"))
    height = int(_call(color_frame, "get_height"))
    fmt = _call(color_frame, "get_format")
    data = np.asanyarray(_call(color_frame, "get_data"))
    fmt_enum = _enum(ob, "OBFormat")
    name = str(fmt)
    if fmt == getattr(fmt_enum, "RGB", None) or fmt == getattr(fmt_enum, "RGB888", None) or "RGB" in name:
        img = np.resize(data, (height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if fmt == getattr(fmt_enum, "BGR", None) or "BGR" in name:
        return np.resize(data, (height, width, 3)).copy()
    if fmt == getattr(fmt_enum, "MJPG", None) or "MJPG" in name or "MJPEG" in name:
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Orbbec MJPG decode failed")
        return decoded
    if fmt == getattr(fmt_enum, "YUYV", None) or "YUYV" in name:
        return cv2.cvtColor(np.resize(data, (height, width, 2)), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == getattr(fmt_enum, "UYVY", None) or "UYVY" in name:
        return cv2.cvtColor(np.resize(data, (height, width, 2)), cv2.COLOR_YUV2BGR_UYVY)
    # Last resort: treat as RGB888.
    return cv2.cvtColor(np.resize(data, (height, width, 3)), cv2.COLOR_RGB2BGR)


def depth_frame_to_meters(depth_frame: Any) -> np.ndarray:
    width = int(_call(depth_frame, "get_width"))
    height = int(_call(depth_frame, "get_height"))
    raw = np.asanyarray(_call(depth_frame, "get_data"))
    if raw.size == width * height * 2 and raw.dtype == np.uint8:
        raw = raw.view(np.uint16)
    depth = np.asarray(raw, dtype=np.uint16).reshape(height, width)
    scale = _call(depth_frame, "get_depth_scale", "getValueScale", default=1.0)
    scale = float(scale)
    # scale is millimeters per unit on most Orbbec modules.
    return (depth.astype(np.float32) * scale) * 1e-3


def _pick_video_profile(pipeline: Any, sensor_type: Any, width: int, height: int, fps: int) -> Any:
    profiles = pipeline.get_stream_profile_list(sensor_type)
    if width > 0 and height > 0:
        try:
            fmt = _enum(ob, "OBFormat")
            wanted = getattr(fmt, "RGB888", getattr(fmt, "YUYV", getattr(fmt, "Y16", None)))
            return profiles.get_video_stream_profile(int(width), int(height), wanted, int(fps))
        except Exception:  # noqa: BLE001
            pass
        try:
            return profiles.get_video_stream_profile(int(width), int(height), 0, int(fps))
        except Exception:  # noqa: BLE001
            pass
    if hasattr(profiles, "get_default_video_stream_profile"):
        return profiles.get_default_video_stream_profile()
    return profiles.get_profile(0).as_video_stream_profile()


def _device_info(device: Any | None) -> dict[str, str]:
    if device is None:
        return {"serial": "", "model": "Orbbec"}
    info = _call(device, "get_device_info")
    serial = str(_call(info, "get_serial_number", "serialNumber", default="") or "")
    model = str(_call(info, "get_name", "name", default="Orbbec") or "Orbbec")
    return {"serial": serial, "model": model}


def _select_device(serial: str) -> Any | None:
    serial = str(serial or "").strip()
    if not serial:
        return None
    ctx = ob.Context()
    listing = _call(ctx, "query_devices", "queryDeviceList")
    by_sn = getattr(listing, "get_device_by_serial_number", None)
    if callable(by_sn):
        try:
            return by_sn(serial)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Orbbec serial {serial!r} not found") from exc
    count = int(_call(listing, "get_count", "deviceCount", default=0) or 0)
    for i in range(count):
        getter = getattr(listing, "get_device_by_index", None) or getattr(listing, "getDevice", None)
        if getter is None:
            break
        dev = getter(i)
        info = _device_info(dev)
        if info["serial"] == serial:
            return dev
    raise RuntimeError(f"Orbbec serial {serial!r} not found ({count} device(s) on the bus)")


def _discover_sdk() -> list[DiscoveredCamera]:
    if _ensure_ob() is None:
        return []
    out: list[DiscoveredCamera] = []
    try:
        ctx = ob.Context()
        listing = _call(ctx, "query_devices", "queryDeviceList")
        count = int(_call(listing, "get_count", "deviceCount", default=0) or 0)
        getter = getattr(listing, "get_device_by_index", None) or getattr(listing, "getDevice", None)
        if getter is None:
            return []
        for i in range(count):
            info = _device_info(getter(i))
            if not info["serial"]:
                continue
            out.append(DiscoveredCamera(driver="orbbec", serial=info["serial"], model=info["model"]))
    except Exception:  # noqa: BLE001
        return []
    return out


def _discover_v4l() -> list[DiscoveredCamera]:
    out: list[DiscoveredCamera] = []
    seen: set[str] = set()
    for n in list_orbbec_v4l_nodes():
        if n.get("index") not in ("", "0"):
            continue
        key = n["serial"] or n["node"]
        if key in seen:
            continue
        seen.add(key)
        out.append(
            DiscoveredCamera(
                driver="orbbec",
                serial=n["serial"] or n["node"],
                model=n["name"] or "Orbbec V4L2",
                extra={"backend": "v4l2", "node": n["node"]},
            )
        )
    return out


def discover_orbbec() -> Iterable[DiscoveredCamera]:
    if _prefer_v4l2():
        v4l = _discover_v4l()
        if v4l:
            return v4l
    sdk = _discover_sdk()
    if sdk:
        return sdk
    return _discover_v4l()


_IPC_HDR = struct.Struct("!Q")


def _ipc_send(sock: socket.socket, obj: object) -> None:
    payload = pickle.dumps(obj, protocol=4)
    sock.sendall(_IPC_HDR.pack(len(payload)) + payload)


def _ipc_recv(sock: socket.socket) -> object:
    raw = _ipc_readexact(sock, _IPC_HDR.size)
    (n,) = _IPC_HDR.unpack(raw)
    if n > 64 * 1024 * 1024:
        raise RuntimeError(f"v1 message too large: {n}")
    return pickle.loads(_ipc_readexact(sock, n))


def _ipc_readexact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("v1 bridge socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _resolve_sdk_root(cfg: OrbbecConfig) -> Path:
    raw = Path(cfg.bundled_sdk_root)
    if raw.is_absolute() and raw.exists():
        return raw
    here = Path(__file__).resolve()
    candidates = [
        (here.parents[2] / raw).resolve(),
        (here.parents[3] / "tmp" / "OrbbecSDK_Python_v1.1.4_linux_x64_release").resolve(),
        Path("/media/camp/EXT_DRIVE/RealUS_playground/tmp/OrbbecSDK_Python_v1.1.4_linux_x64_release"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return raw


def _params_from_v1(raw: dict[str, Any]) -> OrbbecFactoryParams:
    color = raw.get("color") or {}
    depth = raw.get("depth")
    k = np.asarray(color.get("K"), dtype=np.float64).reshape(3, 3)
    dist = np.asarray(color.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1)
    size = tuple(int(x) for x in (color.get("image_size") or (640, 480)))
    color_m = align_factory_pinhole_to_stream(
        PinholeModel(K=k, dist=dist, image_size=(size[0], size[1]), source=str(color.get("source") or "factory")),
        size,
    )
    depth_m = None
    if isinstance(depth, dict) and depth.get("K") is not None:
        dk = np.asarray(depth["K"], dtype=np.float64).reshape(3, 3)
        dd = np.asarray(depth.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1)
        ds = tuple(int(x) for x in (depth.get("image_size") or size))
        depth_m = PinholeModel(K=dk, dist=dd, image_size=(ds[0], ds[1]), source="factory")
    t_cd = raw.get("T_color_depth")
    T = None
    if t_cd is not None:
        T = np.asarray(t_cd, dtype=np.float64).reshape(4, 4)
    return OrbbecFactoryParams(
        serial=str(raw.get("serial") or ""),
        model=str(raw.get("model") or "Orbbec"),
        color=color_m,
        depth=depth_m,
        T_color_depth=T,
    )


class _V1Bridge:
    def __init__(self, proc: subprocess.Popen, sock: socket.socket, sock_path: str) -> None:
        self.proc = proc
        self.sock = sock
        self.sock_path = sock_path

    @classmethod
    def start(cls, cfg: OrbbecConfig) -> "_V1Bridge":
        py = Path(cfg.v1_python)
        if not py.is_file():
            raise RuntimeError(f"v1 python missing: {py}")
        sdk = _resolve_sdk_root(cfg)
        if not sdk.exists():
            raise RuntimeError(f"v1 SDK root missing: {sdk}")
        bridge = Path(__file__).resolve().parents[3] / "scripts" / "orbbec_v1_bridge.py"
        if not bridge.is_file():
            raise RuntimeError(f"v1 bridge missing: {bridge}")
        sock_path = f"/tmp/orbbec_v1_{os.getpid()}_{time.time_ns()}.sock"
        env = os.environ.copy()
        c_lib = sdk / "python3.9" / "lib" / "c_lib"
        env["LD_LIBRARY_PATH"] = f"{c_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        env["PYTHONNOUSERSITE"] = "1"
        proc = subprocess.Popen(
            [str(py), str(bridge), sock_path, str(sdk)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(sdk / "python3.9" / "Samples"),
        )
        err_tail: list[bytes] = []

        def _drain() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                err_tail.append(line)
                if len(err_tail) > 80:
                    del err_tail[:40]

        threading.Thread(target=_drain, daemon=True).start()
        deadline = time.monotonic() + 12.0
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                extra = b"".join(err_tail[-20:]).decode("utf-8", "replace")
                raise RuntimeError(f"v1 bridge exited {proc.returncode}: {extra[-800:]}")
            assert proc.stdout is not None
            readable, _, _ = select.select([proc.stdout], [], [], 0.15)
            if readable:
                line = proc.stdout.readline()
                if line.strip() == b"READY":
                    ready = True
                    break
        if not ready:
            proc.kill()
            raise RuntimeError("v1 bridge did not become READY")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(8.0)
        try:
            sock.connect(sock_path)
        except Exception:
            proc.kill()
            raise
        return cls(proc, sock, sock_path)

    def request(self, obj: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
        self.sock.settimeout(timeout)
        _ipc_send(self.sock, obj)
        reply = _ipc_recv(self.sock)
        if not isinstance(reply, dict):
            raise RuntimeError("v1 bridge returned a non-dict")
        return reply

    def close(self) -> None:
        try:
            self.request({"op": "close"}, timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.sock.close()
        except Exception:  # noqa: BLE001
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        try:
            Path(self.sock_path).unlink()
        except OSError:
            pass
