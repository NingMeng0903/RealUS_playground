"""Genesis-side registry and ZMQ subscriber for anatomy assets."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_drawer import AnatomyLbsDrawer
from projects.genesis_ue_sync.multiview_realtime.track_stream import TOPIC_ANATOMY_ASSET_V1
from projects.genesis_ue_sync.sim_platform.simulation.runtime import GenesisPlatformRuntime

logger = logging.getLogger(__name__)


def _color_from_payload(payload: dict[str, Any], default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    raw = payload.get("color_rgba")
    if raw is None:
        return default
    vals = [float(v) for v in list(raw)]
    if len(vals) != 4:
        return default
    return tuple(vals)  # type: ignore[return-value]


class AnatomyAssetRegistry:
    def __init__(
        self,
        runtime: GenesisPlatformRuntime,
        *,
        default_color_rgba: tuple[float, float, float, float] = (0.8, 0.05, 0.05, 0.85),
        default_transparent_alpha: float = 0.35,
    ) -> None:
        self.runtime = runtime
        self.default_color_rgba = tuple(float(v) for v in default_color_rgba)
        self.default_transparent_alpha = float(default_transparent_alpha)
        self._drawers: dict[str, AnatomyLbsDrawer] = {}

    def upsert(self, *, model_id: str, asset_npz: Path | str, color_rgba: tuple[float, float, float, float] | None = None) -> AnatomyLbsDrawer:
        model = str(model_id)
        old = self._drawers.pop(model, None)
        if old is not None:
            old.clear_node()
        drawer = AnatomyLbsDrawer.from_npz(
            self.runtime,
            path=asset_npz,
            model_id=model,
            color_rgba=color_rgba or self.default_color_rgba,
        )
        self._drawers[model] = drawer
        logger.info("anatomy asset upsert model_id=%s asset=%s", model, asset_npz)
        return drawer

    def delete(self, model_id: str) -> None:
        drawer = self._drawers.pop(str(model_id), None)
        if drawer is not None:
            drawer.clear_node()
            logger.info("anatomy asset deleted model_id=%s", model_id)

    def clear_all(self) -> None:
        for drawer in list(self._drawers.values()):
            drawer.clear_node()
        self._drawers.clear()
        logger.info("all anatomy assets cleared")

    def apply_control(self, payload: dict[str, Any]) -> None:
        action = str(payload.get("action", "")).strip().lower()
        model_id = str(payload.get("model_id", "patient_anatomy"))
        if action == "upsert":
            asset_npz = payload.get("asset_npz")
            if not asset_npz:
                raise ValueError("anatomy upsert requires asset_npz")
            self.upsert(
                model_id=model_id,
                asset_npz=str(asset_npz),
                color_rgba=_color_from_payload(payload, self.default_color_rgba),
            )
            return
        if action == "delete":
            self.delete(model_id)
            return
        if action == "clear_all":
            self.clear_all()
            return
        drawer = self._drawers.get(model_id)
        if drawer is None:
            logger.warning("anatomy action ignored for unknown model_id=%s action=%s", model_id, action)
            return
        if action == "set_visible":
            drawer.set_visible(bool(payload.get("visible", True)))
        elif action == "set_opacity":
            drawer.set_opacity(float(payload.get("opacity", drawer.opacity)))
        elif action == "restore_opacity":
            drawer.restore_opacity()
        elif action == "set_render_mode":
            drawer.set_render_mode(str(payload.get("mode", "opaque")), transparent_alpha=self.default_transparent_alpha)
        else:
            logger.warning("unknown anatomy asset action=%s", action)

    def draw_all(
        self,
        pose_axis_angle: Any,
        *,
        transl: Any | None = None,
        shape_hash: str = "",
    ) -> bool:
        """Draw only assets baked for the incoming SMPL-X body shape."""
        incoming = str(shape_hash or "")
        drawn = False
        for model_id, drawer in list(self._drawers.items()):
            expected = str((drawer.asset.metadata or {}).get("shape_hash", ""))
            if incoming and expected and incoming != expected:
                logger.error(
                    "anatomy drive rejected model_id=%s shape_hash=%s asset_shape_hash=%s",
                    model_id,
                    incoming,
                    expected,
                )
                drawer.clear_node()
                continue
            drawn = bool(drawer.draw(pose_axis_angle, transl=transl)) or drawn
        return drawn

    def canonical_pelvis(self) -> np.ndarray | None:
        """Canonical-frame pelvis joint shared by registered assets (None if empty)."""
        for drawer in self._drawers.values():
            joints = np.asarray(drawer.asset.rest_joints, dtype=np.float32).reshape(-1, 3)
            if joints.size:
                return joints[0]
        return None

    @property
    def model_ids(self) -> list[str]:
        return sorted(self._drawers)


class AnatomyAssetSubscriber:
    def __init__(self, registry: AnatomyAssetRegistry, *, connect: str, topic: str = TOPIC_ANATOMY_ASSET_V1) -> None:
        self.registry = registry
        self.connect = str(connect)
        self.topic = str(topic).encode("utf-8")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: Any = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._recv_loop, name="AnatomyAssetSubscriber", daemon=True)
        self._thread.start()
        logger.info("anatomy asset subscriber started connect=%s", self.connect)

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass

    def close(self) -> None:
        self.stop()

    def _recv_loop(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        self._sock = sock
        sock.setsockopt(zmq.RCVTIMEO, 100)
        sock.connect(self.connect)
        sock.setsockopt(zmq.SUBSCRIBE, self.topic)
        while not self._stop.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue
            except Exception:
                if self._stop.is_set():
                    break
                continue
            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[-1].decode("utf-8"))
                if str(payload.get("payload_kind")) != "anatomy_asset":
                    continue
                self.registry.apply_control(payload)
            except Exception as exc:
                logger.warning("anatomy asset control failed: %s", exc)
