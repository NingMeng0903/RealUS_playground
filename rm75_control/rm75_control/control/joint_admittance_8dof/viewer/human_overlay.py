"""Optional human/anatomy/canonical overlays for the RM75 Genesis twin (Window B).

Requires REALUS/Among_US PYTHONPATH (src/) and genesis env. Soft-imports so twin
still runs without perception packages.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TwinHumanOverlayConfig:
    track_subscribe: str = "tcp://127.0.0.1:5598"
    anatomy_subscribe: str = "tcp://127.0.0.1:5601"
    canonical_bind: str = "tcp://127.0.0.1:5599"
    canonical_human_source: str = "none"  # none | robot | fitted
    smplx_npz: Path | None = None
    track_mesh_rgba: tuple[int, int, int, int] = (250, 122, 31, 55)
    anatomy_opaque: bool = True
    enable_track: bool = True
    enable_anatomy: bool = True
    enable_canonical: bool = True


class TwinHumanOverlay:
    """Poll track mesh + anatomy; optionally publish fitted human on canonical ZMQ."""

    def __init__(self, scene: Any, config: TwinHumanOverlayConfig) -> None:
        self._scene = scene
        self._cfg = config
        self._track = None
        self._anatomy_reg = None
        self._anatomy_sub = None
        self._canonical_pub = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_pose55: np.ndarray | None = None
        self._latest_transl: np.ndarray | None = None
        self._robot_q_fn: Callable[[], list[float] | np.ndarray | None] | None = None

    def set_robot_q_provider(self, fn: Callable[[], list[float] | np.ndarray | None]) -> None:
        self._robot_q_fn = fn

    def start(self) -> None:
        if self._cfg.enable_track:
            self._start_track()
        if self._cfg.enable_anatomy:
            self._start_anatomy()
        if self._cfg.enable_canonical and self._cfg.canonical_human_source in ("fitted", "robot"):
            self._start_canonical()
        if self._cfg.smplx_npz is not None and Path(self._cfg.smplx_npz).is_file():
            try:
                from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import load_easymocap_smplx_fit_drive

                pose55, transl = load_easymocap_smplx_fit_drive(self._cfg.smplx_npz)
                self._latest_pose55 = pose55
                self._latest_transl = transl
                logger.info("loaded static smplx fit drive from %s", self._cfg.smplx_npz)
            except Exception as exc:
                logger.warning("failed to load smplx npz: %s", exc)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="twin-human-overlay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._track is not None:
            try:
                self._track.stop()
            except Exception:
                pass
        if self._anatomy_sub is not None:
            try:
                self._anatomy_sub.stop()
            except Exception:
                pass

    def _start_track(self) -> None:
        try:
            from projects.genesis_ue_sync.multiview_realtime.ingress.track_pose_subscriber import TrackPoseSubscriber
        except Exception as exc:
            logger.warning("TrackPoseSubscriber unavailable: %s", exc)
            return
        # TrackPoseSubscriber expects a GenesisPlatformRuntime-like object with scene debug draw.
        # For RailGenesisScene we attach a thin adapter if needed.
        runtime = getattr(self._scene, "amongus_runtime", None) or _RailSceneRuntimeAdapter(self._scene)
        self._track = TrackPoseSubscriber(
            runtime,
            connect=str(self._cfg.track_subscribe),
            device="cuda",
            default_betas=np.zeros(10, dtype=np.float32),
            mesh_rgba=self._cfg.track_mesh_rgba,
        )
        self._track.start()
        logger.info("track subscribe %s", self._cfg.track_subscribe)

    def _start_anatomy(self) -> None:
        try:
            from projects.genesis_ue_sync.anatomy_retarget.genesis_control import (
                AnatomyAssetRegistry,
                AnatomyAssetSubscriber,
            )
        except Exception as exc:
            logger.warning("anatomy overlay unavailable: %s", exc)
            return
        runtime = getattr(self._scene, "amongus_runtime", None) or _RailSceneRuntimeAdapter(self._scene)
        self._anatomy_reg = AnatomyAssetRegistry(runtime, default_color_rgba=(0.2, 0.75, 0.35, 0.55))
        self._anatomy_sub = AnatomyAssetSubscriber(self._anatomy_reg, connect=str(self._cfg.anatomy_subscribe))
        self._anatomy_sub.start()
        logger.info("anatomy subscribe %s", self._cfg.anatomy_subscribe)

    def _ensure_anatomy_opaque(self) -> None:
        if self._anatomy_reg is None or not self._cfg.anatomy_opaque:
            return
        for model_id in self._anatomy_reg.model_ids:
            drawer = self._anatomy_reg._drawers.get(model_id)
            if drawer is None:
                continue
            if getattr(drawer, "_realus_opaque_layer", False):
                continue
            try:
                drawer.set_render_mode("opaque")
                drawer._realus_opaque_layer = True
            except Exception:
                pass

    def _start_canonical(self) -> None:
        import zmq

        try:
            from projects.genesis_ue_sync.integrations.controller_bus.stream_schemas import (
                TOPIC_CANONICAL_SCENE_V1,
            )
        except Exception:
            TOPIC_CANONICAL_SCENE_V1 = "amongus_canonical_v1"
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 200)
        sock.bind(str(self._cfg.canonical_bind))
        self._canonical_pub = (sock, TOPIC_CANONICAL_SCENE_V1.encode("utf-8"))
        logger.info("canonical PUB %s", self._cfg.canonical_bind)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                logger.debug("overlay poll error: %s", exc)
            self._stop.wait(1.0 / 30.0)

    def poll_once(self) -> None:
        if self._track is not None:
            try:
                self._track.poll_draw()
            except Exception:
                pass
            drive = None
            try:
                drive = self._track.latest_anatomy_drive()
            except Exception:
                drive = None
            if drive is not None:
                self._latest_pose55, self._latest_transl = drive
        if self._anatomy_reg is not None and self._latest_pose55 is not None:
            try:
                from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import anatomy_transl_from_track_drive

                self._ensure_anatomy_opaque()
                pelvis = self._anatomy_reg.canonical_pelvis()
                transl = anatomy_transl_from_track_drive(
                    self._latest_pose55,
                    self._latest_transl,
                    pelvis,
                )
                shape_hash = self._track.latest_anatomy_shape_hash() if self._track is not None else ""
                self._anatomy_reg.draw_all(self._latest_pose55, transl=transl, shape_hash=shape_hash)
            except Exception:
                pass
        if self._canonical_pub is not None:
            if self._cfg.canonical_human_source == "robot":
                self._publish_canonical_robot_only()
            elif self._latest_pose55 is not None:
                self._publish_canonical()

    def _publish_canonical_robot_only(self) -> None:
        import json
        import time

        robot_entities: dict[str, Any] = {}
        if self._robot_q_fn is not None:
            q = self._robot_q_fn()
            if q is not None:
                qv = np.asarray(q, dtype=np.float32).reshape(-1)
                robot_entities["robot_main"] = {
                    "joint_positions": [float(v) for v in qv.tolist()],
                }
        if not robot_entities:
            return
        now = int(time.time_ns())
        payload = {
            "schema_version": 1,
            "sim_step_index": 0,
            "frame_index": 0,
            "wall_time_ns": now,
            "sim_time_ns": now,
            "source_time_ns": now,
            "clock_domain": "realus_twin",
            "robot_entities": robot_entities,
            "human": {},
            "objects": {},
            "contacts": [],
            "extras": {"canonical_human_source": "none"},
        }
        sock, topic = self._canonical_pub
        sock.send_multipart([topic, json.dumps(payload, ensure_ascii=True).encode("utf-8")])

    def _publish_canonical(self) -> None:
        import json
        import time

        from projects.genesis_ue_sync.sim_platform.sync.canonical_human_motion import (
            amongus_human_payload_from_motion_frame,
        )

        pose55 = np.asarray(self._latest_pose55, dtype=np.float32).reshape(-1)
        # pose55 is flat 55*3; canonical wants root rvec + 23*3 body. Map first 24 joints worth.
        root = pose55[:3]
        body21 = pose55[3 : 3 + 21 * 3]
        # Pad to 23 body joints (hands zero) for UE SMPL bone list.
        body23 = np.zeros(23 * 3, dtype=np.float32)
        body23[: body21.size] = body21
        smpl_pose_row = np.concatenate([root, body23]).astype(np.float32)
        transl = np.asarray(self._latest_transl if self._latest_transl is not None else [0, 0, 0], dtype=np.float32)
        human = amongus_human_payload_from_motion_frame(
            frame_index=0,
            motion_fps=30.0,
            root_translation_world_m=transl,
            smpl_pose_row=smpl_pose_row,
        )
        robot_entities: dict[str, Any] = {}
        if self._robot_q_fn is not None:
            q = self._robot_q_fn()
            if q is not None:
                qv = np.asarray(q, dtype=np.float32).reshape(-1)
                robot_entities["robot_main"] = {
                    "joint_positions": [float(v) for v in qv.tolist()],
                }
        now = int(time.time_ns())
        payload = {
            "schema_version": 1,
            "sim_step_index": 0,
            "frame_index": 0,
            "wall_time_ns": now,
            "sim_time_ns": now,
            "source_time_ns": now,
            "clock_domain": "realus_twin",
            "robot_entities": robot_entities,
            "human": human,
            "objects": {},
            "contacts": [],
            "extras": {"canonical_human_source": "fitted_easymocap"},
        }
        sock, topic = self._canonical_pub
        sock.send_multipart([topic, json.dumps(payload, ensure_ascii=True).encode("utf-8")])


class _RailSceneRuntimeAdapter:
    """Minimal adapter so TrackPoseSubscriber / AnatomyAssetRegistry can draw on RailGenesisScene."""

    def __init__(self, scene: Any) -> None:
        self.scene = getattr(scene, "scene", scene)
        self._debug_objects: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scene, name)
