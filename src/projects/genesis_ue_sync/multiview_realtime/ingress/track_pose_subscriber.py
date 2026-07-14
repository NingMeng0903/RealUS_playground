from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.viz.track_capsule_drawer import TrackCapsuleDrawer
from projects.genesis_ue_sync.multiview_realtime.viz.track_mesh_drawer import TrackMeshDrawer
from projects.genesis_ue_sync.multiview_realtime.viz.track_vertices_drawer import TrackVerticesDrawer
from projects.genesis_ue_sync.multiview_realtime.track_stream import TOPIC_MULTIVIEW_TRACK_V1

logger = logging.getLogger(__name__)


@dataclass
class ReceivedTrackPose:
    frame_index: int
    timestamp_ns: int
    payload_kind: str
    translation_m: np.ndarray
    keypoints3d: np.ndarray | None = None
    keypoints3d_schema: str = "body25"
    pose_aa: np.ndarray | None = None
    betas: np.ndarray | None = None
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    mesh_schema: str = ""
    anatomy_pose_aa: np.ndarray | None = None
    anatomy_transl: np.ndarray | None = None
    reason: str = ""
    shape_hash: str = ""
    pose_hash: str = ""
    gender: str = ""
    betas: np.ndarray | None = None


class TrackPoseSubscriber:
    """SUB latest track keypoints from the dedicated multiview track worker."""

    def __init__(
        self,
        runtime: Any,
        *,
        connect: str,
        topic: str = TOPIC_MULTIVIEW_TRACK_V1,
        mesh_rgba: tuple[int, int, int, int] = (250, 122, 31, 235),
        display_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        smpl_model_dir: str = "dataset/intermediate/humans/body_models/smpl",
        device: str = "cpu",
        default_betas: np.ndarray | None = None,
        display_blend_alpha: float = 0.85,
        draw_keypoints_fallback: bool = False,
    ) -> None:
        self._mesh_drawer = TrackMeshDrawer(
            runtime,
            smpl_model_dir=smpl_model_dir,
            mesh_rgba=mesh_rgba,
            device=device,
            display_offset_m=display_offset_m,
        )
        self._capsule_drawer = TrackCapsuleDrawer(
            runtime,
            mesh_rgba=mesh_rgba,
            display_offset_m=display_offset_m,
        )
        self._vertices_drawer = TrackVerticesDrawer(
            runtime,
            mesh_rgba=mesh_rgba,
            display_offset_m=display_offset_m,
        )
        self._topic = str(topic).encode("utf-8")
        self._connect = str(connect)
        self._latest: ReceivedTrackPose | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock = None
        self._logged_first_pose = False
        self._logged_first_draw = False
        self._logged_draw_failure = False
        self._last_drawn_frame_index: int | None = None
        self._default_betas = (
            np.asarray(default_betas, dtype=np.float32).reshape(-1)[:10]
            if default_betas is not None
            else np.zeros(10, dtype=np.float32)
        )
        self._target_frame_index: int | None = None
        self._target_pose_aa: np.ndarray | None = None
        self._target_transl: np.ndarray | None = None
        self._target_betas: np.ndarray | None = None
        self._display_pose_aa: np.ndarray | None = None
        self._display_transl: np.ndarray | None = None
        self._display_blend_alpha = float(np.clip(display_blend_alpha, 0.0, 1.0))
        self._draw_keypoints_fallback = bool(draw_keypoints_fallback)

    @classmethod
    def from_env(cls, runtime: Any) -> "TrackPoseSubscriber | None":
        raw = str(os.environ.get("AMONGUS_GENESIS_TRACK_SUBSCRIBE", "") or "").strip()
        if not raw or raw.lower() in ("0", "false", "no", "off"):
            return None
        rgba_raw = os.environ.get("AMONGUS_GENESIS_TRACK_MESH_RGBA", "250,122,31,132")
        parts = [int(x.strip()) for x in str(rgba_raw).split(",") if x.strip()]
        if len(parts) != 4:
            parts = [250, 122, 31, 132]
        device = str(os.environ.get("AMONGUS_GENESIS_TRACK_DEVICE", "cpu") or "cpu")
        off_raw = os.environ.get("AMONGUS_GENESIS_TRACK_DISPLAY_OFFSET_M", "")
        off_parts = [float(x.strip()) for x in str(off_raw).split(",") if x.strip()]
        offset = tuple(off_parts) if len(off_parts) == 3 else (0.0, 0.0, 0.0)
        smpl_dir = str(os.environ.get("AMONGUS_GENESIS_TRACK_SMPL_DIR", "dataset/intermediate/humans/body_models/smpl"))
        blend_alpha = float(os.environ.get("AMONGUS_GENESIS_TRACK_BLEND_ALPHA", "0.85") or "0.85")
        return cls(
            runtime,
            connect=raw,
            mesh_rgba=tuple(parts),
            device=device,
            display_offset_m=offset,
            smpl_model_dir=smpl_dir,
            display_blend_alpha=blend_alpha,
            draw_keypoints_fallback=str(os.environ.get("AMONGUS_GENESIS_TRACK_DRAW_KEYPOINTS_FALLBACK", "")).strip().lower()
            in {"1", "true", "yes", "on"},
        )

    def start(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq required for track pose subscriber.") from exc
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 50)
        sock.connect(self._connect)
        sock.setsockopt(zmq.SUBSCRIBE, self._topic)
        self._sock = sock
        self._thread = threading.Thread(target=self._recv_loop, name="track-pose-sub", daemon=True)
        self._thread.start()
        logger.info("TrackPoseSubscriber connect=%s", self._connect)

    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                import zmq

                parts = self._sock.recv_multipart()
            except zmq.Again:
                continue
            except Exception:
                if self._stop.is_set():
                    break
                continue
            if len(parts) < 2:
                continue
            try:
                data = json.loads(parts[-1].decode("utf-8"))
                translation = np.asarray(data.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
                kind = str(data.get("payload_kind", "keypoints3d"))
                pose = ReceivedTrackPose(
                    frame_index=int(data["frame_index"]),
                    timestamp_ns=int(data.get("timestamp_ns", 0)),
                    payload_kind=kind,
                    translation_m=translation,
                    reason=str(data.get("reason", "")),
                )
                if kind == "smpl_pose":
                    pose.pose_aa = np.asarray(data["pose_aa"], dtype=np.float32).reshape(-1)
                    raw_betas = data.get("betas")
                    if raw_betas is not None and float(np.max(np.abs(np.asarray(raw_betas)))) > 0.0:
                        pose.betas = np.asarray(raw_betas, dtype=np.float32).reshape(-1)
                    else:
                        pose.betas = self._default_betas.copy()
                elif kind == "mesh_vertices":
                    pose.vertices = np.asarray(data["vertices"], dtype=np.float32).reshape(-1, 3)
                    pose.faces = np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3)
                    pose.mesh_schema = str(data.get("mesh_schema", ""))
                    pose.shape_hash = str(data.get("shape_hash", ""))
                    pose.pose_hash = str(data.get("pose_hash", ""))
                    pose.gender = str(data.get("gender", ""))
                    if "betas" in data:
                        pose.betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)[:10]
                    if "smplx_pose_aa_165" in data:
                        full = np.asarray(data["smplx_pose_aa_165"], dtype=np.float32).reshape(-1)
                        if full.size != 165:
                            raise ValueError(f"invalid smplx_pose_aa_165 length {full.size}")
                        pose.anatomy_pose_aa = full
                        pose.anatomy_transl = np.asarray(data.get("Th", translation), dtype=np.float32).reshape(3)
                    elif "Rh" in data and "poses" in data:
                        from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55

                        pose.anatomy_pose_aa = easymocap_fit_to_smplx55(
                            data["Rh"], data["poses"], gender=str(data.get("gender", "male"))
                        ).reshape(-1)
                        if "Th" in data:
                            pose.anatomy_transl = np.asarray(data["Th"], dtype=np.float32).reshape(3)
                        else:
                            pose.anatomy_transl = translation.copy()
                elif kind == "clear":
                    pass
                else:
                    pose.keypoints3d = np.asarray(data["keypoints3d"], dtype=np.float32).reshape(-1, 4)
                    pose.keypoints3d_schema = str(data.get("keypoints3d_schema", "body25"))
                with self._lock:
                    self._latest = pose
                if not self._logged_first_pose:
                    self._logged_first_pose = True
                    logger.info(
                        "track pose received frame=%s kind=%s trans_m=%s",
                        pose.frame_index,
                        kind,
                        [round(float(v), 3) for v in pose.translation_m.tolist()],
                    )
            except Exception as exc:
                logger.warning("track pose parse failed: %s", exc)

    def latest_anatomy_drive(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (pose_aa flat, transl) from latest UE/terminal-8 SMPL-X fit, if present."""
        with self._lock:
            pose = self._latest
        if pose is None or str(pose.payload_kind) == "clear":
            return None
        if pose.anatomy_pose_aa is not None and pose.anatomy_transl is not None:
            return (
                np.asarray(pose.anatomy_pose_aa, dtype=np.float32).reshape(-1),
                np.asarray(pose.anatomy_transl, dtype=np.float32).reshape(3),
            )
        if pose.payload_kind == "smpl_pose" and pose.pose_aa is not None:
            return (
                np.asarray(pose.pose_aa, dtype=np.float32).reshape(-1),
                np.asarray(pose.translation_m, dtype=np.float32).reshape(3),
            )
        return None

    def poll_draw(self) -> bool:
        with self._lock:
            pose = self._latest
        if pose is None:
            return False
        try:
            if pose.payload_kind == "clear":
                self._mesh_drawer.clear()
                self._capsule_drawer.clear()
                self._vertices_drawer.clear()
                self._target_frame_index = None
                self._target_pose_aa = None
                self._target_transl = None
                self._target_betas = None
                self._display_pose_aa = None
                self._display_transl = None
                return True
            if pose.payload_kind == "smpl_pose" and pose.pose_aa is not None and pose.betas is not None:
                self._capsule_drawer.clear()
                self._vertices_drawer.clear()
                return self._poll_draw_smpl(pose)
            if pose.payload_kind == "mesh_vertices" and pose.vertices is not None and pose.faces is not None:
                if self._last_drawn_frame_index == int(pose.frame_index):
                    return False
                self._mesh_drawer.clear()
                self._capsule_drawer.clear()
                drew = self._vertices_drawer.draw(pose.vertices, pose.faces)
                if drew:
                    self._last_drawn_frame_index = int(pose.frame_index)
                    if not self._logged_first_draw:
                        self._logged_first_draw = True
                        logger.info("track orange mesh vertices drawn in Genesis viewer")
                return drew
            if pose.keypoints3d is not None:
                if not self._draw_keypoints_fallback:
                    self._capsule_drawer.clear()
                    return False
                if self._last_drawn_frame_index == int(pose.frame_index):
                    return False
                self._mesh_drawer.clear()
                self._vertices_drawer.clear()
                self._capsule_drawer.draw(pose.keypoints3d, pose.keypoints3d_schema)
                self._last_drawn_frame_index = int(pose.frame_index)
                if not self._logged_first_draw:
                    self._logged_first_draw = True
                    logger.info("track orange capsule human drawn in Genesis viewer")
                return True
            return False
        except Exception as exc:
            logger.warning("track draw failed: %s", exc)
            if not self._logged_draw_failure:
                self._logged_draw_failure = True
                print(f"amass_bed_capsule_demo: track orange draw failed: {exc}", flush=True)
            return False

    def _poll_draw_smpl(self, pose: ReceivedTrackPose) -> bool:
        assert pose.pose_aa is not None and pose.betas is not None
        target_pose = np.asarray(pose.pose_aa, dtype=np.float32).reshape(-1)
        target_trans = np.asarray(pose.translation_m, dtype=np.float32).reshape(3)
        target_betas = np.asarray(pose.betas, dtype=np.float32).reshape(-1)
        if self._target_frame_index != int(pose.frame_index):
            self._target_frame_index = int(pose.frame_index)
            self._target_pose_aa = target_pose.copy()
            self._target_transl = target_trans.copy()
            self._target_betas = target_betas.copy()
            if self._display_pose_aa is None:
                self._display_pose_aa = target_pose.copy()
                self._display_transl = target_trans.copy()

        assert self._target_pose_aa is not None and self._target_transl is not None
        assert self._display_pose_aa is not None and self._display_transl is not None
        alpha = float(self._display_blend_alpha)
        if not np.allclose(self._display_pose_aa, self._target_pose_aa, rtol=0.0, atol=1.0e-4):
            self._display_pose_aa = (1.0 - alpha) * self._display_pose_aa + alpha * self._target_pose_aa
        if not np.allclose(self._display_transl, self._target_transl, rtol=0.0, atol=1.0e-5):
            self._display_transl = (1.0 - alpha) * self._display_transl + alpha * self._target_transl
        if (
            np.allclose(self._display_pose_aa, self._target_pose_aa, rtol=0.0, atol=1.0e-4)
            and np.allclose(self._display_transl, self._target_transl, rtol=0.0, atol=1.0e-5)
        ):
            self._display_pose_aa = self._target_pose_aa.copy()
            self._display_transl = self._target_transl.copy()

        betas = self._target_betas if self._target_betas is not None else target_betas
        drew = bool(self._mesh_drawer.draw(self._display_pose_aa, betas, self._display_transl))
        if not drew:
            return False
        if not self._logged_first_draw:
            self._logged_first_draw = True
            logger.info("track orange SMPL mesh drawn in Genesis viewer")
        return True

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._vertices_drawer.clear()
