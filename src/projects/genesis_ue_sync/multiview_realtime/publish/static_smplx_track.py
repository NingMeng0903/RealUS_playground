"""Publish saved Terminal-8 SMPL-X fit results to the Genesis track ZMQ topic."""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.track_stream import (
    DEFAULT_TRACK_PUB_BIND,
    TOPIC_MULTIVIEW_TRACK_V1,
    track_keypoints3d_to_dict,
    track_mesh_vertices_to_dict,
    track_pose_to_dict,
)

logger = logging.getLogger(__name__)

_SUPPORTED_SMPLX_GENDERS = frozenset(("male", "female", "neutral"))


def _stored_smplx_gender(smpl_npz: Any) -> str | None:
    """Read optional result provenance, preserving legacy NPZ compatibility."""

    if "gender" not in smpl_npz.files:
        return None
    raw = np.asarray(smpl_npz["gender"]).reshape(-1)
    if raw.size != 1:
        raise ValueError("smplx_result.npz gender metadata must be one string")
    value = raw[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    stored = str(value).strip().lower()
    if stored not in _SUPPORTED_SMPLX_GENDERS:
        raise ValueError(
            "smplx_result.npz has unsupported gender metadata; "
            f"expected one of male, female, neutral, got {stored!r}"
        )
    return stored


def smplx_output_root(repo: Path | None = None) -> Path:
    root = repo or Path(os.environ.get("REALUS_PROJECT_ROOT", "."))
    return Path(os.environ.get("REALUS_SMPLX_OUTPUT_ROOT", root / "smplx_outputs"))


def resolve_moment_dir(
    *,
    run: str | Path | None = None,
    moment_dir: str | Path | None = None,
    npz: str | Path | None = None,
    output_root: Path | None = None,
) -> Path:
    """Resolve a moment folder that contains ``smplx_result.npz``."""
    if moment_dir is not None:
        path = Path(moment_dir)
        if not (path / "smplx_result.npz").is_file() and npz is None:
            raise FileNotFoundError(f"smplx_result.npz not found under {path}")
        return path

    if npz is not None:
        npz_path = Path(npz)
        if not npz_path.is_file():
            raise FileNotFoundError(f"npz not found: {npz_path}")
        return npz_path.parent

    if run is None:
        raise ValueError("one of --run, --moment-dir, or --npz is required")

    run_path = Path(run)
    if not run_path.is_absolute():
        run_path = (Path(output_root) if output_root is not None else smplx_output_root()) / run_path
    moment = run_path / "moment_0000"
    if not (moment / "smplx_result.npz").is_file():
        raise FileNotFoundError(f"smplx_result.npz not found under {moment}")
    return moment


def sequence_moments(sequence_dir: Path | str) -> list[dict[str, Any]]:
    """Read successful moment rows and source timing from a sequence index."""

    root = Path(sequence_dir)
    index_path = root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"SMPL-X sequence index not found: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = payload.get("index") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"SMPL-X sequence index has no index list: {index_path}")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("fit_ok") is False:
            continue
        raw_dir = row.get("moment_dir")
        moment = Path(str(raw_dir)) if raw_dir else root / f"moment_{int(row.get('moment_id', len(out))):04d}"
        if not moment.is_absolute():
            moment = root / moment
        if not (moment / "smplx_result.npz").is_file():
            continue
        out.append(
            {
                "moment_dir": moment,
                "frame_index": int(row.get("frame_index", row.get("capture_frame_index", 0)) or 0),
                "timestamp_ns": int(row.get("timestamp_ns", 0) or 0),
                "moment_id": int(row.get("moment_id", len(out)) or 0),
            }
        )
    if not out:
        raise ValueError(f"SMPL-X sequence contains no fitted moments: {index_path}")
    return out


def load_keypoints3d_for_publish(moment_dir: Path) -> np.ndarray:
    kp_path = Path(moment_dir) / "easymocap_output" / "keypoints3d" / "000000.json"
    data = json.loads(kp_path.read_text(encoding="utf-8"))
    rows = data[0].get("keypoints3d") if isinstance(data, list) and data else None
    if rows is None:
        raise ValueError(f"Missing keypoints3d payload in {kp_path}")
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def build_static_track_payload(
    *,
    moment_dir: Path,
    publish_kind: str = "smplx_mesh",
    frame_index: int = 0,
    timestamp_ns: int | None = None,
    gender: str = "male",
) -> dict[str, Any]:
    moment_dir = Path(moment_dir)
    ts = int(time.time_ns() if timestamp_ns is None else timestamp_ns)
    gender = str(gender).strip().lower()

    if str(publish_kind) == "smplx_mesh" and gender not in _SUPPORTED_SMPLX_GENDERS:
        raise ValueError(
            "SMPL-X mesh publication gender must be one of male, female, neutral; "
            f"got {gender!r}"
        )

    if str(publish_kind) == "smpl_pose":
        smpl_npz = np.load(moment_dir / "smplx_result.npz")
        poses = np.asarray(smpl_npz["poses"], dtype=np.float32).reshape(-1)
        if poses.size != 72:
            raise ValueError(
                f"smpl_pose publish requires 72D SMPL pose, got {poses.size}; use publish_kind=keypoints3d."
            )
        return track_pose_to_dict(
            frame_index=int(frame_index),
            timestamp_ns=ts,
            pose_aa=poses,
            betas=np.asarray(smpl_npz["shapes"], dtype=np.float32).reshape(-1)[:10],
            translation_m=np.asarray(smpl_npz["Th"], dtype=np.float32).reshape(-1)[:3],
            pelvis_world_m=(
                np.asarray(smpl_npz["pelvis_world_m"], dtype=np.float32).reshape(3)
                if "pelvis_world_m" in smpl_npz
                else None
            ),
        )

    if str(publish_kind) == "smplx_mesh":
        smpl_npz = np.load(moment_dir / "smplx_result.npz")
        stored_gender = _stored_smplx_gender(smpl_npz)
        if stored_gender is not None and stored_gender != gender:
            raise ValueError(
                "SMPL-X result gender metadata does not match the requested "
                f"CLI gender: result={stored_gender!r}, requested={gender!r}; "
                "rerun with the matching --gender or use a legacy result without metadata."
            )
        vertices = np.asarray(smpl_npz["vertices"], dtype=np.float32).reshape(-1, 3)
        if "faces" not in smpl_npz:
            raise ValueError("smplx_result.npz missing faces; rerun fitting with the updated pipeline.")
        faces = np.asarray(smpl_npz["faces"], dtype=np.int32).reshape(-1, 3)
        trans = np.asarray(smpl_npz["Th"], dtype=np.float32).reshape(3)
        if "root_align_offset" in smpl_npz:
            trans = trans + np.asarray(smpl_npz["root_align_offset"], dtype=np.float32).reshape(3)
        from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
            easymocap_fit_to_smplx55,
            smplx_pose_hash,
            smplx_shape_hash,
        )

        poses87 = np.asarray(smpl_npz["poses"], dtype=np.float32).reshape(-1)
        pose55 = easymocap_fit_to_smplx55(smpl_npz["Rh"], poses87, gender=gender)
        betas = np.asarray(smpl_npz["shapes"], dtype=np.float32).reshape(-1)[:10]
        pelvis_world_m = (
            np.asarray(smpl_npz["pelvis_world_m"], dtype=np.float32).reshape(3)
            if "pelvis_world_m" in smpl_npz
            else None
        )
        return track_mesh_vertices_to_dict(
            frame_index=int(frame_index),
            timestamp_ns=ts,
            vertices=vertices,
            faces=faces,
            translation_m=trans,
            mesh_schema="smplx_vertices",
            Rh=np.asarray(smpl_npz["Rh"], dtype=np.float32).reshape(3),
            Th=trans,
            poses=poses87,
            smplx_pose_aa_165=pose55.reshape(-1),
            pelvis_world_m=pelvis_world_m,
            betas=betas,
            gender=gender,
            shape_hash=smplx_shape_hash(betas, gender=gender),
            pose_hash=smplx_pose_hash(pose55, trans),
        )

    kp3d = load_keypoints3d_for_publish(moment_dir)
    conf = kp3d[:, 3] > 0.05
    trans = kp3d[8, :3] if kp3d.shape[0] > 8 and bool(conf[8]) else np.nanmean(kp3d[conf, :3], axis=0)
    return track_keypoints3d_to_dict(
        frame_index=int(frame_index),
        timestamp_ns=ts,
        keypoints3d=kp3d,
        schema="body25",
        translation_m=trans,
    )


class SmplxTrackPublisher:
    """Reusable 5598 publisher for live sequence output and replay."""

    def __init__(self, bind: str = DEFAULT_TRACK_PUB_BIND, *, startup_delay_s: float = 0.35) -> None:
        self.bind = str(bind)
        self.startup_delay_s = max(0.0, float(startup_delay_s))
        self._ctx: Any = None
        self._sock: Any = None

    def open(self) -> "SmplxTrackPublisher":
        if self._sock is not None:
            return self
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq required for Genesis SMPL-X publish.") from exc
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 200)
        try:
            self._sock.bind(self.bind)
        except zmq.ZMQError as exc:
            self.close()
            raise OSError(f"ZMQ bind {self.bind!r} failed: {exc}") from exc
        if self.startup_delay_s:
            time.sleep(self.startup_delay_s)
        return self

    def send_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._sock is None:
            self.open()
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self._sock.send_multipart([TOPIC_MULTIVIEW_TRACK_V1.encode("utf-8"), body])
        return payload

    def send_moment(
        self,
        *,
        moment_dir: Path,
        frame_index: int,
        timestamp_ns: int,
        publish_kind: str = "smplx_mesh",
        gender: str = "male",
    ) -> dict[str, Any]:
        payload = build_static_track_payload(
            moment_dir=Path(moment_dir),
            publish_kind=publish_kind,
            frame_index=int(frame_index),
            timestamp_ns=int(timestamp_ns),
            gender=gender,
        )
        return self.send_payload(payload)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = None
        self._ctx = None

    def __enter__(self) -> "SmplxTrackPublisher":
        return self.open()

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


def publish_smplx_sequence(
    sequence_dir: Path | str,
    *,
    bind: str = DEFAULT_TRACK_PUB_BIND,
    publish_kind: str = "smplx_mesh",
    gender: str = "male",
    loop: bool = False,
    time_scale: float = 1.0,
    rate_hz: float = 5.0,
    max_cycles: int | None = None,
) -> dict[str, Any]:
    """Replay fitted moments while preserving each source frame and timestamp."""

    rows = sequence_moments(sequence_dir)
    scale = float(time_scale)
    if scale < 0.0:
        raise ValueError("time_scale must be non-negative")
    cycles = 0
    sent = 0
    publisher = SmplxTrackPublisher(bind)
    try:
        publisher.open()
        while True:
            for index, row in enumerate(rows):
                publisher.send_moment(
                    moment_dir=Path(row["moment_dir"]),
                    frame_index=int(row["frame_index"]),
                    timestamp_ns=int(row["timestamp_ns"]),
                    publish_kind=publish_kind,
                    gender=gender,
                )
                sent += 1
                if index + 1 < len(rows):
                    current_ts = int(row["timestamp_ns"])
                    next_ts = int(rows[index + 1]["timestamp_ns"])
                    if scale > 0.0 and next_ts > current_ts:
                        time.sleep((next_ts - current_ts) * 1.0e-9 / scale)
                    else:
                        time.sleep(1.0 / max(float(rate_hz), 1.0e-6))
            cycles += 1
            if not loop or (max_cycles is not None and cycles >= int(max_cycles)):
                break
    finally:
        publisher.close()
    return {
        "bind": str(bind),
        "topic": TOPIC_MULTIVIEW_TRACK_V1,
        "payload_kind": str(publish_kind),
        "sent": int(sent),
        "cycles": int(cycles),
        "sequence_dir": str(Path(sequence_dir).resolve()),
        "loop": bool(loop),
        "time_scale": float(scale),
    }


def publish_static_smplx_track(
    *,
    moment_dir: Path,
    bind: str = DEFAULT_TRACK_PUB_BIND,
    duration_s: float | None = None,
    rate_hz: float = 5.0,
    publish_kind: str = "smplx_mesh",
    frame_index: int = 0,
    timestamp_ns: int | None = None,
    loop: bool = False,
    gender: str = "male",
) -> dict[str, Any]:
    """Publish a saved fit to Genesis track ZMQ.

    When ``duration_s`` is None and ``loop`` is False, sends a short burst (~2 s)
    so late-joining SUB sockets (twin already running) still receive the mesh.
    When ``loop`` is True, repeats until KeyboardInterrupt.
    """
    try:
        import zmq
    except ImportError as exc:
        raise ImportError("pyzmq required for Genesis static publish.") from exc

    moment_dir = Path(moment_dir)
    payload = build_static_track_payload(
        moment_dir=moment_dir,
        publish_kind=publish_kind,
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        gender=gender,
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 200)
    try:
        sock.bind(str(bind))
    except zmq.ZMQError as exc:
        sock.close(linger=0)
        raise OSError(f"ZMQ bind {bind!r} failed: {exc}") from exc
    topic = TOPIC_MULTIVIEW_TRACK_V1.encode("utf-8")
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    rate = max(float(rate_hz), 1.0e-6)
    dt = 1.0 / rate
    sent = 0

    if duration_s is None and not loop:
        hold_s = 2.0
    elif duration_s is None and loop:
        hold_s = None
    else:
        hold_s = max(0.1, float(duration_s))

    logger.info(
        "static smplx publish bind=%s kind=%s moment=%s loop=%s hold_s=%s",
        bind,
        publish_kind,
        moment_dir,
        loop,
        hold_s,
    )
    time.sleep(0.35)

    stop = False

    def _on_sig(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    if loop:
        signal.signal(signal.SIGINT, _on_sig)
        signal.signal(signal.SIGTERM, _on_sig)

    try:
        if hold_s is None:
            next_log = time.perf_counter() + 2.0
            while not stop:
                sock.send_multipart([topic, body])
                sent += 1
                now = time.perf_counter()
                if now >= next_log:
                    logger.info("static smplx publishing… sent=%d (Ctrl+C to stop)", sent)
                    next_log = now + 2.0
                time.sleep(dt)
        else:
            deadline = time.perf_counter() + hold_s
            while time.perf_counter() < deadline and not stop:
                sock.send_multipart([topic, body])
                sent += 1
                time.sleep(dt)
    finally:
        sock.close(linger=0)

    return {
        "bind": str(bind),
        "topic": TOPIC_MULTIVIEW_TRACK_V1,
        "payload_kind": str(payload.get("payload_kind")),
        "sent": int(sent),
        "duration_s": duration_s,
        "hold_s": hold_s,
        "rate_hz": float(rate_hz),
        "moment_dir": str(moment_dir.resolve()),
        "stopped": bool(stop),
    }
