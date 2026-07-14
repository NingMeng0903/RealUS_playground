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
        run_path = smplx_output_root() / run_path
    moment = run_path / "moment_0000"
    if not (moment / "smplx_result.npz").is_file():
        raise FileNotFoundError(f"smplx_result.npz not found under {moment}")
    return moment


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
        )

    if str(publish_kind) == "smplx_mesh":
        smpl_npz = np.load(moment_dir / "smplx_result.npz")
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
    sock.bind(str(bind))
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
            while not stop:
                sock.send_multipart([topic, body])
                sent += 1
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
    }
