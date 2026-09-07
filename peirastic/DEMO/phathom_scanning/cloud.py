"""Subscribe the wrist Orbbec cloud and lift it into rail_base."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rm75_control.control.joint_admittance_8dof.param_model.paths import (
    DEFAULT_URDF,
    GENERATED_URDF,
)
from rm75_control.control.joint_admittance_8dof.viewer.orbbec_cloud import (
    DEFAULT_ORBBEC_CLOUD_BIND,
    DEFAULT_ORBBEC_CLOUD_TOPIC,
    RailBaseLink7FK,
    T_world_cam,
    load_T_link7_cam,
    transform_points,
    unpack_cloud_multipart,
)


def resolve_urdf(path: Path | str | None = None) -> Path:
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        raise FileNotFoundError(p)
    for cand in (GENERATED_URDF, DEFAULT_URDF):
        if Path(cand).is_file():
            return Path(cand)
    raise FileNotFoundError("no 8-DOF URDF for rail_base→link_7 FK")


def recv_camera_cloud(
    *,
    subscribe: str = DEFAULT_ORBBEC_CLOUD_BIND,
    topic: str = DEFAULT_ORBBEC_CLOUD_TOPIC,
    timeout_s: float = 3.0,
    frames: int = 3,
) -> tuple[dict, np.ndarray, np.ndarray]:
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(str(subscribe))
    sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    sock.setsockopt(zmq.RCVTIMEO, int(max(timeout_s, 0.2) * 1000.0))
    last = None
    deadline = time.monotonic() + float(timeout_s)
    try:
        while time.monotonic() < deadline:
            try:
                parts = sock.recv_multipart()
            except zmq.Again as exc:
                if last is None:
                    raise TimeoutError(
                        f"no Orbbec cloud on {subscribe} topic={topic}"
                    ) from exc
                break
            last = unpack_cloud_multipart(parts)
            frames -= 1
            if frames <= 0:
                break
    finally:
        sock.close(0)
    if last is None:
        raise TimeoutError(f"no Orbbec cloud on {subscribe} topic={topic}")
    return last


def cloud_in_rail_base(
    xyz_cam: np.ndarray,
    q8: np.ndarray,
    *,
    urdf_path: Path | str | None = None,
    handeye=None,
) -> np.ndarray:
    fk = RailBaseLink7FK(resolve_urdf(urdf_path))
    T_rb_l7 = fk.T_railbase_link7(np.asarray(q8, dtype=np.float64))
    T_l7_cam = load_T_link7_cam(handeye)
    T = T_world_cam(np.eye(4), T_rb_l7, T_l7_cam)
    return transform_points(T, xyz_cam)
