"""Wrist cloud in the controller's rail_base (not the Genesis rail_base).

The slider viewer adds a pedestal to its URDF. Its identically named root is
276 mm below the controller root in the current model. Motion targets must use
the control URDF throughout; no viewer/world translation belongs in this chain.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import DEFAULT_URDF
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
    if Path(DEFAULT_URDF).is_file():
        return Path(DEFAULT_URDF)
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
