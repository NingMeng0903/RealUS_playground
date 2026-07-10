"""Pose vector adapters for anatomy assets driven by SMPL/SMPL-X streams."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


SMPLX_RUNTIME_JOINT_COUNT = 55


def easymocap_fit_to_smplx55(Rh: Any, poses: Any) -> np.ndarray:
    """Map EasyMocap mv1p params (Rh + poses 87D) to [55, 3] SMPL-X axis-angle."""
    root = np.asarray(Rh, dtype=np.float32).reshape(3)
    flat = np.asarray(poses, dtype=np.float32).reshape(-1)
    out = np.zeros((SMPLX_RUNTIME_JOINT_COUNT, 3), dtype=np.float32)
    out[0] = root
    if flat.size >= 66:
        body22 = flat[:66].reshape(22, 3)
        out[1:22] = body22[1:22]
    return out


def axis_angle_to_rotation(axis_angle: Any) -> np.ndarray:
    """Single axis-angle vector [3] -> rotation matrix [3, 3]."""
    aa = np.asarray(axis_angle, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(aa))
    if angle < 1.0e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z = (aa / angle).tolist()
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def easymocap_drive_translation(Rh: Any, Th: Any, pelvis: Any) -> np.ndarray:
    """Convert EasyMocap (Rh, Th) into the translation expected by anatomy LBS.

    EasyMocap applies Rh about the canonical-frame origin (verts_world = R @ v + Th),
    while the anatomy LBS rotates the root about the canonical pelvis joint.
    Compensation: Th_eff = Th + R @ pelvis - pelvis.
    """
    R = axis_angle_to_rotation(Rh)
    p = np.asarray(pelvis, dtype=np.float32).reshape(3)
    t = np.asarray(Th, dtype=np.float32).reshape(3)
    return (t + R @ p - p).astype(np.float32)


def load_easymocap_smplx_fit_drive(
    npz_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load static UE/terminal-8 SMPL-X fit params for anatomy drive (pose55 flat, Th).

    The returned Th is the raw EasyMocap translation (plus root_align_offset when
    available); apply ``easymocap_drive_translation`` with the asset pelvis before
    feeding it into anatomy LBS.
    """
    data = np.load(Path(npz_path))
    Rh = np.asarray(data["Rh"], dtype=np.float32).reshape(3)
    poses = np.asarray(data["poses"], dtype=np.float32).reshape(-1)
    Th = np.asarray(data["Th"], dtype=np.float32).reshape(3)
    if "root_align_offset" in data.files:
        Th = Th + np.asarray(data["root_align_offset"], dtype=np.float32).reshape(3)
    pose55 = easymocap_fit_to_smplx55(Rh, poses).reshape(-1)
    return pose55, Th


def pose_to_smplx55_axis_angle(pose: Any) -> np.ndarray:
    """Return a [55, 3] SMPL-X runtime pose from common axis-angle layouts.

    Supported inputs:
    - 72D SMPL axis-angle: copy root + first 21 body joints, ignore SMPL hand end joints.
    - 87D EasyMocap SMPL-X: copy the first 66 body values; hand PCA/face terms are ignored.
    - 165D SMPL-X full axis-angle: reshape directly to 55 joints.
    - [J, 3] arrays: copy up to 55 joints.
    """
    arr = np.asarray(pose, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 3:
        out = np.zeros((SMPLX_RUNTIME_JOINT_COUNT, 3), dtype=np.float32)
        n = min(SMPLX_RUNTIME_JOINT_COUNT, int(arr.shape[0]))
        out[:n] = arr[:n]
        return out
    flat = arr.reshape(-1)
    out = np.zeros((SMPLX_RUNTIME_JOINT_COUNT, 3), dtype=np.float32)
    if flat.size == 72:
        smpl = flat.reshape(24, 3)
        out[:22] = smpl[:22]
        return out
    if flat.size == 87:
        out[:22] = flat[:66].reshape(22, 3)
        return out
    if flat.size == 165:
        return flat.reshape(SMPLX_RUNTIME_JOINT_COUNT, 3).astype(np.float32)
    if flat.size % 3 == 0:
        rows = flat.reshape(-1, 3)
        n = min(SMPLX_RUNTIME_JOINT_COUNT, int(rows.shape[0]))
        out[:n] = rows[:n]
        return out
    raise ValueError(f"Unsupported pose shape for SMPL-X anatomy drive: {arr.shape}")

