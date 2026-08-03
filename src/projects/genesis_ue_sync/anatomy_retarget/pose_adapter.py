"""Pose vector adapters for anatomy assets driven by SMPL/SMPL-X streams."""

from __future__ import annotations

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


SMPLX_RUNTIME_JOINT_COUNT = 55


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_smplx_model_path(gender: str) -> Path:
    requested = str(gender).strip().upper() or "MALE"
    roots = [
        _repo_root() / "ref_code_library" / "EasyMocap" / "data" / "smplx" / "smplx",
        _repo_root() / "ref_code_library" / "InteractVLM" / "data" / "body_models" / "smplx",
    ]
    for root in roots:
        candidate = root / f"SMPLX_{requested}.pkl"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"SMPL-X {requested} model is required to decode EasyMocap hand PCA")


@lru_cache(maxsize=8)
def _hand_pca_components(model_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the exact six-component hand bases used by EasyMocap SMPL-X."""
    with Path(model_path).open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    left = np.asarray(payload["hands_componentsl"], dtype=np.float32)[:6, :45]
    right = np.asarray(payload["hands_componentsr"], dtype=np.float32)[:6, :45]
    if left.shape != (6, 45) or right.shape != (6, 45):
        raise ValueError(f"Invalid SMPL-X hand PCA bases in {model_path}: {left.shape}, {right.shape}")
    return left, right


def easymocap_fit_to_smplx55(
    Rh: Any,
    poses: Any,
    *,
    gender: str = "male",
    model_path: str | Path | None = None,
    closed_mouth: bool = True,
) -> np.ndarray:
    """Map EasyMocap ``Rh + poses`` to the full [55, 3] SMPL-X pose.

    EasyMocap stores SMPL-X as body66 + left/right hand PCA6 + head9.  The
    official SMPL-X runtime order is body22 + head3 + left/right hand15.
    ``use_flat_mean=True`` is hard-coded by EasyMocap, so no MANO mean is added.
    """
    root = np.asarray(Rh, dtype=np.float32).reshape(3)
    flat = np.asarray(poses, dtype=np.float32).reshape(-1)
    out = np.zeros((SMPLX_RUNTIME_JOINT_COUNT, 3), dtype=np.float32)
    out[0] = root
    if flat.size == 165:
        full = flat.reshape(SMPLX_RUNTIME_JOINT_COUNT, 3).copy()
        full[0] = root
        if closed_mouth:
            full[22] = 0.0
        return full.astype(np.float32)
    if flat.size != 87:
        raise ValueError(f"Expected EasyMocap SMPL-X 87D or full 165D pose, got {flat.size}")
    body22 = flat[:66].reshape(22, 3)
    out[1:22] = body22[1:22]
    out[22:25] = flat[78:87].reshape(3, 3)
    if closed_mouth:
        # SMPL-X joint 22 is jaw.  V7 retains the authored tongue and bakes a
        # closed-mouth oral compound, so capture jaw motion is intentionally
        # suppressed unless a caller opts into an open-mouth asset.
        out[22] = 0.0
    resolved = Path(model_path).expanduser().resolve() if model_path is not None else _default_smplx_model_path(gender)
    left_basis, right_basis = _hand_pca_components(str(resolved))
    out[25:40] = (flat[66:72].reshape(1, 6) @ left_basis).reshape(15, 3)
    out[40:55] = (flat[72:78].reshape(1, 6) @ right_basis).reshape(15, 3)
    return out


def smplx_shape_hash(betas: Any, *, gender: str = "male") -> str:
    beta = np.asarray(betas, dtype=np.float32).reshape(-1)[:10]
    digest = hashlib.sha256(str(gender).lower().encode("utf-8") + beta.tobytes()).hexdigest()
    return digest[:20]


def smplx_pose_hash(pose55: Any, transl: Any | None = None) -> str:
    pose = np.asarray(pose55, dtype=np.float32).reshape(55, 3)
    payload = pose.tobytes()
    if transl is not None:
        payload += np.asarray(transl, dtype=np.float32).reshape(3).tobytes()
    return hashlib.sha256(payload).hexdigest()[:20]


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


def anatomy_transl_from_track_drive(
    pose55_flat: Any,
    Th: Any,
    pelvis: Any | None,
) -> np.ndarray:
    """Pelvis-compensated translation for anatomy LBS from a flat pose55 + raw Th."""
    th = np.asarray(Th, dtype=np.float32).reshape(3)
    if pelvis is None:
        return th
    rh = np.asarray(pose55_flat, dtype=np.float32).reshape(-1)[:3]
    return easymocap_drive_translation(rh, th, pelvis)


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
    *,
    gender: str = "male",
    model_path: str | Path | None = None,
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
    pose55 = easymocap_fit_to_smplx55(
        Rh, poses, gender=gender, model_path=model_path
    ).reshape(-1)
    return pose55, Th


def smplh156_to_smplx55(pose156: Any) -> np.ndarray:
    """Map SMPL-H axis-angle (52*3=156) to SMPL-X runtime [55, 3].

    SMPL-H layout is body22 + left_hand15 + right_hand15.  SMPL-X inserts three
    face joints (jaw/leye/reye) after the body block; those stay zero here.
    """
    flat = np.asarray(pose156, dtype=np.float32).reshape(-1)
    if flat.size != 156:
        raise ValueError(f"Expected SMPL-H 156D pose, got {flat.size}")
    joints = flat.reshape(52, 3)
    out = np.zeros((SMPLX_RUNTIME_JOINT_COUNT, 3), dtype=np.float32)
    out[:22] = joints[:22]
    out[25:40] = joints[22:37]
    out[40:55] = joints[37:52]
    out[22] = 0.0
    return out


def pose_to_smplx55_axis_angle(pose: Any) -> np.ndarray:
    """Return a [55, 3] SMPL-X runtime pose from common axis-angle layouts.

    Supported inputs:
    - 72D SMPL axis-angle: copy root + first 21 body joints, ignore SMPL hand end joints.
    - 87D EasyMocap SMPL-X: decode body, face and both six-component hand PCA vectors.
    - 156D SMPL-H full axis-angle: body22 + hands30, face joints zero-padded.
    - 165D SMPL-X full axis-angle: reshape directly to 55 joints.
    - [J, 3] arrays: copy up to 55 joints.
    """
    arr = np.asarray(pose, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 3:
        out = np.zeros((SMPLX_RUNTIME_JOINT_COUNT, 3), dtype=np.float32)
        n = min(SMPLX_RUNTIME_JOINT_COUNT, int(arr.shape[0]))
        out[:n] = arr[:n]
        out[22] = 0.0
        return out
    flat = arr.reshape(-1)
    out = np.zeros((SMPLX_RUNTIME_JOINT_COUNT, 3), dtype=np.float32)
    if flat.size == 72:
        smpl = flat.reshape(24, 3)
        out[:22] = smpl[:22]
        out[22] = 0.0
        return out
    if flat.size == 87:
        # This generic adapter has no separate EasyMocap ``Rh`` argument. Callers
        # that have ``Rh`` should use ``easymocap_fit_to_smplx55`` directly.
        return easymocap_fit_to_smplx55(flat[:3], flat)
    if flat.size == 156:
        return smplh156_to_smplx55(flat)
    if flat.size == 165:
        out = flat.reshape(SMPLX_RUNTIME_JOINT_COUNT, 3).astype(np.float32)
        out[22] = 0.0
        return out
    if flat.size % 3 == 0:
        rows = flat.reshape(-1, 3)
        n = min(SMPLX_RUNTIME_JOINT_COUNT, int(rows.shape[0]))
        out[:n] = rows[:n]
        out[22] = 0.0
        return out
    raise ValueError(f"Unsupported pose shape for SMPL-X anatomy drive: {arr.shape}")
