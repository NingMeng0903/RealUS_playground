"""Unmodified held-out motion frames and explicit synthetic SMPL-X shapes.

AMASS supplies rotations only here: its SMPL-H shape coefficients are never
transferred to SMPL-X.  The selected frames were excluded from the V12 fit,
which used the two local captures and synthetic deep-flex poses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .pose_adapter import smplh156_to_smplx55


DEFAULT_AMASS_ROOT = Path(
    "/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/amass_hf/raw"
)

# Fixed source frames, chosen for distinct motion families without support
# filtering, joint-angle clipping, root removal, or per-candidate selection.
HELD_OUT_POSE_SPECS_V13 = (
    (
        "heldout_jumping_jacks",
        "DFaust_67/50002/50002_jumping_jacks_poses.npz",
        102,
    ),
    (
        "heldout_punching",
        "DFaust_67/50002/50002_punching_poses.npz",
        213,
    ),
    (
        "heldout_kicking",
        "BioMotionLab_NTroje/rub086/0023_kicking1_poses.npz",
        327,
    ),
    (
        "heldout_sitting",
        "BioMotionLab_NTroje/rub086/0014_sitting1_poses.npz",
        262,
    ),
    ("heldout_walking", "HumanEva/S1/Walking_3_poses.npz", 1530),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pose_v13(pose: Any, *, label: str = "pose") -> np.ndarray:
    """Require exactly 55 finite axis-angle rotations; never repair a pose."""
    array = np.asarray(pose, dtype=np.float64)
    if array.shape != (55, 3):
        raise ValueError(f"{label} must have shape (55, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite rotations")
    result = array.astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} rotations exceed float32 range")
    return result


def load_held_out_poses_v13(
    amass_root: Path | str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Return fixed SMPL-X poses plus source path, SHA-256 and frame provenance.

    The SMPL-H-to-SMPL-X adapter inserts three zero facial rotations.  Every
    source body and hand rotation, including global root orientation, is retained
    to float32 precision.  Source translation is recorded separately so a caller
    can apply it equally to anatomy and skin when needed.
    """
    root = Path(amass_root) if amass_root is not None else DEFAULT_AMASS_ROOT
    root = root.expanduser().resolve()
    poses: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for label, relative_path, frame_index in HELD_OUT_POSE_SPECS_V13:
        path = root / relative_path
        with np.load(path, allow_pickle=False) as data:
            raw = np.asarray(data["poses"])
            if raw.ndim != 2 or raw.shape[1] != 156:
                raise ValueError(f"{path}: expected AMASS [frames, 156] poses")
            if not 0 <= frame_index < len(raw):
                raise ValueError(f"{path}: frame {frame_index} is unavailable")
            source_pose = np.asarray(raw[frame_index], dtype=np.float64)
            if not np.all(np.isfinite(source_pose)):
                raise ValueError(f"{path}: frame {frame_index} is non-finite")
            frame_count = len(raw)
            frame_rate = float(np.asarray(data["mocap_framerate"]).item())
            if not np.isfinite(frame_rate) or frame_rate <= 0:
                raise ValueError(f"{path}: invalid mocap frame rate")
            translation = np.asarray(data["trans"][frame_index], dtype=np.float64)
            if translation.shape != (3,) or not np.all(np.isfinite(translation)):
                raise ValueError(f"{path}: invalid frame translation")
            source_gender = str(np.asarray(data["gender"]).item())

        pose = validate_pose_v13(smplh156_to_smplx55(source_pose), label=label)
        # Assert the actual mapping, rather than merely declaring that poses
        # were not clipped: reinserting the two hand blocks must recover input.
        recovered = np.concatenate((pose[:22], pose[25:40], pose[40:55]))
        source_float32 = source_pose.astype(np.float32).reshape(52, 3)
        if not np.array_equal(recovered, source_float32):
            raise ValueError(f"{label}: adapter changed a source rotation")
        poses[label] = pose
        provenance[label] = {
            "source_dataset": "AMASS",
            "source_path": str(path),
            "source_sha256": _sha256(path),
            "source_pose_layout": "smplh156",
            "frame_index": frame_index,
            "frame_count": frame_count,
            "frame_rate_hz": frame_rate,
            "time_seconds": frame_index / frame_rate,
            "source_gender": source_gender,
            "source_translation_m": translation.tolist(),
            "source_shape_used": False,
            "root_rotation_preserved": True,
            "pose_clipped": False,
            "face_joints_zero_padded": [22, 23, 24],
            "pose_sha256_float32": hashlib.sha256(pose.tobytes()).hexdigest(),
        }
    return poses, provenance


def synthetic_betas_v13(amplitude: float = 1.5) -> dict[str, np.ndarray]:
    """SMPL-X mean and ±PC1/±PC2 stress shapes, not captured human identities.

These are coefficients for the caller's actual SMPL-X model.  No AMASS or
BEDLAM shape transfer is performed.  PC signs describe coefficient signs,
not assumed height or weight labels; measure the generated geometry instead.
    """
    value = float(amplitude)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("synthetic beta amplitude must be finite and positive")
    if value > np.finfo(np.float32).max:
        raise ValueError("synthetic beta amplitude exceeds float32 range")
    result = {"synthetic_mean": np.zeros(10, dtype=np.float32)}
    for component in (0, 1):
        for sign, suffix in ((-1.0, "negative"), (1.0, "positive")):
            beta = np.zeros(10, dtype=np.float32)
            beta[component] = sign * value
            result[f"synthetic_pc{component + 1}_{suffix}"] = beta
    return result


__all__ = [
    "DEFAULT_AMASS_ROOT",
    "HELD_OUT_POSE_SPECS_V13",
    "load_held_out_poses_v13",
    "synthetic_betas_v13",
    "validate_pose_v13",
]
