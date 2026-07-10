"""SMPL axis-angle (CRISP body order) → CRISP MuJoco DFS capsule URDF Euler targets.

Mirrors the retarget slice of archived ``crisp_real2sim_bridge`` without training imports.
Joint layout: 23 bodies × 3 (intrinsic Euler) = 69 scalars; root translation + orient are separate DOFs.

**Euler axis order:** SciPy intrinsic convention: ``seq`` uses capital letters (e.g. ``XYZ``, ``ZYX``).
Default ``XYZ``. Override with env ``AMONGUS_CAPSULE_EULER_SEQ`` (e.g. ``ZYX``) if your Genesis/MuJoCo
stack matches a different decomposition than the URDF ``ex/ey/ez`` chain (trial for visual alignment).
"""

from __future__ import annotations

import os

import numpy as np

SMPL_CRISP_BODY_NAMES: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
)

SMPL_MUJOCO_KINEMATIC_ORDER: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
)


def smpl_mujoco_permutation_from_crisp() -> np.ndarray:
    name_to_smpl = {n: i for i, n in enumerate(SMPL_CRISP_BODY_NAMES)}
    out: list[int] = []
    for name in SMPL_MUJOCO_KINEMATIC_ORDER:
        if name not in name_to_smpl:
            raise KeyError(f"Unknown joint {name} for CRISP SMPL layout.")
        out.append(int(name_to_smpl[name]))
    return np.asarray(out, dtype=np.int64)


def axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = aa / theta
    x, y, z = float(k[0]), float(k[1]), float(k[2])
    c, s = np.cos(theta), np.sin(theta)
    c1 = 1.0 - c
    return np.array(
        [
            [x * x * c1 + c, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, y * y * c1 + c, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, z * z * c1 + c],
        ],
        dtype=np.float64,
    )


def capsule_intrinsic_euler_seq() -> str:
    """3-letter intrinsic SciPy euler sequence (capital X/Y/Z only), default ``XYZ``."""

    raw = os.environ.get("AMONGUS_CAPSULE_EULER_SEQ", "XYZ").strip().upper()
    if len(raw) != 3 or any(c not in "XYZ" for c in raw):
        return "XYZ"
    return raw


def _intrinsic_xyz_euler_manual_fallback(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sb = float(R[0, 2])
    c00 = float(R[0, 0])
    c01 = float(R[0, 1])
    cb = float(np.sqrt(max(0.0, c00 * c00 + c01 * c01)))
    eps = 1e-7
    if cb > eps:
        ex = float(np.arctan2(-R[1, 2], R[2, 2]))
        ey = float(np.arctan2(sb, cb))
        ez = float(np.arctan2(-R[0, 1], R[0, 0]))
        return np.array([ex, ey, ez], dtype=np.float32)
    ey = float(np.arctan2(sb, cb))
    ez = 0.0
    if sb > 0.0:
        ex = float(np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1]))
    else:
        ex = float(np.arctan2(R[0, 1] - R[1, 0], R[0, 0] + R[1, 1]))
    return np.array([ex, ey, ez], dtype=np.float32)


def intrinsic_euler_from_rotmat(R: np.ndarray, *, euler_seq: str | None = None) -> np.ndarray:
    """Intrinsic Euler angles (radians) from rotation matrix; ``euler_seq`` defaults to env / ``XYZ``."""

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    seq = capsule_intrinsic_euler_seq() if euler_seq is None else str(euler_seq).strip().upper()
    if len(seq) != 3 or any(c not in "XYZ" for c in seq):
        seq = "XYZ"
    try:
        from scipy.spatial.transform import Rotation as Rsci

        return np.asarray(Rsci.from_matrix(R).as_euler(seq), dtype=np.float32)
    except Exception:
        if seq != "XYZ":
            return _intrinsic_xyz_euler_manual_fallback(R)
        return _intrinsic_xyz_euler_manual_fallback(R)


def intrinsic_xyz_euler_from_rotmat(R: np.ndarray) -> np.ndarray:
    """Deprecated name: uses ``AMONGUS_CAPSULE_EULER_SEQ`` / default ``XYZ``."""

    return intrinsic_euler_from_rotmat(R)


def retarget_smpl_aa_to_crisp_mujoco_euler(
    pose_params: np.ndarray,
    *,
    joint_count: int,
    perm: np.ndarray | None = None,
) -> np.ndarray:
    pose = np.asarray(pose_params, dtype=np.float64).reshape(-1)
    if perm is None:
        perm = smpl_mujoco_permutation_from_crisp()
    parts: list[np.ndarray] = []
    for mi in range(1, 24):
        sj = int(perm[mi])
        start = 3 * sj
        if start + 3 <= pose.size:
            aa = pose[start : start + 3]
        else:
            aa = np.zeros(3, dtype=np.float64)
        euler = intrinsic_euler_from_rotmat(axis_angle_to_rotmat(aa))
        parts.append(np.asarray(euler, dtype=np.float32))
    out = np.concatenate(parts, axis=0).astype(np.float32)
    jc = max(int(joint_count), 0)
    if jc <= 0:
        return out
    if out.size < jc:
        return np.pad(out, (0, jc - out.size)).astype(np.float32)
    return out[:jc].astype(np.float32)


def pack_floating_capsule_dof(
    pose_aa: np.ndarray,
    trans_world: np.ndarray,
    *,
    body_euler_count: int = 69,
) -> np.ndarray:
    """6-DOF floating base (xyz + intrinsic euler of root axis-angle) + body Euler joints."""
    pose_aa = np.asarray(pose_aa, dtype=np.float64).reshape(-1)
    trans_world = np.asarray(trans_world, dtype=np.float32).reshape(3)
    body = retarget_smpl_aa_to_crisp_mujoco_euler(pose_aa.astype(np.float32), joint_count=body_euler_count)
    base = 6
    full = np.zeros((base + body.size,), dtype=np.float32)
    full[0:3] = trans_world
    glo = pose_aa[:3]
    full[3:6] = intrinsic_euler_from_rotmat(axis_angle_to_rotmat(glo))
    full[6 : 6 + body.size] = body
    return full
