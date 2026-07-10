"""Pack / unpack SMPL axis-angle motion for MJCF (free + ball + hinge) Genesis ``q`` vector."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.sim_platform.embodiments.crisp_smpl_euler_retarget import axis_angle_to_rotmat
from projects.genesis_ue_sync.sim_platform.embodiments.mjcf_loader import load_mjcf_dof_layout

_SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

_SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

_PROXY_TO_SMPL_NAME: dict[str, str] = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip",
    "R_Hip": "right_hip",
    "Torso": "spine1",
    "L_Knee": "left_knee",
    "R_Knee": "right_knee",
    "Spine": "spine2",
    "L_Ankle": "left_ankle",
    "R_Ankle": "right_ankle",
    "Chest": "spine3",
    "L_Toe": "left_foot",
    "R_Toe": "right_foot",
    "Neck": "neck",
    "L_Thorax": "left_collar",
    "R_Thorax": "right_collar",
    "Head": "head",
    "L_Shoulder": "left_shoulder",
    "R_Shoulder": "right_shoulder",
    "L_Elbow": "left_elbow",
    "R_Elbow": "right_elbow",
    "L_Wrist": "left_wrist",
    "R_Wrist": "right_wrist",
    "L_Hand": "left_hand",
    "R_Hand": "right_hand",
}


def _smpl_index_for_proxy(proxy_body: str) -> int:
    smpl_name = _PROXY_TO_SMPL_NAME[proxy_body]
    return int(_SMPL_JOINT_NAMES.index(smpl_name))


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rsci

    q_xyzw = np.asarray(Rsci.from_matrix(R).as_quat(), dtype=np.float64).reshape(4)
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    return np.array([w, x, y, z], dtype=np.float32)


def _quat_wxyz_to_rotmat(qwxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rsci

    w, x, y, z = [float(v) for v in np.asarray(qwxyz, dtype=np.float64).reshape(4)]
    q_xyzw = np.array([x, y, z, w], dtype=np.float64)
    return np.asarray(Rsci.from_quat(q_xyzw).as_matrix(), dtype=np.float64)


def _genesis_quat_wxyz_to_dof_euler(q_wxyz: np.ndarray) -> np.ndarray:
    import genesis as gs

    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    return np.asarray(gs.utils.geom.quat_to_xyz(q, rpy=False, degrees=False), dtype=np.float32)


def _genesis_dof_euler_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
    import genesis as gs

    e = np.asarray(euler, dtype=np.float64).reshape(3)
    q = gs.utils.geom.xyz_to_quat(e, rpy=False, degrees=False)
    return np.asarray(q, dtype=np.float32).reshape(4)


def _rot_about_unit_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    a = a / n
    x, y, z = float(a[0]), float(a[1]), float(a[2])
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _hinge_angle_from_R(R: np.ndarray, axis: np.ndarray) -> float:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return 0.0
    a = a / n
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, a))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v = np.cross(a, ref)
    nv = float(np.linalg.norm(v))
    if nv < 1e-9:
        return 0.0
    v = v / nv
    w = R @ v
    s = float(np.dot(a, np.cross(v, w)))
    c = float(np.dot(v, w))
    return float(np.arctan2(s, c))


def fk_smpl24_rot_mats(pose_aa72: np.ndarray) -> np.ndarray:
    """World rotation matrices (24,3,3) from SMPL 72-vector (axis-angle)."""

    p = np.asarray(pose_aa72, dtype=np.float64).reshape(-1)
    if p.size < 72:
        p = np.pad(p, (0, 72 - int(p.size)))
    R = np.zeros((24, 3, 3), dtype=np.float64)
    R[0] = axis_angle_to_rotmat(p[:3])
    for j in range(1, 24):
        lo = 3 + 3 * (j - 1)
        hi = 3 + 3 * j
        Rloc = axis_angle_to_rotmat(p[lo:hi])
        pj = int(_SMPL_PARENTS[j])
        R[j] = R[pj] @ Rloc
    return R


def pack_smpl_pose_to_mjcf_q(
    pose_axis_angle_row: np.ndarray,
    root_translation_world_m: np.ndarray,
    *,
    layout_path: Path | str,
    apply_pelvis_com_offset: bool = True,
) -> np.ndarray:
    """Pack SMPL pose into Genesis ``n_dofs`` vector per ``dof_layout.json`` segments."""

    layout = load_mjcf_dof_layout(Path(layout_path))
    n = int(layout["total_dofs"])
    out = np.zeros((n,), dtype=np.float32)
    Rg = fk_smpl24_rot_mats(pose_axis_angle_row)
    t = np.asarray(root_translation_world_m, dtype=np.float64).reshape(3)
    if apply_pelvis_com_offset:
        pelvis_off = np.asarray(
            layout.get("pelvis_com_offset_body_frame") or [0.0, 0.0, 0.0], dtype=np.float64
        ).reshape(3)
    else:
        pelvis_off = np.zeros((3,), dtype=np.float64)
    t_corr = t + (Rg[0] @ pelvis_off)
    out[0:3] = t_corr.astype(np.float32)
    out[3:6] = _genesis_quat_wxyz_to_dof_euler(_rotmat_to_quat_wxyz(Rg[0]))
    off = 6
    from scipy.spatial.transform import Rotation as Rsci

    for seg in layout["segments"][1:]:
        kind = str(seg["kind"])
        name = str(seg["body"])
        j = _smpl_index_for_proxy(name)
        pj = int(_SMPL_PARENTS[j])
        Rrel = Rg[pj].T @ Rg[j]
        if kind == "ball":
            rv_off = np.asarray(seg.get("frame_offset_rotvec") or [0.0, 0.0, 0.0], dtype=np.float64).reshape(3)
            if float(np.linalg.norm(rv_off)) > 1e-12:
                R_off = np.asarray(Rsci.from_rotvec(rv_off).as_matrix(), dtype=np.float64)
                Rrel = R_off.T @ Rrel @ R_off
            out[off : off + 3] = _genesis_quat_wxyz_to_dof_euler(_rotmat_to_quat_wxyz(Rrel))
            off += 3
        elif kind == "hinge":
            axis = np.asarray(seg["axis_parent_frame"], dtype=np.float64).reshape(3)
            lo, hi = float(seg["range"][0]), float(seg["range"][1])
            th = _hinge_angle_from_R(Rrel, axis)
            th = float(np.clip(th, lo, hi))
            out[off] = np.float32(th)
            off += 1
        else:
            raise ValueError(f"Unknown MJCF layout segment kind: {kind}")
    if off != n:
        raise ValueError(f"MJCF pack size mismatch: wrote {off}, layout expects {n}")
    return out


def smpl_pose_axis_angle_from_mjcf_q(
    q: np.ndarray,
    *,
    layout_path: Path | str,
    apply_pelvis_com_offset: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse: Genesis ``get_dofs_position`` vector → (pose_aa72, root_translation_world_m)."""

    from scipy.spatial.transform import Rotation as Rsci

    layout = load_mjcf_dof_layout(Path(layout_path))
    qh = np.asarray(q, dtype=np.float64).reshape(-1)
    if qh.size != int(layout["total_dofs"]):
        raise ValueError(f"q size {qh.size} != layout total_dofs {layout['total_dofs']}")
    if apply_pelvis_com_offset:
        pelvis_off = np.asarray(
            layout.get("pelvis_com_offset_body_frame") or [0.0, 0.0, 0.0], dtype=np.float64
        ).reshape(3)
    else:
        pelvis_off = np.zeros((3,), dtype=np.float64)
    R = np.zeros((24, 3, 3), dtype=np.float64)
    R[0] = _quat_wxyz_to_rotmat(_genesis_dof_euler_to_quat_wxyz(qh[3:6]))
    trans_corr = qh[0:3].astype(np.float64).reshape(3)
    trans = (trans_corr - (R[0] @ pelvis_off)).astype(np.float32)
    off = 6
    for seg in layout["segments"][1:]:
        kind = str(seg["kind"])
        name = str(seg["body"])
        j = _smpl_index_for_proxy(name)
        pj = int(_SMPL_PARENTS[j])
        if kind == "ball":
            Rloc = _quat_wxyz_to_rotmat(_genesis_dof_euler_to_quat_wxyz(qh[off : off + 3]))
            rv_off = np.asarray(seg.get("frame_offset_rotvec") or [0.0, 0.0, 0.0], dtype=np.float64).reshape(3)
            if float(np.linalg.norm(rv_off)) > 1e-12:
                R_off = np.asarray(Rsci.from_rotvec(rv_off).as_matrix(), dtype=np.float64)
                Rloc = R_off @ Rloc @ R_off.T
            R[j] = R[pj] @ Rloc
            off += 3
        elif kind == "hinge":
            axis = np.asarray(seg["axis_parent_frame"], dtype=np.float64).reshape(3)
            th = float(qh[off])
            Rloc = _rot_about_unit_axis(axis, th)
            R[j] = R[pj] @ Rloc
            off += 1
        else:
            raise ValueError(f"Unknown MJCF layout segment kind: {kind}")
    pose = np.zeros(72, dtype=np.float32)
    pose[:3] = np.asarray(Rsci.from_matrix(R[0]).as_rotvec(), dtype=np.float32).reshape(3)
    for j in range(1, 24):
        pj = int(_SMPL_PARENTS[j])
        Rloc = R[pj].T @ R[j]
        pose[3 + 3 * (j - 1) : 3 + 3 * j] = np.asarray(Rsci.from_matrix(Rloc).as_rotvec(), dtype=np.float32).reshape(3)
    return pose, trans


def capsule_packed_q_from_smpl_mjcf(
    *,
    pose_axis_angle_row: np.ndarray,
    root_translation_world_m: np.ndarray,
    layout_path: Path | str,
    apply_pelvis_com_offset: bool = True,
) -> np.ndarray:
    """Genesis ``set_robot_joint_positions`` vector for MJCF humanoid."""

    return pack_smpl_pose_to_mjcf_q(
        pose_axis_angle_row,
        root_translation_world_m,
        layout_path=layout_path,
        apply_pelvis_com_offset=apply_pelvis_com_offset,
    )