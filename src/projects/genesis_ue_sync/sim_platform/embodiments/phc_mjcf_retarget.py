"""Map SMPL axis-angle into PHC bundled MJCF Genesis ``q`` (FREE: xyz+euler + hinge angles)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rsci

from projects.genesis_ue_sync.sim_platform.embodiments.crisp_smpl_euler_retarget import intrinsic_euler_from_rotmat
from projects.genesis_ue_sync.sim_platform.embodiments.mjcf_loader import load_mjcf_dof_layout
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_mjcf_retarget import (
    _genesis_dof_euler_to_quat_wxyz,
    _genesis_quat_wxyz_to_dof_euler,
    _hinge_angle_from_R,
    _quat_wxyz_to_rotmat,
    _rot_about_unit_axis,
    _rotmat_to_quat_wxyz,
    _smpl_index_for_proxy,
    fk_smpl24_rot_mats,
)

_SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)


def _phc_upright_global_post_rotation_matrix() -> np.ndarray:
    """Same quaternion post-multiply as PHC ``convert_amass_isaac.py`` (``upright_start``).

    Raw AMASS/SMPL world rotations ``Rg`` are per-joint right-multiplied by ``fix^{-1}`` with
    ``fix = scipy.spatial.transform.Rotation.from_quat([x,y,z,w])``, default ``(0.5,0.5,0.5,0.5)``,
    matching bundled ``phc/data/assets/mjcf/smpl_*_humanoid.xml`` training convention.

    Disable with ``AMONGUS_PHC_UPRIGHT_FIX=0``. Override quaternion (xyzw, comma-separated) with
    ``AMONGUS_PHC_UPRIGHT_QUAT``.
    """

    if os.environ.get("AMONGUS_PHC_UPRIGHT_FIX", "1").strip().lower() in ("0", "false", "no", "off"):
        return np.eye(3, dtype=np.float64)
    raw = os.environ.get("AMONGUS_PHC_UPRIGHT_QUAT", "0.5,0.5,0.5,0.5").strip()
    parts = [float(x) for x in raw.split(",")]
    if len(parts) != 4:
        parts = [0.5, 0.5, 0.5, 0.5]
    q_fix = Rsci.from_quat(np.asarray(parts, dtype=np.float64))
    return q_fix.inv().as_matrix()


def pack_smpl_pose_to_phc_bundled_mjcf_q(
    pose_axis_angle_row: np.ndarray,
    root_translation_world_m: np.ndarray,
    *,
    layout_path: Path | str,
) -> np.ndarray:
    """Pack SMPL 72-vector into flat Genesis ``dofs`` vector for ``phc_bundled_mjcf`` layout JSON."""

    layout = load_mjcf_dof_layout(Path(layout_path))
    if str(layout.get("mjcf_layout_tag")) != "phc_bundled_mjcf":
        raise ValueError(f"Expected phc_bundled_mjcf layout, got {layout.get('mjcf_layout_tag')!r}")
    segs: list[dict] = list(layout["segments"])
    n = int(layout["total_dofs"])
    out = np.zeros((n,), dtype=np.float32)
    Rg = fk_smpl24_rot_mats(pose_axis_angle_row)
    R_post = _phc_upright_global_post_rotation_matrix()
    if not np.allclose(R_post, np.eye(3)):
        Rg = np.einsum("nij,jk->nik", Rg, R_post)
    t = np.asarray(root_translation_world_m, dtype=np.float64).reshape(3)
    off = 0
    si = 0
    while si < len(segs):
        seg = segs[si]
        kind = str(seg.get("kind"))
        if kind == "free_mujoco":
            n_free = int(seg["n"])
            if n_free != 6:
                raise ValueError(
                    f"free_mujoco expects n=6 (Genesis MJCF FREE joint dofs, not MuJoCo qpos). Got {n_free}. "
                    "Regenerate layout JSON (re-run sync / prepare_proxy, or delete *_dof_layout.json)."
                )
            out[off : off + 3] = t.astype(np.float32)
            qwxyz = _rotmat_to_quat_wxyz(Rg[0])
            out[off + 3 : off + 6] = _genesis_quat_wxyz_to_dof_euler(qwxyz)
            off += 6
            si += 1
            continue
        if kind != "hinge":
            raise ValueError(f"Unexpected PHC segment {kind!r} at index {si}")
        body = str(seg["body"])
        hinge_group: list[dict] = []
        while si < len(segs) and str(segs[si].get("kind")) == "hinge" and str(segs[si].get("body")) == body:
            hinge_group.append(segs[si])
            si += 1
        jidx = _smpl_index_for_proxy(body)
        pj = int(_SMPL_PARENTS[jidx])
        Rrel = Rg[pj].T @ Rg[jidx]
        if len(hinge_group) == 3:
            # PHC lists hinges x, y, z on one body: intrinsic XYZ (SciPy capital seq). SciPy 1.15+:
            # lowercase "xyz" = extrinsic — wrong here.
            euler = intrinsic_euler_from_rotmat(Rrel, euler_seq="XYZ")
            for hi, ang in enumerate(euler):
                bounds = hinge_group[hi].get("range_rad")
                lo, hi_b = (-np.pi, np.pi) if bounds is None else (float(bounds[0]), float(bounds[1]))
                out[off] = np.float32(np.clip(float(ang), lo, hi_b))
                off += 1
        else:
            for hg in hinge_group:
                axis = np.asarray(hg.get("axis_world_hint") or [1.0, 0.0, 0.0], dtype=np.float64).reshape(3)
                bounds = hg.get("range_rad")
                lo, hi_b = (-np.pi, np.pi) if bounds is None else (float(bounds[0]), float(bounds[1]))
                th = _hinge_angle_from_R(Rrel, axis)
                out[off] = np.float32(np.clip(th, lo, hi_b))
                off += 1
    if off != n:
        raise ValueError(f"PHC MJCF pack mismatch: wrote {off}, layout expects {n}")
    return out


def capsule_packed_q_from_smpl_phc_mjcf(
    *,
    pose_axis_angle_row: np.ndarray,
    root_translation_world_m: np.ndarray,
    layout_path: Path | str,
) -> np.ndarray:
    return pack_smpl_pose_to_phc_bundled_mjcf_q(
        pose_axis_angle_row,
        root_translation_world_m,
        layout_path=layout_path,
    )


def smpl_pose_row_from_phc_bundled_q(
    *,
    pose_ref: np.ndarray,
    q_ref: np.ndarray,
    q_opt: np.ndarray,
    layout_path: Path | str,
) -> np.ndarray:
    """Invert PHC bundled MJCF ``q`` to a full SMPL pose row (same width as ``pose_ref``)."""

    import json

    pose = np.zeros_like(np.asarray(pose_ref, dtype=np.float32))
    q0 = np.asarray(q_ref, dtype=np.float32).reshape(-1)
    q1 = np.asarray(q_opt, dtype=np.float32).reshape(-1)
    if q0.shape != q1.shape:
        return pose
    layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    segs: list[dict] = list(layout.get("segments", []))
    if str(layout.get("mjcf_layout_tag")) != "phc_bundled_mjcf":
        return pose
    R_phc = np.tile(np.eye(3, dtype=np.float64).reshape(1, 3, 3), (24, 1, 1))
    off = 0
    si = 0
    while si < len(segs):
        seg = segs[si]
        kind = str(seg.get("kind", ""))
        if kind == "free_mujoco":
            if int(seg.get("n", 6)) == 6 and q1.shape[0] >= off + 6:
                R_phc[0] = _quat_wxyz_to_rotmat(_genesis_dof_euler_to_quat_wxyz(q1[off + 3 : off + 6]))
            off += int(seg.get("n", 6))
            si += 1
            continue
        if kind != "hinge":
            off += int(seg.get("n", 1))
            si += 1
            continue
        body = str(seg.get("body", ""))
        group: list[dict] = []
        start_off = off
        while si < len(segs) and str(segs[si].get("kind", "")) == "hinge" and str(segs[si].get("body", "")) == body:
            group.append(segs[si])
            si += 1
            off += 1
        try:
            jidx = int(_smpl_index_for_proxy(body))
        except Exception:
            continue
        p0 = 3 * jidx
        p1 = p0 + 3
        if p1 > pose.shape[0] or start_off + len(group) > q0.shape[0]:
            continue
        try:
            pj = int(_SMPL_PARENTS[jidx])
            if len(group) == 3:
                Rrel = np.asarray(
                    Rsci.from_euler(
                        "XYZ",
                        q1[start_off : start_off + 3].astype(np.float64),
                        degrees=False,
                    ).as_matrix(),
                    dtype=np.float64,
                )
            else:
                Rrel = np.eye(3, dtype=np.float64)
                for gi, hg in enumerate(group):
                    axis = np.asarray(hg.get("axis_world_hint") or [1.0, 0.0, 0.0], dtype=np.float64).reshape(3)
                    Rrel = Rrel @ _rot_about_unit_axis(axis, float(q1[start_off + gi]))
            R_phc[jidx] = R_phc[pj] @ Rrel
        except Exception:
            continue
    R_post = _phc_upright_global_post_rotation_matrix()
    R_smpl = np.einsum("nij,jk->nik", R_phc, R_post.T)
    try:
        pose[:3] = np.asarray(Rsci.from_matrix(R_smpl[0]).as_rotvec(), dtype=np.float32).reshape(3)
        for j in range(1, 24):
            pj = int(_SMPL_PARENTS[j])
            Rloc = R_smpl[pj].T @ R_smpl[j]
            pose[3 + 3 * (j - 1) : 3 + 3 * j] = np.asarray(
                Rsci.from_matrix(Rloc).as_rotvec(),
                dtype=np.float32,
            ).reshape(3)
    except Exception:
        return np.asarray(pose_ref, dtype=np.float32).copy()
    return pose
