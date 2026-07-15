"""Linear blend skinning for retargeted anatomy assets."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

from .pose_adapter import pose_to_smplx55_axis_angle
from .rigged_asset import AnatomyRiggedAsset

_DEBUG_LOG = "/media/camp/EXT_DRIVE/RealUS_playground/.cursor/debug-10238d.log"


def _agent_log(*, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    # #region agent log
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "sessionId": "10238d",
                        "runId": "post-fix",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    },
                    default=str,
                )
                + "\n"
            )
    except OSError:
        pass
    # #endregion


_CUDA_ASSET_CACHE: dict[int, tuple[Any, Any, Any]] = {}


def _dense_asset_weights(asset: AnatomyRiggedAsset) -> np.ndarray:
    if asset.lbs_weights is not None:
        return np.asarray(asset.lbs_weights, dtype=np.float32)
    if asset.driver_indices is None or asset.driver_weights is None or asset.source_bone_names is None:
        raise ValueError("asset has no usable skinning weights")
    dense = np.zeros((asset.vertices_rest.shape[0], len(asset.source_bone_names)), dtype=np.float32)
    rows = np.arange(dense.shape[0])
    for slot in range(asset.driver_indices.shape[1]):
        np.add.at(dense, (rows, asset.driver_indices[:, slot]), asset.driver_weights[:, slot])
    return dense


def _cuda_requested() -> bool:
    value = str(os.environ.get("AMONGUS_ANATOMY_LBS_DEVICE", "auto")).strip().lower()
    if value in {"cpu", "off", "false", "0"}:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _skin_vertices_cuda(
    asset: AnatomyRiggedAsset,
    transforms: np.ndarray,
    transl: Any | None,
) -> np.ndarray:
    import torch

    key = id(asset)
    cached = _CUDA_ASSET_CACHE.get(key)
    if cached is None:
        if asset.driver_indices is None or asset.driver_weights is None:
            from .rigged_asset import sparse_driver_weights

            indices, weights = sparse_driver_weights(asset.lbs_weights)
        else:
            indices, weights = asset.driver_indices, asset.driver_weights
        vertices_t = torch.as_tensor(asset.vertices_rest, dtype=torch.float32, device="cuda")
        indices_t = torch.as_tensor(indices, dtype=torch.long, device="cuda")
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device="cuda")
        cached = (vertices_t, indices_t, weights_t)
        _CUDA_ASSET_CACHE[key] = cached
    vertices_t, indices_t, weights_t = cached
    tf = torch.as_tensor(transforms, dtype=torch.float32, device="cuda")
    selected = tf[indices_t]
    blended = torch.sum(selected * weights_t[..., None, None], dim=1)
    ones = torch.ones((vertices_t.shape[0], 1), dtype=torch.float32, device="cuda")
    homo = torch.cat((vertices_t, ones), dim=1)
    posed = torch.bmm(blended, homo.unsqueeze(-1))[:, :3, 0]
    if transl is not None:
        posed = posed + torch.as_tensor(transl, dtype=torch.float32, device="cuda").reshape(1, 3)
    return posed.detach().cpu().numpy().astype(np.float32, copy=False)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    rows = np.asarray(axis_angle, dtype=np.float32).reshape(-1, 3)
    out = np.tile(np.eye(3, dtype=np.float32), (rows.shape[0], 1, 1))
    angles = np.linalg.norm(rows, axis=1)
    for idx, angle in enumerate(angles.tolist()):
        if float(angle) < 1.0e-8:
            continue
        axis = rows[idx] / float(angle)
        x, y, z = [float(v) for v in axis.tolist()]
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        out[idx] = np.asarray(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=np.float32,
        )
    return out


def joint_global_transforms(
    *,
    pose_axis_angle: Any,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    pose = pose_to_smplx55_axis_angle(pose_axis_angle)
    joints = np.asarray(rest_joints, dtype=np.float32).reshape(-1, 3)
    pa = np.asarray(parents, dtype=np.int32).reshape(-1)
    n = min(int(joints.shape[0]), int(pa.shape[0]), int(pose.shape[0]))
    rot = axis_angle_to_matrix(pose[:n])
    out = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    for idx in range(n):
        local = np.eye(4, dtype=np.float32)
        local[:3, :3] = rot[idx]
        if idx == 0 or int(pa[idx]) < 0:
            local[:3, 3] = joints[idx]
            out[idx] = local
        else:
            parent = int(pa[idx])
            local[:3, 3] = joints[idx] - joints[parent]
            out[idx] = out[parent] @ local
    return out


def _rigid_frame(origin: np.ndarray, primary: np.ndarray, plane: np.ndarray) -> np.ndarray:
    x = np.asarray(primary - origin, dtype=np.float64)
    x /= max(float(np.linalg.norm(x)), 1.0e-10)
    z = np.cross(x, np.asarray(plane - origin, dtype=np.float64))
    z /= max(float(np.linalg.norm(z)), 1.0e-10)
    y = np.cross(z, x)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.stack((x, y, z), axis=1)
    out[:3, 3] = origin
    return out


def _interpolate_rigid(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    t = float(np.clip(alpha, 0.0, 1.0))
    Ra = np.asarray(a[:3, :3], dtype=np.float64)
    Rb = np.asarray(b[:3, :3], dtype=np.float64)
    delta = Rotation.from_matrix(Ra.T @ Rb).as_rotvec()
    R = Ra @ Rotation.from_rotvec(t * delta).as_matrix()
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R.astype(np.float32)
    out[:3, 3] = ((1.0 - t) * a[:3, 3] + t * b[:3, 3]).astype(np.float32)
    return out


def _segment_frame(origin: np.ndarray, endpoint: np.ndarray, reference_x: np.ndarray) -> np.ndarray:
    """Stable limb/head frame with its Y axis fixed by anatomical endpoints."""
    y = np.asarray(endpoint - origin, dtype=np.float64)
    y /= max(float(np.linalg.norm(y)), 1.0e-10)
    x = np.asarray(reference_x, dtype=np.float64)
    x -= float(x @ y) * y
    if float(np.linalg.norm(x)) < 1.0e-8:
        # A clavicle can be almost parallel to world X.  Choose the least
        # aligned canonical axis instead of producing a singular frame.
        x = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(y)))]
        x -= float(x @ y) * y
    x /= max(float(np.linalg.norm(x)), 1.0e-10)
    z = np.cross(x, y)
    z /= max(float(np.linalg.norm(z)), 1.0e-10)
    x = np.cross(y, z)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.stack((x, y, z), axis=1)
    out[:3, 3] = origin
    return out


def _endpoint_segment_delta(
    *,
    rest_a: np.ndarray,
    rest_b: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    rest_reference_x: np.ndarray,
    proximal_delta: np.ndarray,
    distal_delta: np.ndarray | None = None,
    twist_alpha: float = 0.0,
) -> np.ndarray:
    """Rigid transform for a limb segment; no blended global translations."""
    if float(np.linalg.norm(rest_b - rest_a)) < 1.0e-8:
        return np.eye(4, dtype=np.float64)
    F0 = _segment_frame(rest_a, rest_b, rest_reference_x)
    reference_delta = np.asarray(proximal_delta, dtype=np.float64)
    if distal_delta is not None and float(twist_alpha) > 0.0:
        reference_delta = _interpolate_rigid(
            np.asarray(proximal_delta, dtype=np.float64),
            np.asarray(distal_delta, dtype=np.float64),
            float(twist_alpha),
        ).astype(np.float64)
    F1 = _segment_frame(pose_a, pose_b, reference_delta[:3, :3] @ rest_reference_x)
    return F1 @ np.linalg.inv(F0)


def _endpoint_segment_pose_frame(
    *,
    rest_a: np.ndarray,
    rest_b: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    rest_reference_x: np.ndarray,
    proximal_delta: np.ndarray,
    distal_delta: np.ndarray | None = None,
    twist_alpha: float = 0.0,
) -> np.ndarray:
    """Return the posed segment frame ``F_seg(pose)`` (not the delta)."""
    reference_delta = np.asarray(proximal_delta, dtype=np.float64)
    if distal_delta is not None and float(twist_alpha) > 0.0:
        reference_delta = _interpolate_rigid(
            np.asarray(proximal_delta, dtype=np.float64),
            np.asarray(distal_delta, dtype=np.float64),
            float(twist_alpha),
        ).astype(np.float64)
    return _segment_frame(
        pose_a,
        pose_b,
        reference_delta[:3, :3] @ rest_reference_x,
    )


def _source_rest_local(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Return Blender bind-local matrices, deriving them for legacy v2 assets."""
    stored = getattr(asset, "source_rest_local", None)
    if stored is not None and np.asarray(stored).shape == np.asarray(asset.source_rest_global).shape:
        return np.asarray(stored, dtype=np.float64)
    global_rest = np.asarray(asset.source_rest_global, dtype=np.float64)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    local = global_rest.copy()
    for bi, parent in enumerate(parents.tolist()):
        if parent >= 0:
            local[bi] = np.linalg.inv(global_rest[parent]) @ global_rest[bi]
    return local


def _uses_segment_coupling(driver_type: str) -> bool:
    return (
        driver_type.startswith("forearm_segment_")
        or driver_type.startswith("clavicle_segment_")
        or driver_type.startswith("humerus_segment_")
        or driver_type.startswith("shin_segment_")
        or driver_type.startswith("knee_chain_")
        or driver_type.startswith("foot_chain_")
        or driver_type in {"head_segment", "head_orientation", "rib_segment"}
        or driver_type.startswith("scapula_")
        or driver_type.startswith("patella_")
    )


def _toe_chain_mask(asset: AnatomyRiggedAsset) -> np.ndarray:
    """All descendants of Blender toe controls (SMPL-X has no toe pose)."""
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    names = [str(name).lower() for name in asset.source_bone_names or []]
    mask = np.asarray(["toes_rotate" in name or "toe_rotate" in name for name in names], dtype=bool)
    changed = True
    while changed:
        inherited = np.asarray(
            [bool(parent >= 0 and mask[int(parent)]) for parent in parents], dtype=bool
        )
        changed = bool(np.any(inherited & ~mask))
        mask |= inherited
    return mask


def _uses_connected_upper_limb_fk(
    bone_index: int, parent_index: int, driver_types: list[str]
) -> bool:
    """Connected-FK for clavicle/humerus/forearm segment bones only.

    Wrist and finger ``direct_joint`` controls stay on absolute SMPL-X joints
    (45a8cf4 / ea38b787).  Mixing connected wrist FK with global finger FK
    creates a ~17 mm frame split at the palm on both hands.
    """
    arm_prefixes = (
        "clavicle_segment_",
        "humerus_segment_",
        "forearm_proximal_",
        "forearm_segment_",
    )
    own = str(driver_types[bone_index]) if bone_index < len(driver_types) else ""
    return own.startswith(arm_prefixes)


def _uses_bind_local_fk(
    bone_index: int,
    parent_index: int,
    driver_types: list[str],
    toe_chain: np.ndarray,
) -> bool:
    """Whether a source bone is solved from its parent's Blender bind frame."""
    return bool(toe_chain[bone_index]) or _uses_connected_upper_limb_fk(
        bone_index, parent_index, driver_types
    )


def _has_foot_chain_ancestor(
    bone_index: int,
    chain_type: str,
    parents: np.ndarray,
    driver_types: list[str],
) -> bool:
    """True when a same-side foot_chain control sits above ``bone_index`` via helpers."""
    current = int(parents[bone_index])
    visited = 0
    while current >= 0 and visited <= len(parents):
        own = str(driver_types[current]) if current < len(driver_types) else ""
        if own == chain_type:
            return True
        if own != "parent_follow":
            return False
        current = int(parents[current])
        visited += 1
    return False


def _segment_pose_frame_for_bone(
    *,
    bi: int,
    driver_type: str,
    asset: AnatomyRiggedAsset,
    rest_points: np.ndarray,
    pose_points: np.ndarray,
    joint_index: dict[str, int],
    joint_delta: np.ndarray,
    pose_global: np.ndarray,
) -> np.ndarray:
    a = int(asset.source_bone_smplx_a[bi])
    b = int(asset.source_bone_smplx_b[bi])
    reference_x = np.asarray(asset.source_rest_global[bi], dtype=np.float64)[:3, 0]
    if driver_type == "head_orientation":
        head = joint_index["head"]
        return np.asarray(pose_global[head], dtype=np.float64)
    if float(np.linalg.norm(rest_points[b] - rest_points[a])) < 1.0e-8:
        raise ValueError(f"degenerate segment joints for bone index {bi}")
    if (
        driver_type.startswith("forearm_segment_")
        or driver_type.startswith("clavicle_segment_")
        or driver_type.startswith("humerus_segment_")
        or driver_type.startswith("shin_segment_")
        or driver_type.startswith("knee_chain_")
        or driver_type.startswith("foot_chain_")
        or driver_type in {"head_segment", "rib_segment"}
        or driver_type.startswith("patella_")
    ) and a != b:
        return _endpoint_segment_pose_frame(
            rest_a=rest_points[a],
            rest_b=rest_points[b],
            pose_a=pose_points[a],
            pose_b=pose_points[b],
            rest_reference_x=reference_x,
            proximal_delta=joint_delta[a],
            distal_delta=joint_delta[b],
            twist_alpha=float(asset.source_bone_blend[bi]),
        )
    if driver_type.startswith("scapula_"):
        side = "left" if driver_type.endswith("left") else "right"
        s, c, h = (joint_index["spine3"], joint_index[f"{side}_collar"], joint_index[f"{side}_shoulder"])
        return _rigid_frame(pose_points[h], pose_points[c], pose_points[s])
    raise ValueError(f"unsupported segment pose frame for driver_type={driver_type}")


def source_bone_skinning_transforms(
    asset: AnatomyRiggedAsset,
    pose_axis_angle: Any,
) -> np.ndarray:
    """Solve all Blender source-bone transforms from a full SMPL-X pose."""
    if asset.source_bone_names is None:
        raise ValueError("source bone transforms requested for a legacy asset")
    pose_global = joint_global_transforms(
        pose_axis_angle=pose_axis_angle,
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    joint_delta = pose_global @ np.linalg.inv(rest_global)
    rest_points = np.asarray(asset.rest_joints, dtype=np.float64)
    pose_points = pose_global[:, :3, 3]
    joint_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    source_delta = np.tile(np.eye(4, dtype=np.float32), (len(asset.source_bone_names), 1, 1))
    types = asset.source_bone_driver_types or ["direct_joint"] * len(asset.source_bone_names)
    for bi, driver_type in enumerate(types):
        a = int(asset.source_bone_smplx_a[bi])
        b = int(asset.source_bone_smplx_b[bi])
        alpha = float(asset.source_bone_blend[bi])
        if (
            driver_type.startswith("forearm_segment_")
            or driver_type.startswith("forearm_proximal_")
            or driver_type.startswith("clavicle_segment_")
            or driver_type.startswith("humerus_segment_")
            or driver_type.startswith("shin_segment_")
            or driver_type.startswith("knee_chain_")
            or driver_type.startswith("foot_chain_")
            or driver_type in {"head_segment", "rib_segment"}
        ) and a != b:
            reference_x = np.asarray(asset.source_rest_global[bi], dtype=np.float64)[:3, 0]
            source_delta[bi] = _endpoint_segment_delta(
                rest_a=rest_points[a], rest_b=rest_points[b],
                pose_a=pose_points[a], pose_b=pose_points[b],
                rest_reference_x=reference_x, proximal_delta=joint_delta[a],
                distal_delta=joint_delta[b], twist_alpha=alpha,
            ).astype(np.float32)
        elif driver_type == "head_orientation":
            head = joint_index["head"]
            source_delta[bi] = joint_delta[head].astype(np.float32)
        elif driver_type.startswith("scapula_"):
            side = "left" if driver_type.endswith("left") else "right"
            s, c, h = (joint_index["spine3"], joint_index[f"{side}_collar"], joint_index[f"{side}_shoulder"])
            F0 = _rigid_frame(rest_points[h], rest_points[c], rest_points[s])
            F1 = _rigid_frame(pose_points[h], pose_points[c], pose_points[s])
            source_delta[bi] = (F1 @ np.linalg.inv(F0)).astype(np.float32)
        elif driver_type.startswith("patella_"):
            side = "left" if driver_type.endswith("left") else "right"
            knee, ankle = (joint_index[f"{side}_knee"], joint_index[f"{side}_ankle"])
            reference_x = np.asarray(asset.source_rest_global[bi], dtype=np.float64)[:3, 0]
            source_delta[bi] = _endpoint_segment_delta(
                rest_a=rest_points[knee], rest_b=rest_points[ankle],
                pose_a=pose_points[knee], pose_b=pose_points[ankle],
                rest_reference_x=reference_x, proximal_delta=joint_delta[knee],
            ).astype(np.float32)
        elif a != b:
            source_delta[bi] = _interpolate_rigid(joint_delta[a], joint_delta[b], alpha)
        else:
            source_delta[bi] = joint_delta[a].astype(np.float32)

    rest_global_bones = np.asarray(asset.source_rest_global, dtype=np.float32)
    coupling = (
        np.asarray(asset.source_segment_coupling, dtype=np.float32)
        if asset.source_segment_coupling is not None and asset.source_segment_coupling.size
        else None
    )
    posed_global = np.empty_like(rest_global_bones)
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    rest_local_bones = _source_rest_local(asset)
    toe_chain = _toe_chain_mask(asset)
    use_connect = (
        np.asarray(asset.source_bone_use_connect, dtype=bool)
        if getattr(asset, "source_bone_use_connect", None) is not None
        else np.zeros(len(asset.source_bone_names), dtype=bool)
    )
    for bi, driver_type in enumerate(types):
        parent = int(source_parents[bi])
        # A Blender follower/helper has no independent SMPL-X control.  Preserve
        # its exact bind-local transform and let the evaluated source hierarchy
        # carry the parent motion.  Previous versions exported this hierarchy
        # but solved every follower as an unrelated global joint.
        if driver_type == "parent_follow" and parent >= 0:
            posed_global[bi] = (
                np.asarray(posed_global[parent], dtype=np.float64) @ rest_local_bones[bi]
            ).astype(np.float32)
            continue
        # SMPL-X has one rigid foot pose and no articulated toe parameters.
        # Preserve Blender's authored foot/toe subtree exactly: only the first
        # foot-chain control receives the SMPL-X segment motion, while all of
        # its descendants retain their bind-local transform.  This is driven
        # by exported rig semantics, not mesh names or spatial thresholds.
        # Toe rotors sit on parent_follow helpers (bone167, …) between two
        # foot_chain controls; they must follow the helper, not the ankle→foot
        # segment applied independently.  Chain roots (Ankle_Rot under eff16)
        # must receive the segment motion themselves.
        if parent >= 0 and driver_type.startswith("foot_chain_"):
            parent_type = str(types[parent]) if parent < len(types) else ""
            chain_type = str(driver_type)
            is_follower = parent_type == chain_type or (
                parent_type == "parent_follow"
                and _has_foot_chain_ancestor(bi, chain_type, source_parents, types)
            )
            if is_follower:
                posed_global[bi] = (
                    np.asarray(posed_global[parent], dtype=np.float64) @ rest_local_bones[bi]
                ).astype(np.float32)
                continue
        # Blender's connected bones share their head with the parent's tail.
        # Keep that bind-local translation exact and take only the desired
        # *local rotation* from the SMPL-X-driven target.  This is hierarchy
        # semantics exported from Blender, so it applies equally to elbows,
        # wrists, knees and ankles without a body-part correction table.
        # Head_Bone intentionally bypasses this branch: it must match the
        # SMPL-X head global orientation, including pitch.
        if (
            parent >= 0
            and (bool(use_connect[bi]) or bool(toe_chain[bi]))
            and driver_type != "head_orientation"
            and _uses_bind_local_fk(bi, parent, types, toe_chain)
        ):
            # Convert the two global SMPL-X-driven deltas into the child's
            # rotation *relative to its Blender parent*.  Using the child's
            # desired global transform directly would apply the parent swing a
            # second time after FK, which is the source of forearm twists.
            relative_delta = np.linalg.inv(source_delta[parent]) @ source_delta[bi]
            local_motion = (
                np.linalg.inv(rest_global_bones[parent])
                @ relative_delta
                @ rest_global_bones[parent]
            )
            local = np.asarray(rest_local_bones[bi], dtype=np.float64).copy()
            local[:3, :3] = (local_motion @ rest_local_bones[bi])[:3, :3]
            posed_global[bi] = (
                np.asarray(posed_global[parent], dtype=np.float64) @ local
            ).astype(np.float32)
            continue
        use_coupling = (
            coupling is not None
            and coupling.shape[0] == len(asset.source_bone_names)
            and _uses_segment_coupling(driver_type)
            and float(np.max(np.abs(coupling[bi] - np.eye(4, dtype=np.float32)))) > 1.0e-6
        )
        if use_coupling:
            try:
                F_pose = _segment_pose_frame_for_bone(
                    bi=bi,
                    driver_type=driver_type,
                    asset=asset,
                    rest_points=rest_points,
                    pose_points=pose_points,
                    joint_index=joint_index,
                    joint_delta=joint_delta,
                    pose_global=pose_global,
                )
                posed_global[bi] = (F_pose @ np.asarray(coupling[bi], dtype=np.float64)).astype(np.float32)
            except ValueError:
                posed_global[bi] = source_delta[bi] @ rest_global_bones[bi]
        else:
            posed_global[bi] = source_delta[bi] @ rest_global_bones[bi]
    # #region agent log
    if asset.source_bone_names is not None:
        bn = list(asset.source_bone_names)
        hand_pairs = [
            ("Wrist_Rotate_L", "Finger_Index_L3", "left"),
            ("Wrist_Rotate_R1", "Finger_Rotate_R4", "right"),
        ]
        hand_rows: list[dict[str, Any]] = []
        for wrist_name, finger_name, side in hand_pairs:
            if wrist_name not in bn or finger_name not in bn:
                continue
            wi, fi = int(bn.index(wrist_name)), int(bn.index(finger_name))
            wrist_pos = np.asarray(posed_global[wi, :3, 3], dtype=np.float64)
            finger_pos = np.asarray(posed_global[fi, :3, 3], dtype=np.float64)
            wrist_global = np.asarray(source_delta[wi, :3, 3], dtype=np.float64)
            finger_global = np.asarray(source_delta[fi, :3, 3], dtype=np.float64)
            hand_rows.append(
                {
                    "side": side,
                    "wrist": wrist_name,
                    "finger": finger_name,
                    "posed_gap_mm": float(np.linalg.norm(finger_pos - wrist_pos) * 1000.0),
                    "source_delta_gap_mm": float(np.linalg.norm(finger_global - wrist_global) * 1000.0),
                    "wrist_connected_fk": bool(
                        _uses_bind_local_fk(wi, int(source_parents[wi]), types, toe_chain)
                    ),
                    "finger_connected_fk": bool(
                        _uses_bind_local_fk(fi, int(source_parents[fi]), types, toe_chain)
                    ),
                    "wrist_use_connect": bool(use_connect[wi]),
                    "finger_use_connect": bool(use_connect[fi]),
                }
            )
        if hand_rows:
            _agent_log(
                hypothesis_id="B",
                location="anatomy_lbs.py:source_bone_skinning_transforms",
                message="hand wrist vs finger frame gap",
                data={"pairs": hand_rows},
            )
    # #endregion
    return posed_global @ np.asarray(asset.source_inverse_bind, dtype=np.float32)


def skin_vertices(
    asset: AnatomyRiggedAsset,
    pose_axis_angle: Any,
    *,
    transl: Any | None = None,
) -> np.ndarray:
    asset.validate()
    vertices = np.asarray(asset.vertices_rest, dtype=np.float32).reshape(-1, 3)
    if asset.source_bone_names is not None:
        transforms = source_bone_skinning_transforms(asset, pose_axis_angle)
    else:
        weights = _dense_asset_weights(asset)
        inverse_bind = np.asarray(asset.inverse_bind, dtype=np.float32)
        global_tf = joint_global_transforms(
            pose_axis_angle=pose_axis_angle,
            rest_joints=asset.rest_joints,
            parents=asset.parents,
        )
        joint_count = min(global_tf.shape[0], inverse_bind.shape[0], weights.shape[1])
        transforms = np.matmul(global_tf[:joint_count], inverse_bind[:joint_count])
    if _cuda_requested():
        return _skin_vertices_cuda(asset, transforms, transl)
    if asset.driver_indices is not None and asset.driver_weights is not None:
        selected = transforms[np.asarray(asset.driver_indices, dtype=np.int64)]
        blended = np.sum(selected * np.asarray(asset.driver_weights, dtype=np.float32)[..., None, None], axis=1)
    else:
        weights = _dense_asset_weights(asset)
        joint_count = min(transforms.shape[0], weights.shape[1])
        blended = np.matmul(weights[:, :joint_count], transforms[:joint_count].reshape(joint_count, 16)).reshape(-1, 4, 4)
    homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
    posed = np.matmul(blended, homo[:, :, None])[:, :3, 0].astype(np.float32)
    if transl is not None:
        posed += np.asarray(transl, dtype=np.float32).reshape(1, 3)
    return posed


def skin_points(
    asset: AnatomyRiggedAsset,
    rest_points: Any,
    *,
    pose_axis_angle: Any,
    transl: Any | None = None,
    anchor_vertices: Any | None = None,
    anchor_weights: Any | None = None,
    neighbor_k: int = 4,
) -> np.ndarray:
    """Skin arbitrary rest points by interpolating LBS weights from nearby mesh vertices."""
    asset.validate()
    pts = np.asarray(rest_points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts.copy()
    verts = np.asarray(asset.vertices_rest if anchor_vertices is None else anchor_vertices, dtype=np.float32).reshape(-1, 3)
    weights = np.asarray(_dense_asset_weights(asset) if anchor_weights is None else anchor_weights, dtype=np.float32)
    k = max(1, min(int(neighbor_k), int(verts.shape[0])))
    try:
        from scipy.spatial import cKDTree

        dist, idx = cKDTree(verts).query(pts, k=k)
    except Exception:
        dist = np.linalg.norm(verts[:, None, :] - pts[None, :, :], axis=2).T
        idx = np.argsort(dist, axis=1)[:, :k]
        dist = np.take_along_axis(dist, idx, axis=1)
    if k == 1:
        dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, 1)
    w_dist = 1.0 / (np.square(dist) + 1.0e-8)
    w_dist /= np.maximum(w_dist.sum(axis=1, keepdims=True), 1.0e-8)
    point_weights = np.zeros((pts.shape[0], weights.shape[1]), dtype=np.float32)
    for ki in range(k):
        point_weights += w_dist[:, ki : ki + 1] * weights[idx[:, ki]]
    point_weights /= np.maximum(point_weights.sum(axis=1, keepdims=True), 1.0e-8)

    if asset.source_bone_names is not None:
        transforms = source_bone_skinning_transforms(asset, pose_axis_angle)
    else:
        inverse_bind = np.asarray(asset.inverse_bind, dtype=np.float32)
        global_tf = joint_global_transforms(
            pose_axis_angle=pose_axis_angle,
            rest_joints=asset.rest_joints,
            parents=asset.parents,
        )
        joint_count = min(global_tf.shape[0], inverse_bind.shape[0], weights.shape[1])
        transforms = np.matmul(global_tf[:joint_count], inverse_bind[:joint_count])
    joint_count = min(transforms.shape[0], weights.shape[1])
    blended = np.matmul(point_weights[:, :joint_count], transforms.reshape(joint_count, 16)).reshape(-1, 4, 4)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    posed = np.matmul(blended, homo[:, :, None])[:, :3, 0].astype(np.float32)
    if transl is not None:
        posed += np.asarray(transl, dtype=np.float32).reshape(1, 3)
    return posed


def compute_point_lbs_weights(
    asset: AnatomyRiggedAsset,
    rest_points: Any,
    *,
    anchor_vertices: Any | None = None,
    anchor_weights: Any | None = None,
    neighbor_k: int = 6,
) -> np.ndarray:
    """Interpolate LBS weights for arbitrary rest points from nearby mesh vertices."""
    asset.validate()
    pts = np.asarray(rest_points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        width = len(asset.source_bone_names) if asset.source_bone_names is not None else len(asset.joint_names)
        return np.zeros((0, width), dtype=np.float32)
    verts = np.asarray(asset.vertices_rest if anchor_vertices is None else anchor_vertices, dtype=np.float32).reshape(-1, 3)
    weights = np.asarray(_dense_asset_weights(asset) if anchor_weights is None else anchor_weights, dtype=np.float32)
    k = max(1, min(int(neighbor_k), int(verts.shape[0])))
    try:
        from scipy.spatial import cKDTree

        dist, idx = cKDTree(verts).query(pts, k=k)
    except Exception:
        dist = np.linalg.norm(verts[:, None, :] - pts[None, :, :], axis=2).T
        idx = np.argsort(dist, axis=1)[:, :k]
        dist = np.take_along_axis(dist, idx, axis=1)
    if k == 1:
        dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, 1)
    w_dist = 1.0 / (np.square(dist) + 1.0e-8)
    w_dist /= np.maximum(w_dist.sum(axis=1, keepdims=True), 1.0e-8)
    point_weights = np.zeros((pts.shape[0], weights.shape[1]), dtype=np.float32)
    for ki in range(k):
        point_weights += w_dist[:, ki : ki + 1] * weights[idx[:, ki]]
    point_weights /= np.maximum(point_weights.sum(axis=1, keepdims=True), 1.0e-8)
    return point_weights


def skin_points_with_weights(
    asset: AnatomyRiggedAsset,
    rest_points: Any,
    point_weights: np.ndarray,
    *,
    pose_axis_angle: Any,
    transl: Any | None = None,
) -> np.ndarray:
    asset.validate()
    pts = np.asarray(rest_points, dtype=np.float32).reshape(-1, 3)
    weights = np.asarray(point_weights, dtype=np.float32)
    if asset.source_bone_names is not None:
        transforms = source_bone_skinning_transforms(asset, pose_axis_angle)
    else:
        inverse_bind = np.asarray(asset.inverse_bind, dtype=np.float32)
        global_tf = joint_global_transforms(
            pose_axis_angle=pose_axis_angle,
            rest_joints=asset.rest_joints,
            parents=asset.parents,
        )
        joint_count = min(global_tf.shape[0], inverse_bind.shape[0], weights.shape[1])
        transforms = np.matmul(global_tf[:joint_count], inverse_bind[:joint_count])
    joint_count = min(transforms.shape[0], weights.shape[1])
    blended = np.matmul(weights[:, :joint_count], transforms.reshape(joint_count, 16)).reshape(-1, 4, 4)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    posed = np.matmul(blended, homo[:, :, None])[:, :3, 0].astype(np.float32)
    if transl is not None:
        posed += np.asarray(transl, dtype=np.float32).reshape(1, 3)
    return posed
