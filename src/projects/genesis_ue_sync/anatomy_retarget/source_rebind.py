"""Keep Blender source-bone bind frames consistent with rest-space warps.

The anatomy vertices retain Blender's original sparse source-bone weights.  Any
canonical, shape, or containment warp must therefore update the source bind
frames as well; otherwise the same weights are evaluated in a different rest
coordinate system and produce detached rigid anatomy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset


def _weighted_rigid(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the proper rigid transform best fitting ``source -> target``."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w /= max(float(w.sum()), 1.0e-12)
    src_center = np.einsum("n,nj->j", w, source)
    dst_center = np.einsum("n,nj->j", w, target)
    x = source - src_center
    y = target - dst_center
    u, _s, vt = np.linalg.svd((x * w[:, None]).T @ y)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1] *= -1.0
        rot = vt.T @ u.T
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = dst_center - rot @ src_center
    return out


def rebind_source_rig(
    asset: AnatomyRiggedAsset,
    *,
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    stage: str,
    minimum_weight: float = 0.05,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Synchronize all source bind frames after a generic rest-space warp.

    Frames are fitted from the vertices influenced by each original Blender
    bone.  This is generic (no body-part rules) and preserves the original
    source weights.  At zero pose every source skinning matrix remains identity.
    """
    if asset.source_bone_names is None or asset.source_rest_global is None:
        return asset, {"stage": stage, "source_rig": "legacy_skip"}
    if asset.driver_indices is None or asset.driver_weights is None:
        raise ValueError("source-rig rebind requires sparse Blender driver weights")
    src = np.asarray(source_vertices, dtype=np.float64)
    dst = np.asarray(target_vertices, dtype=np.float64)
    if src.shape != dst.shape or src.shape != np.asarray(asset.vertices_rest).shape:
        raise ValueError("source/target vertices must match the anatomy rest mesh")
    idx = np.asarray(asset.driver_indices, dtype=np.int32)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    old = np.asarray(asset.source_rest_global, dtype=np.float64)
    new = old.copy()
    bone_transforms = np.tile(np.eye(4, dtype=np.float64), (len(asset.source_bone_names), 1, 1))
    residuals: list[float] = []
    fitted = 0
    for bone in range(len(asset.source_bone_names)):
        mask = idx == bone
        row_weight = np.where(mask, weights, 0.0).sum(axis=1)
        selected = row_weight >= float(minimum_weight)
        if int(np.count_nonzero(selected)) < 3:
            continue
        transform = _weighted_rigid(src[selected], dst[selected], row_weight[selected])
        predicted = src[selected] @ transform[:3, :3].T + transform[:3, 3]
        residuals.append(float(np.sqrt(np.average(np.sum((predicted - dst[selected]) ** 2, axis=1), weights=row_weight[selected]))))
        new[bone] = transform @ old[bone]
        bone_transforms[bone] = transform
        fitted += 1
    # Preserve the actual Blender FK relation for bones that have no
    # independent SMPL-X control.  Fitting every helper/follower globally was
    # enough to keep zero-pose LBS identity, but it silently broke the local
    # shoulder/elbow/neck hierarchy used at runtime.
    parents = np.asarray(asset.source_bone_parents, dtype=np.int32)
    types = list(asset.source_bone_driver_types or [])
    use_connect = (
        np.asarray(asset.source_bone_use_connect, dtype=bool)
        if getattr(asset, "source_bone_use_connect", None) is not None
        else np.zeros(len(parents), dtype=bool)
    )
    old_local = (
        np.asarray(asset.source_rest_local, dtype=np.float64).copy()
        if asset.source_rest_local is not None
        else old.copy()
    )
    if asset.source_rest_local is None:
        for bone, parent in enumerate(parents.tolist()):
            if int(parent) >= 0:
                old_local[bone] = np.linalg.inv(old[int(parent)]) @ old[bone]
    hierarchy_preserved = 0
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) < 0:
            continue
        if bone >= len(types):
            continue
        is_upper_limb_connected = bool(use_connect[bone]) and (
            str(types[bone]).startswith(
                ("clavicle_segment_", "humerus_segment_", "forearm_segment_")
            )
            or str(types[int(parent)]).startswith(
                ("clavicle_segment_", "humerus_segment_", "forearm_segment_")
            )
        )
        is_chain_follower = (
            str(types[bone]).startswith("foot_chain_")
            and int(parent) >= 0
            and str(types[int(parent)]) == str(types[bone])
        )
        if (
            str(types[bone]) != "parent_follow"
            and not is_upper_limb_connected
            and not is_chain_follower
        ):
            continue
        new[bone] = new[int(parent)] @ old_local[bone]
        hierarchy_preserved += 1
    bone_transforms = new @ np.linalg.inv(old)
    inverse = np.linalg.inv(new).astype(np.float32)
    updates: dict[str, Any] = {
        "source_rest_global": new.astype(np.float32),
        "source_inverse_bind": inverse,
    }
    if asset.source_rest_local is not None:
        local = new.copy()
        for bone, parent in enumerate(parents.tolist()):
            if int(parent) >= 0:
                local[bone] = np.linalg.inv(new[int(parent)]) @ new[bone]
        updates["source_rest_local"] = local.astype(np.float32)
    for field_name in ("source_bone_head", "source_bone_tail"):
        value = getattr(asset, field_name)
        if value is None:
            continue
        points = np.asarray(value, dtype=np.float64)
        moved = np.einsum("bij,bj->bi", bone_transforms[:, :3, :3], points)
        moved += bone_transforms[:, :3, 3]
        updates[field_name] = moved.astype(np.float32)
    meta = dict(asset.metadata or {})
    history = list(meta.get("source_rig_rebind", []))
    history.append({"stage": str(stage), "fitted_bones": fitted})
    meta["source_rig_rebind"] = history
    result = type(asset)(**{**asset.__dict__, **updates, "metadata": meta})
    return result, {
        "stage": str(stage),
        "fitted_bones": int(fitted),
        "unfitted_bones": int(len(asset.source_bone_names) - fitted),
        "weighted_fit_rms_m": float(np.mean(residuals)) if residuals else 0.0,
        "weighted_fit_max_m": float(np.max(residuals)) if residuals else 0.0,
        "hierarchy_preserved_followers": int(hierarchy_preserved),
    }


def source_bind_roundtrip(asset: AnatomyRiggedAsset) -> dict[str, Any]:
    """Evaluate the exact zero-pose source LBS identity invariant."""
    if asset.source_rest_global is None or asset.source_inverse_bind is None:
        return {"source_rig": "legacy_skip"}
    skin = np.asarray(asset.source_rest_global, dtype=np.float64) @ np.asarray(asset.source_inverse_bind, dtype=np.float64)
    identity_error = np.max(np.abs(skin - np.eye(4)[None, :, :]))
    return {"max_matrix_error": float(identity_error), "pass": bool(identity_error <= 1.0e-6)}
