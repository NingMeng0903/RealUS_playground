"""Fit subject target bind frames without mutating Blender authored binds.

The anatomy vertices retain Blender's original sparse source-bone weights.  Any
The authored source frames remain the immutable Blender authority.  Rest-space
fitting updates a separate target bind, while source-driver coupling connects
the runtime SMPL-X frame to that final bind.
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


def _weighted_similarity(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return a proper rigid frame update plus the fitted uniform length scale."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w /= max(float(w.sum()), 1.0e-12)
    src_center = np.einsum("n,nj->j", w, source)
    dst_center = np.einsum("n,nj->j", w, target)
    x = source - src_center
    y = target - dst_center
    u, _singular, vt = np.linalg.svd((x * w[:, None]).T @ y)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    rotated = x @ rotation.T
    denominator = float(np.einsum("n,nj,nj->", w, x, x))
    scale = (
        float(np.einsum("n,nj,nj->", w, y, rotated)) / denominator
        if denominator > 1.0e-12
        else 1.0
    )
    scale = float(np.clip(scale, 0.25, 4.0))
    translation = dst_center - scale * (rotation @ src_center)
    rigid = np.eye(4, dtype=np.float64)
    rigid[:3, :3] = rotation
    rigid[:3, 3] = translation
    return rigid, scale


def rebind_source_rig(
    asset: AnatomyRiggedAsset,
    *,
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    stage: str,
    minimum_weight: float = 0.05,
    bone_mask: np.ndarray | None = None,
    fallback_to_soft: bool = True,
    anchor_joint_local: bool = False,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Synchronize all source bind frames after a generic rest-space warp.

    Frames are fitted from the vertices influenced by each original Blender
    bone.  This is generic (no body-part rules) and preserves the original
    source weights.  At zero pose every source skinning matrix remains identity.
    """
    if asset.source_bone_names is None or asset.target_bind_global is None:
        return asset, {"stage": stage, "source_rig": "legacy_skip"}
    if asset.driver_indices is None or asset.driver_weights is None:
        raise ValueError("source-rig rebind requires sparse Blender driver weights")
    src = np.asarray(source_vertices, dtype=np.float64)
    dst = np.asarray(target_vertices, dtype=np.float64)
    if src.shape != dst.shape or src.shape != np.asarray(asset.vertices_rest).shape:
        raise ValueError("source/target vertices must match the anatomy rest mesh")
    idx = np.asarray(asset.driver_indices, dtype=np.int32)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    old = np.asarray(asset.target_bind_global, dtype=np.float64)
    new = old.copy()
    fitted_mask = np.zeros(len(asset.source_bone_names), dtype=bool)
    bone_scales = np.ones(len(asset.source_bone_names), dtype=np.float64)
    residuals: list[float] = []
    fitted = 0
    bone_only = np.asarray(bone_mask, dtype=bool) if bone_mask is not None else None
    bone_preferred_fits = 0
    soft_fallback_fits = 0
    for bone in range(len(asset.source_bone_names)):
        mask = idx == bone
        row_weight = np.where(mask, weights, 0.0).sum(axis=1)
        selected = row_weight >= float(minimum_weight)
        if bone_only is not None:
            selected_bone = selected & bone_only
            # Anatomical bone geometry is the authority for a skeleton frame.
            # Some authored helpers only influence vessels, nerves or organs;
            # retain the full weighted fallback for those bones rather than
            # dropping their bind update altogether.
            if int(np.count_nonzero(selected_bone)) >= 3:
                selected = selected_bone
                bone_preferred_fits += 1
            elif fallback_to_soft:
                soft_fallback_fits += 1
            else:
                selected = selected_bone
        if int(np.count_nonzero(selected)) < 3:
            continue
        transform, scale = _weighted_similarity(
            src[selected], dst[selected], row_weight[selected]
        )
        predicted = scale * (src[selected] @ transform[:3, :3].T) + transform[:3, 3]
        residuals.append(float(np.sqrt(np.average(np.sum((predicted - dst[selected]) ** 2, axis=1), weights=row_weight[selected]))))
        new[bone, :3, :3] = transform[:3, :3] @ old[bone, :3, :3]
        new[bone, :3, 3] = (
            scale * (transform[:3, :3] @ old[bone, :3, 3])
            + transform[:3, 3]
        )
        bone_scales[bone] = scale
        fitted_mask[bone] = True
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
        np.asarray(asset.target_bind_local, dtype=np.float64).copy()
        if asset.target_bind_local is not None
        else old.copy()
    )
    if asset.target_bind_local is None:
        for bone, parent in enumerate(parents.tolist()):
            if int(parent) >= 0:
                old_local[bone] = np.linalg.inv(old[int(parent)]) @ old[bone]
    # Deform followers carry the authored vertex weights while their parent
    # rotation controllers often carry none.  Infer those unsupported
    # controller frames from a fitted connected/follower child before filling
    # any remaining hierarchy gaps.  The former parent-first overwrite threw
    # away every fitted finger frame and left rotations around the old palm.
    inferred_controllers = 0
    supported = fitted_mask.copy()
    for bone in range(len(parents) - 1, -1, -1):
        parent = int(parents[bone])
        if parent < 0 or not supported[bone] or supported[parent]:
            continue
        if str(types[bone]) != "bind_follow" and not bool(use_connect[bone]):
            continue
        new[parent] = new[bone] @ np.linalg.inv(old_local[bone])
        bone_scales[parent] = bone_scales[bone]
        supported[parent] = True
        inferred_controllers += 1

    hierarchy_preserved = 0
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) < 0 or supported[bone]:
            continue
        if bone >= len(types):
            continue
        if str(types[bone]) != "bind_follow" and not bool(use_connect[bone]):
            continue
        new[bone] = new[int(parent)] @ old_local[bone]
        bone_scales[bone] = bone_scales[int(parent)]
        supported[bone] = True
        hierarchy_preserved += 1

    # A joint-local controller is the source-rig rotation center for its
    # mapped SMPL-X degree of freedom.  Weighted rigid fitting recovers its
    # axis but not that pivot (unweighted controllers can remain 1-2 cm away
    # from the mapped anatomy).  Anchor the fitted origin before rebuilding
    # locals so source FK rotates around the subject-beta joint center.
    anchored_joint_local = 0
    hard_anchor = np.zeros(len(parents), dtype=bool)
    if anchor_joint_local:
        smplx_a = np.asarray(asset.source_bone_smplx_a, dtype=np.int64)
        smplx_b = np.asarray(asset.source_bone_smplx_b, dtype=np.int64)
        target_joints = np.asarray(asset.rest_joints, dtype=np.float64)
        for bone, mode in enumerate(types):
            parent = int(parents[bone])
            starts_new_segment = bool(
                str(mode) == "segment_root"
                and (
                    parent < 0
                    or not bool(use_connect[bone])
                    or int(smplx_a[parent]) != int(smplx_a[bone])
                    or int(smplx_b[parent]) != int(smplx_b[bone])
                )
            )
            if str(mode) != "joint_local" and not starts_new_segment:
                continue
            joint = int(smplx_a[bone])
            if joint < 0 or joint >= len(target_joints):
                raise ValueError(f"anchored source bone {bone} has no valid target joint")
            new[bone, :3, 3] = target_joints[joint]
            supported[bone] = True
            hard_anchor[bone] = True
            if str(mode) == "joint_local":
                anchored_joint_local += 1

    # Blender connected bones share one physical joint even when the volume
    # map changes segment length.  Keep each fitted rotation, but place the
    # child origin at the mapped parent tail so the runtime cannot open a gap.
    old_tail = (
        np.asarray(asset.target_bone_tail, dtype=np.float64)
        if asset.target_bone_tail is not None
        else np.asarray(asset.source_bone_tail, dtype=np.float64)
    )
    old_origin = old[:, :3, 3]
    mapped_head = new[:, :3, 3].copy()
    mapped_tail = np.empty_like(mapped_head)
    for bone in range(len(new)):
        rotation_delta = new[bone, :3, :3] @ np.linalg.inv(old[bone, :3, :3])
        mapped_tail[bone] = mapped_head[bone] + bone_scales[bone] * (
            rotation_delta @ (old_tail[bone] - old_origin[bone])
        )

    connected_anchors = 0
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) < 0 or not bool(use_connect[bone]):
            continue
        if bool(hard_anchor[bone]):
            # A mapped rotation center or a semantic limb-boundary anchor is
            # authoritative.  Pull the connected parent tail to that center
            # instead of overwriting the center with the old parent length.
            mapped_tail[int(parent)] = mapped_head[bone]
            connected_anchors += 1
            continue
        anchor = mapped_tail[int(parent)].copy()
        shift = anchor - mapped_head[bone]
        new[bone, :3, 3] = anchor
        mapped_head[bone] = anchor
        mapped_tail[bone] += shift
        connected_anchors += 1
    inverse = np.linalg.inv(new).astype(np.float32)
    updates: dict[str, Any] = {
        "target_rest_global": new.astype(np.float32),
        "target_inverse_bind": inverse,
    }
    if asset.target_bind_local is not None:
        local = new.copy()
        for bone, parent in enumerate(parents.tolist()):
            if int(parent) >= 0:
                local[bone] = np.linalg.inv(new[int(parent)]) @ new[bone]
        updates["target_rest_local"] = local.astype(np.float32)
    updates["target_bone_head"] = mapped_head.astype(np.float32)
    updates["target_bone_tail"] = mapped_tail.astype(np.float32)
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
        "controllers_inferred_from_weighted_children": int(inferred_controllers),
        "connected_shared_anchors_enforced": int(connected_anchors),
        "bone_preferred_fits": int(bone_preferred_fits),
        "soft_fallback_fits": int(soft_fallback_fits),
        "joint_local_centers_anchored": int(anchored_joint_local),
        "bone_length_scale_min": float(np.min(bone_scales)),
        "bone_length_scale_max": float(np.max(bone_scales)),
        "bone_length_scale_p99": float(np.quantile(bone_scales, 0.99)),
    }


def rebind_selected_source_bones(
    asset: AnatomyRiggedAsset,
    *,
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    bone_names: tuple[str, ...] | list[str],
    stage: str,
    minimum_weight: float = 0.05,
    bone_mask: np.ndarray | None = None,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Rebind named bones while preserving every other fitted global frame.

    Local matrices of children are recomputed from the preserved globals.  A
    child's world-space bind therefore cannot move merely because its parent
    was one of the geometrically refitted bones.
    """
    names = list(asset.source_bone_names or [])
    requested = [str(name) for name in bone_names]
    missing = sorted(set(requested) - set(names))
    if missing:
        raise ValueError(f"unknown selected source bone(s): {missing}")
    if not requested:
        return asset, {"stage": str(stage), "fitted_bones": 0, "selected_bones": []}

    candidate, full_report = rebind_source_rig(
        asset,
        source_vertices=source_vertices,
        target_vertices=target_vertices,
        stage=stage,
        minimum_weight=minimum_weight,
        bone_mask=bone_mask,
        fallback_to_soft=False,
        anchor_joint_local=False,
    )
    selected = np.asarray([names.index(name) for name in requested], dtype=np.int64)
    global_bind = np.asarray(asset.target_bind_global, dtype=np.float64).copy()
    candidate_global = np.asarray(candidate.target_bind_global, dtype=np.float64)
    global_bind[selected] = candidate_global[selected]

    head = np.asarray(asset.target_bone_head, dtype=np.float64).copy()
    tail = np.asarray(asset.target_bone_tail, dtype=np.float64).copy()
    head[selected] = np.asarray(candidate.target_bone_head, dtype=np.float64)[selected]
    tail[selected] = np.asarray(candidate.target_bone_tail, dtype=np.float64)[selected]
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    local_bind = global_bind.copy()
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            local_bind[bone] = np.linalg.inv(global_bind[int(parent)]) @ global_bind[bone]

    metadata = dict(asset.metadata or {})
    history = list(metadata.get("source_rig_rebind", []))
    history.append(
        {
            "stage": str(stage),
            "fitted_bones": int(len(selected)),
            "policy": "selected_global_frames_preserve_unselected_v1",
        }
    )
    metadata["source_rig_rebind"] = history
    metadata["selective_rebind_bones"] = requested
    result = type(asset)(
        **{
            **asset.__dict__,
            "target_rest_global": global_bind.astype(np.float32),
            "target_rest_local": local_bind.astype(np.float32),
            "target_inverse_bind": np.linalg.inv(global_bind).astype(np.float32),
            "target_bone_head": head.astype(np.float32),
            "target_bone_tail": tail.astype(np.float32),
            "metadata": metadata,
        }
    )
    from .anatomy_lbs import with_source_driver_coupling

    result = with_source_driver_coupling(result)
    report = {
        **full_report,
        "stage": str(stage),
        "fitted_bones": int(len(selected)),
        "selected_bones": requested,
        "unselected_global_frames_preserved": True,
    }
    return result, report


def source_bind_roundtrip(asset: AnatomyRiggedAsset) -> dict[str, Any]:
    """Evaluate the exact zero-pose source LBS identity invariant."""
    if asset.target_bind_global is None or asset.runtime_inverse_bind is None:
        return {"source_rig": "legacy_skip"}
    skin = np.asarray(asset.target_bind_global, dtype=np.float64) @ np.asarray(asset.runtime_inverse_bind, dtype=np.float64)
    identity_error = np.max(np.abs(skin - np.eye(4)[None, :, :]))
    return {"max_matrix_error": float(identity_error), "pass": bool(identity_error <= 1.0e-6)}
