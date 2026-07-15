"""Static rest-space head/skull and foot-chain alignment.

Head uses a neck→head axial stretch from config (skull/brain move with the
crown; cervical bones and neck vessels interpolate along the same axis).
Feet align each foot_chain subtree length/rotation to the SMPL-X ankle→foot segment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .anatomy_lbs import joint_global_transforms
from .rigged_asset import AnatomyRiggedAsset


def _has_ancestor(bone: int, ancestor: int, parents: np.ndarray) -> bool:
    current = int(bone)
    visited = 0
    while current >= 0 and visited <= len(parents):
        if current == int(ancestor):
            return True
        current = int(parents[current])
        visited += 1
    return False


def _head_subtree_bones(asset: AnatomyRiggedAsset, head_index: int) -> set[int]:
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    return {
        int(bi)
        for bi in range(len(parents))
        if _has_ancestor(bi, int(head_index), parents) or int(bi) == int(head_index)
    }


def _cervical_bones(asset: AnatomyRiggedAsset) -> set[int]:
    names = list(asset.source_bone_names or [])
    bones: set[int] = set()
    for bi, name in enumerate(names):
        if str(name).startswith("Spine_C") and str(name)[7:].isdigit():
            bones.add(int(bi))
    return bones


def _dominant_source_bone(asset: AnatomyRiggedAsset, start: int, stop: int) -> int | None:
    if asset.driver_indices is None or asset.driver_weights is None or asset.source_bone_names is None:
        return None
    indices = np.asarray(asset.driver_indices[start:stop], dtype=np.int64).reshape(-1)
    weights = np.asarray(asset.driver_weights[start:stop], dtype=np.float64).reshape(-1)
    mass = np.bincount(indices, weights=weights, minlength=len(asset.source_bone_names))
    return int(np.argmax(mass)) if mass.size and float(mass.max()) > 0.0 else None


def _resolve_head_offset(
    asset: AnatomyRiggedAsset, config: dict[str, Any] | None
) -> tuple[np.ndarray, str]:
    """Prefer the static config offset; optionally fall back to a one-shot estimate."""
    cfg = dict(config or {})
    raw = cfg.get("head_rest_offset_m")
    if raw is not None:
        offset = np.asarray(raw, dtype=np.float64).reshape(3)
        return offset, "static_config"
    # Legacy fallback for configs that omit the static offset.
    if asset.source_bone_names is None or "Head_Bone" not in asset.source_bone_names:
        return np.zeros(3, dtype=np.float64), "missing_head_bone"
    joint_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    if "head" not in joint_index:
        return np.zeros(3, dtype=np.float64), "missing_smplx_head"
    rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    smpl_head = rest_global[joint_index["head"], :3, 3]
    skull_centroid = None
    if asset.source_vertex_ranges is not None and asset.source_mesh_names is not None:
        for (start, stop), mesh_name in zip(asset.source_vertex_ranges, asset.source_mesh_names):
            if "skull" not in str(mesh_name).lower():
                continue
            block = np.asarray(asset.vertices_rest[int(start) : int(stop)], dtype=np.float64)
            if len(block) >= 3:
                skull_centroid = block.mean(axis=0)
                break
    if skull_centroid is None:
        return np.zeros(3, dtype=np.float64), "missing_skull_mesh"
    offset = np.zeros(3, dtype=np.float64)
    vertical_gap = float(skull_centroid[1] - smpl_head[1])
    if vertical_gap < -0.002:
        offset[1] = float(-vertical_gap * 0.85)
    return offset, "auto_skull_centroid"


def _axial_stretch_weights(
    points: np.ndarray,
    neck_anchor: np.ndarray,
    head_tip: np.ndarray,
) -> np.ndarray:
    axis = np.asarray(head_tip, dtype=np.float64) - np.asarray(neck_anchor, dtype=np.float64)
    span = float(np.linalg.norm(axis))
    if span < 1.0e-6:
        return np.ones(len(points), dtype=np.float64)
    axis_dir = axis / span
    s = (np.asarray(points, dtype=np.float64) - neck_anchor) @ axis_dir
    return np.clip(s / span, 0.0, 1.0)


def _axial_stretch_points(
    points: np.ndarray,
    neck_anchor: np.ndarray,
    head_tip: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    weights = _axial_stretch_weights(points, neck_anchor, head_tip)
    return np.asarray(points, dtype=np.float64) + weights[:, None] * offset


def _is_head_mesh(mesh_name: str, tissue: str) -> bool:
    lower = str(mesh_name).lower()
    if any(token in lower for token in ("skull", "jaw", "mandible", "maxilla", "brain", "cranium")):
        return True
    if str(tissue) == "bone" and any(token in lower for token in ("head", "teeth", "tooth", "hyoid")):
        return True
    return False


def _is_neck_soft_mesh(mesh_name: str, tissue: str) -> bool:
    lower = str(mesh_name).lower()
    if str(tissue) not in {"vessel", "nerve", "organ", "connective"}:
        return False
    return any(
        token in lower
        for token in (
            "carotid",
            "jugular",
            "vertebral",
            "cervical",
            "nuchal",
            "thyroid",
            "trachea",
            "esophagus",
            "larynx",
            "pharynx",
            "neck",
        )
    )


def _scale_along_axis(
    points: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Scale displacement along ``axis`` about ``pivot`` (perpendicular part unchanged)."""
    if abs(float(scale) - 1.0) < 1.0e-8:
        return np.asarray(points, dtype=np.float64)
    unit = np.asarray(axis, dtype=np.float64)
    unit /= max(float(np.linalg.norm(unit)), 1.0e-12)
    rel = np.asarray(points, dtype=np.float64) - pivot
    along = (rel @ unit)[:, None] * unit
    perp = rel - along
    return pivot + perp + float(scale) * along


def _head_bone_local_z_axis(asset: AnatomyRiggedAsset, head_index: int) -> np.ndarray:
    rest_global = np.asarray(asset.source_rest_global, dtype=np.float64)
    return rest_global[int(head_index), :3, 2].copy()


def _resolve_skull_z_scale(config: dict[str, Any] | None) -> float:
    cfg = dict(config or {})
    return float(cfg.get("head_skull_local_z_scale", 1.0))


def _neck_anchor_and_head_tip(asset: AnatomyRiggedAsset) -> tuple[np.ndarray, np.ndarray]:
    names = list(asset.source_bone_names or [])
    rest_global = np.asarray(asset.source_rest_global, dtype=np.float64)
    if "Spine_C7" in names:
        neck_anchor = rest_global[names.index("Spine_C7"), :3, 3].copy()
    elif "neck" in asset.joint_names:
        neck_anchor = np.asarray(asset.rest_joints[asset.joint_names.index("neck")], dtype=np.float64)
    else:
        neck_anchor = rest_global[names.index("Head_Bone"), :3, 3].copy()
    head_tip = rest_global[names.index("Head_Bone"), :3, 3].copy()
    if asset.source_bone_head is not None:
        head_tip = np.asarray(asset.source_bone_head[names.index("Head_Bone")], dtype=np.float64)
    return neck_anchor, head_tip


def calibrate_head_rest_offset(
    asset: AnatomyRiggedAsset,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Stretch neck→head and optionally compress skull along Head_Bone local Z."""
    if asset.source_bone_names is None or asset.source_rest_global is None:
        return asset, {"applied": False, "reason": "legacy_asset"}
    names = list(asset.source_bone_names)
    if "Head_Bone" not in names:
        return asset, {"applied": False, "reason": "missing_head_bone"}
    head_index = int(names.index("Head_Bone"))
    offset, source = _resolve_head_offset(asset, config)
    z_scale = _resolve_skull_z_scale(config)
    if float(np.linalg.norm(offset)) < 1.0e-4 and abs(z_scale - 1.0) < 1.0e-4:
        return asset, {
            "applied": False,
            "reason": "within_tolerance",
            "offset_m": [0.0, 0.0, 0.0],
            "offset_source": source,
            "skull_local_z_scale": z_scale,
        }

    neck_anchor, head_tip = _neck_anchor_and_head_tip(asset)
    head_bones = _head_subtree_bones(asset, head_index)
    cervical_bones = _cervical_bones(asset)
    stretch_bones = set(head_bones) | cervical_bones

    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    mask = np.zeros(len(vertices), dtype=bool)
    skull_mask = np.zeros(len(vertices), dtype=bool)
    if asset.source_vertex_ranges is not None and asset.source_mesh_names is not None:
        tissues = list(asset.source_tissues or ["bone"] * len(asset.source_mesh_names))
        for (start, stop), mesh_name, tissue in zip(
            np.asarray(asset.source_vertex_ranges, dtype=np.int64),
            asset.source_mesh_names,
            tissues,
        ):
            start_i, stop_i = int(start), int(stop)
            block = vertices[start_i:stop_i]
            bone = _dominant_source_bone(asset, start_i, stop_i)
            if _is_head_mesh(str(mesh_name), str(tissue)):
                if bone is not None and int(bone) in head_bones:
                    if float(np.linalg.norm(offset)) >= 1.0e-4:
                        block = _axial_stretch_points(block, neck_anchor, head_tip, offset)
                    vertices[start_i:stop_i] = block
                    mask[start_i:stop_i] = True
                    skull_mask[start_i:stop_i] = True
                continue
            if str(tissue) == "bone" and bone is not None and int(bone) in cervical_bones:
                if float(np.linalg.norm(offset)) >= 1.0e-4:
                    vertices[start_i:stop_i] = _axial_stretch_points(block, neck_anchor, head_tip, offset)
                    mask[start_i:stop_i] = True
                continue
            if _is_neck_soft_mesh(str(mesh_name), str(tissue)) and float(np.linalg.norm(offset)) >= 1.0e-4:
                weights = _axial_stretch_weights(block, neck_anchor, head_tip)
                if float(weights.max()) > 1.0e-6:
                    vertices[start_i:stop_i] = block + weights[:, None] * offset
                    mask[start_i:stop_i] = True

    if not np.any(mask) and abs(z_scale - 1.0) < 1.0e-4:
        return asset, {
            "applied": False,
            "reason": "no_head_meshes",
            "offset_m": offset.tolist(),
            "offset_source": source,
            "skull_local_z_scale": z_scale,
        }

    rest_global = np.asarray(asset.source_rest_global, dtype=np.float64).copy()
    if float(np.linalg.norm(offset)) >= 1.0e-4:
        for bi in stretch_bones:
            point = rest_global[int(bi), :3, 3]
            if int(bi) in head_bones:
                weight = 1.0
            else:
                weight = float(_axial_stretch_weights(point.reshape(1, 3), neck_anchor, head_tip)[0])
            rest_global[int(bi), :3, 3] = point + weight * offset

    bone_head = (
        np.asarray(asset.source_bone_head, dtype=np.float64).copy()
        if asset.source_bone_head is not None
        else None
    )
    bone_tail = (
        np.asarray(asset.source_bone_tail, dtype=np.float64).copy()
        if asset.source_bone_tail is not None
        else None
    )
    if float(np.linalg.norm(offset)) >= 1.0e-4:
        for field_name, points in (("head", bone_head), ("tail", bone_tail)):
            if points is None:
                continue
            for bi in stretch_bones:
                if int(bi) in head_bones:
                    weight = 1.0
                else:
                    weight = float(
                        _axial_stretch_weights(points[int(bi)].reshape(1, 3), neck_anchor, head_tip)[0]
                    )
                points[int(bi)] += weight * offset

    skull_z_axis = _head_bone_local_z_axis(asset, head_index)
    if bone_head is not None:
        skull_pivot = bone_head[int(head_index)].copy()
    else:
        skull_pivot = rest_global[int(head_index), :3, 3].copy()

    if abs(z_scale - 1.0) >= 1.0e-4:
        if np.any(skull_mask):
            vertices[skull_mask] = _scale_along_axis(
                vertices[skull_mask], skull_pivot, skull_z_axis, z_scale
            )
        for bi in head_bones:
            rest_global[int(bi), :3, 3] = _scale_along_axis(
                rest_global[int(bi), :3, 3].reshape(1, 3),
                skull_pivot,
                skull_z_axis,
                z_scale,
            )[0]
            if bone_head is not None:
                bone_head[int(bi)] = _scale_along_axis(
                    bone_head[int(bi)].reshape(1, 3), skull_pivot, skull_z_axis, z_scale
                )[0]
            if bone_tail is not None:
                bone_tail[int(bi)] = _scale_along_axis(
                    bone_tail[int(bi)].reshape(1, 3), skull_pivot, skull_z_axis, z_scale
                )[0]

    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    rest_local = rest_global.copy()
    for bi, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            rest_local[bi] = np.linalg.inv(rest_global[int(parent)]) @ rest_global[bi]

    updates: dict[str, Any] = {
        "vertices_rest": vertices.astype(np.float32),
        "source_rest_global": rest_global.astype(np.float32),
        "source_rest_local": rest_local.astype(np.float32),
        "source_inverse_bind": np.linalg.inv(rest_global).astype(np.float32),
    }
    if asset.registration_reference is not None:
        reference = np.asarray(asset.registration_reference, dtype=np.float64).copy()
        if np.any(mask):
            reference[mask] = vertices[mask]
        updates["registration_reference"] = reference.astype(np.float32)
    if bone_head is not None:
        updates["source_bone_head"] = bone_head.astype(np.float32)
    if bone_tail is not None:
        updates["source_bone_tail"] = bone_tail.astype(np.float32)

    meta = dict(asset.metadata or {})
    meta["head_rest_calibration"] = {
        "offset_m": [float(v) for v in offset.tolist()],
        "offset_source": source,
        "mode": "neck_axial_stretch_plus_skull_z_scale",
        "skull_local_z_scale": float(z_scale),
        "neck_anchor_m": [float(v) for v in neck_anchor.tolist()],
        "head_tip_m": [float(v) for v in head_tip.tolist()],
        "head_bones": int(len(head_bones)),
        "cervical_bones": int(len(cervical_bones)),
        "vertex_count": int(np.count_nonzero(mask)),
        "skull_vertex_count": int(np.count_nonzero(skull_mask)),
    }
    updates["metadata"] = meta
    return type(asset)(**{**asset.__dict__, **updates}), dict(meta["head_rest_calibration"], applied=True)


def _foot_chain_root(asset: AnatomyRiggedAsset, side: str) -> int | None:
    names = list(asset.source_bone_names or [])
    types = list(asset.source_bone_driver_types or [])
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    target = f"foot_chain_{side}"
    members = [i for i, t in enumerate(types) if str(t) == target]
    if not members:
        return None
    for bi in members:
        parent = int(parents[bi])
        if parent < 0 or str(types[parent]) != target:
            return int(bi)
    # Prefer the authored Ankle_Rot control when present.
    for name in (f"Ankle_Rot_{'L' if side == 'left' else 'R'}",):
        if name in names:
            return int(names.index(name))
    return int(members[0])


def _foot_subtree_bones(asset: AnatomyRiggedAsset, root: int) -> set[int]:
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    types = list(asset.source_bone_driver_types or [])
    target = str(types[root])
    bones = {int(root)}
    changed = True
    while changed:
        changed = False
        for bi, parent in enumerate(parents.tolist()):
            if int(bi) in bones:
                continue
            if int(parent) not in bones:
                continue
            own = str(types[bi]) if bi < len(types) else ""
            if own == target or own == "parent_follow":
                bones.add(int(bi))
                changed = True
    return bones


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation that maps unit vector ``a`` onto unit vector ``b``."""
    ua = np.asarray(a, dtype=np.float64)
    ub = np.asarray(b, dtype=np.float64)
    ua /= max(float(np.linalg.norm(ua)), 1.0e-12)
    ub /= max(float(np.linalg.norm(ub)), 1.0e-12)
    cross = np.cross(ua, ub)
    cos = float(np.clip(ua @ ub, -1.0, 1.0))
    if float(np.linalg.norm(cross)) < 1.0e-8:
        if cos > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(ua)))]
        axis = axis - ua * float(axis @ ua)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - cos) / max(float(cross @ cross), 1.0e-12))


def calibrate_foot_rest_alignment(asset: AnatomyRiggedAsset) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Rotate each foot_chain so Blender ankle→tip matches SMPL-X ankle→foot direction.

    SMPL-X ``foot`` is a mid-foot joint, not the toe tip, so we never scale the
    authored Blender foot length down to that short segment.
    """
    if asset.source_bone_names is None or asset.source_rest_global is None:
        return asset, {"applied": False, "reason": "legacy_asset"}
    joint_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    rest_joints = np.asarray(asset.rest_joints, dtype=np.float64)
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    rest_global = np.asarray(asset.source_rest_global, dtype=np.float64).copy()
    bone_head = (
        np.asarray(asset.source_bone_head, dtype=np.float64).copy()
        if asset.source_bone_head is not None
        else None
    )
    bone_tail = (
        np.asarray(asset.source_bone_tail, dtype=np.float64).copy()
        if asset.source_bone_tail is not None
        else None
    )
    reference = (
        np.asarray(asset.registration_reference, dtype=np.float64).copy()
        if asset.registration_reference is not None
        else None
    )
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    report_sides: dict[str, Any] = {}
    touched = 0

    for side in ("left", "right"):
        ankle_name = f"{side}_ankle"
        foot_name = f"{side}_foot"
        if ankle_name not in joint_index or foot_name not in joint_index:
            continue
        root = _foot_chain_root(asset, side)
        if root is None:
            continue
        bones = _foot_subtree_bones(asset, root)
        smpl_ankle = rest_joints[joint_index[ankle_name]]
        smpl_foot = rest_joints[joint_index[foot_name]]
        smpl_vec = smpl_foot - smpl_ankle
        smpl_len = float(np.linalg.norm(smpl_vec))
        if smpl_len < 1.0e-4:
            continue

        root_pos = rest_global[root, :3, 3]
        # Prefer Arch_Rot as the direction probe: it sits on the ankle→toe axis
        # without being as noisy as an anonymous distal phalanx helper.
        names = list(asset.source_bone_names)
        arch_name = f"Arch_Rot_{'L' if side == 'left' else 'R'}"
        if arch_name in names and int(names.index(arch_name)) in bones:
            tip = rest_global[int(names.index(arch_name)), :3, 3]
        else:
            tip = root_pos.copy()
            tip_dist = 0.0
            for bi in bones:
                point = rest_global[int(bi), :3, 3]
                if bone_tail is not None:
                    point = bone_tail[int(bi)]
                dist = float(np.linalg.norm(point - root_pos))
                if dist > tip_dist:
                    tip_dist = dist
                    tip = point
        blender_vec = tip - root_pos
        blender_len = float(np.linalg.norm(blender_vec))
        if blender_len < 1.0e-4:
            continue

        rotation = _rotation_between(blender_vec, smpl_vec)
        angle = float(np.degrees(np.arccos(np.clip(
            (blender_vec / blender_len) @ (smpl_vec / smpl_len), -1.0, 1.0
        ))))
        # Skip near-identity corrections so we do not churn rest frames.
        if angle < 1.0 and float(np.linalg.norm(root_pos - smpl_ankle)) < 0.005:
            report_sides[side] = {
                "root_bone": str(asset.source_bone_names[root]),
                "bones": int(len(bones)),
                "applied": False,
                "reason": "within_tolerance",
                "rotation_deg": angle,
            }
            continue

        def _map_point(point: np.ndarray) -> np.ndarray:
            return smpl_ankle + rotation @ (point - root_pos)

        for bi in bones:
            rest_global[int(bi), :3, :3] = rotation @ rest_global[int(bi), :3, :3]
            rest_global[int(bi), :3, 3] = _map_point(rest_global[int(bi), :3, 3])
            if bone_head is not None:
                bone_head[int(bi)] = _map_point(bone_head[int(bi)])
            if bone_tail is not None:
                bone_tail[int(bi)] = _map_point(bone_tail[int(bi)])

        if asset.source_vertex_ranges is not None:
            for (start, stop), tissue in zip(
                np.asarray(asset.source_vertex_ranges, dtype=np.int64),
                asset.source_tissues or [],
            ):
                # Only rigid bone meshes follow the foot rest alignment. Soft
                # vessels/nerves stay with the volume registration.
                if str(tissue) != "bone":
                    continue
                bone = _dominant_source_bone(asset, int(start), int(stop))
                if bone is None or int(bone) not in bones:
                    continue
                block = vertices[int(start) : int(stop)]
                vertices[int(start) : int(stop)] = smpl_ankle + (rotation @ (block - root_pos).T).T
                if reference is not None:
                    ref_block = reference[int(start) : int(stop)]
                    reference[int(start) : int(stop)] = (
                        smpl_ankle + (rotation @ (ref_block - root_pos).T).T
                    )
                touched += int(stop - start)

        report_sides[side] = {
            "root_bone": str(asset.source_bone_names[root]),
            "bones": int(len(bones)),
            "applied": True,
            "blender_length_m": blender_len,
            "smplx_length_m": smpl_len,
            "rotation_deg": angle,
            "ankle_shift_m": float(np.linalg.norm(root_pos - smpl_ankle)),
        }

    if not report_sides:
        return asset, {"applied": False, "reason": "no_foot_chains"}
    if not any(bool(v.get("applied")) for v in report_sides.values()):
        return asset, {"applied": False, "reason": "within_tolerance", "sides": report_sides}

    rest_local = rest_global.copy()
    for bi, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            rest_local[bi] = np.linalg.inv(rest_global[int(parent)]) @ rest_global[bi]

    updates: dict[str, Any] = {
        "vertices_rest": vertices.astype(np.float32),
        "source_rest_global": rest_global.astype(np.float32),
        "source_rest_local": rest_local.astype(np.float32),
        "source_inverse_bind": np.linalg.inv(rest_global).astype(np.float32),
    }
    if bone_head is not None:
        updates["source_bone_head"] = bone_head.astype(np.float32)
    if bone_tail is not None:
        updates["source_bone_tail"] = bone_tail.astype(np.float32)
    if reference is not None:
        updates["registration_reference"] = reference.astype(np.float32)
    meta = dict(asset.metadata or {})
    meta["foot_rest_calibration"] = {"sides": report_sides, "vertex_count": int(touched)}
    updates["metadata"] = meta
    return type(asset)(**{**asset.__dict__, **updates}), dict(meta["foot_rest_calibration"], applied=True)
