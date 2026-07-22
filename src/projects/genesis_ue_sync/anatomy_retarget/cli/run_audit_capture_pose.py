"""Audit one live Stage-1 anatomy drive against a fitted SMPL-X capture.

This command is deliberately an *offline inspection* tool.  It accepts a
published schema-6 ``anatomy_rigged.npz`` and a terminal-8
``smplx_result.npz``, evaluates the persisted source-rig coupling directly,
and writes artifacts that can be opened without Blender:

``objs/``
    Posed/full anatomy, fitted SMPL-X shell, bone and vessel subsets, plus
    SMPL-X/source-rig line skeletons.
``overlays/``
    Full body and targeted head, elbow, hand, knee and shoulder overlays.
``capture_audit.json``
    Runtime-contract, containment-sample and source-rig health evidence.

No Blender process is started and no per-pose cache is read.  Consequently a
successful audit proves that this particular capture was evaluated through the
same reusable Stage-1 source rig that a new SMPL-X pose will use at runtime.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    joint_global_transforms,
    skin_vertices,
    source_bone_skinning_transforms,
)
from projects.genesis_ue_sync.anatomy_retarget.bone_segment_diagnostics import (
    write_bone_segment_diagnostics,
)
from projects.genesis_ue_sync.anatomy_retarget.obj_io import write_obj
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_drive_translation,
    easymocap_fit_to_smplx55,
    smplx_pose_hash,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset, load_rigged_asset
from projects.genesis_ue_sync.anatomy_retarget.stage1_contract import stage1_runtime_contract


DEFAULT_ASSET = Path("outputs/anatomy_retarget/latest_asset/anatomy_rigged.npz")
DEFAULT_OUTPUT = Path("outputs/anatomy_retarget/capture_audit")

TISSUE_COLORS: dict[str, str] = {
    "bone": "#e8e2c7",
    "vessel": "#d03b52",
    "nerve": "#a45cc5",
    "organ": "#4aa59d",
    "heart": "#d13e3e",
    "connective_tissue": "#76a46f",
    "unknown": "#cc6c4b",
}

REGION_TOKENS: dict[str, tuple[str, ...]] = {
    "head_neck": (
        "skull", "cranium", "brain", "cerebr", "cerebell", "ear", "neck", "cerv",
        "hyoid", "mandible", "jaw", "teeth", "atlas", "axis", "face", "eye", "tongue",
        "carotid", "jugular",
    ),
    "shoulder_girdle": (
        "clavicle", "scapula", "shoulder", "humerus", "subclavian", "axillary",
    ),
    "left_elbow": (
        "humerus", "radius", "ulna", "elbow", "forearm", "brachial", "basilic", "cephalic",
    ),
    "right_elbow": (
        "humerus", "radius", "ulna", "elbow", "forearm", "brachial", "basilic", "cephalic",
    ),
    "left_hand": (
        "hand", "finger", "phalan", "metacarp", "carpal", "wrist", "thumb", "index", "middle",
        "ring", "pinky", "palmar", "digital", "radial", "ulnar",
    ),
    "right_hand": (
        "hand", "finger", "phalan", "metacarp", "carpal", "wrist", "thumb", "index", "middle",
        "ring", "pinky", "palmar", "digital", "radial", "ulnar",
    ),
    "left_hip": (
        "ilium", "sacrum", "pelvis", "femur", "hip", "sciatic", "femoral",
    ),
    "right_hip": (
        "ilium", "sacrum", "pelvis", "femur", "hip", "sciatic", "femoral",
    ),
    "left_knee": (
        "femur", "tibia", "fibula", "patella", "knee", "popliteal", "saphen", "peroneal",
    ),
    "right_knee": (
        "femur", "tibia", "fibula", "patella", "knee", "popliteal", "saphen", "peroneal",
    ),
    "left_ankle": (
        "tibia", "fibula", "talus", "calcaneus", "ankle", "peroneal", "tibial",
    ),
    "right_ankle": (
        "tibia", "fibula", "talus", "calcaneus", "ankle", "peroneal", "tibial",
    ),
    "left_foot": (
        "foot", "talus", "calcaneus", "navicular", "cuboid", "cuneiform", "metatarsal", "phalanx",
    ),
    "right_foot": (
        "foot", "talus", "calcaneus", "navicular", "cuboid", "cuneiform", "metatarsal", "phalanx",
    ),
}

# Stage 1 has to preserve a reusable runtime rig *and* keep the transferred
# anatomy inside the fixed subject body for representative poses.  These are
# deliberately stricter than a visual-warning threshold: an escaped hand,
# knee, or head must stop a candidate from being called a Stage-1 success.
STAGE1_POSE_ACCEPTANCE_LIMITS = {
    "inside_fraction": 0.995,
    "max_outside_m": 0.002,
}

REGION_RADII_M = {
    "head_neck": 0.18,
    "shoulder_girdle": 0.12,
    "left_elbow": 0.10,
    "right_elbow": 0.10,
    "left_hand": 0.10,
    "right_hand": 0.10,
    "left_hip": 0.14,
    "right_hip": 0.14,
    "left_knee": 0.12,
    "right_knee": 0.12,
    "left_ankle": 0.11,
    "right_ankle": 0.11,
    "left_foot": 0.13,
    "right_foot": 0.13,
}


def stage1_pose_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    """Return the publication verdict for one pose-specific Stage-1 audit."""
    failures: list[str] = []
    runtime = report.get("stage1_runtime_contract", {})
    if not bool(runtime.get("passed")):
        failures.append("runtime contract failed")
    if not bool(report.get("anatomy_finite")):
        failures.append("posed anatomy contains non-finite vertices")

    required_regions = tuple(REGION_TOKENS)
    for label, containment in (
        ("rest", report.get("rest_containment", {})),
        ("posed", report.get("containment", {})),
    ):
        regions = containment.get("regions", {}) if isinstance(containment, dict) else {}
        for region in required_regions:
            metric = regions.get(region)
            if not isinstance(metric, dict) or int(metric.get("sample_count", 0)) <= 0:
                failures.append(f"{label}/{region}: containment samples missing")
                continue
            inside = float(metric.get("inside_fraction", 0.0))
            outside = float(metric.get("max_outside_m", float("inf")))
            if inside < STAGE1_POSE_ACCEPTANCE_LIMITS["inside_fraction"]:
                failures.append(
                    f"{label}/{region}: inside_fraction={inside:.4f} < "
                    f"{STAGE1_POSE_ACCEPTANCE_LIMITS['inside_fraction']:.4f}"
                )
            if outside > STAGE1_POSE_ACCEPTANCE_LIMITS["max_outside_m"]:
                failures.append(
                    f"{label}/{region}: max_outside_m={outside:.6f} > "
                    f"{STAGE1_POSE_ACCEPTANCE_LIMITS['max_outside_m']:.6f}"
                )

    chain = report.get("bone_chain_diagnostics", {})
    if isinstance(chain, dict) and chain.get("available") and not chain.get("passed"):
        failures.append(f"bone-chain diagnostics failed ({int(chain.get('failure_count', 0))} findings)")
    return {
        "limits": dict(STAGE1_POSE_ACCEPTANCE_LIMITS),
        "required_regions": list(required_regions),
        "passed": not failures,
        "failures": failures,
    }


def _as_vec3(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be three finite values")
    return array.reshape(3)


def _finite_mesh(vertices: np.ndarray, faces: np.ndarray, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if not len(verts) or not len(tris):
        raise ValueError(f"{label} must contain vertices and triangle faces")
    if not np.all(np.isfinite(verts)):
        raise ValueError(f"{label} contains non-finite vertices")
    if int(tris.min()) < 0 or int(tris.max()) >= len(verts):
        raise ValueError(f"{label} has a face index outside its vertices")
    return verts, tris


def _load_motion(
    motion_npz: Path,
    *,
    gender: str,
    smplx_model: Path | None,
) -> dict[str, Any]:
    with np.load(motion_npz, allow_pickle=False) as data:
        required = {"Rh", "Th", "poses", "vertices", "faces"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{motion_npz} is not a SMPL-X result; missing {missing}")
        rh = _as_vec3(data["Rh"], name="Rh")
        th = _as_vec3(data["Th"], name="Th")
        poses = np.asarray(data["poses"], dtype=np.float32).reshape(-1)
        fit_vertices, fit_faces = _finite_mesh(data["vertices"], data["faces"], label="SMPL-X fit")
        root_align = (
            _as_vec3(data["root_align_offset"], name="root_align_offset")
            if "root_align_offset" in data.files
            else np.zeros(3, dtype=np.float32)
        )
        betas = np.asarray(data["shapes"], dtype=np.float32).reshape(-1) if "shapes" in data.files else None

    # A 165-D result does not need a SMPL-X PKL.  87-D EasyMocap hand PCA does,
    # so preserve the requested gender/model rather than silently substituting
    # a different hand basis.
    pose55 = easymocap_fit_to_smplx55(
        rh,
        poses,
        gender=gender,
        model_path=smplx_model,
    )
    return {
        "Rh": rh,
        "Th": th,
        "pose55": np.asarray(pose55, dtype=np.float32).reshape(55, 3),
        "fit_vertices": fit_vertices,
        "fit_faces": fit_faces,
        "root_align_offset": root_align,
        "betas": betas,
        "source_pose_dimensions": int(poses.size),
    }


def _read_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertex and polygon topology needed for a canonical skin overlay."""
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                values = line.split()
                if len(values) >= 4:
                    vertices.append([float(values[1]), float(values[2]), float(values[3])])
            elif line.startswith("f "):
                values = line.split()[1:]
                ids: list[int] = []
                for value in values:
                    token = value.split("/", 1)[0]
                    if not token:
                        continue
                    index = int(token)
                    ids.append(index - 1 if index > 0 else len(vertices) + index)
                for index in range(1, len(ids) - 1):
                    triangles.append([ids[0], ids[index], ids[index + 1]])
    if not vertices:
        raise ValueError(f"canonical SMPL-X OBJ {path} has no vertices")
    vert_array = np.asarray(vertices, dtype=np.float32)
    face_array = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    if len(face_array):
        _finite_mesh(vert_array, face_array, label="canonical SMPL-X OBJ")
    return vert_array, face_array


def _canonical_reference(canonical_dir: Path | None) -> dict[str, Any] | None:
    """Load the fixed subject body used by Stage-1, optionally with LBS data.

    The canonical mesh defines the body shape the anatomy asset was fitted to.
    The companion weights file lets the audit pose that same body for every
    capture, which avoids treating a changed capture beta as an anatomy error.
    """
    if canonical_dir is None:
        return None
    location = Path(canonical_dir)
    candidate = location if location.suffix.lower() == ".obj" else location / "smpl_canonical_tpose.obj"
    if not candidate.is_file():
        return None
    vertices, faces = _read_obj_mesh(candidate)
    if not len(faces):
        return None
    root = candidate.parent
    result: dict[str, Any] = {
        "vertices": vertices,
        "faces": faces,
        "object_path": candidate,
        "weights": None,
        "manifest_betas": None,
        "manifest_path": None,
    }
    manifest_path = root / "source_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            beta_values = manifest.get("betas")
            if beta_values is not None:
                result["manifest_betas"] = np.asarray(beta_values, dtype=np.float32).reshape(-1)[:10]
            result["manifest_path"] = manifest_path
        except (json.JSONDecodeError, OSError, ValueError):
            # The rest OBJ is still useful if an older cache has a malformed
            # manifest; do not substitute a beta from an unrelated capture.
            pass
    weights_path = root / "smpl_canonical_weights.npz"
    if not weights_path.is_file():
        return result
    with np.load(weights_path, allow_pickle=False) as payload:
        required = {"lbs_weights", "faces", "rest_joints", "parents", "inverse_bind"}
        if not required.issubset(payload.files):
            return result
        weights = np.asarray(payload["lbs_weights"], dtype=np.float32)
        weight_faces = np.asarray(payload["faces"], dtype=np.int32)
        rest_joints = np.asarray(payload["rest_joints"], dtype=np.float32)
        parents = np.asarray(payload["parents"], dtype=np.int32).reshape(-1)
        inverse_bind = np.asarray(payload["inverse_bind"], dtype=np.float32)
    try:
        _finite_mesh(vertices, weight_faces, label="canonical SMPL-X weights")
    except ValueError:
        return result
    joint_count = int(weights.shape[1]) if weights.ndim == 2 else 0
    if (
        weights.shape != (len(vertices), joint_count)
        or rest_joints.shape != (joint_count, 3)
        or parents.shape != (joint_count,)
        or inverse_bind.shape != (joint_count, 4, 4)
        or not np.all(np.isfinite(weights))
    ):
        return result
    result["faces"] = weight_faces
    result["weights"] = {
        "path": weights_path,
        "lbs_weights": weights,
        "rest_joints": rest_joints,
        "parents": parents,
        "inverse_bind": inverse_bind,
    }
    return result


def _skin_canonical_reference(
    canonical: dict[str, Any],
    *,
    pose55: np.ndarray,
    rh: np.ndarray,
    th: np.ndarray,
    extra_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose the fixed canonical SMPL-X body through its own persisted LBS."""
    payload = canonical.get("weights")
    if not isinstance(payload, dict):
        raise ValueError("canonical reference does not include SMPL-X LBS weights")
    rest_joints = np.asarray(payload["rest_joints"], dtype=np.float32)
    parents = np.asarray(payload["parents"], dtype=np.int32)
    inverse_bind = np.asarray(payload["inverse_bind"], dtype=np.float32)
    weights = np.asarray(payload["lbs_weights"], dtype=np.float32)
    global_transforms = joint_global_transforms(
        pose_axis_angle=pose55,
        rest_joints=rest_joints,
        parents=parents,
    )
    transforms = np.matmul(global_transforms, inverse_bind)
    blended = np.matmul(weights, transforms.reshape(transforms.shape[0], 16)).reshape(-1, 4, 4)
    vertices = np.asarray(canonical["vertices"], dtype=np.float32)
    homogeneous = np.concatenate((vertices, np.ones((len(vertices), 1), dtype=np.float32)), axis=1)
    posed = np.matmul(blended, homogeneous[:, :, None])[:, :3, 0].astype(np.float32)
    translation = easymocap_drive_translation(rh, th, rest_joints[0]) + np.asarray(extra_translation, dtype=np.float32)
    return posed + translation.reshape(1, 3), translation


def _mesh_tissues(asset: AnatomyRiggedAsset) -> list[str]:
    tissue_values = list(asset.source_tissues or [])
    if len(tissue_values) != len(asset.source_mesh_names):
        return ["unknown"] * len(asset.source_mesh_names)
    return [str(value).strip().lower() or "unknown" for value in tissue_values]


def _mesh_sides(asset: AnatomyRiggedAsset) -> list[str]:
    values = list(asset.source_sides or [])
    if len(values) != len(asset.source_mesh_names):
        return [""] * len(asset.source_mesh_names)
    return [str(value).strip().lower() for value in values]


def _mesh_mask(asset: AnatomyRiggedAsset, mesh_indices: Iterable[int]) -> np.ndarray:
    mask = np.zeros(len(asset.vertices_rest), dtype=bool)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    for mesh_index in mesh_indices:
        start, stop = ranges[int(mesh_index)]
        mask[int(start) : int(stop)] = True
    return mask


def _tissue_mask(asset: AnatomyRiggedAsset, tissues: set[str]) -> np.ndarray:
    mesh_indices = [
        index
        for index, tissue in enumerate(_mesh_tissues(asset))
        if tissue in tissues
    ]
    return _mesh_mask(asset, mesh_indices)


def _side_matches(name: str, semantic_side: str, requested_side: str) -> bool:
    if semantic_side in {"left", "right"}:
        return semantic_side == requested_side
    parts = re.findall(r"[a-z]+|[0-9]+", str(name).lower())
    markers = {"l", "left"} if requested_side == "left" else {"r", "right"}
    return bool(markers.intersection(parts))


_EXACT_REGION_TOKENS = {"ear", "face", "hand", "foot", "ring", "middle", "index"}


def _region_name_matches(name: str, region: str) -> bool:
    lower = str(name).lower()
    parts = re.findall(r"[a-z]+|[0-9]+", lower)
    if region.endswith("_hand") and "foot" in parts:
        return False
    for token in REGION_TOKENS[region]:
        if token in _EXACT_REGION_TOKENS:
            if token in parts:
                return True
        elif token in lower:
            return True
    return False


def _region_joint_names(asset: AnatomyRiggedAsset, region: str) -> list[str]:
    available = set(asset.joint_names)
    if region == "head_neck":
        requested = ["neck", "head", "jaw", "left_eye_smplhf", "right_eye_smplhf"]
    elif region == "shoulder_girdle":
        requested = ["spine3", "left_collar", "right_collar", "left_shoulder", "right_shoulder"]
    elif region.endswith("_elbow"):
        requested = [region]
    elif region.endswith("_knee"):
        requested = [region]
    elif region.endswith("_hip"):
        requested = [region]
    elif region.endswith("_ankle"):
        requested = [region]
    elif region.endswith("_foot"):
        side = region.split("_", 1)[0]
        requested = [
            f"{side}_ankle",
            f"{side}_foot",
            f"{side}_big_toe",
            f"{side}_small_toe",
        ]
    elif region.endswith("_hand"):
        side = region.split("_", 1)[0]
        requested = [f"{side}_wrist"] + [
            f"{side}_{finger}{level}"
            for finger in ("thumb", "index", "middle", "ring", "pinky")
            for level in (1, 2, 3)
        ]
    else:
        requested = []
    return [name for name in requested if name in available]


def _region_material_mask(asset: AnatomyRiggedAsset, region: str) -> np.ndarray:
    names = _region_joint_names(asset, region)
    if not names:
        return np.zeros(len(asset.vertices_rest), dtype=bool)
    ids = [asset.joint_names.index(name) for name in names]
    anchors = np.asarray(asset.rest_joints, dtype=np.float64)[ids]
    points = np.asarray(asset.vertices_rest, dtype=np.float64)
    distance = np.min(np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=2), axis=1)
    mask = distance <= float(REGION_RADII_M[region])
    if np.any(mask):
        return mask
    # Tiny synthetic fixtures may have no vertex within a real anatomical
    # radius.  Preserve their diagnostic coverage without changing production
    # assets, where the spatial material domain is authoritative.
    requested_side = (
        "left" if region.startswith("left_") else "right" if region.startswith("right_") else ""
    )
    selected = [
        index
        for index, (mesh_name, side) in enumerate(
            zip(asset.source_mesh_names, _mesh_sides(asset), strict=True)
        )
        if _region_name_matches(str(mesh_name), region)
        and (not requested_side or _side_matches(str(mesh_name), side, requested_side))
    ]
    return _mesh_mask(asset, selected)


def _region_mask(asset: AnatomyRiggedAsset, region: str) -> np.ndarray:
    return _region_material_mask(asset, region)


def _subset_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray,
    *,
    comment: str,
) -> dict[str, int]:
    """Write a self-contained mesh subset while preserving only local face indices."""
    selected_faces = np.asarray(faces, dtype=np.int32)[np.all(mask[np.asarray(faces, dtype=np.int64)], axis=1)]
    if not len(selected_faces):
        write_obj(path, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), comment=comment)
        return {"vertices": 0, "faces": 0}
    used = np.unique(selected_faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    write_obj(path, np.asarray(vertices, dtype=np.float32)[used], remap[selected_faces], comment=comment)
    return {"vertices": int(len(used)), "faces": int(len(selected_faces))}


def _write_line_skeleton(
    path: Path,
    *,
    heads: np.ndarray,
    tails: np.ndarray,
    names: Iterable[str],
    comment: str,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {comment}\n")
        vertex_index = 1
        for head, tail, name in zip(heads, tails, names, strict=True):
            handle.write(f"o {str(name).replace(' ', '_')}\n")
            handle.write(f"v {float(head[0]):.6f} {float(head[1]):.6f} {float(head[2]):.6f}\n")
            handle.write(f"v {float(tail[0]):.6f} {float(tail[1]):.6f} {float(tail[2]):.6f}\n")
            handle.write(f"l {vertex_index} {vertex_index + 1}\n")
            vertex_index += 2
            count += 1
    return count


def _write_joint_skeleton(
    path: Path,
    *,
    joints: np.ndarray,
    parents: np.ndarray,
    names: Iterable[str],
    comment: str,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(joints, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {comment}\n")
        for name, point in zip(names, points, strict=True):
            handle.write(f"o {str(name).replace(' ', '_')}\n")
            handle.write(f"v {float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f}\n")
        for child, parent in enumerate(np.asarray(parents, dtype=np.int32).tolist()):
            if child and int(parent) >= 0:
                handle.write(f"l {int(parent) + 1} {child + 1}\n")
    return int(len(points))


def _sample_points(points: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, num=max(1, int(maximum)), dtype=np.int64)
    return values[indices]


def _fit_axes(axis: Any, point_sets: Iterable[np.ndarray], first: int, second: int) -> None:
    usable = [np.asarray(points, dtype=np.float32).reshape(-1, 3) for points in point_sets if len(points)]
    if not usable:
        return
    joined = np.concatenate(usable, axis=0)
    low = np.min(joined[:, (first, second)], axis=0)
    high = np.max(joined[:, (first, second)], axis=0)
    radius = max(float(np.max(high - low)) * 0.54, 0.025)
    center = 0.5 * (low + high)
    axis.set_xlim(float(center[0] - radius), float(center[0] + radius))
    axis.set_ylim(float(center[1] - radius), float(center[1] + radius))


def _draw_overlay(
    path: Path,
    *,
    title: str,
    smpl_vertices: np.ndarray,
    anatomy_groups: dict[str, np.ndarray],
    point_limit: int = 18000,
    region_points: np.ndarray | None = None,
) -> None:
    """Draw deterministic orthographic overlays; SMPL-X always remains visible."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    path.parent.mkdir(parents=True, exist_ok=True)
    body = np.asarray(smpl_vertices, dtype=np.float32).reshape(-1, 3)
    if region_points is not None and len(region_points):
        focus = np.asarray(region_points, dtype=np.float32).reshape(-1, 3)
        low = focus.min(axis=0) - 0.035
        high = focus.max(axis=0) + 0.035
        crop = body[np.all((body >= low) & (body <= high), axis=1)]
        if len(crop):
            body = crop
    body = _sample_points(body, point_limit)
    groups = {
        tissue: _sample_points(points, max(500, point_limit // max(1, len(anatomy_groups))))
        for tissue, points in anatomy_groups.items()
        if len(points)
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    views = ((0, 1, "front"), (2, 1, "side"), (0, 2, "top"))
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d0a000", markersize=5, label="SMPL-X skin"),
    ]
    for tissue in groups:
        handles.append(
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=TISSUE_COLORS.get(tissue, TISSUE_COLORS["unknown"]),
                markersize=5, label=tissue.replace("_", " "),
            )
        )
    for axis, (first, second, view_name) in zip(axes, views, strict=True):
        axis.scatter(body[:, first], body[:, second], s=0.55, c="#d0a000", alpha=0.24, rasterized=True)
        for tissue, points in groups.items():
            axis.scatter(
                points[:, first], points[:, second], s=1.1,
                c=TISSUE_COLORS.get(tissue, TISSUE_COLORS["unknown"]), alpha=0.72, rasterized=True,
            )
        axis.set_title(view_name)
        axis.set_aspect("equal")
        _fit_axes(axis, [body, *groups.values()], first, second)
        axis.grid(alpha=0.12)
    figure.suptitle(title)
    figure.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), fontsize=8)
    figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _group_points(asset: AnatomyRiggedAsset, vertices: np.ndarray, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    values = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    for tissue, (start, stop) in zip(_mesh_tissues(asset), ranges, strict=True):
        points = values[int(start) : int(stop)]
        if mask is not None:
            points = points[np.asarray(mask[int(start) : int(stop)], dtype=bool)]
        if len(points):
            groups[tissue].append(points)
    return {tissue: np.concatenate(chunks, axis=0) for tissue, chunks in groups.items()}


def _sample_containment(
    asset: AnatomyRiggedAsset,
    posed_vertices: np.ndarray,
    fit_vertices: np.ndarray,
    fit_faces: np.ndarray,
    *,
    samples_per_mesh: int,
) -> dict[str, Any]:
    """Sample signed SMPL-X skin distance per mesh without making the audit huge."""
    if samples_per_mesh < 1:
        return {"available": False, "reason": "containment sampling disabled"}
    try:
        import igl
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"igl unavailable: {exc}"}

    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    point_chunks: list[np.ndarray] = []
    descriptors: list[tuple[str, str, str, int]] = []
    tissues = _mesh_tissues(asset)
    sides = _mesh_sides(asset)
    for mesh_index, ((start, stop), name, tissue, side) in enumerate(
        zip(ranges, asset.source_mesh_names, tissues, sides, strict=True)
    ):
        count = int(stop - start)
        if not count:
            continue
        take = min(count, int(samples_per_mesh))
        local = np.linspace(int(start), int(stop) - 1, num=take, dtype=np.int64)
        point_chunks.append(np.asarray(posed_vertices, dtype=np.float64)[local])
        descriptors.extend(
            (str(name), tissue, side, mesh_index, int(vertex_index))
            for vertex_index in local.tolist()
        )
    if not point_chunks:
        return {"available": False, "reason": "asset has no anatomy vertices"}
    points = np.concatenate(point_chunks, axis=0)
    try:
        signed, _faces, _closest, _normal = igl.signed_distance(
            points,
            np.asarray(fit_vertices, dtype=np.float64),
            np.asarray(fit_faces, dtype=np.int32),
        )
    except Exception as exc:  # pragma: no cover - malformed third-party mesh
        return {"available": False, "reason": f"igl.signed_distance failed: {exc}"}
    values = np.asarray(signed, dtype=np.float64).reshape(-1)
    if len(values) != len(descriptors) or not np.all(np.isfinite(values)):
        return {"available": False, "reason": "signed-distance output is non-finite or mismatched"}

    by_tissue: dict[str, list[float]] = defaultdict(list)
    by_region: dict[str, list[float]] = defaultdict(list)
    by_mesh: dict[str, list[float]] = defaultdict(list)
    by_region_tissue: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_region_mesh: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    region_masks = {region: _region_material_mask(asset, region) for region in REGION_TOKENS}
    for distance, (name, tissue, side, _mesh_index, vertex_index) in zip(
        values.tolist(), descriptors, strict=True
    ):
        by_tissue[tissue].append(float(distance))
        by_mesh[name].append(float(distance))
        for region, material_mask in region_masks.items():
            if material_mask[vertex_index]:
                by_region[region].append(float(distance))
                by_region_tissue[region][tissue].append(float(distance))
                by_region_mesh[region][name].append(float(distance))

    def summarize(distances: list[float]) -> dict[str, Any]:
        array = np.asarray(distances, dtype=np.float64)
        outside = array > 0.0
        return {
            "sample_count": int(len(array)),
            "inside_fraction": float(np.mean(~outside)),
            "outside_count": int(np.count_nonzero(outside)),
            "max_outside_m": float(max(0.0, float(np.max(array)))),
            "signed_distance_p95_m": float(np.quantile(array, 0.95)),
            "minimum_signed_distance_m": float(np.min(array)),
        }

    region_metrics: dict[str, dict[str, Any]] = {}
    for name, distances in sorted(by_region.items()):
        if not distances:
            continue
        metric = summarize(distances)
        metric["tissues"] = {
            tissue: summarize(values)
            for tissue, values in sorted(by_region_tissue[name].items())
        }
        metric["meshes"] = {
            mesh: summarize(values)
            for mesh, values in sorted(by_region_mesh[name].items())
        }
        region_metrics[name] = metric
    return {
        "available": True,
        "method": "igl.signed_distance",
        "sign_convention": "negative is inside the oriented SMPL-X shell; positive is outside",
        "samples_per_mesh": int(samples_per_mesh),
        "sample_count": int(len(values)),
        "meshes": {name: summarize(distances) for name, distances in sorted(by_mesh.items())},
        "tissues": {name: summarize(distances) for name, distances in sorted(by_tissue.items())},
        "regions": region_metrics,
    }


def _source_rig_report(
    asset: AnatomyRiggedAsset,
    pose55: np.ndarray,
    transl: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    transforms = source_bone_skinning_transforms(asset, pose55)
    heads_rest = np.asarray(asset.target_bone_head, dtype=np.float32).reshape(-1, 3)
    tails_rest = np.asarray(asset.target_bone_tail, dtype=np.float32).reshape(-1, 3)
    rotations = np.asarray(transforms, dtype=np.float32)[:, :3, :3]
    offsets = np.asarray(transforms, dtype=np.float32)[:, :3, 3]
    heads_posed = np.einsum("bij,bj->bi", rotations, heads_rest) + offsets + transl.reshape(1, 3)
    tails_posed = np.einsum("bij,bj->bi", rotations, tails_rest) + offsets + transl.reshape(1, 3)
    rest_lengths = np.linalg.norm(tails_rest - heads_rest, axis=1)
    posed_lengths = np.linalg.norm(tails_posed - heads_posed, axis=1)
    valid = rest_lengths > 1.0e-8
    ratios = posed_lengths[valid] / rest_lengths[valid]
    report = {
        "source_bone_count": int(len(heads_rest)),
        "finite": bool(np.all(np.isfinite(heads_posed)) and np.all(np.isfinite(tails_posed))),
        "zero_length_rest_bones": int(np.count_nonzero(~valid)),
        "length_ratio_min": float(np.min(ratios)) if len(ratios) else None,
        "length_ratio_max": float(np.max(ratios)) if len(ratios) else None,
        "length_ratio_p99": float(np.quantile(ratios, 0.99)) if len(ratios) else None,
    }
    return report, heads_rest, tails_rest, np.stack((heads_posed, tails_posed), axis=0)


def audit_capture_pose(
    *,
    asset_npz: Path,
    motion_npz: Path,
    output_dir: Path,
    canonical_dir: Path | None = None,
    gender: str = "male",
    smplx_model: Path | None = None,
    apply_root_align: bool = False,
    containment_samples_per_mesh: int = 96,
    write_bone_chain_report: bool = True,
) -> dict[str, Any]:
    """Evaluate one capture and return the persisted audit report.

    This is intentionally public so release tests can exercise the same path
    used by the command line without a Blender dependency.
    """
    asset_npz = Path(asset_npz)
    motion_npz = Path(motion_npz)
    out = Path(output_dir)
    objects = out / "objs"
    overlays = out / "overlays"
    objects.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)

    asset = load_rigged_asset(asset_npz)
    motion = _load_motion(motion_npz, gender=gender, smplx_model=smplx_model)
    extra = motion["root_align_offset"] if apply_root_align else np.zeros(3, dtype=np.float32)
    pelvis = np.asarray(asset.rest_joints, dtype=np.float32).reshape(-1, 3)[0]
    drive_translation = easymocap_drive_translation(motion["Rh"], motion["Th"], pelvis) + extra
    pose55 = np.asarray(motion["pose55"], dtype=np.float32).reshape(55, 3)
    anatomy_rest = np.asarray(asset.vertices_rest, dtype=np.float32)
    anatomy_posed = skin_vertices(asset, pose55, transl=drive_translation)
    capture_smpl_posed = np.asarray(motion["fit_vertices"], dtype=np.float32) + extra.reshape(1, 3)
    capture_smpl_faces = np.asarray(motion["fit_faces"], dtype=np.int32)
    canonical = _canonical_reference(canonical_dir)
    reference_smpl_posed = capture_smpl_posed
    reference_smpl_faces = capture_smpl_faces
    reference_kind = "capture_fit_fallback"
    canonical_drive_translation: np.ndarray | None = None
    if canonical is not None and canonical.get("weights") is not None:
        reference_smpl_posed, canonical_drive_translation = _skin_canonical_reference(
            canonical,
            pose55=pose55,
            rh=motion["Rh"],
            th=motion["Th"],
            extra_translation=extra,
        )
        reference_smpl_faces = np.asarray(canonical["faces"], dtype=np.int32)
        reference_kind = "canonical_reposed_common_body"

    report: dict[str, Any] = {
        "audit_version": 1,
        "asset_npz": str(asset_npz),
        "motion_npz": str(motion_npz),
        "runtime_only": True,
        "requires_blender_at_runtime": False,
        "requires_pose_rebake": False,
        "gender": str(gender),
        "smplx_model": None if smplx_model is None else str(smplx_model),
        "source_pose_dimensions": int(motion["source_pose_dimensions"]),
        "pose_hash": smplx_pose_hash(pose55, drive_translation),
        "root_align_offset_m": [float(value) for value in motion["root_align_offset"]],
        "root_align_applied": bool(apply_root_align),
        "drive_translation_m": [float(value) for value in drive_translation],
        "artifact_files": [],
    }
    if motion["betas"] is not None:
        report["betas"] = [float(value) for value in np.asarray(motion["betas"]).reshape(-1)[:10]]

    capture_beta = None if motion["betas"] is None else np.asarray(motion["betas"], dtype=np.float32).reshape(-1)[:10]
    canonical_beta = None if canonical is None else canonical.get("manifest_betas")
    beta_l2 = None
    beta_max_abs = None
    if capture_beta is not None and canonical_beta is not None and len(capture_beta) == len(canonical_beta):
        delta = np.asarray(capture_beta, dtype=np.float64) - np.asarray(canonical_beta, dtype=np.float64)
        beta_l2 = float(np.linalg.norm(delta))
        beta_max_abs = float(np.max(np.abs(delta)))
    report["reference_surface"] = {
        "kind": reference_kind,
        "canonical_available": canonical is not None,
        "canonical_weights_available": bool(canonical is not None and canonical.get("weights") is not None),
        "canonical_object": None if canonical is None else str(canonical["object_path"]),
        "canonical_weights": None if canonical is None or canonical.get("weights") is None else str(canonical["weights"]["path"]),
        "canonical_manifest": None if canonical is None or canonical.get("manifest_path") is None else str(canonical["manifest_path"]),
        "capture_vs_canonical_beta_l2": beta_l2,
        "capture_vs_canonical_beta_max_abs": beta_max_abs,
        "beta_mismatch": None if beta_l2 is None else bool(beta_l2 > 1.0e-5),
        "capture_fit_is_not_used_for_common_body_containment": bool(reference_kind == "canonical_reposed_common_body"),
        "canonical_drive_translation_m": None if canonical_drive_translation is None else [float(value) for value in canonical_drive_translation],
        "asset_vs_canonical_pelvis_delta_m": None
        if canonical is None or canonical.get("weights") is None
        else [
            float(value)
            for value in (
                np.asarray(asset.rest_joints, dtype=np.float32)[0]
                - np.asarray(canonical["weights"]["rest_joints"], dtype=np.float32)[0]
            )
        ],
    }

    report["stage1_runtime_contract"] = stage1_runtime_contract(asset)
    report["anatomy_vertex_count"] = int(len(anatomy_posed))
    report["capture_smplx_vertex_count"] = int(len(capture_smpl_posed))
    report["reference_smplx_vertex_count"] = int(len(reference_smpl_posed))
    report["anatomy_finite"] = bool(np.all(np.isfinite(anatomy_posed)))
    tissue_counts: dict[str, int] = defaultdict(int)
    for tissue in _mesh_tissues(asset):
        tissue_counts[tissue] += 1
    report["tissue_semantics"] = {
        "source_tissues_available": bool(
            asset.source_tissues is not None and len(asset.source_tissues) == len(asset.source_mesh_names)
        ),
        "mesh_counts": dict(sorted(tissue_counts.items())),
    }

    artifact_counts: dict[str, dict[str, int]] = {}
    write_obj(objects / "anatomy_rest.obj", anatomy_rest, asset.faces, comment="schema-6 anatomy rest pose")
    write_obj(objects / "anatomy_posed.obj", anatomy_posed, asset.faces, comment="schema-6 anatomy live SMPL-X drive")
    write_obj(
        objects / "smplx_capture_fit_posed.obj",
        capture_smpl_posed,
        capture_smpl_faces,
        comment="fitted SMPL-X capture shell; beta may differ from the Stage-1 canonical body",
    )
    write_obj(
        objects / "smplx_posed.obj",
        reference_smpl_posed,
        reference_smpl_faces,
        comment=f"common-body validation SMPL-X surface ({reference_kind})",
    )
    if reference_kind == "canonical_reposed_common_body":
        write_obj(
            objects / "smplx_canonical_reposed.obj",
            reference_smpl_posed,
            reference_smpl_faces,
            comment="fixed canonical SMPL-X body posed with this capture",
        )
    bone_mask = _tissue_mask(asset, {"bone"})
    vessel_mask = _tissue_mask(asset, {"vessel"})
    artifact_counts["bones_rest"] = _subset_obj(
        objects / "bones_rest.obj", anatomy_rest, asset.faces, bone_mask, comment="anatomy bone subset rest pose"
    )
    artifact_counts["bones_posed"] = _subset_obj(
        objects / "bones_posed.obj", anatomy_posed, asset.faces, bone_mask, comment="anatomy bone subset live drive"
    )
    artifact_counts["vessels_rest"] = _subset_obj(
        objects / "vessels_rest.obj", anatomy_rest, asset.faces, vessel_mask, comment="anatomy vessel subset rest pose"
    )
    artifact_counts["vessels_posed"] = _subset_obj(
        objects / "vessels_posed.obj", anatomy_posed, asset.faces, vessel_mask, comment="anatomy vessel subset live drive"
    )
    report["mesh_artifact_counts"] = artifact_counts

    rig_report, source_heads_rest, source_tails_rest, source_lines_posed = _source_rig_report(
        asset, pose55, drive_translation
    )
    report["source_rig"] = rig_report
    report["source_rig"]["rest_line_count"] = _write_line_skeleton(
        objects / "source_rig_rest.obj",
        heads=source_heads_rest,
        tails=source_tails_rest,
        names=asset.source_bone_names or [],
        comment="subject-fitted source rig rest endpoints",
    )
    report["source_rig"]["posed_line_count"] = _write_line_skeleton(
        objects / "source_rig_posed.obj",
        heads=source_lines_posed[0],
        tails=source_lines_posed[1],
        names=asset.source_bone_names or [],
        comment="source rig evaluated by persisted SMPL-X drivers",
    )
    smpl_rest_joints = np.asarray(asset.rest_joints, dtype=np.float32)
    smpl_posed_joints = joint_global_transforms(
        pose_axis_angle=pose55,
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    )[:, :3, 3] + drive_translation.reshape(1, 3)
    _write_joint_skeleton(
        objects / "smplx55_skeleton_rest.obj",
        joints=smpl_rest_joints,
        parents=asset.parents,
        names=asset.joint_names,
        comment="SMPL-X 55 target rest skeleton",
    )
    _write_joint_skeleton(
        objects / "smplx55_skeleton_posed.obj",
        joints=smpl_posed_joints,
        parents=asset.parents,
        names=asset.joint_names,
        comment="SMPL-X 55 target capture skeleton",
    )

    report["containment"] = _sample_containment(
        asset,
        anatomy_posed,
        reference_smpl_posed,
        reference_smpl_faces,
        samples_per_mesh=containment_samples_per_mesh,
    )

    posed_groups = _group_points(asset, anatomy_posed)
    _draw_overlay(
        overlays / "posed_full_anatomy_overlay.png",
        title=f"Live Stage-1 anatomy over SMPL-X {reference_kind.replace('_', ' ')}",
        smpl_vertices=reference_smpl_posed,
        anatomy_groups=posed_groups,
    )
    _draw_overlay(
        overlays / "posed_bones_vessels_overlay.png",
        title=f"Live bone and vessel overlap over SMPL-X {reference_kind.replace('_', ' ')}",
        smpl_vertices=reference_smpl_posed,
        anatomy_groups={
            "bone": anatomy_posed[bone_mask],
            "vessel": anatomy_posed[vessel_mask],
        },
    )
    _draw_overlay(
        overlays / "posed_capture_fit_overlay.png",
        title="Live anatomy over direct captured SMPL-X fit (diagnostic only)",
        smpl_vertices=capture_smpl_posed,
        anatomy_groups={"bone": anatomy_posed[bone_mask], "vessel": anatomy_posed[vessel_mask]},
    )
    regional_written: list[str] = []
    for region in REGION_TOKENS:
        mask = _region_mask(asset, region)
        if not np.any(mask):
            continue
        focus = anatomy_posed[mask]
        _draw_overlay(
            overlays / "posed_regions" / f"{region}.png",
            title=f"Live Stage-1 {region.replace('_', ' ')} over SMPL-X {reference_kind.replace('_', ' ')}",
            smpl_vertices=reference_smpl_posed,
            anatomy_groups=_group_points(asset, anatomy_posed, mask),
            point_limit=9000,
            region_points=focus,
        )
        regional_written.append(region)
    report["posed_regional_overlays"] = regional_written

    report["canonical_tpose"] = {"available": canonical is not None, "path": None if canonical_dir is None else str(canonical_dir)}
    if canonical is not None:
        smpl_rest = np.asarray(canonical["vertices"], dtype=np.float32)
        smpl_rest_faces = np.asarray(canonical["faces"], dtype=np.int32)
        report["rest_containment"] = _sample_containment(
            asset,
            anatomy_rest,
            smpl_rest,
            smpl_rest_faces,
            samples_per_mesh=containment_samples_per_mesh,
        )
        write_obj(objects / "smplx_rest.obj", smpl_rest, smpl_rest_faces, comment="subject canonical SMPL-X T-pose shell")
        _draw_overlay(
            overlays / "rest_full_anatomy_overlay.png",
            title="Stage-1 anatomy rest pose over subject canonical SMPL-X",
            smpl_vertices=smpl_rest,
            anatomy_groups=_group_points(asset, anatomy_rest),
        )
        _draw_overlay(
            overlays / "rest_bones_vessels_overlay.png",
            title="Stage-1 bone and vessel rest overlap over canonical SMPL-X",
            smpl_vertices=smpl_rest,
            anatomy_groups={"bone": anatomy_rest[bone_mask], "vessel": anatomy_rest[vessel_mask]},
        )
        rest_regions: list[str] = []
        for region in REGION_TOKENS:
            mask = _region_mask(asset, region)
            if not np.any(mask):
                continue
            _draw_overlay(
                overlays / "rest_regions" / f"{region}.png",
                title=f"Stage-1 rest {region.replace('_', ' ')} over canonical SMPL-X",
                smpl_vertices=smpl_rest,
                anatomy_groups=_group_points(asset, anatomy_rest, mask),
                point_limit=9000,
                region_points=anatomy_rest[mask],
            )
            rest_regions.append(region)
        report["rest_regional_overlays"] = rest_regions
    else:
        report["rest_containment"] = {
            "available": False,
            "reason": "canonical Stage-1 body is required for rest-pose containment",
        }

    if write_bone_chain_report:
        try:
            bone_chain = write_bone_segment_diagnostics(
                asset,
                pose_axis_angle=pose55,
                transl=drive_translation,
                output_path=out / "bone_chain_diagnostics.json",
            )
            report["bone_chain_diagnostics"] = {
                "available": True,
                "passed": bool(bone_chain.get("passed")),
                "failure_count": int(len(bone_chain.get("failures", []))),
            }
        except Exception as exc:  # A visual audit should remain available for a bad chain.
            report["bone_chain_diagnostics"] = {"available": False, "error": str(exc)}
    else:
        report["bone_chain_diagnostics"] = {"available": False, "reason": "disabled by caller"}

    report["stage1_pose_acceptance"] = stage1_pose_acceptance(report)
    report["artifact_files"] = sorted(
        str(path.relative_to(out))
        for path in out.rglob("*")
        if path.is_file() and path.name != "capture_audit.json"
    )
    (out / "capture_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset-npz", type=Path, default=DEFAULT_ASSET, help="published schema-6 anatomy asset")
    parser.add_argument(
        "--motion-npz",
        type=Path,
        required=True,
        action="append",
        help="capture smplx_result.npz; repeat for a capture-pose audit matrix",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=None,
        help="canonical-cache directory (or smpl_canonical_tpose.obj) for rest-pose overlays",
    )
    parser.add_argument("--gender", choices=("male", "female", "neutral"), default="male")
    parser.add_argument("--smplx-model", type=Path, default=None, help="override SMPL-X PKL used to decode 87-D hand PCA")
    parser.add_argument(
        "--apply-root-align",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="add root_align_offset to both the anatomy and fitted SMPL-X output",
    )
    parser.add_argument(
        "--containment-samples-per-mesh",
        type=int,
        default=96,
        help="deterministic signed-distance samples per anatomy mesh; 0 disables containment sampling",
    )
    parser.add_argument(
        "--bone-chain-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also run source-bone chain diagnostics for this exact capture pose",
    )
    parser.add_argument(
        "--enforce-stage1-pose-acceptance",
        action="store_true",
        help="return non-zero unless every audited capture satisfies the Stage-1 pose gate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    motions = [Path(value) for value in args.motion_npz]
    labels: set[str] = set()
    matrix: list[dict[str, Any]] = []
    for index, motion in enumerate(motions):
        # ``smplx_result.npz`` repeats for every capture, so use the terminal
        # capture directory as the matrix label when that layout is present.
        raw_label = motion.parent.parent.name if motion.parent.name.startswith("moment_") else motion.stem
        label = "".join(char if char.isalnum() or char in "-_" else "_" for char in raw_label) or f"capture_{index:02d}"
        candidate = label
        suffix = 2
        while candidate in labels:
            candidate = f"{label}_{suffix}"
            suffix += 1
        labels.add(candidate)
        output = Path(args.output_dir) if len(motions) == 1 else Path(args.output_dir) / candidate
        report = audit_capture_pose(
            asset_npz=args.asset_npz,
            motion_npz=motion,
            output_dir=output,
            canonical_dir=args.canonical_dir,
            gender=args.gender,
            smplx_model=args.smplx_model,
            apply_root_align=bool(args.apply_root_align),
            containment_samples_per_mesh=int(args.containment_samples_per_mesh),
            write_bone_chain_report=bool(args.bone_chain_diagnostics),
        )
        matrix.append(
            {
                "label": candidate,
                "motion_npz": str(motion),
                "output_dir": str(output),
                "stage1_runtime_contract_passed": bool(report["stage1_runtime_contract"]["passed"]),
                "stage1_pose_acceptance_passed": bool(report["stage1_pose_acceptance"]["passed"]),
                "reference_surface": report["reference_surface"],
                "containment_available": bool(report.get("containment", {}).get("available")),
            }
        )
        print(f"INFO capture audit written -> {output}")
        print(f"INFO Stage-1 runtime contract passed={report['stage1_runtime_contract']['passed']}")
        print(f"INFO Stage-1 pose acceptance passed={report['stage1_pose_acceptance']['passed']}")
        containment = report.get("containment", {})
        if containment.get("available"):
            print(f"INFO containment samples={containment['sample_count']} method={containment['method']}")
        else:
            print(f"WARN containment unavailable: {containment.get('reason')}")
    if len(matrix) > 1:
        matrix_path = Path(args.output_dir) / "capture_audit_matrix.json"
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.write_text(
            json.dumps({"audit_version": 1, "cases": matrix}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"INFO capture audit matrix written -> {matrix_path}")
    if args.enforce_stage1_pose_acceptance:
        failed = [case["label"] for case in matrix if not case["stage1_pose_acceptance_passed"]]
        if failed:
            print(f"ERROR Stage-1 pose acceptance rejected: {', '.join(failed)}")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
