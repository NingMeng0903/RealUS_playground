"""Evaluate two compiled anatomy packages on captures and frozen V15 motion.

This is a replay and review-data CLI.  It never fits, rebinds, searches for
nearest points, or calls Blender/Genesis.  The ``before`` and ``candidate``
arrays in each geometry archive are evaluated with the exact V14 runtime that
is stored in the two packages.  ``source_vertices`` is a calibrated-source
controller sparse-LBS diagnostic from the before package, which gives the
renderer a stable historical comparison in addition to before/after geometry.

By default the command writes nine geometry archives: T-pose, both capture
poses, and the midpoint of each of the six immutable V15 BABEL/AMASS clips.
The first and last frames of those clips can be evaluated as lightweight
metrics with ``--endpoint-metrics``; they are deliberately not written as
large geometry archives unless they are the selected midpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.chain_containment_v1 import (
    _signed_distance,
)
from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    CompiledAnatomyV14,
    load_compiled_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    _smplx_joint_kinematics_v7,
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.sparse_lbs_v14 import SparseLBSV14
from projects.genesis_ue_sync.anatomy_retarget.validation_motion_v15 import (
    CLIPS_V15,
    DEFAULT_AMASS_ROOT,
    load_frozen_validation_clip_v15,
    load_validation_clip_v15,
)


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
DEFAULT_CAPTURE_213328 = ROOT / (
    "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"
)
DEFAULT_CAPTURE_213712 = ROOT / (
    "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz"
)
DEFAULT_VALIDATION_ROOT = ROOT / (
    "outputs/anatomy_retarget/v15_frozen_validation_20260908_001"
)

TISSUE_CODES: dict[str, int] = {
    "bone": 0,
    "vessel": 1,
    "nerve": 2,
    "organ": 3,
    "heart": 4,
    "connective": 5,
    "connective_tissue": 5,
}
VESSEL_LIMIT_M = 0.001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    """Convert NumPy values to strict JSON and reject hidden non-finite data."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("report contains a non-finite float")
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def _manifest_identity(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if not isinstance(files, Mapping):
        raise ValueError(f"{root}: compiled manifest has no file inventory")
    return {
        "path": str(root),
        "manifest_sha256": _sha256(manifest_path),
        "files": {str(name): str(value) for name, value in files.items()},
        "artifact_kind": manifest.get("artifact_kind"),
        "schema_version": manifest.get("schema_version"),
        "publishable": bool(manifest.get("publishable", False)),
        "anatomical_passed": bool(manifest.get("anatomical_passed", False)),
    }


def _source_hashes(asset: Any) -> dict[str, str]:
    names = (
        "vertices_rest",
        "faces",
        "driver_indices",
        "driver_weights",
        "target_inverse_bind",
        "source_mesh_controller_bones",
        "source_vertex_ranges",
    )
    result: dict[str, str] = {}
    for name in names:
        value = getattr(asset, name, None)
        if value is not None:
            result[name] = _array_sha256(value)
    return result


def _validate_pair(
    before: CompiledAnatomyV14,
    after: CompiledAnatomyV14,
    before_root: Path,
    after_root: Path,
) -> dict[str, Any]:
    before_beta = np.asarray(before.betas, dtype=np.float32).reshape(-1)
    after_beta = np.asarray(after.betas, dtype=np.float32).reshape(-1)
    if before_beta.shape != (10,) or after_beta.shape != (10,):
        raise ValueError("compiled packages must each contain exactly ten betas")
    if not np.isfinite(before_beta).all() or not np.isfinite(after_beta).all():
        raise ValueError("compiled beta values must be finite")
    if not np.array_equal(before_beta, after_beta):
        raise ValueError(
            "compiled beta mismatch: before and after must use the exact same "
            "subject shape"
        )

    before_asset = before.source_asset
    after_asset = after.source_asset
    equality: dict[str, bool] = {}
    for name in (
        "vertices_rest",
        "faces",
        "driver_indices",
        "driver_weights",
        "source_mesh_controller_bones",
        "source_vertex_ranges",
    ):
        left = getattr(before_asset, name, None)
        right = getattr(after_asset, name, None)
        equality[name] = bool(
            left is not None
            and right is not None
            and np.shape(left) == np.shape(right)
            and np.array_equal(left, right)
        )
    if not equality["vertices_rest"] or not equality["faces"]:
        raise ValueError("before/after source topology or rest vertices differ")
    if not equality["driver_indices"] or not equality["driver_weights"]:
        raise ValueError("before/after authored sparse weights differ")

    faces = np.asarray(before_asset.faces, dtype=np.int32)
    vertices = np.asarray(before_asset.vertices_rest, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("source anatomy faces must have shape [F, 3]")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("source anatomy vertices must have shape [V, 3]")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("source anatomy faces contain out-of-range indices")
    if not hasattr(before_asset, "target_inverse_bind"):
        raise ValueError("source asset lacks target_inverse_bind for source replay")
    return {
        "beta_equal_exact": True,
        "betas_float32": before_beta.tolist(),
        "weights_faces_equality": equality,
        "source_hashes_before": _source_hashes(before_asset),
        "source_hashes_after": _source_hashes(after_asset),
        "before": _manifest_identity(before_root),
        "after": _manifest_identity(after_root),
    }


def _load_capture_pose(
    path: Path,
    *,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a capture through the established adapter and retain native Th."""

    # This import is intentionally local: importing the evaluator for parser
    # tests must not initialize any capture or Blender-related module.
    from projects.genesis_ue_sync.anatomy_retarget.cli.run_material_matrix_v13 import (
        _load_capture,
    )

    path = Path(path).expanduser().resolve()
    _capture_betas, pose, source_sha = _load_capture(path, model_path=model_path)
    with np.load(path, allow_pickle=False) as data:
        if "Th" not in data.files:
            raise ValueError(f"{path}: capture lacks Th")
        th = np.asarray(data["Th"], dtype=np.float64).reshape(-1)
        if th.shape != (3,) or not np.isfinite(th).all():
            raise ValueError(f"{path}: Th must be finite with shape (3,)")
        offset = np.zeros(3, dtype=np.float64)
        if "root_align_offset" in data.files:
            offset = np.asarray(data["root_align_offset"], dtype=np.float64).reshape(-1)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError(f"{path}: root_align_offset must be finite with shape (3,)")
    transl = th + offset
    pose = _finite_array(pose, (55, 3), "capture pose").astype(np.float32)
    return pose, transl, {
        "path": str(path),
        "sha256": str(source_sha),
        "capture_betas_used": False,
        "native_Th_m": th.tolist(),
        "root_align_offset_m": offset.tolist(),
        "native_translation_m": transl.tolist(),
        "pose_layout": "smplx55_axis_angle",
    }


def _load_validation_inputs(
    validation_root: Path,
    amass_root: Path,
    *,
    require_frozen: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the immutable V15 archives, with an explicit raw-source fallback."""

    root = Path(validation_root).expanduser().resolve()
    if root.is_dir() and (root / "manifest.json").is_file():
        manifest_sha = _sha256(root / "manifest.json")
        clips = {
            spec.name: load_frozen_validation_clip_v15(root, spec.name)
            for spec in CLIPS_V15
        }
        return clips, {
            "mode": "frozen_v15",
            "root": str(root),
            "manifest_sha256": manifest_sha,
            "used_for_fit": False,
            "clip_count": len(clips),
        }
    if require_frozen:
        raise FileNotFoundError(
            f"V15 frozen validation root is required but missing manifest: {root}"
        )
    clips = {
        spec.name: load_validation_clip_v15(spec, amass_root=amass_root)
        for spec in CLIPS_V15
    }
    return clips, {
        "mode": "raw_v15_spec_fallback",
        "root": str(root),
        "manifest_sha256": None,
        "amass_root": str(Path(amass_root).expanduser().resolve()),
        "used_for_fit": False,
        "clip_count": len(clips),
        "warning": "frozen V15 manifest absent; exact audited specs were loaded directly",
    }


def _vertex_tissue_codes(asset: Any) -> np.ndarray:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    tissues = list(asset.source_tissues or ())
    if len(ranges) != len(tissues):
        raise ValueError("source mesh ranges and tissues have different lengths")
    labels = np.full(len(asset.vertices_rest), -1, dtype=np.int8)
    for (start, stop), tissue in zip(ranges.tolist(), tissues):
        label = str(tissue).strip().lower()
        if label not in TISSUE_CODES:
            raise ValueError(f"unknown source tissue label: {tissue!r}")
        lo, hi = int(start), int(stop)
        if lo < 0 or hi <= lo or hi > len(labels):
            raise ValueError(f"invalid source mesh vertex range: {(lo, hi)}")
        if np.any(labels[lo:hi] >= 0):
            raise ValueError("source mesh vertex ranges overlap")
        labels[lo:hi] = TISSUE_CODES[label]
    if np.any(labels < 0):
        raise ValueError("source mesh ranges do not label every anatomy vertex")
    return labels


def _descendants(parents: np.ndarray, roots: Iterable[int]) -> np.ndarray:
    selected = np.zeros(len(parents), dtype=bool)
    selected[np.asarray(tuple(roots), dtype=np.int64)] = True
    changed = True
    while changed:
        changed = False
        for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
            if not selected[index] and int(parent) >= 0 and selected[int(parent)]:
                selected[index] = True
                changed = True
    return np.flatnonzero(selected)


def _arm_mesh_mask(asset: Any, side: str) -> np.ndarray:
    """Return source bone mesh components belonging to one arm.

    The controller hierarchy is authoritative for the normal assets.  The
    name fallback makes old schema snapshots diagnosable if a collar root is
    absent, while still selecting bone tissue only.
    """

    names = [str(value) for value in asset.source_bone_names or ()]
    roots = [
        index
        for index, name in enumerate(names)
        if name.lower() == f"clavicle_rot_{side}".lower()
    ]
    controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64).reshape(-1)
    if roots:
        bones = set(_descendants(np.asarray(asset.source_bone_parents), roots).tolist())
        return np.isin(controllers, tuple(sorted(bones)))

    side_token = "_l" if side.lower() == "l" else "_r"
    mesh_names = [str(value).lower() for value in asset.source_mesh_names or ()]
    arm_words = ("scapula", "clavicle", "humerus", "radius", "ulna", "forearm", "wrist", "hand")
    return np.asarray(
        [
            side_token in name
            and any(word in name for word in arm_words)
            for name in mesh_names
        ],
        dtype=bool,
    )


def _region_ids(asset: Any, vertex_tissue: np.ndarray) -> dict[str, np.ndarray]:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    mesh_count = len(ranges)
    bone_mesh = vertex_tissue[ranges[:, 0]] == TISSUE_CODES["bone"]
    vessel_ids = np.flatnonzero(vertex_tissue == TISSUE_CODES["vessel"])
    all_bone_ids = np.flatnonzero(vertex_tissue == TISSUE_CODES["bone"])

    def mesh_vertices(mask: np.ndarray) -> np.ndarray:
        if mask.shape != (mesh_count,):
            raise ValueError("mesh mask has an invalid shape")
        chunks = [
            np.arange(int(start), int(stop), dtype=np.int64)
            for include, (start, stop) in zip(mask.tolist(), ranges.tolist())
            if include
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    left = mesh_vertices(bone_mesh & _arm_mesh_mask(asset, "L"))
    right = mesh_vertices(bone_mesh & _arm_mesh_mask(asset, "R"))
    return {
        "left_arm": left,
        "right_arm": right,
        "all_bones": all_bone_ids,
        "vessels": vessel_ids,
    }


def _distance_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {
            "available": False,
            "vertex_count": 0,
            "max_signed_m": 0.0,
            "max_outside_m": 0.0,
            "max_outside_mm": 0.0,
            "outside_count": 0,
            "outside_over_1mm_count": 0,
        }
    if not np.isfinite(values).all():
        raise ValueError("signed-distance values contain non-finite entries")
    maximum = float(np.max(values))
    outside = values > 0.0
    outside_max = max(0.0, maximum)
    result = {
        "available": True,
        "vertex_count": int(len(values)),
        "max_signed_m": maximum,
        "max_outside_m": outside_max,
        "max_outside_mm": outside_max * 1000.0,
        "outside_count": int(np.count_nonzero(outside)),
        "outside_over_1mm_count": int(np.count_nonzero(values > VESSEL_LIMIT_M)),
    }
    return result


def _surface_metrics(
    vertices: np.ndarray,
    *,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    asset: Any,
    region_ids: Mapping[str, np.ndarray],
    include_per_mesh: bool,
) -> dict[str, Any]:
    signed = np.asarray(_signed_distance(vertices, skin, skin_faces), dtype=np.float64)
    if signed.shape != (len(vertices),) or not np.isfinite(signed).all():
        raise ValueError("signed-distance query returned invalid values")
    regions = {
        name: _distance_summary(signed[np.asarray(ids, dtype=np.int64)])
        for name, ids in region_ids.items()
    }
    vessel_summary = regions["vessels"]
    vessel_summary["vessel_over_1mm_fail"] = bool(
        vessel_summary["max_outside_m"] > VESSEL_LIMIT_M
    )
    result: dict[str, Any] = {
        "signed_distance_convention": "negative_inside_positive_outside",
        "regions": regions,
        "vessels_over_1mm_fail": bool(
            regions["vessels"].get("vessel_over_1mm_fail", False)
        ),
        "per_mesh_included": bool(include_per_mesh),
    }
    if include_per_mesh:
        ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
        names = list(asset.source_mesh_names or ())
        tissues = list(asset.source_tissues or ())
        per_mesh: dict[str, Any] = {}
        for index, ((start, stop), name, tissue) in enumerate(zip(ranges.tolist(), names, tissues)):
            key = f"mesh_{index}:{name}"
            per_mesh[key] = {
                "mesh_index": int(index),
                "name": str(name),
                "tissue": str(tissue),
                **_distance_summary(signed[int(start):int(stop)]),
            }
        result["per_mesh"] = per_mesh
    return result


def _skin_joints(
    model: Mapping[str, np.ndarray],
    *,
    betas: np.ndarray,
    pose: np.ndarray,
    transl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skin, skin_faces = smplx_body_surface_v7(
        model,
        betas=betas,
        pose_axis_angle=pose,
    )
    _rest_joints, globals_, _rest_to_pose = _smplx_joint_kinematics_v7(
        model,
        betas=betas,
        pose_axis_angle=pose,
    )
    skin = np.asarray(skin, dtype=np.float64) + transl.reshape(1, 3)
    joints = np.asarray(globals_, dtype=np.float64)[:, :3, 3] + transl.reshape(1, 3)
    skin_faces = np.asarray(skin_faces, dtype=np.int32)
    if skin.ndim != 2 or skin.shape[1] != 3 or joints.shape != (55, 3):
        raise ValueError("SMPL-X skin/joints have invalid shapes")
    if not np.isfinite(skin).all() or not np.isfinite(joints).all():
        raise ValueError("SMPL-X skin/joints contain non-finite values")
    return skin, skin_faces, joints


def _source_vertices(
    compiled: CompiledAnatomyV14,
    source_lbs: SparseLBSV14,
    pose: np.ndarray,
    transl: np.ndarray,
) -> np.ndarray:
    asset = compiled.source_asset
    globals_ = np.asarray(compiled.source_globals(pose), dtype=np.float64)
    inverse_bind = np.asarray(asset.target_inverse_bind, dtype=np.float64)
    if inverse_bind.shape != (len(globals_), 4, 4):
        raise ValueError("source target_inverse_bind has an invalid shape")
    vertices = source_lbs(globals_ @ inverse_bind)
    vertices = np.asarray(vertices, dtype=np.float64) + transl.reshape(1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError("source replay produced non-finite vertices")
    return vertices


def _evaluate_frame(
    *,
    before: CompiledAnatomyV14,
    after: CompiledAnatomyV14,
    source_lbs: SparseLBSV14,
    model: Mapping[str, np.ndarray],
    pose: np.ndarray,
    transl: np.ndarray,
    asset: Any,
    vertex_tissue: np.ndarray,
    region_ids: Mapping[str, np.ndarray],
    include_per_mesh: bool,
) -> dict[str, Any]:
    pose = _finite_array(pose, (55, 3), "pose").astype(np.float32)
    transl = _finite_array(transl, (3,), "transl")
    before_vertices = np.asarray(before.apply_pose(pose, transl), dtype=np.float64)
    candidate_vertices = np.asarray(after.apply_pose(pose, transl), dtype=np.float64)
    source_vertices = _source_vertices(before, source_lbs, pose, transl)
    skin, skin_faces, joints = _skin_joints(
        model, betas=np.asarray(before.betas, dtype=np.float64), pose=pose, transl=transl
    )
    for name, vertices in (
        ("before", before_vertices),
        ("candidate", candidate_vertices),
        ("source", source_vertices),
    ):
        if vertices.shape != np.asarray(asset.vertices_rest).shape or not np.isfinite(vertices).all():
            raise ValueError(f"{name} anatomy vertices have an invalid shape or value")
    metrics = {
        "before": _surface_metrics(
            before_vertices,
            skin=skin,
            skin_faces=skin_faces,
            asset=asset,
            region_ids=region_ids,
            include_per_mesh=include_per_mesh,
        ),
        "candidate": _surface_metrics(
            candidate_vertices,
            skin=skin,
            skin_faces=skin_faces,
            asset=asset,
            region_ids=region_ids,
            include_per_mesh=include_per_mesh,
        ),
        "source": _surface_metrics(
            source_vertices,
            skin=skin,
            skin_faces=skin_faces,
            asset=asset,
            region_ids=region_ids,
            include_per_mesh=include_per_mesh,
        ),
    }
    return {
        "before_vertices": before_vertices.astype(np.float32),
        "candidate_vertices": candidate_vertices.astype(np.float32),
        "source_vertices": source_vertices.astype(np.float32),
        "faces": np.asarray(asset.faces, dtype=np.int32),
        "skin_vertices": skin.astype(np.float32),
        "skin_faces": skin_faces,
        "smplx_joints": joints.astype(np.float32),
        "pose": pose,
        "transl": transl.astype(np.float32),
        "vertex_tissue": np.asarray(vertex_tissue, dtype=np.int8),
        "metrics": metrics,
    }


def _save_geometry(path: Path, arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        before_vertices=np.asarray(arrays["before_vertices"], dtype=np.float32),
        candidate_vertices=np.asarray(arrays["candidate_vertices"], dtype=np.float32),
        source_vertices=np.asarray(arrays["source_vertices"], dtype=np.float32),
        faces=np.asarray(arrays["faces"], dtype=np.int32),
        pose=np.asarray(arrays["pose"], dtype=np.float32),
        skin_vertices=np.asarray(arrays["skin_vertices"], dtype=np.float32),
        skin_faces=np.asarray(arrays["skin_faces"], dtype=np.int32),
        smplx_joints=np.asarray(arrays["smplx_joints"], dtype=np.float32),
        vertex_tissue=np.asarray(arrays["vertex_tissue"], dtype=np.int8),
        transl=np.asarray(arrays["transl"], dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps(_json_ready(metadata), sort_keys=True, allow_nan=False)
        ),
    )
    return _sha256(path)


def _pose_record(
    *,
    label: str,
    role: str,
    pose: np.ndarray,
    transl: np.ndarray,
    clip_name: str | None,
    frame_id: int | None,
    fps: float | None,
) -> dict[str, Any]:
    return {
        "label": label,
        "role": role,
        "clip": clip_name,
        "frame_id": frame_id,
        "fps": fps,
        "pose_sha256": _array_sha256(np.asarray(pose, dtype=np.float32)),
        "translation_m": np.asarray(transl, dtype=np.float64).tolist(),
    }


def evaluate_compiled_pair_v15(
    before_compiled: str | Path,
    after_compiled: str | Path,
    output: str | Path,
    *,
    smplx_model: str | Path = DEFAULT_MODEL,
    capture_213328: str | Path = DEFAULT_CAPTURE_213328,
    capture_213712: str | Path = DEFAULT_CAPTURE_213712,
    validation_root: str | Path = DEFAULT_VALIDATION_ROOT,
    amass_root: str | Path = DEFAULT_AMASS_ROOT,
    endpoint_metrics: bool = False,
    require_frozen_validation: bool = False,
) -> dict[str, Any]:
    """Build the V15 geometry review pack and return its JSON report."""

    before_root = Path(before_compiled).expanduser().resolve()
    after_root = Path(after_compiled).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output_root}")

    model_path, model_sha = require_frozen_smplx_male_v7(smplx_model)
    model = load_smplx_model_v7(model_path)
    before = load_compiled_subject(before_root)
    after = load_compiled_subject(after_root)
    pair_identity = _validate_pair(before, after, before_root, after_root)

    capture_specs = (
        ("capture_213328", Path(capture_213328)),
        ("capture_213712", Path(capture_213712)),
    )
    captures: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for name, path in capture_specs:
        captures[name] = _load_capture_pose(path, model_path=model_path)
    clips, validation_identity = _load_validation_inputs(
        Path(validation_root),
        Path(amass_root),
        require_frozen=require_frozen_validation,
    )

    asset = before.source_asset
    vertex_tissue = _vertex_tissue_codes(asset)
    region_ids = _region_ids(asset, vertex_tissue)
    source_lbs = SparseLBSV14(
        np.asarray(asset.vertices_rest, dtype=np.float64),
        np.asarray(asset.driver_indices, dtype=np.int64),
        np.asarray(asset.driver_weights, dtype=np.float64),
    )

    frame_plan: list[dict[str, Any]] = []
    # The plan always names all 21 protocol frames, even when endpoint metrics
    # are disabled.  This prevents a nine-NPZ review from being mistaken for a
    # different validation set.
    frame_plan.append(_pose_record(
        label="tpose", role="tpose", pose=np.zeros((55, 3), dtype=np.float32),
        transl=np.zeros(3), clip_name=None, frame_id=None, fps=None,
    ))
    for name, (pose, transl, _info) in captures.items():
        frame_plan.append(_pose_record(
            label=name, role="capture", pose=pose, transl=transl,
            clip_name=None, frame_id=None, fps=None,
        ))
    for spec in CLIPS_V15:
        clip = clips[spec.name]
        for role, position in (("first", 0), ("middle", clip.frame_count // 2), ("last", clip.frame_count - 1)):
            index = int(position)
            frame_plan.append(_pose_record(
                label=f"{spec.name}_{role}", role=role,
                pose=clip.poses[index], transl=clip.transl[index],
                clip_name=spec.name, frame_id=int(clip.frame_ids[index]), fps=clip.fps,
            ))

    output_root.mkdir(parents=True, exist_ok=False)
    geometry_root = output_root / "geometry"
    geometry_root.mkdir()
    report: dict[str, Any] = {
        "schema_version": 15,
        "artifact_kind": "CompiledPairEvaluationV15",
        "immutable": True,
        "validation_only": True,
        "used_for_fit": False,
        "runtime_opt": False,
        "runtime_optimization": False,
        "runtime_blender": False,
        "anatomical_passed": False,
        "bone_skin_policy": "visual_review",
        "vessel_skin_limit_m": VESSEL_LIMIT_M,
        "vessel_over_1mm_is_fail": True,
        "before_compiled": pair_identity["before"],
        "after_compiled": pair_identity["after"],
        "smplx_model": {"path": str(model_path), "sha256": model_sha},
        "betas_float32": pair_identity["betas_float32"],
        "beta_equal_exact": pair_identity["beta_equal_exact"],
        "weights_faces_equality": pair_identity["weights_faces_equality"],
        "source_hashes_before": pair_identity["source_hashes_before"],
        "source_hashes_after": pair_identity["source_hashes_after"],
        "source_vertices_semantics": (
            "calibrated-source diagnostic: before.source_asset.vertices_rest transported "
            "by before.source_globals (including the before package motion response) "
            "through before.source_asset.target_inverse_bind and source sparse weights"
        ),
        "translation_policy": (
            "native transl is added equally to before/candidate/source anatomy, "
            "SMPL-X skin, and SMPL-X joints"
        ),
        "capture_sources": {},
        "validation_source": validation_identity,
        "validation_clips": {
            spec.name: {
                "sid": int(spec.sid),
                "babel_label": spec.babel_label,
                "source_path": str(clips[spec.name].source_path),
                "source_sha256": str(clips[spec.name].source_sha256),
                "source_fps": float(clips[spec.name].fps),
                "frame_ids": np.asarray(clips[spec.name].frame_ids, dtype=np.int64).tolist(),
                "used_for_fit": False,
            }
            for spec in CLIPS_V15
        },
        "frame_plan_count": len(frame_plan),
        "geometry_default_count": 9,
        "endpoint_metrics_enabled": bool(endpoint_metrics),
        "frame_plan": frame_plan,
        "regions": {
            name: {"vertex_count": int(len(ids))}
            for name, ids in region_ids.items()
        },
        "frames": {},
        "geometry_files": {},
        "errors": [],
    }
    for name, (_pose, _transl, info) in captures.items():
        report["capture_sources"][name] = info

    def evaluate_and_record(
        *,
        label: str,
        role: str,
        pose: np.ndarray,
        transl: np.ndarray,
        clip_name: str | None,
        frame_id: int | None,
        fps: float | None,
        write_geometry: bool,
        include_per_mesh: bool,
    ) -> None:
        base = _pose_record(
            label=label, role=role, pose=pose, transl=transl,
            clip_name=clip_name, frame_id=frame_id, fps=fps,
        )
        try:
            evaluated = _evaluate_frame(
                before=before,
                after=after,
                source_lbs=source_lbs,
                model=model,
                pose=pose,
                transl=transl,
                asset=asset,
                vertex_tissue=vertex_tissue,
                region_ids=region_ids,
                include_per_mesh=include_per_mesh,
            )
            base["status"] = "evaluated"
            base["metrics"] = evaluated["metrics"]
            base["vessel_over_1mm_fail"] = bool(any(
                evaluated["metrics"][variant]["vessels_over_1mm_fail"]
                for variant in ("before", "candidate", "source")
            ))
            base["candidate_vs_before_max_displacement_m"] = float(np.max(
                np.linalg.norm(
                    np.asarray(evaluated["candidate_vertices"], dtype=np.float64)
                    - np.asarray(evaluated["before_vertices"], dtype=np.float64),
                    axis=1,
                )
            ))
            if write_geometry:
                filename = f"{label}.npz"
                path = geometry_root / filename
                base["geometry_file"] = str(path.relative_to(output_root))
                base["geometry_sha256"] = _save_geometry(path, evaluated, base)
                report["geometry_files"][label] = base["geometry_file"]
            else:
                base["geometry_file"] = None
                base["per_mesh_metrics"] = "omitted_for_lightweight_endpoint"
        except Exception as exc:  # preserve an auditable frame-level failure
            base["status"] = "evaluation_error"
            base["error"] = f"{type(exc).__name__}: {exc}"
            report["errors"].append({"label": label, "error": base["error"]})
        report["frames"][label] = base

    evaluate_and_record(
        label="tpose", role="tpose", pose=np.zeros((55, 3), dtype=np.float32),
        transl=np.zeros(3), clip_name=None, frame_id=None, fps=None,
        write_geometry=True, include_per_mesh=True,
    )
    for name, (pose, transl, _info) in captures.items():
        evaluate_and_record(
            label=name, role="capture", pose=pose, transl=transl,
            clip_name=None, frame_id=None, fps=None,
            write_geometry=True, include_per_mesh=True,
        )
    for spec in CLIPS_V15:
        clip = clips[spec.name]
        middle = clip.frame_count // 2
        evaluate_and_record(
            label=f"{spec.name}_middle", role="middle",
            pose=clip.poses[middle], transl=clip.transl[middle],
            clip_name=spec.name, frame_id=int(clip.frame_ids[middle]), fps=clip.fps,
            write_geometry=True, include_per_mesh=True,
        )
        if endpoint_metrics:
            for role, index in (("first", 0), ("last", clip.frame_count - 1)):
                evaluate_and_record(
                    label=f"{spec.name}_{role}", role=role,
                    pose=clip.poses[index], transl=clip.transl[index],
                    clip_name=spec.name, frame_id=int(clip.frame_ids[index]), fps=clip.fps,
                    write_geometry=False, include_per_mesh=False,
                )
        else:
            for role in ("first", "last"):
                label = f"{spec.name}_{role}"
                record = next(item for item in frame_plan if item["label"] == label)
                report["frames"][label] = {
                    **record,
                    "status": "not_evaluated",
                    "reason": "endpoint_metrics_disabled",
                    "geometry_file": None,
                }

    report["evaluated_frame_count"] = int(sum(
        record.get("status") == "evaluated" for record in report["frames"].values()
    ))
    report["geometry_file_count"] = len(report["geometry_files"])
    report["anatomical_passed"] = False
    report_path = output_root / "report.json"
    # Hash the final report bytes in a sidecar.  Keeping the digest outside the
    # JSON avoids a self-referential report hash and makes the recorded value
    # directly checkable with ``sha256sum report.json``.
    report_path.write_text(
        json.dumps(_json_ready(report), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    report_digest = _sha256(report_path)
    (output_root / "report.sha256").write_text(report_digest + "\n", encoding="utf-8")
    report["report_sha256"] = report_digest
    report["report_sha256_sidecar"] = "report.sha256"
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-compiled", type=Path, required=True)
    parser.add_argument("--after-compiled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture-213328", type=Path, default=DEFAULT_CAPTURE_213328)
    parser.add_argument("--capture-213712", type=Path, default=DEFAULT_CAPTURE_213712)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--amass-root", type=Path, default=DEFAULT_AMASS_ROOT)
    parser.add_argument(
        "--endpoint-metrics",
        action="store_true",
        help="evaluate six clip first/last frames without writing endpoint geometry NPZs",
    )
    parser.add_argument(
        "--require-frozen-validation",
        action="store_true",
        help="fail if the immutable V15 validation manifest is absent",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_compiled_pair_v15(
        args.before_compiled,
        args.after_compiled,
        args.output,
        smplx_model=args.smplx_model,
        capture_213328=args.capture_213328,
        capture_213712=args.capture_213712,
        validation_root=args.validation_root,
        amass_root=args.amass_root,
        endpoint_metrics=args.endpoint_metrics,
        require_frozen_validation=args.require_frozen_validation,
    )
    print(json.dumps({
        "output": str(Path(args.output).expanduser().resolve()),
        "geometry_file_count": report.get("geometry_file_count"),
        "evaluated_frame_count": report.get("evaluated_frame_count"),
        "errors": len(report.get("errors", [])),
        "anatomical_passed": False,
    }, ensure_ascii=False, sort_keys=True))
    return 1 if report.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
