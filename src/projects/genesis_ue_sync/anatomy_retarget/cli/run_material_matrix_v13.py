"""Run the once-compiled material transport V13 matrix.

The matrix is deliberately a diagnostic artifact.  It authenticates the
frozen 142 source operator, calibration, male SMPL-X model, and capture pair,
then evaluates the V11/V12e rest pair through one reusable pose map.  Soft
attachments are compiled once per subject and are never searched or fitted in
the pose loop.  Every result keeps ``publishable=false``; a passing numerical
cell is evidence about this experiment, not a release claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.absolute_poke_v12 import (
    absolute_poke_metrics,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.anchored_rest_fit_v11 import (
    build_anchored_rest_fit_v11,
)
from projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_OPERATOR_RUNTIME_DIGEST,
    EXPECTED_ORACLE_SHA256,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_containment_v1 import (
    _signed_distance,
    _vertex_areas,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    _weighted_rest_correction,
)
from projects.genesis_ue_sync.anatomy_retarget.deep_flex_poses_v12 import (
    build_deep_flex_poses_v12,
    verify_deep_flex_poses_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_plausibility_v12 import (
    joint_plausibility_metrics_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.linkage_v12 import (
    evaluate_linkage_v12,
    tube_bone_offset_metrics_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.material_attachment_v13 import (
    compile_material_attachment_map_v13,
)
from projects.genesis_ue_sync.anatomy_retarget.material_runtime_v13 import (
    MaterialRuntimeV13,
    load_material_runtime_v13,
    recover_shared_weight_rest,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import build_pose_map_v1
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import (
    pose_whole_chain_vertices_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
    AnatomyRiggedAsset,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    _smplx_joint_kinematics_v7,
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.validation_poses_v13 import (
    DEFAULT_AMASS_ROOT,
    load_held_out_poses_v13,
    synthetic_betas_v13,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)


# The CLI has one extra ``cli/`` directory compared with the sibling modules;
# repository root is therefore ``parents[5]`` here.
ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
DEFAULT_CALIBRATION = ROOT / (
    "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/"
    "anatomical_calibration_v1"
)
DEFAULT_ORACLE = ROOT / (
    "outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001/"
    "blender_link_oracle_v7.npz"
)
DEFAULT_MODEL = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
DEFAULT_CAPTURE_213328 = ROOT / "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"
DEFAULT_CAPTURE_213712 = ROOT / "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz"
DEFAULT_V11 = ROOT / "outputs/anatomy_retarget/v11_candidates/chain_retarget_v11_anchored_001"
DEFAULT_CANDIDATE = ROOT / (
    "outputs/anatomy_retarget/v12_candidates/"
    "chain_retarget_v12e_forearm_mesh_001"
)

CAPTURE_SHA256 = {
    "213328": "c7a6c3783dc7b764e1f8013ab0a8a45d0380b81c97ac929f67c7a5a526eecbc1",
    "213712": "9887848b7b086d71a875beea50b1d7c7819a11c7b67996fe0d83f451da79b689",
}

TISSUE_CODES = {
    "bone": 0,
    "vessel": 1,
    "nerve": 2,
    "organ": 3,
    "heart": 4,
    "connective": 5,
    "connective_tissue": 5,
}
TISSUE_NAMES = ("bone", "vessel", "nerve", "organ", "heart", "connective")
SELECTED_COMPARISON_POSES = (
    "tpose",
    "pose_213328",
    "heldout_sitting",
    "heldout_kicking",
    "flex_knee_elbow_120",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    """Convert NumPy values into strict JSON values without hiding NaNs."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        if not np.isfinite(result):
            return None
        return result
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _exception_status(exc: BaseException) -> str:
    """Distinguish an expected pose-domain miss from an evaluation bug."""

    text = str(exc).lower()
    support_markers = (
        "outside its support",
        "outside the support",
        "exceeds support",
        "support ball",
        "no local support",
        "out_of_support",
    )
    if any(marker in text for marker in support_markers):
        return "unsupported_pose"
    return "evaluation_error"


def _load_subject_v11(path: Path) -> ChainRestFitSubjectV1:
    """Load an anchored v11/v12e subject without trusting its pass flags."""

    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    npz_path = root / "whole_chain_rest_fit_subject_v11.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The matrix writer labels the directory-level artifact as a
    # ``WholeChainRestFitSubjectV11`` even though ``build_report`` records the
    # inner ``AnchoredRestFitV11`` method.  Authenticate both fields rather
    # than routing this through the older v1 loader, whose manifest contract is
    # intentionally different.
    if manifest.get("artifact_kind") != "WholeChainRestFitSubjectV11":
        raise ValueError(f"{root}: expected WholeChainRestFitSubjectV11 subject")
    if dict(manifest.get("build_report", {})).get("artifact_kind") != "AnchoredRestFitV11":
        raise ValueError(f"{root}: subject build report is not anchored v11")
    if manifest.get("publishable") is not False or manifest.get("trusted_latest_updated") is not False:
        raise ValueError(f"{root}: subject manifest is not frozen non-publishable")
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            "betas", "vertices_prefit", "vertices_final", "faces", "bone_parents",
            "B_prefit", "B_final", "C_bone", "target_local_bind", "inverse_bind",
            "prefit_anatomical_frames", "final_anatomical_frames", "smplx_joints_tpose",
            "station_frame_translation", "centerline_points", "mesh_policy",
            "moved_vertex_ids", "pelvis_cage_vertex_ids", "pelvis_cage_displacements",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{root}: subject NPZ is missing fields {missing}")
        value = ChainRestFitSubjectV1(
            source_operator_digest=str(manifest["source_operator_digest"]),
            calibration_digest=str(manifest["calibration_digest"]),
            source_subject_digest=str(manifest["source_subject_digest"]),
            smplx_model_sha256=str(manifest["smplx_model_sha256"]),
            capture_sha256=str(manifest["capture_sha256"]),
            subject_label=str(manifest["subject_label"]),
            betas=np.asarray(data["betas"], dtype=np.float64),
            vertices_prefit=np.asarray(data["vertices_prefit"], dtype=np.float32),
            vertices_final=np.asarray(data["vertices_final"], dtype=np.float32),
            faces=np.asarray(data["faces"], dtype=np.int32),
            bone_parents=np.asarray(data["bone_parents"], dtype=np.int32),
            B_prefit=np.asarray(data["B_prefit"], dtype=np.float64),
            B_final=np.asarray(data["B_final"], dtype=np.float64),
            C_bone=np.asarray(data["C_bone"], dtype=np.float64),
            target_local_bind=np.asarray(data["target_local_bind"], dtype=np.float64),
            inverse_bind=np.asarray(data["inverse_bind"], dtype=np.float64),
            prefit_anatomical_frames=np.asarray(data["prefit_anatomical_frames"], dtype=np.float64),
            final_anatomical_frames=np.asarray(data["final_anatomical_frames"], dtype=np.float64),
            smplx_joints_tpose=np.asarray(data["smplx_joints_tpose"], dtype=np.float64),
            station_frame_translation=np.asarray(data["station_frame_translation"], dtype=np.float64),
            centerline_points=np.asarray(data["centerline_points"], dtype=np.float64),
            mesh_policy=np.asarray(data["mesh_policy"]).copy(),
            moved_vertex_ids=np.asarray(data["moved_vertex_ids"], dtype=np.int32),
            build_report=dict(manifest.get("build_report", {})),
            pelvis_cage_vertex_ids=np.asarray(data["pelvis_cage_vertex_ids"], dtype=np.int32),
            pelvis_cage_displacements=np.asarray(data["pelvis_cage_displacements"], dtype=np.float64),
        )
    try:
        value.validate()
    except ValueError as exc:
        # V12e intentionally carries the forearm shaft mesh after V11's bind
        # restore.  The old ChainRestFitSubjectV1 validator predates that
        # mesh-only extension and rejects only its copied-vertex policy check;
        # all shape, affine, bind, and index checks above have already run.
        note = str(value.build_report.get("terminal_policy_note", ""))
        if "mesh-only" not in note or "outside its lower bone policy" not in str(exc):
            raise
    return value


def _load_capture(path: Path, *, model_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        if "shapes" not in data.files or "Rh" not in data.files or "poses" not in data.files:
            raise ValueError(f"{path}: EasyMocap capture lacks shapes/Rh/poses")
        betas = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
        pose = easymocap_fit_to_smplx55(
            data["Rh"], data["poses"], model_path=model_path
        )
    return betas, np.asarray(pose, dtype=np.float32), _sha256(path)


def _tissue_ids(asset: Any) -> dict[str, np.ndarray]:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    labels = [str(value).strip().lower() for value in asset.source_tissues]
    result: dict[str, list[np.ndarray]] = {name: [] for name in TISSUE_NAMES}
    for label, (start, stop) in zip(labels, ranges.tolist()):
        canonical = "connective" if label == "connective_tissue" else label
        if canonical in result:
            result[canonical].append(np.arange(int(start), int(stop), dtype=np.int64))
    return {
        name: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        for name, chunks in result.items()
    }


def _bone_surface(asset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-vertex bone faces and their source mesh component labels."""

    faces = np.asarray(asset.faces, dtype=np.int64).reshape(-1, 3)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    names = [str(value) for value in asset.source_mesh_names]
    tissues = [str(value).strip().lower() for value in asset.source_tissues]
    selected: list[int] = []
    components: list[str] = []
    for mesh, ((start, stop), tissue, name) in enumerate(zip(ranges.tolist(), tissues, names)):
        if tissue != "bone":
            continue
        member = np.all((faces >= int(start)) & (faces < int(stop)), axis=1)
        ids = np.flatnonzero(member)
        selected.extend(ids.tolist())
        components.extend([f"mesh_{mesh}:{name}"] * len(ids))
    if not selected:
        raise ValueError("asset has no bone surface faces for material attachments")
    order = np.asarray(selected, dtype=np.int64)
    return faces[order], order, np.asarray(components, dtype="<U128")


def _vertex_tissue_codes(asset: Any) -> np.ndarray:
    values = np.full(len(asset.vertices_rest), -1, dtype=np.int8)
    for label, ids in _tissue_ids(asset).items():
        values[ids] = TISSUE_CODES[label]
    if np.any(values < 0):
        raise ValueError("asset has vertices without a recognised tissue label")
    return values


def _surface_metrics(
    vertices: np.ndarray,
    *,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    asset: Any,
    ids_by_tissue: Mapping[str, np.ndarray],
    area_reference: np.ndarray,
) -> dict[str, Any]:
    """Measure one signed-distance field for all vertices, then slice tissues."""

    signed = np.asarray(_signed_distance(vertices, skin, skin_faces), dtype=np.float64)
    if signed.shape != (len(vertices),) or not np.all(np.isfinite(signed)):
        raise ValueError("signed-distance result is invalid")
    areas = _vertex_areas(np.asarray(area_reference, dtype=np.float64), asset.faces)
    result: dict[str, Any] = {}
    for tissue, ids in ids_by_tissue.items():
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids):
            result[tissue] = {
                "available": False,
                "passed": False,
                "reason": "tissue has no vertices",
            }
            continue
        values = signed[ids]
        weights = np.asarray(areas[ids], dtype=np.float64)
        total_area = float(np.sum(weights))
        outside = values > 0.0
        outside_values = np.maximum(values, 0.0)
        outside_area = (
            float(np.sum(weights[outside]) / total_area)
            if total_area > 0.0
            else float("nan")
        )
        inside_vertices = float(np.mean(~outside))
        inside_area = 1.0 - outside_area if np.isfinite(outside_area) else float("nan")
        max_outside = float(np.max(outside_values))
        passed = bool(
            np.isfinite(outside_area)
            and max_outside <= 0.002
            and inside_vertices >= 0.995
            and inside_area >= 0.995
        )
        result[tissue] = {
            "available": True,
            "passed": passed,
            "vertex_count": int(len(ids)),
            "outside_vertex_count": int(np.count_nonzero(outside)),
            "outside_area_fraction": outside_area,
            "inside_vertex_fraction": inside_vertices,
            "inside_area_fraction": inside_area,
            "max_signed_distance_m": float(np.max(values)),
            "max_outside_m": max_outside,
            "signed_p95_m": float(np.quantile(values, 0.95)),
            "outside_p95_m": float(np.quantile(outside_values, 0.95)),
            "area_weighted_outside_depth_m": float(
                np.sum(weights * outside_values) / total_area
            ) if total_area > 0.0 else float("nan"),
            "gate": {
                "max_outside_m": 0.002,
                "inside_fraction": 0.995,
                "inside_fraction_kind": "vertex_and_rest_area",
            },
        }
    return {
        "available": True,
        "passed": bool(all(bool(row.get("passed")) for row in result.values())),
        "tissues": result,
        "signed_distance_convention": "negative_inside",
        "area_reference": "original_materialized_rest_vertices",
    }


def _tube_edge_ratio_metrics(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    asset: Any,
    minimum_edge_length_m: float = 1.0e-8,
) -> dict[str, Any]:
    faces = np.asarray(asset.faces, dtype=np.int64)
    ids = _tissue_ids(asset)
    tube = np.zeros(len(source), dtype=bool)
    tube[np.concatenate((ids["vessel"], ids["nerve"]))] = True
    mask = np.all(tube[faces], axis=1)
    tube_faces = faces[mask]
    if not len(tube_faces):
        raise ValueError("tube surface has no triangle edges")
    edges = np.concatenate(
        (tube_faces[:, [0, 1]], tube_faces[:, [1, 2]], tube_faces[:, [2, 0]])
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    before = np.linalg.norm(source[edges[:, 1]] - source[edges[:, 0]], axis=1)
    after = np.linalg.norm(candidate[edges[:, 1]] - candidate[edges[:, 0]], axis=1)
    valid = before > float(minimum_edge_length_m)
    if not np.any(valid):
        raise ValueError("all tube edges are below the minimum length threshold")
    ratio = after[valid] / before[valid]
    finite = ratio[np.isfinite(ratio)]
    if len(finite) != len(ratio):
        raise ValueError("tube edge ratios are non-finite")
    return {
        "available": True,
        "passed": bool(
            float(np.min(finite)) >= 0.5 and float(np.max(finite)) <= 2.0
        ),
        "edge_count": int(len(finite)),
        "minimum_source_edge_length_m": float(minimum_edge_length_m),
        "ratio_min": float(np.min(finite)),
        "ratio_p01": float(np.quantile(finite, 0.01)),
        "ratio_median": float(np.median(finite)),
        "ratio_p99": float(np.quantile(finite, 0.99)),
        "ratio_p999": float(np.quantile(finite, 0.999)),
        "ratio_max": float(np.max(finite)),
        "ratio_bounds": [0.5, 2.0],
        "zero_edge_policy": "exclude_source_edges_at_or_below_threshold",
    }


def _pose_joints_and_skin(
    model: Mapping[str, np.ndarray],
    *,
    betas: np.ndarray,
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skin, faces = smplx_body_surface_v7(model, betas=betas, pose_axis_angle=pose)
    _rest_joints, globals_, _rest_to_pose = _smplx_joint_kinematics_v7(
        model, betas=betas, pose_axis_angle=pose
    )
    joints = np.asarray(globals_, dtype=np.float64)[:, :3, 3]
    return np.asarray(skin, dtype=np.float64), np.asarray(faces, dtype=np.int32), joints


def _make_capture_poses(
    captures: Mapping[str, np.ndarray],
    *,
    model: Mapping[str, np.ndarray],
    betas: np.ndarray,
    target_deg: float = 120.0,
) -> dict[str, np.ndarray]:
    def joints_of(pose: np.ndarray) -> np.ndarray:
        return _pose_joints_and_skin(model, betas=betas, pose=pose)[2]

    deep = build_deep_flex_poses_v12(
        captures=captures, joints_of=joints_of, target_deg=float(target_deg)
    )
    checked = verify_deep_flex_poses_v12(
        deep, joints_of=joints_of, target_deg=float(target_deg)
    )
    if not checked.get("passed"):
        raise ValueError(f"deep-flex pose verification failed: {checked.get('failures')}")
    return {
        "tpose": np.zeros((55, 3), dtype=np.float32),
        "pose_213328": np.asarray(captures["213328"], dtype=np.float32),
        "pose_213712": np.asarray(captures["213712"], dtype=np.float32),
        **{name: np.asarray(pose, dtype=np.float32) for name, pose in deep.items()},
    }


def _compile_attachment(
    asset: AnatomyRiggedAsset,
    target_rest: np.ndarray,
    soft_ids: np.ndarray,
    *, guided: bool = True,
) -> Any:
    bone_faces, _face_ids, components = _bone_surface(asset)
    guide_args = dict(
        point_driver_indices=np.asarray(asset.driver_indices)[soft_ids],
        point_driver_weights=np.asarray(asset.driver_weights)[soft_ids],
        surface_driver_indices=asset.driver_indices,
        surface_driver_weights=asset.driver_weights,
    ) if guided else {}
    return compile_material_attachment_map_v13(
        np.asarray(asset.vertices_rest, dtype=np.float64),
        bone_faces,
        np.asarray(target_rest, dtype=np.float64),
        bone_faces,
        np.asarray(asset.vertices_rest, dtype=np.float64)[soft_ids],
        source_face_component_ids=components,
        target_face_component_ids=components,
        max_components=4,
        **guide_args,
    )


def _save_comparison(
    path: Path,
    *,
    label: str,
    pose: np.ndarray,
    faces: np.ndarray,
    skin_faces: np.ndarray,
    source_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    skin_vertices: np.ndarray,
    smplx_joints: np.ndarray,
    vertex_tissue: np.ndarray,
    weights_vertices: np.ndarray,
    raw_vertices: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        faces=np.asarray(faces, dtype=np.int32),
        skin_faces=np.asarray(skin_faces, dtype=np.int32),
        source_vertices=np.asarray(source_vertices, dtype=np.float32),
        candidate_vertices=np.asarray(candidate_vertices, dtype=np.float32),
        skin_vertices=np.asarray(skin_vertices, dtype=np.float32),
        smplx_joints=np.asarray(smplx_joints, dtype=np.float32),
        pose=np.asarray(pose, dtype=np.float32),
        vertex_tissue=np.asarray(vertex_tissue, dtype=np.int8),
        weights_vertices=np.asarray(weights_vertices, dtype=np.float32),
        raw_vertices=np.asarray(raw_vertices, dtype=np.float32),
        metadata_label=np.asarray(str(label)),
        metadata_json=np.asarray(json.dumps(_json_ready(metadata), sort_keys=True, allow_nan=False)),
    )


def _subject_paths(root: Path, subject: str) -> Path:
    return Path(root).expanduser().resolve() / f"subject_{subject}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture-213328", type=Path, default=DEFAULT_CAPTURE_213328)
    parser.add_argument("--capture-213712", type=Path, default=DEFAULT_CAPTURE_213712)
    parser.add_argument("--v11-shadow", type=Path, default=DEFAULT_V11)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--amass-hf-root", type=Path, default=DEFAULT_AMASS_ROOT)
    parser.add_argument("--subjects", default="213328,213712")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="also build fresh synthetic_pc1_negative and synthetic_pc2_positive V11 stress subjects",
    )
    parser.add_argument(
        "--no-attachments",
        action="store_true",
        help="skip one-shot material attachment compilation (weights-only diagnostic)",
    )
    parser.add_argument("--deep-flex-deg", type=float, default=120.0)
    parser.add_argument("--unguided-attachments", action="store_true",
                        help="ablation: select material attachments by surface distance alone")
    return parser


def _auth_subject(
    value: ChainRestFitSubjectV1,
    *,
    subject: str,
    capture_sha: str | None,
    capture_betas: np.ndarray | None,
    operator_digest: str,
    calibration_digest: str,
    model_sha: str,
) -> None:
    if value.source_operator_digest != operator_digest:
        raise ValueError(f"{subject}: source operator digest differs")
    if value.calibration_digest != calibration_digest:
        raise ValueError(f"{subject}: calibration digest differs")
    if value.smplx_model_sha256 != model_sha:
        raise ValueError(f"{subject}: SMPL-X model digest differs")
    if capture_sha is not None and value.capture_sha256 != capture_sha:
        raise ValueError(f"{subject}: capture digest differs")
    if capture_betas is not None and not np.allclose(
        np.asarray(value.betas, dtype=np.float64), np.asarray(capture_betas, dtype=np.float64), atol=1.0e-7, rtol=0.0
    ):
        raise ValueError(f"{subject}: candidate betas differ from capture betas")


def _build_synthetic_subject(
    label: str,
    betas: np.ndarray,
    *,
    operator: Any,
    calibration: Any,
    model: Mapping[str, np.ndarray],
    model_sha: str,
    flex_pose: np.ndarray,
) -> tuple[ChainRestFitSubjectV1, ChainRestFitSubjectV1]:
    """Build a fresh V1 rest fit and anchor it as a V11 stress subject."""

    from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
        build_whole_chain_rest_fit_v1,
    )

    digest = hashlib.sha256(np.asarray(betas, dtype=np.float32).tobytes()).hexdigest()
    fresh = build_whole_chain_rest_fit_v1(
        operator,
        calibration,
        betas=betas,
        subject_label=f"{label}_fresh_v1_no_reseat",
        capture_sha256=f"synthetic:{digest}",
        smplx_model=model,
        smplx_model_sha256=model_sha,
        gender="male",
        containment_pose_axis_angle=flex_pose,
    )
    anchored = build_anchored_rest_fit_v11(
        fresh,
        asset=materialize_subject(operator, betas=betas, gender="male").rigged_asset,
        calibration=calibration,
        carry_mesh_controllers=(),
    )
    anchored_report = dict(anchored.build_report)
    anchored_report.update(
        {
            "synthetic_stress_subject": True,
            "synthetic_source": "fresh_v1_then_anchored_v11_no_reseat_optimize",
            "publishable": False,
        }
    )
    from dataclasses import replace

    anchored = replace(anchored, build_report=anchored_report)
    return fresh, anchored


def _run_subject(
    subject: str,
    *,
    args: argparse.Namespace,
    operator: Any,
    calibration: Any,
    model: Mapping[str, np.ndarray],
    model_sha: str,
    oracle: Path,
    operator_digest: str,
    calibration_digest: str,
    capture_betas: Mapping[str, np.ndarray],
    capture_poses: Mapping[str, np.ndarray],
    heldout_poses: Mapping[str, np.ndarray],
    synthetic: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if synthetic:
        # Synthetic labels are beta keys, not capture identities.
        beta = np.asarray(synthetic_betas_v13()[subject], dtype=np.float64)
        base, candidate = _build_synthetic_subject(
            subject,
            beta,
            operator=operator,
            calibration=calibration,
            model=model,
            model_sha=model_sha,
            flex_pose=np.asarray(capture_poses["213328"], dtype=np.float32),
        )
        subject_provenance = {
            "kind": "synthetic_v11_stress",
            "beta_key": subject,
            "fresh_v1_then_anchored_v11_no_reseat_optimize": True,
        }
    else:
        base = _load_subject_v11(_subject_paths(args.v11_shadow, subject))
        candidate = _load_subject_v11(_subject_paths(args.candidate, subject))
        _auth_subject(
            base,
            subject=f"v11:{subject}",
            capture_sha=CAPTURE_SHA256[subject],
            capture_betas=capture_betas[subject],
            operator_digest=operator_digest,
            calibration_digest=calibration_digest,
            model_sha=model_sha,
        )
        _auth_subject(
            candidate,
            subject=f"candidate:{subject}",
            capture_sha=CAPTURE_SHA256[subject],
            capture_betas=capture_betas[subject],
            operator_digest=operator_digest,
            calibration_digest=calibration_digest,
            model_sha=model_sha,
        )
        beta = np.asarray(candidate.betas, dtype=np.float64)
        subject_provenance = {"kind": "capture", "capture_subject": subject}

    if not np.array_equal(base.faces, candidate.faces):
        raise ValueError(f"{subject}: v11 and candidate topology differ")
    materialized = materialize_subject(operator, betas=beta, gender="male")
    asset = materialized.rigged_asset
    if not np.array_equal(asset.faces, candidate.faces):
        raise ValueError(f"{subject}: candidate topology differs from source operator")
    target_rest, rest_report = recover_shared_weight_rest(base, candidate, asset)
    pose_map = build_pose_map_v1(
        candidate,
        asset=asset,
        calibration=calibration,
        oracle_path=oracle,
        source_operator_digest=operator_digest,
    )
    soft_ids = np.flatnonzero(
        np.isin(_vertex_tissue_codes(asset), [1, 2, 3, 4, 5])
    ).astype(np.int64)
    attachment = None
    attachment_report: dict[str, Any] = {"enabled": False, "reason": "--no-attachments"}
    if not args.no_attachments:
        attachment_started = time.perf_counter()
        attachment = _compile_attachment(asset, target_rest, soft_ids,
                                         guided=not args.unguided_attachments)
        attachment_report = {
            "enabled": True,
            "compile_seconds": float(time.perf_counter() - attachment_started),
            "report": dict(attachment.report),
        }
    runtime = MaterialRuntimeV13(
        source_asset=asset,
        pose_map=pose_map,
        target_rest=np.asarray(target_rest, dtype=np.float64),
        soft_ids=soft_ids,
        attachment=attachment,
    )

    runtime_root = Path(args.output).expanduser().resolve() / "subjects" / f"subject_{subject}" / "runtime"
    runtime_root.parent.mkdir(parents=True, exist_ok=True)
    runtime.save(
        runtime_root,
        provenance={
            "subject": subject,
            "candidate": str(args.candidate),
            "v11_shadow": str(args.v11_shadow),
            "source_operator_digest": operator_digest,
            "calibration_digest": calibration_digest,
            "smplx_model_sha256": model_sha,
            "publishable": False,
            **subject_provenance,
        },
    )
    reloaded = load_material_runtime_v13(runtime_root)

    poses = {
        "tpose": np.zeros((55, 3), dtype=np.float32),
        "pose_213328": np.asarray(capture_poses["213328"], dtype=np.float32),
        "pose_213712": np.asarray(capture_poses["213712"], dtype=np.float32),
        **{name: np.asarray(value, dtype=np.float32) for name, value in heldout_poses.items()},
    }
    deep_poses = _make_capture_poses(
        {"213328": capture_poses["213328"], "213712": capture_poses["213712"]},
        model=model,
        betas=beta,
        target_deg=float(args.deep_flex_deg),
    )
    for name, pose in deep_poses.items():
        if name.startswith("flex_"):
            poses[name] = pose

    ids_by_tissue = _tissue_ids(asset)
    tissue_codes = _vertex_tissue_codes(asset)
    area_reference = np.asarray(asset.vertices_rest, dtype=np.float64)
    cells: dict[str, Any] = {}
    selected_cache: dict[str, dict[str, Any]] = {}
    source_linkage: dict[str, dict[str, Any]] = {}
    before_linkage: dict[str, dict[str, Any]] = {}
    weights_linkage: dict[str, dict[str, Any]] = {}
    candidate_linkage: dict[str, dict[str, Any]] = {}
    replay: dict[str, Any] = {}
    absolute_poke: dict[str, Any] | None = None
    for pose_name, pose in poses.items():
        pose_started = time.perf_counter()
        try:
            raw_vertices = runtime.apply_pose(pose, mode="source")
            weights_vertices = runtime.apply_pose(pose, mode="weights")
            selected_mode = "attachments" if attachment is not None else "weights"
            candidate_vertices = runtime.apply_pose(pose, mode=selected_mode)
            before_vertices, target_globals = pose_whole_chain_vertices_v10(
                candidate,
                pose_map,
                source_asset=asset,
                pose_axis_angle=pose,
            )
            skin, skin_faces, joints = _pose_joints_and_skin(
                model, betas=beta, pose=pose
            )
            before_tissue_metrics = _surface_metrics(
                before_vertices,
                skin=skin,
                skin_faces=skin_faces,
                asset=asset,
                ids_by_tissue=ids_by_tissue,
                area_reference=area_reference,
            )
            weights_tissue_metrics = _surface_metrics(
                weights_vertices,
                skin=skin,
                skin_faces=skin_faces,
                asset=asset,
                ids_by_tissue=ids_by_tissue,
                area_reference=area_reference,
            )
            if np.array_equal(candidate_vertices, weights_vertices):
                tissue_metrics = weights_tissue_metrics
            else:
                tissue_metrics = _surface_metrics(
                    candidate_vertices,
                    skin=skin,
                    skin_faces=skin_faces,
                    asset=asset,
                    ids_by_tissue=ids_by_tissue,
                    area_reference=area_reference,
                )
            target_joints = joint_plausibility_metrics_v12(
                candidate_vertices,
                calibration=calibration,
                smplx_joints=joints,
            )
            raw_joints = joint_plausibility_metrics_v12(
                raw_vertices,
                calibration=calibration,
                smplx_joints=joints,
            )
            source_linkage[pose_name] = tube_bone_offset_metrics_v12(
                np.asarray(asset.vertices_rest, dtype=np.float64),
                raw_vertices,
                asset=asset,
            )
            before_linkage[pose_name] = tube_bone_offset_metrics_v12(
                np.asarray(asset.vertices_rest, dtype=np.float64),
                before_vertices,
                asset=asset,
            )
            weights_linkage[pose_name] = tube_bone_offset_metrics_v12(
                np.asarray(asset.vertices_rest, dtype=np.float64),
                weights_vertices,
                asset=asset,
            )
            candidate_linkage[pose_name] = tube_bone_offset_metrics_v12(
                np.asarray(asset.vertices_rest, dtype=np.float64),
                candidate_vertices,
                asset=asset,
            )
            edge_ratio = _tube_edge_ratio_metrics(
                raw_vertices, candidate_vertices, asset=asset
            )
            if absolute_poke is None and pose_name == "tpose":
                absolute_poke = {
                    "pose": pose_name,
                    "candidate": absolute_poke_metrics(
                        candidate_vertices,
                        asset=asset,
                        skin=skin,
                        skin_faces=skin_faces,
                        area_reference=area_reference,
                    ),
                }
            if pose_name in SELECTED_COMPARISON_POSES:
                selected_cache[pose_name] = {
                    "pose": pose,
                    "faces": asset.faces,
                    "skin_faces": skin_faces,
                    "source_vertices": before_vertices,
                    "candidate_vertices": candidate_vertices,
                    "skin_vertices": skin,
                    "smplx_joints": joints,
                    "weights_vertices": weights_vertices,
                    "raw_vertices": raw_vertices,
                }
            cells[pose_name] = {
                "available": True,
                "passed": bool(tissue_metrics.get("passed")) and bool(edge_ratio.get("passed")),
                "selected_mode": selected_mode,
                "before_v12e": before_tissue_metrics,
                "weights": weights_tissue_metrics,
                "tissue_metrics": tissue_metrics,
                "joint_plausibility_target": target_joints,
                "joint_plausibility_raw": raw_joints,
                "tube_linkage_source": source_linkage[pose_name],
                "tube_linkage_before_v12e": before_linkage[pose_name],
                "tube_linkage_weights": weights_linkage[pose_name],
                "tube_linkage_candidate": candidate_linkage[pose_name],
                "tube_edge_ratio": edge_ratio,
                "elapsed_seconds": float(time.perf_counter() - pose_started),
            }
            if pose_name in {"heldout_sitting", "heldout_kicking"}:
                before = runtime.apply_pose(pose, mode="weights")
                after = reloaded.apply_pose(pose, mode="weights")
                replay[f"{pose_name}:weights"] = {
                    "exact": bool(np.array_equal(before, after)),
                    "max_abs_error_m": float(np.max(np.abs(before.astype(np.float64) - after.astype(np.float64)))),
                }
                if attachment is not None:
                    before = runtime.apply_pose(pose, mode="attachments")
                    after = reloaded.apply_pose(pose, mode="attachments")
                    replay[f"{pose_name}:attachments"] = {
                        "exact": bool(np.array_equal(before, after)),
                        "max_abs_error_m": float(np.max(np.abs(before.astype(np.float64) - after.astype(np.float64)))),
                    }
            _write_json(
                Path(args.output).expanduser().resolve() / "progress.json",
                {
                    "schema_version": 13,
                    "artifact_kind": "MaterialMatrixV13Progress",
                    "publishable": False,
                    "subject": subject,
                    "pose": pose_name,
                    "status": "complete",
                    "cells": cells,
                },
            )
            print(f"subject={subject} pose={pose_name} mode={selected_mode} pass={cells[pose_name]['passed']}", flush=True)
        except Exception as exc:  # every unsupported/error cell is an explicit failure
            status = _exception_status(exc)
            cells[pose_name] = {
                "available": False,
                "passed": False,
                "status": status,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": float(time.perf_counter() - pose_started),
            }
            _write_json(
                Path(args.output).expanduser().resolve() / "progress.json",
                {
                    "schema_version": 13,
                    "artifact_kind": "MaterialMatrixV13Progress",
                    "publishable": False,
                    "subject": subject,
                    "pose": pose_name,
                    "status": status,
                    "cells": cells,
                },
            )
            print(f"subject={subject} pose={pose_name} status={status} error={type(exc).__name__}", flush=True)

    comparison_root = Path(args.output).expanduser().resolve() / "subjects" / f"subject_{subject}" / "comparisons"
    metadata_base = {
        "schema_version": 13,
        "artifact_kind": "MaterialMatrixComparisonV13",
        "subject": subject,
        "selected_mode": "attachments" if attachment is not None else "weights",
        "publishable": False,
        "source_label": "BEFORE_V12E_pose_whole_chain_vertices_v10",
        "candidate_label": "attachments" if attachment is not None else "shared_weight_rest",
        **subject_provenance,
    }
    for pose_name, arrays in selected_cache.items():
        _save_comparison(
            comparison_root / f"{pose_name}.npz",
            label=pose_name,
            metadata=metadata_base,
            **arrays,
            vertex_tissue=tissue_codes,
        )
    linkage_gate = evaluate_linkage_v12(
        candidate_linkage,
        reference=source_linkage,
        reference_topology_digest=(
            next(iter(source_linkage.values()))["topology_digest"] if source_linkage else None
        ),
    ) if source_linkage else {
        "available": False, "passed": False, "reason": "no linkage cells"
    }
    pass_cells = bool(cells) and all(bool(cell.get("passed")) for cell in cells.values())
    replay_passed = bool(replay) and all(bool(row.get("exact")) for row in replay.values())
    return {
        "subject": subject,
        "synthetic": synthetic,
        "betas": beta.tolist(),
        "rest_recovery": rest_report,
        "attachment": attachment_report,
        "runtime_root": str(runtime_root),
        "runtime_reload": {"exact_heldout_replay": replay_passed, "cells": replay},
        "absolute_poke_bones_once": absolute_poke,
        "linkage_gate": linkage_gate,
        "cells": cells,
        "passed": bool(pass_cells and linkage_gate.get("passed", False) and replay_passed),
        "publishable": False,
        # Publication and source-baseline quality claims stay closed even when
        # every candidate diagnostic gate happens to pass.
        "source_baseline_quality_claim": False,
        "diagnostic_cells_passed": pass_cells,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite matrix output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 13,
        "artifact_kind": "MaterialMatrixV13",
        "publishable": False,
        "trusted_latest_updated": False,
        "source_baseline_quality_claim": False,
        "synthetic_requested": bool(args.synthetic),
        "subjects": {},
        "failures": [],
    }
    try:
        operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
        operator_digest = operator.runtime_digest(validate=False)
        if operator_digest != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError("matrix requires frozen rebuild_012 source operator")
        oracle = args.oracle.expanduser().resolve()
        if _sha256(oracle) != EXPECTED_ORACLE_SHA256:
            raise ValueError("matrix requires frozen Blender link oracle")
        model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
        model = load_smplx_model_v7(model_path)
        calibration = load_anatomical_calibration_v1(
            args.calibration.expanduser().resolve(),
            operator=operator,
            required_scope="full_main_chain",
        )
        calibration_digest = _calibration_content_digest(calibration)
        if calibration.source_operator_digest != operator_digest:
            raise ValueError("calibration is bound to a different source operator")
        capture_paths = {
            "213328": args.capture_213328.expanduser().resolve(),
            "213712": args.capture_213712.expanduser().resolve(),
        }
        capture_betas: dict[str, np.ndarray] = {}
        capture_poses: dict[str, np.ndarray] = {}
        capture_digests: dict[str, str] = {}
        for label, path in capture_paths.items():
            betas, pose, digest = _load_capture(path, model_path=model_path)
            if digest != CAPTURE_SHA256[label]:
                raise ValueError(f"capture {label} differs from frozen digest")
            capture_betas[label] = betas
            capture_poses[label] = pose
            capture_digests[label] = digest
        heldout, heldout_provenance = load_held_out_poses_v13(args.amass_hf_root)
        report["provenance"] = {
            "operator": str(args.operator.expanduser().resolve()),
            "operator_runtime_digest": operator_digest,
            "calibration": str(args.calibration.expanduser().resolve()),
            "calibration_digest": calibration_digest,
            "oracle": str(oracle),
            "oracle_sha256": _sha256(oracle),
            "smplx_model": str(model_path),
            "smplx_model_sha256": model_sha,
            "captures": {label: {"path": str(capture_paths[label]), "sha256": capture_digests[label]} for label in capture_paths},
            "held_out": heldout_provenance,
            "candidate": str(args.candidate.expanduser().resolve()),
            "v11_shadow": str(args.v11_shadow.expanduser().resolve()),
        }
        subjects = [part.strip() for part in str(args.subjects).split(",") if part.strip()]
        unknown = sorted(set(subjects) - set(capture_betas))
        if unknown:
            raise ValueError(f"unknown capture subjects: {unknown}")
        # Pose generation is subject-specific because shaped SMPL-X joints set
        # the bisection target.  Keep the held-out rotations shared.
        for subject in subjects:
            try:
                row = _run_subject(
                    subject,
                    args=args,
                    operator=operator,
                    calibration=calibration,
                    model=model,
                    model_sha=model_sha,
                    oracle=oracle,
                    operator_digest=operator_digest,
                    calibration_digest=calibration_digest,
                    capture_betas=capture_betas,
                    capture_poses=capture_poses,
                    heldout_poses=heldout,
                )
                report["subjects"][subject] = row
                _write_json(output / "progress.json", report)
            except Exception as exc:
                status = _exception_status(exc)
                failure = {
                    "subject": subject,
                    "status": status,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report["failures"].append(failure)
                report["subjects"][subject] = failure
                _write_json(output / "progress.json", report)
                print(f"subject={subject} status={status} error={type(exc).__name__}", flush=True)
        if args.synthetic:
            synthetic_keys = ("synthetic_pc1_negative", "synthetic_pc2_positive")
            for key in synthetic_keys:
                try:
                    row = _run_subject(
                        key,
                        args=args,
                        operator=operator,
                        calibration=calibration,
                        model=model,
                        model_sha=model_sha,
                        oracle=oracle,
                        operator_digest=operator_digest,
                        calibration_digest=calibration_digest,
                        capture_betas=capture_betas,
                        capture_poses=capture_poses,
                        heldout_poses=heldout,
                        synthetic=True,
                    )
                    report["subjects"][key] = row
                    _write_json(output / "progress.json", report)
                except Exception as exc:
                    status = _exception_status(exc)
                    failure = {
                        "subject": key,
                        "status": status,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    report["failures"].append(failure)
                    report["subjects"][key] = failure
                    _write_json(output / "progress.json", report)
                    print(f"subject={key} status={status} error={type(exc).__name__}", flush=True)
    except Exception as exc:
        status = _exception_status(exc)
        report["failures"].append({"status": status, "error": f"{type(exc).__name__}: {exc}"})
        report["status"] = status
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        _write_json(output / "matrix_manifest.json", report)
        print(f"matrix status={status} error={type(exc).__name__}", flush=True)
        return 1
    report["status"] = "complete"
    report["passed"] = bool(
        report["subjects"]
        and not report["failures"]
        and all(bool(row.get("passed")) for row in report["subjects"].values())
    )
    report["elapsed_seconds"] = float(time.perf_counter() - started)
    _write_json(output / "matrix_manifest.json", report)
    _write_json(output / "progress.json", report)
    print(
        f"matrix subjects={len(report['subjects'])} failures={len(report['failures'])} publishable=false",
        flush=True,
    )
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
