"""Evaluate the bounded V13 knee station-cage hypothesis on frozen V12e data.

This is an offline feasibility experiment.  It applies each shortening level
to the same V12e rest value, poses the result with ``pose_whole_chain_vertices_v10``,
and measures the femur/shank/patella against the subject's posed SMPL-X skin.
The report stays non-publishable: station motion does not solve soft/material
linkage, which still requires the material attachment mapping experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

# Keep numerical libraries single threaded when this CLI is launched without
# an explicit shell environment.  These are process-wide and must be set
# before importing NumPy/SciPy.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.spatial import cKDTree

from projects.genesis_ue_sync.anatomy_retarget.absolute_poke_v12 import (
    absolute_poke_metrics,
    bone_mesh_group_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_OPERATOR_RUNTIME_DIGEST,
    EXPECTED_ORACLE_SHA256,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_material_matrix_v13 import (
    _load_capture,
    _load_subject_v11,
    _make_capture_poses,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_station_cage_v13 import (
    CAP_END,
    apply_knee_station_cage_v13,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import (
    build_pose_map_v10,
    pose_whole_chain_vertices_v10,
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
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)


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
DEFAULT_CANDIDATE = ROOT / (
    "outputs/anatomy_retarget/v12_candidates/"
    "chain_retarget_v12e_forearm_mesh_001"
)
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v13_station_cage_20260908_001"

CAPTURE_SHA256 = {
    "213328": "c7a6c3783dc7b764e1f8013ab0a8a45d0380b81c97ac929f67c7a5a526eecbc1",
    "213712": "9887848b7b086d71a875beea50b1d7c7819a11c7b67996fe0d83f451da79b689",
}

SHORTENING_MM = (0, 10, 20, 30, 35)
POSE_NAMES = ("tpose", "pose_213328", "pose_213712", "bilateral_knee_120", "heldout_sitting")
GROUPS = tuple(f"{part}_{side}" for side in ("L", "R") for part in ("femur", "shank", "patella"))
JOINT_INDEX = {"L": {"hip": 1, "knee": 4, "ankle": 7}, "R": {"hip": 2, "knee": 5, "ankle": 8}}
CONTROLLER_NAME = {
    "L": {"hip": "Femur_Rot_L", "knee": "Knee_Rotate_L", "ankle": "Ankle_Rot_L"},
    "R": {"hip": "Femur_Rot_R", "knee": "Knee_Rotate_R", "ankle": "Ankle_Rot_R"},
}


class CellTimeout(TimeoutError):
    """Raised when one pose/level cell exceeds the configured wall time."""


@contextlib.contextmanager
def _deadline(seconds: float) -> Iterator[None]:
    """Bound one cell on Unix without hiding a timeout as an evaluation error."""

    limit = float(seconds)
    if limit <= 0.0 or not hasattr(signal, "setitimer"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise(_signum: int, _frame: Any) -> None:
        raise CellTimeout(f"cell exceeded timeout of {limit:.3f} seconds")

    signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _status_for_exception(exc: BaseException) -> str:
    if isinstance(exc, CellTimeout):
        return "timeout"
    text = str(exc).lower()
    if "unsupported" in text or "outside the support" in text or "support ball" in text:
        return "unsupported_pose"
    if "jacobian" in text or "shortening" in text or "station cage" in text:
        return "rejected_geometry"
    return "evaluation_error"


def _bone_ids(asset: Any, group: str) -> np.ndarray:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    ids: list[np.ndarray] = []
    for name, tissue, (start, stop) in zip(
        asset.source_mesh_names, asset.source_tissues, ranges.tolist()
    ):
        if str(tissue).strip().lower() != "bone":
            continue
        if bone_mesh_group_v12(str(name)) == group:
            ids.append(np.arange(int(start), int(stop), dtype=np.int64))
    return np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)


def _cap_ids(
    rest: np.ndarray,
    ids: np.ndarray,
    proximal: np.ndarray,
    distal: np.ndarray,
    *,
    proximal_cap: bool,
) -> np.ndarray:
    axis_vector = np.asarray(distal, dtype=np.float64) - np.asarray(proximal, dtype=np.float64)
    length = float(np.linalg.norm(axis_vector))
    if length <= 1.0e-12:
        return np.empty(0, dtype=np.int64)
    axis = axis_vector / length
    t = ((np.asarray(rest)[ids] - np.asarray(proximal)) @ axis) / length
    threshold = 1.0 - CAP_END if proximal_cap else CAP_END
    mask = t <= threshold if proximal_cap else t >= threshold
    return np.asarray(ids[mask], dtype=np.int64)


def _nearest_stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64).reshape(-1, 3)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 3)
    if not len(left) or not len(right):
        return {"available": False, "reason": "empty_cap"}
    left_to_right = cKDTree(right).query(left, k=1)[0]
    right_to_left = cKDTree(left).query(right, k=1)[0]
    distances = np.concatenate((np.asarray(left_to_right), np.asarray(right_to_left)))
    if not np.all(np.isfinite(distances)):
        raise ValueError("seam distances contain non-finite values")
    return {
        "available": True,
        "min_m": float(np.min(distances)),
        "p50_m": float(np.quantile(distances, 0.50)),
        "p95_m": float(np.quantile(distances, 0.95)),
        "max_m": float(np.max(distances)),
        "sample_count": int(len(distances)),
    }


def _seam_metrics(
    vertices: np.ndarray,
    *,
    cap_ids: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("L", "R"):
        femur = np.asarray(cap_ids[side]["femur_distal"], dtype=np.int64)
        shank = np.asarray(cap_ids[side]["shank_proximal"], dtype=np.int64)
        patella = np.asarray(cap_ids[side]["patella"], dtype=np.int64)
        result[side] = {
            "femur_shank": _nearest_stats(vertices[femur], vertices[shank]),
            "femur_patella": _nearest_stats(vertices[femur], vertices[patella]),
            "shank_patella": _nearest_stats(vertices[shank], vertices[patella]),
        }
    return result


def _joint_station_metrics(
    target_globals: np.ndarray,
    smplx_joints: np.ndarray,
    *,
    bone_names: list[str],
) -> dict[str, Any]:
    globals_ = np.asarray(target_globals, dtype=np.float64)
    joints = np.asarray(smplx_joints, dtype=np.float64)
    result: dict[str, Any] = {}
    for side in ("L", "R"):
        side_result: dict[str, Any] = {}
        for station in ("hip", "knee", "ankle"):
            controller = CONTROLLER_NAME[side][station]
            index = bone_names.index(controller)
            joint_index = JOINT_INDEX[side][station]
            station_point = globals_[index, :3, 3]
            target_point = joints[joint_index]
            difference = station_point - target_point
            side_result[station] = {
                "controller": controller,
                "smplx_joint_index": int(joint_index),
                "station_position_m": station_point.tolist(),
                "smplx_position_m": target_point.tolist(),
                "distance_m": float(np.linalg.norm(difference)),
                "signed_delta_m": difference.tolist(),
            }
        result[side] = side_result
    return result


def _tissue_codes(asset: Any) -> np.ndarray:
    code_by_tissue = {
        "bone": 0,
        "vessel": 1,
        "nerve": 2,
        "organ": 3,
        "heart": 4,
        "connective": 5,
        "connective_tissue": 5,
    }
    codes = np.full(len(asset.vertices_rest), -1, dtype=np.int8)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    for tissue, (start, stop) in zip(asset.source_tissues, ranges.tolist()):
        key = str(tissue).strip().lower()
        if key not in code_by_tissue:
            raise ValueError(f"unrecognised asset tissue {tissue!r}")
        codes[int(start) : int(stop)] = code_by_tissue[key]
    if np.any(codes < 0):
        raise ValueError("asset has vertices without a tissue code")
    return codes


def _pose_skin_joints(
    model: Mapping[str, np.ndarray], *, betas: np.ndarray, pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skin, skin_faces = smplx_body_surface_v7(model, betas=betas, pose_axis_angle=pose)
    _rest_joints, globals_, _rest_to_pose = _smplx_joint_kinematics_v7(
        model, betas=betas, pose_axis_angle=pose
    )
    joints = np.asarray(globals_, dtype=np.float64)[:, :3, 3]
    return np.asarray(skin, dtype=np.float64), np.asarray(skin_faces, dtype=np.int32), joints


def _station_caps(value: Any, asset: Any) -> dict[str, dict[str, np.ndarray]]:
    rest = np.asarray(value.vertices_final, dtype=np.float64)
    bind = np.asarray(value.B_final, dtype=np.float64)
    names = [str(name) for name in asset.source_bone_names]
    caps: dict[str, dict[str, np.ndarray]] = {}
    for side in ("L", "R"):
        hip = bind[names.index(CONTROLLER_NAME[side]["hip"]), :3, 3]
        knee = bind[names.index(CONTROLLER_NAME[side]["knee"]), :3, 3]
        ankle = bind[names.index(CONTROLLER_NAME[side]["ankle"]), :3, 3]
        femur_ids = _bone_ids(asset, f"femur_{side}")
        shank_ids = _bone_ids(asset, f"shank_{side}")
        patella_ids = _bone_ids(asset, f"patella_{side}")
        caps[side] = {
            "femur_distal": _cap_ids(rest, femur_ids, hip, knee, proximal_cap=False),
            "shank_proximal": _cap_ids(rest, shank_ids, knee, ankle, proximal_cap=True),
            "patella": patella_ids,
        }
    return caps


def _group_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in GROUPS:
        if group not in metrics:
            result[group] = {"available": False, "reason": "group_missing"}
            continue
        row = dict(metrics[group])
        count = int(row.get("vertex_count", 0))
        outside = int(row.get("outside_count", 0))
        row["outside_vertex_fraction"] = float(outside / count) if count else None
        row["available"] = True
        result[group] = row
    return result


def _level_score(level_row: Mapping[str, Any]) -> tuple[float, float]:
    max_values: list[float] = []
    p95_values: list[float] = []
    for cell in level_row.get("poses", {}).values():
        if cell.get("status") != "complete":
            return (float("inf"), float("inf"))
        for row in cell.get("groups", {}).values():
            if not row.get("available"):
                return (float("inf"), float("inf"))
            max_values.append(float(row["max_outside_m"]))
            p95_values.append(float(row["outside_p95_m"]))
    if not max_values:
        return (float("inf"), float("inf"))
    return (max(max_values), max(p95_values))


def _save_comparison(
    path: Path,
    *,
    pose_name: str,
    pose: np.ndarray,
    faces: np.ndarray,
    skin_faces: np.ndarray,
    source_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    skin_vertices: np.ndarray,
    smplx_joints: np.ndarray,
    vertex_tissue: np.ndarray,
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
        pose_name=np.asarray(str(pose_name)),
        metadata_json=np.asarray(json.dumps(_json_ready(metadata), sort_keys=True, allow_nan=False)),
    )


def _render_arrays(
    value: Any,
    pose_map: Any,
    *,
    asset: Any,
    model: Mapping[str, np.ndarray],
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_vertices, posed_globals = pose_whole_chain_vertices_v10(
        value, pose_map, source_asset=asset, pose_axis_angle=pose
    )
    skin, skin_faces, joints = _pose_skin_joints(
        model, betas=np.asarray(value.betas, dtype=np.float64), pose=pose
    )
    return np.asarray(candidate_vertices), skin, skin_faces, joints, np.asarray(posed_globals)


def _evaluate_level(
    value: Any,
    pose_map: Any,
    *,
    asset: Any,
    model: Mapping[str, np.ndarray],
    poses: Mapping[str, np.ndarray],
    cap_ids: Mapping[str, Mapping[str, np.ndarray]],
    area_reference: np.ndarray,
    timeout_seconds: float,
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    bone_names = [str(name) for name in asset.source_bone_names]
    for pose_name in POSE_NAMES:
        started = time.perf_counter()
        try:
            with _deadline(timeout_seconds):
                pose = np.asarray(poses[pose_name], dtype=np.float32)
                candidate_vertices, skin, skin_faces, smplx_joints, posed_globals = _render_arrays(
                    value, pose_map, asset=asset, model=model, pose=pose
                )
                all_metrics = absolute_poke_metrics(
                    candidate_vertices,
                    asset=asset,
                    skin=skin,
                    skin_faces=skin_faces,
                    area_reference=area_reference,
                )
                groups = _group_metrics(all_metrics)
                seams = _seam_metrics(candidate_vertices, cap_ids=cap_ids)
                station = _joint_station_metrics(
                    posed_globals, smplx_joints, bone_names=bone_names
                )
                cells[pose_name] = {
                    "status": "complete",
                    "available": True,
                    "groups": groups,
                    "seams": seams,
                    "joint_station_distances": station,
                    "elapsed_seconds": float(time.perf_counter() - started),
                }
        except Exception as exc:
            cells[pose_name] = {
                "status": _status_for_exception(exc),
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": float(time.perf_counter() - started),
            }
    return {"poses": cells, "score": _level_score({"poses": cells})}


def _add_seam_deltas(levels: Mapping[str, Any], baseline_key: str = "0mm") -> None:
    baseline = levels.get(baseline_key, {})
    baseline_poses = baseline.get("poses", {})
    for level in levels.values():
        for pose_name, cell in level.get("poses", {}).items():
            base_cell = baseline_poses.get(pose_name, {})
            base_seams = base_cell.get("seams", {})
            for side, pairs in cell.get("seams", {}).items():
                for pair_name, row in pairs.items():
                    base_row = base_seams.get(side, {}).get(pair_name, {})
                    if row.get("available") and base_row.get("available"):
                        row["delta_vs_baseline_m"] = float(row["min_m"] - base_row["min_m"])
                        row["p95_delta_vs_baseline_m"] = float(row["p95_m"] - base_row["p95_m"])
                    else:
                        row["delta_vs_baseline_m"] = None
                        row["p95_delta_vs_baseline_m"] = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--capture-213328", type=Path, default=DEFAULT_CAPTURE_213328)
    parser.add_argument("--capture-213712", type=Path, default=DEFAULT_CAPTURE_213712)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--amass-hf-root", type=Path, default=DEFAULT_AMASS_ROOT)
    parser.add_argument("--subjects", default="213328,213712")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cell-timeout-seconds",
        type=float,
        default=900.0,
        help="per-pose timeout; <=0 disables it (default 900)",
    )
    return parser


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
    capture_poses: Mapping[str, np.ndarray],
    capture_betas: Mapping[str, np.ndarray],
    heldout_sitting: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    subject_root = Path(args.candidate).expanduser().resolve() / f"subject_{subject}"
    original = _load_subject_v11(subject_root)
    if original.source_operator_digest != operator_digest:
        raise ValueError(f"{subject}: source operator digest mismatch")
    if original.calibration_digest != calibration_digest:
        raise ValueError(f"{subject}: calibration digest mismatch")
    if original.smplx_model_sha256 != model_sha:
        raise ValueError(f"{subject}: SMPL-X model digest mismatch")
    if original.capture_sha256 != CAPTURE_SHA256[subject]:
        raise ValueError(f"{subject}: V12e capture digest mismatch")
    if not np.allclose(np.asarray(original.betas), capture_betas[subject], atol=1.0e-7, rtol=0.0):
        raise ValueError(f"{subject}: V12e beta values differ from capture")
    materialized = materialize_subject(
        operator, betas=np.asarray(original.betas), gender="male"
    )
    asset = materialized.rigged_asset
    if not np.array_equal(np.asarray(original.faces), np.asarray(asset.faces)):
        raise ValueError(f"{subject}: V12e faces differ from source asset")
    poses = {
        "tpose": np.zeros((55, 3), dtype=np.float32),
        "pose_213328": np.asarray(capture_poses["213328"], dtype=np.float32),
        "pose_213712": np.asarray(capture_poses["213712"], dtype=np.float32),
        "heldout_sitting": np.asarray(heldout_sitting, dtype=np.float32),
    }
    # The deep-flex solver is shape-specific; use the same SMPL-X subject
    # beta that will be used for containment and retain only both-knee 120°.
    deep = _make_capture_poses(
        {"213328": capture_poses["213328"], "213712": capture_poses["213712"]},
        model=model,
        betas=np.asarray(original.betas, dtype=np.float64),
        target_deg=120.0,
    )
    if "flex_knee_both_120" not in deep:
        raise ValueError("deep-flex builder did not return flex_knee_both_120")
    poses["bilateral_knee_120"] = np.asarray(deep["flex_knee_both_120"], dtype=np.float32)
    cap_ids = _station_caps(original, asset)
    area_reference = np.asarray(asset.vertices_rest, dtype=np.float64)
    level_rows: dict[str, Any] = {}
    original_maps: dict[str, Any] = {}
    for shortening_mm in SHORTENING_MM:
        key = f"{shortening_mm}mm"
        amount = float(shortening_mm) / 1000.0
        level_started = time.perf_counter()
        try:
            # Every level starts from exactly the loaded V12e value; no level
            # inherits a previous station displacement.
            candidate, cage_report = apply_knee_station_cage_v13(
                original, asset, {"L": amount, "R": amount}
            )
            pose_map = build_pose_map_v10(
                candidate,
                asset=asset,
                calibration=calibration,
                oracle_path=oracle,
                source_operator_digest=operator_digest,
            )
            original_maps[key] = (candidate, pose_map)
            evaluated = _evaluate_level(
                candidate,
                pose_map,
                asset=asset,
                model=model,
                poses=poses,
                cap_ids=cap_ids,
                area_reference=area_reference,
                timeout_seconds=float(args.cell_timeout_seconds),
            )
            level_rows[key] = {
                "status": "complete",
                "shortening_m_by_side": {"L": amount, "R": amount},
                "cage_report": cage_report,
                "poses": evaluated["poses"],
                "score": evaluated["score"],
                "elapsed_seconds": float(time.perf_counter() - level_started),
            }
            print(
                f"subject={subject} level={key} status=complete score={evaluated['score']}",
                flush=True,
            )
        except Exception as exc:
            level_rows[key] = {
                "status": _status_for_exception(exc),
                "shortening_m_by_side": {"L": amount, "R": amount},
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": float(time.perf_counter() - level_started),
            }
            print(
                f"subject={subject} level={key} status={level_rows[key]['status']} error={type(exc).__name__}",
                flush=True,
            )
    _add_seam_deltas(level_rows)
    available = {
        key: row for key, row in level_rows.items()
        if row.get("status") == "complete" and np.isfinite(row.get("score", (np.inf, np.inf))[0])
    }
    if not available:
        raise RuntimeError(f"{subject}: no complete station-cage level")
    baseline_key = "0mm"
    if baseline_key not in available:
        raise RuntimeError(f"{subject}: baseline 0mm level is unavailable")
    best_key = min(available, key=lambda key: tuple(available[key]["score"]))
    tissue_codes = _tissue_codes(asset)
    comparison_root = Path(args.output).expanduser().resolve() / "subjects" / f"subject_{subject}" / "comparisons"
    comparison_files: dict[str, dict[str, str]] = {"baseline": {}, "best": {}}
    for pose_name in POSE_NAMES:
        pose = poses[pose_name]
        source_value, source_map = original_maps[baseline_key]
        best_value, best_map = original_maps[best_key]
        source_vertices, skin, skin_faces, joints, _source_globals = _render_arrays(
            source_value, source_map, asset=asset, model=model, pose=pose
        )
        best_vertices, best_skin, best_skin_faces, best_joints, _best_globals = _render_arrays(
            best_value, best_map, asset=asset, model=model, pose=pose
        )
        if not np.array_equal(skin_faces, best_skin_faces):
            raise ValueError(f"{subject}/{pose_name}: skin faces changed between baseline and best")
        if not np.allclose(skin, best_skin, atol=0.0, rtol=0.0):
            raise ValueError(f"{subject}/{pose_name}: skin vertices changed between baseline and best")
        metadata = {
            "schema_version": 13,
            "artifact_kind": "StationCageComparisonV13",
            "subject": subject,
            "pose": pose_name,
            "baseline_level": baseline_key,
            "best_level": best_key,
            "hypothesis": True,
            "publishable": False,
            "soft_material_linkage": "not evaluated; requires material attachment mapping",
        }
        base_path = comparison_root / f"baseline_{pose_name}.npz"
        best_path = comparison_root / f"best_{pose_name}.npz"
        _save_comparison(
            base_path,
            pose_name=pose_name,
            pose=pose,
            faces=asset.faces,
            skin_faces=skin_faces,
            source_vertices=source_vertices,
            candidate_vertices=source_vertices,
            skin_vertices=skin,
            smplx_joints=joints,
            vertex_tissue=tissue_codes,
            metadata={**metadata, "selection": "baseline"},
        )
        _save_comparison(
            best_path,
            pose_name=pose_name,
            pose=pose,
            faces=asset.faces,
            skin_faces=best_skin_faces,
            source_vertices=source_vertices,
            candidate_vertices=best_vertices,
            skin_vertices=best_skin,
            smplx_joints=best_joints,
            vertex_tissue=tissue_codes,
            metadata={**metadata, "selection": "best"},
        )
        comparison_files["baseline"][pose_name] = str(base_path)
        comparison_files["best"][pose_name] = str(best_path)
    # Keep one explicit non-best pair for visual audit: the selected best is
    # often 0 mm, which would make a renderer comparison visually identical.
    # This file always compares the same original V12e source (0 mm) with the
    # independently compiled 30 mm bilateral station cage.
    explicit_level = "30mm"
    explicit_pose_name = "bilateral_knee_120"
    if explicit_level in original_maps:
        explicit_value, explicit_map = original_maps[explicit_level]
        source_value, source_map = original_maps[baseline_key]
        explicit_pose = poses[explicit_pose_name]
        explicit_source, explicit_skin, explicit_skin_faces, explicit_joints, _ = _render_arrays(
            source_value, source_map, asset=asset, model=model, pose=explicit_pose
        )
        explicit_candidate, explicit_candidate_skin, explicit_candidate_faces, explicit_candidate_joints, _ = _render_arrays(
            explicit_value, explicit_map, asset=asset, model=model, pose=explicit_pose
        )
        if not np.array_equal(explicit_skin_faces, explicit_candidate_faces):
            raise ValueError(f"{subject}/{explicit_pose_name}: explicit comparison skin faces changed")
        if not np.allclose(explicit_skin, explicit_candidate_skin, atol=0.0, rtol=0.0):
            raise ValueError(f"{subject}/{explicit_pose_name}: explicit comparison skin changed")
        explicit_path = comparison_root / "30mm_bilateral_knee_120.npz"
        _save_comparison(
            explicit_path,
            pose_name=explicit_pose_name,
            pose=explicit_pose,
            faces=asset.faces,
            skin_faces=explicit_skin_faces,
            source_vertices=explicit_source,
            candidate_vertices=explicit_candidate,
            skin_vertices=explicit_candidate_skin,
            smplx_joints=explicit_candidate_joints,
            vertex_tissue=tissue_codes,
            metadata={
                "schema_version": 13,
                "artifact_kind": "StationCageExplicit30mmComparisonV13",
                "subject": subject,
                "pose": explicit_pose_name,
                "source_level": baseline_key,
                "candidate_level": explicit_level,
                "source_is_original_v12e": True,
                "hypothesis": True,
                "publishable": False,
                "soft_material_linkage": "not evaluated; requires material attachment mapping",
            },
        )
        comparison_files["explicit_30mm_bilateral_knee_120"] = str(explicit_path)
    return {
        "subject": subject,
        "status": "complete",
        "hypothesis": True,
        "publishable": False,
        "soft_material_linkage": {
            "evaluated": False,
            "required_next_step": "material attachment mapping and soft-tissue containment evaluation",
        },
        "shortening_levels_mm": list(SHORTENING_MM),
        "levels": level_rows,
        "baseline_level": baseline_key,
        "best_level": best_key,
        "best_score": list(available[best_key]["score"]),
        "comparisons": comparison_files,
        "pose_provenance": {
            "tpose": "zero axis-angle",
            "capture_213328": "frozen EasyMocap capture",
            "capture_213712": "frozen EasyMocap capture",
            "bilateral_knee_120": "shape-specific deep-flex solver, target anatomical angle 120 deg",
            "heldout_sitting": "fixed AMASS frame, rotations only; shape remains subject-specific",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite station-cage output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 13,
        "artifact_kind": "StationCageFeasibilityV13",
        "hypothesis": True,
        "publishable": False,
        "trusted_latest_updated": False,
        "soft_material_linkage_evaluated": False,
        "soft_material_linkage_required": "material attachment mapping is a separate required stage",
        "containment_metric": "absolute signed point-to-SMPL-X-skin distance; bone groups only",
        "seam_metric": "symmetric nearest cap-vertex distance; delta is relative to same-pose 0mm baseline",
        "shortening_levels_mm": list(SHORTENING_MM),
        "no_clamp_policy": "invalid or rejected levels are recorded; no amount is silently clamped",
        "subjects": {},
        "failures": [],
    }
    try:
        operator = load_source_operator(Path(args.operator).expanduser().resolve(), mmap=True)
        operator_digest = operator.runtime_digest(validate=False)
        if operator_digest != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError("station cage evaluation requires the frozen 142 source operator")
        calibration = load_anatomical_calibration_v1(
            Path(args.calibration).expanduser().resolve(),
            operator=operator,
            required_scope="full_main_chain",
        )
        calibration_digest = _calibration_content_digest(calibration)
        oracle = Path(args.oracle).expanduser().resolve()
        if _sha256(oracle) != EXPECTED_ORACLE_SHA256:
            raise ValueError("station cage evaluation requires the frozen Blender oracle")
        model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
        model = load_smplx_model_v7(model_path)
        capture_paths = {
            "213328": Path(args.capture_213328).expanduser().resolve(),
            "213712": Path(args.capture_213712).expanduser().resolve(),
        }
        capture_betas: dict[str, np.ndarray] = {}
        capture_poses: dict[str, np.ndarray] = {}
        capture_provenance: dict[str, Any] = {}
        for subject, path in capture_paths.items():
            betas, pose, digest = _load_capture(path, model_path=model_path)
            if digest != CAPTURE_SHA256[subject]:
                raise ValueError(f"capture {subject} differs from the frozen digest")
            capture_betas[subject] = np.asarray(betas, dtype=np.float64)
            capture_poses[subject] = np.asarray(pose, dtype=np.float32)
            capture_provenance[subject] = {"path": str(path), "sha256": digest}
        heldout, heldout_provenance = load_held_out_poses_v13(args.amass_hf_root)
        if "heldout_sitting" not in heldout:
            raise ValueError("fixed AMASS held-out pose set lacks heldout_sitting")
        report["provenance"] = {
            "operator": str(Path(args.operator).expanduser().resolve()),
            "operator_runtime_digest": operator_digest,
            "calibration": str(Path(args.calibration).expanduser().resolve()),
            "calibration_digest": calibration_digest,
            "oracle": str(oracle),
            "oracle_sha256": _sha256(oracle),
            "smplx_model": str(model_path),
            "smplx_model_sha256": model_sha,
            "candidate": str(Path(args.candidate).expanduser().resolve()),
            "captures": capture_provenance,
            "heldout": heldout_provenance["heldout_sitting"],
            "runtime_threads": 1,
        }
        subjects = [part.strip() for part in str(args.subjects).split(",") if part.strip()]
        for subject in subjects:
            if subject not in capture_betas:
                raise ValueError(f"unknown subject {subject!r}; expected 213328 or 213712")
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
                    capture_poses=capture_poses,
                    capture_betas=capture_betas,
                    heldout_sitting=heldout["heldout_sitting"],
                )
                report["subjects"][subject] = row
                _write_json(output / "progress.json", report)
            except Exception as exc:
                failure = {
                    "subject": subject,
                    "status": _status_for_exception(exc),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report["subjects"][subject] = failure
                report["failures"].append(failure)
                _write_json(output / "progress.json", report)
                print(f"subject={subject} status={failure['status']} error={type(exc).__name__}", flush=True)
    except Exception as exc:
        report["status"] = _status_for_exception(exc)
        report["failures"].append({"status": report["status"], "error": f"{type(exc).__name__}: {exc}"})
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        _write_json(output / "matrix_manifest.json", report)
        return 1
    report["status"] = "complete"
    report["passed"] = False
    report["elapsed_seconds"] = float(time.perf_counter() - started)
    _write_json(output / "matrix_manifest.json", report)
    _write_json(output / "progress.json", report)
    print(
        f"station_cage subjects={len(report['subjects'])} failures={len(report['failures'])} publishable=false",
        flush=True,
    )
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
