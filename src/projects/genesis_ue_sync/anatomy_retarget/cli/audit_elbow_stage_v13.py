"""Causal V11-to-V12e left-elbow stage audit for subject 213328.

This bounded diagnostic compares the frozen V11 anchored rest fit with the
V12e independent forearm mesh re-seat on only the 213328 T-pose and its own
capture pose.  Both stages are posed with the same V10 FK helper.  The report
queries the frozen elbow validation domains against the complete opposing
bone meshes, and writes renderer-compatible comparison NPZ files whose
``source_vertices`` are V11 and ``candidate_vertices`` are V12e.

It does not fit, optimize, or alter either retarget method.  The point-to-
triangle measurements are local sampled-domain diagnostics, not a whole-mesh
intersection proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_OPERATOR_RUNTIME_DIGEST,
    EXPECTED_ORACLE_SHA256,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.audit_capture_articular_surfaces_v13 import (
    _mesh_layout,
    _one_query,
    _query_specs,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import (
    build_pose_map_v10,
    pose_whole_chain_vertices_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.v11_artifacts import (
    load_chain_retarget_v11_subject,
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
DEFAULT_V11 = ROOT / "outputs/anatomy_retarget/v11_candidates/chain_retarget_v11_anchored_001"
DEFAULT_V12E = ROOT / "outputs/anatomy_retarget/v12_candidates/chain_retarget_v12e_forearm_mesh_001"
DEFAULT_COMPARISON_ROOT = ROOT / "outputs/anatomy_retarget/v13_material_20260908_001"
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v13_elbow_stage_audit_20260908_001"

SUBJECT = "213328"
POSES = ("tpose", "pose_213328")
LEFT_ELBOW_QUERY_NAMES = (
    "left_elbow_humerus_to_radius",
    "left_elbow_humerus_to_ulna",
)
FOREARM_MESH_TO_CONTROLLER = {
    "Radius_L": "Forearm_Twist_L",
    "Ulna_L": "Forearm_Bone_L",
}
FOREARM_MESHES = tuple(FOREARM_MESH_TO_CONTROLLER)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


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


def _load_comparison(path: Path) -> dict[str, Any]:
    required = {
        "faces",
        "skin_faces",
        "skin_vertices",
        "smplx_joints",
        "pose",
        "vertex_tissue",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing renderer fields {missing}")
        faces = np.asarray(data["faces"], dtype=np.int32)
        skin_faces = np.asarray(data["skin_faces"], dtype=np.int32)
        skin = np.asarray(data["skin_vertices"], dtype=np.float32)
        joints = np.asarray(data["smplx_joints"], dtype=np.float32)
        pose = np.asarray(data["pose"], dtype=np.float32).reshape(55, 3)
        tissue = np.asarray(data["vertex_tissue"], dtype=np.int8)
        metadata = None
        if "metadata_json" in data.files:
            try:
                metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {"parse_error": True}
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path}: faces must be [F,3]")
    if skin_faces.ndim != 2 or skin_faces.shape[1] != 3:
        raise ValueError(f"{path}: skin_faces must be [F,3]")
    if skin.ndim != 2 or skin.shape[1] != 3:
        raise ValueError(f"{path}: skin_vertices must be [V,3]")
    if joints.shape != (55, 3):
        raise ValueError(f"{path}: smplx_joints must be [55,3]")
    if tissue.ndim != 1 or (len(faces) and len(tissue) <= int(np.max(faces))):
        raise ValueError(f"{path}: invalid vertex_tissue")
    if not np.all(np.isfinite(skin)) or not np.all(np.isfinite(joints)):
        raise ValueError(f"{path}: skin/joints contain non-finite values")
    return {
        "path": path,
        "sha256": _sha256(path),
        "faces": faces,
        "skin_faces": skin_faces,
        "skin_vertices": skin,
        "smplx_joints": joints,
        "pose": pose,
        "vertex_tissue": tissue,
        "metadata": metadata,
    }


def _fit_rigid_transform(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit ``after ~= before @ R.T + t`` and return (SE3, max residual)."""

    first = np.asarray(before, dtype=np.float64).reshape(-1, 3)
    second = np.asarray(after, dtype=np.float64).reshape(-1, 3)
    if first.shape != second.shape or len(first) < 3:
        raise ValueError("rigid fit requires matching point sets with at least 3 points")
    center_first = np.mean(first, axis=0)
    center_second = np.mean(second, axis=0)
    centered_first = first - center_first
    centered_second = second - center_second
    covariance = centered_first.T @ centered_second
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    translation = center_second - rotation @ center_first
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    reconstructed = first @ rotation.T + translation
    residual = np.linalg.norm(reconstructed - second, axis=1)
    return transform, float(np.max(residual))


def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(np.asarray(rotation, dtype=np.float64))) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def _domain_ids(calibration: Any, part: str) -> np.ndarray:
    key = f"elbow/left/{part}.validation"
    if key not in calibration.domains:
        raise KeyError(f"missing calibration domain {key}")
    return np.asarray(calibration.domains[key], dtype=np.int64).reshape(-1)


def _domain_displacement(
    before: np.ndarray,
    after: np.ndarray,
    ids: np.ndarray,
) -> dict[str, Any]:
    first = np.asarray(before, dtype=np.float64)[ids]
    second = np.asarray(after, dtype=np.float64)[ids]
    displacement = np.linalg.norm(second - first, axis=1)
    return {
        "vertex_count": int(len(displacement)),
        "mean_m": float(np.mean(displacement)),
        "p05_m": float(np.quantile(displacement, 0.05)),
        "p95_m": float(np.quantile(displacement, 0.95)),
        "max_m": float(np.max(displacement)),
        "center_shift_m": float(np.linalg.norm(np.mean(second, axis=0) - np.mean(first, axis=0))),
    }


def _forearm_reseat_diagnostics(
    v11: Any,
    v12e: Any,
    *,
    asset: Any,
    calibration: Any,
) -> dict[str, Any]:
    names = [str(name) for name in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    pivot_ids = np.unique(
        np.concatenate([_domain_ids(calibration, part) for part in ("humerus", "radius", "ulna")])
    )
    pivot = np.mean(np.asarray(v11.vertices_final, dtype=np.float64)[pivot_ids], axis=0)
    report_clusters = v12e.build_report.get("terminal_reseat_v12", {}).get("clusters", {})
    diagnostics: dict[str, Any] = {
        "pivot_definition": "mean of elbow/left/{humerus,radius,ulna}.validation vertices in V11 rest",
        "v11_material_elbow_pivot_m": pivot.tolist(),
        "per_mesh": {},
        "per_pose_domain_displacement": {},
    }
    for mesh_name, controller in FOREARM_MESH_TO_CONTROLLER.items():
        mesh_index = names.index(mesh_name)
        start, stop = map(int, ranges[mesh_index])
        ids = np.arange(start, stop, dtype=np.int64)
        before = np.asarray(v11.vertices_final, dtype=np.float64)[ids]
        after = np.asarray(v12e.vertices_final, dtype=np.float64)[ids]
        transform, residual = _fit_rigid_transform(before, after)
        point_shift = np.linalg.norm(after - before, axis=1)
        pivot_shift = float(np.linalg.norm(_apply(transform, pivot[None, :])[0] - pivot))
        cluster_report = report_clusters.get(controller, {})
        diagnostics["per_mesh"][mesh_name] = {
            "controller": controller,
            "vertex_range": [start, stop],
            "vertex_count": int(len(ids)),
            "fit_transform_v11_to_v12e": transform.tolist(),
            "fit_reconstruction_max_m": residual,
            "fit_translation_norm_m": float(np.linalg.norm(transform[:3, 3])),
            "fit_rotation_deg": _rotation_angle_deg(transform[:3, :3]),
            "material_elbow_pivot_shift_m": pivot_shift,
            "material_point_shift_mean_m": float(np.mean(point_shift)),
            "material_point_shift_p95_m": float(np.quantile(point_shift, 0.95)),
            "material_point_shift_max_m": float(np.max(point_shift)),
            "terminal_reseat_report": {
                key: cluster_report[key]
                for key in (
                    "mesh_only",
                    "translation_m",
                    "rotation_deg",
                    "max_vertex_shift_m",
                    "moved_vertex_count",
                    "root_origin_shift_m",
                )
                if key in cluster_report
            },
        }
    for pose_name in POSES:
        pose_data = _POSE_CACHE[pose_name]
        diagnostics["per_pose_domain_displacement"][pose_name] = {
            mesh_name: _domain_displacement(
                pose_data["v11_vertices"],
                pose_data["v12e_vertices"],
                _domain_ids(calibration, "radius" if mesh_name == "Radius_L" else "ulna"),
            )
            for mesh_name in FOREARM_MESHES
        }
    return diagnostics


def _compact_query(query: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": query["name"],
        "query_domain": query["query_domain"],
        "target_mesh": query["target_mesh"],
        "query_vertex_count": query["query_vertex_count"],
        "unsigned_distance_m": query["unsigned_distance_m"],
        "signed_distance_m": query["signed_distance_m"],
        "negative_signed_gt_0.1mm_count": query["negative_signed_gt_0.1mm_count"],
        "negative_signed_gt_0.1mm_fraction": query["negative_signed_gt_0.1mm_fraction"],
        "target_mesh_quality": query["target_mesh_quality"],
    }


def _save_comparison(
    path: Path,
    *,
    pose_name: str,
    cell: Mapping[str, Any],
    source_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        faces=np.asarray(cell["faces"], dtype=np.int32),
        skin_faces=np.asarray(cell["skin_faces"], dtype=np.int32),
        source_vertices=np.asarray(source_vertices, dtype=np.float32),
        candidate_vertices=np.asarray(candidate_vertices, dtype=np.float32),
        skin_vertices=np.asarray(cell["skin_vertices"], dtype=np.float32),
        smplx_joints=np.asarray(cell["smplx_joints"], dtype=np.float32),
        pose=np.asarray(cell["pose"], dtype=np.float32),
        vertex_tissue=np.asarray(cell["vertex_tissue"], dtype=np.int8),
        metadata_label=np.asarray("V11_to_V12e_left_elbow_stage"),
        metadata_json=np.asarray(json.dumps(_json_ready(metadata), sort_keys=True, allow_nan=False)),
        subject=np.asarray(SUBJECT),
        pose_name=np.asarray(pose_name),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--v11", type=Path, default=DEFAULT_V11)
    parser.add_argument("--v12e", type=Path, default=DEFAULT_V12E)
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V11 → V12e 左肘阶段因果审查",
        "",
        "213328 的 T-pose 和自身采集 pose。`source_vertices` 是 frozen V11 anchored，",
        "`candidate_vertices` 是 V12e forearm mesh-only reseat；两者使用同一 V10 FK、",
        "同一 skin/joints/pose。骨面数值是冻结 validation 域顶点到完整对侧骨三角面的",
        "局部诊断，不是全网格相交证明。",
        "",
        f"- status: `{report.get('status')}`; passed: `{report.get('passed')}`; publishable: `false`",
        f"- V11 rest method: `{report.get('stage_labels', {}).get('source')}`",
        f"- V12e rest method: `{report.get('stage_labels', {}).get('candidate')}`",
        "",
        "## 左肘骨面指标（mm）",
        "",
        "`closest/p05` 为无符号点到三角面距离；负 signed count 的阈值为 0.1 mm。",
        "",
        "| pose | direction | V11 closest/p05 | V12e closest/p05 | V11 negative | V12e negative |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pose_name, cell in report.get("cells", {}).items():
        for query_name in LEFT_ELBOW_QUERY_NAMES:
            source = cell["queries"]["source"][query_name]
            candidate = cell["queries"]["candidate"][query_name]
            def mm(q: Mapping[str, Any]) -> str:
                d = q["unsigned_distance_m"]
                return f"{d['closest_m'] * 1000:.3f}/{d['p05_m'] * 1000:.3f}"
            lines.append(
                f"| `{pose_name}` | `{query_name.removeprefix('left_elbow_')}` | "
                f"{mm(source)} | {mm(candidate)} | "
                f"{source['negative_signed_gt_0.1mm_count']} | "
                f"{candidate['negative_signed_gt_0.1mm_count']} |"
            )
    lines.extend(["", "## 左前臂 mesh-only 重座位移（mm）", ""])
    lines.append("`material_elbow_pivot_shift` 是把 V11 肘材料支点代入每个 V11→V12e 刚体变换后的位移；同时报告实际骨面点位移。")
    lines.extend(["", "| mesh | controller | rotation | affine translation | elbow pivot shift | point p95/max |", "|---|---|---:|---:|---:|---:|"])
    for mesh_name, row in report.get("forearm_reseat", {}).get("per_mesh", {}).items():
        lines.append(
            f"| `{mesh_name}` | `{row['controller']}` | {row['fit_rotation_deg']:.3f}° | "
            f"{row['fit_translation_norm_m'] * 1000:.3f} | "
            f"{row['material_elbow_pivot_shift_m'] * 1000:.3f} | "
            f"{row['material_point_shift_p95_m'] * 1000:.3f}/{row['material_point_shift_max_m'] * 1000:.3f} |"
        )
    lines.extend(["", "## 因果解释", ""])
    lines.append(str(report.get("causal_interpretation", "")))
    lines.extend([
        "",
        "V11/V12e 关节面查询使用相同冻结 calibration 域和同一 material topology；V12e 的",
        "`Elbow_Rot_L`、`Forearm_Bone_L`、`Forearm_Twist_L` bind 变化与 mesh-only 诊断需结合",
        "报告中的逐字段 digest/位移记录阅读。这个 cell 只能定位阶段增量，不能证明生理接触。",
        "",
    ])
    return "\n".join(lines)


# Filled while auditing cells; kept module-local so the rest diagnostic can
# report pose-specific material support displacement without reloading arrays.
_POSE_CACHE: dict[str, dict[str, np.ndarray]] = {}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty elbow-stage output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 13,
        "artifact_kind": "ElbowStageAuditV13",
        "subject": SUBJECT,
        "poses": list(POSES),
        "status": "running",
        "passed": False,
        "publishable": False,
        "production_solver_modified": False,
        "stage_labels": {
            "source": "frozen V11 anchored rest fit",
            "candidate": "V12e bounded rigid forearm mesh-only reseat",
        },
    }
    try:
        operator_path = args.operator.expanduser().resolve()
        operator = load_source_operator(operator_path, mmap=True)
        operator_digest = operator.runtime_digest(validate=False)
        if operator_digest != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError("elbow stage audit requires frozen rebuild_012 source operator")
        oracle = args.oracle.expanduser().resolve()
        oracle_sha = _sha256(oracle)
        if oracle_sha != EXPECTED_ORACLE_SHA256:
            raise ValueError("elbow stage audit requires frozen Blender oracle")
        calibration = load_anatomical_calibration_v1(
            args.calibration.expanduser().resolve(),
            operator=operator,
            required_scope="full_main_chain",
        )
        calibration_digest = _calibration_content_digest(calibration)
        v11, v11_meta = load_chain_retarget_v11_subject(
            args.v11.expanduser().resolve() / f"subject_{SUBJECT}"
        )
        v12e, v12e_meta = load_chain_retarget_v11_subject(
            args.v12e.expanduser().resolve() / f"subject_{SUBJECT}"
        )
        for label, value in (("V11", v11), ("V12e", v12e)):
            if value.source_operator_digest != operator_digest:
                raise ValueError(f"{label}: source operator digest mismatch")
            if value.calibration_digest != calibration_digest:
                raise ValueError(f"{label}: calibration digest mismatch")
            if np.asarray(value.betas).shape != np.asarray(v11.betas).shape or not np.allclose(
                value.betas, v11.betas, atol=1.0e-8, rtol=0.0
            ):
                raise ValueError(f"{label}: subject beta mismatch")
        if not np.array_equal(v11.faces, v12e.faces):
            raise ValueError("V11/V12e topology differs")
        materialized = materialize_subject(operator, betas=np.asarray(v11.betas), gender="male")
        asset = materialized.rigged_asset
        if not np.array_equal(asset.faces, v11.faces):
            raise ValueError("V11 topology differs from materialized source asset")
        layout = _mesh_layout(asset, np.asarray(asset.faces, dtype=np.int32))
        query_specs = tuple(spec for spec in _query_specs() if spec["name"] in LEFT_ELBOW_QUERY_NAMES)
        if len(query_specs) != len(LEFT_ELBOW_QUERY_NAMES):
            raise ValueError("left elbow query specification is incomplete")
        pose_maps = {
            "source": build_pose_map_v10(
                v11,
                asset=asset,
                calibration=calibration,
                oracle_path=oracle,
                source_operator_digest=operator_digest,
            ),
            "candidate": build_pose_map_v10(
                v12e,
                asset=asset,
                calibration=calibration,
                oracle_path=oracle,
                source_operator_digest=operator_digest,
            ),
        }
        comparison_root = args.comparison_root.expanduser().resolve()
        report["provenance"] = {
            "operator": str(operator_path),
            "operator_runtime_digest": operator_digest,
            "calibration": str(args.calibration.expanduser().resolve()),
            "calibration_digest": calibration_digest,
            "oracle": str(oracle),
            "oracle_sha256": oracle_sha,
            "v11_subject": str(args.v11.expanduser().resolve() / f"subject_{SUBJECT}"),
            "v12e_subject": str(args.v12e.expanduser().resolve() / f"subject_{SUBJECT}"),
            "comparison_root": str(comparison_root),
            "v11_manifest_sha256": _sha256(args.v11.expanduser().resolve() / f"subject_{SUBJECT}" / "manifest.json"),
            "v12e_manifest_sha256": _sha256(args.v12e.expanduser().resolve() / f"subject_{SUBJECT}" / "manifest.json"),
        }
        report["bind_field_equality"] = {
            field: bool(np.array_equal(getattr(v11, field), getattr(v12e, field)))
            for field in ("B_prefit", "B_final", "C_bone", "target_local_bind", "inverse_bind")
        }
        bone_names = [str(name) for name in asset.source_bone_names]
        report["left_elbow_bind_field_equality"] = {
            bone_name: {
                field: bool(
                    np.array_equal(
                        getattr(v11, field)[bone_names.index(bone_name)],
                        getattr(v12e, field)[bone_names.index(bone_name)],
                    )
                )
                for field in ("B_prefit", "B_final", "C_bone", "target_local_bind", "inverse_bind")
            }
            for bone_name in ("Elbow_Rot_L", "Forearm_Bone_L", "Forearm_Twist_L")
        }
        cells: dict[str, Any] = {}
        for pose_name in POSES:
            comparison_path = comparison_root / "subjects" / f"subject_{SUBJECT}" / "comparisons" / f"{pose_name}.npz"
            cell = _load_comparison(comparison_path)
            if not np.array_equal(cell["faces"], asset.faces):
                raise ValueError(f"{comparison_path}: topology differs from source asset")
            if len(cell["vertex_tissue"]) != len(asset.vertices_rest):
                raise ValueError(
                    f"{comparison_path}: vertex_tissue length {len(cell['vertex_tissue'])} "
                    f"does not match materialized vertex count {len(asset.vertices_rest)}"
                )
            source_vertices, _ = pose_whole_chain_vertices_v10(
                v11,
                pose_maps["source"],
                source_asset=asset,
                pose_axis_angle=cell["pose"],
            )
            candidate_vertices, _ = pose_whole_chain_vertices_v10(
                v12e,
                pose_maps["candidate"],
                source_asset=asset,
                pose_axis_angle=cell["pose"],
            )
            _POSE_CACHE[pose_name] = {
                "v11_vertices": np.asarray(source_vertices, dtype=np.float64),
                "v12e_vertices": np.asarray(candidate_vertices, dtype=np.float64),
            }
            # The old V13 comparison source is V12e; this cross-check makes
            # the stage output reproducibly aligned with the Genesis renderer.
            with np.load(comparison_path, allow_pickle=False) as old:
                old_v12e = np.asarray(old["source_vertices"], dtype=np.float64)
            v12e_alignment = np.linalg.norm(np.asarray(candidate_vertices, dtype=np.float64) - old_v12e, axis=1)
            queries = {
                "source": {
                    str(spec["name"]): _compact_query(
                        _one_query(
                            spec,
                            vertices=source_vertices,
                            calibration=calibration,
                            layout=layout,
                        )
                    )
                    for spec in query_specs
                },
                "candidate": {
                    str(spec["name"]): _compact_query(
                        _one_query(
                            spec,
                            vertices=candidate_vertices,
                            calibration=calibration,
                            layout=layout,
                        )
                    )
                    for spec in query_specs
                },
            }
            comparison_out = output / "comparisons" / f"{pose_name}.npz"
            metadata = {
                "schema_version": 13,
                "artifact_kind": "ElbowStageComparisonV13",
                "subject": SUBJECT,
                "pose": pose_name,
                "source_label": "V11_frozen_anchored_rest_fit",
                "candidate_label": "V12e_forearm_mesh_only_reseat",
                "pose_runtime": "pose_whole_chain_vertices_v10",
                "skin_joints_pose_shared": True,
                "publishable": False,
                "renderer_source_geometry": "V11",
                "renderer_candidate_geometry": "V12e",
            }
            _save_comparison(
                comparison_out,
                pose_name=pose_name,
                cell=cell,
                source_vertices=source_vertices,
                candidate_vertices=candidate_vertices,
                metadata=metadata,
            )
            cells[pose_name] = {
                "comparison_input": {"path": str(comparison_path), "sha256": cell["sha256"]},
                "comparison_output": str(comparison_out),
                "pose55_sha256": _array_sha256(cell["pose"]),
                "renderer_shared_skin_joints_pose": True,
                "v12e_alignment_to_existing_source": {
                    "max_m": float(np.max(v12e_alignment)),
                    "rms_m": float(np.sqrt(np.mean(v12e_alignment**2))),
                    "p99_m": float(np.quantile(v12e_alignment, 0.99)),
                },
                "queries": queries,
            }
            _write_json(
                output / "progress.json",
                {
                    "status": "running",
                    "completed_poses": list(cells),
                    "subject": SUBJECT,
                    "output": str(output),
                },
            )
        report["cells"] = cells
        report["forearm_reseat"] = _forearm_reseat_diagnostics(
            v11,
            v12e,
            asset=asset,
            calibration=calibration,
        )
        # Stage attribution is deliberately stated in terms of the observed
        # local sampled-domain increment, without calling it global contact.
        deltas: dict[str, Any] = {}
        for pose_name, cell in cells.items():
            deltas[pose_name] = {}
            for query_name in LEFT_ELBOW_QUERY_NAMES:
                source = cell["queries"]["source"][query_name]
                candidate = cell["queries"]["candidate"][query_name]
                deltas[pose_name][query_name] = {
                    "closest_delta_m_candidate_minus_source": (
                        candidate["unsigned_distance_m"]["closest_m"]
                        - source["unsigned_distance_m"]["closest_m"]
                    ),
                    "p05_delta_m_candidate_minus_source": (
                        candidate["unsigned_distance_m"]["p05_m"]
                        - source["unsigned_distance_m"]["p05_m"]
                    ),
                    "negative_signed_count_delta_candidate_minus_source": (
                        candidate["negative_signed_gt_0.1mm_count"]
                        - source["negative_signed_gt_0.1mm_count"]
                    ),
                }
        report["stage_deltas"] = deltas
        report["causal_interpretation"] = (
            "V11 is the rest-fit baseline. If its sampled elbow distances are already large, "
            "the anchored rest fit contributes a pre-existing separation; the candidate-minus-source "
            "delta isolates the additional V12e forearm mesh-only reseat increment. The saved metrics "
            "show both terms explicitly and do not treat max unsigned distance as a joint gap."
        )
        report["status"] = "complete"
        report["passed"] = True
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        _write_json(output / "report.json", report)
        (output / "README.md").write_text(_markdown(report), encoding="utf-8")
        _write_json(
            output / "progress.json",
            {"status": "complete", "completed_poses": list(cells), "subject": SUBJECT, "output": str(output)},
        )
        print(f"elbow_stage subject={SUBJECT} poses={','.join(POSES)} passed=true publishable=false", flush=True)
        return 0
    except Exception as exc:
        report.update(
            {
                "status": "evaluation_error",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
        _write_json(output / "report.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
