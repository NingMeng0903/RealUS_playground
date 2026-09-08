"""Audit left-elbow bone surfaces before/after the bounded joint-lock ablation.

The input files are the already-generated ``joint_lock_best`` comparison NPZs
for subject 213328.  This CLI performs no posing, fitting, or optimization:
it reuses the frozen-domain point-to-triangle audit and reports V12e source
versus the common fixed-pivot lock candidate for four real poses.

The lock files are a bone-only experiment.  Their soft tissue is not moved by
the lock and this report therefore makes no soft-tissue or whole-body claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
)
from projects.genesis_ue_sync.anatomy_retarget.cli.audit_capture_articular_surfaces_v13 import (
    _load_cell,
    _mesh_layout,
    _one_query,
    _query_specs,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
DEFAULT_CALIBRATION = ROOT / (
    "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/"
    "anatomical_calibration_v1"
)
DEFAULT_INPUT = ROOT / "outputs/anatomy_retarget/v13_forearm_joint_lock_20260908_001"
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v13_elbow_joint_lock_audit_20260908_001"
SUBJECT = "213328"
POSES = ("tpose", "pose_213328", "heldout_sitting", "heldout_kicking")
LEFT_ELBOW_QUERY_NAMES = (
    "left_elbow_humerus_to_radius",
    "left_elbow_humerus_to_ulna",
)


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
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _compact(query: Mapping[str, Any]) -> dict[str, Any]:
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V12e → left-elbow joint-lock 骨面审查",
        "",
        "这是已生成的 joint-lock NPZ 的骨面后验诊断。source 是 V12e，candidate 是",
        "common fixed-pivot lock；查询使用 frozen elbow validation 域顶点到完整对侧骨",
        "三角面的距离。该 ablation 只改 Radius_L/Ulna_L 骨实验，软组织没有跟随。",
        "",
        f"- status: `{report.get('status')}`; passed: `{report.get('passed')}`; publishable: `false`",
        "- soft tissue follow-through: `false`",
        "- signed negative threshold: `0.1 mm`",
        "",
        "| pose | direction | V12e closest/p05 | lock closest/p05 | V12e signed min | lock signed min | V12e negative | lock negative |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pose_name, cell in report.get("cells", {}).items():
        for query_name in LEFT_ELBOW_QUERY_NAMES:
            source = cell["queries"]["source"][query_name]
            candidate = cell["queries"]["candidate"][query_name]
            source_u = source["unsigned_distance_m"]
            candidate_u = candidate["unsigned_distance_m"]
            source_s = source["signed_distance_m"]["min_m"] * 1000
            candidate_s = candidate["signed_distance_m"]["min_m"] * 1000
            lines.append(
                f"| `{pose_name}` | `{query_name.removeprefix('left_elbow_')}` | "
                f"{source_u['closest_m'] * 1000:.3f}/{source_u['p05_m'] * 1000:.3f} | "
                f"{candidate_u['closest_m'] * 1000:.3f}/{candidate_u['p05_m'] * 1000:.3f} | "
                f"{source_s:.3f} | {candidate_s:.3f} | "
                f"{source['negative_signed_gt_0.1mm_count']} | "
                f"{candidate['negative_signed_gt_0.1mm_count']} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这些数值只回答局部骨面采样在 lock 后如何变化；它们不能证明整段骨面无相交，",
            "也不能证明软组织仍然随骨合理联动。joint-lock report 另有外皮 containment 指标，",
            "但本表没有把软组织的变化误算为骨面改善。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty lock audit output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 13,
        "artifact_kind": "ElbowJointLockAuditV13",
        "subject": SUBJECT,
        "poses": list(POSES),
        "status": "running",
        "passed": False,
        "publishable": False,
        "bone_only_experiment": True,
        "soft_tissue_follow_through": False,
    }
    try:
        operator_path = args.operator.expanduser().resolve()
        operator = load_source_operator(operator_path, mmap=True)
        operator_digest = operator.runtime_digest(validate=False)
        if operator_digest != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError("joint-lock audit requires frozen rebuild_012 source operator")
        calibration_path = args.calibration.expanduser().resolve()
        calibration = load_anatomical_calibration_v1(
            calibration_path,
            operator=operator,
            required_scope="full_main_chain",
        )
        calibration_digest = _calibration_content_digest(calibration)
        input_root = args.input.expanduser().resolve()
        query_specs = tuple(spec for spec in _query_specs() if spec["name"] in LEFT_ELBOW_QUERY_NAMES)
        if len(query_specs) != len(LEFT_ELBOW_QUERY_NAMES):
            raise ValueError("left elbow query specification is incomplete")
        cells: dict[str, Any] = {}
        layout = None
        for pose_name in POSES:
            path = input_root / f"subject_{SUBJECT}_joint_lock_best_{pose_name}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            cell = _load_cell(path)
            if layout is None:
                layout = _mesh_layout(operator.template_asset, cell["faces"])
            else:
                if not np.array_equal(cell["faces"], np.asarray(operator.template_asset.faces, dtype=np.int32)):
                    raise ValueError(f"{path}: topology differs from source asset")
            if len(cell["source_vertices"]) != len(operator.template_asset.vertices_rest):
                raise ValueError(f"{path}: source vertex count differs from materialized source")
            queries = {
                "source": {
                    str(spec["name"]): _compact(
                        _one_query(
                            spec,
                            vertices=cell["source_vertices"],
                            calibration=calibration,
                            layout=layout,
                        )
                    )
                    for spec in query_specs
                },
                "candidate": {
                    str(spec["name"]): _compact(
                        _one_query(
                            spec,
                            vertices=cell["candidate_vertices"],
                            calibration=calibration,
                            layout=layout,
                        )
                    )
                    for spec in query_specs
                },
            }
            deltas: dict[str, Any] = {}
            for query_name in LEFT_ELBOW_QUERY_NAMES:
                source = queries["source"][query_name]
                candidate = queries["candidate"][query_name]
                deltas[query_name] = {
                    "closest_delta_m_candidate_minus_source": candidate["unsigned_distance_m"]["closest_m"] - source["unsigned_distance_m"]["closest_m"],
                    "p05_delta_m_candidate_minus_source": candidate["unsigned_distance_m"]["p05_m"] - source["unsigned_distance_m"]["p05_m"],
                    "signed_min_delta_m_candidate_minus_source": candidate["signed_distance_m"]["min_m"] - source["signed_distance_m"]["min_m"],
                    "negative_count_delta_candidate_minus_source": candidate["negative_signed_gt_0.1mm_count"] - source["negative_signed_gt_0.1mm_count"],
                }
            cells[pose_name] = {
                "input": {"path": str(path), "sha256": _sha256(path)},
                "metadata": cell.get("metadata"),
                "pose55_sha256": hashlib.sha256(np.ascontiguousarray(cell["pose"]).tobytes()).hexdigest(),
                "queries": queries,
                "deltas": deltas,
            }
            _write_json(
                output / "progress.json",
                {"status": "running", "completed_poses": list(cells), "output": str(output)},
            )
        report.update(
            {
                "status": "complete",
                "passed": True,
                "provenance": {
                    "operator": str(operator_path),
                    "operator_runtime_digest": operator_digest,
                    "calibration": str(calibration_path),
                    "calibration_digest": calibration_digest,
                    "input_root": str(input_root),
                    "query_policy": "frozen elbow validation domains to complete opposing bone triangles",
                },
                "cells": cells,
                "interpretation": (
                    "The source/candidate delta is a local bone-surface change from V12e to the "
                    "common fixed-pivot lock. Soft tissue was not transported by this bone-only "
                    "experiment, so no tissue-linkage or whole-body acceptance claim is made."
                ),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
        _write_json(output / "report.json", report)
        (output / "README.md").write_text(_markdown(report), encoding="utf-8")
        _write_json(
            output / "progress.json",
            {"status": "complete", "completed_poses": list(cells), "output": str(output)},
        )
        print(f"elbow_joint_lock subject={SUBJECT} poses={','.join(POSES)} passed=true publishable=false", flush=True)
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
