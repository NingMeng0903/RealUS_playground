"""Run the schema-v6 beta/pose release matrix from one neutral source asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from projects.genesis_ue_sync.anatomy_retarget.canonical_export import (
    export_canonical_tpose,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget import (
    _runtime_pose_matrix_report,
    _signed_distance_containment_report,
)
from projects.genesis_ue_sync.anatomy_retarget.provenance import atomic_write_json
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
    load_rigged_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.shape_volume import (
    _load_obj,
    apply_subject_beta_shape,
)
from projects.genesis_ue_sync.anatomy_retarget.tube_graph import (
    build_asset_attachment_graphs,
    tube_graph_metrics,
)
from projects.genesis_ue_sync.anatomy_retarget.validation_matrix import (
    beta_cases,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-asset", type=Path, required=True)
    parser.add_argument("--reference-canonical-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gender", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--principal-dimensions", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named case; unselected existing reports are retained.",
    )
    return parser.parse_args()


def _case_failures(
    *,
    shape_report: dict[str, Any],
    containment: dict[str, Any],
    tube_metrics: dict[str, dict[str, Any]],
    runtime: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if int(shape_report.get("inverted_tetrahedra", -1)) != 0:
        failures.append("inverted tetrahedra")
    if float(shape_report.get("minimum_jacobian_ratio", -1.0)) < 0.05:
        failures.append("minimum Jacobian ratio below 0.05")
    if int(shape_report.get("outside_query_count", -1)) != 0:
        failures.append("beta cage outside query")
    for name, metrics in tube_metrics.items():
        if float(metrics["length_ratio_p99"]) > 1.12:
            failures.append(f"{name}: tube p99")
        if float(metrics["length_ratio_max"]) > 1.30:
            failures.append(f"{name}: tube maximum")
        if float(metrics["length_ratio_min"]) < 1.0 / 1.30:
            failures.append(f"{name}: tube minimum")
    for pose_name, pose in (runtime.get("cases") or {}).items():
        for name, metrics in (pose.get("soft_meshes") or {}).items():
            if float(metrics["ratio_q99"]) > 1.12:
                failures.append(f"{pose_name}/{name}: q99")
            if float(metrics["ratio_max"]) > 1.30:
                failures.append(f"{pose_name}/{name}: maximum")
    over_limit = containment.get("over_limit_count", {})
    if isinstance(over_limit, dict):
        soft_over_limit = sum(
            int(count)
            for tissue, count in over_limit.items()
            if str(tissue) != "bone"
        )
    else:
        soft_over_limit = int(over_limit)
    if soft_over_limit != 0:
        failures.append("subject containment")
    return sorted(set(failures))


def main() -> int:
    args = _parse_args()
    reference_root = args.reference_canonical_dir.expanduser().resolve()
    manifest = json.loads(
        (reference_root / "source_manifest.json").read_text(encoding="utf-8")
    )
    real_betas = np.asarray(manifest.get("betas", []), dtype=np.float32)
    if not len(real_betas):
        raise ValueError("reference canonical manifest has no betas")
    gender = str(args.gender or manifest.get("gender") or "neutral")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_asset = load_rigged_asset(args.source_asset)
    neutral_graphs = build_asset_attachment_graphs(source_asset)
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for case_name, betas in beta_cases(
        real_betas,
        principal_dimensions=args.principal_dimensions,
        sigma=args.sigma,
    ).items():
        case_report_path = output_root / f"{case_name}.json"
        if args.case and case_name not in set(args.case):
            if not case_report_path.is_file():
                raise ValueError(
                    f"selected matrix omits {case_name!r} without an existing report"
                )
            reports[case_name] = json.loads(
                case_report_path.read_text(encoding="utf-8")
            )
            continue
        canonical_dir = output_root / "canonical" / case_name
        try:
            export_canonical_tpose(
                betas=betas,
                output_dir=canonical_dir,
                gender=gender,
                device=args.device,
                source=f"validation_matrix:{case_name}",
            )
            subject_asset, shape_report = apply_subject_beta_shape(
                source_asset,
                canonical_dir=canonical_dir,
                config=config,
            )
            body_vertices, body_faces = _load_obj(
                canonical_dir / "smpl_canonical_tpose.obj"
            )
            containment = _signed_distance_containment_report(
                subject_asset,
                anatomy_vertices=subject_asset.vertices_rest,
                surface_vertices=body_vertices,
                surface_faces=body_faces,
                stage=case_name,
            )
            tube_metrics = {
                name: tube_graph_metrics(graph, subject_asset.vertices_rest)
                for name, graph in neutral_graphs.items()
            }
            runtime = _runtime_pose_matrix_report(
                subject_asset,
                tube_graphs=neutral_graphs,
            )
            failures = _case_failures(
                shape_report=shape_report,
                containment=containment,
                tube_metrics=tube_metrics,
                runtime=runtime,
            )
            reports[case_name] = {
                "betas": np.asarray(betas, dtype=float).tolist(),
                "passed": not failures,
                "failures": failures,
                "shape": shape_report,
                "containment": containment,
                "tube_graphs": tube_metrics,
                "runtime_pose_matrix": runtime,
            }
        except Exception as exc:
            reports[case_name] = {
                "betas": np.asarray(betas, dtype=float).tolist(),
                "passed": False,
                "failures": [f"{type(exc).__name__}: {exc}"],
            }
        atomic_write_json(case_report_path, reports[case_name])
    report = {
        "passed": all(bool(case["passed"]) for case in reports.values()),
        "case_count": len(reports),
        "cases": reports,
    }
    atomic_write_json(output_root / "validation_matrix.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
