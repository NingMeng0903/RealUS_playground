#!/usr/bin/env python3
"""Measure the frozen V2 terminal contract without building a new solver result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_containment_v1 import (
    _signed_distance,
    _summary,
    _vertex_areas,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_chain_validation_v1 import (
    _source_pose_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_retarget_v2 import (
    DynamicMainChainSubjectV2,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.terminal_containment_contract_v2 import (
    terminal_containment_contract_v2,
    terminal_containment_regions_v2,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    load_whole_chain_rest_fit_v1,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_saved_v2_candidate(
    path: Path,
    *,
    baseline: Any,
) -> tuple[DynamicMainChainSubjectV2, dict[str, Any]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_kind") != "DynamicMainChainSubjectV2"
        or manifest.get("baseline_commit")
        != "142ece5f0bc646978ae3e8c9add76deea71c26a2"
        or manifest.get("smplx_gender") != "male"
        or manifest.get("publishable") is not False
        or manifest.get("trusted_latest_updated") is not False
    ):
        raise ValueError(f"invalid saved V2 candidate manifest: {path}")
    npz = path / str(manifest["npz"])
    if _sha256(npz) != manifest.get("npz_sha256"):
        raise ValueError(f"saved V2 candidate digest mismatch: {path}")
    with np.load(npz, allow_pickle=False) as data:
        fields = {
            key: np.asarray(data[key]).copy()
            for key in data.files
        }
    build_report = dict(manifest.get("build_report", {}))
    if "changed_parent_local_bind_indices" not in build_report:
        roots = build_report.get("terminal_bind_root_indices")
        if not isinstance(roots, list) or len(roots) != 4:
            raise ValueError("saved V2 candidate has no auditable changed-bind roots")
        build_report["changed_parent_local_bind_indices"] = sorted(
            int(root) for root in roots
        )
        build_report["compatibility_normalization"] = (
            "changed_parent_local_bind_indices_from_terminal_bind_root_indices"
        )
    inherited = dict(baseline.__dict__)
    inherited.update(fields)
    inherited["build_report"] = build_report
    value = DynamicMainChainSubjectV2(**inherited)
    value.validate()
    return value, manifest


def _region_metrics(
    signed: np.ndarray,
    *,
    ids: np.ndarray,
    lookup: np.ndarray,
    area_weights: np.ndarray,
) -> dict[str, Any]:
    rows = lookup[np.asarray(ids, dtype=np.int64)]
    if np.any(rows < 0):
        raise ValueError("feasibility region lies outside the signed-distance query")
    return _summary(signed[rows], area_weights[np.asarray(ids, dtype=np.int64)])


def _mesh_regions(asset: Any, query_ids: np.ndarray) -> dict[str, np.ndarray]:
    query_mask = np.zeros(len(asset.vertices_rest), dtype=bool)
    query_mask[np.asarray(query_ids, dtype=np.int64)] = True
    result: dict[str, np.ndarray] = {}
    for name, tissue, (start, stop) in zip(
        asset.source_mesh_names, asset.source_tissues, asset.source_vertex_ranges
    ):
        start_i, stop_i = int(start), int(stop)
        if str(tissue).strip().lower() != "bone":
            continue
        ids = np.arange(start_i, stop_i, dtype=np.int64)
        selected = ids[query_mask[start_i:stop_i]]
        if len(selected):
            result[str(name)] = selected
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feasibility artifact: {output}")
    operator_path = args.operator.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    oracle_path = args.oracle.expanduser().resolve()
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    capture_paths = {
        "213328": args.capture_213328.expanduser().resolve(),
        "213712": args.capture_213712.expanduser().resolve(),
    }
    operator = load_source_operator(operator_path, mmap=True)
    calibration = load_anatomical_calibration_v1(
        calibration_path, operator=operator, required_scope="full_main_chain"
    )
    model = load_smplx_model_v7(model_path)
    poses = {"tpose": np.zeros((55, 3), dtype=np.float64)}
    betas: dict[str, np.ndarray] = {}
    for label, path in capture_paths.items():
        with np.load(path, allow_pickle=False) as data:
            betas[label] = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
            poses[f"pose_{label}"] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )

    subjects: dict[str, Any] = {}
    candidate_manifests: dict[str, Any] = {}
    for label in ("213328", "213712"):
        baseline = load_whole_chain_rest_fit_v1(
            args.baseline_root.expanduser().resolve() / f"subject_{label}",
            operator=operator,
            calibration=calibration,
            smplx_model=model,
            smplx_model_sha256=model_sha,
            recheck=False,
        )
        candidate, manifest = _load_saved_v2_candidate(
            args.candidate_root.expanduser().resolve() / f"subject_{label}",
            baseline=baseline,
        )
        subjects[label] = (baseline, candidate)
        candidate_manifests[label] = manifest

    matrix: dict[str, Any] = {}
    frozen_contract: dict[str, Any] | None = None
    for subject_label, (baseline, candidate) in subjects.items():
        asset = materialize_subject(
            operator, betas=betas[subject_label], gender="male"
        ).rigged_asset
        regions = terminal_containment_regions_v2(asset)
        contract = terminal_containment_contract_v2(asset)
        if frozen_contract is None:
            frozen_contract = contract
        elif contract["contract_digest"] != frozen_contract["contract_digest"]:
            raise ValueError("terminal region contract changes across male betas")
        side_regions = {
            name: ids
            for name, ids in regions.items()
            if name.startswith(("left_", "right_"))
        }
        query_ids = np.unique(np.concatenate(list(side_regions.values())))
        lookup = np.full(len(candidate.vertices_final), -1, dtype=np.int64)
        lookup[query_ids] = np.arange(len(query_ids), dtype=np.int64)
        foot_meshes = _mesh_regions(
            asset, np.union1d(regions["foot_major"], regions["toe_phalanges"])
        )
        area_weights = _vertex_areas(candidate.vertices_prefit, candidate.faces)
        pose_map = build_pose_map_v1(
            candidate,
            asset=asset,
            calibration=calibration,
            oracle_path=oracle_path,
            source_operator_digest=operator.runtime_digest(validate=False),
        )
        cells: dict[str, Any] = {}
        for pose_label, pose in poses.items():
            baseline_vertices, _ = _source_pose_vertices(candidate, asset, pose)
            candidate_vertices, _ = pose_whole_chain_vertices(
                candidate,
                pose_map,
                source_asset=asset,
                pose_axis_angle=pose,
                include_tube_transport_preview=False,
            )
            skin, faces = smplx_body_surface_v7(
                model, betas=betas[subject_label], pose_axis_angle=pose
            )
            combined = np.concatenate(
                (
                    np.asarray(baseline_vertices, dtype=np.float64)[query_ids],
                    np.asarray(candidate_vertices, dtype=np.float64)[query_ids],
                )
            )
            signed = _signed_distance(combined, skin, faces)
            baseline_signed, candidate_signed = np.split(signed, 2)
            cells[pose_label] = {
                "regions": {
                    name: {
                        "baseline_142": _region_metrics(
                            baseline_signed,
                            ids=ids,
                            lookup=lookup,
                            area_weights=area_weights,
                        ),
                        "saved_v2_3_candidate": _region_metrics(
                            candidate_signed,
                            ids=ids,
                            lookup=lookup,
                            area_weights=area_weights,
                        ),
                    }
                    for name, ids in side_regions.items()
                },
                "foot_meshes": {
                    name: {
                        "baseline_142": _region_metrics(
                            baseline_signed,
                            ids=ids,
                            lookup=lookup,
                            area_weights=area_weights,
                        ),
                        "saved_v2_3_candidate": _region_metrics(
                            candidate_signed,
                            ids=ids,
                            lookup=lookup,
                            area_weights=area_weights,
                        ),
                    }
                    for name, ids in foot_meshes.items()
                },
            }
        matrix[subject_label] = {"cells": cells}

    package = {
        "schema_version": 2,
        "artifact_kind": "TerminalContainmentFeasibilityV2",
        "purpose": "read_only_contract_freeze_before_solver",
        "contract": frozen_contract,
        "matrix": matrix,
        "sources": {
            "baseline_root": str(args.baseline_root.expanduser().resolve()),
            "candidate_root": str(args.candidate_root.expanduser().resolve()),
            "candidate_manifest_sha256": {
                label: _sha256(
                    args.candidate_root.expanduser().resolve()
                    / f"subject_{label}"
                    / "manifest.json"
                )
                for label in candidate_manifests
            },
        },
        "provenance": {
            "smplx_gender": "male",
            "smplx_model_sha256": model_sha,
            "capture_sha256": {
                label: _sha256(path) for label, path in capture_paths.items()
            },
            "operator_manifest_sha256": _sha256(operator_path / "manifest.json"),
            "calibration_manifest_sha256": _sha256(
                calibration_path / "manifest.json"
            ),
            "oracle_sha256": _sha256(oracle_path),
        },
        "solver_was_run": False,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(package, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"TerminalContainmentFeasibilityV2 -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
