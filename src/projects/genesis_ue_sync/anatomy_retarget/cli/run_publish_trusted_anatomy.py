#!/usr/bin/env python3
"""Validate and publish the last trusted reusable Stage-1 anatomy asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget import (
    _file_digest,
    _publish_upsert,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import load_rigged_asset
from projects.genesis_ue_sync.anatomy_retarget.stage1_contract import stage1_runtime_contract


DEFAULT_TRUSTED_ASSET = Path(
    "outputs/anatomy_retarget/stage1_shared_bind_field_v71/runs/6/"
    "853a0950958a7b45c7f3350c9005c5920a85a81beae85c6a0164e51b04750d26/"
    "anatomy_rigged.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-npz", type=Path, default=DEFAULT_TRUSTED_ASSET)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5601")
    parser.add_argument("--model-id", default="patient_anatomy")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_publishable_asset(path: Path) -> tuple[object, dict[str, object]]:
    asset_path = path.expanduser().resolve()
    if not asset_path.is_file():
        raise ValueError(f"anatomy asset does not exist: {asset_path}")
    asset = load_rigged_asset(asset_path, validate=True)
    metadata = dict(asset.metadata or {})
    failures: list[str] = []
    evidence: dict[str, object] = {}
    for filename in ("run_status.json", "quality_report.json"):
        evidence_path = asset_path.parent / filename
        if not evidence_path.is_file():
            continue
        try:
            report = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{filename} is unreadable: {exc}")
            continue
        evidence[filename] = report
        if not isinstance(report, dict) or report.get("passed") is not True:
            failures.append(
                f"{filename} explicitly refuses publication: "
                f"state={report.get('state') if isinstance(report, dict) else None!r}"
            )
    runtime_contract = stage1_runtime_contract(asset)
    if not bool(runtime_contract.get("passed", False)):
        failures.append(
            "Stage-1 runtime contract failed: "
            f"missing={runtime_contract.get('required_runtime_fields_missing', [])}, "
            f"pose_cache_absent={runtime_contract.get('pose_cache_absent')}, "
            f"zero_pose_vertex_error_m={runtime_contract.get('zero_pose_vertex_error_m')}"
        )
    if asset.source_bind_vertices is None:
        failures.append("source_bind_vertices is missing")
    if asset.source_bone_names is None or len(asset.source_bone_names) != 235:
        failures.append("complete 235-bone Blender source rig is missing")
    if asset.driver_indices is None or asset.driver_weights is None:
        failures.append("full sparse Blender source weights are missing")
        driver_width = 0
    else:
        driver_width = int(np.asarray(asset.driver_indices).shape[1])
        if driver_width != 14:
            failures.append(f"expected full 14-slot weights, found {driver_width}")
    if asset.pose_cache_vertices is not None or str(asset.pose_cache_hash):
        failures.append("pose-specific vertex cache is present")
    rejected_keys = sorted(
        key
        for key in (
            "source_rotation_distribution_v1",
            "source_local_action_coupling_v1",
        )
        if key in metadata
    )
    if rejected_keys:
        failures.append(f"rejected experimental coupling is present: {rejected_keys}")

    zero = np.zeros((55, 3), dtype=np.float32)
    posed_zero = skin_vertices(asset, zero)
    roundtrip = np.linalg.norm(
        np.asarray(posed_zero, dtype=np.float64)
        - np.asarray(asset.vertices_rest, dtype=np.float64),
        axis=1,
    )
    roundtrip_max = float(np.max(roundtrip))
    if not np.isfinite(roundtrip_max) or roundtrip_max > 1.0e-5:
        failures.append(f"T-pose round-trip error is {roundtrip_max:.9g} m")
    if failures:
        raise ValueError("asset refused:\n- " + "\n- ".join(failures))
    return asset, {
        "asset": str(asset_path),
        "content_hash": _file_digest(asset_path),
        "vertices": int(len(asset.vertices_rest)),
        "source_bones": int(len(asset.source_bone_names)),
        "driver_width": driver_width,
        "t_pose_roundtrip_max_m": roundtrip_max,
        "runtime_backend": runtime_contract.get("runtime_backend"),
        "runtime_contract_version": runtime_contract.get("contract_version"),
        "requires_blender_at_runtime": runtime_contract.get("requires_blender_at_runtime"),
        "requires_pose_rebake": runtime_contract.get("requires_pose_rebake"),
        "pose_cache_present": False,
        "quality_evidence": sorted(evidence),
    }


def main() -> int:
    args = parse_args()
    try:
        _asset, report = validate_publishable_asset(args.asset_npz)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print("trusted anatomy asset check passed")
    for key, value in report.items():
        print(f"  {key}={value}")
    if args.dry_run:
        print("dry-run: publish skipped")
        return 0
    sent = _publish_upsert(
        bind=str(args.bind),
        model_id=str(args.model_id),
        asset_npz=Path(report["asset"]),
        color_rgba=(0.8, 0.05, 0.05, 0.85),
        duration_s=float(args.duration_s),
        rate_hz=float(args.rate_hz),
    )
    print(
        f"published anatomy control action=upsert model_id={args.model_id} "
        f"sent={sent} bind={args.bind}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
