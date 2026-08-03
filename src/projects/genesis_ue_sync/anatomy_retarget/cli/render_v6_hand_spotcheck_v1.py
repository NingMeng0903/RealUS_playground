"""Genesis spot-check for V6 hand/wrist/ankle review (bones_only + outside_heatmap)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _measure_frames,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.render_stage1_baseline_compare_v1 import (
    _render_modes,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    load_whole_chain_rest_fit_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import source_bone_posed_global
from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import (
    _weighted_rest_correction,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--v6-shadow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cuda", choices=("cuda", "cpu"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    operator = load_source_operator(args.operator.resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.resolve(),
        operator=operator,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    capture = args.capture_213328.resolve()
    with np.load(capture, allow_pickle=False) as data:
        betas = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
        pose = easymocap_fit_to_smplx55(
            data["Rh"], data["poses"], model_path=model_path
        )
    asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
    value = load_whole_chain_rest_fit_v1(
        args.v6_shadow.resolve() / "subject_213328",
        operator=operator,
        calibration=calibration,
        smplx_model=model,
        smplx_model_sha256=model_sha,
        recheck=False,
    )
    pose_map = build_pose_map_v1(
        value,
        asset=asset,
        calibration=calibration,
        oracle_path=args.oracle.resolve(),
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    poses = {
        "tpose": np.zeros((55, 3), dtype=np.float32),
        "pose_213328": pose,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "V6HandWristAnkleSpotCheck",
        "publishable": False,
        "subjects": {"213328": {"poses": {}}},
    }
    for pose_name, pose_aa in poses.items():
        skin, skin_faces = smplx_body_surface_v7(
            model, betas=betas, pose_axis_angle=pose_aa
        )
        # Pack A baseline (142 materialize)
        src_g = source_bone_posed_global(asset, pose_aa)
        verts_a = _weighted_rest_correction(
            value.vertices_prefit,
            asset.driver_indices,
            asset.driver_weights,
            src_g @ np.linalg.inv(value.B_prefit),
        )
        frames_a, _, _ = _measure_frames(
            verts_a,
            calibration.domains,
            calibration.joint_domain_bases,
            partition="validation",
        )
        out_a = output / "packA_142" / pose_name
        out_a.mkdir(parents=True, exist_ok=False)
        report_a = _render_modes(
            output=out_a,
            vertices=np.asarray(verts_a, dtype=np.float32),
            asset=asset,
            skin=skin,
            skin_faces=skin_faces,
            frames=frames_a,
            backend=args.backend,
        )
        # Pack V6
        verts_v6, _ = pose_whole_chain_vertices(
            value,
            pose_map,
            source_asset=asset,
            pose_axis_angle=pose_aa,
        )
        frames_v6, _, _ = _measure_frames(
            verts_a,  # same camera look-ats as 142 for fair compare
            calibration.domains,
            calibration.joint_domain_bases,
            partition="validation",
        )
        out_v6 = output / "packV6" / pose_name
        out_v6.mkdir(parents=True, exist_ok=False)
        report_v6 = _render_modes(
            output=out_v6,
            vertices=np.asarray(verts_v6, dtype=np.float32),
            asset=asset,
            skin=skin,
            skin_faces=skin_faces,
            frames=frames_v6,
            backend=args.backend,
        )
        manifest["subjects"]["213328"]["poses"][pose_name] = {
            "packA_142": report_a,
            "packV6": report_v6,
        }
    manifest["elapsed_seconds"] = float(time.perf_counter() - started)
    manifest["capture_sha256"] = _sha256(capture)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"V6HandWristAnkleSpotCheck -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
