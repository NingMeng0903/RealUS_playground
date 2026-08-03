"""Genesis spot-check for V6: elbow/knee/hand/wrist/ankle vs Pack B/C references."""

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


SPOT_CAMERAS = (
    "left_elbow_ap",
    "left_elbow_lateral",
    "left_knee_ap",
    "left_knee_lateral",
    "left_wrist_ap",
    "left_hand_oblique",
    "left_ankle_ap",
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
    parser.add_argument("--backend", default="cuda")
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
    value = load_whole_chain_rest_fit_v1(
        args.v6_shadow.resolve() / "subject_213328",
        operator=operator,
        calibration=calibration,
        smplx_model=model,
        smplx_model_sha256=model_sha,
        recheck=False,
    )
    asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
    pose_map = build_pose_map_v1(
        value,
        asset=asset,
        calibration=calibration,
        oracle_path=args.oracle.resolve(),
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "V6GenesisSpotCheck",
        "subject": "213328",
        "poses": {},
        "publishable": False,
    }
    for pose_name, pose_aa in (
        ("tpose", np.zeros((55, 3), dtype=np.float32)),
        ("pose_213328", pose),
    ):
        vertices, _ = pose_whole_chain_vertices(
            value,
            pose_map,
            source_asset=asset,
            pose_axis_angle=pose_aa,
        )
        skin, skin_faces = smplx_body_surface_v7(
            model, betas=betas, pose_axis_angle=pose_aa
        )
        frames, _widths, _details = _measure_frames(
            np.asarray(value.vertices_prefit, dtype=np.float64)
            if pose_name == "tpose"
            else vertices,
            calibration.domains,
            calibration.joint_domain_bases,
            partition="validation",
        )
        # Prefer anatomical frames from posed candidate bones for camera look-ats.
        frames, _widths, _details = _measure_frames(
            vertices,
            calibration.domains,
            calibration.joint_domain_bases,
            partition="validation",
        )
        pose_out = output / pose_name
        pose_out.mkdir(parents=True, exist_ok=False)
        cell = _render_modes(
            output=pose_out,
            vertices=np.asarray(vertices, dtype=np.float32),
            asset=asset,
            skin=skin,
            skin_faces=skin_faces,
            frames=frames,
            backend=args.backend,
        )
        # Keep a focused index of the cameras asked for in review.
        focused = {}
        for layer in ("bones_only", "outside_heatmap"):
            rgb_dir = pose_out / layer / "rgb"
            for cam in SPOT_CAMERAS:
                rgb = rgb_dir / f"{cam}.png"
                if rgb.is_file():
                    focused[f"{layer}/{cam}"] = {
                        "path": str(rgb),
                        "sha256": _sha256(rgb),
                    }
        cell["focused_spot_cameras"] = focused
        report["poses"][pose_name] = {
            "contact_sheets": {
                layer: cell["layers"][layer].get("contact_sheet")
                for layer in cell.get("layers", {})
            },
            "focused_spot_cameras": focused,
            "three_layer_contact_sheet": cell.get("three_layer_contact_sheet"),
        }
    report["elapsed_seconds"] = float(time.perf_counter() - started)
    report["smplx_model_sha256"] = model_sha
    report["capture_sha256"] = _sha256(capture)
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"V6GenesisSpotCheck -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
