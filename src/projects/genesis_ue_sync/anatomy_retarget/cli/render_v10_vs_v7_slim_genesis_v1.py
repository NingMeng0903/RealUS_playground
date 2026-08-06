"""Slim Genesis compare: V7 right-multiply vs V10 joint-anchored FK."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import (
    pose_whole_chain_vertices_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v10_artifacts import (
    load_chain_retarget_v10_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.v11_artifacts import (
    load_chain_retarget_v11_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    load_whole_chain_rest_fit_v1,
)


KEEP_CAMERAS = (
    "left_knee_ap",
    "left_knee_lateral",
    "right_knee_ap",
    "right_knee_lateral",
    "left_elbow_ap",
    "left_elbow_lateral",
    "left_hand_oblique",
    "whole_ap",
)
KEEP_LAYERS = ("bones_only", "outside_heatmap", "full_anatomy")


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
    parser.add_argument("--v7-shadow", type=Path, required=True)
    parser.add_argument(
        "--v10-shadow",
        type=Path,
        required=True,
        help="V10 or V11 subject matrix root (auto-detects v10/v11 npz)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--keep-set",
        type=Path,
        default=Path(
            "outputs/anatomy_retarget/v10_candidates/review_slim_v10_vs_v7"
        ),
    )
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--delete-full-after-slim", action="store_true")
    return parser


def _render_one(
    *,
    label: str,
    vertices: np.ndarray,
    asset: Any,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    frames: np.ndarray,
    output: Path,
    backend: str,
    method: Any,
) -> dict[str, Any]:
    shadow_out = output / label
    shadow_out.mkdir(parents=True, exist_ok=False)
    cell = _render_modes(
        output=shadow_out,
        vertices=np.asarray(vertices, dtype=np.float32),
        asset=asset,
        skin=skin,
        skin_faces=skin_faces,
        frames=frames,
        backend=backend,
    )
    slim: dict[str, Any] = {"layers": {}}
    slim_root = output / "slim" / label
    slim_root.mkdir(parents=True, exist_ok=True)
    for layer in KEEP_LAYERS:
        layer_dir = shadow_out / layer / "rgb"
        kept = {}
        for cam in KEEP_CAMERAS:
            src = layer_dir / f"{cam}.png"
            if not src.is_file():
                continue
            dst = slim_root / layer / f"{cam}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            kept[cam] = {"path": str(dst), "sha256": _sha256(dst)}
        slim["layers"][layer] = kept
    return {"label": label, "method": method, "render": cell, "slim": slim}


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
    skin, skin_faces = smplx_body_surface_v7(
        model, betas=betas, pose_axis_angle=pose
    )
    oracle = args.oracle.resolve()

    v7_value = load_whole_chain_rest_fit_v1(
        args.v7_shadow.resolve() / "subject_213328",
        operator=operator,
        calibration=calibration,
        smplx_model=model,
        smplx_model_sha256=model_sha,
        recheck=False,
    )
    v7_pm = build_pose_map_v1(
        v7_value,
        asset=asset,
        calibration=calibration,
        oracle_path=oracle,
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    v7_verts, _ = pose_whole_chain_vertices(
        v7_value, v7_pm, source_asset=asset, pose_axis_angle=pose
    )

    candidate_root = args.v10_shadow.resolve() / "subject_213328"
    if (candidate_root / "whole_chain_rest_fit_subject_v11.npz").exists():
        v10_value, v10_meta = load_chain_retarget_v11_subject(candidate_root)
    else:
        v10_value, v10_meta = load_chain_retarget_v10_subject(candidate_root)
    v10_pm = build_pose_map_v1(
        v10_value,
        asset=asset,
        calibration=calibration,
        oracle_path=oracle,
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    v10_verts, _ = pose_whole_chain_vertices_v10(
        v10_value,
        v10_pm,
        source_asset=asset,
        pose_axis_angle=pose,
        segment_scales=v10_meta.get("segment_scales"),
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "V10VsV7SlimGenesisCompare",
        "subject": "213328",
        "pose": "pose_213328",
        "publishable": False,
        "review_layers": {
            "bones_only": "skeleton + translucent SMPL-X overlay",
            "outside_heatmap": "poke-through severity (red)",
            "full_anatomy": "Blender soft tissue / vessels / organs",
        },
        "shadows": {},
    }
    for label, verts, value in (
        ("v7", v7_verts, v7_value),
        ("v10", v10_verts, v10_value),
    ):
        frames, _widths, _details = _measure_frames(
            verts,
            calibration.domains,
            calibration.joint_domain_bases,
            partition="validation",
        )
        report["shadows"][label] = _render_one(
            label=label,
            vertices=verts,
            asset=asset,
            skin=skin,
            skin_faces=skin_faces,
            frames=frames,
            output=output,
            backend=args.backend,
            method=value.build_report.get("method"),
        )
    report["elapsed_seconds"] = float(time.perf_counter() - started)
    report["smplx_model_sha256"] = model_sha
    report["capture_sha256"] = _sha256(capture)
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    keep = args.keep_set.expanduser().resolve()
    if keep.exists():
        shutil.rmtree(keep)
    keep.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output / "slim", keep)
    if args.delete_full_after_slim:
        for label in ("v7", "v10"):
            full = output / label
            if full.exists():
                shutil.rmtree(full)
    print(f"V10VsV7SlimGenesisCompare -> {output}")
    print(f"slim keep-set -> {keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
