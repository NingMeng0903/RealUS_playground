"""Acceptance render pack: 2 betas x 3+ poses for a V1 / V10 / V11 candidate.

The only Genesis entry point that accepted a V10/V11 shadow was
``render_v10_vs_v7_slim_genesis_v1``, hardcoded to one subject and one pose, so
the 2 beta x 3 pose matrix that MD/todo_ana.md section 9.5 requires could not
be produced for the current candidates at all.  This CLI closes that hole.

Cameras come from the SMPL-X skin and the frozen validation frames only, never
from candidate geometry (section 9.2), and the manifest records that fact plus
a SHA-256 over the camera table.
"""

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
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
from projects.genesis_ue_sync.anatomy_retarget.cli.render_stage1_baseline_compare_v1 import (
    _render_modes,
)
from projects.genesis_ue_sync.anatomy_retarget.deep_flex_poses_v12 import (
    build_deep_flex_poses_v12,
    measure_hinges_deg,
    verify_deep_flex_poses_v12,
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


SUBJECTS = ("213328", "213712")

# Kept after the full tree is dropped: enough to judge the four fixed review
# questions without holding ~1.7 GB per candidate on a 96%-full disk.
SLIM_CAMERAS = (
    "whole_ap",
    "whole_lateral",
    "left_knee_ap",
    "left_knee_lateral",
    "right_knee_ap",
    "left_elbow_lateral",
    "left_hand_oblique",
    "left_foot_oblique",
    "feet_top",
    "pelvis_context",
)
SLIM_LAYERS = ("bones_only", "outside_heatmap", "bones_tubes", "full_anatomy")


def _sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


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
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="shadow root holding subject_<label> dirs (V1, V10 or V11)",
    )
    parser.add_argument(
        "--candidate-name",
        default="candidate",
        help="label recorded in the manifest and used in output paths",
    )
    parser.add_argument(
        "--subjects",
        default=",".join(SUBJECTS),
        help="comma separated beta labels; section 9.5 wants both",
    )
    parser.add_argument(
        "--pack-a-baseline",
        action="store_true",
        help="render the raw 142 materialize instead of a shadow candidate",
    )
    parser.add_argument("--deep-flex-deg", type=float, default=120.0)
    parser.add_argument("--no-deep-flex", action="store_true")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--keep-full",
        action="store_true",
        help="keep every camera; default drops the full tree after slimming",
    )
    return parser


def _load_candidate(root: Path, label: str, **v1_kwargs: Any) -> tuple[Any, str]:
    """Load whichever shadow format the candidate directory holds."""

    subject = root / f"subject_{label}"
    if not subject.is_dir():
        raise FileNotFoundError(f"candidate is missing {subject}")
    if (subject / "whole_chain_rest_fit_subject_v11.npz").exists():
        value, _meta = load_chain_retarget_v11_subject(subject)
        return value, "v11"
    if (subject / "whole_chain_rest_fit_subject_v10.npz").exists():
        value, _meta = load_chain_retarget_v10_subject(subject)
        return value, "v10"
    return load_whole_chain_rest_fit_v1(subject, **v1_kwargs), "v1"


def _slim(cell_dir: Path, slim_root: Path) -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for layer in SLIM_LAYERS:
        layer_dir = cell_dir / layer / "rgb"
        if not layer_dir.is_dir():
            continue
        for camera in SLIM_CAMERAS:
            source = layer_dir / f"{camera}.png"
            if not source.is_file():
                continue
            target = slim_root / layer / f"{camera}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            kept.setdefault(layer, {})[camera] = {
                "path": str(target),
                "sha256": _sha256(target),
            }
    return kept


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing pack: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    subjects = [part.strip() for part in str(args.subjects).split(",") if part.strip()]
    operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.expanduser().resolve(),
        operator=operator,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    regressor = np.asarray(model["J_regressor"], dtype=np.float64)
    oracle = args.oracle.expanduser().resolve()
    digest = operator.runtime_digest(validate=False)

    captures = {
        "213328": args.capture_213328.expanduser().resolve(),
        "213712": args.capture_213712.expanduser().resolve(),
    }
    betas_by_subject: dict[str, np.ndarray] = {}
    capture_poses: dict[str, np.ndarray] = {}
    for label, path in captures.items():
        with np.load(path, allow_pickle=False) as data:
            betas_by_subject[label] = np.asarray(
                data["shapes"], dtype=np.float64
            ).reshape(-1)[:10]
            capture_poses[f"pose_{label}"] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )

    manifest: dict[str, Any] = {
        "schema_version": 12,
        "artifact_kind": "AcceptancePackV12",
        "candidate_name": str(args.candidate_name),
        "candidate_root": str(args.candidate.expanduser().resolve()),
        "publishable": False,
        "trusted_latest_updated": False,
        "camera_source": "smplx_skin_and_frozen_142_validation_frames",
        "candidate_camera_read": False,
        "smplx_model_sha256": model_sha,
        "subjects": {},
    }

    for label in subjects:
        if label not in betas_by_subject:
            raise KeyError(f"no capture for subject {label}")
        betas = betas_by_subject[label]
        asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset

        def joints_of(pose: np.ndarray, _betas: np.ndarray = betas) -> np.ndarray:
            skin, _faces = smplx_body_surface_v7(
                model,
                betas=_betas,
                pose_axis_angle=np.asarray(pose, dtype=np.float32).reshape(55, 3),
            )
            return regressor @ np.asarray(skin, dtype=np.float64)

        poses: dict[str, np.ndarray] = {
            "tpose": np.zeros((55, 3), dtype=np.float32),
            **capture_poses,
        }
        deep_flex: dict[str, Any] = {"enabled": not args.no_deep_flex}
        if not args.no_deep_flex:
            synthetic = build_deep_flex_poses_v12(
                captures={
                    "213328": capture_poses["pose_213328"],
                    "213712": capture_poses["pose_213712"],
                },
                joints_of=joints_of,
                target_deg=float(args.deep_flex_deg),
            )
            verification = verify_deep_flex_poses_v12(
                synthetic, joints_of=joints_of, target_deg=float(args.deep_flex_deg)
            )
            if not verification["passed"]:
                raise SystemExit(
                    f"deep-flex poses missed their target: {verification['failures']}"
                )
            deep_flex["verification"] = verification
            poses.update(synthetic)

        value = None
        artifact_format = "pack_a_142"
        if not args.pack_a_baseline:
            value, artifact_format = _load_candidate(
                args.candidate.expanduser().resolve(),
                label,
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
                oracle_path=oracle,
                source_operator_digest=digest,
            )

        subject_report: dict[str, Any] = {
            "artifact_format": artifact_format,
            "hinge_angles_deg": {},
            "deep_flex": deep_flex,
            "poses": {},
        }
        for pose_name, pose in poses.items():
            pose_aa = np.asarray(pose, dtype=np.float32).reshape(55, 3)
            if args.pack_a_baseline:
                vertices = np.asarray(skin_vertices(asset, pose_aa), dtype=np.float32)
            elif artifact_format == "v1":
                vertices, _ = pose_whole_chain_vertices(
                    value, pose_map, source_asset=asset, pose_axis_angle=pose_aa
                )
            else:
                vertices, _ = pose_whole_chain_vertices_v10(
                    value, pose_map, source_asset=asset, pose_axis_angle=pose_aa
                )
            skin, skin_faces = smplx_body_surface_v7(
                model, betas=betas, pose_axis_angle=pose_aa
            )
            frames, _widths, _details = _measure_frames(
                np.asarray(vertices, dtype=np.float32),
                calibration.domains,
                calibration.joint_domain_bases,
                partition="validation",
            )
            cell_dir = output / f"subject_{label}" / pose_name
            cell_dir.mkdir(parents=True, exist_ok=False)
            cell = _render_modes(
                output=cell_dir,
                vertices=np.asarray(vertices, dtype=np.float32),
                asset=asset,
                skin=skin,
                skin_faces=skin_faces,
                frames=frames,
                backend=args.backend,
                include_acceptance_views=True,
            )
            slim_root = output / "slim" / f"subject_{label}" / pose_name
            cell["slim"] = _slim(cell_dir, slim_root)
            cell["camera_manifest_sha256"] = _sha256_bytes(
                json.dumps(cell["camera_manifest"], sort_keys=True).encode("utf-8")
            )
            if not args.keep_full:
                for layer in SLIM_LAYERS:
                    shutil.rmtree(cell_dir / layer, ignore_errors=True)
                shutil.rmtree(cell_dir / "mesh_assets", ignore_errors=True)
                cell["full_tree_deleted"] = True
            subject_report["poses"][pose_name] = cell
            subject_report["hinge_angles_deg"][pose_name] = measure_hinges_deg(
                pose_aa, joints_of=joints_of
            )
            print(
                f"{args.candidate_name:14s} subject_{label} {pose_name:20s} rendered",
                flush=True,
            )
        manifest["subjects"][label] = subject_report

    manifest["elapsed_seconds"] = float(time.perf_counter() - started)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nAcceptancePackV12 -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
