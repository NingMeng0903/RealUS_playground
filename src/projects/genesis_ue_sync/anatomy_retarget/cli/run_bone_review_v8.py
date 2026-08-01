#!/usr/bin/env python3
"""Build and Genesis-render the untrusted V8.14 bone review candidate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.bone_review_candidate_v8 import (
    build_bone_review_operator_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.bone_review_pack_v8 import (
    ReviewPoseV8,
    write_bone_review_cell_v8,
    write_bone_review_pack_manifest_v8,
    write_bone_review_sweep_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_drive_translation,
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    load_subject_runtime,
    materialize_subject,
    save_source_operator,
    save_subject_runtime,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_BASELINE = REPO_ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_014_bone_review"
DEFAULT_SMPLX_MODEL = (
    REPO_ROOT
    / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_NEUTRAL.pkl"
)
CAPTURES = {
    "213328": REPO_ROOT / "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz",
    "213712": REPO_ROOT / "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz",
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _capture_betas(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        beta = np.asarray(data["shapes"], dtype=np.float32).reshape(-1)[:10]
    if beta.shape != (10,) or not np.all(np.isfinite(beta)):
        raise ValueError(f"capture {path} has invalid SMPL-X betas")
    return beta


def _capture_pose(
    path: Path,
    *,
    model_path: Path,
    pelvis: np.ndarray,
) -> ReviewPoseV8:
    with np.load(path, allow_pickle=False) as data:
        pose = easymocap_fit_to_smplx55(
            data["Rh"],
            data["poses"],
            gender="neutral",
            model_path=model_path,
        )
        translation = easymocap_drive_translation(
            data["Rh"], data["Th"], pelvis
        )
    return ReviewPoseV8(
        label=path.parents[1].name.removeprefix("20260713_"),
        pose_axis_angle=np.asarray(pose, dtype=np.float32).reshape(55, 3),
        transl=np.asarray(translation, dtype=np.float32).reshape(3),
        source=str(path),
    )


def _run_build(args: argparse.Namespace) -> int:
    baseline_path = args.baseline.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"candidate output already exists: {output}")
    started = time.perf_counter()
    baseline = load_source_operator(baseline_path)
    candidate = build_bone_review_operator_v8(
        baseline,
        baseline_path=baseline_path,
    )
    operator_path = output / "source_operator_v8"
    save_source_operator(operator_path, candidate)
    l0_seconds = time.perf_counter() - started
    subjects: dict[str, Any] = {}
    for label, capture in CAPTURES.items():
        subject_started = time.perf_counter()
        subject = materialize_subject(
            candidate,
            betas=_capture_betas(capture),
            gender="male",
        )
        subject_path = output / f"subject_{label}"
        save_subject_runtime(subject_path, subject)
        auth = (
            subject.audit_report.get("tube_coupling", {})
            .get("final_rest_authentication", {})
        )
        frozen = dict(auth.get("frozen_digest_match", {}))
        if not frozen or not all(bool(value) for value in frozen.values()):
            raise ValueError(f"subject {label} failed frozen tube authentication")
        subjects[label] = {
            "path": str(subject_path),
            "runtime_digest": subject.runtime_digest(validate=False),
            "materialize_seconds": time.perf_counter() - subject_started,
            "tube_frozen_authentication": frozen,
            "pelvis_harmonic_cage_v8": subject.audit_report.get(
                "pelvis_harmonic_cage_v8", {}
            ),
            "functional_joint_frames_v8": subject.audit_report.get(
                "functional_joint_frames_v8", {}
            ),
            "leg_centerline_v810": subject.audit_report.get(
                "leg_centerline_v810", {}
            ),
        }
    manifest = {
        "schema_version": 8,
        "artifact_kind": "BoneReviewCandidateV8",
        "baseline": str(baseline_path),
        "baseline_runtime_digest": baseline.runtime_digest(validate=False),
        "operator": str(operator_path),
        "operator_runtime_digest": candidate.runtime_digest(validate=False),
        "l0_seconds": l0_seconds,
        "subjects": subjects,
        "source_commit_baseline": "142ece5f0bc646978ae3e8c9add76deea71c26a2",
        "old_rebuild_013_data_reused": False,
        "old_29e_data_reused": False,
        "vessel_geometry_repair": "not_run",
        "publishable": False,
        "human_signature": "pending",
        "latest_asset_updated": False,
    }
    _write_json(output / "candidate_manifest.json", manifest)
    print(
        f"BoneReviewCandidateV8 operator={manifest['operator_runtime_digest']} "
        f"subjects={len(subjects)} publishable=false -> {output}"
    )
    return 0


def _run_render(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    operator_path = output / "source_operator_v8"
    operator = load_source_operator(operator_path)
    candidate_manifest_path = output / "candidate_manifest.json"
    if not candidate_manifest_path.is_file():
        raise ValueError("candidate_manifest.json is required for 142 comparison")
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    baseline_path = Path(str(candidate_manifest["baseline"])).expanduser().resolve()
    baseline_operator = load_source_operator(baseline_path)
    model_path = args.smplx_model.expanduser().resolve()
    body_model = load_smplx_model_v7(model_path)
    selected = None
    if args.only_cell:
        if "/" not in args.only_cell:
            raise ValueError("--only-cell must use SUBJECT/POSE")
        selected = tuple(args.only_cell.split("/", 1))
    cells: list[dict[str, Any]] = []
    sweeps: list[dict[str, Any]] = []
    for subject_label in CAPTURES:
        subject = load_subject_runtime(output / f"subject_{subject_label}")
        baseline_subject = materialize_subject(
            baseline_operator,
            betas=subject.betas,
            gender=subject.gender,
        )
        poses = [
            ReviewPoseV8(
                label="tpose",
                pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
                transl=np.zeros(3, dtype=np.float32),
                source="zero",
            )
        ]
        poses.extend(
            _capture_pose(
                capture,
                model_path=model_path,
                pelvis=np.asarray(subject.rigged_asset.rest_joints[0]),
            )
            for capture in CAPTURES.values()
        )
        for pose in poses:
            if selected is not None and selected != (subject_label, pose.label):
                continue
            cell_dir = output / "bone_review_pack_v8" / subject_label / pose.label
            print(f"Genesis BoneReviewCellV8 {subject_label}/{pose.label}")
            cells.append(
                write_bone_review_cell_v8(
                    subject_label=subject_label,
                    subject=subject,
                    baseline_subject=baseline_subject,
                    pose=pose,
                    body_model=body_model,
                    output_dir=cell_dir,
                    backend=args.backend,
                    resolution=(args.resolution, args.resolution),
                    include_bed_robot_scene=(
                        subject_label == "213712" and pose.label == "213712"
                    ),
                )
            )
        if selected is None:
            print(f"BoneReviewSweepV8 {subject_label}")
            sweeps.append(
                write_bone_review_sweep_v8(
                    subject_label=subject_label,
                    subject=subject,
                    baseline_subject=baseline_subject,
                    output_dir=(
                        output / "bone_review_pack_v8" / "sweeps" / subject_label
                    ),
                )
            )
    if selected is None:
        manifest = write_bone_review_pack_manifest_v8(
            output_dir=output / "bone_review_pack_v8",
            operator_runtime_digest=operator.runtime_digest(validate=False),
            operator_manifest=operator_path / "manifest.json",
            cells=cells,
            sweeps=sweeps,
        )
        print(f"BoneReviewPackV8 cells={len(cells)} publishable=false -> {manifest}")
    else:
        print(f"BoneReviewCellV8 complete -> {selected[0]}/{selected[1]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="create the fail-closed candidate")
    build.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.set_defaults(func=_run_build)
    render = commands.add_parser("render", help="render the 2x3 Genesis review matrix")
    render.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    render.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX_MODEL)
    render.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    render.add_argument("--resolution", type=int, default=640)
    render.add_argument("--only-cell", default="")
    render.set_defaults(func=_run_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
