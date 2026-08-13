"""Independent review handoff: the section 9.4 tree and the 9.5 image set.

Nothing in the repo produced ``independent_genesis_review*`` output, so the
section 11 stop condition ("10-image independent handoff, agent status
accepted_for_user_genesis_review") could not physically be reached.  This CLI
builds that pack from an acceptance render pack plus a hinge sweep it renders
itself.

Coverage is reported item by item.  A missing required image sets the decision
to ``needs_rerender``; the pack never silently ships short.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _measure_frames,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
from projects.genesis_ue_sync.anatomy_retarget.cli.render_acceptance_pack_v12 import (
    _load_candidate,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.render_stage1_baseline_compare_v1 import (
    _contact_sheet,
    _render_modes,
)
from projects.genesis_ue_sync.anatomy_retarget.deep_flex_poses_v12 import (
    build_hinge_sweep_poses_v12,
    measure_hinges_deg,
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
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)


# Section 9.5 tile list for images 1-6.  Every entry is (layer, camera).
MATRIX_TILES: tuple[tuple[str, str], ...] = (
    ("bones_only", "whole_ap"),
    ("bones_only", "whole_lateral"),
    ("outside_heatmap", "whole_ap"),
    ("outside_heatmap", "pelvis_context"),
    ("bones_only", "left_knee_ap"),
    ("bones_only", "left_knee_lateral"),
    ("bones_only", "right_knee_ap"),
    ("outside_heatmap", "left_knee_ap"),
    ("bones_only", "left_elbow_lateral"),
    ("outside_heatmap", "left_hand_oblique"),
    ("outside_heatmap", "left_foot_oblique"),
    ("outside_heatmap", "feet_top"),
)

SWEEP_JOINT_CAMERAS = (
    "left_knee_lateral",
    "right_knee_lateral",
    "left_elbow_lateral",
    "right_elbow_lateral",
)
SWEEP_ANGLES_DEG = (0.0, 40.0, 80.0, 120.0)

LINKAGE_TILES: tuple[tuple[str, str], ...] = (
    ("bones_tubes", "whole_ap"),
    ("bones_tubes", "whole_lateral"),
    ("bones_tubes", "pelvis_context"),
    ("bones_tubes", "left_knee_ap"),
    ("bones_tubes", "left_elbow_lateral"),
    ("bones_tubes", "left_hand_oblique"),
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
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument(
        "--acceptance-pack",
        type=Path,
        required=True,
        help="output of render_acceptance_pack_v12 (supplies images 1-6, 10)",
    )
    parser.add_argument("--subjects", default="213328,213712")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _slim_tile(pack: Path, subject: str, pose: str, layer: str, camera: str) -> Path:
    return pack / "slim" / f"subject_{subject}" / pose / layer / f"{camera}.png"


def _compose(
    tiles: Sequence[tuple[Path, str]], target: Path
) -> tuple[bool, list[str]]:
    """Write one contact sheet; report which tiles were unavailable."""

    present = [(path, label) for path, label in tiles if path.is_file()]
    missing = [label for path, label in tiles if not path.is_file()]
    if not present:
        return False, [label for _path, label in tiles]
    target.parent.mkdir(parents=True, exist_ok=True)
    _contact_sheet(
        [path for path, _label in present],
        [label for _label, label in ((p, l) for p, l in present)],
        target,
        {},
    )
    return len(missing) == 0, missing


def _render_sweep(
    *,
    args: argparse.Namespace,
    subject: str,
    output: Path,
) -> dict[str, Any]:
    """Render the hinge sweep this pack needs for images 7 and 8."""

    operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.expanduser().resolve(),
        operator=operator,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    regressor = np.asarray(model["J_regressor"], dtype=np.float64)

    capture_paths = {
        "213328": args.capture_213328.expanduser().resolve(),
        "213712": args.capture_213712.expanduser().resolve(),
    }
    capture_poses: dict[str, np.ndarray] = {}
    betas: np.ndarray | None = None
    for label, path in capture_paths.items():
        with np.load(path, allow_pickle=False) as data:
            if label == subject:
                betas = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
            capture_poses[label] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )
    if betas is None:
        raise KeyError(f"no capture for subject {subject}")

    asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset

    def joints_of(pose: np.ndarray) -> np.ndarray:
        skin, _faces = smplx_body_surface_v7(
            model,
            betas=betas,
            pose_axis_angle=np.asarray(pose, dtype=np.float32).reshape(55, 3),
        )
        return regressor @ np.asarray(skin, dtype=np.float64)

    value, artifact_format = _load_candidate(
        args.candidate.expanduser().resolve(),
        subject,
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
        oracle_path=args.oracle.expanduser().resolve(),
        source_operator_digest=operator.runtime_digest(validate=False),
    )

    sweep_poses = build_hinge_sweep_poses_v12(
        captures=capture_poses, joints_of=joints_of, angles_deg=SWEEP_ANGLES_DEG
    )
    cells: dict[str, Any] = {}
    for pose_name, pose in sweep_poses.items():
        pose_aa = np.asarray(pose, dtype=np.float32).reshape(55, 3)
        if artifact_format == "v1":
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
        cell_dir = output / pose_name
        cell_dir.mkdir(parents=True, exist_ok=True)
        cells[pose_name] = _render_modes(
            output=cell_dir,
            vertices=np.asarray(vertices, dtype=np.float32),
            asset=asset,
            skin=skin,
            skin_faces=skin_faces,
            frames=frames,
            backend=args.backend,
            camera_names=SWEEP_JOINT_CAMERAS,
            include_acceptance_views=True,
        )
        cells[pose_name]["measured_hinges_deg"] = measure_hinges_deg(
            pose_aa, joints_of=joints_of
        )
        # Pack-A ghost so a reviewer can see what the sweep looks like with no
        # rest fit at all, per the section 9.5 "142 baseline ghost" ask.
        ghost = np.asarray(skin_vertices(asset, pose_aa), dtype=np.float32)
        ghost_dir = output / f"{pose_name}_142_ghost"
        ghost_dir.mkdir(parents=True, exist_ok=True)
        cells[f"{pose_name}_142_ghost"] = _render_modes(
            output=ghost_dir,
            vertices=ghost,
            asset=asset,
            skin=skin,
            skin_faces=skin_faces,
            frames=frames,
            backend=args.backend,
            camera_names=SWEEP_JOINT_CAMERAS,
            include_acceptance_views=True,
        )
        print(f"  sweep {pose_name} rendered", flush=True)
    return {"artifact_format": artifact_format, "cells": cells}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing review: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    pack = args.acceptance_pack.expanduser().resolve()
    pack_manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    subjects = [part.strip() for part in str(args.subjects).split(",") if part.strip()]

    handoff = output / "handoff"
    handoff.mkdir(parents=True, exist_ok=True)
    coverage: list[dict[str, Any]] = []
    index = 0

    # Images 1-6: two betas x three poses.
    for subject in subjects:
        for pose in ("tpose", "pose_213328", "pose_213712"):
            index += 1
            tiles = [
                (
                    _slim_tile(pack, subject, pose, layer, camera),
                    f"{layer}/{camera}",
                )
                for layer, camera in MATRIX_TILES
            ]
            target = handoff / f"{index:02d}_beta{subject}_{pose}.png"
            complete, missing = _compose(tiles, target)
            coverage.append(
                {
                    "image": target.name,
                    "requirement": "section_9_5_matrix_cell",
                    "required": True,
                    "complete": complete,
                    "missing_tiles": missing,
                    "sha256": _sha256(target) if target.is_file() else None,
                }
            )

    # Images 7-8: bilateral knee/elbow sweep strips, one per beta.
    sweeps: dict[str, Any] = {}
    for subject in subjects:
        index += 1
        sweep_root = output / "sweeps" / f"subject_{subject}"
        sweep_root.mkdir(parents=True, exist_ok=True)
        print(f"rendering sweep for subject_{subject}", flush=True)
        sweeps[subject] = _render_sweep(
            args=args, subject=subject, output=sweep_root
        )
        tiles: list[tuple[Path, str]] = []
        for camera in SWEEP_JOINT_CAMERAS:
            for angle in SWEEP_ANGLES_DEG:
                pose_name = f"sweep_{int(round(angle)):03d}"
                tiles.append(
                    (
                        sweep_root / pose_name / "bones_only" / "rgb" / f"{camera}.png",
                        f"{camera}@{int(round(angle))}deg",
                    )
                )
                tiles.append(
                    (
                        sweep_root
                        / f"{pose_name}_142_ghost"
                        / "bones_only"
                        / "rgb"
                        / f"{camera}.png",
                        f"{camera}@{int(round(angle))}deg_142",
                    )
                )
        target = handoff / f"{index:02d}_beta{subject}_sweeps.png"
        complete, missing = _compose(tiles, target)
        coverage.append(
            {
                "image": target.name,
                "requirement": "section_9_5_sweep_strip",
                "required": True,
                "complete": complete,
                "missing_tiles": missing,
                "sha256": _sha256(target) if target.is_file() else None,
            }
        )

    # Image 9: section 9.5 asks for a fixed bed/robot context purely as a scale
    # check.  That scene only exists inside the rejected bone_review_pack_v8
    # (candidate-driven cameras, section 9.1), so this substitutes a whole-body
    # scale reference across both betas and records the substitution.
    index += 1
    scale_tiles = [
        (
            _slim_tile(pack, subject, pose, "bones_only", camera),
            f"beta{subject}/{pose}/{camera}",
        )
        for subject in subjects
        for pose in ("tpose",)
        for camera in ("whole_ap", "whole_lateral")
    ]
    target = handoff / f"{index:02d}_scale_reference.png"
    complete, missing = _compose(scale_tiles, target)
    coverage.append(
        {
            "image": target.name,
            "requirement": "section_9_5_bed_robot_context",
            "required": False,
            "substituted": "whole_body_scale_reference_with_10mm_bar",
            "substitution_reason": (
                "the fixed bed/robot scene only exists in bone_review_pack_v8, "
                "which section 9.1 rejects for candidate-driven cameras"
            ),
            "complete": complete,
            "missing_tiles": missing,
            "sha256": _sha256(target) if target.is_file() else None,
        }
    )

    # Image 10: bones + tubes linkage.
    index += 1
    linkage_tiles = [
        (
            _slim_tile(pack, subjects[0], "pose_213328", layer, camera),
            f"{layer}/{camera}",
        )
        for layer, camera in LINKAGE_TILES
    ]
    target = handoff / f"{index:02d}_bones_tubes_linkage.png"
    complete, missing = _compose(linkage_tiles, target)
    coverage.append(
        {
            "image": target.name,
            "requirement": "section_9_5_bones_tubes_linkage",
            "required": True,
            "complete": complete,
            "missing_tiles": missing,
            "sha256": _sha256(target) if target.is_file() else None,
        }
    )

    incomplete = [
        entry["image"]
        for entry in coverage
        if entry.get("required") and not entry["complete"]
    ]
    decision = "needs_rerender" if incomplete else "ready_for_independent_review"

    (output / "camera_manifest.json").write_text(
        json.dumps(
            {
                "source": pack_manifest.get("camera_source"),
                "candidate_camera_read": pack_manifest.get("candidate_camera_read"),
                "per_cell_sha256": {
                    subject: {
                        pose: cell.get("camera_manifest_sha256")
                        for pose, cell in body["poses"].items()
                    }
                    for subject, body in pack_manifest.get("subjects", {}).items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "review_decision.json").write_text(
        json.dumps(
            {
                "schema_version": 12,
                "artifact_kind": "IndependentReviewHandoffV12",
                "candidate_name": str(args.candidate_name),
                "candidate_root": str(args.candidate.expanduser().resolve()),
                "acceptance_pack": str(pack),
                # The reviewing agent fills in the verdict; this writer only
                # states whether the pack is even reviewable.
                "decision": decision,
                "incomplete_required_images": incomplete,
                "publishable": False,
                "trusted_latest_updated": False,
                "reviewer_scope": "independent_visual_acceptance_only",
                "section_9_5_coverage": coverage,
                "sweep_angles_deg": list(SWEEP_ANGLES_DEG),
                "sweep_measured_hinges_deg": {
                    subject: {
                        pose: cell.get("measured_hinges_deg")
                        for pose, cell in body["cells"].items()
                        if "measured_hinges_deg" in cell
                    }
                    for subject, body in sweeps.items()
                },
                "elapsed_seconds": float(time.perf_counter() - started),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nIndependentReviewHandoffV12 decision={decision} -> {output}")
    return 0 if decision != "needs_rerender" else 1


if __name__ == "__main__":
    raise SystemExit(main())
