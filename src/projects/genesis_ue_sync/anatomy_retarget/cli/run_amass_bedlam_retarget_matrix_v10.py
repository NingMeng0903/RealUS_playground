"""BEDLAM2+AMASS beta×pose matrix for V10 joint-anchored FK.

For each beta×pose cell, require:
  - predicted pivot error < 2 mm (all 235 controllers)
  - hinge seating error < 2 mm (12 anatomical joints)
  - terminal hand/foot non-regression vs 142
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_OPERATOR_RUNTIME_DIGEST,
    EXPECTED_ORACLE_SHA256,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_gates_v10 import (
    evaluate_terminal_pose_regression_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_amass_bedlam_retarget_matrix_v6 import (
    FROZEN_AMASS_MOTIONS,
    FROZEN_BEDLAM2_MOTIONS,
    _load_capture_betas,
    _pick_frame,
    _resolve_bedlam_file,
    _sample_bedlam_betas,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import (
    POSE_MAP_V10_COMPOSITION,
    apply_pose_map_global_v10,
    build_pose_map_v10,
    check_pose_map_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_pivot_diag_v10 import (
    evaluate_pose_pivot_diag_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.segment_similarity_rest_v10 import (
    apply_segment_similarity_to_subject_v10,
    build_segment_similarity_rest_v10,
    controller_segment_scales_v10,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v10_artifacts import (
    load_chain_retarget_v10_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    FROZEN_CAPTURE_SHA256,
    build_whole_chain_rest_fit_v1,
    load_whole_chain_rest_fit_v1,
)


MATRIX_KIND = "AmassBedlamRetargetMatrixV10"
MATRIX_SCHEMA = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument(
        "--v10-shadow",
        type=Path,
        required=True,
        help="Existing chain_retarget_v10_* root with subject_* folders",
    )
    parser.add_argument(
        "--v7-baseline",
        type=Path,
        default=None,
        help="Optional V7 baseline used when a beta has no V10 subject yet",
    )
    parser.add_argument(
        "--bedlam2-motions",
        type=Path,
        default=Path(
            "/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/bedlam2/motions"
        ),
    )
    parser.add_argument(
        "--amass-hf-root",
        type=Path,
        default=Path("/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/amass_hf/raw"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-bedlam-betas", type=int, default=2)
    parser.add_argument(
        "--apply-segment-similarity",
        action="store_true",
        help="Apply N3 segment similarity for betas without a prebuilt V10 subject",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite matrix: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    started = time.perf_counter()
    try:
        operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
        if operator.runtime_digest(validate=False) != EXPECTED_OPERATOR_RUNTIME_DIGEST:
            raise ValueError("matrix requires frozen 142 operator")
        calibration = load_anatomical_calibration_v1(
            args.calibration.expanduser().resolve(),
            operator=operator,
            required_scope="full_main_chain",
        )
        oracle = args.oracle.expanduser().resolve()
        if _sha256(oracle) != EXPECTED_ORACLE_SHA256:
            raise ValueError("matrix requires frozen Blender oracle")
        model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
        model = load_smplx_model_v7(model_path)
        captures = {
            "213328": args.capture_213328.expanduser().resolve(),
            "213712": args.capture_213712.expanduser().resolve(),
        }
        capture_sha256s = {label: _sha256(path) for label, path in captures.items()}
        if capture_sha256s != FROZEN_CAPTURE_SHA256:
            raise ValueError("capture digests differ from frozen pair")

        bedlam_root = args.bedlam2_motions.expanduser().resolve()
        amass_root = args.amass_hf_root.expanduser().resolve()
        v10_root = args.v10_shadow.expanduser().resolve()

        beta_specs: list[dict[str, Any]] = []
        exclude_betas: list[np.ndarray] = []
        for label, path in captures.items():
            betas = _load_capture_betas(path)
            subject_dir = v10_root / f"subject_{label}"
            beta_specs.append(
                {
                    "label": label,
                    "source": "capture",
                    "betas": betas,
                    "subject_dir": subject_dir if subject_dir.is_dir() else None,
                }
            )
            exclude_betas.append(betas)
        for stem, betas in _sample_bedlam_betas(
            bedlam_root, args.extra_bedlam_betas, exclude=exclude_betas
        ):
            beta_specs.append(
                {
                    "label": f"bedlam_{stem}",
                    "source": "bedlam2",
                    "betas": betas,
                    "subject_dir": None,
                }
            )

        motion_cells: list[dict[str, Any]] = [
            {"pose_id": "tpose", "source": "synthetic", "kind": "tpose"},
            {"pose_id": "pose_213328", "source": "capture", "kind": "capture"},
            {"pose_id": "pose_213712", "source": "capture", "kind": "capture"},
        ]
        for name, kind in zip(
            FROZEN_BEDLAM2_MOTIONS, ("upper", "upper", "lower", "lower")
        ):
            path = _resolve_bedlam_file(bedlam_root, name)
            with np.load(path, allow_pickle=False) as data:
                frame_index, pose55 = _pick_frame(data["poses"], kind=kind)
            motion_cells.append(
                {
                    "pose_id": f"bedlam2_{path.stem}_{kind}_f{frame_index}",
                    "source": "bedlam2",
                    "kind": kind,
                    "path": str(path),
                    "frame_index": frame_index,
                    "pose55": pose55,
                    "path_sha256": _sha256(path),
                }
            )
        for rel in FROZEN_AMASS_MOTIONS:
            path = (amass_root / rel).resolve()
            if not path.is_file():
                matches = sorted(amass_root.rglob(Path(rel).name))
                if not matches:
                    raise FileNotFoundError(f"AMASS motion missing: {rel}")
                path = matches[0]
            with np.load(path, allow_pickle=False) as data:
                frame_index, pose55 = _pick_frame(data["poses"], kind="full")
            motion_cells.append(
                {
                    "pose_id": f"amass_{path.stem}_f{frame_index}",
                    "source": "amass_hf",
                    "kind": "full",
                    "path": str(path),
                    "frame_index": frame_index,
                    "pose55": pose55,
                    "path_sha256": _sha256(path),
                }
            )

        capture_poses: dict[str, np.ndarray] = {
            "tpose": np.zeros((55, 3), dtype=np.float32),
        }
        for label, path in captures.items():
            with np.load(path, allow_pickle=False) as data:
                capture_poses[f"pose_{label}"] = easymocap_fit_to_smplx55(
                    data["Rh"], data["poses"], model_path=model_path
                )

        cells_out: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for beta_spec in beta_specs:
            label = str(beta_spec["label"])
            betas = np.asarray(beta_spec["betas"], dtype=np.float64)
            subject_dir = beta_spec["subject_dir"]
            segment_scales = None
            if subject_dir is not None:
                value, meta = load_chain_retarget_v10_subject(subject_dir)
                segment_scales = meta.get("segment_scales")
                rest_report = {"passed": True, "loaded_from": str(subject_dir)}
            else:
                # Build from V7-style whole-chain rest, then optional similarity.
                value = build_whole_chain_rest_fit_v1(
                    operator,
                    calibration,
                    betas=betas,
                    subject_label=label,
                    capture_sha256="matrix_extra_beta",
                    smplx_model=model,
                    smplx_model_sha256=model_sha,
                )
                asset_tmp = materialize_subject(
                    operator, betas=betas, gender="male"
                ).rigged_asset
                if args.apply_segment_similarity:
                    similarity = build_segment_similarity_rest_v10(
                        value, asset=asset_tmp, calibration=calibration
                    )
                    value = apply_segment_similarity_to_subject_v10(value, similarity)
                    segment_scales = controller_segment_scales_v10(similarity)
                rest_report = {"passed": True, "built_for_matrix": True}

            asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
            pose_map = build_pose_map_v10(
                value,
                asset=asset,
                calibration=calibration,
                oracle_path=oracle,
                source_operator_digest=operator.runtime_digest(validate=False),
            )
            pose_check = check_pose_map_v10(pose_map, value, source_asset=asset)

            poses_for_eval: dict[str, np.ndarray] = {}
            for motion in motion_cells:
                pose_id = str(motion["pose_id"])
                if pose_id in capture_poses:
                    poses_for_eval[pose_id] = capture_poses[pose_id]
                else:
                    poses_for_eval[pose_id] = np.asarray(
                        motion["pose55"], dtype=np.float32
                    )

            posed_globals = {
                name: apply_pose_map_global_v10(
                    pose_map,
                    source_asset=asset,
                    pose_axis_angle=pose,
                    segment_scales=segment_scales,
                )
                for name, pose in poses_for_eval.items()
            }
            pivot = evaluate_pose_pivot_diag_v10(
                pose_map,
                source_asset=asset,
                calibration=calibration,
                poses=poses_for_eval,
                composition=POSE_MAP_V10_COMPOSITION,
                posed_globals=posed_globals,
            )
            terminal = evaluate_terminal_pose_regression_v10(
                value,
                pose_map,
                asset=asset,
                smplx_model=model,
                poses=poses_for_eval,
                segment_scales=segment_scales,
            )
            tube_ok = int(value.build_report.get("tube_transport_application_count", 1)) == 1
            cell = {
                "beta_label": label,
                "beta_source": beta_spec["source"],
                "betas": betas.tolist(),
                "rest_passed": bool(rest_report.get("passed", False)),
                "pose_map_passed": bool(pose_check.get("passed")),
                "pivot_passed": bool(pivot.get("passed")),
                "max_predicted_pivot_error_m": float(pivot["max_predicted_error_m"]),
                "max_hinge_seating_error_m": float(pivot["max_hinge_seating_error_m"]),
                "tube_transport_application_count_ok": tube_ok,
                "terminal_passed": bool(terminal["passed"]),
                "terminal": {
                    pose: {
                        "passed": cell["passed"],
                        "hand_foot_mean_delta": cell["hand_foot_mean_delta"],
                        "hand_foot_mean_candidate": cell["hand_foot_mean_candidate"],
                        "hand_foot_mean_baseline_142": cell[
                            "hand_foot_mean_baseline_142"
                        ],
                        "n_collapse": int(cell["n_collapse"]),
                    }
                    for pose, cell in terminal["cells"].items()
                },
                "pivot_by_pose": {
                    pose: {
                        "max_predicted_error_m": row["max_predicted_error_m"],
                        "passed": row["passed"],
                    }
                    for pose, row in pivot["poses"].items()
                },
            }
            cells_out.append(cell)
            ok = (
                cell["rest_passed"]
                and cell["pose_map_passed"]
                and cell["pivot_passed"]
                and cell["terminal_passed"]
                and tube_ok
            )
            if not ok:
                failures.append(cell)
            _write_json(temporary / f"cell_{label}.json", cell)

        matrix = {
            "schema_version": MATRIX_SCHEMA,
            "artifact_kind": MATRIX_KIND,
            "publishable": False,
            "trusted_latest_updated": False,
            "vessel_repair_started": False,
            "among_us_copied_into_realus": False,
            "bedlam2_motions_root": str(bedlam_root),
            "amass_hf_root": str(amass_root),
            "v10_shadow": str(v10_root),
            "pose_map_composition": POSE_MAP_V10_COMPOSITION,
            "frozen_bedlam2_motion_names": list(FROZEN_BEDLAM2_MOTIONS),
            "frozen_amass_motion_rels": list(FROZEN_AMASS_MOTIONS),
            "n_beta": len(beta_specs),
            "n_pose_per_beta": len(motion_cells),
            "n_cells": len(cells_out),
            "passed": len(failures) == 0 and len(cells_out) == len(beta_specs),
            "failures": failures,
            "cells": cells_out,
            "elapsed_seconds": float(time.perf_counter() - started),
            "smplx_model_sha256": model_sha,
        }
        _write_json(temporary / "manifest.json", matrix)
        catalog = []
        for motion in motion_cells:
            entry = {k: v for k, v in motion.items() if k != "pose55"}
            catalog.append(entry)
        _write_json(temporary / "motion_catalog.json", {"motions": catalog})
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"{MATRIX_KIND} passed={matrix['passed']} "
        f"cells={matrix['n_cells']} seconds={matrix['elapsed_seconds']:.3f} -> {output}"
    )
    return 0 if matrix["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
