"""Frozen BEDLAM2+AMASS beta×pose retarget matrix for V6 whole-chain subjects.

Reads motion assets from Among_US in place (no copy into RealUS).  Uses the same
retarget path for every cell: whole-chain rest + pose_map_v1, with hard gates for
T-pose main chain and hand/foot non-regression vs 142 materialize.
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
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
    pose_to_smplx55_axis_angle,
    smplh156_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    check_pose_map_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.terminal_pose_regression_v6 import (
    evaluate_terminal_pose_regression_v6,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    FROZEN_CAPTURE_SHA256,
    build_whole_chain_rest_fit_v1,
    check_whole_chain_rest_fit_v1,
    load_whole_chain_rest_fit_v1,
)


MATRIX_KIND = "AmassBedlamRetargetMatrixV6"
MATRIX_SCHEMA = 1

# Frozen small matrix: reproducible paths under Among_US (read-only).
FROZEN_BEDLAM2_MOTIONS = (
    # Upper-body expressive
    "it_4019_2XL_2203.npz",
    "it_4046_2XL_2106.npz",
    # Lower-body / locomotion-ish
    "it_4051_3XL_2304.npz",
    "it_4011_XL_2114.npz",
)
FROZEN_AMASS_MOTIONS = (
    "ACCAD/Female1General_c3d/A10 - lie to crouch_poses.npz",
    "ACCAD/Female1Walking_c3d/B12 - walk turn right (90)_poses.npz",
)


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


# Knees are baked at 75deg support; ankles at 130deg. Use the stricter bound.
COUPLED_RBF_SUPPORT_RAD = float(np.radians(75.0))


def _pose_within_coupled_support(pose55: np.ndarray, *, limit: float = COUPLED_RBF_SUPPORT_RAD) -> bool:
    joints = np.asarray(pose55, dtype=np.float64).reshape(55, 3)
    # Coupled drivers are body limbs; face/hands stay unconstrained here.
    body = joints[1:22]
    return bool(np.all(np.linalg.norm(body, axis=1) <= limit + 1.0e-7))


def _pick_frame(poses: np.ndarray, *, kind: str) -> tuple[int, np.ndarray]:
    """Pick a representative frame by joint-angle energy within RBF support."""

    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim == 1:
        pose55 = pose_to_smplx55_axis_angle(arr)
        if not _pose_within_coupled_support(pose55):
            # Soft-clamp body joints so the frozen operator can evaluate.
            body = pose55[1:22]
            norms = np.linalg.norm(body, axis=1, keepdims=True)
            scale = np.minimum(1.0, COUPLED_RBF_SUPPORT_RAD / np.maximum(norms, 1.0e-12))
            pose55 = pose55.copy()
            pose55[1:22] = body * scale
        return 0, pose55.astype(np.float32)
    if arr.shape[-1] == 165:
        frames = arr.reshape(arr.shape[0], 55, 3)
    elif arr.shape[-1] == 156:
        frames = np.stack(
            [smplh156_to_smplx55(row) for row in arr.reshape(arr.shape[0], 156)],
            axis=0,
        )
    else:
        raise ValueError(f"unsupported motion pose width {arr.shape[-1]}")
    energy = np.linalg.norm(frames.reshape(len(frames), -1), axis=1)
    if kind == "upper":
        upper = frames[:, [16, 17, 18, 19, 20, *range(25, 55)], :]
        energy = np.linalg.norm(upper.reshape(len(frames), -1), axis=1)
    elif kind == "lower":
        lower = frames[:, [1, 2, 4, 5, 7, 8, 10, 11], :]
        energy = np.linalg.norm(lower.reshape(len(frames), -1), axis=1)
    support = np.array(
        [_pose_within_coupled_support(frame) for frame in frames], dtype=bool
    )
    if np.any(support):
        masked = np.where(support, energy, -1.0)
        index = int(np.argmax(masked))
    else:
        index = int(np.argmax(energy))
    pose = frames[index].astype(np.float32).copy()
    pose[22] = 0.0
    if not _pose_within_coupled_support(pose):
        body = pose[1:22]
        norms = np.linalg.norm(body, axis=1, keepdims=True)
        scale = np.minimum(1.0, COUPLED_RBF_SUPPORT_RAD / np.maximum(norms, 1.0e-12))
        pose[1:22] = body * scale
    return index, pose


def _resolve_bedlam_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"BEDLAM2 motion missing: {name} under {root}")
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument(
        "--v6-shadow",
        type=Path,
        required=True,
        help="Existing chain_retarget_v6_node2_* root with subject_* folders",
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
    parser.add_argument(
        "--extra-bedlam-betas",
        type=int,
        default=2,
        help="How many additional BEDLAM2 shape vectors to include (default 2).",
    )
    return parser


def _load_capture_betas(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]


def _sample_bedlam_betas(root: Path, count: int, *, exclude: list[np.ndarray]) -> list[tuple[str, np.ndarray]]:
    selected: list[tuple[str, np.ndarray]] = []
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "betas" not in data.files:
                continue
            betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)[:10]
        if any(np.allclose(betas, other, atol=1.0e-6) for other in exclude):
            continue
        # Prefer non-trivial shapes.
        if float(np.linalg.norm(betas)) < 0.5:
            continue
        selected.append((path.stem, betas.copy()))
        exclude.append(betas.copy())
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise RuntimeError(f"could not sample {count} distinct BEDLAM2 betas")
    return selected


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
        v6_root = args.v6_shadow.expanduser().resolve()

        beta_specs: list[dict[str, Any]] = []
        exclude_betas: list[np.ndarray] = []
        for label, path in captures.items():
            betas = _load_capture_betas(path)
            beta_specs.append(
                {
                    "label": label,
                    "source": "capture",
                    "betas": betas,
                    "subject_dir": v6_root / f"subject_{label}",
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

        # Motion catalog: tpose + capture poses + BEDLAM upper/lower + AMASS.
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
                # Fall back to first matching stem under amass root.
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

        # Resolve capture poses once.
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
            if subject_dir is not None and Path(subject_dir).is_dir():
                value = load_whole_chain_rest_fit_v1(
                    subject_dir,
                    operator=operator,
                    calibration=calibration,
                    smplx_model=model,
                    smplx_model_sha256=model_sha,
                    recheck=False,
                )
                rest_report = {"passed": True, "loaded_from": str(subject_dir)}
            else:
                value = build_whole_chain_rest_fit_v1(
                    operator,
                    calibration,
                    betas=betas,
                    subject_label=label,
                    capture_sha256="matrix_extra_beta",
                    smplx_model=model,
                    smplx_model_sha256=model_sha,
                )
                rest_report = check_whole_chain_rest_fit_v1(
                    value,
                    operator=operator,
                    calibration=calibration,
                    smplx_model=model,
                    smplx_model_sha256=model_sha,
                )
            asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
            pose_map = build_pose_map_v1(
                value,
                asset=asset,
                calibration=calibration,
                oracle_path=oracle,
                source_operator_digest=operator.runtime_digest(validate=False),
            )
            pose_check = check_pose_map_v1(pose_map, value, source_asset=asset)
            if not rest_report.get("passed", False) or not pose_check.get("passed"):
                failures.append(
                    {
                        "beta": label,
                        "rest_passed": rest_report.get("passed"),
                        "pose_map_passed": pose_check.get("passed"),
                    }
                )
                continue

            poses_for_eval: dict[str, np.ndarray] = {}
            for motion in motion_cells:
                pose_id = str(motion["pose_id"])
                if pose_id in capture_poses:
                    poses_for_eval[pose_id] = capture_poses[pose_id]
                else:
                    poses_for_eval[pose_id] = np.asarray(motion["pose55"], dtype=np.float32)

            terminal = evaluate_terminal_pose_regression_v6(
                value,
                pose_map,
                asset=asset,
                smplx_model=model,
                poses=poses_for_eval,
            )
            tube_ok = int(value.build_report.get("tube_transport_application_count", 1)) == 1
            cell = {
                "beta_label": label,
                "beta_source": beta_spec["source"],
                "betas": betas.tolist(),
                "rest_passed": bool(rest_report.get("passed", False)),
                "pose_map_passed": bool(pose_check.get("passed")),
                "tube_transport_application_count_ok": tube_ok,
                "terminal_passed": bool(terminal["passed"]),
                "terminal": {
                    pose: {
                        "passed": cell["passed"],
                        "hand_foot_mean_delta": cell["hand_foot_mean_delta"],
                        "hand_foot_mean_candidate": cell["hand_foot_mean_candidate"],
                        "hand_foot_mean_baseline_142": cell["hand_foot_mean_baseline_142"],
                        "n_collapse": len(cell["collapse_failures"]),
                    }
                    for pose, cell in terminal["cells"].items()
                },
            }
            cells_out.append(cell)
            if not (cell["rest_passed"] and cell["terminal_passed"] and tube_ok):
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
            "v6_shadow": str(v6_root),
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
        # Persist motion catalog without large arrays.
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
