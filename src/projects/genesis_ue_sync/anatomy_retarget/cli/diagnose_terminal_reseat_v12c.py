"""213328-only V12c reseat diagnostic: hindfoot / forefoot mm and area."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import source_bone_posed_global
from projects.genesis_ue_sync.anatomy_retarget.deep_flex_poses_v12 import (
    build_deep_flex_poses_v12,
    verify_deep_flex_poses_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v10 import FOOT_ROOTS
from projects.genesis_ue_sync.anatomy_retarget.terminal_reseat_v12 import (
    FOREFOOT_ROOTS,
    reseat_subject_terminals_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.v11_artifacts import (
    load_chain_retarget_v11_subject,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--v11-subject", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.expanduser().resolve(),
        operator=operator,
        required_scope="full_main_chain",
    )
    model_path, _sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    value, _meta = load_chain_retarget_v11_subject(args.v11_subject.expanduser().resolve())
    asset = materialize_subject(
        operator, betas=np.asarray(value.betas), gender="male"
    ).rigged_asset
    recorded: dict[str, np.ndarray] = {}
    for label, path in (
        ("213328", args.capture_213328),
        ("213712", args.capture_213712),
    ):
        with np.load(path.expanduser().resolve(), allow_pickle=False) as data:
            recorded[f"pose_{label}"] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )
    rest_skin, rest_faces = smplx_body_surface_v7(
        model,
        betas=np.asarray(value.betas),
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
    )
    inverse_source_bind = np.linalg.inv(
        np.asarray(
            source_bone_posed_global(asset, np.zeros((55, 3), dtype=np.float32)),
            dtype=np.float64,
        )
    )
    regressor = np.asarray(model["J_regressor"], dtype=np.float64)
    betas = np.asarray(value.betas, dtype=np.float64)

    def joints_of(pose: np.ndarray) -> np.ndarray:
        skin, _faces = smplx_body_surface_v7(
            model,
            betas=betas,
            pose_axis_angle=np.asarray(pose, dtype=np.float32).reshape(55, 3),
        )
        return regressor @ np.asarray(skin, dtype=np.float64)

    synthetic = build_deep_flex_poses_v12(
        captures={"213328": recorded["pose_213328"], "213712": recorded["pose_213712"]},
        joints_of=joints_of,
    )
    verification = verify_deep_flex_poses_v12(synthetic, joints_of=joints_of)
    if not verification["passed"]:
        raise SystemExit(f"deep-flex poses missed their target: {verification['failures']}")
    frames = []
    for pose in (*recorded.values(), *synthetic.values()):
        pose_aa = np.asarray(pose, dtype=np.float32).reshape(55, 3)
        pose_skin, pose_faces = smplx_body_surface_v7(
            model, betas=betas, pose_axis_angle=pose_aa
        )
        frames.append(
            {
                "skin": pose_skin,
                "skin_faces": pose_faces,
                "source_transforms": np.asarray(
                    source_bone_posed_global(asset, pose_aa), dtype=np.float64
                )
                @ inverse_source_bind,
            }
        )
    reseated = reseat_subject_terminals_v12(
        value,
        asset=asset,
        calibration=calibration,
        skin=rest_skin,
        skin_faces=rest_faces,
        pose_frames=frames,
    )
    report = reseated.build_report["terminal_reseat_v12"]
    wanted = (*FOOT_ROOTS, *FOREFOOT_ROOTS)
    summary = {
        "method": report.get("method"),
        "fitted_pose_count": report.get("fitted_pose_count"),
        "clusters": {},
    }
    print("cluster                 Tmm   deg   out_mm  area   frames_mm")
    for name in wanted:
        row = report["clusters"][name]
        summary["clusters"][name] = {
            "translation_m": row["translation_m"],
            "rotation_deg": row["rotation_deg"],
            "max_outside_before_m": row["max_outside_before_m"],
            "max_outside_after_m": row["max_outside_after_m"],
            "outside_area_before": row["outside_area_before"],
            "outside_area_after": row["outside_area_after"],
            "rejected": row["rejected_full_cluster_regression"],
            "frame_max_outside_before_m": row["frame_max_outside_before_m"],
            "frame_max_outside_after_m": row["frame_max_outside_after_m"],
            "frame_outside_area_before": row["frame_outside_area_before"],
            "frame_outside_area_after": row["frame_outside_area_after"],
        }
        print(
            f"{name:22s} {row['translation_m']*1000:5.1f} {row['rotation_deg']:5.1f} "
            f"{row['max_outside_before_m']*1000:5.1f}->{row['max_outside_after_m']*1000:5.1f} "
            f"{row['outside_area_before']:.3f}->{row['outside_area_after']:.3f} "
            f"{[round(v*1000,1) for v in row['frame_max_outside_after_m']]}"
        )
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
