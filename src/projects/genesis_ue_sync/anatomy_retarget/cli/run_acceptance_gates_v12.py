"""Run every V12 acceptance gate for a candidate over 2 betas x N poses.

Three gate families, all measured independently of anything the candidate
reports about itself, all referenced to the raw 142 materialize (Pack A):

* absolute posed poke-through per bone group (absolute_poke_v12)
* hip seating, section 3.3 hinge-axis error, ankle mortise
  (joint_plausibility_v12)
* posed vessel/nerve linkage invariants (linkage_v12)

Pose set is T-pose, both frozen captures, and the synthetic deep-flex poses,
because the captures alone never reach a deep right knee or a simultaneous
knee-and-elbow bend.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.absolute_poke_v12 import (
    absolute_poke_metrics,
    compare_absolute_poke_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
from projects.genesis_ue_sync.anatomy_retarget.cli.render_acceptance_pack_v12 import (
    _load_candidate,
)
from projects.genesis_ue_sync.anatomy_retarget.deep_flex_poses_v12 import (
    build_deep_flex_poses_v12,
    measure_hinges_deg,
    verify_deep_flex_poses_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_plausibility_v12 import (
    compare_joint_plausibility_v12,
    joint_plausibility_metrics_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.linkage_v12 import (
    evaluate_linkage_v12,
    tube_bone_offset_metrics_v12,
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
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeatable; PATH holds subject_<label> dirs (V1, V10 or V11)",
    )
    parser.add_argument("--subjects", default="213328,213712")
    parser.add_argument("--deep-flex-deg", type=float, default=120.0)
    parser.add_argument("--no-deep-flex", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _posed_vertices(
    kind: str,
    value: Any,
    pose_map: Any,
    *,
    asset: Any,
    pose_aa: np.ndarray,
) -> np.ndarray:
    if kind == "pack_a":
        return np.asarray(skin_vertices(asset, pose_aa), dtype=np.float32)
    if kind == "v1":
        vertices, _ = pose_whole_chain_vertices(
            value, pose_map, source_asset=asset, pose_axis_angle=pose_aa
        )
        return vertices
    vertices, _ = pose_whole_chain_vertices_v10(
        value, pose_map, source_asset=asset, pose_axis_angle=pose_aa
    )
    return vertices


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    candidates: dict[str, Path] = {}
    for entry in args.candidate:
        if "=" not in entry:
            raise SystemExit(f"--candidate needs NAME=PATH, got {entry!r}")
        name, _, path = entry.partition("=")
        candidates[name.strip()] = Path(path.strip()).expanduser().resolve()

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

    capture_paths = {
        "213328": args.capture_213328.expanduser().resolve(),
        "213712": args.capture_213712.expanduser().resolve(),
    }
    betas_by_subject: dict[str, np.ndarray] = {}
    capture_poses: dict[str, np.ndarray] = {}
    for label, path in capture_paths.items():
        with np.load(path, allow_pickle=False) as data:
            betas_by_subject[label] = np.asarray(
                data["shapes"], dtype=np.float64
            ).reshape(-1)[:10]
            capture_poses[f"pose_{label}"] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )

    report: dict[str, Any] = {
        "schema_version": 12,
        "artifact_kind": "AcceptanceGatesV12",
        "publishable": False,
        "trusted_latest_updated": False,
        "reference": "pack_a_142_materialize",
        "subjects": {},
    }
    verdicts: dict[str, dict[str, bool]] = {}

    for label in [part.strip() for part in args.subjects.split(",") if part.strip()]:
        betas = betas_by_subject[label]
        asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
        rest = np.asarray(asset.vertices_rest, dtype=np.float64)

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
            poses.update(synthetic)

        skins = {
            name: smplx_body_surface_v7(model, betas=betas, pose_axis_angle=pose)
            for name, pose in poses.items()
        }
        stations = {name: joints_of(pose) for name, pose in poses.items()}

        measured: dict[str, dict[str, Any]] = {}
        for name, root in {"pack_a": None, **candidates}.items():
            if root is None:
                kind, value, pose_map = "pack_a", None, None
            else:
                value, kind = _load_candidate(
                    root,
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
            poke: dict[str, Any] = {}
            joints: dict[str, Any] = {}
            linkage: dict[str, Any] = {}
            for pose_name, pose in poses.items():
                pose_aa = np.asarray(pose, dtype=np.float32).reshape(55, 3)
                vertices = _posed_vertices(
                    kind, value, pose_map, asset=asset, pose_aa=pose_aa
                )
                skin, skin_faces = skins[pose_name]
                poke[pose_name] = absolute_poke_metrics(
                    vertices, asset=asset, skin=skin, skin_faces=skin_faces
                )
                joints[pose_name] = joint_plausibility_metrics_v12(
                    vertices, calibration=calibration, smplx_joints=stations[pose_name]
                )
                linkage[pose_name] = tube_bone_offset_metrics_v12(
                    rest, vertices, asset=asset
                )
            measured[name] = {
                "artifact_format": kind,
                "poke": poke,
                "joints": joints,
                "linkage": linkage,
            }
            print(f"  measured {name} on subject_{label}", flush=True)

        subject_report: dict[str, Any] = {"candidates": {}}
        for name, blocks in measured.items():
            poke_gate = compare_absolute_poke_v12(
                blocks["poke"], reference=measured["pack_a"]["poke"]
            )
            joint_gate = compare_joint_plausibility_v12(
                blocks["joints"], reference=measured["pack_a"]["joints"]
            )
            linkage_gate = evaluate_linkage_v12(
                blocks["linkage"], reference=measured["pack_a"]["linkage"]
            )
            passed = bool(
                poke_gate["passed"] and joint_gate["passed"] and linkage_gate["passed"]
            )
            subject_report["candidates"][name] = {
                "artifact_format": blocks["artifact_format"],
                "passed": passed,
                "absolute_poke_v12": poke_gate,
                "joint_plausibility_v12": joint_gate,
                "linkage_v12": linkage_gate,
            }
            verdicts.setdefault(name, {})[label] = passed
            print(
                f"    {name:12s} subject_{label} passed={passed} "
                f"poke={poke_gate['passed']} joints={joint_gate['passed']} "
                f"linkage={linkage_gate['passed']}",
                flush=True,
            )
        subject_report["hinge_angles_deg"] = {
            name: measure_hinges_deg(pose, joints_of=joints_of)
            for name, pose in poses.items()
        }
        report["subjects"][label] = subject_report

    report["verdicts"] = {
        name: {"per_subject": cells, "passed": all(cells.values())}
        for name, cells in verdicts.items()
    }
    report["elapsed_seconds"] = float(time.perf_counter() - started)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== verdicts ===")
    for name, entry in report["verdicts"].items():
        print(f"{name:12s} passed={entry['passed']}  {entry['per_subject']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
