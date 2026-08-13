"""Absolute posed poke-through table: Pack A / V7 / V10-hybrid / V11.

Measurement only — no rest fit, no pose map, no geometry is modified.  This is
the instrument the V12 gate needs: every existing containment gate is relative
to a V7 baseline that already pokes, so nothing in the ladder can express
"small poke-through is fine, large posed poke-through is not".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.absolute_poke_v12 import (
    WHOLE_BODY_GROUP,
    absolute_poke_metrics,
    compare_absolute_poke_v12,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
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


REPORT_GROUPS = (
    WHOLE_BODY_GROUP,
    "femur_L",
    "femur_R",
    "patella_L",
    "shank_L",
    "shank_R",
    "foot_L",
    "foot_R",
    "humerus_L",
    "humerus_R",
    "forearm_L",
    "forearm_R",
    "hand_L",
    "hand_R",
    "pelvis",
    "spine",
    "cervical",
    "thoracic",
    "lumbar",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--v7-shadow", type=Path, required=True)
    parser.add_argument("--hybrid-shadow", type=Path, required=True)
    parser.add_argument("--v11-shadow", type=Path, required=True)
    parser.add_argument("--subject", default="213328")
    parser.add_argument(
        "--deep-flex-deg",
        type=float,
        default=120.0,
        help="anatomical hinge angle for the synthetic deep-flex poses",
    )
    parser.add_argument(
        "--no-deep-flex",
        action="store_true",
        help="captures only; the frozen captures never reach a deep right knee",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _markdown_table(candidates: dict[str, Any], poses: list[str]) -> str:
    lines: list[str] = []
    for pose_name in poses:
        lines.append(f"\n### {pose_name}\n")
        header = "| group | " + " | ".join(candidates) + " |"
        lines.append(header)
        lines.append("|---" * (len(candidates) + 1) + "|")
        for group in REPORT_GROUPS:
            cells = []
            for name in candidates:
                entry = candidates[name][pose_name].get(group)
                if entry is None:
                    cells.append("-")
                    continue
                cells.append(
                    f"{entry['max_outside_m'] * 1000:.1f} / "
                    f"{entry['poke_p95_all_m'] * 1000:.1f} / "
                    f"{entry['outside_area_fraction'] * 100:.1f}%"
                )
            lines.append(f"| {group} | " + " | ".join(cells) + " |")
    lines.append(
        "\nCell = max outside mm / all-vertex p95 outside mm / outside area fraction."
        "\nArea weights come from the shared rest geometry, not the posed vertices.\n"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    label = str(args.subject).strip()
    operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.expanduser().resolve(),
        operator=operator,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    oracle = args.oracle.expanduser().resolve()
    digest = operator.runtime_digest(validate=False)

    with np.load(args.capture_213328.expanduser().resolve(), allow_pickle=False) as data:
        betas = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]

    poses: dict[str, np.ndarray] = {"tpose": np.zeros((55, 3), dtype=np.float32)}
    for name, path in (
        ("pose_213328", args.capture_213328),
        ("pose_213712", args.capture_213712),
    ):
        with np.load(path.expanduser().resolve(), allow_pickle=False) as data:
            poses[name] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )

    regressor = np.asarray(model["J_regressor"], dtype=np.float64)

    def joints_of(pose: np.ndarray) -> np.ndarray:
        skin, _faces = smplx_body_surface_v7(
            model,
            betas=betas,
            pose_axis_angle=np.asarray(pose, dtype=np.float32).reshape(55, 3),
        )
        return regressor @ np.asarray(skin, dtype=np.float64)

    captured_angles = {
        name: measure_hinges_deg(pose, joints_of=joints_of)
        for name, pose in poses.items()
    }
    deep_flex: dict[str, Any] = {"enabled": not args.no_deep_flex}
    if not args.no_deep_flex:
        synthetic = build_deep_flex_poses_v12(
            captures={"213328": poses["pose_213328"], "213712": poses["pose_213712"]},
            joints_of=joints_of,
            target_deg=float(args.deep_flex_deg),
        )
        verification = verify_deep_flex_poses_v12(
            synthetic, joints_of=joints_of, target_deg=float(args.deep_flex_deg)
        )
        if not verification["passed"]:
            raise SystemExit(
                f"deep-flex poses did not reach target: {verification['failures']}"
            )
        deep_flex["verification"] = verification
        poses.update(synthetic)

    asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
    skins = {
        name: smplx_body_surface_v7(model, betas=betas, pose_axis_angle=pose)
        for name, pose in poses.items()
    }

    v7_value = load_whole_chain_rest_fit_v1(
        args.v7_shadow.expanduser().resolve() / f"subject_{label}",
        operator=operator,
        calibration=calibration,
        smplx_model=model,
        smplx_model_sha256=model_sha,
        recheck=False,
    )
    hybrid_value, _hybrid_meta = load_chain_retarget_v10_subject(
        args.hybrid_shadow.expanduser().resolve() / f"subject_{label}"
    )
    v11_value, _v11_meta = load_chain_retarget_v11_subject(
        args.v11_shadow.expanduser().resolve() / f"subject_{label}"
    )
    pose_maps = {
        name: build_pose_map_v1(
            value,
            asset=asset,
            calibration=calibration,
            oracle_path=oracle,
            source_operator_digest=digest,
        )
        for name, value in (
            ("v7", v7_value),
            ("hybrid", hybrid_value),
            ("v11", v11_value),
        )
    }

    candidates: dict[str, dict[str, Any]] = {}
    for name in ("pack_A_142", "v7", "hybrid", "v11"):
        candidates[name] = {}
        for pose_name, pose in poses.items():
            pose_aa = np.asarray(pose, dtype=np.float32).reshape(55, 3)
            if name == "pack_A_142":
                vertices = skin_vertices(asset, pose_aa)
            elif name == "v7":
                vertices, _ = pose_whole_chain_vertices(
                    v7_value, pose_maps["v7"], source_asset=asset, pose_axis_angle=pose_aa
                )
            else:
                value = hybrid_value if name == "hybrid" else v11_value
                vertices, _ = pose_whole_chain_vertices_v10(
                    value,
                    pose_maps[name],
                    source_asset=asset,
                    pose_axis_angle=pose_aa,
                )
            skin, skin_faces = skins[pose_name]
            candidates[name][pose_name] = absolute_poke_metrics(
                vertices, asset=asset, skin=skin, skin_faces=skin_faces
            )
            entry = candidates[name][pose_name][WHOLE_BODY_GROUP]
            print(
                f"{name:12s} {pose_name:12s} "
                f"max={entry['max_outside_m'] * 1000:6.1f} mm  "
                f"p95={entry['poke_p95_all_m'] * 1000:5.2f} mm  "
                f"outside_area={entry['outside_area_fraction'] * 100:5.2f}%",
                flush=True,
            )

    # Pack A is the linkage baseline every later version claims to improve on,
    # so it is the reference the absolute gate scores against.
    gate = {
        name: compare_absolute_poke_v12(
            candidates[name], reference=candidates["pack_A_142"]
        )
        for name in ("v7", "hybrid", "v11")
    }
    print()
    for name, verdict in gate.items():
        worst = sorted(
            verdict["failures"], key=lambda f: -f["regression_m"]
        )[:3]
        detail = (
            "; ".join(
                f"{f['pose']}/{f['group']} +{f['regression_m'] * 1000:.1f} mm"
                for f in worst
            )
            or "none"
        )
        print(
            f"gate {name:8s} passed={str(verdict['passed']):5s} "
            f"target_met={str(verdict['target_met']):5s}  worst: {detail}"
        )

    report = {
        "schema_version": 12,
        "artifact_kind": "AbsolutePokeTableV12",
        "subject": label,
        "publishable": False,
        "metric": "signed_point_to_mesh_distance_area_weighted",
        "note": (
            "pack_A_142 is the 31133af materialize baseline; hybrid/v11 use the "
            "current joint-anchored FK with identity-142 terminals."
        ),
        "captured_hinge_angles_deg": captured_angles,
        "deep_flex": deep_flex,
        "shadows": {
            "v7": str(args.v7_shadow.expanduser().resolve()),
            "hybrid": str(args.hybrid_shadow.expanduser().resolve()),
            "v11": str(args.v11_shadow.expanduser().resolve()),
        },
        "candidates": candidates,
        "gate_vs_pack_a": gate,
    }
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    table = _markdown_table(candidates, list(poses))
    out.with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
