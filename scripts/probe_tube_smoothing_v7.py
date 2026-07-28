"""Does smoothing tube patch weights reduce posed centerline kinks?

``vessel.centerline`` fails because posed vessels turn far more sharply than at
rest.  The suspected cause is that authored skin weights are near-binary, so
adjacent patches hand off from one bone to the next in a single step and the tube
bends at the hand-off.  This re-bakes the tube coefficients in memory at several
smoothing strengths so the effect is measured before paying for a template
rebake.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
from projects.genesis_ue_sync.anatomy_retarget.tube_frames_v7 import (
    bake_tube_material_frames_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    load_subject_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.vessel_gates_v7 import (
    VesselGateThresholdsV7,
    _evaluate_centerline,
)

_TUBE_TISSUES = {"vessel", "nerve"}


def _load_pose(spec: str) -> np.ndarray:
    if spec == "zero":
        return np.zeros((55, 3), dtype=np.float64)
    return np.asarray(
        np.load(spec)["pose_axis_angle"], dtype=np.float64
    ).reshape(55, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--iterations", default="0,3,8,20")
    parser.add_argument(
        "--detail",
        default="",
        help="comma separated mesh names to report branch-level detail for",
    )
    args = parser.parse_args()
    detail_names = [
        name.strip() for name in args.detail.split(",") if name.strip()
    ]

    subject = load_subject_asset(args.subject)
    asset = subject.rigged_asset
    pose = _load_pose(args.pose)
    zero = np.zeros((55, 3), dtype=np.float64)

    ids_by_mesh = {}
    for name, tissue, (start, stop) in zip(
        asset.source_mesh_names,
        asset.source_tissues,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64),
    ):
        if str(tissue).strip().lower() in _TUBE_TISSUES:
            ids_by_mesh[str(name)] = np.arange(int(start), int(stop))

    thresholds = VesselGateThresholdsV7()
    faces = np.asarray(asset.faces, dtype=np.int64)
    results = {}
    for token in args.iterations.split(","):
        iterations = int(token)
        coefficients, _report = bake_tube_material_frames_v7(
            asset, weight_smoothing_iterations=iterations
        )
        rest = skin_vertices(
            asset, zero, transl=None, runtime_coefficients=coefficients
        ).astype(np.float64)
        posed = skin_vertices(
            asset, pose, transl=None, runtime_coefficients=coefficients
        ).astype(np.float64)
        summary, components = _evaluate_centerline(
            posed_vertices=posed,
            reference_vertices=rest,
            vertex_ids_by_mesh=ids_by_mesh,
            faces=faces,
            thresholds=thresholds,
        )
        results[iterations] = {
            "pass": bool(summary.get("pass")),
            "failed_components": summary.get("failed_components"),
            "worst_component": summary.get("worst_component"),
            "worst_max_turn_increase_deg": summary.get(
                "worst_max_turn_increase_deg"
            ),
            "spinal_cord_deg": float(
                (components.get("Spinal_Cord") or {}).get(
                    "max_turn_increase_deg", float("nan")
                )
            ),
        }
        for name in detail_names:
            report = components.get(name)
            if report is None:
                continue
            branches = report.get("branches") or {}
            failing = {
                branch_name: {
                    key: branch[key]
                    for key in (
                        "vertex_count",
                        "bin_count",
                        "reference_max_turn_deg",
                        "posed_max_turn_deg",
                        "max_turn_increase_deg",
                        "q99_turn_increase_deg",
                        "worst_posed_turn_position",
                    )
                    if key in branch
                }
                for branch_name, branch in branches.items()
                if not branch.get("skipped", False)
                and not branch.get("pass", True)
            }
            results[iterations][f"detail::{name}"] = {
                "branch_count": len(branches),
                "worst_branch": report.get("worst_branch"),
                "failing_branches": failing,
            }
    print(json.dumps(results, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
