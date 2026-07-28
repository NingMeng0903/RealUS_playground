"""Compare the authored knee hinge axis against the rest condyle geometry.

The knee hinge should sit on the transepicondylar axis.  If the baked axis is far
off it, the leg IK has to spend implausible hip twist to reconcile the authored
hinge with any sagittal drive, which is a bake defect rather than a solve defect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    load_subject_asset,
)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    return value / max(float(np.linalg.norm(value)), 1.0e-12)


def _acute_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = abs(float(np.dot(_unit(a), _unit(b))))
    return float(np.degrees(np.arccos(min(cosine, 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--domains", required=True)
    args = parser.parse_args()

    subject = load_subject_asset(args.subject)
    asset = subject.rigged_asset
    names = list(asset.source_bone_names)
    rest = np.asarray(asset.source_rest_global, dtype=np.float64)
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    domains = json.loads(Path(args.domains).read_text())["domains"]
    metadata = asset.metadata or {}
    leg_solve = metadata.get("source_leg_hinge_solve_v1") or {}
    hinge_splines = metadata.get("source_knee_hinge_splines_v7") or {}

    report: dict[str, object] = {}
    for side, suffix in (("left", "L"), ("right", "R")):
        entry = leg_solve.get(side)
        if entry is None:
            report[side] = "missing leg solve entry"
            continue
        femur = names.index(f"Femur_Rot_{suffix}")
        knee = names.index(f"Knee_Rotate_{suffix}")
        ankle = names.index(f"Tibia_Bone_{suffix}")

        axis_femur_local = _unit(np.asarray(entry["hinge_axis_femur_local"]))
        hinge_world = _unit(rest[femur, :3, :3] @ axis_femur_local)

        lateral = np.asarray(domains[f"{side}/femoral_condyle_lateral"], dtype=np.int64)
        medial = np.asarray(domains[f"{side}/femoral_condyle_medial"], dtype=np.int64)
        epicondylar = np.mean(vertices[lateral], axis=0) - np.mean(
            vertices[medial], axis=0
        )

        femur_dir = _unit(rest[knee, :3, 3] - rest[femur, :3, 3])
        shank_dir = _unit(rest[ankle, :3, 3] - rest[knee, :3, 3])
        sagittal_normal = np.cross(femur_dir, shank_dir)

        side_report: dict[str, object] = {
            "hinge_vs_epicondylar_deg": _acute_deg(hinge_world, epicondylar),
            "hinge_vs_femur_long_axis_deg": _acute_deg(hinge_world, femur_dir),
            "hinge_vs_rest_shank_deg": _acute_deg(hinge_world, shank_dir),
            "rest_sagittal_plane_valid": bool(
                float(np.linalg.norm(sagittal_normal)) > 1.0e-6
            ),
            "hinge_world": hinge_world.tolist(),
            "epicondylar_unit": _unit(epicondylar).tolist(),
            "femur_long_axis": femur_dir.tolist(),
        }

        spline = hinge_splines.get(str(knee))
        if spline is not None:
            spline_axis = _unit(np.asarray(spline["axis_local"], dtype=np.float64))
            spline_world = _unit(rest[knee, :3, :3] @ spline_axis)
            side_report["spline_axis_vs_leg_solve_axis_deg"] = _acute_deg(
                spline_world, hinge_world
            )
            side_report["spline_axis_vs_epicondylar_deg"] = _acute_deg(
                spline_world, epicondylar
            )
        report[side] = side_report

    print(json.dumps(report, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
