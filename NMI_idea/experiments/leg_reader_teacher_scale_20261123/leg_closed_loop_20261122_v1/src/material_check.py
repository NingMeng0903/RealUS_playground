"""Small CPU-only material-chart smoke check on the training CT pack.

This check deliberately loads only ``ct_geometry.npz``.  Blender holdout data,
labels, and evaluation traces are not opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import material_motion
import motion


def _run(scanner, mat, spec, controls, seed, profile, *, feedback, use_dird=True):
    trial = motion.Trial(0, int(seed), profile)
    node = scanner.start(spec, trial, scanner.model())
    plan = mat.plan_probe(node, spec, 1, controls, 32, 85.0, use_dird=use_dird)
    result = mat.run_probe(node, spec, trial, plan, 32, 85.0, feedback=feedback,
                           use_dird=use_dird)
    return plan, result


def check(pack: Path) -> dict:
    scanner = motion.Scanner.training(pack, shape=0)
    spec = motion.definition(0, *scanner.ct_bounds)
    material = material_motion.MaterialMotion(scanner)
    trial = motion.Trial(0, 701, "ordinary")
    node = scanner.start(spec, trial, scanner.model())
    features = material.features(node, spec)

    base_plan, base = _run(scanner, material, spec, (.55, .30, .45), 703, "ordinary", feedback=False)
    fb_plan, feedback = _run(scanner, material, spec, (.55, .30, .45), 703, "ordinary", feedback=True)
    shape_plan, shape = _run(scanner, material, spec, (.55, .30, -.45), 703, "ordinary", feedback=False)
    no_dird_plan, no_dird = _run(scanner, material, spec, (.55, .30, .45), 703, "ordinary",
                                 feedback=False, use_dird=False)
    revisit_plan = material.plan_probe(node, spec, 2, (.55, .70, .45), 32, 85.0)
    seen = []

    def stop(anchor_node, info):
        seen.append({"station": float(anchor_node.station), "sector": float(anchor_node.sector), **info})
        return True

    stopped = None
    if revisit_plan is not None:
        revisit_trial = motion.Trial(0, 709, "stress")
        revisit_node = scanner.start(spec, revisit_trial, scanner.model())
        stopped = material.run_probe(revisit_node, spec, revisit_trial, revisit_plan, 32, 85.0,
                                     feedback=False, stop=stop)

    dird_chart_probe = None
    if material.dird.metadata["available"]:
        # These are public chart coordinates in the supported knee shaft
        # patch.  They test the virtual SE(3) field bridge independently of
        # the CT route's proximal start station.
        chart_path = np.asarray([[68.0, 4.0], [70.0, 8.0], [72.0, 12.0],
                                 [74.0, 16.0], [76.0, 20.0]], dtype=float)
        dird_chart_probe = material.dird.score_trajectory(chart_path, material.atlas)

    trajectory_changed = bool(base.legal and feedback.legal and
                              base.trajectory.shape == feedback.trajectory.shape and
                              not np.array_equal(base.trajectory, feedback.trajectory))
    parameter_changed = bool(base.legal and shape.legal and
                             not np.array_equal(base.trajectory, shape.trajectory))
    stop_anchor = bool(stopped is None or (stopped.legal and stopped.stopped and
                                           stopped.node.station == node.station and
                                           stopped.node.sector == node.sector))
    return {
        "status": "passed" if all((base.legal, feedback.legal, parameter_changed,
                                    trajectory_changed, stop_anchor)) else "failed",
        "pack": str(pack),
        "features": {"names": list(material_motion.MATERIAL_FEATURE_NAMES),
                     "shape": list(features.shape), "finite": bool(np.isfinite(features).all())},
        "dird": material.dird.metadata,
        "atlas": material.atlas.metadata,
        "base": {"legal": base.legal, "reason": base.reason, "ticks": base.ticks,
                 "distance_mm": base.distance, "feedback_updates": base.feedback_updates,
                 "plan": None if base_plan is None else base_plan.applied_fraction,
                 "dird_report": None if base.dird_report is None else base.dird_report},
        "feedback": {"legal": feedback.legal, "reason": feedback.reason,
                     "ticks": feedback.ticks, "distance_mm": feedback.distance,
                     "feedback_updates": feedback.feedback_updates,
                     "dird_report": None if feedback.dird_report is None else feedback.dird_report},
        "dird_ablation": {
            "with_dird_status": None if base.dird_report is None else base.dird_report.get("status"),
            "without_dird_status": None if no_dird.dird_report is None else no_dird.dird_report.get("status"),
            "with_dird_legal": bool(base.legal),
            "without_dird_legal": bool(no_dird.legal),
            "virtual_chart_probe": dird_chart_probe,
        },
        "parameter_effect": parameter_changed,
        "feedback_trajectory_effect": trajectory_changed,
        "stop": {"tested": stopped is not None, "anchor_recovered": stop_anchor,
                  "opportunities": len(seen), "first": seen[0] if seen else None},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = check(args.pack)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
