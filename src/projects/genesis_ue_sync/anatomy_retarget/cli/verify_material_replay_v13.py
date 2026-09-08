"""Save exact source arrays and independently replay a completed V13 matrix.

This creates a new artifact. Original non-exact replay reports are retained.
Geometric failure flags are not overridden by successful serialization.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .run_material_matrix_v13 import DEFAULT_OPERATOR, load_source_operator, materialize_subject
from ..material_runtime_v13 import load_material_runtime_v13


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    args = parser.parse_args()
    matrix = json.loads((args.matrix / "matrix_manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    operator = load_source_operator(args.operator, mmap=True)
    if operator.runtime_digest(validate=False) != matrix["provenance"]["operator_runtime_digest"]:
        raise ValueError("source operator differs from matrix provenance")
    report = {"publishable": False, "geometry_revalidated": False,
              "matrix": str(args.matrix.resolve()), "subjects": {}}
    for label, subject in matrix["subjects"].items():
        if "runtime_root" not in subject:
            report["subjects"][label] = {"status": "no_runtime", "passed": False}
            continue
        original = load_material_runtime_v13(Path(subject["runtime_root"]))
        asset = materialize_subject(operator, betas=np.asarray(subject["betas"]), gender="male").rigged_asset
        runtime = replace(original, source_asset=asset)
        target = args.output / label
        runtime.save(target, provenance={"replaces_legacy_quantized_runtime": subject["runtime_root"],
                                        "geometric_acceptance": False, "betas": subject["betas"]})
        reloaded = load_material_runtime_v13(target)
        cells = {}
        for pose_name in ("heldout_sitting", "heldout_kicking"):
            comparison = Path(subject["runtime_root"]).parent / "comparisons" / f"{pose_name}.npz"
            with np.load(comparison, allow_pickle=False) as data:
                pose = data["pose"].copy()
                reference = {"weights": data["weights_vertices"].copy(),
                             "attachments": data["candidate_vertices"].copy()}
            for mode in ("weights", "attachments"):
                before = runtime.apply_pose(pose, mode=mode)
                after = reloaded.apply_pose(pose, mode=mode)
                cells[f"{pose_name}:{mode}"] = {
                    "exact_save_reload": bool(np.array_equal(before, after)),
                    "reload_max_abs_m": float(np.max(np.abs(before - after))),
                    "matches_original_matrix_geometry": bool(np.array_equal(before, reference[mode])),
                    "matrix_max_abs_m": float(np.max(np.abs(before - reference[mode]))),
                }
        report["subjects"][label] = {"cells": cells, "passed": all(
            cell["exact_save_reload"] and cell["matches_original_matrix_geometry"] for cell in cells.values())}
        print(label, report["subjects"][label], flush=True)
    report["passed"] = all(row["passed"] for row in report["subjects"].values())
    (args.output / "replay_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
