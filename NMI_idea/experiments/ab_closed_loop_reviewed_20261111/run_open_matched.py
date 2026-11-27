"""Matched-deadline open-baffle control using the same reviewed runner."""

import argparse
import ast
import json
from pathlib import Path

from acoustics import paired_scenarios
from fixture import MotionRollouts, NativeMotionEnvironment, make_rig
from planning import PublicModel
from run_closed_loop import (
    HERE, FREEZE_SOURCES, NOISE_SEED, TEST_COUNT, TEST_SEED, create_retry_catalog,
    dump, paired_rows, sha,
)


def constrained_public_timing(root, index):
    frozen = root / f"fixture_{index:02}_constrained" / "FROZEN.json"
    if frozen.exists():
        return json.loads(frozen.read_text())["planning"]
    # Public timings are printed before building the retry catalog and before
    # the first hidden draw. Only this public line is read.
    line = (root / f"constrained_{index}.log").read_text().splitlines()[0]
    return ast.literal_eval(line.split(" public timing ", 1)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--constrained-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public = constrained_public_timing(args.constrained_run, args.index)
    deadline = public["public_timing_witness"]["deadline_s"]
    if deadline is None:
        raise ValueError("no frozen constrained witness deadline")
    with NativeMotionEnvironment() as native:
        rig = make_rig(args.index, "open", native)
        destination = args.output / rig.name
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / "RESULT.json").exists():
            raise FileExistsError(f"result exists in {destination}")
        model = PublicModel(rig, MotionRollouts(rig, native, destination / "calibration"))
        model.calibrate()
        create_retry_catalog(model, deadline)
        frozen = {
            "fixture": rig.public_metadata(), "planning": model.public_report(),
            "matched_constrained_deadline_s": deadline,
            "matched_constrained_timing": public["public_timing_witness"],
            "source_sha256": {p: sha(HERE / p) for p in (*FREEZE_SOURCES, "run_open_matched.py")},
            "motion_fingerprint": native.replay._state_fingerprint(),
            "anatomy_seed": TEST_SEED, "observation_seed": NOISE_SEED, "planned_anatomy_count": TEST_COUNT,
        }
        dump(destination / "FROZEN.json", frozen)
        print(rig.name, "deadline", deadline, "open cross-window times",
              model.witness["cross_window_time"], flush=True)
        physical = MotionRollouts(rig, native, destination / "execution", model=model.rollouts)
        cases = paired_scenarios(TEST_SEED, TEST_COUNT)
        rows = paired_rows(model, physical, cases, {"witness": (deadline, True)})
        for row in rows:
            row["variant"] = "open_witness"
        physical.finish()
        dump(destination / "RESULT.json", {**frozen, "records": rows,
                                          "unique_native_execution_routes": len(physical.rows),
                                          "distinct_executed_capture_states": len(physical.cache),
                                          "identical_state_cache_hits": physical.cache_hits}, compact=True)
        print(rig.name, "matched open control done", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()

