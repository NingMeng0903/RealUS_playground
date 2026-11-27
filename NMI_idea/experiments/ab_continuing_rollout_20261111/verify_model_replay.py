"""Recompute all frozen calibration predictions after performance-only edits."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from belief_model import ModelParams, CompleteOption
from train_calibrate import CAL_SEED, CAL_COUNT, calibration_rows, calibrate_option


def main():
    here = Path(__file__).resolve().parent
    model_dir = here / "models" / "v3"
    model_path = model_dir / "MODEL.json"
    model_bytes = model_path.read_bytes()
    model = ModelParams.from_dict(json.loads(model_bytes))
    old = [json.loads(line) for line in (model_dir / "CALIBRATION_RECORDS.jsonl").read_text().splitlines()]
    start = time.perf_counter()
    _, report = calibrate_option(
        model, calibration_rows(CAL_SEED, CAL_COUNT),
        option_factory=lambda action: CompleteOption((action, action), max_retries=2, gh_order=33),
    )
    new = report["records"]
    key = lambda row: (row["object_id"], row["prefix_length"], row["option_name"])
    old_map = {key(row): row for row in old}
    assert len(old_map) == len(new) == 1280
    errors = []
    for row in new:
        before = old_map[key(row)]
        assert row["input_sha256"] == before["input_sha256"]
        assert row["label"] == before["label"]
        assert row["subject_weight"] == before["subject_weight"]
        errors.append(abs(row["raw_p"] - before["raw_p"]))
    assert model_path.read_bytes() == model_bytes
    result = {
        "scope": "same 128 calibration subjects; numerical equivalence, not independent efficacy data",
        "records": len(new), "subjects": CAL_COUNT,
        "max_raw_probability_abs_difference": float(max(errors)),
        "mean_raw_probability_abs_difference": float(np.mean(errors)),
        "same_inputs_labels_weights": True,
        "passed_at_1e_8": bool(max(errors) <= 1e-8),
        "elapsed_s": time.perf_counter() - start,
        "original_model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "original_belief_source_sha256": "6d71e17fcb8f027da4d9a8482ce3f778c6446eb579742a59ac1b39c6a56ff3ab",
        "current_belief_source_sha256": hashlib.sha256((here / "belief_model.py").read_bytes()).hexdigest(),
        "model_artifact_modified": False,
    }
    path = model_dir / "PERFORMANCE_CALIBRATION_REPLAY.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed_at_1e_8"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
