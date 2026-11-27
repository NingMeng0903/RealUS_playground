"""Bounded dense-update equivalence/performance audit for N4.

This script compares the dense path with the pre-dense N4 snapshot.  It only
uses public calibration packets and never reads hidden truth for prediction.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from acoustics import PublicObservation
from train_calibrate import calibration_rows
import belief_model as dense


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "snapshots" / "n4_before_dense_update_20261111" / "belief_model.py"
FIELDS = ("particle_weights", "mode_weights", "component_weights",
          "precisions", "informations")
COMPLETE_FIELDS = ("raw_complete_p", "calibrated_p", "risk_score",
                   "p_target_identity", "p_geometry_gate", "p_quality_gate",
                   "p_joint_first_attempt", "p_false_early_stop",
                   "p_public_success_stop", "p_ever_qualified")


def _load_reference():
    # The snapshot's cached Numba signatures were compiled under the short
    # module name ``old`` during the source audit; retain that name so cache
    # loading is deterministic rather than silently recompiling a different
    # module environment.
    spec = importlib.util.spec_from_file_location("old", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["old"] = module
    spec.loader.exec_module(module)
    return module


def _build(module, rows, object_id):
    records = sorted((row for row in rows
                      if row["object_id"] == object_id and row["option_name"] == "left"),
                     key=lambda row: row["prefix_length"])
    belief = module.new_belief()
    states = {0: belief.clone()}
    previous = 0
    for row in records:
        for step, prefix_row in enumerate(row["prefix"][previous:row["prefix_length"]], start=previous):
            action = module.ProbeAction(**prefix_row["action"])
            belief.update(prefix_row["packet"].event_id, action, prefix_row["packet"])
            states[step + 1] = belief.clone()
        previous = row["prefix_length"]
    return records, states


def _packet_like(packet, event_id):
    return type(packet)(event_id, packet.address, packet.window, packet.elapsed_s,
                        packet.exit_state, packet.detected, packet.confidence,
                        packet.quality, packet.center_offset_m,
                        packet.axis_angle_rad, packet.candidate_type)


def _timed_update(module, state, prefix_row, repeats=5):
    values = []
    for index in range(repeats):
        event_id = f"dense-bench-{index}"
        branch = state.clone()
        branch._transition_cache = None
        branch._prediction_cache.clear()
        branch._complete_cache.clear()
        action = module.ProbeAction(**prefix_row["action"])
        packet = _packet_like(prefix_row["packet"], event_id)
        started = time.perf_counter()
        branch.update(event_id, action, packet)
        values.append(time.perf_counter() - started)
    return {"mean_s": float(np.mean(values)), "std_s": float(np.std(values))}


def _timed_acceptance(actions, repeats=5):
    batch_values = []
    scalar_values = []
    for _ in range(repeats):
        branch = dense.new_belief()
        started = time.perf_counter()
        branch.predict_acceptance_batch(actions)
        batch_values.append(time.perf_counter() - started)
        branch = dense.new_belief()
        started = time.perf_counter()
        for action in actions:
            branch.public_predict(action)
        scalar_values.append(time.perf_counter() - started)
    return {
        "batch_mean_s": float(np.mean(batch_values)),
        "batch_std_s": float(np.std(batch_values)),
        "scalar_mean_s": float(np.mean(scalar_values)),
        "scalar_std_s": float(np.std(scalar_values)),
    }


def main():
    reference = _load_reference()
    rows = calibration_rows(721502, 2)
    max_error = {name: 0.0 for name in
                 ("particle_weights", "mode_weights", "component_weights",
                  "posterior_moments", "public_probabilities", "complete_probabilities")}
    for object_id in (0, 1):
        dense_rows, dense_states = _build(dense, rows, object_id)
        reference_rows, reference_states = _build(reference, rows, object_id)
        for prefix in (1, 2, 4, 6):
            current = dense_states[prefix]
            expected = reference_states[prefix]
            for name in ("particle_weights", "mode_weights", "component_weights"):
                max_error[name] = max(max_error[name],
                                      float(np.max(np.abs(getattr(current, name)
                                                          - getattr(expected, name)))))
            current_mean, current_cov = current.posterior_moments()
            expected_mean, expected_cov = expected.posterior_moments()
            max_error["posterior_moments"] = max(
                max_error["posterior_moments"],
                float(max(np.max(np.abs(current_mean - expected_mean)),
                          np.max(np.abs(current_cov - expected_cov)))))
            current_action = dense.ProbeAction(**dense_rows[0]["terminal"])
            expected_action = reference.ProbeAction(**reference_rows[0]["terminal"])
            current_packet = current.public_predict(current_action)
            expected_packet = expected.public_predict(expected_action)
            max_error["public_probabilities"] = max(
                max_error["public_probabilities"],
                float(max(abs(current_packet.p_candidate - expected_packet.p_candidate),
                          abs(current_packet.p_missing - expected_packet.p_missing),
                          abs(current_packet.p_public_accept - expected_packet.p_public_accept),
                          abs(current_packet.p_correct_accept - expected_packet.p_correct_accept),
                          abs(current_packet.p_false_accept - expected_packet.p_false_accept))))
            current_complete = current.predict_complete_option(
                dense.CompleteOption((current_action, current_action), max_retries=2, gh_order=33))
            expected_complete = expected.predict_complete_option(
                reference.CompleteOption((expected_action, expected_action), max_retries=2, gh_order=33))
            max_error["complete_probabilities"] = max(
                max_error["complete_probabilities"],
                float(max(abs(float(getattr(current_complete, field))
                             - float(getattr(expected_complete, field)))
                          for field in COMPLETE_FIELDS)))
    dense_rows, dense_states = _build(dense, rows, 0)
    actions = [dense.ProbeAction(**row["terminal"]) for row in rows[:12]]
    batch = dense.new_belief().predict_acceptance_batch(actions)
    scalar = [dense.new_belief().public_predict(action) for action in actions]
    batch_error = max(max(abs(value.p_correct_accept - prediction.p_correct_accept),
                          abs(value.p_false_accept - prediction.p_false_accept),
                          abs(value.p_public_accept - prediction.p_public_accept))
                      for value, prediction in zip(batch, scalar))
    prefix_row = next(row for row in dense_rows if row["prefix_length"] == 1)["prefix"][0]
    _, reference_states = _build(reference, rows, 0)
    timing = {
        "dense_update_prefix6": _timed_update(dense, dense_states[6], prefix_row),
        "reference_update_prefix6": _timed_update(reference, reference_states[6], prefix_row),
        "acceptance_12": _timed_acceptance(actions),
    }
    payload = {
        "version": "n4-dense-update-v1",
        "source": {"current": hashlib.sha256((ROOT / "belief_model.py").read_bytes()).hexdigest(),
                   "test": hashlib.sha256((ROOT / "test_belief_model.py").read_bytes()).hexdigest(),
                   "reference": hashlib.sha256(REFERENCE.read_bytes()).hexdigest()},
        "seed": 721502,
        "objects": [0, 1],
        "prefixes": [1, 2, 4, 6],
        "max_error": max_error,
        "acceptance_batch": {"actions": 12, "max_abs_error": batch_error},
        "timing": timing,
        "contract": {"particles": 512, "max_components": 4,
                     "reference_fallback": True,
                     "public_mixture_components_in_acceptance_batch": False}
    }
    (ROOT / "DENSE_EQUIVALENCE.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
