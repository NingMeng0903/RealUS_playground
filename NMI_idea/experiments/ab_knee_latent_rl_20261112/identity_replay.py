"""Replay audit for confident prefix identity errors.

This is an evaluator-only audit.  It rebuilds a case from the saved test seed,
replays the three saved post-prefix captures (kind, time, site and view), and
lets the public unlabelled filter update itself.  Evaluator labels are read
only from the reconstructed frame after the filter update; they are never
passed to :class:`belief.Filter`.

The report deliberately separates two effects:

* ``old_evidence``: the posterior assigned to the *same* witness selected by
  the erroneous prefix decision, after the rechecks;
* ``fresh_replacement``: a new witness selected for the same identity after
  the rechecks.

The source experiment and atlas are treated as frozen.  This file is a
stand-alone audit and does not alter the experiment or its checkpoints.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from belief import CONFIDENCE
from experiment import prepared, prefix_hash, seed_at
from knee_world import Candidate, World


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"
METHODS = ("rule", "fqi_512_7", "cql_512_7", "sac_512_7")
SPLIT = "test"
TRACE_TOL = 2.0e-10


def _json_rows(results: Path):
    for path in sorted(results.glob("test_*.jsonl")):
        with path.open() as stream:
            for line in stream:
                line = line.strip()
                if line:
                    yield path, json.loads(line)


def _array_equal(a: Any, b: Any, tol: float = TRACE_TOL) -> bool:
    aa = np.asarray(a, dtype=float).reshape(-1, 2)
    bb = np.asarray(b, dtype=float).reshape(-1, 2)
    return aa.shape == bb.shape and bool(np.allclose(aa, bb, atol=tol, rtol=tol))


def _frame_from_saved(world: World, case: Any, item: dict, kind: str):
    """Recreate a saved capture without exposing its evaluator label."""
    site = int(item["site"])
    view = int(item["view"])
    t = float(item["t"])
    action = Candidate(site, view, 0.0, 0.0, t, str(kind))
    frame = world.observe(case, action)
    # World.observe is seeded by the physical clock/site/view, so these are
    # exact checks of the saved capture, not a newly sampled observation.
    if abs(frame.t - t) > TRACE_TOL or abs(frame.q - float(item["q"])) > TRACE_TOL:
        raise AssertionError("saved capture clock/pose does not replay exactly")
    if frame.site != site or frame.view != view:
        raise AssertionError("saved capture site/view does not replay exactly")
    if not _array_equal(frame.centroids, item.get("centroids_mm", [])):
        raise AssertionError("saved capture centroids do not replay exactly")
    return frame


def _truth_class(label: int) -> int:
    # Filter class 2 is the public clutter/unassigned class; world evaluator
    # uses -1 for clutter.
    return 2 if int(label) < 0 else int(label)


def _find_witness(state, key):
    for witness in state.belief.witnesses:
        if witness.key == key:
            return witness
    return None


def _prefix_errors(prefix):
    keys, confidence = prefix.belief.decisions()
    labels = [int(prefix.labels.get(key, -2)) if key is not None else -2
              for key in keys]
    errors = []
    for identity, (key, conf, label) in enumerate(zip(keys, confidence, labels)):
        if key is None or float(conf) < CONFIDENCE or label == identity:
            continue
        witness = _find_witness(prefix, key)
        if witness is None:
            raise AssertionError(f"prefix decision key {key!r} not in witness set")
        initial_prob = prefix.belief.probability(witness)
        errors.append({
            "identity": int(identity),
            "key": [int(key[0]), int(key[1])],
            "assigned_label": int(identity),
            "evaluator_label": int(label),
            "kind": "clutter" if label < 0 else "wrong_branch",
            "initial_confidence": float(conf),
            "initial_posterior": initial_prob.tolist(),
            "initial_map_class": int(np.argmax(initial_prob)),
        })
    return errors


def _audit_run(world: World, row: dict, method: str):
    run = row["runs"][method]
    case = world.case(int(row["seed"]))
    prefix = prepared(world, case)
    if int(row["prefix_acquisitions"]) != len(prefix.frames):
        raise AssertionError("prefix acquisition count changed during replay")
    if prefix_hash(prefix) != run["prefix_sha256"]:
        raise AssertionError("paired prefix hash mismatch")

    prefix_errors = _prefix_errors(prefix)
    if not prefix_errors:
        return None

    # The saved trace consists of the common prefix followed by exactly the
    # three method-specific captures.  The action kind is retained as a
    # provenance field; the public filter receives only the frame contents.
    prefix_len = len(prefix.frames)
    saved_tail = run.get("trace", [])[prefix_len:]
    kinds = list(run.get("actions", []))
    if len(saved_tail) != len(kinds):
        raise AssertionError("saved trace/action count mismatch")

    state = prefix
    capture_checks = []
    for item, kind in zip(saved_tail, kinds):
        frame = _frame_from_saved(world, case, item, kind)
        # State.updated calls the unlabelled public filter.  The frame's
        # labels are only copied to state.labels for evaluator bookkeeping.
        state = state.updated(world, frame, retain=world.identity_site(frame.site))
        capture_checks.append({
            "kind": str(kind), "t": float(item["t"]), "site": int(item["site"]),
            "view": int(item["view"]), "centroids_match": True,
        })

    final_keys, final_conf = state.belief.decisions()
    saved_keys = run.get("witness_keys", [None, None])
    saved_conf = np.asarray(run.get("confidence", [0., 0.]), dtype=float)
    replay_conf = np.asarray(final_conf, dtype=float)
    conf_error = float(np.max(np.abs(replay_conf - saved_conf)))
    final_exact = bool(conf_error <= TRACE_TOL)
    final_key_exact = tuple(final_keys) == tuple(
        tuple(key) if key is not None else None for key in saved_keys
    )

    old_rows = []
    for entry in prefix_errors:
        key = tuple(entry["key"])
        old = _find_witness(state, key)
        if old is None:
            # A witness is retained by Filter.copy/observe.  Missing it would
            # signal an implementation regression, not a successful reset.
            raise AssertionError(f"historical witness {key!r} disappeared")
        final_prob = state.belief.probability(old)
        identity = int(entry["identity"])
        truth_class = _truth_class(entry["evaluator_label"])
        initial_map = int(entry["initial_map_class"])
        final_map = int(np.argmax(final_prob))
        selected_key = final_keys[identity]
        selected_label = int(state.labels.get(selected_key, -2)) if selected_key is not None else -2
        replacement = bool(
            selected_key is not None and tuple(selected_key) != key
            and float(replay_conf[identity]) >= CONFIDENCE
            and selected_label == identity
        )
        old_rows.append({
            **entry,
            "final_posterior": final_prob.tolist(),
            "final_assigned_posterior": float(final_prob[identity]),
            "assigned_posterior_delta": float(final_prob[identity] - entry["initial_posterior"][identity]),
            "disconfirmed_below_0.87": bool(float(final_prob[identity]) < CONFIDENCE),
            # A true MAP flip is deliberately measured on the historical
            # witness, not on a newly selected witness.  This should remain
            # zero: correction should disconfirm stale evidence and let a
            # fresh witness replace it, rather than silently relabelling it.
            "true_map_flip": bool(initial_map != truth_class and final_map == truth_class),
            "final_map_class": final_map,
            "truth_class": truth_class,
            "replacement": replacement,
            "replacement_key": (list(selected_key) if selected_key is not None else None),
            "replacement_evaluator_label": selected_label,
        })

    old_disconfirmed = sum(item["disconfirmed_below_0.87"] for item in old_rows)
    true_map_flips = sum(item["true_map_flip"] for item in old_rows)
    replacements = [item for item in old_rows if item["replacement"]]
    return {
        "index": int(row["index"]),
        "seed": int(row["seed"]),
        "method": method,
        "prefix_error_records": old_rows,
        "capture_replay": capture_checks,
        "saved_final_confidence": saved_conf.tolist(),
        "replayed_final_confidence": replay_conf.tolist(),
        "max_final_confidence_error": conf_error,
        "final_confidence_exact": final_exact,
        "final_keys_exact": final_key_exact,
        "final_keys": [list(key) if key is not None else None for key in final_keys],
        "old_evidence": {
            "records": len(old_rows),
            "disconfirmed_below_0.87": int(old_disconfirmed),
            "true_map_flip": int(true_map_flips),
            "initial_assigned_posterior_mean": float(np.mean([x["initial_posterior"][x["assigned_label"]] for x in old_rows])),
            "final_assigned_posterior_mean": float(np.mean([x["final_assigned_posterior"] for x in old_rows])),
            "wrong_branch_records": int(sum(x["kind"] == "wrong_branch" for x in old_rows)),
            "clutter_records": int(sum(x["kind"] == "clutter" for x in old_rows)),
        },
        "fresh_replacement": {
            "records": len(replacements),
            "correct_witness_replacement": int(len(replacements)),
            "wrong_branch_replacement": int(sum(x["kind"] == "wrong_branch" for x in replacements)),
            "clutter_replacement": int(sum(x["kind"] == "clutter" for x in replacements)),
        },
    }


def audit(results: Path):
    world = World()
    rows = list(_json_rows(results))
    if len(rows) != 256:
        raise AssertionError(f"expected 256 final test cases, found {len(rows)}")
    seen = {int(row["index"]) for _, row in rows}
    if seen != set(range(256)):
        raise AssertionError("final test indices are not exactly 0..255")

    audits = []
    all_method_counts = {}
    for method in METHODS:
        method_rows = []
        for _, row in rows:
            result = _audit_run(world, row, method)
            if result is not None:
                method_rows.append(result)
        all_method_counts[method] = {
            "confident_wrong_prefix_cases": len(method_rows),
            "prefix_error_records": int(sum(len(x["prefix_error_records"]) for x in method_rows)),
            "capture_replays": int(sum(len(x["capture_replay"]) for x in method_rows)),
            "final_confidence_exact_cases": int(sum(x["final_confidence_exact"] for x in method_rows)),
            "final_key_exact_cases": int(sum(x["final_keys_exact"] for x in method_rows)),
            "max_final_confidence_error": float(max((x["max_final_confidence_error"] for x in method_rows), default=0.)),
            "old_evidence": {
                "records": int(sum(x["old_evidence"]["records"] for x in method_rows)),
                "disconfirmed_below_0.87": int(sum(x["old_evidence"]["disconfirmed_below_0.87"] for x in method_rows)),
                "true_map_flip": int(sum(x["old_evidence"]["true_map_flip"] for x in method_rows)),
                "wrong_branch_records": int(sum(x["old_evidence"]["wrong_branch_records"] for x in method_rows)),
                "clutter_records": int(sum(x["old_evidence"]["clutter_records"] for x in method_rows)),
            },
            "fresh_replacement": {
                "records": int(sum(x["fresh_replacement"]["records"] for x in method_rows)),
                "correct_witness_replacement": int(sum(x["fresh_replacement"]["correct_witness_replacement"] for x in method_rows)),
                "wrong_branch_replacement": int(sum(x["fresh_replacement"]["wrong_branch_replacement"] for x in method_rows)),
                "clutter_replacement": int(sum(x["fresh_replacement"]["clutter_replacement"] for x in method_rows)),
            },
        }
        audits.extend(method_rows)
    return {
        "schema": "identity_replay_v1",
        "source": "results/test_*.jsonl",
        "split": SPLIT,
        "cases": len(rows),
        "methods": list(METHODS),
        "confidence_threshold": CONFIDENCE,
        "replay_contract": {
            "prefix": "prepared(World.case(seed))",
            "captures": "saved trace tail, using saved action.kind and frame t/site/view",
            "filter_labels": "unlabelled; evaluator labels are never passed to Filter",
            "true_map_flip": "historical erroneous witness MAP becomes its evaluator truth class",
            "replacement": "new selected witness for the same identity has evaluator label equal to identity",
        },
        "summary": all_method_counts,
        "audits": audits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS / "IDENTITY_REPLAY.json")
    args = parser.parse_args()
    report = audit(args.results)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
