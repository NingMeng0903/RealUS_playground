"""Same-endpoint ProDMP-shape ablation for selected learned-B holdout cases.

The selector is frozen: this script only replays cases whose *logged* learned
action was a recovery action (left, right, short, or revisit).  It regenerates
the public prefix, holds the selected action, longitudinal path, and planned
lateral endpoint fixed, replaces the first action's learned null-space shape
by the endpoint-only ProDMP, and then uses the same feedback-enabled live tail.

This is a mechanism ablation on policy-selected cases, not a deployable
all-case controller comparison.  It never chooses a new action from test
outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_RESULT = REPO.parent / "recovery-training-result"
DEFAULT_RECORDS = DEFAULT_RESULT / "test_fixed.jsonl.gz"
DEFAULT_WEIGHTS = DEFAULT_RESULT / "prodmp_weights.npz"
ELIGIBLE_ACTIONS = ("left", "right", "short", "revisit")
COMPARE_METRICS = (
    "success", "wrong", "fresh_local", "first_block_probe_local",
    "first_block_accepted", "first_block_correct_accepted", "first_block_wrong_accepted",
    "coverage", "arrived", "tcp_mm", "turn_deg", "seconds", "frames",
)


def _import_helpers():
    """Import the target run module before controller mutates sys.path."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(1, str(REPO))
    # Importing run first is important: the legacy experiment also has run.py.
    import run as recovery_run  # pylint: disable=import-outside-toplevel
    from controller import (  # pylint: disable=import-outside-toplevel
        BLOCK_STEPS, FRAME_STRIDE, MAX_TRAVEL, Candidate, admissible,
    )
    return recovery_run, BLOCK_STEPS, FRAME_STRIDE, MAX_TRAVEL, Candidate, admissible


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path):
    if path.suffix != ".gz":
        raise ValueError(f"expected gzipped JSONL records: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc


def _assert_public_prefix(row, state, prefix, *, index):
    """Verify that regeneration starts from the same public state and trace."""
    expected_seed = 1600000 + index
    if row.get("seed") != expected_seed:
        raise AssertionError(f"case {index}: seed {row.get('seed')} != frozen {expected_seed}")
    logged = row.get("prefix") or {}
    logged_motion = np.asarray(logged.get("motion_t_s_x_vs_vx_q", []), dtype=float)
    regenerated_motion = np.asarray(prefix.motion, dtype=float)
    np.testing.assert_allclose(logged_motion, regenerated_motion, atol=1.e-10, rtol=0.,
                               err_msg=f"case {index}: public prefix motion changed")
    if len(logged.get("packets", [])) != len(prefix.packets):
        raise AssertionError(f"case {index}: public prefix packet count changed")
    checkpoint = row.get("checkpoint")
    if checkpoint is None:
        raise AssertionError(f"case {index}: missing checkpoint")
    for name, value in vars(state).items():
        expected = checkpoint.get(name)
        if name == "belief":
            np.testing.assert_allclose(np.asarray(expected, dtype=float), value,
                                       atol=1.e-10, rtol=0.,
                                       err_msg=f"case {index}: checkpoint belief changed")
        elif isinstance(value, (float, np.floating)):
            if expected is None or not np.isclose(float(expected), float(value), atol=1.e-10, rtol=0.):
                raise AssertionError(f"case {index}: checkpoint {name} changed")
        elif expected != value:
            raise AssertionError(f"case {index}: checkpoint {name} changed")


def _logged_first_block(row, action, prefix_len, *, index):
    """Extract the selected action's first block from the archived suffix."""
    suffix = (row.get("suffixes") or {}).get("learned") or {}
    trace = np.asarray(suffix.get("motion_t_s_x_vs_vx_q", []), dtype=float)
    offset = prefix_len - 1  # archive() starts suffixes at the final prefix state
    blocks = suffix.get("blocks", [])
    match = next((block for block in blocks if block[0] == action and block[1] >= offset), None)
    if match is None:
        raise AssertionError(f"case {index}: logged learned suffix has no first {action} block")
    lo, hi = int(match[1] - offset), int(match[2] - offset)
    if lo < 0 or hi > len(trace) or hi <= lo:
        raise AssertionError(f"case {index}: invalid archived {action} block range")
    return trace[lo:hi], match


def _first_block(record, action, prefix_len):
    offset = prefix_len - 1
    match = next((block for block in record.blocks if block[0] == action and block[1] >= offset), None)
    if match is None:
        raise AssertionError(f"replay has no first {action} block")
    return np.asarray(record.motion[match[1]:match[2]], dtype=float), match


def _state_after(state):
    return {"frame": int(state.frame), "s": float(state.s), "x": float(state.x),
            "vs": float(state.vs), "vx": float(state.vx), "travel": float(state.travel)}


def _first_accepts(record, before_frame, after_frame):
    entries = [entry for entry in record.accepts
               if before_frame < int(entry["frame"]) <= after_frame]
    return {
        "count": len(entries),
        "correct": int(sum(bool(entry["identity_ok"] and entry["center_ok"]) for entry in entries)),
        "wrong": int(sum(not bool(entry["identity_ok"]) for entry in entries)),
    }


def _rollout(recovery_run, ctrl, scene, before, prefix, candidate, *, zero_shape):
    """Run one opening action, record its handoff, then use the common live tail."""
    state, record = recovery_run.fork(before, prefix)
    if zero_shape:
        original_shape = ctrl.shape
        # This applies to the opening plan's visual reconditioning as well as
        # the initial plan.  The endpoint and longitudinal component remain in
        # the Candidate passed below.
        ctrl.shape = lambda *args: ctrl.zero.copy()
        try:
            ctrl.execute(scene, state, record, candidate, False)
        finally:
            ctrl.shape = original_shape
    else:
        ctrl.execute(scene, state, record, candidate, False)
    first_block, block = _first_block(record, candidate.name, len(prefix.motion))
    after = _state_after(state)
    evidence = recovery_run.acquisition(ctrl, scene, before, state, record, False)
    first_probe = bool(evidence["probe_local"])
    first_accepts = _first_accepts(record, before.frame, state.frame)
    ctrl.finish(scene, state, record, False, allow_recover=True)
    result = recovery_run.measures(ctrl, record, state) | evidence
    result["fresh_local"] = recovery_run.fresh_evidence(
        record, evidence["query_bin"], before.frame, state.frame)
    probability = ctrl.world.identity_prob(state.belief, recovery_run.IDENTITY_STATION,
                                           before.angle, alias=False)
    result["terminal_identity_correct"] = bool(
        (probability >= .5) == evidence["identity_truth"]
        and max(probability, 1. - probability) >= recovery_run.IDENTITY_ACCEPT)
    result["first_block_probe_local"] = first_probe
    result["first_block_accepted"] = first_accepts["count"]
    result["first_block_correct_accepted"] = first_accepts["correct"]
    result["first_block_wrong_accepted"] = first_accepts["wrong"]
    return result, record, first_block, block, after


def _zero_candidate(ctrl, state, candidate, BLOCK_STEPS, MAX_TRAVEL, Candidate, admissible):
    """Create endpoint-only ProDMP plan with the original s/x endpoint."""
    plan = ctrl.primitive.condition(0, BLOCK_STEPS, state.x, state.vx,
                                   candidate.lateral.goal, 0., ctrl.zero)
    is_admissible = bool(admissible(plan))
    travel, turn = ctrl.geometry.cost(candidate.s, plan.x)
    hard_travel_ok = bool(state.travel + travel <= MAX_TRAVEL + 1.e-9)
    if not is_admissible or not hard_travel_ok:
        return None, {
            "admissible": is_admissible,
            "hard_travel_ok": hard_travel_ok,
            "travel_mm": float(travel),
            "turn_deg": float(turn),
            "planned_endpoint_s": float(candidate.s[-1]),
            "planned_endpoint_x_original_mm": float(candidate.lateral.x[-1]),
            "planned_endpoint_x_zero_mm": float(plan.x[-1]),
            "failure": "inadmissible" if not is_admissible else "hard_travel",
        }
    replacement = Candidate(candidate.name, candidate.s.copy(), candidate.vs.copy(), plan,
                             float(travel), float(turn), candidate.goal_dx)
    return replacement, {
        "admissible": is_admissible,
        "hard_travel_ok": hard_travel_ok,
        "travel_mm": float(travel),
        "turn_deg": float(turn),
        "planned_endpoint_s": float(candidate.s[-1]),
        "planned_endpoint_x_original_mm": float(candidate.lateral.x[-1]),
        "planned_endpoint_x_zero_mm": float(plan.x[-1]),
        "failure": None,
    }


def _small_result(result):
    out = {key: result.get(key) for key in COMPARE_METRICS}
    out["mean_abs_lateral_mm"] = result.get("mean_abs_lateral_mm")
    out["terminal_identity_correct"] = result.get("terminal_identity_correct")
    return out


def _paired(rows, key):
    if not rows:
        return {"n": 0, "delta": None, "ci95": [None, None], "wins": 0, "losses": 0}
    diff = np.asarray([float(row["ablation"][key]) - float(row["baseline"][key]) for row in rows])
    rng = np.random.default_rng(84122)
    bootstrap = rng.integers(len(diff), size=(4000, len(diff)))
    ci = np.quantile(diff[bootstrap].mean(axis=1), [.025, .975]).tolist()
    return {"n": len(diff), "delta_ablation_minus_baseline": float(diff.mean()), "ci95": ci,
            "wins": int(np.sum(diff > 0.)), "losses": int(np.sum(diff < 0.)),
            "ties": int(np.sum(diff == 0.))}


def run(records, weights, out, *, limit=None):
    recovery_run, BLOCK_STEPS, FRAME_STRIDE, MAX_TRAVEL, Candidate, admissible = _import_helpers()
    records, weights, out = Path(records).resolve(), Path(weights).resolve(), Path(out).resolve()
    sections = recovery_run.load_sections()
    with np.load(weights, allow_pickle=False) as archive:
        coefficients = {name: archive[name] for name in archive.files}
    world = recovery_run.World()
    ctrl = recovery_run.Controller(world, sections, coefficients)
    selected = []
    skipped = Counter()
    all_rows = list(_records(records))
    rows_to_process = all_rows if limit is None else all_rows[:limit]
    for index, row in enumerate(rows_to_process):
        action = (row.get("actions") or {}).get("learned")
        if action not in ELIGIBLE_ACTIONS:
            skipped[action or "missing"] += 1
            continue
        if row.get("angle_mode") != "fixed":
            raise ValueError(f"case {index}: same-endpoint ablation expects fixed holdout")
        scene = recovery_run.case(world, "test", index, "fixed")
        before, prefix = recovery_run.context(ctrl, scene, alias=False)
        _assert_public_prefix(row, before, prefix, index=index)
        choices = ctrl.options(before, False)
        if action not in choices:
            raise AssertionError(f"case {index}: logged learned action {action} absent from regenerated menu")
        candidate = choices[action]
        logged_first, logged_block = _logged_first_block(row, action, len(prefix.motion), index=index)
        baseline, baseline_record, baseline_first, baseline_block, baseline_after = _rollout(
            recovery_run, ctrl, scene, before, prefix, candidate, zero_shape=False)
        np.testing.assert_allclose(baseline_first, logged_first, atol=1.e-9, rtol=0.,
                                   err_msg=f"case {index}: regenerated learned shape/feedback path changed")
        if tuple(baseline_block) != tuple(logged_block):
            raise AssertionError(f"case {index}: regenerated action block metadata changed")
        open_count = min(FRAME_STRIDE + 1, len(candidate.lateral.x), len(baseline_first))
        np.testing.assert_allclose(baseline_first[:open_count, 1], candidate.s[:open_count],
                                   atol=1.e-9, rtol=0., err_msg=f"case {index}: s path changed")
        np.testing.assert_allclose(baseline_first[:open_count, 2], candidate.lateral.x[:open_count],
                                   atol=1.e-9, rtol=0., err_msg=f"case {index}: opening ProDMP shape changed")
        logged_outcome = row.get("learned") or {}
        for key in ("success", "wrong", "fresh_local", "tcp_mm", "turn_deg", "seconds", "frames"):
            if key not in logged_outcome:
                continue
            expected, actual = logged_outcome[key], baseline[key]
            if isinstance(expected, bool):
                if bool(expected) != bool(actual):
                    raise AssertionError(f"case {index}: replayed {key} disagrees with logged learned outcome")
            elif not np.isclose(float(expected), float(actual), atol=1.e-8, rtol=0.):
                raise AssertionError(f"case {index}: replayed {key} disagrees with logged learned outcome")
        zero_candidate, feasibility = _zero_candidate(
            ctrl, before, candidate, BLOCK_STEPS, MAX_TRAVEL, Candidate, admissible)
        if zero_candidate is None:
            skipped["ablation_" + feasibility["failure"]] += 1
            selected.append({"record_index": index, "seed": row["seed"], "action": action,
                             "status": "failed_feasibility", "feasibility": feasibility,
                             "baseline": _small_result(baseline)})
            continue
        ablation, ab_record, ab_first, ab_block, ab_after = _rollout(
            recovery_run, ctrl, scene, before, prefix, zero_candidate, zero_shape=True)
        if ab_block != baseline_block:
            raise AssertionError(f"case {index}: ablation block metadata changed")
        endpoint = {
            "planned_s_mm": float(zero_candidate.s[-1]),
            "planned_s_delta_mm": float(zero_candidate.s[-1] - candidate.s[-1]),
            "planned_x_original_mm": float(candidate.lateral.x[-1]),
            "planned_x_zero_mm": float(zero_candidate.lateral.x[-1]),
            "planned_x_delta_mm": float(zero_candidate.lateral.x[-1] - candidate.lateral.x[-1]),
            "actual_s_baseline_mm": baseline_after["s"], "actual_s_ablation_mm": ab_after["s"],
            "actual_s_delta_mm": ab_after["s"] - baseline_after["s"],
            "actual_x_baseline_mm": baseline_after["x"], "actual_x_ablation_mm": ab_after["x"],
            "actual_x_delta_mm": ab_after["x"] - baseline_after["x"],
        }
        open_count = min(FRAME_STRIDE + 1, len(ab_first), len(baseline_first))
        whole_count = min(len(ab_first), len(baseline_first))
        differences = {
            "opening_max_abs_s_mm": float(np.max(np.abs(ab_first[:open_count, 1] - baseline_first[:open_count, 1]))),
            "opening_max_abs_x_mm": float(np.max(np.abs(ab_first[:open_count, 2] - baseline_first[:open_count, 2]))),
            "first_block_max_abs_s_mm": float(np.max(np.abs(ab_first[:whole_count, 1] - baseline_first[:whole_count, 1]))),
            "first_block_max_abs_x_mm": float(np.max(np.abs(ab_first[:whole_count, 2] - baseline_first[:whole_count, 2]))),
            "first_block_max_abs_vx_mm_s": float(np.max(np.abs(ab_first[:whole_count, 4] - baseline_first[:whole_count, 4]))),
        }
        selected.append({
            "record_index": index, "seed": row["seed"], "family": row.get("family"),
            "action": action, "status": "evaluated", "logged_block": logged_block,
            "feasibility": feasibility, "endpoint": endpoint, "trajectory_difference": differences,
            "baseline": _small_result(baseline), "ablation": _small_result(ablation),
        })
    evaluated = [row for row in selected if row["status"] == "evaluated"]
    paired = {key: _paired(evaluated, key) for key in COMPARE_METRICS}
    paired["mean_abs_lateral_mm"] = _paired(evaluated, "mean_abs_lateral_mm")
    trajectory = [row["trajectory_difference"] for row in evaluated]
    source_files = [records, weights, HERE / "controller.py", HERE / "run.py", HERE / "world.py"]
    data = {
        "schema": "ab_recovery_same_endpoint_skill_ablation_v1",
        "scope": ("Selected learned-B recovery cases only: the logged policy selected the opening action; "
                  "the ablation replaces its ProDMP null-space shape by endpoint-only zero, including "
                  "opening-action visual reconditioning, and then hands off to the same live tail. "
                  "This is not a deployable all-case contrast and does not reselect actions or tune on test outcomes."),
        "selection": {"records": str(records), "rule": "logged actions.learned in left/right/short/revisit",
                      "eligible_actions": list(ELIGIBLE_ACTIONS), "rows_read": len(rows_to_process),
                      "selected": len(selected), "evaluated": len(evaluated), "skipped": dict(skipped)},
        "same_endpoint_definition": {
            "opening_action_fixed": True, "longitudinal_s_and_vs_fixed": True,
            "planned_lateral_endpoint_fixed": True,
            "zero_shape": "ctrl.primitive.condition(..., candidate.lateral.goal, ..., ctrl.zero)",
            "reconditioning_shape": "zero for every ctrl.execute reconditioning call in opening block",
            "live_tail": "same Controller.finish(... allow_recover=True) after opening block",
        },
        "paired_public_comparison": paired,
        "trajectory_difference_summary": {
            key: {"mean": float(np.mean([row[key] for row in trajectory])) if trajectory else None,
                  "max": float(np.max([row[key] for row in trajectory])) if trajectory else None,
                  "min": float(np.min([row[key] for row in trajectory])) if trajectory else None}
            for key in ("opening_max_abs_s_mm", "opening_max_abs_x_mm", "first_block_max_abs_s_mm",
                        "first_block_max_abs_x_mm", "first_block_max_abs_vx_mm_s")
        },
        "source_sha256": {path.name: _sha256(path) for path in source_files},
        "cases": selected,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="integration-only prefix of holdout rows; default is all rows")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    out = args.out or args.records.with_name("same_endpoint_ablation.json")
    report = run(args.records, args.weights, out, limit=args.limit)
    print("selected={selected} evaluated={evaluated} skipped={skipped}".format(**report["selection"]))
    print("summary=" + str(out.resolve()))


if __name__ == "__main__":
    main()
