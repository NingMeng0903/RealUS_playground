"""Deterministic per-case audit of the v4 whole-route action tape.

This is deliberately a read-only replay of the frozen v4 reader and action
engine.  It does not train or modify the v4 study.  The only outputs are the
case-level forensic records and an aggregate manifest under ``run/forensics``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "leg_scan_reader_feedback_20261123_v4"
OUT = HERE / "run" / "forensics"
V4_RUN = V4 / "run"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    temporary.replace(path)


def _load_json(path):
    with path.open() as stream:
        return json.load(stream)


def _case_from_meta(meta, Case):
    # The archived action tape stores the authoritative case dictionary.  The
    # v4 Case dataclass grew a default challenge field, so older metadata is
    # accepted without inventing any seed or layout value.
    return Case(**meta["case"])


def _readers(freeze, export_cpu, load_public):
    model = export_cpu(load_public(freeze["updated"]["weights"]))
    readers = {}
    for route in ("femoral", "groin_to_distal_calf"):
        rule = freeze["thresholds_and_reader"][route]
        readers[route] = {}
        risk = freeze["risk"][route].get("fixed_graph")
        threshold = rule["variants"]["fixed_graph"]["threshold"]
        for arm in ("selected", "graph"):
            readers[route][arm] = {
                "method": "fixed_graph",
                "threshold": threshold,
                "risk": risk,
                "position_threshold": freeze.get("position_threshold", .75),
                "model": model,
            }
    return readers


def _float_equal(a, b, *, atol=2.e-5, rtol=2.e-5):
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol))


def _suffix_sequence(node, start_frame):
    """Actual continuation observations after the review has rejoined."""
    return [[float(frame[0]), float(frame[1]), int(frame[8])]
            for frame in node.reader.frames[start_frame:]]


def _spatial_sequence(sequence):
    return [[float(station), float(sector)] for station, sector, _tick in sequence]


def _sequence_hash(sequence):
    payload = json.dumps(sequence, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _echo_hash(frames):
    """Hash measured centroids and quality, before the reader's posteriors."""
    fingerprint = hashlib.sha256()
    for frame in frames:
        points = np.asarray(frame[2], dtype="<f8")
        fingerprint.update(np.asarray([len(points)], dtype="<i4").tobytes())
        fingerprint.update(points.tobytes())
        fingerprint.update(np.asarray([frame[4]], dtype="<f8").tobytes())
    return fingerprint.hexdigest()


def _fixed_suffix(q, rejoined, poses, length_mm, tape, cut, region, score_branch):
    """Replay the MPC pose path with a prescribed acquisition clock and history."""
    reference_tick = q.node.tick
    last_tick = int(poses[-1][-1] - reference_tick + rejoined.tick)
    if last_tick > q.limits.ticks or rejoined.distance + length_mm > q.limits.mm + 1.e-6:
        return None
    node = rejoined.fork()
    before = len(node.reader.frames)
    for station, sector, tick in poses:
        q.env.scanner._observe(node, q.spec, tape, station, sector, tick - reference_tick + rejoined.tick)
    node.station, node.sector, node.tick = q.spec[2], int(round(poses[-1][1])) % 24, last_tick
    node.distance += length_mm
    _, detail, old = score_branch(q, node, cut, region, complete_scan=True)
    return {"strict": bool(np.all(detail[:, 0] == 0)), "old_terminal": int(old),
            "region_labels": detail[:, 0].astype(int).tolist(), "region_capture": detail[:, 1].astype(bool).tolist(),
            "suffix_echo_sha256": _echo_hash(node.reader.frames[before:]), "terminal_tick": last_tick}


def _sequence_echo(left, right):
    first_difference = None
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            first_difference = index
            break
    if first_difference is None and len(left) != len(right):
        first_difference = min(len(left), len(right))
    return {
        "same_length": bool(len(left) == len(right)),
        "same_sequence": bool(first_difference is None),
        "first_difference_index": first_difference,
        "left_frames": int(len(left)),
        "right_frames": int(len(right)),
        "left_sha256": _sequence_hash(left),
        "right_sha256": _sequence_hash(right),
    }


def _assert_equal(expected, actual, path):
    """Assert archive replay agreement while tolerating float32 serialization."""
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"{path}: keys differ {set(expected)} != {set(actual)}")
        for key in expected:
            _assert_equal(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: length {len(expected)} != {len(actual)}")
        for index, (x, y) in enumerate(zip(expected, actual, strict=True)):
            _assert_equal(x, y, f"{path}[{index}]")
        return
    if isinstance(expected, (int, bool)) and isinstance(actual, (int, bool)):
        if expected != actual:
            raise AssertionError(f"{path}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        if not _float_equal(expected, actual):
            raise AssertionError(f"{path}: {expected!r} != {actual!r}")
        return
    if expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def _candidate_stage(q, node, cut, region, live_state, truth_at, confidence):
    """Public candidate IDs and scores; no hidden labels are returned here."""
    public, state = live_state(q.env, q.spec, node, q.reader)
    score = confidence(public, state, q.reader["risk"], max_frame=cut)
    p = state["probabilities"][:, 0]
    use = np.flatnonzero((public["frames"] < cut) & (public["regions"] == region))
    if len(use):
        winner = int(use[np.argmax(score[use])])
    else:
        winner = -1
    winner_score = None if winner < 0 else float(score[winner])
    winner_id = None if winner < 0 else [int(public["frames"][winner]), int(public["slots"][winner])]
    accepted = bool(winner >= 0 and score[winner] >= q.reader["threshold"])
    submitted_id = winner_id if accepted else None

    all_candidates = []
    for index in use:
        all_candidates.append({
            "id": [int(public["frames"][index]), int(public["slots"][index])],
            "station": float(public["stations"][index]),
            "sector": float(public["sectors"][index]),
            "target_probability": float(p[index]),
            "confidence": float(score[index]),
            "accepted_if_winner": bool(score[index] >= q.reader["threshold"]),
            "is_public_winner": bool(index == winner),
        })

    # The fixed anchor/view is the last public candidate from the prefix view
    # in this region (not a truth label).  It lets later graph evidence be
    # inspected against exactly the same candidate ID.
    anchor = None
    if len(use):
        anchor = int(use[np.argmax(public["frames"][use] * 10_000 + public["slots"][use])])
    anchor_info = None
    if anchor is not None:
        anchor_info = {
            "id": [int(public["frames"][anchor]), int(public["slots"][anchor])],
            "target_probability": float(p[anchor]),
            "confidence": float(score[anchor]),
            "accepted_if_winner": bool(score[anchor] >= q.reader["threshold"]),
            "is_public_winner": bool(anchor == winner),
        }
    return {
        "region": int(region),
        "frame_cut_exclusive": int(cut),
        "threshold": float(q.reader["threshold"]),
        "winner_id": winner_id,
        "winner_target_probability": winner_score if winner < 0 else float(p[winner]),
        "winner_confidence": winner_score,
        "accepted": accepted,
        "submitted_id": submitted_id,
        "fixed_anchor_view": anchor_info,
        "candidates": all_candidates,
        "frame_count": int(len(public["frames"])),
    }


def _stage_truth_label(q, node, cut, region, live_state, truth_at, confidence):
    """Score-only companion for a public stage snapshot."""
    public, state = live_state(q.env, q.spec, node, q.reader)
    private = truth_at(node, public, q.trial)
    score = confidence(public, state, q.reader["risk"], max_frame=cut)
    use = np.flatnonzero((public["frames"] < cut) & (public["regions"] == region))
    if not len(use):
        return -99
    best = int(use[np.argmax(score[use])])
    return int(private["labels"][best]) if score[best] >= q.reader["threshold"] else -99


def _region_diagnosis(record):
    outcomes = [int(x) for x in record["region_outcomes"]]
    selected = [int(x) for x in record["selected_ids"]]
    rows = []
    for region, (outcome, selected_id, captured) in enumerate(zip(outcomes, selected, record["capture"], strict=True)):
        rows.append({
            "region": int(region),
            "captured": bool(captured),
            "submitted": selected_id != -99,
            "selected_truth_role_for_score": selected_id,
            "status": "correct" if outcome == 1 else ("wrong" if outcome == 0 else "not_submitted"),
        })
    return rows


def _compare_region_records(a, b):
    """Return region-level sets without attributing an unexplained cause."""
    ao = np.asarray(a["region_outcomes"], dtype=np.int8)
    bo = np.asarray(b["region_outcomes"], dtype=np.int8)
    return {
        "correct_in_both": np.flatnonzero((ao == 1) & (bo == 1)).astype(int).tolist(),
        "failures_in_both": np.flatnonzero((ao != 1) & (bo != 1)).astype(int).tolist(),
        "correct_only_in_a": np.flatnonzero((ao == 1) & (bo != 1)).astype(int).tolist(),
        "correct_only_in_b": np.flatnonzero((ao != 1) & (bo == 1)).astype(int).tolist(),
        "wrong_only_in_a": np.flatnonzero((ao == 0) & (bo != 0)).astype(int).tolist(),
        "wrong_only_in_b": np.flatnonzero((ao != 0) & (bo == 0)).astype(int).tolist(),
    }


def _decision_transition(before, after):
    """Classify the public old-pool decision without using truth labels."""
    same_id = before["winner_id"] == after["winner_id"]
    same_acceptance = before["accepted"] == after["accepted"]
    if same_id:
        if before["accepted"] and not after["accepted"]:
            category = "abstention_same_id"
        elif not before["accepted"] and after["accepted"]:
            category = "adoption_same_id"
        elif not same_acceptance:
            category = "same_id_acceptance_change"
        elif before["winner_target_probability"] != after["winner_target_probability"] \
                or before["winner_confidence"] != after["winner_confidence"]:
            category = "same_id_probability_update"
        else:
            category = "same_id_no_change"
    elif after["accepted"]:
        category = "replacement_adopted"
    elif before["accepted"]:
        category = "replacement_to_abstention"
    else:
        category = "replacement_remains_abstention"
    return {
        "category": category,
        "same_winner_id": bool(same_id),
        "same_acceptance": bool(same_acceptance),
        "before_accepted": bool(before["accepted"]),
        "after_accepted": bool(after["accepted"]),
    }


def _region_status(record, region):
    """Classify a failed delivery using public acquisition/submission fields."""
    if int(record["region_outcomes"][region]) == 1:
        return "correct"
    if not bool(record["capture"][region]):
        return "not_captured"
    if int(record["selected_ids"][region]) == -99:
        return "abstain"
    return "wrong"


def _after_review_vs_hold(result):
    """Compare public winner IDs/acceptance after review, without labels."""
    oracle = result["arms"]["oracle_review"]["geometry_and_stages"]["actual"]["old_candidate_stage"]["after_review"]
    hold = result["arms"]["equal_time_hold"]["geometry_and_stages"]["actual"]["old_candidate_stage"]["after_review"]
    same_id = oracle["winner_id"] == hold["winner_id"]
    oa, ha = bool(oracle["accepted"]), bool(hold["accepted"])
    if same_id and oa and ha:
        category = "same_id_both_accept"
    elif same_id and oa and not ha:
        category = "same_id_oracle_accept_hold_abstain"
    elif same_id and not oa and ha:
        category = "same_id_oracle_abstain_hold_accept"
    elif same_id:
        category = "same_id_both_abstain"
    elif oa and ha:
        category = "replacement_both_accept"
    elif oa and not ha:
        category = "replacement_oracle_accept_hold_abstain"
    elif not oa and ha:
        category = "replacement_oracle_abstain_hold_accept"
    else:
        category = "replacement_both_abstain"
    return {
        "category": category,
        "same_winner_id": bool(same_id),
        "oracle_winner_id": oracle["winner_id"],
        "hold_winner_id": hold["winner_id"],
        "oracle_accepted": oa,
        "hold_accepted": ha,
        "oracle_submitted_id": oracle["submitted_id"],
        "hold_submitted_id": hold["submitted_id"],
    }


def _replay_case(payload):
    """Worker entrypoint; all imports happen here for spawn-safe CPU workers."""
    sys.path.insert(0, str(V4))
    # Importing v4's modules after V4 is first is important: active_trial's
    # study_worlds must be the archived corrected world, not a v3 module.
    from active_trial import ACTIONS, FUTURE_SEEDS, TAPES, init_action_study, review, reached_prefix, score_branch
    from action_eval import record
    from fit_public import load_public
    from readout_state import confidence, export_cpu
    from study_worlds import DEFAULT_PACK, complete, motion, truth_at
    from environment import Case

    freeze = _load_json(V4_RUN / "reader_freeze.json")
    readers = _readers(freeze, export_cpu, load_public)
    init_action_study(DEFAULT_PACK, readers)

    row = payload["row"]
    case = _case_from_meta(row["meta"], Case)
    q = reached_prefix(case)
    env, spec, prefix, trial, limits = q.env, q.spec, q.node, q.trial, q.limits
    cut = len(prefix.reader.frames)
    region = int(row["meta"]["prefix_region"])
    if cut != int(row["meta"]["prefix_frame_count"]):
        raise AssertionError(f"{case.identifier}: frame cut {cut} != archived {row['meta']['prefix_frame_count']}")
    if int(prefix.station) != int(row["meta"]["prefix_station"]) or int(prefix.sector) != int(row["meta"]["prefix_sector"]):
        raise AssertionError(f"{case.identifier}: prefix pose does not reproduce archive")
    if int(q.slot) != int(row["meta"]["review_slot_ticks"]):
        raise AssertionError(f"{case.identifier}: review slot does not reproduce archive")

    before_public = _candidate_stage(q, prefix, cut, region, __import__("active_trial").live_state,
                                     truth_at, confidence)
    before_label = _stage_truth_label(q, prefix, cut, region, __import__("active_trial").live_state,
                                      truth_at, confidence)
    base_old = score_branch(q, prefix, cut, region, complete_scan=False)
    if int(base_old) != int(before_label):
        raise AssertionError(f"{case.identifier}: before label mismatch")

    offsets = {f"{name}_{i}": offset for name, seq in FUTURE_SEEDS.items() for i, offset in enumerate(seq)}
    tapes = [trial if name == "actual" else trial.branch(prefix.tick, case.tape_seed + offsets[name]) for name in TAPES]
    # The report's natural test uses actual, proof_0, proof_1 in this order.
    scored_names = ("actual", "proof_0", "proof_1")
    scored_tapes = {name: tapes[TAPES.index(name)] for name in scored_names}

    arms, actual_rejoin = {}, {}
    for arm_name in ("mpc", "equal_time_hold", "oracle_review"):
        archived = row["arms"][arm_name]
        action_name = archived["action"]
        action = ACTIONS.index(action_name)
        replay_scores = []
        arm_stages = {}
        branch_geometry = {}
        for stream_name in scored_names:
            tape = scored_tapes[stream_name]
            cur, trajectory = review(env, spec, prefix, tape, action, q.plans, q.slot, q.caps)
            after_public = _candidate_stage(q, cur, cut, region, __import__("active_trial").live_state,
                                            truth_at, confidence)
            after_label = _stage_truth_label(q, cur, cut, region, __import__("active_trial").live_state,
                                             truth_at, confidence)
            # run_search records the post-review old label for the actual and
            # proof tapes.  The two search tapes are used only to choose the
            # action and retain the common prefix label.
            logged_after = after_label if action else int(base_old)
            rejoin = {
                "station": int(cur.station), "sector": int(cur.sector), "tick": int(cur.tick),
                "distance_mm": float(cur.distance),
                "elapsed_time_s": float(cur.tick * motion.CLOCK_S),
                "same_common_boundary": bool(
                    cur.station == prefix.station and cur.sector == prefix.sector
                    and cur.tick == prefix.tick + (q.slot if action else 0)
                ),
            }
            rejoin_frame_count = len(cur.reader.frames)
            if stream_name == "actual":
                actual_rejoin[arm_name] = cur.fork()
            complete(env, spec, cur, tape, limits)
            suffix_sequence = _suffix_sequence(cur, rejoin_frame_count)
            values, detail, old_terminal = score_branch(q, cur, cut, region, complete_scan=True)
            peak = max((abs(motion._sector_delta(item[2], prefix.sector)) for item in trajectory), default=0.)
            extent = max((abs(item[1] - prefix.station) for item in trajectory), default=0.)
            actual = record(values, detail, cur.tick * motion.CLOCK_S, cur.distance, logged_after, old_terminal)
            replay_scores.append(actual)
            # Per-stage public IDs only use the old candidate pool (< cut).
            terminal_public = _candidate_stage(q, cur, cut, region, __import__("active_trial").live_state,
                                               truth_at, confidence)
            before_pool_ids = {tuple(item["id"]) for item in before_public["candidates"]}
            terminal_pool_ids = {tuple(item["id"]) for item in terminal_public["candidates"]}
            persisted_pool_ids = sorted(before_pool_ids & terminal_pool_ids)
            suffix_regions = list(range(region + 1, 6))
            legal_budget = bool(cur.station == spec[2] and cur.tick <= limits.ticks and cur.distance <= limits.mm + 1.e-6)
            suffix_all_correct = bool(all(actual["region_outcomes"][r] == 1 for r in suffix_regions))
            old_end_correct = bool(int(old_terminal) == 0)
            branch_geometry[stream_name] = {
                "action": action_name,
                "action_index": int(action),
                "trajectory": {
                    "frames": int(len(trajectory)),
                    "peak_sector_delta": float(peak),
                    "extent_station": float(extent),
                    "rejoin": rejoin,
                },
                "rejoin_frame_count": int(rejoin_frame_count),
                "suffix_after_rejoin": {
                    "definition": "all scanner observations appended after the review rejoin frame and before terminal completion",
                    "sequence_station_sector_tick": suffix_sequence,
                    "sequence_station_sector": _spatial_sequence(suffix_sequence),
                    "frame_count": int(len(suffix_sequence)),
                    "sha256": _sequence_hash(suffix_sequence),
                    "pose_only_sha256": _sequence_hash(_spatial_sequence(suffix_sequence)),
                    "acquired_echo_sha256": _echo_hash(cur.reader.frames[rejoin_frame_count:]),
                },
                "terminal": {
                    "station": int(cur.station), "sector": int(cur.sector), "tick": int(cur.tick),
                    "distance_mm": float(cur.distance), "elapsed_time_s": float(cur.tick * motion.CLOCK_S),
                    "legal_budget": legal_budget,
                },
                "old_candidate_stage": {
                    "before": before_public,
                    "after_review": after_public,
                    "terminal": terminal_public,
                },
                "terminal_old_pool": {
                    "before_candidate_count": int(len(before_pool_ids)),
                    "terminal_candidate_count": int(len(terminal_pool_ids)),
                    "persists": bool(bool(before_pool_ids) and bool(persisted_pool_ids)),
                    "persisted_candidate_ids": [list(item) for item in persisted_pool_ids],
                },
                "decision_transitions": {
                    "before_to_after_review": _decision_transition(before_public, after_public),
                    "after_review_to_terminal": _decision_transition(after_public, terminal_public),
                },
                "old_label_score_only": {"before": int(base_old), "after_review": int(after_label),
                                         "terminal": int(old_terminal)},
                "joint": {
                    "definition": "old candidate from frames<cut correct at terminal AND every region index > prefix_region correct AND complete within time/distance budget",
                    "old_candidate_end_correct": old_end_correct,
                    "required_suffix_regions": suffix_regions,
                    "suffix_all_correct": suffix_all_correct,
                    "complete_and_budget_legal": legal_budget,
                    "joint": bool(old_end_correct and suffix_all_correct and legal_budget),
                },
                "region_diagnosis": _region_diagnosis(actual),
            }

        # Use the archived three score records in the same order and verify
        # every scalar/list used by the report, including selected truth-role
        # IDs.  Public frame/slot IDs are an additional forensic field.
        for index, (expected, actual) in enumerate(zip(archived["scores"], replay_scores, strict=True)):
            _assert_equal(expected, actual, f"{case.identifier}.{arm_name}.scores[{index}]")
        arms[arm_name] = {
            "action": action_name,
            "action_index": int(action),
            "scores": replay_scores,
            "recorded_scores": archived["scores"],
            "geometry_and_stages": branch_geometry,
        }

    # The stage IDs for the actual/proof branches are identical in the archive
    # except when a new branch changes graph state. Store all arms and compare
    # the explicit paired region outcomes without claiming a causal mechanism.
    compare = {}
    for left, right in (("oracle_review", "equal_time_hold"), ("oracle_review", "mpc")):
        compare[f"{left}_vs_{right}"] = {}
        for index, stream_name in enumerate(scored_names):
            compare[f"{left}_vs_{right}"][stream_name] = _compare_region_records(
                arms[left]["scores"][index], arms[right]["scores"][index])

    suffix_echo = {}
    for stream_name in scored_names:
        suffix_echo[stream_name] = {}
        for left, right in (("equal_time_hold", "mpc"), ("oracle_review", "mpc"),
                            ("oracle_review", "equal_time_hold")):
            left_sequence = arms[left]["geometry_and_stages"][stream_name]["suffix_after_rejoin"]
            right_sequence = arms[right]["geometry_and_stages"][stream_name]["suffix_after_rejoin"]
            full = _sequence_echo(left_sequence["sequence_station_sector_tick"],
                                  right_sequence["sequence_station_sector_tick"])
            full["pose_only"] = _sequence_echo(
                _spatial_sequence(left_sequence["sequence_station_sector_tick"]),
                _spatial_sequence(right_sequence["sequence_station_sector_tick"]))
            full["same_acquired_echoes"] = left_sequence["acquired_echo_sha256"] == right_sequence["acquired_echo_sha256"]
            suffix_echo[stream_name][f"{left}_vs_{right}"] = full

    after_review_vs_hold = _after_review_vs_hold({"arms": arms})

    reference = arms["mpc"]["geometry_and_stages"]["actual"]
    poses = reference["suffix_after_rejoin"]["sequence_station_sector_tick"]
    forward_length = reference["terminal"]["distance_mm"] - actual_rejoin["mpc"].distance
    delay = actual_rejoin["mpc"].fork()
    delay.tick += q.slot
    timed_arms = {"delay_without_images": delay, "delay_with_hold_images": actual_rejoin["equal_time_hold"],
                  "delay_with_review_images": actual_rejoin["oracle_review"]}
    timed = {name: _fixed_suffix(q, node, poses, forward_length, trial, cut, region, score_branch)
             for name, node in timed_arms.items()}
    if len({v["suffix_echo_sha256"] for v in timed.values() if v is not None}) > 1:
        raise AssertionError("Fixed pose and absolute clock gave different raw measurements")

    actual_oracle = arms["oracle_review"]["scores"][0]
    actual_hold = arms["equal_time_hold"]["scores"][0]
    actual_mpc = arms["mpc"]["scores"][0]
    return {
        "case": asdict(case),
        "layout_id": row["meta"]["layout_id"],
        "route": case.task,
        "selection_membership": {
            "old_review_rescue_vs_hold": bool(actual_oracle["old_after_review"] == 0 and actual_hold["old_after_review"] != 0),
            "mpc_strict_success_oracle_strict_loss": bool(actual_mpc["strict"] == 1 and actual_oracle["strict"] == 0),
        },
        "prefix": {
            "station": int(prefix.station), "sector": int(prefix.sector), "region": int(region),
            "frame_cut_exclusive": int(cut), "tick": int(prefix.tick), "distance_mm": float(prefix.distance),
            "review_slot_ticks": int(q.slot),
            "pose_boundary_definition": "all arms rejoin at prefix station/sector and prefix tick + review_slot_ticks",
        },
        "reader": {"method": q.reader["method"], "threshold": float(q.reader["threshold"]),
                   "position_threshold": float(q.reader["position_threshold"])},
        "before": {"public_candidates": before_public, "old_label_score_only": int(base_old)},
        "arms": arms,
        "paired_region_diagnostics": compare,
        "suffix_echo_by_tape": suffix_echo,
        "actual_prescribed_mpc_path": timed,
        "after_review_vs_hold": after_review_vs_hold,
        "suffix_definition": {
            "required_regions": list(range(region + 1, 6)),
            "meaning": "regions with index greater than the selected prefix region; this is a diagnostic partition, not a claim about why a branch failed",
            "timing_causality": "not inferred from association; path, pose, elapsed time and readout outcomes are recorded separately",
        },
    }


def _worker(payload):
    return _replay_case(payload)


def _arm_group_metrics(results, arm, stream):
    branches = [result["arms"][arm]["geometry_and_stages"][stream] for result in results]
    scores = [result["arms"][arm]["scores"][("actual", "proof_0", "proof_1").index(stream)]
              for result in results]
    return {
        "cases": int(len(branches)),
        "terminal_old_pool_persists": int(sum(branch["terminal_old_pool"]["persists"] for branch in branches)),
        "final_old_section_correct": int(sum(branch["joint"]["old_candidate_end_correct"] for branch in branches)),
        "required_suffix_all_correct": int(sum(branch["joint"]["suffix_all_correct"] for branch in branches)),
        "strict": int(sum(score["strict"] == 1 for score in scores)),
        "joint": int(sum(branch["joint"]["joint"] for branch in branches)),
        "complete_and_budget_legal": int(sum(branch["joint"]["complete_and_budget_legal"] for branch in branches)),
    }


def _pair_region_aggregate(results, probe, reference, stream):
    index = ("actual", "proof_0", "proof_1").index(stream)
    split_counts = {
        "at_or_before_prefix": {"common_failures": 0, "newly_lost_in_probe": 0,
                                "newly_rescued_in_probe": 0, "common_failure_status_pairs": Counter(),
                                "newly_lost_probe_status": Counter()},
        "after_prefix": {"common_failures": 0, "newly_lost_in_probe": 0,
                          "newly_rescued_in_probe": 0, "common_failure_status_pairs": Counter(),
                          "newly_lost_probe_status": Counter()},
    }
    per_region = {str(region): {"common_failures": 0, "newly_lost_in_probe": 0,
                                "newly_rescued_in_probe": 0} for region in range(6)}
    for result in results:
        prefix_region = int(result["prefix"]["region"])
        a = result["arms"][probe]["scores"][index]
        b = result["arms"][reference]["scores"][index]
        for region in range(6):
            status_probe = _region_status(a, region)
            status_reference = _region_status(b, region)
            split = "at_or_before_prefix" if region <= prefix_region else "after_prefix"
            row = split_counts[split]
            if status_probe != "correct" and status_reference != "correct":
                row["common_failures"] += 1
                row["common_failure_status_pairs"][f"{status_probe}/{status_reference}"] += 1
                per_region[str(region)]["common_failures"] += 1
            if status_probe != "correct" and status_reference == "correct":
                row["newly_lost_in_probe"] += 1
                row["newly_lost_probe_status"][status_probe] += 1
                per_region[str(region)]["newly_lost_in_probe"] += 1
            if status_probe == "correct" and status_reference != "correct":
                row["newly_rescued_in_probe"] += 1
                per_region[str(region)]["newly_rescued_in_probe"] += 1
    for row in split_counts.values():
        row["common_failure_status_pairs"] = dict(row["common_failure_status_pairs"])
        row["newly_lost_probe_status"] = dict(row["newly_lost_probe_status"])
    return {
        "probe": probe,
        "reference": reference,
        "stream": stream,
        "split_definition": "at_or_before_prefix means region <= prefix_region; after_prefix means region > prefix_region",
        "failure_status_definition": "not_captured if capture=false; abstain if captured but selected_id=-99; wrong otherwise; correct is excluded from failures",
        "by_prefix_split": split_counts,
        "by_region": per_region,
    }


def _group_report(results, name, probe, reference):
    report = {
        "case_count": int(len(results)),
        "case_ids": [result["case"]["identifier"] for result in results],
        "arms_by_tape": {
            arm: {stream: _arm_group_metrics(results, arm, stream)
                  for stream in ("actual", "proof_0", "proof_1")}
            for arm in ("mpc", "equal_time_hold", "oracle_review")
        },
        "paired_region_failures": {
            stream: _pair_region_aggregate(results, probe, reference, stream)
            for stream in ("actual", "proof_0", "proof_1")
        },
        "after_review_old_winner_vs_hold": dict(Counter(
            result["after_review_vs_hold"]["category"] for result in results)),
        "suffix_echo_vs_references": {},
    }
    comparisons = (("delay_without_images", "original_mpc"), ("delay_with_hold_images", "delay_without_images"),
                   ("delay_with_review_images", "delay_with_hold_images"))
    timed_counts = {}
    for arm, ref in comparisons:
        rows = []
        for case in results:
            by_action = case["actual_prescribed_mpc_path"]
            if by_action[arm] is None or (ref != "original_mpc" and by_action[ref] is None):
                continue
            baseline = (case["arms"]["mpc"]["scores"][0]["selected_ids"] if ref == "original_mpc"
                        else by_action[ref]["region_labels"])
            first_suffix = case["prefix"]["region"] + 1
            rows.append((np.asarray(baseline[first_suffix:]) == 0,
                         np.asarray(by_action[arm]["region_labels"][first_suffix:]) == 0))
        timed_counts[f"{arm}_vs_{ref}"] = {"cases": len(rows),
            "suffix_newly_correct": int(sum((~a & b).sum() for a,b in rows)),
            "suffix_newly_incorrect": int(sum((a & ~b).sum() for a,b in rows)),
            "suffix_all_correct_a": int(sum(b.all() for a,b in rows)),
            "suffix_all_correct_reference": int(sum(a.all() for a,b in rows))}
    delayed_losses, archived_losses = set(), set()
    for case in results:
        delayed = case["actual_prescribed_mpc_path"]["delay_without_images"]
        if delayed is None:
            continue
        mpc = case["arms"]["mpc"]["scores"][0]["selected_ids"]
        active = case["arms"]["oracle_review"]["scores"][0]["selected_ids"]
        for region in range(case["prefix"]["region"] + 1, 6):
            key = (case["case"]["identifier"], region)
            if mpc[region] == 0 and delayed["region_labels"][region] != 0:
                delayed_losses.add(key)
            if mpc[region] == 0 and active[region] != 0:
                archived_losses.add(key)
    timed_counts["matched_suffix_losses_on_same_case_and_region"] = {
        "shared": len(delayed_losses & archived_losses), "delay_only": len(delayed_losses - archived_losses),
        "archived_review_only": len(archived_losses - delayed_losses)}
    report["fixed_mpc_pose_path_timing_diagnostic"] = timed_counts
    for stream in ("actual", "proof_0", "proof_1"):
        report["suffix_echo_vs_references"][stream] = {}
        for pair in ("equal_time_hold_vs_mpc", "oracle_review_vs_mpc", "oracle_review_vs_equal_time_hold"):
            values = [result["suffix_echo_by_tape"][stream][pair] for result in results]
            report["suffix_echo_vs_references"][stream][pair] = {
                "same_pose_time_sequence_cases": int(sum(value["same_sequence"] for value in values)),
                "same_pose_only_sequence_cases": int(sum(value["pose_only"]["same_sequence"] for value in values)),
                "same_acquired_echo_cases": int(sum(value["same_acquired_echoes"] for value in values)),
                "same_length_cases": int(sum(value["same_length"] for value in values)),
                "cases": int(len(values)),
            }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 6:
        raise SystemExit("--workers must be in [1, 6]")

    source = _load_json(V4_RUN / "action_results.json")
    rows = source["routes"]["groin_to_distal_calf"]["per_case"]
    old_rescue = []
    mpc_loss = []
    for index, row in enumerate(rows):
        oracle = row["arms"]["oracle_review"]["scores"][0]
        hold = row["arms"]["equal_time_hold"]["scores"][0]
        mpc = row["arms"]["mpc"]["scores"][0]
        if oracle["old_after_review"] == 0 and hold["old_after_review"] != 0:
            old_rescue.append(index)
        if mpc["strict"] == 1 and oracle["strict"] == 0:
            mpc_loss.append(index)
    union = sorted(set(old_rescue) | set(mpc_loss))
    if args.limit:
        union = union[:args.limit]
    payloads = [{"index": i, "row": rows[i]} for i in union]

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for result in pool.map(_worker, payloads):
            results.append(result)
    results.sort(key=lambda x: x["case"]["identifier"])

    for result in results:
        identifier = result["case"]["identifier"]
        _write_json(OUT / f"case_{identifier}.json", result)

    streams = ("actual", "proof_0", "proof_1")
    transition_counts = {}
    joint_counts = {}
    for arm in ("mpc", "equal_time_hold", "oracle_review"):
        transition_counts[arm] = dict(Counter(
            result["arms"][arm]["geometry_and_stages"]["actual"]["decision_transitions"]
            ["before_to_after_review"]["category"] for result in results))
        joint_counts[arm] = {}
        for stream in streams:
            branches = [result["arms"][arm]["geometry_and_stages"][stream] for result in results]
            joint_counts[arm][stream] = {
                "joint": int(sum(branch["joint"]["joint"] for branch in branches)),
                "old_candidate_end_correct": int(sum(branch["joint"]["old_candidate_end_correct"] for branch in branches)),
                "suffix_all_correct": int(sum(branch["joint"]["suffix_all_correct"] for branch in branches)),
                "complete_and_budget_legal": int(sum(branch["joint"]["complete_and_budget_legal"] for branch in branches)),
            }

    selected_comparisons = {}
    for left, right in (("oracle_review", "equal_time_hold"), ("oracle_review", "mpc")):
        deltas = []
        for result in results:
            left_score = result["arms"][left]["scores"][0]["strict"]
            right_score = result["arms"][right]["scores"][0]["strict"]
            deltas.append(left_score - right_score)
        selected_comparisons[f"{left}_vs_{right}_actual"] = {
            "strict_net": int(sum(deltas)),
            "rescue_cases": int(sum(delta > 0 for delta in deltas)),
            "loss_cases": int(sum(delta < 0 for delta in deltas)),
        }

    rescue_group = [result for result in results if result["selection_membership"]["old_review_rescue_vs_hold"]]
    loss_group = [result for result in results if result["selection_membership"]["mpc_strict_success_oracle_strict_loss"]]
    group_reports = {
        "old_review_rescue_vs_hold_22": _group_report(rescue_group, "old_review_rescue_vs_hold_22",
                                                       "oracle_review", "equal_time_hold"),
        "mpc_success_oracle_loss_25": _group_report(loss_group, "mpc_success_oracle_loss_25",
                                                    "oracle_review", "mpc"),
    }

    aggregate = {
        "source": str(V4_RUN / "action_results.json"),
        "reader_freeze": str(V4_RUN / "reader_freeze.json"),
        "route": "groin_to_distal_calf",
        "case_count_total": len(rows),
        "old_review_rescue_vs_hold_count": len(old_rescue),
        "mpc_strict_success_oracle_strict_loss_count": len(mpc_loss),
        "union_count": len(set(old_rescue) | set(mpc_loss)),
        "intersection_count": len(set(old_rescue) & set(mpc_loss)),
        "selected_case_indices": union,
        "selected_case_ids": [rows[i]["meta"]["case"]["identifier"] for i in union],
        "old_review_rescue_case_ids": [rows[i]["meta"]["case"]["identifier"] for i in old_rescue],
        "mpc_loss_case_ids": [rows[i]["meta"]["case"]["identifier"] for i in mpc_loss],
        "replay_assertions": "all archived actual/proof score fields and selected truth-role IDs matched deterministic replay within float32 tolerance",
        "actual_tape_selected_union_comparisons": selected_comparisons,
        "actual_before_to_after_review_transition_counts": transition_counts,
        "joint_counts_by_arm_and_tape": joint_counts,
        "separate_group_reports": group_reports,
        "public_id_definition": "candidate IDs are [frame_index, slot] from public history; truth labels are used only in score-only fields and region diagnostics",
        "suffix_definition": "all regions with numeric index greater than that case's prefix_region; joint additionally requires old terminal label 0 and legal complete budget",
        "timing_caution": "pose/time differences are recorded; no path or timing cause is inferred from graph association",
        "case_output_files": [str(OUT / f"case_{result['case']['identifier']}.json") for result in results],
    }
    _write_json(OUT / "summary.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
