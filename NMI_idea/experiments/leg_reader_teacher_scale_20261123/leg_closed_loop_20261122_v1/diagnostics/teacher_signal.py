#!/usr/bin/env python3
"""Audit privileged teacher signals in committed training shards.

The unit of the primary counts is an eligible teacher-sampled prefix. Branch
replays are evaluated for rescue, harm, and utility signs, but are never added
to the prefix or scan totals. The script only reads committed training shards
and writes two JSON artifacts beside itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_BATCH_ROOT = Path(__file__).resolve().parents[1] / "training" / "seed_276382173" / "teacher_ppo" / "batches"
DEFAULT_OUT = Path(__file__).resolve().parent


def _read_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def _safe_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).lower() in {"true", "1", "yes"}


def _sign(value: float) -> int:
    return int(value > 0.) - int(value < 0.)


def _sign_label(gains):
    signs = [_sign(float(value)) for value in gains]
    if signs and all(sign > 0 for sign in signs):
        return "positive_both"
    if signs and all(sign < 0 for sign in signs):
        return "negative_both"
    if any(sign > 0 for sign in signs) and any(sign < 0 for sign in signs):
        return "mixed"
    if any(sign > 0 for sign in signs):
        return "positive_any"
    if any(sign < 0 for sign in signs):
        return "negative_any"
    return "zero"


def _new_branch_counts():
    return {
        "eligible_prefixes": 0,
        "legal_tested_branches": 0,
        "legal_tested_action_branches": 0,
        "legal_skill_counts": {"continue": 0, "window": 0, "revisit": 0, "other": 0},
        "utility": {"positive_both": 0, "negative_both": 0, "mixed": 0,
                     "positive_any": 0, "negative_any": 0,
                     "positive_on_at_least_one_tape": 0,
                     "negative_on_at_least_one_tape": 0, "zero": 0},
        "robust_rescue_both_tape": 0,
        "robust_harm_both_tape": 0,
        "strict_harm_any_tape": 0,
        "replay_pairs_examined": 0,
    }


def _skill_name(skill):
    return {0: "continue", 1: "window", 2: "revisit"}.get(int(skill), "other")


def _branch_is_legal(branch):
    if branch.get("feasible", True) is False:
        return False
    replays = branch.get("replays", [])
    return all(replay.get("feasible", replay.get("legal", True)) is not False
               for replay in replays)


def _valid_replays(branch):
    if not _branch_is_legal(branch):
        return []
    replays = branch.get("replays", [])
    if len(replays) < 2:
        return []
    valid = [replay for replay in replays
             if replay.get("feasible", replay.get("legal", True)) is not False]
    return valid if len(valid) >= 2 else []


def _safe_counter_update(target, source):
    target["eligible_prefixes"] += source["eligible_prefixes"]
    target["legal_tested_branches"] += source["legal_tested_branches"]
    target["legal_tested_action_branches"] += source["legal_tested_action_branches"]
    for key, value in source["legal_skill_counts"].items():
        target["legal_skill_counts"][key] += value
    for key, value in source["utility"].items():
        target["utility"][key] += value
    for key in ("robust_rescue_both_tape", "robust_harm_both_tape",
                "strict_harm_any_tape", "replay_pairs_examined"):
        target[key] += source[key]


def _group(target, key):
    if key not in target:
        target[key] = _new_branch_counts()
    return target[key]


def _episode_group(episode):
    source = episode.get("source", "unknown")
    profile = episode.get("profile", "unknown")
    challenge = int(_safe_bool(episode.get("challenge", False)))
    return f"source_{source}|profile_{profile}|challenge_{challenge}"


def _branch_origin(branch):
    return str(branch.get("origin", "unknown"))


def _full_tail_record(row, branches):
    tail = row.get("full_tail")
    if not isinstance(tail, dict):
        return None
    candidate = tail.get("candidate")
    if not isinstance(candidate, int) or not 0 <= candidate < len(branches):
        return {"available": True, "candidate_valid": False}
    local = branches[candidate].get("paired_gain", [])
    terminal = tail.get("paired_terminal_gain", [])
    if not isinstance(local, list) or not isinstance(terminal, list):
        return {"available": True, "candidate_valid": False}
    length = min(len(local), len(terminal))
    local, terminal = local[:length], terminal[:length]
    tolerance = 1.e-12
    # The training logger treats an exact zero as its own sign class. Keep that
    # convention here and expose the tolerance so re-runs are auditable.
    disagreement = [bool((abs(float(a)) <= tolerance) != (abs(float(b)) <= tolerance)
                         or float(a) * float(b) < -(tolerance ** 2))
                    for a, b in zip(local, terminal)]
    strict_opposite = [bool(float(a) * float(b) < -(tolerance ** 2))
                       for a, b in zip(local, terminal)]
    stored = tail.get("sign_conflict")
    return {
        "available": True, "candidate_valid": True,
        "local_sign": [_sign(float(value)) for value in local],
        "terminal_sign": [_sign(float(value)) for value in terminal],
        "sign_disagreement": disagreement,
        "strict_opposite_nonzero": strict_opposite,
        "sign_tolerance": tolerance,
        "sign_definition": "zero is a separate sign class, matching recorded sign_conflict",
        "stored_sign_conflict_matches": stored is None or list(stored) == disagreement,
    }


def audit(batch_root: Path, max_batch: int | None = None, max_episode: int | None = None):
    batch_root = Path(batch_root)
    batch_dirs, commit_records, failures = [], [], []
    for path in sorted(batch_root.glob("[0-9][0-9][0-9][0-9][0-9]")):
        commit_path = path / "commit.json"
        if not commit_path.is_file():
            continue
        commit = json.loads(commit_path.read_text())
        batch = int(commit.get("batch", int(path.name)))
        if max_batch is not None and batch > max_batch:
            continue
        if commit.get("partition") != "training" or commit.get("arm") != "teacher_ppo":
            failures.append({"batch": batch, "reason": "unexpected_commit_identity"})
            continue
        if not (path / "teacher.jsonl.gz").is_file() or not (path / "episodes.jsonl.gz").is_file():
            failures.append({"batch": batch, "reason": "missing_teacher_or_episode_log"})
            continue
        batch_dirs.append(path)
        commit_records.append(commit)

    episodes = {}
    episode_batch = {}
    for path in batch_dirs:
        commit = json.loads((path / "commit.json").read_text())
        batch = int(commit["batch"])
        expected = int(commit["stop_episode"]) - int(commit["start_episode"])
        count = 0
        for _, episode in _read_jsonl(path / "episodes.jsonl.gz"):
            identifier = episode.get("identifier")
            if identifier in episodes:
                failures.append({"batch": batch, "reason": "duplicate_episode_identifier", "identifier": identifier})
            episodes[identifier] = episode
            episode_batch[identifier] = batch
            count += 1
        if count != expected:
            failures.append({"batch": batch, "reason": "episode_count_mismatch", "expected": expected, "observed": count})

    summary = {
        "protocol": {
            "batch_root": str(batch_root),
            "selected_max_batch": max_batch,
            "selected_max_episode": max_episode,
            "included_batches": [int(record["batch"]) for record in commit_records],
            "commits": len(commit_records),
        },
        "prefixes": {
            "teacher_rows": 0, "eligible_teaching_prefixes": 0,
            "ineligible_teacher_rows": 0, "student_active_prefixes": 0,
        },
        "branches": _new_branch_counts(),
        "by_action_type": {},
        "by_source_profile_challenge": {},
        "student_action": {
            "active_legal_prefixes": 0, "active_ineligible_or_illegal": 0,
            "resimulated_robust_rescue_both_tape": 0,
            "resimulated_robust_harm_both_tape": 0,
            "resimulated_strict_harm_any_tape": 0,
            "actual_same_prefix_rescue_using_student_outcome": 0,
            "actual_same_prefix_harm_using_student_outcome": 0,
            "semantics": {
                "active": "eligible teacher row has a feasible origin=student branch with skill 1 or 2",
                "resimulated": "student branch's two teacher replays satisfy the strict paired outcome rule",
                "actual_same_prefix": "teacher row student_outcome is the observed rollout outcome at the selected prefix region, compared with the continue replay pair",
            },
        },
        "full_tail": {
            "rows_available": 0, "rows_candidate_valid": 0,
            "tape_sign_disagreement": 0, "rows_with_sign_disagreement": 0,
            "strict_opposite_nonzero_tapes": 0,
            "stored_conflict_match": 0, "stored_conflict_mismatch": 0,
            "sign_definition": "zero is a separate sign class, exact magnitudes within 1e-12 are zero",
        },
        "prefix_level": {
            "eligible_prefixes_with_robust_rescue_both_tape": 0,
            "eligible_prefixes_with_robust_harm_both_tape": 0,
            "eligible_prefixes_with_positive_both_tape_utility": 0,
            "eligible_prefixes_with_negative_both_tape_utility": 0,
            "eligible_prefixes_with_mixed_paired_utility": 0,
            "definitions": {
                "rescue": "continue outcome is missed or wrong on both replays and action outcome is correct on both",
                "harm": "continue outcome is correct on both replays and action outcome is missed or wrong on both",
                "mixed_paired_utility": "the action paired_gain has at least one positive and one negative tape value",
                "deduplication": "each eligible teacher row contributes at most one prefix to each field",
            },
        },
        "verification": {
            "episode_rows": len(episodes), "teacher_rows_matching_episode": 0,
            "shared_layout_match": 0, "shared_layout_mismatch": 0,
            "student_tape_matches_episode": 0, "student_tape_mismatch": 0,
            "future_seed_distinct": 0, "future_seed_equal": 0,
            "replays_excluded_from_scan_counts": True,
            "replay_records_examined": 0, "malformed_rows": 0,
            "failures": failures,
        },
    }

    by_origin = summary["by_action_type"]
    by_group = summary["by_source_profile_challenge"]
    for path in batch_dirs:
        commit = json.loads((path / "commit.json").read_text())
        for line_number, row in _read_jsonl(path / "teacher.jsonl.gz"):
            episode = episodes.get(row.get("episode"))
            if episode is None:
                summary["prefixes"]["teacher_rows"] += 1
                summary["verification"]["malformed_rows"] += 1
                failures.append({"batch": commit["batch"], "line": line_number, "reason": "teacher_episode_missing"})
                continue
            if max_episode is not None and int(episode.get("identifier", "_0").rsplit("_", 1)[-1]) > max_episode:
                continue
            summary["prefixes"]["teacher_rows"] += 1
            summary["verification"]["teacher_rows_matching_episode"] += 1
            if not row.get("eligible_prefix", False):
                summary["prefixes"]["ineligible_teacher_rows"] += 1
                continue
            summary["prefixes"]["eligible_teaching_prefixes"] += 1
            if row.get("fixed_layout_id") == episode.get("layout_id"):
                summary["verification"]["shared_layout_match"] += 1
            else:
                summary["verification"]["shared_layout_mismatch"] += 1
                failures.append({"batch": commit["batch"], "line": line_number, "reason": "layout_mismatch"})
            if row.get("student_tape_seed") == episode.get("tape_seed"):
                summary["verification"]["student_tape_matches_episode"] += 1
            else:
                summary["verification"]["student_tape_mismatch"] += 1
            student_seed, independent_seed = row.get("student_tape_seed"), row.get("independent_tape_seed")
            if student_seed is not None and independent_seed is not None and student_seed != independent_seed:
                summary["verification"]["future_seed_distinct"] += 1
            else:
                summary["verification"]["future_seed_equal"] += 1
            branches = row.get("branches", [])
            base = _new_branch_counts()
            base["eligible_prefixes"] = 1
            action_student_branch = None
            prefix_rescue = False
            prefix_harm = False
            prefix_positive_both = False
            prefix_negative_both = False
            prefix_mixed = False
            continue_branch = next((branch for branch in branches
                                    if int(branch.get("skill", 0)) == 0
                                    and _branch_origin(branch) == "continue"), None)
            group = _group(by_group, _episode_group(episode))
            seen_origins = set()
            for branch in branches:
                skill = int(branch.get("skill", 0))
                origin = _branch_origin(branch)
                origin_counts = _group(by_origin, origin)
                if origin not in seen_origins:
                    origin_counts["eligible_prefixes"] += 1
                    seen_origins.add(origin)
                legal = _branch_is_legal(branch)
                summary["verification"]["replay_records_examined"] += len(branch.get("replays", []))
                if legal:
                    base["legal_tested_branches"] += 1
                    base["legal_skill_counts"][_skill_name(skill)] += 1
                    origin_counts["legal_tested_branches"] += 1
                    origin_counts["legal_skill_counts"][_skill_name(skill)] += 1
                if origin == "student" and skill > 0:
                    action_student_branch = branch
                if skill == 0 or not legal:
                    continue
                base["legal_tested_action_branches"] += 1
                origin_counts["legal_tested_action_branches"] += 1
                replays = _valid_replays(branch)
                if not replays:
                    continue
                gains = branch.get("paired_gain", [])
                if isinstance(gains, list) and len(gains) >= 2:
                    label = _sign_label(gains[:2])
                    base["utility"][label] += 1
                    prefix_positive_both |= label == "positive_both"
                    prefix_negative_both |= label == "negative_both"
                    prefix_mixed |= label == "mixed"
                    if any(float(value) > 0. for value in gains[:2]):
                        base["utility"]["positive_on_at_least_one_tape"] += 1
                        origin_counts["utility"]["positive_on_at_least_one_tape"] += 1
                    if any(float(value) < 0. for value in gains[:2]):
                        base["utility"]["negative_on_at_least_one_tape"] += 1
                        origin_counts["utility"]["negative_on_at_least_one_tape"] += 1
                    origin_counts["utility"][label] += 1
                base["replay_pairs_examined"] += 1
                origin_counts["replay_pairs_examined"] += 1
                if continue_branch is None:
                    continue
                baseline_replays = _valid_replays(continue_branch)
                if len(baseline_replays) < 2:
                    continue
                baseline_outcomes = [replay.get("outcome", -1) for replay in baseline_replays[:2]]
                action_outcomes = [replay.get("outcome", -1) for replay in replays[:2]]
                if all(outcome != 1 for outcome in baseline_outcomes) and all(outcome == 1 for outcome in action_outcomes):
                    base["robust_rescue_both_tape"] += 1
                    prefix_rescue = True
                    origin_counts["robust_rescue_both_tape"] += 1
                if all(outcome == 1 for outcome in baseline_outcomes) and all(outcome != 1 for outcome in action_outcomes):
                    base["robust_harm_both_tape"] += 1
                    prefix_harm = True
                    origin_counts["robust_harm_both_tape"] += 1
                if all(outcome == 1 for outcome in baseline_outcomes) and any(outcome != 1 for outcome in action_outcomes):
                    base["strict_harm_any_tape"] += 1
                    origin_counts["strict_harm_any_tape"] += 1

            _safe_counter_update(summary["branches"], base)
            _safe_counter_update(group, base)
            _safe_counter_update(_group(by_origin, "__all__"), base)
            summary["prefix_level"]["eligible_prefixes_with_robust_rescue_both_tape"] += int(prefix_rescue)
            summary["prefix_level"]["eligible_prefixes_with_robust_harm_both_tape"] += int(prefix_harm)
            summary["prefix_level"]["eligible_prefixes_with_positive_both_tape_utility"] += int(prefix_positive_both)
            summary["prefix_level"]["eligible_prefixes_with_negative_both_tape_utility"] += int(prefix_negative_both)
            summary["prefix_level"]["eligible_prefixes_with_mixed_paired_utility"] += int(prefix_mixed)
            if action_student_branch is not None and _branch_is_legal(action_student_branch):
                summary["prefixes"]["student_active_prefixes"] += 1
                summary["student_action"]["active_legal_prefixes"] += 1
                student_replays = _valid_replays(action_student_branch)
                continue_replays = _valid_replays(continue_branch) if continue_branch else []
                if len(student_replays) >= 2 and len(continue_replays) >= 2:
                    baseline = [item.get("outcome", -1) for item in continue_replays[:2]]
                    active = [item.get("outcome", -1) for item in student_replays[:2]]
                    if all(value != 1 for value in baseline) and all(value == 1 for value in active):
                        summary["student_action"]["resimulated_robust_rescue_both_tape"] += 1
                    if all(value == 1 for value in baseline) and all(value != 1 for value in active):
                        summary["student_action"]["resimulated_robust_harm_both_tape"] += 1
                    if all(value == 1 for value in baseline) and any(value != 1 for value in active):
                        summary["student_action"]["resimulated_strict_harm_any_tape"] += 1
                    observed = row.get("student_outcome", -1)
                    if all(value != 1 for value in baseline) and observed == 1:
                        summary["student_action"]["actual_same_prefix_rescue_using_student_outcome"] += 1
                    if all(value == 1 for value in baseline) and observed != 1:
                        summary["student_action"]["actual_same_prefix_harm_using_student_outcome"] += 1
            elif action_student_branch is not None:
                summary["student_action"]["active_ineligible_or_illegal"] += 1

            tail = _full_tail_record(row, branches)
            if tail is not None:
                summary["full_tail"]["rows_available"] += 1
                if tail.get("candidate_valid"):
                    summary["full_tail"]["rows_candidate_valid"] += 1
                    count = sum(tail["sign_disagreement"])
                    summary["full_tail"]["tape_sign_disagreement"] += count
                    summary["full_tail"]["rows_with_sign_disagreement"] += int(count > 0)
                    summary["full_tail"]["strict_opposite_nonzero_tapes"] += sum(tail["strict_opposite_nonzero"])
                    if tail["stored_sign_conflict_matches"]:
                        summary["full_tail"]["stored_conflict_match"] += 1
                    else:
                        summary["full_tail"]["stored_conflict_mismatch"] += 1

    summary["verification"]["failure_count"] = len(failures)
    summary["verification"]["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--max-batch", type=int)
    parser.add_argument("--max-episode", type=int)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = audit(args.batch_root, args.max_batch, args.max_episode)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "teacher_signal_summary.json"
    verify_path = args.out_dir / "teacher_signal_verify.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    verification = {
        "status": "passed" if result["verification"]["failure_count"] == 0 else "failed",
        "summary": str(summary_path), "verification": result["verification"],
        "selected_batches": result["protocol"]["included_batches"],
    }
    verify_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(summary_path), "verification": str(verify_path),
                      "eligible_teaching_prefixes": result["prefixes"]["eligible_teaching_prefixes"],
                      "legal_tested_action_branches": result["branches"]["legal_tested_action_branches"],
                      "verification_status": verification["status"]}, indent=2))


if __name__ == "__main__":
    main()

