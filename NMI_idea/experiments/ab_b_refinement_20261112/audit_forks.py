"""Audit real ``B_once``/``locked`` switch states from paired public runs.

This is a targeted fork audit, not another rollout-wide comparison.  It keeps
only completed paired rows, verifies the public common head and the logged
locked tail, reconstructs the frozen prefix/head belief without using
evaluation labels for planning, and then recomputes the predictive support at
the exact scheduled suffix state.  A counterfactual is executed only when a
stronger fixed tail differs from both logged tails; it forces one public-
eligible action through a temporary :class:`Planner` subclass and delegates
the remaining continuation to the ordinary locked policy.

Example::

    python audit_forks.py --runs paired_0 paired_6 --out FORK_AUDIT.json

The simulator may draw a private truth solely to regenerate the frozen public
prefix and to produce public packets for a forced continuation.  The truth,
labels, and simulator state are never used to choose an audit tail.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "ab_fisher_continuation_20261112"
sys.path.insert(0, str(HERE))
import refine_rechecks as rr  # noqa: E402
from predictive import predictive_sets  # noqa: E402
from evidence import EvidenceBook  # noqa: E402


_PAIR = ("B_once", "locked")
_PUBLIC_KEYS = ("pose", "action", "centroids_mm")
_TOL = 1.0e-8


def _jsonable(value: Any) -> Any:
    """Convert numpy values and non-finite numbers to JSON-safe values."""

    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_number(left: Any, right: Any) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=_TOL, atol=_TOL))
    except (TypeError, ValueError):
        return left == right


def _same_public(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key in _PUBLIC_KEYS:
        if key in ("pose", "action"):
            if left.get(key) != right.get(key):
                return False
            continue
        lv = left.get(key, []) or []
        rv = right.get(key, []) or []
        if len(lv) != len(rv) or any(not _same_number(a, b) for a, b in zip(lv, rv)):
            return False
    return True


def _read_shard(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    runs_path = path if path.name == "RUNS.jsonl" else path / "RUNS.jsonl"
    if not runs_path.is_file():
        raise FileNotFoundError(f"missing RUNS.jsonl in {path}")
    rows: list[dict[str, Any]] = []
    with runs_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "seed" not in row or "arm" not in row:
                raise ValueError(f"invalid run row at {runs_path}:{line_number}")
            rows.append(row)
    config_path = runs_path.with_name("CONFIG.json")
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    return rows, config


def _collect(runs: Sequence[Path]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    cases: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    configs: dict[str, dict[str, Any]] = {}
    for path in runs:
        rows, config = _read_shard(Path(path))
        configs[str(Path(path))] = config
        for row in rows:
            seed, arm = str(row["seed"]), str(row["arm"])
            previous = cases[seed].get(arm)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting duplicate seed={seed!r}, arm={arm!r}")
            cases[seed][arm] = row
    return cases, configs


def _steps(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [step for step in row.get("steps", []) if isinstance(step, dict)]


def _first_public_divergence(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[int, bool]:
    left_steps, right_steps = _steps(left), _steps(right)
    index = 0
    while index < min(len(left_steps), len(right_steps)) and _same_public(left_steps[index], right_steps[index]):
        index += 1
    return index, bool(index < min(len(left_steps), len(right_steps)) or len(left_steps) != len(right_steps))


def _initial_belief(scan) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(scan.dim, dtype=float)
    covariance = np.asarray(scan.prior, dtype=float).copy()
    covariance[-1, -1] = rr.STRESS.slip_sd_mm**2
    return mean, covariance


def _as_action(actions: np.ndarray, index: int) -> list[int]:
    return [int(value) for value in np.asarray(actions[int(index)]).tolist()]


def _action_index(planner: rr.Planner, pose: int, action: Sequence[int]) -> int | None:
    try:
        return int(planner.index[int(pose)][tuple(int(v) for v in action)])
    except (KeyError, TypeError, ValueError):
        return None


def _tail_choice_action(planner: rr.Planner, pose: int, choice) -> list[int]:
    return _as_action(planner.actions[int(pose)], int(choice.index))


def _config_value(configs: Mapping[str, Mapping[str, Any]], key: str, default: Any) -> Any:
    for config in configs.values():
        if key in config and config[key] is not None:
            return config[key]
    return default


def _build_frozen() -> tuple[Any, Any, Any, dict[str, str]]:
    geometry = rr.DYNAMIC / "DYNAMIC_GEOMETRY_FROZEN.npz"
    with np.load(geometry) as data:
        scan = rr.DynamicScan(data)
        support = rr.build_support(scan, data)
    edge = rr.Edges(scan)
    model = rr.Recheck(scan, edge)
    source_paths = [
        HERE / "audit_forks.py",
        HERE / "refine_rechecks.py",
        HERE / "evidence.py",
        HERE / "predictive.py",
        BASE / "certification.py",
        rr.DYNAMIC / "dynamic_scan.py",
        rr.DYNAMIC / "stress_observation.py",
        geometry,
    ]
    hashes = {path.name: _sha256(path) for path in source_paths}
    hashes["edge_check_mode"] = "capability_evaluator"
    return scan, support, model, hashes


def _reconstruct_head(
    scan,
    support,
    model,
    planner: rr.Planner,
    seed: int,
    once_row: Mapping[str, Any],
    locked_row: Mapping[str, Any],
    predictive_z: float,
    path_cap: float,
    cap: int,
) -> dict[str, Any]:
    """Regenerate prefix and reconstruct the exact state before the shared first."""

    prefix_count = int(once_row.get("prefix_acquisitions", 0))
    truth, _rejects = rr.draw_truth(scan, int(seed), support)
    regenerated = scan.run(rr.STRESS, int(seed), "A", truth)["steps"]
    prefix = regenerated[:prefix_count]
    prefix_ok = len(regenerated) == prefix_count
    book = EvidenceBook(model, gate="predictive", predictive_z=float(predictive_z))
    belief = _initial_belief(scan)
    for step in prefix:
        belief = book.assimilate(
            belief,
            int(step["pose"]),
            tuple(int(v) for v in step["action"]),
            step.get("observation_mm", []),
            step.get("eval_only_labels", []),
        )

    once_steps, locked_steps = _steps(once_row), _steps(locked_row)
    divergence, diverged = _first_public_divergence(once_row, locked_row)
    result: dict[str, Any] = {
        "seed": int(seed),
        "completed_pair": True,
        "strict_success_both": bool(once_row.get("complete", False) and locked_row.get("complete", False)),
        "prefix_acquisitions": prefix_count,
        "regenerated_prefix_steps": len(prefix),
        "prefix_regeneration_match": bool(prefix_ok and len(prefix) == prefix_count),
        "first_divergence_index": int(divergence),
        "public_diverged": bool(diverged),
        "shared_head_length": int(divergence),
        "shared_head_actions_and_public_centroids": True,
        "suffix_tail_state": False,
    }
    if not prefix_ok or not diverged:
        result["reason"] = "prefix_regeneration_failed" if not prefix_ok else "no_public_divergence"
        return result
    if divergence <= 0 or divergence >= min(len(once_steps), len(locked_steps)):
        result["reason"] = "divergence_has_no_common_first_suffix_head"
        result["suffix_tail_state"] = bool(
            divergence < len(once_steps)
            and bool(once_steps[divergence].get("tail", False))
            and bool(locked_steps[divergence].get("tail", False))
        )
        return result
    if not bool(once_steps[divergence].get("tail", False)):
        result["reason"] = "first_divergence_is_not_a_B_once_tail"
        return result
    if not bool(locked_steps[divergence].get("tail", False)):
        result["reason"] = "locked_step_at_divergence_is_not_a_tail"
        return result

    # Verify every common-head public action/centroid independently of the
    # private prefix.  Only public fields participate in this check.
    result["shared_head_actions_and_public_centroids"] = all(
        _same_public(once_steps[index], locked_steps[index]) for index in range(divergence)
    )
    if not result["shared_head_actions_and_public_centroids"]:
        result["reason"] = "shared_head_public_mismatch"
        return result

    # The first suffix is scheduled after common step divergence-1.  Keep its
    # pre-update belief/book for _plan_suffix, then ingest that common step to
    # obtain the exact post-head state passed to Planner.choose.
    for step in once_steps[: max(0, divergence - 1)]:
        belief = book.assimilate(
            belief,
            int(step["pose"]),
            tuple(int(v) for v in step["action"]),
            step.get("centroids_mm", []),
            step.get("eval_only_labels", []),
        )
    pre_book = book.copy()
    pre_belief = (belief[0].copy(), belief[1].copy())
    counts_before, _, _ready_before, _ = book.status(belief)
    common_first = once_steps[divergence - 1]
    first_pose = int(common_first["pose"])
    first_action = tuple(int(v) for v in common_first["action"])
    head_step = once_steps[divergence]
    head_pose = int(head_step["pose"])
    head_action = tuple(int(v) for v in head_step["action"])
    start = int(np.random.default_rng(np.random.SeedSequence([int(seed), 81])).integers(len(scan.moving)))
    scheduled_pose = int(scan.pose(rr.STRESS, prefix_count + divergence - 1, start))
    expected_head_pose = int(scan.pose(rr.STRESS, prefix_count + divergence, start))
    result["schedule_pose_match"] = bool(scheduled_pose == first_pose and expected_head_pose == head_pose)
    result["suffix_tail_state"] = bool(
        once_steps[divergence].get("tail", False)
        and locked_steps[divergence].get("tail", False)
    )
    result["common_first"] = {"pose": first_pose, "action": list(first_action)}
    result["head_exit"] = {"pose": first_pose, "action": list(first_action)}
    result["suffix_pose"] = head_pose
    head_distance_before_first = float(
        sum(float(step.get("length_m", 0.0)) for step in once_steps[: max(0, divergence - 1)])
    )
    result["head_distance_m_before_common_first"] = head_distance_before_first
    belief = book.assimilate(
        belief, first_pose, first_action, common_first.get("centroids_mm", []),
        common_first.get("eval_only_labels", []),
    )
    counts_after, _, _ready_after, _ = book.status(belief)
    head_book = book
    head_belief = (belief[0].copy(), belief[1].copy())
    previous = (first_pose, *first_action)

    # Use the original planner's support at exactly the state where it created
    # the scheduled suffix.  This call is deliberately retained even though
    # the score table below is recomputed with explicit public present/usable
    # classes.
    arm_once = rr.ARMS["B_once"]
    arm_locked = rr.ARMS["locked"]
    first_index = _action_index(planner, first_pose, first_action)
    if first_index is None:
        result["reason"] = "common_first_action_not_in_planner_bank"
        return result
    distance_at_head = head_distance_before_first + float(common_first.get("length_m", 0.0))
    program_once, expectation_once = planner._plan_suffix(
        pre_book,
        pre_belief,
        counts_before,
        first_pose,
        first_index,
        head_pose,
        arm_once,
        path_cap - distance_at_head,
    )
    program_locked, expectation_locked = planner._plan_suffix(
        pre_book,
        pre_belief,
        counts_before,
        first_pose,
        first_index,
        head_pose,
        arm_locked,
        path_cap - distance_at_head,
    )
    result["plan_suffix_support"] = bool(program_once is not None and program_locked is not None)
    if program_once is None or program_locked is None:
        result["reason"] = "no_planner_suffix_support"
        return result

    # Validate the actual logged choices using the exact post-head state and
    # the original Planner.choose rules.
    once_choice = planner.choose(
        program_once,
        head_pose,
        previous,
        head_belief,
        counts_before,
        counts_after,
        arm_once,
        False,
    )
    locked_choice = planner.choose(
        program_locked,
        head_pose,
        previous,
        head_belief,
        counts_before,
        counts_after,
        arm_locked,
        False,
    )
    once_choice_action = _tail_choice_action(planner, head_pose, once_choice)
    locked_choice_action = _tail_choice_action(planner, head_pose, locked_choice)
    logged_locked = [int(v) for v in locked_steps[divergence].get("action", [])]
    logged_once = [int(v) for v in once_steps[divergence].get("action", [])]
    logged_tail_choice = once_steps[divergence].get("tail_choice") or {}
    result["choice_checks"] = {
        "planner_B_once_matches_logged": once_choice_action == logged_once,
        "planner_locked_matches_logged": locked_choice_action == logged_locked,
        "logged_tail_choice_locked_matches_locked_row": logged_tail_choice.get("locked") == logged_locked,
        "logged_B_once_action": logged_once,
        "logged_locked_action": logged_locked,
        "planner_B_once_action": once_choice_action,
        "planner_locked_action": locked_choice_action,
        "logged_tail_choice": logged_tail_choice,
        "post_head_usable": bool(np.any(np.asarray(counts_after) > np.asarray(counts_before))),
    }

    # Give the control both public flags, including their actual values.
    model_terms = model.model_cache[(first_pose, *first_action)]
    samples = predictive_sets(
        pre_belief,
        model_terms,
        half_width=rr.WIDTH,
        p_detect=rr.PD,
        noise_mixture=rr.STRESS.noise_mixture,
        clutter_rate=rr.STRESS.clutter_rate,
        count=int(planner.n_predictive),
    )
    options = np.asarray(program_locked.options, dtype=int)
    outcome_rows: list[dict[str, Any]] = []
    score_rows: list[np.ndarray] = []
    for values, weight in samples:
        outcome_book = pre_book.copy()
        outcome_belief = outcome_book.assimilate(pre_belief, first_pose, first_action, values)
        outcome_counts, _, outcome_ready, _ = outcome_book.status(outcome_belief)
        progress = bool(np.any(np.asarray(outcome_counts) > np.asarray(counts_before)))
        present = bool(values)
        if bool(np.all(outcome_ready)):
            scores = np.zeros(len(options), dtype=float)
        else:
            scores = np.asarray(
                planner.score(
                    head_pose,
                    previous,
                    outcome_belief,
                    outcome_counts,
                    arm_locked,
                    options,
                    exact=True,
                ),
                dtype=float,
            )
        score_rows.append(scores)
        outcome_rows.append(
            {
                "values": list(values),
                "weight": float(weight),
                "observations_present": present,
                "progress_only": progress,
                "ready": bool(np.all(outcome_ready)),
                "after_counts": np.asarray(outcome_counts, dtype=float).tolist(),
            }
        )
    scores_table = np.asarray(score_rows, dtype=float)
    weights = np.asarray([row["weight"] for row in outcome_rows], dtype=float)
    progress_mask = np.asarray([row["progress_only"] for row in outcome_rows], dtype=bool)
    present_mask = np.asarray([row["observations_present"] for row in outcome_rows], dtype=bool)
    actual_present = bool(common_first.get("centroids_mm", []))
    actual_progress = bool(np.any(np.asarray(counts_after) > np.asarray(counts_before)))

    def _best(mask: np.ndarray) -> tuple[int | None, float]:
        mass = float(weights[mask].sum()) if np.any(mask) else 0.0
        if mass <= 0.0 or len(options) == 0:
            return None, mass
        expected = (weights[mask, None] * scores_table[mask]).sum(axis=0) / mass
        return int(options[int(np.argmax(expected))]), mass

    actual_group = (present_mask == actual_present) & (progress_mask == actual_progress)
    strong_index, strong_mass = _best(actual_group)
    fallback_mass = float(weights.sum())
    if strong_index is None:
        strong_index = program_locked.locked[actual_progress]
        strong_source = "progress_only_fallback_no_joint_class_mass"
    else:
        strong_source = "matched_actual_present_and_progress_flags"
    coarse_actions: dict[str, list[int] | None] = {}
    coarse_mass: dict[str, float] = {}
    for present, progress in ((False, False), (False, True), (True, False), (True, True)):
        index, mass = _best((present_mask == present) & (progress_mask == progress))
        key = f"present={int(present)},progress={int(progress)}"
        coarse_mass[key] = float(mass)
        coarse_actions[key] = None if index is None else _as_action(planner.actions[head_pose], index)
    logged_locked_tuple = tuple(logged_locked)
    logged_once_tuple = tuple(logged_once)
    strong_action = None if strong_index is None else _as_action(planner.actions[head_pose], strong_index)
    coarse_changes = sum(
        int(action is not None and tuple(action) != logged_locked_tuple)
        for action in coarse_actions.values()
    )
    result["predicted_outcomes"] = outcome_rows
    result["predicted_score_support"] = {
        "candidate_count": int(len(options)),
        "candidate_actions": [_as_action(planner.actions[head_pose], int(index)) for index in options],
        "planner_suffix_expectation_B_once": float(expectation_once),
        "planner_suffix_expectation_locked": float(expectation_locked),
        "strong_mass": float(strong_mass),
        "all_outcome_mass": float(fallback_mass),
        "strong_tail_source": strong_source,
        "actual_public_flags": [actual_present, actual_progress],
        "strong_fixed_tail": strong_action,
        "coarse_class_alternate_tail": coarse_actions,
        "coarse_class_masses": coarse_mass,
        "coarse_class_changes_survive": int(coarse_changes),
        "strong_differs_from_old_locked": bool(strong_action is not None and tuple(strong_action) != logged_locked_tuple),
        "strong_differs_from_actual_B_once": bool(strong_action is not None and tuple(strong_action) != logged_once_tuple),
    }
    result["actual_trace"] = {
        "B_once_completed": bool(once_row.get("complete", False)),
        "locked_completed": bool(locked_row.get("complete", False)),
        "B_once_capture_count": int(once_row.get("prefix_acquisitions", 0)) + len(once_steps),
        "locked_capture_count": int(locked_row.get("prefix_acquisitions", 0)) + len(locked_steps),
        "B_once_recheck_capture_count": len(once_steps),
        "locked_recheck_capture_count": len(locked_steps),
        "B_once_recheck_travel_m": float(once_row.get("recheck_path_mm", 0.0)) / 1000.0,
        "locked_recheck_travel_m": float(locked_row.get("recheck_path_mm", 0.0)) / 1000.0,
        "B_once_tail_changes": int(sum(bool(step.get("changed_suffix", False)) for step in once_steps)),
        "locked_tail_changes": int(sum(bool(step.get("changed_suffix", False)) for step in locked_steps)),
    }

    # The pre-counterfactual report is intentionally emitted before any forced
    # simulator continuation is started.
    print(
        json.dumps(
            {
                "event": "pre_counterfactual",
                "seed": int(seed),
                "coarse_class_alternate_tail": coarse_actions,
                "coarse_class_changes_survive": int(coarse_changes),
                "strong_fixed_tail": strong_action,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    if strong_action is None:
        result["counterfactual"] = {"ran": False, "reason": "no_strong_tail"}
        return result
    strong_tuple = tuple(strong_action)
    forced_index = _action_index(planner, head_pose, strong_action)
    public_eligible = bool(
        forced_index is not None
        and planner._within_budget(
            previous,
            head_pose,
            int(forced_index),
            path_cap - distance_at_head,
            rr.ARMS["locked"],
        )
    )
    if not public_eligible:
        result["counterfactual"] = {"ran": False, "reason": "strong_tail_not_public_eligible"}
        return result
    if strong_tuple == logged_locked_tuple:
        result["counterfactual"] = {
            "ran": False,
            "reused_arm": "locked",
            "reason": "strong_tail_equals_old_locked",
            "completion": bool(locked_row.get("complete", False)),
            "capture_count": int(locked_row.get("prefix_acquisitions", 0)) + len(locked_steps),
            "recheck_travel_m": float(locked_row.get("recheck_path_mm", 0.0)) / 1000.0,
        }
        return result
    if strong_tuple == logged_once_tuple:
        result["counterfactual"] = {
            "ran": False,
            "reused_arm": "B_once",
            "reason": "strong_tail_equals_actual_B_once",
            "completion": bool(once_row.get("complete", False)),
            "capture_count": int(once_row.get("prefix_acquisitions", 0)) + len(once_steps),
            "recheck_travel_m": float(once_row.get("recheck_path_mm", 0.0)) / 1000.0,
        }
        return result

    result["counterfactual"] = _run_forced_tail(
        scan=scan,
        model=model,
        planner=planner,
        seed=int(seed),
        truth=truth,
        start=start,
        prefix_count=prefix_count,
        divergence=divergence,
        path_cap=path_cap - distance_at_head,
        head_book=head_book,
        head_belief=head_belief,
        head_counts=counts_after,
        previous=previous,
        program=program_locked,
        forced_pose=head_pose,
        forced_index=int(forced_index),
        predictive_z=float(predictive_z),
        cap=int(cap),
    )
    result["counterfactual"]["total_recheck_travel_m"] = (
        distance_at_head + result["counterfactual"]["recheck_travel_m"]
    )
    return result


class _ForcedChoicePlanner(rr.Planner):
    """Force exactly one scheduled tail, then use the normal locked policy."""

    def __init__(self, *args, forced_pose: int, forced_index: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_pose = int(forced_pose)
        self._forced_index = int(forced_index)
        self._forced_used = False

    def choose(self, program, pose, previous, belief, before, after, arm, switched):
        if not self._forced_used and int(pose) == self._forced_pose:
            self._forced_used = True
            usable = bool(np.any(np.asarray(after) > np.asarray(before)))
            return rr.Choice(
                self._forced_index,
                self._forced_index,
                usable,
                0.0,
                float(getattr(program, "predicted_gain", 0.0)),
            )
        return super().choose(program, pose, previous, belief, before, after, arm, switched)


def _run_forced_tail(
    *,
    scan,
    model,
    planner: rr.Planner,
    seed: int,
    truth: np.ndarray,
    start: int,
    prefix_count: int,
    divergence: int,
    path_cap: float,
    head_book: EvidenceBook,
    head_belief: tuple[np.ndarray, np.ndarray],
    head_counts: np.ndarray,
    previous: tuple[int, ...],
    program,
    forced_pose: int,
    forced_index: int,
    predictive_z: float,
    cap: int,
) -> dict[str, Any]:
    """Execute one forced public tail, then normal ``locked`` continuation."""

    forced = _ForcedChoicePlanner(
        model,
        travel_weight=planner.travel_weight,
        n_predictive=planner.n_predictive,
        n_first=planner.n_first,
        switch_margin=planner.switch_margin,
        predictive_z=predictive_z,
        forced_pose=forced_pose,
        forced_index=forced_index,
    )
    # The override is triggered at the exact post-head belief/exit, just as
    # the original run invoked choose after the common first action.
    scheduled = forced.choose(
        program,
        forced_pose,
        previous,
        head_belief,
        head_counts,
        head_counts,
        rr.ARMS["locked"],
        False,
    )
    book = head_book.copy()
    belief = (head_belief[0].copy(), head_belief[1].copy())
    counts = np.asarray(head_counts, dtype=float).copy()
    _, _, ready, _ = book.status(belief)
    steps: list[dict[str, Any]] = []
    distance = 0.0
    # Distance before the common head is not included in the supplied state;
    # this audit reports the forced continuation travel only.
    previous_state = previous
    switched = False
    forced_once = False
    for k in range(divergence, int(cap)):
        if bool(np.all(ready)):
            break
        pose = int(scan.pose(rr.STRESS, prefix_count + k, start))
        next_pose = int(scan.pose(rr.STRESS, prefix_count + k + 1, start))
        remaining = path_cap - distance
        plan = None
        tail = scheduled
        scheduled = None
        if tail is not None:
            index = int(tail.index)
            action = tuple(int(v) for v in forced.actions[pose][index])
            is_tail = True
            tail_choice = {
                "locked": _as_action(forced.actions[pose], int(tail.locked_index)),
                "usable": bool(tail.usable),
                "gain": float(tail.gain),
                "predicted_gain": float(tail.predicted_gain),
            }
            forced_once = True
        else:
            index, plan = forced.plan(
                book,
                pose,
                next_pose,
                previous_state,
                belief,
                counts,
                rr.ARMS["locked"],
                remaining,
                k + 1 == int(cap),
            )
            if index is None:
                break
            index = int(index)
            action = tuple(int(v) for v in forced.actions[pose][index])
            is_tail = False
            tail_choice = None
        if not model.edge.allow(previous_state, (pose, *action)):
            break
        motion = forced.motion(
            previous_state,
            pose,
            action,
            rr.ARMS["locked"].learned_primitive,
        )
        if motion.length_m > remaining + 1.0e-10:
            break
        before = counts.copy()
        values, labels = rr.observe(model, pose, action, truth, seed, k)
        belief = book.assimilate(belief, pose, action, values, labels)
        counts, _, ready, _ = book.status(belief)
        changed = bool(is_tail and action != tuple(tail_choice["locked"])) if tail_choice else False
        steps.append(
            {
                "pose": pose,
                "action": list(action),
                "tail": is_tail,
                "centroids_mm": list(values),
                "length_m": float(motion.length_m),
                "cumulative_m": float(distance + motion.length_m),
                "changed_suffix": changed,
                "evidence_gains": np.flatnonzero(np.asarray(counts) > np.asarray(before)).tolist(),
                "tail_choice": tail_choice,
            }
        )
        distance += float(motion.length_m)
        previous_state = (pose, *action)
        if plan is not None:
            scheduled = forced.choose(
                plan,
                next_pose,
                previous_state,
                belief,
                before,
                counts,
                rr.ARMS["locked"],
                switched,
            )
        switched = switched or changed
    _, _, ready, inliers = book.status(belief)
    wrong = [i for i, events in enumerate(inliers) if len(events) >= rr.VERIFY
             and not all(correct for _, correct in events[-rr.VERIFY:])]
    maximum_error = float(np.max(np.abs(model.query @ (belief[0] - truth))))
    return {
        "ran": True,
        "forced_choice_override": bool(forced_once),
        "public_eligible": True,
        "arm_after_forced_tail": "locked",
        "online_stop": bool(np.all(ready)),
        "completion": bool(np.all(ready) and not wrong and maximum_error <= rr.BAND),
        "incorrect_witnesses": wrong,
        "map_max_error_mm": maximum_error,
        "recheck_capture_count": len(steps),
        "capture_count_including_prefix_and_head": int(prefix_count + divergence + len(steps)),
        "recheck_travel_m": float(distance),
        "steps": steps,
        "common_head_preserved": True,
    }


def audit_forks(
    runs: Sequence[Path],
    *,
    max_cases: int | None = None,
    predictive_count: int | None = None,
) -> dict[str, Any]:
    """Run the targeted completed-pair fork audit and return JSON-ready data."""

    cases, configs = _collect(runs)
    scan, support, model, hashes = _build_frozen()
    input_hashes: dict[str, str] = {}
    for path in runs:
        path = Path(path)
        runs_path = path if path.name == "RUNS.jsonl" else path / "RUNS.jsonl"
        if runs_path.is_file():
            input_hashes[str(runs_path)] = _sha256(runs_path)
        config_path = runs_path.with_name("CONFIG.json")
        if config_path.is_file():
            input_hashes[str(config_path)] = _sha256(config_path)
    count = int(predictive_count or _config_value(configs, "n_predictive", 8))
    count = max(1, count)
    predictive_z = float(_config_value(configs, "predictive_z", 2.0))
    configured_path_cap = _config_value(configs, "path_cap", None)
    path_cap = float("inf") if configured_path_cap is None else float(configured_path_cap)
    cap = int(_config_value(configs, "cap", 80))
    planner = rr.Planner(
        model,
        travel_weight=float(_config_value(configs, "travel_weight", 1.4)),
        n_predictive=count,
        n_first=int(_config_value(configs, "n_first", 3)),
        switch_margin=float(_config_value(configs, "switch_margin", 0.006)),
        predictive_z=predictive_z,
    )
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    candidates = []
    for seed, arms in sorted(cases.items(), key=lambda item: int(item[0])):
        missing = [arm for arm in _PAIR if arm not in arms]
        if missing:
            skipped.append({"seed": int(seed), "reason": "missing_arm", "missing": missing})
            continue
        once_row, locked_row = arms["B_once"], arms["locked"]
        divergence, diverged = _first_public_divergence(once_row, locked_row)
        if diverged:
            candidates.append((int(seed), once_row, locked_row))
    if max_cases is not None:
        candidates = candidates[: max(0, int(max_cases))]
    for seed, once_row, locked_row in candidates:
        records.append(
            _reconstruct_head(
                scan,
                support,
                model,
                planner,
                seed,
                once_row,
                locked_row,
                predictive_z,
                path_cap,
                cap,
            )
        )
    positive_seeds = [
        int(record["seed"])
        for record in records
        if record.get("shared_head_actions_and_public_centroids")
        and record.get("choice_checks", {}).get("planner_B_once_matches_logged")
        and record.get("choice_checks", {}).get("planner_locked_matches_logged")
        and record.get("choice_checks", {}).get("logged_tail_choice_locked_matches_locked_row")
    ]
    strong_both = [
        int(record["seed"])
        for record in records
        if record.get("predicted_score_support", {}).get("strong_differs_from_old_locked")
        and record.get("predicted_score_support", {}).get("strong_differs_from_actual_B_once")
    ]
    actual_captures = [record["actual_trace"]["B_once_capture_count"] for record in records]
    locked_captures = [record["actual_trace"]["locked_capture_count"] for record in records]
    actual_travel = [record["actual_trace"]["B_once_recheck_travel_m"] for record in records]
    locked_travel = [record["actual_trace"]["locked_recheck_travel_m"] for record in records]
    return {
        "schema": "ab_b_refinement_fork_audit_v1",
        "scope": "Targeted audit of actual completed B_once-vs-locked switch states with a common public head; not a new rollout-wide comparison.",
        "arms": list(_PAIR),
        "runs": [str(Path(path)) for path in runs],
        "predictive_count": count,
        "predictive_z": predictive_z,
        "cap": cap,
        "path_cap_m": path_cap if math.isfinite(path_cap) else None,
        "candidate_completed_divergent_pairs": len(candidates),
        "audited_records": len(records),
        "skipped": skipped,
        "records": records,
        "source_sha256": hashes,
        "input_sha256": input_hashes,
        "positive_check_seeds": positive_seeds,
        "strong_tail_differs_from_both_seeds": strong_both,
        "aggregates": {
            "completed_divergent_records": len(records),
            "positive_check_records": len(positive_seeds),
            "strict_success_both_records": int(sum(record.get("strict_success_both", False) for record in records)),
            "strong_tail_differs_from_both_count": len(strong_both),
            "mean_B_once_capture_count": float(np.mean(actual_captures)) if actual_captures else None,
            "mean_locked_capture_count": float(np.mean(locked_captures)) if locked_captures else None,
            "mean_B_once_recheck_travel_m": float(np.mean(actual_travel)) if actual_travel else None,
            "mean_locked_recheck_travel_m": float(np.mean(locked_travel)) if locked_travel else None,
            "coarse_class_changes_survive_total": int(sum(
                record.get("predicted_score_support", {}).get("coarse_class_changes_survive", 0)
                for record in records
            )),
        },
        "checks": {
            "all_shared_heads_public_equal": bool(all(record.get("shared_head_actions_and_public_centroids", False) for record in records)) if records else None,
            "all_logged_locked_choices_match": bool(all(record.get("choice_checks", {}).get("logged_tail_choice_locked_matches_locked_row", False) for record in records if record.get("choice_checks"))) if records else None,
            "counterfactuals_run": int(sum(bool(record.get("counterfactual", {}).get("ran", False)) for record in records)),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="paired shard directories or RUNS.jsonl paths")
    parser.add_argument("--out", type=Path, required=True, help="JSON output path, or directory receiving AUDIT_FORKS.json")
    parser.add_argument("--max-cases", type=int, default=None, help="audit only the first completed divergent pairs")
    parser.add_argument("--predictive-count", type=int, default=None, help="override the deterministic predictive support count")
    args = parser.parse_args(argv)
    result = audit_forks(args.runs, max_cases=args.max_cases, predictive_count=args.predictive_count)
    output = args.out
    if output.suffix.lower() != ".json":
        output.mkdir(parents=True, exist_ok=True)
        output = output / "AUDIT_FORKS.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(_jsonable(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

