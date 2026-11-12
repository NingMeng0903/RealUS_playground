"""Audit the logged six-root replay in the B reference development cohort.

This is deliberately a read-only analysis of ``development_b_reverse``.  The
challenge runner logged ``evaluator_root_audit`` only on the MPC row, but that
record contains the result of executing every legal first root and then using
the same actual tail replanner.  The selected root in each other arm is joined
back to that record; no planner, evaluator, or simulator is re-run here.

The report separates three quantities that are easy to conflate:

* the strict result of the root selected online;
* the after-the-fact upper bound obtained by allowing any of the six executed
  roots (the hindsight cap); and
* the model's forecast rank for the selected root versus the strict result
  actually obtained by that root.

The input is a development cohort, not an independent confirmation cohort.
All roots for one case share the prefix, stress realization, and the
view-keyed deterministic ghost jitter used by ``clutter_model.render_observation``.
Consequently ``any_first_success`` is diagnostic headroom, not a deployable
oracle policy estimate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


ARMS = ("MPC", "B_feedback", "B_challenge", "Ref_fixed", "Near_fixed")
ADAPTIVE_ARMS = ("MPC", "B_feedback", "B_challenge")
COST_KEYS = ("travel_mm", "turn_deg", "elapsed_s", "stroke_mm")
WRONG_CAP = 0.02
ROOT_COUNT = 6


def _finite_mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return None if not values else float(mean(values))


def _finite_sd(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return None if len(values) < 2 else float(pstdev(values))


def _rates(count: int, n: int) -> dict[str, Any]:
    return {"count": int(count), "rate": None if n == 0 else float(count / n)}


def _wilson(count: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    """A compact descriptive interval; not used for promotion decisions."""

    if n <= 0:
        return None
    p = count / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / den
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def _binomial_tail(k: int, n: int) -> float:
    """Two-sided sign-test p-value for wins among non-ties."""

    if n <= 0:
        return 1.0
    lo = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2.0**n
    return float(min(1.0, 2.0 * lo))


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _root_label(root: dict[str, Any]) -> dict[str, Any]:
    first = [int(v) for v in root["first"]]
    return {"first": first, "site": first[0], "view": first[1]}


def _choice_from_forecast(forecast: list[dict[str, Any]], arm: str) -> int:
    """Reproduce the frozen choice rule in challenge.py from logged scores."""

    if arm == "MPC":
        return max(range(len(forecast)), key=lambda j: float(forecast[j]["fixed"]))
    if arm == "B_feedback":
        return max(range(len(forecast)),
                   key=lambda j: min(float(v) for v in forecast[j]["feedback"]))
    if arm == "B_challenge":
        eligible = [
            j for j, row in enumerate(forecast)
            if float(row["paired_upper_wrong"]) <= WRONG_CAP
            and float(row["paired_lower_gain"]) > 0.0
        ]
        baseline = max(range(len(forecast)), key=lambda j: float(forecast[j]["fixed"]))
        return max(eligible, key=lambda j: float(forecast[j]["paired_lower_gain"])) \
            if eligible else baseline
    raise ValueError(f"unknown arm {arm}")


def _decision_score(forecast: list[dict[str, Any]], arm: str, j: int) -> float:
    if arm == "MPC":
        return float(forecast[j]["fixed"])
    if arm == "B_feedback":
        return min(float(v) for v in forecast[j]["feedback"])
    if arm == "B_challenge":
        return float(forecast[j]["paired_lower_gain"])
    raise ValueError(f"no forecast score for {arm}")


def _rank_desc(values: list[float], index: int) -> int:
    order = sorted(range(len(values)), key=lambda j: (-values[j], j))
    return int(order.index(index) + 1)


def _action_relation(mpc_root: tuple[int, int] | None,
                     root: tuple[int, int] | None) -> str:
    if mpc_root is None or root is None:
        return "missing"
    if mpc_root == root:
        return "same"
    if mpc_root[0] == root[0]:
        return "tilt_only"
    return "reference_site_switch"


def _cost_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [[float(v) for v in row["costs"]] for row in rows]
    if not values:
        return {k: {"mean": None, "sd": None} for k in COST_KEYS}
    return {
        key: {"mean": float(mean(column)), "sd": float(pstdev(column))}
        for key, column in zip(COST_KEYS, zip(*values))
    }


def _confidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confidences = [float(row["confidence"]) for row in rows]
    accepted = [row for row in rows if row.get("reason") == "accepted"]
    rejected = [row for row in rows if row.get("reason") != "accepted"]
    wrong = [row for row in rows if row.get("wrong")]
    success = [row for row in rows if row.get("success")]
    return {
        "all_mean": _finite_mean(confidences),
        "accepted_mean": _finite_mean(row["confidence"] for row in accepted),
        "unconfirmed_mean": _finite_mean(row["confidence"] for row in rejected),
        "success_mean": _finite_mean(row["confidence"] for row in success),
        "wrong_mean": _finite_mean(row["confidence"] for row in wrong),
        "all_sd": _finite_sd(confidences),
        "accepted_count": len(accepted),
        "unconfirmed_count": len(rejected),
        "wrong_count": len(wrong),
    }


def _paired_summary(arm_rows: list[dict[str, Any]],
                    mpc_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(arm_rows) != len(mpc_rows):
        raise ValueError("paired rows have different lengths")
    wins = [int(a["success"] and not b["success"]) for a, b in zip(arm_rows, mpc_rows)]
    losses = [int(b["success"] and not a["success"]) for a, b in zip(arm_rows, mpc_rows)]
    wrong_harm = [int(a["wrong"] and not b["wrong"]) for a, b in zip(arm_rows, mpc_rows)]
    wrong_avoided = [int(b["wrong"] and not a["wrong"]) for a, b in zip(arm_rows, mpc_rows)]
    deltas = [
        [float(x) - float(y) for x, y in zip(a["costs"], b["costs"])]
        for a, b in zip(arm_rows, mpc_rows)
    ]
    columns = list(zip(*deltas)) if deltas else [[] for _ in COST_KEYS]
    discordant = sum(wins) + sum(losses)
    return {
        "wins": int(sum(wins)),
        "losses": int(sum(losses)),
        "net_success_count": int(sum(wins) - sum(losses)),
        "success_difference": float((sum(wins) - sum(losses)) / len(arm_rows))
        if arm_rows else None,
        "success_sign_test_p": _binomial_tail(sum(wins), discordant),
        "wrong_harm": int(sum(wrong_harm)),
        "wrong_avoided": int(sum(wrong_avoided)),
        "wrong_difference": float((sum(wrong_harm) - sum(wrong_avoided)) / len(arm_rows))
        if arm_rows else None,
        "cost_differences": {
            key: float(mean(column)) if column else None
            for key, column in zip(COST_KEYS, columns)
        },
        "win_indices": [int(i) for i, value in enumerate(wins) if value],
        "loss_indices": [int(i) for i, value in enumerate(losses) if value],
    }


def _root_audit_summary(mpc_rows: list[dict[str, Any]],
                        arm_rows_by_name: dict[str, list[dict[str, Any]]],
                        roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n = len(mpc_rows)
    output = []
    for j, label in enumerate(roots):
        audit = [row["evaluator_root_audit"][j] for row in mpc_rows]
        success = sum(bool(row["success"]) for row in audit)
        wrong = sum(bool(row["wrong"]) for row in audit)
        reasons = Counter(str(row.get("reason")) for row in audit)
        costs = [row["costs"] for row in audit]
        cost_columns = list(zip(*costs)) if costs else [[] for _ in COST_KEYS]
        selected_by = {
            arm: sum(row.get("first_index") == j for row in cohort)
            for arm, cohort in arm_rows_by_name.items()
        }
        output.append({
            "index": j,
            **label,
            "success": _rates(success, n),
            "wrong": _rates(wrong, n),
            "accepted": _rates(success + wrong, n),
            "unconfirmed": _rates(n - success - wrong, n),
            "reasons": dict(sorted(reasons.items())),
            "mean_costs": {
                key: float(mean(column)) if column else None
                for key, column in zip(COST_KEYS, cost_columns)
            },
            "sd_costs": {
                key: float(pstdev(column)) if len(column) > 1 else None
                for key, column in zip(COST_KEYS, cost_columns)
            },
            "selected_by_arm": selected_by,
        })
    return output


def _forecast_summary(mpc_rows: list[dict[str, Any]],
                      arm_rows: list[dict[str, Any]],
                      arm: str) -> dict[str, Any]:
    selected_ranks = []
    top_success = 0
    top_wrong = 0
    selected_pred_gain = []
    selected_pred_wrong = []
    predicted_positive = 0
    positive_wins = 0
    positive_losses = 0
    actual_wins = 0
    actual_losses = 0
    score_success = []
    score_failure = []
    per_case = []
    for mpc, row in zip(mpc_rows, arm_rows):
        forecast = mpc["forecast"]
        scores = [_decision_score(forecast, arm, j) for j in range(len(forecast))]
        selected = int(row["first_index"])
        chosen = _choice_from_forecast(forecast, arm)
        audit = mpc["evaluator_root_audit"]
        selected_audit = audit[selected]
        rank = _rank_desc(scores, selected)
        selected_ranks.append(rank)
        top_success += int(bool(audit[chosen]["success"]))
        top_wrong += int(bool(audit[chosen]["wrong"]))
        gain = float(forecast[selected]["paired_lower_gain"])
        wrong_delta = float(forecast[selected]["paired_upper_wrong"])
        selected_pred_gain.append(gain)
        selected_pred_wrong.append(wrong_delta)
        predicted_positive += int(gain > 0.0)
        baseline_success = bool(audit[int(mpc["first_index"])]["success"])
        actual_success = bool(selected_audit["success"])
        actual_win = actual_success and not baseline_success
        actual_loss = baseline_success and not actual_success
        positive_wins += int(gain > 0.0 and actual_win)
        positive_losses += int(gain > 0.0 and actual_loss)
        actual_wins += int(actual_win)
        actual_losses += int(actual_loss)
        for j, value in enumerate(scores):
            if audit[j]["success"]:
                score_success.append(value)
            else:
                score_failure.append(value)
        per_case.append({
            "index": int(row["index"]),
            "selected_root": selected,
            "choice_recomputed": chosen,
            "choice_match": bool(chosen == selected),
            "selected_rank": rank,
            "selected_score": float(scores[selected]),
            "paired_lower_gain": gain,
            "paired_upper_wrong": wrong_delta,
            "hindsight_success": actual_success,
            "hindsight_mpc_success": baseline_success,
            "actual_benefit": int(actual_win) - int(actual_loss),
        })
    top_n = len(mpc_rows)
    return {
        "score": {
            "name": {
                "MPC": "fixed",
                "B_feedback": "min(feedback)",
                "B_challenge": "paired_lower_gain with wrong<=0.02 gate; fallback fixed baseline",
            }[arm],
            "selected_mean": _finite_mean(selected_pred_gain if arm != "MPC" else
                                           [float(m["forecast"][int(r["first_index"])]["fixed"])
                                            for m, r in zip(mpc_rows, arm_rows)]),
            "selected_paired_lower_gain_mean": _finite_mean(selected_pred_gain),
            "selected_paired_upper_wrong_mean": _finite_mean(selected_pred_wrong),
            "successful_root_score_mean": _finite_mean(score_success),
            "failed_root_score_mean": _finite_mean(score_failure),
        },
        "choice_recomputed_matches_logged": sum(
            item["choice_match"] for item in per_case
        ),
        "n": top_n,
        "selected_rank_mean": _finite_mean(selected_ranks),
        "selected_rank_histogram": dict(sorted(Counter(selected_ranks).items())),
        "forecast_top_success": _rates(top_success, top_n),
        "forecast_top_wrong": _rates(top_wrong, top_n),
        "predicted_positive_gain": _rates(predicted_positive, top_n),
        "positive_gain_wins": int(positive_wins),
        "positive_gain_losses": int(positive_losses),
        "actual_selected_wins_over_mpc": int(actual_wins),
        "actual_selected_losses_to_mpc": int(actual_losses),
        "per_case": per_case,
    }


def _arm_summary(mpc_rows: list[dict[str, Any]],
                 arm_rows: list[dict[str, Any]],
                 roots: list[dict[str, Any]],
                 arm: str) -> dict[str, Any]:
    mpc_indices = [int(row["first_index"]) if row.get("first_index") is not None else None
                   for row in mpc_rows]
    arm_indices = [int(row["first_index"]) if row.get("first_index") is not None else None
                   for row in arm_rows]
    mpc_goals = [tuple(roots[i]["first"]) if i is not None else None for i in mpc_indices]
    arm_goals = [tuple(roots[i]["first"]) if i is not None else None for i in arm_indices]
    relations = Counter(_action_relation(a, b) for a, b in zip(mpc_goals, arm_goals))
    return {
        "n": len(arm_rows),
        "success": _rates(sum(bool(row["success"]) for row in arm_rows), len(arm_rows)),
        "success_wilson_95": _wilson(sum(bool(row["success"]) for row in arm_rows), len(arm_rows)),
        "wrong": _rates(sum(bool(row["wrong"]) for row in arm_rows), len(arm_rows)),
        "accepted": _rates(sum(row.get("reason") == "accepted" for row in arm_rows), len(arm_rows)),
        "reasons": dict(sorted(Counter(str(row.get("reason")) for row in arm_rows).items())),
        "confidence": _confidence_summary(arm_rows),
        "costs": _cost_summary(arm_rows),
        "first_root_counts": dict(sorted(Counter(arm_indices).items(), key=lambda item: str(item[0]))),
        "action_relation_to_mpc": dict(sorted(relations.items())),
        "paired_vs_mpc": None if arm == "MPC" else _paired_summary(arm_rows, mpc_rows),
        "forecast": None if arm not in ADAPTIVE_ARMS else _forecast_summary(mpc_rows, arm_rows, arm),
    }


def _headroom_cases(mpc_rows: list[dict[str, Any]],
                    arm_rows_by_name: dict[str, list[dict[str, Any]]],
                    roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for i, mpc in enumerate(mpc_rows):
        audit = mpc["evaluator_root_audit"]
        successful = [j for j, root in enumerate(audit) if root["success"]]
        if not successful or mpc["success"]:
            continue
        selected = {}
        for arm, cohort in arm_rows_by_name.items():
            row = cohort[i]
            j = row.get("first_index")
            selected[arm] = {
                "root_index": None if j is None else int(j),
                "first": None if j is None else roots[int(j)]["first"],
                "success": bool(row["success"]),
                "wrong": bool(row["wrong"]),
            }
        cases.append({
            "index": int(mpc["index"]),
            "seed": int(mpc["seed"]),
            "mpc_root_index": int(mpc["first_index"]),
            "mpc_first": roots[int(mpc["first_index"])]["first"],
            "successful_roots": [
                {"index": j, **roots[j]} for j in successful
            ],
            "selected": selected,
        })
    return cases


def _audit_stratum(cells: dict[int, dict[str, dict[str, Any]]],
                   task: str,
                   estimator: str) -> dict[str, Any]:
    indices = sorted(cells)
    mpc_rows = [cells[i]["MPC"] for i in indices]
    audits = [row.get("evaluator_root_audit") for row in mpc_rows]
    if any(not isinstance(audit, list) or len(audit) != ROOT_COUNT for audit in audits):
        raise ValueError(f"{task}/{estimator}: expected six audit roots on every MPC row")
    root_first = [tuple(int(v) for v in item["first"]) for item in audits[0]]
    if any([tuple(int(v) for v in item["first"]) for item in audit] != root_first
           for audit in audits[1:]):
        raise ValueError(f"{task}/{estimator}: audit root ordering changed across cases")
    roots = [_root_label(item) for item in audits[0]]
    arm_rows = {
        arm: [cells[i][arm] for i in indices]
        for arm in ARMS
    }
    selected_mismatch = []
    for i, mpc in enumerate(mpc_rows):
        for arm in ARMS:
            row = arm_rows[arm][i]
            j = row.get("first_index")
            if j is None:
                continue
            audit = mpc["evaluator_root_audit"][int(j)]
            fields = ("success", "wrong", "reason", "costs")
            if any(row.get(field) != audit.get(field) for field in fields):
                selected_mismatch.append({"index": int(mpc["index"]), "arm": arm})
    root_success = [
        sum(bool(mpc["evaluator_root_audit"][j]["success"]) for mpc in mpc_rows)
        for j in range(ROOT_COUNT)
    ]
    baseline = sum(bool(row["success"]) for row in mpc_rows)
    any_success = sum(any(bool(item["success"]) for item in mpc["evaluator_root_audit"])
                      for mpc in mpc_rows)
    headroom = _headroom_cases(mpc_rows, arm_rows, roots)
    return {
        "n": len(indices),
        "case_indices": indices,
        "roots": roots,
        "hindsight_cap": {
            "baseline_mpc_success": _rates(baseline, len(indices)),
            "any_legal_root_success": _rates(any_success, len(indices)),
            "optimistic_headroom_count": int(any_success - baseline),
            "optimistic_headroom_rate": float((any_success - baseline) / len(indices)),
            "max_single_root_success": _rates(max(root_success), len(indices)),
            "max_single_root_index": int(max(range(ROOT_COUNT), key=lambda j: root_success[j])),
            "headroom_case_count": len(headroom),
            "headroom_cases": headroom,
        },
        "root_rates": _root_audit_summary(mpc_rows, arm_rows, roots),
        "arms": {
            arm: _arm_summary(mpc_rows, arm_rows[arm], roots, arm)
            for arm in ARMS
        },
        "checks": {
            "audit_rows": len(audits),
            "audit_root_count_all_six": all(len(audit) == ROOT_COUNT for audit in audits),
            "selected_row_vs_audit_mismatches": selected_mismatch,
            "selected_row_vs_audit_mismatch_count": len(selected_mismatch),
        },
    }


def _cross_model(rows_by_key: dict[tuple[int, str, str, str], dict[str, Any]],
                 tasks: list[str],
                 estimators: list[str]) -> dict[str, Any]:
    result = {}
    if "nominal" not in estimators or "fitted" not in estimators:
        return result
    for task in tasks:
        task_result = {}
        for arm in ADAPTIVE_ARMS:
            nominal = [rows_by_key[(i, task, "nominal", arm)]
                       for i in sorted({key[0] for key in rows_by_key if key[1] == task})]
            fitted = [rows_by_key[(i, task, "fitted", arm)]
                      for i in sorted({key[0] for key in rows_by_key if key[1] == task})]
            relations = Counter()
            action_diffs = []
            for nrow, frow in zip(nominal, fitted):
                na = tuple(nrow.get("first_index") if nrow.get("first_index") is not None else -1 for _ in [0])
                fa = tuple(frow.get("first_index") if frow.get("first_index") is not None else -1 for _ in [0])
                if na == fa:
                    relations["same_index"] += 1
                elif nrow.get("first_index") is None or frow.get("first_index") is None:
                    relations["missing"] += 1
                else:
                    # Root labels are available in the paired MPC audit row of the
                    # nominal cell; the index relation is enough for this cohort
                    # because candidate order is frozen and shared.
                    nidx, fidx = int(nrow["first_index"]), int(frow["first_index"])
                    # Site groups are two roots per site, ordered by view 3,7,8.
                    nsite_group, fsite_group = nidx // 3, fidx // 3
                    relations["reference_site_switch" if nsite_group != fsite_group
                              else "tilt_only"] += 1
                action_diffs.append({
                    "index": int(nrow["index"]),
                    "nominal_first_index": nrow.get("first_index"),
                    "fitted_first_index": frow.get("first_index"),
                    "nominal_success": bool(nrow["success"]),
                    "fitted_success": bool(frow["success"]),
                    "fitted_minus_nominal_success": int(bool(frow["success"]))
                    - int(bool(nrow["success"])),
                })
            task_result[arm] = {
                "n": len(nominal),
                "relations": dict(sorted(relations.items())),
                "fitted_success_minus_nominal_success": sum(
                    item["fitted_minus_nominal_success"] for item in action_diffs
                ),
                "action_diffs": action_diffs,
            }
        result[task] = task_result
    return result


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def run(cases_path: Path, settings_path: Path | None = None) -> dict[str, Any]:
    rows = _load(cases_path)
    if not rows:
        raise ValueError(f"empty cases file: {cases_path}")
    rows_by_key: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["index"]), str(row["task"]), str(row["estimator"]), str(row["arm"]))
        if key in rows_by_key:
            raise ValueError(f"duplicate row {key}")
        rows_by_key[key] = row
    indices = sorted({key[0] for key in rows_by_key})
    tasks = sorted({key[1] for key in rows_by_key})
    estimators = sorted({key[2] for key in rows_by_key})
    arms = sorted({key[3] for key in rows_by_key})
    missing = [
        (i, task, estimator, arm)
        for i in indices for task in tasks for estimator in estimators for arm in ARMS
        if (i, task, estimator, arm) not in rows_by_key
    ]
    if missing:
        raise ValueError(f"missing paired rows, first examples: {missing[:5]}")

    cells = {}
    for task in tasks:
        for estimator in estimators:
            cells[f"{task}/{estimator}"] = _audit_stratum(
                {
                    i: {arm: rows_by_key[(i, task, estimator, arm)] for arm in ARMS}
                    for i in indices
                }, task, estimator)

    settings: dict[str, Any] = {}
    if settings_path is not None and settings_path.is_file():
        settings = json.loads(settings_path.read_text())
    seed_values = sorted({int(row["seed"]) for row in rows})
    view_noise = {
        "renderer": "clutter_model.render_observation",
        "persistent_ghost_jitter_sd_mm": [1.0, 0.68],
        "deterministic_seed_rule": "SeedSequence([case.seed, 77631, tick, site, view])",
        "interpretation": "same case/prefix and view-keyed jitter are shared by paired arms; roots are not independent noise draws",
    }
    result = {
        "schema": "reference_root_audit.v1",
        "source": {
            "cases": str(cases_path.resolve()),
            "settings": None if settings_path is None else str(settings_path.resolve()),
            "settings_snapshot": settings,
            "n_jsonl_rows": len(rows),
            "n_independent_case_indices": len(indices),
            "case_indices": indices,
            "seed_values": seed_values,
            "tasks": tasks,
            "estimators": estimators,
            "arms": arms,
            "split": settings.get("split", "unknown"),
            "seed_base": settings.get("seed_base"),
            "developer_cohort": True,
            "no_relaunch_or_tuning": True,
        },
        "audit_design": {
            "audited_arm": "MPC row contains six evaluator_root_audit entries",
            "first_root_count": ROOT_COUNT,
            "tail": "each root was executed by the logged runner and then selected a shared actual tail replanner",
            "identity_gate": "strict evaluated final identity and completion gate in the logged result; success excludes wrong/unconfirmed",
            "hindsight_cap": "any strict success among the six executed roots, selected after observing evaluator outcomes",
            "view_keyed_noise": view_noise,
            "limitations": [
                "The root audit stores strict success/wrong/reason/cost, not unselected-root confidence or raw final labels.",
                "The six-root maximum is an after-the-fact diagnostic ceiling, not an online policy and not a causal B estimate.",
                "The cohort is the development split (32 shared case draws), not an independent confirmation set.",
                "The benchmark executor and acoustic renderer are simulated; this report is not a native-robot or moving-phantom validation.",
            ],
        },
        "cells": cells,
        "cross_model_nominal_vs_fitted": _cross_model(rows_by_key, tasks, estimators),
        "global_checks": {
            "all_mpc_rows_have_six_root_audits": all(
                isinstance(rows_by_key[(i, task, estimator, "MPC")].get("evaluator_root_audit"), list)
                and len(rows_by_key[(i, task, estimator, "MPC")]["evaluator_root_audit"]) == ROOT_COUNT
                for i in indices for task in tasks for estimator in estimators
            ),
            "root_audit_rows": sum(
                bool(rows_by_key[(i, task, estimator, "MPC")].get("evaluator_root_audit"))
                for i in indices for task in tasks for estimator in estimators
            ),
            "selected_results_match_audit": all(
                not cell["checks"]["selected_row_vs_audit_mismatches"]
                for cell in cells.values()
            ),
            "forecast_choice_recomputed_matches_logged": {
                key: {
                    arm: value["arms"][arm]["forecast"]["choice_recomputed_matches_logged"]
                    for arm in ADAPTIVE_ARMS
                }
                for key, value in cells.items()
            },
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("development_b_reverse") / "cases.jsonl",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(__file__).with_name("development_b_reverse") / "settings.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("reference_root_audit.json"),
    )
    args = parser.parse_args()
    report = run(args.cases, args.settings)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "cells": sorted(report["cells"]),
        "selected_results_match_audit": report["global_checks"]["selected_results_match_audit"],
    }, indent=2))


if __name__ == "__main__":
    main()
