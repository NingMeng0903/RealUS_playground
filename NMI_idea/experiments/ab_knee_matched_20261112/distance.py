"""Distance-matched supplementary knee recheck experiment.

This is a fresh driver around the frozen experiment in
``../ab_knee_latent_rl_20261112``.  Frozen world, belief, policy, learner and
outcome code are imported read-only.  No Fisher score is used and no learner is
retrained.  The normal interface is::

    distance.py dev --out distance
    distance.py test --out distance --test-start 256 --test-count 128
    distance.py summary --out distance

The dev pass fixes an outcome-blind CQL distance-price interpolation.  The
held-out pass shares the full rule first sweep and applies one cumulative path
cap to both methods.  The same cap is also applied at both levels of the
rule's two-level lookahead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "ab_knee_latent_rl_20261112"
if not PREVIOUS.exists():
    raise FileNotFoundError(f"frozen knee experiment not found: {PREVIOUS}")
sys.path.insert(0, str(PREVIOUS))

from belief import Filter  # noqa: E402
from experiment import (  # noqa: E402
    BUDGET, BASES, HORIZONS, QUERY_DELAY, State, apply, evaluate_outcome,
    prepared, prefix_hash, seed_at,
)
from knee_world import GEOMETRY, World  # noqa: E402
from policy import RULE_DEPTH, imagined_frames, leaf_value, proposal_features  # noqa: E402
import learners  # noqa: E402


# 80 mm preserves the public first-step 44--61 mm candidate family and most
# one-follow-up continuations while still bounding three supplementary actions.
DEFAULT_CAP_MM = 80.0
DEFAULT_DEV_COUNT = 64
DEFAULT_TEST_START = 256
DEFAULT_TEST_COUNT = 128
DEFAULT_BOOTSTRAPS = 4000
DEFAULT_MODEL = PREVIOUS / "results" / "cql_512_7.pt"
DEFAULT_LAMBDAS = (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.012,
                   0.024, 0.048)
PRICE_SEED = 190417
MIX_SEED_SALT = 910031


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def lambda_key(value: float) -> str:
    return f"cql_lambda_{float(value):.8f}"


def action_distance(action: Any) -> float:
    return float(action.distance_mm) if action is not None else math.inf


def cap_mask(actions: Iterable[Any], used_mm: float, cap_mm: float) -> np.ndarray:
    """Apply the common cumulative path cap to already edge-legal actions."""
    distances = np.asarray([action_distance(a) for a in actions], dtype=float)
    # 1e-9 only absorbs accumulation at exactly the cap.
    return np.isfinite(distances) & (float(used_mm) + distances <= float(cap_mm) + 1.e-9)


def candidate_envelope(world: World, prefix: State, cap_mm: float) -> dict[str, Any]:
    actions = world.candidates(prefix.belief.reading, prefix.previous)
    distances = [action_distance(a) for a in actions if a is not None]
    legal = cap_mask(actions, 0., cap_mm)
    return {
        "initial_action_count": int(len(actions)),
        "initial_legal_count": int(legal.sum()),
        "initial_min_nonzero_distance_mm": float(min((d for d in distances if d > 0.), default=0.)),
        "initial_max_distance_mm": float(max(distances, default=0.)),
        "cap_mm": float(cap_mm),
    }


def capped_rule_scores(world: World, belief: Filter, previous: Any,
                       actions: tuple[Any, ...], remaining: int,
                       used_mm: float, cap_mm: float, *, depth: int = RULE_DEPTH) -> np.ndarray:
    """Frozen rule score with the cap masked at both lookahead levels."""
    scores = np.full(len(actions), -np.inf, dtype=float)
    legal = cap_mask(actions, used_mm, cap_mm)
    from policy import travel_cost
    for slot, action in enumerate(actions):
        if not legal[slot]:
            continue
        outcomes = []
        draws = 5 if depth == RULE_DEPTH else 3
        for reading, obs in imagined_frames(world, belief, action, draws, depth=depth):
            updated = belief.copy()
            updated.observe(world, reading, action.site, action.view, obs,
                            event=-remaining, retain=world.identity_site(action.site))
            if depth > 1 and remaining > 1:
                tail = world.candidates(reading, action)
                values = capped_rule_scores(
                    world, updated, action, tail, remaining - 1,
                    used_mm + float(action.distance_mm), cap_mm, depth=depth - 1,
                )
                outcomes.append(float(np.max(values)) if np.any(np.isfinite(values))
                                else leaf_value(world, updated))
            else:
                outcomes.append(leaf_value(world, updated))
        # Keep original travel_cost; the cap is a feasibility constraint only.
        scores[slot] = float(np.mean(outcomes)) - travel_cost(action)
    return scores


def distance_masked_proposals(world: World, state: State, remaining: int,
                              used_mm: float, cap_mm: float):
    actions = world.candidates(state.belief.reading, state.previous)
    features, model_mask = proposal_features(world, state.belief, actions,
                                             remaining, BUDGET)
    mask = np.asarray(model_mask, dtype=bool) & cap_mask(actions, used_mm, cap_mm)
    if not np.any(mask):
        raise RuntimeError("no legal action remains under cumulative distance cap")
    return actions, features, mask


def run_rule(world: World, case: Any, prefix: State, cap_mm: float,
              *, trace: bool = False) -> dict[str, Any]:
    state = prefix
    actions = []
    used_mm = 0.0
    for step in range(BUDGET):
        remaining = BUDGET - step
        candidates = world.candidates(state.belief.reading, state.previous)
        mask = cap_mask(candidates, used_mm, cap_mm)
        if not np.any(mask):
            raise RuntimeError("rule has no legal action under cumulative cap")
        values = capped_rule_scores(
            world, state.belief, state.previous, candidates, remaining,
            used_mm, cap_mm, depth=RULE_DEPTH,
        )
        values = np.where(mask, values, -np.inf)
        slot = int(np.argmax(values))
        action = candidates[slot]
        if action is None:
            raise RuntimeError("rule selected an illegal candidate")
        state = apply(world, case, state, action)
        actions.append(action)
        used_mm += float(action.distance_mm)
    row = evaluate_outcome(world, case, prefix, state, actions, trace=trace)
    row.update({
        "policy": "rule",
        "distance_cap_mm": float(cap_mm),
        "cap_satisfied": bool(row["travel_mm"] <= cap_mm + 1.e-8),
    })
    return row


def run_cql(world: World, case: Any, prefix: State, model: Any,
            lambda_mm: float, cap_mm: float, *, trace: bool = False) -> dict[str, Any]:
    state = prefix
    actions = []
    used_mm = 0.0
    for step in range(BUDGET):
        remaining = BUDGET - step
        candidates, features, mask = distance_masked_proposals(
            world, state, remaining, used_mm, cap_mm,
        )
        scores = learners.scores(model, features[None], mask[None])[0]
        prices = np.asarray([action_distance(a) for a in candidates], dtype=float)
        # Avoid inf*0 warnings for edge-illegal slots; they remain masked.
        prices = np.where(np.isfinite(prices), prices, 0.)
        scores = np.where(mask, scores - float(lambda_mm) * prices, -np.inf)
        slot = int(np.argmax(scores))
        action = candidates[slot]
        if action is None or not mask[slot]:
            raise RuntimeError("CQL selected an illegal candidate")
        state = apply(world, case, state, action)
        actions.append(action)
        used_mm += float(action.distance_mm)
    row = evaluate_outcome(world, case, prefix, state, actions, trace=trace)
    row.update({
        "policy": "cql_price",
        "lambda_distance": float(lambda_mm),
        "distance_cap_mm": float(cap_mm),
        "cap_satisfied": bool(row["travel_mm"] <= cap_mm + 1.e-8),
    })
    return row


def case_prefix(world: World, split: str, index: int):
    seed = seed_at(split, index)
    case = world.case(seed)
    prefix = prepared(world, case)
    return seed, case, prefix


def nested_value(row: Mapping[str, Any], path: tuple[Any, ...]) -> float:
    value: Any = row
    for key in path:
        value = value[key]
    return float(value)


def mean_nested(rows: list[dict[str, Any]], path: tuple[Any, ...]) -> float:
    return float(np.mean([nested_value(row, path) for row in rows]))


def row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Raw arm summary retaining requested identity, geometry and timing fields."""
    if not rows:
        raise ValueError("cannot summarize empty arm")

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    def pooled(path: tuple[Any, ...]) -> float:
        values = np.asarray([nested_value(row, path) for row in rows], dtype=float)
        return float(np.sqrt(np.mean(np.square(values))))

    def count(predicate) -> int:
        return int(sum(bool(predicate(row)) for row in rows))

    return {
        "n": int(len(rows)),
        "strict_pass": count(lambda r: r["strict"]),
        "strict_rate": float(np.mean([bool(r["strict"]) for r in rows])),
        "both_correct": count(lambda r: r["both_correct"]),
        "wrong_accept": count(lambda r: r["wrong_accept"]),
        "wrong_clutter": count(lambda r: r["wrong_clutter"]),
        "wrong_branch": count(lambda r: r["wrong_branch"]),
        "partial_or_no_confirmation": count(lambda r: r["confirmed_labels"] < 2),
        "material_within_3mm": count(lambda r: r["material_within_3mm"]),
        "prefix_provisional_wrong": int(sum(r["prefix_provisional_wrong"] for r in rows)),
        "prefix_confident_wrong": int(sum(r["prefix_confident_wrong"] for r in rows)),
        "prefix_reassociated": int(sum(r["prefix_reassociated"] for r in rows)),
        "prefix_wrong_replaced": int(sum(r["prefix_wrong_replaced"] for r in rows)),
        "prefix_clutter_replaced": int(sum(r["prefix_clutter_replaced"] for r in rows)),
        "prefix_branch_replaced": int(sum(r["prefix_branch_replaced"] for r in rows)),
        "prefix_confident_wrong_replaced": int(sum(r["prefix_confident_wrong_replaced"] for r in rows)),
        "new_confident_wrong": int(sum(r["new_confident_wrong"] for r in rows)),
        "current_rmse_mm": pooled(("current", "rmse_mm")),
        "future_half_s_rmse_mm": pooled(("future", 0, "rmse_mm")),
        "future_one_s_rmse_mm": pooled(("future", 1, "rmse_mm")),
        "common_clock_rmse_mm": pooled(("common_clock", "rmse_mm")),
        "future_half_s_90_coverage": mean_nested(rows, ("future", 0, "coordinate_90_coverage")),
        "future_one_s_90_coverage": mean_nested(rows, ("future", 1, "coordinate_90_coverage")),
        "common_clock_90_coverage": mean_nested(rows, ("common_clock", "coordinate_90_coverage")),
        "z_rmse_mm": pooled(("z_error_mm",)),
        "mean_duration_s": mean("elapsed_s"),
        "mean_travel_mm": mean("travel_mm"),
        "mean_captures": mean("captures"),
        "cap_violations": count(lambda r: not r["cap_satisfied"]),
    }


def _scalar(row: Mapping[str, Any], field: str) -> float:
    value = row[field]
    return float(value)


def paired_delta(rule: list[dict[str, Any]], variant: list[dict[str, Any]],
                 random: np.random.Generator, *, bootstraps: int = DEFAULT_BOOTSTRAPS) -> dict[str, Any]:
    """Case-paired bootstrap for identity, time/travel and world RMSE."""
    if len(rule) != len(variant):
        raise ValueError("paired arms must have equal case count")
    n = len(rule)
    indices = random.integers(0, n, size=(int(bootstraps), n))

    def delta(field: str) -> dict[str, Any]:
        differences = np.asarray([_scalar(b, field) - _scalar(a, field)
                                  for a, b in zip(rule, variant)], dtype=float)
        samples = differences[indices].mean(axis=1)
        low, high = np.quantile(samples, [.025, .975])
        return {
            "mean": float(differences.mean()), "ci95": [float(low), float(high)],
            "wins": int(np.sum(differences > 0.)), "losses": int(np.sum(differences < 0.)),
        }

    def pooled_delta(path: tuple[Any, ...]) -> dict[str, Any]:
        left = np.asarray([nested_value(a, path) for a in rule], dtype=float)
        right = np.asarray([nested_value(b, path) for b in variant], dtype=float)
        values = np.sqrt((right[indices] ** 2).mean(axis=1)) - np.sqrt((left[indices] ** 2).mean(axis=1))
        return {
            "mean": float(np.sqrt(np.mean(right ** 2)) - np.sqrt(np.mean(left ** 2))),
            "ci95": [float(v) for v in np.quantile(values, [.025, .975])],
        }

    return {
        "strict_pass": delta("strict"),
        "both_correct": delta("both_correct"),
        "wrong_accept": delta("wrong_accept"),
        "travel_mm": delta("travel_mm"),
        "elapsed_s": delta("elapsed_s"),
        "fixed_clock_rmse_delta_mm": pooled_delta(("common_clock", "rmse_mm")),
        "future_half_s_rmse_delta_mm": pooled_delta(("future", 0, "rmse_mm")),
        "future_one_s_rmse_delta_mm": pooled_delta(("future", 1, "rmse_mm")),
    }


def mean_ci(values: np.ndarray, random: np.random.Generator,
            *, bootstraps: int = DEFAULT_BOOTSTRAPS) -> list[float]:
    values = np.asarray(values, dtype=float)
    indices = random.integers(0, len(values), size=(int(bootstraps), len(values)))
    return [float(v) for v in np.quantile(values[indices].mean(axis=1), [.025, .975])]


def cost_only_match(rule_rows: list[dict[str, Any]], ladder_rows: dict[float, list[dict[str, Any]]],
                    random: np.random.Generator, *, bootstraps: int) -> dict[str, Any]:
    """Global outcome-blind travel match; bootstrap refits one global weight."""
    rule_travel = float(np.mean([row["travel_mm"] for row in rule_rows]))
    ladder = []
    for value in sorted(ladder_rows):
        values = ladder_rows[value]
        ladder.append((float(value), float(np.mean([row["travel_mm"] for row in values]))))
    brackets = []
    for (low_l, low_t), (high_l, high_t) in zip(ladder[:-1], ladder[1:]):
        if (low_t - rule_travel) * (high_t - rule_travel) <= 0. and abs(low_t - high_t) > 1.e-12:
            p_low = float(np.clip((rule_travel - high_t) / (low_t - high_t), 0., 1.))
            brackets.append((abs(low_t - high_t), low_l, high_l, p_low))
    if brackets:
        _, low_lambda, high_lambda, p_low = min(brackets, key=lambda x: (x[0], x[1]))
        mode = "bracket_interpolation"
    else:
        nearest = min(ladder, key=lambda x: (abs(x[1] - rule_travel), x[0]))
        low_lambda = high_lambda = nearest[0]
        p_low = 1.0
        mode = "nearest_travel_fallback"

    low = ladder_rows[low_lambda]
    high = ladder_rows[high_lambda]
    n = len(rule_rows)
    indices = random.integers(0, n, size=(int(bootstraps), n))
    rule_t = np.asarray([row["travel_mm"] for row in rule_rows], dtype=float)
    low_t = np.asarray([row["travel_mm"] for row in low], dtype=float)
    high_t = np.asarray([row["travel_mm"] for row in high], dtype=float)

    # Re-select only the adjacent travel bracket inside each bootstrap sample.
    # This keeps matching global (one weight per replicate), while avoiding the
    # artificial travel CI caused by clipping a weight from one fixed sample.
    all_lambdas = [item[0] for item in ladder]
    all_travel = np.asarray([
        [row["travel_mm"] for row in ladder_rows[value]] for value in all_lambdas
    ], dtype=float)
    boot_rule_t = rule_t[indices].mean(axis=1)
    boot_travel = all_travel[:, indices].mean(axis=2)
    boot_low_index = np.empty(len(indices), dtype=int)
    boot_high_index = np.empty(len(indices), dtype=int)
    boot_p = np.empty(len(indices), dtype=float)
    for sample_index, target in enumerate(boot_rule_t):
        options = []
        for j in range(len(all_lambdas) - 1):
            lower = boot_travel[j, sample_index]
            upper = boot_travel[j + 1, sample_index]
            if ((lower - target) * (upper - target) <= 0.
                    and abs(lower - upper) > 1.e-12):
                weight = float(np.clip((target - upper) / (lower - upper), 0., 1.))
                options.append((abs(lower - upper), j, j + 1, weight))
        if options:
            _, j, k, weight = min(options, key=lambda item: (item[0], item[1]))
        else:
            flat = np.abs(boot_travel[:, sample_index] - target)
            j = k = int(np.argmin(flat))
            weight = 1.0
        boot_low_index[sample_index] = j
        boot_high_index[sample_index] = k
        boot_p[sample_index] = weight

    def metric_delta(field: str) -> dict[str, Any]:
        rule = np.asarray([_scalar(row, field) for row in rule_rows], dtype=float)
        lo = np.asarray([_scalar(row, field) for row in low], dtype=float)
        hi = np.asarray([_scalar(row, field) for row in high], dtype=float)
        all_metric = np.asarray([
            [_scalar(row, field) for row in ladder_rows[value]] for value in all_lambdas
        ], dtype=float)
        sample = (boot_p * all_metric[boot_low_index[:, None], indices].mean(axis=1)
                  + (1. - boot_p) * all_metric[boot_high_index[:, None], indices].mean(axis=1)
                  - rule[indices].mean(axis=1))
        observed = p_low * lo.mean() + (1. - p_low) * hi.mean() - rule.mean()
        return {"mean": float(observed),
                "ci95": [float(v) for v in np.quantile(sample, [.025, .975])],
                "wins": int(np.sum(p_low * lo + (1. - p_low) * hi > rule)),
                "losses": int(np.sum(p_low * lo + (1. - p_low) * hi < rule))}

    def rmse_delta(path: tuple[Any, ...]) -> dict[str, Any]:
        rule = np.asarray([nested_value(row, path) for row in rule_rows], dtype=float)
        lo = np.asarray([nested_value(row, path) for row in low], dtype=float)
        hi = np.asarray([nested_value(row, path) for row in high], dtype=float)
        all_metric = np.asarray([
            [nested_value(row, path) for row in ladder_rows[value]] for value in all_lambdas
        ], dtype=float)
        sample = np.sqrt(
            boot_p * (all_metric[boot_low_index[:, None], indices] ** 2).mean(axis=1)
            + (1. - boot_p) * (all_metric[boot_high_index[:, None], indices] ** 2).mean(axis=1)
        )
        sample -= np.sqrt((rule[indices] ** 2).mean(axis=1))
        observed = np.sqrt(p_low * np.mean(lo ** 2) + (1. - p_low) * np.mean(hi ** 2))
        observed -= np.sqrt(np.mean(rule ** 2))
        return {"mean": float(observed),
                "ci95": [float(v) for v in np.quantile(sample, [.025, .975])]}

    fixed_mean_travel = p_low * low_t.mean() + (1. - p_low) * high_t.mean()
    return {
        "mode": mode, "outcome_blind": True,
        "target_rule_mean_travel_mm": rule_travel,
        "lambda_low": float(low_lambda), "lambda_high": float(high_lambda),
        "weight_low": float(p_low), "weight_high": float(1. - p_low),
        "matched_mean_travel_mm": float(fixed_mean_travel),
        "bootstrap_weight_refit": True, "bootstrap_bracket_refit": True,
        "paired_vs_rule": {
            "strict_pass": metric_delta("strict"),
            "both_correct": metric_delta("both_correct"),
            "wrong_accept": metric_delta("wrong_accept"),
            "travel_mm": metric_delta("travel_mm"),
            "elapsed_s": metric_delta("elapsed_s"),
            "fixed_clock_rmse_delta_mm": rmse_delta(("common_clock", "rmse_mm")),
            "future_half_s_rmse_delta_mm": rmse_delta(("future", 0, "rmse_mm")),
            "future_one_s_rmse_delta_mm": rmse_delta(("future", 1, "rmse_mm")),
        },
    }


def select_price(dev_rows: list[dict[str, Any]], lambdas: tuple[float, ...],
                 cap_mm: float) -> dict[str, Any]:
    """Select/interpolate using only independent-dev actual travel."""
    grouped: dict[float, list[dict[str, Any]]] = {float(l): [] for l in lambdas}
    rules = []
    for row in dev_rows:
        rules.append(row["rule"])
        for key, value in row["cql"].items():
            grouped[float(value["lambda_distance"])].append(value)
    rule_travel = float(np.mean([r["travel_mm"] for r in rules]))
    ladder = []
    for value in sorted(grouped):
        rows = grouped[value]
        if not rows:
            raise ValueError(f"missing dev price rows for lambda {value}")
        ladder.append({
            "lambda_distance": float(value), "n": len(rows),
            "mean_travel_mm": float(np.mean([r["travel_mm"] for r in rows])),
            "mean_elapsed_s": float(np.mean([r["elapsed_s"] for r in rows])),
            "cap_violations": int(sum(not r["cap_satisfied"] for r in rows)),
        })

    # Larger prices should weakly reduce travel.  Use the nearest bracket that
    # contains the rule's observed mean; if dev discreteness leaves no bracket,
    # choose the nearest travel point.  Neither branch inspects outcomes.
    candidates = []
    for lower, upper in zip(ladder[:-1], ladder[1:]):
        t_low, t_high = lower["mean_travel_mm"], upper["mean_travel_mm"]
        if (t_low - rule_travel) * (t_high - rule_travel) <= 0. and abs(t_low - t_high) > 1.e-12:
            p_low = (rule_travel - t_high) / (t_low - t_high)
            candidates.append((abs(t_low - t_high), lower, upper, float(np.clip(p_low, 0., 1.))))
    if candidates:
        _, lower, upper, p_low = min(candidates, key=lambda x: (x[0], x[1]["lambda_distance"]))
        mode = "bracket_interpolation"
    else:
        nearest = min(ladder, key=lambda x: (abs(x["mean_travel_mm"] - rule_travel), x["lambda_distance"]))
        lower = upper = nearest
        p_low = 1.0
        mode = "nearest_travel_fallback"
    return {
        "cap_mm": float(cap_mm),
        "target_rule_mean_travel_mm": rule_travel,
        "mode": mode,
        "lambda_low": float(lower["lambda_distance"]),
        "lambda_high": float(upper["lambda_distance"]),
        "probability_choose_lambda_low": float(p_low),
        "lambda_ladder": ladder,
        "selection_uses_outcomes": False,
        "selection_basis": "dev actual cumulative supplementary travel only",
    }


def run_dev(args, world: World, model: Any) -> dict[str, Any]:
    out = args.out
    path = out / "DEV_LADDER.jsonl"
    rows = []
    with path.open("w") as stream:
        for index in range(args.dev_start, args.dev_start + args.dev_count):
            seed, case, prefix = case_prefix(world, "dev", index)
            rule = run_rule(world, case, prefix, args.cap_mm, trace=False)
            cql = {}
            for value in args.lambdas:
                cql[str(float(value))] = run_cql(
                    world, case, prefix, model, float(value), args.cap_mm, trace=False,
                )
            row = {
                "split": "dev", "index": int(index), "seed": int(seed),
                "prefix_acquisitions": len(prefix.frames), "prefix_sha256": prefix_hash(prefix),
                "candidate_envelope": candidate_envelope(world, prefix, args.cap_mm),
                "rule": rule, "cql": cql,
            }
            rows.append(row)
            stream.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
            stream.flush()
            print(f"dev {index} / {args.dev_start + args.dev_count - 1}", flush=True)
    selection = select_price(rows, args.lambdas, args.cap_mm)
    write_json(out / "PRICE_SELECTION.json", selection)
    dev_summary = {
        "split": "dev", "cases": len(rows), "indices": [r["index"] for r in rows],
        "cap_mm": float(args.cap_mm), "lambdas": [float(v) for v in args.lambdas],
        "rule": row_summary([r["rule"] for r in rows]),
        "cql_ladder": {
            lambda_key(float(v)): row_summary([r["cql"][str(float(v))] for r in rows])
            for v in args.lambdas
        },
        "price_selection": selection,
        "selection_random_seed": PRICE_SEED,
    }
    write_json(out / "DEV_SUMMARY.json", dev_summary)
    return selection


def selected_lambda_for_case(seed: int, selection: Mapping[str, Any]) -> tuple[float, float, float]:
    low = float(selection["lambda_low"])
    high = float(selection["lambda_high"])
    probability_low = float(selection["probability_choose_lambda_low"])
    if low == high:
        return low, 0.0, 1.0
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), MIX_SEED_SALT]))
    uniform = float(rng.random())
    chosen = low if uniform < probability_low else high
    return float(chosen), uniform, probability_low


def run_test(args, world: World, model: Any, selection: Mapping[str, Any]) -> None:
    out = args.out
    path = out / "TEST_LADDER.jsonl"
    lambdas = tuple(float(value) for value in args.lambdas)
    low = float(selection["lambda_low"])
    high = float(selection["lambda_high"])
    with path.open("w") as stream:
        for index in range(args.test_start, args.test_start + args.test_count):
            seed, case, prefix = case_prefix(world, "test", index)
            rule = run_rule(world, case, prefix, args.cap_mm, trace=True)
            cql = {}
            for value in lambdas:
                cql[str(float(value))] = run_cql(
                    world, case, prefix, model, float(value), args.cap_mm,
                    trace=(float(value) == low or float(value) == high),
                )
            chosen, uniform, probability_low = selected_lambda_for_case(seed, selection)
            selected_row = cql[str(float(chosen))]
            row = {
                "split": "test", "index": int(index), "seed": int(seed),
                "prefix_acquisitions": len(prefix.frames), "prefix_sha256": prefix_hash(prefix),
                "candidate_envelope": candidate_envelope(world, prefix, args.cap_mm),
                "rule": rule, "cql": cql,
                "selected": {
                    "policy": "cql_price_dev_travel_mix", "lambda_distance": float(chosen),
                    "lambda_low": low, "lambda_high": high,
                    "probability_choose_lambda_low": float(probability_low),
                    "selection_uniform": float(uniform), "selection_is_outcome_blind": True,
                    "run": selected_row,
                },
            }
            if len({rule["prefix_sha256"], selected_row["prefix_sha256"]}) != 1:
                raise RuntimeError("paired prefix hash mismatch")
            if any(not arm["cap_satisfied"] for arm in [rule, *cql.values()]):
                raise RuntimeError("distance cap violation")
            stream.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
            stream.flush()
            print(f"test {index} / {args.test_start + args.test_count - 1}", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def summarize_test(args, rows: list[dict[str, Any]], selection: Mapping[str, Any]) -> dict[str, Any]:
    random = np.random.default_rng(93681)
    rule_rows = [row["rule"] for row in rows]
    selected_rows = [row["selected"]["run"] for row in rows]
    ladder = {}
    for value in args.lambdas:
        key = str(float(value))
        arm = [row["cql"][key] for row in rows]
        ladder[lambda_key(float(value))] = {
            "lambda_distance": float(value), "summary": row_summary(arm),
            "paired_vs_rule": paired_delta(rule_rows, arm, random, bootstraps=args.bootstraps),
        }
    cost_ladder_rows = {
        float(value): [row["cql"][str(float(value))] for row in rows]
        for value in args.lambdas
    }
    cost_match = cost_only_match(rule_rows, cost_ladder_rows, random,
                                 bootstraps=args.bootstraps)
    summary = {
        "split": "test", "cases": len(rows),
        "indices": [int(row["index"]) for row in rows],
        "case_seed_range": [int(min(row["seed"] for row in rows)),
                            int(max(row["seed"] for row in rows))],
        "distance_cap_mm": float(args.cap_mm),
        "first_sweep": {
            "shared": True, "prefix_acquisitions": int(rows[0]["prefix_acquisitions"]),
            "unique_prefix_hashes": int(len({row["prefix_sha256"] for row in rows})),
            "description": "same full moving-knee rule first sweep; no Fisher score or Fisher-based selection",
        },
        "raw_arms": {
            "rule": row_summary(rule_rows),
            "cql_price_dev_travel_mix_actual": row_summary(selected_rows),
        },
        "paired_vs_rule": {
            "cql_price_dev_travel_mix_actual": paired_delta(
                rule_rows, selected_rows, random, bootstraps=args.bootstraps),
        },
        "cost_frontier": ladder,
        "cost_frontier_match": cost_match,
        "price_selection": dict(selection),
        "bootstrap": {"replicates": int(args.bootstraps), "seed": 93681,
                       "unit": "case-paired", "outcome_blind_matching": True},
        "protocol_checks": {
            "all_cap_satisfied": bool(
                all(row["rule"]["cap_satisfied"] for row in rows)
                and all(row["selected"]["run"]["cap_satisfied"] for row in rows)),
            "all_budget_three": bool(
                all(row["rule"]["captures"] == BUDGET for row in rows)
                and all(row["selected"]["run"]["captures"] == BUDGET for row in rows)),
            "selection_uses_outcome": False,
            "selection_uses_travel_only": True,
            "world_is_3d": True,
        },
    }
    write_json(args.out / "TEST_SUMMARY.json", summary)
    return summary


def plot_3d(args, rows: list[dict[str, Any]], summary: Mapping[str, Any], world: World) -> None:
    """Simulated capture poses and two forecast comparisons."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # pragma: no cover
        (args.out / "PLOT_SKIPPED.txt").write_text(f"Plotly unavailable: {exc}\n")
        return

    both_move = [r for r in rows if min(r["rule"]["travel_mm"],
                                       r["selected"]["run"]["travel_mm"]) > 1.e-8]
    chosen = min(both_move or rows, key=lambda r: abs(
        r["rule"]["travel_mm"] - r["selected"]["run"]["travel_mm"]))
    prefix = chosen["rule"]["trace"][:chosen["prefix_acquisitions"]]
    rule = chosen["rule"]
    selected_run = chosen["selected"]["run"]
    fig = make_subplots(
        rows=2, cols=2, specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}],
                                [None, {"type": "xy"}]],
        column_widths=[.58, .42], vertical_spacing=.17,
        subplot_titles=("Example capture poses and vessel geometry",
                        "0.5 s prediction after each policy's last frame",
                        "Prediction at the same absolute time"),
    )

    def add_path(frames, name, color, *, visible=True):
        xyz = np.asarray([frame["probe_xyz_mm"] for frame in frames], dtype=float)
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines+markers",
            name=name, line={"color": color, "width": 4, "dash": "dot"}, visible=visible,
            marker={"color": color, "size": 4},
            text=[f"Station {f['site']} · {f['t'] - prefix[-1]['t']:.2f} s after sweep" for f in frames],
            hovertemplate="%{text}<extra>%{fullData.name}</extra>",
        ), row=1, col=1)

    add_path(prefix, "Shared rule sweep", "#9aa7ad", visible="legendonly")
    add_path([prefix[-1], *rule["trace"][chosen["prefix_acquisitions"]:]],
             f"Rule in example · {rule['travel_mm']:.1f} mm", "#206f83")
    add_path([prefix[-1], *selected_run["trace"][chosen["prefix_acquisitions"]:]],
             f"Dev-calibrated CQL in example · {selected_run['travel_mm']:.1f} mm", "#8055a4")

    truth = np.asarray(rule["common_clock"]["truth_xyz_mm"], dtype=float)
    groups = [[], []]
    for i, (_, label) in enumerate(world.query_sites):
        groups[int(label)].append(i)
    for label, indices in enumerate(groups):
        xyz = truth[indices]
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines+markers",
            name=("True main vessel", "True branch")[label],
            line={"color": "#22272b", "width": 5}, marker={"color": "#22272b", "size": 3},
        ), row=1, col=1)

    frontier = summary["cost_frontier"]
    xs = [frontier[lambda_key(float(v))]["summary"]["mean_travel_mm"] for v in args.lambdas]
    rule_summary = summary["raw_arms"]["rule"]
    match = summary["cost_frontier_match"]
    for row, (metric, difference) in enumerate((
        ("future_half_s_rmse_mm", "future_half_s_rmse_delta_mm"),
        ("common_clock_rmse_mm", "fixed_clock_rmse_delta_mm"),
    ), start=1):
        ys = [frontier[lambda_key(float(v))]["summary"][metric] for v in args.lambdas]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name="Frozen CQL price ladder",
            line={"color": "#8055a4"}, marker={"size": 6}, showlegend=row == 1,
            text=[f"Distance price {float(v):.5f}" for v in args.lambdas],
            hovertemplate="%{text}<br>Travel %{x:.2f} mm<br>RMSE %{y:.3f} mm<extra></extra>",
        ), row=row, col=2)
        fig.add_trace(go.Scatter(
            x=[rule_summary["mean_travel_mm"]], y=[rule_summary[metric]],
            mode="markers", name="Rule cohort mean", showlegend=row == 1,
            marker={"color": "#206f83", "size": 10, "symbol": "diamond"},
        ), row=row, col=2)
        fig.add_trace(go.Scatter(
            x=[match["matched_mean_travel_mm"]],
            y=[rule_summary[metric] + match["paired_vs_rule"][difference]["mean"]],
            mode="markers", name="CQL at matched mean travel", showlegend=row == 1,
            marker={"color": "#8055a4", "size": 11, "symbol": "x"},
        ), row=row, col=2)
        fig.update_yaxes(title_text="3D RMSE (mm)", row=row, col=2)
    fig.update_scenes(aspectmode="data", xaxis_title="World X (mm)", yaxis_title="World Y (mm)",
                      zaxis_title="World Z (mm)", camera={"eye": {"x": 2.0, "y": 1.8, "z": 1.25}})
    fig.update_xaxes(title_text="Mean supplementary travel (mm)", row=2, col=2)
    fig.update_layout(
        template="plotly_white", height=780,
        title=("Distance-matched knee recheck<br>"
               "<sup>Dotted lines connect simulated captures · Distance matching is across the cohort</sup>"),
        legend={"orientation": "h", "y": -0.13}, margin={"l": 26, "r": 24, "t": 110, "b": 112},
    )
    fig.write_html(args.out / "MATCHED_3D.html", include_plotlyjs=True)
    write_json(args.out / "PLOT_CASE.json", {
        "index": int(chosen["index"]), "seed": int(chosen["seed"]),
        "selected_lambda": float(chosen["selected"]["lambda_distance"]),
        "rule_travel_mm": float(rule["travel_mm"]),
        "cql_travel_mm": float(selected_run["travel_mm"]),
        "uses_simulated_frames_only": True,
    })


def protocol_text(args) -> str:
    return f"""Distance-matched supplementary knee recheck
Date: 2026-11-12

Frozen source
------------
World, prepared prefix, BUDGET={BUDGET}, belief/threshold/evaluate_outcome,
original policy.py and frozen cql_512_7.pt are loaded read-only from:
  {PREVIOUS}
The moving-knee world is 3-D. No Fisher score, Fisher feature, or Fisher-based
selection is used.

Paired protocol
---------------
Each arm starts from the identical full rule first sweep (28 simulated frames)
for a case and executes exactly {BUDGET} supplementary acquisitions. A common
per-case cumulative actual path cap of {args.cap_mm:.1f} mm is enforced on each
candidate. The same mask is applied to CQL and strong rule at every actual step;
the rule lookahead applies it at both levels. Frozen geometric edge validity is
also required.

Price selection
---------------
On independent dev indices {args.dev_start}..{args.dev_start + args.dev_count - 1},
CQL scores are adjusted by -lambda_distance*candidate_additional_distance_mm.
The bracket and fixed mixture probability target the strong-rule dev mean travel
using travel alone. Identity, RMSE, strict status and hidden truth are not read
for selection. Held-out cases use a deterministic seeded outcome-blind mixture;
the expected weighted arm is reported separately from the actual sampled arm.

Held-out protocol
-----------------
Test indices {args.test_start}..{args.test_start + args.test_count - 1} are fresh
seeds relative to the frozen 0..255 held-out run. Raw rows retain strict and
mis-ID branch/clutter/prefix replacement errors, 0.5/1.0 s and shared-clock
3-D world RMSE, elapsed time, modeled actual travel, and simulated traces. Confidence intervals
use {args.bootstraps} case-paired bootstrap replicates.

Outputs
-------
DEV_LADDER.jsonl, DEV_SUMMARY.json, PRICE_SELECTION.json,
TEST_LADDER.jsonl, TEST_SUMMARY.json, MATCHED_3D.html, PLOT_CASE.json,
and MANIFEST.json are written under this distance/ directory.
"""


def manifest(args, selection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source_names = ("belief.py", "knee_world.py", "policy.py", "experiment.py", "learners.py")
    return {
        "driver": sha(HERE / "distance.py"), "frozen_root": str(PREVIOUS),
        "frozen_sources": {name: sha(PREVIOUS / name) for name in source_names},
        "geometry": sha(GEOMETRY),
        "model": {"path": str(args.model), "sha256": sha(args.model)},
        "python": sys.executable,
        "config": {
            "cap_mm": float(args.cap_mm), "budget": int(BUDGET),
            "dev_start": int(args.dev_start), "dev_count": int(args.dev_count),
            "test_start": int(args.test_start), "test_count": int(args.test_count),
            "lambdas": [float(v) for v in args.lambdas], "bootstraps": int(args.bootstraps),
        },
        "selection": dict(selection) if selection is not None else None,
        "constraints": {
            "first_sweep_shared": True, "fisher_used": False, "new_training": False,
            "distance_cap_applied_to_rule_lookahead": True,
            "distance_cap_applied_to_cql": True, "selection_outcome_blind": True,
            "3d_measured_traces_only": True,
        },
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dev", "test", "summary", "all"))
    parser.add_argument("--out", type=Path, default=HERE / "distance")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cap-mm", type=float, default=DEFAULT_CAP_MM)
    parser.add_argument("--dev-start", type=int, default=0)
    parser.add_argument("--dev-count", type=int, default=DEFAULT_DEV_COUNT)
    parser.add_argument("--test-start", type=int, default=DEFAULT_TEST_START)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cap_mm <= 0.:
        raise ValueError("cap-mm must be positive")
    if args.dev_count <= 0 or args.test_count <= 0:
        raise ValueError("case counts must be positive")
    args.lambdas = tuple(sorted(set(float(v) for v in args.lambdas)))
    if any(v < 0. for v in args.lambdas):
        raise ValueError("distance prices must be nonnegative")
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    (args.out / "PROTOCOL.txt").write_text(protocol_text(args))
    world = World()
    model = learners.load(args.model, device="cpu")
    selection = None
    if args.mode in ("dev", "all"):
        selection = run_dev(args, world, model)
    elif (args.out / "PRICE_SELECTION.json").exists():
        selection = json.loads((args.out / "PRICE_SELECTION.json").read_text())
    else:
        raise FileNotFoundError("PRICE_SELECTION.json; run distance.py dev first")
    if args.mode in ("test", "all"):
        run_test(args, world, model, selection)
    if args.mode in ("summary", "all"):
        if selection is None:
            selection = json.loads((args.out / "PRICE_SELECTION.json").read_text())
        rows = load_jsonl(args.out / "TEST_LADDER.jsonl")
        summary = summarize_test(args, rows, selection)
        plot_3d(args, rows, summary, world)
    write_json(args.out / "MANIFEST.json", manifest(args, selection))
    print(json.dumps({"mode": args.mode, "out": str(args.out), "selection": selection},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
