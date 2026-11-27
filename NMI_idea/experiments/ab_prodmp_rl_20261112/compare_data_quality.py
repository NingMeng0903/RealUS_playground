#!/usr/bin/env python3
"""Paired data-quality check for frozen suffix policies.

The command evaluates the same generated episode seed with four arms: the
frozen rule, the original offline-Q checkpoint, a Q checkpoint trained with
additional uniform data, and a Q checkpoint trained with additional Fisher
data.  It writes one compact JSON record per paired case and a separate JSON
summary.  The expensive rollout implementation remains in ``rl_rollouts``;
this module only orchestrates paired calls and computes auditable summaries.

Example::

    python compare_data_quality.py --seed 2411202619 --start 0 --n 2 \
      --checkpoint original=frozen_q/POLICY.npz \
      --checkpoint uniform=uniform_fit/POLICY.npz \
      --checkpoint fisher=fisher_fit/POLICY.npz \
      --out compare.jsonl

The default ``--n 1`` is intentionally small.  Larger holdout runs should be
sharded explicitly with ``--start``/``--n`` and paired by the same seed base.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fit_q import FrozenQ  # noqa: E402
import rl_rollouts  # noqa: E402


ARM_NAMES = ("rule", "original", "uniform", "fisher")
RL_ARMS = ("original", "uniform", "fisher")
OPTIONAL_RL_ARMS = ("uniform_matched",)
ARM_ALIASES = {
    "rule": "rule",
    "rl": "original",
    "orig": "original",
    "original": "original",
    "original_rl": "original",
    "original-rl": "original",
    "uniform": "uniform",
    "add_uniform": "uniform",
    "additional_uniform": "uniform",
    "additional-uniform": "uniform",
    "additional_uniform_rl": "uniform",
    "additional-uniform-rl": "uniform",
    "uniform_rl": "uniform",
    "uniform_matched": "uniform_matched",
    "uniform-matched": "uniform_matched",
    "additional_uniform_matched": "uniform_matched",
    "additional-uniform-matched": "uniform_matched",
    "uniform_matched_rl": "uniform_matched",
    "uniform-matched-rl": "uniform_matched",
    "fisher": "fisher",
    "add_fisher": "fisher",
    "additional_fisher": "fisher",
    "additional-fisher": "fisher",
    "additional_fisher_rl": "fisher",
    "additional-fisher-rl": "fisher",
    "fisher_rl": "fisher",
}


def _strict_success(run: Mapping[str, Any]) -> bool:
    """Read the strict result field used by both current and older rollouts."""

    if "strict_complete" in run:
        return bool(run["strict_complete"])
    if "complete" in run:
        return bool(run["complete"])
    raise KeyError("rollout has neither strict_complete nor complete")


def _capture_count(run: Mapping[str, Any]) -> int:
    if "steps" in run:
        return int(len(run["steps"]))
    for key in ("captures", "rechecks", "n_captures"):
        if key in run:
            return int(run[key])
    raise KeyError("rollout has no capture/step count")


def _path_m(run: Mapping[str, Any]) -> float:
    if "recheck_path_mm" in run:
        return float(run["recheck_path_mm"]) / 1000.0
    for key in ("path_m", "travel_m", "distance_m"):
        if key in run:
            return float(run[key])
    raise KeyError("rollout has no path/travel cost")


def _mean_return(run: Mapping[str, Any]) -> float:
    """Use the protocol's undiscounted strict-completion return."""

    if "undiscounted_return" in run:
        return float(run["undiscounted_return"])
    return float(_strict_success(run)) - 0.002 * _capture_count(run) - 0.05 * _path_m(run)


def _arm_metrics(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {
            "n": 0,
            "strict_successes": 0,
            "success": 0,
            "strict_success_rate": None,
            "false_stops": 0,
            "no_stops": 0,
            "mean_captures": None,
            "mean_rechecks": None,
            "mean_path_m": None,
            "mean_travel_m": None,
            "mean_return": None,
            "mean_undiscounted_return": None,
        }
    successes = np.asarray([int(_strict_success(run)) for run in runs], dtype=np.int64)
    captures = np.asarray([_capture_count(run) for run in runs], dtype=np.float64)
    paths = np.asarray([_path_m(run) for run in runs], dtype=np.float64)
    returns = np.asarray([_mean_return(run) for run in runs], dtype=np.float64)
    return {
        "n": int(len(runs)),
        "strict_successes": int(successes.sum()),
        "success": int(successes.sum()),
        "strict_success_rate": float(successes.mean()),
        "false_stops": int(sum(bool(run.get("online_stop", False)) and not _strict_success(run) for run in runs)),
        "no_stops": int(sum(not bool(run.get("online_stop", False)) for run in runs)),
        "mean_captures": float(captures.mean()),
        "mean_rechecks": float(captures.mean()),
        "mean_path_m": float(paths.mean()),
        "mean_travel_m": float(paths.mean()),
        "mean_return": float(returns.mean()),
        "mean_undiscounted_return": float(returns.mean()),
    }


def _exact_binomial_two_sided(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value on non-draw discordances."""

    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    observed = math.comb(n, wins) / (2.0**n)
    probability = 0.0
    for k in range(n + 1):
        mass = math.comb(n, k) / (2.0**n)
        if mass <= observed + 1.0e-15:
            probability += mass
    return float(min(1.0, probability))


def _paired(cases: Mapping[int, Mapping[str, Mapping[str, Any]]], first: str, second: str) -> dict[str, Any]:
    first_runs = [case[first] for case in cases.values()]
    second_runs = [case[second] for case in cases.values()]
    first_success = np.asarray([int(_strict_success(run)) for run in first_runs], dtype=np.int64)
    second_success = np.asarray([int(_strict_success(run)) for run in second_runs], dtype=np.int64)
    success_delta = first_success - second_success
    wins = int(np.count_nonzero(success_delta > 0))
    losses = int(np.count_nonzero(success_delta < 0))
    first_captures = np.asarray([_capture_count(run) for run in first_runs], dtype=np.float64)
    second_captures = np.asarray([_capture_count(run) for run in second_runs], dtype=np.float64)
    first_paths = np.asarray([_path_m(run) for run in first_runs], dtype=np.float64)
    second_paths = np.asarray([_path_m(run) for run in second_runs], dtype=np.float64)
    first_returns = np.asarray([_mean_return(run) for run in first_runs], dtype=np.float64)
    second_returns = np.asarray([_mean_return(run) for run in second_runs], dtype=np.float64)
    exact_p = _exact_binomial_two_sided(wins, losses)
    return {
        "first": first,
        "second": second,
        "n": int(len(cases)),
        "success_wins": wins,
        "success_losses": losses,
        "success_draws": int(len(cases) - wins - losses),
        "discordant": int(wins + losses),
        "success_delta_pp": float(100.0 * success_delta.mean()) if len(cases) else 0.0,
        "exact_binomial_p_two_sided": exact_p,
        "exact_p_two_sided": exact_p,
        "capture_delta": float((first_captures - second_captures).mean()) if len(cases) else 0.0,
        "path_delta_m": float((first_paths - second_paths).mean()) if len(cases) else 0.0,
        "return_delta": float((first_returns - second_returns).mean()) if len(cases) else 0.0,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _episode_seed(seed_base: int, index: int) -> int:
    """Match rl_rollouts' deterministic seed derivation exactly."""

    return int(np.random.SeedSequence([int(seed_base), int(index)]).generate_state(1)[0])


def _parse_checkpoints(items: Iterable[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--checkpoint must be NAME=PATH, got {item!r}")
        raw_name, raw_path = item.split("=", 1)
        name = ARM_ALIASES.get(raw_name.strip().lower())
        if name not in RL_ARMS + OPTIONAL_RL_ARMS:
            expected = ", ".join(RL_ARMS + OPTIONAL_RL_ARMS)
            raise ValueError(f"unknown checkpoint arm {raw_name!r}; expected {expected}")
        if name in parsed:
            raise ValueError(f"duplicate checkpoint arm {name!r}")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        parsed[name] = path
    missing = [name for name in RL_ARMS if name not in parsed]
    if missing:
        raise ValueError(f"missing checkpoints for {', '.join(missing)}")
    return parsed


def _run_one(
    run_episode: Any,
    *,
    planner: Any,
    scan: Any,
    support: Any,
    model: Any,
    episode_seed: int,
    cap: int,
    policy: str,
    q_model: Any,
    prepared_episode: Any | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Invoke the current rollout API, reusing the prepared prefix when available."""

    kwargs = {
        "planner": planner,
        "scan": scan,
        "support": support,
        "model": model,
        "episode_seed": episode_seed,
        "mode": "evaluate",
        "policy": policy,
        "q_model": q_model,
        "cap": cap,
    }
    # Current rl_rollouts exposes prepared_episode explicitly.  Keep a small
    # fallback for an older checkout that has no preparation hook at all.
    if prepared_episode is not None:
        kwargs["prepared_episode"] = prepared_episode
    result = run_episode(**kwargs)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("run_episode must return (run, executed_decisions)")
    return result


def _prefix_fields(run: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that must remain common before the first learned suffix choice."""

    fields: dict[str, Any] = {}
    for key in ("prefix_acquisitions", "initial_certified", "prefix_steps"):
        if key in run:
            fields[key] = run[key]
    if not fields:
        raise KeyError("rollout has no shared-prefix fields")
    return fields


def _assert_shared_prefix(
    case_index: int,
    runs: Mapping[str, Mapping[str, Any]],
    arms: Sequence[str] = ARM_NAMES,
) -> dict[str, Any]:
    reference_arm = arms[0]
    reference = _prefix_fields(runs[reference_arm])
    for arm in arms[1:]:
        current = _prefix_fields(runs[arm])
        if _jsonable(current) != _jsonable(reference):
            raise AssertionError(
                f"shared prefix mismatch for case index {case_index}: {reference_arm} != {arm}"
            )
    return reference


def _slim_run(run: Mapping[str, Any]) -> dict[str, Any]:
    slim_steps = []
    for step in run.get("steps", []):
        slim_steps.append(
            {
                "pose": int(step["pose"]),
                "action": list(step["action"]),
                "path_m": step.get("path_m", []),
                "path_skill": step.get("path_skill"),
                "evidence_gains": step.get("evidence_gains", []),
                "tail": bool(step.get("tail", False)),
                "tail_choice": step.get("tail_choice"),
            }
        )
    return {
        "strict_complete": _strict_success(run),
        "complete": bool(run.get("complete", _strict_success(run))),
        "online_stop": bool(run.get("online_stop", False)),
        "captures": _capture_count(run),
        "path_m": _path_m(run),
        "return": _mean_return(run),
        "map_max_error_mm": float(run["map_max_error_mm"]) if "map_max_error_mm" in run else None,
        "incorrect_witnesses": run.get("incorrect_witnesses", []),
        "terminal_reason": run.get("terminal_reason"),
        "scheduled_decisions": int(run.get("scheduled_decisions", 0)),
        "executed_decisions": int(run.get("executed_decisions", 0)),
        "actual_switch_count": int(run.get("actual_switch_count", 0)),
        "steps": slim_steps,
    }


def evaluate(
    *,
    seed_base: int,
    start: int,
    n: int,
    out: Path,
    checkpoints: Mapping[str, Path],
    cap: int = 80,
    n_predictive: int = 8,
    n_first: int = 3,
    switch_margin: float = 0.006,
    summary_out: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a paired seed block and write JSONL plus summary."""

    if start < 0 or n <= 0 or cap <= 0:
        raise ValueError("start must be nonnegative; n and cap must be positive")
    checkpoint_models = {name: FrozenQ(path) for name, path in checkpoints.items()}
    active_arms = list(ARM_NAMES)
    if "uniform_matched" in checkpoint_models:
        active_arms.append("uniform_matched")
    scan, support, model, planner = rl_rollouts._make_model_planner(
        n_predictive=n_predictive,
        n_first=n_first,
        switch_margin=switch_margin,
    )
    prepare = getattr(rl_rollouts, "prepare_episode", None)
    run_episode = rl_rollouts.run_episode
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cases: dict[int, dict[str, dict[str, Any]]] = {}
    used_seeds: set[int] = set()
    with out.open("w", encoding="utf-8") as stream:
        for index in range(start, start + n):
            episode_seed = _episode_seed(seed_base, index)
            if episode_seed in used_seeds:
                raise AssertionError(f"duplicate episode_seed in requested block: {episode_seed}")
            used_seeds.add(episode_seed)
            # The current collector's exact hook is
            # ``prepare_episode(scan, support, episode_seed)`` and returns
            # ``(truth, rejects, prefix)``.  Older rollouts simply omit it.
            prepared = prepare(scan, support, episode_seed) if prepare is not None else None
            full_runs: dict[str, dict[str, Any]] = {}
            for arm in active_arms:
                if arm == "rule":
                    run, _ = _run_one(
                        run_episode,
                        planner=planner,
                        scan=scan,
                        support=support,
                        model=model,
                        episode_seed=episode_seed,
                        cap=cap,
                        policy="rule",
                        q_model=None,
                        prepared_episode=prepared,
                    )
                else:
                    run, _ = _run_one(
                        run_episode,
                        planner=planner,
                        scan=scan,
                        support=support,
                        model=model,
                        episode_seed=episode_seed,
                        cap=cap,
                        policy="rl",
                        q_model=checkpoint_models[arm],
                        prepared_episode=prepared,
                    )
                full_runs[arm] = run
            prefix = _assert_shared_prefix(index, full_runs, active_arms)
            cases[episode_seed] = full_runs
            record = {
                "seed_base": int(seed_base),
                "seed_index": int(index),
                "episode_seed": int(episode_seed),
                "shared_prefix": prefix,
                "arms": {arm: _slim_run(full_runs[arm]) for arm in active_arms},
            }
            stream.write(json.dumps(_jsonable(record), sort_keys=True, allow_nan=False) + "\n")
            stream.flush()

    summary = {
        "format_version": 1,
        "method": "paired_data_quality",
        "seed_base": int(seed_base),
        "start": int(start),
        "n_requested": int(n),
        "unique_cases": int(len(cases)),
        "cap": int(cap),
        "checkpoint_paths": {
            arm: str(Path(checkpoints[arm]).expanduser().resolve())
            for arm in RL_ARMS + tuple(name for name in OPTIONAL_RL_ARMS if name in checkpoints)
        },
        "arms": {arm: _arm_metrics([case[arm] for case in cases.values()]) for arm in active_arms},
        "paired": {
            "original_vs_rule": _paired(cases, "original", "rule"),
            "uniform_vs_rule": _paired(cases, "uniform", "rule"),
            "fisher_vs_rule": _paired(cases, "fisher", "rule"),
            "fisher_vs_uniform": _paired(cases, "fisher", "uniform"),
        },
        "shared_prefix_assertion": "passed",
        "jsonl": str(out),
    }
    if "uniform_matched" in checkpoint_models:
        # This optional arm is the predeclared matched-loss sensitivity check;
        # the standard rule/original/uniform/fisher outputs above are kept
        # unchanged when the checkpoint is not supplied.
        summary["paired"]["uniform_matched_vs_rule"] = _paired(cases, "uniform_matched", "rule")
        summary["paired"]["uniform_matched_vs_uniform"] = _paired(cases, "uniform_matched", "uniform")
    if summary_out is None:
        summary_out = out.with_suffix(".summary.json")
    summary_out = summary_out.expanduser().resolve()
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", "--seed-base", dest="seed_base", type=int, default=2411202619)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1, help="number of paired cases in this shard (default: 1)")
    parser.add_argument("--cap", type=int, default=80)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="required: original, uniform, fisher; optional: uniform_matched",
    )
    parser.add_argument("--out", type=Path, required=True, help="per-case JSONL output path")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--n-predictive", "--n_predictive", dest="n_predictive", type=int, default=8)
    parser.add_argument("--n-first", "--n_first", dest="n_first", type=int, default=3)
    parser.add_argument("--switch-margin", "--switch_margin", dest="switch_margin", type=float, default=0.006)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoints = _parse_checkpoints(args.checkpoint)
    evaluate(
        seed_base=args.seed_base,
        start=args.start,
        n=args.n,
        out=args.out,
        checkpoints=checkpoints,
        cap=args.cap,
        n_predictive=args.n_predictive,
        n_first=args.n_first,
        switch_margin=args.switch_margin,
        summary_out=args.summary_out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke
    raise SystemExit(main())
