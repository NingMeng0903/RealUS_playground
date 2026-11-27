"""Paired, anatomy-agnostic probe-teacher generation pilot.

``study.py`` owns the ultrasound world and action adapter.  This file owns the
split ledger, frozen proposal libraries, and paired statistics.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_EXP = HERE.parent / "ab_search_embedding_a_20261117"
if str(REPO_EXP) not in sys.path:
    sys.path.insert(0, str(REPO_EXP))

FAMILIES = ("connection", "parallel")
CONTEXTS = ("natural", "burst")
METHODS = ("random", "direct", "evidence", "precommit")
SPLITS = {"discover": (71_000_000, 12), "qualify": (72_000_000, 10), "test": (73_000_000, 16)}
SEED_GAP = 10_000
MAX_LIBRARY = 4
BOOTSTRAP = 2000


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialise {type(value)!r}")


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, default=jsonable, allow_nan=False, indent=2) + "\n")


def fingerprint(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path: Path, rows):
    pending = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(pending, "wt", encoding="utf-8", compresslevel=3) as stream:
        for row in rows:
            stream.write(json.dumps(row, default=jsonable, allow_nan=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def read_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def import_study():
    return importlib.import_module("study")


def ensure_configured():
    study = import_study()
    if not getattr(study, "_RUNNER_CONFIGURED", False):
        study._RUNNER_SOURCE_DIGESTS = study.configure()
        study._RUNNER_CONFIGURED = True
    return study


def code_digests():
    study_path = HERE / "study.py"
    scores_path = HERE / "teacher_scores.py"
    return {"run.py": fingerprint(HERE / "run.py"), "study.py": fingerprint(study_path),
            "teacher_scores.py": fingerprint(scores_path)}


def cells():
    return [(family, context) for family in FAMILIES for context in CONTEXTS]


def jobs_for(split: str, limit=None):
    base, count = SPLITS[split]
    jobs = []
    for index, cell in enumerate(cells()):
        for offset in range(count):
            jobs.append({"family": cell[0], "context": cell[1], "seed": base + index * SEED_GAP + offset})
    return jobs if limit is None else jobs[:limit]


def call_scenario(job):
    study = ensure_configured()
    result = study.scenario((job["family"], job["context"]), int(job["seed"]))
    if not isinstance(result, dict):
        raise TypeError("study.scenario must return a dictionary")
    return result | {"family": job["family"], "context": job["context"], "seed": job["seed"]}


def call_discover(row):
    study = ensure_configured()
    result = study.discover(row)
    if not isinstance(result, dict):
        raise TypeError("study.discover must return a dictionary")
    return result | {"family": row["family"], "context": row["context"], "seed": row["seed"]}


def call_qualify(job):
    row, library = job
    study = ensure_configured()
    result = study.qualify(row, library)
    if not isinstance(result, dict):
        raise TypeError("study.qualify must return a dictionary")
    return result | {"family": row["family"], "context": row["context"], "seed": row["seed"]}


def call_confirm(job):
    row, library, fixed = job
    study = ensure_configured()
    result = study.confirm(row, library, fixed)
    if not isinstance(result, dict):
        raise TypeError("study.confirm must return a dictionary")
    return result | {"family": row["family"], "context": row["context"], "seed": row["seed"]}


def run_jobs(function, jobs, workers, label):
    if not jobs:
        return []
    if workers <= 1:
        return [function(job) for job in jobs]
    from diagnose_b_selection import run_parallel
    return run_parallel(function, jobs, workers, label)


def discover(out: Path, workers: int, limit=None):
    study = ensure_configured()
    jobs = jobs_for("discover", limit)
    scenarios = run_jobs(call_scenario, jobs, workers, "discovery scenarios")
    rows = run_jobs(call_discover, scenarios, workers, "teacher discovery")
    scenario_path, archive_path = out / "discovery_scenarios.jsonl.gz", out / "discovery.jsonl.gz"
    write_rows(scenario_path, scenarios)
    write_rows(archive_path, rows)
    dump_json(out / "discovery_config.json", {
        "split": jobs, "same_public_selector": True, "same_action_budget": True,
        "source_digests": getattr(study, "_RUNNER_SOURCE_DIGESTS", {}), "code_digests": code_digests(),
        "scenario_sha256": fingerprint(scenario_path), "archive_sha256": fingerprint(archive_path),
    })
    return rows


def scalar(value, default=0.):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def outcome_mean(outcomes, key):
    vals = [scalar(row.get(key), 0.) for row in outcomes if isinstance(row, dict)]
    return float(np.mean(vals)) if vals else 0.


def arm_outcomes(value):
    if not isinstance(value, dict):
        return value if isinstance(value, list) else []
    value = value.get("outcomes", value.get("metrics", []))
    return [value] if isinstance(value, dict) else value if isinstance(value, list) else []


def action_outcomes(row, action):
    table = row.get("qualification", {})
    if not isinstance(table, dict):
        return []
    return arm_outcomes(table.get(action))


def proposal_ids(row, method):
    proposals = row.get("proposals", {})
    values = proposals.get(method, []) if isinstance(proposals, dict) else []
    return [str(item) for item in values]


def paired_stats(outcomes, incumbent):
    n = min(len(outcomes), len(incumbent))
    if not n:
        return {"n": 0, "complete_delta": 0., "wrong_delta": 0., "identity_delta": 0.,
                "tcp_delta": 0., "turn_delta": 0., "seconds_delta": 0., "wins": 0}
    keys = ("corrected_complete", "final_wrong", "new_identity_mistake", "tcp_mm", "turn_degs", "seconds")
    d = {key: float(np.mean([scalar(outcomes[i].get(key)) - scalar(incumbent[i].get(key))
                             for i in range(n)])) for key in keys}
    wins = sum(scalar(outcomes[i].get("corrected_complete")) > scalar(incumbent[i].get("corrected_complete"))
               and scalar(outcomes[i].get("final_wrong")) <= scalar(incumbent[i].get("final_wrong"))
               for i in range(n))
    return {"n": n, "complete_delta": d["corrected_complete"], "wrong_delta": d["final_wrong"],
            "identity_delta": d["new_identity_mistake"], "tcp_delta": d["tcp_mm"],
            "turn_delta": d["turn_degs"], "seconds_delta": d["seconds"], "wins": int(wins)}


def freeze_method(rows, method):
    scored = []
    for row in rows:
        incumbent = str(row.get("incumbent", "mpc"))
        base = action_outcomes(row, incumbent)
        for action in proposal_ids(row, method):
            stats = paired_stats(action_outcomes(row, action), base)
            if not stats["n"]:
                continue
            # Discovery ranking only.  Holdout rows are never used here.
            score = (stats["complete_delta"] - 1.5 * max(0., stats["wrong_delta"])
                     - .002 * max(0., stats["tcp_delta"]) - .02 * max(0., stats["seconds_delta"]))
            scored.append((score, stats["complete_delta"], -stats["wrong_delta"],
                           -stats["tcp_delta"], str(action), row["seed"], stats,
                           bool(stats["complete_delta"] > 0. and stats["wrong_delta"] <= 0.
                                and stats["wins"] > 0)))
    scored.sort(reverse=True)
    positive = [item for item in scored if item[7]]
    slate = positive if positive else []
    selected = []
    for item in slate:
        if item[4] not in selected:
            selected.append(item[4])
        if len(selected) == MAX_LIBRARY:
            break
    ranking = [{"action": item[4], "seed": item[5], "score": item[0], "positive": item[7], "stats": item[6]}
               for item in scored[:MAX_LIBRARY * 3]]
    return selected, ranking


def freeze(out: Path):
    rows = read_rows(out / "discovery.jsonl.gz")
    libraries, rankings = {}, {}
    for method in METHODS:
        libraries[method], rankings[method] = freeze_method(rows, method)
    value = {"libraries": libraries, "fixed": {family: "mpc" for family in FAMILIES},
             "rankings": rankings, "source_sha256": fingerprint(out / "discovery.jsonl.gz"),
             "max_library": MAX_LIBRARY, "families": list(FAMILIES), "contexts": list(CONTEXTS),
             "code_digests": code_digests()}
    dump_json(out / "frozen.json", value)
    return value


def qualify(out: Path, workers: int, limit=None):
    frozen = json.loads((out / "frozen.json").read_text())
    jobs = jobs_for("qualify", limit)
    scenarios = run_jobs(call_scenario, jobs, workers, "qualification scenarios")
    rows = run_jobs(call_qualify, [(row, frozen["libraries"]) for row in scenarios],
                    workers, "qualification replay")
    path = out / "qualification.jsonl.gz"
    write_rows(path, rows)
    fixed, fixed_ranking = choose_fixed(rows)
    dump_json(out / "qualification_config.json", {
        "split": jobs, "source_frozen_sha256": fingerprint(out / "frozen.json"),
        "archive_sha256": fingerprint(path), "same_libraries": True,
        "fixed_choices": fixed, "fixed_rankings": fixed_ranking, "code_digests": code_digests(),
    })
    return rows


def choose_fixed(rows):
    by_family = defaultdict(lambda: defaultdict(list))
    available_counts = defaultdict(lambda: defaultdict(int))
    family_counts = defaultdict(int)
    for row in rows:
        family_counts[row["family"]] += 1
        available = set(row.get("available", ()))
        table = row.get("qualification", row.get("arms", {}))
        if not isinstance(table, dict):
            continue
        for action, values in table.items():
            if str(action) == "mpc" or str(action) not in available:
                continue
            outcomes = arm_outcomes(values)
            if outcomes:
                by_family[row["family"]][str(action)].extend(outcomes)
                available_counts[row["family"]][str(action)] += 1
    fixed, ranking = {}, {}
    for family in FAMILIES:
        eligible = {action for action, count in available_counts[family].items()
                    if count >= .8 * max(1, family_counts[family])}
        candidates = []
        for action, outcomes in by_family[family].items():
            if eligible and action not in eligible:
                continue
            candidates.append((outcome_mean(outcomes, "corrected_complete"),
                               -outcome_mean(outcomes, "final_wrong"),
                               -outcome_mean(outcomes, "tcp_mm"), action))
        candidates.sort(reverse=True)
        fixed[family] = candidates[0][3] if candidates else "wait"
        ranking[family] = [{"action": x[3], "complete": x[0], "wrong": -x[1], "tcp_mm": -x[2]}
                           for x in candidates]
    return fixed, ranking


def test(out: Path, workers: int, limit=None):
    frozen = json.loads((out / "frozen.json").read_text())
    if frozen.get("source_sha256") != fingerprint(out / "discovery.jsonl.gz"):
        raise ValueError("frozen proposal library does not match the discovery archive")
    qualified = read_rows(out / "qualification.jsonl.gz")
    qualification_config = json.loads((out / "qualification_config.json").read_text())
    frozen_sha = fingerprint(out / "frozen.json")
    qualification_sha = fingerprint(out / "qualification.jsonl.gz")
    if qualification_config.get("source_frozen_sha256") != frozen_sha:
        raise ValueError("qualification was generated from a different frozen proposal library")
    if qualification_config.get("archive_sha256") != qualification_sha:
        raise ValueError("qualification archive changed after its manifest was written")
    if qualification_config.get("code_digests") != code_digests():
        raise ValueError("qualification was generated with different runner or adapter code")
    fixed = qualification_config.get("fixed_choices", {})
    fixed_ranking = qualification_config.get("fixed_rankings", {})
    if set(fixed) != set(FAMILIES):
        raise ValueError("qualification manifest has no fixed choice for every family")
    jobs = jobs_for("test", limit)
    scenarios = run_jobs(call_scenario, jobs, workers, "holdout scenarios")
    rows = run_jobs(call_confirm, [(row, frozen["libraries"], fixed) for row in scenarios],
                    workers, "holdout paired closed loops")
    required_arms = {"mpc", "stock_mpc", "best_fixed", "public_stock", "random",
                     "direct", "evidence", "precommit", "existing_b"}
    required_fields = {"corrected_complete", "final_wrong", "tcp_mm", "turn_degs", "seconds"}
    for row in rows:
        missing = required_arms - set(row.get("arms", {}))
        if missing:
            raise ValueError(f"holdout row {row.get('seed')} is missing arms {sorted(missing)}")
        for arm in required_arms:
            outcomes = arm_outcomes(row["arms"][arm])
            if not outcomes or any(not required_fields <= set(outcome) for outcome in outcomes):
                raise ValueError(f"holdout row {row.get('seed')} has incomplete metrics for {arm}")
    path = out / "holdout.jsonl.gz"
    write_rows(path, rows)
    report = summarize(rows)
    report.update({"fixed_choices": fixed, "fixed_rankings": fixed_ranking,
                   "source_frozen_sha256": frozen_sha, "archive_sha256": fingerprint(path),
                   "code_digests": code_digests()})
    dump_json(out / "holdout_summary.json", report)
    return report


def mean_arm(arm):
    outcomes = arm_outcomes(arm)
    keys = ("corrected_complete", "final_wrong", "new_identity_mistake", "new_image_and_complete",
            "tcp_mm", "turn_degs", "seconds", "frames", "fresh_query_confirmed")
    return {key: outcome_mean(outcomes, key) for key in keys}


def bootstrap_difference(pairs, key, seed=18_111):
    values = np.asarray([b.get(key, 0.) - a.get(key, 0.) for a, b in pairs], dtype=float)
    if not len(values):
        return {"n": 0, "mean": 0., "lower": 0., "upper": 0.}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    return {"n": int(len(values)), "mean": float(values.mean()),
            "lower": float(np.quantile(means, .025)), "upper": float(np.quantile(means, .975))}


def summarize_core(rows):
    arms = sorted({arm for row in rows for arm in row.get("arms", {})})
    report = {"arms": {}, "paired": {}, "by_family_context": {}}
    for arm in arms:
        values = [mean_arm(row["arms"][arm]) for row in rows if arm in row.get("arms", {})]
        report["arms"][arm] = {
            "scenes": len(values),
            "corrected_complete": float(np.mean([v["corrected_complete"] for v in values])) if values else 0.,
            "final_wrong": float(np.mean([v["final_wrong"] for v in values])) if values else 0.,
            "new_identity_mistake": float(np.mean([v["new_identity_mistake"] for v in values])) if values else 0.,
            "new_image_and_complete": float(np.mean([v["new_image_and_complete"] for v in values])) if values else 0.,
            "tcp_mm": float(np.mean([v["tcp_mm"] for v in values])) if values else 0.,
            "turn_degs": float(np.mean([v["turn_degs"] for v in values])) if values else 0.,
            "seconds": float(np.mean([v["seconds"] for v in values])) if values else 0.,
        }
    reference = "mpc"
    for arm in arms:
        if arm == reference:
            continue
        pairs = []
        for row in rows:
            if arm not in row.get("arms", {}) or reference not in row.get("arms", {}):
                continue
            pairs.append((mean_arm(row["arms"][reference]), mean_arm(row["arms"][arm])))
        report["paired"][f"{arm}_vs_{reference}"] = {
            key: bootstrap_difference(pairs, key)
            for key in ("corrected_complete", "final_wrong", "new_identity_mistake", "new_image_and_complete",
                        "tcp_mm", "turn_degs", "seconds")
        }
    return report


def paired_against(rows, left, right):
    pairs = [(mean_arm(row["arms"][right]), mean_arm(row["arms"][left]))
             for row in rows if left in row.get("arms", {}) and right in row.get("arms", {})]
    return {key: bootstrap_difference(pairs, key)
            for key in ("corrected_complete", "final_wrong", "new_identity_mistake",
                        "new_image_and_complete", "tcp_mm", "turn_degs", "seconds")}


def summarize(rows):
    report = summarize_core(rows)
    report["decision_comparisons"] = {}
    for source in ("evidence", "precommit", "random", "direct", "public_stock"):
        report["decision_comparisons"][f"{source}_vs_best_fixed"] = paired_against(rows, source, "best_fixed")
    report["decision_comparisons"]["evidence_vs_precommit"] = paired_against(rows, "evidence", "precommit")
    report["decision_comparisons"]["evidence_vs_random"] = paired_against(rows, "evidence", "random")
    report["decision_comparisons"]["direct_vs_best_fixed"] = paired_against(rows, "direct", "best_fixed")
    for family in FAMILIES:
        for context in CONTEXTS:
            subset = [row for row in rows if row.get("family") == family and row.get("context") == context]
            if subset:
                report["by_family_context"][f"{family}/{context}"] = summarize_core(subset)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("discover", "freeze", "qualify", "test", "all"))
    parser.add_argument("--out", type=Path, default=HERE / "results" / "teacher_generation_v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="optional deterministic smoke limit per stage")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("discover", "all"):
        discover(args.out, args.workers, args.limit)
    if args.stage in ("freeze", "all"):
        freeze(args.out)
    if args.stage in ("qualify", "all"):
        qualify(args.out, args.workers, args.limit)
    if args.stage in ("test", "all"):
        test(args.out, args.workers, args.limit)


if __name__ == "__main__":
    main()
