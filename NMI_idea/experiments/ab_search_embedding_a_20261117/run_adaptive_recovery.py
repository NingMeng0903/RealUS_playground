"""Reproduce the fixed-pose intervention dataset and paired public teacher study."""

import argparse
from collections import Counter
from dataclasses import replace
import gzip
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np

from recovery_world import FAMILIES, RECOVERY, RecoveryWorld, Scene, noise_tape
from adaptive_controller import (CENTER_MM, EVIDENCE_MODES, FRAMES_PER_BLOCK, LEGACY_EVIDENCE, QUALIFIED_EVIDENCE,
                                 AdaptiveController, FreeCandidate, final_metrics, station_bin)
from adaptive_teacher import cem_discover, choose_public, fit_nominal_primitive, replace_shape


HERE = Path(__file__).resolve().parent
SECTIONS = HERE.parent / "ab_streaming_prodmp_two_vessel_20261113" / "SECTIONS.json"
SPLIT_SEEDS = {"discover": 6_210_000, "paired": 7_210_000}
MARGINS = {"tcp_mm": 2., "turn_degs": 2., "seconds": .10, "frames": 1.}
OUTCOMES = ("corrected_complete", "old_strict_complete", "final_wrong", "ever_wrong", "new_centered_mistake",
            "fresh_query_confirmed", "target_reappeared", "tcp_mm", "turn_degs", "seconds", "frames",
            "feedback_revisions", "wrong_certificates", "wrong_card_seconds", "retractions")
WORKERS = {}


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"unsupported JSON value {type(obj)}")


def shared_prefix(controller, scene):
    state, record = controller.initial_scan(scene, detailed=True)
    for _ in range(2):
        action = controller.choose_mpc(state, expanded=False)
        if action is None:
            break
        controller.execute(scene, state, record, action)
    return state, record


def init_worker(fitted, *, evidence_mode=LEGACY_EVIDENCE):
    sections = json.loads(SECTIONS.read_text())
    with np.load(RECOVERY / "results" / "prodmp_weights.npz") as weights:
        coefficients = dict(weights)
    for family in FAMILIES:
        WORKERS[family] = AdaptiveController(RecoveryWorld(family), sections, coefficients, fitted,
                                              evidence_mode=evidence_mode)


def root_report(root, prefix):
    active = {(w.frame, j) for j, w in enumerate(root.ledger) if w is not None}
    wrong_bins = [e["bin"] for e in prefix.certificates if not e["identity_ok"] and (e["frame"], e["bin"]) in active]
    truth_in_packet = 0 in prefix.packets[-1]["roles_truth"]
    if wrong_bins:
        kind = "confident_wrong"
    elif not truth_in_packet:
        kind = "target_invisible"
    elif root.slot < 0:
        kind = "visible_unassociated"
    else:
        kind = "tracking"
    return {"category": kind, "wrong_bins": wrong_bins, "public_trigger": root.trigger(),
            "target_in_packet": truth_in_packet, "query_bin": root.query_bin(), "frame": root.frame}


def trace(record, *, after_frame=-1):
    public_keys = ("frame", "s", "x", "q", "y", "association", "accepted_slot", "belief", "qualified_coverage")
    packets = [p for p in record.packets if p["frame"] > after_frame]
    return {"public": {"packets": [{k: p[k] for k in public_keys if k in p} for p in packets],
                       "executed_motion": [p for p in record.motion if p[0] > .1 * after_frame - 1e-7],
                       "decisions": [d for d in record.decisions if d["frame"] >= after_frame],
                       "references": [r for r in record.references if r["frame"] >= after_frame],
                       "retractions": [r for r in record.retractions if r["frame"] > after_frame]},
            "evaluator": {"packet_roles": [[p["frame"], p["roles_truth"], p["target_truth"]] for p in packets],
                          "certificates": [a for a in record.certificates if a["frame"] > after_frame],
                          "legacy_accepts": [a for a in record.accepts if a["frame"] > after_frame]}}


def public_root(root):
    return {k: v for k, v in vars(root).items() if k != "ledger"} | {
        "ledger": [None if witness is None else vars(witness) for witness in root.ledger]}


def tape_for(latent, seed, replica, purpose):
    return Scene(latent, noise_tape(np.random.default_rng([seed, replica, purpose])))


def score_outcome(state, record, root):
    metrics = final_metrics(state, record, root_frame=root.frame, query_bin=root.query_bin())
    suffix = [p for p in record.packets if p["frame"] > root.frame]
    metrics["new_centered_mistake"] = any(
        p["accepted_slot"] >= 0 and p["roles_truth"][p["accepted_slot"]] != 0
        and abs(p["y"][p["accepted_slot"]][0]) <= CENTER_MM for p in suffix)
    early = [p for p in suffix if p["frame"] <= root.frame + FRAMES_PER_BLOCK
             and station_bin(p["s"]) == root.query_bin()]
    metrics["first_block_true_image"] = any(0 in p["roles_truth"] for p in early)
    return metrics


def real_trial(controller, root, prefix, scene, leading, *, expanded=True, detailed=False):
    state, record = replace(root, belief=root.belief.copy()), prefix.copy()
    record.detailed = detailed
    for action in leading:
        if controller.can_continue(state) and action is not None:
            controller.execute(scene, state, record, action)
    controller.tail(scene, state, record, expanded=expanded, open_blocks=max(0, 2 - len(leading)))
    return score_outcome(state, record, root), record


def archive_discovery(job):
    family, seed, config = job
    controller = WORKERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    base, base_search, ranked, candidates = cem_discover(
        controller, root, scene.latent, seed, population=config.population,
        iterations=config.iterations, samples=config.search_samples)
    finalists = {"mpc": base, "wait": controller.options_shared(root).get("wait")}
    if ranked:
        finalists["cem"] = ranked[0][0]
        negative_pool = [item for item in ranked[1:] if item[1]["expected"]["corrected_complete"]
                         < ranked[0][1]["expected"]["corrected_complete"]]
        if negative_pool:
            finalists["nearby_failure"] = min(negative_pool, key=lambda item: np.linalg.norm(
                np.asarray(item[0].request) - np.asarray(finalists["cem"].request)))[0]
    verification = {}
    for label, action in finalists.items():
        if action is None:
            continue
        trials = []
        for replica in range(config.replicas):
            future = tape_for(scene.latent, seed, replica, 927)
            metrics, run = real_trial(controller, root, prefix, future, [action], detailed=replica == 0)
            row = {"replica": replica, "metrics": metrics}
            if replica == 0:
                row["trace"] = trace(run, after_frame=root.frame)
            trials.append(row)
        verification[label] = trials
    example = None
    if "cem" in verification and "mpc" in verification:
        counts = lambda label: (sum(r["metrics"]["corrected_complete"] for r in verification[label]),
                                sum(r["metrics"]["new_centered_mistake"] for r in verification[label]))
        good, wrong = counts("cem")
        base_good, base_wrong = counts("mpc")
        gains = (good > base_good and good >= np.ceil(2 * config.replicas / 3) and wrong <= base_wrong)
        if gains:
            action = finalists["cem"]
            example = {"request": action.request,
                       "nominal_x_vx": np.column_stack((action.lateral.x, action.lateral.v)).tolist()}
    return {"family": family, "seed": seed, "checkpoint": root_report(root, prefix),
            "public_root": public_root(root), "prefix": trace(prefix), "hidden_index_evaluator": scene.latent,
            "mpc_search": base_search, "candidates": candidates, "verification": verification, "example": example}


def paired_case(job):
    family, seed, config = job
    controller = WORKERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    initial, initial_report = choose_public(controller, root, samples=config.public_samples, remaining_open=1)
    rows = []
    for replica in range(config.replicas):
        future = tape_for(scene.latent, seed, replica, 1101)
        outputs, traces = {}, {}

        for name, extended in (("mpc_original_menu", False), ("mpc_shared_menu", True)):
            outputs[name], record = real_trial(controller, root, prefix, future, [], expanded=extended,
                                                detailed=replica == 0)
            if replica == 0:
                traces[name] = trace(record, after_frame=root.frame)

        state, record = replace(root, belief=root.belief.copy()), prefix.copy()
        record.detailed = replica == 0
        after_first = None
        pair_error = 0.
        replaced, requested_fit = False, False
        if initial is not None and controller.can_continue(state):
            controller.execute(future, state, record, initial)
            after_first = public_root(state)
            single_state, single_record = replace(state, belief=state.belief.copy()), record.copy()
            controller.tail(future, single_state, single_record, open_blocks=1)
            outputs["public_first_only"] = score_outcome(single_state, single_record, root)
            if controller.learned is not None:
                fitted, replaced = replace_shape(controller, root, initial)
                requested_fit = isinstance(initial, FreeCandidate)
                pair_error = float(np.max(abs(fitted.lateral.x - initial.lateral.x)))
                outputs["prodmp_same_first"], fit_record = real_trial(
                    controller, root, prefix, future, [fitted], detailed=replica == 0)
                if replica == 0:
                    traces["prodmp_same_first"] = trace(fit_record, after_frame=root.frame)
            if controller.can_continue(state):
                next_action, next_report = choose_public(controller, state, samples=config.public_samples,
                                                         remaining_open=0)
                if next_action is not None:
                    controller.execute(future, state, record, next_action)
            else:
                next_report = None
        else:
            outputs["public_first_only"] = outputs["mpc_shared_menu"]
            next_report = None
        controller.tail(future, state, record)
        outputs["public_adaptive"] = score_outcome(state, record, root)
        if replica == 0:
            traces["public_adaptive"] = trace(record, after_frame=root.frame)
        rows.append({"replica": replica, "outcomes": outputs, "public_state_after_first": after_first,
                     "second_decision": next_report, "fit_requested": requested_fit,
                     "fit_executed": replaced, "fit_max_nominal_error_mm": pair_error,
                     "traces": traces})
    return {"family": family, "seed": seed, "checkpoint": root_report(root, prefix),
            "hidden_index_evaluator": scene.latent, "public_root": public_root(root), "prefix": trace(prefix),
            "first_decision": initial_report, "verification": rows}


def paired_statistics(cases, arms, fields, extract):
    data = np.asarray([[[np.mean([replica[arm][metric] for replica in extract(case)]) for metric in fields]
                       for arm in arms] for case in cases], dtype=float)
    centers = data.mean(axis=0)
    differences = data[:, 1:] - data[:, :1]
    draws = np.random.default_rng(3102).integers(len(cases), size=(10_000, len(cases)))
    low, high = np.quantile(differences[draws].mean(axis=1), [.025, .975], axis=0)
    return {"worlds": len(cases), "replicas_per_world": [len(extract(c)) for c in cases[:1]][0],
            "means": {arm: dict(zip(fields, row)) for arm, row in zip(arms, centers)},
            "vs_reference": {arm: {key: {"delta": float(centers[i + 1, j] - centers[0, j]),
                                          "ci95": [float(low[i, j]), float(high[i, j])]}
                                  for j, key in enumerate(fields)} for i, arm in enumerate(arms[1:])}}


def summarize_discovery(rows, controller, out):
    examples = [row["example"] for row in rows if row["example"] is not None]
    coefficients = fit_nominal_primitive(controller.primitive, examples)
    if coefficients is not None:
        np.save(out / "prodmp_nominal_fit.npy", coefficients)
    cohort = [row for row in rows if "mpc" in row["verification"] and "cem" in row["verification"]]
    def extract(row):
        return [{arm: row["verification"][arm][i]["metrics"] for arm in ("mpc", "cem")}
                for i in range(len(row["verification"]["mpc"]))]
    result = {"worlds": len(rows), "paired_worlds": len(cohort), "positives_for_nominal_fit": len(examples),
              "fitted": coefficients is not None, "checkpoint_types": dict(Counter(r["checkpoint"]["category"] for r in rows))}
    if cohort:
        result["comparison"] = paired_statistics(cohort, ("mpc", "cem"), OUTCOMES, extract)
    return result


def summarize_paired(rows):
    common = ["mpc_shared_menu", "mpc_original_menu", "public_first_only", "public_adaptive"]
    def extract(case):
        return [replica["outcomes"] for replica in case["verification"]]
    groups = {family: [row for row in rows if row["family"] == family] for family in FAMILIES}
    natural = groups["connection"] + groups["parallel"]
    result = {"cohorts": {family: paired_statistics(cases, common, OUTCOMES, extract)
                          for family, cases in groups.items() if cases},
              "root_counts": {family: dict(Counter(row["checkpoint"]["category"] for row in cases))
                              for family, cases in groups.items()},
              "execution_equivalence_margins": MARGINS}
    if natural:
        study = paired_statistics(natural, common, OUTCOMES, extract)
        result["natural"] = study
        delta = study["vs_reference"]["public_adaptive"]
        equivalent = {key: delta[key]["ci95"][0] >= -margin and delta[key]["ci95"][1] <= margin
                      for key, margin in MARGINS.items()}
        result["gate"] = {"success": delta["corrected_complete"]["delta"] >= .05
                           and delta["corrected_complete"]["ci95"][0] > 0., "cost": equivalent,
                          "final_wrong": delta["final_wrong"]["ci95"][1] <= .02,
                          "new_wrong": delta["new_centered_mistake"]["ci95"][1] <= .02}
        result["second_step_vs_first_only"] = paired_statistics(
            natural, ("public_first_only", "public_adaptive"), OUTCOMES, extract)
    repairs = [row for row in rows if row["checkpoint"]["wrong_bins"]]
    result["preexisting_error_cases"] = {"worlds": len(repairs)}
    if repairs:
        result["preexisting_error_cases"]["arms"] = {
            arm: {"repaired_any_bin": sum(bool(trial["outcomes"][arm]["repaired_root_bins"])
                                           for case in repairs for trial in case["verification"]),
                  "completed": sum(trial["outcomes"][arm]["corrected_complete"]
                                   for case in repairs for trial in case["verification"]),
                  "trials": sum(len(case["verification"]) for case in repairs)} for arm in common}
    fit_cases = [row for row in rows if all(r["fit_requested"] for r in row["verification"])]
    if fit_cases:
        result["same_request_prodmp"] = paired_statistics(
            fit_cases, ("public_first_only", "prodmp_same_first"), OUTCOMES, extract)
        result["same_request_prodmp"]["executed_fit"] = sum(r["fit_executed"] for c in fit_cases for r in c["verification"])
        result["same_request_prodmp"]["max_nominal_fit_error_mm"] = max(
            r["fit_max_nominal_error_mm"] for case in fit_cases for r in case["verification"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("discover", "paired"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=8, help="independent worlds per requested family")
    parser.add_argument("--families", choices=FAMILIES, nargs="+", default=FAMILIES)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--search-samples", type=int, default=3)
    parser.add_argument("--public-samples", type=int, default=6)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--evidence-mode", choices=EVIDENCE_MODES, default=QUALIFIED_EVIDENCE)
    config = parser.parse_args()
    if min(config.cases, config.replicas, config.population, config.iterations, config.search_samples,
           config.public_samples, config.jobs) <= 0 or config.offset < 0:
        parser.error("counts must be positive and offset nonnegative")
    if len(config.families) != len(set(config.families)) or config.offset + config.cases > 90_000:
        parser.error("families must be distinct and each case namespace has 90,000 entries")
    config.out.mkdir(parents=True, exist_ok=True)
    fitted = np.load(config.fit) if config.fit else None
    init_worker(fitted, evidence_mode=config.evidence_mode)
    jobs = [(family, SPLIT_SEEDS[config.phase] + 100_000 * FAMILIES.index(family) + config.offset + index, config)
            for family in config.families for index in range(config.cases)]
    runner = archive_discovery if config.phase == "discover" else paired_case
    started = perf_counter()
    cache = []
    with gzip.open(config.out / "cases.jsonl.gz", "wt", compresslevel=3) as stream:
        if config.jobs == 1:
            for row in map(runner, jobs):
                stream.write(json.dumps(row, default=jsonable, allow_nan=False, separators=(",", ":")) + "\n")
                cache.append(row)
        else:
            with get_context("fork").Pool(min(config.jobs, len(jobs))) as pool:
                for row in pool.imap(runner, jobs):
                    stream.write(json.dumps(row, default=jsonable, allow_nan=False, separators=(",", ":")) + "\n")
                    cache.append(row)
                    if len(cache) % 8 == 0:
                        print(f"{config.phase} {len(cache)}/{len(jobs)}", flush=True)
    summary = (summarize_discovery(cache, WORKERS[config.families[0]], config.out) if config.phase == "discover"
               else summarize_paired(cache))
    summary["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()}
    summary["elapsed_s"] = perf_counter() - started
    (config.out / "summary.json").write_text(json.dumps(summary, default=jsonable, indent=2, allow_nan=False) + "\n")
    print(f"{config.phase} complete in {summary['elapsed_s']:.1f}s: {config.out}", flush=True)


if __name__ == "__main__":
    main()
