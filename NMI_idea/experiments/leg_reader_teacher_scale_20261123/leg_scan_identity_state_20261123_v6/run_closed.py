"""Run development teaching and frozen identity-state action comparisons."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from v6_data import BASE, DEFAULT_PACK, ROOT, ROUTES, SEED_BASES, STAGES, atomic_json, collect, digest, read_json
from live_programs import (PROGRAMS, REFERENCE, SCORE_STREAMS, SEARCH_STREAM_OFFSETS, TEACH_ACTIONS, case_rollout,
                           init_live, pack_rollouts)
from paired_value import fit_student, select_teacher
from reader_study import frozen_reader, warm_bundle
from route_metrics import totals
from scale_worlds import FUTURE_SEEDS
import numpy as np


ALL_METRICS = ("strict", "joint", "wrong_branch", "wrong_region_count", "old_pool_correct", "suffix_valid", "time_s", "travel_mm")
DESIGN = {"candidate": "locked v5 public anchor vs common MPC on whole route",
          "identity": "matched objects_dynamic vs unary_dynamic",
          "teacher": "adaptive search vs locked v5 public anchor on preselected test subset",
          "pass": "proof mean strict 95% lower bound positive, wrong-branch 95% upper bound <= .03, actual strict delta nonnegative",
          "false_branch_noninferiority_margin": .03}


def resample(difference, *, seed=180_411_381):
    values = np.asarray(difference, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), (3500, len(values)))
    return {"delta": float(values.mean()), "lower95": float(np.quantile(values[samples].mean(axis=1), .025)),
            "upper95": float(np.quantile(values[samples].mean(axis=1), .975))}


def contrast(active, reference):
    result = {}
    for metric in ALL_METRICS:
        difference = np.asarray([[float(a[band][metric]) - float(b[band][metric]) for band in SCORE_STREAMS]
                                 for a, b in zip(active, reference, strict=True)])
        comparisons = {band: resample(difference[:, i]) | {"added_or_increased": int((difference[:, i] > 0).sum()),
                       "removed_or_decreased": int((difference[:, i] < 0).sum())} for i, band in enumerate(SCORE_STREAMS)}
        comparisons["independent_tape_expected"] = resample(difference[:, 1:].mean(axis=1))
        result[metric] = comparisons
    return result


def metric_section(rows):
    actual = [x["actual"] for x in rows]; overall = totals(actual)
    section = []
    for region in range(6):
        labels = np.asarray([r["selected_ids"][region] for r in actual])
        seen = np.asarray([r["capture"][region] for r in actual])
        section.append({"region": region + 1, "captured": int(seen.sum()), "delivered_correct": int((labels == 0).sum()),
                        "delivered_other_branch": int((labels > 0).sum()), "artifact_delivered": int(((labels < 0) & (labels != -99)).sum()),
                        "seen_and_abstained": int((seen & (labels == -99)).sum())})
    return overall | {"regions": section, "joint": sum(r["joint"] for r in actual),
                      "old_pool_correct": sum(r["old_pool_correct"] for r in actual), "suffix_valid": sum(r["suffix_valid"] for r in actual),
                      "mean_time_s": float(np.mean([v["time_s"] for v in actual])),
                      "mean_travel_mm": float(np.mean([v["travel_mm"] for v in actual])),
                      "mean_correct_mm": float(np.mean([v["mm"] for v in actual])),
                      "mean_longest_mm": float(np.mean([v["longest_mm"] for v in actual])),
                      "independent_strict": {stream: sum(r[stream]["strict"] for r in rows) for stream in SCORE_STREAMS[1:]}}


def retrieve(rows, action):
    return [{band: r["outcomes"][band][r[action] if action in ("teacher", "student") else action] for band in SCORE_STREAMS}
            for r in rows]


def gate(compare):
    return (compare["strict"]["independent_tape_expected"]["lower95"] > 0. and
            compare["wrong_branch"]["independent_tape_expected"]["upper95"] <= DESIGN["false_branch_noninferiority_margin"] and
            compare["strict"]["actual"]["delta"] >= 0.)


def trace_summary(rows):
    tr = [r["outcomes"]["actual"]["objects_dynamic"] for r in rows if r["trigger"]]
    by = [r["outcomes"]["actual"] for r in rows if r["trigger"]]
    same = [i for arm in tr for i in arm["trace"] if i.get("two_hypotheses") and i.get("executed", True)]
    stopped = [r["objects_stop"] for r in by]; go = [r["objects_no_stop"] for r in by]
    rf = [r["rejoin_trace"] for r in stopped]
    evidence = [a["endpoint_evidence"] for a in stopped if a["endpoint_evidence"] is not None]
    times = np.asarray([a["time_s"] - b["time_s"] for a,b in zip(stopped, go, strict=True)])
    return {"triggered": len(tr), "same_geometry_comparisons": len(same),
            "same_geometry_window_changed_by_identity_posterior": sum(i["same_geometry_unary"] != i["chosen"] for i in same),
            "actual_object_vs_unary_forward_windows_differ": sum(
                [z[1] for z in a["objects_dynamic"]["forward_path"]] != [z[1] for z in a["unary_dynamic"]["forward_path"]] for a in by),
            "dynamic_landmark_not_nominal": sum(a["requested_target_station"] != r["anchor"]["station"]
                for a, r in zip(stopped, [r for r in rows if r["trigger"]], strict=True)),
            "natural_forward_stops": sum(x["completion_stop"] is not None for x in stopped),
            "natural_prodmp_early_stops": sum(x["after_review"] is not None and x["after_review"]["early"] for x in stopped),
            "target_relation_observed_at_endpoint": sum(x.get("distinct_competing_or_relation", False) for x in evidence),
            "endpoint_completion_reasons": dict(Counter(x["reason"] for x in evidence)),
            "physical_time_saved_cases": int((times < -1.e-6).sum()),
            "physical_time_increased_cases": int((times > 1.e-6).sum()),
            "rejoin_choices_differ_from_local_mpc": sum(i["chosen"] != i["mpc"] for plans in rf for i in plans)}


def report(test):
    result = {}
    for route in ROUTES:
        rows = [r for r in test if r["case"]["task"] == route]
        runs = {m: retrieve(rows, m) for m in (*PROGRAMS, "student")}
        comparisons = {m: {ref: contrast(a, runs[ref]) for ref in ("mpc", REFERENCE) if ref != m} for m,a in runs.items()}
        comparison_objects = contrast(runs["objects_dynamic"], runs["unary_dynamic"])
        effect_stop = contrast(runs["objects_stop"], runs["objects_no_stop"])
        subset = [r for r in rows if r["teacher"] is not None]
        selected = retrieve(subset, "teacher")
        reference = retrieve(subset, REFERENCE)
        teach_comparison = contrast(selected, reference)
        changed = 0
        for r in subset:
            shortlist = r["adaptive_actions"]
            if len(shortlist) > 1:
                provisional, _ = select_teacher({a: s[:2] for a,s in r["paired_samples"].items()}, REFERENCE)
                changed += provisional != r["teacher"]
        result[route] = {"cases": len(rows), "scores": {a: metric_section(r) for a,r in runs.items()},
                         "contrasts": comparisons, "identity_vs_unary": comparison_objects,
                         "real_stop_vs_no_stop": effect_stop,
                         "teacher": {"cases": len(subset), "scores": metric_section(selected), "vs_locked_v5": teach_comparison,
                                      "selected": dict(Counter(r["teacher"] for r in subset)),
                                      "adaptive_layouts": sum(len(r["adaptive_actions"]) > 1 for r in subset),
                                      "choice_changed_with_replication": changed},
                         "student_choices": dict(Counter(r["student"] for r in rows)), "executed_mechanics": trace_summary(rows),
                         "gates": {"v5_candidate_confirmation": gate(comparisons[REFERENCE]["mpc"]),
                                   "identity_extra_contribution": gate(comparison_objects), "teacher_large_distillation": gate(teach_comparison)}}
    return result


def verify_rosters(out, dev, tests):
    layouts = [r["layout_id"] for r in dev + tests]
    from v6_data import load_shard
    for path in sorted((out / "reader_fresh").glob("*/commit.json")):
        data, rows = load_shard(path.parent)
        if digest(data) != read_json(path)["data_sha256"]:
            raise RuntimeError("Fresh reader test changed after freezing")
        layouts.extend(r["layout_id"] for r in rows)
    if len(set(layouts)) != len(layouts):
        raise RuntimeError("Fresh stages reused anatomy")
    return len(layouts)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run/confirmed")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    out, atlas = args.out.resolve(), args.atlas_pack.resolve()
    fpath, path = out / "reader_freeze.json", out / "closed_freeze.json"
    if not (out / "contracts.json").exists():
        raise RuntimeError("Prefix order and pinned v5 execution contracts must pass first")
    freeze = read_json(fpath)
    if freeze["usability_sha256"] != digest(out / "usability_bundle.json"):
        raise RuntimeError("Frozen usability estimate changed")
    if any(digest(ROOT / name) != hash_value for name, hash_value in freeze["reader_source_sha256"].items()):
        raise RuntimeError("Frozen reader source changed")
    rules = {"experiment": ROOT.name, "base_commit": BASE, "atlas_sha256": digest(atlas), "reader_freeze_sha256": digest(fpath),
             "source_sha256": {n: digest(ROOT / n) for n in ("live_programs.py", "anchor_logic.py", "paired_value.py", "run_closed.py")},
             "gates": DESIGN, "seeds": SEED_BASES, "stages": STAGES, "actions": list(PROGRAMS), "teacher_actions": list(TEACH_ACTIONS),
             "search_offsets": list(SEARCH_STREAM_OFFSETS), "proof_offsets": list(FUTURE_SEEDS["proof"]),
             "teacher_test_selection": "layout ordinal//2 modulo 3 equals zero", "dird": False}
    if path.exists() and read_json(path) != rules:
        raise RuntimeError("Frozen closed-loop protocol changed")
    atomic_json(path, rules)
    prepared = frozen_reader(warm_bundle(out, atlas), freeze)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_live, initargs=(atlas, prepared)) as pool:
        _, dev = collect(pool, out, "closed_dev", case_rollout, pack_rollouts)
    order = [{"features": r["features"], "paired_samples": r["paired_samples"], "layout": r["layout_id"],
              "reference": REFERENCE} for r in dev]
    model_path = out / "student_manifest.json"
    if model_path.exists():
        training = read_json(model_path)
        if training["closed_protocol"] != digest(path):
            raise RuntimeError("Student protocol diverged from fitted model")
        fitted = read_json(out / "student_bundle.json")
        if training["bundle_sha256"] != digest(out / "student_bundle.json"):
            raise RuntimeError("Student model changed after fitting")
    else:
        if not read_json(out / "gpu_preflight.json")["cuda_optimizer_step_verified"]:
            raise RuntimeError("CUDA student preflight unavailable")
        fitted, training = fit_student(order, TEACH_ACTIONS, out / "student_fit", seed=251_140_607)
        if fitted["reference"] != REFERENCE or not training["cuda_optimizer_step_verified"]:
            raise RuntimeError("Student reference or CUDA optimizer update did not verify")
        atomic_json(out / "student_bundle.json", fitted)
        training = training | {"closed_protocol": digest(path), "bundle_sha256": digest(out / "student_bundle.json"),
                               "actual_pre_update_student_episodes": len(dev), "training_layouts": sorted(r["layout"] for r in order)}
        atomic_json(model_path, training)
    print("Small CUDA action-value model frozen before held-out simulation", flush=True)
    if fitted["reference"] != REFERENCE:
        raise RuntimeError("Student outputs use the wrong paired baseline")
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_live, initargs=(atlas, prepared, fitted)) as pool:
        _, test = collect(pool, out, "closed_test", case_rollout, pack_rollouts)
    results = report(test)
    verified_layouts = verify_rosters(out, dev, test)
    atomic_json(out / "closed_results.json", {"protocol_sha256": digest(path), "student_freeze_sha256": digest(model_path),
         "routes": results, "new_layouts": verified_layouts, "reward_labeled_dev_layouts": len(dev),
         "complete_scoring_trajectories": sum(r["scoring_executions"] for r in dev + test),
         "reference_reused_when_ineligible": True, "reference": REFERENCE})
    for name, r in results.items():
        print(name, {k: (v["strict"], v["wrong_branch"], v["joint"]) for k,v in r["scores"].items()}, r["gates"], flush=True)


if __name__ == "__main__":
    main()
