"""Run the two fixed atlas routes and test public embeddings on the same acquired frames."""

from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "leg_closed_loop_20261122_v1" / "src"
sys.path.insert(0, str(V1))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["LEG_ENABLE_DIRD"] = "0"
os.environ["LEG_ENABLE_MATERIAL_ATLAS"] = "1"

import numpy as np

import motion
from connected_reader import rerank
from public_embedding import BASELINE_FEATURE_NAMES, EXTENDED_FEATURE_NAMES, extract
from route_metrics import attributes, outcomes, paired, totals
from storage import atomic_json, digest, read_json, stash
from online_reader import PublicHistoryReader
from environment import spec_for


BASE = len(BASELINE_FEATURE_NAMES)
FEATURES = (BASELINE_FEATURE_NAMES + tuple(f"available_{v}" for v in BASELINE_FEATURE_NAMES)
            + EXTENDED_FEATURE_NAMES[BASE:] + tuple(f"available_{v}" for v in EXTENDED_FEATURE_NAMES[BASE:]))
POLICIES = ("mpc", "active_mpc", "teacher_ppo", "teacher_ppo_no_stop", "ppo", "ppo_no_stop")
LEGACY_THRESHOLD = .82
FIT_SEED_OFFSET = 71_981_270_002_109
PILOT_EXPERIMENT = "leg_named_target_routes_v2_20261123"


def shape_public_reader(env, spec, acquired, mixed):
    reader = PublicHistoryReader(env.model, spec[3], True)
    for i, frame in enumerate(acquired):
        public = (*frame[:3], None, *frame[4:])
        reader.frames.append(public)
        reader.mixed_posterior.append(mixed[i])
        reader._append_frame(i, public)
    return reader


def features_of(env, spec, item):
    reader = shape_public_reader(env, spec, item.frames, item.mixed)
    frame_data = extract(reader, spec, env.material)
    full = frame_data["extended"]
    present = frame_data["extended_mask"]
    ff, sl = frame_data["frame_index"], frame_data["slot"]
    fields = np.c_[full[:, :BASE], present[:, :BASE], full[:, BASE:], present[:, BASE:]].astype(np.float32)
    keyed = {(int(i), int(j)): v for region in rerank(reader, spec) for i, j, v in region}
    old = frame_data["baseline"][:, 0]
    track = np.asarray([keyed[(int(i), int(j))] for i, j in zip(ff, sl)], dtype=np.float32)
    labels = np.asarray([item.frames[i][3][j] for i, j in zip(ff, sl)], dtype=np.int8)
    station = np.asarray([item.frames[i][0] for i in ff], dtype=float)
    sector = np.asarray([item.frames[i][1] for i in ff], dtype=float)
    return {"features": fields, "frame": ff, "slot": sl, "label": labels,
            "station": station, "sector": sector, "region": frame_data["region"],
            "v1_history": old, "connected_rank": track, "complete": item.metrics["complete"],
            "harmonic_candidate_count": int(present[:, EXTENDED_FEATURE_NAMES.index("harmonic_supported")].sum())}


def geometric_opportunity(env, case, spec, item):
    target = item.layout_centers if item.layout_centers is not None else env.scanner._anatomy.centers[case.source]
    surface = env.scanner.surface
    tests = []
    for s in range(spec[1], spec[2] + 1):
        pos = target[s, 0]
        has = False
        if np.isfinite(pos).all():
            for sector in np.flatnonzero(env.scanner.valid[s]):
                x = surface.project(s, int(sector), pos)
                if -25. <= x[0] <= 25. and 3. <= x[1] <= 80. and surface.shadow(s, int(sector), x[None])[0] < .95:
                    has = True
                    break
        tests.append(has)
    return {"stations": tests, "regions": [bool(np.any(np.asarray(tests)[spec[4] == r])) for r in range(6)]}


def threshold_for(rows, method, expected):
    best = [max(scores[a["region"] == region], default=-1.)
            for a in rows for region in range(6) for scores in [a[method]]]
    eligible = [s for s in best if s >= 0.]
    points = np.unique(np.r_[0., 1., eligible])
    threshold = min(points, key=lambda t: (abs(sum(i >= t for i in eligible) - expected), -t))
    return float(threshold)


def same_submit_thresholds(data, methods):
    bundle = {}
    for task in sorted({case.task for case, _, _, _, _ in data}):
        ref = [a for case, _, _, a, _ in data if case.task == task]
        target_count = sum(outcomes(a, a["v1_history"], LEGACY_THRESHOLD)["submitted"] for a in ref)
        bundle[task] = {"v1_history": LEGACY_THRESHOLD}
        for method in methods:
            if method != "v1_history":
                bundle[task][method] = threshold_for(ref, method, target_count)
    return bundle


def record_run_stage(out, stage, rows):
    values = [(case, arm, item) for case, arm, item, _, _ in rows]
    stash(out / stage / "00000", values, {"experiment": PILOT_EXPERIMENT, "stage": stage})
    offsets = np.cumsum([0, *(len(r["label"]) for _, _, _, r, _ in rows)])
    names = ("features", "frame", "slot", "station", "sector", "region", "v1_history", "connected_rank")
    payload = {k: np.concatenate([row[k] for _, _, _, row, _ in rows], axis=0) for k in names}
    payload["offsets"] = offsets
    payload["feature_order"] = np.asarray(FEATURES)
    np.savez_compressed(out / stage / "public_candidates.npz", **payload)
    np.savez_compressed(out / stage / "candidate_targets.npz", offsets=offsets,
                        labels=np.concatenate([a["label"] for _, _, _, a, _ in rows]))


def report_routes(rows, methods, thresholds):
    tracks = defaultdict(dict)
    for case, arm, item, features, geom in rows:
        scores = {method: outcomes(features, features[method], thresholds[case.task][method]) for method in methods}
        same_count = scores["v1_history"]["submitted"]
        eq = {m: outcomes(features, features[m], thresholds[case.task][m], same_count) for m in methods}
        tracks[case.task][case.identifier, arm] = {
            "case": asdict(case), "scores": scores, "equal_per_scan_submissions": eq,
            "geometric": geom, "probe_proposals": item.metrics["proposed"], "probes": item.metrics["probes"],
            "early_stops": item.metrics["early_stops"], "time_s": item.metrics["time_s"],
            "travel_mm": item.metrics["travel_mm"], "material_harmonic_candidates": features["harmonic_candidate_count"],
            "layout_id": item.metrics.get("layout_id", ""),
            "budget_ticks": item.metrics["cap_ticks"], "budget_mm": item.metrics["cap_mm"],
        }
    result = {}
    for task, panels in tracks.items():
        case_ids = sorted({key[0] for key in panels})
        arms = sorted({key[1] for key in panels})
        contrasts, embeddings = {}, {}
        mpc = [panels[key, "mpc"]["scores"]["v1_history"] for key in case_ids]
        cluster = [panels[key, "mpc"]["layout_id"] or key for key in case_ids]
        for arm in arms:
            panel = [panels[key, arm] for key in case_ids]
            outcomes_by_case = [x["scores"]["v1_history"] for x in panel]
            for i, case_id in enumerate(case_ids):
                alts = [panels[case_id, other]["scores"]["v1_history"]["capture"] for other in arms if other != arm]
                panel[i]["attribution"] = attributes(outcomes_by_case[i], panel[i]["geometric"]["regions"], alts)
            counts = totals(outcomes_by_case)
            counts["failure_attribution"] = {
                k: int(sum(x["attribution"][k] for x in panel)) for k in panel[0]["attribution"]
            }
            counts["duration_mean_s"] = float(np.mean([x["time_s"] for x in panel]))
            counts["travel_mean_mm"] = float(np.mean([x["travel_mm"] for x in panel]))
            counts["probe_count"] = sum(x["probes"] for x in panel)
            counts["early_stops"] = sum(x["early_stops"] for x in panel)
            counts["paired_vs_mpc"] = paired(outcomes_by_case, mpc, cluster, 745787)
            if arm.endswith("_no_stop"):
                first = arm.removesuffix("_no_stop")
                with_stop = [panels[k, first]["scores"]["v1_history"] for k in case_ids]
                counts["paired_vs_same_policy_stop_on"] = paired(outcomes_by_case, with_stop, cluster, 745907)
            contrasts[arm] = counts
            embeddings[arm] = {m: totals([x["scores"][m] for x in panel]) | {
                "equal_submissions": totals(x["equal_per_scan_submissions"][m] for x in panel),
                "paired_equal_vs_v1": paired([x["equal_per_scan_submissions"][m] for x in panel],
                                             [x["equal_per_scan_submissions"]["v1_history"] for x in panel],
                                             cluster, 749913),
            } for m in methods}
        shared_layout = [all(panels[k, a]["layout_id"] == panels[k, "mpc"]["layout_id"]
                             and panels[k, a]["budget_ticks"] == panels[k, "mpc"]["budget_ticks"]
                             and abs(panels[k, a]["budget_mm"] - panels[k, "mpc"]["budget_mm"]) <= 1e-5
                             for a in arms) for k in case_ids]
        if not all(shared_layout):
            raise RuntimeError(f"Paired layout or budget differs in {task}")
        result[task] = {"cases": len(case_ids), "same_layout_and_budgets": True,
                        "policy_arms": contrasts, "same_track_readouts": embeddings,
                        "per_scan": [{"arm": arm} | e for (_, arm), e in panels.items()]}
    return result


def print_summary(res):
    print("task arm strict/N wrong_branch_scans rescued/lost(strict) observed_oracle/N")
    for task, route in res.items():
        for arm, val in route["policy_arms"].items():
            pair = val["paired_vs_mpc"]["strict"]
            print(task, arm, f"{val['strict']}/{val['cases']}", val["wrong_branch"],
                  f"{pair['rescued_or_added']}/{pair['lost_or_removed']}",
                  f"{val['acquired_ceiling']}/{val['cases']}", flush=True)


def cli():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "fixed_pilot")
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def work(job):
    from route_rollouts import run_case

    case, arms = job
    measured = []
    first_frame = None
    geometry = None
    for scene, name, row in run_case(case, arms):
        item = row.rollout
        if isinstance(item.metrics.get("layout"), Mapping):
            item.metrics["layout"] = dict(item.metrics["layout"])
        env = row.environment
        spec = spec_for(env.scanner, case)
        first = item.frames[0]
        if first_frame is None:
            first_frame = first
        else:
            if first_frame[0] != first[0] or first_frame[1] != first[1] or first_frame[8] != first[8]:
                raise RuntimeError("Paired starts differ")
            np.testing.assert_array_equal(first_frame[2], first[2])
        if geometry is None:
            geometry = geometric_opportunity(env, scene, spec, item)
        features = features_of(env, spec, item)
        public = features["v1_history"]
        check = outcomes(features, public, LEGACY_THRESHOLD)
        if check["region_outcomes"] != item.metrics["region_outcomes"]:
            raise RuntimeError("The same-track v1 results disagree with the executed reader")
        measured.append((scene, name, item, features, geometry))
    return measured


def reader_gate(report):
    findings = {}
    for task, route in report.items():
        mpc = route["same_track_readouts"]["mpc"]
        if "full_public_embedding" not in mpc:
            continue
        full = mpc["full_public_embedding"]["equal_submissions"]
        local = mpc["local_embedding"]["equal_submissions"]
        old = mpc["v1_history"]["equal_submissions"]
        if not (full["submitted"] == local["submitted"] == old["submitted"]):
            raise RuntimeError("Reader feature effects require matched submission counts")
        findings[task] = {
            "same_track_fewer_wrong_vs_v1": full["wrong_region_count"] < old["wrong_region_count"],
            "same_track_fewer_wrong_vs_matched_local": full["wrong_region_count"] < local["wrong_region_count"],
            "full_wrong_regions": full["wrong_region_count"],
            "local_wrong_regions": local["wrong_region_count"],
            "v1_wrong_regions": old["wrong_region_count"],
            "checked_submission_count": full["submitted"],
            "teacher_holdout_eligible": full["wrong_region_count"] < min(local["wrong_region_count"], old["wrong_region_count"]),
        }
    return findings


def main(args):
    from route_rollouts import ROUTES, initialize

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pilot = initialize()
    calibration = [case for task in ROUTES for case in pilot.case_plan("calibration", task)]
    initial = [replace(case, identifier=case.identifier.replace("calibration", "fit"), split="fit",
                       world_seed=case.world_seed + FIT_SEED_OFFSET, tape_seed=case.tape_seed + FIT_SEED_OFFSET,
                       action_seed=case.action_seed + FIT_SEED_OFFSET)
               for case in calibration]
    evaluation = [case for task in ROUTES for case in pilot.case_plan("evaluation", task)]
    plans = {"reader_fit": (initial, ("mpc", "active_mpc")),
             "threshold_calibration": (calibration, ("mpc", "active_mpc")),
             "fixed_evaluation": (evaluation, POLICIES)}
    if args.baseline_only:
        plans = {name: value for name, value in plans.items() if name == "fixed_evaluation"}
    protocol = {
        "experiment": PILOT_EXPERIMENT, "source_pack": dict(pilot.metadata),
        "source_sha256": {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))},
        "shared_engine_sha256": {p.name: digest(p) for p in sorted(V1.glob("*.py"))},
        "prior_controls_sha256": digest(ROOT / "prior_control.json"),
        "route_eligibility": {task: sorted({case.source for case in evaluation if case.task == task}) for task in ROUTES},
        "case_plans": {stage: {"cases": [asdict(case) for case in plan], "arms": arms}
                       for stage, (plan, arms) in plans.items()},
        "holdout": "fixed generated atlas validation layouts and unused bodies, not Blender nor new human anatomy",
        "dird": False, "clock_s": motion.CLOCK_S, "actuation": "shared constrained ProDMP and MPC",
        "reader_fit": "matched GPU identity and cross-view association supervision",
        "same_track_gate": "fewer incorrect submitted regions with equal per-scan counts against v1 and matched local within each route",
        "future_teacher_gate": "use a frozen reader and public continuation on fresh cases",
    }
    atomic_json(out / "protocol.json", protocol)
    data = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=initialize) as process:
        for stage, (cases, arms) in plans.items():
            rows = [part for job in process.map(work, [(case, arms) for case in cases], chunksize=1) for part in job]
            if len(rows) != len(cases) * len(arms):
                raise RuntimeError("Fixed panel is incomplete")
            record_run_stage(out, stage, rows)
            data[stage] = rows
            print(f"committed {stage}: {len(rows)} complete scans", flush=True)
    methods = ["v1_history", "connected_rank"]
    trained = {}
    summaries = {}
    if not args.baseline_only:
        from reader_fit import fit, score

        trained, summaries = fit([a for _, _, _, a, _ in data["reader_fit"]], 2 * BASE, FEATURES, out / "weights")
        methods.extend(trained)
        print(f"updated {len(trained)} GPU readout variants", flush=True)
        for stage in ("threshold_calibration", "fixed_evaluation"):
            for _, _, _, values, _ in data[stage]:
                for method, reader in trained.items():
                    values[method] = score(values["features"], reader)
        thresholds = same_submit_thresholds(data["threshold_calibration"], methods)
    else:
        thresholds = {task: {method: LEGACY_THRESHOLD for method in methods} for task in ROUTES}
    results = report_routes(data["fixed_evaluation"], methods, thresholds)
    output = {"experiment": PILOT_EXPERIMENT, "protocol_sha256": digest(out / "protocol.json"),
              "methods": methods, "thresholds_development": thresholds, "gpu_readouts": summaries,
              "routes": results, "embedding_gate": reader_gate(results), "dird_enabled": False}
    atomic_json(out / "results.json", output)
    if summaries:
        frozen = {
            "experiment": PILOT_EXPERIMENT,
            "protocol_sha256": output["protocol_sha256"],
            "readers": {name: {"weights": info["weights"], "sha256": digest(info["weights"])}
                        for name, info in summaries.items()},
            "thresholds_development": thresholds, "approved_routes_for_teacher": {
                key: "full_public_embedding" for key, value in output["embedding_gate"].items()
                if value["teacher_holdout_eligible"]},
        }
        atomic_json(out / "reader_freeze.json", frozen)
    print_summary(results)


if __name__ == "__main__":
    main(cli())
