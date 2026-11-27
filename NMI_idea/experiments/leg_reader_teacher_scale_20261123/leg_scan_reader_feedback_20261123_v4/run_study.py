"""Validate corrected readout, causal feedback, and the executable action menu."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from study_worlds import (DEFAULT_PACK, EXPERIMENT, FEATURES, PRIOR, PRIOR_MANIFEST, READER_SCANS, ROOT, ROUTES,
                          SHARD, acquire, atomic_json, collect, completed, digest, init_worlds, pack_acquired, read_json)

import numpy as np


ACTION_COUNTS = {"search_dev": 320, "search_test": 512, "feedback_test": 192}
ACTION_SHARD = 32
POSITION_THRESHOLD = .75


def recorded_layouts(rows):
    counts = {}
    for route in ROUTES:
        collection = [m for m in rows if m["case"]["task"] == route]
        probes = Counter(m.get("probe_skill", 0) for m in collection)
        counts[route] = {"unique_layouts": len({m["layout_id"] for m in collection}),
                         "cases": len(collection), "frames": sum(m.get("frame_count", 0) for m in collection),
                         "histories": sum(len(m.get("histories", ())) for m in collection),
                         "windows": probes[1], "revisits": probes[2],
                         "probe_region_3": sum(m.get("probe_in_region") == 2 and m.get("probe_skill", 0) != 0
                                               for m in collection)}
    return counts


def reader_phase(args):
    from reader_eval import calibrate, choose_readers, validate_readers
    from fit_public import fit_public, load_public

    out = args.out
    prior = read_json(args.prior_manifest)
    if digest(args.prior_reader) != prior["readers"]["scaled_identity"]["sha256"]:
        raise RuntimeError("v3 identity/association starting weights did not match the frozen manifest")
    if not read_json(out / "gpu_preflight.json")["cuda_optimizer_step_verified"]:
        raise RuntimeError("The CUDA preflight must pass")
    acquisition = read_json(out / "reader_acquisition.json")
    if acquisition["counts"] != READER_SCANS or acquisition["atlas_sha256"] != digest(args.atlas_pack):
        raise RuntimeError("Reader data does not match the declared protocol")
    paths, fit_meta = completed(out, "fit", READER_SCANS["fit"], SHARD)
    print(f"Fitting corrected identity and instance association on CUDA from {len(fit_meta)} physical scans", flush=True)
    manifest = out / "reader_weights" / "public_fit_manifest.json"
    if manifest.is_file():
        checked = read_json(manifest)
        if checked["epochs"] != args.reader_epochs or checked["prior_sha256"] != digest(args.prior_reader) \
                or checked["shards"] != [str(p) for p in paths]:
            raise RuntimeError("An existing CUDA fit has different inputs")
        result = checked["weights"]
    else:
        result = fit_public(paths, args.prior_reader, out / "reader_weights", epochs=args.reader_epochs)
    if result["device"] != "cuda:0" or not result["parameter_update_verified"] or digest(result["weights"]) != result["sha256"]:
        raise RuntimeError("The updated reader checkpoint did not verify")
    models = {"v3_scaled": load_public(args.prior_reader, "cuda:0"), "updated": load_public(result["weights"], "cuda:0")}
    if any(model["feature_order"] != FEATURES for model in models.values()):
        raise RuntimeError("Inconsistent public feature order")
    with np.load(args.atlas_pack, allow_pickle=False) as pack:
        links = pack["links"]
    cal, cal_meta = completed(out, "calibrate", READER_SCANS["calibrate"], SHARD)
    dev, dev_meta = completed(out, "select_reader", READER_SCANS["select_reader"], SHARD)
    prior_thresholds = {route: values["scaled_identity"] for route, values in prior["thresholds"].items()}
    risk = calibrate(cal, models, links)
    print("Calibrated the confidence of each already selected winner", flush=True)
    rules = choose_readers(dev, models, links, risk, prior_thresholds)
    frozen = {"updated": result, "prior_reader": str(args.prior_reader), "prior_sha256": digest(args.prior_reader),
              "thresholds_and_reader": rules, "risk": risk, "geometry_and_history_contract": "public_v4",
              "source_sha256": {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))},
              "fit_manifest_sha256": digest(manifest), "position_threshold": POSITION_THRESHOLD,
              "calibration_and_selection_distinct": True}
    if (out / "reader_freeze.json").exists():
        existing = read_json(out / "reader_freeze.json")
        if existing["updated"]["sha256"] != result["sha256"] or existing["thresholds_and_reader"] != rules:
            raise RuntimeError("Reader was already frozen with another development rule")
    atomic_json(out / "reader_freeze.json", frozen)
    print("Readers frozen by route before fresh test collection", {r: rule["selected"] for r, rule in rules.items()}, flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_worlds,
                             initargs=(args.atlas_pack,)) as pool:
        test, test_meta = collect(pool, out, "reader_test", READER_SCANS["reader_test"], acquire, pack_acquired, SHARD)
    scopes = (fit_meta, cal_meta, dev_meta, test_meta)
    pools = [{x["layout_id"] for x in rows} for rows in scopes]
    if sum(map(len, pools)) != len(set.union(*pools)):
        raise RuntimeError("Reader splits reused a layout")
    report = validate_readers(test, models, links, risk, rules)
    atomic_json(out / "reader_results.json", {"routes": report, "layout_counts": {
        k: recorded_layouts(rows) for k, rows in zip(READER_SCANS, scopes, strict=True)},
        "reader_freeze_sha256": digest(out / "reader_freeze.json")})
    for route, row in report.items():
        print("reader", route, {m: (r["strict"], r["wrong_region_count"], r["wrong_branch"])
                                for m, r in row["readouts"]["matched"].items()}, flush=True)


def action_phase(args):
    from action_eval import feedback_report, gates, panel, search_report
    from active_trial import (ACTIONS, DEV_TAPES, TAPES, init_action_study, pack_search, run_feedback, run_search)
    from dual_value import calibrated_actions, calibrate_dual, fit_dual, load_dual, predict_dual
    from fit_public import load_public
    from readout_state import export_cpu

    out = args.out
    source = out / "reader_freeze.json"
    freeze = read_json(source)
    prior, updated = freeze["prior_reader"], freeze["updated"]["weights"]
    if digest(prior) != freeze["prior_sha256"] or digest(updated) != freeze["updated"]["sha256"]:
        raise RuntimeError("Action study readers did not match their freeze")
    m = {"v3_scaled": export_cpu(load_public(prior)), "updated": export_cpu(load_public(updated))}
    readers = {}
    for route in ROUTES:
        rule = freeze["thresholds_and_reader"][route]
        readers[route] = {}
        for arm, method in (("selected", rule["selected"]), ("graph", "fixed_graph")):
            readers[route][arm] = {"method": method, "threshold": rule["variants"][method]["threshold"],
                                  "risk": freeze["risk"][route].get(method), "position_threshold": POSITION_THRESHOLD,
                                  "model": m["v3_scaled" if method == "v3_scaled" else "updated"]}
    code_sha = {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))}
    protocol = {"experiment": EXPERIMENT, "reader_freeze_sha256": digest(source), "action_menu": list(ACTIONS),
                "counts": ACTION_COUNTS, "dev_tapes": list(DEV_TAPES), "test_tapes": list(TAPES),
                "test_sampling": "predefined natural atlas layout distribution with no recoverability filter",
                "dird": False, "map": "planned and executed in the material chart", "source_sha256": code_sha}
    config = out / "action_protocol.json"
    if config.exists() and protocol != read_json(config):
        raise RuntimeError("Action protocol changed after freezing")
    atomic_json(config, protocol)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_action_study,
                             initargs=(args.atlas_pack, readers)) as pool:
        # Actor feedback is measured on paired trajectories before inspecting
        # natural action-test outcomes. Both feedback arms use the same frozen
        # graph to evaluate their acquired evidence.
        _, fb = collect(pool, out, "feedback_test", ACTION_COUNTS["feedback_test"], run_feedback, pack_search, ACTION_SHARD)
        control = feedback_report(fb)
        atomic_json(out / "feedback_results.json", control)
        for route, r in control.items():
            print("feedback", route, "probe paths changed", r["probe_paths_changed"], "MPC continuations changed",
                  r["continuation_paths_changed"], flush=True)

        dev, _ = collect(pool, out, "search_dev", ACTION_COUNTS["search_dev"], run_search, pack_search, ACTION_SHARD)
        rows, meta = panel(dev)
        dev_report = search_report(rows, meta, natural_test=False)
        decision = gates(dev_report)
        atomic_json(out / "action_development.json", {"results": dev_report, "teacher_training_eligibility": decision})
        print("Independent action-menu development gate", decision, flush=True)
        active = {}
        labels = np.asarray([row["case"]["task"] for row in meta])
        value_fits = {}
        for n, route in enumerate(ROUTES):
            ids = np.flatnonzero(labels == route)
            training, calibration = np.split(np.random.default_rng(4371 + n).permutation(ids), [len(ids) * 3 // 4])
            passed = decision[route]["passes"]
            # A small CUDA contract run always checks the training/inference
            # transforms using measured data. It cannot enter evaluation
            # when the independent executable action gate is unmet.
            selected = training if passed else training[:24]
            path = out / "teacher_weights" / f"{route}.pt"
            report = fit_dual(rows["features"][selected], rows["valid"][selected], rows["outcomes"][selected, :, :2], path,
                              epochs=25 if passed else 2)
            value_fits[route] = report | {"purpose": "action model" if passed else "CUDA parameterization smoke"}
            if passed:
                bundle = load_dual(path)
                preds = predict_dual(bundle, rows["features"][calibration], rows["valid"][calibration])
                rule = calibrate_dual(preds, rows["valid"][calibration], rows["outcomes"][calibration, :, 2:])
                active[route] = (bundle, rule)
                value_fits[route]["calibrated_rule"] = rule
        atomic_json(out / "teacher_gate_freeze.json", {"decision": decision, "value_fits": value_fits,
                                                        "natural_test_selection_independent": True,
                                                        "readers_sha256": digest(source)})
        test, _ = collect(pool, out, "search_test", ACTION_COUNTS["search_test"], run_search, pack_search, ACTION_SHARD)
    natural, ledger = panel(test)
    tasks = np.asarray([case["case"]["task"] for case in ledger])
    if any(case["scope"] != "natural" for case in ledger):
        raise RuntimeError("The final action cohort must remain natural")
    choices = {}
    for task, (bundle, rule) in active.items():
        subset = tasks == task
        valid = natural["valid"][subset]
        choices[task] = calibrated_actions(predict_dual(bundle, natural["features"][subset], valid), valid, rule)
    report = search_report(natural, ledger, natural_test=True, learned=choices)
    atomic_json(out / "action_results.json", {"routes": report, "teacher_gate_sha256": digest(out / "teacher_gate_freeze.json"),
                                               "independent_test_complete": True,
                                               "dev_continuations": int(rows["valid"].sum()) * len(DEV_TAPES),
                                               "natural_test_continuations": int(natural["valid"].sum()) * len(TAPES)})
    for route, row in report.items():
        for arm, val in row["arms"].items():
            facts = val["totals_by_tape"][0]
            pairing = val["contrasts"]["mpc"]["by_tape"][0]["strict"]
            print("natural test", route, arm, f"strict={facts['strict']}/{row['cases']}",
                  f"wrong_branch={facts['wrong_branch']}",
                  f"rescue/loss={pairing['rescued_or_added']}/{pairing['lost_or_removed']}", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("reader", "actions", "all"), default="all")
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--prior-reader", type=Path, default=PRIOR)
    parser.add_argument("--prior-manifest", type=Path, default=PRIOR_MANIFEST)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reader-epochs", type=int, default=3)
    args = parser.parse_args()
    for name in ("out", "atlas_pack", "prior_reader", "prior_manifest"):
        setattr(args, name, getattr(args, name).resolve())
    if args.phase in ("reader", "all"):
        reader_phase(args)
    if args.phase in ("actions", "all"):
        action_phase(args)


if __name__ == "__main__":
    main()
