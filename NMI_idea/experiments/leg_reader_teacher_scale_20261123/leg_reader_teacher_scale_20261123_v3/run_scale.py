"""Run one reader-to-teacher update and independent paired validation cycle."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path

from scale_worlds import (DEFAULT_PACK, EXPERIMENT, FEATURES, FUTURE_SEEDS, PRIOR, ROOT, ROUTES, SEED_RANGES, V2,
                          atomic_json, cases_for, collect, complete, digest, init_worlds, load_shard, read_json,
                          start, vectors)
from collect_scale import READER_COUNTS, READER_SHARD

import numpy as np


TEACHER_COUNTS = {"teacher_fit": 2048, "teacher_calibration": 384, "teacher_test": 768}
TEACHER_SHARD = 48
PRIOR_FREEZE = V2 / "fixed_pilot" / "reader_freeze.json"


def stage_paths(out, stage, count):
    paths, ledger = [], []
    for i in range(0, count, READER_SHARD):
        folder = out / stage / f"{i // READER_SHARD:05d}"
        path, meta = load_shard(folder)
        commit = read_json(folder / "commit.json")
        if digest(path) != commit["data_sha256"] or digest(folder / "cases.jsonl.gz") != commit["metadata_sha256"]:
            raise RuntimeError(f"Reader data checksum mismatch in {folder}")
        paths.append(path)
        ledger.extend(meta)
    if (len(ledger) != count or not all(row["complete"] for row in ledger)
            or len({row["layout_id"] for row in ledger}) != count or any(row["case"]["split"] != stage for row in ledger)):
        raise RuntimeError(f"{stage} did not contain {count} unique complete scans")
    return paths, ledger


def collection_counts(metadata):
    scopes = {}
    for task in ROUTES:
        rows = [row for row in metadata if row["case"]["task"] == task]
        info = {"scans": len(rows), "layouts": len({r["layout_id"] for r in rows}),
                "window_generated": sum(r["window_generated"] for r in rows)}
        if rows and "frames" in rows[0]:
            info.update({"acquisition": dict(Counter(row["acquisition"] for row in rows)),
                         "latest_unique_frames": sum(row["frames"] for row in rows),
                         "harmonic_candidates": sum(row["harmonic_candidates"] for row in rows),
                         "original_window_seen": sum(row["reference_window_visited"] for row in rows),
                         "both_windows_seen": sum(row["reference_window_visited"] and row["alternative_window_visited"]
                                                  for row in rows)})
        scopes[task] = info
    return scopes


def reader_stage(args):
    from scale_readout_eval import calibration, compare_readouts
    from scaled_reader import fit_readers, load_reader

    out = args.out
    prior = read_json(args.prior_freeze)
    if digest(args.prior_reader) != prior["readers"]["full_public_embedding"]["sha256"]:
        raise RuntimeError("The warm-start reader does not match its v2 freeze")
    device = read_json(out / "gpu_preflight.json")
    if not device["cuda_optimizer_step_verified"]:
        raise RuntimeError("CUDA preflight must pass before fitting")
    protocol = read_json(out / "reader_collection.json")
    counts = protocol["counts"]
    if counts != READER_COUNTS or digest(args.atlas_pack) != protocol["inputs_sha256"][args.atlas_pack.name]:
        raise RuntimeError("Reader collection does not match the production protocol")
    prior_thresholds = {route: thresholds["full_public_embedding"]
                        for route, thresholds in prior["thresholds_development"].items()}
    fit, fit_meta = stage_paths(out, "reader_fit", counts["reader_fit"])
    print(f"Fitting both reader losses on CUDA with {len(fit_meta)} unique complete scans", flush=True)
    fit_file = out / "reader_weights" / "reader_update_manifest.json"
    if fit_file.exists():
        trained = read_json(fit_file)
        if (trained["prior_sha256"] != digest(args.prior_reader) or trained["epochs"] != args.reader_epochs
                or trained["shards"] != [str(path) for path in fit]):
            raise RuntimeError("Existing reader training manifest differs from frozen inputs")
    else:
        trained = fit_readers(fit, args.prior_reader, out / "reader_weights", epochs=args.reader_epochs)
    paths = {name: info["weights"] for name, info in trained["readers"].items()}
    for info in trained["readers"].values():
        if not info["parameter_update_verified"] or info["device"] != "cuda:0" or digest(info["weights"]) != info["sha256"]:
            raise RuntimeError("Reader checkpoint did not verify against the CUDA training manifest")
    models = {name: load_reader(path, "cuda:0") for name, path in {"v2_reader": args.prior_reader, **paths}.items()}
    if any(model["feature_order"] != FEATURES for model in models.values()):
        raise RuntimeError("Reader checkpoints have different public schemas")
    development, dev_meta = stage_paths(out, "reader_calibration", counts["reader_calibration"])
    thresholds = calibration(development, models, prior_thresholds)
    freeze = {"experiment": EXPERIMENT, "reader_fit_manifest_sha256": digest(fit_file), "thresholds": thresholds,
              "calibration": "match v2 submissions by route on new development tracks",
              "prior_sha256": digest(args.prior_reader), "readers": trained["readers"],
              "teacher_reader": "updated_reader", "teacher_reader_sha256": trained["readers"]["updated_reader"]["sha256"]}
    atomic_json(out / "reader_freeze.json", freeze)
    validation, val_meta = stage_paths(out, "reader_test", counts["reader_test"])
    groups = [{r["layout_id"] for r in rows} for rows in (fit_meta, dev_meta, val_meta)]
    if sum(map(len, groups)) != len(set.union(*groups)):
        raise RuntimeError("Reader split layouts overlap")
    reader_result = compare_readouts(validation, models, thresholds)
    counts = {k: collection_counts(rows) for k, rows in zip(READER_COUNTS, (fit_meta, dev_meta, val_meta), strict=True)}
    atomic_json(out / "reader_results.json", {"routes": reader_result, "collection": counts,
                                               "reader_freeze_sha256": digest(out / "reader_freeze.json")})
    for route, row in reader_result.items():
        print(route, "equal submissions",
              {key: {"wrong_regions": d["wrong_region_count"], "strict": d["strict"], "wrong_branch": d["wrong_branch"]}
               for key, d in row["counts"]["matched"].items()}, flush=True)


def teacher_stage(args):
    from scale_teacher import ACTIONS, ACTION_FEATURES, TAPES, TARGETS, cpu_bundle, init_teacher, pack_prefixes, run_prefix
    from scale_teacher_eval import evaluate_teacher, fit_selection, load_panel, stable_signals
    from scaled_reader import load_reader
    from scaled_value import fit_value, load_value, predict_value

    out = args.out
    freeze_file = out / "reader_freeze.json"
    reader_freeze = read_json(freeze_file)
    reader = reader_freeze["readers"][reader_freeze["teacher_reader"]]
    if digest(reader["weights"]) != reader_freeze["teacher_reader_sha256"]:
        raise RuntimeError("The reader changed after freezing")
    thresholds = {k: v[reader_freeze["teacher_reader"]] for k, v in reader_freeze["thresholds"].items()}
    bundle = cpu_bundle(load_reader(reader["weights"]))
    protocol = {"experiment": EXPERIMENT, "reader_freeze_sha256": digest(freeze_file),
                "atlas_sha256": digest(args.atlas_pack), "counts": TEACHER_COUNTS, "case_seeds": SEED_RANGES,
                "noise_seed_offsets": FUTURE_SEEDS, "tapes": TAPES, "review_actions": ACTIONS,
                "public_features": ACTION_FEATURES, "private_targets": TARGETS,
                "feedback": True, "stop": False, "dird": False,
                "continuation": "same constrained public MPC after every action",
                "fit": "expected paired outcome rather than imitation of a privileged best action",
                "raw_code_sha256": {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))}}
    teacher_protocol = out / "teacher_protocol.json"
    if teacher_protocol.exists():
        checked = read_json(teacher_protocol)
        if (checked["reader_freeze_sha256"] != protocol["reader_freeze_sha256"]
                or checked["raw_code_sha256"] != protocol["raw_code_sha256"]):
            raise RuntimeError("The frozen teacher run must be resumed with the original code and reader")
    atomic_json(teacher_protocol, protocol)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_teacher,
                             initargs=(args.atlas_pack, bundle, thresholds)) as pool:
        fit, fit_meta = collect(pool, "teacher_fit", TEACHER_COUNTS["teacher_fit"], out, run_prefix, pack_prefixes, TEACHER_SHARD)
        print(f"Fitting the public teacher decision model on CUDA with {len(fit_meta)} layout prefixes", flush=True)
        training = fit_value(fit, out / "teacher_weights", epochs=args.teacher_epochs)
        if not training["changed_on_cuda"] or training["device"] != "cuda:0" or digest(training["weights"]) != training["sha256"]:
            raise RuntimeError("Action-value CUDA update failed verification")
        development, dev_meta = collect(pool, "teacher_calibration", TEACHER_COUNTS["teacher_calibration"], out,
                                        run_prefix, pack_prefixes, TEACHER_SHARD)
        value = load_value(training["weights"], "cuda:0")
        data, _ = load_panel(development)
        rules = fit_selection(data, predict_value(data["features"], data["valid"], value))
        panel_freeze = {"protocol_sha256": digest(teacher_protocol), "reader_freeze_sha256": digest(freeze_file),
                        "training": training, "selection_rules": rules,
                        "selection_inputs": "frozen reader, public prefix and legal action geometry",
                        "test_world_seeds": "distinct from reader, teacher fit and calibration"}
        atomic_json(out / "teacher_freeze.json", panel_freeze)
        print("Teacher weight, thresholds, and action rule frozen before independent collection", flush=True)
        tested, test_meta = collect(pool, "teacher_test", TEACHER_COUNTS["teacher_test"], out,
                                    run_prefix, pack_prefixes, TEACHER_SHARD)

    all_layouts = [{r["layout_id"] for r in stage} for stage in (fit_meta, dev_meta, test_meta)]
    if sum(map(len, all_layouts)) != len(set.union(*all_layouts)):
        raise RuntimeError("Teacher split layouts overlap")
    data, meta = load_panel(tested)
    evaluated = evaluate_teacher(data, meta, predict_value(data["features"], data["valid"], value), rules)
    train_panel, _ = load_panel(fit)
    training_signals = {}
    for route in ROUTES:
        idx = train_panel["route"] == route
        training_signals[route] = stable_signals(train_panel["outcomes"][idx], train_panel["valid"][idx], 1, (0, 1))
    counters = {stage: collection_counts(ledger) for stage, ledger in zip(TEACHER_COUNTS, (fit_meta, dev_meta, test_meta), strict=True)}
    atomic_json(out / "teacher_results.json", {"routes": evaluated, "teacher_freeze_sha256": digest(out / "teacher_freeze.json"),
                                               "training_signals": training_signals, "collection": counters,
                                               "full_continuations_by_split": {"teacher_fit": int(train_panel["valid"].sum()) * 2,
                                                                               "teacher_test": int(data["valid"].sum()) * 5}})
    print_teacher(evaluated)


def print_teacher(routes):
    for route, row in routes.items():
        for name, arm in row["arms"].items():
            strict = arm["contrasts"]["mpc"]["strict"]
            print(route, name, f"strict {arm['counts']['strict']}/{row['cases']}",
                  f"wrong branch {arm['counts']['wrong_branch']}",
                  f"rescued/lost vs MPC {strict['rescued_or_added']}/{strict['lost_or_removed']}", flush=True)


def next_phase(out):
    results = read_json(out / "reader_results.json")
    teacher = read_json(out / "teacher_results.json")
    choices = {}
    for route in ROUTES:
        row = results["routes"][route]
        reader = row["counts"]["fixed"]
        diff = row["wrong_region_contrasts"]["v2_reader"]
        improved_readout = (diff["layout95_delta_per_scan"][1] < 0.
                            and reader["updated_reader"]["strict"] >= reader["v2_reader"]["strict"])
        update = teacher["routes"][route]["arms"]["updated_teacher"]
        hold, mpc = (update["contrasts"][arm] for arm in ("equal_time_hold", "mpc"))
        improved_teacher = (hold["strict"]["clustered95"][0] > 0.
                            and min(hold["proof"]["strict_delta_counts"]) > 0
                            and hold["wrong_branch"]["delta_per_scan"] <= 0.
                            and mpc["strict"]["delta_per_scan"] >= 0.)
        if improved_teacher and improved_readout:
            next_test = "Small online PPO roll-in with frozen reader/value, independent student-visited prefixes and equal-budget MPC comparison."
        elif not improved_readout:
            next_test = "Trace false top candidates on held-out routes; fit alternate-window association supervised on new development layouts."
        else:
            next_test = "Collect across-window/anchor views at public ambiguous prefixes; verify more reproducible rescue than loss before supervision."
        choices[route] = {"reader_gate": improved_readout, "teacher_gate": improved_teacher,
                          "next_experiment": next_test,
                          "candidates_with_two_tape_rescue": teacher["routes"][route]["available_signal_vs_hold"]["robust_rescue_actions"]}
    atomic_json(out / "next_experiment.json", choices)


def smoke(args):
    """Exercise real continuation, public label invariance and inference parity."""
    from public_embedding import extract as reference_extract
    from fast_embedding import extract
    from scale_teacher import init_teacher, run_prefix, cpu_bundle, probabilities
    from scaled_reader import load_reader, predict

    prior = load_reader(args.prior_reader)
    bundle = cpu_bundle(prior)
    thresholds = {k: v["full_public_embedding"] for k, v in read_json(args.prior_freeze)["thresholds_development"].items()}
    init_worlds(args.atlas_pack)
    checks = {}
    for case in cases_for("smoke", 4):
        env, spec, node, trial, limits = start(case)
        complete(env, spec, node, trial, limits)
        before = vectors(env, spec, node)
        fast = extract(node.reader, spec, env.material)
        reference = reference_extract(node.reader, spec, env.material)
        for key in ("extended", "extended_mask", "frame_index", "slot", "region"):
            np.testing.assert_allclose(fast[key], reference[key], rtol=0., atol=1.e-6)
        node.reader.frames = [(*f[:3], np.roll(f[3], 1), *f[4:]) for f in node.reader.frames]
        replaced = vectors(env, spec, node)
        np.testing.assert_array_equal(before["features"], replaced["features"])
        np.testing.assert_allclose(probabilities(before["features"], bundle), predict(before["features"], prior), atol=1.e-6)
        checks[case.identifier] = {"route": case.task, "anatomy": trial.layout.identifier, "ticks": node.tick,
                                   "labels_invariant": True, "cpu_inference_matches_torch": True}
    init_teacher(args.atlas_pack, bundle, thresholds)
    for case in cases_for("teacher_calibration", 4):
        case = replace(case, identifier=case.identifier.replace("calibration", "smoke"),
                       world_seed=case.world_seed + 9_100_000, tape_seed=case.tape_seed + 9_100_000,
                       action_seed=case.action_seed + 9_100_000)
        batch, meta = run_prefix(case)
        checks[case.identifier] = {"route": case.task, "legal_actions": int(batch["valid"].sum()),
                                   "layout": meta["layout_id"], "complete": meta["complete"]}
    atomic_json(args.out / "smoke_checks.json", checks)
    print(f"Verified {len(checks)} cases for public feature and continuation contracts", flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("smoke", "reader", "teacher", "all"), default="all")
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--prior-reader", type=Path, default=PRIOR)
    p.add_argument("--prior-freeze", type=Path, default=PRIOR_FREEZE)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--reader-epochs", type=int, default=4)
    p.add_argument("--teacher-epochs", type=int, default=32)
    args = p.parse_args()
    for field in ("out", "atlas_pack", "prior_reader", "prior_freeze"):
        setattr(args, field, getattr(args, field).resolve())
    if args.phase == "smoke":
        smoke(args)
        return
    if args.phase in ("reader", "all"):
        reader_stage(args)
    if args.phase in ("teacher", "all"):
        teacher_stage(args)
        next_phase(args.out)


if __name__ == "__main__":
    main()
