"""Run local skill learning, reader adaptation and independent v10 pairing."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
import json

import numpy as np

from recovery_task import (DEFAULT_PACK, EVAL_TAPES, ROOT, ROSTER, ROUTES, SEARCH_TAPES, TEACH_PROOF, V9,
                           Live, initialize, roster)
from recovery_policy import fit_options, read_bundle, rollout, update_online
from recovery_teacher import focus_prefix, strong
from reader_v9 import PUBLIC_FIELDS
from study_v9 import atomic_json, commit_shard, digest, freeze, read_json, verify_shard


_CPU_ACTORS = {}
_V9_READER = None
_MODULES = ("recovery_task.py", "recovery_motion.py", "recovery_teacher.py", "recovery_policy.py",
            "recovery_reader.py", "run_recovery.py")


def actor(path):
    if path not in _CPU_ACTORS:
        _CPU_ACTORS[path] = read_bundle(path)
    return _CPU_ACTORS[path]


def strong_job(job):
    case, width, snapshots, *follow = job
    live, focus = focus_prefix(case)
    ep, report, reader_data = strong(live, focus, width=width, keep_snapshots=snapshots,
                                    teach_continuations=bool(follow and follow[0]))
    return ep, report, reader_data


def collect_reader(job):
    case, path, split = job
    live, focus = focus_prefix(case)
    snapshots = [live.snapshot(split, "targeted_failure_prefix")]
    baseline = live.fork()
    baseline.finish()
    snapshots.append(baseline.snapshot(split, "mpc_terminal"))
    acquired = rollout(live, actor(path), case.action_seed, greedy=False, record=False, snapshot_split=split)
    snapshots.extend(acquired.pop("snapshots"))
    return {}, {"case": asdict(case), "layout_id": acquired["metrics"]["layout_id"], "focus": focus+1,
                "baseline": baseline.grade(), "policy": acquired["metrics"]}, snapshots


def online_job(job):
    case, path, guided, prefix = job
    live, focus = focus_prefix(case) if prefix else (Live(case), None)
    run = rollout(live, actor(path), case.action_seed, greedy=False, query=guided)
    steps = run["experience"]
    x = {key: np.stack([s[key] for s in steps]) for key in ("x", "mask", "value", "reward", "logp", "action")}
    return x, {"case": asdict(case), "layout_id": run["metrics"]["layout_id"], "metrics": run["metrics"],
               "focus": None if focus is None else focus + 1, "start_station": run["start_station"],
               "teacher": run["teacher"]}, []


def evaluation_job(job):
    case, arms, tapes, compare_reader = job
    global _V9_READER
    from recovery_reader import load_bundle, predict
    outcomes = {}
    for k, noise in enumerate(tapes):
        by_arm = {}
        baseline = Live(case, noise)
        acquired = rollout(baseline, None, case.action_seed, record=False)
        by_arm["MPC_new"] = acquired["metrics"]
        if compare_reader:
            if _V9_READER is None:
                _V9_READER = load_bundle(V9 / "run" / "reader_bundle.json")
            p = baseline.get()[0]
            legacy_m, _ = baseline.scan.grade(p, predict(p, _V9_READER, case.task))
            by_arm["MPC_v9"] = by_arm["MPC_new"] | legacy_m
        for name, policy, greedy in arms:
            run = rollout(Live(case, noise), actor(policy), case.action_seed, greedy, record=False)
            by_arm[name] = run["metrics"]
        layout = acquired["metrics"]["layout_id"]
        if not all(m["layout_id"] == layout and m["completed_legal"] for m in by_arm.values()):
            raise RuntimeError("The independent routes are not legal paired layouts")
        outcomes[("actual", "proof_0", "proof_1")[k]] = by_arm
    return {}, {"case": asdict(case), "layout_id": layout, "outcomes": outcomes}, []


def _save_snapshots(path, snapshots):
    path.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0] + [len(s["labels"]) for s in snapshots])
    array = {key: np.concatenate([s["public"][key] for s in snapshots]) for key in PUBLIC_FIELDS}
    array |= {key: np.concatenate([s[key] for s in snapshots]) for key in ("labels", "instance")}
    array["features"] = array["features"].astype(np.float16)
    temporary = path / "snapshots.pending"
    with temporary.open("wb") as f:
        np.savez_compressed(f, offsets=offsets, **array)
    temporary.replace(path / "snapshots.npz")
    temporary = path / "metadata.pending"
    with temporary.open("w") as f:
        for snap in snapshots:
            f.write(json.dumps({k: v for k, v in snap.items() if k not in ("public", "labels", "instance")}) + "\n")
    temporary.replace(path / "snapshots.jsonl")
    atomic_json(path / "snapshot_commit.json", {name: digest(path / name) for name in ("snapshots.npz", "snapshots.jsonl")})


def _snapshots_verified(path):
    return all(digest(path / name) == expected for name, expected in read_json(path / "snapshot_commit.json").items())


def collect(pool, root, cases, worker, args, *, mode="none", save_snapshots=False, chunk=12):
    x_paths, logs = [], []
    for offset in range(0, len(cases), chunk):
        selected = cases[offset:offset + chunk]
        dest = root / f"{offset//chunk:04d}"
        snapshots_path = dest / "observer"
        if not (dest / "commit.json").exists():
            completed = list(pool.map(worker, (args(c, offset + i) for i, c in enumerate(selected)), chunksize=1))
            samples, rows, saved = zip(*completed, strict=True)
            if mode == "strong":
                arrays = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
            elif mode == "online":
                arrays = {k: np.concatenate([s[k] for s in samples]) for k in samples[0]}
                arrays["offsets"] = np.cumsum([0] + [len(s["reward"]) for s in samples])
            else:
                arrays = {"cases": np.arange(len(samples))}
            if save_snapshots:
                _save_snapshots(snapshots_path, [s for data in saved for s in data])
            commit_shard(dest, arrays, rows)
        p, rows = verify_shard(dest)
        if save_snapshots and not _snapshots_verified(snapshots_path):
            raise RuntimeError("A reader data shard no longer matches its digest")
        ids = [r["case"]["identifier"] if isinstance(r["case"], dict) else r["case"] for r in rows]
        if ids != [c.identifier for c in selected]:
            raise RuntimeError("Mismatched completion roster")
        x_paths.append(p)
        logs.extend(rows)
        print(f"{root.name} {offset+len(rows)}/{len(cases)}", flush=True)
    if len({r["layout_id"] for r in logs}) != len(cases):
        raise RuntimeError("Layout duplicated inside the run")
    return x_paths, logs


def learning_rows(paths, logs=None):
    out = []
    head = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            keys = [key for key in arrays.files if key != "offsets"]
            if logs is None:
                for i in range(len(arrays["x"])):
                    out.append({key: arrays[key][i] for key in ("x", "mask", "values")})
                    if "follow_mask" in arrays:
                        for x, mask, values in zip(arrays["follow_x"][i], arrays["follow_mask"][i],
                                                   arrays["follow_values"][i], strict=True):
                            if mask.any():
                                out.append({"x": x, "mask": mask, "values": values})
            else:
                for first, last in zip(arrays["offsets"][:-1], arrays["offsets"][1:], strict=True):
                    experience = [{key: arrays[key][i] for key in keys} for i in range(first, last)]
                    out.append({"experience": experience, "teacher": logs[head]["teacher"]})
                    head += 1
    return out


def teacher_report(rows):
    output = {}
    for route in ROUTES:
        sub = [r for r in rows if r["route"] == route]
        base, chosen = [r["proof_baseline"] for r in sub], [r["proof_teacher"] for r in sub]
        strict = [int(c["strict"]) - int(b["strict"]) for b, c in zip(base, chosen, strict=True)]
        f = [r["focus"]-1 for r in sub]
        local = [int(c["selected_ids"][j] == 0) - int(b["selected_ids"][j] == 0)
                 for b, c, j in zip(base, chosen, f, strict=True)]
        output[route] = {"layouts": len(sub), "baseline_strict": sum(b["strict"] for b in base),
                         "teacher_strict": sum(c["strict"] for c in chosen),
                         "rescue": strict.count(1), "loss": strict.count(-1),
                         "target_region_fixed": local.count(1), "target_region_lost": local.count(-1),
                         "teacher_wrong": sum(c["wrong_branch"] for c in chosen),
                         "baseline_wrong": sum(b["wrong_branch"] for b in base),
                         "replayed": sum(r["valid_replay"] for r in sub),
                         "post_recovery_labels": sum(r.get("post_recovery_labels", 0) for r in sub),
                         "two_step_programs": sum(len(r["sequence"]) == 2 for r in sub),
                         "positive_deploy_commands": sum(any(g["skill"] not in ("advance", "submit") for g in r["sequence"]) for r in sub),
                         "selection_avg": float(np.mean([r["selection_gain"] for r in sub])),
                         "proof_avg": float(np.mean([r["proof_gain"] for r in sub]))}
    return output


def ci(d):
    gen = np.random.default_rng(40_998_431)
    d = np.asarray(d, dtype=float)
    sims = d[gen.integers(0, len(d), (4000, len(d)))].mean(axis=1)
    return [float(d.mean()), *np.percentile(sims, (2.5, 97.5)).tolist()]


def measure(rows):
    out = {}
    for route in ROUTES:
        r = [x["outcomes"] for x in rows if x["case"]["task"] == route]
        arm_names = r[0]["actual"]
        tapes = tuple(r[0])
        out[route] = {}
        for name in arm_names:
            actual = [y["actual"][name] for y in r]
            evidence = Counter()
            actions = Counter()
            for y in actual:
                actions.update(y["actions"])
                evidence.update(layouts=1, strict=y["strict"], wrong_branch=y["wrong_branch"], seen_all=y["acquired_ceiling"],
                                repair_count=y["repairs"], backward=y["negative_station_moves"],
                                earlier_region=y["earlier_region_visits"], multiple_repair_regions=len(y["recovery_regions"]) >= 2)
            result = dict(evidence) | {"actions": dict(actions),
                      "per_region_correct": np.sum([np.asarray(y["selected_ids"]) == 0 for y in actual], axis=0).tolist(),
                      "per_region_missing": np.sum([~np.asarray(y["capture"]) for y in actual], axis=0).tolist(),
                      "per_region_wrong_named": np.sum([np.asarray(y["selected_ids"]) > 0 for y in actual], axis=0).tolist(),
                      "per_region_seen_rejected": np.sum([np.asarray(y["capture"]) & (np.asarray(y["selected_ids"]) == -99)
                                                          for y in actual], axis=0).tolist(),
                      "all_seen_failed": sum(y["acquired_ceiling"] and not y["strict"] for y in actual),
                      "time_s": float(np.mean([y["time_s"] for y in actual])),
                      "travel_mm": float(np.mean([y["travel_mm"] for y in actual])),
                      "correct_mm": float(np.mean([y["mm"] for y in actual])),
                      "longest_mm": float(np.mean([y["longest_mm"] for y in actual]))}
            result["strict_by_tape"] = [sum(y[t][name]["strict"] for y in r) for t in tapes]
            result["wrong_by_tape"] = [sum(y[t][name]["wrong_branch"] for y in r) for t in tapes]
            result["paired"] = {}
            for ref in ("MPC_new", "MPC_v9", "student_BC"):
                if ref == name or ref not in arm_names:
                    continue
                contrast = {}
                for m in ("strict", "wrong_branch", "acquired_ceiling", "time_s", "travel_mm"):
                    d = np.array([[float(y[t][name][m]) - float(y[t][ref][m]) for t in tapes] for y in r])
                    contrast[m] = {"add": int((d[:, 0] > 0).sum()), "loss": int((d[:, 0] < 0).sum()),
                                   "actual95": ci(d[:, 0])}
                    if len(tapes) > 1:
                        contrast[m]["proof95"] = ci(d[:, 1:].mean(axis=1))
                result["paired"][ref] = contrast
            out[route][name] = result
    return out


def pool(args, reader):
    return ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                               initializer=initialize, initargs=(args.atlas_pack, reader))


def training(args, out, initial_reader, seed):
    strong_out, reader_out, skill0 = out / "strong", out / "reader_data", out / "skill_v9_reader"
    if (out / "adapted_reader" / "reader_manifest.json").exists():
        return
    with pool(args, initial_reader) as work:
        smoke = list(roster("smoke", 16))
        p, preflight = collect(work, out / "smoke", smoke, strong_job, lambda c, _: (c, 1, False), mode="strong")
        if not (out / "smoke_skill" / "training.json").exists():
            fit_options(learning_rows(p), out / "smoke_skill", seed-3, iterations=30)
        atomic_json(out / "smoke_summary.json", teacher_report(preflight))
        p, reports = collect(work, strong_out, list(roster("strong", args.strong)), strong_job,
                             lambda c, _: (c, 2, False), mode="strong")
        atomic_json(out / "strong_training_summary.json", teacher_report(reports))
        if not (skill0 / "training.json").exists():
            fit_options(learning_rows(p), skill0, seed, iterations=185)
        snapshot_sets = ((args.reader_train, "reader_train", "train"), (args.reader_cal, "reader_cal", "cal"),
                         (args.reader_dev, "reader_dev", "dev"))
        for count, stage, name in snapshot_sets:
            collect(work, reader_out / name, list(roster(stage, count)), collect_reader,
                    lambda c, _: (c, skill0 / "actor.json", name), save_snapshots=True)
    from recovery_reader import fit_reader
    fit_reader(reader_out, initial_reader, out / "adapted_reader", seed+31, epochs=args.reader_epochs, batch_size=16)


def run(args):
    out = args.out.resolve()
    gpu = read_json(out / "gpu_preflight.json")
    if not gpu.get("cuda_optimizer_step_verified"):
        raise RuntimeError("GPU preflight did not complete")
    seed = 261124101
    initial_reader = V9 / "run" / "reader_bundle.json"
    settings = {"base": "f115a663ec260779bc6a7dcb9b17e88a1354dd71", "sources": {name: digest(ROOT / name) for name in _MODULES},
                "v9_sources": {name: digest(V9 / name) for name in ("public_v9.py", "study_v9.py", "scan_v9.py", "reader_v9.py")},
                "atlas_sha256": digest(args.atlas_pack), "initial_reader_sha256": digest(initial_reader),
                "gpu_preflight_sha256": digest(out / "gpu_preflight.json"), "seeds": ROSTER, "optimizer_seed": seed,
                "search_tapes": SEARCH_TAPES, "teacher_proof_tape": TEACH_PROOF, "test_tapes": EVAL_TAPES,
                "cohorts": {k: v for k, v in vars(args).items() if isinstance(v, int) and k != "workers"},
                "dird": False, "motion": "contact and speed checked; no run-level motion allowance; elapsed and distance recorded",
                "repair_policy": "compare routine progress with one normalized group of unresolved goals; settled regular views stay closed",
                "readiness": "accepted regular regions; acceptance, score >=.86 and target probability >=.65 for later problem regions",
                "selection": "shared reader, actual greedy full-route dev first whole strict then wrong; femoral strict then wrong"}
    reuse = out / "reused_motion_trials.json"
    if reuse.exists():
        settings["reused_motion_trials_sha256"] = digest(reuse)
    lineage = out / "reader_collection_lineage.json"
    if lineage.exists():
        settings["reader_collection_lineage_sha256"] = digest(lineage)
    protocol_hash = freeze(out, "protocol.json", settings)
    training(args, out, initial_reader, seed)
    adapted = out / "adapted_reader" / "reader_bundle.json"
    reader_hash = digest(adapted)
    freeze(out, "compose_freeze.json", {"protocol_sha256": protocol_hash, "reader_sha256": reader_hash})
    with pool(args, adapted) as work:
        folder = out / "compose_teacher"
        p, found = collect(work, folder, list(roster("compose_teacher", args.compose_teacher)), strong_job,
                           lambda c, _: (c, 2, False, True), mode="strong")
        atomic_json(out / "strong_composition_summary.json", teacher_report(found))
        # A fresh layout subset and a distinct proof tape evaluate the teacher's
        # ability to preserve existing regions before calling it a demonstration.
        _, expert_dev = collect(work, out / "strong_dev", list(roster("adapt", args.teacher_dev)), strong_job,
                                lambda c, _: (c, 2, False))
        atomic_json(out / "strong_development_summary.json", teacher_report(expert_dev))
        skill_path = out / "skill_adapted"
        if not (skill_path / "training.json").exists():
            fit_options(learning_rows(p), skill_path, seed+44, iterations=220)
        initial = skill_path / "actor.json"
        dev, candidates = list(roster("dev", args.dev)), []
        behavior = initial
        histories = []
        for r in range(args.rounds+1):
            if r:
                folder = out / "online" / f"{r:02d}"
                declared = list(roster("online", args.online_batch, r))
                lock = freeze(out, f"online_{r:02d}.json", {"protocol_sha256": protocol_hash, "reader_sha256": reader_hash,
                                                          "behavior_actor_sha256": digest(behavior), "batch": r})
                inputs, logs = collect(work, folder, declared, online_job,
                                       lambda c, i: (c, behavior, i % 4 == 0, i % 2 == 0), mode="online")
                if not (folder / "training.json").exists():
                    update_online(learning_rows(inputs, logs), behavior, folder, seed+60+r)
                counts = read_json(folder / "training.json")
                counts |= {"full_start": sum(q["focus"] is None for q in logs),
                           "prefix_start": sum(q["focus"] is not None for q in logs),
                           "protocol_sha256": lock, "shards_sha256": [digest(p.parent / "commit.json") for p in inputs]}
                atomic_json(folder / "evidence.json", counts)
                histories.append(counts)
                behavior = folder / "actor.json"
            _, check = collect(work, out / "development" / f"{r:02d}", dev, evaluation_job,
                               lambda c, _: (c, (("candidate", behavior, True),), (None,), True), chunk=12)
            dev_result = measure(check)
            atomic_json(out / "development" / f"{r:02d}" / "results.json", dev_result)
            whole, femoral = dev_result[ROUTES[1]]["candidate"], dev_result[ROUTES[0]]["candidate"]
            key = (whole["strict"], -whole["wrong_branch"], femoral["strict"], -femoral["wrong_branch"])
            candidates.append((key, behavior, r))
            print(f"Development {r} {key}", flush=True)
        key, deployed, selected_round = max(candidates, key=lambda x: x[0])
        policy_freeze = freeze(out, "independent_freeze.json", {"protocol_sha256": protocol_hash, "reader_sha256": reader_hash,
                                                               "selection_key": key, "candidate": selected_round, "student_sha256": digest(deployed),
                                                               "student_BC_sha256": digest(initial)})
        arms = (("student_BC", initial, True), ("student_BC_sample", initial, False), ("student", deployed, True))
        _, raw = collect(work, out / "independent", list(roster("test", args.test)), evaluation_job,
                         lambda c, _: (c, arms, EVAL_TAPES, True), chunk=12)
    if digest(adapted) != reader_hash or any(digest(ROOT / name) != sha for name, sha in settings["sources"].items()):
        raise RuntimeError("The online or evaluation run mutated a frozen training input")
    compact = []
    for r in raw:
        effects = {tape: {arm: {k: item[k] for k in ("strict", "wrong_branch", "selected_ids", "capture", "time_s", "travel_mm",
                                                     "mm", "actions", "recovery_regions", "negative_station_moves", "earlier_region_visits")}
                          for arm, item in methods.items()} for tape, methods in r["outcomes"].items()}
        compact.append({"case": r["case"], "layout_id": r["layout_id"], "outcomes": effects})
    atomic_json(out / "paired_outcomes.json", compact)
    results = {"protocol_sha256": protocol_hash, "freeze_sha256": policy_freeze, "paired_sha256": digest(out / "paired_outcomes.json"),
               "online": histories, "selected_round": selected_round,
               "independent_layouts": len(raw), "stats": measure(raw)}
    atomic_json(out / "results.json", results)
    print({route: {arm: (a["strict"], a["wrong_branch"]) for arm, a in rows.items()} for route, rows in results["stats"].items()},
          flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--strong", type=int, default=144)
    p.add_argument("--reader-train", type=int, default=132)
    p.add_argument("--reader-cal", type=int, default=36)
    p.add_argument("--reader-dev", type=int, default=48)
    p.add_argument("--reader-epochs", type=int, default=8)
    p.add_argument("--compose-teacher", type=int, default=168)
    p.add_argument("--teacher-dev", type=int, default=48)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--online-batch", type=int, default=64)
    p.add_argument("--dev", type=int, default=60)
    p.add_argument("--test", type=int, default=160)
    run(p.parse_args())


if __name__ == "__main__":
    main()
