"""Mine v10 failures and test evidence-conditioned identity recovery on new layouts."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from evidence import ROOT, V10, execute_view, information, labels_for, query, resolve, update_belief, views
from recovery_task import DEFAULT_PACK, ROUTES, Live, initialize, roster
from recovery_policy import read_bundle, rollout
from recovery_reader import load_bundle, predict
from run_recovery import _save_snapshots
from study_v9 import atomic_json, commit_shard, digest, freeze, read_json, verify_shard


V10_RUN = V10 / "run"
READER = V10_RUN / "adapted_reader" / "reader_bundle.json"
SEEDS = {"mine": 8_158_690_000_000_000, "cal": 8_158_691_000_000_000, "dev": 8_158_692_000_000_000,
         "teach": 8_158_693_000_000_000, "select": 8_158_694_000_000_000, "test": 8_158_695_000_000_000}
REVEALS = (32_000_139, 42_000_193)
PROOF = 45_000_399
TEST_TAPES = (None, 104_231_865, 104_251_253)
_ACTOR = _V10_READER = _WORK_READER = _MIXED_READER = _MODEL = None


def cases(stage, count):
    family = "test" if stage == "test" else "dev" if stage in ("dev", "select") else "reader_train"
    for j, c in enumerate(roster(family, count)):
        seed = SEEDS[stage] + j * 29
        yield replace(c, identifier=f"v11_{stage}_{j:05d}", split=f"v11_{stage}", world_seed=seed, tape_seed=seed+1,
                      action_seed=seed+2, challenge=False)


def init(atlas, reader, actor, model=None, mixed_reader=None):
    global _ACTOR, _V10_READER, _WORK_READER, _MIXED_READER, _MODEL
    initialize(atlas, READER)
    _ACTOR = read_bundle(actor)
    _V10_READER = load_bundle(READER)
    _WORK_READER = load_bundle(reader)
    _MIXED_READER = None if mixed_reader is None else load_bundle(mixed_reader)
    if model is not None:
        from evidence_model import load_model
        _MODEL = load_model(model)


def acquire(case, offset):
    live = Live(case, offset)
    rollout(live, _ACTOR, case.action_seed, greedy=True, record=False)
    p = live.scan.prefix.update(live.scan.node)
    return live, p


def assessed(live, public, reader):
    p = predict(public, reader, live.case.task)
    return p, live.scan.grade(public, p)[0]


def mine_job(case):
    layout, public = acquire(case, None)
    p, grade = assessed(layout, public, _V10_READER)
    assessments, candidates = [], []
    targeted = (5,) if case.task == "femoral" else (3, 4, 5)
    for r in targeted:
        state = query(public, p, r)
        if state is None:
            assessments.append({"region": r+1, "capture": False, "anchor": False})
            continue
        truth = labels_for(layout, state, public)
        high = state.incumbent_accepted and state.incumbent_score >= .86 and state.incumbent_target >= .65
        assessments.append({"region": r+1, "capture": grade["capture"][r], "anchor": True,
                            "old_high_and_closed": high, "old_score": state.incumbent_score,
                            "old_target": state.incumbent_target, **truth,
                            "selected": grade["selected_ids"][r], "has_alternative": state.anchors[1] is not None})
        priority = 1.5 * (grade["selected_ids"][r] != 0) + .75 * (not grade["capture"][r]) + .45 * (r == 4)
        if high and truth["anchor_labels"][0] != 0:
            priority += 1.5
        candidates.append((priority, r, state))
    split = {"v11_mine": "train", "v11_cal": "cal", "v11_dev": "dev"}[case.split]
    category = "normal" if grade["strict"] else "hard"
    data = [layout.snapshot(split, f"{category}_student_terminal")]
    options = []
    if split == "train" and candidates:
        _, r, state = max(candidates, key=lambda x: x[0])
        options = views(layout, state)
        if options:
            idx = int(np.random.default_rng(case.action_seed).integers(1, len(options)))
            future, _, _, end, evidence = execute_view(layout, options[idx], state, _V10_READER, REVEALS[0])
            data.append(future.snapshot("train", "recovery_student_terminal"))
    return {}, {"case": asdict(case), "layout_id": grade["layout_id"], "terminal": outcome(grade),
                "regions": assessments, "additional_observed_trajectory": bool(options)}, data


def outcome(grade):
    return {key: grade[key] for key in ("strict", "wrong_branch", "wrong_region_count", "selected_ids", "capture",
                                        "mm", "longest_mm", "completed_legal", "time_s", "travel_mm")}


def deltas(old, new, region):
    a, b = old["selected_ids"][region], new["selected_ids"][region]
    return [int(new["strict"])-int(old["strict"]), int(b == 0)-int(a == 0),
            sum(v > 0 for v in new["selected_ids"]) - sum(v > 0 for v in old["selected_ids"]),
            sum(v in (-1, -2) for v in new["selected_ids"]) - sum(v in (-1, -2) for v in old["selected_ids"])]


def teacher_job(case):
    live, p = acquire(case, None)
    pred, old = assessed(live, p, _WORK_READER)
    records = []
    array = {"context": [], "features": [], "identity": [], "event": [], "deltas": [], "row": []}
    if case.task != "femoral":
        flagged = [r for r in (3, 4, 5) if old["selected_ids"][r] != 0]
        number = int(case.identifier.rsplit("_", 1)[-1])
        # A fixed panel and an independently chosen difficult region preserve
        # negative outcomes alongside real failures and avoid duplicate rows.
        agenda = list(dict.fromkeys((4 if number % 3 != 2 else 5, *(flagged[:1] or (3,)))))
        for region in agenda:
            state = query(p, pred, region)
            allowed = [] if state is None else views(live, state)
            if not allowed:
                continue
            truth = labels_for(live, state, p)
            panel = []
            mean = []
            for choice in allowed:
                trials = []
                for offset in REVEALS:
                    _, _, _, end, evidence = execute_view(live, choice, state, _WORK_READER, offset)
                    delta = deltas(old, end, region)
                    for field, value in (("context", state.features), ("features", choice.features),
                                         ("identity", truth["hypothesis"]), ("event", evidence["event"]),
                                         ("deltas", delta), ("row", len(records))):
                        array[field].append(value)
                    trials.append(evidence | {"delta": delta})
                panel.append(trials)
                mean.append(float(np.mean([teacher_utility(item["delta"]) for item in trials])))
            best = int(np.argmax(mean))
            _, _, _, proof, details = execute_view(live, allowed[best], state, _WORK_READER, PROOF)
            _, _, _, repeated, repeat_detail = execute_view(live, allowed[0], state, _WORK_READER, PROOF)
            records.append({"region": region+1, **truth, "closed_high": bool(
                state.incumbent_accepted and state.incumbent_score >= .86 and state.incumbent_target >= .65),
                "before": outcome(old), "search": panel, "chosen": allowed[best].name,
                "proof_teacher": outcome(proof), "proof_repeat": outcome(repeated), "proof_event": details,
                "repeat_event": repeat_detail})
    return array, {"case": asdict(case), "layout_id": old["layout_id"], "queries": records}, []


def teacher_utility(delta):
    return float(np.dot(delta, (1., .24, -.40, -.16)))


def control_job(job):
    case, tapes, settings = job
    from evidence_model import infer, prior_for

    panel = {}
    for tag, offset in zip(("actual", "proof_0", "proof_1"), tapes):
        live, public = acquire(case, offset)
        pred, baseline = assessed(live, public, _WORK_READER)
        _, v10 = assessed(live, public, _V10_READER)
        if case.task == "femoral":
            baseline = v10
        observed = {k: outcome(baseline) for k in ("base", "prior_readout", "repeat", "geometry", "information",
                                                   "task", "task_old_image")}
        observed["v10"] = outcome(v10)
        if _MIXED_READER is not None:
            observed["mixed_readout"] = outcome(assessed(live, public, _MIXED_READER)[1])
        detail = {}
        if case.task != "femoral":
            contenders = []
            for r in (4, 5, 3):
                state = query(public, pred, r)
                if state is not None:
                    prior = prior_for(_MODEL, state)
                    urgency = 1. - prior[0] + (.035 if r == 4 else 0.)
                    if urgency >= settings["trigger"]:
                        contenders.append((urgency, state, prior))
            for _, state, prior in sorted(contenders, key=lambda x: x[0], reverse=True):
                legal = views(live, state)
                if not legal:
                    continue
                likelihood, change = infer(_MODEL, legal)
                gains = [information(prior, l) for l in likelihood]
                task = [teacher_utility(d) + .30*g for d, g in zip(change, gains, strict=True)]
                observed["prior_readout"] = outcome(live.scan.grade(
                    public, resolve(state, public, pred, prior, settings["trust"]))[0])
                if max(task) < settings["advantage"]:
                    break
                choices = {"repeat": 0, "geometry": int(np.argmax([v.visibility for v in legal])),
                           "information": int(np.argmax(gains)), "task": int(np.argmax(task))}
                cache = {}
                for arm, index in choices.items():
                    if index not in cache:
                        cache[index] = execute_view(live, legal[index], state, _WORK_READER)
                    branch, updated, new, measured, effect = cache[index]
                    observed[arm] = outcome(measured)
                    detail[arm] = effect | {"predicted_information": gains[index], "task_value": task[index]}
                    if arm == "task":
                        post = update_belief(prior, likelihood[index], effect["event"])
                        unambiguous_new_support = effect["event"] == 1 and post[1] > prior[1] + .025
                        readout = resolve(state, updated, new, post, settings["trust"]) if unambiguous_new_support else new
                        phenotype = branch.scan.grade(updated, readout)[0]
                        observed["task_old_image"] = outcome(phenotype)
                        detail["memory"] = {"changed": readout is not new, "old_hypothesis_prior": prior.tolist(),
                                            "posterior": post.tolist(), "anchors": labels_for(live, state, public),
                                            "frame_slot_keys": [a.key if a else None for a in state.anchors],
                                            "region": state.region + 1}
                detail["target_region"] = state.region + 1
                break
        if not all(x["completed_legal"] for x in observed.values()):
            raise RuntimeError("An arm in the panel did not complete")
        panel[tag] = {"arms": observed, "executed": detail}
    return {}, {"case": asdict(case), "layout_id": baseline["layout_id"], "tapes": panel}, []


def collect(pool, folder, roster_cases, fn, setup=None, save_reader=False):
    records, paths = [], []
    for first in range(0, len(roster_cases), 12):
        chunk = roster_cases[first:first+12]
        where = folder / f"{first//12:04d}"
        if not (where / "commit.json").is_file():
            done = list(pool.map(fn, chunk if setup is None else [setup(c) for c in chunk]))
            features, outcomes, snapshots = zip(*done, strict=True)
            packed = ({"counts": np.array([len(r["queries"]) for r in outcomes])} if fn is teacher_job else
                      {"rows": np.arange(len(chunk))})
            if fn is teacher_job:
                for field in features[0]:
                    elements = [x for y in features for x in y[field]]
                    packed[field] = np.stack(elements) if elements else np.empty(0)
                packed["layout_index"] = np.array([i for i, y in enumerate(features) for _ in y["identity"]])
            if save_reader:
                _save_snapshots(where / "observer", [v for row in snapshots for v in row])
            commit_shard(where, packed, outcomes)
        path, row = verify_shard(where)
        if [r["case"] for r in row] != [asdict(c) for c in chunk]:
            raise RuntimeError("Unexpected layout roster in the committed data")
        records.extend(row)
        paths.append(path)
        print(folder.name, len(records), "/", len(roster_cases), flush=True)
    if len({r["layout_id"] for r in records}) != len(records):
        raise RuntimeError("Repeated layout in the same cohort")
    return paths, records


def pool(args, reader, actor, model=None, mixed_reader=None):
    return ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=init,
                               initargs=(args.atlas, reader, actor, model, mixed_reader))


def audit(rows):
    result = {}
    for route in ROUTES:
        selected = [r for r in rows if r["case"]["task"] == route]
        per_region = {}
        for i in (4, 5, 6):
            c = [x for r in selected for x in r["regions"] if x["region"] == i]
            c = [x for x in c if x["anchor"]]
            wrong = [x for x in c if x["old_high_and_closed"] and x["anchor_labels"][0] != 0]
            per_region[i] = {"anchors": len(c), "misses": sum(not x["capture"] for x in c),
                             "high_confidence_wrong": len(wrong),
                             "named": sum(x["anchor_labels"][0] > 0 for x in wrong),
                             "artifact_or_mix": sum(x["anchor_labels"][0] in (-1, -2) for x in wrong),
                             "old_alternative_true": sum(x["hypothesis"] == 1 for x in wrong)}
        result[route] = {"layouts": len(selected), "strict": sum(r["terminal"]["strict"] for r in selected), "regions": per_region}
    return result


def run(args):
    out = args.out
    if not read_json(out / "gpu_preflight.json")["cuda_optimizer_step_verified"]:
        raise RuntimeError("CUDA is required")
    frozen = read_json(V10_RUN / "independent_freeze.json")
    actor = ((V10_RUN / "online" / f"{frozen['candidate']:02d}") if frozen["candidate"] else
             V10_RUN / "skill_adapted") / "actor.json"
    if digest(actor) != frozen["student_sha256"] or digest(READER) != frozen["reader_sha256"]:
        raise RuntimeError("v10 lock mismatch")
    with pool(args, READER, actor) as work:
        mined = []
        for name, n in (("mine", args.mine), ("cal", args.cal), ("dev", args.dev)):
            _, rows = collect(work, out / "acquisition" / name, list(cases(name, n)), mine_job, save_reader=True)
            mined.extend(rows)
    atomic_json(out / "confidence_audit.json", audit(mined))
    if args.acquire_only:
        return
    from pipeline import finish
    finish(args, out, actor, mined)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--mine", type=int, default=144)
    p.add_argument("--cal", type=int, default=48)
    p.add_argument("--dev", type=int, default=48)
    p.add_argument("--teach", type=int, default=288)
    p.add_argument("--select", type=int, default=72)
    p.add_argument("--test", type=int, default=120)
    p.add_argument("--acquire-only", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
