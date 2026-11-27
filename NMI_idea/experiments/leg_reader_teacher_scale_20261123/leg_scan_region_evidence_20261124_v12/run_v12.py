"""Freeze and evaluate a student that can revisit several anatomical regions."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import data_v12
from closed_scan import Engine
from evidence import execute_view, information, query, views
from evidence_model import infer as v11_infer, load_model, prior_for
from fit_v12 import hydrate, plain
from pipeline import paired_delta
from recovery_reader import load_bundle, predict as read_reader
from recovery_task import DEFAULT_PACK, Live
from region_policy import SEARCH, VERIFY_TAPE, fit_actor, pack_teacher, roll, summarize_teacher, teacher_job
from replay_v11 import MIXED, ROOT, RUN11, actor_path
from run_v11 import READER, outcome, teacher_utility
from study_v9 import atomic_json, commit_shard, digest, freeze, read_json, verify_shard
from track_state import from_logits, linear, features
from scipy.special import expit, softmax


FIRST = {"teach_fit": 132, "teach_cal": 42}
REFINE = {"adapt_fit": 48, "adapt_cal": 24}
POLICY_DEV = 48
TEST = 108
FEMORAL = 36
TAPES = (None, 176_145_763, 176_215_945)
PRESET = {"margin": .06, "r5_weight": .035}
_V11_MODEL = _V11_READER = None


def boot(atlas, actor, association, reader, blend, compare):
    global _V11_MODEL, _V11_READER
    data_v12.init(atlas, actor, association, reader, blend)
    if compare:
        _V11_READER = load_bundle(MIXED)
        _V11_MODEL = load_model(RUN11/"model"/"model.json")


def pool(args, association, reader, blend, compare=False):
    return ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=boot,
                                initargs=(args.atlas, actor_path(), association, reader, blend, compare))


def roster(name, n):
    purpose = {"adapt_fit": "teach_fit", "adapt_cal": "teach_cal"}.get(name)
    if purpose is None:
        return list(data_v12.cases(name, n))
    from dataclasses import replace

    return [replace(c, identifier=f"{c.identifier}_adapt_{i:05d}", split=f"v12_{name}",
                    world_seed=c.world_seed + 320_000, tape_seed=c.tape_seed + 320_000,
                    action_seed=c.action_seed + 320_000) for i, c in enumerate(data_v12.cases(purpose, n))]


def gather(work, root, name, n, worker, setup, packer, meta):
    planned = roster(name, n)
    if len(planned) != n:
        raise RuntimeError("The planned roster is incomplete")
    arrays, records = [], []
    for first in range(0, n, 12):
        at = root/name/f"{first//12:04d}"
        group = planned[first:first+12]
        if not (at/"commit.json").exists():
            done = list(work.map(worker, map(setup, group)))
            commit_shard(at, packer(done), [meta(x) for x in done])
        path, rows = verify_shard(at)
        if [r["case"] for r in rows] != [asdict(c) for c in group]:
            raise RuntimeError("Cached data differ from the frozen schedule")
        arrays.append(path)
        records.extend(rows)
        print(root.name, name, len(records), "/", n, flush=True)
    if len({r["layout_id"] for r in records}) != n:
        raise RuntimeError("One cohort contains repeat layouts")
    return arrays, records


def _load_targets(paths):
    blocks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as block:
            blocks.append(dict(block))
    return {k: np.concatenate([b[k] for b in blocks], axis=0) for k in blocks[0]}


def train_stages(args, edge, reader, blend):
    out = args.out
    root = out/"teacher"
    models = out/"models"
    panels, buckets = {}, {}
    with pool(args, edge, reader, blend) as work:
        for name, nr in FIRST.items():
            shards, meta = gather(work, root, name, nr, teacher_job, lambda c: (c, "one_step_v10", None),
                                  pack_teacher, lambda x: x[2])
            panels[name], buckets[name] = meta, _load_targets(shards)
    if not (models/"student_r0.json").exists():
        first = fit_actor(buckets["teach_fit"], buckets["teach_cal"], models, "student_r0", 34264)
    else:
        first = hydrate(read_json(models/"student_r0.json"))
    follow = {"model": first, "config": PRESET}
    follow_hash = freeze(out, "teacher_adaptation_freeze.json", {"student_sha256": digest(models/"student_r0.json"),
                         "follow_rule": PRESET, "eval_tapes": (SEARCH, VERIFY_TAPE), "rows": REFINE})
    with pool(args, edge, reader, blend) as work:
        for name, nr in REFINE.items():
            shards, meta = gather(work, root, name, nr, teacher_job, lambda c: (c, "current_student_suffix", follow),
                                  pack_teacher, lambda x: x[2])
            panels[name], buckets[name] = meta, _load_targets(shards)
    merged = {k: np.concatenate((buckets["teach_fit"][k], buckets["adapt_fit"][k]), axis=0)
              for k in buckets["teach_fit"]}
    if not (models/"student_r1.json").exists():
        revised = fit_actor(merged, buckets["adapt_cal"], models, "student_r1", 34266, start=first, epochs=16)
    else:
        revised = hydrate(read_json(models/"student_r1.json"))
    atomic_json(out/"teacher_summary.json", plain({name: summarize_teacher(rows) for name, rows in panels.items()} |
                {"adaptation_freeze_sha256": follow_hash}))
    return (first, revised), panels


def _grade(live, public, prediction):
    grad = live.scan.grade(public, prediction)[0]
    if not grad["completed_legal"]:
        raise RuntimeError("An accepted test arm did not reach the terminal station")
    return outcome(grad)


def _v11_window(engine, live, at_end):
    # Match the first v11 frozen selector and use exactly one execution for
    # both frozen readouts. Retain the same duration and exit pose.
    p = at_end.public
    mix = read_reader(p, _V11_READER, live.case.task)
    contenders = []
    for r in (4, 5, 3):
        state = query(p, mix, r)
        if state is None:
            continue
        b = prior_for(_V11_MODEL, state)
        urgency = 1. - b[0] + (.035 if r == 4 else 0.)
        if urgency >= .20:
            contenders.append((urgency, state, b))
    for _, state, prior in sorted(contenders, key=lambda x: x[0], reverse=True):
        menu = views(live, state)
        if not menu:
            continue
        l, delta = v11_infer(_V11_MODEL, menu)
        value = [teacher_utility(d) + .30*information(prior, q) for d, q in zip(delta, l, strict=True)]
        if max(value) < 0.:
            break
        selected = max(menu, key=lambda a: a.visibility)
        branch, q, _, measured, effect = execute_view(live, selected, state, _V11_READER)
        return (outcome(measured), _grade(branch, q, read_reader(q, engine.old, live.case.task)),
                effect | {"region": state.region+1})
    return _grade(live, p, mix), _grade(live, p, at_end.old), {}


def _review_stats(live, at_end, actions):
    from study_worlds import truth_at

    truth = truth_at(live.scan.node, at_end.public, live.scan.tape)
    indexes = {k: i for i, k in enumerate(zip(at_end.public["frames"], at_end.public["slots"], strict=True))}
    for action in actions:
        for track in action["objects"]:
            i = indexes[tuple(track["key"])]
            track["identity_label"] = int(truth["labels"][i])
        if action["keys"]:
            k = action["keys"][1 if action["program"] == "trace_b" else 0]
            first = truth["instance"][indexes[tuple(k)]] if k is not None else -1
            action["bridge_same_object"] = int(sum(first >= 0 and truth["instance"][indexes[tuple(key)]] == first
                                                   for key in action["bridge_keys"]))


def _masked_reread(engine, live, end):
    p, base = end.public, end.old
    x, ids = features(p, base, live.scan.env.scanner.surface, engine.edge, material=False)
    logits = linear(engine.decoder, x)
    no = from_logits(p, base, ids, softmax(logits[:, :8]/engine.decoder["temperature"], axis=1),
                     expit(logits[:, 8]/engine.decoder["rank_temperature"]), engine.blend)
    return _grade(live, p, no)


def validate_job(job):
    case, configs = job
    engine = data_v12._ENGINE
    base, panel = Live(case), {}
    engine.to_frontier(base, 3)
    base_end = base.fork()
    q, old = engine.finish(base_end, update=False)
    panel["v10"] = outcome(old)
    panel["same_track_new"] = _grade(base_end, q.public, engine.read(base_end).decision)
    if case.task == "groin_to_distal_calf":
        panel["v11_geometry_mixed"], panel["v11_geometry"], _ = _v11_window(engine, base_end, q)
    for label, model, config in configs:
        _, _, grade, _, _ = roll(engine, base.fork(), model, "joint", config)
        panel[label] = outcome(grade)
    return {"case": asdict(case), "layout_id": old["layout_id"], "arms": panel}


def paired_job(job):
    case, selected = job
    engine, model, config = data_v12._ENGINE, selected["model"], selected["config"]
    tapes = {}
    for tag, offset in zip(("actual", "proof_0", "proof_1"), TAPES, strict=True):
        start = Live(case, offset)
        engine.to_frontier(start, 3)
        base = start.fork()
        q, old = engine.finish(base, update=False)
        old_frame = outcome(old)
        results = {"v10": old_frame}
        paths = {}
        if case.task == "femoral":
            for k in ("v11_geometry", "v11_geometry_mixed", "same_track_new", "geometry_r5", "joint",
                      "joint_old_readout", "joint_no_material_readout", "sequential", "no_identity_feedback"):
                results[k] = old_frame
        else:
            results["same_track_new"] = _grade(base, q.public, engine.read(base).decision)
            results["v11_geometry_mixed"], results["v11_geometry"], paths["v11"] = _v11_window(engine, base, q)
            for arm in ("geometry_r5", "joint", "sequential", "no_identity_feedback"):
                branch, at_end, g, g_old, trace = roll(engine, start.fork(), model, arm, config)
                _review_stats(branch, at_end, trace)
                results[arm], paths[arm] = outcome(g), trace
                if arm == "joint":
                    results["joint_old_readout"] = outcome(g_old)
                    results["joint_no_material_readout"] = _masked_reread(engine, branch, at_end)
        tapes[tag] = {"arms": results, "actions": paths}
    return {"case": asdict(case), "layout_id": old["layout_id"], "tapes": tapes}


def summary(rows):
    panel = {}
    references = ("v10", "v11_geometry", "geometry_r5", "sequential", "no_identity_feedback", "joint_old_readout")
    for k in rows[0]["tapes"]["actual"]["arms"]:
        by = [r["tapes"]["actual"]["arms"][k] for r in rows]
        ids = np.asarray([x["selected_ids"] for x in by])
        panel[k] = {"strict": [sum(r["tapes"][t]["arms"][k]["strict"] for r in rows)
                              for t in ("actual", "proof_0", "proof_1")],
                    "wrong": [sum(r["tapes"][t]["arms"][k]["wrong_branch"] for r in rows)
                              for t in ("actual", "proof_0", "proof_1")],
                    "r5_arterial": int((ids[:, 4] == 1).sum()), "correct_regions": (ids == 0).sum(0).tolist(),
                    "captured_regions": np.asarray([r["capture"] for r in by]).sum(0).tolist(),
                    "mean_seconds": float(np.mean([r["time_s"] for r in by])),
                    "mean_mm": float(np.mean([r["travel_mm"] for r in by])),
                    "deltas": {ref: {m: paired_delta(rows, k, ref, m)
                               for m in ("strict", "wrong_branch")} for ref in references if ref != k}}
    events = {}
    for arm in ("geometry_r5", "joint", "sequential", "no_identity_feedback"):
        c = Counter()
        for r in rows:
            sample = r["tapes"]["actual"]["actions"].get(arm, [])
            c.update(episodes_with_review=bool(sample), episodes_with_multi_region=len({x["region"] for x in sample})>1)
            for p in sample:
                c.update(programs=1, before_last_station=p["before_end"], r5=p["region"]==5,
                         trace_programs=p["program"].startswith("trace"), anchored=p["program"].startswith("trace") and
                         p["reached"][0]==p["anchor_station"], branch_supported=p["bridge_same_object"]>=2,
                         r5_winner_changed=p["region"]==5 and p["old_r5"]!=p["new_r5"],
                         correct_target_identity_promoted=sum(o["identity_label"]==0 and o["role_before"]!=0 and
                             o["role_after"]==0 and o["new_supported_views"]>0 for o in p["objects"]),
                         correct_target_identity_lost=sum(o["identity_label"]==0 and o["role_before"]==0 and
                             o["role_after"]!=0 and o["new_supported_views"]>0 for o in p["objects"]))
        events[arm] = dict(c)
    return {"layouts": len(rows), "arms": panel, "events": events}


def _select_dev(rows, labels):
    res = {}
    for k in rows[0]["arms"]:
        values = [r["arms"][k] for r in rows]
        res[k] = (sum(r["strict"] for r in values), sum(r["wrong_branch"] for r in values),
                  sum(r["selected_ids"][4] == 0 for r in values))
    winner = max(labels, key=lambda k: (res[k][0], -res[k][1], res[k][2]))
    return winner, res


def run(args):
    out = args.out
    asset = out/"models"
    association = hydrate(read_json(asset/"association.json"))
    reader = hydrate(read_json(asset/"reader.json"))
    read_train = read_json(asset/"reader_training.json")
    blend = read_train["blend"]
    reader_manifest = read_json(asset/"reader_manifest.json")
    if reader_manifest["association_sha256"] != digest(asset/"association.json") or reader_manifest["reader_sha256"] != digest(
            asset/"reader.json"):
        raise RuntimeError("Unexpected reader or association weights")
    old = read_json(out/"reader_data_freeze.json")
    extra = ("data_v12.py", "closed_scan.py", "track_state.py", "fit_v12.py", "region_policy.py", "run_v12.py", "replay_v11.py")
    lock = freeze(out, "protocol.json", {"source": {k: digest(ROOT/k) for k in extra}, "atlas_sha256": digest(args.atlas),
                  "reader_data_sha256": digest(out/"reader_data_freeze.json"), "reader_manifest_sha256": digest(
                      asset/"reader_manifest.json"), "actor_sha256": digest(actor_path()), "v10_reader_sha256": digest(READER),
                  "geometry_reader_sha256": digest(MIXED), "v11_model_sha256": digest(RUN11/"model"/"model.json"),
                  "heldout_tapes": TAPES, "search_tapes": SEARCH, "teacher_proof": VERIFY_TAPE,
                  "settings_seed": data_v12.SEEDS, "refine_seed_offset": 320_000,
                  "main_comparison": "joint versus v11 geometry with v10 reader, strict delivery and wrong branch",
                  "counts": FIRST | REFINE | {"policy_dev": POLICY_DEV, "test": TEST, "femoral": FEMORAL}})
    if any(digest(ROOT/k) != v for k, v in old["sources"].items()):
        raise RuntimeError("Motion or public input code changed after reader collection")
    students, teachers = train_stages(args, association, reader, blend)
    configurations = [(f"r{i}_v{j}", s, c) for i, s in enumerate(students) for j, c in enumerate((
        {"margin": .015, "r5_weight": .065}, PRESET, {"margin": .10, "r5_weight": .02}))]
    with pool(args, association, reader, blend, compare=True) as work:
        _, dev = gather(work, out/"evaluation", "policy_dev", POLICY_DEV, validate_job, lambda c: (c, configurations),
                        lambda d: {"rows": np.arange(len(d))}, lambda d:d)
        candidate, report = _select_dev(dev, [c[0] for c in configurations])
        _, student, config = next(c for c in configurations if c[0] == candidate)
        model_asset = asset/f"student_{candidate[:2]}.json"
        selected = {"model": student, "config": config}
        atomic_json(out/"policy_development.json", {"selected": candidate, "outcomes": report})
        test_hash = freeze(out, "test_freeze.json", {"protocol_sha256": lock, "model": candidate, "rule": config,
                           "actor_sha256": digest(model_asset), "reader_sha256": digest(asset/"reader.json"),
                           "association_sha256": digest(asset/"association.json"), "blend": blend})
        independent = {}
        for name, count in (("test", TEST), ("femoral", FEMORAL)):
            _, panel = gather(work, out/"evaluation", name, count, paired_job, lambda c: (c, selected),
                              lambda d: {"rows": np.arange(len(d))}, lambda d:d)
            independent[name] = panel
    observed = [r["layout_id"] for rows in (*teachers.values(), dev, *independent.values()) for r in rows]
    fitted = [i for rows in reader_manifest["layouts"].values() for i in rows]
    all_ids = observed + fitted
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("A training or evaluation layout leaked across cohorts")
    output = {}
    for name, rows in independent.items():
        path = out/f"{name}_paired.json"
        atomic_json(path, rows)
        output[name] = summary(rows) | {"paired_sha256": digest(path)}
    atomic_json(out/"results.json", output | {"freeze_sha256": test_hash, "disjoint_layouts": len(all_ids)})
    print({arm: (s["strict"], s["wrong"]) for arm, s in output["test"]["arms"].items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"run")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=12)
    run(parser.parse_args())
