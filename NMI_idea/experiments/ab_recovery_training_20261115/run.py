"""Collect causal scan data, fit recovery skills and compare paired controllers."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from controller import (
    CENTER_MM, EDGES, FRAME_COUNT, FRAME_STRIDE, HERE, IDENTITY_ACCEPT, IDENTITY_STATION, KINDS, LEGACY,
    NAMES, Controller, fork, skill_features, station_bin,
)
from world import World


SPLIT_BASE = {"motion": 700000, "branch": 800000, "dev": 900000, "test": 1600000, "alias": 1800000}
WORLD_SPLIT = {"motion": "train", "branch": "train", "dev": "dev", "test": "heldout", "alias": "stress"}
ARMS = ("mpc", "fixed", "learned")
ELIGIBLE_METRICS = ("probe_local", "fresh_local", "identity_correct", "identity_correct_before",
                    "var_drop_mm2", "sq_position_error", "interval_covered", "ambiguity_correct",
                    "identity_certified", "terminal_identity_correct")
OUTCOME_METRICS = ("success", "wrong", "wrong_certification", "wrong_confident_frames",
                   "false_complete", "coverage", "arrived",
                   "hit_rate", "mean_abs_lateral_mm", "tcp_mm", "turn_deg", "seconds", "frames")
MARGINS = {"tcp_mm": 2., "turn_deg": 2., "seconds": .1, "frames": 1.}
_case_job = None


def install_case_job(job):
    global _case_job
    _case_job = job


def execute_case_job(index):
    return _case_job(index)


def case_records(job, count, jobs):
    if jobs == 1:
        yield from map(job, range(count))
        return
    # The Linux study runner shares the read-only controller/model with workers.
    context = get_context("fork")
    with context.Pool(min(count, jobs), initializer=install_case_job, initargs=(job,)) as pool:
        yield from pool.imap(execute_case_job, range(count), chunksize=1)


def load_sections():
    return json.loads((LEGACY / "SECTIONS.json").read_text(encoding="utf-8"))


def json_record(stream, row):
    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")


def archive(record, *, prefix_frames=0, prefix_samples=0):
    return {
        "packet_offset": prefix_frames,
        "motion_offset": prefix_samples,
        "packets": record.packets[prefix_frames:],
        "motion_t_s_x_vs_vx_q": record.motion[prefix_samples:],
        "decisions": record.decisions,
        "accepts": record.accepts,
        "blocks": record.blocks,
    }


def context(ctrl, scene, *, perturb=None, alias=False):
    state, record = ctrl.initial(scene, alias)
    return ctrl.finish(scene, state, record, alias, allow_recover=False, stop_at_trigger=True, rng=perturb)


def checkpoint(state):
    public = dict(vars(state))
    public["belief"] = state.belief.tolist()
    return public


def clean_target_packet(row):
    slot = row["accepted_slot"]
    return (slot >= 0 and row["roles_truth"][slot] == 0
            and abs(row["y"][slot][0]) <= CENTER_MM
            and abs(row["target_truth"][0] - row["x"]) <= CENTER_MM)


def shape_examples(ctrl, record, blocks, x_data, w_data, used):
    for name, lo, hi in blocks:
        frames = record.packets[lo // FRAME_STRIDE + 1:(hi - 1) // FRAME_STRIDE + 1]
        if not any(clean_target_packet(row) for row in frames):
            continue
        if any(p["accepted_slot"] >= 0 and p["roles_truth"][p["accepted_slot"]] != 0 for p in frames):
            continue
        trajectory = np.array(record.motion[lo:hi])
        x, v = trajectory[:, 2], trajectory[:, 4]
        kind = KINDS[name]
        x_data[kind].append(skill_features(x[0], v[0], x[-1]))
        w_data[kind].append(ctrl.primitive.fit(x, v))
        used[name] += 1


def case(world, split, index, mode="fixed"):
    seed = SPLIT_BASE[split] + index
    scene = world.scene(seed, WORLD_SPLIT[split], FRAME_COUNT, False, alias=split == "alias")
    if mode != "fixed":
        # The fixed and moving arms share anatomy and every exogenous disturbance.
        t = np.linspace(0., 1., FRAME_COUNT)
        direction = 1. if mode == "flexion" else -1.
        scene = world.replace_angles(scene, 15. + direction * 6. * np.sin(np.pi * t))
    return scene


def fit_motion(world, sections, n, out):
    ctrl = Controller(world, sections)
    x_data, w_data = defaultdict(list), defaultdict(list)
    used, generated = Counter(), Counter()
    with gzip.open(out / "motion_demonstrations.jsonl.gz", "wt", encoding="utf-8", compresslevel=2) as stream:
        for index in range(n):
            scene = case(world, "motion", index)
            rng = np.random.default_rng([SPLIT_BASE["motion"], index, 9921])
            before, prefix = context(ctrl, scene, perturb=rng)
            full_s, full_r = fork(before, prefix)
            ctrl.finish(scene, full_s, full_r, False, allow_recover=False, rng=rng)
            generated.update(block[0] for block in full_r.blocks)
            shape_examples(ctrl, full_r, full_r.blocks, x_data, w_data, used)
            branches = {}
            if before.trigger():
                for name, candidate in ctrl.options(before, False).items():
                    if name == "advance":
                        continue
                    state, record = fork(before, prefix)
                    ctrl.execute(scene, state, record, candidate, False)
                    generated[name] += 1
                    n_before = used[name]
                    shape_examples(ctrl, record, record.blocks[-1:], x_data, w_data, used)
                    branches[name] = {
                        "used_for_fit": used[name] != n_before,
                        **archive(record, prefix_frames=len(prefix.packets), prefix_samples=len(prefix.motion) - 1),
                    }
            json_record(stream, {"seed": SPLIT_BASE["motion"] + index, "family": scene.family,
                                 "trigger": before.trigger(), "checkpoint": checkpoint(before),
                                 "teacher": archive(full_r), "branches": branches})
    coef, diagnostic = {}, {}
    for name, rows in x_data.items():
        x, y = np.array(rows), np.array(w_data[name])
        reg = len(x) * .008 * np.eye(x.shape[-1])
        coef[name] = np.linalg.solve(x.T @ x + reg, x.T @ y)
        project = (coef[name] @ ctrl.primitive.null) @ ctrl.primitive.null.T
        diagnostic[name] = {"segments": len(x), "shape_norm": float(np.linalg.norm(coef[name])),
                            "fit_shape_rmse": float(np.sqrt(np.mean((x @ coef[name] - y) ** 2))),
                            "nullspace_error": float(np.max(abs(coef[name] - project)))}
    np.savez_compressed(out / "prodmp_weights.npz", **coef)
    return coef, {"episodes": n, "generated_segments": dict(generated), "used_segments": dict(used),
                  "conditioned_fits": diagnostic}


def measures(ctrl, record, state):
    values = record.details(state)
    values["success"] = values.pop("complete")
    values["false_complete"] = bool(values["arrived"] and not values["success"]
                                    and not state.uncovered())
    values["hit_rate"] = sum(clean_target_packet(p) for p in record.packets) / len(record.packets)
    values["mean_abs_lateral_mm"] = float(np.mean([abs(p["target_truth"][0] - p["x"])
                                                   for p in record.packets]))
    trace = np.array(record.motion)
    values["tcp_mm"], values["turn_deg"] = ctrl.geometry.cost(trace[:, 1], trace[:, 2])
    values["tcp_cost_error"] = abs(values["tcp_mm"] - state.travel)
    values["longitudinal_mm"] = float(abs(np.diff(trace[:, 1])).sum())
    values["max_x_speed"] = float(np.max(abs(trace[:, 4])))
    values["max_s_speed"] = float(np.max(abs(trace[:, 3])))
    return values


def fresh_evidence(record, bin_id, from_frame, to_frame):
    return any(clean_target_packet(row) for row in record.packets
               if from_frame < row["frame"] <= to_frame and station_bin(row["s"]) == bin_id)


def acquisition(ctrl, scene, before, after, record, alias):
    q = before.query_bin()
    query_s = float(np.mean(EDGES[q:q + 2]))
    u0 = ctrl.uncertainty(before, q, alias, intervals=True)
    u1 = ctrl.uncertainty(after, q, alias, angle=before.angle, intervals=True)
    truth = ctrl.world.observe(scene, before.frame, query_s, before.x)[2]
    identity = ctrl.world.true_identity(scene, before.frame, IDENTITY_STATION)
    role0 = ctrl.world.identity_prob(before.belief, IDENTITY_STATION, before.angle, alias=alias)
    role1 = ctrl.world.identity_prob(after.belief, IDENTITY_STATION, before.angle, alias=alias)
    correct0 = (role0 >= .5) == identity
    correct1 = (role1 >= .5) == identity
    low, high = np.array(u1["interval90"]).T
    return {
        "query_bin": q, "query_station": query_s, "query_angle": before.angle,
        "target_truth": truth.tolist(), "identity_truth": bool(identity), "pre": u0, "post": u1,
        "identity_station": IDENTITY_STATION, "identity_pre": role0, "identity_post": role1,
        "probe_local": fresh_evidence(record, q, before.frame, after.frame),
        "identity_correct": bool(correct1), "identity_correct_before": bool(correct0),
        "identity_certified": bool(max(role1, 1. - role1) >= IDENTITY_ACCEPT),
        "ambiguous_before": bool(.1 < role0 < .9),
        "ambiguity_correct": bool(correct1 and max(role1, 1. - role1) >= IDENTITY_ACCEPT),
        "var_drop_mm2": float(sum(u0["var"]) - sum(u1["var"])),
        "sq_position_error": float(np.sum((np.array(u1["mean"]) - truth) ** 2)),
        "sq_error_before": float(np.sum((np.array(u0["mean"]) - truth) ** 2)),
        "interval_covered": float(np.mean((low <= truth) & (truth <= high))),
    }


def run_action(ctrl, scene, before, prefix, action, *, alias):
    state, record = fork(before, prefix)
    ctrl.execute(scene, state, record, action, alias)
    evidence = acquisition(ctrl, scene, before, state, record, alias)
    ctrl.finish(scene, state, record, alias, allow_recover=True)
    result = measures(ctrl, record, state) | evidence
    result["fresh_local"] = fresh_evidence(record, evidence["query_bin"], before.frame, state.frame)
    probability = ctrl.world.identity_prob(state.belief, IDENTITY_STATION, before.angle, alias=alias)
    result["terminal_identity_correct"] = bool((probability >= .5) == evidence["identity_truth"]
                                               and max(probability, 1. - probability) >= IDENTITY_ACCEPT)
    return result, record


def fit_value(rows):
    x = np.array([row["features"] for row in rows])
    outcomes = np.array([[row["success"], row["wrong"], row["fresh_local"]] for row in rows], dtype=float)
    models = []
    for column in outcomes.T:
        model = HistGradientBoostingRegressor(max_iter=65, max_leaf_nodes=9, learning_rate=.07,
                                              min_samples_leaf=18, l2_regularization=4., random_state=902)
        model.fit(x, column)
        models.append(model)
    return models


def predictions(ctrl, state, choices, alias, models):
    x = np.array([ctrl.policy_features(state, c, alias) for c in choices.values()])
    pred = np.column_stack([np.clip(model.predict(x), 0., 1.) for model in models])
    return {name: row for name, row in zip(choices, pred, strict=True)}


def value_score(result, distance):
    success, wrong, local = result
    return float(success - wrong + .13 * local - .009 * distance)


def collect_branches(ctrl, n, out, split, jobs):
    def run_case(index):
        scene = case(ctrl.world, split, index)
        state, prefix = context(ctrl, scene)
        menu = ctrl.options(state, False) if state.trigger() else {}
        row = {"seed": SPLIT_BASE[split] + index, "family": scene.family,
               "trigger": bool(menu), "checkpoint": checkpoint(state), "prefix": archive(prefix)}
        if not menu:
            return row
        mpc = ctrl.mpc(state, menu, False).name
        candidates = {}
        for name, option in menu.items():
            result, record = run_action(ctrl, scene, state, prefix, option, alias=False)
            candidates[name] = dict(result, action=name, distance_first=option.distance,
                                    features=ctrl.policy_features(state, option, False).tolist(),
                                    suffix=archive(record, prefix_frames=len(prefix.packets),
                                                   prefix_samples=len(prefix.motion) - 1))
        row |= {"baseline_action": mpc, "candidates": candidates}
        if split == "dev":
            favorable = max(candidates, key=lambda name: (
                candidates[name]["success"], not candidates[name]["wrong"],
                candidates[name]["fresh_local"], -candidates[name]["tcp_mm"], name == mpc))
            replicate = ctrl.world.resample_tail(scene, state.frame, 271)
            witness, _ = run_action(ctrl, replicate, state, prefix, menu[favorable], alias=False)
            repeated_mpc = witness
            if favorable != mpc:
                repeated_mpc, _ = run_action(ctrl, replicate, state, prefix, menu[mpc], alias=False)
            row["replicate_witness"] = {
                "first_tape_best_action": favorable, "baseline_action": mpc,
                "first_tape": {name: {key: candidates[name][key] for key in OUTCOME_METRICS}
                               for name in (favorable, mpc)},
                "replicate": {favorable: witness, mpc: repeated_mpc},
            }
        return row

    training, outcomes, replica_rows = [], [], []
    file = out / f"{split}_branches.jsonl.gz"
    with gzip.open(file, "wt", encoding="utf-8", compresslevel=2) as stream:
        for row in case_records(run_case, n, jobs):
            json_record(stream, row)
            if row["trigger"]:
                scores = {name: {key: value for key, value in candidate.items() if key != "suffix"}
                          for name, candidate in row["candidates"].items()}
                training.extend(scores.values())
                outcomes.append((row["baseline_action"], scores))
            if "replicate_witness" in row:
                replica_rows.append(row["replicate_witness"])
    return training, outcomes, replica_rows


def freeze_fixed(rows):
    value = {}
    for choice in NAMES:
        runs = [menu.get(choice, menu[mpc]) for mpc, menu in rows]
        value[choice] = float(np.mean([value_score([r["success"], r["wrong"], r["fresh_local"]],
                                                   r["distance_first"]) for r in runs]))
    return max(value, key=value.get), value


def brier(models, rows):
    x = np.array([row["features"] for row in rows])
    y = np.array([[row["success"], row["wrong"], row["fresh_local"]] for row in rows], dtype=float)
    forecasts = np.column_stack([np.clip(model.predict(x), 0., 1.) for model in models])
    mean_brier = ((forecasts - y) ** 2).mean(axis=0).tolist()
    return dict(zip(("success", "wrong", "local"), mean_brier, strict=True))


def stage_a(ctrl, scene, *, alias=False):
    state, record = ctrl.initial(scene, alias)
    ctrl.finish(scene, state, record, alias, allow_recover=False)
    return measures(ctrl, record, state), record


def assess(ctrl, old, models, fixed, n, out, *, mode, split, jobs):
    alias = split == "alias"

    def run_case(index):
        scene = case(ctrl.world, split, index, mode)
        state, prefix = context(ctrl, scene, alias=alias)
        decisions, arms, arms_trace = {}, {}, {}
        a0, a0_trace = stage_a(old, scene, alias=alias)
        a1_state, a1_trace = fork(state, prefix)
        ctrl.finish(scene, a1_state, a1_trace, alias, allow_recover=False)
        a1 = measures(ctrl, a1_trace, a1_state)
        choices = ctrl.options(state, alias) if state.trigger() else {}
        if choices:
            mpc = ctrl.mpc(state, choices, alias).name
            values = predictions(ctrl, state, choices, alias, models)
            learned = max(choices, key=lambda name: value_score(values[name], choices[name].distance))
            decisions = dict(mpc=mpc, fixed=fixed if fixed in choices else mpc, learned=learned)
            memo = {}
            for arm, name in decisions.items():
                if name not in memo:
                    memo[name] = run_action(ctrl, scene, state, prefix, choices[name], alias=alias)
                arms[arm], trace = memo[name]
                arms_trace[arm] = archive(trace, prefix_frames=len(prefix.packets),
                                          prefix_samples=len(prefix.motion) - 1)
            logged_predictions = {key: value.tolist() for key, value in values.items()}
        else:
            arms = {arm: a1 for arm in ARMS}
            arms_trace = {arm: {} for arm in ARMS}
            logged_predictions = {}
        row = {"seed": SPLIT_BASE[split] + index, "family": scene.family, "angle_mode": mode,
               "trigger": bool(choices), "actions": decisions, "B_predictions": logged_predictions,
               "a_old": a0, "a_new": a1, **arms}
        return row, dict(row, prefix=archive(prefix), suffixes=arms_trace, checkpoint=checkpoint(state),
                         a_old_trace=archive(a0_trace), a_new_trace=archive(a1_trace))

    all_rows = []
    output = out / f"{split}_{mode}.jsonl.gz"
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=2) as stream:
        for row, detail in case_records(run_case, n, jobs):
            all_rows.append(row)
            json_record(stream, detail)
    return all_rows


def means(rows, arm, metrics):
    return {key: float(np.mean([row[arm][key] for row in rows])) for key in metrics}


def contrast(rows, arm, base, metrics):
    bootstrap = np.random.default_rng(84122).integers(len(rows), size=(4000, len(rows)))
    compare = {}
    for key in metrics:
        diff = np.array([float(row[arm][key]) - float(row[base][key]) for row in rows])
        pair = dict(delta=float(diff.mean()),
                    ci95=np.quantile(diff[bootstrap].mean(axis=1), [.025, .975]).tolist())
        if key in ("success", "wrong", "fresh_local", "identity_correct"):
            pair |= {"wins": int(sum(diff > 0)), "losses": int(sum(diff < 0))}
        if key in MARGINS:
            pair["within_margin"] = (pair["ci95"][0] >= -MARGINS[key] and pair["ci95"][1] <= MARGINS[key])
        compare[key] = pair
    return compare


def summarize(rows):
    trigger = [row for row in rows if row["trigger"]]
    summary = {
        "n": len(rows), "triggers": len(trigger),
        "arms": {arm: means(rows, arm, OUTCOME_METRICS) for arm in ("a_old", "a_new", *ARMS)},
        "paired_a": contrast(rows, "a_new", "a_old", OUTCOME_METRICS),
        "paired_b": {a: contrast(rows, "learned", a, OUTCOME_METRICS) for a in ("mpc", "fixed")},
        "max_handoff_error": max(row[arm]["max_handoff_error"] for row in rows for arm in ("a_old", "a_new", *ARMS)),
        "max_tcp_cost_error": max(row[arm]["tcp_cost_error"] for row in rows for arm in ("a_old", "a_new", *ARMS)),
    }
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    summary["family"] = {
        family: dict(n=len(group), **{arm: means(group, arm, ("success", "wrong", "tcp_mm", "seconds"))
                                    for arm in ("a_old", "a_new", *ARMS)})
        for family, group in sorted(groups.items())
    }
    if trigger:
        ambiguous = [row for row in trigger if row["mpc"]["ambiguous_before"]]
        summary["trigger_metrics"] = {
            "ambiguous_before": len(ambiguous),
            "arms": {arm: means(trigger, arm, (*OUTCOME_METRICS, *ELIGIBLE_METRICS, "sq_error_before"))
                     for arm in ARMS},
            "learned_vs_mpc": contrast(trigger, "learned", "mpc", (*OUTCOME_METRICS, *ELIGIBLE_METRICS)),
            "actions": {arm: dict(Counter(row["actions"][arm] for row in trigger)) for arm in ARMS},
            "selection_diff": sum(row["actions"]["mpc"] != row["actions"]["learned"] for row in trigger),
        }
        if ambiguous:
            summary["trigger_metrics"]["ambiguous_arms"] = {
                arm: means(ambiguous, arm, ("ambiguity_correct", "identity_correct", "wrong",
                                            "fresh_local", "success", "var_drop_mm2", "terminal_identity_correct"))
                for arm in ARMS
            }
    return summary


def write_report(out, data):
    temp = out / ".summary.incomplete"
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(out / "summary.json")


def replica_summary(rows):
    comparisons = {}
    for tape in ("first_tape", "replicate"):
        paired = [{"first_tape_choice": row[tape][row["first_tape_best_action"]],
                   "mpc": row[tape][row["baseline_action"]]} for row in rows]
        comparisons[tape] = contrast(paired, "first_tape_choice", "mpc",
                                     ("success", "wrong", "tcp_mm", "turn_deg", "seconds", "frames"))
    return {"contexts": len(rows), "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion-count", type=int, default=128)
    parser.add_argument("--branch-count", type=int, default=192)
    parser.add_argument("--dev-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=96)
    parser.add_argument("--alias-count", type=int, default=24)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--preflight", action="store_true", help="use disjoint integration-only seed namespaces")
    parser.add_argument("--fixed-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.motion_count, args.branch_count, args.dev_count, args.test_count, args.jobs) < 1:
        parser.error("positive training, development, holdout and worker counts required")
    if max(args.motion_count, args.branch_count, args.dev_count, args.test_count, args.alias_count) >= 100000:
        parser.error("counts must fit in the frozen disjoint seed namespaces")
    if args.preflight:
        SPLIT_BASE.update({split: seed + 3000000 for split, seed in SPLIT_BASE.items()})
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(f"use a fresh run directory: {out}")
    started = perf_counter()
    sections, world = load_sections(), World()
    with np.load(LEGACY / "verification" / "DEMONSTRATIONS.npz") as archive:
        legacy = archive["shape_regression"]
    old = Controller(world, sections, {"advance": legacy * np.array([[1.], [5.], [6.]])})
    motion, shape_report = fit_motion(world, sections, args.motion_count, out)
    ctrl = Controller(world, sections, motion)
    print(f"Motion fit: {shape_report}", flush=True)
    rows, train_actions, _ = collect_branches(ctrl, args.branch_count, out, "branch", args.jobs)
    validation, dev_actions, replicas = collect_branches(ctrl, args.dev_count, out, "dev", args.jobs)
    if min(len(train_actions), len(dev_actions)) < 2:
        raise RuntimeError("insufficient independent recovery contexts")
    models = fit_value(rows)
    fixed, fixed_values = freeze_fixed(dev_actions)
    with open(out / "value_model.pkl", "wb") as stream:
        pickle.dump(models, stream, protocol=4)
    sources = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
               for file in (HERE / "controller.py", HERE / "run.py", HERE / "world.py",
                            LEGACY / "SECTIONS.json", LEGACY / "primitive.py",
                            LEGACY / "verification" / "DEMONSTRATIONS.npz")}
    data = {
        "scope": "Synthetic anonymous centroid packets on registered Among_US section geometry, ideal material-chart motion and reference-pose Cartesian costs.",
        "seed_bases": SPLIT_BASE, "source_sha256": sources, "motion": shape_report,
        "branch_train": {
            "scenes": args.branch_count, "eligible": len(train_actions), "pairs": len(rows),
            "actions": dict(Counter(row["action"] for row in rows)),
            "success_by_action": {a: sum(row["action"] == a and row["success"] for row in rows)
                                  for a in NAMES},
        },
        "dev": {"scenes": args.dev_count, "eligible": len(dev_actions), "pairs": len(validation),
                "fixed": fixed, "scores": fixed_values, "brier": brier(models, validation),
                "headroom_diagnostic": replica_summary(replicas)},
        "cost_margins": MARGINS, "holdout": {},
    }
    write_report(out, data)
    print(f"B fit: {data['branch_train']}; development fixed action: {fixed}", flush=True)
    modes = ("fixed",) if args.fixed_only else ("fixed", "flexion", "extension")
    for mode in modes:
        outcomes = assess(ctrl, old, models, fixed, args.test_count, out, mode=mode, split="test",
                          jobs=args.jobs)
        data["holdout"][mode] = summarize(outcomes)
        write_report(out, data)
        print(mode, {arm: int(round(v["success"] * len(outcomes))) for arm, v in data["holdout"][mode]["arms"].items()},
              f"n={len(outcomes)}, eligible={data['holdout'][mode]['triggers']}", flush=True)
    if args.alias_count:
        outcomes = assess(ctrl, old, models, fixed, args.alias_count, out, mode="fixed", split="alias",
                          jobs=args.jobs)
        data["holdout"]["alias"] = summarize(outcomes)
    data["runtime_s"] = perf_counter() - started
    write_report(out, data)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
