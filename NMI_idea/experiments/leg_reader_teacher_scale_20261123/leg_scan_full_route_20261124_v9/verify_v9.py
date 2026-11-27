"""Recount the v9 scans and replay saved policy decisions from committed shards."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from policy_v9 import Actor
from study_v9 import DEFAULT_PACK, ROOT, ROUTES, assert_hashes, atomic_json, digest, read_json, roster, verify_shard


LANES = ("ppo_only", "teacher")
TAPES = ("actual", "proof_0", "proof_1")
FIELDS = ("strict", "wrong_branch", "capture", "selected_ids", "mm", "longest_mm", "time_s", "travel_mm",
          "actions", "steps", "completed_legal")
CONTRASTS = (("MPC_new", "MPC_old"), ("student_old", "MPC_old"), ("student_new", "MPC_new"),
             ("student_new", "student_old"), ("PPO_only_old", "MPC_old"), ("PPO_only_new", "MPC_new"),
             ("student_new", "PPO_only_new"), ("student_old", "PPO_only_old"), ("student_new", "MPC_old"))


def require(condition, failure):
    if not condition:
        raise ValueError(failure)


def cohort(path, planned):
    shards = [verify_shard(lock.parent) for lock in sorted(path.glob("*/commit.json"))]
    rows = [row for _, records in shards for row in records]
    require([r["case"] for r in rows] == [asdict(c) for c in planned], f"Roster mismatch in {path}")
    require(all(r["complete"] and r["case"]["source"] in (0, 1, 2) for r in rows), f"Incomplete or non-atlas scan in {path}")
    require(len({r["layout_id"] for r in rows}) == len(rows), f"Duplicated layout in {path}")
    return shards, rows


def check_trace(trace, metrics, actions):
    require(metrics["completed_legal"] and trace[0]["tick"] == 1 and trace[-1]["action"] == "finish",
            "Missing legal terminal action")
    require(trace[-1]["next_tick"] == metrics["terminal_clock"] and metrics["time_s"] > 0, "Terminal clock mismatch")
    require(Counter(t["action"] for t in trace) == actions, "Executed actions disagree with trace")
    for t, n in zip(trace, trace[1:]):
        require((t["next_station"], t["next_sector"], t["next_tick"]) == (n["station"], n["sector"], n["tick"]),
                "Trace is discontinuous")
    for t in trace:
        require(t["next_tick"] - t["tick"] == t["duration"] and t["duration"] >= 0 and t["distance_mm"] >= -1e-9,
                "Unphysical time or distance")
        require((t["duration"] == 0) == (t["action"] == "finish"), "Invalid zero-duration action")
        if t["action"].startswith("revisit"):
            require(t["probe"] and t["probe"]["ticks"] > 0, "Revisit skipped physical execution")


def check_teacher(row, actions, obs, mask, hidden):
    teacher = row["teacher"]
    require(teacher["layout"] == row["layout_id"], "Changed anatomy in teacher label")
    trace = row["trace"][teacher["step"]]
    require((trace["station"], trace["tick"]) == (teacher["station"], teacher["tick"]), "Teacher used a different prefix")
    np.testing.assert_allclose(teacher["observations"], obs, atol=1e-6)
    np.testing.assert_allclose(teacher["history"], hidden, atol=1e-6)
    np.testing.assert_array_equal(teacher["mask"], mask)
    offered = [actions[i] for i in teacher["actions"]]
    require(all(mask[i] for i in teacher["actions"]) and 2 <= len(offered) <= 3 and len(teacher["bands"]) == 2,
            "Teacher mask or tape budget violated")
    ref = actions[teacher["reference"]]
    stats = Counter(paired_visits=1)
    for j, band in enumerate(teacher["bands"]):
        require(set(band) == set(offered), "Missing a same-tape action")
        for i, key in zip(teacher["actions"], offered, strict=True):
            d = band[key]["value"] - band[ref]["value"]
            np.testing.assert_allclose(d, teacher["label_differences"][str(i)][j], atol=1e-12)
            stats["executed_branches"] += 1
            if key == ref:
                continue
            stats["strict_rescues"] += int(band[key]["strict"] and not band[ref]["strict"])
            stats["strict_losses"] += int(band[ref]["strict"] and not band[key]["strict"])
            stats["wrong_added"] += int(band[key]["wrong_branch"] and not band[ref]["wrong_branch"])
            stats["wrong_removed"] += int(band[ref]["wrong_branch"] and not band[key]["wrong_branch"])
            stats["positive" if d > 1e-7 else "negative" if d < -1e-7 else "zero"] += 1
    delta = np.array([teacher["label_differences"][str(i)] for i in teacher["actions"]])
    mean = delta.mean(axis=1)
    stats["selected_and_scored_same_tapes_sum"] += float(mean.max())
    stats["selected_on_one_scored_on_other_sum"] += float(
        (delta[delta[:, 0].argmax(), 1] + delta[delta[:, 1].argmax(), 0]) / 2)
    return stats


def replay(shards, bundle, action_names):
    actor = Actor(bundle)
    details = Counter()
    errors = {"logp": 0., "value": 0.}
    for path, rows in shards:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        offsets = arrays["offsets"]
        starts = zip(offsets[:-1], offsets[1:], rows, strict=True)
        for start, stop, row in starts:
            trace = row["trace"]
            check_trace(trace, row["metrics"], row["actions"])
            require(len(trace) == stop - start == row["steps"], "Rollout offsets differ from the trace")
            np.testing.assert_allclose(arrays["reward"][start:stop].sum(), row["total_reward"], atol=1e-8)
            hidden = actor.zero()
            for index, actual in zip(range(start, stop), trace, strict=True):
                action = arrays["action"][index]
                mask, obs = arrays["mask"][index], arrays["obs"][index]
                require(mask[action] and action_names[action] == actual["action"] and
                        arrays["duration"][index] == actual["duration"], "Replay differs from physical action")
                prob, hidden = actor.step(obs, hidden, mask)
                old_value = actor.value(hidden, arrays["priv"][index])
                errors["logp"] = max(errors["logp"], abs(np.log(max(1.e-15, prob[action])) - arrays["logp"][index]))
                errors["value"] = max(errors["value"], abs(old_value - arrays["value"][index]))
                if row["teacher"] and row["teacher"]["step"] == index - start:
                    details.update(check_teacher(row, action_names, obs, mask, hidden))
    require(max(errors.values()) < 1e-5, "Stored behaviour policy differs from its recurrent replay")
    return details, errors


def manifest(path, protocol_hash):
    row = read_json(path / "manifest.json")
    require(row["protocol_sha256"] == protocol_hash and row["actor_sha256"] == digest(path / "actor.json") and
            row["checkpoint_sha256"] == digest(path / "checkpoint.pt"), f"Checkpoint changed in {path}")
    return row


def training(out, protocol, protocol_hash):
    smoke_n, bc_n, batch_n = (protocol[k] for k in ("common_smoke", "common_BC", "cases_per_round"))
    _, smoke = cohort(out / "smoke", roster("smoke", smoke_n))
    _, demos = cohort(out / "bc_episodes", roster("bc", bc_n))
    bc = manifest(out / "bc_checkpoint", protocol_hash)
    require(bc["counts"]["complete"] == len(demos) and bc["bc"]["cuda_optimizer_step_verified"], "BC checkpoint invalid")
    machine = read_json(out / "full_stack_smoke.json")
    require(machine["protocol_sha256"] == protocol_hash and machine["metrics"]["complete"] == len(smoke) and
            machine["cuda"]["cuda_optimizer_step_verified"], "Full-stack check failed")
    counts = {lane: Counter() for lane in LANES}
    per_route = {lane: {} for lane in LANES}
    previous = dict.fromkeys(LANES, out / "bc_checkpoint" / "actor.json")
    teacher_counts, teacher_fit = Counter(), []
    behaviour_error = {"logp": 0., "value": 0.}
    train_ids = set()
    freeze_actors = read_json(out / "independent_freeze.json")["actors"]
    for r in range(protocol["rounds_per_branch"]):
        expected = list(roster("train", batch_n, r))
        ids = []
        for lane in LANES:
            path = out / "training" / lane / f"{r:03d}"
            m = manifest(path, protocol_hash)
            update = m["updates"]
            require(m["batch"] == r and m["lane"] == lane and m["behavior_actor_sha256"] == digest(previous[lane]) and
                    m["reader_sha256"] == protocol["reader_sha256"] and update["device"].startswith("cuda") and
                    update["actor_delta_l2"] > 0 and update["cuda_optimizer_step_verified"], "Broken on-policy checkpoint chain")
            shards, rows = cohort(path / "episodes", expected)
            require(m["episode_shards_sha256"] == [digest(p.parent / "commit.json") for p, _ in shards],
                    "Training checkpoint points to different episode shards")
            info, errors = replay(shards, read_json(previous[lane]), protocol["actions"])
            behaviour_error = {k: max(errors[k], behaviour_error[k]) for k in behaviour_error}
            require(m["counts"]["complete"] == len(rows) and m["counts"]["macros"] == sum(r["steps"] for r in rows),
                    "Checkpoint counts differ from completed episodes")
            previous[lane] = path / "actor.json"
            if lane == "teacher":
                fit = m["teacher"]["fit"]
                teacher_dir = path / "teacher_fit"
                require(m["teacher"]["paired_visits"] == info["paired_visits"] and
                        fit["device"].startswith("cuda") and fit["optimizer_steps"] > 0 and fit["parameters_updated"] and
                        fit["checkpoint_sha256"] == digest(teacher_dir / "teacher_checkpoint.pt") and
                        fit["bundle_sha256"] == digest(teacher_dir / "teacher_bundle.json"), "Teacher CUDA update missing")
                teacher_counts.update(info)
                teacher_fit.append({"batch": r, "cuda_updates": fit["optimizer_steps"], "validation": fit["validation"]})
            else:
                require(m["teacher"] is None and not info, "PPO-only lane used a teacher")
            ids.append([(row["layout_id"], row["reader"]) for row in rows])
            counts[lane].update({k: m["counts"][k] for k in ("complete", "macros")})
            counts[lane]["cuda_ppo_steps"] += update["optimizer_steps"]
            for row in rows:
                route = row["case"]["task"]
                counted = per_route[lane].setdefault(route, Counter())
                counted.update(completed=1, macros=row["steps"])
        require(ids[0] == ids[1], "Train arms did not share layouts and reader assignment")
        batch_ids = {item[0] for item in ids[0]}
        require(not train_ids & batch_ids, "A train layout was collected in two updates")
        train_ids.update(batch_ids)
    for lane, info in freeze_actors.items():
        path = out / f"deploy_{lane}.json"
        used = out / "training" / lane / info["checkpoint"] / "actor.json"
        source_actor = read_json(used)
        del source_actor["critic"]
        require(digest(path) == info["actor_only_sha256"] and digest(used) == info["source_actor_sha256"] and
                read_json(path) == source_actor, "Independent deployment contains unexpected weights or a training critic")
    partitions = ({"smoke": {r["layout_id"] for r in smoke}, "demonstration": {r["layout_id"] for r in demos},
                   "training": train_ids})
    paired_visits = teacher_counts["paired_visits"]
    selection = {"same_tapes_average_margin": teacher_counts.pop("selected_and_scored_same_tapes_sum") / paired_visits,
                 "cross_tape_average_margin": teacher_counts.pop("selected_on_one_scored_on_other_sum") / paired_visits}
    fits = [row["validation"] for row in teacher_fit]
    n = sum(row["measured"] for row in fits)
    sign_n = sum(row["positive"] + row["negative"] for row in fits)
    fit_validation = {"labelled_actions": n, "nonzero_labelled_actions": sign_n,
                      "mse": sum(row["mse"] * row["measured"] for row in fits) / n,
                      "zero_mse": sum(row["zero_mse"] * row["measured"] for row in fits) / n,
                      "sign_accuracy": sum(row["signed_accuracy"] * (row["positive"] + row["negative"])
                                           for row in fits if row["positive"] + row["negative"]) / sign_n}
    return partitions, {"counts": counts, "route_counts": per_route, "behaviour_replay_max_error": behaviour_error,
                        "teacher_branches": teacher_counts, "teacher_branch_selection": selection,
                        "teacher_validation": fit_validation, "teacher_cuda_rounds": teacher_fit}


def ci(delta):
    samples = np.random.default_rng(180_411_381).integers(0, len(delta), (3500, len(delta)))
    bounds = np.percentile(delta[samples].mean(axis=1), [2.5, 97.5])
    return [float(delta.mean()), *bounds.tolist()]


def paired(rows, arm, other, key):
    d = np.array([[float(r["outcomes"][t][arm][key]) - r["outcomes"][t][other][key] for t in TAPES] for r in rows])
    return d, {"actual": ci(d[:, 0]), "proof_layout_ci95": ci(d[:, 1:].mean(axis=1)),
               "added": int(np.count_nonzero(d[:, 0] > 0)), "lost": int(np.count_nonzero(d[:, 0] < 0))}


def result_panel(rows, frozen):
    panel = {}
    for route in ROUTES:
        subset = [row for row in rows if row["case"]["task"] == route]
        arms = {}
        for arm in subset[0]["outcomes"]["actual"]:
            cases = [[row["outcomes"][t][arm] for row in subset] for t in TAPES]
            x = cases[0]
            capture = np.array([item["capture"] for item in x])
            selected = np.array([item["selected_ids"] for item in x])
            failure = (selected != 0).sum(axis=1)
            arms[arm] = {"layouts": len(x), "strict": [sum(item["strict"] for item in s) for s in cases],
                         "wrong_branch": [sum(item["wrong_branch"] for item in s) for s in cases],
                         "acquisition_ceiling": [sum(all(item["capture"]) for item in s) for s in cases],
                         "wrong_delivery": sum(item["wrong_delivery"] for item in x),
                         "one_failed_region": int(np.count_nonzero(failure == 1)),
                         "multi_failed_region": int(np.count_nonzero(failure > 1)),
                         "all_captured_failed": int(np.count_nonzero(capture.all(axis=1) & (failure > 0))),
                         "artifacts_by_region": ((selected < 0) & (selected != -99)).sum(axis=0).tolist(),
                         "rejected_with_capture_by_region": ((selected == -99) & capture).sum(axis=0).tolist(),
                         "correct_by_region": (selected == 0).sum(axis=0).tolist(),
                         "named_branch_by_region": (selected > 0).sum(axis=0).tolist()}
            require(arms[arm]["strict"][0] == frozen[route][arm]["strict"] and
                    arms[arm]["wrong_branch"][0] == frozen[route][arm]["wrong_branch"], "Published actual counts disagree")
        contrasts = {}
        for arm, other in CONTRASTS:
            contrasts[f"{arm}_vs_{other}"] = {}
            for key in ("strict", "wrong_branch", "time_s", "travel_mm", "mm", "longest_mm"):
                _, result = paired(subset, arm, other, key)
                contrasts[f"{arm}_vs_{other}"][key] = result
                expected = frozen[route][arm]["paired"].get(other, {}).get(key)
                if expected is not None:
                    for interval in ("actual", "proof_layout_ci95"):
                        np.testing.assert_allclose(result[interval], expected[interval], atol=1e-12)
                    require(result["added"] == expected["added"] and result["lost"] == expected["lost"],
                            "Paired rescue or loss count differs")
        newer, _ = paired(subset, "student_new", "MPC_new", "strict")
        older, _ = paired(subset, "student_old", "MPC_old", "strict")
        diff = newer - older
        panel[route] = {"arms": arms, "contrasts": contrasts,
                        "strict_factorial_interaction": {"actual": ci(diff[:, 0]),
                                                         "proof_layout_ci95": ci(diff[:, 1:].mean(axis=1))}}
    return panel


def supplementary_bc(out, cases, protocol, protocol_hash):
    folder = out / "bc_ablation"
    frozen = folder / "freeze.json"
    lock = read_json(frozen)
    paired_results = read_json(folder / "paired_results.json")
    require(lock["source_sha256"] == digest(ROOT / "bc_ablation_v9.py") and
            lock["bc_actor_sha256"] == digest(out / "bc_checkpoint" / "actor.json") and
            lock["primary_freeze_sha256"] == digest(out / "independent_freeze.json") and
            lock["protocol_sha256"] == protocol_hash and digest(frozen) == paired_results["freeze_sha256"],
            "Supplementary BC lock differs")
    _, bc_rows = cohort(folder / "independent", roster("test", protocol["independent_test_layouts"]))
    compact = []
    for main_row, bc_row in zip(cases, bc_rows, strict=True):
        require(main_row["case"] == bc_row["case"] and main_row["layout_id"] == bc_row["layout_id"], "BC panel mismatch")
        outcomes = {}
        for tape in TAPES:
            record = bc_row["outcomes"][tape]["BC_new"]
            require(record["completed_legal"] and record["layout_id"] == main_row["layout_id"], "BC route invalid")
            if tape == "actual":
                check_trace(record["trace"], record, record["actions"])
            outcomes[tape] = {key: record[key] for key in FIELDS}
        compact.append({field: bc_row["case"][field] for field in ("identifier", "task")} |
                       {"layout_id": bc_row["layout_id"], "outcomes": outcomes})
    for route in ROUTES:
        subset = [row | {"outcomes": {tape: row["outcomes"][tape] | bc["outcomes"][tape] for tape in TAPES}}
                  for row, bc in zip(cases, bc_rows, strict=True) if row["case"]["task"] == route]
        expected = paired_results["scores"][route]
        for arm, other in (("BC_new", "MPC_new"), ("student_new", "BC_new"), ("PPO_only_new", "BC_new")):
            for key in ("strict", "wrong_branch"):
                _, observed = paired(subset, arm, other, key)
                require(observed == expected[arm]["paired"][other][key], "BC result differs from its committed shards")
    path = out / "bc_outcomes.json"
    atomic_json(path, compact)
    return {"physical_executions": len(compact) * len(TAPES), "scores_sha256": digest(path),
            "freeze_sha256": digest(frozen)}


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--with-bc", action="store_true")
    args = p.parse_args()
    out = args.out
    protocol_file = out / "full_route_protocol.json"
    test_file = out / "independent_freeze.json"
    protocol = read_json(protocol_file)
    reader_protocol = read_json(out / "reader_protocol.json")
    protocol_hash = digest(protocol_file)
    results = read_json(out / "full_route_results.json")
    assert_hashes(protocol["sources"])
    assert_hashes(reader_protocol["sources"])
    require(protocol["atlas_sha256"] == digest(args.atlas_pack) and
            protocol["reader_sha256"] == digest(out / "reader_bundle.json") and
            protocol_hash == results["protocol_sha256"] and digest(test_file) == results["freeze_sha256"],
            "Frozen data, reader or protocol changed")
    partitions, evidence = training(out, protocol, protocol_hash)
    _, dev = cohort(out / "dev" / "ppo_only" / "bc", roster("dev", protocol["development_layouts"]))
    _, cases = cohort(out / "independent", roster("test", protocol["independent_test_layouts"]))
    partitions.update(development={r["layout_id"] for r in dev}, independent={r["layout_id"] for r in cases})
    require(len(set.union(*partitions.values())) == sum(map(len, partitions.values())), "Overlap between experiment cohorts")
    reader_partitions = {k: set(v) for k, v in reader_protocol["layouts"].items()}
    all_partitions = list(partitions.values()) + list(reader_partitions.values())
    require(len(set.union(*all_partitions)) == sum(map(len, all_partitions)), "Reader and policy cohorts overlap")
    require({lane: evidence["counts"][lane]["complete"] for lane in LANES} == results["student_complete"],
            "Reported online size differs from actual completed size")
    compact = []
    for row in cases:
        observed = {}
        for tape, arms in row["outcomes"].items():
            require(arms["MPC_old"]["trace"] == arms["MPC_new"]["trace"], "MPC readouts did not share an observed path")
            observed[tape] = {}
            for arm, record in arms.items():
                require(record["layout_id"] == row["layout_id"] and record["completed_legal"], "Paired layouts or legality differ")
                if tape == "actual":
                    check_trace(record["trace"], record, record["actions"])
                observed[tape][arm] = {key: record[key] for key in FIELDS}
        compact.append({field: row["case"][field] for field in ("identifier", "task", "source", "shape", "profile")} |
                       {"layout_id": row["layout_id"], "outcomes": observed})
    panel = result_panel(cases, results["scores"])
    pairing_path = out / "paired_outcomes.json"
    atomic_json(pairing_path, compact)
    checks = evidence | {"protocol_sha256": protocol_hash, "freeze_sha256": digest(test_file),
                         "independent_score_rows": len(cases) * len(TAPES) * len(protocol["arms"]),
                         "independent_physical_executions": len(cases) * len(TAPES) * (len(protocol["arms"]) - 1),
                         "partition_layouts": {k: len(v) for k, v in partitions.items()},
                         "reader_partition_layouts": {k: len(v) for k, v in reader_partitions.items()},
                         "paired_outcomes_sha256": digest(pairing_path), "panel": panel}
    if args.with_bc:
        checks["supplementary_bc"] = supplementary_bc(out, cases, protocol, protocol_hash)
    atomic_json(out / "verification.json", checks)
    print("Verified", checks["partition_layouts"], checks["counts"], checks["teacher_branches"])


if __name__ == "__main__":
    main()
