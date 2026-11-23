"""Recount acquired recoveries and route outcomes from the committed v10 logs."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np

from recovery_task import DEFAULT_PACK, ROOT, ROUTES, V9
from recovery_policy import _predict, read_bundle
from study_v9 import atomic_json, digest, read_json, verify_shard


def rows(folder):
    locks = sorted(folder.glob("*/commit.json"))
    return [record for path in locks for record in verify_shard(path.parent)[1]]


def check_motion(record):
    trace = record["trace"]
    assert record["completed_legal"] and trace[-1]["skill"] == "submit"
    assert trace[-1]["next_tick"] == record["terminal_clock"]
    assert len(trace) == record["steps"]
    assert all(item["ticks"] == item["next_tick"] - item["tick"] and item["ticks"] >= 0 for item in trace)
    assert all(item["ticks"] > 0 for item in trace[:-1]) and trace[-1]["ticks"] == 0
    for old, new in zip(trace, trace[1:]):
        assert (old["next_station"], old["next_sector"], old["next_tick"]) == (new["station"], new["sector"], new["tick"])
    taken = [item for item in trace if item["skill"] in ("revisit", "recover", "disambiguate")]
    assert len(taken) == record["repairs"]
    assert len({(item["goal_region"], item["goal_station"], item["goal_sector"]) for item in taken}) == len(taken)
    for item in taken:
        assert (item["next_station"], item["next_sector"]) == (item["goal_station"], item["goal_sector"])
    return taken


def check_outcome(record):
    tags = record["selected_ids"]
    assert len(tags) == 6 and len(record["capture"]) == 6
    assert record["strict"] == (all(x == 0 for x in tags) and record["completed_legal"])
    assert record["wrong_branch"] == any(x > 0 for x in tags)
    assert all(y or x != 0 for x, y in zip(tags, record["capture"], strict=True))


def independent(out, report):
    tested = rows(out / "independent")
    summary, longest, decisions = {}, 0, 0
    for name in ROUTES:
        cases = [row for row in tested if row["case"]["task"] == name]
        assert len(set(row["layout_id"] for row in cases)) == len(cases)
        arm_names = tuple(cases[0]["outcomes"]["actual"])
        for row in cases:
            for arms in row["outcomes"].values():
                assert arms["MPC_v9"]["trace"] == arms["MPC_new"]["trace"]
                for arm in arms.values():
                    check_outcome(arm)
                    check_motion(arm)
                    longest = max(longest, arm["terminal_clock"])
                    decisions = max(decisions, arm["steps"])
        data = {}
        for arm in arm_names:
            scored = [row["outcomes"]["actual"][arm] for row in cases]
            strict = [sum(row["outcomes"][tape][arm]["strict"] for row in cases)
                      for tape in ("actual", "proof_0", "proof_1")]
            assert strict == report["stats"][name][arm]["strict_by_tape"]
            assert sum(s["wrong_branch"] for s in scored) == report["stats"][name][arm]["wrong_branch"]
            by_region = np.count_nonzero(np.array([s["selected_ids"] for s in scored]) == 0, axis=0)
            assert by_region.tolist() == report["stats"][name][arm]["per_region_correct"]
            lost = sum(np.any(np.asarray(s["selected_ids"]) != 0) and s["acquired_ceiling"] for s in scored)
            data[arm] = {"all_seen_failed": int(lost), "strict": strict, "layouts": len(scored),
                         "backward": sum(s["negative_station_moves"] for s in scored)}
        summary[name] = data
    return tested, {"panel": summary, "maximum_terminal_tick": longest, "maximum_macros": decisions}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    arg = parser.parse_args()
    out = arg.out
    protocol, report = (read_json(out / f"{name}.json") for name in ("protocol", "results"))
    assert digest(out / "protocol.json") == report["protocol_sha256"]
    assert digest(out / "independent_freeze.json") == report["freeze_sha256"]
    assert digest(out / "paired_outcomes.json") == report["paired_sha256"]
    assert digest(arg.atlas_pack) == protocol["atlas_sha256"]
    assert all(digest(ROOT / name) == sha for name, sha in protocol["sources"].items())
    assert all(digest(V9 / name) == sha for name, sha in protocol["v9_sources"].items())
    assert digest(V9 / "run" / "reader_bundle.json") == protocol["initial_reader_sha256"]
    reader_dir = out / "adapted_reader"
    reader = read_json(reader_dir / "reader_manifest.json")
    freeze = read_json(out / "independent_freeze.json")
    assert reader["bundle_sha256"] == digest(reader_dir / "reader_bundle.json") == freeze["reader_sha256"]
    assert reader["device"].startswith("cuda") and all(x > 0 for x in reader["max_absolute_parameter_updates"].values())
    assert reader["checkpoint_sha256"] == digest(reader_dir / "reader_latest.pt")
    tested, result = independent(out, report)
    assert len(tested) == protocol["cohorts"]["test"]
    teacher = {name: rows(out / name) for name in ("strong", "compose_teacher", "strong_dev")}
    outcomes = {}
    for key, sampled in teacher.items():
        count = Counter()
        for case in sampled:
            b, a = case["proof_baseline"], case["proof_teacher"]
            check_outcome(b)
            check_outcome(a)
            check_motion(a)
            count.update(layouts=1, rescue=a["strict"] and not b["strict"], loss=b["strict"] and not a["strict"],
                         backward=a["negative_station_moves"], two_options=len(case["sequence"]) == 2)
        outcomes[key] = dict(count)
    online, total, true_start, adaptation = [], 0, 0, Counter()
    behavior = out / "skill_adapted" / "actor.json"
    max_log_error = 0.
    for ledger in sorted((out / "online").glob("*/evidence.json")):
        evidence = read_json(ledger)
        lock = out / f"online_{ledger.parent.name}.json"
        assert digest(lock) == evidence["protocol_sha256"] and read_json(lock)["behavior_actor_sha256"] == digest(behavior)
        source = read_bundle(behavior)
        sampled = rows(ledger.parent)
        assert len(sampled) == evidence["complete_routes"] == protocol["cohorts"]["online_batch"]
        behavior = ledger.parent / "actor.json"
        assert digest(behavior) == evidence["actor_sha256"] and evidence["actor_delta_l2"] > 0. and evidence["device"].startswith("cuda")
        for shard in sorted(ledger.parent.glob("*/commit.json")):
            data, _ = verify_shard(shard.parent)
            with np.load(data, allow_pickle=False) as x:
                for feats, mask, index, stored in zip(x["x"], x["mask"], x["action"], x["logp"], strict=True):
                    assert mask[index]
                    p = _predict(source, feats, mask)[1][index]
                    max_log_error = max(max_log_error, abs(np.log(max(float(p), 1.e-20)) - float(stored)))
        for r in sampled:
            check_outcome(r["metrics"])
            check_motion(r["metrics"])
            true_start += r["focus"] is None
            t = r["teacher"]
            if t:
                assert r["layout_id"] == t["layout_id"] and len(t["outcomes"]) == 2
                move = r["metrics"]["trace"][t["step"]]
                assert (move["station"], move["tick"]) == (t["station"], t["tick"])
                adaptation["paired_prefixes"] += 1
                for b in t["outcomes"]:
                    d = int(b["alternative_strict"]) - int(b["reference_strict"])
                    adaptation["strict_rescue"] += d > 0
                    adaptation["strict_loss"] += d < 0
        online.extend(sampled)
        total += len(sampled)
    cohorts = [tested, online, *teacher.values(), *[rows(out / "reader_data" / part) for part in ("train", "cal", "dev")]]
    combined = [r["layout_id"] for group in cohorts for r in group]
    assert len(combined) == len(set(combined))
    assert max_log_error < 1.e-5
    picked = out / "online" / f"{freeze['candidate']:02d}" if freeze["candidate"] else out / "skill_adapted"
    assert digest(picked / "actor.json") == freeze["student_sha256"]
    result.update(teacher=outcomes, online=total, starting_at_route_origin=true_start, adaptation=dict(adaptation),
                  verified_cohorts_disjoint=len(combined), max_online_logp_replay_error=max_log_error)
    atomic_json(out / "verification.json", result)
    print("Verified", len(tested), "independent layouts,", total, "online scans,", dict(adaptation))


if __name__ == "__main__":
    main()
