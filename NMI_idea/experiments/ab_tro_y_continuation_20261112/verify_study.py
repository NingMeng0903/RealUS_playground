"""Recompute the paired endpoint and costs from the frozen rollout records."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
ARMS = ("online_B", "online_MPC", "frozen_B")
RESERVES = ("narrow", "wide")
METRICS = ("travel_mm", "turn_deg", "duration_s")
MARGINS = (2.0, 2.0, 0.10)
BOOTSTRAP = 20000


def load_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_cost(motion):
    samples = motion["samples"]
    time = np.asarray(samples["t_s"])
    xyz = np.asarray(samples["atlas_xyz_mm"])
    pose = Rotation.from_matrix(samples["robot_tcp_rotation"])
    assert len(time) == len(xyz) == len(pose) == len(samples["q"])
    assert np.all(np.diff(time) > 0)
    distance = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
    turn = np.rad2deg((pose[:-1].inv() * pose[1:]).magnitude()).sum() if len(time) > 1 else 0.
    cost = np.array([distance, turn, time[-1] - time[0]])
    assert np.allclose(cost, [motion[name] for name in METRICS], rtol=0., atol=1.e-8)
    return cost


def check_run(run, shared, plan):
    assert run["prefix_sha"] == shared["prefix_sha"]
    assert run["common_probe_sha"] == shared["common_probe_sha"]
    assert run["common_probe_visible_for_evaluation"] == shared["common_probe_visible_for_evaluation"]
    if run["arm"] == "frozen_B" and "map_marginal_change_l1" in run:
        assert run["map_marginal_change_l1"] <= 1.e-10
    stages = run["stages"]
    last = shared["probe_motion"]["samples"]
    cost = np.zeros(len(METRICS))
    for stage in stages:
        motion, frame = stage["motion"], stage["frame"]
        samples = motion["samples"]
        assert np.allclose(last["q"][-1], samples["q"][0], atol=1.e-10, rtol=0.)
        assert np.isclose(last["t_s"][-1], samples["t_s"][0], atol=1.e-10, rtol=0.)
        last = samples
        cost += trace_cost(motion)
        assert (frame is not None) == motion["legal_completed"]
        if frame:
            assert np.isclose(frame["t"], samples["t_s"][-1], atol=1.e-10, rtol=0.)
            assert [frame["site"], frame["view"]] == motion["goal"]
            assert len(frame["centroids_mm"]) == len(stage["labels_for_evaluation"])
    assert np.allclose(cost, [run[name] for name in METRICS], rtol=0., atol=1.e-8)
    executed = (len(stages) == 2 and all(s["motion"]["legal_completed"] for s in stages)
                and cost[0] <= plan["shared_post_caps"]["travel_mm"] + 1.e-9
                and cost[1] <= plan["shared_post_caps"]["turn_deg"] + 1.e-9
                and cost[2] <= plan["shared_post_caps"]["duration_s"] + 1.e-9
                and sum(s["motion"]["stroke_mm"] for s in stages) <= run["reserve_mm"] + 1.e-9)
    assert bool(run["legal_completed"]) == executed
    if len(stages) == 2:
        tails = plan["mechanic_preflight"]["legal_graph"][run["reserve"]][str(run["first_view"])]
        assert stages[-1]["motion"]["path_id"] in tails
    claim = run.get("final_claim")
    correct = False
    if claim and claim["confirmed"]:
        assert claim["event"] == 30 and claim["site"] == 4 and executed
        assert claim["confidence"] >= plan["planning"]["confidence_threshold"]
        assert stages[-1]["frame"]["t"] > shared["common_probe_frame"]["t"]
        label = stages[-1]["labels_for_evaluation"][claim["ordinal"]]
        assert run["label_for_evaluation"] == label
        correct = label == 1
        assert run["false_confirm"] == (not correct)
    assert bool(run["strict"]) == (executed and correct)


def confidence(draws, quantiles=(.025, .975)):
    return np.quantile(draws, quantiles, axis=0).tolist()


def compare(left, right, subject_draws):
    def field(name, rows):
        return np.asarray([[float(row.get(name, False)) for row in reserves] for reserves in rows])

    success = field("strict", left) - field("strict", right)
    wrong = field("false_confirm", left) - field("false_confirm", right)
    cost = np.stack([field(m, left) - field(m, right) for m in METRICS], axis=-1)
    delta = success.mean(axis=1)
    delta_ci = confidence(delta[subject_draws].mean(axis=1))
    wrong_ci = confidence(wrong.mean(axis=1)[subject_draws].mean(axis=1))
    cost_ci = np.asarray(confidence(cost.mean(axis=1)[subject_draws].mean(axis=1),
                                    (.05 / 12, 1. - .05 / 12))).T
    by_reserve = {}
    for index, reserve in enumerate(RESERVES):
        changes = success[:, index]
        wins = int((changes > 0).sum())
        losses = int((changes < 0).sum())
        by_reserve[reserve] = {
            "effect": float(changes.mean()),
            "ci_95": confidence(changes[subject_draws].mean(axis=1)),
            "win_loss": [wins, losses],
            "exact_p": float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.,
        }
    width = success[:, 0] - success[:, 1]
    return {
        "effect": float(delta.mean()), "ci_95": delta_ci,
        "success_gate": bool(delta.mean() >= .05 and delta_ci[0] > 0.),
        "wrong_effect": float(wrong.mean()), "wrong_ci_95": wrong_ci,
        "wrong_gate": bool(wrong_ci[1] <= .02),
        "actual_cost_delta": dict(zip(METRICS, cost.mean(axis=(0, 1)).tolist())),
        "cost_ci_simultaneous_95": dict(zip(METRICS, cost_ci.tolist())),
        "cost_equivalent": bool(np.all(cost_ci[:, 0] >= -np.asarray(MARGINS))
                                and np.all(cost_ci[:, 1] <= MARGINS)),
        "by_reserve": by_reserve,
        "narrow_minus_wide": {"effect": float(width.mean()),
                              "ci_95": confidence(width[subject_draws].mean(axis=1))},
    }


def report_cases(cases, ordered):
    counts = {}
    for arm in ARMS:
        counts[arm] = {}
        for index, reserve in enumerate(RESERVES):
            rows = [pair[index] for pair in ordered[arm]]
            successful = sum(r["strict"] for r in rows)
            prospective_alternatives = sum(
                sum(x > 0 for x in r.get("planned", {}).get("adaptive_tail_histogram", {}).values()) > 1
                for r in rows
            )
            switches = sum(bool(r.get("tail_switched")) for r in rows)
            counts[arm][reserve] = dict(
                success=int(successful), total=len(rows), legal=sum(r["legal_completed"] for r in rows),
                wrong=sum(bool(r.get("false_confirm")) for r in rows),
                wrong_trunk=sum(bool(r.get("false_confirm_trunk")) for r in rows),
                wrong_clutter=sum(bool(r.get("false_confirm_clutter")) for r in rows),
                target_present=sum(bool(r.get("target_present")) for r in rows),
                with_prospective_tail_alternatives=prospective_alternatives,
                tail_switches=switches,
                switches_with_strict_success=sum(bool(r.get("tail_switched")) and r["strict"] for r in rows),
                choices=dict(Counter(str((r.get("first_view"), r.get("actual_tail_view"))) for r in rows)),
                mean_actual_cost={m: float(np.mean([r[m] for r in rows])) for m in METRICS})
    draws = np.random.default_rng(20261112219).integers(0, len(cases), (BOOTSTRAP, len(cases)))
    contrasts = {}
    for arm in ARMS[1:]:
        info = compare(ordered[ARMS[0]], ordered[arm], draws)
        paired = list(zip(np.ravel(ordered[ARMS[0]]), np.ravel(ordered[arm])))
        info["different_first_view"] = sum(a.get("first_view") != b.get("first_view") for a, b in paired)
        info["different_tail_view"] = sum(a.get("actual_tail_view") != b.get("actual_tail_view") for a, b in paired)
        info["B_switch_with_paired_win"] = sum(bool(a.get("tail_switched")) and a["strict"]
                                             and not b["strict"] for a, b in paired)
        contrasts[arm] = info
    return dict(subjects=len(cases), arms=counts, online_B_minus=contrasts)


def main(out):
    frozen = HERE / "FROZEN_MANIFEST.json"
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["frozen_manifest_sha256"] == sha(frozen)
    assert manifest["split"] == "holdout"
    for source, expected in manifest["sources"].items():
        assert sha(Path(source)) == expected
    complete = json.loads((out / "COMPLETE.json").read_text())
    assert complete["common_sha256"] == sha(out / "COMMON.jsonl")
    assert complete["rollouts_sha256"] == sha(out / "RUNS.jsonl")
    common, rows = load_lines(out / "COMMON.jsonl"), load_lines(out / "RUNS.jsonl")
    assert len(common) == complete["common_rows"] == manifest["cohort_n"]
    assert len(rows) == complete["rollout_rows"] == len(common) * len(ARMS) * len(RESERVES)
    assert [r["seed"] for r in common] == manifest["holdout_seed_list"]
    shared = {r["seed"]: r for r in common}
    key = {(r["seed"], r["reserve"], r["arm"]): r for r in rows}
    assert len(key) == len(rows)
    for item in common:
        assert len(item["prefix_frames"]) == 28
        trace_cost(item["probe_motion"])
        probe = item["common_probe_frame"]
        assert item["common_probe_visible_for_evaluation"] == (1 in (item["probe_labels_for_evaluation"] or []))
        payload = json.dumps(probe, sort_keys=True, allow_nan=False, separators=(",", ":"))
        assert hashlib.sha256(payload.encode()).hexdigest() == item["common_probe_sha"]
    for row in rows:
        check_run(row, shared[row["seed"]], manifest)
    cases = list(shared)
    eligible = [s for s in cases if shared[s]["common_probe_visible_for_evaluation"]]
    def ordered(seeds):
        return {arm: [[key[seed, reserve, arm] for reserve in RESERVES] for seed in seeds] for arm in ARMS}
    summary = {
        "source_manifest": sha(frozen),
        "common_probe": {
            "available": sum(item["common_probe_frame"] is not None for item in common),
            "target_present": len(eligible),
            "both_present": sum(item["common_probe_visible_for_evaluation"] and
                                item["common_probe_trunk_visible_for_evaluation"] for item in common),
            "empty": sum(item["common_probe_empty"] for item in common),
        },
        "primary": report_cases(cases, ordered(cases)),
        "target_present_sensitivity": report_cases(eligible, ordered(eligible)) if eligible else None,
    }
    (out / "VERIFIED_SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["common_probe"], indent=2))
    print(json.dumps(summary["primary"], indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "holdout")
