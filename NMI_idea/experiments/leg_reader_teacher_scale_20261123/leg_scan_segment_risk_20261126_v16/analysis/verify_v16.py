"""Recompute the frozen v16 contrasts from exported paired execution records."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("masked_mean", "masked_full", "read_mean", "read_full")
CONTRASTS = (("masked_mean", "read_mean"), ("masked_full", "read_full"),
             ("masked_mean", "masked_full"), ("read_mean", "read_full"))
METRICS = ("joint", "pass", "seen_and_geometry", "target_seen", "target_delivered",
           "wrong_branch", "certified")
BOOTSTRAP_SEED = 7_616_815
RESAMPLES = 6000


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha(record):
    blob = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def result(row, arm):
    return row["arms"][arm]["result"] if "arms" in row else {}


def score(row, arm):
    choice = row["arms"][arm]
    return choice["scores"][choice["index"]]


def value(row, arm, metric):
    outcome = result(row, arm)
    if not outcome:
        return 0
    if metric == "wrong_branch":
        return int(outcome["delivered_role"] is not None and outcome["delivered_role"] > 0)
    if metric == "seen_and_geometry":
        return int(outcome["pass"] and outcome["target_seen"])
    if metric == "certified":
        return int(score(row, arm)["certified"])
    return int(outcome[metric])


def action_change(row, a, b):
    selected = [row["candidate_menu"][row["arms"][key]["index"]] for key in (a, b)]
    r1, r2 = [np.asarray(s["orientation"]) for s in selected]
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(r1.T @ r2) - 1.) / 2., -1., 1.))))
    sector = selected[0]["sector"] != selected[1]["sector"]
    return sector, angle


def _audit(row, pulses, alpha):
    if "skip" in row:
        assert "arms" not in row
        return
    assert tuple(row["arms"]) == ARMS
    assert len(row["candidate_menu"]) >= 2
    assert sha(row["candidate_menu"]) == row["menu_sha256"]
    assert row["prefix_sha256"] != row["post_sha256"]
    probe = row["probe"]
    assert row["prefix_pose"] == probe["start"] and probe["start"][:2] == probe["exit"][:2]
    assert probe["exit"][2] - probe["start"][2] == row["new_frames"] > 0
    assert probe["outbound"] + probe["return"]
    for mode in ("masked", "read"):
        belief = row[mode]
        assert abs(sum(belief["probabilities"]) - 1.) <= 1.e-6
        assert abs(sum(belief["weights"]) + belief["probabilities"][-1] - 1.) <= 1.e-6
        assert len(belief["weights"]) == len(belief["centers"]) == len(belief["tangents"])
    first_pulse = row["shared_pulse_start"]
    execution_map = {}
    for key, choice in row["arms"].items():
        outcome = choice["result"]
        assert len(choice["scores"]) == len(row["candidate_menu"])
        def rank(i):
            res = choice["scores"][i]
            risk = res["risk"] if res["risk"] is not None else np.inf
            return (not res["certified"], -round(res["lower"], 10), risk,
                    row["candidate_menu"][i]["travel_ticks"], i)
        assert min(range(len(choice["scores"])), key=rank) == choice["index"]
        assert all(s["certified"] == (s["lower"] >= 1. - alpha) for s in choice["scores"])
        assert outcome["pulse_ticks"] == [first_pulse, first_pulse + pulses - 1]
        assert outcome["approach"]["tick"] == probe["exit"][2]
        assert outcome["leaving_pose"][1] == row["candidate_menu"][choice["index"]]["sector"]
        echo = outcome["selected_pulse"]
        role = None if echo is None else outcome["echo_labels"][echo[0] - first_pulse][echo[1]]
        assert role == outcome["delivered_role"]
        assert outcome["target_seen"] == any(0 in labels for labels in outcome["echo_labels"])
        assert outcome["target_delivered"] == (role == 0)
        assert outcome["joint"] == (outcome["pass"] and outcome["target_delivered"])
        assert outcome["pass"] == (choice["index"] in row["true_geometry_passing_actions"])
        assert execution_map.setdefault(choice["index"], outcome) == outcome
    assert row["unique_executed"] == len(execution_map)


def audit(rows, protocol, stage):
    by_layout = defaultdict(list)
    for row in rows:
        assert row["case"]["split"] == f"v16_{stage}"
        _audit(row, protocol["sweep_pulses"], protocol["alpha"])
        by_layout[row["case"]["identifier"]].append(row)
    assert len(by_layout) == protocol["counts"][stage]
    n_bands = len(protocol["bands"]) if stage in ("blind", "shear") else 1
    for block in by_layout.values():
        assert sorted(r["band"] for r in block) == list(range(n_bands))
        for other in block[1:]:
            for field in ("layout_id", "prefix_pose", "prefix_sha256", "menu_sha256", "masked",
                          "old_object_truth_audit", "object_keys", "skip", "shear"):
                assert other.get(field) == block[0].get(field), field
            if "probe" in other:
                assert other["probe"] == block[0]["probe"]
    assert len({r[0]["layout_id"] for r in by_layout.values()}) == len(by_layout)
    return {"layouts": len(by_layout), "paired_records": len(rows), "checks_passed": True,
            "executed_unique_actions": sum(r.get("unique_executed", 0) for r in rows),
            "unshared_menus": 0, "axial_max_error_mm": max(r.get("axial_renderer_max_error_mm", 0.) for r in rows),
            "shared_actual_contact_and_acoustic_clock": True}


def counts(rows):
    summary = {}
    for arm in ARMS:
        totals = {m: sum(value(r, arm, m) for r in rows) for m in METRICS}
        outcomes = [result(r, arm) for r in rows if "arms" in r]
        totals["rejected_echo"] = sum(x["delivered_role"] is None for x in outcomes)
        totals["artifact"] = sum(x["delivered_role"] is not None and x["delivered_role"] < 0 for x in outcomes)
        totals["certified_incorrect_geometry"] = sum(value(r, arm, "certified") and not value(r, arm, "pass")
                                                    for r in rows)
        totals["certified_joint"] = sum(value(r, arm, "certified") and value(r, arm, "joint") for r in rows)
        totals["moment_only_certified"] = sum(score(r, arm)["markov_upper"] <= .20 for r in rows if "arms" in r)
        totals["true_geometry_opportunity"] = sum(bool(r.get("true_geometry_passing_actions")) for r in rows)
        peaks = [x["peak"] for x in outcomes if "peak" in x]
        totals["true_peak_median"] = float(np.median(peaks)) if peaks else None
        totals["mean_program_ticks_without_sync"] = float(np.mean([x["program_ticks"] for x in outcomes])) if outcomes else None
        summary[arm] = totals
    return summary


def differences(rows, ref, arm):
    amounts = {}
    for metric in METRICS:
        delta = [value(r, arm, metric) - value(r, ref, metric) for r in rows]
        amounts[metric] = {"gained": sum(d > 0 for d in delta), "lost": sum(d < 0 for d in delta),
                           "net": sum(delta)}
    changes = [action_change(r, ref, arm) for r in rows if "arms" in r]
    amounts["decisions"] = {"raw_changes": sum(r["arms"][ref]["index"] != r["arms"][arm]["index"]
                                               for r in rows if "arms" in r),
                            "sector_changes": sum(c for c, angle in changes),
                            "orientation_changes_ge_1_degree": sum(angle >= 1. for c, angle in changes),
                            "physical_changes": sum(c or angle >= 1. for c, angle in changes)}
    return amounts


def intervals(rows):
    by_layout = defaultdict(list)
    for r in rows:
        by_layout[r["layout_id"]].append(r)
    blocks = list(by_layout.values())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(blocks), size=(RESAMPLES, len(blocks)))
    def interval(values):
        sample = np.asarray(values) * 100
        return {"mean_pp": float(sample.mean()),
                "ci95_pp": np.quantile(sample[draw].mean(axis=1), [.025, .975]).tolist()}
    out = {}
    for a, b in CONTRASTS:
        out[f"{b}_vs_{a}"] = {m: interval([np.mean([value(r, b, m) - value(r, a, m) for r in group])
                                            for group in blocks]) for m in METRICS}
    out["information_by_full_posterior_interaction"] = {
        m: interval([np.mean([value(r, "read_full", m) - value(r, "masked_full", m)
                             - value(r, "read_mean", m) + value(r, "masked_mean", m) for r in group])
                     for group in blocks]) for m in METRICS}
    return out


def evidence(rows):
    eligible = [r for r in rows if "arms" in r]
    tv = [.5 * np.sum(np.abs(np.asarray(r["masked"]["probabilities"]) - r["read"]["probabilities"]))
          for r in eligible]
    exclusive = [r for r in eligible if len(r["old_object_truth_audit"]) == 2
                 and r["old_object_truth_audit"].count(0) == 1
                 and all(x >= 0 for x in r["old_object_truth_audit"])]
    target_movement = [r["read"]["probabilities"][r["old_object_truth_audit"].index(0)]
                       - r["masked"]["probabilities"][r["old_object_truth_audit"].index(0)] for r in exclusive]
    switched = [(r, r["arms"]["masked_full"], r["arms"]["read_full"]) for r in eligible
                if r["arms"]["masked_full"]["index"] != r["arms"]["read_full"]["index"]]
    newly_certified = [
        r for r, masked, read in switched
        if read["scores"][read["index"]]["certified"] and not masked["scores"][read["index"]]["certified"]
    ]
    return {"records": len(rows), "executed": len(eligible),
            "probe_pulses": sum(r["new_frames"] for r in eligible),
            "probe_echo_candidates": sum(r["new_echoes"] for r in eligible),
            "belief_tv_ge_0p02": int(sum(d >= .02 for d in tv)),
            "belief_tv_median": float(np.median(tv)) if tv else None,
            "A_B_argmax_changed": sum(int(np.argmax(r["read"]["probabilities"][:2])) !=
                                      int(np.argmax(r["masked"]["probabilities"][:2])) for r in eligible),
            "exclusive_real_branch_records": len(exclusive),
            "target_mass_mean_delta": float(np.mean(target_movement)) if target_movement else None,
            "target_mass_increased_ge_0p02": int(sum(x >= .02 for x in target_movement)),
            "target_mass_decreased_ge_0p02": int(sum(x <= -.02 for x in target_movement)),
            "full_selected_action_newly_certified": len(newly_certified),
            "full_newly_certified_selected_joint": sum(value(r, "read_full", "joint") for r in newly_certified),
            "full_newly_certified_selected_joint_rescue": sum(
                value(r, "read_full", "joint") and not value(r, "masked_full", "joint") for r in newly_certified)}


def panel(rows, protocol, stage):
    arms = counts(rows)
    r = {"records": len(rows), "unexecuted": dict(Counter(r["skip"] for r in rows if "skip" in r)),
         "verification": audit(rows, protocol, stage), "arms": arms, "evidence": evidence(rows),
         "pooled_pairs": {f"{b}_vs_{a}": differences(rows, a, b) for a, b in CONTRASTS},
         "clustered_intervals": intervals(rows)}
    for dimension, key in (("band", lambda x: str(x["band"])), ("region", lambda x: str(x["region"])),
                           ("profile", lambda x: x["case"]["profile"])):
        r[dimension] = {}
        for name in sorted({key(x) for x in rows}):
            group = [x for x in rows if key(x) == name]
            r[dimension][name] = {"records": len(group), "arms": counts(group),
                                  "pairs": {f"{b}_vs_{a}": differences(group, a, b) for a, b in CONTRASTS}}
    if stage == "shear":
        stress_only = [row for row in rows if row["shear"]["accepted"]]
        approved = [row["shear"] for row in rows if row["band"] == 0 and row["shear"]["accepted"]]
        r["shear"] = {"accepted_layouts": len(approved),
                      "median_peak_mm": float(np.median([x["peak_mm"] for x in approved])),
                      "median_added_slope_deg": float(np.median([x["added_slope_deg"] for x in approved])),
                      "accepted_only_records": len(stress_only),
                      "accepted_only_arms": counts(stress_only)}
    return r


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=("smoke", "development", "blind", "shear"))
    p.add_argument("--summary", type=Path)
    args = p.parse_args()
    protocol = json.loads((args.out / "protocol.json").read_text())
    stages = [args.stage] if args.stage else ["blind", "shear"]
    combined = {stage: panel(read(args.out / f"{stage}_paired.json.gz"), protocol, stage) for stage in stages}
    result_file = args.summary or args.out / "results.json"
    result_file.write_text(json.dumps(combined, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    for stage, summary in combined.items():
        print(stage, summary["records"], {a: (c["joint"], c["pass"], c["wrong_branch"], c["certified"])
                                         for a, c in summary["arms"].items()})


if __name__ == "__main__":
    main()
