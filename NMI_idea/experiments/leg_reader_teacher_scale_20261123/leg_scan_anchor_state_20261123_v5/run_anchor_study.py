"""Small frozen integrated test of object identity and physical anchor selection."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from study_data import (DEFAULT_PACK, EXPERIMENT, ROOT, ROUTES, STAGE_COUNTS, atomic_json, collect, digest, read_json)
import numpy as np
from fit_public import load_public
from readout_state import export_cpu
from route_metrics import paired, totals
from anchor_experiment import DEV_STREAMS, PROGRAMS, TEST_STREAMS, anchor_case, init_integrated, pack_anchors
from anchor_policy import sensitivity_check


def interval(rows, refs, metric, group):
    result = paired([v | {"strict": v[metric]} for v in rows],
                    [v | {"strict": v[metric]} for v in refs], group, 174_660_221)["strict"]
    return result


def expected_effect(selected, reference, group):
    # Each layout contributes an expected outcome across its independent
    # tapes. The bootstrap unit is the layout, not an individual rendering.
    a = np.mean(selected, axis=1)
    b = np.mean(reference, axis=1)
    keyed = {}
    for key, delta in zip(group, a - b, strict=True):
        keyed.setdefault(key, []).append(float(delta))
    values = np.array([np.mean(v) for v in keyed.values()])
    rng = np.random.default_rng(5_168_671)
    samples = rng.integers(0, len(values), (3500, len(values)))
    return {"delta": float(values.mean()), "paired95": np.quantile(values[samples].mean(axis=1), [.025, .975]).tolist()}


def counts(rows):
    return totals(rows) | {"joint": int(sum(row["joint"] for row in rows)),
                           "old_delivered_correct": int(sum(row["old_correct"] for row in rows)),
                           "suffix_correct": int(sum(row["suffix_valid"] for row in rows)),
                           "mean_time_s": float(np.mean([row["time_s"] for row in rows])),
                           "mean_travel_mm": float(np.mean([row["travel_mm"] for row in rows])),
                           "mean_correct_mm": float(np.mean([row["correct_mm"] for row in rows])),
                           "mean_contiguous_mm": float(np.mean([row["contiguous_mm"] for row in rows]))}


def physical_comparison(a, b):
    u, v = a["forward_path"], b["forward_path"]
    if [x["station"] for x in u] != [x["station"] for x in v]:
        raise RuntimeError("Anchor approaches differ in signed progress")
    x, y = (np.array([step["world_mm"] for step in seq]) for seq in (u, v))
    turns = [abs((p["sector"] - q["sector"] + 12) % 24 - 12) for p,q in zip(u, v, strict=True)]
    return {"max_world_difference_mm": float(np.linalg.norm(x - y, axis=1).max()),
            "max_angle_difference_deg": float(15. * max(turns)), "sectors_changed": bool(max(turns) > 0)}


def policy_report(meta, *, test):
    scoring = ("actual", "proof_0", "proof_1") if test else ("proof_0", "proof_1")
    result = {}
    for route in ROUTES:
        cases = [m for m in meta if m["case"]["task"] == route]
        layouts = [m["layout_id"] for m in cases]
        decks = {name: {stream: [m["programs"][stream][name] for m in cases] for stream in scoring} for name in PROGRAMS}
        choices = [max(PROGRAMS, key=lambda a: (sum(m["programs"][k][a]["utility"] for k in ("search_0", "search_1")), a == "mpc"))
                   for m in cases]
        public = [m["public_policy"] for m in cases]
        for name, deck in (("privileged_search", choices), ("public_closed_loop", public)):
            decks[name] = {stream: [m["programs"][stream][a] for m, a in zip(cases, deck, strict=True)] for stream in scoring}
        proof = ("proof_0", "proof_1")
        panel = {}
        for name, arms in decks.items():
            results = {"scores": {b: counts(arm) for b, arm in arms.items()}, "contrasts": {}}
            for baseline in ("mpc", "equal_time_hold"):
                ref = decks[baseline]
                comparison = {}
                for metric in ("strict", "joint", "wrong_branch"):
                    effect = {b: interval(arms[b], ref[b], metric, layouts) for b in scoring}
                    value, reference = (np.asarray([[r[metric] for r in row[item]] for item in proof], dtype=float).T
                                        for row in (arms, ref))
                    effect["proof_expected"] = expected_effect(value, reference, layouts)
                    effect["same_layout_rescued_both"] = int(((value - reference) > 0).all(axis=1).sum())
                    effect["same_layout_harmed_both"] = int(((value - reference) < 0).all(axis=1).sum())
                    comparison[metric] = effect
                results["contrasts"][baseline] = comparison
            panel[name] = results
        primary = scoring[0]
        motion = [physical_comparison(p, h) for p,h in zip(decks["anchor_information"][primary],
                                                           decks["equal_time_hold"][primary], strict=True)]
        ranked = [v for r in decks["anchor_information"][primary] for v in r["ranking"] if v["hypotheses"] == 2]
        active = [r for r in decks["anchor_information"][primary]]
        fixed = [(m["origin_fixed"], r["fixed_at_boundary"]) for m,r in zip(cases, active, strict=True)]
        witnesses = [(a, b) for a, b in fixed if a is not None and b is not None]
        trajectories = {
            "physically_changed_approaches": sum(v["sectors_changed"] for v in motion),
            "median_max_pose_difference_mm": float(np.median([v["max_world_difference_mm"] for v in motion])),
            "median_max_angle_difference_deg": float(np.median([v["max_angle_difference_deg"] for v in motion])),
            "ranked_decisions_with_two_hypotheses": len(ranked),
            "hypotheses_preferring_different_windows": sum(len(set(x["hypothesis_preference_slots"])) == 2 for x in ranked),
            "counterfactual_swap_changes_choice": sum(x["chosen"] != x["posterior_swap_slot"] for x in ranked),
            "same_geometry_unary_disagrees": sum(x["chosen"] != x["unary_same_geometry_slot"] for x in ranked),
            "decisions_comparable_to_previous_evidence": sum(x["previous_evidence_same_geometry_slot"] is not None for x in ranked),
            "same_geometry_choice_changes_after_new_evidence": sum(x["previous_evidence_same_geometry_slot"] is not None
                                                                   and x["chosen"] != x["previous_evidence_same_geometry_slot"] for x in ranked),
            "chosen_slot_differs_from_mpc": sum(x["chosen"] != x["mpc_slot"] for x in ranked),
            "evidence_early_stops": sum(r["evidence_stop"] is not None for r in active),
            "actual_after_anchor_revisits": sum(r["after_anchor"] is not None and r["after_anchor"]["actual_ticks"] > 0 for r in active),
            "old_fixed_candidate_with_non_neighbor_late_link": sum(b["far_non_neighbor_frames"] > 0 for _, b in witnesses),
            "old_fixed_candidate_probability_changed_with_non_neighbor_late_link": sum(
                b["far_non_neighbor_frames"] > 0 and abs(b["target_probability"] - a["target_probability"]) > .01 for a,b in witnesses),
            "fixed_truth_old_improved_with_non_neighbor_late_link": sum(
                b["far_non_neighbor_frames"] > 0 and a["truth"] == 0 and b["target_probability"] - a["target_probability"] > .01
                for a,b in witnesses),
        }
        result[route] = {"cases": len(cases), "programs": panel, "public_program_counts": dict(Counter(public)),
                         "search_program_counts": dict(Counter(choices)), "actual_closed_loop_mechanics": trajectories,
                         "possible_anchor_distance_mm": [min(r["boundary"]["signed_forward_travel_mm"] for r in active),
                                                         max(r["boundary"]["signed_forward_travel_mm"] for r in active)]}
    return result


def eligibility(report):
    verdict = {}
    for route, panel in report.items():
        by_ref = panel["programs"]["privileged_search"]["contrasts"]
        verdict[route] = {"action_expected_gate": all(by_ref[k]["strict"]["proof_expected"]["paired95"][0] > 0.
                    and by_ref[k]["joint"]["proof_expected"]["delta"] > 0.
                    and by_ref[k]["wrong_branch"]["proof_expected"]["delta"] <= 0.
                    for k in ("mpc", "equal_time_hold")), "mechanical_sensitivity_executed": True}
    return verdict


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    out = args.out.resolve()
    freeze_path = out / "reader_freeze.json"
    freeze = read_json(freeze_path)
    if digest(freeze["fit"]["weights"]) != freeze["fit"]["sha256"]:
        raise RuntimeError("The deployment reader did not verify")
    p = export_cpu(load_public(freeze["fit"]["weights"]))
    readers = {task: {"model": p, "edge_model": freeze["edge_models"]["prematch"],
               "risk": freeze["risk"][task]["prematch_track"],
               "threshold": freeze["thresholds"][task]["methods"]["prematch_track"]["threshold"]} for task in ROUTES}
    checks = sensitivity_check()
    atomic_json(out / "mechanism_sensitivity.json", checks)
    rules = {"experiment": EXPERIMENT, "reader_freeze_sha256": digest(freeze_path), "section_confidence_frozen": True,
             "state": "whole-history linked observations by frame and slot, dual trajectory forecasts",
             "joint_endpoint": "fixed old pool correct at terminal, all later regions correct with physically forward post-prefix acquisitions and legal completion",
             "anchor_suffix": "same route.advance for all arms", "clock_comparator": "same signed anchor station and elapsed clock, selected sectors may differ",
             "programs": PROGRAMS, "dev_noise": DEV_STREAMS, "test_noise": TEST_STREAMS,
             "case_counts": {k: STAGE_COUNTS[k] for k in ("anchor_dev", "anchor_test")},
             "dird": False, "source_sha256": {name: digest(ROOT / name) for name in
                                               ("study_data.py", "object_tracks.py", "forward_anchor.py", "anchor_policy.py",
                                                "anchor_experiment.py", "run_anchor_study.py")}}
    if (out / "integrated_freeze.json").exists():
        from json import dumps, loads
        if read_json(out / "integrated_freeze.json") != loads(dumps(rules)):
            raise RuntimeError("Integrated evaluation protocol changed")
    atomic_json(out / "integrated_freeze.json", rules)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_integrated,
                             initargs=(args.atlas_pack.resolve(), readers)) as pool:
        _, dev = collect(pool, out, "anchor_dev", anchor_case, pack_anchors, size=16)
        df = policy_report(dev, test=False)
        dgate = eligibility(df)
        atomic_json(out / "anchor_development.json", {"analysis": df, "gate": dgate})
        print("Independent development evidence", dgate, flush=True)
        _, test = collect(pool, out, "anchor_test", anchor_case, pack_anchors, size=16)
    checks = [r["layout_id"] for r in dev + test]
    if len(set(checks)) != len(checks):
        raise RuntimeError("Development and test layouts overlap")
    result = policy_report(test, test=True)
    folded = {"protocol_sha256": digest(out / "integrated_freeze.json"), "development_gate": dgate,
              "independent_test_gate": eligibility(result), "routes": result,
              "all_candidates_explicitly_executed_continuations": sum(len(m["programs"]) * len(PROGRAMS) for m in dev + test)}
    atomic_json(out / "anchor_results.json", folded)
    for route, row in result.items():
        print(route, {a: (p["scores"]["actual"]["strict"], p["scores"]["actual"]["joint"], p["scores"]["actual"]["wrong_branch"])
                      for a, p in row["programs"].items()}, flush=True)


if __name__ == "__main__":
    main()
