"""Independently tabulate archived robotic ultrasound studies from case rows."""
import argparse
from collections import defaultdict
import json
import lzma
from pathlib import Path
from statistics import mean

import numpy as np


def read_rows(paths):
    for path in paths:
        opener = lzma.open if path.suffix == ".xz" else Path.open
        with opener(path, "rt") as stream:
            yield from (json.loads(line) for line in stream if line.strip())


def read_json(path):
    return json.loads(path.read_text())


def unique_cases(rows, key="seed"):
    values = list(rows)
    assert len({row[key] for row in values}) == len(values)
    return values


def discordance(candidate, baseline):
    deltas = [int(a) - int(b) for a, b in zip(candidate, baseline, strict=True)]
    return {
        "wins": sum(value > 0 for value in deltas),
        "losses": sum(value < 0 for value in deltas),
        "delta_pp": 100 * mean(deltas),
    }


def general_summary(arms, baseline=None):
    result = {}
    for name, rows in arms.items():
        result[name] = {
            "n": len(rows),
            "correct": sum(row["strict"] for row in rows),
            "wrong": sum(row["wrong_accept"] for row in rows),
            "mean_tcp_mm": mean(row["travel_mm"] for row in rows),
            "mean_duration_s": mean(row["elapsed_s"] for row in rows),
        }
        if baseline and name != baseline:
            result[name]["paired_completion"] = discordance(
                [row["strict"] for row in rows], [row["strict"] for row in arms[baseline]]
            )
    return result


def prodmp(original):
    root = original / "ab_prodmp_rl_20261112"
    cases = defaultdict(dict)
    for pattern, forced_name in [
        ("control_[0-9]*/RUNS.jsonl", None),
        ("eval_q_[0-9]*/RUNS.jsonl", "rl"),
        ("eval_mc_[0-9]*/RUNS.jsonl", "mc"),
        ("replicate_[0-9]*/RUNS.jsonl", None),
    ]:
        for row in read_rows(sorted(root.glob(pattern))):
            arm = forced_name or row.get("arm", row.get("policy"))
            arm = {"locked_seen": "rule", "B_seen": "immediate"}.get(arm, arm)
            case = cases[row["seed"]]
            assert arm not in case
            case[arm] = row

    expected = {"rule", "immediate", "rl"}
    assert len(cases) == 96
    for arms in cases.values():
        assert set(arms).issuperset(expected)
        assert len({row["prefix_acquisitions"] for row in arms.values()}) == 1
        traces = [row["prefix_steps"] for row in arms.values() if "prefix_steps" in row]
        assert all(trace == traces[0] for trace in traces)

    train = set()
    n_transitions, nonrule = 0, 0
    for path in sorted(root.glob("train_[0-9]*/TRANSITIONS.npz")):
        with np.load(path, allow_pickle=False) as table:
            train.update(table["episode_seed"].tolist())
            n_transitions += len(table["actions"])
            nonrule += int(np.count_nonzero(table["actions"] != 0))
    assert (len(train), n_transitions, nonrule) == (128, 2329, 720)
    assert not train.intersection(cases)
    out = {"training": {"n_cases": len(train), "transitions": n_transitions, "nonrule": nonrule}}
    for name, cohort in (
        ("primary", [row for row in cases.values() if "mc" in row]),
        ("replication", [row for row in cases.values() if "mc" not in row]),
        ("pooled", list(cases.values())),
    ):
        report = {}
        for arm in sorted(cohort[0]):
            if name == "pooled" and arm == "mc":
                continue
            rows = [row[arm] for row in cohort]
            report[arm] = {
                "n": len(rows),
                "correct": sum(row["complete"] for row in rows),
                "false_stops": sum(row["online_stop"] and not row["complete"] for row in rows),
                "mean_rechecks": mean(len(row["steps"]) for row in rows),
                "mean_reference_mm": mean(row["recheck_path_mm"] for row in rows),
                "posthoc_40_captures_2500_mm": sum(
                    row["complete"] and len(row["steps"]) <= 40 and row["recheck_path_mm"] <= 2500
                    for row in rows
                ),
            }
            if arm != "rule":
                report[arm]["paired_completion"] = discordance(
                    [row["complete"] for row in rows], [row["rule"]["complete"] for row in cohort]
                )
        out[name] = report

    quality = unique_cases(
        read_rows(sorted((root / "data_quality_20261112").glob("eval_*.jsonl"))),
        "episode_seed",
    )
    assert len(quality) == 48 and not train.intersection(row["episode_seed"] for row in quality)
    out["data_quality_correct_counts"] = {
        arm: sum(row["arms"][arm]["strict_complete"] for row in quality) for arm in quality[0]["arms"]
    }
    identity = unique_cases(read_rows(sorted((root / "identity_pair_20261112").glob("test_*.jsonl"))))
    out["one_switch_no_fisher"] = {
        arm: {
            "n": len(identity),
            "correct": sum(row["arms"][arm]["strict_complete"] for row in identity),
            "mean_captures": mean(row["arms"][arm]["captures"] for row in identity),
            "mean_reference_mm": 1000 * mean(row["arms"][arm]["path_m"] for row in identity),
        }
        for arm in ("rule", "posterior", "paired")
    }
    target = read_json(root / "frozen_q/TRAINING.json")["target_audit"]
    out["fqi_max_bellman_target"] = max(row["target_max"] for row in target["iterations"])
    return out


def knee(original):
    root = original / "ab_knee_latent_rl_20261112"
    rows = unique_cases(read_rows(sorted((root / "results").glob("test_*.jsonl"))))
    assert len(rows) == 256
    training = read_json(root / "results/TRAIN_MANIFEST.json")
    assert not set(training["case_seeds"]).intersection(row["seed"] for row in rows)
    for row in rows:
        assert row["prefix_acquisitions"] == 28
        reference = row["runs"]["rule"]["trace"][:28]
        assert [frame["site"] for frame in reference] == list(range(27, -1, -1))
        for arm in row["runs"].values():
            assert arm["captures"] == 3
            assert arm["trace"][:28] == reference
    result = general_summary({
        key: [row["runs"][key] for row in rows]
        for key in ("rule", "fqi_512_7", "cql_512_7", "sac_512_7")
    }, "rule")
    for key in result:
        result[key]["prefix_confident_wrong_replaced"] = sum(
            row["runs"][key]["prefix_confident_wrong_replaced"] for row in rows
        )
        result[key]["fresh_witnesses_per_target"] = [
            sum((witness := row["runs"][key]["witness_keys"][target]) is not None
                and witness[0] >= 28 for row in rows)
            for target in (0, 1)
        ]
    result["strict_across_train_sizes_seeds"] = {
        key: sum(row["runs"][key]["strict"] for row in rows)
        for key in rows[0]["runs"] if key != "rule"
    }
    match_root = original / "ab_knee_matched_20261112/distance"
    matched = unique_cases(read_rows([match_root / "TEST_LADDER.jsonl"]))
    assert len(matched) == 128
    assert not {row["seed"] for row in matched}.intersection(row["seed"] for row in rows)
    for row in matched:
        assert row["prefix_acquisitions"] == 28
        for arm in [row["rule"], row["selected"]["run"], *row["cql"].values()]:
            assert arm["prefix_sha256"] == row["prefix_sha256"]
            assert arm["captures"] == 3 and arm["travel_mm"] <= 80.0 + 1e-8
    result["distance_matched"] = general_summary({
        "rule": [row["rule"] for row in matched],
        "cql_cap_only": [row["cql"]["0.0"] for row in matched],
        "cql_dev_price": [row["selected"]["run"] for row in matched],
    }, "rule")
    return result


def first_scan(original):
    root = original / "ab_live_anchor_firstscan_20261112"
    result = {}
    for experiment in ("first_scan", "first_scan_matched_features"):
        rows = unique_cases(read_rows([root / experiment / "TEST.jsonl"]))
        assert len(rows) == 128
        for row in rows:
            assert row["fork_timestamps_common"]
            for arm in row["arms"].values():
                assert [frame["site"] for frame in arm["trace"]] == list(range(27, -1, -1))
                assert all(frame["view"] == 3 for frame in arm["trace"] if frame["site"] not in (4, 3, 2))
        result[experiment] = general_summary({
            key: [row["arms"][key] for row in rows] for key in rows[0]["arms"]
        }, "fixed_rule")
    return result


def certification(original):
    out = {}
    for folder, patterns, key in (
        ("ab_fisher_continuation_20261112", ("recheck_holdout", "recheck_replication"), "fisher_continuation"),
        ("ab_b_refinement_20261112", ("seen_primary", "seen_rep"), "full_posterior_vs_status"),
    ):
        arms = defaultdict(list)
        for prefix in patterns:
            for row in read_rows(sorted((original / folder).glob(f"{prefix}_[0-9]*/RUNS.jsonl"))):
                arms[row["arm"]].append(row)
        for rows in arms.values():
            unique_cases(rows)
            assert len(rows) == 72
        out[key] = {
            arm: {
                "n": len(rows), "correct": sum(row["complete"] for row in rows),
                "mean_captures": mean(len(row["steps"]) for row in rows),
                "mean_reference_mm": mean(row["recheck_path_mm"] for row in rows),
            }
            for arm, rows in arms.items()
        }
        if key == "full_posterior_vs_status":
            by_seed = {row["seed"]: row for row in arms["locked_seen"]}
            assert all(row["complete"] == by_seed[row["seed"]]["complete"] for row in arms["B_seen"])
        else:
            by_seed = {row["seed"]: row for row in arms["fixed"]}
            assert all(row["complete"] == by_seed[row["seed"]]["complete"] for row in arms["B"])
            def actions(row):
                return [(step["pose"], step["action"]) for step in row["steps"]]
            assert all(actions(row) == actions(by_seed[row["seed"]]) for row in arms["B"])
    return out


def fisher_demo(original):
    root = original / "ab_fisher_prodmp_20261112"
    return {
        file: {
            cost: {
                condition: {
                    arm: {
                        "anatomy_rmse_mm": mean(row["mse"] for row in values) ** 0.5,
                        "within_3mm": mean(row["within_3mm"] for row in values),
                        "mean_travel_deg": mean(row["travel_deg"] for row in values),
                    }
                    for arm, values in arms.items() if arm.startswith("prodmp")
                }
                for condition, arms in report["raw_by_context"].items()
            }
            for cost, report in read_json(root / file).items() if cost in ("equal_36deg", "common_48deg_cap")
        }
        for file in ("DEMO_STUDY_basis12.json", "outcome_teacher/SUMMARY.json")
    }


def map_recheck(original):
    raw = defaultdict(list)
    for row in read_rows([original / "ab_map_recheck_20261112/results/TEST.jsonl"]):
        assert row["prefix_sites"] == list(range(27, -1, -1))
        assert all(arm["captures"] == 3 for arm in row["arms"].values())
        raw[row["mode"]].append(row)
    return {
        mode: general_summary({arm: [row["arms"][arm] for row in rows]
                               for arm in rows[0]["arms"]}, "online")
        for mode, rows in raw.items()
    }


def moving_y(repo):
    root = repo / "ab_tro_y_continuation_20261112/holdout"
    common = unique_cases(read_rows([root / "COMMON.jsonl.xz"]))
    lookup = {row["seed"]: row for row in common}
    assert len(common) == 128
    assert all([frame["site"] for frame in row["prefix_frames"]] == list(range(27, -1, -1)) for row in common)
    cells = defaultdict(list)
    for row in read_rows([root / "RUNS.jsonl.xz"]):
        shared = lookup[row["seed"]]
        assert (row["prefix_sha"], row["common_probe_sha"]) == (shared["prefix_sha"], shared["common_probe_sha"])
        assert len(row["stages"]) == 2 and row["legal_completed"] and row["final_claim"]["event"] == 30
        cells[row["reserve"], row["arm"]].append(row)
    result = {}
    for (reserve, arm), rows in cells.items():
        unique_cases(rows)
        assert len(rows) == 128
        result[f"{reserve}/{arm}"] = {
            "n": len(rows), "correct": sum(row["strict"] for row in rows),
            "wrong": sum(row["false_confirm"] for row in rows),
            "mean_tcp_mm": mean(row["travel_mm"] for row in rows),
            "mean_turn_deg": mean(row["turn_deg"] for row in rows),
            "mean_duration_s": mean(row["duration_s"] for row in rows),
            "tail_switched": sum(row["tail_switched"] for row in rows),
        }
    return result


def full_b(repo):
    root = repo / "ab_full_b_rollout_20261113/data"
    result = {}
    for split in ("confirmation", "cost_diagnostic"):
        cells = defaultdict(list)
        for row in read_rows([root / split / "cases.jsonl.xz"]):
            cells[row["task"], row["initial"], row["arm"]].append(row)
        for (task, initial, arm), rows in cells.items():
            unique_cases(rows, "case_seed")
            result[f"{split}/{task}/{initial}/{arm}"] = {
                "n": len(rows), "correct": sum(row["success"] for row in rows),
                "wrong": sum(row["wrong"] for row in rows),
                "mean_tcp_mm": mean(row["path_mm"] for row in rows),
                "mean_turn_deg": mean(row["turn_deg"] for row in rows),
                "mean_duration_s": mean(row["motion_s"] for row in rows),
            }
    return result


def measure(rows):
    values = np.asarray(rows)
    return {
        "n_rows": len(values), "correct": int(values[:, 0].sum()), "wrong": int(values[:, 1].sum()),
        "mean_tcp_mm": float(values[:, 2].mean()), "mean_turn_deg": float(values[:, 3].mean()),
        "mean_duration_s": float(values[:, 4].mean()), "mean_rail_mm": float(values[:, 5].mean()),
        "mean_captures": float(values[:, 6].mean()),
    }


def compact_studies(repo):
    result = {}
    for exp, split in (
        ("ab_continuation_b_20261113", "confirmation"),
        ("ab_relational_b_20261113", "heldout"),
    ):
        cells = defaultdict(list)
        for row in read_rows([repo / exp / "data" / split / "cases.jsonl.xz"]):
            if exp.startswith("ab_continuation"):
                for arm, result_arm in row["arms"].items():
                    cells[row["task"], row["depth"], arm].append(result_arm["measured"])
            else:
                for name, context in row["contexts"].items():
                    cells[row["task"], name, "no_probe_mpc"].append(context["fresh_no_probe"]["measured"])
                    for arm, result_arm in context["fresh"].items():
                        cells[row["task"], name, arm].append(result_arm["measured"])
        result[exp] = {"/".join(map(str, key)): measure(rows) for key, rows in cells.items()}
    return result


def inverse(repo):
    root = repo / "ab_inverse_acquisition_b_20261113/data/screen"
    result = defaultdict(lambda: defaultdict(list))
    archived = read_json(root / "summary.json")
    for row in read_rows([root / "cases.jsonl.xz"]):
        values = {key: np.asarray(item) for key, item in row["measurements"].items()}
        task = result[row["task"]]
        for label in ("MPC", "coarse", "original", "fixed", "claim_only", "fixed_angles"):
            task[label].extend(values[label].tolist())
        half = len(row["repeat_seeds"]) // 2
        for i, (rank, test) in enumerate(((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half)))):
            delta = {name: (values[name][rank] - values["MPC"][rank]).mean(axis=0)
                     for name in row["candidates"]}
            unrestricted = max(delta, key=lambda name: delta[name][0])
            eligible = [name for name, diff in delta.items() if np.all(diff[2:5] <= [2. + 1e-12, 2. + 1e-12, .1 + 1e-12])]
            screened = max(eligible, key=lambda name: delta[name][0])
            stored = archived[row["task"]]["selections"][row["index"]]["halves"][i]
            assert (screened, unrestricted) == (stored["choice"], stored["unconstrained"])
            task["split_cost_screened"].extend(values[screened][test].tolist())
            task["split_hard_caps"].extend(values[unrestricted][test].tolist())
    return {task: {arm: measure(rows) for arm, rows in arms.items()} for task, arms in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="Original RealUS experiment directory")
    parser.add_argument("--archive", type=Path, required=True, help="Review checkout experiment directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "prodmp_rl": prodmp(args.original),
        "knee_latent": knee(args.original),
        "first_scan": first_scan(args.original),
        "certification": certification(args.original),
        "fisher_demo": fisher_demo(args.original),
        "map_recheck": map_recheck(args.original),
        "moving_y": moving_y(args.archive),
        "full_rollout": full_b(args.archive),
        "compact_studies": compact_studies(args.archive),
        "inverse": inverse(args.archive),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
