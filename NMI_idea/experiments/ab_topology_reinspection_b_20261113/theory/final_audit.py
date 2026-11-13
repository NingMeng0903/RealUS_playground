"""Read-only independent recomputation of the frozen result records."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", type=Path, required=True)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
root = args.stage
repo = args.repo
bank_path = repo / "NMI_idea/experiments/ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz"
with lzma.open(bank_path, "rt") as stream:
    bank = json.load(stream)
sites = {int(row["id"]): int(row["site"]) for row in bank["nodes"]}
expected = {1: (4, 3), 2: (4, 3), 3: (5, 4), 4: (5, 4)}
report = {"cohorts": {}, "checks": {}}

for cohort in ("confirm", "shape"):
    data, base, freeze = {}, None, None
    for seed in (7, 29, 61):
        directory = root / f"{cohort}_seed{seed}"
        summary = json.loads((directory / "summary.json").read_text())
        if freeze is None:
            freeze = summary["settings"]["source_sha256"]
        assert summary["settings"]["source_sha256"] == freeze
        for path in sorted(directory.glob("*.jsonl")):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 256
            assert [row["index"] for row in rows] == list(range(256))
            if base is None:
                base = rows
            for row, common in zip(rows, base):
                assert all(row[key] == common[key] for key in ("truth", "family", "positive", "prefix_hash", "prefix"))
                assert row["positive"] == int(row["truth"] > 0)
                assert row["captures"] == 5 + len(row["trace"])
                assert abs(row["seconds"] - 1.6 * row["captures"]) < 1e-9
                assert np.all(np.asarray(row["suffix_cost"]) <= np.array([65, 90, 250]) + 1e-8)
                assert np.all(np.asarray(row["prefix_cost"]) <= np.array([142, 114, 401]) + 1e-8)
                true_frames = False
                if row["prediction"] > 0:
                    chosen = row["chosen"]
                    times = chosen["frames"]
                    assert min(times) > 5 and times[1] == times[0] + 1
                    observations = {trace[0]: trace for trace in row["trace"]}
                    assert [sites[node] for node in chosen["nodes"]] == list(expected[row["prediction"]])
                    assert all(observations[t][1] == a and observations[t][2] == p
                               for t, a, p in zip(times, chosen["nodes"], chosen["packets"]))
                    true_frames = all(observations[t][3][slot] == 0 for t, slot in zip(times, chosen["slots"]))
                assert row["correct_images"] == int(true_frames)
                success = row["prediction"] == row["truth"] and (
                    row["truth"] == 0 or (true_frames and row["fork_error_mm"] is not None and row["fork_error_mm"] <= 5))
                assert row["success"] == int(success)
                assert row["wrong"] == int(row["prediction"] >= 0 and not success)
            data[f"{path.stem}_{seed}"] = rows

    metrics = {}
    for arm in ("mpc", "rollout", "witness", "fisher", "greedy", "nearest", "prescan_only"):
        rows = data[f"{arm}_7"]
        metrics[arm] = {key: sum(row[key] for row in rows) for key in ("success", "wrong", "affirmative_error")}
        metrics[arm]["positive_success"] = sum(row["success"] for row in rows if row["positive"])
        metrics[arm]["mean_cost"] = {key: float(np.mean([row[key] for row in rows]))
                                     for key in ("tcp_mm", "turn_deg", "rail_mm", "seconds")}
    for arm in ("ppo", "mlp", "cql", "bc", "bc_mlp"):
        metrics[arm] = {
            "seeds": [7, 29, 61],
            "positive_success": [sum(row["success"] for row in data[f"{arm}_{seed}"] if row["positive"])
                                 for seed in (7, 29, 61)],
            "all_success": [sum(row["success"] for row in data[f"{arm}_{seed}"]) for seed in (7, 29, 61)],
        }
    contrasts = {}
    rng = np.random.default_rng(726181)
    for first, second in (("mpc", "nearest"), ("rollout", "mpc"), ("witness", "mpc")):
        a, b = data[f"{first}_7"], data[f"{second}_7"]
        result = {}
        for stratum in ("all", "positive"):
            indices = [i for i, row in enumerate(a) if stratum == "all" or row["positive"]]
            delta = np.asarray([a[i]["success"] - b[i]["success"] for i in indices])
            resample = rng.integers(len(indices), size=(20000, len(indices)))
            result[stratum] = {
                "n": len(indices), "wins": int((delta == 1).sum()), "losses": int((delta == -1).sum()),
                "delta_pp": float(100 * delta.mean()),
                "ci95_pp": (100 * np.quantile(delta[resample].mean(1), [.025, .975])).tolist(),
            }
        contrasts[f"{first}_vs_{second}"] = result
    report["cohorts"][cohort] = {"metrics": metrics, "independent_contrasts": contrasts}
    report["checks"][cohort] = {
        "records": sum(map(len, data.values())),
        "source_hashes_equal_across_training_seeds": True,
        "same_case_truth_and_prefix_across_arms_and_seeds": True,
        "fresh_pair_identity_success_and_cost_recomputed": True,
        "current_source_matches_freeze": {
            key: hashlib.sha256((root / key).read_bytes()).hexdigest() == value
            for key, value in freeze.items() if key.endswith(".py") and (root / key).exists()
        },
    }

args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"checks": report["checks"], "contrasts": {
    key: value["independent_contrasts"] for key, value in report["cohorts"].items()}}, indent=2))
