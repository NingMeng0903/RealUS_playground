"""Read the frozen scan logs and calculate paired episode-level differences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired_difference(x, y, rng):
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    draws = rng.integers(0, len(diff), size=(5000, len(diff)))
    return {"mean": float(diff.mean()), "bootstrap_95": np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist()}


def compare(rows):
    rng = np.random.default_rng(20261112)
    result = {}
    for regime in ("matched", "continuous", "temporary_dropout", "permanent_occlusion"):
        group = {m: {r["episode"]: r for r in rows if r["regime"] == regime and r["method"] == m}
                 for m in ("A_joint", "A_independent", "B_joint", "B_open_loop", "B_endpoint")}
        for a, b in (("B_joint", "A_joint"), ("A_joint", "A_independent"),
                     ("B_joint", "B_open_loop"), ("B_joint", "B_endpoint")):
            ids = sorted(group[a].keys() & group[b].keys())
            if not ids:
                continue
            aa, bb = ([group[m][i] for i in ids] for m in (a, b))
            if any(x["seed"] != y["seed"] for x, y in zip(aa, bb)):
                raise ValueError("mismatched episode pairing")
            record = {"n": len(ids), "a_only_success": sum(x["success"] and not y["success"] for x, y in zip(aa, bb)),
                      "b_only_success": sum(y["success"] and not x["success"] for x, y in zip(aa, bb))}
            for key in ("success", "frames", "segments", "path_mm", "failed_attempts"):
                record[key] = paired_difference([r[key] for r in aa], [r[key] for r in bb], rng)
            for key in ("rmse_mm", "joint_90_radius_mm", "joint_90_covered"):
                record[key] = paired_difference([r["final_map"][key] for r in aa],
                                                [r["final_map"][key] for r in bb], rng)
            result[f"{regime}/{a}_minus_{b}"] = record
    return result


def action_agreement(path):
    sequences = {}
    for line in path.open():
        row = json.loads(line)
        if row["method"] in ("B_joint", "B_open_loop"):
            sequences.setdefault(row["episode"], {}).setdefault(row["method"], []).append(row["path_xy_mm"])
    paired = [r for r in sequences.values() if len(r) == 2]
    return {"paired_episodes": len(paired),
            "same_first_path": sum(r["B_joint"][0] == r["B_open_loop"][0] for r in paired),
            "same_complete_path_sequence": sum(r["B_joint"] == r["B_open_loop"] for r in paired)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    (args.run / "SUMMARY.json").read_text()
    rows = list(map(json.loads, (args.run / "EPISODES.jsonl").open()))
    output = {"unit": "paired independent synthetic episode, all censoring outcomes retained",
              "comparisons": compare(rows), "action_agreement": action_agreement(args.run / "PUBLIC_SCAN.jsonl")}
    (args.run / "PAIRED.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["action_agreement"], indent=2))


if __name__ == "__main__":
    main()
