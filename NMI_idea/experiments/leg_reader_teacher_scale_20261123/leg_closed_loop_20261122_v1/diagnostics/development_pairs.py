"""Re-score committed, unseen development scans against paired CPU baselines."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["LEG_ENABLE_DIRD"] = "1"
os.environ["LEG_ENABLE_MATERIAL_ATLAS"] = "1"

from run_online import BatchJob, CT_PACK, DEV_SEED, pool, harvest, protect_holdout
from environment import Arm, TRAIN_SEEDS, run_cases
from storage import atomic_json, digest, read_json, read_lines, stash
from audits import summarize
import numpy as np


PAIRED_METRICS = ("pass", "wrong_delivery", "correct_regions", "correct_mm", "captured_regions",
                  "capture_mm", "probes", "time_s", "travel_mm")


def match(learned, baseline, seed):
    if set(learned) != set(baseline):
        raise RuntimeError("Development pairing keys differ")
    flags = {"rescued_regions": 0, "lost_regions": 0, "wrong_added": 0, "wrong_removed": 0,
             "wrong_scans_added": 0, "wrong_scans_removed": 0,
             "rescued_strict_scans": 0, "lost_strict_scans": 0,
             "rescued_scans_at_least_one_region": 0, "lost_scans_at_least_one_region": 0,
             "rescues_after_executed_probe": 0, "losses_after_executed_probe": 0}
    details, bands = [], defaultdict(list)
    for key in sorted(learned):
        a, b = learned[key], baseline[key]
        for field in ("task", "profile", "source", "shape", "world_seed", "tape_seed", "layout_id"):
            if a[field] != b[field]:
                raise RuntimeError(f"Paired scene does not match in {field} for {key}")
        aa, bb = np.asarray(a["region_outcomes"]), np.asarray(b["region_outcomes"])
        rescues = np.flatnonzero((aa == 1) & (bb != 1))
        losses = np.flatnonzero((aa != 1) & (bb == 1))
        flags["rescued_regions"] += len(rescues)
        flags["lost_regions"] += len(losses)
        flags["wrong_added"] += int(((aa == 0) & (bb != 0)).sum())
        flags["wrong_removed"] += int(((aa != 0) & (bb == 0)).sum())
        flags["wrong_scans_added"] += int(a["wrong_delivery"] > b["wrong_delivery"])
        flags["wrong_scans_removed"] += int(a["wrong_delivery"] < b["wrong_delivery"])
        flags["rescued_strict_scans"] += int(a["pass"] > b["pass"])
        flags["lost_strict_scans"] += int(a["pass"] < b["pass"])
        flags["rescued_scans_at_least_one_region"] += bool(len(rescues))
        flags["lost_scans_at_least_one_region"] += bool(len(losses))
        if a["probes"]:
            flags["rescues_after_executed_probe"] += len(rescues)
            flags["losses_after_executed_probe"] += len(losses)
        values = np.array([a[m] - b[m] for m in PAIRED_METRICS])
        bands[a["band"]].append(values)
        details.append({"case": key, "source": a["source"], "profile": a["profile"], "task": a["task"],
                        "layout_id": a["layout_id"], "probes": a["probes"],
                        "reference_outcomes": bb, "student_outcomes": aa,
                        "rescued": rescues, "lost": losses})
    rng = np.random.default_rng(seed)
    clusters = np.asarray([np.mean(bands[band], axis=0) for band in sorted(bands)])
    draws = rng.integers(len(clusters), size=(4000, len(clusters)))
    interval = np.quantile(clusters[draws].mean(axis=1), [.025, .975], axis=0).T
    comparisons = {name: {"mean_delta_per_scan": float(value), "band_bootstrap_95": limits}
                   for name, value, limits in zip(PAIRED_METRICS, clusters.mean(axis=0), interval)}
    return flags | {"pairs": len(learned), "random_band_clusters": len(clusters),
                    "delta": comparisons, "case_details": details}


def arm_map(rows, name):
    output = {}
    for row in rows:
        if row["arm"] != name:
            continue
        if row["identifier"] in output or not row["complete"] or row["split"] != "development":
            raise RuntimeError("Nonunique, incomplete or nondevelopment case in paired audit")
        output[row["identifier"]] = row
    return output


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    protect_holdout()
    protocol_path = ROOT / "protocol.json"
    seed = TRAIN_SEEDS[0]
    protocol = read_json(protocol_path)
    training = ROOT / "training" / f"seed_{seed}"
    teacher = training / "teacher_ppo"
    destination = ROOT / "diagnostics" / "development"
    baseline = destination / "baselines" / "00000"
    if not baseline.is_dir():
        smoke = read_json(ROOT / "smoke" / "verification.json")
        cases = list(run_cases(protocol["count"]["development_cases"], "development", DEV_SEED))
        initial = teacher / "rollout_weights" / "0000000.npz"
        arms = (Arm("mpc"), Arm("active_mpc"), Arm("initial_student", "initial"))
        jobs = [BatchJob(case, arms, {"initial": str(initial)}, smoke["threshold"]) for case in cases]
        with pool(SimpleNamespace(workers=args.workers, pack=CT_PACK)) as executor:
            data = harvest(executor, jobs, chunksize=1)
        stake = {"partition": "development", "protocol_sha256": digest(protocol_path),
                 "teacher_initial_weights_sha256": digest(initial), "threshold": smoke["threshold"]}
        stash(baseline, data, stake)
    summary = {"scope": "fixed development shapes and measurement tapes",
               "protocol_sha256": digest(protocol_path), "training_seed": seed, "baselines": {}, "student": {}}
    reference = list(read_lines(baseline / "episodes.jsonl.gz"))
    for name in ("mpc", "active_mpc", "initial_student"):
        summary["baselines"][name] = summarize(arm_map(reference, name).values())
    loaded = {}
    for arm in ("teacher_ppo", "ppo"):
        directory = training / arm / "validation"
        for folder in sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9]")):
            commit = read_json(folder / "commit.json")
            rows = arm_map(read_lines(folder / "episodes.jsonl.gz"), arm)
            count = commit["training_complete_scans"]
            loaded[arm, count] = rows
            summary["student"][f"{arm}_{count}"] = {
                "training_complete_scans": count, "totals": summarize(rows.values()),
                "paired": {name: match(rows, arm_map(reference, name), seed + count)
                            for name in ("mpc", "active_mpc", "initial_student")}
            }
    if not summary["student"]:
        raise RuntimeError("No committed development validation to pair")
    summary["teacher_vs_ppo"] = {
        str(count): match(rows, loaded["ppo", count], seed + count)
        for (arm, count), rows in loaded.items() if arm == "teacher_ppo" and ("ppo", count) in loaded
    }
    atomic_json(destination / "paired_checks.json", summary)
    for key, row in summary["student"].items():
        p = row["paired"]["mpc"]
        a = row["paired"]["active_mpc"]
        print(f"{key}  {p['pairs']} pairs    MPC {p['rescued_regions']} rescued / {p['lost_regions']} lost"
              f"    active MPC {a['rescued_regions']} rescued / {a['lost_regions']} lost", flush=True)


if __name__ == "__main__":
    main()
