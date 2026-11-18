"""Summarize open-duration outcomes and the difference between B and image servo."""

import argparse
from collections import Counter

import numpy as np

from run_open import ARMS, HERE, PRIMARY, p, write_json


MODES = ("original", "extended", "open")
MEASURES = (PRIMARY, "final_wrong", "new_identity_mistake", "tcp_mm", "turn_degs", "seconds")


def scopes(rows):
    for key in ("family", "context"):
        for tag in sorted({r[key] for r in rows}):
            yield tag, [r for r in rows if r[key] == tag]


def summarize(rows):
    def trials(mode, name):
        return [trial for row in rows for trial in row["outcomes"][mode][name]]

    data = {(mode, name): np.asarray([[[t[key] for key in MEASURES] for t in row["outcomes"][mode][name]]
                                     for row in rows]) for mode in MODES for name in ARMS}
    rng = np.random.default_rng(24385)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row["family"], row["context"]), []).append(i)
    bootstrap = np.concatenate([rng.choice(indices, (8000, len(indices)), replace=True)
                                for indices in groups.values()], axis=1)

    def stats(d):
        value = d.mean(axis=1)
        interval = np.quantile(value[bootstrap].mean(axis=1), [.025, .975], axis=0)
        return {k: {"mean": float(value[:, i].mean()), "ci95": interval[:, i].tolist()}
                for i, k in enumerate(MEASURES)}

    record = {}
    for mode in MODES:
        arms = {}
        for name in ARMS:
            x = data[mode, name]
            y = trials(mode, name)
            arms[name] = {
                "correct": int(x[:, :, 0].sum()), "final_wrong": int(x[:, :, 1].sum()),
                "interim_wrong": int(x[:, :, 2].sum()),
                "cost": dict(zip(MEASURES[3:], x[:, :, 3:].mean(axis=(0, 1)).tolist(), strict=True)),
                "longest_s": max(t["seconds"] for t in y),
                "termination": dict(Counter(t["reason"] for t in y)),
                "retained_query_images": sum(t["new_query_image_retained"] for t in y),
                "correct_after_8s": sum(t[PRIMARY] for t in y if t["seconds"] > 8.00001),
                "stationary_blocks": sum(t["tail_wait_blocks"] for t in y),
                "fallback_moves": sum(t["recovery_moves"] for t in y),
                "high_level_probe_count": dict(Counter(t["probes"] for t in y)),
                "visual_corrections_in_probes": sum(t["servo_in_b_probes"] for t in y),
                "visual_corrections_after_handoff": sum(t["servo_after_handoff"] for t in y),
            }
        record[mode] = {
            "arms": arms,
            "psdp_vs_random": stats(data[mode, "psdp"] - data[mode, "random_data"]),
            "psdp_vs_same_first_direct": stats(data[mode, "psdp"] - data[mode, "psdp_direct"]),
            "relative_to_cap": {name: stats(data[mode, name] - data["original", name]) for name in ARMS},
        }
    originals = [(row["root_triggers_mpc_recovery"], row["outcomes"]["original"]["psdp"][0])
                 for row in rows]
    record["checkpoint"] = {
        "independent_cases": len(rows),
        "tapes_per_case": data["original", "psdp"].shape[1],
        "root_calls_to_psdp": len(rows),
        "first_consecutive_probe_after_normal_mpc_would_not_trigger": sum(
            not trigger and out["first"] != p.HANDOFF for trigger, out in originals),
        "first_change_after_normal_mpc_would_not_trigger": sum(
            not trigger and out["changed_from_mpc_first"] for trigger, out in originals),
        "root_trigger_fired": sum(trigger for trigger, _ in originals),
        "psdp_changes_the_first_mpc_command": sum(out["changed_from_mpc_first"] for _, out in originals),
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=HERE / "results/main", type=type(HERE))
    args = parser.parse_args()
    results = {}
    for cohort in ("test", "drift"):
        rows = p.base.read_rows(args.out / f"{cohort}.jsonl.gz")
        results[cohort] = {name: summarize(group) for name, group in (("all", rows), *scopes(rows))}
        print(cohort)
        for mode in MODES:
            arms = results[cohort]["all"][mode]["arms"]
            print(mode, {name: (a["correct"], a["final_wrong"], round(a["cost"]["seconds"], 2), a["termination"])
                         for name, a in arms.items()})
        print(results[cohort]["all"]["checkpoint"], flush=True)
    write_json(args.out / "summary.json", results)


if __name__ == "__main__":
    main()
