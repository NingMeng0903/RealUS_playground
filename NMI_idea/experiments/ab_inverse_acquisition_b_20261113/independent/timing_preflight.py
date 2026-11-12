"""Public-model check of arrival-time choice without outcome case labels."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from audit_paths import load_bank


def run(repo, bank_path):
    sys.path.insert(0, str(repo / "NMI_idea/experiments/ab_full_b_rollout_20261113"))
    import acoustics

    bank = json.loads(load_bank(bank_path))
    nodes = {n["id"]: n for n in bank["nodes"]}
    world = acoustics.setup_world(repo)
    clocks = np.arange(13) * .75
    durations = clocks[1:4]
    report = {"durations_s": durations.tolist(), "model_atoms": 256, "tasks": {}}
    stored_rates = np.array([edge["max_joint_rate_fraction"] for edge in bank["edges"]])
    stored_durations = np.array([edge["duration_s"] for edge in bank["edges"]])
    retimed = stored_rates[:, None] * stored_durations[:, None] / durations
    report["speed_screen"] = {"paths": len(stored_rates), "passed_by_duration": (retimed <= 1. + 1.e-10).sum(axis=0).tolist(),
                              "max_rate_fraction_by_duration": retimed.max(axis=0).tolist()}
    for task, site, label in (("daughter", 4, 1), ("trunk", 6, 0)):
        g = acoustics.Geometry(world, bank["nodes"], label)
        like, correct, _, _ = acoustics.model_tables(g, clocks, 256, "joint")
        start = bank["task_starts"][task]
        anchor = like[0, g.node_to_address[start]]
        mass = anchor.mean(axis=-1)
        active = mass > 1.e-14
        beliefs = anchor[active] / anchor[active].sum(axis=1, keepdims=True)
        direct = {nodes[e["goal"]]["view"] for e in bank["edges"]
                  if e["start"] == start and nodes[e["goal"]]["site"] == site}
        rows = []
        for view in (3, 7, 8):
            address = g.index[site, view]
            utilities = []
            for tick in (1, 2, 3):
                observation = beliefs @ like[tick, address].T
                target_joint = beliefs @ correct[tick, address].T
                accept = (observation > 1.e-260) & (target_joint >= .87 * observation)
                accept[:, 0] = False
                utilities.append(((2. * target_joint - observation) * accept).sum(axis=-1))
            value = np.asarray(utilities).T
            expected = mass[active] @ value
            rows.append(dict(view=view, checked_direct_from_start=view in direct, expected_signed_claim_utility=expected.tolist(),
                             best_timing_public_mass=np.bincount(np.argmax(value, axis=1),
                                                                 weights=mass[active], minlength=3).tolist(),
                             public_conditioned_minus_fixed=float(mass[active] @ value.max(axis=1) - expected.max())))
        report["tasks"][task] = {"anchor_support": int(active.sum()), "first_target_views": rows}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo, args.bank), indent=2))
