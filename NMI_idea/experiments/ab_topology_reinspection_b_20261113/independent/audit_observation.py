"""Enumerate the observation-only topology and identity signal in the public bank."""
import argparse
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import sys

import numpy as np


def fine_to_coarse(grid):
    x_groups = [0, 0, 0, 1, 2, 2, 2] if grid == 3 else [0, 0, 1, 2, 3, 4, 4]
    z_groups = [0, 0, 0, 1, 2, 2, 2]
    return np.asarray([1 + x * 3 + z for x in x_groups for z in z_groups])


def assignment_weights(grid):
    bins = 1 + grid * 3
    packets = tuple(combinations_with_replacement(range(bins), 3))
    packet_ids = {packet: i for i, packet in enumerate(packets)}
    assignments = np.asarray(list(product(range(bins), repeat=3)), dtype=np.int64)
    ordered = np.sort(assignments, axis=1)
    packet = np.asarray([packet_ids[tuple(row)] for row in ordered], dtype=np.int64)
    matches = (ordered == assignments[:, :1]) & (assignments[:, :1] != 0)
    matches = matches / np.maximum(matches.sum(axis=1, keepdims=True), 1)
    return assignments, packet, len(packets), matches


def packet_law(detect, cdf, grid, weights):
    assignments, packet, n_packets, matches = weights
    pmf = np.diff(cdf, axis=-1, prepend=0)
    factors = np.empty((*detect.shape, grid * 3 + 1))
    factors[..., 0] = 1.0 - detect
    mapping = fine_to_coarse(grid)
    for obs in range(1, factors.shape[-1]):
        factors[..., obs] = detect * pmf[..., mapping == obs].sum(axis=-1)
    per_assignment = np.prod(factors[:, np.arange(3), assignments], axis=-1)
    total = np.stack([np.bincount(packet, weights=row, minlength=n_packets) for row in per_assignment])
    joint = np.stack([
        np.stack([np.bincount(packet, weights=row * matches[:, k], minlength=n_packets) for k in range(3)])
        for row in per_assignment
    ])
    np.testing.assert_allclose(total.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(joint.sum(axis=(1, 2)), detect[:, 0], rtol=0.0, atol=1e-7)
    return total, joint


def mutual_information(total, labels):
    ordered = np.unique(labels)
    combined = total.mean(axis=0)
    by_label = np.stack([total[labels == label].sum(axis=0) / len(labels) for label in ordered])
    independence = by_label.sum(axis=1, keepdims=True) * combined[None]
    ratio = np.divide(by_label, independence, out=np.ones_like(by_label), where=by_label > 0)
    return float(np.sum(by_label * np.log2(ratio)))


def inspect(world_module, geometry, seed, count, split, grid):
    cases = world_module.cases(geometry, seed, count, split)
    weights = assignment_weights(grid)
    thresholds = (0.65, 0.75, 0.87)
    metrics = []
    conditionally_correct = {
        position: {str(c): [] for c in thresholds} for position in ("near", "far")
    }
    near = 4 + (cases.labels - 1) // 2
    positive = cases.labels > 0
    for site in np.unique(geometry.sites):
        for view in np.unique(geometry.views):
            node = np.flatnonzero((geometry.sites == site) & (geometry.views == view))[0]
            # One fixed post-prescan clock is an observation-law diagnostic.
            total, joint = packet_law(cases.detect[4, node], cases.cdf[4, node], grid, weights)
            metrics.append({
                "site": int(site),
                "view": int(view),
                "target_label_bits": mutual_information(total, cases.labels),
                "topology_bits": mutual_information(total, cases.families),
                "has_daughter_bits": mutual_information(total, positive),
            })
            if view != 3:
                continue
            best = joint.max(axis=1)
            for name, goal in (("near", near), ("far", near - 1)):
                selected = (goal == site) & positive
                for threshold in thresholds:
                    right = (best * (best >= threshold * total)).sum(axis=1)
                    conditionally_correct[name][str(threshold)].extend(right[selected].tolist())
    return {
        "single_frame_information": metrics,
        "truth_conditioned_target_slot_correct_acceptance": {
            name: {threshold: float(np.mean(values)) for threshold, values in stats.items()}
            for name, stats in conditionally_correct.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.stage.resolve()))
    import world

    geometry = world.load(args.repo)
    result = {"world_sha256": hashlib.sha256(Path(world.__file__).read_bytes()).hexdigest()}
    for grid in (3, 5):
        result[str(grid)] = {
            split: inspect(world, geometry, 1032853, 256, split, grid)
            for split in ("development", "stress", "shape_holdout")
        }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
