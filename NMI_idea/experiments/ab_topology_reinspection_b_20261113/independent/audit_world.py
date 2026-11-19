"""Check fixed-bank parity and public/private geometry in held-out-style draws."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from audit_bank import recheck


def inspect(world_module, geometry, seed, count, split):
    cases = world_module.cases(geometry, seed, count, split)
    assert np.isfinite(cases.centers).all()
    assert np.isfinite(cases.detect).all() and np.all((cases.detect >= 0) & (cases.detect <= 1))
    assert np.all(cases.cdf[..., -1] == 1) and np.min(np.diff(cases.cdf, axis=-1)) >= -1e-7
    assert np.array_equal(cases.families == 0, cases.labels != 0)
    assert np.isfinite(cases.fork_mm[cases.labels != 0]).all()
    assert np.isnan(cases.fork_mm[cases.labels == 0]).all()
    center = {
        site: int(np.flatnonzero((geometry.sites == site) & (geometry.views == 3))[0])
        for site in np.unique(geometry.sites)
    }
    separations = {"far": [], "near": []}
    for label in range(1, 5):
        cases_with_label = cases.labels == label
        near = 4 + (label - 1) // 2
        fork = cases.fork_mm[cases_with_label]
        assert np.all((geometry.material_s[near + 1] < fork) & (fork < geometry.material_s[near + 2]))
        for name, site in (("far", near - 1), ("near", near)):
            node = center[site]
            assert np.all(cases.detect[1, node, cases_with_label, 0] > 0)
            signed = cases.centers[1, node, cases_with_label, 0, 0] - cases.centers[1, node, cases_with_label, 1, 0]
            assert np.all((signed > 0) == (label % 2 == 0))
            separations[name].extend(np.abs(signed).tolist())
    positive = cases.families == 0
    assert np.all(cases.detect[:, center[6], cases.labels == 1, 0] == 0)
    assert np.all(cases.detect[:, center[6], cases.labels == 2, 0] == 0)
    assert np.all(cases.detect[:, center[7], positive, 0] == 0)
    assert np.all(cases.detect[:, :, cases.families == 1, 0] == 0)
    return {
        "n": count,
        "family_counts": np.bincount(cases.families, minlength=4).tolist(),
        "target_label_counts": np.bincount(cases.labels, minlength=5).tolist(),
        "median_far_lateral_separation_mm": float(np.median(separations["far"])),
        "median_near_lateral_separation_mm": float(np.median(separations["near"])),
        "near_separation_under_1_5_mm": int(np.count_nonzero(np.asarray(separations["near"]) < 1.5)),
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
    nodes, costs, _ = recheck(args.repo / "NMI_idea/experiments")
    assert len(nodes) == len(geometry.nodes)
    assert len(costs) == int(np.isfinite(geometry.cost[..., 0]).sum())
    for (start, goal), cost in costs.items():
        np.testing.assert_allclose(geometry.cost[start, goal], cost[:3], rtol=0, atol=1e-8)
    result = {
        "world_sha256": hashlib.sha256(Path(world.__file__).read_bytes()).hexdigest(),
        "nominal": inspect(world, geometry, 1032853, 256, "development"),
        "stress": inspect(world, geometry, 1032853, 256, "stress"),
        "held_shape": inspect(world, geometry, 1032853, 256, "shape_holdout"),
        "edge_count": len(costs),
        "clock_cadence_s": float(np.diff(geometry.clock_s)[0]),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
