"""Evaluate the same decision policies with only the learned shape removed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from run import PARENT, finite_benchmark, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "learned", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = args.repo.resolve() / PARENT
    sys.path.insert(0, str(root))
    import benchmark as previous

    geom_path = root / "GEOMETRY.json"
    shape_path = args.learned / "PRODMP_SHAPE_CHECK.json"
    learned_path = args.learned / "SUMMARY.json"
    geometry = json.loads(geom_path.read_text())
    shape = json.loads(shape_path.read_text())
    learned = json.loads(learned_path.read_text())["exact"]
    costs = np.asarray([[cost if cost is not None else [np.inf] * 4 for cost in row]
                        for row in shape["endpoint_only_costs"]])
    bench = finite_benchmark(previous, geometry, costs)
    endpoint, _ = previous.summarize_exact(bench, args.out / "prescan_states.jsonl")
    delta = {
        arm: {metric: values[metric] - endpoint["total"][arm][metric] for metric in values}
        for arm, values in learned["total"].items()
    }
    payload = {
        "scope": "secondary exact finite calculation using the checked endpoint-only ProDMP costs",
        "changed_component": "learned 2-D mean weights replaced by the old endpoint-only weights",
        "input_sha256": {p.name: sha256(p) for p in (geom_path, shape_path, learned_path)},
        "same_path_feasibility_count": int(np.isfinite(costs).all(axis=-1).sum()),
        "learned_minus_endpoint_only": delta,
        "b_mpc_success_interaction": (learned["adaptive_over_mpc"] - endpoint["adaptive_over_mpc"]),
        "endpoint_only_exact": endpoint,
    }
    write_json(args.out / "SUMMARY.json", payload)
    print(json.dumps(delta, indent=2), flush=True)


if __name__ == "__main__":
    main()
