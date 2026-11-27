"""Recompute the old one-step centering candidate-removal diagnostic."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "ab_theory_recheck_20261110/response_learning_geometry"
sys.path.insert(0, str(SOURCE))
from run_benchmark import CANDIDATES, make_samples


def main():
    data = make_samples(512, 720972, "ood", CANDIDATES, False)
    target = np.array([.40, -.30], dtype=np.float32)
    predicted_local = np.einsum(
        "nij,nkj->nki", data["R"].transpose(0, 2, 1), data["u_world"]
    )
    predicted_cost = np.linalg.norm(predicted_local[..., [0, 2]] - target, axis=-1)
    actual_cost = np.linalg.norm(data["total_local"][..., [0, 2]] - target, axis=-1)
    qualified = actual_cost <= .20
    row = np.arange(len(actual_cost))
    chosen = np.argmin(predicted_cost, axis=1)
    without_fixed = 1 + np.argmin(predicted_cost[:, 1:], axis=1)
    counts = {
        "analytic_selects_fixed": int((chosen == 0).sum()),
        "analytic_success_all": int(qualified[row, chosen].sum()),
        "analytic_success_without_fixed": int(qualified[row, without_fixed].sum()),
        "fixed_alone_success": int(qualified[:, 0].sum()),
        "oracle_success_all": int(qualified.any(axis=1).sum()),
        "oracle_success_without_fixed": int(qualified[:, 1:].any(axis=1).sum()),
    }
    stored = json.loads((HERE / "OLD_CANDIDATE_DIAGNOSTIC.json").read_text())
    for name, value in counts.items():
        if stored[name] != value:
            raise ValueError(f"{name}: recomputed {value}, stored {stored[name]}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
