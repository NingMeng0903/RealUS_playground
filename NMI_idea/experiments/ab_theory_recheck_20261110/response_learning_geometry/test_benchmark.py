#!/usr/bin/env python3
"""Independent checks for the synthetic response geometry benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_benchmark as bench


ROOT = Path(__file__).resolve().parent


def check() -> None:
    results = json.loads((ROOT / "RESULTS.json").read_text(encoding="utf-8"))
    raw = np.load(ROOT / "RAW_RESULTS.npz")
    assert results["status"] == "synthetic_conditioned_response_geometry_benchmark"
    assert results["protocol"]["train_sizes"] == [128, 512, 2048]
    assert results["protocol"]["train_seeds"] == [101, 202, 303]
    assert len(results["runs"]) == 45  # 5 models x 3 sizes x 3 seeds
    assert len(list((ROOT / "weights").glob("*.pt"))) == 27
    assert raw["candidate_ood_u_local"].shape == (512, 32, 3)
    assert raw["test_ood_u_local"].shape == (512, 3)

    # Rebuild the declared plant invariances independently of learned models.
    data = bench.make_samples(16, 720970, "ood", None, negative=False)
    q = bench.plant_residual_local(data["s"], data["u_local"], data["n_local"], data["t_local"])
    q_flip = bench.plant_residual_local(data["s"], data["u_local"], data["n_local"], -data["t_local"])
    assert np.allclose(q, q_flip, atol=1e-7)
    Q = bench.random_rotations(np.random.default_rng(720971), 1, "ood")[0]
    transformed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    transformed["u_world"] = np.einsum("ij,nj->ni", Q, data["u_world"])
    transformed["n_world"] = np.einsum("ij,nj->ni", Q, data["n_world"])
    transformed["t_world"] = np.einsum("ij,nj->ni", Q, data["t_world"])
    transformed["R"] = np.einsum("ij,njk->nik", Q, data["R"])
    assert np.allclose(transformed["R"].transpose(0, 2, 1) @ transformed["u_world"][..., None],
                       data["R"].transpose(0, 2, 1) @ data["u_world"][..., None], atol=1e-5)

    # The typed basis is loaded from the saved largest model and must be
    # sign-invariant in t; this is an independent inference check, not a
    # reliance on the summary table.
    typed = bench.TypedResidualBasis().to(bench.DEVICE)
    payload = torch.load(ROOT / "weights/typed_residual_basis_n2048_seed101.pt", map_location=bench.DEVICE, weights_only=False)
    typed.load_state_dict(payload["state_dict"]); typed.eval()
    flip_data = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    flip_data["t_world"] = -data["t_world"]
    pred = bench.neural_predict(typed, data)
    pred_flip = bench.neural_predict(typed, flip_data)
    assert np.max(np.abs(pred - pred_flip)) < 1e-5

    # All response metrics and geometry checks are finite; typed/local gradient
    # checks are explicitly recorded rather than inferred from smoothness.
    for name, row in results["geometry_checks_selected_n2048_seed101"].items():
        for value in row.values():
            assert np.isfinite(value)
    for name, row in results["gradient_checks_selected_n2048_seed101"].items():
        assert row["n"] > 0
        assert np.isfinite(row["mean_fro_error"])
        assert np.isfinite(row["max_fro_error"])
    assert results["online_rls"]["probe_count"] == 3
    assert results["protocol"]["success_threshold_mm"] == 0.5

    # Negative q=0 plant was retained in every learned run.
    for run in results["runs"]:
        assert "negative_response" in run
        assert run["negative_response"]["n"] == 512 * 32
    assert results["interpretation"]["typed_basis"].startswith("Manual exact")

    print("independent response geometry checks: PASS")


if __name__ == "__main__":
    check()
