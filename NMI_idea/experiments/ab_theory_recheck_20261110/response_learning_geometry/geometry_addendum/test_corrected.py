#!/usr/bin/env python3
"""Independent checks for the corrected signless/e3nn addendum."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_corrected as corr


ROOT = Path(__file__).resolve().parent


def check() -> None:
    result = json.loads((ROOT / "RESULTS.json").read_text(encoding="utf-8"))
    raw = np.load(ROOT / "RAW_RESULTS.npz")
    assert result["status"] == "corrected_signless_e3nn_addendum"
    assert len(result["positive_runs"]) == 45
    assert len(result["negative_runs"]) == 45
    assert result["protocol"]["task_ood"] == 720972
    assert result["protocol"]["task_id"] == 720973
    assert result["protocol"]["success_threshold_original"] == 0.5
    assert result["protocol"]["task_threshold_normalized"] == 0.2

    align = result["e3nn_basis_alignment"]
    assert align["vendor_e3nn_version"] == "0.6.0"
    assert align["b1_max_abs"] < 1e-5
    assert align["b2_max_abs"] < 1e-5

    for row in result["gradients"].values():
        assert row["n"] == 16
        assert row["mean_fro_error"] < 1e-2
        assert row["max_fro_error"] < 2e-2

    data = corr.base.make_samples(12, 721140, "ood", None, False)
    model = corr.WorldDyadicMLP().to(corr.DEVICE).float()
    payload = torch.load(ROOT / "weights/world_dyadic_mlp_positive_n2048_seed101.pt", map_location=corr.DEVICE, weights_only=False)
    model.load_state_dict(payload["state_dict"]); model.eval()
    flip = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    flip["t_world"] = -data["t_world"]
    assert np.max(np.abs(corr.predict(model, data) - corr.predict(model, flip))) < 1e-5

    assert raw["negative_test_q_world"].shape == (512, 3)
    assert raw["positive_test_ood_q_world"].shape == (512, 3)
    task = result["task_scale_rows"]
    task_id = result["task_scale_id_rows"]
    assert task["oracle"]["success_rate"] == 1.0
    assert task["analytic_zero_residual"]["success_rate"] < 1.0
    assert task["e3nn_typed_basis"]["success_rate"] >= task["analytic_zero_residual"]["success_rate"]
    assert task_id["oracle"]["success_rate"] >= 0.0
    assert raw["task_id_candidate_u_local"].shape == (512, 32, 3)
    print("independent corrected geometry/e3nn checks: PASS")


if __name__ == "__main__":
    check()
