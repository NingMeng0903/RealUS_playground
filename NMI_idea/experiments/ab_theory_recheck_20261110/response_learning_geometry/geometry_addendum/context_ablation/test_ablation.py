#!/usr/bin/env python3
"""Independent checks for the fixed-basis context ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import run_ablation as abl


ROOT = Path(__file__).resolve().parent


def check() -> None:
    result = json.loads((ROOT / "RESULTS.json").read_text(encoding="utf-8"))
    raw = np.load(ROOT / "RAW_RESULTS.npz")
    assert result["status"] == "context_ablation_fixed_basis"
    assert result["protocol"]["test_seed"] == 720984
    assert result["protocol"]["n_test"] == 512
    assert result["protocol"]["candidate_count"] == 32
    assert result["protocol"]["success_threshold_mm"] == 2.0
    assert set(result["models"]) == {"constant_basis_ls", "typed_no_context", "typed_full_context"}
    for row in result["models"].values():
        assert row["response"]["units"] == "mm"
        assert set(row["targets"]) == {"target_pos", "target_neg", "target_alt"}
        for target in row["targets"].values():
            assert target["success_threshold_mm"] == 2.0
            assert len(target["chosen"]) == 512

    # Recompute the constant coefficient LS from public labels and the exact
    # shared basis, without consulting saved coefficients.
    train = abl.corrected.base.make_samples(abl.TRAIN_N, abl.TRAIN_DATA_SEED, "train", None, False)
    coeff = abl.fit_constant_coeff(train)
    assert np.allclose(coeff, raw["constant_coeff"], atol=2e-7, rtol=2e-7)

    # The no-context model is sign-invariant because its inputs use only the
    # dyadic/contraction representation, not signed t.
    test = abl.corrected.base.make_samples(12, abl.TEST_SEED, "ood", None, False)
    model = abl.TypedNoContext().to(abl.DEVICE).float()
    payload = torch.load(ROOT / "weights/typed_no_context_n2048_seed101.pt", map_location=abl.DEVICE, weights_only=False)
    model.load_state_dict(payload["state_dict"]); model.eval()
    flip = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in test.items()}
    flip["t_world"] = -test["t_world"]
    assert np.max(np.abs(abl.predict_model(model, test) - abl.predict_model(model, flip))) < 1e-5

    # All source hashes in RESULTS and manifest are reproducible.
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["weights_sha256"].items():
        assert hashlib.sha256((ROOT / "weights" / name).read_bytes()).hexdigest() == digest
    for name, path in result["source_paths"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == result["source_hashes"][name]

    print("independent context-ablation checks: PASS")


if __name__ == "__main__":
    check()
