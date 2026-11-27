#!/usr/bin/env python3
"""Bounded causal context ablation with a fixed typed geometric basis."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
ADDENDUM = ROOT.parent
PARENT = ADDENDUM.parent
if str(ADDENDUM) not in sys.path:
    sys.path.insert(0, str(ADDENDUM))
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
import run_corrected as corrected  # noqa: E402

DEVICE = corrected.DEVICE
TRAIN_N = 2048
TRAIN_SEED = 101
TRAIN_DATA_SEED = 100000 + TRAIN_N * 10 + TRAIN_SEED
TEST_SEED = 720984
N_TEST = 512
TARGETS = {
    "target_pos": np.asarray([0.40, -0.30], dtype=np.float32),
    "target_neg": np.asarray([-0.40, 0.30], dtype=np.float32),
    "target_alt": np.asarray([0.20, 0.40], dtype=np.float32),
}
PHYSICAL_SCALE_MM = 10.0
SUCCESS_THRESHOLD_NORMALIZED = 0.20


class TypedNoContext(corrected.nn.Module):
    """Same vendor e3nn contractions, with only invariant geometry inputs."""

    def __init__(self):
        super().__init__()
        from e3nn import o3

        self.dot = o3.ElementwiseTensorProduct("1o", "1o", ["0e"])
        self.scalar_vector = o3.ElementwiseTensorProduct("0e", "1o", ["1o"])
        self.scalar_net = nn.Sequential(
            nn.Linear(4, corrected.HIDDEN),
            nn.Tanh(),
            nn.Linear(corrected.HIDDEN, corrected.HIDDEN),
            nn.Tanh(),
            nn.Linear(corrected.HIDDEN, 3),
        )

    def forward(self, s, u_world, n_world, t_world, R):
        u = torch.einsum("nij,nj->ni", R.transpose(-1, -2), u_world)
        n = torch.einsum("nij,nj->ni", R.transpose(-1, -2), n_world)
        t = torch.einsum("nij,nj->ni", R.transpose(-1, -2), t_world)
        un = np.sqrt(3.0) * self.dot(n, u)
        ut = np.sqrt(3.0) * self.dot(t, u)
        nt = (n * t).sum(-1, keepdim=True)
        scalar = torch.cat([(u * u).sum(-1, keepdim=True), un, ut * ut, nt * nt], dim=-1)
        coeff = self.scalar_net(scalar)
        b0 = u
        b1 = self.scalar_vector(un, n)
        b2 = self.scalar_vector(ut, t)
        q_local = coeff[:, 0:1] * b0 + coeff[:, 1:2] * b1 + coeff[:, 2:3] * b2
        return torch.einsum("nij,nj->ni", R, q_local)


def basis_local(data):
    u = data["u_local"]
    n = data["n_local"]
    t = data["t_local"]
    if u.ndim == 3:
        n = n[:, None, :]
        t = t[:, None, :]
    b0 = u
    b1 = n * np.sum(n * u, axis=-1, keepdims=True)
    b2 = t * np.sum(t * u, axis=-1, keepdims=True)
    return np.stack([b0, b1, b2], axis=-1).astype(np.float32)


def fit_constant_coeff(train):
    B = basis_local(train)
    q_label_local = np.einsum("nij,nj->ni", train["R"].transpose(0, 2, 1), train["label_q_world"])
    coeff = np.linalg.lstsq(B.reshape(-1, 3), q_label_local.reshape(-1), rcond=None)[0].astype(np.float32)
    return coeff


def constant_predict(coeff, data):
    B = basis_local(data)
    q_local = np.einsum("n...jc,c->n...j", B, coeff)
    return np.einsum("nij,n...j->n...i", data["R"], q_local)


def fit_no_context(train):
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    model = TypedNoContext().to(DEVICE).float()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tensors = corrected.flatten_tensors(train)
    n = tensors[0].shape[0]
    gen = torch.Generator(device="cpu").manual_seed(TRAIN_SEED)
    for _ in range(corrected.EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, corrected.BATCH_SIZE):
            idx = perm[start : start + corrected.BATCH_SIZE].to(DEVICE)
            s, u, normal, tangent, R, y = [x[idx] for x in tensors]
            pred = model(s, u, normal, tangent, R)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    return model


def predict_model(model, data):
    return corrected.predict(model, data)


def physical_response_metrics(pred, data):
    m = corrected.response_metrics(pred, data)
    for key in ("rmse_vector_mm", "mean_norm_mm", "p95_norm_mm", "max_norm_mm"):
        m[key] *= PHYSICAL_SCALE_MM
    m["units"] = "mm"
    return m


def physical_action_metrics(pred, data, target):
    m = corrected.candidate_metrics(pred, data, target, SUCCESS_THRESHOLD_NORMALIZED)
    for key in ("mean_actual_error_mm_normalized", "mean_oracle_error_normalized", "mean_regret_normalized", "p95_regret_normalized"):
        m[key.replace("_normalized", "_mm")] = m[key] * PHYSICAL_SCALE_MM
    m["success_threshold_mm"] = 2.0
    m["target_mm"] = (target * PHYSICAL_SCALE_MM).tolist()
    return m


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    train = corrected.base.make_samples(TRAIN_N, TRAIN_DATA_SEED, "train", None, False)
    test = corrected.base.make_samples(N_TEST, TEST_SEED, "ood", corrected.base.CANDIDATES, False)
    coeff = fit_constant_coeff(train)
    no_context = fit_no_context(train)
    full_path = ADDENDUM / "weights/e3nn_typed_basis_positive_n2048_seed101.pt"
    full_payload = torch.load(full_path, map_location=DEVICE, weights_only=False)
    full = corrected.E3NNTypedBasis().to(DEVICE).float()
    full.load_state_dict(full_payload["state_dict"]); full.eval()
    torch.save({"model": "constant_basis_ls", "coeff": coeff}, ROOT / "weights/constant_basis_ls_n2048_seed101.pt")
    torch.save({"model": "typed_no_context", "state_dict": no_context.state_dict()}, ROOT / "weights/typed_no_context_n2048_seed101.pt")
    torch.save(full_payload, ROOT / "weights/typed_full_context_source_snapshot.pt")

    models = {
        "constant_basis_ls": (coeff, lambda d: constant_predict(coeff, d)),
        "typed_no_context": (no_context, lambda d: predict_model(no_context, d)),
        "typed_full_context": (full, lambda d: predict_model(full, d)),
    }
    raw = {
        "train_s": train["s"],
        "train_label_q_world": train["label_q_world"],
        "train_u_local": train["u_local"],
        "test_s": test["s"],
        "test_R": test["R"],
        "test_u_local": test["u_local"],
        "test_total_local": test["total_local"],
        "test_q_world": test["q_world"],
        "constant_coeff": coeff,
    }
    rows = {}
    for name, (artifact, predict_fn) in models.items():
        pred = predict_fn(test)
        raw[f"prediction_{name}"] = pred
        raw[f"error_{name}"] = pred - test["q_world"]
        rows[name] = {"response": physical_response_metrics(pred, test)}
        rows[name]["targets"] = {}
        for target_name, target in TARGETS.items():
            rows[name]["targets"][target_name] = physical_action_metrics(pred, test, target)

    source_files = {
        "corrected_protocol": ADDENDUM / "PROTOCOL.txt",
        "task_scale_protocol": ADDENDUM / "TASK_SCALE_ADDENDUM.txt",
        "corrected_source": ADDENDUM / "run_corrected.py",
        "full_weight_source": full_path,
    }
    result = {
        "status": "context_ablation_fixed_basis",
        "protocol": {
            "train_n": TRAIN_N,
            "train_seed": TRAIN_SEED,
            "train_data_seed": TRAIN_DATA_SEED,
            "test_seed": TEST_SEED,
            "n_test": N_TEST,
            "candidate_count": corrected.base.CANDIDATES,
            "physical_scale_mm_per_unit": PHYSICAL_SCALE_MM,
            "success_threshold_mm": 2.0,
            "targets_normalized": {k: v.tolist() for k, v in TARGETS.items()},
            "model_contract": "same b0/b1/b2; only context s is removed for no-context model",
        },
        "constant_coefficients": coeff.tolist(),
        "models": rows,
        "source_hashes": {k: source_hash(v) for k, v in source_files.items()},
        "source_paths": {k: str(v) for k, v in source_files.items()},
        "interpretation": {
            "question": "Does public context s add predictive/action value once geometric basis is held fixed?",
            "task_scope": "Three centering target readouts only; not long-axis or acoustic-window transfer.",
            "limits": "Synthetic plant, candidate-set evaluation, no claim of real anatomy or e3nn necessity.",
        },
    }
    np.savez_compressed(ROOT / "RAW_RESULTS.npz", **raw)
    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (ROOT / "REPORT.txt").write_text(make_report(result), encoding="utf-8")
    return result


def fmt(x): return f"{float(x):.6g}"


def make_report(r):
    lines = [
        "Context ablation with fixed typed geometric basis",
        "==================================================",
        "Units: response/action errors are physical mm (1 normalized unit=10 mm); success threshold=2 mm.",
        "All three models share b0=u, b1=n(n.u), b2=t(t.u); only context conditioning differs.",
        "",
        "Model | response RMSE mm | target + success/regret mm | target - success/regret mm | target alt success/regret mm",
    ]
    for name, row in r["models"].items():
        rr = row["response"]["rmse_vector_mm"]
        vals = []
        for target in ("target_pos", "target_neg", "target_alt"):
            q = row["targets"][target]
            vals.append(f"{fmt(q['success_rate'])}/{fmt(q['mean_regret_mm'])}")
        lines.append(f"{name} | {fmt(rr)} | " + " | ".join(vals))
    lines.extend([
        "",
        "Interpretation",
        "--------------",
        "This isolates the incremental effect of public context s after holding the geometric basis and candidate budget fixed. The three target columns are new centering readouts only; they are not long-axis or acoustic-window transfer. Results remain synthetic and do not establish e3nn necessity or real ultrasound performance.",
    ])
    return "\n".join(lines) + "\n"


def write_manifest():
    names = ["PROTOCOL.txt", "run_ablation.py", "test_ablation.py", "RAW_RESULTS.npz", "RESULTS.json", "REPORT.txt"]
    weights = sorted((ROOT / "weights").glob("*.pt"))
    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "files_sha256": {n: source_hash(ROOT / n) for n in names},
        "weights_sha256": {p.name: source_hash(p) for p in weights},
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(); write_manifest()
