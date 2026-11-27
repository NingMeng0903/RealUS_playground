#!/usr/bin/env python3
"""Corrected signless-input / real-e3nn addendum.

It preserves the first benchmark and writes only below geometry_addendum/.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
import run_benchmark as base  # noqa: E402

VENDOR = Path("/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_validation_20261110/theory/vendor")
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from e3nn import o3  # noqa: E402


DEVICE = base.DEVICE
HIDDEN = base.HIDDEN
EPOCHS = base.EPOCHS
BATCH_SIZE = base.BATCH_SIZE
TRAIN_SIZES = base.TRAIN_SIZES
TRAIN_SEEDS = base.TRAIN_SEEDS
TASK_TRAIN_NEG_SEEDS = (721050, 721051, 721052)
NEG_TEST_SEED = 721053
TARGET_XZ = base.TARGET_XZ_MM
SUCCESS_THRESHOLD = base.SUCCESS_THRESHOLD_MM
TEST_OOD = base.TEST_SEED_OOD
TEST_ID = base.TEST_SEED_ID
TASK_OOD_SEED = 720972
TASK_ID_SEED = 720973
FD_STEP = 1e-4


def model_input(s, u_world, n_world, t_world, R, local: bool):
    if local:
        u = torch.einsum("nij,nj->ni", R.transpose(-1, -2), u_world)
        n = torch.einsum("nij,nj->ni", R.transpose(-1, -2), n_world)
        T_world = t_world.unsqueeze(-1) * t_world.unsqueeze(-2)
        T = torch.einsum("nij,njk,nkl->nil", R.transpose(-1, -2), T_world, R)
    else:
        u, n = u_world, n_world
        T = t_world.unsqueeze(-1) * t_world.unsqueeze(-2)
    return torch.cat([s, u, n, T.reshape(T.shape[0], 9)], dim=-1), u, n, T


class WorldDyadicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(19, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 3))

    def forward(self, s, u_world, n_world, t_world, R):
        x, _, _, _ = model_input(s, u_world, n_world, t_world, R, False)
        return self.net(x)


class LocalDyadicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(19, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 3))

    def forward(self, s, u_world, n_world, t_world, R):
        x, _, _, _ = model_input(s, u_world, n_world, t_world, R, True)
        q_local = self.net(x)
        return torch.einsum("nij,nj->ni", R, q_local)


class E3NNTypedBasis(nn.Module):
    """True vendor-e3nn contractions plus scalar coefficient MLP."""

    def __init__(self):
        super().__init__()
        self.dot = o3.ElementwiseTensorProduct("1o", "1o", ["0e"])
        self.scalar_vector = o3.ElementwiseTensorProduct("0e", "1o", ["1o"])
        self.scalar_net = nn.Sequential(nn.Linear(8, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 3))

    def forward(self, s, u_world, n_world, t_world, R):
        u = torch.einsum("nij,nj->ni", R.transpose(-1, -2), u_world)
        n = torch.einsum("nij,nj->ni", R.transpose(-1, -2), n_world)
        t = torch.einsum("nij,nj->ni", R.transpose(-1, -2), t_world)
        un = np.sqrt(3.0) * self.dot(n, u)
        ut = np.sqrt(3.0) * self.dot(t, u)
        nt = (n * t).sum(-1, keepdim=True)
        scalar = torch.cat([s, (u * u).sum(-1, keepdim=True), un, ut * ut, nt * nt], dim=-1)
        coeff = self.scalar_net(scalar)
        b0 = u
        b1 = self.scalar_vector(un, n)
        b2 = self.scalar_vector(ut, t)
        q_local = coeff[:, 0:1] * b0 + coeff[:, 1:2] * b1 + coeff[:, 2:3] * b2
        return torch.einsum("nij,nj->ni", R, q_local)


MODELS = {"world_dyadic_mlp": WorldDyadicMLP, "probe_local_dyadic_mlp": LocalDyadicMLP, "e3nn_typed_basis": E3NNTypedBasis}


def flatten_tensors(data):
    return base.flatten_training_data(data)


def fit(model_name: str, data: Dict[str, np.ndarray], seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODELS[model_name]().to(DEVICE).float()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tensors = flatten_tensors(data)
    n = tensors[0].shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE].to(DEVICE)
            s, u, normal, tangent, R, y = [x[idx] for x in tensors]
            pred = model(s, u, normal, tangent, R)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, data: Dict[str, np.ndarray]) -> np.ndarray:
    return base.neural_predict(model, data)


def response_metrics(pred, data):
    return base.response_metrics(pred, data)


def global_fit(data):
    return base.global_linear_fit(data)


def global_predict(B, data):
    return base.global_linear_predict(B, data)


def candidate_metrics(pred_q, data, target=TARGET_XZ, threshold=SUCCESS_THRESHOLD):
    R = data["R"]
    pred_local = np.einsum("nij,nkj->nki", R.transpose(0, 2, 1), data["u_world"] + pred_q)
    true_cost = np.linalg.norm(data["total_local"][..., [0, 2]] - target, axis=-1)
    pred_cost = np.linalg.norm(pred_local[..., [0, 2]] - target, axis=-1)
    chosen = np.argmin(pred_cost, axis=1)
    rows = np.arange(pred_q.shape[0])
    actual = true_cost[rows, chosen]
    oracle = true_cost.min(axis=1)
    return {
        "mean_actual_error_mm_normalized": float(actual.mean()),
        "success_rate": float((actual <= threshold).mean()),
        "mean_oracle_error_normalized": float(oracle.mean()),
        "oracle_success_rate": float((oracle <= threshold).mean()),
        "mean_regret_normalized": float((actual - oracle).mean()),
        "p95_regret_normalized": float(np.quantile(actual - oracle, 0.95)),
        "chosen": chosen.tolist(),
    }


def rls_predict(data: Dict[str, np.ndarray], seed: int):
    rng = np.random.default_rng(seed)
    n = data["s"].shape[0]
    probe_local = base.RLS_PROBE_MAG * np.eye(3, dtype=np.float32)[None].repeat(n, 0)
    probe_world = np.einsum("nij,nkj->nki", data["R"], probe_local)
    n_world = np.repeat(data["n_world"][:, None], 3, axis=1)
    t_world = np.repeat(data["t_world"][:, None], 3, axis=1)
    s = np.repeat(data["s"][:, None], 3, axis=1)
    R_T = data["R"].transpose(0, 2, 1)
    n_local = np.einsum("nij,nkj->nki", R_T, n_world)
    t_local = np.einsum("nij,nkj->nki", R_T, t_world)
    q_local = base.plant_residual_local(s, probe_local, n_local, t_local)
    q_world = np.einsum("nij,nkj->nki", data["R"], q_local)
    q_obs = q_world + rng.normal(0.0, base.LABEL_NOISE_STD, q_world.shape).astype(np.float32)
    pred = np.empty_like(data["u_world"])
    for i in range(n):
        B = np.linalg.lstsq(probe_world[i], q_obs[i], rcond=None)[0]
        pred[i] = data["u_world"][i] @ B
    return pred


def gradient_check(name, model, data, n_points=16):
    rng = np.random.default_rng(721100 + len(name))
    idxs = rng.choice(data["s"].shape[0], min(n_points, data["s"].shape[0]), replace=False)
    errs = []
    for i in idxs:
        s = torch.from_numpy(data["s"][i : i + 1]).to(DEVICE).float()
        R = torch.from_numpy(data["R"][i : i + 1]).to(DEVICE).float()
        nw = torch.from_numpy(data["n_world"][i : i + 1]).to(DEVICE).float()
        tw = torch.from_numpy(data["t_world"][i : i + 1]).to(DEVICE).float()
        u0 = torch.from_numpy(data["u_local"][i : i + 1]).to(DEVICE).float()

        def fn(ul):
            uw = torch.einsum("nij,nj->ni", R, ul)
            qw = model(s, uw, nw, tw, R)
            return torch.einsum("nij,nj->ni", R.transpose(-1, -2), qw)[0]

        jac = torch.autograd.functional.jacobian(fn, u0).detach().cpu().numpy()
        jac = jac[:, 0, :]  # output(3), batch(1), input(3): same local coordinates
        ul = data["u_local"][i].copy()
        fd = np.zeros((3, 3), dtype=np.float32)
        for j in range(3):
            up = ul.copy(); um = ul.copy(); up[j] += FD_STEP; um[j] -= FD_STEP
            yp = fn(torch.from_numpy(up[None]).to(DEVICE).float()).detach().cpu().numpy()
            ym = fn(torch.from_numpy(um[None]).to(DEVICE).float()).detach().cpu().numpy()
            fd[:, j] = (yp - ym) / (2 * FD_STEP)
        errs.append(float(np.linalg.norm(jac - fd)))
    return {"n": len(errs), "mean_fro_error": float(np.mean(errs)), "max_fro_error": float(np.max(errs))}


def e3nn_basis_alignment():
    torch.manual_seed(721120)
    e = E3NNTypedBasis().to(DEVICE).float()
    u = torch.randn(32, 3, device=DEVICE)
    n = torch.randn(32, 3, device=DEVICE); n = n / n.norm(dim=-1, keepdim=True)
    t = torch.randn(32, 3, device=DEVICE); t = t / t.norm(dim=-1, keepdim=True)
    un = np.sqrt(3.0) * e.dot(n, u)
    ut = np.sqrt(3.0) * e.dot(t, u)
    b1_e = e.scalar_vector(un, n)
    b2_e = e.scalar_vector(ut, t)
    b1 = n * (n * u).sum(-1, keepdim=True)
    b2 = t * (t * u).sum(-1, keepdim=True)
    return {
        "b1_max_abs": float((b1_e - b1).abs().max().detach().cpu()),
        "b2_max_abs": float((b2_e - b2).abs().max().detach().cpu()),
        "vendor_e3nn_version": __import__("e3nn").__version__,
    }


def main():
    print(f"device={DEVICE}, vendor={VENDOR}")
    positive_runs = []
    negative_runs = []
    raw = {}
    test_ood = base.make_samples(512, TEST_OOD, "ood", None, False)
    test_id = base.make_samples(512, TEST_ID, "id", None, False)
    cand = base.make_samples(512, TEST_OOD + 100, "ood", base.CANDIDATES, False)
    neg_test = base.make_samples(512, NEG_TEST_SEED, "ood", None, True)
    for prefix, data in (("positive_test_ood", test_ood), ("positive_test_id", test_id), ("positive_candidate_ood", cand), ("negative_test", neg_test)):
        for key in ("s", "R", "n_local", "t_local", "n_world", "t_world", "u_local", "u_world", "q_world", "total_local", "total_world", "label_q_world"):
            raw[f"{prefix}_{key}"] = data[key]

    selected = {}
    n2048 = {}
    for n in TRAIN_SIZES:
        for seed in TRAIN_SEEDS:
            train = base.make_samples(n, 100000 + n * 10 + seed, "train", None, False)
            neg_seed = TASK_TRAIN_NEG_SEEDS[TRAIN_SEEDS.index(seed)]
            neg_train = base.make_samples(n, 200000 + n * 10 + neg_seed, "train", None, True)
            raw[f"positive_train_n{n}_seed{seed}_s"] = train["s"]
            raw[f"negative_train_n{n}_seed{seed}_s"] = neg_train["s"]
            B = global_fit(train); negB = global_fit(neg_train)
            artifacts = {"analytic_zero_residual": None, "global_fixed_linear": B}
            neg_artifacts = {"analytic_zero_residual": None, "global_fixed_linear": negB}
            for model_name in MODELS:
                model = fit(model_name, train, seed)
                neg_model = fit(model_name, neg_train, neg_seed)
                artifacts[model_name] = model; neg_artifacts[model_name] = neg_model
                torch.save({"model_name": model_name, "n": n, "seed": seed, "state_dict": model.state_dict()}, ROOT / "weights" / f"{model_name}_positive_n{n}_seed{seed}.pt")
                torch.save({"model_name": model_name, "n": n, "seed": neg_seed, "state_dict": neg_model.state_dict()}, ROOT / "weights" / f"{model_name}_negative_n{n}_seed{neg_seed}.pt")

            for name, artifact in artifacts.items():
                p_id = base.analytic_predict(test_id) if name == "analytic_zero_residual" else (global_predict(B, test_id) if name == "global_fixed_linear" else predict(artifact, test_id))
                p_ood = base.analytic_predict(test_ood) if name == "analytic_zero_residual" else (global_predict(B, test_ood) if name == "global_fixed_linear" else predict(artifact, test_ood))
                p_cand = base.analytic_predict(cand) if name == "analytic_zero_residual" else (global_predict(B, cand) if name == "global_fixed_linear" else predict(artifact, cand))
                positive_runs.append({"n": n, "seed": seed, "model": name, "response_id": response_metrics(p_id, test_id), "response_ood": response_metrics(p_ood, test_ood), "action_ood": candidate_metrics(p_cand, cand)})
                negA = neg_artifacts[name]
                pn = base.analytic_predict(neg_test) if name == "analytic_zero_residual" else (global_predict(negB, neg_test) if name == "global_fixed_linear" else predict(negA, neg_test))
                negative_runs.append({"n": n, "seed": seed, "model": name, "response_negative_test": response_metrics(pn, neg_test)})
                if n == 2048 and seed == 101:
                    n2048[name] = artifact

    # Online RLS and corrected geometry use the largest positive weights.
    rls = rls_predict(cand, TEST_OOD + 301)
    task_ood = base.make_samples(512, TASK_OOD_SEED, "ood", base.CANDIDATES, False)
    task_id = base.make_samples(512, TASK_ID_SEED, "id", None, False)
    task_id_cand = base.make_samples(512, TASK_ID_SEED + 100, "id", base.CANDIDATES, False)
    task_target = np.asarray([0.40, -0.30], dtype=np.float32)
    task_threshold = 0.20
    task_rows = {}
    for name, artifact in n2048.items():
        p = base.analytic_predict(task_ood) if name == "analytic_zero_residual" else (global_predict(global_fit(base.make_samples(2048, 120480 + 101, "train", None, False)), task_ood) if name == "global_fixed_linear" else predict(artifact, task_ood))
        task_rows[name] = candidate_metrics(p, task_ood, task_target, task_threshold)
    rls_task = rls_predict(task_ood, TASK_OOD_SEED + 301)
    task_rows["online_RLS_3probe"] = candidate_metrics(rls_task, task_ood, task_target, task_threshold)
    task_rows["online_RLS_3probe"]["probe_count"] = base.RLS_PROBE_COUNT
    task_rows["oracle"] = {
        "success_rate": float((np.linalg.norm(task_ood["total_local"][..., [0, 2]] - task_target, axis=-1).min(axis=1) <= task_threshold).mean()),
        "mean_oracle_error_normalized": float(np.linalg.norm(task_ood["total_local"][..., [0, 2]] - task_target, axis=-1).min(axis=1).mean()),
    }
    task_id_rows = {}
    for name, artifact in n2048.items():
        p_id = base.analytic_predict(task_id_cand) if name == "analytic_zero_residual" else (global_predict(global_fit(base.make_samples(2048, 120480 + 101, "train", None, False)), task_id_cand) if name == "global_fixed_linear" else predict(artifact, task_id_cand))
        task_id_rows[name] = candidate_metrics(p_id, task_id_cand, task_target, task_threshold)
    rls_task_id = rls_predict(task_id_cand, TASK_ID_SEED + 301)
    task_id_rows["online_RLS_3probe"] = candidate_metrics(rls_task_id, task_id_cand, task_target, task_threshold)
    task_id_rows["online_RLS_3probe"]["probe_count"] = base.RLS_PROBE_COUNT
    task_id_rows["oracle"] = {
        "success_rate": float((np.linalg.norm(task_id_cand["total_local"][..., [0, 2]] - task_target, axis=-1).min(axis=1) <= task_threshold).mean()),
        "mean_oracle_error_normalized": float(np.linalg.norm(task_id_cand["total_local"][..., [0, 2]] - task_target, axis=-1).min(axis=1).mean()),
    }
    raw["task_ood_u_local"] = task_ood["u_local"]
    raw["task_ood_q_world"] = task_ood["q_world"]
    raw["task_id_u_local"] = task_id["u_local"]
    raw["task_id_q_world"] = task_id["q_world"]
    raw["task_id_candidate_u_local"] = task_id_cand["u_local"]

    # Geometry checks for corrected n2048/seed101 models.
    geom = {}
    grads = {}
    base_query = test_ood
    Q = base.random_rotations(np.random.default_rng(721130), 1, "ood")[0]
    transformed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in base_query.items()}
    transformed["u_world"] = np.einsum("ij,nj->ni", Q, base_query["u_world"])
    transformed["n_world"] = np.einsum("ij,nj->ni", Q, base_query["n_world"])
    transformed["t_world"] = np.einsum("ij,nj->ni", Q, base_query["t_world"])
    transformed["R"] = np.einsum("ij,njk->nik", Q, base_query["R"])
    for name, artifact in n2048.items():
        p = base.analytic_predict(base_query) if name == "analytic_zero_residual" else (global_predict(global_fit(base.make_samples(2048, 120480 + 101, "train", None, False)), base_query) if name == "global_fixed_linear" else predict(artifact, base_query))
        pQ = base.analytic_predict(transformed) if name == "analytic_zero_residual" else (global_predict(global_fit(base.make_samples(2048, 120480 + 101, "train", None, False)), transformed) if name == "global_fixed_linear" else predict(artifact, transformed))
        flip = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in base_query.items()}; flip["t_world"] = -base_query["t_world"]
        pf = base.analytic_predict(flip) if name == "analytic_zero_residual" else (global_predict(global_fit(base.make_samples(2048, 120480 + 101, "train", None, False)), flip) if name == "global_fixed_linear" else predict(artifact, flip))
        eq = pQ - np.einsum("ij,nj->ni", Q, p)
        geom[name] = {"so3_mean": float(np.linalg.norm(eq, axis=1).mean()), "so3_p95": float(np.quantile(np.linalg.norm(eq, axis=1), .95)), "signflip_mean": float(np.linalg.norm(pf - p, axis=1).mean()), "signflip_p95": float(np.quantile(np.linalg.norm(pf - p, axis=1), .95))}
        if name in MODELS:
            grads[name] = gradient_check(name, artifact, base_query)

    alignment = e3nn_basis_alignment()
    result = {"status": "corrected_signless_e3nn_addendum", "protocol": {"positive_test_ood": TEST_OOD, "positive_test_id": TEST_ID, "negative_test": NEG_TEST_SEED, "task_ood": TASK_OOD_SEED, "task_id": TASK_ID_SEED, "task_target_normalized": task_target.tolist(), "task_threshold_normalized": task_threshold, "success_threshold_original": SUCCESS_THRESHOLD, "device": str(DEVICE), "vendor": str(VENDOR)}, "positive_runs": positive_runs, "negative_runs": negative_runs, "task_scale_rows": task_rows, "task_scale_id_rows": task_id_rows, "geometry": geom, "gradients": grads, "e3nn_basis_alignment": alignment, "interpretation": {"first_results_preserved": True, "task_saturation": "Original 0.50 mm success regime is retained; task-scale regime is a posthoc development addendum.", "scope": "Synthetic response mechanism only."}}
    np.savez_compressed(ROOT / "RAW_RESULTS.npz", **raw)
    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (ROOT / "REPORT.txt").write_text(make_report(result), encoding="utf-8")
    return result


def fmt(x): return f"{float(x):.6g}"


def make_report(r):
    lines = ["Corrected signless-input / real-e3nn addendum", "==============================================", "Previous manual benchmark is preserved; this report is an addendum.", "", "Task-scale addendum: 1 normalized unit=10 mm, target=[4,-3] mm, success<=2 mm (normalized target [.4,-.3], threshold .2).", "model | task success | task regret | oracle error"]
    for n, row in r["task_scale_rows"].items():
        if n == "oracle":
            lines.append(f"oracle: success={fmt(row.get('success_rate', float('nan')))}, oracle={fmt(row.get('mean_oracle_error_normalized', float('nan')))}")
        else:
            lines.append(f"{n}: success={fmt(row.get('success_rate', float('nan')))}, regret={fmt(row.get('mean_regret_normalized', float('nan')))}, oracle={fmt(row.get('mean_oracle_error_normalized', float('nan')))}")
    lines.append("ID task-scale (seed 720973):")
    for n, row in r["task_scale_id_rows"].items():
        if n == "oracle":
            lines.append(f"  oracle: success={fmt(row.get('success_rate', float('nan')))}, oracle={fmt(row.get('mean_oracle_error_normalized', float('nan')))}")
        else:
            lines.append(f"  {n}: success={fmt(row.get('success_rate', float('nan')))}, regret={fmt(row.get('mean_regret_normalized', float('nan')))}")
    lines.extend(["", "Corrected geometry (n2048/seed101)", "----------------------------------"])
    for n, row in r["geometry"].items():
        g = r["gradients"].get(n, {})
        lines.append(f"{n}: SO3={fmt(row['so3_mean'])}/{fmt(row['so3_p95'])}, signflip={fmt(row['signflip_mean'])}/{fmt(row['signflip_p95'])}, FD={fmt(g.get('mean_fro_error', float('nan')))}/{fmt(g.get('max_fro_error', float('nan')))}")
    lines.extend(["", "Negative train/test (independent q=0 plant)", "------------------------------------------"])
    for model in sorted({x['model'] for x in r['negative_runs']}):
        rows = [x for x in r['negative_runs'] if x['model'] == model and x['n'] == 2048 and x['seed'] == 101]
        if rows:
            lines.append(f"{model}: negative-test response RMSE={fmt(rows[0]['response_negative_test']['rmse_vector_mm'])} mm")
    lines.extend(["", f"e3nn vendor basis alignment: b1={fmt(r['e3nn_basis_alignment']['b1_max_abs'])}, b2={fmt(r['e3nn_basis_alignment']['b2_max_abs'])}, version={r['e3nn_basis_alignment']['vendor_e3nn_version']}", "", "Negative train/test are independent from positive data and reported separately. The task-scale experiment is posthoc; if fixed linear/RLS or all methods are adequate, no learning necessity is claimed."])
    return "\n".join(lines) + "\n"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest():
    names = ["PROTOCOL.txt", "TASK_SCALE_ADDENDUM.txt", "run_corrected.py", "test_corrected.py", "RAW_RESULTS.npz", "RESULTS.json", "REPORT.txt"]
    out = {"python": platform.python_version(), "torch": torch.__version__, "files_sha256": {n: sha(ROOT / n) for n in names}, "weights_sha256": {p.name: sha(p) for p in sorted((ROOT / 'weights').glob('*.pt'))}}
    (ROOT / "MANIFEST.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(); write_manifest()
