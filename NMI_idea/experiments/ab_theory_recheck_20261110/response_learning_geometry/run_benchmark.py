#!/usr/bin/env python3
"""Synthetic conditioned response benchmark.

The plant and all evaluation contracts are frozen in PROTOCOL.txt.  This is a
mechanism benchmark, not a real ultrasound/contact implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
WEIGHT_DIR = ROOT / "weights"
WEIGHT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

TRAIN_SIZES = (128, 512, 2048)
TRAIN_SEEDS = (101, 202, 303)
TEST_SEED_OOD = 720950
TEST_SEED_ID = 720951
LABEL_NOISE_STD = 0.01
EPOCHS = 100
BATCH_SIZE = 256
HIDDEN = 64
TARGET_XZ_MM = np.asarray([0.40, -0.30], dtype=np.float32)
SUCCESS_THRESHOLD_MM = 0.50
CANDIDATES = 32
RLS_PROBE_MAG = 0.5
RLS_PROBE_COUNT = 3
FD_STEP = 1e-4


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-8)


def skew(axis: np.ndarray) -> np.ndarray:
    x, y, z = axis
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=np.float32))
    K = skew(axis)
    I = np.eye(3, dtype=np.float32)
    return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def random_rotations(rng: np.random.Generator, n: int, mode: str) -> np.ndarray:
    axes = normalize(rng.normal(size=(n, 3)).astype(np.float32))
    if mode == "train":
        angles = rng.uniform(-np.deg2rad(15.0), np.deg2rad(15.0), size=n)
    elif mode == "id":
        angles = rng.uniform(-np.deg2rad(15.0), np.deg2rad(15.0), size=n)
    elif mode == "ood":
        # Uniform unit quaternions, converted to SO(3).
        q = rng.normal(size=(n, 4)).astype(np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        w, x, y, z = q.T
        R = np.empty((n, 3, 3), dtype=np.float32)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R
    else:
        raise ValueError(mode)
    return np.stack([axis_angle_rotation(a, ang) for a, ang in zip(axes, angles)]).astype(np.float32)


def sample_conditions(rng: np.random.Generator, n: int) -> np.ndarray:
    depth = rng.uniform(0.2, 1.0, size=n)
    force = rng.uniform(0.0, 1.0, size=n)
    compliance = rng.uniform(0.3, 1.3, size=n)
    history = rng.uniform(-1.0, 1.0, size=n)
    # Public normalized values used by the declared plant.
    return np.stack(
        [2.0 * (depth - 0.2) / 0.8 - 1.0, 2.0 * force - 1.0, 2.0 * (compliance - 0.3) - 1.0, history],
        axis=1,
    ).astype(np.float32)


def plant_residual_local(
    s: np.ndarray, u_local: np.ndarray, n_local: np.ndarray, t_local: np.ndarray
) -> np.ndarray:
    """Declared smooth anisotropic compliance plant, vectorized."""

    d, f, k, h = [s[..., i] for i in range(4)]
    a = np.sum(n_local * u_local, axis=-1)
    v = np.sum(t_local * u_local, axis=-1)
    alpha = 0.22 + 0.08 * d + 0.05 * f + 0.03 * h
    beta = 0.12 + 0.06 * k + 0.04 * f**2
    gamma = 0.10 + 0.05 * (1.0 - d) + 0.04 * np.cos(np.pi * h)
    b0 = u_local
    b1 = n_local * a[..., None]
    b2 = t_local * v[..., None]
    return (
        alpha[..., None] * b0
        + beta[..., None] * (1.0 + 0.35 * np.tanh(2.0 * f) * a**2)[..., None] * b1
        + gamma[..., None] * (1.0 + 0.20 * d * v**2)[..., None] * b2
    ).astype(np.float32)


def make_samples(
    n: int,
    seed: int,
    rotation_mode: str,
    actions_per_context: int | None = None,
    negative: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate independent contexts and public noisy response labels."""

    rng = np.random.default_rng(seed)
    if actions_per_context is None:
        n_context = n
        n_action = 1
    else:
        n_context = n
        n_action = actions_per_context
    s = sample_conditions(rng, n_context)
    R = random_rotations(rng, n_context, rotation_mode)
    n_local = normalize(rng.normal(size=(n_context, 3)).astype(np.float32))
    t_local = normalize(rng.normal(size=(n_context, 3)).astype(np.float32))
    if actions_per_context is None:
        direction = normalize(rng.normal(size=(n_context, 3)).astype(np.float32))
        magnitude = rng.uniform(0.15, 0.8, size=n_context).astype(np.float32)
        u_local = direction * magnitude[:, None]
    else:
        direction = normalize(rng.normal(size=(n_context, n_action, 3)).astype(np.float32))
        magnitude = rng.uniform(0.15, 0.8, size=(n_context, n_action)).astype(np.float32)
        u_local = direction * magnitude[..., None]
        # Candidate set includes a deterministic moderate action direction for
        # stable oracle coverage without changing candidate count.
        u_local[:, 0, :] = np.asarray([0.35, 0.0, -0.25], dtype=np.float32)
    n_local_b = n_local[:, None, :] if actions_per_context is not None else n_local
    t_local_b = t_local[:, None, :] if actions_per_context is not None else t_local
    s_b = s[:, None, :] if actions_per_context is not None else s
    q_local = np.zeros_like(u_local) if negative else plant_residual_local(s_b, u_local, n_local_b, t_local_b)
    total_local = u_local + q_local
    if actions_per_context is not None:
        u_world = np.einsum("nij,nkj->nki", R, u_local)
    else:
        u_world = np.einsum("nij,nj->ni", R, u_local)
    n_world = np.einsum("nij,nj->ni", R, n_local)
    t_world = np.einsum("nij,nj->ni", R, t_local)
    if actions_per_context is not None:
        q_world = np.einsum("nij,nkj->nki", R, q_local)
        total_world = np.einsum("nij,nkj->nki", R, total_local)
    else:
        q_world = np.einsum("nij,nj->ni", R, q_local)
        total_world = np.einsum("nij,nj->ni", R, total_local)
    noise = np.random.default_rng(seed + 991).normal(0.0, LABEL_NOISE_STD, size=q_world.shape).astype(np.float32)
    return {
        "s": s,
        "R": R,
        "n_local": n_local,
        "t_local": t_local,
        "n_world": n_world,
        "t_world": t_world,
        "u_local": u_local.astype(np.float32),
        "u_world": u_world.astype(np.float32),
        "q_local": q_local.astype(np.float32),
        "q_world": q_world.astype(np.float32),
        "total_local": total_local.astype(np.float32),
        "total_world": total_world.astype(np.float32),
        "label_q_world": (q_world + noise).astype(np.float32),
        "label_noise_world": noise,
        "negative": np.asarray(negative),
    }


class WorldMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(13, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 3))

    def forward(self, s, u_world, n_world, t_world, R):
        x = torch.cat([s, u_world, n_world, t_world], dim=-1)
        return self.net(x)


class LocalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(13, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 3))

    def forward(self, s, u_world, n_world, t_world, R):
        u = torch.einsum("nij,nj->ni", R.transpose(-1, -2), u_world)
        n = torch.einsum("nij,nj->ni", R.transpose(-1, -2), n_world)
        t = torch.einsum("nij,nj->ni", R.transpose(-1, -2), t_world)
        q_local = self.net(torch.cat([s, u, n, t], dim=-1))
        return torch.einsum("nij,nj->ni", R, q_local)


class TypedResidualBasis(nn.Module):
    """Manual exact vector contractions with scalar coefficient network."""

    def __init__(self):
        super().__init__()
        self.scalar_net = nn.Sequential(nn.Linear(8, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 3))

    def forward(self, s, u_world, n_world, t_world, R):
        u = torch.einsum("nij,nj->ni", R.transpose(-1, -2), u_world)
        n = torch.einsum("nij,nj->ni", R.transpose(-1, -2), n_world)
        t = torch.einsum("nij,nj->ni", R.transpose(-1, -2), t_world)
        un = (u * n).sum(-1, keepdim=True)
        ut = (u * t).sum(-1, keepdim=True)
        nt = (n * t).sum(-1, keepdim=True)
        scalar = torch.cat([s, (u * u).sum(-1, keepdim=True), un, ut * ut, nt * nt], dim=-1)
        coeff = self.scalar_net(scalar)
        b0 = u
        b1 = n * un
        b2 = t * ut
        q_local = coeff[:, 0:1] * b0 + coeff[:, 1:2] * b1 + coeff[:, 2:3] * b2
        return torch.einsum("nij,nj->ni", R, q_local)


MODEL_CLASSES = {"world_mlp": WorldMLP, "probe_local_mlp": LocalMLP, "typed_residual_basis": TypedResidualBasis}


def tensor_batch(data: Dict[str, np.ndarray], target: str = "label_q_world"):
    return (
        torch.from_numpy(data["s"]).to(DEVICE),
        torch.from_numpy(data["u_world"]).reshape(-1, 3).to(DEVICE),
        torch.from_numpy(data["n_world"]).repeat_interleave(data["u_world"].shape[1] if data["u_world"].ndim == 3 else 1, dim=0).to(DEVICE),
        torch.from_numpy(data["t_world"]).repeat_interleave(data["u_world"].shape[1] if data["u_world"].ndim == 3 else 1, dim=0).to(DEVICE),
        torch.from_numpy(data["R"]).repeat_interleave(data["u_world"].shape[1] if data["u_world"].ndim == 3 else 1, dim=0).to(DEVICE),
        torch.from_numpy(data[target]).reshape(-1, 3).to(DEVICE),
    )


def flatten_training_data(data: Dict[str, np.ndarray]):
    nrep = data["u_world"].shape[1] if data["u_world"].ndim == 3 else 1
    if nrep == 1:
        return tensor_batch(data)
    flat = {}
    for key in ("s", "u_world", "n_world", "t_world", "R", "label_q_world"):
        arr = data[key]
        flat[key] = np.repeat(arr[:, None, ...], nrep, axis=1) if key in ("s", "n_world", "t_world", "R") else arr
        flat[key] = flat[key].reshape((-1,) + flat[key].shape[2:])
    return tensor_batch(flat)


def fit_neural(model_name: str, data: Dict[str, np.ndarray], seed: int) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODEL_CLASSES[model_name]().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tensors = flatten_training_data(data)
    n = tensors[0].shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE].to(DEVICE)
            s, u, normal, tangent, R, y = [x[idx] for x in tensors]
            pred = model(s, u, normal, tangent, R)
            loss = ((pred - y) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def neural_predict(model: nn.Module, data: Dict[str, np.ndarray]) -> np.ndarray:
    nrep = data["u_world"].shape[1] if data["u_world"].ndim == 3 else 1
    flat = {}
    for key in ("s", "u_world", "n_world", "t_world", "R"):
        arr = data[key]
        if nrep > 1 and key in ("s", "n_world", "t_world", "R"):
            arr = np.repeat(arr[:, None, ...], nrep, axis=1)
        flat[key] = arr.reshape((-1,) + arr.shape[2:]) if nrep > 1 else arr
    s = torch.from_numpy(flat["s"]).to(DEVICE)
    u = torch.from_numpy(flat["u_world"]).to(DEVICE)
    n = torch.from_numpy(flat["n_world"]).to(DEVICE)
    t = torch.from_numpy(flat["t_world"]).to(DEVICE)
    R = torch.from_numpy(flat["R"]).to(DEVICE)
    pred = model(s, u, n, t, R).detach().cpu().numpy()
    return pred.reshape(data["u_world"].shape)


def global_linear_fit(data: Dict[str, np.ndarray]) -> np.ndarray:
    u = data["u_world"].reshape(-1, 3)
    q = data["label_q_world"].reshape(-1, 3)
    return np.linalg.lstsq(u, q, rcond=None)[0].astype(np.float32)


def global_linear_predict(B: np.ndarray, data: Dict[str, np.ndarray]) -> np.ndarray:
    return np.einsum("...i,ij->...j", data["u_world"], B)


def analytic_predict(data: Dict[str, np.ndarray]) -> np.ndarray:
    return np.zeros_like(data["u_world"], dtype=np.float32)


def response_metrics(pred_q: np.ndarray, data: Dict[str, np.ndarray]) -> Dict[str, float]:
    err = pred_q - data["q_world"]
    norms = np.linalg.norm(err, axis=-1)
    return {
        "n": int(norms.size),
        "rmse_vector_mm": float(np.sqrt(np.mean(err**2))),
        "mean_norm_mm": float(norms.mean()),
        "p95_norm_mm": float(np.quantile(norms, 0.95)),
        "max_norm_mm": float(norms.max()),
    }


def action_metrics(pred_q: np.ndarray, data: Dict[str, np.ndarray]) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    # Inputs have shape (contexts,candidates,3).
    R = data["R"]
    pred_total_local = np.einsum("nij,nkj->nki", R.transpose(0, 2, 1), data["u_world"] + pred_q)
    true_total_local = data["total_local"]
    pred_cost = np.linalg.norm(pred_total_local[..., [0, 2]] - TARGET_XZ_MM, axis=-1)
    true_cost_all = np.linalg.norm(true_total_local[..., [0, 2]] - TARGET_XZ_MM, axis=-1)
    chosen = np.argmin(pred_cost, axis=1)
    rows = np.arange(pred_q.shape[0])
    actual = true_cost_all[rows, chosen]
    oracle = np.min(true_cost_all, axis=1)
    regret = actual - oracle
    metrics = {
        "mean_actual_error_mm": float(actual.mean()),
        "p95_actual_error_mm": float(np.quantile(actual, 0.95)),
        "success_rate_at_0.50mm": float(np.mean(actual <= SUCCESS_THRESHOLD_MM)),
        "mean_oracle_error_mm": float(oracle.mean()),
        "oracle_success_rate": float(np.mean(oracle <= SUCCESS_THRESHOLD_MM)),
        "mean_regret_mm": float(regret.mean()),
        "p95_regret_mm": float(np.quantile(regret, 0.95)),
        "max_regret_mm": float(regret.max()),
    }
    return metrics, {"chosen": chosen, "actual": actual, "oracle": oracle, "regret": regret, "pred_cost": pred_cost}


def rls_predict_candidates(data: Dict[str, np.ndarray], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = data["s"].shape[0]
    probe_local = RLS_PROBE_MAG * np.eye(3, dtype=np.float32)[None, :, :].repeat(n, axis=0)
    probe_world = np.einsum("nij,nkj->nki", data["R"], probe_local)
    probe_n = np.repeat(data["n_world"][:, None, :], 3, axis=1)
    probe_t = np.repeat(data["t_world"][:, None, :], 3, axis=1)
    s = np.repeat(data["s"][:, None, :], 3, axis=1)
    q_true_local = plant_residual_local(s, probe_local, np.einsum("nij,nkj->nki", data["R"].transpose(0, 2, 1), probe_n), np.einsum("nij,nkj->nki", data["R"].transpose(0, 2, 1), probe_t))
    q_true_world = np.einsum("nij,nkj->nki", data["R"], q_true_local)
    noise = rng.normal(0.0, LABEL_NOISE_STD, size=q_true_world.shape).astype(np.float32)
    q_obs = q_true_world + noise
    pred = np.empty_like(data["u_world"])
    for i in range(n):
        B = np.linalg.lstsq(probe_world[i], q_obs[i], rcond=None)[0]
        pred[i] = data["u_world"][i] @ B
    return pred, q_obs


def predict_method(model_name: str, artifact, data: Dict[str, np.ndarray]) -> np.ndarray:
    if model_name == "analytic_zero_residual":
        return analytic_predict(data)
    if model_name == "global_fixed_linear":
        return global_linear_predict(artifact, data)
    if model_name in MODEL_CLASSES:
        return neural_predict(artifact, data)
    raise KeyError(model_name)


def aggregate(values: List[float]) -> Dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def geometry_checks(models: Dict[str, object], base: Dict[str, np.ndarray], seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    Q = random_rotations(rng, 1, "ood")[0]
    transformed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in base.items()}
    transformed["u_world"] = np.einsum("ij,nj->ni", Q, base["u_world"])
    transformed["n_world"] = np.einsum("ij,nj->ni", Q, base["n_world"])
    transformed["t_world"] = np.einsum("ij,nj->ni", Q, base["t_world"])
    transformed["R"] = np.einsum("ij,njk->nik", Q, base["R"])
    out = {}
    for name, artifact in models.items():
        p = predict_method(name, artifact, base)
        p_Q = predict_method(name, artifact, transformed)
        equiv = p_Q - np.einsum("ij,nj->ni", Q, p)
        flip = {k: v for k, v in base.items()}
        flip["t_world"] = -base["t_world"]
        p_flip = predict_method(name, artifact, flip)
        out[name] = {
            "so3_equivariance_mean_norm_mm": float(np.linalg.norm(equiv, axis=1).mean()),
            "so3_equivariance_p95_norm_mm": float(np.quantile(np.linalg.norm(equiv, axis=1), 0.95)),
            "t_signflip_mean_norm_mm": float(np.linalg.norm(p_flip - p, axis=1).mean()),
            "t_signflip_p95_norm_mm": float(np.quantile(np.linalg.norm(p_flip - p, axis=1), 0.95)),
        }
    return out


def gradient_check(model_name: str, model: nn.Module, data: Dict[str, np.ndarray], n_points: int = 16) -> Dict[str, float]:
    rng = np.random.default_rng(720960 + len(model_name))
    idxs = rng.choice(data["s"].shape[0], size=min(n_points, data["s"].shape[0]), replace=False)
    errors = []
    for i in idxs:
        s = torch.from_numpy(data["s"][i : i + 1]).to(DEVICE).requires_grad_(False)
        R = torch.from_numpy(data["R"][i : i + 1]).to(DEVICE).requires_grad_(False)
        n_world = torch.from_numpy(data["n_world"][i : i + 1]).to(DEVICE).requires_grad_(False)
        t_world = torch.from_numpy(data["t_world"][i : i + 1]).to(DEVICE).requires_grad_(False)
        u0 = torch.from_numpy(data["u_world"][i : i + 1]).to(DEVICE).requires_grad_(True)

        def fn(u):
            return model(s, u, n_world, t_world, R)[0]

        jac = torch.autograd.functional.jacobian(fn, u0).detach().cpu().numpy()
        u_np = data["u_world"][i].copy()
        fd = np.zeros((3, 3), dtype=np.float32)
        for j in range(3):
            plus = u_np.copy(); minus = u_np.copy()
            plus[j] += FD_STEP; minus[j] -= FD_STEP
            plus_t = torch.from_numpy(plus[None]).to(DEVICE)
            minus_t = torch.from_numpy(minus[None]).to(DEVICE)
            fd[:, j] = ((fn(plus_t) - fn(minus_t)) / (2 * FD_STEP)).detach().cpu().numpy()
        errors.append(float(np.linalg.norm(jac - fd)))
    return {"n": len(errors), "mean_fro_error": float(np.mean(errors)), "max_fro_error": float(np.max(errors))}


def save_raw_arrays(arrays: Dict[str, np.ndarray]) -> None:
    cleaned = {k: np.asarray(v) for k, v in arrays.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(ROOT / "RAW_RESULTS.npz", **cleaned)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> Dict[str, object]:
    print(f"device={DEVICE}")
    all_runs = []
    raw: Dict[str, np.ndarray] = {}
    global_artifacts = {}

    # Fixed independent test sets and their raw evidence.
    test_ood = make_samples(512, TEST_SEED_OOD, "ood", None, negative=False)
    test_id = make_samples(512, TEST_SEED_ID, "id", None, negative=False)
    candidate_ood = make_samples(512, TEST_SEED_OOD + 100, "ood", CANDIDATES, negative=False)
    negative_ood = make_samples(512, TEST_SEED_OOD + 200, "ood", CANDIDATES, negative=True)
    for prefix, data in (("test_ood", test_ood), ("test_id", test_id), ("candidate_ood", candidate_ood), ("negative_ood", negative_ood)):
        for key in ("s", "R", "n_local", "t_local", "n_world", "t_world", "u_local", "u_world", "q_local", "q_world", "total_local", "total_world", "label_q_world"):
            raw[f"{prefix}_{key}"] = data[key]

    for n in TRAIN_SIZES:
        for seed in TRAIN_SEEDS:
            train = make_samples(n, 100000 + n * 10 + seed, "train", None, negative=False)
            for key in ("s", "R", "n_local", "t_local", "n_world", "t_world", "u_local", "u_world", "q_world", "label_q_world"):
                raw[f"train_n{n}_seed{seed}_{key}"] = train[key]
            global_B = global_linear_fit(train)
            global_artifacts[(n, seed)] = global_B
            model_names = ["analytic_zero_residual", "global_fixed_linear"] + list(MODEL_CLASSES.keys())
            artifacts = {"analytic_zero_residual": None, "global_fixed_linear": global_B}
            train_loss = {"analytic_zero_residual": float(np.mean(train["q_world"] ** 2)), "global_fixed_linear": float(np.mean((global_predict := global_linear_predict(global_B, train)) - train["q_world"]) ** 2)}
            for model_name in MODEL_CLASSES:
                model = fit_neural(model_name, train, seed)
                path = WEIGHT_DIR / f"{model_name}_n{n}_seed{seed}.pt"
                torch.save({"model_name": model_name, "n": n, "seed": seed, "state_dict": model.state_dict()}, path)
                artifacts[model_name] = model
                with torch.no_grad():
                    train_pred = neural_predict(model, train)
                train_loss[model_name] = float(np.mean((train_pred - train["q_world"]) ** 2))

            for model_name, artifact in artifacts.items():
                pred_id = predict_method(model_name, artifact, test_id)
                pred_ood = predict_method(model_name, artifact, test_ood)
                pred_cand = predict_method(model_name, artifact, candidate_ood)
                pred_neg = predict_method(model_name, artifact, negative_ood)
                action, _ = action_metrics(pred_cand, candidate_ood)
                neg_action, _ = action_metrics(pred_neg, negative_ood)
                run = {
                    "n": n,
                    "seed": seed,
                    "model": model_name,
                    "train_mse_mm2": train_loss[model_name],
                    "response_id": response_metrics(pred_id, test_id),
                    "response_ood": response_metrics(pred_ood, test_ood),
                    "action_ood": action,
                    "negative_response": response_metrics(pred_neg, negative_ood),
                    "negative_action": neg_action,
                }
                all_runs.append(run)

    # Strong online RLS baseline uses test-context probes, not learned free data.
    rls_pred_query, rls_probe_labels = rls_predict_candidates(test_ood, TEST_SEED_OOD + 300)
    rls_pred_cand, rls_probe_labels_cand = rls_predict_candidates(candidate_ood, TEST_SEED_OOD + 301)
    rls_action, rls_action_arrays = action_metrics(rls_pred_cand, candidate_ood)
    rls_response = response_metrics(rls_pred_query, test_ood)
    rls_record = {
        "model": "online_RLS_3probe",
        "probe_count": RLS_PROBE_COUNT,
        "probe_magnitude_mm": RLS_PROBE_MAG,
        "response_ood": rls_response,
        "action_ood": rls_action,
        "negative": "not applicable as a learned positive-plant test; RLS is a per-context adaptive baseline",
    }
    raw["rls_probe_labels_query"] = rls_probe_labels
    raw["rls_probe_labels_candidate"] = rls_probe_labels_cand
    raw["rls_pred_query"] = rls_pred_query
    raw["rls_pred_candidate"] = rls_pred_cand
    raw["rls_chosen_candidate"] = rls_action_arrays["chosen"]

    # Geometry checks use the largest model with seed 101 plus fixed baselines.
    selected = {}
    n = 2048; seed = 101
    train = make_samples(n, 100000 + n * 10 + seed, "train", None, negative=False)
    selected["analytic_zero_residual"] = None
    selected["global_fixed_linear"] = global_artifacts[(n, seed)]
    selected_models = {}
    for model_name in MODEL_CLASSES:
        model = MODEL_CLASSES[model_name]().to(DEVICE)
        payload = torch.load(WEIGHT_DIR / f"{model_name}_n{n}_seed{seed}.pt", map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["state_dict"]); model.eval()
        selected[model_name] = model
        selected_models[model_name] = model
    geometry = geometry_checks(selected, test_ood, TEST_SEED_OOD + 400)
    gradients = {name: gradient_check(name, model, test_ood) for name, model in selected_models.items()}

    # Save selected model outputs for geometry and negative controls.
    for name, artifact in selected.items():
        raw[f"selected_ood_pred_{name}"] = predict_method(name, artifact, test_ood)
        raw[f"selected_candidate_pred_{name}"] = predict_method(name, artifact, candidate_ood)
        raw[f"selected_negative_pred_{name}"] = predict_method(name, artifact, negative_ood)

    def summarize(group_key: str):
        groups = {}
        for run in all_runs:
            groups.setdefault((run["model"], run["n"]), []).append(run)
        out = {}
        for (model, size), rows in groups.items():
            out[f"{model}_n{size}"] = {
                "response_ood_rmse_mm": aggregate([x["response_ood"]["rmse_vector_mm"] for x in rows]),
                "response_id_rmse_mm": aggregate([x["response_id"]["rmse_vector_mm"] for x in rows]),
                "action_success": aggregate([x["action_ood"]["success_rate_at_0.50mm"] for x in rows]),
                "action_regret_mm": aggregate([x["action_ood"]["mean_regret_mm"] for x in rows]),
                "negative_response_rmse_mm": aggregate([x["negative_response"]["rmse_vector_mm"] for x in rows]),
            }
        return out

    result = {
        "status": "synthetic_conditioned_response_geometry_benchmark",
        "protocol": {
            "train_sizes": list(TRAIN_SIZES),
            "train_seeds": list(TRAIN_SEEDS),
            "test_seed_ood": TEST_SEED_OOD,
            "test_seed_id": TEST_SEED_ID,
            "epochs": EPOCHS,
            "hidden": HIDDEN,
            "label_noise_std_mm": LABEL_NOISE_STD,
            "candidate_count": CANDIDATES,
            "target_xz_mm": TARGET_XZ_MM.tolist(),
            "success_threshold_mm": SUCCESS_THRESHOLD_MM,
            "device": str(DEVICE),
        },
        "runs": all_runs,
        "aggregate": summarize("response_ood"),
        "online_rls": rls_record,
        "geometry_checks_selected_n2048_seed101": geometry,
        "gradient_checks_selected_n2048_seed101": gradients,
        "interpretation": {
            "scope": "Synthetic contact-response mechanism only; no anatomical or real-ultrasound claim.",
            "typed_basis": "Manual exact vector contractions; e3nn package unavailable, so this is not reported as a completed e3nn library implementation.",
            "baseline_rule": "If local MLP matches typed basis, retain the simpler local model; if online RLS matches or wins, report that honestly.",
            "oracle": "Candidate-set oracle uses the declared plant only as an evaluation ceiling.",
        },
    }
    save_raw_arrays(raw)
    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (ROOT / "REPORT.txt").write_text(make_report(result), encoding="utf-8")
    return result


def fmt(x):
    return f"{float(x):.5g}"


def make_report(result):
    lines = [
        "Synthetic conditioned-response geometry benchmark",
        "=================================================",
        "Scope: synthetic contact plant only; not real ultrasound or full A/B validation.",
        f"Device={result['protocol']['device']}; test OOD seed={result['protocol']['test_seed_ood']}; ID seed={result['protocol']['test_seed_id']}",
        "",
        "Selected n2048/seed101 geometry checks",
        "--------------------------------------",
        "model | SO3 equiv mean/p95 | t-signflip mean/p95 | FD gradient mean/max",
    ]
    for name, g in result["geometry_checks_selected_n2048_seed101"].items():
        grad = result["gradient_checks_selected_n2048_seed101"].get(name, {})
        lines.append(
            f"{name} | {fmt(g['so3_equivariance_mean_norm_mm'])}/{fmt(g['so3_equivariance_p95_norm_mm'])} | "
            f"{fmt(g['t_signflip_mean_norm_mm'])}/{fmt(g['t_signflip_p95_norm_mm'])} | "
            f"{fmt(grad.get('mean_fro_error', float('nan')))}/{fmt(grad.get('max_fro_error', float('nan')))}"
        )
    lines.extend(["", "Data efficiency aggregate (OOD response RMSE / action success / regret)", "---------------------------------------------------------------------"])
    for key, row in result["aggregate"].items():
        lines.append(
            f"{key}: RMSE={fmt(row['response_ood_rmse_mm']['mean'])}±{fmt(row['response_ood_rmse_mm']['std'])} mm, "
            f"success={fmt(row['action_success']['mean'])}±{fmt(row['action_success']['std'])}, "
            f"regret={fmt(row['action_regret_mm']['mean'])}±{fmt(row['action_regret_mm']['std'])} mm, "
            f"negative-RMSE={fmt(row['negative_response_rmse_mm']['mean'])} mm"
        )
    rls = result["online_rls"]
    lines.extend(
        [
            "",
            f"online_RLS_3probe: response RMSE={fmt(rls['response_ood']['rmse_vector_mm'])} mm, "
            f"success={fmt(rls['action_ood']['success_rate_at_0.50mm'])}, "
            f"regret={fmt(rls['action_ood']['mean_regret_mm'])} mm, explicit probes={rls['probe_count']}",
            "",
            "Interpretation",
            "--------------",
            "The typed basis is a geometry-constrained synthetic mechanism model; coordinate and sample-efficiency results must be read against the probe-local MLP and online three-probe RLS. Matching local/typed performance is a valid negative result for an e3nn-specific claim. None of these results establishes anatomical contact truth or complete A/B performance.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_manifest():
    names = ["PROTOCOL.txt", "run_benchmark.py", "test_benchmark.py", "RAW_RESULTS.npz", "RESULTS.json", "REPORT.txt"]
    manifest = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(DEVICE),
        "files_sha256": {name: sha256(ROOT / name) for name in names},
        "weights_sha256": {p.name: sha256(p) for p in sorted(WEIGHT_DIR.glob("*.pt"))},
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
    write_manifest()
