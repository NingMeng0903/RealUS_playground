"""Masked offline learners for the knee recheck experiment.

The action dimension is a small, state-dependent set of legal ProDMP
continuations.  Every learner in this module consumes the same transition
arrays and the same boolean action mask.  Invalid actions are removed from
Bellman targets, conservative penalties, and the discrete SAC policy.

The public interface is intentionally small::

    model = train(data, "fqi", seed=7)
    value = scores(model, features, mask)
    save(model, path)
    model = load(path)

``features`` and ``next_features`` have shape ``(N, A, D)``.  ``mask`` and
``next_mask`` have shape ``(N, A)``.  A terminal row may have no valid next
action.  ``action`` contains the selected legal action for each row.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


METHODS = ("fqi", "cql", "sac")
_METHOD_ALIASES = {
    "fitted_q": "fqi",
    "fitted_q_iteration": "fqi",
    "discrete_cql": "cql",
    "offline_cql": "cql",
    "discrete_sac": "sac",
    "offline_sac": "sac",
}
VERSION = 1


def _canonical_method(method: str) -> str:
    canonical = _METHOD_ALIASES.get(str(method).lower(), str(method).lower())
    if canonical not in METHODS:
        raise ValueError(f"method must be one of {METHODS} or their named aliases")
    return canonical


def _device(value: str | torch.device | None) -> torch.device:
    if value is None or str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _check_data(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {
        "features", "mask", "action", "reward", "next_features", "next_mask", "done"
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"missing transition fields: {sorted(missing)}")
    features = np.asarray(data["features"], dtype=np.float32)
    next_features = np.asarray(data["next_features"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=bool)
    next_mask = np.asarray(data["next_mask"], dtype=bool)
    action = np.asarray(data["action"], dtype=np.int64).reshape(-1)
    reward = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
    done = np.asarray(data["done"], dtype=bool).reshape(-1)
    if features.ndim != 3 or next_features.ndim != 3:
        raise ValueError("features and next_features must be (N, A, D)")
    n, actions, dim = features.shape
    if next_features.shape != (n, actions, dim):
        raise ValueError("next_features must have the same (N, A, D) shape")
    if mask.shape != (n, actions) or next_mask.shape != (n, actions):
        raise ValueError("mask and next_mask must be (N, A)")
    if any(array.shape[0] != n for array in (action, reward, done)):
        raise ValueError("transition vectors must have N rows")
    if n == 0 or actions == 0 or dim == 0:
        raise ValueError("transition arrays cannot be empty")
    if np.any(action < 0) or np.any(action >= actions):
        raise ValueError("action contains an out-of-range index")
    if not np.all(mask[np.arange(n), action]):
        raise ValueError("action selects an invalid masked candidate")
    if not np.isfinite(features).all() or not np.isfinite(next_features).all():
        raise ValueError("features contain a non-finite value")
    if not np.isfinite(reward).all():
        raise ValueError("reward contains a non-finite value")
    return {
        "features": np.ascontiguousarray(features),
        "mask": np.ascontiguousarray(mask),
        "action": np.ascontiguousarray(action),
        "reward": np.ascontiguousarray(reward),
        "next_features": np.ascontiguousarray(next_features),
        "next_mask": np.ascontiguousarray(next_mask),
        "done": np.ascontiguousarray(done),
    }


def _to_tensors(data: Mapping[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    checked = _check_data(data)
    return {
        "features": torch.as_tensor(checked["features"], device=device),
        "mask": torch.as_tensor(checked["mask"], device=device),
        "action": torch.as_tensor(checked["action"], device=device),
        "reward": torch.as_tensor(checked["reward"], device=device),
        "next_features": torch.as_tensor(checked["next_features"], device=device),
        "next_mask": torch.as_tensor(checked["next_mask"], device=device),
        "done": torch.as_tensor(checked["done"], device=device),
    }


class CandidateMLP(nn.Module):
    """A shared candidate scorer, preserving permutation symmetry over actions."""

    def __init__(self, dim: int, hidden: int = 96, depth: int = 2, output: int = 1):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        layers: list[nn.Module] = []
        input_dim = int(dim)
        for _ in range(int(depth)):
            layers.extend((nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, output))
        self.net = nn.Sequential(*layers)
        # Zero output is a stable neutral initialization for finite-horizon
        # fitted targets and makes the initial discrete policy uniform.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("candidate features must be (N, A, D)")
        n, actions, dim = features.shape
        return self.net(features.reshape(n * actions, dim)).reshape(n, actions, -1)[..., 0]


def _masked_max(values: Tensor, mask: Tensor) -> Tensor:
    """Maximum over legal actions, with zero for a terminal empty set."""

    legal = mask.to(dtype=torch.bool)
    safe = values.masked_fill(~legal, -torch.inf)
    result = safe.max(dim=1).values
    return torch.where(legal.any(dim=1), result, torch.zeros_like(result))


def _masked_log_softmax(logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    legal = mask.to(dtype=torch.bool)
    safe = logits.masked_fill(~legal, -torch.inf)
    log_prob = torch.log_softmax(safe, dim=1)
    prob = torch.exp(log_prob).masked_fill(~legal, 0.0)
    # Rows with no legal action do not occur for a live state.  Keeping zero
    # probabilities there avoids NaNs in a terminal policy target.
    empty = ~legal.any(dim=1)
    if empty.any():
        log_prob = torch.where(empty[:, None], torch.zeros_like(log_prob), log_prob)
    return log_prob, prob


def _taken(values: Tensor, action: Tensor) -> Tensor:
    return values.gather(1, action[:, None]).squeeze(1)


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if batch_size >= n:
        return np.arange(n)
    return rng.integers(0, n, size=batch_size, dtype=np.int64)


def _hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


@dataclass
class Learner:
    method: str
    q1: CandidateMLP | None
    q2: CandidateMLP | None
    actor: CandidateMLP | None
    metadata: dict[str, Any]
    device: torch.device

    def action_scores(self, features: np.ndarray | Tensor, mask: np.ndarray | Tensor) -> np.ndarray:
        """Return policy/action scores with illegal entries set to ``-inf``."""

        if isinstance(features, np.ndarray):
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        else:
            x = features.to(self.device, dtype=torch.float32)
        if isinstance(mask, np.ndarray):
            m = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        else:
            m = mask.to(self.device, dtype=torch.bool)
        if x.ndim != 3 or m.shape != x.shape[:2]:
            raise ValueError("features/mask must have shapes (N,A,D)/(N,A)")
        with torch.no_grad():
            if self.method == "sac":
                assert self.actor is not None
                result = self.actor(x)
            else:
                assert self.q1 is not None
                result = self.q1(x)
            result = result.masked_fill(~m, -torch.inf)
        return result.detach().cpu().numpy()

    def state_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.q1 is not None:
            result["q1"] = self.q1.state_dict()
        if self.q2 is not None:
            result["q2"] = self.q2.state_dict()
        if self.actor is not None:
            result["actor"] = self.actor.state_dict()
        return result


def _new_model(method: str, *, dim: int, hidden: int, depth: int, device: torch.device,
               alpha: float = .10) -> Learner:
    method = _canonical_method(method)
    q1 = CandidateMLP(dim, hidden, depth).to(device)
    if method == "fqi":
        return Learner(method, q1, None, None,
                       {"dim": dim, "hidden": hidden, "depth": depth, "alpha": alpha}, device)
    q2 = CandidateMLP(dim, hidden, depth).to(device)
    actor = CandidateMLP(dim, hidden, depth).to(device) if method == "sac" else None
    return Learner(method, q1, q2, actor,
                   {"dim": dim, "hidden": hidden, "depth": depth, "alpha": alpha}, device)


def _fqi(model: Learner, t: dict[str, Tensor], *, steps: int, gamma: float, seed: int,
         batch_size: int) -> None:
    assert model.q1 is not None
    q = model.q1
    n = int(t["action"].shape[0])
    updates = max(1, int(steps))
    # Five frozen-target blocks are enough to retain fitted-iteration semantics
    # while allowing a caller to use a smaller smoke-test budget.
    blocks = min(5, updates)
    per_block = max(1, updates // blocks)
    rng = np.random.default_rng(seed)
    for block in range(blocks):
        target = copy.deepcopy(q).to(model.device).eval()
        with torch.no_grad():
            next_values = _masked_max(target(t["next_features"]), t["next_mask"])
            y = t["reward"] + float(gamma) * (~t["done"]).float() * next_values
        optimizer = torch.optim.Adam(q.parameters(), lr=3.e-3, weight_decay=1.e-5)
        for _ in range(per_block):
            idx = torch.as_tensor(_batch_indices(n, batch_size, rng), device=model.device)
            prediction = _taken(q(t["features"][idx]), t["action"][idx])
            loss = F.smooth_l1_loss(prediction, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 10.)
            optimizer.step()


def _cql(model: Learner, t: dict[str, Tensor], *, steps: int, gamma: float, seed: int,
         batch_size: int, cql_alpha: float) -> None:
    assert model.q1 is not None and model.q2 is not None
    q1, q2 = model.q1, model.q2
    target1, target2 = copy.deepcopy(q1).to(model.device).eval(), copy.deepcopy(q2).to(model.device).eval()
    optim = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-3, weight_decay=1e-5)
    n = int(t["action"].shape[0])
    rng = np.random.default_rng(seed)
    period = max(1, int(steps) // 5)
    for update in range(max(1, int(steps))):
        idx = torch.as_tensor(_batch_indices(n, batch_size, rng), device=model.device)
        with torch.no_grad():
            next_value = _masked_max(torch.minimum(target1(t["next_features"][idx]),
                                                   target2(t["next_features"][idx])), t["next_mask"][idx])
            y = t["reward"][idx] + float(gamma) * (~t["done"][idx]).float() * next_value
        values1 = q1(t["features"][idx])
        values2 = q2(t["features"][idx])
        taken1, taken2 = _taken(values1, t["action"][idx]), _taken(values2, t["action"][idx])
        td = .5 * (F.smooth_l1_loss(taken1, y) + F.smooth_l1_loss(taken2, y))
        safe1 = values1.masked_fill(~t["mask"][idx], -torch.inf)
        safe2 = values2.masked_fill(~t["mask"][idx], -torch.inf)
        cql = .5 * ((torch.logsumexp(safe1, dim=1) - taken1).mean()
                    + (torch.logsumexp(safe2, dim=1) - taken2).mean())
        loss = td + float(cql_alpha) * cql
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10.)
        optim.step()
        if (update + 1) % period == 0:
            _hard_update(target1, q1)
            _hard_update(target2, q2)


def _sac(model: Learner, t: dict[str, Tensor], *, steps: int, gamma: float, seed: int,
         batch_size: int, alpha: float) -> None:
    assert model.q1 is not None and model.q2 is not None and model.actor is not None
    q1, q2, actor = model.q1, model.q2, model.actor
    target1, target2 = copy.deepcopy(q1).to(model.device).eval(), copy.deepcopy(q2).to(model.device).eval()
    critic_optim = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-3, weight_decay=1e-5)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-3, weight_decay=1e-5)
    n = int(t["action"].shape[0])
    rng = np.random.default_rng(seed)
    period = max(1, int(steps) // 5)
    temperature = float(alpha)
    for update in range(max(1, int(steps))):
        idx = torch.as_tensor(_batch_indices(n, batch_size, rng), device=model.device)
        with torch.no_grad():
            next_logits = actor(t["next_features"][idx])
            next_logp, next_prob = _masked_log_softmax(next_logits, t["next_mask"][idx])
            next_q = torch.minimum(target1(t["next_features"][idx]), target2(t["next_features"][idx]))
            next_value = (next_prob * (next_q - temperature * next_logp).masked_fill(~t["next_mask"][idx], 0.)).sum(1)
            y = t["reward"][idx] + float(gamma) * (~t["done"][idx]).float() * next_value
        values1 = q1(t["features"][idx])
        values2 = q2(t["features"][idx])
        critic_loss = .5 * (F.smooth_l1_loss(_taken(values1, t["action"][idx]), y)
                            + F.smooth_l1_loss(_taken(values2, t["action"][idx]), y))
        critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10.)
        critic_optim.step()

        logits = actor(t["features"][idx])
        logp, prob = _masked_log_softmax(logits, t["mask"][idx])
        with torch.no_grad():
            policy_q = torch.minimum(q1(t["features"][idx]), q2(t["features"][idx]))
        actor_loss = (prob * (temperature * logp - policy_q).masked_fill(~t["mask"][idx], 0.)).sum(1).mean()
        actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.)
        actor_optim.step()
        if (update + 1) % period == 0:
            _hard_update(target1, q1)
            _hard_update(target2, q2)


def train(data: Mapping[str, np.ndarray], method: str, *, seed: int = 0,
          steps: int = 1000, device: str | torch.device | None = "auto",
          gamma: float = 1.0, hidden: int = 96, depth: int = 2,
          batch_size: int = 256, cql_alpha: float = .50,
          alpha: float = .10) -> Learner:
    """Train one masked learner on a fixed transition table.

    ``steps`` is the number of gradient updates for CQL/SAC and the total
    fitted-iteration updates for FQI, distributed over five frozen-target
    blocks.  No evaluation labels or hidden simulator state are accepted by
    this interface.
    """

    checked = _check_data(data)
    method = _canonical_method(method)
    if not np.isfinite(gamma) or gamma < 0.:
        raise ValueError("gamma must be finite and nonnegative")
    target_device = _device(device)
    torch.manual_seed(int(seed))
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    tensors = _to_tensors(checked, target_device)
    model = _new_model(method, dim=checked["features"].shape[-1], hidden=hidden,
                       depth=depth, device=target_device, alpha=alpha)
    model.metadata.update({
        "version": VERSION, "seed": int(seed), "steps": int(steps), "gamma": float(gamma),
        "batch_size": int(batch_size), "cql_alpha": float(cql_alpha),
    })
    if method == "fqi":
        _fqi(model, tensors, steps=steps, gamma=gamma, seed=seed, batch_size=batch_size)
    elif method == "cql":
        _cql(model, tensors, steps=steps, gamma=gamma, seed=seed,
             batch_size=batch_size, cql_alpha=cql_alpha)
    else:
        _sac(model, tensors, steps=steps, gamma=gamma, seed=seed,
             batch_size=batch_size, alpha=alpha)
    return model


def scores(model: Learner, features: np.ndarray | Tensor, mask: np.ndarray | Tensor) -> np.ndarray:
    """Score every candidate for action selection, with invalid entries masked."""

    if not isinstance(model, Learner):
        raise TypeError("model must be returned by learners.train or learners.load")
    return model.action_scores(features, mask)


def save(model: Learner, path: str | Path) -> None:
    if not isinstance(model, Learner):
        raise TypeError("model must be returned by learners.train")
    torch.save({"version": VERSION, "method": model.method, "metadata": model.metadata,
                "state": model.state_dict()}, Path(path))


def load(path: str | Path, device: str | torch.device | None = "cpu") -> Learner:
    target_device = _device(device)
    checkpoint = torch.load(Path(path), map_location=target_device, weights_only=False)
    if int(checkpoint.get("version", -1)) != VERSION:
        raise ValueError("unsupported learner checkpoint version")
    metadata = dict(checkpoint["metadata"])
    model = _new_model(checkpoint["method"], dim=int(metadata["dim"]),
                       hidden=int(metadata["hidden"]), depth=int(metadata["depth"]),
                       device=target_device, alpha=float(metadata.get("alpha", .1)))
    model.metadata = metadata
    state = checkpoint["state"]
    for name, module in (("q1", model.q1), ("q2", model.q2), ("actor", model.actor)):
        if module is not None:
            if name not in state:
                raise ValueError(f"checkpoint lacks {name}")
            module.load_state_dict(state[name])
    return model


def _smoke() -> None:
    rng = np.random.default_rng(13)
    n, actions, dim = 48, 4, 7
    features = rng.normal(size=(n, actions, dim)).astype(np.float32)
    mask = np.ones((n, actions), dtype=bool)
    mask[::3, 2] = False
    action = np.array([int(rng.choice(np.flatnonzero(row))) for row in mask], dtype=np.int64)
    next_features = rng.normal(size=features.shape).astype(np.float32)
    next_mask = mask.copy()
    next_mask[::5] = False
    reward = (features[np.arange(n), action, 0] > 0.).astype(np.float32)
    done = np.zeros(n, dtype=bool)
    done[::7] = True
    data = dict(features=features, mask=mask, action=action, reward=reward,
                next_features=next_features, next_mask=next_mask, done=done)
    for method in METHODS:
        model = train(data, method, seed=19, steps=25, device="cpu", batch_size=16)
        values = scores(model, features[:5], mask[:5])
        assert values.shape == (5, actions) and np.isneginf(values[0, 2])
        path = Path(f"/tmp/knee_{method}_learner.pt")
        save(model, path)
        restored = load(path)
        assert np.allclose(values, scores(restored, features[:5], mask[:5]), equal_nan=True)
        path.unlink()
        print(method, "smoke ok", float(np.nanmax(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
    else:
        parser.error("use --smoke for the standalone check; import train/scores for experiments")


if __name__ == "__main__":
    main()
