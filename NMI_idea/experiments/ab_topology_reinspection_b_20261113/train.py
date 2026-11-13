"""Cheap online PPO and fixed-replay CQL training for the topology screen.

This module only depends on the public environment contract used by the
topology experiment.  The environment supplies public feature vectors and
legal-action masks; private labels are read only from ``outcomes()`` after an
episode has terminated for evaluation.  The learner implementation lives in
``learner.py``.

The training routines intentionally use small, complete episode batches.  PPO
warms each recurrent state with the five public prescan frames, then excludes
that prefix and padded post-terminal frames from the loss.  CQL first freezes a
mixed greedy/random logged replay and never changes that source while fitting.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

try:  # Package import when called as ``work.topology_b.train``.
    from .learner import ActorCritic, QNetwork, cql_update, ppo_update
except ImportError:  # Direct script/import from the topology directory.
    from learner import ActorCritic, QNetwork, cql_update, ppo_update


__all__ = ["train_ppo", "train_cql", "neural_actions", "evaluate"]


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The environment and these small MLPs are CPU-friendly.  Avoid spawning a
    # large worker pool when this file is used for a smoke run on a workstation.
    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except RuntimeError:
        pass


def _device(requested: str | torch.device) -> torch.device:
    chosen = torch.device(requested)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return chosen


def _as_batch_features(value: Any, *, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(np.asarray(value), device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _as_batch_mask(value: Any, *, device: torch.device, action_count: int) -> Tensor:
    tensor = torch.as_tensor(np.asarray(value), device=device, dtype=torch.bool)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[-1] != action_count:
        raise ValueError(f"expected legal mask [B,{action_count}], got {tuple(tensor.shape)}")
    return tensor


def _prefix(episode: Any, *, device: torch.device, action_count: int, feature_dim: int) -> Tuple[Tensor, Tensor]:
    if not hasattr(episode, "prefix_features"):
        raise AttributeError("episode must provide prefix_features()")
    features = torch.as_tensor(np.asarray(episode.prefix_features()), device=device, dtype=torch.float32)
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3 or features.shape[-1] != feature_dim:
        raise ValueError(f"expected prefix [B,T,{feature_dim}], got {tuple(features.shape)}")
    # Prescan frames are public warm-up observations rather than actions.  A
    # fully legal mask keeps Categorical well-defined while train_mask excludes
    # these frames from every PPO loss.
    mask = torch.ones(
        features.shape[0], features.shape[1], action_count,
        device=device,
        dtype=torch.bool,
    )
    return features, mask


def _current(
    episode: Any,
    *,
    device: torch.device,
    action_count: int,
    feature_dim: int,
) -> Tuple[Tensor, Tensor]:
    features, legal = episode.features()
    features = _as_batch_features(features, device=device)
    legal = _as_batch_mask(legal, device=device, action_count=action_count)
    if features.shape[0] != legal.shape[0] or features.shape[-1] != feature_dim:
        raise ValueError(
            f"episode features/mask mismatch: {tuple(features.shape)}, {tuple(legal.shape)}"
        )
    return features, legal


def _done(episode: Any, batch_size: int) -> np.ndarray:
    value = np.asarray(getattr(episode, "done"), dtype=bool).reshape(-1)
    if value.size == 1 and batch_size != 1:
        value = np.repeat(value, batch_size)
    if value.size != batch_size:
        raise ValueError(f"episode.done has {value.size} entries, expected {batch_size}")
    return value.copy()


def _numpy_vector(value: Any, size: int, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.size == 1 and size != 1:
        array = np.repeat(array, size)
    if array.size != size:
        raise ValueError(f"expected {size} values, got {array.size}")
    return array


def _load_init(model: torch.nn.Module, init: Any, device: torch.device) -> bool:
    """Load a state dict/path when supplied; unsupported values are ignored."""

    if init is None:
        return False
    state: Any = init
    try:
        if isinstance(init, torch.nn.Module):
            state = init.state_dict()
        elif isinstance(init, (str, Path)):
            state = torch.load(Path(init), map_location=device, weights_only=False)
        if isinstance(state, dict):
            for key in ("model", "state_dict", "policy"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
            if all(isinstance(k, str) for k in state):
                model.load_state_dict(state, strict=False)
                return True
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return False


def _safe_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _sample_masked(logits: Tensor, legal: Tensor, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
    # learner.ActorCritic already applies the mask; retaining this helper keeps
    # action sampling in one place and handles a caller-supplied model too.
    safe = legal.to(dtype=torch.bool)
    any_legal = safe.any(dim=-1, keepdim=True)
    safe = torch.where(any_legal, safe, torch.ones_like(safe))
    logits = logits.masked_fill(~safe, torch.finfo(logits.dtype).min)
    distribution = torch.distributions.Categorical(logits=logits)
    if generator is None:
        action = distribution.sample()
    else:
        # Categorical.sample did not accept a generator on older torch builds;
        # use the global seeded stream for compatibility.
        action = distribution.sample()
    return action, distribution.log_prob(action)


def _greedy_masked(logits: Tensor, legal: Tensor) -> Tensor:
    safe = legal.to(dtype=torch.bool)
    any_legal = safe.any(dim=-1, keepdim=True)
    safe = torch.where(any_legal, safe, torch.ones_like(safe))
    return logits.masked_fill(~safe, torch.finfo(logits.dtype).min).argmax(dim=-1)


def _random_legal(legal: np.ndarray, rng: np.random.Generator, abstain: int) -> np.ndarray:
    legal = np.asarray(legal, dtype=bool)
    out = np.full(legal.shape[0], int(abstain), dtype=np.int64)
    for row, allowed in enumerate(legal):
        choices = np.flatnonzero(allowed)
        if choices.size:
            out[row] = int(rng.choice(choices))
    return out


def _gae(
    rewards: Tensor,
    values: Tensor,
    done: Tensor,
    train_mask: Tensor,
    bootstrap: Tensor,
    *,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """Compute masked GAE; terminal rewards use gamma=1 by default."""

    bsz, time_steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    next_value = bootstrap
    next_advantage = torch.zeros(bsz, device=rewards.device, dtype=rewards.dtype)
    for t in range(time_steps - 1, -1, -1):
        valid = train_mask[:, t]
        nonterminal = (~done[:, t]) & valid
        delta = rewards[:, t] + float(gamma) * nonterminal.to(rewards.dtype) * next_value - values[:, t]
        gae = delta + float(gamma * lam) * nonterminal.to(rewards.dtype) * next_advantage
        advantages[:, t] = torch.where(valid, gae, torch.zeros_like(gae))
        returns[:, t] = torch.where(valid, advantages[:, t] + values[:, t], torch.zeros_like(values[:, t]))
        next_value = torch.where(valid, values[:, t], torch.zeros_like(next_value))
        next_advantage = torch.where(valid, gae, torch.zeros_like(gae))
    return returns, advantages


def _ppo_rollout(
    model: ActorCritic,
    env: Any,
    indices: np.ndarray,
    max_steps: int,
    device: torch.device,
) -> Tuple[Dict[str, Tensor], int]:
    """Collect one complete batched rollout from public environment state."""

    action_count = model.action_count
    feature_dim = model.state_dim
    episode = env.reset(np.asarray(indices, dtype=np.int64))
    prefix, prefix_mask = _prefix(
        episode, device=device, action_count=action_count, feature_dim=feature_dim
    )
    batch_size = prefix.shape[0]
    if batch_size != len(indices):
        raise ValueError("env.reset returned a different batch size")
    done_np = _done(episode, batch_size)
    hidden: Optional[Tensor]
    model.eval()
    with torch.no_grad():
        _, _, hidden = model.sequence(prefix, prefix_mask, h=None)

    feature_rows: List[Tensor] = []
    mask_rows: List[Tensor] = []
    action_rows: List[Tensor] = []
    old_logp_rows: List[Tensor] = []
    value_rows: List[Tensor] = []
    reward_rows: List[Tensor] = []
    done_rows: List[Tensor] = []
    train_rows: List[Tensor] = []

    for _ in range(max(0, int(max_steps))):
        if bool(done_np.all()):
            break
        features, legal = _current(
            episode, device=device, action_count=action_count, feature_dim=feature_dim
        )
        active = torch.as_tensor(~done_np, device=device, dtype=torch.bool)
        with torch.no_grad():
            logits, values, hidden = model.step(features, legal, h=hidden)
            actions, logp = _sample_masked(logits, legal)
        # Explicit abstention is the no-op for already terminated rows.  Their
        # frames are padded out of the loss and never become training inputs.
        actions = torch.where(active, actions, torch.full_like(actions, action_count - 1))
        logp = torch.where(active, logp, torch.zeros_like(logp))
        values = torch.where(active, values, torch.zeros_like(values))

        reward_np, done_after = episode.step(actions.detach().cpu().numpy().astype(np.int64))
        reward_np = _numpy_vector(reward_np, batch_size, dtype=np.float32)
        done_after = _numpy_vector(done_after, batch_size, dtype=bool) | done_np
        # Retain only public observations for inactive/padded rows.
        stored_features = features.detach().clone()
        stored_legal = legal.detach().clone()
        stored_features[~active] = 0.0
        stored_legal[~active] = False
        reward = torch.as_tensor(reward_np, device=device, dtype=torch.float32)
        reward = torch.where(active, reward, torch.zeros_like(reward))

        feature_rows.append(stored_features)
        mask_rows.append(stored_legal)
        action_rows.append(actions.detach())
        old_logp_rows.append(logp.detach())
        value_rows.append(values.detach())
        reward_rows.append(reward)
        done_rows.append(torch.as_tensor(done_after, device=device, dtype=torch.bool))
        train_rows.append(active)
        done_np = done_after

    horizon = len(feature_rows)
    if horizon == 0:
        empty = {
            "x": prefix.new_zeros((batch_size, prefix.shape[1], feature_dim)),
            "legal": prefix_mask.clone(),
            "actions": torch.zeros(batch_size, prefix.shape[1], dtype=torch.long, device=device),
            "old_logp": prefix.new_zeros((batch_size, prefix.shape[1])),
            "returns": prefix.new_zeros((batch_size, prefix.shape[1])),
            "advantages": prefix.new_zeros((batch_size, prefix.shape[1])),
            "train_mask": torch.zeros(batch_size, prefix.shape[1], dtype=torch.bool, device=device),
        }
        return empty, 0

    live_x = torch.stack(feature_rows, dim=1)
    live_legal = torch.stack(mask_rows, dim=1)
    live_actions = torch.stack(action_rows, dim=1)
    live_old_logp = torch.stack(old_logp_rows, dim=1)
    live_values = torch.stack(value_rows, dim=1)
    live_rewards = torch.stack(reward_rows, dim=1)
    live_done = torch.stack(done_rows, dim=1)
    live_train = torch.stack(train_rows, dim=1)

    with torch.no_grad():
        if bool(done_np.all()):
            bootstrap = torch.zeros(batch_size, device=device, dtype=torch.float32)
        else:
            next_features, next_legal = _current(
                episode, device=device, action_count=action_count, feature_dim=feature_dim
            )
            _, bootstrap, _ = model.step(next_features, next_legal, h=hidden)
            bootstrap = torch.where(
                torch.as_tensor(done_np, device=device, dtype=torch.bool),
                torch.zeros_like(bootstrap),
                bootstrap,
            )
    returns, advantages = _gae(
        live_rewards,
        live_values,
        live_done,
        live_train,
        bootstrap.detach(),
        gamma=1.0,
        lam=0.95,
    )

    prefix_len = prefix.shape[1]
    sequence_x = torch.cat((prefix, live_x), dim=1)
    sequence_legal = torch.cat((prefix_mask, live_legal), dim=1)
    zeros_prefix_long = torch.zeros(batch_size, prefix_len, dtype=torch.long, device=device)
    zeros_prefix = prefix.new_zeros((batch_size, prefix_len))
    false_prefix = torch.zeros(batch_size, prefix_len, dtype=torch.bool, device=device)
    rollout = {
        "x": sequence_x,
        "legal": sequence_legal,
        "actions": torch.cat((zeros_prefix_long, live_actions), dim=1),
        "old_logp": torch.cat((zeros_prefix, live_old_logp), dim=1),
        "returns": torch.cat((zeros_prefix, returns), dim=1),
        "advantages": torch.cat((zeros_prefix, advantages), dim=1),
        "train_mask": torch.cat((false_prefix, live_train), dim=1),
        "reward": live_rewards,
    }
    return rollout, int(live_train.sum().item())


def neural_actions(model: torch.nn.Module, episode: Any) -> np.ndarray:
    """Return deterministic masked actions for an episode's current state.

    For an :class:`ActorCritic`, the public prescan prefix warms the GRU before
    the current frame.  For a :class:`QNetwork`, actions are simply masked Q
    argmaxes.  The returned array is always ``int64[B]``.
    """

    device, _ = next(model.parameters()).device, next(model.parameters()).dtype
    action_count = int(getattr(model, "action_count"))
    feature_dim = int(getattr(model, "state_dim"))
    features, legal = _current(
        episode, device=device, action_count=action_count, feature_dim=feature_dim
    )
    model.eval()
    with torch.no_grad():
        if isinstance(model, ActorCritic):
            prefix, prefix_mask = _prefix(
                episode, device=device, action_count=action_count, feature_dim=feature_dim
            )
            _, _, hidden = model.sequence(prefix, prefix_mask, h=None)
            logits, _, _ = model.step(features, legal, h=hidden)
            actions = _greedy_masked(logits, legal)
        elif isinstance(model, QNetwork):
            actions = _greedy_masked(model(features), legal)
        else:
            raise TypeError("model must be ActorCritic or QNetwork")
    return actions.detach().cpu().numpy().astype(np.int64)


def _outcome_metrics(outcomes: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not outcomes:
        return {"n": 0.0, "success": 0.0, "wrong": 0.0, "objective": 0.0,
                "tcp_mm": 0.0, "turn_deg": 0.0, "seconds": 0.0, "captures": 0.0}

    def mean(name: str) -> float:
        return float(np.mean([float(row.get(name, 0.0) or 0.0) for row in outcomes]))

    success = mean("success")
    wrong = mean("wrong")
    tcp_mm = mean("tcp_mm")
    turn_deg = mean("turn_deg")
    seconds = mean("seconds")
    captures = mean("captures")
    # The shared prefix adds a constant across policies on the same cases.
    rail_mm = mean("rail_mm")
    objective = success - 1.5 * wrong - .001*tcp_mm - .0005*turn_deg - .0001*rail_mm - .012*captures
    positive = [row for row in outcomes if row.get("positive",0)]
    negatives = [row for row in outcomes if "positive" in row and not row["positive"]]
    return {
        "n": float(len(outcomes)),
        "success": success,
        "wrong": wrong,
        "objective": float(objective),
        "tcp_mm": tcp_mm,
        "turn_deg": turn_deg,
        "seconds": seconds,
        "captures": captures,
        "rail_mm": rail_mm,
        "positive_n": len(positive),
        "positive_success": sum(int(row["success"]) for row in positive),
        "negative_n": len(negatives),
        "negative_success": sum(int(row["success"]) for row in negatives),
    }


def evaluate(model: torch.nn.Module, env: Any, *, batch: int = 128) -> List[Dict[str, Any]]:
    """Run a deterministic public-history policy over all environment cases."""

    n_cases = int(getattr(env, "n_cases"))
    feature_dim = int(getattr(env, "feature_dim"))
    action_count = int(getattr(env, "action_count"))
    device = next(model.parameters()).device
    outputs: List[Dict[str, Any]] = []
    model.eval()
    for start in range(0, n_cases, max(1, int(batch))):
        indices = np.arange(start, min(n_cases, start + max(1, int(batch))), dtype=np.int64)
        episode = env.reset(indices)
        bsz = len(indices)
        done_np = _done(episode, bsz)
        hidden: Optional[Tensor] = None
        if isinstance(model, ActorCritic):
            prefix, prefix_mask = _prefix(
                episode, device=device, action_count=action_count, feature_dim=feature_dim
            )
            with torch.no_grad():
                _, _, hidden = model.sequence(prefix, prefix_mask, h=None)
        # The world is finite; 64 acquisitions is a safety cap for malformed
        # mock environments while allowing the actual bank to terminate.
        for _ in range(64):
            if bool(done_np.all()):
                break
            features, legal = _current(
                episode, device=device, action_count=action_count, feature_dim=feature_dim
            )
            with torch.no_grad():
                if isinstance(model, ActorCritic):
                    logits, _, hidden = model.step(features, legal, h=hidden)
                    actions = _greedy_masked(logits, legal)
                elif isinstance(model, QNetwork):
                    actions = _greedy_masked(model(features), legal)
                else:
                    raise TypeError("model must be ActorCritic or QNetwork")
            actions = torch.where(
                torch.as_tensor(~done_np, device=device, dtype=torch.bool),
                actions,
                torch.full_like(actions, action_count - 1),
            )
            _, done_after = episode.step(actions.detach().cpu().numpy().astype(np.int64))
            done_np = _numpy_vector(done_after, bsz, dtype=bool) | done_np
        if hasattr(episode, "outcomes"):
            rows = episode.outcomes()
            if rows is not None:
                outputs.extend(list(rows))
    return outputs


def _checkpoint_payload(model: torch.nn.Module, *, kind: str, seed: int, interactions: int, update: int) -> Dict[str, Any]:
    config = {
        "kind": kind,
        "seed": int(seed),
        "interactions": int(interactions),
        "state_dim": int(getattr(model, "state_dim")),
        "action_count": int(getattr(model, "action_count")),
    }
    if isinstance(model, ActorCritic):
        config.update({"hidden": int(model.hidden), "recurrent": bool(model.recurrent)})
    elif isinstance(model, QNetwork):
        config.update({"hidden": int(model.hidden)})
    return {"model": model.state_dict(), "config": config, "update": int(update)}


def train_ppo(
    env: Any,
    dev: Any,
    *,
    seed: int,
    interactions: int,
    output: Path,
    recurrent: bool = True,
    device: str = "cpu",
    init: Any = None,
) -> Tuple[ActorCritic, Dict[str, Any]]:
    """Train a recurrent/no-GRU PPO policy on public batched episodes."""

    _seed_everything(seed)
    torch_device = _device(device)
    n_cases = int(getattr(env, "n_cases"))
    feature_dim = int(getattr(env, "feature_dim"))
    action_count = int(getattr(env, "action_count"))
    if n_cases <= 0 or feature_dim <= 0 or action_count <= 0:
        raise ValueError("env must expose positive n_cases, feature_dim and action_count")
    interactions = max(0, int(interactions))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = ActorCritic(feature_dim, action_count, recurrent=bool(recurrent)).to(torch_device)
    init_loaded = _load_init(model, init, torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    cursor = 0
    remaining = interactions
    update = 0
    train_curve: List[Dict[str, Any]] = []
    dev_curve: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_path: Optional[Path] = None
    batch_cap = min(256, n_cases)
    episode_cap = 64

    while remaining > 0:
        batch_size = min(batch_cap, remaining)
        indices = (np.arange(batch_size, dtype=np.int64) + cursor) % n_cases
        cursor = (cursor + batch_size) % n_cases
        # Keep the declared interaction cap exact even when a full batch is
        # used: B*T never exceeds remaining, while early terminals use fewer.
        max_steps = min(episode_cap, max(1, remaining // max(1, batch_size)))
        rollout, used = _ppo_rollout(model, env, indices, max_steps, torch_device)
        if used <= 0:
            # A mock may return already-terminated rows.  Move to another
            # cohort rather than spinning forever.
            remaining -= min(batch_size, remaining)
            continue
        stats = ppo_update(
            model,
            optimizer,
            rollout["x"],
            rollout["legal"],
            rollout["actions"],
            rollout["old_logp"],
            rollout["returns"],
            rollout["advantages"],
            rollout["train_mask"],
            epochs=2,
            batch=batch_size,
            clip=0.2,
            entropy=0.02,
        )
        remaining -= min(used, remaining)
        update += 1
        train_row: Dict[str, Any] = {
            "update": update,
            "interactions": interactions - remaining,
            "frames": used,
            "mean_reward": float(rollout["reward"].sum().detach().cpu()) / max(1, used),
            **stats,
        }
        train_curve.append(train_row)

        if dev is not None and (update == 1 or update % 10 == 0 or remaining == 0):
            dev_metrics = _outcome_metrics(evaluate(model, dev, batch=128))
            dev_metrics["update"] = update
            dev_curve.append(dev_metrics)
            if dev_metrics["objective"] >= best_score:
                best_score = float(dev_metrics["objective"])
                best_path = output / "ppo_best.pt"
                _safe_save(
                    best_path,
                    {**_checkpoint_payload(model, kind="ppo", seed=seed, interactions=interactions, update=update),
                     "dev": dev_metrics},
                )

        _safe_save(
            output / "ppo_last.pt",
            {**_checkpoint_payload(model, kind="ppo", seed=seed, interactions=interactions, update=update),
             "dev": dev_curve[-1] if dev_curve else None},
        )

    # If no interaction was requested, still provide a loadable checkpoint.
    if update == 0:
        _safe_save(output / "ppo_last.pt", _checkpoint_payload(model, kind="ppo", seed=seed, interactions=interactions, update=0))
    selected = "final"
    if best_path is not None and best_path.exists():
        checkpoint = torch.load(best_path, map_location=torch_device, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        selected = "best_dev"
    final_path = output / "ppo_final.pt"
    _safe_save(final_path, _checkpoint_payload(model, kind="ppo", seed=seed, interactions=interactions, update=update))
    summary: Dict[str, Any] = {
        "kind": "ppo",
        "seed": int(seed),
        "interactions": int(interactions),
        "actual_interactions": int(interactions - remaining),
        "recurrent": bool(recurrent),
        "device": str(torch_device),
        "init_loaded": bool(init_loaded),
        "selection": selected,
        "selection_objective": "mean(success)-1.5*mean(wrong)-.001*TCP-.0005*turn-.0001*rail-.012*captures",
        "best_dev_objective": None if best_path is None else float(best_score),
        "checkpoint": str(final_path),
        "train_curve": train_curve,
        "dev_curve": dev_curve,
    }
    _write_json(output / "ppo_summary.json", summary)
    return model, summary


def _collect_replay(
    env: Any,
    *,
    interactions: int,
    seed: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Collect one fixed mixed greedy/random replay from public transitions."""

    n_cases = int(getattr(env, "n_cases"))
    feature_dim = int(getattr(env, "feature_dim"))
    action_count = int(getattr(env, "action_count"))
    rng = np.random.default_rng(int(seed) + 2111)
    batch_cap = min(256, n_cases)
    cursor = 0
    remaining = max(0, int(interactions))
    rows: Dict[str, List[np.ndarray]] = {key: [] for key in ("x", "mask", "a", "r", "next_x", "next_mask", "done")}
    while remaining > 0:
        batch_size = min(batch_cap, remaining)
        indices = (np.arange(batch_size, dtype=np.int64) + cursor) % n_cases
        cursor = (cursor + batch_size) % n_cases
        episode = env.reset(indices)
        done_np = _done(episode, batch_size)
        for _ in range(64):
            if bool(done_np.all()) or remaining <= 0:
                break
            features_t, legal_t = _current(
                episode, device=device, action_count=action_count, feature_dim=feature_dim
            )
            features = features_t.detach().cpu().numpy().astype(np.float32)
            legal = legal_t.detach().cpu().numpy().astype(bool)
            try:
                greedy = np.asarray(episode.actions(method="greedy"), dtype=np.int64).reshape(-1)
            except TypeError:
                greedy = np.asarray(episode.actions("greedy"), dtype=np.int64).reshape(-1)
            if greedy.size == 1 and batch_size != 1:
                greedy = np.repeat(greedy, batch_size)
            if greedy.size != batch_size:
                raise ValueError("episode.actions(method='greedy') must return [B]")
            random_actions = _random_legal(legal, rng, action_count - 1)
            use_greedy = rng.random(batch_size) < 0.5
            actions = np.where(use_greedy, greedy, random_actions).astype(np.int64)
            actions = np.where(done_np, action_count - 1, actions)
            reward, done_after = episode.step(actions)
            reward = _numpy_vector(reward, batch_size, dtype=np.float32)
            done_after = _numpy_vector(done_after, batch_size, dtype=bool) | done_np
            next_features_t, next_legal_t = _current(
                episode, device=device, action_count=action_count, feature_dim=feature_dim
            )
            next_features = next_features_t.detach().cpu().numpy().astype(np.float32)
            next_legal = next_legal_t.detach().cpu().numpy().astype(bool)
            active = ~done_np
            if bool(active.any()):
                # Never append more transitions than the declared fixed
                # interaction cap, even when the last batched step has more
                # active rows than remain.
                take = np.flatnonzero(active)[:remaining]
                if take.size:
                    rows["x"].append(features[take])
                    rows["mask"].append(legal[take])
                    rows["a"].append(actions[take])
                    rows["r"].append(reward[take])
                    rows["next_x"].append(next_features[take])
                    rows["next_mask"].append(next_legal[take])
                    rows["done"].append(done_after[take])
                    remaining -= int(take.size)
            done_np = done_after
        if not rows["x"] and batch_size:
            remaining -= min(batch_size, remaining)

    if not rows["x"]:
        return {
            "x": np.zeros((0, feature_dim), np.float32),
            "mask": np.zeros((0, action_count), bool),
            "a": np.zeros((0,), np.int64),
            "r": np.zeros((0,), np.float32),
            "next_x": np.zeros((0, feature_dim), np.float32),
            "next_mask": np.zeros((0, action_count), bool),
            "done": np.zeros((0,), bool),
        }
    return {key: np.concatenate(value, axis=0) for key, value in rows.items()}


def train_cql(
    env: Any,
    dev: Any,
    *,
    seed: int,
    interactions: int,
    output: Path,
    device: str = "cpu",
) -> Tuple[QNetwork, Dict[str, Any]]:
    """Fit a masked discrete CQL baseline on one fixed logged replay."""

    _seed_everything(seed)
    torch_device = _device(device)
    n_cases = int(getattr(env, "n_cases"))
    feature_dim = int(getattr(env, "feature_dim"))
    action_count = int(getattr(env, "action_count"))
    if n_cases <= 0 or feature_dim <= 0 or action_count <= 0:
        raise ValueError("env must expose positive n_cases, feature_dim and action_count")
    interactions = max(0, int(interactions))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    replay = _collect_replay(
        env, interactions=interactions, seed=seed, device=torch_device
    )
    model = QNetwork(feature_dim, action_count).to(torch_device)
    target = QNetwork(feature_dim, action_count).to(torch_device)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    replay_n = int(replay["x"].shape[0])
    rng = np.random.default_rng(int(seed) + 3109)
    updates = max(0, min(2400, int(math.ceil(replay_n / 256.0)) * 6))
    curve: List[Dict[str, Any]] = []
    for update in range(1, updates + 1):
        sample_n = min(256, replay_n)
        sample = rng.choice(replay_n, size=sample_n, replace=replay_n < sample_n)
        stats = cql_update(
            model,
            target,
            optimizer,
            torch.as_tensor(replay["x"][sample], device=torch_device),
            torch.as_tensor(replay["mask"][sample], device=torch_device),
            torch.as_tensor(replay["a"][sample], device=torch_device),
            torch.as_tensor(replay["r"][sample], device=torch_device),
            torch.as_tensor(replay["next_x"][sample], device=torch_device),
            torch.as_tensor(replay["next_mask"][sample], device=torch_device),
            torch.as_tensor(replay["done"][sample], device=torch_device),
            alpha=0.2,
            gamma=1.0,
        )
        if update % 32 == 0 or update == updates:
            target.load_state_dict(model.state_dict())
        curve.append({"update": update, **stats})

    if dev is not None:
        dev_metrics = _outcome_metrics(evaluate(model, dev, batch=128))
    else:
        dev_metrics = _outcome_metrics([])
    dev_curve = [{"update": updates, **dev_metrics}]
    final_path = output / "cql_final.pt"
    _safe_save(final_path, _checkpoint_payload(model, kind="cql", seed=seed, interactions=interactions, update=updates))
    summary: Dict[str, Any] = {
        "kind": "cql",
        "seed": int(seed),
        "interactions": int(interactions),
        "replay_size": replay_n,
        "updates": updates,
        "device": str(torch_device),
        "fixed_mixed_replay": True,
        "replay_greedy_fraction": 0.5,
        "dev": dev_metrics,
        "dev_curve": dev_curve,
        "curve": curve,
        "checkpoint": str(final_path),
    }
    _write_json(output / "cql_summary.json", summary)
    return model, summary

