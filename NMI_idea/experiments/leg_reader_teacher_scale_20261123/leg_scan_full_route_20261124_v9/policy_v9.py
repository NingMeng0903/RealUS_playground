"""Recurrent actor and training-only privileged critic for full-route v9."""

from __future__ import annotations

import numpy as np
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F


FORMAT = "recurrent_asymmetric_route_policy_v9"
ACTOR_CAP = 10_000
BC_EXPLORATION = .17
GAMMA = 1.
GAE_LAMBDA = .92
PPO_CLIP = .20
ENTROPY_WEIGHT = .015
VALUE_WEIGHT = .5
BATCH_ROUTES = 24
BC_EPOCHS = 4
PPO_EPOCHS = 4


class Policy(nn.Module):
    """An actor over public history and a critic over history and simulator state."""

    def __init__(self, obs_dim, priv_dim, n_actions):
        super().__init__()
        if obs_dim < 1 or priv_dim < 1 or n_actions < 2:
            raise ValueError("Expected positive observation and privileged widths and at least two actions")
        widths = [w for w in range(1, 33)
                  if 6 * w * w + w * (obs_dim + n_actions + 7) + n_actions <= ACTOR_CAP]
        if not widths:
            raise ValueError("Observation or action space exceeds the actor parameter budget")
        self.obs_dim = obs_dim
        self.priv_dim = priv_dim
        self.n_actions = n_actions
        self.width = widths[-1]
        self.projection = nn.Linear(obs_dim, self.width)
        self.memory = nn.GRUCell(self.width, self.width)
        self.action = nn.Linear(self.width, n_actions)
        self.critic_in = nn.Linear(self.width + priv_dim, 32)
        self.critic_out = nn.Linear(32, 1)
        nn.init.orthogonal_(self.action.weight, gain=.01)
        nn.init.zeros_(self.action.bias)
        nn.init.zeros_(self.critic_out.bias)

    def actor_parameters(self):
        return (*self.projection.parameters(), *self.memory.parameters(), *self.action.parameters())

    def critic_parameters(self):
        return (*self.critic_in.parameters(), *self.critic_out.parameters())

    def _value(self, history, privileged):
        joined = torch.cat((history, privileged), dim=-1)
        return self.critic_out(torch.tanh(self.critic_in(joined))).squeeze(-1)

    def forward(self, obs, h, priv=None):
        h_next = self.memory(torch.tanh(self.projection(obs)), h)
        value = self._value(h_next, priv) if priv is not None else None
        return self.action(h_next), h_next, value

    def _sequence(self, obs, priv):
        projected = torch.tanh(self.projection(obs))
        hidden = projected.new_zeros((obs.shape[0], self.width))
        states = []
        for step in projected.unbind(dim=1):
            hidden = self.memory(step, hidden)
            states.append(hidden)
        history = torch.stack(states, dim=1)
        return self.action(history), self._value(history, priv)


def export(model):
    """Export JSON-compatible float32 weights, with separate actor and critic groups."""
    weights = model.state_dict()
    critic = {"critic_in", "critic_out"}
    return {
        "format": FORMAT,
        "obs_dim": model.obs_dim,
        "priv_dim": model.priv_dim,
        "n_actions": model.n_actions,
        "hidden_dim": model.width,
        "actor": {key: value.detach().cpu().tolist()
                  for key, value in weights.items() if key.split(".")[0] not in critic},
        "critic": {key: value.detach().cpu().tolist()
                   for key, value in weights.items() if key.split(".")[0] in critic},
    }


class Actor:
    """CPU rollout policy. The actor step works with an actor-only export."""

    def __init__(self, bundle):
        if bundle["format"] != FORMAT:
            raise ValueError("Invalid v9 policy export")
        self.width = int(bundle["hidden_dim"])
        self.n_actions = int(bundle["n_actions"])
        self.actor = {key: np.asarray(value, np.float32) for key, value in bundle["actor"].items()}
        self.critic = {key: np.asarray(value, np.float32) for key, value in bundle.get("critic", {}).items()}

    def zero(self):
        return np.zeros(self.width, dtype=np.float32)

    def step(self, obs, h, mask):
        w = self.actor
        obs = np.asarray(obs, dtype=np.float32)
        h = np.asarray(h, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_actions,) or not mask.any():
            raise ValueError("Expected a nonempty legal action mask")

        embedded = np.tanh(w["projection.weight"] @ obs + w["projection.bias"])
        input_gates = w["memory.weight_ih"] @ embedded + w["memory.bias_ih"]
        history_gates = w["memory.weight_hh"] @ h + w["memory.bias_hh"]
        width = self.width
        reset = expit(input_gates[:width] + history_gates[:width])
        update = expit(input_gates[width:2 * width] + history_gates[width:2 * width])
        candidate = np.tanh(input_gates[2 * width:] + reset * history_gates[2 * width:])
        h_next = (h - candidate) * update + candidate
        logits = w["action.weight"] @ h_next + w["action.bias"]
        logits[~mask] = -np.inf
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return probabilities, h_next

    def value(self, h_next, priv):
        w = self.critic
        if not w:
            raise ValueError("Policy export has no training critic")
        joined = np.concatenate((h_next, priv), dtype=np.float32)
        hidden = np.tanh(w["critic_in.weight"] @ joined + w["critic_in.bias"])
        return float((w["critic_out.weight"] @ hidden + w["critic_out.bias"])[0])


def _episodes(model, episodes, on_policy):
    prepared = []
    for episode in episodes:
        obs = np.asarray(episode["obs"], dtype=np.float32)
        n = len(obs)
        if n < 1:
            raise ValueError("Empty route episode")
        shapes = {
            "obs": ((n, model.obs_dim), np.float32),
            "priv": ((n, model.priv_dim), np.float32),
            "mask": ((n, model.n_actions), bool),
            "action": ((n,), np.int64),
            "duration": ((n,), np.float32),
        }
        if on_policy or "reward" in episode:
            shapes["reward"] = ((n,), np.float32)
        if on_policy:
            shapes["logp"] = ((n,), np.float32)
            shapes["value"] = ((n,), np.float32)
        row = {}
        for key, (shape, dtype) in shapes.items():
            value = np.asarray(episode[key], dtype=dtype)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid episode field {key}")
            row[key] = value
        actions = row["action"]
        if ((actions < 0) | (actions >= model.n_actions)).any() or not row["mask"][np.arange(n), actions].all():
            raise ValueError("Episode contains an illegal action")
        if (row["duration"] < 0).any() or not np.equal(row["duration"], np.floor(row["duration"])).all():
            raise ValueError("Macro durations must be nonnegative integer ticks")
        if on_policy and (row["logp"] > 1e-6).any():
            raise ValueError("Behavior log-probabilities must be nonpositive")

        if on_policy:
            old_v = row["value"]
            td_error = row["reward"] + GAMMA * np.r_[old_v[1:], np.float32(0.)] - old_v
            advantages = np.empty(n, dtype=np.float32)
            tail = 0.
            for step in reversed(range(n)):
                tail = td_error[step] + GAMMA * GAE_LAMBDA * tail
                advantages[step] = tail
            row["advantage"] = advantages
            row["return"] = advantages + old_v
        else:
            reward = row.get("reward")
            row["value_valid"] = np.full(n, reward is not None, dtype=bool)
            row["return"] = (np.cumsum(reward[::-1], dtype=np.float32)[::-1]
                             if reward is not None else np.zeros(n, dtype=np.float32))
        if on_policy:
            row["teacher_mask"] = np.zeros_like(row["mask"])
            row["teacher_q"] = np.zeros_like(row["mask"], dtype=np.float32)
            if "teacher_q" in episode or "teacher_mask" in episode:
                teacher_q = np.asarray(episode["teacher_q"], dtype=np.float32)
                teacher_mask = np.asarray(episode["teacher_mask"], dtype=bool)
                if teacher_q.shape != row["mask"].shape or teacher_mask.shape != row["mask"].shape:
                    raise ValueError("Teacher scores and masks must align with the policy action space")
                teacher_mask = teacher_mask & row["mask"]
                if not np.isfinite(teacher_q[teacher_mask]).all():
                    raise ValueError("Labelled legal teacher scores must be finite")
                row["teacher_mask"] = teacher_mask
                row["teacher_q"] = np.where(teacher_mask, teacher_q, 0.)
        row.pop("reward", None)
        prepared.append(row)
    if not prepared:
        raise ValueError("Training requires at least one complete route")
    if on_policy:
        all_advantages = np.concatenate([row["advantage"] for row in prepared])
        center, scale = float(all_advantages.mean()), max(float(all_advantages.std()), 1e-8)
        for row in prepared:
            row["advantage"] = (row["advantage"] - center) / scale
    return prepared


def _padded(rows, device):
    lengths = np.array([len(row["action"]) for row in rows])
    padded_shape = (len(rows), int(lengths.max()))
    valid = np.arange(padded_shape[1])[None, :] < lengths[:, None]
    arrays = {}
    for key in rows[0]:
        example = rows[0][key]
        padded = np.zeros(padded_shape + example.shape[1:], dtype=example.dtype)
        for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
            padded[index, :length] = row[key]
        arrays[key] = padded
    # Padded timesteps contribute no loss. Give softmax a finite dummy action.
    arrays["mask"][~valid, 0] = True
    arrays["valid"] = valid
    return {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}, int(lengths.sum())


def _bc_loss(model, batch):
    logits, values = model._sequence(batch["obs"], batch["priv"])
    valid = batch["valid"]
    legal = batch["mask"][valid]
    actions = batch["action"][valid]
    logp = F.log_softmax(logits[valid].masked_fill(~legal, -torch.inf), dim=-1)
    chosen = logp.gather(-1, actions[:, None]).squeeze(-1)
    alternatives = legal.sum(dim=-1) - 1
    alt_sum = logp.masked_fill(~legal, 0.).sum(dim=-1) - chosen
    spread = BC_EXPLORATION * (alternatives > 0)
    actor_loss = (-(1. - spread) * chosen - spread * alt_sum / alternatives.clamp_min(1)).mean()
    value_valid = batch["value_valid"][valid]
    squared_error = (values[valid] - batch["return"][valid]).square()
    critic_loss = (squared_error * value_valid).sum() / value_valid.sum().clamp_min(1)
    zero = actor_loss.new_zeros(())
    return actor_loss + VALUE_WEIGHT * critic_loss, (actor_loss, critic_loss, zero, zero, zero, zero)


def _ppo_loss(model, batch, teacher_weight):
    logits, values = model._sequence(batch["obs"], batch["priv"])
    valid = batch["valid"]
    legal = batch["mask"][valid]
    logits = logits[valid].masked_fill(~legal, -torch.inf)
    logp = F.log_softmax(logits, dim=-1)
    chosen = logp.gather(-1, batch["action"][valid][:, None]).squeeze(-1)
    log_ratio = chosen - batch["logp"][valid]
    ratio = log_ratio.exp()
    advantage = batch["advantage"][valid].detach()
    clipped = ratio.clamp(1. - PPO_CLIP, 1. + PPO_CLIP)
    actor_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    critic_loss = F.mse_loss(values[valid], batch["return"][valid].detach())
    entropy = -(logp.exp() * logp.masked_fill(~legal, 0.)).sum(dim=-1).mean()
    kl = (ratio - 1. - log_ratio).mean()
    clip_fraction = ((ratio - 1.).abs() > PPO_CLIP).float().mean()

    teacher_loss = actor_loss.new_zeros(())
    if teacher_weight:
        labelled = batch["teacher_mask"][valid]
        counts = labelled.sum(dim=-1)
        eligible = counts > 1
        # Use the original legal set as the finite fallback on unlabelled rows.
        subset = torch.where(eligible[:, None], labelled, legal)
        distribution = F.softmax(logits.masked_fill(~subset, -torch.inf), dim=-1)
        q = batch["teacher_q"][valid].detach()
        baseline = q.sum(dim=-1, keepdim=True) / counts[:, None].clamp_min(1)
        teacher_loss = -((distribution * (q - baseline)).sum(dim=-1) * eligible).sum()
        teacher_loss = teacher_loss / eligible.sum().clamp_min(1)
    loss = actor_loss + VALUE_WEIGHT * critic_loss - ENTROPY_WEIGHT * entropy + teacher_weight * teacher_loss
    return loss, (actor_loss, critic_loss, entropy, teacher_loss, kl, clip_fraction)


def _train(model, opt, episodes, seed, *, on_policy, teacher_weight):
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("v9 policy updates require CUDA")
    if not np.isfinite(teacher_weight) or teacher_weight < 0.:
        raise ValueError("Teacher weight must be finite and nonnegative")
    rows = _episodes(model, episodes, on_policy)
    generator = np.random.default_rng(seed)
    epochs = PPO_EPOCHS if on_policy else BC_EPOCHS
    actors = model.actor_parameters()
    critics = model.critic_parameters()
    before = [param.detach().clone() for param in (*actors, *critics)]
    totals = torch.zeros(7, device=device)
    max_grad = torch.zeros((), device=device)
    steps = transitions = 0
    model.train()
    for _ in range(epochs):
        order = generator.permutation(len(rows))
        for start in range(0, len(order), BATCH_ROUTES):
            indices = order[start:start + BATCH_ROUTES]
            batch, n = _padded([rows[index] for index in indices], device)
            opt.zero_grad(set_to_none=True)
            if on_policy:
                loss, metrics = _ppo_loss(model, batch, teacher_weight)
            else:
                loss, metrics = _bc_loss(model, batch)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            totals += n * torch.stack([loss.detach(), *(metric.detach() for metric in metrics)])
            max_grad = torch.maximum(max_grad, grad.detach())
            steps += 1
            transitions += n

    changed = torch.stack([(old - new.detach()).square().sum()
                           for old, new in zip(before, (*actors, *critics), strict=True)])
    actor_delta = changed[:len(actors)].sum().sqrt().item()
    critic_delta = changed[len(actors):].sum().sqrt().item()
    durations = np.concatenate([row["duration"] for row in rows])
    names = ("loss", "actor_loss", "critic_loss", "entropy", "teacher_loss", "kl", "clip_fraction")
    metrics = dict(zip(names, (totals / transitions).tolist(), strict=True))
    metrics.update({
        "device": str(device),
        "actor_parameters": sum(param.numel() for param in actors),
        "critic_parameters": sum(param.numel() for param in critics),
        "optimizer_steps": steps,
        "gradient_norm_max": max_grad.item(),
        "actor_delta_l2": actor_delta,
        "critic_delta_l2": critic_delta,
        "cuda_optimizer_step_verified": bool(steps and max_grad.item() > 0. and actor_delta + critic_delta > 0.),
        "routes": len(rows),
        "macro_steps": len(durations),
        "duration_ticks_min": int(durations.min()),
        "duration_ticks_max": int(durations.max()),
        "duration_ticks_total": int(durations.sum(dtype=np.float64)),
        "gamma_per_macro": GAMMA,
        "gae_lambda_per_macro": GAE_LAMBDA,
        "teacher_labelled_states": (sum(int((row["teacher_mask"].sum(axis=-1) > 1).sum()) for row in rows)
                                    if on_policy else 0),
    })
    return metrics


def bc(model, opt, episodes, seed):
    """Fit the soft demonstration policy and available privileged return targets."""
    return _train(model, opt, episodes, seed, on_policy=False, teacher_weight=0.)


def ppo(model, opt, episodes, seed, *, teacher_weight=0.):
    """Fit clipped on-policy actor and critic losses on complete recurrent routes."""
    return _train(model, opt, episodes, seed, on_policy=True, teacher_weight=teacher_weight)
