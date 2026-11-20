"""Recurrent PPO with a public-observation policy and asymmetric value function."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


MOVE = 0
SETTLE_CONFIRM = 1
DELTA_DIM = 4


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, privileged_dim: int, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.privileged_dim = privileged_dim
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.recurrent = nn.GRU(hidden, hidden, batch_first=True)
        self.mode_head = nn.Linear(hidden, 2)
        self.delta_head = nn.Linear(hidden, DELTA_DIM)
        self.log_std = nn.Parameter(torch.full((DELTA_DIM,), -0.5))
        self.value_head = nn.Sequential(
            nn.Linear(hidden + privileged_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def _tensor(self, value):
        return torch.as_tensor(
            value, dtype=self.log_std.dtype, device=self.log_std.device
        )

    def _encode(self, observations, hidden):
        features, hidden = self.recurrent(
            self.encoder(observations).unsqueeze(0), hidden
        )
        return features.squeeze(0), hidden

    @torch.no_grad()
    def burn_in(self, observations, hidden=None):
        """Recompute the hidden state from a public prefix without gradient or loss."""
        observations = self._tensor(observations).reshape(-1, self.obs_dim)
        if observations.shape[0]:
            _, hidden = self._encode(observations, hidden)
        return hidden

    def _distributions(self, features):
        mode = Categorical(logits=self.mode_head(features))
        delta = Normal(self.delta_head(features), self.log_std.clamp(-5, 1).exp())
        return mode, delta

    def _value(self, features, privileged):
        inputs = torch.cat((features.detach(), privileged), dim=-1)
        return self.value_head(inputs).squeeze(-1)

    @staticmethod
    def _delta_log_prob(distribution, latent):
        # Stable log derivative of tanh, including for large latent magnitudes.
        log_jacobian = 2 * (np.log(2) - latent - F.softplus(-2 * latent))
        return (distribution.log_prob(latent) - log_jacobian).sum(dim=-1)

    def _log_prob(self, mode_distribution, delta_distribution, modes, latents):
        delta_log_prob = self._delta_log_prob(delta_distribution, latents)
        return mode_distribution.log_prob(modes) + torch.where(
            modes == MOVE, delta_log_prob, torch.zeros_like(delta_log_prob)
        )

    @torch.no_grad()
    def act(self, obs, hidden, privileged=None, deterministic=False):
        features, new_hidden = self._encode(self._tensor(obs).unsqueeze(0), hidden)
        mode_distribution, delta_distribution = self._distributions(features)
        mode = (
            mode_distribution.logits.argmax(dim=-1)
            if deterministic
            else mode_distribution.sample()
        )
        if mode.item() == MOVE:
            latent = (
                delta_distribution.mean if deterministic else delta_distribution.sample()
            )
        else:
            latent = torch.zeros_like(delta_distribution.mean)
        log_prob = self._log_prob(mode_distribution, delta_distribution, mode, latent)
        value = None
        if privileged is not None:
            value = self._value(features, self._tensor(privileged).unsqueeze(0)).item()
        action = {
            "mode": int(mode.item()),
            "latent": latent[0].cpu().numpy(),
            "delta": latent[0].tanh().cpu().numpy(),
        }
        return action, log_prob.item(), value, new_hidden

    def evaluate_sequence(self, observations, privileged, modes, latents, warmup):
        """Evaluate a complete rollout in chronological order."""
        hidden = self.burn_in(warmup)
        features, _ = self._encode(self._tensor(observations), hidden)
        mode_distribution, delta_distribution = self._distributions(features)
        modes = torch.as_tensor(modes, dtype=torch.long, device=features.device)
        log_prob = self._log_prob(
            mode_distribution, delta_distribution, modes, self._tensor(latents)
        )
        # Joint entropy = mode entropy + P(move) times conditional delta entropy.
        delta_entropy = -self._delta_log_prob(
            delta_distribution, delta_distribution.rsample()
        )
        entropy = mode_distribution.entropy() + (
            mode_distribution.probs[:, MOVE] * delta_entropy
        )
        return log_prob, self._value(features, self._tensor(privileged)), entropy


@dataclass
class Episode:
    warmup: np.ndarray
    observations: np.ndarray
    privileged: np.ndarray
    modes: np.ndarray
    latents: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray

    def advantages_and_returns(self, gamma: float, gae_lambda: float):
        """Compute GAE with zero bootstrap at the terminal transition."""
        advantages = np.empty_like(self.values)
        next_value = 0.0
        next_advantage = 0.0
        for index in range(len(advantages) - 1, -1, -1):
            delta = self.rewards[index] + gamma * next_value - self.values[index]
            next_advantage = delta + gamma * gae_lambda * next_advantage
            advantages[index] = next_advantage
            next_value = self.values[index]
        return advantages, advantages + self.values


def update_ppo(
    model,
    optimizer,
    episodes,
    *,
    epochs=4,
    gamma=1.0,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """Run full-batch PPO epochs over frozen, complete on-policy rollouts."""
    targets = [episode.advantages_and_returns(gamma, gae_lambda) for episode in episodes]
    all_advantages = np.concatenate([advantages for advantages, _ in targets])
    advantage_mean = all_advantages.mean()
    advantage_std = all_advantages.std() + 1e-8
    total_steps = len(all_advantages)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        for episode, (advantages, returns) in zip(episodes, targets):
            log_probs, values, entropy = model.evaluate_sequence(
                episode.observations,
                episode.privileged,
                episode.modes,
                episode.latents,
                episode.warmup,
            )
            advantages = model._tensor((advantages - advantage_mean) / advantage_std)
            log_ratio = log_probs - model._tensor(episode.log_probs)
            ratio = log_ratio.exp()
            clipped_ratio = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
            value_loss = (values - model._tensor(returns)).square()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            (loss.sum() / total_steps).backward()
            stats["policy_loss"] += policy_loss.detach().sum().item() / total_steps
            stats["value_loss"] += value_loss.detach().sum().item() / total_steps
            stats["entropy"] += entropy.detach().sum().item() / total_steps
            approx_kl = (ratio.detach() - 1) - log_ratio.detach()
            stats["approx_kl"] += approx_kl.sum().item() / total_steps
        nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()
    return stats
