"""CPU actor–critic for scan continuation and continuous ProDMP rechecks."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Decision:
    recheck: bool
    latent: np.ndarray
    action: np.ndarray
    log_probability: float
    value: float
    probability: float


class ScanPolicy(nn.Module):
    """The skill head is discrete, and the conditional control head is continuous."""

    def __init__(self, feature_count):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(feature_count, 64), nn.Tanh(), nn.Linear(64, 48), nn.Tanh()
        )
        self.skill = nn.Linear(48, 1)
        self.control = nn.Linear(48, 3)
        self.value = nn.Linear(48, 1)
        self.log_scale = nn.Parameter(torch.full((3,), -.45))
        nn.init.orthogonal_(self.skill.weight, gain=.08)
        nn.init.constant_(self.skill.bias, -.28)
        nn.init.orthogonal_(self.control.weight, gain=.08)
        nn.init.zeros_(self.control.bias)

    def forward(self, observations):
        x = self.body(observations)
        return self.skill(x).squeeze(-1), self.control(x), self.value(x).squeeze(-1)

    def likelihood(self, observations, eligible, skill, latent):
        logits, means, values = self(observations)
        scale = self.log_scale.clamp(-2.6, .3).exp()
        normal = torch.distributions.Normal(means, scale)
        log_skill = -F.binary_cross_entropy_with_logits(logits, skill, reduction="none")
        # The Jacobian is required for the bounded controls. The null skill
        # has no continuous action and contributes no Gaussian likelihood.
        tanh_jacobian = 2 * (np.log(2.) - latent - F.softplus(-2 * latent))
        log_motor = (normal.log_prob(latent) - tanh_jacobian).sum(-1)
        entropy_skill = torch.distributions.Bernoulli(logits=logits).entropy()
        entropy = eligible * (entropy_skill + logits.sigmoid() * normal.entropy().sum(-1))
        return eligible * log_skill + skill * log_motor, values, entropy

    @torch.inference_mode()
    def choose(self, features, eligible, rng=None, threshold=.5):
        observation = torch.as_tensor(features, dtype=torch.float32)
        logit, mean, value = self(observation)
        probability = float(logit.sigmoid())
        recheck = bool(eligible and (rng.random() < probability if rng is not None
                                     else probability >= threshold))
        latent = mean.detach().clone()
        if recheck and rng is not None:
            noise = torch.from_numpy(rng.standard_normal(3).astype(np.float32))
            latent += noise * self.log_scale.clamp(-2.6, .3).exp()
        if not recheck:
            latent.zero_()
        log_probability = self.likelihood(
            observation, observation.new_tensor(float(eligible)),
            observation.new_tensor(float(recheck)), latent,
        )[0]
        return Decision(recheck, latent.numpy(), latent.tanh().numpy(),
                        float(log_probability), float(value), probability)


def update_ppo(policy, optimizer, episodes, rng, *, epochs=6, minibatch=128,
               discount=.985, gae_decay=.93, clip=.2):
    transitions = []
    returns, advantages = [], []
    for steps in episodes:
        episode_advantages = []
        next_value = 0.
        advantage = 0.
        for step in reversed(steps):
            delta = step["reward"] + discount * next_value - step["value"]
            advantage = delta + discount * gae_decay * advantage
            episode_advantages.append(advantage)
            next_value = step["value"]
        episode_advantages.reverse()
        transitions.extend(steps)
        advantages.extend(episode_advantages)
        returns.extend(step["value"] + a for step, a in zip(steps, episode_advantages))

    observations = torch.tensor(np.asarray([s["features"] for s in transitions]), dtype=torch.float32)
    eligible = torch.tensor([s["eligible"] for s in transitions], dtype=torch.float32)
    skills = torch.tensor([s["selected_skill"] == "recheck" for s in transitions], dtype=torch.float32)
    latent = torch.tensor(np.asarray([s["latent"] for s in transitions]), dtype=torch.float32)
    old_logs = torch.tensor([s["log_probability"] for s in transitions], dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    selectable = eligible.bool()
    if selectable.any():
        scale = advantages[selectable].std(unbiased=False).clamp_min(.01)
        advantages = (advantages - advantages[selectable].mean()) / scale

    losses, grad_norms = [], []
    for _ in range(epochs):
        indices = rng.permutation(len(transitions))
        for offset in range(0, len(indices), minibatch):
            idx = indices[offset:offset + minibatch]
            logp, values, entropy = policy.likelihood(observations[idx], eligible[idx],
                                                      skills[idx], latent[idx])
            ratio = (logp - old_logs[idx]).exp()
            advantage = advantages[idx]
            mask = eligible[idx]
            actor = -(mask * torch.minimum(ratio * advantage,
                       ratio.clamp(1 - clip, 1 + clip) * advantage)).sum() / mask.sum().clamp_min(1)
            critic = F.smooth_l1_loss(values, returns[idx])
            entropy = entropy.sum() / mask.sum().clamp_min(1)
            loss = actor + .55 * critic - .013 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(policy.parameters(), .7)
            optimizer.step()
            losses.append((actor.item(), critic.item(), entropy.item(),
                           ((old_logs[idx] - logp) * mask).sum().item() / mask.sum().clamp_min(1).item()))
            grad_norms.append(float(norm))
    actor, critic, entropy, kl = np.mean(losses, axis=0)
    return {"steps": len(transitions), "eligible_steps": int(eligible.sum()),
            "recheck_proposals": int(skills.sum()), "minibatch_updates": len(losses),
            "actor_loss": float(actor), "critic_loss": float(critic),
            "entropy": float(entropy), "approximate_kl": float(kl),
            "gradient_norm": float(np.mean(grad_norms))}
