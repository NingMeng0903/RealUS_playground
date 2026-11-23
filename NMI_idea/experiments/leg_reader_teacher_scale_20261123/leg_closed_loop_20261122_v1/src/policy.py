"""GPU policy, CPU inference actor, and SMDP PPO updates for leg scans.

The training policy is deliberately small. Rollout workers use :class:`NumpyActor`
so they never inherit a CUDA context. ``ScanPolicy`` is moved to CUDA by the
runner before an optimizer is constructed, and PPO tensors then stay there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HIGH_SKILLS = ("continue", "window", "revisit")
SKILL_TO_INDEX = {name: index for index, name in enumerate(HIGH_SKILLS)}


def _skill_index(value) -> int:
    if isinstance(value, str):
        if value in SKILL_TO_INDEX:
            return SKILL_TO_INDEX[value]
        if value in ("recheck", "stop"):
            return 1
        return 0
    return int(np.clip(int(value), 0, 2))


def _normalise_mask(mask, stage: int = 0) -> np.ndarray:
    """Return a three-slot mask, retaining legal continuation."""
    if mask is None:
        mask = (True, True, True) if stage == 0 else (True, True, False)
    if isinstance(mask, (bool, np.bool_)):
        mask = (True, bool(mask), bool(mask)) if mask else (True, False, False)
    values = np.asarray(list(mask), dtype=bool).reshape(-1)
    if values.size < 3:
        values = np.pad(values, (0, 3 - values.size), constant_values=False)
    values = values[:3].copy()
    if not values.any():
        values[0] = True
    if stage == 1:
        values[2] = False
        if not values[:2].any():
            values[0] = True
    return values


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked, dim=-1)


def _tanh_log_prob(means: torch.Tensor, log_scale: torch.Tensor,
                   latent: torch.Tensor) -> torch.Tensor:
    scale = log_scale.clamp(-2.6, .3).exp()
    normal = torch.distributions.Normal(means, scale)
    jacobian = 2. * (np.log(2.) - latent - F.softplus(-2. * latent))
    return (normal.log_prob(latent) - jacobian).sum(-1)


@dataclass
class Decision:
    """A sampled skill and its bounded continuous motor command."""

    skill: int
    latent: np.ndarray
    action: np.ndarray
    log_probability: float
    value: float
    probabilities: np.ndarray

    @property
    def recheck(self) -> bool:
        """Compatibility alias used by the development runner."""
        return bool(self.skill > 0)

    @property
    def probability(self) -> float:
        return float(self.probabilities[min(1, len(self.probabilities) - 1)])


class ScanPolicy(nn.Module):
    """Shared belief encoder with separate high-level and stop heads."""

    def __init__(self, feature_count: int, hidden: int = 64):
        super().__init__()
        self.feature_count = int(feature_count)
        self.body = nn.Sequential(
            nn.Linear(self.feature_count, hidden), nn.Tanh(),
            nn.Linear(hidden, 48), nn.Tanh(),
        )
        self.skill_high = nn.Linear(48, 3)
        self.stop_head = nn.Linear(48, 2)
        # The two active skills get separate conditional motor distributions.
        # Slots 0:3 are window and 3:6 are revisit.
        self.control = nn.Linear(48, 6)
        self.value_high = nn.Linear(48, 1)
        self.value_stop = nn.Linear(48, 1)
        self.log_scale = nn.Parameter(torch.full((3,), -.45))
        nn.init.orthogonal_(self.skill_high.weight, gain=.08)
        nn.init.constant_(self.skill_high.bias, 0.)
        nn.init.orthogonal_(self.stop_head.weight, gain=.08)
        nn.init.constant_(self.stop_head.bias, 0.)
        nn.init.orthogonal_(self.control.weight, gain=.08)
        nn.init.zeros_(self.control.bias)
        self.gain_head = nn.Sequential(
            nn.Linear(48 + 3 + 2, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    @property
    def skill(self):
        """Legacy singular name for code that only inspected the skill head."""
        return self.skill_high

    @property
    def value(self):
        return self.value_high

    def forward(self, observations: torch.Tensor, stage=0):
        x = self.body(observations)
        stage_tensor = torch.as_tensor(stage, device=x.device)
        if stage_tensor.ndim == 0:
            stage_is_stop = bool(stage_tensor.item())
            logits = self.stop_head(x) if stage_is_stop else self.skill_high(x)
            value = self.value_stop(x) if stage_is_stop else self.value_high(x)
        else:
            stage_is_stop = stage_tensor.to(dtype=torch.bool).reshape(-1, 1)
            high_logits, stop_logits = self.skill_high(x), self.stop_head(x)
            logits = torch.where(stage_is_stop, F.pad(stop_logits, (0, 1), value=-30.), high_logits)
            high_value, stop_value = self.value_high(x), self.value_stop(x)
            value = torch.where(stage_is_stop, stop_value, high_value)
        if logits.shape[-1] == 2:
            means = torch.zeros((*logits.shape[:-1], 6), device=x.device, dtype=x.dtype)
        else:
            means = self.control(x)
        return logits, means, value.squeeze(-1)

    def predict_gain(self, observations, skills, actions):
        """Predict signed utility relative to continue for each active action."""
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.log_scale.device)
        if obs.ndim == 1:
            obs = obs[None]
        hidden = self.body(obs)
        skill = torch.as_tensor(skills, dtype=torch.long, device=obs.device).reshape(-1).clamp(0, 2)
        action = torch.as_tensor(actions, dtype=torch.float32, device=obs.device).reshape(-1, 3)
        one_hot = F.one_hot((skill - 1).clamp(0, 1), num_classes=2).to(obs.dtype)
        one_hot = torch.where((skill > 0)[:, None], one_hot, torch.zeros_like(one_hot))
        return self.gain_head(torch.cat((hidden, action, one_hot), dim=-1)).squeeze(-1)

    def _stage_tensors(self, observations, stage, masks):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.log_scale.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        n = obs.shape[0]
        stage_tensor = torch.as_tensor(stage, dtype=torch.long, device=obs.device)
        if stage_tensor.ndim == 0:
            stage_tensor = stage_tensor.expand(n)
        stage_tensor = stage_tensor.reshape(-1)
        if torch.is_tensor(masks):
            mask = masks.to(device=obs.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(n, -1)
            if mask.shape[-1] < 3:
                mask = F.pad(mask, (0, 3 - mask.shape[-1]), value=False)
            mask = mask[..., :3].clone()
            mask[:, 0] |= ~mask.any(dim=1)
            mask[stage_tensor == 1, 2] = False
        elif masks is None:
            mask_array = np.asarray([_normalise_mask(None, int(s)) for s in stage_tensor.detach().cpu().tolist()])
            mask = torch.as_tensor(mask_array, dtype=torch.bool, device=obs.device)
        else:
            if isinstance(masks, (bool, np.bool_)):
                masks = [masks] * n
            mask_array = np.asarray([_normalise_mask(m, int(s))
                                     for m, s in zip(masks, stage_tensor.detach().cpu().tolist())])
            mask = torch.as_tensor(mask_array, dtype=torch.bool, device=obs.device)
        logits, means, values = self(obs, stage_tensor)
        if logits.shape[-1] == 2:
            logits = F.pad(logits, (0, 1), value=-30.)
            mask[:, 2] = False
        return obs, stage_tensor, mask, logits, means, values

    def evaluate(self, observations, stage, masks, skills, latent):
        obs, stage_tensor, mask, logits, means, values = self._stage_tensors(observations, stage, masks)
        skills = torch.as_tensor(skills, dtype=torch.long, device=obs.device).reshape(-1).clamp(0, 2)
        latent = torch.as_tensor(latent, dtype=torch.float32, device=obs.device).reshape(-1, 3)
        probabilities = _masked_softmax(logits, mask)
        categorical = torch.log(probabilities.gather(1, skills[:, None]).clamp_min(1.e-8)).squeeze(1)
        # Select the conditional control distribution associated with the skill.
        control_index = (skills - 1).clamp(0, 1)[:, None] * 3
        offsets = control_index + torch.arange(3, device=obs.device)[None]
        selected_means = means.gather(1, offsets)
        motor = _tanh_log_prob(selected_means, self.log_scale, latent)
        active = (skills > 0) & (stage_tensor == 0)
        log_probability = categorical + active.to(categorical.dtype) * motor
        category_entropy = -(probabilities * torch.log(probabilities.clamp_min(1.e-8))
                              * mask.to(probabilities.dtype)).sum(-1)
        entropy = category_entropy + active.to(category_entropy.dtype) * (
            torch.distributions.Normal(selected_means, self.log_scale.clamp(-2.6, .3).exp()).entropy().sum(-1))
        return log_probability, values, entropy, probabilities

    def likelihood(self, observations, eligible=None, skill=None, latent=None, *, stage=0, masks=None):
        """Evaluate likelihood with a compatibility path for the old binary API."""
        if masks is None and eligible is not None:
            masks = [((True, bool(e), bool(e)) if int(stage) == 0 else (True, bool(e), False))
                     for e in np.asarray(eligible).reshape(-1)]
        if skill is None:
            skill = np.zeros(len(np.asarray(observations)), dtype=int)
        if latent is None:
            latent = np.zeros((len(np.asarray(observations)), 3), dtype=np.float32)
        return self.evaluate(observations, stage, masks, skill, latent)[:3]

    @torch.inference_mode()
    def choose(self, features, mask=True, rng=None, threshold=None, *, stage=0):
        actor = NumpyActor(self.state_dict())
        return actor.choose(features, mask, rng=rng, stage=stage, threshold=threshold)

    @torch.inference_mode()
    def stop(self, features, rng=None):
        return self.choose(features, (True, True, False), rng=rng, stage=1)


class NumpyActor:
    """Small allocation-free CPU actor reconstructed from policy weights."""

    def __init__(self, weights: Mapping):
        self.w = {}
        for name, value in weights.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            self.w[name] = np.asarray(value)
        self.log_scale = np.clip(self.w.get("log_scale", np.full(3, -.45)), -2.6, .3)

    def _linear(self, name, x):
        return x @ self.w[f"{name}.weight"].T + self.w[f"{name}.bias"]

    def _body(self, features):
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        x = np.tanh(self._linear("body.0", x))
        return np.tanh(self._linear("body.2", x))

    @staticmethod
    def _softmax(logits, mask):
        masked = np.where(mask, logits, -1.e9)
        masked -= np.max(masked)
        p = np.exp(masked)
        p[~mask] = 0.
        total = p.sum()
        return p / total if total > 0 else np.array([1., 0., 0.])

    def _decision(self, features, mask, rng, stage, threshold):
        h = self._body(features)
        stop = stage == 1
        raw = self._linear("stop_head", h) if stop else self._linear("skill_high", h)
        logits = np.full(3, -1.e9, dtype=np.float32)
        logits[:2 if stop else 3] = raw
        legal = _normalise_mask(mask, stage)
        probabilities = self._softmax(logits, legal)
        if rng is None:
            skill = int(np.argmax(probabilities))
        elif threshold is not None and stage == 1:
            skill = int(probabilities[1] >= float(threshold))
            if not legal[skill]:
                skill = 0
        else:
            skill = int(rng.choice(3, p=probabilities))
        latent = np.zeros(3, dtype=np.float32)
        means = np.zeros(3, dtype=np.float32)
        if not stop:
            all_means = self._linear("control", h).astype(np.float32)
            if all_means.size < 6:
                all_means = np.tile(all_means, 2)
            means = all_means[(skill - 1) * 3:skill * 3] if skill > 0 else np.zeros(3, np.float32)
            if skill > 0:
                latent = means + (rng.standard_normal(3).astype(np.float32)
                                  * np.exp(self.log_scale).astype(np.float32) if rng is not None else 0.)
        logp = float(np.log(max(probabilities[skill], 1.e-8)))
        if not stop and skill > 0:
            scale = np.exp(self.log_scale)
            log_normal = -.5 * (((latent - means) / scale) ** 2 + 2. * self.log_scale + np.log(2. * np.pi)).sum()
            jacobian = 2. * (np.log(2.) - latent - np.logaddexp(0., -2. * latent))
            logp += float(log_normal - jacobian.sum())
        value_name = "value_stop" if stop else "value_high"
        value = float(self._linear(value_name, h).reshape(-1)[0])
        return Decision(skill, latent, np.tanh(latent).astype(np.float32), logp, value,
                        probabilities.astype(np.float32))

    def choose(self, features, mask=True, rng=None, *, stage=0, threshold=None):
        return self._decision(features, mask, rng, int(stage), threshold)

    def stop(self, features, rng=None):
        return self._decision(features, (True, True, False), rng, 1, None)


def _normalise_step(step):
    stage = int(step.get("stage", 0))
    skill = _skill_index(step.get("skill", step.get("selected_skill", 0)))
    mask = _normalise_mask(step.get("mask", step.get("eligible", None)), stage)
    return {
        "features": np.asarray(step["features"], dtype=np.float32),
        "stage": stage,
        "mask": mask,
        "skill": skill,
        "latent": np.asarray(step.get("latent", np.zeros(3)), dtype=np.float32),
        "old_logp": float(step.get("old_logp", step.get("log_probability", 0.))),
        "value": float(step.get("value", 0.)),
        "reward": float(step.get("reward", 0.)),
        "ticks": int(step.get("ticks", 1)),
        "done": bool(step.get("done", False)),
    }


def update_ppo(policy: ScanPolicy, optimizer, episodes: Iterable[Sequence[dict]], rng=None,
               *, epochs=4, minibatch=256, discount=.985, gae_decay=.93, clip=.2):
    """Perform SMDP PPO. ``ticks`` supplies the between-decision discount."""
    raw_episodes = list(episodes)
    normalised_episodes = [[_normalise_step(step) for step in episode] for episode in raw_episodes]
    steps = [step for episode in normalised_episodes for step in episode]
    if not steps:
        return {"steps": 0, "eligible_steps": 0, "minibatch_updates": 0}
    advantages, returns = [], []
    for episode in normalised_episodes:
        next_value, gae = 0., 0.
        local_adv = []
        for step in reversed(episode):
            gamma = float(discount) ** max(0, int(step["ticks"]))
            continuation = 0. if step["done"] else 1.
            delta = step["reward"] + gamma * continuation * next_value - step["value"]
            gae = delta + gamma * float(gae_decay) * continuation * gae
            local_adv.append(gae)
            next_value = step["value"]
        local_adv.reverse()
        advantages.extend(local_adv)
        returns.extend(s["value"] + a for s, a in zip(episode, local_adv))
    device = policy.log_scale.device
    features = torch.as_tensor(np.asarray([s["features"] for s in steps]), dtype=torch.float32, device=device)
    stage = torch.as_tensor([s["stage"] for s in steps], dtype=torch.long, device=device)
    masks = torch.as_tensor(np.asarray([s["mask"] for s in steps]), dtype=torch.bool, device=device)
    skill = torch.as_tensor([s["skill"] for s in steps], dtype=torch.long, device=device)
    latent = torch.as_tensor(np.asarray([s["latent"] for s in steps]), dtype=torch.float32, device=device)
    old_logp = torch.as_tensor([s["old_logp"] for s in steps], dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(.01)
    n = len(steps)
    metrics = []
    for _ in range(int(epochs)):
        order = torch.randperm(n, device=device)
        for index in order.split(int(minibatch)):
            logp, values, entropy, _ = policy.evaluate(features[index], stage[index], masks[index],
                                                       skill[index], latent[index])
            ratio = (logp - old_logp[index]).exp().clamp(1.e-5, 1.e5)
            advantage = advantages[index]
            surrogate = torch.minimum(ratio * advantage,
                                      ratio.clamp(1. - clip, 1. + clip) * advantage)
            actor_loss = -surrogate.mean()
            critic_loss = F.smooth_l1_loss(values, returns[index])
            entropy_loss = entropy.mean()
            loss = actor_loss + .55 * critic_loss - .013 * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), .7)
            optimizer.step()
            metrics.append(torch.stack((actor_loss.detach(), critic_loss.detach(),
                                        entropy_loss.detach(), (old_logp[index] - logp).mean().detach(),
                                        grad_norm.detach())))
    if metrics:
        summary = torch.stack(metrics).mean(dim=0).detach().cpu().numpy()
    else:
        summary = np.zeros(5)
    # One compact transfer at the end covers all scalar counters.
    counters = torch.cat((masks[:, 1:].any(dim=1).sum().reshape(1),
                          torch.stack([(skill == i).sum() for i in range(3)]))).detach().cpu().tolist()
    return {"steps": n, "eligible_steps": int(counters[0]),
            "skill_counts": [int(value) for value in counters[1:]],
            "minibatch_updates": len(metrics), "actor_loss": float(summary[0]),
            "critic_loss": float(summary[1]), "entropy": float(summary[2]),
            "approximate_kl": float(summary[3]), "gradient_norm": float(summary[4]),
            "optimizer_cuda": bool(any(p.is_cuda for group in optimizer.param_groups for p in group["params"]))}
