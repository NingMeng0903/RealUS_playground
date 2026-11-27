"""Small masked recurrent PPO and discrete CQL learners.

The module intentionally has no environment or simulator dependencies.  The
policy consumes only the public state feature vector and a legal-action mask;
the caller is responsible for constructing belief/history features and for
computing returns/advantages.  A rollout prefix (for example, the common
prescan) can simply be concatenated to a training sequence and excluded from
``train_mask``.  Calling :meth:`ActorCritic.sequence` with ``h=None`` resets
the GRU at an episode boundary, while passing the returned hidden state warms
the next chunk.

The implementation is deliberately compact so it can be used as a baseline in
the topology experiment without importing the simulator.  Invalid masks are
handled defensively: an all-false action mask is interpreted as "no mask" for
that row, and malformed logged actions are excluded from the PPO/CQL loss
instead of causing a categorical-distribution failure.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_


__all__ = ["ActorCritic", "QNetwork", "ppo_update", "cql_update"]


def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
    """Return the device/dtype used by the module's first parameter."""

    parameter = next(module.parameters(), None)
    if parameter is None:  # pragma: no cover - all classes below have params
        return torch.device("cpu"), torch.get_default_dtype()
    return parameter.device, parameter.dtype


def _as_float_tensor(value: Tensor, module: nn.Module) -> Tensor:
    """Move an input to the module's device and floating-point dtype."""

    device, dtype = _module_device_dtype(module)
    if not torch.is_floating_point(value):
        value = value.float()
    return value.to(device=device, dtype=dtype)


def _bool_mask(mask: Tensor, *, device: torch.device, shape: Tuple[int, ...]) -> Tensor:
    """Convert and validate a legal-action mask."""

    mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(f"legal mask has shape {tuple(mask.shape)}, expected {shape}")
    # A padded row is sometimes represented by an all-false mask.  Keeping a
    # valid categorical distribution is safer than producing NaNs; callers
    # still control whether such a row contributes through train_mask.
    any_legal = mask.any(dim=-1, keepdim=True)
    return torch.where(any_legal, mask, torch.ones_like(mask))


def _masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """Set illegal action logits to a finite, dtype-safe negative value."""

    # ``finfo.min`` works for float16 as well as float32/64, unlike a literal
    # ``-1e9`` which cannot be represented by every floating dtype.
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class ActorCritic(nn.Module):
    """A small masked actor-critic with an optional one-layer GRU.

    Parameters
    ----------
    state_dim:
        Number of public state features.
    action_count:
        Number of discrete actions represented by the legal mask.
    hidden:
        Width of the state MLP, GRU and policy/value heads.
    recurrent:
        If false, use the same MLP and heads without a GRU (the intended
        no-memory ablation).
    """

    def __init__(
        self,
        state_dim: int,
        action_count: int,
        hidden: int = 96,
        recurrent: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_count <= 0 or hidden <= 0:
            raise ValueError("state_dim, action_count and hidden must be positive")
        self.state_dim = int(state_dim)
        self.action_count = int(action_count)
        self.hidden = int(hidden)
        self.recurrent = bool(recurrent)

        # Two small nonlinearities are enough for the baseline and make the
        # recurrent/no-recurrent comparison differ only in memory.
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
        )
        self.gru = nn.GRU(self.hidden, self.hidden, batch_first=True) if self.recurrent else None
        self.policy_head = nn.Linear(self.hidden, self.action_count)
        self.value_head = nn.Linear(self.hidden, 1)

    def _hidden(self, h: Optional[Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Normalize a caller hidden state to GRU's ``[1, B, H]`` format."""

        if h is None:
            return torch.zeros(1, batch_size, self.hidden, device=device, dtype=dtype)
        h = torch.as_tensor(h, device=device, dtype=dtype)
        if h.ndim == 2:
            h = h.unsqueeze(0)
        if h.ndim != 3 or h.shape[0] != 1 or h.shape[1] != batch_size or h.shape[2] != self.hidden:
            raise ValueError(
                "hidden state must have shape [B, hidden] or [1, B, hidden], "
                f"got {tuple(h.shape)} for B={batch_size}, hidden={self.hidden}"
            )
        return h

    def step(
        self,
        x: Tensor,
        mask: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Evaluate one step.

        ``x`` is ``[B, F]`` and ``mask`` is ``[B, A]``.  The returned logits
        already contain the action mask, values are ``[B]``, and recurrent
        hidden state is ``[1, B, hidden]`` (or ``None`` in the ablation).
        """

        if x.ndim != 2:
            raise ValueError(f"step expects x[B,F], got {tuple(x.shape)}")
        if x.shape[-1] != self.state_dim:
            raise ValueError(f"expected {self.state_dim} state features, got {x.shape[-1]}")
        x = _as_float_tensor(x, self)
        legal = _bool_mask(mask, device=x.device, shape=(x.shape[0], self.action_count))

        features = self.encoder(x)
        next_h: Optional[Tensor]
        if self.recurrent:
            assert self.gru is not None
            h0 = self._hidden(h, x.shape[0], x.device, features.dtype)
            sequence_features, next_h = self.gru(features.unsqueeze(1), h0)
            features = sequence_features[:, 0]
        else:
            # A no-GRU ablation has no recurrent state.  Ignore a supplied
            # hidden state rather than accidentally carrying information.
            next_h = None

        logits = _masked_logits(self.policy_head(features), legal)
        values = self.value_head(features).squeeze(-1)
        return logits, values, next_h

    def sequence(
        self,
        x: Tensor,
        mask: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Evaluate a full sequence.

        ``x`` is ``[B, T, F]`` and ``mask`` is ``[B, T, A]``.  A fresh call
        with ``h=None`` resets every sequence.  To warm a recurrent policy,
        prepend the public prefix to ``x`` and use a ``train_mask`` that only
        includes the post-prefix frames in the loss.
        """

        if x.ndim != 3:
            raise ValueError(f"sequence expects x[B,T,F], got {tuple(x.shape)}")
        if x.shape[-1] != self.state_dim:
            raise ValueError(f"expected {self.state_dim} state features, got {x.shape[-1]}")
        x = _as_float_tensor(x, self)
        batch_size, time_steps = x.shape[:2]
        legal = _bool_mask(
            mask,
            device=x.device,
            shape=(batch_size, time_steps, self.action_count),
        )

        features = self.encoder(x.reshape(batch_size * time_steps, self.state_dim))
        features = features.reshape(batch_size, time_steps, self.hidden)
        next_h: Optional[Tensor]
        if self.recurrent:
            assert self.gru is not None
            h0 = self._hidden(h, batch_size, x.device, features.dtype)
            features, next_h = self.gru(features, h0)
        else:
            next_h = None

        logits = _masked_logits(self.policy_head(features), legal)
        values = self.value_head(features).squeeze(-1)
        return logits, values, next_h

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Alias for :meth:`step` for convenient policy use."""

        return self.step(x, mask, h)


class QNetwork(nn.Module):
    """A small discrete-action Q network."""

    def __init__(self, state_dim: int, action_count: int, hidden: int = 128) -> None:
        super().__init__()
        if state_dim <= 0 or action_count <= 0 or hidden <= 0:
            raise ValueError("state_dim, action_count and hidden must be positive")
        self.state_dim = int(state_dim)
        self.action_count = int(action_count)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.action_count),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.state_dim:
            raise ValueError(f"expected {self.state_dim} state features, got {x.shape[-1]}")
        if x.ndim < 2:
            raise ValueError(f"QNetwork expects at least [B,F], got {tuple(x.shape)}")
        x = _as_float_tensor(x, self)
        leading = x.shape[:-1]
        q = self.net(x.reshape(-1, self.state_dim))
        return q.reshape(*leading, self.action_count)


def _zero_ppo_stats() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "valid_frames": 0.0,
        "updates": 0.0,
    }


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    legal: Tensor,
    actions: Tensor,
    old_logp: Tensor,
    returns: Tensor,
    advantages: Tensor,
    train_mask: Tensor,
    *,
    epochs: int = 4,
    batch: int = 64,
    clip: float = 0.2,
    entropy: float = 0.02,
) -> Dict[str, float]:
    """Run masked PPO updates over whole recurrent sequences.

    ``batch`` is the number of complete sequences in a minibatch.  Time is
    never flattened, so each GRU call starts at a sequence boundary and sees
    the complete prefix/history.  ``train_mask`` can exclude a warm-up prefix
    and padded frames.  Logged actions that are out of range or illegal are
    ignored rather than crashing a categorical distribution.
    """

    if not isinstance(model, ActorCritic):
        raise TypeError("model must be an ActorCritic")
    if x.ndim != 3:
        raise ValueError(f"ppo_update expects x[B,T,F], got {tuple(x.shape)}")
    if x.shape[-1] != model.state_dim:
        raise ValueError(f"expected {model.state_dim} state features, got {x.shape[-1]}")
    batch_size, time_steps = x.shape[:2]
    if tuple(legal.shape) != (batch_size, time_steps, model.action_count):
        raise ValueError("legal must have shape [B,T,action_count]")
    expected_bt = (batch_size, time_steps)
    for name, value in (
        ("actions", actions),
        ("old_logp", old_logp),
        ("returns", returns),
        ("advantages", advantages),
        ("train_mask", train_mask),
    ):
        if tuple(value.shape) != expected_bt:
            raise ValueError(f"{name} must have shape [B,T], got {tuple(value.shape)}")

    stats = _zero_ppo_stats()
    if batch_size == 0 or time_steps == 0 or int(epochs) <= 0:
        return stats

    device, dtype = _module_device_dtype(model)
    legal_raw = torch.as_tensor(legal, device=device, dtype=torch.bool)
    legal_safe = _bool_mask(
        legal_raw,
        device=device,
        shape=(batch_size, time_steps, model.action_count),
    )
    has_legal = legal_raw.any(dim=-1)
    action_long = torch.as_tensor(actions, device=device, dtype=torch.long)
    action_in_range = (action_long >= 0) & (action_long < model.action_count)
    action_safe = action_long.clamp(0, model.action_count - 1)
    action_legal = legal_safe.gather(-1, action_safe.unsqueeze(-1)).squeeze(-1)
    learning_mask = torch.as_tensor(train_mask, device=device, dtype=torch.bool)
    # An all-false mask is retained as a finite distribution for forward
    # evaluation, but it is not a learnable action row (it is normally a
    # padded frame).
    learning_mask = learning_mask & has_legal & action_in_range & action_legal

    # Normalize once across all valid learning frames, not independently per
    # minibatch.  This is important when a batch contains variable-length
    # episodes or a long warm-up prefix.
    adv = torch.as_tensor(advantages, device=device, dtype=dtype)
    valid_adv = adv[learning_mask]
    if valid_adv.numel() == 0:
        return stats
    adv_mean = valid_adv.mean()
    adv_std = valid_adv.std(unbiased=False)
    adv_norm = (adv - adv_mean) / (adv_std + 1e-8)

    old_logp_t = torch.as_tensor(old_logp, device=device, dtype=dtype).detach()
    returns_t = torch.as_tensor(returns, device=device, dtype=dtype)
    x_t = torch.as_tensor(x, device=device)
    # ``_as_float_tensor`` in model.sequence performs the final dtype cast;
    # keeping x_t's original dtype here avoids an unnecessary copy for float
    # inputs while still accepting numpy/integer tensors.
    legal_t = legal_raw

    model.train()
    sequence_batch = max(1, int(batch))
    n_epochs = max(0, int(epochs))
    sum_policy = sum_value = sum_entropy = sum_loss = 0.0
    update_count = 0
    valid_count = int(learning_mask.sum().item())

    for _ in range(n_epochs):
        order = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, sequence_batch):
            indices = order[start : start + sequence_batch]
            # The mask is intentionally not used to reset h inside a sequence:
            # callers can place all episode/prescan prefixes in separate rows,
            # and padded rows only affect the loss through learning_mask.
            logits, values, _ = model.sequence(x_t[indices], legal_t[indices], h=None)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(action_safe[indices])
            entropy_term = dist.entropy()

            valid = learning_mask[indices]
            # Index valid frames before arithmetic so arbitrary padded values
            # (including NaN placeholders in returns/log-probs) cannot leak
            # through a ``0 * NaN`` multiplication into the loss.
            valid_flat = valid.reshape(-1)
            if not bool(valid_flat.any()):
                continue
            new_logp_valid = new_logp.reshape(-1)[valid_flat]
            old_logp_valid = old_logp_t[indices].reshape(-1)[valid_flat]
            ratio = torch.exp((new_logp_valid - old_logp_valid).clamp(-20.0, 20.0))
            clipped_ratio = ratio.clamp(1.0 - float(clip), 1.0 + float(clip))
            advantages_batch = adv_norm[indices].reshape(-1)[valid_flat]
            surrogate = torch.minimum(ratio * advantages_batch, clipped_ratio * advantages_batch)
            policy_loss = -surrogate.mean()
            # Huber limits the impact of an occasional large bootstrap target;
            # gradient norm clipping below protects both actor and critic.
            value_error = F.smooth_l1_loss(
                values.reshape(-1)[valid_flat],
                returns_t[indices].reshape(-1)[valid_flat],
                reduction="mean",
                beta=1.0,
            )
            value_loss = 0.5 * value_error
            entropy_mean = entropy_term.reshape(-1)[valid_flat].mean()
            total = policy_loss + value_loss - float(entropy) * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            sum_policy += float(policy_loss.detach().cpu())
            sum_value += float(value_loss.detach().cpu())
            sum_entropy += float(entropy_mean.detach().cpu())
            sum_loss += float(total.detach().cpu())
            update_count += 1

    if update_count:
        inv_updates = 1.0 / update_count
        stats.update(
            loss=sum_loss * inv_updates,
            policy_loss=sum_policy * inv_updates,
            value_loss=sum_value * inv_updates,
            entropy=sum_entropy * inv_updates,
            valid_frames=float(valid_count),
            updates=float(update_count),
        )
    return stats


def _zero_cql_stats() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "td_loss": 0.0,
        "cql_penalty": 0.0,
        "q_taken": 0.0,
        "target_q": 0.0,
        "valid": 0.0,
    }


def cql_update(
    model: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    x: Tensor,
    mask: Tensor,
    a: Tensor,
    r: Tensor,
    next_x: Tensor,
    next_mask: Tensor,
    done: Tensor,
    *,
    alpha: float = 0.2,
    gamma: float = 0.99,
) -> Dict[str, float]:
    """Perform one masked discrete CQL TD update.

    The replay source is deliberately external: this function only consumes a
    fixed logged batch.  ``mask`` and ``next_mask`` constrain both the sampled
    action value and the log-sum-exp conservative term.  Rows with malformed
    logged actions are skipped safely.
    """

    if not isinstance(model, QNetwork) or not isinstance(target, QNetwork):
        raise TypeError("model and target must be QNetwork instances")
    if tuple(mask.shape) != (x.shape[0], model.action_count):
        raise ValueError("mask must have shape [B, action_count]")
    if tuple(next_mask.shape) != (next_x.shape[0], model.action_count):
        raise ValueError("next_mask must have shape [B, action_count]")
    if x.ndim != 2 or next_x.ndim != 2:
        raise ValueError("x and next_x must have shape [B,F]")
    n = x.shape[0]
    if next_x.shape[0] != n:
        raise ValueError("x and next_x must have the same batch size")

    stats = _zero_cql_stats()
    if n == 0:
        return stats

    device, dtype = _module_device_dtype(model)
    x_t = torch.as_tensor(x, device=device)
    next_x_t = torch.as_tensor(next_x, device=device)
    mask_raw = torch.as_tensor(mask, device=device, dtype=torch.bool)
    next_mask_raw = torch.as_tensor(next_mask, device=device, dtype=torch.bool)
    mask_safe = _bool_mask(mask_raw, device=device, shape=(n, model.action_count))
    next_mask_safe = _bool_mask(next_mask_raw, device=device, shape=(n, model.action_count))
    has_legal = mask_raw.any(dim=-1)
    next_has_legal = next_mask_raw.any(dim=-1)
    action = torch.as_tensor(a, device=device, dtype=torch.long).reshape(-1)
    reward = torch.as_tensor(r, device=device, dtype=dtype).reshape(-1)
    terminal = torch.as_tensor(done, device=device, dtype=dtype).reshape(-1)
    if action.numel() != n or reward.numel() != n or terminal.numel() != n:
        raise ValueError("a, r and done must each contain B values")
    action_in_range = (action >= 0) & (action < model.action_count)
    action_safe = action.clamp(0, model.action_count - 1)
    action_legal = mask_safe.gather(-1, action_safe.unsqueeze(-1)).squeeze(-1)
    valid = has_legal & action_in_range & action_legal
    if not bool(valid.any()):
        return stats

    model.train()
    q_all = model(x_t)
    with torch.no_grad():
        q_next = target(next_x_t)
        q_next_masked = _masked_logits(q_next, next_mask_safe)
        next_best = q_next_masked.max(dim=-1).values
        # There is no valid bootstrap action for a padded next state.  Treat
        # it as a zero continuation rather than accidentally selecting from
        # the all-false-mask fallback distribution.
        next_best = torch.where(next_has_legal, next_best, torch.zeros_like(next_best))
        target_q_all = reward + float(gamma) * (1.0 - terminal.clamp(0.0, 1.0)) * next_best

    q_taken_all = q_all.gather(-1, action_safe.unsqueeze(-1)).squeeze(-1)
    q_taken = q_taken_all[valid]
    target_q = target_q_all[valid]
    q_masked = _masked_logits(q_all, mask_safe)
    conservative = torch.logsumexp(q_masked[valid], dim=-1) - q_taken
    cql_penalty = conservative.mean()
    td_loss = F.smooth_l1_loss(q_taken, target_q)
    total = td_loss + float(alpha) * cql_penalty

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    stats.update(
        loss=float(total.detach().cpu()),
        td_loss=float(td_loss.detach().cpu()),
        cql_penalty=float(cql_penalty.detach().cpu()),
        q_taken=float(q_taken.detach().mean().cpu()),
        target_q=float(target_q.detach().mean().cpu()),
        valid=float(valid.sum().item()),
    )
    return stats

