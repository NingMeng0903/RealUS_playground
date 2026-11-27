"""Teacher-behaviour cloning for a public-history topology policy.

The teacher is obtained only through ``episode.actions(method=teacher)``.
Features, legal masks and the five-frame public prescan prefix are retained;
private labels and evaluator outcomes never enter the policy input.  Each
trajectory is padded only after its terminal step, and cross-entropy is
computed on valid teacher actions in the unpadded suffix.  Two stochastic
tapes per selected case are collected by default so a reset-sampled nuisance
field is represented in the fixed training set.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

try:
    from .learner import ActorCritic
    from .train import _device, evaluate
except ImportError:  # Direct import from work/topology_b.
    from learner import ActorCritic
    from train import _device, evaluate


__all__ = ["clone_policy"]

REPEATS = 2
MAX_STEPS = 128
BATCH_SIZE = 256
EPOCHS = 32


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except RuntimeError:
        pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True), encoding="utf-8")


def _as_prefix(episode: Any, feature_dim: int) -> np.ndarray:
    value = np.asarray(episode.prefix_features(), dtype=np.float32)
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[-1] != feature_dim:
        raise ValueError(f"expected prefix [B,T,{feature_dim}], got {tuple(value.shape)}")
    return value


def _as_current(episode: Any, feature_dim: int, action_count: int) -> Tuple[np.ndarray, np.ndarray]:
    features, legal = episode.features()
    features = np.asarray(features, dtype=np.float32)
    legal = np.asarray(legal, dtype=bool)
    if features.ndim == 1:
        features = features[None]
    if legal.ndim == 1:
        legal = legal[None]
    if features.ndim != 2 or features.shape[-1] != feature_dim:
        raise ValueError(f"expected features [B,{feature_dim}], got {tuple(features.shape)}")
    if legal.shape != (features.shape[0], action_count):
        raise ValueError(f"expected legal [B,{action_count}], got {tuple(legal.shape)}")
    return features, legal


def _teacher_actions(episode: Any, teacher: str, legal: np.ndarray, action_count: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        actions = np.asarray(episode.actions(method=teacher), dtype=np.int64).reshape(-1)
    except TypeError:
        actions = np.asarray(episode.actions(teacher), dtype=np.int64).reshape(-1)
    batch_size = legal.shape[0]
    if actions.size == 1 and batch_size != 1:
        actions = np.repeat(actions, batch_size)
    if actions.size != batch_size:
        raise ValueError("episode.actions(method=teacher) must return [B]")
    in_range = (actions >= 0) & (actions < action_count)
    safe = np.clip(actions, 0, action_count - 1)
    valid = in_range & legal[np.arange(batch_size), safe]
    # A malformed teacher action still needs a public executable fallback to
    # advance the episode.  It is excluded from the imitation loss by valid.
    fallback = safe.copy()
    for row in np.flatnonzero(~valid):
        choices = np.flatnonzero(legal[row])
        fallback[row] = int(choices[0]) if choices.size else action_count - 1
    return fallback.astype(np.int64), valid.astype(bool)


def _collect_teacher(
    env: Any,
    *,
    case_ids: np.ndarray,
    feature_dim: int,
    action_count: int,
    teacher: str,
    repeats: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Collect complete public teacher sequences, preserving prefix frames."""

    trajectories: List[Dict[str, Any]] = []
    counters = {"requested": 0, "complete": 0, "truncated": 0, "invalid_teacher": 0, "valid_frames": 0}
    case_ids = np.asarray(case_ids, dtype=np.int64).reshape(-1)
    if case_ids.size == 0:
        return trajectories, counters
    for repeat in range(int(repeats)):
        for start in range(0, case_ids.size, min(BATCH_SIZE, case_ids.size)):
            ids = case_ids[start : start + min(BATCH_SIZE, case_ids.size)]
            episode = env.reset(ids)
            prefix = _as_prefix(episode, feature_dim)
            if prefix.shape[0] != ids.size:
                raise ValueError("env.reset returned a different prefix batch size")
            done = np.asarray(getattr(episode, "done"), dtype=bool).reshape(-1)
            if done.size == 1 and ids.size != 1:
                done = np.repeat(done, ids.size)
            if done.size != ids.size:
                raise ValueError("episode.done has a different batch size")
            rows: List[Dict[str, Any]] = [
                {"case_id": int(case_id), "repeat": repeat, "prefix": prefix[row].copy(),
                 "x": [], "legal": [], "actions": [], "valid": []}
                for row, case_id in enumerate(ids)
            ]
            counters["requested"] += int(ids.size)
            for _ in range(MAX_STEPS):
                if bool(done.all()):
                    break
                features, legal = _as_current(episode, feature_dim, action_count)
                actions, valid = _teacher_actions(episode, teacher, legal, action_count)
                active = ~done
                for row in np.flatnonzero(active):
                    rows[row]["x"].append(features[row].copy())
                    rows[row]["legal"].append(legal[row].copy())
                    rows[row]["actions"].append(int(actions[row]))
                    rows[row]["valid"].append(bool(valid[row]))
                    if not valid[row]:
                        counters["invalid_teacher"] += 1
                # Abstention is the public no-op for rows that already ended.
                step_actions = np.where(active, actions, action_count - 1).astype(np.int64)
                _, done_after = episode.step(step_actions)
                done_after = np.asarray(done_after, dtype=bool).reshape(-1)
                if done_after.size == 1 and ids.size != 1:
                    done_after = np.repeat(done_after, ids.size)
                if done_after.size != ids.size:
                    raise ValueError("episode.step returned a different done batch size")
                done = done_after | done
            for row_index, row in enumerate(rows):
                row["complete"] = bool(done[row_index])
                valid_count = int(np.asarray(row["valid"], dtype=bool).sum())
                if row["complete"] and valid_count:
                    counters["complete"] += 1
                    counters["valid_frames"] += valid_count
                    trajectories.append(row)
                else:
                    counters["truncated"] += 1
    return trajectories, counters


def _pad_dataset(
    trajectories: Sequence[Dict[str, Any]],
    *,
    feature_dim: int,
    action_count: int,
    repeats: int,
) -> Dict[str, Tensor]:
    if not trajectories:
        raise ValueError("teacher collection produced no complete valid trajectories")
    prefix_len = int(np.asarray(trajectories[0]["prefix"]).shape[0])
    max_suffix = max(len(row["x"]) for row in trajectories)
    batch_size = len(trajectories)
    time_steps = prefix_len + max_suffix
    x = torch.zeros(batch_size, time_steps, feature_dim, dtype=torch.float32)
    legal = torch.zeros(batch_size, time_steps, action_count, dtype=torch.bool)
    actions = torch.zeros(batch_size, time_steps, dtype=torch.long)
    train_mask = torch.zeros(batch_size, time_steps, dtype=torch.bool)
    weights = torch.zeros(batch_size, time_steps, dtype=torch.float32)
    for row_index, row in enumerate(trajectories):
        prefix = np.asarray(row["prefix"], dtype=np.float32)
        plen = prefix.shape[0]
        x[row_index, :plen] = torch.from_numpy(prefix)
        legal[row_index, :plen] = True
        suffix_len = len(row["x"])
        if suffix_len:
            x[row_index, plen : plen + suffix_len] = torch.from_numpy(
                np.asarray(row["x"], dtype=np.float32)
            )
            legal[row_index, plen : plen + suffix_len] = torch.from_numpy(
                np.asarray(row["legal"], dtype=bool)
            )
            actions[row_index, plen : plen + suffix_len] = torch.as_tensor(row["actions"], dtype=torch.long)
            valid = torch.as_tensor(row["valid"], dtype=torch.bool)
            train_mask[row_index, plen : plen + suffix_len] = valid
            valid_count = int(valid.sum())
            if valid_count:
                # Fixed per-case weighting: the two reset tapes together give
                # each selected case equal total weight, independent of its
                # trajectory length.
                weights[row_index, plen : plen + suffix_len] = valid.to(torch.float32) / float(
                    max(1, int(repeats)) * valid_count
                )
    return {"x": x, "legal": legal, "actions": actions, "train_mask": train_mask, "weights": weights}


def _evaluate_dataset(model: ActorCritic, dataset: Dict[str, Tensor], device: torch.device) -> Dict[str, float]:
    x = dataset["x"].to(device)
    legal = dataset["legal"].to(device)
    actions = dataset["actions"].to(device)
    valid = dataset["train_mask"].to(device)
    weights = dataset["weights"].to(device)
    if not bool(valid.any()):
        return {"loss": 0.0, "accuracy": 0.0, "frames": 0.0}
    model.eval()
    with torch.no_grad():
        logits, _, _ = model.sequence(x, legal, h=None)
        logits_flat = logits.reshape(-1, model.action_count)
        actions_flat = actions.reshape(-1)
        valid_flat = valid.reshape(-1)
        ce = F.cross_entropy(logits_flat[valid_flat], actions_flat[valid_flat], reduction="none")
        weight = weights.reshape(-1)[valid_flat]
        loss = (ce * weight).sum() / weight.sum().clamp_min(1e-8)
        prediction = logits_flat[valid_flat].argmax(dim=-1)
        accuracy = (prediction == actions_flat[valid_flat]).to(torch.float32).mean()
    return {"loss": float(loss.cpu()), "accuracy": float(accuracy.cpu()), "frames": float(valid.sum().item())}


def _env_metrics(outcomes: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Report the frozen public dev objective used for policy comparison."""

    if not outcomes:
        return {"n": 0.0, "success": 0.0, "wrong": 0.0, "tcp_mm": 0.0,
                "turn_deg": 0.0, "rail_mm": 0.0, "captures": 0.0, "objective": 0.0}

    def mean(name: str) -> float:
        return float(np.mean([float(row.get(name, 0.0) or 0.0) for row in outcomes]))

    success = mean("success")
    wrong = mean("wrong")
    tcp_mm = mean("tcp_mm")
    turn_deg = mean("turn_deg")
    rail_mm = mean("rail_mm")
    captures = mean("captures")
    objective = success - 1.5 * wrong - 1e-3 * tcp_mm - 5e-4 * turn_deg - 1e-4 * rail_mm - 1.2e-2 * captures
    return {
        "n": float(len(outcomes)),
        "success": success,
        "wrong": wrong,
        "tcp_mm": tcp_mm,
        "turn_deg": turn_deg,
        "rail_mm": rail_mm,
        "captures": captures,
        "objective": float(objective),
    }


def _fit(
    model: ActorCritic,
    train_data: Dict[str, Tensor],
    dev_data: Optional[Dict[str, Tensor]],
    *,
    device: torch.device,
    epochs: int = EPOCHS,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    x = train_data["x"].to(device)
    legal = train_data["legal"].to(device)
    actions = train_data["actions"].to(device)
    valid = train_data["train_mask"].to(device)
    weights = train_data["weights"].to(device)
    count = x.shape[0]
    train_curve: List[Dict[str, float]] = []
    dev_curve: List[Dict[str, float]] = []
    if count == 0 or not bool(valid.any()):
        return train_curve, dev_curve
    batch = min(64, count)
    for epoch in range(1, max(0, int(epochs)) + 1):
        model.train()
        order = torch.randperm(count, device=device)
        total_loss = total_correct = total_frames = 0.0
        for start in range(0, count, batch):
            indices = order[start : start + batch]
            logits, _, _ = model.sequence(x[indices], legal[indices], h=None)
            valid_flat = valid[indices].reshape(-1)
            if not bool(valid_flat.any()):
                continue
            logits_flat = logits.reshape(-1, model.action_count)[valid_flat]
            actions_flat = actions[indices].reshape(-1)[valid_flat]
            frame_weights = weights[indices].reshape(-1)[valid_flat]
            ce = F.cross_entropy(logits_flat, actions_flat, reduction="none")
            loss = (ce * frame_weights).sum() / frame_weights.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                total_loss += float((ce * frame_weights).sum().cpu())
                total_correct += float((logits_flat.argmax(dim=-1) == actions_flat).sum().cpu())
                total_frames += float(valid_flat.sum().item())
        train_metrics = {
            "epoch": float(epoch),
            "loss": total_loss / max(1e-8, float(weights[valid].sum().detach().cpu())),
            "accuracy": total_correct / max(1.0, total_frames),
            "frames": total_frames,
        }
        train_curve.append(train_metrics)
        if dev_data is not None:
            metrics = _evaluate_dataset(model, dev_data, device)
            metrics["epoch"] = float(epoch)
            dev_curve.append(metrics)
    return train_curve, dev_curve


def clone_policy(
    env: Any,
    dev: Any,
    *,
    seed: int,
    cases: int,
    output: Path,
    recurrent: bool,
    device: str = "cpu",
    teacher: str = "mpc",
) -> Tuple[ActorCritic, Dict[str, Any]]:
    """Clone a public teacher into a recurrent or MLP actor-critic policy."""

    _seed_everything(seed)
    torch_device = _device(device)
    n_cases = int(getattr(env, "n_cases"))
    feature_dim = int(getattr(env, "feature_dim"))
    action_count = int(getattr(env, "action_count"))
    if n_cases <= 0 or feature_dim <= 0 or action_count <= 0:
        raise ValueError("env must expose positive n_cases, feature_dim and action_count")
    case_count = min(max(1, int(cases)), n_cases)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Fixed case order/tape count makes recurrent and no-GRU calls comparable
    # when the caller reuses seed, env and cases.  A deterministic teacher env
    # already has one fixed tape per case; only a reset-stochastic env needs a
    # second tape.
    repeats = REPEATS if bool(getattr(env, "stochastic", False)) else 1
    case_ids = np.arange(case_count, dtype=np.int64)
    train_rows, train_counts = _collect_teacher(
        env,
        case_ids=case_ids,
        feature_dim=feature_dim,
        action_count=action_count,
        teacher=teacher,
        repeats=repeats,
        device=torch_device,
    )
    if not train_rows:
        raise RuntimeError("teacher collection produced no complete trajectories")
    train_data = _pad_dataset(
        train_rows,
        feature_dim=feature_dim,
        action_count=action_count,
        repeats=repeats,
    )

    dev_rows: Optional[List[Dict[str, Any]]] = None
    dev_data: Optional[Dict[str, Tensor]] = None
    dev_counts: Dict[str, int] = {}
    if dev is not None:
        dev_n = min(case_count, int(getattr(dev, "n_cases")))
        if dev_n > 0:
            dev_rows, dev_counts = _collect_teacher(
                dev,
                case_ids=np.arange(dev_n, dtype=np.int64),
                feature_dim=feature_dim,
                action_count=action_count,
                teacher=teacher,
                repeats=1,
                device=torch_device,
            )
            if dev_rows:
                dev_data = _pad_dataset(
                    dev_rows,
                    feature_dim=feature_dim,
                    action_count=action_count,
                    repeats=1,
                )

    model = ActorCritic(feature_dim, action_count, recurrent=bool(recurrent)).to(torch_device)
    train_curve, dev_curve = _fit(model, train_data, dev_data, device=torch_device)
    train_quality = _evaluate_dataset(model, train_data, torch_device)
    dev_quality = dev_curve[-1] if dev_curve else ({"loss": 0.0, "accuracy": 0.0, "frames": 0.0})
    dev_env_quality: Dict[str, Any] = {}
    if dev is not None:
        try:
            dev_env_quality = _env_metrics(evaluate(model, dev, batch=128))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # Teacher-frame quality remains available even for a lightweight
            # mock dev object that does not implement evaluator outcomes.
            dev_env_quality = {"error": str(exc)}
    checkpoint = output / "clone_policy.pt"
    payload = {
        "model": model.state_dict(),
        "config": {
            "kind": "imitation",
            "state_dim": feature_dim,
            "action_count": action_count,
            "hidden": int(model.hidden),
            "recurrent": bool(recurrent),
            "teacher": str(teacher),
        },
    }
    torch.save(payload, checkpoint)
    summary: Dict[str, Any] = {
        "kind": "imitation",
        "seed": int(seed),
        "teacher": str(teacher),
        "recurrent": bool(recurrent),
        "device": str(torch_device),
        "cases": int(case_count),
        "repeats": repeats,
        "max_steps": MAX_STEPS,
        "weighting": "each selected case contributes equal total weight across its reset tapes; valid frame weight=1/(repeats*valid_frames_in_tape)",
        "loss": "masked cross-entropy on valid teacher actions in the unpadded suffix; five-frame prefix is warm-only",
        "train_collection": train_counts,
        "dev_collection": dev_counts,
        "train_quality": train_quality,
        "dev_quality": dev_quality,
        "dev_env_quality": dev_env_quality,
        "dev_objective": "success-1.5*wrong-0.001*tcp_mm-0.0005*turn_deg-0.0001*rail_mm-0.012*captures",
        "train_curve": train_curve,
        "dev_curve": dev_curve,
        "checkpoint": str(checkpoint),
    }
    _write_json(output / "imitation_summary.json", summary)
    return model, summary

