"""Paired teacher selection and a small public-state action-value model.

This module deliberately keeps the teacher and the student separate:

* :func:`recheck` only decides which action comparisons are still uncertain
  enough to spend two more *paired* search tapes on.
* :func:`select_teacher` consumes paired deltas against a frozen reference and
  makes a conservative decision.  It never sees a test split or a hidden
  anatomical label.
* :func:`fit_student` learns the empirical public-history deltas from those
  comparisons.  The actor input is the supplied public feature vector and the
  targets are only the observed soft/strict/wrong deltas.

The numerical conventions are important.  A positive ``strict_delta`` means
more strict deliveries than the frozen-v5 reference, while a positive
``wrong_branch_delta`` means *more* wrong-branch cases and is therefore bad.
The reference has a zero delta by definition.

The student is intentionally a small regression model rather than PPO.  It is
trained on CUDA when ``fit_student`` is called, but the returned predictor is
CPU/NumPy-only so that policy evaluation cannot silently initialise CUDA in a
rollout worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_TRIPLE_NAMES = ("soft", "strict", "wrong_branch")
_DEFAULT_REFERENCE = "continue"
_EPS = 1.0e-9


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _triples(values: Any) -> list[tuple[float, float, float]]:
    """Return finite triples, ignoring malformed rows.

    Search tapes are allowed to be interrupted.  Skipping a malformed tail is
    preferable to letting one broken tape change the action order; the caller
    can inspect the resulting sample count in the returned trace/manifest.
    """

    if values is None:
        return []
    try:
        iterator = iter(values)
    except TypeError:
        return []
    result: list[tuple[float, float, float]] = []
    for row in iterator:
        try:
            if len(row) < 3:  # type: ignore[arg-type]
                continue
            parsed = tuple(_as_float(row[index]) for index in range(3))
        except (TypeError, IndexError):
            continue
        if all(item is not None for item in parsed):
            result.append((float(parsed[0]), float(parsed[1]), float(parsed[2])))
    return result


def _stats(values: Sequence[float]) -> dict[str, Any]:
    """Summarise one paired delta stream with a conservative normal CI.

    One observation is treated as low signal and receives a unit standard
    error.  This avoids a false zero-width interval, while keeping all values
    finite and JSON-compatible.
    """

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "se": 1.0,
            "ci95": [-1.96, 1.96],
            "signal": False,
        }
    mean = float(array.mean())
    if count == 1:
        # A single paired tape does not estimate variance.  Deltas are bounded
        # reward-like quantities in this experiment, so one unit is a safe
        # neutral scale rather than pretending the variance is zero.
        std = 1.0
        se = 1.0
    else:
        std = float(array.std(ddof=1))
        se = float(std / math.sqrt(count))
        if not math.isfinite(se):
            se = 1.0
    half = float(1.96 * max(se, 1.0e-12))
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95": [float(mean - half), float(mean + half)],
        "signal": bool(count >= 2),
    }


def _estimate(action: str, values: Sequence[Sequence[float]] | Any) -> dict[str, Any]:
    samples = _triples(values)
    channels = [_stats([row[index] for row in samples]) for index in range(3)]
    return {
        "action": str(action),
        "n": len(samples),
        "soft": channels[0],
        "strict": channels[1],
        "wrong_branch": channels[2],
    }


def _action_estimates(
    paired_samples: Mapping[str, Sequence[Sequence[float]]] | None,
    reference: str,
) -> dict[str, dict[str, Any]]:
    if paired_samples is None:
        paired_samples = {}
    estimates: dict[str, dict[str, Any]] = {}
    # Preserve caller order for reproducible trace files; tie-breaking later
    # uses the action name so independent dict construction cannot change the
    # result.
    for key, values in paired_samples.items():
        action = str(key)
        if action not in estimates:
            estimates[action] = _estimate(action, values)
    if reference not in estimates:
        estimates[reference] = _estimate(reference, [])
    return estimates


def _paired_margin(estimate: Mapping[str, Any]) -> float:
    strict = estimate["strict"]
    wrong = estimate["wrong_branch"]
    soft = estimate["soft"]
    strict_width = float(strict["ci95"][1] - strict["ci95"][0])
    wrong_width = float(wrong["ci95"][1] - wrong["ci95"][0])
    soft_width = float(soft["ci95"][1] - soft["ci95"][0])
    # Strict and wrong are equally important for deciding whether to repeat a
    # tape; soft is only a small tie-breaker.
    return float(strict_width + wrong_width + 0.10 * soft_width)


def recheck(
    paired_samples: Mapping[str, Sequence[Sequence[float]]] | None,
    reference: str,
) -> list[str]:
    """Return actions worth two additional paired independent tapes.

    All entries are deltas against ``reference`` on the *same* tape.  An action
    is eligible when its strict or wrong-branch comparison still straddles the
    reference (or has fewer than two tapes), while actions with a clear strict
    loss and clear wrong-branch degradation are not rechecked.  The returned
    list is ordered with the reference first and contains at most three
    actions.  No held-out/test data enter this decision.
    """

    reference = str(reference)
    estimates = _action_estimates(paired_samples, reference)
    candidates: list[tuple[float, str]] = []
    for action, estimate in estimates.items():
        if action == reference:
            continue
        strict = estimate["strict"]
        wrong = estimate["wrong_branch"]
        soft = estimate["soft"]
        count = int(estimate["n"])
        strict_low, strict_high = map(float, strict["ci95"])
        wrong_low, wrong_high = map(float, wrong["ci95"])
        strict_uncertain = count < 2 or strict_low <= 0.0 <= strict_high
        wrong_uncertain = count < 2 or wrong_low <= 0.0 <= wrong_high
        clear_bad = strict_high < 0.0 and wrong_low > 0.0
        # Do not spend the extra tapes on an action that is already clearly
        # worse.  If either channel is still uncertain, it remains eligible.
        if clear_bad or not (strict_uncertain or wrong_uncertain):
            continue
        # A very large observed strict loss is not rescued by a positive soft
        # score; require at least a plausible strict upside for rechecking.
        if strict_high < -0.25 and float(soft["ci95"][1]) <= 0.0:
            continue
        uncertainty = _paired_margin(estimate)
        plausible_upside = max(0.0, strict_high) + 0.05 * max(0.0, float(soft["ci95"][1]))
        priority = float(uncertainty + plausible_upside - 0.10 * max(0.0, -strict_low))
        candidates.append((priority, action))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [reference] + [action for _, action in candidates[:2]]


def _teacher_score(estimate: Mapping[str, Any]) -> tuple[float, bool, dict[str, float]]:
    """Conservative score for one non-reference action."""

    count = int(estimate["n"])
    strict = estimate["strict"]
    wrong = estimate["wrong_branch"]
    soft = estimate["soft"]
    strict_mean = float(strict["mean"])
    strict_lcb = float(strict["ci95"][0])
    wrong_ucb = float(wrong["ci95"][1])
    soft_mean = float(soft["mean"])
    # With no paired signal, reference is safer.  Once at least two tapes are
    # available, strict lower confidence is primary.  Wrong branches are a
    # risk penalty; soft utility is deliberately capped and cannot overwhelm a
    # severe strict/wrong degradation.
    if count < 2:
        return (-math.inf, False, {"strict_lcb": strict_lcb, "wrong_ucb": wrong_ucb})
    severe = strict_mean < -0.05 or wrong_ucb > 0.25
    score = strict_lcb - 0.75 * max(0.0, wrong_ucb) + 0.05 * float(np.clip(soft_mean, -1.0, 1.0))
    # A large adverse mean remains disqualifying even when a soft reward is
    # positive.  This gate is intentionally conservative for deployment.
    eligible = bool(strict_lcb > 0.0 and not (strict_mean < -0.10 or wrong_ucb > 0.50))
    if severe:
        score -= 0.25 * max(0.0, -strict_mean) + 0.25 * max(0.0, wrong_ucb - 0.25)
    return (
        float(score),
        eligible,
        {
            "strict_lcb": strict_lcb,
            "wrong_ucb": wrong_ucb,
            "strict_mean": strict_mean,
            "wrong_mean": float(wrong["mean"]),
            "soft_mean": soft_mean,
        },
    )


def select_teacher(
    paired_samples: Mapping[str, Sequence[Sequence[float]]] | None,
    reference: str,
) -> tuple[str, dict[str, Any]]:
    """Select a conservative teacher action from paired search tapes.

    The frozen ``reference`` is a zero-delta baseline.  A candidate must have
    at least two paired tapes and a positive strict lower confidence estimate
    before it can displace that baseline.  The trace is JSON-compatible and
    records every action's estimates, sample count, and the score mix.
    """

    reference = str(reference)
    estimates = _action_estimates(paired_samples, reference)
    best_action = reference
    best_score = 0.0
    trace_estimates: dict[str, Any] = {}
    for action in sorted(estimates):
        estimate = estimates[action]
        if action == reference:
            trace_estimates[action] = {
                **estimate,
                "score": 0.0,
                "eligible": True,
                "strict_lcb": 0.0,
                "wrong_ucb": 0.0,
            }
            continue
        score, eligible, diagnostics = _teacher_score(estimate)
        trace_estimates[action] = {
            **estimate,
            "score": float(score) if math.isfinite(score) else None,
            "eligible": bool(eligible),
            **{key: float(value) for key, value in diagnostics.items()},
        }
        if eligible and score > best_score + _EPS:
            best_action = action
            best_score = float(score)
    trace = {
        "reference": reference,
        "selected": best_action,
        "estimates": trace_estimates,
        "mix": {
            "primary": "mean_strict_delta_with_95pct_lower_bound",
            "strict_lcb_weight": 1.0,
            "wrong_branch_upper_risk_weight": 0.75,
            "soft_delta_weight": 0.05,
            "minimum_paired_tapes": 2,
            "reference_tie_epsilon": _EPS,
            "paired_against_reference": True,
            "decision_scope": "training_or_development_tapes_only",
        },
    }
    return best_action, trace


def _infer_reference(rows: Sequence[Mapping[str, Any]], actions: Sequence[str]) -> str:
    names = [str(action) for action in actions]
    for row in rows:
        candidate = row.get("reference")
        if candidate is not None and str(candidate) in names:
            return str(candidate)
    if _DEFAULT_REFERENCE in names:
        return _DEFAULT_REFERENCE
    if not names:
        raise ValueError("actions must contain at least one action")
    return names[0]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _output_paths(out: str | os.PathLike[str]) -> tuple[Path, Path]:
    requested = Path(out)
    if requested.suffix.lower() == ".pt":
        checkpoint = requested
        manifest = requested.with_suffix(".json")
    elif requested.suffix.lower() == ".json":
        manifest = requested
        checkpoint = requested.with_suffix(".pt")
    else:
        requested.mkdir(parents=True, exist_ok=True)
        checkpoint = requested / "student_value.pt"
        manifest = requested / "student_value.json"
    return checkpoint, manifest


def _safe_layout(row: Mapping[str, Any], index: int) -> str:
    value = row.get("layout", f"__missing_layout_{index}")
    return str(value)


def _targets_for_row(
    row: Mapping[str, Any],
    action_names: Sequence[str],
    reference: str,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((len(action_names), 3), dtype=np.float32)
    mask = np.zeros((len(action_names), 3), dtype=np.float32)
    samples_by_action = row.get("paired_samples", {})
    if not isinstance(samples_by_action, Mapping):
        samples_by_action = {}
    for index, action in enumerate(action_names):
        if action == reference:
            # Reference deltas are exactly zero by construction, even when a
            # caller includes redundant all-zero rows under that key.
            mask[index, :] = 1.0
            continue
        samples = _triples(samples_by_action.get(action, []))
        if not samples:
            continue
        target[index, :] = np.asarray(np.mean(np.asarray(samples, dtype=np.float64), axis=0), dtype=np.float32)
        mask[index, :] = 1.0
    return target, mask


def _normalise_features(
    rows: Sequence[Mapping[str, Any]], train_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    vectors: list[np.ndarray] = []
    for index, row in enumerate(rows):
        value = row.get("features")
        if value is None:
            raise ValueError(f"row {index} is missing public features")
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise ValueError(f"row {index} has empty public features")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"row {index} has non-finite public features")
        vectors.append(array)
    input_dim = int(vectors[0].size)
    if any(int(vector.size) != input_dim for vector in vectors):
        raise ValueError("all public feature vectors must have the same fixed dimension")
    train_array = np.asarray([vectors[index] for index in train_indices], dtype=np.float64)
    feature_mean = train_array.mean(axis=0).astype(np.float32)
    feature_scale = train_array.std(axis=0).astype(np.float32)
    feature_scale = np.where(feature_scale > 1.0e-6, feature_scale, 1.0).astype(np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix = (matrix - feature_mean) / feature_scale
    return matrix, feature_mean, input_dim, feature_scale


def _cpu_state_dict(torch: Any, state_dict: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {str(key): value.detach().cpu().numpy().copy() for key, value in state_dict.items()}


def fit_student(
    rows: Sequence[Mapping[str, Any]],
    actions: Sequence[str],
    out: str | os.PathLike[str],
    *,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a tiny CUDA action-value regressor from paired public histories.

    ``rows`` contain no actor-side truth labels.  Each action target is the
    empirical mean of its paired soft/strict/wrong deltas, and absent actions
    are masked from the loss.  Layouts, rather than individual rows, define the
    train/dev split to prevent tape leakage across the same synthetic anatomy.
    """

    if not rows:
        raise ValueError("fit_student requires at least one public-history row")
    action_names: list[str] = []
    for action in actions:
        name = str(action)
        if name not in action_names:
            action_names.append(name)
    reference = _infer_reference(rows, action_names)
    if reference not in action_names:
        action_names.insert(0, reference)

    # Import lazily so CPU-only worker processes can still import recheck and
    # select_teacher without initialising a CUDA context.
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("fit_student requires PyTorch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("fit_student requires an available CUDA device")

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))
    layout_names = sorted({_safe_layout(row, index) for index, row in enumerate(rows)})
    layout_order = list(layout_names)
    rng.shuffle(layout_order)
    if len(layout_order) >= 2:
        dev_count = max(1, int(round(0.20 * len(layout_order))))
        dev_layouts = set(layout_order[:dev_count])
        train_layouts = set(layout_order[dev_count:])
        if not train_layouts:
            train_layouts = set(layout_order)
            dev_layouts = set()
    else:
        train_layouts = set(layout_order)
        dev_layouts = set()
    train_indices = [index for index, row in enumerate(rows) if _safe_layout(row, index) in train_layouts]
    dev_indices = [index for index, row in enumerate(rows) if _safe_layout(row, index) in dev_layouts]
    if not train_indices:
        train_indices = list(range(len(rows)))
        dev_indices = []

    feature_matrix, feature_mean, input_dim, feature_scale = _normalise_features(rows, train_indices)
    target_matrix: list[np.ndarray] = []
    mask_matrix: list[np.ndarray] = []
    for row in rows:
        target, mask = _targets_for_row(row, action_names, reference)
        target_matrix.append(target)
        mask_matrix.append(mask)
    targets = np.asarray(target_matrix, dtype=np.float32)
    masks = np.asarray(mask_matrix, dtype=np.float32)
    train_targets = targets[train_indices]
    train_masks = masks[train_indices]
    observed = train_targets[train_masks > 0.0]
    if observed.size == 0:
        raise ValueError("paired_samples contain no usable action targets")
    # A shared scale per output channel keeps the strict/wrong channels in a
    # comparable numerical range without changing their deployment semantics.
    target_center = np.zeros(3, dtype=np.float32)
    target_scale = np.ones(3, dtype=np.float32)
    for channel in range(3):
        values = train_targets[:, :, channel][train_masks[:, :, channel] > 0.0]
        if values.size:
            target_center[channel] = float(values.mean())
            spread = float(values.std())
            target_scale[channel] = max(spread, 1.0e-3)
    targets_z = (targets - target_center.reshape(1, 1, 3)) / target_scale.reshape(1, 1, 3)

    output_dim = int(3 * len(action_names))
    hidden_candidates = (32, 24, 16, 12, 8, 4)
    hidden_dim = next(
        (candidate for candidate in hidden_candidates if (input_dim + 1) * candidate + (candidate + 1) * output_dim < 10000),
        None,
    )
    if hidden_dim is None:
        raise ValueError(f"public feature dimension {input_dim} is too large for a <10k-parameter student")
    parameter_count = int((input_dim + 1) * hidden_dim + (hidden_dim + 1) * output_dim)
    device = torch.device("cuda:0")
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, output_dim),
    ).to(device)
    initial = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    feature_tensor = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    target_tensor = torch.as_tensor(targets_z.reshape(len(rows), output_dim), dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(masks.reshape(len(rows), output_dim), dtype=torch.float32, device=device)
    train_index_array = np.asarray(train_indices, dtype=np.int64)
    batch_size = min(64, max(8, len(train_indices)))
    epochs = 60 if len(train_indices) < 512 else 32
    losses: list[float] = []
    dev_losses: list[float] = []
    optimizer_steps = 0
    # A fixed host permutation per epoch makes the fit reproducible while
    # avoiding a CUDA DataLoader in rollout workers.
    for epoch in range(epochs):
        shuffled = rng.permutation(train_index_array)
        model.train()
        epoch_loss = 0.0
        epoch_weight = 0.0
        for start in range(0, len(shuffled), batch_size):
            batch = torch.as_tensor(shuffled[start : start + batch_size], dtype=torch.long, device=device)
            prediction = model(feature_tensor.index_select(0, batch))
            target_batch = target_tensor.index_select(0, batch)
            mask_batch = mask_tensor.index_select(0, batch)
            # Channel weighting prevents the low-scale soft utility from
            # dominating strict delivery and wrong-branch risk.
            channel_weight = torch.tensor([0.25, 1.0, 1.0], dtype=torch.float32, device=device)
            channel_weight = channel_weight.repeat(len(action_names))
            loss_values = F.smooth_l1_loss(prediction, target_batch, reduction="none")
            loss_values = loss_values * mask_batch * channel_weight.reshape(1, -1)
            denominator = torch.clamp((mask_batch * channel_weight.reshape(1, -1)).sum(), min=1.0)
            loss = loss_values.sum() / denominator
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            batch_weight = float(mask_batch.sum().detach().item())
            epoch_loss += float(loss.detach().item()) * max(batch_weight, 1.0)
            epoch_weight += max(batch_weight, 1.0)
        losses.append(float(epoch_loss / max(epoch_weight, 1.0)))
        if dev_indices:
            model.eval()
            with torch.no_grad():
                dev_index_tensor = torch.as_tensor(dev_indices, dtype=torch.long, device=device)
                dev_prediction = model(feature_tensor.index_select(0, dev_index_tensor))
                dev_target = target_tensor.index_select(0, dev_index_tensor)
                dev_mask = mask_tensor.index_select(0, dev_index_tensor)
                dev_weight = torch.tensor([0.25, 1.0, 1.0], dtype=torch.float32, device=device).repeat(len(action_names))
                dev_values = F.smooth_l1_loss(dev_prediction, dev_target, reduction="none") * dev_mask * dev_weight.reshape(1, -1)
                dev_denominator = torch.clamp((dev_mask * dev_weight.reshape(1, -1)).sum(), min=1.0)
                dev_losses.append(float((dev_values.sum() / dev_denominator).item()))
    torch.cuda.synchronize()
    # Confirm that an optimizer update really changed a CUDA parameter, rather
    # than merely constructing a CUDA graph or running a forward pass.
    changed = any(not torch.equal(p.detach(), old) for p, old in zip(model.parameters(), initial, strict=True))
    cuda_optimizer_step_verified = bool(changed and optimizer_steps > 0 and all(p.is_cuda for p in model.parameters()))
    cpu_state = _cpu_state_dict(torch, model.state_dict())
    checkpoint, manifest_path = _output_paths(out)
    checkpoint_payload = {
        "format": "v6_paired_value_cpu_predictor",
        "architecture": {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim},
        "actions": action_names,
        "reference": reference,
        "state_dict": {key: value for key, value in cpu_state.items()},
        "optimizer": optimizer.state_dict(),
        "epochs": epochs,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": rng.bit_generator.state,
    }
    _atomic_torch_save(torch, checkpoint_payload, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    train_layout_list = sorted(train_layouts)
    dev_layout_list = sorted(dev_layouts)
    manifest: dict[str, Any] = {
        "format": "v6_paired_value_manifest",
        "version": 1,
        "actions": action_names,
        "reference": reference,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "parameter_count": parameter_count,
        "rows": len(rows),
        "train_rows": len(train_indices),
        "dev_rows": len(dev_indices),
        "layouts": len(layout_names),
        "train_layouts": train_layout_list,
        "validation_layouts": dev_layout_list,
        "seed": int(seed),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_device": str(torch.cuda.get_device_name(device)),
        "cuda_optimizer_step_verified": cuda_optimizer_step_verified,
        "optimizer": "AdamW",
        "optimizer_steps": int(optimizer_steps),
        "epochs": int(epochs),
        "target_fields": list(_TRIPLE_NAMES),
        "target_center": [float(value) for value in target_center],
        "target_scale": [float(value) for value in target_scale],
        "feature_mean": [float(value) for value in feature_mean],
        "feature_scale": [float(value) for value in feature_scale],
        "channel_loss_weights": {"soft": 0.25, "strict": 1.0, "wrong_branch": 1.0},
        "train_loss": losses,
        "validation_loss": dev_losses,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "actor_input": "fixed_public_features_only; no true labels or hidden layout",
        "target_definition": "paired empirical mean delta versus frozen-v5 reference",
    }
    _atomic_json(manifest_path, manifest)
    bundle: dict[str, Any] = {
        "format": "v6_paired_value_cpu_predictor",
        "version": 1,
        "actions": action_names,
        "reference": reference,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "state_dict": cpu_state,
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_scale": np.asarray(feature_scale, dtype=np.float32),
        "target_center": np.asarray(target_center, dtype=np.float32),
        "target_scale": np.asarray(target_scale, dtype=np.float32),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path),
    }
    return bundle, manifest


def _bundle_array(bundle: Mapping[str, Any], key: str) -> np.ndarray:
    value = bundle.get(key)
    if value is None:
        raise ValueError(f"student bundle is missing {key}")
    return np.asarray(value, dtype=np.float32)


def student_scores(features: Sequence[float], bundle: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Predict action deltas deterministically on CPU from a returned bundle."""

    action_names = [str(action) for action in bundle.get("actions", [])]
    if not action_names:
        raise ValueError("student bundle has no actions")
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    input_dim = int(bundle.get("input_dim", vector.size))
    if vector.size != input_dim:
        raise ValueError(f"expected {input_dim} public features, got {vector.size}")
    mean = _bundle_array(bundle, "feature_mean").reshape(-1)
    scale = _bundle_array(bundle, "feature_scale").reshape(-1)
    x = (vector - mean) / np.where(scale > 1.0e-6, scale, 1.0)
    state = bundle.get("state_dict", {})
    if not isinstance(state, Mapping):
        raise ValueError("student bundle has invalid state_dict")
    # The tiny network is a Linear-Tanh-Linear Sequential model.  Accept torch
    # tensors as well as the NumPy state returned by fit_student.
    weight1 = np.asarray(state.get("0.weight"), dtype=np.float32)
    bias1 = np.asarray(state.get("0.bias"), dtype=np.float32)
    weight2 = np.asarray(state.get("2.weight"), dtype=np.float32)
    bias2 = np.asarray(state.get("2.bias"), dtype=np.float32)
    hidden = np.tanh(weight1 @ x + bias1)
    normalized = weight2 @ hidden + bias2
    center = _bundle_array(bundle, "target_center").reshape(1, 3)
    target_scale = _bundle_array(bundle, "target_scale").reshape(1, 3)
    values = normalized.reshape(len(action_names), 3) * target_scale + center
    reference = str(bundle.get("reference", action_names[0]))
    result: dict[str, dict[str, float]] = {}
    for index, action in enumerate(action_names):
        if action == reference:
            # Deltas are defined relative to this action; keeping the baseline
            # exactly zero makes greedy comparisons deterministic.
            row = np.zeros(3, dtype=np.float32)
        else:
            row = values[index]
        result[action] = {
            "soft": float(row[0]),
            "strict": float(row[1]),
            "wrong_branch": float(row[2]),
        }
    return result


def student_action(
    features: Sequence[float],
    bundle: Mapping[str, Any],
    reference: str,
) -> str:
    """Choose a conservative action from public-state predicted deltas."""

    reference = str(reference)
    scores = student_scores(features, bundle)
    if reference not in scores:
        reference = str(bundle.get("reference", reference))
    best_action = reference
    best_score = 0.0
    for action in sorted(scores):
        if action == reference:
            continue
        values = scores[action]
        strict = float(values["strict"])
        wrong = float(values["wrong_branch"])
        soft = float(np.clip(values["soft"], -1.0, 1.0))
        # The learned model has no uncertainty head, so require positive strict
        # gain and use wrong-branch risk as a first-class penalty.  Soft utility
        # is deliberately too small to compensate for severe harm.
        if strict <= 0.0:
            continue
        score = strict - 0.75 * max(0.0, wrong) + 0.05 * soft
        if strict < -0.10 or wrong > 0.50:
            continue
        if score > best_score + _EPS:
            best_action = action
            best_score = score
    return best_action


def _smoke() -> dict[str, Any]:
    """Bounded synthetic API/CUDA smoke used by developers, not experiments."""

    paired = {
        "continue": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "anchor": [[0.02, 0.03, 0.00], [0.01, -0.02, 0.02]],
        "window": [[0.01, 0.00, 0.01]],
    }
    shortlist = recheck(paired, "continue")
    action, trace = select_teacher(paired, "continue")
    if shortlist[0] != "continue" or "continue" not in shortlist:
        raise AssertionError("recheck must include reference first")
    rows = []
    for index in range(6):
        rows.append(
            {
                "features": [float(index), float(index % 2), 1.0],
                "layout": f"layout-{index // 2}",
                "paired_samples": {
                    "continue": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "anchor": [[0.03 + 0.01 * (index % 2), 0.04, 0.00], [0.02, 0.03, 0.01]],
                },
            }
        )
    with tempfile.TemporaryDirectory(prefix="v6-paired-value-smoke-") as directory:
        bundle, manifest = fit_student(rows, ["continue", "anchor", "window"], directory, seed=17)
        predicted = student_scores(rows[0]["features"], bundle)
        selected = student_action(rows[0]["features"], bundle, "continue")
        if manifest["parameter_count"] >= 10000 or not manifest["cuda_optimizer_step_verified"]:
            raise AssertionError("student is not a bounded verified CUDA update")
    return {
        "shortlist": shortlist,
        "teacher": action,
        "teacher_samples": trace["estimates"]["anchor"]["n"],
        "student_action": selected,
        "student_actions": sorted(predicted),
        "cuda_optimizer_step_verified": bool(manifest["cuda_optimizer_step_verified"]),
        "parameter_count": int(manifest["parameter_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v6 paired teacher/student bounded smoke")
    parser.add_argument("--smoke", action="store_true", help="run synthetic API and CUDA smoke")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(_smoke(), sort_keys=True))
    else:
        parser.error("pass --smoke; this module is an importable API, not a rollout entry point")


if __name__ == "__main__":
    main()
