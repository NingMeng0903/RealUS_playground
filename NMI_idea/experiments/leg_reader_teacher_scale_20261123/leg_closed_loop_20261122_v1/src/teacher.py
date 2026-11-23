"""Privileged counterfactual scoring and conditional soft-teacher updates.

Teacher branches are evaluated on paired future tapes.  Feasible signed returns
are first regressed by an action-conditional gain head, then aggregated per
skill before the actor receives a soft preference target.  Consequently, the
number of sampled branches for one skill cannot by itself increase that skill's
weight, and equal positive/negative hidden-world outcomes cancel.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from policy import ScanPolicy, _normalise_mask, _skill_index, _tanh_log_prob
from soft_identity import region_potential


def review_utility(node, region, initial_frames, initial_tick, initial_distance, initial_potential):
    """Evaluate a completed teacher branch with soft and strict identity terms."""
    reader = node.reader
    choice = reader.select(region)
    outcome = -1 if choice is None else int(reader.frames[choice[0]][3][choice[1]] == 0)
    captured = sum(region == frame[5] and 0 in frame[3] for frame in reader.frames[initial_frames:])
    cost = .006 * (node.tick - initial_tick) + .001 * (node.distance - initial_distance)
    potential_change = region_potential(reader, region) - initial_potential
    value = (1.4 * (outcome == 1) - 1.65 * (outcome == 0)
             + .085 * (captured > 0) + .75 * potential_change - cost)
    return {"correct": int(outcome == 1), "wrong": int(outcome == 0),
            "captured": captured, "soft_identity_delta": potential_change,
            "incremental_time_ticks": node.tick - initial_tick,
            "incremental_distance_mm": node.distance - initial_distance,
            "utility": value}


def _branch_skill(branch):
    value = branch.get("skill")
    if value is not None:
        return _skill_index(value)
    candidate = str(branch.get("candidate", "continue")).lower()
    if candidate in {"continue", "baseline", "none"}:
        return 0
    if "revisit" in candidate or "return" in candidate or "回访" in candidate:
        return 2
    return 1


def _feasible(branch):
    if branch.get("feasible", branch.get("legal", True)) is False:
        return False
    replays = branch.get("replays")
    if replays is None:
        return True
    # A branch may have one failed future tape while the paired tape remains a
    # useful regression target. Filtering happens again per replay below.
    return any(replay.get("feasible", replay.get("legal", True)) is not False
               for replay in replays)


def _paired_returns(branch):
    """Return feasible per-tape returns and whether they are already gains."""
    if not _feasible(branch):
        return np.empty(0, dtype=np.float32), False
    replays = branch.get("replays")
    if replays is not None:
        values = []
        valid_indices = []
        for index, replay in enumerate(replays):
            if replay.get("feasible", replay.get("legal", True)) is False:
                continue
            if "utility" in replay:
                values.append(float(replay["utility"]))
                valid_indices.append(index)
        if values:
            # If paired_gain is present, preserve its signed per-tape values but
            # remove entries whose corresponding replay was infeasible.
            if "paired_gain" in branch:
                gains = np.asarray(branch["paired_gain"], dtype=np.float32).reshape(-1)
                if len(gains) == len(replays):
                    gains = gains[np.asarray(valid_indices, dtype=int)]
                return gains, True
            return np.asarray(values, dtype=np.float32), False
    if "paired_gain" in branch:
        gains = np.asarray(branch["paired_gain"], dtype=np.float32).reshape(-1)
        return gains, True
    if "gains" in branch:
        gains = np.asarray(branch["gains"], dtype=np.float32).reshape(-1)
        return gains, True
    if "utility" in branch:
        return np.asarray([float(branch["utility"])], dtype=np.float32), False
    return np.empty(0, dtype=np.float32), False


def _example_groups(row):
    """Group feasible ``(signed return, control)`` samples by high-level skill."""
    branches = list(row.get("branches", []))
    if not branches:
        branches = [{"skill": row.get("target_skill", 0),
                     "controls": row.get("target_controls", np.zeros(3)),
                     "paired_gain": [0., 0.]}]
    baseline = None
    parsed = []
    for branch in branches:
        values, are_gains = _paired_returns(branch)
        skill = _branch_skill(branch)
        if len(values) and skill == 0 and not are_gains:
            baseline = values.copy()
        parsed.append((branch, skill, values, are_gains))
    groups = {0: [], 1: [], 2: []}
    for branch, skill, values, are_gains in parsed:
        if not len(values) or not _feasible(branch):
            continue
        if are_gains:
            gains = values
        elif baseline is not None:
            count = min(len(values), len(baseline))
            gains = values[:count] - baseline[:count]
        else:
            gains = values
        controls = np.asarray(branch.get("controls", branch.get("action", np.zeros(3))), dtype=np.float32).reshape(3)
        controls = np.clip(controls, -.999, .999)
        groups[skill].extend((float(gain), controls.copy()) for gain in gains if np.isfinite(gain))
    # The continue baseline is the reference action and is always available as
    # zero expected gain, even when its rollout was omitted from the log.
    if not groups[0]:
        groups[0].append((0., np.zeros(3, dtype=np.float32)))
    return groups


def _softmax_np(values, beta):
    values = np.asarray(values, dtype=np.float64)
    scaled = float(beta) * (values - np.max(values))
    weights = np.exp(np.clip(scaled, -60., 60.))
    return weights / max(weights.sum(), 1.e-12)


def _teacher_update(policy, optimizer, examples, *, beta=4., entropy_weight=.002,
                    gain_weight=.65, actor_weight=1.):
    device = policy.log_scale.device
    groups = [_example_groups(row) for row in examples]
    observations = np.asarray([row["features"] for row in examples], dtype=np.float32)
    stages = np.asarray([int(row.get("stage", 0)) for row in examples], dtype=np.int64)
    masks = np.asarray([_normalise_mask(row.get("mask"), int(row.get("stage", 0)))
                        for row in examples], dtype=bool)

    # Explicit per-tape gain regression. Continue's return is zero by definition.
    gain_features, gain_skills, gain_actions, gain_targets = [], [], [], []
    for group in groups:
        for skill in (1, 2):
            for gain, controls in group[skill]:
                gain_skills.append(skill)
                gain_actions.append(controls)
                gain_targets.append(gain)
    for _ in groups:
        gain_skills.append(0)
        gain_actions.append(np.zeros(3, dtype=np.float32))
        gain_targets.append(0.)
    # Keep one observation aligned with every gain sample. This is intentionally
    # a plain list to avoid ever exposing hidden labels to the actor.
    for index, group in enumerate(groups):
        active_count = sum(len(group[s]) for s in (1, 2))
        gain_features.extend([observations[index]] * (active_count + 1))
    gain_x = torch.as_tensor(np.asarray(gain_features), dtype=torch.float32, device=device)
    gain_skill = torch.as_tensor(gain_skills, dtype=torch.long, device=device)
    gain_action = torch.as_tensor(np.asarray(gain_actions), dtype=torch.float32, device=device)
    gain_target = torch.as_tensor(gain_targets, dtype=torch.float32, device=device)
    gain_prediction = policy.predict_gain(gain_x, gain_skill, gain_action)
    gain_loss = F.smooth_l1_loss(gain_prediction, gain_target)

    x = torch.as_tensor(observations, dtype=torch.float32, device=device)
    stage = torch.as_tensor(stages, dtype=torch.long, device=device)
    mask = torch.as_tensor(masks, dtype=torch.bool, device=device)
    logits, controls, _ = policy(x, stage)
    if logits.shape[-1] == 2:
        logits = F.pad(logits, (0, 1), value=-30.)
    probabilities = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1)
    target_probabilities = np.zeros((len(groups), 3), dtype=np.float32)
    target_controls = np.zeros((len(groups), 2, 3), dtype=np.float32)
    conflicting_expectations = []
    for index, group in enumerate(groups):
        expected = np.zeros(3, dtype=np.float32)
        for skill in (1, 2):
            if group[skill]:
                expected[skill] = float(np.mean([item[0] for item in group[skill]]))
                weights = _softmax_np([item[0] for item in group[skill]], beta)
                target_controls[index, skill - 1] = np.sum(
                    np.asarray([item[1] for item in group[skill]]) * weights[:, None], axis=0)
        legal = masks[index].copy()
        expected[0] = 0.
        expected[~legal] = -1.e9
        target_probabilities[index] = _softmax_np(expected, beta) * legal
        target_probabilities[index] /= max(target_probabilities[index].sum(), 1.e-12)
        conflicting_expectations.append(float(np.sum(target_probabilities[index] * np.clip(expected, -1.e5, 1.e5))))
    target = torch.as_tensor(target_probabilities, dtype=torch.float32, device=device)
    actor_loss = -(target * torch.log(probabilities.clamp_min(1.e-8))).sum(-1).mean()
    motor_loss = torch.zeros((), dtype=torch.float32, device=device)
    motor_count = 0
    for index in range(len(groups)):
        if stages[index] != 0:
            continue
        for skill in (1, 2):
            if not groups[index][skill]:
                continue
            mean = controls[index, (skill - 1) * 3:skill * 3][None]
            latent = torch.as_tensor(np.arctanh(target_controls[index, skill - 1]),
                                     dtype=torch.float32, device=device)[None]
            logp = _tanh_log_prob(mean, policy.log_scale, latent)[0]
            motor_loss = motor_loss - target[index, skill] * logp
            motor_count += 1
    if motor_count:
        motor_loss = motor_loss / motor_count
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1.e-8))
                * mask.to(probabilities.dtype)).sum(-1).mean()
    objective = actor_weight * (actor_loss + motor_loss) + gain_weight * gain_loss - entropy_weight * entropy
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), .7)
    optimizer.step()

    scalars = torch.stack((gain_loss.detach(), actor_loss.detach(), motor_loss.detach(),
                           entropy.detach(), grad_norm.detach())).cpu().tolist()
    return {"teacher_examples": len(examples),
            "teacher_positives": int(sum(any(groups[i][s] for s in (1, 2)) for i in range(len(groups)))),
            "teacher_negatives": int(sum(not any(groups[i][s] for s in (1, 2)) for i in range(len(groups)))),
            "teacher_motor_targets": int(motor_count), "teacher_optimizer_updates": 1,
            "teacher_branches": int(sum(len(groups[i][s]) for i in range(len(groups)) for s in (1, 2))),
            "teacher_gain_samples": len(gain_targets), "teacher_gain_loss": float(scalars[0]),
            "teacher_actor_loss": float(scalars[1]), "teacher_motor_loss": float(scalars[2]),
            "teacher_entropy": float(scalars[3]), "teacher_gradient_norm": float(scalars[4]),
            "teacher_expected_gain_mean": float(np.mean(conflicting_expectations)),
            "teacher_conflicting_prefixes_near_zero": int(sum(abs(v) < .05 for v in conflicting_expectations))}


def fit_teacher(policy: ScanPolicy, optimizer, examples, *, beta=4., entropy_weight=.002):
    """Fit a conditional gain head and soft skill/parameter preference targets."""
    if not examples:
        return {"teacher_examples": 0, "teacher_positives": 0,
                "teacher_negatives": 0, "teacher_motor_targets": 0,
                "teacher_optimizer_updates": 0, "teacher_branches": 0}
    return _teacher_update(policy, optimizer, examples, beta=beta,
                           entropy_weight=entropy_weight)

