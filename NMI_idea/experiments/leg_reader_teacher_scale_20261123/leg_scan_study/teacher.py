"""Privileged branch evaluations on student-sampled development states."""

import numpy as np
import torch
from torch.nn import functional as F

from soft_identity import region_potential


def review_utility(node, region, initial_frames, initial_tick, initial_distance, initial_potential):
    reader = node.reader
    choice = reader.select(region)
    outcome = -1 if choice is None else int(reader.frames[choice[0]][3][choice[1]] == 0)
    captured = sum(region == frame[5] and 0 in frame[3] for frame in reader.frames[initial_frames:])
    cost = .006 * (node.tick - initial_tick) + .001 * (node.distance - initial_distance)
    potential_change = region_potential(reader, region) - initial_potential
    value = 1.4 * (outcome == 1) - 1.65 * (outcome == 0) + .085 * (captured > 0) + .75 * potential_change - cost
    return {"correct": int(outcome == 1), "wrong": int(outcome == 0),
            "captured": captured, "soft_identity_delta": potential_change, "incremental_time_ticks": node.tick - initial_tick,
            "incremental_distance_mm": node.distance - initial_distance, "utility": value}


def fit_teacher(policy, optimizer, examples):
    """Give the learning policy skill labels and viable controls for robust branches."""
    if not examples:
        return {"teacher_examples": 0, "teacher_positives": 0, "teacher_negatives": 0,
                "teacher_motor_targets": 0, "teacher_optimizer_updates": 0}
    x = torch.as_tensor(np.asarray([row["features"] for row in examples]), dtype=torch.float32)
    skill = torch.as_tensor([row["target_skill"] for row in examples], dtype=torch.float32)
    controls = torch.as_tensor(np.asarray([row["target_controls"] for row in examples]), dtype=torch.float32)
    active = skill.bool()
    logits, means, _ = policy(x)
    skill_before, motor_before = logits.detach().sigmoid(), means.detach().tanh()
    skill_loss = F.binary_cross_entropy_with_logits(logits, skill)
    control_loss = F.smooth_l1_loss(means[active].tanh(), controls[active]) if active.any() else means.sum() * 0.
    optimizer.zero_grad(set_to_none=True)
    (.37 * skill_loss + .48 * control_loss).backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), .7)
    optimizer.step()
    with torch.no_grad():
        trained_logits, trained_means, _ = policy(x)
        skill_shift = (trained_logits.sigmoid() - skill_before).abs().mean()
        motor_shift = (trained_means.tanh() - motor_before).abs().mean()
    return {"teacher_examples": len(examples), "teacher_positives": int(active.sum()),
            "teacher_negatives": len(examples) - int(active.sum()),
            "teacher_motor_targets": int(active.sum()), "teacher_optimizer_updates": 1,
            "teacher_skill_loss": float(skill_loss.detach()), "teacher_control_loss": float(control_loss.detach()),
            "teacher_skill_probability_mean_abs_shift": float(skill_shift),
            "teacher_control_mean_abs_shift": float(motor_shift)}
