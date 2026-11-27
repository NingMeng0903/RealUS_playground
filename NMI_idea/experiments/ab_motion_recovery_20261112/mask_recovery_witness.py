#!/usr/bin/env python3
"""Paired synthetic observability witness for mask-only material-section recovery.

The evaluator moves the interior by an unobserved axial shift. The policies
have the same old tracked masks, external rigid pose and calibrated probe
coordinates. Neither policy reads the shift or an image texture channel.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SLIPS = (-5, 5)
LOCAL = (0, -5, 5)
SWEEP = (0, 5, 10, 15, 20, 25, 30)
CANDIDATES = tuple(sorted(set(LOCAL + SWEEP)))
REFERENCE_X = 0
TOLERANCE_MM = 2.0
AXIS = np.linspace(-11.0, 11.0, 177)
Y, DEPTH_OFFSET = np.meshgrid(AXIS, AXIS)
THETA = np.deg2rad(18.0)
BODY_ROTATION = np.array([
    [np.cos(THETA), -np.sin(THETA), 0.0],
    [np.sin(THETA), np.cos(THETA), 0.0],
    [0.0, 0.0, 1.0],
])
BODY_ORIGIN = np.array([-18.0, 12.0, 7.0])


def tube_mask(material_x, with_branch):
    """Short-axis segmentation of a lumen with a widening branch separation."""
    main = Y**2 + DEPTH_OFFSET**2 <= 2.6**2
    if not with_branch or material_x < 14:
        return main
    lateral = min(8.0, 0.82 * (material_x - 14.0))
    branch = (Y - lateral)**2 + DEPTH_OFFSET**2 <= 1.6**2
    return main | branch


def signature(mask):
    return sha256(np.packbits(mask).tobytes()).hexdigest()


def record_reference_masks(with_branch):
    return {
        x: signature(tube_mask(x, with_branch))
        for x in range(-10, 36, 5)
    }


def capture(body_x, hidden_slip, with_branch):
    """Evaluator only. Project the calibrated world-frame plane to body x."""
    requested_world_origin = BODY_ORIGIN + BODY_ROTATION @ np.array([body_x, 0.0, 0.0])
    x_recovered = (BODY_ROTATION.T @ (requested_world_origin - BODY_ORIGIN))[0]
    if abs(x_recovered - body_x) > 1e-10:
        raise AssertionError("inconsistent external rigid coordinate transformation")
    return signature(tube_mask(round(x_recovered) - hidden_slip, with_branch))


def predicted_partition(archive, position, modes):
    return Counter(archive[position - mode] for mode in modes)


def active_position(archive, previous, modes, observed_positions):
    """Pick a tracked slice with distinct predicted masks for likely modes."""
    available = [x for x in CANDIDATES if x not in observed_positions]
    choices = [(x, len(predicted_partition(archive, x, modes))) for x in available]
    distinguishing = [(x, count) for x, count in choices if count > 1]
    if not distinguishing:
        return None
    return min(distinguishing, key=lambda item: (-item[1], abs(item[0] - previous), item[0]))[0]


def estimate(policy, archive, observe, prior):
    """Decision logic uses only observed masks and the same reference archive."""
    if policy == "geometry_replay":
        position = REFERENCE_X + prior[0] if len(prior) == 1 else REFERENCE_X
        return {"return_body_x_mm": position, "observations": [(position, observe(position))]}
    modes = list(prior)
    observed = []
    if len(modes) > 1:
        planned_views = LOCAL if policy == "local_full_history" else SWEEP
        sequence = iter(planned_views[1:])
        next_x = REFERENCE_X
        while next_x is not None:
            actual = observe(next_x)
            observed.append((next_x, actual))
            modes = [mode for mode in modes if archive[next_x - mode] == actual]
            if len(modes) <= 1:
                break
            if policy == "active_mask_recovery":
                next_x = active_position(archive, next_x, modes, {x for x, _ in observed})
            else:
                next_x = next(sequence, None)
    if len(modes) != 1:
        return {"return_body_x_mm": None, "observations": observed, "remaining_slips_mm": modes}
    position = REFERENCE_X + modes[0]
    observed.append((position, observe(position)))
    return {"return_body_x_mm": position, "observations": observed, "remaining_slips_mm": modes}


def evaluate(condition, hidden_slip, with_branch, revealed=False):
    archive = record_reference_masks(with_branch)
    policies = ("geometry_replay", "local_full_history", "classical_fixed_sweep", "active_mask_recovery")
    rows = []
    for policy in policies:
        prior = (hidden_slip,) if revealed else SLIPS
        # Only this evaluator callback closes over the private displacement.
        observe = lambda x: capture(x, hidden_slip, with_branch)
        result = estimate(policy, archive, observe, prior)
        chosen = result["return_body_x_mm"]
        error = None if chosen is None else abs(chosen - (REFERENCE_X + hidden_slip))
        status = "abstain" if chosen is None else "correct" if error <= TOLERANCE_MM else "wrong"
        xs = [REFERENCE_X, *(x for x, _ in result["observations"])]
        rows.append({
            "condition": condition, "policy": policy, "private_slip_mm": hidden_slip,
            "axial_error_mm": error, "status": status,
            "total_travel_mm": sum(abs(b - a) for a, b in zip(xs[:-1], xs[1:])),
            **result,
        })
    return rows


def main():
    assert all(capture(x, SLIPS[0], True) == capture(x, SLIPS[1], True) for x in LOCAL)
    assert capture(15, SLIPS[0], True) != capture(15, SLIPS[1], True)
    assert all(capture(x, SLIPS[0], False) == capture(x, SLIPS[1], False) for x in CANDIDATES)
    rows = [
        row for condition, branch, revealed in (
            ("tracked_branch", True, False),
            ("uniform_tube_symmetry", False, False),
            ("inner_registration_revealed", True, True),
        )
        for slip in SLIPS
        for row in evaluate(condition, slip, branch, revealed)
    ]
    report = {
        "scope": "two exact synthetic worlds, no image noise, no native controller or clinical data",
        "known_external_pose": {
            "rotation": BODY_ROTATION.tolist(), "translation_mm": BODY_ORIGIN.tolist(),
        },
        "fixed_short_axis_view_normal_world": BODY_ROTATION[:, 0].tolist(),
        "reference_target_mm": REFERENCE_X,
        "position_tolerance_mm": TOLERANCE_MM,
        "unobserved_inner_slips_mm": list(SLIPS),
        "local_views_mm": list(LOCAL),
        "first_discriminating_grid_position_mm": active_position(
            record_reference_masks(True), REFERENCE_X, SLIPS, {0}
        ),
        "results": rows,
    }
    (HERE / "MASK_RECOVERY_WITNESS.json").write_text(json.dumps(report, indent=2) + "\n")
    for condition in dict.fromkeys(r["condition"] for r in rows):
        for policy in dict.fromkeys(r["policy"] for r in rows):
            counts = Counter(r["status"] for r in rows if r["condition"] == condition and r["policy"] == policy)
            print(condition, policy, dict(counts))


if __name__ == "__main__":
    main()
