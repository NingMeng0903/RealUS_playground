"""Paired streaming servo ablation on a registered static artery and vein."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from primitive import (  # noqa: E402
    BLOCK_STEPS,
    FRAME_STRIDE,
    MAX_LATERAL_ACCEL_MM_S2,
    MAX_LATERAL_SPEED_MM_S,
    SAMPLE_DT_S,
    TOTAL_STEPS,
    LateralPrimitive,
    revise,
)
from sensor import Sensor  # noqa: E402


TRAIN_SEED = 2026111320
TEST_SEED = 2026111321
TRAIN_CASES = 128
TEST_CASES = 256
FRAME_STEPS = np.arange(0, TOTAL_STEPS + 1, FRAME_STRIDE)
FRAME_DT = FRAME_STRIDE * SAMPLE_DT_S
TICKS = np.arange(TOTAL_STEPS + 1) * SAMPLE_DT_S
N_COVERAGE_BINS = 6
CENTER_TOL_MM = 3.0
IDENTITY_MIN = 0.95
ARMS = ("learned_boundary_feedback", "endpoint_stream_feedback", "learned_stream_feedback")


def read_frame(sensor, scene, belief, frame, s, x):
    packet, roles, truth = sensor.observe(scene, frame, s, x)
    belief, probs = sensor.update(belief, packet, s, x)
    slot = int(np.argmax(probs)) if len(probs) and np.max(probs) >= IDENTITY_MIN else -1
    return belief, packet, probs, slot, roles, truth


def frame_log():
    n = len(FRAME_STEPS)
    return {
        "packets_mm": np.full((n, 2, 2), np.nan),
        "artery_probs": np.full((n, 2), np.nan),
        "role_truth": np.full((n, 2), -1, dtype=int),
        "selected_slot": np.full(n, -1, dtype=int),
        "true_lateral_mm": np.empty(n),
        "next_goal_mm": np.full(n, np.nan),
        "changed_next_frame_mm": np.zeros(n),
    }


def record(log, frame, packet, probs, slot, roles, truth):
    count = len(packet)
    log["packets_mm"][frame, :count] = packet
    log["artery_probs"][frame, :count] = probs
    log["role_truth"][frame, :count] = roles
    log["selected_slot"][frame] = slot
    log["true_lateral_mm"][frame] = truth[0]


def nominal_velocity(nominal, step):
    left = max(step - FRAME_STRIDE, 0)
    right = min(step + FRAME_STRIDE, TOTAL_STEPS)
    return (nominal[right] - nominal[left]) / (SAMPLE_DT_S * (right - left))


def teacher(sensor, scene, path, nominal):
    """Generate demonstrations with causal, rate-limited image servo feedback."""
    x, v = np.empty(len(TICKS)), np.empty(len(TICKS))
    x[0], v[0] = nominal[0], 0.0
    belief = sensor.prior()
    log = frame_log()
    velocity_command = np.zeros(len(FRAME_STEPS))
    for frame, step in enumerate(FRAME_STEPS):
        belief, packet, probs, slot, roles, truth = read_frame(sensor, scene, belief, frame, path[step], x[step])
        record(log, frame, packet, probs, slot, roles, truth)
        if step == TOTAL_STEPS:
            break
        lateral_error = 0.0 if slot < 0 else packet[slot, 0]
        command = np.clip(nominal_velocity(nominal, step) + 5.0 * lateral_error,
                          -MAX_LATERAL_SPEED_MM_S, MAX_LATERAL_SPEED_MM_S)
        velocity_command[frame] = command
        for tick in range(step, step + FRAME_STRIDE):
            cap = MAX_LATERAL_ACCEL_MM_S2 * SAMPLE_DT_S
            v[tick + 1] = v[tick] + np.clip(command - v[tick], -cap, cap)
            x[tick + 1] = x[tick] + 0.5 * SAMPLE_DT_S * (v[tick] + v[tick + 1])
    return dict(x_mm=x, vx_mm_s=v, teacher_velocity_mm_s=velocity_command, **log)


def train(sensor, primitive, path, nominal, count):
    rng = np.random.default_rng(TRAIN_SEED)
    logs = [teacher(sensor, sensor.scene(rng, len(FRAME_STEPS)), path, nominal) for _ in range(count)]
    features, targets = [], []
    for log in logs:
        for start in range(0, TOTAL_STEPS, BLOCK_STEPS):
            sl = slice(start, start + BLOCK_STEPS + 1)
            x, v = log["x_mm"][sl], log["vx_mm_s"][sl]
            features.append([1.0, x[-1] - x[0], v[0]])
            targets.append(primitive.fit(x, v))
    features, targets = np.asarray(features), np.asarray(targets)
    # Learn a shape prior conditioned on the commanded displacement and start velocity.
    regularizer = np.diag([0.0, 0.1, 0.1])
    regression = np.linalg.solve(features.T @ features + regularizer, features.T @ targets)
    return regression, dict(**stack(logs), shape_weights=targets, shape_regression=regression)


def execute(sensor, primitive, scene, path, nominal, regression, streaming):
    x, v = np.empty(len(TICKS)), np.empty(len(TICKS))
    x[0], v[0] = nominal[0], 0.0
    belief = sensor.prior()
    log = frame_log()
    plan = None
    state_jump = np.zeros(2)
    for frame, step in enumerate(FRAME_STEPS):
        belief, packet, probs, slot, roles, truth = read_frame(sensor, scene, belief, frame, path[step], x[step])
        record(log, frame, packet, probs, slot, roles, truth)
        if step == TOTAL_STEPS:
            break
        block_start = step % BLOCK_STEPS == 0
        end = (step // BLOCK_STEPS + 1) * BLOCK_STEPS
        if block_start or (streaming and slot >= 0):
            delta = sensor.offset_mean(belief)[0]
            if slot >= 0:
                measured_delta = packet[slot, 0] + x[step] - nominal[step]
                delta = 0.8 * delta + 0.2 * measured_delta
            goal = nominal[end] + np.clip(delta, -7.0, 7.0)
            goal_v = nominal_velocity(nominal, end)
            previous = None if block_start else plan
            shape = np.asarray([1.0, goal - x[step], v[step]]) @ regression
            plan = revise(primitive, step, end, x[step], v[step], goal, goal_v, shape,
                          previous, nominal[end] - nominal[step])
            if plan is not previous:
                state_jump = np.maximum(state_jump, np.abs([plan.x[0] - x[step], plan.v[0] - v[step]]))
                if previous is not None:
                    old_index = step + FRAME_STRIDE - previous.step
                    log["changed_next_frame_mm"][frame] = plan.x[FRAME_STRIDE] - previous.x[old_index]
        log["next_goal_mm"][frame] = plan.goal
        future = slice(step + 1 - plan.step, step + FRAME_STRIDE + 1 - plan.step)
        x[step + 1:step + FRAME_STRIDE + 1] = plan.x[future]
        v[step + 1:step + FRAME_STRIDE + 1] = plan.v[future]
    return dict(x_mm=x, vx_mm_s=v, state_jump=state_jump, **log)


def stack(logs):
    return {key: np.stack([log[key] for log in logs]) for key in logs[0]}


def score(log, path, origins, axes):
    bins = np.minimum((path[FRAME_STEPS] / path[-1] * N_COVERAGE_BINS).astype(int), N_COVERAGE_BINS - 1)
    slots = log["selected_slot"]
    frame = np.flatnonzero(slots >= 0)
    slot = slots[frame]
    centered = np.abs(log["packets_mm"][frame, slot, 0]) <= CENTER_TOL_MM
    accepted = frame[centered]
    truth = log["role_truth"][accepted, slots[accepted]]
    error = np.abs(log["true_lateral_mm"])
    correct = accepted[(truth == 0) & (error[accepted] <= CENTER_TOL_MM)]
    covered = len(np.unique(bins[correct]))
    wrong = bool(np.any(truth != 0))
    missing = np.flatnonzero(np.all(log["role_truth"] != 0, axis=1))
    longest_missing = max((len(run) for run in np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)), default=0)
    positions = origins + log["x_mm"][:, None] * axes[:, :, 1]
    travel = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    rotation = Rotation.from_matrix(axes)
    turn = np.rad2deg((rotation[:-1].inv() * rotation[1:]).magnitude()).sum()
    success = covered == N_COVERAGE_BINS and not wrong
    declared_complete = len(np.unique(bins[accepted])) == N_COVERAGE_BINS
    return dict(success=success, wrong=wrong, declared_complete=declared_complete,
                false_complete=declared_complete and not success,
                coverage=covered / N_COVERAGE_BINS, mean_abs_lateral_mm=float(error.mean()),
                longest_target_miss_s=longest_missing * FRAME_DT, tcp_mm=float(travel),
                turn_deg=float(turn), duration_s=float(TICKS[-1]), frames=len(FRAME_STEPS),
                midsegment_motion_changes=int(np.count_nonzero(np.abs(log["changed_next_frame_mm"]) >= 1.0e-3)))


def compare(left, right, bootstrap):
    out = {}
    for key in left[0]:
        delta = np.asarray([float(a[key]) - float(b[key]) for a, b in zip(left, right, strict=True)])
        out[key] = {"mean_delta": float(delta.mean()),
                    "paired_95pct_ci": np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist()}
        if key in ("success", "wrong"):
            out[key]["left_only"] = int(np.sum(delta == 1))
            out[key]["right_only"] = int(np.sum(delta == -1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sections = json.loads((HERE / "SECTIONS.json").read_text())
    sensor, primitive = Sensor(sections), LateralPrimitive()
    grid = np.asarray(sections["s_mm"])
    path = np.linspace(grid[0], grid[-1], len(TICKS))
    nominal = sensor.nominal(path)[:, 0, 0]
    origins = np.stack([np.interp(path, grid, column) for column in np.asarray(sections["origin_atlas_mm"]).T], axis=-1)
    axes = Slerp(grid, Rotation.from_matrix(np.asarray(sections["axes_atlas"])))(path).as_matrix()
    training_count, test_count = (6, 8) if args.smoke else (TRAIN_CASES, TEST_CASES)
    regression, demonstrations = train(sensor, primitive, path, nominal, training_count)
    np.savez_compressed(args.out / "DEMONSTRATIONS.npz", t_s=TICKS, s_mm=path, frame_steps=FRAME_STEPS, **demonstrations)
    rng = np.random.default_rng(TEST_SEED)
    traces = {arm: [] for arm in ARMS}
    scores = {arm: [] for arm in ARMS}
    scenes = []
    for _ in range(test_count):
        scene = sensor.scene(rng, len(FRAME_STEPS))
        scenes.append(scene)
        for arm in ARMS:
            weights = regression if arm.startswith("learned") else np.zeros_like(regression)
            log = execute(sensor, primitive, scene, path, nominal, weights, arm.endswith("stream_feedback"))
            traces[arm].append(log)
            scores[arm].append(score(log, path, origins, axes))
    arrays = {arm: stack(logs) for arm, logs in traces.items()}
    np.savez_compressed(args.out / "TRACES.npz", t_s=TICKS, s_mm=path, frame_steps=FRAME_STEPS, arms=ARMS,
                        **{key: np.stack([arrays[arm][key] for arm in ARMS]) for key in arrays[ARMS[0]]},
                        **{key: np.stack([getattr(scene, key) for scene in scenes])
                           for key in ("offset", "bias", "clear", "noise")})
    bootstrap = np.random.default_rng(2026111322).integers(test_count, size=(10_000, test_count))
    summary = {
        "scope": "static registered anatomy, synthetic centroid packets, ideal chart-space plant",
        "train_cases": training_count,
        "test_cases": test_count,
        "train_seed": TRAIN_SEED,
        "test_seed": TEST_SEED,
        "means": {arm: {key: float(np.mean([row[key] for row in scores[arm]])) for key in scores[arm][0]}
                  for arm in ARMS},
        "completed": {arm: int(sum(row["success"] for row in scores[arm])) for arm in ARMS},
        "max_state_jump_x_mm_vx_mm_s": {arm: arrays[arm]["state_jump"].max(axis=0).tolist() for arm in ARMS},
        "max_abs_speed_mm_s": {arm: float(np.abs(arrays[arm]["vx_mm_s"]).max()) for arm in ARMS},
        "max_sampled_acceleration_mm_s2": {arm: float(np.abs(np.diff(arrays[arm]["vx_mm_s"])).max() / SAMPLE_DT_S)
                                            for arm in ARMS},
        "feedback_effect": compare(scores[ARMS[2]], scores[ARMS[0]], bootstrap),
        "learned_shape_effect": compare(scores[ARMS[2]], scores[ARMS[1]], bootstrap),
        "cases": scores,
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed": summary["completed"], "test_cases": test_count}, indent=2))


if __name__ == "__main__":
    main()
