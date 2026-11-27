"""Re-run the plotted case with online C2 references and frozen task scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation

import rl_rollouts as rollout
from audit_trajectory import ARMS, DYNAMIC, load_rows, reconstruct
from smooth_switch import BodyClock, SmoothAtlas, TimedProbe, learned_shapes, rest_jet

HERE = Path(__file__).resolve().parent
GEOMETRY = DYNAMIC / "DYNAMIC_GEOMETRY_FROZEN.npz"
SEED = 201568239
OUT = HERE / "smooth_switch_result"


class ReferenceRecorder:
    def __init__(self, atlas, shapes, moving, slot_s):
        self.atlas, self.shapes = atlas, shapes
        self.moving, self.slot_s = moving, slot_s
        self.segments = []
        self.states = []

    def append(self, previous, goal, skill):
        previous, goal = tuple(map(int, previous)), tuple(map(int, goal))
        if not self.segments:
            pairs = zip(self.moving, (*self.moving[1:], self.moving[0]))
            index = next(i for i, pair in enumerate(pairs) if pair == (previous[0], goal[0]))
            lower, upper = self.atlas.angles[[self.moving[0], max(self.moving)]]
            self.clock = BodyClock(self.slot_s, index, .5 * (lower + upper), .5 * (upper - lower))
            self.states.append(previous)
        if self.states[-1] != previous:
            raise ValueError("the next action must start at the executed exit")
        t = self.slot_s * len(self.segments)
        material = rest_jet(self.atlas.material(previous))
        if self.segments:
            material = self.segments[-1].material_jet(self.slot_s)[:, 0, :]
        shape = self.shapes.get(skill)
        ref = TimedProbe(self.atlas, self.clock, shape, t, self.slot_s, material, self.atlas.material(goal))
        end_theta = self.clock.sample(t + self.slot_s)[0]
        np.testing.assert_allclose(end_theta, self.atlas.angles[goal[0]], atol=1.e-12, rtol=0.)
        self.segments.append(ref)
        self.states.append(goal)


def pairs_difference(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)


def endpoint_state(segment, elapsed):
    value, rate, acc = segment.world_jet(elapsed)
    pos, rot, twist = segment.atlas.pose_and_twist(value, rate)
    return pos[0], rot[0], twist[0], acc[0]


def join_error(left, right, elapsed):
    p0, r0, v0, a0 = endpoint_state(left, elapsed)
    p1, r1, v1, a1 = endpoint_state(right, 0.)
    return np.array([np.linalg.norm(p1 - p0) * 1000., np.rad2deg((r1 * r0.inv()).magnitude()),
                     np.linalg.norm(v1[:3] - v0[:3]) * 1000.,
                     np.rad2deg(np.linalg.norm(v1[3:] - v0[3:])),
                     np.linalg.norm(a1[:3] - a0[:3]) * 1000.,
                     np.rad2deg(np.linalg.norm(a1[3:] - a0[3:]))])


JOIN_KEYS = ("position_mm", "orientation_deg", "linear_velocity_mm_s", "angular_velocity_deg_s",
             "linear_acceleration_mm_s2", "rotation_chart_acceleration_deg_s2")


def audit_reference(record, baseline, scan, transform):
    atlas, slot = record.atlas, record.slot_s
    states, segments = record.states, record.segments
    _, old_paths, _ = reconstruct(scan, baseline, states[0])
    dense = old_paths[1]
    old = dense @ transform[:3, :3].T + transform[:3, 3]
    joins = np.array([join_error(a, b, slot) for a, b in zip(segments, segments[1:])])
    assert len(joins) and joins.max() < 1.e-7
    position_knots = atlas.targets[..., :3, 3]
    old_drift_in, old_drift_out = [], []
    offsets, ids = np.unique(atlas.offsets, return_index=True)
    piecewise = RegularGridInterpolator((atlas.angles, atlas.stations, offsets), position_knots[:, :, ids])
    grid = np.linspace(0., slot, old.shape[1])
    new, twists, clock_times, deviation = [], [], [], []
    p_err, r_err, fd_xyz, fd_omega = [], [], [], []
    for start, end, segment in zip(states, states[1:], segments):
        p, q = start[0], end[0]
        old_drift_out.append((position_knots[(q, *start[1:])] - position_knots[start]) / slot)
        old_drift_in.append((position_knots[end] - position_knots[(p, *end[1:])]) / slot)
        value, rate, _ = segment.world_jet(grid, acceleration=False)
        points, rot, twist = atlas.pose_and_twist(value, rate)
        target = atlas.targets[end]
        p_err.append(np.linalg.norm(points[-1] - target[:3, 3]))
        r_err.append((rot[-1] * Rotation.from_matrix(target[:3, :3]).inv()).magnitude())
        absolute = segment.time_s + grid
        material = segment.material_jet(grid)[0]
        linear = piecewise(np.column_stack((record.clock.sample(absolute)[0], material[:, :2])))
        deviation.extend(pairs_difference(points, linear))
        new.append(points)
        twists.append(twist)
        clock_times.append(absolute)

        # Check analytic feedforward against SE(3) finite differences, including moving-body terms.
        dt = 1.e-4
        at = slot * np.array([.17, .49, .78])
        minus, minus_rate, _ = segment.world_jet(at - dt, acceleration=False)
        plus, plus_rate, _ = segment.world_jet(at + dt, acceleration=False)
        xm, rm, _ = atlas.pose_and_twist(minus, minus_rate)
        xp, rp, _ = atlas.pose_and_twist(plus, plus_rate)
        v, vrate, _ = segment.world_jet(at, acceleration=False)
        _, _, expected = atlas.pose_and_twist(v, vrate)
        fd_xyz.extend(pairs_difference((xp - xm) / (2. * dt), expected[:, :3]))
        fd_omega.extend(pairs_difference((rp * rm.inv()).as_rotvec() / (2. * dt), expected[:, 3:]))

    old_jumps = pairs_difference(old_drift_in[:-1], old_drift_out[1:])
    assert max(p_err) < 1.e-10 and max(r_err) < 1.e-8
    assert max(fd_xyz) < 1.e-6 and max(fd_omega) < 1.e-6
    old_v = np.gradient(old, grid, axis=1, edge_order=2)
    twist = np.asarray(twists)
    return {
        "moves": len(segments),
        "joins_checked": len(joins),
        "old_linear_velocity_jump_max_mm_s": float(1000. * old_jumps.max()),
        "max_join_error": dict(zip(JOIN_KEYS, joins.max(axis=0).tolist())),
        "max_capture_error_mm": float(1000. * max(p_err)),
        "max_capture_orientation_error_deg": float(np.rad2deg(max(r_err))),
        "max_fd_linear_error_mm_s": float(1000. * max(fd_xyz)),
        "max_fd_angular_error_deg_s": float(np.rad2deg(max(fd_omega))),
        "surface_interpolation_difference_mm_p95_max": (1000. * np.percentile(deviation, [95, 100])).tolist(),
        "peak_linear_speed_mm_s": float(1000. * np.linalg.norm(twist[..., :3], axis=-1).max()),
        "peak_angular_speed_deg_s": float(np.rad2deg(np.linalg.norm(twist[..., 3:], axis=-1).max())),
        "segments_using_learned_shape": sum(s.use_shape and r["path_skill"] != "endpoint_only"
                                             for s, r in zip(segments, baseline["steps"])),
        "material_corridor_fallbacks": sum(s.shape is not None and not s.use_shape for s in segments),
    }, (np.asarray(clock_times), old, np.asarray(new), old_v, twist[..., :3])


def causal_fork(record, row, scan, model, planner):
    """Find two public observation outcomes that the existing B policy routes differently."""
    from predictive import predictive_sets

    study = rollout.study
    belief = rollout._initial_belief(scan)
    book = rollout.branches.DetectionBook(model, gate="predictive", predictive_z=2.)
    for step in row["prefix_steps"]:
        belief = book.assimilate(
            belief, step["pose"], tuple(step["action"]), step["observation_mm"], step["eval_only_labels"],
        )
    arm = study.ARMS["B_seen"]
    for i, step in enumerate(row["steps"][:-1]):
        previous, current, following = record.states[i:i + 3]
        pose, action, next_pose = current[0], tuple(current[1:]), following[0]
        before = book.status(belief)[0]
        if not step["tail"]:
            index, program = planner.plan(book, pose, next_pose, previous, belief, before, arm, np.inf, False)
            assert tuple(planner.actions[pose][index]) == action
            if program is not None:
                outcomes = predictive_sets(
                    belief, model.model_cache[current], half_width=study.WIDTH, p_detect=study.PD,
                    noise_mixture=study.STRESS.noise_mixture, clutter_rate=study.STRESS.clutter_rate, count=8,
                )
                choices = {}
                for observed, weight in outcomes:
                    if weight <= 0.:
                        continue
                    copy = book.copy()
                    posterior = copy.assimilate(belief, pose, action, observed, [-1] * len(observed))
                    after, _, ready, _ = copy.status(posterior)
                    if np.all(ready):
                        continue
                    planner.public_book = copy
                    selected = planner.choose(program, next_pose, current, posterior, before, after, arm, False)
                    goal = (next_pose, *map(int, planner.actions[next_pose][selected.index]))
                    choices.setdefault(goal, [float(x) for x in observed])
                    if len(choices) == 2:
                        break
                if len(choices) == 2:
                    incoming = record.segments[i]
                    switch = incoming.time_s + incoming.duration
                    frozen = incoming.world_jet(np.linspace(0., incoming.duration, 65))[0].copy()
                    jet = incoming.material_jet(incoming.duration)[:, 0]
                    cases = []
                    for goal, observed in choices.items():
                        skill = planner.motion(current, next_pose, tuple(goal[1:]), True).skill
                        tail = TimedProbe(record.atlas, record.clock, record.shapes.get(skill), switch,
                                          record.slot_s, jet, record.atlas.material(goal))
                        error = join_error(incoming, tail, incoming.duration)
                        assert error.max() < 1.e-7
                        cases.append({"observation_mm": observed, "selected": list(goal),
                                      "join_error": dict(zip(JOIN_KEYS, error.tolist()))})
                    assert np.array_equal(frozen, incoming.world_jet(np.linspace(0., incoming.duration, 65))[0])
                    return {"same_probe": list(current), "probe_index": i + 1, "cases": cases,
                            "prefix_samples_bit_identical": True}
        belief = book.assimilate(belief, pose, action, step["centroids_mm"], step["eval_only_labels"])
    raise RuntimeError("the fixture no longer contains an observation-dependent fork")


def midcourse_switch(record):
    left = record.segments[1]
    elapsed = .53 * left.duration
    state = left.material_jet(elapsed)[:, 0]
    new_goal = left.goal.copy()
    new_goal[1] = 0.
    target_change = float(np.linalg.norm(left.goal - new_goal))
    assert target_change > 0.
    right = TimedProbe(record.atlas, record.clock, left.shape, left.time_s + elapsed,
                       left.duration - elapsed, state, new_goal)
    error = join_error(left, right, elapsed)
    material_speed = float(1000. * np.linalg.norm(state[1, :2]))
    assert material_speed > 0. and error.max() < 1.e-7
    np.testing.assert_allclose(right.material_jet(right.duration)[0, 0], new_goal, atol=1.e-12, rtol=0.)
    return {"incoming_relative_speed_mm_s": material_speed,
            "lateral_target_change_mm": 1000. * target_change,
            "join_error": dict(zip(JOIN_KEYS, error.tolist()))}


def render(path, traces, slot):
    fig = make_subplots(
        rows=2, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}], [{"colspan": 2}, None]],
        subplot_titles=("原世界插值", "修复后的同目标参考", "RL 第 14–16 次补采附近 · 世界 Y 速度"),
        vertical_spacing=.13, horizontal_spacing=.06, row_heights=[.66, .34],
    )
    for arm, (times, old, new, old_v, new_v) in traces.items():
        name, color = ARMS[arm]
        for col, points in enumerate((old, new), 1):
            x, y, z = (1000. * points.reshape(-1, 3)).T
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, name=name, mode="lines",
                                       line={"color": color, "width": 4}, legendgroup=arm,
                                       showlegend=col == 1, hoverinfo="skip"), row=1, col=col)
            x, y, z = (1000. * points[:, -1]).T
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, name=name, mode="markers",
                                       marker={"color": color, "size": 3}, legendgroup=arm, showlegend=False,
                                       text=[f"补采 {i}" for i in range(1, len(points) + 1)]), row=1, col=col)
        if arm == "rl":
            for label, speed, color in (("原参考", old_v, "#777777"), ("补丁", new_v, "#b13e5d")):
                fig.add_trace(go.Scatter(
                    x=(times[13:16] - 13. * slot).reshape(-1),
                    y=(1000. * speed[13:16, :, 1]).reshape(-1), name=label, line={"color": color},
                ), row=2, col=1)
    for t in (slot, 2. * slot):
        fig.add_shape(type="line", x0=t, x1=t, y0=0., y1=1.,
                      xref="x", yref="y domain", line={"dash": "dot", "color": "#888"})
    fig.update_scenes(aspectmode="data", camera={"eye": {"x": 1.6, "y": 1.6, "z": 1.6}},
                      xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)")
    fig.update_xaxes(title_text="相对时间 (s)", row=2, col=1)
    fig.update_yaxes(title_text="参考 Y 速度 (mm/s)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=880,
                      title=f"同一病例的 ProDMP 平滑衔接 · {SEED}<br>"
                            "<sup>3D 曲线为参考 · 圆点为同一采集位姿 · 速度图使用共同合成时钟</sup>",
                      legend={"orientation": "h", "y": -.1, "groupclick": "togglegroup"})
    fig.write_html(path, include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot-s", type=float, default=30.)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if not np.isfinite(args.slot_s) or args.slot_s <= 0.:
        parser.error("--slot-s must be positive and finite")
    baseline = load_rows([HERE / "control_24", HERE / "eval_q_24"], [])[SEED]
    scan, support, model, planner = rollout._make_model_planner(n_predictive=8, n_first=3, switch_margin=.006)
    with np.load(GEOMETRY, allow_pickle=False) as geometry:
        atlas, transform = SmoothAtlas(geometry), geometry["robot_from_atlas"]
    shapes = learned_shapes(scan.primitive)
    checkpoint = rollout._load_q("rl", HERE / "frozen_q" / "POLICY.npz")
    report = {"seed": SEED, "clock_slot_s": args.slot_s, "arms": {}}
    traces, runs, records = {}, {}, {}
    for policy in ("rule", "rl"):
        record = ReferenceRecorder(atlas, shapes, scan.moving, args.slot_s)
        run, _ = rollout.run_episode(
            planner, scan, support, model, episode_seed=SEED, mode="evaluate", policy=policy,
            q_model=checkpoint if policy == "rl" else None, cap=80, reference_step=record.append,
        )
        for key in ("complete", "online_stop", "incorrect_witnesses"):
            assert run[key] == baseline[policy][key]
        assert len(run["steps"]) == len(baseline[policy]["steps"])
        for new, old in zip(run["steps"], baseline[policy]["steps"]):
            for key in ("pose", "action", "centroids_mm", "eval_only_labels", "evidence_gains", "changed_suffix"):
                assert new[key] == old[key], (policy, key)
        results, traces[policy] = audit_reference(record, baseline[policy], scan, transform)
        report["arms"][policy] = dict(results, public_trace_matches=True, strict_complete=run["complete"],
                                     adaptive_tail_switches=run["actual_switch_count"])
        runs[policy], records[policy] = run, record
        print(policy, len(record.segments), "segments checked", flush=True)
    report["causal_b_fork"] = causal_fork(records["rule"], runs["rule"], scan, model, planner)
    report["nonzero_velocity_replan"] = midcourse_switch(records["rule"])
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.dumps(rollout._jsonable(report), indent=2, allow_nan=False)
    (args.out / "SUMMARY.json").write_text(summary + "\n")
    render(args.out / "SMOOTH_SWITCH_3D.html", traces, args.slot_s)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
