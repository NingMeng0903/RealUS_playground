"""Rebuild paired pilot measurements, updates, motion and causal behavior."""

import argparse
from dataclasses import replace
import json
import lzma
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_filter import (CENTER_TOL_MM, FOV_MAX_MM, FOV_MIN_MM, FRAME_STRIDE, IDENTITY_MIN,
                          SAMPLE_DT_S, configure_experiment, geometry_indices, independent_condition,
                          pointwise_project)


EXT = Path("/media/camp/EXT_DRIVE")
REPAIR = EXT / "RealUS_playground/NMI_idea/experiments/ab_validation_20261110/repair_20261110"
BASE = "mpc_ou"
RESPONSE_ARM = "mpc_ou_g"
ARMS = (BASE, RESPONSE_ARM)


def rebuild_packet(frame, pose, geometry, envelope, angle, scene, motion):
    response = np.array([scene.slope_mm_per_deg * (angle - 15.)])
    offset = scene.sensor.offset + [motion, 0.]
    measured = pointwise_project(geometry, pose, offset[None], response, envelope)[0]
    physical = pointwise_project(geometry, pose, (offset - scene.sensor.bias)[None], response, envelope)[0]
    raw = measured + scene.sensor.noise[frame]
    visible = scene.sensor.clear[frame] & np.all(raw >= FOV_MIN_MM, axis=-1) & np.all(raw <= FOV_MAX_MM, axis=-1)
    selected = np.flatnonzero(visible)
    selected = selected[np.argsort(raw[selected, 0], kind="stable")]
    return raw[selected], selected, physical[0, 0]


def prove_run(key, run, traces, angles, atlas, capability, scene, motion, extrema):
    from primitive import Plan, admissible

    q, fk, xv = (traces[f"{key}_{suffix}"] for suffix in ("q", "fk_m", "x_v"))
    np.testing.assert_allclose(fk, np.stack([capability.fk(item) for item in q]), rtol=0., atol=1.e-13)
    path = traces["station_ref_mm"]
    target = atlas.poses(path[:len(q)], xv[:, 0], angles[:len(q)])
    for q_sample, pose in zip(q, target, strict=True):
        check = capability.check_q(q_sample, pose)
        assert check["valid"]
        extrema["fk_mm"] = max(extrema["fk_mm"], 1000. * check["fk_position_error_m"])
        extrema["fk_deg"] = max(extrema["fk_deg"], np.rad2deg(check["fk_rotation_error_rad"]))
        extrema["self_clearance_mm"] = min(extrema["self_clearance_mm"],
                                           1000. * check["min_self_collision_distance_m"])

    assert admissible(Plan(0, xv[:, 0], xv[:, 1]))
    max_rate = np.max(np.abs(np.diff(q, axis=0)) / (SAMPLE_DT_S * capability.kin.v_max), initial=0.)
    assert max_rate <= 1. + 1.e-10
    extrema["max_rate_fraction"] = max(extrema["max_rate_fraction"], max_rate)
    correct_bins, accepted_bins = set(), set()
    any_wrong = False
    count_revisions = 0
    for frame, saved in enumerate(run["frames"]):
        step = frame * FRAME_STRIDE
        angle = angles[step]
        physical_geometry = atlas.vessels(angle, physical=True)
        packet, roles, aria = rebuild_packet(frame, fk[step], physical_geometry, atlas.envelope, angle,
                                             scene, motion[frame])
        np.testing.assert_allclose(packet, np.asarray(saved["packet_mm"]).reshape(-1, 2), rtol=0., atol=1.e-10)
        np.testing.assert_array_equal(roles, saved["role_truth"])
        if np.isfinite(aria):
            np.testing.assert_allclose(saved["physical_lateral_mm"], aria, rtol=0., atol=1.e-10)
        else:
            assert saved["physical_lateral_mm"] is None
            extrema["physical_misses"] += 1
        probs = np.asarray(saved["artery_probability"])
        slot = int(probs.argmax()) if len(probs) and probs.max() >= IDENTITY_MIN else -1
        accepted = slot >= 0 and abs(packet[slot, 0]) <= CENTER_TOL_MM
        wrong = bool(accepted and roles[slot] != 0)
        correct = bool(accepted and not wrong and np.isfinite(aria) and abs(aria) <= CENTER_TOL_MM)
        assert (slot, accepted, correct, wrong) == (
            saved["selected_slot"], saved["accepted"], saved["correct"], saved["wrong"]
        )
        any_wrong |= wrong
        bin_number = min(4, int(5 * (path[step] - path[0]) / (path[-1] - path[0])))
        if correct:
            correct_bins.add(bin_number)
        if accepted:
            accepted_bins.add(bin_number)
        count_revisions += saved.get("next_frame_shift_from_revision_mm", 0.) >= .001
    complete = run["failure"] is None and len(correct_bins) == 5 and not any_wrong
    assert run["success"] == complete
    assert run["wrong"] == any_wrong
    assert run["coverage"] == len(correct_bins)
    assert run["false_complete"] == (len(accepted_bins) == 5 and not complete)
    assert run["revisions"] == count_revisions
    assert run["max_chart_handoff_x_ref_mm"] <= 1.e-10
    assert run["max_chart_handoff_v_ref_mm_s"] <= 1.e-10
    rotations = Rotation.from_matrix(fk[:, :3, :3])
    turn = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude()).sum()
    travel = 1000. * np.linalg.norm(np.diff(fk[:, :3, 3], axis=0), axis=1).sum()
    np.testing.assert_allclose([run["tcp_mm"], run["turn_deg"], run["duration_s"]],
                               [travel, turn, (len(q) - 1) * SAMPLE_DT_S], rtol=0., atol=1.e-10)
    extrema["paths"] += 1
    extrema["robot_samples"] += len(q)
    extrema["exposures"] += len(run["frames"])
    extrema["effective_revisions"] += count_revisions
    return q, fk, xv


def restore_scene(record, traces, tape_key):
    from knee_response_scan import KneeScene
    from sensor import Scene

    return KneeScene(Scene(np.asarray(record["effective_static_mm"]), np.asarray(record["calibration_bias_mm"]),
                            traces[f"{tape_key}_clear"], traces[f"{tape_key}_noise_mm"]),
                     record["true_slope_mm_per_deg"]), traces[f"{tape_key}_z_mm"]


def restart_independent(model, fit):
    return (fit.sum(axis=(1, 3))[:, None, :, None]
            * model.residual.z_prior[None, :, None, None]
            * model.visibility_prior[None, None, None, :])


def replay_response(model, beliefs, run, poses, angles, atlas, *, compare_saved=False):
    index = geometry_indices(model)
    saved_errors = []
    for frame, row in enumerate(run["frames"]):
        step = frame * FRAME_STRIDE
        angle = angles[step]
        geometry = atlas.vessels(angle)
        means = pointwise_project(geometry, poses[step], model.offsets_mm,
                                  model.slopes_mm_per_deg * (angle - 15.), atlas.envelope)
        packet = np.asarray(row["packet_mm"]).reshape(-1, 2)
        for key, belief in beliefs.items():
            updated, identity = independent_condition(model, belief, packet, means, index)
            beliefs[key] = updated
            if compare_saved:
                expected = np.asarray(row["artery_probability"])
                error = float(np.max(np.abs(expected - identity), initial=0.))
                np.testing.assert_allclose(expected, identity, rtol=0., atol=3.e-13)
                np.testing.assert_allclose(row["slope_mean_mm_per_deg"], updated.sum(axis=(0, 1, 3)) @ model.slope_grid,
                                           rtol=0., atol=1.e-13)
                saved_errors.append(error)
    return max(saved_errors, default=0.)


def check_time_hooks(scan):
    """Trace how candidate timing and feedback index their geometry."""
    from primitive import admissible

    original = {name: getattr(scan, name) for name in ("acquisition_score", "project_hypotheses",
                                                       "project_estimate", "revise")}
    events = {"candidate_scores": 0, "feedback_attempts": 0, "validated_replacements": 0}
    replacements = {}
    active_calls = {"score": None, "revision": None}

    def check_score(trajectory, model, belief, frames):
        start = trajectory.plan.step
        calls = []
        active_calls["score"] = calls
        result = original["acquisition_score"](trajectory, model, belief, frames)
        active_calls["score"] = None
        assert [t for t, _ in calls] == [start + f * FRAME_STRIDE for f in (1, 6, 10)]
        assert all(np.array_equal(pose, trajectory.poses[step - start]) for step, pose in calls)
        events["candidate_scores"] += 1
        return result

    def check_revise(old, step, end, x, v, q, model, belief, pose, packet, slot):
        calls = []
        active_calls["revision"] = calls
        new = original["revise"](old, step, end, x, v, q, model, belief, pose, packet, slot)
        active_calls["revision"] = None
        forecast = (end - step) // FRAME_STRIDE
        assert [(t, n) for t, n, _ in calls[:3]] == [(end, forecast)] * 3
        np.testing.assert_allclose(calls[1][2], scan.poses_at(end, scan.nominal_x[end]), rtol=0., atol=1.e-13)
        if len(calls) > 3:
            assert len(calls) == 4
            t, n, observed = calls[3]
            assert (t, n) == (step, 0)
            np.testing.assert_array_equal(observed, pose)
        if new is not old:
            events["validated_replacements"] += 1
            assert admissible(new.plan)
            assert len(new.plan.x) == end - step + 1
            assert new.plan.step == step
            np.testing.assert_allclose([x, v], [new.plan.x[0], new.plan.v[0]], rtol=0., atol=1.e-13)
            np.testing.assert_allclose(new.plan.v[-1], scan.nominal_v[end], rtol=0., atol=3.e-13)
            np.testing.assert_array_equal(new.joints[0], q)
            replacements[step] = new.poses[FRAME_STRIDE]
        events["feedback_attempts"] += 1
        return new

    def hypotheses(step, pose, model):
        calls = active_calls["score"]
        if calls is not None:
            calls.append((step, pose))
        return original["project_hypotheses"](step, pose, model)

    def estimate(step, pose, model, belief, frames=0):
        calls = active_calls["revision"]
        if calls is not None:
            calls.append((step, frames, pose))
        return original["project_estimate"](step, pose, model, belief, frames)

    scan.acquisition_score = check_score
    scan.project_hypotheses = hypotheses
    scan.project_estimate = estimate
    scan.revise = check_revise
    return original, events, replacements


def check_causal(scan, learned, scene, motion, pilot_run, pilot_state):
    from knee_response_scan import KneeScene

    hooks, events, replacements = check_time_hooks(scan)
    run, state = scan.execute(scene, RESPONSE_ARM, motion, initial_belief=learned)
    for name, method in hooks.items():
        setattr(scan, name, method)
    for actual, archived in zip(state, pilot_state, strict=True):
        np.testing.assert_allclose(actual, archived, rtol=0., atol=1.e-13)
    assert run["revisions"] == pilot_run["revisions"]
    assert run["revisions"] > 0
    assert events["validated_replacements"] >= run["revisions"]
    for step, expected in replacements.items():
        np.testing.assert_array_equal(state[1][step + FRAME_STRIDE], expected)

    observation = scan.observation
    def mutate_labels_and_truth(scene, frame, pose, z):
        packet, order, truth = observation(scene, frame, pose, z)
        return packet, 1 - order, truth + 18.
    scan.observation = mutate_labels_and_truth
    manipulated, changed_labels = scan.execute(scene, RESPONSE_ARM, motion, initial_belief=learned)
    scan.observation = observation
    for unchanged, test in zip(state, changed_labels, strict=True):
        np.testing.assert_array_equal(unchanged, test)
    assert manipulated["decisions"] == run["decisions"]
    assert manipulated["wrong"] != run["wrong"] or manipulated["coverage"] != run["coverage"]

    boundary = next(frame for frame in range(14, 20) if run["frames"][frame]["packet_mm"])
    dropout = scene.sensor.clear.copy()
    dropout[boundary:] = False
    noise = scene.sensor.noise.copy()
    noise[boundary:] += 10.
    private_motion = motion.copy()
    private_motion[boundary:] += 4.
    future = KneeScene(replace(scene.sensor, clear=dropout, noise=noise), scene.slope_mm_per_deg)
    changed_run, changed_state = scan.execute(future, RESPONSE_ARM, private_motion, initial_belief=learned)
    sample_boundary = boundary * FRAME_STRIDE + 1
    for unchanged, test in zip(state, changed_state, strict=True):
        np.testing.assert_array_equal(unchanged[:sample_boundary], test[:sample_boundary])
    assert run["frames"][:boundary] == changed_run["frames"][:boundary]
    assert run["frames"][boundary]["packet_mm"] != changed_run["frames"][boundary]["packet_mm"]
    assert all(row["packet_mm"] == [] for row in changed_run["frames"][boundary:])
    events["causal_prefix_exposures"] = boundary
    return events


def audit_pilot(experiment, among, repair, pilot, case_id, causal):
    from atlas_geometry import ScanTemplate
    from atlas_scan import FRAME_DT_S, load_witness
    from knee_response_benchmark import learn_from_visit, restart
    from knee_response_scan import KneeAtlas, KneeScan
    from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import CapabilityEvaluator
    from pose_response import PoseResponseFilter
    from sensor import Sensor

    with lzma.open(pilot / "CASES.json.xz", "rt") as source:
        report = json.load(source)
    record = report["cases"][0] if case_id is None else next(row for row in report["cases"] if row["case"] == case_id)
    case_prefix = f"{record['cohort']}_{record['case']}"
    arrays = np.load(pilot / "TRACES.npz", allow_pickle=False)
    template = ScanTemplate(among, repair)
    atlas = KneeAtlas(template, check_midpoints=False)
    capability = CapabilityEvaluator(device="cpu")
    _, witness = load_witness(experiment.parent / "ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz")
    with np.load(experiment / "verification/DEMONSTRATIONS.npz", allow_pickle=False) as frozen:
        regression = frozen["shape_regression"]
    assert atlas.knee_station_mm == report["knee_side_station_mm"]
    np.testing.assert_array_equal(atlas.envelope, arrays["envelope"])
    direction, duration = record["holdout_direction"], record["holdout_duration_s"]
    q_tape = arrays[f"knee_tape_{direction}_{duration}"]
    phase = np.minimum(np.arange(len(q_tape)) * SAMPLE_DT_S / duration, 1.)
    np.testing.assert_array_equal(q_tape, 15. + 8. * direction * np.sin(np.pi * phase) ** 2)
    sensor = Sensor()
    model = PoseResponseFilter(sensor, FRAME_DT_S)
    stats = dict(case=record["case"], cohort=record["cohort"], fk_mm=0., fk_deg=0.,
                 self_clearance_mm=np.inf, max_rate_fraction=0., paths=0, robot_samples=0,
                 exposures=0, physical_misses=0, effective_revisions=0)
    history = {}
    for visit, run in record["visits"].items():
        angle, _ = report["visits"][visit]
        angles = np.full_like(q_tape, angle)
        key = f"{case_prefix}_{visit}"
        scene, motion = restore_scene(record, arrays, key)
        q, positions, _ = prove_run(key, run, arrays, angles, atlas, capability, scene, motion, stats)
        history[visit] = run, positions, angles
        assert len(run["frames"]) == report["acquisition_frames"] and len(q) == len(q_tape)
    for left, right in (("ref_1", "low_1"), ("ref_2", "high_2")):
        for field in ("clear", "noise_mm", "z_mm"):
            np.testing.assert_array_equal(arrays[f"{case_prefix}_{left}_{field}"],
                                          arrays[f"{case_prefix}_{right}_{field}"])

    start_fit = {"reference": model.prior()}
    replay_response(model, start_fit, *history["ref_0"], atlas)
    for label, visits in report["calibration_histories"].items():
        learned = {"result": start_fit["reference"]}
        for visit in visits[1:]:
            learned["result"] = restart_independent(model, learned["result"])
            replay_response(model, learned, *history[visit], atlas)
        posterior = learned["result"].sum(axis=(0, 1, 3))
        np.testing.assert_allclose(posterior, record["histories"][label]["posterior"]["weights"], rtol=0., atol=1.e-13)
        if label in ("single", "same_angle"):
            np.testing.assert_allclose(posterior, model.slope_prior, rtol=0., atol=1.e-13)
        history[label] = restart_independent(model, learned["result"])

    tape_key = f"{case_prefix}_multiple_angles_{BASE}"
    scene, motion = restore_scene(record, arrays, tape_key)
    initial_joint = initial_packet = None
    assoc = 0.
    for label, data in record["histories"].items():
        for arm, run in data["arms"].items():
            key = f"{case_prefix}_{label}_{arm}"
            q, poses, xv = prove_run(key, run, arrays, q_tape, atlas, capability, scene, motion, stats)
            packet = run["frames"][0]["packet_mm"]
            if initial_joint is None:
                initial_joint, initial_packet = q[0], packet
            np.testing.assert_array_equal(q[0], initial_joint)
            assert packet == initial_packet
            if arm == RESPONSE_ARM:
                assoc = max(assoc, replay_response(model, {"learned": history[label]}, run, poses, q_tape, atlas,
                                                   compare_saved=True))
                if label == "multiple_angles":
                    diverse_state = q, poses, xv
                    diverse_run = run
    stats["max_association_replay_error"] = assoc

    if causal:
        scan = KneeScan(atlas, capability, witness, regression, q_tape)
        models = {arm: scan.models[arm] for arm in ARMS}
        live = {arm: model.prior() for arm, model in models.items()}
        for visit in report["calibration_histories"]["multiple_angles"]:
            run, poses, angle_tape = history[visit]
            cscan = KneeScan(atlas, capability, witness, regression, angle_tape)
            live = learn_from_visit(cscan, run, poses, models, live)
            live = restart(models, live)
        stats.update(check_causal(scan, live[RESPONSE_ARM], scene, motion, diverse_run, diverse_state))
    return {name: value.item() if isinstance(value, np.generic) else value for name, value in stats.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--among-us", type=Path, default=EXT / "Among_US")
    parser.add_argument("--repair", type=Path, default=REPAIR)
    parser.add_argument("--pilot", type=Path, default=Path(__file__).resolve().parents[1] / "pilot")
    parser.add_argument("--case", type=int)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()
    experiment = configure_experiment(args.experiment)
    print(json.dumps(audit_pilot(experiment, args.among_us, args.repair, args.pilot, args.case, args.causal),
                     indent=2, allow_nan=False))
