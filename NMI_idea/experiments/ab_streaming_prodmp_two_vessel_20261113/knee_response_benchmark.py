"""Paired acquisitions, posterior replay and moving-knee ProDMP MPC."""

import argparse
from dataclasses import replace
import json
import lzma
from pathlib import Path

import numpy as np

from atlas_geometry import REFERENCE_KNEE_DEG, ScanTemplate
from atlas_scan import FRAME_COUNT, FRAME_DT_S, HERE, load_witness, sha
from knee_response_scan import KneeAtlas, KneeScan, KneeScene, RESPONSE_ARM, central_interval
from primitive import FRAME_STRIDE, SAMPLE_DT_S, TOTAL_STEPS
from residual import motion_tape
from sensor import CENTER_TOL_MM, IDENTITY_MIN


SEED = 2026111513
BASE = "mpc_ou"
ARMS = (BASE, RESPONSE_ARM)
FIXED_ANGLE = REFERENCE_KNEE_DEG
ACQUISITIONS = {
    "single": ("ref_0",),
    "same_angle": ("ref_0", "ref_1", "ref_2"),
    "multiple_angles": ("ref_0", "low_1", "high_2"),
}
VISITS = {
    "ref_0": (FIXED_ANGLE, 0),
    "ref_1": (FIXED_ANGLE, 1),
    "ref_2": (FIXED_ANGLE, 2),
    "low_1": (5., 1),
    "high_2": (25., 2),
}
HORIZONS = (1, 5, 10)


def heldout_angles(direction, duration):
    time_s = np.arange(TOTAL_STEPS + 1) * SAMPLE_DT_S
    phase = np.clip(time_s / duration, 0., 1.)
    return FIXED_ANGLE + direction * 8. * np.sin(np.pi * phase) ** 2


def slope_statistics(model, belief):
    weights = model.slope_distribution(belief)
    grid = model.slope_grid
    mean = float(weights @ grid)
    return dict(mean=mean, sd=float(np.sqrt(weights @ (grid - mean) ** 2)),
                interval90=central_interval(grid, weights).tolist(),
                weights=weights.tolist())


def public_history(run, poses):
    for frame, row in enumerate(run["frames"]):
        yield frame, poses[frame * FRAME_STRIDE], np.asarray(row["packet_mm"]).reshape(-1, 2), row


def learn_from_visit(scan, run, poses, models, priors):
    beliefs = {}
    for arm, model in models.items():
        belief = priors[arm]
        for frame, pose, packet, _ in public_history(run, poses):
            means = scan.project_hypotheses(frame * FRAME_STRIDE, pose, model)
            belief, _ = model.update_projected(belief, packet, means)
        beliefs[arm] = belief
    return beliefs


def restart(models, beliefs):
    return {arm: model.start_scan(beliefs[arm]) for arm, model in models.items()}


def common_history_replay(scan, run, poses, models, priors, scene, motion):
    """Score all forecasts and associations on the same realized public FK and packets."""
    results = {name: {arm: {"forecasts": [], "association": []} for arm in models} for name in priors}
    live = {name: beliefs.copy() for name, beliefs in priors.items()}
    for frame, pose, packet, row in public_history(run, poses):
        step = frame * FRAME_STRIDE
        labels = np.asarray(row["role_truth"]) == 0
        for arm, model in models.items():
            means = scan.project_hypotheses(step, pose, model)
            for name in priors:
                result = results[name][arm]
                belief, identity = model.update_projected(live[name][arm], packet, means)
                live[name][arm] = belief
                result["association"].append(identity.tolist())
                for horizon in HORIZONS:
                    future = frame + horizon
                    if future >= FRAME_COUNT:
                        continue
                    future_step = future * FRAME_STRIDE
                    feature = scan.feature_at(future_step)
                    mean, interval = scan.offset_forecast(future_step, model, belief, horizon)
                    truth = scene.sensor.offset[0] + motion[future] + scene.slope_mm_per_deg * feature
                    result["forecasts"].append(dict(
                        origin=frame, horizon=horizon, mean=mean, interval90=interval.tolist(), truth=float(truth),
                        active=bool(abs(feature) >= 1.), missing_artery=bool(not labels.any()),
                    ))
    for name in priors:
        results[name]["heldout_posterior"] = slope_statistics(models[RESPONSE_ARM], live[name][RESPONSE_ARM])
    return results


def frozen_visit(sensor, case, visit, cohort, persistent, slope):
    # Visit seeds are indexed independently of the chosen angle and estimator.
    world_code = int(cohort == "null")
    packet_rng = np.random.default_rng([SEED, world_code, case, visit, 1])
    motion_rng = np.random.default_rng([SEED, world_code, case, visit, 2])
    tape = replace(sensor.scene(packet_rng, FRAME_COUNT), offset=persistent.offset, bias=persistent.bias)
    return KneeScene(tape, slope), motion_tape(motion_rng, FRAME_COUNT, FRAME_DT_S, "ou")


def evaluate_case(case, cohort, calibration_scans, heldout_scans, traces):
    direction = 1 if case % 2 == 0 else -1
    duration = (2.4, 2.8)[(case // 2) % 2]
    scan = heldout_scans[direction, duration]
    model = {arm: scan.models[arm] for arm in ARMS}
    patient_rng = np.random.default_rng([SEED, int(cohort == "null"), case, 0])
    persistent = scan.sensor.scene(patient_rng, 0)
    slope = 0. if cohort == "null" else float(patient_rng.choice([-1., 1.]) * patient_rng.uniform(.12, .24))
    record = dict(
        case=case, cohort=cohort, true_slope_mm_per_deg=slope, holdout_direction=direction,
        holdout_duration_s=duration, effective_static_mm=persistent.offset.tolist(),
        calibration_bias_mm=persistent.bias.tolist(), visits={}, histories={},
    )
    # Paired histories share the reference prefix and each visit's disturbance tape.
    calibration = {}
    prefix_traces = f"{cohort}_{case}"
    for key, (angle, visit) in VISITS.items():
        acquisition_scan = calibration_scans[angle]
        scene, motion = frozen_visit(scan.sensor, case, visit, cohort, persistent, slope)
        result, (joints, poses, lateral) = acquisition_scan.execute(scene, BASE, motion)
        archive_trace(traces, f"{prefix_traces}_{key}", scene, motion, joints, poses, lateral)
        record["visits"][key] = result
        calibration[key] = (acquisition_scan, result, poses)
    priors = {arm: state.prior() for arm, state in model.items()}
    initial = learn_from_visit(*calibration["ref_0"], model, priors)
    np.testing.assert_allclose(model[RESPONSE_ARM].slope_distribution(initial[RESPONSE_ARM]),
                               model[RESPONSE_ARM].slope_prior, rtol=0., atol=1.e-13)
    scene, motion = frozen_visit(scan.sensor, case, 3, cohort, persistent, slope)
    next_run = {}
    for name, visits in ACQUISITIONS.items():
        belief = initial.copy()
        for visit in visits[1:]:
            belief = learn_from_visit(*calibration[visit], model, restart(model, belief))
        record["histories"][name] = dict(
            posterior=slope_statistics(model[RESPONSE_ARM], belief[RESPONSE_ARM]), arms={})
        next_run[name] = restart(model, belief)
        for arm in ARMS:
            outcome, (joints, poses, lateral) = scan.execute(
                scene, arm, motion, initial_belief=next_run[name][arm])
            record["histories"][name]["arms"][arm] = outcome
            key = f"{prefix_traces}_{name}_{arm}"
            archive_trace(traces, key, scene, motion, joints, poses, lateral,
                          store_tape=(name == "multiple_angles" and arm == BASE))
            if name == "multiple_angles" and arm == BASE:
                common = outcome, poses
    record["replay"] = common_history_replay(scan, *common, model, next_run, scene, motion)
    return record


def archive_trace(traces, key, scene, motion, joints, poses, lateral, *, store_tape=True):
    traces[f"{key}_q"] = joints
    traces[f"{key}_fk_m"] = poses
    traces[f"{key}_x_v"] = lateral
    if store_tape:
        traces[f"{key}_clear"] = scene.sensor.clear
        traces[f"{key}_noise_mm"] = scene.sensor.noise
        traces[f"{key}_z_mm"] = motion


def build_scans(atlas, capability, witness, regression):
    calibration = {angle: KneeScan(atlas, capability, witness, regression,
                                   np.full(TOTAL_STEPS + 1, angle))
                   for angle in sorted({angle for angle, _ in VISITS.values()})}
    evaluation = {(direction, duration): KneeScan(atlas, capability, witness, regression,
                                                   heldout_angles(direction, duration))
                  for direction in (-1, 1) for duration in (2.4, 2.8)}
    return calibration, evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--among-us", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", choices=("response", "null"), default="response")
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--first-case", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cases < 1 or args.first_case < 0:
        parser.error("require --cases > 0 and --first-case >= 0")
    from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import CapabilityEvaluator

    args.out.mkdir(parents=True, exist_ok=True)
    atlas = KneeAtlas(ScanTemplate(args.among_us, args.repair))
    print("atlas interpolation checked", flush=True)
    bank = HERE.parent / "ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz"
    demonstrations = HERE / "verification/DEMONSTRATIONS.npz"
    capability = CapabilityEvaluator(device="cpu")
    witness_id, witness = load_witness(bank)
    with np.load(demonstrations, allow_pickle=False) as frozen:
        calibration, evaluation = build_scans(atlas, capability, witness, frozen["shape_regression"])
    traces = dict(station_ref_mm=calibration[FIXED_ANGLE].path, envelope=atlas.envelope,
                  vessel_stations_mm=atlas.template.s_mm, witness_q=witness)
    for (direction, duration), scan in evaluation.items():
        traces[f"knee_tape_{direction}_{duration}"] = scan.body_angles
    model = calibration[FIXED_ANGLE].models[RESPONSE_ARM]
    report = dict(
        scope="Synthetic anonymous centroids; programmed observable body angles; sampled RM75 FK and in-segment ProDMP",
        cohort=args.cohort, seed=SEED, calibration_histories=ACQUISITIONS, visits=VISITS,
        common_history="multiple_angles / mpc_ou", slope_grid_mm_per_deg=model.slope_grid.tolist(),
        slope_prior=model.slope_prior.tolist(), frame_dt_s=FRAME_DT_S, joint_dt_s=SAMPLE_DT_S,
        horizon_frames=HORIZONS, acquisition_frames=FRAME_COUNT,
        atlas_angles_deg=atlas.angles.tolist(), knee_side_station_mm=atlas.knee_station_mm,
        closest_artery_to_knee_joint_mm=atlas.knee_distance_mm, interpolation=atlas.interpolation_checks,
        shape_geometry=atlas.reference.metadata, kinematic_witness_id=witness_id,
        model_input_sha256={path.name: sha(path) for path in (bank, demonstrations)},
        cases=[],
    )
    for case in range(args.first_case, args.first_case + args.cases):
        record = evaluate_case(case, args.cohort, calibration, evaluation, traces)
        report["cases"].append(record)
        summary = record["histories"]["multiple_angles"]
        print(f"{args.cohort} case {case}: k={record['true_slope_mm_per_deg']:.3f}, "
              f"estimate={summary['posterior']['mean']:.3f}, "
              f"success={[summary['arms'][arm]['success'] for arm in ARMS]}", flush=True)
    with lzma.open(args.out / "CASES.json.xz", "wt", preset=3) as file:
        json.dump(report, file, allow_nan=False, separators=(",", ":"))
        file.write("\n")
    np.savez_compressed(args.out / "TRACES.npz", **traces)


if __name__ == "__main__":
    main()
