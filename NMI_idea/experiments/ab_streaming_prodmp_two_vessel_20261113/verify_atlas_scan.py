"""Check saved scan histories against fixed material addresses, FK and the sensor tape."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from atlas_scan import FRAME_COUNT, FRAME_DT_S, RESIDUAL_ARMS, ScanTemplate, project, score, sha
from atlas_geometry import frame
from primitive import FRAME_STRIDE, SAMPLE_DT_S, Plan, admissible
from residual import ResidualFilter, motion_tape, offset_stats
from sensor import CENTER_TOL_MM, IDENTITY_MIN, Scene, Sensor


def coordinate_error(left, right):
    delta = np.asarray(left, float) - np.asarray(right, float)
    delta[..., 0] = np.angle(np.exp(1j * delta[..., 0]))
    return np.max(np.abs(delta.reshape(-1, 3)), axis=0)


def verify(template, capability, run):
    report_path, trace_path = run / "SUMMARY.json", run / "TRACES.npz"
    report = json.loads(report_path.read_text())
    metadata = report["geometry"]
    shape = template.instantiate((0., 0.), metadata["knee_deg"], beta=metadata["beta"])
    assert metadata == shape.metadata
    sensor = Sensor()
    motion = report.get("motion")
    models = {arm: ResidualFilter(sensor, FRAME_DT_S, name, motion["filter_grid_mm"])
              for arm, name in RESIDUAL_ARMS.items()} if motion else {}
    scene_rng = np.random.default_rng(report["seed"])
    motion_rng = np.random.default_rng(motion["seed"]) if motion else None
    assert len(report["cases"]) == report["case_count"]

    max_label_error = np.zeros(3)
    max_roundtrip_m, max_fk_mm, max_fk_deg, max_rate = 0., 0., 0., 0.
    max_relative_rotation_error = 0.
    min_clearance = float("inf")
    row_count, sample_count, frame_count = 0, 0, 0
    with np.load(trace_path, allow_pickle=False) as traces:
        if motion:
            np.testing.assert_array_equal(traces["offset_axes"], shape.offset_axes)
        np.testing.assert_allclose(traces["nominal_vessels_m"], shape.vessels_m, atol=1.e-12, rtol=0.)
        np.testing.assert_array_equal(traces["vessel_material_xi"], template.vessel_xi)
        max_label_error = coordinate_error(shape.xyz_to_xi(shape.vessels_m), template.vessel_xi)
        max_roundtrip_m = float(np.linalg.norm(shape.xi_to_xyz(template.vessel_xi) - shape.vessels_m,
                                                axis=-1).max())

        station = traces["sample_stations_ref_mm"]
        sample_pose = shape.pose(station, np.zeros_like(station))
        np.testing.assert_allclose(sample_pose, traces["sample_material_probe_poses"], atol=1.e-12, rtol=0.)
        for lateral in (-20., -10., 0.):
            rotation = shape.pose(station, lateral)[:, :3, :3]
            for s, physical in zip(station, rotation, strict=True):
                nodes, weights = template.surface_address(s, lateral)
                source_frame = frame(weights @ template.tangent[nodes], weights @ template.inward[nodes])
                target_frame = frame(weights @ shape.skin_tangent[nodes], weights @ shape.skin_inward[nodes])
                relative = (template.robot_from_atlas[:3, :3] @ target_frame).T @ physical
                expected_relative = source_frame.T @ template.reference_probe_frame(s).as_matrix()
                max_relative_rotation_error = max(max_relative_rotation_error,
                                                  float(np.max(np.abs(relative - expected_relative))))
        assert max_relative_rotation_error <= 1.e-12
        expected_h, expected_theta = [], []
        for value in station:
            nodes, weights = shape.surface_address(value, 0.)
            expected_h.append(weights @ template.chart.h.ravel()[nodes])
            expected_theta.append(np.angle(weights @ np.exp(1j * template.chart.theta.ravel()[nodes])))
        expected = np.column_stack((expected_theta, expected_h, np.zeros_like(station)))
        sample_labels = shape.xyz_to_xi(sample_pose[:, :3, 3])
        max_label_error = np.maximum(max_label_error, coordinate_error(sample_labels, expected))
        max_roundtrip_m = max(max_roundtrip_m, float(np.linalg.norm(
            shape.xi_to_xyz(sample_labels) - sample_pose[:, :3, 3], axis=-1).max()))
        assert np.all(max_label_error <= [1.e-6, 1.e-7, 1.e-7])
        assert max_roundtrip_m <= 1.e-6

        path = traces["station_ref_mm"]
        for case in report["cases"]:
            index = case["case"]
            scene = Scene(np.asarray(case["hidden_static_offset_mm"]), np.asarray(case["hidden_calibration_bias_mm"]),
                          traces[f"{index}_clear"], traces[f"{index}_noise_mm"])
            generated = sensor.scene(scene_rng, FRAME_COUNT)
            for field in ("offset", "bias", "clear", "noise"):
                np.testing.assert_array_equal(getattr(scene, field), getattr(generated, field))
            z_mm = traces[f"{index}_z_mm"] if motion else np.zeros(FRAME_COUNT)
            if motion:
                np.testing.assert_array_equal(z_mm, motion_tape(motion_rng, FRAME_COUNT, FRAME_DT_S, motion["kind"]))
            first_q = first_packet = None
            for arm, row in case["arms"].items():
                key = f"{index}_{arm}"
                q, poses, x_v = (traces[f"{key}_{suffix}"] for suffix in ("q", "fk_m", "x_v"))
                if first_q is None:
                    first_q = q[0]
                np.testing.assert_array_equal(q[0], first_q)
                assert len(q) == len(poses) == len(x_v)
                assert (len(q) - 1) % FRAME_STRIDE == 0
                if len(q) > 1:
                    assert admissible(Plan(0, x_v[:, 0], x_v[:, 1]))
                targets = shape.pose(path[:len(q)], x_v[:, 0])
                realized = np.stack([capability.fk(item) for item in q])
                np.testing.assert_allclose(poses, realized, atol=1.e-12, rtol=0.)
                checks = [capability.check_q(item, target) for item, target in zip(q, targets, strict=True)]
                assert all(item["valid"] for item in checks)
                max_fk_mm = max(max_fk_mm, 1000. * max(item["fk_position_error_m"] for item in checks))
                max_fk_deg = max(max_fk_deg, float(np.rad2deg(max(item["fk_rotation_error_rad"] for item in checks))))
                min_clearance = min(min_clearance, 1000. * min(item["min_self_collision_distance_m"] for item in checks))
                if len(q) > 1:
                    rate = np.abs(np.diff(q, axis=0)) / (SAMPLE_DT_S * capability.kin.v_max)
                    max_rate = max(max_rate, float(rate.max()))
                    assert rate.max() <= 1. + 1.e-10
                model = models.get(arm, sensor)
                belief = model.prior()
                frames = []
                for f, saved in enumerate(row["frames"]):
                    step = f * FRAME_STRIDE
                    pose = realized[step]
                    effective = scene.offset + [z_mm[f], 0.]
                    measured = project(shape, pose, effective)[0]
                    physical = project(shape, pose, effective - scene.bias)[0]
                    packet, role, true_u = sensor.observe_means(scene, f, measured, physical)
                    means = project(shape, pose, model.offsets_mm)
                    belief, assoc = model.update_projected(belief, packet, means)
                    if motion:
                        mean, interval = offset_stats(model, belief)
                        residual = model.residual_mean(belief) if arm in models else 0.
                        np.testing.assert_allclose(saved["offset_mean_mm"], mean, rtol=0., atol=1.e-10)
                        np.testing.assert_array_equal(saved["offset_interval90_mm"], interval)
                        np.testing.assert_allclose(saved["dynamic_lateral_mean_mm"], residual, rtol=0., atol=1.e-10)
                    if f == 0:
                        if first_packet is None:
                            first_packet = packet
                        np.testing.assert_array_equal(packet, first_packet)
                    np.testing.assert_allclose(np.asarray(saved["packet_mm"]).reshape(-1, 2), packet, rtol=0., atol=1.e-10)
                    np.testing.assert_array_equal(saved["role_truth"], role)
                    np.testing.assert_allclose(saved["artery_probability"], assoc, rtol=0., atol=1.e-10)
                    slot = int(assoc.argmax()) if len(assoc) and assoc.max() >= IDENTITY_MIN else -1
                    accepted = slot >= 0 and abs(packet[slot, 0]) <= CENTER_TOL_MM
                    wrong = accepted and role[slot] != 0
                    correct = accepted and not wrong and np.isfinite(true_u[0]) and abs(true_u[0]) <= CENTER_TOL_MM
                    if np.isfinite(true_u[0]):
                        np.testing.assert_allclose(saved["physical_lateral_mm"], true_u[0], rtol=0., atol=1.e-10)
                    else:
                        assert saved["physical_lateral_mm"] is None
                    for field, value in (("selected_slot", slot), ("accepted", accepted),
                                         ("wrong", wrong), ("correct", correct)):
                        assert saved[field] == value
                    frames.append(dict(station_ref_mm=float(path[step]), accepted=accepted, correct=correct, wrong=wrong))
                verified = score(frames, row["failure"], realized, path, [], 0, 0., 0.)
                for field in ("success", "wrong", "false_complete", "coverage", "tcp_mm", "turn_deg", "duration_s"):
                    np.testing.assert_allclose(row[field], verified[field], rtol=0., atol=1.e-10)
                row_count += 1
                sample_count += len(q)
                frame_count += len(frames)
    return {"run": run.name, "beta": metadata["beta"], "knee_deg": metadata["knee_deg"],
            "material_binding_sha256": metadata["material_binding"]["sha256"],
            "inputs_sha256": {path.name: sha(path) for path in (report_path, trace_path)},
            "max_label_error_rad_h_d": max_label_error.tolist(), "max_coordinate_roundtrip_m": max_roundtrip_m,
            "max_relative_rotation_matrix_element_error": max_relative_rotation_error,
            "paths": row_count, "joint_samples": sample_count, "acquisitions": frame_count,
            "max_position_tracking_mm": max_fk_mm, "max_rotation_tracking_deg": max_fk_deg,
            "max_joint_rate_fraction": max_rate, "minimum_robot_self_clearance_mm": min_clearance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--among-us", required=True, type=Path)
    parser.add_argument("--repair", required=True, type=Path)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import CapabilityEvaluator

    template = ScanTemplate(args.among_us, args.repair)
    capability = CapabilityEvaluator(device="cpu")
    records = []
    for run in args.runs:
        verified = verify(template, capability, run)
        records.append(verified)
        print(json.dumps(verified), flush=True)

    reference = args.runs[0]
    ref_report = json.loads((reference / "SUMMARY.json").read_text())
    with np.load(reference / "TRACES.npz", allow_pickle=False) as traces:
        reference_poses = traces["sample_material_probe_poses"].copy()
        reference_vessels = traces["nominal_vessels_m"].copy()
        reference_tapes = {name: traces[name].copy() for name in traces.files
                           if name.endswith(("_clear", "_noise_mm", "_z_mm"))}
    for run, row in zip(args.runs, records, strict=True):
        cohort = json.loads((run / "SUMMARY.json").read_text())
        with np.load(run / "TRACES.npz", allow_pickle=False) as traces:
            poses = traces["sample_material_probe_poses"]
            relative = Rotation.from_matrix(reference_poses[:, :3, :3]).inv() * Rotation.from_matrix(poses[:, :3, :3])
            row["max_material_probe_origin_change_vs_first_mm"] = float(
                1000. * np.linalg.norm(poses[:, :3, 3] - reference_poses[:, :3, 3], axis=-1).max())
            row["max_material_probe_rotation_change_vs_first_deg"] = float(np.rad2deg(relative.magnitude()).max())
            row["max_material_vessel_point_change_vs_first_mm"] = float(
                1000. * np.linalg.norm(traces["nominal_vessels_m"] - reference_vessels, axis=-1).max())
            common_cases = min(len(cohort["cases"]), len(ref_report["cases"]))
            for i, case in enumerate(cohort["cases"][:common_cases]):
                for k in ("hidden_static_offset_mm", "hidden_calibration_bias_mm"):
                    np.testing.assert_array_equal(case[k], ref_report["cases"][i][k])
                for suffix in ("clear", "noise_mm"):
                    key = f"{i}_{suffix}"
                    np.testing.assert_array_equal(traces[key], reference_tapes[key])
                if cohort.get("motion") == ref_report.get("motion") and cohort.get("motion"):
                    key = f"{i}_z_mm"
                    np.testing.assert_array_equal(traces[key], reference_tapes[key])
            row["matching_disturbance_tapes_vs_first"] = common_cases
    args.out.write_text(json.dumps({"runs": records}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
