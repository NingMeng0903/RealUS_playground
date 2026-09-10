#!/usr/bin/env python3
"""Reproduce the bounded design-v2 diagnosis; never read acceptance runs.

Reads frozen simulation sources, design seeds 0--9, and the already generated
recovery-event table. Imports only the frozen plant/features for static renderer
checks. It does not run a controller, change simulation code, or use hardware.
Writes one JSON report and an exact input fingerprint index beside that report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ("healthy", "left_gap", "right_gap", "both_edges", "curvature",
             "delayed_execution", "shadow")
VARIANTS = ("baseline", "full", "full_consistency")
SOURCES = (
    "peirastic/apps/contact_qp_experiment.py",
    "peirastic/contact_qp/plant.py", "peirastic/contact_qp/features.py",
    "peirastic/contact_qp/history.py", "peirastic/contact_qp/qp.py",
    "peirastic/scan_path.py", "peirastic/realman8dof/session.py",
    "peirastic/realman8dof/force/contact_nominal.py",
    "peirastic/configs/controller.yaml",
    "rm75_control/rm75_control/control/joint_admittance_8dof/loop.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "design_v2" / "diagnosis.json")
    args = parser.parse_args()
    source = ROOT / "MD/contact_qp/design_v2"
    inputs = {}

    def record(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        inputs[str(path.relative_to(ROOT))] = dict(sha256=sha(raw), bytes=len(raw))
        return raw

    manifest = json.loads(record(source / "manifest.json"))
    metrics = json.loads(record(source / "metrics.json"))
    assert len(metrics) == 840
    assert {row["seed"] for row in metrics} == set(range(10)), "design-only guard"
    assert len({(r["scenario"], r["seed"], r["variant"]) for r in metrics}) == 840
    recovery_path = ROOT / "MD/contact_qp/analysis_artifacts/design_v2/recovery_events.csv"
    recovery = list(csv.DictReader(record(recovery_path).decode().splitlines()))
    assert {int(row["seed"]) for row in recovery} <= set(range(10))
    snapshot = source / "sources.zip"
    record(snapshot)
    with zipfile.ZipFile(snapshot) as archive:
        frozen_track = yaml.safe_load(archive.read("peirastic/configs/controller.yaml"))["cartesian_track"]
        assert frozen_track["k_task_lin"] == 12. and frozen_track["fb_lpf_tau_s"] == 0.
        source_hashes = {}
        for name in SOURCES:
            frozen = archive.read(name)
            current = ROOT / name
            local = current.read_bytes() if current.exists() else None
            source_hashes[name] = dict(frozen_sha256=sha(frozen),
                                      current_sha256=sha(local) if local else None,
                                      current_matches_snapshot=local == frozen)

    cache = {}

    def trace(scenario, seed, variant):
        assert seed in range(10), "design-only guard"
        key = scenario, seed, variant
        if key not in cache:
            path = source / scenario / str(seed) / (variant + ".npz")
            record(path)
            with np.load(path) as npz:
                cache[key] = {name: npz[name].copy() for name in npz.files}
        return cache[key]

    pc, fc = manifest["plant"], manifest["features"]
    dt = pc["dt_s"]
    assert dt == .005 and pc["warmup_s"] == 1.5
    assert pc["image_period_s"] == .04
    delivery_delay = pc["image_delay_s"] + pc["feature_processing_delay_s"]
    delivery_ticks = int(np.ceil(delivery_delay / dt - 1e-12))
    image_stride = round(pc["image_period_s"] / dt)
    active_start_tick = round(pc["warmup_s"] / dt)
    assert delivery_ticks == 32 and image_stride == 8 and active_start_tick == 300
    c_min = manifest["qp"]["c_min"]
    assert c_min == .5
    delay_rows = []
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            counts = np.zeros(8, dtype=np.int64)
            ages = []
            mismatches = 0
            bad_any_fraction = []
            for seed in range(10):
                data = trace(scenario, seed, variant)
                sc = data["phase"] == "scan"
                time = data["time_s"][sc]
                ticks = np.rint(time / dt).astype(int) + active_start_tick
                assert np.array_equal(ticks, np.arange(active_start_tick, active_start_tick+len(time)))
                # Inferred from exact source scheduling. No per-frame timestamp
                # is claimed to be present in the saved NPZ.
                capture_ticks = ((ticks-delivery_ticks)//image_stride)*image_stride
                capture_index = capture_ticks-active_start_tick
                age = (ticks-capture_ticks)*dt
                ages.extend((float(age.min()), float(age.max())))
                q = np.stack((data["quality_left"][sc], data["quality_right"][sc]), axis=1)
                acoustic = np.stack((data["acoustic_left"][sc], data["acoustic_right"][sc]), axis=1)
                request, good = q < c_min, acoustic >= .9
                bad_any_fraction.append(float(np.any(request, axis=1).mean()))
                have = capture_index >= 0
                old_good = acoustic[capture_index[have]] >= .9
                rq, current = request[have], good[have]
                counts += (request.sum(), (request & good).sum(), rq.sum(),
                           (rq & ~old_good).sum(), (rq & ~old_good & current).sum(),
                           (rq & old_good).sum(), (rq & old_good & current).sum(), request.size)
                changed = np.r_[False, np.any(np.diff(q, axis=0) != 0, axis=1)]
                expected = ticks % image_stride == 0
                expected[0] = False  # no preceding observation in this trace
                mismatches += int(np.sum(changed != expected))
            events = [row for row in recovery if row["scenario"] == scenario and row["variant"] == variant]
            durations = np.array([float(row["duration_s"]) for row in events if row["duration_s"]])
            names = ("request_window_ticks", "request_current_good_window_ticks",
                     "request_with_in_scan_capture_window_ticks", "request_capture_bad_window_ticks",
                     "request_capture_bad_current_good_window_ticks", "request_capture_good_window_ticks",
                     "request_capture_good_current_good_window_ticks", "all_window_ticks")
            row = dict(scenario=scenario, variant=variant,
                       counts={key: int(value) for key, value in zip(names, counts)},
                       current_good_fraction_given_request=float(counts[1]/counts[0]) if counts[0] else None,
                       current_good_fraction_given_request_and_capture_bad=float(counts[4]/counts[3]) if counts[3] else None,
                       any_edge_below_c_min_tick_fraction_mean=float(np.mean(bad_any_fraction)),
                       inferred_image_age_min_s=min(ages), inferred_image_age_max_s=max(ages),
                       observation_change_schedule_mismatches=mismatches,
                       recovery_event_count=len(events), confirmed_recovery_count=len(durations),
                       unconfirmed_recovery_count=len(events)-len(durations),
                       confirmed_recovery_duration_min_median_max_s=
                       [float(durations.min()), float(np.median(durations)), float(durations.max())] if len(durations) else None,
                       confirmed_recovery_duration_below_delivery_delay_count=int(np.sum(durations < delivery_delay)),
                       confirmed_recovery_median_over_delivery_delay=float(np.median(durations)/delivery_delay) if len(durations) else None)
            delay_rows.append(row)

    failures = []
    for row in metrics:
        if row["variant"] != "full" or row["status"] != "aborted":
            continue
        data = trace(row["scenario"], row["seed"], "full")
        i = np.flatnonzero(data["phase"] == "scan")[-1]
        assert data["reference_s"][i] == 3.
        # Frozen config gain_y=12, feedback LPF=0; y-axis rotation leaves Y
        # unchanged. Positive forward motion implies next actual path >= last.
        # With reference feedforward now zero and alpha in [0,1], this is a
        # conservative H-admitted upper bound, not a logged rejected command.
        y_upper = max(0., frozen_track["k_task_lin"]*(pc["path_length_m"]-float(data["path_m"][i])))
        y_lower = float(data["sent_twist"][i, 1])-manifest["qp"]["max_acceleration"][1]*dt
        assert np.all(data["measured_twist"][:, 1] >= -1e-12)
        assert y_lower > y_upper
        failures.append(dict(scenario=row["scenario"], seed=row["seed"], reason=row["reason"],
                             last_scan_time_s=float(data["time_s"][i]),
                             last_scan_path_m=float(data["path_m"][i]),
                             last_scan_reference_s=float(data["reference_s"][i]),
                             last_sent_y_m_s=float(data["sent_twist"][i, 1]),
                             next_h_y_upper_bound_m_s=y_upper,
                             next_mechanical_y_lower_bound_m_s=y_lower,
                             bounds_disjoint=True))
    assert len(failures) == 8

    shadow = {}
    for variant in ("baseline", "full", "matched_scan_speed", "full_consistency"):
        rows = [row for row in metrics if row["scenario"] == "shadow" and row["variant"] == variant]
        assert len(rows) == 10
        stats = {key: float(np.mean([row[key] for row in rows])) for key in
                 ("rmse", "absolute_mean_bias", "peak_absolute_error", "elapsed_s", "angular_travel_rad")}
        for seed in range(10):
            trace("shadow", seed, variant)
        data = trace("shadow", 0, variant)
        sc = data["phase"] == "scan"
        f = data["force_n"][sc]
        delta = (data["qp_twist"]-data["nominal_twist"])[sc, 2]
        in_band = abs(f-4) <= .1+1e-12
        stats["seed_0_scan_diagnostic"] = dict(
            mean_force_n=float(f.mean()), in_force_sign_band_fraction=float(in_band.mean()),
            force_above_4p1_fraction=float(np.mean(f > 4.1)),
            mean_additional_vn_in_band_m_s=float(delta[in_band].mean()) if in_band.any() else None,
            mean_additional_vn_outside_band_m_s=float(delta[~in_band].mean()) if (~in_band).any() else None,
            integral_positive_additional_vn_m=float(np.maximum(delta, 0).sum()*dt),
            gamma_left_min=float(data["gamma_left"][sc].min()),
            gamma_left_max=float(data["gamma_left"][sc].max()))
        shadow[variant] = stats

    # Import directly from the archived source bundle, not mutable worktree code.
    sys.path.insert(0, str(snapshot)+"/peirastic")
    plant_module = importlib.import_module("contact_qp.plant")
    feature_module = importlib.import_module("contact_qp.features")
    assert str(snapshot) in plant_module.__file__
    assert str(snapshot) in feature_module.__file__
    config = plant_module.PlantConfig(**pc)
    features = feature_module.FeatureConfig(**fc)
    renderer = {}
    for scenario in ("healthy", "shadow"):
        plant = plant_module.FiniteAreaPlant(scenario, 0, config)
        plant.z = .005
        plant._update_contact()
        original = [plant.render(i) for i in range(48)]
        plant.z = .0051
        plant._update_contact()
        deeper = [plant.render(i) for i in range(48)]
        q = np.array([feature_module.window_quality(feature_module.random_walk_confidence(frame, features), features)
                      for frame in original])
        renderer[scenario] = dict(seed=0, fixed_state_depth_m=.005, deeper_depth_m=.0051,
                                  identical_frames_after_100um_press=int(sum(np.array_equal(a, b) for a, b in zip(original, deeper))),
                                  frames_compared=48,
                                  quality_min_median_max=np.quantile(q, (0., .5, 1.), axis=0).tolist(),
                                  quality_peak_to_peak_by_window=np.ptp(q, axis=0).tolist(),
                                  maximum_absolute_adjacent_frame_quality_change_by_window=abs(np.diff(q, axis=0)).max(axis=0).tolist(),
                                  rendered_frames_sha256=sha(np.stack(original).tobytes()),
                                  window_order=["left", "center", "right"])

    # Existing independently computed static feasibility is evidence, not rerun.
    static_report = ROOT / "MD/contact_qp/ACCEPTANCE_METHOD.md"
    record(static_report)
    static_result = ROOT / "MD/contact_qp/acceptance_artifacts/static_feasibility_design.json"
    record(static_result)
    result = dict(schema_version=1, scope="DESIGN_ONLY_NOT_ACCEPTANCE", seeds=list(range(10)),
                  analysis_sha256=sha(Path(__file__).read_bytes()),
                  evidence_kinds=dict(npz="recorded simulation values",
                      image_age="inferred from frozen scheduling; not a saved NPZ field",
                      endpoint_bounds="analytic conservative bound using recorded last state and frozen reference/config",
                      renderer="static forward calculations from frozen source; no controller execution",
                      static_feasibility="previous independent design-only grid audit, not recomputed here"),
                  frozen_source_hashes=source_hashes,
                  delay_method=dict(dt_s=dt, image_period_s=pc["image_period_s"],
                                    declared_delivery_delay_s=delivery_delay, delivered_after_ticks=delivery_ticks,
                                    capture_tick_formula="8 * floor((300 + scan_index - 32) / 8)",
                                    request="quality < 0.5 for the same individual left/right window",
                                    latent_good="same acoustic window fraction >= 0.90",
                                    captured_truth_rule="discard inferred capture before scan start; otherwise index exact 5ms trace",
                                    uncertainty="Age inferred from deterministic simulator, not hardware timestamp validation"),
                  delay_and_recovery=delay_rows, full_endpoint_failures=failures,
                  shadow=shadow, static_renderer=renderer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    index_path = args.output.with_name("diagnosis_inputs.json")
    index_raw = (json.dumps(dict(sorted(inputs.items())), indent=2, ensure_ascii=False)+"\n").encode()
    index_path.write_bytes(index_raw)
    result["input_index"] = dict(path=str(index_path.relative_to(ROOT)), sha256=sha(index_raw), files=len(inputs),
                                 npz_files=sum(name.endswith(".npz") for name in inputs))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output.relative_to(ROOT)), sha256=sha(args.output.read_bytes()),
                          input_files=len(inputs), full_endpoint_failures=len(failures),
                          schedule_mismatches=sum(row["observation_change_schedule_mismatches"] for row in delay_rows))))


if __name__ == "__main__":
    main()
