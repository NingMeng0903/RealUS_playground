"""Independent SENSOR_V2 feature generation and train/calibration fitting.

The generator is a training/evaluation utility only.  Policy code receives the
returned ``PublicObservation`` and never receives ``V2Truth`` or a scenario.
Splits are independent: train 721501/512, calibration 721502/128, test
721503/128, with observation-noise family 721504.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from acoustics import (PRIOR_SD, ProbeAction, PublicObservation, fixture_actions,
                       make_scenario, sensor_sd)
from belief_model import CompleteOption, ModelParams, new_belief

TRAIN_SEED, CAL_SEED, TEST_SEED, NOISE_SEED = 721501, 721502, 721503, 721504
TRAIN_COUNT, CAL_COUNT, TEST_COUNT = 512, 128, 128
CALIBRATION_VERSION = "v3_prefix_subject_weighted_dwell_white_hmm_single_transition_joint_gh"


def _seed(*parts):
    raw = ":".join(map(str, (NOISE_SEED, *parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _uniform(seed, *parts):
    return np.random.default_rng(_seed(seed, *parts))


def sensor_v2_action(action: ProbeAction, dwell_s: float) -> ProbeAction:
    """Return the same physical action with one allowed diagnostic dwell."""
    if float(dwell_s) not in (0.15, 0.30, 0.60):
        raise ValueError("SENSOR_V2 probe dwell must be .15, .30, or .60 s")
    return ProbeAction(action.window, action.xy_m, action.yaw_rad, float(dwell_s), action.kind)


@dataclass(frozen=True)
class V2Truth:
    present_before: bool
    present_after: bool
    target_association: bool
    clutter: bool
    target_quality: float
    true_center_offset_m: float
    true_axis_angle_rad: float
    geometry_gate: bool
    quality_gate: bool
    public_stop: bool
    correct_complete: bool


def render_sensor_v2(scenario, action: ProbeAction, *, event_id: str,
                     seed: int = NOISE_SEED, present: bool = True,
                     elapsed_s: float | None = None):
    """Render one V2 packet; hidden truth is returned only to the evaluator."""
    rng = _uniform(seed, scenario.seed, scenario.address, event_id, action.window,
                   action.xy_m, action.yaw_rad, action.duration_s)
    present_before = bool(present)
    p_loss = .03 if present_before else 0.
    p_recover = 0.70 if not present_before else 0.
    present_after = bool(rng.random() >= p_loss) if present_before else bool(rng.random() < p_recover)
    # SENSOR_V2 publishes signed errors.  The old helper returns absolute
    # errors for its legacy evaluator, so compute the signed means here and
    # apply abs only to support/gate decisions.
    x, y = map(float, action.xy_m)
    target_y = scenario.true_curve_y(x)
    beam_y = y - scenario.target_depth_m * math.tan(action.tilt_rad)
    center_error = float(beam_y - target_y)
    vessel_axis = math.atan(scenario.curve_slope + scenario.curve_curvature * x)
    axis_error = float((action.tilt_rad - vessel_axis + math.pi) % (2. * math.pi) - math.pi)
    quality_geo = float(.64 - 11. * min(abs(center_error), .022) - .38 * abs(axis_error))
    target_quality = float(quality_geo + action.as_feature() @ scenario.response + sensor_sd(action) * rng.normal())
    support = bool(abs(center_error) <= .022)
    target_detected = bool(present_after and support and rng.random() < .80)
    clutter = bool((not target_detected) and rng.random() < .10)
    if target_detected:
        center = float(center_error + .0005 * math.sqrt(.15 / action.duration_s) * rng.normal())
        axis = float(axis_error + .03 * math.sqrt(.15 / action.duration_s) * rng.normal())
        quality = target_quality
        confidence = float(1. / (1. + math.exp(-np.clip((quality - .34) / .14, -60., 60.))))
        candidate = True
    elif clutter:
        center = float(rng.uniform(-.022, .022))
        axis = float(rng.uniform(-.35, .35))
        quality = float(rng.normal(.60, .12))
        confidence = float(1. / (1. + math.exp(-np.clip((quality - .34) / .14, -60., 60.))))
        candidate = True
    else:
        center = axis = quality = float("nan")
        confidence = 0.0
        candidate = False
    elapsed = float(action.duration_s if elapsed_s is None else elapsed_s)
    packet = PublicObservation(event_id, scenario.address, action.window, elapsed, "checked",
                               candidate, confidence, quality, center, axis,
                               "candidate" if candidate else "missing")
    public_stop = bool(candidate and quality >= .58 and abs(center) <= .0045 and abs(axis) <= .22)
    geometry_gate = bool(abs(center_error) <= .0045 and abs(axis_error) <= .22)
    quality_gate = bool(target_quality >= .58)
    correct_complete = bool(target_detected and geometry_gate and quality_gate and public_stop)
    return packet, V2Truth(present_before, present_after, target_detected, clutter,
                           target_quality, center_error, axis_error, geometry_gate,
                           quality_gate, public_stop, correct_complete)


def action_grid():
    base = fixture_actions()
    names = ("probe_left", "probe_right", "probe_center_left", "probe_center_right")
    return tuple(sensor_v2_action(base[name], dwell) for name in names for dwell in (.15, .30, .60))


def feature_rows(seed: int, count: int, *, repeats: int = 1):
    """Generate public packet rows plus evaluator-only labels for one split."""
    rows = []
    actions = action_grid()
    for index in range(int(count)):
        scenario = make_scenario(_seed(seed, "scenario", index), f"left_leg:train:{index:04d}")
        present = True
        for visit in range(int(repeats)):
            for action_index, action in enumerate(actions):
                event_id = f"{index}:{visit}:{action_index}"
                packet, truth = render_sensor_v2(scenario, action, event_id=event_id,
                                                 present=present, seed=NOISE_SEED)
                present = truth.present_after
                rows.append({"split_seed": int(seed), "object_id": index,
                             "visit": visit, "action_index": action_index,
                             "action": {"window": action.window, "xy_m": action.xy_m,
                                        "yaw_rad": action.yaw_rad,
                                        "duration_s": action.duration_s, "kind": action.kind},
                             "packet": packet, "truth": truth})
    return rows


PREFIX_LENGTHS = (0, 1, 2, 4, 6)


def _action_dict(action):
    return {"window": action.window, "xy_m": action.xy_m,
            "yaw_rad": action.yaw_rad, "duration_s": action.duration_s,
            "kind": action.kind}


def _packet_dict(packet):
    """Stable public-only serialization for hashes and calibration audit."""
    return {name: getattr(packet, name) for name in (
        "event_id", "address", "window", "elapsed_s", "exit_state", "detected",
        "confidence", "quality", "center_offset_m", "axis_angle_rad",
        "candidate_type")}


def _input_hash(row, seed):
    """Hash the public prefix/terminal input used for one raw prediction."""
    payload = {"seed": int(seed), "object_id": int(row["object_id"]),
               "prefix_length": int(row["prefix_length"]),
               "prefix": [{"action": item["action"],
                           "packet": _packet_dict(item["packet"])}
                          for item in row.get("prefix", ())],
               "terminal": row.get("terminal", row.get("action")),
               "option_name": row.get("option_name", "left")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_hashes():
    """Hashes recorded with artifacts so failed-contract outputs are traceable."""
    paths = {name: Path(__file__).with_name(name) for name in (
        "train_calibrate.py", "belief_model.py", "acoustics.py", "DESIGN_PROTOCOL.txt")}
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items() if path.exists()}


def terminal_actions():
    """Public fixed local terminal poses; native validity is supplied outside."""
    base = fixture_actions()
    return {
        "left": ProbeAction(base["terminal_left"].window, base["terminal_left"].xy_m,
                             base["terminal_left"].yaw_rad, 1.0, "terminal"),
        "right": ProbeAction(base["terminal_right"].window, base["terminal_right"].xy_m,
                              base["terminal_right"].yaw_rad, 1.0, "terminal"),
    }


def _simulate_complete_option(scenario, action, present, event_prefix):
    correct = False
    ever_qualified = False
    false_stop = False
    attempts = 0
    for attempt in range(2):
        packet, truth = render_sensor_v2(
            scenario, action, event_id=f"{event_prefix}:terminal:{attempt}",
            present=present, seed=NOISE_SEED)
        present = truth.present_after
        attempts += 1
        ever_qualified |= bool(truth.target_association and truth.geometry_gate and truth.quality_gate)
        if truth.public_stop:
            correct = bool(truth.correct_complete)
            false_stop = not correct
            break
    return {"label": bool(correct), "ever_qualified": bool(ever_qualified),
            "false_stop": bool(false_stop), "attempts": attempts}


def calibration_rows(seed: int, count: int):
    """Generate object-weighted prefix/option calibration records.

    Prefixes are common fixed public actions of lengths 0,1,2,4,6.  Each
    prefix is replayed into a fresh public belief, then both terminal options
    are predicted and evaluated with the same two-attempt public verifier.
    The actual native route validity is deliberately not inferred here; the
    pose manifest records that these are canonical public local poses and the
    native adapter supplies checked validity in the integrated run.
    """
    diagnostics = action_grid()
    terminals = terminal_actions()
    rows = []
    for index in range(int(count)):
        scenario = make_scenario(_seed(seed, "calibration", index), f"left_leg:cal:{index:04d}")
        present = True
        prefix_packets = []
        prefix_states = [present]
        for step, action in enumerate(diagnostics[:max(PREFIX_LENGTHS)]):
            packet, truth = render_sensor_v2(
                scenario, action, event_id=f"cal:{index}:prefix:{step}",
                present=present, seed=NOISE_SEED)
            present = truth.present_after
            prefix_packets.append({"action": _action_dict(action), "packet": packet})
            prefix_states.append(present)
        for prefix_length in PREFIX_LENGTHS:
            prefix = prefix_packets[:prefix_length]
            for option_name, terminal in terminals.items():
                outcome = _simulate_complete_option(
                    scenario, terminal, prefix_states[prefix_length],
                    f"cal:{index}:prefix:{prefix_length}:{option_name}")
                rows.append({
                    "object_id": index,
                    "seed": int(seed),
                    "prefix_length": prefix_length,
                    "option_name": option_name,
                    "prefix": prefix,
                    "terminal": _action_dict(terminal),
                    "label": outcome["label"],
                    "ever_qualified": outcome["ever_qualified"],
                    "false_stop": outcome["false_stop"],
                    "attempts": outcome["attempts"],
                    "record_weight": 1. / (len(PREFIX_LENGTHS) * len(terminals)),
                })
    return rows


def complete_option_rows(seed: int, count: int):
    """Backward-compatible no-prefix view; main calibration uses all prefixes."""
    rows = calibration_rows(seed, count)
    return [row for row in rows if row["prefix_length"] == 0 and row["option_name"] == "left"]


def _fit_platt(raw, outcomes, weights=None):
    raw = np.clip(np.asarray(raw, float), 1e-5, 1. - 1e-5)
    outcomes = np.asarray(outcomes, float)
    weights = np.ones_like(raw) if weights is None else np.asarray(weights, float)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    z = np.log(raw / (1. - raw))
    def objective(x):
        p = 1. / (1. + np.exp(-np.clip(x[0] * z + x[1], -60., 60.)))
        loss = outcomes * np.log(np.clip(p, 1e-12, 1.)) + (1. - outcomes) * np.log(np.clip(1. - p, 1e-12, 1.))
        return float(-np.sum(weights * loss))
    result = minimize(objective, np.array((1., 0.)), bounds=((0., 20.), (-20., 20.)), method="L-BFGS-B")
    return float(max(result.x[0], 0.)), float(result.x[1])


def fit_model(train_rows):
    """Fit response/noise/detector parameters from labelled train rows.

    The true ``scenario.response`` is never read.  Training geometry and
    target/clutter labels are permitted by the frozen protocol.
    """
    response_groups = {}
    target_detected = target_opportunities = clutter_count = clutter_opportunities = 0
    loss_transitions = recover_transitions = 0
    loss_opportunities = recover_opportunities = 0
    center_residuals, axis_residuals, clutter_quality = [], [], []
    # Rows retain only public packet and truth; scenario values are not stored.
    # Geometry is reconstructed from the public training action and the declared
    # public prior by the generator, not by reading response coefficients.
    for row in train_rows:
        packet, truth = row["packet"], row["truth"]
        action = ProbeAction(**row["action"])
        if truth.present_before:
            loss_opportunities += 1
            loss_transitions += int(not truth.present_after)
        else:
            recover_opportunities += 1
            recover_transitions += int(truth.present_after)
        if truth.present_after and abs(truth.true_center_offset_m) <= .022:
            target_opportunities += 1
            target_detected += int(truth.target_association)
        if not truth.target_association:
            clutter_opportunities += 1
            clutter_count += int(truth.clutter)
        if truth.target_association and packet.detected and np.isfinite(packet.quality):
            # The public regression target is packet quality minus declared
            # nominal geometry; no hidden response coefficient is consumed.
            nominal = .64 - 11. * min(abs(truth.true_center_offset_m), .022) - .38 * abs(truth.true_axis_angle_rad)
            noise_scale = math.sqrt(.15 / action.duration_s)
            response_groups.setdefault(row["object_id"], []).append(
                (action.as_feature(), packet.quality - nominal, noise_scale))
            if np.isfinite(packet.center_offset_m):
                center_residuals.append(((packet.center_offset_m - truth.true_center_offset_m), noise_scale))
            if np.isfinite(packet.axis_angle_rad):
                axis_residuals.append(((packet.axis_angle_rad - truth.true_axis_angle_rad), noise_scale))
        if truth.clutter and np.isfinite(packet.quality): clutter_quality.append(packet.quality)
    if response_groups:
        all_design = np.asarray([row[0] for rows in response_groups.values() for row in rows])
        all_values = np.asarray([row[1] for rows in response_groups.values() for row in rows])
        all_scales = np.asarray([row[2] for rows in response_groups.values() for row in rows])
        weighted_design = all_design / all_scales[:, None]
        weighted_values = all_values / all_scales
        mean, *_ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
        object_fits, residual_squares, information_matrices = [], [], []
        for rows in response_groups.values():
            if len(rows) < 2:
                continue
            design = np.asarray([row[0] for row in rows]); values = np.asarray([row[1] for row in rows])
            scales = np.asarray([row[2] for row in rows])
            design_w = design / scales[:, None]; values_w = values / scales
            fitted, *_ = np.linalg.lstsq(design_w, values_w, rcond=None)
            object_fits.append(fitted)
            residual_squares.extend(((values - design @ fitted) / scales) ** 2)
            information_matrices.append(np.linalg.pinv(design_w.T @ design_w))
        # Whitened residuals estimate the .15 s baseline noise.  The
        # measurement covariance correction uses the same weighted design.
        dof = max(len(residual_squares) - 2 * len(object_fits), 1)
        sensor_sd = float(np.sqrt(np.sum(residual_squares) / dof)) if residual_squares else .035
        if len(object_fits) >= 2:
            covariance = np.cov(np.asarray(object_fits).T, bias=True)
            measurement = np.mean(np.asarray(information_matrices), axis=0) * sensor_sd ** 2
            covariance = covariance - measurement
            covariance = np.diag(np.maximum(np.diag(covariance), 1e-8))
        else:
            covariance = np.diag(PRIOR_SD ** 2)
    else:
        mean, covariance, sensor_sd = np.zeros(2), np.diag(PRIOR_SD ** 2), .035
    params = ModelParams(response_mean=np.asarray(mean), response_covariance=np.asarray(covariance),
        sensor_sd_at_015=max(sensor_sd, .005),
                         target_detect_probability=target_detected / max(target_opportunities, 1),
                         clutter_probability=clutter_count / max(clutter_opportunities, 1),
                         present_to_lost=loss_transitions / max(loss_opportunities, 1),
                         lost_to_present=recover_transitions / max(recover_opportunities, 1),
                         center_noise_sd_at_015=max(float(np.sqrt(np.mean([(value / scale) ** 2 for value, scale in center_residuals]))) if center_residuals else .0005, 1e-5),
                         axis_noise_sd_at_015=max(float(np.sqrt(np.mean([(value / scale) ** 2 for value, scale in axis_residuals]))) if axis_residuals else .03, 1e-4),
                         clutter_quality_mean=float(np.mean(clutter_quality)) if clutter_quality else .60,
                         clutter_quality_sd=max(float(np.std(clutter_quality)) if clutter_quality else .12, .01))
    return params


def calibrate_option(model, calibration_rows, *, option_factory=None):
    """Fit Platt on object-weighted prefix-conditioned complete options."""
    # Replay each subject's common prefix incrementally once.  Replaying every
    # prefix from zero would multiply the expensive bounded-mixture updates by
    # 1+2+4+6 and would not change the public posterior.
    grouped = {}
    for row in calibration_rows:
        grouped.setdefault(row["object_id"], []).append(row)
    raw_by_key = {}
    for object_id, subject_rows in grouped.items():
        belief = new_belief(model)
        previous_length = 0
        for prefix_length in sorted({row["prefix_length"] for row in subject_rows}):
            representative = next(row for row in subject_rows if row["prefix_length"] == prefix_length)
            prefix = representative.get("prefix", ())
            for prefix_row in prefix[previous_length:prefix_length]:
                prefix_action = ProbeAction(**prefix_row["action"])
                belief.update(prefix_row["packet"].event_id, prefix_action, prefix_row["packet"])
            for row in (row for row in subject_rows if row["prefix_length"] == prefix_length):
                action = ProbeAction(**row["terminal"] if "terminal" in row else row["action"])
                option = option_factory(action) if option_factory else CompleteOption((action, action), max_retries=2, gh_order=33)
                raw_by_key[(object_id, row["prefix_length"], row.get("option_name", "left"))] = belief.predict_complete_option(option).raw_complete_p
            previous_length = prefix_length
    raw, labels, weights = [], [], []
    for row in calibration_rows:
        key = (row["object_id"], row["prefix_length"], row.get("option_name", "left"))
        raw.append(raw_by_key[key])
        labels.append(float(row["label"] if "label" in row else row["truth"].correct_complete))
        weights.append(float(row.get("record_weight", 1.)))
    a, b = _fit_platt(raw, labels, weights)
    by_prefix = {}
    for row, prediction in zip(calibration_rows, raw):
        key = f"p{row.get('prefix_length', 0)}_{row.get('option_name', 'left')}"
        by_prefix.setdefault(key, {"n": 0, "raw": [], "labels": []})
        by_prefix[key]["n"] += 1; by_prefix[key]["raw"].append(prediction); by_prefix[key]["labels"].append(float(row["label"]))
    for value in by_prefix.values():
        value["raw_mean"] = float(np.mean(value.pop("raw")))
        value["label_rate"] = float(np.mean(value.pop("labels")))
    seed_values = {int(row.get("seed", CAL_SEED)) for row in calibration_rows}
    if len(seed_values) != 1:
        raise ValueError("calibration rows must have one reproducible calibration seed")
    calibration_seed = next(iter(seed_values))
    records = []
    for row, prediction, label, weight in zip(calibration_rows, raw, labels, weights):
        terminal = row.get("terminal", row.get("action", {}))
        records.append({"object_id": int(row["object_id"]),
                        "prefix_length": int(row.get("prefix_length", 0)),
                        "option_name": row.get("option_name", "left"),
                        "window": terminal.get("window"),
                        "raw_p": float(prediction), "label": int(bool(label)),
                        "subject_weight": float(weight),
                        "input_sha256": _input_hash(row, calibration_seed),
                        "seed": calibration_seed})
    report = {"calibration_version": CALIBRATION_VERSION, "platt_a": a, "platt_b": b,
        "n_subjects": len({row["object_id"] for row in calibration_rows}),
        "n_records": len(calibration_rows),
        "subject_weighting": {"records_per_subject": len(PREFIX_LENGTHS) * 2,
                              "per_record_weight": 1. / (len(PREFIX_LENGTHS) * 2),
                              "subject_total_weight": 1.0},
        "raw_mean": float(np.average(raw, weights=weights)),
        "label_rate": float(np.average(labels, weights=weights)),
        "by_prefix_option": by_prefix,
        "pose_manifest": {"source": "fixture_actions canonical local poses",
                          "native_replay": False,
                          "terminal_dwell_s": 1.0,
                          "native_validity": "supplied by integrated native adapter"},
        "calibration_seed": calibration_seed,
        "records_file": "CALIBRATION_RECORDS.jsonl",
        "records": records,
        "source_hashes": source_hashes()}
    return ModelParams(**{**model.to_dict(), "platt_a": a, "platt_b": b}), report


def split_manifest():
    return {"train": {"seed": TRAIN_SEED, "count": 512},
            "calibration": {"seed": CAL_SEED, "count": 128},
            "test": {"seed": TEST_SEED, "count": 128}, "noise_seed": NOISE_SEED,
            "calibration_version": CALIBRATION_VERSION}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=512)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    train = feature_rows(TRAIN_SEED, args.train_count, repeats=1)
    model = fit_model(train)
    calibration = calibration_rows(CAL_SEED, CAL_COUNT)
    model, calibration_report = calibrate_option(model, calibration,
        option_factory=lambda action: CompleteOption((action, action), max_retries=2, gh_order=33))
    (args.output / "MODEL.json").write_text(json.dumps(model.to_dict(), indent=2) + "\n")
    (args.output / "CALIBRATION.json").write_text(json.dumps(calibration_report, indent=2) + "\n")
    (args.output / "SPLIT_MANIFEST.json").write_text(json.dumps(split_manifest(), indent=2) + "\n")
    with (args.output / "CALIBRATION_RECORDS.jsonl").open("w") as stream:
        for record in calibration_report["records"]:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"model": model.to_dict(), "calibration": calibration_report, "splits": split_manifest()}, indent=2))


if __name__ == "__main__":
    main()
