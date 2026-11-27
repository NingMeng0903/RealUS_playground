"""Step 3 reader benchmarks. Run as ``python -m reader.eval_reader``.

Readers see only the public scan interface. Truth is joined outside inference for
training records, paired metrics and segment audits. The frozen Blender evaluation
has a separate, authorized data-loading path and never fits anything.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np
from scipy.stats import binomtest, wilcoxon

from sim.anatomy_assets import (
    AnatomySource, NYSORA_STATIONS, OPEN3DMODEL_RIGHT, SPLIT_PROTOCOLS, TrainingAnatomyBank, VesselRole,
)
from sim.metric_graph_gp import ORDINARY_MATERIAL, STRESS_MATERIAL

from .belief_ipe import ROLE_COUNT
from .baselines import make_readers
from .dynamic_factor_graph import DynamicFactorGraphReader
from .split_prior import DynamicScanSimulator, build_split_prior
from .station_classifier import (
    STATION_IDS,
    fit_segment_certifier,
    fit_station_scorer,
    load_station_checkpoint,
    make_segment_record,
    make_training_record,
    uncalibrated_segment_certifier,
)

FULL = "Full_AGPS_Reader"
METHODS = ("B0_Rigid27", "B1_NoTopology_AbsXYZ", "B2_NoPullback", "B3_NoDiscount",
           "B4_NoBeliefIPE", FULL)
BLENDER = "FULL_TO_BLENDER"
HOLDOUT_ENV = "AGPS_EVAL_HOLDOUT_ALLOWED"
CHECKPOINT_SCHEMA = 6
ALPHA = .20
TOLERANCE_MM = 8.
OFFSETS_MM = (0., 10., 20., 30., 40.)
_N_MASKS = 1 << ROLE_COUNT
_MASK_SIZES = np.array([mask.bit_count() for mask in range(_N_MASKS)], dtype=np.intp)
SCORING_PROTOCOL = "anonymous-exact-contributor-mask-v1"
TRACKING_PROTOCOL = "fixed-rendered-frame-role-opportunities-v1"
DIRECT_STATIONS = (STATION_IDS[0], STATION_IDS[1], STATION_IDS[4])
INFERRED_STATIONS = STATION_IDS[2:4]
PROTOCOL_STATIONS = {
    "LOAO_OPEN3D": STATION_IDS,
    "LOAO_BODYPARTS": DIRECT_STATIONS,
    "FAMILY_HELD_OUT_ATLASES": STATION_IDS[:4],
    BLENDER: DIRECT_STATIONS,
}
COHORTS = {name: i for i, name in enumerate((
    "fit", "temperature", "conformal_calibration", "matched_train_risk", "held_risk",
    "offset", "dynamic", "retroactive", "station", "inferred_station",
    "role_fit", "role_temperature",
), start=1)}


@dataclass(frozen=True)
class Tape:
    episode_id: str
    source_id: str
    seed: int
    station_id: str | None = None


def _tape(seed, cohort, index, source=None, station=None, population=None):
    root = np.random.SeedSequence([seed, COHORTS[cohort], index])
    sampling, noise = root.spawn(2)
    if population is not None:
        rng = np.random.default_rng(sampling)
        source, stations = population[rng.integers(len(population))]
        station = stations[rng.integers(len(stations))]
    noise_seed = int(noise.generate_state(1, dtype=np.uint64)[0])
    return Tape(f"{cohort}:{seed}:{index}", source, noise_seed, station)


def _balanced_tapes(seed, cohort, pairs, per_pair):
    return [_tape(seed, cohort, i, source, station)
            for i, (source, station) in enumerate(pair for pair in pairs for _ in range(per_pair))]


def _station_population(sim, sources, stations):
    return [(source, tuple(s for s in sim.station_ids(source) if s in stations)) for source in sources]


def _iid_tapes(seed, cohort, population, count):
    if not population or any(not stations for _, stations in population):
        missing = [source for source, stations in population if not stations]
        raise ValueError(f"{cohort} requires a supported station on every source; missing: {missing}")
    return [_tape(seed, cohort, i, population=population) for i in range(count)]


def _episode(sim, tape, benchmark, frames, *, offset_mm=0., augment=None):
    return sim.episode(tape.source_id, tape.seed, benchmark=benchmark, frames=frames,
                       station_id=tape.station_id, offset_mm=offset_mm, augment=augment)


def _run(reader, episode, *, timed=False, after_step=None):
    reader.reset()
    estimates, elapsed = [], []
    for i, frame in enumerate(episode.frames, 1):
        inputs = (frame.obs, frame.probe_pose, frame.f_contact, frame.q_hat, frame.Sigma_q, frame.dt)
        if timed:
            start = perf_counter_ns()
            estimate = reader.step(*inputs)
            elapsed.append((perf_counter_ns() - start) * 1e-6)
        else:
            estimate = reader.step(*inputs)
        estimates.append(estimate)
        if after_step is not None:
            after_step(i)
    return estimates, np.asarray(elapsed, dtype=float)


def _training_records(reader, episodes):
    records = []
    for tape, episode in episodes:
        estimates, _ = _run(reader, episode)
        records.extend(make_training_record(tape.source_id, tape.episode_id, frame, estimate, episode.station_id)
                       for frame, estimate in zip(episode.frames, estimates, strict=True))
    return records


def _caliber_prior(episodes):
    diameters = []
    for _, episode in episodes:
        for frame in episode.frames:
            for echo, roles in zip(frame.obs.echoes, frame.echo_targets, strict=True):
                if roles and np.isfinite(echo[2]):
                    scale = max(.2, 1. + echo[3]) if np.isfinite(echo[3]) else 1.
                    diameters.append(float(echo[2] / (scale * np.sqrt(len(roles)))))
    if not diameters:
        raise ValueError("train-only caliber fit needs at least one lumen echo")
    return {"caliber_mean_mm": float(np.mean(diameters)), "caliber_std_mm": max(float(np.std(diameters)), 2.5)}


def _dev_filter(prior, kinematics, episodes, caliber):
    # Fix the GP and acoustic model first. Select only the correlation weights
    # on an independent, train-source-only dev cohort before fitting the head.
    results = []
    for rho_corr, rho_novel in ((.15, .85), (.20, .90), (.25, .95)):
        params = dict(caliber, rho_corr=rho_corr, rho_novel=rho_novel)
        reader = DynamicFactorGraphReader(prior, kinematics, **params)
        nll = 0.
        for _, episode in episodes:
            estimates, _ = _run(reader, episode)
            for frame, estimate in zip(episode.frames, estimates, strict=True):
                for probs, contributors in zip(estimate.identity_probabilities, frame.echo_targets, strict=True):
                    mass = probs[_echo_mask(contributors)]
                    nll -= float(np.log(max(mass, 1.e-12)))
        results.append((nll, params))
    return min(results, key=lambda result: result[0])[1]


def _segment_record(tape, episode, estimates):
    return make_segment_record(tape.source_id, tape.episode_id, episode.frames, estimates)


def _blender_plan(args):
    return {name: getattr(args, name) for name in (
        "seed", "episodes", "station_episodes", "frames", "risk_episodes", "bootstrap")}


def _fit(args, bank, split, prior, sim):
    population = _station_population(sim, split.train_source_ids, STATION_IDS)
    pairs = [(source, station) for source, stations in population for station in stations]
    profiles = (None, ORDINARY_MATERIAL, STRESS_MATERIAL)
    fit_repeats = max(args.fit_episodes_per_station, -(-1000 // (len(pairs) * args.segment_frames)))
    fit_repeats += -fit_repeats % len(profiles)  # every source/station pair gets the same profile mixture
    fit_tapes = _balanced_tapes(args.seed, "fit", pairs, fit_repeats)
    temp_tapes = _balanced_tapes(args.seed, "temperature", pairs, args.temperature_episodes_per_station)
    cal_tapes = _iid_tapes(args.seed, "conformal_calibration", population, args.calibration_episodes)
    fit_episodes = [(tape, _episode(sim, tape, "station", args.segment_frames, augment=profiles[i % len(profiles)]))
                    for i, tape in enumerate(fit_tapes)]
    dev_episodes = [(tape, _episode(sim, tape, "station", args.segment_frames)) for tape in temp_tapes]
    # Moving surfaces, rechecks and large offsets train the role head without
    # station targets or reuse of an independent conformal calibration tape.
    source_cases = [(source, None) for source in split.train_source_ids]
    for cohort, dest, repeats in (("role_fit", fit_episodes, 12), ("role_temperature", dev_episodes, 4)):
        for i, tape in enumerate(_balanced_tapes(args.seed, cohort, source_cases, repeats)):
            benchmark = ("dynamic", "retroactive", "offset")[i % 3]
            offset = (20., 40.)[(i // 3) % 2] if benchmark == "offset" else 0.
            dest.append((tape, _episode(sim, tape, benchmark, args.frames, offset_mm=offset)))
    # Augment head fitting only. Nuisance caliber, dev/temperature, the
    # separate calibration population and all evaluation draws remain nominal.
    augmented_profiles = {profile.name for profile in profiles if profile is not None}
    caliber_episodes = (item for item in fit_episodes if item[1].provenance["material_profile"] not in augmented_profiles)
    filter_params = _dev_filter(prior, sim.kinematics, dev_episodes, _caliber_prior(caliber_episodes))
    base_reader = DynamicFactorGraphReader(prior, sim.kinematics, compute_ipe=True, **filter_params)
    fit_records = _training_records(base_reader, fit_episodes)
    temp_records = _training_records(base_reader, dev_episodes)
    scorer = fit_station_scorer(fit_records, temp_records, split, device=args.device, seed=args.seed,
                                epochs=args.epochs, batch_size=args.batch_size)
    frozen_reader = DynamicFactorGraphReader(prior, sim.kinematics, scorer=scorer, **filter_params)
    records = []
    for tape in cal_tapes:
        episode = _episode(sim, tape, "conformal", args.segment_frames)
        estimates, _ = _run(frozen_reader, episode)
        records.append(_segment_record(tape, episode, estimates))
    certifier = fit_segment_certifier(records, split, alpha=ALPHA, tolerance_mm=TOLERANCE_MM,
                                      scorer=scorer)
    provenance = {
        "schema_version": CHECKPOINT_SCHEMA,
        "protocol": args.split,
        "scoring_protocol": SCORING_PROTOCOL,
        "acquisition_protocol": "paired nominal probe schedule; exact contributor identities learned on training sources only",
        "tracking_protocol": TRACKING_PROTOCOL,
        "source_sha256": _source_sha256(),
        "checkpoint_only_training": args.train_only,
        "seed": args.seed,
        "training_source_ids": list(split.train_source_ids),
        "training_source_provenance": {source: dict(bank.source(source).provenance)
                                       for source in split.train_source_ids},
        "shared_geometry_provenance": {key: bank.provenance[key] for key in (
            "coordinate_frame", "native_graph_file", "neutral_smplx_surface_file",
            "neutral_smplx_surface_sha256", "bone_occupancy_file")},
        "station_ids": list(STATION_IDS),
        "segment_frames": args.segment_frames,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "training_device": args.device,
        "fit_frames": len(fit_records),
        "fit_episodes_per_source_station": fit_repeats,
        "temperature_frames": len(temp_records),
        "filter_parameters": filter_params,
        "temperature": {"echo_identity": float(scorer.identity_temperature),
                        "station": float(scorer.station_temperature)},
        "conformal_alpha": ALPHA,
        "conformal_tolerance_mm": TOLERANCE_MM,
        "conformal_sampling": "IID uniform training source, then uniform supported station, then independent scan seed",
        "conformal_population": {source: list(stations) for source, stations in population},
        "fit_tapes": [{**asdict(tape), "benchmark": episode.provenance["benchmark"],
                       "offset_mm": episode.provenance["offset_mm"],
                       "material_profile": episode.provenance["material_profile"]} for tape, episode in fit_episodes],
        "temperature_tapes": [{**asdict(tape), "benchmark": episode.provenance["benchmark"],
                               "offset_mm": episode.provenance["offset_mm"],
                               "material_profile": episode.provenance["material_profile"]}
                              for tape, episode in dev_episodes],
        "calibration_tapes": [asdict(tape) for tape in cal_tapes],
    }
    if OPEN3DMODEL_RIGHT in split.train_source_ids:
        provenance["training_source_provenance"][OPEN3DMODEL_RIGHT]["registered_surface_sha256"] = (
            bank.provenance["open3d_registered_surface_sha256"])
    if args.split == BLENDER:
        provenance["locked_blender_evaluation"] = _blender_plan(args)
    return scorer, certifier, provenance


def _load(args, split, population):
    try:
        scorer, certifier, provenance = load_station_checkpoint(args.checkpoint, split)
    except ValueError as exc:
        raise ValueError(f"incompatible Step 3 checkpoint (evaluator schema {CHECKPOINT_SCHEMA} required): {exc}") from exc
    if (provenance.get("schema_version") != CHECKPOINT_SCHEMA
            or provenance.get("scoring_protocol") != SCORING_PROTOCOL
            or provenance.get("tracking_protocol") != TRACKING_PROTOCOL):
        raise ValueError(f"incompatible Step 3 checkpoint: evaluator schema {CHECKPOINT_SCHEMA} "
                         "requires refitting with the current observation/acquisition protocol")
    cal_tapes = provenance.get("calibration_tapes", [])
    expected = {source: list(stations) for source, stations in population}
    if (certifier is None
            or provenance.get("protocol") != args.split or len(cal_tapes) < 200
            or certifier.calibration_segments != len(cal_tapes)
            or certifier.alpha != ALPHA or certifier.tolerance_mm != TOLERANCE_MM
            or provenance.get("training_source_ids") != list(split.train_source_ids)
            or provenance.get("conformal_population") != expected
            or provenance.get("conformal_alpha") != ALPHA
            or set(provenance.get("filter_parameters", ())) != {"rho_corr", "rho_novel", "caliber_mean_mm", "caliber_std_mm"}
            or provenance.get("conformal_tolerance_mm") != TOLERANCE_MM):
        raise ValueError("checkpoint lacks compatible Step 3 split/conformal provenance")
    if args.split == BLENDER:
        if provenance.get("checkpoint_only_training") is not True:
            raise ValueError("Blender evaluation requires a frozen checkpoint from --train-only")
        if provenance.get("locked_blender_evaluation") != _blender_plan(args):
            raise ValueError("Blender benchmark parameters must match the plan frozen by --train-only")
    return scorer, certifier, provenance


@dataclass(frozen=True)
class _TapeTruth:
    centers: np.ndarray
    observed: np.ndarray  # one opportunity per unique rendered role/frame, including all merge contributors
    supported: np.ndarray  # all finite truth centers in the slice, whether rendered or not
    echo_masks: tuple[np.ndarray, ...]
    echo_counts_by_size: np.ndarray

    def acquisition(self):
        counts = self.echo_counts_by_size
        finite_truth = int(self.supported.sum())
        rendered_finite_truth = int((self.observed & self.supported).sum())
        return {
            "frames": len(self.observed),
            "frames_with_lumen_echo": int(self.observed.any(axis=1).sum()),
            "rendered_roles": [role.name for role in VesselRole if self.observed[:, role].any()],
            "rendered_frame_role_opportunities": int(self.observed.sum()),
            "rendered_frame_role_opportunities_by_role": dict(zip(
                (role.name for role in VesselRole), self.observed.sum(axis=0).tolist(), strict=True)),
            "all_slice_truth_frame_role_opportunities": finite_truth,
            "finite_truth_frame_roles_with_lumen_echo": rendered_finite_truth,
            "rendered_fraction_on_finite_slice_truth": rendered_finite_truth / finite_truth if finite_truth else None,
            "rendered_frame_role_opportunities_without_finite_truth": int(self.observed.sum()) - rendered_finite_truth,
            "rendered_echoes": int(counts.sum()),
            "rendered_lumen_echoes": int(counts[1:].sum()),
            "rendered_echoes_by_contributor_count": counts.tolist(),
            "empty_echo_tape": not counts.any(),
            "empty_lumen_tape": not counts[1:].any(),
        }


def _echo_mask(contributors):
    mask = 0
    for role in contributors:
        mask |= 1 << int(role)
    return mask


def _tape_truth(episode):
    centers = np.stack([frame.centers_mm for frame in episode.frames])
    supported = np.isfinite(centers).all(axis=-1)
    observed = np.zeros_like(supported)
    echo_masks = []
    for i, frame in enumerate(episode.frames):
        for contributors in frame.echo_targets:
            if contributors:
                observed[i, list(contributors)] = True
        echo_masks.append(np.fromiter(map(_echo_mask, frame.echo_targets), dtype=np.intp,
                                      count=len(frame.echo_targets)))
    counts = np.bincount(_MASK_SIZES[np.concatenate(echo_masks)], minlength=ROLE_COUNT + 1)
    return _TapeTruth(centers, observed, supported, tuple(echo_masks), counts)


@dataclass
class _EpisodeMeasures:
    role_counts: np.ndarray  # truth mass, prediction count, intersecting mass by class
    role_confusion: np.ndarray
    squared_errors: np.ndarray  # rendered frame-role pairs only, NaN for unmeasured/non-opportunities
    all_slice_squared_errors: np.ndarray
    tape_truth: _TapeTruth
    exact_masks_correct_by_size: np.ndarray
    latency_ms: np.ndarray
    terminal_echo_correct: bool | None
    station_counts: np.ndarray
    station_correct: bool | None
    flips: np.ndarray  # wrong -> right, previously wrong, right -> wrong, previously right
    exact_flips: np.ndarray
    prefix_echo_before: bool | None
    prefix_echo_after: bool | None
    prefix_history_latency_ms: float | None
    history_latency_ms: float | None


def _echo_labels(frame, estimate):
    predicted = np.argmax(estimate.role_probabilities, axis=1)
    truth = np.zeros((len(frame.echo_targets), 7), dtype=float)
    for i, roles in enumerate(frame.echo_targets):
        truth[i, list(roles) if roles else [6]] = 1. / max(len(roles), 1)
    if len(predicted) != len(truth):
        raise ValueError("reader returned a different number of echoes than on the tape")
    mass = truth[np.arange(len(predicted)), predicted]
    return truth, predicted, mass


def _exact_mask_hits(estimate, truth_masks):
    probs = estimate.identity_probabilities
    if probs is None or probs.shape != (len(truth_masks), _N_MASKS):
        raise ValueError("reader must return one posterior over exact contributor bitmasks per echo")
    return np.argmax(probs, axis=1) == truth_masks


def _as_outcome(hits):
    return bool(hits[0]) if len(hits) else None


def _measure(reader, episode, benchmark, tape_truth):
    prefix = None
    prefix_history_ms = None
    stop = max(1, len(episode.frames) // 2)

    def replay_prefix(frame_count):
        nonlocal prefix, prefix_history_ms
        if frame_count == stop:
            start = perf_counter_ns()
            prefix = reader.relabel_history()
            prefix_history_ms = (perf_counter_ns() - start) * 1e-6

    estimates, elapsed = _run(reader, episode, timed=True,
                              after_step=replay_prefix if benchmark == "retroactive" else None)
    role = np.zeros((3, 7))
    role_confusion = np.zeros((7, 7))
    exact_correct = np.zeros(ROLE_COUNT + 1, dtype=int)
    hits = []
    for frame, estimate, masks in zip(episode.frames, estimates, tape_truth.echo_masks, strict=True):
        truth, predicted, mass = _echo_labels(frame, estimate)
        role[0] += truth.sum(axis=0)
        role[1] += np.bincount(predicted, minlength=7)
        role[2] += np.bincount(predicted, weights=mass, minlength=7)
        np.add.at(role_confusion.T, predicted, truth)
        exact_correct += np.bincount(_MASK_SIZES[masks[_exact_mask_hits(estimate, masks)]],
                                    minlength=ROLE_COUNT + 1)
        hits.append(mass > 0.)
    centers = np.stack([est.centers_mm for est in estimates])
    errors = np.square(centers - tape_truth.centers).sum(axis=-1)
    errors[~tape_truth.supported | ~np.isfinite(errors)] = np.nan
    observed_errors = np.where(tape_truth.observed, errors, np.nan)
    stations = np.zeros((len(STATION_IDS), len(STATION_IDS) + 1), dtype=int)
    station_correct = before = after = history_ms = None
    flips = np.zeros(4, dtype=int)
    exact_flips = np.zeros(4, dtype=int)
    if benchmark == "station":
        truth_station = STATION_IDS.index(episode.station_id)
        probabilities = np.mean([estimate.station_probabilities for estimate in estimates], axis=0)
        prediction = int(np.argmax(probabilities)) if np.any(probabilities > 0.) else len(STATION_IDS)
        stations[truth_station, prediction] = 1
        station_correct = truth_station == prediction
    if benchmark == "retroactive":
        start = perf_counter_ns()
        revised = reader.relabel_history()
        history_ms = (perf_counter_ns() - start) * 1e-6
        if prefix is None or len(prefix) != stop or len(revised) != len(estimates):
            raise ValueError("relabel_history must return one estimate per frame acquired at each endpoint")
        # Use identical replay inference/decision operators on all prefix slots
        # at the two endpoints; only the second endpoint has later evidence.
        for frame, initial, revised_estimate, masks in zip(episode.frames[:stop], prefix, revised[:stop],
                                                           tape_truth.echo_masks[:stop], strict=True):
            old, new = (_echo_labels(frame, est)[2] > 0. for est in (initial, revised_estimate))
            old_exact, new_exact = (_exact_mask_hits(est, masks) for est in (initial, revised_estimate))
            flips += ((~old & new).sum(), (~old).sum(), (old & ~new).sum(), old.sum())
            exact_flips += ((~old_exact & new_exact).sum(), (~old_exact).sum(),
                            (old_exact & ~new_exact).sum(), old_exact.sum())
            if before is None and len(old):
                before, after = _as_outcome(old), _as_outcome(new)
    return _EpisodeMeasures(role, role_confusion, observed_errors, errors, tape_truth, exact_correct, elapsed,
                            _as_outcome(hits[-1]), stations, station_correct, flips, exact_flips, before, after,
                            prefix_history_ms, history_ms)


def _divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)


def _bootstrap_weights(strata, replicates, seed):
    """Resample whole episode indices, identically for every paired method."""
    groups = {}
    for row, key in enumerate(strata):
        groups.setdefault(key, []).append(row)
    weights = np.zeros((replicates + 1, len(strata)), dtype=np.int32)
    weights[0] = 1  # first row computes the point estimate, not a resample
    rng = np.random.default_rng(seed)
    for group in groups.values():
        draws = rng.choice(group, size=(replicates, len(group)), replace=True)
        np.add.at(weights, (np.arange(1, replicates + 1)[:, None], draws), 1)
    return weights


def _number(value):
    return float(value) if np.isfinite(value) else None


def _interval(values):
    boot = values[1:][np.isfinite(values[1:])]
    return {"value": _number(values[0]),
            "ci95": np.quantile(boot, [.025, .975]).tolist() if len(boot) else None,
            "defined_bootstrap_draws": len(boot)}


def _role_f1(counts):
    truth, pred, tp = np.moveaxis(counts, -2, 0)
    return np.where(truth > 0, _divide(2 * tp, truth + pred), np.nan)


def _macro(f1):
    return _divide(np.nansum(f1, axis=-1), np.isfinite(f1).sum(axis=-1))


def _aggregates(runs, weights, stations):
    role = np.einsum("bn,nrc->brc", weights, np.stack([r.role_counts for r in runs]))
    role_f1 = _role_f1(role)

    def tracking(errors_name, opportunity_mask_name):
        centers = [(np.nansum(values), np.isfinite(values).sum(), getattr(run.tape_truth, opportunity_mask_name).sum())
                   for run in runs for values in (getattr(run, errors_name),)]
        sse, measured, opportunities = (weights @ np.asarray(centers)).T
        finite_rmse = np.sqrt(_divide(sse, measured))
        # No population estimate or interval if any original opportunity was
        # unmeasured. A bootstrap draw omitting those tapes is not a remedy.
        rmse = np.where((measured == opportunities) & (measured[0] == opportunities[0]), finite_rmse, np.nan)
        return rmse, finite_rmse, _divide(measured, opportunities), int(measured[0]), int(opportunities[0])

    observed = tracking("squared_errors", "observed")
    all_slice = tracking("all_slice_squared_errors", "supported")
    exact_correct = weights @ np.stack([r.exact_masks_correct_by_size for r in runs])
    exact_count = weights @ np.stack([r.tape_truth.echo_counts_by_size for r in runs])
    latency = np.asarray([(r.latency_ms.sum(), len(r.latency_ms)) for r in runs])
    total_ms, steps = (weights @ latency).T
    out = {
        "role_macro_f1": _macro(role_f1[:, :6]),
        "null_f1": role_f1[:, 6],
        "fractional_role_accuracy_including_null": _divide(role[:, 2].sum(axis=-1), role[:, 0].sum(axis=-1)),
        "exact_mask_accuracy_including_null": _divide(exact_correct.sum(axis=1), exact_count.sum(axis=1)),
        "exact_lumen_mask_accuracy": _divide(exact_correct[:, 1:].sum(axis=1), exact_count[:, 1:].sum(axis=1)),
        "tracking_rmse_mm": observed[0],
        "tracking_conditional_finite_rmse_mm": observed[1],
        "tracking_center_coverage": observed[2],
        "all_slice_role_rmse_mm": all_slice[0],
        "all_slice_role_conditional_finite_rmse_mm": all_slice[1],
        "all_slice_role_center_coverage": all_slice[2],
        "mean_step_ms": _divide(total_ms, steps),
    }
    details = {
        "role_truth_mass": role[0, 0].tolist(),
        "role_f1_by_column": [_number(value) for value in role_f1[0]],
        "role_confusion_true_by_predicted": np.sum([r.role_confusion for r in runs], axis=0).tolist(),
        "tracking_centers_measured": observed[3],
        "tracking_centers_observed": observed[4],
        "tracking_centers_unmeasured": observed[4] - observed[3],
        "all_slice_role_centers_measured": all_slice[3],
        "all_slice_role_centers_supported": all_slice[4],
        "all_slice_role_centers_unmeasured": all_slice[4] - all_slice[3],
        "exact_mask_accuracy_by_true_contributor_count": {
            str(size): {"echo_slots": int(exact_count[0, size]), "correct_echo_slots": int(exact_correct[0, size]),
                        "accuracy": _interval(_divide(exact_correct[:, size], exact_count[:, size]))}
            for size in range(ROLE_COUNT + 1)
        },
        "step_latency_p50_p95_ms": np.quantile(np.concatenate([r.latency_ms for r in runs]), [.5, .95]).tolist(),
    }
    if runs[0].history_latency_ms is not None:
        for count_name, attribute, prefix in (("prefix_flip_counts", "flips", ""),
                                              ("prefix_exact_mask_flip_counts", "exact_flips", "exact_mask_")):
            wr, w, rw, r = (weights @ np.stack([getattr(run, attribute) for run in runs])).T
            out[prefix + "wrong_to_right"] = _divide(wr, w)
            out[prefix + "right_to_wrong"] = _divide(rw, r)
            details[count_name] = dict(wrong_to_right=int(wr[0]), previously_wrong=int(w[0]),
                                       right_to_wrong=int(rw[0]), previously_right=int(r[0]))
        out["mean_relabel_history_ms"] = (
            weights @ np.asarray([run.history_latency_ms for run in runs])) / weights.sum(axis=1)
        out["mean_prefix_relabel_history_ms"] = (
            weights @ np.asarray([run.prefix_history_latency_ms for run in runs])) / weights.sum(axis=1)
        details["prefix_echo_mcnemar_before_to_after"] = _mcnemar(
            [run.prefix_echo_after for run in runs], [run.prefix_echo_before for run in runs])
    if stations:
        confusion = np.einsum("bn,nij->bij", weights, np.stack([r.station_counts for r in runs]))
        truth, pred = confusion.sum(axis=2), confusion.sum(axis=1)[:, :len(STATION_IDS)]
        f1 = np.where(truth > 0, _divide(2 * np.diagonal(confusion, axis1=1, axis2=2), truth + pred), np.nan)
        columns = [STATION_IDS.index(s) for s in stations]
        selected = f1[:, columns]
        out["station_macro_f1"] = np.where(np.isfinite(selected).all(axis=1), _macro(selected), np.nan)
        details["station_f1_by_class"] = dict(zip(STATION_IDS, map(_number, f1[0]), strict=True))
        details["station_confusion_true_by_predicted"] = confusion[0].tolist()
        details["missing_expected_station_classes"] = [s for s, j in zip(stations, columns, strict=True)
                                                       if truth[0, j] == 0]
    return out, details


def _mcnemar(full, other):
    paired = tuple(zip(full, other, strict=True))
    pairs = [(a, b) for a, b in paired if a is not None and b is not None]
    wins = sum(bool(a) and not b for a, b in pairs)
    losses = sum(bool(b) and not a for a, b in pairs)
    p = (float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.) if pairs else None
    return {"paired_episodes": len(pairs), "excluded_episodes": len(paired) - len(pairs),
            "first_only_correct": wins, "second_only_correct": losses,
            "p_two_sided": p}


def _signed_rank(first, second, alternative):
    delta = np.asarray(first) - np.asarray(second)
    delta = delta[np.isfinite(delta)]
    nonzero = np.count_nonzero(delta)
    result = (float(wilcoxon(delta, alternative=alternative).pvalue) if nonzero else 1.) if len(delta) else None
    return {"paired_episodes": len(delta), "nonzero_differences": int(nonzero),
            "alternative_first_vs_second": alternative, "p": result}


def _paired_rmse(first, second, metadata):
    a, b, counts = [], [], []
    for i, j, tape in zip(first, second, metadata, strict=True):
        one, two = i.squared_errors, j.squared_errors
        observed = i.tape_truth.observed
        finite_first, finite_second = np.isfinite(one), np.isfinite(two)
        common = finite_first & finite_second
        matched, opportunities = int(common.sum()), int(observed.sum())
        counts.append({"episode_id": tape["episode_id"], "opportunities": opportunities, "common": matched,
                       "first_only": int((finite_first & ~finite_second).sum()),
                       "second_only": int((finite_second & ~finite_first).sum()),
                       "neither": int((observed & ~finite_first & ~finite_second).sum())})
        if matched:
            a.append(np.sqrt(one[common].mean()))
            b.append(np.sqrt(two[common].mean()))
    return {
        **_signed_rank(a, b, "less"),
        "episodes_total": len(counts),
        "episodes_without_opportunities": sum(c["opportunities"] == 0 for c in counts),
        "episodes_with_opportunities_but_no_common_prediction": sum(c["opportunities"] > 0 and c["common"] == 0
                                                                     for c in counts),
        "episodes_with_partial_common_predictions": sum(0 < c["common"] < c["opportunities"] for c in counts),
        "matched_frame_role_centers": sum(c["common"] for c in counts),
        "total_frame_role_opportunities": sum(c["opportunities"] for c in counts),
        "by_tape": counts,
    }


def _collect(readers, sim, tapes, benchmark, frames, *, offset_mm=0.):
    measured = {name: [] for name in METHODS}
    metadata = []
    for i, tape in enumerate(tapes):
        episode = _episode(sim, tape, benchmark, frames, offset_mm=offset_mm)
        truth = _tape_truth(episode)
        metadata.append({**asdict(tape), "h_start_mm": float(episode.h_start_mm),
                         "geometry": episode.provenance["geometry"], **truth.acquisition()})
        if benchmark in ("dynamic", "offset"):
            metadata[-1]["physical_cue_frames"] = {
                "compression": sum(4.3 <= f.f_contact <= 4.5 and any(
                    labels and np.isfinite(echo[3]) for echo, labels in zip(f.obs.echoes, f.echo_targets, strict=True))
                    for f in episode.frames),
                "pulse": sum(f.probe_pose.dwell_s >= .35 and any(
                    labels and np.isfinite(echo[4]) for echo, labels in zip(f.obs.echoes, f.echo_targets, strict=True))
                    for f in episode.frames),
            }
        if benchmark == "offset":
            metadata[-1]["offset_policy"] = episode.provenance["offset_policy"]
        elif benchmark == "retroactive":
            metadata[-1].update({key: episode.provenance[key] for key in (
                "rewindow_deg", "rewindow_step", "echo_counts_by_rewindow")})
        # Rotate inference order to reduce systematically assigning warm-cache effects to Full.
        for name in METHODS[i % len(METHODS):] + METHODS[:i % len(METHODS)]:
            measured[name].append(_measure(readers[name], episode, benchmark, truth))
    return measured, metadata


def _acquisition_summary(metadata):
    finite_truth = sum(tape["all_slice_truth_frame_role_opportunities"] for tape in metadata)
    rendered_finite_truth = sum(tape["finite_truth_frame_roles_with_lumen_echo"] for tape in metadata)
    return {
        "frames_without_lumen_echo": sum(tape["frames"] - tape["frames_with_lumen_echo"] for tape in metadata),
        "frames": sum(tape["frames"] for tape in metadata),
        "empty_echo_tapes": sum(tape["empty_echo_tape"] for tape in metadata),
        "empty_lumen_tapes": sum(tape["empty_lumen_tape"] for tape in metadata),
        "rendered_frame_role_opportunities": sum(tape["rendered_frame_role_opportunities"] for tape in metadata),
        "all_slice_truth_frame_role_opportunities": finite_truth,
        "finite_truth_frame_roles_with_lumen_echo": rendered_finite_truth,
        "rendered_fraction_on_finite_slice_truth": rendered_finite_truth / finite_truth if finite_truth else None,
        "rendered_frame_role_opportunities_without_finite_truth": sum(
            tape["rendered_frame_role_opportunities_without_finite_truth"] for tape in metadata),
        "rendered_lumen_echoes": sum(tape["rendered_lumen_echoes"] for tape in metadata),
        "rendered_echoes_by_contributor_count": [
            sum(tape["rendered_echoes_by_contributor_count"][i] for tape in metadata)
            for i in range(ROLE_COUNT + 1)],
    }


def _benchmark_report(runs, metadata, weights, benchmark, stations=()):
    report = {"episodes": len(metadata), "tapes": metadata, "methods": {}, "paired_full_minus_baseline": {}}
    report["acquisition"] = _acquisition_summary(metadata)
    if not metadata:
        report["status"] = "not_estimable_no_supported_episodes"
        if stations:
            report["expected_station_classes"] = list(stations)
        return report, {}
    report["acquisition"]["rendered_role_episode_counts_by_station"] = {
        station: {role.name: sum(tape["station_id"] == station and role.name in tape["rendered_roles"]
                                 for tape in metadata)
                  for role in VesselRole}
        for station in sorted({tape["station_id"] for tape in metadata if tape["station_id"] is not None})}
    if benchmark in ("dynamic", "offset"):
        report["acquisition"]["episodes_with_at_least_two_cue_frames"] = {
            key: sum(tape["physical_cue_frames"][key] >= 2 for tape in metadata)
            for key in ("compression", "pulse")
        }
    if benchmark == "retroactive":
        report["acquisition"]["echo_counts_by_rewindow"] = {
            phase: {key: sum(tape["echo_counts_by_rewindow"][phase][key] for tape in metadata) for key in counts}
            for phase, counts in metadata[0]["echo_counts_by_rewindow"].items()
        }
        report["acquisition"]["echo_count_policy"] = (
            "counts rendered echo groups by simulator provenance; a mixed group contributes to both ghost and lumen totals"
        )
    if stations:
        report["expected_station_classes"] = list(stations)
    aggregates = {}
    for name in METHODS:
        aggregates[name], details = _aggregates(runs[name], weights, stations)
        details["tracking_by_tape"] = [
            {"episode_id": tape["episode_id"],
             "rendered_centers_measured": int(np.isfinite(run.squared_errors).sum()),
             "rendered_centers_unmeasured": int((run.tape_truth.observed & ~np.isfinite(run.squared_errors)).sum()),
             "all_slice_centers_measured": int(np.isfinite(run.all_slice_squared_errors).sum()),
             "all_slice_centers_unmeasured": int((run.tape_truth.supported & ~np.isfinite(run.all_slice_squared_errors)).sum())}
            for run, tape in zip(runs[name], metadata, strict=True)]
        report["methods"][name] = {**details, **{key: _interval(v) for key, v in aggregates[name].items()}}
    for baseline in METHODS[:-1]:
        first, second = runs[FULL], runs[baseline]
        endpoint = {"station": "station_correct", "retroactive": "prefix_echo_after"}.get(benchmark, "terminal_echo_correct")
        comparison = {
            "delta": {key: _interval(value - aggregates[baseline][key]) for key, value in aggregates[FULL].items()},
            "mcnemar_endpoint": endpoint,
            "mcnemar": _mcnemar([getattr(r, endpoint) for r in first], [getattr(r, endpoint) for r in second]),
        }
        if benchmark == "dynamic":
            comparison["common_center_rmse_wilcoxon"] = _paired_rmse(first, second, metadata)
        report["paired_full_minus_baseline"][baseline] = comparison
    return report, aggregates


def _weights(tapes, args, domain, *, stratified=True):
    strata = [(tape.source_id, tape.station_id) if stratified else None for tape in tapes]
    seed = np.random.SeedSequence([args.seed, 100 + domain])
    return _bootstrap_weights(strata, args.bootstrap, seed)


def _station_benchmark(args, sim, readers, sources, stations, cohort="station"):
    population = _station_population(sim, sources, stations)
    pairs = [(source, station) for source, available in population for station in available]
    tapes = _balanced_tapes(args.seed, cohort, pairs, args.station_episodes)
    runs, metadata = _collect(readers, sim, tapes, "station", args.frames)
    report, _ = _benchmark_report(runs, metadata, _weights(tapes, args, COHORTS[cohort]), "station", stations)
    report["source_station_support"] = {source: list(available) for source, available in population}
    return report


def _scan_benchmarks(args, sim, readers, sources, stations):
    cases = [(source, None) for source in sources]
    tapes = _balanced_tapes(args.seed, "offset", cases, args.episodes)
    weights = _weights(tapes, args, COHORTS["offset"])
    offsets, offset_metrics = {}, {}
    for offset in OFFSETS_MM:
        runs, metadata = _collect(readers, sim, tapes, "offset", args.frames, offset_mm=offset)
        offsets[str(int(offset))], offset_metrics[offset] = _benchmark_report(runs, metadata, weights, "offset")
    report = {"offset": {
        "offset_axis": "commanded circumferential material-arc mm, not achieved Euclidean translation",
        "probe_aim": ("fixed nominal neutral-skin probe schedule aimed at the training-only trunk; all paired offset "
                      "levels reuse probe and tracker inputs, and only source anatomy is arc shifted"),
        "by_offset_mm": offsets,
        "paired_role_macro_f1_40_minus_0_mm": {
            method: _interval(offset_metrics[40.][method]["role_macro_f1"] - offset_metrics[0.][method]["role_macro_f1"])
            for method in METHODS},
    }}
    for benchmark in ("dynamic", "retroactive"):
        tapes = _balanced_tapes(args.seed, benchmark, cases, args.episodes)
        runs, metadata = _collect(readers, sim, tapes, benchmark, args.frames)
        report[benchmark], _ = _benchmark_report(runs, metadata, _weights(tapes, args, COHORTS[benchmark]), benchmark)
    report["station"] = _station_benchmark(args, sim, readers, sources, stations)
    return report


def _risk_rates(outcomes, weights):
    missed, wrong_cert, certified = (weights @ outcomes.astype(np.int64)).T
    count = weights.sum(axis=1)
    return {"miscoverage": _divide(missed, count), "false_certification": _divide(wrong_cert, count),
            "acceptance_rate": _divide(certified, count), "conditional_failure": _divide(wrong_cert, certified)}


def _risk_benchmark(args, sim, readers, certifier, provenance, sources, stations, *, matched):
    cohort = "matched_train_risk" if matched else "held_risk"
    population = (_station_population(sim, sources, stations) if not matched else
                  [(source, tuple(available)) for source, available in provenance["conformal_population"].items()])
    report = {
        "scope": "same synthetic source/station mixture as calibration, fresh tapes" if matched else
                 "shifted held anatomy/source mixture; diagnostic only",
        "rank_coverage_claim_applicable_to_full_marginal_miscoverage": matched,
        "independent_anatomies_claimed": False,
        "episode_sampling_population": {source: list(available) for source, available in population},
        "segment_frames": provenance["segment_frames"],
        "methods": {},
    }
    if not matched:
        # Held source omissions stay visible instead of samplers silently resampling failures.
        population = [(source, available) for source, available in population if available]
        if not population:
            report["status"] = "not_estimable_no_supported_episodes"
            return report
    # The root is frozen for the matched audit, even when evaluating a reused checkpoint.
    seed = provenance["seed"] if matched else args.seed
    tapes = _iid_tapes(seed, cohort, population, args.risk_episodes)
    weights = _weights(tapes, args, COHORTS[cohort], stratified=False)
    nominal = uncalibrated_segment_certifier(alpha=ALPHA, tolerance_mm=TOLERANCE_MM)
    records = {name: [] for name in METHODS}
    report["tapes"] = []
    for tape in tapes:
        episode = _episode(sim, tape, "conformal", provenance["segment_frames"])
        report["tapes"].append({**asdict(tape), "h_start_mm": float(episode.h_start_mm),
                                **_tape_truth(episode).acquisition()})
        for name in METHODS:
            estimates, _ = _run(readers[name], episode)
            records[name].append(_segment_record(tape, episode, estimates))
    report["acquisition"] = _acquisition_summary(report["tapes"])
    results, rates = {}, {}
    policies = {name: certifier if name == FULL else nominal for name in METHODS}
    policies["Full_AGPS_Reader_nominal_reference"] = nominal
    for name, policy in policies.items():
        outcomes = policy.outcomes(records[name if name in records else FULL])
        results[name] = outcomes
        rates[name] = _risk_rates(outcomes, weights)
        audit = policy.audit(outcomes=outcomes)
        if not matched and name == FULL:
            audit["frozen_training_rank_bound_not_applicable_here"] = audit.pop("rank_violation_bound")
            audit["rank_coverage_scope"] = None
            for stratum in audit["strata"].values():
                stratum["frozen_training_rank_bound_not_applicable_here"] = stratum.pop("rank_violation_bound")
        audit["certification_status"] = "some_certificates" if audit["certified_segments"] else "all_abstained"
        report["methods"][name] = {
            **audit,
            "threshold_policy": "split_conformal" if name == FULL else "nominal_unadjusted_no_rank_claim",
            "cluster_bootstrap": {key: _interval(value) for key, value in rates[name].items()},
        }
    report["paired_full_minus_comparator"] = {
        name: {"delta": {key: _interval(value - rates[name][key]) for key, value in rates[FULL].items()},
               "coverage_mcnemar": _mcnemar(~results[FULL][:, 0], ~results[name][:, 0])}
        for name in results if name != FULL}
    return report


def _gate(args, benchmarks, risk, held_risk):
    if args.split == BLENDER:
        return {"status": "diagnostic_only_no_gate"}
    checks, engineering_controls = {}, {}

    def criterion(name, point, op, threshold, *, engineering=False):
        passed = None if point is None else {">=": point >= threshold, "<=": point <= threshold, "<": point < threshold}[op]
        target = engineering_controls if engineering else checks
        target[name] = {"value": point, "criterion": f"{op} {threshold:g}",
                        "status": "insufficient_evidence" if passed is None else "pass" if passed else "fail"}
        return target[name]

    def metric(bench, key, reader=FULL):
        return bench["methods"].get(reader, {}).get(key, {}).get("value")

    if args.split == "LOAO_OPEN3D":
        offset = benchmarks["offset"]["by_offset_mm"]["40"]
        dynamic = benchmarks["dynamic"]
        retro = benchmarks["retroactive"]
        criterion("offset_40_full_role_f1", metric(offset, "role_macro_f1"), ">=", .85)
        criterion("offset_40_rigid27_role_f1", metric(offset, "role_macro_f1", "B0_Rigid27"), "<", .40)
        baseline = metric(offset, "role_macro_f1", "B1_NoTopology_AbsXYZ")
        full = metric(offset, "role_macro_f1")
        criterion("offset_40_topology_advantage", None if baseline is None or full is None else baseline - full, "<", 0.)
        criterion("dynamic_role_f1", metric(dynamic, "role_macro_f1"), ">=", .85)
        dynamic_full = dynamic["methods"].get(FULL, {})
        measured, opportunities = (dynamic_full.get(key, 0) for key in
                                   ("tracking_centers_measured", "tracking_centers_observed"))
        complete = 0 < opportunities == measured
        tracking_check = criterion("dynamic_tracking_rmse_mm",
                                   metric(dynamic, "tracking_rmse_mm") if complete else None, "<=", 4.)
        tracking_check.update(measured_frame_roles=measured, rendered_frame_role_opportunities=opportunities,
                              unmeasured_frame_roles=opportunities - measured,
                              conditional_finite_rmse_mm=metric(dynamic, "tracking_conditional_finite_rmse_mm"))
        if not complete:
            tracking_check["reason"] = ("incomplete_predictions_or_geometry" if opportunities else
                                        "no_rendered_frame_role_opportunities")
        common = dynamic["paired_full_minus_baseline"].get("B2_NoPullback", {}).get("common_center_rmse_wilcoxon", {})
        paired_check = criterion("dynamic_vs_no_pullback_wilcoxon", common.get("p"), "<", .01)
        paired_check["scope"] = "episodic signed-rank comparison conditional on common finite predictions"
        paired_check["paired_episodes"] = common.get("paired_episodes", 0)
        times = dynamic_full.get("step_latency_p50_p95_ms", ())
        criterion("dynamic_tracking_median_step_ms", times[0] if times else None, "<=", 5.)
        criterion("retroactive_wrong_to_right", metric(retro, "wrong_to_right"), ">=", .60)
        criterion("retroactive_right_to_wrong", metric(retro, "right_to_wrong"), "<", .05)
        full_flip = metric(retro, "wrong_to_right")
        b3_flip = metric(retro, "wrong_to_right", "B3_NoDiscount")
        criterion("retroactive_flip_advantage_vs_no_discount",
                  None if full_flip is None or b3_flip is None else b3_flip - full_flip, "<", 0.)
        criterion("station_S1_S5_macro_f1", metric(benchmarks["station"], "station_macro_f1"), ">=", .85)
        b4 = metric(benchmarks["station"], "station_macro_f1", "B4_NoBeliefIPE")
        full = metric(benchmarks["station"], "station_macro_f1")
        criterion("station_belief_ipe_advantage", None if b4 is None or full is None else b4 - full, "<", 0.)
    elif args.split == "FAMILY_HELD_OUT_ATLASES":
        criterion("station_S1_S4_macro_f1", metric(benchmarks["station"], "station_macro_f1"), ">=", .80)
    conformal = risk["methods"][FULL]
    criterion("matched_certification_acceptance", conformal["acceptance_rate"], ">=", .40, engineering=True)
    criterion("matched_in_source_whole_segment_miscoverage",
              conformal["miscoverage"]["rate"] if not conformal["cutoff_unbounded"] else None, "<=", ALPHA)
    conditional = conformal["conditional_failure"]["rate"]
    comparison = risk["methods"]["B3_NoDiscount"]["conditional_failure"]["rate"]
    criterion("matched_certified_segment_conditional_violation", conditional, "<=", ALPHA)
    criterion("matched_conditional_violation_advantage_vs_no_discount",
              None if conditional is None or comparison is None else conditional - comparison, "<", 0.)
    criterion("matched_unconditional_false_certification", conformal["false_certification"]["rate"], "<=", ALPHA)
    held = held_risk["methods"].get(FULL, {})
    criterion("held_source_certification_acceptance", held.get("acceptance_rate"), ">=", .40, engineering=True)
    criterion("held_source_unconditional_false_certification", held.get("false_certification", {}).get("rate"), "<=", ALPHA)
    held_conditional = held.get("conditional_failure", {}).get("rate")
    held_baseline = held_risk["methods"].get("B3_NoDiscount", {}).get("conditional_failure", {}).get("rate")
    criterion("held_source_conditional_violation", held_conditional, "<=", ALPHA)
    criterion("held_source_conditional_violation_advantage_vs_no_discount",
              None if held_conditional is None or held_baseline is None else held_conditional - held_baseline, "<", 0.)
    statuses = {test["status"] for test in (*checks.values(), *engineering_controls.values())}
    return {"status": "fail" if "fail" in statuses else "insufficient_evidence" if "insufficient_evidence" in statuses else "pass",
            "checks": checks, "engineering_controls": engineering_controls,
            "engineering_controls_scope": "pre-existing minimum acceptance checks, not a theoretical guarantee",
            "scope": "protocol-specific point-estimate criteria; no held-atlas or conditional conformal guarantee"}


def _legacy_route_diagnostic(legacy, comparable_h):
    report = {}
    for name, route in legacy.routes.items():
        rows = route.global_row_indices
        roles = np.asarray(route.role_ids)
        supported = route.supported[:, roles]
        reachable = route.physically_reachable_mask[:, roles]
        contact = legacy.valid_contact[rows].any(axis=1)
        valid = supported & reachable
        comparable = np.isin(route.h_mm, comparable_h)
        report[name] = {
            "station_rows": len(rows),
            "rows_all_raw_slots_outside_bone": int(legacy.station_physically_reachable_mask[rows].sum()),
            "rows_with_valid_contact": int(contact.sum()),
            "rows_with_all_requested_slots_supported_reachable_and_valid_contact": int((valid.all(axis=1) & contact).sum()),
            "rows_sampled_by_named_direct_thigh_reader_benchmark": int(comparable.sum()),
            "by_legacy_slot": {str(int(role)): {
                "supported": int(supported[:, j].sum()),
                "missing": int((~supported[:, j]).sum()),
                "supported_bone_excluded": int((supported[:, j] & ~reachable[:, j]).sum()),
                "supported_physically_reachable": int(valid[:, j].sum()),
                "reachable_with_valid_contact": int((valid[:, j] & contact).sum()),
                "reachable_without_valid_contact": int((valid[:, j] & ~contact).sum()),
                "adjacent_supported_links": int(route.links[:, role].sum()),
                "adjacent_physically_reachable_links": int((route.links[:, role] & valid[:-1, j] & valid[1:, j]).sum()),
            } for j, role in enumerate(roles)},
        }
    return {"scope": "archived route geometry and bone/contact feasibility, not a capture/continuation trial",
            "reader_scope": "only the separate canonical thigh station evaluations; no raw legacy slot is relabelled as DFA",
            "station_rows": len(legacy.h_mm),
            "physically_reachable_station_rows": int(legacy.station_physically_reachable_mask.sum()),
            "role_identities": legacy.role_identities, "routes": report, "provenance": legacy.provenance}


def _blender_sources(thigh, stations, *, inferred):
    # Keep the real legacy arterial gap and topology. Only R_DFA's inferred samples
    # can enter the inferred panel; do not mark its physical reachability as known.
    allowed = (thigh.support_source != 0 if inferred else thigh.directly_supported)
    centers = np.where(allowed[..., None], thigh.centers_mm, np.nan)
    names = np.where(allowed, thigh.named_roles, "")
    links = thigh.raw_geometry_links.copy()
    links[:, VesselRole.DFA] = inferred & (np.abs(np.diff(thigh.h_mm)) <= 8.)
    links &= allowed[:-1] & allowed[1:]
    source = AnatomySource(thigh.source_id, thigh.h_mm, centers, allowed, links, names,
                           {**thigh.provenance, "inferred_R_DFA_enabled": inferred})
    windows = {}
    for station in stations:
        j = STATION_IDS.index(station)
        roles = [role for role in NYSORA_STATIONS[station] if role != VesselRole.DFA]
        mask = (thigh.station_named_role_support[:, j] & thigh.valid_contact.any(axis=1)
                & thigh.physically_reachable_mask[:, roles].all(axis=1))
        windows[station] = thigh.h_mm[mask]
    return source, windows


def _blender_eval(args, bank, split, sim, readers, certifier, provenance):
    if (args.split != BLENDER or args.train_only or not args.checkpoint or
            os.environ.get(HOLDOUT_ENV) != "1" or provenance.get("checkpoint_only_training") is not True):
        raise RuntimeError("Blender requires explicit authorization and a frozen --train-only checkpoint")
    from sim.holdout_eval_assets import load_blender_legacy_routes, load_blender_nysora_thigh

    thigh = load_blender_nysora_thigh()
    legacy = load_blender_legacy_routes()
    direct, direct_windows = _blender_sources(thigh, DIRECT_STATIONS, inferred=False)
    inferred, inferred_windows = _blender_sources(thigh, INFERRED_STATIONS, inferred=True)
    sim.register_evaluation_sources((direct,), station_windows_by_source={direct.source_id: direct_windows})
    inferred_sim = DynamicScanSimulator(bank, split, kinematics=sim.kinematics)
    inferred_sim.register_evaluation_sources((inferred,), station_windows_by_source={inferred.source_id: inferred_windows})
    report = _scan_benchmarks(args, sim, readers, split.test_source_ids, DIRECT_STATIONS)
    report["station"]["evidence"] = "direct named roles only; R_DFA absent from these simulated observations"
    report["station_inferred_DFA"] = _station_benchmark(
        args, inferred_sim, readers, split.test_source_ids, INFERRED_STATIONS, "inferred_station")
    report["station_inferred_DFA"]["evidence"] = "R_DFA support_source=2, retargeted path; physical reachability unknown for inferred DFA"
    report["blender_support"] = {
        "station_truth_scope": "named role co-presence surrogate with archived contact and bone checks, not annotated junction or clinical station labels",
        "direct_candidate_h_mm": {key: value.tolist() for key, value in direct_windows.items()},
        "inferred_candidate_h_mm": {key: value.tolist() for key, value in inferred_windows.items()},
        "direct_main_artery_missing_rows_minus_545_to_minus_520": int((
            ~thigh.directly_supported[:, VesselRole.MAIN_ARTERY]
            & (thigh.h_mm >= -545.) & (thigh.h_mm <= -520.)).sum()),
        "provenance": thigh.provenance,
    }
    comparable = [tape["h_start_mm"] for tape in report["station"]["tapes"]]
    report["physical_legacy_routes"] = _legacy_route_diagnostic(legacy, comparable)
    report["held_conformal_diagnostic"] = _risk_benchmark(
        args, sim, readers, certifier, provenance, split.test_source_ids, DIRECT_STATIONS, matched=False)
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLIT_PROTOCOLS, default="LOAO_OPEN3D")
    parser.add_argument("--out", type=Path, help="Benchmark JSON (required unless --train-only)")
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--checkpoint", type=Path, help="Reuse a frozen scorer and conformal certifier; no refit")
    checkpoints.add_argument("--train-and-freeze", "--checkpoint-out", dest="checkpoint_out", type=Path,
                             help="Save newly fitted scorer, certifier and provenance")
    parser.add_argument("--train-only", action="store_true", help="Only write a training checkpoint; do not evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="Torch fitting device")
    parser.add_argument("--episodes", type=int, default=16, help="Tapes per test source for each scan comparison")
    parser.add_argument("--station-episodes", type=int, default=6, help="Tapes per supported test source/station")
    parser.add_argument("--frames", type=int, default=8, help="Frames per scan benchmark tape")
    parser.add_argument("--segment-frames", type=int, default=4, help="Frames per training/risk tape; frozen in checkpoint")
    parser.add_argument("--fit-episodes-per-station", type=int, default=4, help="Fit tapes per supported train source/station")
    parser.add_argument("--temperature-episodes-per-station", type=int, default=2, help="Separate temperature tapes per train source/station")
    parser.add_argument("--calibration-episodes", type=int, default=200, help="Independent train-source calibration tapes (minimum 200)")
    parser.add_argument("--risk-episodes", type=int, default=80, help="Independent tapes per risk evaluation population")
    parser.add_argument("--bootstrap", type=int, default=500, help="Number of paired episode bootstrap draws")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def _validate(parser, args):
    if args.train_only and (args.checkpoint_out is None or args.out is not None):
        parser.error("--train-only requires --checkpoint-out and does not write --out")
    if not args.train_only and args.out is None:
        parser.error("--out is required for evaluation")
    allow_blender = args.split == BLENDER and not args.train_only
    if allow_blender:
        if os.environ.get(HOLDOUT_ENV) != "1" or args.checkpoint is None:
            parser.error(f"{BLENDER} evaluation requires {HOLDOUT_ENV}=1 and --checkpoint")
    elif HOLDOUT_ENV in os.environ:
        parser.error(f"unset {HOLDOUT_ENV} for every training or training-atlas evaluation invocation")
    if args.checkpoint_out is not None and args.out is not None and args.checkpoint_out.resolve() == args.out.resolve():
        parser.error("--checkpoint-out and --out must be different files")
    if args.checkpoint_out is not None and args.checkpoint_out.exists():
        parser.error("--checkpoint-out may not overwrite an existing checkpoint")
    if args.checkpoint is not None and args.out.resolve() == args.checkpoint.resolve():
        parser.error("--out may not overwrite the frozen --checkpoint")
    minimums = {"seed": 0, "frames": 2, "segment_frames": 1, "calibration_episodes": 200}
    for name in ("episodes", "station_episodes", "fit_episodes_per_station", "temperature_episodes_per_station",
                 "risk_episodes", "bootstrap", "epochs", "batch_size"):
        minimums[name] = 1
    for name, low in minimums.items():
        if getattr(args, name) < low:
            parser.error(f"--{name.replace('_', '-')} must be >= {low}")
    return allow_blender


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Only modules used by the training/atlas pipeline. Never discover sources by
# traversing these packages: sim also contains authorization-gated modules.
_PIPELINE_SOURCE_FILES = (
    "reader/__init__.py",
    "reader/baselines.py",
    "reader/belief_ipe.py",
    "reader/dynamic_factor_graph.py",
    "reader/eval_reader.py",
    "reader/joint_association.py",
    "reader/split_prior.py",
    "reader/station_classifier.py",
    "sim/__init__.py",
    "sim/anatomy_assets.py",
    "sim/audit_training_support.py",
    "sim/material_chart.py",
    "sim/metric_graph_gp.py",
    "sim/physics_ultrasound_kernel.py",
    "sim/smplx_dynamic_atlas.py",
)


def _source_sha256():
    root = Path(__file__).resolve().parent.parent
    return {name: _sha256(root / name) for name in _PIPELINE_SOURCE_FILES}


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    blender = _validate(parser, args)
    split = SPLIT_PROTOCOLS[args.split]
    bank = TrainingAnatomyBank.load()
    prior = build_split_prior(bank, split)
    sim = DynamicScanSimulator(bank, split)
    if args.checkpoint is None:
        scorer, certifier, provenance = _fit(args, bank, split, prior, sim)
    else:
        population = _station_population(sim, split.train_source_ids, STATION_IDS)
        scorer, certifier, provenance = _load(args, split, population)
    checkpoint_path = args.checkpoint or args.checkpoint_out
    if args.checkpoint_out is not None:
        scorer.save(args.checkpoint_out, certifier=certifier, metadata=provenance)
    if args.train_only:
        print(f"Frozen Step 3 checkpoint: {checkpoint_path}")
        return 0
    if checkpoint_path is None:
        with TemporaryDirectory(prefix="agps-checkpoint-") as directory:
            ephemeral = Path(directory) / "reader.pt"
            scorer.save(ephemeral, certifier=certifier, metadata=provenance)
            checkpoint_sha256 = _sha256(ephemeral)
    else:
        checkpoint_sha256 = _sha256(checkpoint_path)
    readers = make_readers(prior, sim.kinematics, scorer=scorer, **provenance["filter_parameters"])
    if set(readers) != set(METHODS):
        raise ValueError(f"expected exactly the six Step 3 readers, got {list(readers)}")
    stations = PROTOCOL_STATIONS[args.split]
    if blender:
        benchmarks = _blender_eval(args, bank, split, sim, readers, certifier, provenance)
        held = benchmarks.pop("held_conformal_diagnostic")
    else:
        benchmarks = _scan_benchmarks(args, sim, readers, split.test_source_ids, stations)
        held = _risk_benchmark(args, sim, readers, certifier, provenance, split.test_source_ids, stations, matched=False)
    matched = _risk_benchmark(args, sim, readers, certifier, provenance, split.train_source_ids, STATION_IDS, matched=True)
    gate = _gate(args, benchmarks, matched, held)
    source_sha256 = _source_sha256()
    if blender:
        # Only reached after _blender_eval has checked the holdout authority
        # and evaluated using the matching locked checkpoint.
        name = "sim/holdout_eval_assets.py"
        source_sha256[name] = _sha256(Path(__file__).resolve().parent.parent / name)
    report = {
        "evaluation_schema_version": CHECKPOINT_SCHEMA,
        "source_sha256": source_sha256,
        "protocol": args.split,
        "relationship": split.relationship,
        "evaluation_source_ids": list(split.test_source_ids),
        "evaluation_seed": args.seed,
        "clinical_station_labels_available": False,
        "measurement_scope": "synthetic label-blind ultrasound echoes and tracker poses, not B-mode or robot trials",
        "methodology": {
            "pairing": "same immutable simulator episode for all six methods; offset levels also share tape seeds",
            "confidence_intervals": "95% percentile episode-cluster bootstrap; paired draws across methods/offsets; source/station stratified for balanced comparisons, IID for risk",
            "bootstrap_draws": args.bootstrap,
            "role_f1": "pooled macro-F1 over vessel roles with truth support; argmax predictions, merged truth contributes 1/k per contributor; NULL reported separately",
            "role_column_order": [role.name for role in VesselRole] + ["NULL"],
            "scoring_protocol": SCORING_PROTOCOL,
            "exact_contributor_mask": ("argmax over the 64 contributor bitmasks; mask zero is NULL. Accuracy is "
                                       "reported overall, for lumen slots, and by true group size, including 0 for NULL."),
            "acquisition": ("fixed simulator truth counts: unique rendered roles per frame, lumen and total echo slots, "
                            "and empty tapes. These counts do not establish capture of any absent target."),
            "rendered_fraction_on_finite_slice_truth": ("fraction of finite simulator slice centers contributing to "
                                                        "a lumen echo; not coverage of targets without measured truth"),
            "tracking_protocol": TRACKING_PROTOCOL,
            "tracking": ("world XYZ Euclidean RMSE over all unique rendered frame-role opportunities (including all "
                         "merge contributors, regardless of predicted identity). Any unmeasured opportunity makes "
                         "the primary estimate undefined. Conditional finite RMSE, coverage and all-slice results "
                         "are diagnostics; all-slice opportunities require finite simulator truth."),
            "paired_tracking": ("Wilcoxon compares within-episode RMSE on pairs both readers measure; report empty "
                                "episodes and all missing/one-sided pairs separately. This is not the primary RMSE."),
            "mcnemar": ("exact two-sided, one outcome per tape: station argmax, first echo of the terminal frame, "
                        "or first available prefix echo for retroactive (membership in merged truth counts as correct); "
                        "empty endpoints are excluded in both arms and counted."),
            "station": "five-way argmax of mean frame posteriors across the acquisition, abstain if unsupported; no test-label masking; macro-F1 over prescribed classes, undefined if any expected class lacks truth",
            "station_confusion_true_row_order": list(STATION_IDS),
            "station_confusion_predicted_column_order": [*STATION_IDS, "ABSTAIN"],
            "retroactive": ("same history replay at midpoint before recheck and at termination; compare decisions "
                            "on every slot in the prefix. Top-1 membership and exact-mask flips reported separately."),
            "latency": "wall-clock reader.step only, per frame; history relabelling timed separately; does not include simulation, metric extraction or reset",
            "conformal": ("Full uses a frozen rank threshold; other readers and the Full nominal control use an "
                          "unadjusted threshold with no rank claim. Tests compare conditional failures to B3 within "
                          "each population; there is no rank guarantee for conditional risk, held-source transfer, "
                          "or acceptance engineering controls. Only rendered lumen identity and geometry can be certified."),
        },
        "checkpoint": {"path": str(checkpoint_path) if checkpoint_path else None,
                       "sha256": checkpoint_sha256,
                       "sha256_scope": "checkpoint file bytes" if checkpoint_path else "ephemeral serialized checkpoint bytes",
                       "reused_without_refit": args.checkpoint is not None},
        "training_provenance": provenance,
        "benchmarks": benchmarks,
        "conformal": {"matched_train_source_test": matched, "held_source_diagnostic": held},
        "gate": gate,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.split}: {gate['status']} — {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
