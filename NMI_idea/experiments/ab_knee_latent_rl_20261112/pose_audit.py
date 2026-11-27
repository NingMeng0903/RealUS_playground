"""Controlled latent-knee audit.

This audit keeps the synthetic Y-vessel subject and atlas supplied by
``knee_world.KneeWorld`` and creates two physical knee histories which share
the same angle and angular speed at a checkpoint.  A common calibration frame
is assimilated first.  Ultrasound is then withheld while the two histories
move through different intermediate knee rates.  The latent filter and the
zero-latent filter receive exactly the same public calibration and pose
readings.  Future forecasts are compared with the evaluator-only points.

The extra history construction is deliberately local to this diagnostic.  It
uses the filter's declared first-order dynamics so that a nonzero difference
at equal angle and speed is an identifiable test of temporal state.  It does
not pass hidden branch labels, z values, or future positions to either filter.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from belief import CONFIDENCE, Filter, Reading, VELOCITY_COUPLING, LAG_TAU
from knee_world import KneeCase, KneeWorld


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "pose_audit_20261112"
SEED_BASE = 31_310_492
N_PAIRS = 64
CAL_TIME = 0.0
REST_TIME = -0.5
CHECKPOINT_TIME = 2.0
FUTURE_TIMES = (2.5, 3.0)
# Central 90% Gaussian interval used for coordinate coverage.
Z90 = 1.6448536269514722
MOTION_TIMES = tuple(float(value) for value in np.linspace(0.0, 2.0, 9))


@dataclass(frozen=True)
class PoseHistory:
    """Public pose schedule with a hidden evaluator state generated below."""

    name: str
    readings: tuple[Reading, ...]

    def at(self, time_s: float) -> Reading:
        for reading in self.readings:
            if abs(reading.t - time_s) <= 1.0e-9:
                return reading
        if time_s >= self.readings[-1].t:
            return Reading(float(time_s), 15.0, 0.0)
        raise KeyError(f"history has no reading at {time_s}")


def seed_at(index: int) -> int:
    return int(np.random.SeedSequence([SEED_BASE, int(index)]).generate_state(1)[0])


def _history_pair() -> tuple[PoseHistory, PoseHistory]:
    # Smooth opposite excursions meet at exactly 15 degrees and zero speed.
    # The continuously sampled readings are the physical-time sensor inputs.
    times = MOTION_TIMES
    phase = np.pi * np.asarray(times)
    flexion_q = 17.0 - 2.0 * np.cos(phase)
    flexion_v = 2.0 * np.pi * np.sin(phase)
    extension_q = 13.0 + 2.0 * np.cos(phase)
    extension_v = -2.0 * np.pi * np.sin(phase)
    def readings(q_values, v_values):
        # Make the paired checkpoint a byte-identical public sensor reading;
        # the two physical histories differ before/after it, not in q/v at
        # the comparison instant.
        result = []
        for t, q, v in zip(times, q_values, v_values):
            if abs(float(t) - CHECKPOINT_TIME) < 1.e-12:
                q, v = 15.0, 0.0
            result.append(Reading(float(t), float(q), float(v)))
        return tuple(result)
    left = PoseHistory("flexion", readings(flexion_q, flexion_v))
    right = PoseHistory("extension", readings(extension_q, extension_v))
    return left, right


class BoundWorld:
    """Bind the worker's case API to the public model expected by Filter."""

    def __init__(self, world: KneeWorld, case: KneeCase, reading: Reading):
        self.base = world
        self.case = case
        self.reading = reading
        self.n = world.site_count
        self.n_views = int(world.scan.views.shape[1])
        self.branch_sites = set(world.branch_sites)

    def set_reading(self, reading: Reading) -> None:
        self.reading = reading

    def identity_site(self, site: int) -> bool:
        return int(site) in self.branch_sites

    def projection(self, q: float, site: int, view: int):
        try:
            result = self.base.projection(float(q), int(site), int(view))
            if isinstance(result, tuple):
                return np.asarray(result[0], dtype=float), np.asarray(result[1], dtype=float)
        except TypeError:
            result = self.base.projection(self.case, self.reading.t, int(site), int(view),
                                          pose_reading=(float(q), float(self.reading.velocity)))
        return np.asarray(result["base_mm"], dtype=float), np.asarray(result["rows"], dtype=float)

    def detection_probability(self, q: float, site: int, view: int) -> float:
        return .91

    def clutter_probability(self, site: int, view: int) -> float:
        return float(self.case.clutter)

    def world_design(self, q: float, sites: Iterable[int]):
        """Return the public atlas design in millimetres.

        The clean worker world exposes this operation as ``world_design(q,
        ((site, label), ...))``.  Keeping the adapter at this boundary is
        important: the filter must consume exactly the same atlas geometry as
        the production sensor model, rather than a private/interpolated copy.
        Integer site inputs are normalized to trunk labels for callers from
        the existing filter API.
        """
        labels = []
        for item in sites:
            if isinstance(item, (tuple, list, np.ndarray)):
                if len(item) != 2:
                    raise ValueError(f"site label must have length two: {item!r}")
                labels.append((int(item[0]), int(item[1])))
            else:
                labels.append((int(item), 0))
        xyz, design = self.base.world_design(float(q), tuple(labels))
        return np.asarray(xyz, dtype=float), np.asarray(design, dtype=float)

    def candidates(self, reading: Reading, previous=None):
        return self.base.legal_actions(reading.t, previous=previous)


def _true_z(z: float, previous: Reading, reading: Reading, *, tau: float,
            coupling: float, process_noise: float = 0.0) -> float:
    dt = float(reading.t - previous.t)
    decay = np.exp(-dt / float(tau))
    return float(decay * z + float(coupling) * float(tau) * (1.0 - decay) * reading.velocity + process_noise)


def _advance_truth(z: float, history: PoseHistory, initial: float, *, tau: float,
                   coupling: float, process_sd: float,
                   rng: np.random.Generator) -> dict[float, float]:
    # Shared rest calibration: the initial latent state decays before motion
    # starts, so the first moving frame is not the only source of z evidence.
    rest = Reading(REST_TIME, 15.0, 0.0)
    first = history.readings[0]
    z = _true_z(initial, rest, first, tau=tau, coupling=coupling,
                process_noise=0.0)
    values = {REST_TIME: float(initial), float(first.t): float(z)}
    for previous, reading in zip(history.readings[:-1], history.readings[1:]):
        z = _true_z(z, previous, reading, tau=tau, coupling=coupling,
                    process_noise=float(rng.normal(0.0, process_sd)))
        values[float(reading.t)] = z
    for previous, reading in zip((history.readings[-1], Reading(2.5, 15.0, 0.0)),
                                 (Reading(2.5, 15.0, 0.0), Reading(3.0, 15.0, 0.0))):
        z = _true_z(z, previous, reading, tau=tau, coupling=coupling,
                    process_noise=float(rng.normal(0.0, process_sd)))
        values[float(reading.t)] = z
    return values


def _actual_point(world: KneeWorld, case: KneeCase, reading: Reading, site: int, label: int, z: float) -> np.ndarray:
    base, design = world.world_design(float(reading.q), ((int(site), int(label)),))
    row = np.asarray(design[0], dtype=float)
    displacement = np.array([float(case.w_subject_mm[int(site)]),
                             float(case.branch_offset_mm) if int(label) == 1 else 0.0,
                             float(z)])
    point_mm = np.asarray(base[0], dtype=float) + row @ displacement
    # Public world_design is in WORLD mm; traces are serialized in metres.
    return point_mm / 1000.0


def _observe(world: BoundWorld, case: KneeCase, reading: Reading, site: int, view: int,
             z: float, rng: np.random.Generator, *, clean: bool = False) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    """Generate an unordered public frame and evaluator labels separately."""

    nominal, rows = world.projection(reading.q, int(site), int(view))
    labels = [0] + ([1] if int(site) in world.branch_sites else [])
    observations: list[tuple[float, float]] = []
    eval_labels: list[int] = []
    points: list[np.ndarray] = []
    for label in labels:
        displacement = np.array([case.w_subject_mm[int(site)],
                                 case.branch_offset_mm if label == 1 else 0.0, z])
        center = nominal[labels.index(label)] + rows[labels.index(label)] @ displacement
        if abs(float(center[0])) > world.base.aperture_mm or (not clean and rng.random() > .94):
            continue
        observations.append((float(center[0] + rng.normal(0., .85)),
                            float(center[1] + rng.normal(0., .4))))
        eval_labels.append(int(label))
        points.append(_actual_point(world.base, case, reading, site, label, z))
    if not clean and rng.random() < min(.22, case.clutter + .06):
        observations.append((float(rng.uniform(-world.base.aperture_mm, world.base.aperture_mm)),
                             float(rng.uniform(3., 55.))))
        eval_labels.append(-1)
        points.append(np.full(3, np.nan))
    order = np.argsort([value[0] for value in observations], kind="stable")
    return (np.asarray([observations[i] for i in order], dtype=float).reshape(-1, 2),
            tuple(eval_labels[i] for i in order),
            np.asarray([points[i] for i in order], dtype=float))


def _calibrate(world: BoundWorld, case: KneeCase,
               states: tuple[tuple[Reading, float], ...], sites: tuple[int, ...],
               seed: int):
    frames = []
    for rest_index, (reading, z) in enumerate(states):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7001, rest_index]))
        for site in sites:
            values, labels, points = _observe(
                world, case, reading, site, world.base.scan.central, z, rng, clean=True)
            frames.append((reading, site, values, labels, points))
    return frames


def _filter_from_calibration(world: BoundWorld, case: KneeCase, calibration, *, latent: bool):
    filt = Filter(latent=latent)
    for reading, site, values, _, _ in calibration:
        world.set_reading(reading)
        filt.observe(
            world,
            reading,
            site,
            world.base.scan.central,
            values,
            event=0,
            retain=world.identity_site(site),
        )
    return filt


def _run_history(world: BoundWorld, case: KneeCase, calibration, history: PoseHistory,
                 sites: tuple[int, ...], latent: bool, motion_frames,
                 z_values: dict[float, float], *, use_motion_us: bool):
    filt = _filter_from_calibration(world, case, calibration, latent=latent)
    motion_trace = []
    for reading, site, values in motion_frames:
        world.set_reading(reading)
        if use_motion_us:
            filt.observe(world, reading, site, world.base.scan.central, values,
                         event=int(round(reading.t * 100.0)), retain=False)
        else:
            filt.advance(reading)
        trace_mean, trace_width = filt.forecast(world, reading.t, sites)
        trace_actual = np.stack([
            _actual_point(world.base, case, reading, trace_site, 0,
                          z_values[reading.t])
            for trace_site in sites
        ])
        motion_trace.append({
            "time_s": float(reading.t),
            "reading": reading.__dict__,
            "site": int(site),
            "z_true_mm": float(z_values[reading.t]),
            "z_estimated_mm": float(filt.stats()[0][2]),
            "mean_m": (trace_mean / 1000.0).tolist(),
            "actual_m": trace_actual.tolist(),
            "width_m": (trace_width / 1000.0).tolist(),
        })
    metrics = {
        "history": history.name,
        "latent": bool(latent),
        "motion_ultrasound": bool(use_motion_us),
        "checkpoint": {
            "time_s": CHECKPOINT_TIME,
            "reading": history.at(CHECKPOINT_TIME).__dict__,
            "true_z_mm": float(z_values[CHECKPOINT_TIME]),
            "estimated_z_mm": float(filt.stats()[0][2]),
        },
        "motion_trace": motion_trace,
        "future": [],
    }
    for time_s in FUTURE_TIMES:
        reading = history.at(time_s)
        world.set_reading(reading)
        means, widths = filt.forecast(world, time_s, sites)
        forecast_weights, _ = filt.prior_at(reading)
        forecast_z = float(np.sum(forecast_weights * filt.parameters[..., 2]))
        forecast_z_var = float(np.sum(forecast_weights * (filt.parameters[..., 2] - forecast_z) ** 2))
        actual = np.stack([_actual_point(world.base, case, reading, site, 0, z_values[time_s]) for site in sites])
        # The clean atlas/filter contract is millimetres.  Keep the trace in
        # metres for plotting, but calculate errors against the mm forecast;
        # mixing the two units creates a spurious 10^3-scale miss.
        actual_mm = 1000.0 * actual
        delta_mm = means - actual_mm
        covered = np.abs(delta_mm) <= Z90 * widths
        metrics["future"].append({
            "time_s": float(time_s),
            "reading": reading.__dict__,
            "z_true_mm": float(z_values[time_s]),
            "z_estimated_mm": forecast_z,
            "z_sd_mm": float(np.sqrt(max(0.0, forecast_z_var))),
            "point_rmse_mm": float(np.sqrt(np.mean(np.sum(delta_mm ** 2, axis=1)))),
            "point_max_error_mm": float(np.max(np.linalg.norm(delta_mm, axis=1))),
            "coverage_coordinate": float(np.mean(covered)),
            "mean_m": (means / 1000.0).tolist(),
            "actual_m": actual.tolist(),
            "width_m": (widths / 1000.0).tolist(),
        })
    return metrics


def _one_pair(world: KneeWorld, seed: int) -> dict[str, object]:
    case = world.case(seed, difficult=True)
    history_a, history_b = _history_pair()
    # High-response single-lumen stations (branch stations are 2/3/4).
    sites = tuple(sorted(set((1, 5, 6)) - set(world.branch_sites)))
    param_rng = np.random.default_rng(np.random.SeedSequence([seed, 7002]))
    initial_z = float(param_rng.normal(0., 3.5))
    true_tau = float(param_rng.uniform(.48, 1.18))
    true_coupling = float(param_rng.uniform(.64, 1.08))
    true_process_sd = float(param_rng.uniform(.06, .18))
    start = history_a.readings[0]
    bound = BoundWorld(world, case, start)
    truth_by_history = {}
    for history_code, history in enumerate((history_a, history_b)):
        truth_rng = np.random.default_rng(np.random.SeedSequence([seed, 7011, history_code]))
        truth_by_history[history.name] = _advance_truth(
            initial_z, history, initial_z, tau=true_tau, coupling=true_coupling,
            process_sd=true_process_sd, rng=truth_rng)
    # The shared rest-to-t=0 transition is deterministic, so both paired
    # histories enter motion with the same calibrated z state.
    assert np.isclose(truth_by_history[history_a.name][REST_TIME],
                      truth_by_history[history_b.name][REST_TIME])
    assert np.isclose(truth_by_history[history_a.name][CAL_TIME],
                      truth_by_history[history_b.name][CAL_TIME])
    calibration = _calibrate(
        bound, case,
        ((Reading(REST_TIME, 15.0, 0.0), truth_by_history[history_a.name][REST_TIME]),
         (start, truth_by_history[history_a.name][CAL_TIME])),
        sites, seed)
    methods: dict[str, list[dict[str, object]]] = {}
    public_motion: dict[str, list[dict[str, object]]] = {}
    for history_code, history in enumerate((history_a, history_b)):
        z_values = truth_by_history[history.name]
        motion_rng = np.random.default_rng(np.random.SeedSequence([seed, 7021, history_code]))
        frames = []
        for index, reading in enumerate(history.readings[1:], start=1):
            site = sites[(index - 1) % len(sites)]
            # Identical seeded read/dropout noise at the paired checkpoint
            # isolates the history-dependent latent state from sensor noise.
            frame_rng = motion_rng
            if abs(reading.t - CHECKPOINT_TIME) < 1.e-12:
                frame_rng = np.random.default_rng(np.random.SeedSequence([seed, 7022, index]))
            values, _, _ = _observe(bound, case, reading, site, bound.base.scan.central,
                                    z_values[reading.t], frame_rng, clean=False)
            frames.append((reading, site, values))
        public_motion[history.name] = [
            {"reading": reading.__dict__, "site": int(site), "n_observations": int(len(values))}
            for reading, site, values in frames
        ]
        methods.setdefault("dynamic_us", []).append(
            _run_history(bound, case, calibration, history, sites, True, frames, z_values,
                         use_motion_us=True))
        methods.setdefault("no_z_us", []).append(
            _run_history(bound, case, calibration, history, sites, False, frames, z_values,
                         use_motion_us=True))
        methods.setdefault("dynamic_withheld", []).append(
            _run_history(bound, case, calibration, history, sites, True, frames, z_values,
                         use_motion_us=False))
    checkpoint_a = methods["dynamic_us"][0]["checkpoint"]
    checkpoint_b = methods["dynamic_us"][1]["checkpoint"]
    assert checkpoint_a["reading"] == checkpoint_b["reading"]
    return {
        "seed": int(seed),
        "sites": list(sites),
        "checkpoint_equal": True,
        "checkpoint": {
            "actual_readings": {
                history_a.name: history_a.at(CHECKPOINT_TIME).__dict__,
                history_b.name: history_b.at(CHECKPOINT_TIME).__dict__,
            },
            "public_readings": {
                history_a.name: history_a.at(CHECKPOINT_TIME).__dict__,
                history_b.name: history_b.at(CHECKPOINT_TIME).__dict__,
            },
            "actual_reading_equal": bool(history_a.at(CHECKPOINT_TIME) == history_b.at(CHECKPOINT_TIME)),
            "public_reading_equal": bool(history_a.at(CHECKPOINT_TIME) == history_b.at(CHECKPOINT_TIME)),
        },
        "true_dynamics": {"tau_s": true_tau, "coupling": true_coupling,
                           "process_sd_mm": true_process_sd},
        "public_motion": public_motion,
        "histories": methods,
    }


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    methods = ("dynamic_us", "no_z_us", "dynamic_withheld")
    rows = {method: [r["histories"][method] for r in records] for method in methods}
    summary: dict[str, object] = {"pairs": len(records), "seed_base": SEED_BASE, "methods": {}}
    for method, pairs in rows.items():
        runs = [run for pair in pairs for run in pair]
        futures = [entry for run in runs for entry in run["future"]]
        checkpoints = [run["checkpoint"] for run in runs]
        summary["methods"][method] = {
            "checkpoint_z_rmse_mm": float(np.sqrt(np.mean([(x["true_z_mm"] - x["estimated_z_mm"]) ** 2 for x in checkpoints]))),
            # Each run has the same three evaluated stations, so pooling the
            # squared Euclidean errors is equivalent to the pooled RMSE.
            "future_point_rmse_mm": float(np.sqrt(np.mean([
                x["point_rmse_mm"] ** 2 for x in futures]))),
            "future_z_rmse_mm": float(np.sqrt(np.mean([(x["z_true_mm"] - x["z_estimated_mm"]) ** 2 for x in futures]))),
            "future_coordinate_coverage": float(np.mean([x["coverage_coordinate"] for x in futures])),
            "per_horizon": {
                str(time_s): {
                    "point_rmse_mm": float(np.sqrt(np.mean([
                        x["point_rmse_mm"] ** 2 for x in futures
                        if abs(x["time_s"] - time_s) < 1.e-9
                    ]))),
                    "z_rmse_mm": float(np.sqrt(np.mean([
                        (x["z_true_mm"] - x["z_estimated_mm"]) ** 2
                        for x in futures if abs(x["time_s"] - time_s) < 1.e-9
                    ]))),
                    "coordinate_coverage": float(np.mean([
                        x["coverage_coordinate"] for x in futures if abs(x["time_s"] - time_s) < 1.e-9
                    ])),
                }
                for time_s in FUTURE_TIMES
            },
        }
    gt_spread, predicted_spread = [], {method: [] for method in methods}
    for record in records:
        for index in range(len(FUTURE_TIMES)):
            a = record["histories"]["dynamic_us"][0]["future"][index]
            b = record["histories"]["dynamic_us"][1]["future"][index]
            gt = np.asarray(a["actual_m"]) - np.asarray(b["actual_m"])
            gt_spread.append(float(np.mean(np.linalg.norm(gt, axis=1)) * 1000.0))
            for method in methods:
                ma = record["histories"][method][0]["future"][index]
                mb = record["histories"][method][1]["future"][index]
                spread = np.asarray(ma["mean_m"]) - np.asarray(mb["mean_m"])
                predicted_spread[method].append(float(np.mean(np.linalg.norm(spread, axis=1)) * 1000.0))
    summary["history_spread_mm"] = {"ground_truth": float(np.mean(gt_spread)),
                                     **{f"{method}_prediction": float(np.mean(values))
                                        for method, values in predicted_spread.items()}}
    summary["equal_checkpoint_readings"] = bool(all(r["checkpoint_equal"] for r in records))
    summary["equal_checkpoint_actual_readings"] = bool(all(
        r["checkpoint"]["actual_reading_equal"] for r in records))
    summary["equal_checkpoint_public_readings"] = bool(all(
        r["checkpoint"]["public_reading_equal"] for r in records))
    return summary


def run(*, n_pairs: int = N_PAIRS) -> dict[str, object]:
    world = KneeWorld(noisy_pose=False)
    # Index zero is reserved for the interactive smoke run; final benchmarks
    # use disjoint deterministic seeds 1..N.
    records = []
    for pair_index in range(1, int(n_pairs) + 1):
        record = _one_pair(world, seed_at(pair_index))
        record["pair_index"] = int(pair_index)
        records.append(record)
    summary = _summarize(records)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS / "TRACE.json").write_text(json.dumps(records[0], indent=2) + "\n")
    with (RESULTS / "PAIRS.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=N_PAIRS)
    args = parser.parse_args()
    print(json.dumps(run(n_pairs=args.pairs), indent=2))


if __name__ == "__main__":
    main()
