"""Archival replay of ``s = r + z`` and a five-gain lag-model mixture.

The frozen ``belief``/``knee_world`` modules and ``runs.rule.trace`` records
are loaded from the original read-only knee experiment.  This is exploratory
evaluator code: it changes no motion or policy and uses no future reading.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = Path("/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_knee_latent_rl_20261112")
DEFAULT_OUT = HERE / "verification" / "RESPONSE.json"
GAINS = np.array([.55, .69, .83, 1., 1.18])
HORIZONS = (.5, 1.)
Z90 = 1.6448536269514722
BOOT = 4000
TOL = 2.e-9


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_archive(archive):
    archive = Path(archive).resolve()
    if not all((archive / name).is_file() for name in ("belief.py", "knee_world.py")):
        raise FileNotFoundError(f"original archive missing belief.py/knee_world.py: {archive}")
    sys.path.insert(0, str(archive))
    importlib.invalidate_caches()
    for name in ("belief", "knee_world"):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != archive / f"{name}.py":
            del sys.modules[name]
    try:
        B = importlib.import_module("belief")
        W = importlib.import_module("knee_world")
    except ImportError as exc:
        raise ImportError("use the original archive/dependencies; do not edit the archive") from exc
    if Path(B.__file__).resolve() != archive / "belief.py" or Path(W.__file__).resolve() != archive / "knee_world.py":
        raise ImportError("non-archive belief/knee_world module was loaded")
    return B, W


def read_rows(results):
    rows = []
    for path in sorted(Path(results).glob("test_[0-9]*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["index"]))
    indices = np.array([int(row["index"]) for row in rows])
    if not len(rows) or not np.array_equal(indices, np.arange(len(rows))):
        raise ValueError("test rows must be non-empty, unique, and contiguous")
    return rows


def shifted_bin(total, r0, r1, dt, tau, process_sd, ndtr):
    """Independent residual-bin matrix; target support is ``total-r1``."""
    if dt <= 1.e-10:
        return np.eye(len(total))
    a = np.exp(-dt / tau)
    mean = a * (total - r0)
    sd = max(.23, process_sd * np.sqrt(1. - a * a))
    mids = ((total - r1)[1:] + (total - r1)[:-1]) * .5
    cdf = ndtr((mids[None, :] - mean[:, None]) / sd)
    return np.diff(np.pad(cdf, ((0, 0), (1, 1)), constant_values=(0., 1.)), axis=1)


def gain_filter_class(B):
    class GainFilter(B.Filter):
        def __init__(self, gain):
            super().__init__(latent=True)
            self.gain = float(gain)

        def transition(self, dt, velocity):
            if dt < -1.e-10:
                raise ValueError("filter time must increase")
            if not self.latent or dt <= 1.e-10:
                return np.eye(len(self.z))
            a = np.exp(-dt / B.LAG_TAU)
            mean = a * self.z + self.gain * B.LAG_TAU * (1. - a) * velocity
            sd = max(.23, B.PROCESS_SD * np.sqrt(1. - a * a))
            mids = (self.z[1:] + self.z[:-1]) * .5
            cdf = B.ndtr((mids[None, :] - mean[:, None]) / sd)
            return np.diff(np.pad(cdf, ((0, 0), (1, 1)), constant_values=(0., 1.)), axis=1)

    return GainFilter


def residual_forecast(world, filt, B, weights, r, time, sites):
    reading = filt.predicted_reading(time)
    if filt.reading is None:
        prior, r1 = weights, 0.
    else:
        dt = reading.t - filt.reading.t
        v = .5 * (reading.velocity + filt.reading.velocity)
        a = np.exp(-dt / B.LAG_TAU)
        r1 = a * r + B.VELOCITY_COUPLING * B.LAG_TAU * (1. - a) * v
        prior = weights @ shifted_bin(B.Z_AXIS, r, r1, dt, B.LAG_TAU, B.PROCESS_SD, B.ndtr)
    pars = np.stack(np.meshgrid(filt.w, filt.b, B.Z_AXIS - r1, indexing="ij"), axis=-1)
    pars = pars.reshape(-1, len(B.Z_AXIS), 3)
    pars[..., 2] += r1
    xyz, design = world.world_design(reading.q, sites)
    samples = xyz + np.einsum("wzk,nck->wznc", pars, design)
    mean = np.einsum("wz,wznc->nc", prior, samples)
    var = np.einsum("wz,wznc->nc", prior, (samples - mean) ** 2)
    qhi, qlo = min(19., reading.q + .15), max(11., reading.q - .15)
    if qhi > qlo:
        xhi, _ = world.world_design(qhi, sites)
        xlo, _ = world.world_design(qlo, sites)
        pose_sd = .26 + .5 * max(0., time - filt.reading.t)
        var += (pose_sd * (xhi - xlo) / (qhi - qlo)) ** 2
    return mean, np.sqrt(var)


def fuse(forecasts, weights):
    means = np.array([item[0] for item in forecasts])
    sds = np.array([item[1] for item in forecasts])
    mean = np.einsum("g,gmc->mc", weights, means)
    var = np.einsum("g,gmc->mc", weights, sds * sds + (means - mean) ** 2)
    return mean, np.sqrt(np.maximum(var, 0.))


def metric(world, case, time, mean, sd):
    error = mean - world.truth_world(case, time, world.query_sites)
    return {
        "rmse_mm": float(np.sqrt(np.mean(np.sum(error * error, axis=1)))),
        "coverage_90": float(np.mean(np.abs(error) <= Z90 * sd)),
    }


def case_replay(row, world, B, GainFilter):
    case, run = world.case(int(row["seed"])), row["runs"]["rule"]
    trace, velocities = run["trace"], []
    qerr = cerr = 0.
    for frame in trace:
        replay = world.observe(case, frame["t"], frame["site"], frame["view"])
        saved = np.asarray(frame["centroids_mm"], float).reshape(-1, 2)
        qerr = max(qerr, abs(replay.q - frame["q"]))
        cerr = max(cerr, float(np.max(np.abs(replay.centroids - saved))) if saved.size else 0.)
        if abs(replay.q - frame["q"]) > 2.e-10 or replay.centroids.shape != saved.shape or cerr > 2.e-10:
            raise AssertionError(f"saved q/centroids do not replay: case {row['index']}")
        velocities.append(replay.velocity)

    old, gains = B.Filter(latent=True), [GainFilter(g) for g in GAINS]
    residual_weights, r, logw = old.weights.copy(), 0., np.log(np.ones(len(GAINS)) / len(GAINS))
    checks = {key: 0. for key in ("transition", "rowsum", "projection", "emission", "evidence", "posterior")}
    for event, (frame, velocity) in enumerate(zip(trace, velocities)):
        reading = B.Reading(frame["t"], frame["q"], velocity)
        previous = old.reading
        if previous is None:
            r1 = 0.
            independent = trans = np.eye(len(B.Z_AXIS))
        else:
            dt, v = reading.t - previous.t, .5 * (reading.velocity + previous.velocity)
            a = np.exp(-dt / B.LAG_TAU)
            r1 = a * r + B.VELOCITY_COUPLING * B.LAG_TAU * (1. - a) * v
            trans = old.transition(dt, v)
            independent = shifted_bin(B.Z_AXIS, r, r1, dt, B.LAG_TAU, B.PROCESS_SD, B.ndtr)
            checks["transition"] = max(checks["transition"], float(np.max(np.abs(trans - independent))))
            checks["rowsum"] = max(checks["rowsum"], float(np.max(np.abs(independent.sum(1) - 1.))))
        obs = np.asarray(frame["centroids_mm"], float).reshape(-1, 2)
        evidence = old.observe(world, reading, frame["site"], frame["view"], obs, event=event, retain=False)
        likelihood, _ = old.emission(world, reading, frame["site"], frame["view"], obs)
        pars = np.stack(np.meshgrid(old.w, old.b, B.Z_AXIS - r1, indexing="ij"), axis=-1).reshape(-1, len(B.Z_AXIS), 3)
        pars[..., 2] += r1
        residual_view = old.copy()
        residual_view.parameters = pars
        residual_likelihood, _ = residual_view.emission(world, reading, frame["site"], frame["view"], obs)
        residual_prior = residual_weights @ independent
        residual_evidence = float(np.sum(residual_prior * residual_likelihood))
        residual_weights = residual_prior * residual_likelihood / residual_evidence
        checks["emission"] = max(checks["emission"], float(np.max(np.abs(likelihood - residual_likelihood))))
        checks["evidence"] = max(checks["evidence"], abs(evidence - residual_evidence))
        checks["posterior"] = max(checks["posterior"], float(np.max(np.abs(old.weights - residual_weights))))
        base, design = world.projection(reading.q, frame["site"], frame["view"])
        old_centers = old.means(base, design, reading)
        residual_centers = base + np.einsum("wzk,ldk->wzld", pars, design)
        checks["projection"] = max(checks["projection"], float(np.max(np.abs(old_centers - residual_centers))))
        r = r1
        evs = [f.observe(world, reading, frame["site"], frame["view"], obs, event=event, retain=False, track=False) for f in gains]
        logw += np.log(evs)

    weights = np.exp(logw - np.max(logw))
    weights /= weights.sum()
    gain_mean = float(weights @ GAINS)
    gain_sd = float(np.sqrt(weights @ (GAINS - gain_mean) ** 2))
    old_metrics, adapted_metrics, physical = {}, {}, {"mean": 0., "sd": 0.}
    for horizon in HORIZONS:
        key = f"{horizon:.1f}"
        time = old.reading.t + horizon
        old_forecast = old.forecast(world, time, world.query_sites)
        adapted_forecast = fuse(
            [f.forecast(world, time, world.query_sites) for f in gains], weights
        )
        residual = residual_forecast(world, old, B, residual_weights, r, time, world.query_sites)
        physical["mean"] = max(physical["mean"], float(np.max(np.abs(old_forecast[0] - residual[0]))))
        physical["sd"] = max(physical["sd"], float(np.max(np.abs(old_forecast[1] - residual[1]))))
        old_metrics[key] = metric(world, case, time, *old_forecast)
        adapted_metrics[key] = metric(world, case, time, *adapted_forecast)
        saved = run["future"][0 if horizon == .5 else 1]
        saved_errors = {
            "saved_mean": np.max(np.abs(old_forecast[0] - np.asarray(saved["mean_xyz_mm"]))),
            "saved_sd": np.max(np.abs(old_forecast[1] - np.asarray(saved["sd_xyz_mm"]))),
            "saved_rmse": abs(old_metrics[key]["rmse_mm"] - saved["rmse_mm"]),
        }
        for name, value in saved_errors.items():
            checks[name] = max(checks.get(name, 0.), float(value))
    checks["physical_mean"], checks["physical_sd"] = physical["mean"], physical["sd"]
    return {
        "index": int(row["index"]),
        "seed": int(row["seed"]),
        "final_t": float(old.reading.t),
        "frames": len(trace),
        "trace_q_max_abs_error": float(qerr),
        "trace_centroids_max_abs_error": float(cerr),
        "true_gain_evaluator_only": float(case.coupling),
        "gain_posterior_mean": gain_mean,
        "gain_posterior_sd": gain_sd,
        "gain_posterior_weights": weights.tolist(),
        "old_fixed_0.83": old_metrics,
        "gain_adaptation": adapted_metrics,
        "paired_delta": {
            h: {k: adapted_metrics[h][k] - old_metrics[h][k]
                for k in ("rmse_mm", "coverage_90")}
            for h in old_metrics
        },
        "checks": checks,
    }


def aggregate(rows, method):
    out = {}
    for h in HORIZONS:
        key = f"{h:.1f}"
        vals = [r[method][key] for r in rows]
        out[key] = {
            "pooled_rmse_mm": float(np.sqrt(np.mean([v["rmse_mm"] ** 2 for v in vals]))),
            "mean_coverage_90": float(np.mean([v["coverage_90"] for v in vals])),
        }
    return out


def paired_bootstrap(rows):
    n = len(rows)
    rng = np.random.default_rng(84062)
    out = {}
    sample = rng.integers(0, n, size=(BOOT, n))
    for h in HORIZONS:
        key = f"{h:.1f}"
        old = np.array([r["old_fixed_0.83"][key]["rmse_mm"] for r in rows])
        new = np.array([r["gain_adaptation"][key]["rmse_mm"] for r in rows])
        oldc = np.array([r["old_fixed_0.83"][key]["coverage_90"] for r in rows])
        newc = np.array([r["gain_adaptation"][key]["coverage_90"] for r in rows])
        delta = np.sqrt(np.mean(new[sample] ** 2, 1)) - np.sqrt(np.mean(old[sample] ** 2, 1))
        dc = np.mean((newc - oldc)[sample], 1)
        out[key] = {
            "rmse_delta_mm": float(np.sqrt(np.mean(new ** 2)) - np.sqrt(np.mean(old ** 2))),
            "rmse_delta_bootstrap_ci95_mm": [float(x) for x in np.quantile(delta, [.025, .975])],
            "adaptation_rmse_wins": int(np.sum(new < old)),
            "adaptation_rmse_losses": int(np.sum(new > old)),
            "coverage_delta": float(np.mean(newc - oldc)),
            "coverage_delta_bootstrap_ci95": [float(x) for x in np.quantile(dc, [.025, .975])],
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--count", type=int, default=None)
    args = p.parse_args()
    archive = args.archive.resolve()
    result_dir = archive / "results"
    all_rows = read_rows(result_dir)
    limit = len(all_rows) if args.count is None else max(1, min(args.count, len(all_rows)))
    rows = all_rows[:limit]
    B, W = load_archive(archive)
    world = W.World()
    GainFilter = gain_filter_class(B)
    cases = [case_replay(row, world, B, GainFilter) for row in rows]
    checks = [r.pop("checks") for r in cases]
    maxes = {name: float(max(c[name] for c in checks)) for name in checks[0]}
    identity_ok = all(maxes[name] <= TOL for name in (
        "transition", "projection", "emission", "evidence", "posterior",
        "physical_mean", "physical_sd"))
    hashes = {
        f"results/{path.name}": digest(path)
        for path in sorted(result_dir.glob("test_[0-9]*.jsonl"))
    }
    for name in ("belief.py", "knee_world.py", "estimator_replay.py"):
        if (archive / name).is_file():
            hashes[name] = digest(archive / name)
    if Path(world.geometry_path).is_file():
        key = f"geometry/{Path(world.geometry_path).name}"
        hashes[key] = digest(world.geometry_path)
    true = np.array([r["true_gain_evaluator_only"] for r in cases])
    means = np.array([r["gain_posterior_mean"] for r in cases])
    sds = np.array([r["gain_posterior_sd"] for r in cases])
    weights = np.array([r["gain_posterior_weights"] for r in cases])
    prior_mean, prior_sd = float(GAINS.mean()), float(np.sqrt(np.mean((GAINS - GAINS.mean()) ** 2)))
    report = {
        "schema": "response_replay_v1",
        "study_label": "archival replay; exploratory; no train split or tuning",
        "source_archive": str(archive),
        "cases_available": len(all_rows),
        "cases_processed": len(cases),
        "gains": GAINS.tolist(),
        "gain_prior": "uniform",
        "time_constant_s": float(B.LAG_TAU),
        "old_fixed_gain": float(B.VELOCITY_COUPLING),
        "horizons_s": list(HORIZONS),
        "identity_decomposition": {
            "equation": "s = r + z",
            "forced_response": "r[k+1]=a*r[k]+g0*tau*(1-a)*mean(measured qdot), r[0]=0",
            "residual_support": "old_total_grid-r[k]",
            "residual_transition": "a*z plus the original noise",
            "old_total_grid": B.Z_AXIS.tolist(),
            "g0": float(B.VELOCITY_COUPLING),
        },
        "gain_adaptation": {
            "conditional_filter": "independent old Filter subclass; only transition coupling changes",
            "evidence": "log-space update by each observe return value",
            "forecast": "total expectation and variance",
            "true_gain": "case.coupling evaluator-only; no policy/training use",
        },
        "input_sha256": hashes,
        "checks": {
            **maxes,
            "trace_q_centroids_exact": all(
                r["trace_q_max_abs_error"] <= 2.e-10
                and r["trace_centroids_max_abs_error"] <= 2.e-10 for r in cases
            ),
            "identity_equivalence_within_numeric_tolerance": identity_ok,
            "old_saved_output_exact": (
                maxes["saved_mean"] <= TOL
                and maxes["saved_sd"] <= TOL
                and maxes["saved_rmse"] <= TOL
            ),
        },
        "aggregate": {
            "old_fixed_0.83": aggregate(cases, "old_fixed_0.83"),
            "gain_adaptation": aggregate(cases, "gain_adaptation"),
            "paired_bootstrap": paired_bootstrap(cases),
        },
        "gain_diagnostics_evaluator_only": {
            "prior_mean": prior_mean,
            "prior_sd": prior_sd,
            "posterior_mean_rmse_vs_true": float(np.sqrt(np.mean((means - true) ** 2))),
            "posterior_sd_mean": float(sds.mean()),
            "posterior_sd_minus_prior_sd": float(sds.mean() - prior_sd),
            "posterior_effective_count_mean": float(np.mean(1. / np.sum(weights ** 2, 1))),
        },
        "reproduction": (
            "PYTHONDONTWRITEBYTECODE=1 /media/camp/EXT_DRIVE/envs/genesis/bin/python "
            "response_replay.py --archive "
            "/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/"
            "ab_knee_latent_rl_20261112 --out verification/RESPONSE.json [--count N]"
        ),
        "caveats": [
            "Archival exploratory replay on traces already used to study old z; not a learned claim.",
            "Five gains are a finite mixture, not continuous identification.",
            "No future readings in policy and no motion changes; true coupling is evaluator-only.",
        ],
        "per_case": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "cases": len(cases),
        "checks": report["checks"],
        "paired_bootstrap": report["aggregate"]["paired_bootstrap"],
    }, indent=2))


if __name__ == "__main__":
    main()
