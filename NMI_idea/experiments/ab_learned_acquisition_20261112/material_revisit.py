"""Isolate spatial model capacity and online adaptation on registered revisits."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from acquisition import Candidate, World


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "material"
SITES = (6, 9, 12, 15, 18, 21)
TRAIN_COUNT = 256
TRAIN_SMALL = 32
TEST_COUNT = 128
TAU = .72
GAIN = .83
MEAS_SD = .48
MODELS = ("affine_32", "affine_256", "field_256_frozen", "field_256_online", "field_broad_online")
SPLITS = {"nominal": 2026111221, "local_deformation": 2026111222}
TRAIN_SEED = 2026111220


@dataclass(frozen=True)
class Observations:
    times: np.ndarray
    sites: np.ndarray
    velocity: np.ndarray
    design: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class Truth:
    static_mm: np.ndarray
    transient_mm: np.ndarray
    q: np.ndarray


def acquisition_clock(world):
    local = np.arange(len(SITES))
    visits = np.concatenate((local, local[::-1], local))
    times = [.21]
    for a, b in zip(visits[:-1], visits[1:]):
        prior = (SITES[a], world.scan.central)
        goal = (SITES[b], world.scan.central)
        if not world.edge_valid(prior, goal):
            raise RuntimeError(f"modeled revisit edge unavailable {prior}, {goal}")
        previous = Candidate(*prior, 0., 0., times[-1], "previous")
        duration = max(world.make_action(q, previous, goal, "revisit").duration
                       for q in world.knot_angles)
        times.append(times[-1] + duration)
    return np.asarray(times), visits


def simulate(world, clock, arc_mm, seed, mode):
    rng = np.random.default_rng(seed)
    times, sites = clock
    case = world.case(seed)
    # Exact registered geometry is shared by all estimators in this isolation test.
    # Change internal spatial bias, independent of the camera or motion model.
    length = rng.uniform(55., 125.)
    cov = 1.1**2 * np.exp(-np.abs(arc_mm[:, None] - arc_mm) / length)
    stable = rng.normal(0., 1.65) + rng.multivariate_normal(np.zeros(len(SITES)), cov)
    if mode == "local_deformation":
        center = rng.uniform(arc_mm[1], arc_mm[-2])
        width = rng.uniform(20., 38.)
        amplitude = rng.choice([-1., 1.]) * rng.uniform(2.4, 4.2)
        stable += amplitude * np.exp(-.5 * ((arc_mm - center) / width) ** 2)
    q, v = case.true_angle_rate(times)
    z = np.array([case.z_mm(t) for t in times])
    design = []
    for angle, site in zip(q, sites):
        _, rows = world.projection(angle, SITES[site], world.scan.central)
        design.append(rows[0, 0, [0, 2]])
    design = np.array(design)
    residual = design[:, 0] * stable[sites] + design[:, 1] * z + rng.normal(0., MEAS_SD, len(times))
    return Observations(times, sites, v, design, residual), Truth(stable, z, q)


def lag_terms(obs):
    a = np.exp(-np.diff(obs.times, prepend=obs.times[0]) / TAU)
    v = np.r_[obs.velocity[0], .5 * (obs.velocity[1:] + obs.velocity[:-1])]
    offset = GAIN * TAU * (1. - a) * v
    return a, offset


def fit_prior(records, basis):
    estimates = []
    for obs in records:
        a, drive = lag_terms(obs)
        forced = np.zeros(len(a))
        for i in range(1, len(a)):
            forced[i] = a[i] * forced[i - 1] + drive[i]
        h = np.column_stack((obs.design[:, :1] * basis[obs.sites],
                             obs.design[:, 1] * np.cumprod(a)))
        target = obs.residual - obs.design[:, 1] * forced
        fitted = np.linalg.solve(h.T @ h + .07 * np.eye(h.shape[1]), h.T @ target)
        estimates.append(fitted[:-1])
    values = np.array(estimates)
    cov = np.atleast_2d(np.cov(values.T))
    cov = .9 * cov + .1 * np.diag(np.diag(cov)) + .25**2 * np.eye(cov.shape[0])
    return values.mean(axis=0), cov


def filter_record(obs, basis, prior, *, frozen=False):
    mean = np.r_[prior[0], 0.]
    cov = np.zeros((len(mean), len(mean)))
    cov[:-1, :-1] = prior[1]
    cov[-1, -1] = 1.8**2
    a, drive = lag_terms(obs)
    predictions, estimates, sd = [], [], []
    for i, (site, row, value) in enumerate(zip(obs.sites, obs.design, obs.residual)):
        mean[-1] = a[i] * mean[-1] + drive[i]
        cov[-1, :] *= a[i]
        cov[:, -1] *= a[i]
        cov[-1, -1] += 1. * (1. - a[i]**2)
        h = np.r_[row[0] * basis[site], row[1]]
        predictions.append(float(h @ mean))
        projection = cov @ h
        variance = float(h @ projection + MEAS_SD**2)
        sd.append(float(np.sqrt(variance)))
        gain = projection / variance
        if frozen:
            # Keep static uncertainty and its cross-covariance while freezing its mean.
            gain[:-1] = 0.
        mean += gain * (value - predictions[-1])
        cov += (variance * np.outer(gain, gain) - np.outer(gain, projection)
                - np.outer(projection, gain))
        estimates.append(mean.copy())
    return np.array(predictions), np.array(sd), np.array(estimates)


def case_score(obs, truth, predictions, sd, estimates, basis):
    target = obs.design[:, 0] * truth.static_mm[obs.sites] + obs.design[:, 1] * truth.transient_mm
    sq = (predictions - target)**2
    cover = np.abs(predictions - obs.residual) <= 1.645 * sd
    after = basis @ estimates[-1, :-1]
    return {
        "first_sq_mm2": float(np.mean(sq[:len(SITES)])),
        "revisit_sq_mm2": float(np.mean(sq[len(SITES):])),
        "revisit_90_predictive_coverage": float(np.mean(cover[len(SITES):])),
        "final_static_sq_mm2": float(np.mean((after - truth.static_mm)**2)),
        "final_transient_error_mm": float(estimates[-1, -1] - truth.transient_mm[-1]),
    }


def summarize(rows):
    rng = np.random.default_rng(2026111223)
    report = {}
    for mode in SPLITS:
        cases = [r["arms"] for r in rows if r["mode"] == mode]
        ids = rng.integers(0, len(cases), size=(4000, len(cases)))
        arms, paired = {}, {}
        for model in MODELS:
            arms[model] = {
                f"{key}_rmse_mm": float(np.sqrt(np.mean([r[model][f"{key}_sq_mm2"] for r in cases])))
                for key in ("first", "revisit", "final_static")
            }
            arms[model]["revisit_90_predictive_coverage"] = float(
                np.mean([r[model]["revisit_90_predictive_coverage"] for r in cases]))
        for a, b in (("affine_256", "affine_32"), ("field_256_online", "affine_256"),
                     ("field_256_online", "field_256_frozen"), ("field_256_online", "field_broad_online")):
            left = np.array([r[a]["revisit_sq_mm2"] for r in cases])
            right = np.array([r[b]["revisit_sq_mm2"] for r in cases])
            delta = np.sqrt(left[ids].mean(axis=1)) - np.sqrt(right[ids].mean(axis=1))
            paired[f"{a}_minus_{b}"] = {
                "delta_rmse_mm": float(np.sqrt(left.mean()) - np.sqrt(right.mean())),
                "ci95": np.quantile(delta, [.025, .975]).tolist(),
            }
        report[mode] = {"cases": len(cases), "arms": arms, "paired_revisit": paired}
    return report


def plot(world, example):
    import plotly.graph_objects as go

    truth, fits = example
    q = truth.q[-1]
    xyz, design = world.world_design(q, [(site, 0) for site in SITES])
    fig = go.Figure()
    curves = {
        "Simulated target": xyz + design[:, :, 0] * truth.static_mm[:, None]
        + design[:, :, 2] * truth.transient_mm[-1],
        "Coarse atlas": xyz,
    }
    for label, name in (("Affine calibration", "affine_256"), ("Per-site calibration", "field_256_online")):
        basis, state = fits[name]
        curves[label] = xyz + design[:, :, 0] * (basis @ state[:-1])[:, None] + design[:, :, 2] * state[-1]
    for (label, curve), color in zip(curves.items(), ("#23313d", "#a7b5be", "#b37243", "#217e87")):
        fig.add_trace(go.Scatter3d(x=curve[:, 0], y=curve[:, 1], z=curve[:, 2], mode="lines+markers",
                                  name=label, line=dict(color=color, width=6), marker=dict(size=4)))
    fig.update_layout(title="Registered vessel estimates after three sweeps", template="plotly_white",
                      height=650, scene=dict(aspectmode="data", xaxis_title="X (mm)",
                                             yaxis_title="Y (mm)", zaxis_title="Z (mm)"))
    fig.write_html(OUTPUT / "MATERIAL_REVISIT_3D.html", include_plotlyjs=True)


def main():
    OUTPUT.mkdir(exist_ok=True)
    world = World()
    clock = acquisition_clock(world)
    arc_mm = world.scan.s[list(SITES)]
    direction = 2. * (arc_mm - np.min(arc_mm)) / np.ptp(arc_mm) - 1.
    affine = np.column_stack((np.ones(len(SITES)), direction))
    field = np.eye(len(SITES))
    broad = 2.8**2 * np.exp(-np.abs(arc_mm[:, None] - arc_mm) / 75.) + .6**2 * field
    train = [simulate(world, clock, arc_mm, TRAIN_SEED + i, "nominal")[0] for i in range(TRAIN_COUNT)]
    fit_field = fit_prior(train, field)
    configurations = {
        "affine_32": (affine, fit_prior(train[:TRAIN_SMALL], affine), False),
        "affine_256": (affine, fit_prior(train, affine), False),
        "field_256_frozen": (field, fit_field, True),
        "field_256_online": (field, fit_field, False),
        "field_broad_online": (field, (np.zeros(len(SITES)), broad), False),
    }
    rows = []
    example = None
    for mode, seed in SPLITS.items():
        for i in range(TEST_COUNT):
            obs, truth = simulate(world, clock, arc_mm, seed * 1000 + i, mode)
            arms, estimates = {}, {}
            for model, (basis, prior, frozen) in configurations.items():
                prediction, sd, states = filter_record(obs, basis, prior, frozen=frozen)
                arms[model] = case_score(obs, truth, prediction, sd, states, basis)
                estimates[model] = basis, states[-1]
            if mode == "local_deformation" and i == 0:
                example = truth, estimates
            rows.append({"mode": mode, "index": i, "arms": arms})
    result = {
        "scope": "Synthetic internal lateral deformation under shared ideal registered pose and visible masks.",
        "train": {"cases": TRAIN_COUNT, "nested_small": TRAIN_SMALL, "seed": TRAIN_SEED,
                  "fit_uses": "Registered mask residuals and measured pose input only."},
        "test": {"cases_per_condition": TEST_COUNT, "seed_bases": SPLITS},
        "sites": SITES, "capture_times_s": clock[0].tolist(),
        "capture_station_index": clock[1].tolist(),
        "summary": summarize(rows), "case_rows": rows,
    }
    (OUTPUT / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(world, example)


if __name__ == "__main__":
    main()
