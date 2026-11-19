"""Calibrated static clutter model for the B validation experiment.

The public runtime API is intentionally small.

``render_observation(world, case, t, site, view)``
    Render one deterministic stress frame and apply the same nonzero
    localization jitter to a persistent ghost in every split.

``FittedMapFilter(calibration=...)``
    A :class:`MapFilter` drop-in whose transition, prior, dynamic state and
    identity likelihood stay unchanged while the false-component chance and
    spatial density use the fitted static model.

``simulated_masks(world, belief, reading, site, view, uniforms)``
    Draw unlabeled predictive masks from the fitted false-component model.

``fit_calibration(...)``
    Fit identity-site and single-lumen clutter parameters on independent
    simulated seeds, evaluate a held-apart validation group, and optionally
    write the lineage/calibration JSON.

The fitting code uses only NumPy/SciPy.  Truth labels are used by fitting and
the independent evaluator only.  Runtime filtering and predictive sampling
consume public centroids, pose packets and the fitted parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import lzma
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.special import ndtr, ndtri


HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "RealUS_playground"
EXPERIMENTS = REPO / "NMI_idea" / "experiments"
MAP_DIR = EXPERIMENTS / "ab_map_recheck_20261112"
KNEE_DIR = EXPERIMENTS / "ab_knee_latent_rl_20261112"
DYNAMIC_DIR = EXPERIMENTS / "ab_leg_dynamic_20261112"
ASSET_DIR = EXPERIMENTS / "ab_tro_y_continuation_20261112" / "review_inputs"
ARCHIVE_ROOT = Path("/media/camp/EXT_DRIVE/RealUS_playground")
ARCHIVE_GEOMETRY = ARCHIVE_ROOT / "NMI_idea" / "experiments" / "ab_leg_dynamic_20261112" / "DYNAMIC_GEOMETRY_FROZEN.npz"

for _path in (str(MAP_DIR), str(KNEE_DIR), str(DYNAMIC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from belief import (  # noqa: E402
    DEPTH,
    HALF_WIDTH,
    IMAGE_SD,
    NEAR_CLUTTER_RATE,
    NEAR_CLUTTER_SD,
    OUTLIER_RATE,
    OUTLIER_SD,
    Reading,
    assignments,
    in_view_probability,
    norm_pdf,
)
from map_model import MapFilter, MapWorld  # noqa: E402


LOW = np.array([-HALF_WIDTH, DEPTH[0]], dtype=float)
HIGH = np.array([HALF_WIDTH, DEPTH[1]], dtype=float)
UNIFORM_DENSITY = 1.0 / float(np.prod(HIGH - LOW))
GHOST_JITTER_SD_MM = np.array([1.0, 0.68], dtype=float)
TRAIN_SEED_BASE = 2026111311
VALIDATION_SEED_BASE = 2026111312
DEFAULT_TRAIN_CASES = 96
DEFAULT_VALIDATION_CASES = 32


def _seed(base: int, index: int) -> int:
    return int(np.random.SeedSequence([int(base), int(index)]).generate_state(1)[0])


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_route() -> tuple[tuple[int, int], ...] | None:
    """Read the frozen route if the portable review record is present."""

    common = EXPERIMENTS / "ab_tro_y_continuation_20261112" / "holdout" / "COMMON.jsonl.xz"
    if not common.is_file():
        return None
    with lzma.open(common, "rt") as stream:
        row = json.loads(next(stream))
    return tuple((int(frame["site"]), int(frame["view"])) for frame in row["prefix_frames"])


def _portable_branch() -> tuple[np.ndarray, ...]:
    path = ASSET_DIR / "POSED_BRANCH.npz"
    if not path.is_file():
        raise FileNotFoundError(f"portable branch snapshot not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["dense"]),
            np.asarray(data["s"]),
            np.asarray(data["daughter"]),
            tuple(map(int, data["sites"])),
            int(data["fork"]),
        )


def build_portable_world(*, mode: str = "stress"):
    """Construct the frozen MapWorld using the portable posed-branch asset.

    This helper avoids the optional robot capability bridge.  It is only used
    for acoustic fitting and validation.  Candidate legality in the benchmark
    remains owned by the benchmark's frozen preflight graph.
    """

    import dynamic_scan
    import knee_world

    class AcousticEdges:
        robot = None

        def __init__(self, scan):
            self.q = scan.q

        def allow(self, start, goal):
            return True

    posed = _portable_branch()
    original_previous = getattr(dynamic_scan, "PREVIOUS", None)
    original_edges = knee_world.Edges
    dynamic_scan.PREVIOUS = ARCHIVE_ROOT / "NMI_idea" / "experiments" / "ab_leg_bifurcation_20261112"
    knee_world.Edges = AcousticEdges
    try:
        world = MapWorld(geometry=ARCHIVE_GEOMETRY, mode=mode)
    finally:
        dynamic_scan.PREVIOUS = original_previous
        knee_world.Edges = original_edges
    route = _canonical_route()
    if route is not None:
        world.initial_route = route
        world.nominal_fork_arrival_s, world.nominal_route_end_s = world._fixed_route_times()
    return world


def _group(world, site: int) -> str:
    return "identity" if bool(world.identity_site(int(site))) else "single"


def _model_table(calibration: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if calibration is None:
        return None
    if "models" in calibration:
        calibration = calibration["models"]
    if "identity" in calibration and "single" in calibration:
        return calibration
    return None


def _params(calibration: Mapping[str, Any] | None, group: str) -> dict[str, Any] | None:
    table = _model_table(calibration)
    if table is None:
        return None
    model = dict(table[group])
    model.setdefault("chance", model.get("observed_clutter_probability", 0.21 if group == "identity" else 0.12))
    model.setdefault("near_weight", model.get("near_lumen_weight", NEAR_CLUTTER_RATE))
    model.setdefault("uniform_weight", 1.0 - float(model["near_weight"]))
    sd = model.get("sd_mm", model.get("near_sd_mm", NEAR_CLUTTER_SD))
    model["sd_mm"] = np.asarray(sd, dtype=float).reshape(2)
    model["chance"] = float(np.clip(model["chance"], 1.e-8, 1. - 1.e-8))
    model["near_weight"] = float(np.clip(model["near_weight"], 1.e-8, 1. - 1.e-8))
    model["uniform_weight"] = 1.0 - model["near_weight"]
    return model


def _component_density(obs: np.ndarray, centers: np.ndarray, sd: np.ndarray, near_weight: float) -> np.ndarray:
    obs = np.asarray(obs, dtype=float).reshape(-1, 2)
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    sd = np.asarray(sd, dtype=float).reshape(2)
    if not len(obs):
        return np.zeros(0, dtype=float)
    near = np.prod(norm_pdf(obs[:, None, :] - centers[None, :, :], sd), axis=-1)
    normalizer = np.maximum(in_view_probability(centers, sd), 1.e-12)
    near = np.mean(near / normalizer[None, :], axis=1)
    return (1.0 - float(near_weight)) * UNIFORM_DENSITY + float(near_weight) * near


def clutter_density(obs: np.ndarray, base: np.ndarray, model: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return the fitted or nominal false-centroid density for each point."""

    if model is None:
        model = {"sd_mm": NEAR_CLUTTER_SD, "near_weight": NEAR_CLUTTER_RATE}
    return _component_density(obs, base, np.asarray(model["sd_mm"], dtype=float), float(model["near_weight"]))


def render_observation(world, case, t: float, site: int, view: int):
    """Render a deterministic frame with nonzero persistent-ghost jitter.

    The base renderer keeps the physical-time dropout and all case-local
    stress variables fixed.  When the deterministic persistent ghost is
    present, its projected centroid receives a reproducible Gaussian
    localization perturbation in both training and challenge use.
    """

    frame = world.observe(case, float(t), int(site), int(view))
    q, _ = case.true_angle_rate(float(t))
    base, _ = world.projection(q, int(site), int(view))
    persistent = world._persistent_clutter(case, q, int(site), int(view), base)
    if persistent is None or -1 not in frame.labels:
        return frame
    tick = round(float(t) * 1000)
    rng = np.random.default_rng(np.random.SeedSequence([int(case.seed), 77631, tick, int(site), int(view)]))
    jitter = rng.normal(0.0, GHOST_JITTER_SD_MM, size=2)
    values = np.asarray(frame.centroids, dtype=float).copy()
    ordinal = int(frame.labels.index(-1))
    values[ordinal] = np.clip(values[ordinal] + jitter, LOW, HIGH)
    order = np.argsort(values[:, 0], kind="stable")
    return replace(frame, centroids=values[order], labels=tuple(frame.labels[index] for index in order))


class FittedMapFilter(MapFilter):
    """MapFilter with a calibrated static false-component likelihood."""

    def __init__(self, calibration: Mapping[str, Any] | None = None, *, latent: bool = True):
        super().__init__(latent=latent)
        self.calibration = calibration

    @classmethod
    def from_json(cls, path: str | Path, *, latent: bool = True):
        return cls(json.loads(Path(path).read_text()), latent=latent)

    def clutter_model(self, world, site: int, view: int) -> dict[str, Any]:
        group = _group(world, site)
        model = _params(self.calibration, group)
        if model is not None:
            return model
        return {"chance": float(world.clutter_probability(site, view)), "near_weight": NEAR_CLUTTER_RATE,
                "uniform_weight": 1.0 - NEAR_CLUTTER_RATE, "sd_mm": np.asarray(NEAR_CLUTTER_SD, dtype=float)}

    def emission(self, world, reading, site, view, obs):
        model = self.clutter_model(world, site, view)
        # Keep nominal behavior exactly when no fit is attached.
        if self.calibration is None:
            return super().emission(world, reading, site, view, obs)
        centers, hit = self.imaging(world, reading, site, view)
        obs = np.asarray(obs, dtype=float).reshape(-1, 2)
        count = len(obs)
        n_lumens = centers.shape[-2]
        errors = obs[None, None, None, :, :] - centers[..., None, :]
        ordinary = np.prod(norm_pdf(errors, IMAGE_SD), axis=-1)
        broad = np.prod(norm_pdf(errors, OUTLIER_SD), axis=-1)
        observed = world.detection_probability(reading.q, site, view) * (
            (1.0 - OUTLIER_RATE) * ordinary + OUTLIER_RATE * broad
        )
        base, _ = world.projection(reading.q, site, view)
        false_density = clutter_density(obs, base, model)
        likelihood = np.zeros_like(self.weights)
        responsibilities = np.zeros((count, 3, *self.weights.shape))
        for assigned, unused in assignments(n_lumens, count):
            if len(unused) > 1:
                continue
            term = np.ones_like(self.weights)
            for label, index in enumerate(assigned):
                term *= observed[..., label, index] if index >= 0 else 1.0 - hit[..., label]
            if unused:
                term *= float(model["chance"]) * false_density[unused[0]]
            else:
                term *= 1.0 - float(model["chance"])
            likelihood += term
            for label, index in enumerate(assigned):
                if index >= 0:
                    responsibilities[index, label] += term
            for index in unused:
                responsibilities[index, 2] += term
        return likelihood, responsibilities


def simulated_masks(world, belief, reading, site: int, view: int, uniforms: np.ndarray):
    """Sample unlabeled masks using the fitted false-component density.

    ``uniforms`` follows the existing 16-column Sobol layout.  Columns 0--9
    retain the original latent/detection/noise convention.  Column 10 chooses
    uniform versus near-lumen clutter, columns 11--14 sample the corresponding
    point and column 15 chooses the near-lumen component for a multi-lumen
    site.
    """

    uniforms = np.asarray(uniforms, dtype=float)
    if uniforms.ndim != 2 or uniforms.shape[1] < 15:
        raise ValueError("uniforms must have shape (N, >=15)")
    group = _group(world, site)
    calibration = getattr(belief, "calibration", None)
    if calibration is None:
        calibration = getattr(world, "clutter_calibration", None)
    model = _params(calibration, group)
    if model is None:
        model = {"chance": float(world.clutter_probability(site, view)), "near_weight": NEAR_CLUTTER_RATE,
                 "sd_mm": np.asarray(NEAR_CLUTTER_SD, dtype=float)}
    prior, _ = belief.prior_at(reading)
    base, rows = world.projection(reading.q, site, view)
    centers = belief.means(base, rows, reading).reshape(-1, len(base), 2)
    cumulative = np.cumsum(prior.ravel())
    cumulative[-1] = 1.0
    sampled = centers[np.searchsorted(cumulative, uniforms[:, 0], side="right")]
    n_lumens = len(base)
    sigma_flag = uniforms[:, 3:3 + n_lumens, None] < OUTLIER_RATE
    sigma = np.where(sigma_flag, OUTLIER_SD, IMAGE_SD)
    normal_values = ndtri(np.clip(uniforms[:, 5:5 + 2 * n_lumens], 1.e-8, 1.0 - 1.e-8))
    normal_values = normal_values.reshape(-1, n_lumens, 2)
    values = sampled + normal_values * sigma
    visible = ((uniforms[:, 1:1 + n_lumens] < world.detection_probability(reading.q, site, view))
               & np.all((values >= LOW) & (values <= HIGH), axis=2))
    sd = np.asarray(model["sd_mm"], dtype=float)
    near_uniforms = np.clip(uniforms[:, 11:13], 1.e-8, 1.0 - 1.e-8)
    uniform_points = LOW + uniforms[:, 13:15] * (HIGH - LOW)
    if n_lumens:
        component = np.minimum((uniforms[:, 15] * n_lumens).astype(int), n_lumens - 1) if uniforms.shape[1] > 15 else np.zeros(len(uniforms), dtype=int)
        chosen_centers = base[component]
    else:
        chosen_centers = np.zeros((len(uniforms), 2))
    lower = ndtr((LOW[None, :] - chosen_centers) / sd[None, :])
    upper = ndtr((HIGH[None, :] - chosen_centers) / sd[None, :])
    near_points = chosen_centers + sd[None, :] * ndtri(lower + near_uniforms * (upper - lower))
    clutter_point = np.where((uniforms[:, 10] < float(model["near_weight"]))[:, None], near_points, uniform_points)
    false_present = uniforms[:, 9] < float(model["chance"])
    result = []
    for index in range(len(uniforms)):
        points = values[index][visible[index]]
        if false_present[index]:
            points = np.vstack((points, clutter_point[index]))
        result.append(points[np.argsort(points[:, 0], kind="stable")])
    return tuple(result)


def _fit_group(points: np.ndarray, centers: np.ndarray, frame_count: int, clutter_frames: int) -> dict[str, Any]:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    if len(points):
        selected = centers
        residual = points - selected
    else:
        residual = np.zeros((0, 2), dtype=float)
        selected = np.zeros((0, 2), dtype=float)
    if len(residual):
        sd = np.array([3.0, 1.5], dtype=float)
        weight = 0.5
        for _ in range(48):
            nearby = np.prod(norm_pdf(residual, sd), axis=1)
            normalizer = np.maximum(in_view_probability(selected, sd), 1.e-12)
            nearby /= normalizer
            uniform = np.full(len(residual), UNIFORM_DENSITY)
            posterior = weight * nearby / np.maximum(weight * nearby + (1.0 - weight) * uniform, 1.e-300)
            weight = float(np.clip((posterior.sum() + 1.0) / (len(residual) + 2.0), .02, .98))
            mass = max(float(posterior.sum()), 1.e-9)
            sd = np.sqrt(np.maximum(np.sum(posterior[:, None] * residual ** 2, axis=0) / mass, .25 ** 2))
            sd = np.clip(sd, .25, 12.0)
    else:
        sd = np.asarray(NEAR_CLUTTER_SD, dtype=float)
        weight = float(NEAR_CLUTTER_RATE)
    chance = float((clutter_frames + 1.0) / (frame_count + 2.0))
    return {
        "chance": chance,
        "observed_clutter_probability": chance,
        "near_weight": float(weight),
        "near_lumen_weight": float(weight),
        "uniform_weight": float(1.0 - weight),
        "sd_mm": sd.tolist(),
        "near_sd_mm": sd.tolist(),
        "frames": int(frame_count),
        "clutter_frames": int(clutter_frames),
        "clutter_points": int(len(points)),
    }


def _association_probability(point, centers, chance, model):
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    point = np.asarray(point, dtype=float).reshape(1, 2)
    nearest = centers[np.argmin(np.linalg.norm(centers - point, axis=1))]
    false = float(chance) * float(clutter_density(point, centers, model)[0])
    target = float((1.0 - OUTLIER_RATE) * np.prod(norm_pdf(point[0] - nearest, IMAGE_SD))
                   + OUTLIER_RATE * np.prod(norm_pdf(point[0] - nearest, OUTLIER_SD)))
    return false / max(false + target, 1.e-300)


def _validation_metrics(world, cases, calibration):
    frame_rows = []
    point_rows = []
    identity_sites = tuple(map(int, world.branch_sites))
    single_sites = tuple(site for site in range(world.n) if site not in identity_sites)
    sites = (*identity_sites, *single_sites[: max(3, len(identity_sites))])
    views_by_site = {site: tuple(int(v) for v in world.view_bank[site][:3]) for site in sites}
    for case in cases:
        for repeat, offset in enumerate((.25, .55, .95, 1.35)):
            t = float(world.nominal_route_end_s + offset + .37 * repeat)
            for site in sites:
                for view in views_by_site[site]:
                    frame = render_observation(world, case, t, site, view)
                    group = _group(world, site)
                    fitted = _params(calibration, group)
                    y = int(-1 in frame.labels)
                    p_fit = float(fitted["chance"])
                    p_nom = float(world.clutter_probability(site, view))
                    frame_rows.append((group, y, p_fit, p_nom))
                    base, _ = world.projection(frame.q, site, view)
                    for label, point in zip(frame.labels, frame.centroids):
                        y_point = int(label == -1)
                        p_fit_point = _association_probability(point, base, p_fit, fitted)
                        nominal = {"sd_mm": np.asarray(NEAR_CLUTTER_SD), "near_weight": NEAR_CLUTTER_RATE}
                        p_nom_point = _association_probability(point, base, p_nom, nominal)
                        point_rows.append((group, y_point, p_fit_point, p_nom_point))

    def brier(rows, column):
        return float(np.mean([(row[column] - row[1]) ** 2 for row in rows])) if rows else None

    def logloss(rows, column):
        return float(np.mean([-row[1] * np.log(np.clip(row[column], 1.e-9, 1. - 1.e-9))
                              - (1 - row[1]) * np.log(np.clip(1. - row[column], 1.e-9, 1. - 1.e-9)) for row in rows])) if rows else None

    def summarize_rows(rows):
        result = {"n": len(rows), "brier_fitted": brier(rows, 2), "brier_nominal": brier(rows, 3),
                  "logloss_fitted": logloss(rows, 2), "logloss_nominal": logloss(rows, 3)}
        for group in ("identity", "single"):
            sub = [row for row in rows if row[0] == group]
            result[group] = {"n": len(sub), "brier_fitted": brier(sub, 2), "brier_nominal": brier(sub, 3),
                             "logloss_fitted": logloss(sub, 2), "logloss_nominal": logloss(sub, 3)}
        return result

    return {"frame_clutter_prediction": summarize_rows(frame_rows),
            "point_clutter_association": summarize_rows(point_rows),
            "frames": len(frame_rows), "points": len(point_rows)}


def fit_calibration(world=None, *, train_cases: int = DEFAULT_TRAIN_CASES,
                    validation_cases: int = DEFAULT_VALIDATION_CASES,
                    train_seed_base: int = TRAIN_SEED_BASE,
                    validation_seed_base: int = VALIDATION_SEED_BASE,
                    output: str | Path | None = None):
    """Fit and validate static clutter parameters on independent seeds."""

    world = build_portable_world() if world is None else world
    identity_sites = tuple(map(int, world.branch_sites))
    single_sites = tuple(site for site in range(world.n) if site not in identity_sites)
    sites = (*identity_sites, *single_sites[: max(3, len(identity_sites))])
    views_by_site = {site: tuple(int(v) for v in world.view_bank[site][:3]) for site in sites}
    records = {"identity": [], "single": []}
    seeds = [_seed(train_seed_base, index) for index in range(train_cases)]
    for seed in seeds:
        case = world.case(seed)
        for repeat, offset in enumerate((.25, .55, .95, 1.35)):
            t = float(world.nominal_route_end_s + offset + .37 * repeat)
            for site in sites:
                group = _group(world, site)
                for view in views_by_site[site]:
                    frame = render_observation(world, case, t, site, view)
                    base, _ = world.projection(frame.q, site, view)
                    rows = records[group]
                    rows.append({"clutter": bool(-1 in frame.labels), "points": [
                        (np.asarray(point, dtype=float), np.asarray(base, dtype=float))
                        for label, point in zip(frame.labels, frame.centroids) if label == -1
                    ]})

    models = {}
    counts = {}
    for group, rows in records.items():
        points = np.array([point for row in rows for point, _ in row["points"]], dtype=float).reshape(-1, 2)
        centers = np.array([base[np.argmin(np.linalg.norm(base - point, axis=1))]
                            for row in rows for point, base in row["points"]], dtype=float).reshape(-1, 2)
        model = _fit_group(points, centers, len(rows), sum(row["clutter"] for row in rows))
        models[group] = model
        counts[group] = {"frames": len(rows), "clutter_frames": sum(row["clutter"] for row in rows),
                         "clutter_points": len(points)}

    validation_seeds = [_seed(validation_seed_base, index) for index in range(validation_cases)]
    validation_cases_data = [world.case(seed) for seed in validation_seeds]
    calibration = {
        "schema": "b_static_clutter_fit_v1",
        "models": models,
        "jitter_sd_mm": GHOST_JITTER_SD_MM.tolist(),
        "train": {"count": train_cases, "seed_base": train_seed_base, "seeds": seeds, "counts": counts},
        "validation": {"count": validation_cases, "seed_base": validation_seed_base, "seeds": validation_seeds},
        "source": {
            "map_model_sha256": _sha256(MAP_DIR / "map_model.py"),
            "belief_sha256": _sha256(KNEE_DIR / "belief.py"),
            "portable_branch_sha256": _sha256(ASSET_DIR / "POSED_BRANCH.npz"),
            "geometry_sha256": _sha256(ARCHIVE_GEOMETRY),
            "ghost_rule": "MapWorld persistent ghost plus deterministic Gaussian localization jitter",
        },
        "validation_metrics": _validation_metrics(world, validation_cases_data, models),
    }
    if output is not None:
        Path(output).write_text(json.dumps(calibration, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return calibration


__all__ = [
    "FittedMapFilter",
    "GHOST_JITTER_SD_MM",
    "build_portable_world",
    "clutter_density",
    "fit_calibration",
    "render_observation",
    "simulated_masks",
]
