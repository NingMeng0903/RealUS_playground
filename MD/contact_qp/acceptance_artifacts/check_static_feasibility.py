"""Read-only design-seed feasibility audit; never executes a controller or hardware.

Writes only this MD artifact directory. This is a static existence calculation,
not a controller success result, dynamic certificate, or held-out evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from peirastic.contact_qp.plant import FiniteAreaPlant, PlantConfig
from peirastic.contact_qp.features import FeatureConfig

OUT = Path(__file__).resolve().parent
MODEL = ROOT / "peirastic/contact_qp/plant.py"
MODEL_SOURCE = MODEL.read_bytes()
MODEL_HASH = hashlib.sha256(MODEL_SOURCE).hexdigest()
MANIFEST = json.loads((ROOT / "MD/contact_qp/acceptance_v1.json").read_text())
SEEDS = MANIFEST["design_seeds"]
assert not set(SEEDS) & set(MANIFEST["acceptance_seeds"])
PATHS = np.linspace(0., .06, 121)
ANGLES = np.linspace(-.35, .35, 281)
FEATURE_CONFIG = FeatureConfig()


def equilibrium(plant, path):
    """Exact static unilateral-spring solution at control Fz=4 N for each angle."""
    c, sx = np.cos(ANGLES)[:, None], np.sin(ANGLES)[:, None]
    world_points_x = plant.world_x + c * plant.x[None, :]
    heights = plant.surface(path=path, time=0., world_x=world_points_x)
    eps = 1e-5  # Same central-difference slope as the actual physical port.
    slope_x = (plant.surface(path=path, time=0., world_x=world_points_x+eps)
               - plant.surface(path=path, time=0., world_x=world_points_x-eps)) / (2*eps)
    effective_k = plant.k[None, :] * (c - sx * slope_x)
    assert np.all(effective_k > 0.), "Monotone force root assumption fails"
    offset = heights + sx * plant.x[None, :]
    order = np.argsort(offset, axis=1)
    off_sorted = np.take_along_axis(offset, order, axis=1)
    k_sorted = np.take_along_axis(effective_k, order, axis=1)
    z_candidates = (4. + np.cumsum(k_sorted * off_sorted, axis=1)) / np.cumsum(k_sorted, axis=1)
    next_offset = np.concatenate((off_sorted[:, 1:], np.full((len(ANGLES), 1), np.inf)), axis=1)
    active_last = np.argmax(z_candidates <= next_offset, axis=1)
    z = z_candidates[np.arange(len(ANGLES)), active_last]
    depth = z[:, None] - offset
    fz = np.sum(effective_k * np.maximum(0., depth), axis=1)
    assert np.max(np.abs(fz - 4.)) < 1e-12
    image_fraction = .5 * (plant.u + 1)
    coupled = depth >= plant.config.coupling_depth_m
    windows = [(image_fraction >= lo) & (image_fraction < hi)
               for lo, hi in FEATURE_CONFIG.lateral_windows]
    quality = np.stack([coupled[:, w].mean(axis=1) for w in windows], axis=1)
    # Minimum compression beyond the 90%-coupled cutoff in each required window.
    clearance = []
    for w in (windows[0], windows[2]):
        needed = int(np.ceil(.90 * w.sum()))
        cutoff = np.sort(depth[:, w], axis=1)[:, -needed]
        clearance.append(cutoff - plant.config.coupling_depth_m)
    margin = np.min(clearance, axis=0)
    return z, quality, margin, windows


records = []
plot_data = None
for scenario in ("both_edges", "left_gap", "right_gap", "curvature", "delayed_execution"):
    for seed in SEEDS:
        plant = FiniteAreaPlant(scenario, seed)
        qualities, zs, margins = [], [], []
        for path in PATHS:
            z, quality, margin, windows = equilibrium(plant, path)
            qualities.append(quality)
            zs.append(z)
            margins.append(margin)
        qualities, zs, margins = map(np.asarray, (qualities, zs, margins))
        edge = qualities[:, :, [0, 2]].min(axis=2)
        feasible = edge >= .90
        center = int(np.argmin(np.abs(ANGLES)))
        record = {
            "scenario": scenario, "seed": seed,
            "sampled_paths": len(PATHS), "sampled_angles": len(ANGLES),
            "window_element_counts": [int(w.sum()) for w in windows],
            "required_coupled_counts": [int(np.ceil(.90 * windows[j].sum())) for j in (0, 2)],
            "paths_with_static_witness": int(feasible.any(axis=1).sum()),
            "theta_zero_min_edge_fraction": float(edge[:, center].min()),
            "theta_zero_min_depth_margin_m": float(margins[:, center].min()),
            "theta_zero_z_min_m": float(zs[:, center].min()),
            "theta_zero_z_max_m": float(zs[:, center].max()),
            "apex_theta_zero_z_m": float(zs[60, center]),
            "apex_theta_zero_edge_fraction": float(edge[60, center]),
            "apex_feasible_angle_min_rad": float(ANGLES[feasible[60]].min()) if feasible[60].any() else None,
            "apex_feasible_angle_max_rad": float(ANGLES[feasible[60]].max()) if feasible[60].any() else None,
            "theta_zero_nominal_scan_peak_abs_vz_m_s": float(np.max(np.abs(np.gradient(zs[:, center], PATHS))) * plant.config.scan_speed_m_s),
            "paths": [
                {"s_m": float(path), "theta_zero_edge_fraction": float(edge[i, center]),
                 "theta_zero_margin_m": float(margins[i, center]),
                 "best_edge_fraction": float(edge[i].max()),
                 "feasible_theta_min_rad": float(ANGLES[feasible[i]].min()) if feasible[i].any() else None,
                 "feasible_theta_max_rad": float(ANGLES[feasible[i]].max()) if feasible[i].any() else None}
                for i, path in enumerate(PATHS)
            ],
        }
        # Independent equality check through the actual mutable plant state.
        for pi, ai in ((0, center), (60, center), (60, 0), (60, len(ANGLES)-1), (120, center)):
            plant.path, plant.theta, plant.z = PATHS[pi], ANGLES[ai], zs[pi, ai]
            plant.velocity[:] = 0.
            plant._update_contact()
            assert abs(plant.control_wrench(noise=False)[2] - 4.) < 1e-12
            assert np.array_equal(plant.window_coupling(FEATURE_CONFIG), qualities[pi, ai])
        records.append(record)
        if scenario == "both_edges" and seed == SEEDS[0]:
            plot_data = edge, zs, margins

assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == MODEL_HASH, "Plant changed during audit; rerun on the frozen source"
metadata = {
    "schema_version": 1, "scope": "STATIC_DESIGN_AUDIT_NOT_ACCEPTANCE",
    "plant_sha256": MODEL_HASH,
    "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "acceptance_sha256": hashlib.sha256((ROOT / "MD/contact_qp/acceptance_v1.json").read_bytes()).hexdigest(),
    "seed_source": "design_seeds_only", "seeds": SEEDS,
    "plant_config": vars(PlantConfig()),
    "feature_config": vars(FEATURE_CONFIG),
    "equilibrium": "sum(k_i*(cos(theta)-sin(theta)*dh_i/dx)*max(z-x_i*sin(theta)-h_i(s,world_x+x_i*cos(theta)),0))=4 N; measured velocities=0; TCP world_x=0",
    "path_grid_m": [0., .06, .0005], "angle_grid_rad": [-.35, .35, .0025],
    "limitations": ["No held-out seeds used", "Static existence only; no closed-loop efficacy",
                    "No dynamic reachability, damping, safety-admission or energy certificate",
                    "Finite grid does not prove global infeasibility when no witness is found",
                    "No independent normal-displacement bound exists in current PlantConfig",
                    "Moving-surface family excluded: time-phase feasibility requires an additional time grid"],
    "runs": records,
}
OUT.mkdir(exist_ok=True)
(OUT / "static_feasibility_model_snapshot.py.txt").write_bytes(MODEL_SOURCE)
(OUT / "static_feasibility_design.json").write_text(json.dumps(metadata, indent=2) + "\n")
edge, zs, margins = plot_data
center = int(np.argmin(np.abs(ANGLES)))
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
mesh = axes[0].pcolormesh(PATHS * 1000., ANGLES, edge.T, vmin=0., vmax=1., shading="nearest")
axes[0].contour(PATHS * 1000., ANGLES, edge.T, levels=[.90], colors="white", linewidths=.9)
axes[0].axhline(0., color="red", lw=.7)
axes[0].set(xlabel="Measured path (mm)", ylabel="Static angle (rad)", title="Both-edge coupling at 4 N")
fig.colorbar(mesh, ax=axes[0], label="min(left,right) fraction")
axes[1].plot(PATHS * 1000., zs[:, center] * 1000.)
axes[1].set(xlabel="Measured path (mm)", ylabel="Normal displacement (mm)", title="Force-equilibrium branch at angle 0")
axes[2].plot(PATHS * 1000., margins[:, center] * 1000.)
axes[2].axhline(0., color="black", lw=.7)
axes[2].set(xlabel="Measured path (mm)", ylabel="90%-coupling clearance (mm)", title="Positive static feasibility margin")
for ax in axes[1:]:
    ax.grid(alpha=.25)
fig.suptitle("Synthetic both_edges, design seed 0; static witness, not controller efficacy", fontsize=11)
fig.savefig(OUT / "both_edges_static_feasibility.png", dpi=160)
plt.close(fig)
summary = {name: {
    "all_sampled_paths_have_witness": all(r["paths_with_static_witness"] == len(PATHS) for r in records if r["scenario"] == name),
    "theta_zero_min_edge_fraction": min(r["theta_zero_min_edge_fraction"] for r in records if r["scenario"] == name),
    "theta_zero_min_depth_margin_m": min(r["theta_zero_min_depth_margin_m"] for r in records if r["scenario"] == name),
    "apex_theta_zero_z_range_m": [min(r["apex_theta_zero_z_m"] for r in records if r["scenario"] == name), max(r["apex_theta_zero_z_m"] for r in records if r["scenario"] == name)],
    "theta_zero_nominal_scan_peak_abs_vz_m_s": max(r["theta_zero_nominal_scan_peak_abs_vz_m_s"] for r in records if r["scenario"] == name),
} for name in dict.fromkeys(r["scenario"] for r in records)}
print(json.dumps({"plant_sha256": MODEL_HASH, "summary": summary}, indent=2))
