"""Plot a fixed held-out Y scan and whole-map errors for the paired cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bifurcation_model import graph_nodes, prior
from leg_chart import LegChart


BRANCHES = (slice(0, 9), np.r_[8, np.arange(9, 17)], np.r_[8, np.arange(17, 25)])
INK = "#12303b"
TEAL = "#007d72"
GOLD = "#b07724"


def plot_graph(axis, graph, style, color, *, alpha=1., lw=1., label=None):
    for i, indices in enumerate(BRANCHES):
        points = graph[indices]
        axis.plot(*points.T, style, color=color, alpha=alpha, lw=lw, label=label if i == 0 else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in (args.run / "EPISODES.jsonl").read_text().splitlines()]
    group = [r for r in rows if r["regime"] == "continuous" and r["method"] == "B_joint"]
    # First registered 45-degree continuous episode, without selecting on outcome.
    example = next((r for r in group if r["pose_deg"] == 45), group[0])
    episode = example["episode"]
    with np.load(args.run / "FINAL_MAPS.npz") as maps:
        i = int(np.flatnonzero((maps["episode"] == episode) & (maps["method"] == "B_joint"))[0])
        n = maps["posterior_count"][i]
        theta = maps["posterior_coefficients"][i, :n]
        weights = np.exp(maps["log_weights"][i, :n])
        truth = maps["evaluation_truth_coefficients"][i]
        actual = maps["evaluation_truth_world_mm"][i]
        estimate = maps["mean_world_mm"][i]
    old, w0 = prior()
    base_graph = graph_nodes(old)
    new_graph = graph_nodes(theta)
    trace = [r for r in map(json.loads, (args.run / "PUBLIC_SCAN.jsonl").open())
             if r["episode"] == episode and r["method"] == "B_joint"]
    route = json.loads((args.run / f"INITIAL_GPS_{example['pose_deg']:02d}.json").read_text())
    goals = np.asarray([r["xy_mm"] for r in route])

    fig = plt.figure(figsize=(13.2, 4.1), layout="constrained", facecolor="white")
    ax0 = fig.add_subplot(131, projection="3d")
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    chart = LegChart(example["pose_deg"])
    x, y = np.meshgrid(np.linspace(-40., 40., 21), np.linspace(-32., 32., 17), indexing="ij")
    skin = chart.world(np.stack((x, y, np.zeros_like(x)), axis=-1))
    center = chart.world(np.array([[0., 0., 0.]]))[0]
    frame = chart.frame(np.array([[0., 0.]]))[0]
    shape = (skin-center) @ frame
    ax0.plot_surface(*np.moveaxis(shape, -1, 0), color="#b5d8db", alpha=.28, linewidth=0)
    surface_gps = chart.world(np.column_stack((goals, np.zeros(len(goals)))))
    ax0.scatter(*((surface_gps-center) @ frame).T, color=GOLD, s=13, depthshade=False, label="Initial skin GPS")
    plot_graph(ax0, (actual-center) @ frame, "--", INK, lw=1.5, label="Evaluation vessel")
    plot_graph(ax0, (estimate-center) @ frame, "-", TEAL, lw=2., label="Posterior mean")
    ax0.set_title(f"Leg atlas at {example['pose_deg']}° knee pose", loc="left", color=INK, fontsize=11)
    ax0.set(xlabel="Longitudinal (mm)", ylabel="Lateral (mm)", zlabel="Inward (mm)")
    ax0.view_init(elev=23, azim=-65)
    ax0.legend(loc="upper left", fontsize=7, frameon=False)

    for path in (np.asarray(r["path_xy_mm"]) for r in trace):
        ax1.plot(*path.T, color=GOLD, alpha=.43, lw=1.25)
    quantiles = np.linspace(.02, .98, 25)
    for j in np.searchsorted(np.cumsum(w0), quantiles):
        plot_graph(ax1, base_graph[j, :, :2], "-", "#7b95ac", alpha=.12)
    for j in np.minimum(n-1, np.searchsorted(np.cumsum(weights), quantiles)):
        plot_graph(ax1, new_graph[j, :, :2], "-", TEAL, alpha=.16)
    plot_graph(ax1, graph_nodes(truth[None])[0, :, :2], "--", INK, lw=1.8, label="Evaluation vessel")
    posterior_chart = np.einsum("h,hij->ij", weights, new_graph)
    plot_graph(ax1, posterior_chart[:, :2], "-", TEAL, lw=2., label="Posterior mean")
    ax1.scatter(*goals.T, color=GOLD, s=13, label="Initial skin GPS")
    ax1.set_title("ProDMP motion and internal estimates", loc="left", color=INK, fontsize=11)
    ax1.set(xlabel="Material longitudinal (mm)", ylabel="Material lateral (mm)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.legend(loc="lower left", fontsize=7, frameon=False)

    limits = np.array([0., max(max(r["initial_map"]["rmse_mm"], r["final_map"]["rmse_mm"]) for r in group) * 1.08])
    ax2.plot(limits, limits, "--", color="#83939b", lw=1.)
    covered = [r for r in group if r["final_map"]["joint_90_covered"]]
    missed = [r for r in group if not r["final_map"]["joint_90_covered"]]
    for data, color, label in ((covered, TEAL, "90% region covered"), (missed, GOLD, "Outside 90% region")):
        ax2.scatter([r["initial_map"]["rmse_mm"] for r in data],
                    [r["final_map"]["rmse_mm"] for r in data], c=color, s=24, alpha=.76, label=label)
    ax2.set(xlim=limits, ylim=limits, xlabel="Initial graph RMSE (mm)", ylabel="Final graph RMSE (mm)")
    ax2.set_title(f"Continuous Y cohort, n={len(group)}", loc="left", color=INK, fontsize=11)
    ax2.legend(loc="upper left", fontsize=7, frameon=False)
    for axis in (ax1, ax2):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.12, linewidth=.5)
        axis.tick_params(labelsize=8)
    fig.savefig(args.output, dpi=175, facecolor="white")
    print(example["episode"], args.output)


if __name__ == "__main__":
    main()
