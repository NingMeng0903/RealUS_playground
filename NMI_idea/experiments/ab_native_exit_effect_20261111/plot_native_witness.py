"""Replot two measured native continuation contrasts from saved trajectories."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

root = Path(__file__).resolve().parent
fig, (path_ax, error_ax) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
colors = ("#187c80", "#b45042")
balanced = root / "balanced_fresh_030"
rows = [json.loads(line) for line in (balanced / "EXECUTED.jsonl").read_text().splitlines()]
manifest = json.loads((root / "fixture_00_constrained" / "native_exit_matrix.jsonl").read_text().splitlines()[0])
xmin, xmax, ymin, ymax = [1000 * x for x in manifest["adapter_fingerprint"]["rig"]["wall"]]
path_ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor="#ece8e2", edgecolor="#999188"))
for row, color in zip((r for r in rows if r["action"] == "left"), colors):
    points = []
    for name in row["records"]:
        with np.load(balanced / "execution" / f"{name}.npz") as trace:
            points.append(1000 * trace["local_xyz_m"][:, :2])
    xy = np.concatenate(points)
    side = "Left" if row["probe"].startswith("side_left") else "Right"
    path_ax.plot(*xy.T, color=color, lw=2.25, label=f"{side} exit · {1000 * row['path_length_m']:.1f} mm")
    path_ax.scatter(*xy[0], color=color, s=28, zorder=4)
    path_ax.scatter(*xy[-1], marker="*", color="#273138", s=120, zorder=5)
path_ax.set(title="Same left-window target", xlabel="Surface x (mm)", ylabel="Surface y (mm)",
            xlim=(-24, 24), ylim=(-8, 25))
path_ax.text((xmin+xmax)/2, 17.5, "keep-out", ha="center", fontsize=8, rotation=90, color="#685d52")
path_ax.set_aspect("equal")
path_ax.legend(loc="lower center", fontsize=8, frameon=False)

short = root / "dwell_sensitivity" / ".30"
short_row = json.loads((short / "RESULT.json").read_text())["links"][-1]
long = root / "late_native_witness" / "after_4"
long_row = next(
    route for route in json.loads((long / "RESULT.json").read_text())["attempts"]
    if route["route"] == "retreat_fork"
)
for directory, names, dwell, color in (
    (short, short_row["traces"], "0.30 s", colors[0]),
    (long, long_row["records"], "0.60 s", colors[1]),
):
    with np.load(directory / "execution" / f"{names[-1]}.npz") as trace:
        error_mm = 1000 * np.linalg.norm(trace["measured_before"][:, :3, 3] - trace["desired"][:, :3, 3], axis=1)
        error_ax.plot(trace["time_s"], error_mm, color=color, lw=2.2,
                      label=f"Scout {dwell} · {error_mm.max():.2f} mm peak")
error_ax.axhline(2, color="#687279", linestyle="--", linewidth=1.3, label="Existing 2 mm gate")
error_ax.set(title="Same continuation, different scout dwell",
             xlabel="Time in final segment (s)", ylabel="Position tracking error (mm)")
error_ax.legend(loc="center right", fontsize=8, frameon=False)
for ax in (path_ax, error_ax):
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(alpha=.13)
fig.savefig(root / "native_witness.png", dpi=180)
