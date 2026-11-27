#!/usr/bin/env python3
"""Render the planned phantom trajectory figure for the proposal.

The figure is deliberately offline: it reads the frozen plan and point-cloud
artifacts from a comparison-study run and never opens a robot, camera, or
shared-memory channel.  The surface interpolation and PLY reader are shared
with ``plan_rgb_figure.py`` so that this small proposal figure uses the same
geometric evidence as the original ICRA figure.

Examples
--------
    python scripts/plot_proposal_trajectory.py
    python scripts/plot_proposal_trajectory.py --study \
        '/media/camp/PEI_T7/icra 2027_contact/Comparison Study' \
        --out outputs/proposal_figures_20261015/fig1_phantom_trajectory
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

# Running ``python scripts/plot_proposal_trajectory.py`` puts ``scripts/``
# rather than the repository root on sys.path.  Add the root explicitly so the
# shared offline phantom figure module resolves without a shell-specific
# PYTHONPATH.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from peirastic.DEMO.phathom_scanning.plan_rgb_figure import (
    _unit,
    _read_ascii_ply,
    colored_surface_grid,
    load_plan,
)


DEFAULT_STUDY = Path("/media/camp/PEI_T7/icra 2027_contact/Comparison Study")
DEFAULT_RUN = Path("ultrapoc/004noise")
DEFAULT_OUT = Path("outputs/proposal_figures_20261015/fig1_phantom_trajectory")

TRAJECTORY_COLOR = "#0072B2"
SURFACE_NORMAL_COLOR = "#D55E00"
OFFSET_NORMAL_COLOR = "#8E44AD"
INK = "#272727"
SURFACE_EDGE = "#A77C68"

# Register the installed paper fonts explicitly.  Matplotlib's font cache
# exposes the TTC as the JP face even though the collection contains the CJK
# glyphs needed here; using a FontProperties object for the Chinese legend
# avoids silently falling back to tofu boxes on clean machines.
CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
TIMES_FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")
if CJK_FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(CJK_FONT_PATH))
if TIMES_FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(TIMES_FONT_PATH))
CJK_FONT = FontProperties(fname=str(CJK_FONT_PATH)) if CJK_FONT_PATH.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _surface_inputs(run: Path):
    """Select the phantom neighbourhood used by the original plan figure."""
    cloud, colors = _read_ascii_ply(run / "detect_cloud.ply")
    top, _ = _read_ascii_ply(run / "detect_top.ply")
    if len(top) == 0:
        raise ValueError("detect_top.ply contains no points")

    # ``plan_rgb_figure.render`` uses the selected top cloud to crop the dense
    # coloured cloud before fitting its surface.  Keep that exact selection so
    # the proposal panel does not introduce a different phantom geometry.
    lo = top.min(axis=0) - 0.010
    hi = top.max(axis=0) + 0.010
    lo[2] -= 0.018
    keep = np.all((cloud >= lo) & (cloud <= hi), axis=1)
    near_cloud, near_colors = cloud[keep], colors[keep]
    if len(near_cloud) < 16:
        near_cloud = top
        near_colors = np.full((len(top), 3), 0.70, dtype=float)
        selection = "detect_top_fallback"
    else:
        selection = "detect_cloud_cropped_by_detect_top_bounds"
    return near_cloud, near_colors, top, selection


def _arrow(ax, origin_m, direction, length_m, color, *, linewidth=1.05, zorder=12):
    """Draw one compact 3D vector with a consistent arrow head."""
    origin = np.asarray(origin_m, dtype=float).reshape(3)
    direction = _unit(direction)
    ax.quiver(
        origin[0], origin[1], origin[2],
        direction[0], direction[1], direction[2],
        length=float(length_m), normalize=False,
        color=color, linewidth=linewidth, arrow_length_ratio=0.22,
        pivot="tail", alpha=1.0, zorder=zorder,
    )


def _relative_mm(points_m, center_m):
    return (np.asarray(points_m, dtype=float) - np.asarray(center_m, dtype=float)) * 1000.0


def _set_equalish_limits(ax, arrays_mm):
    points = np.vstack([np.asarray(a, dtype=float).reshape(-1, 3) for a in arrays_mm])
    lo = np.nanmin(points, axis=0)
    hi = np.nanmax(points, axis=0)
    span = np.maximum(hi - lo, 1.0)
    pad = np.array([0.055, 0.055, 0.12]) * span + np.array([2.0, 2.0, 2.0])
    lo -= pad
    hi += pad
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_zlim(float(lo[2]), float(hi[2]))
    # Keep the surface readable while retaining its measured aspect.  The Z
    # range is much smaller than X/Y for this phantom, so a mild vertical
    # exaggeration avoids a visually flat panel without changing coordinates.
    ax.set_box_aspect(np.maximum(hi - lo, 1.0) * np.array([1.0, 1.0, 1.28]))


def _write_trace_csv(path: Path, poses, surface_normals, command_normals, center_m):
    path.parent.mkdir(parents=True, exist_ok=True)
    poses = np.asarray(poses, dtype=float)
    surface_normals = np.asarray(surface_normals, dtype=float)
    command_normals = np.asarray(command_normals, dtype=float)
    xyz_mm = _relative_mm(poses[:, :3], center_m)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "waypoint", "x_mm", "y_mm", "z_mm",
            "surface_nx", "surface_ny", "surface_nz",
            "offset_nx", "offset_ny", "offset_nz",
        ])
        for i, (xyz, normal, command) in enumerate(zip(xyz_mm, surface_normals, command_normals)):
            writer.writerow([int(i), *[f"{float(v):.9f}" for v in xyz],
                             *[f"{float(v):.9f}" for v in normal],
                             *[f"{float(v):.9f}" for v in command]])


def build(study: Path, run_rel: Path, out: Path, *, arrow_count: int = 6):
    run = Path(study) / run_rel
    plan_path = run / "plan.json"
    cloud_path = run / "detect_cloud.ply"
    top_path = run / "detect_top.ply"
    for path in (plan_path, cloud_path, top_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = load_plan(run)
    cloud, colors, top, surface_selection = _surface_inputs(run)
    surface_normal = _unit(np.nanmean(np.asarray(plan.normals), axis=0))
    mesh = colored_surface_grid(
        cloud, colors, plan.right, plan.far, surface_normal,
        pitch_m=0.0032, smooth_sigma=1.7,
    )

    poses = np.asarray(plan.poses, dtype=float)
    surface_normals = np.asarray(plan.normals, dtype=float)
    command_normals = np.asarray(plan.command_normals, dtype=float)
    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"expected poses with xyz columns, got {poses.shape}")
    if surface_normals.shape != (len(poses), 3) or command_normals.shape != (len(poses), 3):
        raise ValueError("plan normals and poses have inconsistent lengths")

    # The scan center is the plan's authored coordinate origin.  Keeping this
    # origin makes the three axes directly interpretable in millimetres while
    # preserving every recorded waypoint exactly.
    center_m = np.asarray(plan_data["scan_center"], dtype=float)
    surface_xyz = np.asarray(mesh.xyz, dtype=float)
    surface_mm = _relative_mm(surface_xyz.reshape(-1, 3), center_m).reshape(surface_xyz.shape)
    path_mm = _relative_mm(poses[:, :3], center_m)
    # Lift the rendered polyline uniformly along each recorded surface normal
    # by 1 mm for visibility.  The authored waypoints and exported CSV remain
    # unchanged; this is only a display layer above the opaque surface.
    path_display_m = poses[:, :3] + 0.001 * surface_normals
    path_display_mm = _relative_mm(path_display_m, center_m)

    arrow_count = max(1, min(int(arrow_count), len(poses)))
    arrow_indices = np.linspace(0, len(poses) - 1, arrow_count, dtype=int)
    # Lift vectors by a small measured plotting-only amount so their tails sit
    # visibly above the surface; this does not modify the stored trajectory.
    origins_m = poses[arrow_indices, :3].copy()
    origins_m += 0.0006 * surface_normals[arrow_indices]

    plt.rcParams.update({
        "font.family": [
            "Times New Roman", "Noto Serif CJK JP", "Noto Serif",
            "DejaVu Serif", "serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9.0,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    mpl.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(7.5, 5.8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    # Disable mplot3d's depth-derived artist reordering.  Explicit zorders
    # below keep the measured surface behind the planned line and vectors.
    ax.computed_zorder = False
    ax.set_facecolor("white")
    try:
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    except AttributeError:
        pass

    ax.plot_surface(
        surface_mm[:, :, 0], surface_mm[:, :, 1], surface_mm[:, :, 2],
        facecolors=np.asarray(mesh.facecolors),
        rstride=1, cstride=1, linewidth=0.0, antialiased=True,
        shade=False, alpha=1.0, zorder=1,
    )
    # The complete planned polyline is retained; no interpolation or shape
    # substitution is used for the displayed trajectory.
    ax.plot(
        path_display_mm[:, 0], path_display_mm[:, 1], path_display_mm[:, 2],
        color=TRAJECTORY_COLOR, linewidth=2.35,
        solid_capstyle="round", solid_joinstyle="round", zorder=8,
    )

    for idx, origin in zip(arrow_indices, origins_m):
        origin_mm = _relative_mm(origin, center_m)
        # Surface normals are shown in warm vermillion; biased command normals
        # are purple, matching the 20-degree offset stored in plan.json.
        _arrow(ax, origin_mm, surface_normals[idx], 15.0, SURFACE_NORMAL_COLOR,
               zorder=12)
        _arrow(ax, origin_mm, command_normals[idx], 14.0, OFFSET_NORMAL_COLOR,
               zorder=13)

    _set_equalish_limits(ax, [surface_mm.reshape(-1, 3), path_display_mm])
    ax.set_xlabel("X (mm)", labelpad=5.5)
    ax.set_ylabel("Y (mm)", labelpad=5.5)
    # The projected 3D z-label is easily clipped by Matplotlib's tight bbox;
    # place it in figure coordinates so it remains visible in every backend.
    ax.set_zlabel("")
    fig.text(0.966, 0.505, "Z (mm)", rotation=90, va="center", ha="center",
             fontsize=10.5, fontfamily="Times New Roman", color=INK)
    ax.view_init(elev=28.0, azim=-62.0)
    # Keep the proposal panel visually light: three meaningful ticks per axis
    # are sufficient for the measured phantom neighbourhood.
    ax.set_xticks([-50.0, 0.0, 50.0])
    ax.set_yticks([-100.0, 0.0, 100.0])
    ax.set_zticks([-20.0, 0.0, 20.0])
    ax.tick_params(axis="both", which="major", pad=1.5, width=0.65, length=3.0)
    ax.tick_params(axis="z", which="major", pad=1.0, width=0.65, length=3.0)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.55, 0.55, 0.55, 0.30)
        axis._axinfo["grid"]["linewidth"] = 0.35

    legend_handles = [
        Line2D([0], [0], color=TRAJECTORY_COLOR, linewidth=2.35, label="规划轨迹"),
        Line2D([0], [0], color=SURFACE_NORMAL_COLOR, linewidth=1.45, label="表面法向"),
        Line2D([0], [0], color=OFFSET_NORMAL_COLOR, linewidth=1.45, label="偏置法向"),
    ]
    legend = ax.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(0.015, 0.985),
        frameon=False, ncol=3, columnspacing=1.05, handlelength=1.7,
        handletextpad=0.35, borderaxespad=0.0,
    )
    if CJK_FONT is not None:
        for text in legend.get_texts():
            text.set_fontproperties(CJK_FONT)
            text.set_fontsize(9.0)
    # Reserve a narrow right margin for the hand-placed Z label and use the
    # full landscape canvas rather than a tight bbox that distorts the ratio.
    fig.subplots_adjust(left=0.025, right=0.935, bottom=0.045, top=0.975)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    png_path = out.with_suffix(".png")
    pdf_path = out.with_suffix(".pdf")
    svg_path = out.with_suffix(".svg")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)

    trace_path = out.with_name(out.name + "_data.csv")
    provenance_path = out.with_name(out.name + "_provenance.json")
    _write_trace_csv(trace_path, poses, surface_normals, command_normals, center_m)
    provenance = {
        "schema": "proposal_phantom_trajectory_v1",
        "figure": "fig1_phantom_trajectory",
        "claim": "planned Lissajous trajectory on the measured phantom surface",
        "study": str(Path(study).resolve()),
        "run": str(run_rel),
        "run_path": str(run.resolve()),
        "source_files": {
            "plan_json": {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path)},
            "detect_cloud_ply": {"path": str(cloud_path.resolve()), "sha256": _sha256(cloud_path)},
            "detect_top_ply": {"path": str(top_path.resolve()), "sha256": _sha256(top_path)},
        },
        "surface_selection": surface_selection,
        "surface_input_points": int(len(cloud)),
        "surface_top_points": int(len(top)),
        "surface_mesh_shape": [int(v) for v in mesh.xyz.shape],
        "trajectory_waypoints": int(len(poses)),
        "trajectory_pattern": str(plan_data["pattern"]),
        "normal_offset_deg": float(plan_data.get("normal_offset_deg") or 0.0),
        "coordinate_frame": str(plan_data["frame"]),
        "coordinate_units": "relative to plan.scan_center, millimetres in figure and CSV",
        "scan_center_m": [float(v) for v in center_m],
        "normal_arrow_indices": [int(v) for v in arrow_indices],
        "plotting_lift_m": 0.0006,
        "outputs": {
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "svg": str(svg_path.resolve()),
            "trajectory_csv": str(trace_path.resolve()),
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [png_path, pdf_path, svg_path, trace_path, provenance_path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN,
                        help="run path relative to --study (default: ultrapoc/004noise)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output basename; suffixes are added automatically")
    parser.add_argument("--arrow-count", type=int, default=6)
    args = parser.parse_args()
    written = build(args.study, args.run, args.out, arrow_count=args.arrow_count)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
