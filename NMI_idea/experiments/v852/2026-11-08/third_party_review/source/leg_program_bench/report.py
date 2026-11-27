"""Offline plots and metrics for a frozen leg-program benchmark run.

The report deliberately consumes the files written by :mod:`run_experiment`
instead of regenerating outcomes.  In particular, it does not simulate new
traces or attach confidence bands to empirical curves.  The experiment is a
Cartesian feature surrogate; it contains no real-robot or B-mode evidence.

Usage::

    PYTHONPATH=work python -m leg_program_bench.report \
        --input work/pilot_smoke3 --output work/pilot_smoke3/report

The required input files are ``summary.json``, ``evaluation.json`` and
``four_cell_measured.json``.  Missing files or malformed curves are reported
as errors rather than being silently filled in.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

# Matplotlib may otherwise try to create a cache in a user directory while a
# report is being generated on a headless machine.  This is intentionally a
# dedicated temporary cache, not a repository artifact.
_MPL_CONFIG = Path("/tmp/leg_program_bench_mpl")
_MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

try:  # normal package invocation
    from .atlas_scene import AtlasLeg
except ImportError:  # pragma: no cover - useful when run as a script
    from atlas_scene import AtlasLeg


CORE_METHODS = (
    "joint_material",
    "joint_anchor",
    "joint_world",
    "geometry_servo",
    "marginal_ablation",
)
BUDGET_VARIANTS = ("joint_material_n8", "joint_material_n16", "joint_material")
METHOD_COLORS = {
    "joint_material": "#1f77b4",
    "joint_anchor": "#ff7f0e",
    "joint_world": "#2ca02c",
    "geometry_servo": "#d62728",
    "marginal_ablation": "#9467bd",
    "joint_material_n8": "#17becf",
    "joint_material_n16": "#8c564b",
}
LANDMARK_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")
LANDMARK_PROGRESS = np.asarray((0.10, 0.50, 0.85), dtype=float)


class ReportInputError(ValueError):
    """Raised when a run output cannot support an honest report."""


def _read_json(input_dir: Path, filename: str, *, required: bool = True) -> tuple[Any, Path | None]:
    """Read a named run artifact, accepting one nested copy for convenience."""

    direct = input_dir / filename
    if direct.is_file():
        path = direct
    else:
        matches = sorted(p for p in input_dir.rglob(filename) if p.is_file())
        if not matches:
            if required:
                raise ReportInputError(f"Missing required input: {input_dir / filename}")
            return None, None
        if len(matches) > 1:
            raise ReportInputError(f"Ambiguous input {filename}; found {matches}")
        path = matches[0]
    try:
        return json.loads(path.read_text()), path
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"Could not read JSON input {path}: {exc}") from exc


def _finite_curve(value: Any, *, label: str, expected: int | None = None) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ReportInputError(f"{label} must be a non-empty JSON list")
    try:
        curve = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ReportInputError(f"{label} is not numeric") from exc
    if curve.ndim != 1 or not np.all(np.isfinite(curve)):
        raise ReportInputError(f"{label} must be a finite one-dimensional curve")
    if expected is not None and len(curve) != expected:
        raise ReportInputError(f"{label} has {len(curve)} samples; expected {expected}")
    return curve


def _positive_count(value: Any, *, label: str) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ReportInputError(f"{label} is not an integer") from exc
    if n < 0:
        raise ReportInputError(f"{label} must be non-negative")
    return n


def _friendly_stratum(name: str) -> str:
    labels = {
        "new_object_known_pose": "new object, known pose",
        "heldout_pose": "test_new_pose",
        "heldout_shape": "test_new_shape",
        "heldout_pose_and_shape": "test_new_pose_and_shape",
    }
    return labels.get(name, name)


def _load_run(input_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ReportInputError(f"Input directory does not exist: {input_dir}")

    summary, summary_path = _read_json(input_dir, "summary.json")
    evaluation, evaluation_path = _read_json(input_dir, "evaluation.json")
    four_cells, four_cell_path = _read_json(input_dir, "four_cell_measured.json")

    if not isinstance(summary, dict):
        raise ReportInputError("summary.json must contain an object")
    if not isinstance(evaluation, list):
        raise ReportInputError("evaluation.json must contain a list")
    if not isinstance(four_cells, list):
        raise ReportInputError("four_cell_measured.json must contain a list")
    protocol = summary.get("protocol")
    rows = summary.get("rows")
    if not isinstance(protocol, dict) or not isinstance(rows, list) or not rows:
        raise ReportInputError("summary.json needs non-empty protocol and rows fields")

    deadlines_value = protocol.get("deadlines")
    deadlines = _finite_curve(deadlines_value, label="protocol.deadlines")
    if len(deadlines) < 2 or np.any(np.diff(deadlines) <= 0):
        raise ReportInputError("protocol.deadlines must contain at least two strictly increasing values")

    parsed_rows: list[dict[str, Any]] = []
    strata: set[str] = set()
    methods: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ReportInputError(f"summary.rows[{index}] must be an object")
        stratum = row.get("stratum")
        method = row.get("method")
        if not isinstance(stratum, str) or not isinstance(method, str):
            raise ReportInputError(f"summary.rows[{index}] needs string stratum and method")
        actual = _finite_curve(
            row.get("mean_finish_curve"),
            label=f"summary.rows[{index}].mean_finish_curve",
            expected=len(deadlines),
        )
        predicted = _finite_curve(
            row.get("mean_predicted_curve"),
            label=f"summary.rows[{index}].mean_predicted_curve",
            expected=len(deadlines),
        )
        n_objects = _positive_count(
            row.get("n_independent_objects", 0),
            label=f"summary.rows[{index}].n_independent_objects",
        )
        parsed = dict(row)
        parsed["_actual"] = actual
        parsed["_predicted"] = predicted
        parsed["_n_objects"] = n_objects
        parsed_rows.append(parsed)
        strata.add(stratum)
        methods.add(method)

    if len(strata) != 4:
        raise ReportInputError(f"Expected four benchmark strata, found {sorted(strata)}")
    missing_core = sorted(set(CORE_METHODS) - methods)
    if missing_core:
        raise ReportInputError(f"summary.json is missing required methods: {missing_core}")

    # Validate the evaluation artifact rather than silently ignoring a corrupt
    # file.  Curves in it are per-object/per-method and use the same deadline
    # grid as summary.json.
    if not evaluation:
        raise ReportInputError("evaluation.json is empty")
    for index, row in enumerate(evaluation):
        if not isinstance(row, dict):
            raise ReportInputError(f"evaluation[{index}] must be an object")
        for key in ("observed_finish", "predicted_finish"):
            _finite_curve(row.get(key), label=f"evaluation[{index}].{key}", expected=len(deadlines))
        for key in ("stratum", "method"):
            if not isinstance(row.get(key), str):
                raise ReportInputError(f"evaluation[{index}].{key} must be a string")

    # A cell is a measured curve indexed by the observed evidence sign y and
    # executed mode.  Keep all values; Gamma(T) is only computed below when
    # all four cells for a stratum have positive sample counts.
    for index, cell in enumerate(four_cells):
        if not isinstance(cell, dict):
            raise ReportInputError(f"four_cell_measured[{index}] must be an object")
        if not isinstance(cell.get("stratum"), str):
            raise ReportInputError(f"four_cell_measured[{index}].stratum must be a string")
        for key in ("y", "mode"):
            if cell.get(key) not in (-1, 1):
                raise ReportInputError(f"four_cell_measured[{index}].{key} must be -1 or 1")
        n_objects = _positive_count(cell.get("n_objects"), label=f"four_cell_measured[{index}].n_objects")
        finish = cell.get("finish")
        if n_objects > 0:
            _finite_curve(finish, label=f"four_cell_measured[{index}].finish", expected=len(deadlines))
        else:
            # run_experiment writes nulls for an empty ± cell.  Preserve the
            # missing measurement and let Gamma(T) omit this stratum.
            if not isinstance(finish, list) or len(finish) != len(deadlines):
                raise ReportInputError(
                    f"four_cell_measured[{index}].finish must be a deadline-length list when n_objects=0"
                )

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in parsed_rows:
        key = (row["stratum"], row["method"])
        if key in rows_by_key:
            raise ReportInputError(f"Duplicate summary row for {key}")
        rows_by_key[key] = row

    models = summary.get("models", {})
    if models is None:
        models = {}
    if not isinstance(models, dict):
        raise ReportInputError("summary.models must be an object when present")

    geometry_backend = str(protocol.get("geometry_backend", ""))
    if not geometry_backend and "native_smplx" in str(protocol.get("plant", "")):
        geometry_backend = "native_smplx"
    geometry_label = (
        "native SMPL-X geometry backend" if geometry_backend == "native_smplx"
        else "Cartesian feature surrogate"
    )

    return {
        "input_dir": input_dir,
        "summary": summary,
        "evaluation": evaluation,
        "four_cells": four_cells,
        "summary_path": summary_path,
        "evaluation_path": evaluation_path,
        "four_cell_path": four_cell_path,
        "protocol": protocol,
        "deadlines": deadlines,
        "rows": parsed_rows,
        "rows_by_key": rows_by_key,
        "strata": sorted(strata),
        "methods": sorted(methods),
        "models": models,
        "geometry_backend": geometry_backend or "cartesian_feature_surrogate",
        "geometry_label": geometry_label,
    }


def _auc(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) != len(x):
        return None
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def _curve_metrics(run: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    x = run["deadlines"]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in run["rows"]:
        actual = row["_actual"]
        predicted = row["_predicted"]
        abs_error = np.abs(actual - predicted)
        out[(row["stratum"], row["method"])] = {
            "n_independent_objects": row["_n_objects"],
            "auc_measured": _auc(x, actual),
            "auc_predicted": _auc(x, predicted),
            "prediction_integrated_absolute_error": _auc(x, abs_error),
            "prediction_max_absolute_error": float(np.max(abs_error)),
            "mean_absolute_error": float(np.mean(abs_error)),
        }
    return out


def _weighted_differences(
    run: dict[str, Any], metric_rows: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    """Compute object-count-weighted measured AUC differences to geometry."""

    result: dict[str, Any] = {}
    geometry_name = "geometry_servo"
    for method in run["methods"]:
        if method == geometry_name:
            continue
        differences: list[float] = []
        weights: list[float] = []
        per_stratum: dict[str, float | None] = {}
        for stratum in run["strata"]:
            candidate = metric_rows.get((stratum, method))
            geometry = metric_rows.get((stratum, geometry_name))
            if candidate is None or geometry is None:
                per_stratum[stratum] = None
                continue
            a = candidate.get("auc_measured")
            g = geometry.get("auc_measured")
            if a is None or g is None:
                per_stratum[stratum] = None
                continue
            difference = float(a - g)
            per_stratum[stratum] = difference
            differences.append(difference)
            weights.append(float(candidate["n_independent_objects"]))
        result[method] = {
            "weighted_auc_difference": (
                float(np.average(differences, weights=weights)) if differences and sum(weights) > 0 else None
            ),
            "per_stratum_auc_difference": per_stratum,
            "weight_definition": "n_independent_objects",
            "total_weight": float(sum(weights)),
        }
    return result


def _budget_variant_info(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    protocol_train_objects = run["protocol"].get("train_objects")
    try:
        protocol_train_objects = int(protocol_train_objects)
    except (TypeError, ValueError):
        protocol_train_objects = None
    result: dict[str, dict[str, Any]] = {}
    for method in BUDGET_VARIANTS:
        match = re.search(r"_n(\d+)$", method)
        objects = int(match.group(1)) if match else protocol_train_objects
        if objects is None or objects <= 0:
            continue
        metadata = run["models"].get(method)
        if metadata is not None and not isinstance(metadata, dict):
            raise ReportInputError(f"summary.models[{method!r}] must be an object")
        expected_records = objects * 6
        result[method] = {
            "label": "full" if method == "joint_material" else method,
            "objects": objects,
            # The benchmark contract is six local records per training object.
            "nrecords": expected_records,
            "metadata_n_records": metadata.get("n_records") if isinstance(metadata, dict) else None,
            "available": method in run["methods"],
        }
    return result


def _write_figure(fig: plt.Figure, output_dir: Path, filename: str) -> str:
    path = output_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return filename


def _plot_finish_curves(run: dict[str, Any], output_dir: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    for ax, stratum in zip(axes_flat, run["strata"]):
        for method in CORE_METHODS:
            row = run["rows_by_key"].get((stratum, method))
            if row is None:
                continue
            color = METHOD_COLORS[method]
            ax.plot(run["deadlines"], row["_actual"], color=color, linewidth=1.8)
            ax.plot(run["deadlines"], row["_predicted"], color=color, linestyle="--", linewidth=1.2)
        ax.set_title(f"{_friendly_stratum(stratum)}\n({stratum})")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Time budget T (s)")
        ax.set_ylabel("Finish probability")
    method_handles = [Line2D([0], [0], color=METHOD_COLORS[m], lw=2, label=m) for m in CORE_METHODS]
    style_handles = [
        Line2D([0], [0], color="black", lw=2, label="measured (solid)"),
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="prospective (dashed)"),
    ]
    fig.legend(handles=method_handles + style_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        f"Finish curves by benchmark stratum — {run['geometry_label']}; no real-robot/B-mode evidence",
        y=1.06,
    )
    fig.tight_layout()
    return _write_figure(fig, output_dir, "finish_curves.png")


def _plot_completion_errors(
    run: dict[str, Any], metric_rows: dict[tuple[str, str], dict[str, Any]], output_dir: Path
) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: maximum error retains stratum identity.
    strata = run["strata"]
    x = np.arange(len(strata), dtype=float)
    methods = [m for m in CORE_METHODS if all((s, m) in metric_rows for s in strata)]
    width = 0.8 / max(1, len(methods))
    for i, method in enumerate(methods):
        values = [metric_rows[(s, method)]["prediction_max_absolute_error"] for s in strata]
        axes[0].bar(x + (i - (len(methods) - 1) / 2) * width, values, width,
                    color=METHOD_COLORS[method], label=method)
    axes[0].set_xticks(x, [_friendly_stratum(s) for s in strata], rotation=25, ha="right")
    axes[0].set_ylabel("Maximum |predicted − measured|")
    axes[0].set_title("Maximum completion-probability error")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel 2: group mean error over all deadline values and all strata.  The
    # object-count weights are explicit and are also written to JSON.
    group_values: dict[str, float | None] = {}
    group_weights: dict[str, float] = {}
    for method in run["methods"]:
        vals: list[float] = []
        weights: list[float] = []
        for stratum in strata:
            metric = metric_rows.get((stratum, method))
            if metric is None:
                continue
            vals.append(float(metric["mean_absolute_error"]))
            weights.append(float(metric["n_independent_objects"]))
        group_values[method] = float(np.average(vals, weights=weights)) if vals and sum(weights) > 0 else None
        group_weights[method] = float(sum(weights))
    group_methods = [m for m in CORE_METHODS if group_values.get(m) is not None]
    axes[1].bar(np.arange(len(group_methods)), [group_values[m] for m in group_methods],
                color=[METHOD_COLORS[m] for m in group_methods])
    axes[1].set_xticks(np.arange(len(group_methods)), group_methods, rotation=30, ha="right")
    axes[1].set_ylabel("Object-weighted mean absolute error")
    axes[1].set_title("Group mean completion-probability error")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.25)

    # Panel 3: the explicit training-budget variants.  x is nrecords =
    # objects*6, never a made-up frame count.  Pose and shape are kept as
    # separate lines so their generalization conditions are visible.
    variants = _budget_variant_info(run)
    plotted_variants = [m for m in BUDGET_VARIANTS if variants.get(m, {}).get("available")]
    pose_strata = [s for s in strata if s == "heldout_pose"]
    shape_strata = [s for s in strata if s == "heldout_shape"]
    for selected, label, color in ((pose_strata, "test_new_pose", "#1f77b4"),
                                   (shape_strata, "test_new_shape", "#d62728")):
        if not selected:
            continue
        xvals: list[float] = []
        yvals: list[float] = []
        for method in plotted_variants:
            metric = metric_rows.get((selected[0], method))
            if metric is None:
                continue
            xvals.append(float(variants[method]["nrecords"]))
            yvals.append(float(metric["mean_absolute_error"]))
        if xvals:
            axes[2].plot(xvals, yvals, marker="o", color=color, label=label)
            for xv, yv, method in zip(xvals, yvals, plotted_variants):
                axes[2].annotate(variants[method]["label"], (xv, yv),
                                 textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    if plotted_variants:
        axes[2].set_xticks([variants[m]["nrecords"] for m in plotted_variants],
                           [f"{variants[m]['label']}\n({variants[m]['nrecords']} records)" for m in plotted_variants])
    axes[2].set_xlabel("Training budget: nrecords = objects × 6")
    axes[2].set_ylabel("Mean absolute error")
    axes[2].set_title("Training-budget variants")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.25)
    if axes[2].lines:
        axes[2].legend()

    fig.suptitle(
        f"Completion-probability errors — {run['geometry_label']}; no real-robot/B-mode evidence",
        y=1.03,
    )
    fig.tight_layout()
    filename = _write_figure(fig, output_dir, "completion_probability_errors.png")
    # These values are returned through the metrics builder; keeping plotting
    # code side-effect free avoids inventing a second source of statistics.
    return filename


def _set_axes_equal(ax: Any, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _atlas_root_from_audit(audit: Any) -> Path | None:
    if not isinstance(audit, dict):
        return None
    value = audit.get("atlas_path")
    if not isinstance(value, str):
        return None
    path = Path(value)
    for parent in (path, *path.parents):
        if parent.name == "Among_US":
            return parent
    return None


def _load_geometry_scene(run: dict[str, Any]) -> tuple[Any, str]:
    """Load the geometry backend declared by the frozen protocol.

    The legacy smoke protocol has an ``AtlasLeg`` Cartesian deformation
    surrogate.  A fresh protocol may declare ``native_smplx``; in that case
    importing the old class would make the geometry panel look valid while
    silently plotting the wrong body model, so it is an explicit error until
    ``SmplxAtlasLeg`` is available.
    """

    protocol = run["protocol"]
    requested = str(protocol.get("geometry_backend", ""))
    if not requested and "native_smplx" in str(protocol.get("plant", "")):
        requested = "native_smplx"
    audit, _ = _read_json(run["input_dir"], "atlas_audit.json", required=False)
    atlas_root = _atlas_root_from_audit(audit)
    if requested == "native_smplx":
        try:
            module = importlib.import_module(f"{__package__}.smplx_scene" if __package__ else "smplx_scene")
            scene_cls = getattr(module, "SmplxAtlasLeg")
        except (ImportError, AttributeError) as exc:
            raise ReportInputError(
                "Protocol declares native_smplx geometry, but SmplxAtlasLeg is unavailable"
            ) from exc
        loader = getattr(scene_cls, "load", None)
        if loader is None:
            raise ReportInputError("SmplxAtlasLeg must expose load() for report geometry")
        try:
            scene = loader(atlas_root) if atlas_root is not None else loader()
        except TypeError:
            # Keep compatibility with a loader whose root is optional but
            # keyword-only; the class remains responsible for asset checks.
            scene = loader()
        return scene, "native SMPL-X geometry backend"

    scene = AtlasLeg.load(atlas_root) if atlas_root is not None else AtlasLeg.load()
    return scene, "Cartesian feature surrogate"


def _plot_atlas_geometry(run: dict[str, Any], output_dir: Path) -> str:
    atlas, backend_label = _load_geometry_scene(run)
    required_methods = (
        "skin_vertices", "skin_faces", "deform_points", "material_curve", "material_to_world"
    )
    missing = [name for name in required_methods if not hasattr(atlas, name)]
    if missing:
        raise ReportInputError(
            "Geometry backend does not expose the report mesh/material API: " + ", ".join(missing)
        )

    # Four panels mirror the held-out combinations in run_experiment: pose and
    # shape are varied independently, while every colored point uses the same
    # canonical material address in every panel.
    configurations = (
        (0.0, (1.0, 1.0), "0°; shape (1.00, 1.00)"),
        (60.0, (1.0, 1.0), "60°; shape (1.00, 1.00)"),
        (0.0, (1.18, 0.88), "0°; shape (1.18, 0.88)"),
        (60.0, (1.18, 0.88), "60°; shape (1.18, 0.88)"),
    )
    progress = np.linspace(0.0, 1.0, 240)
    material_curve = atlas.material_curve(progress)
    landmark_xi = atlas.material_curve(LANDMARK_PROGRESS)
    # A full skin mesh is expensive to render at native size; every displayed
    # triangle is nevertheless an actual triangle from atlas.skin_faces.
    stride = max(1, int(np.ceil(len(atlas.skin_faces) / 3500)))

    fig = plt.figure(figsize=(15, 11))
    for index, (knee_deg, shape, title) in enumerate(configurations, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        skin = atlas.deform_points(atlas.skin_vertices, knee_deg=knee_deg, shape=shape)
        faces = atlas.skin_faces[::stride]
        triangles = skin[faces]
        mesh = Poly3DCollection(triangles, facecolor="#aeb7bf", edgecolor="none", alpha=0.18)
        ax.add_collection3d(mesh)
        centerline = atlas.material_to_world(material_curve, knee_deg=knee_deg, shape=shape)
        ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color="#111111", linewidth=1.5)
        landmarks = atlas.material_to_world(landmark_xi, knee_deg=knee_deg, shape=shape)
        for p, point, color in zip(LANDMARK_PROGRESS, landmarks, LANDMARK_COLORS):
            ax.scatter(*point, color=color, s=35, depthshade=False)
            ax.text(*point, f"  p={p:.2f}", color=color, fontsize=8)
        _set_axes_equal(ax, np.vstack((skin, centerline, landmarks)))
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=22, azim=-70)
    legend = [
        Line2D([0], [0], color="#111111", lw=1.5, label="atlas centerline"),
        *[Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=7,
                 label=f"same material address p={p:.2f}")
          for p, c in zip(LANDMARK_PROGRESS, LANDMARK_COLORS)],
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        f"Atlas full-skin geometry and invariant material landmarks — {backend_label}",
        y=1.03,
    )
    fig.tight_layout()
    return _write_figure(fig, output_dir, "atlas_geometry.png")


def _gamma_curves(run: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cells_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for cell in run["four_cells"]:
        key = (cell["stratum"], int(cell["y"]), int(cell["mode"]))
        if key in cells_by_key:
            raise ReportInputError(f"Duplicate four-cell entry for {key}")
        cells_by_key[key] = cell

    gamma: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for stratum in run["strata"]:
        entries = {(y, mode): cells_by_key.get((stratum, y, mode)) for y in (-1, 1) for mode in (-1, 1)}
        counts = {f"y{y:+d}_mode{mode:+d}": (int(c["n_objects"]) if c is not None else 0)
                  for (y, mode), c in entries.items()}
        metadata[stratum] = {"cell_n_objects": counts, "included": False}
        if any(c is None or int(c["n_objects"]) <= 0 for c in entries.values()):
            continue
        # Gamma(T) is the four-cell interaction contrast:
        # f(+,+) - f(+,-) - f(-,+) + f(-,-).
        pp = _finite_curve(entries[(1, 1)]["finish"], label=f"{stratum} y+ mode+")
        pm = _finite_curve(entries[(1, -1)]["finish"], label=f"{stratum} y+ mode-")
        mp = _finite_curve(entries[(-1, 1)]["finish"], label=f"{stratum} y- mode+")
        mm = _finite_curve(entries[(-1, -1)]["finish"], label=f"{stratum} y- mode-")
        gamma[stratum] = pp - pm - mp + mm
        metadata[stratum]["included"] = True
        metadata[stratum]["formula"] = "f(+,+) - f(+,-) - f(-,+) + f(-,-)"
        metadata[stratum]["gamma_T"] = [float(v) for v in gamma[stratum]]
    return gamma, metadata


def _plot_gamma(run: dict[str, Any], output_dir: Path) -> tuple[str | None, dict[str, Any]]:
    gamma, metadata = _gamma_curves(run)
    if not gamma:
        return None, metadata
    fig, ax = plt.subplots(figsize=(9, 6))
    for stratum in run["strata"]:
        curve = gamma.get(stratum)
        if curve is not None:
            ax.plot(run["deadlines"], curve, linewidth=1.8, label=_friendly_stratum(stratum))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time budget T (s)")
    ax.set_ylabel("Gamma(T) four-cell interaction")
    ax.set_title("Measured four-cell Gamma(T) — only strata with all ± cells")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _write_figure(fig, output_dir, "gamma_T_four_cell.png"), metadata


def _metrics_summary(
    run: dict[str, Any],
    metric_rows: dict[tuple[str, str], dict[str, Any]],
    plots: Iterable[str],
    gamma_metadata: dict[str, Any],
) -> dict[str, Any]:
    nested_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    auc_measured: dict[str, dict[str, float | None]] = {}
    pred_iae: dict[str, dict[str, float | None]] = {}
    pred_max: dict[str, dict[str, float | None]] = {}
    mean_abs: dict[str, dict[str, float | None]] = {}
    for stratum in run["strata"]:
        nested_metrics[stratum] = {}
        auc_measured[stratum] = {}
        pred_iae[stratum] = {}
        pred_max[stratum] = {}
        mean_abs[stratum] = {}
        for method in run["methods"]:
            metric = metric_rows.get((stratum, method))
            if metric is None:
                continue
            safe = {
                key: (int(value) if key == "n_independent_objects" else value)
                for key, value in metric.items()
            }
            nested_metrics[stratum][method] = safe
            auc_measured[stratum][method] = metric["auc_measured"]
            pred_iae[stratum][method] = metric["prediction_integrated_absolute_error"]
            pred_max[stratum][method] = metric["prediction_max_absolute_error"]
            mean_abs[stratum][method] = metric["mean_absolute_error"]

    variants = _budget_variant_info(run)
    # The panel has two explicit group-level summaries for each budget variant:
    # test_new_pose and test_new_shape.  Missing variants remain missing.
    budget_errors: dict[str, dict[str, float | None]] = {}
    for method in BUDGET_VARIANTS:
        if method not in variants:
            continue
        budget_errors[method] = {}
        for label, stratum in (("test_new_pose", "heldout_pose"), ("test_new_shape", "heldout_shape")):
            metric = metric_rows.get((stratum, method))
            budget_errors[method][label] = metric["mean_absolute_error"] if metric is not None else None
        budget_errors[method]["nrecords"] = variants[method]["nrecords"]

    # Group mean pointwise error uses explicit object-count weights.  It is
    # stored separately from the integrated absolute error to avoid conflating
    # probability-seconds with a dimensionless average error.
    group_mean: dict[str, float | None] = {}
    group_weights: dict[str, float] = {}
    for method in run["methods"]:
        vals: list[float] = []
        weights: list[float] = []
        for stratum in run["strata"]:
            metric = metric_rows.get((stratum, method))
            if metric is not None:
                vals.append(metric["mean_absolute_error"])
                weights.append(float(metric["n_independent_objects"]))
        group_mean[method] = float(np.average(vals, weights=weights)) if vals and sum(weights) > 0 else None
        group_weights[method] = float(sum(weights))

    return {
        "geometry_backend": run["geometry_backend"],
        "surrogate_label": run["geometry_label"],
        "evidence_scope": {
            "real_robot": False,
            "b_mode_ultrasound": False,
            "note": (
                "These run outputs use the native SMPL-X geometry backend, not real robot or B-mode evidence."
                if run["geometry_backend"] == "native_smplx"
                else "These run outputs use an independent geometry proxy, not real robot or B-mode evidence."
            ),
        },
        "source_files": {
            "summary": str(run["summary_path"]),
            "evaluation": str(run["evaluation_path"]),
            "four_cell_measured": str(run["four_cell_path"]),
        },
        "deadline_grid_seconds": [float(v) for v in run["deadlines"]],
        "strata": run["strata"],
        "methods": run["methods"],
        "curve_metrics": nested_metrics,
        "auc_measured_over_deadline_budget_grid": auc_measured,
        "prediction_integrated_absolute_error": pred_iae,
        "prediction_max_absolute_error": pred_max,
        "mean_absolute_error": mean_abs,
        "group_mean_absolute_error": group_mean,
        "group_mean_weight_definition": "n_independent_objects",
        "group_mean_total_weights": group_weights,
        "weighted_method_differences_vs_geometry": _weighted_differences(run, metric_rows),
        "training_budget_variants": variants,
        "training_budget_group_errors": budget_errors,
        "four_cell_gamma_T": gamma_metadata,
        "plots": list(plots),
    }


def build_report(input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build all available data-backed plots and return the saved metrics."""

    run = _load_run(Path(input_dir))
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    metric_rows = _curve_metrics(run)
    plots: list[str] = []
    plots.append(_plot_finish_curves(run, output))
    plots.append(_plot_completion_errors(run, metric_rows, output))
    plots.append(_plot_atlas_geometry(run, output))
    gamma_plot, gamma_metadata = _plot_gamma(run, output)
    if gamma_plot is not None:
        plots.append(gamma_plot)

    metrics = _metrics_summary(run, metric_rows, plots, gamma_metadata)
    metrics_path = output / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="run_experiment output directory")
    parser.add_argument("--output", required=True, help="directory for PNGs and metrics_summary.json")
    args = parser.parse_args(argv)
    try:
        metrics = build_report(args.input, args.output)
    except (ReportInputError, FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": str(Path(args.output).resolve()), "plots": metrics["plots"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

