"""Render candidate-independent Node-1 anatomical calibration diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    JOINT_SPECS,
    _measure_frames,
    build_anatomical_calibration_v1,
    check_anatomical_calibration_v1,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_BLEND_SHA256,
    EXPECTED_ORACLE_SHA256,
    EXPECTED_OPERATOR_RUNTIME_DIGEST,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


SCHEMA_VERSION = 1
COLORS = {
    "bone": "#e8dcc5",
    "fit": "#2979ff",
    "validation": "#00d9e8",
    "station": "#e000d4",
    "controller": "#54e68b",
    "axis": "#ffd400",
    "residual": "#ef2929",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project(points: np.ndarray, view: str) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if view == "ap":
        return value[:, (0, 1)]
    if view == "lateral":
        return value[:, (2, 1)]
    raise ValueError(view)


def _ids_for_joint(calibration: Any, joint_index: int, partition: str) -> np.ndarray:
    values = []
    for base in calibration.joint_domain_bases[joint_index].tolist():
        if str(base):
            values.append(np.asarray(calibration.domains[f"{base}.{partition}"], dtype=np.int64))
    return np.unique(np.concatenate(values))


def _mesh_ids_for_domains(asset: Any, ids: np.ndarray) -> np.ndarray:
    selected = []
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    for start, stop in ranges:
        if np.any((ids >= int(start)) & (ids < int(stop))):
            stride = max(1, int(np.ceil((int(stop) - int(start)) / 3500)))
            selected.append(np.arange(int(start), int(stop), stride, dtype=np.int64))
    return np.unique(np.concatenate(selected)) if selected else ids


def _closest_axis_point(point: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return origin + float(np.dot(point - origin, axis)) * axis


def _draw_tile(
    ax: Any,
    *,
    vertices: np.ndarray,
    bone_ids: np.ndarray,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray,
    pivot: np.ndarray,
    axis: np.ndarray,
    station: np.ndarray,
    controller: np.ndarray,
    view: str,
    title: str,
    metric: str,
    camera: dict[str, Any],
    extra_centers: np.ndarray | None = None,
) -> None:
    bone_2d = _project(vertices[bone_ids], view)
    fit_2d = _project(vertices[fit_ids], view)
    validation_2d = _project(vertices[validation_ids], view)
    pivot_2d = _project(pivot.reshape(1, 3), view)[0]
    station_2d = _project(station.reshape(1, 3), view)[0]
    controller_2d = _project(controller.reshape(1, 3), view)[0]
    half = 0.055
    axis_end = np.stack((pivot - half * axis, pivot + half * axis), axis=0)
    axis_2d = _project(axis_end, view)
    residual_end = _closest_axis_point(station, pivot, axis)
    residual_2d = _project(np.stack((station, residual_end), axis=0), view)
    controller_residual_2d = _project(np.stack((controller, pivot), axis=0), view)

    ax.scatter(bone_2d[:, 0], bone_2d[:, 1], s=0.35, c=COLORS["bone"], alpha=0.22)
    ax.scatter(fit_2d[:, 0], fit_2d[:, 1], s=4.0, c=COLORS["fit"], alpha=0.85)
    ax.scatter(validation_2d[:, 0], validation_2d[:, 1], s=4.0, c=COLORS["validation"], alpha=0.85)
    ax.plot(axis_2d[:, 0], axis_2d[:, 1], color=COLORS["axis"], linewidth=2.0)
    ax.plot(residual_2d[:, 0], residual_2d[:, 1], color=COLORS["residual"], linewidth=1.5)
    ax.plot(
        controller_residual_2d[:, 0], controller_residual_2d[:, 1],
        color=COLORS["controller"], linewidth=1.0, linestyle="--",
    )
    ax.scatter(*pivot_2d, s=38, c=COLORS["validation"], edgecolors="black", linewidths=0.4)
    ax.scatter(*station_2d, s=34, c=COLORS["station"], marker="x", linewidths=1.8)
    ax.scatter(*controller_2d, s=26, c=COLORS["controller"], marker="+", linewidths=1.5)
    if extra_centers is not None:
        centers_2d = _project(np.asarray(extra_centers, dtype=np.float64), view)
        ax.scatter(
            centers_2d[:, 0], centers_2d[:, 1], s=46,
            facecolors="none", edgecolors=("white", "#ff9f1c"), linewidths=1.4,
        )
    x0, x1 = camera["xlim"]
    y0, y1 = camera["ylim"]
    ax.plot((x0 + 0.05 * (x1 - x0), x0 + 0.05 * (x1 - x0) + 0.010),
            (y0 + 0.06 * (y1 - y0), y0 + 0.06 * (y1 - y0)),
            color="white", linewidth=2.5)
    ax.text(x0 + 0.05 * (x1 - x0), y0 + 0.085 * (y1 - y0), "10 mm", color="white", fontsize=7)
    ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal")
    ax.set_title(title, color="white")
    ax.text(0.01, 0.01, metric, transform=ax.transAxes, fontsize=6.5, color="white", va="bottom")
    ax.set_facecolor("#15191f")
    ax.tick_params(labelsize=6, colors="#aeb5bf")
    for spine in ax.spines.values():
        spine.set_color("#59616c")


def _camera_from_validation(vertices: np.ndarray, ids: np.ndarray, view: str) -> dict[str, Any]:
    points = _project(vertices[ids], view)
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    extent = np.maximum(upper - lower, 0.020)
    center = 0.5 * (lower + upper)
    half = 0.75 * float(np.max(extent)) + 0.025
    return {"xlim": [float(center[0] - half), float(center[0] + half)],
            "ylim": [float(center[1] - half), float(center[1] + half)]}


def render_review(
    *, operator_path: Path, calibration_path: Path, output_path: Path
) -> Path:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite review: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    operator = load_source_operator(operator_path, mmap=True)
    if operator.runtime_digest(validate=False) != EXPECTED_OPERATOR_RUNTIME_DIGEST:
        raise ValueError("review requires the frozen 142 operator")
    candidate = load_anatomical_calibration_v1(
        calibration_path, operator=operator, required_scope="lower_chain"
    )
    check = check_anatomical_calibration_v1(candidate, operator=operator)
    if not check["passed_lower_chain"]:
        raise ValueError("review refuses a lower-chain calibration that failed checking")
    expected = build_anatomical_calibration_v1(
        operator,
        source_blend_sha256=EXPECTED_BLEND_SHA256,
        blender_oracle_sha256=EXPECTED_ORACLE_SHA256,
    )
    asset = operator.template_asset
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    validation_frames, _widths, details = _measure_frames(
        vertices, expected.domains, expected.joint_domain_bases, partition="validation"
    )

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))
    camera_manifest: dict[str, Any] = {}
    images: dict[str, str] = {}
    try:
        for kind in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist"):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
            for column, side in enumerate(("left", "right")):
                index = next(
                    i for i, spec in enumerate(JOINT_SPECS)
                    if spec.kind == kind and spec.side == side
                )
                fit_ids = _ids_for_joint(expected, index, "fit")
                validation_ids = _ids_for_joint(expected, index, "validation")
                bone_ids = _mesh_ids_for_domains(asset, np.union1d(fit_ids, validation_ids))
                pivot = validation_frames[index, :3, 3]
                axis = validation_frames[index, :3, 0]
                station = expected.station_rest_global[index, :3, 3]
                controller = expected.controller_rest_global[index, :3, 3]
                joint = check["joints"][JOINT_SPECS[index].name]
                metric = (
                    f"validation-derived | raw_station_gate=false\n"
                    f"center={joint['fit_validation_center_error_m'] * 1000:.3f} mm | "
                    f"axis={joint['fit_validation_axis_error_deg']:.3f} deg"
                )
                if kind == "hip":
                    metric += (
                        f" | head/socket={details[index]['head_socket_error_m'] * 1000:.3f} mm"
                    )
                    extra_centers = np.asarray(
                        (
                            details[index]["head_center_m"],
                            details[index]["socket_center_m"],
                        ),
                        dtype=np.float64,
                    )
                else:
                    extra_centers = None
                for row, view in enumerate(("ap", "lateral")):
                    camera = _camera_from_validation(vertices, validation_ids, view)
                    camera_manifest[f"{kind}/{side}/{view}"] = camera
                    _draw_tile(
                        axes[row, column], vertices=vertices, bone_ids=bone_ids,
                        fit_ids=fit_ids, validation_ids=validation_ids,
                        pivot=pivot, axis=axis, station=station, view=view,
                        controller=controller,
                        title=f"{side} {kind} — {view.upper()}", metric=metric,
                        camera=camera, extra_centers=extra_centers,
                    )
            fig.suptitle(
                "Node 1 frozen-142 validation | fit blue | validation/pivot cyan | "
                "axis yellow | raw station magenta | controller green | residual red",
                fontsize=10, color="white",
            )
            image_path = temporary / f"{kind}_bilateral_contact_sheet.png"
            fig.savefig(image_path, dpi=150, facecolor="#0d1015")
            plt.close(fig)
            images[image_path.name] = _sha256(image_path)

        fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
        all_ids = np.unique(np.concatenate([
            _ids_for_joint(expected, index, "validation")
            for index in range(len(JOINT_SPECS))
        ]))
        source_ids = np.arange(0, len(vertices), max(1, len(vertices) // 50000), dtype=np.int64)
        source_2d = _project(vertices[source_ids], "ap")
        ax.scatter(source_2d[:, 0], source_2d[:, 1], s=0.15, c=COLORS["bone"], alpha=0.12)
        for index, spec in enumerate(JOINT_SPECS):
            pivot = validation_frames[index, :3, 3]
            station = expected.station_rest_global[index, :3, 3]
            controller = expected.controller_rest_global[index, :3, 3]
            line = _project(np.stack((pivot, station)), "ap")
            ax.plot(line[:, 0], line[:, 1], c=COLORS["residual"], linewidth=0.7)
            ax.scatter(*_project(pivot.reshape(1, 3), "ap")[0], c=COLORS["validation"], s=18)
            ax.scatter(*_project(station.reshape(1, 3), "ap")[0], c=COLORS["station"], s=18, marker="x")
            ax.scatter(*_project(controller.reshape(1, 3), "ap")[0], c=COLORS["controller"], s=16, marker="+")
            label = _project(pivot.reshape(1, 3), "ap")[0]
            ax.text(label[0], label[1], spec.name, color="white", fontsize=5)
        camera = _camera_from_validation(vertices, all_ids, "ap")
        ax.set(xlim=camera["xlim"], ylim=camera["ylim"], aspect="equal")
        ax.set_facecolor("#15191f")
        ax.set_title("Frozen 142 Node 1 overview — validation-derived frames", color="white")
        camera_manifest["overview/ap"] = camera
        overview = temporary / "overview_ap.png"
        fig.savefig(overview, dpi=150, facecolor="#0d1015")
        plt.close(fig)
        images[overview.name] = _sha256(overview)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "AnatomicalCalibrationReviewV1",
            "candidate_calibration_digest": check["calibration_digest"],
            "source_operator_digest": operator.runtime_digest(validate=False),
            "domain_source": "deterministically_rebuilt_from_frozen_142_operator",
            "camera_source": "frozen_validation_domain_bbox_only",
            "candidate_frames_used": False,
            "candidate_bbox_used": False,
            "raw_station_gate": False,
            "derived_source_ids": ["shoulder", "wrist"],
            "images": images,
            "camera_manifest": camera_manifest,
            "publishable": False,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
    except Exception:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = render_review(
        operator_path=args.operator.expanduser().resolve(),
        calibration_path=args.calibration.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
    )
    print(f"AnatomicalCalibrationReviewV1 -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
