"""Independent Genesis review for the two-beta lower-chain rest-fit shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import trimesh

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    JOINT_SPECS,
    _measure_frames,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import (
    load_chain_rest_fit_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_chain_rest_fit_shadow_v1 import (
    EXPECTED_MODEL_SHA256,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


COLORS = {
    "skin": (0.90, 0.58, 0.43, 0.18),
    "baseline": (0.15, 0.38, 0.95, 0.24),
    "candidate": (0.91, 0.86, 0.73, 1.0),
    "station": (0.92, 0.03, 0.72, 1.0),
    "pivot": (0.02, 0.82, 0.86, 1.0),
    "axis": (0.96, 0.76, 0.02, 1.0),
    "residual": (0.95, 0.03, 0.02, 1.0),
}
LOWER_NAMES = (
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export(path: Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
        validate=False,
    )
    mesh.export(path)
    return path


def _cylinder_between(first: np.ndarray, second: np.ndarray, radius: float) -> trimesh.Trimesh:
    vector = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-8:
        return trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
    align = trimesh.geometry.align_vectors((0.0, 0.0, 1.0), vector / length)
    transform = np.asarray(align, dtype=np.float64)
    transform[:3, 3] = 0.5 * (np.asarray(first) + np.asarray(second))
    mesh.apply_transform(transform)
    return mesh


def _markers(
    frames: np.ndarray,
    stations: np.ndarray,
    joint_names: tuple[str, ...],
) -> dict[str, trimesh.Trimesh]:
    spheres_station = []
    spheres_pivot = []
    axes = []
    residuals = []
    for frame, station, joint_name in zip(frames, stations, joint_names):
        pivot = frame[:3, 3]
        axis = frame[:3, 0]
        closest = (
            pivot
            if joint_name.endswith("hip")
            else pivot + float(np.dot(station - pivot, axis)) * axis
        )
        station_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.005)
        station_sphere.apply_translation(station)
        pivot_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.0045)
        pivot_sphere.apply_translation(pivot)
        spheres_station.append(station_sphere)
        spheres_pivot.append(pivot_sphere)
        if not joint_name.endswith("hip"):
            axes.append(_cylinder_between(pivot - 0.055 * axis, pivot + 0.055 * axis, 0.0015))
        residuals.append(_cylinder_between(station, closest, 0.0013))
    return {
        "station": trimesh.util.concatenate(spheres_station),
        "pivot": trimesh.util.concatenate(spheres_pivot),
        "axis": trimesh.util.concatenate(axes),
        "residual": trimesh.util.concatenate(residuals),
    }


def _save_modalities(
    root: Path,
    rendered: dict[str, dict[str, Any]],
    camera_manifest: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    from PIL import Image, ImageDraw

    records = []
    for camera, payload in rendered.items():
        rgb = np.asarray(payload["rgb"])
        if rgb.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0
            rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)
        rgb = rgb[..., :3]
        camera_spec = camera_manifest[camera]
        camera_distance = float(
            np.linalg.norm(
                np.asarray(camera_spec["pos"], dtype=np.float64)
                - np.asarray(camera_spec["lookat"], dtype=np.float64)
            )
        )
        lookat_plane_height = 2.0 * camera_distance * np.tan(
            np.radians(float(camera_spec["fov"])) * 0.5
        )
        scale_bar_pixels = max(
            2, int(round(0.010 / lookat_plane_height * float(rgb.shape[0])))
        )
        annotated = Image.fromarray(rgb)
        draw = ImageDraw.Draw(annotated)
        x0, y0 = 22, int(rgb.shape[0]) - 27
        x1 = x0 + scale_bar_pixels
        draw.line((x0 - 1, y0, x1 + 1, y0), fill=(0, 0, 0), width=7)
        draw.line((x0, y0, x1, y0), fill=(255, 255, 255), width=3)
        draw.line((x0, y0 - 4, x0, y0 + 4), fill=(255, 255, 255), width=2)
        draw.line((x1, y0 - 4, x1, y0 + 4), fill=(255, 255, 255), width=2)
        draw.text((x0, y0 - 18), "10 mm", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        rgb = np.asarray(annotated)
        depth = np.asarray(payload["depth"], dtype=np.float32).squeeze()
        segmentation = np.asarray(payload["segmentation"])
        segmentation_mask = (
            np.any(segmentation != 0, axis=-1)
            if segmentation.ndim == 3 else segmentation != 0
        )
        rgb_path = root / "rgb" / f"{camera}.png"
        depth_path = root / "depth" / f"{camera}.npy"
        segmentation_path = root / "segmentation" / f"{camera}.png"
        for path in (rgb_path, depth_path, segmentation_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(rgb_path, rgb)
        np.save(depth_path, depth, allow_pickle=False)
        imageio.imwrite(segmentation_path, segmentation_mask.astype(np.uint8) * 255)
        depth_valid = np.isfinite(depth) & (depth > 0.0)
        records.append(
            {
                "camera": camera,
                "rgb": str(rgb_path),
                "rgb_sha256": _sha256(rgb_path),
                "depth": str(depth_path),
                "depth_sha256": _sha256(depth_path),
                "segmentation": str(segmentation_path),
                "segmentation_sha256": _sha256(segmentation_path),
                "foreground_fraction": float(np.mean(segmentation_mask)),
                "depth_valid_fraction": float(np.mean(depth_valid)),
                "scale_bar_m": 0.010,
                "scale_bar_pixels_at_lookat_plane": int(scale_bar_pixels),
                "scale_bar_projection": "perspective_at_frozen_camera_lookat_plane",
                "pass": bool(np.any(segmentation_mask) and np.any(depth_valid)),
            }
        )
    return records


def _contact_sheet(
    paths: list[Path], labels: list[str], output: Path, metrics: dict[str, str]
) -> Path:
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGB") for path in paths]
    columns = 4
    thumb = (400, 300)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb[0], rows * 340), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        image.thumbnail(thumb)
        x = index % columns * thumb[0]
        y = index // columns * 340
        sheet.paste(image, (x, y))
        text = label + (" | " + metrics[label] if label in metrics else "")
        draw.text((x + 5, y + 304), text, fill=(238, 238, 238))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def _render_subject(
    *, value: Any, operator: Any, calibration: Any, model: Any,
    model_sha: str, subject_path: Path, output: Path, backend: str,
) -> dict[str, Any]:
    from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
        GenesisPlatformRuntime,
        GenesisRuntimeConfig,
        MeshEntityConfig,
        StaticCameraConfig,
    )

    asset = operator.template_asset
    final_validation, _widths, _details = _measure_frames(
        value.vertices_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    lookup = {spec.name: index for index, spec in enumerate(JOINT_SPECS)}
    lower_indices = [lookup[name] for name in LOWER_NAMES]
    frames = final_validation[lower_indices]
    station_ids = (1, 2, 4, 5, 7, 8)
    stations = (
        value.smplx_joints_tpose[np.asarray(station_ids, dtype=np.int64)]
        + value.station_frame_translation.reshape(1, 3)
    )
    skin, skin_faces = smplx_body_surface_v7(
        model,
        betas=value.betas,
        pose_axis_angle=np.zeros((55, 3), dtype=np.float64),
    )
    # The beta-prefit anatomy and SMPL-X skin are already in the same canonical
    # metric frame.  station_frame_translation maps motion stations only; it
    # must not be applied a second time to the rendered skin surface.

    context_ids = np.zeros(len(value.vertices_final), dtype=bool)
    context_ids[value.moved_vertex_ids] = True
    for name, (start, stop) in zip(
        asset.source_mesh_names, np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    ):
        if name in {"Ilium_L", "Ilium_R", "Sacrum"}:
            context_ids[int(start):int(stop)] = True
    rows = np.all(context_ids[np.asarray(value.faces, dtype=np.int64)], axis=1)
    assets = output / "mesh_assets"
    entities = [
        ("smplx_skin", _export(assets / "skin.obj", skin, skin_faces), COLORS["skin"]),
        ("baseline_142", _export(assets / "baseline.obj", value.vertices_prefit, value.faces[rows]), COLORS["baseline"]),
        ("candidate", _export(assets / "candidate.obj", value.vertices_final, value.faces[rows]), COLORS["candidate"]),
    ]
    marker_meshes = _markers(frames, stations, LOWER_NAMES)
    for name, mesh in marker_meshes.items():
        marker_path = assets / f"{name}.obj"
        mesh.export(marker_path)
        entities.append((name, marker_path, COLORS[name]))

    cameras = []
    camera_manifest = {}
    center = 0.5 * (np.min(stations, axis=0) + np.max(stations, axis=0))
    lower_height = float(np.max(stations[:, 1]) - np.min(stations[:, 1])) + 0.32
    overview_distance = max(1.20, 0.52 * lower_height / np.tan(np.radians(18.0)))
    camera_manifest["overview_ap"] = {
        "pos": (float(center[0]), float(center[1]), float(center[2] + overview_distance)),
        "lookat": tuple(float(v) for v in center),
        "up": (0.0, 1.0, 0.0), "fov": 36.0,
    }
    camera_manifest["overview_oblique"] = {
        "pos": tuple(
            float(v)
            for v in center
            + np.asarray((0.72 * overview_distance, 0.12, 0.72 * overview_distance))
        ),
        "lookat": tuple(float(v) for v in center),
        "up": (0.0, 1.0, 0.0), "fov": 36.0,
    }
    for name, station in zip(LOWER_NAMES, stations):
        distance = 0.55 if name.endswith("hip") else 0.38
        for view, offset in (
            ("ap", np.asarray((0.0, 0.0, distance))),
            ("lateral", np.asarray((distance, 0.0, 0.0))),
            ("oblique", np.asarray((0.72 * distance, 0.08 * distance, 0.72 * distance))),
        ):
            camera_manifest[f"{name}_{view}"] = {
                "pos": tuple(float(v) for v in station + offset),
                "lookat": tuple(float(v) for v in station),
                "up": (0.0, 1.0, 0.0), "fov": 34.0,
            }
    for name, spec in camera_manifest.items():
        cameras.append(
            StaticCameraConfig(
                name=name, res=(640, 480), pos=spec["pos"], lookat=spec["lookat"],
                up=spec["up"], fov=spec["fov"], near=0.01, far=10.0, gui=False,
            )
        )

    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=backend, show_viewer=False, show_fps=False,
            enable_collision=False, gravity=(0.0, 0.0, 0.0),
            plane_reflection=False, ambient_light=(0.42, 0.42, 0.42),
        )
    )
    try:
        runtime.initialize()
        for name, path, color in entities:
            runtime.add_mesh_entity(
                MeshEntityConfig(
                    name=name, file=path, color=color, fixed=True, collision=False
                )
            )
        for camera in cameras:
            runtime.add_camera(camera)
        runtime.build()
        rendered = runtime.render_all_cameras(
            modalities=("rgb", "depth", "segmentation"), force_render=True
        )
        records = _save_modalities(output, rendered, camera_manifest)
    finally:
        runtime.close()
    if not all(record["pass"] for record in records):
        raise ValueError("Genesis lower-chain render has an empty modality")
    metrics = {}
    check = json.loads((subject_path.parent / f"subject_{value.subject_label}_check.json").read_text())
    for name in LOWER_NAMES:
        joint = check["joints"][name]
        if name.endswith("hip"):
            joint_index = LOWER_NAMES.index(name)
            raw_station = value.smplx_joints_tpose[station_ids[joint_index]]
            raw_to_pivot_mm = float(
                np.linalg.norm(raw_station - frames[joint_index, :3, 3]) * 1000.0
            )
            metric = (
                f"head/socket={joint['head_socket_error_m'] * 1000:.2f}mm "
                f"raw-hip->pivot(report-only)={raw_to_pivot_mm:.2f}mm"
            )
        else:
            metric = (
                f"station-axis={joint['mapped_station_to_axis_m'] * 1000:.2f}mm "
                f"axis={joint['fit_validation_axis_error_deg']:.2f}deg"
            )
        metrics[f"{name}_ap"] = metric
        metrics[f"{name}_lateral"] = metric
        metrics[f"{name}_oblique"] = metric
    rgb_paths = [Path(record["rgb"]) for record in records]
    labels = [record["camera"] for record in records]
    sheet = _contact_sheet(
        rgb_paths, labels, output / f"subject_{value.subject_label}_contact_sheet.png", metrics
    )
    return {
        "subject": value.subject_label,
        "subject_content_digest": json.loads((subject_path / "manifest.json").read_text())["content_digest"],
        "candidate_frames_used_for_camera": False,
        "candidate_bbox_used_for_camera": False,
        "camera_source": "mapped_smplx_lower_stations_only",
        "scale_bar_m": 0.010,
        "camera_manifest": camera_manifest,
        "renders": records,
        "contact_sheet": str(sheet),
        "contact_sheet_sha256": _sha256(sheet),
        "publishable": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Genesis review: {output}")
    output.mkdir(parents=True)
    operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration, operator=operator, required_scope="lower_chain"
    )
    model_path = args.smplx_model.expanduser().resolve()
    model_sha = _sha256(model_path)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise ValueError("Genesis review model differs from the frozen model")
    model = load_smplx_model_v7(model_path)
    matrix = args.matrix.expanduser().resolve()
    reports = []
    for label in ("213328", "213712"):
        subject_path = matrix / f"subject_{label}"
        value = load_chain_rest_fit_v1(
            subject_path,
            operator=operator,
            calibration=calibration,
            smplx_model=model,
            smplx_model_sha256=model_sha,
            recheck=True,
        )
        reports.append(
            _render_subject(
                value=value, operator=operator, calibration=calibration,
                model=model, model_sha=model_sha, subject_path=subject_path,
                output=output / label, backend=args.backend,
            )
        )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "ChainRestFitGenesisReviewV1",
        "matrix": str(matrix),
        "subjects": reports,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_sha = _sha256(manifest_path)
    (output / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="utf-8"
    )
    print(f"ChainRestFitGenesisReviewV1 subjects=2 -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
