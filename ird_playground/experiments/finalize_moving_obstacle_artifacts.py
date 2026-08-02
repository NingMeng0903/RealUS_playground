"""Validate and finalize the moving-ellipsoid release artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


WHITELIST = {
    "summary.json",
    "moving_obstacle_guidance.npz",
    "optimization_history.json",
    "artifact_manifest.json",
    "diffusion_guidance_summary.json",
    "diffusion_guidance_foundation.npz",
    "diffusion_guidance_no_learning.mp4",
    "guidance_recovery_heatmap.png",
    "u_band_moving_obstacle.mp4",
    "u_band_cone_reachability.png",
    "region_ird_field_nearest_along_s.mp4",
    "region_ird_field_optimized_along_s.mp4",
    "region_ird_field_conditioned_along_s.mp4",
    "optimization_conditioned_field_evolution.mp4",
    "region_ird_gradient.png",
    "dual_phase_diagnostics.png",
    "ellipse_section_projection.png",
    "ellipse_trajectory.png",
    "optimized_joint_path.png",
    "qpik_continuity_contrast.png",
    "qpik_joint_guidance.png",
    "trajectory_controls_vs_s.png",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_video(path: Path) -> dict[str, object]:
    reader = imageio.get_reader(path)
    try:
        frame_count = int(reader.count_frames())
        if frame_count < 3:
            raise RuntimeError(f"{path.name} has fewer than three frames")
        indices = (0, frame_count // 2, frame_count - 1)
        frames = [np.asarray(reader.get_data(index)) for index in indices]
    finally:
        reader.close()
    shape = tuple(int(value) for value in frames[0].shape)
    if any(frame.shape != frames[0].shape for frame in frames):
        raise RuntimeError(f"{path.name} changes frame shape")
    if any(not np.isfinite(frame).all() or float(frame.std()) < 1.0 for frame in frames):
        raise RuntimeError(f"{path.name} contains a blank or invalid audit frame")
    return {"frame_count": frame_count, "frame_shape": shape, "audited_frames": indices}


def validate_png(path: Path) -> dict[str, object]:
    image = np.asarray(imageio.imread(path))
    if image.ndim < 2 or min(image.shape[:2]) < 64 or float(image.std()) < 1.0:
        raise RuntimeError(f"{path.name} is blank or too small")
    return {"shape": tuple(int(value) for value in image.shape)}


def validate_release(out: Path) -> dict[str, object]:
    required = WHITELIST - {"artifact_manifest.json"}
    missing = sorted(name for name in required if not (out / name).is_file())
    if missing:
        raise RuntimeError(f"missing release artifacts: {missing}")

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    diffusion = json.loads(
        (out / "diffusion_guidance_summary.json").read_text(encoding="utf-8")
    )
    timing = summary["timing"]
    obstacle = summary["obstacle"]
    reachability = summary["reachability"]
    mask = summary["visualization_mask_audit"]
    if not summary["valid"] or int(timing["benchmark_runs"]) != 30:
        raise RuntimeError("moving summary is not a valid 30-run release")
    if float(timing["warm_request_to_certified_q_p95_seconds"]) > 5.0:
        raise RuntimeError("warm request-to-certified-q P95 exceeds five seconds")
    if float(obstacle["segment_min_tcp_signed_distance_m"]) < 0.003:
        raise RuntimeError("dense ellipsoid margin is below three millimetres")
    if float(reachability["minimum_selected_clearance"]) < 5.0:
        raise RuntimeError("selected raw IRD is below the safety threshold")
    if (
        int(mask["start"]["deep_red_core_points"]) != 0
        or int(mask["end"]["deep_red_core_points"]) != 0
        or int(mask["encounter"]["deep_red_core_points"]) <= 0
        or int(mask["encounter"]["soft_halo_points"]) <= 0
    ):
        raise RuntimeError("conditioned visualization mask audit failed")
    if not diffusion["valid"] or int(diffusion["samples"]) != 48:
        raise RuntimeError("diffusion foundation summary is incomplete")

    moving = np.load(out / "moving_obstacle_guidance.npz")
    foundation = np.load(out / "diffusion_guidance_foundation.npz")
    for key in (
        "T_tcp_ref", "q_ref", "qdot_ff", "rail_ref_m", "ird_clearance",
        "conditioned_clearance", "obstacle_rotation_world", "obstacle_semiaxes_m",
    ):
        if key not in moving:
            raise RuntimeError(f"moving NPZ misses {key}")
    if moving["q_ref"].shape != (81, 8) or not np.isfinite(moving["q_ref"]).all():
        raise RuntimeError("moving NPZ has an invalid rail+7R trajectory")
    if not np.array_equal(moving["rail_ref_m"], moving["q_ref"][:, 0]):
        raise RuntimeError("rail reference and q_ref[:,0] disagree")
    recovered = np.asarray(foundation["recovered"], dtype=bool)
    if recovered.shape != (48,) or int(recovered.sum()) != int(diffusion["success_count"]):
        raise RuntimeError("diffusion NPZ and summary recovery counts disagree")

    media: dict[str, object] = {}
    for path in sorted(out.iterdir()):
        if path.name not in required:
            continue
        if path.suffix.lower() == ".mp4":
            media[path.name] = validate_video(path)
        elif path.suffix.lower() == ".png":
            media[path.name] = validate_png(path)
    return {"media": media, "moving": summary, "diffusion": diffusion}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    validation = validate_release(out)
    extras = sorted(
        path.name for path in out.iterdir() if path.is_file() and path.name not in WHITELIST
    )
    if not args.clean:
        print(json.dumps({
            "validation": {
                "moving_valid": bool(validation["moving"]["valid"]),
                "benchmark_runs": int(validation["moving"]["timing"]["benchmark_runs"]),
                "diffusion_success_count": int(validation["diffusion"]["success_count"]),
                "diffusion_samples": int(validation["diffusion"]["samples"]),
            },
            "rebuildable_extras": extras,
        }, indent=2))
        return 0
    for name in extras:
        (out / name).unlink()

    artifacts = {}
    for name in sorted(WHITELIST - {"artifact_manifest.json"}):
        path = out / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema": "moving-obstacle-ellipsoid-artifacts-v1",
        "directory": str(out),
        "retained_files": sorted(WHITELIST),
        "deleted_rebuildable_files": extras,
        "directory_matches_whitelist": True,
        "validation": {
            "moving_valid": bool(validation["moving"]["valid"]),
            "benchmark_runs": int(validation["moving"]["timing"]["benchmark_runs"]),
            "warm_pipeline_p95_seconds": float(
                validation["moving"]["timing"]["warm_request_to_certified_q_p95_seconds"]
            ),
            "diffusion_success_count": int(validation["diffusion"]["success_count"]),
            "diffusion_samples": int(validation["diffusion"]["samples"]),
            "media": validation["media"],
        },
        "artifacts": artifacts,
        "manifest_hash_note": "artifact_manifest.json omits its own recursive hash",
    }
    (out / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    actual = {path.name for path in out.iterdir() if path.is_file()}
    if actual != WHITELIST:
        raise RuntimeError(
            f"directory does not match whitelist; extra={sorted(actual-WHITELIST)}, "
            f"missing={sorted(WHITELIST-actual)}"
        )
    print(json.dumps(manifest["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
