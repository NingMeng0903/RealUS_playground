"""Minimal ``tracking_result.json`` for Genesis viz when the full pipeline did not finish."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle
from projects.genesis_ue_sync.tracking.pipeline import TrackingPipelineConfig
from projects.genesis_ue_sync.sim_platform.datasets import HumanMotionSequence


def write_minimal_tracking_result_json(cfg: TrackingPipelineConfig, *, overwrite: bool = False) -> Path:
    """Write ``cfg.output_root / tracking_result.json`` if ``motion_sequence.npz`` exists."""
    out = Path(cfg.output_root)
    motion = out / "motion_sequence.npz"
    if not motion.is_file():
        raise FileNotFoundError(f"Missing motion sequence: {motion}. Run U-HMR / tracking first.")

    result_path = out / "tracking_result.json"
    if result_path.is_file() and not overwrite:
        return result_path

    seq = HumanMotionSequence.load(motion)
    pc_dir = out / "pointcloud"
    pc_dir.mkdir(parents=True, exist_ok=True)
    raw_path = pc_dir / "raw_points.npy"
    filt_path = pc_dir / "filtered_points.npy"
    if not raw_path.is_file():
        np.save(raw_path, np.zeros((0, 3), dtype=np.float32))
    if not filt_path.is_file():
        np.save(filt_path, np.zeros((0, 3), dtype=np.float32))
    raw_pts = np.load(raw_path).astype(np.float32).reshape(-1, 3)
    filt_pts = np.load(filt_path).astype(np.float32).reshape(-1, 3)

    cal_json = out / "calibration_bundle.json"
    cal_path_str = str(cal_json) if cal_json.is_file() else str(cfg.calibration_path)
    try:
        bundle = load_calibration_bundle(cfg.calibration_path, scene_spec_path=cfg.scene_spec_path)
        camera_ids = bundle.ordered_camera_ids()
    except Exception:
        camera_ids = []

    result: dict[str, Any] = {
        "config_path": str(cfg.config_path),
        "scene_spec_path": str(cfg.scene_spec_path),
        "calibration_path": str(cfg.calibration_path),
        "calibration_bundle_json": cal_path_str,
        "run_meta_path": str(cfg.run_meta_path),
        "input_fps": float(cfg.input_fps),
        "output_root": str(out),
        "motion_sequence_path": str(motion.resolve()),
        "heatmap_paths": {},
        "vit_videos": {"per_camera": {}, "strip": None},
        "mask_paths": {},
        "mask_meta_path": str(out / "genesis_masks" / "mask_meta.json"),
        "pointcloud": {
            "raw_points_path": str(raw_path.resolve()),
            "filtered_points_path": str(filt_path.resolve()),
            "frame_json": [],
            "raw_count": int(raw_pts.shape[0]),
            "filtered_count": int(filt_pts.shape[0]),
        },
        "frame_count": int(seq.frame_count),
        "camera_ids": camera_ids,
        "pipeline_sampling": {
            "frame_start": int(cfg.frame_start),
            "frame_step": int(cfg.frame_step),
            "frame_limit": cfg.frame_limit,
        },
        "viz_note": "minimal_summary_for_genesis_viz_only",
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result_path


__all__ = ["write_minimal_tracking_result_json"]
