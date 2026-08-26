"""Reload calibration parameters from disk into shared in-memory caches."""
from __future__ import annotations

from multicam_calib.io.results import ExtrinsicsSet, Intrinsics, extrinsics_rel_path, load_extrinsics, load_intrinsics_map


def refresh_intrinsics_cache(cache: dict[str, Intrinsics]) -> None:
    """Replace ``cache`` contents with the latest ``intrinsics.yaml``."""
    fresh = load_intrinsics_map()
    cache.clear()
    cache.update(fresh)


def load_stage1_extrinsics() -> ExtrinsicsSet | None:
    """Read the latest Stage 1 relative extrinsics from disk."""
    return load_extrinsics(extrinsics_rel_path())


def stage1_rmse_label() -> str:
    ext = load_stage1_extrinsics()
    if ext is None:
        return "Stage1: (not run)"
    rmse = ext.metadata.get("total_rmse_px")
    if rmse is None:
        return "Stage1: (no RMSE)"
    mm = ext.metadata.get("board_pose_disagreement_mean_mm")
    try:
        mm_f = float(mm)
    except (TypeError, ValueError):
        mm_f = float("nan")
    if mm is not None and mm_f == mm_f:
        return f"Stage1: {float(rmse):.3f} px / {mm_f:.2f} mm"
    return f"Stage1: {float(rmse):.3f} px"
