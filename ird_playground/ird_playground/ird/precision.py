"""Source-resolution checks for physically meaningful IRD acceptance.

An MLP can interpolate a voxel field smoothly, but it cannot recover a reachability
boundary more accurately than the labels that trained it.  This module makes that
limit explicit in evaluation reports, rather than conflating interpolation error
with physical localization accuracy.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def source_resolution_report(
    gt_npz: str | Path,
    *,
    target_position_error_m: float = 1.0e-4,
) -> dict[str, float | bool | str]:
    """Return a conservative spatial-accuracy gate from GT provenance.

    For occupancy labels sampled at voxel centers, a boundary can lie anywhere
    within the final cell.  Half a voxel width is therefore an optimistic lower
    bound on localizable boundary position; it is not a measured robot error.
    """
    gt_npz = Path(gt_npz).resolve()
    provenance = gt_npz.with_suffix(".yaml")
    if not provenance.is_file():
        return {
            "source_resolution_known": False,
            "source_resolution_pass": False,
            "source_resolution_reason": f"missing GT provenance: {provenance}",
        }

    meta = yaml.safe_load(provenance.read_text(encoding="utf-8")) or {}
    # Continuous FK/IK GT records its verified boundary tolerance directly.
    # Its uncertainty is governed by both bisection termination and the IK
    # residual that decides each side of the bracket, not by a voxel width.
    if str(meta.get("label_kind", "")).startswith("continuous_fk"):
        bisection_m = float(meta.get("physical_boundary_tolerance_m", 0.0))
        ik_m = float(meta.get("ik_position_tolerance_m", 0.0))
        lower_bound_m = max(bisection_m, ik_m)
        if lower_bound_m <= 0.0:
            return {
                "source_resolution_known": False,
                "source_resolution_pass": False,
                "source_resolution_reason": "continuous GT provenance has invalid tolerances",
            }
        target_m = float(target_position_error_m)
        return {
            "source_resolution_known": True,
            "source_kind": "continuous_fk_multiseed_ik",
            "source_boundary_lower_bound_m": lower_bound_m,
            "target_position_error_m": target_m,
            "source_resolution_pass": bool(lower_bound_m <= target_m),
            "source_resolution_reason": "max(IK position residual, SE(3) bisection termination)",
        }
    raw_map = meta.get("map_dir")
    if not raw_map:
        return {
            "source_resolution_known": False,
            "source_resolution_pass": False,
            "source_resolution_reason": "GT provenance has no map_dir",
        }
    map_dir = Path(str(raw_map)).expanduser()
    manifest_path = map_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return {
            "source_resolution_known": False,
            "source_resolution_pass": False,
            "source_resolution_reason": f"missing capability manifest: {manifest_path}",
        }
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    step_m = float((manifest.get("grid") or {}).get("step_m", 0.0))
    if step_m <= 0.0:
        return {
            "source_resolution_known": False,
            "source_resolution_pass": False,
            "source_resolution_reason": "capability manifest has invalid grid.step_m",
        }
    lower_bound_m = 0.5 * step_m
    target_m = float(target_position_error_m)
    return {
        "source_resolution_known": True,
        "source_voxel_step_m": step_m,
        "source_boundary_lower_bound_m": lower_bound_m,
        "target_position_error_m": target_m,
        "source_resolution_pass": bool(lower_bound_m <= target_m),
        "source_resolution_reason": (
            "voxel-center labels: optimistic boundary-localization lower bound is half a voxel"
        ),
    }
