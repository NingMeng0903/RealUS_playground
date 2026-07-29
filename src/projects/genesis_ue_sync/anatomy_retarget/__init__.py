"""Offline anatomy retargeting utilities for the Genesis/SMPL-X pipeline.

The public compatibility exports are lazy.  Importing a focused runtime
submodule must not initialize the optional drawer/tracking/geometry stack;
that cost otherwise dominates a short-lived V8 pose or beta process.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULE = {
    "AnatomyRiggedAsset": ".rigged_asset",
    "load_rigged_asset": ".rigged_asset",
    "save_rigged_asset": ".rigged_asset",
    "AnatomyLbsDrawer": ".anatomy_drawer",
    "AnatomyAssetRegistry": ".genesis_control",
    "AnatomyAssetSubscriber": ".genesis_control",
    "SourceOperatorV7": ".v7_artifacts",
    "SubjectAssetV7": ".v7_artifacts",
    "apply_subject_pose": ".v7_artifacts",
    "load_source_operator": ".v7_artifacts",
    "load_subject_asset": ".v7_artifacts",
    "materialize_subject": ".v7_artifacts",
    "save_source_operator": ".v7_artifacts",
    "save_subject_asset": ".v7_artifacts",
    "apply_tube_material_frames_v7": ".tube_frames_v7",
    "bake_tube_material_frames_v7": ".tube_frames_v7",
    "tube_material_frame_metrics_v7": ".tube_frames_v7",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "AnatomyAssetRegistry",
    "AnatomyAssetSubscriber",
    "AnatomyLbsDrawer",
    "AnatomyRiggedAsset",
    "SourceOperatorV7",
    "SubjectAssetV7",
    "apply_subject_pose",
    "apply_tube_material_frames_v7",
    "bake_tube_material_frames_v7",
    "load_rigged_asset",
    "load_source_operator",
    "load_subject_asset",
    "materialize_subject",
    "save_rigged_asset",
    "save_source_operator",
    "save_subject_asset",
    "tube_material_frame_metrics_v7",
]
