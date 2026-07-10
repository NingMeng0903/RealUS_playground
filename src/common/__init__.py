"""Common reusable helpers for the reorganized workspace."""

from common.project import ProjectPaths, discover_project_root, project_paths
from common.types import MotionSequenceManifest

__all__ = [
    "MotionSequenceManifest",
    "ProjectPaths",
    "discover_project_root",
    "lower_shell_mask",
    "project_paths",
    "support_plane_shift",
    "support_plane_shift_masked",
]


def __getattr__(name: str):
    if name in {"lower_shell_mask", "support_plane_shift", "support_plane_shift_masked"}:
        from common import geometry_support as _geometry_support

        return getattr(_geometry_support, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
