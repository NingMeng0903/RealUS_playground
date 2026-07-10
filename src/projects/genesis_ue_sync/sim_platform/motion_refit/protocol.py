"""Offline SMPL motion refit contracts (implementation stubs live outside this module)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ContactPlaneConstraint:
    plane_z_m: float
    body_vertex_indices: tuple[int, ...] = ()
    max_penetration_m: float = 0.002
    stiffness: float = 1.0


@dataclass(frozen=True)
class JointLimitConstraint:
    pose_indices: tuple[int, ...]
    lower_rad: np.ndarray
    upper_rad: np.ndarray


@dataclass(frozen=True)
class RefitOptions:
    optimize_global_orient: bool = True
    optimize_body_pose: bool = True
    optimize_transl: bool = True
    temporal_smooth_lambda: float = 0.05
    foot_sliding_lambda: float = 0.0
    capsule_margin_m: float = 0.01


@dataclass(frozen=True)
class RefitRequest:
    """Input bundle for an offline refit pass."""

    sequence_npz_path: Path
    scene_metadata_path: Path | None = None
    human_placement_path: Path | None = None
    constraints: tuple[Any, ...] = ()
    options: RefitOptions = field(default_factory=RefitOptions)
    seed: int = 0


@dataclass(frozen=True)
class RefitResult:
    """Output aligned motion in the same HumanMotionSequence-compatible NPZ layout."""

    output_npz_path: Path
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SmplMotionRefitter(Protocol):
    """Pluggable optimizer (IK, dynamics filter, or QP trajectory smoother)."""

    def refit(self, request: RefitRequest) -> RefitResult:
        ...


def describe_refit_pipeline_stages() -> list[str]:
    """Documented extension order for contributors."""
    return [
        "stage_a_root_lower_body_support_contact",
        "stage_b_joint_limits_temporal_smooth",
        "stage_c_capsule_proxy_nonpenetration",
        "stage_d_full_body_sparse_ik",
    ]
