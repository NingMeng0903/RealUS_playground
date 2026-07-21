"""Shared constants and RM75 loader used by GPU reachability GT builders."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np

CLEARANCE_INTERIOR = -1
CLEARANCE_POSITION = 0
CLEARANCE_ROTATION = 1


def block_ids(features: np.ndarray, width_m: float) -> np.ndarray:
    ijk = np.floor(features[:, :3] / float(width_m)).astype(np.int64)
    return ijk[:, 0] * 1_000_000_000_000 + ijk[:, 1] * 1_000_000 + ijk[:, 2]


def reachability_modules():
    try:
        from rm75_control.tools.reachability.kinematics.ik_dls import ik_dls, ik_dls_multiseed
        from rm75_control.tools.reachability.kinematics.ik_seeds import SeedPoolConfig, build_seed_pool, halton_matrix
        from rm75_control.tools.reachability.build.self_collision import SelfCollisionFilter
        from rm75_control.tools.reachability.kinematics.model_locked_rail import build_locked_rail_model
        return ik_dls, ik_dls_multiseed, SeedPoolConfig, build_seed_pool, halton_matrix, SelfCollisionFilter, build_locked_rail_model
    except ModuleNotFoundError as exc:
        if exc.name not in {"Robotic_Arm", "Robotic_Arm.rm_ctypes_wrap"}:
            raise
    for name in list(sys.modules):
        if name == "rm75_control" or name.startswith("rm75_control."):
            sys.modules.pop(name, None)
    package_root = Path(__file__).resolve().parents[3] / "rm75_control" / "rm75_control"
    package = types.ModuleType("rm75_control")
    package.__path__ = [str(package_root)]
    sys.modules["rm75_control"] = package
    from rm75_control.tools.reachability.kinematics.ik_dls import ik_dls, ik_dls_multiseed
    from rm75_control.tools.reachability.kinematics.ik_seeds import SeedPoolConfig, build_seed_pool, halton_matrix
    from rm75_control.tools.reachability.build.self_collision import SelfCollisionFilter
    from rm75_control.tools.reachability.kinematics.model_locked_rail import build_locked_rail_model
    return ik_dls, ik_dls_multiseed, SeedPoolConfig, build_seed_pool, halton_matrix, SelfCollisionFilter, build_locked_rail_model


__all__ = [
    "CLEARANCE_INTERIOR",
    "CLEARANCE_POSITION",
    "CLEARANCE_ROTATION",
    "block_ids",
    "reachability_modules",
]
