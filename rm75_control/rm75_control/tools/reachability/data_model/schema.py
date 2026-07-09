"""Config dataclasses and the on-disk file layout for the capability map.

Only pure-Python types live here so that `data_model` stays cheap to import
(no Pinocchio, no PyVista).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class VoxelGridConfig:
    """Axis-aligned 3-D voxel grid in the arm-base (rail_base @ rail_y=0) frame.

    Defaults cover the RM75-6F reachable ball (~1.09 m arm reach) with 3 cm
    voxels; ~312 k cells before sparse filtering, ~90 k reachable.
    """

    origin_m: tuple[float, float, float] = (-1.10, -1.10, -0.30)
    step_m: float = 0.03
    shape: tuple[int, int, int] = (74, 74, 57)


@dataclass(frozen=True)
class OrientationGridConfig:
    """5-DOF tool-axis discretisation on the unit sphere.

    ``kind='icosphere'`` with ``subdiv=3`` → 642 vertices, mean spacing ~11°
    (finer than the user-requested 15°, chosen because Fibonacci with N=200 is
    a close alternative but has a slight bias at the poles).
    """

    kind: Literal["icosphere", "fibonacci"] = "icosphere"
    subdiv: int = 3
    fibonacci_n: int = 200


@dataclass(frozen=True)
class RollGridConfig:
    """Optional 6-DOF add-on: roll around the tool axis.

    Default ``enabled=False`` follows Zacharias (2013): 5-DOF is enough for the
    canonical D(x) reachability index and keeps the bitmask ~24× smaller.
    """

    enabled: bool = False
    step_deg: float = 15.0


@dataclass(frozen=True)
class BitmaskLayout:
    """How ``bitmask.npy`` is packed on disk.

    For a 5-DOF map with ``n_orient`` tool-axis samples::

        bitmask.shape == (n_voxels, ceil(n_orient / 8))  # uint8

    For a 6-DOF map with roll grid of size ``n_roll``::

        bitmask.shape == (n_voxels, n_orient, ceil(n_roll / 8))  # uint8

    ``voxels.npz`` always carries the (n_voxels, 3) int32 ``ijk`` array and the
    (n_voxels,) float32 ``d_value`` D(x) reachability index alongside optional
    mean manipulability etc.
    """

    n_orient: int
    n_roll: int = 0

    @property
    def per_voxel_bytes(self) -> int:
        if self.n_roll == 0:
            return (self.n_orient + 7) // 8
        return self.n_orient * ((self.n_roll + 7) // 8)


@dataclass
class MapMeta:
    """Free-form metadata persisted to ``manifest.yaml`` next to the map."""

    schema_version: str = "1.0"
    urdf_path: str = ""
    urdf_sha256: str = ""
    tcp_frame: str = "tcp"
    git_sha: str = ""
    build_wall_s: float = 0.0
    mc_samples: int = 0
    ik_refined: bool = False
    with_roll: bool = False
    extra: dict = field(default_factory=dict)
