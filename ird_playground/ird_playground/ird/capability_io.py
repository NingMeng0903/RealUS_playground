"""File-format CapabilityMap loader (no rm75_control package import).

Reads the same on-disk layout as ``rm75_control.tools.reachability`` CapabilityMap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class SimpleVoxelGrid:
    origin_m: np.ndarray
    step_m: float
    shape: tuple[int, int, int]

    def center_of(self, ijk: np.ndarray) -> np.ndarray:
        arr = np.asarray(ijk, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        c = self.origin_m[None, :] + self.step_m * (arr + 0.5)
        return c[0] if single else c


@dataclass
class SimpleOrientations:
    vectors: np.ndarray

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])


@dataclass
class LoadedCapabilityMap:
    grid: SimpleVoxelGrid
    orientations: SimpleOrientations
    roll: object | None
    voxel_ids: np.ndarray
    bitmask: np.ndarray
    d_value: np.ndarray
    mu_mean: np.ndarray | None
    n_orient: int
    manifest: dict


def unpack_bits_5dof(packed: np.ndarray, n_orient: int) -> np.ndarray:
    m, n_bytes = packed.shape
    out = np.zeros((m, n_bytes * 8), dtype=bool)
    for k in range(8):
        out[:, k::8] = ((packed >> k) & 1).astype(bool)
    return out[:, :n_orient]


def load_capability_map_dir(map_dir: str | Path, *, mmap: bool = True) -> LoadedCapabilityMap:
    p = Path(map_dir)
    manifest = yaml.safe_load((p / "manifest.yaml").read_text(encoding="utf-8"))
    g = manifest["grid"]
    grid = SimpleVoxelGrid(
        origin_m=np.asarray(g["origin_m"], dtype=np.float64),
        step_m=float(g["step_m"]),
        shape=tuple(int(s) for s in g["shape"]),
    )
    vectors = np.load(p / "orientations.npy").astype(np.float64)
    voxels = np.load(p / "voxels.npz")
    bitmask = np.load(p / "bitmask.npy", mmap_mode=("r" if mmap else None))
    mu = voxels["mu_mean"] if "mu_mean" in voxels.files else None
    n_orient = int(manifest["layout"]["n_orient"])
    roll = manifest.get("roll")
    return LoadedCapabilityMap(
        grid=grid,
        orientations=SimpleOrientations(vectors=vectors),
        roll=roll,
        voxel_ids=voxels["ijk"].astype(np.int32),
        bitmask=bitmask,
        d_value=voxels["d_value"].astype(np.float32),
        mu_mean=(mu.astype(np.float32) if mu is not None else None),
        n_orient=n_orient,
        manifest=manifest,
    )
