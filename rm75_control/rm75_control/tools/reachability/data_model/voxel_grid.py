"""Axis-aligned uniform 3-D voxel grid.

Coordinates are in the arm-base frame (i.e. ``rail_base`` with ``rail_y=0``);
translating the grid along +Y in world = shifting the base y_b, which is the
inversion invariance the online query exploits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.tools.reachability.data_model.schema import VoxelGridConfig


@dataclass(frozen=True)
class VoxelGrid:
    """(nx, ny, nz) uniform grid with origin at ``origin_m`` and cube step ``step_m``.

    Voxel ``(i, j, k)`` covers the AABB
    ``[origin + step*(i,j,k), origin + step*(i+1,j+1,k+1)]`` with centre
    ``origin + step*(i+0.5, j+0.5, k+0.5)``.
    """

    origin_m: np.ndarray  # (3,) float64
    step_m: float
    shape: tuple[int, int, int]

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: VoxelGridConfig) -> "VoxelGrid":
        return cls(
            origin_m=np.asarray(cfg.origin_m, dtype=np.float64),
            step_m=float(cfg.step_m),
            shape=tuple(int(s) for s in cfg.shape),  # type: ignore[arg-type]
        )

    def __post_init__(self) -> None:
        if self.origin_m.shape != (3,):
            raise ValueError(f"origin_m must have shape (3,), got {self.origin_m.shape}")
        if self.step_m <= 0.0:
            raise ValueError(f"step_m must be > 0, got {self.step_m}")
        if len(self.shape) != 3 or any(s <= 0 for s in self.shape):
            raise ValueError(f"shape must be 3 positive ints, got {self.shape}")

    # ---- basics ------------------------------------------------------------
    @property
    def n_voxels(self) -> int:
        return int(self.shape[0] * self.shape[1] * self.shape[2])

    @property
    def bbox_m(self) -> tuple[np.ndarray, np.ndarray]:
        lo = self.origin_m.copy()
        hi = self.origin_m + self.step_m * np.array(self.shape, dtype=np.float64)
        return lo, hi

    # ---- world <-> index ---------------------------------------------------
    def idx_of(self, p: np.ndarray) -> np.ndarray:
        """Return int32 (i,j,k) for one point (3,) or a batch (N,3).

        Uses ``floor((p - origin)/step)``; out-of-range indices are clipped to
        ``[-1, shape]`` so the caller can detect them with :meth:`in_bounds`.
        """
        arr = np.asarray(p, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.shape[-1] != 3:
            raise ValueError(f"expected trailing dim 3, got {arr.shape}")
        rel = (arr - self.origin_m[None, :]) / self.step_m
        ijk = np.floor(rel).astype(np.int32)
        return ijk[0] if single else ijk

    def in_bounds(self, ijk: np.ndarray) -> np.ndarray:
        """Boolean mask (scalar or (N,)) for indices inside the grid."""
        arr = np.asarray(ijk, dtype=np.int32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        s = np.asarray(self.shape, dtype=np.int32)
        ok = np.all((arr >= 0) & (arr < s[None, :]), axis=1)
        return bool(ok[0]) if single else ok

    def center_of(self, ijk: np.ndarray) -> np.ndarray:
        """Voxel centre(s) in metres."""
        arr = np.asarray(ijk, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        c = self.origin_m[None, :] + self.step_m * (arr + 0.5)
        return c[0] if single else c

    def flat(self, ijk: np.ndarray) -> np.ndarray:
        """Row-major flat index; scalar or (N,)."""
        arr = np.asarray(ijk, dtype=np.int64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        ny, nz = self.shape[1], self.shape[2]
        f = arr[:, 0] * (ny * nz) + arr[:, 1] * nz + arr[:, 2]
        return int(f[0]) if single else f

    def unflat(self, flat: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`flat`."""
        f = np.asarray(flat, dtype=np.int64)
        single = f.ndim == 0
        if single:
            f = f[None]
        ny, nz = self.shape[1], self.shape[2]
        i = f // (ny * nz)
        rem = f - i * (ny * nz)
        j = rem // nz
        k = rem - j * nz
        out = np.stack([i, j, k], axis=-1).astype(np.int32)
        return out[0] if single else out

    # ---- helpers -----------------------------------------------------------
    def all_centers(self) -> np.ndarray:
        """(n_voxels, 3) centres, row-major on (i,j,k)."""
        nx, ny, nz = self.shape
        ii, jj, kk = np.meshgrid(
            np.arange(nx, dtype=np.float64),
            np.arange(ny, dtype=np.float64),
            np.arange(nz, dtype=np.float64),
            indexing="ij",
        )
        c = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)
        return self.origin_m[None, :] + self.step_m * (c + 0.5)

    def centers_inside_ball(self, radius_m: float, center: np.ndarray | None = None) -> np.ndarray:
        """Return (M,3) int32 ``ijk`` of voxels whose centre lies within ``radius_m``
        of ``center`` (defaults to origin of the arm base = (0,0,0))."""
        if center is None:
            center = np.zeros(3, dtype=np.float64)
        centers = self.all_centers()
        d2 = np.sum((centers - center[None, :]) ** 2, axis=1)
        mask = d2 <= radius_m * radius_m
        nx, ny, nz = self.shape
        ii, jj, kk = np.meshgrid(
            np.arange(nx, dtype=np.int32),
            np.arange(ny, dtype=np.int32),
            np.arange(nz, dtype=np.int32),
            indexing="ij",
        )
        ijk = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)
        return ijk[mask]
