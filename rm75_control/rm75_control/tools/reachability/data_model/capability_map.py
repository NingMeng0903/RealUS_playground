"""In-memory capability map plus save / load (mmap) helpers.

On-disk layout under ``<dir>/``::

    manifest.yaml       - free-form MapMeta + all grid configs (human-readable)
    voxels.npz          - ijk (M,3) int32, d_value (M,) float32, mu_mean opt.
    orientations.npy    - (n_orient, 3) float32 unit vectors
    faces.npy           - (F, 3) int64 icosphere faces (optional; direction viz)
    bitmask.npy         - packed uint8 array, see BitmaskLayout

Design:

* The voxel array is *sparse* – we only store voxels that received at least one
  reachable orientation during the build, indexed by their (i, j, k) tuple.
* Query is by (i, j, k); a compact hash map from ijk → row index is rebuilt on
  load (dense in memory but O(M) rather than O(nx*ny*nz)).
* ``bitmask.npy`` is opened with ``np.load(..., mmap_mode='r')`` so multiple
  processes share pages and the on-disk file drives resident memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from rm75_control.tools.reachability.data_model.orientation_grid import (
    IcosphereToolAxisGrid,
    RollGrid,
    ToolAxisGrid,
)
from rm75_control.tools.reachability.data_model.schema import (
    BitmaskLayout,
    MapMeta,
    OrientationGridConfig,
    RollGridConfig,
    VoxelGridConfig,
)
from rm75_control.tools.reachability.data_model.voxel_grid import VoxelGrid


@dataclass
class CapabilityMap:
    grid: VoxelGrid
    orientations: ToolAxisGrid
    roll: RollGrid | None
    layout: BitmaskLayout
    voxel_ids: np.ndarray  # (M, 3) int32 sparse indices
    bitmask: np.ndarray    # (M, ceil(N_ori/8)) uint8, or (M, N_ori, ceil(N_roll/8))
    d_value: np.ndarray    # (M,) float32
    mu_mean: np.ndarray | None = None  # (M,) float32
    meta: MapMeta = field(default_factory=MapMeta)
    _lookup: dict[tuple[int, int, int], int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._lookup = {
            (int(a), int(b), int(c)): row
            for row, (a, b, c) in enumerate(self.voxel_ids)
        }

    # ---- query -------------------------------------------------------------
    @property
    def n_reachable_voxels(self) -> int:
        return int(self.voxel_ids.shape[0])

    def row_of(self, ijk: tuple[int, int, int]) -> int | None:
        return self._lookup.get((int(ijk[0]), int(ijk[1]), int(ijk[2])))

    def is_reachable(self, ijk: tuple[int, int, int], orient_idx: int) -> bool:
        row = self.row_of(ijk)
        if row is None:
            return False
        if self.roll is None:
            byte, bit = divmod(int(orient_idx), 8)
            return bool(self.bitmask[row, byte] & (1 << bit))
        # 6-DOF: any roll reachable → True
        return bool(np.any(self.bitmask[row, int(orient_idx)]))

    def any_orient_reachable(
        self, ijk: tuple[int, int, int], orient_indices: np.ndarray
    ) -> bool:
        """True if at least one of the given orientation indices is reachable."""
        row = self.row_of(ijk)
        if row is None:
            return False
        if self.roll is None:
            bytes_ = orient_indices >> 3
            bits = (1 << (orient_indices & 7)).astype(np.uint8)
            return bool(np.any(self.bitmask[row, bytes_] & bits))
        return bool(np.any(self.bitmask[row, orient_indices]))

    def d_grid(self) -> np.ndarray:
        """Dense (nx, ny, nz) D(x) with NaN where the voxel was not visited."""
        nx, ny, nz = self.grid.shape
        out = np.full((nx, ny, nz), np.nan, dtype=np.float32)
        i, j, k = self.voxel_ids[:, 0], self.voxel_ids[:, 1], self.voxel_ids[:, 2]
        out[i, j, k] = self.d_value
        return out

    # ---- IO ----------------------------------------------------------------
    def save(self, dir_path: str | Path) -> Path:
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": self.meta.schema_version,
            "urdf_path": self.meta.urdf_path,
            "urdf_sha256": self.meta.urdf_sha256,
            "tcp_frame": self.meta.tcp_frame,
            "git_sha": self.meta.git_sha,
            "build_wall_s": float(self.meta.build_wall_s),
            "mc_samples": int(self.meta.mc_samples),
            "ik_refined": bool(self.meta.ik_refined),
            "with_roll": bool(self.meta.with_roll),
            "extra": dict(self.meta.extra),
            "grid": {
                "origin_m": [float(x) for x in self.grid.origin_m],
                "step_m": float(self.grid.step_m),
                "shape": list(self.grid.shape),
            },
            "orientations": {
                "kind": "icosphere" if isinstance(self.orientations, IcosphereToolAxisGrid) else "fibonacci",
                "n": self.orientations.n,
                "subdiv": getattr(self.orientations, "subdiv", None),
            },
            "roll": None if self.roll is None else {"step_deg": self.roll.step_deg, "n": self.roll.n},
            "layout": {"n_orient": self.layout.n_orient, "n_roll": self.layout.n_roll},
            "counts": {
                "n_reachable_voxels": int(self.voxel_ids.shape[0]),
                "n_grid_voxels": int(self.grid.n_voxels),
            },
        }
        (p / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        d_arrays: dict[str, np.ndarray] = {
            "ijk": self.voxel_ids.astype(np.int32),
            "d_value": self.d_value.astype(np.float32),
        }
        if self.mu_mean is not None:
            d_arrays["mu_mean"] = self.mu_mean.astype(np.float32)
        np.savez(p / "voxels.npz", **d_arrays)
        np.save(p / "orientations.npy", self.orientations.vectors.astype(np.float32))
        if isinstance(self.orientations, IcosphereToolAxisGrid):
            np.save(p / "faces.npy", self.orientations.faces.astype(np.int64))
        np.save(p / "bitmask.npy", self.bitmask.astype(np.uint8))
        return p

    @classmethod
    def load(cls, dir_path: str | Path, *, mmap: bool = True) -> "CapabilityMap":
        p = Path(dir_path)
        manifest = yaml.safe_load((p / "manifest.yaml").read_text())
        g = manifest["grid"]
        grid = VoxelGrid(
            origin_m=np.asarray(g["origin_m"], dtype=np.float64),
            step_m=float(g["step_m"]),
            shape=tuple(int(s) for s in g["shape"]),  # type: ignore[arg-type]
        )
        o = manifest["orientations"]
        vectors = np.load(p / "orientations.npy").astype(np.float64)
        if o["kind"] == "icosphere":
            faces_path = p / "faces.npy"
            faces = np.load(faces_path).astype(np.int64) if faces_path.exists() else np.zeros((0, 3), np.int64)
            orient: ToolAxisGrid = IcosphereToolAxisGrid(
                vectors=vectors, faces=faces, subdiv=int(o.get("subdiv") or 0)
            )
        else:
            orient = ToolAxisGrid(vectors=vectors)
        roll_cfg = manifest.get("roll")
        roll = None if roll_cfg is None else RollGrid(step_deg=float(roll_cfg["step_deg"]))
        layout = BitmaskLayout(
            n_orient=int(manifest["layout"]["n_orient"]),
            n_roll=int(manifest["layout"]["n_roll"]),
        )
        voxels = np.load(p / "voxels.npz")
        mmap_mode = "r" if mmap else None
        bitmask = np.load(p / "bitmask.npy", mmap_mode=mmap_mode)
        mu = voxels["mu_mean"] if "mu_mean" in voxels.files else None
        meta = MapMeta(
            schema_version=str(manifest.get("schema_version", "1.0")),
            urdf_path=str(manifest.get("urdf_path", "")),
            urdf_sha256=str(manifest.get("urdf_sha256", "")),
            tcp_frame=str(manifest.get("tcp_frame", "tcp")),
            git_sha=str(manifest.get("git_sha", "")),
            build_wall_s=float(manifest.get("build_wall_s", 0.0)),
            mc_samples=int(manifest.get("mc_samples", 0)),
            ik_refined=bool(manifest.get("ik_refined", False)),
            with_roll=bool(manifest.get("with_roll", False)),
            extra=dict(manifest.get("extra", {}) or {}),
        )
        return cls(
            grid=grid,
            orientations=orient,
            roll=roll,
            layout=layout,
            voxel_ids=voxels["ijk"].astype(np.int32),
            bitmask=bitmask,
            d_value=voxels["d_value"].astype(np.float32),
            mu_mean=(mu.astype(np.float32) if mu is not None else None),
            meta=meta,
        )


# ---------------------------------------------------------------------------
# Bitmask packing helpers (used by the builder + tests)
# ---------------------------------------------------------------------------
def pack_bits_5dof(bool_matrix: np.ndarray) -> np.ndarray:
    """(M, n_orient) bool → (M, ceil(n_orient/8)) uint8 little-bit-endian.

    Bit ``k`` inside byte ``b`` corresponds to ``orient_idx = 8*b + k``.
    """
    if bool_matrix.dtype != np.bool_:
        bool_matrix = bool_matrix.astype(bool)
    m, n_orient = bool_matrix.shape
    n_bytes = (n_orient + 7) // 8
    padded = np.zeros((m, n_bytes * 8), dtype=bool)
    padded[:, :n_orient] = bool_matrix
    packed = np.zeros((m, n_bytes), dtype=np.uint8)
    for k in range(8):
        packed |= (padded[:, k::8].astype(np.uint8) << k)
    return packed


def unpack_bits_5dof(packed: np.ndarray, n_orient: int) -> np.ndarray:
    """Inverse of :func:`pack_bits_5dof`."""
    m, n_bytes = packed.shape
    out = np.zeros((m, n_bytes * 8), dtype=bool)
    for k in range(8):
        out[:, k::8] = ((packed >> k) & 1).astype(bool)
    return out[:, :n_orient]


def d_value_from_bitmask(packed: np.ndarray, n_orient: int) -> np.ndarray:
    """Reachability index D(x) = (# reachable orientations) / n_orient."""
    counts = np.zeros(packed.shape[0], dtype=np.int32)
    for k in range(8):
        counts += ((packed >> k) & 1).sum(axis=1).astype(np.int32)
    # trim last-byte padding
    if n_orient % 8 != 0:
        overshoot = (packed.shape[1] * 8) - n_orient
        # subtract padding bits (they are always 0 by construction of pack_bits_5dof)
        del overshoot  # kept as a comment marker; padding is zeros so no correction needed
    return (counts.astype(np.float32) / float(n_orient)).astype(np.float32)
