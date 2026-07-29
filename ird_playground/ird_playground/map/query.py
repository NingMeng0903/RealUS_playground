"""Differentiable sampling of a stored 5-D tensor field."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from ird_playground.map.build_flange_tensor import CHART_NAMES


def _wrap_periodic(values: "torch.Tensor", lo: "torch.Tensor", hi: "torch.Tensor") -> "torch.Tensor":
    span = hi - lo
    return lo + torch.remainder(values - lo, span)


class TensorField(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Differentiable multilinear sampler over a regular 5-D grid."""

    def __init__(
        self,
        values: "torch.Tensor | np.ndarray",
        axes: tuple["torch.Tensor | np.ndarray", ...] | None = None,
        *,
        periodic_axes: tuple[int, ...] = (3, 4),
        device: str | "torch.device | None" = None,
    ) -> None:
        if torch is None:
            raise ImportError("torch required")
        super().__init__()
        if axes is None:
            raise ValueError("axes are required")
        if len(axes) != values.ndim:
            raise ValueError("axes length must match field dimensionality")
        self.periodic_axes = tuple(int(a) for a in periodic_axes)
        axis_tensors = [
            torch.as_tensor(np.asarray(a, dtype=np.float32), device=device) for a in axes
        ]
        for idx, axis in enumerate(axis_tensors):
            self.register_buffer(f"axis_{idx}", axis, persistent=False)
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32, device=device))
        self.ndim = int(self.values.ndim)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        values_key: str = "sdf",
        device: str | None = None,
    ) -> "TensorField":
        blob = np.load(Path(path), allow_pickle=False)
        values = blob[values_key] if values_key in blob.files else blob["occupancy"].astype(np.float32)
        axes = tuple(blob[name] for name in CHART_NAMES if name in blob.files)
        if len(axes) != values.ndim:
            raise ValueError(f"expected {values.ndim} axis arrays in {path}")
        return cls(values, axes, device=device)

    def _axis(self, dim: int) -> "torch.Tensor":
        return getattr(self, f"axis_{dim}")

    def _normalize_coord(self, coord: "torch.Tensor", dim: int) -> "torch.Tensor":
        axis = self._axis(dim)
        lo, hi = axis[0], axis[-1]
        if dim in self.periodic_axes:
            coord = _wrap_periodic(coord, lo, hi)
        coord = coord.clamp(lo, hi)
        idx = (coord - lo) / (hi - lo).clamp_min(1.0e-8)
        return idx * 2.0 - 1.0

    def _sample_axis(
        self,
        field: "torch.Tensor",
        coord: "torch.Tensor",
        dim: int,
    ) -> "torch.Tensor":
        grid = self._normalize_coord(coord, dim)
        while grid.ndim < field.ndim:
            grid = grid.unsqueeze(-1)
        # ``grid_sample`` expects the spatial dimension being resampled last.
        moved = field.movedim(dim, -1).unsqueeze(1)
        sample_grid = grid.unsqueeze(-1)
        out = F.grid_sample(
            moved,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(1).squeeze(-1)
        return out.movedim(-1, dim)

    def score(self, coords: "torch.Tensor") -> "torch.Tensor":
        """Sample clearance at chart coordinates ``(*, 5)`` with autograd support."""
        if coords.shape[-1] != self.ndim:
            raise ValueError(f"expected trailing dim {self.ndim}, got {coords.shape[-1]}")
        field = self.values
        for dim in range(self.ndim):
            field = self._sample_axis(field, coords[..., dim], dim)
        return field

    def score_world(self, T_tcp_world: "torch.Tensor", T_axis_world: "torch.Tensor") -> "torch.Tensor":
        raise NotImplementedError(
            "TensorField stores chart coordinates; map SE(3) poses to chart coords "
            "before calling score()."
        )


__all__ = ["TensorField"]
