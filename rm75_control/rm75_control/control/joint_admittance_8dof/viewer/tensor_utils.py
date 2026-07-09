"""Convert Genesis / torch DOF buffers to host numpy (CUDA backend returns cuda tensors)."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    """Same pattern as Among_US ``GenesisPlatformRuntime._to_numpy``."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
