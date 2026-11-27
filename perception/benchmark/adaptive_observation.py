"""Causal, source-clock filtering of measured 3D landmarks.

The speed-dependent cutoff follows the One Euro filter. Confidence and
visibility remain measured; newly visible points start from the current
measurement rather than an earlier hidden location.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationFilterState:
    previous: np.ndarray
    filtered: np.ndarray
    velocity: np.ndarray
    valid: np.ndarray
    timestamp_ns: int


class AdaptiveObservationFilter:
    def __init__(self, *, min_cutoff_hz: float = 1.0, speed_beta: float = 20.0,
                 min_confidence: float = .05):
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.speed_beta = float(speed_beta)
        self.min_confidence = float(min_confidence)
        if not np.isfinite(self.min_cutoff_hz) or self.min_cutoff_hz <= 0:
            raise ValueError("min_cutoff_hz must be positive and finite")
        if not np.isfinite(self.speed_beta) or self.speed_beta < 0:
            raise ValueError("speed_beta must be nonnegative and finite")
        self.state: ObservationFilterState | None = None

    def snapshot(self) -> ObservationFilterState | None:
        state = self.state
        if state is None:
            return None
        return ObservationFilterState(*(value.copy() for value in
            (state.previous, state.filtered, state.velocity, state.valid)), state.timestamp_ns)

    def restore(self, state: ObservationFilterState | None) -> None:
        self.state = state

    def apply(self, points: np.ndarray, timestamp_ns: int | None) -> np.ndarray:
        result = np.asarray(points, dtype=np.float32).copy()
        if result.shape != (67, 4):
            raise ValueError("Expected measured landmarks [67,4]")
        # Untimed initialization/warmup must not seed tracking history.
        if timestamp_ns is None:
            return result
        valid = np.isfinite(result).all(axis=1) & (result[:, 3] > self.min_confidence)
        xyz = np.where(valid[:, None], result[:, :3], 0)
        old = self.state
        if old is None:
            filtered, velocity = xyz.copy(), np.zeros_like(xyz)
        else:
            dt = (int(timestamp_ns) - old.timestamp_ns) * 1e-9
            if dt <= 0:
                raise ValueError("Observation timestamp_ns must increase strictly")
            continuing = valid & old.valid
            derivative = (xyz - old.previous) / dt
            derivative_alpha = 1 / (1 + 1 / (2 * np.pi * dt))
            velocity = derivative_alpha * derivative + (1 - derivative_alpha) * old.velocity
            cutoff = self.min_cutoff_hz + self.speed_beta * np.linalg.norm(velocity, axis=1)
            alpha = 1 / (1 + 1 / (2 * np.pi * cutoff * dt))
            filtered = alpha[:, None] * xyz + (1 - alpha[:, None]) * old.filtered
            filtered[~continuing] = xyz[~continuing]
            velocity[~continuing] = 0
        result[valid, :3] = filtered[valid]
        self.state = ObservationFilterState(xyz.copy(), filtered.copy(), velocity.copy(),
                                            valid.copy(), int(timestamp_ns))
        return result
