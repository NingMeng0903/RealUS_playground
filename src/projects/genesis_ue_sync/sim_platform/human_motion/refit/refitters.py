"""Compatibility shells for retired refitters.

The runtime path is now the implicit kinodynamic engine implemented by
:class:`ImplicitKinodynamicRefitController` and exposed at the package root.
Legacy SMPL-only sliding-window / Adam refitters are intentionally removed —
the shells below keep the old import names alive so accidental imports fail
loudly instead of silently activating a stale optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _RemovedRefitter:
    name: str

    def __post_init__(self) -> None:
        raise RuntimeError(
            f"{self.name} was removed. Use ImplicitKinodynamicRefitController "
            "(`from projects.genesis_ue_sync.sim_platform.human_motion.refit import "
            "ImplicitKinodynamicRefitController`)."
        )


class SimplePhysicsRefitter(_RemovedRefitter):  # noqa: D401 - back-compat shell
    """Removed; replaced by the implicit kinodynamic refit."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__(name="SimplePhysicsRefitter")

    def refit(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover - guard path
        raise RuntimeError("SimplePhysicsRefitter was removed.")


@dataclass(frozen=True)
class HamiltonianLossWeights:
    """Stub kept so callers reading old configs do not fail on import."""

    tracking_root_xy: float = 0.0
    tracking_root_z: float = 0.0
    tracking_global_orient: float = 0.0
    tracking_spine: float = 0.0
    tracking_limbs: float = 0.0
    tracking_hands: float = 0.0
    contact: float = 0.0
    joint_damping: float = 0.0
    temporal_smooth: float = 0.0
    symmetry: float = 0.0
    vposer_prior: float = 0.0
    pd_effort: float = 0.0
    root_drift: float = 0.0
    bed_sdf: float = 0.0
    langevin_temperature: float = 0.0


@dataclass(frozen=True)
class RefitMvpOptions:
    support_margin_m: float = 0.015
    temporal_smooth_lambda: float = 0.05
    joint_limit_rad: float = 3.141592653589793
    lowpass_passes: int = 1
    loss_weights: HamiltonianLossWeights = HamiltonianLossWeights()


def removed_refit_entry_point(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "The MVP refit entry point was removed. Use ImplicitKinodynamicRefitController."
    )


__all__ = [
    "HamiltonianLossWeights",
    "RefitMvpOptions",
    "SimplePhysicsRefitter",
    "removed_refit_entry_point",
]
