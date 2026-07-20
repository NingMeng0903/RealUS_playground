"""Feature conventions for Point IRD (6-D baseline / 8-D roll-ready)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    kind: str = "pu6"
    dim: int = 6
    use_roll: bool = False
    roll_harmonics: tuple[int, ...] = (1, 2, 4, 8)
    surface_frame_version: str = "btn_v1"
    tool_axis: str = "tcp_plus_z"


def make_feature_spec(kind: str = "pu6") -> FeatureSpec:
    kind = str(kind).lower().strip()
    if kind in {"pu6", "pu", "6", "natural_pu"}:
        return FeatureSpec(kind="pu6", dim=6, use_roll=False)
    if kind in {"pu_roll8", "pu8", "8", "natural_pu_roll"}:
        return FeatureSpec(kind="pu_roll8", dim=8, use_roll=True)
    raise ValueError(f"unsupported feature kind: {kind!r}")
