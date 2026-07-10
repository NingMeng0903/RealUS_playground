from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence


@dataclass(frozen=True)
class CameraViewFrame:
    camera_id: str
    image_path: Path
    timestamp_ns: int
    intrinsics: list[list[float]] | None = None
    extrinsics: list[list[float]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiViewHumanRecoveryRequest:
    sequence_id: str
    views: dict[str, list[CameraViewFrame]]
    world_frame: str = "world"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiViewHumanRecoveryResult:
    sequence_id: str
    motion_sequence: HumanMotionSequence
    world_frame: str = "world"
    diagnostics: dict[str, Any] = field(default_factory=dict)
