from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


def host_clock_fields(*, source_time_ns: int | None = None) -> dict[str, int]:
    """Host CLOCK_REALTIME stamp for live streams (ZMQ / SHM / files)."""
    wall_time_ns = int(time.time_ns())
    source = wall_time_ns if source_time_ns is None else int(source_time_ns)
    return {
        "source_time_ns": source,
        "sim_time_ns": source,
        "wall_time_ns": wall_time_ns,
    }


@dataclass(frozen=True)
class ClockSnapshot:
    sim_step_index: int
    sim_time_ns: int
    wall_time_ns: int
    source_time_ns: int
    clock_domain: str = "genesis_sim"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sim_step_index": int(self.sim_step_index),
            "sim_time_ns": int(self.sim_time_ns),
            "wall_time_ns": int(self.wall_time_ns),
            "source_time_ns": int(self.source_time_ns),
            "clock_domain": str(self.clock_domain),
        }


class ClockService:
    """Canonical timebase for Genesis-led sessions.

    Genesis simulation time is authoritative by default. External systems such as ROS or hardware
    drivers should pass their own timestamp as source_time_ns while keeping sim_time_ns in this domain.
    """

    def __init__(self, *, dt: float, clock_domain: str = "genesis_sim") -> None:
        self.dt = float(dt)
        self.clock_domain = str(clock_domain or "genesis_sim")

    @classmethod
    def from_runtime(cls, runtime: Any) -> "ClockService":
        cfg = getattr(runtime, "config", None)
        return cls(
            dt=float(getattr(cfg, "dt", 0.01) or 0.01),
            clock_domain=str(getattr(cfg, "clock_domain", "genesis_sim") or "genesis_sim"),
        )

    def sim_time_ns_for_step(self, step_index: int) -> int:
        dt_ns = int(max(float(self.dt), 1e-9) * 1e9)
        return int(step_index) * dt_ns

    def snapshot(self, step_index: int, *, source_time_ns: int | None = None) -> ClockSnapshot:
        wall_ns = time.time_ns()
        return ClockSnapshot(
            sim_step_index=int(step_index),
            sim_time_ns=self.sim_time_ns_for_step(int(step_index)),
            wall_time_ns=int(wall_ns),
            source_time_ns=int(wall_ns if source_time_ns is None else source_time_ns),
            clock_domain=self.clock_domain,
        )
