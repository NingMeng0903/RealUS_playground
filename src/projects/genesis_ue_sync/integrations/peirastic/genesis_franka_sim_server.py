"""Backward-compatible entrypoint for PEIRASTIC-protocol Genesis simulation.

Prefer importing ``GenesisRobotSimPeirasticBridge`` from ``integrations.controller_bus`` for new code."""

from __future__ import annotations

from projects.genesis_ue_sync.integrations.controller_bus.peirastic_robot_sim_bridge import (
    GenesisRobotSimPeirasticBridge,
    GenesisRobotSimPeirasticConfig,
    ensure_peirastic_on_path,
)

GenesisFrankaSimConfig = GenesisRobotSimPeirasticConfig
GenesisFrankaSimServer = GenesisRobotSimPeirasticBridge

__all__ = [
    "GenesisFrankaSimConfig",
    "GenesisFrankaSimServer",
    "GenesisRobotSimPeirasticBridge",
    "GenesisRobotSimPeirasticConfig",
    "ensure_peirastic_on_path",
]
