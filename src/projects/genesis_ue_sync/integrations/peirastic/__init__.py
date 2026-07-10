from __future__ import annotations

__all__ = [
    "GenesisFrankaSimConfig",
    "GenesisFrankaSimServer",
    "GenesisRobotSimPeirasticBridge",
    "GenesisRobotSimPeirasticConfig",
    "ensure_peirastic_on_path",
]


def __getattr__(name: str):
    if name in {"GenesisRobotSimPeirasticBridge", "GenesisRobotSimPeirasticConfig", "ensure_peirastic_on_path"}:
        from projects.genesis_ue_sync.integrations.controller_bus import peirastic_robot_sim_bridge as bridge

        return getattr(bridge, name)
    if name in {"GenesisFrankaSimConfig", "GenesisFrankaSimServer"}:
        from projects.genesis_ue_sync.integrations.peirastic import genesis_franka_sim_server as server

        return getattr(server, name)
    raise AttributeError(name)
