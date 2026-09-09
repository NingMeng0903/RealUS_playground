"""Normal-axis speed envelope shared by the active force controllers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PressEnvelopeConfig:
    """Linear-region press envelope.  Caps are runtime, not certificates."""

    # Zero disables an override; YAML parsing retains the established defaults.
    soft_approach_m_s: float = 0.0
    first_touch_m_s: float = 0.0
    max_force_axis_m_s: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict) -> "PressEnvelopeConfig":
        root = raw if isinstance(raw, dict) else {}
        controller = root.get("hybrid_motion", root.get("controller", root))
        if not isinstance(controller, dict):
            controller = root
        block = controller.get("press_envelope", root.get("press_envelope", {}))
        if not isinstance(block, dict):
            block = {}
        return cls(
            soft_approach_m_s=float(block.get("soft_approach_m_s", 0.020)),
            first_touch_m_s=float(block.get("first_touch_m_s", 0.012)),
            max_force_axis_m_s=float(block.get("max_force_axis_m_s", 0.0)),
        )
