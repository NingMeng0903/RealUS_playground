"""1610 ball-screw geometry: mm ↔ motor revolutions + instruction pulses."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PositionCommand:
    """Internal absolute position segment (Pr P1 registers)."""

    revolutions: int
    pulses: int
    speed_rpm: int


def mm_to_position_command(
    travel_mm: float,
    *,
    lead_mm: float = 10.0,
    gear_ratio: float = 1.0,
    pulses_per_rev: int = 10_000,
    speed_rpm: int = 200,
) -> PositionCommand:
    """Convert linear travel (mm) to LW100 internal position command fields.

    Parameters
    ----------
    travel_mm:
        Signed distance along the screw (+/-).
    lead_mm:
        Screw lead in mm/rev (1610 → 10 mm).
    gear_ratio:
        Motor revolutions per screw revolution (1.0 = direct coupling).
    pulses_per_rev:
        ``FA11`` — instruction pulses per motor revolution.
    speed_rpm:
        ``FD-4`` segment speed (r/min).
    """
    if lead_mm <= 0.0:
        raise ValueError(f"lead_mm must be > 0, got {lead_mm}")
    if gear_ratio <= 0.0:
        raise ValueError(f"gear_ratio must be > 0, got {gear_ratio}")
    if pulses_per_rev <= 0:
        raise ValueError(f"pulses_per_rev must be > 0, got {pulses_per_rev}")

    total_revs = (float(travel_mm) / lead_mm) * gear_ratio
    sign = 1 if total_revs >= 0.0 else -1
    abs_revs = abs(total_revs)
    whole = int(math.floor(abs_revs + 1e-12))
    frac = abs_revs - whole
    pulses = int(round(frac * pulses_per_rev))
    if pulses >= pulses_per_rev:
        whole += 1
        pulses = 0
    revolutions = sign * whole
    if sign < 0:
        pulses = -pulses if pulses else 0
    return PositionCommand(revolutions=revolutions, pulses=pulses, speed_rpm=int(speed_rpm))


def position_command_to_mm(
    cmd: PositionCommand,
    *,
    lead_mm: float = 10.0,
    gear_ratio: float = 1.0,
    pulses_per_rev: int = 10_000,
) -> float:
    """Inverse of ``mm_to_position_command`` (approximate for display)."""
    motor_revs = float(cmd.revolutions) + float(cmd.pulses) / float(pulses_per_rev)
    return (motor_revs / gear_ratio) * lead_mm
