"""Per-machine crop presets (search window, resolution, starting box)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MachinePreset:
    id: str
    name: str
    frame_width: int
    frame_height: int
    init_cbox: list[int]
    final_cbox: list[int]
    jpeg_quality: int = 80
    hflip: bool = False
    color: bool = False


SONOSCAPE_E2 = MachinePreset(
    id="sonoscape_e2",
    name="SonoScape E2",
    frame_width=1920,
    frame_height=1080,
    # Search window used by auto-crop (HDMI B-mode lives inside this).
    init_cbox=[550, 1650, 150, 920],
    # Tuned starting crop: convex apex ~115, linear/convex width from live E2.
    final_cbox=[559, 1611, 115, 920],
    jpeg_quality=80,
)

PRESETS: tuple[MachinePreset, ...] = (SONOSCAPE_E2,)
DEFAULT_PRESET_ID = SONOSCAPE_E2.id


def list_presets() -> list[MachinePreset]:
    return list(PRESETS)


def get_preset(preset_id: str) -> MachinePreset | None:
    key = str(preset_id).strip().lower()
    for preset in PRESETS:
        if preset.id == key or preset.name.lower() == key:
            return preset
    return None
