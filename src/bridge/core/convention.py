from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CanonicalConvention:
    name: str
    handedness: str
    up_axis: str
    units: str
    image_origin: str | None = None
    camera_forward_axis: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


CANONICAL_GENESIS_CONVENTION = CanonicalConvention(
    name='genesis_world',
    handedness='right',
    up_axis='z',
    units='meters',
    image_origin='top_left',
    camera_forward_axis='+z',
    notes=(
        'Canonical project frame is Genesis-aligned world.',
        'Internal bridge math keeps a metric right-handed Z-up world.',
    ),
)

UE_WORLD_CONVENTION = CanonicalConvention(
    name='ue_world',
    handedness='left',
    up_axis='z',
    units='meters',
    notes=(
        'UE authoring is left-handed with X forward, Y right, Z up.',
    ),
)

OPENCV_CAMERA_CONVENTION = CanonicalConvention(
    name='opencv_camera',
    handedness='right',
    up_axis='-y',
    units='pixels_and_meters',
    image_origin='top_left',
    camera_forward_axis='+z',
    notes=(
        'Camera axes are x right, y down, z forward.',
    ),
)

BLENDER_WORLD_CONVENTION = CanonicalConvention(
    name='blender_world',
    handedness='right',
    up_axis='z',
    units='meters',
    notes=(
        'Blender world is right-handed Z-up.',
        'Blender camera/object local axes may still need adapter-specific basis handling.',
    ),
)

BEDLAM_WORLD_CONVENTION = CanonicalConvention(
    name='bedlam_unreal_world',
    handedness='left',
    up_axis='z',
    units='meters',
    notes=(
        'BEDLAM world data is authored on top of Unreal conventions.',
        'SMPL pelvis and shape offsets are a semantic caveat, not a generic axis-flip issue.',
    ),
)
