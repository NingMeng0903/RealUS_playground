"""Shared V8 source-rig FK policy and validation.

New V8 assets use selective source-local FK: only authored articulation links
retain their Blender-local translations, while independently mapped controls
remain anchored to their SMPL-X driver frames.  Legacy full-local-FK assets are
accepted for read compatibility but are never emitted by the selective baker.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


SOURCE_FK_POLICY_KEY_V4 = "source_fk_policy_v4"
SELECTIVE_AUTHORITY_FK_POLICY_V4 = "selective_authority"
LEGACY_FULL_LOCAL_FK_POLICY = "legacy_full_local_fk_v2"


_V71_ARTICULATION_LOCAL_BONE_NAMES_V4 = {
    "left": (
        "Knee_Rotate_L",
        "Tibia_Bone_L",
        "Tibia_Twist_L",
        "Elbow_Rot_L",
        "Forearm_Bone_L",
        "Forearm_Twist_L",
    ),
    "right": (
        "Knee_Rotate_R",
        "Tibia_Bone_R",
        "Tibia_Twist_R",
        "Elbow_Rot_R",
        "Forearm_Bone_R",
        "Forearm_Twist_R",
    ),
}
_V71_LEG_COMPOUND_ROOT_NAMES_V1 = {
    "left": {
        "femur": "Femur_Rot_L",
        "knee": "Knee_Rotate_L",
        "shank": "Tibia_Bone_L",
        "ankle": "Ankle_Rot_L",
        "arch": "Arch_Rot_L",
        "toes_prefix": "Toes_Rotate_L",
    },
    "right": {
        "femur": "Femur_Rot_R",
        "knee": "Knee_Rotate_R",
        "shank": "Tibia_Bone_R",
        "ankle": "Ankle_Rot_R",
        "arch": "Arch_Rot_R",
        "toes_prefix": "Toes_Rotate_R",
    },
}
_V71_LEG_RUNTIME_DRIVER_SEMANTICS_V811 = {
    "left": {
        "femur": ("left_hip", "segment_root"),
        "knee": ("left_knee", "segment_root"),
        "shank": ("left_knee", "segment_root"),
        "ankle": ("left_ankle", "rigid_group"),
        "arch": ("left_foot", "joint_local"),
    },
    "right": {
        "femur": ("right_hip", "segment_root"),
        "knee": ("right_knee", "segment_root"),
        "shank": ("right_knee", "segment_root"),
        "ankle": ("right_ankle", "rigid_group"),
        "arch": ("right_foot", "joint_local"),
    },
}
_V71_LEG_RUNTIME_STATION_LABELS_V811 = (
    "femur",
    "knee",
    "shank",
    "ankle",
    "arch",
)
_SELECTIVE_RUNTIME_LEGACY_KEYS_V811 = (
    "source_leg_hinge_solve_v1",
    "source_knee_hinge_splines_v7",
    "source_tibia_glide_splines_v7",
    "source_patella_v71_response_v8",
)
_SMPLX_HAND_CONTROLLER_PARTS_V4 = frozenset(
    (
        "wrist",
        "thumb1",
        "thumb2",
        "thumb3",
        "index1",
        "index2",
        "index3",
        "middle1",
        "middle2",
        "middle3",
        "ring1",
        "ring2",
        "ring3",
        "pinky1",
        "pinky2",
        "pinky3",
    )
)
_V71_NON_HAND_DIRECT_ANCHOR_NAMES_V4 = frozenset(
    (
        "Spine_C7",
        "Head_Bone",
        "Jaw_Bone_base",
    )
)


def articulation_local_fk_bones_v4(bone_names: Sequence[str] | None) -> list[int]:
    """Return the connected knee/elbow mechanism links which keep local pivots."""

    prefixes = (
        "knee_rotate_",
        "tibia_bone_",
        "tibia_twist_",
        "elbow_rot_",
        "forearm_bone_",
        "forearm_twist_",
    )
    return [
        int(index)
        for index, name in enumerate(bone_names or ())
        if str(name).lower().startswith(prefixes)
    ]


def direct_smplx_hand_controllers_v4(asset: Any) -> list[int]:
    """Return semantic wrist/finger controls, independent of source indices."""

    raw_modes = getattr(asset, "source_bone_driver_types", None)
    raw_mapped = getattr(asset, "source_bone_smplx_a", None)
    raw_joint_names = getattr(asset, "joint_names", None)
    modes = [] if raw_modes is None else list(raw_modes)
    mapped = [] if raw_mapped is None else list(raw_mapped)
    joint_names = [] if raw_joint_names is None else list(raw_joint_names)
    if len(modes) != len(mapped):
        return []
    direct: list[int] = []
    for bone, (mode, joint_id) in enumerate(zip(modes, mapped, strict=True)):
        if str(mode) not in {"joint_local", "direct_joint"}:
            continue
        joint = (
            str(joint_names[int(joint_id)])
            if 0 <= int(joint_id) < len(joint_names)
            else ""
        )
        side, separator, part = joint.partition("_")
        if side not in {"left", "right"} or not separator:
            continue
        if part == "wrist" or part.startswith(
            ("thumb", "index", "middle", "ring", "pinky")
        ):
            direct.append(int(bone))
    return direct


def _required_unique_bone_indices_v4(
    names: Sequence[str],
    required: Sequence[str],
    *,
    label: str,
) -> dict[str, int]:
    """Resolve the fixed V71 semantic names without accepting aliases."""

    result: dict[str, int] = {}
    for required_name in required:
        matches = [
            int(index)
            for index, name in enumerate(names)
            if str(name) == str(required_name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"selective V71 FK requires exactly one {label} bone "
                f"{required_name!r}"
            )
        result[str(required_name)] = matches[0]
    return result


def _production_leg_compound_roots_v1(names: Sequence[str]) -> dict[str, Any]:
    """Build the complete, bilateral V71 leg/foot semantic chain."""

    result: dict[str, Any] = {}
    toe_counts: list[int] = []
    for side in ("left", "right"):
        required = _V71_LEG_COMPOUND_ROOT_NAMES_V1[side]
        indices = _required_unique_bone_indices_v4(
            names,
            tuple(required[label] for label in ("femur", "knee", "shank", "ankle", "arch")),
            label=f"{side} leg",
        )
        toes = [
            int(index)
            for index, name in enumerate(names)
            if str(name).startswith(str(required["toes_prefix"]))
        ]
        if not toes:
            raise ValueError(
                "selective V71 FK requires at least one "
                f"{side} toe-chain root"
            )
        toe_counts.append(len(toes))
        result[side] = {
            label: indices[str(required[label])]
            for label in ("femur", "knee", "shank", "ankle", "arch")
        }
        result[side]["toes"] = toes
    if toe_counts[0] != toe_counts[1]:
        raise ValueError(
            "selective V71 FK requires symmetric left/right toe-chain roots"
        )
    return result


def _production_hand_controller_bones_v4(asset: Any) -> set[int]:
    """Return exactly the bilateral SMPL-X wrist/finger controller set."""

    expected = set(direct_smplx_hand_controllers_v4(asset))
    raw_names = getattr(asset, "joint_names", None)
    raw_mapped = getattr(asset, "source_bone_smplx_a", None)
    raw_modes = getattr(asset, "source_bone_driver_types", None)
    names = [] if raw_names is None else list(raw_names)
    mapped = [] if raw_mapped is None else list(raw_mapped)
    modes = [] if raw_modes is None else list(raw_modes)
    semantic: dict[str, set[str]] = {"left": set(), "right": set()}
    for bone in expected:
        if bone >= len(mapped) or bone >= len(modes):
            raise ValueError("selective V71 FK hand controller metadata is incomplete")
        joint_id = int(mapped[bone])
        if joint_id < 0 or joint_id >= len(names):
            raise ValueError("selective V71 FK hand controller joint is invalid")
        side, separator, part = str(names[joint_id]).partition("_")
        if (
            str(modes[bone]) not in {"joint_local", "direct_joint"}
            or side not in semantic
            or not separator
            or part not in _SMPLX_HAND_CONTROLLER_PARTS_V4
        ):
            raise ValueError(
                "selective V71 FK hand controller set must map wrists and "
                "all three joints of every finger"
            )
        semantic[side].add(part)
    if (
        len(expected) != 32
        or semantic["left"] != _SMPLX_HAND_CONTROLLER_PARTS_V4
        or semantic["right"] != _SMPLX_HAND_CONTROLLER_PARTS_V4
    ):
        raise ValueError(
            "selective V71 FK requires both wrists and all 30 SMPL-X "
            "finger controllers"
        )
    return expected


def _validate_production_hand_bind_follow_children_v4(
    asset: Any,
    controllers: set[int],
) -> None:
    """Require the authored hand-control terminal links when hierarchy exists."""

    raw_parents = getattr(asset, "source_bone_parents", None)
    if raw_parents is None:
        return
    raw_modes = getattr(asset, "source_bone_driver_types", None)
    names = list(getattr(asset, "source_bone_names", None) or ())
    if raw_modes is None:
        raise ValueError(
            "selective V71 FK hand bind-follow validation requires driver modes"
        )
    try:
        parents = tuple(int(value) for value in raw_parents)
        modes = tuple(str(value) for value in raw_modes)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "selective V71 FK hand bind-follow hierarchy is invalid"
        ) from error
    if len(parents) != len(names) or len(modes) != len(names):
        raise ValueError(
            "selective V71 FK hand bind-follow hierarchy does not match bones"
        )
    missing = sorted(
        controller
        for controller in controllers
        if not any(
            int(parent) == controller and modes[child] == "bind_follow"
            for child, parent in enumerate(parents)
        )
    )
    if missing:
        raise ValueError(
            "selective V71 FK wrist/finger controllers require a direct "
            f"bind_follow child: {missing}"
        )


def leg_compound_roots_v1(bone_names: Sequence[str] | None) -> dict[str, Any]:
    """Resolve the authored bilateral leg/foot controller roots by name."""

    names = list(bone_names or ())

    def index(name: str) -> int | None:
        try:
            return int(names.index(name))
        except ValueError:
            return None

    result: dict[str, Any] = {}
    for side, suffix in (("left", "L"), ("right", "R")):
        entry: dict[str, Any] = {}
        for label, name in (
            ("femur", f"Femur_Rot_{suffix}"),
            ("knee", f"Knee_Rotate_{suffix}"),
            ("shank", f"Tibia_Bone_{suffix}"),
            ("ankle", f"Ankle_Rot_{suffix}"),
            ("arch", f"Arch_Rot_{suffix}"),
        ):
            value = index(name)
            if value is not None:
                entry[label] = value
        toe_roots = [
            int(bone)
            for bone, name in enumerate(names)
            if str(name).startswith(f"Toes_Rotate_{suffix}")
        ]
        if toe_roots:
            entry["toes"] = toe_roots
        if entry:
            result[side] = entry
    return result


def build_selective_fk_metadata_v4(
    asset: Any,
    metadata: Mapping[str, Any] | None = None,
    *,
    extra_direct_bone_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Return detached metadata for the new selective-authority contract."""

    result = copy.deepcopy(dict(metadata or {}))
    names = list(getattr(asset, "source_bone_names", None) or ())
    direct = set(direct_smplx_hand_controllers_v4(asset))
    for name in extra_direct_bone_names:
        if str(name) in names:
            direct.add(int(names.index(str(name))))
    result.update(
        {
            SOURCE_FK_POLICY_KEY_V4: SELECTIVE_AUTHORITY_FK_POLICY_V4,
            "source_full_local_fk_v2": False,
            "source_joint_local_fk_v1": False,
            "source_connected_local_fk_v3": False,
            "source_local_fk_bones_v3": articulation_local_fk_bones_v4(names),
            "source_direct_driver_bones_v1": sorted(direct),
            "source_leg_compound_roots_v1": leg_compound_roots_v1(names),
        }
    )
    return result


def _validated_bone_list(
    metadata: Mapping[str, Any],
    key: str,
    *,
    bone_count: int,
) -> tuple[int, ...]:
    raw = metadata.get(key, ())
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{key} must be a list of source-bone indices")
    values = tuple(int(value) for value in raw)
    if len(set(values)) != len(values):
        raise ValueError(f"{key} may not contain duplicate source bones")
    if any(value < 0 or value >= int(bone_count) for value in values):
        raise ValueError(f"{key} contains an invalid source-bone index")
    return values


def _validate_leg_roots(value: Any, *, bone_count: int) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError("source_leg_compound_roots_v1 must be a mapping")
    for side, entry in value.items():
        if str(side) not in {"left", "right"} or not isinstance(entry, Mapping):
            raise ValueError("source_leg_compound_roots_v1 has an invalid side")
        for label, raw in entry.items():
            values = raw if isinstance(raw, (list, tuple)) else (raw,)
            if str(label) not in {"femur", "knee", "shank", "ankle", "arch", "toes"}:
                raise ValueError(
                    "source_leg_compound_roots_v1 has an unknown compound"
                )
            indices = tuple(int(item) for item in values)
            if not indices or any(
                item < 0 or item >= int(bone_count) for item in indices
            ):
                raise ValueError(
                    "source_leg_compound_roots_v1 contains an invalid source bone"
                )


def validate_source_fk_policy_v8(
    metadata: Mapping[str, Any] | None,
    *,
    bone_count: int,
    bone_names: Sequence[str] | None = None,
    require_selective: bool = False,
) -> str:
    """Validate a new selective policy or a legacy full-FK compatibility pack."""

    values = dict(metadata or {})
    policy = values.get(SOURCE_FK_POLICY_KEY_V4)
    if policy is None:
        if require_selective:
            raise ValueError(f"{SOURCE_FK_POLICY_KEY_V4} must be explicitly selective")
        if values.get("source_full_local_fk_v2") is not True:
            raise ValueError(
                "V8 requires either selective source_fk_policy_v4 or legacy "
                "source_full_local_fk_v2=true"
            )
        return LEGACY_FULL_LOCAL_FK_POLICY
    if str(policy) != SELECTIVE_AUTHORITY_FK_POLICY_V4:
        raise ValueError(f"unsupported {SOURCE_FK_POLICY_KEY_V4}={policy!r}")
    if values.get("source_full_local_fk_v2") is not False:
        raise ValueError(
            "selective source_fk_policy_v4 requires source_full_local_fk_v2=false"
        )
    if values.get("source_joint_local_fk_v1") is not False:
        raise ValueError(
            "selective source_fk_policy_v4 forbids joint-local FK fallback"
        )
    if values.get("source_connected_local_fk_v3", False) is not False:
        raise ValueError(
            "selective source_fk_policy_v4 forbids connected-local FK fallback"
        )
    local = set(
        _validated_bone_list(
            values, "source_local_fk_bones_v3", bone_count=bone_count
        )
    )
    direct = set(
        _validated_bone_list(
            values, "source_direct_driver_bones_v1", bone_count=bone_count
        )
    )
    if local & direct:
        raise ValueError(
            "selective source FK bones and direct driver bones must be disjoint"
        )
    names = tuple(str(name) for name in (bone_names or ()))
    if names:
        if len(names) != int(bone_count):
            raise ValueError("source FK policy bone_names do not match bone_count")
        expected_local = set(articulation_local_fk_bones_v4(names))
        if local != expected_local:
            raise ValueError(
                "source_local_fk_bones_v3 must exactly match the knee/elbow "
                "mechanism links"
            )
    _validate_leg_roots(
        values.get("source_leg_compound_roots_v1"), bone_count=bone_count
    )
    if names and int(bone_count) == 235:
        if len(set(names)) != len(names):
            raise ValueError("selective V71 FK source bone names must be unique")
        expected_local_names = tuple(
            name
            for side in ("left", "right")
            for name in _V71_ARTICULATION_LOCAL_BONE_NAMES_V4[side]
        )
        expected_local_indices = set(
            _required_unique_bone_indices_v4(
                names,
                expected_local_names,
                label="knee/elbow mechanism",
            ).values()
        )
        if local != expected_local_indices:
            raise ValueError(
                "source_local_fk_bones_v3 must contain exactly the 12 bilateral "
                "knee/elbow mechanism bones"
            )
        expected_roots = _production_leg_compound_roots_v1(names)
        if values.get("source_leg_compound_roots_v1") != expected_roots:
            raise ValueError(
                "source_leg_compound_roots_v1 must contain the complete "
                "bilateral V71 leg/foot chain"
            )
    return SELECTIVE_AUTHORITY_FK_POLICY_V4


def validate_source_fk_asset_policy_v8(
    asset: Any,
    *,
    require_selective: bool = False,
) -> str:
    """Validate policy structure plus production V71 hand semantics."""

    names = tuple(str(name) for name in (asset.source_bone_names or ()))
    policy = validate_source_fk_policy_v8(
        asset.metadata,
        bone_count=len(names),
        bone_names=names,
        require_selective=require_selective,
    )
    if policy != SELECTIVE_AUTHORITY_FK_POLICY_V4 or len(names) != 235:
        return policy

    expected_hand = _production_hand_controller_bones_v4(asset)
    direct = set(
        _validated_bone_list(
            dict(asset.metadata or {}),
            "source_direct_driver_bones_v1",
            bone_count=len(names),
        )
    )
    if not expected_hand.issubset(direct):
        missing = sorted(expected_hand - direct)
        raise ValueError(
            "source_direct_driver_bones_v1 is missing wrist/finger "
            f"controllers: {missing}"
        )
    allowed_direct = expected_hand | {
        int(index)
        for index, name in enumerate(names)
        if name in _V71_NON_HAND_DIRECT_ANCHOR_NAMES_V4
    }
    unexpected = sorted(direct - allowed_direct)
    if unexpected:
        raise ValueError(
            "source_direct_driver_bones_v1 contains unsupported direct "
            f"drivers: {unexpected}"
        )
    _validate_production_hand_bind_follow_children_v4(asset, expected_hand)
    return policy


def _is_source_descendant_v811(
    child: int,
    ancestor: int,
    parents: Sequence[int],
) -> bool:
    """Return whether a source bone is inside one specific parent chain."""

    cursor = int(child)
    visited: set[int] = set()
    while cursor >= 0:
        if cursor == int(ancestor):
            return True
        if cursor >= len(parents) or cursor in visited:
            return False
        visited.add(cursor)
        cursor = int(parents[cursor])
    return False


def selective_leg_runtime_roots_v811(asset: Any) -> dict[int, int]:
    """Resolve the selective V71 guide-driven leg roots for runtime FK.

    Legacy full-FK packs deliberately return an empty mapping so their
    read-only pose replay remains unchanged.  New V71 selective packs must
    prove that their metadata names a complete anatomical chain, that each
    active controller drives the expected SMPL-X station, and that the
    authored source hierarchy actually connects the declared chain.
    """

    metadata = dict(getattr(asset, "metadata", None) or {})
    if metadata.get(SOURCE_FK_POLICY_KEY_V4) != SELECTIVE_AUTHORITY_FK_POLICY_V4:
        return {}

    names = tuple(str(name) for name in (getattr(asset, "source_bone_names", None) or ()))
    validate_source_fk_asset_policy_v8(asset, require_selective=True)
    if len(names) != 235:
        # Small synthetic rigs may exercise the general selective contract,
        # but the fixed V71 leg semantics only apply to the production rig.
        return {}
    if metadata.get("source_anatomical_guide_fk_v810") is not True:
        raise ValueError(
            "selective V71 runtime leg authority requires "
            "source_anatomical_guide_fk_v810=true"
        )
    if getattr(asset, "source_driver_rest_joints", None) is None:
        raise ValueError(
            "selective V71 runtime leg authority requires anatomical guide joints"
        )
    legacy = [
        key for key in _SELECTIVE_RUNTIME_LEGACY_KEYS_V811 if metadata.get(key)
    ]
    if legacy:
        raise ValueError(
            "selective V71 runtime leg authority forbids legacy leg solvers: "
            f"{legacy}"
        )

    raw_parents = getattr(asset, "source_bone_parents", None)
    raw_mapped = getattr(asset, "source_bone_smplx_a", None)
    raw_modes = getattr(asset, "source_bone_driver_types", None)
    joint_names = tuple(str(name) for name in (getattr(asset, "joint_names", None) or ()))
    if raw_parents is None or raw_mapped is None or raw_modes is None:
        raise ValueError(
            "selective V71 runtime leg authority requires source hierarchy and drivers"
        )
    try:
        parents = tuple(int(value) for value in raw_parents)
        mapped = tuple(int(value) for value in raw_mapped)
        modes = tuple(str(value) for value in raw_modes)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "selective V71 runtime leg authority has invalid source driver data"
        ) from error
    if (
        len(parents) != len(names)
        or len(mapped) != len(names)
        or len(modes) != len(names)
    ):
        raise ValueError(
            "selective V71 runtime leg authority source driver lengths disagree"
        )

    roots = metadata.get("source_leg_compound_roots_v1")
    if not isinstance(roots, Mapping):
        raise ValueError(
            "selective V71 runtime leg authority requires a complete leg chain"
        )
    local = set(
        _validated_bone_list(
            metadata,
            "source_local_fk_bones_v3",
            bone_count=len(names),
        )
    )
    direct = set(
        _validated_bone_list(
            metadata,
            "source_direct_driver_bones_v1",
            bone_count=len(names),
        )
    )
    runtime_roots: dict[int, int] = {}
    for side in ("left", "right"):
        entry = roots.get(side)
        if not isinstance(entry, Mapping):
            raise ValueError(
                "selective V71 runtime leg authority requires bilateral leg chains"
            )
        resolved: dict[str, int] = {}
        for label in _V71_LEG_RUNTIME_STATION_LABELS_V811:
            try:
                bone = int(entry[label])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "selective V71 runtime leg authority has an incomplete "
                    f"{side} station chain"
                ) from error
            expected_joint, expected_mode = _V71_LEG_RUNTIME_DRIVER_SEMANTICS_V811[
                side
            ][label]
            if (
                bone < 0
                or bone >= len(names)
                or mapped[bone] < 0
                or mapped[bone] >= len(joint_names)
                or joint_names[mapped[bone]] != expected_joint
                or modes[bone] != expected_mode
            ):
                raise ValueError(
                    "selective V71 runtime leg authority has an invalid "
                    f"{side}.{label} driver"
                )
            if bone in runtime_roots:
                raise ValueError(
                    "selective V71 runtime leg authority reuses a station root"
                )
            resolved[label] = bone
            runtime_roots[bone] = mapped[bone]

        for parent_label, child_label in zip(
            _V71_LEG_RUNTIME_STATION_LABELS_V811,
            _V71_LEG_RUNTIME_STATION_LABELS_V811[1:],
            strict=True,
        ):
            if not _is_source_descendant_v811(
                resolved[child_label], resolved[parent_label], parents
            ):
                raise ValueError(
                    "selective V71 runtime leg authority has a disconnected "
                    f"{side} {parent_label}/{child_label} chain"
                )
        toe_roots = entry.get("toes")
        if not isinstance(toe_roots, (list, tuple)) or not toe_roots:
            raise ValueError(
                "selective V71 runtime leg authority requires toe-chain roots"
            )
        for raw_toe in toe_roots:
            toe = int(raw_toe)
            if (
                toe < 0
                or toe >= len(names)
                or modes[toe] != "bind_follow"
                or not _is_source_descendant_v811(toe, resolved["arch"], parents)
            ):
                raise ValueError(
                    "selective V71 runtime leg authority has an invalid "
                    f"{side} toe-chain root"
                )
        for label in ("knee", "shank"):
            if resolved[label] not in local:
                raise ValueError(
                    "selective V71 runtime leg authority requires local FK for "
                    f"{side}.{label}"
                )
        if set(resolved.values()) & direct:
            raise ValueError(
                "selective V71 runtime leg roots may not be direct hand drivers"
            )
    return runtime_roots


__all__ = [
    "LEGACY_FULL_LOCAL_FK_POLICY",
    "SELECTIVE_AUTHORITY_FK_POLICY_V4",
    "SOURCE_FK_POLICY_KEY_V4",
    "articulation_local_fk_bones_v4",
    "build_selective_fk_metadata_v4",
    "direct_smplx_hand_controllers_v4",
    "leg_compound_roots_v1",
    "selective_leg_runtime_roots_v811",
    "validate_source_fk_asset_policy_v8",
    "validate_source_fk_policy_v8",
]
