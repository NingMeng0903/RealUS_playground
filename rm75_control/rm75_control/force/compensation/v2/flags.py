"""Split dynamic-kinematics / rotational-inertia flags (no global regressor shim)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class DynamicKinematicsMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    APPLY = "apply"


def _mode_from_yaml(value: Any) -> DynamicKinematicsMode:
    """YAML 1.1 parses unquoted ``off``/``on`` as bool, not the mode name."""

    if value is None:
        return DynamicKinematicsMode.OFF
    if isinstance(value, bool):
        return DynamicKinematicsMode.APPLY if value else DynamicKinematicsMode.OFF
    if isinstance(value, (int, float)):
        return DynamicKinematicsMode.APPLY if float(value) else DynamicKinematicsMode.OFF
    text = str(value).strip().lower()
    if text in ("", "off", "false", "0", "no"):
        return DynamicKinematicsMode.OFF
    if text in ("on", "true", "1", "yes"):
        return DynamicKinematicsMode.APPLY
    return DynamicKinematicsMode(text)


def resolve_online_flags(force_cfg: dict[str, Any]) -> tuple[DynamicKinematicsMode, bool, bool]:
    """Online YAML only.

    New keys win. Legacy ``use_inertia`` maps to *both* dynamic kinematics and
    rotational inertia (old causal observer). Does not touch ``build_dataset``.
    """
    f = dict(force_cfg or {})
    has_new = any(
        k in f
        for k in ("use_dynamic_kinematics", "use_rotational_inertia", "dynamic_kinematics_mode")
    )
    if has_new:
        mode = _mode_from_yaml(f.get("dynamic_kinematics_mode", "off"))
        use_dyn = bool(f.get("use_dynamic_kinematics", mode != DynamicKinematicsMode.OFF))
        use_rot = bool(f.get("use_rotational_inertia", False))
        if bool(f.get("use_rotational_inertia", False)) and not bool(
            f.get("use_dynamic_kinematics", mode != DynamicKinematicsMode.OFF)
        ):
            raise ValueError("use_rotational_inertia requires use_dynamic_kinematics")
        if mode == DynamicKinematicsMode.OFF:
            use_dyn = False
            use_rot = False
        elif mode in (DynamicKinematicsMode.OBSERVE, DynamicKinematicsMode.APPLY):
            use_dyn = True
        return mode, use_dyn, use_rot

    legacy = bool(f.get("use_inertia", False))
    mode = DynamicKinematicsMode.APPLY if legacy else DynamicKinematicsMode.OFF
    return mode, legacy, legacy
