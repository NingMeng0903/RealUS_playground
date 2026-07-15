"""Source-asset semantic roles used by the retarget runtime.

These are source naming semantics, not spatial patches: the Blender anatomy
asset already distinguishes skull and foot phalanx meshes by name.  Keeping
the policy here makes the exceptions auditable and prevents generic skin
containment from changing rigid anatomy that SMPL-X does not parameterise.
"""

from __future__ import annotations


def is_cranial_shell_mesh(name: str) -> bool:
    normalized = str(name).lower()
    return "upper_skull" in normalized or "cranium" in normalized


def is_foot_toe_mesh(name: str) -> bool:
    normalized = str(name).lower()
    return "foot" in normalized and any(
        token in normalized for token in ("phalan", "toe")
    )


def exempt_from_rigid_containment(name: str) -> bool:
    """Rigid parts that have no SMPL-X articulation/containment target."""
    return is_cranial_shell_mesh(name) or is_foot_toe_mesh(name)
