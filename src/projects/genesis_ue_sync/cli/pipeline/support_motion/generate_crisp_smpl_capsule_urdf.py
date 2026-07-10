#!/usr/bin/env python3
"""Emit a floating-base capsule URDF whose joint order matches CRISP MuJoco SMPL DFS (69 Euler scalars).

Joint order: for each body in SMPL_MUJOCO_KINEMATIC_ORDER[1:], three continuous joints (X,Y,Z Euler)
named ``{Body}_ex``, ``{Body}_ey``, ``{Body}_ez``. This matches``retarget_smpl_aa_to_crisp_mujoco_euler`` in ``crisp_real2sim_bridge.py``.

Run from repo root:
  python scripts/pipeline/support_motion/generate_crisp_smpl_capsule_urdf.py
"""

from __future__ import annotations

from pathlib import Path

REPO = next(parent.parent for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents) if parent.name == "src")

# Duplicated from ``crisp_real2sim_bridge`` so this script runs without importing the full package.
SMPL_CRISP_BODY_NAMES: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
)

SMPL_MUJOCO_KINEMATIC_ORDER: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
)

# SMPL_CRISP_BODY_NAMES index -> parent index (-1 for root)
_PARENT_SMPL: tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)


def _parent_name(child: str) -> str:
    idx = SMPL_CRISP_BODY_NAMES.index(child)
    p = _PARENT_SMPL[idx]
    if p < 0:
        raise ValueError(f"{child} has no parent")
    return SMPL_CRISP_BODY_NAMES[p]


# Child body -> bone origin (m) on first Euler joint, parent-link frame; Z-up sim, legs mostly -Z.
_BONE_ORIGIN: dict[str, tuple[float, float, float]] = {
    "L_Hip": (0.0, 0.10, -0.04),
    "R_Hip": (0.0, -0.10, -0.04),
    "Torso": (0.0, 0.0, 0.06),
    "L_Knee": (0.0, 0.0, -0.38),
    "R_Knee": (0.0, 0.0, -0.38),
    "L_Ankle": (0.0, 0.0, -0.36),
    "R_Ankle": (0.0, 0.0, -0.36),
    "Spine": (0.0, 0.0, 0.07),
    "L_Toe": (0.05, 0.0, -0.08),
    "R_Toe": (0.05, 0.0, -0.08),
    "Chest": (0.0, 0.0, 0.09),
    "Neck": (0.0, 0.0, 0.12),
    "Head": (0.0, 0.0, 0.09),
    "L_Thorax": (0.0, 0.08, 0.05),
    "L_Shoulder": (0.0, 0.11, 0.0),
    "L_Elbow": (0.0, 0.0, -0.28),
    "L_Wrist": (0.0, 0.0, -0.25),
    "L_Hand": (0.0, 0.0, -0.08),
    "R_Thorax": (0.0, -0.08, 0.05),
    "R_Shoulder": (0.0, -0.11, 0.0),
    "R_Elbow": (0.0, 0.0, -0.28),
    "R_Wrist": (0.0, 0.0, -0.25),
    "R_Hand": (0.0, 0.0, -0.08),
}

# (length, radius) capsule along local X after rpy="-1.5708 0 0" (vertical segment along -Z)
_CAPSULE: dict[str, tuple[float, float]] = {
    "Pelvis": (0.16, 0.085),
    "L_Hip": (0.12, 0.055),
    "R_Hip": (0.12, 0.055),
    "Torso": (0.12, 0.065),
    "L_Knee": (0.30, 0.045),
    "R_Knee": (0.30, 0.045),
    "L_Ankle": (0.12, 0.04),
    "R_Ankle": (0.12, 0.04),
    "Spine": (0.10, 0.06),
    "L_Toe": (0.08, 0.035),
    "R_Toe": (0.08, 0.035),
    "Chest": (0.12, 0.07),
    "Neck": (0.08, 0.04),
    "Head": (0.12, 0.055),
    "L_Thorax": (0.08, 0.04),
    "L_Shoulder": (0.10, 0.04),
    "L_Elbow": (0.22, 0.035),
    "L_Wrist": (0.10, 0.032),
    "L_Hand": (0.06, 0.04),
    "R_Thorax": (0.08, 0.04),
    "R_Shoulder": (0.10, 0.04),
    "R_Elbow": (0.22, 0.035),
    "R_Wrist": (0.10, 0.032),
    "R_Hand": (0.06, 0.04),
}


def _link_dummy(name: str) -> str:
    return f"""	<link name="{name}">
		<inertial>
			<origin rpy="0 0 0" xyz="0 0 0"/>
			<mass value="0"/>
			<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
		</inertial>
	</link>
"""


def _link_body(name: str) -> str:
    ln, rad = _CAPSULE.get(name, (0.1, 0.04))
    mass = 0.9 + ln * 1.8
    inertia = 0.02 + ln * 0.04
    return f"""	<link name="{name}">
		<inertial>
			<origin rpy="0 0 0" xyz="0 0 {-0.5 * ln:.5f}"/>
			<mass value="{mass:.5f}"/>
			<inertia ixx="{inertia:.5f}" ixy="0" ixz="0" iyy="{inertia:.5f}" iyz="0" izz="{inertia * 0.2:.5f}"/>
		</inertial>
		<collision>
			<origin rpy="-1.5708 0 0" xyz="0 0 {-0.5 * ln:.5f}"/>
			<geometry>
				<capsule length="{ln:.5f}" radius="{rad:.5f}"/>
			</geometry>
		</collision>
	</link>
"""


def _joint_continuous(name: str, parent: str, child: str, origin_xyz: str, axis: str) -> str:
    return f"""	<joint name="{name}" type="continuous">
		<parent link="{parent}"/>
		<child link="{child}"/>
		<dynamics damping="0.5" friction="0.0001"/>
		<origin rpy="0 0 0" xyz="{origin_xyz}"/>
		<axis xyz="{axis}"/>
	</joint>
"""


def build_urdf() -> str:
    lines: list[str] = [
        '<?xml version="1.0"?>',
        '<robot name="crisp_smpl_capsule_humanoid">',
        _link_body("Pelvis"),
    ]
    for b in SMPL_MUJOCO_KINEMATIC_ORDER[1:]:
        lines.append(_link_dummy(f"{b}__e0"))
        lines.append(_link_dummy(f"{b}__e1"))
        lines.append(_link_body(b))

    for b in SMPL_MUJOCO_KINEMATIC_ORDER[1:]:
        parent = _parent_name(b)
        ox, oy, oz = _BONE_ORIGIN[b]
        o = f"{ox:.5f} {oy:.5f} {oz:.5f}"
        lines.append(_joint_continuous(f"{b}_ex", parent, f"{b}__e0", o, "1 0 0"))
        lines.append(_joint_continuous(f"{b}_ey", f"{b}__e0", f"{b}__e1", "0 0 0", "0 1 0"))
        lines.append(_joint_continuous(f"{b}_ez", f"{b}__e1", b, "0 0 0", "0 0 1"))

    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def main() -> None:
    out = REPO / "assets" / "humanoid" / "crisp_smpl_capsule" / "humanoid.urdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build_urdf()
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
