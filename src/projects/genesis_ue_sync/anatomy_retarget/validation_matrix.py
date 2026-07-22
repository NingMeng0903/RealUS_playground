"""Deterministic beta/pose cases used by release validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidationCase:
    name: str
    betas: np.ndarray
    pose_axis_angle: np.ndarray


def beta_cases(
    real_betas: np.ndarray,
    *,
    principal_dimensions: int = 4,
    sigma: float = 2.0,
) -> dict[str, np.ndarray]:
    real = np.asarray(real_betas, dtype=np.float32).reshape(-1)
    cases: dict[str, np.ndarray] = {
        "beta_zero": np.zeros_like(real),
        "beta_real": real.copy(),
    }
    for index in range(min(max(0, int(principal_dimensions)), len(real))):
        positive = np.zeros_like(real)
        negative = np.zeros_like(real)
        positive[index] = float(sigma)
        negative[index] = -float(sigma)
        cases[f"beta_{index:02d}_plus_{sigma:g}sigma"] = positive
        cases[f"beta_{index:02d}_minus_{sigma:g}sigma"] = negative
    return cases


def pose_cases(joint_names: list[str]) -> dict[str, np.ndarray]:
    index = {name: i for i, name in enumerate(joint_names)}
    count = len(joint_names)

    def pose(**rotations: tuple[float, float, float]) -> np.ndarray:
        value = np.zeros((count, 3), dtype=np.float32)
        for name, axis_angle in rotations.items():
            if name not in index:
                raise ValueError(f"validation pose requires missing joint {name!r}")
            value[index[name]] = np.asarray(axis_angle, dtype=np.float32)
        return value

    return {
        "pose_zero": pose(),
        "pose_upper_limb_flex": pose(
            left_shoulder=(0.0, 0.0, 0.7),
            right_shoulder=(0.0, 0.0, -0.7),
            left_elbow=(0.0, -1.1, 0.0),
            right_elbow=(0.0, 1.1, 0.0),
            left_wrist=(0.25, 0.0, 0.0),
            right_wrist=(-0.25, 0.0, 0.0),
        ),
        "pose_shoulder_girdle_elevation": pose(
            **{
                name: rotation
                for name, rotation in (
                    ("left_collar", (0.0, 0.0, 0.30)),
                    ("right_collar", (0.0, 0.0, -0.30)),
                    ("left_shoulder", (0.0, 0.0, 0.45)),
                    ("right_shoulder", (0.0, 0.0, -0.45)),
                )
                if name in index
            }
        ),
        "pose_lower_limb_flex": pose(
            left_hip=(0.65, 0.0, 0.0),
            right_hip=(0.35, 0.0, 0.0),
            # The fitted SMPL-X captures and the exported Blender knee driver
            # both use positive local X for anatomical flexion.  The former
            # negative sign exercised extension and hid the bent-knee failure.
            left_knee=(1.0, 0.0, 0.0),
            right_knee=(0.65, 0.0, 0.0),
            left_ankle=(0.25, 0.0, 0.0),
            right_ankle=(0.15, 0.0, 0.0),
        ),
        # A hinge-only knee is insufficient for fitted human captures: hip
        # yaw/roll and knee varus/twist occur together, so the rigid anatomy
        # and the blended skin see different three-dimensional trajectories.
        # Keep this deterministic and subject-independent while exercising the
        # full SO(3) range that exposed the former patella/femur failure.
        "pose_lower_limb_multiaxis": pose(
            left_hip=(-0.20, -0.25, 0.55),
            right_hip=(-0.10, 0.20, -0.35),
            left_knee=(0.70, 0.60, -1.45),
            right_knee=(0.40, 0.30, 0.20),
            left_ankle=(0.75, 0.40, 0.30),
            right_ankle=(0.65, -0.25, -0.45),
        ),
        "pose_upper_limb_multiaxis": pose(
            left_shoulder=(0.35, 0.10, -1.15),
            right_shoulder=(0.35, -0.45, 1.20),
            left_elbow=(0.55, -0.55, -0.15),
            right_elbow=(-0.40, 0.65, 0.05),
            left_wrist=(0.70, -0.05, 0.25),
            right_wrist=(0.50, 0.25, -0.10),
        ),
        "pose_full_body_multiaxis": pose(
            **{
                name: rotation
                for name, rotation in (
                    ("pelvis", (-0.10, 0.25, 1.25)),
                    ("left_hip", (-0.20, -0.25, 0.55)),
                    ("right_hip", (-0.10, 0.20, -0.35)),
                    ("left_knee", (0.70, 0.60, -1.45)),
                    ("right_knee", (0.40, 0.30, 0.20)),
                    ("left_ankle", (0.75, 0.40, 0.30)),
                    ("right_ankle", (0.65, -0.25, -0.45)),
                    ("left_shoulder", (0.35, 0.10, -1.15)),
                    ("right_shoulder", (0.35, -0.45, 1.20)),
                    ("left_elbow", (0.55, -0.55, -0.15)),
                    ("right_elbow", (-0.40, 0.65, 0.05)),
                    ("left_wrist", (0.70, -0.05, 0.25)),
                    ("right_wrist", (0.50, 0.25, -0.10)),
                    ("neck", (-0.32, 0.10, -0.20)),
                    ("head", (0.20, -0.10, 0.20)),
                )
                if name in index
            }
        ),
        "pose_finger_flex": pose(
            **{
                f"{side}_{finger}{level}": (0.0, 0.0, 0.45 + 0.1 * level)
                for side in ("left", "right")
                for finger in ("thumb", "index", "middle", "ring", "pinky")
                for level in (1, 2, 3)
            }
        ),
        "pose_axial_twist": pose(
            pelvis=(0.0, 0.7, 0.0),
            left_hip=(0.0, 0.45, 0.0),
            right_hip=(0.0, -0.45, 0.0),
            left_shoulder=(0.45, 0.0, 0.0),
            right_shoulder=(-0.45, 0.0, 0.0),
        ),
        "pose_head_neck_flex": pose(
            **{
                name: rotation
                for name, rotation in (
                    ("neck", (0.32, 0.0, 0.0)),
                    ("head", (0.28, 0.0, 0.0)),
                )
                if name in index
            }
        ),
    }


def release_validation_matrix(
    real_betas: np.ndarray,
    joint_names: list[str],
    *,
    principal_dimensions: int = 4,
) -> list[ValidationCase]:
    betas = beta_cases(real_betas, principal_dimensions=principal_dimensions)
    poses = pose_cases(joint_names)
    return [
        ValidationCase(name=f"{beta_name}__{pose_name}", betas=beta, pose_axis_angle=pose)
        for beta_name, beta in betas.items()
        for pose_name, pose in poses.items()
    ]
