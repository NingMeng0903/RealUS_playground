"""Manipulability nullspace task tests."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.model import RobotKinematics, deg2rad
from rm75_control.control.joint_admittance.tasks.manipulability_task import (
    ManipulabilityTask,
    ManipulabilityTaskConfig,
)
from rm75_control.control.joint_admittance.tasks.nullspace_task import NullspaceTaskConfig
from rm75_control.control.joint_admittance.tasks.secondary_composer import SecondaryComposer


def test_manipulability_gradient_nonzero_near_singularity():
    kin = RobotKinematics()
    task = ManipulabilityTask(kin, ManipulabilityTaskConfig(k_mu=1.0, eps_rad=1e-3))
    # Straight q=0 has μ=0 and a numerically flat ∇μ; use the contact-like posture
    # from the failed on-robot move (J1 ~ -166 deg).
    q = deg2rad(np.array([-166.0, -30.0, 80.0, 5.0, -90.0, 60.0, 0.0]))
    grad = task.gradient(q)
    assert task.last_mu < 0.05
    assert float(np.linalg.norm(grad)) > 1e-6


def test_manipulability_replaces_centering_when_active():
    kin = RobotKinematics()
    from rm75_control.control.joint_admittance.tasks.nullspace_task import JointCenteringTask

    centering = JointCenteringTask.from_kinematics(
        kin,
        NullspaceTaskConfig(
            k_center=2.0,
            activation=0.85,
            q_nominal_rad=deg2rad(np.array([0.0, -45.0, 0.0, 90.0, 0.0, 45.0, 0.0])),
        ),
    )
    manip = ManipulabilityTask(kin, ManipulabilityTaskConfig(k_mu=0.8, eps_rad=1e-3))
    comp = SecondaryComposer(centering, None, manipulability=manip, v_max=kin.v_max, max_qdot_frac=0.2)
    q = deg2rad(np.array([-166.0, -30.0, 80.0, 5.0, -90.0, 60.0, 0.0]))
    qdot_center = comp.compose(q, None, np.zeros(7), arm_suppressed=True, centering_suppressed=False)
    qdot_manip = comp.compose(
        q, None, np.zeros(7), arm_suppressed=True,
        centering_suppressed=True, manipulability_active=True,
        sigma_min=0.05,
    )
    assert float(np.linalg.norm(qdot_center)) > 0.0
    assert float(np.linalg.norm(qdot_manip)) > 0.0
    # Directions differ: centering pulls toward q_nominal, manip ascends μ.
    cos = float(np.dot(qdot_center, qdot_manip) / (
        np.linalg.norm(qdot_center) * np.linalg.norm(qdot_manip) + 1e-12
    ))
    assert cos < 0.99


def test_joint_ik_manipulability_toggle():
    from rm75_control.control.joint_admittance.config import build_joint_ik_config
    from rm75_control.control.joint_admittance.loop import JointIkController
    import yaml
    from pathlib import Path

    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    cfg = build_joint_ik_config(raw)
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    assert inner.manipulability_task is not None
    inner.set_manipulability_active(True)
    assert inner._manipulability_active
    inner.set_manipulability_active(False)
    assert not inner._manipulability_active
