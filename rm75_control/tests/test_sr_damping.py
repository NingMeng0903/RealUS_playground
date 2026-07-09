"""Tests for Chiaverini / Nakamura singularity-robust nullspace projection."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.ik_types import (
    SrDampingConfig,
    project_onto_task_nullspace,
    sr_damping_lambda,
)
from rm75_control.control.joint_admittance.model import RobotKinematics


def test_sr_damping_lambda_increases_as_sigma_drops():
    cfg = SrDampingConfig(lam0=0.05, sigma_ref=0.08)
    lam_ok = sr_damping_lambda(0.10, cfg)
    lam_bad = sr_damping_lambda(0.01, cfg)
    assert lam_ok == cfg.lam0
    assert lam_bad > lam_ok * 10.0


def test_nullspace_releases_secondary_near_singularity():
    kin = RobotKinematics()
    q = np.zeros(kin.nv)
    J = kin.jacobian(q)
    qdot0 = np.array([0.1, -0.2, 0.15, 0.05, -0.1, 0.08, 0.12])
    cfg = SrDampingConfig(lam0=0.05, sigma_ref=0.08)
    sigma = kin.singular_values(J)
    sigma_min = float(sigma.min())

    out_tight = project_onto_task_nullspace(J, qdot0, damping=1e-4)
    out_sr = project_onto_task_nullspace(J, qdot0, sigma_min=sigma_min, sr_cfg=cfg)

    # Near a singularity, SR damping should pass MORE of qdot0 through N (ratio closer to 1).
    ratio_tight = float(np.linalg.norm(out_tight) / (np.linalg.norm(qdot0) + 1e-12))
    ratio_sr = float(np.linalg.norm(out_sr) / (np.linalg.norm(qdot0) + 1e-12))
    assert ratio_sr >= ratio_tight
