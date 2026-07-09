"""Paths for legacy velocity-admittance scan logs."""

from __future__ import annotations

from pathlib import Path

from rm75_control.force.compensation.paths import CONFIG_FORCE, CONFIG_ROBOT, PHI_JSON, REPO

VA_DATA_DIR = REPO / "data" / "velocity_admittance"
LOG_DIR = VA_DATA_DIR / "logs"
# Deprecated Cartesian demos removed; joint admittance configs are canonical.
CONFIG_ADMITTANCE = REPO / "configs" / "joint_admittance.yaml"
CONFIG_SIN_TOOL_Y_Z2N = CONFIG_ADMITTANCE
CONFIG_D_TO_A_SIN_TOOL_Y = REPO / "configs" / "joint_admittance_8dof.yaml"
CONFIG_HUMAN_SOFT_SCAN = CONFIG_ADMITTANCE
