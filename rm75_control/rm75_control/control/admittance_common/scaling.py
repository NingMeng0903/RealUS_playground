"""Scale admittance config for runtime ``desired_z`` CLI overrides."""

from __future__ import annotations

from rm75_control.control.admittance_common.controller import AdmittanceConfig


def scale_admittance_for_desired_z(raw: dict, desired_z_n: float) -> AdmittanceConfig:
    """Load the one physical controller without setpoint-dependent retuning.

    ``desired_z_n`` is intentionally accepted for call-site compatibility.
    The controller's normalized proactive error is what equalises tracking;
    mass, Dimeas scale and damping limits stay fixed between 1 N and 5 N.
    """
    del desired_z_n
    return AdmittanceConfig.from_dict(raw)
