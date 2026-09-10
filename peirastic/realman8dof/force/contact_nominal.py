"""One production nominal core, shared by contact-QP adapter and experiments."""
from __future__ import annotations

from peirastic.scan_path import SCAN_FORCE_AXES, force_profile
from .config import build_force_controller
from .legacy import LegacyForceLaw
from .torque_tilt import LegacyForceWithTilt, TorqueTilt, TorqueTiltConfig


def icra_contact_payload():
    return dict(force_profile("icra"), force_axes=list(SCAN_FORCE_AXES), control_frame="tool",
                desired_z=4., max_vz_tool_m_s=.01, v_seek_free_m_s=.01)


def build_contact_nominal(dt_s=.005):
    controller, raw, force = build_force_controller(dt_s, payload=icra_contact_payload())
    if force != 4.:
        raise ValueError("contact-QP nominal requires the frozen 4 N target")
    law = LegacyForceWithTilt(LegacyForceLaw(controller), TorqueTilt(TorqueTiltConfig.from_dict(raw)))
    return law, raw


def build_contact_position(reference):
    import numpy as np
    import yaml
    from peirastic.configs import DEFAULT_CONTROLLER_YAML
    from rm75_control.control.joint_admittance_8dof.loop import CartesianTrackConfig, CartesianTrackOuterLoop
    gains = yaml.safe_load(DEFAULT_CONTROLLER_YAML.read_text())["cartesian_track"]
    cfg = CartesianTrackConfig(k_task=np.array([gains["k_task_lin"]]*3+[gains["k_task_rot"]]*3),
                max_pos_err_m=gains["max_pos_err_m"], max_rot_err_rad=gains["max_rot_err_rad"],
                fb_lpf_tau_s=gains["fb_lpf_tau_s"], control_frame="tool",
                track_axes=1.-np.array(SCAN_FORCE_AXES))
    return CartesianTrackOuterLoop(reference, cfg)
