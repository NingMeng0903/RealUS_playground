"""TRACK_HYBRID local compilation keeps caller supplied phase labels."""

from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

from peirastic.configs import DEFAULT_CONTROLLER_YAML
from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof.session import compile_request
from rm75_control.control.joint_admittance_8dof.api import CompileContext
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


_SEED = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])


def _ctx() -> CompileContext:
    raw = yaml.safe_load(DEFAULT_CONTROLLER_YAML.read_text())
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.native_shm_prefix = f"rm75_wbc_hybrid_label_{os.getpid()}"
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    inner.reset(_SEED)
    return CompileContext(
        kin=kin,
        inner=inner,
        euler_order=cfg.euler_order,
        control_frame=cfg.control_frame,
        v_scale=cfg.v_scale,
    )


@pytest.mark.parametrize("reference", ["hold", "polyline", "ellipse"])
@pytest.mark.parametrize("use_tff_split", [False, True])
def test_track_hybrid_label_survives_local_compile(
    reference: str, use_tff_split: bool
) -> None:
    ctx = _ctx()
    pose = ctx.kin.fk_pose(_SEED)
    payload = {
        "reference": reference,
        "label": f"scan_{reference}",
        "duration_s": 1.0,
        "desired_z": 0.0,
        "use_tff_split": use_tff_split,
    }
    if reference == "polyline":
        payload["points"] = [
            pose[:3].tolist(),
            (pose[:3] + np.array([0.01, 0.0, 0.0])).tolist(),
        ]
        payload["rpy"] = pose[3:].tolist()
    elif reference == "ellipse":
        payload.update(
            amplitude_x_m=0.005,
            amplitude_y_m=0.010,
            period_s=2.0,
            soft_start=False,
        )

    phase = compile_request(
        ctx,
        ModeRequest(Mode.TRACK_HYBRID, payload),
        dt=0.005,
    )

    assert phase.label == f"scan_{reference}"


def test_track_hybrid_label_defaults_for_local_compile() -> None:
    phase = compile_request(
        _ctx(),
        ModeRequest(
            Mode.TRACK_HYBRID,
            {"reference": "hold", "duration_s": 1.0, "desired_z": 0.0},
        ),
        dt=0.005,
    )

    assert phase.label == "track_hybrid"
