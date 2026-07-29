from __future__ import annotations

import numpy as np
import pytest

from ird_playground.ird.canonical import (
    FLANGE_J7_INVARIANT_INDEX,
    FLANGE_ROLL_INDEX,
    canonical_flange_from_se3_features,
    canonical_from_se3_features,
)
from ird_playground.ird.robot_model import RobotModelSpec


def _locked_probe45_model(spec: RobotModelSpec):
    """Build the rail-locked pinocchio model, or skip if unavailable."""
    from ird_playground.ird.gt_common import reachability_modules

    *_, build_locked_rail_model = reachability_modules()[1:]
    return build_locked_rail_model(
        spec.kinematics_urdf,
        rail_locked_at_m=spec.rail_locked_at_m,
        tcp_frame=spec.tcp_frame,
    )


def _sweep_se3_features(pin, model, q, joint_index, frame_ids, n=31):
    """Sweep one joint and return per-frame ``se3_rot6d9`` feature stacks."""
    out = {name: [] for name in frame_ids}
    q = np.array(q, dtype=float)
    for value in np.linspace(model.q_lower[joint_index], model.q_upper[joint_index], n):
        q[joint_index] = value
        pin.forwardKinematics(model.model, model.data, q)
        pin.updateFramePlacements(model.model, model.data)
        for name, frame_id in frame_ids.items():
            pose = model.data.oMf[frame_id]
            out[name].append(
                np.concatenate(
                    (pose.translation, pose.rotation[:, 0], pose.rotation[:, 1])
                )
            )
    return {name: np.asarray(rows, dtype=np.float64) for name, rows in out.items()}


def test_probe45_contract_matches_current_urdfs():
    spec = RobotModelSpec.default_probe45()
    spec.validate()
    assert spec.rail_limits_m() == pytest.approx((0.0, 0.8))
    T = spec.root_to_j1_axis()
    assert T[:3, 3] == pytest.approx([0.0, -0.4, 0.2405])
    manifest = spec.to_manifest()
    assert len(manifest["kinematics_urdf_sha256"]) == 64
    assert len(manifest["collision_urdf_sha256"]) == 64
    assert any(path.endswith("probe45.stl") for path in manifest["collision_mesh_sha256"])


def test_real_fk_q1_sweep_is_invariant_only_after_axis_transform():
    pin = pytest.importorskip("pinocchio")
    try:
        from ird_playground.ird.gt_common import reachability_modules

        *_, build_locked_rail_model = reachability_modules()[1:]
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rm75 reachability package unavailable")

    spec = RobotModelSpec.default_probe45()
    model = build_locked_rail_model(
        spec.kinematics_urdf,
        rail_locked_at_m=spec.rail_locked_at_m,
        tcp_frame=spec.tcp_frame,
    )
    q = 0.35 * model.q_lower + 0.65 * model.q_upper
    features = []
    for q1 in np.linspace(model.q_lower[0], model.q_upper[0], 31):
        q[0] = q1
        pin.forwardKinematics(model.model, model.data, q)
        pin.updateFramePlacements(model.model, model.data)
        pose = model.data.oMf[model.tcp_id]
        features.append(
            np.concatenate(
                (pose.translation, pose.rotation[:, 0], pose.rotation[:, 1])
            )
        )
    features_np = np.asarray(features, dtype=np.float32)
    wrong = canonical_from_se3_features(features_np)
    fixed = canonical_from_se3_features(
        features_np,
        T_root_axis=spec.root_to_j1_axis(),
    )
    assert float(np.max(np.ptp(wrong, axis=0))) > 0.1
    assert float(np.max(np.ptp(fixed, axis=0))) < 2.0e-5


def test_real_fk_q7_sweep_fixes_flange_z_block_but_moves_tcp_chart():
    """Sweeping q7 must leave the flange z-block fixed and move the TCP chart.

    RM4D's assumption 2 (the last joint's axis is the tool approach axis) holds
    at the flange and fails at the probe45 TCP. So the five flange components
    built from the flange z axis, which is the J7 axis, are exactly invariant,
    while the legacy TCP chart moves by an amount comparable to its own dynamic
    range. The three components built from the flange x axis carry the flange
    roll gamma and must move, because gamma is a real degree of freedom the
    chart deliberately keeps rather than a symmetry it may quotient.
    """
    pin = pytest.importorskip("pinocchio")
    spec = RobotModelSpec.default_probe45()
    try:
        model = _locked_probe45_model(spec)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rm75 reachability package unavailable")

    q = 0.35 * model.q_lower + 0.65 * model.q_upper
    features = _sweep_se3_features(
        pin, model, q, 6, {"tcp": model.tcp_id}
    )["tcp"].astype(np.float32)

    axis = spec.root_to_j1_axis()
    tool = spec.tool_frame().T_flange_tcp
    tcp_chart = canonical_from_se3_features(features, T_root_axis=axis)
    flange_chart = canonical_flange_from_se3_features(
        features, tool, T_root_axis=axis
    )
    spread = np.ptp(flange_chart, axis=0)

    assert float(np.max(np.ptp(tcp_chart, axis=0))) > 0.1
    assert float(np.max(spread[list(FLANGE_J7_INVARIANT_INDEX)])) < 2.0e-5
    assert float(np.min(spread[list(FLANGE_ROLL_INDEX)])) > 0.1


def test_real_fk_q1_sweep_is_invariant_in_flange_chart():
    """Base yaw is the one symmetry the flange chart does quotient."""
    pin = pytest.importorskip("pinocchio")
    spec = RobotModelSpec.default_probe45()
    try:
        model = _locked_probe45_model(spec)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rm75 reachability package unavailable")

    q = 0.35 * model.q_lower + 0.65 * model.q_upper
    features = _sweep_se3_features(
        pin, model, q, 0, {"tcp": model.tcp_id}
    )["tcp"].astype(np.float32)

    tool = spec.tool_frame().T_flange_tcp
    wrong = canonical_flange_from_se3_features(features, tool)
    fixed = canonical_flange_from_se3_features(
        features, tool, T_root_axis=spec.root_to_j1_axis()
    )
    assert float(np.max(np.ptp(wrong, axis=0))) > 0.1
    assert float(np.max(np.ptp(fixed, axis=0))) < 2.0e-5


def test_tool_frame_matches_pinocchio_flange_to_tcp_placement():
    """``T_flange_tcp`` parsed from URDF must reproduce pinocchio's own FK.

    A frame-convention error here would silently corrupt every chart value, so
    it is checked against the solver that generates the ground truth rather
    than against the URDF text it was parsed from.
    """
    pin = pytest.importorskip("pinocchio")
    spec = RobotModelSpec.default_probe45()
    try:
        model = _locked_probe45_model(spec)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rm75 reachability package unavailable")

    flange_id = model.model.getFrameId(spec.flange_frame)
    assert flange_id < model.model.nframes, f"missing frame {spec.flange_frame}"

    rng = np.random.default_rng(11)
    tool = spec.tool_frame().T_flange_tcp
    for _ in range(8):
        q = model.q_lower + rng.random(len(model.q_lower)) * (
            model.q_upper - model.q_lower
        )
        pin.forwardKinematics(model.model, model.data, q)
        pin.updateFramePlacements(model.model, model.data)
        T_flange = model.data.oMf[flange_id].homogeneous
        T_tcp = model.data.oMf[model.tcp_id].homogeneous
        assert np.allclose(T_flange @ tool, T_tcp, atol=1.0e-9)
