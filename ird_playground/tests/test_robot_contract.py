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


def test_axis_counterexample_8d_ambiguity_resolved_by_9d():
    """On the J1 axis the old 8-D embedding collides; 9-D separates the orbits.

    Constructed pair: identical ``(p_z, r, u_x·ẑ, u_z·ẑ)`` and zero mixed
    products, but opposite ``u_y·ẑ``.  Orbit distance is large while the
    truncated 8-D embedding difference is exactly zero.
    """
    torch = pytest.importorskip("torch")
    from ird_playground.ird.canonical import (
        FLANGE_CANONICAL_DIM,
        FLANGE_RADIAL_EPS,
        canonical_flange_invariants_torch,
    )

    def rotation_with_third_row(third_row: np.ndarray) -> np.ndarray:
        n = np.asarray(third_row, dtype=np.float64)
        n = n / np.linalg.norm(n)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        r0 = tmp - n * np.dot(tmp, n)
        r0 = r0 / np.linalg.norm(r0)
        r1 = np.cross(n, r0)  # r0 × r1 = n  → right-handed rows
        return np.stack((r0, r1, n), axis=0)

    a, b = 0.3, 0.8
    c = float(np.sqrt(max(0.0, 1.0 - a * a - b * b)))
    R_a = rotation_with_third_row(np.array([a, b, c]))
    R_b = rotation_with_third_row(np.array([a, -b, c]))
    p = np.array([0.0, 0.0, 0.35], dtype=np.float64)

    p_t = torch.as_tensor(np.stack((p, p)), dtype=torch.float64)
    R_t = torch.as_tensor(np.stack((R_a, R_b)), dtype=torch.float64)
    emb = canonical_flange_invariants_torch(p_t, R_t).numpy()
    assert emb.shape[-1] == FLANGE_CANONICAL_DIM

    emb8_a, emb8_b = emb[0, :8], emb[1, :8]
    assert float(np.max(np.abs(emb8_a - emb8_b))) == pytest.approx(0.0, abs=1e-12)
    assert abs(float(emb[0, 8] - emb[1, 8])) > 0.5
    assert float(emb[0, 8]) == pytest.approx(b, abs=1e-9)
    assert float(emb[1, 8]) == pytest.approx(-b, abs=1e-9)

    # Finite derivative of r = sqrt(px²+py²+eps) at the axis tip.
    p0 = torch.tensor([[0.0, 0.0, 0.35]], dtype=torch.float64, requires_grad=True)
    R0 = torch.as_tensor(R_a[None], dtype=torch.float64)
    e0 = canonical_flange_invariants_torch(p0, R0)
    e0[0, 1].backward()
    assert p0.grad is not None
    assert torch.isfinite(p0.grad).all()
    # At px=py=0, ∂r/∂p = 0 and r = sqrt(eps).
    assert float(e0[0, 1].detach()) == pytest.approx(np.sqrt(FLANGE_RADIAL_EPS), rel=0, abs=1e-12)
    assert float(torch.linalg.vector_norm(p0.grad[0, :2])) < 1e-9


def test_collision_guard_rejects_side_branch_under_rail_base(tmp_path):
    """Subtree-complement guard must catch gantry-like side-branch collisions.

    An ancestor-only walk would silently allow a ``cable_carrier`` hanging off
    ``rail_base``; the complement of ``joint_1``'s subtree must reject it.
    ``base_link`` remains whitelisted.
    """
    import shutil
    import xml.etree.ElementTree as ET

    from ird_playground.ird.robot_model import RobotModelSpec

    spec = RobotModelSpec.default_probe45()
    mutated = tmp_path / "side_branch.collision.urdf"
    shutil.copy(spec.collision_urdf, mutated)
    root = ET.parse(mutated).getroot()
    carrier = ET.SubElement(root, "link", {"name": "cable_carrier"})
    coll = ET.SubElement(carrier, "collision")
    geom = ET.SubElement(coll, "geometry")
    ET.SubElement(geom, "box", {"size": "0.05 0.05 0.05"})
    joint = ET.SubElement(root, "joint", {"name": "rail_to_cable_carrier", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": "rail_base"})
    ET.SubElement(joint, "child", {"link": "cable_carrier"})
    ET.SubElement(joint, "origin", {"xyz": "0.3 0 0", "rpy": "0 0 0"})
    ET.ElementTree(root).write(mutated, encoding="unicode")

    bad = RobotModelSpec(
        kinematics_urdf=spec.kinematics_urdf,
        collision_urdf=mutated,
        collision_pairs=spec.collision_pairs,
    )
    with pytest.raises(ValueError, match="cable_carrier"):
        bad.assert_base_yaw_invariant_collision_model()

    # Whitelist still exempts base_link on the stock collision URDF.
    spec.assert_base_yaw_invariant_collision_model()

