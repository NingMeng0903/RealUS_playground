"""Smoke tests for controller-aligned SRS point labeling."""

from __future__ import annotations

import numpy as np
import pytest


def test_srs_label_imports_offline():
    """Offline env must import without the Robotic_Arm SDK."""
    import ird_playground.ird.srs_label as mod

    assert hasattr(mod, "SrsLabelConfig")
    assert hasattr(mod, "srs_reachable_single")


def test_srs_label_fk_pose_branch_gate():
    """Correct branch ⇒ reachable; fixed branch is respected (no silent OR)."""
    pin = pytest.importorskip("pinocchio")
    from ird_playground.ird.gt_common import reachability_modules
    from ird_playground.ird.robot_model import RobotModelSpec
    from ird_playground.ird.srs_label import (
        SrsLabelConfig,
        branch_and_psi_from_q7,
        srs_reachable_single,
        _srs_api,
    )

    *_, build_locked_rail_model = reachability_modules()[1:]
    spec = RobotModelSpec.default_probe45()
    model = build_locked_rail_model(
        spec.kinematics_urdf,
        rail_locked_at_m=spec.rail_locked_at_m,
        tcp_frame=spec.tcp_frame,
    )
    # Safe posture away from shoulder/wrist gimbal.
    q = np.array([0.30, 0.70, -0.20, 0.90, 0.15, 0.60, -0.40], dtype=float)
    pin.forwardKinematics(model.model, model.data, q)
    pin.updateFramePlacements(model.model, model.data)
    pose = model.data.oMf[model.tcp_id]
    R = np.asarray(pose.rotation, dtype=float).copy()
    p = np.asarray(pose.translation, dtype=float).copy()
    branch, psi_home = branch_and_psi_from_q7(q)

    srs = _srs_api()
    y_rail = float(srs.shoulder_y_from_q_rail(spec.rail_locked_at_m))

    cfg_ok = SrsLabelConfig(branch_id=branch, psi_home_rad=psi_home, y_rail_m=y_rail)
    ok, best_psi, best_b, q_out = srs_reachable_single(R, p, cfg_ok)
    assert ok, "self-FK pose must be point-reachable on the generating branch"
    assert best_b == branch
    assert q_out is not None
    assert best_psi is not None
    assert abs(best_psi - psi_home) <= cfg_ok.psi_grid_step_rad + 1e-9
    assert srs.branch_from_q(q_out) == branch
    # Round-trip through the independent URDF FK validates the actual angled
    # probe transform, not merely a successful analytical IK return value.
    pin.forwardKinematics(model.model, model.data, q_out)
    pin.updateFramePlacements(model.model, model.data)
    recovered = model.data.oMf[model.tcp_id]
    assert np.linalg.norm(np.asarray(recovered.translation) - p) < 0.0005
    from scipy.spatial.transform import Rotation
    assert Rotation.from_matrix(np.asarray(recovered.rotation) @ R.T).magnitude() < np.deg2rad(1.0)

    # Alternate branches may still reach the same TCP (SRS redundancy), but the
    # labeler must lock to the requested branch — never silently OR all eight.
    wrong = (branch ^ 0b001) & 0b111  # flip wrist bit
    cfg_alt = SrsLabelConfig(branch_id=wrong, psi_home_rad=psi_home, y_rail_m=y_rail)
    ok_alt, _, best_alt, q_alt = srs_reachable_single(R, p, cfg_alt)
    if ok_alt:
        assert best_alt == wrong
        assert srs.branch_from_q(q_alt) == wrong
        assert not np.allclose(q_alt, q_out, atol=1e-3)
    # Manifest records point-only semantics.
    man = cfg_ok.to_manifest()
    assert man["reachability_kind"] == "point"
    assert man["models_path_reachability"] is False
    assert man["branch_id"] == branch
    assert man["tool_mode"] == srs.TOOL_MODE_FLANGE_TCP
    np.testing.assert_allclose(man["T_flange_tcp"], spec.tool_frame().T_flange_tcp, atol=1e-12)
    np.testing.assert_allclose(man["tcp_offset_xyz"], spec.tool_frame().T_flange_tcp[:3, 3], atol=1e-12)



def test_srs_label_requires_branch_id():
    from ird_playground.ird.srs_label import SrsLabelConfig

    with pytest.raises(TypeError):
        SrsLabelConfig()  # type: ignore[call-arg]


def test_srs_label_passes_custom_rigid_tool_without_legacy_enum(monkeypatch):
    """The full configured transform reaches every candidate, including rotation."""
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    import ird_playground.ird.srs_label as mod

    calls = []
    def ik(pose, psi, branch, **kwargs):
        calls.append((np.array(pose), branch, kwargs))
        return np.zeros(7)

    # Deliberately provides only the real public enum, not the removed alias.
    fake = SimpleNamespace(TOOL_MODE_FLANGE_TCP="flange_tcp", srs_ik=ik)
    monkeypatch.setattr(mod, "_srs_api", lambda: fake)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.31, -0.43, 0.72]).as_matrix()
    transform[:3, 3] = [0.012, -0.026, 0.151]
    cfg = mod.SrsLabelConfig(branch_id=5, y_rail_m=-0.2, T_flange_tcp=transform)
    ok, _, branch, _ = mod.srs_reachable_single(np.eye(3), np.array([0.1, 0.2, 0.3]), cfg)
    assert ok and branch == 5 and calls
    for pose, requested_branch, kwargs in calls:
        assert requested_branch == 5
        assert kwargs["tool_mode"] == "flange_tcp"
        np.testing.assert_allclose(kwargs["R_flange_tcp"], transform[:3, :3])
        np.testing.assert_allclose(kwargs["t_flange_tcp"], transform[:3, 3])
        np.testing.assert_allclose(pose[:3], [0.1, 0.2, 0.3])


def test_gpu_builder_uses_supplied_robot_tool_transform():
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from ird_playground.ird.gpu_pose_gt import _make_srs_cfg

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("y", 0.37).as_matrix()
    transform[:3, 3] = [0.02, -0.03, 0.19]
    spec = SimpleNamespace(rail_locked_at_m=0.21,
                           tool_frame=lambda: SimpleNamespace(T_flange_tcp=transform))
    cfg = _make_srs_cfg(spec, 6)
    assert cfg.branch_id == 6
    assert cfg.y_rail_m == pytest.approx(-0.19)
    np.testing.assert_allclose(cfg.T_flange_tcp, transform)


def test_srs_label_rejects_reflected_tool_frame():
    from ird_playground.ird.srs_label import SrsLabelConfig
    transform = np.eye(4)
    transform[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        SrsLabelConfig(branch_id=0, T_flange_tcp=transform)
