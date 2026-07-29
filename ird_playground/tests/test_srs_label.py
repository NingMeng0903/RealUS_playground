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
    R = np.asarray(pose.rotation, dtype=float)
    p = np.asarray(pose.translation, dtype=float)
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



def test_srs_label_requires_branch_id():
    from ird_playground.ird.srs_label import SrsLabelConfig

    with pytest.raises(TypeError):
        SrsLabelConfig()  # type: ignore[call-arg]
