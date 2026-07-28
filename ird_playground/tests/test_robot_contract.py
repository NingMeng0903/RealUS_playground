from __future__ import annotations

import numpy as np
import pytest

from ird_playground.ird.canonical import canonical_from_se3_features
from ird_playground.ird.robot_model import RobotModelSpec


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
