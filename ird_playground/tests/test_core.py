"""Retained kinematics, collision, probe, and signed-operator infrastructure tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ird_playground.probe.transform import default_ultrasound_probe, load_probe_yaml


def test_default_probe_and_yaml_contract():
    probe = default_ultrasound_probe()
    assert np.allclose(probe.translation_m, [0.05, 0.0, 0.07], atol=1.0e-9)
    assert np.allclose(probe.rotation_matrix()[:, 2], [1.0, 0.0, 0.0], atol=1.0e-6)
    path = Path(__file__).resolve().parents[1] / "configs/probe_default.yaml"
    assert load_probe_yaml(path).name.startswith("ultrasound")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pinocchio") is None
    or __import__("importlib").util.find_spec("torch") is None,
    reason="pinocchio and torch are required",
)
def test_torch_rm75_fk_and_batched_ik_match_pinocchio():
    import pinocchio as pin
    import torch

    from ird_playground.ird.gt_common import reachability_modules
    from ird_playground.ird.torch_kinematics import TorchRM75Kinematics

    *_, build_locked_rail_model = reachability_modules()
    locked = build_locked_rail_model()
    kin = TorchRM75Kinematics.from_locked_model(locked, dtype=torch.float64)
    rng = np.random.default_rng(11)
    Q = locked.q_lower + rng.random((5, 7)) * (locked.q_upper - locked.q_lower)
    p_t, R_t = kin.fk(torch.as_tensor(Q, dtype=torch.float64))
    for i, q in enumerate(Q):
        pin.forwardKinematics(locked.model, locked.data, q)
        pin.updateFramePlacement(locked.model, locked.data, locked.tcp_id)
        pose = locked.data.oMf[locked.tcp_id]
        assert np.allclose(p_t[i].numpy(), pose.translation, atol=1.0e-9)
        assert np.allclose(R_t[i].numpy(), pose.rotation, atol=1.0e-9)
    target_q = torch.as_tensor(Q[:3], dtype=torch.float64)
    target_p, target_R = kin.fk(target_q)
    result = kin.ik_dls(target_p, target_R, (target_q + 0.08).clamp(kin.q_lower, kin.q_upper), max_iter=100)
    assert bool(result.ok.all())


def test_collision_checked_gpu_ik_rejects_colliding_solutions():
    import torch

    from ird_playground.ird.torch_kinematics import BatchIkResult, select_collision_free_ik

    class FakeCollisionFilter:
        def free_mask(self, q):
            return np.asarray(q)[:, 0] >= 0.0

    q = torch.zeros(2, 3, 7)
    q[0, :, 0] = torch.tensor([-0.2, 0.1, 0.3])
    q[1, :, 0] = torch.tensor([-0.3, -0.2, -0.1])
    ok = torch.ones(2, 3, dtype=torch.bool)
    pos = torch.tensor([[1.0e-5, 5.0e-5, 1.0e-4], [1.0e-5, 2.0e-5, 3.0e-5]])
    checked = select_collision_free_ik(
        BatchIkResult(q=q, ok=ok, pos_error_m=pos, rot_error_rad=torch.zeros_like(pos), iterations=1),
        FakeCollisionFilter(),
    )
    assert checked.reachable.tolist() == [True, False]
    assert int(checked.seed_index[0]) == 1
    assert torch.isnan(checked.q[1]).all()


def test_signed_production_config_loads():
    from ird_playground.neural.train_signed import load_signed_train_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_signed_train_config(root / "configs/rm4d_signed_production.yaml", root=root)
    assert cfg.lambda_signed_value == 4.0
    assert cfg.lambda_eikonal == 0.0
    assert Path(cfg.gt_npz).is_file()
