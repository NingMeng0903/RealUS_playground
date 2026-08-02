"""Retained kinematics, collision, probe, and signed-operator infrastructure tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ird_playground.probe.transform import default_ultrasound_probe, load_probe_yaml


def test_default_probe_and_yaml_contract():
    probe = default_ultrasound_probe()
    assert np.allclose(probe.translation_m, [0.0, -0.01523, 0.12135], atol=1.0e-9)
    assert np.isclose(
        np.degrees(np.arccos(probe.rotation_matrix()[2, 2])), 49.9002404,
        atol=1.0e-6,
    )
    path = Path(__file__).resolve().parents[1] / "configs/probe_default.yaml"
    assert load_probe_yaml(path).name == "probe45_physical"


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
    gt = root / "data/ird/rm4d_signed_production.npz"
    if not gt.is_file():
        pytest.skip(
            "rm4d_signed_production.npz archived pending Phase-3 regen after flange GT rebuild"
        )
    cfg = load_signed_train_config(root / "configs/rm4d_signed_production.yaml", root=root)
    assert cfg.lambda_signed_value > 0.0
    assert cfg.lambda_boundary_slope > 0.0
    assert cfg.lambda_generic_eikonal == 0.0
    assert cfg.sdf_target_scale > 1.0
    assert Path(cfg.gt_npz).is_file()


def test_signed_smoke_config_defaults():
    from ird_playground.ird.canonical_gt import CanonicalGtConfig
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs/rm4d_signed_smoke.yaml").read_text())
    assert float(raw["loss"]["boundary_slope"]) > 0.0
    assert float(raw["loss"]["generic_eikonal"]) == 0.0
    assert CanonicalGtConfig.__dataclass_fields__["edt_npz"] is not None


def test_task_cone_returns_recoverable_selected_pose_and_weights():
    import torch

    from ird_playground.region.task_cone import TaskConeConfig, TaskConeReachability

    class FakeField:
        def score_world(self, tcp, axis):
            del axis
            return tcp[..., 0, 0] + 0.2 * tcp[..., 1, 0]

    cone = TaskConeReachability(
        TaskConeConfig(tip_half_angle_deg=20.0, roll_half_range_deg=15.0, samples=16)
    )
    tcp = torch.eye(4).repeat(3, 1, 1)
    axis = torch.eye(4).repeat(3, 1, 1)
    result = cone(FakeField(), tcp, axis)
    assert result.sample_tcp.shape == (3, 16, 4, 4)
    assert result.free_weights.shape == (3, 16)
    assert torch.allclose(result.free_weights.sum(dim=-1), torch.ones(3))
    rows = torch.arange(3)
    assert torch.allclose(result.selected_tcp, result.sample_tcp[rows, result.best_index])
    assert torch.allclose(
        result.selected_rotvec_local,
        cone.rotation_offsets_local[result.best_index],
    )
    base, conditioned = cone.query_conditioned(
        FakeField(),
        tcp,
        axis,
        torch.full((3, 16), 0.1),
        nearest_cost=torch.arange(16, dtype=torch.float32).expand(3, -1),
        clearance_target=-1.0,
    )
    assert base.sample_clearance.shape == (3, 16)
    assert conditioned.valid.tolist() == [True, True, True]


def test_waypoint_cliff_lifts_segment_slopes_to_both_adjacent_waypoints():
    import torch

    from experiments.ellipse_vessel_ird_demo import waypoint_cliff

    s = torch.linspace(0.0, 1.0, 5)
    values = torch.tensor([0.0, 0.0, 2.0, 2.0, 2.0])
    cliff = waypoint_cliff(values, s)
    assert cliff.shape == values.shape
    assert torch.allclose(cliff, torch.tensor([0.0, 8.0, 8.0, 0.0, 0.0]))
