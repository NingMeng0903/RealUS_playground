"""Unit tests for probe, region A, and neural point field (synthetic GT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ird_playground.probe.se3 import (
    complete_frame_from_tool_axis,
    delta_T_tcp_inv_base,
    features_from_delta_T,
    mat4_from_Rt,
    se3_exp,
)
from ird_playground.probe.transform import default_ultrasound_probe, load_probe_yaml
from ird_playground.region.aggregate import (
    OrientationExtent,
    PositionExtent,
    aggregate_mean_softmin,
    sample_anisotropic_xi,
    softmin,
)


def test_default_probe_composition():
    p = default_ultrasound_probe()
    assert np.allclose(p.translation_m, [0.05, 0.0, 0.07], atol=1e-9)
    R = p.rotation_matrix()
    assert np.allclose(R[:, 2], [1.0, 0.0, 0.0], atol=1e-6)


def test_probe_yaml_roundtrip(tmp_path):
    src = Path(__file__).resolve().parents[1] / "configs" / "probe_default.yaml"
    if not src.exists():
        pytest.skip("configs/probe_default.yaml missing")
    p = load_probe_yaml(src)
    assert p.name.startswith("ultrasound")


def test_delta_T_features_dim():
    R = complete_frame_from_tool_axis([0, 0, 1])
    T = mat4_from_Rt(R, [0.3, 0.1, 0.2])
    dT = delta_T_tcp_inv_base(T)
    f = features_from_delta_T(dT)
    assert f.shape == (6,)
    assert np.allclose(f[:3], [0.3, 0.1, 0.2], atol=1e-6)
    assert np.allclose(f[3:], R[:, 2], atol=1e-6)


def test_softmin_approaches_min():
    v = np.array([0.9, 0.2, 0.8])
    s = softmin(v, tau=1e-4)
    assert abs(s - 0.2) < 1e-3


def test_region_aggregate_not_mean_only():
    from ird_playground.region.aggregate import aggregate_mq

    v = np.array([1.0, 1.0, 0.0])
    q = np.array([0.8, 0.7, 0.1])
    rs = aggregate_mq(v, q, tau=0.05, lambda_q=0.5)
    assert rs.mean_score > rs.softmin_score
    assert rs.m_robust == rs.softmin_score
    assert abs(rs.q_region - float(q.mean())) < 1e-6
    assert rs.min_score == 0.0
    rs2 = aggregate_mean_softmin(v, lam=0.6, tau=0.05)
    assert rs2.softmin_score < rs2.mean_score


def test_anisotropic_extents_respect_bounds():
    xi = sample_anisotropic_xi(
        PositionExtent(0.02, 0.01, 0.002),
        OrientationExtent(8.0, 5.0, 3.0),
        64,
        seed=0,
    )
    assert xi.shape == (64, 6)
    assert np.max(np.abs(xi[:, 0])) <= 0.02 + 1e-9
    assert np.max(np.abs(xi[:, 2])) <= 0.002 + 1e-9


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_optimization_cost_uses_reach_logit():
    import torch
    from ird_playground.neural.cost import optimization_cost

    logit = torch.tensor([2.0, -2.0])
    margin = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.5, 0.5])
    c = optimization_cost(logit, margin, q)
    assert c[0] < c[1]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_lambda_rail_ad_and_robust_region():
    import torch
    from ird_playground.ird.query_base import cost_vs_lambda_rail_torch, lambda_rail_grad_ad_fd
    from ird_playground.neural.model import NeuralIRD, NeuralIRDPoint
    from ird_playground.region.local_region import local_region_cost, make_joint_sobol_ellipsoid_cone
    from ird_playground.traj.manifold import SyntheticVesselSkinManifold

    model = NeuralIRDPoint(hidden=64, depth=2, num_freqs_u=2, use_physical_pe=False)
    net = NeuralIRD(model, device="cpu")
    man = SyntheticVesselSkinManifold()

    g = lambda_rail_grad_ad_fd(net, man, n=6, seed=0)
    assert g["lambda_ad_fd_rel"] < 0.35
    assert g["rail_ad_fd_rel"] < 0.35

    lam = torch.tensor(0.15, requires_grad=True)
    rail = torch.tensor(0.0, requires_grad=True)
    cost_vs_lambda_rail_torch(net, man, lam, rail)["cost"].backward()
    assert lam.grad is not None and abs(float(lam.grad)) + abs(float(rail.grad)) > 0

    eps = make_joint_sobol_ellipsoid_cone(32, seed=0, device="cpu")
    assert eps.shape == (32, 5)
    assert torch.allclose(eps[0], torch.zeros(5))  # center included
    N = 8
    lam_c = torch.linspace(0.05, 0.35, N, requires_grad=True)
    rail_c = torch.zeros(N, requires_grad=True)
    out = local_region_cost(net, lam_c, rail_c, man, local_eps=eps)
    assert out["point_cost"].shape == (N, 32)
    out["cost"].backward()
    assert float(lam_c.grad.abs().sum()) > 0


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_p1_smoke():
    from ird_playground.neural.model import NeuralIRD, NeuralIRDPoint
    from ird_playground.traj.manifold import SyntheticVesselSkinManifold
    from ird_playground.traj.p1_optimize import P1Config, optimize_p1_lambda_rail

    net = NeuralIRD(
        NeuralIRDPoint(hidden=64, depth=2, num_freqs_u=2, use_physical_pe=False),
        device="cpu",
    )
    man = SyntheticVesselSkinManifold()
    res = optimize_p1_lambda_rail(
        net,
        man,
        cfg=P1Config(n_ctrl=5, n_knots_eval=12, region_k=16, steps=5, lr=1e-2),
    )
    assert len(res["history"]) == 5
    assert np.isfinite(res["final_loss"])
    assert res["lambda"].shape == (12,)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_train_synthetic_and_region_a(tmp_path):
    from ird_playground.neural.model import NeuralIRD
    from ird_playground.neural.train import TrainConfig, differentiability_smoke, train_point_field
    from ird_playground.region.aggregate import region_score_a

    ckpt = tmp_path / "m.pt"
    cfg = TrainConfig(
        gt_npz=None,
        synthetic_n=4096,
        epochs=25,
        batch_size=256,
        hidden=128,
        depth=3,
        num_freqs=4,
        num_freqs_u=2,
        use_physical_pe=False,
        warmup_steps=0,
        lr=3e-3,
        hardneg_every=0,
        checkpoint=str(ckpt),
        seed=0,
        num_workers=0,
        lambda_cls=1.0,
        lambda_margin=0.0,
        lambda_q=0.0,
        lambda_local=0.0,
        wandb_enable=False,
    )
    result = train_point_field(cfg)
    assert ckpt.exists()
    assert result["val_metrics"]["boundary_iou"] > 0.5
    assert result["val_metrics"]["mae_m"] < 1.5

    net = NeuralIRD.load(ckpt)
    assert differentiability_smoke(net) >= 0.0

    T_mu = mat4_from_Rt(np.eye(3), np.array([0.2, 0.0, 0.1]))
    rs = region_score_a(net, T_mu=T_mu, num_samples=16, seed=0)
    assert np.isfinite(rs.score)
    assert rs.num_samples == 16

    from ird_playground.ird.query_base import rail_y_grad_ad_fd

    rail = rail_y_grad_ad_fd(net, n=8, seed=0)
    assert rail["rail_ad_fd_rel"] < 0.5
    assert rail["rail_sign_agree"] >= 0.5


def test_load_neural_point_yaml():
    from ird_playground.neural.train import load_train_config

    root = Path(__file__).resolve().parents[1]
    cfg_a = load_train_config(root / "configs/train_phase_a.yaml", root=root)
    assert cfg_a.phase == "A"
    assert cfg_a.selection_metric == "iou"
    assert cfg_a.init_checkpoint is None
    assert cfg_a.lambda_margin == 0.0 and cfg_a.lambda_q == 0.0
    assert cfg_a.freeze_backbone_cls_epochs == 0
    assert cfg_a.feature_kind == "pu6"
    assert "phase_a" in cfg_a.checkpoint_dir

    cfg_b = load_train_config(root / "configs/train_phase_b.yaml", root=root)
    assert cfg_b.phase == "B"
    assert cfg_b.selection_metric == "joint"
    assert cfg_b.init_checkpoint is not None
    assert cfg_b.freeze_backbone_cls_epochs == 5
    assert cfg_b.freeze_cls_epochs == 5  # alias
    assert cfg_b.lambda_margin > 0 and cfg_b.lambda_q > 0
    assert cfg_b.lr_trunk is not None and cfg_b.lr_cls_head is not None
    assert cfg_b.warm_start_iou_min == 0.80
    assert "phase_b" in cfg_b.checkpoint_dir

    cfg_cont = load_train_config(root / "configs/train_continuous_pilot.yaml", root=root)
    assert cfg_cont.p_wavelengths_m is not None
    assert min(cfg_cont.p_wavelengths_m) <= 0.0015
    assert cfg_cont.early_stop_patience == 12

    # missing GT must hard-fail (not silently synthetic)
    import pytest
    import yaml

    bad = root / "configs" / "_tmp_bad_gt.yaml"
    bad.write_text(
        yaml.dump(
            {
                "data": {"gt_npz": "data/ird/does_not_exist.npz"},
                "training": {"phase": "A"},
                "loss": {"lambda_cls": 1.0, "lambda_margin": 0.0, "lambda_q": 0.0},
                "io": {
                    "checkpoint": "data/checkpoints/phase_a/selected.pt",
                    "checkpoint_dir": "data/checkpoints/phase_a",
                    "report": "data/reports/x.json",
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        with pytest.raises(FileNotFoundError):
            load_train_config(bad, root=root)
    finally:
        bad.unlink(missing_ok=True)


def test_source_resolution_gate_uses_capability_manifest(tmp_path):
    from ird_playground.ird.precision import source_resolution_report
    import yaml

    map_dir = tmp_path / "map"
    map_dir.mkdir()
    (map_dir / "manifest.yaml").write_text(
        yaml.safe_dump({"grid": {"step_m": 0.015}}), encoding="utf-8"
    )
    gt = tmp_path / "gt.npz"
    gt.touch()
    gt.with_suffix(".yaml").write_text(
        yaml.safe_dump({"map_dir": str(map_dir)}), encoding="utf-8"
    )
    report = source_resolution_report(gt, target_position_error_m=1.0e-4)
    assert report["source_resolution_known"] is True
    assert report["source_boundary_lower_bound_m"] == pytest.approx(0.0075)
    assert report["source_resolution_pass"] is False


def test_source_resolution_gate_uses_continuous_gt_tolerance(tmp_path):
    from ird_playground.ird.precision import source_resolution_report
    import yaml

    gt = tmp_path / "continuous.npz"
    gt.touch()
    gt.with_suffix(".yaml").write_text(
        yaml.safe_dump(
            {
                "label_kind": "continuous_fk_multiseed_ik_se3_bisection_v1",
                "physical_boundary_tolerance_m": 2.0e-4,
                "ik_position_tolerance_m": 5.0e-5,
            }
        ),
        encoding="utf-8",
    )
    report = source_resolution_report(gt, target_position_error_m=2.0e-4)
    assert report["source_resolution_known"] is True
    assert report["source_resolution_pass"] is True
    assert report["source_boundary_lower_bound_m"] == pytest.approx(2.0e-4)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pinocchio") is None,
    reason="pinocchio not installed",
)
def test_continuous_fk_ik_gt_smoke():
    from ird_playground.ird.continuous_gt import ContinuousGtConfig, build_continuous_ird_gt
    from ird_playground.ird.export_gt import assert_gt_contract

    arrays, meta = build_continuous_ird_gt(
        ContinuousGtConfig(
            n_fk_interior=8,
            n_boundary_rays=6,
            n_random_seeds=1,
            ray_max_pos_m=0.20,
            ray_max_rot_deg=20.0,
            boundary_tol_m=5.0e-4,
            ik_tol_pos_m=1.0e-4,
            ik_max_iter=50,
            seed=3,
        )
    )
    assert_gt_contract(arrays)
    assert meta["n_boundary_rays_accepted"] > 0
    assert arrays["label_kind"][0] == 4
    assert "boundary_id" in arrays and "boundary_signed_m" in arrays


def test_dmp_task_requires_explicit_calibration(tmp_path):
    from ird_playground.traj.dmp_task import load_dmp_task_spec, load_task_tcp_poses
    import yaml

    traj = tmp_path / "traj.npz"
    np.savez(
        traj,
        s=np.array([0.0, 1.0]),
        p_target=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        r_target=np.repeat(np.eye(3)[None, :, :], 2, axis=0),
    )
    cfg = tmp_path / "task.yaml"
    cfg.write_text(yaml.safe_dump({"trajectory_npz": str(traj)}), encoding="utf-8")
    with pytest.raises(ValueError, match="T_arm_base_from_patient"):
        load_dmp_task_spec(cfg)
    cfg.write_text(
        yaml.safe_dump({"trajectory_npz": str(traj), "T_arm_base_from_patient": np.eye(4).tolist()}),
        encoding="utf-8",
    )
    phase, poses = load_task_tcp_poses(load_dmp_task_spec(cfg))
    assert phase.shape == (2,) and poses.shape == (2, 4, 4)


def test_feature_spec_and_warm_start_mismatch(tmp_path):
    import torch

    from ird_playground.neural.feature_spec import make_feature_spec
    from ird_playground.neural.model import NeuralIRDPoint
    from ird_playground.neural.train import TrainConfig, validate_phase_config

    assert make_feature_spec("pu6").dim == 6
    assert make_feature_spec("pu_roll8").dim == 8
    assert make_feature_spec("pu_roll8").use_roll

    m6 = NeuralIRDPoint(feature_kind="pu6", hidden=64, depth=2)
    m8 = NeuralIRDPoint(feature_kind="pu_roll8", hidden=64, depth=2)
    assert m6.in_dim == 6 and m8.in_dim == 8
    x8 = torch.zeros(2, 8)
    x8[:, 5] = 1.0
    x8[:, 6] = 1.0
    out = m8(x8)
    assert out[0].shape == (2, 1)

    # architecture mismatch must raise on warm-start load path
    bad_ck = tmp_path / "bad_hidden.pt"
    torch.save(
        {
            "state_dict": NeuralIRDPoint(feature_kind="pu6", hidden=32, depth=2).state_dict(),
            "model_cfg": {
                "in_dim": 6,
                "feature_kind": "pu6",
                "num_freqs": 6,
                "num_freqs_u": 5,
                "hidden": 32,
                "depth": 2,
                "use_physical_pe": True,
            },
        },
        bad_ck,
    )
    model = NeuralIRDPoint(feature_kind="pu6", hidden=64, depth=2)
    blob = torch.load(bad_ck, map_location="cpu", weights_only=False)
    raised = False
    try:
        if blob["model_cfg"]["hidden"] != model.hidden:
            raise RuntimeError("Warm-start architecture mismatch")
        model.load_state_dict(blob["state_dict"], strict=True)
    except RuntimeError:
        raised = True
    assert raised

    cfg = TrainConfig(phase="B", init_checkpoint=None, lambda_margin=0.1)
    try:
        validate_phase_config(cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_phase_b_freeze_groups():
    import torch

    from ird_playground.neural.model import NeuralIRDPoint

    model = NeuralIRDPoint(feature_kind="pu6", hidden=64, depth=2)
    trunk = list(model.stem.parameters())
    for b in model.blocks:
        trunk.extend(list(b.parameters()))
    cls = list(model.head_cls.parameters())
    aux = list(model.head_margin.parameters()) + list(model.head_q.parameters())
    for p in trunk + cls:
        p.requires_grad = False
    for p in aux:
        p.requires_grad = True
    assert not next(model.stem.parameters()).requires_grad
    assert not next(model.head_cls.parameters()).requires_grad
    assert next(model.head_margin.parameters()).requires_grad
    for p in trunk + cls:
        p.requires_grad = True
    assert all(p.requires_grad for p in model.stem.parameters())


def test_ird_viz_gt_only(tmp_path):
    from ird_playground.ird.export_gt import make_synthetic_ird_gt
    from ird_playground.viz.ird_compare import features_to_xyz, render_ird_comparison

    arrays = make_synthetic_ird_gt(2000, seed=0)
    out = tmp_path / "ird.png"
    render_ird_comparison(
        xyz=features_to_xyz(arrays["features"]),
        gt=arrays["d"],
        pred=None,
        out_path=out,
        max_points=1500,
    )
    assert out.exists() and out.stat().st_size > 1000


def test_se3_exp_identity():
    T = se3_exp(np.zeros(6))
    assert np.allclose(T, np.eye(4), atol=1e-9)
