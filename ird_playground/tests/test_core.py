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
    # TCP +Z should align with link7 +X
    assert np.allclose(R[:, 2], [1.0, 0.0, 0.0], atol=1e-6)


def test_probe_yaml_roundtrip(tmp_path):
    root = tmp_path / "probe.yaml"
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "configs"
        / "probe_default.yaml"
    )
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
    # legacy wrapper still exposes softmin < mean
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
        warmup_steps=0,
        lr=3e-3,
        hardneg_every=0,
        checkpoint=str(ckpt),
        seed=0,
        num_workers=0,
        lambda_local=0.0,
        wandb_enable=False,
    )
    result = train_point_field(cfg)
    assert ckpt.exists()
    assert result["val_metrics"]["boundary_iou"] > 0.5
    assert result["val_metrics"]["mae_m"] < 1.5

    net = NeuralIRD.load(ckpt)
    g = differentiability_smoke(net)
    assert g >= 0.0

    T_mu = mat4_from_Rt(np.eye(3), np.array([0.2, 0.0, 0.1]))
    rs = region_score_a(net, T_mu=T_mu, num_samples=16, seed=0)
    assert np.isfinite(rs.score)
    assert np.isfinite(rs.m_robust)
    assert 0.0 <= rs.q_region <= 1.0
    assert rs.num_samples == 16

    from ird_playground.ird.query_base import rail_y_grad_ad_fd

    rail = rail_y_grad_ad_fd(net, n=8, seed=0)
    assert rail["rail_ad_fd_rel"] < 0.5
    assert rail["rail_sign_agree"] >= 0.5


def test_load_neural_point_yaml():
    from ird_playground.neural.train import load_train_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_train_config(root / "configs/train_config.yaml", root=root)
    assert cfg.epochs >= 1
    assert cfg.hidden >= 64
    assert cfg.batch_size >= 1


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
