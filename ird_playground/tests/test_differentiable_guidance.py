from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from ird_playground.optimization.differentiable_energy import (
    DifferentiableTrajectoryEnergy,
    TrajectoryEnergyConfig,
    cubic_bspline_basis,
    cubic_bspline_matrices,
    gauss_legendre_path_samples,
    optimize_guidance_controls,
)
from ird_playground.region.task_cone import TaskConeConfig, TaskConeReachability


class _AnalyticField(torch.nn.Module):
    def score_world(self, tcp: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        del axis
        p = tcp[..., :3, 3]
        # Clearance remains above five and still has a useful spatial gradient.
        return 8.0 + 20.0 * p[..., 0] - 3.0 * p[..., 2]


def _pose_decoder(theta: torch.Tensor, path_y: torch.Tensor) -> torch.Tensor:
    batch, width = theta.shape
    T = torch.eye(4, dtype=theta.dtype, device=theta.device).repeat(batch, width, 1, 1)
    T[..., 0, 3] = 0.1 * torch.cos(theta)
    T[..., 1, 3] = path_y
    T[..., 2, 3] = 0.1 * torch.sin(theta)
    T[..., 0, 0] = torch.cos(theta)
    T[..., 0, 2] = torch.sin(theta)
    T[..., 2, 0] = -torch.sin(theta)
    T[..., 2, 2] = torch.cos(theta)
    return T


def _energy(obstacle_x: float = 0.4, *, dtype=torch.float64):
    width = 11
    basis = cubic_bspline_basis(np.linspace(0.0, 1.0, width), 5)
    return DifferentiableTrajectoryEnergy(
        _AnalyticField(),
        basis=torch.as_tensor(basis, dtype=dtype),
        baseline_theta=torch.linspace(-0.2, 0.2, width, dtype=dtype),
        path_y=torch.linspace(0.0, 0.1, width, dtype=dtype),
        baseline_rail=torch.full((width,), 0.4, dtype=dtype),
        obstacle_centers=torch.tensor([[obstacle_x, 0.05, 0.0]] * width, dtype=dtype),
        T_rail_axis0=torch.eye(4, dtype=dtype),
        pose_decoder=_pose_decoder,
        config=TrajectoryEnergyConfig(clearance_output_scale=10.0),
    ).to(dtype=dtype)


def test_batched_shapes_bounds_and_all_guidance_gradients_are_finite():
    energy = _energy()
    controls = torch.randn(3, 5, 5, dtype=torch.float64, requires_grad=True)
    output = energy(controls)
    assert output.energy.shape == (3,)
    assert output.raw_clearance.shape == (3, 11)
    assert output.decoded.tcp.shape == (3, 11, 4, 4)
    assert output.decoded.normalized_velocity.shape == (3, 10, 5)
    assert output.decoded.normalized_curvature.shape == (3, 9, 5)
    assert torch.all(output.decoded.tip_xy.norm(dim=-1) <= np.deg2rad(20.0) + 1e-10)
    assert torch.all(output.decoded.roll.abs() <= np.deg2rad(20.0) + 1e-10)
    assert torch.all((output.decoded.rail >= 0.0) & (output.decoded.rail <= 0.8))
    output.energy.sum().backward()
    assert controls.grad is not None
    assert torch.isfinite(controls.grad).all()
    assert torch.count_nonzero(controls.grad).item() > 0


def test_obstacle_context_does_not_modify_raw_ird():
    controls = torch.zeros(2, 5, 5, dtype=torch.float64)
    far = _energy(0.4)(controls)
    near = _energy(0.1)(controls)
    assert torch.equal(far.raw_clearance, near.raw_clearance)
    assert not torch.equal(far.obstacle_signed_distance, near.obstacle_signed_distance)


def test_current_controls_recompute_all_live_query_channels_and_gradient():
    class PoseRailField(torch.nn.Module):
        def score_world(self, tcp, axis):
            return (
                8.0 + 10.0 * tcp[..., 0, 3] + 2.0 * tcp[..., 0, 2]
                + 3.0 * axis[..., 1, 3]
            )

    width = 11
    basis = cubic_bspline_basis(np.linspace(0.0, 1.0, width), 5)
    energy = DifferentiableTrajectoryEnergy(
        PoseRailField(),
        basis=torch.as_tensor(basis, dtype=torch.float64),
        baseline_theta=torch.linspace(-0.2, 0.2, width, dtype=torch.float64),
        path_y=torch.linspace(0.0, 0.1, width, dtype=torch.float64),
        baseline_rail=torch.full((width,), 0.4, dtype=torch.float64),
        obstacle_centers=torch.tensor([[0.08, 0.05, 0.02]] * width, dtype=torch.float64),
        T_rail_axis0=torch.eye(4, dtype=torch.float64),
        pose_decoder=_pose_decoder,
        angle_query=TaskConeReachability(TaskConeConfig(samples=64, seed=11)),
        config=TrajectoryEnergyConfig(clearance_output_scale=10.0),
    ).to(dtype=torch.float64)
    initial = torch.zeros(1, 5, 5, dtype=torch.float64, requires_grad=True)
    changed = initial.detach().clone()
    changed[0, 2] = torch.tensor([0.35, 0.12, -0.08, 0.17, 0.25])
    changed.requires_grad_(True)
    output0 = energy(initial)
    output1 = energy(changed)
    for left, right in (
        (output0.decoded.tcp, output1.decoded.tcp),
        (output0.decoded.rail, output1.decoded.rail),
        (output0.raw_clearance, output1.raw_clearance),
        (output0.angle_conditioned_clearance, output1.angle_conditioned_clearance),
        (output0.obstacle_signed_distance, output1.obstacle_signed_distance),
        (output0.conditioned_clearance, output1.conditioned_clearance),
    ):
        assert not torch.allclose(left, right)
    grad0 = torch.autograd.grad(output0.conditioned_clearance.sum(), initial)[0]
    grad1 = torch.autograd.grad(output1.conditioned_clearance.sum(), changed)[0]
    assert torch.isfinite(grad0).all() and torch.isfinite(grad1).all()
    assert torch.count_nonzero(grad0).item() > 0
    assert not torch.allclose(grad0, grad1)


def test_reachability_gradient_is_nonzero_above_acceptance_threshold():
    energy = _energy()
    controls = torch.zeros(1, 5, 5, dtype=torch.float64, requires_grad=True)
    output = energy(controls)
    assert output.minimum_clearance.item() > 5.0
    grad = torch.autograd.grad(output.regrets["reachability"].sum(), controls)[0]
    assert torch.linalg.vector_norm(grad[..., 0]).item() > 1e-7


def test_theta_guidance_ad_matches_finite_difference():
    energy = _energy()
    controls = torch.zeros(1, 5, 5, dtype=torch.float64, requires_grad=True)
    value = energy(controls).energy.sum()
    ad = torch.autograd.grad(value, controls)[0][0, 2, 0].item()
    eps = 1e-5
    plus = controls.detach().clone(); plus[0, 2, 0] += eps
    minus = controls.detach().clone(); minus[0, 2, 0] -= eps
    fd = ((energy(plus).energy - energy(minus).energy) / (2.0 * eps)).item()
    assert np.isfinite(ad) and np.isfinite(fd)
    assert np.isclose(ad, fd, rtol=2e-3, atol=2e-6)


def test_forward_graph_has_no_discrete_or_detaching_operations():
    source = Path(__file__).parents[1] / "ird_playground/optimization/differentiable_energy.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DifferentiableTrajectoryEnergy")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"decode", "forward"}]
    text = ast.unparse(ast.Module(body=methods, type_ignores=[]))
    for forbidden in ("detach", "no_grad", ".numpy", "argmax", "argmin"):
        assert forbidden not in text


def test_basis_rejects_extrapolation_and_nonconvex_rows():
    for samples in ([-0.01, 0.5, 1.0], [0.0, 0.5, 1.01]):
        try:
            cubic_bspline_basis(np.asarray(samples), 5)
        except ValueError as exc:
            assert "[0,1]" in str(exc)
        else:
            raise AssertionError("B-spline extrapolation was accepted")
    energy = _energy()
    bad = energy.basis.detach().clone(); bad[0] = 0.0; bad[0, :2] = torch.tensor([2.0, -1.0])
    try:
        DifferentiableTrajectoryEnergy(
            _AnalyticField(), basis=bad, baseline_theta=energy.baseline_theta,
            path_y=energy.path_y, baseline_rail=energy.baseline_rail,
            obstacle_centers=energy.obstacle_centers, T_rail_axis0=energy.T_rail_axis0,
            pose_decoder=_pose_decoder,
        )
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("non-convex basis row was accepted")


def test_analytic_spline_derivatives_do_not_depend_on_sample_spacing():
    controls = torch.zeros(1, 5, 5, dtype=torch.float64)
    controls[0, :, 0] = torch.linspace(-0.4, 0.5, 5)
    controls[0, :, 4] = torch.linspace(-0.2, 0.3, 5)
    values = []
    for order in (4, 6):
        samples, weights = gauss_legendre_path_samples(np.linspace(0.0, 1.0, 5), order=order)
        B, dB, ddB = cubic_bspline_matrices(samples, 5)
        width = len(samples)
        energy = DifferentiableTrajectoryEnergy(
            _AnalyticField(), basis=B, velocity_basis=dB, curvature_basis=ddB,
            baseline_theta=torch.zeros(width), path_y=torch.linspace(0.0, 0.1, width),
            baseline_rail=torch.full((width,), 0.4),
            obstacle_centers=torch.tensor([[0.4, 0.05, 0.0]] * width),
            T_rail_axis0=torch.eye(4), pose_decoder=_pose_decoder,
            quadrature_weights=torch.as_tensor(weights),
        ).to(dtype=torch.float64)
        values.append((energy(controls).regrets["continuity"].item(), energy(controls).regrets["curvature"].item()))
    assert np.allclose(values[0], values[1], atol=1e-12)


def test_field_parameters_are_frozen_by_energy_contract():
    class ParametricField(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(2.0))
        def score_world(self, tcp, axis):
            del axis
            return 8.0 + self.scale * tcp[..., 0, 3]
    field = ParametricField()
    base = _energy()
    energy = DifferentiableTrajectoryEnergy(
        field, basis=base.basis, baseline_theta=base.baseline_theta,
        path_y=base.path_y, baseline_rail=base.baseline_rail,
        obstacle_centers=base.obstacle_centers, T_rail_axis0=base.T_rail_axis0,
        pose_decoder=_pose_decoder,
    ).to(dtype=torch.float64)
    controls = torch.zeros(1, 5, 5, dtype=torch.float64, requires_grad=True)
    energy(controls).energy.sum().backward()
    assert field.scale.grad is None
    assert controls.grad is not None


def test_batched_dynamic_obstacle_context_has_finite_gradient_and_preserves_raw_ird():
    energy = _energy()
    controls = torch.zeros(2, 5, 5, dtype=torch.float64, requires_grad=True)
    context = torch.tensor(
        [[[0.11, 0.05, 0.0]] * 11, [[0.35, 0.05, 0.0]] * 11],
        dtype=torch.float64, requires_grad=True,
    )
    contextual = energy(controls, obstacle_centers=context)
    baseline = energy(controls.detach())
    assert torch.equal(contextual.raw_clearance, baseline.raw_clearance)
    grad = torch.autograd.grad(contextual.regrets["obstacle"].sum(), context)[0]
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() > 0


def test_optimizer_preserves_fixed_control_values():
    energy = _energy(dtype=torch.float32)
    initial = torch.randn(2, 5, 5, dtype=torch.float32)
    fixed = torch.zeros(1, 5, 5, dtype=torch.float32)
    mask = torch.zeros(1, 5, 1, dtype=torch.bool)
    mask[:, (0, 4)] = True
    result = optimize_guidance_controls(
        energy, initial, max_steps=6, learning_rate=0.01,
        fixed_control_mask=mask, fixed_control_values=fixed,
    )
    assert torch.equal(result.controls[:, (0, 4)], torch.zeros(2, 2, 5))


def test_optimizer_requires_all_three_stages():
    energy = _energy(dtype=torch.float32)
    initial = torch.zeros(1, 5, 5, dtype=torch.float32)
    try:
        optimize_guidance_controls(energy, initial, max_steps=2)
    except ValueError as exc:
        assert "three-stage" in str(exc)
    else:
        raise AssertionError("two-step run bypassed the three-stage contract")
