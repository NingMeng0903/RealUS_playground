from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from perception.benchmark.orientation_constraints import (  # noqa: E402
    ORIENTATION_GROUPS,
    build_orientation_targets,
    orientation_loss,
)


def _observations(*, only_first: bool = False):
    points = torch.zeros((1, 67, 3), dtype=torch.float64)
    confidence = torch.zeros((1, 67), dtype=torch.float64)
    triangle = torch.tensor([[0.0, -0.10, 0.0], [0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]], dtype=points.dtype)
    for number, group in enumerate(ORIENTATION_GROUPS):
        points[:, list(group.indices)] = triangle + torch.tensor([number * 0.3, 0.0, 0.2], dtype=points.dtype)
        if not only_first or number == 0:
            confidence[:, list(group.indices)] = 1.0
    return points, confidence


def _rotate_z(points, angle: float):
    value = torch.tensor(angle, dtype=points.dtype)
    cosine, sine = torch.cos(value), torch.sin(value)
    rotation = torch.stack((
        torch.stack((cosine, -sine, value * 0)),
        torch.stack((sine, cosine, value * 0)),
        torch.tensor([0.0, 0.0, 1.0], dtype=points.dtype),
    ))
    return points @ rotation.T


def test_direction_loss_ignores_translation_and_positive_scale():
    target, confidence = _observations()
    prepared = build_orientation_targets(target, confidence)
    prediction = target * 2.4 + torch.tensor([2.0, -3.0, 0.7], dtype=target.dtype)
    torch.testing.assert_close(orientation_loss(prediction, prepared), torch.tensor(0.0, dtype=target.dtype), atol=1e-24, rtol=0)
    assert prepared.weights.shape == (1, 5)
    torch.testing.assert_close(prepared.weights, torch.ones_like(prepared.weights))


def test_direction_loss_detects_rotation_and_is_coordinate_rotation_invariant():
    target, confidence = _observations()
    prediction = _rotate_z(target, torch.pi / 2)
    loss = orientation_loss(prediction, build_orientation_targets(target, confidence))
    # A 90-degree rotation changes two unit axes by squared distance 2 each.
    torch.testing.assert_close(loss, torch.tensor(4.0 / 9.0, dtype=target.dtype))
    common_target = _rotate_z(target, 0.37)
    common_prediction = _rotate_z(prediction, 0.37)
    torch.testing.assert_close(loss, orientation_loss(common_prediction, build_orientation_targets(common_target, confidence)))


def test_three_point_minimum_confidence_controls_absolute_group_weight():
    target, confidence = _observations(only_first=True)
    prediction = _rotate_z(target, 0.5)
    full = orientation_loss(prediction, build_orientation_targets(target, confidence))
    confidence[:, ORIENTATION_GROUPS[0].indices[1]] = 0.5
    half = orientation_loss(prediction, build_orientation_targets(target, confidence))
    torch.testing.assert_close(half, full * 0.5)


@pytest.mark.parametrize("failure", ["missing", "nan", "short_width", "short_length", "collinear", "almost_collinear"])
def test_unreliable_or_degenerate_observed_groups_do_not_contribute(failure):
    target, confidence = _observations(only_first=True)
    p0, p1, p2 = ORIENTATION_GROUPS[0].indices
    if failure == "missing":
        confidence[:, p1] = 0.299
    elif failure == "nan":
        target[:, p2, 0] = float("nan")
    elif failure == "short_width":
        target[:, p1] = target[:, p2] + torch.tensor([0.01, 0.0, 0.0], dtype=target.dtype)
    elif failure == "short_length":
        target[:, p0] = (target[:, p1] + target[:, p2]) * 0.5 + torch.tensor([0.0, 0.001, 0.0], dtype=target.dtype)
    elif failure == "collinear":
        target[:, p0] = (target[:, p1] + target[:, p2]) * 0.5 + torch.tensor([0.1, 0.0, 0.0], dtype=target.dtype)
    else:
        target[:, p0] = (target[:, p1] + target[:, p2]) * 0.5 + torch.tensor([0.1, 0.01, 0.0], dtype=target.dtype)
    prepared = build_orientation_targets(target, confidence)
    prediction = torch.nan_to_num(target).clone().requires_grad_(True)
    loss = orientation_loss(prediction, prepared)
    assert prepared.weights.count_nonzero() == 0
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.count_nonzero() == 0


def test_nonzero_direction_loss_has_finite_nonzero_gradient():
    target, confidence = _observations()
    prediction = _rotate_z(target, 0.2).detach().requires_grad_(True)
    loss = orientation_loss(prediction, build_orientation_targets(target, confidence))
    loss.backward()
    assert loss.item() > 0
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    # A common translation cannot change the direction penalty.
    torch.testing.assert_close(prediction.grad.sum(dim=1), torch.zeros((1, 3), dtype=target.dtype), atol=1e-12, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_collapsed_predicted_frame_retains_penalty_and_finite_gradient(dtype):
    target, confidence = _observations(only_first=True)
    target, confidence = target.to(dtype=dtype), confidence.to(dtype=dtype)
    prediction = target.clone()
    prediction[:, list(ORIENTATION_GROUPS[0].indices)] = 0.0
    prediction.requires_grad_(True)
    loss = orientation_loss(prediction, build_orientation_targets(target, confidence))
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(1.0 / 3.0, dtype=target.dtype))
    assert torch.isfinite(prediction.grad).all()


def test_orientation_weight_validation_precedes_asset_loading():
    from perception.benchmark.hybrik_backend import InitializedHybrIKX

    with pytest.raises(ValueError, match="orientation_weight must be finite and nonnegative"):
        InitializedHybrIKX(Path("missing-init.npz"), device="cpu", orientation_weight=-0.01)
    with pytest.raises(ValueError, match="orientation_weight requires refine_steps > 0"):
        InitializedHybrIKX(Path("missing-init.npz"), device="cpu", orientation_weight=0.01)
