import numpy as np
import pytest

torch = pytest.importorskip("torch")

from perception.benchmark.occlusion_rotation import (
    leg_occlusion_weights, leg_world_rotations, occlusion_rotation_loss,
)


def test_only_missing_chain_receives_occlusion_weights():
    points = np.ones((67, 4), np.float32)
    np.testing.assert_array_equal(leg_occlusion_weights(points), np.zeros(4))
    points[10, 3] = 0
    np.testing.assert_array_equal(leg_occlusion_weights(points), [0, 0, .5, .5])
    points[11, :3] = np.nan
    np.testing.assert_array_equal(leg_occlusion_weights(points), [0, 0, 1, 1])


def test_world_prior_detects_upstream_knee_rotation_even_with_unchanged_foot():
    current = torch.eye(3).repeat(1, 55, 1, 1)
    previous = leg_world_rotations(current)
    # Right foot local rotation stays identity, but knee rotates the entire foot.
    current[:, 5] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    current.requires_grad_(True)
    weights = torch.tensor([[0., 0., .5, .5]])
    loss = occlusion_rotation_loss(current, previous, weights, dt_s=.5)
    assert loss.item() > .1
    loss.backward()
    assert current.grad[:, 5].abs().sum().item() > 0
    assert current.grad[:, 4].abs().sum().item() == 0
    assert occlusion_rotation_loss(current, previous, torch.zeros_like(weights), dt_s=.5).item() == 0
    torch.testing.assert_close(
        occlusion_rotation_loss(current, previous, weights, dt_s=1.), loss / 4)
