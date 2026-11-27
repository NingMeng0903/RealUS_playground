import numpy as np
import pytest

torch = pytest.importorskip("torch")
from perception.benchmark.hybrik_backend import InitializedHybrIKX, refinement_confidence, refinement_landmarks


def test_dwpose_shoulders_use_native_fk_without_changing_other_landmarks():
    vertices = torch.arange(201, dtype=torch.float32).reshape(1, 67, 3).requires_grad_()
    native = (torch.arange(165, dtype=torch.float32).reshape(1, 55, 3) + 1000).requires_grad_()
    regressor = torch.eye(67)
    predicted = refinement_landmarks(regressor, vertices, native, convention="dwpose")
    torch.testing.assert_close(predicted[:, 5], native[:, 16])
    torch.testing.assert_close(predicted[:, 2], native[:, 17])
    other = [i for i in range(67) if i not in (2, 5)]
    torch.testing.assert_close(predicted[:, other], vertices[:, other])
    predicted.sum().backward()
    assert torch.isfinite(native.grad).all() and torch.isfinite(vertices.grad).all()
    torch.testing.assert_close(native.grad[:, [16, 17]], torch.ones((1, 2, 3)))
    torch.testing.assert_close(vertices.grad[:, [2, 5]], torch.zeros((1, 2, 3)))
    legacy = refinement_landmarks(regressor, vertices, None, convention="easymocap")
    torch.testing.assert_close(legacy, vertices)


def test_derived_dwpose_neck_is_excluded_without_filling_missing_observations():
    points = np.column_stack((np.zeros((67, 3)), np.linspace(.1, 1., 67))).astype(np.float32)
    original = points.copy()
    valid = np.ones(67, dtype=bool)
    valid[[2, 28]] = False
    confidence = refinement_confidence(points, valid, convention="dwpose")
    assert confidence[1] == confidence[2] == confidence[28] == 0
    other = [i for i in range(67) if i not in (1, 2, 28)]
    np.testing.assert_array_equal(confidence[other], points[other, 3])
    np.testing.assert_array_equal(points, original)
    legacy = refinement_confidence(points, valid, convention="easymocap")
    assert legacy[1] == points[1, 3]


def test_unknown_measurement_convention_is_rejected():
    with pytest.raises(ValueError, match="convention"):
        refinement_confidence(np.ones((67, 4)), np.ones(67, bool), convention="unknown")


@pytest.mark.parametrize("weight", [0., -1., float("nan")])
def test_invalid_landmark_weight_is_rejected_before_loading_model(weight):
    with pytest.raises(ValueError, match="weights must be positive"):
        InitializedHybrIKX(None, device="cpu", pose_initialization="previous",
                           refine_steps=80, body_landmark_weight=weight)
