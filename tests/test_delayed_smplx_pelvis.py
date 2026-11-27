from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from projects.genesis_ue_sync.multiview_realtime.easymocap.delayed_smplx import (
    easymocap_pelvis_world_m,
    easymocap_smplx_joints_world,
)


class _FakeBodyModel:
    device = torch.device("cpu")
    dtype = torch.float32
    model_type = "smplx"
    use_pose_blending = True
    use_shape_blending = True
    v_template = torch.zeros((2, 3), dtype=torch.float32)
    shapedirs = torch.zeros((2, 3, 20), dtype=torch.float32)
    posedirs = torch.zeros((0, 6), dtype=torch.float32)
    J_regressor = torch.zeros((55, 2), dtype=torch.float32)
    parents = torch.tensor([-1] + [0] * 54, dtype=torch.long)
    weights = torch.zeros((2, 55), dtype=torch.float32)
    J_shaped = None

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def extend_pose(self, poses):
        return poses

    def lbs(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        assert kwargs["pose2rot"] is True
        # The native SMPL-X joint 0 is deliberately distinct from Th so the
        # test proves that the fitted world pelvis is exported, rather than
        # forwarding the optimizer translation unchanged.
        base = torch.zeros((1, 55, 3), dtype=torch.float32)
        base[:, 0] = torch.tensor([1.0, 2.0, 3.0])
        base[:, 1] = torch.tensor([4.0, 5.0, 6.0])
        return torch.zeros((1, 2, 3), dtype=torch.float32), base


def _params() -> dict[str, np.ndarray]:
    return {
        "Rh": np.zeros((1, 3), dtype=np.float32),
        "Th": np.asarray([[0.25, -0.5, 0.75]], dtype=np.float32),
        "poses": np.zeros((1, 3), dtype=np.float32),
        "shapes": np.zeros((1, 10), dtype=np.float32),
        "expression": np.zeros((1, 10), dtype=np.float32),
    }


def test_native_smplx_joints_request_and_world_pelvis() -> None:
    model = _FakeBodyModel()
    joints = easymocap_smplx_joints_world(model, _params())
    np.testing.assert_allclose(joints[0], [1.25, 1.5, 3.75])
    np.testing.assert_allclose(easymocap_pelvis_world_m(model, _params()), [1.25, 1.5, 3.75])
    assert len(model.calls) == 2
    assert model.calls[0]["pose2rot"] is True
