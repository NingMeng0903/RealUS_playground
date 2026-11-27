"""Independent mathematical regression for the approved production fix."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, '/media/camp/EXT_DRIVE/RealUS_playground/ird_playground')
from ird_playground.ird.torch_kinematics import so3_exp


def skew(x):
    z = x[..., 0] * 0.
    return torch.stack([z, -x[..., 2], x[..., 1], x[..., 2], z, -x[..., 0],
                        -x[..., 1], x[..., 0], z], dim=-1).reshape(*x.shape[:-1], 3, 3)


@pytest.mark.parametrize('r', [[0., 0., 0.], [1e-10, -2e-10, 3e-10],
    [1e-5, -2e-5, 3e-5], [.2, -.5, .7], [3.141592653589793, 0., 0.],
    [0., 0., 6.283185307179586]])
def test_value_jacobian_and_gradcheck_against_matrix_exponential(r):
    x = torch.tensor(r, dtype=torch.float64, requires_grad=True)
    assert torch.allclose(so3_exp(x), torch.matrix_exp(skew(x)), atol=2e-14, rtol=2e-14)
    assert np.allclose(so3_exp(x).detach(), Rotation.from_rotvec(r).as_matrix(), atol=2e-14)
    jac = torch.autograd.functional.jacobian(so3_exp, x)
    ref = torch.autograd.functional.jacobian(lambda y: torch.matrix_exp(skew(y)), x)
    assert torch.allclose(jac, ref, atol=2e-13, rtol=2e-12)
    assert torch.autograd.gradcheck(so3_exp, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_batched_float32_and_identity_second_derivatives():
    x = torch.tensor([[0., 0., 0.], [1e-9, -1e-8, 2e-8], [.3, .1, -.4]], requires_grad=True)
    assert torch.allclose(so3_exp(x), torch.matrix_exp(skew(x)), atol=2e-7, rtol=1e-6)
    gradient = torch.autograd.grad(so3_exp(x)[:, 2, 1].sum(), x)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0].item() == pytest.approx(1.)
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(so3_exp, (zero,), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_taylor_switch_is_continuous_to_numerical_precision():
    for factor in (1. - 1e-8, 1. + 1e-8):
        x = torch.tensor([1e-4 * factor, 0., 0.], dtype=torch.float64, requires_grad=True)
        assert torch.allclose(so3_exp(x), torch.matrix_exp(skew(x)), atol=1e-15, rtol=1e-14)
        assert torch.autograd.gradcheck(so3_exp, (x,), eps=1e-7, atol=1e-8, rtol=1e-6)
