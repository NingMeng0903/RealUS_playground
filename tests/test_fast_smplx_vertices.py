from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from perception.benchmark.fast_smplx import CachedSMPLXVertexForward


@pytest.fixture(scope="module", autouse=True)
def _load_official_lbs_package():
    # The production loader installs the isolated official SMPL-X package;
    # constructing a full 518 MB body model is unnecessary for this CPU test.
    from perception.benchmark.hybrik_backend import load_official_layer

    load_official_layer()


def _fake_model(seed: int = 17):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vertices = 24
    betas = 2
    expressions = 2
    joints = 55
    parents = torch.tensor([-1] + [(index - 1) // 2 for index in range(1, joints)], dtype=torch.long)
    return SimpleNamespace(
        parents=parents,
        v_template=torch.randn((vertices, 3), generator=generator) * 0.1,
        shapedirs=torch.randn((vertices, 3, betas), generator=generator) * 0.01,
        expr_dirs=torch.randn((vertices, 3, expressions), generator=generator) * 0.01,
        posedirs=torch.randn(((joints - 1) * 9, vertices * 3), generator=generator) * 0.01,
        J_regressor=torch.softmax(torch.randn((joints, vertices), generator=generator), dim=1),
        lbs_weights=torch.softmax(torch.randn((vertices, joints), generator=generator), dim=1),
    )


def _rotations(batch: int = 2, seed: int = 29):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn((batch, 55, 3, 3), generator=generator)
    q, _ = torch.linalg.qr(raw.reshape(-1, 3, 3))
    q = q.reshape(batch, 55, 3, 3)
    determinant = torch.linalg.det(q)
    q[..., :, -1] *= torch.where(determinant < 0, -1.0, 1.0).unsqueeze(-1)
    return q.detach().contiguous()


def _official_vertices(model, betas, expression, pose, *, root_align):
    from _realus_hybrik_smplx.lbs import lbs

    shape_dirs = torch.cat([model.shapedirs, model.expr_dirs], dim=-1)
    shape_components = torch.cat([betas, expression], dim=-1)
    vertices, joints = lbs(
        shape_components,
        pose,
        model.v_template,
        shape_dirs,
        model.posedirs,
        model.J_regressor,
        model.parents,
        model.lbs_weights,
        pose2rot=False,
    )
    if root_align:
        vertices = vertices - joints[:, :1]
    return vertices


def _official_vertices_and_joints(model, betas, expression, pose, *, root_align):
    from _realus_hybrik_smplx.lbs import lbs

    shape_dirs = torch.cat([model.shapedirs, model.expr_dirs], dim=-1)
    shape_components = torch.cat([betas, expression], dim=-1)
    vertices, joints = lbs(
        shape_components,
        pose,
        model.v_template,
        shape_dirs,
        model.posedirs,
        model.J_regressor,
        model.parents,
        model.lbs_weights,
        pose2rot=False,
    )
    if root_align:
        root = joints[:, :1]
        vertices = vertices - root
        joints = joints - root
    return vertices, joints


@pytest.mark.parametrize("root_align", [False, True])
def test_cached_vertices_match_official_lbs_and_pose_gradients(root_align):
    model = _fake_model()
    betas = torch.tensor([[0.2, -0.1], [-0.05, 0.13]], dtype=torch.float32)
    expression = torch.tensor([[0.03, -0.02], [0.01, 0.04]], dtype=torch.float32)
    pose = _rotations()
    candidate = CachedSMPLXVertexForward(model, betas, expression)

    reference_pose = pose.clone().requires_grad_(True)
    candidate_pose = pose.clone().requires_grad_(True)
    reference = _official_vertices(
        model, betas, expression, reference_pose, root_align=root_align
    )
    actual = candidate(candidate_pose, root_align=root_align)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)

    weights = torch.linspace(0.5, 1.5, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    reference_grad = torch.autograd.grad((reference * weights).square().mean(), reference_pose)[0]
    candidate_grad = torch.autograd.grad((actual * weights).square().mean(), candidate_pose)[0]
    torch.testing.assert_close(candidate_grad, reference_grad, atol=1e-5, rtol=1e-5)
    assert actual.shape == (2, 24, 3)


def test_shape_and_expression_inputs_are_frozen_without_gradient_accumulation():
    model = _fake_model(seed=31)
    betas = torch.tensor([[0.1, -0.2], [0.05, 0.08]], requires_grad=True)
    expression = torch.tensor([[0.02, 0.01], [-0.01, 0.03]], requires_grad=True)
    pose = _rotations(seed=37).requires_grad_(True)
    candidate = CachedSMPLXVertexForward(model, betas, expression)

    loss = candidate(pose).square().mean()
    loss.backward()

    assert pose.grad is not None
    assert betas.grad is None
    assert expression.grad is None
    assert candidate.v_shaped.requires_grad is False
    assert candidate.rest_joints.requires_grad is False
    assert candidate.relative_joints.requires_grad is False


@pytest.mark.parametrize("root_align", [False, True])
def test_return_joints_matches_native_fk_and_is_pose_differentiable(root_align):
    model = _fake_model(seed=43)
    betas = torch.tensor([[0.12, -0.07], [-0.03, 0.09]], dtype=torch.float32)
    expression = torch.tensor([[0.01, -0.02], [0.02, 0.03]], dtype=torch.float32)
    candidate = CachedSMPLXVertexForward(model, betas, expression)

    reference_pose = _rotations(seed=47).requires_grad_(True)
    candidate_pose = reference_pose.detach().clone().requires_grad_(True)
    reference_vertices, reference_joints = _official_vertices_and_joints(
        model,
        betas,
        expression,
        reference_pose,
        root_align=root_align,
    )
    actual_vertices, actual_joints = candidate(
        candidate_pose,
        root_align=root_align,
        return_joints=True,
    )
    torch.testing.assert_close(actual_vertices, reference_vertices, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual_joints, reference_joints, atol=1e-6, rtol=1e-5)

    # Vertices and FK joints must participate in one differentiable pose graph;
    # this also catches an accidental detached/recomputed joint output.
    reference_loss = reference_vertices.square().mean() + reference_joints.square().mean()
    actual_loss = actual_vertices.square().mean() + actual_joints.square().mean()
    reference_grad = torch.autograd.grad(reference_loss, reference_pose)[0]
    actual_grad = torch.autograd.grad(actual_loss, candidate_pose)[0]
    torch.testing.assert_close(actual_grad, reference_grad, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(actual_joints).all()
    assert torch.isfinite(actual_grad).all()

    # The default remains the original vertex-only API, while __call__ accepts
    # the new keyword exactly like forward.
    default_output = candidate(candidate_pose.detach(), root_align=root_align)
    torch.testing.assert_close(default_output, actual_vertices.detach(), atol=1e-6, rtol=1e-5)
