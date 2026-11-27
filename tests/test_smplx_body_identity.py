from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from projects.genesis_ue_sync.sim_platform.datasets import human_sequence as hs
from projects.genesis_ue_sync.sim_platform.human_runtime import gt_smpl_display as display


def sequence(*, betas_dim=16, width=165, metadata=None, frames=2):
    return hs.HumanMotionSequence(
        source_dataset="test", sequence_name="identity", source_path="", model_type="smplx",
        fps=30, gender="neutral", betas=np.arange(betas_dim, dtype=np.float32) / 10,
        poses=np.arange(frames * width, dtype=np.float32).reshape(frames, width),
        trans=np.arange(frames * 3, dtype=np.float32).reshape(frames, 3) / 10,
        metadata=dict(metadata or {}),
    )


def test_native55_face_and_hands_are_distinct_slots():
    seq = sequence(metadata={"pose_layout": "smplx55"})
    kwargs = hs._build_smpl_kwargs(seq, torch_device="cpu", frame_slice=slice(1, 2))
    for name, expected in {
        "global_orient": seq.poses[1:2, :3], "body_pose": seq.poses[1:2, 3:66],
        "jaw_pose": seq.poses[1:2, 66:69], "leye_pose": seq.poses[1:2, 69:72],
        "reye_pose": seq.poses[1:2, 72:75], "left_hand_pose": seq.poses[1:2, 75:120],
        "right_hand_pose": seq.poses[1:2, 120:165], "betas": seq.betas[None], "transl": seq.trans[1:2],
    }.items():
        np.testing.assert_array_equal(kwargs[name].numpy(), expected)


@pytest.mark.parametrize("width", [156, 165])
def test_unmarked_legacy_arrays_keep_their_original_layout(width):
    seq = sequence(width=width, betas_dim=10)
    kwargs = hs._build_smpl_kwargs(seq, torch_device="cpu")
    np.testing.assert_array_equal(kwargs["left_hand_pose"].numpy(), seq.poses[:, 66:111])
    np.testing.assert_array_equal(kwargs["right_hand_pose"].numpy(), seq.poses[:, 111:156])
    if width == 165:
        np.testing.assert_array_equal(kwargs["jaw_pose"].numpy(), seq.poses[:, 156:159])
        np.testing.assert_array_equal(kwargs["leye_pose"].numpy(), seq.poses[:, 159:162])
        np.testing.assert_array_equal(kwargs["reye_pose"].numpy(), seq.poses[:, 162:165])
    else:
        for key in ("jaw_pose", "leye_pose", "reye_pose"):
            np.testing.assert_array_equal(kwargs[key].numpy(), np.zeros((2, 3), dtype=np.float32))


@pytest.mark.parametrize("layout,width", [(None, 66), (None, 111), (None, 156), (None, 165), ("smplx55", 165)])
def test_smplx_optional_defaults_match_the_evaluation_batch(layout, width):
    seq = sequence(width=width, frames=4, metadata={"pose_layout": layout} if layout else {})
    kwargs = hs._build_smpl_kwargs(seq, torch_device="cpu", frame_slice=slice(1, 4))
    assert kwargs["expression"].shape == (3, 10)
    np.testing.assert_array_equal(kwargs["expression"].numpy(), np.zeros((3, 10), dtype=np.float32))
    for key in ("jaw_pose", "leye_pose", "reye_pose"):
        assert kwargs[key].shape == (3, 3)
        if not layout and width < 159:
            np.testing.assert_array_equal(kwargs[key].numpy(), np.zeros((3, 3), dtype=np.float32))
    for key in ("left_hand_pose", "right_hand_pose"):
        assert kwargs[key].shape == (3, 45)
    if not layout and width < 111:
        np.testing.assert_array_equal(kwargs["left_hand_pose"].numpy(), np.zeros((3, 45), dtype=np.float32))
    if not layout and width < 156:
        np.testing.assert_array_equal(kwargs["right_hand_pose"].numpy(), np.zeros((3, 45), dtype=np.float32))


def test_native55_does_not_silently_accept_an_old156_array():
    with pytest.raises(ValueError, match="requires 165"):
        hs._build_smpl_kwargs(sequence(width=156, metadata={"pose_layout": "smplx55"}), torch_device="cpu")


def test_smpl_payload_remains_body_only_with_ten_betas():
    seq = dataclasses.replace(sequence(width=72), model_type="smpl")
    kwargs = hs._build_smpl_kwargs(seq, torch_device="cpu")
    np.testing.assert_array_equal(kwargs["body_pose"].numpy(), seq.poses[:, 3:72])
    assert kwargs["betas"].shape == (2, 10)
    assert "left_hand_pose" not in kwargs


def test_explicit_model_assets_and_beta_dimensions_do_not_share_cached_models(tmp_path, monkeypatch):
    first_path = tmp_path / "locked_head.npz"
    second_path = tmp_path / "standard.npz"
    first_path.touch()
    second_path.touch()
    calls = []

    class Model:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def to(self, device):
            return self

    def unexpected_create(**kwargs):
        raise AssertionError("An explicit model file must use SMPLX directly")

    monkeypatch.setitem(sys.modules, "smplx", SimpleNamespace(SMPLX=Model, create=unexpected_create))
    monkeypatch.setattr(hs, "_SMPL_MODEL_CACHE", {})
    first = sequence(metadata={"body_model_path": str(first_path), "pose_layout": "smplx55", "body_model_variant": "locked_head"})
    second = dataclasses.replace(first, metadata={**first.metadata, "body_model_path": str(second_path)})
    shorter = dataclasses.replace(first, betas=first.betas[:10])
    m1 = hs._create_smpl_model(first, torch.device("cpu"))
    assert hs._create_smpl_model(first, torch.device("cpu")) is m1
    assert hs._create_smpl_model(second, torch.device("cpu")) is not m1
    assert hs._create_smpl_model(shorter, torch.device("cpu")) is not m1
    assert len(calls) == 3
    assert [item["num_betas"] for item in calls] == [16, 16, 10]
    assert calls[0]["model_path"] == str(first_path.resolve())
    assert calls[1]["model_path"] == str(second_path.resolve())


def test_missing_explicit_model_fails_instead_of_using_a_different_template(tmp_path):
    seq = sequence(metadata={"body_model_path": str(tmp_path / "absent.npz")})
    with pytest.raises(FileNotFoundError, match="Explicit SMPL-X body model"):
        hs._smpl_model_cache_key(seq, torch.device("cpu"))


def test_default_smplx_loader_receives_the_actual_beta_dimension(tmp_path, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(to=lambda device: object())

    monkeypatch.setitem(sys.modules, "smplx", SimpleNamespace(create=create))
    monkeypatch.setattr(hs, "resolve_body_model_dir", lambda model_type: tmp_path)
    monkeypatch.setattr(hs, "_SMPL_MODEL_CACHE", {})
    hs._create_smpl_model(sequence(betas_dim=16), torch.device("cpu"))
    assert calls[0]["model_type"] == "smplx"
    assert calls[0]["num_betas"] == 16


@pytest.mark.parametrize("builder", ["shape", "pose_sequence", "pose_zero_betas"])
def test_neutral_geometry_preserves_explicit_locked_model_and_sixteen_betas(builder, tmp_path, monkeypatch):
    model_path = tmp_path / "locked_head_neutral.npz"
    model_path.touch()
    seq = sequence(metadata={"body_model_path": str(model_path), "body_model_variant": "locked_head", "pose_layout": "smplx55"})
    create_calls = []
    payloads = []

    class Model:
        lbs_weights = torch.ones((3, 55)) / 55

        def __init__(self, **kwargs):
            assert kwargs["model_path"] == str(model_path.resolve())
            assert kwargs["num_betas"] == 16
            create_calls.append(kwargs)

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            payloads.append(kwargs)
            return SimpleNamespace(vertices=torch.zeros((1, 3, 3)), joints=torch.zeros((1, 55, 3)))

    def unexpected_default(model_type):
        raise AssertionError("Neutral geometry must retain its explicit model asset")

    monkeypatch.setitem(sys.modules, "smplx", SimpleNamespace(SMPLX=Model))
    monkeypatch.setattr(hs, "resolve_body_model_dir", unexpected_default)
    monkeypatch.setattr(hs, "_SMPL_MODEL_CACHE", {})
    if builder == "shape":
        vertices, joints = hs.build_shape_neutral_geometry(seq, device="cpu")
    else:
        mode = "neutral" if builder == "pose_zero_betas" else "sequence"
        vertices, joints, weights, info = hs.build_pose_neutral_template_geometry(seq, device="cpu", betas_mode=mode)
        assert weights.shape == (3, 55)
        assert len(info["betas"]) == 16
    assert vertices.shape == (3, 3)
    assert joints.shape == (55, 3)
    assert len(create_calls) == 1
    expected_betas = np.zeros((1, 16), dtype=np.float32) if builder == "pose_zero_betas" else seq.betas[None]
    np.testing.assert_array_equal(payloads[0]["betas"].numpy(), expected_betas)
    assert set(payloads[0]) >= {"jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose"}
    assert seq.metadata["body_model_path"] == str(model_path)


@pytest.mark.parametrize("default_device", ["meta", "cuda"])
@pytest.mark.parametrize("model_kind", ["smpl", "smplx_default", "smplx_explicit"])
def test_model_construction_and_renderer_survive_non_cpu_default_device(default_device, model_kind, tmp_path, monkeypatch):
    if default_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA default-device regression needs CUDA")
    model_path = tmp_path / "locked_head.npz"
    model_path.touch()
    seq = sequence(metadata={"body_model_path": str(model_path), "pose_layout": "smplx55"})
    if model_kind == "smplx_default":
        seq = dataclasses.replace(seq, metadata={})
    elif model_kind == "smpl":
        seq = dataclasses.replace(seq, model_type="smpl", poses=seq.poses[:, :72], metadata={})
    constructed = []

    class Model:
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        def __init__(self, **kwargs):
            # This is the NumPy/Torch operation used by upstream mean-pose
            # creation. Meta/CUDA defaults fail here unless construction
            # explicitly chooses CPU, rather than transferring afterwards.
            mean = np.concatenate((np.ones(3), torch.zeros(3)))
            assert mean.shape == (6,)
            constructed.append(kwargs)

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **kwargs):
            batch = kwargs["transl"].shape[0]
            return SimpleNamespace(vertices=torch.zeros((batch, 3, 3), device=self.device),
                                   joints=torch.zeros((batch, 55, 3), device=self.device))

    monkeypatch.setitem(sys.modules, "smplx", SimpleNamespace(SMPL=Model, SMPLX=Model, create=Model))
    monkeypatch.setattr(hs, "resolve_body_model_dir", lambda model_type: tmp_path)
    monkeypatch.setattr(hs, "_SMPL_MODEL_CACHE", {})
    previous_device = torch.get_default_device()
    try:
        torch.set_default_device(default_device)
        vertices, joints = hs.evaluate_smpl_sequence(seq, device="cpu")
        renderer = display.GtSmplFrameRenderer(seq, color=(1, 2, 3, 4), device="cpu")
        viewer_vertices, viewer_joints = renderer._forward_local_body(seq.poses[0])
        assert len(constructed) == 1
        if model_kind.startswith("smplx"):
            assert constructed[0]["num_betas"] == 16
        assert vertices.shape == (2, 3, 3)
        assert joints.shape == (2, 55, 3)
        assert viewer_vertices.shape == (3, 3)
        assert viewer_joints.shape == (55, 3)
        assert torch.get_default_device().type == default_device
    finally:
        torch.set_default_device(previous_device)


@pytest.mark.parametrize("layout", [None, "smplx55"])
def test_viewer_and_batch_evaluation_send_the_same_pose_and_shape(layout, monkeypatch):
    seq = sequence(metadata={"pose_layout": layout} if layout else {})
    calls = []

    class Model:
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        def __call__(self, **kwargs):
            calls.append({key: value.clone() for key, value in kwargs.items()})
            batch = kwargs["transl"].shape[0]
            return SimpleNamespace(vertices=torch.zeros((batch, 3, 3)), joints=torch.zeros((batch, 55, 3)))

    model = Model()
    monkeypatch.setattr(hs, "_create_smpl_model", lambda seq, device: model)
    monkeypatch.setattr(display, "_create_smpl_model", lambda seq, device: model)
    hs.evaluate_smpl_sequence(seq, device="cpu")
    renderer = display.GtSmplFrameRenderer(seq, color=(1, 2, 3, 4), device="cpu")
    renderer._forward_local_body(seq.poses[1], transl_m=seq.trans[1])
    assert calls[0].keys() == calls[1].keys()
    for key in calls[0]:
        torch.testing.assert_close(calls[0][key][1:2], calls[1][key], rtol=0, atol=0)
