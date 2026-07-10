"""Offline fixed-shape calibration from multiview Body25 observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.fitting.body25_smpl24 import BODY25_SMPL24_PAIRS
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import human_sequence_from_smpl_pkl


BODY25_CORE_JOINTS: tuple[int, ...] = (1, 2, 5, 8, 9, 12)

BODY25_SHAPE_LIMBS: tuple[tuple[int, int], ...] = (
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
)

_BODY25_TO_SMPL24 = {int(body25): int(smpl24) for body25, smpl24 in BODY25_SMPL24_PAIRS}
SMPL24_SHAPE_LIMBS: tuple[tuple[int, int], ...] = tuple(
    (_BODY25_TO_SMPL24[a], _BODY25_TO_SMPL24[b])
    for a, b in BODY25_SHAPE_LIMBS
    if a in _BODY25_TO_SMPL24 and b in _BODY25_TO_SMPL24
)


@dataclass(frozen=True)
class ShapeFrameQualityConfig:
    min_valid_body25: int = 12
    min_core_body25: int = 4
    min_valid_shape_limbs: int = 6
    confidence_threshold: float = 0.3

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ShapeFrameQualityConfig":
        payload = dict(payload or {})
        return cls(
            min_valid_body25=max(1, int(payload.get("min_valid_body25", cls.min_valid_body25))),
            min_core_body25=max(0, int(payload.get("min_core_body25", cls.min_core_body25))),
            min_valid_shape_limbs=max(1, int(payload.get("min_valid_shape_limbs", cls.min_valid_shape_limbs))),
            confidence_threshold=float(payload.get("confidence_threshold", cls.confidence_threshold)),
        )


@dataclass(frozen=True)
class SmplShapeCalibrationConfig:
    smpl_model_dir: str = "dataset/intermediate/humans/body_models/smpl"
    device: str = "cuda"
    max_iter: int = 30
    lr: float = 0.08
    reg_shapes_weight: float = 5.0e-3
    init_betas: tuple[float, ...] = (0.0,) * 10

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SmplShapeCalibrationConfig":
        payload = dict(payload or {})
        raw = list(payload.get("init_betas") or [0.0] * 10)[:10]
        betas = tuple(float(v) for v in raw)
        if len(betas) < 10:
            betas = betas + (0.0,) * (10 - len(betas))
        return cls(
            smpl_model_dir=str(payload.get("smpl_model_dir", cls.smpl_model_dir)),
            device=str(payload.get("device", cls.device)),
            max_iter=max(1, int(payload.get("max_iter", cls.max_iter))),
            lr=float(payload.get("lr", cls.lr)),
            reg_shapes_weight=float(payload.get("reg_shapes_weight", cls.reg_shapes_weight)),
            init_betas=betas,
        )


def evaluate_shape_frame_quality(
    keypoints3d_body25: np.ndarray,
    config: ShapeFrameQualityConfig | None = None,
) -> tuple[bool, dict[str, Any]]:
    cfg = config or ShapeFrameQualityConfig()
    kp = np.asarray(keypoints3d_body25, dtype=np.float32).reshape(-1, 4)
    conf = kp[:, 3] if kp.shape[0] else np.zeros(0, dtype=np.float32)
    valid = conf >= float(cfg.confidence_threshold)
    valid_body25 = int(np.sum(valid))
    valid_core = int(sum(1 for idx in BODY25_CORE_JOINTS if idx < kp.shape[0] and bool(valid[idx])))
    valid_limbs = 0
    for a, b in BODY25_SHAPE_LIMBS:
        if a < kp.shape[0] and b < kp.shape[0] and bool(valid[a]) and bool(valid[b]):
            valid_limbs += 1
    reasons: list[str] = []
    if valid_body25 < int(cfg.min_valid_body25):
        reasons.append("min_valid_body25")
    if valid_core < int(cfg.min_core_body25):
        reasons.append("min_core_body25")
    if valid_limbs < int(cfg.min_valid_shape_limbs):
        reasons.append("min_valid_shape_limbs")
    diagnostics = {
        "valid": not reasons,
        "valid_body25": valid_body25,
        "valid_core_body25": valid_core,
        "valid_shape_limbs": valid_limbs,
        "reasons": reasons,
    }
    return not reasons, diagnostics


def _observed_limb_tensors(
    keypoints3d_frames: np.ndarray,
    *,
    confidence_threshold: float,
    device: Any,
) -> tuple[Any, Any, Any]:
    import torch

    frames = np.asarray(keypoints3d_frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1] < 25 or frames.shape[2] < 4:
        raise ValueError(f"Expected keypoints3d frames with shape (F, >=25, 4), got {frames.shape}.")
    lengths: list[float] = []
    weights: list[float] = []
    limb_indices: list[int] = []
    for frame in frames:
        for limb_i, (a, b) in enumerate(BODY25_SHAPE_LIMBS):
            ca = float(frame[a, 3])
            cb = float(frame[b, 3])
            if ca < confidence_threshold or cb < confidence_threshold:
                continue
            length = float(np.linalg.norm(frame[b, :3] - frame[a, :3]))
            if not np.isfinite(length) or length <= 1.0e-4:
                continue
            lengths.append(length)
            weights.append(min(ca, cb))
            limb_indices.append(limb_i)
    if not lengths:
        raise ValueError("No valid Body25 shape limbs after filtering.")
    return (
        torch.tensor(limb_indices, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.float32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
    )


def optimize_smpl_betas_from_body25(
    keypoints3d_frames: np.ndarray,
    config: SmplShapeCalibrationConfig | None = None,
    *,
    confidence_threshold: float = 0.3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit fixed SMPL betas from multiframe Body25 limb lengths."""
    cfg = config or SmplShapeCalibrationConfig()
    import torch

    from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import _create_smpl_model, resolve_torch_device

    device = resolve_torch_device(cfg.device)
    sequence = human_sequence_from_smpl_pkl(Path(str(cfg.smpl_model_dir)), betas=np.zeros(10, dtype=np.float32))
    model = _create_smpl_model(sequence, device)
    model.eval()

    limb_index, obs_lengths, obs_weights = _observed_limb_tensors(
        keypoints3d_frames,
        confidence_threshold=float(confidence_threshold),
        device=device,
    )
    smpl_limb_pairs = torch.tensor(SMPL24_SHAPE_LIMBS, dtype=torch.long, device=device)
    init = np.asarray(cfg.init_betas, dtype=np.float32).reshape(10)
    betas = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32, device=device))
    zeros_pose = torch.zeros((1, 72), dtype=torch.float32, device=device)
    zeros_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
    optimizer = torch.optim.LBFGS(
        [betas],
        lr=float(cfg.lr),
        max_iter=int(cfg.max_iter),
        line_search_fn="strong_wolfe",
    )
    loss_history: list[float] = []

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        out = model(
            betas=betas[None, :],
            global_orient=zeros_pose[:, :3],
            body_pose=zeros_pose[:, 3:72],
            transl=zeros_trans,
        )
        joints = out.joints[0, :24, :]
        src = joints[smpl_limb_pairs[:, 0]]
        dst = joints[smpl_limb_pairs[:, 1]]
        pred_all = torch.linalg.norm(dst - src, dim=1)
        pred = pred_all[limb_index]
        err = pred - obs_lengths
        loss_shape = torch.sum(obs_weights * err.square()) / torch.clamp(torch.sum(obs_weights), min=1.0)
        loss_reg = torch.sum(betas.square())
        loss = loss_shape + float(cfg.reg_shapes_weight) * loss_reg
        loss.backward()
        loss_history.append(float(loss.detach().cpu().item()))
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final_betas = betas.detach().cpu().numpy().astype(np.float32)
    diagnostics = {
        "num_frames": int(np.asarray(keypoints3d_frames).shape[0]),
        "num_limb_observations": int(obs_lengths.numel()),
        "loss_initial": float(loss_history[0]) if loss_history else None,
        "loss_final": float(loss_history[-1]) if loss_history else None,
        "beta_norm": float(np.linalg.norm(final_betas)),
        "optimizer": "torch_lbfgs",
        "max_iter": int(cfg.max_iter),
        "reg_shapes_weight": float(cfg.reg_shapes_weight),
        "shape_limb_count": int(len(BODY25_SHAPE_LIMBS)),
    }
    return final_betas, diagnostics


def write_shape_calibration_outputs(
    output_dir: Path,
    *,
    betas: np.ndarray,
    diagnostics: dict[str, Any],
    keypoints3d_frames: np.ndarray | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    betas_path = output_dir / "betas.npy"
    diag_path = output_dir / "diagnostics.json"
    np.save(betas_path, np.asarray(betas, dtype=np.float32).reshape(10))
    if keypoints3d_frames is not None:
        np.save(output_dir / "keypoints3d_body25.npy", np.asarray(keypoints3d_frames, dtype=np.float32))
    diag_path.write_text(json.dumps(diagnostics, ensure_ascii=True, indent=2), encoding="utf-8")
    return betas_path, diag_path
