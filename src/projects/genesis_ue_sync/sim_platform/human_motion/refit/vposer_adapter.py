from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on the active runtime env
    torch = None  # type: ignore[assignment]

from projects.genesis_ue_sync.sim_platform.human_motion.dependencies import human_motion_dependencies


def _as_body_pose63(poses: torch.Tensor) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for VPoser.")
    pose = poses.float()
    if pose.shape[-1] == 72:
        return pose[..., 3:66]
    if pose.shape[-1] == 69:
        return pose[..., :63]
    if pose.shape[-1] >= 63:
        return pose[..., :63]
    raise ValueError(f"VPoser expects at least 63 body-pose values, got {pose.shape[-1]}.")


def _latent_mean(encoded: Any) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for VPoser.")
    for attr in ("mean", "loc"):
        value = getattr(encoded, attr, None)
        if value is not None:
            return value
    if isinstance(encoded, (tuple, list)) and encoded:
        first = encoded[0]
        for attr in ("mean", "loc"):
            value = getattr(first, attr, None)
            if value is not None:
                return value
        if torch.is_tensor(first):
            return first
    if torch.is_tensor(encoded):
        return encoded
    raise TypeError(f"Cannot read VPoser latent mean from {type(encoded)!r}.")


def _torch_load_checkpoint(path: Path, *, map_location: Any) -> Any:
    if torch is None:
        raise ImportError("torch is required for VPoser.")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_vposer_v2_checkpoint(model_dir: str, *, map_location: Any) -> Any:
    if torch is None:
        raise ImportError("torch is required for VPoser.")
    model_path = Path(model_dir)
    checkpoints = sorted(
        list((model_path / "snapshots").glob("*.ckpt")) + list((model_path / "snapshots").glob("*.pt")),
        key=lambda p: p.stat().st_mtime,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No VPoser checkpoint found under {model_path / 'snapshots'}")
    cfg_path = next(iter(sorted(model_path.glob("*.yaml"))), None)
    if cfg_path is None:
        raise FileNotFoundError(f"No VPoser yaml config found under {model_path}")

    from human_body_prior.models.vposer_model import VPoser
    from omegaconf import OmegaConf

    model = VPoser(OmegaConf.load(cfg_path))
    checkpoint = _torch_load_checkpoint(checkpoints[-1], map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected VPoser checkpoint payload: {type(state_dict)!r}")
    clean_state = {
        key.removeprefix("vp_model."): value
        for key, value in state_dict.items()
        if key.removeprefix("vp_model.") in model.state_dict()
    }
    if not clean_state:
        first_key = next(iter(state_dict), "")
        raise ValueError(f"Checkpoint does not contain VPoser weights (first key: {first_key!r})")
    model.load_state_dict(clean_state, strict=False)
    return model


@dataclass
class VPoserAdapter:
    model_dir: Path
    human_body_prior_root: Path
    device: str = "cpu"
    enabled: bool = True
    load_backend: str = ""

    def __post_init__(self) -> None:
        self.model = None
        self.warning = ""
        self.load_backend = ""
        if not bool(self.enabled):
            self.warning = "VPoser disabled by configuration."
            return
        try:
            if torch is None:
                raise ImportError("torch is required for VPoser.")
            hbp = self.human_body_prior_root.expanduser().resolve()
            if hbp.is_dir() and str(hbp) not in sys.path:
                sys.path.insert(0, str(hbp))
            if not self.model_dir.expanduser().exists():
                raise FileNotFoundError(f"VPoser model dir not found: {self.model_dir}")
            model_dir = str(self.model_dir.expanduser().resolve())
            dev = torch.device(self.device)
            comp_device = "cpu" if str(self.device).lower() == "cpu" else "gpu"
            model = None
            errors: list[str] = []
            try:
                from human_body_prior.tools.model_loader import load_vposer

                model, _ = load_vposer(model_dir, vp_model="snapshot")
                self.load_backend = "load_vposer"
            except Exception as exc:
                errors.append(f"load_vposer: {exc}")
            if model is None:
                try:
                    from human_body_prior.models.vposer_model import VPoser
                    from human_body_prior.tools.model_loader import load_model

                    model, _ = load_model(
                        model_dir,
                        model_code=VPoser,
                        remove_words_in_model_weights="vp_model.",
                        comp_device=comp_device,
                    )
                    self.load_backend = "human_body_prior_sequential"
                except Exception as exc:
                    errors.append(f"load_model: {exc}")
            if model is None:
                try:
                    model = _load_vposer_v2_checkpoint(model_dir, map_location=dev)
                    self.load_backend = "direct_vposer_v2"
                except Exception as exc:
                    errors.append(f"direct_vposer_v2: {exc}")
            if model is None:
                from projects.genesis_ue_sync.sim_platform.human_motion.refit.legacy_vposer_smpl import (
                    load_legacy_vposer_smpl_checkpoint,
                )

                model = load_legacy_vposer_smpl_checkpoint(model_dir, map_location=dev)
                self.load_backend = "legacy_vposer_v1_bodyprior"
            model = model.to(dev)
            model.eval()
            self.model = model
        except Exception as exc:
            self.model = None
            self.load_backend = ""
            details = f" ({'; '.join(errors)})" if "errors" in locals() and errors else ""
            self.warning = f"VPoser unavailable: {exc}{details}"

    @classmethod
    def from_dependencies(cls, *, device: str = "cpu", enabled: bool = True) -> "VPoserAdapter":
        deps = {dep.name: dep for dep in human_motion_dependencies()}
        return cls(
            model_dir=deps["VPoser-weights"].resolved_path(),
            human_body_prior_root=deps["human_body_prior"].resolved_path(),
            device=device,
            enabled=enabled,
        )

    @property
    def available(self) -> bool:
        return self.model is not None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "available": bool(self.available),
            "model_dir": str(self.model_dir),
            "human_body_prior_root": str(self.human_body_prior_root),
            "device": str(self.device),
            "load_backend": self.load_backend or "",
            "warning": self.warning,
        }

    def encode(self, poses: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError(self.warning or "VPoser is not available.")
        body_pose = _as_body_pose63(poses).reshape(-1, 63).to(torch.device(self.device))
        encoded = self.model.encode(body_pose)
        return _latent_mean(encoded)

    def prior_loss(self, poses: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return poses.new_zeros(())
        latent = self.encode(poses)
        return torch.mean(latent * latent)

    def latent_norm(self, poses: torch.Tensor) -> float:
        if self.model is None:
            return 0.0
        with torch.no_grad():
            latent = self.encode(poses)
            return float(torch.linalg.vector_norm(latent, dim=-1).mean().detach().cpu())
