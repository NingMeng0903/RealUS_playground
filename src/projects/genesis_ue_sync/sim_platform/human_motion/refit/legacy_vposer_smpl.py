# Legacy SMPL VPoser (bodyprior_* weights), compatible with checkpoints such as TR00_E096.pt.
# Rotations use human_body_prior.tools.tgm_conversion (no torchgeometry dependency).

from __future__ import annotations

import glob
import os.path as osp
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from human_body_prior.tools.tgm_conversion import (
    angle_axis_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
)


class ContinousRotReprDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, module_input: torch.Tensor) -> torch.Tensor:
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


class LegacyVPoserSmpl(nn.Module):
    """Matches ref_code_library/vposer_v1_0/vposer_smpl.py layout for state_dict keys."""

    def __init__(
        self,
        num_neurons: int,
        latentD: int,
        data_shape: tuple[int, int, int],
        *,
        use_cont_repr: bool = True,
    ) -> None:
        super().__init__()
        self.latentD = latentD
        self.use_cont_repr = use_cont_repr
        n_features = int(data_shape[0] * data_shape[1] * data_shape[2])
        self.num_joints = int(data_shape[1])

        self.bodyprior_enc_bn1 = nn.BatchNorm1d(n_features)
        self.bodyprior_enc_fc1 = nn.Linear(n_features, num_neurons)
        self.bodyprior_enc_bn2 = nn.BatchNorm1d(num_neurons)
        self.bodyprior_enc_fc2 = nn.Linear(num_neurons, num_neurons)
        self.bodyprior_enc_mu = nn.Linear(num_neurons, latentD)
        self.bodyprior_enc_logvar = nn.Linear(num_neurons, latentD)
        self.dropout = nn.Dropout(p=0.1, inplace=False)

        self.bodyprior_dec_fc1 = nn.Linear(latentD, num_neurons)
        self.bodyprior_dec_fc2 = nn.Linear(num_neurons, num_neurons)
        if self.use_cont_repr:
            self.rot_decoder = ContinousRotReprDecoder()
        self.bodyprior_dec_out = nn.Linear(num_neurons, self.num_joints * 6)

    def encode(self, pin: torch.Tensor) -> torch.distributions.normal.Normal:
        xout = pin.view(pin.size(0), -1)
        xout = self.bodyprior_enc_bn1(xout)
        xout = F.leaky_relu(self.bodyprior_enc_fc1(xout), negative_slope=0.2)
        xout = self.bodyprior_enc_bn2(xout)
        xout = self.dropout(xout)
        xout = F.leaky_relu(self.bodyprior_enc_fc2(xout), negative_slope=0.2)
        return torch.distributions.normal.Normal(
            self.bodyprior_enc_mu(xout),
            F.softplus(self.bodyprior_enc_logvar(xout)),
        )

    def decode(self, zin: torch.Tensor, output_type: str = "matrot") -> torch.Tensor:
        assert output_type in ("matrot", "aa")
        xout = F.leaky_relu(self.bodyprior_dec_fc1(zin), negative_slope=0.2)
        xout = self.dropout(xout)
        xout = F.leaky_relu(self.bodyprior_dec_fc2(xout), negative_slope=0.2)
        xout = self.bodyprior_dec_out(xout)
        if self.use_cont_repr:
            xout = self.rot_decoder(xout)
        else:
            xout = torch.tanh(xout)
        xout = xout.view([-1, 1, self.num_joints, 9])
        if output_type == "aa":
            return self.matrot2aa(xout)
        return xout

    def forward(
        self,
        pin: torch.Tensor,
        input_type: str = "matrot",
        output_type: str = "matrot",
    ) -> dict[str, Any]:
        del input_type
        assert output_type in ("matrot", "aa")
        q_z = self.encode(pin)
        q_z_sample = q_z.rsample()
        prec = self.decode(q_z_sample)
        if output_type == "aa":
            prec = self.matrot2aa(prec)
        return {"pose": prec, "mean": q_z.mean, "std": q_z.scale}

    @staticmethod
    def matrot2aa(pose_matrot: torch.Tensor) -> torch.Tensor:
        batch_size = pose_matrot.size(0)
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0, 1])
        pose = rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose: torch.Tensor) -> torch.Tensor:
        batch_size = pose.size(0)
        pose_body_matrot = (
            angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3]
            .contiguous()
            .view(batch_size, 1, -1, 9)
        )
        return pose_body_matrot


def _read_model_hparams(expr_dir: Path) -> tuple[int, int, tuple[int, int, int], bool]:
    latent_d = 32
    num_neurons = 512
    data_shape = (1, 21, 3)
    use_cont_repr = True
    yaml_files = sorted(glob.glob(str(expr_dir / "*.yaml")))
    if yaml_files:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(yaml_files[0])
        mp = cfg.get("model_params") or {}
        latent_d = int(mp.get("latentD", latent_d))
        num_neurons = int(mp.get("num_neurons", num_neurons))
    ini_files = sorted(glob.glob(str(expr_dir / "*.ini")))
    if ini_files:
        import configparser

        cp = configparser.ConfigParser()
        cp.read(ini_files[0])
        sec = cp["All"] if "All" in cp else cp[cp.sections()[0]]
        if sec.get("latentD"):
            latent_d = int(sec.get("latentD"))
        if sec.get("num_neurons"):
            num_neurons = int(sec.get("num_neurons"))
        if sec.get("use_cont_repr"):
            use_cont_repr = sec.get("use_cont_repr").strip().lower() in ("1", "true", "yes")
        ds = sec.get("data_shape")
        if ds:
            import ast

            parsed = ast.literal_eval(ds.strip())
            if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
                data_shape = (int(parsed[0]), int(parsed[1]), int(parsed[2]))
    return num_neurons, latent_d, data_shape, use_cont_repr


def load_legacy_vposer_smpl_checkpoint(expr_dir: str | Path, *, map_location: str | torch.device) -> LegacyVPoserSmpl:
    expr_dir = Path(expr_dir).expanduser().resolve()
    snap_dir = expr_dir / "snapshots"
    candidates = sorted(
        glob.glob(str(snap_dir / "*.ckpt")) + glob.glob(str(snap_dir / "*.pt")),
        key=osp.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No .pt/.ckpt under {snap_dir}")
    ckpt_path = candidates[-1]
    raw: Any = torch.load(ckpt_path, map_location=map_location)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Unexpected checkpoint format: {ckpt_path}")
    first_key = next(iter(state_dict.keys()))
    if not str(first_key).startswith("bodyprior_"):
        raise ValueError(
            f"Checkpoint does not look like legacy bodyprior weights (first key: {first_key!r})"
        )
    num_neurons, latent_d, data_shape, use_cont_repr = _read_model_hparams(expr_dir)
    model = LegacyVPoserSmpl(
        num_neurons,
        latent_d,
        data_shape,
        use_cont_repr=use_cont_repr,
    )
    model.load_state_dict(state_dict, strict=True)
    return model
