"""Bed support-plane SDF penalty for SMPL-X fitting (world frame, meters)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec


@dataclass(frozen=True)
class BedSdfConfig:
    margin_m: float = 0.008
    weight: float = 80.0
    max_iter: int = 12
    lr: float = 0.25
    optimize_keys: tuple[str, ...] = ("Rh", "Th")


def bed_top_z_from_scene_spec(scene_spec_path: str | None) -> float:
    spec = load_sync_scene_spec(scene_spec_path)
    if spec.support_surface is None:
        raise RuntimeError(f"Scene has no support_surface: {scene_spec_path}")
    return float(spec.support_surface_top_z)


def bed_penetration_loss(
    vertices_world: np.ndarray,
    *,
    bed_top_z: float,
    margin_m: float,
) -> tuple[float, int]:
    """Scalar penetration loss and count of vertices below bed_top - margin."""
    verts = np.asarray(vertices_world, dtype=np.float64).reshape(-1, 3)
    z = verts[:, 2]
    penetration = float(bed_top_z) + float(margin_m) - z
    active = penetration > 0.0
    if not np.any(active):
        return 0.0, 0
    pen = penetration[active]
    return float(np.mean(pen * pen)), int(np.sum(active))


def refine_params_with_bed_sdf(
    body_model: Any,
    params: dict[str, Any],
    *,
    bed_top_z: float,
    cfg: BedSdfConfig | None = None,
    optimize_keys: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """LBFGS refinement: penalize vertices penetrating below bed top plane."""
    import torch
    from easymocap.pyfitting.lbfgs import LBFGS
    from easymocap.pyfitting.optimize import FittingMonitor, grad_require

    cfg = cfg or BedSdfConfig()
    device = body_model.device
    work = {k: np.asarray(params[k], dtype=np.float32).copy() for k in params}
    torch_params = {k: torch.tensor(work[k], device=device, dtype=torch.float32) for k in work}
    opt_keys_list = list(optimize_keys if optimize_keys is not None else cfg.optimize_keys)
    opt_keys = [k for k in opt_keys_list if k in torch_params and k != "shapes"]
    opt_list = [torch_params[k] for k in opt_keys]
    grad_require(opt_list, True)
    optimizer = LBFGS(opt_list, line_search_fn="strong_wolfe", max_iter=int(cfg.max_iter))

    bed_z = float(bed_top_z)
    margin = float(cfg.margin_m)
    weight = float(cfg.weight)

    def _verts() -> torch.Tensor:
        kw: dict[str, Any] = {
            "Rh": torch_params["Rh"],
            "Th": torch_params["Th"],
            "poses": torch_params["poses"],
            "shapes": torch_params["shapes"],
            "return_verts": True,
            "return_tensor": True,
        }
        if "expression" in torch_params:
            kw["expression"] = torch_params["expression"]
        v = body_model(**kw)
        if isinstance(v, (list, tuple)):
            v = v[0]
        return v.reshape(-1, 3)

    def closure(debug: bool = False):
        optimizer.zero_grad()
        verts = _verts()
        pen = margin + bed_z - verts[:, 2]
        active = pen > 0.0
        if bool(torch.any(active)):
            loss_bed = torch.mean(pen[active] ** 2)
        else:
            loss_bed = torch.sum(verts[:, 2] * 0.0)
        loss = weight * loss_bed
        if debug:
            return {
                "bed_sdf": float(loss_bed.detach().cpu().item()),
                "total": float(loss.detach().cpu().item()),
                "penetrating_verts": int(torch.sum(active).detach().cpu().item()),
            }
        loss.backward()
        return loss

    monitor = FittingMonitor(ftol=1e-5)
    monitor.run_fitting(optimizer, closure, opt_list)
    monitor.close()
    grad_require(opt_list, False)
    diag = closure(debug=True)
    out = {k: torch_params[k].detach().cpu().numpy() for k in torch_params}
    for k, v in out.items():
        work[k] = np.asarray(v, dtype=np.float32)
    return work, {str(k): float(v) for k, v in diag.items()}
