"""GPU-only rerender of moving-ellipsoid region-field videos from certified data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_EXPERIMENTS = Path(__file__).resolve().parent
_ROOT = _EXPERIMENTS.parent
for _path in (_ROOT, _EXPERIMENTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ellipse_vessel_ird_demo import DemoConfig  # noqa: E402
from moving_obstacle_u_band_demo import render_region_field_videos  # noqa: E402
from ird_playground.ird.robot_model import load_robot_model_spec  # noqa: E402
from ird_playground.neural.signed_field import ReachabilitySDF  # noqa: E402
from ird_playground.region.task_cone import TaskConeConfig, TaskConeReachability  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/rm4d_signed/selected.pt"))
    parser.add_argument("--robot-spec", type=Path, default=Path("configs/robot_probe45.yaml"))
    parser.add_argument("--source", type=Path, default=Path("data/reports/ellipse_vessel_ird_demo_projection"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports/moving_obstacle_u_band"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU-only region-field rerender requires CUDA")

    source = np.load(resolve(args.source) / "qpik_guidance.npz")
    moving = np.load(resolve(args.out_dir) / "moving_obstacle_guidance.npz")
    spec = load_robot_model_spec(resolve(args.robot_spec))
    field = ReachabilitySDF.load(
        resolve(args.checkpoint), device=str(device), expected_robot=spec, allow_stale=True
    ).model.eval()
    for parameter in field.parameters():
        parameter.requires_grad_(False)
    cfg = DemoConfig()
    axis0 = torch.as_tensor(spec.root_to_j1_axis().astype(np.float32), device=device)
    task_cone = TaskConeReachability(TaskConeConfig(
        tip_half_angle_deg=cfg.tip_half_angle_deg,
        roll_half_range_deg=cfg.roll_half_range_deg,
        samples=64,
        seed=17,
    )).to(device)
    render_region_field_videos(
        resolve(args.out_dir), field, task_cone, cfg,
        np.asarray(source["s"]), np.asarray(source["path_y_m"]),
        np.asarray(source["initial_surface_theta_rad"]),
        np.asarray(source["initial_rail_y"]),
        np.asarray(moving["surface_theta_rad"]), np.asarray(moving["rail_ref_m"]),
        np.asarray(moving["task_rotvec_local"]), np.asarray(moving["ird_clearance"]),
        np.asarray(moving["conditioned_clearance"]),
        np.asarray(moving["obstacle_center_world"]),
        np.asarray(moving["obstacle_rotation_world"]),
        np.asarray(moving["obstacle_semiaxes_m"]),
        float(np.asarray(moving["obstacle_safe_margin_m"])),
        float(np.asarray(moving["obstacle_planner_margin_m"])), axis0, fps=args.fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
