"""Independent GT, neural prediction, and gradient plots for signed IRD."""

from __future__ import annotations

from pathlib import Path

import json
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from ird_playground.ird.gpu_pose_gt import GpuPoseGtConfig, _probe_collision_filter
from ird_playground.ird.gt_common import reachability_modules
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.ird.torch_kinematics import TorchRM75Kinematics, select_collision_free_ik
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.probe.transform import default_ultrasound_probe


def horizontal_probe_rotation() -> np.ndarray:
    """Fixed base-to-TCP rotation with the configured probe's +Z axis horizontal."""
    return default_ultrasound_probe().rotation_matrix().astype(np.float32)


def _tcp_pose_matrices(
    base_in_tcp: "torch.Tensor",
    R_base_tcp: "torch.Tensor",
) -> "torch.Tensor":
    """Build ``T_tcp_in_rail_base`` from base-origin samples expressed in TCP."""
    R = R_base_tcp.expand(len(base_in_tcp), 3, 3)
    p = -(R @ base_in_tcp[..., None]).squeeze(-1)
    bottom = base_in_tcp.new_zeros(len(base_in_tcp), 1, 4)
    bottom[:, 0, 3] = 1.0
    return torch.cat((torch.cat((R, p[..., None]), dim=-1), bottom), dim=-2)


def neural_clearance(
    field: ReachabilitySDF,
    base_in_tcp: np.ndarray,
    R_base_tcp: np.ndarray,
    *,
    T_axis_world: np.ndarray | None = None,
    gradient: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Score flange-chart signed IRD on fixed-orientation TCP samples."""
    b = torch.as_tensor(base_in_tcp, dtype=torch.float32, device=field.device)
    b.requires_grad_(gradient)
    R = torch.as_tensor(R_base_tcp, dtype=torch.float32, device=field.device)
    T_tcp = _tcp_pose_matrices(b, R)
    if T_axis_world is None:
        axis = torch.eye(4, dtype=torch.float32, device=field.device)
    else:
        axis = torch.as_tensor(T_axis_world, dtype=torch.float32, device=field.device)
    clearance = field.score_world(T_tcp, axis)
    grad = None
    if gradient:
        grad = torch.autograd.grad(clearance.sum(), b)[0].detach().cpu().numpy()
    return clearance.detach().cpu().numpy(), grad


def solve_ird_grid_gt(
    base_in_tcp: np.ndarray,
    R_base_tcp: np.ndarray,
    *,
    seed_npz: str | Path,
    n_ik_seeds: int = 16,
    batch_size: int = 512,
    seed: int = 71,
    robot_spec_path: str | Path | None = None,
) -> np.ndarray:
    """Label fixed-orientation IRD base positions with GPU IK and collision."""
    if torch is None:
        raise ImportError("torch required")
    arrays = np.load(seed_npz, allow_pickle=False)
    q_pool_np = arrays["q_best"][arrays["reachable"] > 0.5]
    q_pool_np = q_pool_np[np.any(q_pool_np != 0.0, axis=1)]
    rng = np.random.default_rng(seed)
    *_, SelfCollisionFilter, build_locked_rail_model = reachability_modules()
    if robot_spec_path is not None:
        spec = load_robot_model_spec(robot_spec_path)
        locked = build_locked_rail_model(
            spec.kinematics_urdf,
            rail_locked_at_m=spec.rail_locked_at_m,
            tcp_frame=spec.tcp_frame,
        )
    else:
        locked = build_locked_rail_model()
    collision_filter, _, _ = _probe_collision_filter(
        GpuPoseGtConfig(), locked, SelfCollisionFilter
    )
    kin = TorchRM75Kinematics.from_locked_model(locked, device="cuda")
    q_pool = torch.as_tensor(q_pool_np, dtype=torch.float32, device=kin.device)
    b = np.asarray(base_in_tcp, dtype=np.float32)
    R_np = np.asarray(R_base_tcp, dtype=np.float32)
    # inv([R^T,b]) gives T_base_tcp=[R,-R b].
    p_np = -(R_np @ b[..., None]).squeeze(-1)
    labels = []
    for start in range(0, len(b), batch_size):
        stop = min(len(b), start + batch_size)
        n = stop - start
        p = torch.as_tensor(p_np[start:stop], device=kin.device)
        R = torch.as_tensor(R_np, device=kin.device).expand(n, 3, 3)
        seed_idx = rng.integers(0, len(q_pool_np), size=(n, n_ik_seeds))
        q0 = q_pool[torch.as_tensor(seed_idx, device=kin.device)]
        result = kin.ik_dls(
            p,
            R,
            q0,
            max_iter=100,
            tol_pos_m=2.0e-4,
            tol_rot_rad=1.0e-3,
        )
        checked = select_collision_free_ik(
            result,
            collision_filter,
            tol_pos_m=2.0e-4,
            tol_rot_rad=1.0e-3,
        )
        labels.append(checked.reachable.cpu().numpy())
        print(f"[viz-gt] {stop}/{len(b)} reachable={sum(int(x.sum()) for x in labels)}", flush=True)
    return np.concatenate(labels).astype(bool)


def _style_3d(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("base x in TCP (m)")
    ax.set_ylabel("base y in TCP (m)")
    ax.set_zlabel("base z in TCP (m)")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-55)


def render_volume_plots(
    base_in_tcp: np.ndarray,
    gt: np.ndarray,
    clearance: np.ndarray,
    *,
    out_dir: str | Path,
    decision_threshold: float = 0.0,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred = clearance >= float(decision_threshold)
    probability = 1.0 / (1.0 + np.exp(-3.0 * np.clip(clearance - float(decision_threshold), -20.0, 20.0)))
    files = {}
    for name, mask, color, title in (
        ("gt", gt, "#1976d2", "GT IRD"),
        ("neural", pred, "#2e7d32", "Neural IRD"),
    ):
        fig = plt.figure(figsize=(9, 8), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        pts = base_in_tcp[mask]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, c=color, alpha=0.45, linewidths=0)
        _style_3d(ax, title)
        path = out / f"horizontal_probe_ird_{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        files[name] = path

    fig = plt.figure(figsize=(16, 7), dpi=160)
    ax1 = fig.add_subplot(121, projection="3d")
    correct = gt == pred
    ax1.scatter(*base_in_tcp[correct].T, s=3, c="#9e9e9e", alpha=0.12, linewidths=0)
    fp = (~gt) & pred
    fn = gt & (~pred)
    if fp.any():
        ax1.scatter(*base_in_tcp[fp].T, s=12, c="#d32f2f", label="FP", linewidths=0)
    if fn.any():
        ax1.scatter(*base_in_tcp[fn].T, s=12, c="#7b1fa2", label="FN", linewidths=0)
    _style_3d(ax1, "Neural vs GT")
    ax1.legend(loc="upper right")
    ax2 = fig.add_subplot(122, projection="3d")
    shown = probability >= 0.05
    scatter = ax2.scatter(*base_in_tcp[shown].T, c=probability[shown], cmap="turbo", vmin=0, vmax=1, s=5, alpha=0.5, linewidths=0)
    _style_3d(ax2, "Neural probability")
    fig.colorbar(scatter, ax=ax2, shrink=0.65, label="probability")
    comparison = out / "horizontal_probe_ird_comparison.png"
    fig.savefig(comparison, bbox_inches="tight")
    plt.close(fig)
    files["comparison"] = comparison
    return files


def render_gradient_slice(
    field: ReachabilitySDF,
    R_base_tcp: np.ndarray,
    *,
    z_value: float,
    xy_limit: float,
    resolution: int,
    gt: np.ndarray,
    out_path: str | Path,
    T_axis_world: np.ndarray | None = None,
    decision_threshold: float = 0.0,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    axis = np.linspace(-xy_limit, xy_limit, resolution, dtype=np.float32)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    b = np.stack((X.reshape(-1), Y.reshape(-1), np.full(X.size, z_value, dtype=np.float32)), axis=1)
    clearance, grad = neural_clearance(
        field, b, R_base_tcp, T_axis_world=T_axis_world, gradient=True
    )
    C = clearance.reshape(resolution, resolution)
    G = grad.reshape(resolution, resolution, 3)
    norm = np.linalg.norm(G[..., :2], axis=-1, keepdims=True)
    unit = G[..., :2] / np.maximum(norm, 1.0e-8)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=170)
    im = ax.contourf(X, Y, C, levels=40, cmap="RdYlBu", alpha=0.9)
    thr = float(decision_threshold)
    ax.contour(X, Y, C, levels=[thr], colors="black", linewidths=2.0)
    ax.contour(X, Y, gt.reshape(resolution, resolution).astype(float), levels=[0.5], colors="#00e676", linewidths=1.8, linestyles="--")
    stride = max(1, resolution // 20)
    ax.quiver(
        X[::stride, ::stride], Y[::stride, ::stride],
        unit[::stride, ::stride, 0], unit[::stride, ::stride, 1],
        color="black", alpha=0.75, pivot="mid", scale=28,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("base x in TCP (m)")
    ax.set_ylabel("base y in TCP (m)")
    ax.set_title("Neural IRD")
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=2.0, label="Neural"),
            Line2D(
                [0], [0], color="#00e676", linewidth=1.8,
                linestyle="--", label="GT (IK + collision)",
            ),
        ],
        loc="upper right",
    )
    fig.colorbar(im, ax=ax, label="clearance")
    out_path = Path(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_viz_report(path: str | Path, metrics: dict) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


__all__ = [
    "horizontal_probe_rotation",
    "neural_clearance",
    "render_gradient_slice",
    "render_volume_plots",
    "solve_ird_grid_gt",
    "write_viz_report",
]
