"""Render independent horizontal-probe GT/neural IRD and gradient figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ird_playground.calib import load_conformal_json
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.viz.signed_ird import (
    horizontal_probe_rotation,
    neural_clearance,
    render_gradient_slice,
    render_volume_plots,
    solve_ird_grid_gt,
    write_viz_report,
)


def _balanced(gt: np.ndarray, pred: np.ndarray) -> float:
    return 0.5 * (float(pred[gt].mean()) + float((~pred[~gt]).mean()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("data/checkpoints/rm4d_signed/selected.pt"))
    ap.add_argument("--seed-gt", type=Path, default=Path("data/ird/gpu_pose_production.npz"))
    ap.add_argument("--robot-spec", type=Path, default=Path("configs/robot_probe45.yaml"))
    ap.add_argument("--conformal", type=Path, default=Path("data/calib/conformal_rm4d_signed.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/reports/ird_visualization"))
    ap.add_argument("--volume-resolution", type=int, default=31)
    ap.add_argument("--slice-resolution", type=int, default=61)
    ap.add_argument("--limit-m", type=float, default=1.1)
    ap.add_argument("--ik-seeds", type=int, default=16)
    ap.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Neural reachable iff clearance >= thr. Default: conformal m_safe if present else 0.",
    )
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    robot_spec = load_robot_model_spec(resolve(args.robot_spec))
    field = ReachabilitySDF.load(resolve(args.checkpoint), expected_robot=robot_spec)
    T_axis = robot_spec.root_to_j1_axis().astype(np.float32)
    thr = args.decision_threshold
    if thr is None:
        conf_path = resolve(args.conformal)
        thr = float(load_conformal_json(conf_path)["m_safe"]) if conf_path.is_file() else 0.0

    R = horizontal_probe_rotation()
    axis = np.linspace(-args.limit_m, args.limit_m, args.volume_resolution, dtype=np.float32)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    volume_b = np.stack((X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), axis=1)
    gt = solve_ird_grid_gt(
        volume_b,
        R,
        seed_npz=resolve(args.seed_gt),
        n_ik_seeds=args.ik_seeds,
        robot_spec_path=resolve(args.robot_spec),
    )
    clearance, _ = neural_clearance(field, volume_b, R, T_axis_world=T_axis)
    pred = clearance >= thr
    paths = render_volume_plots(
        volume_b, gt, clearance, out_dir=out, decision_threshold=thr
    )

    counts = np.array([gt[volume_b[:, 2] == z].sum() for z in axis])
    best_z = float(axis[int(np.argmax(counts))])
    sa = np.linspace(-args.limit_m, args.limit_m, args.slice_resolution, dtype=np.float32)
    SX, SY = np.meshgrid(sa, sa, indexing="xy")
    slice_b = np.stack((SX.reshape(-1), SY.reshape(-1), np.full(SX.size, best_z, dtype=np.float32)), axis=1)
    slice_gt = solve_ird_grid_gt(
        slice_b,
        R,
        seed_npz=resolve(args.seed_gt),
        n_ik_seeds=max(args.ik_seeds, 24),
        seed=72,
        robot_spec_path=resolve(args.robot_spec),
    )
    gradient = render_gradient_slice(
        field,
        R,
        z_value=best_z,
        xy_limit=args.limit_m,
        resolution=args.slice_resolution,
        gt=slice_gt,
        out_path=out / "horizontal_probe_ird_gradient.png",
        T_axis_world=T_axis,
        decision_threshold=thr,
    )
    paths["gradient"] = gradient
    metrics = {
        "chart": "flange_9d",
        "decision_threshold": float(thr),
        "volume_points": int(len(gt)),
        "volume_gt_reachable": int(gt.sum()),
        "volume_neural_reachable": int(pred.sum()),
        "volume_accuracy": float((gt == pred).mean()),
        "volume_balanced_accuracy": _balanced(gt, pred),
        "volume_false_positive_rate": float(pred[~gt].mean()) if (~gt).any() else 0.0,
        "volume_false_negative_rate": float((~pred[gt]).mean()) if gt.any() else 0.0,
        "slice_base_z_in_tcp_m": best_z,
        "slice_points": int(len(slice_gt)),
        "slice_gt_reachable": int(slice_gt.sum()),
        "files": {k: str(v) for k, v in paths.items()},
    }
    report = out / "horizontal_probe_ird_visualization.json"
    write_viz_report(report, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
