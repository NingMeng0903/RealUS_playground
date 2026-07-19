"""Evaluate point field vs GT + optimization-oriented P2 checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ird_playground.neural.metrics import (
    PassThresholds,
    evaluate_optimization_suite,
    point_field_pass,
)
from ird_playground.ird.export_gt import load_ird_gt, make_synthetic_ird_gt
from ird_playground.neural.model import NeuralIRD
from ird_playground.neural.train import evaluate_point_field, load_train_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    ap.add_argument("--gt-npz", type=Path, default=None)
    ap.add_argument("--synthetic-n", type=int, default=2048)
    ap.add_argument("--skip-opt", action="store_true", help="Skip gradient/rail/region suite")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    train_cfg = load_train_config(cfg_path, root=root) if cfg_path.exists() else None

    ckpt = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    net = NeuralIRD.load(ckpt)

    if args.gt_npz is not None:
        arrays = load_ird_gt(args.gt_npz if args.gt_npz.is_absolute() else root / args.gt_npz)
    elif train_cfg is not None and train_cfg.gt_npz:
        arrays = load_ird_gt(train_cfg.gt_npz)
    else:
        arrays = make_synthetic_ird_gt(args.synthetic_n, seed=1)

    metrics = evaluate_point_field(net, arrays)
    if not args.skip_opt:
        metrics.update(evaluate_optimization_suite(net, arrays, seed=0))

    thr = PassThresholds(
        mae_max=train_cfg.mae_max if train_cfg else 0.35,
        spearman_min=train_cfg.spearman_min if train_cfg else 0.70,
        boundary_iou_min=train_cfg.boundary_iou_min if train_cfg else 0.70,
        grad_cosine_min=train_cfg.grad_cosine_min if train_cfg else 0.30,
        ascent_improve_min=train_cfg.ascent_improve_min if train_cfg else 0.40,
        rail_ad_fd_rel_max=train_cfg.rail_ad_fd_rel_max if train_cfg else 0.25,
        rail_sign_agree_min=train_cfg.rail_sign_agree_min if train_cfg else 0.80,
        region_improve_min=train_cfg.region_improve_min if train_cfg else 0.40,
    )
    ok = point_field_pass(metrics, thr)
    metrics["pass"] = ok
    metrics["thresholds"] = {
        "mae_max": thr.mae_max,
        "spearman_min": thr.spearman_min,
        "boundary_iou_min": thr.boundary_iou_min,
        "grad_cosine_min": thr.grad_cosine_min,
        "ascent_improve_min": thr.ascent_improve_min,
        "rail_ad_fd_rel_max": thr.rail_ad_fd_rel_max,
        "rail_sign_agree_min": thr.rail_sign_agree_min,
        "region_improve_min": thr.region_improve_min,
    }
    report = root / "data/reports/eval_point.json"
    report.parent.mkdir(parents=True, exist_ok=True)

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, (int, float)):
            return obj
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    payload = _jsonify(metrics)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
