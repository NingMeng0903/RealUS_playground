"""Fit independent geometric-zero and one-sided false-accept calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from ird_playground.calib import (
    false_acceptance_report,
    fit_unreachable_safety_threshold,
    fit_zero_bias,
)
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.ird.splits import FiveWaySplitConfig, five_way_split_indices
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.neural.train_signed import load_signed_train_config, require_source_pose_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/rm4d_signed_production.yaml"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output", type=Path, default=Path("data/calib/conformal_rm4d_signed.json"))
    ap.add_argument("--allow-stale-checkpoint", action="store_true")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    cfg = load_signed_train_config(cfg_path, root=root)
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    robot_spec_path = raw_cfg.get("build", {}).get("robot_spec")
    if robot_spec_path is not None and not Path(robot_spec_path).is_absolute():
        robot_spec_path = root / robot_spec_path
    robot_spec = load_robot_model_spec(robot_spec_path)

    ckpt = args.checkpoint or Path(cfg.checkpoint)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    field = ReachabilitySDF.load(
        ckpt, expected_robot=robot_spec, allow_stale=args.allow_stale_checkpoint
    )

    gt_path = Path(cfg.gt_npz)
    gt = np.load(gt_path, allow_pickle=False)
    meta = json.loads(str(gt["meta_json"].item())) if "meta_json" in gt.files else {}
    if meta.get("zero_boundary_schema") != "on_manifold_se3_interpolation_v1":
        raise ValueError("dataset lacks independently encoded on-manifold zero poses")
    arrays = {key: gt[key] for key in gt.files if key != "meta_json"}
    source = require_source_pose_id(arrays)
    splits = five_way_split_indices(
        arrays["boundary_id"],
        source,
        seed=cfg.seed,
        config=FiveWaySplitConfig(
            selection_fraction=cfg.val_fraction,
            zero_calibration_fraction=cfg.zero_calibration_fraction,
            safety_calibration_fraction=cfg.safety_calibration_fraction,
            test_fraction=cfg.test_fraction,
        ),
    )

    zero_idx = splits["zero_calibration"]
    zero_mask = (
        (arrays["boundary_id"][zero_idx] >= 0)
        & (arrays["classification_weight"][zero_idx] == 0)
        & np.isclose(arrays["sdf_target"][zero_idx], 0.0)
    )
    zero_rows = zero_idx[zero_mask]
    zero_result = fit_zero_bias(
        field.score_np(arrays["canonical"][zero_rows]), seed=cfg.seed + 91
    )

    safety_idx = splits["safety_calibration"]
    safety_mask = (
        (arrays["classification_weight"][safety_idx] > 0)
        & (arrays["reachable"][safety_idx] < 0.5)
    )
    safety_rows = safety_idx[safety_mask]
    safety_scores = field.score_np(arrays["canonical"][safety_rows]) - zero_result.zero_bias
    safety_result = fit_unreachable_safety_threshold(safety_scores, alpha=args.alpha)

    test_idx = splits["test"]
    test_idx = test_idx[arrays["classification_weight"][test_idx] > 0]
    test_scores = field.score_np(arrays["canonical"][test_idx]) - zero_result.zero_bias
    test_report = false_acceptance_report(
        test_scores,
        arrays["reachable"][test_idx] > 0.5,
        safety_result.safety_threshold,
    )

    out = {
        "schema": "ird_clearance_calibration_v2",
        **zero_result.to_dict(),
        **safety_result.to_dict(),
        "threshold": safety_result.safety_threshold,
        "m_safe": zero_result.zero_bias + safety_result.safety_threshold,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256(ckpt),
        "dataset": str(gt_path),
        "dataset_sha256": _sha256(gt_path),
        "split_seed": cfg.seed,
        "test_report": test_report,
        "output_scale": cfg.sdf_target_scale,
        "metric": meta.get("metric"),
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
