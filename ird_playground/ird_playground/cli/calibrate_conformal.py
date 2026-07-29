"""Fit split-conformal clearance threshold for the signed IRD field.

Writes ``threshold`` (ρ) so conservative clearance is ``score − ρ``.
Optimization / precheck should require ``score ≥ ρ`` (or calibrated ≥ 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ird_playground.calib.conformal import (
    empirical_coverage,
    fit_split_conformal,
)
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.neural.train_signed import (
    _split_indices,
    load_signed_train_config,
    require_source_pose_id,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/rm4d_signed_production.yaml"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.05, help="Miscoverage level (default 5%).")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/calib/conformal_rm4d_signed.json"),
    )
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
        ckpt,
        expected_robot=robot_spec,
        allow_stale=args.allow_stale_checkpoint,
    )

    gt = np.load(cfg.gt_npz, allow_pickle=False)
    arrays = {k: gt[k] for k in gt.files if k != "meta_json"}
    source = require_source_pose_id(arrays)
    tr_idx, va_idx = _split_indices(
        arrays["boundary_id"], cfg.val_fraction, cfg.seed, source
    )
    cls_w = arrays.get("classification_weight", np.ones(len(arrays["reachable"]), dtype=np.float32))
    # Fit on train supervised rows; report coverage on val.
    tr = tr_idx[cls_w[tr_idx] > 0]
    va = va_idx[cls_w[va_idx] > 0]
    # Cap calib size for speed while keeping coverage stable.
    rng = np.random.default_rng(cfg.seed + 91)
    if len(tr) > 200_000:
        tr = rng.choice(tr, size=200_000, replace=False)

    scores_tr = field.score_np(arrays["canonical"][tr])
    y_tr = arrays["reachable"][tr] > 0.5
    result = fit_split_conformal(scores_tr, y_tr, alpha=float(args.alpha))

    scores_va = field.score_np(arrays["canonical"][va])
    y_va = arrays["reachable"][va] > 0.5
    cov = empirical_coverage(scores_va, y_va, result.threshold, margin=0.0)

    out = {
        **result.to_dict(),
        "checkpoint": str(ckpt),
        "m_safe": float(result.threshold),
        "note": "Require raw score >= m_safe, or use calibrated_clearance = score - threshold >= 0.",
        "val_coverage": cov,
        "n_fit": int(len(tr)),
        "n_val": int(len(va)),
        "sdf_target_scale": float(
            (raw_cfg.get("loss") or {}).get("sdf_target_scale", 1.0)
        ),
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
