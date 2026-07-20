"""Train generic Neural IRD point field (hyperparams from YAML)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ird_playground.neural.train import load_train_config, train_point_field


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_phase_a.yaml"),
        help="Training YAML (configs/train_phase_a.yaml or train_phase_b.yaml)",
    )
    ap.add_argument(
        "--gt-npz",
        type=Path,
        default=None,
        help="Optional override of data.gt_npz",
    )
    ap.add_argument("--checkpoint", type=Path, default=None, help="Optional override of io.checkpoint")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing train config: {cfg_path}")

    cfg = load_train_config(cfg_path, root=root)
    if args.gt_npz is not None:
        cfg.gt_npz = str(args.gt_npz if args.gt_npz.is_absolute() else root / args.gt_npz)
    if args.checkpoint is not None:
        cfg.checkpoint = str(args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint)

    result = train_point_field(cfg)
    report = Path(cfg.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result["val_metrics"], indent=2))
    print(f"checkpoint → {result['checkpoint']}")
    print(f"report → {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
