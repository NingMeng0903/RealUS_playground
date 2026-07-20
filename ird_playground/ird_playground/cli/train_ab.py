"""Run the full Phase A -> Phase B Neural IRD training pipeline back-to-back.

Trains Phase A (classification-only) to completion, then warm-starts Phase B
(margin + q) from the checkpoint Phase A *actually* produced in this run —
`cfg_b.init_checkpoint` is overridden with `result_a["checkpoint"]`, regardless
of whatever `init_checkpoint` path is written in the Phase B YAML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ird_playground.neural.train import load_train_config, train_point_field, validate_phase_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config-a",
        "--phase-a",
        type=Path,
        default=Path("configs/train_phase_a.yaml"),
        help="Phase A training YAML",
    )
    ap.add_argument(
        "--config-b",
        "--phase-b",
        type=Path,
        default=Path("configs/train_phase_b.yaml"),
        help="Phase B training YAML",
    )
    ap.add_argument(
        "--gt-npz",
        type=Path,
        default=None,
        help="Optional override of data.gt_npz for both phases",
    )
    ap.add_argument(
        "--skip-a",
        action="store_true",
        help="Skip Phase A training; warm-start Phase B from its existing checkpoint",
    )
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else root / p

    path_a = _resolve(args.config_a)
    path_b = _resolve(args.config_b)
    if not path_a.exists():
        raise FileNotFoundError(f"missing Phase A config: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"missing Phase B config: {path_b}")

    cfg_a = load_train_config(path_a, root=root)
    cfg_b = load_train_config(path_b, root=root)
    if args.gt_npz is not None:
        gt = str(_resolve(args.gt_npz))
        cfg_a.gt_npz = gt
        cfg_b.gt_npz = gt

    if args.skip_a:
        checkpoint_a = cfg_b.init_checkpoint or cfg_a.checkpoint
        if not checkpoint_a or not Path(checkpoint_a).is_file():
            raise FileNotFoundError(
                f"--skip-a requires an existing Phase A checkpoint, not found: {checkpoint_a}"
            )
        result_a: dict = {"checkpoint": str(checkpoint_a), "history": [], "val_metrics": {}}
        print(f"[train_ab] --skip-a: reusing checkpoint {checkpoint_a}", flush=True)
    else:
        print(f"[train_ab] === Phase A: {path_a} ===", flush=True)
        result_a = train_point_field(cfg_a)
        report_a = Path(cfg_a.report)
        report_a.parent.mkdir(parents=True, exist_ok=True)
        report_a.write_text(json.dumps(result_a, indent=2, default=str), encoding="utf-8")
        print(f"[train_ab] Phase A checkpoint -> {result_a['checkpoint']}", flush=True)

    # Phase B always warm-starts from the checkpoint Phase A produced *in this
    # run* — not whatever init_checkpoint the Phase B YAML happens to list.
    cfg_b.init_checkpoint = result_a["checkpoint"]
    validate_phase_config(cfg_b)

    print(
        f"[train_ab] === Phase B: {path_b} (init_checkpoint={cfg_b.init_checkpoint}) ===",
        flush=True,
    )
    result_b = train_point_field(cfg_b)
    report_b = Path(cfg_b.report)
    report_b.parent.mkdir(parents=True, exist_ok=True)
    report_b.write_text(json.dumps(result_b, indent=2, default=str), encoding="utf-8")
    print(f"[train_ab] Phase B checkpoint -> {result_b['checkpoint']}", flush=True)

    summary = {
        "phase_a_checkpoint": result_a["checkpoint"],
        "phase_b_checkpoint": result_b["checkpoint"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
