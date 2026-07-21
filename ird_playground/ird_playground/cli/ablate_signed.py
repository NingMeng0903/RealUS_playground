"""Train fixed RM4D data-scale partitions and report quality/time tradeoffs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from ird_playground.neural.train_signed import load_signed_train_config, train_signed_field


LEVELS = (
    ("quarter", 125_000, 23_000),
    ("half", 250_000, 46_000),
    ("full", 500_000, 91_738),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/rm4d_signed_production.yaml"))
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = args.config if args.config.is_absolute() else root / args.config
    base = load_signed_train_config(config, root=root)
    results = []
    tmp = root / "data/checkpoints/_ablation"
    for name, global_rows, groups in LEVELS:
        checkpoint = tmp / f"{name}.pt"
        report = root / f"data/reports/_ablation_{name}.json"
        cfg = replace(
            base,
            epochs=args.epochs,
            early_stop_patience=5,
            max_global_rows=global_rows,
            max_boundary_groups=groups,
            checkpoint=str(checkpoint),
            report=str(report),
        )
        print(f"[ablation] {name}: global={global_rows} boundary_groups={groups}", flush=True)
        result = train_signed_field(cfg)
        results.append({"name": name, **{k: result[k] for k in ("n_total", "n_global", "n_boundary_groups", "elapsed_seconds")}, "metrics": result["metrics"]})
        checkpoint.unlink(missing_ok=True)
        report.unlink(missing_ok=True)
    if tmp.is_dir() and not any(tmp.iterdir()):
        tmp.rmdir()
    output = root / "data/reports/ablation_rm4d_scale.json"
    output.write_text(json.dumps({"epochs": args.epochs, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"report -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
