"""Train the RM4D signed reachability operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ird_playground.neural.train_signed import load_signed_train_config, train_signed_field


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/rm4d_signed_production.yaml"))
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    path = args.config if args.config.is_absolute() else root / args.config
    result = train_signed_field(load_signed_train_config(path, root=root))
    print(json.dumps(result["metrics"], indent=2))
    print(f"checkpoint -> {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
