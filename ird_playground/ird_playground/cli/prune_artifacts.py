"""Prune old Neural IRD checkpoints and logs; dry-run unless --apply is passed."""

from __future__ import annotations

import argparse
from pathlib import Path


KEEP_CHECKPOINT_NAMES = {"selected.pt", "best.pt", "best_iou.pt", "best_joint.pt", "best_margin.pt", "latest.pt"}


def _candidates(root: Path, keep_epochs: int) -> list[Path]:
    doomed: list[Path] = []
    for path in sorted(root.rglob("*.pt")):
        if path.name in KEEP_CHECKPOINT_NAMES:
            continue
        if path.name.startswith("epoch_"):
            try:
                epoch = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if epoch > keep_epochs:
                doomed.append(path)
    for path in sorted(root.rglob("*.log")):
        doomed.append(path)
    return doomed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--keep-epochs", type=int, default=0, help="Keep epoch_NNNN.pt through this epoch")
    ap.add_argument("--apply", action="store_true", help="Actually delete candidates")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    candidates = _candidates(root, max(0, int(args.keep_epochs)))
    bytes_total = sum(p.stat().st_size for p in candidates if p.exists())
    for path in candidates:
        print(path)
    print(f"candidates={len(candidates)} reclaim_bytes={bytes_total} apply={args.apply}")
    if args.apply:
        for path in candidates:
            path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
