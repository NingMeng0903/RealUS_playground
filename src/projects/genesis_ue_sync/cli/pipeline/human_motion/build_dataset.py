from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a small generated/refit human motion dataset from prompt lines.")
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--scene-spec", type=Path, default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    parse_args()
    raise SystemExit(
        "build_dataset: the old SimplePhysicsRefitter dataset path was removed. "
        "Generate motions first, then run the Genesis PHC runtime with --online-hamiltonian."
    )


if __name__ == "__main__":
    main()
