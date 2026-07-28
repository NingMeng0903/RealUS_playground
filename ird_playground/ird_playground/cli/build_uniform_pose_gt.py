"""Build uniform full-workspace GPU pose GT for signed IRD."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.ird.gpu_uniform_pose_gt import (
    UniformPoseGtConfig,
    build_uniform_pose_gt,
    save_uniform_pose_gt,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/gpu_uniform_pose_production.yaml"))
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    path = args.config if args.config.is_absolute() else root / args.config
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = dict(raw.get("sampling") or {})
    for key in ("seed_gt_npz", "output_npz", "robot_spec"):
        if key in data and not Path(data[key]).is_absolute():
            data[key] = str(root / data[key])
    cfg = UniformPoseGtConfig(**data)
    arrays, meta = build_uniform_pose_gt(cfg)
    save_uniform_pose_gt(cfg.output_npz, arrays, meta)
    print(f"wrote {cfg.output_npz} N={meta['n']} reachable={meta['n_reachable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
