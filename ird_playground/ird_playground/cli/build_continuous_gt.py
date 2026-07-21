"""Build continuous FK/multi-seed-IK IRD GT from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.ird.continuous_gt import ContinuousGtConfig, build_continuous_ird_gt
from ird_playground.ird.export_gt import assert_gt_contract, save_ird_gt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/continuous_gt_smoke.yaml"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    out = Path(raw.get("out", "data/ird/continuous_ik_gt.npz"))
    if args.out is not None:
        out = args.out
    if not out.is_absolute():
        out = root / out
    cfg = ContinuousGtConfig(**dict(raw.get("sampling") or {}))
    arrays, meta = build_continuous_ird_gt(cfg)
    assert_gt_contract(arrays)
    meta["config_path"] = str(config_path)
    save_ird_gt(out, arrays, meta)
    accepted = int(
        meta.get(
            "n_boundary_rays_accepted",
            int(meta.get("n_position_curves", 0)) + int(meta.get("n_rotation_curves", 0)),
        )
    )
    print(f"wrote {out} N={len(arrays['reachable'])} accepted_curves={accepted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
