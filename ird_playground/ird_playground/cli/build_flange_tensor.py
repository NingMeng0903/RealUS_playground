"""CLI: build 5-D flange-chart occupancy from collision-free FK samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.map.build_flange_tensor import (
    FlangeOccupancyConfig,
    build_flange_occupancy,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--step-m", type=float, default=None)
    ap.add_argument("--step-deg", type=float, default=None)
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    raw: dict = {}
    if args.config is not None:
        config_path = args.config if args.config.is_absolute() else root / args.config
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sampling = dict(raw.get("sampling") or raw)
    for key in ("robot_spec", "output_npz"):
        if key in sampling and sampling[key] is not None and not Path(str(sampling[key])).is_absolute():
            sampling[key] = str(root / str(sampling[key]))
    # CLI overrides.
    if args.out is not None:
        sampling["output_npz"] = str(args.out if args.out.is_absolute() else root / args.out)
    if args.n_samples is not None:
        sampling["n_samples"] = int(args.n_samples)
    if args.batch_size is not None:
        sampling["batch_size"] = int(args.batch_size)
    if args.seed is not None:
        sampling["seed"] = int(args.seed)
    if args.device is not None:
        sampling["device"] = str(args.device)
    if args.step_m is not None:
        sampling["step_m"] = float(args.step_m)
    if args.step_deg is not None:
        sampling["step_deg"] = float(args.step_deg)
    if "output_npz" not in sampling:
        sampling["output_npz"] = str(root / "data/maps/flange_occupancy.npz")
    # Drop unknown keys that are not dataclass fields.
    allowed = {f.name for f in FlangeOccupancyConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    cfg_kwargs = {k: v for k, v in sampling.items() if k in allowed}
    cfg = FlangeOccupancyConfig(**cfg_kwargs)
    arrays, meta = build_flange_occupancy(cfg)
    shape = meta["axes"]["shape"]
    print(
        f"wrote {cfg.output_npz} shape={shape} "
        f"occupied={meta['n_occupied_voxels']} "
        f"frac={meta['occupancy_fraction']:.6f} "
        f"fk_pos={meta['n_fk_positives']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
