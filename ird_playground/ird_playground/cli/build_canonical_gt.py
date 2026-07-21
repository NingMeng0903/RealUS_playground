"""Build RM4D invariant signed-field GT from collision-checked pose samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.ird.canonical_gt import CanonicalGtConfig, build_canonical_gt, save_canonical_gt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/rm4d_signed_production.yaml"))
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    path = args.config if args.config.is_absolute() else root / args.config
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    build = dict(raw.get("build") or {})
    for key in ("source_npz", "output_npz"):
        if key in build and not Path(build[key]).is_absolute():
            build[key] = str(root / build[key])
    if "auxiliary_npz" in build:
        build["auxiliary_npz"] = tuple(
            str(Path(p) if Path(p).is_absolute() else root / p)
            for p in build["auxiliary_npz"]
        )
    cfg = CanonicalGtConfig(**build)
    arrays, meta = build_canonical_gt(cfg)
    save_canonical_gt(cfg.output_npz, arrays, meta)
    print(f"wrote {cfg.output_npz} N={meta['n']} groups={meta['n_boundary_groups']} oriented={meta['n_oriented_groups']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
