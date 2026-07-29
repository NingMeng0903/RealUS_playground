"""CLI: anisotropic weighted 5-D EDT on a flange occupancy tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

from ird_playground.map.signed_distance import FlangeEdtConfig, build_flange_edt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--occupancy", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    occ = args.occupancy if args.occupancy.is_absolute() else root / args.occupancy
    if args.out is None:
        out = occ.with_name(occ.stem.replace("occupancy", "sdf") + ".npz")
        if out == occ:
            out = occ.with_name(occ.stem + "_sdf.npz")
    else:
        out = args.out if args.out.is_absolute() else root / args.out
    cfg = FlangeEdtConfig(occupancy_npz=str(occ), output_npz=str(out))
    _arrays, meta = build_flange_edt(cfg)
    stats = meta["sdf_stats"]
    print(f"wrote {out}")
    print(f"warning: {meta['warning']}")
    print(
        f"sdf min={stats['min']:.4f} max={stats['max']:.4f} "
        f"mean={stats['mean']:.4f} occupied_frac={stats['occupied_fraction']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
