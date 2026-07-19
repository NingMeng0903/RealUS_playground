"""Export IRD GT NPZ from a capability map (sampling from YAML)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.ird.export_gt import IrdGtConfig, export_ird_gt_from_capability_map, save_ird_gt
from ird_playground.ird.capability_io import load_capability_map_dir
from ird_playground.ird.map_loader import resolve_map_dir


def load_ird_gt_config(path: Path, *, root: Path) -> tuple[Path, Path, IrdGtConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    samp = dict(raw.get("sampling") or {})
    map_dir = Path(raw.get("map_dir", ""))
    out = Path(raw.get("out", "data/ird/gt_samples.npz"))
    if not map_dir.is_absolute():
        map_dir = (root / map_dir).resolve()
    if not out.is_absolute():
        out = root / out

    n_int = samp.get("n_interior")
    n_bnd = samp.get("n_boundary")
    n_ext = samp.get("n_exterior")
    n_pos = int(samp.get("n_positive", 700_000))
    n_neg = int(samp.get("n_negative", 500_000))
    if n_int is None and n_bnd is None and n_ext is None:
        n_tot = n_pos + n_neg
        n_int = int(round(0.35 * n_tot))
        n_bnd = int(round(0.40 * n_tot))
        n_ext = max(0, n_tot - n_int - n_bnd)
    else:
        n_int = int(n_int or 0)
        n_bnd = int(n_bnd or 0)
        n_ext = int(n_ext or 0)

    cfg = IrdGtConfig(
        n_interior=n_int,
        n_boundary=n_bnd,
        n_exterior=n_ext,
        n_positive=n_pos,
        n_negative=n_neg,
        max_orients_per_voxel=int(samp.get("max_orients_per_voxel", 24)),
        hard_negative_frac=float(samp.get("hard_negative_frac", 0.45)),
        hard_negative_radius_m=float(samp.get("hard_negative_radius_m", 0.06)),
        sigma_p_m=float(samp.get("sigma_p_m", 0.03)),
        sigma_r_deg=float(samp.get("sigma_r_deg", 10.0)),
        boundary_d_lo=float(samp.get("boundary_d_lo", 0.02)),
        boundary_d_hi=float(samp.get("boundary_d_hi", 0.08)),
        bbox_margin_m=float(samp.get("bbox_margin_m", 0.20)),
        comfort_from=str(samp.get("comfort_from", "auto")),
        k_candidates=int(samp.get("k_candidates", 4)),
        seed=int(samp.get("seed", 0)),
    )
    return map_dir, out, cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/ird_gt_config.yaml"))
    ap.add_argument("--map", type=Path, default=None, help="Override map_dir")
    ap.add_argument("--out", type=Path, default=None, help="Override out")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    map_dir, out, cfg = load_ird_gt_config(cfg_path, root=root)
    if args.map is not None:
        map_dir = resolve_map_dir(args.map if args.map.is_absolute() else root / args.map)
    else:
        map_dir = resolve_map_dir(map_dir)
    if args.out is not None:
        out = args.out if args.out.is_absolute() else root / args.out

    cm = load_capability_map_dir(map_dir, mmap=True)
    arrays = export_ird_gt_from_capability_map(cm, cfg)
    save_ird_gt(
        out,
        arrays,
        meta={
            "map_dir": str(map_dir),
            "config": str(cfg_path),
            "n_interior": cfg.n_interior,
            "n_boundary": cfg.n_boundary,
            "n_exterior": cfg.n_exterior,
            "sigma_p_m": cfg.sigma_p_m,
            "sigma_r_deg": cfg.sigma_r_deg,
            "max_orients_per_voxel": cfg.max_orients_per_voxel,
            "seed": cfg.seed,
            "n_total": int(arrays["features"].shape[0]),
            "stratification": "0.35_interior_0.40_boundary_0.25_exterior",
            "note": "m_gt is a truncated margin label, not a strict SDF",
        },
    )
    print(f"wrote {out}  N={arrays['features'].shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
