"""Render 3 TCP mounts × (reachability map + global IRD) = 6 paper figures.

Example::

    cd /media/camp/EXT_DRIVE/RealUS_playground
    source ird_playground/env.sh
    export PYTHONPATH="$PWD/ird_playground:$PYTHONPATH"

    # Optional: rebuild capability maps for mounts that are missing
    #   cd rm75_control && python scripts/build_coll_map.py \\
    #     --config configs/reachability/rm75_6f_3cm_15deg_coll_probe45.yaml
    #   python scripts/build_coll_map.py \\
    #     --config configs/reachability/rm75_6f_3cm_15deg_coll_tcp220.yaml

    python -m ird_playground.cli.viz_mount_compare \\
      --config configs/mount_compare.yaml \\
      --skip-missing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ird_playground.viz.mount_compare import (
    load_mount_compare_config,
    render_mount_compare,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mount_compare.yaml"),
        help="mount_compare YAML (default: configs/mount_compare.yaml under ird_playground)",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="RealUS_playground root (default: inferred from config path)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="override output directory from the YAML",
    )
    ap.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="optional mount ids to render (default: all)",
    )
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="skip mounts whose capability map is not built yet",
    )
    args = ap.parse_args(argv)

    ird_root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else ird_root / args.config
    repo = args.repo_root
    if repo is None:
        repo = ird_root.parent if (ird_root.parent / "rm75_control").is_dir() else ird_root

    config = load_mount_compare_config(cfg_path, repo_root=repo)
    if args.out_dir is not None:
        out = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
        object.__setattr__(config, "out_dir", out.resolve())  # type: ignore[misc]
        # frozen dataclass — rebuild
        from dataclasses import replace

        config = replace(config, out_dir=out.resolve())
    if args.only:
        wanted = set(args.only)
        from dataclasses import replace

        config = replace(
            config,
            mounts=tuple(m for m in config.mounts if m.id in wanted),
        )
        missing = wanted - {m.id for m in config.mounts}
        if missing:
            raise SystemExit(f"unknown mount id(s): {sorted(missing)}")

    report = render_mount_compare(config, skip_missing=bool(args.skip_missing))
    out_json = config.out_dir / "mount_compare_report.json"
    config.out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
