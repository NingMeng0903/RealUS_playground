#!/usr/bin/env python3
"""Publish a saved SMPL-X capture (orange mesh) to Genesis track ZMQ.

Split from ``run_smplx_capture.py``: recognition stays in capture; this script
only republishes an existing ``smplx_result.npz``. Twin (window B) can already
be running with ``--track-subscribe`` — no restart needed.

Examples::

  # After A/B/C are up, publish latest saved run (loops until Ctrl+C):
  source env.sh
  $PY perception/apps/publish_saved_smplx.py --run 20260713_213712

  # One-shot burst (~2 s), enough if twin is already subscribed:
  $PY perception/apps/publish_saved_smplx.py --run 20260713_213712 --once

  # Timed hold:
  $PY perception/apps/publish_saved_smplx.py --run 20260713_213712 --duration-s 120
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> int:
    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))

    from projects.genesis_ue_sync.multiview_realtime.publish.static_smplx_track import (
        publish_static_smplx_track,
        resolve_moment_dir,
        smplx_output_root,
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=str, help="Capture run dir name or path under smplx_outputs/")
    src.add_argument("--moment-dir", type=Path, help="Explicit moment_0000 directory")
    src.add_argument("--npz", type=Path, help="Direct path to smplx_result.npz")
    ap.add_argument("--output-root", type=Path, default=smplx_output_root(repo))
    ap.add_argument("--publish-bind", type=str, default="tcp://127.0.0.1:5598")
    ap.add_argument(
        "--publish-kind",
        type=str,
        default="smplx_mesh",
        choices=["smplx_mesh", "keypoints3d", "smpl_pose"],
        help="Orange mesh uses smplx_mesh (default).",
    )
    ap.add_argument("--rate-hz", type=float, default=5.0)
    ap.add_argument("--gender", choices=["male", "female", "neutral"], default="male")
    ap.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Timed publish then exit. Omit with default loop until Ctrl+C.",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Short burst (~2 s) then exit; twin already subscribed needs no restart.",
    )
    ap.add_argument(
        "--loop",
        action="store_true",
        help="Repeat until Ctrl+C (default when neither --once nor --duration-s).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    moment = resolve_moment_dir(
        run=args.run,
        moment_dir=args.moment_dir,
        npz=args.npz,
        output_root=args.output_root,
    )

    duration_s = 2.0 if args.once else args.duration_s
    loop = bool(args.loop) or (duration_s is None and not args.once)

    try:
        diag = publish_static_smplx_track(
            moment_dir=moment,
            bind=str(args.publish_bind),
            duration_s=duration_s,
            rate_hz=float(args.rate_hz),
            publish_kind=str(args.publish_kind),
            loop=loop,
            gender=str(args.gender),
        )
    except KeyboardInterrupt:
        logging.info("publish stopped")
        return 0
    except OSError as exc:
        # Typical: Address already in use — another publisher still on 5598.
        logging.error("publish bind failed (%s): %s", args.publish_bind, exc)
        logging.error("kill the other publisher, or pass --publish-bind tcp://127.0.0.1:<port>")
        return 1

    if diag.get("stopped"):
        logging.info(
            "publish stopped kind=%s sent=%s bind=%s",
            diag.get("payload_kind"),
            diag.get("sent"),
            diag.get("bind"),
        )
    else:
        logging.info(
            "published kind=%s sent=%s bind=%s moment=%s",
            diag.get("payload_kind"),
            diag.get("sent"),
            diag.get("bind"),
            diag.get("moment_dir"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
