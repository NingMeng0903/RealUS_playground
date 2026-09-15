#!/usr/bin/env python3
"""Render the dual-view plan figure from today's biased detect + live Orbbec RGB.

    python scripts/render_plan_rgb_figure.py
    python scripts/render_plan_rgb_figure.py --grab
"""
from __future__ import annotations

import argparse
from pathlib import Path

from peirastic.DEMO.phathom_scanning.plan_rgb_figure import (
    BIASED_DIR,
    DEFAULT_CAPTURE,
    DEFAULT_OUT,
    UNBIASED_DIR,
    grab_orbbec,
    load_plan,
    write_figure,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biased", type=Path, default=BIASED_DIR)
    parser.add_argument("--unbiased", type=Path, default=UNBIASED_DIR)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grab", action="store_true", help="open Orbbec and write --capture")
    parser.add_argument("--offline", action="store_true", help="splat the saved detect cloud")
    args = parser.parse_args()

    if args.grab:
        grab_orbbec(args.capture)
    capture = None if args.offline or not args.capture.is_file() else args.capture
    written = write_figure(
        args.out,
        unbiased=load_plan(args.unbiased),
        biased=load_plan(args.biased),
        capture=capture,
        run_fallback=args.biased,
    )
    print("wrote", written)
    print("wrote", written.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
