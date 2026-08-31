#!/usr/bin/env python3
"""Backward-compatible entry for Cartesian trajectory planning.

Prefer::

    python -m peirastic.DEMO.cartesian
"""

from peirastic.DEMO.cartesian import main

if __name__ == "__main__":
    print(
        "[NOTE] this is Cartesian trajectory planning; "
        "prefer python -m peirastic.DEMO.cartesian",
        flush=True,
    )
    raise SystemExit(main())
