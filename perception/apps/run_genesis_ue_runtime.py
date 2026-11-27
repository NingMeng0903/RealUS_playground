#!/usr/bin/env python3
"""Unified RealUS Genesis runtime entrypoint.

This is a thin entrypoint over the existing ``amass_bed_capsule_demo``.  It
keeps one Genesis scene/runtime and one canonical 5599 publisher while giving
the RealUS checkout sensible defaults:

* calibrated RealUS bed/rail scene;
* CMU 114/114_11 AMASS motion (override with ``--amass-npz``);
* simulation robot state (override with ``--robot-state-source shm``).

Examples::

  PYTHONPATH=src:. python -m perception.apps.run_genesis_ue_runtime \
      --backend cuda --show-viewer --loop --frame-step 4

  # Mirror a running C++/peirastic controller through read-only SHM.
  PYTHONPATH=src:.:rm75_control python -m perception.apps.run_genesis_ue_runtime \
      --backend cuda --show-viewer --robot-state-source shm \
      --robot-state-wait 15

  # Let the fitted EasyMocap/SMPL-X stream on 5598 drive the canonical human.
  PYTHONPATH=src:. python -m perception.apps.run_genesis_ue_runtime \
      --human-source fitted --track-subscribe tcp://127.0.0.1:5598
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_REPO = Path(os.environ.get("REALUS_PROJECT_ROOT") or Path(__file__).resolve().parents[2]).resolve()
for _path in (_REPO, _REPO / "src", _REPO / "rm75_control"):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

DEFAULT_AMASS_REL = "dataset/raw/humans/amass_hf/raw/CMU/114/114_11_poses.npz"
DEFAULT_SMPL_REL = "dataset/intermediate/humans/body_models/smpl"
DEFAULT_SCENE = "configs/scenes/realus_bed_rail_scene.yaml"


def _has_option(argv: list[str], name: str) -> bool:
    prefix = f"{name}="
    return any(arg == name or arg.startswith(prefix) for arg in argv)


def _resolve_default_amass() -> str:
    """Find the shared CMU clip without copying it into RealUS."""
    candidates = [
        Path(os.environ["REALUS_AMASS_NPZ"]).expanduser()
        for _ in (0,)
        if str(os.environ.get("REALUS_AMASS_NPZ", "")).strip()
    ]
    candidates.extend(
        [
            _REPO / DEFAULT_AMASS_REL,
            Path(os.environ.get("AMONGUS_PROJECT_ROOT", "")).expanduser() / DEFAULT_AMASS_REL,
            _REPO.parent / "Among_US" / DEFAULT_AMASS_REL,
        ]
    )
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file():
            return str(path)
    searched = ", ".join(str(path) for path in candidates)
    raise SystemExit(
        "run_genesis_ue_runtime: default CMU clip is missing; searched "
        f"{searched}. Pass --amass-npz /path/to/114_11_poses.npz or set REALUS_AMASS_NPZ."
    )


def _configure_default_body_models() -> None:
    """Point the AMASS placement/evaluator at an existing shared SMPL tree.

    RealUS intentionally does not copy the large body-model files.  Keep an
    explicit user setting authoritative, then use the checkout or its sibling
    Among_US data tree when the default AMASS path is selected.
    """
    if str(os.environ.get("AMONGUS_SMPL_MODEL_DIR", "") or "").strip():
        return
    candidates = [
        Path(os.environ["REALUS_SMPL_MODEL_DIR"]).expanduser()
        for _ in (0,)
        if str(os.environ.get("REALUS_SMPL_MODEL_DIR", "")).strip()
    ]
    candidates.extend(
        [
            _REPO / DEFAULT_SMPL_REL,
            Path(os.environ.get("AMONGUS_PROJECT_ROOT", "")).expanduser() / DEFAULT_SMPL_REL,
            _REPO.parent / "Among_US" / DEFAULT_SMPL_REL,
        ]
    )
    for candidate in candidates:
        path = candidate.resolve()
        if (path / "SMPL_NEUTRAL.pkl").is_file():
            os.environ["AMONGUS_SMPL_MODEL_DIR"] = str(path)
            return


def _with_defaults(argv: list[str]) -> list[str]:
    out = list(argv)
    if not _has_option(out, "--amass-npz"):
        # Preserve ``--help`` even on a machine where the optional shared
        # Among_US dataset has not been mounted yet.
        default_amass = (
            DEFAULT_AMASS_REL
            if any(x in {"-h", "--help"} for x in out)
            else _resolve_default_amass()
        )
        out.extend(("--amass-npz", default_amass))
    if not _has_option(out, "--scene-spec"):
        out.extend(("--scene-spec", DEFAULT_SCENE))
    if not _has_option(out, "--robot-state-source"):
        out.extend(("--robot-state-source", "simulation"))
    return out


def main(argv: list[str] | None = None) -> int:
    """Delegate to the existing Genesis app after applying RealUS defaults."""
    from projects.genesis_ue_sync.sim_platform.apps.genesis_viz import amass_bed_capsule_demo

    _configure_default_body_models()

    # The legacy app uses argparse internally.  Keep its parser and behavior
    # in one place rather than creating a second scene implementation.
    original = sys.argv
    sys.argv = [original[0], *_with_defaults(list(sys.argv[1:] if argv is None else argv))]
    try:
        result = amass_bed_capsule_demo.main()
    finally:
        sys.argv = original
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
