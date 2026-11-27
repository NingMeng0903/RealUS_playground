"""Filesystem inputs for this experiment's simulator."""

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "local_assets"
PRIMITIVE = "NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/primitive.py"
REPO_ROOT = next((parent for parent in HERE.parents if (parent / PRIMITIVE).is_file()), None)
if REPO_ROOT is None:
    raise RuntimeError("Run this experiment inside the full RealUS_playground checkout.")


def input_path(variable, filename):
    configured = os.environ.get(variable)
    return Path(configured).expanduser().resolve() if configured else ASSETS / filename


ANATOMY_ARCHIVE = input_path("REALUS_GB_ANATOMY", "registered_torso.npz")
SMPLX_ARCHIVE = input_path("REALUS_GB_SMPLX", "SMPLX_MALE.npz")
ATLAS_ARCHIVE = input_path("REALUS_GB_ATLAS", "thorax_atlas.npz")
