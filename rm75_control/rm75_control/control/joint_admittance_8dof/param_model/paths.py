"""Paths for parametric slider/rail model and Genesis viewer."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_DIR / "config"
ASSETS_DIR = PACKAGE_DIR / "assets"
VIEWER_DIR = PACKAGE_DIR / "viewer"
URDF_CACHE_DIR = ASSETS_DIR / ".genesis_urdf_cache"

DEFAULT_SPEC_YAML = CONFIG_DIR / "slider_rail.yaml"
DEFAULT_URDF = ASSETS_DIR / "RM75-6F-8dof.genesis.urdf"
GENERATED_URDF = ASSETS_DIR / "RM75-6F-8dof.slider.generated.urdf"
CUDA_SHIM_DIR = VIEWER_DIR / ".cuda_shim"
