"""Canonical on-disk layout for leg volume coordinate datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

LegSide = Literal["left", "right"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def leg_volume_dataset_root() -> Path:
    return repo_root() / "dataset/processed/anatomy_retarget/leg_volume_coordinates"


def leg_volume_atlas_dir() -> Path:
    return leg_volume_dataset_root() / "atlas"


def leg_volume_layered_atlas_dir() -> Path:
    """Final registered/Butterfly + layered Laplace3D atlas used for material lookup."""
    return leg_volume_dataset_root() / "atlas_layered_laplace3d"


def leg_volume_production_dir() -> Path:
    """Clean production package for material atlas, vessels, bake files, and figures."""
    return leg_volume_dataset_root() / "production"


def leg_volume_production_atlas_dir() -> Path:
    return leg_volume_production_dir() / "atlas"


def leg_volume_production_vessels_dir() -> Path:
    return leg_volume_production_dir() / "vessels"


def leg_volume_production_figures_dir() -> Path:
    return leg_volume_production_dir() / "figures"


def leg_volume_bake_dir() -> Path:
    return leg_volume_dataset_root() / "bake"


def leg_volume_figures_dir() -> Path:
    return leg_volume_bake_dir() / "figures"


def atlas_path(side: LegSide) -> Path:
    return leg_volume_atlas_dir() / f"atlas_{side}.npz"


def layered_atlas_path(side: LegSide) -> Path:
    return leg_volume_layered_atlas_dir() / f"atlas_{side}.npz"


def production_atlas_path(side: LegSide) -> Path:
    return leg_volume_production_atlas_dir() / f"atlas_{side}.npz"


def production_vessel_material_path() -> Path:
    return leg_volume_production_vessels_dir() / "vessel_material_coordinates.npz"


def resolve_repo_path(raw: str | Path, *, base_dir: Path | None = None) -> Path:
    """Resolve config paths relative to the repository root."""
    text = str(raw).strip()
    if not text:
        raise ValueError("Path must not be empty.")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = repo_root()
    return (root / path).resolve()


def ensure_dataset_layout() -> None:
    leg_volume_atlas_dir().mkdir(parents=True, exist_ok=True)
    leg_volume_figures_dir().mkdir(parents=True, exist_ok=True)
    leg_volume_production_atlas_dir().mkdir(parents=True, exist_ok=True)
    leg_volume_production_vessels_dir().mkdir(parents=True, exist_ok=True)
    leg_volume_production_figures_dir().mkdir(parents=True, exist_ok=True)
