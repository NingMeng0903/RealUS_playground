"""Smoke tests for three-mount RM/IRD compare config."""

from __future__ import annotations

from pathlib import Path

from ird_playground.viz.mount_compare import load_mount_compare_config


def test_mount_compare_config_loads_three_mounts():
    root = Path(__file__).resolve().parents[1]
    cfg = load_mount_compare_config(
        root / "configs/mount_compare.yaml",
        repo_root=root.parent,
    )
    ids = [m.id for m in cfg.mounts]
    assert ids == ["probe45", "tcp220", "horizontal"]
    assert cfg.style.clim == (0.0, 0.18)
    # tcp220 must target the dedicated coll map (not legacy probe45 @ 10M).
    assert cfg.mounts[1].map_dir.name == "rm75_6f_3cm_15deg_coll_tcp220"
    assert "tcp220" in cfg.mounts[1].robot_urdf.name
    assert cfg.mounts[2].map_dir.is_dir()
