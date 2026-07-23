from __future__ import annotations

import sys

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget import (
    _array_content_key,
    _quality_failure_blocks_publish,
    parse_args,
)


def test_quality_failure_is_advisory_by_default() -> None:
    assert not _quality_failure_blocks_publish(passed=False)


def test_quality_failure_blocks_only_in_strict_mode() -> None:
    assert _quality_failure_blocks_publish(passed=False, enforce_quality_gate=True)


def test_quality_success_never_blocks_publish() -> None:
    assert not _quality_failure_blocks_publish(passed=True)


def test_diagnostics_only_requires_explicit_cli_flag(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_anatomy_retarget"])
    assert not parse_args().diagnostics_only

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_anatomy_retarget", "--diagnostics-only"],
    )
    assert parse_args().diagnostics_only


def test_rigid_skin_clamping_is_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_anatomy_retarget"])
    assert not parse_args().stage1_clamp_rigid_to_skin

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_anatomy_retarget", "--stage1-clamp-rigid-to-skin"],
    )
    assert parse_args().stage1_clamp_rigid_to_skin


def test_pose_cache_geometry_key_tracks_exact_rest_geometry() -> None:
    first = np.zeros((4, 3), dtype=np.float32)
    second = first.copy()
    second[2, 1] = 1.0e-4

    assert _array_content_key(first) == _array_content_key(first.copy())
    assert _array_content_key(first) != _array_content_key(second)
