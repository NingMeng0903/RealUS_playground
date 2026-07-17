from __future__ import annotations

import sys

from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget import (
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
