from __future__ import annotations

from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget import (
    _quality_failure_blocks_publish,
)


def test_quality_failure_is_diagnostic_by_default() -> None:
    assert not _quality_failure_blocks_publish(passed=False, enforce_quality_gate=False)


def test_explicit_quality_enforcement_blocks_publish() -> None:
    assert _quality_failure_blocks_publish(passed=False, enforce_quality_gate=True)


def test_quality_success_never_blocks_publish() -> None:
    assert not _quality_failure_blocks_publish(passed=True, enforce_quality_gate=True)
