"""Pure path-precedence tests for the signed evaluation CLI."""

from pathlib import Path

from ird_playground.cli.eval_signed import resolve_eval_path


ROOT = Path("/repo/ird_playground")


def test_config_relative_path_is_rooted():
    raw = {"io": {"conformal": "data/calib/custom_v41.json"}}
    assert resolve_eval_path(
        None, raw, "conformal", root=ROOT, default="data/calib/legacy.json"
    ) == ROOT / "data/calib/custom_v41.json"


def test_absolute_config_path_is_preserved():
    absolute = Path("/mnt/calibration/custom_v41.json")
    raw = {"io": {"conformal": str(absolute)}}
    assert resolve_eval_path(
        None, raw, "conformal", root=ROOT, default="data/calib/legacy.json"
    ) == absolute


def test_cli_path_takes_precedence_over_config_and_resolves_relative():
    raw = {"io": {"conformal": "data/calib/from_config.json"}}
    assert resolve_eval_path(
        Path("runs/calibration/from_cli.json"),
        raw,
        "conformal",
        root=ROOT,
        default="data/calib/legacy.json",
    ) == ROOT / "runs/calibration/from_cli.json"


def test_legacy_default_is_kept_when_config_key_is_absent():
    # ``io.report`` is the training-report key in older configs; it must not
    # accidentally become the evaluation output path.
    raw = {"io": {"report": "data/reports/train.json"}}
    assert resolve_eval_path(
        None,
        raw,
        "eval_report",
        root=ROOT,
        default="data/reports/eval_rm4d_signed.json",
    ) == ROOT / "data/reports/eval_rm4d_signed.json"

