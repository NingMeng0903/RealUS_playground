"""Shadow-only CLI for topology-preserving anatomical chain calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    FULL_MAIN_CHAIN_SCOPE,
    build_anatomical_calibration_v1,
    check_anatomical_calibration_v1,
    load_anatomical_calibration_v1,
    save_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_BLEND_SHA256,
    EXPECTED_ORACLE_SHA256,
    EXPECTED_OPERATOR_RUNTIME_DIGEST,
    check_blender_link_oracle_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_frozen_operator(path: Path):
    operator = load_source_operator(path.expanduser().resolve(), mmap=True)
    digest = operator.runtime_digest(validate=False)
    if digest != EXPECTED_OPERATOR_RUNTIME_DIGEST:
        raise ValueError("shadow chain CLI requires the frozen 142 rebuild_012 operator")
    return operator


def _calibrate(args: argparse.Namespace) -> int:
    operator = _load_frozen_operator(args.operator)
    oracle = args.oracle_npz.expanduser().resolve()
    oracle_report_path = args.oracle_report.expanduser().resolve()
    if not oracle.is_file():
        raise FileNotFoundError(f"Blender oracle is missing: {oracle}")
    if not oracle_report_path.is_file():
        raise FileNotFoundError(f"Blender oracle report is missing: {oracle_report_path}")
    oracle_report = check_blender_link_oracle_v7(
        oracle_npz=oracle,
        oracle_report=oracle_report_path,
        operator_path=args.operator,
        require_full_action=True,
    )
    if not oracle_report["passed"] or not oracle_report["performance_pass"]:
        raise ValueError("frozen Blender linkage oracle failed independent parity")
    calibration = build_anatomical_calibration_v1(
        operator,
        source_blend_sha256=EXPECTED_BLEND_SHA256,
        blender_oracle_sha256=EXPECTED_ORACLE_SHA256,
    )
    report = check_anatomical_calibration_v1(calibration, operator=operator)
    output = save_anatomical_calibration_v1(
        args.output.expanduser().resolve(),
        calibration,
        operator=operator,
        checker_report=report,
        accepted_scope=FULL_MAIN_CHAIN_SCOPE,
    )
    print(
        f"AnatomicalCalibrationV1 passed={str(report['passed']).lower()} "
        f"joints={report['joint_count']} seconds={report['elapsed_seconds']:.3f} "
        f"-> {output}"
    )
    return 0 if report["passed"] else 1


def _check(args: argparse.Namespace) -> int:
    operator = _load_frozen_operator(args.operator)
    calibration = load_anatomical_calibration_v1(
        args.calibration,
        operator=operator,
        require_complete=True,
        required_scope=FULL_MAIN_CHAIN_SCOPE,
    )
    report = check_anatomical_calibration_v1(calibration, operator=operator)
    _write_json(args.output_json.expanduser().resolve(), report)
    print(
        f"AnatomicalCalibrationCheckV1 passed={str(report['passed']).lower()} "
        f"joints={report['joint_count']} -> {args.output_json}"
    )
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser(
        "calibrate-source", help="build and independently check the L0 calibration"
    )
    calibrate.add_argument("--operator", type=Path, required=True)
    calibrate.add_argument("--oracle-npz", type=Path, required=True)
    calibrate.add_argument("--oracle-report", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.set_defaults(handler=_calibrate)

    check = commands.add_parser(
        "check-calibration", help="recheck an existing calibration artifact"
    )
    check.add_argument("--operator", type=Path, required=True)
    check.add_argument("--calibration", type=Path, required=True)
    check.add_argument("--output-json", type=Path, required=True)
    check.set_defaults(handler=_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
