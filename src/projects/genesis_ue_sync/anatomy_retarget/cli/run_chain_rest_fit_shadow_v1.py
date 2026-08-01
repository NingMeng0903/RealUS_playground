"""Build the fixed two-beta lower-chain rest-fit shadow matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.blender_link_oracle_v7 import (
    EXPECTED_OPERATOR_RUNTIME_DIGEST,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_rest_fit_v1 import (
    build_lower_chain_rest_fit_v1,
    check_chain_rest_fit_v1,
    save_chain_rest_fit_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


EXPECTED_MODEL_SHA256 = "af7ebc82e44cf098598685474c0592049ddfaca8e850feb0c2b88343f9aacee3"
EXPECTED_CAPTURES = {
    "213328": "c7a6c3783dc7b764e1f8013ab0a8a45d0380b81c97ac929f67c7a5a526eecbc1",
    "213712": "9887848b7b086d71a875beea50b1d7c7819a11c7b67996fe0d83f451da79b689",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Node 2 matrix: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    operator = load_source_operator(args.operator.expanduser().resolve(), mmap=True)
    if operator.runtime_digest(validate=False) != EXPECTED_OPERATOR_RUNTIME_DIGEST:
        raise ValueError("Node 2 shadow requires the frozen 142 operator")
    calibration = load_anatomical_calibration_v1(
        args.calibration,
        operator=operator,
        required_scope="lower_chain",
    )
    model_path = args.smplx_model.expanduser().resolve()
    model_sha = _sha256(model_path)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise ValueError("SMPL-X male model differs from the frozen input")
    model = load_smplx_model_v7(model_path)
    capture_paths = {
        "213328": args.capture_213328.expanduser().resolve(),
        "213712": args.capture_213712.expanduser().resolve(),
    }
    subjects: dict[str, dict] = {}
    for label, capture in capture_paths.items():
        capture_sha = _sha256(capture)
        if capture_sha != EXPECTED_CAPTURES[label]:
            raise ValueError(f"capture {label} differs from the frozen input")
        with np.load(capture, allow_pickle=False) as data:
            betas = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
        subject_started = time.perf_counter()
        value = build_lower_chain_rest_fit_v1(
            operator,
            calibration,
            betas=betas,
            subject_label=label,
            capture_sha256=capture_sha,
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
        checker = check_chain_rest_fit_v1(
            value,
            operator=operator,
            calibration=calibration,
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
        if not checker["passed"]:
            raise ValueError(f"Node 2 subject {label} failed independent checking")
        subject_path = output / f"subject_{label}"
        save_chain_rest_fit_v1(
            subject_path,
            value,
            operator=operator,
            calibration=calibration,
            smplx_model=model,
            smplx_model_sha256=model_sha,
        )
        _write_json(output / f"subject_{label}_check.json", checker)
        subjects[label] = {
            "path": str(subject_path),
            "content_digest": checker["content_digest"],
            "build_seconds": checker["build_seconds"],
            "check_seconds": checker["elapsed_seconds"],
            "end_to_end_seconds": float(time.perf_counter() - subject_started),
            "passed": True,
        }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "ChainRestFitMatrixV1",
        "accepted_scope": "lower_chain_shadow",
        "source_operator_digest": operator.runtime_digest(validate=False),
        "calibration_digest": _calibration_content_digest(calibration),
        "smplx_model_sha256": model_sha,
        "subjects": subjects,
        "elapsed_seconds": float(time.perf_counter() - started),
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
    }
    _write_json(output / "manifest.json", manifest)
    print(
        f"ChainRestFitMatrixV1 passed=true subjects=2 "
        f"seconds={manifest['elapsed_seconds']:.3f} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
