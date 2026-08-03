#!/usr/bin/env python3
"""Build and independently check the frozen two-beta Male V4 matrix on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget._quarantine_v4.dynamic_main_chain_retarget_v4 import (
    build_dynamic_main_chain_retarget_v4,
    save_dynamic_main_chain_subject_v4,
)
from projects.genesis_ue_sync.anatomy_retarget._quarantine_v4.dynamic_main_chain_validation_v4 import (
    check_dynamic_main_chain_retarget_v4,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    BASELINE_COMMIT,
    FROZEN_CAPTURE_SHA256,
)


MATRIX_SCHEMA_VERSION_V4 = 4
MATRIX_KIND_V4 = "DynamicMainChainMatrixV4"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _defaults() -> dict[str, Path]:
    root = _repo_root()
    return {
        "operator": root
        / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8",
        "calibration": root
        / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006"
        / "anatomical_calibration_v1",
        "oracle": root
        / "outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001"
        / "blender_link_oracle_v7.npz",
        "smplx_model": root
        / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl",
        "capture_213328": root
        / "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz",
        "capture_213712": root
        / "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz",
    }


def _parser() -> argparse.ArgumentParser:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, default=defaults["operator"])
    parser.add_argument("--calibration", type=Path, default=defaults["calibration"])
    parser.add_argument("--oracle", type=Path, default=defaults["oracle"])
    parser.add_argument("--smplx-model", type=Path, default=defaults["smplx_model"])
    parser.add_argument(
        "--capture-213328", type=Path, default=defaults["capture_213328"]
    )
    parser.add_argument(
        "--capture-213712", type=Path, default=defaults["capture_213712"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-device", type=int, default=0)
    return parser


def _require_cuda(device_index: int) -> dict[str, Any]:
    if device_index < 0:
        raise RuntimeError("--cuda-device must be non-negative")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on deployment image.
        raise RuntimeError("V4 matrix build requires CUDA-enabled PyTorch") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() <= device_index:
        raise RuntimeError(
            "V4 matrix build requires a visible CUDA GPU; check CUDA_VISIBLE_DEVICES"
        )
    torch.cuda.set_device(device_index)
    probe = torch.empty(1, device=f"cuda:{device_index}")
    probe.fill_(1.0)
    torch.cuda.synchronize(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "required": True,
        "available": True,
        "device_index": int(device_index),
        "device_name": str(properties.name),
        "device_capability": list(torch.cuda.get_device_capability(device_index)),
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    cuda = _require_cuda(args.cuda_device)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V4 matrix: {output}")
    operator_path = args.operator.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    oracle_path = args.oracle.expanduser().resolve()
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    captures = {
        "213328": args.capture_213328.expanduser().resolve(),
        "213712": args.capture_213712.expanduser().resolve(),
    }
    capture_sha = {label: _sha256(path) for label, path in captures.items()}
    if capture_sha != FROZEN_CAPTURE_SHA256:
        raise ValueError("V4 CLI capture inputs differ from the frozen SHA-256 set")

    operator = load_source_operator(operator_path, mmap=True)
    calibration = load_anatomical_calibration_v1(
        calibration_path, operator=operator, required_scope="full_main_chain"
    )
    model = load_smplx_model_v7(model_path)
    betas: dict[str, np.ndarray] = {}
    poses: dict[str, np.ndarray] = {}
    for label, capture in captures.items():
        with np.load(capture, allow_pickle=False) as data:
            betas[label] = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
            poses[f"pose_{label}"] = np.asarray(
                easymocap_fit_to_smplx55(
                    data["Rh"], data["poses"], model_path=model_path
                ),
                dtype=np.float64,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        provenance = {
            "baseline_commit": BASELINE_COMMIT,
            "operator_manifest_sha256": _sha256(operator_path / "manifest.json"),
            "calibration_manifest_sha256": _sha256(
                calibration_path / "manifest.json"
            ),
            "oracle_sha256": _sha256(oracle_path),
            "smplx_model_sha256": model_sha,
            "capture_sha256s": capture_sha,
            "smplx_gender": "male",
            "cuda": cuda,
        }
        subjects: dict[str, Any] = {}
        passed = True
        for label in ("213328", "213712"):
            value = build_dynamic_main_chain_retarget_v4(
                operator,
                calibration,
                betas=betas[label],
                subject_label=label,
                capture_sha256=capture_sha[label],
                smplx_model=model,
                smplx_model_sha256=model_sha,
                recorded_poses=poses,
                gender="male",
                fit_device=f"cuda:{args.cuda_device}",
            )
            checker = check_dynamic_main_chain_retarget_v4(
                value,
                operator=operator,
                calibration=calibration,
                smplx_model=model,
                smplx_model_path=model_path,
                capture_paths=captures,
                oracle_path=oracle_path,
            )
            subject_path = temporary / f"subject_{label}"
            save_dynamic_main_chain_subject_v4(
                subject_path,
                value,
                checker_report=checker,
                provenance=provenance,
            )
            checker_path = temporary / f"subject_{label}_check_v4.json"
            checker_path.write_text(
                json.dumps(checker, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            build_seconds = float(value.build_report.get("elapsed_seconds", float("inf")))
            subject_passed = bool(checker["passed"] and build_seconds < 30.0)
            subjects[label] = {
                "path": subject_path.name,
                "checker": checker_path.name,
                "checker_sha256": _sha256(checker_path),
                "passed": subject_passed,
                "build_seconds": build_seconds,
                "checker_seconds": float(checker["elapsed_seconds"]),
            }
            passed = passed and subject_passed

        elapsed = float(time.perf_counter() - started)
        passed = bool(passed and elapsed < 120.0)
        manifest = {
            "schema_version": MATRIX_SCHEMA_VERSION_V4,
            "artifact_kind": MATRIX_KIND_V4,
            "baseline_commit": BASELINE_COMMIT,
            "subjects": subjects,
            "passed": passed,
            "decision": "needs_rerender" if passed else "rejected_for_redesign",
            "accepted_scope": "full_main_chain_shadow_v4" if passed else "none",
            "smplx_gender": "male",
            "smplx_model_sha256": model_sha,
            "capture_sha256s": capture_sha,
            "cuda": cuda,
            "cold_elapsed_seconds": elapsed,
            "cold_limit_seconds": 120.0,
            "per_beta_limit_seconds": 30.0,
            "publishable": False,
            "trusted_latest_updated": False,
            "vessel_repair_started": False,
            "complete": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"{MATRIX_KIND_V4} passed={str(passed).lower()} "
        f"cuda={cuda['device_name']} -> {output}"
    )
    return 0 if passed else 2


def _entrypoint() -> int:
    try:
        return main()
    except Exception as exc:
        print(f"{MATRIX_KIND_V4} failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
