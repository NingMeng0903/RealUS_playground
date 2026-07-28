#!/usr/bin/env python3
"""Build and evaluate fail-closed schema-v7 anatomy artifacts.

The expensive geometry/contact solver writes a prepared bake-data NPZ.  This
entry point packages that data, materializes one beta, and applies arbitrary
poses without importing or invoking Blender in the runtime stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_drive_translation,
    load_easymocap_smplx_fit_drive,
    smplx_pose_hash,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import load_rigged_asset
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
)
from projects.genesis_ue_sync.anatomy_retarget.operator_bake_v7 import (
    build_prepared_bake_data_v7,
    save_prepared_bake_data_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    ANATOMY_V7_SCHEMA_VERSION,
    SourceOperatorV7,
    apply_subject_pose,
    load_source_operator,
    load_subject_asset,
    materialize_subject,
    rigged_asset_digest,
    save_source_operator,
    save_subject_asset,
)


_BAKE_PREFIXES = {
    "fixed_domain__": "fixed_material_domains",
    "joint_spline__": "joint_splines",
    "contact_envelope__": "contact_envelopes",
    "vessel_avoidance__": "vessel_avoidance_fields",
    "runtime_coefficient__": "runtime_coefficients",
}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _prepared_bake_data(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"prepared bake data does not exist: {path}")
    required = {
        "beta_vertex_basis",
        "beta_rest_joint_basis",
        "beta_bind_twist_basis",
        "internal_handle_basis",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"prepared bake data is missing arrays: {missing}")
        result: dict[str, Any] = {
            name: np.asarray(data[name]).copy() for name in required
        }
        for prefix, field_name in _BAKE_PREFIXES.items():
            values = {
                name[len(prefix) :]: np.asarray(data[name]).copy()
                for name in data.files
                if name.startswith(prefix) and name[len(prefix) :]
            }
            result[field_name] = values
    return result


def _load_betas(args: argparse.Namespace) -> np.ndarray:
    if args.betas is not None:
        result = np.asarray(args.betas, dtype=np.float32)
    else:
        path = Path(args.betas_file).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"beta input does not exist: {path}")
        if path.suffix.lower() == ".npy":
            result = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        else:
            with np.load(path, allow_pickle=False) as data:
                key = "shapes" if "shapes" in data.files else "betas"
                if key not in data.files:
                    raise ValueError(f"{path} must contain shapes or betas")
                result = np.asarray(data[key], dtype=np.float32)
    result = result.reshape(-1)
    if result.shape != (10,) or not np.all(np.isfinite(result)):
        raise ValueError("beta input must contain exactly 10 finite values")
    return result


def _load_pose(args: argparse.Namespace, subject: Any) -> tuple[np.ndarray, np.ndarray]:
    if args.zero_pose:
        return np.zeros((55, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)
    if args.motion_npz is not None:
        pose, raw_translation = load_easymocap_smplx_fit_drive(
            Path(args.motion_npz).expanduser().resolve(),
            gender=subject.gender,
            model_path=args.smplx_model,
        )
        translation = easymocap_drive_translation(
            np.asarray(pose, dtype=np.float32).reshape(55, 3)[0],
            raw_translation,
            subject.rigged_asset.rest_joints[0],
        )
        return np.asarray(pose, dtype=np.float32).reshape(55, 3), translation
    path = Path(args.pose_npy).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"pose input does not exist: {path}")
    if path.suffix.lower() == ".npy":
        pose = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        translation = np.zeros(3, dtype=np.float32)
    else:
        with np.load(path, allow_pickle=False) as data:
            key = "pose_axis_angle" if "pose_axis_angle" in data.files else "pose"
            if key not in data.files:
                raise ValueError(f"{path} must contain pose_axis_angle or pose")
            pose = np.asarray(data[key], dtype=np.float32)
            translation = (
                np.asarray(data["transl"], dtype=np.float32).reshape(3)
                if "transl" in data.files
                else np.zeros(3, dtype=np.float32)
            )
    try:
        pose = pose.reshape(55, 3)
    except ValueError as exc:
        raise ValueError("pose input must contain exactly 55x3 values") from exc
    if args.translation is not None:
        translation = np.asarray(args.translation, dtype=np.float32)
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(translation)):
        raise ValueError("pose/translation input contains non-finite values")
    return pose, translation


def _add_bake_template(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "bake-template",
        help="package a validated source rig and prepared offline V7 fields",
    )
    parser.add_argument("--source-asset", type=Path, required=True)
    parser.add_argument(
        "--bake-data",
        type=Path,
        help=(
            "optional already-prepared NPZ; omit to run the one-time beta/"
            "material-field bake"
        ),
    )
    parser.add_argument("--source-reference", type=Path)
    parser.add_argument("--uncorrected-source-asset", type=Path)
    parser.add_argument("--fixed-domains", type=Path)
    parser.add_argument("--template-betas-file", type=Path)
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--cage", type=Path)
    parser.add_argument("--smplx-model", type=Path)
    parser.add_argument("--prepared-output", type=Path)
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--blender-version", required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--correction-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=_run_bake_template)


def _add_bake_patella_oracle(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "bake-patella-oracle",
        help="freeze the canonical patella response from the V71 Action export",
    )
    parser.add_argument("--action-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.set_defaults(handler=_run_bake_patella_oracle)


def _add_materialize_beta(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "materialize-beta",
        help="materialize one pose-independent SubjectAssetV7",
    )
    parser.add_argument("--operator", type=Path, required=True)
    beta_group = parser.add_mutually_exclusive_group(required=True)
    beta_group.add_argument("--betas", type=float, nargs=10)
    beta_group.add_argument("--betas-file", type=Path)
    parser.add_argument("--gender", default="male")
    parser.add_argument(
        "--patella-oracle",
        type=Path,
        help="frozen patella oracle; omit only to reproduce a pre-oracle candidate",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=_run_materialize_beta)


def _add_apply_pose(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "apply-pose",
        help="evaluate SubjectAssetV7 without Blender or pose rebaking",
    )
    parser.add_argument("--subject", type=Path, required=True)
    pose_group = parser.add_mutually_exclusive_group(required=True)
    pose_group.add_argument("--pose-npy", type=Path)
    pose_group.add_argument("--motion-npz", type=Path)
    pose_group.add_argument("--zero-pose", action="store_true")
    parser.add_argument("--translation", type=float, nargs=3)
    parser.add_argument("--smplx-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=_run_apply_pose)


def _add_diagnose_matrix(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "diagnose-matrix",
        help="run the fail-closed V7 acceptance matrix over subjects × poses",
    )
    parser.add_argument("--operator", type=Path)
    parser.add_argument(
        "--subject",
        action="append",
        required=True,
        dest="subjects",
        metavar="LABEL=PATH",
        help="repeatable subject LABEL=PATH (at least one)",
    )
    parser.add_argument(
        "--pose",
        action="append",
        required=True,
        dest="poses",
        metavar="LABEL=PATH|LABEL=zero",
        help="repeatable pose LABEL=PATH or LABEL=zero (at least one)",
    )
    parser.add_argument("--domains", type=Path, required=True)
    parser.add_argument("--patella-oracle", type=Path, required=True)
    parser.add_argument("--action-oracle", type=Path)
    parser.add_argument("--sweep-count", type=int, default=13)
    parser.add_argument("--vertices-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=_run_diagnose_matrix)


def _add_evidence_pack(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "evidence-pack",
        help="generate the V7 acceptance evidence PNG pack with JSON sidecars",
    )
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument(
        "--subject",
        action="append",
        required=True,
        dest="subjects",
        metavar="LABEL=PATH",
        help="repeatable subject LABEL=PATH (at least one)",
    )
    parser.add_argument(
        "--pose",
        action="append",
        required=True,
        dest="poses",
        metavar="LABEL=PATH|LABEL=zero",
        help="repeatable pose LABEL=PATH or LABEL=zero (at least one)",
    )
    parser.add_argument("--domains", type=Path, required=True)
    parser.add_argument("--patella-oracle", type=Path, required=True)
    parser.add_argument("--posed-dir", type=Path)
    parser.add_argument("--sweep-count", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.set_defaults(handler=_run_evidence_pack)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_bake_template(subparsers)
    _add_bake_patella_oracle(subparsers)
    _add_materialize_beta(subparsers)
    _add_apply_pose(subparsers)
    _add_diagnose_matrix(subparsers)
    _add_evidence_pack(subparsers)
    return parser


def _run_bake_template(args: argparse.Namespace) -> int:
    source_asset_path = args.source_asset.expanduser().resolve()
    source_blend = args.source_blend.expanduser().resolve()
    if not source_asset_path.is_file():
        raise ValueError(f"source asset does not exist: {source_asset_path}")
    if not source_blend.is_file() or source_blend.suffix.lower() != ".blend":
        raise ValueError(f"source blend does not exist or is not .blend: {source_blend}")
    source_asset = load_rigged_asset(source_asset_path, validate=True)
    beta_bake_report: dict[str, Any] | None = None
    if args.bake_data is not None:
        prepared_path = args.bake_data.expanduser().resolve()
        prepared = _prepared_bake_data(prepared_path)
    else:
        required = {
            "--source-reference": args.source_reference,
            "--uncorrected-source-asset": args.uncorrected_source_asset,
            "--fixed-domains": args.fixed_domains,
            "--template-betas-file": args.template_betas_file,
            "--canonical-dir": args.canonical_dir,
            "--cage": args.cage,
            "--smplx-model": args.smplx_model,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "one-time bake-template is missing " + ", ".join(missing)
            )
        source_reference_path = args.source_reference.expanduser().resolve()
        template_beta_path = args.template_betas_file.expanduser().resolve()
        source_reference = load_rigged_asset(
            source_reference_path, validate=True
        )
        uncorrected_template = load_rigged_asset(
            args.uncorrected_source_asset.expanduser().resolve(),
            validate=True,
        )
        template_betas = np.asarray(
            np.load(template_beta_path, allow_pickle=False), dtype=np.float32
        ).reshape(-1)
        source_asset, prepared, beta_bake_report = build_prepared_bake_data_v7(
            corrected_template_asset=source_asset,
            uncorrected_template_asset=uncorrected_template,
            source_reference_asset=source_reference,
            fixed_domains=FrozenJointMaterialDomainsV7.load_json(
                args.fixed_domains.expanduser().resolve()
            ),
            template_betas=template_betas,
            canonical_dir=args.canonical_dir.expanduser().resolve(),
            cage_path=args.cage.expanduser().resolve(),
            smplx_model_path=args.smplx_model.expanduser().resolve(),
        )
        prepared_path = (
            args.prepared_output.expanduser().resolve()
            if args.prepared_output is not None
            else args.output.expanduser().resolve().with_suffix(
                ".prepared.npz"
            )
        )
        save_prepared_bake_data_v7(prepared_path, prepared)
    provenance = {
        "source_asset_digest": rigged_asset_digest(source_asset),
        "source_asset_file_digest": _file_digest(source_asset_path),
        "source_blend_digest": _file_digest(source_blend),
        "blender_version": str(args.blender_version).strip(),
        "prepared_bake_data_digest": _file_digest(prepared_path),
    }
    correction_report = _read_json(
        args.correction_report.expanduser().resolve(),
        label="correction report",
    )
    if beta_bake_report is not None:
        correction_report = {
            **correction_report,
            "operator_beta_bake_v7": beta_bake_report,
        }
    operator = SourceOperatorV7(
        template_asset=source_asset,
        beta_vertex_basis=prepared["beta_vertex_basis"],
        beta_rest_joint_basis=prepared["beta_rest_joint_basis"],
        beta_bind_twist_basis=prepared["beta_bind_twist_basis"],
        internal_handle_basis=prepared["internal_handle_basis"],
        fixed_material_domains=prepared["fixed_material_domains"],
        joint_splines=prepared["joint_splines"],
        contact_envelopes=prepared["contact_envelopes"],
        vessel_avoidance_fields=prepared["vessel_avoidance_fields"],
        runtime_coefficients=prepared["runtime_coefficients"],
        provenance=provenance,
        correction_report=correction_report,
        quality_report=_read_json(
            args.quality_report.expanduser().resolve(), label="quality report"
        ),
    )
    output = save_source_operator(args.output.expanduser().resolve(), operator)
    print(f"SourceOperatorV7 {operator.content_digest()} -> {output}")
    return 0


def _run_bake_patella_oracle(args: argparse.Namespace) -> int:
    from projects.genesis_ue_sync.anatomy_retarget.patella_oracle_v7 import (
        extract_patella_law_v7,
        save_patella_oracle_v7,
    )

    action = args.action_oracle.expanduser().resolve()
    if not action.is_file():
        raise ValueError(f"V71 action oracle does not exist: {action}")
    law = extract_patella_law_v7(action)
    output = save_patella_oracle_v7(args.output.expanduser().resolve(), law)
    summary = {
        "schema_version": ANATOMY_V7_SCHEMA_VERSION,
        "artifact": "AnatomyPatellaOracleV7",
        "content_digest": law.content_digest(),
        "action_source_digest": law.action_source_digest,
        "action_frame_count": int(law.action_frame_count),
        "sides": {
            side: {
                "response_slope": float(law.response_slope[side]),
                "response_max_residual_deg": float(
                    law.response_max_residual_deg[side]
                ),
                "keyed_frame_count": int(law.keyed_frame_count[side]),
                "observed_max_flexion_deg": float(
                    law.observed_max_flexion_deg[side]
                ),
                "penetration_envelope_m": float(law.penetration_envelope_m[side]),
            }
            for side in ("left", "right")
        },
        "corridor_m": [float(law.corridor_min_m), float(law.corridor_max_m)],
    }
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(f"AnatomyPatellaOracleV7 {law.content_digest()} -> {output}")
    return 0


def _load_patella_law(args: argparse.Namespace) -> Any | None:
    path = getattr(args, "patella_oracle", None)
    if path is None:
        return None
    from projects.genesis_ue_sync.anatomy_retarget.patella_oracle_v7 import (
        load_patella_oracle_v7,
    )

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"patella oracle does not exist: {resolved}")
    return load_patella_oracle_v7(resolved)


def _run_materialize_beta(args: argparse.Namespace) -> int:
    operator = load_source_operator(args.operator.expanduser().resolve())
    subject = materialize_subject(
        operator,
        betas=_load_betas(args),
        gender=str(args.gender),
        patella_law=_load_patella_law(args),
    )
    output = save_subject_asset(args.output.expanduser().resolve(), subject)
    print(
        f"SubjectAssetV7 {subject.content_digest()} cache_key={subject.cache_key} "
        f"publishable=false -> {output}"
    )
    return 0


def _run_apply_pose(args: argparse.Namespace) -> int:
    subject = load_subject_asset(args.subject.expanduser().resolve())
    pose, translation = _load_pose(args, subject)
    vertices = apply_subject_pose(
        subject,
        pose_axis_angle=pose,
        transl=translation,
        # load_subject_asset has already verified the exact embedded payload,
        # decoded it, and run the full structural validation.
        validate=False,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pose_digest = smplx_pose_hash(pose, translation)
    np.savez(
        output,
        schema_version=np.asarray(ANATOMY_V7_SCHEMA_VERSION, dtype=np.int32),
        artifact_kind=np.asarray("AnatomyPoseEvaluationV7"),
        subject_digest=np.asarray(subject.content_digest()),
        pose_digest=np.asarray(pose_digest),
        pose_axis_angle=np.asarray(pose, dtype=np.float32),
        transl=np.asarray(translation, dtype=np.float32),
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(subject.rigged_asset.faces, dtype=np.int32),
    )
    print(
        f"AnatomyPoseEvaluationV7 subject={subject.content_digest()} "
        f"pose={pose_digest} -> {output}"
    )
    return 0


def _run_diagnose_matrix(args: argparse.Namespace) -> int:
    from projects.genesis_ue_sync.anatomy_retarget.acceptance_matrix_v7 import (
        _json_ready,
        load_matrix_pose_v7,
        load_matrix_subject_v7,
        parse_label_value_pair,
        run_acceptance_matrix_v7,
    )
    from projects.genesis_ue_sync.anatomy_retarget.patella_oracle_v7 import (
        load_patella_oracle_v7,
    )

    from projects.genesis_ue_sync.anatomy_retarget.vessel_gates_v7 import (
        vessel_topology_digest_v7,
    )

    operator_digest = ""
    source_topology_digest: str | None = None
    if args.operator is not None:
        operator = load_source_operator(args.operator.expanduser().resolve())
        operator_digest = str(operator.content_digest())
        # Independent reference for the vessel topology gate: the pre-beta
        # template's own tube selection, not the materialized subject's.
        source_topology_digest = vessel_topology_digest_v7(operator.template_asset)

    subjects = []
    for raw in args.subjects:
        label, value = parse_label_value_pair(raw)
        subjects.append(load_matrix_subject_v7(label, value))

    poses = []
    for raw in args.poses:
        label, value = parse_label_value_pair(raw)
        poses.append(load_matrix_pose_v7(label, value))

    domains_path = args.domains.expanduser().resolve()
    if not domains_path.is_file():
        raise ValueError(f"fixed domains do not exist: {domains_path}")
    oracle_path = args.patella_oracle.expanduser().resolve()
    if not oracle_path.is_file():
        raise ValueError(f"patella oracle does not exist: {oracle_path}")

    report = run_acceptance_matrix_v7(
        subjects=subjects,
        poses=poses,
        domains=FrozenJointMaterialDomainsV7.load_json(domains_path),
        law=load_patella_oracle_v7(oracle_path),
        action_oracle_path=(
            None
            if args.action_oracle is None
            else args.action_oracle.expanduser().resolve()
        ),
        operator_digest=operator_digest,
        source_topology_digest=source_topology_digest,
        sweep_count=int(args.sweep_count),
        vertices_dir=(
            None
            if args.vertices_dir is None
            else args.vertices_dir.expanduser().resolve()
        ),
    )

    for cell_key, cell in report["cells"].items():
        status = "PASS" if cell.get("passed") else "FAIL"
        failures = ",".join(cell.get("failures") or []) or "-"
        print(f"{status} {cell_key} {failures}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"AcceptanceMatrixV7 passed={report['passed']} -> {output}")
    return 0


def _run_evidence_pack(args: argparse.Namespace) -> int:
    import shlex
    import sys

    from projects.genesis_ue_sync.anatomy_retarget.acceptance_matrix_v7 import (
        _json_ready,
        load_matrix_pose_v7,
        load_matrix_subject_v7,
        parse_label_value_pair,
    )
    from projects.genesis_ue_sync.anatomy_retarget.evidence_pack_v7 import (
        generate_evidence_pack_v7,
    )
    from projects.genesis_ue_sync.anatomy_retarget.patella_oracle_v7 import (
        load_patella_oracle_v7,
    )

    operator_path = args.operator.expanduser().resolve()
    if not operator_path.is_file():
        raise ValueError(f"source operator does not exist: {operator_path}")
    operator = load_source_operator(operator_path)
    operator_digest = str(operator.content_digest())

    subjects = []
    for raw in args.subjects:
        label, value = parse_label_value_pair(raw)
        subjects.append(load_matrix_subject_v7(label, value))

    poses = []
    for raw in args.poses:
        label, value = parse_label_value_pair(raw)
        poses.append(load_matrix_pose_v7(label, value))

    domains_path = args.domains.expanduser().resolve()
    if not domains_path.is_file():
        raise ValueError(f"fixed domains do not exist: {domains_path}")
    oracle_path = args.patella_oracle.expanduser().resolve()
    if not oracle_path.is_file():
        raise ValueError(f"patella oracle does not exist: {oracle_path}")

    command = " ".join(shlex.quote(part) for part in sys.argv)
    manifest = generate_evidence_pack_v7(
        subjects=subjects,
        poses=poses,
        domains=FrozenJointMaterialDomainsV7.load_json(domains_path),
        law=load_patella_oracle_v7(oracle_path),
        operator_digest=operator_digest,
        output_dir=args.output_dir.expanduser().resolve(),
        posed_dir=(
            None
            if args.posed_dir is None
            else args.posed_dir.expanduser().resolve()
        ),
        sweep_count=int(args.sweep_count),
        command=command,
    )

    for item in manifest["files"]:
        print(item["png"])
    for missing in manifest["missing_views"]:
        print(
            "MISSING "
            f"{missing['beta']}/{missing['pose']}/{missing['view']}: "
            f"{missing['reason']}"
        )

    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(
        f"EvidencePackV7 complete={manifest['complete']} "
        f"files={manifest['generated_file_count']} -> {manifest['output_dir']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
