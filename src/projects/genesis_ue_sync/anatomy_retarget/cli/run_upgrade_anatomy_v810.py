#!/usr/bin/env python3
"""Upgrade a reviewed V8 operator into the untrusted V8.10 rebuild candidate."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    with_source_driver_coupling,
)
from projects.genesis_ue_sync.anatomy_retarget.leg_centerline_v810 import (
    LEG_CENTERLINE_SCHEMA_VERSION_V810,
    build_leg_centerline_coefficients_v810,
)
from projects.genesis_ue_sync.anatomy_retarget.reference_fit_v8 import (
    apply_v810_reference_policies,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
    load_rigged_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.tube_frames_v8 import (
    bake_tube_coupling_v8,
    tube_coupling_pack_to_runtime_fields_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    rigged_asset_digest,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
    save_source_operator,
)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--foot-product", type=Path, required=True)
    parser.add_argument("--beta-a", type=Path, required=True)
    parser.add_argument("--reference-a", type=Path, required=True)
    parser.add_argument("--beta-b", type=Path, required=True)
    parser.add_argument("--reference-b", type=Path, required=True)
    parser.add_argument("--gender", default="male")
    parser.add_argument(
        "--algorithm-version",
        default="leg-centerline-oral-vessel-v8.10",
    )
    parser.add_argument(
        "--oracle-version",
        default="contact-independent-v8.10",
    )
    parser.add_argument(
        "--correction-version",
        default="rebuild-013-centerline-v8.10",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    operator_path = args.operator.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    operator = load_source_operator(operator_path)
    foot_path = args.foot_product.expanduser().resolve()
    foot_product = load_rigged_asset(foot_path)
    template, reference_report = apply_v810_reference_policies(
        operator.template_asset,
        foot_product=foot_product,
    )
    template = with_source_driver_coupling(template)
    preliminary_mechanism = {
        str(name): np.asarray(value).copy()
        for name, value in operator.mechanism_coefficients.items()
    }
    preliminary_mechanism[
        "leg_centerline_v810.schema_version"
    ] = np.asarray([LEG_CENTERLINE_SCHEMA_VERSION_V810], dtype=np.int32)
    preliminary = replace(
        operator,
        template_asset=template,
        mechanism_coefficients=preliminary_mechanism,
        provenance={
            **dict(operator.provenance),
            "source_asset_digest": rigged_asset_digest(template),
        },
    )
    preliminary.validate()
    beta_a_path = args.beta_a.expanduser().resolve()
    beta_b_path = args.beta_b.expanduser().resolve()
    beta_a = np.asarray(np.load(beta_a_path, allow_pickle=False), dtype=np.float32)
    beta_b = np.asarray(np.load(beta_b_path, allow_pickle=False), dtype=np.float32)
    source_a = materialize_subject(
        preliminary,
        betas=beta_a,
        gender=args.gender,
    ).rigged_asset
    source_b = materialize_subject(
        preliminary,
        betas=beta_b,
        gender=args.gender,
    ).rigged_asset
    reference_a_path = args.reference_a.expanduser().resolve()
    reference_b_path = args.reference_b.expanduser().resolve()
    reference_a = load_rigged_asset(reference_a_path)
    reference_b = load_rigged_asset(reference_b_path)
    leg_coefficients, leg_report = build_leg_centerline_coefficients_v810(
        samples=(
            (beta_a, source_a, reference_a),
            (beta_b, source_b, reference_b),
        ),
        domains=operator.fixed_material_domains,
    )

    tube_pack, tube_report = bake_tube_coupling_v8(template)
    runtime_coefficients = {
        str(name): np.asarray(value).copy()
        for name, value in operator.runtime_coefficients.items()
        if not str(name).startswith("tube_coupling_v8.")
    }
    runtime_coefficients.update(tube_coupling_pack_to_runtime_fields_v8(tube_pack))
    mechanism_coefficients = {
        str(name): np.asarray(value).copy()
        for name, value in operator.mechanism_coefficients.items()
    }
    mechanism_coefficients.update(leg_coefficients)
    provenance = {
        **dict(operator.provenance),
        "source_asset_digest": rigged_asset_digest(template),
        "v810_parent_operator_runtime_digest": operator.runtime_digest(
            validate=False
        ),
        "v810_parent_operator_audit_digest": operator.audit_digest(),
        "v810_template_digest": rigged_asset_digest(template),
        "v810_foot_product_file_digest": _file_digest(foot_path),
        "v810_beta_a_file_digest": _file_digest(beta_a_path),
        "v810_reference_a_file_digest": _file_digest(reference_a_path),
        "v810_beta_b_file_digest": _file_digest(beta_b_path),
        "v810_reference_b_file_digest": _file_digest(reference_b_path),
    }
    correction_report = {
        **dict(operator.correction_report),
        "passed": False,
        "publishable": False,
        "tongue": "no_tongue_display; oral_visibility_policy_v2",
        "pelvis_correction": "identity",
        "leg_centerline_v810": leg_report,
        "reference_policies_v810": reference_report,
        "tube_coupling_final_rest_v810": tube_report,
    }
    upgraded = replace(
        operator,
        template_asset=template,
        mechanism_coefficients=mechanism_coefficients,
        runtime_coefficients=runtime_coefficients,
        algorithm_version=args.algorithm_version,
        oracle_version=args.oracle_version,
        correction_version=args.correction_version,
        provenance=provenance,
        correction_report=correction_report,
        quality_report={
            "publishable": False,
            "reason": (
                "rebuild_013 requires Genesis visual review and independent "
                "V8.10 acceptance before release"
            ),
        },
    )
    upgraded.validate()
    saved = save_source_operator(output, upgraded)
    print(
        "SourceOperatorV8 V8.10 "
        f"runtime={upgraded.runtime_digest(validate=False)} "
        f"vertices={leg_report['vertex_count']} "
        f"tube={tube_report.get('backend')} publishable=false -> {saved}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
