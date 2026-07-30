#!/usr/bin/env python3
"""Upgrade a reviewed V8 operator into the untrusted V8.11 candidate."""

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
)
from projects.genesis_ue_sync.anatomy_retarget.reference_fit_v8 import (
    apply_v810_reference_policies,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
    load_rigged_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.tube_frames_v8 import (
    bake_tube_coupling_v8,
    tube_coupling_pack_from_runtime_fields_v8,
    tube_coupling_pack_to_runtime_fields_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
    rigged_asset_digest,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    save_source_operator,
)
from projects.genesis_ue_sync.anatomy_retarget.version_v8 import (
    SOURCE_OPERATOR_ALGORITHM_VERSION,
    SOURCE_OPERATOR_CORRECTION_VERSION,
    SOURCE_OPERATOR_ORACLE_VERSION,
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
    parser.add_argument(
        "--algorithm-version",
        default=SOURCE_OPERATOR_ALGORITHM_VERSION,
    )
    parser.add_argument(
        "--oracle-version",
        default=SOURCE_OPERATOR_ORACLE_VERSION,
    )
    parser.add_argument(
        "--correction-version",
        default=SOURCE_OPERATOR_CORRECTION_VERSION,
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

    tube_pack, tube_report = bake_tube_coupling_v8(template)
    parent_tube_pack = tube_coupling_pack_from_runtime_fields_v8(
        operator.runtime_coefficients
    )
    frozen_digest_match = {
        "topology": (
            tube_pack.topology_digest == parent_tube_pack.topology_digest
        ),
        "domain": tube_pack.domain_digest == parent_tube_pack.domain_digest,
        "weight": tube_pack.weight_digest == parent_tube_pack.weight_digest,
    }
    if not all(frozen_digest_match.values()):
        raise ValueError(
            "V8.10 L0 tube pack changed a frozen topology/domain/weight digest"
        )
    tube_report = {
        **tube_report,
        "final_template_rest_authentication": {
            "parent_rest_digest": parent_tube_pack.rest_digest,
            "template_rest_digest": tube_pack.rest_digest,
            "topology_digest": tube_pack.topology_digest,
            "domain_digest": tube_pack.domain_digest,
            "weight_digest": tube_pack.weight_digest,
            "frozen_digest_match": frozen_digest_match,
        },
    }
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
    mechanism_coefficients[
        "leg_centerline_v810.schema_version"
    ] = np.asarray([LEG_CENTERLINE_SCHEMA_VERSION_V810], dtype=np.int32)
    provenance = {
        **dict(operator.provenance),
        "source_asset_digest": rigged_asset_digest(template),
        "v810_parent_operator_runtime_digest": operator.runtime_digest(
            validate=False
        ),
        "v810_parent_operator_audit_digest": operator.audit_digest(),
        "v810_template_digest": rigged_asset_digest(template),
        "v810_foot_product_file_digest": _file_digest(foot_path),
    }
    leg_report = {
        "schema_version": LEG_CENTERLINE_SCHEMA_VERSION_V810,
        "method": "single_pass_contact_first_joint_chain_v810",
        "calibration_reference": "none_beta_specific_materialize",
        "ba9_used_for_coefficients": False,
        "runtime_station_module": False,
        "pelvis_correction": "identity",
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
        "leg=single-pass-contact-chain "
        f"tube={tube_report.get('backend')} publishable=false -> {saved}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
