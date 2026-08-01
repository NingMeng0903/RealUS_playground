"""Build the untrusted V8.14 bone-review operator from the frozen 142 baseline."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .anatomy_lbs import with_source_driver_coupling
from .articular_fit_v8 import calibrate_coupled_joint_roll_glide_v8
from .functional_joint_v8 import (
    build_functional_joint_frames_v8,
    build_pelvis_harmonic_cage_v8,
)
from .leg_centerline_v810 import LEG_CENTERLINE_SCHEMA_VERSION_V810
from .operator_bake_v8 import build_frozen_domains_v8
from .reference_fit_v8 import apply_v810_reference_policies
from .tube_frames_v8 import (
    tube_coupling_pack_from_runtime_fields_v8,
)
from .v7_artifacts import rigged_asset_digest
from .v8_artifacts import SourceOperatorV8


BONE_REVIEW_ALGORITHM_VERSION = "functional-joint-retarget-v8.14"
BONE_REVIEW_ORACLE_VERSION = "frozen-contact-v71-score-v8.14"
BONE_REVIEW_CORRECTION_VERSION = "pelvis-harmonic-cage-v8.14"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tube_frozen_authentication(
    baseline: SourceOperatorV8,
    candidate_pack: Any,
) -> dict[str, Any]:
    baseline_pack = tube_coupling_pack_from_runtime_fields_v8(
        baseline.runtime_coefficients
    )
    matches = {
        "rest_vertices": bool(
            np.array_equal(
                np.asarray(candidate_pack.rest_vertices_m),
                np.asarray(baseline_pack.rest_vertices_m),
            )
        ),
        "topology": candidate_pack.topology_digest == baseline_pack.topology_digest,
        "domain": candidate_pack.domain_digest == baseline_pack.domain_digest,
        "driver_indices": bool(
            np.array_equal(
                np.asarray(candidate_pack.driver_indices),
                np.asarray(baseline_pack.driver_indices),
            )
        ),
        "driver_weights": bool(
            np.array_equal(
                np.asarray(candidate_pack.driver_weights),
                np.asarray(baseline_pack.driver_weights),
            )
        ),
        "weight_digest": candidate_pack.weight_digest == baseline_pack.weight_digest,
    }
    if not all(matches.values()):
        failed = sorted(name for name, passed in matches.items() if not passed)
        raise ValueError(
            "bone-review operator changed frozen tube material: " + ", ".join(failed)
        )
    return {
        "passed": True,
        "bitwise_matches": matches,
        "baseline_rest_digest": baseline_pack.rest_digest,
        "candidate_rest_digest": candidate_pack.rest_digest,
        "topology_digest": candidate_pack.topology_digest,
        "domain_digest": candidate_pack.domain_digest,
        "weight_digest": candidate_pack.weight_digest,
    }


def build_bone_review_operator_v8(
    baseline: SourceOperatorV8,
    *,
    baseline_path: Path | str | None = None,
) -> SourceOperatorV8:
    """Re-bake V8.14 fields without importing old rebuild_013 candidate data.

    ``rebuild_012`` is the frozen L0 source used by the 142 implementation.
    Its foot product is already composed into the common topology; applying the
    V8.10 rigid-foot policy against that same composed product is an identity
    geometry operation which adds only the reviewed policy metadata.
    """

    baseline.validate()
    template, reference_report = apply_v810_reference_policies(
        baseline.template_asset,
        foot_product=baseline.template_asset,
    )
    if not np.array_equal(
        np.asarray(template.vertices_rest),
        np.asarray(baseline.template_asset.vertices_rest),
    ):
        raise ValueError("self-authored rigid-foot policy changed the 142 rest geometry")
    domains = build_frozen_domains_v8(template, baseline.fixed_material_domains)
    template = with_source_driver_coupling(template)
    functional_frames = build_functional_joint_frames_v8(template, domains=domains)
    frame_metadata = {
        str(name).removeprefix("functional_joint_v8."): np.asarray(value).tolist()
        for name, value in functional_frames.coefficient_fields().items()
    }
    template = replace(
        template,
        metadata={
            **dict(template.metadata or {}),
            "functional_joint_frames_v8": frame_metadata,
        },
    )
    template = with_source_driver_coupling(template)
    template, coupled_report = calibrate_coupled_joint_roll_glide_v8(
        template,
        domains=domains,
        maximum_translation_m=0.002 / 1.35,
        runtime_maximum_translation_m=0.002,
    )
    pelvis_fields, pelvis_report = build_pelvis_harmonic_cage_v8(
        template,
        domains=domains,
    )
    mechanism = {
        str(name): np.asarray(value).copy()
        for name, value in baseline.mechanism_coefficients.items()
        if not str(name).startswith(("functional_joint_v8.", "pelvis_cage_v8."))
    }
    mechanism.update(functional_frames.coefficient_fields())
    mechanism.update(pelvis_fields)
    mechanism["leg_centerline_v810.schema_version"] = np.asarray(
        [LEG_CENTERLINE_SCHEMA_VERSION_V810], dtype=np.int32
    )

    # The 142 operator's tube pack contains the already-reviewed routed rest
    # material.  Those coordinates intentionally differ from the raw template
    # tube meshes, so re-baking here would silently undo that route.  Bone-only
    # review inherits the pack byte-for-byte; subject materialization remains
    # responsible for expressing it in the subject bind frames.
    tube_pack = tube_coupling_pack_from_runtime_fields_v8(
        baseline.runtime_coefficients
    )
    tube_auth = _tube_frozen_authentication(baseline, tube_pack)
    runtime = {
        str(name): np.asarray(value).copy()
        for name, value in baseline.runtime_coefficients.items()
    }
    tube_report = {
        "available": True,
        "passed": True,
        "backend": tube_pack.backend,
        "tube_vertex_count": int(len(tube_pack.vertex_ids)),
        "source_bone_count": int(tube_pack.source_bone_count),
        "influence_slots": int(tube_pack.influence_slots),
        "rest_geometry_repaired": False,
        "inheritance": "bitwise_from_142_frozen_runtime_pack",
    }

    baseline_location = None if baseline_path is None else Path(baseline_path).resolve()
    provenance = {
        **dict(baseline.provenance),
        "source_asset_digest": rigged_asset_digest(template),
        "bone_review_baseline": "142ece5_rebuild_012_frozen_l0",
        "bone_review_baseline_runtime_digest": baseline.runtime_digest(validate=False),
        "bone_review_baseline_audit_digest": baseline.audit_digest(),
        "old_rebuild_013_data_reused": False,
        "old_29e_data_reused": False,
    }
    if baseline_location is not None:
        manifest_path = baseline_location / "manifest.json"
        if manifest_path.is_file():
            provenance["bone_review_baseline_manifest_sha256"] = _file_sha256(
                manifest_path
            )

    candidate = replace(
        baseline,
        template_asset=template,
        fixed_material_domains=domains,
        mechanism_coefficients=mechanism,
        runtime_coefficients=runtime,
        algorithm_version=BONE_REVIEW_ALGORITHM_VERSION,
        oracle_version=BONE_REVIEW_ORACLE_VERSION,
        correction_version=BONE_REVIEW_CORRECTION_VERSION,
        provenance=provenance,
        correction_report={
            **dict(baseline.correction_report),
            "passed": False,
            "publishable": False,
            "scope": "pelvis_and_appendicular_bone_review_only",
            "reference_policies_v810": reference_report,
            "functional_joint_frames_v8": dict(functional_frames.report),
            "coupled_joint_roll_glide_v814": coupled_report,
            "pelvis_harmonic_cage_v8": pelvis_report,
            "tube_coupling_final_rest_v8": {
                **dict(tube_report),
                "frozen_authentication": tube_auth,
            },
            "blood_vessel_geometry_repair": "deferred_until_bone_review_signature",
        },
        quality_report={
            "publishable": False,
            "human_signature": "pending",
            "reason": "BoneReviewPackV8 and Genesis review are not signed",
        },
    )
    candidate.validate()
    return candidate


__all__ = [
    "BONE_REVIEW_ALGORITHM_VERSION",
    "BONE_REVIEW_CORRECTION_VERSION",
    "BONE_REVIEW_ORACLE_VERSION",
    "build_bone_review_operator_v8",
]
