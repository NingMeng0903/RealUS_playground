"""Build an auditable long right-leg surface/internal path asset.

This is a data-only bridge between two already sealed experiment assets:

* ``contact_multichart_right.npz`` supplies the complete retained right-leg
  skin as exact intrinsic surface chart addresses;
* ``right_template_prior_chart_addresses.npz`` supplies the authorized old
  template path and its original physical XYZ points; and
* the repaired shaft ROI tetra atlas is used only as a strict physical
  inverse for ``internal_xi``.  Legacy ``xi_skin``/old ``h`` labels are never
  copied into the repaired coordinate output.

The emitted core path is ordered as COM_FEM -> SUPFEMV -> POPV -> POST_TIB,
with 234 aligned rows.  Points outside the repaired atlas domain or outside
its actual tetra geometry remain ``internal_valid=False`` and have NaN
``internal_xi``; they are never projected or silently snapped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent
VALIDATION_ROOT = EXPERIMENT_ROOT.parent / "ab_validation_20261110" / "repair_20261110"
SURFACE_PATH = VALIDATION_ROOT / "harmonic_full_domain_followup" / "contact_multichart_right.npz"
TEMPLATE_PATH = VALIDATION_ROOT / "harmonic_full_domain_followup" / "right_template_prior_chart_addresses.npz"
ATLAS_PATH = VALIDATION_ROOT / "harmonic" / "shaft_roi_v2" / "atlas_right.npz"
INTERNAL_SOURCE_PATH = VALIDATION_ROOT / "servo" / "prepared_scene" / "prebake" / "right_thigh_patch_and_vessels.npz"
CORE_LABELS = ("R_COM_FEM_V", "R_SUPFEMV", "R_POPV", "R_POST_TIB_V")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _decode_surface_addresses(
    surface: Any,
    chart_ids: np.ndarray,
    coordinates_uv_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode intrinsic surface addresses with the sealed atlas data.

    This is the same piecewise-affine chart decode as ``SurfaceMaterialAtlas``
    but kept local so the experiment does not import or modify production
    code.  Every address must have at least one containing chart triangle;
    multiple matches must agree physically.
    """

    vertices = np.asarray(surface["vertices"], dtype=np.float64)
    faces = np.asarray(surface["faces"], dtype=np.int64)
    indptr = np.asarray(surface["chart_indptr"], dtype=np.int64)
    chart_faces = np.asarray(surface["chart_faces"], dtype=np.int64)
    chart_uv = np.asarray(surface["chart_uv"], dtype=np.float64)
    out = np.empty((len(chart_ids), 3), dtype=np.float64)
    face_ids = np.full(len(chart_ids), -1, dtype=np.int64)
    barycentric = np.full((len(chart_ids), 3), np.nan, dtype=np.float64)
    for i, (chart_id_raw, coordinate) in enumerate(zip(chart_ids, coordinates_uv_m, strict=True)):
        chart_id = int(chart_id_raw)
        if chart_id < 0 or chart_id + 1 >= indptr.size:
            raise ValueError(f"surface chart id out of range at row {i}: {chart_id}")
        first, last = int(indptr[chart_id]), int(indptr[chart_id + 1])
        rows = np.arange(first, last, dtype=np.int64)
        triangles = chart_uv[rows]
        matrix = np.transpose(triangles[:, 1:] - triangles[:, :1], (0, 2, 1))
        rhs = np.asarray(coordinate, dtype=np.float64).reshape(1, 2) - triangles[:, 0]
        determinant = matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
        good_matrix = np.abs(determinant) > 1.0e-14
        local = np.full((len(rows), 2), np.nan, dtype=np.float64)
        local[good_matrix, 0] = (
            rhs[good_matrix, 0] * matrix[good_matrix, 1, 1]
            - matrix[good_matrix, 0, 1] * rhs[good_matrix, 1]
        ) / determinant[good_matrix]
        local[good_matrix, 1] = (
            matrix[good_matrix, 0, 0] * rhs[good_matrix, 1]
            - rhs[good_matrix, 0] * matrix[good_matrix, 1, 0]
        ) / determinant[good_matrix]
        bary = np.concatenate([1.0 - local.sum(axis=1, keepdims=True), local], axis=1)
        valid = good_matrix & np.all(bary >= -1.0e-10, axis=1) & np.all(bary <= 1.0 + 1.0e-10, axis=1)
        matches = np.flatnonzero(valid)
        if not len(matches):
            raise ValueError(f"surface address does not decode at row {i}, chart={chart_id}")
        physical = []
        for match in matches:
            face = int(chart_faces[rows[match]])
            physical.append((bary[match] @ vertices[faces[face]], face, bary[match]))
        if physical[1:] and max(float(np.linalg.norm(item[0] - physical[0][0])) for item in physical[1:]) > 1.0e-9:
            raise ValueError(f"ambiguous intrinsic surface address at row {i}, chart={chart_id}")
        out[i], face_ids[i], barycentric[i] = physical[0]
    return out, face_ids, barycentric


def _strict_physical_inverse(
    atlas: Any,
    points_m: np.ndarray,
    *,
    candidate_k: int = 256,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Invert original XYZ through actual repaired atlas tetrahedra.

    Returns ``(valid, xi, residual_m, tetra_index)``.  No nearest point is
    emitted for an outside query.  The theta interpolation unwraps each
    containing tetra around its first vertex before restoring ``[0, 2pi)``.
    """

    from scipy.spatial import cKDTree

    vertices = np.asarray(atlas["harmonic_vertices"], dtype=np.float64)
    tets = np.asarray(atlas["harmonic_tets"], dtype=np.int64)
    theta = np.asarray(atlas["harmonic_theta"], dtype=np.float64)
    h = np.asarray(atlas["harmonic_h"], dtype=np.float64)
    d = np.asarray(atlas["harmonic_d"], dtype=np.float64)
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    centroid_tree = cKDTree(vertices[tets].mean(axis=1))
    _, candidates = centroid_tree.query(points, k=min(int(candidate_k), len(tets)))
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    valid_out = np.zeros(len(points), dtype=bool)
    xi_out = np.full((len(points), 3), np.nan, dtype=np.float64)
    residual_out = np.full(len(points), np.nan, dtype=np.float64)
    tet_out = np.full(len(points), -1, dtype=np.int64)
    for start in range(0, len(points), int(batch_size)):
        stop = min(start + int(batch_size), len(points))
        query = points[start:stop]
        selected = candidates[start:stop]
        tet_vertices = vertices[tets[selected]]
        origin = tet_vertices[:, :, 0, :]
        matrix = np.transpose(tet_vertices[:, :, 1:, :] - origin[:, :, None, :], (0, 1, 3, 2))
        rhs = (query[:, None, :] - origin)[..., None]
        local = np.linalg.solve(matrix, rhs)[..., 0]
        bary = np.concatenate([1.0 - local.sum(axis=2, keepdims=True), local], axis=2)
        inside = np.all(bary >= -1.0e-7, axis=2) & np.all(bary <= 1.0 + 1.0e-7, axis=2)
        score = np.where(inside, np.min(bary, axis=2), -np.inf)
        chosen = np.argmax(score, axis=1)
        good = np.isfinite(score[np.arange(stop - start), chosen])
        for row in np.flatnonzero(good):
            candidate_slot = int(chosen[row])
            weights = np.clip(np.asarray(bary[row, candidate_slot], dtype=np.float64), 0.0, 1.0)
            weights /= float(weights.sum())
            tet_index = int(selected[row, candidate_slot])
            tet = tets[tet_index]
            raw_theta = theta[tet]
            reference = float(raw_theta[0])
            unwrapped_theta = reference + ((raw_theta - reference + np.pi) % (2.0 * np.pi) - np.pi)
            valid_out[start + row] = True
            tet_out[start + row] = tet_index
            xi_out[start + row] = np.asarray(
                [np.mod(weights @ unwrapped_theta, 2.0 * np.pi), weights @ h[tet], weights @ d[tet]],
                dtype=np.float64,
            )
            residual_out[start + row] = float(np.linalg.norm(weights @ vertices[tet] - query[row]))
    return valid_out, xi_out, residual_out, tet_out


def _ordered_core_rows(template: Any) -> np.ndarray:
    labels = np.asarray(template["labels"]).astype(str)
    rows = np.concatenate([np.flatnonzero(labels == label) for label in CORE_LABELS]).astype(np.int64)
    if len(rows) != 234:
        raise ValueError(f"expected 234 ordered core rows, got {len(rows)}")
    if any(np.any(np.diff(np.asarray(template["source_global_point_indices"])[rows_for_label]) < 0) for label in CORE_LABELS for rows_for_label in [np.flatnonzero(labels == label)]):
        raise ValueError("template rows within a vessel label are not source-ordered")
    return rows


def _contiguous_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(np.asarray(valid, dtype=bool))
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index != previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def build_assets(
    *,
    output_path: Path,
    audit_path: Path,
    surface_path: Path = SURFACE_PATH,
    template_path: Path = TEMPLATE_PATH,
    atlas_path: Path = ATLAS_PATH,
    internal_source_path: Path = INTERNAL_SOURCE_PATH,
) -> dict[str, Any]:
    for path in (surface_path, template_path, atlas_path, internal_source_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    surface = np.load(surface_path, allow_pickle=False)
    template = np.load(template_path, allow_pickle=False)
    atlas = np.load(atlas_path, allow_pickle=False)
    # Load the baked internal source only to establish provenance and its
    # independent chain length; output coordinates come from template original
    # XYZ rows and repaired physical inversion below.
    internal_source = np.load(internal_source_path, allow_pickle=True)
    rows = _ordered_core_rows(template)
    labels = np.asarray(template["labels"])[rows].astype(str)
    vessel_m = np.asarray(template["source_original_vessel_points"], dtype=np.float64)[rows]
    chart_ids = np.asarray(template["chart_ids"], dtype=np.int64)[rows]
    coordinates_uv_m = np.asarray(template["coordinates_uv_m"], dtype=np.float64)[rows]
    skin_m, surface_face_ids, surface_barycentric = _decode_surface_addresses(surface, chart_ids, coordinates_uv_m)
    prior_skin_m = np.asarray(template["canonical_reprojected_points"], dtype=np.float64)[rows]
    prior_surface_decode_error_m = np.linalg.norm(skin_m - prior_skin_m, axis=1)
    valid, internal_xi, internal_residual_m, internal_tet_index = _strict_physical_inverse(atlas, vessel_m)
    runs = _contiguous_runs(valid)
    source_global_indices = np.asarray(template["source_global_point_indices"], dtype=np.int64)[rows]
    old_legacy_xi = np.asarray(template["legacy_xi"], dtype=np.float64)[rows]
    vessel_step_m = np.linalg.norm(np.diff(vessel_m, axis=0), axis=1)
    skin_step_m = np.linalg.norm(np.diff(skin_m, axis=0), axis=1)
    run_rows: list[dict[str, Any]] = []
    for start, end in runs:
        run_rows.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "point_count": int(end - start + 1),
                "label_start": str(labels[start]),
                "label_end": str(labels[end]),
                "source_global_index_start": int(source_global_indices[start]),
                "source_global_index_end": int(source_global_indices[end]),
                "internal_length_m": float(np.sum(vessel_step_m[start:end])),
                "internal_length_mm": float(np.sum(vessel_step_m[start:end]) * 1000.0),
                "skin_length_m": float(np.sum(skin_step_m[start:end])),
                "skin_length_mm": float(np.sum(skin_step_m[start:end]) * 1000.0),
                "repaired_h_start": float(internal_xi[start, 1]),
                "repaired_h_end": float(internal_xi[end, 1]),
                "repaired_d_min": float(np.nanmin(internal_xi[start : end + 1, 2])),
                "repaired_d_max": float(np.nanmax(internal_xi[start : end + 1, 2])),
            }
        )
    source_meta: dict[str, Any] = {
        "surface_asset": {"path": str(surface_path.resolve()), "sha256": _sha256(surface_path)},
        "template_prior_asset": {"path": str(template_path.resolve()), "sha256": _sha256(template_path)},
        "repaired_atlas_asset": {"path": str(atlas_path.resolve()), "sha256": _sha256(atlas_path)},
        "baked_internal_source_asset": {"path": str(internal_source_path.resolve()), "sha256": _sha256(internal_source_path)},
        "legacy_vessel_material_asset": {
            "path": "/media/camp/EXT_DRIVE/Among_US/dataset/processed/anatomy_retarget/leg_volume_coordinates/production/vessels/vessel_material_coordinates.npz",
            "role": "upstream source provenance only; its old xi_skin labels were not used",
        },
    }
    metadata = _jsonable(
        {
            "schema": "long_leg_assets_v1",
            "status": "bounded_repaired_internal_attachment",
            "ordering": list(CORE_LABELS),
            "row_count": int(len(rows)),
            "surface_contract": "full retained right-leg contact multichart surface, d=0 only",
            "internal_contract": "strict containing-tetra inverse on repaired atlas_right; no nearest projection",
            "repaired_atlas_domain": {
                "h": [float(np.min(atlas["harmonic_h"])), float(np.max(atlas["harmonic_h"]))],
                "theta": "periodic radians",
                "d": [float(np.min(atlas["harmonic_d"])), float(np.max(atlas["harmonic_d"]))],
            },
            "old_xi_warning": "legacy_xi is retained only as provenance/debug; do not use it as repaired internal coordinates",
            "source_assets": source_meta,
            "source_internal_bake_summary": {
                "baked_trunk_label": str(np.asarray(internal_source["trunk_label"]).item()),
                "baked_trunk_points": int(len(internal_source["trunk_vessel_points"])),
                "baked_internal_field_keys": ["vessel_points", "vessel_surface_points", "vessel_theta", "vessel_h"],
            },
            "decode_checks": {
                "surface_address_decode_max_error_m": float(np.max(prior_surface_decode_error_m)),
                "surface_address_decode_median_error_m": float(np.median(prior_surface_decode_error_m)),
                "strict_internal_inverse_residual_max_m": float(np.nanmax(internal_residual_m)),
                "strict_internal_valid_count": int(np.sum(valid)),
                "strict_internal_invalid_count": int(np.sum(~valid)),
            },
            "contiguous_supported_runs": run_rows,
            "full_ordered_path_length_m": float(np.sum(vessel_step_m)),
            "full_ordered_path_length_mm": float(np.sum(vessel_step_m) * 1000.0),
            "full_ordered_skin_length_m": float(np.sum(skin_step_m)),
            "full_ordered_skin_length_mm": float(np.sum(skin_step_m) * 1000.0),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        vessel_m=vessel_m,
        skin_m=skin_m,
        labels=labels,
        chart_ids=chart_ids,
        coordinates_uv_m=coordinates_uv_m,
        internal_valid=valid,
        internal_xi=internal_xi,
        internal_residual_m=internal_residual_m,
        internal_tet_index=internal_tet_index,
        source_global_point_indices=source_global_indices,
        surface_face_ids=surface_face_ids,
        surface_barycentric=surface_barycentric,
        legacy_xi=old_legacy_xi,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "LONG_LEG_ASSETS.npz")
    parser.add_argument("--audit-output", type=Path, default=EXPERIMENT_ROOT / "LONG_LEG_ASSETS.audit.json")
    parser.add_argument("--surface", type=Path, default=SURFACE_PATH)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--atlas", type=Path, default=ATLAS_PATH)
    parser.add_argument("--internal-source", type=Path, default=INTERNAL_SOURCE_PATH)
    args = parser.parse_args(argv)
    metadata = build_assets(
        output_path=args.output,
        audit_path=args.audit_output,
        surface_path=args.surface,
        template_path=args.template,
        atlas_path=args.atlas,
        internal_source_path=args.internal_source,
    )
    print(json.dumps({"status": metadata["status"], "output": str(args.output), "audit": str(args.audit_output), "runs": metadata["contiguous_supported_runs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

