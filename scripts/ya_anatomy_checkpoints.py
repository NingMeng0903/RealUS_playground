#!/usr/bin/env python3
"""YA metrics for two-stage anatomy alignment (Hypothesis 1 + skeleton checks).

Measures soft-tissue containment against the subject SMPL-X surface, plus
cervical chain / head / clavicle / femur diagnostics.  Works on schema-3 and
schema-6 NPZ files (raw load; does not require validate=True).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_raw(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {name: data[name] for name in data.files}


def _as_str_list(arr: Any) -> list[str]:
    return [str(x) for x in np.asarray(arr).tolist()]


def _smplx_surface(canonical_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    sys.path.insert(0, str(_repo_root() / "src"))
    from projects.genesis_ue_sync.anatomy_retarget.containment import load_body_surface

    return load_body_surface(canonical_dir / "smpl_canonical_tpose.obj")


def _signed_distance(points: np.ndarray, sv: np.ndarray, sf: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(_repo_root() / "src"))
    from projects.genesis_ue_sync.anatomy_retarget.containment import signed_distance

    values, _closest, _normal = signed_distance(points, sv, sf)
    return np.asarray(values, dtype=np.float64)


def _tissue_report(
    sdf: np.ndarray,
    ranges: np.ndarray,
    tissues: list[str],
    meshes: list[str],
) -> dict[str, Any]:
    by_tissue: dict[str, dict[str, float | int]] = {}
    by_mesh: list[dict[str, Any]] = []
    for (start, stop), tissue, mesh in zip(ranges, tissues, meshes):
        local = sdf[int(start) : int(stop)]
        if local.size == 0:
            continue
        outside = int(np.count_nonzero(local > 0.0))
        entry = by_tissue.setdefault(
            tissue,
            {"vertex_count": 0, "outside_count": 0, "max_outside_m": 0.0},
        )
        entry["vertex_count"] = int(entry["vertex_count"]) + int(local.size)
        entry["outside_count"] = int(entry["outside_count"]) + outside
        entry["max_outside_m"] = max(
            float(entry["max_outside_m"]), float(np.max(local))
        )
        if outside:
            by_mesh.append(
                {
                    "mesh": mesh,
                    "tissue": tissue,
                    "outside_count": outside,
                    "vertex_count": int(local.size),
                    "outside_fraction": float(outside / local.size),
                    "max_outside_m": float(np.max(local)),
                }
            )
    for tissue, entry in by_tissue.items():
        n = max(int(entry["vertex_count"]), 1)
        entry["outside_fraction"] = float(int(entry["outside_count"]) / n)
    by_mesh.sort(key=lambda row: (-int(row["outside_count"]), -float(row["max_outside_m"])))
    soft_keys = ("vessel", "nerve", "organ")
    soft_out = sum(int(by_tissue.get(k, {}).get("outside_count", 0)) for k in soft_keys)
    soft_n = sum(int(by_tissue.get(k, {}).get("vertex_count", 0)) for k in soft_keys)
    soft_max = max(
        (float(by_tissue.get(k, {}).get("max_outside_m", 0.0)) for k in soft_keys),
        default=0.0,
    )
    return {
        "by_tissue": by_tissue,
        "top_outside_meshes": by_mesh[:25],
        "soft_vessel_nerve_organ": {
            "outside_count": int(soft_out),
            "vertex_count": int(soft_n),
            "outside_fraction": float(soft_out / max(soft_n, 1)),
            "max_outside_m": float(soft_max),
        },
    }


def _cervical_chain(vertices: np.ndarray, ranges: np.ndarray, meshes: list[str], tissues: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (start, stop), mesh, tissue in zip(ranges, meshes, tissues):
        if str(tissue).lower() != "bone":
            continue
        lower = str(mesh).lower()
        if not (
            lower in {f"c{i}" for i in range(1, 8)}
            or lower.startswith("c1")
            or lower.startswith("c2")
            or "atlas" in lower
            or "axis" in lower
            or (lower.startswith("c") and any(ch.isdigit() for ch in lower[:3]))
        ):
            if lower not in {f"c{i}" for i in range(1, 8)} and not any(
                lower.startswith(f"c{i}") for i in range(1, 8)
            ):
                continue
        pts = vertices[int(start) : int(stop)]
        rows.append(
            {
                "mesh": mesh,
                "count": int(stop - start),
                "center": pts.mean(axis=0).tolist(),
                "y_min": float(pts[:, 1].min()),
                "y_max": float(pts[:, 1].max()),
            }
        )
    rows.sort(key=lambda r: -float(r["y_min"]))
    gaps: list[dict[str, Any]] = []
    for a, b in zip(rows, rows[1:]):
        gap = float(a["y_min"]) - float(b["y_max"])
        gaps.append(
            {
                "from": a["mesh"],
                "to": b["mesh"],
                "gap_m": gap,
                "gap_mm": gap * 1000.0,
            }
        )
    return {"vertebrae": rows, "adjacent_gaps": gaps}


def _axial_mask_hit_table(meshes: list[str], tissues: list[str]) -> list[dict[str, Any]]:
    """Mirror the fast-merge vertebra selection bug/fix surface."""
    vertebra_exact = {*(f"c{i}" for i in range(1, 8))}
    rows = []
    for mesh, tissue in zip(meshes, tissues):
        if str(tissue).lower() != "bone":
            continue
        lower = str(mesh).lower()
        if not (
            lower in vertebra_exact
            or any(lower.startswith(f"c{i}") for i in range(1, 8))
            or "atlas" in lower
            or "axis" in lower
        ):
            continue
        exact = lower in vertebra_exact
        prefix = any(lower.startswith(f"c{i}") for i in range(1, 8))
        rows.append(
            {
                "mesh": mesh,
                "exact_cN_match": exact,
                "prefix_cN_match": prefix,
                "would_take_16fa_axial_with_exact_only": exact,
                "would_take_16fa_axial_with_prefix": prefix or exact,
            }
        )
    return rows


def _soft_residual(
    vertices: np.ndarray,
    reference: np.ndarray | None,
    ranges: np.ndarray,
    tissues: list[str],
) -> dict[str, Any] | None:
    if reference is None or reference.shape != vertices.shape:
        return None
    soft = np.zeros(len(vertices), dtype=bool)
    for (start, stop), tissue in zip(ranges, tissues):
        if str(tissue).lower() != "bone":
            soft[int(start) : int(stop)] = True
    if not np.any(soft):
        return None
    delta = np.linalg.norm(vertices[soft] - reference[soft], axis=1)
    return {
        "soft_vertex_count": int(np.count_nonzero(soft)),
        "soft_residual_mean_m": float(np.mean(delta)),
        "soft_residual_max_m": float(np.max(delta)),
        "soft_residual_p99_m": float(np.quantile(delta, 0.99)),
        "clamped_at_30mm": bool(float(np.max(delta)) <= 0.0300001 and float(np.max(delta)) >= 0.029),
    }


def _hypothesis1_gate(soft_summary: dict[str, Any]) -> dict[str, Any]:
    """Gate: most vessel/nerve/organ vertices inside SMPL-X.

    Allow small distal leaks: fail if soft outside fraction > 5% OR max > 25 mm.
    """
    frac = float(soft_summary.get("outside_fraction", 1.0))
    max_m = float(soft_summary.get("max_outside_m", 1.0))
    passed = frac <= 0.05 and max_m <= 0.025
    return {
        "name": "hypothesis1_smplx_harmonic_containment",
        "passed": bool(passed),
        "outside_fraction": frac,
        "max_outside_m": max_m,
        "thresholds": {"outside_fraction_max": 0.05, "max_outside_m": 0.025},
        "reason": (
            "ok"
            if passed
            else (
                f"soft outside_fraction={frac:.4f} (limit 0.05) "
                f"max_outside_m={max_m:.4f} (limit 0.025)"
            )
        ),
    }


def _lower_limb_report(
    vertices: np.ndarray,
    sdf: np.ndarray,
    ranges: np.ndarray,
    tissues: list[str],
    canonical_dir: Path,
) -> dict[str, Any]:
    """Measure containment in the four thigh/calf work regions."""
    skeleton = json.loads((canonical_dir / "smpl_canonical_skeleton.json").read_text())
    joints = np.asarray(skeleton["rest_joints_subject"], dtype=np.float64)
    soft = np.zeros(len(vertices), dtype=bool)
    for (start, stop), tissue in zip(ranges, tissues):
        if str(tissue).lower() in {"vessel", "nerve", "organ"}:
            soft[int(start) : int(stop)] = True
    output: dict[str, Any] = {}
    for side, hip, knee, ankle in (("left", 1, 4, 7), ("right", 2, 5, 8)):
        hip_point, knee_point, ankle_point = joints[[hip, knee, ankle]]
        axis = hip_point - ankle_point
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        projection = (vertices - ankle_point) @ axis
        radial = np.linalg.norm(
            vertices - (ankle_point + projection[:, None] * axis), axis=1
        )
        knee_t = float(np.dot(knee_point - ankle_point, axis))
        hip_t = float(np.dot(hip_point - ankle_point, axis))
        for part, lower, upper in (
            ("thigh", knee_t - 0.04, hip_t + 0.04),
            ("calf", -0.04, knee_t + 0.04),
        ):
            mask = soft & (projection >= lower) & (projection <= upper) & (radial < 0.19)
            values = sdf[mask]
            outside = int(np.count_nonzero(values > 0.0))
            output[f"{side}_{part}"] = {
                "vertex_count": int(len(values)),
                "outside_count": outside,
                "outside_fraction": float(outside / max(len(values), 1)),
                "max_outside_m": float(max(0.0, float(np.max(values)))) if len(values) else 0.0,
                "p99_m": float(np.quantile(values, 0.99)) if len(values) else 0.0,
            }
    return output


def analyze_asset(
    asset_path: Path,
    *,
    canonical_dir: Path,
    label: str | None = None,
) -> dict[str, Any]:
    raw = _load_raw(asset_path)
    vertices = np.asarray(raw["vertices_rest"], dtype=np.float64)
    ranges = np.asarray(raw["source_vertex_ranges"], dtype=np.int64)
    tissues = _as_str_list(raw["source_tissues"])
    meshes = _as_str_list(raw["source_mesh_names"])
    schema = int(np.asarray(raw["schema_version"]).reshape(-1)[0]) if "schema_version" in raw else 0
    meta = raw["metadata"].item() if "metadata" in raw else {}
    if not isinstance(meta, dict):
        meta = {}

    sv, sf = _smplx_surface(canonical_dir)
    sdf = _signed_distance(vertices, sv, sf)
    containment = _tissue_report(sdf, ranges, tissues, meshes)
    gate = _hypothesis1_gate(containment["soft_vessel_nerve_organ"])

    reference = None
    if "harmonic_reference_vertices" in raw and np.asarray(raw["harmonic_reference_vertices"]).size:
        reference = np.asarray(raw["harmonic_reference_vertices"], dtype=np.float64).reshape(-1, 3)

    cage_report = None
    for key in ("source_skin_volume_report", "registration_report", "shape_report"):
        blob = meta.get(key)
        if isinstance(blob, dict):
            if "outside_query_count" in blob:
                cage_report = {
                    "source": key,
                    "outside_query_count": blob.get("outside_query_count"),
                    "outside_soft_material_count": blob.get("outside_soft_material_count"),
                    "minimum_jacobian_ratio": blob.get("minimum_jacobian_ratio"),
                }
                break
            nested = blob.get("source_skin_volume") if key == "registration_report" else None
            if isinstance(nested, dict) and "outside_query_count" in nested:
                cage_report = {
                    "source": f"{key}.source_skin_volume",
                    "outside_query_count": nested.get("outside_query_count"),
                    "outside_soft_material_count": nested.get("outside_soft_material_count"),
                    "minimum_jacobian_ratio": nested.get("minimum_jacobian_ratio"),
                }
                break

    return {
        "label": label or asset_path.name,
        "asset_path": str(asset_path.resolve()),
        "schema_version": schema,
        "shape_hash": meta.get("shape_hash"),
        "canonical_dir": str(canonical_dir.resolve()),
        "smplx_containment": containment,
        "hypothesis1_gate": gate,
        "lower_limb_containment": _lower_limb_report(
            vertices, sdf, ranges, tissues, canonical_dir
        ),
        "skin_glass_or_cage_report_from_metadata": cage_report,
        "cervical_chain": _cervical_chain(vertices, ranges, meshes, tissues),
        "axial_mask_hit_table": _axial_mask_hit_table(meshes, tissues),
        "soft_residual_vs_harmonic_reference": _soft_residual(
            vertices, reference, ranges, tissues
        ),
        "note": (
            "Skin_Glass cage outside_query_count=0 does not imply SMPL-X containment; "
            "hypothesis1_gate uses subject smpl_canonical_tpose.obj SDF."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", type=Path, action="append", default=[], help="anatomy_rigged.npz (repeatable)")
    ap.add_argument("--label", type=str, action="append", default=[], help="optional label per --asset")
    ap.add_argument(
        "--canonical-dir",
        type=Path,
        default=_repo_root()
        / "outputs/anatomy_retarget/canonical_cache/34deaeada36cdc4a505d",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=_repo_root() / "outputs/anatomy_retarget/ya_hypothesis1_report.json",
    )
    args = ap.parse_args()
    if not args.asset:
        raise SystemExit("pass at least one --asset")
    labels = list(args.label)
    while len(labels) < len(args.asset):
        labels.append(None)

    reports = []
    for path, label in zip(args.asset, labels):
        print(f"analyzing {path} ...", flush=True)
        report = analyze_asset(path, canonical_dir=args.canonical_dir, label=label)
        soft = report["smplx_containment"]["soft_vessel_nerve_organ"]
        gate = report["hypothesis1_gate"]
        print(
            f"  [{report['label']}] soft outside={soft['outside_count']}/{soft['vertex_count']} "
            f"({100 * soft['outside_fraction']:.2f}%) max_mm={soft['max_outside_m'] * 1000:.1f} "
            f"gate={'PASS' if gate['passed'] else 'FAIL'} — {gate['reason']}",
            flush=True,
        )
        reports.append(report)

    payload = {
        "canonical_dir": str(args.canonical_dir.resolve()),
        "assets": reports,
        "any_hypothesis1_pass": any(r["hypothesis1_gate"]["passed"] for r in reports),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}", flush=True)
    return 0 if payload["any_hypothesis1_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
