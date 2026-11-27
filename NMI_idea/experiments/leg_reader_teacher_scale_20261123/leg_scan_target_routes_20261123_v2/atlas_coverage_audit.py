#!/usr/bin/env python3
"""Audit the named deep-venous route in the *training* atlas package.

This is intentionally a read-only evidence audit.  It only opens
``ct_reference_topology.json.gz`` and ``ct_geometry.npz`` (the four training
sources and their coarse role arrays); it never opens a Blender asset.  The
topology file uses positive axial station coordinates (480..1280 mm), while
the scan package uses neutral SMPL-X coordinates (-480..-1280 mm).  The
conversion used in this report is therefore ``neutral_mm = -station_mm``.

The audit keeps two kinds of evidence separate:

* the exact named-route graph (native labels, source junctions and measured
  same-source inferred bridges); and
* the coarse ``ct_geometry`` role rows used by the historical scanner.  A
  non-empty role-0 row is not treated as proof that all four deep-vein names
  are present or joined.

Usage from the handoff root::

    python leg_scan_target_routes_20261123_v2/atlas_coverage_audit.py

The resulting JSON and CSV are written below ``--out-dir``.  Both inputs are
read only and are hashed in the JSON manifest.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROUTE = (
    "common femoral vein",
    "femoral vein",
    "popliteal vein",
    "posterior tibial vein",
)

# These are the explicitly named incidence relationships in the legacy
# scanner's anatomy graph.  The graph deliberately does not create a fibular
# continuation edge.
BRANCHES = {
    "great saphenous vein": {
        "incidence": "saphenofemoral",
        "legacy_relation": "joins deep_target at saphenofemoral junction",
    },
    "deep femoral vein": {
        "incidence": "proximal thigh",
        "legacy_relation": "joins deep_target in proximal thigh",
    },
    "fibular vein": {
        "incidence": None,
        "legacy_relation": "separate calf vessel; no new edge emitted",
    },
}

ROLE_LABELS = (
    "deep_target (coarse role; four named route members)",
    "arterial_competitor",
    "great_saphenous",
    "deep_femoral_or_fibular",
    "small_saphenous_or_other",
    "unmodeled_or_source_specific",
)

SOURCE_ORDER = (
    "BodyParts3D right",
    "BodyParts3D mirrored left",
    "Open3DModel right",
    "MedSeg",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_name(value: str) -> str:
    """Normalise source labels without conflating different vessels."""

    text = str(value).strip().lower().replace("_", " ")
    # Open3DModel uses a trailing ``.r`` and MedSeg uses ``NN Right ``.
    text = re.sub(r"\.r\b", "", text)
    text = re.sub(r"^\s*\d+\s+right\s+", "", text)
    text = re.sub(r"\s+right\s+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars/arrays for deterministic JSON output."""

    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _runs(values: Iterable[int], *, step: int = 1) -> list[dict[str, int]]:
    """Return contiguous runs from a sorted integer station sequence."""

    vals = sorted({int(v) for v in values})
    if not vals:
        return []
    out: list[dict[str, int]] = []
    start = previous = vals[0]
    for current in vals[1:]:
        if current - previous > step:
            out.append(
                {
                    "start": start,
                    "end": previous,
                    "count": ((previous - start) // step) + 1,
                }
            )
            start = current
        previous = current
    out.append(
        {
            "start": start,
            "end": previous,
            "count": ((previous - start) // step) + 1,
        }
    )
    return out


def _missing_gaps(values: Iterable[int], *, step: int = 1) -> list[dict[str, int]]:
    vals = sorted({int(v) for v in values})
    gaps: list[dict[str, int]] = []
    for left, right in zip(vals, vals[1:]):
        missing = ((right - left) // step) - 1
        if missing > 0:
            gaps.append(
                {
                    "from_station_mm": left,
                    "to_station_mm": right,
                    "axial_gap_mm": right - left,
                    "missing_integer_stations": missing,
                }
            )
    return gaps


def _node_summary(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    stations = sorted({int(round(float(node["station_mm"]))) for node in nodes})
    return {
        "node_count": len(nodes),
        "station_count": len(stations),
        "station_min_mm": min(stations) if stations else None,
        "station_max_mm": max(stations) if stations else None,
        "neutral_min_mm": -max(stations) if stations else None,
        "neutral_max_mm": -min(stations) if stations else None,
        "station_runs": _runs(stations),
        "internal_station_gaps": _missing_gaps(stations),
        "provenance_counts": dict(Counter(str(n.get("provenance", "")) for n in nodes)),
    }


def _canonical_edges(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Add exact normalized endpoint names to each two-node edge."""

    nodes = {int(node["id"]): node for node in source.get("nodes", [])}
    records: list[dict[str, Any]] = []
    for edge in source.get("edges", []):
        endpoint_ids = edge.get("nodes", [])
        if len(endpoint_ids) < 2:
            continue
        left = nodes.get(int(endpoint_ids[0]))
        right = nodes.get(int(endpoint_ids[1]))
        if left is None or right is None:
            continue
        records.append(
            {
                "edge": edge,
                "left": left,
                "right": right,
                "left_name": normalise_name(left["name"]),
                "right_name": normalise_name(right["name"]),
            }
        )
    return records


def _edge_evidence(edge_record: Mapping[str, Any]) -> dict[str, Any]:
    edge = edge_record["edge"]
    left = edge_record["left"]
    right = edge_record["right"]
    evidence = edge.get("evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    # Keep the measured fields and avoid copying node payloads into every row.
    keep = (
        "from_mm",
        "to_mm",
        "station_extent_delta_mm",
        "endpoint_distance_mm",
        "axial_gap_mm",
        "two_sided_sample_span_mm",
        "dense_probe_step_mm",
        "probes",
        "supported_probes",
        "source_support_types",
        "max_blank_run_mm",
        "native_shared_voxel_faces",
        "local_vertex_pairs_le_0p8mm",
        "min_local_surface_vertex_gap_mm",
        "lateral_residual_mm",
        "status",
    )
    result = {
        "left": edge_record["left_name"],
        "right": edge_record["right_name"],
        "provenance": str(edge.get("provenance", "")),
        "node_ids": [int(left["id"]), int(right["id"])],
        "station_mm": [int(round(float(left["station_mm"]))), int(round(float(right["station_mm"])))],
    }
    result.update({key: _jsonable(evidence[key]) for key in keep if key in evidence})
    return result


def _matching_candidate_gaps(
    source: Mapping[str, Any], left_name: str, right_name: str
) -> list[dict[str, Any]]:
    """Return declared unsupported/long-gap evidence for one exact pair."""

    matches: list[dict[str, Any]] = []
    for candidate in source.get("gaps_and_join_candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        left = normalise_name(candidate.get("left", ""))
        right = normalise_name(candidate.get("right", ""))
        pair = {left, right}
        if pair == {left_name, right_name} and left and right:
            matches.append(dict(candidate))
    return matches


def _same_name_edges(source: Mapping[str, Any], canonical: str) -> list[dict[str, Any]]:
    return [
        _edge_evidence(record)
        for record in _canonical_edges(source)
        if record["left_name"] == canonical and record["right_name"] == canonical
    ]


def _transition_edges(
    source: Mapping[str, Any], left_name: str, right_name: str
) -> list[dict[str, Any]]:
    target = {left_name, right_name}
    return [
        _edge_evidence(record)
        for record in _canonical_edges(source)
        if {record["left_name"], record["right_name"]} == target
    ]


def _cross_role_branch_edges(
    source: Mapping[str, Any], branch: str, main_route: Iterable[str]
) -> list[dict[str, Any]]:
    names = set(main_route)
    return [
        _edge_evidence(record)
        for record in _canonical_edges(source)
        if (record["left_name"] == branch and record["right_name"] in names)
        or (record["right_name"] == branch and record["left_name"] in names)
    ]


def _source_graph_audit(source_name: str, source: Mapping[str, Any]) -> dict[str, Any]:
    nodes = source.get("nodes", [])
    by_name: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_name.setdefault(normalise_name(node["name"]), []).append(node)

    named_segments: dict[str, Any] = {}
    for name in ROUTE + tuple(BRANCHES):
        segment_nodes = by_name.get(name, [])
        summary = _node_summary(segment_nodes)
        summary["canonical_name"] = name
        summary["role_declared"] = any(
            name == normalise_name(identity)
            for role in source.get("role_identities", [])
            for identity in (role if isinstance(role, list) else [role])
        )
        summary["same_name_inferred_edges"] = [
            item
            for item in _same_name_edges(source, name)
            if item.get("provenance") in {"inferred", "inferred_junction"}
        ]
        summary["same_name_inferred_edge_count"] = len(summary["same_name_inferred_edges"])
        named_segments[name] = summary

    transitions: list[dict[str, Any]] = []
    for left_name, right_name in zip(ROUTE, ROUTE[1:]):
        records = _transition_edges(source, left_name, right_name)
        provenance_counts = dict(Counter(str(record["provenance"]) for record in records))
        transition = {
            "from": left_name,
            "to": right_name,
            "edge_count": len(records),
            "provenance_counts": provenance_counts,
            "edges": records,
            "candidate_gap_or_join_records": []
            if records
            else _jsonable(_matching_candidate_gaps(source, left_name, right_name)),
            "status": "supported_edge" if records else "unsupported_transition",
        }
        # A transition edge can be geometrically measured but still not native
        # label contact.  Keep the distinction explicit for the parent audit.
        transition["has_native_source_junction"] = provenance_counts.get("source_junction", 0) > 0
        transition["has_measured_inferred_bridge"] = any(
            record["provenance"] in {"inferred", "inferred_junction"} for record in records
        )
        transitions.append(transition)

    exact_nodes_present = all(named_segments[name]["node_count"] > 0 for name in ROUTE)
    exact_segments_contiguous = all(
        not named_segments[name]["internal_station_gaps"] for name in ROUTE
    )
    all_transitions_supported = all(item["edge_count"] > 0 for item in transitions)
    all_transitions_native = all(
        item["edge_count"] > 0
        and set(item["provenance_counts"]).issubset({"sampled", "source_junction"})
        for item in transitions
    )
    inferred_count = sum(item["has_measured_inferred_bridge"] for item in transitions)

    # The historical role row names this as a four-member deep target even
    # when a source only has a combined/coarse row.  ``femoral vein`` alone is
    # not silently promoted to a common femoral label.
    coarse_alias = not exact_nodes_present and (
        named_segments["common femoral vein"]["node_count"] == 0
        and named_segments["femoral vein"]["node_count"] > 0
    )
    if exact_nodes_present and exact_segments_contiguous and all_transitions_native:
        support_class = "exact_full_named_route_native_edges"
    elif exact_nodes_present and exact_segments_contiguous and all_transitions_supported:
        support_class = "exact_full_named_route_with_declared_inferred_edges"
    elif coarse_alias and all(
        item["edge_count"] > 0 for item in transitions[1:]
    ):
        support_class = (
            "femoral_to_posterior_named_route_only; common_femoral_is_coarse_role_alias"
        )
    elif source_name == "MedSeg" and transitions[0]["has_native_source_junction"]:
        support_class = "native_common_to_femoral_only; distal_named_route_not_present"
    else:
        support_class = "partial_named_segments_and_or_unsupported_transitions"
    return {
        "source_name": source_name,
        "source_file_label": source.get("source"),
        "node_count": len(nodes),
        "edge_count": len(source.get("edges", [])),
        "role_identities": _jsonable(source.get("role_identities", [])),
        "named_segments": named_segments,
        "route_transitions": transitions,
        "route_summary": {
            "exact_named_label_coverage_complete": bool(exact_nodes_present),
            "exact_segments_have_no_internal_1mm_gaps": bool(exact_segments_contiguous),
            "full_named_route_supported_by_declared_edges": bool(
                exact_nodes_present and exact_segments_contiguous and all_transitions_supported
            ),
            "full_named_route_native_only": bool(
                exact_nodes_present and exact_segments_contiguous and all_transitions_native
            ),
            "coarse_role_geometry_only_for_missing_names": bool(coarse_alias),
            "support_class": support_class,
            "measured_same_source_inferred_bridge_count": int(inferred_count),
            "unsupported_route_transitions": [
                f"{item['from']} -> {item['to']}"
                for item in transitions
                if item["edge_count"] == 0
            ],
        },
        "branch_audit": {},
        "declared_unsupported_or_preserved_gaps": _jsonable(
            source.get("gaps_and_join_candidates", [])
        ),
    }


def _add_branch_audit(source: Mapping[str, Any], graph_audit: dict[str, Any]) -> None:
    result: dict[str, Any] = {}
    for branch, declaration in BRANCHES.items():
        segment = graph_audit["named_segments"][branch]
        cross_edges = _cross_role_branch_edges(source, branch, ROUTE)
        branch_edges = segment["same_name_inferred_edges"]
        row = {
            "canonical_name": branch,
            "legacy_incidence": declaration["incidence"],
            "legacy_relation": declaration["legacy_relation"],
            "named_segment": segment,
            "cross_role_edges_to_deep_route": cross_edges,
            "cross_role_incidence_status": (
                "measured_graph_edge" if cross_edges else "no_cross_role_edge_in_training_graph"
            ),
            "same_source_inferred_bridge_count": len(branch_edges),
            "same_source_inferred_bridges": branch_edges,
            "declared_same_identity_bridge_candidates": _jsonable(
                [
                    candidate
                    for candidate in source.get("gaps_and_join_candidates", [])
                    if normalise_name(candidate.get("name", "")) == branch
                    and candidate.get("status") == "inferred_same_identity_bridge"
                ]
            ),
        }
        if branch == "fibular vein":
            row["must_not_be_promoted_to_deep_route"] = True
        result[branch] = row
    graph_audit["branch_audit"] = result


def _bool_runs(mask: np.ndarray) -> list[dict[str, int]]:
    return _runs(np.flatnonzero(mask).tolist(), step=1)


def _coarse_role_audit(
    geometry: Mapping[str, np.ndarray], source_index: int, source_name: str, shape_index: int
) -> dict[str, Any]:
    centers = np.asarray(geometry["centers"])
    status = np.asarray(geometry["status"])
    links = np.asarray(geometry["links"])
    edge_status = np.asarray(geometry["edge_status"])
    neutral_grid = np.asarray(geometry["neutral_grid"], dtype=float)

    # ct_geometry uses shape-major centers and source-major status/links.
    source_centers = centers[shape_index, source_index]
    rows: list[dict[str, Any]] = []
    for role_index, role_label in enumerate(ROLE_LABELS):
        finite = np.isfinite(source_centers[:, role_index]).all(axis=1)
        present_indices = np.flatnonzero(finite)
        link_row = links[source_index, :, role_index].astype(bool)
        edge_row = edge_status[source_index, :, role_index]
        link_true_indices = np.flatnonzero(link_row)
        row: dict[str, Any] = {
            "role_index": role_index,
            "role_label": role_label,
            "support_count": int(finite.sum()),
            "station_count": int(finite.sum()),
            "neutral_station_min_mm": float(neutral_grid[present_indices[-1]])
            if len(present_indices)
            else None,
            "neutral_station_max_mm": float(neutral_grid[present_indices[0]])
            if len(present_indices)
            else None,
            "positive_station_min_mm": float(-neutral_grid[present_indices[0]])
            if len(present_indices)
            else None,
            "positive_station_max_mm": float(-neutral_grid[present_indices[-1]])
            if len(present_indices)
            else None,
            "finite_station_runs": [
                {
                    "neutral_start_mm": float(neutral_grid[run["start"]]),
                    "neutral_end_mm": float(neutral_grid[run["end"]]),
                    "count": run["count"],
                }
                for run in _bool_runs(finite)
            ],
            "missing_station_runs_between_support": [
                {
                    "neutral_start_mm": float(neutral_grid[run["start"]]),
                    "neutral_end_mm": float(neutral_grid[run["end"]]),
                    "count": run["count"],
                }
                for run in _bool_runs(~finite)
                if len(present_indices)
                and run["start"] > int(present_indices[0])
                and run["end"] < int(present_indices[-1])
            ],
            "link_count": int(link_row.sum()),
            "link_false_count": int((~link_row).sum()),
            "link_true_neutral_runs": [
                {
                    "neutral_start_mm": float(neutral_grid[run["start"]]),
                    "neutral_end_mm": float(neutral_grid[run["end"] + 1]),
                    "count": run["count"],
                }
                for run in _bool_runs(link_row)
            ],
            "edge_status_counts": {
                str(int(key)): int(value)
                for key, value in Counter(edge_row.tolist()).items()
            },
            "shape_index": int(shape_index),
            "source_index": int(source_index),
        }
        rows.append(row)
    return {
        "source_name": source_name,
        "source_index": int(source_index),
        "shape_index": int(shape_index),
        "role_rows": rows,
        "interpretation": (
            "These are coarse role rows from ct_geometry, not proof of exact named-route "
            "coverage or named transitions."
        ),
    }


def _load_topology(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("sources"), Mapping):
        raise ValueError("ct_reference_topology.json.gz has no source mapping")
    return data


def _load_geometry(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def run_audit(topology_path: Path, geometry_path: Path, out_dir: Path, shape_index: int = 0) -> dict[str, Any]:
    topology = _load_topology(topology_path)
    geometry = _load_geometry(geometry_path)
    sources = topology["sources"]
    missing_sources = [name for name in SOURCE_ORDER if name not in sources]
    if missing_sources:
        raise ValueError(f"training topology missing expected sources: {missing_sources}")
    if shape_index < 0 or shape_index >= int(geometry["centers"].shape[0]):
        raise ValueError(f"shape_index {shape_index} outside centers shape {geometry['centers'].shape}")
    if geometry["centers"].shape[1] != len(SOURCE_ORDER):
        raise ValueError(f"unexpected source axis in centers: {geometry['centers'].shape}")

    graph_audits: dict[str, Any] = {}
    coarse_audits: dict[str, Any] = {}
    for source_index, source_name in enumerate(SOURCE_ORDER):
        graph = _source_graph_audit(source_name, sources[source_name])
        _add_branch_audit(sources[source_name], graph)
        graph_audits[source_name] = graph
        coarse_audits[source_name] = _coarse_role_audit(
            geometry, source_index, source_name, shape_index
        )

    neutral_grid = np.asarray(geometry["neutral_grid"], dtype=float)
    route_start = float(np.max(neutral_grid))
    route_end = float(np.min(neutral_grid))
    summary_by_source = {
        name: graph_audits[name]["route_summary"] for name in SOURCE_ORDER
    }
    result: dict[str, Any] = {
        "audit": {
            "name": "leg_scan_target_routes_20261123_v2.atlas_coverage_audit",
            "version": 2,
            "status": "passed",
            "read_only": True,
            "training_inputs_only": True,
            "blender_opened": False,
            "blender_input_in_scope": False,
            "source_order": list(SOURCE_ORDER),
            "shape_index_for_coarse_geometry": int(shape_index),
        },
        "inputs": {
            "topology_path": str(topology_path),
            "topology_sha256": _sha256(topology_path),
            "geometry_path": str(geometry_path),
            "geometry_sha256": _sha256(geometry_path),
            "topology_coordinates": topology.get("coordinates"),
            "topology_station_step_mm": topology.get("station_step_mm"),
            "geometry_keys": sorted(geometry),
        },
        "intended_route": {
            "ordered_named_route": list(ROUTE),
            "direction": "proximal/groin to distal/calf",
            "neutral_start_mm": route_start,
            "neutral_end_mm": route_end,
            "neutral_station_count": int(len(neutral_grid)),
            "neutral_station_step_mm": float(
                abs(float(np.median(np.diff(neutral_grid))))
            )
            if len(neutral_grid) > 1
            else None,
            "topology_station_start_mm": float(-route_start),
            "topology_station_end_mm": float(-route_end),
            "legacy_scan_task_window": {
                "name": "groin_to_distal_calf",
                "neutral_start_mm": -525.0,
                "neutral_end_mm": -1255.0,
                "note": "Historical task window; full ct_geometry grid is -480..-1280 mm.",
            },
            "branches": {
                "great_saphenous_vein": "saphenofemoral incidence; do not treat a role row as a measured cross-role join",
                "deep_femoral_vein": "proximal-thigh incidence; preserve identity",
                "fibular_vein": "separate calf vessel; no continuation edge",
            },
        },
        "sources": graph_audits,
        "coarse_role_geometry": coarse_audits,
        "source_summary": summary_by_source,
        "verification": {
            "expected_source_count": len(SOURCE_ORDER),
            "observed_source_count": len(sources),
            "expected_shape_axis": int(geometry["centers"].shape[0]),
            "geometry_centers_shape": list(geometry["centers"].shape),
            "geometry_status_shape": list(geometry["status"].shape),
            "geometry_links_shape": list(geometry["links"].shape),
            "no_blender_path_opened": True,
            "all_expected_sources_present": not missing_sources,
            "source_summary": {
                name: {
                    "exact_labels_complete": bool(
                        graph_audits[name]["route_summary"]["exact_named_label_coverage_complete"]
                    ),
                    "full_route_declared_edges": bool(
                        graph_audits[name]["route_summary"][
                            "full_named_route_supported_by_declared_edges"
                        ]
                    ),
                    "same_source_inferred_bridge_count": int(
                        graph_audits[name]["route_summary"][
                            "measured_same_source_inferred_bridge_count"
                        ]
                    ),
                    "unsupported_transitions": graph_audits[name]["route_summary"][
                        "unsupported_route_transitions"
                    ],
                }
                for name in SOURCE_ORDER
            },
        },
    }
    result = _jsonable(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "atlas_coverage_audit.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(json_path)

    csv_path = out_dir / "route_coverage.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source",
                "record_kind",
                "name_or_transition",
                "node_count_or_edge_count",
                "station_min_mm",
                "station_max_mm",
                "provenance_or_status",
                "note",
            ),
        )
        writer.writeheader()
        for source_name in SOURCE_ORDER:
            graph = graph_audits[source_name]
            for name in ROUTE + tuple(BRANCHES):
                item = graph["named_segments"][name]
                writer.writerow(
                    {
                        "source": source_name,
                        "record_kind": "named_segment",
                        "name_or_transition": name,
                        "node_count_or_edge_count": item["node_count"],
                        "station_min_mm": item["station_min_mm"],
                        "station_max_mm": item["station_max_mm"],
                        "provenance_or_status": ";".join(
                            f"{k}:{v}" for k, v in sorted(item["provenance_counts"].items())
                        ),
                        "note": "exact named graph; role geometry is reported separately",
                    }
                )
            for transition in graph["route_transitions"]:
                writer.writerow(
                    {
                        "source": source_name,
                        "record_kind": "route_transition",
                        "name_or_transition": f"{transition['from']} -> {transition['to']}",
                        "node_count_or_edge_count": transition["edge_count"],
                        "station_min_mm": "",
                        "station_max_mm": "",
                        "provenance_or_status": transition["status"],
                        "note": json.dumps(
                            transition["provenance_counts"], ensure_ascii=False, sort_keys=True
                        ),
                    }
                )
    csv_tmp.replace(csv_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path("leg_scan_study/topology/ct_reference_topology.json.gz"),
        help="training-only named graph (never a Blender topology)",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=Path("leg_scan_study/topology/ct_geometry.npz"),
        help="training-only coarse role geometry",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("leg_scan_target_routes_20261123_v2"),
    )
    parser.add_argument("--shape", type=int, default=0, help="SMPL-X shape row for coarse geometry")
    args = parser.parse_args()
    result = run_audit(args.topology, args.geometry, args.out_dir, shape_index=args.shape)
    print(json.dumps(result["verification"], ensure_ascii=False, indent=2))
    print(f"wrote {args.out_dir / 'atlas_coverage_audit.json'}")
    print(f"wrote {args.out_dir / 'route_coverage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

