"""Project the audited connected venous routes onto the seven SMPL-X shapes.

This module deliberately has a small input surface.  It reads only the training
``ct_geometry.npz`` and ``ct_reference_topology.json.gz`` packs.  In particular,
it does not import the rendering stack and it never opens a Blender or native
MedSeg asset.  The output is a Scanner-compatible training pack with the source
axis restricted to the three geometry sources used by the fixed pilot.

The old coarse projection selected a named ring independently at every station.
That is unsafe at a bifurcation: it can jump from a proximal component to a
disconnected posterior-tibial component.  Here each role-0 projection is a
single, ordered graph path.  The only synthetic sample is the 750-mm O3D sample,
which lies strictly inside the audited 746 -> 751 inferred-junction edge.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_NAMES = (
    "BodyParts3D right",
    "BodyParts3D mirrored left",
    "Open3DModel right",
)
SOURCE_INDICES = (0, 1, 2)
ROUTE_SPECS = {
    "femoral": {"neutral_start_mm": -595.0, "neutral_end_mm": -790.0},
    "groin_to_distal_calf": {
        "neutral_start_mm": -525.0,
        "neutral_end_mm": -1255.0,
    },
}

# These codes are the historical topology pack codes.  In particular, an
# inferred_junction is kept distinct from sampled in the JSON audit but maps to
# the existing edge_status value used by Scanner (2 = inferred).
STATUS_MISSING = 0
STATUS_SAMPLED = 1
STATUS_INTERPOLATED = 2
EDGE_MISSING = 0
EDGE_SAMPLED = 1
EDGE_INFERRED = 2
EDGE_SOURCE_JUNCTION = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_topology(path: Path) -> dict[str, Any]:
    # The caller supplies the training graph.  Keep this loader intentionally
    # narrow so a future refactor cannot accidentally reach a holdout graph.
    if "blender" in path.name.lower() or "holdout" in path.name.lower():
        raise ValueError(f"Refusing a holdout topology input: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _node_map(graph: dict[str, Any], component_index: int) -> tuple[dict[int, dict[str, Any]], set[int]]:
    nodes = {int(node["id"]): node for node in graph["nodes"]}
    component = {int(node_id) for node_id in graph["components"][component_index]}
    return nodes, component


def _edge_map(graph: dict[str, Any]) -> dict[frozenset[int], dict[str, Any]]:
    result: dict[frozenset[int], dict[str, Any]] = {}
    for edge in graph["edges"]:
        node_ids = [int(value) for value in edge["nodes"]]
        if len(node_ids) != 2:
            continue
        result[frozenset(node_ids)] = edge
    return result


def _edge_json(edge: dict[str, Any], *, nodes: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return a compact, JSON-safe edge audit retaining both endpoint names."""

    result: dict[str, Any] = {
        "nodes": [int(value) for value in edge["nodes"]],
        "provenance": str(edge.get("provenance", "unknown")),
    }
    if nodes is not None:
        result["names"] = [str(nodes[int(value)]["name"]) for value in edge["nodes"]]
        result["stations_mm"] = [int(nodes[int(value)]["station_mm"]) for value in edge["nodes"]]
    evidence = edge.get("evidence")
    if evidence is not None:
        # Evidence is already composed of JSON scalars in the handoff graph.
        result["evidence"] = evidence
    return result


def _node_json(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(node["id"]),
        "station_mm": int(node["station_mm"]),
        "name": str(node["name"]),
        "xyz_mm": [float(value) for value in node["xyz_mm"]],
        "surface_component": int(node.get("surface_component", -1)),
        "provenance": str(node.get("provenance", "unknown")),
    }


def _build_path(
    source_name: str,
    graph: dict[str, Any],
    *,
    component_index: int,
    segments: tuple[tuple[str, int, int], ...],
    bridge_edges: tuple[tuple[int, int], ...] = (),
    path_kind: str = "connected_chain",
) -> dict[str, Any]:
    """Build one monotone named path without crossing components.

    ``segments`` are inclusive station/name ranges in the graph's positive
    station coordinate.  A bridge edge may connect two ranges whose stations
    have a gap or whose names change.  Every selected node is checked against
    the declared component and every consecutive pair is checked against the
    graph edge table.
    """

    nodes, component = _node_map(graph, component_index)
    by_name_station: dict[tuple[str, int], dict[str, Any]] = {}
    for node_id in component:
        node = nodes[node_id]
        by_name_station[(str(node["name"]), int(node["station_mm"]))] = node

    selected: dict[int, dict[str, Any]] = {}
    for name, first, last in segments:
        for station in range(int(first), int(last) + 1):
            node = by_name_station.get((name, station))
            if node is None:
                raise ValueError(
                    f"{source_name}: missing {name} at graph station {station} in component {component_index}"
                )
            selected[station] = node

    ordered = [selected[station] for station in sorted(selected)]
    edge_lookup = _edge_map(graph)
    ordered_edges: list[dict[str, Any]] = []
    bridge_keys = {frozenset((int(a), int(b))) for a, b in bridge_edges}
    for left, right in zip(ordered, ordered[1:]):
        key = frozenset((int(left["id"]), int(right["id"])))
        edge = edge_lookup.get(key)
        if edge is None:
            raise ValueError(
                f"{source_name}: path leaves graph between {left['id']} and {right['id']}"
            )
        ordered_edges.append(edge)

    # A bridge is required to be an actual edge and to occur in the selected
    # ordered path.  This guards against silently replacing a missing section by
    # a geometric interpolation.
    for a, b in bridge_edges:
        key = frozenset((int(a), int(b)))
        edge = edge_lookup.get(key)
        if edge is None:
            raise ValueError(f"{source_name}: declared bridge edge {a}<->{b} is absent")
        if key not in {frozenset((int(x["id"]), int(y["id"]))) for x, y in zip(ordered, ordered[1:])}:
            raise ValueError(f"{source_name}: declared bridge edge {a}<->{b} is not on path")

    order_by_id = {int(node["id"]): index for index, node in enumerate(ordered)}
    bridge_records = []
    for left_id, right_id in bridge_edges:
        left = nodes[int(left_id)]
        right = nodes[int(right_id)]
        bridge_records.append(
            {
                "left_id": int(left_id),
                "right_id": int(right_id),
                "left_station_mm": int(left["station_mm"]),
                "right_station_mm": int(right["station_mm"]),
                "left_name": str(left["name"]),
                "right_name": str(right["name"]),
                "edge": _edge_json(edge_lookup[frozenset((int(left_id), int(right_id)))], nodes=nodes),
                "left_order_index": order_by_id[int(left_id)],
                "right_order_index": order_by_id[int(right_id)],
            }
        )

    return {
        "source": source_name,
        "component_index": int(component_index),
        "component_size": len(component),
        "component_node_ids": sorted(component),
        "segments": [
            {"name": str(name), "start_station_mm": int(first), "end_station_mm": int(last)}
            for name, first, last in segments
        ],
        "ordered_nodes": ordered,
        "ordered_edges": ordered_edges,
        "order_by_id": order_by_id,
        "bridge_records": bridge_records,
        "path_kind": path_kind,
        "selected_by_station": selected,
        "edge_lookup": edge_lookup,
        "nodes": nodes,
    }


def _main_paths(graphs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select the audited connected chain for each of the three training sources."""

    # Component indices are those in ct_reference_topology.json.gz.  The audit
    # calls the O3D filtered venous component "component 0"; in the complete
    # graph it is component 1 because the arterial component sorts first.
    return {
        "BodyParts3D right": _build_path(
            "BodyParts3D right",
            graphs["BodyParts3D right"],
            component_index=2,
            segments=(
                ("femoral vein", 480, 750),
                ("popliteal vein", 751, 914),
            ),
            bridge_edges=((270, 282),),
        ),
        # The mirrored source has separate femoral and popliteal components.
        # Keep its verified femoral fragment for nuisance geometry, but do not
        # invent a bridge to make either route eligible.
        "BodyParts3D mirrored left": _build_path(
            "BodyParts3D mirrored left",
            graphs["BodyParts3D mirrored left"],
            component_index=6,
            segments=(("femoral vein", 480, 756),),
            path_kind="femoral_fragment_no_verified_join",
        ),
        "Open3DModel right": _build_path(
            "Open3DModel right",
            graphs["Open3DModel right"],
            component_index=1,
            segments=(
                ("Femoral_vein.r", 480, 746),
                ("Popliteal_vein.r", 751, 909),
                ("Posterior_tibial_vein.r", 910, 1264),
            ),
            bridge_edges=((266, 267), (425, 426)),
        ),
    }


def _competitor_paths(graphs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    """Choose the largest disconnected same-name posterior-tibial branch."""

    choices = {
        "BodyParts3D right": ("posterior tibial vein", 7),
        "BodyParts3D mirrored left": ("posterior tibial vein", 7),
        # This is component 6 in the complete O3D graph (the connected-chain
        # component above is complete-graph component 1).
        "Open3DModel right": ("Posterior_tibial_vein.r", 6),
    }
    result: dict[str, dict[str, Any] | None] = {}
    for source, (name, component_index) in choices.items():
        graph = graphs[source]
        nodes, component = _node_map(graph, component_index)
        stations = sorted(int(nodes[node_id]["station_mm"]) for node_id in component)
        if not stations:
            result[source] = None
            continue
        result[source] = _build_path(
            source,
            graph,
            component_index=component_index,
            segments=((name, stations[0], stations[-1]),),
            path_kind="disconnected_same_name_competitor",
        )
    return result


def _bridge_for_station(path: dict[str, Any], station_mm: int) -> dict[str, Any] | None:
    for bridge in path["bridge_records"]:
        left = int(bridge["left_station_mm"])
        right = int(bridge["right_station_mm"])
        if left < station_mm < right:
            return bridge
    return None


def _point_at(path: dict[str, Any], station_mm: int) -> dict[str, Any] | None:
    """Return an exact node or a permitted point inside one audited bridge."""

    exact = path["selected_by_station"].get(int(station_mm))
    if exact is not None:
        order_index = path["order_by_id"][int(exact["id"])]
        return {
            "supported": True,
            "synthetic": False,
            "station_mm": int(station_mm),
            "node_id": int(exact["id"]),
            "node_ids": [int(exact["id"])],
            "name": str(exact["name"]),
            "xyz_mm": [float(value) for value in exact["xyz_mm"]],
            "order_index": order_index,
            "node": exact,
        }

    bridge = _bridge_for_station(path, int(station_mm))
    if bridge is None:
        return None
    left = path["nodes"][int(bridge["left_id"])]
    right = path["nodes"][int(bridge["right_id"])]
    left_station = float(left["station_mm"])
    right_station = float(right["station_mm"])
    fraction = (float(station_mm) - left_station) / (right_station - left_station)
    xyz = (1.0 - fraction) * np.asarray(left["xyz_mm"], dtype=float) + fraction * np.asarray(
        right["xyz_mm"], dtype=float
    )
    return {
        "supported": True,
        "synthetic": True,
        "station_mm": int(station_mm),
        "node_id": -1,
        "node_ids": [int(left["id"]), int(right["id"])],
        "name": f"{left['name']} -> {right['name']} (confirmed bridge)",
        "xyz_mm": [float(value) for value in xyz],
        "order_index": int(bridge["left_order_index"]),
        "left_order_index": int(bridge["left_order_index"]),
        "right_order_index": int(bridge["right_order_index"]),
        "bridge": bridge,
        "fraction": float(fraction),
    }


def _edges_between(path: dict[str, Any], start_index: int, end_index: int) -> list[dict[str, Any]]:
    if end_index < start_index:
        raise ValueError(f"Non-monotone path request {start_index}->{end_index}")
    return path["ordered_edges"][start_index:end_index]


def _interval_edges(path: dict[str, Any], previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """Graph edges represented by one target-grid interval."""

    if previous.get("synthetic"):
        bridge = previous["bridge"]["edge"]
        start = int(previous["right_order_index"])
        end = int(current["order_index"])
        return [bridge, *_edges_between(path, start, end)]
    if current.get("synthetic"):
        start = int(previous["order_index"])
        end = int(current["left_order_index"])
        bridge = current["bridge"]["edge"]
        return [*_edges_between(path, start, end), bridge]
    return _edges_between(path, int(previous["order_index"]), int(current["order_index"]))


def _edge_code(edges: list[dict[str, Any]]) -> int:
    if not edges:
        return EDGE_MISSING
    provenance = {str(edge.get("provenance", "unknown")) for edge in edges}
    if provenance <= {"sampled"}:
        return EDGE_SAMPLED
    if "source_junction" in provenance:
        return EDGE_SOURCE_JUNCTION
    return EDGE_INFERRED


def _polar_project(
    base_point_xz: np.ndarray,
    station_index: int,
    origins: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Move a neutral point with the legacy section-wise polar radius ratio."""

    base_origin = np.asarray(origins[0, station_index], dtype=float)
    delta = np.asarray(base_point_xz, dtype=float) - base_origin
    if not np.isfinite(delta).all():
        return np.full((len(origins), 2), np.nan, dtype=float)
    length = float(np.linalg.norm(delta))
    if length < 1.0e-12:
        return np.asarray(origins[:, station_index], dtype=float).copy()
    theta = float(np.mod(np.arctan2(delta[1], delta[0]), 2.0 * np.pi))
    sector = theta * radii.shape[2] / (2.0 * np.pi)
    lo = int(np.floor(sector)) % radii.shape[2]
    hi = (lo + 1) % radii.shape[2]
    weight = sector - np.floor(sector)
    base_radius = (1.0 - weight) * float(radii[0, station_index, lo]) + weight * float(
        radii[0, station_index, hi]
    )
    if not np.isfinite(base_radius) or base_radius <= 0.0:
        raise ValueError(f"Invalid neutral radius at station index {station_index}")
    result = np.empty((len(origins), 2), dtype=float)
    for shape in range(len(origins)):
        radius = (1.0 - weight) * float(radii[shape, station_index, lo]) + weight * float(
            radii[shape, station_index, hi]
        )
        result[shape] = np.asarray(origins[shape, station_index], dtype=float) + delta * (radius / base_radius)
    return result


def _station_record(path: dict[str, Any], station_mm: int, previous: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    point = _point_at(path, int(station_mm))
    if point is None:
        return (
            {
                "station_mm": int(station_mm),
                "neutral_mm": float(-station_mm),
                "supported": False,
                "synthetic": False,
                "target_name": None,
                "node_id": -1,
                "node_ids": [],
                "edge_provenance": [],
                "reason": "no_node_on_selected_component_or_confirmed_bridge",
            },
            None,
        )
    edges = [] if previous is None else _interval_edges(path, previous, point)
    return (
        {
            "station_mm": int(station_mm),
            "neutral_mm": float(-station_mm),
            "supported": True,
            "synthetic": bool(point.get("synthetic", False)),
            "target_name": str(point["name"]),
            "node_id": int(point["node_id"]),
            "node_ids": [int(value) for value in point["node_ids"]],
            "edge_provenance": [_edge_json(edge, nodes=path["nodes"]) for edge in edges],
            "reason": "confirmed_bridge_only" if point.get("synthetic") else "exact_graph_node",
        },
        point,
    )


def _route_grid(neutral_grid: np.ndarray, spec: dict[str, float]) -> np.ndarray:
    lo = float(min(spec["neutral_start_mm"], spec["neutral_end_mm"]))
    hi = float(max(spec["neutral_start_mm"], spec["neutral_end_mm"]))
    selected = neutral_grid[(neutral_grid >= lo) & (neutral_grid <= hi)]
    # The source grid is descending.  Keep that historical order and require
    # exact 5-mm endpoints rather than silently shortening a route.
    if selected.size == 0 or not np.isclose(selected[0], spec["neutral_start_mm"]) or not np.isclose(
        selected[-1], spec["neutral_end_mm"]
    ):
        raise ValueError(f"Route bounds are not represented by neutral_grid: {spec}")
    return selected


def _project(
    geometry_path: Path,
    topology_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    with np.load(geometry_path, allow_pickle=False) as old:
        required = {"ys", "origins", "radii", "centers", "status", "neutral_grid", "links", "edge_status"}
        missing = required.difference(old.files)
        if missing:
            raise ValueError(f"ct_geometry is missing required arrays: {sorted(missing)}")
        ys = np.asarray(old["ys"], dtype=float)
        origins = np.asarray(old["origins"], dtype=float)
        radii = np.asarray(old["radii"], dtype=float)
        old_centers = np.asarray(old["centers"], dtype=float)
        old_status = np.asarray(old["status"], dtype=np.uint8)
        neutral_grid = np.asarray(old["neutral_grid"], dtype=float)
        old_links = np.asarray(old["links"], dtype=bool)
        old_edge_status = np.asarray(old["edge_status"], dtype=np.uint8)

    if ys.shape[0] != 7 or old_centers.shape[:3] != (7, 4, len(neutral_grid)):
        raise ValueError(f"Unexpected source geometry shapes: ys={ys.shape}, centers={old_centers.shape}")
    if old_centers.shape[3:] != (6, 2):
        raise ValueError(f"Unexpected role geometry shape: {old_centers.shape}")
    if neutral_grid.ndim != 1 or len(neutral_grid) != 161:
        raise ValueError(f"Expected the historical 161-station grid, got {neutral_grid.shape}")

    topology = _load_topology(topology_path)
    all_graphs = topology.get("sources", {})
    if any(name not in all_graphs for name in SOURCE_NAMES):
        raise ValueError("Training topology does not contain all three requested sources")
    graphs = {name: all_graphs[name] for name in SOURCE_NAMES}
    paths = _main_paths(graphs)
    competitors = _competitor_paths(graphs)

    # Preserve the posture and the previously validated nuisance roles.  Only
    # role 0 (connected venous chain) and spare role 5 are reconstructed below.
    centers = np.full((7, 3, len(neutral_grid), 6, 2), np.nan, dtype=float)
    centers[:, :, :, 1:5] = old_centers[:, :3, :, 1:5]
    status = np.zeros((3, len(neutral_grid), 6), dtype=np.uint8)
    status[:, :, 1:5] = old_status[:3, :, 1:5]
    links = np.zeros((3, len(neutral_grid) - 1, 6), dtype=bool)
    links[:, :, 1:5] = old_links[:3, :, 1:5]
    edge_status = np.zeros((3, len(neutral_grid) - 1, 6), dtype=np.uint8)
    edge_status[:, :, 1:5] = old_edge_status[:3, :, 1:5]
    selected_node_ids = np.full((3, len(neutral_grid), 6), -1, dtype=np.int32)
    synthetic_bridge = np.zeros((3, len(neutral_grid), 6), dtype=bool)

    station_values = np.rint(-neutral_grid).astype(int)
    station_index = {int(station): index for index, station in enumerate(station_values)}
    source_audit: dict[str, Any] = {}

    for source_index, source_name in enumerate(SOURCE_NAMES):
        path = paths[source_name]
        competitor = competitors[source_name]
        path_points: dict[int, dict[str, Any] | None] = {}
        competitor_points: dict[int, dict[str, Any] | None] = {}
        for index, station in enumerate(station_values):
            point = _point_at(path, int(station))
            path_points[int(station)] = point
            if point is not None:
                p0 = np.asarray(point["xyz_mm"], dtype=float)[[0, 2]]
                centers[:, source_index, index, 0] = _polar_project(p0, index, origins, radii)
                status[source_index, index, 0] = STATUS_INTERPOLATED if point.get("synthetic") else STATUS_SAMPLED
                selected_node_ids[source_index, index, 0] = int(point["node_id"])
                synthetic_bridge[source_index, index, 0] = bool(point.get("synthetic", False))
            if competitor is not None:
                cpoint = _point_at(competitor, int(station))
                competitor_points[int(station)] = cpoint
                if cpoint is not None:
                    cp0 = np.asarray(cpoint["xyz_mm"], dtype=float)[[0, 2]]
                    centers[:, source_index, index, 5] = _polar_project(cp0, index, origins, radii)
                    status[source_index, index, 5] = STATUS_SAMPLED
                    selected_node_ids[source_index, index, 5] = int(cpoint["node_id"])

        # Build edge reachability on the actual target grid.  A missing point
        # breaks the link; no nearest-neighbour or long-range fill is allowed.
        for index in range(len(station_values) - 1):
            previous = path_points[int(station_values[index])]
            current = path_points[int(station_values[index + 1])]
            if previous is not None and current is not None:
                interval = _interval_edges(path, previous, current)
                edge_status[source_index, index, 0] = _edge_code(interval)
                links[source_index, index, 0] = edge_status[source_index, index, 0] > 0
            if competitor is not None:
                previous_c = competitor_points.get(int(station_values[index]))
                current_c = competitor_points.get(int(station_values[index + 1]))
                if previous_c is not None and current_c is not None:
                    interval_c = _interval_edges(competitor, previous_c, current_c)
                    edge_status[source_index, index, 5] = _edge_code(interval_c)
                    links[source_index, index, 5] = edge_status[source_index, index, 5] > 0

        source_audit[source_name] = {
            "source_index": source_index,
            "path_component_index": path["component_index"],
            "audit_component_index": 0
            if source_name == "Open3DModel right"
            else path["component_index"],
            "path_component_size": path["component_size"],
            "path_kind": path["path_kind"],
            "path_station_span_mm": [
                int(path["ordered_nodes"][0]["station_mm"]),
                int(path["ordered_nodes"][-1]["station_mm"]),
            ],
            "path_nodes": [_node_json(node) for node in path["ordered_nodes"]],
            "path_edges": [_edge_json(edge, nodes=path["nodes"]) for edge in path["ordered_edges"]],
            "bridge_edges": path["bridge_records"],
            "competitor": None
            if competitor is None
            else {
                "component_index": competitor["component_index"],
                "audit_filtered_venous_component_rank": 1,
                "component_size": competitor["component_size"],
                "path_kind": competitor["path_kind"],
                "station_span_mm": [
                    int(competitor["ordered_nodes"][0]["station_mm"]),
                    int(competitor["ordered_nodes"][-1]["station_mm"]),
                ],
                "path_nodes": [_node_json(node) for node in competitor["ordered_nodes"]],
                "path_edges": [_edge_json(edge, nodes=competitor["nodes"]) for edge in competitor["ordered_edges"]],
            },
        }

    # Keep the output self-consistent even if the old pack used a uint8 view for
    # links: links are exactly the non-missing edge statuses for every rebuilt
    # role, while preserved nuisance roles remain byte-for-byte equivalent.
    if not np.array_equal(links[:, :, 0], edge_status[:, :, 0] > 0):
        raise AssertionError("role-0 link/status mismatch")
    if not np.array_equal(links[:, :, 5], edge_status[:, :, 5] > 0):
        raise AssertionError("role-5 link/status mismatch")

    routes: dict[str, Any] = {}
    for route_name, spec in ROUTE_SPECS.items():
        route_neutral = _route_grid(neutral_grid, spec)
        route_stations = [int(round(-float(value))) for value in route_neutral]
        per_source: dict[str, Any] = {}
        eligible: list[int] = []
        for source_index, source_name in enumerate(SOURCE_NAMES):
            path = paths[source_name]
            records = []
            previous = None
            for station in route_stations:
                record, current = _station_record(path, station, previous)
                records.append(record)
                previous = current
            complete = all(bool(record["supported"]) for record in records)
            # A connected route requires the selected graph component to cover
            # the full requested range.  The mirrored femoral fragment is not
            # eligible even though its proximal records are individually exact.
            if path["path_kind"] != "connected_chain":
                complete = False
            if complete:
                eligible.append(source_index)
            per_source[source_name] = {
                "source_index": source_index,
                "eligible": bool(complete),
                "reason": "complete_connected_chain"
                if complete
                else (
                    "no_verified_join_between_femoral_and_popliteal"
                    if path["path_kind"] != "connected_chain"
                    else "selected_component_does_not_cover_requested_range"
                ),
                "station_records": records,
            }
        routes[route_name] = {
            **spec,
            "station_count": len(route_stations),
            "station_mm": route_stations,
            "eligible_source_indices": eligible,
            "eligible_sources": [SOURCE_NAMES[index] for index in eligible],
            "per_source": per_source,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / "atlas_routes.npz"
    np.savez_compressed(
        pack_path,
        ys=ys,
        origins=origins,
        radii=radii,
        centers=centers,
        neutral_grid=neutral_grid,
        status=status,
        links=links,
        edge_status=edge_status,
        selected_node_ids=selected_node_ids,
        synthetic_bridge=synthetic_bridge,
    )

    manifest: dict[str, Any] = {
        "schema": "leg_scan_target_routes_20261123_v2",
        "status": "generated",
        "inputs": {
            "ct_geometry": str(geometry_path),
            "ct_reference_topology": str(topology_path),
            "ct_geometry_sha256": _sha256(geometry_path),
            "ct_reference_topology_sha256": _sha256(topology_path),
        },
        "holdout_opened": False,
        "raw_medseg_opened": False,
        "blender_opened": False,
        "source_names": list(SOURCE_NAMES),
        "source_indices": list(SOURCE_INDICES),
        "array_shapes": {
            "ys": list(ys.shape),
            "origins": list(origins.shape),
            "radii": list(radii.shape),
            "centers": list(centers.shape),
            "neutral_grid": list(neutral_grid.shape),
            "status": list(status.shape),
            "links": list(links.shape),
            "edge_status": list(edge_status.shape),
        },
        "status_codes": {
            "missing": STATUS_MISSING,
            "sampled": STATUS_SAMPLED,
            "interpolated_confirmed_bridge": STATUS_INTERPOLATED,
        },
        "edge_status_codes": {
            "missing": EDGE_MISSING,
            "sampled": EDGE_SAMPLED,
            "inferred": EDGE_INFERRED,
            "source_junction": EDGE_SOURCE_JUNCTION,
        },
        "projection": {
            "route_axis": "positive graph station maps to negative neutral_grid",
            "shape_transport": "legacy move_section equivalent: per-section polar radius ratio",
            "nuisance_roles": "roles 1-4 copied from the three-source handoff pack",
            "role_0": "one monotone graph-edge-respecting named deep-venous chain",
            "role_5": "largest disconnected same-name posterior-tibial component per source",
            "unsupported_fill": "none; only target -750 O3D is interpolated inside 746->751 confirmed bridge",
            "o3d_audit_component_index": 0,
            "o3d_complete_graph_component_index": 1,
        },
        "sources": source_audit,
        "routes": routes,
        "pack_sha256": _sha256(pack_path),
    }
    manifest_path = output_dir / "atlas_routes_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "leg_scan_study/topology/ct_geometry.npz",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "leg_scan_study/topology/ct_reference_topology.json.gz",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    manifest = _project(args.geometry.resolve(), args.topology.resolve(), args.out.resolve())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "pack": str(args.out.resolve() / "atlas_routes.npz"),
                "routes": {
                    name: {
                        "eligible_source_indices": value["eligible_source_indices"],
                        "station_count": value["station_count"],
                    }
                    for name, value in manifest["routes"].items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

