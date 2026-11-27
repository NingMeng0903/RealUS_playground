"""Rebuild conservative training topology and validate the native MedSeg route."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
import gzip
import io
import json
import os
import sys

import numpy as np

from native_ct import build_native, components, ct_route, route_audit, write_native_pack
from section_graph import SectionGraph


STATIONS = np.arange(-480., -1280.1, -5.)
TRAIN_SOURCES = ("BodyParts3D right", "BodyParts3D mirrored left", "Open3DModel right", "MedSeg")
DEFAULT_PROJECT = Path("/home/camp/Documents/Codex/2026-11-13/ieee-t-ro-b-mpc-b")
DRIVE = Path("/media/camp/PEI_T7")
LABELS = DRIVE / "Leg_Vascular_Reference_Extraction/medseg_ct/groin_labels.nii.gz"
BASELINE = (DRIVE / "1_team meeting/atlas_ppt_build/assets/blender_topology_holdout/"
            "ct_continuous_prodmp/ct_reference_topology.json")
OUT = Path(__file__).resolve().parent
STATUS = {"missing": 0, "sampled": 1, "interpolated": 2}
EDGE = {"missing": 0, "sampled": 1, "inferred": 2, "source_junction": 3}


def sha(path):
    with Path(path).open("rb") as f:
        chunks = iter(lambda: f.read(1024 * 1024), b"")
        h = sha256()
        for data in chunks:
            h.update(data)
        return h.hexdigest()


def write_json(path, data, *, compress=False):
    if compress:
        with Path(path).open("wb") as stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8") as text:
                    json.dump(data, text, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    text.write("\n")
    else:
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_graph(path):
    """Read one of the separated source graph outputs."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def load_ct_training(path=OUT / "medseg_ct_route.npz", shape=0):
    """Read 31 consecutive CT-labelled stations in native or a selected neutral/drawn shape."""
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k] for k in z.files}
    if "shape_stations_mm" in result:
        result["stations_mm"] = result["shape_stations_mm"][shape]
        result["role_centers_xz_mm"] = result["shape_centers_xz_mm"][shape]
        result["skin_origins_mm"] = result["shape_origins_mm"][shape]
        result["skin_radii_mm"] = result["shape_radii_mm"][shape]
    return result


def training_input_guard():
    """Fail on a held-out vascular/evaluator file read during the training CLI."""
    blocked = {"vessel_centerlines_rest.obj", "arterial_paths.json", "geometry_stages.npz",
               "test_evaluator_labels.jsonl.gz", "test_observations.jsonl.gz",
               "blender_holdout_topology.json.gz", "blender_holdout_geometry_links.npz"}

    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            if Path(os.fsdecode(args[0])).name in blocked:
                raise PermissionError(f"Holdout read blocked in training process: {args[0]}")

    sys.addaudithook(audit)


def import_source(project):
    path = project / "work/leg_reference_transfer/blender_holdout"
    sys.path.insert(0, str(path))
    import geometry as g
    import shape_training as s
    return g, s


def region_checker(origin, radii, occupancy):
    def inside(xyz):
        grid = -STATIONS
        x, y, z = xyz.T
        center = np.column_stack([np.interp(-y, grid, c) for c in origin.T])
        direction = xyz[:, [0, 2]] - center
        angle = np.mod(np.arctan2(direction[:, 1], direction[:, 0]), 2 * np.pi)
        angle_idx = angle * radii.shape[1] / (2 * np.pi)
        lo = np.floor(angle_idx).astype(int)
        hi = (lo + 1) % radii.shape[1]
        weight = angle_idx - lo
        limit = np.array([(1 - w) * np.interp(-yy, grid, radii[:, a])
                          + w * np.interp(-yy, grid, radii[:, b])
                          for yy, w, a, b in zip(y, weight, lo, hi, strict=True)])
        if (np.linalg.norm(direction, axis=1) > limit - .25).any():
            return "outside_skin"
        if (occupancy(xyz) > .55).any():
            return "high_population_bone_occupancy"
        return None
    return inside


def source_contacts(native):
    contacts = Counter()
    for edge in native["verified_junction_edges"]:
        contacts[tuple(edge["labels"])] += edge["shared_voxel_faces"]
    return contacts


def graph_summary(graph):
    return {"nodes": len(graph["nodes"]),
            "node_provenance": dict(Counter(n["provenance"] for n in graph["nodes"])),
            "edges": len(graph["edges"]),
            "edge_provenance": dict(Counter(e["provenance"] for e in graph["edges"])),
            "components": len(graph["components"]),
            "largest_components": sorted((len(c) for c in graph["components"]), reverse=True)[:8],
            "audit_outcomes": dict(Counter(x["status"] for x in graph["gaps_and_join_candidates"]))}


def resolve_declared(graph, groups, native=None):
    for role, names in enumerate(groups):
        available = [n for n in names if graph.by_name.get(n)]
        for a, b in pairwise(available):
            if role in (0, 1):
                if graph.mode == "mesh":
                    graph.mesh_junction(a, b, native)
                else:
                    graph.line_junction(a, b)
            else:
                graph.audit.append({"left": a, "right": b,
                                    "status": "shared_classifier_role_distinct_vessels"})


def graph_projection(graph):
    """Project graph visibility to legacy stations and export actual edge reachability."""
    shape = (len(STATIONS), len(graph.groups))
    points = np.full((*shape, 2), np.nan)
    status = np.zeros(shape, dtype=np.uint8)
    links = np.zeros((len(STATIONS) - 1, len(graph.groups)), dtype=np.uint8)
    chosen = np.full(shape, -1, dtype=np.int32)
    adjacency = defaultdict(list)
    for edge in graph.edges:
        a, b = (graph.nodes[i] for i in edge["nodes"])
        kind = edge["provenance"]
        if b.axis > a.axis:
            adjacency[a.id].append((b.id, kind))
    states = {"sampled": EDGE["sampled"], "source_junction": EDGE["source_junction"],
              "inferred": EDGE["inferred"], "inferred_junction": EDGE["inferred"]}
    for col, group in enumerate(graph.groups):
        per_station = defaultdict(list)
        rank = {name: i for i, name in enumerate(group)}
        for name in group:
            for n in graph.by_name.get(name, ()):
                if n.axis % 5 == 0:
                    per_station[n.axis].append(n)
        # A real, named ring retains precedence over a synthetic point.
        for s, y in enumerate(STATIONS):
            rows = per_station[-int(y)]
            if rows:
                n = max(rows, key=lambda r: (r.provenance != "interpolated",
                                             rank[r.name], r.area or 0., -r.id))
                points[s, col] = n.xyz[[0, 2]]
                status[s, col] = STATUS["interpolated" if n.provenance == "interpolated" else "sampled"]
                chosen[s, col] = n.id
        for s, (a, b) in enumerate(pairwise(chosen[:, col])):
            if a < 0 or b < 0:
                continue
            max_axis = graph.nodes[b].axis
            pending = [(int(a), EDGE["sampled"])]
            visited = set()
            available = set()
            while pending:
                entry = pending.pop()
                node, state = entry
                if entry in visited:
                    continue
                visited.add(entry)
                if node == b:
                    available.add(state)
                for nxt, label in adjacency[node]:
                    if graph.nodes[nxt].axis > max_axis or graph.nodes[nxt].name not in rank:
                        continue
                    code = states[label]
                    updated = EDGE["inferred"] if EDGE["inferred"] in (state, code) else max(state, code)
                    pending.append((nxt, updated))
            if available:
                for code in (EDGE["sampled"], EDGE["source_junction"], EDGE["inferred"]):
                    if code in available:
                        links[s, col] = code
                        break
    return points, status, chosen, links


def body_data(g, s):
    with np.load(s.SMPL, allow_pickle=False) as data:
        skin = data["v_template"].astype(float) * 1000.
        directions = data["shapedirs"][..., :3].astype(float) * 1000.
        regressor, faces = data["J_regressor"], data["f"]
    return skin, directions, regressor, faces, s.guides(skin, regressor @ skin)


def shape_project(s, data, y, centers):
    skin, directions, regressor, faces, guides = data
    c0, r0 = s.fixed_profiles(skin, faces, guides, y)
    moved = []
    for beta in s.BETAS:
        body = skin + np.einsum("vij,j->vi", directions, beta)
        marks = s.guides(body, regressor @ body)
        ys = np.interp(y, guides[:, 0, 1], marks[:, 0, 1])
        origin, radii = s.fixed_profiles(body, faces, marks, ys)
        if centers.ndim == 4:
            xz = np.stack([s.move_section(centers[:, i], c0, r0, origin, radii, i)
                           for i in range(len(ys))], axis=1)
        else:
            xz = np.stack([s.move_section(centers[i], c0, r0, origin, radii, i)
                           for i in range(len(ys))])
        moved.append((ys, origin, radii, xz))
    return tuple(np.stack(x) for x in zip(*moved, strict=True))


def native_shape_projection(native, g, s, data):
    from align_cloud_sources import medseg_map, section_outline
    from native_ct import ROLES

    target, guides = g.target_model()
    registration, _ = medseg_map(guides, section_outline(target))
    nodes, _ = ct_route(native)
    axis = np.array([n["station"] for n in nodes])
    centers = np.full((len(axis), len(ROLES), 3), np.nan)
    label_ids = np.zeros((len(axis), len(ROLES)), dtype=np.uint8)
    for i, role in enumerate(ROLES):
        for n in native["nodes"]:
            j = n["station"] - axis[0]
            if n["label_id"] not in role or j < 0 or j >= len(axis):
                continue
            if np.isfinite(centers[j, i]).any():
                # A role assignment is undefined if multiple rings survive on a layer.
                centers[j, i] = np.inf
                label_ids[j, i] = 0
                continue
            if np.isinf(centers[j, i]).any():
                continue
            centers[j, i] = n["xyz_mm"]
            label_ids[j, i] = n["label_id"]
    centers[np.isinf(centers)] = np.nan
    defined = np.isfinite(centers).all(axis=-1)
    aligned = np.full_like(centers, np.nan)
    source = centers[defined][:, [0, 2, 1]] * [-1., 1., -1.]
    aligned[defined] = registration.transform(source)
    if any(np.ptp(layer[np.isfinite(layer[:, 1]), 1]) > 1.e-4 for layer in aligned):
        raise ValueError("CT label registration did not preserve shared axial planes")
    ys, origins, radii, output = shape_project(s, data, aligned[:, 0, 1], aligned[..., [0, 2]])
    return {"source_role_available": defined, "source_role_label_ids": label_ids,
            "neutral_role_xyz_mm": aligned,
            "shape_stations_mm": ys, "shape_origins_mm": origins, "shape_radii_mm": radii,
            "shape_centers_xz_mm": output, "shape_betas": np.array(s.BETAS)}


def legacy_comparison(path, graphs):
    if not path.is_file():
        return None
    prior = load_graph(path)
    report = {}
    for name in TRAIN_SOURCES:
        rows = prior["sources"][name]
        now = graphs[name]
        old = [g for role in rows for g in role["gaps"]]
        latest = []
        for gap in old:
            left, right = gap["parts"]
            a, b = int(-gap["from_mm"]), int(-gap["to_mm"])
            matches = [x["status"] for x in now["gaps_and_join_candidates"]
                       if (x.get("left"), x.get("right")) == (left, right)
                       or x.get("name") == left == right
                       and a - 3 <= x.get("from_mm", -1) <= b + 3]
            local = [n["station_mm"] for n in now["nodes"] if n["name"] in (left, right)
                     and a < n["station_mm"] < b and n["provenance"] != "interpolated"]
            latest.append({"from_mm": a, "to_mm": b, "parts": [left, right],
                           "legacy": gap["status"], "new_inside_sampled_station_count": len(set(local)),
                           "new_local_outcomes": dict(Counter(matches)),
                           "note": "gap_exceeds_15mm" if b - a > 15 else None})
        report[name] = latest
    return {"input": str(path), "sha256": sha(path), "changes": report}


def run_train(options):
    g, s = import_source(options.project)
    out = options.out
    native = build_native(options.labels)
    target = route_audit(native)
    body = body_data(g, s)
    neutral, _, _, faces, guides = body
    origin, radii = s.fixed_profiles(neutral, faces, guides, STATIONS)
    inside = region_checker(origin, radii, g.bone_prior())
    aligned = g.read_mesh_pack(g.ATLAS_PACK)
    medseg = {name: g.trimesh.Trimesh(*aligned[f"MedSeg|{name}"], process=False)
              for role in s.CT_KEYS for name in role}
    output, graphs = [], {}
    for i, (name, surfaces, groups) in enumerate(
            zip(TRAIN_SOURCES, (*g.training_meshes(), medseg), (g.BP, g.BP, g.OPEN, s.CT_KEYS), strict=True)):
        current = SectionGraph(name, groups, surfaces, "mesh", g.Slicer, inside)
        current.sample()
        current.repair()
        resolve_declared(current, groups, source_contacts(native) if i == 3 else None)
        current.audit_unfilled_gaps()
        current.audit_cross_class_endpoints()
        graphs[name] = current.public()
        output.append(graph_projection(current))
    write_json(out / "ct_reference_topology.json.gz",
               {"coordinates": "neutral SMPL-X mm", "sources": graphs,
                "target_role": 0, "source_local_only": True, "station_step_mm": 1.},
               compress=True)
    write_json(out / "medseg_native_graph.json", native)
    registration = native_shape_projection(native, g, s, body)
    write_native_pack(out, native, registration)
    centers, status, chosen, links = (np.stack(x) for x in zip(*output, strict=True))
    ys, origins, radii, anatomy = shape_project(s, body, STATIONS, centers)
    indices = np.flatnonzero(status[3, :, 0] == STATUS["sampled"])
    np.savez_compressed(out / "ct_geometry.npz", ys=ys, origins=origins, radii=radii, centers=anatomy,
                        status=status, ct_bounds=indices[[0, -1]], neutral_grid=STATIONS,
                        selected_node_ids=chosen, edge_status=links, links=links > 0)
    audit = {"coordinate_axes": "SMPL-X x left, y superior, z anterior; source CT RAS",
             "training_input_sha256": {str(p): sha(p) for p in (g.ATLAS_PACK, g.NEW_OPEN, g.OCCUPANCY,
                         s.SMPL, options.labels, DRIVE / "MedSeg_Pelvic_Vasculature/ct.nii.gz")},
             "blender_vascular_inputs_loaded": False,
             "training_holdout_open_guard_enabled_by_cli": True,
             "native_CT_target": target,
             "native_graph": {"nodes": len(native["nodes"]), "sampled_edges": len(native["sampled_edges"]),
                              "verified_junction_edges": len(native["verified_junction_edges"]),
                              "components": len(components(
                                  [n["id"] for n in native["nodes"]],
                                  [e["nodes"] for e in native["sampled_edges"]
                                   + native["verified_junction_edges"]])),
                              "rejected_edges": len(native["rejected_edges"]),
                              "rejected_reasons": dict(Counter(e["reason"] for e in native["rejected_edges"]))},
             "aligned_legacy_grid_CT_target": {
                 "stations": len(indices), "bounds_mm": STATIONS[indices[[0, -1]]].tolist(),
                 "missing_stations_mm": STATIONS[np.setdiff1d(np.arange(indices[0], indices[-1] + 1), indices)].tolist(),
                 "disconnected_links_at_mm": STATIONS[indices[:-1][links[3, indices[:-1], 0] == 0]].tolist(),
                 "edge_provenance_codes": links[3, indices[:-1], 0].tolist(),
                 "codes": EDGE},
             "shape_count": len(s.BETAS), "scope": "one labelled CT participant, synthetic SMPL-X shapes",
             "sources": {name: graph_summary(value) for name, value in graphs.items()},
             "legacy_gap_comparison": legacy_comparison(options.baseline, graphs)}
    write_json(out / "training_topology_audit.json", audit)
    print(json.dumps({"native_CT_target": audit["native_CT_target"],
                      "aligned_CT_target": audit["aligned_legacy_grid_CT_target"],
                      "sources": audit["sources"]}, ensure_ascii=False, indent=2), flush=True)


def run_holdout(options):
    g, s = import_source(options.project)
    data = body_data(g, s)
    neutral, _, _, faces, guides = data
    origin, radii = s.fixed_profiles(neutral, faces, guides, STATIONS)
    source_path = g.PREVIOUS / "evidence/arterial_paths.json"
    named_paths = load_graph(source_path)
    curves = g.neutral_lines() | {k: np.asarray(v["neutral"]) for k, v in named_paths.items()
                                 if k.startswith("R_")}
    graph = SectionGraph("Blender right held-out", g.BLENDER, curves, "line", g.Slicer,
                         region_checker(origin, radii, g.bone_prior()))
    graph.sample()
    resolve_declared(graph, g.BLENDER)
    graph.line_junction("R_COM_FEM_V", "R_SAPH_V", branch=True)
    graph.audit_unfilled_gaps()
    graph.audit_cross_class_endpoints()
    record = graph.public()
    centers, status, chosen, links = graph_projection(graph)
    offset = int(np.flatnonzero(STATIONS == g.YS[0])[0])
    stop = offset + len(g.YS)
    edge_status = links[offset:stop - 1]
    np.savez_compressed(options.out / "blender_holdout_geometry_links.npz",
                        neutral_grid=STATIONS[offset:stop], centers=centers[offset:stop],
                        status=status[offset:stop], selected_node_ids=chosen[offset:stop],
                        edge_status=edge_status, links=edge_status > 0)
    inputs = [source_path, g.PREVIOUS / "evidence/geometry_stages.npz",
              s.SMPL_DIR / "vessel_centerlines_rest.obj"]
    write_json(options.out / "blender_holdout_topology.json.gz",
               {"coordinates": "neutral SMPL-X mm", "source": record,
                "station_step_mm": 1., "split": "evaluation only",
                "input_sha256": {str(p): sha(p) for p in inputs}}, compress=True)
    write_json(options.out / "blender_holdout_topology_audit.json", graph_summary(record))
    print(json.dumps(graph_summary(record), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "holdout"))
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--out", type=Path, default=OUT)
    options = parser.parse_args()
    options.out.mkdir(exist_ok=True, parents=True)
    if options.mode == "train":
        training_input_guard()
        run_train(options)
    else:
        run_holdout(options)
