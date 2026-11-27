"""Construct and query MedSeg topology directly from the expert voxel labels."""

from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy import ndimage

LABELS = {
    19: "right superficial epigastric vein(s)",
    21: "right common femoral artery",
    23: "right common femoral vein",
    27: "right great saphenous vein",
    29: "right deep femoral vein",
    35: "right femoral artery",
    36: "right deep femoral artery",
    39: "right femoral vein",
}
ROLES = ((23, 39), (21, 35), (27,), (29,), (19,), (36,))
DECLARED_TRUNK = {(23, 39), (21, 35)}
DECLARED_BRANCH = {(23, 27), (23, 29), (21, 36)}


def kind(label):
    return "artery" if label in (21, 35, 36) else "vein"


def components(nodes, edges):
    groups = {int(node): int(node) for node in nodes}

    def root(node):
        while node != groups[node]:
            groups[node] = groups[groups[node]]
            node = groups[node]
        return node

    for a, b in edges:
        groups[root(a)] = root(b)
    out = {}
    for node in groups:
        out.setdefault(root(node), []).append(node)
    return sorted(out.values(), key=lambda group: (-len(group), group[0]))


def contact_pairs(a, b):
    both = (a > 0) & (b > 0) & (a != b)
    pairs = np.sort(np.column_stack((a[both], b[both])), axis=1)
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    return zip(unique, counts, strict=True)


def build_native(path):
    import nibabel as nib

    image = nib.load(path)
    voxels = np.asanyarray(image.dataobj)
    nodes, measured, accepted, rejected = [], [], [], []
    # Each connected labelled cross-section is its own node.
    node_volume = np.zeros(voxels.shape, dtype=np.int32)
    for label, name in LABELS.items():
        for station in np.flatnonzero((voxels == label).any(axis=(0, 1))):
            mask = voxels[:, :, station] == label
            regions, count = ndimage.label(mask)
            for region in range(1, count + 1):
                ij = np.argwhere(regions == region)
                xyz = nib.affines.apply_affine(image.affine, [*ij.mean(axis=0), station])
                node = len(nodes) + 1
                node_volume[ij[:, 0], ij[:, 1], station] = node
                nodes.append({"id": node, "station": int(station), "label_id": label,
                              "name": name, "kind": kind(label), "voxel_count": len(ij),
                              "ijk_centroid": [*ij.mean(axis=0).tolist(), int(station)],
                              "xyz_mm": xyz.tolist(), "provenance": "sampled_native_label"})
    counts = Counter()
    for axis in range(3):
        a = np.moveaxis(node_volume, axis, 0)
        for (i, j), count in contact_pairs(a[:-1], a[1:]):
            counts[int(i), int(j)] += int(count)
    for (i, j), count in sorted(counts.items()):
        a, b = nodes[i - 1], nodes[j - 1]
        labels = tuple(sorted((a["label_id"], b["label_id"])))
        record = {"nodes": [i, j], "labels": list(labels), "shared_voxel_faces": count}
        if labels[0] == labels[1]:
            measured.append(record | {"evidence": "same_label_voxel_faces"})
        elif labels in DECLARED_TRUNK:
            accepted.append(record | {"evidence": "declared_trunk_shared_voxel_faces"})
        elif labels in DECLARED_BRANCH:
            accepted.append(record | {"evidence": "declared_side_branch_shared_voxel_faces"})
        else:
            reason = "artery_vein" if a["kind"] != b["kind"] else "undeclared_identity"
            rejected.append(record | {"reason": reason})
    return {"source": "MedSeg CT right", "path": str(path), "coords": "native RAS mm",
            "spacing_mm": [float(x) for x in image.header.get_zooms()[:3]],
            "nodes": nodes, "sampled_edges": measured, "verified_junction_edges": accepted,
            "rejected_edges": rejected}


def ct_route(graph, role=0):
    """Return the unique gap-free ordered native role path or raise ValueError."""
    labels = ROLES[role]
    nodes = sorted((n for n in graph["nodes"] if n["label_id"] in labels),
                   key=lambda n: n["station"])
    if not nodes:
        raise ValueError(f"CT role {role} has no source support")
    by_id = {n["id"]: n for n in nodes}
    relevant = [e for e in graph["sampled_edges"] + graph["verified_junction_edges"]
                if set(e["nodes"]) <= by_id.keys()]
    stations = [n["station"] for n in nodes]
    actual_edges = {tuple(sorted(e["nodes"])) for e in relevant}
    expected_edges = {tuple(sorted((a["id"], b["id"]))) for a, b in pairwise(nodes)}
    if (stations != list(range(stations[0], stations[-1] + 1))
            or actual_edges != expected_edges):
        raise ValueError(f"CT role {role} is ambiguous or disconnected")
    return nodes, relevant


def route_audit(graph):
    nodes, edges = ct_route(graph)
    stations = [n["station"] for n in nodes]
    relevant = [n["id"] for n in nodes]
    span = (stations[-1] - stations[0]) * graph["spacing_mm"][2]
    return {"native_station_bounds": [stations[0], stations[-1]], "station_count": len(stations),
            "axial_span_mm": span, "missing_stations": [], "inferred_nodes": 0,
            "nodes": len(nodes), "edges": len(edges),
            "components": len(components(relevant, [e["nodes"] for e in edges])),
            "label_transition": next(e for e in edges if e["labels"] == [23, 39]),
            "label_voxel_counts": {str(i): sum(n["voxel_count"] for n in nodes
                                               if n["label_id"] == i) for i in (23, 39)}}


def write_native_pack(out, graph, aligned=None):
    nodes, edges = ct_route(graph)
    arrays = dict(native_layer=np.array([n["station"] for n in nodes], dtype=np.int16),
                  voxel_ijk=np.array([n["ijk_centroid"] for n in nodes]),
                  native_ras_mm=np.array([n["xyz_mm"] for n in nodes]),
                  label_id=np.array([n["label_id"] for n in nodes], dtype=np.uint8),
                  shared_voxel_faces=np.array([
                      next(e["shared_voxel_faces"] for e in edges
                           if set(e["nodes"]) == {a["id"], b["id"]})
                      for a, b in pairwise(nodes)], dtype=np.int16))
    if aligned is not None:
        arrays.update(aligned)
    np.savez_compressed(Path(out) / "medseg_ct_route.npz", **arrays)
