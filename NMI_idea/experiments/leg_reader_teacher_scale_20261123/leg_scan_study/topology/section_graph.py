"""Build a source-local topology without collapsing rings to classifier roles."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
import math

import numpy as np
from scipy.spatial import cKDTree, distance_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

from native_ct import components

MIN_AREA = 0.6
MAX_SKIP_MM = 12.
GAP_LIMIT_MM = 15.
LOCAL_CONTACT_MM = 0.8


@dataclass
class Section:
    id: int
    xyz: np.ndarray
    name: str
    component: int
    area: float | None
    provenance: str

    def __post_init__(self):
        self.axis = int(round(-self.xyz[1]))

    def public(self):
        return {"id": self.id, "xyz_mm": self.xyz.round(5).tolist(),
                "station_mm": self.axis, "name": self.name,
                "surface_component": self.component, "area_mm2": self.area,
                "provenance": self.provenance}


def vector(a, b):
    return b.xyz - a.xyz


def cosine(u, v):
    return float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1.e-9))


def weld_components(mesh):
    vertices = np.asarray(mesh.vertices)
    _, index = np.unique(np.round(vertices, 4), axis=0, return_inverse=True)
    faces = index[mesh.faces]
    edges = faces[:, [(0, 1), (1, 2), (2, 0)]].reshape(-1, 2)
    n = int(index.max()) + 1
    matrix = coo_matrix((np.ones(len(edges), dtype=bool), edges.T), shape=(n, n))
    _, labels = connected_components(matrix, directed=False, return_labels=True)
    return labels[index]


class MeshEvidence:
    def __init__(self, mesh, slicer):
        self.mesh = mesh
        self.slicer = slicer(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        self.components = weld_components(mesh)
        self.face_labels = self.components[mesh.faces[:, 0]]
        y = mesh.vertices[mesh.faces, 1]
        self.ylo, self.yhi = y.min(axis=1), y.max(axis=1)

    def sections(self, y, area=MIN_AREA):
        return [(r.xyz.copy(), int(self.components[r.edges[0, 0]]), float(r.area))
                for r in self.slicer.at(y, min_area=area) if r.xyz[0] < 0.]

    def support(self, xyz, component, radius):
        tolerance = max(1.5, .65 * radius)
        rings = [(np.linalg.norm(p - xyz), comp) for p, comp, _ in
                 self.sections(xyz[1], area=.06)]
        intact = [d for d, comp in rings if d <= tolerance and comp == component]
        if len(intact) == 1:
            return "closed_source_ring"
        # Missing rings occasionally have open, but still surrounding,
        # source side-wall segments. Points on one side are insufficient.
        faces = np.flatnonzero((self.face_labels == component) &
                               (self.ylo <= xyz[1]) & (self.yhi >= xyz[1]))
        cuts = trimesh.intersections.mesh_plane(
            self.mesh, [0., 1., 0.], [0., xyz[1], 0.], local_faces=faces)
        delta = np.unique(np.round((cuts.reshape(-1, 3) - xyz)[:, [0, 2]], 4), axis=0)
        distances = np.linalg.norm(delta, axis=1)
        near = delta[(distances > .2 * radius) & (distances < max(2.2, 1.65 * radius))]
        theta = np.sort(np.arctan2(near[:, 0], near[:, 1]))
        if len(theta) >= 4 and np.diff(np.r_[theta, theta[0] + 2 * np.pi]).max() < np.pi:
            return "surrounding_source_wall"
        return None


def line_sections(curve, y):
    start, end = curve[:-1], curve[1:]
    hits = (np.minimum(start[:, 1], end[:, 1]) < y) & (
        np.maximum(start[:, 1], end[:, 1]) >= y)
    fraction = (y - start[hits, 1]) / (end[hits, 1] - start[hits, 1])
    return [(xyz, 0, None) for xyz in start[hits] + fraction[:, None] * (end[hits] - start[hits])]


class SectionGraph:
    def __init__(self, source, groups, surfaces, mode, slicer, inside):
        self.source, self.groups, self.mode = source, groups, mode
        self.surfaces, self.inside = surfaces, inside
        self.assets = {name: MeshEvidence(surfaces[name], slicer) for group in groups
                       for name in group if name in surfaces} if mode == "mesh" else {}
        self.nodes, self.edges, self.audit = [], [], []
        self.by_name = defaultdict(list)
        self.before, self.after = {}, {}

    def add(self, xyz, name, component, area, provenance):
        node = Section(len(self.nodes), xyz, name, component, area, provenance)
        self.nodes.append(node)
        self.by_name[name].append(node)
        return node

    def edge(self, a, b, kind, evidence=None, *, track=False):
        item = {"nodes": [a.id, b.id], "provenance": kind}
        if evidence:
            item["evidence"] = evidence
        self.edges.append(item)
        if track:
            self.before[b.id] = a
            self.after[a.id] = b

    def sample(self):
        for group in self.groups:
            for name in group:
                if name not in self.surfaces:
                    continue
                extent = self.assets[name].mesh.vertices if self.mode == "mesh" else self.surfaces[name]
                begin = max(480, int(math.ceil(-extent[:, 1].max())))
                end = min(1280, int(math.floor(-extent[:, 1].min())))
                layers = []
                for axis in range(begin, end + 1):
                    raw = (self.assets[name].sections(-float(axis)) if self.mode == "mesh"
                           else line_sections(self.surfaces[name], -float(axis)))
                    tag = "sampled_mesh_ring" if self.mode == "mesh" else "sampled_source_polyline"
                    layers.append([self.add(x, name, c, area, tag) for x, c, area in raw])
                for left, right in pairwise(layers):
                    self.measured_edges(left, right)

    def measured_edges(self, left, right):
        if not left or not right:
            return
        dist = distance_matrix([n.xyz for n in left], [n.xyz for n in right])
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                limit = 4.0 if self.mode == "mesh" else 4.5
                if a.component != b.component or dist[i, j] > limit:
                    dist[i, j] = np.inf
        for i, a in enumerate(left):
            j = int(dist[i].argmin())
            if not np.isfinite(dist[i, j]) or i != int(dist[:, j].argmin()):
                continue
            b = right[j]
            alt = min(np.delete(dist[i], j).min(initial=np.inf),
                      np.delete(dist[:, j], i).min(initial=np.inf))
            if alt - dist[i, j] < 1.25:
                self.audit.append({"status": "ambiguous_ring_association", "nodes": [a.id, b.id],
                                   "station_mm": a.axis, "name": a.name})
                continue
            if self.mode == "mesh" and dist[i, j] > 3.2:
                mid = (a.xyz + b.xyz) / 2
                radius = math.sqrt(.5 * (a.area + b.area) / math.pi)
                if self.assets[a.name].support(mid, a.component, radius) != "closed_source_ring":
                    self.audit.append({"status": "steep_step_without_midpoint_support",
                                       "nodes": [a.id, b.id], "station_mm": a.axis, "name": a.name})
                    continue
            self.edge(a, b, "sampled", track=True)

    def tangents(self, a, b):
        start, end = a, b
        for _ in range(4):
            start = self.before.get(start.id, start)
            end = self.after.get(end.id, end)
        return (vector(start, a), vector(b, end),
                min(a.axis - start.axis, end.axis - b.axis))

    def direction(self, a, b):
        incoming, outgoing, support = self.tangents(a, b)
        delta = vector(a, b)
        span = b.axis - a.axis
        balance = .5 * (incoming / max(1., -incoming[1]) + outgoing / max(1., -outgoing[1]))
        return {"endpoint_distance_mm": float(np.linalg.norm(delta)),
                "axial_gap_mm": span, "two_sided_sample_span_mm": support,
                "endpoint_tangent_cosines": [cosine(incoming, delta), cosine(outgoing, delta)],
                "lateral_residual_mm": float(np.linalg.norm((delta - span * balance)[[0, 2]]))}

    @staticmethod
    def tangent_ok(stats):
        span = stats["axial_gap_mm"]
        return (stats["two_sided_sample_span_mm"] >= 2
                and min(stats["endpoint_tangent_cosines"]) >= .64
                and stats["endpoint_distance_mm"] <= 1.6 * span + 2
                and stats["lateral_residual_mm"] <= max(4., .6 * span))

    def bridge_proposals(self):
        proposals = []
        for name, rows in self.by_name.items():
            ends = [n for n in rows if n.id not in self.after]
            starts = [n for n in rows if n.id not in self.before]
            for a in ends:
                for b in starts:
                    span = b.axis - a.axis
                    if span < 1 or span > GAP_LIMIT_MM:
                        continue
                    if np.linalg.norm(vector(a, b)) > 1.6 * span + 5:
                        continue
                    item = {"name": name, "nodes": [a.id, b.id], "from_mm": a.axis,
                            "to_mm": b.axis, **self.direction(a, b)}
                    if a.component != b.component:
                        self.audit.append(item | {"status": "disconnected_source_components"})
                    elif not self.tangent_ok(item):
                        self.audit.append(item | {"status": "unsupported_endpoint_direction"})
                    elif self.mode == "mesh" and span <= MAX_SKIP_MM:
                        proposals.append((a, b, item))
        return proposals

    def repair(self):
        proposals = self.bridge_proposals()
        alternatives = Counter(n.id for a, b, _ in proposals for n in (a, b))
        for a, b, item in proposals:
            if alternatives[a.id] > 1 or alternatives[b.id] > 1:
                self.audit.append(item | {"status": "ambiguous_endpoint_match"})
                continue
            u = np.linspace(0., 1., 2 * (b.axis - a.axis) + 1)[1:-1]
            curve = a.xyz + u[:, None] * vector(a, b)
            radius = math.sqrt(.5 * (a.area + b.area) / math.pi)
            support = [self.assets[a.name].support(p, a.component, radius) for p in curve]
            checked = int(sum(s is not None for s in support))
            missing = np.flatnonzero([s is None for s in support])
            blocks = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
            blank = max(map(len, blocks), default=0)
            proof = {"dense_probe_step_mm": .5, "probes": len(curve), "supported_probes": checked,
                     "source_support_types": dict(Counter(s for s in support if s)),
                     "max_blank_run_mm": .5 * blank}
            if checked < .8 * len(curve) or blank > 4:
                self.audit.append(item | proof | {"status": "insufficient_local_source_samples"})
                continue
            if (reason := self.inside(curve)):
                self.audit.append(item | proof | {"status": reason})
                continue
            other_rings = [n for n in self.by_name[a.name] if a.axis < n.axis < b.axis and
                           np.linalg.norm(n.xyz - (a.xyz + (n.axis - a.axis) * vector(a, b) /
                                                   (b.axis - a.axis))) < max(2., 1.2 * radius)]
            if other_rings:
                self.audit.append(item | proof | {"status": "competing_supported_ring"})
                continue
            previous = a
            for xyz in curve[1::2]:
                node = self.add(xyz, a.name, a.component, None, "interpolated")
                self.edge(previous, node, "inferred", proof)
                previous = node
            self.edge(previous, b, "inferred", proof)
            self.audit.append(item | proof | {"status": "inferred_same_identity_bridge"})

    def audit_unfilled_gaps(self):
        for name, rows in self.by_name.items():
            stations = sorted({n.axis for n in rows})
            for a, b in pairwise(stations):
                if b - a > GAP_LIMIT_MM:
                    self.audit.append({"name": name, "from_mm": a, "to_mm": b,
                                       "axial_gap_mm": b - a, "status": "long_gap_same_identity"})

    @staticmethod
    def vascular_class(name):
        if name in {"R_CFA", "R_FA", "R_PA", "R_CALF_M", "R_CALF_L"} or "artery" in name.lower():
            return "artery"
        return "vein"

    def audit_cross_class_endpoints(self):
        ends = [n for n in self.nodes if n.provenance != "interpolated" and n.id not in self.after]
        starts = [n for n in self.nodes if n.provenance != "interpolated" and n.id not in self.before]
        for a in ends:
            for b in starts:
                if not 0 <= b.axis - a.axis <= GAP_LIMIT_MM:
                    continue
                if self.vascular_class(a.name) == self.vascular_class(b.name):
                    continue
                dist = np.linalg.norm(vector(a, b))
                if dist <= GAP_LIMIT_MM:
                    self.audit.append({"left": a.name, "right": b.name, "nodes": [a.id, b.id],
                                       "from_mm": a.axis, "to_mm": b.axis,
                                       "endpoint_distance_mm": float(dist),
                                       "status": "artery_vein_endpoint_candidate"})

    def track_start(self, node):
        while node.id in self.before:
            node = self.before[node.id]
        return node.id

    def line_junction(self, left, right, branch=False):
        a, b = self.surfaces[left], self.surfaces[right]
        if branch:
            gap = np.linalg.norm(a - b[0], axis=1).min()
            point_a = point_b = b[0]
        else:
            point_a, point_b = a[-1], b[0]
            gap = np.linalg.norm(point_a - point_b)
        proof = {"raw_endpoint_gap_mm": float(gap), "left": left, "right": right}
        if gap > 1.e-6:
            status = "long_gap" if gap > GAP_LIMIT_MM else "missing_shared_source_endpoint"
        else:
            a_rows = self.by_name[left]
            b_rows = self.by_name[right]
            if branch:
                a_node = min(a_rows, key=lambda n: np.linalg.norm(n.xyz - point_a))
                b_node = min(b_rows, key=lambda n: np.linalg.norm(n.xyz - point_b))
            else:
                a_node = max(a_rows, key=lambda n: n.axis)
                b_node = min(b_rows, key=lambda n: n.axis)
            proof.update(junction_xyz_mm=point_a.tolist(), nodes=[a_node.id, b_node.id])
            self.edge(a_node, b_node, "source_junction", proof)
            status = "verified_source_branch" if branch else "verified_source_continuation"
        self.audit.append(proof | {"status": status})

    def mesh_junction(self, left, right, native_contacts):
        a_rows, b_rows = self.by_name[left], self.by_name[right]
        a_end = max(n.axis for n in a_rows)
        b_start = min(n.axis for n in b_rows)
        proof = {"left": left, "right": right, "from_mm": a_end, "to_mm": b_start,
                 "station_extent_delta_mm": b_start - a_end}
        if b_start - a_end > GAP_LIMIT_MM:
            self.audit.append(proof | {"status": "long_gap"})
            return
        candidates = []
        for a in a_rows:
            if a.axis < a_end - 12 or a.provenance == "interpolated":
                continue
            for b in b_rows:
                if b.axis > b_start + 12 or b.provenance == "interpolated":
                    continue
                if not 1 <= b.axis - a.axis <= MAX_SKIP_MM:
                    continue
                stat = self.direction(a, b)
                if self.tangent_ok(stat):
                    candidates.append((a, b, stat))
        if not candidates:
            self.audit.append(proof | {"status": "unsupported_join_direction"})
            return
        verified = []
        a_obj, b_obj = self.assets[left], self.assets[right]
        for a, b, stat in candidates:
            low, high = -b.axis - 2, -a.axis + 2
            av, bv = [v.mesh.vertices[(v.components == node.component)
                                   & (v.mesh.vertices[:, 1] >= low)
                                   & (v.mesh.vertices[:, 1] <= high)
                                   & (np.linalg.norm(v.mesh.vertices[:, [0, 2]] -
                                       node.xyz[[0, 2]], axis=1) < 13.)]
                      for v, node in ((a_obj, a), (b_obj, b))]
            if not len(av) or not len(bv):
                continue
            distance, _ = cKDTree(bv).query(av)
            count = int(np.count_nonzero(distance <= LOCAL_CONTACT_MM))
            if count >= 3:
                # At a verified native interface the link must terminate on
                # the bordering slices, even when the ROI centroid shifts.
                score = (10. * stat["axial_gap_mm"] if native_contacts else 0.)
                score += stat["lateral_residual_mm"] + .25 * stat["endpoint_distance_mm"]
                verified.append((score, a, b, stat, count, float(distance.min())))
        if not verified:
            self.audit.append(proof | {"status": "no_local_source_contact"})
            return
        verified.sort(key=lambda row: row[0])
        _, a, b, stat, count, nearest = verified[0]
        # Multiple rings may belong to one connected source mesh. Membership
        # alone cannot establish which lumen continues into the next label.
        chosen_track = (self.track_start(a), self.track_start(b))
        for _, other_a, other_b, _, _, _ in verified[1:]:
            if chosen_track != (self.track_start(other_a), self.track_start(other_b)):
                self.audit.append(proof | {"status": "ambiguous_source_join"})
                return
        record = proof | stat | {"nodes": [a.id, b.id], "local_vertex_pairs_le_0p8mm": count,
                                 "min_local_surface_vertex_gap_mm": nearest}
        if native_contacts:
            labels = tuple(sorted((int(left.split()[0]), int(right.split()[0]))))
            if not (faces := native_contacts.get(labels)):
                self.audit.append(record | {"status": "missing_native_label_contact"})
                return
            record["native_shared_voxel_faces"] = faces
            tag = "source_junction"
        else:
            tag = "inferred_junction"
        if (reason := self.inside(np.linspace(a.xyz, b.xyz, 2 * (b.axis - a.axis) + 1)[1:-1])):
            self.audit.append(record | {"status": reason})
            return
        self.edge(a, b, tag, record)
        self.audit.append(record | {"status": tag})

    def public(self):
        graph = {"source": self.source, "nodes": [n.public() for n in self.nodes],
                 "edges": self.edges, "gaps_and_join_candidates": self.audit,
                 "role_identities": self.groups}
        grouped = components(range(len(self.nodes)), [e["nodes"] for e in self.edges])
        graph["components"] = grouped  # Each list preserves all node IDs.
        return graph
