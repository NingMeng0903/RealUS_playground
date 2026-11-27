"""Fast, topology-preserving static anatomy variants for training rollouts.

Only the packaged CT/atlas training geometry is read here.  Blender geometry,
holdout labels, and evaluation replay files are intentionally outside this
module's inputs.  A generated layout changes centerline locations while keeping
the finite/support/link masks of its selected training atlas byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


_ROLE_LABELS = {23: 0, 39: 0, 21: 1, 35: 1, 27: 2, 29: 3, 19: 4, 36: 5}
_SOURCE_NAMES = {
    0: "BodyParts3D_right",
    1: "BodyParts3D_mirrored_left",
    2: "Open3DModel_right",
    3: "MedSeg_CT",
}


@dataclass(frozen=True)
class Layout:
    """Immutable centerline layout on the scanner's native station grid."""

    centers: np.ndarray
    identifier: str
    metadata: Mapping
    stations: np.ndarray
    links: np.ndarray
    shared_vertices: tuple = ()

    def __post_init__(self):
        for array in (self.centers, self.stations, self.links):
            array.setflags(write=False)

    def at(self, station):
        """Interpolate only across a supported native link.

        Unsupported gaps return NaN for that role, preserving the original
        topology instead of silently creating a long blended fill.
        """
        value = float(station)
        if value < 0 or value > len(self.stations) - 1:
            return np.full((self.centers.shape[1], 2), np.nan, dtype=np.float64)
        left = int(np.floor(value))
        frac = value - left
        if frac <= 1.e-10 or left == len(self.stations) - 1:
            output = self.centers[left].copy()
        else:
            right = left + 1
            output = np.full((self.centers.shape[1], 2), np.nan, dtype=np.float64)
            finite = np.isfinite(self.centers[left]).all(axis=-1) & np.isfinite(self.centers[right]).all(axis=-1)
            finite &= self.links[left]
            output[finite] = ((1. - frac) * self.centers[left, finite]
                              + frac * self.centers[right, finite])
        for index, roles, point in self.shared_vertices:
            separation = abs(index - value)
            if separation >= .52:
                continue
            weight = (1. - (separation / .52) ** 2) ** 2
            output[roles] += weight * (point - output[roles])
        return output


class LayoutFactory:
    """Create bounded static perturbations from read-only training atlases."""

    def __init__(self, pack: Path, scanner, shape: int):
        self.pack = Path(pack)
        if self.pack.is_dir():
            self.pack = self.pack / "ct_geometry.npz"
        self.scanner = scanner
        self.shape = int(shape)
        self._cache = {}
        with np.load(self.pack, allow_pickle=False) as data:
            self._ct_geometry = {name: np.array(data[name], copy=True) for name in data.files}
        self._ct_geometry["centers"].setflags(write=False)
        self._ct_geometry["links"].setflags(write=False)
        graph_path = self.pack.with_name("medseg_native_graph.json")
        self._native_graph = json.loads(graph_path.read_text()) if graph_path.is_file() else {}
        route_path = self.pack.with_name("medseg_ct_route.npz")
        if route_path.is_file():
            with np.load(route_path, allow_pickle=False) as route:
                self._route = {name: np.array(route[name], copy=True) for name in route.files}
        else:
            self._route = {}
        source_centers = np.asarray(scanner._anatomy.centers, dtype=np.float64)
        source_links = np.asarray(scanner._anatomy.links, dtype=bool)
        if source_centers.ndim != 4 or source_centers.shape[2:] != (6, 2):
            raise ValueError(f"unexpected training center shape {source_centers.shape}")
        self._source_centers = np.array(source_centers, copy=True)
        self._source_links = np.array(source_links, copy=True)
        self._source_centers.setflags(write=False)
        self._source_links.setflags(write=False)
        self.stations = np.asarray(scanner.stations, dtype=np.float64).copy()
        self.stations.setflags(write=False)
        if len(self.stations) != source_centers.shape[1]:
            raise ValueError("layout and scanner station grids differ")
        self._atlas_centers = self._source_centers[:3]
        self._bounds = self._estimate_differential_bounds()
        self._junctions = self._find_junctions()
        self._geometry_reference = [self._geometry_audit(centers, index)
                                    for index, centers in enumerate(self._source_centers)]

    @property
    def source_names(self):
        return dict(_SOURCE_NAMES)

    def _estimate_differential_bounds(self):
        """Estimate robust role-wise displacement envelopes from atlas spread."""
        centers = self._atlas_centers
        bounds = np.full((6, 2), 1.5, dtype=np.float64)
        deviations = []
        # Relative vectors distinguish vessel/artery/saphenous motion from a
        # common chart translation.  The first three roles are prioritized.
        for a, b in ((0, 1), (0, 2), (1, 2), (0, 3), (0, 4)):
            finite = np.isfinite(centers[:, :, a]).all(-1) & np.isfinite(centers[:, :, b]).all(-1)
            if not finite.any():
                continue
            relative = centers[:, :, a] - centers[:, :, b]
            median = np.zeros(relative.shape[1:], dtype=np.float64)
            valid_columns = finite.any(axis=0)
            median[valid_columns] = np.nanmedian(
                np.where(finite[:, valid_columns, None], relative[:, valid_columns], np.nan), axis=0)
            spread = np.abs(relative - median[None])
            spread = spread[finite]
            if len(spread):
                q = np.nanquantile(spread, .9, axis=0)
                bounds[a] = np.maximum(bounds[a], np.minimum(q * .9 + .8, 8.))
                bounds[b] = np.maximum(bounds[b], np.minimum(q * .9 + .8, 8.))
                deviations.append({"roles": [a, b], "q10_q90_abs_mm": q.tolist(),
                                   "samples": int(len(spread))})
        # Use direct atlas deviations where a role is absent from relative pairs.
        for role in range(6):
            finite = np.isfinite(centers).all(-1)[:, :, role]
            if finite.any():
                median = np.zeros(centers[:, :, role].shape[1:], dtype=np.float64)
                valid_columns = finite.any(axis=0)
                median[valid_columns] = np.nanmedian(
                    np.where(finite[:, valid_columns, None], centers[:, valid_columns, role], np.nan), axis=0)
                spread = np.abs(centers[:, :, role] - median[None])[finite]
                if len(spread):
                    bounds[role] = np.maximum(bounds[role], np.minimum(np.nanquantile(spread, .9, axis=0) + .7, 8.))
        return {"role_bounds_mm": bounds, "relative_deviations": deviations}

    def _find_junctions(self):
        """Project supported native CT incidences onto the same shaped surface."""
        nodes = {int(n["id"]): n for n in self._native_graph.get("nodes", [])}
        if not nodes or "shape_stations_mm" not in self._route:
            return []
        shape_axis = self._route["shape_stations_mm"][self.shape]
        physical_y = self.scanner.surface.y
        grouped = {}
        for edge in self._native_graph["verified_junction_edges"]:
            labels = tuple(map(int, edge["labels"]))
            roles = tuple(sorted({_ROLE_LABELS[label] for label in labels}))
            layer = np.mean([nodes[int(value)]["station"] for value in edge["nodes"]])
            centroid_y = np.interp(layer, np.arange(len(shape_axis)), shape_axis)
            station = int(np.argmin(abs(physical_y - centroid_y)))
            entry = grouped.setdefault(labels, [])
            entry.append((station, roles, edge, float(layer)))
        output = []
        low, high = self.scanner.ct_bounds
        finite = np.isfinite(self._source_centers[3]).all(axis=-1)
        for index, (labels, edges) in enumerate(sorted(grouped.items())):
            station, roles, best, layer = max(edges, key=lambda row: row[2]["shared_voxel_faces"])
            nearby = np.arange(max(low, station - 1), min(high, station + 1) + 1)
            eligible = nearby[finite[nearby][:, roles].all(axis=1)]
            supported = bool(len(eligible))
            if supported:
                station = int(eligible[np.argmin(abs(eligible - station))])
            output.append({
                "junction_id": f"medseg_native_{index:02d}", "labels": list(labels), "roles": list(roles),
                "station_index": station, "native_layer": layer, "supported": supported,
                "incident_edges": [{"nodes": row["nodes"], "labels": row["labels"],
                                     "native_shared_faces": row["shared_voxel_faces"],
                                     "synthetic_shared_vertex": f"medseg_native_{index:02d}"}
                                    for _, _, row, _ in edges],
            })
        return output

    def _select_source(self, source, rng):
        if isinstance(source, (int, np.integer)):
            index = int(np.clip(source, 0, len(self._source_centers) - 1))
            kind = "synthetic"
        else:
            name = str(source).lower()
            aliases = {"bodyparts3d_right": 0, "bodyparts_right": 0, "right": 0,
                       "bodyparts3d_mirrored_left": 1, "mirrored_left": 1, "left": 1,
                       "open3dmodel_right": 2, "open3d_right": 2, "open3d": 2,
                       "medseg": 3, "medseg_ct": 3, "ct": 3}
            if name in aliases:
                index, kind = aliases[name], "synthetic"
            elif name in {"ordinary", "reference", "atlas"}:
                index, kind = int(rng.integers(3)), "ordinary_reference"
            else:
                index, kind = int(rng.integers(3)), "synthetic"
        return index, kind

    def _segments(self, finite, links):
        segments, current = [], []
        for station in range(len(finite)):
            if finite[station] and (not current or (station > 0 and links[station - 1])):
                current.append(station)
            else:
                if current:
                    segments.append(current)
                current = [station] if finite[station] else []
        if current:
            segments.append(current)
        return segments

    def _warp(self, base, source, rng, profile):
        links = self._source_links[source]
        finite = np.isfinite(base).all(-1)
        field = np.zeros_like(base)
        factor = 1.25 if profile == "stress" else 1.
        bounds = self._bounds["role_bounds_mm"] * factor
        for role in range(base.shape[1]):
            for component in self._segments(finite[:, role], links[:, role]):
                ix = np.asarray(component, dtype=int)
                if len(ix) == 1:
                    field[ix, role] = rng.normal(0., .35, (1, 2))
                    continue
                knots = np.unique(np.r_[ix[::max(1, len(ix) // 7)], ix[-1]])
                shifts = np.clip(rng.normal(0., .48, (len(knots), 2)), -1., 1.) * bounds[role]
                interval = np.clip(np.searchsorted(knots, ix, side="right") - 1, 0, len(knots) - 2)
                progress = (ix - knots[interval]) / (knots[interval + 1] - knots[interval])
                u = (progress ** 2 * (3. - 2. * progress))[:, None]
                field[ix, role] = (1. - u) * shifts[interval] + u * shifts[interval + 1]
        junctions = []
        if source != 3:
            return field, junctions
        axis = self.scanner.surface.y
        for event in self._junctions:
            station, roles = event["station_index"], event["roles"]
            if not event["supported"]:
                continue
            both = [direction for direction in (-1, 1)
                    if 0 <= station + direction < len(axis)
                    and all(links[min(station, station + direction), role] for role in roles)]
            step = 0. if not both else float(rng.choice(both) * rng.uniform(.12, .40))
            ix = np.arange(max(0, station - 3), min(len(base), station + 4))
            common = field[station, roles].mean(axis=0)
            for role in roles:
                # An axial reparameterization uses only pre-existing support.
                # The local angular offset vanishes at the common junction.
                keep = ix[finite[ix, role]]
                radial = keep - station
                taper = np.maximum(0., 1. - (radial / 3.5) ** 2) ** 2
                source_point = keep - step * taper
                lo, hi = np.floor(source_point).astype(int), np.ceil(source_point).astype(int)
                legal = ((lo >= 0) & (hi < len(base)))
                lo, hi = np.clip(lo, 0, len(base) - 1), np.clip(hi, 0, len(base) - 1)
                legal &= finite[lo, role] & finite[hi, role]
                legal &= (lo == hi) | links[np.minimum(lo, len(links) - 1), role]
                u = (source_point - lo)[:, None]
                shift = (1. - u) * base[lo, role] + u * base[hi, role] - base[keep, role]
                field[keep[legal], role] += shift[legal]
                field[keep, role] += taper[:, None] * (common - field[station, role])
                tangent = rng.uniform(-.50, .50, 2) * (role != roles[0])
                field[keep, role] += (radial * taper)[:, None] * tangent
            original = base[station, roles].mean(axis=0)
            new_index = station + step
            y = float(np.interp(new_index, np.arange(len(axis)), axis))
            vertex = original + common
            junctions.append(event | {
                "original_xyz_mm": [float(original[0]), float(axis[station]), float(original[1])],
                "shared_xyz_mm": [float(vertex[0]), y, float(vertex[1])],
                "moved_station_index": float(new_index), "axial_shift_mm": y - axis[station], "applied": True,
                "representation": "synthetic shared vertex carrying the source incidence records",
            })
        return field, junctions

    def _sample_positions(self, centers, source):
        finite = np.isfinite(centers).all(-1)
        station, role = np.where(finite)
        mid, neighbor = np.where(self._source_links[source] & finite[1:] & finite[:-1])
        low = np.r_[station, mid]
        upper = np.r_[station, mid + 1]
        xz = np.concatenate((centers[station, role],
                             (centers[mid, neighbor] + centers[mid + 1, neighbor]) * .5))
        return low, upper, xz

    def _geometry_audit(self, candidate, source):
        surface = self.scanner.surface
        low, upper, points = self._sample_positions(candidate, source)
        origin, radii, neutral_origin, neutral_radii, neutral_y = surface.shape
        local = points - (origin[low] + origin[upper]) * .5
        angle = (np.arctan2(local[:, 1], local[:, 0]) * 24. / (2. * np.pi)) % 24.
        a = angle.astype(int) % 24
        b, weight = (a + 1) % 24, angle % 1.
        radial = lambda table: ((1. - weight) * (table[low, a] + table[upper, a])
                                + weight * (table[low, b] + table[upper, b])) * .5
        radius = radial(radii)
        outward = np.linalg.norm(local, axis=1) - radius
        xz = (neutral_origin[low] + neutral_origin[upper]) * .5 + local * (radial(neutral_radii) / radius)[:, None]
        y = (neutral_y[low] + neutral_y[upper]) * .5
        occupation = surface.bone(np.column_stack((xz[:, 0], y, xz[:, 1])))
        return outward, occupation

    def _checks(self, base, candidate, source):
        finite = np.isfinite(base).all(-1)
        if not np.array_equal(finite, np.isfinite(candidate).all(-1)):
            return False, {"support_preserved": False}
        old_out, old_bone = self._geometry_reference[source]
        new_out, new_bone = self._geometry_audit(candidate, source)
        old_invalid = old_out > 0.
        new_outside = int(np.sum(new_out > np.maximum(0., old_out) + .08))
        new_conflicts = int(np.sum((old_bone <= .95) & (new_bone > .95)))
        crossings = 0
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 5)):
            both = finite[:, a] & finite[:, b]
            old = np.linalg.norm(base[both, a] - base[both, b], axis=-1)
            new = np.linalg.norm(candidate[both, a] - candidate[both, b], axis=-1)
            crossings += int(np.sum((old > 2.) & (new < .55)))
        return not (new_outside or new_conflicts or crossings), {
            "support_preserved": True, "links_preserved": True, "geometry_samples": len(new_out),
            "new_outside_skin": new_outside, "source_outside_skin": int(old_invalid.sum()),
            "source_severe_bone_conflicts": int(np.sum(old_bone > .95)),
            "new_severe_bone_conflicts": new_conflicts, "new_crossings": crossings,
        }

    def build(self, source, seed: int, profile="ordinary", split="train") -> Layout:
        key = (str(source), int(seed), str(profile), str(split))
        if key in self._cache:
            return self._cache[key]
        split_seed = int.from_bytes(hashlib.sha256(str(split).encode()).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), self.shape, split_seed]))
        source_index, kind = self._select_source(source, rng)
        base = np.array(self._source_centers[source_index], copy=True)
        links = np.array(self._source_links[source_index], copy=True)
        unchanged = kind == "ordinary_reference" or (str(profile).lower() == "ordinary" and rng.random() < .18)
        rejection_reasons = {}
        checks = {"support_preserved": True, "links_preserved": True}
        candidate = base.copy()
        junction_shifts = []
        attempts = 0
        if not unchanged:
            for attempts in range(1, 9):
                field, junction_shifts = self._warp(base, source_index, rng, profile)
                # Keep the same random-varying geometry, reducing only its
                # displacement when a source geometry leaves little room.
                amplitude = 1.
                for amplitude in (1., .72, .45, .22, .10):
                    candidate = base + amplitude * field
                    accepted, check = self._checks(base, candidate, source_index)
                    if accepted:
                        break
                    name = "skin_bone_or_crossing"
                    rejection_reasons[name] = rejection_reasons.get(name, 0) + 1
                if accepted:
                    for event in junction_shifts:
                        old = np.asarray(event["original_xyz_mm"])
                        new = np.asarray(event["shared_xyz_mm"])
                        point = old + amplitude * (new - old)
                        event["shared_xyz_mm"] = point.tolist()
                        event["axial_shift_mm"] = float(point[1] - old[1])
                        s = event["station_index"]
                        event["moved_station_index"] = s + amplitude * (event["moved_station_index"] - s)
                        curves = []
                        for role in event["roles"]:
                            s = event["station_index"]
                            adjacent = [j for j in (s - 1, s + 1)
                                        if 0 <= j < len(base)
                                        and self._source_links[source_index, min(s, j), role]]
                            tangent = []
                            for j in adjacent:
                                x, z = candidate[j, role] - point[[0, 2]]
                                dy = self.scanner.surface.y[j] - point[1]
                                tangent.append(float(np.degrees(np.arctan2(np.hypot(x, z), abs(dy)))))
                            curves.append({"role": role, "attachment_vertex": event["junction_id"],
                                           "local_incidence_degrees": tangent})
                        event["attachments"] = curves
                    checks.update(check | {"accepted_amplitude": amplitude})
                    break
            else:
                candidate = base.copy()
                junction_shifts = []
                unchanged = True
                checks["bounded_retry_fallback_reference"] = True
        if unchanged:
            _, check = self._checks(base, base, source_index)
            checks.update(check)
        candidate = np.asarray(candidate, dtype=np.float64)
        candidate.setflags(write=False)
        links.setflags(write=False)
        finite = np.isfinite(candidate).all(-1)
        junction_payload = json.dumps(junction_shifts, sort_keys=True, separators=(",", ":")).encode()
        payload = (np.ascontiguousarray(candidate).tobytes()
                   + np.ascontiguousarray(links).tobytes() + junction_payload)
        identifier = hashlib.sha256(payload).hexdigest()[:24]
        metadata = {
            "layout_type": "ordinary_reference" if unchanged else "synthetic",
            "source": _SOURCE_NAMES[source_index], "source_index": int(source_index),
            "profile": str(profile), "split": str(split), "shape": int(self.shape),
            "identifier_basis": "static_centers_links_and_moved_junctions_sha256",
            "displacement_limits_mm": np.asarray(self._bounds["role_bounds_mm"]).tolist(),
            "relative_warp_quantiles": self._bounds["relative_deviations"],
            "attempts": int(attempts), "rejections": rejection_reasons,
            "unchanged_reference": bool(unchanged), "finite_counts_by_role": finite.sum(0).astype(int).tolist(),
            "links_true_by_role": links.sum(0).astype(int).tolist(),
            "accepted_checks": checks, "junctions": junction_shifts,
            "junction_provenance": {
                "native_graph_applied_source_indices": [3],
                "atlas_edge_status_inferred_or_confirmed_counts": (
                    np.sum(self._ct_geometry.get("edge_status", np.zeros((4, 0, 6), dtype=np.uint8))[:3] >= 2,
                           axis=(1, 2)).astype(int).tolist()),
            },
            "native_support_preserved": bool(np.array_equal(finite, np.isfinite(base).all(-1))),
            "no_blender_reads": True,
        }
        shared = tuple((event["moved_station_index"], np.asarray(event["roles"]),
                        np.asarray(event["shared_xyz_mm"])[[0, 2]])
                       for event in junction_shifts if len(event["roles"]) > 1)
        layout = Layout(candidate, identifier, metadata, self.stations, links, shared)
        self._cache[key] = layout
        if len(self._cache) > 24:
            self._cache.pop(next(iter(self._cache)))
        return layout
