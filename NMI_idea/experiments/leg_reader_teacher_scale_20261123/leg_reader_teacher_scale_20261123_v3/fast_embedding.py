"""Fast public-only candidate embedding for the v3 scale runs.

This module is a drop-in replacement for the v2 ``public_embedding`` module.
It keeps the v2 feature schema and numerical operations, while avoiding the
most expensive repeated work in a scan prefix.  In particular, echo points
and same-frame separation are computed once per frame, atlas priors are
cached by station, and the local history search uses station windows.

Only anonymous reader frames are consumed.  The inherited label field is
never inspected and no latent layout or evaluator state is accessed.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


# The scale runner can import this module from a fresh Python process.  Keep
# the v2 helpers available even when its directory is not already on sys.path.
_V2_DIR = Path(__file__).resolve().parents[1] / "leg_scan_target_routes_20261123_v2"
if str(_V2_DIR) not in sys.path:
    sys.path.insert(0, str(_V2_DIR))
from connected_reader import _candidate_records as _legacy_candidate_records  # noqa: E402
from connected_reader import _public_evidence as _legacy_public_evidence  # noqa: E402
from connected_reader import _target_prior as _legacy_target_prior  # noqa: E402
from public_embedding import (  # noqa: E402
    ATLAS_FEATURE_NAMES,
    BASELINE_FEATURE_NAMES,
    EXTENDED_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    MATERIAL_FEATURE_NAMES,
    _atlas_features as _legacy_atlas_features,
    _finite,
    _history_features as _legacy_history_features,
    _material_features as _legacy_material_features,
    _merge_values,
    _route_bounds,
    _same_frame_separation as _legacy_same_frame_separation,
)
from connected_reader import (  # noqa: E402
    ATLAS_DISTANCE_MM,
    FRAME_STATION_RADIUS,
    SECTORS,
    STATIC_CORRELATION_MM,
    TRACK_DISTANCE_MM,
    _circular_distance,
    _region_stations,
)


def _frame_points(reader, frame):
    """Transform all anonymous echoes in one frame in one surface call.

    ``Surface.absolute`` is vectorized over the last echo dimension.  A
    conservative fallback retains the scalar v2 behavior for compatible
    reader implementations that do not support vector inputs.
    """
    echoes = frame[2]
    try:
        array = np.asarray(echoes, dtype=float)
        surface = reader.model.surface
        contact, normal, tangent = surface.pose(frame[0], frame[1])
        points = contact[None, :] + array[:, 0, None] * tangent[None, :] \
            - array[:, 1, None] * normal[None, :]
        if points.shape == (len(array), 2):
            return np.asarray(points, dtype=float)
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        pass
    result = []
    for echo in echoes:
        try:
            point = np.asarray(reader.absolute(frame[0], frame[1], echo), dtype=float)
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
            point = np.empty(0, dtype=float)
        result.append(point if point.shape == (2,) else np.empty(0, dtype=float))
    return result


def _candidate_records_fast(reader, spec):
    """v2 candidate records with frame and station-local caches."""
    # Keep the public evidence definition exactly as in v2.  ``ranked`` reads
    # only the public reader state and does not inspect frame labels.
    existing = _legacy_public_evidence(reader)
    frames = list(getattr(reader, "frames", ()))
    records = []
    prior_cache = {}
    frame_point_cache = {}
    for frame_index, frame in enumerate(frames):
        station, sector, echoes = frame[0], frame[1], frame[2]
        region = int(frame[5]) if len(frame) > 5 else -1
        if region < 0 or region >= 6:
            continue
        posterior = np.asarray(frame[6], dtype=float).reshape(-1)
        unknown = np.asarray(frame[7], dtype=float).reshape(-1)
        quality = float(np.clip(frame[4], 0., 1.))
        points = frame_point_cache.setdefault(frame_index, _frame_points(reader, frame))
        station_key = int(np.clip(np.floor(float(station) + .5), 0,
                                  getattr(getattr(reader, "model", None), "surface", frame).priors.shape[1] - 1)) \
            if getattr(getattr(getattr(reader, "model", None), "surface", None), "priors", None) is not None else None
        if station_key not in prior_cache:
            prior_cache[station_key] = _legacy_target_prior(reader, station) if station_key is not None else None
        target = prior_cache[station_key]
        for slot, point in enumerate(points):
            if not isinstance(point, np.ndarray) or point.shape != (2,) or not np.isfinite(point).all():
                continue
            p = float(np.clip(posterior[slot] if slot < len(posterior) else 0., 0., 1.))
            u = float(np.clip(unknown[slot] if slot < len(unknown) else 1., 0., 1.))
            atlas_score = 0.
            if target is not None:
                atlas_score = float(np.exp(-.5 * (np.linalg.norm(target - point, axis=-1).min() / ATLAS_DISTANCE_MM) ** 2))
            records.append({
                "frame": frame_index, "slot": slot, "region": region,
                "station": float(station), "sector": float(sector), "world": point,
                "posterior": p, "unknown": u, "quality": quality,
                "base": float(existing.get((frame_index, slot),
                                           .62 * p + .08 * atlas_score + .04 * quality + .02 * (1. - u))),
                "atlas": atlas_score,
                "view": (round(float(station)), round(float(sector))),
                "track_support": 0., "bidirectional": 0., "correlation": 0.,
            })
    return records, _region_stations(spec), frame_point_cache


def _track_support_fast(records):
    """Compute the v2 support graph using vectorized station-local distances."""
    if not records:
        return
    # The v2 implementation writes these defaults even when a candidate has
    # no admissible neighbors.  Initialize them before the fast early exits.
    for record in records:
        record["track_support"] = 0.0
        record["bidirectional"] = 0.0
        record["correlation"] = 1.0
    station = np.asarray([r["station"] for r in records], dtype=float)
    sector = np.asarray([r["sector"] for r in records], dtype=float)
    world = np.asarray([r["world"] for r in records], dtype=float)
    quality = np.asarray([r["quality"] for r in records], dtype=float)
    posterior = np.asarray([r["posterior"] for r in records], dtype=float)
    bins = defaultdict(list)
    for index, value in enumerate(np.floor(station).astype(int)):
        bins[int(value)].append(index)
    for index, record in enumerate(records):
        neighbors = []
        first = int(np.floor(station[index]))
        for station_bin in range(first - 4, first + 5):
            neighbors.extend(bins.get(station_bin, ()))
        if not neighbors:
            continue
        neighbors = np.asarray(neighbors, dtype=np.int32)
        neighbors = neighbors[neighbors != index]
        if not len(neighbors):
            continue
        ds = np.abs(station[neighbors] - station[index])
        da = np.abs((sector[neighbors] - sector[index] + SECTORS / 2.) % SECTORS - SECTORS / 2.)
        admissible = ((.60 <= ds) & (ds <= FRAME_STATION_RADIUS)) | ((ds <= 2.05) & (da >= .75))
        neighbors = neighbors[admissible]
        ds, da = ds[admissible], da[admissible]
        if not len(neighbors):
            continue
        distance = np.linalg.norm(world[neighbors] - world[index], axis=1)
        admissible = distance <= TRACK_DISTANCE_MM
        neighbors, ds, da, distance = neighbors[admissible], ds[admissible], da[admissible], distance[admissible]
        if not len(neighbors):
            continue
        correlation = np.exp(-.5 * (ds / STATIC_CORRELATION_MM) ** 2 - .5 * (da / .75) ** 2)
        affinity = np.exp(-.5 * (distance / 6.5) ** 2)
        affinity *= .60 + .40 * np.minimum(quality[index], quality[neighbors])
        affinity *= .55 + .45 * np.sqrt(np.minimum(posterior[index], posterior[neighbors]))
        affinity *= 1. - .70 * correlation
        # A view is counted once.  The first occurrence is retained, matching
        # the original ordered set logic and its boundary-view semantics.
        views = set()
        kept = []
        for pos, other in enumerate(neighbors):
            value = float(affinity[pos])
            if not value or not np.isfinite(value):
                continue
            view = records[int(other)]["view"]
            if view in views:
                continue
            views.add(view)
            kept.append((int(other), value))
        if not kept:
            continue
        all_values = [value for _, value in kept]
        left = [value for other, value in kept if station[other] < station[index] - .20]
        right = [value for other, value in kept if station[other] > station[index] + .20]
        record["track_support"] = max(all_values, default=0.)
        record["bidirectional"] = min(max(left, default=0.), max(right, default=0.))
        record["correlation"] = 1. - min(1., max(all_values, default=0.))


def _same_frame_separations(frame_points, frame_index, slot):
    points = frame_points.get(frame_index, ())
    if not isinstance(points, np.ndarray) or points.ndim != 2 or slot >= len(points):
        return _legacy_same_frame_separation
    point = points[slot]
    if point.shape != (2,) or not np.isfinite(point).all():
        return _legacy_same_frame_separation
    others = np.delete(points, slot, axis=0)
    if not len(others):
        return 0., False
    finite = np.isfinite(others).all(axis=1)
    if not finite.any():
        return 0., False
    minimum = float(np.linalg.norm(others[finite] - point, axis=1).min())
    return float(np.clip(minimum / 40., 0., 2.)), True


def _material_features_fast(reader, material, record, beam, begin, span, caches):
    """v2 material features with memoized chart contacts and mappings."""
    station, sector = record["station"], record["sector"]
    surface = getattr(getattr(reader, "model", None), "surface", None)
    values = np.zeros(len(MATERIAL_FEATURE_NAMES), dtype=float)
    mask = np.zeros(len(MATERIAL_FEATURE_NAMES), dtype=bool)
    if surface is None:
        return values, mask
    nearest = int(np.clip(round(station), 0, len(surface.y) - 1))
    center = surface.cross_origin[nearest]
    local = record["world"] - center
    angle = np.arctan2(local[1], local[0])
    if surface.shape is None:
        radius = np.linalg.norm(surface.contact[nearest] - center, axis=1).mean()
    else:
        radii = surface.shape[1]
        angular = (angle * 24. / (2. * np.pi)) % 24.
        lo = int(angular) % 24
        ratio = angular % 1.
        radius = (1. - ratio) * radii[nearest, lo] + ratio * radii[nearest, (lo + 1) % 24]
    values[:6] = ((station - begin) / span, *(local / max(1., radius)),
                  np.sin(angle), np.cos(angle), 1.)
    mask[:6] = True
    adapter = getattr(material, "atlas", material)
    if adapter is None or not hasattr(adapter, "map_pose"):
        return values, mask
    key = (station, sector)
    mapped = caches.setdefault(key, {})
    if "contact" not in mapped:
        try:
            mapped["contact"] = adapter.map_pose(station, sector, depth_mm=0.)
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
            mapped["contact"] = None
    contact = mapped["contact"]
    if contact is None:
        return values, mask
    try:
        if "pose" not in mapped:
            mapped["pose"] = surface.pose(station, sector)
        _pos, _normal, tangent = mapped["pose"]
        offset = np.array([tangent[0], 0., tangent[1]]) * beam[0]
        chart = adapter.chart
        xyz = np.array([contact.x_mm + offset @ contact.basis[:, 0],
                        contact.y_mm + offset @ contact.basis[:, 1], beam[1]])
        theta, h, d = np.asarray(chart.xi(xyz)).reshape(-1)
        values[6:] = (np.sin(theta), np.cos(theta), h, d, 1.,
                      min(2., contact.registration_error_mm / 200.))
        mask[6:] = True
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        pass
    return values, mask


def _atlas_features_fast(reader, record, caches):
    surface = getattr(getattr(reader, "model", None), "surface", None)
    priors = getattr(surface, "priors", None)
    links = getattr(surface, "reference_links", None)
    features = np.zeros(len(ATLAS_FEATURE_NAMES))
    mask = np.zeros(len(features), dtype=bool)
    if priors is None:
        return features, mask
    station = int(np.clip(np.floor(record["station"] + .5), 0, priors.shape[1] - 1))
    if station not in caches:
        caches[station] = np.asarray(priors[:, station, :6])
    points = caches[station]
    distances = np.where(np.isfinite(points).all(axis=-1),
                         np.linalg.norm(points - record["world"], axis=-1), np.inf)
    finite = np.isfinite(distances)
    for role in range(min(points.shape[1], 6)):
        if finite[:, role].any():
            nearest = float(distances[:, role].min())
            features[2 * role:2 * role + 2] = np.exp(-.5 * (nearest / 25.) ** 2), nearest / 100.
            mask[2 * role:2 * role + 2] = True
    target = distances[:, 0]
    if not np.isfinite(target).any():
        return features, mask
    affinity = np.exp(-.5 * (target / 22.) ** 2)
    p = affinity / max(affinity.sum(), 1.e-12)
    tail = 12
    features[tail:tail + 2] = np.isfinite(target).mean(), -p @ np.log(np.maximum(p, 1.e-12)) / np.log(max(len(p), 2))
    mask[tail:tail + 2] = True
    if links is not None:
        proximal = links[:, station - 1, 0] if station > 0 else np.zeros(len(p), dtype=bool)
        distal = links[:, station, 0] if station < links.shape[1] else np.zeros(len(p), dtype=bool)
        features[tail + 2:] = p @ proximal, p @ distal, p @ (proximal & distal), affinity.mean()
        mask[tail + 2:] = True
    return features, mask


def _history_features_fast(records, index, by_bin):
    record = records[index]
    station = record["station"]
    candidates = []
    for value in range(int(np.floor(station - 3.2)), int(np.floor(station + 3.2)) + 2):
        candidates.extend(by_bin.get(value, ()))
    # Preserve the original all-record order, since the first occurrence of a
    # repeated rounded view determines its region contribution.
    candidates.sort()
    views = set()
    neighbors = 0
    cross = 0
    for other_index in candidates:
        other = records[other_index]
        if other_index == index or abs(other["station"] - station) > 3.2:
            continue
        if other["view"] in views:
            continue
        views.add(other["view"])
        neighbors += 1
        cross += other["region"] != record["region"]
    return np.asarray((record.get("track_support", 0.), record.get("bidirectional", 0.),
                       record.get("correlation", 0.), min(1., len(views) / 8.),
                       min(1., neighbors / 12.), float(cross / max(1, neighbors))), dtype=float)


def extract(reader, spec, material=None):
    """Extract the exact v2 public feature schema from an anonymous prefix."""
    records, _region_stations, frame_points = _candidate_records_fast(reader, spec)
    _track_support_fast(records)
    begin, _end, span = _route_bounds(spec)
    max_tick = max((_finite(frame[8]) for frame in reader.frames if len(frame) > 8), default=1.)
    baseline_rows, extended_rows = [], []
    baseline_masks, extended_masks = [], []
    frame_indices, slots, regions = [], [], []
    by_bin = defaultdict(list)
    for index, record in enumerate(records):
        by_bin[int(np.floor(record["station"]))].append(index)
    material_cache, atlas_cache = {}, {}
    for index, record in enumerate(records):
        frame = reader.frames[record["frame"]]
        station, sector = record["station"], record["sector"]
        merge, merge_mask = _merge_values(reader, record["frame"], record["slot"])
        separation, separation_mask = _same_frame_separations(frame_points, record["frame"], record["slot"])
        if callable(separation):
            separation, separation_mask = separation(reader, record["frame"], record["slot"])
        beam = np.asarray(frame[2][record["slot"]], dtype=float)
        baseline = np.asarray((
            record.get("base", 0.), record["posterior"], record["unknown"], merge,
            separation, beam[0] / 25., beam[1] / 80., np.clip((station - begin) / span, -1., 2.),
            np.sin(sector * 2. * np.pi / 24.), np.cos(sector * 2. * np.pi / 24.),
            np.clip(_finite(frame[8]) / max_tick, 0., 2.), record["quality"],
        ), dtype=float)
        baseline_mask = np.ones(len(BASELINE_FEATURE_NAMES), dtype=bool)
        baseline_mask[3] = merge_mask
        baseline_mask[4] = separation_mask
        material_values, material_mask = _material_features_fast(reader, material, record, beam, begin, span, material_cache)
        atlas_values, atlas_mask = _atlas_features_fast(reader, record, atlas_cache)
        history_values = _history_features_fast(records, index, by_bin)
        baseline_rows.append(baseline)
        baseline_masks.append(baseline_mask)
        extended_rows.append(np.concatenate((baseline, material_values, atlas_values, history_values)))
        extended_masks.append(np.concatenate((baseline_mask, material_mask, atlas_mask,
                                               np.ones(len(history_values), dtype=bool))))
        frame_indices.append(record["frame"]); slots.append(record["slot"]); regions.append(record["region"])
    baseline = np.asarray(baseline_rows, dtype=np.float32).reshape(-1, len(BASELINE_FEATURE_NAMES))
    extended = np.asarray(extended_rows, dtype=np.float32).reshape(-1, len(EXTENDED_FEATURE_NAMES))
    result = {
        "frame_index": np.asarray(frame_indices, dtype=np.int32), "slot": np.asarray(slots, dtype=np.int16),
        "region": np.asarray(regions, dtype=np.int8), "baseline": baseline,
        "baseline_mask": np.asarray(baseline_masks, dtype=bool).reshape(-1, len(BASELINE_FEATURE_NAMES)),
        "baseline_names": BASELINE_FEATURE_NAMES, "extended": extended,
        "extended_mask": np.asarray(extended_masks, dtype=bool).reshape(-1, len(EXTENDED_FEATURE_NAMES)),
        "extended_names": EXTENDED_FEATURE_NAMES,
        "schema": {"baseline_dimensions": len(BASELINE_FEATURE_NAMES),
                   "extended_dimensions": len(EXTENDED_FEATURE_NAMES),
                   "index_semantics": "row i identifies reader.frames[frame_index[i]][2][slot[i]]",
                   "prefix_semantics": "only reader.frames supplied by caller are read",
                   "label_access": False, "private_source_access": False,
                   "material_scope": "slice-normalized candidate coordinates plus masked, projected harmonic shaft ROI",
                   "appearance_scope": "anonymous centroid only, B-mode and segmentation appearance unavailable"},
    }
    if not np.isfinite(baseline).all() or not np.isfinite(extended).all():
        raise RuntimeError("public embedding contains non-finite values")
    return result


__all__ = ["extract", "BASELINE_FEATURE_NAMES", "MATERIAL_FEATURE_NAMES",
           "ATLAS_FEATURE_NAMES", "HISTORY_FEATURE_NAMES", "EXTENDED_FEATURE_NAMES"]
