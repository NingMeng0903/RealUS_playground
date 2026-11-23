"""Public candidate embeddings for the v2 connected deep-leg routes.

``extract(reader, spec, material=None)`` consumes only the supplied reader
prefix. It returns stable per-frame/slot indices, a compact baseline matrix,
and an extended public-only matrix with masks and names. The label field in
the inherited frame tuple is intentionally never inspected.
"""

from __future__ import annotations

import numpy as np

from connected_reader import _candidate_records, _track_support


REGION_COUNT = 6
ROLE_COUNT = 6

BASELINE_FEATURE_NAMES = (
    "public_evidence", "target_posterior", "unknown_posterior", "merge_posterior",
    "same_frame_rival_separation", "echo_lateral", "echo_depth", "probe_station_fraction", "probe_sector_sin",
    "probe_sector_cos", "probe_tick_fraction", "probe_quality",
)

MATERIAL_FEATURE_NAMES = (
    "material_axial_fraction", "material_cross_section_u", "material_cross_section_v",
    "material_sector_sin", "material_sector_cos", "material_supported",
    "harmonic_theta_sin", "harmonic_theta_cos", "harmonic_h", "harmonic_d",
    "harmonic_supported", "harmonic_registration_error",
)

ATLAS_FEATURE_NAMES = tuple(
    name
    for role in range(ROLE_COUNT)
    for name in (f"atlas_role_{role}_proximity", f"atlas_role_{role}_distance")
) + (
    "atlas_target_hypothesis_count", "atlas_target_hypothesis_entropy",
    "atlas_proximal_link_support", "atlas_distal_link_support", "atlas_same_component_support",
    "atlas_supported_hypothesis_mass",
)

HISTORY_FEATURE_NAMES = (
    "history_track_support", "history_bidirectional_support",
    "history_correlation_discount", "history_distinct_view_count",
    "history_neighbor_frame_count", "history_cross_window_fraction",
)

EXTENDED_FEATURE_NAMES = BASELINE_FEATURE_NAMES + MATERIAL_FEATURE_NAMES + ATLAS_FEATURE_NAMES + HISTORY_FEATURE_NAMES


def _finite(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _route_bounds(spec):
    begin, end = float(spec[1]), float(spec[2])
    return begin, end, max(1.0, end - begin)


def _merge_values(reader, frame_index, slot):
    mixed = getattr(reader, "mixed_posterior", None)
    if mixed is None or frame_index >= len(mixed):
        return 0.0, False
    values = np.asarray(mixed[frame_index], dtype=float).reshape(-1)
    if slot >= len(values) or not np.isfinite(values[slot]):
        return 0.0, False
    return float(np.clip(values[slot], 0.0, 1.0)), True


def _same_frame_separation(reader, frame_index, slot):
    frame = reader.frames[frame_index]
    if slot >= len(frame[2]):
        return 0.0, False
    point = np.asarray(reader.absolute(frame[0], frame[1], frame[2][slot]), dtype=float)
    distances = []
    for other_slot, echo in enumerate(frame[2]):
        if other_slot == slot:
            continue
        try:
            other = np.asarray(reader.absolute(frame[0], frame[1], echo), dtype=float)
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        if other.shape == (2,) and np.isfinite(other).all():
            distances.append(float(np.linalg.norm(point - other)))
    if not distances:
        return 0.0, False
    return float(np.clip(min(distances) / 40.0, 0.0, 2.0)), True


def _material_features(reader, material, record, beam, begin, span):
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
    try:
        contact = adapter.map_pose(station, sector, depth_mm=0.)
        if contact is not None:
            _pos, _normal, tangent = surface.pose(station, sector)
            offset = np.array([tangent[0], 0., tangent[1]]) * beam[0]
            chart = adapter.chart
            xyz = np.array([contact.x_mm + offset @ contact.basis[:, 0],
                            contact.y_mm + offset @ contact.basis[:, 1], beam[1]])
            theta, h, d = np.asarray(chart.xi(xyz)).reshape(-1)
            values[6:] = (np.sin(theta), np.cos(theta), h, d, 1.,
                          min(2., contact.registration_error_mm / 200.))
            mask[6:] = True
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        # A public chart coordinate is available only in its verified ROI.
        pass
    return values, mask


def _atlas_features(reader, record):
    surface = getattr(getattr(reader, "model", None), "surface", None)
    priors = getattr(surface, "priors", None)
    links = getattr(surface, "reference_links", None)
    features = np.zeros(len(ATLAS_FEATURE_NAMES))
    mask = np.zeros(len(features), dtype=bool)
    if priors is None:
        return features, mask
    station = int(np.clip(np.floor(record["station"] + .5), 0, priors.shape[1] - 1))
    points = priors[:, station, :ROLE_COUNT]
    distances = np.where(np.isfinite(points).all(axis=-1),
                         np.linalg.norm(points - record["world"], axis=-1), np.inf)
    finite = np.isfinite(distances)
    for role in range(min(points.shape[1], ROLE_COUNT)):
        if finite[:, role].any():
            nearest = float(distances[:, role].min())
            features[2 * role:2 * role + 2] = np.exp(-.5 * (nearest / 25.) ** 2), nearest / 100.
            mask[2 * role:2 * role + 2] = True
    target = distances[:, 0]
    if not np.isfinite(target).any():
        return features, mask
    affinity = np.exp(-.5 * (target / 22.) ** 2)
    p = affinity / max(affinity.sum(), 1.e-12)
    tail = ROLE_COUNT * 2
    features[tail:tail + 2] = np.isfinite(target).mean(), -p @ np.log(np.maximum(p, 1.e-12)) / np.log(max(len(p), 2))
    mask[tail:tail + 2] = True
    if links is not None:
        proximal = links[:, station - 1, 0] if station > 0 else np.zeros(len(p), dtype=bool)
        distal = links[:, station, 0] if station < links.shape[1] else np.zeros(len(p), dtype=bool)
        features[tail + 2:] = p @ proximal, p @ distal, p @ (proximal & distal), affinity.mean()
        mask[tail + 2:] = True
    return features, mask


def _history_features(records, record_index):
    record = records[record_index]
    views = set()
    neighbors = 0
    cross = 0
    for index, other in enumerate(records):
        if index == record_index or abs(other["station"] - record["station"]) > 3.2:
            continue
        if other["view"] in views:
            continue
        views.add(other["view"])
        neighbors += 1
        cross += other["region"] != record["region"]
    return np.asarray((record.get("track_support", 0.0), record.get("bidirectional", 0.0),
                       record.get("correlation", 0.0), min(1.0, len(views) / 8.0),
                       min(1.0, neighbors / 12.0), float(cross / max(1, neighbors))), dtype=float)


def extract(reader, spec, material=None):
    """Extract baseline and extended public candidate embeddings.

    The returned dictionary contains integer ``frame_index`` and ``slot``
    arrays, float matrices, matching masks, explicit schema names, and no
    evaluator or private-source fields.
    """
    records, _region_stations = _candidate_records(reader, spec)
    _track_support(records)
    begin, _end, span = _route_bounds(spec)
    max_tick = max((_finite(frame[8]) for frame in reader.frames if len(frame) > 8), default=1.0)
    baseline_rows, extended_rows = [], []
    baseline_masks, extended_masks = [], []
    frame_indices, slots, regions = [], [], []
    for index, record in enumerate(records):
        frame = reader.frames[record["frame"]]
        station, sector = record["station"], record["sector"]
        merge, merge_mask = _merge_values(reader, record["frame"], record["slot"])
        separation, separation_mask = _same_frame_separation(reader, record["frame"], record["slot"])
        tick = _finite(frame[8]) if len(frame) > 8 else 0.0
        beam = np.asarray(frame[2][record["slot"]], dtype=float)
        baseline = np.asarray((
            record.get("base", 0.0), record["posterior"], record["unknown"], merge,
            separation, beam[0] / 25., beam[1] / 80., np.clip((station - begin) / span, -1.0, 2.0),
            np.sin(sector * 2.0 * np.pi / 24.0), np.cos(sector * 2.0 * np.pi / 24.0),
            np.clip(tick / max_tick, 0.0, 2.0), record["quality"],
        ), dtype=float)
        baseline_mask = np.ones(len(BASELINE_FEATURE_NAMES), dtype=bool)
        baseline_mask[3] = merge_mask
        baseline_mask[4] = separation_mask
        material_values, material_mask = _material_features(reader, material, record, beam, begin, span)
        atlas_values, atlas_mask = _atlas_features(reader, record)
        history_values = _history_features(records, index)
        extended = np.concatenate((baseline, material_values, atlas_values, history_values))
        extended_mask = np.concatenate((baseline_mask, material_mask, atlas_mask,
                                        np.ones(len(history_values), dtype=bool)))
        baseline_rows.append(baseline); baseline_masks.append(baseline_mask)
        extended_rows.append(extended); extended_masks.append(extended_mask)
        frame_indices.append(record["frame"]); slots.append(record["slot"]); regions.append(record["region"])
    baseline = np.asarray(baseline_rows, dtype=np.float32).reshape(-1, len(BASELINE_FEATURE_NAMES))
    extended = np.asarray(extended_rows, dtype=np.float32).reshape(-1, len(EXTENDED_FEATURE_NAMES))
    result = {
        "frame_index": np.asarray(frame_indices, dtype=np.int32),
        "slot": np.asarray(slots, dtype=np.int16), "region": np.asarray(regions, dtype=np.int8),
        "baseline": baseline,
        "baseline_mask": np.asarray(baseline_masks, dtype=bool).reshape(-1, len(BASELINE_FEATURE_NAMES)),
        "baseline_names": BASELINE_FEATURE_NAMES,
        "extended": extended,
        "extended_mask": np.asarray(extended_masks, dtype=bool).reshape(-1, len(EXTENDED_FEATURE_NAMES)),
        "extended_names": EXTENDED_FEATURE_NAMES,
        "schema": {
            "baseline_dimensions": len(BASELINE_FEATURE_NAMES),
            "extended_dimensions": len(EXTENDED_FEATURE_NAMES),
            "index_semantics": "row i identifies reader.frames[frame_index[i]][2][slot[i]]",
            "prefix_semantics": "only reader.frames supplied by caller are read",
            "label_access": False, "private_source_access": False,
            "material_scope": "slice-normalized candidate coordinates plus masked, projected harmonic shaft ROI",
            "appearance_scope": "anonymous centroid only, B-mode and segmentation appearance unavailable",
        },
    }
    if not np.isfinite(baseline).all() or not np.isfinite(extended).all():
        raise RuntimeError("public embedding contains non-finite values")
    return result


__all__ = ["extract", "BASELINE_FEATURE_NAMES", "MATERIAL_FEATURE_NAMES",
           "ATLAS_FEATURE_NAMES", "HISTORY_FEATURE_NAMES", "EXTENDED_FEATURE_NAMES"]
