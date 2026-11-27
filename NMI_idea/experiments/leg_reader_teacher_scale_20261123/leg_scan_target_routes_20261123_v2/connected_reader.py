"""Public-only connected-path candidate reranking for the v2 leg routes.

``rerank(reader, spec)`` returns six ranked candidate lists.  A candidate is
represented by ``(frame_index, slot, score)``.  The routine consumes the
reader's anonymous frame archive and public ranked evidence only.  It never
indexes the label field at tuple position three and never applies an entry
posterior threshold.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


REGION_COUNT = 6
SECTORS = 24
FRAME_STATION_RADIUS = 3.2
TRACK_DISTANCE_MM = 16.0
ATLAS_DISTANCE_MM = 17.0
STATIC_CORRELATION_MM = .75


def _circular_distance(first, second, period=SECTORS):
    return np.abs((float(first) - float(second) + period / 2.) % period - period / 2.)


def _region_stations(spec):
    """Return station membership for the route's six named regions."""
    begin, end = int(spec[1]), int(spec[2])
    labels = np.asarray(spec[4], dtype=int).reshape(-1)
    stations = np.arange(begin, end + 1, dtype=int)
    if len(labels) != len(stations):
        # A malformed or older spec should still yield a deterministic route
        # partition rather than silently assigning frames by list position.
        labels = np.minimum((np.arange(len(stations)) * REGION_COUNT) // max(1, len(stations)), REGION_COUNT - 1)
    return {region: stations[labels == region] for region in range(REGION_COUNT)}


def _target_prior(reader, station):
    """Get the public target atlas offset at a station, if it is available."""
    surface = getattr(getattr(reader, "model", None), "surface", None)
    priors = getattr(surface, "priors", None)
    if priors is None:
        return None
    values = np.asarray(priors, dtype=float)
    if values.ndim != 4 or values.shape[-1] != 2:
        return None
    location = int(np.clip(np.floor(float(station) + .5), 0, values.shape[1] - 1))
    target = values[:, location, 0]
    target = target[np.isfinite(target).all(axis=-1)]
    if not len(target):
        return None
    return target


def _world_point(reader, frame, slot):
    station, sector, echoes = frame[0], frame[1], frame[2]
    if slot >= len(echoes):
        return None
    try:
        point = np.asarray(reader.absolute(station, sector, echoes[slot]), dtype=float)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    return point if point.shape == (2,) and np.isfinite(point).all() else None


def _public_evidence(reader):
    """Index existing continuous ranking evidence without reading labels."""
    result = {}
    for region in range(REGION_COUNT):
        try:
            frame_ids, slots, evidence = reader.ranked(region)
        except (AttributeError, TypeError, ValueError):
            continue
        for frame, slot, value in zip(frame_ids, slots, evidence):
            result[(int(frame), int(slot))] = float(np.clip(value, 0., 1.))
    return result


def _candidate_records(reader, spec):
    region_stations = _region_stations(spec)
    frame_rows = list(getattr(reader, "frames", ()))
    records = []
    existing = _public_evidence(reader)
    for frame_index, frame in enumerate(frame_rows):
        # Tuple fields are station, sector, anonymous echoes, labels, quality,
        # public region, target posterior, unknown posterior, tick.  Labels
        # are intentionally never read.
        station, sector, echoes = frame[0], frame[1], frame[2]
        region = int(frame[5]) if len(frame) > 5 else -1
        if region < 0 or region >= REGION_COUNT:
            continue
        posterior = np.asarray(frame[6], dtype=float).reshape(-1)
        unknown = np.asarray(frame[7], dtype=float).reshape(-1)
        quality = float(np.clip(frame[4], 0., 1.))
        for slot in range(len(echoes)):
            point = _world_point(reader, frame, slot)
            if point is None:
                continue
            p = float(np.clip(posterior[slot] if slot < len(posterior) else 0., 0., 1.))
            u = float(np.clip(unknown[slot] if slot < len(unknown) else 1., 0., 1.))
            atlas = _target_prior(reader, station)
            atlas_score = 0.
            if atlas is not None:
                atlas_score = float(np.exp(-.5 * (np.linalg.norm(atlas - point, axis=-1).min() / ATLAS_DISTANCE_MM) ** 2))
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
    return records, region_stations


def _track_support(records):
    """Compute local same-track support with static-view deduplication."""
    by_bin = defaultdict(list)
    for index, record in enumerate(records):
        by_bin[int(np.floor(record["station"]))].append(index)
    for index, record in enumerate(records):
        neighbors = []
        for station_bin in range(int(np.floor(record["station"])) - 4,
                                int(np.floor(record["station"])) + 5):
            neighbors.extend(by_bin.get(station_bin, ()))
        left, right, all_support = [], [], []
        views = set()
        for other_index in neighbors:
            if other_index == index:
                continue
            other = records[other_index]
            delta_station = abs(record["station"] - other["station"])
            delta_angle = _circular_distance(record["sector"], other["sector"])
            if not (.60 <= delta_station <= FRAME_STATION_RADIUS or
                    (delta_station <= 2.05 and delta_angle >= .75)):
                continue
            distance = float(np.linalg.norm(record["world"] - other["world"]))
            if distance > TRACK_DISTANCE_MM:
                continue
            # A repeated static view is one evidence packet.  It still enters
            # the maximum, but never accumulates as independent support.
            correlation = float(np.exp(-.5 * (delta_station / STATIC_CORRELATION_MM) ** 2
                                       - .5 * (delta_angle / .75) ** 2))
            affinity = float(np.exp(-.5 * (distance / 6.5) ** 2)
                             * (.60 + .40 * min(record["quality"], other["quality"]))
                             * (.55 + .45 * np.sqrt(min(record["posterior"], other["posterior"]))))
            affinity *= 1. - .70 * correlation
            if not affinity:
                continue
            view = other["view"]
            if view in views:
                continue
            views.add(view)
            all_support.append(affinity)
            if other["station"] < record["station"] - .20:
                left.append(affinity)
            elif other["station"] > record["station"] + .20:
                right.append(affinity)
        # A candidate with evidence on both sides of the route gets a modest
        # coherence bonus.  One-sided support remains useful at route ends.
        left_max = max(left, default=0.)
        right_max = max(right, default=0.)
        record["track_support"] = max(all_support, default=0.)
        record["bidirectional"] = min(left_max, right_max)
        record["correlation"] = 1. - min(1., max(all_support, default=0.))


def _score(record):
    # Existing anonymous evidence is the anchor.  Continuous public posterior,
    # quality, unknown explanation, weak atlas proximity and connected-track
    # support refine it without a hard entry gate.
    value = (
        .52 * record["base"] + .16 * record["posterior"]
        + .07 * record["quality"] + .04 * (1. - record["unknown"])
        + .06 * record["atlas"] + .11 * record["track_support"]
        + .04 * record["bidirectional"]
    )
    return float(np.clip(value, 0., 1.))


def rerank(reader, spec):
    """Return six same-track ranked candidate lists for the connected route.

    The output is ordered by region ``0`` through ``5``.  Every candidate in
    a region remains in that region's list, including low-posterior and
    unknown candidates.  Neighboring regions contribute only through the
    support graph, which prevents a boundary frame from being silently
    relabeled.
    """
    records, _region_stations = _candidate_records(reader, spec)
    if not records:
        return [[] for _ in range(REGION_COUNT)]
    _track_support(records)
    output = [[] for _ in range(REGION_COUNT)]
    for record in records:
        output[record["region"]].append((record["frame"], record["slot"], _score(record)))
    for region in range(REGION_COUNT):
        output[region].sort(key=lambda item: (-item[2], item[0], item[1]))
    return output


__all__ = ["rerank"]
