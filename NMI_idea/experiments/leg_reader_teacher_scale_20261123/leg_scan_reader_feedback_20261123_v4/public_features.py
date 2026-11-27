"""Corrected public geometry and history features for the v4 reader study.

The v3 reader's public feature contract is retained at 96 columns.  This
module fixes the atlas distance reduction in the public reader and gives the
last six history columns their literal meanings.  It never reads evaluator
labels or a synthetic layout.  A caller that asks for ``legacy=True`` must
pass a replay made with the old reader.  That mode is deliberately kept for a
same-track v3 baseline and is never used to train the corrected reader.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "leg_closed_loop_20261122_v1" / "src"
V2 = ROOT / "leg_scan_target_routes_20261123_v2"
V3 = ROOT / "leg_reader_teacher_scale_20261123_v3"
# Keep the v1 module names ahead of the v2/v3 helpers.  The helpers themselves
# only import the v1 public reader and the v2 embedding constants.
for path in (V1, V2, V3):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fast_embedding import (  # noqa: E402
    ATLAS_FEATURE_NAMES,
    BASELINE_FEATURE_NAMES,
    EXTENDED_FEATURE_NAMES,
    MATERIAL_FEATURE_NAMES,
    _atlas_features_fast,
    _candidate_records_fast,
    _finite,
    _material_features_fast,
    _merge_values,
    _route_bounds,
    _same_frame_separations,
)
from online_reader import PublicHistoryReader  # noqa: E402
from scale_worlds import VectorHistoryReader  # noqa: E402
from connected_reader import _circular_distance  # noqa: E402


REGION_COUNT = 6
SECTORS = 24
SUPPORT_ANGLE = .75
STATIC_STATION = .55
STATIC_ANGLE = .75
BASE = len(BASELINE_FEATURE_NAMES)
TAIL = EXTENDED_FEATURE_NAMES[BASE:]
FEATURES = (BASELINE_FEATURE_NAMES + tuple(f"available_{name}" for name in BASELINE_FEATURE_NAMES)
            + TAIL + tuple(f"available_{name}" for name in TAIL))


class CorrectedPublicHistoryReader(VectorHistoryReader):
    """Public reader whose atlas distances reduce over sources, then roles."""

    corrected_public_geometry = True

    def _new_geometry(self, frame):
        station, sector, observed, _labels, quality, _region, posterior, unknown, _tick = frame
        observed = np.asarray(observed, dtype=float)
        posterior = np.asarray(posterior, dtype=float).reshape(-1)
        unknown = np.asarray(unknown, dtype=float).reshape(-1)
        if not len(observed):
            return np.empty((0, 2)), np.empty(0)
        world = np.asarray([self.absolute(station, sector, xy) for xy in observed], dtype=float)
        surface = self.model.surface
        location = int(np.clip(np.floor(float(station) + .5), 0, len(surface.y) - 1))
        # priors are [source, station, role, xy].  The previous reduction
        # removed the role axis first, which made every candidate inherit the
        # closest source's closest role.  Keep role identity through the
        # source reduction so role 0 is the target and roles 1..4 are rivals.
        prior = np.asarray(surface.priors[:, location, :5], dtype=float)
        valid = np.isfinite(prior).all(axis=-1)
        distance = np.linalg.norm(world[:, None, None] - prior[None], axis=-1)
        distance = np.where(valid[None], distance, 120.)
        role_distance = distance.min(axis=1)
        target = role_distance[:, 0]
        rival = role_distance[:, 1:].min(axis=1) if role_distance.shape[1] > 1 else target + 20.
        absolute = np.exp(-.5 * (target / 17.) ** 2)
        margin = 1. / (1. + np.exp(np.clip((target - rival) / 5., -30., 30.)))
        relation = np.zeros(len(observed), dtype=float)
        if len(observed) > 1:
            pair_distance = np.linalg.norm(world[:, None] - world[None, :], axis=-1)
            affinity = np.exp(-.5 * (pair_distance / 7.5) ** 2)
            affinity[np.diag_indices_from(affinity)] = 0.
            relation = np.clip(affinity.max(axis=1), 0., 1.)
        q = float(np.clip(quality, 0., 1.))
        posterior = np.resize(np.clip(posterior, 0., 1.), len(observed))
        unknown = np.resize(np.clip(unknown, 0., 1.), len(observed))
        score = (.62 * posterior + .08 * absolute + .06 * margin
                 + .02 * relation + .04 * q + .02 * (1. - unknown))
        return world, np.clip(score, 0., 1.)


def _history_features_corrected(records, index, by_bin):
    """Return the six history values with frame/view/window semantics.

    ``distinct_view_count`` counts rounded probe poses.  ``neighbor_frame_count``
    counts unique acquired frame ids, so two candidates in one frame do not
    become two observations.  ``cross_window_fraction`` is the fraction of
    those neighboring frames whose actual probe-sector angle differs by at
    least ``SUPPORT_ANGLE``.  ``correlation_discount`` is one minus the largest
    same-static-view correlation among those unique frames.  This differs from
    the old feature, which used ``1 - track_support`` and compared region ids.
    """
    record = records[index]
    station = float(record["station"])
    nearby = []
    for bucket in range(int(np.floor(station - 3.2)), int(np.floor(station + 3.2)) + 2):
        nearby.extend(by_bin.get(bucket, ()))
    nearby.sort()
    frame_rows = {}
    for other_index in nearby:
        if other_index == index:
            continue
        other = records[other_index]
        if other["frame"] == record["frame"]:
            continue
        if abs(float(other["station"]) - station) > 3.2:
            continue
        # One row per acquired frame.  Candidate slots within a frame have the
        # same public station and sector and are not independent views.
        frame_rows.setdefault(int(other["frame"]), other)

    neighbors = list(frame_rows.values())
    views = {(round(float(other["station"])), round(float(other["sector"])))
             for other in neighbors}
    frame_count = len(neighbors)
    cross_count = sum(
        float(_circular_distance(float(other["sector"]), float(record["sector"]), SECTORS)) >= SUPPORT_ANGLE
        for other in neighbors
    )
    correlations = []
    for other in neighbors:
        ds = abs(float(other["station"]) - station)
        da = float(_circular_distance(float(other["sector"]), float(record["sector"]), SECTORS))
        correlations.append(float(np.exp(-.5 * (ds / STATIC_STATION) ** 2
                                        - .5 * (da / STATIC_ANGLE) ** 2)))
    max_correlation = max(correlations, default=0.)
    return np.asarray((
        float(record.get("track_support", 0.)),
        float(record.get("bidirectional", 0.)),
        1. - max_correlation,
        min(1., len(views) / 8.),
        min(1., frame_count / 12.),
        float(cross_count / frame_count) if frame_count else 0.,
    ), dtype=float)


def _feature_matrix(reader, spec, material, *, legacy):
    """Build a v3-shaped matrix and retain all index/geometry side channels."""
    if legacy:
        if isinstance(reader, CorrectedPublicHistoryReader):
            raise ValueError("legacy=True requires a legacy PublicHistoryReader replay")
        from fast_embedding import extract as legacy_extract

        extracted = legacy_extract(reader, spec, material)
        full, present = extracted["extended"], extracted["extended_mask"]
        features = np.c_[full[:, :BASE], present[:, :BASE], full[:, BASE:], present[:, BASE:]]
        records, _regions, points = _candidate_records_fast(reader, spec)
        point_values = np.asarray([points[r["frame"]][r["slot"]] for r in records], dtype=np.float32).reshape(-1, 2)
        return features.astype(np.float32), extracted, records, point_values

    if not isinstance(reader, CorrectedPublicHistoryReader):
        raise ValueError("corrected public_arrays requires CorrectedPublicHistoryReader or replay_reader")
    records, _regions, frame_points = _candidate_records_fast(reader, spec)
    # The fast candidate routine uses reader.ranked(), which now contains the
    # corrected public geometry.  Its support values remain the established
    # v3 public support graph and are label-free.
    from fast_embedding import _track_support_fast

    _track_support_fast(records)
    begin, _end, span = _route_bounds(spec)
    max_tick = max((_finite(frame[8]) for frame in reader.frames if len(frame) > 8), default=1.)
    by_bin = defaultdict(list)
    for index, record in enumerate(records):
        by_bin[int(np.floor(record["station"]))].append(index)
    baseline_rows, extended_rows = [], []
    baseline_masks, extended_masks = [], []
    frame_indices, slots, regions = [], [], []
    stations, sectors, qualities, points = [], [], [], []
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
        material_values, material_mask = _material_features_fast(
            reader, material, record, beam, begin, span, material_cache
        )
        atlas_values, atlas_mask = _atlas_features_fast(reader, record, atlas_cache)
        history_values = _history_features_corrected(records, index, by_bin)
        extended = np.concatenate((baseline, material_values, atlas_values, history_values))
        extended_mask = np.concatenate((baseline_mask, material_mask, atlas_mask,
                                        np.ones(len(history_values), dtype=bool)))
        baseline_rows.append(baseline)
        baseline_masks.append(baseline_mask)
        extended_rows.append(extended)
        extended_masks.append(extended_mask)
        frame_indices.append(record["frame"])
        slots.append(record["slot"])
        regions.append(record["region"])
        stations.append(station)
        sectors.append(sector)
        qualities.append(frame[4])
        points.append(frame_points[record["frame"]][record["slot"]])
    baseline = np.asarray(baseline_rows, dtype=np.float32).reshape(-1, BASE)
    full = np.asarray(extended_rows, dtype=np.float32).reshape(-1, len(EXTENDED_FEATURE_NAMES))
    masks = np.asarray(extended_masks, dtype=bool).reshape(-1, len(EXTENDED_FEATURE_NAMES))
    extracted = {
        "frame_index": np.asarray(frame_indices, dtype=np.int32),
        "slot": np.asarray(slots, dtype=np.int16),
        "region": np.asarray(regions, dtype=np.int8),
        "baseline": baseline,
        "baseline_mask": np.asarray(baseline_masks, dtype=bool).reshape(-1, BASE),
        "extended": full,
        "extended_mask": masks,
        "baseline_names": BASELINE_FEATURE_NAMES,
        "extended_names": EXTENDED_FEATURE_NAMES,
    }
    features = np.c_[full[:, :BASE], masks[:, :BASE], full[:, BASE:], masks[:, BASE:]]
    point_values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return features.astype(np.float32), extracted, records, point_values


def public_arrays(reader, spec, material=None, *, legacy=False):
    """Return corrected public arrays in the exact v3 96-column order.

    The returned ``features`` contain values and availability bits in v3
    order. ``feature_mask`` is the unflattened 48-column mask. ``frames``,
    ``slots``, ``regions``, ``stations``, ``sectors`` and ``points`` index only
    anonymous candidates.  ``v1_scores`` is the old public evidence column,
    useful for same-track comparisons.
    """
    features, extracted, records, points = _feature_matrix(reader, spec, material, legacy=legacy)
    frame_index = extracted["frame_index"]
    slot = extracted["slot"]
    frame_rows = [reader.frames[int(i)] for i in frame_index]
    return {
        "features": np.asarray(features, dtype=np.float32),
        "feature_mask": np.asarray(extracted["extended_mask"], dtype=bool),
        "baseline": np.asarray(extracted["baseline"], dtype=np.float32),
        "baseline_mask": np.asarray(extracted["baseline_mask"], dtype=bool),
        "frames": frame_index,
        "slots": slot,
        "regions": np.asarray(extracted["region"], dtype=np.int8),
        "stations": np.asarray([row[0] for row in frame_rows], dtype=np.float32),
        "sectors": np.asarray([row[1] for row in frame_rows], dtype=np.float32),
        "points": points,
        "quality": np.asarray([row[4] for row in frame_rows], dtype=np.float32),
        "v1_scores": np.asarray(extracted["baseline"][:, 0], dtype=np.float32),
        "feature_order": tuple(FEATURES),
        "legacy": bool(legacy),
        "schema": {
            "geometry": "distance[candidate, source, role] -> min(source) per role",
            "history_distinct_view_count": "unique rounded station,sector among neighboring frames",
            "history_neighbor_frame_count": "unique acquired frame ids, candidate slots deduplicated",
            "history_cross_window_fraction": "neighbor frames with angular sector delta >= 0.75 / neighbor frames",
            "history_correlation_discount": "1 - max static pose correlation over neighboring frames",
            "label_access": False,
            "private_source_access": False,
        },
    }


def replay_reader(reader, frames, mixed=None, *, legacy=False):
    """Replay anonymous frames into a corrected or legacy public reader.

    ``frames`` may retain the evaluator label slot because it is replaced with
    ``None`` before insertion.  ``mixed`` is the supplied public mixed-posterior
    sequence.  A corrected replay deliberately does not call ``push`` because
    that would regenerate posteriors and change the frozen trajectory.
    """
    if legacy:
        replay = VectorHistoryReader(reader.model, reader.roles, reader.history)
    else:
        replay = CorrectedPublicHistoryReader(reader.model, reader.roles, reader.history)
    replay.w = reader.w.copy()
    replay.state = reader.state
    mixed_values = list(mixed) if mixed is not None else list(getattr(reader, "mixed_posterior", ()))
    if len(mixed_values) != len(frames):
        raise ValueError("mixed posterior count must equal frame count")
    for index, (frame, value) in enumerate(zip(frames, mixed_values, strict=True)):
        if len(frame) < 9:
            raise ValueError("public frame must contain station, sector, echoes, and tick fields")
        public = (*frame[:3], None, *frame[4:])
        replay.frames.append(public)
        replay.mixed_posterior.append(np.asarray(value, dtype=float).copy())
        replay._append_frame(index, public)
    if frames:
        replay.state = (frames[-1][0], frames[-1][1])
    return replay


__all__ = [
    "CorrectedPublicHistoryReader", "FEATURES", "public_arrays", "replay_reader",
    "BASELINE_FEATURE_NAMES", "EXTENDED_FEATURE_NAMES", "MATERIAL_FEATURE_NAMES",
    "ATLAS_FEATURE_NAMES",
]
