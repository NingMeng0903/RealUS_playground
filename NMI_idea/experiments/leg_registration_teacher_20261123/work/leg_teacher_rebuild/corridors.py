"""Budgeted multi-region sweeps on the v2 contact chart. No student is involved."""
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "leg_disambiguation_learning"))
import simulate as scan

# The original runner points at the lab disk. A portable copy can be supplied for replay.
import scan_kernel
if "LEG_BONE_PRIOR" in os.environ:
    scan_kernel.BONE_PRIOR = Path(os.environ["LEG_BONE_PRIOR"])

MODES = ("reference", "nominal", "individual", "pair")
OFFSETS = (-3, -2, -1, 0, 1, 2, 3)


def view_fields(env, centers):
    surface = env.scanner.surface
    angles = np.arange(24) * np.pi / 12.
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    tangents = np.stack([-normals[:, 1], normals[:, 0]], axis=-1)
    local = centers[:, None] - surface.contact[:, :, None, :]
    x = np.sum(local * tangents[None, :, None, :], axis=-1)
    z = -np.sum(local * normals[None, :, None, :], axis=-1)
    inside = (np.abs(x) <= 25.) & (z >= 3.) & (z <= 80.)
    u = np.clip(np.rint((np.nan_to_num(x) + 25.) / 2.5), 0, 20).astype(int)
    v = np.clip(np.rint((np.nan_to_num(z) - 4.) / 2.), 0, 38).astype(int)
    s, a = np.indices(x.shape[:2])
    gate = surface.ray[s[..., None], a[..., None], u, v]
    edge = np.minimum.reduce([25. - np.abs(x), z - 3., 80. - z])
    det = np.where(inside, (.55 + .45 * np.clip(edge / 3., 0., 1.)) * (1. - .96 * gate), 0.)
    d = np.nan_to_num((x[:, :, 1:] - x[:, :, :1])**2 / 2.8**2 +
                      (z[:, :, 1:] - z[:, :, :1])**2 / 2.1**2, nan=1.e4)
    confusable = np.exp(-.5 * d) * det[:, :, 1:]
    isolation = det[:, :, 0] * (1. - confusable.max(axis=-1))
    # A pair can be followed only while both contours are in view and spatially resolved.
    rival = np.argmin(np.nan_to_num(np.linalg.norm(centers[:, 1:] - centers[:, :1], axis=-1),
                                    nan=1.e4), axis=1)
    facing = np.broadcast_to(rival[:, None, None], d.shape[:2] + (1,))
    joint = np.take_along_axis(det[:, :, 1:] * (1. - np.exp(-.5 * d)), facing, axis=-1)[:, :, 0]
    return np.stack([det[:, :, 0], isolation, det[:, :, 0] * joint], axis=-1)


@lru_cache(maxsize=10)
def nominal_fields(env):
    return view_fields(env, env.scanner._anatomy.centers[2])


@lru_cache(maxsize=80)
def scaffold(env, first, last, start_sector):
    route = env.route
    fields = nominal_fields(env)
    tail = np.where(env.scanner.valid[last], 0., -np.inf)
    policy = []
    for s in range(last - 1, first - 1, -1):
        a = route.sectors
        score = .58 * fields[s + 1, a, 0] + .42 * fields[s + 1, a, 1]
        score -= .075 * route.distance[s] + .16 * route.clock[s]
        q = np.where(route.legal[s], tail[a] + score, -np.inf)
        policy.append(np.argmax(q, axis=1))
        tail = np.max(q, axis=1)
    if not np.isfinite(tail[start_sector]):
        raise RuntimeError("No legal scaffold")
    sectors = [start_sector]
    for row in reversed(policy):
        sectors.append(int(route.sectors[sectors[-1], row[sectors[-1]]]))
    return np.asarray(sectors)


@dataclass
class Path:
    sectors: np.ndarray
    slots: np.ndarray
    ticks: int
    distance: float
    offset: int


def path(env, s, sectors, offset=0):
    stations = np.arange(s, s + len(sectors) - 1)
    delta = (np.diff(sectors) + 12) % 24 - 12
    if np.max(np.abs(delta)) > 2:
        return None
    slots = np.asarray([0, 2, 4, 3, 1])[delta]
    a = sectors[:-1]
    route = env.route
    if not route.legal[stations, a, slots].all():
        return None
    return Path(sectors, slots, int(route.clock[stations, a, slots].sum()),
                float(route.distance[stations, a, slots].sum()), offset)


def design(env, spec, start, limits, trial):
    first, last = spec[1:3]
    base = scaffold(env, first, last, int(start.sector))
    nominal, individual = nominal_fields(env), view_fields(env, trial.layout.centers)
    main = path(env, first, base)
    if main is None or main.ticks + start.tick > limits.ticks or main.distance > limits.mm:
        raise RuntimeError("Scaffold exceeds common budget")
    regions = np.array([scan.MOTION.region_of(spec, s) for s in range(first, last + 1)])
    tick_slack = min(20, (limits.ticks - main.ticks - start.tick) // 6)
    mm_slack = (limits.mm - main.distance) / 6.
    plans = {mode: {} for mode in MODES}
    for region in range(6):
        indices = np.flatnonzero(regions == region)
        length = min(9, len(indices) - 1)
        i = int(indices[(len(indices) - length - 1) // 2])
        s = first + i
        reference = path(env, s, base[i:i + length + 1])
        max_ticks = reference.ticks + tick_slack
        targets = np.sin(np.pi * np.arange(length + 1) / length)
        options = []
        for offset in OFFSETS:
            sectors = (reference.sectors + np.rint(offset * targets).astype(int)) % 24
            candidate = path(env, s, sectors, offset)
            if candidate and candidate.ticks <= max_ticks and candidate.distance <= reference.distance + mm_slack:
                options.append(candidate)
        def choose(mode):
            if mode == "reference":
                return reference, length // 2
            field = nominal if mode == "nominal" else individual
            best, selected, hold_at = -np.inf, None, None
            for candidate in options:
                stations = np.arange(s, s + length + 1)
                views = field[stations, candidate.sectors]
                merit = (.18 * views[:, 0] + .32 * views[:, 1] + .50 * views[:, 2]
                         if mode == "pair" else .13 * views[:, 0] + .87 * views[:, 1])
                j = int(merit[1:-1].argmax() + 1)
                quality = .55 * np.mean(merit[1:]) + .45 * merit[j]
                if quality > best or (quality == best and abs(candidate.offset) < abs(selected.offset)):
                    best, selected, hold_at = quality, candidate, j
            return selected, hold_at
        for mode in MODES:
            candidate, at = choose(mode)
            plans[mode][s] = (candidate, at, max_ticks - candidate.ticks, region)
    return base, plans


def observe_hold(env, node, spec, trial, ticks):
    for _ in range(ticks):
        node.tick += 1
        env.scanner._observe(node, spec, trial, float(node.station), float(node.sector), node.tick)


def execute(env, node, trial, spec, limits, base, plan):
    # Each corridor starts and finishes at the same public pose and time for every study arm.
    first = spec[1]
    spans, offsets, ends = [], [], []
    while node.station < spec[2]:
        s = int(node.station)
        if s not in plan:
            i = s - first
            edge = path(env, s, base[i:i + 2])
            scan.ref.reference.apply_route_slot(env, spec, node, trial, limits, int(edge.slots[0]))
            continue
        candidate, at, extra, region = plan[s]
        begin = len(node.reader.frames)
        for j, slot in enumerate(candidate.slots, start=1):
            scan.ref.reference.apply_route_slot(env, spec, node, trial, limits, int(slot))
            if j == at:
                observe_hold(env, node, spec, trial, extra)
        spans.append((begin, len(node.reader.frames)))
        offsets.append([region, int(candidate.offset), int(at), int(extra)])
        ends.append((int(node.tick), int(node.station), int(node.sector)))
    scan.ref.finish(env, node, spec, trial, limits)
    end = node.reader.frames[-1]
    if abs(end[0] - node.station) > 1.e-7 or abs((end[1] - node.sector + 12.) % 24. - 12.) > 1.e-7:
        raise RuntimeError("Terminal frame disagrees with physical endpoint")
    return node.reader.frames, spans, offsets, ends
