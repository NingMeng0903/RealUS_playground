"""Physically executed forward motion to a public topology anchor.

This module is deliberately a low-level actuator.  It does not choose a
reader, a teacher, or a reward.  A caller supplies a public callback that can
choose one of the legal one-station MPC slots, or return ``"resume"`` to stop
the anchor phase and use the common continuation from the current *actual*
pose.

The chart is the atlas-only two-dimensional material chart used by the v2/v4
experiments.  The public topology is ``surface.reference_links`` with shape
``[source, edge, role]``.  ``find_anchor`` chooses the nominal -750 mm
femoral or -910 mm whole-route seam and requires one source to support the
complete signed route, not merely a local edge.  No anatomy labels, layout
objects, or future observations are passed to the callback.

Public API
----------
``find_anchor(env, spec, ...)`` returns :class:`AnchorTarget`.
``mpc_candidates(env, spec, node, limits)`` returns ``(sectors, legal,
scores)``.  ``scores`` is the exact v1 route score with illegal entries set to
``-inf``; ``argmax(scores)`` is therefore the ordinary MPC slot.
``step_slot(env, spec, node, tape, limits, slot)`` executes one legal chart
step in-place and returns ``(node, record)``.
``run_anchor(...)`` executes the authorized prefix-to-anchor steps, invokes
the public callback at each step, then runs the same MPC tail to the route
end.  It returns ``(node, log)``.

The callback contract is ``choose(context)``.  ``context`` contains only
numeric/public values (latest label-free frame, label-free frame summaries,
posterior, candidate sectors/legal/scores, and remaining budget).  It may
return an integer slot, ``{"slot": slot}``, ``{"sector": absolute_sector}``,
or ``"resume"``.  An illegal choice is rejected before any motion occurs.
``after_anchor`` is an optional hook called only after the real anchor pose
has been reached.  It may return ``None`` (normal tail), a physically produced
``MotionResult``/node, or ``(node, record)``; the hook itself is responsible
for using ``MaterialMotion.run_probe`` if it requests a legal short review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import inspect

import numpy as np


FEMORAL_SEAM_MM = -750.0
WHOLE_SEAM_MM = -910.0
DEFAULT_PREFIX_STATIONS = 4
MIN_PREFIX_STATIONS = 3
MAX_PREFIX_STATIONS = 6


@dataclass(frozen=True)
class AnchorTarget:
    """A public, source-backed named route anchor.

    ``station`` is an integer chart station.  ``source`` is selected from the
    public ``reference_links`` array and is not read from the hidden trial
    layout.  ``prefix_station`` is a recommended starting station upstream of
    the anchor; the actuator still accepts any actual node not past the
    anchor and records the actual one.
    """

    route: str
    name: str
    station: int
    seam_mm: float
    prefix_station: int
    prefix_stations: int
    source: int
    role: int
    route_begin: int
    route_end: int
    signed_start_mm: float
    signed_anchor_mm: float
    signed_end_mm: float
    link_edges: int
    prefix_link_edges: int

    def as_dict(self):
        return asdict(self)


def _infer_route(spec):
    """Infer the two supported route names from the station span."""

    span = int(spec[2]) - int(spec[1])
    return "groin_to_distal_calf" if span >= 70 else "femoral"


def _route_seam(route, seam_mm):
    if seam_mm is not None:
        return float(seam_mm)
    route = str(route)
    if route in {"femoral", "femoral_to_popliteal", "femoral_vein"}:
        return FEMORAL_SEAM_MM
    if route in {"groin_to_distal_calf", "whole", "full", "whole_leg"}:
        return WHOLE_SEAM_MM
    raise ValueError(f"Unknown atlas route {route!r}; provide seam_mm explicitly")


def _public_links(env, role):
    links = getattr(getattr(env.scanner, "surface", None), "reference_links", None)
    if links is None:
        raise ValueError("Atlas public reference_links are unavailable")
    links = np.asarray(links, dtype=bool)
    if links.ndim != 3:
        raise ValueError(f"reference_links must have [source, edge, role], got {links.shape}")
    if not 0 <= int(role) < links.shape[2]:
        raise ValueError(f"Topology role {role} is outside public link array {links.shape}")
    return links


def find_anchor(env, spec, *, route=None, seam_mm=None,
                prefix_stations=DEFAULT_PREFIX_STATIONS, role=0, source=None):
    """Return a source-backed nominal anchor using only public chart data.

    The route is scanned in increasing chart station index.  The atlas neutral
    coordinates must decrease along this direction (proximal/groin to distal),
    which guards against accidentally using an unsigned or reversed anchor.
    Every edge of the requested route must be supported by the same public
    source for ``role``.  This is intentionally stricter than finding a single
    locally supported edge at the nominal seam.
    """

    route = _infer_route(spec) if route is None else str(route)
    prefix_stations = int(prefix_stations)
    if not MIN_PREFIX_STATIONS <= prefix_stations <= MAX_PREFIX_STATIONS:
        raise ValueError("prefix_stations must be between 3 and 6")
    begin, end = int(spec[1]), int(spec[2])
    if end <= begin:
        raise ValueError(f"Invalid forward route bounds {begin}:{end}")
    stations = np.asarray(env.scanner.stations, dtype=float)
    if stations.ndim != 1 or end >= len(stations) or begin < 0:
        raise ValueError("Route bounds fall outside scanner stations")
    seam = _route_seam(route, seam_mm)
    anchor = int(np.argmin(np.abs(stations - seam)))
    if not begin + prefix_stations <= anchor <= end:
        raise ValueError(
            f"Nominal seam {seam:g} maps to station {anchor}, outside route "
            f"{begin}:{end} with a {prefix_stations}-station prefix")
    prefix = anchor - prefix_stations
    # Signed chart orientation is part of the route contract.  Do not silently
    # reinterpret an unsigned coordinate or reverse the route in this helper.
    if not (stations[begin] > stations[end] and stations[prefix] > stations[anchor]):
        raise ValueError("Atlas route is not a signed proximal-to-distal decreasing chart")

    links = _public_links(env, role)
    if links.shape[1] < end:
        raise ValueError(f"reference_links has only {links.shape[1]} edges for route end {end}")
    # Include the full requested route, not only the prefix-to-seam segment.
    route_edges = links[:, begin:end, int(role)]
    candidates = np.flatnonzero(np.all(route_edges, axis=1))
    if source is not None:
        selected = int(source)
        if selected not in candidates:
            raise ValueError(
                f"Requested source {selected} lacks complete public role-{role} "
                f"coverage on route {begin}:{end}; candidates={candidates.tolist()}")
    elif len(candidates):
        selected = int(candidates[0])
    else:
        raise ValueError(
            f"No same-source public role-{role} coverage for signed route {begin}:{end}")
    prefix_edges = links[selected, prefix:anchor, int(role)]
    if len(prefix_edges) != prefix_stations or not bool(np.all(prefix_edges)):
        raise ValueError("Selected public source does not cover the complete prefix-to-anchor span")

    if route in {"femoral", "femoral_to_popliteal", "femoral_vein"}:
        name = "femoral_to_popliteal_anchor"
        canonical_route = "femoral"
    else:
        name = "popliteal_to_posterior_tibial_anchor"
        canonical_route = "groin_to_distal_calf"
    return AnchorTarget(
        route=canonical_route, name=name, station=anchor, seam_mm=seam,
        prefix_station=prefix, prefix_stations=prefix_stations, source=selected,
        role=int(role), route_begin=begin, route_end=end,
        signed_start_mm=float(stations[begin]), signed_anchor_mm=float(stations[anchor]),
        signed_end_mm=float(stations[end]), link_edges=int(route_edges.shape[1]),
        prefix_link_edges=int(prefix_edges.size),
    )


# Descriptive aliases make the intended public-only nature explicit to callers.
public_anchor = find_anchor
anchor_for_route = find_anchor


def _validate_node_bounds(env, spec, node):
    s = int(node.station)
    if not int(spec[1]) <= s <= int(spec[2]):
        raise ValueError(f"Node station {s} is outside route {spec[1]}:{spec[2]}")
    if not np.isfinite(float(node.tick)) or not np.isfinite(float(node.distance)):
        raise ValueError("Node clock or distance is not finite")


def mpc_candidates(env, spec, node, limits):
    """Return the exact common-route MPC candidates at ``node``.

    The score is copied from :meth:`environment.Route.advance`, including its
    one-step lookahead and the ``-.013`` physical distance penalty.  Illegal
    candidates are ``-inf`` in the returned score vector, while ``legal`` is
    returned separately for diagnostics.
    """

    _validate_node_bounds(env, spec, node)
    route = env.route
    s, a = int(node.station), int(node.sector)
    if s >= int(spec[2]):
        raise ValueError("MPC has no forward candidate at the terminal station")
    legal = np.asarray(route.bounded_candidates(node, spec, limits), dtype=bool)
    sectors = np.asarray(route.sectors[a], dtype=int).copy()
    model, w = node.reader.model, node.reader.w
    forecast = route.predictive(model, s + 1)[sectors] @ w
    scores = forecast[:, 0] + .19 * forecast[:, 1] - .013 * route.distance[s, a]
    if s + 1 < int(spec[2]):
        next_detection = route.predictive(model, s + 2)[:, 0] @ w
        future = next_detection[route.sectors[sectors]] - .013 * route.distance[s + 1, sectors]
        forward = np.max(np.where(route.legal[s + 1, sectors], future, -1.), axis=1)
        scores = scores + .43 * forward
    scores = np.where(legal, scores, -np.inf)
    return sectors, legal, np.asarray(scores, dtype=float)


def _path_for_slot(env, spec, node, slot):
    route = env.route
    s, a = int(node.station), int(node.sector)
    slot = int(slot)
    if not 0 <= slot < len(route.deltas):
        raise ValueError(f"MPC slot {slot} is outside 0:{len(route.deltas)}")
    key = s, a, slot
    path = route.cache.get(key)
    if path is None:
        ticks = int(route.clock[s, a, slot])
        if ticks < 1:
            raise RuntimeError(f"Invalid zero-clock route step at {key}")
        fraction = np.arange(ticks + 1, dtype=float) / ticks
        positions = np.column_stack((s + fraction, a + route.deltas[slot] * fraction))
        lengths = env.scanner._path_metrics(positions, spec)
        if lengths is None:
            raise RuntimeError(f"Sampled common path violates chart constraints at {key}")
        path = route.cache[key] = (positions[1:], np.asarray(lengths, dtype=float))
    return path


def step_slot(env, spec, node, tape, limits, slot):
    """Execute one legal slot with the common physical sampling clock.

    The node is mutated exactly as ``Route.advance`` mutates it.  A record is
    returned for the caller's audit ledger; it contains only poses, clock,
    distance, and public candidate scores.
    """

    _validate_node_bounds(env, spec, node)
    route = env.route
    s, a = int(node.station), int(node.sector)
    if s >= int(spec[2]):
        raise ValueError("Cannot step past the terminal station")
    sectors, legal, scores = mpc_candidates(env, spec, node, limits)
    slot = int(slot)
    if not 0 <= slot < len(sectors) or not bool(legal[slot]):
        raise ValueError(
            f"Illegal route slot {slot} at {(s, a)}; legal={legal.astype(int).tolist()}")
    target = int(sectors[slot])
    path = _path_for_slot(env, spec, node, slot)
    positions, lengths = path
    before_tick, before_distance = int(node.tick), float(node.distance)
    # These are the same per-tick observations as Route.advance.  The tape is
    # supplied by the caller and is passed unchanged to the simulator.
    for (station, sector), length in zip(positions, lengths, strict=True):
        node.tick += 1
        node.distance += float(length)
        env.scanner._observe(node, spec, tape, float(station), float(sector), node.tick)
    node.station, node.sector = s + 1, target
    if node.tick > limits.ticks or node.distance > limits.mm + 1.e-6:
        raise RuntimeError("Executed route slot exceeded the shared clock or distance budget")
    if len(node.reader.frames) != node.tick:
        raise RuntimeError("Physical route step dropped an acquisition frame")
    record = {
        "kind": "forward_step", "slot": slot, "from_station": s,
        "from_sector": a, "to_station": int(node.station), "to_sector": target,
        "ticks": int(node.tick - before_tick), "distance_mm": float(node.distance - before_distance),
        "tick_before": before_tick, "tick_after": int(node.tick),
        "distance_before_mm": before_distance, "distance_after_mm": float(node.distance),
        "candidate_sectors": sectors.tolist(), "candidate_legal": legal.astype(bool).tolist(),
        "candidate_scores": scores.tolist(), "path_end": [float(positions[-1, 0]), float(positions[-1, 1])],
    }
    return node, record


def _public_frame(frame):
    """Remove evaluator labels from one internal reader frame."""

    # (station, sector, echoes, labels, quality, region, posterior, unknown, tick)
    return {
        "station": float(frame[0]), "sector": float(frame[1]),
        "echoes": np.asarray(frame[2], dtype=float).copy(),
        "quality": float(frame[4]), "region": int(frame[5]),
        "posterior": np.asarray(frame[6], dtype=float).copy(),
        "unknown": np.asarray(frame[7], dtype=float).copy(),
        "tick": None if frame[8] is None else int(frame[8]),
    }


def _public_context(env, spec, node, limits, anchor, sectors, legal, scores, log):
    reader = node.reader
    frames = tuple(_public_frame(frame) for frame in reader.frames)
    current = {
        "station": int(node.station), "sector": int(node.sector),
        "tick": int(node.tick), "distance_mm": float(node.distance),
    }
    return {
        "node": current, "anchor": anchor.as_dict() if isinstance(anchor, AnchorTarget) else int(anchor),
        "candidate_sectors": np.asarray(sectors, dtype=int).copy(),
        "legal": np.asarray(legal, dtype=bool).copy(),
        "scores": np.asarray(scores, dtype=float).copy(),
        "mpc_slot": int(np.argmax(scores)) if np.any(np.isfinite(scores)) else None,
        "remaining_ticks": int(limits.ticks - node.tick),
        "remaining_distance_mm": float(limits.mm - node.distance),
        "route_end": int(spec[2]), "frames": frames,
        "posterior_weights": np.asarray(reader.w, dtype=float).copy(),
        "decision_count": len(log.get("decisions", ())),
    }


def _call_choose(choose, context):
    if choose is None:
        return "mpc"
    # The documented callback takes one context.  A keyword-compatible
    # callback is convenient for callers and remains label-free.
    try:
        signature = inspect.signature(choose)
    except (TypeError, ValueError):
        return choose(context)
    parameters = list(signature.parameters.values())
    positional = [p for p in parameters if p.kind in
                  (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters) and not positional:
        return choose(**context)
    if len(positional) == 0:
        return choose()
    if len(positional) >= 2:
        # A sanitized current-pose mapping is the first argument; the complete
        # public context remains the second.  Neither exposes reader labels.
        return choose(context["node"], context)
    return choose(context)


def _decision_slot(decision, sectors, legal, mpc_slot):
    if decision is None or decision == "mpc":
        return mpc_slot, "mpc"
    if isinstance(decision, str):
        if decision.lower() in {"resume", "continue", "stop"}:
            return None, decision.lower()
        raise ValueError(f"Unknown anchor decision {decision!r}")
    if isinstance(decision, Mapping):
        action = str(decision.get("action", "slot")).lower()
        if action in {"resume", "continue", "stop"}:
            return None, action
        if "slot" in decision:
            slot = int(decision["slot"])
        elif "sector" in decision or "target_sector" in decision:
            requested = float(decision.get("sector", decision.get("target_sector")))
            # Select the nearest legal candidate, retaining the route's exact
            # five-slot action set and modulo sector geometry.
            delta = np.abs((np.asarray(sectors, dtype=float) - requested + 12.) % 24. - 12.)
            slot = int(np.argmin(np.where(legal, delta, np.inf)))
        else:
            raise ValueError("Anchor decision mapping needs slot, sector, or action")
        return slot, "callback"
    if np.isscalar(decision):
        return int(decision), "callback"
    raise ValueError(f"Unsupported anchor decision type {type(decision).__name__}")


def _apply_hook_result(result, node, limits, spec=None):
    """Normalize an optional physically executed after-anchor hook result."""

    if result is None or isinstance(result, str):
        return node, {"action": "resume" if result else "none"}
    hook_log = None
    candidate = result
    if isinstance(result, Mapping):
        candidate = result.get("node", node)
        hook_log = dict(result.get("log", {}))
    elif isinstance(result, tuple) and len(result) == 2:
        candidate, hook_log = result
    elif hasattr(result, "node") and hasattr(result, "legal"):
        if not bool(result.legal):
            raise RuntimeError(f"after_anchor MaterialMotion result is illegal: {result.reason}")
        candidate = result.node
        hook_log = {"ticks": int(result.ticks), "distance_mm": float(result.distance),
                    "reason": str(result.reason)}
    if not (hasattr(candidate, "reader") and hasattr(candidate, "station")
            and hasattr(candidate, "sector") and hasattr(candidate, "tick")
            and hasattr(candidate, "distance")):
        raise TypeError("after_anchor must return None, a legal MotionResult, a Node, or (Node, log)")
    if int(candidate.tick) < int(node.tick) or float(candidate.distance) + 1.e-7 < float(node.distance):
        raise RuntimeError("after_anchor moved the physical clock or distance backwards")
    if spec is not None and not int(spec[1]) <= int(candidate.station) <= int(spec[2]):
        raise RuntimeError("after_anchor returned a station outside the route")
    if int(candidate.tick) > int(limits.ticks) or float(candidate.distance) > float(limits.mm) + 1.e-6:
        raise RuntimeError("after_anchor exceeded the shared envelope")
    if len(candidate.reader.frames) != int(candidate.tick):
        raise RuntimeError("after_anchor returned an unsynchronized node")
    return candidate, hook_log or {"action": "node"}


def run_anchor(env, spec, node, tape, limits, anchor, choose=None, *, max_extra_ticks=None,
               after_anchor=None):
    """Run real forward steps to ``anchor`` then the common MPC tail.

    ``max_extra_ticks`` bounds only the anchor-phase clock increase from the
    supplied node.  If it would be exceeded, the callback phase ends at the
    current actual pose and the common tail resumes; no pose or clock is
    fabricated.  With ``None`` (the default), the shared route envelope is the
    only bound.
    """

    _validate_node_bounds(env, spec, node)
    target = anchor if isinstance(anchor, AnchorTarget) else None
    target_station = int(anchor.station if target is not None else
                         (anchor["station"] if isinstance(anchor, Mapping) else anchor))
    if not int(spec[1]) <= target_station <= int(spec[2]):
        raise ValueError(f"Anchor station {target_station} is outside route")
    if int(node.station) > target_station:
        raise ValueError("Anchor is upstream of the actual node; refusing to reverse silently")
    if max_extra_ticks is not None and float(max_extra_ticks) < 0:
        raise ValueError("max_extra_ticks must be non-negative or None")
    max_extra_ticks = None if max_extra_ticks is None else int(max_extra_ticks)
    start_tick = int(node.tick)
    log = {
        "anchor": target.as_dict() if target is not None else {"station": target_station},
        "start": {"station": int(node.station), "sector": int(node.sector),
                  "tick": int(node.tick), "distance_mm": float(node.distance)},
        "decisions": [], "steps": [], "tail_steps": [], "reached_anchor": False,
        "resume_reason": None, "after_anchor": None,
    }

    while int(node.station) < target_station:
        sectors, legal, scores = mpc_candidates(env, spec, node, limits)
        mpc_slot = int(np.argmax(scores)) if np.any(np.isfinite(scores)) else None
        context = _public_context(env, spec, node, limits, target or target_station,
                                  sectors, legal, scores, log)
        decision = _call_choose(choose, context)
        slot, reason = _decision_slot(decision, sectors, legal, mpc_slot)
        entry = {"station": int(node.station), "sector": int(node.sector),
                 "decision": decision if isinstance(decision, (str, int, float, type(None))) else dict(decision),
                 "reason": reason, "mpc_slot": mpc_slot}
        if slot is None:
            log["resume_reason"] = reason
            log["decisions"].append(entry)
            break
        if max_extra_ticks is not None:
            expected_ticks = int(env.route.clock[int(node.station), int(node.sector), int(slot)])
            if int(node.tick) - start_tick + expected_ticks > max_extra_ticks:
                log["resume_reason"] = "max_extra_ticks"
                entry["reason"] = "max_extra_ticks"
                log["decisions"].append(entry)
                break
        log["decisions"].append(entry)
        node, step = step_slot(env, spec, node, tape, limits, slot)
        log["steps"].append(step)

    if int(node.station) == target_station:
        log["reached_anchor"] = True
        if after_anchor is not None:
            node, hook_log = _apply_hook_result(
                after_anchor(env, spec, node, tape, limits), node, limits, spec)
            log["after_anchor"] = hook_log

    # The suffix is deliberately Route.advance, not a hidden truth-aware
    # continuation.  This is the exact common tail used by the baseline arm.
    while int(node.station) < int(spec[2]):
        before = (int(node.station), int(node.sector), int(node.tick), float(node.distance))
        env.route.advance(node, spec, tape, limits)
        after = (int(node.station), int(node.sector), int(node.tick), float(node.distance))
        if after[2] <= before[2] or after[0] != before[0] + 1:
            raise RuntimeError("Common tail did not make one physical forward route step")
        log["tail_steps"].append({"from_station": before[0], "from_sector": before[1],
                                  "to_station": after[0], "to_sector": after[1],
                                  "ticks": after[2] - before[2],
                                  "distance_mm": after[3] - before[3]})
    if int(node.station) != int(spec[2]) or int(node.tick) > int(limits.ticks) \
            or float(node.distance) > float(limits.mm) + 1.e-6:
        raise RuntimeError("Anchor continuation exceeded the shared terminal envelope")
    if len(node.reader.frames) != int(node.tick):
        raise RuntimeError("Anchor continuation is not synchronized with physical clock")
    log["end"] = {"station": int(node.station), "sector": int(node.sector),
                   "tick": int(node.tick), "distance_mm": float(node.distance)}
    log["anchor_ticks"] = int(node.tick - start_tick - sum(x["ticks"] for x in log["tail_steps"]))
    log["tail_ticks"] = int(sum(x["ticks"] for x in log["tail_steps"]))
    return node, log


__all__ = ["AnchorTarget", "anchor_for_route", "find_anchor", "public_anchor",
           "mpc_candidates", "step_slot", "run_anchor"]

