"""Current-frame geometry checks for fused SMPL-X hand observations.

The detector/triangulator can occasionally produce a coherent hand cluster
with a bad Body25 wrist, or a wrist with a detached hand cloud.  This module
only lowers confidence on observations that violate the frozen rest-model
geometry.  It never changes coordinates, fills a missing point, or consults a
previous frame/ground truth.

The input layout is ``Body25 + left hand21 + right hand21``.  Body25 wrists
are index 7 (left) and 4 (right); each hand block starts with its wrist/root.
Reach limits are deliberately generous for articulated motion:

* wrist/root reach: ``1.35 * rest path length + 0.020 m``;
* an individual finger edge: ``1.50 * rest edge + 0.025 m``.

When the Body25 wrist is used as the anchor, the model's rest Body25-wrist to
hand-root separation is added to the reach.  These limits reject detached
clusters while retaining bent fingers and global translation/rotation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


BODY_WRIST_INDEX = {"left": 7, "right": 4}
HAND_ROOT_INDEX = {"left": 25, "right": 46}
HAND_SLICE = {"left": slice(25, 46), "right": slice(46, 67)}

# Local hand indices.  Each chain begins at the hand root (local index 0),
# followed by its MCP/PIP/DIP/tip landmarks.
FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
MCP_INDICES = (1, 5, 9, 13, 17)

WRIST_REACH_SCALE = 1.35
WRIST_SLACK_M = 0.020
EDGE_REACH_SCALE = 1.50
EDGE_SLACK_M = 0.025
ANCHOR_DISAGREEMENT_SLACK_M = 0.025
UNANCHORED_KEEP_CONFIDENCE = 0.50
_EPS = 1e-12


def _as_finite_float(value: Any) -> float | None:
    """Return a JSON-safe float, or ``None`` for a non-finite value."""

    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _finite_distance(first: np.ndarray, second: np.ndarray) -> float | None:
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        return None
    value = float(np.linalg.norm(first - second))
    return value if np.isfinite(value) else None


def _rest_paths(rest: np.ndarray, root_index: int) -> tuple[np.ndarray, dict[int, float]]:
    """Build per-local-index rest path lengths and edge-path lookup.

    A finger's reach is measured along its kinematic path instead of by the
    straight root-to-tip distance.  This is what keeps a valid flexed finger
    from being rejected merely because its endpoint is folded back toward the
    palm.
    """

    local_rest = rest[root_index : root_index + 21]
    path = np.full(21, np.nan, dtype=np.float64)
    path[0] = 0.0
    edge_paths: dict[int, float] = {}
    for chain in FINGER_CHAINS:
        previous = 0
        accumulated = 0.0
        for local_index in chain:
            edge = _finite_distance(local_rest[previous], local_rest[local_index])
            if edge is None:
                accumulated = float("nan")
            elif np.isfinite(accumulated):
                accumulated += edge
            path[local_index] = accumulated
            if edge is not None:
                edge_paths[(previous << 8) | local_index] = edge
            previous = local_index
    return path, edge_paths


def _path_between(
    local_a: int,
    local_b: int,
    *,
    local_rest: np.ndarray,
) -> float | None:
    """Return the rest length along one finger chain from ``a`` to ``b``."""

    if local_a == local_b:
        return 0.0
    for chain in FINGER_CHAINS:
        sequence = (0,) + chain
        try:
            start = sequence.index(local_a)
            end = sequence.index(local_b)
        except ValueError:
            continue
        if start > end:
            return None
        total = 0.0
        for before, after in zip(sequence[start:end], sequence[start + 1 : end + 1]):
            edge = _finite_distance(local_rest[before], local_rest[after])
            if edge is None:
                return None
            total += edge
        return total
    return None


def _anchor_score(
    points: np.ndarray,
    valid: np.ndarray,
    *,
    anchor_index: int,
    hand_start: int,
    path_lengths: np.ndarray,
    base_offset: float,
) -> tuple[float, int]:
    """Confidence-weighted in-reach count and MCP support for an anchor."""

    if not valid[anchor_index]:
        return 0.0, 0
    anchor = points[anchor_index, :3]
    score = 0.0
    mcp_support = 0
    for local_index in range(1, 21):
        global_index = hand_start + local_index
        if not valid[global_index] or not np.isfinite(path_lengths[local_index]):
            continue
        distance = _finite_distance(points[global_index, :3], anchor)
        if distance is None:
            continue
        limit = base_offset + WRIST_REACH_SCALE * float(path_lengths[local_index]) + WRIST_SLACK_M
        if distance <= limit:
            confidence = float(np.clip(points[global_index, 3], 0.0, 1.0))
            score += confidence
            if local_index in MCP_INDICES:
                mcp_support += 1
    return score, mcp_support


def _valid_current(points: np.ndarray, min_confidence: float) -> np.ndarray:
    confidence = points[:, 3]
    return (
        np.isfinite(points[:, :3]).all(axis=1)
        & np.isfinite(confidence)
        & (confidence > min_confidence)
    )


def filter_hand_observations(
    points: np.ndarray,
    rest_landmarks: np.ndarray,
    *,
    min_confidence: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mask detached current-frame hand observations using frozen rest geometry.

    Parameters
    ----------
    points:
        A ``(67, 4)`` array containing xyz and confidence.  The returned array
        is a copy.  Rejected observations retain their xyz coordinates and get
        confidence zero.
    rest_landmarks:
        A ``(67, 3)`` frozen SMPL-X landmark layout for the subject's beta.
        It supplies only rest distances; it is never compared to the current
        world position.
    min_confidence:
        Minimum confidence for an observation to participate in anchor and
        reach checks.  Missing/non-finite observations are handled as absent.

    Returns
    -------
    filtered, metadata:
        ``filtered`` is a confidence-masked copy.  ``metadata`` contains only
        JSON-safe values, including a 67-element rejected mask and reasons.
    """

    points = np.asarray(points)
    rest = np.asarray(rest_landmarks)
    if points.shape != (67, 4):
        raise ValueError(f"Expected points shape (67, 4), got {points.shape}")
    if rest.shape != (67, 3):
        raise ValueError(f"Expected rest_landmarks shape (67, 3), got {rest.shape}")
    if not np.isfinite(min_confidence) or min_confidence < 0.0:
        raise ValueError("min_confidence must be finite and non-negative")
    # A non-finite rest model cannot define a geometry policy.  Current-frame
    # NaNs, by contrast, are expected and are dealt with as missing below.
    if not np.isfinite(rest).all():
        raise ValueError("rest_landmarks must be finite")

    filtered = np.array(points, copy=True)
    raw_valid = _valid_current(points, float(min_confidence))
    # ``valid`` is the effective state after an unsupported anchor is masked;
    # ``raw_valid`` remains the original observation state for provenance.
    valid = raw_valid.copy()
    # Make missing confidence explicit in the returned observation.  xyz is
    # deliberately left untouched so diagnostics can still inspect NaNs.
    filtered[~raw_valid, 3] = 0.0
    rejected = np.zeros(67, dtype=bool)
    reasons = ["accepted" if bool(value) else "input_missing_or_low_confidence" for value in raw_valid]
    hands: dict[str, dict[str, Any]] = {}

    for side in ("left", "right"):
        body_index = BODY_WRIST_INDEX[side]
        root_index = HAND_ROOT_INDEX[side]
        hand_start = HAND_SLICE[side].start
        local_rest = rest[hand_start : hand_start + 21]
        path_lengths, _ = _rest_paths(rest, hand_start)
        rest_separation = _finite_distance(rest[body_index], rest[root_index])
        # The supplied rest model is finite, but retain a defensive branch so
        # metadata never contains NaN if this helper is reused with odd models.
        if rest_separation is None:
            rest_separation = 0.0

        body_input_valid = bool(raw_valid[body_index])
        root_input_valid = bool(raw_valid[root_index])
        valid_mcp_count = int(sum(bool(raw_valid[hand_start + index]) for index in MCP_INDICES))
        body_score = body_mcp_support = 0.0
        root_score = root_mcp_support = 0
        if body_input_valid:
            body_score, body_mcp_support = _anchor_score(
                points,
                raw_valid,
                anchor_index=body_index,
                hand_start=hand_start,
                path_lengths=path_lengths,
                base_offset=rest_separation,
            )
        if root_input_valid:
            root_score, root_mcp_support = _anchor_score(
                points,
                raw_valid,
                anchor_index=root_index,
                hand_start=hand_start,
                path_lengths=path_lengths,
                base_offset=0.0,
            )

        body_anchor_unusable = bool(body_input_valid and body_score <= _EPS and valid_mcp_count >= 3)
        root_anchor_unusable = bool(root_input_valid and root_score <= _EPS and valid_mcp_count >= 3)
        anchor_unusable = {
            body_index: body_anchor_unusable,
            root_index: root_anchor_unusable,
        }
        for anchor_index, unusable in anchor_unusable.items():
            if not unusable:
                continue
            # Mask the candidate before anchor selection.  The raw confidence
            # is retained in metadata below; xyz is left unchanged.
            valid[anchor_index] = False
            filtered[anchor_index, 3] = 0.0
            rejected[anchor_index] = True
            reasons[anchor_index] = "unsupported_wrist_anchor"

        body_valid = bool(valid[body_index])
        root_valid = bool(valid[root_index])
        gap = _finite_distance(points[body_index, :3], points[root_index, :3]) if body_valid and root_valid else None
        agreement_limit = WRIST_REACH_SCALE * rest_separation + ANCHOR_DISAGREEMENT_SLACK_M
        anchor_choice: str
        anchor_reason: str
        if body_valid and root_valid:
            if gap is not None and gap <= agreement_limit:
                # The hand-root coordinate is the more direct hand anchor when
                # both wrist measurements describe the same current cluster.
                anchor_choice = "hand_root"
                anchor_reason = "anchors_agree_use_hand_root"
            elif root_score > body_score or (
                abs(root_score - body_score) <= _EPS
                and float(points[root_index, 3]) >= float(points[body_index, 3])
            ):
                anchor_choice = "hand_root"
                anchor_reason = "higher_confidence_weighted_hand_score"
            else:
                anchor_choice = "body_wrist"
                anchor_reason = "higher_confidence_weighted_body_score"
        elif root_valid:
            anchor_choice = "hand_root"
            anchor_reason = "only_hand_root_anchor_valid"
        elif body_valid:
            anchor_choice = "body_wrist"
            anchor_reason = "only_body_wrist_anchor_valid"
        else:
            anchor_choice = "unanchored"
            anchor_reason = "both_body_wrist_and_hand_root_missing"

        hand_global = np.arange(hand_start, hand_start + 21, dtype=np.int64)
        hand_rejected = np.zeros(21, dtype=bool)
        # Keep the effective rejection state visible to the hand-level
        # accounting and chain walk.  ``reasons`` still carries the dedicated
        # unsupported-anchor reason for this root.
        hand_rejected[0] = root_anchor_unusable
        hand_reasons = [reasons[index] for index in hand_global]

        if anchor_choice == "unanchored":
            # Keep strong isolated measurements without pretending that one is
            # a missing wrist/root.  The caller can decide whether to use them.
            for local_index, global_index in enumerate(hand_global):
                if not valid[global_index]:
                    continue
                if float(points[global_index, 3]) < UNANCHORED_KEEP_CONFIDENCE:
                    hand_rejected[local_index] = True
                    hand_reasons[local_index] = "unanchored_low_confidence_suppressed"
                else:
                    hand_reasons[local_index] = "unanchored_high_confidence_retained"
        else:
            anchor_index = root_index if anchor_choice == "hand_root" else body_index
            anchor = points[anchor_index, :3]
            base_offset = 0.0 if anchor_choice == "hand_root" else rest_separation
            for local_index, global_index in enumerate(hand_global):
                if not valid[global_index] or not np.isfinite(path_lengths[local_index]):
                    continue
                distance = _finite_distance(points[global_index, :3], anchor)
                limit = base_offset + WRIST_REACH_SCALE * float(path_lengths[local_index]) + WRIST_SLACK_M
                if distance is None or distance > limit:
                    hand_rejected[local_index] = True
                    hand_reasons[local_index] = "outside_anchor_reach"

            # A bad MCP must not automatically discard valid distal points.  A
            # chain is checked against its nearest accepted ancestor, summing
            # rest edges when an intervening ancestor was rejected.
            for chain in FINGER_CHAINS:
                sequence = (0,) + chain
                for position in range(1, len(sequence)):
                    local_index = sequence[position]
                    if hand_rejected[local_index] or not valid[hand_global[local_index]]:
                        continue
                    nearest = None
                    for prior_position in range(position - 1, -1, -1):
                        candidate = sequence[prior_position]
                        if not hand_rejected[candidate] and valid[hand_global[candidate]]:
                            nearest = candidate
                            break
                    if nearest is None:
                        # Anchor reach remains the only safe test when no
                        # observed ancestor is available.
                        continue
                    rest_path = _path_between(nearest, local_index, local_rest=local_rest)
                    if rest_path is None:
                        continue
                    distance = _finite_distance(
                        points[hand_global[local_index], :3],
                        points[hand_global[nearest], :3],
                    )
                    limit = EDGE_REACH_SCALE * rest_path + EDGE_SLACK_M
                    if distance is None or distance > limit:
                        hand_rejected[local_index] = True
                        hand_reasons[local_index] = "outside_accepted_ancestor_chain_reach"

        for local_index, global_index in enumerate(hand_global):
            reasons[global_index] = hand_reasons[local_index]
            if hand_rejected[local_index]:
                rejected[global_index] = True
                filtered[global_index, 3] = 0.0

        body_wrist_rejected = False
        if (
            anchor_choice == "hand_root"
            and body_valid
            and root_valid
            and gap is not None
            and gap > agreement_limit
            and root_mcp_support >= 3
        ):
            # Preserve the coherent hand root and mask only the competing
            # Body25 wrist.  Coordinates remain untouched for diagnostics.
            filtered[body_index, 3] = 0.0
            rejected[body_index] = True
            reasons[body_index] = "disagreeing_body_wrist_rejected_for_coherent_hand_root"
            body_wrist_rejected = True

        hands[side] = {
            "anchor_choice": anchor_choice,
            "anchor_reason": anchor_reason,
            "unanchored": anchor_choice == "unanchored",
            "body_wrist_index": body_index,
            "hand_root_index": root_index,
            # ``*_valid`` is the effective post-guard state.  The input state
            # and raw confidence remain explicit for diagnostics.
            "body_wrist_valid": body_valid,
            "hand_root_valid": root_valid,
            "body_wrist_input_valid": body_input_valid,
            "hand_root_input_valid": root_input_valid,
            "body_wrist_confidence": _as_finite_float(points[body_index, 3]),
            "hand_root_confidence": _as_finite_float(points[root_index, 3]),
            "body_wrist_raw_confidence": _as_finite_float(points[body_index, 3]),
            "hand_root_raw_confidence": _as_finite_float(points[root_index, 3]),
            "body_wrist_anchor_unusable": body_anchor_unusable,
            "hand_root_anchor_unusable": root_anchor_unusable,
            "anchor_unusable": {
                "body_wrist": body_anchor_unusable,
                "hand_root": root_anchor_unusable,
            },
            "anchor_unusable_reasons": {
                "body_wrist": "unsupported_wrist_anchor" if body_anchor_unusable else None,
                "hand_root": "unsupported_wrist_anchor" if root_anchor_unusable else None,
            },
            "valid_observed_mcp_count": valid_mcp_count,
            "body_score": _as_finite_float(body_score) or 0.0,
            "hand_root_score": _as_finite_float(root_score) or 0.0,
            "hand_root_mcp_support": int(root_mcp_support),
            "body_wrist_mcp_support": int(body_mcp_support),
            "anchor_gap_m": _as_finite_float(gap),
            "anchor_agreement_limit_m": float(agreement_limit),
            "body_wrist_rejected": body_wrist_rejected,
            "rejected_hand_count": int(hand_rejected.sum()),
        }

    metadata = {
        "policy_version": 3,
        "policy": "current_frame frozen rest-model hand geometry; coordinates unchanged; no previous frame or GT",
        "layout": "Body25 + left hand21 + right hand21",
        "rejected_mask": [bool(value) for value in rejected],
        "input_valid_mask": [bool(value) for value in raw_valid],
        "effective_valid_mask": [bool(value) for value in valid],
        "reasons": list(reasons),
        "hands": hands,
        "thresholds": {
            "min_confidence": float(min_confidence),
            "wrist_reach_scale": float(WRIST_REACH_SCALE),
            "wrist_slack_m": float(WRIST_SLACK_M),
            "edge_reach_scale": float(EDGE_REACH_SCALE),
            "edge_slack_m": float(EDGE_SLACK_M),
            "anchor_disagreement_slack_m": float(ANCHOR_DISAGREEMENT_SLACK_M),
            "unanchored_keep_confidence": float(UNANCHORED_KEEP_CONFIDENCE),
        },
        # Keep descriptive aliases alongside the short names for consumers
        # that serialize benchmark evidence without knowing this module's
        # internal naming conventions.
        "rejected_mask67": [bool(value) for value in rejected],
        "reasons_per_point": list(reasons),
        "per_hand": hands,
        "per_hand_anchor_choice": {side: info["anchor_choice"] for side, info in hands.items()},
        "rejected_count": int(rejected.sum()),
    }
    return filtered, metadata


__all__ = [
    "BODY_WRIST_INDEX",
    "ANCHOR_DISAGREEMENT_SLACK_M",
    "EDGE_REACH_SCALE",
    "EDGE_SLACK_M",
    "FINGER_CHAINS",
    "HAND_ROOT_INDEX",
    "HAND_SLICE",
    "MCP_INDICES",
    "UNANCHORED_KEEP_CONFIDENCE",
    "WRIST_REACH_SCALE",
    "WRIST_SLACK_M",
    "filter_hand_observations",
]
