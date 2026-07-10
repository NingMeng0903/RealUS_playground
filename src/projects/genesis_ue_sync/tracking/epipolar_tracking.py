from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle
from projects.genesis_ue_sync.tracking.heatmap_ops import HeatPoint, peak_points
from projects.genesis_ue_sync.tracking.triangulation import (
    TriangulatedPoint,
    bilinear_sample,
    epipolar_line,
    fundamental_from_calibrations,
    sample_epipolar_line,
    triangulate_linear,
)


@dataclass(frozen=True)
class EpipolarTrackerConfig:
    source_camera_id: str | None = None
    max_source_points: int = 32
    source_min_distance: int = 12
    source_threshold_quantile: float = 0.995
    target_line_samples: int = 256
    target_peak_threshold: float = 0.6
    max_matches_per_source: int = 2
    max_reprojection_error_px: float = 25.0


@dataclass(frozen=True)
class MatchedObservation:
    camera_id: str
    xy: tuple[float, float]
    score: float
    line: tuple[float, float, float] | None = None


@dataclass
class FramePointCloudResult:
    frame_idx: int
    source_camera_id: str
    source_candidates: list[HeatPoint]
    triangulated_points: list[TriangulatedPoint]
    debug: dict[str, Any] = field(default_factory=dict)


def _source_valid_mask(source_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(source_mask, dtype=bool)
    return (~mask).astype(np.float32)


def _best_target_peak(
    source_camera_id: str,
    target_camera_id: str,
    *,
    source_xy: tuple[float, float],
    source_calibration,
    target_calibration,
    target_heatmap: np.ndarray,
    line_samples: int,
    score_threshold: float,
) -> MatchedObservation | None:
    F = fundamental_from_calibrations(source_calibration, target_calibration)
    line = epipolar_line(F, source_xy)
    samples = sample_epipolar_line(line, width=target_heatmap.shape[1], height=target_heatmap.shape[0], samples=line_samples)
    if samples.size == 0:
        return None
    scores = bilinear_sample(target_heatmap, samples)
    if scores.size == 0:
        return None
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < float(score_threshold):
        return None
    best_xy = tuple(float(v) for v in samples[best_idx].tolist())
    return MatchedObservation(
        camera_id=target_camera_id,
        xy=best_xy,
        score=best_score,
        line=tuple(float(v) for v in line.tolist()),
    )


def track_obstacles_frame(
    *,
    frame_idx: int,
    heatmaps: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    calibration: CalibrationBundle,
    config: EpipolarTrackerConfig,
) -> FramePointCloudResult:
    camera_ids = list(heatmaps.keys())
    if not camera_ids:
        raise ValueError("track_obstacles_frame requires at least one heatmap.")
    source_camera_id = config.source_camera_id or camera_ids[0]
    if source_camera_id not in heatmaps:
        raise KeyError(f"Source camera '{source_camera_id}' is missing from frame heatmaps.")
    source_candidates = peak_points(
        heatmaps[source_camera_id],
        max_points=int(config.max_source_points),
        min_distance=int(config.source_min_distance),
        threshold_quantile=float(config.source_threshold_quantile),
        valid_mask=_source_valid_mask(masks[source_camera_id]),
    )
    triangulated: list[TriangulatedPoint] = []
    debug_matches: list[dict[str, Any]] = []
    for candidate in source_candidates:
        observations = [
            (
                source_camera_id,
                MatchedObservation(
                    camera_id=source_camera_id,
                    xy=candidate.as_xy(),
                    score=float(candidate.score),
                    line=None,
                ),
            )
        ]
        for target_camera_id in camera_ids:
            if target_camera_id == source_camera_id:
                continue
            matched = _best_target_peak(
                source_camera_id,
                target_camera_id,
                source_xy=candidate.as_xy(),
                source_calibration=calibration.camera(source_camera_id),
                target_calibration=calibration.camera(target_camera_id),
                target_heatmap=np.asarray(heatmaps[target_camera_id], dtype=np.float32),
                line_samples=int(config.target_line_samples),
                score_threshold=float(config.target_peak_threshold),
            )
            if matched is not None:
                observations.append((target_camera_id, matched))
            if len(observations) - 1 >= int(config.max_matches_per_source):
                break
        if len(observations) < 2:
            continue
        calibration_observations = [
            (calibration.camera(camera_id), match.xy)
            for camera_id, match in observations
        ]
        xyz_world, reproj = triangulate_linear(calibration_observations)
        if reproj > float(config.max_reprojection_error_px):
            continue
        score = float(np.mean([match.score for _, match in observations]))
        triangulated.append(
            TriangulatedPoint(
                xyz_world=xyz_world,
                reprojection_error_px=float(reproj),
                observations={camera_id: match.xy for camera_id, match in observations},
                score=score,
            )
        )
        debug_matches.append(
            {
                "source_xy": candidate.as_xy(),
                "source_score": float(candidate.score),
                "observations": {
                    camera_id: {
                        "xy": [float(match.xy[0]), float(match.xy[1])],
                        "score": float(match.score),
                        "line": list(match.line) if match.line is not None else None,
                    }
                    for camera_id, match in observations
                },
                "reprojection_error_px": float(reproj),
            }
        )
    return FramePointCloudResult(
        frame_idx=int(frame_idx),
        source_camera_id=source_camera_id,
        source_candidates=source_candidates,
        triangulated_points=triangulated,
        debug={"matches": debug_matches},
    )


__all__ = [
    "EpipolarTrackerConfig",
    "FramePointCloudResult",
    "MatchedObservation",
    "track_obstacles_frame",
]
