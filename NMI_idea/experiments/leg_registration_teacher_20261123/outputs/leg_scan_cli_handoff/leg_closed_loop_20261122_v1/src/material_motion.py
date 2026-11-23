"""Material-chart and dIRD adapter for online leg-scan actions.

The scan simulator exposes a two-dimensional material chart, with station and
sector as its coordinates.  This module keeps that chart separate from the
reader and policy.  A probe is planned with the same integrated ProDMP basis as
``motion.Scanner.prodmp`` and is executed one frame at a time so that the latest
public echo can recondition the remaining trajectory.

The dIRD checkpoint shipped with the robot project is a score-only field for a
physical TCP frame.  The adapter can project the verified shaft LegChart into a
declared virtual SE(3) frame and interpolate cached field scores across rail
and orientation variants.  That bridge is explicitly unregistered and never
becomes an execution certificate.  A caller can enable loading of the existing
field with ``LEG_ENABLE_DIRD=1`` for an explicit audit, while ordinary
training remains lightweight and does not import Blender or holdout geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
from typing import Callable, Any

import numpy as np

import motion
from material_assets import MaterialChartBridge


_DIRD_FIELD_CACHE: dict[str, object] = {}


MATERIAL_FEATURE_NAMES = (
    "material_station_fraction",
    "material_sector_sin",
    "material_sector_cos",
    "material_contact_x_mm",
    "material_contact_y_mm",
    "material_contact_z_mm",
    "material_normal_x",
    "material_normal_z",
    "material_valid",
    "material_frame_quality",
    "material_posterior_peak",
    "material_history_frames",
    "material_dird_available",
    "material_atlas_supported",
    "material_atlas_x_fraction",
    "material_atlas_y_fraction",
    "material_atlas_registration_error",
    "material_virtual_dird_score",
    "material_virtual_dird_score_valid",
)


def _finite_controls(controls: Any) -> np.ndarray:
    values = np.asarray(controls, dtype=float).reshape(-1)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("controls must be a finite length-3 vector")
    if np.any(np.abs(values) > 1.0 + 1.e-9):
        raise ValueError("controls must lie in [-1, 1]")
    return np.clip(values, -1.0, 1.0)


def _as_float_pair(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite length-2 chart coordinate")
    return result


@dataclass(frozen=True)
class MaterialPlan:
    """A checked, endpoint-conditioned probe plan on the material chart."""

    skill: int
    controls: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    shape: np.ndarray
    segments: tuple
    applied_fraction: float
    use_dird: bool
    dird_metadata: dict

    @property
    def ticks(self) -> int:
        return int(sum(max(0, len(item[0]) - 1) for item in self.segments))

    @property
    def distance(self) -> float:
        total = 0.0
        for positions, _, _, _, speed in self.segments:
            costs = self._costs(positions, speed)
            if costs is None:
                return float("inf")
            total += float(costs.sum())
        return total

    @staticmethod
    def _costs(positions: np.ndarray, speed: float) -> np.ndarray | None:
        # This method is replaced by MaterialMotion._plan_costs when plans are
        # made.  Keeping a conservative fallback makes the object inspectable
        # even if a caller serializes only its trajectory arrays.
        if len(positions) < 2:
            return np.empty(0, dtype=float)
        delta = np.diff(np.asarray(positions, dtype=float), axis=0)
        return np.linalg.norm(delta, axis=1) * float(speed) * motion.CLOCK_S


@dataclass
class MaterialMotionResult:
    node: Any
    ticks: int
    distance: float
    legal: bool
    reason: str
    trajectory: np.ndarray
    feedback_updates: int = 0
    applied_fraction: float = 0.0
    stopped: bool = False
    stop_opportunities: tuple = ()
    plan: MaterialPlan | None = None
    exit: dict | None = None
    dird_report: dict | None = None


class _DirdAdapter:
    """Lazy score-only access to the validated robot field.

    The field consumes physical probe45 TCP transforms.  The material-assets
    bridge supplies a declared virtual chart transform for score-only audits.
    It remains unregistered and is never presented as a physical command path.
    """

    def __init__(self) -> None:
        repo = Path("/media/camp/EXT_DRIVE/RealUS_playground")
        source = Path(__file__).resolve().parents[2] / "reference_source" / "legacy_dird" / "dird_probe.py"
        enabled = os.environ.get("LEG_ENABLE_DIRD", "0") == "1"
        self.metadata = {
            "requested": bool(enabled),
            "source_present": bool(source.is_file() and repo.is_dir()),
            "available": False,
            "status": "score_only_unregistered",
            "physical_execution": False,
            "frame_mapping": "2d_material_chart_to_physical_tcp_not_calibrated",
            "public_feature_scaling": "tanh(raw_score/20)",
            "ready_for_execution": False,
            "source": str(source),
        }
        self._field = None
        self._load_error = None
        self._grid_key = None
        self._grid_x = None
        self._grid_y = None
        self._grid_scores = {}
        self._grid_scores_by_rail = {}
        # Do not load the large robot checkpoint during the normal training
        # loop.  An explicit audit can request the dependency and receives a
        # truthful failure if the robot environment is incomplete.
        if enabled and self.metadata["source_present"]:
            try:
                cache_key = str(source)
                if cache_key in _DIRD_FIELD_CACHE:
                    self._field = _DIRD_FIELD_CACHE[cache_key]
                else:
                    spec = importlib.util.spec_from_file_location("_leg_dird_probe", source)
                    if spec is None or spec.loader is None:
                        raise ImportError("cannot create dIRD module spec")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._field = module.DirdProbe(allow_stale_audit=True)
                    _DIRD_FIELD_CACHE[cache_key] = self._field
                self.metadata["available"] = True
                self.metadata["status"] = "score_only_virtual_unregistered"
            except Exception as exc:  # pragma: no cover - host package dependent
                self._load_error = f"{type(exc).__name__}: {exc}"
                self.metadata["load_error"] = self._load_error

    def score(self, *_args, **_kwargs) -> None:
        """Return no score until a caller supplies a calibrated physical frame."""
        return None

    @staticmethod
    def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=float).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm <= 1.e-12:
            return np.eye(3)
        axis = axis / norm
        x, y, z = axis
        c, s = np.cos(float(angle)), np.sin(float(angle))
        return np.array(((c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s),
                         (y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s),
                         (z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c))), dtype=float)

    def _virtual_transforms(self, bridge, x_values, y_values, orientation_offset=0.0):
        chart = bridge.chart
        points = np.stack(np.meshgrid(np.asarray(x_values, dtype=float),
                                      np.asarray(y_values, dtype=float), indexing="ij"), axis=-1).reshape(-1, 2)
        world = chart.world(np.column_stack([points, np.full(len(points), 12.0)]))
        basis = chart.frame(points)
        transforms = np.tile(np.eye(4), (len(points), 1, 1))
        for index in range(len(points)):
            rotation = basis[index]
            if np.linalg.det(rotation) < 0.0:
                rotation = rotation.copy()
                rotation[:, 2] *= -1.0
            if abs(float(orientation_offset)) > 1.e-12:
                rotation = self._rodrigues(rotation[:, 2], float(orientation_offset)) @ rotation
            transforms[index, :3, :3] = rotation
            transforms[index, :3, 3] = np.asarray(world[index], dtype=float) / 1000.0
        return transforms, points

    def prepare_grid(self, bridge, *, count: int = 9) -> dict:
        """Precompute score-only virtual field values on a public chart grid."""
        if self._field is None:
            return {"status": "unavailable", "available": False}
        chart = bridge.chart
        if chart is None:
            return {"status": "atlas_unavailable", "available": False}
        key = (str(bridge.atlas_path), int(count), int(bridge.pose_deg))
        if self._grid_key == key:
            return {"status": "cached", "available": True, "grid_shape": [count, count]}
        count = max(3, int(count))
        self._grid_x = np.linspace(chart.x_bounds_mm[0], chart.x_bounds_mm[1], count)
        self._grid_y = np.linspace(chart.y_bounds_mm[0], chart.y_bounds_mm[1], count)
        low, high = map(float, self._field.rail_limits)
        rails = np.stack([np.full(count * count, low),
                          np.full(count * count, (low + high) / 2.0),
                          np.full(count * count, high)])
        self._grid_scores = {}
        self._grid_scores_by_rail = {}
        orientation_offsets = (-0.25, 0.0, 0.25)
        for offset in orientation_offsets:
            transforms, _ = self._virtual_transforms(bridge, self._grid_x, self._grid_y, offset)
            ranking = self._field.rank_modes(transforms, np.eye(4), rails)
            by_input = {}
            for row in ranking.get("modes", []):
                by_input[int(row["input_index"])] = np.asarray(row["waypoint_scores"], dtype=float)
            if len(by_input) != 3:
                raise RuntimeError("dIRD virtual chart grid did not return all rail variants")
            rail_grid = np.stack([by_input[i] for i in range(3)]).reshape(3, count, count)
            self._grid_scores_by_rail[float(offset)] = rail_grid
            self._grid_scores[float(offset)] = np.max(rail_grid, axis=0)
        self._grid_key = key
        self.metadata.update({
            "virtual_precompute": {
                "status": "score_only_virtual_grid",
                "grid_shape": [count, count],
                "orientation_offsets_rad": list(orientation_offsets),
                "rail_variants_m": [low, (low + high) / 2.0, high],
                "interpolation": "bilinear chart x/y, nearest orientation variant",
                "base_world": np.eye(4).tolist(),
                "ready_for_execution": False,
            }
        })
        return self.metadata["virtual_precompute"]

    @staticmethod
    def _bilinear(grid: np.ndarray, x: float, y: float, x_grid: np.ndarray, y_grid: np.ndarray) -> float:
        ux = np.clip((float(x) - x_grid[0]) / max(x_grid[-1] - x_grid[0], 1.e-9) * (len(x_grid) - 1), 0., len(x_grid) - 1.)
        uy = np.clip((float(y) - y_grid[0]) / max(y_grid[-1] - y_grid[0], 1.e-9) * (len(y_grid) - 1), 0., len(y_grid) - 1.)
        ix, iy = min(int(np.floor(ux)), len(x_grid) - 2), min(int(np.floor(uy)), len(y_grid) - 2)
        ax, ay = ux - ix, uy - iy
        return float((1 - ax) * (1 - ay) * grid[ix, iy] + ax * (1 - ay) * grid[ix + 1, iy]
                     + (1 - ax) * ay * grid[ix, iy + 1] + ax * ay * grid[ix + 1, iy + 1])

    def score_poses(self, poses, bridge, *, precompute_count: int = 9) -> dict:
        """Interpolate cached virtual dIRD scores for mapped atlas poses."""
        if self._field is None:
            return {"status": "unavailable", "available": False, "supported_fraction": 0.0}
        mapped = list(poses)
        if not mapped:
            return {"status": "no_supported_poses", "available": True, "supported_fraction": 0.0}
        grid_report = self.prepare_grid(bridge, count=precompute_count)
        offsets = tuple(float(v) for v in (-0.25, 0.0, 0.25))
        scores = []
        for pose in mapped:
            values = [[self._bilinear(self._grid_scores_by_rail[offset][rail], pose.x_mm, pose.y_mm,
                                      self._grid_x, self._grid_y) for rail in range(3)]
                      for offset in offsets]
            scores.append(values)
        arr = np.asarray(scores, dtype=float)
        best = arr.max(axis=(1, 2))
        mean_by_orientation = arr.mean(axis=0)
        orientation_index, rail_index = np.unravel_index(int(np.argmax(mean_by_orientation)), mean_by_orientation.shape)
        rail_limits = tuple(float(v) for v in self._field.rail_limits)
        rail_values = (rail_limits[0], (rail_limits[0] + rail_limits[1]) / 2.0, rail_limits[1])
        return {
            "status": "score_only_virtual_interpolated",
            "available": True,
            "supported_fraction": 1.0,
            "pose_count": int(len(mapped)),
            "score_min": float(best.min()),
            "score_mean": float(best.mean()),
            "score_max": float(best.max()),
            "orientation_score_min": [float(v) for v in arr.min(axis=(0, 2))],
            "orientation_offsets_rad": list(offsets),
            "selected_virtual_mode": {
                "orientation_offset_rad": float(offsets[orientation_index]),
                "rail_m": float(rail_values[rail_index]),
                "selection": "highest_mean_interpolated_score_over_endpoint_or_midpoint",
            },
            "best_score_is_execution_ready": False,
            "grid": grid_report,
        }

    def score_trajectory(self, positions, bridge, *, depth_mm: float = 12.0) -> dict:
        mapped, mapping = bridge.map_trajectory(np.asarray(positions, dtype=float), depth_mm=depth_mm)
        report = {"mapping": mapping}
        report.update(self.score_poses(mapped, bridge))
        report["supported_fraction"] = float(mapping["supported_fraction"])
        if report.get("available") and mapping["supported_fraction"] < 1.0:
            report["status"] = "score_only_virtual_partial"
        return report


class MaterialMotion:
    """Plan and run public-observation-conditioned material chart probes."""

    def __init__(self, scanner) -> None:
        self.scanner = scanner
        self.dird = _DirdAdapter()
        atlas_enabled = os.environ.get("LEG_ENABLE_MATERIAL_ATLAS", "1") != "0"
        self.atlas = MaterialChartBridge(scanner, enabled=atlas_enabled)
        self._cost_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def features(self, node, spec, *, use_dird: bool = True) -> np.ndarray:
        """Return fixed public material and reader features.

        Only chart geometry, the public history reader and the latest public
        frame are used.  Hidden anatomy, labels and future tape values never
        enter this vector.
        """
        surface = self.scanner.surface
        station = float(node.station)
        sector = float(node.sector) % motion.SECTORS
        route_span = max(1.0, float(spec[2] - spec[1]))
        station_fraction = (station - float(spec[1])) / route_span
        contact, normal, _tangent = surface.pose(station, sector)
        valid = float(surface.legal(station, sector))
        frame = node.reader.frames[-1]
        quality = float(np.clip(frame[4], 0.0, 1.0))
        posterior = float(np.max(frame[6], initial=0.0))
        atlas_pose = self.atlas.map_pose(station, sector, depth_mm=12.0)
        atlas_supported = float(atlas_pose is not None)
        atlas_x = 0.0 if atlas_pose is None else atlas_pose.x_mm / 42.0
        atlas_y = 0.0 if atlas_pose is None else atlas_pose.y_mm / 36.0
        atlas_error = 0.0 if atlas_pose is None else min(atlas_pose.registration_error_mm / 200.0, 2.0)
        dird_score = 0.0
        dird_score_valid = 0.0
        if use_dird and atlas_pose is not None and self.dird.metadata["available"]:
            score_report = self.dird.score_poses([atlas_pose], self.atlas)
            if score_report.get("available") and score_report.get("score_mean") is not None:
                # The checkpoint score is uncalibrated.  Bound its public
                # feature contribution while retaining raw values in reports.
                dird_score = float(np.tanh(float(score_report["score_mean"]) / 20.0))
                dird_score_valid = 1.0
        vector = np.array((
            station_fraction,
            np.sin(sector * motion.ANGLE_STEP),
            np.cos(sector * motion.ANGLE_STEP),
            float(contact[0]) / 100.0,
            float(motion._between(surface.y, station)) / 1400.0,
            float(contact[1]) / 100.0,
            float(normal[0]),
            float(normal[1]),
            valid,
            quality,
            posterior,
            min(1.0, len(node.reader.frames) / 64.0),
            float(self.dird.metadata["available"] if use_dird else 0.0),
            atlas_supported,
            atlas_x,
            atlas_y,
            atlas_error,
            dird_score,
            dird_score_valid,
        ), dtype=np.float32)
        if vector.shape != (len(MATERIAL_FEATURE_NAMES),) or not np.isfinite(vector).all():
            raise RuntimeError("material feature construction produced invalid values")
        return vector

    @staticmethod
    def _skill_goal(start: np.ndarray, controls: np.ndarray, skill: int, spec) -> np.ndarray:
        if skill == 1:
            # A local in-place window can sweep angularly and move at most 35%
            # of the normal axial action range before returning to its anchor.
            axial = 0.35 * float(controls[1])
        elif skill == 2:
            # A revisit always starts toward the already-scanned side.  The
            # magnitude remains policy-controlled but is never exactly zero.
            axial = -max(0.30, abs(float(controls[1])))
        else:
            raise ValueError("skill must be 1 (in-place window) or 2 (backward revisit)")
        goal = start + np.array((motion.AXIAL_RANGE * axial,
                                 motion.ANGLE_RANGE * float(controls[0])))
        goal[0] = np.clip(goal[0], float(spec[1]), float(spec[2]))
        return goal

    def _plan_costs(self, positions: np.ndarray, spec, speed: float) -> np.ndarray | None:
        return self.scanner._path_metrics(np.asarray(positions, dtype=float), spec, speed)

    def _dird_selection_report(self, segments, *, use_dird: bool) -> dict:
        """Score only endpoint and midpoint chart realizations for ranking."""
        positions = np.asarray(segments[0][0], dtype=float)
        probe_positions = np.asarray((positions[len(positions) // 2], positions[-1]), dtype=float)
        mapped, mapping = self.atlas.map_trajectory(probe_positions, depth_mm=12.0)
        report = {
            "mapping": mapping,
            "selection_supported": False,
            "selection_score": None,
            "status": "disabled_for_ablation" if not use_dird else "unsupported_endpoint_or_midpoint",
        }
        if use_dird and self.dird.metadata["available"] and len(mapped) == len(probe_positions):
            scores = self.dird.score_poses(mapped, self.atlas)
            report.update({
                "selection_supported": bool(scores.get("available")),
                "selection_score": scores.get("score_mean"),
                "selected_virtual_mode": scores.get("selected_virtual_mode"),
                "status": "score_only_virtual_endpoint_midpoint",
                "orientation_offsets_rad": scores.get("orientation_offsets_rad"),
            })
        return report

    @staticmethod
    def _reject(node, reason: str, plan: MaterialPlan | None = None) -> MaterialMotionResult:
        return MaterialMotionResult(node=node, ticks=0, distance=0.0, legal=False,
                                    reason=reason, trajectory=np.empty((0, 5), dtype=float),
                                    plan=plan)

    def plan_probe(self, node, spec, skill, controls, max_ticks, max_mm, *, use_dird=True):
        controls = _finite_controls(controls)
        skill = int(skill)
        if skill not in (1, 2):
            raise ValueError("skill must be 1 (in-place window) or 2 (backward revisit)")
        max_ticks = max(0, int(max_ticks))
        max_mm = float(max_mm)
        if not np.isfinite(max_mm) or max_mm < 0.0:
            raise ValueError("max_mm must be finite and nonnegative")
        start = np.array((float(node.station), float(node.sector)), dtype=float)
        goal = self._skill_goal(start, controls, skill, spec)
        shape = float(controls[2]) * np.array((0.0, 0.60), dtype=float)
        if np.linalg.norm(goal - start) < motion.EPS and np.linalg.norm(shape) < motion.EPS:
            return None
        # Reduce an invalid or over-budget target while keeping the action
        # direction observable.  For each feasible endpoint, construct two
        # equivalent ProDMP realizations.  They share the exact endpoint while
        # retaining the policy's shape in the first candidate and using a
        # slightly slower, smoother realization in the second candidate.
        for fraction in (1.0, 0.75, 0.5, 0.25):
            applied_goal = start + fraction * (goal - start)
            if not self.scanner.surface.legal(*applied_goal):
                continue
            local_shape = fraction * shape
            variants = []
            for variant_name, variant_shape, rate_scale in (
                    ("policy_shape_fast", local_shape, 1.0),
                    ("smoothed_shape_slow", .65 * local_shape, .97)):
                segments = self.scanner._fit_loop(
                    node, spec, applied_goal, variant_shape, max_ticks, max_mm,
                    prodmp=True, speed=motion.MAX_SPEED_MM_S * rate_scale,
                )
                if segments is None:
                    continue
                selection = self._dird_selection_report(segments, use_dird=use_dird)
                variants.append({
                    "name": variant_name,
                    "shape": np.asarray(variant_shape, dtype=float),
                    "rate_scale": float(rate_scale),
                    "segments": tuple(segments),
                    "selection": selection,
                })
            if not variants:
                continue
            if len(variants) == 1:
                # If the direct smoother was rejected by a tight contact
                # corridor, derive a second realization from the feasible
                # trajectory itself.  It keeps the exact endpoint and frame
                # count while changing the forcing shape and the permitted
                # speed cap, so it remains a genuine executable alternative.
                source = variants[0]
                alternate_segments = []
                alternate_shape = .45 * np.asarray(source["shape"], dtype=float)
                valid_alternate = True
                for poses, velocities, target, _shape, speed in source["segments"]:
                    ticks = len(poses) - 1
                    alt_positions, alt_velocities = motion.PRIMITIVE.segment(
                        np.asarray(poses[0], dtype=float), np.asarray(velocities[0], dtype=float),
                        np.asarray(target, dtype=float), ticks, alternate_shape,
                    )
                    alt_cost = self._plan_costs(alt_positions, spec, speed * .97)
                    if alt_cost is None:
                        valid_alternate = False
                        break
                    alternate_segments.append((alt_positions, alt_velocities, target,
                                               alternate_shape.copy(), speed * .97))
                if valid_alternate:
                    alternate_selection = self._dird_selection_report(tuple(alternate_segments), use_dird=use_dird)
                    variants.append({
                        "name": "feasible_shape_rate_fallback",
                        "shape": alternate_shape.copy(),
                        "rate_scale": float(source["rate_scale"] * .97),
                        "segments": tuple(alternate_segments),
                        "selection": alternate_selection,
                    })
            eligible = [v for v in variants if v["selection"]["selection_supported"]]
            if eligible:
                chosen = max(eligible, key=lambda value: float(value["selection"]["selection_score"]))
                chart_score_applied = True
            else:
                chosen = variants[0]
                chart_score_applied = False
            segments = chosen["segments"]
            flat_positions = np.concatenate([np.asarray(item[0], dtype=float) for item in segments], axis=0)
            atlas_mapped, atlas_mapping = self.atlas.map_trajectory(flat_positions, depth_mm=12.0)
            if use_dird and self.dird.metadata["available"]:
                dird_report = self.dird.score_trajectory(flat_positions, self.atlas, depth_mm=12.0)
            else:
                dird_report = {"status": "disabled_for_ablation", "available": bool(self.dird.metadata["available"]),
                               "mapping": atlas_mapping, "supported_fraction": float(atlas_mapping["supported_fraction"])}
            metadata = dict(self.dird.metadata)
            metadata.update({
                "used": bool(use_dird and self.dird.metadata["available"]),
                "chart_score_applied": bool(chart_score_applied),
                "atlas_mapping": atlas_mapping,
                "dird_report": dird_report,
                "atlas_supported_pose_count": len(atlas_mapped),
                "equivalent_plan_candidates": [
                    {"name": v["name"], "rate_scale": v["rate_scale"],
                     "shape": v["shape"].tolist(), "selection": v["selection"]}
                    for v in variants
                ],
                "chosen_virtual_mode": chosen["selection"].get("selected_virtual_mode"),
                "chosen_shape": chosen["shape"].tolist(),
                "chosen_rate_scale": chosen["rate_scale"],
                "selection_status": "dird_ranked_equivalent_plans" if chart_score_applied else "policy_shape_default",
            })
            return MaterialPlan(
                skill=skill,
                controls=controls.copy(),
                start=start.copy(),
                goal=applied_goal.copy(),
                shape=chosen["shape"].copy(),
                segments=tuple(segments),
                applied_fraction=float(fraction),
                use_dird=bool(use_dird),
                dird_metadata=metadata,
            )
        return None

    @staticmethod
    def _feedback_shape(frame) -> float:
        """Smooth public-frame correction in angular chart units."""
        z = np.asarray(frame[2], dtype=float)
        if z.size == 0:
            return 0.0
        z = z.reshape(-1, 2)
        posterior = np.asarray(frame[6], dtype=float).reshape(-1)
        if posterior.size != len(z):
            posterior = np.ones(len(z), dtype=float)
        posterior = np.clip(posterior, 0.0, None)
        total = float(posterior.sum())
        weights = posterior / total if total > 1.e-12 else np.full(len(z), 1.0 / len(z))
        quality = float(np.clip(frame[4], 0.0, 1.0))
        confidence = float(np.clip(np.max(posterior, initial=0.0), 0.0, 1.0))
        # Quality and confidence attenuate continuously, including low-quality
        # frames.  No entry threshold is used for trajectory feedback.
        gain = quality * (0.20 + 0.80 * confidence)
        lateral = float(np.sum(weights * np.clip(z[:, 0], -25.0, 25.0)))
        return float(np.clip(gain * lateral / 18.0, -0.65, 0.65))

    def _direct_return(self, current, anchor, spec, max_ticks, max_mm, velocity, shape):
        start = _as_float_pair(current, "current")
        goal = _as_float_pair(anchor, "anchor").copy()
        if np.linalg.norm(start - goal) < motion.EPS:
            return None
        # Keep the angular endpoint on the locally unwrapped branch.  This
        # avoids a 23.9 -> 0.0 discontinuity when a return crosses the sector
        # seam while preserving the actual physical pose modulo 24 sectors.
        goal[1] = start[1] + motion._sector_delta(start[1], goal[1])
        for ticks in range(2, int(max_ticks) + 1):
            positions, velocities = motion.PRIMITIVE.segment(
                start, np.asarray(velocity, dtype=float), goal, ticks,
                -0.35 * np.asarray(shape, dtype=float),
            )
            costs = self._plan_costs(positions, spec, motion.MAX_SPEED_MM_S)
            if costs is not None and float(costs.sum()) <= max_mm + motion.EPS:
                return positions, velocities, costs
        return None

    def _execute_positions(self, output, spec, trial, positions, velocities, costs,
                           path, *, feedback=False, max_mm=np.inf, total=0.0,
                           target=None, shape=None, updates=0):
        """Execute a direct segment and return updated execution state."""
        positions = np.asarray(positions, dtype=float).copy()
        velocities = np.asarray(velocities, dtype=float).copy()
        costs = np.asarray(costs, dtype=float).copy()
        target = positions[-1] if target is None else np.asarray(target, dtype=float)
        shape = np.zeros(2, dtype=float) if shape is None else np.asarray(shape, dtype=float)
        tail = 0.0
        for k in range(len(positions) - 1):
            remain = len(positions) - 1 - k
            if feedback and remain >= 2:
                correction = self._feedback_shape(output.reader.frames[-1])
                strength = min(1.0, remain / max(1, len(positions) - 1)) ** 2
                updated_shape = shape * strength + np.array((0.0, strength * correction))
                suffix, suffix_velocity = motion.PRIMITIVE.segment(
                    positions[k], velocities[k], target, remain, updated_shape,
                )
                suffix_cost = self._plan_costs(suffix, spec, motion.MAX_SPEED_MM_S)
                if suffix_cost is not None and total + float(suffix_cost.sum()) <= max_mm + motion.EPS:
                    if np.linalg.norm(suffix[1] - positions[k + 1]) > 1.e-6:
                        positions[k:], velocities[k:], costs[k:] = suffix, suffix_velocity, suffix_cost
                        updates += 1
            s, a = positions[k + 1]
            total += float(costs[k])
            output.tick += 1
            output.station, output.sector = float(s), float(motion.wrap_sector(a))
            self.scanner._observe(output, spec, trial, float(s), float(a), output.tick)
            path.append((output.tick, float(s), output.sector, float(costs[k]), float(total)))
        return output, path, total, updates

    def run_probe(self, node, spec, trial, plan, max_ticks, max_mm, *, feedback=True, stop=None,
                  use_dird=True, copy_node=True):
        if plan is None:
            return self._reject(node, "no_feasible_plan")
        if not isinstance(plan, MaterialPlan):
            raise TypeError("plan must be a MaterialPlan returned by plan_probe")
        max_ticks = max(0, int(max_ticks))
        max_mm = float(max_mm)
        start_tick = int(node.tick)
        output = node.fork() if copy_node else node
        path = []
        total = 0.0
        updates = 0
        stopped = False
        opportunities = []
        exit_info = None
        dird_report = plan.dird_metadata.get("dird_report") if isinstance(plan.dird_metadata, dict) else None

        segment_costs = []
        for segment in plan.segments:
            segment_costs.append(self._plan_costs(np.asarray(segment[0], dtype=float), spec, segment[4]))
        if any(cost is None for cost in segment_costs):
            return self._reject(node, "invalid_plan_path", plan)
        tail_costs = [float(sum(np.asarray(cost).sum() for cost in segment_costs[index + 1:]))
                      for index in range(len(segment_costs))]
        exit_checks = set()
        if plan.segments:
            outbound_ticks = max(0, len(plan.segments[0][0]) - 1)
            if outbound_ticks >= 3:
                exit_checks.update({max(0, int(round(outbound_ticks / 3.0)) - 1),
                                    max(0, int(round(2.0 * outbound_ticks / 3.0)) - 1)})

        for segment_index, segment in enumerate(plan.segments):
            positions, velocities, target, shape, speed = segment
            positions = np.asarray(positions, dtype=float).copy()
            velocities = np.asarray(velocities, dtype=float).copy()
            costs = np.asarray(segment_costs[segment_index], dtype=float).copy()
            for k in range(len(positions) - 1):
                if output.tick - start_tick >= max_ticks or total + float(costs[k]) > max_mm + motion.EPS:
                    return MaterialMotionResult(output, output.tick - start_tick, total, False,
                                                "runtime_budget", np.asarray(path), updates,
                                                plan.applied_fraction, stopped, tuple(opportunities),
                                                plan, exit_info, dird_report)
                remain = len(positions) - 1 - k
                if feedback and remain >= 2:
                    correction = self._feedback_shape(output.reader.frames[-1])
                    strength = min(1.0, remain / max(1, len(positions) - 1)) ** 2
                    updated_shape = np.asarray(shape, dtype=float) * strength
                    updated_shape += np.array((0.0, strength * correction))
                    suffix, suffix_velocity = motion.PRIMITIVE.segment(
                        positions[k], velocities[k], np.asarray(target, dtype=float), remain, updated_shape,
                    )
                    suffix_cost = self._plan_costs(suffix, spec, speed)
                    if (suffix_cost is not None
                            and total + float(suffix_cost.sum()) + tail_costs[segment_index] <= max_mm + motion.EPS):
                        if np.linalg.norm(suffix[1] - positions[k + 1]) > 1.e-6:
                            positions[k:], velocities[k:], costs[k:] = suffix, suffix_velocity, suffix_cost
                            updates += 1
                s, a = positions[k + 1]
                ds = float(costs[k])
                total += ds
                output.tick += 1
                output.station, output.sector = float(s), float(motion.wrap_sector(a))
                self.scanner._observe(output, spec, trial, float(s), float(a), output.tick)
                path.append((output.tick, float(s), output.sector, ds, float(total)))

                # A callback is considered only during the outbound segment,
                # after at least one frame, and only when a legal direct return
                # to the original anchor fits in the remaining budget.
                if stop is not None and segment_index == 0 and k in exit_checks and remain >= 2:
                    available_ticks = max_ticks - (output.tick - start_tick)
                    available_mm = max_mm - total
                    current = np.array((float(output.station), float(output.sector)))
                    return_plan = self._direct_return(
                        current, plan.start, spec, available_ticks, available_mm,
                        velocities[k + 1], np.asarray(shape, dtype=float),
                    )
                    return_ticks = 0 if return_plan is None else len(return_plan[0]) - 1
                    early_enough = (output.tick - start_tick) + return_ticks < plan.ticks
                    if return_plan is not None and early_enough:
                        info = {
                            "elapsed_ticks": float(output.tick - start_tick),
                            "remaining_ticks": float(available_ticks),
                            "return_reserve_ticks": float(return_ticks),
                            "elapsed_distance_mm": float(total),
                            "remaining_mm": float(available_mm),
                            "return_reserve_mm": float(np.asarray(return_plan[2]).sum()),
                            "station": float(output.station),
                            "sector": float(output.sector),
                            "segment_index": float(segment_index),
                        }
                        opportunities.append(dict(info))
                        if bool(stop(output, info)):
                            stopped = True
                            exit_info = dict(info)
                            ret_pos, ret_vel, ret_cost = return_plan
                            output, path, total, updates = self._execute_positions(
                                output, spec, trial, ret_pos, ret_vel, ret_cost, path,
                                feedback=feedback, max_mm=max_mm, total=total,
                                target=ret_pos[-1], shape=np.asarray(shape), updates=updates,
                            )
                            # The return trajectory ends at the actual anchor.
                            output.station = int(round(plan.start[0]))
                            output.sector = int(round(plan.start[1])) % motion.SECTORS
                            break
            if stopped:
                break
        output.distance += total
        output.visits += 1
        if (abs(output.station - plan.start[0]) < 1.e-6
                and abs(motion._sector_delta(output.sector, plan.start[1])) < 1.e-6):
            output.station = int(round(plan.start[0]))
            output.sector = int(round(plan.start[1])) % motion.SECTORS
        return MaterialMotionResult(output, output.tick - start_tick, total, True,
                                    "stopped_and_returned" if stopped else "completed",
                                    np.asarray(path, dtype=float).reshape(-1, 5), updates,
                                    plan.applied_fraction, stopped, tuple(opportunities),
                                    plan, exit_info, dird_report)


__all__ = ["MATERIAL_FEATURE_NAMES", "MaterialPlan", "MaterialMotionResult", "MaterialMotion"]
