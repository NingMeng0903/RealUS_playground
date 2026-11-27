"""Verified material-chart bridge for the 2-D leg simulator.

The simulator's probe chart is registered by station height and a public
surface contact.  The attached LegChart atlas provides a bounded right-leg
shaft patch with an explicit ``(x, y, z)`` millimetre chart and strict
tetrahedral interpolation.  This module projects the simulator chart into that
patch and records where the projection is unsupported.  It never reads Blender
holdout data and it never turns the virtual transform into a physical robot
registration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

import motion


DEFAULT_CHART_SOURCE = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/"
    "ab_leg_bifurcation_20261112/leg_chart.py"
)
DEFAULT_ATLAS_PATH = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/"
    "ab_validation_20261110/repair_20261110/harmonic/shaft_roi_v2/atlas_right.npz"
)

_BUNDLED = Path(__file__).resolve().parents[4] / "assets"
if (_BUNDLED / "leg_chart.py").is_file():
    DEFAULT_CHART_SOURCE = _BUNDLED / "leg_chart.py"
    DEFAULT_ATLAS_PATH = _BUNDLED / "atlas_right.npz"

_CHART_CACHE: dict[tuple[str, str, int], object] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_chart_class(source: Path):
    spec = importlib.util.spec_from_file_location("_leg_material_chart", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import LegChart from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LegChart


@dataclass(frozen=True)
class AtlasPose:
    """One strictly mapped virtual atlas pose."""

    station: float
    sector: float
    x_mm: float
    y_mm: float
    z_mm: float
    world_mm: np.ndarray
    basis: np.ndarray
    transform_world_tcp: np.ndarray
    registration_error_mm: float


class MaterialChartBridge:
    """Map public simulator poses to the audited shaft LegChart.

    The bridge is deliberately lazy.  Loading the LegChart takes roughly
    0.4 seconds and about 140 MB in the genesis environment, so one chart is
    shared per process and only material-enabled workers construct it.
    """

    def __init__(self, scanner, *, chart_source: Path | str = DEFAULT_CHART_SOURCE,
                 atlas_path: Path | str = DEFAULT_ATLAS_PATH, pose_deg: int = 0,
                 enabled: bool = True):
        self.scanner = scanner
        self.chart_source = Path(chart_source).expanduser().resolve()
        self.atlas_path = Path(atlas_path).expanduser().resolve()
        self.pose_deg = int(pose_deg)
        self.enabled = bool(enabled)
        self._chart = None
        self._x_grid = None
        self._y_grid = None
        self._loaded = False
        self._load_error = None
        self._supported_cache: dict[tuple[int, int, int], AtlasPose | None] = {}
        self._station_x_cache: dict[int, float | None] = {}
        self._station_x_grid: np.ndarray | None = None

    @property
    def chart(self):
        if not self.enabled:
            return None
        if self._loaded:
            return self._chart
        self._loaded = True
        try:
            if not self.chart_source.is_file() or not self.atlas_path.is_file():
                raise FileNotFoundError("LegChart source or atlas asset is unavailable")
            cache_key = (str(self.chart_source), str(self.atlas_path), self.pose_deg)
            if cache_key in _CHART_CACHE:
                self._chart = _CHART_CACHE[cache_key]
            else:
                chart_class = _load_chart_class(self.chart_source)
                self._chart = chart_class(self.pose_deg, atlas_path=self.atlas_path)
                _CHART_CACHE[cache_key] = self._chart
            x = np.linspace(self._chart.x_bounds_mm[0], self._chart.x_bounds_mm[1], 257)
            world = self._chart.world(np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]))
            world_y = np.asarray(world[:, 1], dtype=float)
            order = np.argsort(world_y)
            self._x_grid = x[order]
            self._y_grid = world_y[order]
            self._station_x_grid = np.asarray([
                np.nan if self.station_to_x(index) is None else self.station_to_x(index)
                for index in range(len(self.scanner.surface.y))
            ], dtype=float)
        except Exception as exc:  # pragma: no cover - host asset dependent
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._chart = None
        return self._chart

    @property
    def metadata(self) -> dict[str, Any]:
        chart = self.chart
        result: dict[str, Any] = {
            "enabled": self.enabled,
            "available": chart is not None,
            "chart_source": str(self.chart_source),
            "atlas_path": str(self.atlas_path),
            "pose_deg": self.pose_deg,
            "scope": "right_leg_shaft_roi_only",
            "holdout_opened": False,
            "registration": {
                "method": "public_simulator_station_y_plus_surface_contact_projected_on_LegChart_frame",
                "world_transform": "virtual_chart_world_from_strict_LegChart_query",
                "physical_robot_registration_verified": False,
                "physical_execution_ready": False,
                "unsupported_outside_chart": True,
                "virtual_se3_handedness_fix": "flip_chart_inward_axis_if_frame_det_negative",
            },
        }
        if self._load_error:
            result["load_error"] = self._load_error
        if chart is not None:
            result["chart_metadata"] = {
                "chart_definition": chart.metadata.get("chart_definition", {}),
                "geometry_assumptions": chart.metadata.get("geometry_assumptions", {}),
                "asset_hashes": {
                    self.atlas_path.name: _sha256(self.atlas_path),
                    self.chart_source.name: _sha256(self.chart_source),
                },
            }
        return result

    @staticmethod
    def _quantized(value: float, step: float = 1.e-4) -> int:
        return int(np.rint(float(value) / step))

    def station_to_x(self, station: float) -> float | None:
        chart = self.chart
        if chart is None or self._y_grid is None or self._x_grid is None:
            return None
        query = float(station)
        rounded = int(np.rint(query))
        if abs(query - rounded) < 1.e-8 and self._station_x_grid is not None and 0 <= rounded < len(self._station_x_grid):
            value = self._station_x_grid[rounded]
            return None if not np.isfinite(value) else float(value)
        cache_key = self._quantized(query)
        if cache_key in self._station_x_cache:
            return self._station_x_cache[cache_key]
        physical_y = float(motion._between(self.scanner.surface.y, query))
        if physical_y < float(self._y_grid[0]) or physical_y > float(self._y_grid[-1]):
            self._station_x_cache[cache_key] = None
            return None
        value = float(np.interp(physical_y, self._y_grid, self._x_grid))
        self._station_x_cache[cache_key] = value
        return value

    def map_pose(self, station: float, sector: float, *, depth_mm: float = 12.0) -> AtlasPose | None:
        """Map one simulator contact to a strict virtual LegChart pose.

        The public simulator contact is used only to obtain a lateral offset in
        the LegChart's verified local frame.  The resulting transform is
        explicitly virtual and remains unsuitable for robot execution.
        """
        chart = self.chart
        if chart is None:
            return None
        if not np.isfinite(depth_mm) or not 0.0 <= float(depth_mm) <= chart.z_bounds_mm[1]:
            raise ValueError("depth_mm must lie in the LegChart depth domain")
        key = (self._quantized(station), self._quantized(sector), self._quantized(depth_mm))
        if key in self._supported_cache:
            return self._supported_cache[key]
        x_mm = self.station_to_x(station)
        if x_mm is None:
            self._supported_cache[key] = None
            return None
        surface = self.scanner.surface
        contact, _normal, _tangent = surface.pose(float(station), float(sector))
        contact_world = np.asarray([contact[0], motion._between(surface.y, float(station)), contact[1]], dtype=float)
        valid = np.asarray(surface.valid[int(np.clip(np.rint(station), 0, len(surface.y) - 1))], dtype=bool)
        cross = np.asarray(surface.contact[int(np.clip(np.rint(station), 0, len(surface.y) - 1))], dtype=float)
        center = np.nanmean(cross[valid], axis=0) if np.any(valid) else np.nanmean(cross, axis=0)
        center_world = np.asarray([center[0], contact_world[1], center[1]], dtype=float)
        try:
            basis_surface = np.asarray(chart.frame(np.asarray([[x_mm, 0.0]], dtype=float))[0], dtype=float)
            y_mm = float(np.dot(contact_world - center_world, basis_surface[:, 1]))
            # The LegChart strict query owns the final mapped skin point.  Its
            # world point and frame are used to construct the virtual TCP pose.
            world_mm = np.asarray(chart.world(np.asarray([x_mm, y_mm, depth_mm], dtype=float)), dtype=float)
            basis = np.asarray(chart.frame(np.asarray([[x_mm, y_mm]], dtype=float))[0], dtype=float)
            # LegChart exposes the ordered (longitudinal, lateral, inward)
            # frame.  Its inward convention is intentionally anatomical and
            # left-handed in this asset.  A virtual SE(3) transform must be a
            # proper rotation, so flip only the virtual third axis and record
            # the correction in metadata.  This does not claim a calibrated
            # probe orientation.
            if np.linalg.det(basis) < 0.0:
                basis[:, 2] *= -1.0
        except Exception:
            self._supported_cache[key] = None
            return None
        registration_error = float(np.linalg.norm(center_world - np.asarray(chart.world(np.asarray([x_mm, 0.0, 0.0])))))
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = basis
        transform[:3, 3] = world_mm / 1000.0
        pose = AtlasPose(float(station), float(sector), float(x_mm), float(y_mm), float(depth_mm),
                         world_mm, basis, transform, registration_error)
        self._supported_cache[key] = pose
        return pose

    def map_trajectory(self, positions: np.ndarray, *, depth_mm: float = 12.0) -> tuple[list[AtlasPose], dict[str, Any]]:
        rows = np.asarray(positions, dtype=float).reshape(-1, 2)
        mapped: list[AtlasPose] = []
        unsupported = 0
        for station, sector in rows:
            # Station support is a compact memoized grid.  Skip distal and
            # proximal poses outside the shaft ROI before touching the costly
            # chart frame/tetra interpolation.
            if self.station_to_x(float(station)) is None:
                unsupported += 1
                continue
            pose = self.map_pose(float(station), float(sector), depth_mm=depth_mm)
            if pose is None:
                unsupported += 1
            else:
                mapped.append(pose)
        return mapped, {
            "requested_pose_count": int(len(rows)),
            "supported_pose_count": int(len(mapped)),
            "supported_fraction": float(len(mapped) / max(1, len(rows))),
            "unsupported_pose_count": int(unsupported),
            "registration_error_mm_median": float(np.median([p.registration_error_mm for p in mapped])) if mapped else None,
            "registration_error_mm_max": float(np.max([p.registration_error_mm for p in mapped])) if mapped else None,
            "depth_mm": float(depth_mm),
            "physical_registration_verified": False,
        }


__all__ = ["AtlasPose", "MaterialChartBridge", "DEFAULT_CHART_SOURCE", "DEFAULT_ATLAS_PATH"]
