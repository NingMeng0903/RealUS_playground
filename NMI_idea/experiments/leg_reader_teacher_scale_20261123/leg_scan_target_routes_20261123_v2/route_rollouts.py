"""Fixed paired pilot runner for the verified v2 atlas route pack.

The runner is intentionally a sibling wrapper rather than a change to the v1
launcher.  It imports the v1 public scanner/environment, constructs a scanner
directly from ``atlas_routes.npz`` (there is no ``ct_bounds`` or fourth source),
and supplies a small atlas-only layout factory.  No MedSeg, native CT, Blender,
or holdout input is accepted by this module.

Typical use::

    pilot = initialize(Path("atlas_routes.npz"))
    case = pilot.case_plan("calibration", "femoral")[0]
    rows = run_case(case, ("mpc", "teacher_ppo", "ppo"), pilot=pilot)

``rows`` has the historical ``(case, arm_name, item)`` layout.  ``item``
proxies the v1 ``Rollout`` fields while exposing ``item.public_reader`` (frames
without labels) and ``item.evaluator`` (private labels/metrics) separately.
All arms use the same deterministic case tape, layout, starting pose and budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V1_SRC = ROOT / "leg_closed_loop_20261122_v1" / "src"
DEFAULT_PACK = Path(__file__).resolve().parent / "atlas_routes.npz"
DEFAULT_CONTROL = Path(__file__).resolve().parent / "prior_control.json"
SOURCE_NAMES = ("BodyParts3D right", "BodyParts3D mirrored left", "Open3DModel right")
CALIBRATION_SHAPES = (0, 1, 2, 5)
EVALUATION_SHAPES = (3, 4, 6)
FEMORAL_SOURCES = (0, 2)
WHOLE_SOURCES = (2,)
PROFILES = ("ordinary", "stress")
ROUTES = ("femoral", "groin_to_distal_calf")
DEFAULT_THRESHOLD = 0.82

_V1: dict[str, Any] | None = None
_ACTIVE: "Pilot | None" = None
_INPUT_GUARD_INSTALLED = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_v1() -> dict[str, Any]:
    global _V1
    if _V1 is not None:
        return _V1
    if not V1_SRC.is_dir():
        raise FileNotFoundError(f"v1 source directory is unavailable: {V1_SRC}")
    # dIRD is explicitly disabled before importing environment/material_motion;
    # every arm also carries dird=False below.
    os.environ["LEG_ENABLE_DIRD"] = "0"
    if str(V1_SRC) not in sys.path:
        sys.path.insert(0, str(V1_SRC))
    import environment
    import motion
    from policy import NumpyActor

    _V1 = {"environment": environment, "motion": motion, "NumpyActor": NumpyActor}
    return _V1


def _install_input_guard() -> None:
    """Reject accidental holdout/native/MedSeg opens after v1 imports settle."""

    global _INPUT_GUARD_INSTALLED
    if _INPUT_GUARD_INSTALLED:
        return
    blocked = ("blender", "holdout", "medseg", "ct_geometry", "native_ct",
               "vessel_centerlines_rest.obj", "test_evaluator_labels")

    def guard(event, args):
        if event not in {"open", "os.open"} or not args or isinstance(args[0], int):
            return
        filename = os.fsdecode(args[0]).lower()
        if any(token in filename for token in blocked):
            raise PermissionError(f"v2 route pilot input guard blocked: {args[0]}")

    sys.addaudithook(guard)
    _INPUT_GUARD_INSTALLED = True


def _validate_pack(path: Path) -> dict[str, np.ndarray]:
    path = path.resolve()
    if path.name != "atlas_routes.npz":
        raise ValueError("v2 pilot accepts only atlas_routes.npz")
    if any(token in str(path).lower() for token in ("blender", "holdout", "medseg", "ct_geometry")):
        raise PermissionError(f"Refusing non-atlas geometry path: {path}")
    required = {
        "ys", "origins", "radii", "centers", "neutral_grid", "status", "links", "edge_status"
    }
    with np.load(path, allow_pickle=False) as source:
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"atlas pack missing arrays: {sorted(missing)}")
        data = {name: np.array(source[name], copy=True) for name in source.files}
    if data["centers"].shape != (7, 3, 161, 6, 2):
        raise ValueError(f"unexpected atlas centers shape: {data['centers'].shape}")
    if data["ys"].shape != (7, 161) or data["origins"].shape != (7, 161, 2):
        raise ValueError("atlas posture/profile arrays are not the fixed seven-shape grid")
    if data["radii"].shape != (7, 161, 24) or data["neutral_grid"].shape != (161,):
        raise ValueError("atlas surface grid is not the fixed 161-station chart")
    if data["status"].shape != (3, 161, 6) or data["links"].shape != (3, 160, 6):
        raise ValueError("atlas source/link axis is not the three-source v2 schema")
    if data["edge_status"].shape != (3, 160, 6):
        raise ValueError("atlas edge_status shape is not compatible with Scanner")
    if not np.array_equal(data["links"], data["edge_status"] > 0):
        raise ValueError("atlas links and edge_status disagree")
    if not np.isfinite(data["ys"]).all() or not np.isfinite(data["origins"]).all() \
            or not np.isfinite(data["radii"]).all():
        raise ValueError("atlas posture/surface arrays contain non-finite values")
    data["_path"] = np.asarray(str(path))
    return data


@dataclass(frozen=True)
class ScopedLayout:
    """Deterministic bounded warp of one atlas source, with links preserved."""

    centers: np.ndarray
    identifier: str
    metadata: Mapping[str, Any]
    stations: np.ndarray
    links: np.ndarray
    shared_vertices: tuple = ()

    def __post_init__(self):
        for array in (self.centers, self.stations, self.links):
            array.setflags(write=False)

    def at(self, station: float) -> np.ndarray:
        value = float(station)
        if value < 0.0 or value > len(self.stations) - 1:
            return np.full((self.centers.shape[1], 2), np.nan, dtype=float)
        left = int(np.floor(value))
        fraction = value - left
        if fraction <= 1.e-10 or left == len(self.stations) - 1:
            return self.centers[left].copy()
        right = left + 1
        output = np.full((self.centers.shape[1], 2), np.nan, dtype=float)
        finite = np.isfinite(self.centers[left]).all(axis=-1)
        finite &= np.isfinite(self.centers[right]).all(axis=-1)
        finite &= self.links[left]
        output[finite] = ((1.0 - fraction) * self.centers[left, finite]
                          + fraction * self.centers[right, finite])
        return output


def _segments(finite: np.ndarray, links: np.ndarray) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    current: list[int] = []
    for station in range(len(finite)):
        connected = bool(finite[station]) and (not current or bool(links[station - 1]))
        if connected:
            current.append(station)
        else:
            if current:
                segments.append(np.asarray(current, dtype=int))
            current = [station] if finite[station] else []
    if current:
        segments.append(np.asarray(current, dtype=int))
    return segments


class AtlasLayoutFactory:
    """Scoped layout factory that reads only the already-projected atlas pack."""

    def __init__(self, data: Mapping[str, np.ndarray], scanner, shape: int, pack_sha256: str):
        self.shape = int(shape)
        self.scanner = scanner
        self.pack_sha256 = str(pack_sha256)
        self.stations = np.asarray(scanner.stations, dtype=float).copy()
        self._centers = np.asarray(data["centers"][self.shape], dtype=float).copy()
        self._links = np.asarray(data["links"], dtype=bool).copy()
        self._centers.setflags(write=False)
        self._links.setflags(write=False)
        self.stations.setflags(write=False)

    def build(self, source: int, world_seed: int, profile: str, split: str) -> ScopedLayout:
        source = int(source)
        if source not in range(3):
            raise ValueError(f"v2 atlas layout source must be 0, 1 or 2, got {source}")
        base = self._centers[source].copy()
        links = self._links[source]
        finite = np.isfinite(base).all(axis=-1)
        # Keep the deformation deliberately local: route eligibility and every
        # unsupported topology gap come from the atlas pack and are never filled.
        sequence = np.random.SeedSequence([
            int(world_seed) & 0xFFFFFFFF,
            int(world_seed >> 32) & 0xFFFFFFFF,
            source,
            int(self.shape),
            int(profile == "stress"),
            0xA71A5,
        ])
        rng = np.random.default_rng(sequence)
        scale = 1.15 if profile == "stress" else .68
        for role in range(base.shape[1]):
            for indices in _segments(finite[:, role], links[:, role]):
                if len(indices) < 2:
                    if len(indices):
                        base[indices, role] += rng.normal(0.0, .12 * scale, (len(indices), 2))
                    continue
                knot_step = max(1, len(indices) // 8)
                knots = np.unique(np.r_[indices[::knot_step], indices[-1]])
                shifts = np.clip(rng.normal(0.0, .55 * scale, (len(knots), 2)), -1.5, 1.5)
                if len(knots) == 1:
                    base[indices, role] += shifts[0]
                    continue
                interval = np.clip(np.searchsorted(knots, indices, side="right") - 1,
                                   0, len(knots) - 2)
                width = np.maximum(1, knots[interval + 1] - knots[interval])
                u = ((indices - knots[interval]) / width).astype(float)
                u = (u * u * (3.0 - 2.0 * u))[:, None]
                base[indices, role] += (1.0 - u) * shifts[interval] + u * shifts[interval + 1]
        base.setflags(write=False)
        identifier = hashlib.sha256(
            f"{self.pack_sha256}:{source}:{self.shape}:{world_seed}:{profile}:{split}".encode()
            + base.tobytes()
        ).hexdigest()[:20]
        metadata = {
            "atlas_only": True,
            "source_index": source,
            "source_name": SOURCE_NAMES[source],
            "shape": self.shape,
            "world_seed": int(world_seed),
            "profile": str(profile),
            "split": str(split),
            "pack_sha256": self.pack_sha256,
            "topology": "projected graph links preserved; no unsupported fill",
        }
        return ScopedLayout(base, f"atlas_v2_{identifier}", metadata,
                            self.stations.copy(), links.copy())


def _scanner_from_atlas(data: Mapping[str, np.ndarray], shape: int):
    v1 = _load_v1()
    motion = v1["motion"]
    import scan_kernel

    order = [0, 1, 2, 3, 4, 5]
    centers = np.asarray(data["centers"][int(shape)], dtype=float)[..., order, :]
    origins = np.asarray(data["origins"][int(shape)], dtype=float)
    radii = np.asarray(data["radii"][int(shape)], dtype=float)
    ys = np.asarray(data["ys"][int(shape)], dtype=float)
    contacts = origins[:, None, :] + radii[..., None] * scan_kernel.NORMALS[None]
    chart = (origins, radii, data["origins"][0], data["radii"][0], data["ys"][0])
    surface = motion.Surface(ys, contacts, centers[:3].copy(), contacts[..., 0] < -1.0, shape=chart)
    links = np.asarray(data["links"], dtype=bool)[..., order]
    edge_status = np.asarray(data["edge_status"], dtype=np.uint8)[..., order]
    surface.reference_links = links[:3].copy()
    anatomy = motion.Anatomy(centers, links=links.copy())
    scanner = motion.Scanner(np.asarray(data["neutral_grid"], dtype=float).copy(), surface,
                             anatomy, ct_bounds=None, edge_status=edge_status.copy())
    scanner.v2_atlas_only = True
    scanner.v2_source_names = SOURCE_NAMES
    scanner.v2_shape = int(shape)
    return scanner


@dataclass(frozen=True)
class PublicReaderSnapshot:
    """Replayed public reader with label-free frames and public ranking methods."""

    reader: Any

    @property
    def frames(self):
        return tuple(self.reader.frames)

    @property
    def mixed_posterior(self):
        return tuple(np.asarray(row, dtype=float).copy() for row in self.reader.mixed_posterior)

    def __getattr__(self, name):
        # Preserve the public Reader API (absolute/ranked/options/public_summary)
        # without exposing the evaluator's ground-truth labels.
        return getattr(self.reader, name)


@dataclass
class RouteRollout:
    """Old Rollout-compatible item plus separated public/evaluator views."""

    case: Any
    arm: str
    rollout: Any
    public_reader: PublicReaderSnapshot
    evaluator: Mapping[str, Any]
    scanner: Any
    model: Any
    environment: Any

    @property
    def metrics(self):
        return self.rollout.metrics

    @property
    def steps(self):
        return self.rollout.steps

    @property
    def frames(self):
        return self.public_reader.frames

    @property
    def mixed(self):
        return self.public_reader.mixed_posterior

    @property
    def probes(self):
        return self.rollout.probes

    @property
    def teaching(self):
        return self.rollout.teaching

    @property
    def audit(self):
        return self.rollout.audit

    @property
    def layout_centers(self):
        return self.rollout.layout_centers


def _public_snapshot(result: Any, model: Any, roles: tuple[int, ...]) -> tuple[PublicReaderSnapshot, dict[str, Any]]:
    v1 = _load_v1()
    public_reader_type = None
    # Import through the already selected v1 source path; this does not read
    # geometry files and keeps the replay on the public observation model.
    if "online_reader" in sys.modules:
        public_reader_type = sys.modules["online_reader"].PublicHistoryReader
    if public_reader_type is None:
        from online_reader import PublicHistoryReader as public_reader_type
    reader = public_reader_type(model, roles, True)
    labels = []
    for frame in result.frames:
        station, sector, echoes, frame_labels, quality, region, posterior, unknown, tick = frame
        anonymous = np.full(len(echoes), -1, dtype=np.int16)
        reader.push(station, sector, np.asarray(echoes, dtype=float).copy(), anonymous,
                    quality, region, tick)
        labels.append(np.asarray(frame_labels, dtype=np.int16).copy())
    # The replay has fully public posterior/history state.  Replace the
    # inherited frame archive's placeholder labels with empty anonymous arrays
    # so downstream public embedding code cannot accidentally read identities.
    public_frames = []
    for frame in reader.frames:
        station, sector, echoes, _anonymous, quality, region, posterior, unknown, tick = frame
        public_frames.append((float(station), float(sector), np.asarray(echoes, dtype=float).copy(),
                              np.empty(0, dtype=np.int16), float(quality), int(region),
                              np.asarray(posterior, dtype=float).copy(),
                              np.asarray(unknown, dtype=float).copy(),
                              None if tick is None else int(tick)))
    reader.frames = list(public_frames)
    evaluator = {
        "labels_by_frame": tuple(labels),
        "target_label_mask_by_frame": tuple((row == 0) for row in labels),
        "metrics": dict(result.metrics),
        "features_contain_labels": False,
        "labels_excluded_from_public_frames": True,
        "private_frames": tuple(result.frames),
    }
    return PublicReaderSnapshot(reader), evaluator


def _between(array: np.ndarray, station: float) -> np.ndarray:
    value = float(np.clip(station, 0.0, len(array) - 1))
    lo = int(np.floor(value))
    frac = value - lo
    if frac <= 1.e-10 or lo == len(array) - 1:
        return np.asarray(array[lo]).copy()
    return (1.0 - frac) * array[lo] + frac * array[lo + 1]


def _accessible_upper(env: Any, result: Any, spec: tuple, source: int) -> dict[str, Any]:
    """Approximate per-region geometric ceiling under public FOV/bone prior."""

    v1 = _load_v1()
    scan_kernel = sys.modules["scan_kernel"]
    scanner = env.scanner
    layout_centers = result.layout_centers
    if layout_centers is None:
        centers = scanner._anatomy.centers[int(source)]
        links = scanner._anatomy.links[int(source)]
    else:
        centers = np.asarray(layout_centers, dtype=float)
        links = scanner._anatomy.links[int(source)]
    station_good = np.zeros(spec[2] - spec[1] + 1, dtype=bool)
    for offset, station in enumerate(range(spec[1], spec[2] + 1)):
        center = _between(centers, station)
        if not np.isfinite(center[0]).all():
            continue
        for sector in np.flatnonzero(scanner.valid[station]):
            means = scanner.surface.project(station, int(sector), center)
            gate = scanner.surface.exact_population_shadow(station, int(sector), means)
            lateral, depth = means[0]
            if (np.isfinite(means[0]).all() and -scan_kernel.LATERAL <= lateral <= scan_kernel.LATERAL
                    and scan_kernel.DEPTH_MIN <= depth <= scan_kernel.DEPTH_MAX
                    and float(gate[0]) < .95):
                station_good[offset] = True
                break
    bins = v1["environment"].station_bins(spec, scanner.surface.y)
    region_good = np.zeros(6, dtype=bool)
    for region in range(6):
        region_good[region] = bool(np.any(station_good[np.asarray(spec[4]) == region]))
    return {
        "method": "role-0 atlas centerline, public echo FOV, Surface.exact_population_shadow",
        "bone_prior": "common population occupancy from v1 public model",
        "accessible_mm_by_region": [float(bins[np.asarray(spec[4]) == region][station_good[np.asarray(spec[4]) == region]].sum())
                                    for region in range(6)],
        "accessible_regions": int(region_good.sum()),
        "accessible_mm": float(bins @ station_good),
        "station_mask": station_good.tolist(),
    }


def _make_case(environment_module: Any, route: str, split: str, ordinal: int, source: int,
               shape: int, profile: str, band: int) -> Any:
    route_code = 101 if route == "femoral" else 211
    split_code = 17 if split == "calibration" else 31
    base = 7_000_000_000 + route_code * 10_000_000 + split_code * 100_000 + ordinal * 10
    # Case streams are disjoint from v1's <=~1e9 seed design and from each
    # other, while all arms receive exactly these same three streams.
    challenge = bool((ordinal + source + band + (profile == "stress")) % 4 == 0)
    return environment_module.Case(
        identifier=f"v2_{split}_{route}_{ordinal:03d}",
        split=split,
        source=int(source),
        shape=int(shape),
        task=route,
        profile=profile,
        world_seed=int(base),
        tape_seed=int(base + 1),
        action_seed=int(base + 2),
        band=int(band),
        challenge=challenge,
    )


def _case_plan(environment_module: Any, split: str, route: str) -> tuple[Any, ...]:
    if split not in ("calibration", "evaluation"):
        raise ValueError("split must be calibration or evaluation")
    if route not in ROUTES:
        raise ValueError(f"unknown v2 route: {route}")
    shapes = CALIBRATION_SHAPES if split == "calibration" else EVALUATION_SHAPES
    cases = []
    ordinal = 0
    if route == "femoral":
        sources = FEMORAL_SOURCES
        bands = (0,) if split == "calibration" else (0, 1)
        for shape in shapes:
            for source in sources:
                for profile in PROFILES:
                    for band in bands:
                        cases.append(_make_case(environment_module, route, split, ordinal,
                                                source, shape, profile, band))
                        ordinal += 1
    else:
        # Source 2 is the only complete connected whole-leg route in the atlas.
        sources = WHOLE_SOURCES
        bands = (0, 1) if split == "calibration" else (0, 1, 2, 3)
        for shape in shapes:
            for profile in PROFILES:
                for band in bands:
                    cases.append(_make_case(environment_module, route, split, ordinal,
                                            sources[0], shape, profile, band))
                    ordinal += 1
    expected = 16 if split == "calibration" else 24
    if len(cases) != expected:
        raise AssertionError(f"{route} {split} plan has {len(cases)} cases, expected {expected}")
    return tuple(cases)


def _arm_table(environment_module: Any) -> dict[str, Any]:
    Arm = environment_module.Arm
    return {
        "mpc": Arm("mpc", dird=False),
        "active_mpc": Arm("active_mpc", dird=False),
        "teacher_ppo": Arm("teacher_ppo", "teacher_ppo", dird=False),
        "ppo": Arm("ppo", "ppo", dird=False),
        "teacher_ppo_no_stop": Arm("teacher_ppo_no_stop", "teacher_ppo", stop=False, dird=False),
        "ppo_no_stop": Arm("ppo_no_stop", "ppo", stop=False, dird=False),
    }


class Pilot:
    """Reusable v2 route pilot with shape-cached scanners/environments."""

    def __init__(self, pack: Path, control_info: Mapping[str, Any] | None = None):
        control_info = dict(control_info or {})
        self.v1 = _load_v1()
        _install_input_guard()
        self.pack = Path(pack)
        if self.pack.is_dir():
            self.pack = self.pack / "atlas_routes.npz"
        self.pack = self.pack.resolve()
        self.data = _validate_pack(self.pack)
        self.pack_sha256 = _sha256(self.pack)
        self.threshold = float(control_info.get("threshold", DEFAULT_THRESHOLD))
        if not np.isfinite(self.threshold) or not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        control_path = Path(control_info.get("prior_control", DEFAULT_CONTROL)).resolve()
        with control_path.open("r", encoding="utf-8") as stream:
            control = json.load(stream)
        self.control_path = control_path
        self.weights = self._load_weights(control, control_info)
        self.actors = {
            name: self.v1["NumpyActor"](weights)
            for name, weights in self.weights.items()
        }
        self.arms = _arm_table(self.v1["environment"])
        self._scanners: dict[int, Any] = {}
        self._environments: dict[int, Any] = {}
        self._factories: dict[int, AtlasLayoutFactory] = {}
        self._plans = {
            (split, route): _case_plan(self.v1["environment"], split, route)
            for split in ("calibration", "evaluation")
            for route in ROUTES
        }
        self.metadata = MappingProxyType({
            "pack": str(self.pack),
            "pack_sha256": self.pack_sha256,
            "control": str(control_path),
            "threshold": self.threshold,
            "dird_enabled": False,
            "sources": list(SOURCE_NAMES),
            "atlas_only": True,
            "case_counts": {f"{split}:{route}": len(cases)
                            for (split, route), cases in self._plans.items()},
        })

    @staticmethod
    def _load_weights(control: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
        arms = control.get("arms", {})
        selected = {
            "teacher_ppo": overrides.get("teacher_ppo_weights", arms.get("teacher_ppo", {}).get("weights_npz")),
            "ppo": overrides.get("ppo_weights", arms.get("ppo", {}).get("weights_npz")),
        }
        result = {}
        for name, raw_path in selected.items():
            if not raw_path:
                raise ValueError(f"prior_control has no frozen weights for {name}")
            path = Path(raw_path).resolve()
            if path.suffix != ".npz" or not path.is_file():
                raise FileNotFoundError(f"frozen {name} weights are unavailable: {path}")
            expected = arms.get(name, {}).get("weights_sha256")
            if expected and _sha256(path) != expected:
                raise ValueError(f"frozen {name} weights hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as pack:
                result[name] = {key: np.array(pack[key], copy=True) for key in pack.files}
        return result

    def _environment(self, shape: int):
        shape = int(shape)
        if shape not in self._environments:
            scanner = _scanner_from_atlas(self.data, shape)
            factory = AtlasLayoutFactory(self.data, scanner, shape, self.pack_sha256)
            env = self.v1["environment"].ScanEnvironment(scanner, factory)
            self._scanners[shape] = scanner
            self._factories[shape] = factory
            self._environments[shape] = env
        return self._environments[shape]

    def cached_environment(self, shape: int):
        """Return the atlas-only cached ScanEnvironment for a shape."""

        return self._environment(shape)

    def cached_scanner(self, shape: int):
        """Return the atlas-only cached Scanner for a shape."""

        return self._environment(shape).scanner

    def case_plan(self, split: str | None = None, route: str | None = None) -> tuple[Any, ...]:
        if split is None and route is None:
            return tuple(case for cases in self._plans.values() for case in cases)
        if split is None or route is None:
            raise ValueError("case_plan requires both split and route, or neither")
        return self._plans[(split, route)]

    def run_case(self, case: Any, arm_names: Iterable[str]) -> list[tuple[Any, str, RouteRollout]]:
        if isinstance(case, str):
            matches = [candidate for candidate in self.case_plan() if candidate.identifier == case]
            if len(matches) != 1:
                raise KeyError(f"unknown or ambiguous v2 case identifier: {case}")
            case = matches[0]
        if case.task not in ROUTES or case.source not in (0, 2):
            raise ValueError("v2 pilot cases must use a fixed route and source 0 or 2")
        names = tuple(arm_names)
        if not names:
            raise ValueError("run_case requires at least one arm")
        unknown = sorted(set(names).difference(self.arms))
        if unknown:
            raise KeyError(f"unknown v2 arm(s): {unknown}")
        env = self._environment(case.shape)
        spec = self.v1["environment"].spec_for(env.scanner, case)
        rows = []
        for name in names:
            arm = self.arms[name]
            actor = self.actors.get(arm.policy)
            result = self.v1["environment"].rollout(
                env, case, arm, self.threshold, actor,
                teacher=False, audit=False, forced=False, budget_scale=1.0,
                trial_type=self.v1["motion"].Trial, diagnostics_ids=False,
            )
            if result.metrics.get("dird_probes", 0) != 0:
                raise AssertionError(f"dIRD unexpectedly active in {case.identifier}/{name}")
            public_reader, evaluator = _public_snapshot(result, env.model, tuple(spec[3]))
            evaluator = dict(evaluator)
            evaluator["accessible_upper"] = _accessible_upper(env, result, spec, case.source)
            evaluator["private_layout_centers"] = None if result.layout_centers is None else np.asarray(
                result.layout_centers, dtype=float).copy()
            item = RouteRollout(case, name, result, public_reader,
                                evaluator, env.scanner, env.model, env)
            rows.append((case, name, item))
        return rows


def initialize(pack: str | Path = DEFAULT_PACK,
               control_info: Mapping[str, Any] | None = None) -> Pilot:
    """Initialize and return the isolated v2 atlas pilot.

    ``control_info`` may override ``prior_control`` or ``threshold`` and may
    provide explicit ``teacher_ppo_weights``/``ppo_weights`` paths.  Frozen
    weight hashes from ``prior_control.json`` are checked by default.
    """

    global _ACTIVE
    _ACTIVE = Pilot(Path(pack), control_info)
    return _ACTIVE


def run_case(case: Any, arm_names: Iterable[str], *, pilot: Pilot | None = None):
    """Run paired arms on one case; returns ``(case, arm, RouteRollout)`` rows."""

    active = pilot or _ACTIVE
    if active is None:
        raise RuntimeError("Call initialize(...) before module-level run_case")
    return active.run_case(case, arm_names)


__all__ = [
    "AtlasLayoutFactory", "Pilot", "PublicReaderSnapshot", "RouteRollout",
    "ScopedLayout", "initialize", "run_case", "CALIBRATION_SHAPES",
    "EVALUATION_SHAPES", "ROUTES",
]

