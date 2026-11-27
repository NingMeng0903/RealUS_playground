#!/usr/bin/env python3
"""Compact trajectory-fitting ablation for richer local motion skills.

This artifact is deliberately a *trajectory* study, not a policy study.  It
reuses the constrained 2-D ProDMP basis from the neighbouring
``ab_leg_bifurcation_20261112`` experiment and fits one mean weight matrix for
each of its three existing skills (``advance``, ``sweep_left``, and
``sweep_right``).  The only ablation is the demonstration support:

``old_36``
    The exact 36 legacy training demonstrations (12 per skill) from
    ``learned_prodmp.py``.
``same_support_72``
    The same legacy 36 plus 36 independently drawn demonstrations from the
    same analytic support (24 per skill total).
``diverse_72``
    The same legacy 36 plus 36 demonstrations with smooth response/exit
    perturbations.  Their metadata names nominal advance, left/right recovery,
    oblique/tilt, and controlled-correction strata; these are target strata,
    not policy branches or extra skills.

All three fitted conditions, plus the endpoint-only baseline, are evaluated on
the same 24 independent held-out motions (8 per skill, 4 legacy-support and 4
rich-support motions).  Held-out physical contexts use the frozen dynamic
geometry's 11/19-degree pose knots, contact offsets, and +/-12-degree central
tilt alternatives.  Endpoint feasibility and the 5-mm projection tolerance
are read from that artifact; no new robot, force, collision, or hardware
limit is invented here.

The target curves are analytic and synthetic.  The output therefore measures
representation/generalisation and a geometry-backed reconstruction proxy only;
it does not claim learned task success or a deployable controller.

The default command writes all artifacts under ``rich_skills/`` next to this
file.  A lightweight ``RICH_SKILL_PLOT_DATA.npz`` is always produced; a 2-D
preview PNG is attempted unless ``--no-plot`` is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "ab_leg_bifurcation_20261112"
DYNAMIC_BASE = HERE.parent / "ab_leg_dynamic_20261112"
LEGACY_MODULE_PATH = BASE / "learned_prodmp.py"
LEGACY_PROTOCOL_PATH = BASE / "prodmp_protocol.json"
LEGACY_DIAGNOSTICS_PATH = BASE / "prodmp_diagnostics.json"
LEGACY_DEMOS_PATH = BASE / "prodmp_demos.npz"
GEOMETRY_PATH = DYNAMIC_BASE / "DYNAMIC_GEOMETRY_FROZEN.npz"
OUT_DIR = HERE / "rich_skills"

SEED = 20261112
LEGACY_SEED = 20261112
SAME_SUPPORT_SEED = 2026111201
DIVERSE_SUPPORT_SEED = 2026111202
HOLDOUT_SEED = 2026111203
TRAIN_PER_SKILL_OLD = 12
TRAIN_PER_SKILL_FULL = 24
TEST_PER_SKILL = 8
PHASE_COUNT = 81
PROJECTION_TOLERANCE_MM_FALLBACK = 5.0  # only used after an explicit metadata check

CONDITION_NAMES = ("endpoint_only", "old_36", "same_support_72", "diverse_72")
TRAINING_CONDITIONS = CONDITION_NAMES[1:]
SKILLS = ("advance", "sweep_left", "sweep_right")
RICH_MODES = (
    "advance_oblique",
    "left_recovery",
    "right_recovery",
    "controlled_correction",
    "tilt_exit",
)


def _load_legacy_module():
    """Load the proven neighbouring module without turning the studies into a package."""

    if not LEGACY_MODULE_PATH.is_file():
        raise FileNotFoundError(LEGACY_MODULE_PATH)
    name = "_legacy_learned_prodmp_for_rich_skill_ablation"
    spec = importlib.util.spec_from_file_location(name, LEGACY_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module spec for {LEGACY_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy_module()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    """Hash a parameter vector/array for an auditable train/test split."""

    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_geometry(path: Path = GEOMETRY_PATH) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate only the frozen geometry facts used by this ablation."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    if "metadata_json" not in arrays:
        raise KeyError(f"{path} has no metadata_json")
    metadata = json.loads(str(arrays["metadata_json"].item()))
    expected_pose = np.asarray([0.0, 11.0, 15.0, 19.0])
    pose = np.asarray(arrays["pose_knee_deg"], dtype=float)
    if pose.shape != expected_pose.shape or not np.allclose(pose, expected_pose, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"unexpected frozen pose schedule: {pose}")
    offsets = np.asarray(arrays["view_offsets_mm"], dtype=float)
    tilts = np.asarray(arrays["view_tilts_deg"], dtype=float)
    expected_offsets = np.asarray([-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 0.0, 0.0])
    expected_tilts = np.asarray([0.0] * 7 + [-12.0, 12.0])
    if not np.allclose(offsets, expected_offsets, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"unexpected frozen contact offsets: {offsets}")
    if not np.allclose(tilts, expected_tilts, atol=1.0e-12, rtol=0.0):
        raise ValueError(f"unexpected frozen contact tilts: {tilts}")
    required = (
        "s_mm",
        "posed_views_m",
        "posed_frames",
        "view_projection_feasible",
        "static_found",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"{path} is missing required geometry arrays: {missing}")
    if arrays["posed_views_m"].shape[:3] != (4, len(arrays["s_mm"]), 9):
        raise ValueError("unexpected posed view table shape")
    if arrays["posed_frames"].shape[:3] != (4, len(arrays["s_mm"]), 9):
        raise ValueError("unexpected posed frame table shape")
    if arrays["view_projection_feasible"].shape != (len(arrays["s_mm"]), 9):
        raise ValueError("unexpected projection feasibility shape")
    if arrays["static_found"].shape != (4, len(arrays["s_mm"]), 9):
        raise ValueError("unexpected static endpoint feasibility shape")
    contact_window = metadata.get("contact_window", {})
    tolerance = float(contact_window.get("projection_tolerance_mm", np.nan))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("frozen geometry has no positive projection tolerance")
    return arrays, metadata


@dataclass(frozen=True)
class GeometryContext:
    """A physical/local chart context decoded from the frozen dynamic table."""

    pose_index: int
    station_start: int
    station_goal: int
    contact_index: int
    pose_deg: float
    offset_mm: float
    tilt_deg: float
    start_world_mm: np.ndarray
    goal_world_mm: np.ndarray
    goal_local_mm: np.ndarray
    start_frame: np.ndarray
    endpoint_static: bool
    projection_feasible: bool

    @property
    def step_length_mm(self) -> float:
        return float(np.linalg.norm(self.goal_local_mm))

    def as_dict(self) -> dict[str, Any]:
        return {
            "pose_index": int(self.pose_index),
            "station_start": int(self.station_start),
            "station_goal": int(self.station_goal),
            "contact_index": int(self.contact_index),
            "pose_deg": float(self.pose_deg),
            "offset_mm": float(self.offset_mm),
            "tilt_deg": float(self.tilt_deg),
            "start_world_mm": self.start_world_mm.tolist(),
            "goal_world_mm": self.goal_world_mm.tolist(),
            "goal_local_mm": self.goal_local_mm.tolist(),
            "step_length_mm": self.step_length_mm,
            "endpoint_static": bool(self.endpoint_static),
            "projection_feasible": bool(self.projection_feasible),
        }


def _make_context(
    geometry: Mapping[str, np.ndarray],
    pose_index: int,
    station_start: int,
    station_goal: int,
    contact_index: int,
) -> GeometryContext:
    posed_views = np.asarray(geometry["posed_views_m"], dtype=float)
    posed_frames = np.asarray(geometry["posed_frames"], dtype=float)
    projection = np.asarray(geometry["view_projection_feasible"], dtype=bool)
    static_found = np.asarray(geometry["static_found"], dtype=bool)
    pose_deg = float(np.asarray(geometry["pose_knee_deg"])[pose_index])
    offsets = np.asarray(geometry["view_offsets_mm"], dtype=float)
    tilts = np.asarray(geometry["view_tilts_deg"], dtype=float)
    start_world = posed_views[pose_index, station_start, contact_index] * 1000.0
    goal_world = posed_views[pose_index, station_goal, contact_index] * 1000.0
    frame = posed_frames[pose_index, station_start, contact_index]
    delta = goal_world - start_world
    # The dynamic geometry stores proper 3-D contact frames.  This ablation
    # uses the first two physical frame axes as a local 2-D chart and records
    # the omitted depth component rather than silently treating it as a limit.
    goal_local = np.asarray([float(frame[:, 0] @ delta), float(frame[:, 1] @ delta)])
    endpoint_static = bool(
        static_found[pose_index, station_start, contact_index]
        and static_found[pose_index, station_goal, contact_index]
    )
    projection_feasible = bool(projection[station_start, contact_index] and projection[station_goal, contact_index])
    if not np.isfinite(goal_local).all() or np.linalg.norm(goal_local) <= 1.0e-8:
        raise ValueError("geometry context has a degenerate local endpoint")
    return GeometryContext(
        pose_index=int(pose_index),
        station_start=int(station_start),
        station_goal=int(station_goal),
        contact_index=int(contact_index),
        pose_deg=pose_deg,
        offset_mm=float(offsets[contact_index]),
        tilt_deg=float(tilts[contact_index]),
        start_world_mm=np.asarray(start_world, dtype=float),
        goal_world_mm=np.asarray(goal_world, dtype=float),
        goal_local_mm=goal_local,
        start_frame=np.asarray(frame, dtype=float),
        endpoint_static=endpoint_static,
        projection_feasible=projection_feasible,
    )


def _available_contexts(
    geometry: Mapping[str, np.ndarray],
    *,
    pose_indices: Sequence[int],
    contact_indices: Sequence[int],
    strides: Sequence[int] = (1,),
    avoid: set[tuple[int, int, int, int]] | None = None,
) -> list[GeometryContext]:
    """Enumerate contexts whose endpoint/contact support is present in the table."""

    avoid = avoid or set()
    station_count = int(np.asarray(geometry["s_mm"]).size)
    projection = np.asarray(geometry["view_projection_feasible"], dtype=bool)
    contexts: list[GeometryContext] = []
    for pose in pose_indices:
        for contact in contact_indices:
            for stride in strides:
                stride = int(stride)
                if stride < 1:
                    raise ValueError("context stride must be positive")
                for station in range(station_count - stride):
                    goal = station + stride
                    key = (int(pose), int(station), int(goal), int(contact))
                    if key in avoid:
                        continue
                    if not (projection[station, contact] and projection[goal, contact]):
                        continue
                    candidate = _make_context(geometry, pose, station, goal, contact)
                    if not candidate.endpoint_static:
                        continue
                    contexts.append(candidate)
    return contexts


def _select_stratified_contexts(
    contexts: Sequence[GeometryContext],
    groups: Sequence[tuple[int, int]],
    count: int,
) -> list[GeometryContext]:
    """Select contexts round-robin over pose/contact strata.

    Enumeration order in the frozen table is station-major within each
    contact.  Selecting the first ``count`` rows would therefore silently
    over-represent one offset.  This helper keeps the ablation explicit about
    outer offsets and the two central tilt alternatives.
    """

    if not groups or count < 1:
        raise ValueError("context groups and count must be nonempty")
    pools: dict[tuple[int, int], list[GeometryContext]] = {tuple(group): [] for group in groups}
    for context in contexts:
        key = (int(context.pose_index), int(context.contact_index))
        if key in pools:
            pools[key].append(context)
    if any(not pools[tuple(group)] for group in groups):
        missing = [tuple(group) for group in groups if not pools[tuple(group)]]
        raise ValueError(f"requested geometry context strata are unavailable: {missing}")
    cursors = {tuple(group): 0 for group in groups}
    selected: list[GeometryContext] = []
    for index in range(int(count)):
        group = tuple(groups[index % len(groups)])
        pool = pools[group]
        selected.append(pool[cursors[group] % len(pool)])
        cursors[group] += 1
    return selected


def _smoothstep3(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = 3.0 * s**2 - 2.0 * s**3
    derivative = 6.0 * s - 6.0 * s**2
    return value, derivative


def _rich_chart(
    skill: str,
    phases: Iterable[float],
    parameters: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth local target with explicit response and late-exit terms.

    ``parameters`` are ``(cross_amplitude, longitudinal_bump, phase,
    frequency, modulation, response_strength, exit_lag)``.  Every added term
    is multiplied by ``sin(pi*s)^2`` (and the late term also by a quintic
    smoothstep), so both terminal position and phase derivative are zero for
    the perturbations.  This makes the physical terminal constraints explicit
    before the constrained ProDMP fit, rather than repairing endpoints after
    fitting.
    """

    skill = str(skill)
    if skill not in SKILLS:
        raise ValueError(f"unknown skill {skill!r}")
    s = np.asarray(phases, dtype=float).reshape(-1)
    if not s.size or not np.isfinite(s).all() or np.any((s < 0.0) | (s > 1.0)):
        raise ValueError("phases must be finite and lie in [0,1]")
    p = np.asarray(parameters, dtype=float).reshape(-1)
    if p.size != 7 or not np.isfinite(p).all():
        raise ValueError("rich parameters must contain seven finite values")
    amplitude, longitudinal_bump, phase, frequency, modulation, response, exit_lag = p
    q, qp = LEGACY._smoothstep5(s)
    envelope = np.sin(np.pi * s) ** 2
    envelope_p = np.pi * np.sin(2.0 * np.pi * s)
    argument = 2.0 * np.pi * frequency * s + phase
    oscillation = np.sin(argument)
    oscillation_p = 2.0 * np.pi * frequency * np.cos(argument)
    longitudinal = q + longitudinal_bump * envelope * oscillation
    longitudinal_p = qp + longitudinal_bump * (envelope_p * oscillation + envelope * oscillation_p)

    modulation_argument = 2.0 * np.pi * (0.7 + 0.25 * frequency) * s + 0.7 * phase
    modulation_shape = 1.0 + modulation * np.cos(modulation_argument)
    modulation_p = (
        -modulation
        * 2.0
        * np.pi
        * (0.7 + 0.25 * frequency)
        * np.sin(modulation_argument)
    )
    response_argument = np.pi * (1.0 + 0.35 * frequency) * s + 0.5 * phase
    response_shape = np.sin(response_argument)
    response_p = np.pi * (1.0 + 0.35 * frequency) * np.cos(response_argument)
    late_q, late_qp = _smoothstep3(s)
    late_shape = late_q * envelope
    late_shape_p = late_qp * envelope + late_q * envelope_p

    signed = 1.0 if skill in ("advance", "sweep_left") else -1.0
    cross = signed * (
        amplitude * envelope * modulation_shape
        + response * envelope * response_shape
        + exit_lag * late_shape
    )
    cross_p = signed * (
        amplitude * (envelope_p * modulation_shape + envelope * modulation_p)
        + response * (envelope_p * response_shape + envelope * response_p)
        + exit_lag * late_shape_p
    )
    return np.column_stack((longitudinal, cross)), np.column_stack((longitudinal_p, cross_p))


def _legacy_to_rich(parameters: Sequence[float]) -> np.ndarray:
    p = np.asarray(parameters, dtype=float).reshape(-1)
    if p.size != 5:
        raise ValueError("legacy parameters must contain five values")
    return np.concatenate((p, [0.0, 0.0]))


def _draw_rich_parameters(
    rng: np.random.Generator,
    skill: str,
    mode: str,
    context: GeometryContext,
) -> np.ndarray:
    """Draw one rich target while tying scale to actual pose/contact context."""

    if skill not in SKILLS:
        raise ValueError(skill)
    if mode not in RICH_MODES:
        raise ValueError(mode)
    if skill == "advance":
        amplitude = float(rng.uniform(0.08, 0.18))
    else:
        amplitude = float(rng.uniform(0.20, 0.46))
    # Geometry changes adjust the perturbation magnitude modestly; the
    # physical/local chart remains endpoint-normalised and architecture-free.
    context_scale = 1.0 + 0.003 * context.pose_deg + 0.003 * abs(context.tilt_deg) + 0.001 * abs(context.offset_mm)
    if mode == "advance_oblique":
        amplitude *= 1.08
    if mode == "tilt_exit":
        amplitude *= 1.04
    response_low, response_high = (0.06, 0.18) if "correction" in mode or "recovery" in mode else (0.0, 0.10)
    exit_low, exit_high = (-0.04, 0.16) if "exit" in mode or "correction" in mode else (-0.06, 0.08)
    return np.asarray(
        [
            amplitude * context_scale,
            rng.uniform(-0.05, 0.05),
            rng.uniform(-np.pi, np.pi),
            float(rng.integers(1, 4)),
            rng.uniform(0.05, 0.18),
            rng.uniform(response_low, response_high) * (1.0 + 0.5 * context.pose_deg / 19.0),
            rng.uniform(exit_low, exit_high) * (1.0 + 0.5 * context.tilt_deg / 12.0),
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class Motion:
    skill: str
    family: str
    mode: str
    parameters: np.ndarray
    phases: np.ndarray
    chart: np.ndarray
    derivative: np.ndarray
    context: GeometryContext


def _fit_weights(
    motion: Motion,
    basis: Any,
) -> np.ndarray:
    return LEGACY._constrained_fit_weights(
        motion.phases,
        motion.chart,
        motion.derivative,
        basis,
        LEGACY.LAMBDA,
    )


def _parameter_matrix(motions: Sequence[Motion]) -> np.ndarray:
    """Store legacy five-parameter and rich seven-parameter targets together."""

    output = np.full((len(motions), 7), np.nan, dtype=float)
    for index, motion in enumerate(motions):
        parameters = np.asarray(motion.parameters, dtype=float).reshape(-1)
        if parameters.size > output.shape[1]:
            raise ValueError("motion parameter vector exceeds the seven-column artifact schema")
        output[index, : parameters.size] = parameters
    return output


def _legacy_training_motions(
    legacy_model: Any,
    geometry: Mapping[str, np.ndarray],
) -> list[Motion]:
    """Wrap the exact 36 legacy demos without re-generating or altering them."""

    contexts = _available_contexts(
        geometry,
        pose_indices=(0,),
        contact_indices=(3,),
    )
    if not contexts:
        raise ValueError("no canonical geometry contexts available for legacy metadata")
    motions: list[Motion] = []
    for si, skill in enumerate(SKILLS):
        params = np.asarray(legacy_model.train_parameters[skill], dtype=float)
        charts = np.asarray(legacy_model.train_demos[skill], dtype=float)
        derivatives = np.asarray(legacy_model.train_derivatives[skill], dtype=float)
        if params.shape[0] != TRAIN_PER_SKILL_OLD:
            raise ValueError("legacy model does not contain the expected 12 demonstrations per skill")
        for index in range(TRAIN_PER_SKILL_OLD):
            motions.append(
                Motion(
                    skill=skill,
                    family="legacy",
                    mode="legacy_nominal",
                    parameters=params[index].copy(),
                    phases=np.linspace(0.0, 1.0, charts.shape[1]),
                    chart=charts[index].copy(),
                    derivative=derivatives[index].copy(),
                    context=contexts[(si * TRAIN_PER_SKILL_OLD + index) % len(contexts)],
                )
            )
    return motions


def _extra_legacy_motions(
    rng: np.random.Generator,
    geometry: Mapping[str, np.ndarray],
    basis: Any,
) -> list[Motion]:
    contexts = _available_contexts(geometry, pose_indices=(0, 2), contact_indices=(3,))
    if not contexts:
        raise ValueError("no canonical/modest-bend contexts available for same-support augmentation")
    phases = np.linspace(0.0, 1.0, PHASE_COUNT)
    motions: list[Motion] = []
    for si, skill in enumerate(SKILLS):
        for index in range(TRAIN_PER_SKILL_OLD):
            parameters = LEGACY._draw_parameters(rng, skill)
            chart, derivative = LEGACY._analytic_chart(skill, phases, parameters)
            motion = Motion(
                skill=skill,
                family="legacy",
                mode="legacy_nominal",
                parameters=np.asarray(parameters, dtype=float),
                phases=phases.copy(),
                chart=chart,
                derivative=derivative,
                context=contexts[(si * TRAIN_PER_SKILL_OLD + index) % len(contexts)],
            )
            # Evaluate the fit now so the data contract is checked before the
            # condition table is assembled; weights are recomputed below for
            # a single, auditable condition fit.
            _fit_weights(motion, basis)
            motions.append(motion)
    return motions


def _extra_rich_motions(
    rng: np.random.Generator,
    geometry: Mapping[str, np.ndarray],
    basis: Any,
) -> list[Motion]:
    contexts = _available_contexts(
        geometry,
        pose_indices=(0, 2),
        contact_indices=(0, 2, 4, 6, 7, 8),
    )
    context_groups = [
        (pose, contact)
        for pose in (0, 2)
        for contact in (0, 2, 4, 6, 7, 8)
    ]
    selected_contexts = _select_stratified_contexts(
        contexts,
        context_groups,
        len(SKILLS) * TRAIN_PER_SKILL_OLD,
    )
    phases = np.linspace(0.0, 1.0, PHASE_COUNT)
    motions: list[Motion] = []
    for si, skill in enumerate(SKILLS):
        for index in range(TRAIN_PER_SKILL_OLD):
            context = selected_contexts[si * TRAIN_PER_SKILL_OLD + index]
            mode = RICH_MODES[(si * TRAIN_PER_SKILL_OLD + index) % len(RICH_MODES)]
            parameters = _draw_rich_parameters(rng, skill, mode, context)
            chart, derivative = _rich_chart(skill, phases, parameters)
            motion = Motion(
                skill=skill,
                family="rich",
                mode=mode,
                parameters=parameters,
                phases=phases.copy(),
                chart=chart,
                derivative=derivative,
                context=context,
            )
            _fit_weights(motion, basis)
            motions.append(motion)
    return motions


def _holdout_motions(
    rng: np.random.Generator,
    geometry: Mapping[str, np.ndarray],
) -> list[Motion]:
    """Build one shared, independent old/rich holdout for every condition."""

    contexts = _available_contexts(
        geometry,
        pose_indices=(1, 3),
        contact_indices=(0, 2, 4, 6, 7, 8),
        strides=(1, 2, 3),
    )
    # Each skill sees the same eight context strata in the shared holdout.  A
    # stratum includes the outer +/-24-mm offsets, +/-12-degree tilts, and
    # inner offsets; pose knots are held out at 11/19 degrees.
    context_groups = (
        (1, 0),  # -24 mm outer offset
        (1, 6),  # +24 mm outer offset
        (1, 7),  # -12 degree central tilt
        (1, 8),  # +12 degree central tilt
        (3, 2),
        (3, 4),
        (3, 7),
        (3, 8),
    )
    selected_contexts = _select_stratified_contexts(
        contexts,
        context_groups,
        len(SKILLS) * TEST_PER_SKILL,
    )
    phases = np.linspace(0.0, 1.0, PHASE_COUNT)
    motions: list[Motion] = []
    for si, skill in enumerate(SKILLS):
        for index in range(TEST_PER_SKILL):
            context = selected_contexts[si * TEST_PER_SKILL + index]
            if index < TEST_PER_SKILL // 2:
                parameters = np.asarray(LEGACY._draw_parameters(rng, skill), dtype=float)
                chart, derivative = LEGACY._analytic_chart(skill, phases, parameters)
                family = "legacy"
                mode = "legacy_holdout"
            else:
                mode = RICH_MODES[(si * TEST_PER_SKILL + index) % len(RICH_MODES)]
                parameters = _draw_rich_parameters(rng, skill, mode, context)
                chart, derivative = _rich_chart(skill, phases, parameters)
                family = "rich"
            motions.append(
                Motion(
                    skill=skill,
                    family=family,
                    mode=mode,
                    parameters=parameters,
                    phases=phases.copy(),
                    chart=chart,
                    derivative=derivative,
                    context=context,
                )
            )
    return motions


def _weight_statistics(motions: Sequence[Motion], basis: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_skill: dict[str, list[np.ndarray]] = {skill: [] for skill in SKILLS}
    for motion in motions:
        by_skill[motion.skill].append(_fit_weights(motion, basis))
    weights = {skill: np.asarray(by_skill[skill], dtype=float) for skill in SKILLS}
    means = {skill: np.mean(weights[skill], axis=0) for skill in SKILLS}
    covariances = {
        skill: np.cov(weights[skill].reshape(weights[skill].shape[0], -1), rowvar=False, ddof=1)
        for skill in SKILLS
    }
    return means, covariances


def _chart_from_weights(phases: np.ndarray, weights: np.ndarray, basis: Any) -> tuple[np.ndarray, np.ndarray]:
    return LEGACY._path_from_chart(phases, np.asarray(weights, dtype=float), basis, LEGACY.LAMBDA)


def _evaluate_condition(
    name: str,
    means: Mapping[str, np.ndarray],
    holdout: Sequence[Motion],
    basis: Any,
    projection_tolerance_mm: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for motion_index, motion in enumerate(holdout):
        if name == "endpoint_only":
            weights = LEGACY._endpoint_only_weights(basis, LEGACY.LAMBDA)
        else:
            weights = np.asarray(means[motion.skill], dtype=float)
        prediction_chart, prediction_derivative = _chart_from_weights(motion.phases, weights, basis)
        target_xy, _ = LEGACY._map_chart_to_xy(
            motion.chart,
            motion.derivative,
            np.zeros(2, dtype=float),
            motion.context.goal_local_mm,
        )
        prediction_xy, prediction_dxy = LEGACY._map_chart_to_xy(
            prediction_chart,
            prediction_derivative,
            np.zeros(2, dtype=float),
            motion.context.goal_local_mm,
        )
        error_mm = np.linalg.norm(prediction_xy - target_xy, axis=1)
        target_terminal = target_xy[-1]
        prediction_terminal = prediction_xy[-1]
        terminal_error = float(np.linalg.norm(prediction_terminal - target_terminal))
        # The exact endpoint/velocity constraints are evaluated in chart space.
        endpoint_position_error = float(
            max(
                np.max(np.abs(prediction_chart[0] - np.asarray([0.0, 0.0]))),
                np.max(np.abs(prediction_chart[-1] - np.asarray([1.0, 0.0]))),
            )
        )
        endpoint_velocity_error = float(
            max(np.max(np.abs(prediction_derivative[0])), np.max(np.abs(prediction_derivative[-1])))
        )
        tolerance_fraction = float(np.mean(error_mm <= projection_tolerance_mm))
        proxy_viable = bool(
            motion.context.endpoint_static
            and motion.context.projection_feasible
            and np.max(error_mm) <= projection_tolerance_mm
        )
        rows.append(
            {
                "motion_index": int(motion_index),
                "skill": motion.skill,
                "family": motion.family,
                "mode": motion.mode,
                "pose_deg": float(motion.context.pose_deg),
                "offset_mm": float(motion.context.offset_mm),
                "tilt_deg": float(motion.context.tilt_deg),
                "station_start": int(motion.context.station_start),
                "station_goal": int(motion.context.station_goal),
                "contact_index": int(motion.context.contact_index),
                "step_length_mm": float(motion.context.step_length_mm),
                "endpoint_static": bool(motion.context.endpoint_static),
                "projection_feasible": bool(motion.context.projection_feasible),
                "rmse_mm": float(np.sqrt(np.mean(error_mm**2))),
                "mean_error_mm": float(np.mean(error_mm)),
                "max_error_mm": float(np.max(error_mm)),
                "terminal_position_error_mm": terminal_error,
                "endpoint_position_error_chart": endpoint_position_error,
                "endpoint_velocity_error_chart": endpoint_velocity_error,
                "within_projection_tolerance_fraction": tolerance_fraction,
                "constraint_viability_proxy": proxy_viable,
                "target_xy": target_xy.tolist(),
                "prediction_xy": prediction_xy.tolist(),
                "prediction_velocity_chart": prediction_dxy.tolist(),
            }
        )
    return {"condition": name, "rows": rows}


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty rows")
    out: dict[str, Any] = {
        "motion_count": int(len(rows)),
        "rmse_mm": float(np.mean([row["rmse_mm"] for row in rows])),
        "mean_error_mm": float(np.mean([row["mean_error_mm"] for row in rows])),
        "max_error_mm_mean": float(np.mean([row["max_error_mm"] for row in rows])),
        "max_error_mm_worst": float(np.max([row["max_error_mm"] for row in rows])),
        "terminal_position_error_mm_max": float(np.max([row["terminal_position_error_mm"] for row in rows])),
        "endpoint_position_error_chart_max": float(np.max([row["endpoint_position_error_chart"] for row in rows])),
        "endpoint_velocity_error_chart_max": float(np.max([row["endpoint_velocity_error_chart"] for row in rows])),
        "within_projection_tolerance_fraction": float(
            np.mean([row["within_projection_tolerance_fraction"] for row in rows])
        ),
        "constraint_viability_proxy_rate": float(np.mean([row["constraint_viability_proxy"] for row in rows])),
        "endpoint_static_rate": float(np.mean([row["endpoint_static"] for row in rows])),
        "projection_feasible_rate": float(np.mean([row["projection_feasible"] for row in rows])),
    }
    for family in ("legacy", "rich"):
        family_rows = [row for row in rows if row["family"] == family]
        if family_rows:
            out[f"{family}_rmse_mm"] = float(np.mean([row["rmse_mm"] for row in family_rows]))
            out[f"{family}_max_error_mm_mean"] = float(np.mean([row["max_error_mm"] for row in family_rows]))
            out[f"{family}_within_projection_tolerance_fraction"] = float(
                np.mean([row["within_projection_tolerance_fraction"] for row in family_rows])
            )
            out[f"{family}_constraint_viability_proxy_rate"] = float(
                np.mean([row["constraint_viability_proxy"] for row in family_rows])
            )
    for skill in SKILLS:
        skill_rows = [row for row in rows if row["skill"] == skill]
        if skill_rows:
            out[f"{skill}_rmse_mm"] = float(np.mean([row["rmse_mm"] for row in skill_rows]))
            out[f"{skill}_constraint_viability_proxy_rate"] = float(
                np.mean([row["constraint_viability_proxy"] for row in skill_rows])
            )
    return out


def _derivative_check() -> float:
    phases = np.linspace(0.0, 1.0, 1001)
    chart, derivative = _rich_chart(
        "sweep_left",
        phases,
        np.asarray([0.27, 0.03, 0.4, 2.0, 0.12, 0.08, 0.05]),
    )
    finite = np.gradient(chart, phases[1] - phases[0], axis=0, edge_order=2)
    return float(np.max(np.abs(finite[2:-2] - derivative[2:-2])))


def _write_plot(plot_path: Path, plot_data: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"status": "unavailable", "reason": str(exc)}
    conditions = list(plot_data["condition_names"])
    target = np.asarray(plot_data["target_xy"])
    predictions = np.asarray(plot_data["prediction_xy"])
    families = np.asarray(plot_data["test_family"]).astype(str)
    selected = [index for index, family in enumerate(families) if family == "rich"][:3]
    if len(selected) < 3:
        selected = list(range(min(3, len(families))))
    fig, axes = plt.subplots(1, len(selected), figsize=(13, 4.0), squeeze=False)
    for axis, index in zip(axes[0], selected):
        axis.plot(target[index, :, 0], target[index, :, 1], color="black", linewidth=2.3, label="target")
        for ci, condition in enumerate(conditions):
            if condition == "endpoint_only":
                style = "--"
                alpha = 0.75
            else:
                style = "-"
                alpha = 0.85
            axis.plot(
                predictions[ci, index, :, 0],
                predictions[ci, index, :, 1],
                linestyle=style,
                alpha=alpha,
                linewidth=1.2,
                label=condition,
            )
        axis.set_title(f"motion {index}: {families[index]}")
        axis.set_xlabel("local tangent (mm)")
        axis.set_ylabel("local normal (mm)")
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="datalim")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return {"status": "written", "path": str(plot_path)}


def run(
    *,
    output_dir: Path = OUT_DIR,
    geometry_path: Path = GEOMETRY_PATH,
    make_plot: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry, geometry_metadata = _load_geometry(geometry_path)
    projection_tolerance_mm = float(geometry_metadata["contact_window"]["projection_tolerance_mm"])
    # Keep this fallback explicit, even though _load_geometry rejects missing
    # values.  It makes the emitted protocol unambiguous if the schema grows.
    if not np.isfinite(projection_tolerance_mm) or projection_tolerance_mm <= 0.0:
        projection_tolerance_mm = PROJECTION_TOLERANCE_MM_FALLBACK

    legacy_model = LEGACY.LearnedProDMP.fit_default(seed=LEGACY_SEED)
    if legacy_model.protocol.get("train_per_skill") != TRAIN_PER_SKILL_OLD:
        raise ValueError("legacy fit did not reproduce the 12-per-skill baseline")
    if not (LEGACY_PROTOCOL_PATH.is_file() and LEGACY_DIAGNOSTICS_PATH.is_file() and LEGACY_DEMOS_PATH.is_file()):
        raise FileNotFoundError("legacy ProDMP protocol/diagnostic/demo artifacts are incomplete")
    basis = LEGACY._IntegralBasis(lam=LEGACY.LAMBDA, quadrature_order=64)
    legacy_train = _legacy_training_motions(legacy_model, geometry)
    same_extra = _extra_legacy_motions(np.random.default_rng(SAME_SUPPORT_SEED), geometry, basis)
    rich_extra = _extra_rich_motions(np.random.default_rng(DIVERSE_SUPPORT_SEED), geometry, basis)
    holdout = _holdout_motions(np.random.default_rng(HOLDOUT_SEED), geometry)
    if len(legacy_train) != 36 or len(same_extra) != 36 or len(rich_extra) != 36 or len(holdout) != 24:
        raise AssertionError("unexpected ablation split sizes")

    means_old = {skill: np.asarray(legacy_model.weight_mean[skill], dtype=float).copy() for skill in SKILLS}
    cov_old = {skill: np.asarray(legacy_model.weight_covariance[skill], dtype=float).copy() for skill in SKILLS}
    same_means, same_cov = _weight_statistics(legacy_train + same_extra, basis)
    rich_means, rich_cov = _weight_statistics(legacy_train + rich_extra, basis)
    condition_means = {
        "old_36": means_old,
        "same_support_72": same_means,
        "diverse_72": rich_means,
    }
    condition_covariances = {
        "old_36": cov_old,
        "same_support_72": same_cov,
        "diverse_72": rich_cov,
    }
    evaluations = {
        condition: _evaluate_condition(
            condition,
            means if condition != "endpoint_only" else {},
            holdout,
            basis,
            projection_tolerance_mm,
        )
        for condition, means in (
            [("endpoint_only", {})]
            + [(name, condition_means[name]) for name in TRAINING_CONDITIONS]
        )
    }
    aggregate = {condition: _aggregate_rows(evaluations[condition]["rows"]) for condition in CONDITION_NAMES}

    # The derivative check catches accidental endpoint-repair or finite
    # difference targets before artifacts are written.
    rich_derivative_error = _derivative_check()
    endpoint_constraint_checks: dict[str, Any] = {}
    for condition in CONDITION_NAMES:
        rows = evaluations[condition]["rows"]
        endpoint_constraint_checks[condition] = {
            "max_chart_position_error": float(max(row["endpoint_position_error_chart"] for row in rows)),
            "max_chart_velocity_error": float(max(row["endpoint_velocity_error_chart"] for row in rows)),
            "max_terminal_position_error_mm": float(max(row["terminal_position_error_mm"] for row in rows)),
        }
    if rich_derivative_error > 2.0e-4:
        raise AssertionError(f"rich analytic derivative check failed: {rich_derivative_error}")
    for condition, checks in endpoint_constraint_checks.items():
        if checks["max_chart_position_error"] > 1.0e-12 or checks["max_chart_velocity_error"] > 1.0e-12:
            raise AssertionError(f"endpoint constraint failed for {condition}: {checks}")

    # One row per condition/motion is convenient for downstream audit tools;
    # trajectory arrays are kept in the separate plot/data artifact.
    all_rows = []
    for condition in CONDITION_NAMES:
        for row in evaluations[condition]["rows"]:
            compact = {key: value for key, value in row.items() if key not in ("target_xy", "prediction_xy", "prediction_velocity_chart")}
            compact["condition"] = condition
            all_rows.append(compact)

    target_xy = np.asarray([row["target_xy"] for row in evaluations["endpoint_only"]["rows"]], dtype=float)
    prediction_xy = np.asarray(
        [
            [row["prediction_xy"] for row in evaluations[condition]["rows"]]
            for condition in CONDITION_NAMES
        ],
        dtype=float,
    )
    plot_data = {
        "condition_names": np.asarray(CONDITION_NAMES, dtype="U32"),
        "skill_names": np.asarray([motion.skill for motion in holdout], dtype="U32"),
        "test_family": np.asarray([motion.family for motion in holdout], dtype="U16"),
        "test_mode": np.asarray([motion.mode for motion in holdout], dtype="U32"),
        "target_xy": target_xy,
        "prediction_xy": prediction_xy,
        "test_pose_deg": np.asarray([motion.context.pose_deg for motion in holdout], dtype=float),
        "test_offset_mm": np.asarray([motion.context.offset_mm for motion in holdout], dtype=float),
        "test_tilt_deg": np.asarray([motion.context.tilt_deg for motion in holdout], dtype=float),
        "test_goal_local_mm": np.asarray([motion.context.goal_local_mm for motion in holdout], dtype=float),
    }
    np.savez_compressed(output_dir / "RICH_SKILL_PLOT_DATA.npz", **plot_data)

    # Save the fitted dataset itself, including independent parameter hashes
    # and context coordinates, without mutating the source study.
    training_groups = {
        "old_36": legacy_train,
        "same_support_72": legacy_train + same_extra,
        "diverse_72": legacy_train + rich_extra,
    }
    train_parameter_hashes = {
        condition: {
            skill: [_array_hash(motion.parameters) for motion in motions if motion.skill == skill]
            for skill in SKILLS
        }
        for condition, motions in training_groups.items()
    }
    holdout_parameter_hashes = [_array_hash(motion.parameters) for motion in holdout]
    train_hash_set = {
        digest
        for condition_hashes in train_parameter_hashes.values()
        for skill_hashes in condition_hashes.values()
        for digest in skill_hashes
    }
    hashes_disjoint = not bool(train_hash_set.intersection(holdout_parameter_hashes))
    dataset_payload: dict[str, np.ndarray] = {}
    for condition, motions in training_groups.items():
        for skill in SKILLS:
            selected = [motion for motion in motions if motion.skill == skill]
            dataset_payload[f"{condition}_{skill}_parameters"] = _parameter_matrix(selected)
            dataset_payload[f"{condition}_{skill}_weights"] = np.asarray(
                [_fit_weights(motion, basis) for motion in selected], dtype=float
            )
            dataset_payload[f"{condition}_{skill}_mean"] = np.asarray(condition_means[condition][skill], dtype=float)
            dataset_payload[f"{condition}_{skill}_covariance"] = np.asarray(
                condition_covariances[condition][skill], dtype=float
            )
            dataset_payload[f"{condition}_{skill}_context"] = np.asarray(
                [
                    [
                        motion.context.pose_deg,
                        motion.context.offset_mm,
                        motion.context.tilt_deg,
                        motion.context.station_start,
                        motion.context.station_goal,
                        motion.context.contact_index,
                    ]
                    for motion in selected
                ],
                dtype=float,
            )
    dataset_payload["holdout_parameters"] = _parameter_matrix(holdout)
    dataset_payload["holdout_context"] = np.asarray(
        [
            [
                motion.context.pose_deg,
                motion.context.offset_mm,
                motion.context.tilt_deg,
                motion.context.station_start,
                motion.context.station_goal,
                motion.context.contact_index,
            ]
            for motion in holdout
        ],
        dtype=float,
    )
    dataset_payload["holdout_goal_local_mm"] = np.asarray([motion.context.goal_local_mm for motion in holdout], dtype=float)
    dataset_payload["endpoint_only_weights"] = np.asarray(
        LEGACY._endpoint_only_weights(basis, LEGACY.LAMBDA), dtype=float
    )
    np.savez_compressed(output_dir / "RICH_SKILL_DATASET.npz", **dataset_payload)

    plot_status = {"status": "not_requested"}
    if make_plot:
        plot_status = _write_plot(output_dir / "RICH_SKILL_PREVIEW.png", plot_data)

    protocol = {
        "schema": "rich-skill-learning-ablation-v1",
        "seed": SEED,
        "source_studies": {
            "legacy_module": str(LEGACY_MODULE_PATH.resolve()),
            "legacy_module_sha256": _sha256(LEGACY_MODULE_PATH),
            "legacy_protocol": str(LEGACY_PROTOCOL_PATH.resolve()),
            "legacy_protocol_sha256": _sha256(LEGACY_PROTOCOL_PATH),
            "legacy_diagnostics": str(LEGACY_DIAGNOSTICS_PATH.resolve()),
            "legacy_diagnostics_sha256": _sha256(LEGACY_DIAGNOSTICS_PATH),
            "legacy_demos": str(LEGACY_DEMOS_PATH.resolve()),
            "legacy_demos_sha256": _sha256(LEGACY_DEMOS_PATH),
            "dynamic_geometry": str(geometry_path.resolve()),
            "dynamic_geometry_sha256": _sha256(geometry_path),
        },
        "architecture": {
            "skills": list(SKILLS),
            "basis": "constrained 2-D integrated ProDMP basis reused from learned_prodmp.py",
            "lambda": float(LEGACY.LAMBDA),
            "quadrature_order": 64,
            "fit": "mean and empirical covariance of per-trajectory constrained weights; endpoint position/phase derivative are exact",
            "policy_or_controller": False,
        },
        "conditions": {
            "old_36": {
                "train_demos_total": 36,
                "train_per_skill": 12,
                "support": "exact legacy learned_prodmp train split",
            },
            "same_support_72": {
                "train_demos_total": 72,
                "train_per_skill": 24,
                "support": "legacy analytic support; legacy 36 plus 36 independent same-support draws",
            },
            "diverse_72": {
                "train_demos_total": 72,
                "train_per_skill": 24,
                "support": "legacy 36 plus 36 smooth response/exit perturbation draws",
                "target_strata": list(RICH_MODES),
            },
            "endpoint_only": {
                "train_demos_total": 0,
                "support": "minimum-norm endpoint position/derivative constraints only",
            },
        },
        "split": {
            "train_phases": PHASE_COUNT,
            "test_phases": PHASE_COUNT,
            "test_per_skill": TEST_PER_SKILL,
            "test_total": len(holdout),
            "test_family_counts": {
                "legacy": int(sum(motion.family == "legacy" for motion in holdout)),
                "rich": int(sum(motion.family == "rich" for motion in holdout)),
            },
            "same_holdout_for_all_conditions": True,
            "independent_holdout_rng_seed": HOLDOUT_SEED,
            "train_test_parameter_hashes_disjoint": hashes_disjoint,
            "train_parameter_hashes": train_parameter_hashes,
            "holdout_parameter_hashes": holdout_parameter_hashes,
            "heldout_context_rule": "poses 11/19 degrees, projection-feasible contact endpoints, including outer offsets and central +/-12-degree tilt alternatives",
        },
        "geometry_contract": {
            "source": str(geometry_path.resolve()),
            "pose_angles_deg": np.asarray(geometry["pose_knee_deg"], dtype=float).tolist(),
            "training_pose_states": [0.0, 15.0],
            "heldout_pose_states": [11.0, 19.0],
            "contact_offsets_mm": np.asarray(geometry["view_offsets_mm"], dtype=float).tolist(),
            "contact_tilts_deg": np.asarray(geometry["view_tilts_deg"], dtype=float).tolist(),
            "modeled_total_fov_mm": float(geometry_metadata["contact_window"]["modeled_total_fov_mm"]),
            "projection_tolerance_mm": projection_tolerance_mm,
            "heldout_endpoint_static_rate": float(np.mean([motion.context.endpoint_static for motion in holdout])),
            "heldout_projection_feasible_rate": float(np.mean([motion.context.projection_feasible for motion in holdout])),
            "dird_margin_chosen_min": float(geometry_metadata["public_robot"]["dird_margin_chosen_min"]),
            "dird_margin_chosen_max": float(geometry_metadata["public_robot"]["dird_margin_chosen_max"]),
            "interpretation": "geometry provides local frames, contact support, endpoint masks, and projection tolerance; no continuous IK/collision/force claim",
        },
        "synthetic_target_contract": {
            "legacy": "quintic smoothstep plus sinusoidal bow from learned_prodmp.py",
            "rich": "same endpoint-normalised chart plus smooth signed response and late-exit terms; analytic derivative",
            "rich_parameter_order": ["cross_amplitude", "longitudinal_bump", "phase", "frequency", "modulation", "response_strength", "exit_lag"],
            "rich_modes_are": "target strata for representation coverage, not extra policy branches",
            "real_data": False,
        },
        "diagnostics": {
            "rich_analytic_derivative_max_abs": rich_derivative_error,
            "endpoint_constraint_checks": endpoint_constraint_checks,
            "legacy_diagnostics_snapshot": {
                key: legacy_model.diagnostics[key]
                for key in (
                    "aggregate_heldout_position_rmse_mm",
                    "aggregate_endpoint_only_position_rmse_mm",
                    "train_test_parameter_hashes_disjoint",
                )
                if key in legacy_model.diagnostics
            },
        },
        "outputs": {
            "summary": "RICH_SKILL_SUMMARY.json",
            "protocol": "RICH_SKILL_PROTOCOL.json",
            "metrics": "RICH_SKILL_METRICS.npz",
            "dataset": "RICH_SKILL_DATASET.npz",
            "plot_data": "RICH_SKILL_PLOT_DATA.npz",
            "preview": "RICH_SKILL_PREVIEW.png" if plot_status.get("status") == "written" else None,
        },
        "scope_notes": [
            "All trajectories are synthetic analytic targets; no real demonstrations or robot execution are claimed.",
            "The study fits constrained trajectory weights only and does not add a learned policy, controller, or success classifier.",
            "constraint_viability_proxy means endpoint geometry mask plus max reconstruction error within the frozen 5-mm projection tolerance; it is not continuous path feasibility.",
            "The frozen dynamic geometry's endpoint/static checks are retained as masks, not used to fabricate intermediate IK or collision results.",
        ],
    }
    (output_dir / "RICH_SKILL_PROTOCOL.json").write_text(json.dumps(_jsonable(protocol), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metric_payload: dict[str, np.ndarray] = {
        "condition_names": np.asarray(CONDITION_NAMES, dtype="U32"),
        "test_skill": np.asarray([motion.skill for motion in holdout], dtype="U32"),
        "test_family": np.asarray([motion.family for motion in holdout], dtype="U16"),
        "test_mode": np.asarray([motion.mode for motion in holdout], dtype="U32"),
        "test_pose_deg": np.asarray([motion.context.pose_deg for motion in holdout], dtype=float),
        "test_offset_mm": np.asarray([motion.context.offset_mm for motion in holdout], dtype=float),
        "test_tilt_deg": np.asarray([motion.context.tilt_deg for motion in holdout], dtype=float),
        "test_step_length_mm": np.asarray([motion.context.step_length_mm for motion in holdout], dtype=float),
        "test_endpoint_static": np.asarray([motion.context.endpoint_static for motion in holdout], dtype=bool),
        "test_projection_feasible": np.asarray([motion.context.projection_feasible for motion in holdout], dtype=bool),
        "rmse_mm": np.asarray([[row["rmse_mm"] for row in evaluations[condition]["rows"]] for condition in CONDITION_NAMES]),
        "mean_error_mm": np.asarray([[row["mean_error_mm"] for row in evaluations[condition]["rows"]] for condition in CONDITION_NAMES]),
        "max_error_mm": np.asarray([[row["max_error_mm"] for row in evaluations[condition]["rows"]] for condition in CONDITION_NAMES]),
        "within_projection_tolerance_fraction": np.asarray(
            [[row["within_projection_tolerance_fraction"] for row in evaluations[condition]["rows"]] for condition in CONDITION_NAMES]
        ),
        "constraint_viability_proxy": np.asarray(
            [[row["constraint_viability_proxy"] for row in evaluations[condition]["rows"]] for condition in CONDITION_NAMES],
            dtype=bool,
        ),
        "terminal_position_error_mm": np.asarray(
            [[row["terminal_position_error_mm"] for row in evaluations[condition]["rows"]] for condition in CONDITION_NAMES]
        ),
    }
    np.savez_compressed(output_dir / "RICH_SKILL_METRICS.npz", **metric_payload)

    summary = {
        "schema": protocol["schema"],
        "status": "completed",
        "protocol": "RICH_SKILL_PROTOCOL.json",
        "metrics": "RICH_SKILL_METRICS.npz",
        "dataset": "RICH_SKILL_DATASET.npz",
        "plot_data": "RICH_SKILL_PLOT_DATA.npz",
        "preview": plot_status,
        "conditions": aggregate,
        "condition_deltas_vs_old_36": {
            condition: {
                "rmse_mm_change": float(aggregate[condition]["rmse_mm"] - aggregate["old_36"]["rmse_mm"]),
                "constraint_viability_proxy_rate_change": float(
                    aggregate[condition]["constraint_viability_proxy_rate"]
                    - aggregate["old_36"]["constraint_viability_proxy_rate"]
                ),
            }
            for condition in ("same_support_72", "diverse_72", "endpoint_only")
        },
        "key_readout": {
            "lower_rmse_is_better": True,
            "higher_constraint_viability_proxy_rate_is_better": True,
            "holdout_is_shared_and_independent": True,
            "rich_support_targets_include_recovery_oblique_controlled_exit": True,
        },
    }
    (output_dir / "RICH_SKILL_SUMMARY.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {"protocol": protocol, "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--geometry", type=Path, default=GEOMETRY_PATH)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    result = run(output_dir=args.output_dir, geometry_path=args.geometry, make_plot=not args.no_plot)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

