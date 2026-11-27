"""Closed-loop local acoustic fixture for the A/B continuation experiment.

The policy-visible object is a local action expressed in the measured contact
frame. The fixture keeps the private vessel curve, response coefficients,
identity, and detector truth separate from that object. This makes it possible
to replay a common first probe across policy arms while evaluating a terminal
image on the same hidden acoustic draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import math
from typing import Sequence

import numpy as np


MM = 1.0e-3
YAW_SCALE_RAD = 0.10
SENSOR_SD = 0.035
PRIOR_SD = np.array([0.065, 0.23], dtype=float)
P_MISS_FLOOR = 0.20
P_CLUTTER = 0.10
QUALITY_THRESHOLD = 0.58
CENTER_TOLERANCE_M = 0.0045
AXIS_TOLERANCE_RAD = 0.22
FORK_SUPPORT_M = 0.022
NOMINAL_DEPTH_M = 0.022


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _hash_seed(seed: int, *parts: object) -> int:
    encoded = ":".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _angle_difference(first: float, second: float) -> float:
    return (float(first) - float(second) + math.pi) % (2.0 * math.pi) - math.pi


def sensor_sd(action: "ProbeAction") -> float:
    """Declared quality noise for both the private renderer and model factors."""

    return float(SENSOR_SD * math.sqrt(0.15 / max(float(action.duration_s), 1.0e-9)))


def sensor_variance(action: "ProbeAction") -> float:
    return sensor_sd(action) ** 2


@dataclass(frozen=True)
class WindowSpec:
    """A public acoustic aperture descriptor.

    Physical corridor validity is checked by the robot executor. The acoustic
    renderer uses only the aperture label and its fixed transit/dwell timing.
    """

    name: str
    center_xy_m: tuple[float, float]
    nominal_yaw_rad: float
    transit_s: float
    dwell_s: float
    aperture_role: str


def constrained_windows() -> tuple[WindowSpec, WindowSpec]:
    return (
        WindowSpec("window_left", (-0.016, 0.018), +0.10, 0.72, 0.34, "left_terminal"),
        WindowSpec("window_right", (+0.016, 0.018), -0.10, 0.72, 0.34, "right_terminal"),
    )


@dataclass(frozen=True)
class ProbeAction:
    window: str
    xy_m: tuple[float, float]
    yaw_rad: float
    duration_s: float
    kind: str = "probe"

    @property
    def xy_mm(self) -> tuple[float, float]:
        return (float(self.xy_m[0] / MM), float(self.xy_m[1] / MM))

    @property
    def yaw_deg(self) -> float:
        return float(np.rad2deg(self.yaw_rad))

    @property
    def tilt_rad(self) -> float:
        """Signed rotation about the material-frame x axis.

        The renderer historically called this field ``yaw_rad``. The physical
        interpretation in this fixture is a signed tilt alpha.
        """

        return float(self.yaw_rad)

    def as_feature(self) -> np.ndarray:
        return np.array([1.0, float(self.yaw_rad) / YAW_SCALE_RAD], dtype=float)


def fixture_actions() -> dict[str, ProbeAction]:
    """Canonical probe and terminal actions for the closed-loop run."""

    return {
        "probe_left": ProbeAction("window_left", (-0.016, 0.013), +0.07, 0.72, "probe"),
        "probe_right": ProbeAction("window_right", (+0.016, 0.013), -0.07, 0.72, "probe"),
        "probe_center_left": ProbeAction("window_left", (0.0, 0.0), +0.07, 0.86, "probe"),
        "probe_center_right": ProbeAction("window_right", (0.0, 0.0), -0.07, 0.86, "probe"),
        "terminal_left": ProbeAction("window_left", (-0.016, 0.018), +0.10, 0.94, "terminal"),
        "terminal_right": ProbeAction("window_right", (+0.016, 0.018), -0.10, 0.94, "terminal"),
    }


@dataclass(frozen=True)
class PublicObservation:
    """Image information available to the planner.

    A missed target carries NaN geometry fields. A candidate has no
    target/distractor truth label, so a false positive is observationally
    indistinguishable until terminal evaluation checks the hidden frame.
    """

    event_id: str
    address: str
    window: str
    elapsed_s: float
    exit_state: str
    detected: bool
    confidence: float
    quality: float
    center_offset_m: float
    axis_angle_rad: float
    candidate_type: str

    @property
    def missing(self) -> bool:
        return not self.detected

    @property
    def center_offset_mm(self) -> float:
        return float(self.center_offset_m / MM) if np.isfinite(self.center_offset_m) else float("nan")

    @property
    def axis_angle_deg(self) -> float:
        return float(np.rad2deg(self.axis_angle_rad)) if np.isfinite(self.axis_angle_rad) else float("nan")


@dataclass(frozen=True)
class HiddenFrame:
    """Private capture metadata kept out of :class:`PublicObservation`."""

    kind: str
    target_quality: float
    target_center_error_m: float
    target_axis_error_rad: float
    target_visible: bool
    detector_draw: float


@dataclass(frozen=True)
class CapturedObservation:
    public: PublicObservation
    hidden: HiddenFrame


@dataclass(frozen=True)
class PrivateScenario:
    seed: int
    address: str
    target_id: int
    curve_offset_m: float
    curve_slope: float
    curve_curvature: float
    target_depth_m: float
    response: np.ndarray  # w=(b,beta), used with H=[1,yaw/.10]
    distractor_xy_m: np.ndarray
    distractor_yaw_rad: float
    acoustic_state: int

    def true_curve_y(self, x_m: float) -> float:
        return 0.018 + self.curve_offset_m + self.curve_slope * float(x_m) + 0.5 * self.curve_curvature * float(x_m) ** 2


@dataclass(frozen=True)
class TerminalTruth:
    target_visible: bool
    target_identity_correct: bool
    quality_pass: bool
    center_error_m: float
    axis_error_rad: float
    fresh_image: bool
    hidden_kind: str


def make_scenario(seed: int, address: str = "left_leg:h0.54:theta1.2") -> PrivateScenario:
    rng = np.random.default_rng(seed)
    response = np.array([rng.normal(0.0, PRIOR_SD[0]), rng.normal(0.0, PRIOR_SD[1])], dtype=float)
    return PrivateScenario(
        seed=int(seed),
        address=address,
        target_id=int(rng.integers(10_000, 99_999)),
        curve_offset_m=float(rng.uniform(-0.003, 0.003)),
        curve_slope=float(rng.uniform(-0.08, 0.08)),
        curve_curvature=float(rng.uniform(-0.35, 0.35)),
        target_depth_m=float(rng.uniform(0.018, 0.026)),
        response=response,
        distractor_xy_m=np.array([rng.uniform(-0.020, 0.020), rng.uniform(0.007, 0.030)], dtype=float),
        distractor_yaw_rad=float(rng.uniform(-math.pi, math.pi)),
        acoustic_state=int(rng.integers(0, 3)),
    )


def paired_scenarios(seed: int, count: int) -> list[PrivateScenario]:
    return [make_scenario(_hash_seed(seed, "scenario", i), f"left_leg:address:{i:04d}") for i in range(count)]


def _window_index(name: str) -> int:
    return 0 if "left" in name else 1


def geometry_value(action: ProbeAction, *, nominal_offset_m: float = 0.0, nominal_slope: float = 0.0, nominal_depth_m: float = NOMINAL_DEPTH_M) -> float:
    """Nominal image quality from the public atlas geometry.

    A positive material-frame tilt rotates +z towards -y. A beam reaching a
    target at depth therefore has the lateral intercept
    ``y_beam = y - depth*tan(alpha)``. The fork score uses this projected
    two-dimensional geometry and keeps a broad 22 mm diagnostic support.
    """

    x, y = map(float, action.xy_m)
    expected_y = 0.018 + nominal_offset_m + nominal_slope * x
    y_beam = y - nominal_depth_m * math.tan(action.tilt_rad)
    center_error = abs(y_beam - expected_y)
    axis_error = abs(_angle_difference(action.yaw_rad, 0.0))
    baseline = 0.64
    return float(baseline - 11.0 * min(center_error, FORK_SUPPORT_M) - 0.38 * axis_error)


def _true_geometry_value(scenario: PrivateScenario, action: ProbeAction) -> tuple[float, float, float]:
    x, y = map(float, action.xy_m)
    target_y = scenario.true_curve_y(x)
    y_beam = y - scenario.target_depth_m * math.tan(action.tilt_rad)
    center_error = abs(y_beam - target_y)
    # Projected 3-D vessel tangent. The fixture's local z coordinate is depth
    # and the vessel has no public out-of-plane slope, so this is a declared
    # x-y tangent approximation rather than a full ultrasound ray model.
    vessel_axis = math.atan(scenario.curve_slope + scenario.curve_curvature * x)
    axis_error = abs(_angle_difference(action.tilt_rad, vessel_axis))
    baseline = 0.64
    score = baseline - 11.0 * min(center_error, FORK_SUPPORT_M) - 0.38 * axis_error
    return float(score), float(center_error), float(axis_error)


def _noise(seed: int, scenario: PrivateScenario, action: ProbeAction, event_id: str, channel: str) -> float:
    stream = _hash_seed(seed, scenario.seed, scenario.address, event_id, action.window, channel)
    return float(np.random.default_rng(stream).normal())


def _draw_detector(seed: int, scenario: PrivateScenario, action: ProbeAction, event_id: str, channel: str) -> float:
    stream = _hash_seed(seed, scenario.seed, scenario.address, event_id, action.window, channel)
    return float(np.random.default_rng(stream).random())


def _capture_hidden(scenario: PrivateScenario, action: ProbeAction, *, event_id: str, seed: int, informative: bool) -> HiddenFrame:
    true_geometry, center_error, axis_error = _true_geometry_value(scenario, action)
    response = float(action.as_feature() @ scenario.response)
    target_quality = true_geometry + response + sensor_sd(action) * _noise(seed, scenario, action, event_id, "quality")
    # The detector law is fixed by the broad fixture support. It does not
    # inspect the latent response coefficient w.
    support = bool(center_error <= FORK_SUPPORT_M)
    target_prob = 0.80 if informative and support else 0.0 if informative else 0.5
    visible = bool(_draw_detector(seed, scenario, action, event_id, "target") < target_prob)
    # Spurious candidates are sampled only after a true target miss.
    clutter = bool((not visible) and _draw_detector(seed, scenario, action, event_id, "clutter") < P_CLUTTER)
    kind = "target" if visible and not clutter else "distractor" if clutter else "missing"
    return HiddenFrame(kind, target_quality, center_error, axis_error, visible, float(_draw_detector(seed, scenario, action, event_id, "target_record")))


def _public_from_hidden(scenario: PrivateScenario, action: ProbeAction, hidden: HiddenFrame, *, event_id: str, elapsed_s: float, exit_state: str, seed: int) -> PublicObservation:
    if hidden.kind == "missing":
        return PublicObservation(event_id, scenario.address, action.window, elapsed_s, exit_state, False, 0.0, float("nan"), float("nan"), float("nan"), "missing")
    if hidden.kind == "target":
        confidence = float(_sigmoid((hidden.target_quality - 0.34) / 0.14))
        # Center/axis are deliberately omitted at the probe stage. Exposing
        # them only for true targets would reveal the target/clutter branch.
        return PublicObservation(event_id, scenario.address, action.window, elapsed_s, exit_state, True, confidence, hidden.target_quality, float("nan"), float("nan"), "candidate")
    fake_quality = 0.12 + 0.08 * _noise(seed, scenario, action, event_id, "clutter_public")
    return PublicObservation(event_id, scenario.address, action.window, elapsed_s, exit_state, True, float(_sigmoid((fake_quality - 0.34) / 0.14)), fake_quality, float("nan"), float("nan"), "candidate")


def _render_nondiagnostic(scenario: PrivateScenario, action: ProbeAction, *, event_id: str, elapsed_s: float, seed: int, exit_state: str) -> CapturedObservation:
    """Produce a public packet independent of hidden anatomy and response."""

    stream = np.random.default_rng(_hash_seed(seed, "nondiagnostic", event_id, action.window, action.xy_m, action.yaw_rad))
    detected = bool(stream.random() < (1.0 - P_CLUTTER))
    if detected:
        quality = float(0.24 + 0.08 * stream.normal())
        confidence = float(_sigmoid((quality - 0.16) / 0.14))
        public = PublicObservation(event_id, scenario.address, action.window, elapsed_s, exit_state, True, confidence, quality, float("nan"), float("nan"), "candidate")
    else:
        public = PublicObservation(event_id, scenario.address, action.window, elapsed_s, exit_state, False, 0.0, float("nan"), float("nan"), float("nan"), "missing")
    hidden = HiddenFrame("nondiagnostic", float("nan"), float("nan"), float("nan"), False, float(stream.random()))
    return CapturedObservation(public, hidden)


def render_capture(scenario: PrivateScenario, action: ProbeAction, *, event_id: str, elapsed_s: float, seed: int = 721311, exit_state: str = "checked", informative: bool = True) -> CapturedObservation:
    if not informative:
        return _render_nondiagnostic(scenario, action, event_id=event_id, elapsed_s=elapsed_s, seed=seed, exit_state=exit_state)
    hidden = _capture_hidden(scenario, action, event_id=event_id, seed=seed, informative=informative)
    public = _public_from_hidden(scenario, action, hidden, event_id=event_id, elapsed_s=elapsed_s, exit_state=exit_state, seed=seed)
    return CapturedObservation(public, hidden)


def render_observation(scenario: PrivateScenario, action: ProbeAction, *, event_id: str, elapsed_s: float, seed: int = 721311, exit_state: str = "checked", informative: bool = True) -> PublicObservation:
    return render_capture(scenario, action, event_id=event_id, elapsed_s=elapsed_s, seed=seed, exit_state=exit_state, informative=informative).public


def terminal_truth(scenario: PrivateScenario, action: ProbeAction, *, event_id: str, seed: int = 721311, capture: CapturedObservation | None = None) -> TerminalTruth:
    """Evaluate the same terminal frame exposed to the policy."""

    capture = capture or render_capture(scenario, action, event_id=event_id, elapsed_s=action.duration_s, seed=seed, informative=True)
    if capture.hidden.kind == "nondiagnostic":
        # The probe channel may be intentionally uninformative, while the
        # terminal acquisition still uses the physical acoustic renderer.
        capture = render_capture(scenario, action, event_id=event_id, elapsed_s=action.duration_s, seed=seed, informative=True)
    hidden = capture.hidden
    quality_pass = bool(hidden.kind == "target" and hidden.target_quality >= QUALITY_THRESHOLD and hidden.target_center_error_m <= CENTER_TOLERANCE_M and hidden.target_axis_error_rad <= AXIS_TOLERANCE_RAD)
    return TerminalTruth(hidden.kind == "target", hidden.kind == "target", quality_pass, hidden.target_center_error_m, hidden.target_axis_error_rad, True, hidden.kind)


def terminal_success(scenario: PrivateScenario, action: ProbeAction, *, event_id: str, elapsed_s: float, deadline_s: float, seed: int = 721311, capture: CapturedObservation | None = None) -> tuple[bool, TerminalTruth]:
    truth = terminal_truth(scenario, action, event_id=event_id, seed=seed, capture=capture)
    return bool(elapsed_s <= deadline_s and truth.quality_pass), truth


@dataclass
class GaussianAddressMemory:
    """Natural parameters for the stable accepted candidate subset."""

    prior_mean: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    prior_covariance: np.ndarray = field(default_factory=lambda: np.diag(PRIOR_SD**2))
    observation_variance: float = SENSOR_SD**2
    precision: np.ndarray | None = None
    information: np.ndarray | None = None
    event_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        prior_precision = np.linalg.inv(np.asarray(self.prior_covariance, dtype=float))
        prior_information = prior_precision @ np.asarray(self.prior_mean, dtype=float)
        self.precision = prior_precision.copy() if self.precision is None else np.asarray(self.precision, dtype=float)
        self.information = prior_information.copy() if self.information is None else np.asarray(self.information, dtype=float)

    def update(self, event_id: str, design: Sequence[float], residual: float, variance: float | None = None) -> bool:
        if str(event_id) in self.event_ids:
            return False
        H = np.asarray(design, dtype=float).reshape(2)
        R = float(self.observation_variance if variance is None else variance)
        if not np.isfinite(R) or R <= 0.0:
            raise ValueError("observation variance must be positive")
        self.precision = self.precision + np.outer(H, H) / R
        self.information = self.information + H * float(residual) / R
        self.event_ids.add(str(event_id))
        return True

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        covariance = np.linalg.inv(self.precision)
        return covariance @ self.information, covariance

    def predict(self, design: Sequence[float], *, geometry_value_: float = 0.0, geometry_value: float | None = None, observation_variance: float | None = None) -> tuple[float, float]:
        if geometry_value is not None:
            geometry_value_ = geometry_value
        H = np.asarray(design, dtype=float).reshape(2)
        mean, covariance = self.posterior()
        variance = self.observation_variance if observation_variance is None else float(observation_variance)
        return float(geometry_value_ + H @ mean), float(H @ covariance @ H + variance)

    def snapshot(self) -> dict[str, object]:
        mean, covariance = self.posterior()
        return {"mean": mean.tolist(), "covariance": covariance.tolist(), "events": sorted(self.event_ids)}


def _normal_pdf(value: float | np.ndarray, mean: float | np.ndarray, variance: float) -> float | np.ndarray:
    variance = max(float(variance), 1.0e-12)
    return np.exp(-0.5 * (np.asarray(value) - np.asarray(mean)) ** 2 / variance) / math.sqrt(2.0 * math.pi * variance)


@dataclass
class MixtureAddressMemory:
    """Association mixture of Gaussian natural-parameter components.

    Each candidate branches into a target component that receives the [1,s]
    Gaussian update and a clutter component that keeps its natural parameters.
    Missing frames only update component weights through the fixed detector law.
    A five-node Gauss-Hermite quality quadrature evaluates each predictive
    detector probability. The prior and all detector constants are fixed.
    """

    prior_mean: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    prior_covariance: np.ndarray = field(default_factory=lambda: np.diag(PRIOR_SD**2))
    observation_variance: float = SENSOR_SD**2
    quadrature_order: int = 5
    components: list[GaussianAddressMemory] | None = None
    weights: np.ndarray | None = None
    event_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.components is None:
            self.components = [GaussianAddressMemory(self.prior_mean.copy(), self.prior_covariance.copy(), self.observation_variance)]
        if self.weights is None:
            self.weights = np.ones(len(self.components), dtype=float) / len(self.components)
        else:
            self.weights = np.asarray(self.weights, dtype=float)
            self.weights /= np.sum(self.weights)

    @staticmethod
    def _copy_component(component: GaussianAddressMemory) -> GaussianAddressMemory:
        return GaussianAddressMemory(component.prior_mean.copy(), component.prior_covariance.copy(), component.observation_variance, component.precision.copy(), component.information.copy(), set(component.event_ids))

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        means, covariances = zip(*(component.posterior() for component in self.components))
        means = np.asarray(means)
        mean = np.sum(self.weights[:, None] * means, axis=0)
        covariance = np.zeros((2, 2), dtype=float)
        for weight, local_mean, local_covariance in zip(self.weights, means, covariances):
            delta = local_mean - mean
            covariance += weight * (local_covariance + np.outer(delta, delta))
        return mean, covariance

    def predict(self, design: Sequence[float], *, geometry_value_: float = 0.0, geometry_value: float | None = None, observation_variance: float | None = None) -> tuple[float, float]:
        if geometry_value is not None:
            geometry_value_ = geometry_value
        H = np.asarray(design, dtype=float).reshape(2)
        variance_for_measurement = self.observation_variance if observation_variance is None else float(observation_variance)
        means, variances = zip(*(component.predict(H, geometry_value_=geometry_value_, observation_variance=variance_for_measurement) for component in self.components))
        mean = float(np.sum(self.weights * np.asarray(means)))
        variance = float(np.sum(self.weights * (np.asarray(variances) + (np.asarray(means) - mean) ** 2)))
        return mean, variance

    def _detect_probability(self, component: GaussianAddressMemory, H: np.ndarray, geometry_value_: float, action: ProbeAction | None = None) -> float:
        """Fixed p_real=.8 on the known broad support, independent of w."""

        if action is None:
            support_probability = 1.0
        else:
            support_probability = geometry_support_probability(action)
        return float(0.80 * support_probability)

    def condition(self, event_id: str, design: Sequence[float], observation: PublicObservation, *, geometry_value_: float = 0.0, geometry_value: float | None = None, observation_variance: float | None = None, action: ProbeAction | None = None) -> bool:
        if str(event_id) in self.event_ids:
            return False
        if geometry_value is not None:
            geometry_value_ = geometry_value
        H = np.asarray(design, dtype=float).reshape(2)
        variance_for_measurement = self.observation_variance if observation_variance is None else float(observation_variance)
        new_components = []
        log_weights = []
        variance_for_measurement = self.observation_variance if observation_variance is None else float(observation_variance)
        if observation.detected and np.isfinite(observation.quality):
            for weight, component in zip(self.weights, self.components):
                predicted_mean, predicted_variance = component.predict(H, geometry_value_=geometry_value_, observation_variance=variance_for_measurement)
                p_real = self._detect_probability(component, H, geometry_value_, action)
                target_likelihood = p_real * float(_normal_pdf(float(observation.quality), predicted_mean, predicted_variance))
                clutter_likelihood = (1.0 - p_real) * P_CLUTTER * float(_normal_pdf(float(observation.quality), 0.12, 0.08**2))
                target_component = self._copy_component(component)
                target_component.update(str(event_id), H, float(observation.quality - geometry_value_), variance_for_measurement)
                new_components.extend([target_component, self._copy_component(component)])
                new_weights = [weight * target_likelihood, weight * clutter_likelihood]
                for branch_weight in new_weights:
                    log_weights.append(math.log(max(float(branch_weight), 1.0e-300)))
        else:
            for weight, component in zip(self.weights, self.components):
                p_detect = self._detect_probability(component, H, geometry_value_, action)
                new_components.append(self._copy_component(component))
                log_weights.append(math.log(max(float(weight * (1.0 - p_detect) * (1.0 - P_CLUTTER)), 1.0e-300)))
        max_log = max(log_weights)
        new_weights_array = np.exp(np.asarray(log_weights) - max_log)
        total = float(np.sum(new_weights_array))
        if not np.isfinite(total) or total <= 0.0:
            return False
        self.components = new_components
        self.weights = new_weights_array / total
        self.event_ids.add(str(event_id))
        return True

    def outcome_probabilities(self, action: ProbeAction, *, geometry_value_: float = 0.0, geometry_value: float | None = None, informative: bool = True, observation_variance: float | None = None) -> dict[str, float]:
        if geometry_value is not None:
            geometry_value_ = geometry_value
        if not informative:
            return {"candidate": 0.0, "clutter": P_CLUTTER, "missing": 1.0 - P_CLUTTER}
        H = response_design(action)
        p_target = float(sum(weight * self._detect_probability(component, H, geometry_value_, action) for weight, component in zip(self.weights, self.components)))
        p_clutter = float((1.0 - p_target) * P_CLUTTER)
        return {"candidate": p_target, "clutter": p_clutter, "missing": max(0.0, 1.0 - p_target - p_clutter)}

    def snapshot(self) -> dict[str, object]:
        mean, covariance = self.posterior()
        return {"mean": mean.tolist(), "covariance": covariance.tolist(), "events": sorted(self.event_ids), "quadrature_order": self.quadrature_order, "component_count": len(self.components)}


def response_design(action: ProbeAction, *, frame_yaw_rad: float = 0.0, frame_yaw_deg: float | None = None) -> np.ndarray:
    if frame_yaw_deg is not None:
        frame_yaw_rad = float(np.deg2rad(frame_yaw_deg))
    return np.array([1.0, _angle_difference(action.yaw_rad, frame_yaw_rad) / YAW_SCALE_RAD], dtype=float)


def condition(memory: GaussianAddressMemory | MixtureAddressMemory, action: ProbeAction, observation: PublicObservation, *, geometry_value_: float = 0.0, geometry_value: float | None = None, event_id: str | None = None) -> bool:
    """Condition the public response belief once per event.

    The association mixture consumes missing packets as detector evidence.
    The simpler Gaussian memory keeps its stable accepted-candidate subset.
    """

    if geometry_value is not None:
        geometry_value_ = geometry_value
    if isinstance(memory, MixtureAddressMemory):
        return memory.condition(observation.event_id if event_id is None else str(event_id), response_design(action), observation, geometry_value_=geometry_value_, observation_variance=sensor_variance(action), action=action)
    if not observation.detected or observation.candidate_type != "candidate" or not np.isfinite(observation.quality) or observation.confidence < 0.50:
        return False
    return memory.update(observation.event_id if event_id is None else str(event_id), response_design(action), float(observation.quality - geometry_value_), sensor_variance(action))


def scan_probability(action: ProbeAction, memory: GaussianAddressMemory | MixtureAddressMemory | None = None, *, geometry_value_: float | None = None, geometry_value: float | None = None, informative: bool = True) -> float:
    if geometry_value is not None:
        geometry_value_ = geometry_value
    if geometry_value_ is None:
        geometry_value_ = geometry_value_fn(action)
    if not informative:
        return 0.5
    design = response_design(action)
    factor = 0.80 * terminal_geometry_probability(action)
    if isinstance(memory, MixtureAddressMemory):
        chance = 0.0
        for weight, component in zip(memory.weights, memory.components):
            mean, variance = component.predict(design, geometry_value_=geometry_value_, observation_variance=sensor_variance(action))
            chance += weight * _normal_cdf((mean - QUALITY_THRESHOLD) / math.sqrt(max(variance, 1.0e-12)))
        return float(factor * chance)
    if memory is None:
        mean = float(geometry_value_)
        variance = float(np.dot(PRIOR_SD**2, design ** 2) + sensor_variance(action))
    else:
        mean, variance = memory.predict(design, geometry_value_=geometry_value_, observation_variance=sensor_variance(action))
    return float(factor * _normal_cdf((mean - QUALITY_THRESHOLD) / math.sqrt(max(variance, 1.0e-12))))


def geometry_value_fn(action: ProbeAction) -> float:
    return geometry_value(action)


@lru_cache(maxsize=512)
def _geometry_probability(action: ProbeAction, terminal: bool) -> float:
    """Integrate uniform center offset exactly and the other public nuisances by quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(5)
    slopes, curvature, depth = np.meshgrid(.08 * nodes, .35 * nodes, NOMINAL_DEPTH_M + .004 * nodes, indexing="ij")
    density = np.einsum("i,j,k->ijk", weights, weights, weights) / 8.0
    x, y = action.xy_m
    error = y - depth * np.tan(action.tilt_rad) - (.018 + slopes * x + .5 * curvature * x**2)
    limit = CENTER_TOLERANCE_M if terminal else FORK_SUPPORT_M
    overlap = np.maximum(0., np.minimum(.003, error + limit) - np.maximum(-.003, error - limit)) / .006
    if terminal:
        overlap *= np.abs(action.tilt_rad - np.arctan(slopes + curvature * x)) <= AXIS_TOLERANCE_RAD
    return float(np.sum(density * overlap))


def geometry_support_probability(action: ProbeAction) -> float:
    return _geometry_probability(action, False)


def detection_probability(action: ProbeAction, *, informative: bool = True) -> float:
    return float(0.80 * geometry_support_probability(action)) if informative else 0.0


def terminal_geometry_probability(action: ProbeAction) -> float:
    return _geometry_probability(action, True)


def predict_outcomes(action: ProbeAction, memory: GaussianAddressMemory | MixtureAddressMemory | None = None, *, geometry_value_: float | None = None, geometry_value: float | None = None, informative: bool = True, visible_prior: float = 0.80, clutter_prior: float = P_CLUTTER) -> tuple[dict[str, object], ...]:
    if geometry_value is not None:
        geometry_value_ = geometry_value
    if geometry_value_ is None:
        geometry_value_ = geometry_value_fn(action)
    H = response_design(action)
    variance_for_measurement = sensor_variance(action)
    if isinstance(memory, MixtureAddressMemory):
        components = list(zip(memory.weights, memory.components))
    elif memory is None:
        components = [(1.0, GaussianAddressMemory())]
    else:
        components = [(1.0, memory)]
    target_nodes, target_weights = np.polynomial.hermite.hermgauss(5)
    clutter_nodes, clutter_weights = np.polynomial.hermite.hermgauss(3)
    target_weights = target_weights / math.sqrt(math.pi)
    clutter_weights = clutter_weights / math.sqrt(math.pi)
    outcomes = []
    for component_weight, component in components:
        predicted_mean, predicted_variance = component.predict(H, geometry_value_=geometry_value_, observation_variance=variance_for_measurement)
        quality_scale = math.sqrt(max(float(predicted_variance), 1.0e-12))
        p_real = detection_probability(action, informative=informative)
        for node, node_weight in zip(target_nodes, target_weights):
            quality = float(predicted_mean + math.sqrt(2.0) * quality_scale * node)
            outcomes.append({"probability": float(component_weight * p_real * node_weight), "detected": True, "candidate_type": "candidate", "quality": quality, "confidence": float(_sigmoid((quality - 0.34) / 0.14))})
        p_clutter = (1.0 - p_real) * clutter_prior
        for node, node_weight in zip(clutter_nodes, clutter_weights):
            clutter_quality = float(0.12 + math.sqrt(2.0) * 0.08 * node)
            outcomes.append({"probability": float(component_weight * p_clutter * node_weight), "detected": True, "candidate_type": "candidate", "quality": clutter_quality, "confidence": float(_sigmoid((clutter_quality - 0.34) / 0.14))})
        p_missing = (1.0 - p_real) * (1.0 - clutter_prior)
        outcomes.append({"probability": float(component_weight * p_missing), "detected": False, "candidate_type": "missing", "quality": float("nan"), "confidence": 0.0})
    total = sum(float(outcome["probability"]) for outcome in outcomes)
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("non-finite public outcome mass")
    outcomes[-1]["probability"] += 1.0 - total
    return tuple(outcomes)


def public_likelihood(observation: PublicObservation) -> np.ndarray:
    return np.array([
        float(observation.detected),
        float(observation.confidence),
        float(observation.quality) if np.isfinite(observation.quality) else 0.0,
        float(observation.center_offset_m) if np.isfinite(observation.center_offset_m) else 0.0,
        float(observation.axis_angle_rad) if np.isfinite(observation.axis_angle_rad) else 0.0,
    ], dtype=float)


def stable_subset_smoke() -> dict[str, object]:
    action = fixture_actions()["probe_left"]
    memory_once = GaussianAddressMemory()
    memory_twice = GaussianAddressMemory()
    H = response_design(action)
    assert memory_once.update("e0", H, 0.2)
    assert not memory_once.update("e0", H, 0.2)
    assert memory_twice.update("e0", H, 0.2)
    assert np.allclose(memory_once.precision, memory_twice.precision)
    assert np.allclose(memory_once.information, memory_twice.information)
    mixture = MixtureAddressMemory()
    observation = render_observation(make_scenario(17), action, event_id="mixture-e0", elapsed_s=0.72)
    first = condition(mixture, action, observation)
    second = condition(mixture, action, observation)
    assert first in (True, False) and second is False
    return {"duplicate_rejected": True, "posterior": memory_once.snapshot(), "mixture": mixture.snapshot()}


def scenario_public_metadata(scenario: PrivateScenario) -> dict[str, object]:
    return {"seed": scenario.seed, "address": scenario.address, "target_id_hidden": True, "response_hidden": True, "windows": [s.name for s in constrained_windows()]}


__all__ = [
    "CapturedObservation", "GaussianAddressMemory", "MixtureAddressMemory", "HiddenFrame", "PrivateScenario", "ProbeAction", "PublicObservation", "TerminalTruth", "WindowSpec",
    "constrained_windows", "fixture_actions", "make_scenario", "paired_scenarios", "geometry_value", "geometry_support_probability", "detection_probability", "terminal_geometry_probability", "sensor_sd", "sensor_variance", "response_design", "render_capture", "render_observation", "terminal_truth", "terminal_success", "scan_probability", "predict_outcomes", "condition", "public_likelihood", "stable_subset_smoke", "scenario_public_metadata",
]
