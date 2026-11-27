"""Action-conditioned chart-space ultrasound world for recovery data.

This is a compact benchmark world, not a B-mode renderer.  It uses the
registered artery/vein section table from the streaming ProDMP experiment,
keeps two smooth structural explanations, and returns anonymous centroids.
Ground-truth hypothesis, roles, clutter, and knee sensitivity live on
``Scene`` only for evaluation.  ``alias=True`` is an explicit negative control
in which both public explanations have identical likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr


HERE = Path(__file__).resolve().parent
SECTIONS = HERE.parent / "ab_streaming_prodmp_two_vessel_20261113" / "SECTIONS.json"

TARGET = 0
DISTRACTOR = 1
CLUTTER = 2
STRUCTURAL_HYPOTHESES = 2
LATERAL_OFFSETS = np.array([-0.8, 0.0, 0.8], dtype=float)
DEPTH_OFFSETS = np.array([-0.5, 0.0, 0.5], dtype=float)
SLOPE_MODES = np.array([-1.0, 0.0, 1.0], dtype=float)
SLOPE_VECTOR = np.array([0.055, 0.035], dtype=float)
HYPOTHESES = STRUCTURAL_HYPOTHESES * len(LATERAL_OFFSETS) * len(DEPTH_OFFSETS) * len(SLOPE_MODES)
N_SOURCES = 3
REFERENCE_KNEE_DEG = 15.0
FOV_MIN = np.array([-25.0, 3.0], dtype=float)
FOV_MAX = np.array([25.0, 52.0], dtype=float)
NOISE_SD = np.array([0.72, 0.52], dtype=float)
CLUTTER_NOISE_SD = np.array([2.4, 1.5], dtype=float)
CLUTTER_DENSITY = 1.0 / float(np.prod(FOV_MAX - FOV_MIN))

# Public hypothesis indexing is fixed so the controller can reshape a belief
# as (connection, lateral offset, depth offset, slope mode).
_LATENT_COMPONENTS = np.asarray(
    [(h, li, di, si)
     for h in range(STRUCTURAL_HYPOTHESES)
     for li in range(len(LATERAL_OFFSETS))
     for di in range(len(DEPTH_OFFSETS))
     for si in range(len(SLOPE_MODES))],
    dtype=int,
)
_LATENT_H = _LATENT_COMPONENTS[:, 0]
_LATENT_LI = _LATENT_COMPONENTS[:, 1]
_LATENT_DI = _LATENT_COMPONENTS[:, 2]
_LATENT_SI = _LATENT_COMPONENTS[:, 3]
_LATENT_OFFSETS = np.column_stack((LATERAL_OFFSETS[_LATENT_LI], DEPTH_OFFSETS[_LATENT_DI]))
_PACKET_ASSIGNMENTS = {
    n: tuple(permutations(range(N_SOURCES), n)) for n in range(N_SOURCES + 1)
}


def _readonly(value: Any, dtype=float) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _normal_pdf(delta: np.ndarray, scale: np.ndarray = NOISE_SD) -> np.ndarray:
    z = delta / scale
    return np.exp(-0.5 * np.sum(z * z, axis=-1)) / (2.0 * np.pi * scale[0] * scale[1])


def _smoothstep(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


@dataclass(frozen=True)
class Scene:
    """Private realization used by the evaluator and packet generator."""

    seed: int
    split: str
    hypothesis: int
    alias: bool
    moving: bool
    angles_deg: np.ndarray
    offset: np.ndarray
    bias: np.ndarray
    knee_slope: np.ndarray
    shadow_s: float
    shadow_width: float
    clutter_shift: np.ndarray
    clear: np.ndarray
    noise: np.ndarray
    clutter_noise: np.ndarray
    family: str

    @property
    def slope_mm_per_deg(self) -> np.ndarray:
        """Evaluator-only sampled sensitivity of the hidden material response."""

        return self.knee_slope


class World:
    """Public probabilistic chart world shared by teacher and learned B."""

    def __init__(self, sections_path: str | Path = SECTIONS):
        with Path(sections_path).open("r", encoding="utf-8") as stream:
            sections = json.load(stream)
        self.s_mm = _readonly(sections["s_mm"])
        centers = np.asarray(sections["centers_mm"], dtype=float)
        if centers.shape != (len(self.s_mm), 2, 2):
            raise ValueError("registered section table must contain two 2-D vessel centers")
        self.centers_mm = _readonly(centers)
        self.station_bounds = (float(self.s_mm[0]), float(self.s_mm[-1]))
        self.transition_s = float(np.mean(self.station_bounds))
        structure_prior = np.full(STRUCTURAL_HYPOTHESES, 0.5)
        lateral_prior = np.array([0.24, 0.52, 0.24], dtype=float)
        depth_prior = np.array([0.22, 0.56, 0.22], dtype=float)
        slope_prior = np.array([0.25, 0.5, 0.25], dtype=float)
        weights = [structure_prior[h] * lateral_prior[li] * depth_prior[di] * slope_prior[si]
                   for h in range(STRUCTURAL_HYPOTHESES)
                   for li in range(3) for di in range(3) for si in range(3)]
        self.prior = _readonly(np.asarray(weights) / np.sum(weights))
        self.public_slope_modes = _readonly(np.array([-0.08, 0.0, 0.08], dtype=float))
        self._latent_h = _LATENT_H
        self._latent_slope = SLOPE_MODES[_LATENT_SI]
        self._latent_offsets = _LATENT_OFFSETS

    @property
    def n_hypotheses(self) -> int:
        return HYPOTHESES

    @property
    def n_structural_hypotheses(self) -> int:
        return STRUCTURAL_HYPOTHESES

    @staticmethod
    def latent_components(index: int) -> tuple[int, int, int, int]:
        """Return ``(connection, lateral-index, depth-index, slope-index)``."""

        index = int(index)
        if index < 0 or index >= HYPOTHESES:
            raise IndexError("latent hypothesis index outside public support")
        h, remainder = divmod(index, 27)
        li, remainder = divmod(remainder, 9)
        di, si = divmod(remainder, 3)
        return h, li, di, si

    @staticmethod
    def latent_index(h: int, lateral_index: int, depth_index: int, slope_index: int) -> int:
        return (((int(h) * 3 + int(lateral_index)) * 3 + int(depth_index)) * 3 + int(slope_index))

    def _interp_centers(self, s: float | np.ndarray) -> np.ndarray:
        query = np.asarray(s, dtype=float)
        if np.any(~np.isfinite(query)) or np.any(query < self.station_bounds[0]) or np.any(query > self.station_bounds[1]):
            raise ValueError(f"s must lie in [{self.station_bounds[0]}, {self.station_bounds[1]}]")
        flat = query.reshape(-1)
        result = np.empty((len(flat), 2, 2), dtype=float)
        for role in range(2):
            for coordinate in range(2):
                result[:, role, coordinate] = np.interp(flat, self.s_mm, self.centers_mm[:, role, coordinate])
        if query.ndim == 0:
            return result[0]
        return result.reshape(query.shape + (2, 2))

    def nominal(self, s: float | np.ndarray) -> np.ndarray:
        """Nominal artery center in absolute chart coordinates."""

        return self._interp_centers(s)[..., TARGET, :]

    def _envelope(self, s: float | np.ndarray) -> np.ndarray:
        width = max(5.0, 0.26 * (self.station_bounds[1] - self.station_bounds[0]))
        return 1.0 - _smoothstep(np.abs(np.asarray(s, dtype=float) - self.transition_s) / width)

    def _tracks(self, s: float, angle_deg: float, hypothesis: int, *, alias: bool = False,
                slope: np.ndarray | None = None, family: str = "nominal") -> tuple[np.ndarray, np.ndarray]:
        base = self._interp_centers(float(s))
        artery, vein = base[TARGET].copy(), base[DISTRACTOR].copy()
        separation = vein - artery
        span = self.station_bounds[1] - self.station_bounds[0]
        start = self.transition_s - 0.16 * span
        width = 0.32 * span
        if family == "heldout":
            start, width = self.transition_s + 1.0 - 0.135 * span, 0.27 * span
        phase = float(_smoothstep((float(s) - start) / width))
        # The first structural explanation follows the nominal pair.  The
        # second rotates the separation vector by pi through the junction.
        # Rotation preserves a nonzero pair separation and ends with the same
        # unordered pair as h0, while the path through the junction differs.
        theta = 0.035 * phase * np.sin(np.pi * phase)
        if int(hypothesis) == 1 and not alias:
            theta += np.pi * phase
        c, s_theta = np.cos(theta), np.sin(theta)
        rotated = np.array([[c * separation[0] - s_theta * separation[1],
                             s_theta * separation[0] + c * separation[1]]])
        plus, minus = 0.5 * rotated[0], -0.5 * rotated[0]
        midpoint = 0.5 * (artery + vein)
        track_minus, track_plus = midpoint + minus, midpoint + plus
        if slope is not None:
            delta_q = float(angle_deg) - REFERENCE_KNEE_DEG
            delta = self._envelope(float(s)) * delta_q * np.asarray(slope, dtype=float)
            track_minus, track_plus = track_minus + delta, track_plus + delta
        return track_minus, track_plus

    def _geometry(self, s: float, angle_deg: float, hypothesis: int, *, alias: bool = False,
                  slope: np.ndarray | None = None, family: str = "nominal") -> tuple[np.ndarray, np.ndarray]:
        track_minus, track_plus = self._tracks(s, angle_deg, hypothesis, alias=alias,
                                               slope=slope, family=family)
        if alias and int(hypothesis) == 1:
            return track_plus, track_minus
        return track_minus, track_plus

    def target_grid(self, s: float, angle: float, *, alias: bool = False) -> np.ndarray:
        """Absolute target position predicted by each public connection model."""
        # Only the two structural hypotheses require geometry evaluation.  The
        # remaining public dimensions are broadcast over those two base pairs.
        base = np.asarray([self._geometry(float(s), float(angle), h, alias=alias)[0]
                           for h in range(STRUCTURAL_HYPOTHESES)], dtype=float)
        response = (self._envelope(float(s)) * (float(angle) - REFERENCE_KNEE_DEG)
                    * self._latent_slope[:, None] * SLOPE_VECTOR[None, :])
        return base[self._latent_h] + self._latent_offsets + response

    def _distractor_grid(self, s: float, angle: float, *, alias: bool = False) -> np.ndarray:
        base = np.asarray([self._geometry(float(s), float(angle), h, alias=alias)[1]
                           for h in range(STRUCTURAL_HYPOTHESES)], dtype=float)
        response = (self._envelope(float(s)) * (float(angle) - REFERENCE_KNEE_DEG)
                    * self._latent_slope[:, None] * SLOPE_VECTOR[None, :])
        return base[self._latent_h] + self._latent_offsets + response

    def _view_visibility(self, means_abs: np.ndarray, s: float, x: float) -> np.ndarray:
        rel = means_abs - np.array([float(x), 0.0])
        inside = np.all((rel >= FOV_MIN) & (rel <= FOV_MAX), axis=-1)
        shadow = np.exp(-0.5 * ((float(s) - self.transition_s) / 2.0) ** 2)
        side = np.exp(-0.5 * ((rel[..., 0] - 4.0) / 3.0) ** 2)
        return inside.astype(float) * np.clip(0.97 - 0.54 * shadow * side, 0.03, 0.97)

    def predict(self, s: float, x: float, angle: float, *, alias: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return public source means and visibility probabilities.

        Means have shape ``(hypotheses, sources, 2)`` and are probe-relative.
        The third source is an anonymous low-probability clutter component.
        """

        targets = self.target_grid(float(s), float(angle), alias=alias)
        distractors = self._distractor_grid(float(s), float(angle), alias=alias)
        midpoint = 0.5 * (targets + distractors)
        means_abs = np.stack((targets, distractors, midpoint + np.array([4.0, 1.0])), axis=1)
        means = means_abs - np.array([float(x), 0.0])
        visibility = self._view_visibility(means_abs.reshape(-1, 2), float(s), float(x)).reshape(HYPOTHESES, N_SOURCES)
        visibility[:, CLUTTER] *= 0.16
        return means, visibility

    def _hidden_geometry(self, scene: Scene, s: float, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        angle = float(scene.angles_deg[frame])
        target, distractor = self._geometry(s, angle, scene.hypothesis, alias=scene.alias,
                                            slope=scene.knee_slope, family=scene.family)
        target = target + scene.offset
        distractor = distractor + scene.offset
        clutter = 0.5 * (target + distractor) + scene.clutter_shift
        if scene.alias and scene.hypothesis == 1:
            role_to_track = np.array([1, 0, 2], dtype=int)
        else:
            role_to_track = np.array([0, 1, 2], dtype=int)
        return np.stack((target, distractor, clutter)), np.array([TARGET, DISTRACTOR, CLUTTER], dtype=int), role_to_track

    def scene(self, seed: int, split: str = "train", n_frames: int = 1,
              moving: bool = False, alias: bool = False) -> Scene:
        """Create a reproducible hidden case.

        The held-out split changes transition location and coupling amplitude.
        The public model never reads case-specific fields.  ``alias`` activates
        the non-identifiable control in which public action likelihoods match.
        """

        if not isinstance(seed, (int, np.integer)) or n_frames < 0:
            raise ValueError("seed must be integer and n_frames nonnegative")
        split = str(split).lower()
        if split not in {"train", "dev", "heldout", "stress"}:
            raise ValueError("split must be train, dev, heldout, or stress")
        rng = np.random.default_rng(int(seed))
        n_frames = int(n_frames)
        phase_offset = rng.random()
        if moving:
            t = np.linspace(0.0, 1.0, max(n_frames, 1))
            amp = 4.0 if split in {"train", "dev"} else 6.0
            angles = REFERENCE_KNEE_DEG + amp * np.sin(2.0 * np.pi * (t + 0.07 * phase_offset))
        else:
            angles = np.full(max(n_frames, 1), REFERENCE_KNEE_DEG, dtype=float)
        slope_scale = 0.055 if split != "heldout" else 0.085
        knee_slope = rng.normal(0.0, slope_scale, size=2)
        if split == "heldout":
            offset = rng.choice(np.array([-0.4, 0.4]), size=2) + rng.normal(0.0, [0.08, 0.06])
        else:
            offset = np.array([rng.choice(LATERAL_OFFSETS), rng.choice(DEPTH_OFFSETS)])
            offset += rng.normal(0.0, [0.08, 0.06])
        bias = rng.normal([0.0, 0.0], [0.22, 0.16])
        shadow_s = float(self.transition_s + rng.normal(0.0, 1.6 if split != "heldout" else 2.2))
        shadow_width = float(rng.uniform(1.2, 2.6))
        clutter_shift = rng.normal([5.0, 0.5], [1.2, 0.7])
        clear = np.empty((max(n_frames, 1), 3), dtype=bool)
        clear[0] = rng.random(3) < np.array([0.84, 0.82, 0.58])
        for frame in range(1, len(clear)):
            persistence = np.array([0.87, 0.84, 0.72])
            clear[frame] = np.where(clear[frame - 1], rng.random(3) < persistence,
                                    rng.random(3) < np.array([0.22, 0.24, 0.35]))
        noise = rng.normal(0.0, NOISE_SD, size=(len(clear), 3, 2))
        clutter_noise = rng.normal(0.0, [1.0, 0.8], size=(len(clear), 2))
        family = "heldout" if split == "heldout" else ("stress" if split == "stress" else "nominal")
        return Scene(int(seed), split, int(rng.integers(STRUCTURAL_HYPOTHESES)), bool(alias), bool(moving),
                     _readonly(angles), _readonly(offset), _readonly(bias), _readonly(knee_slope),
                     float(shadow_s), float(shadow_width), _readonly(clutter_shift),
                     _readonly(clear, bool), _readonly(noise), _readonly(clutter_noise), family)

    def observe(self, scene: Scene, frame: int, s: float, x: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return anonymous packet, evaluator roles, and absolute true target."""

        if not isinstance(scene, Scene) or frame < 0 or frame >= len(scene.clear):
            raise ValueError("frame must index a Scene returned by scene()")
        means, role_ids, role_to_track = self._hidden_geometry(scene, float(s), int(frame))
        rel = means - np.array([float(x), 0.0]) + scene.bias
        visible = scene.clear[frame, role_to_track].copy()
        shadow = np.exp(-0.5 * ((float(s) - scene.shadow_s) / scene.shadow_width) ** 2)
        shadow *= np.exp(-0.5 * ((rel[:, 0] - 4.0) / 3.0) ** 2)
        visible &= shadow < 0.82
        visible &= np.all((rel >= FOV_MIN) & (rel <= FOV_MAX), axis=1)
        noise = scene.noise[frame, role_to_track]
        noise[CLUTTER] = scene.clutter_noise[frame]
        packet = rel[visible] + noise[visible]
        roles = role_ids[visible]
        # A noisy centroid that leaves the public window is not returned as a
        # packet.  This keeps the evaluator and the public likelihood on the
        # same bounded observation domain.
        if len(packet):
            in_fov = np.all((packet >= FOV_MIN) & (packet <= FOV_MAX), axis=1)
            packet, roles = packet[in_fov], roles[in_fov]
        if len(packet):
            order = np.argsort(packet[:, 0], kind="stable")
            packet, roles = packet[order], roles[order]
        else:
            packet = np.empty((0, 2), dtype=float)
            roles = np.empty((0,), dtype=int)
        return np.asarray(packet, dtype=float), np.asarray(roles, dtype=int), means[TARGET].copy()

    def true_identity(self, scene: Scene, frame: int = 0, station: float | None = None) -> bool:
        """Evaluator-only event matching ``identity_prob``.

        The event is that the requested target lies on the right side of the
        other visible track in the local chart lateral coordinate.
        """

        if not isinstance(scene, Scene) or frame < 0 or frame >= len(scene.angles_deg):
            raise ValueError("frame must index a Scene returned by scene()")
        if station is None:
            station = float(self.transition_s)
        target, distractor = self._geometry(float(station), float(scene.angles_deg[frame]), scene.hypothesis,
                                            alias=scene.alias, slope=scene.knee_slope, family=scene.family)
        return bool(target[0] > distractor[0])

    @staticmethod
    def replace_angles(scene: Scene, angles_deg: np.ndarray) -> Scene:
        """Return a paired schedule with every hidden tape held fixed."""

        angles = np.asarray(angles_deg, dtype=float)
        if angles.shape != scene.angles_deg.shape or np.any(~np.isfinite(angles)):
            raise ValueError("replacement angle schedule must match the scene tape")
        return replace(scene, angles_deg=_readonly(angles))

    def resample_tail(self, scene: Scene, from_frame: int, replica: int = 0) -> Scene:
        """Resample only stochastic tapes after a public decision checkpoint."""

        if not isinstance(scene, Scene) or from_frame < 0 or from_frame >= len(scene.clear):
            raise ValueError("from_frame must be an existing scene frame")
        if not isinstance(replica, (int, np.integer)):
            raise ValueError("replica must be an integer")
        rng = np.random.default_rng([int(scene.seed), 0xA53E, int(replica)])
        clear = np.asarray(scene.clear, dtype=bool).copy()
        noise = np.asarray(scene.noise, dtype=float).copy()
        clutter_noise = np.asarray(scene.clutter_noise, dtype=float).copy()
        for frame in range(int(from_frame) + 1, len(clear)):
            persistence = np.array([0.87, 0.84, 0.72])
            clear[frame] = np.where(clear[frame - 1], rng.random(3) < persistence,
                                    rng.random(3) < np.array([0.22, 0.24, 0.35]))
            noise[frame] = rng.normal(0.0, NOISE_SD, size=(3, 2))
            clutter_noise[frame] = rng.normal(0.0, [1.0, 0.8], size=2)
        return replace(scene, clear=_readonly(clear, bool), noise=_readonly(noise),
                       clutter_noise=_readonly(clutter_noise))

    def _likelihood_batch(self, packet: np.ndarray, means: np.ndarray,
                          visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate anonymous packets with an injective source assignment sum.

        Each of the three public sources emits at most one centroid.  A source
        is detected with probability ``visibility`` before the noisy centroid
        is clipped to the image. An unassigned source may either miss or
        produce a centroid outside the field of view. The observed packet is
        anonymous, so all injective packet-to-source assignments are summed.
        """

        packet = np.asarray(packet, dtype=float)
        means = np.asarray(means, dtype=float)
        visibility = np.asarray(visibility, dtype=float)
        if packet.ndim != 2 or packet.shape[1:] != (2,) or np.any(~np.isfinite(packet)):
            raise ValueError("packet must have shape (N,2) and finite coordinates")
        if means.ndim != 3 or means.shape[1:] != (N_SOURCES, 2):
            raise ValueError("means must have shape (H,3,2)")
        if visibility.shape != means.shape[:2]:
            raise ValueError("visibility must have shape (H,3)")
        if np.any(packet < FOV_MIN) or np.any(packet > FOV_MAX):
            raise ValueError("packet lies outside public FOV")
        if len(packet) > N_SOURCES:
            raise ValueError("anonymous packet contains more sources than the public model")

        detect = np.clip(visibility, 0.0, 1.0)
        n_hyp = means.shape[0]
        scales = np.stack((NOISE_SD, NOISE_SD, CLUTTER_NOISE_SD))
        mass = np.prod(ndtr((FOV_MAX - means) / scales) - ndtr((FOV_MIN - means) / scales), axis=-1)
        mass[:, CLUTTER] = .72 * mass[:, CLUTTER] + .28
        miss = 1.0 - detect * mass
        if len(packet) == 0:
            return np.prod(miss, axis=1), np.empty((n_hyp, 0), dtype=float)

        # densities[h, packet_index, source].  Clutter uses a broad Gaussian
        # plus a bounded uniform component so an outlying clutter point does
        # not make a hypothesis exactly impossible.
        delta = packet[None, :, None, :] - means[:, None, :, :]
        densities = np.empty((n_hyp, len(packet), N_SOURCES), dtype=float)
        densities[..., TARGET] = _normal_pdf(delta[..., TARGET, :], NOISE_SD)
        densities[..., DISTRACTOR] = _normal_pdf(delta[..., DISTRACTOR, :], NOISE_SD)
        clutter_pdf = _normal_pdf(delta[..., CLUTTER, :], CLUTTER_NOISE_SD)
        densities[..., CLUTTER] = 0.72 * clutter_pdf + 0.28 * CLUTTER_DENSITY

        likelihood = np.zeros(n_hyp, dtype=float)
        target_mass = np.zeros((n_hyp, len(packet)), dtype=float)
        for assignment in _PACKET_ASSIGNMENTS[len(packet)]:
            assigned = np.asarray(assignment, dtype=int)
            used = np.zeros(N_SOURCES, dtype=bool)
            used[assigned] = True
            p_emit = np.prod(detect[:, assigned], axis=1)
            p_miss = np.prod(miss[:, ~used], axis=1)
            point_density = np.prod(densities[:, np.arange(len(packet)), assigned], axis=1)
            weight = p_emit * p_miss * point_density
            likelihood += weight
            for packet_index, source in enumerate(assignment):
                if source == TARGET:
                    target_mass[:, packet_index] += weight
        likelihood = np.maximum(likelihood, 1.0e-300)
        return likelihood, np.clip(target_mass / likelihood[:, None], 0.0, 1.0)

    def _likelihood(self, packet: np.ndarray, means: np.ndarray,
                    visibility: np.ndarray) -> tuple[float, np.ndarray]:
        """Scalar compatibility wrapper around :meth:`_likelihood_batch`."""

        likelihood, responsibility = self._likelihood_batch(
            packet, np.asarray(means, dtype=float)[None, ...],
            np.asarray(visibility, dtype=float)[None, ...])
        return float(likelihood[0]), responsibility[0]

    def update(self, belief: np.ndarray, packet: np.ndarray, s: float, x: float, angle: float,
               *, alias: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Bayesian update with anonymous target, distractor and clutter."""

        belief = np.asarray(belief, dtype=float)
        if belief.shape != (HYPOTHESES,) or np.any(~np.isfinite(belief)) or np.any(belief < 0.0) or belief.sum() <= 0.0:
            raise ValueError(f"belief must be a positive finite vector of shape ({HYPOTHESES},)")
        belief = belief / belief.sum()
        means, visibility = self.predict(float(s), float(x), float(angle), alias=alias)
        likelihood, responsibilities = self._likelihood_batch(packet, means, visibility)
        weighted = belief * likelihood
        mass = float(weighted.sum())
        if not np.isfinite(mass) or mass <= 0.0:
            return self.prior.copy(), np.zeros(len(packet), dtype=float)
        posterior = weighted / mass
        association = np.sum(posterior[:, None] * responsibilities, axis=0) if len(packet) else np.empty(0)
        return posterior, association

    def sample(self, belief: np.ndarray, s: float, x: float, angle: float,
               rng: np.random.Generator, *, alias: bool = False) -> np.ndarray:
        """Draw one hypothetical public packet using the supplied belief."""

        belief = np.asarray(belief, dtype=float)
        belief = belief / belief.sum()
        h = int(rng.choice(HYPOTHESES, p=belief))
        means, visibility = self.predict(float(s), float(x), float(angle), alias=alias)
        packet = []
        for source in (TARGET, DISTRACTOR, CLUTTER):
            probability = float(np.clip(visibility[h, source], 0.0, 1.0))
            if rng.random() < probability:
                if source == CLUTTER:
                    if rng.random() < 0.72:
                        location = means[h, source] + rng.normal(0.0, CLUTTER_NOISE_SD)
                    else:
                        location = rng.uniform(FOV_MIN, FOV_MAX)
                else:
                    location = means[h, source] + rng.normal(0.0, NOISE_SD)
                if np.all((location >= FOV_MIN) & (location <= FOV_MAX)):
                    packet.append(location)
        if not packet:
            return np.empty((0, 2), dtype=float)
        return np.asarray(sorted(packet, key=lambda value: value[0]), dtype=float)

    def identity_prob(self, belief: np.ndarray, station: float, angle: float, *, alias: bool = False) -> float:
        """Posterior probability that target is right of the other track."""

        belief = np.asarray(belief, dtype=float)
        if belief.shape != (HYPOTHESES,) or belief.sum() <= 0.0:
            raise ValueError(f"belief must have {HYPOTHESES} positive hypotheses")
        target = self.target_grid(float(station), float(angle), alias=alias)
        distractor = self._distractor_grid(float(station), float(angle), alias=alias)
        event = target[:, 0] > distractor[:, 0]
        return float(np.sum((belief / belief.sum()) * event))


__all__ = ["Scene", "World", "TARGET", "DISTRACTOR", "CLUTTER", "HYPOTHESES", "N_SOURCES"]
