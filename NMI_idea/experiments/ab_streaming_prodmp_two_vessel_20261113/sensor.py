"""Streaming two-vessel sensor and exact latent-state Bayesian filter.

The public section table supplies the nominal artery/vein centroids along the
streaming path.  A case has one effective discrete probe offset and a small
continuous calibration bias.  The image mean uses the effective offset while
the physical anatomy uses ``effective - bias``.  The filter deliberately keeps
the two-role clear/dropout process in its state instead of making consecutive
frame marginals independent.

Hidden state rows are ``(effective_offset_index, clear_state_index)`` and
``STATE_INDEX`` documents the clear-state order:

    ``[(0, 0), (0, 1), (1, 0), (1, 1)]``

where each pair is ``(artery_state, vein_state)`` and each role state is
``0 = dropped, 1 = clear``.  ``prior()`` is an ``(H, 4)`` array with one row
per effective offset hypothesis.  One Markov transition is applied before
every observation update; because the returned prior starts in stationarity,
the first transition leaves its clear-state marginal unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np
from scipy.special import ndtr


OFFSET_LATERAL_MM = np.arange(-6.0, 6.0 + 1.0, 2.0)
OFFSET_DEPTH_MM = np.arange(-2.0, 2.0 + 1.0, 2.0)
OFFSET_SD_MM = np.array([3.5, 1.5], dtype=float)
BIAS_SD_MM = np.array([0.55, 0.30], dtype=float)
NOISE_SD_MM = np.array([1.0, 0.68], dtype=float)
FOV_MIN_MM = np.array([-25.0, 3.0], dtype=float)
FOV_MAX_MM = np.array([25.0, 52.0], dtype=float)

# Roles are always in physical/public order, independent of packet order.
ARTERY = 0
VEIN = 1
STATE_INDEX = ((0, 0), (0, 1), (1, 0), (1, 1))
P_CLEAR = np.array([[0.72, 0.28], [0.06, 0.94]], dtype=float)
STATIONARY_ROLE = np.array([0.06 / (0.28 + 0.06), 0.28 / (0.28 + 0.06)], dtype=float)


@dataclass(frozen=True)
class Scene:
    """One frozen sensor case and its frame-wise latent dropouts/noise."""

    offset: np.ndarray
    bias: np.ndarray
    clear: np.ndarray
    noise: np.ndarray


def _freeze(array: np.ndarray) -> np.ndarray:
    """Make a scene array immutable while retaining NumPy's public API."""

    array = np.asarray(array)
    array.setflags(write=False)
    return array


def _normal_cell_prior(values: np.ndarray, sd: float) -> np.ndarray:
    """Integrate a normal over each grid point's Voronoi cell."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(np.diff(values) <= 0):
        raise ValueError("normal-prior support must be strictly increasing")
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError("normal-prior scale must be positive")
    if len(values) == 1:
        edges = np.array([-np.inf, np.inf])
    else:
        edges = np.r_[-np.inf, (values[:-1] + values[1:]) / 2.0, np.inf]
    masses = np.diff(ndtr(edges / sd))
    return masses / masses.sum()


def _as_scalar(value: Any, name: str) -> float:
    value = np.asarray(value, dtype=float)
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be one finite scalar millimetre value")
    return float(value)


class Sensor:
    """Exact static-world packet simulator and streaming Bayesian filter."""

    def __init__(self, sections: dict):
        if not isinstance(sections, dict):
            raise TypeError("sections must be the decoded SECTIONS.json dictionary")
        try:
            s_mm = np.asarray(sections["s_mm"], dtype=float)
            centers_mm = np.asarray(sections["centers_mm"], dtype=float)
        except KeyError as exc:
            raise ValueError(f"sections is missing {exc.args[0]!r}") from exc
        if s_mm.ndim != 1 or len(s_mm) < 2 or np.any(~np.isfinite(s_mm)) or np.any(np.diff(s_mm) <= 0.0):
            raise ValueError("sections s_mm must be finite and strictly increasing")
        if centers_mm.shape != (len(s_mm), 2, 2) or np.any(~np.isfinite(centers_mm)):
            raise ValueError("sections centers_mm must have shape (len(s_mm), 2, 2)")

        self._s_mm = np.array(s_mm, dtype=float, copy=True)
        self._centers_mm = np.array(centers_mm, dtype=float, copy=True)
        self._offsets_mm = np.asarray(
            [(lateral, depth) for lateral in OFFSET_LATERAL_MM for depth in OFFSET_DEPTH_MM],
            dtype=float,
        )
        lateral_prior = _normal_cell_prior(OFFSET_LATERAL_MM, OFFSET_SD_MM[0])
        depth_prior = _normal_cell_prior(OFFSET_DEPTH_MM, OFFSET_SD_MM[1])
        self._offset_prior = np.asarray(
            [lateral_prior[i] * depth_prior[j]
             for i in range(len(OFFSET_LATERAL_MM))
             for j in range(len(OFFSET_DEPTH_MM))],
            dtype=float,
        )
        self._state_transition = np.asarray(
            [
                [P_CLEAR[a0, b0] * P_CLEAR[a1, b1] for b0, b1 in STATE_INDEX]
                for a0, a1 in STATE_INDEX
            ],
            dtype=float,
        )
        self._noise_sd = np.array(NOISE_SD_MM, dtype=float, copy=True)
        self._stationary_state = np.asarray(
            [STATIONARY_ROLE[a0] * STATIONARY_ROLE[a1] for a0, a1 in STATE_INDEX],
            dtype=float,
        )

        # Keep user-visible constants useful for audit scripts without making
        # internal arrays accidentally mutable through a returned view.
        self.offsets_mm = self._offsets_mm.copy()
        self.offset_prior = self._offset_prior.copy()
        self.state_index = tuple(STATE_INDEX)
        self.offsets_mm.setflags(write=False)
        self.offset_prior.setflags(write=False)

    @property
    def n_hypotheses(self) -> int:
        """Number of effective discrete offset hypotheses (21)."""

        return len(self._offsets_mm)

    def nominal(self, s_mm) -> np.ndarray:
        """Linearly interpolate public nominal centers at path coordinate(s)."""

        query = np.asarray(s_mm, dtype=float)
        if np.any(~np.isfinite(query)) or np.any(query < self._s_mm[0]) or np.any(query > self._s_mm[-1]):
            raise ValueError(f"s_mm must lie in [{self._s_mm[0]}, {self._s_mm[-1]}]")
        flat = query.reshape(-1)
        values = np.empty((len(flat), 2, 2), dtype=float)
        for role in range(2):
            for coordinate in range(2):
                values[:, role, coordinate] = np.interp(
                    flat, self._s_mm, self._centers_mm[:, role, coordinate]
                )
        if query.ndim == 0:
            return values[0]
        return values.reshape(query.shape + (2, 2))

    def scene(self, rng: np.random.Generator, n_frames: int) -> Scene:
        """Draw one effective offset, calibration bias, dropout chain, and noise tape."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if not isinstance(n_frames, (int, np.integer)) or n_frames < 0:
            raise ValueError("n_frames must be a nonnegative integer")
        n_frames = int(n_frames)

        offset_index = int(rng.choice(self.n_hypotheses, p=self._offset_prior))
        offset = self._offsets_mm[offset_index].copy()
        bias = rng.normal(loc=0.0, scale=BIAS_SD_MM).astype(float)

        clear = np.empty((n_frames, 2), dtype=bool)
        if n_frames:
            clear[0] = rng.random(2) < STATIONARY_ROLE[1]
            for frame in range(1, n_frames):
                clear[frame] = rng.random(2) < np.where(clear[frame - 1], P_CLEAR[1, 1], P_CLEAR[0, 1])
        noise = rng.normal(loc=0.0, scale=self._noise_sd, size=(n_frames, 2, 2)).astype(float)
        return Scene(_freeze(offset), _freeze(bias), _freeze(clear), _freeze(noise))

    def observe(self, scene: Scene, k: int, s_mm, x_mm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return an anonymous FOV-clipped packet, evaluator roles, and physical lateral truth."""

        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene returned by Sensor.scene")
        if not isinstance(k, (int, np.integer)) or not 0 <= int(k) < len(scene.clear):
            raise IndexError("frame k is outside the scene tape")
        k = int(k)
        x_mm = _as_scalar(x_mm, "x_mm")
        if scene.offset.shape != (2,) or scene.bias.shape != (2,) or scene.noise.shape != (len(scene.clear), 2, 2):
            raise ValueError("scene has incompatible array shapes")

        center = self.nominal(s_mm)
        if center.shape != (2, 2):
            raise ValueError("observe requires one scalar s_mm")
        physical_shift = scene.offset - scene.bias
        true_u = center[:, 0] + physical_shift[0] - x_mm
        # The calibration bias cancels in the observed mean: anatomy moves by
        # effective-bias, while the sensor's case bias adds back bias.
        observed_mean = center + scene.offset - np.array([x_mm, 0.0])
        measured = observed_mean + scene.noise[k]
        visible = scene.clear[k] & np.all(measured >= FOV_MIN_MM, axis=1) & np.all(measured <= FOV_MAX_MM, axis=1)
        roles = np.flatnonzero(visible).astype(int)
        if len(roles):
            order = np.argsort(measured[roles, 0], kind="stable")
            roles = roles[order]
            packet = np.asarray(measured[roles], dtype=float)
        else:
            packet = np.empty((0, 2), dtype=float)
            roles = np.empty((0,), dtype=int)
        return packet, roles, np.asarray(true_u, dtype=float)

    def prior(self) -> np.ndarray:
        """Return the normalized joint prior over offset and clear-state rows."""

        return self._offset_prior[:, None] * self._stationary_state[None, :]

    @staticmethod
    def _normal_pdf(point: np.ndarray, mean: np.ndarray) -> float:
        delta = (np.asarray(point, dtype=float) - mean) / NOISE_SD_MM
        return float(np.exp(-0.5 * np.dot(delta, delta)) / (2.0 * pi * np.prod(NOISE_SD_MM)))

    @staticmethod
    def _rectangle_mass(mean: np.ndarray) -> float:
        mean = np.asarray(mean, dtype=float)
        masses = ndtr((FOV_MAX_MM - mean) / NOISE_SD_MM) - ndtr((FOV_MIN_MM - mean) / NOISE_SD_MM)
        return float(np.prod(masses))

    @staticmethod
    def _validate_packet(packet) -> np.ndarray:
        packet = np.asarray(packet, dtype=float)
        if packet.ndim != 2 or packet.shape[1:] != (2,) or packet.shape[0] > 2:
            raise ValueError("packet must have shape (N, 2) with N <= 2")
        if np.any(~np.isfinite(packet)):
            raise ValueError("packet coordinates must be finite")
        if np.any(packet < FOV_MIN_MM) or np.any(packet > FOV_MAX_MM):
            raise ValueError("packet coordinates must lie inside the sensor field of view")
        return packet

    @staticmethod
    def _normalize_belief(belief, shape: tuple[int, int]) -> np.ndarray:
        belief = np.asarray(belief, dtype=float)
        if belief.shape != shape or np.any(~np.isfinite(belief)) or np.any(belief < 0.0):
            raise ValueError(f"belief must be a nonnegative finite array with shape {shape}")
        total = float(belief.sum())
        if total <= 0.0:
            raise ValueError("belief must have positive mass")
        return belief / total

    def _emission(self, packet: np.ndarray, s_mm, x_mm) -> tuple[np.ndarray, np.ndarray]:
        """Return state-wise packet likelihood and artery-slot association mass."""

        packet = self._validate_packet(packet)
        x_mm = _as_scalar(x_mm, "x_mm")
        center = self.nominal(s_mm)
        if center.shape != (2, 2):
            raise ValueError("update requires one scalar s_mm")
        means = center + self._offsets_mm[:, None, :] - np.array([x_mm, 0.0])
        n = len(packet)
        likelihood = np.zeros((self.n_hypotheses, 4), dtype=float)
        association = np.zeros((self.n_hypotheses, 4, n), dtype=float)

        for h in range(self.n_hypotheses):
            present_prob = np.array([self._rectangle_mass(means[h, role]) for role in range(2)])
            present_density = np.empty((2, n), dtype=float)
            for role in range(2):
                for slot in range(n):
                    present_density[role, slot] = self._normal_pdf(packet[slot], means[h, role])
            for state, (artery_clear, vein_clear) in enumerate(STATE_INDEX):
                clear = (bool(artery_clear), bool(vein_clear))
                missing = np.array(
                    [1.0 if not clear[role] else 1.0 - present_prob[role] for role in range(2)],
                    dtype=float,
                )
                if n == 0:
                    likelihood[h, state] = missing[0] * missing[1]
                elif n == 1:
                    artery_assignment = present_density[ARTERY, 0] * missing[VEIN] if clear[ARTERY] else 0.0
                    vein_assignment = missing[ARTERY] * present_density[VEIN, 0] if clear[VEIN] else 0.0
                    likelihood[h, state] = artery_assignment + vein_assignment
                    association[h, state, 0] = artery_assignment
                else:  # n == 2: exact sum over the two role-to-slot permutations.
                    if clear[ARTERY] and clear[VEIN]:
                        artery_first = present_density[ARTERY, 0] * present_density[VEIN, 1]
                        artery_second = present_density[ARTERY, 1] * present_density[VEIN, 0]
                        likelihood[h, state] = artery_first + artery_second
                        association[h, state, 0] = artery_first
                        association[h, state, 1] = artery_second
        return likelihood, association

    def update(self, belief, packet, s_mm, x_mm) -> tuple[np.ndarray, np.ndarray]:
        """Apply one clear-state transition and the exact packet association update."""

        belief = self._normalize_belief(belief, (self.n_hypotheses, len(STATE_INDEX)))
        predicted = belief @ self._state_transition
        likelihood, association = self._emission(packet, s_mm, x_mm)
        joint = predicted * likelihood
        mass = float(joint.sum())
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("packet has zero probability under the supplied belief")
        new_belief = joint / mass
        arterial_probs = np.sum(predicted[:, :, None] * association, axis=(0, 1)) / mass
        return new_belief, arterial_probs

    def offset_mean(self, belief) -> np.ndarray:
        """Return the posterior mean effective offset in millimetres."""

        belief = self._normalize_belief(belief, (self.n_hypotheses, len(STATE_INDEX)))
        return np.sum(self._offsets_mm * belief.sum(axis=1)[:, None], axis=0)


__all__ = ["Scene", "Sensor", "STATE_INDEX"]
