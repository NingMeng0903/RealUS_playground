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
CENTER_TOL_MM = 3.0
IDENTITY_MIN = 0.95

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


class UnsupportedObservation(ValueError):
    """The observed packet has no support under the predictive belief."""


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

    def __init__(self, sections: dict | None = None):
        self._s_mm = self._centers_mm = None
        if sections is not None:
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
            self._s_mm = s_mm.copy()
            self._centers_mm = centers_mm.copy()
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
        self._sigma = np.sqrt(2.0 * pi) * self._noise_sd
        self._clear = np.asarray(STATE_INDEX, dtype=bool)
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

        if self._s_mm is None:
            raise ValueError("section-based queries require a nominal section table")
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
        """Return anonymous observations for the legacy section-table geometry."""
        lateral = np.array([_as_scalar(x_mm, "x_mm"), 0.0])
        center = self.nominal(s_mm)
        if center.shape != (2, 2):
            raise ValueError("observe requires one scalar s_mm")
        self._validate_scene_frame(scene, k)
        return self._packet(scene, k, center + scene.offset - lateral,
                            center + (scene.offset - scene.bias) - lateral)

    @staticmethod
    def _validate_scene_frame(scene, k):
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene returned by Sensor.scene")
        if not isinstance(k, (int, np.integer)) or not 0 <= int(k) < len(scene.clear):
            raise IndexError("frame k is outside the scene tape")
        if scene.offset.shape != (2,) or scene.bias.shape != (2,) or scene.noise.shape != (len(scene.clear), 2, 2):
            raise ValueError("scene has incompatible array shapes")

    def observe_means(self, scene, k, observed_mean, physical_mean):
        """Add calibrated noise and persistent dropouts to projected means."""
        self._validate_scene_frame(scene, k)
        return self._packet(scene, k, observed_mean, physical_mean)

    @staticmethod
    def _packet(scene, k, observed_mean, physical_mean):
        true_u = physical_mean[:, 0]
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
        return self._emission_means(packet, means)

    def _emission_means(self, packet, means):
        """Marginalize anonymous slot assignments at projected hypothesis means."""
        n = len(packet)
        visible = np.isfinite(means).all(axis=-1)
        means = np.where(visible[..., None], means, 0.0)
        present = (ndtr((FOV_MAX_MM - means) / self._noise_sd) -
                   ndtr((FOV_MIN_MM - means) / self._noise_sd)).prod(axis=-1)
        clear = self._clear[None, :, :] & visible[:, None, :]
        missing = 1.0 - clear * present[:, None, :]
        if n == 0:
            return missing.prod(axis=-1), np.empty((self.n_hypotheses, 4, 0))
        error = (packet - means[:, :, None, :]) / self._noise_sd
        density = (np.exp(-0.5 * error**2) / self._sigma).prod(axis=-1)
        p = clear[..., None] * density[:, None, :, :]
        if n == 1:
            artery = p[..., ARTERY, 0] * missing[..., VEIN]
            vein = p[..., VEIN, 0] * missing[..., ARTERY]
            return artery + vein, artery[..., None]
        a0 = p[..., ARTERY, 0] * p[..., VEIN, 1]
        a1 = p[..., ARTERY, 1] * p[..., VEIN, 0]
        return a0 + a1, np.stack((a0, a1), axis=-1)

    def update(self, belief, packet, s_mm, x_mm) -> tuple[np.ndarray, np.ndarray]:
        """Apply one clear-state transition and the exact packet association update."""

        likelihood, association = self._emission(packet, s_mm, x_mm)
        return self._condition(belief, likelihood, association)

    def update_projected(self, belief, packet, means_mm):
        """Update at the current measured probe pose, with one centre per hypothesis."""
        means = np.asarray(means_mm, dtype=float)
        if means.shape != (self.n_hypotheses, 2, 2):
            raise ValueError("means_mm must have shape (hypotheses, roles, 2)")
        likelihood, association = self._emission_means(self._validate_packet(packet), means)
        return self._condition(belief, likelihood, association)

    def _condition(self, belief, likelihood, association):
        belief = self._normalize_belief(belief, (self.n_hypotheses, len(STATE_INDEX)))
        predicted = belief @ self._state_transition
        joint = predicted * likelihood
        mass = float(joint.sum())
        if not np.isfinite(mass) or mass <= 0.0:
            raise UnsupportedObservation("packet has zero probability under the supplied belief")
        new_belief = joint / mass
        arterial_probs = np.sum(predicted[:, :, None] * association, axis=(0, 1)) / mass
        return new_belief, arterial_probs

    def predicted_hit(self, belief, means_mm, frames):
        """Probability of a target packet near centre at a future exposure."""
        means = np.asarray(means_mm, dtype=float)
        lo, hi = FOV_MIN_MM.copy(), FOV_MAX_MM.copy()
        lo[0], hi[0] = -CENTER_TOL_MM, CENTER_TOL_MM
        delta = (ndtr((hi - means) / self._noise_sd) -
                 ndtr((lo - means) / self._noise_sd)).prod(axis=-1)
        predicted = belief @ np.linalg.matrix_power(self._state_transition, frames)
        clear_artery = predicted[:, self._clear[:, ARTERY]].sum(axis=1)
        return float(clear_artery @ np.where(np.isfinite(delta[:, ARTERY]), delta[:, ARTERY], 0.0))

    def offset_mean(self, belief) -> np.ndarray:
        """Return the posterior mean effective offset in millimetres."""

        belief = self._normalize_belief(belief, (self.n_hypotheses, len(STATE_INDEX)))
        return np.sum(self._offsets_mm * belief.sum(axis=1)[:, None], axis=0)


__all__ = ["Scene", "Sensor", "STATE_INDEX"]
