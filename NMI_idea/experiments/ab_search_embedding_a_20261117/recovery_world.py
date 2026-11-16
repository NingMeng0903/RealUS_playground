"""Small fixed-pose chart experiment with a matched finite observation model."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys

import numpy as np

RECOVERY = Path(__file__).resolve().parent.parent / "ab_recovery_training_20261115"
sys.path.insert(0, str(RECOVERY))
from world import (  # noqa: E402
    CLUTTER_NOISE_SD, FOV_MAX, FOV_MIN, NOISE_SD, REFERENCE_KNEE_DEG, World,
)


FAMILIES = ("connection", "parallel", "wrong_prior", "alias")
FRAME_COUNT = 81
PACKET_OUTAGE = .06


@dataclass(frozen=True)
class NoiseTape:
    emit: np.ndarray
    error: np.ndarray
    uniform_clutter: np.ndarray
    outage: np.ndarray


def noise_tape(rng, frames=FRAME_COUNT):
    return NoiseTape(rng.random((frames, 4)), rng.standard_normal((frames, 3, 2)),
                     rng.uniform(FOV_MIN, FOV_MAX, (frames, 2)), rng.random(frames))


@dataclass(frozen=True)
class Scene:
    latent: int
    tape: NoiseTape


class RecoveryWorld:
    """A static hypothesis keeps geometry, shadow and clutter linked across a rollout."""

    def __init__(self, family, reference=None):
        if family not in FAMILIES:
            raise ValueError(f"unknown experiment family {family}")
        self.family = family
        self.reference = reference if reference is not None else World()
        self.station_bounds = self.reference.station_bounds
        self.angles_deg = np.full(FRAME_COUNT, REFERENCE_KNEE_DEG)
        self.grid = np.linspace(*self.station_bounds, 513)
        # The two backgrounds, two shadow lanes and three offsets are latent.
        self.hypotheses = np.asarray(list(product(range(2), (-4., 0., 4.), (-1., 1.), range(2))))
        track, lateral, side, layout = self.hypotheses.T
        structure = np.array([.975, .025]) if family == "wrong_prior" else np.array([.5, .5])
        self.prior = structure[track.astype(int)] * np.where(lateral == 0., .50, .25) / 4.
        self.shape = self._geometry_grid(track, lateral, side, layout)
        self.shadow_center = np.where(layout == 0., 12.4, 15.4)

    def _geometry_grid(self, track, lateral, side, layout):
        base = self.reference._interp_centers(self.grid)
        pairs = np.broadcast_to(base[:, None], (len(self.grid), len(track), 2, 2)).copy()
        s = self.grid[:, None]
        if self.family in ("connection", "wrong_prior"):
            center = np.where(layout == 0., 12.4, 15.4)
            phase = np.clip((s - center + 3.) / 6., 0., 1.)
            angle = np.pi * phase**2 * (3. - 2. * phase) * track
            direction = (pairs[:, :, 1] - pairs[:, :, 0]) * .5
            a, d = direction[..., 0], direction[..., 1]
            rot = np.stack((np.cos(angle) * a - np.sin(angle) * d,
                            np.sin(angle) * a + np.cos(angle) * d), axis=-1)
            mean = pairs.mean(axis=2)
            pairs[:, :, 0], pairs[:, :, 1] = mean - rot, mean + rot
        else:
            # Both tracks remain continuous and unbranched. A local depth
            # change is the only anatomical identity cue in the parallel task.
            swapped = track.astype(bool)
            pairs[:, swapped] = pairs[:, swapped, ::-1]
            if self.family == "parallel":
                center = np.where(layout == 0., 12.4, 15.4)
                pairs[:, :, 0, 1] -= 4.2 * np.exp(-.5 * ((s - center) / 2.2)**2)
        pairs[..., 0] += lateral[None, :, None]
        clutter = pairs.mean(axis=2) + np.stack((4.8 + 1.6 * side, .7 + .35 * layout), axis=-1)
        return np.concatenate((pairs, clutter[:, :, None]), axis=2)

    def nominal(self, s):
        return self.reference.nominal(s)

    def _at(self, s):
        grid_index = np.clip(float(s) * (len(self.grid) - 1) / self.grid[-1], 0., len(self.grid) - 1.)
        lo = min(int(grid_index), len(self.grid) - 2)
        return self.shape[lo] + (grid_index - lo) * (self.shape[lo + 1] - self.shape[lo])

    def target_grid(self, s, angle, *, alias=False):
        return self._at(s)[:, 0]

    def identity_prob(self, belief, station, angle, *, alias=False):
        positions = self._at(station)
        return float(belief @ (positions[:, 0, 0] > positions[:, 1, 0]))

    def predict(self, s, x, angle, *, alias=False):
        centers = self._at(s)
        means = centers - [x, 0.]
        inside = np.all((means >= FOV_MIN) & (means <= FOV_MAX), axis=-1)
        _, offset, side, _ = self.hypotheses.T
        # A narrow coupling/shadow lane attenuates both structures. Lateral
        # movement changes it; the separate packet outage is action independent.
        lane_x = np.interp(s, self.reference.s_mm, self.reference.centers_mm[:, 0, 0]) + .45 * offset + 1.5 * side
        core = np.exp(-.5 * ((float(s) - self.shadow_center) / 2.6)**4)
        attenuation = 1. - .995 * core / (1. + ((x - lane_x) / 4.5)**6)
        visibility = inside * np.column_stack((.94 * attenuation, .94 * attenuation,
                                                np.full(len(offset), .24)))
        return means, visibility

    def likelihood(self, packet, s, x, angle):
        means, visibility = self.predict(s, x, angle)
        mass, association = self.reference._likelihood_batch(packet, means, visibility)
        if not len(packet):
            mass = PACKET_OUTAGE + (1. - PACKET_OUTAGE) * mass
        return mass, association

    def update(self, belief, packet, s, x, angle, *, alias=False):
        mass, conditional = self.likelihood(packet, s, x, angle)
        posterior = belief * mass
        posterior /= posterior.sum()
        return posterior, posterior @ conditional

    def draw(self, seed):
        rng = np.random.default_rng(seed)
        weights = self.prior.copy()
        if self.family == "wrong_prior":
            # Deliberate stress cohort. Inference keeps its original atlas prior.
            weights[self.hypotheses[:, 0] == 0.] = 0.
        return Scene(int(rng.choice(len(weights), p=weights / weights.sum())), noise_tape(rng))

    def _packet(self, index, s, x, emit, error, uniform_clutter, outage):
        means, visibility = self.predict(s, x, REFERENCE_KNEE_DEG)
        mu, p = means[index], visibility[index]
        target = mu[0] + [x, 0.]
        if outage < PACKET_OUTAGE:
            return np.empty((0, 2)), np.empty(0, dtype=int), target
        # Couple the disturbances by physical track (left/right), not hidden
        # identity. Exact label swaps then produce exactly the same public y.
        order = np.r_[np.lexsort((mu[:2, 1], mu[:2, 0])), 2]
        noisy = mu[order] + error * np.vstack((NOISE_SD, NOISE_SD, CLUTTER_NOISE_SD))
        if emit[3] >= .72:
            noisy[2] = uniform_clutter
        seen = (emit[:3] < p[order]) & np.all((noisy >= FOV_MIN) & (noisy <= FOV_MAX), axis=1)
        kept = np.flatnonzero(seen)
        kept = kept[np.argsort(noisy[kept, 0], kind="stable")]
        return noisy[kept], order[kept], target

    def observe(self, scene, frame, s, x):
        tape = scene.tape
        return self._packet(scene.latent, s, x, tape.emit[frame], tape.error[frame],
                            tape.uniform_clutter[frame], tape.outage[frame])

    def sample(self, belief, s, x, angle, rng, *, alias=False):
        index = int(rng.choice(len(belief), p=belief))
        tape = noise_tape(rng, 1)
        return self.observe(Scene(index, tape), 0, s, x)[0]
