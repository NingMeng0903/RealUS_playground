"""V6.1 whole-segment risk for a contact-preserving oblique ultrasound sweep."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# Columns of the probe frame are image lateral, plane normal and beam depth.
SCALE = np.array([4., 3., np.sin(np.deg2rad(10.)), np.sin(np.deg2rad(10.))])
SWEEP_DEG = 4.
SWEEP = np.deg2rad(SWEEP_DEG)
MAX_ATTITUDE_DEG = 30.
ALPHA = .20
PULSES = 7
ANGULAR_RATE_DEG_S = 35.


@dataclass(frozen=True)
class Belief:
    centers: np.ndarray
    tangents: np.ndarray
    weights: np.ndarray
    objects: tuple
    probabilities: tuple

    @property
    def known(self):
        return float(self.weights.sum())

    def mean(self):
        w = self.weights / self.known
        return w @ self.centers, unit(w @ self.tangents)


@dataclass(frozen=True)
class Sweep:
    sector: int
    tag: str
    middle: np.ndarray

    @property
    def axis(self):
        return self.middle[:, 0]

    def frame(self, fraction):
        angle = (np.asarray(fraction) - .5) * SWEEP
        rotation = Rotation.from_rotvec(angle[..., None] * self.axis).as_matrix()
        return rotation @ self.middle


def unit(vector):
    size = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(size, 1.e-12)


def base_frame(sector):
    angle = sector * np.pi / 12.
    radial = np.array([np.cos(angle), 0., np.sin(angle)])
    lateral = np.array([-radial[2], 0., radial[0]])
    return np.column_stack((lateral, [0., 1., 0.], -radial))


def aim(origin, neutral, center, tangent):
    """Aim toward a public center without exceeding the chart's lean envelope."""
    beam = unit(center - origin)
    inward = neutral[:, 2]
    cosine = beam @ inward
    limit = np.deg2rad(21.)
    if cosine < np.cos(limit):
        sideways = unit(beam - cosine * inward)
        beam = np.cos(limit) * inward + np.sin(limit) * sideways
    normal = unit(tangent - (tangent @ beam) * beam)
    if np.linalg.norm(normal) < .5:
        return neutral
    lateral = np.cross(normal, beam)
    return np.column_stack((lateral, normal, beam))


def attitude(neutral, frame):
    return Rotation.from_matrix(frame @ neutral.T).magnitude()


def candidates(surface, station, approaches, belief):
    """Only the pre-probe belief and checked chart contact paths enter the menu."""
    representatives = []
    if belief.known > 0:
        representatives.append(("mean", *belief.mean()))
        first = 0
        for key, n in belief.objects:
            indices = slice(first, first + n)
            w = belief.weights[indices]
            if w.sum() > 0:
                w = w / w.sum()
                representatives.append((key, w @ belief.centers[indices],
                                        unit(w @ belief.tangents[indices])))
            first += n
    result = []
    for sector in sorted(approaches):
        origin = surface.world(station, sector)
        neutral = base_frame(sector)
        frames = [("axial", neutral)]
        frames.extend((tag, aim(origin, neutral, c, t)) for tag, c, t in representatives)
        seen = set()
        for tag, frame in frames:
            for offset in (0., -6., 6.):
                middle = Rotation.from_rotvec(np.deg2rad(offset) * frame[:, 0]).as_matrix() @ frame
                if attitude(neutral, middle) + SWEEP / 2 > np.deg2rad(MAX_ATTITUDE_DEG):
                    continue
                key = tuple(np.round(middle.ravel(), 7))
                if key not in seen:
                    seen.add(key)
                    result.append(Sweep(sector, f"{tag}/{offset:+g}", middle))
    return result


def residual(frame, displacement, tangent):
    """Weighted PointAt and short-axis residual, with signed probe-frame axes."""
    x, y, z = np.moveaxis(frame, -1, 0)
    terms = np.broadcast_arrays(np.sum(x * displacement, axis=-1), np.sum(y * displacement, axis=-1),
                                np.sum(x * tangent, axis=-1), np.sum(z * tangent, axis=-1))
    return np.stack(terms, axis=-1) / SCALE


def derivative_bound(displacement):
    """Uniform norm bound on the first and second angular derivatives."""
    length = np.linalg.norm(displacement, axis=-1)
    return np.sqrt((length / SCALE[0]) ** 2 + (length / SCALE[1]) ** 2 + 2. / SCALE[2] ** 2)


def _checks(sweep, origin, centers, tangents):
    d = centers - origin
    frame = sweep.frame(0.)
    r = residual(frame, d, tangents)
    derivative = np.cross(sweep.axis, frame.T).T
    j = residual(derivative, d, tangents)
    endpoint = r + SWEEP * j
    remainder = .5 * SWEEP ** 2 * derivative_bound(d)
    peak = np.maximum(np.linalg.norm(r, axis=-1), np.linalg.norm(endpoint, axis=-1))
    # V6.1's sign/facing condition is checked separately from the quadratic residual.
    depth = d @ frame[:, 2]
    facing = tangents @ frame[:, 1]
    visible_depth = (depth - SWEEP * np.linalg.norm(d, axis=-1) >= 3.) & (
        depth + SWEEP * np.linalg.norm(d, axis=-1) <= 80.)
    forward = facing > SWEEP
    return peak + remainder, r, endpoint, remainder, visible_depth & forward


def evaluate(sweep, origin, belief, full):
    """Finite-support version of V6.1 with endpoint convexity and a Taylor bound.

    Count all passing hypotheses over the same segment. Unresolved mass counts
    as a possible violation. The mean arm collapses the very same belief into
    one conditional center and direction, with its resolved mass preserved.
    """
    mass = belief.known
    if mass <= 1.e-8:
        return {"lower": 0., "risk": None, "markov_upper": 1., "certified": False}
    if full:
        centers, tangents, weights = belief.centers, belief.tangents, belief.weights
    else:
        center, tangent = belief.mean()
        centers, tangents, weights = center[None], tangent[None], np.array([mass])
    upper, r, endpoint, remainder, facing = _checks(sweep, origin, centers, tangents)
    good = (upper <= 1.) & facing
    lower = float(np.clip(weights @ good, 0., 1.))
    # The original moment-only sufficient bound is logged as a comparison.
    rho = remainder.max()
    endpoint_moments = weights @ np.stack(((r ** 2).sum(axis=1), (endpoint ** 2).sum(axis=1)), axis=1)
    miss = 1. - float(weights @ facing)
    markov = 1. if rho >= 1. else min(1., miss + endpoint_moments.sum() / (1. - rho) ** 2)
    # Back up a failed certificate with the smallest predicted full-segment
    # peak. This is an exploratory action and cannot be marked certified.
    risk = float(weights @ (upper ** 2 + 4. * ~facing) / mass)
    return {"lower": lower, "risk": risk, "markov_upper": markov, "certified": lower >= 1. - ALPHA,
            "upper_residual_mean": float(weights @ upper / mass)}


def choose(surface, station, menu, approaches, belief, full):
    scores = [evaluate(a, surface.world(station, a.sector), belief, full) for a in menu]
    def rank(i):
        result = scores[i]
        risk = result["risk"] if result["risk"] is not None else np.inf
        return (not result["certified"], -round(result["lower"], 10), risk,
                approaches[menu[i].sector].travel_ticks, i)
    return min(range(len(menu)), key=rank), scores


def validate_segment(sweep, origin, center, tangents, depth_min=3., depth_max=80.):
    """Independent dense true-geometry verification with Lipschitz bracketing."""
    d = center - origin
    for points in (129, 1025, 8193):
        frame = sweep.frame(np.linspace(0., 1., points))
        errors = residual(frame[:, None], d, tangents)
        peak = float(np.linalg.norm(errors, axis=-1).max())
        delta = SWEEP / (2 * (points - 1))
        enclosure = peak + delta * derivative_bound(d)
        depth = frame[:, :, 2] @ d
        facing = frame[:, :, 1] @ tangents.T
        dz = delta * np.linalg.norm(d)
        pass_domain = (depth.min() - dz >= depth_min and depth.max() + dz <= depth_max
                       and facing.min() > delta)
        if (enclosure <= 1. and pass_domain) or peak > 1. or points == 8193:
            break
    worst = np.max(np.abs(errors), axis=(0, 1))
    return {"pass": bool(enclosure <= 1. and pass_domain), "peak": peak, "upper": float(enclosure),
            "near_grid_boundary": bool(peak <= 1. < enclosure),
            "lateral_mm": float(worst[0] * SCALE[0]), "off_plane_mm": float(worst[1] * SCALE[1]),
            "angle_deg": float(np.rad2deg(np.arcsin(np.clip(
                np.linalg.norm(errors[..., 2:], axis=-1).max() * SCALE[2], 0., 1.)))),
            "points": points}
