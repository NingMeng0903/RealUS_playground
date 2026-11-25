"""Withheld geometry challenge: axial shear of an existing synthetic leg layout."""

from dataclasses import replace
import hashlib

import numpy as np


def _shape_checks(surface, centers, links, active):
    """Audit affected station and supported mid-edge samples on the v15 chart."""
    finite = np.isfinite(centers).all(axis=-1)
    stations, roles = np.where(finite & active[:, None])
    edges, next_roles = np.where(finite[:-1] & finite[1:] & links &
                                 (active[:-1] | active[1:])[:, None])
    left, right = np.r_[stations, edges], np.r_[stations, edges + 1]
    points = np.concatenate((centers[stations, roles],
                             .5 * (centers[edges, next_roles] + centers[edges + 1, next_roles])))
    origins, radii, base_origin, base_radii, base_y = surface.shape
    local = points - .5 * (origins[left] + origins[right])
    angle = (np.arctan2(local[:, 1], local[:, 0]) * 12. / np.pi) % 24.
    a = angle.astype(int) % 24
    b, mix = (a + 1) % 24, angle % 1.

    def radius(grid):
        return .5 * ((1. - mix) * (grid[left, a] + grid[right, a])
                     + mix * (grid[left, b] + grid[right, b]))

    width = radius(radii)
    outward = np.linalg.norm(local, axis=-1) - width
    xz = .5 * (base_origin[left] + base_origin[right]) + local * (radius(base_radii) / width)[:, None]
    bone = surface.bone(np.column_stack((xz[:, 0], .5 * (base_y[left] + base_y[right]), xz[:, 1])))
    return outward, np.nan_to_num(bone, nan=.15)


class LocalShear:
    """Keep vessel identity and topology fixed while changing current tangents."""

    def __init__(self, base, station):
        self.base, self.station = base, station

    def build(self, source, seed, profile, split):
        layout = self.base.build(source, seed, profile, split)
        surface = self.base.scanner.surface
        y = surface.y
        rng = np.random.default_rng(seed + 70105)
        azimuth = rng.uniform(0., 2. * np.pi)
        slope = rng.uniform(.34, .56) * np.array([np.cos(azimuth), np.sin(azimuth)])
        distance = y - y[self.station]
        field = distance[:, None] * np.exp(-.5 * (distance[:, None] / 18.) ** 2) * slope
        active = np.linalg.norm(field, axis=1) > 1.e-6
        old_skin, old_bone = _shape_checks(surface, layout.centers, layout.links, active)
        for amplitude in (1., .72, .45, .25):
            changed = layout.centers + amplitude * field[:, None]
            skin, bone = _shape_checks(surface, changed, layout.links, active)
            if np.any(skin > np.maximum(0., old_skin) + .08):
                continue
            if np.any(bone > np.maximum(.35, old_bone + .045)):
                continue
            vertices = tuple((s, roles, point + amplitude * np.array(
                [np.interp(s, np.arange(len(y)), component) for component in field.T]))
                             for s, roles, point in layout.shared_vertices)
            # A common displacement keeps every adjacency and same-section
            # separation. Move any shared vertex with its incident curves.
            suffix = hashlib.sha256(layout.identifier.encode() + changed.tobytes()).hexdigest()[:20]
            info = {"station": self.station, "accepted": True, "factor": amplitude,
                    "geometry_samples": len(skin),
                    "peak_mm": float(amplitude * np.linalg.norm(field, axis=1).max()),
                    "added_slope_deg": float(np.rad2deg(np.arctan(amplitude * np.linalg.norm(slope))))}
            return replace(layout, identifier=f"v16_shear_{suffix}", centers=changed, shared_vertices=vertices,
                           metadata=layout.metadata | {"v16_shear": info})
        return replace(layout, metadata=layout.metadata | {"v16_shear": {"station": self.station,
                                                                       "accepted": False}})
