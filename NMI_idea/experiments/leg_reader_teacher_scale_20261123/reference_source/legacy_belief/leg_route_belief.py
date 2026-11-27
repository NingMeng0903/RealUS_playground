"""Associate unlabelled ultrasound centroids with the full-leg anatomy prior."""

from argparse import ArgumentParser
from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.special import ndtr


ASSET = Path(__file__).resolve().parent / "results/leg_proximal_geometry/right_groin_to_ankle.npz"
GSV_EXTENSION = "R_SAPH_V_MEDIAL_CALF_EXTENSION"
GSV = "R_SAPH_V"
CFV = "R_COM_FEM_V"


def vessel_identity(label):
    return {GSV_EXTENSION: GSV, "R_DEEP_FEM_CONNECTOR": "R_DEEP_FEM_V"}.get(label, label)


def transport(points, weights, transforms):
    if transforms is None:
        return points
    homogeneous = np.column_stack((points, np.ones(len(points))))
    weights = weights.astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("nj,jab,nb->na", weights, transforms[:, :3], homogeneous)


@dataclass(frozen=True)
class View:
    target: str
    labels: tuple[str, ...]
    centers_mm: np.ndarray
    coarse: np.ndarray
    rotation: np.ndarray


class RegisteredLeg:
    def __init__(self, path=ASSET):
        with np.load(path) as data:
            self.asset = {name: data[name] for name in data.files}

    def anatomy(self, transforms=None):
        data = self.asset
        vertices = transport(data["vertices_m"], data["vertex_lbs_weights"], transforms)
        lines = np.split(vertices, data["point_offsets"][1:-1])
        meshes = {}
        for kind in ("artery", "vein"):
            points = transport(data[f"native_{kind}_vertices_m"], data[f"native_{kind}_lbs_weights"], transforms)
            meshes[kind] = trimesh.Trimesh(points, data[f"native_{kind}_faces"], process=False)
        return tuple(zip(data["labels"], lines, strict=True)), meshes

    @cached_property
    def rest_anatomy(self):
        return self.anatomy()

    def project(self, pose, target, transforms=None):
        rotation = pose[:3, :3]
        origin = pose[:3, 3]
        lines, meshes = self.rest_anatomy if transforms is None else self.anatomy(transforms)
        target = vessel_identity(target)
        named = []
        for label, line in lines:
            xy = (line - origin) @ rotation * 1000.
            z = xy[:, 2]
            on_plane = abs(z) <= 1.e-7
            z0, z1 = z[:-1], z[1:]
            hits = (z0 * z1 < 0.) & ~on_plane[:-1] & ~on_plane[1:]
            t = -z0[hits] / (z1[hits] - z0[hits])
            crossing = xy[:-1, :2][hits] + t[:, None] * np.diff(xy[:, :2], axis=0)[hits]
            for center in np.vstack((xy[on_plane, :2], crossing)):
                name = vessel_identity(label)
                if all(name != other or np.linalg.norm(center - point) > 1. for other, point in named):
                    named.append((name, center))
        # Collapse the duplicate representation of the same lumen. Distinct
        # named branches are kept, even if the current slice does not resolve them.
        extras = []
        for kind, mesh in meshes.items():
            section = mesh.section(plane_normal=rotation[:, 2], plane_origin=origin)
            if section is None:
                continue
            contours = [line[:-1] for line in section.discrete
                        if len(line) >= 4 and np.linalg.norm(line[0] - line[-1]) < 1.e-7]
            for line in contours:
                polygon = ((line - origin) @ rotation[:, :2]) * 1000.
                next_point = np.roll(polygon, -1, axis=0)
                area = polygon[:, 0] * next_point[:, 1] - next_point[:, 0] * polygon[:, 1]
                if abs(area.sum()) < .4:
                    continue
                center = area @ (polygon + next_point) / (3. * area.sum())
                radius = np.sqrt(abs(area.sum()) / (2. * np.pi))
                if kind == "vein" and any(np.linalg.norm(center - point) <= radius + 1.5 for _, point in named):
                    continue
                extras.append((f"mesh_{kind}", center))
        named.sort(key=lambda item: item[0] != target)
        labels = tuple(name for name, _ in named + extras)
        centers = np.asarray([point for _, point in named + extras]).reshape(-1, 2)
        coarse = np.array([False] * len(named) + [True] * len(extras), dtype=bool)
        return View(target, labels, centers, coarse, rotation)


def associate(missed, ratios):
    """Marginalize single-source matches, misses and Poisson false contours."""
    hypotheses, count, channels = ratios.shape
    bits = 1 << np.arange(count)
    masks = np.arange(1 << count)
    available = [(masks[(masks & bit) == 0], bit) for bit in bits]
    suffixes = [None] * (channels + 1)
    suffixes[-1] = np.ones((hypotheses, len(masks)))
    for channel in range(channels - 1, -1, -1):
        tail = suffixes[channel + 1]
        updated = missed[:, channel, None] * tail
        for k, (source, bit) in enumerate(available):
            updated[:, source] += ratios[:, k, channel, None] * tail[:, source | bit]
        suffixes[channel] = updated
    mass = suffixes[0][:, 0]
    prefix = np.zeros_like(suffixes[0])
    prefix[:, 0] = 1.
    matches = np.zeros_like(ratios)
    for channel in range(channels):
        updated = missed[:, channel, None] * prefix
        for k, (source, bit) in enumerate(available):
            claimed = source | bit
            weight = ratios[:, k, channel, None] * prefix[:, source]
            updated[:, claimed] += weight
            matches[:, k, channel] = (weight * suffixes[channel + 1][:, claimed]).sum(axis=1)
        prefix = updated
    return mass, matches / np.maximum(mass[:, None, None], 1.e-100)


class RouteBelief:
    def __init__(self, half_width_mm=25., depth_mm=75., *, competitors=True):
        self.lower = np.array([-half_width_mm, 3.])
        self.upper = np.array([half_width_mm, depth_mm])
        self.competitors = competitors
        # Local slice approximation to a shared body-frame registration offset.
        offsets = np.array([-6., -3., 0., 3., 6.])
        offsets = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)
        self.offsets = np.column_stack((offsets[:, 0], np.zeros(len(offsets)), offsets[:, 1]))
        self.weights = np.exp(-.5 * ((offsets / 3.5) ** 2).sum(axis=1))
        self.weights /= self.weights.sum()
        self.history = []
        self.last_pose = None
        self.source_probabilities = {}

    def update(self, view, measured_centroids_mm, *, position_m=None, sensitivity=1.):
        packet = np.asarray(measured_centroids_mm, dtype=float).reshape(-1, 2)
        if not np.isfinite(packet).all():
            raise ValueError("centroids must be finite in probe coordinates")
        if not 0. <= sensitivity <= 1.:
            raise ValueError("sensitivity must be between zero and one")
        target = np.array([label == view.target for label in view.labels], dtype=bool)
        keep = np.ones(len(target), dtype=bool) if self.competitors else target
        sigma = np.where(view.coarse[keep, None], [2.5, 2.2], [1.8, 1.5])
        means = view.centers_mm[None, keep] + (self.offsets @ view.rotation[:, :2])[:, None]
        # Ongoing segmentation failure is plausible even inside the geometric window.
        detect = sensitivity * np.where(view.coarse[keep], .52, .75)
        mass = np.prod(ndtr((self.upper - means) / sigma) - ndtr((self.lower - means) / sigma), axis=-1)
        visible = mass.max(axis=0) > 1.e-6
        means, sigma, detect, mass = means[:, visible], sigma[visible], detect[visible], mass[:, visible]
        labels = np.asarray(view.labels, dtype=str)[keep][visible]
        missed = 1. - detect * mass
        delta = (packet[None, :, None] - means[:, None]) / sigma
        density = np.exp(-.5 * (delta ** 2).sum(axis=-1)) / (2. * np.pi * sigma.prod(axis=-1))
        clutter = .38 / np.prod(self.upper - self.lower)
        in_window = ((packet >= self.lower) & (packet <= self.upper)).all(axis=1)
        mass, association = associate(missed, density * (in_window[:, None] * detect) / clutter)
        if not len(packet):
            mass = .15 + .85 * mass
        strength = 1.
        if position_m is not None and self.last_pose is not None:
            movement_mm = 1000. * np.linalg.norm(position_m - self.last_pose)
            strength = .4 + .6 * min(movement_mm / 5., 1.)
        posterior = self.weights * np.maximum(mass, 1.e-100) ** strength
        self.weights = posterior / posterior.sum()
        self.last_pose = position_m
        target_probability = association[:, :, labels == view.target].sum(axis=2)
        self.history.append(target_probability)
        averaged = np.einsum("h,hfc->fc", self.weights, association)
        self.source_probabilities = {name: averaged[:, labels == name].sum(axis=1) for name in dict.fromkeys(labels)}
        self.source_probabilities["unexplained"] = np.clip(1. - averaged.sum(axis=1), 0., 1.)
        return self.weights @ target_probability

    def retained_probability(self, frame):
        """Re-score saved candidates using all geometry evidence available now."""
        return self.weights @ self.history[frame]


def run_stream(path, anatomy, half_width, depth):
    belief = RouteBelief(half_width, depth)
    with path.open() as stream:
        for row in map(json.loads, stream):
            pose = np.asarray(row["pose_m"])
            transforms = np.asarray(row["joint_transforms"]) if "joint_transforms" in row else None
            view = anatomy.project(pose, row["target"], transforms)
            confidence = belief.update(view, row["centroids_mm"], position_m=pose[:3, 3],
                                       sensitivity=row.get("sensitivity", 1.))
            result = {"target": view.target, "target_probabilities": confidence.tolist(),
                      "source_probabilities": {name: p.tolist() for name, p in belief.source_probabilities.items()}}
            if "recheck_frame" in row:
                result["rechecked_probabilities"] = belief.retained_probability(row["recheck_frame"]).tolist()
            print(json.dumps(result), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path, help="JSONL of measured probe poses and centroids")
    parser.add_argument("--asset", type=Path, default=ASSET)
    parser.add_argument("--half-width-mm", type=float, default=25.)
    parser.add_argument("--depth-mm", type=float, default=75.)
    args = parser.parse_args()
    run_stream(args.frames, RegisteredLeg(args.asset), args.half_width_mm, args.depth_mm)


if __name__ == "__main__":
    main()
