"""Rigid bone boundary targets driven by the unchanged authored LBS weights.

The small weighted moments are baked once. Runtime evaluates those moments
and a 3x3 polar decomposition, with no spatial search or iterative fitting.
This is a boundary-target generator, not a full anatomy retargeter: consumers
must transport the resulting residual to soft material and audit articulation.
In particular, making bones rigid does not remove pre-existing intersections.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np


def _digest(*values):
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


@dataclass
class BoneComponentsV15:
    names: np.ndarray
    ranges: np.ndarray
    rest: np.ndarray
    centers: np.ndarray
    counts: np.ndarray
    controller_ids: np.ndarray
    component_ids: np.ndarray
    weight_sums: np.ndarray
    first_moments: np.ndarray
    centered_moments: np.ndarray
    cross_moments: np.ndarray
    source_contract_digest: str
    controller_count: int

    def __post_init__(self):
        for name in ('names', 'ranges', 'rest', 'centers', 'counts',
                     'controller_ids', 'component_ids', 'weight_sums',
                     'first_moments', 'centered_moments', 'cross_moments'):
            value = np.array(getattr(self, name), copy=True)
            if value.dtype.kind in 'fc' and not np.isfinite(value).all():
                raise ValueError(f'non-finite bone component {name}')
            value.setflags(write=False)
            setattr(self, name, value)
        size = len(self.names)
        k = len(self.controller_ids)
        if (not size or self.ranges.shape != (size, 2)
                or self.centers.shape != (size, 3)
                or self.counts.shape != (size,) or np.any(self.counts < 3)
                or self.rest.ndim != 2 or self.rest.shape[1] != 3):
            raise ValueError('invalid bone component geometry')
        if (self.ranges.dtype.kind not in 'iu'
                or self.controller_ids.dtype.kind not in 'iu'
                or self.component_ids.dtype.kind not in 'iu'
                or self.component_ids.shape != (k,)
                or self.weight_sums.shape != (k,)
                or self.first_moments.shape != (k, 3)
                or self.centered_moments.shape != (k, 3)
                or self.cross_moments.shape != (k, 3, 3)
                or np.any(self.controller_ids < 0)
                or np.any(self.controller_ids >= self.controller_count)
                or np.any(self.component_ids < 0) or np.any(self.component_ids >= size)
                or np.any(self.weight_sums <= 0)):
            raise ValueError('invalid bone controller moments')
        if (np.any(self.ranges[:, 0] < 0) or np.any(self.ranges[:, 1] > len(self.rest))
                or np.any(self.ranges[:, 1] - self.ranges[:, 0] != self.counts)
                or np.any(self.ranges[1:, 0] < self.ranges[:-1, 1])):
            raise ValueError('invalid or overlapping bone ranges')
        if len(str(self.source_contract_digest)) != 64:
            raise ValueError('missing original topology/weight identity')

    @classmethod
    def from_asset(cls, asset, target_rest=None, *, mesh_names=None):
        rest = np.asarray(asset.vertices_rest if target_rest is None else target_rest,
                          dtype=np.float64)
        indices = np.asarray(asset.driver_indices)
        weights = np.asarray(asset.driver_weights, dtype=np.float64)
        controller_count = len(asset.source_bone_names)
        if (rest.shape != np.shape(asset.vertices_rest) or not np.isfinite(rest).all()
                or indices.dtype.kind not in 'iu' or indices.shape != weights.shape
                or indices.shape[0] != len(rest) or np.any(indices < 0)
                or np.any(indices >= controller_count) or not np.isfinite(weights).all()
                or np.any(weights < 0)
                or not np.allclose(weights.sum(1), 1, atol=2e-6, rtol=0)):
            raise ValueError('invalid original sparse weights or target rest')
        requested = None if mesh_names is None else set(mesh_names)
        labels, ranges, centers, counts = [], [], [], []
        controller_ids, component_ids, sums, first, centered, cross = [], [], [], [], [], []
        for name, tissue, (start, stop) in zip(asset.source_mesh_names,
                                              asset.source_tissues,
                                              asset.source_vertex_ranges):
            # The authored render category also includes intervertebral discs.
            # Preserve their original flexible response; a render tissue code
            # is not sufficient evidence that a component must be rigid.
            flexible_disc = str(name).startswith('Disc_')
            if (str(tissue) != 'bone' or flexible_disc
                    or (requested is not None and name not in requested)):
                continue
            start, stop = int(start), int(stop)
            points = rest[start:stop]
            if len(points) < 3 or np.linalg.matrix_rank(points - points.mean(0), tol=1e-12) < 2:
                raise ValueError(f'degenerate bone component: {name}')
            center = points.mean(0)
            component = len(labels)
            labels.append(name); ranges.append((start, stop)); centers.append(center); counts.append(len(points))
            slots = indices[start:stop]
            slot_weights = weights[start:stop]
            for controller in np.unique(slots[slot_weights > 0]):
                w = np.sum(np.where(slots == controller, slot_weights, 0), axis=1)
                controller_ids.append(controller); component_ids.append(component)
                sums.append(w.sum())
                first.append(w @ points)
                centered.append(w @ (points - center))
                cross.append(np.einsum('i,ij,ik->jk', w, points-center, points))
        if requested is not None and requested != set(labels):
            raise ValueError(f'unknown/non-bone or flexible component names: {sorted(requested-set(labels))}')
        if not labels:
            raise ValueError('no bone components')
        return cls(np.asarray(labels), np.asarray(ranges), rest,
                   np.asarray(centers), np.asarray(counts), np.asarray(controller_ids),
                   np.asarray(component_ids), np.asarray(sums), np.asarray(first),
                   np.asarray(centered), np.asarray(cross),
                   _digest(asset.faces, asset.driver_indices, asset.driver_weights,
                           asset.source_bone_parents), controller_count)

    def transforms(self, skinning_transforms):
        """Return the exact least-squares rigid target of each original LBS bone.

        Baked weighted moments reproduce the dense point covariance. This
        analytic polar evaluation does not optimize joints or rebind geometry.
        """
        transforms = np.asarray(skinning_transforms, dtype=np.float64)
        if (transforms.shape != (self.controller_count, 4, 4)
                or not np.isfinite(transforms).all()
                or not np.allclose(transforms[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0)):
            raise ValueError('invalid controller skinning transforms')
        rotations = transforms[:, :3, :3]
        if (not np.allclose(rotations.swapaxes(1, 2) @ rotations, np.eye(3), atol=1e-5, rtol=0)
                or np.any(np.linalg.det(rotations) <= 0)):
            raise ValueError('controller transforms must be proper rigid, without scale')
        selected = transforms[self.controller_ids]
        r, t = selected[:, :3, :3], selected[:, :3, 3]
        point_sum = np.zeros_like(self.centers)
        covariance = np.zeros((len(self.names), 3, 3), dtype=np.float64)
        np.add.at(point_sum, self.component_ids,
                  np.einsum('kij,kj->ki', r, self.first_moments) + self.weight_sums[:, None] * t)
        np.add.at(covariance, self.component_ids,
                  self.cross_moments @ r.swapaxes(1, 2)
                  + self.centered_moments[:, :, None] * t[:, None, :])
        u, singular, vt = np.linalg.svd(covariance)
        if np.any(singular[:, 1] <= np.maximum(singular[:, 0], 1e-30) * 1e-10):
            raise ValueError('bone rigid response loses rank at this pose')
        sign = np.ones((len(self.names), 3)); sign[:, 2] = np.linalg.det(vt.swapaxes(1, 2) @ u.swapaxes(1, 2))
        rotation = (vt.swapaxes(1, 2) * sign[:, None, :]) @ u.swapaxes(1, 2)
        result = np.tile(np.eye(4), (len(self.names), 1, 1))
        result[:, :3, :3] = rotation
        result[:, :3, 3] = point_sum / self.counts[:, None] - np.einsum('bij,bj->bi', rotation, self.centers)
        return result

    def boundary_targets(self, skinning_transforms):
        transforms = self.transforms(skinning_transforms)
        ids = np.concatenate([np.arange(start, stop) for start, stop in self.ranges])
        targets = np.concatenate([self.rest[start:stop] @ transform[:3, :3].T + transform[:3, 3]
                                  for (start, stop), transform in zip(self.ranges, transforms)])
        return ids, targets, transforms

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values['schema'] = np.array('bone_components_v15')
        values['payload_digest'] = np.array(_digest(*[np.asarray(values[name]) for name in sorted(values)]))
        with path.open('xb') as handle:
            np.savez_compressed(handle, **values)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        signature = str(values.pop('payload_digest').item())
        if signature != _digest(*[values[name] for name in sorted(values)]):
            raise ValueError('bone component payload identity mismatch')
        if str(values.pop('schema').item()) != 'bone_components_v15':
            raise ValueError('unsupported bone component schema')
        values['source_contract_digest'] = str(values['source_contract_digest'].item())
        values['controller_count'] = int(values['controller_count'].item())
        return cls(**values)
