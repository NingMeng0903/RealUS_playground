"""Experimental baked visceral volume response with a shared material field.

The original sparse LBS is the motion authority. A closed organ supplies a
volume signal, expressed as a fixed cubic polynomial of its controller
transforms (Cauchy--Binet). No pose samples, nearest rebinding, or optimization
are used at playback. This is a local corrective experiment, not a complete
bone/skin constrained anatomy transfer or an anatomical acceptance gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path

import numpy as np


def mesh_volume(vertices, faces):
    p = np.asarray(vertices, dtype=float)
    q = (p - p.mean(axis=0))[np.asarray(faces, dtype=int)]
    return float(np.einsum('ij,ij->i', q[:, 0], np.cross(q[:, 1], q[:, 2])).sum() / 6)


def _closed_oriented(faces):
    f = np.asarray(faces, dtype=int)
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges, inv, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                    return_inverse=True, return_counts=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    balance = np.bincount(inv, weights=signs, minlength=len(edges))
    if not len(edges) or np.any(counts != 2) or np.any(balance != 0):
        raise ValueError('volume authority requires a closed consistently oriented mesh')


@dataclass
class BakedVolumeV16:
    controller_ids: np.ndarray
    origin: np.ndarray
    triples: np.ndarray
    coefficients: np.ndarray
    rest_volume: float

    @classmethod
    def compile(cls, points, faces, indices, weights):
        p = np.asarray(points, dtype=float)
        f = np.asarray(faces, dtype=int)
        _closed_oriented(f)
        ids = np.asarray(indices, dtype=int)
        weights = np.asarray(weights, dtype=float)
        if (p.ndim != 2 or p.shape[1] != 3 or ids.shape != weights.shape
                or ids.shape[0] != len(p) or np.any(ids < 0)
                or not np.isfinite(p).all() or not np.isfinite(weights).all()
                or np.any(weights < 0)
                or not np.allclose(weights.sum(axis=1), 1, atol=2e-6, rtol=0)):
            raise ValueError('invalid original sparse material weights or points')
        controllers = np.unique(ids[weights > 0])
        if len(controllers) > 12:
            raise ValueError('this experimental volume polynomial supports at most 12 controllers')
        origin = p.mean(axis=0)
        phi = np.zeros((len(p), len(controllers), 4))
        homogeneous = np.column_stack((p - origin, np.ones(len(p))))
        for k, c in enumerate(controllers):
            mass = np.where(ids == c, weights, 0).sum(axis=1)
            phi[:, k] = mass[:, None] * homogeneous
        phi = phi.reshape(len(p), -1)
        # An independent constant column removes a common translation exactly
        # without renormalizing the authored weights. Their row sums can differ
        # from one by the source asset's small floating-point tolerance.
        phi = np.column_stack((phi, np.ones(len(p))))
        triples = np.asarray(list(combinations(range(phi.shape[1]), 3)), dtype=np.int32)
        coefficients = np.zeros(len(triples))
        # Bounded memory: each face's 3x3 minor is an algebraic coefficient.
        for start in range(0, len(triples), 24):
            selected = triples[start:start + 24]
            minors = phi[f[:, :, None, None], selected[None, None, :, :]]
            minors = minors.transpose(0, 2, 1, 3)
            coefficients[start:start + len(selected)] = np.linalg.det(minors).sum(axis=0) / 6
        volume = mesh_volume(p, f)
        if abs(volume) < 1e-10:
            raise ValueError('volume authority is degenerate')
        return cls(controllers, origin, triples, coefficients, volume)

    def evaluate(self, transforms):
        t = np.asarray(transforms, dtype=float)[self.controller_ids]
        center = np.einsum('nij,j->ni', t[:, :3, :3], self.origin) + t[:, :3, 3]
        # Remove a common translation before the polynomial to avoid large
        # AMASS world translations causing cancellation of closed volumes.
        rows = np.concatenate((t[:, :3, :3].transpose(0, 2, 1), center[:, None]), axis=1)
        if int(self.triples.max()) == len(t) * 4:
            rows = np.concatenate((rows.reshape(-1, 3), -center[:1]), axis=0)
        else:
            # Preserve the exact replay of the first two experimental packs;
            # their earlier coefficient layout had no independent constant.
            # This legacy path inherits their documented ~1e-8 volume error.
            rows[:, 3] -= center[:1].copy()
            rows = rows.reshape(-1, 3)
        determinants = np.linalg.det(rows[self.triples])
        return float(self.coefficients @ determinants)


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def compile_shared_field(base, mesh_name='Large_Intestine', *, reach_m=.04, bone_guard_m=.04):
    """Sample one continuous rest field on every tissue, without mesh owners.

    The field is radial near the selected organ, decays over ``reach_m`` and
    vanishes at bone surfaces. All neighbouring soft materials use the same
    spatial function. Bone and cranial geometry are preserved exactly.
    """
    import igl
    a = base.source_asset
    names = list(a.source_mesh_names)
    ranges = np.asarray(a.source_vertex_ranges, dtype=int)
    tissues = np.asarray(a.source_tissues)
    index = names.index(mesh_name)
    start, stop = ranges[index]
    faces = np.asarray(a.faces, dtype=int)
    own_faces = faces[(faces[:, 0] >= start) & (faces[:, 0] < stop)] - start
    rest = np.asarray(base.target_rest, dtype=float)
    organ = rest[start:stop]
    volume = BakedVolumeV16.compile(organ, own_faces, base.indices[start:stop], base.weights[start:stop])
    protected = np.zeros(len(rest), dtype=bool)
    for (s, e), tissue in zip(ranges, tissues):
        if tissue == 'bone':
            protected[s:e] = True
    bone_faces = faces[protected[faces].all(axis=1)]
    bounds = (organ.min(axis=0) - reach_m, organ.max(axis=0) + reach_m)
    query_ids = np.flatnonzero((rest >= bounds[0]).all(axis=1) &
                               (rest <= bounds[1]).all(axis=1) & ~protected)
    query = rest[query_ids]
    sq, _, _ = igl.point_mesh_squared_distance(query, organ, own_faces.astype(np.int32))
    winding = np.asarray(igl.winding_number(organ, own_faces.astype(np.int32), query)).ravel()
    outside_distance = np.where(np.abs(winding) >= .5, 0, np.sqrt(np.maximum(sq, 0)))
    envelope = 1 - _smoothstep(outside_distance / reach_m)
    bone_sq, _, _ = igl.point_mesh_squared_distance(query, rest, bone_faces.astype(np.int32))
    envelope *= _smoothstep(np.sqrt(np.maximum(bone_sq, 0)) / bone_guard_m)
    field = np.zeros_like(rest)
    field[query_ids] = envelope[:, None] * (query - volume.origin)
    from .soft_constraints import unique_mesh_edges
    edges = unique_mesh_edges(faces)
    rest_edges = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    field_edges = np.linalg.norm(field[edges[:, 0]] - field[edges[:, 1]], axis=1)
    nondegenerate = rest_edges > 1e-10
    if np.any((~nondegenerate) & (field_edges > 1e-10)):
        raise ValueError('coincident material points receive inconsistent correction')
    gradient_bound = float(np.max(field_edges[nondegenerate] / rest_edges[nondegenerate]))
    return volume, field, dict(mesh_name=mesh_name, vertex_range=[int(start), int(stop)],
        reach_m=reach_m, bone_guard_m=bone_guard_m,
        nonzero_vertices=int(np.count_nonzero(np.linalg.norm(field, axis=1) > 0)),
        fit_pose_data_used=False, shared_spatial_field=True,
        rest_edge_field_lipschitz_bound=gradient_bound,
        max_amplitude_for_15_percent_rest_edge_change=.15 / max(gradient_bound, 1e-12),
        skin_boundary_constrained=False, organ_bone_collision_free=False,
        note='A bounded shared corrective candidate; bone taper is not a collision guarantee.')


@dataclass
class VisceralRuntimeV16:
    base: object
    volume: BakedVolumeV16
    field: np.ndarray
    metadata: dict
    max_amplitude: float = .08

    def __post_init__(self):
        from .sparse_lbs_v14 import SparseLBSV14
        if self.field.shape != self.base.target_rest.shape or not np.isfinite(self.field).all():
            raise ValueError('invalid fixed material field')
        self.field = np.array(self.field, dtype=float, copy=True)
        self.field.setflags(write=False)
        self._field_lbs = SparseLBSV14(self.field, self.base.indices, self.base.weights)

    def apply_pose(self, pose55, transl=None, *, return_report=False):
        # Obtain the translated base directly to preserve its exact float32
        # rounding on every unchanged vertex. Volume removes common translation
        # algebraically; the material correction is a vector (zero translation).
        before, globals_ = self.base.apply_pose(pose55, transl, return_globals=True)
        transforms = globals_ @ self.base.target_inverse
        posed_volume = self.volume.evaluate(transforms)
        ratio = posed_volume / self.volume.rest_volume
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError('base organ volume inverted or collapsed')
        requested = (0.0 if abs(ratio - 1) < 1e-7
                     else float(np.cbrt(1 / ratio) - 1))
        amplitude = float(np.clip(requested, -self.max_amplitude, self.max_amplitude))
        rotations = transforms.copy()
        rotations[:, :3, 3] = 0
        correction = amplitude * self._field_lbs(rotations)
        after = np.asarray(before, dtype=float) + correction
        result = after.astype(np.float32)
        report = dict(base_volume_ratio=ratio, requested_amplitude=requested,
            applied_amplitude=amplitude, response_saturated=abs(requested) > self.max_amplitude,
            max_displacement_mm=float(np.linalg.norm(correction, axis=1).max() * 1000),
            anatomical_passed=False)
        return (result, report) if return_report else result

    def save(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        self.base.save(root / 'base')
        np.savez_compressed(root / 'field.npz', field=self.field,
            controller_ids=self.volume.controller_ids, origin=self.volume.origin,
            triples=self.volume.triples, coefficients=self.volume.coefficients,
            rest_volume=self.volume.rest_volume)
        manifest = dict(artifact_kind='ExperimentalVisceralRuntimeV16',
            anatomical_passed=False, publishable=False,
            base_manifest_sha256=_sha(root / 'base/manifest.json'),
            field_sha256=_sha(root / 'field.npz'), max_amplitude=self.max_amplitude,
            metadata=self.metadata, runtime_nearest_search=False,
            runtime_optimization=False, original_weights_preserved=True)
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_visceral_runtime_v16(root):
    from .consistent_runtime_v14 import load_compiled_subject
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('artifact_kind') != 'ExperimentalVisceralRuntimeV16':
        raise ValueError('wrong experimental package kind')
    if (_sha(root / 'base/manifest.json') != manifest['base_manifest_sha256'] or
            _sha(root / 'field.npz') != manifest['field_sha256']):
        raise ValueError('visceral package digest mismatch')
    base = load_compiled_subject(root / 'base')
    with np.load(root / 'field.npz', allow_pickle=False) as d:
        volume = BakedVolumeV16(d['controller_ids'], d['origin'], d['triples'],
                                d['coefficients'], float(d['rest_volume']))
        return VisceralRuntimeV16(base, volume, d['field'], manifest['metadata'], manifest['max_amplitude'])
