"""Persist an explicitly calibrated driver response beside the frozen rig.

This artifact contains response declarations, rigid reference frames and
already baked coupling matrices. Loading and evaluating it never rebuilds
coupling, fits a response, or modifies geometry, topology or skin weights.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import copy
from pathlib import Path

import numpy as np


SCHEMA = 'baked_source_motion_response_v14'
ARRAY_FIELDS = ('source_bone_smplx_a', 'source_bone_smplx_b',
                'source_bone_frame_joints', 'source_bone_blend',
                'source_driver_coupling', 'target_rest_global',
                'target_rest_local', 'target_inverse_bind')
ALLOWED_FIELDS = set(ARRAY_FIELDS) | {'source_bone_driver_types', 'metadata'}


def _signature(asset):
    h = hashlib.sha256()
    for name in ARRAY_FIELDS + ('source_bone_parents',):
        value = getattr(asset, name)
        a = np.ascontiguousarray(value)
        h.update(name.encode()); h.update(a.dtype.str.encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    h.update(json.dumps(list(asset.source_bone_names)).encode())
    h.update(json.dumps(list(asset.source_bone_driver_types)).encode())
    return h.hexdigest()


def _content_signature(asset):
    h = hashlib.sha256()
    for name in ('vertices_rest', 'faces', 'driver_indices', 'driver_weights'):
        a = np.ascontiguousarray(getattr(asset, name))
        h.update(name.encode()); h.update(a.dtype.str.encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def frozen_asset_copy_v14(asset):
    """Detach runtime buffers from caller-owned source arrays and lock them."""
    result = copy.deepcopy(asset)
    for value in vars(result).values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return result


def _rigid(array, name):
    a = np.asarray(array)
    if a.shape != (235, 4, 4) or not np.isfinite(a).all():
        raise ValueError(f'{name} must contain 235 finite rigid frames')
    r = a[:, :3, :3]
    if (not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=2e-6, rtol=0)
            or not np.allclose(np.linalg.det(r), 1, atol=2e-6, rtol=0)):
        raise ValueError(f'{name} contains scale, shear or reflection')


@dataclass
class BakedMotionResponseV14:
    source_signature: str
    arrays: dict
    driver_types: list
    metadata_updates: dict
    provenance: dict
    source_content_signature: str | None = None

    @classmethod
    def from_assets(cls, original, effective, *, provenance):
        # Fail if a response builder also edited a mesh, original weights,
        # hierarchy, driver axes or an unrelated runtime field.
        for field in fields(original):
            if field.name in ALLOWED_FIELDS:
                continue
            before, after = getattr(original, field.name), getattr(effective, field.name)
            if before is after:
                continue
            equal = (np.array_equal(before, after) if isinstance(before, np.ndarray)
                     or isinstance(after, np.ndarray) else before == after)
            if not equal:
                raise ValueError(f'motion calibration changed forbidden field {field.name}')
        before_meta, after_meta = original.metadata or {}, effective.metadata or {}
        for key, value in before_meta.items():
            if key not in after_meta or after_meta[key] != value:
                raise ValueError(f'motion calibration changed original metadata {key}')
        updates = {key: value for key, value in after_meta.items() if key not in before_meta}
        response = cls(_signature(original),
                       {name: np.array(getattr(effective, name), copy=True) for name in ARRAY_FIELDS},
                       list(effective.source_bone_driver_types), updates, dict(provenance),
                       source_content_signature=_content_signature(original))
        response.apply(original)
        return response

    def _validate_arrays(self):
        if set(self.arrays) != set(ARRAY_FIELDS):
            raise ValueError('invalid baked response array inventory')
        expected_shapes = dict(source_bone_smplx_a=(235,), source_bone_smplx_b=(235,),
                               source_bone_frame_joints=(235, 3), source_bone_blend=(235,))
        for name, shape in expected_shapes.items():
            value = np.asarray(self.arrays[name])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'invalid response {name}')
            if name != 'source_bone_blend' and value.dtype.kind not in 'iu':
                raise ValueError(f'response joint identifiers must be integers: {name}')
        for name in ('source_bone_smplx_a', 'source_bone_smplx_b', 'source_bone_frame_joints'):
            if np.any(self.arrays[name] < -1) or np.any(self.arrays[name] >= 55):
                raise ValueError(f'response joint identifier out of range: {name}')
        if len(self.driver_types) != 235 or any(not isinstance(t, str) for t in self.driver_types):
            raise ValueError('invalid response driver types')
        if np.any(self.arrays['source_bone_blend'] < 0) or np.any(self.arrays['source_bone_blend'] > 1):
            raise ValueError('response blend must be within [0, 1]')
        if not np.array_equal(self.arrays['source_bone_frame_joints'][:, 0], self.arrays['source_bone_smplx_a']):
            raise ValueError('response frame primary joint disagrees with driver a')
        _rigid(self.arrays['source_driver_coupling'], 'driver coupling')
        _rigid(self.arrays['target_rest_global'], 'motion reference bind')
        _rigid(self.arrays['target_rest_local'], 'motion reference local bind')
        _rigid(self.arrays['target_inverse_bind'], 'motion inverse bind')

    def apply(self, original, *, source_authenticated=False):
        if _signature(original) != self.source_signature:
            raise ValueError('baked response belongs to a different source motion reference')
        if self.source_content_signature is None:
            # Early packages authenticated geometry/weights in the containing
            # compiled manifest, rather than in this response file itself.
            if not source_authenticated:
                raise ValueError('legacy response requires its authenticated compiled source package')
        elif _content_signature(original) != self.source_content_signature:
            raise ValueError('baked response source geometry or original weights changed')
        self._validate_arrays()
        if any(t not in set(original.source_bone_driver_types) | {'joint_local'} for t in self.driver_types):
            raise ValueError('invalid response driver types')
        global_bind = self.arrays['target_rest_global']
        local_bind = global_bind.copy()
        for bone, parent in enumerate(original.source_bone_parents):
            if parent >= 0:
                local_bind[bone] = np.linalg.inv(global_bind[parent]) @ global_bind[bone]
        if (not np.allclose(local_bind, self.arrays['target_rest_local'], atol=2e-6, rtol=0)
                or not np.allclose(np.linalg.inv(global_bind), self.arrays['target_inverse_bind'], atol=2e-6, rtol=0)):
            raise ValueError('calibrated global, local and inverse bind disagree')
        if set(self.metadata_updates) & set(original.metadata or {}):
            raise ValueError('response cannot replace original runtime metadata')
        if set(self.metadata_updates) - {'source_collar_response_v14', 'source_forearm_response_v14'}:
            raise ValueError('response contains unrecognized runtime metadata')
        metadata = dict(original.metadata or {}); metadata.update(self.metadata_updates)
        return frozen_asset_copy_v14(replace(original, **self.arrays,
                       source_bone_driver_types=list(self.driver_types), metadata=metadata))

    def save(self, path):
        payload = dict(schema=SCHEMA, source_signature=self.source_signature,
                       driver_types=self.driver_types, metadata_updates=self.metadata_updates,
                       provenance=self.provenance, source_content_signature=self.source_content_signature)
        np.savez_compressed(path, **self.arrays,
                            metadata_json=np.asarray(json.dumps(payload, allow_nan=False)))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            if set(data.files) != set(ARRAY_FIELDS) | {'metadata_json'}:
                raise ValueError('invalid baked response file inventory')
            payload = json.loads(str(data['metadata_json']))
            if payload.pop('schema') != SCHEMA:
                raise ValueError('unsupported baked response schema')
            result = cls(arrays={name: data[name].copy() for name in ARRAY_FIELDS}, **payload)
            result._validate_arrays()
            return result
