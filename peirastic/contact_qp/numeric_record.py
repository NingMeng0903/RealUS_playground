"""Lossless JSON numeric payloads, separate from human-facing null diagnostics."""
from collections.abc import Mapping
import math
import numpy as np

SCHEMA = 'contact_numeric_replay_v1'


def encode(value):
    if isinstance(value, np.ndarray):
        return {'__ndarray__': encode(value.tolist()), 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, np.generic): return encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'__float__': 'nan' if math.isnan(value) else '+inf' if value > 0 else '-inf'}
    if isinstance(value, Mapping): return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise TypeError('unsupported numeric record value: '+type(value).__name__)


def decode(value):
    if isinstance(value, list): return [decode(v) for v in value]
    if isinstance(value, dict):
        if '__float__' in value:
            return {'nan': float('nan'), '+inf': float('inf'), '-inf': -float('inf')}[value['__float__']]
        if '__ndarray__' in value:
            dtype = np.dtype(value['dtype'])
            if dtype.kind not in 'biuf': raise ValueError('numeric dtype required')
            return np.asarray(decode(value['__ndarray__']), dtype=dtype).reshape(value['shape'])
        return {k: decode(v) for k, v in value.items()}
    return value
