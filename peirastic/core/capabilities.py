"""Startup-only Python handler advertisement; independent of native IPC ABI."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid
from realus_clock import header_path


SOURCE_TIMEBASE_CAPABILITY='contact_qp.source_timebase_bilinear_v1'
DIFFERENTIAL_REPAIR_CAPABILITY='contact_qp.differential_repair_v8_r2'
CONFIDENCE_BALANCE_CAPABILITY='contact_qp.confidence_balance_v8_r3'
LOGICAL_COMMAND_BUDGET_CAPABILITY='contact_qp.logical_command_budget_v1'
CONTINUOUS_VISUAL_CAPABILITY='contact_qp.continuous_visual_v1'
NOMINAL_TASK_POWER_CAPABILITY='contact_qp.nominal_task_power_v1'
TRANSIENT_FEEDBACK_CAPABILITY='contact_qp.transient_feedback_grace_v1'
PAUSE_VISUAL_FEEDBACK_CAPABILITY='contact_qp.pause_visual_feedback_v1'


def study_capabilities(config):
    active=config.get('mode','baseline')=='active'
    required=['contact_qp.active_v1' if active else 'contact_qp.recording_v1']
    if active and (config.get('source') or {}).get('timebase')=='variable_step_bilinear_v1':
        required.append(SOURCE_TIMEBASE_CAPABILITY)
    if active and (config.get('qp') or {}).get('allocation_policy')=='differential_repair_v8':
        required.append(DIFFERENTIAL_REPAIR_CAPABILITY)
        if ((config.get('qp') or {}).get('differential_repair') or {}).get('revision')=='v8r3_confidence_balance':
            required.append(CONFIDENCE_BALANCE_CAPABILITY)
        if ((config.get('qp') or {}).get('differential_repair') or {}).get('permission_mode')=='continuous':
            required.append(CONTINUOUS_VISUAL_CAPABILITY)
    if active and config.get('energy_constraint_enabled') is True:
        required.append(LOGICAL_COMMAND_BUDGET_CAPABILITY)
        if (config.get('energy') or {}).get('task_power_source')=='nominal_command':
            required.append(NOMINAL_TASK_POWER_CAPABILITY)
    if active and float((config.get('feature') or {}).get('dropout_grace_s',0.))>0:
        required.append(TRANSIENT_FEEDBACK_CAPABILITY)
    if active and (config.get('feature') or {}).get('dropout_policy')=='pause_visual':
        required.append(PAUSE_VISUAL_FEEDBACK_CAPABILITY)
    return tuple(required)


def _identity(path):
    st = Path(path).stat()
    return [st.st_dev, st.st_ino]


def capability_path(name):
    return header_path(name).with_name(name + '_capabilities_v1.json')


class CapabilityAdvertisement:
    def __init__(self, hub, handlers):
        self.path = capability_path(hub.ctl_name)
        self.instance = uuid.uuid4().hex
        payload = dict(schema=1, instance=self.instance, capabilities=sorted(handlers),
                       ctl_identity=_identity('/dev/shm/' + hub.ctl_name),
                       header_identity=_identity(header_path(hub.ctl_name)))
        fd, tmp = tempfile.mkstemp(prefix=self.path.name + '.', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(payload, stream)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def close(self):
        try:
            if json.loads(self.path.read_text()).get('instance') == self.instance:
                self.path.unlink()
        except (OSError, ValueError):
            pass


def require_capability(client, capability, *, max_age_s=.5):
    """Read only: fail before any request or force override is published."""
    try:
        payload = json.loads(capability_path(client.ctl_name).read_text())
        st = os.fstat(client._ctl_shm._fd)
        stamp = float(client._ctl[0]['t_mono'])
        age = time.monotonic() - stamp
        matching = (payload.get('schema') == 1 and isinstance(payload.get('instance'), str)
                 and bool(payload['instance']) and capability in payload.get('capabilities', [])
                 and payload.get('ctl_identity') == [st.st_dev, st.st_ino]
                 and payload['ctl_identity'] == _identity('/dev/shm/' + client.ctl_name)
                 and payload.get('header_identity') == _identity(header_path(client.ctl_name)))
        if matching and not (math.isfinite(age) and 0 <= age <= max_age_s):
            raise RuntimeError(
                f'Controller unresponsive: heartbeat age={age:.3f}s (limit {max_age_s:.3f}s); '
                f'{capability} is advertised, but no request was sent. '
                'Recover the stopped/unresponsive controller before retrying.')
        valid = matching
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        valid = False
    if not valid:
        raise RuntimeError('Running controller does not advertise ' + capability
                           + '; restart Window A with the updated Python controller before requesting contact_qp')
