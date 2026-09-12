"""Bounded asynchronous study records and optional latest feature reception.

Neither component attaches to a robot or changes a controller command. File
serialization and feature transport run outside the control callback.
"""
from __future__ import annotations
import json
import math
import os
import socket
from pathlib import Path
from queue import Queue, Full, Empty
from threading import Event, Thread
import time
import uuid
import hashlib
import subprocess
import atexit
from threading import Lock
from dataclasses import asdict,is_dataclass
from collections.abc import Mapping
import numpy as np

_OPEN_SINKS=set()
_SINK_LOCK=Lock()


def drain_study_records():
    """Wait only from final resource cleanup or interpreter shutdown."""
    with _SINK_LOCK:sinks=tuple(_OPEN_SINKS)
    for sink in sinks:sink.close(wait=True)


atexit.register(drain_study_records)


def json_value(value):
    if is_dataclass(value):return json_value(asdict(value))
    if isinstance(value,np.ndarray):return json_value(value.tolist())
    if isinstance(value,np.generic):return json_value(value.item())
    if isinstance(value,float):return value if math.isfinite(value) else None
    if isinstance(value,Mapping):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    if value is None or isinstance(value,(str,int,bool)):return value
    return str(value)


def study_fingerprints(baseline,config):
    """Resolve dirty source and effective configuration once, before ticks."""
    root=Path(__file__).resolve().parents[2]
    sources={}
    for relative in ('realman8dof/modes/contact_qp.py','realman8dof/modes/contact_recording.py',
            'realman8dof/modes/contact_active.py','contact_qp/runtime_source.py','contact_qp/runtime_config.py',
            'contact_qp/execution.py','contact_qp/rocking_smoothing.py','contact_qp/repair_policy.py',
            'contact_qp/command_budget.py','contact_qp/qp.py','contact_qp/port_constraint.py','realman8dof/force/legacy.py',
            'realman8dof/force/torque_tilt.py','realman8dof/force/nominal_transaction.py','configs/force.yaml'):
        path=root/relative
        if path.is_file():sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    shared_controller=root.parent/'rm75_control/rm75_control/control/admittance_common/controller.py'
    sources['rm75_control/admittance_common/controller.py']=hashlib.sha256(shared_controller.read_bytes()).hexdigest()
    for relative in ('admittance_common/observer.py','admittance_common/variable_step_filter.py',
                     'joint_admittance_8dof/loop.py','joint_admittance_8dof/wbc_rt/client.py',
                     'joint_admittance_8dof/wbc_rt/protocol.py'):
        path=shared_controller.parent.parent/relative
        if path.is_file():sources['rm75_control/'+relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    controller=getattr(baseline,'controller',None)
    effective=json_value(dict(force=getattr(controller,'cfg',None),study=config))
    encoded=json.dumps(effective,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    try:
        baseline_head=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,
                                     check=True,timeout=2.).stdout.strip()
    except (OSError,subprocess.SubprocessError):baseline_head=None
    return dict(git_baseline=baseline_head,source_sha256=sources,effective_configuration=effective,
                configuration_sha256=hashlib.sha256(encoded).hexdigest())


def clock_metadata():
    namespace=os.environ.get('REALUS_CLOCK_NAMESPACE','default')
    result=dict(host=socket.gethostname(),clock_domain='host_monotonic',clock_namespace=namespace,
                clock_id=None,shared_clock=None,clock_metadata_status='unavailable')
    try:result['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:result['boot_id']=None
    path=Path(os.environ.get('REALUS_CLOCK_DIR','/dev/shm'))/f'realus_clock_{os.getuid()}_{namespace}_v1.json'
    try:
        shared=json.loads(path.read_text())
        if shared.get('boot_id')==result['boot_id']:
            result.update(clock_id=shared.get('clock_id'),shared_clock=shared,clock_metadata_status='matched_current_boot')
    except (OSError,ValueError):pass
    return result


class ContactRecordSink:
    """Snapshot before enqueue; bounded queue loss is explicit in every record."""
    def __init__(self,path,*,capacity=8192):
        self.path=Path(path).expanduser().resolve()
        self.session_id=uuid.uuid4().hex
        self.queue=Queue(maxsize=int(capacity))
        self.dropped=0;self.error=None;self.closed=False
        self._finish=Event()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        # Fail synchronously if the destination cannot be created.
        self._file=self.path.open('x',encoding='utf-8')
        with _SINK_LOCK:_OPEN_SINKS.add(self)
        self._thread=Thread(target=self._write,name='contact-study-record',daemon=True)
        self._thread.start()

    def emit(self,event,**fields):
        if self.closed:return False
        record=json_value(dict(schema='contact_study_v1',session_id=self.session_id,
             event=event,record_monotonic_s=time.monotonic(),dropped_records=self.dropped,**fields))
        try:self.queue.put_nowait(record)
        except Full:self.dropped+=1;return False
        return True

    def _write(self):
        flush=time.monotonic()
        try:
            while not self._finish.is_set() or not self.queue.empty():
                try:record=self.queue.get(timeout=.05)
                except Empty:continue
                self._file.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
                if time.monotonic()-flush>=.2:
                    self._file.flush();flush=time.monotonic()
            self._file.write(json.dumps(dict(schema='contact_study_v1',session_id=self.session_id,
                event='recording_close',record_monotonic_s=time.monotonic(),
                dropped_records=self.dropped,writer_error=self.error),allow_nan=False)+'\n')
            self._file.flush()
        except Exception as exc:
            self.error=str(exc)
        finally:
            self._file.close()
            with _SINK_LOCK:_OPEN_SINKS.discard(self)

    def close(self,*,wait=True):
        self.closed=True;self._finish.set()
        if wait:
            self._thread.join(timeout=2.)
            if self._thread.is_alive():self.error='writer_did_not_finish_before_close_timeout'


class FeatureReceiver:
    """Latest immutable observation; no image processing in the controller."""
    def __init__(self,endpoint,*,topic='contact_qp_features_v1'):
        self.endpoint=str(endpoint);self.topic=str(topic)
        self.observation=None;self.error=None;self.received_count=0
        self._finish=Event()
        self._thread=Thread(target=self._receive,name='contact-study-features',daemon=True)
        self._thread.start()

    def _receive(self):
        from peirastic.realman8dof.force.contact_observer import ConfidenceSubscriber
        subscriber=None
        try:
            if self.topic!='contact_qp_features_v1':raise ValueError('unsupported feature topic')
            subscriber=ConfidenceSubscriber(self.endpoint)
            while not self._finish.is_set():
                observation=subscriber.snapshot()
                if observation is not self.observation:
                    self.observation=observation;self.received_count+=1
                self.error=subscriber.last_error or None
                self._finish.wait(.01)
        except Exception as exc:self.error=str(exc)
        finally:
            if subscriber is not None:subscriber.close()

    def close(self,*,wait=True):
        self._finish.set()
        if wait:self._thread.join(timeout=.5)
