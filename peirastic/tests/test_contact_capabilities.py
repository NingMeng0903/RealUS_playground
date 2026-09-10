import json
import time
import uuid
import pytest
from peirastic.core.ipc import CommandClient, CommandHub
from peirastic.core.capabilities import CapabilityAdvertisement, capability_path


def test_capability_rejects_old_or_stale_service_without_command():
    hub=CommandHub(prefix='cq_test_'+uuid.uuid4().hex+'_')
    client=CommandClient(prefix=hub.ctl_name.removesuffix('peirastic_ctl_v2'))
    advertisement=None
    try:
        with pytest.raises(RuntimeError):client.require_capability('contact_qp.recording_v1')
        assert client.snapshot()['cmd_seq']==0
        advertisement=CapabilityAdvertisement(hub,{'contact_qp.recording_v1'})
        client.require_capability('contact_qp.recording_v1')
        with pytest.raises(RuntimeError):client.require_capability('contact_qp.active_v1')
        hub._ctl[0]['t_mono']=time.monotonic()-1
        with pytest.raises(RuntimeError):client.require_capability('contact_qp.recording_v1')
        hub._ctl[0]['t_mono']=time.monotonic()
        path=capability_path(hub.ctl_name)
        payload=json.loads(path.read_text());payload['header_identity']=[0,0]
        path.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError):client.require_capability('contact_qp.recording_v1')
        assert client.snapshot()['cmd_seq']==0
    finally:
        if advertisement:advertisement.close()
        client.close();hub.close()
