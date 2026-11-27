"""Read-only telemetry added to an isolated native source copy, never production."""
import json
import re
import numpy as np
from common import HERE, BASELINE, save, sha256
from qp_replay import QPReplayBridge
from rm75_control.control.joint_admittance_8dof.wbc_rt import client


def main():
    binary = HERE / 'native_rail_diagnostic/build/wbc_rt'
    header = (HERE / 'native_rail_diagnostic/build/wbc_build_id.hpp').read_text()
    expected = re.search('WBC_SRC_HASH "([a-f0-9]+)"',header).group(1)
    def validate_instrumented_hash(embedded):
        if embedded != expected:
            raise RuntimeError('isolated instrumented binary does not match its own frozen source hash')
    client.assert_native_matches_tree = validate_instrumented_hash
    bridge = QPReplayBridge('native','urdf')
    bridge.raw['inner']['native_bin'] = str(binary)
    seed = json.load(open('initialization_sensitivity_seed.json'))['q_seed']
    path = np.load(BASELINE / 'dird/trajectory_paths.npz')['no_dird_T_tcp_railbase'][1]
    result = bridge.query_pose_path(path,seed,return_trace=True,return_state=False,
        provenance='isolated instrumented same native source copy; only read-only print additions')
    save('instrumented_native_result.json', {'production_binary_overwritten':False,
        'instrumented_binary':str(binary),'binary_sha256':sha256(binary),
        'instrumented_tree_embedded_hash':expected,
        'source':str(HERE/'native_rail_diagnostic/source/native/wbc_rt/src/inner.cpp'),
        'source_sha256':sha256(HERE/'native_rail_diagnostic/source/native/wbc_rt/src/inner.cpp'),
        'instrumentation_scope':'qhat/qprev/observer/command logging only around t=1003.39..1003.43; no arithmetic/control changes',
        'result':result})
    print('steps',result['n_steps'],'failures',[(r.get('tick'),r.get('violations')) for r in result['failures']],flush=True)


if __name__=='__main__':main()
