"""Fresh native replay of an archived near-identical-pose contrast."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

here = Path(__file__).resolve().parent
source = here.parent / "ab_continuing_rollout_20261111"
archive = source / "study_v4_closed_loop_20261111/B_rollout/fixture_00/cases_0000_0001/dev/fixture_00_constrained/native_shared/planning/checkpoints"
sys.path.insert(0, str(source))
from native_adapter import NativeAdapter, _state_fingerprint, _state_from_payload

sources = {
    4: archive / "acbc475b5405d0156af7c8f0.json",
    5: archive / "3e2a64178fc712f3eaf2d325.json",
}
prefix = int(sys.argv[1])
out = here / "late_native_witness" / f"after_{prefix}"
out.mkdir(parents=True, exist_ok=True)
data = {n: json.loads(path.read_text()) for n, path in sources.items()}
states = {n: _state_from_payload(row["input_state"]) for n, row in data.items()}
s4, s5 = states[4], states[5]
s, old = states[prefix], data[prefix]
result = {
    "source": str(sources[prefix]), "prefix": prefix,
    "action": old["action_id"], "entry_fingerprint": _state_fingerprint(s),
    "tcp_delta_mm": float(1000 * np.linalg.norm(s4.pose[:3, 3] - s5.pose[:3, 3])),
    "tcp_delta_deg": float(np.rad2deg(Rotation.from_matrix(s4.pose[:3, :3].T @ s5.pose[:3, :3]).magnitude())),
    "rail_delta_mm": float(1000 * abs(s4.native.q[0] - s5.native.q[0])),
    "max_joint_delta_deg": float(np.rad2deg(np.max(np.abs(s4.native.q[1:] - s5.native.q[1:])))),
}
with NativeAdapter(0, "constrained", out, reuse_checked=False) as adapter:
    assert old["input_fingerprint"] == _state_fingerprint(s)
    assert old["adapter_fingerprint"]["sha256"] == adapter._adapter_fingerprint()["sha256"]
    try:
        event = adapter.execute(s, old["action_id"])
    except RuntimeError as exc:
        result.update(status="exception", exception=str(exc))
    else:
        result.update(status=event.result["reason"], passed=event.feasible,
                      selected=event.route_id, output_fingerprint=_state_fingerprint(event.output))
    result["attempts"] = [{
        "route": row["candidate"], "valid": row["valid"],
        "position_max_mm": max(
            (1000 * step["same_tick_position_error_max_m"] for step in row["steps"]
             if step["same_tick_position_error_max_m"] is not None), default=None),
        "records": [step["record"] for step in row["steps"]],
    } for row in adapter._execution.rows]
    adapter.finish()
(out / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(prefix, result["status"], result["attempts"])
