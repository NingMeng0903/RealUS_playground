"""Matched native motion test of three scout dwells at one archived state."""
import json
from pathlib import Path
import sys

from coverage_order import NativeAdapter, native_link
from fixture import single_dwell_action
from native_adapter import _state_fingerprint, _state_from_payload

here = Path(__file__).resolve().parent
source = here.parent / "ab_continuing_rollout_20261111/study_v4_closed_loop_20261111/B_rollout/fixture_00/cases_0000_0001/dev/fixture_00_constrained/native_shared/planning/checkpoints/2a921a2997242aaf7a1d55ea.json"
raw = json.loads(source.read_text())
entry = _state_from_payload(raw["input_state"])
arm = sys.argv[1]
out = here / "dwell_sensitivity" / arm
out.mkdir(parents=True, exist_ok=True)
target = "side_left__single_dwell_0.6"
order = [target] if arm == "direct" else [single_dwell_action("side_right", float(arm)), target]
with NativeAdapter(0, "constrained", out, reuse_checked=False) as adapter:
    assert raw["input_fingerprint"] == _state_fingerprint(entry)
    assert raw["adapter_fingerprint"]["sha256"] == adapter._adapter_fingerprint()["sha256"]
    state = entry
    links = []
    for action in order:
        state, record = native_link(adapter, state, action, execution=True)
        links.append(record)
        if state is None:
            break
    adapter.finish()
result = {
    "source": str(source), "root": _state_fingerprint(entry),
    "arm": arm, "prescribed_target": target, "order": order, "links": links,
    "complete": state is not None,
    "motion_s": None if state is None else state.elapsed - entry.elapsed,
}
(out / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(arm, result["complete"], result["motion_s"])
