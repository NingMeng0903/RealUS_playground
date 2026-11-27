"""Verify completeness, phase, actual-state linkage and provenance of a merged catalog."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROBES = ("side_left", "side_right", "fork_minus", "fork_plus", "fork_plus_shift3")
WINDOWS = ("left", "right")


def audit(path):
    root = path.parent.parent.parent
    data = json.loads(path.read_text())
    by_prefix = {}
    accepted = 0
    attempts = []
    for key, entry in data["transitions"].items():
        p = tuple(entry["sequence"])
        name = entry["capture"]
        assert key == f"{'>'.join(p) if p else 'START'}|capture={name}"
        by_prefix.setdefault(p, {})[name] = entry
        accepted += int(entry["valid"])
        assert entry["check_metrics"]["all_linkage_pass"], key
        origin = Path(entry.get("source_run_root", root))
        for link in (entry["entry_state_link"], entry["exit_state_link"]):
            assert (origin / link).exists(), (key, origin, link)
        for r in entry["dird"]["attempts"]:
            for trace in r["linkage"]["trace_links"]:
                assert (origin / trace["path"]).exists(), trace["path"]
            attempts.append(r)

    def walk(prefix, before, probes, terminals):
        if terminals >= 2:
            return
        expected = WINDOWS + (PROBES if not terminals and probes < 2 else ())
        options = by_prefix[prefix]
        assert set(options) == set(expected), (prefix, sorted(options), sorted(expected))
        for name in expected:
            row = options[name]
            assert row["entry_state_signature"] == before, (prefix, name)
            if not row["valid"]:
                continue
            end = row["exit_state_signature"]
            assert row["actual_elapsed_s"] > 0
            assert row["actual_exit_elapsed_s"] == end["elapsed_s"]
            assert abs(before["elapsed_s"]+row["actual_elapsed_s"]-end["elapsed_s"]) < 1e-9
            assert end["key"] == [*prefix, name]
            walk((*prefix, name), end, probes + int(name in PROBES), terminals + int(name in WINDOWS))

    walk((), by_prefix[()]["left"]["entry_state_signature"], 0, 0)
    good_steps = [s for r in attempts if r["valid"] for s in r["steps"]]
    return {
        "layout": data["layout"], "transitions": len(data["transitions"]), "accepted": accepted,
        "route_attempts": len(attempts), "rejected_routes": sum(not r["valid"] for r in attempts),
        "control_ticks_all": sum(s["control_ticks"] for r in attempts for s in r["steps"]),
        "max_accepted_position_mm": 1000*max(s["same_tick_position_error_max_m"] for s in good_steps),
        "max_accepted_orientation_rad": max(s["same_tick_rotation_error_max_rad"] for s in good_steps),
        "max_accepted_contact_gap_mm": 1000*max(s["synthetic_contact_gap_max_m"] for s in good_steps),
        "max_accepted_pause_count": max(s["paused"] for s in good_steps),
        "min_accepted_jerk_margin": min(s["configured_jerk_margin_min"] for s in good_steps
                                        if s["configured_jerk_margin_min"] is not None),
        "max_reconstruction_error": max(s["reconstruction_error"] for s in good_steps),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.catalogs], indent=2))
