"""Build a speed-derated graph from two already checked motion clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build(fast, slow, max_rate):
    inputs = [json.loads(path.read_text()) for path in (fast, slow)]
    nodes = [[(node["id"], node["site"], node["view"], node["q"]) for node in bank["nodes"]]
             for bank in inputs]
    if nodes[0] != nodes[1]:
        raise ValueError("retiming requires identical checked endpoint configurations")
    available = {}
    for bank in inputs:
        for edge in bank["edges"]:
            if edge["max_joint_rate_fraction"] <= max_rate + 1.e-10:
                pair = edge["start"], edge["goal"]
                if pair not in available or edge["duration_s"] < available[pair]["duration_s"]:
                    available[pair] = edge
    main = inputs[0]
    durations = sorted({e["duration_s"] for e in available.values()})
    main["edges"] = list(available.values())
    main["status"] = "shortest_checked_profile_under_derating"
    main["summary"] = dict(main["summary"], edge_count=len(available),
                           edge_duration_s=durations[0] if len(durations) == 1 else None,
                           all_retained_edges_uniform_duration=int(len(durations) == 1),
                           edge_duration_options_s=durations, rate_derating=max_rate)
    main["retiming"] = {"max_joint_rate_fraction": max_rate,
                        "parent_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (fast, slow)}}
    return main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fast", type=Path)
    parser.add_argument("slow", type=Path)
    parser.add_argument("--rate", type=float, default=.30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.fast, args.slow, args.rate)
    args.out.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
