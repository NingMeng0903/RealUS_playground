"""Reproduce fixed-path MPC witness selection and evaluate candidate survival."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np


BASE = Path("/home/camp/Documents/Codex/2026-11-13/ieee-t-ro-b-mpc-b")
ORIGINAL_CODE = BASE / "work/leg_reference_transfer/blender_holdout"
ORIGINAL_DATA = Path("/media/camp/PEI_T7/1_team meeting/atlas_ppt_build/assets/blender_topology_holdout")


def public_model(code, data):
    sys.path.insert(0, str(code))
    import scan

    class PublicStore(scan.Store):
        def __init__(self):
            # The archived pack stores all four sources in one array.
            # Give the model only the three prior sources and the public skin.
            with np.load(data / "geometry.npz", allow_pickle=False) as archive:
                self.centers = archive["centers"][:3].copy()
                self.contact = archive["contacts"]
                self.valid = archive["valid_contact"]
            self.bone = scan.bone_prior()
            self._make_fields()

        def truth(self, *args):
            raise RuntimeError("The public replay has no simulated target query")

    scan.STORE = PublicStore()
    return scan, scan.Model(8, True, True)


def public_replay(scan, model, data):
    """The only stream read here contains anonymous, acquired public packets."""
    rows = {}
    choices = {}
    with gzip.open(data / "test_observations.jsonl.gz", "rt") as source:
        for episode in map(json.loads, source):
            key = episode["episode"]
            reader = scan.Reader(model, scan.TASKS[episode["task"]][2], True)
            for idx, packet in enumerate(episode["packets"]):
                assert packet["index"] == idx
                candidates = np.array(packet["candidates_lateral_depth_mm"]).reshape(-1, 2)
                reader.push(packet["station_index"], packet["view_sector"], candidates,
                            None, packet["signal_quality"], packet["visible_region"] - 1)
            selected = [reader.select(region) for region in range(6)]
            choices[key] = [None if pick is None else [int(pick[0]), int(pick[1])]
                            for pick in selected]
            rows[key] = (episode["task"], episode["condition"], episode["seed"],
                         [(frame[5], frame[6]) for frame in reader.frames])
    return choices, rows


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def score(data, choices, public, output):
    """The evaluator first opens the identity file after every selection is frozen."""
    counts = defaultdict(Counter)
    region_records = []
    scored = set()
    with gzip.open(data / "test_evaluator_labels.jsonl.gz", "rt") as ground_truth:
        for labels in map(json.loads, ground_truth):
            key = labels["episode"]
            task, condition, seed, packets = public[key]
            assert key not in scored
            scored.add(key)
            regions = [[] for _ in range(6)]
            for idx, ((region, probs), annotated) in enumerate(zip(packets, labels["packets"], strict=True)):
                roles = annotated["candidate_roles"]
                assert annotated["index"] == idx and len(roles) == len(probs)
                regions[region].extend(zip(roles, map(float, probs), strict=True))
            for region, (observations, witness) in enumerate(zip(regions, choices[key], strict=True)):
                target_scores = [p for identity, p in observations if identity == 0]
                competitor_scores = [p for identity, p in observations if identity != 0]
                best = max(target_scores, default=0.0)
                best_rival = max(competitor_scores, default=0.0)
                result = None
                if witness is not None:
                    frame, slot = witness
                    assert packets[frame][0] == region
                    result = labels["packets"][frame]["candidate_roles"][slot] == 0
                flags = dict(regions=1, target_seen=int(bool(target_scores)),
                             target_passes_p012=int(best >= .12), target_passes_p055=int(best >= .55),
                             target_top_strict=int(bool(target_scores) and best > best_rival),
                             correct=int(result is True), wrong=int(result is False),
                             abstain=int(result is None))
                row = dict(episode=key, task=task, condition=condition, seed=seed, region=region,
                           max_target_p=best if target_scores else None,
                           max_competitor_p=best_rival if competitor_scores else None, **flags)
                region_records.append(row)
                counts["all"].update(flags)
                counts[task].update(flags)
    assert scored == choices.keys()
    with (output / "mpc_funnel_regions.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(region_records[0]))
        writer.writeheader()
        writer.writerows(region_records)
    write_json(output / "mpc_funnel_by_task.json", {key: dict(value) for key, value in counts.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ORIGINAL_DATA)
    parser.add_argument("--code", type=Path, default=ORIGINAL_CODE)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    parser.add_argument("--public-only", action="store_true")
    opts = parser.parse_args()
    engine, model = public_model(opts.code, opts.data)
    choices, public = public_replay(engine, model, opts.data)
    opts.out.mkdir(exist_ok=True, parents=True)
    write_json(opts.out / "mpc_witnesses.json", choices)
    print(f"{len(choices)} public records, {sum(pick is not None for values in choices.values() for pick in values)} selected", flush=True)
    if not opts.public_only:
        score(opts.data, choices, public, opts.out)


if __name__ == "__main__":
    main()
