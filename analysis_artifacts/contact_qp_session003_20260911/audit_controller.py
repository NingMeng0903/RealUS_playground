"""Offline audit of the reported attempt; never attaches to a controller."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np

from peirastic.apps.contact_qp_features import process_parts
from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.attempt / "contact_qp.jsonl").open()]
    start = next(r for r in records if r["event"] == "study_start")
    config = start["config"]
    controls = [r for r in records if r["event"] == "control_sample"]
    episodes = [ast.literal_eval(r["repair_episode"]) if isinstance(r["repair_episode"], str)
                else r["repair_episode"] for r in controls]
    extractor = FeatureExtractor(FeatureConfig(**config["feature"]["config"]))
    offline = []
    with h5py.File(args.attempt / "raw.h5", "r") as f:
        frame_count = len(f["ultrasound/jpeg"])
        scan_start = json.loads((args.attempt / "failure.json").read_text())["stages"]["tracking_observed"]
        for index in [150, 300, 500, 700, 870]:
            metadata = f["ultrasound/metadata_json"][index]
            if isinstance(metadata, str):
                metadata = metadata.encode()
            jpeg = f["ultrasound/jpeg"][index].tobytes()
            received = float(f["ultrasound/received_monotonic_ns"][index]) * 1e-9
            obs = process_parts([b"amongus_camera_frame_v1", metadata, jpeg], extractor, received)
            meta = json.loads(metadata)
            offline.append(dict(
                raw_frame_index=index,
                capture_relative_to_scan_s=(int(f["ultrasound/timestamp_ns"][index]) - scan_start) * 1e-9,
                crop_box=meta.get("crop_box"), hflip=meta.get("hflip"),
                observation=obs.to_dict(),
                window_matches=obs.window_version == config["feature"]["window_version"],
                registration_matches=obs.registration_version == config["feature"]["registration_version"],
                note="Offline recomputation, not a feature received by the controller.",
            ))
    last_id = controls[-1]["control_id"]
    last_events = [r for r in records if r.get("control_id") == last_id and r["event"] != "control_sample"]
    last_energy = next(r for r in reversed(records) if r["event"] == "logical_command_energy")
    summary = dict(
        attempt=str(args.attempt), session_id=start["session_id"],
        mode=start["mode"], revision=config["qp"]["differential_repair"]["revision"],
        control_samples=len(controls), raw_ultrasound_frames=frame_count,
        null_features=sum(r["feature"] is None for r in controls),
        image_compatible=sum(bool(r["image_compatible"]) for r in controls),
        repair_reasons=dict(Counter(r["reason"] for r in episodes)),
        repair_allowed=sum(bool(r["repair_allowed"]) for r in episodes),
        max_abs_differential_request_m_s=max(abs(r["allocation_diagnostics"]["differential_request_m_s"]) for r in controls),
        alpha_quantiles=np.quantile([r["alpha"] for r in controls], [0, .5, .95, 1]).tolist(),
        last_reference_s=controls[-1]["reference_s"],
        expected_registration_version=config["feature"]["registration_version"],
        offline_frames=offline,
        last_candidate_source=controls[-1]["source"],
        last_candidate_events=last_events,
        final_energy={k: last_energy[k] for k in ("balance_j", "available_j", "energy_constraint_enabled", "latched_reason")},
        stop=next(r for r in reversed(records) if r["event"] == "stop"),
    )
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
