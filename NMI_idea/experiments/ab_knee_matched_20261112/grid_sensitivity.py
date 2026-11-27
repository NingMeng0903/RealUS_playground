"""Resolution-only replay on the same held-knee observations."""
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import separation
from verify_separation import ARMS, FIELDS, NAMES, Z90, run_errors
import belief


ROOT = Path(__file__).resolve().parent / "separation"


def public_signature(row):
    return json.dumps((row["prefix"]["sha256"], row["schedule"], [
        row["trace"]["arms"][arm]["frames"] for arm in ARMS
    ]), sort_keys=True)


def replay(path):
    old = json.loads(path.read_text())
    new = separation._case_record(old["case_index"], WORLD, old["case_seed"])
    if public_signature(old) != public_signature(new):
        raise ValueError(f"grid comparison changed observations for case {old['case_index']}")
    return new


def metrics(rows):
    result = {}
    for arm in ARMS:
        for name in ("latent", "latent_no_post_stop_US"):
            runs = [r["arms"][arm][name] for r in rows]
            summary = dict(zip(NAMES, np.sqrt(np.mean([run_errors(r) for r in runs], axis=0)).tolist()))
            for field, sd, short in zip(FIELDS, ("w_sd_mm", "z_common_sd_mm", "z_start_smoothed_sd_mm"),
                                        ("w", "z_end", "z_stop")):
                summary[f"{short}_90_coverage"] = float(np.mean([abs(r[field]) <= Z90 * r[sd] for r in runs]))
            result[f"{arm}/{name}"] = summary
    return result


def main():
    global WORLD
    paths = sorted((ROOT / "raw").glob("case_*.json"))
    old = [json.loads(path.read_text()) for path in paths]
    assert [r["case_index"] for r in old] == list(range(384, 512))
    original = np.asarray(belief.W_AXIS)
    WORLD = separation.KneeWorld(noisy_pose=True)
    belief.W_AXIS = np.linspace(original[0], original[-1], 25)
    with get_context("fork").Pool(8) as pool:
        rows = pool.map(replay, paths, chunksize=4)
    output = {
        "cases": len(rows),
        "identical_public_observations": True,
        "scope": "Post hoc inference-resolution sensitivity on the same cases and exact captures. Other axes and parameters fixed.",
        "coarse_step_mm": float(original[1]-original[0]),
        "fine_step_mm": float(belief.W_AXIS[1]-belief.W_AXIS[0]),
        "coarse": metrics(old), "fine": metrics(rows),
    }
    (ROOT / "GRID_SENSITIVITY.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for arm in ("burst_one_site/latent_no_post_stop_US", "separated_one_site/latent", "separated_two_response/latent"):
        print(arm, output["coarse"][arm], output["fine"][arm], flush=True)


if __name__ == "__main__":
    main()
