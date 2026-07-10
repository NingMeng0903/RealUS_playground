from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    HumanMotionSequence,
    load_amass_sequence,
)
from projects.genesis_ue_sync.sim_platform.human_motion.proxy import LocalCapsuleProxyProvider, PhcProxyProvider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare or inspect a physical human proxy for a motion sequence.",
        epilog=(
            "Examples (repo root, PYTHONPATH=src):\n"
            "  AMASS raw npz:\n"
            "    python -m projects.genesis_ue_sync.cli.pipeline.human_motion.prepare_proxy \\\n"
            "      --sequence-npz dataset/raw/humans/amass_hf/raw/CMU/114/114_11_poses.npz \\\n"
            "      --input-format amass --provider phc\n"
            "  HumanMotionSequence.save npz (has metadata_json):\n"
            "    python -m ... --sequence-npz outputs/foo_sequence.npz --input-format amongus\n"
            "  PHC with beta-shaped MJCF (SMPLSim SMPL_Robot; needs SMPL model dir + torch/lxml/mujoco):\n"
            "    AMONGUS_PHC_MJCF_SOURCE=smpl_robot python -m ... --provider phc --sequence-npz <npz>\n"
            "    Optional: AMONGUS_SMPLSIM_ROOT (default ref_code_library/SMPLSim).\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sequence-npz",
        type=Path,
        required=True,
        help="Motion .npz (AMASS-style or HumanMotionSequence.save). Relative paths are under the repo root.",
    )
    p.add_argument(
        "--input-format",
        choices=("auto", "amass", "amongus"),
        default="auto",
        help="auto: detect from keys; amass: poses/trans/betas; amongus: HumanMotionSequence.save format.",
    )
    p.add_argument("--provider", choices=("local_capsule", "phc"), default="local_capsule")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--force-rewrite", action="store_true")
    return p.parse_args()


def _load_sequence(path: Path, fmt: str) -> HumanMotionSequence:
    if fmt == "amongus":
        return HumanMotionSequence.load(path)
    if fmt == "amass":
        return load_amass_sequence(path)
    with np.load(path, allow_pickle=True) as z:
        files = set(z.files)
        if "metadata_json" in files and "source_dataset" in files and "poses" in files:
            return HumanMotionSequence.load(path)
        if "poses" in files and "trans" in files and "betas" in files:
            return load_amass_sequence(path)
    raise ValueError(
        f"Cannot detect sequence format in {path} (keys include {sorted(files)[:20]}...). "
        "Pass --input-format amass or amongus."
    )


def main() -> None:
    args = parse_args()
    repo = project_paths(__file__).root
    npz_path = Path(args.sequence_npz)
    if not npz_path.is_absolute():
        npz_path = (repo / npz_path).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Sequence npz not found: {npz_path}\n"
            "Use a real path, e.g. dataset/raw/humans/amass_hf/raw/CMU/114/114_11_poses.npz (AMASS), "
            "or an AmongUs HumanMotionSequence .npz from a prior export."
        )
    seq = _load_sequence(npz_path, str(args.input_format))
    if args.provider == "phc":
        provider = PhcProxyProvider(force_rewrite=bool(args.force_rewrite))
        payload = provider.prepare(seq).to_json_dict()
    else:
        provider = LocalCapsuleProxyProvider(force_rewrite=bool(args.force_rewrite))
        payload = provider.prepare(seq).to_json_dict()
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
