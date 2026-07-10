#!/usr/bin/env python3
"""Download publicly hosted U-HMR auxiliary SMPL files into smpl_assets_dir.

Fetches:
  - smpl_mean_params.npz from the SPIN project data bundle (UPenn visiondata mirror)
  - SMPL_to_J19.pkl from a Hugging Face mirror (4D-Humans community snapshot)

Does NOT download SMPL_NEUTRAL.pkl: that file is distributed only under the SMPL
license from https://smpl.is.tue.mpg.de/ (manual registration + download).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


SPIN_DATA_TAR_URL = "http://visiondata.cis.upenn.edu/spin/data.tar.gz"
SMPL_TO_J19_URL = (
    "https://huggingface.co/camenduru/4D-Humans/resolve/"
    "f5f35fe4ce8f4f9e989d2cad0def6a019fb116a2/SMPL_to_J19.pkl"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("dataset/extra/smpl_assets"),
        help="Output directory (repo-relative or absolute).",
    )
    args = parser.parse_args()
    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    mean_out = dest / "smpl_mean_params.npz"
    j19_out = dest / "SMPL_to_J19.pkl"
    neutral = dest / "SMPL_NEUTRAL.pkl"

    if not mean_out.is_file():
        with tempfile.TemporaryDirectory(prefix="spin_data_") as tmp:
            tar_path = Path(tmp) / "data.tar"
            _download(SPIN_DATA_TAR_URL, tar_path)
            with tarfile.open(tar_path, "r") as archive:
                member = "data/smpl_mean_params.npz"
                try:
                    extracted = archive.extractfile(member)
                except KeyError as exc:
                    available = "\n".join(sorted(archive.getnames())[:40])
                    raise FileNotFoundError(
                        f"Missing {member} in SPIN data bundle. First entries:\n{available}"
                    ) from exc
                if extracted is None:
                    raise RuntimeError(f"Could not read {member} from SPIN data bundle.")
                mean_out.write_bytes(extracted.read())
        print(f"Wrote {mean_out}")
    else:
        print(f"Skip existing {mean_out}")

    if not j19_out.is_file():
        _download(SMPL_TO_J19_URL, j19_out)
        print(f"Wrote {j19_out}")
    else:
        print(f"Skip existing {j19_out}")

    if not neutral.is_file():
        print(
            "\nSMPL_NEUTRAL.pkl is still missing.\n"
            "Download SMPL (neutral) from https://smpl.is.tue.mpg.de/ after accepting the license,\n"
            "then copy SMPL_NEUTRAL.pkl into:\n"
            f"  {dest}\n"
        )
        return 1

    print(f"OK: all three files present under {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
