"""Recover verified runtime inputs from the existing leg scan replay ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def restore(archive, records):
    manifest = json.loads((ROOT / "replay_inputs.json").read_text())
    expected = dict(manifest["runtime"])
    if records:
        expected.update(manifest["records"])

    with ZipFile(archive) as source:
        members = set(source.namelist())
        missing = sorted(set(expected).difference(members))
        if missing:
            names = "\n".join(missing[:8])
            raise ValueError(f"This replay package does not include {len(missing)} required files.\n{names}")

        cached = {}
        for name, checksum in expected.items():
            path = ROOT / name
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError(f"Replay path leaves the experiment directory: {name}")
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Existing file differs from the published replay input: {name}")
            data = source.read(name)
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError(f"Replay input differs from the recorded checksum: {name}")
            cached[path] = data

        for path, data in cached.items():
            if path.is_file():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                temporary.write(data)
                temp_path = Path(temporary.name)
            temp_path.replace(path)
    print(f"Verified {len(expected)} replay inputs under {ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--with-records", action="store_true",
                        help="Install the archived trajectories for historical scores and audits.")
    options = parser.parse_args()
    restore(options.archive, options.with_records)
