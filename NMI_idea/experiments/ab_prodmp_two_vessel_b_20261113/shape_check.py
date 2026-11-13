"""Reproduce the matched-endpoint learned versus endpoint-only motion audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from audit import HERE, INHERITED, read_json, shape_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--robot-repo', required=True, type=Path)
    parser.add_argument('--results', type=Path, default=HERE / 'verification')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    repo, robot_repo = args.repo.resolve(), args.robot_repo.resolve()
    geometry = read_json(repo / INHERITED / 'GEOMETRY.json')
    source = args.results / 'MOTION.json'
    if not source.exists():
        source = source.with_suffix('.json.xz')
    # Reuse the original generator without modifying the published results.
    with TemporaryDirectory() as temporary:
        working = Path(temporary)
        (working / source.name).symlink_to(source.resolve(strict=True))
        result = shape_check(repo, robot_repo, working, geometry)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
