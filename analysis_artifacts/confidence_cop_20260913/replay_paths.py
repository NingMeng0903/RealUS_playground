"""Run explicit disjoint paths through the unchanged, hash-pinned replay code."""
import argparse
import json
import os
from pathlib import Path

import replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=int,required=True)
    parser.add_argument('paths',nargs='+')
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    root=Path(__file__).parent
    hashes=replay.source_hashes()
    for identity in args.paths:
        output=root/'results'/identity
        path=output/'summary.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old.get('replay_sha256')==replay.REPLAY_SHA256 and old.get('source_sha256')==hashes:
                continue
        replay.replay_scan(root/'cache'/identity,output,hashes)


if __name__=='__main__':
    main()
