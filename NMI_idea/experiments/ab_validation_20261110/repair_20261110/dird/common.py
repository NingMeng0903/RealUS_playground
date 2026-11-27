from pathlib import Path
import sys
import json

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parents[1]
sys.path.insert(0, str(BASELINE / 'dird'))
from adapter import CapabilityBridge, clean_json, sha256, SOURCE, ROBOT


def save(name, x):
    (HERE / name).write_text(json.dumps(clean_json(x), indent=2, allow_nan=False))
