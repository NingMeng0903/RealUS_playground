"""HDMI ultrasound frame grabber: crop + ZMQ CameraFrame publish."""

# The common clock is a workspace protocol shared across the separate envs.
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[3]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
