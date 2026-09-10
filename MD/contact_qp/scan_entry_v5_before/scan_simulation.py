"""Virtual motion and synthetic multimodal recordings for workflow acceptance.

No robot/camera IPC is attached. Output is explicitly marked simulated.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np

import record
from peirastic.scan_path import ForearmReference, lift_pose
from realus_clock import get_clock


class SimRobot:
    def __init__(self, args):
        self.args = args
        self.time_ns = get_clock().now_ns()
        self.events = []
        self.stages = {}
        self.automatic = False

    def begin_teaching(self):
        self.events.append("pad")

    def idle_tick(self):
        pass

    def check(self):
        pass

    def capture(self, hand, point):
        sign = 1 if hand == "RH" else -1
        pose = np.array([0.22, sign * 0.12, 0.28, np.pi, 0.0, 0.03])
        if point == "P":
            pose += np.array([0.22, 0, 0.008, 0.025, 0.02, 0])
        self.time_ns += 1_000_000_000
        self.events.append(f"teach_{hand}_{point}")
        return dict(pose_m_rad=pose, q_deg=np.zeros(7), rail_m=0.4,
                    wrench_raw_sensor=np.zeros(6), timestamp_ns=self.time_ns,
                    timestamp_mono_ns=self.time_ns-get_clock().offset_ns, seq=len(self.events),
                    clock_id=get_clock().clock_id,
                    header=get_clock().header("world", timestamp_ns=self.time_ns))

    def prepare(self, spec):
        self.automatic = True
        self.events += ["withdraw_before_movej", "movej_standoff"]
        self.spec = spec
        self.time_ns += 1_000_000_000

    def scan(self, spec, recording):
        recording.poll()
        self.events += ["seek", "gate_4n", "scan", "retract", "park"]
        ref = ForearmReference(spec)
        self.stages = dict(seek_begin=self.time_ns,
                           tracking_observed=self.time_ns+500_000_000,
                           retract_begin=self.time_ns+round((0.5+ref.duration_s)*1e9),
                           retract_end=self.time_ns+round((4+ref.duration_s)*1e9))
        self.time_ns = self.stages["retract_end"]
        self.automatic = False
        return dict(self.stages)

    def stop(self):
        self.events.append("stop")
        self.automatic = False

    def close(self):
        self.events.append("close")


class SimRecorder:
    def __init__(self, directory, args, robot):
        self.path = Path(directory) / "raw.h5"
        self.args, self.robot = args, robot
        self.closed = False

    def wait_ready(self, tick):
        tick()
        self.robot.events.append("recorder_ready")

    def poll(self, **kwargs):
        return True

    def close(self, *, require_success=False):
        if self.closed:
            return
        self.closed = True
        if not require_success:
            return
        ref = ForearmReference(self.robot.spec)
        stages = self.robot.stages
        origin = stages["seek_begin"]
        end = (stages["retract_end"]-origin)/1e9
        options = record.parse_args(["--repo", str(self.args.repo), "--raw-only"])
        writer = record.Writer(self.path, record.Clock(), options)
        writer.file.attrs["simulated"] = True
        counts = Counter()
        batches = {g: [] for g in record.RECORDED_GROUPS}
        for i, t in enumerate(np.arange(0, end, 0.01)):
            stamp = origin + round(float(t)*1e9)
            pose = ref.sample(max(0, t-0.5)).pose_d
            scanning = t < 0.5+ref.duration_s
            force = 4.3 if 0.4 <= t < 0.7+ref.duration_s else 0.0
            if not scanning:
                progress = min(1, (t-0.5-ref.duration_s)/3)
                pose = pose + progress * (lift_pose(pose)-pose)
            timing = dict(timestamp_ns=stamp, timestamp_mono_ns=stamp-get_clock().offset_ns)
            rows = dict(
                tcp=dict(timing, pose_m_rad=pose, q_deg=np.zeros(7), rail_m=0.4,
                         wrench_raw_sensor=np.array([0, 0, force, 0, 0, 0]), seq=i+1),
                force=dict(timing, wrench_tcp=np.array([0, 0, force, 0, 0, 0]),
                           contact_force_n=force, force_control_active=scanning, mode_valid=True,
                           mode=4 if scanning else 3, seq=i+1))
            if i % 4 == 0:
                # A valid synthetic 1×1 grayscale JPEG, generated in memory only.
                if i == 0:
                    from io import BytesIO
                    from PIL import Image

                    encoded = BytesIO()
                    Image.new("L", (1, 1)).save(encoded, format="JPEG")
                    jpeg = np.frombuffer(encoded.getvalue(), dtype=np.uint8)
                meta = dict(get_clock().metadata("us_prob", timestamp_ns=stamp), simulated=True)
                rows["ultrasound"] = dict(timing, jpeg=jpeg, width=1, height=1, frame_seq=i//4,
                                           metadata_json=json.dumps(meta))
            if i % 100 == 0:
                rows["clock"] = dict(timing, wall_time_ns=stamp)
            for group, row in rows.items():
                batches[group].append(row)
                counts[group] += 1
            if i % 200 == 199:
                for group, batch in batches.items():
                    if batch:
                        writer.append(group, batch)
                        batch.clear()
        for group, batch in batches.items():
            if batch:
                writer.append(group, batch)
        writer.finish(SimpleNamespace(counts=counts, diagnostics={}), None)
