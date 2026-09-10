#!/usr/bin/env python3
"""Two taught points → RH/LH × L/C/S × DtP/PtD, one Enter per scan.

This entrypoint commands the existing controller. --simulate and --preview
never attach to robot IPC and write outside the patient directory by default.
C/S noisy lateral peaks are randomized within 10-15 mm; L stays straight.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import select
import sys
import tempfile
import time

REPO = Path(os.environ.get("REALUS_PROJECT_ROOT", "/media/camp/EXT_DRIVE/RealUS_playground"))
for path in (REPO, REPO / "rm75_control", REPO / "src"):
    sys.path.insert(0, str(path))

if __name__ == "__main__":
    from rm75_control.control.admittance_common.observer_runtime import prepare_observer_process
    prepare_observer_process()

import numpy as np

from peirastic.scan_path import ForearmReference, SCAN_ORDER, make_spec, force_profile
from realus_clock import get_clock
from scan_contact import contact_policy
from scan_io import (RecorderProcess, allocate_subject, finish_trial, jsonable,
                     preview_paths, save_json, trial_name)

DATA_ROOT = Path("/media/camp/yameng/icra 2027/uncalibrated")


def wait_enter(prompt, tick=lambda: None):
    print(f"[ENTER] {prompt}", flush=True)
    while True:
        tick()
        ready, _, _ = select.select([sys.stdin], [], [], 0.02)
        if ready:
            line = sys.stdin.readline()
            if not line:
                raise KeyboardInterrupt("stdin closed")
            if line.strip().lower() in ("q", "qq", "quit", "exit"):
                raise KeyboardInterrupt("operator exit")
            if not line.strip():
                return
            print("Enter to continue; q to quit.", flush=True)


class Session:
    def __init__(self, args, robot, *, recorder_factory=RecorderProcess, ask=wait_enter):
        self.args, self.robot = args, robot
        self.recorder_factory, self.ask = recorder_factory, ask
        self.directory = allocate_subject(args.data_root)
        self.manifest = dict(schema="icra_scan_session_v1", subject=self.directory.name,
                             status="started", simulated=bool(args.simulate),
                             clock=get_clock().description, hands={}, trials=[],
                             contact_policy=contact_policy(),
                             force_profile=args.force_profile, force_overrides=force_profile(args.force_profile))
        self.active_recorder = None
        self.active_attempt = None
        self.save()

    def save(self):
        save_json(self.directory / "session.json", self.manifest)

    def teach(self, hand):
        self.robot.begin_teaching()
        points = []
        for point, label in (("D", "wrist (D)"), ("P", "forearm near elbow (P)")):
            while True:
                self.ask(f"{hand}: pad L3 hybrid to {label}; hold still, then Enter.", self.robot.idle_tick)
                try:
                    sample = self.robot.capture(hand, point)
                    points.append(sample)
                    print(f"[POINT] {hand} {point} xyz={np.round(sample['pose_m_rad'][:3], 4)}", flush=True)
                    break
                except RuntimeError as exc:
                    print(f"[RETRY POINT] {exc}", flush=True)
        specs = [make_spec(points[0]["pose_m_rad"], points[1]["pose_m_rad"], shape, direction,
                           secrets.randbits(63), speed=self.args.speed_m_s, side=self.args.curve_side)
                 for shape, direction in SCAN_ORDER]
        self.manifest["hands"][hand] = dict(distal=points[0], proximal=points[1], paths=specs)
        self.save()
        preview = preview_paths(self.directory, hand, specs)
        print(f"[PREVIEW] {preview}\n[PLAN] 6 paths; {self.args.speed_m_s*1000:g} mm/s; Enter per path.", flush=True)
        if self.args.open_preview:
            import subprocess
            try:
                subprocess.Popen(["xdg-open", str(preview)], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            except OSError as exc:
                print(f"[PREVIEW] Open preview manually: {exc}", flush=True)
        return specs

    def save_trial(self, raw, destination, metadata, trial, attempt):
        """Retry disk operations without repeating an already finished motion."""
        while True:
            trial["status"] = attempt["status"] = "saving"
            try:
                self.save()
                print(f"[SAVE] {destination.name}", flush=True)
                return finish_trial(raw, destination, metadata, keep_raw=self.args.keep_raw)
            except OSError as exc:
                # Publication may have succeeded before a directory fsync
                # error. Never overwrite that file or repeat its robot scan.
                if destination.exists():
                    raise RuntimeError(f"File exists after save error: {destination}. Stop and verify.") from exc
                trial["status"] = attempt["status"] = "save_pending"
                attempt["save_error"] = str(exc)
                try:
                    self.save()
                except OSError:
                    pass  # Keep the pending raw path in memory if disk is offline.
                print(f"[SAVE FAILED] {exc}\n[RAW] {raw}", flush=True)
                if self.args.simulate:
                    raise
                self.ask("Enter to retry SAVE ONLY; q to quit. No robot motion.", self.robot.idle_tick)

    def run_trial(self, hand, spec):
        name = trial_name(hand, spec)
        destination = self.directory / name
        if destination.exists():
            raise FileExistsError(f"File exists: {destination}")
        trial = dict(name=name, status="pending", path=spec, attempts=[])
        self.manifest["trials"].append(trial)
        while True:
            action = "rescan + record" if trial["attempts"] else "scan + record"
            self.ask(f"{name}: Enter to {action}; q to quit.", self.robot.idle_tick)
            attempt_dir = self.directory / "attempts" / destination.stem / f"{len(trial['attempts'])+1:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            self.active_attempt = attempt_dir
            attempt = dict(directory=str(attempt_dir.relative_to(self.directory)), status="running")
            trial["attempts"].append(attempt)
            trial["status"] = "running"
            self.save()
            recorder = None
            try:
                self.robot.prepare(spec)
                print(f"[RECORD] Waiting for streams.", flush=True)
                recorder = self.recorder_factory(attempt_dir, self.args)
                self.active_recorder = recorder
                recorder.wait_ready(self.robot.check)
                stages = self.robot.scan(spec, recorder)
                recorder.close(require_success=True)
                self.active_recorder = None
                attempt.update(motion_completed=True, stages=stages)
                metadata = dict(hand=hand, path=spec, stages=stages,
                                taught={k: self.manifest["hands"][hand][k] for k in ("distal", "proximal")},
                                force_profile=self.args.force_profile,
                                force_overrides=force_profile(self.args.force_profile),
                                simulated=bool(self.args.simulate))
                episode = self.save_trial(recorder.path, destination, metadata, trial, attempt)
                attempt.update(status="completed", episode=episode)
                attempt.pop("save_error", None)
                trial["status"] = "completed"
                self.save()
                self.active_attempt = None
                print(f"[SAVED] {name}", flush=True)
                return
            except Exception as exc:
                try:
                    self.robot.stop()
                except Exception as stop_error:
                    print(f"[STOP ERROR] {stop_error}", flush=True)
                if recorder is not None:
                    try:
                        recorder.close()
                    except Exception as close_error:
                        print(f"[RECORDER ERROR] {close_error}", flush=True)
                self.active_recorder = None
                if destination.exists():
                    trial["status"] = "saved_metadata_error"
                    raise RuntimeError(f"File saved but metadata failed: {destination}: {exc}") from exc
                attempt.update(status="failed", error=str(exc),
                               stages=dict(getattr(self.robot, "stages", {})))
                trial["status"] = "retry_pending"
                save_json(attempt_dir / "failure.json", attempt)
                self.save()
                self.active_attempt = None
                print(f"[FAILED] {name}: {exc}\n[RETRY] Restore control if needed, then Enter. No auto-reset.", flush=True)
                if self.args.simulate:
                    raise  # An unattended simulation must not loop forever on a bug.

    def run(self):
        print(f"[SESSION] {self.directory}\n[ORDER] RH 6, then LH 6.", flush=True)
        try:
            for hand in ("RH", "LH"):
                if hand == "LH":
                    self.ask("RH done. Switch to LH; Enter for pad teaching.", self.robot.idle_tick)
                while True:
                    try:
                        specs = self.teach(hand)
                        break
                    except ValueError as exc:
                        print(f"[RETEACH] {exc} Teach both points again.", flush=True)
                for index, spec in enumerate(specs, 1):
                    print(f"[NEXT] {hand} {index}/6: {trial_name(hand, spec)}", flush=True)
                    self.run_trial(hand, spec)
            self.manifest["status"] = "completed"
            self.save()
            print(f"[DONE] 12 scans saved: {self.directory}", flush=True)
            return 0
        except KeyboardInterrupt:
            self.manifest["status"] = "interrupted"
            if self.manifest["trials"] and self.manifest["trials"][-1]["status"] == "running":
                self.manifest["trials"][-1]["status"] = "interrupted"
                self.manifest["trials"][-1]["attempts"][-1]["status"] = "interrupted"
            print("[STOP] Stopping. Saved files and raw attempts kept.", flush=True)
            return 130
        except Exception as exc:
            self.manifest.update(status="failed", error=str(exc))
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        finally:
            # Motion stop always precedes waiting on disk/recorder shutdown.
            if self.manifest["status"] != "completed":
                try:
                    self.robot.stop()
                except Exception as exc:
                    print(f"[STOP ERROR] {exc}", file=sys.stderr)
            if self.active_recorder is not None:
                try:
                    self.active_recorder.close()
                except Exception as exc:
                    print(f"[RECORDER ERROR] {exc}", file=sys.stderr)
            if self.active_attempt is not None:
                save_json(self.active_attempt / "failure.json", dict(outcome="interrupted"))
            self.save()
            self.robot.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--recorder-python", type=Path,
                        default=Path(os.environ.get("ICRA_RECORD_PYTHON", "/media/camp/EXT_DRIVE/envs/genesis/bin/python")))
    parser.add_argument("--state-shm", default="rm75_state")
    parser.add_argument("--force-shm", default=None)
    parser.add_argument("--shm-prefix", default="")
    parser.add_argument("--us-endpoint", default="tcp://127.0.0.1:17359")
    parser.add_argument("--force-profile", choices=("icra", "baseline"), default="icra")
    parser.add_argument("--speed-m-s", type=float, default=0.02)
    parser.add_argument("--curve-side", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--simulate", action="store_true", help="virtual robot and synthetic HDF5; no robot IPC")
    parser.add_argument("--auto-enter", action="store_true", help="simulate only: accept all operator prompts")
    parser.add_argument("--preview", type=Path, help="plan only from JSON {distal:[6], proximal:[6]}; no robot IPC")
    parser.add_argument("--open-preview", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    if args.auto_enter and not args.simulate:
        parser.error("--auto-enter is only allowed with --simulate")
    if args.preview and args.simulate:
        parser.error("choose --preview or --simulate")
    if not np.isfinite(args.speed_m_s) or not 0 < args.speed_m_s <= 0.02:
        parser.error("--speed-m-s must be in (0, 0.02]")
    if args.data_root is None:
        args.data_root = (Path(tempfile.mkdtemp(prefix="icra-offline-"))
                          if args.simulate or args.preview else DATA_ROOT)
    if args.open_preview is None:
        args.open_preview = not (args.simulate or args.preview)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.preview:
        points = json.loads(args.preview.read_text())
        specs = [make_spec(points["distal"], points["proximal"], shape, direction, index,
                           speed=args.speed_m_s, side=args.curve_side)
                 for index, (shape, direction) in enumerate(SCAN_ORDER)]
        args.data_root.mkdir(parents=True, exist_ok=True)
        print(preview_paths(args.data_root, "preview", specs))
        return 0
    if args.simulate:
        from scan_simulation import SimRobot, SimRecorder
        robot = SimRobot(args)
        factory = lambda directory, options: SimRecorder(directory, options, robot)
    else:
        from scan_robot import Robot
        robot, factory = Robot(args), RecorderProcess
    ask = (lambda prompt, tick: (tick(), print(f"[SIM ENTER] {prompt}"))) if args.auto_enter else wait_enter
    session = Session(args, robot, recorder_factory=factory, ask=ask)
    return session.run()


if __name__ == "__main__":
    raise SystemExit(main())
