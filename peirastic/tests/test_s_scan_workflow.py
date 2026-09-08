"""Exercise the three Enter steps with a robot double; never connects to SHM."""

from types import SimpleNamespace

import numpy as np
import pytest

from peirastic.DEMO.phathom_scanning import s_scan
from peirastic.core.ipc import Status
from peirastic.core.modes import Mode


@pytest.mark.parametrize("failure", [None, "ptp", "contact"])
@pytest.mark.parametrize("pattern", ["raster", "lissajous"])
def test_scan_sequence_and_failure_stop(monkeypatch, tmp_path, failure, pattern):
    now = [10.0]
    monkeypatch.setattr(s_scan, "CLOSE_TIMEOUT_S", 0.5)

    def monotonic():
        now[0] += 0.02
        return now[0]

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(s_scan, "time", SimpleNamespace(monotonic=monotonic, sleep=sleep))
    poses = np.array([[0.2, 0.2, 0.168, np.pi, 0, 0],
                      [0.3, 0.2, 0.168, np.pi, 0, 0],
                      [0.3, 0.24, 0.168, np.pi, 0, 0]])
    normal = np.array([0., 0., 1.])
    standoff = poses[0].copy()
    standoff[2] += 0.04
    lift = poses[-1].copy()
    lift[2] += 0.01
    # A curved face can have a reference normal different from the start's
    # local normal. Contact seeking must use the actual start orientation.
    reference_normal = np.array([0.3, 0., np.sqrt(0.91)])
    plan = SimpleNamespace(poses=poses, standoff=standoff, lift=lift, normal=reference_normal,
                           n_rows=3, stride_m=0.04, u_span_m=0.1, v_span_m=0.04,
                           length_m=0.14, pattern=pattern)
    hit = SimpleNamespace(n_points=500, normal=normal, corner=poses[0, :3])
    calls = []
    enters = []

    class Arm:
        def __init__(self):
            self.live = np.array([0.35, 0.19, 0.4, np.pi, 0, 0])
            self.mode = Mode.SERVO_TWIST_HOLD
            self.label = "idle"
            self.last_seq = 0
            self.hybrid_t0 = None

        def _snapshot(self):
            msg, done, force = self.label, 0, 0.0
            if self.hybrid_t0 is not None and self.mode == Mode.TRACK_HYBRID:
                elapsed = now[0] - self.hybrid_t0
                tracking = failure != "contact" and elapsed >= 0.3
                stage = "tracking" if tracking else "approach"
                progress = max(0.0, elapsed - 0.3) if tracking else 0.0
                force = 4.0 if tracking else -1.2
                msg = (f"{self.label}:{stage} t={progress:.2f} "
                       f"contact={int(tracking)} vz_cmd=0.0")
                if tracking:
                    self.live = poses[-1].copy()
                    self.live[2] -= 0.003
                    done = self.last_seq if progress >= self.duration else 0
            return dict(status=Status.RUNNING, mode=self.mode, msg=msg,
                        t_mono=now[0], done_seq=done, f_ext_z=force, dof=8)

        def set_dof(self, dof, **kw):
            calls.append(("dof", dof))
            return 0

        def movej(self, q, **kw):
            calls.append(("movej", list(q)))
            return 0

        def cartesian(self, pose, **kw):
            calls.append(("ptp", list(pose)))
            assert kw["block"] == 1
            if failure == "ptp":
                return 1
            self.live = np.array(pose)
            return 0

        def hfpc(self, path, **kw):
            calls.append((kw["label"], path))
            assert kw["law"] == "tff"
            assert kw["force_axes"] == s_scan.FORCE_AXES
            self.mode, self.label = Mode.TRACK_HYBRID, kw["label"]
            self.last_seq += 1
            assert kw["wait_for_contact"] is True
            self.hybrid_t0 = now[0]
            self.duration = kw["duration_s"]
            return 0

        def cartesian_track(self, **kw):
            calls.append(("lift", kw["poses"]))
            self.mode, self.label = Mode.TRACK_CARTESIAN, kw["label"]
            self.live = np.array(kw["poses"][-1])
            return 0

        def set_force_control(self, **kw):
            return 0

        def set_force_raw_override(self, payload):
            return 0

        def clear_force_control(self):
            calls.append(("clear_force", None))

        def set_arm_stop(self):
            calls.append(("stop", None))

        def close(self):
            calls.append(("close", None))

    arm = Arm()

    class Recorder:
        stopped = False

        def __init__(self, directory, *, read_status, desired_force_n):
            assert directory == tmp_path
            assert callable(read_status)
            assert desired_force_n == 4.0

        def start(self):
            calls.append(("record_start", None))

        def check_error(self):
            pass

        def stop(self):
            if not self.stopped:
                calls.append(("record_stop", None))
                self.stopped = True

        def finish(self, outcome):
            self.stop()
            calls.append(("record_finish", outcome))
            return {"csv": tmp_path / "force_trace.csv", "png": tmp_path / "force_trace.png"}

    def detect(_arm, args):
        assert args.pattern == pattern
        assert args.scan_length_m == 0.14
        assert args.scan_width_m == 0.08
        args.capture_dir = tmp_path
        return hit, plan

    monkeypatch.setattr(s_scan, "PeirasticArm", lambda: arm)
    monkeypatch.setattr(s_scan, "ForceTraceRecorder", Recorder)
    monkeypatch.setattr(s_scan, "_detect", detect)
    monkeypatch.setattr(s_scan, "_live_q8", lambda *a: s_scan.load_detect_pose())
    monkeypatch.setattr(s_scan, "_live_pose", lambda *a, **kw: arm.live.tolist())
    monkeypatch.setattr(s_scan, "_wait_enter", enters.append)
    cli = [] if pattern == "raster" else ["--pattern", pattern]
    assert s_scan.main(cli) == (0 if failure is None else 1)
    names = [name for name, _ in calls]
    assert names[:3] == ["dof", "movej", "ptp"]
    np.testing.assert_allclose(calls[2][1], standoff)
    if failure:
        assert "stop" in names
        assert names.count("phathom_s_scan") == (1 if failure == "contact" else 0)
        assert "lift" not in names
        assert names.count("movej") == 1
        if failure == "contact":
            assert "aborted" in dict(calls)["record_finish"]
            assert names.index("stop") < names.index("record_finish")
        else:
            assert "record_start" not in names
    else:
        assert len(enters) == 3
        assert names == ["dof", "movej", "ptp", "record_start", "phathom_s_scan",
                         "clear_force", "record_stop", "lift", "movej", "record_finish", "close"]
        assert dict(calls)["record_finish"] == "completed"
        actual_lift = dict(calls)["lift"]
        np.testing.assert_allclose(np.subtract(actual_lift[-1][:3], actual_lift[0][:3]),
                                   [0, 0, 0.01], atol=1e-9)
        assert actual_lift[-1][2] == pytest.approx(0.175)  # from measured indentation
        np.testing.assert_allclose(dict(calls)["phathom_s_scan"], plan.poses)
        import csv
        with (tmp_path / "motion_trace.csv").open() as stream:
            motion_rows = list(csv.DictReader(stream))
        assert motion_rows
        assert {r["stage"] for r in motion_rows} == {"approach", "tracking"}
        assert all(len(r) == 23 and None not in r for r in motion_rows)
        assert float(motion_rows[0]["surface_gap_mm"]) == pytest.approx(40.0)


def test_standoff_requires_orientation_as_well_as_position(monkeypatch):
    target = np.array([0.2, 0.2, 0.208, np.pi, 0, 0])
    live = target.copy()
    live[3] = 0.0
    monkeypatch.setattr(s_scan, "_latched", lambda arm: False)
    monkeypatch.setattr(s_scan, "_live_pose", lambda *a, **kw: live)
    arm = SimpleNamespace(_snapshot=lambda: {})
    assert not s_scan._await_tcp(arm, target, timeout_s=0.01)


def test_contact_seek_stops_below_detected_surface(monkeypatch):
    monkeypatch.setattr(s_scan, "_latched", lambda arm: False)
    monkeypatch.setattr(s_scan, "_live_pose", lambda *a, **kw: [0.2, 0.2, 0.15, np.pi, 0, 0])
    arm = SimpleNamespace(_snapshot=lambda: dict(
        mode=Mode.TRACK_HYBRID, status=Status.RUNNING, t_mono=s_scan.time.monotonic(),
        msg="phathom_s_scan:approach t=0.00 contact=0 vz_cmd=12.0", f_ext_z=0.0))
    plan = SimpleNamespace(poses=np.array([[0.2, 0.2, 0.168, np.pi, 0, 0]]))
    with pytest.raises(RuntimeError, match="10 mm below"):
        s_scan._monitor_hybrid_scan(arm, plan, duration_s=1.0, seq=1)


def test_stale_telemetry_cannot_confirm_contact_or_complete_scan(monkeypatch):
    monkeypatch.setattr(s_scan, "_latched", lambda arm: False)
    arm = SimpleNamespace(_snapshot=lambda: dict(
        mode=Mode.TRACK_HYBRID, status=Status.RUNNING, t_mono=s_scan.time.monotonic() - 1,
        done_seq=1, msg="phathom_s_scan:tracking t=3.00 contact=1 vz_cmd=0.0", f_ext_z=4.0))
    with pytest.raises(RuntimeError, match="stale controller telemetry"):
        s_scan._monitor_hybrid_scan(arm, None, duration_s=1.0, seq=1)


def test_failed_plan_keeps_replayable_cloud_and_detection(tmp_path, monkeypatch):
    import json

    x, y = np.meshgrid(np.linspace(-0.1, 0.1, 11), np.linspace(-0.1, 0.1, 11))
    xyz = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.18)])
    rgb = np.tile([0.55, 0.35, 0.18], (len(xyz), 1))
    hit = SimpleNamespace(points=xyz, centroid=xyz.mean(axis=0),
                          normal=np.array([0., 0., 1.]), n_points=len(xyz))
    monkeypatch.setattr(s_scan, "detect_phantom_top", lambda *a, **kw: hit)
    monkeypatch.setattr(s_scan, "plan_s_scan", lambda *a, **kw: (_ for _ in ()).throw(
        ValueError("point-cloud support: test gap")))
    args = s_scan._parse_args(["--plan-dir", str(tmp_path)])
    q8 = np.asarray(s_scan.load_detect_pose())
    with pytest.raises(ValueError, match="point-cloud support"):
        s_scan._plan_capture(args, xyz_cam=xyz, rgb=rgb, q8=q8, xyz=xyz)
    directory = next(tmp_path.iterdir())
    with np.load(directory / "capture.npz") as capture:
        np.testing.assert_array_equal(capture["xyz_cam"], xyz)
        np.testing.assert_array_equal(capture["q8"], q8)
    assert json.loads((directory / "failure.json").read_text())["stage"] == "surface_planning"
    assert (directory / "failure.png").is_file()


def test_replay_never_attaches_to_live_controller(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    np.savez(source, xyz_cam=np.ones((10, 3)), rgb=np.ones((10, 3)),
             q8=s_scan.load_detect_pose())
    monkeypatch.setattr(s_scan, "PeirasticArm", lambda: pytest.fail("offline replay attached to hardware"))
    monkeypatch.setattr(s_scan, "cloud_in_rail_base", lambda xyz, q: xyz)
    monkeypatch.setattr(s_scan, "_plan_capture", lambda *a, **kw:
                        (None, SimpleNamespace(n_rows=3, poses=np.zeros((10, 6)))))
    assert s_scan.main(["--replay", str(source), "--plan-dir", str(tmp_path)]) == 0
