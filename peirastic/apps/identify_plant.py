#!/usr/bin/env python3
"""Window C: free-space plant identification for τ_eff / G_v(s).

Window A must be running to servo the arm.  Default sequence: tool-Z
steps (±2/5/10/20/40/80 mm/s) and a 0.2–5 Hz chirp.

``--air-campaign`` writes its *own* 200 Hz CSV (command + MotionBus +
Fz + TDPA shadow) and runs ``--analyze-air`` when the sequence ends.
Do not analyse the Window A hybrid log for this campaign.  Plots under
``MD/todo_controller_logs/id_air_<stamp>/`` are required; no plots
means the campaign is incomplete.  Do not add stop-reverse here.

``--stop-reverse`` is plant identification: +10/+20/+40/+80 mm/s to 0 and
to −40 mm/s.  Analyse with ``--analyze-stop --input-column twist_vz``.
``--backup-replay`` sends the shield's a/j-limited ``u_b``.  Certificate
replay reads ``v_actual`` from the 200 Hz ``peirastic_motion`` SHM, not
the Window A CSV (flushed every 200 rows).  Stale motion aborts to zero.
Analyse with ``--analyze-stop --event-log``.  Independent backup val
needs ``--val-event-log``.  That is still not a certificate until
every event reaches T, jerk-state covering passes, and
``stop_dx_ub.certified: true``.  Do not set certified true until
the terminal box proof and backup-table state are complete.

Does not enable force mode.  Bidirectional_flow stays observe/off.
``--analyze-tn`` fits Γ_d + min-phase T_n from ``vel_ff_vz → vz_achieved_tool``.
``--replay-cdyob`` shadows the observer on an existing hybrid CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from peirastic.core.ipc import CommandClient, MotionBus, Status, TwistBus
from peirastic.core.modes import Mode, ModeRequest
from peirastic.core.session import request_dof, stop_before_dof

STEPS_MM_S = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
STOP_SPEEDS_MM_S = (10.0, 20.0, 40.0, 80.0)
REVERSE_MM_S = -40.0


def _install_servo_mode(client: CommandClient) -> None:
    """Install SERVO_TWIST and wait for Window A's post-on_enter ACK."""

    req = ModeRequest(Mode.SERVO_TWIST, {"filter": False})
    seq = client.set_mode(req)
    ret = client.wait_installed(seq, req.mode)
    if ret != 0:
        raise RuntimeError(f"SERVO_TWIST install failed ({ret})")


def _parse_speeds_mm_s(raw: str) -> tuple[float, ...]:
    text = str(raw).strip()
    if not text:
        return STOP_SPEEDS_MM_S
    return tuple(float(x) for x in text.split(",") if x.strip())


def _force_yaml_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "force.yaml"


def _tool_z_twist(vz_m_s: float) -> np.ndarray:
    tw = np.zeros(6, dtype=float)
    tw[2] = float(vz_m_s)
    return tw


def _write_vz(bus: TwistBus, vz_m_s: float, hz: float) -> None:
    bus.write(_tool_z_twist(vz_m_s), hz=hz, connected=True)


def _wait_or_estop(client: CommandClient, seconds: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        tel = client.snapshot()
        if int(tel["status"]) == int(Status.ESTOP):
            print("[ESTOP] " + str(tel["msg"]), flush=True)
            return False
        time.sleep(0.005)
    return True


def run_sequence(
    *,
    prefix: str,
    hold_s: float,
    rest_s: float,
    chirp_s: float,
    chirp_amp_m_s: float,
    hz: float,
    include_steps: bool = True,
    steps_mm_s: tuple[float, ...] | None = None,
    chirp_f0_hz: float = 0.2,
    chirp_f1_hz: float = 5.0,
    chirp_amps_m_s: tuple[float, ...] | None = None,
) -> int:
    client = CommandClient(prefix=prefix)
    bus = TwistBus(prefix=prefix, create=False)
    # DOF changes are boundary requests; stop the idle/live SERVO phase
    # explicitly so the daemon can commit the requested 8-DOF structure
    # before the first plant-identification sample.  A zero twist alone is
    # intentionally not a boundary.
    previous_dof: int | None = None
    try:
        stop_before_dof(client)
        previous_dof = request_dof(client, 8)
        _install_servo_mode(client)
    except BaseException:
        try:
            if previous_dof in (7, 8):
                stop_before_dof(client)
                request_dof(client, previous_dof)
        finally:
            bus.close()
            client.close()
        raise
    print("[MODE] SERVO_TWIST identify_plant  filter OFF", flush=True)
    dt = 1.0 / max(hz, 1.0)
    try:
        _write_vz(bus, 0.0, hz)
        if not _wait_or_estop(client, rest_s):
            return 130
        step_list = STEPS_MM_S if steps_mm_s is None else steps_mm_s
        if include_steps:
            for mm_s in step_list:
                vz = mm_s / 1000.0
                for sign in (1.0, -1.0):
                    cmd = sign * vz
                    print(
                        f"[STEP] vz={cmd:+.4f} m/s hold={hold_s:.2f}s",
                        flush=True,
                    )
                    t0 = time.monotonic()
                    while time.monotonic() - t0 < hold_s:
                        _write_vz(bus, cmd, hz)
                        tel = client.snapshot()
                        if int(tel["status"]) == int(Status.ESTOP):
                            print("[ESTOP] " + str(tel["msg"]), flush=True)
                            return 130
                        time.sleep(dt)
                    t1 = time.monotonic()
                    while time.monotonic() - t1 < rest_s:
                        _write_vz(bus, 0.0, hz)
                        time.sleep(dt)
        amps = (chirp_amp_m_s,) if chirp_amps_m_s is None else chirp_amps_m_s
        f0 = max(float(chirp_f0_hz), 1e-3)
        f1 = max(float(chirp_f1_hz), f0 + 1e-3)
        for amp in amps:
            print(
                f"[CHIRP] {amp*1000:.1f} mm/s  {f0:.2f}–{f1:.1f} Hz  {chirp_s:.1f}s",
                flush=True,
            )
            t0 = time.monotonic()
            while True:
                t = time.monotonic() - t0
                if t >= chirp_s:
                    break
                k = math.log(f1 / f0) / max(chirp_s, 1e-6)
                phase = 2.0 * math.pi * f0 * (math.exp(k * t) - 1.0) / k
                vz = float(amp) * math.sin(phase)
                _write_vz(bus, vz, hz)
                tel = client.snapshot()
                if int(tel["status"]) == int(Status.ESTOP):
                    print("[ESTOP] " + str(tel["msg"]), flush=True)
                    return 130
                time.sleep(dt)
            if not _wait_or_estop(client, rest_s):
                return 130
        _write_vz(bus, 0.0, hz)
        print("[OK] sequence complete — analyse the Window A CSV with --analyze or --analyze-air", flush=True)
        return 0
    except KeyboardInterrupt:
        _write_vz(bus, 0.0, hz)
        client.stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        try:
            stop_before_dof(client)
            if previous_dof in (7, 8):
                request_dof(client, previous_dof)
        except Exception as exc:
            print(f"[DOF] restore {previous_dof} failed: {exc}", flush=True)
        bus.close()
        client.close()


def _hold_cmd(bus: TwistBus, client: CommandClient, vz_m_s: float, seconds: float, hz: float) -> bool:
    dt = 1.0 / max(hz, 1.0)
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        _write_vz(bus, vz_m_s, hz)
        tel = client.snapshot()
        if int(tel["status"]) == int(Status.ESTOP):
            print("[ESTOP] " + str(tel["msg"]), flush=True)
            return False
        time.sleep(dt)
    return True


def _motion_fresh_or_wait(
    motion: MotionBus,
    last_seq: int,
    *,
    max_age_s: float,
    poll_s: float = 0.001,
) -> tuple[dict | None, str]:
    """Wait for the next even generation; do not reuse a sample.

    ``seq_stale`` / torn copies are publish jitter, not a certificate fault,
    until ``max_age_s`` elapses.  ``age_total`` / invalid / empty still abort
    immediately so a stale or dead SHM is never used as ``v_actual``.
    """
    limit = max(float(max_age_s), 0.0)
    deadline = time.monotonic() + limit
    last_why = "seq_stale"
    while True:
        row, why = motion.fresh(last_seq, max_age_s=limit)
        if row is not None:
            return row, ""
        last_why = str(why or "seq_stale")
        if last_why not in ("seq_stale", "torn"):
            return None, last_why
        if time.monotonic() >= deadline:
            return None, last_why
        time.sleep(max(float(poll_s), 0.0))


def run_stop_reverse_sequence(
    *,
    prefix: str,
    hold_s: float,
    short_hold_s: float,
    rest_s: float,
    hz: float,
    speeds_mm_s: tuple[float, ...] = STOP_SPEEDS_MM_S,
) -> int:
    """Free-space stop / reverse identification.  No force loop.

    This sequence identifies the plant from commanded tool-Z to achieved
    speed.  Instantaneous ``→ 0`` / ``→ −40 mm/s`` is *not* the shield
    backup ``u_b`` (which is a/j limited).  Measured Δx is therefore a
    plant-ID statistic, not Δx_b^ub, unless the CSV command matches the
    shield backup.
    """
    client = CommandClient(prefix=prefix)
    bus = TwistBus(prefix=prefix, create=False)
    previous_dof: int | None = None
    try:
        stop_before_dof(client)
        previous_dof = request_dof(client, 8)
        _install_servo_mode(client)
    except BaseException:
        try:
            if previous_dof in (7, 8):
                stop_before_dof(client)
                request_dof(client, previous_dof)
        finally:
            bus.close()
            client.close()
        raise
    print("[MODE] SERVO_TWIST identify_plant --stop-reverse  filter OFF", flush=True)
    try:
        if not _hold_cmd(bus, client, 0.0, rest_s, hz):
            return 130
        for mm_s in speeds_mm_s:
            vz = mm_s / 1000.0
            rev = REVERSE_MM_S / 1000.0
            for label, dwell in (("settled", hold_s), ("accel", short_hold_s)):
                print(
                    f"[STOP] +{mm_s:.0f}→0  {label} hold={dwell:.2f}s",
                    flush=True,
                )
                if not _hold_cmd(bus, client, vz, dwell, hz):
                    return 130
                if not _hold_cmd(bus, client, 0.0, rest_s, hz):
                    return 130
                print(
                    f"[REV]  +{mm_s:.0f}→{REVERSE_MM_S:.0f}  {label} hold={dwell:.2f}s",
                    flush=True,
                )
                if not _hold_cmd(bus, client, vz, dwell, hz):
                    return 130
                if not _hold_cmd(bus, client, rev, rest_s, hz):
                    return 130
                if not _hold_cmd(bus, client, 0.0, rest_s, hz):
                    return 130
        print(
            "[OK] stop/reverse complete — analyse with --analyze-stop. "
            "This is plant ID (u_cmd→v), not a certified Δx_b^ub unless "
            "the logged command equals the a/j-limited backup.",
            flush=True,
        )
        return 0
    except KeyboardInterrupt:
        _write_vz(bus, 0.0, hz)
        client.stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        try:
            stop_before_dof(client)
            if previous_dof in (7, 8):
                request_dof(client, previous_dof)
        except Exception as exc:
            print(f"[DOF] restore {previous_dof} failed: {exc}", flush=True)
        bus.close()
        client.close()


def run_backup_replay_sequence(
    *,
    prefix: str,
    hold_s: float,
    short_hold_s: float,
    rest_s: float,
    hz: float,
    speeds_mm_s: tuple[float, ...],
    event_log: Path,
    window_a_csv: Path | None = None,
    motion_max_age_s: float = 0.015,
) -> int:
    """Play the shield's a/j-limited ``u_b``, not an instant −40 mm/s step.

    Closed-loop replay requires the 200 Hz ``peirastic_motion`` SHM.
    Window A CSV is flushed every 200 rows and is not a feedback source.
    """
    from collections import deque

    import yaml

    from rm75_control.control.admittance_common.delay_safety_shield import (
        DelaySafetyShield,
        SafetyShieldConfig,
    )

    dt = 1.0 / max(hz, 1.0)
    raw = {}
    ypath = _force_yaml_path()
    if ypath.is_file():
        loaded = yaml.safe_load(ypath.read_text())
        raw = loaded if isinstance(loaded, dict) else {}
    sh = DelaySafetyShield(SafetyShieldConfig.from_dict(raw), dt)
    client = CommandClient(prefix=prefix)
    bus = TwistBus(prefix=prefix, create=False)
    try:
        motion = MotionBus(prefix=prefix, create=False)
    except Exception as exc:
        bus.close()
        client.close()
        print(
            f"[ERR] peirastic_motion SHM missing ({exc}).  Restart Window A "
            "after this build.  CSV is not 200 Hz feedback; aborting "
            "certificate-grade --backup-replay.",
            flush=True,
        )
        return 2
    previous_dof: int | None = None
    try:
        stop_before_dof(client)
        previous_dof = request_dof(client, 8)
        _install_servo_mode(client)
    except BaseException:
        try:
            if previous_dof in (7, 8):
                stop_before_dof(client)
                request_dof(client, previous_dof)
        finally:
            bus.close()
            client.close()
            motion.close()
        raise
    print(
        "[MODE] SERVO_TWIST identify_plant --backup-replay  filter OFF  "
        f"a_max={sh.cfg.a_max_m_s2} j_max={sh.cfg.j_max_m_s3} "
        f"u_retract={sh.cfg.u_retract_m_s}  "
        f"motion SHM max_age={1e3 * motion_max_age_s:.0f} ms  "
        "(CSV is not closed-loop)",
        flush=True,
    )
    rows: list[dict[str, str]] = []
    event_n = 0
    last_seq = -1

    def _abort_stale(reason: str) -> int:
        _write_vz(bus, 0.0, hz)
        print(
            f"[ABORT] motion feedback {reason}; sent zero.  "
            "Not a stop certificate.",
            flush=True,
        )
        return 2

    try:
        if not _hold_cmd(bus, client, 0.0, rest_s, hz):
            return 130
        for mm_s in speeds_mm_s:
            vz = mm_s / 1000.0
            for label, dwell in (("settled", hold_s), ("accel", short_hold_s)):
                event_n += 1
                eid = f"{event_n:03d}_{mm_s:.0f}_{label}"
                if not _hold_cmd(bus, client, vz, dwell, hz):
                    return 130
                delay_n = max(sh._delay_steps(), 1)
                sh.reset()
                sh._delay = deque([vz] * delay_n, maxlen=max(delay_n, 1))
                sh._u_prev = vz
                sh._u_prev2 = vz
                t_trigger_mono = time.monotonic()
                print(
                    f"[BACKUP] event_id={eid} trigger=backup_to_terminal "
                    f"phase={label} v0_cmd={mm_s:.0f} mm/s hold={dwell:.2f}s",
                    flush=True,
                )
                hold_acc = 0.0
                n_max = int(round((rest_s + 0.50) / dt))
                for tick in range(n_max):
                    row_m, why = _motion_fresh_or_wait(
                        motion, last_seq, max_age_s=motion_max_age_s
                    )
                    if row_m is None:
                        return _abort_stale(why)
                    last_seq = int(row_m["seq"])
                    sh._sync_plant_from_measurement(
                        float(row_m["v_tcp_z"]), float(row_m["a_tcp_z_plus"])
                    )
                    if tick == 0:
                        print(
                            f"[BACKUP] trigger v0_meas="
                            f"{1e3 * float(row_m['v_tcp_z']):.1f} mm/s "
                            f"a0+={float(row_m['a_tcp_z_plus']):.3f} "
                            f"t_wall={row_m['t_wall_s']:.4f}",
                            flush=True,
                        )
                    u = sh.backup_command(
                        sh._u_prev,
                        sh._u_prev2,
                        released=False,
                        v_pred=float(sh._v_plant),
                    )
                    t_unix = time.time()
                    _write_vz(bus, u, hz)
                    sh._commit_sent(u, keep_measured_state=True)
                    queue_u = ";".join(f"{float(x):.6f}" for x in sh._delay)
                    t_csv_now = None
                    if window_a_csv is not None:
                        _v_csv, _a_csv, t_csv_now = _tail_window_a_motion(window_a_csv)
                        del _v_csv, _a_csv
                    rows.append(
                        {
                            "event_id": eid,
                            "trigger": "backup_to_terminal",
                            "phase": label,
                            "tick": str(tick),
                            "t_unix_s": f"{t_unix:.6f}",
                            "t_mono_s": f"{time.monotonic() - t_trigger_mono:.6f}",
                            "t_wall_csv_s": (
                                f"{float(row_m['t_wall_s']):.6f}"
                                if math.isfinite(float(row_m["t_wall_s"]))
                                else ("" if t_csv_now is None else f"{t_csv_now:.6f}")
                            ),
                            "u_b": f"{u:.6f}",
                            "v0_cmd": f"{vz:.6f}",
                            "v_actual": f"{float(row_m['v_tcp_z']):.6f}",
                            "a0_plus": f"{float(row_m['a_tcp_z_plus']):.6f}",
                            "q_remain_m": f"{sh.queue_remain_m():.8f}",
                            "queue_u": queue_u,
                        }
                    )
                    tel = client.snapshot()
                    if int(tel["status"]) == int(Status.ESTOP):
                        print("[ESTOP] " + str(tel["msg"]), flush=True)
                        return 130
                    q_ok = sh.queue_press() <= float(sh.cfg.queue_clear_m_s) + 1e-9
                    v_ok = abs(float(sh._v_plant)) <= float(sh.cfg.v_hold_m_s) + 1e-9
                    if (
                        abs(u) <= float(sh.cfg.queue_clear_m_s) + 1e-9
                        and v_ok
                        and q_ok
                    ):
                        hold_acc += dt
                        if hold_acc + 1e-12 >= 0.050:
                            break
                    else:
                        hold_acc = 0.0
                    time.sleep(dt)
                if not _hold_cmd(bus, client, 0.0, rest_s, hz):
                    return 130
        event_log.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "event_id",
            "trigger",
            "phase",
            "tick",
            "t_unix_s",
            "t_mono_s",
            "t_wall_csv_s",
            "u_b",
            "v0_cmd",
            "v_actual",
            "a0_plus",
            "q_remain_m",
            "queue_u",
        ]
        with event_log.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"[OK] backup replay complete — events={event_n} log={event_log}. "
            "Analyse Window A CSV with --analyze-stop --event-log "
            f"{event_log} --input-column twist_vz. "
            "This is still not a certificate until independent val and "
            "stop_dx_ub.certified: true.",
            flush=True,
        )
        return 0
    except KeyboardInterrupt:
        _write_vz(bus, 0.0, hz)
        client.stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        try:
            stop_before_dof(client)
            if previous_dof in (7, 8):
                request_dof(client, previous_dof)
        except Exception as exc:
            print(f"[DOF] restore {previous_dof} failed: {exc}", flush=True)
        bus.close()
        client.close()
        motion.close()


def _simulate_first_order(
    cmd: np.ndarray,
    *,
    delay_steps: int,
    tp_s: float,
    dt: float,
) -> np.ndarray:
    alpha = math.exp(-dt / max(tp_s, 1e-4))
    v = 0.0
    delay = [0.0] * max(int(delay_steps), 0)
    out = np.zeros_like(cmd)
    for i, u in enumerate(cmd):
        if delay:
            u_app = delay[0]
            delay = delay[1:] + [float(u)]
        else:
            u_app = float(u)
        v = alpha * v + (1.0 - alpha) * u_app
        out[i] = v
    return out


def _tail_window_a_motion(
    path: Path | None,
) -> tuple[float | None, float | None, float | None]:
    """Latest ``(v_z, [a_z]+, t_wall_csv)`` from a growing Window A CSV."""
    if path is None:
        return None, None, None
    csv_path = Path(path)
    if not csv_path.is_file():
        return None, None, None
    try:
        data = csv_path.read_bytes()
    except OSError:
        return None, None, None
    if len(data) > 131072:
        chunk = data[:2048] + b"\n" + data[-65536:]
        text = chunk.decode("utf-8", "replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 2:
            header = lines[0]
            body = lines[1:]
            if "," in header and "t_wall" in header:
                text = header + "\n" + "\n".join(body)
            else:
                text = data[-65536:].decode("utf-8", "replace")
    else:
        text = data.decode("utf-8", "replace")
    try:
        reader = csv.DictReader(text.splitlines())
        rows = [r for r in reader if r]
    except Exception:
        return None, None, None
    samples: list[tuple[float, float]] = []
    for row in rows:
        try:
            t = float(row.get("t_wall_s") or "nan")
            v = float(
                row.get("twist_achieved_vz")
                or row.get("v_tcp_z_actual")
                or row.get("vz_achieved_tool")
                or "nan"
            )
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and math.isfinite(v):
            samples.append((t, v))
    if not samples:
        return None, None, None
    t1, v1 = samples[-1]
    a_plus = 0.0
    if len(samples) >= 2:
        t0, v0 = samples[-2]
        dts = t1 - t0
        if dts > 1e-4:
            a_plus = max((v1 - v0) / dts, 0.0)
    return v1, a_plus, t1


def _horizon_error_ub(
    cmd: np.ndarray,
    ach: np.ndarray,
    *,
    delay_steps: int,
    tp_s: float,
    dt: float,
    horizon: int,
    margin: float,
) -> tuple[list[float], list[float]]:
    ev = [0.0] * horizon
    n = int(cmd.size)
    alpha = math.exp(-dt / max(tp_s, 1e-4))
    for k in range(0, n - horizon, max(horizon // 4, 1)):
        delay = [float(cmd[max(k - delay_steps + j, 0)]) for j in range(delay_steps)]
        v = float(ach[k])
        for i in range(horizon):
            idx = k + i
            u = float(cmd[idx]) if idx < n else 0.0
            if delay_steps > 0:
                u_app = delay[0]
                delay = delay[1:] + [u]
            else:
                u_app = u
            v = alpha * v + (1.0 - alpha) * u_app
            if idx + 1 < n:
                ev[i] = max(ev[i], abs(float(ach[idx + 1]) - v))
    ev_ub = [float(e + margin) for e in ev]
    ex_ub: list[float] = []
    acc = 0.0
    for e in ev_ub:
        acc += dt * e
        ex_ub.append(acc)
    return ev_ub, ex_ub


def analyze_csv(
    path: Path,
    *,
    horizon: int = 40,
    write_yaml: Path | None = None,
    margin: float = 0.002,
) -> int:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[ERR] empty csv {path}", flush=True)
        return 1
    cmd = np.array([float(r.get("twist_vz") or "nan") for r in rows], dtype=float)
    ach = np.array(
        [float(r.get("twist_achieved_vz") or r.get("v_tcp_z_actual") or "nan")
         for r in rows],
        dtype=float,
    )
    t = np.array([float(r.get("t_wall_s") or "nan") for r in rows], dtype=float)
    mask = np.isfinite(cmd) & np.isfinite(ach) & np.isfinite(t)
    cmd, ach, t = cmd[mask], ach[mask], t[mask]
    if cmd.size < 50:
        print("[ERR] not enough finite twist samples", flush=True)
        return 1
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.005
    split = max(int(0.7 * cmd.size), 50)
    cmd_tr, ach_tr = cmd[:split], ach[:split]
    cmd_va, ach_va = cmd[split:], ach[split:]
    if cmd_va.size < horizon + 5:
        cmd_tr, ach_tr = cmd[: cmd.size // 2], ach[: cmd.size // 2]
        cmd_va, ach_va = cmd[cmd.size // 2 :], ach[cmd.size // 2 :]

    cmd0 = cmd_tr - np.mean(cmd_tr)
    ach0 = ach_tr - np.mean(ach_tr)
    corr = np.correlate(ach0, cmd0, mode="full")
    lags = np.arange(-cmd0.size + 1, ach0.size)
    i = int(np.argmax(corr))
    lag_ticks = max(int(lags[i]), 0)
    t0_s = lag_ticks * dt
    best_tp = 0.060
    best_err = float("inf")
    for tp in np.linspace(0.020, 0.150, 14):
        pred = _simulate_first_order(
            cmd_tr, delay_steps=lag_ticks, tp_s=float(tp), dt=dt
        )
        err = float(np.mean((pred - ach_tr) ** 2))
        if err < best_err:
            best_err = err
            best_tp = float(tp)
    ev_ub, ex_ub = _horizon_error_ub(
        cmd_va,
        ach_va,
        delay_steps=lag_ticks,
        tp_s=best_tp,
        dt=dt,
        horizon=int(horizon),
        margin=float(margin),
    )
    rho = float("nan")
    if lag_ticks > 0 and cmd.size > lag_ticks + 2:
        rho = float(np.corrcoef(cmd[:-lag_ticks], ach[lag_ticks:])[0, 1])
    elif cmd.size > 2:
        rho = float(np.corrcoef(cmd, ach)[0, 1])
    print(f"[ID] file={path}", flush=True)
    print(f"[ID] n={cmd.size} train={cmd_tr.size} val={cmd_va.size} dt≈{dt*1000:.2f} ms", flush=True)
    print(
        f"[ID] T0≈{t0_s*1000:.1f} ms  Tp≈{best_tp*1000:.1f} ms  "
        f"corr≈{rho:.3f}  (placeholder plant, not a certificate)",
        flush=True,
    )
    print(
        f"[ID] val ē_v(1)={ev_ub[0]:.4f} ē_v({horizon})={ev_ub[-1]:.4f} m/s  "
        f"ē_x({horizon})={ex_ub[-1]*1000:.2f} mm",
        flush=True,
    )
    yaml_block = (
        "safety_shield:\n"
        "  plant:\n"
        f"    t0_s: {t0_s:.4f}\n"
        f"    tp_s: {best_tp:.4f}\n"
        f"    horizon_steps: {int(horizon)}\n"
        "  velocity_error_ub_m_s:\n"
        + "".join(f"    - {e:.6f}\n" for e in ev_ub)
        + "  position_error_ub_m:\n"
        + "".join(f"    - {e:.7f}\n" for e in ex_ub)
        + "  position_error_ub_plus_m:\n"
        + "".join(f"    - {e:.7f}\n" for e in ex_ub)
    )
    print(yaml_block, flush=True)
    print(
        "[ID] write these bounds into hybrid_motion.safety_shield after a "
        "separate validation set covers the operating envelope. "
        "Do not treat this fit as a passivity proof.",
        flush=True,
    )
    if write_yaml is not None:
        write_yaml.write_text(yaml_block)
        print(f"[ID] wrote {write_yaml}", flush=True)
    return 0


def _load_twist_csv(
    path: Path,
    *,
    input_column: str = "twist_vz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    twist = np.array([float(r.get("twist_vz") or "nan") for r in rows], dtype=float)
    sent = np.array([float(r.get("u_sent") or "nan") for r in rows], dtype=float)
    ach = np.array(
        [float(r.get("twist_achieved_vz") or r.get("v_tcp_z_actual") or "nan")
         for r in rows],
        dtype=float,
    )
    t = np.array([float(r.get("t_wall_s") or "nan") for r in rows], dtype=float)
    col = str(input_column).strip().lower()
    if col not in ("twist_vz", "u_sent"):
        raise ValueError("input-column must be twist_vz or u_sent")
    cmd = twist if col == "twist_vz" else sent
    mask = np.isfinite(cmd) & np.isfinite(ach) & np.isfinite(t)
    return cmd[mask], ach[mask], t[mask], twist[mask], sent[mask]


def _load_event_log(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _event_trigger_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """First tick of each event_id, in log order."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        eid = str(row.get("event_id") or "").strip()
        if not eid or eid in seen:
            continue
        if str(row.get("trigger") or "") != "backup_to_terminal":
            continue
        tick = str(row.get("tick") or "0").strip()
        if tick not in ("", "0"):
            continue
        seen.add(eid)
        out.append(row)
    if not out:
        # Logs without a tick column: first row of each event_id.
        seen.clear()
        for row in rows:
            eid = str(row.get("event_id") or "").strip()
            if not eid or eid in seen:
                continue
            seen.add(eid)
            out.append(row)
    return out


def _align_trigger_index(
    cmd: np.ndarray,
    t: np.ndarray,
    trigger: dict[str, str],
    *,
    search_from: int = 1,
) -> int | None:
    """Map an explicit trigger to the Window A sample that starts backup.

    Prefers the first logged ``u_b`` against ``v0_cmd → u_b`` so integration
    starts at the true trigger tick, not a 3 mm/s edge and not the last
    hold sample.  ``t_wall_csv_s`` is the fallback.
    """
    try:
        v0 = float(trigger.get("v0_cmd") or "nan")
        u0 = float(trigger.get("u_b") or "nan")
    except ValueError:
        v0, u0 = float("nan"), float("nan")
    start = max(int(search_from), 1)
    if math.isfinite(v0) and math.isfinite(u0):
        for k in range(start, int(cmd.size)):
            prev = float(cmd[k - 1])
            now = float(cmd[k])
            if (
                abs(prev - v0) <= 0.0015
                and abs(now - u0) <= 2.0e-4
                and now < prev - 1e-6
            ):
                return int(k)
    t_csv = trigger.get("t_wall_csv_s") or ""
    if t_csv.strip():
        try:
            t0 = float(t_csv)
        except ValueError:
            t0 = float("nan")
        if math.isfinite(t0) and t.size:
            idx = int(np.argmin(np.abs(t - t0)))
            if abs(float(t[idx]) - t0) <= 0.020 and idx >= start - 1:
                return max(idx, 0)
    return None


def _step_dt(t: np.ndarray, fallback: float) -> np.ndarray:
    """Per-sample Δt.  Large gaps are *not* clipped; callers must invalidate."""
    n = int(t.size)
    dt = np.full(n, max(float(fallback), 1e-4))
    if n > 1:
        d = np.diff(t)
        dt[1:] = d
        dt[0] = d[0] if d.size else dt[0]
    return np.maximum(dt, 1e-4)


def _has_dt_gap(dt_arr: np.ndarray, start: int, n: int, max_s: float) -> bool:
    sl = dt_arr[int(start) : int(start) + int(n)]
    if sl.size == 0:
        return False
    return bool(float(np.max(sl)) > float(max_s) + 1e-12)


def _event_end(cmd: np.ndarray, start: int) -> int:
    """Stop integrating at the next press trial, not at end-of-file."""
    saw_nonpos = False
    for k in range(int(start), int(cmd.size)):
        u = float(cmd[k])
        if u <= 0.002:
            saw_nonpos = True
        elif saw_nonpos and u > 0.008:
            return int(k)
    return int(cmd.size)


def _event_metrics(
    cmd: np.ndarray,
    ach: np.ndarray,
    t: np.ndarray,
    *,
    start: int,
    settle_m_s: float,
    horizon: int,
    v_hold_m_s: float = 0.015,
    a_hold_m_s2: float = 0.15,
    u_clear_m_s: float = 0.015,
    hold_s: float = 0.050,
    delay_s: float = 0.050,
    end: int | None = None,
    dt_gap_s: float = 0.050,
) -> dict[str, float]:
    """Press-positive stop metrics.  Negative speed never reduces Δx_b^+."""
    del settle_m_s
    dt_arr = _step_dt(t, 0.005)
    stop = int(end) if end is not None else _event_end(cmd, start)
    stop = min(max(stop, start + 2), int(ach.size))
    search_n = int(stop - start)
    if search_n < 2:
        return {}
    if _has_dt_gap(dt_arr, start, search_n, dt_gap_s):
        return {
            "v0": float(ach[start]),
            "a0": 0.0,
            "dx_press": float("nan"),
            "dx_retract": float("nan"),
            "n_press": float("nan"),
            "n_b": float("nan"),
            "reached_T": 0.0,
            "horizon": float(horizon),
            "q_press": 0.0,
            "q_remain_m": 0.0,
            "u_prev": 0.0,
            "a_cmd": 0.0,
            "q_front": 0.0,
            "gap_invalid": 1.0,
        }
    v0 = float(ach[start])
    a0 = (
        (float(ach[start]) - float(ach[start - 1])) / float(dt_arr[start])
        if start > 0
        else 0.0
    )
    a0_plus = max(a0, 0.0)
    delay_n = max(int(round(float(delay_s) / max(float(np.median(dt_arr)), 1e-4))), 1)
    dx_press = 0.0
    dx_retract = 0.0
    n_press = 0
    hold_acc = 0.0
    n_b = float("nan")
    reached = False
    for i in range(search_n):
        idx = start + i
        v = float(ach[idx])
        dts = float(dt_arr[idx])
        dx_press += max(v, 0.0) * dts
        dx_retract += max(-v, 0.0) * dts
        if v > 0.0:
            n_press = i + 1
        u = float(cmd[idx])
        a_act = (v - float(ach[idx - 1])) / dts if idx > 0 else 0.0
        a_cmd = (u - float(cmd[idx - 1])) / dts if idx > 0 else 0.0
        q0 = start + i + 1 - delay_n
        pending = cmd[max(q0, 0) : idx + 1]
        queue_ok = (
            pending.size == 0
            or float(np.max(np.abs(pending))) <= float(u_clear_m_s) + 1e-9
        )
        terminal = (
            abs(v) <= float(v_hold_m_s) + 1e-9
            and abs(a_act) <= float(a_hold_m_s2) + 1e-9
            and abs(u) <= float(u_clear_m_s) + 1e-9
            and abs(a_cmd) <= float(a_hold_m_s2) + 1e-9
            and queue_ok
        )
        if terminal:
            hold_acc += dts
            if hold_acc + 1e-12 >= float(hold_s) and not reached:
                n_b = float(i + 1)
                reached = True
        else:
            hold_acc = 0.0
    inflight = cmd[max(start - delay_n, 0) : start]
    dt_med = float(np.median(dt_arr))
    q_remain = (
        dt_med * float(np.sum(np.maximum(inflight, 0.0))) if inflight.size else 0.0
    )
    q_press = float(max(float(np.max(inflight)), 0.0)) if inflight.size else 0.0
    u_prev = float(cmd[start - 1]) if start > 0 else 0.0
    u_prev2 = float(cmd[start - 2]) if start > 1 else u_prev
    dt0 = float(dt_arr[start]) if start < dt_arr.size else dt_med
    a_cmd = (u_prev - u_prev2) / dt0 if dt0 > 1e-12 else 0.0
    q_front = float(inflight[0]) if inflight.size else 0.0
    return {
        "v0": v0,
        "a0": a0_plus,
        "a0_signed": a0,
        "dx_press": dx_press,
        "dx_retract": dx_retract,
        "n_press": float(n_press),
        "n_b": n_b,
        "reached_T": 1.0 if reached else 0.0,
        "horizon": float(horizon),
        "q_press": q_press,
        "q_remain_m": q_remain,
        "u_prev": max(u_prev, 0.0),
        "a_cmd": max(a_cmd, 0.0),
        "q_front": max(q_front, 0.0),
        "gap_invalid": 0.0,
    }


def _rollout_fopdt(
    cmd: np.ndarray,
    ach: np.ndarray,
    k: int,
    horizon: int,
    delay_steps: int,
    tp_s: float,
    dt_arr: np.ndarray,
) -> np.ndarray:
    d = max(int(delay_steps), 0)
    delay = [float(cmd[max(k - d + 1 + j, 0)]) for j in range(d)]
    v = float(ach[k])
    out = np.zeros(horizon)
    for i in range(horizon):
        idx = k + 1 + i
        u = float(cmd[idx]) if idx < cmd.size else 0.0
        dts = float(dt_arr[min(idx, max(dt_arr.size - 1, 0))])
        alpha = math.exp(-dts / max(float(tp_s), 1e-4))
        if delay:
            u_app = delay[0]
            delay = delay[1:] + [u]
        else:
            u_app = u
        v = alpha * v + (1.0 - alpha) * u_app
        out[i] = v
    return out


def _rollout_second(
    cmd: np.ndarray,
    ach: np.ndarray,
    k: int,
    horizon: int,
    wn: float,
    z: float,
    dt_arr: np.ndarray,
) -> np.ndarray:
    y = float(ach[k])
    dts0 = float(dt_arr[k]) if k < dt_arr.size else 0.005
    yd = (float(ach[k]) - float(ach[k - 1])) / dts0 if k > 0 else 0.0
    out = np.zeros(horizon)
    for i in range(horizon):
        idx = k + 1 + i
        u = float(cmd[idx]) if idx < cmd.size else 0.0
        dts = float(dt_arr[min(idx, max(dt_arr.size - 1, 0))])
        ydd = (wn * wn) * (u - y) - 2.0 * z * wn * yd
        yd += ydd * dts
        y += yd * dts
        out[i] = y
    return out


def _rollout_arx(
    cmd: np.ndarray,
    ach: np.ndarray,
    k: int,
    horizon: int,
    theta: np.ndarray,
    na: int,
    nb: int,
) -> np.ndarray:
    yhat: dict[int, float] = {}

    def y_at(j: int) -> float:
        if j in yhat:
            return yhat[j]
        if 0 <= j <= k:
            return float(ach[j])
        return 0.0

    def u_at(j: int) -> float:
        if 0 <= j < cmd.size:
            return float(cmd[j])
        return 0.0

    out = np.zeros(horizon)
    for i in range(1, horizon + 1):
        t = k + i
        phi = [y_at(t - 1 - j) for j in range(na)] + [u_at(t - j) for j in range(nb)]
        pred = float(np.dot(theta, np.asarray(phi, dtype=float)))
        yhat[t] = pred
        out[i - 1] = pred
    return out


def _open_loop_envelopes(
    cmd: np.ndarray,
    ach: np.ndarray,
    t: np.ndarray,
    *,
    origins: list[int],
    horizon: int,
    rollout_fn,
    origin_dt_gap_s: float = 0.050,
) -> tuple[list[float], list[float], list[float], list[float], int]:
    """Unified ê_v, ê_x, ê_{x,+}, ê_a from the same origins: v̂(k+i|k)."""
    ev = [0.0] * horizon
    ex = [0.0] * horizon
    ex_plus = [0.0] * horizon
    ea = [0.0] * horizon
    dt_arr = _step_dt(t, 0.005)
    used = 0
    for k in origins:
        if k + horizon >= ach.size or k < 0:
            continue
        if _has_dt_gap(dt_arr, k, horizon + 1, origin_dt_gap_s):
            continue
        pred = rollout_fn(k)
        acc = 0.0
        acc_plus_act = 0.0
        acc_plus_hat = 0.0
        prev_act = float(ach[k])
        prev_hat = float(ach[k])
        used += 1
        for i in range(horizon):
            actual = float(ach[k + i + 1])
            ev[i] = max(ev[i], abs(actual - float(pred[i])))
            acc += float(dt_arr[k + i + 1]) * (actual - float(pred[i]))
            ex[i] = max(ex[i], abs(acc))
            dts = float(dt_arr[k + i + 1])
            acc_plus_act += dts * max(actual, 0.0)
            acc_plus_hat += dts * max(float(pred[i]), 0.0)
            ex_plus[i] = max(ex_plus[i], max(acc_plus_act - acc_plus_hat, 0.0))
            if dts > 1e-9:
                a_act = (actual - prev_act) / dts
                a_hat = (float(pred[i]) - prev_hat) / dts
                ea[i] = max(ea[i], abs(a_act - a_hat))
            prev_act = actual
            prev_hat = float(pred[i])
    return ev, ex, ex_plus, ea, used


def _fit_arx(cmd: np.ndarray, ach: np.ndarray, na: int = 3, nb: int = 3) -> np.ndarray | None:
    rows = []
    rhs = []
    for k in range(max(na, nb), int(cmd.size)):
        phi = [float(ach[k - 1 - i]) for i in range(na)] + [
            float(cmd[k - i]) for i in range(nb)
        ]
        rows.append(phi)
        rhs.append(float(ach[k]))
    if len(rows) < 20:
        return None
    theta, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    return np.asarray(theta, dtype=float)


def _snap_v_cover_m_s(v0: float) -> float:
    mm = abs(float(v0)) * 1000.0
    if mm <= 1e-9:
        return 0.0
    return math.ceil(mm / 10.0 - 1e-12) * 0.010


_STOP_COVER_KEYS = (
    "v0_m_s",
    "a0_m_s2",
    "q_remain_m",
    "u_prev_m_s",
    "a_cmd_m_s2",
    "q_front_m_s",
)


def _stop_point_covers(hi: dict[str, float], lo: dict[str, float]) -> bool:
    return all(
        float(hi.get(k, 0.0)) + 1e-12 >= float(lo.get(k, 0.0))
        for k in _STOP_COVER_KEYS
    )


def _monotonic_stop_bins(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Covering closure on ``(v,a,q_remain,u_prev,a_cmd,q_front)``.

    Each event stays a point.  A dominating corner inflates ``dx`` / ``N_b``
    so a high-``a`` or high-``u_prev`` short stop cannot undercut a longer
    coast.  ``q_remain`` is still not a proven permutation abstraction.
    """
    pts: list[dict[str, float]] = []
    for r in rows:
        if float(r.get("gap_invalid", 0.0)) >= 0.5:
            continue
        if float(r.get("reached_T", 0.0)) < 0.5:
            continue
        if not math.isfinite(float(r.get("dx_press", float("nan")))):
            continue
        if not math.isfinite(float(r.get("n_b", float("nan")))):
            continue
        pts.append(
            {
                "v0_m_s": _snap_v_cover_m_s(float(r["v0"])),
                "a0_m_s2": max(float(r["a0"]), 0.0),
                "q_press_m_s": max(float(r.get("q_press", 0.0)), 0.0),
                "q_remain_m": max(float(r.get("q_remain_m", 0.0)), 0.0),
                "u_prev_m_s": max(float(r.get("u_prev", 0.0)), 0.0),
                "a_cmd_m_s2": max(float(r.get("a_cmd", 0.0)), 0.0),
                "q_front_m_s": max(float(r.get("q_front", 0.0)), 0.0),
                "dx_ub_m": float(r["dx_press"]),
                "n_b": float(r["n_b"]),
            }
        )
    for i, pi in enumerate(pts):
        dx = float(pi["dx_ub_m"])
        nb = float(pi["n_b"])
        for pj in pts:
            if _stop_point_covers(pi, pj):
                dx = max(dx, float(pj["dx_ub_m"]))
                nb = max(nb, float(pj["n_b"]))
        pts[i] = {**pi, "dx_ub_m": dx, "n_b": nb}
    kept: list[dict[str, float]] = []
    for i, pi in enumerate(pts):
        dominated = False
        for j, pj in enumerate(pts):
            if i == j:
                continue
            if (
                _stop_point_covers(pj, pi)
                and float(pj["dx_ub_m"]) + 1e-12 >= float(pi["dx_ub_m"])
                and float(pj["n_b"]) + 1e-12 >= float(pi["n_b"])
                and (
                    any(
                        float(pj[k]) > float(pi[k]) + 1e-12
                        for k in _STOP_COVER_KEYS
                    )
                    or float(pj["dx_ub_m"]) > float(pi["dx_ub_m"]) + 1e-12
                    or float(pj["n_b"]) > float(pi["n_b"]) + 1e-12
                )
            ):
                dominated = True
                break
        if not dominated:
            kept.append(pi)
    kept.sort(
        key=lambda b: (
            b["v0_m_s"],
            b["a0_m_s2"],
            b["q_remain_m"],
            b["u_prev_m_s"],
            b["a_cmd_m_s2"],
            b["q_front_m_s"],
        )
    )
    return kept


def _lookup_stop_cover(
    bins: list[dict[str, float]],
    *,
    v0: float,
    a0: float,
    q_remain_m: float,
    u_prev: float = 0.0,
    a_cmd: float = 0.0,
    q_front: float = 0.0,
) -> tuple[float, float]:
    query = {
        "v0_m_s": abs(float(v0)),
        "a0_m_s2": max(float(a0), 0.0),
        "q_remain_m": max(float(q_remain_m), 0.0),
        "u_prev_m_s": max(float(u_prev), 0.0),
        "a_cmd_m_s2": max(float(a_cmd), 0.0),
        "q_front_m_s": max(float(q_front), 0.0),
    }
    covering = [b for b in bins if _stop_point_covers(b, query)]
    if not covering:
        return float("inf"), float("inf")
    best = min(covering, key=lambda b: float(b["dx_ub_m"]))
    return float(best["dx_ub_m"]), float(best.get("n_b", 0.0))


def _validate_stop_lookup(
    bins: list[dict[str, float]],
    events: list[dict[str, float]],
    *,
    n_trigger: int | None = None,
    n_aligned: int | None = None,
    n_missed: int = 0,
) -> dict[str, int]:
    over_dx = 0
    over_nb = 0
    uncovered = 0
    terminal_fail = 0
    n_gap = 0
    n_invalid = 0
    n_checked = 0
    for ev in events:
        if float(ev.get("gap_invalid", 0.0)) >= 0.5:
            n_gap += 1
            continue
        if not math.isfinite(float(ev.get("dx_press", float("nan")))):
            n_invalid += 1
            continue
        n_checked += 1
        reached = float(ev.get("reached_T", 0.0)) >= 0.5
        nb = float(ev.get("n_b", float("nan")))
        if (not reached) or (not math.isfinite(nb)):
            terminal_fail += 1
            continue
        d_ub, n_ub = _lookup_stop_cover(
            bins,
            v0=float(ev["v0"]),
            a0=float(ev["a0"]),
            q_remain_m=float(ev.get("q_remain_m", 0.0)),
            u_prev=float(ev.get("u_prev", 0.0)),
            a_cmd=float(ev.get("a_cmd", 0.0)),
            q_front=float(ev.get("q_front", 0.0)),
        )
        if not math.isfinite(d_ub):
            uncovered += 1
            continue
        if float(ev["dx_press"]) > d_ub + 1e-12:
            over_dx += 1
        if (not math.isfinite(n_ub)) or nb > n_ub + 1e-12:
            over_nb += 1
    n_trig = int(n_trigger) if n_trigger is not None else len(events)
    n_al = int(n_aligned) if n_aligned is not None else len(events)
    n_valid = n_checked
    complete = int(
        n_trig == n_al == n_valid == n_checked
        and n_trig > 0
        and int(n_missed) == 0
        and n_gap == 0
        and n_invalid == 0
        and terminal_fail == 0
        and uncovered == 0
        and over_dx == 0
        and over_nb == 0
    )
    return {
        "n": n_checked,
        "n_trigger": n_trig,
        "n_aligned": n_al,
        "n_valid": n_valid,
        "n_checked": n_checked,
        "n_missed": int(n_missed),
        "n_gap": n_gap,
        "n_invalid": n_invalid,
        "over_dx": over_dx,
        "over_nb": over_nb,
        "uncovered": uncovered,
        "terminal_fail": terminal_fail,
        "complete": complete,
    }


def _stop_dx_yaml_block(rows: list[dict[str, float]], *, source: str) -> str:
    bins = _monotonic_stop_bins(rows)
    lines = [
        "safety_shield:",
        "  stop_dx_ub:",
        "    certified: false",
        f"    source: {source}",
        "    note: plant-ID or unvalidated backup replay; not Δx_b^ub until",
        "      independent val covers every event and certified is set true.",
        "      Covering on (v0,a0,q_remain,u_prev,a_cmd,q_front).",
        "      q_remain is not a proven delay-queue permutation abstraction.",
        "      a0 is [a_actual]+.  q_remain_m is dt Σ [u]+ of the delay line.",
        "      D_b covers backup-from-now; the shield uses Δx_1(u(λ))+D_b(ξ_1).",
        "    bins:",
    ]
    if not bins:
        lines.append("      []")
    for b in bins:
        lines.append(
            "      - "
            f"{{v0_m_s: {b['v0_m_s']:.4f}, a0_m_s2: {b['a0_m_s2']:.4f}, "
            f"q_press_m_s: {b['q_press_m_s']:.4f}, "
            f"q_remain_m: {b['q_remain_m']:.8f}, "
            f"u_prev_m_s: {b['u_prev_m_s']:.4f}, "
            f"a_cmd_m_s2: {b['a_cmd_m_s2']:.4f}, "
            f"q_front_m_s: {b['q_front_m_s']:.4f}, "
            f"dx_ub_m: {b['dx_ub_m']:.7f}, n_b: {int(b['n_b'])}}}"
        )
    return "\n".join(lines) + "\n"


def _command_stop_edges(cmd: np.ndarray, *, dt: float) -> list[tuple[str, int]]:
    min_drop = max(0.5 * 1.20 * max(dt, 1e-4), 0.001)
    edges: list[tuple[str, int]] = []
    in_backup = False
    for k in range(1, cmd.size):
        prev, now = float(cmd[k - 1]), float(cmd[k])
        if prev > 0.008 and abs(now) <= 0.002:
            edges.append(("stop", k))
            in_backup = False
        elif prev > 0.008 and now < -0.020:
            edges.append(("reverse", k))
            in_backup = False
        elif prev > 0.008 and (prev - now) >= min_drop and now > -0.020:
            if not in_backup:
                edges.append(("backup", k))
                in_backup = True
        elif abs(now) <= 0.002:
            in_backup = False
    return edges


def _event_log_edges(
    cmd: np.ndarray,
    t: np.ndarray,
    event_log: Path,
) -> tuple[list[tuple[str, int]], int, int]:
    triggers = _event_trigger_rows(_load_event_log(Path(event_log)))
    edges: list[tuple[str, int]] = []
    search_from = 1
    missed = 0
    for trig in triggers:
        idx = _align_trigger_index(cmd, t, trig, search_from=search_from)
        if idx is None:
            missed += 1
            continue
        edges.append(("backup", int(idx)))
        search_from = int(idx) + 1
    return edges, len(triggers), missed


def analyze_stop_reverse(
    path: Path,
    *,
    horizon: int = 40,
    settle_m_s: float = 0.003,
    val_path: Path | None = None,
    input_column: str = "twist_vz",
    write_yaml: Path | None = None,
    dt_gap_s: float = 0.050,
    event_log: Path | None = None,
    val_event_log: Path | None = None,
) -> int:
    """Plant-ID stop metrics.  Not a backup certificate unless input is u_b."""
    cmd, ach, t, twist, sent = _load_twist_csv(path, input_column=input_column)
    if cmd.size < 80:
        print("[ERR] not enough finite command samples", flush=True)
        return 1
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.005
    dt_arr_all = _step_dt(t, dt)
    n_twist = int(np.sum(np.isfinite(twist)))
    n_sent = int(np.sum(np.isfinite(sent)))
    print(
        f"[ID-SR] input_column={input_column}  twist_vz finite={n_twist}  "
        f"u_sent finite={n_sent}",
        flush=True,
    )
    if input_column == "twist_vz":
        print(
            "[ID-SR] SERVO_TWIST plant ID uses twist_vz.  "
            "u_sent may be empty or stale in this mode; do not prefer it.",
            flush=True,
        )
    else:
        print(
            "[ID-SR] using u_sent (backup replay).  Instantaneous →0/−40 "
            "on twist_vz is not this column.",
            flush=True,
        )
    print(
        f"[ID-SR] timestamp dt median={1e3 * dt:.2f} ms  "
        f"max={1e3 * float(np.max(dt_arr_all)):.2f} ms  "
        f"gap_limit={1e3 * dt_gap_s:.0f} ms (gaps invalidate the event)",
        flush=True,
    )
    print(
        "[ID-SR] this CSV is plant ID unless the logged command equals the "
        "a/j-limited backup.  Reverse N_b includes any held −40 mm/s and is "
        "not backup-to-terminal time.",
        flush=True,
    )
    edges: list[tuple[str, int]] = []
    if event_log is not None and Path(event_log).is_file():
        edges, n_trig, missed = _event_log_edges(cmd, t, Path(event_log))
        print(
            f"[ID-SR] event-log={event_log} triggers={n_trig} "
            f"aligned={len(edges)} missed={missed} "
            "(explicit trigger, not a 3 mm/s edge)",
            flush=True,
        )
    else:
        edges = _command_stop_edges(cmd, dt=dt)
        if event_log is not None:
            print(
                f"[ID-SR] event-log {event_log} missing; falling back to "
                "command edges (may miss the first jerk-limited ticks)",
                flush=True,
            )
    if not edges:
        print("[ERR] no +v→0 / +v→−40 / backup-decel edges in this CSV", flush=True)
        return 1
    print(
        f"[ID-SR] file={path} n={cmd.size} dt≈{dt*1000:.2f} ms edges={len(edges)}",
        flush=True,
    )
    by_kind: dict[str, list[dict[str, float]]] = {
        "stop": [],
        "reverse": [],
        "backup": [],
    }
    expand = 0
    gap_n = 0
    for kind, idx in edges:
        met = _event_metrics(
            cmd,
            ach,
            t,
            start=idx,
            settle_m_s=settle_m_s,
            horizon=horizon,
            end=_event_end(cmd, idx),
            dt_gap_s=dt_gap_s,
        )
        if not met:
            continue
        if float(met.get("gap_invalid", 0.0)) >= 0.5:
            gap_n += 1
            print(
                f"[ID-SR] {kind}@{idx} invalid: timestamp gap > {1e3*dt_gap_s:.0f} ms",
                flush=True,
            )
            continue
        by_kind[kind].append(met)
        if not met["reached_T"] or (
            math.isfinite(met["n_b"]) and met["n_b"] > horizon
        ):
            expand += 1
    if gap_n:
        print(f"[ID-SR] dropped {gap_n} events with timestamp gaps", flush=True)
    for kind, rows in by_kind.items():
        if not rows:
            print(f"[ID-SR] {kind}: none", flush=True)
            continue
        dx = np.array([r["dx_press"] for r in rows])
        n_press = np.array([r["n_press"] for r in rows])
        nb = np.array([r["n_b"] for r in rows])
        reached = np.array([r["reached_T"] for r in rows])
        v0 = np.array([r["v0"] for r in rows])
        finite_nb = nb[np.isfinite(nb)]
        print(
            f"[ID-SR] {kind}: n={len(rows)}  reached_T={int(np.sum(reached))}/{len(rows)}  "
            f"Δx^+ p50={1e3*float(np.median(dx)):.2f} mm  "
            f"p95={1e3*float(np.percentile(dx, 95)):.2f} mm  "
            f"N_press p50={float(np.median(n_press)):.1f}  "
            f"N_b p50={float(np.median(finite_nb)) if finite_nb.size else float('nan'):.1f}  "
            f"|v0| p50={1e3*float(np.median(np.abs(v0))):.1f} mm/s",
            flush=True,
        )
        bins: dict[int, list[float]] = {}
        for r in rows:
            key = int(1000.0 * abs(float(r["v0"])) // 10.0) * 10
            bins.setdefault(key, []).append(float(r["dx_press"]))
        parts = [
            f"{lo}–{lo+10}mm/s:{1e3*max(vals):.2f}mm"
            for lo, vals in sorted(bins.items())
        ]
        print(
            f"[ID-SR] {kind} Δx^+(v0) by |v0| bin (max, not a global max): "
            + "; ".join(parts),
            flush=True,
        )
        if kind == "reverse":
            print(
                "[ID-SR] reverse N_b includes held −40 mm/s; not shield N_b.",
                flush=True,
            )
        if int(np.sum(reached)) < len(rows):
            print(
                f"[ID-SR] {kind}: {len(rows) - int(np.sum(reached))} events never "
                f"entered the hold set.  Expand --horizon (now {horizon}); "
                "do not set N_b = horizon.",
                flush=True,
            )
    if expand:
        print(
            f"[ID-SR] {expand} events need a longer horizon or did not reach T.",
            flush=True,
        )
    table_rows = by_kind["backup"] or by_kind["stop"]
    table_src = "backup_replay" if by_kind["backup"] else "plant_step_stop"
    yaml_block = _stop_dx_yaml_block(table_rows, source=table_src)
    print(yaml_block, flush=True)
    print(
        "[ID-SR] copy stop_dx_ub into hybrid_motion.safety_shield only as a "
        "development table.  Leave certified: false until backup replay "
        "and independent val pass.  a0 is [a_actual]+; q_remain_m is dt Σ [u]+. "
        "Table also stores u_prev, a_cmd, q_front.  Still not certified. "
        "The shield uses Δx_1(u(λ))+D_b(ξ_1), not max(model, D_b(ξ)).",
        flush=True,
    )
    if write_yaml is not None:
        write_yaml.write_text(yaml_block)
        print(f"[ID-SR] wrote {write_yaml}", flush=True)

    bins_now = _monotonic_stop_bins(table_rows)
    independent = val_path is not None
    val_lookup_fail = False
    if independent:
        cmd_val, ach_val, t_val, _twv, _sev = _load_twist_csv(
            val_path, input_column=input_column
        )
        dt_val = float(np.median(np.diff(t_val))) if t_val.size > 2 else dt
        used_val_log = False
        if val_event_log is not None and Path(val_event_log).is_file():
            val_edges, n_vt, n_vm = _event_log_edges(
                cmd_val, t_val, Path(val_event_log)
            )
            used_val_log = True
            print(
                f"[ID-SR] val-event-log={val_event_log} triggers={n_vt} "
                f"aligned={len(val_edges)} missed={n_vm}",
                flush=True,
            )
        else:
            val_edges = _command_stop_edges(cmd_val, dt=dt_val)
            if table_src == "backup_replay":
                print(
                    "[ID-SR] FAIL: backup table needs --val-event-log.  "
                    "Command edges can miss the first jerk-limited ticks.  "
                    "Leave certified: false.",
                    flush=True,
                )
            elif val_event_log is not None:
                print(
                    f"[ID-SR] val-event-log {val_event_log} missing; "
                    "using command edges.",
                    flush=True,
                )
        val_events: list[dict[str, float]] = []
        n_reverse = 0
        for kind, idx in val_edges:
            if kind == "reverse":
                n_reverse += 1
                continue
            met = _event_metrics(
                cmd_val,
                ach_val,
                t_val,
                start=idx,
                settle_m_s=settle_m_s,
                horizon=horizon,
                end=_event_end(cmd_val, idx),
                dt_gap_s=dt_gap_s,
            )
            if not met:
                val_events.append(
                    {
                        "v0": 0.0,
                        "a0": 0.0,
                        "dx_press": float("nan"),
                        "n_b": float("nan"),
                        "reached_T": 0.0,
                        "gap_invalid": 1.0,
                    }
                )
            else:
                val_events.append(met)
        if used_val_log:
            n_trigger = int(n_vt)
            n_aligned = int(len(val_edges))
            n_missed = int(n_vm)
        else:
            n_trigger = int(len(val_edges) - n_reverse)
            n_aligned = int(len(val_events))
            n_missed = 0
        report = _validate_stop_lookup(
            bins_now,
            val_events,
            n_trigger=n_trigger,
            n_aligned=n_aligned,
            n_missed=n_missed,
        )
        print(
            f"[ID-SR] stop-lookup val trigger={report['n_trigger']} "
            f"aligned={report['n_aligned']} valid={report['n_valid']} "
            f"checked={report['n_checked']} missed={report['n_missed']} "
            f"gap={report['n_gap']} over_dx={report['over_dx']} "
            f"over_Nb={report['over_nb']} uncovered={report['uncovered']} "
            f"terminal_fail={report['terminal_fail']}",
            flush=True,
        )
        lookup_fail = bool(
            report["complete"] == 0
            or (table_src == "backup_replay" and not used_val_log)
        )
        val_lookup_fail = lookup_fail
        if report["n_trigger"] == 0:
            print(
                "[ID-SR] FAIL: --val produced no stop/backup events for lookup.",
                flush=True,
            )
        elif lookup_fail:
            print(
                "[ID-SR] FAIL: lookup val needs "
                "N_trigger=N_aligned=N_valid=N_checked and "
                "N_missed=N_gap=N_terminalFail=N_uncovered=N_overDx="
                "N_overNb=0.  A gapped or unaligned event is a fail, "
                "not a skip.  Leave certified: false.",
                flush=True,
            )
        else:
            print(
                "[ID-SR] lookup covers every independent stop/backup event "
                "(Δx^+ ≤ D_b^ub and N_b ≤ N_b^ub).  Still do not set "
                "certified: true: terminal D_T is a box-sup, last ē_v(N) "
                "is not ē_v(∞), and q_remain is not a proven queue "
                "permutation abstraction.",
                flush=True,
            )
    else:
        print(
            "[ID-SR] WARNING: stop-lookup events were not checked on an "
            "independent --val file.  Plant 70/30 is not that check.",
            flush=True,
        )

    independent = val_path is not None
    if independent:
        cmd_va, ach_va, t_va, _tw, _se = _load_twist_csv(
            val_path, input_column=input_column
        )
        cmd_tr, ach_tr, t_tr = cmd, ach, t
        print(f"[ID-SR] independent validation file={val_path}", flush=True)
    else:
        split = max(int(0.7 * cmd.size), 80)
        cmd_tr, ach_tr, t_tr = cmd[:split], ach[:split], t[:split]
        cmd_va, ach_va, t_va = cmd[split:], ach[split:], t[split:]
        if cmd_va.size < horizon + 5:
            mid = cmd.size // 2
            cmd_tr, ach_tr, t_tr = cmd[:mid], ach[:mid], t[:mid]
            cmd_va, ach_va, t_va = cmd[mid:], ach[mid:], t[mid:]
        print(
            "[ID-SR] WARNING: same-run 70/30 split is development only. "
            "A certificate needs a separate validation run (pose, payload, "
            "and preferably another date/boot).  Sequence is speed-increasing, "
            "so this split can become low-speed train / high-speed val.",
            flush=True,
        )

    cmd0 = cmd_tr - np.mean(cmd_tr)
    ach0 = ach_tr - np.mean(ach_tr)
    corr = np.correlate(ach0, cmd0, mode="full")
    lags = np.arange(-cmd0.size + 1, ach0.size)
    lag = max(int(lags[int(np.argmax(corr))]), 0)
    dt_tr = _step_dt(t_tr, dt)
    best_tp = 0.020
    best_err = float("inf")
    dt_med = float(np.median(dt_tr))
    for tp in np.linspace(0.010, 0.080, 15):
        pred = _simulate_first_order(
            cmd_tr, delay_steps=lag, tp_s=float(tp), dt=dt_med
        )
        err = float(np.mean((pred - ach_tr) ** 2))
        if err < best_err:
            best_err = err
            best_tp = float(tp)
    best_wn, best_z, best_e2 = 40.0, 0.8, float("inf")
    dt_va = _step_dt(t_va, dt)

    def _sim_second_train(u: np.ndarray, wn: float, z: float) -> np.ndarray:
        y = np.zeros_like(u)
        yd = 0.0
        for i, ui in enumerate(u):
            ydd = (wn * wn) * (float(ui) - (y[i - 1] if i else 0.0)) - 2.0 * z * wn * yd
            yd += ydd * dt_med
            y[i] = (y[i - 1] if i else 0.0) + yd * dt_med
        return y

    for wn in np.linspace(20.0, 80.0, 8):
        for z in (0.5, 0.8, 1.1, 1.4):
            pred = _sim_second_train(cmd_tr, float(wn), float(z))
            err = float(np.mean((pred - ach_tr) ** 2))
            if err < best_e2:
                best_e2 = err
                best_wn, best_z = float(wn), float(z)
    theta = _fit_arx(cmd_tr, ach_tr)
    step = max(horizon // 4, 1)
    origins = list(range(0, max(cmd_va.size - horizon - 2, 1), step))
    ev_f, ex_f, ex_f_plus, ea_f, n_f = _open_loop_envelopes(
        cmd_va,
        ach_va,
        t_va,
        origins=origins,
        horizon=horizon,
        rollout_fn=lambda k: _rollout_fopdt(
            cmd_va, ach_va, k, horizon, lag, best_tp, dt_va
        ),
        origin_dt_gap_s=dt_gap_s,
    )
    ev_2, ex_2, ex_2_plus, ea_2, _n_2 = _open_loop_envelopes(
        cmd_va,
        ach_va,
        t_va,
        origins=origins,
        horizon=horizon,
        rollout_fn=lambda k: _rollout_second(
            cmd_va, ach_va, k, horizon, best_wn, best_z, dt_va
        ),
        origin_dt_gap_s=dt_gap_s,
    )
    if theta is None:
        ev_a, ex_a, ex_a_plus, ea_arx = (
            [0.0] * horizon,
            [0.0] * horizon,
            [0.0] * horizon,
            [0.0] * horizon,
        )
    else:
        ev_a, ex_a, ex_a_plus, ea_arx, _n_a = _open_loop_envelopes(
            cmd_va,
            ach_va,
            t_va,
            origins=origins,
            horizon=horizon,
            rollout_fn=lambda k: _rollout_arx(
                cmd_va, ach_va, k, horizon, theta, 3, 3
            ),
            origin_dt_gap_s=dt_gap_s,
        )
    def _emax(xs: list[float]) -> float:
        return max(xs) if xs else float("nan")

    print(
        f"[ID-SR] open-loop ê(k+i|k) origins={n_f} (same starts for all models)  "
        f"FOPDT(T0={lag * dt * 1000:.1f}ms,Tp={best_tp * 1000:.1f}ms) "
        f"ê_v max={_emax(ev_f):.4f} ê_x max={1e3 * _emax(ex_f):.2f}mm "
        f"ê_x,+ max={1e3 * _emax(ex_f_plus):.2f}mm ê_a max={_emax(ea_f):.3f}  "
        f"2nd(wn={best_wn:.1f},z={best_z:.2f}) "
        f"ê_v max={_emax(ev_2):.4f} ê_x max={1e3 * _emax(ex_2):.2f}mm "
        f"ê_x,+ max={1e3 * _emax(ex_2_plus):.2f}mm ê_a max={_emax(ea_2):.3f}  "
        f"ARX ê_v max={_emax(ev_a):.4f} ê_x max={1e3 * _emax(ex_a):.2f}mm "
        f"ê_x,+ max={1e3 * _emax(ex_a_plus):.2f}mm ê_a max={_emax(ea_arx):.3f}",
        flush=True,
    )
    print(
        "[ID-SR] pick the smallest *open-loop* 1–H envelope.  Use ê_{x,+} "
        "for force indent; signed ê_x can cancel.  Correlation is not the "
        "criterion.  Do not shrink ê on the validation set.",
        flush=True,
    )
    print(
        "[ID-SR] first-contact feasible only if "
        "F_enter+ΔF_unc+K_ub Δx_b^ub(v0,a0,q0)+ΔF_active^ub ≤ F_max.  "
        "Plant-step Δx^+ is not that bound.  If the inequality fails at low "
        "v0, do not shrink the tube.",
        flush=True,
    )
    return 1 if val_lookup_fail else 0


def _finite_col(rows: list[dict[str, str]], key: str) -> np.ndarray:
    out = np.empty(len(rows), dtype=float)
    for i, row in enumerate(rows):
        try:
            out[i] = float(row.get(key) or "nan")
        except (TypeError, ValueError):
            out[i] = float("nan")
    return out


def _load_tool_z_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Command/measurement pair in the same tool-Z frame."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    t = _finite_col(rows, "t_wall_s")
    u = _finite_col(rows, "vel_ff_vz")
    y = _finite_col(rows, "vz_achieved_tool")
    mask = np.isfinite(t) & np.isfinite(u) & np.isfinite(y)
    t, u, y = t[mask], u[mask], y[mask]
    if t.size < 8:
        raise ValueError(f"{path} has no finite vel_ff_vz/vz_achieved_tool pair")
    order = np.argsort(t, kind="stable")
    return t[order], u[order], y[order]


def _resample_uniform(
    t: np.ndarray, u: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dt = float(np.median(np.diff(t))) if t.size > 2 else 0.005
    if not math.isfinite(dt) or dt <= 1e-4:
        dt = 0.005
    t0 = float(t[0])
    t1 = float(t[-1])
    n = max(int(round((t1 - t0) / dt)) + 1, 8)
    tu = t0 + dt * np.arange(n)
    return tu, np.interp(tu, t, u), np.interp(tu, t, y), dt


def _simulate_fopdt_k(
    cmd: np.ndarray,
    *,
    delay_steps: int,
    tp_s: float,
    gain: float,
    dt: float,
) -> np.ndarray:
    y = _simulate_first_order(
        cmd, delay_steps=delay_steps, tp_s=tp_s, dt=dt
    )
    return float(gain) * y


def _fit_fopdt(
    cmd: np.ndarray, ach: np.ndarray, dt: float
) -> tuple[int, float, float, float]:
    best = (float("inf"), 10, 0.020, 1.0)
    cmd0 = cmd - np.mean(cmd)
    ach0 = ach - np.mean(ach)
    if cmd0.size > 4 and float(np.std(cmd0)) > 1e-8:
        corr = np.correlate(ach0, cmd0, mode="full")
        lags = np.arange(-cmd0.size + 1, ach0.size)
        lag0 = int(max(lags[int(np.argmax(corr))], 0))
    else:
        lag0 = 10
    # Keep the correlation delay (communication Γ_d).  Do not trade it
    # against Tp: that inflates the residual time constant and the
    # derivative in T_n^{-1}.
    delay_lo = max(lag0 - 2, 0)
    delay_hi = lag0 + 2
    for delay in range(delay_lo, delay_hi + 1):
        for tp in np.linspace(0.008, 0.040, 17):
            pred = _simulate_first_order(
                cmd, delay_steps=delay, tp_s=float(tp), dt=dt
            )
            den = float(np.dot(pred, pred))
            gain = float(np.dot(pred, ach) / den) if den > 1e-12 else 1.0
            gain = float(np.clip(gain, 0.7, 1.3))
            err = float(np.mean((gain * pred - ach) ** 2))
            if err < best[0]:
                best = (err, delay, float(tp), gain)
    return best[1], best[2], best[3], math.sqrt(best[0])


def _simulate_second_zero(
    cmd: np.ndarray,
    *,
    delay_steps: int,
    wn: float,
    z: float,
    z_zero: float,
    dt: float,
) -> np.ndarray:
    """T(s) = (s/z_zero + 1) / (s^2/wn^2 + 2 z s/wn + 1) after a delay."""
    y = np.zeros_like(cmd)
    yd = 0.0
    delay = [0.0] * max(int(delay_steps), 0)
    wn = max(float(wn), 1.0)
    z = max(float(z), 0.2)
    z_zero = max(float(z_zero), 1.0)
    for i, u in enumerate(cmd):
        if delay:
            u_app = delay[0]
            delay = delay[1:] + [float(u)]
        else:
            u_app = float(u)
        prev = y[i - 1] if i else 0.0
        # Input zero: u_f = u + ú/z_zero, ú ≈ (u - u_prev)/dt
        u_prev = delay[-1] if delay else (float(cmd[i - 1]) if i else u_app)
        u_f = u_app + (u_app - u_prev) / max(dt * z_zero, 1e-6)
        ydd = (wn * wn) * (u_f - prev) - 2.0 * z * wn * yd
        yd += ydd * dt
        y[i] = prev + yd * dt
    return y


def _band_mag_phase_err(
    cmd: np.ndarray,
    ach: np.ndarray,
    pred: np.ndarray,
    dt: float,
    f_lo: float = 0.2,
    f_hi: float = 5.0,
) -> tuple[float, float]:
    n = int(cmd.size)
    if n < 32:
        return float("nan"), float("nan")
    win = np.hanning(n)
    freq = np.fft.rfftfreq(n, dt)
    band = (freq >= f_lo) & (freq <= f_hi)
    if not np.any(band):
        return float("nan"), float("nan")
    u = np.fft.rfft(cmd * win)
    y = np.fft.rfft(ach * win)
    p = np.fft.rfft(pred * win)
    eps = 1e-12
    mag_err = float(
        np.mean(np.abs(np.abs(y[band] / (u[band] + eps)) - np.abs(p[band] / (u[band] + eps))))
    )
    ang_y = np.angle(y[band] / (u[band] + eps))
    ang_p = np.angle(p[band] / (u[band] + eps))
    phase_err = float(np.mean(np.abs(np.angle(np.exp(1j * (ang_y - ang_p))))))
    return mag_err, phase_err


def analyze_tn(
    path: Path,
    *,
    val_path: Path | None = None,
    write_yaml: Path | None = None,
    fit_speed_max_m_s: float = 0.025,
) -> int:
    """Fit a shadow-only Γ_d + T_n candidate on tool-Z step logs.

    Step validation does not certify phase in the intended Q band.  Active
    operation still requires PRBS/multisine (or an equivalent FRF experiment)
    and leaves ``active_model_validated`` false.
    """
    t, u, y = _load_tool_z_csv(path)
    tu, uu, yy, dt = _resample_uniform(t, u, y)
    broadband_excitation = np.unique(np.round(uu, decimals=5)).size > 100
    # Fit on the contiguous record.  Masking low-speed samples would
    # stitch distant ticks together and destroy the delay.
    delay, tp, gain, train_rmse = _fit_fopdt(uu, yy, dt)
    print(
        f"[ID-TN] corr_lag={delay} ticks  (Γ_d pinned near cross-correlation)",
        flush=True,
    )
    pred = _simulate_fopdt_k(
        uu, delay_steps=delay, tp_s=tp, gain=gain, dt=dt
    )
    low = np.abs(uu) <= fit_speed_max_m_s + 1e-9
    train_low = (
        math.sqrt(float(np.mean((pred[low] - yy[low]) ** 2)))
        if np.any(low)
        else train_rmse
    )
    train_all = math.sqrt(float(np.mean((pred - yy) ** 2)))
    mag_err, phase_err = _band_mag_phase_err(uu, yy, pred, dt)
    pole = math.exp(-dt / max(tp, 1e-4))
    print(
        f"[ID-TN] train={path}  dt={1e3 * dt:.2f}ms  "
        f"pair=vel_ff_vz→vz_achieved_tool  "
        f"excitation={'broadband/chirp' if broadband_excitation else 'step'}",
        flush=True,
    )
    print(
        f"[ID-TN] FOPDT T0={delay * dt * 1000:.1f}ms  Tp={tp * 1000:.1f}ms  "
        f"K={gain:.3f}  pole={pole:.4f}  min_phase=yes  dc≈{gain:.3f}",
        flush=True,
    )
    print(
        f"[ID-TN] train RMSE (≤{1e3 * fit_speed_max_m_s:.0f} mm/s)="
        f"{1e3 * train_low:.2f} mm/s  all-speeds={1e3 * train_all:.2f} mm/s  "
        f"step-spectrum diagnostic |T|err[0.2-5Hz]={mag_err:.3f}  "
        f"∠err={phase_err:.3f} rad",
        flush=True,
    )

    val_rmse = float("nan")
    if val_path is not None:
        tv, uv, yv = _load_tool_z_csv(val_path)
        _t2, uv, yv, dtv = _resample_uniform(tv, uv, yv)
        pred_v = _simulate_fopdt_k(
            uv, delay_steps=max(int(round(delay * dt / dtv)), 0),
            tp_s=tp, gain=gain, dt=dtv,
        )
        val_rmse = math.sqrt(float(np.mean((pred_v - yv) ** 2)))
        high = np.abs(uv) > 0.030
        high_rmse = (
            math.sqrt(float(np.mean((pred_v[high] - yv[high]) ** 2)))
            if np.any(high)
            else float("nan")
        )
        print(
            f"[ID-TN] val={val_path}  RMSE={1e3 * val_rmse:.2f} mm/s  "
            f"40/80 RMSE={1e3 * high_rmse:.2f} mm/s",
            flush=True,
        )

    best2 = float("inf")
    best2_p = (40.0, 0.8, 80.0)
    for wn in (25.0, 40.0, 55.0, 70.0):
        for z in (0.7, 1.0, 1.3):
            for z0 in (40.0, 80.0, 160.0):
                p2 = _simulate_second_zero(
                    uu, delay_steps=delay, wn=wn, z=z, z_zero=z0, dt=dt
                )
                err = float(np.mean((p2 - yy) ** 2))
                if err < best2:
                    best2 = err
                    best2_p = (wn, z, z0)
    second_rmse = math.sqrt(best2)
    choose_second = second_rmse + 1e-4 < 0.75 * train_all
    print(
        f"[ID-TN] 2nd+zero wn={best2_p[0]:.0f} z={best2_p[1]:.2f} "
        f"z0={best2_p[2]:.0f} RMSE={1e3 * second_rmse:.2f} mm/s  "
        f"{'selected' if choose_second else 'not selected (keep FOPDT)'}",
        flush=True,
    )
    t0_s = delay * dt
    yaml_block = (
        "hybrid_motion:\n"
        "  cdyob:\n"
        "    mode: shadow\n"
        f"    t0_s: {t0_s:.4f}\n"
        f"    tp_s: {tp:.4f}\n"
        "    omega_q_hz: 0.75\n"
        "    pn_m: 0.0\n"
        "    v_corr_max_m_s: 0.003\n"
        "    blend_s: 0.30\n"
        "    active_press_max_m_s: 0.010\n"
        "    active_retract_max_m_s: 0.010\n"
        "    active_q_max_hz: 1.0\n"
        "    active_force_ratio: 0.90\n"
        "    active_settle_speed_m_s: 0.003\n"
        "    active_settle_hold_s: 0.20\n"
        "    active_model_validated: false\n"
    )
    print("[ID-TN] yaml:\n" + yaml_block, flush=True)
    if write_yaml is not None:
        write_yaml.write_text(yaml_block)
        print(f"[ID-TN] wrote {write_yaml}", flush=True)
    if choose_second:
        print(
            "[ID-TN] WARNING: second-order looked better; still shipping FOPDT "
            "unless a chirp confirms the extra zero.",
            flush=True,
        )
    if broadband_excitation:
        print(
            "[ID-TN] ACTIVE BLOCKED: one broadband/chirp record is not an "
            "independent validation.  Repeat it in a separate log and compare "
            "the target-Q-band FRF before active_model_validated=true.",
            flush=True,
        )
    else:
        print(
            "[ID-TN] ACTIVE BLOCKED: stop/step data do not bound 1–3 Hz phase. "
            "Collect PRBS/multisine before setting active_model_validated=true.",
            flush=True,
        )
    return 0


def _col_or_nan(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float("nan")


def replay_cdyob(path: Path) -> int:
    """Shadow the paper CDYOB on a recorded hybrid CSV.  Not a closed-loop claim."""
    from rm75_control.control.admittance_common.cdyob import (
        CdyobConfig,
        CombinedDynamicsYob,
    )

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 20:
        print(f"[CDYOB-REPLAY] too few rows in {path}", flush=True)
        return 1
    cfg = CdyobConfig(
        mode="shadow",
        omega_q_hz=0.75,
        t0_s=0.050,
        tp_s=0.026,
        v_corr_max_m_s=0.003,
        blend_s=0.30,
    )
    yob = CombinedDynamicsYob(cfg)
    dt_med = 0.005
    t = _finite_col(rows, "t_wall_s")
    if np.isfinite(t).sum() > 4:
        dts = np.diff(t[np.isfinite(t)])
        dts = dts[(dts > 1e-4) & (dts < 0.05)]
        if dts.size:
            dt_med = float(np.median(dts))
    corrs: list[float] = []
    unclip: list[float] = []
    v_r_seen: list[float] = []
    dob_seen: list[float] = []
    sat = 0
    contact_n = 0
    for i, row in enumerate(rows):
        dt = _col_or_nan(row, "dt_actual_s")
        if not math.isfinite(dt) or dt <= 0.0:
            dt = dt_med
        v_nom = _col_or_nan(row, "v_force_z", "u_nom_raw", "vel_ff_vz")
        v_meas = _col_or_nan(row, "vz_achieved_tool")
        force = _col_or_nan(row, "fz")
        if not math.isfinite(v_nom):
            v_nom = 0.0
        if not math.isfinite(force):
            force = 0.0
        # Logged fz is tool-Z (press-negative on this stack).  Observer uses
        # press-positive force, same as controller.force_normal_filtered.
        force_n = -force if math.isfinite(force) else 0.0
        v_meas_n = (
            None if not math.isfinite(v_meas) else float(v_meas)
        )
        yob.update(
            float(v_nom),
            v_meas_m_s=v_meas_n,
            force_n=force_n,
            dt_s=float(dt),
            mass_z=1.0,
            damping_z=40.0,
            apply_scale=0.0,
        )
        sent = _col_or_nan(
            row, "u_sent", "u_nom_capped", "vel_ff_vz", "v_force_z"
        )
        yob.commit_sent(
            float(sent) if math.isfinite(sent) else float(v_nom),
            dt_s=float(dt),
        )
        tel = yob.telemetry
        state = str(row.get("physical_contact_state") or "")
        in_contact = state in (
            "contact",
            "confirmed",
            "held",
            "stable",
        ) or str(row.get("contact_present") or "0") in ("1", "true", "True")
        if in_contact or not any(
            str(row.get(k) or "").strip() for k in ("physical_contact_state",)
        ):
            unclip.append(float(tel.pert_unclipped))
            corrs.append(float(tel.pert_clipped))
            sat += int(bool(tel.saturated))
            v_r = _col_or_nan(row, "v_r_z")
            dob = _col_or_nan(row, "u_dob_z")
            if math.isfinite(v_r):
                v_r_seen.append(v_r)
            if math.isfinite(dob):
                dob_seen.append(dob)
        if in_contact:
            contact_n += 1
    arr = np.asarray(unclip, dtype=float)
    clip = np.asarray(corrs, dtype=float)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    p95 = float(np.percentile(np.abs(arr), 95)) if arr.size else 0.0
    # 2.73 Hz tone in the unclipped correction (offline, open-loop).
    tone = float("nan")
    if arr.size > 64:
        freq = np.fft.rfftfreq(arr.size, dt_med)
        spec = np.abs(np.fft.rfft(arr - np.mean(arr))) ** 2
        idx = int(np.argmin(np.abs(freq - 2.73)))
        tone = float(spec[idx]) if spec.size else float("nan")
    v_r_p95 = (
        float(np.percentile(np.abs(np.asarray(v_r_seen)), 95))
        if v_r_seen
        else 0.0
    )
    dob_p95 = (
        float(np.percentile(np.abs(np.asarray(dob_seen)), 95))
        if dob_seen
        else 0.0
    )
    baseline_known = bool(v_r_seen) and bool(dob_seen)
    baseline_compatible = (
        baseline_known and v_r_p95 < 1e-6 and dob_p95 < 1e-6
    )
    runtime_shadow_rows = sum(
        1
        for row in rows
        if str(row.get("cdyob_mode") or "").strip().lower() == "shadow"
        and math.isfinite(_col_or_nan(row, "cdyob_pert_unclipped"))
    )
    print(
        f"[CDYOB-REPLAY] {path.name}  n={len(rows)} used={arr.size} "
        f"contact_rows≈{contact_n}  "
        f"|pert| p95={1e3 * p95:.2f} mm/s  peak={1e3 * peak:.2f} mm/s  "
        f"clip={sat}/{arr.size}  2.73Hz pwr={tone:.3e}",
        flush=True,
    )
    print(
        "[CDYOB-REPLAY] A-only baseline="
        f"{'yes' if baseline_compatible else 'NO' if baseline_known else 'unknown'}  "
        f"|v_r|p95={1e3 * v_r_p95:.2f} mm/s  |u_dob|p95={dob_p95:.3f} N",
        flush=True,
    )
    print(
        f"[CDYOB-REPLAY] runtime shadow telemetry rows={runtime_shadow_rows}  "
        f"contact rows={contact_n}",
        flush=True,
    )
    print(
        "[CDYOB-REPLAY] open-loop shadow only.  Does not claim closed-loop "
        "suppression.  polarity check: +force (press) should not produce a "
        f"sustained +pert (mean={1e3 * float(np.mean(arr)):.2f} mm/s).",
        flush=True,
    )
    if not baseline_known or runtime_shadow_rows == 0:
        print(
            "[CDYOB-REPLAY] NOT A SHADOW RUN: controller CDYOB telemetry is "
            "blank (for example, plain servo_twist rather than hybrid force).",
            flush=True,
        )
    elif not baseline_compatible:
        print(
            "[CDYOB-REPLAY] NOT AN ACTIVE PREDICTOR: this log contains v_r/DOB. "
            "Record A-only off baseline, then A-only CDYOB shadow.",
            flush=True,
        )
    del clip
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="peirastic plant identification")
    parser.add_argument("--shm-prefix", default="")
    parser.add_argument("--hold-s", type=float, default=0.80)
    parser.add_argument("--rest-s", type=float, default=0.40)
    parser.add_argument("--chirp-s", type=float, default=8.0)
    parser.add_argument("--chirp-amp-mm-s", type=float, default=20.0)
    parser.add_argument("--chirp-f0", type=float, default=0.2)
    parser.add_argument("--chirp-f1", type=float, default=5.0)
    parser.add_argument(
        "--chirp-amps-mm-s",
        type=str,
        default="",
        help="comma list; with --air-campaign default 8,15,25",
    )
    parser.add_argument(
        "--air-campaign",
        action="store_true",
        help="dense 4–80 mm/s steps + three chirps (default 0.2–8 Hz; pass --chirp-f1 10 to sweep to 10 Hz); no stop-reverse",
    )
    parser.add_argument(
        "--tdpa-press",
        action="store_true",
        help=(
            "open-loop seek + constant press for De Stefano §IV sign check; "
            "force loop stays off — do not use hybrid F* tracking"
        ),
    )
    parser.add_argument(
        "--identify",
        action="store_true",
        help=(
            "write plant+contact ID from a finished air-campaign dir "
            "(--air-out) and --analyze-tdpa press CSVs; does not drive the arm"
        ),
    )
    parser.add_argument(
        "--analyze-air",
        type=str,
        default="",
        help="comma-separated CSVs; write required plots to --air-out",
    )
    parser.add_argument(
        "--analyze-tdpa",
        type=str,
        default="",
        help=(
            "score longest continuous press window on a --tdpa-press or "
            "hybrid CSV (needs tdpa_e_obs_j)"
        ),
    )
    parser.add_argument(
        "--air-out",
        type=str,
        default="",
        help="plot directory (default MD/todo_controller_logs/id_air_<stamp>)",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="",
        help="with --air-campaign, campaign CSV (default <air-out>/air_campaign.csv)",
    )
    parser.add_argument("--hz", type=float, default=200.0)
    parser.add_argument("--analyze", type=str, default="", help="analyse an existing CSV")
    parser.add_argument(
        "--analyze-stop",
        type=str,
        default="",
        help="analyse a --stop-reverse CSV for Δx^+ / N_press / N_b",
    )
    parser.add_argument(
        "--analyze-tn",
        type=str,
        default="",
        help="fit Γ_d + T_n from vel_ff_vz → vz_achieved_tool",
    )
    parser.add_argument(
        "--replay-cdyob",
        type=str,
        default="",
        help="shadow paper CDYOB on an existing hybrid CSV",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="",
        help="independent validation CSV for --analyze-stop (not a 70/30 split)",
    )
    parser.add_argument(
        "--input-column",
        choices=("twist_vz", "u_sent"),
        default="twist_vz",
        help="SERVO_TWIST plant ID uses twist_vz; backup replay may use u_sent",
    )
    parser.add_argument(
        "--dt-gap-s",
        type=float,
        default=0.050,
        help="timestamp gap that invalidates a stop event (not silently clipped)",
    )
    parser.add_argument(
        "--speeds-mm-s",
        type=str,
        default="",
        help="comma list for --stop-reverse / --backup-replay (default 10,20,40,80)",
    )
    parser.add_argument(
        "--backup-replay",
        action="store_true",
        help="play a/j-limited shield backup instead of instant →0/−40",
    )
    parser.add_argument(
        "--event-log",
        type=str,
        default="",
        help="backup-replay event log (write path, or --analyze-stop align file)",
    )
    parser.add_argument(
        "--window-a-csv",
        type=str,
        default="",
        help="optional Window A CSV for event-log t_wall notes (not 200 Hz v)",
    )
    parser.add_argument(
        "--motion-max-age-s",
        type=float,
        default=0.015,
        help="abort --backup-replay if feedback_age + SHM publish age exceeds this",
    )
    parser.add_argument(
        "--val-event-log",
        type=str,
        default="",
        help="independent backup val event log (not command edges)",
    )
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--write-yaml", type=str, default="")
    parser.add_argument("--error-margin", type=float, default=0.002)
    parser.add_argument(
        "--stop-reverse",
        action="store_true",
        help="run +10/20/40/80 → 0 and → −40 mm/s (settled and accel)",
    )
    parser.add_argument(
        "--chirp-only",
        action="store_true",
        help="run only the 0.2–5 Hz chirp; do not repeat step identification",
    )
    parser.add_argument(
        "--short-hold-s",
        type=float,
        default=0.050,
        help="nonzero-accel dwell before stop/reverse",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the sequence without talking to Window A",
    )
    args = parser.parse_args()
    if args.identify:
        from datetime import datetime

        from peirastic.apps.identify_air import write_identification_report

        air_dir = Path(args.air_out) if args.air_out else Path(
            "MD/todo_controller_logs/id_air_20260829_202437"
        )
        presses = [
            Path(p.strip())
            for p in str(args.analyze_tdpa).split(",")
            if p.strip()
        ]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.log_csv) if args.log_csv else Path(
            "MD/todo_controller_logs"
        ) / f"id_fit_{stamp}"
        report = write_identification_report(
            air_dir=air_dir,
            press_paths=presses,
            out_dir=out,
        )
        plant = report["plant"]
        print(
            f"[ID] out={out} T0={1e3 * plant['t0_s']:.1f} ms "
            f"Tp={1e3 * plant['tp_s']:.1f} ms "
            f"write_fopdt={plant['write_single_fopdt']} "
            f"td_band={plant['td_is_band']} "
            f"surfaces={report['tdpa_sign']['n_surfaces']} "
            f"sign={report['tdpa_sign']['ok']}",
            flush=True,
        )
        if not plant["write_single_fopdt"]:
            print("[ID] T0 spread > 8 ms: do not write a single FOPDT", flush=True)
        return 0 if report["tdpa_sign"]["ok"] or not presses else 2
    if args.analyze_tdpa:
        from peirastic.apps.identify_air import analyze_tdpa_contact

        verdict = analyze_tdpa_contact(Path(args.analyze_tdpa))
        print(f"[TDPA] {verdict}", flush=True)
        return 0 if verdict.get("ok") else 2
    if args.analyze_air:
        from datetime import datetime

        from peirastic.apps.identify_air import AIR_CHIRP_F1_HZ, analyze_air_paths

        paths = [Path(p.strip()) for p in str(args.analyze_air).split(",") if p.strip()]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.air_out) if args.air_out else Path(
            "MD/todo_controller_logs"
        ) / f"id_air_{stamp}"
        result = analyze_air_paths(
            paths,
            out_dir=out,
            chirp_f0_hz=float(args.chirp_f0),
            chirp_f1_hz=(
                AIR_CHIRP_F1_HZ
                if abs(float(args.chirp_f1) - 5.0) < 1e-12
                else float(args.chirp_f1)
            ),
        )
        print(
            f"[ID-AIR] out={out} edges={len(result.edges)} chirps={len(result.chirps)} "
            f"linear≤{result.linear_speed_mm_s:.1f} mm/s "
            f"T0_spread={1e3 * result.t0_spread_s:.1f} ms",
            flush=True,
        )
        if math.isfinite(result.t0_spread_s) and result.t0_spread_s > 0.008:
            print(
                "[ID-AIR] T0 spread > 8 ms: do not write a single FOPDT into yaml",
                flush=True,
            )
        return 0
    speeds = _parse_speeds_mm_s(args.speeds_mm_s)
    if args.analyze_tn:
        return analyze_tn(
            Path(args.analyze_tn),
            val_path=Path(args.val) if args.val else None,
            write_yaml=Path(args.write_yaml) if args.write_yaml else None,
        )
    if args.replay_cdyob:
        return replay_cdyob(Path(args.replay_cdyob))
    if args.analyze_stop:
        elog = Path(args.event_log) if args.event_log else None
        return analyze_stop_reverse(
            Path(args.analyze_stop),
            horizon=int(args.horizon),
            val_path=Path(args.val) if args.val else None,
            input_column=str(args.input_column),
            write_yaml=Path(args.write_yaml) if args.write_yaml else None,
            dt_gap_s=float(args.dt_gap_s),
            event_log=elog,
            val_event_log=Path(args.val_event_log) if args.val_event_log else None,
        )
    if args.analyze:
        return analyze_csv(
            Path(args.analyze),
            horizon=int(args.horizon),
            write_yaml=Path(args.write_yaml) if args.write_yaml else None,
            margin=float(args.error_margin),
        )
    if args.dry_run:
        if args.backup_replay:
            print(
                f"[DRY] backup-replay mm/s={speeds} hold={args.hold_s} "
                f"short={args.short_hold_s} rest={args.rest_s}",
                flush=True,
            )
        elif args.stop_reverse:
            print(
                f"[DRY] stop-reverse mm/s={speeds} →0 and →{REVERSE_MM_S} "
                f"hold={args.hold_s} short={args.short_hold_s} rest={args.rest_s} "
                "(plant ID, not Δx_b^ub)",
                flush=True,
            )
        elif args.tdpa_press:
            from peirastic.apps.identify_air import (
                TDPA_ABORT_N,
                TDPA_CONTACT_N,
                TDPA_PRESS_M_S,
                TDPA_PRESS_S,
                TDPA_SEEK_M_S,
                TDPA_SEEK_MAX_S,
                TDPA_TARGET_N,
            )

            press_mm = speeds[0] if args.speeds_mm_s.strip() else 1e3 * TDPA_PRESS_M_S
            press_s = (
                float(args.hold_s) if float(args.hold_s) >= 1.5 else TDPA_PRESS_S
            )
            print(
                f"[DRY] tdpa-press SERVO_TWIST force-loop OFF  "
                f"seek {1e3 * TDPA_SEEK_M_S:.0f} mm/s until F>{TDPA_CONTACT_N:.1f} N "
                f"(max {TDPA_SEEK_MAX_S:.0f}s) then +{press_mm:.0f} mm/s for "
                f"{press_s:.1f}s (no stop at {TDPA_TARGET_N:.1f} N)  "
                f"abort motion F>{TDPA_ABORT_N:.1f} N then score  "
                "writes own csv; not hybrid F* tracking",
                flush=True,
            )
        elif args.air_campaign:
            from peirastic.apps.identify_air import (
                AIR_CHIRP_AMPS_MM_S,
                AIR_CHIRP_S,
                AIR_STEPS_MM_S,
            )

            print(
                f"[DRY] air-campaign steps mm/s={AIR_STEPS_MM_S} "
                f"hold={max(args.hold_s, 1.0)} rest={max(args.rest_s, 0.5)} "
                f"chirp={AIR_CHIRP_S}s amps={AIR_CHIRP_AMPS_MM_S} 0.2–8 Hz "
                "writes own air_campaign.csv (not Window A log)",
                flush=True,
            )
        else:
            print(
                f"[DRY] steps mm/s={(() if args.chirp_only else STEPS_MM_S)} "
                f"hold={args.hold_s} rest={args.rest_s} "
                f"chirp={args.chirp_s}s amp={args.chirp_amp_mm_s} mm/s",
                flush=True,
            )
        return 0
    if args.backup_replay:
        return run_backup_replay_sequence(
            prefix=str(args.shm_prefix),
            hold_s=float(args.hold_s),
            short_hold_s=float(args.short_hold_s),
            rest_s=float(args.rest_s),
            hz=float(args.hz),
            speeds_mm_s=speeds,
            event_log=Path(args.event_log) if args.event_log else Path("identify_backup_events.csv"),
            window_a_csv=Path(args.window_a_csv) if args.window_a_csv else None,
            motion_max_age_s=float(args.motion_max_age_s),
        )
    if args.stop_reverse:
        return run_stop_reverse_sequence(
            prefix=str(args.shm_prefix),
            hold_s=float(args.hold_s),
            short_hold_s=float(args.short_hold_s),
            rest_s=float(args.rest_s),
            hz=float(args.hz),
            speeds_mm_s=speeds,
        )
    if args.tdpa_press:
        from datetime import datetime

        from peirastic.apps.identify_air import (
            TDPA_PRESS_M_S,
            TDPA_PRESS_S,
            run_tdpa_press_campaign,
        )

        press_m_s = (
            float(speeds[0]) / 1000.0 if args.speeds_mm_s.strip() else TDPA_PRESS_M_S
        )
        press_s = float(args.hold_s) if float(args.hold_s) >= 1.5 else TDPA_PRESS_S
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.air_out) if args.air_out else Path(
            "MD/todo_controller_logs"
        ) / f"id_tdpa_{stamp}"
        log = Path(args.log_csv) if args.log_csv else out / "tdpa_press.csv"
        return run_tdpa_press_campaign(
            prefix=str(args.shm_prefix),
            hz=float(args.hz),
            log_csv=log,
            press_m_s=press_m_s,
            press_s=press_s,
        )
    if args.air_campaign:
        from datetime import datetime

        from peirastic.apps.identify_air import (
            AIR_CHIRP_AMPS_MM_S,
            AIR_CHIRP_F0_HZ,
            AIR_CHIRP_F1_HZ,
            AIR_CHIRP_S,
            AIR_HOLD_S,
            AIR_REST_S,
            AIR_STEPS_MM_S,
            run_air_campaign,
        )

        extra_amps = _parse_speeds_mm_s(args.chirp_amps_mm_s)
        chirp_amps = (
            tuple(x / 1000.0 for x in extra_amps)
            if args.chirp_amps_mm_s.strip()
            else tuple(x / 1000.0 for x in AIR_CHIRP_AMPS_MM_S)
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.air_out) if args.air_out else Path(
            "MD/todo_controller_logs"
        ) / f"id_air_{stamp}"
        log = Path(args.log_csv) if args.log_csv else out / "air_campaign.csv"
        return run_air_campaign(
            prefix=str(args.shm_prefix),
            hold_s=max(float(args.hold_s), AIR_HOLD_S),
            rest_s=max(float(args.rest_s), AIR_REST_S),
            chirp_s=(
                max(float(args.chirp_s), AIR_CHIRP_S)
                if float(args.chirp_s) <= 8.0
                else float(args.chirp_s)
            ),
            chirp_amps_m_s=chirp_amps,
            steps_mm_s=AIR_STEPS_MM_S if not args.speeds_mm_s.strip() else speeds,
            hz=float(args.hz),
            log_csv=log,
            out_dir=out,
            chirp_f0_hz=float(args.chirp_f0),
            # Parser default --chirp-f1 is 5 Hz (short sequence). Air
            # campaign keeps 8 Hz unless the operator passes --chirp-f1.
            chirp_f1_hz=(
                AIR_CHIRP_F1_HZ
                if abs(float(args.chirp_f1) - 5.0) < 1e-12
                else float(args.chirp_f1)
            ),
        )
    extra_amps = _parse_speeds_mm_s(args.chirp_amps_mm_s)
    return run_sequence(
        prefix=str(args.shm_prefix),
        hold_s=float(args.hold_s),
        rest_s=float(args.rest_s),
        chirp_s=float(args.chirp_s),
        chirp_amp_m_s=float(args.chirp_amp_mm_s) / 1000.0,
        hz=float(args.hz),
        include_steps=not bool(args.chirp_only),
        chirp_f0_hz=float(args.chirp_f0),
        chirp_f1_hz=float(args.chirp_f1),
        chirp_amps_m_s=(
            tuple(x / 1000.0 for x in extra_amps) if args.chirp_amps_mm_s.strip() else None
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
