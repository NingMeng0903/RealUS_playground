#!/usr/bin/env python3
"""Window C: free-space plant identification for τ_eff / G_v(s).

Requires Window A running with --log-csv.  Default sequence: tool-Z
steps (±2/5/10/20/40/80 mm/s) and a 0.2–5 Hz chirp.

``--stop-reverse`` is plant identification: +10/+20/+40/+80 mm/s to 0 and
to −40 mm/s.  Analyse with ``--analyze-stop --input-column twist_vz``.
``--backup-replay`` sends the shield's a/j-limited ``u_b``.  Pass
``--window-a-csv`` so each tick syncs measured ``v`` / ``[a]_+``, and
analyse with ``--analyze-stop --event-log`` so Window A is aligned to
the explicit trigger, not a 3 mm/s edge.  That is still not a
certificate until independent val and ``stop_dx_ub.certified: true``.

Does not enable force mode.  Bidirectional_flow / CDYOB stay observe/off.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from peirastic.core.ipc import CommandClient, Status, TwistBus
from peirastic.core.modes import Mode, ModeRequest

STEPS_MM_S = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
STOP_SPEEDS_MM_S = (10.0, 20.0, 40.0, 80.0)
REVERSE_MM_S = -40.0


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
) -> int:
    client = CommandClient(prefix=prefix)
    bus = TwistBus(prefix=prefix, create=False)
    client.set_mode(ModeRequest(Mode.SERVO_TWIST, {}))
    print("[MODE] SERVO_TWIST identify_plant", flush=True)
    dt = 1.0 / max(hz, 1.0)
    try:
        _write_vz(bus, 0.0, hz)
        if not _wait_or_estop(client, rest_s):
            return 130
        for mm_s in STEPS_MM_S:
            vz = mm_s / 1000.0
            for sign in (1.0, -1.0):
                cmd = sign * vz
                print(f"[STEP] vz={cmd:+.4f} m/s hold={hold_s:.2f}s", flush=True)
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
        print(
            f"[CHIRP] {chirp_amp_m_s*1000:.1f} mm/s  0.2–5 Hz  {chirp_s:.1f}s",
            flush=True,
        )
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= chirp_s:
                break
            f0, f1 = 0.2, 5.0
            k = math.log(f1 / f0) / max(chirp_s, 1e-6)
            phase = 2.0 * math.pi * f0 * (math.exp(k * t) - 1.0) / k
            vz = chirp_amp_m_s * math.sin(phase)
            _write_vz(bus, vz, hz)
            tel = client.snapshot()
            if int(tel["status"]) == int(Status.ESTOP):
                print("[ESTOP] " + str(tel["msg"]), flush=True)
                return 130
            time.sleep(dt)
        _write_vz(bus, 0.0, hz)
        print("[OK] sequence complete — analyse the Window A CSV with --analyze", flush=True)
        return 0
    except KeyboardInterrupt:
        _write_vz(bus, 0.0, hz)
        client.stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
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
    client.set_mode(ModeRequest(Mode.SERVO_TWIST, {}))
    print("[MODE] SERVO_TWIST identify_plant --stop-reverse", flush=True)
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
) -> int:
    """Play the shield's a/j-limited ``u_b``, not an instant −40 mm/s step.

    Each tick prefers measured ``v`` / ``[a]_+`` from Window A.  Without
    ``--window-a-csv`` this is only an open-loop model-generated backup
    sequence; do not treat it as the online shield.
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
    client.set_mode(ModeRequest(Mode.SERVO_TWIST, {}))
    closed_loop = window_a_csv is not None and Path(window_a_csv).is_file()
    print(
        "[MODE] SERVO_TWIST identify_plant --backup-replay  "
        f"a_max={sh.cfg.a_max_m_s2} j_max={sh.cfg.j_max_m_s3} "
        f"u_retract={sh.cfg.u_retract_m_s} "
        f"{'closed-loop v_actual' if closed_loop else 'OPEN-LOOP model v (not a certificate)'}",
        flush=True,
    )
    rows: list[dict[str, str]] = []
    event_n = 0
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
                v_meas, a_plus, t_csv = _tail_window_a_motion(window_a_csv)
                if v_meas is not None:
                    sh._sync_plant_from_measurement(v_meas, a_plus)
                t_trigger_wall = time.time()
                t_trigger_mono = time.monotonic()
                print(
                    f"[BACKUP] event_id={eid} trigger=backup_to_terminal "
                    f"phase={label} v0_cmd={mm_s:.0f} mm/s "
                    f"v0_meas={('nan' if v_meas is None else f'{1e3 * v_meas:.1f}')} mm/s "
                    f"a0+={('nan' if a_plus is None else f'{a_plus:.3f}')} "
                    f"t_wall_csv={('nan' if t_csv is None else f'{t_csv:.4f}')} "
                    f"hold={dwell:.2f}s",
                    flush=True,
                )
                hold_acc = 0.0
                n_max = int(round((rest_s + 0.50) / dt))
                for tick in range(n_max):
                    v_now, a_now, t_csv_now = _tail_window_a_motion(window_a_csv)
                    if v_now is not None:
                        sh._sync_plant_from_measurement(v_now, a_now)
                    v_pred = (
                        float(sh._v_plant)
                        if v_now is not None
                        else float(vz)
                    )
                    u = sh.backup_command(
                        sh._u_prev,
                        sh._u_prev2,
                        released=False,
                        v_pred=v_pred,
                    )
                    t_unix = time.time()
                    if tick == 0:
                        t_trigger_wall = t_unix
                    _write_vz(bus, u, hz)
                    sh._commit_sent(u, keep_measured_state=v_now is not None)
                    queue_u = ";".join(f"{float(x):.6f}" for x in sh._delay)
                    rows.append(
                        {
                            "event_id": eid,
                            "trigger": "backup_to_terminal",
                            "phase": label,
                            "tick": str(tick),
                            "t_unix_s": f"{t_unix:.6f}",
                            "t_mono_s": f"{time.monotonic() - t_trigger_mono:.6f}",
                            "t_wall_csv_s": (
                                "" if t_csv_now is None else f"{t_csv_now:.6f}"
                            ),
                            "u_b": f"{u:.6f}",
                            "v0_cmd": f"{vz:.6f}",
                            "v_actual": (
                                "" if v_now is None else f"{v_now:.6f}"
                            ),
                            "a0_plus": (
                                "" if a_now is None else f"{a_now:.6f}"
                            ),
                            "q_remain_m": f"{sh.queue_remain_m():.8f}",
                            "queue_u": queue_u,
                        }
                    )
                    tel = client.snapshot()
                    if int(tel["status"]) == int(Status.ESTOP):
                        print("[ESTOP] " + str(tel["msg"]), flush=True)
                        return 130
                    q_ok = sh.queue_press() <= float(sh.cfg.queue_clear_m_s) + 1e-9
                    v_ok = (
                        abs(float(sh._v_plant)) <= float(sh.cfg.v_hold_m_s) + 1e-9
                        if v_now is not None
                        else True
                    )
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
                del t_trigger_wall
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
        bus.close()
        client.close()


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
) -> tuple[list[float], list[float], list[float], int]:
    """Unified ê_v(i), ê_x(i), ê_{x,+}(i) from the same origins: v̂(k+i|k)."""
    ev = [0.0] * horizon
    ex = [0.0] * horizon
    ex_plus = [0.0] * horizon
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
    return ev, ex, ex_plus, used


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


def _monotonic_stop_bins(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    groups: dict[int, list[dict[str, float]]] = {}
    for r in rows:
        if float(r.get("gap_invalid", 0.0)) >= 0.5:
            continue
        if not math.isfinite(float(r.get("dx_press", float("nan")))):
            continue
        lo = int(1000.0 * abs(float(r["v0"])) // 10.0) * 10
        groups.setdefault(lo, []).append(r)
    out: list[dict[str, float]] = []
    run_dx = 0.0
    for lo in sorted(groups):
        rs = groups[lo]
        dx = max(float(r["dx_press"]) for r in rs)
        run_dx = max(run_dx, dx)
        a0 = max(max(float(r["a0"]), 0.0) for r in rs)
        q0 = max(float(r.get("q_press", 0.0)) for r in rs)
        q_remain = max(float(r.get("q_remain_m", 0.0)) for r in rs)
        nbs = [float(r["n_b"]) for r in rs if math.isfinite(float(r["n_b"]))]
        out.append(
            {
                "v0_m_s": (lo + 10) / 1000.0,
                "a0_m_s2": a0,
                "q_press_m_s": q0,
                "q_remain_m": q_remain,
                "dx_ub_m": run_dx,
                "n_b": float(max(nbs) if nbs else 0.0),
            }
        )
    return out


def _stop_dx_yaml_block(rows: list[dict[str, float]], *, source: str) -> str:
    bins = _monotonic_stop_bins(rows)
    lines = [
        "safety_shield:",
        "  stop_dx_ub:",
        "    certified: false",
        f"    source: {source}",
        "    note: plant-ID or unvalidated backup replay; not Δx_b^ub until",
        "      independent val covers the envelope and certified is set true.",
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
            f"q_remain_m: {b['q_remain_m']:.8f}, dx_ub_m: {b['dx_ub_m']:.7f}, "
            f"n_b: {int(b['n_b'])}}}"
        )
    return "\n".join(lines) + "\n"


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
    min_drop = max(0.5 * 1.20 * max(dt, 1e-4), 0.001)
    edges: list[tuple[str, int]] = []
    if event_log is not None and Path(event_log).is_file():
        triggers = _event_trigger_rows(_load_event_log(Path(event_log)))
        search_from = 1
        missed = 0
        for trig in triggers:
            idx = _align_trigger_index(cmd, t, trig, search_from=search_from)
            if idx is None:
                missed += 1
                print(
                    f"[ID-SR] event_id={trig.get('event_id')} not aligned "
                    "(need t_wall_csv_s or v0_cmd→u_b match)",
                    flush=True,
                )
                continue
            edges.append(("backup", int(idx)))
            search_from = int(idx) + 1
        print(
            f"[ID-SR] event-log={event_log} triggers={len(triggers)} "
            f"aligned={len(edges)} missed={missed} "
            "(explicit trigger, not a 3 mm/s edge)",
            flush=True,
        )
    else:
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
        "The shield uses Δx_1(u(λ))+D_b(ξ_1), not max(model, D_b(ξ)).",
        flush=True,
    )
    if write_yaml is not None:
        write_yaml.write_text(yaml_block)
        print(f"[ID-SR] wrote {write_yaml}", flush=True)

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
    ev_f, ex_f, ex_f_plus, n_f = _open_loop_envelopes(
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
    ev_2, ex_2, ex_2_plus, _n_2 = _open_loop_envelopes(
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
        ev_a, ex_a, ex_a_plus = [0.0] * horizon, [0.0] * horizon, [0.0] * horizon
    else:
        ev_a, ex_a, ex_a_plus, _n_a = _open_loop_envelopes(
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
        f"ê_x,+ max={1e3 * _emax(ex_f_plus):.2f}mm  "
        f"2nd(wn={best_wn:.1f},z={best_z:.2f}) "
        f"ê_v max={_emax(ev_2):.4f} ê_x max={1e3 * _emax(ex_2):.2f}mm "
        f"ê_x,+ max={1e3 * _emax(ex_2_plus):.2f}mm  "
        f"ARX ê_v max={_emax(ev_a):.4f} ê_x max={1e3 * _emax(ex_a):.2f}mm "
        f"ê_x,+ max={1e3 * _emax(ex_a_plus):.2f}mm",
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
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="peirastic plant identification")
    parser.add_argument("--shm-prefix", default="")
    parser.add_argument("--hold-s", type=float, default=0.80)
    parser.add_argument("--rest-s", type=float, default=0.40)
    parser.add_argument("--chirp-s", type=float, default=8.0)
    parser.add_argument("--chirp-amp-mm-s", type=float, default=20.0)
    parser.add_argument("--hz", type=float, default=200.0)
    parser.add_argument("--analyze", type=str, default="", help="analyse an existing CSV")
    parser.add_argument(
        "--analyze-stop",
        type=str,
        default="",
        help="analyse a --stop-reverse CSV for Δx^+ / N_press / N_b",
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
        help="Window A --log-csv path so --backup-replay syncs measured v/[a]+",
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
    speeds = _parse_speeds_mm_s(args.speeds_mm_s)
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
        else:
            print(
                f"[DRY] steps mm/s={STEPS_MM_S} hold={args.hold_s} rest={args.rest_s} "
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
    return run_sequence(
        prefix=str(args.shm_prefix),
        hold_s=float(args.hold_s),
        rest_s=float(args.rest_s),
        chirp_s=float(args.chirp_s),
        chirp_amp_m_s=float(args.chirp_amp_mm_s) / 1000.0,
        hz=float(args.hz),
    )


if __name__ == "__main__":
    raise SystemExit(main())
