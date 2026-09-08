"""Background force trace for the phantom hybrid scan.

The recorder is deliberately independent from the motion client.  It samples
Window A's status and the compensated-force relay from its own thread, and it
only accepts samples while the requested ``TRACK_HYBRID`` phase is live.  The
default relay is a read-only ``ForceExtBus`` subscriber; an injected bus is
owned by the caller and is never stopped by this class.
"""

from __future__ import annotations

import csv
import json
import math
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from peirastic.core.ipc import Status
from peirastic.core.modes import Mode

TRACK_HYBRID_VALUE = int(Mode.TRACK_HYBRID)
_TERMINAL_STATUS_VALUES = frozenset(
    (int(Status.ERROR), int(Status.STOPPED), int(Status.ESTOP))
)
_ACTIVE_STAGES = frozenset(("approach", "tracking"))
_STATUS_MAX_AGE_S = 0.5
_STATUS_MAX_FUTURE_S = 0.1
_FORCE_MAX_AGE_S = 0.5
_FORCE_MAX_FUTURE_S = 0.5
_MESSAGE_FIELD_RE = re.compile(r"(?:^|\s)(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)")

CSV_FIELDS = (
    "sample_t_mono_s",
    "status_t_mono_s",
    "force_seq",
    "force_t_s",
    "fx_n",
    "fy_n",
    "fz_n",
    "mx_nm",
    "my_nm",
    "mz_nm",
    "desired_fz_n",
    "stage",
    "contact",
    "vz_cmd_mm_s",
    "valid",
    "stale",
    "wrench_complete",
    "force_source",
    "mode",
    "status_msg",
)


def _status_field(status: Any, name: str, default: Any = None) -> Any:
    try:
        if isinstance(status, Mapping):
            return status.get(name, default)
        return getattr(status, name, default)
    except Exception:
        return default


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    return number if math.isfinite(number) else float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return bool(float(value))
        except (TypeError, ValueError, OverflowError):
            return bool(default)
    try:
        text = str(value).strip().lower()
    except Exception:
        return bool(default)
    if text in {"1", "true", "yes", "on", "contact", "active"}:
        return True
    if text in {"0", "false", "no", "off", "none", "", "stale"}:
        return False
    return bool(default)


def _mode_is_hybrid(value: Any) -> bool:
    try:
        return int(value) == TRACK_HYBRID_VALUE
    except (TypeError, ValueError, OverflowError):
        try:
            text = str(value).strip().upper()
        except Exception:
            return False
        return text in {"TRACK_HYBRID", "MODE.TRACK_HYBRID"}


def _status_is_terminal(value: Any) -> bool:
    try:
        if int(value) in _TERMINAL_STATUS_VALUES:
            return True
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        text = str(value).strip().upper()
    except Exception:
        return False
    return text in {
        "ERROR",
        "STOPPED",
        "ESTOP",
        "STATUS.ERROR",
        "STATUS.STOPPED",
        "STATUS.ESTOP",
    }


def _message_fields(message: str) -> dict[str, str]:
    try:
        return {match.group("key"): match.group("value") for match in _MESSAGE_FIELD_RE.finditer(message)}
    except Exception:
        return {}


def _active_stage(status: Any, label: str) -> tuple[bool, str, str]:
    message_value = _status_field(status, "msg", "")
    try:
        message = str(message_value or "").strip()
    except Exception:
        message = ""
    if message == label:
        stage = str(_status_field(status, "stage", "active") or "active").strip().lower()
        return True, stage if stage else "active", message
    prefix = f"{label}:"
    if not message.startswith(prefix):
        return False, "", message
    token = message[len(prefix) :].split(None, 1)[0].strip().lower()
    if token not in _ACTIVE_STAGES:
        return False, "", message
    return True, token, message


def _parse_message_number(fields: Mapping[str, str], key: str) -> float:
    value = fields.get(key)
    if value is None:
        return float("nan")
    # Controller messages currently use plain numeric values.  Accept a
    # trailing unit too so a diagnostic message remains useful if that text
    # changes to ``vz_cmd=+1.2mm/s``.
    match = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value)
    return _finite_float(match.group(0) if match else value)


class ForceTraceRecorder:
    """Record compensated wrench samples during one phantom scan.

    ``read_status`` should return the Window A snapshot dictionary (or an
    object exposing the same fields).  A supplied ``force_bus`` must expose
    ``read() -> (ok, seq, t_s, wrench6)``.  The recorder owns only its default
    bus; an injected bus remains the caller's resource and is never stopped.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        read_status: Callable[[], Any],
        desired_force_n: float,
        label: str = "phathom_s_scan",
        sample_hz: float = 100.0,
        force_bus: Any | None = None,
    ) -> None:
        if not callable(read_status):
            raise TypeError("read_status must be callable")
        desired = _finite_float(desired_force_n)
        if not math.isfinite(desired):
            raise ValueError("desired_force_n must be finite")
        hz = _finite_float(sample_hz)
        if not math.isfinite(hz) or hz <= 0.0:
            raise ValueError("sample_hz must be finite and positive")
        text_label = str(label).strip()
        if not text_label:
            raise ValueError("label must be non-empty")

        self.directory = Path(directory)
        self.read_status = read_status
        self.desired_force_n = desired
        self.label = text_label
        self.sample_hz = hz
        self.csv_path = self.directory / "force_trace.csv"
        self.png_path = self.directory / "force_trace.png"
        self.summary_path = self.directory / "force_trace_summary.json"

        self._owns_force_bus = force_bus is None
        if force_bus is None:
            # Import lazily so offline tests and dry-run tools do not attach to
            # shared memory merely by importing this module.
            from rm75_control.control.admittance_common.state_relay import ForceExtBus

            force_bus = ForceExtBus()
        self.force_bus = force_bus

        self._state_lock = threading.RLock()
        self._file_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._csv_file = None
        self._csv_writer: csv.DictWriter | None = None
        self._started = False
        self._stopped = False
        self._finished = False
        self._result: dict[str, str] | None = None
        self.error: Exception | None = None
        self._force_error_count = 0
        self._force_error_text = ""
        self._last_status_t_mono: float | None = None
        self._last_force_seq: int | None = None
        self._rows = 0
        self._valid_rows = 0
        self._stale_rows = 0
        self._complete_rows = 0
        self._fallback_rows = 0
        self._invalid_rows = 0
        self._duplicate_rows = 0
        self._force_seq_gap_count = 0
        self._stage_counts: Counter[str] = Counter()
        self._source_counts: Counter[str] = Counter()
        self._sample_times: list[float] = []
        self._last_flush_mono = 0.0

    @property
    def running(self) -> bool:
        with self._state_lock:
            return bool(self._thread is not None and self._thread.is_alive())

    def check_error(self) -> None:
        """Raise a worker/source error so the caller can stop the scan early."""

        error = self.error
        if error is not None:
            raise RuntimeError(f"force trace recorder failed: {error}") from error

    def start(self) -> None:
        """Create the CSV synchronously, then start the sampling thread."""

        with self._state_lock:
            if self._started:
                raise RuntimeError("force trace recorder already started")
            if self._finished:
                raise RuntimeError("force trace recorder already finished")
            self.directory.mkdir(parents=True, exist_ok=True)
            self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
            self._csv_writer.writeheader()
            self._csv_file.flush()
            self._last_flush_mono = time.monotonic()
            self._stop_event.clear()
            self._started = True
            self._stopped = False
            self._thread = threading.Thread(
                target=self._run,
                name="phathom-force-trace",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop sampling without closing the CSV or rendering the plot.

        This is intentionally idempotent: the scan stops sampling immediately
        after contact tracking, while ``finish`` calls it again in ``finally``.
        """

        with self._state_lock:
            if self._stopped:
                return
            self._stopped = True
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
            if thread.is_alive():
                error = RuntimeError("force trace sampler did not stop within 3 seconds")
                self._set_error(error)
                raise error

    def finish(self, outcome: str) -> dict[str, str]:
        """Stop, close, plot, and summarize one trace; repeated calls reuse paths."""

        with self._state_lock:
            if self._finished:
                return dict(self._result or self._paths())
            if not self._started:
                # A failed setup before ``start`` still gets an explicit empty
                # trace, making aborted runs distinguishable from missing logs.
                self.directory.mkdir(parents=True, exist_ok=True)
                self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
                self._csv_writer.writeheader()
                self._csv_file.flush()
                self._started = True
                self._stopped = True
            else:
                # Marking stopped under the lock prevents a concurrent finish
                # from starting a second join; the actual join is outside it.
                pass

        self.stop()
        errors: list[Exception] = []
        try:
            self._close_csv()
        except Exception as exc:
            errors.append(exc)

        if self._owns_force_bus:
            try:
                stop_bus = getattr(self.force_bus, "stop", None)
                if callable(stop_bus):
                    stop_bus()
            except Exception as exc:
                errors.append(exc)

        plot_error: Exception | None = None
        try:
            self._plot()
        except Exception as exc:
            plot_error = exc
            errors.append(exc)

        summary_errors = list(errors)
        if self.error is not None and all(self.error is not error for error in summary_errors):
            summary_errors.insert(0, self.error)
        summary = self._summary(outcome, plot_error=plot_error, errors=summary_errors)
        try:
            self.summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            errors.append(exc)

        paths = self._paths()
        with self._state_lock:
            self._result = dict(paths)
            self._finished = True
        if self.error is not None:
            errors.insert(0, self.error)
        if errors:
            unique: list[Exception] = []
            seen: set[int] = set()
            for error in errors:
                if id(error) not in seen:
                    seen.add(id(error))
                    unique.append(error)
            message = "; ".join(str(error) for error in unique)
            raise RuntimeError(f"force trace finalization failed: {message}") from unique[0]
        return dict(paths)

    def _paths(self) -> dict[str, str]:
        return {
            "csv": str(self.csv_path),
            "png": str(self.png_path),
            "summary": str(self.summary_path),
        }

    def _set_error(self, error: Exception) -> None:
        with self._state_lock:
            if self.error is None:
                self.error = error

    def _record_force_error(self, error: Exception) -> None:
        with self._state_lock:
            self._force_error_count += 1
            if not self._force_error_text:
                self._force_error_text = f"{type(error).__name__}: {error}"
            if self.error is None:
                self.error = error

    def _run(self) -> None:
        period = 1.0 / self.sample_hz
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_tick:
                self._stop_event.wait(next_tick - now)
                continue
            next_tick += period
            if next_tick < now - period:
                next_tick = now + period
            try:
                self._sample_once(now)
            except Exception as exc:
                self._set_error(exc)
                self._stop_event.set()
                break

    def _sample_once(self, sample_t_mono: float) -> None:
        try:
            status = self.read_status()
        except Exception as exc:
            self._set_error(exc)
            self._stop_event.set()
            return
        if status is None or _status_is_terminal(_status_field(status, "status")):
            return
        if not _mode_is_hybrid(_status_field(status, "mode")):
            return
        active, stage, message = _active_stage(status, self.label)
        if not active:
            return

        status_t = _finite_float(_status_field(status, "t_mono"))
        status_age = (
            sample_t_mono - status_t if math.isfinite(status_t) else float("nan")
        )
        status_duplicate = bool(
            math.isfinite(status_t)
            and self._last_status_t_mono is not None
            and status_t <= self._last_status_t_mono
        )
        if math.isfinite(status_t) and (
            self._last_status_t_mono is None or status_t > self._last_status_t_mono
        ):
            self._last_status_t_mono = status_t
        message_values = _message_fields(message)
        contact_value = _status_field(status, "contact", None)
        if contact_value is None:
            contact_value = _status_field(status, "contact_present", None)
        if contact_value is None:
            contact_value = message_values.get("contact")
        contact = 1 if _safe_bool(contact_value, False) else 0
        vz_value = _status_field(status, "vz_cmd", None)
        if vz_value is None:
            vz_cmd = _parse_message_number(message_values, "vz_cmd")
        else:
            vz_cmd = _finite_float(vz_value)
        stale = _safe_bool(
            _status_field(
                status,
                "stale",
                _status_field(status, "command_stale", False),
            ),
            False,
        )
        stale = bool(
            stale
            or status_duplicate
            or not math.isfinite(status_t)
            or status_age < -_STATUS_MAX_FUTURE_S
            or status_age > _STATUS_MAX_AGE_S
        )

        force_ok = False
        force_stale = False
        force_timestamp_stale = False
        force_complete = False
        force_source = "none"
        force_seq = 0
        force_t_s = float("nan")
        wrench = np.full(6, np.nan, dtype=float)
        duplicate = False
        try:
            result = self.force_bus.read()
            if not isinstance(result, (tuple, list)) or len(result) != 4:
                raise ValueError("force_bus.read() must return (ok, seq, t_s, wrench6)")
            ok, raw_seq, raw_t_s, raw_wrench = result
            try:
                force_seq = int(raw_seq)
            except (TypeError, ValueError, OverflowError):
                force_seq = 0
            force_t_s = _finite_float(raw_t_s)
            if self._last_force_seq is not None and force_seq > self._last_force_seq + 1:
                self._force_seq_gap_count += force_seq - self._last_force_seq - 1
            if (
                self._last_force_seq is not None
                and force_seq > 0
                and force_seq <= self._last_force_seq
            ):
                duplicate = True
                force_stale = True
                self._duplicate_rows += 1
                force_source = "force_bus_duplicate"
            elif force_seq > 0:
                self._last_force_seq = force_seq
            vector = np.asarray(raw_wrench, dtype=float).reshape(-1)
            force_age = (
                sample_t_mono - force_t_s
                if math.isfinite(force_t_s)
                else float("nan")
            )
            force_timestamp_stale = bool(
                not math.isfinite(force_t_s)
                or force_age < -_FORCE_MAX_FUTURE_S
                or force_age > _FORCE_MAX_AGE_S
            )
            if (
                _safe_bool(ok, False)
                and force_seq > 0
                and not duplicate
                and not force_timestamp_stale
                and vector.size >= 6
                and np.all(np.isfinite(vector[:6]))
            ):
                wrench = vector[:6].copy()
                force_ok = True
                force_complete = True
                force_source = "force_bus_tool_compensated"
            elif not duplicate:
                force_source = (
                    "force_bus_stale" if force_timestamp_stale else "force_bus_invalid"
                )
        except Exception as exc:
            self._record_force_error(exc)
            force_source = "force_bus_error"

        if stale:
            # A stale controller snapshot cannot be used to turn a prior force
            # value into a fresh sample.
            wrench[:] = np.nan
            force_ok = False
            force_complete = False
            force_stale = True
            force_source = "stale_controller"
        elif not force_ok and not duplicate:
            fz_fallback = _finite_float(_status_field(status, "f_ext_z"))
            if math.isfinite(fz_fallback):
                wrench[:] = np.nan
                wrench[2] = fz_fallback
                force_ok = True
                force_source = "controller_f_ext_z_fallback"
                self._fallback_rows += 1
            else:
                wrench[:] = np.nan

        if force_stale or (force_timestamp_stale and not force_ok):
            stale = True
        valid = bool(force_ok and not stale)
        if not valid:
            wrench[:] = np.nan
        row = {
            "sample_t_mono_s": float(sample_t_mono),
            "status_t_mono_s": float(status_t),
            "force_seq": int(force_seq),
            "force_t_s": float(force_t_s),
            "fx_n": float(wrench[0]),
            "fy_n": float(wrench[1]),
            "fz_n": float(wrench[2]),
            "mx_nm": float(wrench[3]),
            "my_nm": float(wrench[4]),
            "mz_nm": float(wrench[5]),
            "desired_fz_n": float(self.desired_force_n),
            "stage": stage,
            "contact": int(contact),
            "vz_cmd_mm_s": float(vz_cmd),
            "valid": int(valid),
            "stale": int(stale),
            "wrench_complete": int(force_complete and valid),
            "force_source": force_source,
            "mode": TRACK_HYBRID_VALUE,
            "status_msg": message,
        }
        self._write_row(row)
        self._rows += 1
        self._sample_times.append(float(sample_t_mono))
        self._stage_counts[stage] += 1
        self._source_counts[force_source] += 1
        if valid:
            self._valid_rows += 1
        else:
            self._invalid_rows += 1
        if stale:
            self._stale_rows += 1
        if force_complete and valid:
            self._complete_rows += 1

    def _write_row(self, row: dict[str, Any]) -> None:
        with self._file_lock:
            if self._csv_writer is None or self._csv_file is None:
                raise RuntimeError("force trace CSV is not open")
            self._csv_writer.writerow(row)
            now = time.monotonic()
            if self._rows % 100 == 0 or now - self._last_flush_mono >= 1.0:
                self._csv_file.flush()
                self._last_flush_mono = now

    def _close_csv(self) -> None:
        with self._file_lock:
            if self._csv_file is None:
                return
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None

    def _read_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.is_file():
            return []
        with self.csv_path.open("r", newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def _csv_number(row: Mapping[str, str], key: str) -> float:
        return _finite_float(row.get(key))

    def _plot(self) -> None:
        # Rendering is intentionally delayed until finish, after sampling and
        # motion cleanup.  Never import pyplot on the control thread.
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        rows = self._read_rows()
        fig, ax = plt.subplots(figsize=(10.0, 5.5))
        try:
            if not rows:
                ax.text(
                    0.5,
                    0.5,
                    "No active TRACK_HYBRID force samples",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{self.label} force trace (empty)")
                ax.set_xlabel("elapsed status sample time [s]")
                ax.set_ylabel("force [N]")
                ax.grid(True, alpha=0.25)
            else:
                t = np.asarray([self._csv_number(row, "sample_t_mono_s") for row in rows], dtype=float)
                finite_t = np.isfinite(t)
                t0 = float(t[finite_t][0]) if np.any(finite_t) else 0.0
                x = t - t0
                colors = {"fx_n": "tab:blue", "fy_n": "tab:orange", "fz_n": "tab:green"}
                labels = {"fx_n": "Fx", "fy_n": "Fy", "fz_n": "Fz"}
                for key in ("fx_n", "fy_n", "fz_n"):
                    values = np.asarray([self._csv_number(row, key) for row in rows], dtype=float)
                    ax.plot(x, values, label=labels[key], color=colors[key], linewidth=1.2)
                ax.axhline(
                    self.desired_force_n,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"target Fz={self.desired_force_n:g} N",
                )
                self._shade_stages(ax, x, rows)
                ax.set_title(f"{self.label} compensated tool force")
                ax.set_xlabel("elapsed status sample time [s]")
                ax.set_ylabel("force [N]")
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(self.png_path, dpi=150)
        finally:
            plt.close(fig)

    @staticmethod
    def _shade_stages(ax, x: np.ndarray, rows: list[dict[str, str]]) -> None:
        if x.size == 0:
            return
        start = 0
        current = rows[0].get("stage", "")
        for index in range(1, len(rows) + 1):
            stage = rows[index].get("stage", "") if index < len(rows) else None
            if stage == current:
                continue
            if current in _ACTIVE_STAGES:
                color = "#b7d7f0" if current == "approach" else "#c6e7c6"
                left = float(x[start])
                right = float(x[index - 1])
                if right <= left:
                    right = left + 1.0e-6
                ax.axvspan(left, right, color=color, alpha=0.18, linewidth=0)
                ax.text(
                    0.5 * (left + right),
                    0.98,
                    current.capitalize(),
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#405060",
                )
            start = index
            current = stage

    def _summary(
        self,
        outcome: str,
        *,
        plot_error: Exception | None,
        errors: list[Exception],
    ) -> dict[str, Any]:
        times = np.asarray(self._sample_times, dtype=float)
        finite_times = times[np.isfinite(times)]
        elapsed = float(finite_times[-1] - finite_times[0]) if finite_times.size >= 2 else 0.0
        actual_hz = float((finite_times.size - 1) / elapsed) if finite_times.size >= 2 and elapsed > 0.0 else 0.0
        error_texts = []
        for error in errors:
            text = f"{type(error).__name__}: {error}"
            if text not in error_texts:
                error_texts.append(text)
        summary: dict[str, Any] = {
            "label": self.label,
            "outcome": str(outcome),
            "empty": self._rows == 0,
            "configured_sample_hz": self.sample_hz,
            "actual_sample_hz": actual_hz,
            "active_samples": self._rows,
            "valid_samples": self._valid_rows,
            "invalid_samples": self._invalid_rows,
            "stale_samples": self._stale_rows,
            "complete_wrench_samples": self._complete_rows,
            "fallback_fz_samples": self._fallback_rows,
            "duplicate_force_samples": self._duplicate_rows,
            "force_seq_gap_count": int(self._force_seq_gap_count),
            "duration_s": elapsed,
            "stage_counts": dict(self._stage_counts),
            "force_source_counts": dict(self._source_counts),
            "force_read_error_count": int(self._force_error_count),
            "force_read_error": self._force_error_text,
            "errors": error_texts,
            "units_frame": {
                "force": "N",
                "moment": "N m",
                "desired_force": "N",
                "vz_cmd": "mm/s",
                "status_time": "monotonic seconds",
                "force_time": "UNIX seconds",
                "wrench_frame": "tool compensated",
            },
            "paths": self._paths(),
        }
        if plot_error is not None:
            summary["plot_error"] = f"{type(plot_error).__name__}: {plot_error}"
        return summary


__all__ = ["CSV_FIELDS", "ForceTraceRecorder"]
