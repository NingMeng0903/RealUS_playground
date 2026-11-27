"""Native transition catalog for paired adaptive experiments.

This module is intentionally a thin runner around the corrected closed-loop
implementation in ``ab_closed_loop_reviewed_20261111``.  It does not duplicate
the ProDMP, dIRD, kinematics, or QP implementation.  A transition is always
checked from the complete native state returned by the previous transition.

The public side of a transition contains only the measured-frame exit and the
``planning.action_at`` action.  The true-surface action is written under the
evaluation boundary for audit, but is never supplied to a planner.  Native
traces and complete controller snapshots are retained so q/qdot/acceleration
continuity can be independently checked after a run.

Examples (Genesis Python, no hardware)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /media/camp/EXT_DRIVE/envs/genesis/bin/python native_catalog.py \
      --stage stage1 --output native_outputs

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      /media/camp/EXT_DRIVE/envs/genesis/bin/python native_catalog.py \
      --stage catalog --output native_outputs --max-actions 6

``--output`` is a parent directory.  Every invocation creates a fresh child;
existing runs are never overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REVIEWED = HERE.parent / "ab_closed_loop_reviewed_20261111"
PROBE_NAMES = (
    "side_left",
    "side_right",
    "fork_minus",
    "fork_plus",
    "fork_plus_shift3",
)
WINDOW_NAMES = ("left", "right")
DEFAULT_LAYOUTS = (0, 1)
SCHEMA = "native_paired_action_catalog_v1"


def _load_reviewed_modules() -> dict[str, Any]:
    """Import the corrected sibling as the only source of motion primitives.

    The experiment directories intentionally use common module names.  Evict
    any module loaded from another experiment before importing, otherwise a
    long-lived test process could silently mix fixtures or controller code.
    """

    sys.path.insert(0, str(REVIEWED))
    for name in ("acoustics", "fixture", "robot_motion", "planning"):
        stale = sys.modules.get(name)
        if stale is None:
            continue
        source = getattr(stale, "__file__", None)
        if source is None or Path(source).resolve().parent != REVIEWED:
            del sys.modules[name]
    importlib.invalidate_caches()
    modules = {
        name: importlib.import_module(name)
        for name in ("acoustics", "fixture", "robot_motion", "planning")
    }
    for name, module in modules.items():
        source = Path(module.__file__).resolve()
        if source.parent != REVIEWED:
            raise RuntimeError(f"{name} imported from {source}, expected {REVIEWED}")
    return modules


def _json_safe(value: Any) -> Any:
    """Convert native diagnostics to strict JSON without dropping snapshots."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "as_dict"):
        return _json_safe(value.as_dict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _json_safe(vars(value))
    return value


def _dump(path: Path, payload: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=indent, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_manifest(modules: Mapping[str, Any]) -> dict[str, Any]:
    files = {}
    for name, module in modules.items():
        source = Path(module.__file__).resolve()
        files[name] = {"path": str(source), "sha256": _sha(source)}
    # These protocol files are part of the corrected sibling's reviewed source
    # boundary even though they are not imported by Python.
    for name in ("PROTOCOL.txt", "IMPLEMENTATION_FREEZE.txt", "ASSOCIATION_CORRECTION.txt"):
        path = REVIEWED / name
        if path.exists():
            files[name] = {"path": str(path), "sha256": _sha(path)}
    return {
        "reviewed_root": str(REVIEWED),
        "modules": files,
        "import_policy": "corrected sibling only; no adaptive_stopping source imports",
    }


def _native_payload(state: Any, motion: Any) -> dict[str, Any]:
    """Return complete q/qdot/acceleration/controller state for persistence."""

    if state.native is None:
        return {
            "q": np.asarray(motion.q_seed, dtype=float).tolist(),
            "qdot": np.zeros(8, dtype=float).tolist(),
            "acceleration": np.zeros(8, dtype=float).tolist(),
            "snapshot": None,
            "snapshot_sha256": None,
        }
    snapshot = _json_safe(state.native.snapshot)
    return {
        "q": np.asarray(state.native.q, dtype=float).tolist(),
        "qdot": np.asarray(state.native.qdot, dtype=float).tolist(),
        "acceleration": np.asarray(state.native.acceleration, dtype=float).tolist(),
        "snapshot": snapshot,
        "snapshot_sha256": _hash_json(snapshot),
    }


def _state_payload(state: Any, motion: Any) -> dict[str, Any]:
    native = _native_payload(state, motion)
    return {
        "key": list(state.key),
        "elapsed_s": float(state.elapsed),
        "pose": _json_safe(np.asarray(state.pose, dtype=float)),
        "valid": bool(state.valid),
        "native": native,
    }


def _state_signature(state: Any, motion: Any) -> dict[str, Any]:
    payload = _state_payload(state, motion)
    native = payload["native"]
    snapshot = native.get("snapshot")
    fingerprint = None
    schema = None
    if isinstance(snapshot, Mapping):
        fingerprint = snapshot.get("fingerprint")
        schema = snapshot.get("schema")
    return {
        "key": list(state.key),
        "elapsed_s": float(state.elapsed),
        "valid": bool(state.valid),
        "pose_sha256": _hash_json(payload["pose"]),
        "q": native["q"],
        "qdot": native["qdot"],
        "acceleration": native["acceleration"],
        "controller_snapshot_sha256": native["snapshot_sha256"],
        "controller_snapshot_schema": schema,
        "controller_fingerprint": _json_safe(fingerprint),
    }


def _action_dict(action: Any, *, frame_role: str) -> dict[str, Any]:
    payload = asdict(action)
    payload["xy_m"] = [float(x) for x in action.xy_m]
    payload["yaw_rad"] = float(action.yaw_rad)
    payload["duration_s"] = float(action.duration_s)
    payload["frame_role"] = frame_role
    return payload


def _relative_pose(modules: Mapping[str, Any], frame: np.ndarray, pose: np.ndarray) -> list[float]:
    local = modules["fixture"].local_pose(frame, pose)
    return [float(x) for x in local]


def _safe_name(value: str) -> str:
    return "_".join("".join(char if char.isalnum() or char in "._-" else "_" for char in value).split())


def _allocate_run(parent: Path, stage: str) -> Path:
    """Create a unique run child and never overwrite an existing output."""

    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = parent / f"native_catalog_{_safe_name(stage)}_{stamp}_{os.getpid()}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base.name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _add_shift_capture(rig: Any, fixture: Any) -> Any:
    """Return a rig copy with the +3 mm local-y fork probe added."""

    # Rig is frozen, but its cached_property dictionary is intentionally
    # mutable.  Make a dataclass copy so the reviewed rig itself is untouched.
    copied = rig.__class__(
        rig.name, rig.address, rig.root_pose, rig.measured, rig.actual_surface,
        rig.wall, rig.main_side, rig.width, rig.terminal_y, rig.sensor_points,
        rig.nominal, rig.atlas_frame, rig.atlas_to_robot,
    )
    captures = dict(rig.captures)
    old = captures["fork_plus"]
    captures["fork_plus_shift3"] = fixture.Capture(
        "fork_plus_shift3", "fork", (float(old.goal[0]), 0.0, float(old.goal[2])), "fork"
    )
    copied.__dict__["captures"] = captures
    return copied


@dataclass
class CatalogRunner:
    """Run checked transitions while retaining audit evidence."""

    modules: Mapping[str, Any]
    rig: Any
    motion: Any
    layout_dir: Path
    max_actions: int
    max_diagnostic_decisions: int
    max_terminal_attempts: int

    def __post_init__(self) -> None:
        self.fixture = self.modules["fixture"]
        self.planning = self.modules["planning"]
        self.rollouts = self.fixture.MotionRollouts(
            self.rig, self.motion, self.layout_dir / "traces"
        )
        self.state_store = self.layout_dir / "native_states"
        self.state_store.mkdir(parents=True, exist_ok=True)
        self.transitions: dict[str, dict[str, Any]] = {}
        self._state_links: dict[str, str] = {}
        self.action_count = 0
        self.diagnostic_decisions = 0
        self.terminal_attempts = 0
        self.linkage_failures: list[dict[str, Any]] = []

    def _state_link(self, state: Any, label: str) -> str:
        signature = _state_signature(state, self.motion)
        identity = _hash_json({"signature": signature, "payload": _state_payload(state, self.motion)})
        if identity in self._state_links:
            return self._state_links[identity]
        filename = f"{_safe_name(label)}__{identity[:16]}.json"
        path = self.state_store / filename
        _dump(path, _state_payload(state, self.motion))
        relative = str(path.relative_to(self.layout_dir.parent.parent))
        self._state_links[identity] = relative
        return relative

    @staticmethod
    def _entry_key(state: Any, capture_name: str) -> str:
        sequence = ">".join(str(item) for item in tuple(state.key)) or "START"
        return f"{sequence}|capture={capture_name}"

    def _verify_route_linkage(self, row: Mapping[str, Any], before: Any, after: Any,
                              *, selected: bool) -> dict[str, Any]:
        """Verify q/qdot links without relaxing any kinematics or tracking gate."""

        steps = list(row.get("steps", ()))
        checks: list[dict[str, Any]] = []
        previous_q = _native_payload(before, self.motion)["q"]
        previous_qdot = _native_payload(before, self.motion)["qdot"]
        passed = True
        for index, step in enumerate(steps):
            start_q = np.asarray(step.get("start_q", []), dtype=float)
            start_qdot = np.asarray(step.get("start_qdot", []), dtype=float)
            expected_q = np.asarray(previous_q, dtype=float)
            expected_qdot = np.asarray(previous_qdot, dtype=float)
            q_ok = bool(start_q.shape == expected_q.shape and np.array_equal(start_q, expected_q))
            qdot_ok = bool(start_qdot.shape == expected_qdot.shape and np.array_equal(start_qdot, expected_qdot))
            checks.append({"step": index, "start_q_exact": q_ok, "start_qdot_exact": qdot_ok})
            passed &= q_ok and qdot_ok
            previous_q = step.get("end_q", previous_q)
            previous_qdot = step.get("end_qdot", previous_qdot)
        if selected and after.native is not None and steps:
            end_q = np.asarray(steps[-1].get("end_q", []), dtype=float)
            end_qdot = np.asarray(steps[-1].get("end_qdot", []), dtype=float)
            q_exit_ok = bool(np.array_equal(end_q, np.asarray(after.native.q, dtype=float)))
            qdot_exit_ok = bool(np.array_equal(end_qdot, np.asarray(after.native.qdot, dtype=float)))
            checks[-1].update({"exit_q_exact": q_exit_ok, "exit_qdot_exact": qdot_exit_ok})
            passed &= q_exit_ok and qdot_exit_ok
        trace_links = []
        for step in steps:
            record = step.get("record")
            if record:
                trace = self.layout_dir / "traces" / f"{record}.npz"
                trace_links.append({"record": record, "path": str(trace.relative_to(self.layout_dir.parent.parent)),
                                    "exists": trace.exists()})
                passed &= trace.exists()
        result = {"passed": bool(passed), "checks": checks, "trace_links": trace_links}
        if not passed:
            self.linkage_failures.append({"route": row.get("route"), "checks": checks})
        return result

    def _dird_attempt(self, row: Mapping[str, Any], linkage: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "route_id": row.get("route"),
            "candidate": row.get("candidate"),
            "valid": bool(row.get("valid", False)),
            "reason": "executed" if bool(row.get("valid", False)) else "native_or_workspace_guard",
            "dird_mean_calibrated_margin": row.get("dird_mean_calibrated_margin"),
            "commanded_duration_prediction_s": row.get("commanded_duration_prediction_s"),
            "duration_s": row.get("duration_s"),
            "steps": row.get("steps", ()),
            "linkage": linkage,
            "calculation_wall_seconds": row.get("calculation_wall_seconds"),
        }

    def transition(self, state: Any, capture_name: str, *, diagnostic: bool = False,
                   terminal: bool = False, independent_repeat: bool = False) -> tuple[Any, dict[str, Any]]:
        """Check one transition from an actual state and record its evidence."""

        if not state.valid:
            raise ValueError(f"cannot continue from invalid state {state.key}")
        # Guards apply to one public-history path.  Branches in the catalog
        # share a checked prefix, but an old/new comparison must not consume
        # the diagnostic budget of its sibling branch.
        path_actions = len(tuple(state.key))
        path_diagnostics = sum(item in PROBE_NAMES for item in tuple(state.key))
        path_terminals = sum(item in WINDOW_NAMES for item in tuple(state.key))
        if path_actions >= self.max_actions:
            raise RuntimeError("common action-count guard reached")
        if diagnostic and path_diagnostics >= self.max_diagnostic_decisions:
            raise RuntimeError("diagnostic-decision guard reached")
        if terminal and path_terminals >= self.max_terminal_attempts:
            raise RuntimeError("terminal-attempt guard reached")
        if capture_name not in PROBE_NAMES and capture_name not in WINDOW_NAMES:
            raise ValueError(f"unknown catalog action {capture_name}")
        if capture_name not in self.rig.captures:
            raise ValueError(f"capture {capture_name} missing from rig")

        before = state
        before_signature = _state_signature(before, self.motion)
        entry_link = self._state_link(before, "entry__" + ("_".join(before.key) or "start"))
        row_start = len(self.rollouts.rows)
        dird_before = int(self.rollouts.dird_queries)
        wall_start = time.perf_counter()
        try:
            result = self.rollouts.capture(before, capture_name)
            after = result.transfer.output
            reason = result.transfer.reason
            error = None
        except Exception as exc:  # retain the defect and re-raise after logging
            after = self.fixture.EpisodeState(
                tuple(before.key) + (capture_name,), before.elapsed, before.pose.copy(),
                before.native, False,
            )
            result = None
            reason = "exception_during_native_check"
            error = {"type": type(exc).__name__, "message": str(exc)}

        rows = self.rollouts.rows[row_start:]
        dird_after = int(self.rollouts.dird_queries)
        after_signature = _state_signature(after, self.motion)
        exit_link = self._state_link(after, "exit__" + ("_".join(after.key) or "start"))

        attempts = []
        for row in rows:
            linkage = self._verify_route_linkage(
                row, before, after, selected=bool(result is not None and row.get("route") == result.transfer.route_id)
            )
            attempts.append(self._dird_attempt(row, linkage))

        if result is None:
            capture = self.rig.captures[capture_name]
            measured_action = self.planning.action_at(capture, after.pose, self.rig.measured)
            true_action = self.planning.action_at(capture, after.pose, self.rig.actual_surface)
            selected_route = None
        else:
            capture = result.capture
            measured_action = self.planning.action_at(capture, after.pose, self.rig.measured)
            true_action = self.planning.action_at(capture, after.pose, self.rig.actual_surface)
            selected_route = result.transfer.route_id

        key = self._entry_key(before, capture_name)
        transition = {
            "key": key,
            "sequence": list(before.key),
            "capture": capture_name,
            "capture_kind": capture.kind,
            "independent_repeat": bool(independent_repeat),
            "entry_state_signature": before_signature,
            "exit_state_signature": after_signature,
            "entry_state_link": entry_link,
            "exit_state_link": exit_link,
            "public": {
                "planning_action": _action_dict(measured_action, frame_role="measured"),
                "measured_exit_pose_world": _json_safe(np.asarray(after.pose, dtype=float)),
                "measured_exit_local": _relative_pose(self.modules, self.rig.measured, after.pose),
                "native_state_exposed": "signature_only",
            },
            "evaluation": {
                "true_surface_action": _action_dict(true_action, frame_role="actual_surface"),
                "boundary": "private_evaluation_only",
            },
            "actual_elapsed_s": float(after.elapsed - before.elapsed),
            "actual_exit_elapsed_s": float(after.elapsed),
            "duration_s_reported": None if result is None else float(result.transfer.duration),
            "valid": bool(after.valid),
            "reason": reason,
            "selected_route_id": selected_route,
            "dird": {
                "query_count_before": dird_before,
                "query_count_after": dird_after,
                "query_increment": dird_after - dird_before,
                "attempt_count": len(attempts),
                "attempts": attempts,
            },
            "check_metrics": {
                "route_attempts": len(attempts),
                "valid_route_attempts": sum(bool(item["valid"]) for item in attempts),
                "all_linkage_pass": all(bool(item["linkage"]["passed"]) for item in attempts),
            },
            "trace_links": [
                link
                for item in attempts
                for link in item["linkage"].get("trace_links", ())
            ],
            # Sharded catalog merges retain the original run root so a loader
            # can resolve relative trace/state links without copying hundreds
            # of megabytes of NPZ files.
            "source_run_root": str(self.layout_dir.parent.parent.resolve()),
            "wall_seconds": time.perf_counter() - wall_start,
            "error": error,
        }
        self.transitions[key] = transition
        self.action_count += 1
        if diagnostic:
            self.diagnostic_decisions += 1
        if terminal:
            self.terminal_attempts += 1
        return after, transition

    def finish(self) -> dict[str, Any]:
        self.rollouts.finish()
        payload = {
            "schema": SCHEMA,
            "layout": self.rig.name,
            "fixture": self.rig.public_metadata(),
            "transitions": self.transitions,
            "guards": {
                "max_actions": self.max_actions,
                "actions_used": self.action_count,
                "max_diagnostic_decisions": self.max_diagnostic_decisions,
                "diagnostic_decisions_used": self.diagnostic_decisions,
                "max_terminal_attempts": self.max_terminal_attempts,
                "terminal_attempts_used": self.terminal_attempts,
                "deadline": None,
                "guard_kind": "fixed_common_action_count_only",
            },
            "summary": {
                "transition_count": len(self.transitions),
                "valid_transition_count": sum(bool(item["valid"]) for item in self.transitions.values()),
                "invalid_transition_count": sum(not bool(item["valid"]) for item in self.transitions.values()),
                "dird_queries": int(self.rollouts.dird_queries),
                "native_route_attempts": sum(
                    int(item["dird"]["attempt_count"]) for item in self.transitions.values()
                ),
                "linkage_failures": len(self.linkage_failures),
                "linkage_pass": not self.linkage_failures,
            },
            "linkage_failures": self.linkage_failures,
        }
        _dump(self.layout_dir / "TRANSITIONS.json", payload)
        return payload


def _stage1_sequences() -> tuple[tuple[str, ...], ...]:
    """Shared and direct +3 mm fork witnesses for both constrained layouts."""

    return (
        ("fork_minus",),
        ("fork_minus", "fork_plus"),
        ("fork_minus", "fork_plus", "left"),
        ("fork_minus", "fork_plus", "right"),
        ("fork_minus", "fork_plus_shift3"),
        ("fork_minus", "fork_plus_shift3", "left"),
        ("fork_minus", "fork_plus_shift3", "right"),
        ("fork_plus_shift3",),
        ("fork_plus_shift3", "left"),
        ("fork_plus_shift3", "right"),
    )


def _catalog_sequences() -> tuple[tuple[str, ...], ...]:
    """Return root/one-probe/two-probe terminal continuations.

    Every public-history state gets both first terminal windows and all ordered
    second-terminal choices (same-window repeat and opposite-window retry).
    The resulting sequence set is intentionally larger than a minimum policy
    witness; the fixed action-count guard is the only common computational
    guard and no candidate is pruned because it is inconvenient.
    """

    sequences: list[tuple[str, ...]] = []
    # Direct terminal continuations are part of the reusable catalog as well.
    for first in WINDOW_NAMES:
        sequences.append((first,))
        for second in WINDOW_NAMES:
            sequences.append((first, second))
    for first in PROBE_NAMES:
        sequences.append((first,))
        for first_terminal in WINDOW_NAMES:
            sequences.append((first, first_terminal))
            for second_terminal in WINDOW_NAMES:
                sequences.append((first, first_terminal, second_terminal))
    for first in PROBE_NAMES:
        for second in PROBE_NAMES:
            prefix = (first, second)
            sequences.append(prefix)
            for window in WINDOW_NAMES:
                sequences.append(prefix + (window,))
                for second_terminal in WINDOW_NAMES:
                    # A repeated terminal is a new acquisition event at a new
                    # public-history key, while the mechanical state is still
                    # checked from the prior terminal exit.  Opposite-window
                    # retries are retained under the same rule.
                    sequences.append(prefix + (window, second_terminal))
    # Stable first-use order and no accidental duplicate keys.
    return tuple(dict.fromkeys(sequences))


def _run_layout(modules: Mapping[str, Any], index: int, mode: str, stage: str,
                run_root: Path, *, max_actions: int, max_diagnostic_decisions: int,
                max_terminal_attempts: int, sequences: Sequence[Sequence[str]]) -> dict[str, Any]:
    fixture = modules["fixture"]
    robot_motion = modules["robot_motion"]
    layout_name = f"fixture_{int(index):02d}_{mode}"
    layout_dir = run_root / "layouts" / layout_name
    layout_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with robot_motion.NativeMotionEnvironment(backend="native") as motion:
        base_rig = fixture.make_rig(int(index), mode, motion)
        rig = _add_shift_capture(base_rig, fixture)
        runner = CatalogRunner(
            modules, rig, motion, layout_dir,
            max_actions=max_actions,
            max_diagnostic_decisions=max_diagnostic_decisions,
            max_terminal_attempts=max_terminal_attempts,
        )
        states: dict[tuple[str, ...], Any] = {(): rig.start}
        errors: list[dict[str, Any]] = []
        for sequence in sequences:
            sequence = tuple(sequence)
            # Replay only the missing suffix; all prior transitions are shared
            # and cache identity includes the complete actual state.
            prefix: tuple[str, ...] = ()
            state = states[prefix]
            for position, name in enumerate(sequence):
                next_prefix = prefix + (name,)
                if next_prefix in states:
                    state = states[next_prefix]
                    prefix = next_prefix
                    continue
                diagnostic = name in PROBE_NAMES
                terminal = name in WINDOW_NAMES
                # The common guard is deliberately action-count based; there
                # is no wall-clock deadline or time-based rejection here.
                try:
                    state, _ = runner.transition(
                        state, name,
                        diagnostic=diagnostic,
                        terminal=terminal,
                        independent_repeat=(position > 0 and name == sequence[position - 1]),
                    )
                except Exception as exc:
                    errors.append({
                        "sequence": list(sequence), "prefix": list(prefix),
                        "capture": name, "type": type(exc).__name__, "message": str(exc),
                    })
                    break
                states[next_prefix] = state
                prefix = next_prefix
                if not state.valid:
                    break
        result = runner.finish()
        result["stage"] = stage
        result["layout"] = layout_name
        result["planned_sequences"] = [list(seq) for seq in sequences]
        result["runner_errors"] = errors
        result["wall_seconds_total"] = time.perf_counter() - started
        _dump(layout_dir / "MANIFEST.json", {
            "schema": SCHEMA,
            "stage": stage,
            "layout": layout_name,
            "index": int(index),
            "mode": mode,
            "planned_sequence_count": len(sequences),
            "result": result["summary"],
            "runner_errors": errors,
            "wall_seconds_total": result["wall_seconds_total"],
            "backend": "native",
            "hardware": False,
            "no_new_hard_limits": True,
        })
        return result


def run(stage: str, output: Path, *, layouts: Sequence[int] = DEFAULT_LAYOUTS,
        mode: str = "constrained", max_actions: int | None = None,
        max_diagnostic_decisions: int = 2, max_terminal_attempts: int = 2,
        shard: str | None = None) -> Path:
    """Run Stage 1 or the reusable catalog and return the fresh run directory."""

    if stage not in ("stage1", "catalog"):
        raise ValueError("stage must be stage1 or catalog")
    if not layouts:
        raise ValueError("at least one constrained layout is required")
    if mode != "constrained":
        raise ValueError("native catalog is scoped to constrained layouts")
    if max_diagnostic_decisions < 1 or max_terminal_attempts < 1:
        raise ValueError("diagnostic and terminal guards must be positive")
    sequences = _stage1_sequences() if stage == "stage1" else _catalog_sequences()
    if shard is not None:
        shard = str(shard)
        if stage != "catalog":
            raise ValueError("--shard is only valid for catalog stage")
        if shard == "root":
            sequences = tuple(seq for seq in sequences if seq and seq[0] in WINDOW_NAMES)
        elif shard in PROBE_NAMES:
            sequences = tuple(seq for seq in sequences if seq and seq[0] == shard)
        else:
            raise ValueError(f"unknown catalog shard {shard!r}")
    if max_actions is None:
        max_actions = 7 if stage == "stage1" else 6
    max_actions = int(max_actions)
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    run_root = _allocate_run(Path(output), stage)
    modules = _load_reviewed_modules()
    manifest = {
        "schema": SCHEMA,
        "stage": stage,
        "shard": shard,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "source_manifest": _source_manifest(modules),
        "backend": "native",
        "hardware": False,
        "no_deadline": True,
        "guards": {
            "max_actions": max_actions,
            "max_diagnostic_decisions": max_diagnostic_decisions,
            "max_terminal_attempts": max_terminal_attempts,
            "guard_kind": "fixed_common_action_count_only",
        },
        "probe_contract": {
            "named_probes": list(PROBE_NAMES),
            "diagnostic_decisions_max": int(max_diagnostic_decisions),
            "repeat_allowed": True,
            "repeat_semantics": "independent_new_acquisition_at_new_public_history_key",
            "terminal_attempts_max": int(max_terminal_attempts),
            "public_planning_inputs": ["measured ProbeAction", "measured-frame exit pose"],
            "evaluation_only_inputs": ["true-surface ProbeAction", "private anatomy/quality"],
            "deadline": None,
        },
        "layout_indexes": [int(index) for index in layouts],
        "sequence_count_per_layout": len(sequences),
        "sequence_contract": [list(seq) for seq in sequences],
        "status": "running",
    }
    _dump(run_root / "MANIFEST.json", manifest)
    # Write the cost estimate before native execution starts.  This is a
    # resource guard report, not a deadline and never rejects an action.
    estimate = {
        "schema": "native_catalog_cost_estimate_v1",
        "stage": stage,
        "layouts": len(layouts),
        "planned_sequences_per_layout": len(sequences),
        "planned_unique_prefixes_per_layout": len({prefix for seq in sequences for n in range(len(seq) + 1) for prefix in [tuple(seq[:n])]}),
        "upper_bound_transition_checks": len({prefix for seq in sequences for n in range(1, len(seq) + 1) for prefix in [tuple(seq[:n])]} ) * len(layouts),
        "max_actions_guard_per_layout": max_actions,
        "native_route_alternatives": "unknown until dIRD-ranked route enumeration",
        "deadline": None,
        "guard_kind": "fixed_common_action_count_only",
    }
    _dump(run_root / "COST_ESTIMATE.json", estimate)

    layout_results = []
    started = time.perf_counter()
    for index in layouts:
        layout_results.append(_run_layout(
            modules, int(index), mode, stage, run_root,
            max_actions=max_actions,
            max_diagnostic_decisions=max_diagnostic_decisions,
            max_terminal_attempts=max_terminal_attempts,
            sequences=sequences,
        ))
        # Stage 1 is intentionally observable before any caller opts into the
        # wider catalog.  The file is written after each completed layout.
        _dump(run_root / "STAGE1_PROGRESS.json", {
            "schema": "native_catalog_stage1_progress_v1",
            "stage": stage,
            "completed_layouts": [result.get("layout") for result in layout_results],
            "layout_summaries": [
                {"layout": result.get("layout"), **result.get("summary", {})}
                for result in layout_results
            ],
            "wall_seconds": time.perf_counter() - started,
        })

    manifest.update({
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds_total": time.perf_counter() - started,
        "layout_summaries": [
            {"layout": result.get("layout"), **result.get("summary", {})}
            for result in layout_results
        ],
        "linkage_pass": all(bool(result.get("summary", {}).get("linkage_pass")) for result in layout_results),
        "valid_transition_count": sum(int(result.get("summary", {}).get("valid_transition_count", 0)) for result in layout_results),
        "invalid_transition_count": sum(int(result.get("summary", {}).get("invalid_transition_count", 0)) for result in layout_results),
    })
    _dump(run_root / "MANIFEST.json", manifest)
    _dump(run_root / "CATALOG.json", {
        "schema": SCHEMA,
        "stage": stage,
        "source_manifest": manifest["source_manifest"],
        "probe_contract": manifest["probe_contract"],
        "guards": manifest["guards"],
        "shard": shard,
        "layouts": [
            {"layout": result.get("layout"), "summary": result.get("summary"),
             "transitions_file": f"layouts/{result.get('layout')}/TRANSITIONS.json"}
            for result in layout_results
        ],
    })
    return run_root


def merge_catalogs(inputs: Sequence[Path], output: Path) -> Path:
    """Merge per-first-action catalog shards with strict duplicate checks."""

    if not inputs:
        raise ValueError("at least one catalog shard is required")
    manifests = []
    layout_rows: dict[str, dict[str, Any]] = {}
    source_runs = []
    for root in inputs:
        root = Path(root).resolve()
        manifest_path = root / "MANIFEST.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("stage") != "catalog":
            raise ValueError(f"not a catalog shard: {manifest_path}")
        manifests.append(manifest)
        source_runs.append(str(root))
        for layout in manifest.get("layout_summaries", ()):
            name = str(layout.get("layout"))
            transitions_path = root / "layouts" / name / "TRANSITIONS.json"
            if not transitions_path.exists():
                raise FileNotFoundError(transitions_path)
            payload = json.loads(transitions_path.read_text(encoding="utf-8"))
            current = layout_rows.setdefault(name, {
                "payload": payload,
                "transitions": {},
                "source_runs": [],
            })
            current["source_runs"].append(str(root))
            for key, item in payload.get("transitions", {}).items():
                canonical = json.dumps(_json_safe(item), sort_keys=True, separators=(",", ":"), allow_nan=False)
                previous = current["transitions"].get(str(key))
                if previous is not None:
                    previous_canonical = json.dumps(_json_safe(previous), sort_keys=True, separators=(",", ":"), allow_nan=False)
                    if previous_canonical != canonical:
                        raise ValueError(f"conflicting duplicate transition {name}:{key}")
                    continue
                current["transitions"][str(key)] = item
    first = manifests[0]
    for manifest in manifests[1:]:
        if manifest.get("source_manifest") != first.get("source_manifest"):
            raise ValueError("catalog shards have different source manifests")
        if manifest.get("probe_contract") != first.get("probe_contract"):
            raise ValueError("catalog shards have different probe contracts")
        if manifest.get("guards", {}).get("max_actions") != first.get("guards", {}).get("max_actions"):
            raise ValueError("catalog shards have different action-count guards")

    run_root = _allocate_run(Path(output), "catalog_merged")
    summaries = []
    for layout_name, current in sorted(layout_rows.items()):
        source_payload = current["payload"]
        merged = dict(source_payload)
        merged["transitions"] = current["transitions"]
        merged["merged_from"] = current["source_runs"]
        merged["summary"] = {
            "transition_count": len(current["transitions"]),
            "valid_transition_count": sum(bool(item.get("valid")) for item in current["transitions"].values()),
            "invalid_transition_count": sum(not bool(item.get("valid")) for item in current["transitions"].values()),
            "linkage_pass": all(bool(item.get("check_metrics", {}).get("all_linkage_pass", False))
                                 for item in current["transitions"].values()),
            "source_shard_count": len(current["source_runs"]),
        }
        path = run_root / "layouts" / layout_name / "TRANSITIONS.json"
        _dump(path, merged)
        summaries.append({"layout": layout_name, "summary": merged["summary"],
                          "transitions_file": str(path.relative_to(run_root))})
    merged_manifest = {
        "schema": SCHEMA,
        "stage": "catalog",
        "shard": "merged",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "source_runs": source_runs,
        "source_manifest": first.get("source_manifest"),
        "probe_contract": first.get("probe_contract"),
        "guards": first.get("guards"),
        "layout_summaries": summaries,
        "duplicate_policy": "identical canonical transition required; conflicting duplicate rejected",
        "trace_link_policy": "transition source_run_root plus relative trace/state links",
        "status": "complete",
    }
    _dump(run_root / "MANIFEST.json", merged_manifest)
    _dump(run_root / "CATALOG.json", {
        "schema": SCHEMA,
        "stage": "catalog",
        "shard": "merged",
        "source_manifest": first.get("source_manifest"),
        "probe_contract": first.get("probe_contract"),
        "guards": first.get("guards"),
        "source_runs": source_runs,
        "layouts": summaries,
    })
    return run_root


def _parse_layouts(value: str) -> tuple[int, ...]:
    indexes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not indexes or any(index < 0 for index in indexes):
        raise argparse.ArgumentTypeError("layouts must be comma-separated nonnegative indexes")
    return indexes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "catalog", "merge"), required=True)
    parser.add_argument("--output", type=Path, default=HERE / "native_outputs",
                        help="parent directory; a fresh child is always created")
    parser.add_argument("--layouts", type=_parse_layouts, default=DEFAULT_LAYOUTS,
                        help="constrained fixture indexes, default 0,1")
    parser.add_argument("--max-actions", type=int, default=None,
                        help="fixed common action-count guard; no deadline")
    parser.add_argument("--max-diagnostic-decisions", type=int, default=2)
    parser.add_argument("--max-terminal-attempts", type=int, default=2)
    parser.add_argument("--shard", choices=("root", *PROBE_NAMES), default=None,
                        help="catalog shard: root terminals or one named first probe")
    parser.add_argument("--inputs", nargs="*", type=Path,
                        help="catalog shard run roots for --stage merge")
    args = parser.parse_args(argv)
    if args.stage == "merge":
        if not args.inputs:
            parser.error("--stage merge requires --inputs RUN_ROOT [...]")
        run_root = merge_catalogs(args.inputs, args.output)
    else:
        run_root = run(
            args.stage, args.output, layouts=args.layouts,
            max_actions=args.max_actions,
            max_diagnostic_decisions=args.max_diagnostic_decisions,
            max_terminal_attempts=args.max_terminal_attempts,
            shard=args.shard,
        )
    print(json.dumps({"run_root": str(run_root), "manifest": str(run_root / "MANIFEST.json"),
                      "catalog": str(run_root / "CATALOG.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

