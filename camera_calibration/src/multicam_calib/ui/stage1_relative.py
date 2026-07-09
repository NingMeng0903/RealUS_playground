"""Stage 1 UI panel: capture -> compute relative extrinsics.

Composed on top of :class:`LiveViewGrid`. This module only wires buttons and
progress reporting; the heavy lifting lives in :mod:`multicam_calib.calib.run_stage1`.
"""
from __future__ import annotations

import traceback
import time
from collections.abc import Callable
from dataclasses import dataclass

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QShortcut,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from multicam_calib.board.apriltag_board import BoardGeometry
from multicam_calib.calib.run_stage1 import Stage1Report, run_stage1
from multicam_calib.io.config import AppConfig
from multicam_calib.io.results import Intrinsics
from multicam_calib.recording.session import RecordingSession
from multicam_calib.ui.live_view import LiveViewGrid
from multicam_calib.ui.param_refresh import refresh_intrinsics_cache


@dataclass
class Stage1Deps:
    aliases: list[str]
    board_geom: BoardGeometry
    intrinsics: dict[str, Intrinsics]
    app_cfg: AppConfig


class _Stage1Worker(QThread):
    """Runs `run_stage1` off the UI thread."""

    finished_ok = pyqtSignal(object)  # Stage1Report
    finished_err = pyqtSignal(str)

    def __init__(self, session: RecordingSession, deps: Stage1Deps) -> None:
        super().__init__()
        self.session = session
        self.deps = deps

    def run(self) -> None:  # noqa: D401 - QThread override
        try:
            refresh_intrinsics_cache(self.deps.intrinsics)
            report = run_stage1(
                session=self.session,
                board_geom=self.deps.board_geom,
                intrinsics=self.deps.intrinsics,
                reference=self.deps.aliases[0],
                app_cfg=self.deps.app_cfg,
            )
        except Exception:  # noqa: BLE001
            self.finished_err.emit(traceback.format_exc())
        else:
            self.finished_ok.emit(report)


class Stage1Panel(QWidget):
    """UI panel that owns a `RecordingSession` and drives Stage 1 calibration."""

    def __init__(
        self,
        *,
        live_view: LiveViewGrid,
        session: RecordingSession,
        deps: Stage1Deps,
        on_extrinsics_saved: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._live = live_view
        self._session = session
        self._deps = deps
        self._on_extrinsics_saved = on_extrinsics_saved
        self._worker: _Stage1Worker | None = None
        self._last_capture_mono_ns: int = 0
        self._capture_debounce_ms = 400

        root = QVBoxLayout(self)
        root.addWidget(self._live, 1)

        session_row = QHBoxLayout()
        self._btn_load_last = QPushButton("Load last session")
        self._btn_load_last.setToolTip("Optional: restore the previous saved capture session into this empty session.")
        session_row.addWidget(self._btn_load_last)
        session_row.addStretch(1)
        root.addLayout(session_row)

        controls = QHBoxLayout()
        self._btn_capture = QPushButton("Capture (Space)")
        self._btn_remove = QPushButton("Remove last")
        self._btn_delete = QPushButton("Delete selected")
        self._btn_clear = QPushButton("Clear")
        self._btn_run = QPushButton("Run calibration")
        controls.addWidget(self._btn_capture)
        controls.addWidget(self._btn_remove)
        controls.addWidget(self._btn_delete)
        controls.addWidget(self._btn_clear)
        controls.addStretch(1)
        controls.addWidget(self._btn_run)
        root.addLayout(controls)

        body = QHBoxLayout()
        self._samples = QListWidget()
        self._samples.setMaximumWidth(320)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        body.addWidget(self._samples)
        body.addWidget(self._log, 1)
        root.addLayout(body, 1)

        self._status = QLabel(self._status_text())
        root.addWidget(self._status)

        self._btn_capture.clicked.connect(self._on_capture)
        self._btn_remove.clicked.connect(self._on_remove_last)
        self._btn_delete.clicked.connect(self._on_delete_selected)
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_load_last.clicked.connect(self._on_load_last)

        # Space on the button can fire twice (shortcut + click) on some Qt builds.
        # Use one panel-level shortcut plus a short debounce instead.
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._on_capture)

        self._update_load_last_button()
        self._refresh_samples_list()

    def on_intrinsics_updated(self) -> None:
        """Called when Stage 0 writes new intrinsics — no restart needed."""
        refresh_intrinsics_cache(self._deps.intrinsics)
        self._log.append("Stage 0 saved new intrinsics — next Run will use them automatically.")

    def _status_text(self) -> str:
        return (
            f"Session: {self._session.session_dir.name}   "
            f"samples: {len(self._session.samples)}   "
            f"cameras: {', '.join(self._deps.aliases)}"
        )

    def _update_load_last_button(self) -> None:
        if RecordingSession.last_exists("stage1_extrinsics"):
            self._btn_load_last.setEnabled(True)
            self._btn_load_last.setText(f"Load {RecordingSession.last_count_label('stage1_extrinsics')}")
        else:
            self._btn_load_last.setEnabled(False)
            self._btn_load_last.setText("Load last session")

    def _on_load_last(self) -> None:
        ans = QMessageBox.question(
            self,
            "Load last session",
            "Replace the current empty session with the last saved capture session?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        loaded = RecordingSession.load_last_into_working(
            stage="stage1_extrinsics",
            aliases=self._deps.aliases,
            detector=self._session.detector,
            recording_cfg=self._deps.app_cfg.recording,
        )
        if loaded is None:
            QMessageBox.information(self, "No last session", "No saved last session found on disk.")
            self._update_load_last_button()
            return
        self._session = loaded
        self._log.append(f"Loaded {RecordingSession.last_count_label('stage1_extrinsics')} into working session.")
        self._refresh_samples_list()

    def _refresh_samples_list(self) -> None:
        min_tags = int(self._deps.app_cfg.calibration.min_tags_per_view)
        self._samples.clear()
        for s in self._session.samples:
            qual = [a for a, v in s.views.items() if v.num_tags() >= min_tags]
            skip = [a for a, v in s.views.items() if v.num_tags() < min_tags]
            parts = [f"BA:{','.join(qual) or 'none'}"]
            if skip:
                parts.append("skip:" + ",".join(f"{a}({s.views[a].num_tags()})" for a in skip))
            item = QListWidgetItem(f"#{s.index:03d}  {'  '.join(parts)}")
            item.setData(Qt.UserRole, s.index)
            self._samples.addItem(item)
        self._status.setText(self._status_text())

    def _on_capture(self) -> None:
        now = time.monotonic_ns()
        if (now - self._last_capture_mono_ns) < self._capture_debounce_ms * 1_000_000:
            return
        self._last_capture_mono_ns = now

        snap = self._live.snapshot_now()
        if snap is None:
            self._log.append("Capture failed: not all cameras have a frame yet.")
            return
        ts_list = [ts for (_, _, ts) in snap.values()]
        spread_ms = (max(ts_list) - min(ts_list)) / 1e6
        if spread_ms > self._deps.app_cfg.sync.max_spread_ms:
            self._log.append(
                f"Rejected: host timestamp spread {spread_ms:.1f} ms exceeds "
                f"{self._deps.app_cfg.sync.max_spread_ms:.0f} ms limit. "
                f"Wait a moment and capture again (board must be still)."
            )
            return
        images = {alias: img for alias, (img, _, _) in snap.items()}
        sample = self._session.add_sample(
            images_bgr=images,
            host_timestamp_ns=int(sum(ts_list) / len(ts_list)),
            metadata={"host_ts_spread_ms": spread_ms},
        )
        counts = ", ".join(f"{a}:{v.num_tags()}" for a, v in sample.views.items())
        min_tags = int(self._deps.app_cfg.calibration.min_tags_per_view)
        qual = [a for a, v in sample.views.items() if v.num_tags() >= min_tags]
        hint = int(self._deps.app_cfg.calibration.min_qualifying_cameras_hint)
        n_cams = len(self._deps.aliases)
        self._log.append(
            f"Sample #{sample.index:03d} captured — spread {spread_ms:.1f} ms\n"
            f"  tags: {counts}\n"
            f"  included in BA: {len(qual)}/{n_cams} → {', '.join(qual) or 'none'}"
        )
        if len(qual) < hint:
            self._log.append(
                f"  Warning: only {len(qual)} camera(s) qualify (recommend ≥{hint}). "
                f"Reposition the board (left / right / head / foot) and capture again."
            )
        self._refresh_samples_list()

    def _on_remove_last(self) -> None:
        s = self._session.remove_last()
        if s is not None:
            self._log.append(f"Removed sample #{s.index:03d}.")
        self._refresh_samples_list()

    def _on_delete_selected(self) -> None:
        item = self._samples.currentItem()
        if item is None:
            self._log.append("Select a sample in the list to delete.")
            return
        sample_index = item.data(Qt.UserRole)
        if sample_index is None:
            return
        ans = QMessageBox.question(
            self,
            "Delete sample",
            f"Delete sample #{int(sample_index):03d}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        removed = self._session.remove_sample_by_index(int(sample_index))
        if removed is None:
            self._log.append(f"Failed to delete #{int(sample_index):03d}.")
            return
        self._log.append(f"Deleted sample #{int(sample_index):03d} (number unchanged for remaining samples).")
        self._refresh_samples_list()

    def _on_clear(self) -> None:
        self._session.clear()
        self._log.append("Cleared all samples.")
        self._refresh_samples_list()

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        refresh_intrinsics_cache(self._deps.intrinsics)
        self._log.append("Running Stage 1 calibration (PnP -> pose averaging -> BA) ...")
        self._status.setText(self._status_text() + "   |   computing …")
        self._btn_run.setEnabled(False)
        self._worker = _Stage1Worker(self._session, self._deps)
        self._worker.finished_ok.connect(self._on_run_ok)
        self._worker.finished_err.connect(self._on_run_err)
        self._worker.start()

    def _on_run_ok(self, report: Stage1Report) -> None:
        self._btn_run.setEnabled(True)
        self._status.setText(self._status_text())
        from multicam_calib.io.results import extrinsics_rel_path

        RecordingSession.archive_working_as_last("stage1_extrinsics")
        self._update_load_last_button()
        self._log.append(report.summary())
        self._log.append(f"Saved: {extrinsics_rel_path()}")
        if self._on_extrinsics_saved is not None:
            self._on_extrinsics_saved()
        QMessageBox.information(
            self,
            "Stage 1 done",
            f"Total RMSE: {report.ba.total_rmse:.3f} px\n\nSaved to:\n{extrinsics_rel_path()}",
        )

    def _on_run_err(self, err: str) -> None:
        self._btn_run.setEnabled(True)
        self._status.setText(self._status_text())
        self._log.append("Stage 1 FAILED:\n" + err)
        QMessageBox.critical(self, "Stage 1 error", err.splitlines()[-1] if err else "unknown")
