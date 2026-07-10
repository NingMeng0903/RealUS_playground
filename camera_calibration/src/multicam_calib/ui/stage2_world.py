"""Stage 2 UI: floor plane, bed height, and bed-corner envelope world alignment."""
from __future__ import annotations

import traceback
from dataclasses import dataclass

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from multicam_calib.board.apriltag_board import BoardGeometry
from multicam_calib.board.detector import AprilTagDetector
from multicam_calib.calib.world_align import (
    Stage2Report,
    basis_from_aligned_state,
    run_stage2_phase,
    validate_bed_capture,
    validate_corner_capture,
    validate_floor_capture,
)
from multicam_calib.io.config import AppConfig, WorldConfig
from multicam_calib.io.results import ExtrinsicsSet, Intrinsics, extrinsics_rel_path, load_extrinsics
from multicam_calib.recording.session import ViewDetections
from multicam_calib.recording.stage2_session import Stage2Phase, Stage2SessionBundle
from multicam_calib.ui.live_view import LiveViewGrid
from multicam_calib.ui.param_refresh import load_stage1_extrinsics, refresh_intrinsics_cache, stage1_rmse_label


PHASE_LABELS: dict[Stage2Phase, str] = {
    "floor": "Ground plane",
    "bed": "Bed plane",
    "corners": "Bed corners",
}

RUN_LABELS: dict[Stage2Phase, str] = {
    "floor": "Run: Ground plane",
    "bed": "Run: Bed plane",
    "corners": "Run: Bed corners",
}

LOAD_LAST_LABELS: dict[Stage2Phase, str] = {
    "floor": "Load last ground",
    "bed": "Load last bed",
    "corners": "Load last corners",
}


@dataclass
class Stage2Deps:
    aliases: list[str]
    board_geom: BoardGeometry
    intrinsics: dict[str, Intrinsics]
    app_cfg: AppConfig
    world_cfg: WorldConfig
    detector: AprilTagDetector


class _Stage2Worker(QThread):
    finished_ok = pyqtSignal(object)
    finished_err = pyqtSignal(str)

    def __init__(
        self,
        bundle: Stage2SessionBundle,
        deps: Stage2Deps,
        stage1: ExtrinsicsSet,
        phase: Stage2Phase,
    ) -> None:
        super().__init__()
        self._bundle = bundle
        self._deps = deps
        self._stage1 = stage1
        self._phase = phase

    def run(self) -> None:
        try:
            report = run_stage2_phase(
                bundle=self._bundle,
                phase=self._phase,
                board_geom=self._deps.board_geom,
                intrinsics=self._deps.intrinsics,
                stage1=self._stage1,
                app_cfg=self._deps.app_cfg,
                world_cfg=self._deps.world_cfg,
            )
        except Exception:
            self.finished_err.emit(traceback.format_exc())
        else:
            self.finished_ok.emit(report)


class Stage2Panel(QWidget):
    def __init__(
        self,
        *,
        live_view: LiveViewGrid,
        bundle: Stage2SessionBundle,
        deps: Stage2Deps,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._live = live_view
        self._bundle = bundle
        self._deps = deps
        self._worker = None

        root = QVBoxLayout(self)
        root.addWidget(self._live, 1)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Capture mode:"))
        self._phase_cb = QComboBox()
        for ph in ("floor", "bed", "corners"):
            self._phase_cb.addItem(PHASE_LABELS[ph], ph)
        mode_row.addWidget(self._phase_cb)
        self._btn_load_last = QPushButton("Load last bed")
        self._btn_load_last.setToolTip("Optional: copy this phase's samples from last/ into working/.")
        mode_row.addWidget(self._btn_load_last)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        controls = QHBoxLayout()
        self._btn_capture = QPushButton("Capture (Space)")
        self._btn_capture.setShortcut("Space")
        self._btn_remove = QPushButton("Remove last")
        self._btn_delete = QPushButton("Delete selected")
        self._btn_clear = QPushButton("Clear this phase")
        self._btn_clear_floor = QPushButton("New floor calibration…")
        self._btn_run = QPushButton()
        self._btn_run.setMinimumWidth(200)
        self._btn_run.setToolTip("Runs only the phase selected in Capture mode (one button, label follows dropdown).")
        controls.addWidget(self._btn_capture)
        controls.addWidget(self._btn_remove)
        controls.addWidget(self._btn_delete)
        controls.addWidget(self._btn_clear)
        controls.addWidget(self._btn_clear_floor)
        controls.addStretch(1)
        controls.addWidget(self._btn_run)
        root.addLayout(controls)

        body = QHBoxLayout()
        self._samples = QListWidget()
        self._samples.setMaximumWidth(360)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        body.addWidget(self._samples)
        body.addWidget(self._log, 1)
        root.addLayout(body, 1)

        self._status = QLabel("")
        root.addWidget(self._status)

        self._btn_capture.clicked.connect(self._on_capture)
        self._btn_remove.clicked.connect(self._on_remove)
        self._btn_delete.clicked.connect(self._on_delete_selected)
        self._btn_clear.clicked.connect(self._on_clear_phase)
        self._btn_clear_floor.clicked.connect(self._on_clear_floor)
        self._btn_run.clicked.connect(self._on_run)
        self._phase_cb.currentIndexChanged.connect(self._on_phase_changed)
        self._btn_load_last.clicked.connect(self._on_load_last)

        self._sync_run_button()
        self._update_load_last_button()
        self._refresh()

    def on_intrinsics_updated(self) -> None:
        """Called when Stage 0 writes new intrinsics — no restart needed."""
        refresh_intrinsics_cache(self._deps.intrinsics)
        self._log.append("Stage 0 saved new intrinsics — Stage 2 will use them on next capture/run.")
        self._refresh()

    def on_stage1_updated(self) -> None:
        """Called when Stage 1 writes new extrinsics — no restart needed."""
        self._log.append(f"Stage 1 saved new extrinsics — Stage 2 will use them ({stage1_rmse_label()}).")
        self._refresh()

    def _update_load_last_button(self) -> None:
        phase = self._current_phase()
        self._btn_load_last.setText(LOAD_LAST_LABELS[phase])
        n = Stage2SessionBundle.last_phase_sample_count(phase)
        if n > 0:
            self._btn_load_last.setEnabled(True)
            self._btn_load_last.setToolTip(
                f"Copy {n} saved {PHASE_LABELS[phase].lower()} sample(s) from last/ into working/."
            )
        else:
            self._btn_load_last.setEnabled(False)
            self._btn_load_last.setToolTip(f"No saved {PHASE_LABELS[phase].lower()} samples in last/.")

    def _on_load_last(self) -> None:
        phase = self._current_phase()
        n_last = Stage2SessionBundle.last_phase_sample_count(phase)
        if n_last == 0:
            QMessageBox.information(
                self,
                "No last data",
                f"No saved {PHASE_LABELS[phase].lower()} samples in last/.",
            )
            self._update_load_last_button()
            return
        n_working = len(self._bundle.phase_session(phase).samples)
        if n_working > 0:
            ans = QMessageBox.warning(
                self,
                LOAD_LAST_LABELS[phase],
                f"working/ already has {n_working} {PHASE_LABELS[phase].lower()} sample(s).\n\n"
                f"Loading from last/ will PERMANENTLY replace them with {n_last} older sample(s).\n"
                f"(There is no undo — only proceed if you mean to restore the old set.)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
        else:
            ans = QMessageBox.question(
                self,
                LOAD_LAST_LABELS[phase],
                f"Copy {n_last} saved {PHASE_LABELS[phase].lower()} sample(s) from last/ into working/?\n"
                "(Other phases in working/ are not changed.)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
        if ans != QMessageBox.Yes:
            return
        if not Stage2SessionBundle.copy_last_phase_to_working(phase):
            QMessageBox.information(self, "No last data", "Copy failed — last/ has no data for this phase.")
            self._update_load_last_button()
            return
        self._bundle.phase_session(phase).load_existing()
        self._bundle.write_manifest()
        self._log.append(
            f"Loaded {n_last} {PHASE_LABELS[phase].lower()} sample(s) from last/ "
            f"(replaced {n_working} working sample(s))."
        )
        inherited = self._bundle.inherit_prereq_alignment_from_last(phase)
        if inherited:
            self._log.append(
                "Also inherited from last/ (needed to Run this phase): " + "; ".join(inherited)
            )
        self._sync_run_button()
        self._refresh()

    def _sync_run_button(self) -> None:
        """Single Run button — label always matches Capture mode dropdown."""
        phase = self._current_phase()
        self._btn_run.setText(RUN_LABELS[phase])

    def _current_phase(self) -> Stage2Phase:
        return self._phase_cb.currentData()

    def _phase_session(self):
        return self._bundle.phase_session(self._current_phase())

    def _on_phase_changed(self, _index: int = 0) -> None:
        self._sync_run_button()
        self._update_load_last_button()
        self._refresh()

    def _refresh(self) -> None:
        wc = self._deps.world_cfg
        state = self._bundle.load_aligned_state()
        phase = self._current_phase()
        self._samples.clear()
        ps = self._bundle.phase_session(phase)
        for s in ps.samples:
            counts = ", ".join(f"{a}:{v.num_tags()}" for a, v in s.views.items())
            meta = s.metadata.get("corner_gate", "")
            extra = f"  {meta}" if meta else ""
            item = QListWidgetItem(f"#{s.index:03d}  {counts}{extra}")
            item.setData(Qt.UserRole, (phase, s.index))
            self._samples.addItem(item)
        aligned = []
        if state.floor_aligned:
            aligned.append(f"floor RMSE {state.floor_plane_residual_mm:.1f}mm")
        if state.bed_aligned and state.bed_height_m is not None:
            aligned.append(f"bed z={state.bed_height_m * 1000:.0f}mm")
        if state.corners_aligned:
            aligned.append("world exported")
        aligned_txt = "   aligned: " + ", ".join(aligned) if aligned else "   aligned: (none yet)"
        cur_n = len(ps.samples)
        min_n = {
            "floor": wc.min_floor_samples,
            "bed": wc.min_bed_samples,
            "corners": wc.min_corner_samples,
        }[phase]
        self._status.setText(
            f"{PHASE_LABELS[phase]}   "
            f"samples: {cur_n}/{min_n}   "
            f"(all phases: floor {len(self._bundle.floor.samples)}, "
            f"bed {len(self._bundle.bed.samples)}, "
            f"corners {len(self._bundle.corners.samples)})"
            f"{aligned_txt}   "
            f"{stage1_rmse_label()}"
        )

    def _snap_images_and_views(self):
        snap = self._live.snapshot_now()
        if snap is None:
            return None
        images = {alias: img for alias, (img, _, _) in snap.items()}
        views: dict[str, ViewDetections] = {}
        for alias, (_, dets, _) in snap.items():
            views[alias] = ViewDetections(
                alias=alias,
                tags=self._deps.detector.detections_to_dict(dets),
            )
        ts_list = [ts for (_, _, ts) in snap.values()]
        return images, views, ts_list

    def _on_capture(self) -> None:
        got = self._snap_images_and_views()
        if got is None:
            self._log.append("Capture failed: no frames yet.")
            return
        images, views, ts_list = got
        spread_ms = (max(ts_list) - min(ts_list)) / 1e6
        if spread_ms > self._deps.app_cfg.sync.max_spread_ms:
            self._log.append(f"Rejected: timestamp spread {spread_ms:.1f} ms > limit.")
            return

        phase = self._current_phase()
        metadata: dict = {"host_ts_spread_ms": spread_ms, "phase": phase}
        state = self._bundle.load_aligned_state()
        host_ts = int(sum(ts_list) / len(ts_list))

        if phase == "floor":
            refresh_intrinsics_cache(self._deps.intrinsics)
            stage1 = load_stage1_extrinsics()
            if stage1 is None:
                self._log.append("Stage 1 extrinsics missing — run Stage 1 first.")
                return
            preview = validate_floor_capture(
                views=views,
                host_timestamp_ns=host_ts,
                board_geom=self._deps.board_geom,
                intrinsics=self._deps.intrinsics,
                stage1=stage1,
                app_cfg=self._deps.app_cfg,
                world_cfg=self._deps.world_cfg,
                state=state,
                bed_samples=self._bundle.bed.samples,
            )
            if not preview.ok:
                self._log.append(f"Floor capture rejected: {preview.message}")
                return
            metadata["floor_gate"] = preview.message

        elif phase == "bed":
            refresh_intrinsics_cache(self._deps.intrinsics)
            stage1 = load_stage1_extrinsics()
            if stage1 is None:
                self._log.append("Stage 1 extrinsics missing — run Stage 1 first.")
                return
            preview = validate_bed_capture(
                views=views,
                board_geom=self._deps.board_geom,
                intrinsics=self._deps.intrinsics,
                stage1=stage1,
                app_cfg=self._deps.app_cfg,
                world_cfg=self._deps.world_cfg,
                state=state,
            )
            if not preview.ok:
                self._log.append(f"Bed capture rejected: {preview.message}")
                return
            metadata["bed_gate"] = preview.message

        elif phase == "corners":
            refresh_intrinsics_cache(self._deps.intrinsics)
            stage1 = load_stage1_extrinsics()
            if stage1 is None:
                self._log.append("Stage 1 extrinsics missing — run Stage 1 first.")
                return
            preview = validate_corner_capture(
                views=views,
                board_geom=self._deps.board_geom,
                intrinsics=self._deps.intrinsics,
                stage1=stage1,
                world_cfg=self._deps.world_cfg,
                app_cfg=self._deps.app_cfg,
                basis=(
                    basis_from_aligned_state(self._bundle.load_aligned_state())
                    if self._bundle.load_aligned_state().floor_aligned
                    else None
                ),
            )
            if not preview.ok:
                self._log.append(f"Corner capture rejected: {preview.message}")
                return
            metadata["corner_gate"] = preview.message

        ps = self._bundle.phase_session(phase)
        sample = ps.add_sample(
            images_bgr=images,
            host_timestamp_ns=host_ts,
            metadata=metadata,
        )
        self._bundle.write_manifest()
        self._log.append(f"[{phase}] Sample #{sample.index:03d} captured — spread {spread_ms:.1f} ms")
        gate = metadata.get("corner_gate") or metadata.get("floor_gate") or metadata.get("bed_gate")
        if gate:
            self._log.append(f"  {gate}")
        self._refresh()

    def _on_remove(self) -> None:
        phase = self._current_phase()
        s = self._bundle.phase_session(phase).remove_last()
        if s is not None:
            self._bundle.write_manifest()
            self._log.append(f"[{phase}] Removed sample #{s.index:03d}.")
        self._refresh()

    def _on_delete_selected(self) -> None:
        item = self._samples.currentItem()
        if item is None:
            self._log.append("Select a sample in the list to delete.")
            return
        data = item.data(Qt.UserRole)
        if not data:
            return
        phase, sample_index = data
        ans = QMessageBox.question(
            self,
            "Delete sample",
            f"Delete [{phase}] sample #{sample_index:03d}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        removed = self._bundle.phase_session(phase).remove_sample_by_index(int(sample_index))
        if removed is None:
            self._log.append(f"Failed to delete [{phase}] #{sample_index:03d}.")
            return
        if phase == "floor":
            state = self._bundle.load_aligned_state()
            state.floor_aligned = False
            state.bed_aligned = False
            state.corners_aligned = False
            state.bed_height_m = None
            state.bed_plane_residual_mm = None
            self._bundle.save_aligned_state(state)
            Stage2SessionBundle._remove_world_results()
        elif phase == "bed":
            state = self._bundle.load_aligned_state()
            state.bed_aligned = False
            state.corners_aligned = False
            state.bed_height_m = None
            state.bed_plane_residual_mm = None
            self._bundle.save_aligned_state(state)
            Stage2SessionBundle._remove_world_results()
        elif phase == "corners":
            state = self._bundle.load_aligned_state()
            state.corners_aligned = False
            self._bundle.save_aligned_state(state)
            Stage2SessionBundle._remove_world_results()
        self._bundle.write_manifest()
        self._log.append(f"[{phase}] Deleted sample #{sample_index:03d} (number unchanged for remaining samples).")
        self._refresh()

    def _on_clear_phase(self) -> None:
        phase = self._current_phase()
        self._bundle.phase_session(phase).clear()
        if phase == "bed":
            self._bundle.invalidate_from_bed()
        elif phase == "floor":
            self._log.append("Use 'New floor calibration' to also clear bed and corners.")
        self._bundle.write_manifest()
        self._log.append(f"[{phase}] Cleared all samples in this phase.")
        self._refresh()

    def _on_clear_floor(self) -> None:
        ans = QMessageBox.question(
            self,
            "New floor calibration",
            "This clears ALL floor, bed, and corner samples and deletes world results. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        self._bundle.floor.clear()
        self._bundle.invalidate_from_floor()
        self._log.append("New floor calibration: cleared floor, bed, corners, and world yaml.")
        self._refresh()

    def _on_run(self) -> None:
        refresh_intrinsics_cache(self._deps.intrinsics)
        stage1 = load_stage1_extrinsics()
        if stage1 is None:
            QMessageBox.warning(self, "Stage 1 missing", f"{extrinsics_rel_path()} not found.")
            return

        phase = self._current_phase()
        wc = self._deps.world_cfg
        state = self._bundle.load_aligned_state()

        if phase == "floor":
            if len(self._bundle.floor.samples) < wc.min_floor_samples:
                QMessageBox.warning(
                    self, "Not enough floor samples", f"Need >= {wc.min_floor_samples} floor captures."
                )
                return
        elif phase == "bed":
            if not state.floor_aligned:
                QMessageBox.warning(self, "Floor not aligned", "Run floor alignment first.")
                return
            if len(self._bundle.bed.samples) < wc.min_bed_samples:
                QMessageBox.warning(self, "Not enough bed samples", f"Need >= {wc.min_bed_samples} bed captures.")
                return
        else:
            if not state.floor_aligned:
                QMessageBox.warning(self, "Floor not aligned", "Run floor alignment first.")
                return
            if not state.bed_aligned:
                QMessageBox.warning(self, "Bed not aligned", "Run bed height alignment first.")
                return
            if len(self._bundle.corners.samples) < wc.min_corner_samples:
                QMessageBox.warning(
                    self, "Not enough corner samples", f"Need >= {wc.min_corner_samples} corner captures."
                )
                return

        self._log.append(f"Running Stage 2 — {RUN_LABELS[phase]}…")
        self._btn_run.setEnabled(False)
        worker = _Stage2Worker(self._bundle, self._deps, stage1, phase)
        self._worker = worker
        worker.finished_ok.connect(self._on_ok)
        worker.finished_err.connect(self._on_err)
        worker.start()

    def _on_ok(self, report: Stage2Report) -> None:
        self._btn_run.setEnabled(True)
        lines = [f"Stage 2 [{report.phase}] done."]
        lines.append(f"  Floor plane RMSE: {report.floor_residual_mm:.2f} mm")
        if report.bed_height_m is not None:
            lines.append(
                f"  Bed height: {report.bed_height_m * 1000:.1f} mm "
                f"(parallel plane RMSE {report.bed_residual_mm:.2f} mm)"
            )
        if report.world_meta is not None and report.phase == "corners":
            m = report.world_meta
            if m.xy_aligned_to_bed:
                lines.append(
                    f"  Bed size (m): {m.bed_size_m[0]:.3f} x {m.bed_size_m[1]:.3f} "
                    f"(world X/Y aligned to bed; pre-align skew {m.bed_xy_skew_deg_pre_align:.1f} deg)"
                )
            else:
                lines.append(
                    f"  Bed size (m): {m.bed_size_m[0]:.3f} x {m.bed_size_m[1]:.3f} "
                    f"(rotated {m.bed_rotation_deg:.1f} deg from world X)"
                )
            lines.append(f"  Origin (floor): {m.bed_center_on_floor}")
            lines.append(f"  Bed center (world): {m.bed_center_world}")
            lines.append("  Exported: calibration_results/genesis_bundle.yaml")
        if report.phase in ("floor", "bed"):
            Stage2SessionBundle.archive_working_phase_as_last(report.phase)
        elif report.phase == "corners":
            Stage2SessionBundle.archive_working_as_last()
        self._log.append("\n".join(lines))
        QMessageBox.information(self, f"Stage 2 [{report.phase}] done", lines[1])
        self._update_load_last_button()
        self._refresh()

    def _on_err(self, err: str) -> None:
        self._btn_run.setEnabled(True)
        self._log.append("Stage 2 FAILED:\n" + err)
        QMessageBox.critical(self, "Stage 2 error", err.splitlines()[-1] if err else "unknown")
