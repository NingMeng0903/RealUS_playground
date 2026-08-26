"""Stage 2 UI: robot hand-eye, bed height, and bed-corner envelope world alignment."""
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
    robot_capture_coverage,
    run_stage2_phase,
    validate_bed_capture,
    validate_corner_capture,
    validate_robot_capture,
)
from multicam_calib.ingress.robot_state import RobotStateReader
from multicam_calib.io.config import AppConfig, RobotConfig, WorldConfig, load_robot, load_world
from multicam_calib.io.results import ExtrinsicsSet, Intrinsics, extrinsics_rel_path, load_extrinsics
from multicam_calib.recording.session import ViewDetections
from multicam_calib.recording.stage2_session import Stage2Phase, Stage2SessionBundle
from multicam_calib.ui.live_view import LiveViewGrid
from multicam_calib.ui.param_refresh import load_stage1_extrinsics, refresh_intrinsics_cache, stage1_rmse_label


PHASE_LABELS: dict[Stage2Phase, str] = {
    "robot": "Robot (hand-eye)",
    "bed": "Bed plane",
    "corners": "Bed corners",
}

RUN_LABELS: dict[Stage2Phase, str] = {
    "robot": "Run: Robot hand-eye",
    "bed": "Run: Bed plane",
    "corners": "Run: Bed corners",
}

LOAD_LAST_LABELS: dict[Stage2Phase, str] = {
    "robot": "Load last robot",
    "bed": "Load last bed",
    "corners": "Load last corners",
}


@dataclass
class Stage2Deps:
    aliases: list[str]
    board_geom: BoardGeometry
    board_geom_ee: BoardGeometry
    intrinsics: dict[str, Intrinsics]
    app_cfg: AppConfig
    world_cfg: WorldConfig
    robot_cfg: RobotConfig
    detector: AprilTagDetector
    detector_ee: AprilTagDetector
    preview_detector: AprilTagDetector
    preview_detector_ee: AprilTagDetector


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
                board_geom_ee=self._deps.board_geom_ee,
                robot_cfg=self._deps.robot_cfg,
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
        self._robot_reader = RobotStateReader(deps.robot_cfg.shm.name)

        root = QVBoxLayout(self)
        root.addWidget(self._live, 1)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Capture mode:"))
        self._phase_cb = QComboBox()
        for ph in ("robot", "bed", "corners"):
            self._phase_cb.addItem(PHASE_LABELS[ph], ph)
        mode_row.addWidget(self._phase_cb)
        mode_row.addWidget(QLabel("Group:"))
        self._group_cb = QComboBox()
        self._group_cb.addItem("Rail scan", "rail_scan")
        self._group_cb.addItem("Pose diversity", "pose_diversity")
        self._group_cb.setToolTip(
            "Label only — does not block capture. Rail scan: freeze the arm and move the rail. "
            "Pose diversity: rotate the wrist to constrain T_tcp_board."
        )
        mode_row.addWidget(self._group_cb)
        self._btn_load_last = QPushButton("Load last robot")
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
        self._btn_clear_floor = QPushButton("New robot calibration…")
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
        self._apply_live_detector()
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
        self._apply_live_detector()
        self._refresh()

    def _apply_live_detector(self) -> None:
        phase = self._current_phase()
        self._group_cb.setEnabled(phase == "robot")
        if phase == "robot":
            self._live.set_board_mode(
                self._deps.detector_ee,
                preview_detector=self._deps.preview_detector_ee,
                min_tags_for_ba=int(self._deps.world_cfg.min_tags_robot_view),
            )
        else:
            self._live.set_board_mode(
                self._deps.detector,
                preview_detector=self._deps.preview_detector,
                min_tags_for_ba=int(self._deps.app_cfg.calibration.min_tags_per_view),
            )

    def _refresh(self) -> None:
        wc = self._deps.world_cfg
        state = self._bundle.load_aligned_state()
        phase = self._current_phase()
        self._samples.clear()
        ps = self._bundle.phase_session(phase)
        for s in ps.samples:
            counts = ", ".join(f"{a}:{v.num_tags()}" for a, v in s.views.items())
            meta = s.metadata.get("corner_gate") or s.metadata.get("robot_gate") or s.metadata.get("bed_gate") or ""
            grp = s.metadata.get("capture_group", "")
            prefix = f"[{grp}] " if grp else ""
            extra = f"  {prefix}{meta}" if (meta or grp) else ""
            item = QListWidgetItem(f"#{s.index:03d}  {counts}{extra}")
            item.setData(Qt.UserRole, (phase, s.index))
            self._samples.addItem(item)
        aligned = []
        if state.floor_aligned:
            tilt = state.baselink_z_tilt_from_world_z_deg
            extra = f", tilt {tilt:.2f}deg" if tilt is not None else ""
            aligned.append(f"robot axes ready{extra}")
        if state.bed_aligned and state.bed_height_m is not None:
            aligned.append(f"bed z={state.bed_height_m * 1000:.0f}mm")
        if state.corners_aligned:
            aligned.append("world exported")
        aligned_txt = "   aligned: " + ", ".join(aligned) if aligned else "   aligned: (none yet)"
        cur_n = len(ps.samples)
        min_n = {
            "robot": wc.min_robot_samples,
            "bed": wc.min_bed_samples,
            "corners": wc.min_corner_samples,
        }[phase]
        cov_txt = ""
        if phase == "robot":
            cov = robot_capture_coverage(ps.samples, min_tags=int(wc.min_tags_robot_view))
            cams = " ".join(f"{a}:{n}" for a, n in sorted(cov["per_camera"].items())) or "none"
            cov_txt = (
                f"   rail {cov['rail_baseline_m']:.3f}m/{cov['n_rail_stations']} sta"
                f"   cams[{cams}]"
                f"   ≥3-cam {cov['n_multicam']}"
                f"   scan {cov['n_rail_scan']} pose {cov['n_pose_diversity']}"
            )
        self._status.setText(
            f"{PHASE_LABELS[phase]}   "
            f"samples: {cur_n}/{min_n}   "
            f"(all phases: robot {len(self._bundle.robot.samples)}, "
            f"bed {len(self._bundle.bed.samples)}, "
            f"corners {len(self._bundle.corners.samples)})"
            f"{cov_txt}"
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
        self._deps.world_cfg = load_world()
        self._deps.robot_cfg = load_robot()
        state = self._bundle.load_aligned_state()
        host_ts = int(sum(ts_list) / len(ts_list))

        if phase == "robot":
            rc = self._deps.robot_cfg
            snap, still = self._robot_reader.wait_still(
                window_s=rc.stillness.window_s,
                trans_m=rc.stillness.trans_m,
                rot_deg=rc.stillness.rot_deg,
                rail_m=rc.stillness.rail_m,
            )
            shm_ok, shm_age = self._robot_reader.is_fresh(rc.shm.max_age_s)
            if snap is None:
                self._log.append(
                    f"Robot capture rejected: {still.message or self._robot_reader.last_error or 'SHM missing'}"
                )
                return
            preview = validate_robot_capture(
                views=views,
                world_cfg=self._deps.world_cfg,
                shm_ok=shm_ok,
                shm_age_s=shm_age,
                still_ok=still.ok,
                still_message=still.message,
                rail_m=snap.rail_m,
                max_age_s=rc.shm.max_age_s,
            )
            if not preview.ok:
                self._log.append(f"Robot capture rejected: {preview.message}")
                return
            metadata["robot_gate"] = preview.message
            metadata["capture_group"] = str(self._group_cb.currentData() or "rail_scan")
            if snap is not None:
                metadata["rail_m"] = float(snap.rail_m)
                metadata["q_deg"] = [float(v) for v in snap.q_deg.tolist()]
                metadata["pose"] = [float(v) for v in snap.pose.tolist()]
                metadata["T_railbase_tcp"] = snap.T_railbase_tcp().tolist()
                metadata["shm_seq"] = int(snap.seq)
                metadata["shm_t_s"] = float(snap.t_s)

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
        gate = metadata.get("corner_gate") or metadata.get("robot_gate") or metadata.get("bed_gate")
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
        if phase == "robot":
            state = self._bundle.load_aligned_state()
            state.floor_aligned = False
            state.bed_aligned = False
            state.corners_aligned = False
            state.bed_height_m = None
            state.bed_plane_residual_mm = None
            state.T_ref_railbase = None
            state.T_tcp_board = None
            state.robot_diagnostics = {}
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
        elif phase == "robot":
            self._log.append("Use 'New robot calibration' to also clear bed and corners.")
        self._bundle.write_manifest()
        self._log.append(f"[{phase}] Cleared all samples in this phase.")
        self._refresh()

    def _on_clear_floor(self) -> None:
        ans = QMessageBox.question(
            self,
            "New robot calibration",
            "This clears ALL robot, bed, and corner samples and deletes world results. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        self._bundle.robot.clear()
        self._bundle.invalidate_from_robot()
        self._log.append("New robot calibration: cleared robot, bed, corners, and world yaml.")
        self._refresh()

    def _on_run(self) -> None:
        refresh_intrinsics_cache(self._deps.intrinsics)
        self._deps.world_cfg = load_world()
        self._deps.robot_cfg = load_robot()
        stage1 = load_stage1_extrinsics()
        if stage1 is None:
            QMessageBox.warning(self, "Stage 1 missing", f"{extrinsics_rel_path()} not found.")
            return

        phase = self._current_phase()
        wc = self._deps.world_cfg
        inherited = self._bundle.inherit_prereq_alignment_from_last(phase)
        if inherited:
            self._log.append("Inherited alignment (no re-run needed): " + "; ".join(inherited))
        state = self._bundle.load_aligned_state()

        if phase == "robot":
            if len(self._bundle.robot.samples) < wc.min_robot_samples:
                QMessageBox.warning(
                    self, "Not enough robot samples", f"Need >= {wc.min_robot_samples} robot captures."
                )
                return
        elif phase == "bed":
            if not state.floor_aligned:
                QMessageBox.warning(self, "Robot not aligned", "Run robot hand-eye first.")
                return
            if len(self._bundle.bed.samples) < wc.min_bed_samples:
                QMessageBox.warning(self, "Not enough bed samples", f"Need >= {wc.min_bed_samples} bed captures.")
                return
        else:
            if not state.floor_aligned:
                QMessageBox.warning(self, "Robot not aligned", "Run robot hand-eye first.")
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
        if report.phase == "robot":
            d = report.robot_diagnostics or {}
            lines.append(
                f"  BA RMSE: {d.get('ba_rmse_px', float('nan')):.3f} px "
                f"({d.get('ba_rmse_at_2m_mm', float('nan')):.1f} mm at 2 m)"
            )
            lines.append(
                f"  +X σ ≈ {d.get('x_axis_sigma_deg', float('nan')):.3f} deg "
                f"({d.get('x_axis_sigma_at_2m_mm', float('nan')):.1f} mm at 2 m)  [approx]"
            )
            lines.append(
                f"  rail baseline {d.get('rail_baseline_m', float('nan')):.3f} m, "
                f"{d.get('n_rail_stations', 0)} stations"
            )
            lines.append(
                f"  rail-axis residual {d.get('rail_axis_residual_deg', float('nan')):.3f} deg, "
                f"base_link tilt {d.get('baselink_z_tilt_from_world_z_deg', float('nan')):.3f} deg"
            )
            lines.append(f"  leave-one-group angle {d.get('leave_one_group_angle_deg', float('nan')):.3f} deg")
            lines.append("  Exported: calibration_results/robot_world.yaml")
        else:
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
                warn = ""
                if abs(m.bed_rotation_deg) > float(self._deps.world_cfg.bed_skew_warn_deg):
                    warn = f"  [warn: |skew| > {self._deps.world_cfg.bed_skew_warn_deg:.1f} deg]"
                lines.append(
                    f"  Bed size (m): {m.bed_size_m[0]:.3f} x {m.bed_size_m[1]:.3f} "
                    f"(rotated {m.bed_rotation_deg:.1f} deg from world X){warn}"
                )
            lines.append(f"  Origin (floor): {m.bed_center_on_floor}")
            lines.append(f"  Bed center (world): {m.bed_center_world}")
            lines.append("  Exported: calibration_results/genesis_bundle.yaml")
        if report.phase in ("robot", "bed"):
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
