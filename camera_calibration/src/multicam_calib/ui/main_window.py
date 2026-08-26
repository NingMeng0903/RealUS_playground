"""Main window: Stage 0/1/2 share the 4-camera stream; Stage 3 is Orbbec RGB-D."""
from __future__ import annotations

from dataclasses import replace

from PyQt5.QtWidgets import QLabel, QMainWindow, QMessageBox, QTabWidget, QVBoxLayout, QWidget

from multicam_calib.board.apriltag_board import BoardGeometry, build_board_geometry
from multicam_calib.board.detector import AprilTagDetector
from multicam_calib.calib.intrinsics import load_intrinsics_for_aliases
from multicam_calib.devices.base import CameraDevice
from multicam_calib.io.config import (
    AppConfig,
    DetectorConfig,
    load_board_ee,
    load_orbbec,
    load_robot,
    load_world,
)
from multicam_calib.recording.session import RecordingSession
from multicam_calib.recording.stage2_session import Stage2SessionBundle
from multicam_calib.recording.sync import CameraStreamThread
from multicam_calib.ui.live_view import LiveViewGrid
from multicam_calib.ui.stage0_intrinsics import Stage0Panel
from multicam_calib.ui.stage1_relative import Stage1Deps, Stage1Panel
from multicam_calib.ui.stage2_world import Stage2Deps, Stage2Panel
from multicam_calib.devices.orbbec import OrbbecRGBDSession
from multicam_calib.ui.stage3_orbbec import Stage3OrbbecPanel
from multicam_calib.ui.stage4_orbbec_intrinsics import Stage4OrbbecIntrinsicsPanel
from multicam_calib.ui.stage5_orbbec_handeye import Stage5OrbbecHandeyePanel
from multicam_calib.ui.param_refresh import refresh_intrinsics_cache


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        aliases: list[str],
        devices: dict[str, CameraDevice],
        board_geom: BoardGeometry,
        app_cfg: AppConfig,
        zmq_hub=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Multi-camera calibration")
        self._aliases = aliases
        self._devices = devices
        self._board_geom = board_geom
        self._app_cfg = app_cfg
        self._zmq_hub = zmq_hub

        self._streams: dict[str, CameraStreamThread]
        snapshot_fn = None
        if zmq_hub is not None:
            self._streams = zmq_hub.stream_threads()  # type: ignore[assignment]
            for s in self._streams.values():
                s.start()
            snapshot_fn = lambda: zmq_hub.capture_snapshot_best_effort(
                attempts=app_cfg.sync.capture_poll_attempts,
                poll_interval_ms=app_cfg.sync.capture_poll_interval_ms,
                use_device_timestamp=app_cfg.sync.use_device_timestamp,
            )
        else:
            self._streams = {alias: CameraStreamThread(alias=alias, device=dev) for alias, dev in devices.items()}
            for s in self._streams.values():
                s.start()

        self._detector = AprilTagDetector(board_geom.config, app_cfg.detector)
        preview_det_cfg = replace(
            app_cfg.detector,
            quad_decimate=app_cfg.detector.preview_quad_decimate,
            nthreads=1,
        )
        self._preview_detector = AprilTagDetector(board_geom.config, preview_det_cfg)
        board_geom_ee = build_board_geometry(load_board_ee())
        self._detector_ee = AprilTagDetector(board_geom_ee.config, app_cfg.detector)
        self._preview_detector_ee = AprilTagDetector(board_geom_ee.config, preview_det_cfg)

        try:
            if zmq_hub is not None:
                self._intrinsics = load_intrinsics_for_aliases(aliases)
            else:
                self._intrinsics = load_intrinsics_for_aliases(aliases, devices=devices)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Intrinsics unavailable", str(exc))
            raise

        expected_wh = (int(app_cfg.stream.width), int(app_cfg.stream.height))
        mismatched = [
            a
            for a, intr in self._intrinsics.items()
            if tuple(int(v) for v in intr.image_size) != expected_wh
        ]
        if mismatched:
            QMessageBox.warning(
                self,
                "Intrinsics resolution mismatch",
                "Saved Stage 0 K is "
                + ", ".join(
                    f"{a}={self._intrinsics[a].image_size[0]}x{self._intrinsics[a].image_size[1]}"
                    for a in mismatched
                )
                + f", but stream is {expected_wh[0]}x{expected_wh[1]}.\n\n"
                "Run Stage 0 at the new resolution before Stage 1/2.",
            )

        stage1_session = RecordingSession.create_fresh_for_ui(
            stage="stage1_extrinsics",
            aliases=aliases,
            detector=self._detector,
            recording_cfg=app_cfg.recording,
        )
        stage2_bundle = Stage2SessionBundle.create_fresh_for_ui(
            aliases=aliases,
            detector=self._detector,
            detector_ee=self._detector_ee,
            recording_cfg=app_cfg.recording,
        )
        world_cfg = load_world()
        robot_cfg = load_robot()

        badge_streams = (
            zmq_hub.stream_threads(source="capture") if zmq_hub is not None else None
        )
        live_kwargs = dict(
            sync_cfg=app_cfg.sync,
            cell_width=480,
            refresh_hz=app_cfg.ui.preview_refresh_hz,
            min_tags_for_ba=app_cfg.calibration.min_tags_per_view,
            preview_detector=self._preview_detector,
            snapshot_fn=snapshot_fn,
            badge_streams=badge_streams,
            empty_frame_hint=(
                f"waiting for Cam publisher at {app_cfg.preview.zmq_connect} "
                "(start run_realsense_camera_publisher.py first)"
                if zmq_hub is not None
                else None
            ),
        )
        self._live_stage1 = LiveViewGrid(
            aliases, self._streams, self._detector, **live_kwargs,
        )
        self._live_stage2 = LiveViewGrid(
            aliases, self._streams, self._detector, **live_kwargs,
        )

        stage0_streams = (
            zmq_hub.stream_threads(source="capture") if zmq_hub is not None else self._streams
        )
        stage0 = Stage0Panel(
            aliases=aliases,
            streams=stage0_streams,
            on_intrinsics_saved=self._reload_intrinsics,
        )
        stage1 = Stage1Panel(
            live_view=self._live_stage1,
            session=stage1_session,
            deps=Stage1Deps(
                aliases=aliases,
                board_geom=board_geom,
                intrinsics=self._intrinsics,
                app_cfg=app_cfg,
            ),
            on_extrinsics_saved=self._on_stage1_saved,
        )
        stage2 = Stage2Panel(
            live_view=self._live_stage2,
            bundle=stage2_bundle,
            deps=Stage2Deps(
                aliases=aliases,
                board_geom=board_geom,
                board_geom_ee=board_geom_ee,
                intrinsics=self._intrinsics,
                app_cfg=app_cfg,
                world_cfg=world_cfg,
                robot_cfg=robot_cfg,
                detector=self._detector,
                detector_ee=self._detector_ee,
                preview_detector=self._preview_detector,
                preview_detector_ee=self._preview_detector_ee,
            ),
        )

        self._stage0_panel = stage0
        self._stage1_panel = stage1
        self._stage2_panel = stage2
        orbbec_cfg = load_orbbec()
        self._orbbec_session = OrbbecRGBDSession(orbbec_cfg)
        self._stage3_panel = Stage3OrbbecPanel(cfg=orbbec_cfg, session=self._orbbec_session)
        self._stage4_panel = Stage4OrbbecIntrinsicsPanel(
            cfg=orbbec_cfg,
            session=self._orbbec_session,
        )
        self._stage5_panel = Stage5OrbbecHandeyePanel(
            cfg=orbbec_cfg,
            session=self._orbbec_session,
            board_geom=board_geom,
            detector=self._detector,
            robot_cfg=robot_cfg,
        )

        tabs = QTabWidget()
        tabs.addTab(stage0, "Stage 0: Intrinsics")
        tabs.addTab(stage1, "Stage 1: Relative Extrinsics")
        tabs.addTab(stage2, "Stage 2: World Alignment")
        tabs.addTab(self._stage3_panel, "Stage 3: Orbbec RGB-D")
        tabs.addTab(self._stage4_panel, "Stage 4: Orbbec Intrinsics")
        tabs.addTab(self._stage5_panel, "Stage 5: Orbbec eye-in-hand")
        tabs.setCurrentIndex(0 if mismatched else 1)
        tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(tabs.currentIndex())

        central = QWidget()
        lay = QVBoxLayout(central)
        mode = (
            f"ZMQ preview ← {app_cfg.preview.zmq_connect}"
            if zmq_hub is not None
            else "local USB"
        )
        header = QLabel(
            f"Cameras: {', '.join(aliases)}   "
            f"Board: {board_geom.config.family} {board_geom.config.rows}x{board_geom.config.cols}   "
            f"Stream: {app_cfg.stream.width}x{app_cfg.stream.height}@{app_cfg.stream.fps}   "
            f"Mode: {mode}"
        )
        header.setStyleSheet("padding: 4px; background: #333; color: white;")
        lay.addWidget(header)
        lay.addWidget(tabs, 1)
        self.setCentralWidget(central)
        self.resize(1600, 1000)

    def _on_tab_changed(self, index: int) -> None:
        self._stage0_panel.set_active(index == 0)
        self._live_stage1.set_active(index == 1)
        self._live_stage2.set_active(index == 2)
        self._stage3_panel.set_active(index == 3)
        self._stage4_panel.set_active(index == 4)
        self._stage5_panel.set_active(index == 5)

    def _reload_intrinsics(self) -> None:
        refresh_intrinsics_cache(self._intrinsics)
        self._stage1_panel.on_intrinsics_updated()
        self._stage2_panel.on_intrinsics_updated()

    def _on_stage1_saved(self) -> None:
        self._stage2_panel.on_stage1_updated()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._live_stage1.stop()
        self._live_stage2.stop()
        self._stage3_panel.shutdown()
        self._stage4_panel.shutdown()
        self._stage5_panel.shutdown()
        self._orbbec_session.close()
        for s in self._streams.values():
            s.stop()
        if self._zmq_hub is not None:
            self._zmq_hub.stop()
        for dev in self._devices.values():
            try:
                dev.close()
            except Exception:  # noqa: BLE001
                pass
        super().closeEvent(event)
