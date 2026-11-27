#!/usr/bin/env bash
# Each role runs in its own terminal. Extra arguments are forwarded verbatim.
set -e

if [ "$#" -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "help" ]; then
  cat <<'USAGE'
Usage: bash scripts/realus_fullflow.sh ROLE [extra arguments]
  setup       regenerate the calibrated scene and physical camera calibration
  plugin      compile and install the UE capture plugin (keeps an old-plugin backup)
  1           UE session watcher
  2           UE TCP -> ZMQ camera mux (17366)
  3           four-camera preview
  4           current eight-DOF scene-init publisher
  5           textured T-pose prepare, then exit
  6           canonical -> UE preview bridge
  play        start PIE after T-pose prepare
  stop-play   stop PIE
  resume-preview restore continuous camera capture after strict recording
  7           Genesis AMASS loop; add --robot-state-source shm for a running controller
  8           UE RGB burst -> DWPose/EasyMocap SMPL-X
  9           anatomy retarget and publish
  gt-log      finite headless Genesis motion log for benchmark replay
  benchmark-log prepend 0.5 seconds of static RGB for shape initialization
  record      strict four-camera benchmark recorder
  sequence    continuous or recorded-RGB SMPL-X fitting
  video       export aligned MP4 files from a recording
  mv4dhmr     official calibrated four-view MV4DHMR benchmark
  hybrik      initialized analytical HybrIK-X on DWPose 3D skeletons
  evaluate    compare reconstructed SMPL-X55 against actual UE bones
  overlays    render reconstructed SMPL-X meshes over the recorded RGB
  annotations export portable actual camera calibration and joint GT
  camera      physical RealSense publisher (17356)
  capture-real physical RGB burst -> SMPL-X
  controller  current peirastic/native C++ controller; connects real hardware

Override SESSION_DIR / REALUS_AMASS_NPZ before running. Default robot state
source is simulation; only the explicit controller role opens robot connections.
USAGE
  exit 0
fi

role="$1"
shift
realus_launch_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${realus_launch_root}"
source env.sh

case "${role}" in
  setup)
    exec "$PY" scripts/build_realus_scene_from_bundle.py --bundle "$CAMERA_CALIB_BUNDLE" \
      --out-scene "$REALUS_SCENE_SPEC" "$@" ;;
  plugin)
    exec "$PY" -m projects.genesis_ue_sync.cli.render.unreal.install_amongus_capture_plugin "$@" ;;
  1|ue)
    exec "$PY" -m projects.genesis_ue_sync.cli.render.unreal.run_ue_scene_session \
      --session-dir "$SESSION_DIR" --watcher-only "$@" ;;
  2|mux)
    exec "$PY" -m projects.genesis_ue_sync.cli.render.unreal.amongus_ue_tcp_camera_mux \
      --listen-host 127.0.0.1 --listen-port 17355 --pub-bind "$REALUS_UE_CAMERA_CONNECT" "$@" ;;
  3|preview)
    exec "$PY" -m projects.genesis_ue_sync.cli.render.unreal.watch_ue_camera_frames \
      --connect "$REALUS_UE_CAMERA_CONNECT" --camera-names cam1 cam2 cam3 cam4 "$@" ;;
  4|scene)
    exec "$PY" -m projects.genesis_ue_sync.cli.render.unreal.run_scene_init_publisher \
      --scene-spec "$REALUS_SCENE_SPEC" --robot-model rm75_6f_8dof \
      --bind tcp://127.0.0.1:5588 --repeat-s 2 "$@" ;;
  5|prepare)
    exec "$PY" -m projects.genesis_ue_sync.cli.render.unreal.run_scene_init_zmq_ue_bridge \
      --connect tcp://127.0.0.1:5588 --session-dir "$SESSION_DIR" \
      --scene-apply-mode prepare --exit-after-first "$@" ;;
  6|bridge)
    exec "$PY" -m projects.genesis_ue_sync.cli.render.unreal.run_canonical_zmq_ue_bridge \
      --canonical-connect tcp://127.0.0.1:5599 --session-dir "$SESSION_DIR" "$@" ;;
  play)
    exec "$PY" perception/apps/ue_session_command.py --session-dir "$SESSION_DIR" --command start_pie "$@" ;;
  stop-play)
    exec "$PY" perception/apps/ue_session_command.py --session-dir "$SESSION_DIR" --command stop_pie "$@" ;;
  resume-preview)
    exec "$PY" perception/apps/ue_session_command.py --session-dir "$SESSION_DIR" --command resume_camera_preview "$@" ;;
  7|runtime)
    realus_viewer_args=()
    if [ "${REALUS_SHOW_VIEWER:-1}" = "1" ]; then realus_viewer_args+=(--show-viewer); fi
    exec "$PY" perception/apps/run_genesis_ue_runtime.py --scene-spec "$REALUS_SCENE_SPEC" \
      --amass-npz "$REALUS_AMASS_NPZ" --robot-model rm75_6f_8dof \
      --backend "${REALUS_GENESIS_BACKEND:-cuda}" "${realus_viewer_args[@]}" \
      --track-subscribe tcp://127.0.0.1:5598 --anatomy-asset-subscribe tcp://127.0.0.1:5601 \
      --loop --frame-step 4 --playback-fps 30 "$@" ;;
  8|capture-ue)
    exec "$PY" perception/apps/run_smplx_capture.py \
      --config configs/tracking/realus_ue_dwpose_easymocap.yaml \
      --connect "$REALUS_UE_CAMERA_CONNECT" --write-debug-images --publish-genesis "$@" ;;
  9|anatomy)
    exec "$PY" perception/apps/run_anatomy_retarget.py --publish-genesis --export-vessels "$@" ;;
  gt-log)
    realus_gt_path="${REALUS_GT_LOG:-${SESSION_DIR}/benchmark_gt.jsonl}"
    if [ -s "$realus_gt_path" ] && [[ " $* " != *" --help "* ]]; then
      echo "GT log already exists: $realus_gt_path. Set REALUS_GT_LOG to a new path." >&2
      exit 2
    fi
    export AMONGUS_GENESIS_CANONICAL_STATE_JSONL="$realus_gt_path"
    export AMONGUS_GENESIS_CANONICAL_ZMQ_BIND=0
    exec "$PY" perception/apps/run_genesis_ue_runtime.py --scene-spec "$REALUS_SCENE_SPEC" \
      --amass-npz "$REALUS_AMASS_NPZ" --robot-model rm75_6f_8dof \
      --backend "${REALUS_GENESIS_BACKEND:-cuda}" --frame-step 4 --playback-fps 30 \
      --frame-limit 120 "$@" ;;
  record)
    exec "$PY" perception/apps/run_ue_benchmark_recorder.py "$@" ;;
  benchmark-log)
    exec "$PY" perception/apps/build_ue_motion_benchmark.py "$@" ;;
  sequence)
    exec "$PY" perception/apps/run_smplx_sequence.py \
      --config configs/tracking/realus_ue_dwpose_easymocap.yaml "$@" ;;
  video)
    exec "$PY" perception/apps/export_ue_benchmark_videos.py "$@" ;;
  mv4dhmr)
    exec "$PY" -m perception.benchmark.mv4dhmr "$@" ;;
  hybrik)
    exec "$PY" perception/apps/run_hybrik_benchmark.py "$@" ;;
  evaluate)
    exec "$PY" perception/apps/evaluate_mocap_benchmark.py "$@" ;;
  overlays)
    exec "$PY" perception/apps/render_mocap_benchmark_overlays.py "$@" ;;
  annotations)
    exec "$PY" perception/apps/export_ue_benchmark_annotations.py "$@" ;;
  camera)
    exec "$REALUS_CAMERA_PY" perception/apps/run_realsense_camera_publisher.py \
      --bundle "$CAMERA_CALIB_BUNDLE" --cameras-yaml "$REALUS_CAMERAS_YAML" \
      --pub-bind tcp://127.0.0.1:17356 --width 1920 --height 1080 --fps 30 "$@" ;;
  capture-real)
    exec "$PY" perception/apps/run_smplx_capture.py \
      --config configs/tracking/realus_dwpose_easymocap.yaml \
      --connect tcp://127.0.0.1:17356 --write-debug-images --publish-genesis "$@" ;;
  controller)
    source rm75_control/env.sh
    exec python -m peirastic.apps.run_controller "$@" ;;
  *) echo "Unknown role: ${role}. Run with --help." >&2; exit 2 ;;
esac
