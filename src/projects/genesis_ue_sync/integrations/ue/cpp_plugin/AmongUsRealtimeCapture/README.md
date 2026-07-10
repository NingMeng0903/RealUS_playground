# AmongUsRealtimeCapture (UE plugin skeleton)

Copy this folder into `<YourUnrealProject>/Plugins/AmongUsRealtimeCapture`, regenerate project files, then build the Editor target.

## Wire-up checklist

1. Add `AmongUsRealtimeCapture` to `Plugins` and enable it inside the editor.
2. Add `UAmongUsTcpCaptureComponent` to a ticking actor; assign `SceneCaptures` + `CameraNames`; optional `SetExternalSimClock` / `SetSessionId` from Blueprint/Python when syncing to Genesis timestamps.
   Scene init (`spawn_amongus_capture_rig`) also sets `CameraFlipU` / `CameraFlipV` per camera from scene spec metadata or near-nadir geometry auto-detection.
3. Point the TCP client at `amongus_ue_tcp_camera_mux.py --listen-port ...`.

This repository ships the mux + metadata contracts so gameplay code stays inside UE while ZMQ fan-out stays in Python.

## TCP framing

```
uint32 meta_len_be
meta_json_utf8 (CameraFrameMetadataV1 fields)
uint32 img_len_be
jpeg_bytes
```
