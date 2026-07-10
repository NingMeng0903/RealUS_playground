# ROS2 bridge mapping

Future ROS2 nodes mirror ZMQ schemas defined in `stream_schemas.py`:

## CameraFrame (ZMQ multipart part 2)

JSON keys align with `CameraFrameMetadataV1`:

- `schema_version`, `session_id`, `source_id`, `camera_name`
- `frame_index`, `sim_time_ns`, `wall_time_ns`, `source_time_ns`
- `encoding`, `width`, `height`
- `intrinsics`, `extrinsics` (optional nested dicts)

ROS mapping: `sensor_msgs/Image` carries compressed/raw pixels; `sensor_msgs/CameraInfo` fills intrinsics; bridge copies remaining keys into metadata JSON before publishing ZMQ or recording datasets.

## CanonicalScene (ZMQ multipart part 2)

Payload mirrors `canonical_scene_state_to_dict`:

- `schema_version`, `sim_step_index`, `frame_index` (duplicate of step index), `sim_time_ns`, `wall_time_ns`
- `robot_entities`, `human`, `objects`, `contacts`, `extras`

ROS mapping: generate `amongus_msgs/CanonicalScene` or flatten into separate robot-specific topics; retain numeric timestamps for clock alignment.

Real cameras publish ROS topics first; a thin bridge converts `Image/CameraInfo` into `CameraFrameMetadataV1` + compressed payload before emitting ZMQ or feeding downstream Python consumers.

Stamp rule:

- `Header.stamp` derives from `sim_time_ns` when replaying simulation or canonical-driven UE renders.
- Live robots retain sensor `source_time_ns`; bridges populate both `source_time_ns` and aligned `sim_time_ns` after clock sync.
