"""multicam_calib — extensible multi-camera extrinsics calibration.

Two calibration stages:
- Stage 1: camera-to-camera relative extrinsics from a moving AprilTag board.
- Stage 2: cameras aligned to a world frame defined by the board on the floor.

An optional Stage 0 refines per-camera pinhole intrinsics from a chessboard.
"""

__all__: list[str] = []
