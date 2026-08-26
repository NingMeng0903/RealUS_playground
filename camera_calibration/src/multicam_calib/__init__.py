"""multicam_calib — extensible multi-camera extrinsics calibration.

Two calibration stages:
- Stage 1: camera-to-camera relative extrinsics from a moving AprilTag board.
- Stage 2: cameras aligned to a world frame from robot-rail geometry (hand-eye),
  then bed height and bed-corner envelope.

An optional Stage 0 refines per-camera pinhole intrinsics from a chessboard.
Stage 3 checks Orbbec RGB-D (D2C / point cloud). Stage 4 is Orbbec RGB
chessboard K/d. Stage 5 is eye-in-hand ``T_link7_cam``.
"""

__all__: list[str] = []
