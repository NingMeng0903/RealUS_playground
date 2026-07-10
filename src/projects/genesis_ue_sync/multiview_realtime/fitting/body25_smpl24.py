"""OpenPose Body25 index -> SMPL-24 joint index (smplx ``out.joints`` layout)."""

from __future__ import annotations

# (body25_index, smpl24_index)
BODY25_SMPL24_PAIRS: tuple[tuple[int, int], ...] = (
    (8, 0),   # MidHip -> pelvis
    (9, 2),   # RHip
    (12, 1),  # LHip
    (10, 5),  # RKnee
    (13, 4),  # LKnee
    (11, 8),  # RAnkle
    (14, 7),  # LAnkle
    (1, 12),  # Neck
    (0, 15),  # Nose -> head
    (2, 17),  # RShoulder
    (5, 16),  # LShoulder
    (3, 19),  # RElbow
    (6, 18),  # LElbow
    (4, 21),  # RWrist
    (7, 20),  # LWrist
)

BODY25_SMPL24_BONE_PAIRS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((8, 9), (0, 2)),    # MidHip -> RHip
    ((9, 10), (2, 5)),   # RHip -> RKnee
    ((10, 11), (5, 8)),  # RKnee -> RAnkle
    ((8, 12), (0, 1)),   # MidHip -> LHip
    ((12, 13), (1, 4)),  # LHip -> LKnee
    ((13, 14), (4, 7)),  # LKnee -> LAnkle
    ((1, 2), (12, 17)),  # Neck -> RShoulder
    ((2, 3), (17, 19)),  # RShoulder -> RElbow
    ((3, 4), (19, 21)),  # RElbow -> RWrist
    ((1, 5), (12, 16)),  # Neck -> LShoulder
    ((5, 6), (16, 18)),  # LShoulder -> LElbow
    ((6, 7), (18, 20)),  # LElbow -> LWrist
    ((8, 1), (0, 12)),   # MidHip -> Neck
    ((1, 0), (12, 15)),  # Neck -> Head
)

BODY25_MID_HIP = 8
