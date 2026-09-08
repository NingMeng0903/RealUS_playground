# Phantom planning regression input

`phantom_top_20260908_051548.csv` contains the 2256 detected top-surface points
from `peirastic/logs/phantom/20260908_051548_662477/detected_top.npy`, in metres
in the controller `rail_base` frame. The first commented JSON line preserves
the surface reference normal, q8 and measured TCP pose from that run's
`failure.json` and `capture.npz`. Coordinates retain 12 significant digits.

This measured case formerly failed because a Delaunay edge of 11.865 mm
exceeded a global nearest-spacing limit of 11.682 mm. It also exercises the
long-edge scan alignment and measured-tool orientation anchor. Tests use no
camera or controller connection.

`phantom_standoff_handoff_20260908.json` records the next capture's TCP
standoff, initial joint state, active tool offset, retained posture attractor
and native fault command/measurement pair. It reproduces the J4 endpoint
inside the online position margin and exercises the corrected IK selection.
The native regression uses private UUID shared memory and ideal feedback.
