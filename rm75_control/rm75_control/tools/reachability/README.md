# rm75_control.tools.reachability

Offline capability map (Zacharias 2013) + online rail-base placement inversion
(Vahrenkamp 2013) for the RM75-6F on Y-axis rail.

Layers:

- `data_model/` — VoxelGrid, ToolAxisGrid, CapabilityMap, frame helpers.
- `kinematics/` — 7-DOF (rail locked) Pinocchio model + batch FK + DLS IK.
- `build/`      — Monte-Carlo + IK-refined offline builder + CLI.
- `inversion/`  — waypoint reach sets + full-scan / prefix optimizers + CLI.
- `viz/`        — PyVista scenes matching Zacharias & Vahrenkamp paper figures.

See [MD/plan.md](../../../MD/plan.md) for the full design.
