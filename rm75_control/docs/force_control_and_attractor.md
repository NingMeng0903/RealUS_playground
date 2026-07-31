# RM75 force control and runtime attractors

The production tool-Z force loop uses hysteretic contact release, a 12 mm/s
setpoint-independent seek velocity, implicit-Euler admittance integration, a
predictive force-space velocity damper, light force-trend damping, and Dimeas
variable inertia. The proactive `v_ref` integrator is normalized by desired
force and remains bidirectional. It is not the full force/motion observer
controller described by Li et al. (2022).

`JointCenteringTask.set_q_target()` accepts an eight-element target while the
controller is running. In the production 8-DOF configuration its RAIL weight
is zero, so this attractor changes only the seven arm-joint preferences. RAIL
motion remains controlled by the Cartesian QP allocation and the independent
`RailExtensionTask.set_rail_pose_target()` target.

The YAML comfortable posture is retained as the default target. If the
controller has entered the sigma-escape zone and subsequently recovers, a
stronger centering pull (gain 3.0, arm-joint cap 35% of the URDF velocity
limit) stays enabled until the seven arm joints are within 0.12 normalized
range of the active target. It is a persistent recovery objective, not a
short timer, and it never supplies a RAIL velocity.

There is currently no unified, thread-safe BirdPlayground streaming API that
atomically updates an arm posture target and a RAIL target. A future trajectory
streamer must route those two targets through their separate owners and define
the update synchronization explicitly.
