Offline current-RM75 execution validation, 2026-11-08

Backend: Genesis CPU position PD, current 8DOF URDF, zero gravity.
The only direct joint-position assignment is reset at the start of an independent episode.
Every executing tick calls control_dofs_position + scene.step and reads measured q/qdot.
Pinocchio FK maps measured q to TCP/image pose. Current configured arm-body self collision is checked densely between actual successive states; a violating state stops the trial.
Genesis collision response is disabled. No body/bed/environment contact or force model is present.

Scope: native SMPL-X public distal reference (.62-.76), 0/60 degree knee x 2 shape controls x 2 reference modes; top two dIRD-ranked rail candidates, 2 shared independent entry states.
The virtual placement and illustrative proper image-to-TCP Rx(pi) transform are inherited unchanged from the frozen strict-checkpoint audit. They are not physical registration/calibration.

Results: 26/32 candidates completed actual dynamic execution; 6 were rejected by prior SRS joint-step checks. 0 entry-collision rejection, 0 dynamic stops. All 8 conditions had a dynamically completed candidate.
Actual successful simulated duration range: 11.09-15.18s, mean 12.92s, including entry and waypoint settling. Prior scan-only retiming mean: 5.12s. These are different quantities.
Wall time for the validation process: 350.27s.
All failures remain in summary.json. Time statistics in metrics.json are explicitly conditional on dynamic completion; failed candidates are not counted as completed scans.

Each trial_NNN.npz contains q_cmd, measured q, qdot, TCP/image poses, simulation/wall time, collision distances and phase. trial_NNN.json contains the actual entry, exit and duration.
This is a path-following execution validation, not A/B/C information timing, a probe-contact experiment, an ultrasound-quality acceptance test, or real-robot evidence.
pose_step_smoke.json separately validates the DLS -> PD -> measured FK interface on a public 4mm target (200 ticks, one reset).
10 meaningful adapter contract tests passed with the project Python environment.

Source API: work/leg_program_bench/joint_execution.py
JointServoBackend(JointServoConfig(backend="genesis_pd", dt=.01), base_world=..., T_image_tcp=...)
reset(q) only at episode start; step(q_cmd), state(), fk(), step_pose(T_image_world) advance or read measured state.
execute_path(q_path, q_entry=None) continues from the last measured exit; passing q_entry explicitly begins an independent reset episode.
No hardware SDK is imported or activated.
