"""Show how much femoral axial twist the leg hinge solve injects vs flexion.

The solve puts the femur long axis on the SMPL-X hip->knee ray and then twists the
femur by ``phi`` about that ray until the authored knee hinge can reach the SMPL-X
shank.  Near full extension the shank is almost parallel to the femur, so the
radial component ``phi`` acts on vanishes and every twist satisfies the cone
constraint about equally well: ``phi`` there is an ill-conditioned quantity rather
than a measurement, and the runtime fades it in over a fixed flexion band.

This probe drives the real baked leg entry with a shank flexed about an axis
tilted away from the authored hinge, which is the situation the SMPL-X drive
actually presents, and prints the raw solved twist next to the twist the fade
lets through.  It answers whether the fade band is wide enough for how fast raw
``phi`` grows as the leg straightens.
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--side", default="left", choices=("left", "right"))
    parser.add_argument(
        "--tilt-deg",
        type=float,
        default=53.3,
        help="angle between the drive flexion axis and the authored hinge",
    )
    args = parser.parse_args()

    from scipy.spatial.transform import Rotation

    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
        solve_leg_hinge_v1,
    )
    from projects.genesis_ue_sync.anatomy_retarget.v7_artifacts import (
        load_subject_asset,
    )

    subject = load_subject_asset(args.subject)
    asset = subject.rigged_asset
    bind = np.asarray(asset.source_rest_global, dtype=np.float64)

    raw = asset.metadata["source_leg_hinge_solve_v1"]
    entry = raw.item() if isinstance(raw, np.ndarray) else raw
    if isinstance(entry, dict) and args.side in entry:
        entry = entry[args.side]

    femur = int(entry["femur_bone"])
    knee = int(entry["knee_bone"])
    ankle_bone = int(entry["ankle_bone"])
    hinge_local = np.asarray(entry["hinge_axis_femur_local"], dtype=np.float64)
    blend_lo = float(entry.get("blend_lo_deg", 5.0))
    blend_hi = float(entry.get("blend_hi_deg", 15.0))

    bind_hip = bind[femur, :3, 3]
    bind_knee = bind[knee, :3, 3]
    bind_ankle = bind[ankle_bone, :3, 3]
    r_bind = bind[femur, :3, :3]

    thigh = bind_knee - bind_hip
    d0 = thigh / float(np.linalg.norm(thigh))
    shank = bind_ankle - bind_knee
    shank_len = float(np.linalg.norm(shank))

    hinge_world = r_bind @ hinge_local
    hinge_world = hinge_world / float(np.linalg.norm(hinge_world))

    # Drive axis: the authored hinge tilted by --tilt-deg about the femur long
    # axis, i.e. a drive that wants the knee to flex in a different plane.
    tilt = Rotation.from_rotvec(d0 * np.radians(args.tilt_deg)).as_matrix()
    drive_axis = tilt @ hinge_world
    drive_axis = drive_axis / float(np.linalg.norm(drive_axis))

    print(
        f"side={args.side} thigh={float(np.linalg.norm(thigh)):.4f} m "
        f"shank={shank_len:.4f} m fade_band=[{blend_lo:.1f},{blend_hi:.1f}] deg "
        f"drive_axis_vs_hinge={args.tilt_deg:.1f} deg"
    )
    print(
        f"{'drive_deg':>9} {'flex_deg':>9} {'raw_phi_deg':>12} "
        f"{'applied_phi_deg':>16} {'theta_deg':>10} {'ankle_err_mm':>13}"
    )
    for drive_deg in (0.0, 0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 40.0, 90.0):
        rot = Rotation.from_rotvec(drive_axis * np.radians(drive_deg)).as_matrix()
        posed_ankle = bind_knee + rot @ shank
        r_femur, theta, _theta_raw, hinge_out = solve_leg_hinge_v1(
            hip=bind_hip,
            knee=bind_knee,
            ankle=posed_ankle,
            bind_hip=bind_hip,
            bind_knee=bind_knee,
            bind_ankle=bind_ankle,
            bind_femur_rotation=r_bind,
            hinge_axis_femur_local=hinge_local,
            driver_femur_rotation=r_bind,
            blend_lo_deg=blend_lo,
            blend_hi_deg=blend_hi,
        )
        applied_phi = float(
            np.dot(Rotation.from_matrix(r_femur @ r_bind.T).as_rotvec(), d0)
        )
        # Undo the fade to recover the twist the closed form actually solved for.
        flex = float(
            np.degrees(
                np.arccos(
                    float(
                        np.clip(
                            np.dot(
                                d0,
                                (posed_ankle - (bind_hip + np.linalg.norm(thigh) * d0))
                                / float(
                                    np.linalg.norm(
                                        posed_ankle
                                        - (bind_hip + np.linalg.norm(thigh) * d0)
                                    )
                                ),
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        )
        fade = np.clip((flex - blend_lo) / (blend_hi - blend_lo), 0.0, 1.0)
        fade = fade * fade * (3.0 - 2.0 * fade)
        raw_phi = applied_phi / fade if fade > 1.0e-9 else float("nan")
        carried = r_femur @ r_bind.T @ (shank / shank_len)
        swung = Rotation.from_rotvec(hinge_out * theta).as_matrix() @ carried
        ankle_err_mm = 1000.0 * float(
            np.linalg.norm(bind_knee + shank_len * swung - posed_ankle)
        )
        print(
            f"{drive_deg:9.1f} {flex:9.3f} {np.degrees(raw_phi):12.3f} "
            f"{np.degrees(applied_phi):16.3f} {np.degrees(theta):10.3f} "
            f"{ankle_err_mm:13.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
