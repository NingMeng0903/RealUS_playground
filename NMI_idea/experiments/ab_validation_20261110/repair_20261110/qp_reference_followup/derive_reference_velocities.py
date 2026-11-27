"""Re-evaluate deterministic references at saved controller ticks; no QP rerun."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from run_reference_protocol import HERE, INPUT, SOURCE_REFERENCE, Analytic6DReference


def main():
    frozen = np.load(INPUT, allow_pickle=False)
    T = frozen['branch4_path0_T']
    points = T[:, :3, 3]
    rpy = Rotation.from_matrix(T[:, :3, :3]).as_euler('xyz')
    for case, klass, speed in [('original_2cm', SOURCE_REFERENCE, .02),
                               ('6d_2cm', Analytic6DReference, .02),
                               ('6d_retime_0p5cm', Analytic6DReference, .005)]:
        source = np.load(HERE / f'{case}_trace.npz', allow_pickle=False)
        t = source['t_s']
        ref = klass(points, rpy=rpy, speed_m_s=speed, soft_start=True, ramp_s=.4)
        samples = [ref.sample(float(tick)) for tick in t]
        Path(HERE / f'{case}_reference.npz').parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(HERE / f'{case}_reference.npz', t_s=t,
            pose_d=np.asarray([x.pose_d for x in samples]),
            vel_ff=np.asarray([x.vel_ff for x in samples]))


if __name__ == '__main__':
    main()
