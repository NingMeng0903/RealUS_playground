"""Initial Franka joint vectors aligned with PEIRASTIC scripts under ``ref_code_library/PEIRASTIC_control``."""

import numpy as np

PEIRASTIC_GOLDEN_RESET_JOINTS = np.array(
    [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ],
    dtype=np.float32,
)

PEIRASTIC_SPACENAV_INIT_JOINTS = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 1.8, 0.0], dtype=np.float32)
