from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.operator_bake_v8 import (
    sanitize_v8_runtime_metadata,
)


def test_runtime_metadata_removes_every_old_leg_and_patella_path() -> None:
    cleaned = sanitize_v8_runtime_metadata(
        {
            "source_full_local_fk_v2": True,
            "source_leg_hinge_solve_v1": {"left": 1},
            "source_knee_hinge_splines_v7": {"left": 2},
            "nested": {
                "source_tibia_glide_splines_v7": {"left": 3},
                "source_patella_v71_response_v8": {"left": 4},
            },
            "unrelated": np.int64(5),
        }
    )
    encoded = repr(cleaned)
    for marker in (
        "source_leg_hinge_solve_v1",
        "source_knee_hinge_splines_v7",
        "source_tibia_glide_splines_v7",
        "source_patella_v71_response_v8",
    ):
        # The audit list records what was removed, but no executable key remains.
        assert marker not in cleaned
        assert marker not in cleaned["nested"]
    assert cleaned["unrelated"] == 5
    assert cleaned["source_full_local_fk_v2"] is True
    assert cleaned["disable_soft_follow"] is True
