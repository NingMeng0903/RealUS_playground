from __future__ import annotations

from pathlib import Path
from typing import Iterable

from projects.genesis_ue_sync.sim_platform.embodiments.loaders.urdf_loader import (
    URDFToolFrames,
    build_embodiment_from_urdf,
)


def _repo_root() -> Path:
    from common.project import project_paths
    return project_paths(__file__).root


def default_panda_urdf_path() -> Path:
    return _repo_root() / "assets" / "robots" / "panda_urdf" / "panda_arm.urdf"


def build_panda_ultrasound_preset(
    *,
    urdf_path: Path | None = None,
    camera_names: Iterable[str] = ("camera_left", "camera_right"),
    image_resolution: tuple[int, int] = (1280, 720),
    camera_baseline_m: float = 0.12,
) -> object:
    return build_embodiment_from_urdf(
        name="panda_ultrasound",
        urdf_path=Path(urdf_path) if urdf_path is not None else default_panda_urdf_path(),
        tool_frames=URDFToolFrames(
            base_frame="panda_link0",
            eef_link="panda_link7",
            tool_frame="panda_probe",
            tcp_frame="panda_probe",
            ultrasound_image_frame="ultrasound_image_frame/probe",
        ),
        camera_names=camera_names,
        image_resolution=image_resolution,
        camera_baseline_m=camera_baseline_m,
        tool_name="ultrasound_probe",
        max_contact_force_n=15.0,
        metadata={
            "preset": "panda_ultrasound",
            "urdf_tool_link": "panda_probe",
            "urdf_tcp_link": "panda_probe",
        },
    )
