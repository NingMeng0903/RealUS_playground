"""Core sequence and mesh utilities for human motion data."""

__all__ = [
    "HumanDatasetLayout",
    "HumanMotionSequence",
    "build_joint_sequence",
    "build_shape_neutral_geometry",
    "build_trimesh_sequence",
    "compute_genesis_matched_root_translation",
    "evaluate_smpl_sequence",
    "export_mesh_sequence",
    "load_amass_sequence",
    "load_bedlam_sequence",
    "load_mesh_sequence_from_manifest",
]


def __getattr__(name):
    if name in {
        "HumanMotionSequence",
        "build_joint_sequence",
        "build_shape_neutral_geometry",
        "build_trimesh_sequence",
        "compute_genesis_matched_root_translation",
        "evaluate_smpl_sequence",
        "export_mesh_sequence",
        "load_amass_sequence",
        "load_bedlam_sequence",
        "load_mesh_sequence_from_manifest",
    }:
        from projects.genesis_ue_sync.sim_platform.datasets import human_sequence as _hs

        return getattr(_hs, name)
    if name == "HumanDatasetLayout":
        from projects.genesis_ue_sync.sim_platform.datasets.layout import HumanDatasetLayout

        return HumanDatasetLayout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
