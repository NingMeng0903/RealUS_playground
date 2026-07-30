import pytest

from projects.genesis_ue_sync.anatomy_retarget.reference_fit_v8 import (
    _is_axial_product_bone,
)


@pytest.mark.parametrize(
    "mesh_name",
    [
        "Upper_Skull",
        "Mandible",
        "C1_Atlas",
        "C2_Axis",
        "C3",
        "C7",
        "T1",
        "T12",
        "L1",
        "L5",
        "Disc_C2_C3",
        "Disc_T12_L1",
        "Rib_1L",
        "Rib_12R",
        "Clavicle_L",
        "Scapula_R",
        "Sternum",
        "Central_Incisor",
        "Canine_4",
        "Premolar_2nd_3",
        "Molar_3rd_4",
    ],
)
def test_head_neck_thorax_compound_uses_one_continuous_authority(
    mesh_name: str,
) -> None:
    assert _is_axial_product_bone(mesh_name)


@pytest.mark.parametrize(
    "mesh_name",
    ["Femur_L", "Tibia_R", "Humerus_L", "Radius_R", "Ulna_L"],
)
def test_long_bones_stay_on_nonshrunk_authority(mesh_name: str) -> None:
    assert not _is_axial_product_bone(mesh_name)
