"""Parameterized link7 → TCP SE(3) probe transform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ProbeTransform:
    """Rigid transform from parent (link7) to TCP."""

    name: str
    translation_m: np.ndarray  # (3,)
    quaternion_xyzw: np.ndarray  # (4,)
    parent_frame: str = "link_7"
    child_frame: str = "tcp"

    def rotation_matrix(self) -> np.ndarray:
        return Rotation.from_quat(self.quaternion_xyzw).as_matrix()

    def matrix(self) -> np.ndarray:
        """4×4 T_parent_tcp (TCP pose expressed in parent)."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation_matrix()
        T[:3, 3] = self.translation_m
        return T

    def pose6_xyz_rpy(self, *, euler_order: str = "xyz") -> np.ndarray:
        """[x,y,z,rx,ry,rz] for RealMan / Pinocchio tcp offset APIs."""
        if (
            euler_order == "xyz"
            and abs(self.quaternion_xyzw[0]) < 1e-9
            and abs(self.quaternion_xyzw[2]) < 1e-9
            and abs(abs(self.quaternion_xyzw[1]) - abs(self.quaternion_xyzw[3])) < 1e-6
        ):
            rpy = np.array([0.0, 0.5 * np.pi, 0.0])
        else:
            rpy = Rotation.from_quat(self.quaternion_xyzw).as_euler(euler_order, degrees=False)
        return np.concatenate([self.translation_m, rpy]).astype(np.float64)

    def urdf_xyz_rpy(self) -> tuple[str, str]:
        """Strings for `<origin xyz=... rpy=.../>` (URDF fixed-axis RPY = xyz)."""
        rpy = self.pose6_xyz_rpy(euler_order="xyz")[3:]
        xyz = " ".join(f"{v:.8f}" for v in self.translation_m)
        rpy_s = " ".join(f"{v:.8f}" for v in rpy)
        return xyz, rpy_s


def default_ultrasound_probe() -> ProbeTransform:
    """Trans_z(0.07)·Rot_y(+π/2)·Trans_z(0.05) composed origin/orientation."""
    Tz1 = np.eye(4)
    Tz1[2, 3] = 0.07
    Ry = np.eye(4)
    Ry[:3, :3] = Rotation.from_euler("y", 0.5 * np.pi).as_matrix()
    Tz2 = np.eye(4)
    Tz2[2, 3] = 0.05
    T = Tz1 @ Ry @ Tz2
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    return ProbeTransform(
        name="ultrasound_probe_default",
        translation_m=T[:3, 3].copy(),
        quaternion_xyzw=quat.astype(np.float64),
    )


def load_probe_yaml(path: str | Path) -> ProbeTransform:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    t = np.asarray(raw["translation_m"], dtype=np.float64).reshape(3)
    q = np.asarray(raw["quaternion_xyzw"], dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    return ProbeTransform(
        name=str(raw.get("name", Path(path).stem)),
        translation_m=t,
        quaternion_xyzw=q,
        parent_frame=str(raw.get("parent_frame", "link_7")),
        child_frame=str(raw.get("child_frame", "tcp")),
    )


def patch_urdf_tcp(
    src_urdf: str | Path,
    dst_urdf: str | Path,
    probe: ProbeTransform,
    *,
    joint_name: str = "link_7_to_tcp",
    add_probe_visual: bool = False,
    probe_length_m: float = 0.05,
    probe_radius_m: float = 0.012,
) -> Path:
    """Rewrite the fixed joint origin for ``joint_name`` and write ``dst_urdf``.

    If ``add_probe_visual``, replace an empty ``<link name="tcp" />`` with a short
    cylinder along TCP +Z so the horizontal mount is visible in PyVista/Genesis.
    """
    import re

    text = Path(src_urdf).read_text(encoding="utf-8")
    xyz, rpy = probe.urdf_xyz_rpy()
    pattern = re.compile(
        rf'(<joint\s+name="{re.escape(joint_name)}"[^>]*>\s*)'
        r"<origin\s+[^/]*/>",
        re.DOTALL,
    )
    repl = rf'\1<origin xyz="{xyz}" rpy="{rpy}" />'
    new_text, n = pattern.subn(repl, text, count=1)
    if n != 1:
        raise ValueError(f"could not patch joint {joint_name!r} in {src_urdf}")

    if add_probe_visual:
        half = 0.5 * float(probe_length_m)
        visual = (
            '<link name="tcp">\n'
            "    <visual>\n"
            f'      <origin xyz="0 0 {half:.6f}" rpy="0 0 0" />\n'
            "      <geometry>\n"
            f'        <cylinder length="{float(probe_length_m):.6f}" '
            f'radius="{float(probe_radius_m):.6f}" />\n'
            "      </geometry>\n"
            '      <material name="probe_cyan"><color rgba="0.2 0.75 0.85 1"/></material>\n'
            "    </visual>\n"
            "  </link>"
        )
        new_text2, n2 = re.subn(
            r'<link\s+name="tcp"\s*/>',
            visual,
            new_text,
            count=1,
        )
        if n2 != 1:
            # already has a tcp link body — leave geometry as-is
            pass
        else:
            new_text = new_text2

    dst = Path(dst_urdf)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new_text, encoding="utf-8")
    return dst


def ensure_probe_visual_urdf(
    *,
    playground_root: str | Path,
    probe_yaml: str | Path | None = None,
    out_name: str = "RM75-probe.genesis.urdf",
) -> Path:
    """Genesis-mesh URDF with horizontal ultrasound TCP + cylinder glyph.

    Mesh ``filename`` entries are rewritten to absolute paths so the file can
    live under ``ird_playground/data/maps/`` without breaking PyVista.
    """
    import re

    root = Path(playground_root).resolve()
    rm75 = root.parent / "rm75_control"
    src = rm75 / "rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.genesis.urdf"
    if not src.is_file():
        raise FileNotFoundError(src)
    yaml_path = Path(probe_yaml) if probe_yaml else root / "configs/probe_default.yaml"
    if not yaml_path.is_absolute():
        yaml_path = root / yaml_path
    probe = load_probe_yaml(yaml_path)
    out = root / "data/maps" / out_name
    patch_urdf_tcp(src, out, probe, add_probe_visual=True)

    mesh_root = src.parent
    text = out.read_text(encoding="utf-8")

    def _abs_mesh(m: re.Match[str]) -> str:
        rel = m.group(1)
        if Path(rel).is_absolute():
            return m.group(0)
        abs_p = (mesh_root / rel).resolve()
        return f'filename="{abs_p}"'

    text = re.sub(r'filename="([^"]+)"', _abs_mesh, text)
    out.write_text(text, encoding="utf-8")
    return out
