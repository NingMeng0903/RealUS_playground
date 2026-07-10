from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


@dataclass
class MeshGeometry:
    positions: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]


def _parse_float_array(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(item) for item in text.split()]


def _parse_matrix(node: ET.Element | None) -> list[list[float]]:
    if node is None or not node.text:
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    values = [float(item) for item in node.text.split()]
    if len(values) != 16:
        raise ValueError(f"Expected 16 matrix values, got {len(values)}")
    row_major = [
        values[0:4],
        values[4:8],
        values[8:12],
        values[12:16],
    ]
    # COLLADA spec uses column-major storage, but some exporters in this asset set
    # serialize affine transforms in row-major text order. Prefer the interpretation
    # whose last row stays near [0, 0, 0, 1].
    column_major = [
        [values[0], values[4], values[8], values[12]],
        [values[1], values[5], values[9], values[13]],
        [values[2], values[6], values[10], values[14]],
        [values[3], values[7], values[11], values[15]],
    ]
    row_major_affine = (
        abs(float(row_major[3][0])) < 1e-8
        and abs(float(row_major[3][1])) < 1e-8
        and abs(float(row_major[3][2])) < 1e-8
        and abs(float(row_major[3][3]) - 1.0) < 1e-8
    )
    column_major_affine = (
        abs(float(column_major[3][0])) < 1e-8
        and abs(float(column_major[3][1])) < 1e-8
        and abs(float(column_major[3][2])) < 1e-8
        and abs(float(column_major[3][3]) - 1.0) < 1e-8
    )
    if row_major_affine and not column_major_affine:
        return row_major
    if column_major_affine and not row_major_affine:
        return column_major
    return row_major


def _matmul4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)] for row in range(4)]


def _transform_point(matrix: list[list[float]], point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def _mirror_y_mesh(mesh: MeshGeometry) -> MeshGeometry:
    # Unreal's OBJ importer on this toolchain mirrors local Y during RH->LH conversion.
    # Pre-mirror here so the imported StaticMesh local frame matches the original COLLADA/URDF frame.
    # Mirroring changes handedness, so face winding must be flipped to keep normals facing out.
    return MeshGeometry(
        positions=[(float(x), float(-y), float(z)) for x, y, z in mesh.positions],
        faces=[(a, c, b) for a, b, c in mesh.faces],
    )


def _parse_geometries(root: ET.Element) -> dict[str, MeshGeometry]:
    geometries: dict[str, MeshGeometry] = {}
    for geometry in root.findall(".//c:library_geometries/c:geometry", NS):
        mesh = geometry.find("c:mesh", NS)
        if mesh is None:
            continue

        source_positions: dict[str, list[tuple[float, float, float]]] = {}
        for source in mesh.findall("c:source", NS):
            float_array = source.find("c:float_array", NS)
            technique = source.find("c:technique_common/c:accessor", NS)
            if float_array is None or technique is None:
                continue
            stride = int(technique.attrib.get("stride", "3"))
            if stride < 3:
                continue
            values = _parse_float_array(float_array.text)
            triples = []
            for idx in range(0, len(values), stride):
                triples.append((values[idx], values[idx + 1], values[idx + 2]))
            source_positions[source.attrib["id"]] = triples

        vertices_map: dict[str, list[tuple[float, float, float]]] = {}
        for vertices in mesh.findall("c:vertices", NS):
            input_node = vertices.find("c:input[@semantic='POSITION']", NS)
            if input_node is None:
                continue
            source_id = input_node.attrib["source"].lstrip("#")
            if source_id in source_positions:
                vertices_map[vertices.attrib["id"]] = source_positions[source_id]

        positions: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []

        for triangles in mesh.findall("c:triangles", NS):
            inputs = triangles.findall("c:input", NS)
            vertex_offset = None
            vertex_source = None
            stride = 0
            for input_node in inputs:
                offset = int(input_node.attrib.get("offset", "0"))
                stride = max(stride, offset + 1)
                if input_node.attrib.get("semantic") == "VERTEX":
                    vertex_offset = offset
                    vertex_source = input_node.attrib["source"].lstrip("#")
            if vertex_offset is None or vertex_source is None:
                continue
            source_vertices = vertices_map.get(vertex_source)
            if source_vertices is None:
                continue
            p_node = triangles.find("c:p", NS)
            if p_node is None or not p_node.text:
                continue
            raw = [int(item) for item in p_node.text.split()]
            for face_start in range(0, len(raw), stride * 3):
                indices = []
                for corner in range(3):
                    base = face_start + corner * stride
                    source_index = raw[base + vertex_offset]
                    positions.append(source_vertices[source_index])
                    indices.append(len(positions))
                faces.append(tuple(indices))

        geometries[geometry.attrib["id"]] = MeshGeometry(positions=positions, faces=faces)
    return geometries


def read_collada_asset_meta(root: ET.Element) -> tuple[str | None, float]:
    """Return (up_axis text e.g. Z_UP or None, unit length in meters)."""
    asset = root.find("c:asset", NS)
    if asset is None:
        return None, 1.0
    up_el = asset.find("c:up_axis", NS)
    up_axis = str(up_el.text).strip() if up_el is not None and up_el.text else None
    unit_el = asset.find("c:unit", NS)
    meter = 1.0
    if unit_el is not None:
        raw = unit_el.attrib.get("meter")
        if raw is not None:
            meter = float(raw)
    return up_axis, meter


def bake_collada_mesh(input_path: Path) -> MeshGeometry:
    """Bake the default visual scene into one triangle mesh (world vertices).

    Node transforms from library_visual_scenes are applied. <up_axis> and <unit> in
    <asset> are not remapped; Panda assets use Z_UP with meter=1.
    """
    tree = ET.parse(input_path)
    root = tree.getroot()
    geometries = _parse_geometries(root)

    scene_ref = root.find("c:scene/c:instance_visual_scene", NS)
    if scene_ref is None:
        raise RuntimeError(f"No visual scene found in {input_path}")
    visual_scene_id = scene_ref.attrib["url"].lstrip("#")
    visual_scene = root.find(f".//c:library_visual_scenes/c:visual_scene[@id='{visual_scene_id}']", NS)
    if visual_scene is None:
        raise RuntimeError(f"Visual scene {visual_scene_id} not found in {input_path}")

    positions: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for node in visual_scene.findall("c:node", NS):
        _collect_geometry_instances(node, identity, geometries, positions, faces)
    return MeshGeometry(positions=positions, faces=faces)


def _collect_geometry_instances(
    node: ET.Element,
    parent_transform: list[list[float]],
    geometries: dict[str, MeshGeometry],
    out_positions: list[tuple[float, float, float]],
    out_faces: list[tuple[int, int, int]],
) -> None:
    current_transform = _matmul4(parent_transform, _parse_matrix(node.find("c:matrix", NS)))

    for instance in node.findall("c:instance_geometry", NS):
        geometry_id = instance.attrib["url"].lstrip("#")
        geometry = geometries.get(geometry_id)
        if geometry is None:
            continue
        base_index = len(out_positions)
        for position in geometry.positions:
            out_positions.append(_transform_point(current_transform, position))
        for face in geometry.faces:
            out_faces.append((face[0] + base_index, face[1] + base_index, face[2] + base_index))

    for child in node.findall("c:node", NS):
        _collect_geometry_instances(child, current_transform, geometries, out_positions, out_faces)


def convert_collada_to_obj(input_path: Path, output_path: Path, *, mirror_y_for_unreal_obj_import: bool = False) -> None:
    baked = bake_collada_mesh(input_path)
    if mirror_y_for_unreal_obj_import:
        baked = _mirror_y_mesh(baked)
    positions, faces = baked.positions, baked.faces

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for x, y, z in positions:
            handle.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            handle.write(f"f {a} {b} {c}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COLLADA .dae files to OBJ using a built-in XML parser.")
    parser.add_argument("input", type=Path, help="Input .dae file or directory")
    parser.add_argument("output_dir", type=Path, help="Output directory for .obj files")
    parser.add_argument(
        "--mirror-y-for-unreal-obj-import",
        action="store_true",
        help="Pre-mirror Y so UE's OBJ importer lands in the original COLLADA local frame.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if input_path.is_dir():
        for dae_path in sorted(input_path.glob("*.dae")):
            convert_collada_to_obj(
                dae_path,
                output_dir / f"{dae_path.stem}.obj",
                mirror_y_for_unreal_obj_import=bool(args.mirror_y_for_unreal_obj_import),
            )
    else:
        convert_collada_to_obj(
            input_path,
            output_dir / f"{input_path.stem}.obj",
            mirror_y_for_unreal_obj_import=bool(args.mirror_y_for_unreal_obj_import),
        )


if __name__ == "__main__":
    main()
