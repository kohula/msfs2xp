"""
An untextured material is drawn in its own flat colour, as in MSFS --
not with some other material's texture -- and bare metal is darkened,
since X-Plane draws its colour as plain paint (white metal glowed white).
"""
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder, flat_quad  # noqa: E402

import mesh_convert
from mesh_convert.convert import untextured_swatch_color


def _texture_of(material_kwargs, metallic=None):
    b = GltfBuilder()
    tex = b.add_texture(b.add_image_data_uri((10, 200, 30, 255)))
    textured = b.add_material("Brick", base_color_texture_index=tex)
    plain = b.add_material("Panel", **material_kwargs)
    if metallic is not None:
        b._materials[plain].setdefault("pbrMetallicRoughness", {})["metallicFactor"] = metallic
    for i, mat in enumerate((textured, plain)):
        positions, normals, uvs, indices = flat_quad(0, 2, 0, 2)
        positions = [(x + 3 * i, z, 0.0) for (x, _y, z) in positions]
        b.add_node(mesh_index=b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat),
                   name=f"N{i}")
    td = Path(tempfile.mkdtemp())
    (td / "m.glb").write_bytes(b.build())
    (td / "objects").mkdir()
    (td / "textures").mkdir()
    res = mesh_convert.convert(td / "m.glb", td / "objects", td / "textures", td / "textures", "0.0", "0.0", "0.0")
    panel = next(p for p in res if "Panel" in p.stem)
    name = next(l.split()[-1] for l in panel.read_text().splitlines() if l.startswith("TEXTURE "))
    return Image.open(td / "textures" / Path(name).name).convert("RGBA").getpixel((0, 0))


class TestUntexturedMaterials(unittest.TestCase):
    def test_own_colour_not_a_borrowed_texture(self):
        px = _texture_of({"base_color_factor": [0.8, 0.1, 0.1, 1.0]})
        self.assertEqual(px, (204, 25, 25, 255))

    def test_white_metal_is_dark_grey(self):
        px = _texture_of({"base_color_factor": [1.0, 1.0, 1.0, 1.0]}, metallic=1.0)
        self.assertEqual(px[:3], untextured_swatch_color((255, 255, 255, 255), 1.0)[:3])
        self.assertLess(px[0], 100)

    def test_unstated_metalness_is_not_metal(self):
        self.assertEqual(untextured_swatch_color((200, 200, 200, 255), 0.0), (200, 200, 200, 255))


if __name__ == "__main__":
    unittest.main()
