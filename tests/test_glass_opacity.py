"""Glass in X-Plane:

- blended MSFS glass is drawn at the chosen glass opacity (its own alpha is
  near zero -- MSFS draws glass by reflections X-Plane doesn't have), and
  at 100% it's drawn opaque;
- an object with blended geometry sits one layer-group step later, so it
  draws after the opaque parts of the same building (which are separate
  .obj files at the same point) instead of hiding them;
- an OPAQUE material's stray texture alpha no longer punches holes;
- a terrain-fitted copy keeps the night (TEXTURE_LIT) and normal textures.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder, flat_quad  # noqa: E402

import mesh_convert
from mesh_convert import mesh_ir


def _convert(material_name, alpha_mode, rgba, glass_opacity=50, extensions=None):
    b = GltfBuilder()
    tex = b.add_texture(b.add_image_data_uri(rgba))
    mat = b.add_material(material_name, base_color_texture_index=tex, alpha_mode=alpha_mode, extensions=extensions)
    positions, normals, uvs, indices = flat_quad(0, 2, 0, 2)
    # Stand the quad up (a wall, not ground) so it isn't treated as draped.
    positions = [(x, z, 0.0) for (x, _y, z) in positions]
    b.add_node(mesh_index=b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat), name="Pane")
    td = Path(tempfile.mkdtemp())
    glb = td / "pane.glb"
    glb.write_bytes(b.build())
    (td / "objects").mkdir()
    (td / "textures").mkdir()
    result = mesh_convert.convert(glb, td / "objects", td / "textures", td / "textures", "0.0", "0.0", "0.0",
                                  glass_opacity=glass_opacity)
    text = result[0].read_text(encoding="utf-8")
    tex_name = next(l.split("/")[-1] for l in text.splitlines() if l.startswith("TEXTURE "))
    return text, Image.open(td / "textures" / tex_name).convert("RGBA")


class TestGlass(unittest.TestCase):
    def test_near_invisible_glass_is_raised_to_the_glass_opacity(self):
        text, img = _convert("Terminal_Glass", "BLEND", (90, 110, 130, 3), glass_opacity=50)
        self.assertIn("ATTR_blend", text)
        self.assertEqual(img.getchannel("A").getextrema(), (128, 128))
        self.assertEqual(img.getpixel((0, 0))[:3], (90, 110, 130), "colour untouched")

    def test_blended_object_draws_after_opaque_objects(self):
        text, _ = _convert("Terminal_Glass", "BLEND", (90, 110, 130, 3))
        self.assertIn("ATTR_layer_group objects 1", text)
        self.assertLess(text.index("ATTR_layer_group"), text.index("POINT_COUNTS"))

    def test_full_opacity_glass_is_drawn_opaque(self):
        text, img = _convert("Terminal_Glass", "BLEND", (90, 110, 130, 3), glass_opacity=100)
        self.assertNotIn("ATTR_blend", text)
        self.assertNotIn("ATTR_layer_group objects", text)
        self.assertEqual(img.getchannel("A").getextrema(), (255, 255))

    def test_opaque_material_with_stray_alpha_gets_an_opaque_copy(self):
        text, img = _convert("Facade", "OPAQUE", (200, 200, 200, 40))
        self.assertIn("_opaque.png", text)
        self.assertEqual(img.getchannel("A").getextrema(), (255, 255))
        self.assertNotIn("ATTR_layer_group objects", text)

    def test_opaque_material_with_opaque_texture_is_left_alone(self):
        text, _ = _convert("Facade", "OPAQUE", (200, 200, 200, 255))
        self.assertNotIn("_opaque.png", text)


class TestMeshIrKeepsTextures(unittest.TestCase):
    def test_write_obj8_writes_night_and_normal_textures(self):
        ir = mesh_ir.MeshIR(name="w", positions=np.zeros((3, 3)), normals=np.zeros((3, 3)), uvs=np.zeros((3, 2)),
                            indices=np.array([0, 1, 2]), texture="../textures/a.png",
                            texture_lit="../textures/a_lit.png", texture_normal="../textures/a_nm.png",
                            normal_metalness=True, alpha_mode="BLEND", is_glass=True)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "w.obj"
            mesh_ir.write_obj8(ir, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("TEXTURE_LIT ../textures/a_lit.png", text)
        self.assertIn("TEXTURE_NORMAL ../textures/a_nm.png\nNORMAL_METALNESS", text)
        self.assertIn("ATTR_layer_group objects 1", text)
        self.assertIn("ATTR_shiny_rat 1.0", text)


if __name__ == "__main__":
    unittest.main()
