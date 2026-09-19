"""
The confirmed real bug: MSFS routinely reuses ONE shared, neutral texture
across many differently-colored/branded objects, relying on the glTF
material's own baseColorFactor to tint it to the object's real paint color
(glTF spec: final color = baseColorTexture.rgb * baseColorFactor.rgb). This
converter read and stored base_color_factor but only ever applied it to the
flat 2x2 swatch used for a genuinely UNTEXTURED material -- a textured
material's baseColorFactor was silently dropped, so a whole building came
out in the shared texture's own neutral gray/white instead of its real
livery color (a DHL cargo building rendering white instead of yellow, with
only its separately-textured logo decal -- which doesn't rely on this tint
-- looking correct).
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


class TestBaseColorTint(unittest.TestCase):
    def _convert_panel(self, texture_rgba, base_color_factor):
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri(texture_rgba, size=(4, 4)))
        mat = b.add_material("TintedWall", base_color_texture_index=tex, base_color_factor=base_color_factor)
        positions, normals, uvs, indices = flat_quad(0, 2, 0, 2)
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        b.add_node(mesh_index=mesh, name="Wall")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "wall.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()
            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            text = result[0].read_text(encoding="utf-8")
            texture_line = next(l for l in text.splitlines() if l.startswith("TEXTURE "))
            texture_name = texture_line.split(None, 1)[1].rsplit("/", 1)[-1]
            img = Image.open(tex_dir / texture_name).convert("RGBA")
            return np.array(img)

    def test_neutral_texture_gets_tinted_to_material_color(self):
        """A near-white shared texture (200,200,200) with a yellow
        baseColorFactor (DHL-style livery) must come out tinted yellow,
        not left in the shared texture's own neutral color -- the exact
        real-world symptom: a building rendering white instead of its
        actual paint color."""
        arr = self._convert_panel((200, 200, 200, 255), [1.0, 0.85, 0.0, 1.0])
        r, g, b_, a = arr[0, 0]
        self.assertAlmostEqual(int(r), 200, delta=2)
        self.assertAlmostEqual(int(g), int(200 * 0.85), delta=2)
        self.assertAlmostEqual(int(b_), 0, delta=2)
        self.assertEqual(int(a), 255, "alpha must be untouched by the color tint")

    def test_default_white_factor_leaves_texture_unmodified(self):
        """baseColorFactor defaults to (1,1,1,1) -- "no tint" -- and must
        be a true no-op (not even a redundant re-encode), same as before
        this fix for every material that doesn't actually use a tint."""
        arr = self._convert_panel((123, 45, 67, 255), [1.0, 1.0, 1.0, 1.0])
        r, g, b_, a = arr[0, 0]
        self.assertEqual((int(r), int(g), int(b_), int(a)), (123, 45, 67, 255))

    def test_no_base_color_factor_at_all_leaves_texture_unmodified(self):
        """Most real materials never set baseColorFactor explicitly --
        must still default to (1,1,1,1) and skip tinting entirely."""
        arr = self._convert_panel((10, 200, 30, 255), None)
        r, g, b_, a = arr[0, 0]
        self.assertEqual((int(r), int(g), int(b_), int(a)), (10, 200, 30, 255))


if __name__ == "__main__":
    unittest.main()
