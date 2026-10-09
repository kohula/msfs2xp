"""Normal maps are written in X-Plane's NORMAL_METALNESS layout: normal in
red/green, metalness in blue, smoothness in alpha (from the material's
metal/roughness texture and factors)."""
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder, flat_quad  # noqa: E402

import mesh_convert


def _convert(with_comp, factors=None, normal_scale=None):
    b = GltfBuilder()
    base = b.add_texture(b.add_image_data_uri((180, 180, 180, 255)))
    normal = b.add_texture(b.add_image_data_uri((100, 150, 255, 255)))
    mat = b.add_material("Wall", base_color_texture_index=base, alpha_mode="OPAQUE")
    m = b._materials[mat]
    m["normalTexture"] = {"index": normal}
    if normal_scale is not None:
        m["normalTexture"]["scale"] = normal_scale
    if with_comp:
        comp = b.add_texture(b.add_image_data_uri((255, 64, 200, 255)))  # AO, roughness 64, metal 200
        m["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": comp}
    m["pbrMetallicRoughness"].update(factors or {})
    positions, normals, uvs, indices = flat_quad(0, 2, 0, 2)
    positions = [(x, z, 0.0) for (x, _y, z) in positions]
    b.add_node(mesh_index=b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat), name="W")
    td = Path(tempfile.mkdtemp())
    (td / "w.glb").write_bytes(b.build())
    (td / "objects").mkdir()
    (td / "textures").mkdir()
    res = mesh_convert.convert(td / "w.glb", td / "objects", td / "textures", td / "textures", "0.0", "0.0", "0.0")
    text = res[0].read_text(encoding="utf-8")
    nm = next(l.split("/")[-1] for l in text.splitlines() if l.startswith("TEXTURE_NORMAL"))
    return text, Image.open(td / "textures" / nm).convert("RGBA").getpixel((0, 0))


class TestNormalMetalness(unittest.TestCase):
    def test_with_metal_roughness_texture(self):
        text, px = _convert(True)
        self.assertIn("NORMAL_METALNESS", text)
        self.assertEqual(px, (100, 150, 200, 255 - 64))

    def test_factors_scale_the_texture(self):
        _, px = _convert(True, {"metallicFactor": 0.5, "roughnessFactor": 0.5})
        self.assertEqual(px[2], 100)
        self.assertEqual(px[3], 255 - 32)

    def test_without_texture_only_stated_factors_count(self):
        _, px = _convert(False)
        self.assertEqual(px[2], 0, "no metal unless the material says so")
        _, px = _convert(False, {"metallicFactor": 1.0, "roughnessFactor": 0.2})
        self.assertEqual(px[2:], (255, 255 - 51))


    def test_normal_scale_weakens_the_bumps(self):
        """normalTexture.scale 0.2 (a glass pane's faint ripple): red/green
        move a fifth of the way from flat (127.5) to the map's values."""
        _, px = _convert(True, normal_scale=0.2)
        self.assertEqual(px[:2], (round(127.5 + (100 - 127.5) * 0.2), round(127.5 + (150 - 127.5) * 0.2)))
        _, px = _convert(True, normal_scale=1.0)
        self.assertEqual(px[:2], (100, 150))


if __name__ == "__main__":
    unittest.main()
