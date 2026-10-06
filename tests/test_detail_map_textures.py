"""
An MSFS detail map (ASOBO_material_detail_map) is a small tile repeated
across a surface at its own UVScale under a blend mask. Its normal map is
not a stand-in for the material's own: read at the model's UVs it
stretched one tile over a whole wall, lit by X-Plane as a large
black-and-white blotch pattern. Its colour texture, when it's the only one,
keeps its tiling.
"""
import unittest

from mesh_convert.convert import find_base_color_texture, find_normal_texture


def _detail(**extra):
    d = {"detailColorTexture": {"index": 3}, "detailNormalTexture": {"index": 4}}
    d.update(extra)
    return {"extensions": {"ASOBO_material_detail_map": d}}


class TestDetailMapTextures(unittest.TestCase):
    def test_detail_normal_is_not_used_as_the_normal_map(self):
        self.assertIsNone(find_normal_texture(_detail()))
        mat = _detail()
        mat["normalTexture"] = {"index": 7}
        self.assertEqual(find_normal_texture(mat), {"index": 7})

    def test_own_base_colour_wins_over_the_detail_map(self):
        mat = _detail(UVScale=8.0)
        mat["pbrMetallicRoughness"] = {"baseColorTexture": {"index": 1}}
        self.assertEqual(find_base_color_texture(mat), {"index": 1})

    def test_detail_colour_alone_keeps_its_tiling(self):
        tex = find_base_color_texture(_detail(UVScale=8.0, UVOffset=[0.5, 0.25]))
        self.assertEqual(tex["index"], 3)
        transform = tex["extensions"]["KHR_texture_transform"]
        self.assertEqual(transform["scale"], [8.0, 8.0])
        self.assertEqual(transform["offset"], [0.5, 0.25])

    def test_detail_colour_without_a_scale_is_unchanged(self):
        self.assertEqual(find_base_color_texture(_detail()), {"index": 3})


if __name__ == "__main__":
    unittest.main()
