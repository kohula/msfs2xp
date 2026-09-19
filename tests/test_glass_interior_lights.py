"""
A window pane (is_glass, alphaMode BLEND) that also carries an emissive
texture -- exactly how "a lit interior room glowing through the glass at
night" is authored in a real airport package -- must still get TEXTURE_LIT.
Confirmed real regression: an earlier version unconditionally skipped
TEXTURE_LIT for every is_glass material to avoid a hypothetical clash with
ATTR_blend, which meant every interior light in every building went dark
(user: "the interior lights of the buildings are not working").
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder, flat_quad  # noqa: E402

import mesh_convert


def _convert_window(material_name="LHBP_B_1_8_Window"):
    b = GltfBuilder()
    base_tex = b.add_texture(b.add_image_data_uri((120, 140, 160, 180)))
    emis_tex = b.add_texture(b.add_image_data_uri((255, 230, 150, 255)))
    mat = b.add_material(material_name, base_color_texture_index=base_tex,
                          emissive_texture_index=emis_tex, alpha_mode="BLEND")
    positions, normals, uvs, indices = flat_quad(0, 3, 0, 3)
    mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
    b.add_node(mesh_index=mesh, name="Window")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        glb = td / "LHBP_B_1_8.glb"
        glb.write_bytes(b.build())
        obj_dir, tex_dir = td / "objects", td / "textures"
        obj_dir.mkdir()
        tex_dir.mkdir()
        result = mesh_convert.convert(glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
        texts = [p.read_text(encoding="utf-8") for p in result]
        return texts


class TestGlassInteriorLights(unittest.TestCase):
    def test_emissive_window_pane_gets_texture_lit(self):
        texts = _convert_window()
        self.assertTrue(any("TEXTURE_LIT" in t for t in texts),
                         "an emissive glass/window material must still get TEXTURE_LIT")

    def test_emissive_window_pane_stays_translucent(self):
        # The fix must not touch the ALREADY-correct glass/alpha handling --
        # only whether TEXTURE_LIT gets written alongside it.
        texts = _convert_window()
        self.assertTrue(any("ATTR_blend" in t for t in texts),
                         "glass/window BLEND alpha must be unaffected by the TEXTURE_LIT fix")

    def test_day_night_switch_material_without_emissive_texture_still_gets_texture_lit(self):
        # The real LHBP buildings light their windows/interiors with
        # ASOBO_material_day_night_switch (or a large emissiveFactor, or a
        # parallax window) and NO dedicated emissive texture -- that was
        # going completely unhandled, so every such interior stayed dark
        # (user: "the interior lights of the buildings don't even light
        # up"). The base colour texture must be synthesised into a
        # TEXTURE_LIT in that case.
        b = GltfBuilder()
        base = b.add_texture(b.add_image_data_uri((90, 110, 140, 255)))
        mat = b.add_material("LHBP_Terminal_Windows_Lit", base_color_texture_index=base,
                              extensions={"ASOBO_material_day_night_switch": {}})
        positions, normals, uvs, indices = flat_quad(0, 3, 0, 3)
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        b.add_node(mesh_index=mesh, name="Windows")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb = td / "term.glb"
            glb.write_bytes(b.build())
            obj_dir, tex_dir = td / "objects", td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()
            result = mesh_convert.convert(glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(any("TEXTURE_LIT" in p.read_text(encoding="utf-8") for p in result),
                            "a day/night-switch material with no emissive texture must still get TEXTURE_LIT")


if __name__ == "__main__":
    unittest.main()
