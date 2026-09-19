"""
mesh_convert.convert synthesizes a night LIGHT_SPILL_CUSTOM for an
emissive-only light FIXTURE (apron pole, flood, wig-wag, street lamp)
that ships no ASOBO_macro_light / KHR_lights_punctual node -- otherwise
X-Plane self-glows the TEXTURE_LIT lens but the fixture casts no light on
the ground and can't flash. Gated on the model NAME (a lit sign / window
/ facade must NOT become a floodlight) plus an emissive builder that is
physically small.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert


def _convert(glb_name, material_name, size=(0.4, 0.3), y=5.0):
    b = GltfBuilder()
    tex = b.add_texture(b.add_image_data_uri((255, 240, 200, 255)))
    mat = b.add_material(material_name, base_color_texture_index=tex)
    hx, hz = size
    positions = [(-hx, y, -hz), (hx, y, -hz), (hx, y + 0.2, -hz), (-hx, y + 0.2, -hz)]
    mesh = b.add_mesh(positions, [0, 1, 2, 0, 2, 3],
                      normals=[(0.0, 0.0, -1.0)] * 4, uvs=[(0.0, 0.0)] * 4, material_index=mat)
    b.add_node(mesh_index=mesh, name="Head")   # NO add_macro_light
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        glb = td / glb_name
        glb.write_bytes(b.build())
        obj_dir, tex_dir = td / "objects", td / "textures"
        obj_dir.mkdir()
        tex_dir.mkdir()
        result = mesh_convert.convert(glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
        lights = next((p for p in result if p.stem.endswith("_lights")), None)
        return lights.read_text(encoding="utf-8") if lights else None


class TestSynthesizedFixtureLights(unittest.TestCase):
    def test_apron_light_fixture_gets_the_stock_night_flood_param_light(self):
        # An emissive-only apron/pole fixture -> X-Plane's own registered
        # param light `full_custom_halo_night` (SPILL_HW_DIR, night-gated
        # in the sim engine, no dataref). NOT LIGHT_SPILL_CUSTOM (unused by
        # stock scenery, didn't render) and NOT a scalar-dataref gate.
        text = _convert("SHS_ApronLight_Test.glb", "Lens_Emis")
        self.assertIsNotNone(text, "an emissive-only light fixture should get a synthesized light")
        self.assertIn("LIGHT_PARAM full_custom_halo_night", text)
        self.assertNotIn("LIGHT_SPILL_CUSTOM", text)
        self.assertNotIn("msfs2xp/", text)
        self.assertNotIn("percent_lights_on", text)
        # LIGHT_PARAM <name> px py pz  R G B A  S  X Y Z  F   (14 tokens)
        pl = next(l for l in text.splitlines() if l.startswith("LIGHT_PARAM ")).split()
        self.assertEqual(len(pl), 14)
        self.assertEqual((float(pl[10]), float(pl[11]), float(pl[12])), (0.0, -1.0, 0.0), "aimed down")

    def test_raised_fixture_gets_a_metre_scale_reach(self):
        # A fixture head sitting well above its base -> the `S` (reach in
        # metres) param is floored to something that actually lights the
        # ground from up there, not the 1..3 the intensity formula caps at.
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((255, 240, 200, 255)))
        matp = b.add_material("Pole", base_color_texture_index=tex)
        mate = b.add_material("Lens_Emis", base_color_texture_index=tex)
        pole = b.add_mesh([(-0.1, 0, -0.1), (0.1, 0, -0.1), (0.1, 14, -0.1), (-0.1, 14, -0.1)],
                          [0, 1, 2, 0, 2, 3], normals=[(0, 0, -1)] * 4, uvs=[(0, 0)] * 4, material_index=matp)
        lens = b.add_mesh([(-0.4, 14, -0.3), (0.4, 14, -0.3), (0.4, 14.3, -0.3), (-0.4, 14.3, -0.3)],
                          [0, 1, 2, 0, 2, 3], normals=[(0, 0, -1)] * 4, uvs=[(0, 0)] * 4, material_index=mate)
        b.add_node(mesh_index=pole, name="Pole")
        b.add_node(mesh_index=lens, name="Head")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb = td / "SHS_ApronLight_Tall.glb"
            glb.write_bytes(b.build())
            od, xd = td / "objects", td / "textures"
            od.mkdir()
            xd.mkdir()
            result = mesh_convert.convert(glb, od, xd, xd, "0.0", "0.0", "0.0")
            pl = next(l for l in
                      next(p for p in result if p.stem.endswith("_lights")).read_text().splitlines()
                      if l.startswith("LIGHT_PARAM ")).split()
            self.assertGreaterEqual(float(pl[9]), 12.0, "reach scales to the ~14 m mount height")

    def test_wigwag_fixture_gets_a_flashing_light(self):
        text = _convert("SHS_WigWag_Test.glb", "WigWag_Emis")
        self.assertIsNotNone(text)
        # A wig-wag / runway-guard uses X-Plane's own built-in animated
        # named light -- self-flashing, no plugin, no custom dataref.
        self.assertIn("LIGHT_NAMED wigwag_y", text)
        self.assertNotIn("msfs2xp/", text)

    def test_lit_sign_is_NOT_turned_into_a_light(self):
        self.assertIsNone(_convert("SHS_outdoor_sign_001.glb", "Sign_Emis"),
                          "a lit sign must not be synthesized into a floodlight")

    def test_lit_facade_is_NOT_turned_into_a_light(self):
        self.assertIsNone(_convert("LHBP_B_1_7_Windows.glb", "Windows_Emis"))

    def test_big_emissive_object_under_a_light_name_is_skipped(self):
        # name passes the gate, but the emissive builder is 60 m wide -> a
        # lit wall, not a fixture head -> no synthesized light
        self.assertIsNone(_convert("SHS_BigLight_Wall.glb", "Panel_Emis", size=(30.0, 0.3)))

    def test_a_real_macro_light_is_not_doubled(self):
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((255, 240, 200, 255)))
        mat = b.add_material("Head_Emis", base_color_texture_index=tex)
        mesh = b.add_mesh([(-0.3, 5, -0.3), (0.3, 5, -0.3), (0.3, 5.2, -0.3), (-0.3, 5.2, -0.3)],
                          [0, 1, 2, 0, 2, 3], normals=[(0.0, 0.0, -1.0)] * 4,
                          uvs=[(0.0, 0.0)] * 4, material_index=mat)
        node = b.add_node(mesh_index=mesh, name="Head")
        b.add_macro_light(node, color=(1.0, 1.0, 0.9), cone_angle=180.0, intensity=5.0,
                          day_night_cycle=True, flash_frequency=0.0)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb = td / "SHS_Generic_Head_Real.glb"   # not a stock-named type -> spill path
            glb.write_bytes(b.build())
            obj_dir, tex_dir = td / "objects", td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()
            result = mesh_convert.convert(glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            text = next(p for p in result if p.stem.endswith("_lights")).read_text(encoding="utf-8")
            n_lights = sum(text.count(c) for c in ("LIGHT_PARAM ", "LIGHT_NAMED ", "LIGHT_SPILL_CUSTOM"))
            self.assertEqual(n_lights, 1, "the real macro_light must not be doubled by a synthesized one")

    def test_downlight_fixture_classification(self):
        # Drives the writer's "emitter axis came through pointing up ->
        # flip it to (0,-1,0)" guard: street lamps / apron poles / floods
        # / base lights light the ground; uplights / beacons / facade
        # washes really do point up and must be left alone.
        from mesh_convert.convert import _is_downlight_fixture
        for n in ("StreetLamp_Type1_Single", "SHS_Clutter_ApronLight_001",
                  "LHBP_BaseLight_Pole_2", "SHS_Floodlight_001"):
            self.assertTrue(_is_downlight_fixture(n), n)
        for n in ("LHBP_Facade_Uplight_3", "Airport_Beacon_Rotating",
                  "Terminal_Wall_Wash_S", "SHS_outdoor_sign_001"):
            self.assertFalse(_is_downlight_fixture(n), n)


if __name__ == "__main__":
    unittest.main()
