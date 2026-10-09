"""
Lamps: emissive lamp clusters without a light source get a halo-only
light; a synthesized fixture light is placed in the output frame (no
second global rotation); a lights-only model converts to a light.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert
from mesh_convert.convert import lamp_head_points, MatBuilder


def _quad(b, mat, cx, cy, cz, h=0.15):
    return b.add_mesh([(cx - h, cy, cz - h), (cx + h, cy, cz - h), (cx + h, cy, cz + h), (cx - h, cy, cz + h)],
                      [0, 1, 2, 0, 2, 3], normals=[(0, -1, 0)] * 4, uvs=[(0, 0)] * 4, material_index=mat)


def _run(b, name, yaw="0.0", **kw):
    td = Path(tempfile.mkdtemp())
    (td / name).write_bytes(b.build())
    (td / "o").mkdir()
    (td / "t").mkdir()
    res = mesh_convert.convert(td / name, td / "o", td / "t", td / "t", "0.0", yaw, "0.0", **kw)
    lights = next((p for p in res if p.stem.endswith("_lights")), None)
    return [l.split() for l in lights.read_text().splitlines() if l.startswith("LIGHT_")] if lights else []


def _canopy_with_lamps():
    b = GltfBuilder()
    tex = b.add_texture(b.add_image_data_uri((120, 120, 120, 255)))
    lit = b.add_texture(b.add_image_data_uri((255, 230, 180, 255)))
    roof = b.add_material("Canopy_Roof", base_color_texture_index=tex)
    lamp = b.add_material("Canopy_Lamp_Emis", base_color_texture_index=tex, emissive_texture_index=lit)
    b.add_node(mesh_index=b.add_mesh([(-10, 4, -5), (10, 4, -5), (10, 4, 5), (-10, 4, 5)], [0, 1, 2, 0, 2, 3],
                                     normals=[(0, -1, 0)] * 4, uvs=[(0, 0)] * 4, material_index=roof), name="Roof")
    for x in (-6.0, 0.0, 6.0):
        b.add_node(mesh_index=_quad(b, lamp, x, 3.9, 0.0), name=f"Lamp{x}")
    return b


class TestLampHeads(unittest.TestCase):
    def test_each_lamp_gets_a_halo(self):
        lines = _run(_canopy_with_lamps(), "Canopy.glb")
        self.assertEqual(len(lines), 3)
        xs = sorted(round(float(l[2]), 2) for l in lines)
        self.assertEqual(xs, [-6.0, 0.0, 6.0])
        for l in lines:
            self.assertEqual(l[1], "full_custom_halo_night")
            self.assertAlmostEqual(float(l[9]), 0.5)  # small: a glow, no ground pool

    def test_switch_off(self):
        self.assertEqual(_run(_canopy_with_lamps(), "Canopy.glb", lamp_glow=False), [])

    def test_lit_windows_are_not_lamps(self):
        b = MatBuilder("Facade_Window_Emis")
        b.vertices = [(0, 0, 0), (0.2, 0, 0), (0.2, 0.2, 0)]
        b.emissive_texture_name = "x.png"
        self.assertEqual(lamp_head_points({0: b}), [])


class TestFixtureLightPlace(unittest.TestCase):
    def test_light_sits_at_its_lens_when_turned(self):
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((255, 240, 200, 255)))
        mat = b.add_material("Lens_Emis", base_color_texture_index=tex)
        b.add_node(mesh_index=_quad(b, mat, 3.0, 12.0, 1.0, h=0.3), name="Head")
        lines = _run(b, "SHS_ApronLight_Arm.glb", yaw="180.0")
        self.assertTrue(lines)
        x, z = float(lines[0][2]), float(lines[0][4])
        self.assertAlmostEqual(x, -3.0, places=3)  # turned 180: where the lens is drawn
        self.assertAlmostEqual(z, -1.0, places=3)


class TestLightOnlyModel(unittest.TestCase):
    def test_lights_only_model_converts(self):
        b = GltfBuilder()
        h = math.radians(90.0) / 2.0
        b.add_node(name="Light", translation=(0, 0, 0), rotation=(math.sin(h), 0, 0, math.cos(h)),
                   extensions={"ASOBO_macro_light": {"color": [1, 0.9, 0.8], "cone_angle": 180,
                                                     "intensity": 400, "day_night_cycle": True}})
        lines = _run(b, "Light_Warm.glb")
        self.assertEqual(len(lines), 1)
        self.assertLess(float(lines[0][11]), -0.9)


if __name__ == "__main__":
    unittest.main()
