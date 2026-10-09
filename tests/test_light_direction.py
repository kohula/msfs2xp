"""
Which way converted lights shine. MSFS's own lights (ASOBO_macro_light,
and MSFS 2024's ASOBO_advanced_light) shine along their node's +Z axis; a
standard KHR_lights_punctual light along -Z. A lamp-post spot is a node
turned ~110 degrees about X so its +Z points at the ground: read the
wrong way round it lit the sky.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert


def _about_x(deg):
    h = math.radians(deg) / 2.0
    return (math.sin(h), 0.0, 0.0, math.cos(h))


def _light_lines(extensions, rotation, khr_defs=None):
    b = GltfBuilder()
    tex = b.add_texture(b.add_image_data_uri((128, 128, 128, 255)))
    mat = b.add_material("Concrete", base_color_texture_index=tex)
    mesh = b.add_mesh([(-0.2, 0, -0.2), (0.2, 0, -0.2), (0.2, 8, -0.2), (-0.2, 8, -0.2)], [0, 1, 2, 0, 2, 3],
                      normals=[(0, 0, -1)] * 4, uvs=[(0, 0)] * 4, material_index=mat)
    b.add_node(mesh_index=mesh, name="Mast")
    b.add_node(name="Lamp", translation=(0.0, 8.0, 0.0), rotation=rotation, extensions=extensions)
    data = b.build()
    if khr_defs is not None:
        import json
        import struct
        # put the KHR light definitions at the top level of the JSON chunk
        jlen = struct.unpack_from("<I", data, 12)[0]
        gltf = json.loads(data[20:20 + jlen])
        gltf.setdefault("extensions", {})["KHR_lights_punctual"] = {"lights": khr_defs}
        js = json.dumps(gltf).encode()
        js += b" " * (-len(js) % 4)
        rest = data[20 + jlen:]
        data = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + len(rest)) + \
            struct.pack("<II", len(js), 0x4E4F534A) + js + rest
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        glb = td / "Mast_Test.glb"
        glb.write_bytes(data)
        od, xd = td / "objects", td / "textures"
        od.mkdir()
        xd.mkdir()
        result = mesh_convert.convert(glb, od, xd, xd, "0.0", "0.0", "0.0")
        lights = next(p for p in result if p.stem.endswith("_lights"))
        return [l.split() for l in lights.read_text().splitlines() if l.startswith("LIGHT_PARAM ")]


class TestLightDirection(unittest.TestCase):
    def test_msfs_light_turned_down_shines_down(self):
        lines = _light_lines({"ASOBO_macro_light": {"color": [1, 1, 1], "cone_angle": 45, "intensity": 10}},
                             _about_x(110.0))
        self.assertEqual(len(lines), 1)
        self.assertLess(float(lines[0][11]), -0.9)

    def test_msfs_2024_advanced_light_is_converted(self):
        ext = {"ASOBO_advanced_light": {"color": [1, 0.9, 0.85], "intensity": 4000, "day_night_cycle": True,
                                        "inner_cone_angle": 100, "outer_cone_angle": 140}}
        lines = _light_lines(ext, _about_x(90.0))
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0][1], "full_custom_halo_night")
        self.assertLess(float(lines[0][11]), -0.9)
        # cone from the outer angle: cos(70 deg), at most
        self.assertLessEqual(float(lines[0][13]), math.cos(math.radians(70.0)) + 1e-3)

    def test_standard_gltf_light_keeps_minus_z(self):
        lines = _light_lines({"KHR_lights_punctual": {"light": 0}}, _about_x(-90.0),
                             khr_defs=[{"type": "spot", "intensity": 5.0, "spot": {"outerConeAngle": 0.5}}])
        self.assertEqual(len(lines), 1)
        self.assertLess(float(lines[0][11]), -0.9)


if __name__ == "__main__":
    unittest.main()
