"""
An AutoPlay clip (<Animation ... typeParam="AutoPlay"/> in the model XML)
becomes a looping OBJ8 animation on the sim clock.
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert
from mesh_convert import autoplay


def _y_turn(deg):
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def _x_turn(deg):
    h = math.radians(deg) / 2.0
    return (math.sin(h), 0.0, 0.0, math.cos(h))


def _convert(xml, quats, times, name="Radar_Spin"):
    b = GltfBuilder()
    tex = b.add_texture(b.add_image_data_uri((180, 180, 180, 255)))
    mat = b.add_material("Dish", base_color_texture_index=tex)
    mesh = b.add_mesh([(-2, 0, 0), (2, 0, 0), (2, 1.5, 0.3), (-2, 1.5, 0.3)], [0, 1, 2, 0, 2, 3],
                      normals=[(0, 0, -1)] * 4, uvs=[(0, 0)] * 4, material_index=mat)
    node = b.add_node(mesh_index=mesh, name="Dish", translation=(0.0, 6.0, 0.0))
    idx = b.add_animation(target_node=node, path="rotation", times=times, values=quats)
    b._animations[idx]["name"] = name
    td = Path(tempfile.mkdtemp())
    (td / "radar_LOD00.glb").write_bytes(b.build())
    (td / "radar.xml").write_text(xml, encoding="utf-8")
    (td / "objects").mkdir()
    (td / "textures").mkdir()
    res = mesh_convert.convert(td / "radar_LOD00.glb", td / "objects", td / "textures", td / "textures",
                               "0.0", "0.0", "0.0")
    return "\n".join(p.read_text() for p in res)


AUTOPLAY_XML = ('<ModelInfo version="1.1" guid="{0}"><LODS><LOD minSize="0" ModelFile="radar_LOD00.glb"/></LODS>'
                '<Animation name="Radar_Spin" guid="" type="Standard" typeParam="AutoPlay"/></ModelInfo>')


class TestAutoplayNames(unittest.TestCase):
    def test_only_autoplay_animations(self):
        xml = ('<Animation name="Spin" typeParam="AutoPlay"/>'
               '<Animation typeParam="autoplay" name="Fan_Blades" guid="x"/>'
               '<Animation name="Door" type="Sim" typeParam="AnimTime"/>')
        self.assertEqual(autoplay.autoplay_animation_names(xml), {"spin", "fan_blades"})


class TestRotationTurn(unittest.TestCase):
    def test_full_turn_is_one_continuous_sweep(self):
        angles = list(range(0, 361, 30))
        axis, swept = autoplay.rotation_turn(_y_turn(0), [_y_turn(a) for a in angles], np.eye(3))
        self.assertAlmostEqual(abs(axis[1]), 1.0, places=6)
        sign = 1.0 if axis[1] > 0 else -1.0
        np.testing.assert_allclose([a * sign for a in swept], angles, atol=1e-3)

    def test_wobble_about_two_axes_is_left_still(self):
        quats = [_y_turn(0), _y_turn(40), _x_turn(40), _y_turn(0)]
        self.assertIsNone(autoplay.rotation_turn(_y_turn(0), quats, np.eye(3)))

    def test_thinning_keeps_a_steady_turn_short(self):
        times = np.linspace(0, 4, 121)
        keys = autoplay.thin_turn(times, times * 90.0)
        self.assertEqual(keys, [(0.0, 0.0), (4.0, 360.0)])


class TestAutoplayConversion(unittest.TestCase):
    def test_radar_loops_on_the_sim_clock(self):
        times = [i * 0.5 for i in range(9)]
        quats = [_y_turn(45.0 * i) for i in range(9)]
        text = _convert(AUTOPLAY_XML, quats, times)
        self.assertIn("ANIM_rotate_begin", text)
        self.assertIn("sim/time/total_running_time_sec", text)
        self.assertIn("ANIM_keyframe_loop 4.0000", text)
        keys = [l.split() for l in text.splitlines() if l.startswith("ANIM_rotate_key")]
        self.assertAlmostEqual(abs(float(keys[-1][2]) - float(keys[0][2])), 360.0, places=2)
        # it turns about the mast (Y), around the dish's own origin 6 m up
        axis = [float(v) for v in next(l for l in text.splitlines()
                                       if l.startswith("ANIM_rotate_begin")).split()[1:4]]
        self.assertAlmostEqual(abs(axis[1]), 1.0, places=4)

    def test_without_autoplay_it_stays_still(self):
        times = [i * 0.5 for i in range(9)]
        quats = [_y_turn(45.0 * i) for i in range(9)]
        text = _convert(AUTOPLAY_XML.replace("AutoPlay", "AnimTime"), quats, times)
        self.assertNotIn("ANIM_", text)


class TestPathAnimation(unittest.TestCase):
    def test_vehicle_on_a_path_moves_and_turns(self):
        """One node with a translation and a rotation channel (a bus driving
        a loop): both are written -- the move, then the turn about the bus's
        own origin -- instead of only the first channel (which slid the bus
        along the path without turning it)."""
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((180, 30, 30, 255)))
        mat = b.add_material("Bus", base_color_texture_index=tex)
        mesh = b.add_mesh([(-1.2, 0, -5), (1.2, 0, -5), (1.2, 3, 5), (-1.2, 3, 5)], [0, 1, 2, 0, 2, 3],
                          normals=[(0, 0, -1)] * 4, uvs=[(0, 0)] * 4, material_index=mat)
        node = b.add_node(mesh_index=mesh, name="Bus")
        times = [i * 1.0 for i in range(9)]
        a = b.add_animation(target_node=node, path="translation", times=times,
                            values=[(20 * math.cos(math.radians(45 * i)), 0.0, 20 * math.sin(math.radians(45 * i)))
                                    for i in range(9)])
        b._animations[a]["name"] = "Bus_Path"
        b._animations[a]["channels"].append({"sampler": 1, "target": {"node": node, "path": "rotation"}})
        r_in = b.add_accessor(np.asarray(times, dtype=np.float32), 5126, "SCALAR")
        r_out = b.add_accessor(np.asarray([_y_turn(-45.0 * i) for i in range(9)], dtype=np.float32), 5126, "VEC4")
        b._animations[a]["samplers"].append({"input": r_in, "output": r_out})
        td = Path(tempfile.mkdtemp())
        (td / "bus_LOD00.glb").write_bytes(b.build())
        (td / "bus.xml").write_text('<ModelInfo><Animation name="Bus_Path" typeParam="AutoPlay"/></ModelInfo>')
        (td / "objects").mkdir()
        (td / "textures").mkdir()
        res = mesh_convert.convert(td / "bus_LOD00.glb", td / "objects", td / "textures", td / "textures",
                                   "0.0", "180.0", "0.0")
        text = "\n".join(p.read_text() for p in res)
        self.assertIn("ANIM_trans_begin sim/time/total_running_time_sec", text)
        self.assertIn("ANIM_rotate_begin", text)
        self.assertEqual(text.count("ANIM_keyframe_loop 8.0000"), 2)
        keys = [l.split() for l in text.splitlines() if l.startswith("ANIM_rotate_key")]
        self.assertAlmostEqual(abs(float(keys[-1][2]) - float(keys[0][2])), 360.0, places=1)


if __name__ == "__main__":
    unittest.main()
