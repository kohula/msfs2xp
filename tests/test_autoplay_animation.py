"""
An AutoPlay clip (<Animation ... typeParam="AutoPlay"/> in the model XML)
becomes a looping OBJ8 animation on the sim clock.
"""
import json
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


def _eval_block(lines, t, point):
    """Apply one OBJ8 ANIM block's commands (outer first) to a point at
    dataref value t (linear keys, looping)."""
    ops, i = [], 0
    while i < len(lines):
        l = lines[i].split()
        if l[0] == "ANIM_trans" and len(l) >= 7:
            ops.append(("t", np.array([float(v) for v in l[1:4]])))
        elif l[0] in ("ANIM_trans_begin", "ANIM_rotate_begin"):
            axis = np.array([float(v) for v in l[1:4]]) if l[0] == "ANIM_rotate_begin" else None
            keys, loop = [], None
            i += 1
            while not lines[i].startswith(("ANIM_trans_end", "ANIM_rotate_end")):
                k = lines[i].split()
                if k[0] == "ANIM_keyframe_loop":
                    loop = float(k[1])
                else:
                    keys.append([float(v) for v in k[1:]])
                i += 1
            tt = t % loop if loop else t
            ks = np.array(keys)
            val = np.array([np.interp(tt, ks[:, 0], ks[:, c]) for c in range(1, ks.shape[1])])
            ops.append(("t", val) if axis is None else ("r", axis, float(val[0])))
        i += 1
    p = np.asarray(point, dtype=float)
    for op in reversed(ops):
        if op[0] == "t":
            p = p + op[1]
        else:
            ax = op[1] / np.linalg.norm(op[1])
            a = math.radians(op[2])
            p = (p * math.cos(a) + np.cross(ax, p) * math.sin(a) + ax * np.dot(ax, p) * (1 - math.cos(a)))
    return p


class TestChainedAnimation(unittest.TestCase):
    def test_wheel_rides_the_bus_path_and_spins(self):
        """A wheel node (spinning) under a bus node (driving a path and
        turning): the wheel's object carries both, so it moves with the bus
        instead of spinning in one place."""
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((20, 20, 20, 255)))
        mat = b.add_material("Tyre", base_color_texture_index=tex)
        mesh = b.add_mesh([(0, -0.4, -0.4), (0, 0.4, -0.4), (0, 0.4, 0.4), (0, -0.4, 0.4)], [0, 1, 2, 0, 2, 3],
                          normals=[(1, 0, 0)] * 4, uvs=[(0, 0)] * 4, material_index=mat)
        wheel = b.add_node(mesh_index=mesh, name="Wheel", translation=(1.2, 0.5, 3.0), top_level=False)
        bus = b.add_node(name="Bus", children=[wheel])
        times = [i * 1.0 for i in range(9)]
        pos = [(20 * math.cos(math.radians(45 * i)), 0.0, 20 * math.sin(math.radians(45 * i))) for i in range(9)]
        rot = [_y_turn(-45.0 * i) for i in range(9)]
        a = b.add_animation(target_node=bus, path="translation", times=times, values=pos)
        b._animations[a]["name"] = "Bus_Path"
        b._animations[a]["channels"].append({"sampler": 1, "target": {"node": bus, "path": "rotation"}})
        b._animations[a]["samplers"].append({
            "input": b.add_accessor(np.asarray(times, dtype=np.float32), 5126, "SCALAR"),
            "output": b.add_accessor(np.asarray(rot, dtype=np.float32), 5126, "VEC4")})
        spin = b.add_animation(target_node=wheel, path="rotation", times=[0.0, 0.5, 1.0],
                               values=[_x_turn(0), _x_turn(180), _x_turn(360)])
        b._animations[spin]["name"] = "Wheel_Spin"
        td = Path(tempfile.mkdtemp())
        (td / "bus_LOD00.glb").write_bytes(b.build())
        (td / "bus.xml").write_text('<ModelInfo><Animation name="Bus_Path" typeParam="AutoPlay"/>'
                                    '<Animation name="Wheel_Spin" typeParam="AutoPlay"/></ModelInfo>')
        (td / "objects").mkdir()
        (td / "textures").mkdir()
        res = mesh_convert.convert(td / "bus_LOD00.glb", td / "objects", td / "textures", td / "textures",
                                   "0.0", "0.0", "0.0")
        obj = next(p for p in res if "Tyre" in p.stem)
        text = obj.read_text().splitlines()
        self.assertTrue((obj.with_suffix(".autoplay.json")).exists(), "the path is recorded for the terrain")
        start = text.index("ANIM_begin")
        end = next(k for k in range(start, len(text)) if text[k].startswith("TRIS"))
        block = text[start + 1:end]
        rec = json.loads(next((td / "objects").glob("*.originoffset.json")).read_text())
        # the wheel's centre: at rest it is at bus(0) + (1.2, 0.5, 3.0)
        centre_rest = np.array([20.0 + 1.2 - rec["x"], 0.5, 3.0 - rec["z"]])
        for t in (0.0, 2.0, 3.5):
            got = _eval_block(block, t, centre_rest) + np.array([rec["x"], 0.0, rec["z"]])
            # glTF interpolates position linearly between keys, the turn by slerp
            i0 = int(math.floor(t))
            f = t - i0
            bus_pos = (1 - f) * np.array(pos[i0]) + f * np.array(pos[min(i0 + 1, 8)])
            # glTF: bus turned by -45*t deg about Y, wheel offset rotated with it
            a = math.radians(-45 * t)
            off = np.array([1.2 * math.cos(a) + 3.0 * math.sin(a), 0.5, -1.2 * math.sin(a) + 3.0 * math.cos(a)])
            np.testing.assert_allclose(got, bus_pos + off, atol=0.15)


if __name__ == "__main__":
    unittest.main()
