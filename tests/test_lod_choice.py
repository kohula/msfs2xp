"""bgl_extractor.choose_lod: convert the most detailed MSFS level that
fits a size-scaled triangle budget, never stepping into a level that
loses LOD0's textures; and convert.draw_distance_m for small props."""
import json
import struct
import unittest

import bgl_extractor
import importlib

convert_module = importlib.import_module("mesh_convert.convert")


def _glb(tris, half_size, textured=True, mat_name="Wall"):
    js = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "accessors": [
            {"count": 4, "type": "VEC3", "componentType": 5126,
             "min": [-half_size, 0, -half_size], "max": [half_size, 0, half_size]},
            {"count": tris * 3, "type": "SCALAR", "componentType": 5125},
        ],
        "materials": [{"name": mat_name,
                       "pbrMetallicRoughness": ({"baseColorTexture": {"index": 0}} if textured else {})}],
    }
    body = json.dumps(js).encode()
    body += b" " * (-len(body) % 4)
    return b"glTF" + struct.pack("<II", 2, 20 + len(body)) + struct.pack("<I4s", len(body), b"JSON") + body


def _lods(*glbs):
    return [(b"GLB\x00", g) for g in glbs]


class TestChooseLod(unittest.TestCase):
    def test_small_prop_steps_down_to_fit_its_budget(self):
        # ~1 m person: budget 8,000 triangles.
        glb, level = bgl_extractor.choose_lod(_lods(_glb(50_000, 0.7), _glb(20_000, 0.7), _glb(4_000, 0.7)))
        self.assertEqual(level, 2)

    def test_large_building_keeps_lod0_under_the_cap(self):
        _, level = bgl_extractor.choose_lod(_lods(_glb(300_000, 150.0), _glb(100_000, 150.0)))
        self.assertEqual(level, 0)

    def test_never_steps_into_a_level_that_loses_textures(self):
        _, level = bgl_extractor.choose_lod(_lods(_glb(50_000, 0.7), _glb(4_000, 0.7, textured=False)))
        self.assertEqual(level, 0)

    def test_far_only_last_level_with_too_little_detail_is_skipped(self):
        xml = '<ModelInfo><LODS><LOD minSize="10"/><LOD minSize="0"/></LODS></ModelInfo>'
        _, level = bgl_extractor.choose_lod(_lods(_glb(200_000, 2.0), _glb(5_000, 2.0)), xml)
        self.assertEqual(level, 0)

    def test_interiors_get_a_quarter_of_the_budget(self):
        lods = _lods(_glb(400_000, 150.0), _glb(90_000, 150.0))
        self.assertEqual(bgl_extractor.choose_lod(lods, name="Terminal_Interior")[1], 1)
        self.assertEqual(bgl_extractor.choose_lod(lods, name="Terminal")[1], 0)

    def test_multi_lod_container_round_trip(self):
        lod0, lod1 = _glb(50_000, 0.7), _glb(4_000, 0.7)
        glbd = b"".join(b"GLB\x00" + struct.pack("<I", len(g)) + g for g in (lod0, lod1))
        body = b"GXML" + struct.pack("<I", 14) + b'<M name="P"/>\x00' + b"GLBD" + struct.pack("<I", len(glbd)) + glbd
        riff = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"GLTF" + body
        name, xml, lods = bgl_extractor.extract_riff_model_lods(riff, 0, len(riff))
        self.assertEqual(name, "P")
        self.assertEqual(len(lods), 2)
        self.assertEqual(bgl_extractor.choose_lod(lods, xml, name)[0], lod1)


class _B:
    def __init__(self, verts):
        self.vertices = verts


class TestDrawDistance(unittest.TestCase):
    def test_props_fade_buildings_dont(self):
        self.assertEqual(convert_module.draw_distance_m([_B([(-0.5, 0, 0), (0.5, 0, 0)])]), 300)
        self.assertEqual(convert_module.draw_distance_m([_B([(-5, 0, 0), (5, 0, 0)])]), 2000)
        self.assertIsNone(convert_module.draw_distance_m([_B([(-60, 0, 0), (60, 0, 0)])]))


if __name__ == "__main__":
    unittest.main()
