"""
scenery_viewer.py's DSF/OBJ8 reader is a matching decoder for what THIS
project's own dsf_compiler.py/mesh_convert write -- pins the round trip:
compile a real tile with dsf_compiler.build_dsf (the same golden-hash-
pinned function tests/test_dsf_compiler.py covers), write a couple of real
OBJ8 .obj files next to it, and confirm scenery_viewer.parse_dsf/
parse_obj8/build_scene recover the same objects/exclusions/positions.
"""
import math
import tempfile
import unittest
from pathlib import Path

import dsf_compiler
import scenery_viewer


class TestSceneryViewer(unittest.TestCase):
    def test_parse_dsf_recovers_objects_and_exclusions(self):
        objects = [
            {"name": "test_obj_a", "lat": 47.1234, "lon": 8.5678, "hdg": 90.0, "agl": 0.0},
            {"name": "test_obj_b", "lat": 47.1250, "lon": 8.5690, "hdg": 180.0, "agl": 0.0},
            {"name": "test_obj_c", "lat": 47.1240, "lon": 8.5680, "hdg": 0.0, "agl": 3.5},
        ]
        exclusions = [{"west": 8.5, "south": 47.1, "east": 8.6, "north": 47.2}]

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_tile.dsf"
            dsf_compiler.build_dsf(47, 8, objects, out_path, exclusions=exclusions)

            tile = scenery_viewer.parse_dsf(out_path)
            self.assertIsNotNone(tile)
            self.assertEqual(len(tile.objects), 3)
            self.assertEqual(len(tile.exclusions), 1)

            excl = tile.exclusions[0]
            self.assertAlmostEqual(excl["west"], 8.5, places=5)
            self.assertAlmostEqual(excl["south"], 47.1, places=5)
            self.assertAlmostEqual(excl["east"], 8.6, places=5)
            self.assertAlmostEqual(excl["north"], 47.2, places=5)
            self.assertEqual(excl["category"], "obj")

            by_name = {o["name"]: o for o in tile.objects}
            self.assertIn("objects/test_obj_a.obj", by_name)
            self.assertIn("objects/test_obj_c.obj", by_name)

            a = by_name["objects/test_obj_a.obj"]
            self.assertAlmostEqual(a["lat"], 47.1234, places=3)
            self.assertAlmostEqual(a["lon"], 8.5678, places=3)
            self.assertAlmostEqual(a["hdg"], 90.0, places=1)
            self.assertFalse(a["is_agl"])

            c = by_name["objects/test_obj_c.obj"]
            self.assertTrue(c["is_agl"], "test_obj_c has agl=3.5 -- must route through the AGL pool")
            self.assertAlmostEqual(c["agl"], 3.5, delta=0.05)

    def test_parse_dsf_no_exclusions_or_agl_pool(self):
        """A tile with only ground-level objects and no exclusions must not
        even attempt to build a (nonexistent) second AGL pool -- confirms
        parse_dsf's pool-count handling matches build_geod_atom's own
        "second pool only when agl_objects is non-empty" behavior."""
        objects = [{"name": "solo", "lat": 40.0, "lon": 10.0, "hdg": 45.0, "agl": 0.0}]
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "solo_tile.dsf"
            dsf_compiler.build_dsf(40, 10, objects, out_path)
            tile = scenery_viewer.parse_dsf(out_path)
            self.assertEqual(len(tile.exclusions), 0)
            self.assertEqual(len(tile.objects), 1)
            self.assertAlmostEqual(tile.objects[0]["lat"], 40.0, places=3)
            self.assertAlmostEqual(tile.objects[0]["lon"], 10.0, places=3)

    def test_parse_obj8_reads_draped_and_rigid_attributes(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            draped_path = td / "draped.obj"
            draped_path.write_text(
                "I\n800\nOBJ\n\nTEXTURE ../textures/asphalt.png\n\n"
                "POINT_COUNTS 4 0 0 6\n\n"
                "VT 0.0 0.0 0.0 0 1 0 0 0\nVT 10.0 0.0 0.0 0 1 0 1 0\n"
                "VT 10.0 0.0 10.0 0 1 0 1 1\nVT 0.0 0.0 10.0 0 1 0 0 1\n\n"
                "IDX 0\nIDX 1\nIDX 2\nIDX 0\nIDX 2\nIDX 3\n\n"
                "ATTR_draped\nATTR_layer_group_draped markings -3\nTRIS 0 6\n",
                encoding="utf-8",
            )
            rigid_path = td / "rigid.obj"
            rigid_path.write_text(
                "I\n800\nOBJ\n\nTEXTURE ../textures/building.png\n\nTILTED\n"
                "POINT_COUNTS 1 0 0 0\n\nVT 0 0 0 0 1 0 0 0\n\nTRIS 0 0\n",
                encoding="utf-8",
            )

            draped_ir = scenery_viewer.parse_obj8(draped_path)
            self.assertTrue(draped_ir["draped"])
            self.assertEqual(draped_ir["layer_offset"], -3)
            self.assertEqual(len(draped_ir["positions"]), 4)
            self.assertFalse(draped_ir["tilted"])

            rigid_ir = scenery_viewer.parse_obj8(rigid_path)
            self.assertFalse(rigid_ir["draped"])
            self.assertTrue(rigid_ir["tilted"])

    def test_build_scene_end_to_end_classifies_draped_vs_rigid_vs_library(self):
        with tempfile.TemporaryDirectory() as td:
            pack_dir = Path(td)
            nav_dir = pack_dir / "Earth nav data"
            nav_dir.mkdir(parents=True)
            objects_dir = pack_dir / "objects"
            objects_dir.mkdir()

            (objects_dir / "pave.obj").write_text(
                "I\n800\nOBJ\n\nTEXTURE ../textures/asphalt.png\n\n"
                "POINT_COUNTS 4 0 0 6\n\n"
                "VT -5.0 0.0 -5.0 0 1 0 0 0\nVT 5.0 0.0 -5.0 0 1 0 1 0\n"
                "VT 5.0 0.0 5.0 0 1 0 1 1\nVT -5.0 0.0 5.0 0 1 0 0 1\n\n"
                "IDX 0\nIDX 1\nIDX 2\nIDX 0\nIDX 2\nIDX 3\n\n"
                "ATTR_draped\nATTR_layer_group_draped markings -5\nTRIS 0 6\n",
                encoding="utf-8",
            )
            (objects_dir / "shed.obj").write_text(
                "I\n800\nOBJ\n\nTEXTURE ../textures/shed.png\n\n"
                "POINT_COUNTS 1 0 0 0\n\nVT 0 0 0 0 1 0 0 0\n\nTRIS 0 0\n",
                encoding="utf-8",
            )

            objects = [
                {"name": "pave", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
                {"name": "shed", "lat": 47.001, "lon": 8.001, "hdg": 45.0, "agl": 0.0},
                {"name": None, "library_path": "lib/airport/lights/beacon.obj",
                 "lat": 47.002, "lon": 8.002, "hdg": 0.0, "agl": 0.0},
            ]
            dsf_dir = nav_dir / "+40+000"
            dsf_dir.mkdir(parents=True)
            dsf_compiler.build_dsf(47, 8, objects, dsf_dir / "+47+008.dsf")

            scene = scenery_viewer.build_scene(pack_dir)
            self.assertEqual(scene["tile_count"], 1)
            self.assertEqual(len(scene["draped"]), 1)
            self.assertEqual(len(scene["rigid"]), 1)
            self.assertEqual(len(scene["library"]), 1)
            self.assertEqual(scene["library"][0]["path"], "lib/airport/lights/beacon.obj")
            self.assertEqual(len(scene["draped"][0]["ring"]), 4)

            # Rendering must not raise on real scene data.
            out_html = pack_dir / "_viewer.html"
            scenery_viewer.render_html(scene, out_html, title="test")
            self.assertTrue(out_html.exists())
            content = out_html.read_text(encoding="utf-8")
            self.assertIn("<svg", content)
            self.assertIn("lib/airport/lights/beacon.obj", content)


if __name__ == "__main__":
    unittest.main()
