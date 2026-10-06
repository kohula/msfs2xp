"""
X-Plane doesn't sink an object below the terrain from a negative AGL
height: a drain tile whose channel sits under the surface stood on the
ground instead, its below-ground part showing above the pavement. The
pipeline bakes the drop into a copy of the geometry placed at height 0.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline
from mesh_convert import mesh_ir


def _tile(name):
    # a 20 m tile, lifted by convert() so its 1.2 m deep channel bottom is y=0
    pos = np.array([[-10, 1.2, -10], [10, 1.2, -10], [10, 1.2, 10], [-10, 1.2, 10],
                    [-1, 0.0, -10], [1, 0.0, -10], [1, 0.0, 10], [-1, 0.0, 10]], dtype=float)
    light = mesh_ir.LightEntry(pos=(0.0, 1.5, 0.0), dir=(0, -1, 0), color=(1, 1, 1), cone_angle=90,
                               size=1.0, dataref=None)
    return mesh_ir.MeshIR(name=name, positions=pos, normals=np.tile([0.0, 1.0, 0.0], (8, 1)),
                          uvs=np.zeros((8, 2)), indices=np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]),
                          lights=[light])


class TestBakeNegativeHeights(unittest.TestCase):
    def test_negative_height_is_baked_into_the_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            mesh_ir.save(_tile("drain"), mesh_ir.sidecar_path_for(obj_dir / "drain.obj"))
            entry = {"name": "drain", "lat": 51.5, "lon": 0.05, "hdg": 0.0, "agl": -1.2}
            tiles = {(51, 0): [entry]}
            self.assertEqual(pipeline._bake_negative_heights(tiles, obj_dir), (1, 0))
            self.assertEqual(entry["agl"], 0.0)
            self.assertEqual(entry["name"], "drain_dn120")
            lowered = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / "drain_dn120.obj"))
            # the surface is back at ground level, the channel under it
            self.assertAlmostEqual(float(lowered.positions[:, 1].max()), 0.0, places=6)
            self.assertAlmostEqual(float(lowered.positions[:, 1].min()), -1.2, places=6)
            self.assertAlmostEqual(lowered.lights[0].pos[1], 0.3, places=6)
            self.assertTrue((obj_dir / "drain_dn120.obj").exists())

    def test_one_copy_per_model_and_depth(self):
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            mesh_ir.save(_tile("drain"), mesh_ir.sidecar_path_for(obj_dir / "drain.obj"))
            a = {"name": "drain", "agl": -1.2}
            b = {"name": "drain", "agl": -1.2}
            c = {"name": "drain", "agl": -0.5}
            self.assertEqual(pipeline._bake_negative_heights({(51, 0): [a, b, c]}, obj_dir), (3, 0))
            self.assertEqual(a["name"], b["name"])
            self.assertEqual(c["name"], "drain_dn50")

    def test_ground_level_raised_draped_and_animated_are_left_alone(self):
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            mesh_ir.save(_tile("drain"), mesh_ir.sidecar_path_for(obj_dir / "drain.obj"))
            decal = _tile("decal")
            decal.draped = True
            mesh_ir.save(decal, mesh_ir.sidecar_path_for(obj_dir / "decal.obj"))
            ground = {"name": "drain", "agl": 0.0}
            raised = {"name": "drain", "agl": 3.0}
            draped = {"name": "decal", "agl": -1.0}
            animated = {"name": "door_anim", "agl": -0.8}  # no sidecar
            library = {"name": None, "library_path": "lib/x.obj", "agl": -1.0}
            tiles = {(51, 0): [ground, raised, draped, animated, library]}
            self.assertEqual(pipeline._bake_negative_heights(tiles, obj_dir), (0, 2))
            self.assertEqual([o["name"] for o in tiles[(51, 0)]], ["drain", "drain", "decal", "door_anim", None])
            self.assertEqual(animated["agl"], -0.8)


if __name__ == "__main__":
    unittest.main()


class TestPlacementReport(unittest.TestCase):
    def test_one_row_per_placement_with_each_height_step(self):
        import csv
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            mesh_ir.save(_tile("drain"), mesh_ir.sidecar_path_for(obj_dir / "drain.obj"))
            kept = {"name": "drain_dn120", "agl": 0.0}
            gone = {"name": "cover", "agl": 0.0}
            cands = [{
                "model": "DrainTile", "title": "Drain tile", "source": "SceneryObject",
                "abs_lat": 51.5, "abs_lon": 0.05, "hdg": 90.0, "alt": 4.6, "is_agl": False,
                "height_offset": -1.2, "mid_y": 0.0, "agl": -1.2,
                "stem_entries": {"drain": (kept, False, False, "negligible"),
                                 "cover": (gone, False, False, "x")},
            }]
            path = obj_dir / "report.csv"
            pipeline._write_placement_report(path, cands, obj_dir, {(51, 0): [kept]}, 5.8)
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual(r["model"], "DrainTile")
            self.assertEqual(r["msfs_alt_is_agl"], "no (MSL)")
            self.assertEqual(r["airport_alt_m"], "5.800")
            self.assertEqual(r["height_above_ground_m"], "-1.200")
            self.assertEqual(r["model_height_m"], "1.20")
            self.assertIn("drain_dn120@+0.00", r["final_parts"])
            self.assertIn("cover=removed", r["final_parts"])
