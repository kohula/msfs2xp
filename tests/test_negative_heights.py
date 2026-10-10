"""
Geometry below an object's zero point (a drain channel, a quay wall into
the water) is given as a negative AGL height on every part of the
placement; the geometry itself stays as modelled.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline
import terrain_fit
from mesh_convert import mesh_ir


def _tile(name, bottom=-1.2):
    # a 20 m tile: surface at y=0, a channel down to `bottom`
    pos = np.array([[-10, 0.0, -10], [10, 0.0, -10], [10, 0.0, 10], [-10, 0.0, 10],
                    [-1, bottom, -10], [1, bottom, -10], [1, bottom, 10], [-1, bottom, 10]], dtype=float)
    return mesh_ir.MeshIR(name=name, positions=pos, normals=np.tile([0.0, 1.0, 0.0], (8, 1)),
                          uvs=np.zeros((8, 2)), indices=np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]))


def _cand(**entries):
    return {"stem_entries": {k: (v[0], False, v[1], "negligible") for k, v in entries.items()}}


class TestSinkBelowZero(unittest.TestCase):
    def tearDown(self):
        terrain_fit._ir_cache.clear()

    def test_depth_becomes_negative_agl_on_every_part(self):
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            mesh_ir.save(_tile("drain"), mesh_ir.sidecar_path_for(obj_dir / "drain.obj"))
            mesh_ir.save(_tile("grate", bottom=-0.3), mesh_ir.sidecar_path_for(obj_dir / "grate.obj"))
            drain, grate = {"name": "drain", "agl": 0.0}, {"name": "grate", "agl": 0.5}
            c = _cand(drain=(drain, False), grate=(grate, False))
            self.assertEqual(pipeline._sink_below_zero([c], obj_dir), 1)
            self.assertAlmostEqual(drain["agl"], -1.2)
            self.assertAlmostEqual(grate["agl"], 0.5 - 1.2)
            self.assertEqual(drain["name"], "drain")  # geometry untouched
            self.assertAlmostEqual(float(mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / "drain.obj"))
                                         .positions[:, 1].min()), -1.2)

    def test_objects_at_or_above_zero_and_draped_parts_stay(self):
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            mesh_ir.save(_tile("slab", bottom=0.0), mesh_ir.sidecar_path_for(obj_dir / "slab.obj"))
            mesh_ir.save(_tile("decal"), mesh_ir.sidecar_path_for(obj_dir / "decal.obj"))
            slab, decal = {"name": "slab", "agl": 0.0}, {"name": "decal", "agl": 0.0}
            self.assertEqual(pipeline._sink_below_zero([_cand(slab=(slab, False), decal=(decal, True))],
                                                       obj_dir), 0)
            self.assertEqual((slab["agl"], decal["agl"]), (0.0, 0.0))


class TestPlacementReport(unittest.TestCase):
    def test_one_row_per_placement_with_each_height_step(self):
        import csv
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            mesh_ir.save(_tile("drain"), mesh_ir.sidecar_path_for(obj_dir / "drain.obj"))
            kept = {"name": "drain", "agl": -1.2}
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
            self.assertIn("drain@-1.20", r["final_parts"])
            self.assertIn("cover=removed", r["final_parts"])


if __name__ == "__main__":
    unittest.main()
