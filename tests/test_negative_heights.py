"""
The per-placement height report (msfs2xp_placements.csv).
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pipeline
from mesh_convert import mesh_ir


def _tile(name, bottom=-1.2):
    # a 20 m tile: surface at y=0, a channel down to `bottom`
    pos = np.array([[-10, 0.0, -10], [10, 0.0, -10], [10, 0.0, 10], [-10, 0.0, 10],
                    [-1, bottom, -10], [1, bottom, -10], [1, bottom, 10], [-1, bottom, 10]], dtype=float)
    return mesh_ir.MeshIR(name=name, positions=pos, normals=np.tile([0.0, 1.0, 0.0], (8, 1)),
                          uvs=np.zeros((8, 2)), indices=np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]))


def _cand(**entries):
    return {"stem_entries": {k: (v[0], False, v[1], "negligible") for k, v in entries.items()}}


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
