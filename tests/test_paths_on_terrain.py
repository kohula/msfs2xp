"""
A looping path (a bus driving round) is raised or lowered onto the X-Plane
ground under each point of the path, per placement.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from dsf_fixture import build_elevation_dsf  # noqa: E402

import pipeline
import terrain_dem

OBJ = """I
800
OBJ

TEXTURE ../textures/bus.png
POINT_COUNTS 3 0 0 3

VT 0 0 0 0 1 0 0 0
VT 1 0 0 0 1 0 0 0
VT 0 1 0 0 1 0 0 0
IDX 0
IDX 1
IDX 2

ANIM_begin
ANIM_trans_begin sim/time/total_running_time_sec
ANIM_trans_key 0.0000 0.00000 0.00000 0.00000
ANIM_trans_key 10.0000 500.00000 0.00000 0.00000
ANIM_keyframe_loop 10.0000
ANIM_trans_end
TRIS 0 3
ANIM_end
"""


class TestPathsOnTerrain(unittest.TestCase):
    def tearDown(self):
        terrain_dem._dem_cache.clear()
        terrain_dem._mesh_cache.clear()

    def test_path_keys_follow_the_ground(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xp = td / "XPlane"
            d = xp / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / "+40+000"
            d.mkdir(parents=True)
            # ground rises 1000 m per degree of longitude eastward
            (d / "+47+008.dsf").write_bytes(build_elevation_dsf([[c * 1000 for c in range(2)] for _ in range(2)]))
            obj_dir = td / "objects"
            obj_dir.mkdir()
            (obj_dir / "bus_Body.obj").write_text(OBJ)
            (obj_dir / "bus_Body.autoplay.json").write_text(json.dumps(
                {"pivot": [0.0, 0.0, 0.0], "keys": [[0.0, 0.0, 0.0, 0.0], [10.0, 500.0, 0.0, 0.0]]}))
            entry = {"name": "bus_Body", "lat": 47.5, "lon": 8.5, "hdg": 0.0, "agl": 0.0}
            tiles = {(47, 8): [entry]}
            self.assertEqual(pipeline._paths_on_terrain(tiles, obj_dir, xp), 1)
            self.assertTrue(entry["name"].startswith("bus_Body_rt_"))
            text = (obj_dir / f"{entry['name']}.obj").read_text()
            keys = [l.split() for l in text.splitlines() if l.startswith("ANIM_trans_key")]
            self.assertAlmostEqual(float(keys[0][3]), 0.0, places=3)
            # 500 m east at lat 47.5 is ~0.00663 deg of longitude: ~6.6 m higher
            self.assertAlmostEqual(float(keys[1][3]), 1000 * 500 / (111320 * 0.6756), delta=0.2)


if __name__ == "__main__":
    unittest.main()
