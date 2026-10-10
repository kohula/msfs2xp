"""
Ground heights come from the terrain mesh X-Plane draws (terrain_mesh.py),
not the elevation raster it was built from: between mesh points the drawn
ground is a flat triangle, which can sit well off the raster.
"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from dsf_fixture import build_mesh_dsf  # noqa: E402

import terrain_dem
import terrain_mesh

RASTER = -32768.0


def _patch(flags=1, near=0.0, far=-1.0):
    return bytes([18, flags]) + struct.pack("<ff", near, far)


def _tri(*idx):
    return bytes([23, len(idx)]) + struct.pack(f"<{len(idx)}H", *idx)


def _xp(td, data, tile=(47, 8)):
    root = Path(td) / "XPlane"
    lat, lon = tile
    d = root / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / "+40+000"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{lat:+03d}{lon:+04d}.dsf").write_bytes(data)
    return root


# the tile corner square split into two triangles: heights 100, 110, 130, 120
SQUARE = [(8.0, 47.0, 100.0), (9.0, 47.0, 110.0), (9.0, 48.0, 130.0), (8.0, 48.0, 120.0)]


class TestTerrainMesh(unittest.TestCase):
    def tearDown(self):
        terrain_dem._dem_cache.clear()
        terrain_dem._mesh_cache.clear()

    def _elev(self, data, lat, lon):
        with tempfile.TemporaryDirectory() as td:
            return terrain_dem.get_elevation(_xp(td, data), lat, lon)

    def test_point_inside_a_triangle_is_interpolated_on_it(self):
        data = build_mesh_dsf(SQUARE, _patch() + _tri(0, 1, 2, 0, 2, 3))
        # on the edge midpoint between (8,47) 100 and (9,48) 130
        self.assertAlmostEqual(self._elev(data, 47.5, 8.5), 115.0, places=1)
        # inside the first triangle: 100 + 10*0.75 + 20*0.25 (lon 8.75, lat 47.25)
        self.assertAlmostEqual(self._elev(data, 47.25, 8.75), 100 + 10 * 0.75 + 20 * 0.25, places=1)

    def test_mesh_wins_over_the_raster(self):
        grid = [[0, 0], [0, 0]]  # the raster says 0 everywhere
        data = build_mesh_dsf(SQUARE, _patch() + _tri(0, 1, 2, 0, 2, 3), grid=grid)
        self.assertAlmostEqual(self._elev(data, 47.5, 8.5), 115.0, places=1)

    def test_raster_heights_for_marked_points(self):
        pts = [(8.0, 47.0, RASTER), (9.0, 47.0, RASTER), (9.0, 48.0, 130.0)]
        grid = [[50, 70], [50, 70]]  # 50 on the west edge, 70 on the east
        data = build_mesh_dsf(pts, _patch() + _tri(0, 1, 2), grid=grid, elev_range=(-32768.0, 33000.0))
        # halfway along the south edge: between 50 and 70
        self.assertAlmostEqual(self._elev(data, 47.0001, 8.5), 60.0, delta=0.1)

    def test_overlays_and_far_lods_are_not_the_ground(self):
        cmds = (_patch(flags=3) + _tri(0, 1, 2)         # overlay
                + _patch(flags=1, near=2000.0) + _tri(0, 1, 2)  # far LOD
                + _patch() + _tri(0, 2, 3))             # the real ground, other half only
        data = build_mesh_dsf(SQUARE, cmds, grid=[[7, 7], [7, 7]])
        self.assertAlmostEqual(self._elev(data, 47.75, 8.25), 100 + 30 * 0.25 + 20 * 0.5, delta=0.5)
        # no base triangle covers this point: raster fallback
        self.assertAlmostEqual(self._elev(data, 47.25, 8.75), 7.0, places=3)

    def test_strips_fans_and_differenced_pools(self):
        strip = bytes([26, 4]) + struct.pack("<4H", 0, 1, 3, 2)
        data = build_mesh_dsf(SQUARE, _patch() + strip, pool_mode=3)
        self.assertAlmostEqual(self._elev(data, 47.5, 8.5), 115.0, places=1)
        fan = bytes([29, 4]) + struct.pack("<4H", 0, 1, 2, 3)
        data = build_mesh_dsf(SQUARE, _patch() + fan)
        self.assertAlmostEqual(self._elev(data, 47.75, 8.25), 100 + 30 * 0.25 + 20 * 0.5, delta=0.5)

    def test_other_commands_are_skipped(self):
        other = (bytes([1]) + struct.pack("<H", 0) + bytes([3, 0]) + bytes([7]) + struct.pack("<H", 0)
                 + bytes([32, 2]) + b"hi" + bytes([12]) + struct.pack("<HB", 0, 2) + struct.pack("<2H", 0, 1))
        data = build_mesh_dsf(SQUARE, other + _patch() + _tri(0, 1, 2, 0, 2, 3))
        self.assertAlmostEqual(self._elev(data, 47.5, 8.5), 115.0, places=1)

    def test_unknown_command_falls_back_to_the_raster(self):
        data = build_mesh_dsf(SQUARE, bytes([99]) + _patch() + _tri(0, 1, 2), grid=[[5, 5], [5, 5]])
        self.assertAlmostEqual(self._elev(data, 47.5, 8.5), 5.0, places=3)

    def test_parsed_mesh_is_cached_on_disk(self):
        data = build_mesh_dsf(SQUARE, _patch() + _tri(0, 1, 2, 0, 2, 3))
        with tempfile.TemporaryDirectory() as td:
            root = _xp(td, data)
            first = terrain_dem.get_elevation(root, 47.5, 8.5)
            terrain_dem._mesh_cache.clear()
            original = terrain_mesh.parse_mesh
            terrain_mesh.parse_mesh = lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-parsed"))
            try:
                self.assertEqual(terrain_dem.get_elevation(root, 47.5, 8.5), first)
            finally:
                terrain_mesh.parse_mesh = original


if __name__ == "__main__":
    unittest.main()
