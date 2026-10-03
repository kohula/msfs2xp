"""
An airport written with "1302 flatten 1" is levelled by X-Plane inside its
boundary, so terrain_dem reports that level there instead of the raw
elevation raster, and terrain_fit then leaves models inside it alone --
they were built for flat ground and now stand on flat ground.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import terrain_dem
import terrain_fit

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from dsf_fixture import build_elevation_dsf  # noqa: E402
import test_terrain_fit  # noqa: E402  (module, not the class: unittest would re-run it here)

import mesh_convert


def _write_bumpy_terrain(root, tile_lat, tile_lon, bump_scale):
    n, mid = 5, 2
    grid = [[int(((r - mid) ** 2 + (c - mid) ** 2) * bump_scale) for c in range(n)] for r in range(n)]
    folder = f"{(tile_lat // 10) * 10:+03d}{(tile_lon // 10) * 10:+04d}"
    dsf_dir = root / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / folder
    dsf_dir.mkdir(parents=True, exist_ok=True)
    (dsf_dir / f"{tile_lat:+03d}{tile_lon:+04d}.dsf").write_bytes(build_elevation_dsf(grid))


SQUARE = [(47.4, 8.4), (47.4, 8.6), (47.6, 8.6), (47.6, 8.4)]


class TestFlatZones(unittest.TestCase):
    def tearDown(self):
        terrain_dem.set_flat_zones([])
        terrain_fit._group_cache.clear()
        terrain_fit._group_transform_cache.clear()

    def test_inside_reports_the_zone_level_outside_the_raster(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_bumpy_terrain(root, 47, 8, bump_scale=1500.0)
            raw_inside = terrain_dem.get_elevation(root, 47.45, 8.45)
            raw_outside = terrain_dem.get_elevation(root, 47.9, 8.9)
            terrain_dem.set_flat_zones([(SQUARE, 12.5)])
            self.assertEqual(terrain_dem.get_elevation(root, 47.45, 8.45), 12.5)
            self.assertEqual(terrain_dem.get_elevation(root, 47.5, 8.5), 12.5)
            self.assertNotEqual(raw_inside, 12.5)
            self.assertEqual(terrain_dem.get_elevation(root, 47.9, 8.9), raw_outside)
            # other raster layers are not levelled
            self.assertIsNone(terrain_dem.flat_elevation(47.9, 8.9))
            self.assertEqual(terrain_dem.flat_zones(), [(SQUARE, 12.5)])
            terrain_dem.set_flat_zones([])
            self.assertEqual(terrain_dem.get_elevation(root, 47.45, 8.45), raw_inside)

    def test_no_install_still_means_no_elevation(self):
        terrain_dem.set_flat_zones([(SQUARE, 12.5)])
        self.assertIsNone(terrain_dem.get_elevation(None, 47.5, 8.5))

    def test_a_building_inside_a_flattened_airport_is_not_fitted(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            _write_bumpy_terrain(xplane_root, 47, 8, bump_scale=1500.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()
            glb = td / "building.glb"
            test_terrain_fit.TestTerrainFit._build_box_glb(None, glb, "BigBuilding", "BuildingTex", half_size=15.0)
            stem = mesh_convert.convert(glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")[0].stem

            # the same building on the same bump IS fitted without the zone
            # (test_terrain_fit.test_large_building_on_steep_terrain_gets_the_precise_warp)
            terrain_dem.set_flat_zones([(SQUARE, 30.0)])
            result_stem, applied, reason = terrain_fit.get_or_create_fitted_group(
                obj_dir, [stem], 47.5, 8.5, 0.0, xplane_root)[stem]
            self.assertFalse(applied)
            self.assertEqual(reason, "negligible")
            self.assertEqual(result_stem, stem)


if __name__ == "__main__":
    unittest.main()
