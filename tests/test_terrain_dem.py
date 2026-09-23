"""
terrain_dem.py ported verbatim. Formalizes the synthetic-DSF-fixture
verification already built ad hoc during this project's development:
tile-path resolution, DEMS/DEMI/DEMD atom parsing, bilinear elevation
sampling, NODATA propagation, and the real 7z-compression unwrap path.

Caveat carried over unchanged from the original verification: never
tested against a real X-Plane installation (none available in this
environment) -- these tests pin the atom-parsing logic against the
documented DSF spec, not against a real compiled tile.
"""
import array
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from dsf_fixture import build_elevation_dsf, build_multi_layer_elevation_dsf, pack_atom, string_table  # noqa: E402

import terrain_dem


class TestTerrainDem(unittest.TestCase):
    def _write_tile(self, root: Path, lat: int, lon: int, dsf_bytes: bytes):
        band_lat = (lat // 10) * 10
        band_lon = (lon // 10) * 10
        folder = f"{band_lat:+03d}{band_lon:+04d}"
        filename = f"{lat:+03d}{lon:+04d}.dsf"
        dsf_dir = root / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / folder
        dsf_dir.mkdir(parents=True, exist_ok=True)
        path = dsf_dir / filename
        path.write_bytes(dsf_bytes)
        return path

    def test_tile_path_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = self._write_tile(root, 37, -123, build_elevation_dsf([[0, 10], [100, 110]]))
            found = terrain_dem.find_dsf_for_latlon(root, 37.5, -122.8)
            self.assertEqual(found, expected)

    def test_exact_post_values_and_bilinear_center(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = [[0, 10, 20], [100, 110, 120], [200, 210, 220]]
            self._write_tile(root, 47, 8, build_elevation_dsf(grid))
            terrain_dem._dem_cache.clear()

            self.assertEqual(terrain_dem.get_elevation(root, 47.5, 8.5), 110.0)  # exact tile-center post
            self.assertEqual(terrain_dem.get_elevation(root, 47.0, 8.0), 0.0)    # SW corner post

    def test_bilinear_quarter_point(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = [[0, 10, 20], [100, 110, 120], [200, 210, 220]]
            self._write_tile(root, 47, 8, build_elevation_dsf(grid))
            terrain_dem._dem_cache.clear()
            v = terrain_dem.get_elevation(root, 47.25, 8.25)
            self.assertAlmostEqual(v, 55.0, places=6)

    def test_nodata_propagates_through_bilinear_blend(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = [[-32768, 10, 20], [100, 110, 120], [200, 210, 220]]
            self._write_tile(root, 47, 8, build_elevation_dsf(grid))
            terrain_dem._dem_cache.clear()

            self.assertIsNone(terrain_dem.get_elevation(root, 47.0, 8.0))   # exact NODATA post
            self.assertIsNone(terrain_dem.get_elevation(root, 47.1, 8.1))   # blend touching NODATA corner
            self.assertIsNotNone(terrain_dem.get_elevation(root, 47.9, 8.9))  # far corner unaffected

    def test_multiple_layers_packed_in_one_dems_atom_all_decode_correctly(self):
        """Real default-global-scenery tiles pack elevation + sea_level +
        bathymetry as three (IMED, DMED) pairs inside ONE top-level DEMS
        atom, in DEMN name-table order. A prior bug treated each DEMS atom
        as exactly one layer and kept only the LAST pair found inside it,
        so "elevation" silently returned a different (wrong-shaped) raster's
        bytes. Pins that all three layers decode independently and
        correctly, matched to the right (IMED, DMED) pair by encounter
        order, not just "whichever pair happened to be read last"."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            elevation_grid = [[100, 110], [120, 130]]
            sea_level_grid = [[1, 1], [1, 1]]
            bathymetry_grid = [[-5, -6], [-7, -8]]
            dsf_bytes = build_multi_layer_elevation_dsf([
                ("elevation", elevation_grid),
                ("sea_level", sea_level_grid),
                ("bathymetry", bathymetry_grid),
            ])
            self._write_tile(root, 47, 19, dsf_bytes)
            terrain_dem._dem_cache.clear()

            dsf_path = terrain_dem.find_dsf_for_latlon(root, 47.5, 19.5)
            layers = terrain_dem.parse_dem_layers(dsf_path)
            self.assertEqual(set(layers.keys()), {"elevation", "sea_level", "bathymetry"})
            self.assertEqual(layers["elevation"].value_at(0, 0), 100.0)
            self.assertEqual(layers["elevation"].value_at(1, 1), 130.0)
            self.assertEqual(layers["sea_level"].value_at(0, 0), 1.0)
            self.assertEqual(layers["bathymetry"].value_at(0, 0), -5.0)
            self.assertEqual(layers["bathymetry"].value_at(1, 1), -8.0)

    def test_real_default_elevation_flags_value_decodes_as_signed_int_not_float(self):
        """A real installed X-Plane default elevation layer's flags value is
        5 (format bits = 1 "Int" + the unrelated Post bit, verified by hand
        against a real X-Plane 11 install) -- confirms the format is read
        via the 2-bit mask/enum (dsf_Raster_Format_Mask) rather than
        misreading bit 0 as an independent "is float" flag, which would
        reinterpret these very int16 sample bytes as float32 noise."""
        payload = struct.pack("<BBHIIff", 1, 2, 5, 2, 2, 1.0, 0.0)
        pairs = pack_atom(b"IMED", payload) + pack_atom(
            b"DMED", array.array("h", [150, 151, 152, 153]).tobytes())
        dsf_bytes = (b"XPLNEDSF" + struct.pack("<I", 1)
                     + pack_atom(b"NFED", pack_atom(b"NMED", string_table(["elevation"])))
                     + pack_atom(b"SMED", pairs))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_tile(root, 47, 19, dsf_bytes)
            terrain_dem._dem_cache.clear()
            layer = terrain_dem.parse_dem_layers(terrain_dem.find_dsf_for_latlon(root, 47.5, 19.5))["elevation"]
            self.assertFalse(layer.is_float)
            self.assertTrue(layer.is_signed)
            self.assertEqual(layer.value_at(0, 0), 150.0)

    def test_missing_root_and_missing_tile_return_none_no_exception(self):
        self.assertIsNone(terrain_dem.get_elevation(None, 47.5, 8.5))
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(terrain_dem.get_elevation(Path(td), 10.0, 10.0))

    def test_7z_compressed_dsf_round_trip(self):
        if not terrain_dem._HAVE_PY7ZR:
            self.skipTest("py7zr not installed")
        import py7zr

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            plain_bytes = build_elevation_dsf([[1000, 1010], [1100, 1110]])
            inner_name = "+41+011.dsf"
            plain_path = td / inner_name
            plain_path.write_bytes(plain_bytes)

            root = td / "XPlaneRoot"
            dsf_dir = root / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / "+40+010"
            dsf_dir.mkdir(parents=True)
            compressed_path = dsf_dir / inner_name
            with py7zr.SevenZipFile(compressed_path, "w") as archive:
                archive.write(plain_path, arcname=inner_name)

            self.assertEqual(compressed_path.read_bytes()[:6], terrain_dem._SEVEN_ZIP_MAGIC)
            terrain_dem._dem_cache.clear()
            v = terrain_dem.get_elevation(root, 41.5, 11.5)
            self.assertEqual(v, 1055.0)  # average of the 4 posts at exact tile center


if __name__ == "__main__":
    unittest.main()
