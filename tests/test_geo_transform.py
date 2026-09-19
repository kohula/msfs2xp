"""
geo_transform.py replaces three independently hand-copied implementations
of the same algebra (main.py's inline placement-offset correction,
terrain_fit.py's local_offset_to_latlon, draped_merge.py's _rotate_xz +
_member_point_to_common_frame). This pins it against the exact numeric
behavior of the OLD (GPU/) implementations, imported directly for
comparison, plus a round-trip property test.
"""
import math
import sys
import unittest
from pathlib import Path

import geo_transform

_GPU_DIR = Path(__file__).resolve().parent.parent.parent / "msfs2xp_v0813_2" / "GPU"


class TestGeoTransform(unittest.TestCase):
    def test_heading_zero_matches_hand_derivation(self):
        lat, lon = geo_transform.local_offset_to_latlon(47.5, 8.5, 0.0, 100.0, -50.0)
        expected_lat = 47.5 - (-50.0 / geo_transform.EARTH_M_PER_DEG)
        expected_lon = 8.5 + (100.0 / (geo_transform.EARTH_M_PER_DEG * math.cos(math.radians(47.5))))
        self.assertAlmostEqual(lat, expected_lat, places=12)
        self.assertAlmostEqual(lon, expected_lon, places=12)

    def test_round_trip_local_to_latlon_and_back(self):
        base_lat, base_lon = 47.123, 8.456
        for heading in (0.0, 30.0, 90.0, 180.0, 270.0, 359.0):
            for local_x, local_z in [(0.0, 0.0), (15.0, -20.0), (-100.0, 200.0), (3.3, 7.7)]:
                lat, lon = geo_transform.local_offset_to_latlon(base_lat, base_lon, heading, local_x, local_z)
                recovered_x, recovered_z = geo_transform.latlon_offset_to_local(base_lat, base_lon, heading, lat, lon)
                self.assertAlmostEqual(recovered_x, local_x, places=6, msg=f"heading={heading}")
                self.assertAlmostEqual(recovered_z, local_z, places=6, msg=f"heading={heading}")

    def test_zero_offset_returns_base_point(self):
        lat, lon = geo_transform.local_offset_to_latlon(47.0, 8.0, 123.0, 0.0, 0.0)
        self.assertAlmostEqual(lat, 47.0, places=12)
        self.assertAlmostEqual(lon, 8.0, places=12)

    def test_metres_per_degree_matches_wgs84(self):
        # Equator: 1 deg lat ~ 110.574 km (meridian), 1 deg lon ~ 111.319 km.
        m_lat0, m_lon0 = geo_transform.metres_per_degree(0.0)
        self.assertAlmostEqual(m_lat0, 110574.3, delta=1.0)
        self.assertAlmostEqual(m_lon0, 111319.5, delta=1.0)
        # Longitude shrinks ~cos(lat) (plus a small ellipsoid term, N growing
        # toward the pole); latitude grows slightly toward the pole.
        m_lat60, m_lon60 = geo_transform.metres_per_degree(60.0)
        self.assertAlmostEqual(m_lon60 / m_lon0, math.cos(math.radians(60.0)), delta=3e-3)
        self.assertGreater(m_lat60, m_lat0)
        # The flat EARTH_M_PER_DEG constant is ~0.35% short E-W at 47N -- the
        # gap this function exists to close for the draped-layer merge.
        _, m_lon47 = geo_transform.metres_per_degree(47.43)
        flat_lon47 = geo_transform.EARTH_M_PER_DEG * math.cos(math.radians(47.43))
        self.assertGreater((m_lon47 - flat_lon47) / m_lon47, 0.002)
        self.assertLess((m_lon47 - flat_lon47) / m_lon47, 0.006)

    def test_matches_old_terrain_fit_implementation(self):
        """Bit-identical to the OLD (GPU/) terrain_fit.local_offset_to_latlon
        -- confirms the port didn't silently change the already-verified math."""
        if not _GPU_DIR.is_dir():
            self.skipTest(f"old GPU/ tree not found at {_GPU_DIR} -- skipping cross-check")
        sys.path.insert(0, str(_GPU_DIR))
        import terrain_fit as old_terrain_fit  # noqa: E402

        cases = [
            (47.5, 8.5, 0.0, 15.0, -10.0),
            (47.5, 8.5, 45.0, 15.0, -10.0),
            (0.5, -122.3, 200.0, -33.0, 44.0),
        ]
        for base_lat, base_lon, heading, x, z in cases:
            new_result = geo_transform.local_offset_to_latlon(base_lat, base_lon, heading, x, z)
            old_result = old_terrain_fit.local_offset_to_latlon(base_lat, base_lon, heading, x, z)
            self.assertEqual(new_result, old_result, f"mismatch for case {(base_lat, base_lon, heading, x, z)}")

    def test_matches_old_draped_merge_two_stage_usage(self):
        """draped_merge.py's own usage pattern is two calls: rotate into a
        member's real lat/lon via its own heading, then re-express relative
        to a shared origin at heading 0. Confirms the new shared functions
        reproduce the OLD hand-inlined version's numeric result exactly for
        this exact call shape."""
        old_path = _GPU_DIR / "draped_merge.py"
        if not old_path.is_file():
            self.skipTest(f"old GPU/draped_merge.py not found at {old_path} -- skipping cross-check")

        # Loaded by explicit file path under a distinct module name (NOT
        # "import draped_merge"): this project's own draped_merge.py
        # shares that exact module name, and since Python caches imports
        # in sys.modules by name, a plain "import draped_merge" here would
        # silently return whichever one -- new or old -- some OTHER test
        # file already imported first, not necessarily this one.
        import importlib.util
        spec = importlib.util.spec_from_file_location("old_draped_merge_module", old_path)
        old_draped_merge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_draped_merge)

        member_lat, member_lon, heading = 47.5, 8.5, 37.0
        origin_lat, origin_lon = 47.0, 8.0
        local_x, local_z = 12.0, -6.0

        old_x, old_z = old_draped_merge._member_point_to_common_frame(
            local_x, local_z, member_lat, member_lon, heading, origin_lat, origin_lon)

        real_lat, real_lon = geo_transform.local_offset_to_latlon(member_lat, member_lon, heading, local_x, local_z)
        new_x, new_z = geo_transform.latlon_offset_to_local(origin_lat, origin_lon, 0.0, real_lat, real_lon)

        self.assertAlmostEqual(new_x, old_x, places=9)
        self.assertAlmostEqual(new_z, old_z, places=9)


if __name__ == "__main__":
    unittest.main()
