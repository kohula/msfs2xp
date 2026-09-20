"""
main._built_up_exclusion_rects -- the roads/rail (sim/exclude_net /
sim/exclude_str) exclusion is a set of rectangles fitted to where the
conversion actually placed objects, not one big fixed radius (user: the
old 4.5km box was "a bit too big"). These check it covers every
placement, stays local, falls back cleanly, and can't explode the prop
count.
"""
import math
import unittest

import main


def _covers(rects, lat, lon):
    return any(r["west"] <= lon <= r["east"] and r["south"] <= lat <= r["north"] for r in rects)


class TestBuiltUpExclusionRects(unittest.TestCase):
    def _cluster(self, lat0, lon0, n=12, spread_m=400.0):
        m = 111320.0
        pts = []
        for i in range(n):
            for j in range(n):
                pts.append((lat0 + (i / n) * spread_m / m,
                            lon0 + (j / n) * spread_m / (m * math.cos(math.radians(lat0)))))
        return [p[0] for p in pts], [p[1] for p in pts]

    def test_every_placement_falls_inside_some_rect(self):
        lats, lons = self._cluster(47.43, 19.26)
        rects = main._built_up_exclusion_rects(lats, lons)
        self.assertTrue(rects)
        for la, lo in zip(lats, lons):
            self.assertTrue(_covers(rects, la, lo), f"{la},{lo} not covered")

    def test_stays_local_does_not_reach_far_field(self):
        """A point ~3 km away from the airport cluster must NOT be inside
        any rectangle -- that's the whole point vs. the old radius box."""
        lats, lons = self._cluster(47.43, 19.26)
        rects = main._built_up_exclusion_rects(lats, lons)
        far_lat = 47.43 + 3000.0 / 111320.0
        self.assertFalse(_covers(rects, far_lat, 19.26))

    def test_too_few_points_returns_empty_for_fallback(self):
        self.assertEqual(main._built_up_exclusion_rects([47.4, 47.4], [19.2, 19.2]), [])

    def test_stray_far_coordinate_bails_to_fallback(self):
        lats, lons = self._cluster(47.43, 19.26)
        lats.append(0.0)       # ~5000 km away -> grid would be enormous
        lons.append(19.26)
        self.assertEqual(main._built_up_exclusion_rects(lats, lons), [])

    def test_rect_count_stays_bounded(self):
        # a big sparse checkerboard would fragment badly at 200 m; the
        # coarsening retry must keep it under the cap or return [].
        m = 111320.0
        lats, lons = [], []
        for i in range(0, 40, 3):
            for j in range(0, 40, 3):
                lats.append(47.4 + i * 250.0 / m)
                lons.append(19.2 + j * 250.0 / (m * math.cos(math.radians(47.4))))
        rects = main._built_up_exclusion_rects(lats, lons, max_rects=80)
        self.assertLessEqual(len(rects), 80)


class TestPolygonInteriorExclusionRects(unittest.TestCase):
    """main._polygon_interior_exclusion_rects -- the shape-aware
    replacement for the old single-combined-bbox exclusion (CONFIRMED REAL
    BUG: unioning a real airport boundary/placement extent with an
    always-present ~5km-wide fixed-radius fallback ballooned the exclusion
    far past the airport, wiping default scenery over a huge area of
    unrelated surrounding city -- confirmed on a real EGLC conversion).
    Rasterizes the REAL boundary ring's own shape instead of just its
    bounding box, so a long thin runway-shaped ring doesn't drag a huge
    square along with it."""

    def _rect_ring(self, lat0, lon0, length_m, width_m):
        """A simple rectangular boundary ring (4 corners), long axis
        north-south -- stands in for a runway-shaped real boundary."""
        m = 111320.0
        dlat = (length_m / 2.0) / m
        dlon = (width_m / 2.0) / (m * math.cos(math.radians(lat0)))
        return [
            (lat0 - dlat, lon0 - dlon), (lat0 - dlat, lon0 + dlon),
            (lat0 + dlat, lon0 + dlon), (lat0 + dlat, lon0 - dlon),
        ]

    def test_covers_points_well_inside_the_ring(self):
        ring = self._rect_ring(47.43, 19.26, length_m=1800.0, width_m=200.0)
        rects = main._polygon_interior_exclusion_rects(ring)
        self.assertTrue(rects)
        self.assertTrue(_covers(rects, 47.43, 19.26), "the ring's own center must be covered")

    def test_elongated_runway_shaped_ring_does_not_balloon_to_a_wide_square(self):
        """The confirmed real failure mode: a long, THIN ring (a runway)
        must stay thin in the exclusion too -- not get treated as if its
        own bounding SQUARE (as wide as it is long) were the real shape."""
        ring = self._rect_ring(47.43, 19.26, length_m=1800.0, width_m=200.0)
        rects = main._polygon_interior_exclusion_rects(ring)
        self.assertTrue(rects)
        m = 111320.0
        # A point offset sideways (east-west, the SHORT axis) by 300m --
        # outside the 200m-wide ring plus a reasonable margin -- must NOT
        # be covered. The ring's bounding box alone (900m x 200m) would
        # already exclude this correctly, but a regression back to a
        # squared-off union would not.
        far_lon = 19.26 + 300.0 / (m * math.cos(math.radians(47.43)))
        self.assertFalse(_covers(rects, 47.43, far_lon))

    def test_stays_local_does_not_reach_far_field(self):
        ring = self._rect_ring(47.43, 19.26, length_m=1800.0, width_m=200.0)
        rects = main._polygon_interior_exclusion_rects(ring)
        far_lat = 47.43 + 3000.0 / 111320.0
        self.assertFalse(_covers(rects, far_lat, 19.26))

    def test_degenerate_ring_returns_empty(self):
        self.assertEqual(main._polygon_interior_exclusion_rects([]), [])
        self.assertEqual(main._polygon_interior_exclusion_rects([(47.4, 19.2), (47.4, 19.3)]), [])


class TestPointsInPolygon(unittest.TestCase):
    def test_classifies_inside_and_outside_a_square(self):
        import numpy as np
        poly_x = np.array([0.0, 10.0, 10.0, 0.0])
        poly_y = np.array([0.0, 0.0, 10.0, 10.0])
        px = np.array([5.0, 15.0, -5.0])
        py = np.array([5.0, 5.0, 5.0])
        result = main._points_in_polygon(px, py, poly_x, poly_y)
        self.assertEqual(list(result), [True, False, False])


class TestGreedyRectsFromMask(unittest.TestCase):
    def test_solid_block_is_one_rect(self):
        import numpy as np
        mask = np.zeros((5, 6), dtype=bool)
        mask[1:4, 1:5] = True
        rects = main._greedy_rects_from_mask(mask)
        self.assertEqual(rects, [(1, 1, 3, 4)])

    def test_disjoint_blobs_each_covered_once(self):
        import numpy as np
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:2, 0:2] = True
        mask[4:6, 4:6] = True
        rects = main._greedy_rects_from_mask(mask)
        covered = np.zeros_like(mask)
        for (y0, x0, y1, x1) in rects:
            covered[y0:y1 + 1, x0:x1 + 1] = True
        self.assertTrue((covered == mask).all())


if __name__ == "__main__":
    unittest.main()
