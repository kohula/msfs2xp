"""
main._built_up_exclusion_rects -- the roads/rail (sim/exclude_net /
sim/exclude_str) exclusion is a set of rectangles fitted to where the
conversion actually placed objects, not one big fixed radius (user: the
old 4.5km box was "a bit too big"). These check it covers every
placement, stays local, falls back cleanly, and can't explode the prop
count.
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import main
from mesh_convert import mesh_ir


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


class TestConvexHull2D(unittest.TestCase):
    def test_square_hull_has_4_vertices(self):
        pts = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [2.0, 2.0]])
        hull = main._convex_hull_2d(pts)
        self.assertEqual(len(hull), 4)

    def test_collinear_points_degenerate_gracefully(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        hull = main._convex_hull_2d(pts)
        self.assertLessEqual(len(hull), 2)

    def test_single_point(self):
        hull = main._convex_hull_2d(np.array([[5.0, 5.0]]))
        self.assertEqual(len(hull), 1)


class TestPerObjectExclusionRects(unittest.TestCase):
    """main._per_object_exclusion_rects -- explicit user instruction: default
    scenery exclusion should cover each converted object's OWN footprint
    using close to the MINIMUM real area needed (1-3m growth), not one
    shared shape covering the whole airport's combined extent, not a
    single bounding box per object (overshoots badly on an elongated/
    angled/irregular footprint), and not even one rectangle per convex-
    hull edge (a diagonal edge's own local axis-aligned box, or the hull
    itself, both over-cover a real concavity -- e.g. the gap between an
    X-shaped building's two arms). A solid, simply-convex object collapses
    to ONE minimal rectangle; a real footprint with two genuinely
    disconnected parts (see TestPerObjectExclusionRectsConcave below)
    splits into separate rectangles that leave the gap between them
    uncovered."""

    def _make_sidecar(self, obj_dir, stem, x_range, z_range):
        ir = mesh_ir.MeshIR(
            name=stem,
            positions=np.array([
                [x_range[0], 0.0, z_range[0]],
                [x_range[1], 0.0, z_range[0]],
                [x_range[1], 0.0, z_range[1]],
                [x_range[0], 0.0, z_range[1]],
            ], dtype=np.float64),
        )
        mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{stem}.obj"))

    def test_square_object_gets_one_minimal_rect_padded_by_2m(self):
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            self._make_sidecar(obj_dir, "Building_A", x_range=(-5.0, 5.0), z_range=(-5.0, 5.0))
            rects = main._per_object_exclusion_rects(
                obj_dir, [(["Building_A"], 47.0, 19.0, 0.0)])
            # A solid square needs only ONE minimal rectangle -- not one
            # per hull edge.
            self.assertEqual(len(rects), 1)
            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * math.cos(math.radians(47.0))
            # Every point on the square's own boundary must be covered.
            self.assertTrue(_covers(rects, 47.0 + 5.0 / m_per_deg_lat, 19.0))
            self.assertTrue(_covers(rects, 47.0, 19.0 + 5.0 / m_per_deg_lon))
            # 2m padding: a point 6.9m out (within the 5+2=7m reach) is
            # covered; a point 7.5m out (beyond it) is not.
            self.assertTrue(_covers(rects, 47.0 + 6.9 / m_per_deg_lat, 19.0))
            self.assertFalse(_covers(rects, 47.0 + 7.5 / m_per_deg_lat, 19.0))

    def test_two_distant_objects_stay_strictly_their_own_area(self):
        """The whole point vs. the old airport-wide shape: the empty gap
        between two well-separated objects must NOT be covered by either
        rect, or by any single combined one."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            self._make_sidecar(obj_dir, "Building_A", x_range=(-2.0, 2.0), z_range=(-2.0, 2.0))
            self._make_sidecar(obj_dir, "Building_B", x_range=(-2.0, 2.0), z_range=(-2.0, 2.0))
            rects = main._per_object_exclusion_rects(obj_dir, [
                (["Building_A"], 47.0000, 19.0000, 0.0),
                (["Building_B"], 47.0100, 19.0000, 0.0),  # ~1.1km north
            ])
            self.assertTrue(_covers(rects, 47.0000, 19.0000))
            self.assertTrue(_covers(rects, 47.0100, 19.0000))
            # Roughly halfway between the two objects -- well outside either
            # object's own small padded footprint.
            self.assertFalse(_covers(rects, 47.0050, 19.0000))

    def test_heading_rotates_the_footprint(self):
        """A footprint that's wide in X and narrow in Z, placed at
        heading=90, must come out wide in the north/south direction and
        narrow east/west -- confirms the same local_offset_to_latlon
        heading convention used everywhere else in the pipeline is applied
        here too, not a naive axis-aligned copy of local X/Z."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            self._make_sidecar(obj_dir, "LongBuilding", x_range=(-20.0, 20.0), z_range=(-2.0, 2.0))
            rects = main._per_object_exclusion_rects(
                obj_dir, [(["LongBuilding"], 47.0, 19.0, 90.0)])
            self.assertTrue(rects)
            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * math.cos(math.radians(47.0))
            lat_span_m = (max(r["north"] for r in rects) - min(r["south"] for r in rects)) * m_per_deg_lat
            lon_span_m = (max(r["east"] for r in rects) - min(r["west"] for r in rects)) * m_per_deg_lon
            self.assertGreater(lat_span_m, lon_span_m,
                                "a 40m-long object at heading 90 must span more north/south than east/west")

    def test_multiple_stems_for_one_placement_combine_into_one_hull(self):
        """generated_stems (multiple .obj siblings from one original model,
        e.g. per-material split) must combine into ONE footprint outline
        (one convex hull covering their union), not be treated as two
        separate, independently-hulled objects."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            self._make_sidecar(obj_dir, "Roof_Mat", x_range=(-5.0, 5.0), z_range=(-5.0, 5.0))
            self._make_sidecar(obj_dir, "Wall_Mat", x_range=(-3.0, 8.0), z_range=(-3.0, 3.0))
            rects = main._per_object_exclusion_rects(
                obj_dir, [(["Roof_Mat", "Wall_Mat"], 47.0, 19.0, 0.0)])
            self.assertTrue(rects)
            # The combined footprint's own far corner (8, 3, from Wall_Mat)
            # must be covered -- proves both stems fed the SAME hull.
            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * math.cos(math.radians(47.0))
            self.assertTrue(_covers(rects, 47.0 + 3.0 / m_per_deg_lat, 19.0 + 8.0 / m_per_deg_lon))

    def test_missing_sidecar_is_skipped_not_crashed(self):
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            rects = main._per_object_exclusion_rects(
                obj_dir, [(["NoSuchStem"], 47.0, 19.0, 0.0)])
            self.assertEqual(rects, [])

    def test_empty_candidates_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(main._per_object_exclusion_rects(Path(td), []), [])


class TestPerObjectExclusionRectsConcave(unittest.TestCase):
    """The actual motivating case: a real object whose true footprint is
    concave or has genuinely disconnected parts (an X/dumbbell shape) must
    not get the gap between its parts excluded just because a convex hull
    of the combined points would cover it. Uses real triangle indices (not
    _make_sidecar's corners-only fixture) so rasterization sees the TRUE
    footprint, not a hull fallback."""

    def _make_triangulated_sidecar(self, obj_dir, stem, quads):
        """quads: [(x0, x1, z0, z1), ...] -- each rectangle triangulated
        into 2 real triangles, all combined into one sidecar, so one stem
        can carry a real, possibly-disconnected multi-part footprint."""
        positions, indices = [], []
        for (x0, x1, z0, z1) in quads:
            base = len(positions)
            positions.extend([[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]])
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        ir = mesh_ir.MeshIR(
            name=stem,
            positions=np.array(positions, dtype=np.float64),
            indices=np.array(indices, dtype=np.int64),
        )
        mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{stem}.obj"))

    def test_two_disconnected_parts_of_one_object_stay_separate(self):
        """A real dumbbell-shaped footprint (two small real quads far
        apart, connected only by an unrelated hull) must NOT get its
        middle gap excluded -- the whole reason for rasterizing real
        triangles instead of a convex hull."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            self._make_triangulated_sidecar(obj_dir, "Dumbbell", quads=[
                (-10.0, -7.0, -1.0, 1.0),   # left blob
                (7.0, 10.0, -1.0, 1.0),     # right blob, far away
            ])
            rects = main._per_object_exclusion_rects(
                obj_dir, [(["Dumbbell"], 47.0, 19.0, 0.0)], pad_m=1.0)
            self.assertTrue(rects)
            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * math.cos(math.radians(47.0))
            self.assertTrue(_covers(rects, 47.0, 19.0 - 8.5 / m_per_deg_lon), "left blob center must be covered")
            self.assertTrue(_covers(rects, 47.0, 19.0 + 8.5 / m_per_deg_lon), "right blob center must be covered")
            # The middle gap -- well outside pad_m=1 of either real quad --
            # must NOT be excluded. A convex hull of the combined points
            # would wrongly cover this whole region.
            self.assertFalse(_covers(rects, 47.0, 19.0), "the empty gap between the two parts must stay open")

    def test_solid_l_shape_does_not_fill_its_own_concave_notch(self):
        """An L-shaped footprint (two overlapping real quads forming an L,
        not a full rectangle) must not have its missing corner excluded --
        that corner is real empty space the object never occupies."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            self._make_triangulated_sidecar(obj_dir, "LShape", quads=[
                (0.0, 10.0, 0.0, 3.0),   # long horizontal arm
                (0.0, 3.0, 0.0, 10.0),   # long vertical arm (shares the corner)
            ])
            rects = main._per_object_exclusion_rects(
                obj_dir, [(["LShape"], 47.0, 19.0, 0.0)], pad_m=0.5)
            self.assertTrue(rects)
            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * math.cos(math.radians(47.0))
            # geo_transform.local_offset_to_latlon: lat = base_lat -
            # rot_z/EARTH_M_PER_DEG -- increasing local Z moves SOUTH, so
            # a point at local (x, z) lands at (base_lat - z/m, base_lon +
            # x/m), not +z.
            # The L's own missing corner (far from both arms, well beyond
            # the 0.5m pad) must stay open.
            self.assertFalse(_covers(rects, 47.0 - 8.0 / m_per_deg_lat, 19.0 + 8.0 / m_per_deg_lon))
            # Both arms themselves must be covered.
            self.assertTrue(_covers(rects, 47.0 - 1.0 / m_per_deg_lat, 19.0 + 8.0 / m_per_deg_lon))
            self.assertTrue(_covers(rects, 47.0 - 8.0 / m_per_deg_lat, 19.0 + 1.0 / m_per_deg_lon))


if __name__ == "__main__":
    unittest.main()
