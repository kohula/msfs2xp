"""
convert() re-centers every model around its own XZ footprint center, and
re-zeroes it so its lowest vertex sits at Y=0 regardless of which
direction it started on -- both explicitly requested so TILTED's rotation
pivot (see convert.py's own file_apply_tilted) sits at the object's true
geometric center instead of wherever the source export's arbitrary local
origin happened to be, and so no object's own local geometry is ever
authored with its base above OR below its own local Y=0. The removed
offset is written to a "<model_stem>.originoffset.json" sidecar for
main.py to fold into the DSF placement instead (x/z into the placement's
lat/lon via the same rotate-by-heading algebra geo_transform.py already
centralizes, y into the placement's AGL offset).

The Y case used to only handle geometry dipping BELOW zero (lifting it
up) -- a positive min_y (the mesh's own lowest point already sitting
above its local origin, e.g. a multi-figure "group" character prop whose
shared pivot was authored at the formation's own center rather than any
one figure's feet) was left completely untouched, baking that arbitrary
positive baseline permanently into the exported mesh with zero
placement-time compensation. Confirmed real-world consequence: visually
similar character props floating at different, seemingly arbitrary
heights above the floor purely because of where their own source authors
happened to put the local origin. Both directions are now normalized the
same way.
"""
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder, flat_quad  # noqa: E402

import mesh_convert


class TestOriginRecentering(unittest.TestCase):
    def _convert(self, td, positions_builder):
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((150, 150, 150, 255)))
        mat = b.add_material("Mat", base_color_texture_index=tex)
        positions, normals, uvs, indices = positions_builder()
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        b.add_node(mesh_index=mesh, name="Thing")

        glb_path = td / "model.glb"
        glb_path.write_bytes(b.build())
        obj_dir = td / "objects"
        tex_dir = td / "textures"
        obj_dir.mkdir()
        tex_dir.mkdir()
        result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
        self.assertTrue(result)
        sidecar = json.loads((obj_dir / "model.originoffset.json").read_text(encoding="utf-8"))
        return result, sidecar

    def _mean_xz(self, obj_path):
        text = obj_path.read_text(encoding="utf-8")
        vt_lines = [l for l in text.splitlines() if l.startswith("VT ")]
        xs = [float(l.split()[1]) for l in vt_lines]
        zs = [float(l.split()[3]) for l in vt_lines]
        return sum(xs) / len(xs), sum(zs) / len(zs)

    def test_off_center_rigid_building_gets_recentered_and_offset_recorded(self):
        """A genuinely 3-D (rigid, TILTED-eligible) box authored entirely in
        the +X/+Z quadrant (footprint center at local (55, 25), not near
        the origin at all) should come out of convert() re-centered around
        (0, 0), with the removed center recorded in the sidecar."""
        def off_center_box():
            hs = 5.0
            cx, cz = 55.0, 25.0
            bx = [(cx - hs, 0, cz - hs), (cx + hs, 0, cz - hs), (cx + hs, 0, cz + hs), (cx - hs, 0, cz + hs),
                  (cx - hs, 6.0, cz - hs), (cx + hs, 6.0, cz - hs), (cx + hs, 6.0, cz + hs), (cx - hs, 6.0, cz + hs)]
            wall_tris = []
            for a, c, d, e in [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
                wall_tris += [(a, c, d), (a, d, e)]
            roof_tris = [(4, 5, 6), (4, 6, 7)]
            indices = [i for tri in (wall_tris + roof_tris) for i in tri]
            return bx, [(0.0, 1.0, 0.0)] * len(bx), [(0.0, 0.0)] * len(bx), indices

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            result, offset = self._convert(td, off_center_box)
            mean_x, mean_z = self._mean_xz(result[0])
            self.assertAlmostEqual(mean_x, 0.0, places=3)
            self.assertAlmostEqual(mean_z, 0.0, places=3)
            self.assertAlmostEqual(offset["x"], 55.0, places=3)
            self.assertAlmostEqual(offset["z"], 25.0, places=3)

    def test_off_center_flat_quad_is_not_recentered_horizontally(self):
        """Draped/flat content (a single flat quad, definitely under the
        0.98 file-wide flat threshold) must NOT be re-centered in X/Z, even
        when far off-center -- confirmed real regression: MSFS often
        represents one continuous surface as SEVERAL separate coincident
        objects (a base asphalt fill + a paint-stripe overlay), each its
        own file/placement sharing the same original BGL anchor lat/lon so
        their raw local coordinates already line up. Re-centering each one
        independently around ITS OWN bbox moves each file's placement
        anchor by a different amount and visibly shifts previously-aligned
        layers apart. TILTED's pivot-centering benefit (the reason rigid
        objects DO get re-centered) doesn't apply to draped geometry at
        all -- it's never TILTED-rotated -- so there is no upside to trade
        away by leaving flat content exactly as authored."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            result, offset = self._convert(td, lambda: flat_quad(50, 60, 20, 30))
            mean_x, mean_z = self._mean_xz(result[0])
            self.assertAlmostEqual(mean_x, 55.0, places=3)
            self.assertAlmostEqual(mean_z, 25.0, places=3)
            self.assertAlmostEqual(offset["x"], 0.0, places=6)
            self.assertAlmostEqual(offset["z"], 0.0, places=6)

    def test_negative_y_geometry_gets_lifted_and_agl_delta_is_negative(self):
        """A genuinely 3-D object (so it isn't flattened by the file-wide
        flat/draped path) with real geometry down to y=-2 should get
        lifted so its lowest vertex is at y=0, with a matching NEGATIVE
        offset["y"] (main.py adds this straight into the placement's own
        "agl" field to pull the object back down by the same amount at
        placement time, keeping its final rendered position unchanged)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            def box():
                hs = 5.0
                bx = [(-hs, -2.0, -hs), (hs, -2.0, -hs), (hs, -2.0, hs), (-hs, -2.0, hs),
                      (-hs, 4.0, -hs), (hs, 4.0, -hs), (hs, 4.0, hs), (-hs, 4.0, hs)]
                wall_tris = []
                for a, c, d, e in [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
                    wall_tris += [(a, c, d), (a, d, e)]
                roof_tris = [(4, 5, 6), (4, 6, 7)]
                indices = [i for tri in (wall_tris + roof_tris) for i in tri]
                return bx, [(0.0, 1.0, 0.0)] * len(bx), [(0.0, 0.0)] * len(bx), indices

            result, offset = self._convert(td, box)
            text = result[0].read_text(encoding="utf-8")
            ys = [float(l.split()[2]) for l in text.splitlines() if l.startswith("VT ")]
            self.assertAlmostEqual(min(ys), 0.0, places=3)
            self.assertAlmostEqual(max(ys), 6.0, places=3)  # 4 - (-2), unchanged span
            self.assertAlmostEqual(offset["y"], -2.0, places=3)

    def test_positive_y_geometry_gets_dropped_and_agl_delta_is_positive(self):
        """The mirror case: a genuinely 3-D object whose lowest vertex sits
        ABOVE its own local origin (y=+3 here, nothing at or below y=0 at
        all) must get shifted DOWN so its lowest vertex lands at y=0, with
        a matching POSITIVE offset["y"] (main.py adds this into the
        placement's own "agl" field to push the object back UP by the same
        amount at placement time, keeping its final rendered position
        unchanged). Before this generalization, min_y >= -0.01 meant this
        object was left completely untouched -- no shift, no recorded
        offset -- permanently baking its +3 baseline into the mesh."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            def box():
                hs = 5.0
                bx = [(-hs, 3.0, -hs), (hs, 3.0, -hs), (hs, 3.0, hs), (-hs, 3.0, hs),
                      (-hs, 9.0, -hs), (hs, 9.0, -hs), (hs, 9.0, hs), (-hs, 9.0, hs)]
                wall_tris = []
                for a, c, d, e in [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
                    wall_tris += [(a, c, d), (a, d, e)]
                roof_tris = [(4, 5, 6), (4, 6, 7)]
                indices = [i for tri in (wall_tris + roof_tris) for i in tri]
                return bx, [(0.0, 1.0, 0.0)] * len(bx), [(0.0, 0.0)] * len(bx), indices

            result, offset = self._convert(td, box)
            text = result[0].read_text(encoding="utf-8")
            ys = [float(l.split()[2]) for l in text.splitlines() if l.startswith("VT ")]
            self.assertAlmostEqual(min(ys), 0.0, places=3)
            self.assertAlmostEqual(max(ys), 6.0, places=3)  # 9 - 3, unchanged span
            self.assertAlmostEqual(offset["y"], 3.0, places=3)

    def test_recenter_uses_median_not_bbox_midpoint_so_sparse_outliers_dont_drag_it(self):
        """CONFIRMED REAL BUG: a real converted LHBP ATC tower shell glTF
        bundles the actual ~20m building together with unrelated nearby
        ground/decal geometry, inflating the combined bbox to ~640x640m --
        the bbox-midpoint recenter this test replaces was computed across
        that whole inflated span, moving the placement's own anchor by the
        resulting large, wrong offset (confirmed ~163m in the real case)
        even though the tower's OWN true center never moved. A median
        recenter only moves once outliers are close to HALF the total
        point count, unlike a bbox midpoint which just TWO extreme corner
        vertices (regardless of how sparse) can drag anywhere -- so a
        dense real structure's own vertices dominate the result as long as
        a sparse unrelated decal stays a small minority of the total,
        which is exactly the real case (a few big flat decal quads next to
        thousands of detailed building vertices).

        Fixture: a dense "building" cluster (repeated boxes -- many
        vertices at local x in {-5, 0(ridge), +5}) plus a sparse single
        "decal" quad far away (x ~= 300, only 4 vertices). Expected offset
        is computed the same way the fixture data is generated, then
        cross-checked as strictly closer to the building's own true range
        than the old bbox-midpoint formula would have been -- not hand-
        derived, so this doesn't depend on getting arithmetic right by
        eye, just on the two formulas genuinely disagreeing here."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            def dense_building_plus_sparse_decal():
                positions, indices = [], []

                def add_gabled_box(cx, cz, hs, y0=0.0, y1=6.0, apex_y=8.0):
                    """Walls + a real gable roof (ridge line AT x=cx, the
                    box's own true center) -- a non-degenerate, physically
                    ordinary building detail, not a synthetic filler point:
                    a few vertices genuinely sitting at the true center is
                    exactly what a real building's own continuous geometry
                    provides for free, which is what keeps the median from
                    landing on the box's +hs face instead of its center
                    (see class docstring)."""
                    nonlocal indices
                    base = len(positions)
                    bx = [(cx - hs, y0, cz - hs), (cx + hs, y0, cz - hs),
                          (cx + hs, y0, cz + hs), (cx - hs, y0, cz + hs),
                          (cx - hs, y1, cz - hs), (cx + hs, y1, cz - hs),
                          (cx + hs, y1, cz + hs), (cx - hs, y1, cz + hs),
                          (cx, apex_y, cz - hs), (cx, apex_y, cz + hs)]  # ridge: indices base+8, base+9
                    positions.extend(bx)
                    for a, c, d, e in [(0, 1, 5, 4), (2, 3, 7, 6)]:
                        indices += [base + a, base + c, base + d, base + a, base + d, base + e]
                    # Two gable-end triangles + two roof-slope quads (as
                    # triangle pairs) meeting at the ridge -- a real,
                    # non-degenerate gabled roof.
                    indices += [base + 4, base + 5, base + 8]
                    indices += [base + 7, base + 6, base + 9]
                    indices += [base + 4, base + 8, base + 9, base + 4, base + 9, base + 7]
                    indices += [base + 5, base + 6, base + 9, base + 5, base + 9, base + 8]

                # Dense real "building": 15 coincident-footprint gabled
                # boxes at true center (0, 0) -- many vertices, none by
                # themselves off-center, plus a real ridge at x=0 each time.
                for _ in range(15):
                    add_gabled_box(0.0, 0.0, hs=5.0)

                # Sparse, far-away "decal": ONE small quad, only 4 vertices.
                decal_base = len(positions)
                positions.extend([(299.0, 0.0, 299.0), (301.0, 0.0, 299.0),
                                   (301.0, 0.0, 301.0), (299.0, 0.0, 301.0)])
                indices += [decal_base, decal_base + 1, decal_base + 2,
                            decal_base, decal_base + 2, decal_base + 3]

                normals = [(0.0, 1.0, 0.0)] * len(positions)
                uvs = [(0.0, 0.0)] * len(positions)
                return positions, normals, uvs, indices

            result, offset = self._convert(td, dense_building_plus_sparse_decal)

            # Recompute both candidate formulas from the SAME raw vertex
            # data the fixture built, independently of convert()'s own
            # internals, so this doesn't just re-assert whatever the
            # implementation happens to do.
            all_positions, _, _, _ = dense_building_plus_sparse_decal()
            all_x = [p[0] for p in all_positions]
            bbox_mid_x = (min(all_x) + max(all_x)) / 2.0
            median_x = statistics.median(all_x)

            self.assertNotAlmostEqual(bbox_mid_x, median_x, places=1,
                                       msg="test setup issue: the two formulas must actually disagree here")
            self.assertAlmostEqual(offset["x"], median_x, places=3)
            # The real proof: median-based recenter stays within (or very
            # close to) the dense building's own true footprint, while the
            # old bbox-midpoint formula was dragged out past it toward the
            # sparse decal.
            self.assertLess(abs(offset["x"]), 10.0,
                             "median recenter must stay near the dense building's own footprint")
            self.assertGreater(abs(bbox_mid_x), 100.0,
                                "test setup issue: expected the bbox-midpoint formula to be dragged far off")

    def test_already_centered_flat_pavement_is_a_no_op(self):
        """A flat file already straddling the origin symmetrically (and
        already-non-negative, as every draped/flat file's own reference
        height clamp already guarantees -- see convert.py's file_reference_
        height clamp) should get a zero offset: nothing to compensate for
        in the placement."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            result, offset = self._convert(td, lambda: flat_quad(-10, 10, -5, 5))
            self.assertAlmostEqual(offset["x"], 0.0, places=6)
            self.assertAlmostEqual(offset["y"], 0.0, places=6)
            self.assertAlmostEqual(offset["z"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
