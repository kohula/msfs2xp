"""
terrain_fit.py rewritten to consume MeshIR sidecars (mesh_convert.mesh_ir)
instead of regex-parsing OBJ8 text, and geo_transform.py instead of its
own duplicated rotation math. Pins the "corrected per model, not per
sub-object" grouping policy: a too-small-to-qualify-alone sibling and the
"_lights" companion must get the IDENTICAL correction as their large
sibling at the same real-world point.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import terrain_dem
import terrain_fit

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from dsf_fixture import build_elevation_dsf  # noqa: E402
from gltf_builder import GltfBuilder, flat_quad  # noqa: E402

import mesh_convert
from mesh_convert import mesh_ir


class TestTerrainFit(unittest.TestCase):
    def _write_sloped_terrain(self, root: Path, tile_lat: int, tile_lon: int, slope_per_post: float):
        width = height = 5
        grid = [[col * slope_per_post for col in range(width)] for _ in range(height)]
        folder = f"{(tile_lat // 10) * 10:+03d}{(tile_lon // 10) * 10:+04d}"
        dsf_dir = root / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / folder
        dsf_dir.mkdir(parents=True, exist_ok=True)
        (dsf_dir / f"{tile_lat:+03d}{tile_lon:+04d}.dsf").write_bytes(build_elevation_dsf(grid))

    def _write_bumpy_terrain(self, root: Path, tile_lat: int, tile_lon: int, bump_scale: float):
        """A real quadratic bump (not a pure linear ramp) -- unlike
        _write_sloped_terrain, a symmetric footprint's own grid samples
        do NOT cancel out to zero when averaged: a linear slope sampled
        symmetrically around its own center averages to exactly the
        center's own value (a pure tilt, which _robust_vertical_shift
        correctly leaves alone -- that's what rotation is for), while a
        real bump/dip genuinely shifts the whole footprint's average
        elevation relative to the anchor, which is what a uniform shift
        SHOULD correct for."""
        N = 5  # same grid size as _write_sloped_terrain, whose spacing is proven to reach a real object's footprint
        mid = N // 2
        grid = [[int(((r - mid) ** 2 + (c - mid) ** 2) * bump_scale) for c in range(N)] for r in range(N)]
        folder = f"{(tile_lat // 10) * 10:+03d}{(tile_lon // 10) * 10:+04d}"
        dsf_dir = root / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / folder
        dsf_dir.mkdir(parents=True, exist_ok=True)
        (dsf_dir / f"{tile_lat:+03d}{tile_lon:+04d}.dsf").write_bytes(build_elevation_dsf(grid))

    def _build_glb(self, path: Path, name: str, texture_name: str, x0, x1, z0, z1, y=0.0):
        b = GltfBuilder()
        tex = b.add_image_data_uri((150, 150, 150, 255), name=texture_name)
        texi = b.add_texture(tex)
        mat = b.add_material(f"{name}Mat", base_color_texture_index=texi)
        positions = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
        normals = [(0.0, 1.0, 0.0)] * 4
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        indices = [0, 1, 2, 0, 2, 3]
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        b.add_node(mesh_index=mesh, name=name)
        path.write_bytes(b.build())

    def _build_box_glb(self, path: Path, name: str, texture_name: str, half_size: float, height: float = 6.0):
        """Walls + roof -- genuinely non-flat, so mesh_convert classifies
        the whole file TILTED (same shape as
        test_flatness_tilted_exclusivity.test_genuine_3d_building_stays_rigid).
        half_size controls footprint: large enough clears terrain_fit's own
        size gate, small enough stays under it."""
        b = GltfBuilder()
        tex = b.add_image_data_uri((150, 150, 150, 255), name=texture_name)
        texi = b.add_texture(tex)
        mat = b.add_material(f"{name}Mat", base_color_texture_index=texi)
        hs = half_size
        bx = [(-hs, 0, -hs), (hs, 0, -hs), (hs, 0, hs), (-hs, 0, hs),
              (-hs, height, -hs), (hs, height, -hs), (hs, height, hs), (-hs, height, hs)]
        wall_tris = []
        for a, c, d, e in [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
            wall_tris += [(a, c, d), (a, d, e)]
        roof_tris = [(4, 5, 6), (4, 6, 7)]
        indices = [i for tri in (wall_tris + roof_tris) for i in tri]
        mesh = b.add_mesh(
            bx, indices, normals=[(0.0, 1.0, 0.0)] * len(bx), uvs=[(0.0, 0.0)] * len(bx), material_index=mat)
        b.add_node(mesh_index=mesh, name=name)
        path.write_bytes(b.build())

    def test_draped_siblings_and_lights_companion_all_get_the_identical_correction(self):
        """Both "Walls" and "Trim" are flat quads here, so convert() emits
        both as ATTR_draped ground-level geometry -- draped content is
        eligible for the real per-vertex terrain warp now (same as rigid
        buildings; ATTR_draped still re-projects it at render time
        regardless, so this doesn't change the main render, only what's
        available to whatever else reads authored Y -- see terrain_fit's
        own module docstring). Trim's own footprint is compact relative to
        ITS OWN bbox (not just the group's shared bbox dominated by Walls),
        so it must NOT be mistaken for sparse/GPU-instanced-looking content
        and skipped -- all three siblings, including the synthetic "_lights"
        companion, must land on the exact same Y correction at their shared
        real-world point, pinning the "one consistent decision for the
        whole group" grouping contract this module has always guaranteed."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)

            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            base_lat, base_lon, heading = 47.5, 8.5, 0.0

            # Walls + Trim as two materials/nodes in ONE glb (like a real
            # MSFS model split by material into several sibling .obj files
            # by one convert() call), not two separate glb files -- convert()
            # now re-centers a whole model around its own shared XZ
            # footprint (see convert.py's re-centering pass), so two
            # genuinely separate files no longer share a coordinate frame
            # the way real convert()-time siblings automatically do.
            b = GltfBuilder()
            walls_tex = b.add_texture(b.add_image_data_uri((150, 150, 150, 255), name="WallsTex"))
            walls_mat = b.add_material("WallsMat", base_color_texture_index=walls_tex)
            trim_tex = b.add_texture(b.add_image_data_uri((150, 150, 150, 255), name="TrimTex"))
            trim_mat = b.add_material("TrimMat", base_color_texture_index=trim_tex)

            wp, wn, wuv, wi = flat_quad(-15, 15, -10, 10)
            walls_mesh = b.add_mesh(wp, wi, normals=wn, uvs=wuv, material_index=walls_mat)
            b.add_node(mesh_index=walls_mesh, name="Walls")

            tp, tn, tuv, ti = flat_quad(14, 15, 9, 10)
            trim_mesh = b.add_mesh(tp, ti, normals=tn, uvs=tuv, material_index=trim_mat)
            b.add_node(mesh_index=trim_mesh, name="Trim")

            model_glb = td / "building.glb"
            model_glb.write_bytes(b.build())
            result = mesh_convert.convert(model_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            walls_stem = next(p.stem for p in result if "Walls" in p.stem)
            trim_stem = next(p.stem for p in result if "Trim" in p.stem)

            # A synthetic "_lights" sidecar sharing the same group, matching
            # what convert() itself would produce for the building's own
            # point lights (built directly since gltf_builder doesn't cover
            # ASOBO_macro_light end-to-end placement here).
            lights_stem = "building_lights"
            lights_ir = mesh_ir.MeshIR(
                name=lights_stem,
                lights=[mesh_ir.LightEntry(pos=(15.0, 8.5, 10.0), dir=(0.0, -1.0, 0.0),
                                            color=(1.0, 1.0, 1.0), cone_angle=360.0, size=1.5, dataref="NULL")],
            )
            mesh_ir.save(lights_ir, mesh_ir.sidecar_path_for(obj_dir / f"{lights_stem}.obj"))

            stems = [walls_stem, trim_stem, lights_stem]
            results = terrain_fit.get_or_create_fitted_group(obj_dir, stems, base_lat, base_lon, heading, xplane_root)

            self.assertTrue(results[walls_stem][1], "large sibling should qualify")
            self.assertTrue(results[trim_stem][1], "too-small-alone sibling should still be corrected as part of the group")
            self.assertTrue(results[lights_stem][1], "lights companion should also be corrected")

            def y_at(obj_stem, target_x, target_z):
                text = (obj_dir / f"{results[obj_stem][0]}.obj").read_text(encoding="utf-8")
                for line in text.splitlines():
                    if line.startswith("VT "):
                        p = line.split()
                        if abs(float(p[1]) - target_x) < 1e-6 and abs(float(p[3]) - target_z) < 1e-6:
                            return float(p[2])
                return None

            walls_y = y_at(walls_stem, 15.0, 10.0)
            trim_y = y_at(trim_stem, 15.0, 10.0)
            self.assertIsNotNone(walls_y)
            self.assertIsNotNone(trim_y)
            self.assertNotAlmostEqual(walls_y, 0.0, places=3, msg="draped geometry must now receive the real terrain warp")
            self.assertAlmostEqual(walls_y, trim_y, places=5,
                                    msg="Trim must get the identical correction as Walls at their shared point, "
                                        "not be skipped as sparse/instanced-looking just because it's small")

            light_text = (obj_dir / f"{results[lights_stem][0]}.obj").read_text(encoding="utf-8")
            # LIGHT_PARAM <name> px py pz ...  -> py is token index 3
            light_line = next(l for l in light_text.splitlines() if l.startswith("LIGHT_PARAM "))
            light_py = float(light_line.split()[3])
            self.assertAlmostEqual(light_py - 8.5, walls_y, places=5,
                                    msg="the lights companion must get the same correction as the draped siblings too")

    def test_skip_draped_positions_leaves_draped_siblings_unwarped_but_still_corrects_lights(self):
        """Per the user's own explicit instruction for the .pol/DSF-polygon
        conversion path: "the terrain fit should be applied to every object
        placed on the ground, except the polygons, as they are going to be
        DRAPED". skip_draped_positions=True must leave every DRAPED
        sibling's own geometry completely untouched (result_stem == the
        original stem, applied=False, reason="skipped_for_polygon_mode") --
        a real DRAPED_POLYGON has no elevation field of its own at all, so
        warping it here would be silently discarded downstream anyway (see
        this module's own docstring for get_or_create_fitted_group). The
        lights companion is a separate, independent point (not connected
        mesh topology) and must still be corrected exactly as before --
        this flag only ever gates the draped position warp, nothing else."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)

            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            base_lat, base_lon, heading = 47.5, 8.5, 0.0

            b = GltfBuilder()
            walls_tex = b.add_texture(b.add_image_data_uri((150, 150, 150, 255), name="WallsTex"))
            walls_mat = b.add_material("WallsMat", base_color_texture_index=walls_tex)
            wp, wn, wuv, wi = flat_quad(-15, 15, -10, 10)
            walls_mesh = b.add_mesh(wp, wi, normals=wn, uvs=wuv, material_index=walls_mat)
            b.add_node(mesh_index=walls_mesh, name="Walls")
            model_glb = td / "building.glb"
            model_glb.write_bytes(b.build())
            result = mesh_convert.convert(model_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            walls_stem = next(p.stem for p in result if "Walls" in p.stem)

            lights_stem = "building_lights"
            lights_ir = mesh_ir.MeshIR(
                name=lights_stem,
                lights=[mesh_ir.LightEntry(pos=(15.0, 8.5, 10.0), dir=(0.0, -1.0, 0.0),
                                            color=(1.0, 1.0, 1.0), cone_angle=360.0, size=1.5, dataref="NULL")],
            )
            mesh_ir.save(lights_ir, mesh_ir.sidecar_path_for(obj_dir / f"{lights_stem}.obj"))

            stems = [walls_stem, lights_stem]
            results = terrain_fit.get_or_create_fitted_group(
                obj_dir, stems, base_lat, base_lon, heading, xplane_root, skip_draped_positions=True)

            self.assertEqual(results[walls_stem], (walls_stem, False, "skipped_for_polygon_mode"))
            self.assertTrue(results[lights_stem][1], "lights companion is still corrected")

    def test_large_building_on_steep_terrain_gets_the_precise_warp(self):
        """CONFIRMED REAL BUG this pins: a large rigid building on
        genuinely sloped terrain used to get ONE uniform shift (every
        vertex moves by the same amount) -- correct for a bump/dip that
        shifts the whole footprint's average elevation, but for a REAL
        SLOPE across a large footprint, one averaged number necessarily
        leaves one end of the building floating and the other sunk/
        underground, no matter how robust the average is (confirmed real
        symptom: large buildings partially underground). Past
        _RIGID_WARP_SLOPE_THRESHOLD_M of real sampled slope, a rigid
        group now gets the SAME per-vertex warp draped content gets
        instead -- some shear risk to architectural detail is an accepted
        trade-off there, since the alternative (part of the building
        buried) is worse. Not TILTED/a rotation (a rotation only fixes a
        genuine tilt, never a uniform anchor-offset error -- see the
        module's own docstring)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_bumpy_terrain(xplane_root, 47, 8, bump_scale=1500.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            building_glb = td / "building.glb"
            self._build_box_glb(building_glb, "BigBuilding", "BuildingTex", half_size=15.0)
            result = mesh_convert.convert(building_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem
            self.assertNotIn("ATTR_draped", result[0].read_text(encoding="utf-8"),
                              "test setup issue: expected this box fixture to convert as rigid")

            original_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{stem}.obj"))

            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], 47.5, 8.5, 0.0, xplane_root)
            result_stem, applied, reason = results[stem]
            self.assertTrue(applied, f"a large rigid building on meaningfully sloped terrain must be corrected (reason={reason})")
            self.assertEqual(reason, "applied_rigid_warp")
            self.assertNotEqual(result_stem, stem, "a corrected copy should have been written")

            corrected_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{result_stem}.obj"))
            self.assertFalse(
                np.allclose(original_ir.positions, corrected_ir.positions, atol=1e-6),
                "expected the warp to actually move the geometry, not be a no-op"
            )

            # X/Z entirely unchanged (only Y moves) -- unlike a uniform
            # shift, different (x, z) columns are free to move by
            # DIFFERENT amounts, each following the real terrain sampled
            # directly under it (this fixture's own box happens to sit
            # exactly at the radially-symmetric bump's own center, so its
            # 4 equidistant corners coincidentally warp by the identical
            # amount here -- see test_oversized_footprint_on_real_slope_
            # gets_the_warp_not_left_partially_underground, on a LINEAR
            # slope instead, for a real per-vertex-varies assertion).
            np.testing.assert_allclose(corrected_ir.positions[:, [0, 2]], original_ir.positions[:, [0, 2]], atol=1e-9)

    def test_tall_building_gets_the_same_warp_as_a_short_one(self):
        """Neither a uniform shift nor this module's per-vertex warp has
        a rotation's "implied displacement scales with distance from the
        pivot" problem (the old "excessive implied roof displacement"
        guard existed only for the rotation this module no longer uses at
        all) -- both key strictly off each vertex's own (x, z), never its
        Y, so a roof directly above its own base moves by the identical
        amount as that base regardless of how tall the building is. This
        tower (150m instead of the passing 6m box's height, otherwise
        identical fixture/terrain, both routed to applied_rigid_warp by
        the real slope) proves that still holds under the warp too."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            tower_glb = td / "tower.glb"
            self._build_box_glb(tower_glb, "TallTower", "TowerTex", half_size=15.0, height=150.0)
            result = mesh_convert.convert(tower_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem
            original_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{stem}.obj"))

            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], 47.5, 8.5, 0.0, xplane_root)
            result_stem, applied, reason = results[stem]
            self.assertTrue(applied)
            self.assertEqual(reason, "applied_rigid_warp")

            corrected_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{result_stem}.obj"))
            roof_mask = original_ir.positions[:, 1] > 100.0
            base_mask = original_ir.positions[:, 1] < 0.1
            self.assertTrue(roof_mask.any() and base_mask.any(), "test setup issue")
            roof_dy = corrected_ir.positions[roof_mask, 1] - original_ir.positions[roof_mask, 1]
            base_dy = corrected_ir.positions[base_mask, 1] - original_ir.positions[base_mask, 1]
            self.assertAlmostEqual(float(roof_dy.mean()), float(base_dy.mean()), places=5,
                                    msg="roof and base share the same (x, z) columns, so the warp must move "
                                        "them by the identical amount -- no rotation-scaling concern")

    def test_oversized_footprint_on_real_slope_gets_the_warp_not_left_partially_underground(self):
        """CONFIRMED REAL BUG this pins: a genuinely large SINGLE building
        (not bundled-unrelated-content -- a real continuous terminal
        structure) can have a large footprint (EGLC's own terminal
        measured 380m wide). Giving an oversized footprint on real sloped
        terrain ONE uniform shift (an earlier version of this test's own
        expectation) has exactly the failure mode a real user reported:
        one averaged number can only ever be exactly right for the
        group's own AVERAGE terrain delta, so a 600m-wide building on a
        genuine slope still comes out with one end floating and the other
        sunk/underground -- worse the larger the footprint, not better.
        Past _RIGID_WARP_SLOPE_THRESHOLD_M of real sampled slope, size
        alone no longer exempts a group from the precise per-vertex warp
        -- unlike the old rotation this replaced, a warp can't "swing" a
        separately-anchored sibling (each vertex samples its own real
        position independently), so there's no oversized-footprint risk
        to guard against here at all."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            building_glb = td / "huge_building.glb"
            self._build_box_glb(building_glb, "HugeBuilding", "HugeBuildingTex", half_size=300.0)
            result = mesh_convert.convert(building_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem
            self.assertNotIn("ATTR_draped", result[0].read_text(encoding="utf-8"),
                              "test setup issue: expected this box fixture to convert as rigid")
            original_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{stem}.obj"))

            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], 47.5, 8.5, 0.0, xplane_root)
            result_stem, applied, reason = results[stem]
            self.assertTrue(applied, "an oversized footprint must still be corrected, not left floating")
            self.assertEqual(reason, "applied_rigid_warp")
            self.assertNotEqual(result_stem, stem, "a corrected copy should have been written")

            corrected_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{result_stem}.obj"))
            roof_mask = original_ir.positions[:, 1] > 3.0
            base_mask = original_ir.positions[:, 1] < 0.1
            self.assertTrue(roof_mask.any() and base_mask.any(), "test setup issue")
            roof_dy = corrected_ir.positions[roof_mask, 1] - original_ir.positions[roof_mask, 1]
            base_dy = corrected_ir.positions[base_mask, 1] - original_ir.positions[base_mask, 1]
            self.assertAlmostEqual(float(roof_dy.mean()), float(base_dy.mean()), places=5,
                                    msg="roof and base share the same (x, z) columns, so still move together")
            # The whole point: on a REAL slope this large, the warp must
            # actually vary across the footprint (unlike a shift) -- one
            # side of a 600m building sits measurably higher than the
            # other on this fixture's slope.
            dy = corrected_ir.positions[:, 1] - original_ir.positions[:, 1]
            self.assertGreater(float(dy.max() - dy.min()), 1.0,
                                msg="expected a real, footprint-scale variation in the correction on a 600m slope")

    def test_small_object_gets_the_precise_warp_no_size_disqualification(self):
        """CONFIRMED REAL BUG this pins: terrain_fit's own correction used
        to be gated to objects with a footprint >=300m2/10m-per-side, with
        X-Plane's own TILTED rotation left as the ONLY correction for
        anything smaller -- but TILTED can only ever fix a genuine local
        SLOPE, never a flat-out wrong anchor elevation (a rotation can't
        move its own origin). That left small/medium objects silently
        uncorrected for exactly the more common real problem. There is no
        disqualification gate any more: this 8m-wide box (small enough to
        have failed the old 10m-per-side gate) on genuinely uneven terrain
        must now be corrected too -- and, being this small, with the
        precise per-vertex warp rather than an averaged shift (see
        _RIGID_WARP_MAX_SIDE_M in the module docstring: real terrain
        barely varies at all across a footprint this size, so the warp
        carries no meaningful shear risk and is strictly more accurate).
        Bumpy, not linearly-sloped, terrain: a symmetric footprint on a
        pure linear slope averages to zero by construction (that's a
        tilt, not an offset -- see _write_bumpy_terrain's own docstring)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_bumpy_terrain(xplane_root, 47, 8, bump_scale=1500.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            small_glb = td / "small_box.glb"
            self._build_box_glb(small_glb, "SmallBox", "SmallBoxTex", half_size=4.0, height=3.0)
            result = mesh_convert.convert(small_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem
            self.assertNotIn("ATTR_draped", result[0].read_text(encoding="utf-8"),
                              "test setup issue: expected this small box fixture to convert as rigid")
            original_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{stem}.obj"))

            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], 47.5, 8.5, 0.0, xplane_root)
            result_stem, applied, reason = results[stem]
            self.assertTrue(applied, f"a small object on meaningfully sloped terrain must now be corrected too (reason={reason})")
            self.assertEqual(reason, "applied_rigid_warp")
            self.assertNotEqual(result_stem, stem, "a corrected copy should have been written")

            corrected_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{result_stem}.obj"))
            self.assertFalse(np.allclose(original_ir.positions, corrected_ir.positions, atol=1e-6),
                              "expected the warp to actually move the geometry, not be a no-op")

    def test_shared_shift_links_a_disqualified_sibling_at_the_same_anchor(self):
        """Universal fix for: one real-world building instance split into
        two placements from different source paths (confirmed real case:
        LHBP's ATC tower -- an SPB-attached exterior shell + a plain-BGL-
        placed interior, at the same real-world anchor, that never share a
        model stem so never reach the same terrain_fit group_key). Both
        the shell and the (now, with no size gate) small interior qualify
        independently, but each samples its OWN (differently-sized)
        footprint, so on genuinely uneven (not just linearly sloped --
        see _write_bumpy_terrain) terrain their independently-computed
        shifts can legitimately differ. apply_shared_shift_to_group, fed
        the shell's cached transform via get_cached_transform, must
        override the interior with the shell's EXACT shift value instead
        -- no anchor-delta bookkeeping needed (unlike the rotation this
        replaced): the shift is a property of the shared real-world
        anchor, not of either object's own local-frame convention or its
        own footprint, so it applies directly regardless of the
        interior's own (different) recenter offset or AGL height."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_bumpy_terrain(xplane_root, 47, 8, bump_scale=1500.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            shell_glb = td / "shell.glb"
            self._build_box_glb(shell_glb, "Shell", "ShellTex", half_size=15.0)
            shell_result = mesh_convert.convert(shell_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            shell_stem = shell_result[0].stem

            interior_glb = td / "interior.glb"
            self._build_box_glb(interior_glb, "Interior", "InteriorTex", half_size=4.0, height=3.0)
            interior_result = mesh_convert.convert(interior_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            interior_stem = interior_result[0].stem
            original_interior_positions = mesh_ir.load(
                mesh_ir.sidecar_path_for(obj_dir / f"{interior_stem}.obj")).positions.copy()

            # Same anchor for both -- the whole point of the scenario.
            anchor_lat, anchor_lon, anchor_hdg = 47.5, 8.5, 0.0

            shell_group_key = (tuple(sorted([shell_stem])), round(anchor_lat, 6), round(anchor_lon, 6),
                                round(anchor_hdg, 2), False)
            shell_results = terrain_fit.get_or_create_fitted_group(
                obj_dir, [shell_stem], anchor_lat, anchor_lon, anchor_hdg, xplane_root)
            _, shell_applied, shell_reason = shell_results[shell_stem]
            self.assertTrue(shell_applied, f"test setup issue: expected the shell to qualify (reason={shell_reason})")

            # The interior, evaluated on its own, ALSO qualifies now (no
            # size gate) -- but its own independently-sampled footprint
            # gives it a genuinely different shift value than the shell's,
            # setting up the real point of this test: explicit linking
            # below must override that with the shell's exact value.
            interior_results_alone = terrain_fit.get_or_create_fitted_group(
                obj_dir, [interior_stem], anchor_lat, anchor_lon, anchor_hdg, xplane_root)
            self.assertTrue(interior_results_alone[interior_stem][1], "test setup issue: expected the interior to also qualify alone")

            transform = terrain_fit.get_cached_transform(shell_group_key)
            self.assertIsNotNone(transform)
            self.assertIsNotNone(transform["vertical_shift"])

            linked_results = terrain_fit.apply_shared_shift_to_group(obj_dir, [interior_stem], transform, xplane_root)
            linked_stem, linked_applied, linked_reason = linked_results[interior_stem]
            self.assertTrue(linked_applied, "the interior must be corrected once linked to the shell's shift")
            self.assertEqual(linked_reason, "applied_shared_shift")
            self.assertNotEqual(linked_stem, interior_stem)

            linked_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{linked_stem}.obj"))
            # X/Z entirely unchanged, every vertex's Y moved by the exact
            # shift value the shell got -- proves it was linked to the
            # shell's own value, overriding whatever the interior's own
            # independent computation (checked above) would have used.
            np.testing.assert_allclose(linked_ir.positions[:, [0, 2]], original_interior_positions[:, [0, 2]], atol=1e-9)
            dy = linked_ir.positions[:, 1] - original_interior_positions[:, 1]
            np.testing.assert_allclose(dy, transform["vertical_shift"], atol=1e-6,
                                        err_msg="every vertex must move by exactly the shell's own shift amount")

    def test_shared_shift_is_a_noop_when_the_source_group_had_no_shift_to_offer(self):
        """get_cached_transform on a group whose cached transform has
        vertical_shift=None (no usable terrain samples at all) must make
        apply_shared_shift_to_group a no-op, not synthesize a spurious
        shift -- propagating "no shift to share" is exactly as valid a
        shared decision as propagating a real one. Constructed directly
        (rather than engineering a real all-samples-failed scenario)
        since the transform dict's own shape is all this checks."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            transform = {"vertical_shift": None, "base_lat": 47.5, "base_lon": 8.5,
                         "heading_deg": 0.0, "origin_elev": 10.0}
            other_stem = "some_other_stem_at_the_same_anchor"
            results = terrain_fit.apply_shared_shift_to_group(obj_dir, [other_stem], transform, None)
            self.assertEqual(results[other_stem], (other_stem, False, "not_applicable"))

    def test_get_cached_transform_returns_none_for_an_unprocessed_group(self):
        self.assertIsNone(terrain_fit.get_cached_transform((("nonexistent_stem",), 1.0, 2.0, 3.0, False)))

    def test_qualifying_group_on_gentle_real_slope_gets_the_ordinary_shift(self):
        """A plain integration check that a large, qualifying group on
        genuinely bumpy but GENTLE real terrain (its sampled corner
        spread stays under _RIGID_WARP_SLOPE_THRESHOLD_M) gets
        applied_vertical_shift with a real, non-zero, consistent
        correction -- confirming the ordinary shift path is still very
        much alive for the large-and-relatively-flat case, not replaced
        outright by the warp. Outlier-rejection math itself is unit-
        tested directly against synthetic samples in
        TestRobustVerticalShift below, where exact DSF-grid/real-world
        post alignment doesn't need to be reasoned about. (A pure linear
        SLOPE, symmetric around the object's own anchor, averages to
        exactly zero by construction -- that's a tilt, which a uniform
        shift correctly leaves alone; a real bump/dip is what a uniform
        shift is actually for, see _write_bumpy_terrain.)"""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_bumpy_terrain(xplane_root, 47, 8, bump_scale=20.0)
            obj_dir = td / "objects"
            obj_dir.mkdir()

            hs = 60.0
            positions = np.array(
                [(-hs, 0.0, -hs), (hs, 0.0, -hs), (hs, 0.0, hs), (-hs, 0.0, hs)], dtype=np.float64)
            ir = mesh_ir.MeshIR(
                name="slope_box", positions=positions,
                normals=np.tile([0.0, 1.0, 0.0], (4, 1)).astype(np.float64),
                uvs=np.zeros((4, 2), dtype=np.float64),
                indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
                texture="t.png", draped=False, footprint_area_m2=None,
            )
            mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / "slope_box.obj"))

            res = terrain_fit.get_or_create_fitted_group(obj_dir, ["slope_box"], 47.5, 8.5, 0.0, xplane_root)
            rstem, applied, reason = res["slope_box"]
            self.assertTrue(applied, f"reason={reason!r}")
            self.assertEqual(reason, "applied_vertical_shift")
            corrected = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{rstem}.obj"))
            dy = corrected.positions[:, 1] - positions[:, 1]
            self.assertAlmostEqual(float(dy.max() - dy.min()), 0.0, places=6, msg="a uniform shift, all 4 corners identical")
            self.assertNotAlmostEqual(float(dy[0]), 0.0, places=3, msg="expected a real, non-zero correction on sloped terrain")

    def test_huge_draped_quad_gets_warped_like_a_building_now(self):
        """Draped/pavement geometry is eligible for the real per-vertex
        terrain warp again, same as rigid buildings -- ATTR_draped still
        re-projects it onto X-Plane's own terrain at render time regardless
        of authored Y, so this doesn't change the main render, but keeps
        authored Y realistically close to the real ground contour instead
        of a flat plane (closes the residual "shadow floating slightly
        above the surface" gap the flat-Y-only fix left behind). This is
        the real 5518m x 3003m "TileSeams" footprint from a converted
        package -- with per-vertex-exact sampling (no shared-bbox grid to
        get misled by a huge footprint) there's no size-based reason to
        disqualify or special-case it at all."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            huge_glb = td / "huge_decal.glb"
            self._build_glb(huge_glb, "HugeDecal", "HugeDecalTex", -2759, 2759, -1502, 1502)
            result = mesh_convert.convert(huge_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem
            self.assertIn("ATTR_draped", result[0].read_text(encoding="utf-8").splitlines(),
                          "test setup issue: expected this flat decal fixture to convert as draped")

            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], 47.5, 8.5, 0.0, xplane_root)
            result_stem, applied, reason = results[stem]
            self.assertTrue(applied, f"a huge but genuinely continuous draped footprint must not be disqualified (reason={reason!r})")

            corrected_text = (obj_dir / f"{result_stem}.obj").read_text(encoding="utf-8")
            self.assertIn("ATTR_draped", corrected_text.splitlines(),
                          "must still be draped -- the main render is unaffected either way")
            y_values = {round(float(line.split()[2]), 5) for line in corrected_text.splitlines() if line.startswith("VT ")}
            self.assertNotEqual(y_values, {0.0},
                                 "a real continuous draped surface should now receive the real terrain warp, not stay flat")

    def test_scattered_points_each_get_their_own_exact_elevation(self):
        """The old grid-interpolation approach's confirmed failure mode: a
        "footprint" bbox spanning many scattered small clusters (like
        GPU-instanced content -- one small motif repeated far apart across
        a wide area) produced a warp with no relationship to any
        individual cluster's real position, because a shared-bbox grid
        blends distant sample points together. Per-vertex-exact sampling
        has no such blending -- each of these two far-apart clusters must
        come out matching a DIRECT, independent elevation computation at
        its own real-world position, not some value interpolated from the
        other cluster or from anywhere in between."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            obj_dir.mkdir()

            stem = "scattered_instanced_like"
            positions = np.array([
                [-2759.0, 0.0, -1502.0], [-2758.0, 0.0, -1502.0], [-2759.0, 0.0, -1501.0],
                [2759.0, 0.0, 1502.0], [2758.0, 0.0, 1502.0], [2759.0, 0.0, 1501.0],
            ], dtype=np.float64)
            normals = np.array([[0.0, 1.0, 0.0]] * 6, dtype=np.float64)
            uvs = np.array([[0.0, 0.0]] * 6, dtype=np.float64)
            indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
            ir = mesh_ir.MeshIR(
                name=stem, positions=positions, normals=normals, uvs=uvs, indices=indices,
                texture="scattered.png", draped=True, footprint_area_m2=2.0,
            )
            mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{stem}.obj"))

            base_lat, base_lon, heading = 47.5, 8.5, 0.0
            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], base_lat, base_lon, heading, xplane_root)
            result_stem, applied, reason = results[stem]
            self.assertTrue(applied, f"reason={reason!r}")

            origin_elev = terrain_dem.get_elevation(xplane_root, base_lat, base_lon)

            def expected_delta(local_x, local_z):
                import geo_transform
                lat, lon = geo_transform.local_offset_to_latlon(base_lat, base_lon, heading, local_x, local_z)
                return terrain_dem.get_elevation(xplane_root, lat, lon) - origin_elev

            corrected_text = (obj_dir / f"{result_stem}.obj").read_text(encoding="utf-8")
            vt_by_xz = {}
            for line in corrected_text.splitlines():
                if line.startswith("VT "):
                    p = line.split()
                    vt_by_xz[(round(float(p[1]), 3), round(float(p[3]), 3))] = float(p[2])

            near_corner_y = vt_by_xz[(-2759.0, -1502.0)]
            far_corner_y = vt_by_xz[(2759.0, 1502.0)]
            self.assertAlmostEqual(near_corner_y, expected_delta(-2759.0, -1502.0), places=3)
            self.assertAlmostEqual(far_corner_y, expected_delta(2759.0, 1502.0), places=3)
            self.assertNotAlmostEqual(near_corner_y, far_corner_y, places=1,
                                       msg="two far-apart clusters on genuinely different real terrain must not "
                                           "come out with the same (grid-blended) correction")

    def test_no_xplane_root_skips_gracefully(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()
            glb = td / "big.glb"
            self._build_glb(glb, "Big", "BigTex", -20, 20, -20, 20)
            result = mesh_convert.convert(glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem
            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], 47.5, 8.5, 0.0, None)
            self.assertEqual(results[stem], (stem, False, "no_xplane_root"))


class TestRobustVerticalShift(unittest.TestCase):
    """_robust_vertical_shift's own math, isolated from any file I/O or
    terrain sampling -- the single number this returns gets applied to
    every vertex of every rigid object in a group, so a bad value here
    (e.g. one spiky sample dragging the average) is as costly as the old
    rotation module's own wrong-signed-rotation risk was."""

    def test_plain_average_when_all_samples_agree(self):
        samples = [(-10.0, -10.0, 2.0), (10.0, -10.0, 2.1), (-10.0, 10.0, 1.9), (10.0, 10.0, 2.0)]
        shift = terrain_fit._robust_vertical_shift(samples)
        self.assertAlmostEqual(shift, 2.0, places=1)

    def test_one_spiky_outlier_is_rejected_not_averaged_in(self):
        """8 samples agreeing closely on ~2.0m, 1 wild outlier at 500m --
        a naive mean would land near 57m (500/9 dominates); the robust
        estimate must stay close to what the consistent majority says."""
        samples = [
            (-10.0, -10.0, 2.0), (0.0, -10.0, 2.1), (10.0, -10.0, 1.9),
            (-10.0, 0.0, 2.0), (10.0, 0.0, 2.2),
            (-10.0, 10.0, 1.8), (0.0, 10.0, 2.0), (10.0, 10.0, 2.1),
            (0.0, 0.0, 500.0),  # the spike -- e.g. a DSF triangulation seam or DEM read error
        ]
        shift = terrain_fit._robust_vertical_shift(samples)
        self.assertLess(abs(shift - 2.0), 1.0, "must track the consistent majority, not the spike")

    def test_two_disagreeing_samples_trusts_both_rather_than_guessing(self):
        """With only 2 samples there's no way to tell which one (if
        either) is the "real" outlier -- must fall back to using both
        (the plain mean) rather than arbitrarily rejecting one."""
        shift = terrain_fit._robust_vertical_shift([(-10.0, 0.0, 1.0), (10.0, 0.0, 3.0)])
        self.assertAlmostEqual(shift, 2.0, places=6)

    def test_no_samples_returns_none(self):
        self.assertIsNone(terrain_fit._robust_vertical_shift([]))

    def test_every_sample_identical_returns_that_value(self):
        """No spread at all to reject against (MAD=0) -- must return the
        common value directly, not divide by zero or otherwise choke."""
        samples = [(-10.0, -10.0, 4.0), (10.0, -10.0, 4.0), (-10.0, 10.0, 4.0), (10.0, 10.0, 4.0)]
        self.assertAlmostEqual(terrain_fit._robust_vertical_shift(samples), 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
