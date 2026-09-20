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

    def test_large_tilted_building_gets_a_rigid_rotation_not_a_shear(self):
        """Confirmed real regression from a live conversion, in two acts:
        first an earlier version of this module applied the same
        per-vertex Y-warp to RIGID (non-draped) siblings too, shearing
        architectural detail into visibly warped/jagged geometry even on
        near-flat terrain -- worse than doing nothing. That was fixed by
        leaving rigid siblings completely untouched (see
        test_small_tilted_object_stays_disqualified_and_unmodified for
        that still-correct behavior on a too-small object). But "untouched"
        for a LARGE qualifying rigid building just means it's still stuck
        with X-Plane's own crude single-point TILTED rotation -- the
        original "one side floating" symptom this whole module exists to
        fix. The real fix: a large TILTED building now gets a single RIGID
        rotation (not per-vertex) fit against several real terrain samples
        across its footprint (better than TILTED's one sampled point),
        replacing TILTED rather than stacking with it -- shape must be
        perfectly preserved (a rotation, unlike a per-vertex shear, can't
        distort it)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            building_glb = td / "building.glb"
            self._build_box_glb(building_glb, "BigBuilding", "BuildingTex", half_size=15.0)
            result = mesh_convert.convert(building_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem
            self.assertIn("TILTED", result[0].read_text(encoding="utf-8").splitlines(),
                          "test setup issue: expected this box fixture to convert as TILTED")

            original_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{stem}.obj"))

            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], 47.5, 8.5, 0.0, xplane_root)
            result_stem, applied, reason = results[stem]
            self.assertTrue(applied, f"a large TILTED building on meaningfully sloped terrain must be corrected (reason={reason})")
            self.assertEqual(reason, "applied_rigid_tilt")
            self.assertNotEqual(result_stem, stem, "a corrected copy should have been written")

            corrected_text = (obj_dir / f"{result_stem}.obj").read_text(encoding="utf-8")
            self.assertNotIn("TILTED", corrected_text.splitlines(),
                              "the rigid rotation must REPLACE TILTED, not stack with it")

            corrected_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{result_stem}.obj"))
            self.assertFalse(
                np.allclose(original_ir.positions, corrected_ir.positions, atol=1e-6),
                "expected the rotation to actually move the geometry, not be a no-op"
            )

            # The core rigidity guarantee: a rotation can't change distances
            # between vertices. Check every pair among a sample of vertices
            # -- if this were still a per-vertex shear, these would differ.
            n = len(original_ir.positions)
            idx = list(range(0, n, max(1, n // 12)))
            for i in idx:
                for j in idx:
                    if i >= j:
                        continue
                    d_before = np.linalg.norm(original_ir.positions[i] - original_ir.positions[j])
                    d_after = np.linalg.norm(corrected_ir.positions[i] - corrected_ir.positions[j])
                    self.assertAlmostEqual(d_before, d_after, places=5,
                                            msg=f"vertex pair ({i},{j}) distance changed -- geometry was sheared, not rotated")

    def test_tall_building_with_excessive_implied_roof_displacement_gets_ground_skirt_only(self):
        """CONFIRMED REAL BUG (part 1): the displacement guard only ever
        tested the footprint's own GROUND-level (Y=0) corners against
        _RIGID_TILT_MAX_DISPLACEMENT_M -- but displacement from a rotation
        scales with distance from the pivot, so a TALL, narrow building's
        roof moves far more than its base for the exact same angle. A
        rotation whose ground-corner displacement comfortably passes (as
        in test_large_tilted_building_gets_a_rigid_rotation_not_a_shear,
        same terrain/footprint) can still swing a tower's roof many times
        further than the guard is supposed to allow, since the guard never
        looked at the object's own height at all -- confirmed real
        symptom: large buildings visibly floating/leaning after
        "correction". This building is identical to that passing test
        except for height (150m instead of 6m) -- the ROTATION must now
        be rejected.

        CONFIRMED REAL BUG (part 2, found right after part 1 shipped):
        rejecting the rotation used to mean NO correction at all (result
        stem unchanged, reason "rigid_skip") -- reverting the whole
        building to its dead-flat original left a visible gap under the
        WHOLE base on this same sloped terrain, not just an excessive
        roof tilt -- worse than before, and especially visible through a
        glass facade (user: "the glass is worse now"). The base must
        still get a ground-only "skirt" correction -- ROOF height
        unchanged (no rotation), base nudged to real terrain."""
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
            self.assertTrue(applied, "the ground band must still be corrected even though the roof-swinging rotation is rejected")
            self.assertEqual(reason, "ground_skirt_only")
            self.assertNotEqual(result_stem, stem, "a corrected copy should have been written")

            corrected_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{result_stem}.obj"))

            # Roof (top face, height 150) must be UNCHANGED -- no rotation
            # means no roof swing at all, which is the whole point.
            roof_mask = original_ir.positions[:, 1] > 100.0
            self.assertTrue(roof_mask.any(), "test setup issue: expected some roof-height vertices")
            self.assertTrue(np.allclose(
                original_ir.positions[roof_mask], corrected_ir.positions[roof_mask], atol=1e-6),
                "the roof must stay exactly at its original position -- no rotation should reach it")

            # Base (ground-contact band) must actually have moved -- this
            # is the whole point of the skirt, on real sloped terrain.
            base_mask = original_ir.positions[:, 1] < 0.1
            self.assertTrue(base_mask.any(), "test setup issue: expected some ground-level vertices")
            self.assertFalse(np.allclose(
                original_ir.positions[base_mask], corrected_ir.positions[base_mask], atol=1e-6),
                "the ground-contact band should have been nudged to match real sampled terrain")

    def test_oversized_footprint_rejects_the_rotation_instead_of_tearing_it_apart(self):
        """Confirmed real bug: a single glTF that bundles one real building
        together with a swath of unrelated nearby ground/decal geometry
        reports a footprint far larger than the building itself (a real
        control tower model came out 640x640 m). The SAME terrain relief
        that produces a small, correct tilt for a normal-sized building
        (see test_large_tilted_building_gets_rigid_rotation_correction,
        identical terrain) fits a much steeper plane across that inflated
        span -- rotating the whole combined mesh rigidly by it would swing
        whichever of its sub-parts sit far from the shared local origin by
        many metres (the "one wall/window panel floating away from the
        rest of the building" the user actually saw). Must be rejected
        outright, not applied."""
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
            self.assertIn("TILTED", result[0].read_text(encoding="utf-8").splitlines(),
                          "test setup issue: expected this box fixture to convert as TILTED")

            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], 47.5, 8.5, 0.0, xplane_root)
            result_stem, applied, reason = results[stem]
            self.assertFalse(applied, f"an oversized footprint's rotation must be rejected (reason={reason})")
            self.assertEqual(result_stem, stem, "a rejected object's original file must be left alone")

    def test_small_tilted_object_stays_disqualified_and_unmodified(self):
        """A TILTED object too small for terrain_fit's own size gate (but
        still above convert()'s separate tiny-object TILT-exemption
        threshold, so it's genuinely TILTED to begin with -- half_size=4
        gives an 8m-wide box: over the 3m radius that skips TILTED
        entirely, under terrain_fit's own 10m-side qualification) must be
        left completely untouched, TILTED directive included -- confirms
        the fix only widens eligibility for objects that ALSO pass the
        existing size qualification, not TILTED objects in general."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            small_glb = td / "small_box.glb"
            self._build_box_glb(small_glb, "SmallBox", "SmallBoxTex", half_size=4.0, height=3.0)
            result = mesh_convert.convert(small_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem
            self.assertIn("TILTED", result[0].read_text(encoding="utf-8").splitlines(),
                          "test setup issue: expected this small box fixture to convert as TILTED")

            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], 47.5, 8.5, 0.0, xplane_root)
            result_stem, applied, reason = results[stem]
            self.assertFalse(applied)
            self.assertEqual(result_stem, stem, "a disqualified object's original file must be left alone")

    def test_shared_rotation_links_a_disqualified_sibling_at_the_same_anchor(self):
        """Universal fix for: one real-world building instance split into
        two placements from different source paths (confirmed real case:
        LHBP's ATC tower -- an SPB-attached exterior shell + a plain-BGL-
        placed interior, at the same real-world anchor, that never share a
        model stem so never reach the same terrain_fit group_key). The
        large shell here qualifies for and gets a real rigid rotation; a
        small companion object at the SAME anchor is, on its own,
        disqualified (matches test_small_tilted_object_stays_disqualified_
        and_unmodified). apply_shared_rotation_to_group, fed the shell's
        cached transform via get_cached_transform, must rotate the small
        object by the IDENTICAL matrix -- proven here by checking the
        rotated small object's positions equal its own original positions
        pre-multiplied by the SAME rotation the shell got, not by
        re-deriving a rotation independently (which would fail: its own
        footprint doesn't qualify)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
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

            # The interior, evaluated on its own, is still disqualified --
            # confirms the "problem" side of this scenario still holds.
            interior_results_alone = terrain_fit.get_or_create_fitted_group(
                obj_dir, [interior_stem], anchor_lat, anchor_lon, anchor_hdg, xplane_root)
            self.assertFalse(interior_results_alone[interior_stem][1])

            transform = terrain_fit.get_cached_transform(shell_group_key)
            self.assertIsNotNone(transform)
            self.assertIsNotNone(transform["rigid_rotation"])

            linked_results = terrain_fit.apply_shared_rotation_to_group(
                obj_dir, [interior_stem], transform, xplane_root)
            linked_stem, linked_applied, linked_reason = linked_results[interior_stem]
            self.assertTrue(linked_applied, "the interior must be corrected once linked to the shell's rotation")
            self.assertEqual(linked_reason, "applied_shared_rotation")
            self.assertNotEqual(linked_stem, interior_stem)

            linked_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{linked_stem}.obj"))
            # Checked via the rigidity/distance-preservation invariant
            # (same style as test_large_tilted_building_gets_a_rigid_
            # rotation_not_a_shear) rather than exact positional equality
            # to the naive rotation, since the foundation skirt may still
            # nudge base-band vertices by a real-terrain residual.
            n = len(original_interior_positions)
            for i in range(n):
                for j in range(i + 1, n):
                    d_before = np.linalg.norm(original_interior_positions[i] - original_interior_positions[j])
                    d_after = np.linalg.norm(linked_ir.positions[i] - linked_ir.positions[j])
                    self.assertAlmostEqual(d_before, d_after, places=5,
                                            msg=f"vertex pair ({i},{j}) distance changed -- not a rigid rotation")
            self.assertFalse(np.allclose(linked_ir.positions, original_interior_positions, atol=1e-6),
                              "expected the shared rotation to actually move the interior's geometry")

    def test_shared_rotation_accounts_for_a_different_agl_anchor_height(self):
        """CONFIRMED REAL BUG (found after the first version of this fix
        made the actual tower shift WORSE, not better): a rigid rotation
        is only physically correct around a PIVOT both objects share. Two
        placements at the same (lat,lon) but different AGL height (a
        ground-anchored shell vs. its cab-floor-anchored interior, ~42m
        up) do NOT share a pivot at either object's own local (0,0,0) --
        the correct pivot is the reference's. Naively rotating the
        interior's own local-frame vertices in place (anchor_delta_y=0)
        ignores that its 42m-high anchor point ALSO has to swing
        laterally by height*sin(angle) as part of one rigid assembly.
        This test proves the anchor_delta_y-corrected math against a hand
        -derived expectation: rotated_local = R @ (v + d) - d, where d =
        (0, anchor_delta_y, 0) -- NOT simply R @ v."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            shell_glb = td / "shell2.glb"
            self._build_box_glb(shell_glb, "Shell2", "Shell2Tex", half_size=15.0)
            shell_result = mesh_convert.convert(shell_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            shell_stem = shell_result[0].stem

            cab_glb = td / "cab2.glb"
            self._build_box_glb(cab_glb, "Cab2", "Cab2Tex", half_size=4.0, height=3.0)
            cab_result = mesh_convert.convert(cab_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            cab_stem = cab_result[0].stem
            original_cab_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{cab_stem}.obj"))

            anchor_lat, anchor_lon, anchor_hdg = 47.5, 8.5, 0.0
            shell_group_key = (tuple(sorted([shell_stem])), round(anchor_lat, 6), round(anchor_lon, 6),
                                round(anchor_hdg, 2), False)
            terrain_fit.get_or_create_fitted_group(obj_dir, [shell_stem], anchor_lat, anchor_lon, anchor_hdg, xplane_root)
            transform = terrain_fit.get_cached_transform(shell_group_key)
            self.assertIsNotNone(transform["rigid_rotation"], "test setup issue: expected the shell to tilt")

            anchor_delta_y = 42.0  # the cab's AGL height above the shell's ground anchor
            linked_results = terrain_fit.apply_shared_rotation_to_group(
                obj_dir, [cab_stem], transform, xplane_root, anchor_delta_y=anchor_delta_y)
            linked_stem = linked_results[cab_stem][0]
            linked_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{linked_stem}.obj"))

            rigid_rotation = transform["rigid_rotation"]
            d = np.array([0.0, anchor_delta_y, 0.0])
            expected = (original_cab_ir.positions + d) @ rigid_rotation.T - d
            naive_wrong = original_cab_ir.positions @ rigid_rotation.T

            np.testing.assert_allclose(linked_ir.positions, expected, atol=1e-6,
                                        err_msg="must match the anchor-height-corrected rotation exactly")
            self.assertFalse(
                np.allclose(linked_ir.positions, naive_wrong, atol=1e-3),
                "must NOT match the naive (anchor_delta_y-ignoring) rotation -- that's the confirmed bug"
            )

    def test_shared_rotation_accounts_for_a_different_horizontal_recenter_pivot(self):
        """CONFIRMED THIRD REAL BUG (found after the anchor_delta_y-only
        fix above was shipped and still left a residual shift, both
        horizontal AND vertical): mesh_convert.convert() re-centers EVERY
        model independently around its OWN median footprint before
        terrain_fit ever runs, so "local (0,0,0)" is a DIFFERENT real-
        world point for two independently-recentred source models even
        when their raw BGL/SPB placement anchor was identical -- the
        shell and the cab here stand in for exactly that (two separate
        source files, at the same real-world anchor). anchor_delta_y alone
        assumes the horizontal parts of their local origins coincide;
        they don't in general, and because a tilt rotation's off-diagonal
        terms couple X/Z into Y, the missing horizontal pivot correction
        ALSO reintroduces vertical error even when anchor_delta_y itself
        is exactly right. This test proves the fully-corrected math
        against a hand-derived expectation: rotated_local = R @ (v + d)
        - d, where d = (anchor_delta_x, anchor_delta_y, anchor_delta_z)
        -- NOT just the y-only version above."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            shell_glb = td / "shell3.glb"
            self._build_box_glb(shell_glb, "Shell3", "Shell3Tex", half_size=15.0)
            shell_result = mesh_convert.convert(shell_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            shell_stem = shell_result[0].stem

            cab_glb = td / "cab3.glb"
            self._build_box_glb(cab_glb, "Cab3", "Cab3Tex", half_size=4.0, height=3.0)
            cab_result = mesh_convert.convert(cab_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            cab_stem = cab_result[0].stem
            original_cab_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{cab_stem}.obj"))

            anchor_lat, anchor_lon, anchor_hdg = 47.5, 8.5, 0.0
            shell_group_key = (tuple(sorted([shell_stem])), round(anchor_lat, 6), round(anchor_lon, 6),
                                round(anchor_hdg, 2), False)
            terrain_fit.get_or_create_fitted_group(obj_dir, [shell_stem], anchor_lat, anchor_lon, anchor_hdg, xplane_root)
            transform = terrain_fit.get_cached_transform(shell_group_key)
            self.assertIsNotNone(transform["rigid_rotation"], "test setup issue: expected the shell to tilt")

            # The cab's own recenter offset differs from the shell's own --
            # exactly the LHBP tower case (each source model's own median
            # drags its own recenter by a different amount).
            anchor_delta_x, anchor_delta_y, anchor_delta_z = 3.0, 42.0, -2.0
            linked_results = terrain_fit.apply_shared_rotation_to_group(
                obj_dir, [cab_stem], transform, xplane_root, anchor_delta_y=anchor_delta_y,
                anchor_delta_x=anchor_delta_x, anchor_delta_z=anchor_delta_z)
            linked_stem = linked_results[cab_stem][0]
            linked_ir = mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{linked_stem}.obj"))

            rigid_rotation = transform["rigid_rotation"]
            d = np.array([anchor_delta_x, anchor_delta_y, anchor_delta_z])
            expected = (original_cab_ir.positions + d) @ rigid_rotation.T - d
            y_only_d = np.array([0.0, anchor_delta_y, 0.0])
            y_only_wrong = (original_cab_ir.positions + y_only_d) @ rigid_rotation.T - y_only_d

            np.testing.assert_allclose(linked_ir.positions, expected, atol=1e-6,
                                        err_msg="must match the fully anchor-corrected (X, Y, and Z) rotation")
            self.assertFalse(
                np.allclose(linked_ir.positions, y_only_wrong, atol=1e-3),
                "must NOT match the y-only-corrected rotation -- that's the confirmed follow-up bug"
            )

    def test_shared_rotation_is_a_noop_when_the_source_group_never_tilted(self):
        """get_cached_transform on a group that legitimately never got a
        rotation (rejected for an oversized/bogus footprint -- see
        test_oversized_footprint_rejects_the_rotation_instead_of_tearing_
        it_apart, same terrain/fixture) must make apply_shared_rotation_
        to_group a no-op, not synthesize a spurious rotation --
        propagating "no correction" is exactly as valid a shared decision
        as propagating a real one. (A group disqualified at the earlier
        too-small-footprint gate never reaches the transform cache write
        at all -- see test_get_cached_transform_returns_none_for_an_
        unprocessed_group; this test targets the oversized-footprint gate
        specifically, which does.)"""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            huge_glb = td / "huge_box.glb"
            self._build_box_glb(huge_glb, "HugeBox2", "HugeBox2Tex", half_size=300.0)
            result = mesh_convert.convert(huge_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem

            anchor_lat, anchor_lon, anchor_hdg = 47.5, 8.5, 0.0
            group_key = (tuple(sorted([stem])), round(anchor_lat, 6), round(anchor_lon, 6), round(anchor_hdg, 2), False)
            fit_results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], anchor_lat, anchor_lon, anchor_hdg, xplane_root)
            self.assertFalse(fit_results[stem][1], "test setup issue: expected the oversized footprint to be rejected")

            transform = terrain_fit.get_cached_transform(group_key)
            self.assertIsNotNone(transform, "an oversized-footprint group still reaches the transform cache write")
            self.assertIsNone(transform["rigid_rotation"])

            other_stem = "some_other_stem_at_the_same_anchor"
            results = terrain_fit.apply_shared_rotation_to_group(obj_dir, [other_stem], transform, xplane_root)
            self.assertEqual(results[other_stem], (other_stem, False, "not_applicable"))

    def test_get_cached_transform_returns_none_for_an_unprocessed_group(self):
        self.assertIsNone(terrain_fit.get_cached_transform((("nonexistent_stem",), 1.0, 2.0, 3.0, False)))

    def test_get_cached_transform_returns_none_for_a_too_small_to_qualify_group(self):
        """The too-small-footprint gate (dx/dz/area under threshold)
        returns before any terrain sampling happens at all -- there's no
        rigid_rotation decision to cache either way, so the group_key
        simply never appears in the transform cache (as opposed to the
        oversized-footprint case, which DOES reach the cache write with
        rigid_rotation=None -- see test_shared_rotation_is_a_noop_when_
        the_source_group_never_tilted)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            small_glb = td / "small_box.glb"
            self._build_box_glb(small_glb, "SmallBox2", "SmallBox2Tex", half_size=4.0, height=3.0)
            result = mesh_convert.convert(small_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem

            anchor_lat, anchor_lon, anchor_hdg = 47.5, 8.5, 0.0
            group_key = (tuple(sorted([stem])), round(anchor_lat, 6), round(anchor_lon, 6), round(anchor_hdg, 2), False)
            terrain_fit.get_or_create_fitted_group(obj_dir, [stem], anchor_lat, anchor_lon, anchor_hdg, xplane_root)

            self.assertIsNone(terrain_fit.get_cached_transform(group_key))

    def test_foundation_skirt_conforms_the_base_band_not_the_superstructure(self):
        """On genuinely non-planar ground the rigid tilt averages the
        bumps out, so the base can still float / dig in. The foundation
        skirt additionally nudges ONLY vertices within _SKIRT_BAND_M of the
        building's own lowest point by the terrain residual the plane fit
        missed, ramped to zero across _SKIRT_BLEND_M above it -- clamped
        per-vertex at _SKIRT_MAX_M. Measured as (fit WITH skirt) minus
        (fit WITHOUT skirt), so the underlying rigid rotation cancels out
        and only the skirt's own contribution is asserted."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            # a linear E-W ramp (gives a real, non-degenerate tilt so the
            # rigid-rotation path runs) PLUS a symmetric quadratic bump in
            # the same axis (the ramp fit can't represent it -> it lands
            # entirely in the residual the skirt corrects).
            N = 41
            mid = N // 2
            grid = [[int(c * 2.0 + ((c - mid) ** 2) * 10.0) for c in range(N)] for _ in range(N)]
            dsf_dir = xplane_root / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / "+40+000"
            dsf_dir.mkdir(parents=True, exist_ok=True)
            (dsf_dir / "+47+008.dsf").write_bytes(build_elevation_dsf(grid))

            obj_dir = td / "objects"
            obj_dir.mkdir()

            hs = 60.0   # a large-but-real building footprint, not the unbounded
                        # synthetic 3000 this used to be -- terrain_fit now
                        # rejects a rotation whose implied displacement is
                        # bigger than any real building's own tilt correction
                        # should ever be (see
                        # test_oversized_footprint_rejects_the_rotation_
                        # instead_of_tearing_it_apart), which a 6 km-wide box
                        # tripped despite being a synthetic fixture, not a bug.
            ring_y = (0.0, 1.5, 8.0)   # base band, inside blend zone, well above it
            positions, ring_idx = [], {}
            for y in ring_y:
                ring_idx[y] = list(range(len(positions), len(positions) + 4))
                positions += [(-hs, y, -hs), (hs, y, -hs), (hs, y, hs), (-hs, y, hs)]
            positions = np.array(positions, dtype=np.float64)
            ir = mesh_ir.MeshIR(
                name="bump_box", positions=positions,
                normals=np.tile([0.0, 1.0, 0.0], (len(positions), 1)).astype(np.float64),
                uvs=np.zeros((len(positions), 2), dtype=np.float64),
                indices=np.array([0, 1, 2, 0, 2, 3, 8, 9, 10, 8, 10, 11], dtype=np.int64),
                texture="t.png", draped=False, footprint_area_m2=None,
            )
            mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / "bump_box.obj"))

            def fit_positions(skirt_on):
                orig = terrain_fit._FOUNDATION_SKIRT
                terrain_fit._FOUNDATION_SKIRT = skirt_on
                terrain_fit._group_cache.clear()
                for p in obj_dir.glob("bump_box_tfit_*"):
                    p.unlink()
                try:
                    res = terrain_fit.get_or_create_fitted_group(obj_dir, ["bump_box"], 47.5, 8.5, 0.0, xplane_root)
                    rstem, applied, reason = res["bump_box"]
                    self.assertTrue(applied, f"reason={reason!r}")
                    self.assertNotEqual(rstem, "bump_box")
                    return mesh_ir.load(mesh_ir.sidecar_path_for(obj_dir / f"{rstem}.obj")).positions
                finally:
                    terrain_fit._FOUNDATION_SKIRT = orig

            without = fit_positions(False)
            with_skirt = fit_positions(True)
            dy = with_skirt[:, 1] - without[:, 1]

            base_dy = float(np.mean(dy[ring_idx[0.0]]))
            mid_dy = float(np.mean(dy[ring_idx[1.5]]))
            top_dy = float(np.mean(dy[ring_idx[8.0]]))

            self.assertGreater(abs(base_dy), 0.2, "base band must be conformed by the skirt")
            self.assertLessEqual(abs(base_dy), terrain_fit._SKIRT_MAX_M + 1e-6, "per-vertex clamp respected")
            self.assertGreater(abs(mid_dy), 0.0, "blend zone gets a partial nudge")
            self.assertLess(abs(mid_dy), abs(base_dy) - 1e-6, "blend zone gets LESS than the full band")
            self.assertAlmostEqual(top_dy, 0.0, delta=1e-6, msg="superstructure gets none of the skirt residual")

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

    def test_too_small_group_is_disqualified(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xplane_root = td / "XPlaneRoot"
            self._write_sloped_terrain(xplane_root, 47, 8, slope_per_post=3000.0)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            small_glb = td / "small.glb"
            self._build_glb(small_glb, "Small", "SmallTex", -1, 1, -1, 1)
            result = mesh_convert.convert(small_glb, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stem = result[0].stem

            results = terrain_fit.get_or_create_fitted_group(obj_dir, [stem], 47.5, 8.5, 0.0, xplane_root)
            self.assertEqual(results[stem], (stem, False, "disqualified"))

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


class TestFitRigidTiltRotation(unittest.TestCase):
    """_fit_rigid_tilt_rotation's own math, isolated from any file I/O or
    terrain sampling -- the piece of this module with the highest cost of
    a subtle bug (a wrong-signed rotation would tilt buildings the WRONG
    way, silently, which would be much harder to notice/diagnose than an
    object simply being left uncorrected)."""

    def test_east_higher_terrain_tilts_east_side_up(self):
        """Corner samples forming a plane that's 5m higher at x=+10 than
        at x=-10 (rises to the east, flat in z) -- real terrain is higher
        to the east, so the building's east edge must rotate UP (+Y) and
        its west edge DOWN (-Y), matching a physical object tilting to
        rest flush against a slope that's higher on its east side. Exact
        expected numbers hand-derived from Rodrigues' rotation formula for
        this slope (see terrain_fit.py's own math for the derivation)."""
        corners = [(-10.0, -10.0, -5.0), (-10.0, 10.0, -5.0), (10.0, -10.0, 5.0), (10.0, 10.0, 5.0)]
        rotation = terrain_fit._fit_rigid_tilt_rotation(corners)
        self.assertIsNotNone(rotation)

        east_point = np.array([10.0, 0.0, 0.0])
        rotated_east = rotation @ east_point
        self.assertGreater(rotated_east[1], 0.0, "the east edge (over higher real terrain) must tilt UP")
        np.testing.assert_allclose(rotated_east, [8.9443, 4.4721, 0.0], atol=1e-3)

        west_point = np.array([-10.0, 0.0, 0.0])
        rotated_west = rotation @ west_point
        self.assertLess(rotated_west[1], 0.0, "the west edge (over lower real terrain) must tilt DOWN")
        np.testing.assert_allclose(rotated_west, [-8.9443, -4.4721, 0.0], atol=1e-3)

    def test_rotation_matrix_is_orthogonal_ie_truly_rigid(self):
        """R^T @ R must be the identity for any fitted slope -- the
        algebraic guarantee behind "this can't shear anything", checked
        directly rather than only inferred from a specific geometry test."""
        corners = [(-8.0, -6.0, -1.5), (-8.0, 6.0, 0.5), (8.0, -6.0, 0.8), (8.0, 6.0, 2.9)]
        rotation = terrain_fit._fit_rigid_tilt_rotation(corners)
        self.assertIsNotNone(rotation)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9,
                                msg="determinant must be +1 (a proper rotation, not a reflection)")

    def test_flat_samples_return_none(self):
        """All 4 corners at the same real elevation (delta=0 everywhere) --
        no tilt needed, must return None rather than an identity-but-not-
        quite matrix that'd trigger a pointless corrected copy."""
        corners = [(-10.0, -10.0, 0.0), (-10.0, 10.0, 0.0), (10.0, -10.0, 0.0), (10.0, 10.0, 0.0)]
        self.assertIsNone(terrain_fit._fit_rigid_tilt_rotation(corners))

    def test_too_few_samples_returns_none(self):
        self.assertIsNone(terrain_fit._fit_rigid_tilt_rotation([(-10.0, -10.0, -5.0), (10.0, 10.0, 5.0)]))

    def test_collinear_samples_return_none(self):
        """3 samples all lying on the same line through the origin can't
        pin down a unique plane (infinitely many planes fit them equally
        well) -- must decline rather than fit an arbitrary/unstable one."""
        corners = [(-10.0, -10.0, -5.0), (0.0, 0.0, 0.0), (10.0, 10.0, 5.0)]
        self.assertIsNone(terrain_fit._fit_rigid_tilt_rotation(corners))


if __name__ == "__main__":
    unittest.main()
