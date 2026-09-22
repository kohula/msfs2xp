"""
mesh_convert.convert() no longer emits TILTED at all (terrain_fit.py's
uniform vertical shift replaces it for every rigid object regardless of
size -- see terrain_fit.py's own module docstring for why a rotation
could never fix the more common anchor-elevation-offset case). This file
used to pin an empirically-discovered invariant -- TILTED and per-node
ATTR_draped must never coexist in one file, confirmed against real
converted taxi signs where mixing them made the object disappear
entirely in X-Plane -- which is now trivially true by construction (no
code path writes TILTED at all), so that specific test was removed.

What's still pinned here is the node-level flatness threshold's
asymmetric-risk rationale:
a node wrongly classified flat/draped costs nothing (ATTR_draped
re-projects onto terrain, discarding authored Y, per X-Plane's own spec),
while a node wrongly left rigid visibly floats/sinks on sloped terrain --
so the per-node threshold (0.90) is deliberately more lenient than the
file-wide TILTED threshold (0.98).
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert


def _box_walls_and_roof(hw=0.05):
    """A thin vertical post/box: 4 wall quads (8 tris, clearly vertical)."""
    px, pz = 0.0, 0.0
    verts = [(px - hw, 0, pz - hw), (px + hw, 0, pz - hw), (px + hw, 2.0, pz - hw), (px - hw, 2.0, pz - hw),
             (px - hw, 0, pz + hw), (px + hw, 0, pz + hw), (px + hw, 2.0, pz + hw), (px - hw, 2.0, pz + hw)]
    quad_faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 7, 3), (1, 5, 6, 2)]
    tris = []
    for a, b, c, d in quad_faces:
        tris += [(a, b, c), (a, c, d)]
    return verts, tris


class TestFlatnessTiltedExclusivity(unittest.TestCase):
    def test_flat_node_inside_an_otherwise_rigid_file_still_drapes(self):
        """Flatness/draping classification is FILE-WIDE (see MatBuilder's
        own is_draped formula) -- a material literally named "decal" is
        the one deliberate exception, via the separate is_decal check
        (`"decal" in raw_mat_name.lower()`), always draped regardless of
        the file-wide verdict. This building+decal fixture exercises
        exactly that combination: the decal-named material must come out
        as its own separate, draped (ATTR_draped) object, while the
        building's own rigid geometry stays rigid (not draped) -- the one
        real-world case where a single glTF file's own nodes can still
        split into both a rigid and a draped output file side by side."""
        b = GltfBuilder()
        tex = b.add_image_data_uri((150, 150, 150, 255))
        texi = b.add_texture(tex)
        building_mat = b.add_material("BuildingMat", base_color_texture_index=texi)
        decal_mat = b.add_material("DecalMat", base_color_texture_index=texi)

        bx = [(-5, 0, -5), (5, 0, -5), (5, 0, 5), (-5, 0, 5), (-5, 6, -5), (5, 6, -5), (5, 6, 5), (-5, 6, 5)]
        wall_tris = []
        for a, c, d, e in [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
            wall_tris += [(a, c, d), (a, d, e)]
        roof_tris = [(4, 5, 6), (4, 6, 7)]
        building_indices = [i for tri in (wall_tris + roof_tris) for i in tri]
        building_mesh = b.add_mesh(
            bx, building_indices,
            normals=[(0.0, 1.0, 0.0)] * len(bx), uvs=[(0.0, 0.0)] * len(bx), material_index=building_mat,
        )
        b.add_node(mesh_index=building_mesh, name="Building")

        # A separate node/material -- its own perfectly flat quad, well
        # under the 0.98 file-wide fraction on its own (the building's 8
        # non-flat wall triangles dominate the file-wide aggregate), but
        # clear above the 0.90 per-node threshold by itself.
        decal_verts = [(10.0, 0.0, 10.0), (12.0, 0.0, 10.0), (12.0, 0.0, 12.0), (10.0, 0.0, 12.0)]
        decal_indices = [0, 1, 2, 0, 2, 3]
        decal_mesh = b.add_mesh(
            decal_verts, decal_indices,
            normals=[(0.0, 1.0, 0.0)] * 4, uvs=[(0.0, 0.0)] * 4, material_index=decal_mat,
        )
        b.add_node(mesh_index=decal_mesh, name="FloorDecal")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "building_with_decal.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            building_obj = next(p for p in result if "BuildingMat" in p.name)
            decal_obj = next(p for p in result if "DecalMat" in p.name)

            building_text = building_obj.read_text(encoding="utf-8")
            decal_text = decal_obj.read_text(encoding="utf-8")

            self.assertNotIn("ATTR_draped", building_text, "the rigid building must not be draped")

            self.assertIn("ATTR_draped", decal_text, "the individually-flat decal node must now drape")
            self.assertNotIn("TILTED", decal_text.splitlines(),
                              "the flat decal must never get TILTED -- the exact combination that made "
                              "a real converted object disappear entirely in X-Plane")

    def test_elevated_decal_material_stays_rigid_not_draped(self):
        """CONFIRMED REAL BUG this pins: a "decal"-named/ASOBO_material_
        decal-tagged material used to be draped unconditionally on name
        alone, with no elevation check at all (unlike is_near_ground_flat's
        own check on the same kind of material one call below). MSFS uses
        that same BLEND-mode decal material type for more than ground-
        level stains/markings -- a rooftop weathering/grime overlay meant
        to stay coincident with its own rigid roof is authored the same
        way. Confirmed against a real converted LHBP building: a
        "roof_decal" material at genuine roof height was draped flat onto
        the ground, far below the roof it was meant to sit on. This
        fixture mirrors that: a decal-named quad sitting AT the building's
        own roof height (not at its ground level) must stay rigid.

        Deliberately NO explicit ground-level floor geometry here -- the
        building's only flat triangles are its roof (2 tris @ y=6), same
        as the decal. This is what makes the fixture actually discriminate
        the real fix (comparing against file_min_height, the lowest CLEAN
        VERTEX in the file -- found here from the walls' own non-flat
        bottom edge, still @ y=0) from the superseded, buggier approach
        (comparing against file_reference_height, the most vertex-heavy
        FLAT band -- which without a ground-level floor would itself
        resolve to y=6, the roof, making the roof_decal wrongly look
        "close to ground" and stay draped)."""
        b = GltfBuilder()
        tex = b.add_image_data_uri((150, 150, 150, 255))
        texi = b.add_texture(tex)
        building_mat = b.add_material("BuildingMat", base_color_texture_index=texi)
        decal_mat = b.add_material("roof_decal", base_color_texture_index=texi)

        bx = [(-5, 0, -5), (5, 0, -5), (5, 0, 5), (-5, 0, 5), (-5, 6, -5), (5, 6, -5), (5, 6, 5), (-5, 6, 5)]
        wall_tris = []
        for a, c, d, e in [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
            wall_tris += [(a, c, d), (a, d, e)]
        roof_tris = [(4, 5, 6), (4, 6, 7)]
        building_indices = [i for tri in (wall_tris + roof_tris) for i in tri]
        building_mesh = b.add_mesh(
            bx, building_indices,
            normals=[(0.0, 1.0, 0.0)] * len(bx), uvs=[(0.0, 0.0)] * len(bx), material_index=building_mat,
        )
        b.add_node(mesh_index=building_mesh, name="Building")

        # A "roof_decal" quad sitting AT the roof's own height (y=6, same
        # as the building's roof triangles above -- not at the file's
        # ground level, y=0) -- the real-world case this fix targets.
        decal_verts = [(-4.0, 6.0, -4.0), (4.0, 6.0, -4.0), (4.0, 6.0, 4.0), (-4.0, 6.0, 4.0)]
        decal_indices = [0, 1, 2, 0, 2, 3]
        decal_mesh = b.add_mesh(
            decal_verts, decal_indices,
            normals=[(0.0, 1.0, 0.0)] * 4, uvs=[(0.0, 0.0)] * 4, material_index=decal_mat,
        )
        b.add_node(mesh_index=decal_mesh, name="RoofDecal")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "building_with_roof_decal.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            decal_obj = next(p for p in result if "roof_decal" in p.name)
            decal_text = decal_obj.read_text(encoding="utf-8")
            self.assertNotIn("ATTR_draped", decal_text,
                              "a decal at genuine roof height must stay rigid, not get projected onto "
                              "the ground far below its authored position")

    def test_mostly_flat_node_with_minor_embossing_still_drapes(self):
        """Flatness/draping is a FILE-WIDE verdict: a node with a small
        amount of embossed/raised detail (here, 10 flat tris + 1 near-
        vertical sliver, ~91% flat on its own) still drapes correctly as
        long as the FILE as a whole clears the 0.98 flat-fraction bar --
        confirmed by pairing it with a large, purely-flat "base fill" node
        (60 tris) that keeps the combined file-wide fraction at ~98.6%,
        comfortably above threshold. Real MSFS airport marking files are
        exactly this shape: mostly pure flat decals (many nodes near 100%
        flat) with an occasional embossed one mixed in -- the embossed
        node's own minor non-flat detail doesn't disqualify the file."""
        b = GltfBuilder()
        tex = b.add_image_data_uri((180, 180, 180, 255))
        texi = b.add_texture(tex)
        fill_mat = b.add_material("FillMat", base_color_texture_index=texi)
        emboss_mat = b.add_material("EmbossedMat", base_color_texture_index=texi)

        # Base fill: 30 quads (60 triangles), all perfectly flat -- keeps
        # the file-wide aggregate comfortably above 0.98 even with the
        # embossed node's 1-in-11 non-flat triangle mixed in.
        fill_verts, fill_tris = [], []
        for i in range(30):
            base = len(fill_verts)
            x0 = -100.0 + i * 6.0
            fill_verts += [(x0, 0.0, -50.0), (x0 + 5.0, 0.0, -50.0), (x0 + 5.0, 0.0, 50.0), (x0, 0.0, 50.0)]
            fill_tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        fill_indices = [i for tri in fill_tris for i in tri]
        fill_mesh = b.add_mesh(
            fill_verts, fill_indices,
            normals=[(0.0, 1.0, 0.0)] * len(fill_verts),
            uvs=[(0.0, 0.0)] * len(fill_verts),
            material_index=fill_mat,
        )
        b.add_node(mesh_index=fill_mesh, name="BaseFill")

        verts, tris = [], []
        for i in range(5):
            base = len(verts)
            x0 = -40.0 + i * 20.0
            verts += [(x0, 0.0, -20.0), (x0 + 15.0, 0.0, -20.0), (x0 + 15.0, 0.0, 20.0), (x0, 0.0, 20.0)]
            tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        # One near-vertical sliver triangle (a raised embossed edge).
        vbase = len(verts)
        verts += [(-40.0, 0.0, -20.0), (-40.0, 0.30, -20.0), (-39.9, 0.0, -20.0)]
        tris.append((vbase, vbase + 1, vbase + 2))

        indices = [i for tri in tris for i in tri]
        mesh = b.add_mesh(
            verts, indices,
            normals=[(0.0, 1.0, 0.0)] * len(verts),
            uvs=[(0.0, 0.0)] * len(verts),
            material_index=emboss_mat,
        )
        b.add_node(mesh_index=mesh, name="EmbossedMarking")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "embossed.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            embossed_obj = next(p for p in result if "Embossed" in p.name)
            text = embossed_obj.read_text(encoding="utf-8")
            self.assertNotIn("TILTED", text.splitlines(), "test setup issue: file-wide fraction should stay above 0.98")
            self.assertIn("ATTR_draped", text, "a 91%-flat node with minor embossing should still drape")

    def test_tiny_sign_skips_tilted_entirely(self):
        """A genuinely tiny non-flat object (a real small taxi sign: panel
        + post, horizontal radius well under 3m) must skip TILTED
        entirely rather than getting X-Plane's single-point-sampled
        rotation -- real terrain slope can't meaningfully vary across a
        footprint this small, while a locally noisy terrain-normal sample
        (a nearby DSF seam) is exactly as likely regardless of the
        object's own size, and a small/thin panel is far more visually
        sensitive to a wrong rotation (edge-on to the camera reads as
        "gone") than a large one -- plausibly the mechanism behind real
        taxi signs reported as intermittently not appearing at all."""
        b = GltfBuilder()
        tex = b.add_image_data_uri((200, 200, 200, 255))
        texi = b.add_texture(tex)
        panel_mat = b.add_material("PanelMat", base_color_texture_index=texi)
        post_mat = b.add_material("PostMat", base_color_texture_index=texi)

        panel_verts = [(-0.5, 2.0, 0.0), (0.5, 2.0, 0.0), (0.5, 2.4, 0.0), (-0.5, 2.4, 0.0)]
        panel_indices = [0, 1, 2, 0, 2, 3]
        panel_mesh = b.add_mesh(
            panel_verts, panel_indices,
            normals=[(0.0, 0.0, -1.0)] * 4, uvs=[(0.0, 0.0)] * 4, material_index=panel_mat,
        )
        post_verts, post_tris = _box_walls_and_roof()
        post_indices = [i for tri in post_tris for i in tri]
        post_mesh = b.add_mesh(
            post_verts, post_indices,
            normals=[(1.0, 0.0, 0.0)] * len(post_verts), uvs=[(0.0, 0.0)] * len(post_verts), material_index=post_mat,
        )
        b.add_node(mesh_index=panel_mesh, name="SignPanel")
        b.add_node(mesh_index=post_mesh, name="SignPost")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "tiny_sign.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            for p in result:
                self.assertNotIn("TILTED", p.read_text(encoding="utf-8").splitlines(),
                                  f"{p.name}: a tiny sign-scale object should skip TILTED entirely")

    def test_genuine_3d_building_stays_rigid(self):
        """A real building (20% flat: roof only) must never be flattened/
        draped, regardless of the lenient per-node threshold."""
        b = GltfBuilder()
        tex = b.add_image_data_uri((150, 150, 150, 255))
        texi = b.add_texture(tex)
        mat = b.add_material("BuildingMat", base_color_texture_index=texi)

        bx = [(-5, 0, -5), (5, 0, -5), (5, 0, 5), (-5, 0, 5), (-5, 6, -5), (5, 6, -5), (5, 6, 5), (-5, 6, 5)]
        wall_tris = []
        for a, c, d, e in [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
            wall_tris += [(a, c, d), (a, d, e)]
        roof_tris = [(4, 5, 6), (4, 6, 7)]
        all_tris = wall_tris + roof_tris
        indices = [i for tri in all_tris for i in tri]
        mesh = b.add_mesh(
            bx, indices,
            normals=[(0.0, 1.0, 0.0)] * len(bx),
            uvs=[(0.0, 0.0)] * len(bx),
            material_index=mat,
        )
        b.add_node(mesh_index=mesh, name="Building")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "building.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            text = result[0].read_text(encoding="utf-8")
            self.assertNotIn("ATTR_draped", text, "a genuine 3D building must never be draped/flattened")

    def _build_ground_layer_glb(self):
        """Mirrors the confirmed real-world EGLC case: one file mixing a
        genuinely 3D building (non-flat walls, vetoes the file-wide 0.98
        verdict) with THREE separate ground-level materials -- a large
        dominant GroundMat at y=0 (sets the file's own reference height),
        a non-"decal"-named PaverMat at y=1.5 (mirrors the real confirmed
        SmallTiles/ConcreteTile case: individually ~100% flat, but a
        small nonzero baked authoring offset), and a RoofMat at y=6.0
        (individually just as flat as PaverMat, but at a real elevated
        height -- must NOT be treated as ground-level just because it's
        flat)."""
        b = GltfBuilder()
        tex = b.add_image_data_uri((160, 160, 160, 255))
        texi = b.add_texture(tex)
        building_mat = b.add_material("BuildingMat", base_color_texture_index=texi)
        ground_mat = b.add_material("GroundMat", base_color_texture_index=texi)
        paver_mat = b.add_material("PaverMat", base_color_texture_index=texi)
        roof_mat = b.add_material("RoofMat", base_color_texture_index=texi)

        wall_verts, wall_tris = _box_walls_and_roof(hw=4.0)
        wall_indices = [i for tri in wall_tris for i in tri]
        wall_mesh = b.add_mesh(
            wall_verts, wall_indices,
            normals=[(1.0, 0.0, 0.0)] * len(wall_verts), uvs=[(0.0, 0.0)] * len(wall_verts),
            material_index=building_mat,
        )
        b.add_node(mesh_index=wall_mesh, name="Building")

        # Large dominant ground fill at y=0 -- several quads so its vertex
        # support clearly outweighs the two single-quad candidates below,
        # keeping the file's own reference height anchored at ~0.
        ground_verts, ground_tris = [], []
        for i in range(4):
            base = len(ground_verts)
            x0 = -100.0 + i * 20.0
            ground_verts += [(x0, 0.0, -50.0), (x0 + 15.0, 0.0, -50.0), (x0 + 15.0, 0.0, 50.0), (x0, 0.0, 50.0)]
            ground_tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        ground_indices = [i for tri in ground_tris for i in tri]
        ground_mesh = b.add_mesh(
            ground_verts, ground_indices,
            normals=[(0.0, 1.0, 0.0)] * len(ground_verts), uvs=[(0.0, 0.0)] * len(ground_verts),
            material_index=ground_mat,
        )
        b.add_node(mesh_index=ground_mesh, name="GroundFill")

        # A grid of small individual tile quads (20 of them, 40 flat tris)
        # plus a couple of small tilted "seam" triangles between them (2
        # non-flat tris) -- NOT an artificially perfect 100%-flat single
        # quad. Real tile/paver geometry has genuine small edge/seam
        # detail; the confirmed real EGLC material this fixture mirrors
        # (ini_GP_GEN_SmallTiles_4m_01) measures 0.9577 flat, not 1.0 --
        # an earlier version of this fixture used a single perfect quad,
        # which passed even the old, too-strict 0.98 per-material
        # threshold and gave false confidence that threshold was correct.
        # This fixture's ~40/42 = 0.952 fraction sits in the same real
        # range, so it only passes at 0.90, not 0.98 -- pinning the actual
        # regression the real threshold had to be tuned against.
        paver_verts, paver_tris = [], []
        for i in range(20):
            base = len(paver_verts)
            x0 = 10.0 + i * 0.5
            paver_verts += [(x0, 1.5, 10.0), (x0 + 0.4, 1.5, 10.0), (x0 + 0.4, 1.5, 10.4), (x0, 1.5, 10.4)]
            paver_tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        seam_base = len(paver_verts)
        paver_verts += [(10.0, 1.5, 10.4), (10.2, 1.7, 10.4), (10.2, 1.5, 10.6)]
        paver_tris += [(seam_base, seam_base + 1, seam_base + 2)]
        seam_base2 = len(paver_verts)
        paver_verts += [(10.5, 1.5, 10.4), (10.7, 1.7, 10.4), (10.7, 1.5, 10.6)]
        paver_tris += [(seam_base2, seam_base2 + 1, seam_base2 + 2)]
        paver_indices = [i for tri in paver_tris for i in tri]
        paver_mesh = b.add_mesh(
            paver_verts, paver_indices,
            normals=[(0.0, 1.0, 0.0)] * len(paver_verts), uvs=[(0.0, 0.0)] * len(paver_verts),
            material_index=paver_mat,
        )
        b.add_node(mesh_index=paver_mesh, name="PaverPatch")

        roof_verts = [(20.0, 6.0, 20.0), (22.0, 6.0, 20.0), (22.0, 6.0, 22.0), (20.0, 6.0, 22.0)]
        roof_indices = [0, 1, 2, 0, 2, 3]
        roof_mesh = b.add_mesh(
            roof_verts, roof_indices,
            normals=[(0.0, 1.0, 0.0)] * 4, uvs=[(0.0, 0.0)] * 4, material_index=roof_mat,
        )
        b.add_node(mesh_index=roof_mesh, name="RoofPatch")

        return b.build()

    def test_near_ground_flat_material_stays_rigid_not_dropped(self):
        """The per-material near-ground-flat DETECTION (convert()'s
        builder.is_near_ground_flat) is informational only: a non-"decal"-
        named material that is individually ~100% flat AND close to the
        file's own ground-level reference used to be DROPPED from the
        output entirely, even though the file-wide verdict fails because
        of the building's genuinely non-flat walls. Reverted per a
        real-world comparison against another converter's output for the
        same EGLC content (pavement/rail-ballast detail near the train):
        its tool keeps this as ordinary rigid geometry instead of omitting
        it, and that reads fine in-sim even if it ends up floating a
        little proud of the ground -- unlike our old DROP, which removed
        real content outright. Confirmed real case this pins: EGLC's
        SmallTiles/ConcreteTile materials, baked ~1.5m off true ground
        level -- this mechanism used to drape them, then dropped them;
        now it leaves them rigid (not draped, not dropped, still eligible
        for terrain_fit's vertical shift like any other rigid object)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "ground_layer.glb"
            glb_path.write_bytes(self._build_ground_layer_glb())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            building_obj = next(p for p in result if "BuildingMat" in p.name)
            building_text = building_obj.read_text(encoding="utf-8")
            self.assertNotIn("ATTR_draped", building_text, "the rigid building must not be draped")

            paver_obj = next((p for p in result if "PaverMat" in p.name), None)
            self.assertIsNotNone(paver_obj,
                                  "a non-decal material that only qualifies via the near-ground-flat "
                                  "fallback must still be written to the output, not dropped")
            paver_text = paver_obj.read_text(encoding="utf-8")
            self.assertNotIn("ATTR_draped", paver_text,
                              "it must stay rigid (not draped), since MSFS's own stacking order "
                              "for this content can't be recovered")
            self.assertTrue(paver_text.strip())

            # CONFIRMED REAL BUG this pins: staying rigid instead of
            # dropped was never enough on its own -- PaverMat is baked at
            # y=1.5 (see the fixture's own docstring), and nothing used to
            # correct that baked-authoring offset, so it kept rendering
            # floating 1.5m above the real ground (y=0, GroundMat/the
            # walls' own base) even after it stopped being dropped. Median,
            # not every vertex: the fixture's own deliberately-tilted seam
            # triangles (see its docstring) have one corner intentionally
            # 0.2m off the flat tile quads' own level, and the snap is a
            # uniform shape-preserving translation, so that relative offset
            # is correctly preserved after the shift -- only the dominant
            # (flat-tile-quad) level is expected to land exactly on 0.0.
            paver_ys = [float(line.split()[2]) for line in paver_text.splitlines() if line.startswith("VT")]
            self.assertTrue(paver_ys)
            self.assertAlmostEqual(float(np.median(paver_ys)), 0.0, places=3,
                                    msg="the near-ground-flat snap must bring this material's baked 1.5m "
                                        "offset down to the object's real local ground level (0.0), not "
                                        "leave it floating at its originally authored height")

    def test_elevated_flat_material_stays_rigid_not_dropped(self):
        """The near-ground-flat DETECTION must NOT fire for a flat
        material that sits meters above the file's own ground level (a
        roof, a bridge deck top) -- flatness alone isn't the signal,
        proximity to ground level is what distinguishes "floating
        pavement" from "a real elevated flat surface that must stay
        rigid". This is exactly the failure mode an earlier, more
        aggressive per-material attempt hit this session (no ground-
        proximity check at all, wrongly flattened chairs/glass/rooftops)
        -- this test pins that it can't happen again via this narrower
        mechanism. This must stay present in the output (rigid) either
        way -- a true near-ground-flat match no longer gets dropped
        either, see test_near_ground_flat_material_stays_rigid_not_dropped."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "ground_layer.glb"
            glb_path.write_bytes(self._build_ground_layer_glb())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            roof_obj = next(p for p in result if "RoofMat" in p.name)
            roof_text = roof_obj.read_text(encoding="utf-8")

            self.assertNotIn("ATTR_draped", roof_text,
                             "an individually-flat but elevated material must not drape just because it's flat")
            self.assertTrue(roof_text.strip(), "it must stay rigid instead, and stay present in the output")


if __name__ == "__main__":
    unittest.main()
