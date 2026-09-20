"""
Pins the empirically-discovered invariant: TILTED and per-node ATTR_draped
must never coexist in one file. Confirmed against real converted taxi
signs -- mixing them (a TILTED file, correctly TILTED since the file's
post/frame geometry makes it well under the file-wide flat threshold,
that ALSO had one of its own nodes independently qualify as flat/draped
per-node) made the object disappear entirely in X-Plane, not just
partially misrender.

Also pins the node-level flatness threshold's asymmetric-risk rationale:
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
    def _build_sign_glb(self):
        """A realistic gantry-style taxi sign: a large, near-perfectly-flat
        PANEL (one material/node) + a genuinely 3-D post/frame (a separate
        material/node) -- the file-wide fraction stays well under 0.98
        (correctly TILTED), while the panel node ALONE would clear even a
        strict per-node threshold on its own.

        Deliberately sized with a horizontal radius ABOVE
        convert()'s own tiny-object TILT exemption threshold (3m) --
        below that, a real small taxi sign now legitimately skips TILTED
        entirely (see convert.py's own comment), which would make this
        fixture no longer exercise the TILTED+draped exclusivity path
        this test exists to pin at all."""
        b = GltfBuilder()
        tex = b.add_image_data_uri((200, 200, 200, 255))
        texi = b.add_texture(tex)
        panel_mat = b.add_material("PanelMat", base_color_texture_index=texi)
        post_mat = b.add_material("PostMat", base_color_texture_index=texi)

        panel_verts, panel_tris = [], []
        for i in range(3):
            base = len(panel_verts)
            x0 = -3.6 + i * 2.4
            panel_verts += [(x0, 2.0, 0.0), (x0 + 2.1, 2.0, 0.0), (x0 + 2.1, 2.6, 0.0), (x0, 2.6, 0.0)]
            panel_tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        panel_indices = [i for tri in panel_tris for i in tri]
        panel_mesh = b.add_mesh(
            panel_verts, panel_indices,
            normals=[(0.0, 0.0, -1.0)] * len(panel_verts),
            uvs=[(0.0, 0.0)] * len(panel_verts),
            material_index=panel_mat,
        )

        post_verts, post_tris = _box_walls_and_roof()
        post_indices = [i for tri in post_tris for i in tri]
        post_mesh = b.add_mesh(
            post_verts, post_indices,
            normals=[(1.0, 0.0, 0.0)] * len(post_verts),
            uvs=[(0.0, 0.0)] * len(post_verts),
            material_index=post_mat,
        )

        b.add_node(mesh_index=panel_mesh, name="SignPanel")
        b.add_node(mesh_index=post_mesh, name="SignPost")
        return b.build()

    def test_tilted_and_draped_never_coexist_in_one_file(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "taxi_sign.glb"
            glb_path.write_bytes(self._build_sign_glb())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            any_tilted = False
            for p in result:
                text = p.read_text(encoding="utf-8")
                is_tilted = "TILTED" in text.splitlines()
                has_draped = "ATTR_draped" in text
                any_tilted = any_tilted or is_tilted
                self.assertFalse(
                    is_tilted and has_draped,
                    f"{p.name} has BOTH TILTED and ATTR_draped -- the exact combination "
                    f"that made real converted taxi signs disappear in X-Plane"
                )
            self.assertTrue(any_tilted, "test setup issue: expected this fixture's file-wide fraction to trigger TILTED")

    def test_flat_node_inside_an_otherwise_tilted_file_still_drapes(self):
        """Flatness/draping classification is FILE-WIDE (see MatBuilder's
        own is_draped formula) -- a material literally named "decal" is
        the one deliberate exception, via the separate is_decal check
        (`"decal" in raw_mat_name.lower()`), always draped regardless of
        the file-wide verdict. This building+decal fixture exercises
        exactly that combination: the decal-named material must come out
        as its own separate, draped (ATTR_draped, no TILTED) object, while
        the building's own rigid geometry still correctly gets TILTED --
        and, critically, NEITHER ends up with both markers on the same
        file (the exact combination that made a real converted object
        disappear entirely, see the test above). This is the one
        real-world case where a single file can still carry both a TILTED
        and a draped builder side by side; the write site's own
        "file_apply_tilted and not builder_is_draped" guard (not just
        "file_apply_tilted") is what keeps it safe."""
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

            self.assertIn("TILTED", building_text.splitlines(), "the rigid building must still get TILTED")
            self.assertNotIn("ATTR_draped", building_text, "the rigid building must not be draped")

            self.assertIn("ATTR_draped", decal_text, "the individually-flat decal node must now drape")
            self.assertNotIn("TILTED", decal_text.splitlines(),
                              "the flat decal must never get TILTED -- the exact combination that made "
                              "a real converted object disappear entirely in X-Plane")

    def test_mostly_flat_node_with_minor_embossing_still_drapes(self):
        """Flatness/draping is a PER-NODE verdict (this module's own
        docstring: the 0.90 per-node threshold): a node with a small
        amount of embossed/raised detail (here, 10 flat tris + 1 near-
        vertical sliver, ~91% flat on its own) still drapes correctly on
        its OWN merit, clearing the lenient 0.90 per-node bar -- paired
        here with a large, purely-flat "base fill" node (60 tris) only to
        keep the combined FILE-WIDE fraction at ~98.6% too, so this
        fixture can't be accidentally passing via TILTED's separate
        file-wide 0.98 threshold instead of the per-node one this test
        actually exists to pin. Real MSFS airport marking files are
        exactly this shape: mostly pure flat decals (many nodes near 100%
        flat) with an occasional embossed one mixed in -- the embossed
        node's own minor non-flat detail doesn't disqualify IT, let alone
        any sibling node in the same file."""
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

    def test_flat_majority_drapes_despite_one_non_flat_primitive_sharing_its_node(self):
        """CONFIRMED REAL BUG on a live EGLC package: one glTF bundles a
        large, genuinely flat apron/taxiway ground-poly (concrete tiles,
        transitions, every painted marking -- ~25 distinct materials)
        together with ONE real-3D slope/transition material -- ALL as
        separate PRIMITIVES inside a SINGLE node's mesh (real MSFS ground-
        poly export shape: one node, one mesh, many materials as
        primitives -- not one node per material). The file-wide fraction
        came out at 89.70% -- short of the 0.98 TILTED bar, which is
        correct (the file does have real 3D content), but the OLD code
        also used that same file-wide verdict to gate draping, so the
        ENTIRE file -- including every individually-flat pavement
        material -- was left rigid. A rigid object never gets X-Plane's
        native per-vertex terrain draping, so it doesn't track the
        compiled terrain precisely -- this is the confirmed mechanism
        behind reported "pavement floating above the ground and
        glitching". A first fix keyed per NODE alone (matching this real
        file's shape, where everything sits in one node) would have been
        just as useless as file-wide -- the key has to be per (node,
        MATERIAL) for a flat majority sharing a node with one non-flat
        material to be told apart at all. Each flat pavement material
        must drape on its own merit; only the genuinely-3D one stays
        rigid."""
        b = GltfBuilder()
        tex = b.add_image_data_uri((170, 170, 170, 255))
        texi = b.add_texture(tex)

        # Several separate, perfectly flat pavement materials -- stands in
        # for concrete tiles / transitions / markings -- PLUS one
        # genuinely-3D slope/transition material (reusing the same near-
        # vertical-wall fixture other tests in this file already rely on
        # to fail BOTH the per-triangle height-variance check and the
        # face-normal-near-vertical check), ALL as primitives of ONE mesh
        # attached to a SINGLE node -- the real EGLC file's exact shape.
        pave_names = ("ConcreteTile", "Transitions", "WhiteMarking", "Grunge")
        primitives = []
        for i, name in enumerate(pave_names):
            mat = b.add_material(name, base_color_texture_index=texi)
            x0 = -60.0 + i * 30.0
            verts = [(x0, 0.0, -20.0), (x0 + 25.0, 0.0, -20.0), (x0 + 25.0, 0.0, 20.0), (x0, 0.0, 20.0)]
            indices = [0, 1, 2, 0, 2, 3]
            primitives.append({
                "attributes": {
                    "POSITION": b.add_positions(verts),
                    "NORMAL": b.add_normals([(0.0, 1.0, 0.0)] * 4),
                    "TEXCOORD_0": b.add_uvs([(0.0, 0.0)] * 4),
                },
                "indices": b.add_indices(indices),
                "material": mat,
            })

        slope_mat = b.add_material("Slope", base_color_texture_index=texi)
        slope_verts, slope_tris = _box_walls_and_roof(hw=1.5)
        slope_indices = [i for tri in slope_tris for i in tri]
        primitives.append({
            "attributes": {
                "POSITION": b.add_positions(slope_verts),
                "NORMAL": b.add_normals([(1.0, 0.0, 0.0)] * len(slope_verts)),
                "TEXCOORD_0": b.add_uvs([(0.0, 0.0)] * len(slope_verts)),
            },
            "indices": b.add_indices(slope_indices),
            "material": slope_mat,
        })

        mesh = b.add_raw_mesh(primitives)
        b.add_node(mesh_index=mesh, name="GroundPolyMainLayer")
        pave_objs = pave_names

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "ground_poly.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            for name in pave_objs:
                obj = next(p for p in result if name in p.name)
                text = obj.read_text(encoding="utf-8")
                self.assertIn("ATTR_draped", text,
                               f"{name}: an individually flat pavement material must drape even though "
                               f"a sibling material in the same node/file isn't flat")
                self.assertNotIn("TILTED", text.splitlines(),
                                  f"{name}: a draped node must never also carry TILTED")

            slope_obj = next(p for p in result if "Slope" in p.name)
            slope_text = slope_obj.read_text(encoding="utf-8")
            self.assertNotIn("ATTR_draped", slope_text,
                              "the genuinely-3D slope material must stay rigid, not be flattened")

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


if __name__ == "__main__":
    unittest.main()
