"""
MSFS translucent (BLEND) materials are very often authored single-sided,
and a common source-side workaround is a SECOND, manually-duplicated,
reverse-wound copy of the same (or a subset of the same) panels so the
surface is visible from the inside too -- confirmed real case: LHBP's ATC
tower cab, "Glass" (covering the cab AND the lower facade) + "Glass-
Obratno" (Russian/Slavic for "reverse", covering just the cab, every one
of its vertices found within Glass's own). mesh_convert.convert() already
forces ANY BLEND builder double-sided (see test_double_sided_glass.py),
which makes the ORIGINAL single mesh visible from both sides on its own --
so the artist's own duplicate becomes pure dead weight, coincident with
part of the original, and two coincident BLEND surfaces cause an
intermittent "sometimes see-through, sometimes solid" render flicker
(X-Plane's blend draw order between them isn't guaranteed stable frame to
frame). This tests the fix: the redundant duplicate is dropped, detected
purely geometrically (a full positional-vertex-subset match), not by name.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder, flat_quad  # noqa: E402

import mesh_convert
from mesh_convert.convert import _vertex_positions_are_subset


class TestVertexPositionsAreSubset(unittest.TestCase):
    def test_exact_subset_within_tolerance_matches(self):
        big = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 2.0), (0.0, 0.0, 2.0),
               (4.0, 0.0, 0.0), (4.0, 0.0, 2.0)]
        small = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 2.0)]
        self.assertTrue(_vertex_positions_are_subset(small, big))

    def test_tiny_jitter_within_tolerance_still_matches(self):
        big = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 2.0), (0.0, 0.0, 2.0)]
        small = [(0.001, 0.0, 0.0), (1.999, 0.0, 0.0)]
        self.assertTrue(_vertex_positions_are_subset(small, big))

    def test_a_vertex_with_no_close_match_fails(self):
        big = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 2.0), (0.0, 0.0, 2.0)]
        small = [(0.0, 0.0, 0.0), (50.0, 0.0, 50.0)]  # second point far outside big
        self.assertFalse(_vertex_positions_are_subset(small, big))

    def test_empty_inputs_are_not_a_subset(self):
        self.assertFalse(_vertex_positions_are_subset([], [(0.0, 0.0, 0.0)]))
        self.assertFalse(_vertex_positions_are_subset([(0.0, 0.0, 0.0)], []))


class TestGlassDuplicateDedup(unittest.TestCase):
    def _convert_two_blend_panes(self, span_a=(0, 4), span_b=None, alpha_mode_b="BLEND"):
        """Builds one glTF with TWO separate BLEND-material meshes/nodes:
        "Glass" spanning span_a (as two adjacent flat_quads, so it has
        more/different vertices than a single quad would), and
        "Glass-Obratno" spanning span_b (reverse-wound). When span_b is
        None, it defaults to exactly the FIRST quad of span_a (a genuine
        positional subset, the real-world LHBP case)."""
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((150, 150, 220, 180)))

        mat_a = b.add_material("Glass", base_color_texture_index=tex, alpha_mode="BLEND")
        x0, x1 = span_a
        xm = (x0 + x1) / 2.0
        pos_a1, norm_a1, uv_a1, idx_a1 = flat_quad(x0, xm, 0, 2)
        pos_a2, norm_a2, uv_a2, idx_a2 = flat_quad(xm, x1, 0, 2)
        positions_a = pos_a1 + pos_a2
        normals_a = norm_a1 + norm_a2
        uvs_a = uv_a1 + uv_a2
        indices_a = idx_a1 + [i + 4 for i in idx_a2]
        mesh_a = b.add_mesh(positions_a, indices_a, normals=normals_a, uvs=uvs_a, material_index=mat_a)
        b.add_node(mesh_index=mesh_a, name="Glass")

        mat_b = b.add_material("Glass-Obratno", base_color_texture_index=tex, alpha_mode=alpha_mode_b)
        if span_b is None:
            positions_b, normals_b, uvs_b, indices_b = pos_a1, norm_a1, uv_a1, idx_a1
        else:
            positions_b, normals_b, uvs_b, indices_b = flat_quad(span_b[0], span_b[1], 0, 2)
        # Reverse winding, matching a real "-Obratno" reverse-wound copy.
        indices_b = [indices_b[i] for i in range(len(indices_b) - 1, -1, -1)]
        mesh_b = b.add_mesh(positions_b, indices_b, normals=normals_b, uvs=uvs_b, material_index=mat_b)
        b.add_node(mesh_index=mesh_b, name="Glass-Obratno")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "tower.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()
            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            return {p.stem: p.read_text(encoding="utf-8") for p in result}

    def test_redundant_reverse_wound_subset_is_dropped(self):
        """The real LHBP case: Glass-Obratno's vertices are a strict
        subset of Glass's own -- it must be dropped entirely, leaving
        Glass alone (now double-sided) to cover the same visual area."""
        objs = self._convert_two_blend_panes()
        names = list(objs.keys())
        glass_names = [n for n in names if "obratno" not in n.lower()]
        obratno_names = [n for n in names if "obratno" in n.lower()]
        self.assertEqual(len(obratno_names), 0, f"Glass-Obratno should have been dropped, got: {names}")
        self.assertEqual(len(glass_names), 1)
        self.assertIn("ATTR_no_cull", objs[glass_names[0]])
        self.assertIn("ATTR_blend", objs[glass_names[0]])

    def test_non_overlapping_blend_panes_are_both_kept(self):
        """Two GENUINELY DIFFERENT glass panels (no shared geometry) must
        both survive -- this is not a blanket "merge any two BLEND
        builders" rule, only an exact positional-subset match qualifies."""
        objs = self._convert_two_blend_panes(span_a=(0, 4), span_b=(100, 104))
        self.assertEqual(len(objs), 2, f"expected both panes kept, got: {list(objs.keys())}")
        for text in objs.values():
            self.assertIn("ATTR_no_cull", text)  # still independently double-sided

    def test_partially_overlapping_pane_is_not_treated_as_a_duplicate(self):
        """A pane sharing only SOME vertices with another (e.g. an
        adjacent tile in a window grid sharing one edge) must NOT be
        dropped -- only a FULL positional-subset match qualifies."""
        objs = self._convert_two_blend_panes(span_a=(0, 4), span_b=(3, 7))
        self.assertEqual(len(objs), 2, f"expected both panes kept, got: {list(objs.keys())}")

    def test_opaque_duplicate_is_not_dropped(self):
        """This is scoped to BLEND materials only -- an OPAQUE duplicate
        (a different real-world situation, e.g. two coplanar opaque
        decals) is left alone; dropping it would delete real content
        this fix was never meant to touch."""
        objs = self._convert_two_blend_panes(alpha_mode_b="OPAQUE")
        self.assertEqual(len(objs), 2, f"expected both kept (one isn't BLEND), got: {list(objs.keys())}")


if __name__ == "__main__":
    unittest.main()
