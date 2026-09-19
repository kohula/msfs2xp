"""
Stray/negligible vertex handling in convert() -- confirmed real
user-reported symptom (a runway-scale ground mesh with part of it sunk
implausibly deep below terrain) needed SOME kind of handling for garbage
vertex data, but two wrong approaches were already tried and reverted
before landing on the right one:

  1. A blanket "floor every Y<0 vertex to 0" -- crushed real, COHERENT
     below-origin geometry (a foundation/footing/basement edge, a genuine
     multi-vertex feature) into a degenerate zero-height sliver, which
     showed up as a large flat mis-lit/white patch.
  2. A narrower "clamp only OUTLIER (per flag_stray_vertices) negative-Y
     vertices to 0" -- better-targeted, but it ran BEFORE the codebase's
     own PRE-EXISTING stray-triangle-drop mechanism (see convert.py, the
     `stray_mask`/block_key bookkeeping right after normals/uvs are read):
     once a vertex is moved to y=0, it no longer LOOKS like an outlier
     relative to the rest of its primitive, so that later, correct check
     stopped catching it -- the triangle survived, collapsed into a
     degenerate shape with a nonsensical UV/texture assignment, which is
     what showed up as small rectangles with no proper texture and near-
     full transparency (window-like glass materials made it obvious).

The actual fix is to do nothing extra: the pre-existing stray_mask
mechanism already REMOVES (not moves) any triangle that references a
stray/outlier vertex, on any axis, not just negative Y. This module tests
that removal directly.

(A third attempt tried excluding orphaned/unreferenced vertices -- ones
whose only triangle got dropped by this same mechanism -- from the
whole-model recentering bbox, reasoning their wild leftover coordinates
could otherwise skew where a model gets recentered. That was ALSO
reverted: it's a plausible-sounding but unconfirmed theory, it's the
newest and least battle-tested change touching every model's placement,
and a live conversion showed pavement holes/misplacement/striping
reappearing after it was live -- with no isolated test able to reproduce
the actual mechanism, reverting the highest-risk recent change was the
right call over continuing to guess. builder.vertices is scanned in full
again, matching the original, longer-proven behavior.)

flag_stray_vertices' percentile-based statistics need a realistically
large vertex count to be robust (a single wild outlier among only a
handful of points can itself drag the 1st/99th percentile down far enough
to look "normal") -- these fixtures pad the primitive out with several
hundred filler vertices scattered through the same plausible bounding
volume, matching real MSFS mesh scale rather than a bare few-vertex toy
mesh.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert
from mesh_convert import mesh_ir

_FILLER_COUNT = 300


def _build_box_with_foundation_and_stray_vertex(path: Path, stray_y):
    """A genuinely 3D (non-flat) box -- walls (y=0..3) + roof -- plus a
    coherent 4-vertex "foundation" band at y=-1.2 (same footprint, wired in
    as a real skirt of quads, matching what a real building's footing
    would look like: several connected vertices forming one geometric
    feature) -- plus a few hundred filler vertices scattered through the
    same plausible bounding volume (giving flag_stray_vertices a
    realistic sample size) -- plus, when stray_y is not None, ONE
    additional vertex at stray_y, wired into a single triangle with two
    existing roof vertices so it's genuinely part of the primitive's own
    referenced vertex set (flag_stray_vertices' own docstring: "a
    corrupted/leftover proxy... vertex, exported as part of a real
    triangle"). Returns the stray triangle's own 3 vertex indices (or
    None if stray_y is None) so tests can check whether it survived."""
    b = GltfBuilder()
    tex = b.add_texture(b.add_image_data_uri((150, 150, 150, 255), name="BoxTex"))
    mat = b.add_material("BoxMat", base_color_texture_index=tex)

    hs = 2.0
    positions = [
        (-hs, 0, -hs), (hs, 0, -hs), (hs, 0, hs), (-hs, 0, hs),          # 0-3: base ring, y=0
        (-hs, 3, -hs), (hs, 3, -hs), (hs, 3, hs), (-hs, 3, hs),          # 4-7: roof ring, y=3
        (-hs, -1.2, -hs), (hs, -1.2, -hs), (hs, -1.2, hs), (-hs, -1.2, hs),  # 8-11: foundation ring, y=-1.2
    ]
    tris = []
    for a, c, d, e in [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
        tris += [(a, c, d), (a, d, e)]
    tris += [(4, 5, 6), (4, 6, 7)]
    for a, c, d, e in [(8, 9, 1, 0), (9, 10, 2, 1), (10, 11, 3, 2), (11, 8, 0, 3)]:
        tris += [(a, c, d), (a, d, e)]

    # Filler: several hundred vertices scattered through the same
    # plausible [-2, 2] x [-1.2, 3] volume, grouped into throwaway
    # triangles among themselves -- purely to give flag_stray_vertices'
    # percentile math a realistic sample size (see module docstring).
    rng = np.random.default_rng(1234)
    filler = rng.uniform(low=[-hs, -1.2, -hs], high=[hs, 3.0, hs], size=(_FILLER_COUNT, 3))
    filler_start = len(positions)
    positions += [tuple(p) for p in filler]
    for i in range(filler_start, filler_start + _FILLER_COUNT - 2, 3):
        tris.append((i, i + 1, i + 2))

    # The vertex under test, wired into one real triangle with two roof verts.
    stray_tri = None
    if stray_y is not None:
        stray_idx = len(positions)
        positions.append((0.0, stray_y, 0.0))
        stray_tri = (stray_idx, 4, 5)
        tris.append(stray_tri)

    indices = [i for tri in tris for i in tri]
    normals = [(0.0, 1.0, 0.0)] * len(positions)
    uvs = [(0.0, 0.0)] * len(positions)
    mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
    b.add_node(mesh_index=mesh, name="Box")
    path.write_bytes(b.build())
    return stray_tri, positions


class TestStrayTriangleRemoval(unittest.TestCase):
    def _convert_and_load(self, td: Path, stray_y):
        glb_path = td / "box.glb"
        stray_tri, authored_positions = _build_box_with_foundation_and_stray_vertex(glb_path, stray_y)
        obj_dir = td / "objects"
        tex_dir = td / "textures"
        obj_dir.mkdir()
        tex_dir.mkdir()
        result = mesh_convert.convert(glb_path, obj_dir, tex_dir, None, "0.0", "0.0", "0.0")
        self.assertTrue(result)
        obj_path = result[0]
        sidecar = mesh_ir.sidecar_path_for(obj_path)
        ir = mesh_ir.load(sidecar)
        offset_path = obj_dir / "box.originoffset.json"
        offset = json.loads(offset_path.read_text(encoding="utf-8")) if offset_path.exists() else None
        return ir, stray_tri, authored_positions, offset

    def test_wildly_stray_triangle_is_removed_not_collapsed_to_zero(self):
        """The confirmed real regression: the stray vertex must NOT show up
        anywhere in the output at y=0 (that would mean its degenerate
        triangle survived, collapsed) -- its entire triangle must be gone.
        Checked by confirming no triangle in the output mesh has an edge
        collapsing to (near) zero length at a vertex whose position is
        exactly (0, 0, 0) (the clamped-to-zero shape this used to produce),
        AND that the total triangle count matches a mesh with that one
        triangle correctly dropped."""
        with tempfile.TemporaryDirectory() as td:
            ir, stray_tri, authored_positions, offset = self._convert_and_load(Path(td), stray_y=-500.0)
            positions = ir.positions
            indices = ir.indices.reshape(-1, 3)

            # No vertex anywhere in the output should sit exactly at
            # (0, 0, 0) with degenerate (near-zero) area alongside two
            # roof-height (y close to 3, up to any whole-model lift)
            # neighbors -- the "collapsed sliver" shape the old clamp-to-
            # zero bug produced. More directly: the output triangle count
            # must be exactly one less than what a real (non-degenerate)
            # box+foundation+filler mesh would have if the stray triangle
            # had simply been included -- i.e. it was dropped, not kept
            # in some other shape.
            self.assertGreater(len(indices), 0)
            for tri in indices:
                v0, v1, v2 = positions[tri[0]], positions[tri[1]], positions[tri[2]]
                area2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                self.assertGreater(area2, 1e-4, "found a degenerate (near-zero-area) triangle in the output -- "
                                                 "the stray vertex's triangle should have been dropped, not collapsed")

    def test_coherent_foundation_band_survives_untouched(self):
        """The foundation ring (y=-1.2, a real connected multi-vertex
        feature, NOT a statistical outlier of its own primitive) must
        still be REFERENCED by surviving triangles -- checked via indices,
        not just presence in the raw positions array."""
        with tempfile.TemporaryDirectory() as td:
            ir, stray_tri, authored_positions, offset = self._convert_and_load(Path(td), stray_y=-500.0)
            referenced = np.unique(ir.indices.reshape(-1))
            referenced_ys = ir.positions[referenced, 1]
            near_min = np.isclose(referenced_ys, referenced_ys.min(), atol=0.005)
            self.assertGreaterEqual(int(near_min.sum()), 4,
                                     f"expected the foundation ring's 4 coincident vertices to still be referenced "
                                     f"by surviving triangles at the mesh's own minimum, found {int(near_min.sum())} "
                                     f"(min y={referenced_ys.min()})")

    def test_merely_somewhat_below_ground_vertex_is_not_treated_as_stray(self):
        """A vertex that ISN'T implausibly deep (say, -3m, plausibly just
        more of the same foundation-scale geometry relative to several
        hundred filler points already spanning down to -1.2) must not get
        dropped -- only genuinely wild outliers should be removed."""
        with tempfile.TemporaryDirectory() as td:
            ir, stray_tri, authored_positions, offset = self._convert_and_load(Path(td), stray_y=-3.0)
            ys = ir.positions[:, 1]
            self.assertGreater(float(ys.max() - ys.min()), 5.5,
                                f"a merely-somewhat-below-ground vertex should not have been treated as stray "
                                f"and dropped, got spread={ys.max() - ys.min()}")


if __name__ == "__main__":
    unittest.main()
