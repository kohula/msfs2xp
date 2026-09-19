"""
A skinned mesh node (JOINTS_0/WEIGHTS_0-style rigging -- common for a
large, complex door/barrier asset) is frequently a scene-graph SIBLING of
the joint that actually drives it, not a node.children descendant of it.
find_animated_ancestor's skin fallback (see convert.py) correctly finds
that driving joint for the ANIM_ block, but a real, confirmed, previously
undocumented-as-fixed gap remained: the mesh's own STATIC world transform
was still built purely from its own (unrelated) node.children parent
chain, with no knowledge of the joint's rest pose at all -- so the mesh's
baked geometry rendered at whatever pose that unrelated parent chain
implied (commonly the origin / bind pose), not the joint's real rest
position. This is the confirmed real-world symptom: a door that renders
permanently open regardless of a correctly-computed ANIM_ close delta,
because the STATIC geometry itself was never moved to the closed
position in the first place.

A second, separate rest-pose bug pinned here: node_local_matrix_at_rest
used to always trust the animation's FIRST keyframe (values[0]) as rest.
Confirmed against a real airport package (MSFS2XP) that this doesn't hold
across every asset -- 28 of 30 real door/barrier hinges in one package
are authored closed-at-values[0], but 2 (both leaves of one double door)
are authored the opposite way, closed-at-values[-1]. Baking rest from a
fixed array position renders that door's static geometry (AND its ANIM_
open direction) backwards: frozen open at rest, and swinging further
open -- not closed -- as the trigger condition is met.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert


class TestSkinnedAnimationRestPose(unittest.TestCase):
    def test_skin_fallback_mesh_uses_joints_rest_world_not_its_own_parent_chain(self):
        b = GltfBuilder()
        tex = b.add_image_data_uri((180, 180, 180, 255))
        texi = b.add_texture(tex)
        mat = b.add_material("DoorMat", base_color_texture_index=texi)

        # Joint node: NO static translation of its own -- its rest pose
        # comes entirely from the animation sampler's first keyframe
        # (node_local_matrix_at_rest), landing at local (10, 0, 0).
        joint_idx = b.add_node(name="DoorHinge")
        b.add_animation(
            target_node=joint_idx, path="translation",
            times=[0.0, 2.0], values=[(10.0, 0.0, 0.0), (10.0, 0.0, -2.0)])

        # Mesh node: a scene-graph ROOT, NOT a child of the joint -- its
        # own naive parent chain resolves to identity (origin). A flat
        # quad centered on its own LOCAL origin, so the final WORLD X
        # coordinate directly reveals which transform was actually used:
        # ~0 means the (buggy) naive parent-chain transform was used,
        # ~10 means the joint's rest transform was correctly applied.
        positions = [(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5)]
        normals = [(0.0, 1.0, 0.0)] * 4
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        indices = [0, 1, 2, 0, 2, 3]
        mesh_idx = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)

        # Identity bind pose: mesh-local space maps directly onto the
        # joint's own space with no additional offset, so the expected
        # corrected world transform is exactly the joint's rest transform.
        identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        skin_idx = b.add_skin(joints=[joint_idx], inverse_bind_matrices=[identity])
        b.add_node(mesh_index=mesh_idx, name="DoorPanel", skin_index=skin_idx)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "skinned_door.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            text = result[0].read_text(encoding="utf-8")
            vt_lines = [l for l in text.splitlines() if l.startswith("VT ")]
            self.assertTrue(vt_lines)
            xs = [float(l.split()[1]) for l in vt_lines]
            mean_x = sum(xs) / len(xs)

            # convert() now re-centers every model around its own XZ
            # footprint before writing VT lines, moving the removed offset
            # into a "<model_stem>.originoffset.json" sidecar instead (see
            # convert.py's own re-centering comments) -- so the world x=10
            # this test pins is now split between the written-out local
            # geometry (re-centered to ~0) and that sidecar's own "x"
            # value, rather than living directly in the VT lines. Summing
            # them back together re-derives the same world position the
            # original (pre-recentering) assertion checked directly.
            sidecar = obj_dir / "skinned_door.originoffset.json"
            self.assertTrue(sidecar.exists())
            offset = json.loads(sidecar.read_text(encoding="utf-8"))

            self.assertAlmostEqual(
                mean_x + offset["x"], 10.0, places=3,
                msg=f"skin-fallback mesh should be positioned at the joint's rest world (x=10), "
                    f"got mean x={mean_x} + origin-offset x={offset['x']} -- looks like the mesh's "
                    f"own (unrelated) parent-chain transform was used instead"
            )


    def test_reversed_keyframe_order_still_bakes_the_identity_pose_as_rest(self):
        """A hinge authored with its OPEN pose at values[0] and its CLOSED
        (identity-rotation) pose at values[-1] -- the reverse of the usual
        authoring order, and exactly the real shape found on one of two
        confirmed-broken hinges in a real airport package's own double
        door. node_local_matrix_at_rest must still bake the identity
        keyframe as rest regardless of which array position it's authored
        at (see _rest_open_rotation_values), not just always trust
        values[0]."""
        b = GltfBuilder()
        tex = b.add_image_data_uri((180, 180, 180, 255))
        texi = b.add_texture(tex)
        mat = b.add_material("DoorMat", base_color_texture_index=texi)

        # Flat quad centered at local (5, 0, 0) relative to the hinge, so
        # its world X directly reveals which keyframe got baked as rest:
        # ~5 means identity (closed, correct); ~0 (rotated onto Z instead)
        # means the OPEN keyframe was wrongly baked as rest.
        positions = [(4.5, 0, -0.5), (5.5, 0, -0.5), (5.5, 0, 0.5), (4.5, 0, 0.5)]
        normals = [(0.0, 1.0, 0.0)] * 4
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        indices = [0, 1, 2, 0, 2, 3]
        mesh_idx = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        mesh_node = b.add_node(mesh_index=mesh_idx, name="DoorPanel", top_level=False)

        hinge_idx = b.add_node(name="DoorHinge", children=[mesh_node])
        open_quat = (0.0, 0.70710678, 0.0, 0.70710678)  # 90 degrees about Y
        identity_quat = (0.0, 0.0, 0.0, 1.0)
        b.add_animation(
            target_node=hinge_idx, path="rotation", times=[0.0, 2.0],
            values=[open_quat, identity_quat])  # reversed: open first, closed last

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "reversed_hinge_door.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)

            text = result[0].read_text(encoding="utf-8")
            vt_lines = [l for l in text.splitlines() if l.startswith("VT ")]
            self.assertTrue(vt_lines)
            xs = [float(l.split()[1]) for l in vt_lines]
            mean_x = sum(xs) / len(xs)

            sidecar = obj_dir / "reversed_hinge_door.originoffset.json"
            self.assertTrue(sidecar.exists())
            offset = json.loads(sidecar.read_text(encoding="utf-8"))

            self.assertAlmostEqual(
                mean_x + offset["x"], 5.0, places=3,
                msg=f"reversed-keyframe hinge should still bake its identity (closed) keyframe as "
                    f"rest regardless of array position, got mean x={mean_x} + origin-offset "
                    f"x={offset['x']} -- looks like values[0] (the open pose here) was baked instead"
            )


if __name__ == "__main__":
    unittest.main()
