"""
Regression test for the confirmed winding/normal instancing bug: previously,
reverse_winding was computed once per NODE from base_world (before the
per-instance loop), while normal_mat was correctly computed per INSTANCE
from base_world @ instance_matrix. A mirrored EXT_mesh_gpu_instancing
instance (a real technique for e.g. "left-facing vs right-facing sign"
variants of one base asset) got correctly-flipped normals but stale,
un-flipped winding -- the object rendered inside-out relative to its own
mirrored position.

This builds a synthetic glTF with one instanced mesh, two instances --
identity and a negative-X-scale mirror -- and confirms both the fixed
behavior AND, by running the identical fixture through the OLD (GPU/)
converter, that the bug this guards against was real and is now closed.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

_GPU_DIR = Path(__file__).resolve().parent.parent.parent / "msfs2xp_v0813_2" / "GPU"


def _signed_area_xz(p0, p1, p2):
    """2D signed area of a triangle projected onto the XZ plane, using the
    SAME (a, b, c) vertex order the triangle's own IDX lines list them in
    -- positive vs. negative sign directly encodes winding direction."""
    x0, _, z0 = p0
    x1, _, z1 = p1
    x2, _, z2 = p2
    return 0.5 * ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0))


def _build_mirrored_instance_glb() -> bytes:
    b = GltfBuilder()
    tex = b.add_image_data_uri((100, 100, 100, 255))
    texi = b.add_texture(tex)
    mat = b.add_material("SignMat", base_color_texture_index=texi)

    # Asymmetric triangle offset from the origin (so mirroring across x=0
    # produces a position range distinguishable from the original,
    # letting the test identify which output triangle came from which
    # instance).
    positions = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.0, 1.0)]
    normals = [(0.0, 1.0, 0.0)] * 3
    uvs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    indices = [0, 1, 2]
    mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)

    b.add_instanced_node(
        mesh, "MirroredSignInstances",
        instance_translations=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        instance_scales=[(1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)],  # instance 0 = identity, instance 1 = mirrored
    )
    return b.build()


def _convert_and_get_signed_areas(convert_fn, glb_bytes: bytes):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        glb_path = td / "mirrored.glb"
        glb_path.write_bytes(glb_bytes)
        obj_dir = td / "objects"
        tex_dir = td / "textures"
        obj_dir.mkdir()
        tex_dir.mkdir()

        result = convert_fn(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
        if not result:
            return None
        text = result[0].read_text(encoding="utf-8")

        vt_positions = [
            tuple(float(x) for x in l.split()[1:4])
            for l in text.splitlines() if l.startswith("VT ")
        ]
        idx_values = [int(l.split()[1]) for l in text.splitlines() if l.startswith("IDX ")]
        if len(idx_values) != 6:
            return None

        tri_a = [vt_positions[i] for i in idx_values[0:3]]
        tri_b = [vt_positions[i] for i in idx_values[3:6]]
        original_tri = tri_a if min(p[0] for p in tri_a) > 0 else tri_b
        mirrored_tri = tri_b if original_tri is tri_a else tri_a
        return _signed_area_xz(*original_tri), _signed_area_xz(*mirrored_tri)


class TestWindingInstancing(unittest.TestCase):
    def test_mirrored_instance_matches_original_winding_sign(self):
        import mesh_convert

        areas = _convert_and_get_signed_areas(mesh_convert.convert, _build_mirrored_instance_glb())
        self.assertIsNotNone(areas, "conversion did not produce the expected 2-triangle output")
        area_original, area_mirrored = areas

        self.assertNotAlmostEqual(area_original, 0.0, msg="degenerate original triangle -- test setup issue")
        self.assertNotAlmostEqual(area_mirrored, 0.0, msg="degenerate mirrored triangle -- test setup issue")

        # A geometric mirror alone flips a triangle's signed-area sign
        # (reflection reverses orientation). The code compensates by ALSO
        # reversing the winding order for a mirrored instance (swaps two
        # vertices when appending indices) -- the whole point of
        # "reverse_winding", so the object doesn't render inside-out.
        # Position-mirror-sign-flip + winding-swap-sign-flip cancel out,
        # so a CORRECTLY fixed mirrored instance ends up with the SAME
        # signed-area sign as the original (both +0.5 for this fixture,
        # hand-verified). See test_old_gpu_version_exhibits_the_bug below
        # for direct confirmation this actually discriminates fixed from
        # buggy behavior, not just an assertion that happens to pass.
        self.assertGreater(
            area_original * area_mirrored, 0.0,
            f"WINDING BUG: original (signed area={area_original}) and mirrored "
            f"(signed area={area_mirrored}) instances must have the SAME sign once "
            f"winding correctly compensates for the per-instance mirror; opposite "
            f"signs mean the mirrored instance kept the node-level (un-mirrored) winding."
        )

    def test_old_gpu_version_exhibits_the_bug(self):
        """Runs the IDENTICAL fixture through the OLD (GPU/) converter --
        confirms the bug this test guards against was real (opposite
        signs there), not a hypothetical. If this test ever fails, the old
        tree changed underneath it or no longer matches what was fixed;
        it is not asserting anything about the new code."""
        if not _GPU_DIR.is_dir():
            self.skipTest(f"old GPU/ tree not found at {_GPU_DIR} -- skipping bug-existed confirmation")
        sys.path.insert(0, str(_GPU_DIR))
        import glb2obj as old_convert_module  # noqa: E402

        areas = _convert_and_get_signed_areas(old_convert_module.convert, _build_mirrored_instance_glb())
        self.assertIsNotNone(areas, "old converter did not produce the expected 2-triangle output")
        area_original, area_mirrored = areas
        self.assertLess(
            area_original * area_mirrored, 0.0,
            "expected the OLD (pre-fix) converter to exhibit the winding bug (opposite signs) "
            "on this exact fixture -- if it no longer does, the bug-existed premise this "
            "regression test is built on needs re-checking"
        )


if __name__ == "__main__":
    unittest.main()
