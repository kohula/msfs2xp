"""
mesh_ir.py: pins the serialization contract independent of conversion
logic (write_obj8(load(saved_meshir)) must reproduce write_obj8(original))
and confirms convert() actually writes a usable sidecar for real converted
geometry.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder, flat_quad  # noqa: E402

import mesh_convert
from mesh_convert import mesh_ir


class TestMeshIrRoundTrip(unittest.TestCase):
    def _sample_ir(self) -> mesh_ir.MeshIR:
        return mesh_ir.MeshIR(
            name="sample",
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            normals=np.array([[0.0, 1.0, 0.0]] * 4),
            uvs=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
            texture="../textures/foo.png",
            draped=True,
            draped_layer_offset=-3,
        )

    def test_save_load_round_trip_preserves_fields(self):
        ir = self._sample_ir()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.meshir.pkl"
            mesh_ir.save(ir, path)
            loaded = mesh_ir.load(path)
        np.testing.assert_array_equal(loaded.positions, ir.positions)
        np.testing.assert_array_equal(loaded.indices, ir.indices)
        self.assertEqual(loaded.texture, ir.texture)
        self.assertEqual(loaded.draped, ir.draped)
        self.assertEqual(loaded.draped_layer_offset, ir.draped_layer_offset)

    def test_write_obj8_is_stable_across_save_load(self):
        ir = self._sample_ir()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            sidecar = td / "sample.meshir.pkl"
            mesh_ir.save(ir, sidecar)
            loaded = mesh_ir.load(sidecar)

            direct_path = td / "direct.obj"
            roundtrip_path = td / "roundtrip.obj"
            mesh_ir.write_obj8(ir, direct_path)
            mesh_ir.write_obj8(loaded, roundtrip_path)

            self.assertEqual(direct_path.read_text(encoding="utf-8"), roundtrip_path.read_text(encoding="utf-8"))

    def test_write_obj8_produces_valid_draped_structure(self):
        ir = self._sample_ir()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.obj"
            mesh_ir.write_obj8(ir, path)
            text = path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("I\n800\nOBJ\n"))
        self.assertIn("TEXTURE ../textures/foo.png", text)
        self.assertEqual(text.count("VT "), 4)
        self.assertEqual(text.count("IDX "), 6)
        self.assertIn("ATTR_draped", text)
        self.assertIn("ATTR_layer_group_draped markings -3", text)
        self.assertIn("TRIS 0 6", text)

    def test_convert_writes_usable_sidecar_for_real_geometry(self):
        b = GltfBuilder()
        tex = b.add_image_data_uri((120, 90, 60, 255), name="TestTex")
        texi = b.add_texture(tex)
        mat = b.add_material("FlatMat", base_color_texture_index=texi)
        pos, norm, uv, idx = flat_quad(0, 10, 0, 5)
        mesh = b.add_mesh(pos, idx, normals=norm, uvs=uv, material_index=mat)
        b.add_node(mesh_index=mesh, name="FlatPlane")
        glb_bytes = b.build()

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "test.glb"
            glb_path.write_bytes(glb_bytes)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            sidecar = mesh_ir.sidecar_path_for(result[0])
            self.assertTrue(sidecar.exists(), "convert() did not write a .meshir.pkl sidecar")

            ir = mesh_ir.load(sidecar)
            self.assertEqual(ir.positions.shape, (4, 3))
            self.assertTrue(ir.draped)
            self.assertEqual(ir.texture, "../textures/testtex.png")

            # Re-serializing the loaded IR must match what convert() itself
            # wrote to the real .obj -- confirms write_obj8 is a faithful
            # re-derivation of convert()'s own draped-object write path,
            # not just an independently-plausible format.
            regenerated_path = td / "regenerated.obj"
            mesh_ir.write_obj8(ir, regenerated_path)
            original_text = result[0].read_text(encoding="utf-8")
            regenerated_text = regenerated_path.read_text(encoding="utf-8")
            self.assertEqual(original_text, regenerated_text)


if __name__ == "__main__":
    unittest.main()
