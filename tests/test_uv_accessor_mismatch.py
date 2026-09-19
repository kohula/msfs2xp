"""
Confirmed real risk: positions/normals/UVs for one primitive are read from
THREE INDEPENDENT accessors, each sliced by the same [win_lo:win_hi+1]
window (see mesh_convert/convert.py's own ASOBO_primitive windowing
comment). glTF's spec requires every attribute on one primitive to share
vertex count, but this project's own comments already document real ASOBO
exports bending spec conventions elsewhere -- if a TEXCOORD (or NORMAL)
accessor were ever genuinely shorter than what a primitive's window needs,
a plain numpy slice truncates silently instead of raising, appending FEWER
UV/normal rows than vertex rows and silently shifting the UV/normal index
for every vertex appended afterward in the same builder. This is the
leading hypothesis for a real reported bug where one ground marking
rendered showing what looked like its entire shared texture atlas at once
(a garbled, index-shifted UV read) instead of its own small cropped decal.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert


class TestUvAccessorMismatch(unittest.TestCase):
    def test_short_texcoord_accessor_falls_back_instead_of_shifting_index(self):
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((180, 180, 180, 255)))
        mat = b.add_material("Mat", base_color_texture_index=tex)

        positions = np.array([(0, 0, 0), (10, 0, 0), (10, 0, 10), (0, 0, 10)], dtype=np.float32)
        normals = np.array([(0.0, 1.0, 0.0)] * 4, dtype=np.float32)
        # Deliberately SHORT: only 2 rows for 4 vertices -- a genuinely
        # malformed/spec-violating export, which is exactly the case this
        # guard exists for.
        short_uvs = np.array([(0.0, 0.0), (1.0, 0.0)], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        pos_acc = b.add_accessor(positions, 5126, "VEC3")
        norm_acc = b.add_accessor(normals, 5126, "VEC3")
        uv_acc = b.add_accessor(short_uvs, 5126, "VEC2")
        idx_acc = b.add_indices(indices)

        prim = {
            "attributes": {"POSITION": pos_acc, "NORMAL": norm_acc, "TEXCOORD_0": uv_acc},
            "indices": idx_acc, "material": mat,
        }
        mesh = b.add_raw_mesh([prim])
        b.add_node(mesh_index=mesh, name="MismatchedQuad")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "mismatched.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            # Must not raise/crash the whole model's conversion.
            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result, "a short TEXCOORD accessor must not abort the whole model's conversion")

            text = result[0].read_text(encoding="utf-8")
            vt_lines = [l for l in text.splitlines() if l.startswith("VT ")]
            self.assertEqual(len(vt_lines), 4, "all 4 vertices must still be written, none dropped by the mismatch")

            # Fallback UV is a uniform (0, 0) for every vertex in this
            # block, not a garbled/index-shifted read from the short array.
            uvs_written = [(float(l.split()[7]), float(l.split()[8])) for l in vt_lines]
            self.assertTrue(all(uv == (0.0, 0.0) for uv in uvs_written),
                             f"expected uniform (0,0) UV fallback for the whole mismatched block, got {uvs_written}")


if __name__ == "__main__":
    unittest.main()
