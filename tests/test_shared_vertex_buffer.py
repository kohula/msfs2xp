"""
Real MSFS/ASOBO glTF exports routinely pack MANY primitives' vertex data
into ONE big shared POSITION/NORMAL/TEXCOORD accessor, with each primitive
selecting only its own small window via the ASOBO_primitive extension's
StartIndex/BaseVertexIndex/PrimitiveCount extras (see convert.py's own
comment at the local_indices computation). Confirmed against a real,
detailed payware airport (LHBP): a single "parking lines" ground-marking
node split by material into 15 separate output .obj files, EVERY one of
which carried ~389,000-395,000 VT lines despite only using a few hundred
to a few thousand of them (one had 6 indices total against 389,216
vertices) -- because convert() read_accessor()'d the FULL shared accessor
for every single material-split builder that touched it, not just the
slice that builder's own primitive actually referenced.

This reproduces that exact shape at a tiny scale: one 1000-vertex shared
accessor, two primitives (different materials, so two separate output
.obj files) each using only a disjoint 4-vertex window of it via
BaseVertexIndex. Each output file must carry only ITS OWN window's worth
of vertices, not the full shared 1000.
"""
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert

_POINT_COUNTS_RE = re.compile(r"POINT_COUNTS (\d+) (\d+) (\d+) (\d+)")


def _point_counts(obj_path: Path):
    text = obj_path.read_text(encoding="utf-8")
    m = _POINT_COUNTS_RE.search(text)
    return tuple(int(x) for x in m.groups())


class TestSharedVertexBuffer(unittest.TestCase):
    def test_material_split_primitives_dont_each_carry_the_full_shared_buffer(self):
        b = GltfBuilder()
        tex_uri = GltfBuilder.make_data_uri((180, 180, 180, 255))
        imgA = b.add_image_uri(tex_uri, name="TexA")
        imgB = b.add_image_uri(tex_uri, name="TexB")
        texA = b.add_texture(imgA)
        texB = b.add_texture(imgB)
        matA = b.add_material("LinesMatA", base_color_texture_index=texA)
        matB = b.add_material("LinesMatB", base_color_texture_index=texB)

        n = 1000
        positions = [(0.0, 0.0, 0.0)] * n
        normals = [(0.0, 1.0, 0.0)] * n
        uvs = [(0.0, 0.0)] * n
        # Quad A lives at buffer indices 0-3.
        positions[0], positions[1], positions[2], positions[3] = (
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0))
        uvs[0], uvs[1], uvs[2], uvs[3] = (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
        # Quad B lives at buffer indices 500-503, far away in the SAME shared buffer.
        positions[500], positions[501], positions[502], positions[503] = (
            (50.0, 0.0, 50.0), (51.0, 0.0, 50.0), (51.0, 0.0, 51.0), (50.0, 0.0, 51.0))
        uvs[500], uvs[501], uvs[502], uvs[503] = (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)

        pos_acc = b.add_positions(positions)
        norm_acc = b.add_normals(normals)
        uv_acc = b.add_uvs(uvs)

        idx_local = [0, 1, 2, 0, 2, 3]
        idxA = b.add_indices(idx_local)
        idxB = b.add_indices(idx_local)

        prims = [
            {
                "attributes": {"POSITION": pos_acc, "NORMAL": norm_acc, "TEXCOORD_0": uv_acc},
                "indices": idxA,
                "material": matA,
                "extras": {"ASOBO_primitive": {"StartIndex": 0, "PrimitiveCount": 2, "BaseVertexIndex": 0}},
            },
            {
                "attributes": {"POSITION": pos_acc, "NORMAL": norm_acc, "TEXCOORD_0": uv_acc},
                "indices": idxB,
                "material": matB,
                "extras": {"ASOBO_primitive": {"StartIndex": 0, "PrimitiveCount": 2, "BaseVertexIndex": 500}},
            },
        ]
        mesh_idx = b.add_raw_mesh(prims)
        b.add_node(mesh_index=mesh_idx, name="SharedBufferNode")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "shared_buffer.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            results = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertEqual(len(results), 2, "one output object per material")

            for obj_path in results:
                vt_count, _, _, idx_count = _point_counts(obj_path)
                self.assertLessEqual(
                    vt_count, 20,
                    f"{obj_path.name}: declared {vt_count} vertices for a single 4-vertex quad -- "
                    f"the FULL {n}-vertex shared accessor is being appended instead of just this "
                    f"primitive's own referenced window (idx_count={idx_count})",
                )


if __name__ == "__main__":
    unittest.main()
