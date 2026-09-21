"""
extract_image's DDS passthrough: X-Plane's OBJ8 TEXTURE line supports
.dds natively, so decoding a source DDS to PNG is pure waste when
nothing about the texture needs to change afterward -- confirmed real
gap found comparing this project's output against a different MSFS->
X-Plane converter's: every one of our textures paid a full decode+
re-encode cost even when unmodified (~7x size/VRAM penalty on one real
texture, for zero quality gain), while the other tool passes DDS
through unchanged in the common case.

allow_dds_passthrough is opt-in per call site, not a blanket toggle:
whichever caller knows whether apply_color_factor/apply_alpha_factor/
apply_emissive_factor (all of which edit decoded pixel data) will run
on this specific texture slot afterward must pass False whenever any of
them might.
"""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

import mesh_convert
convert_module = importlib.import_module("mesh_convert.convert")

# A minimal payload that only needs to satisfy extract_image's own
# passthrough check (raw_bytes[:4] == b"DDS ") -- passthrough never
# decodes it, so it doesn't need to be a real, fully-valid DDS.
_FAKE_DDS_BYTES = b"DDS " + b"\x00" * 124 + b"FAKEPIXELDATA"
_FAKE_PNG_BYTES = None  # filled in by setUpModule


def setUpModule():
    global _FAKE_PNG_BYTES
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (2, 2), (200, 120, 60, 255)).save(buf, "PNG")
    _FAKE_PNG_BYTES = buf.getvalue()


def _make_gltf_with_embedded_image(raw_bytes):
    gltf = {
        "images": [{"name": "SomeTexture", "bufferView": 0}],
        "bufferViews": [{"bufferIndex": 0, "byteOffset": 0, "byteLength": len(raw_bytes)}],
    }
    buffers = [raw_bytes]
    return gltf, buffers


class TestDdsPassthrough(unittest.TestCase):
    def test_dds_bytes_pass_through_unmodified_when_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            textures_dir = Path(td)
            gltf, buffers = _make_gltf_with_embedded_image(_FAKE_DDS_BYTES)
            out_name = convert_module.extract_image(
                gltf, buffers, 0, Path(td) / "model.glb", textures_dir, None, {}, (255, 255, 255, 255),
                allow_dds_passthrough=True,
            )
            self.assertTrue(out_name.endswith(".dds"), f"expected a .dds passthrough, got {out_name!r}")
            written = (textures_dir / out_name).read_bytes()
            self.assertEqual(written, _FAKE_DDS_BYTES, "passthrough must write the exact original bytes, no re-encoding")

    def test_dds_bytes_still_decoded_when_passthrough_not_allowed(self):
        """The default (allow_dds_passthrough=False, matching every
        pre-existing call site that never opted in) must keep decoding --
        this is the same-as-before path a caller with a pending color/
        alpha/emissive factor modification relies on."""
        with tempfile.TemporaryDirectory() as td:
            textures_dir = Path(td)
            gltf, buffers = _make_gltf_with_embedded_image(_FAKE_DDS_BYTES)
            out_name = convert_module.extract_image(
                gltf, buffers, 0, Path(td) / "model.glb", textures_dir, None, {}, (255, 255, 255, 255),
            )
            # This fake payload isn't a real decodable DDS, so it falls
            # through to the flat-color-stub fallback -- what matters
            # here is only that it did NOT take the passthrough path.
            self.assertTrue(out_name.endswith(".png"), f"expected a decoded .png (not passthrough), got {out_name!r}")

    def test_non_dds_bytes_are_unaffected_by_passthrough_flag(self):
        """allow_dds_passthrough=True must not change behavior for a
        source that genuinely isn't DDS -- only the b"DDS " magic bytes
        trigger it."""
        with tempfile.TemporaryDirectory() as td:
            textures_dir = Path(td)
            gltf, buffers = _make_gltf_with_embedded_image(_FAKE_PNG_BYTES)
            out_name = convert_module.extract_image(
                gltf, buffers, 0, Path(td) / "model.glb", textures_dir, None, {}, (255, 255, 255, 255),
                allow_dds_passthrough=True,
            )
            self.assertTrue(out_name.endswith(".png"))
            with open(textures_dir / out_name, "rb") as f:
                self.assertTrue(f.read().startswith(b"\x89PNG"), "must still be a real decoded/re-saved PNG")

    def test_tinted_material_disables_passthrough_for_base_color(self):
        """Integration-level check on the real call site in convert():
        a non-white baseColorFactor needs apply_color_factor to bake the
        tint in, which requires a real decoded PNG -- the base-color
        texture slot must NOT pass a DDS through unmodified in that case,
        even though DDS passthrough is supported in general."""
        sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
        from gltf_builder import GltfBuilder  # noqa: E402

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            b = GltfBuilder()
            tex = b.add_image_data_uri((10, 10, 10, 255), name="TintedTex")  # a real, valid embedded image
            texi = b.add_texture(tex)
            mat = b.add_material("TintedMat", base_color_texture_index=texi, base_color_factor=[0.2, 0.4, 0.6, 1.0])
            positions = [(-5, 0, -5), (5, 0, -5), (5, 0, 5), (-5, 0, 5)]
            normals = [(0.0, 1.0, 0.0)] * 4
            uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            indices = [0, 1, 2, 0, 2, 3]
            mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
            b.add_node(mesh_index=mesh, name="TintedQuad")

            glb_path = td / "tinted.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            obj_text = result[0].read_text(encoding="utf-8")
            tex_line = next(line for line in obj_text.splitlines() if line.startswith("TEXTURE "))
            self.assertTrue(tex_line.endswith(".png"),
                             f"a tinted material's base color texture must be a decoded PNG, got: {tex_line!r}")


if __name__ == "__main__":
    unittest.main()
