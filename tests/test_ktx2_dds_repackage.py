"""
main.repackage_ktx2_to_dds / decode_or_repackage_ktx2 -- the real fix for
the confirmed texture-size gap (one real texture: 699KB as a passed-
through .dds vs 5.1MB fully decoded to .png). The FIRST attempt at this
(mesh_convert.convert's DDS-passthrough) turned out to never fire on real
data: real MSFS packages reference textures externally as .ktx2, which
Step 2's own bulk pass (this module) already fully decodes to PNG before
any model conversion starts -- that's the actual dominant path (~90%+ of
real textures), not the rare embedded-glTF-DDS case. This repackages a
KTX2's own GPU-native block-compressed payload straight into a DDS
container instead of decoding to raw pixels and re-encoding, for
everything except normal maps (which still need decode_ktx2_to_png's real
Z-channel reconstruction -- BC5 normal maps only carry X/Y).
"""
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import main


def _build_fake_ktx2(vk_format, width, height, compressed_bytes, supercompression=0):
    """A minimal, structurally-valid KTX2 file wrapping compressed_bytes
    as its single mip level -- enough for main.parse_ktx2_header (which
    only reads by field index, not full spec compliance) to parse
    correctly. Real block CONTENT doesn't matter for these tests; only
    that repackaging preserves it bit-for-bit and that both decode paths
    agree on what it means."""
    magic = b"\xabKTX 20\xbb\r\n\x1a\n"
    header = struct.pack("<17I", vk_format, 1, width, height, 0, 1, 1, 1, supercompression,
                          0, 0, 0, 0, 0, 0, 0, 0)
    offset = len(magic) + len(header) + 24  # + one 24-byte level-index entry
    level_index = struct.pack("<3Q", offset, len(compressed_bytes), len(compressed_bytes))
    return magic + header + level_index + compressed_bytes


def _fake_block_bytes(n, seed=0):
    import random
    rnd = random.Random(seed)
    return bytes(rnd.randrange(256) for _ in range(n))


class TestBuildDdsBytes(unittest.TestCase):
    def test_starts_with_dds_magic(self):
        out = main._build_dds_bytes(4, 4, b"\x00" * 8, b"DXT1", block_size=8)
        self.assertTrue(out.startswith(b"DDS "))

    def test_fourcc_lands_at_byte_84(self):
        out = main._build_dds_bytes(8, 8, b"\x00" * 32, b"DXT5", block_size=16)
        self.assertEqual(out[84:88], b"DXT5")

    def test_dx10_payload_starts_at_byte_148(self):
        payload = b"\xAB" * 16
        out = main._build_dds_bytes(4, 4, payload, b"DX10", dxgi_format=98, block_size=16)
        self.assertEqual(out[84:88], b"DX10")
        self.assertEqual(out[148:], payload)

    def test_no_dx10_payload_starts_at_byte_128(self):
        payload = b"\xCD" * 8
        out = main._build_dds_bytes(4, 4, payload, b"DXT1", block_size=8)
        self.assertEqual(out[128:], payload)


class TestRepackageKtx2ToDds(unittest.TestCase):
    def test_bc1_repackages_and_decodes_identically_to_full_decode(self):
        """The core correctness property: repackaging must change nothing
        about the final pixels, only the container -- decoding the
        repackaged .dds (via mesh_convert.convert's own reader) must match
        decode_ktx2_to_png's full decode of the ORIGINAL .ktx2 exactly."""
        import importlib
        convert_module = importlib.import_module("mesh_convert.convert")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            compressed = _fake_block_bytes(8)  # one 4x4 BC1 block
            ktx2_path = td / "some_albd.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 4, 4, compressed))

            dds_path = td / "some_albd.dds"
            self.assertIs(main.repackage_ktx2_to_dds(ktx2_path, dds_path), True)
            self.assertTrue(dds_path.read_bytes().startswith(b"DDS "))

            png_path = td / "some_albd_full.png"
            self.assertIs(main.decode_ktx2_to_png(ktx2_path, png_path), True)
            expected = np.array(Image.open(png_path).convert("RGBA"))

            redecoded_png = td / "redecoded.png"
            self.assertTrue(convert_module.decode_dds_bytes_to_png(dds_path.read_bytes(), redecoded_png))
            actual = np.array(Image.open(redecoded_png).convert("RGBA"))
            np.testing.assert_array_equal(actual, expected)

    def test_bc7_dx10_header_round_trips_through_the_projects_own_reader(self):
        import importlib
        convert_module = importlib.import_module("mesh_convert.convert")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            compressed = _fake_block_bytes(16)  # one 4x4 BC7 block
            ktx2_path = td / "some_roof.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC7_UNORM_BLOCK, 4, 4, compressed))

            dds_path = td / "some_roof.dds"
            self.assertIs(main.repackage_ktx2_to_dds(ktx2_path, dds_path), True)

            png_path = td / "some_roof_full.png"
            self.assertIs(main.decode_ktx2_to_png(ktx2_path, png_path), True)
            expected = np.array(Image.open(png_path).convert("RGBA"))

            redecoded_png = td / "redecoded.png"
            self.assertTrue(convert_module.decode_dds_bytes_to_png(dds_path.read_bytes(), redecoded_png))
            actual = np.array(Image.open(redecoded_png).convert("RGBA"))
            np.testing.assert_array_equal(actual, expected)

    def test_basis_supercompression_is_not_repackaged(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "some_tex.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(
                main.VK_FORMAT_BC7_UNORM_BLOCK, 4, 4, _fake_block_bytes(16),
                supercompression=main._KTX2_SUPERCOMPRESSION_BASIS))
            result = main.repackage_ktx2_to_dds(ktx2_path, td / "out.dds")
            self.assertNotEqual(result, True)
            self.assertIn("Basis", str(result))

    def test_unsupported_vkformat_is_not_repackaged(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "some_tex.ktx2"
            # VK_FORMAT_R8G8B8A8_UNORM: decode_ktx2_to_png supports it
            # (plain uncompressed pixels), but repackage_ktx2_to_dds
            # deliberately doesn't bother (no size win to chase there).
            ktx2_path.write_bytes(_build_fake_ktx2(
                main.VK_FORMAT_R8G8B8A8_UNORM, 2, 2, _fake_block_bytes(2 * 2 * 4)))
            result = main.repackage_ktx2_to_dds(ktx2_path, td / "out.dds")
            self.assertNotEqual(result, True)
            self.assertIn("Unsupported", str(result))


class TestDecodeOrRepackageKtx2(unittest.TestCase):
    def test_non_normal_texture_becomes_dds(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "roof_albd.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 4, 4, _fake_block_bytes(8)))
            result = main.decode_or_repackage_ktx2(ktx2_path, td)
            self.assertIs(result, True)
            self.assertTrue((td / "roof_albd.dds").exists())
            self.assertFalse((td / "roof_albd.png").exists())

    def test_normal_map_never_becomes_dds(self):
        """_norm textures MUST stay on the full decode+Z-reconstruct path
        -- a raw BC5 passthrough would silently drop the reconstructed Z
        channel and break lighting in-sim."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "roof_norm.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC5_UNORM_BLOCK, 4, 4, _fake_block_bytes(16)))
            result = main.decode_or_repackage_ktx2(ktx2_path, td)
            self.assertIs(result, True)
            self.assertTrue((td / "roof_norm.png").exists())
            self.assertFalse((td / "roof_norm.dds").exists())

    def test_unsupported_format_falls_back_to_full_decode(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "flat_comp.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(
                main.VK_FORMAT_R8G8B8A8_UNORM, 2, 2, _fake_block_bytes(2 * 2 * 4)))
            result = main.decode_or_repackage_ktx2(ktx2_path, td)
            self.assertIs(result, True)
            self.assertTrue((td / "flat_comp.png").exists())
            self.assertFalse((td / "flat_comp.dds").exists())


if __name__ == "__main__":
    unittest.main()
