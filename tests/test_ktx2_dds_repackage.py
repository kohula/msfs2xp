"""
main.repackage_ktx2_to_dds / decode_or_repackage_ktx2 -- repackages a
KTX2's own GPU-native block-compressed payload straight into a DDS
container instead of decoding to raw pixels and re-encoding, for
everything except normal maps (which still need decode_ktx2_to_png's real
Z-channel reconstruction -- BC5 normal maps only carry X/Y).

ONLY BC1/BC3 (legacy "DXT1"/"DXT5" FourCC) are repackaged -- CONFIRMED
REAL REGRESSION: an earlier version also repackaged BC4/BC5/BC7 via a
DDS_HEADER_DXT10 extension, which round-tripped fine through this
project's OWN reader (decode_dds_bytes_to_png) but made "almost
everything" grey with "Some scenery textures could not be loaded" in
real X-Plane 11 -- its DDS loader has no confirmed DX10/BC4-7 support
(X-Plane's own official DDSTool manual documents only DXT1/DXT3/DXT5).
Every format that can't be safely repackaged falls back to the full
decode-to-PNG path instead of guessing again.

INCLUDES A REAL MIP CHAIN -- CONFIRMED REAL BUG this also fixes: even
after the DX10 fix above, real X-Plane Log.txt output still reported
"missing" textures for structurally-valid DXT1/DXT5 files (correct
magic bytes, correct FourCC, plausible size). A byte-for-byte header
comparison against a different, real, working MSFS->X-Plane converter's
own DDS output for the exact same source texture found the real cause:
theirs had a real dwMipMapCount (11, for a 1024x1024 texture) with
DDSD_MIPMAPCOUNT/DDSCAPS_MIPMAP/DDSCAPS_COMPLEX all set; this project's
single-level output had none of those, and X-Plane 11's DDS loader
rejects a single-level file outright. The source KTX2 already has a
real mip chain (parse_ktx2_all_levels reads every level, not just
level 0, unlike parse_ktx2_header which only decode_ktx2_to_png/
decode_or_repackage_ktx2's PNG fallback still use -- PNG has no mip
concept at all, so that path never needed this).
"""
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import main


def _build_fake_ktx2(vk_format, width, height, compressed_bytes, supercompression=0, extra_levels=None):
    """A minimal, structurally-valid KTX2 file wrapping compressed_bytes
    as level 0 (the base/highest-res level), optionally followed by
    extra_levels (smaller mip levels, level 1 first) -- enough for
    main.parse_ktx2_header / parse_ktx2_all_levels (which only read by
    field index, not full spec compliance) to parse correctly. Real
    block CONTENT doesn't matter for these tests; only that repackaging
    preserves it bit-for-bit and that both decode paths agree on what
    it means."""
    levels = [compressed_bytes] + list(extra_levels or [])
    level_count = len(levels)
    magic = b"\xabKTX 20\xbb\r\n\x1a\n"
    header = struct.pack("<17I", vk_format, 1, width, height, 0, 1, 1, level_count, supercompression,
                          0, 0, 0, 0, 0, 0, 0, 0)
    cur_offset = len(magic) + len(header) + level_count * 24
    level_index = b""
    data = b""
    for lvl in levels:
        level_index += struct.pack("<3Q", cur_offset, len(lvl), len(lvl))
        data += lvl
        cur_offset += len(lvl)
    return magic + header + level_index + data


def _fake_block_bytes(n, seed=0):
    import random
    rnd = random.Random(seed)
    return bytes(rnd.randrange(256) for _ in range(n))


class TestBuildDdsBytes(unittest.TestCase):
    def test_starts_with_dds_magic(self):
        out = main._build_dds_bytes(4, 4, [b"\x00" * 8], b"DXT1", block_size=8)
        self.assertTrue(out.startswith(b"DDS "))

    def test_fourcc_lands_at_byte_84(self):
        out = main._build_dds_bytes(8, 8, [b"\x00" * 32], b"DXT5", block_size=16)
        self.assertEqual(out[84:88], b"DXT5")

    def test_dx10_payload_starts_at_byte_148(self):
        payload = b"\xAB" * 16
        out = main._build_dds_bytes(4, 4, [payload], b"DX10", dxgi_format=98, block_size=16)
        self.assertEqual(out[84:88], b"DX10")
        self.assertEqual(out[148:], payload)

    def test_no_dx10_payload_starts_at_byte_128(self):
        payload = b"\xCD" * 8
        out = main._build_dds_bytes(4, 4, [payload], b"DXT1", block_size=8)
        self.assertEqual(out[128:], payload)

    def test_single_level_has_no_mipmap_flags(self):
        """A single-level file keeps the OLD, already-proven-fine
        convention (no DDSD_MIPMAPCOUNT/DDSCAPS_MIPMAP, dwMipMapCount=0)
        -- only a real multi-level file needs the new flags."""
        out = main._build_dds_bytes(4, 4, [b"\x00" * 8], b"DXT1", block_size=8)
        dwFlags, dwHeight, dwWidth, dwPitchOrLinearSize, dwDepth, dwMipMapCount = struct.unpack_from("<6I", out, 8)
        self.assertEqual(dwFlags & 0x20000, 0, "DDSD_MIPMAPCOUNT must not be set for a single level")
        self.assertEqual(dwMipMapCount, 0)
        dwCaps = struct.unpack_from("<I", out, 76 + 32)[0]
        self.assertEqual(dwCaps & 0x400000, 0, "DDSCAPS_MIPMAP must not be set for a single level")

    def test_multi_level_sets_real_mipmap_count_and_flags(self):
        """CONFIRMED REAL BUG this pins: X-Plane 11's DDS loader rejects
        a single-level file outright ("missing texture" even with valid
        magic bytes and FourCC) -- a real converter's own working DDS
        output for the same source texture had dwMipMapCount=11 (for a
        1024x1024 texture) with DDSD_MIPMAPCOUNT/DDSCAPS_MIPMAP/
        DDSCAPS_COMPLEX all set. A multi-level file here must match."""
        level0 = b"\x00" * 8  # 4x4, 1 BC1 block
        level1 = b"\x11" * 8  # 2x2, still 1 block (BC1 pads up to a full block)
        level2 = b"\x22" * 8  # 1x1, still 1 block
        out = main._build_dds_bytes(4, 4, [level0, level1, level2], b"DXT1", block_size=8)
        dwFlags, dwHeight, dwWidth, dwPitchOrLinearSize, dwDepth, dwMipMapCount = struct.unpack_from("<6I", out, 8)
        self.assertEqual(dwFlags & 0x20000, 0x20000, "DDSD_MIPMAPCOUNT must be set")
        self.assertEqual(dwMipMapCount, 3)
        dwCaps = struct.unpack_from("<I", out, 76 + 32)[0]
        self.assertEqual(dwCaps & 0x8, 0x8, "DDSCAPS_COMPLEX must be set")
        self.assertEqual(dwCaps & 0x400000, 0x400000, "DDSCAPS_MIPMAP must be set")
        # dwPitchOrLinearSize is level 0's own size only, never the sum.
        self.assertEqual(dwPitchOrLinearSize, 8)
        # All three levels' bytes present, in order, right after the header.
        self.assertEqual(out[128:], level0 + level1 + level2)


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

    def test_real_multi_level_source_produces_a_real_mip_chain_end_to_end(self):
        """CONFIRMED REAL BUG this pins, end-to-end through
        repackage_ktx2_to_dds itself (not just _build_dds_bytes in
        isolation): a source KTX2 with a real mip chain (3 levels here,
        matching how real MSFS textures are actually authored) must
        produce a DDS whose header correctly declares that same mip
        chain (X-Plane 11 rejects a single-level DDS outright, even one
        with fully valid magic bytes and FourCC -- confirmed via a
        byte-for-byte comparison against a different, real, working
        converter's own DDS output for the same source texture), AND
        whose level-0 pixels still decode identically to a full PNG
        decode of the source's own level 0."""
        import importlib
        convert_module = importlib.import_module("mesh_convert.convert")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            level0 = _fake_block_bytes(32, seed=1)  # 8x8 -> 2x2 blocks -> 4 BC1 blocks
            level1 = _fake_block_bytes(8, seed=2)    # 4x4 -> 1 block
            level2 = _fake_block_bytes(8, seed=3)    # 2x2 -> 1 block (padded up to a full block)
            ktx2_path = td / "some_wall.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(
                main.VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 8, 8, level0,
                extra_levels=[level1, level2]))

            dds_path = td / "some_wall.dds"
            self.assertIs(main.repackage_ktx2_to_dds(ktx2_path, dds_path), True)
            dds_bytes = dds_path.read_bytes()

            dwMipMapCount = struct.unpack_from("<I", dds_bytes, 28)[0]
            self.assertEqual(dwMipMapCount, 3, "must declare the real 3-level mip chain")
            dwFlags = struct.unpack_from("<I", dds_bytes, 8)[0]
            self.assertEqual(dwFlags & 0x20000, 0x20000, "DDSD_MIPMAPCOUNT must be set")
            dwCaps = struct.unpack_from("<I", dds_bytes, 76 + 32)[0]
            self.assertEqual(dwCaps & 0x400000, 0x400000, "DDSCAPS_MIPMAP must be set")

            png_path = td / "some_wall_full.png"
            self.assertIs(main.decode_ktx2_to_png(ktx2_path, png_path), True)
            expected = np.array(Image.open(png_path).convert("RGBA"))

            redecoded_png = td / "redecoded.png"
            self.assertTrue(convert_module.decode_dds_bytes_to_png(dds_bytes, redecoded_png))
            actual = np.array(Image.open(redecoded_png).convert("RGBA"))
            np.testing.assert_array_equal(actual, expected)

    def test_bc3_repackages_and_decodes_identically_to_full_decode(self):
        import importlib
        convert_module = importlib.import_module("mesh_convert.convert")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            compressed = _fake_block_bytes(16)  # one 4x4 BC3 block
            ktx2_path = td / "some_wall.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC3_UNORM_BLOCK, 4, 4, compressed))

            dds_path = td / "some_wall.dds"
            self.assertIs(main.repackage_ktx2_to_dds(ktx2_path, dds_path), True)
            self.assertEqual(dds_path.read_bytes()[84:88], b"DXT5",
                              "must use the legacy FourCC, never a DX10 header")

            png_path = td / "some_wall_full.png"
            self.assertIs(main.decode_ktx2_to_png(ktx2_path, png_path), True)
            expected = np.array(Image.open(png_path).convert("RGBA"))

            redecoded_png = td / "redecoded.png"
            self.assertTrue(convert_module.decode_dds_bytes_to_png(dds_path.read_bytes(), redecoded_png))
            actual = np.array(Image.open(redecoded_png).convert("RGBA"))
            np.testing.assert_array_equal(actual, expected)

    def test_non_power_of_2_size_is_not_repackaged(self):
        """CONFIRMED REAL BUG: X-Plane 11's DDS loader rejects a DDS
        whose width or height isn't a power of 2 outright ("the texture
        ... is DDS but its size is not power of 2", TEX_obj.cpp:568),
        even with a correct FourCC and a real mip chain -- confirmed on
        a real 1048x1048 source texture (not a power of 2; 1024, the
        nearest one, is)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "some_background.ktx2"
            # 1048x1048: same non-power-of-2 size as the real confirmed case.
            ktx2_path.write_bytes(_build_fake_ktx2(
                main.VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 1048, 1048, _fake_block_bytes(8)))
            result = main.repackage_ktx2_to_dds(ktx2_path, td / "out.dds")
            self.assertNotEqual(result, True)
            self.assertIn("power-of-2", str(result))

    def test_power_of_2_size_is_still_repackaged(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "some_background.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(
                main.VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 1024, 1024, _fake_block_bytes(8)))
            result = main.repackage_ktx2_to_dds(ktx2_path, td / "out.dds")
            self.assertIs(result, True)

    def test_bc7_is_never_repackaged(self):
        """Regression: X-Plane 11's DDS loader has no confirmed DX10/BC7
        support -- confirmed real breakage ("almost everything" grey,
        "Some scenery textures could not be loaded") when an earlier
        version repackaged BC7 via a DDS_HEADER_DXT10 extension. BC7 is
        the common format for modern MSFS albedo/PBR textures, so this
        one matters most."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "some_roof.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC7_UNORM_BLOCK, 4, 4, _fake_block_bytes(16)))
            result = main.repackage_ktx2_to_dds(ktx2_path, td / "out.dds")
            self.assertNotEqual(result, True)
            self.assertIn("Unsupported", str(result))

    def test_bc4_is_never_repackaged(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "some_mask.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC4_UNORM_BLOCK, 4, 4, _fake_block_bytes(8)))
            result = main.repackage_ktx2_to_dds(ktx2_path, td / "out.dds")
            self.assertNotEqual(result, True)
            self.assertIn("Unsupported", str(result))

    def test_bc5_is_never_repackaged(self):
        """BC5 is excluded from repackaging entirely (not just via the
        "_norm" filename routing in decode_or_repackage_ktx2) since it
        also needs the unsupported DX10 header regardless of what it's
        used for."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "some_data.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC5_UNORM_BLOCK, 4, 4, _fake_block_bytes(16)))
            result = main.repackage_ktx2_to_dds(ktx2_path, td / "out.dds")
            self.assertNotEqual(result, True)
            self.assertIn("Unsupported", str(result))

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

    def test_no_crash_when_the_stale_file_is_already_gone(self):
        """CONFIRMED REAL CRASH this pins: decode_or_repackage_ktx2 runs
        under a ProcessPoolExecutor, one worker per .ktx2 file, and two
        different SOURCE .ktx2 files can legitimately clean to the SAME
        stem (two differently-pathed copies of one shared texture) and
        race on cleaning up the same stale cross-format file
        concurrently. An exists()-then-unlink() TOCTOU window let a
        second worker's unlink() raise FileNotFoundError after a first
        worker already removed the same file -- an unhandled exception
        that silently aborted the ENTIRE pipeline mid-Step-2, before
        model conversion ever started, with no "[error]"-tagged log line
        and exit code 0. Both directions (repackaging to .dds with no
        stale .png to clean up, and falling back to .png with no stale
        .dds to clean up) must be a clean no-op, not a crash."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "roof_albd.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 4, 4, _fake_block_bytes(8)))
            # No stale .png sitting here at all (already removed by
            # another concurrent worker, or never existed).
            result = main.decode_or_repackage_ktx2(ktx2_path, td)
            self.assertIs(result, True)

            ktx2_path2 = td / "roof2_albd.ktx2"
            ktx2_path2.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC7_UNORM_BLOCK, 4, 4, _fake_block_bytes(16)))
            # No stale .dds sitting here either.
            result2 = main.decode_or_repackage_ktx2(ktx2_path2, td)
            self.assertIs(result2, True)

    def test_stale_dds_from_an_older_run_is_removed_when_now_falling_back_to_png(self):
        """CONFIRMED REAL BUG: run_convert_resume.py's output textures
        folder isn't cleared between runs. A .dds written under an OLDER
        version of repackage_ktx2_to_dds (before BC4/BC5/BC7 were
        excluded) can still be sitting there after an upgrade makes this
        run correctly fall back to .png for the same stem -- and
        extract_image's own "prefer an existing .dds" check has no way
        to know that leftover file is stale, reintroducing the exact
        DX10-header breakage that was supposedly fixed. Confirmed in
        real X-Plane Log.txt output: "we are missing the texture" for
        several *_albd.dds files that had both a stale bad .dds and a
        fresh, correct .png sitting side by side."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            stale_dds = td / "roof_albd.dds"
            stale_dds.write_bytes(b"DDS " + b"\x00" * 200)  # stands in for an old DX10-header file

            ktx2_path = td / "roof_albd.ktx2"
            # BC7: unsupported for repackaging, must fall back to .png.
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC7_UNORM_BLOCK, 4, 4, _fake_block_bytes(16)))
            result = main.decode_or_repackage_ktx2(ktx2_path, td)
            self.assertIs(result, True)
            self.assertTrue((td / "roof_albd.png").exists())
            self.assertFalse(stale_dds.exists(), "the stale .dds must be removed, not left to shadow the fresh .png")

    def test_stale_png_from_an_older_run_is_removed_when_now_repackaging_to_dds(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            stale_png = td / "roof_albd.png"
            stale_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

            ktx2_path = td / "roof_albd.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 4, 4, _fake_block_bytes(8)))
            result = main.decode_or_repackage_ktx2(ktx2_path, td)
            self.assertIs(result, True)
            self.assertTrue((td / "roof_albd.dds").exists())
            self.assertFalse(stale_png.exists(), "the stale .png must be removed, not left to shadow the fresh .dds")

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

    def test_bc7_non_normal_texture_falls_back_to_png_not_dds(self):
        """End-to-end regression for the real breakage: a non-"_norm"
        BC7 texture (the common case -- most modern MSFS albedo textures)
        must still become .png, not a DX10-header .dds X-Plane 11 can't
        load."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "roof_albd.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC7_UNORM_BLOCK, 4, 4, _fake_block_bytes(16)))
            result = main.decode_or_repackage_ktx2(ktx2_path, td)
            self.assertIs(result, True)
            self.assertTrue((td / "roof_albd.png").exists())
            self.assertFalse((td / "roof_albd.dds").exists())


class TestAtomicWrites(unittest.TestCase):
    """decode_ktx2_to_png and repackage_ktx2_to_dds both write to a
    per-call unique temp path and os.replace() into place, instead of
    writing straight to the final output path -- CONFIRMED REAL CRASH
    this fixes: this runs under a ProcessPoolExecutor, one worker per
    .ktx2 file, and two different source .ktx2 files can legitimately
    clean to the same stem (the same collision class already fixed once
    for decode_or_repackage_ktx2's cross-format cleanup). Two workers
    both writing straight to the SAME output path concurrently
    interleaved their writes into a corrupted PNG (a real IDAT CRC
    error), which is a hard X-Plane crash ("THREAD FATAL ASSERT"), not
    just a bad-looking texture. These tests only confirm the ordinary,
    non-racing case leaves no leftover temp file behind -- a true
    concurrent race isn't practical to simulate directly in a unit
    test."""

    def test_decode_ktx2_to_png_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "roof_albd.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 4, 4, _fake_block_bytes(8)))
            png_path = td / "roof_albd.png"
            self.assertIs(main.decode_ktx2_to_png(ktx2_path, png_path), True)
            self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG"))
            leftovers = [p.name for p in td.iterdir() if ".tmp_" in p.name]
            self.assertEqual(leftovers, [], f"leftover temp file(s): {leftovers}")

    def test_repackage_ktx2_to_dds_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ktx2_path = td / "roof_albd.ktx2"
            ktx2_path.write_bytes(_build_fake_ktx2(main.VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 4, 4, _fake_block_bytes(8)))
            dds_path = td / "roof_albd.dds"
            self.assertIs(main.repackage_ktx2_to_dds(ktx2_path, dds_path), True)
            self.assertTrue(dds_path.read_bytes().startswith(b"DDS "))
            leftovers = [p.name for p in td.iterdir() if ".tmp_" in p.name]
            self.assertEqual(leftovers, [], f"leftover temp file(s): {leftovers}")


if __name__ == "__main__":
    unittest.main()
