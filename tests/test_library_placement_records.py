"""bgl_extractor.decode_library_placement: both placement record sizes,
GUID anchored on the record end, and the placement scale."""
import struct
import unittest

import bgl_extractor

GUID = bytes(range(0x10, 0x20))


def _lonlat(lat, lon):
    return struct.pack("<II", round((lon + 180.0) * (805306368.0 / 360.0)),
                       round((90.0 - lat) * (536870912.0 / 180.0)))


def _head(rec_type, size, lat, lon, heading_u16):
    return (struct.pack("<HH", rec_type, size) + _lonlat(lat, lon) + struct.pack("<i", 2500)
            + struct.pack("<HHHH", 1, 0, 0, heading_u16) + struct.pack("<HH", 0, 0) + b"\xAA" * 16)


def _libobj(lat, lon, heading_u16, scale, hires=None):
    rec = _head(0x000B, 92 if hires else 64, lat, lon, heading_u16)
    if hires:
        plat, plon, fine = hires
        rec += struct.pack("<dd", plat, plon) + b"\x00" * 8 + struct.pack("<I", fine)
    return rec + GUID + struct.pack("<f", scale)


def _container(lat, lon, scale):
    return _head(0x001B, 64, lat, lon, 0) + struct.pack("<f", scale) + GUID


class TestLibraryPlacement(unittest.TestCase):
    def test_classic_64_byte_record(self):
        rec = _libobj(47.43, 19.25, 0x4000, 1.0)
        self.assertEqual(len(rec), 64)
        d = bgl_extractor.decode_library_placement(rec)
        self.assertEqual(d["guid"], GUID.hex())
        self.assertAlmostEqual(d["hdg"], 90.0)
        self.assertAlmostEqual(d["alt"], 2.5)
        self.assertTrue(d["is_agl"])
        self.assertEqual(d["scale"], 1.0)

    def test_92_byte_record_guid_precise_position_and_heading(self):
        fine = (0x4000 << 16) | 0x1234
        rec = _libobj(47.43, 19.25, 0x4000, 2.5, hires=(47.4300001234, 19.2500004321, fine))
        self.assertEqual(len(rec), 92)
        d = bgl_extractor.decode_library_placement(rec)
        self.assertEqual(d["guid"], GUID.hex(), "GUID is read from the end, not offset 44")
        self.assertEqual(d["lat"], 47.4300001234)
        self.assertEqual(d["lon"], 19.2500004321)
        self.assertAlmostEqual(d["hdg"], fine * 360.0 / 2 ** 32)
        self.assertAlmostEqual(d["scale"], 2.5)

    def test_precise_copy_ignored_when_it_disagrees(self):
        rec = _libobj(47.43, 19.25, 0, 1.0, hires=(10.0, 10.0, 0x12345678))
        d = bgl_extractor.decode_library_placement(rec)
        self.assertAlmostEqual(d["lat"], 47.43, places=5)
        self.assertAlmostEqual(d["hdg"], 0.0)

    def test_container_guid_and_scale(self):
        d = bgl_extractor.decode_library_placement(_container(47.43, 19.25, 0.5))
        self.assertEqual(d["guid"], GUID.hex())
        self.assertAlmostEqual(d["scale"], 0.5)

    def test_garbage_scale_falls_back_to_one(self):
        d = bgl_extractor.decode_library_placement(_libobj(47.43, 19.25, 0, float("nan")))
        self.assertEqual(d["scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
