"""
airport_layout.decode_airport_layout against synthetic airport records
built field by field to the documented layouts (see airport_layout's
module docstring). Real BGL content is proprietary and can't be
reproduced here, so these pin the format understanding itself:

- the sub-record chain is found for both the MSFS 2020 head (0x003C /
  0x0056, sub-records from 0x44) and the MSFS 2024 head (0x0113, from 0x5C);
- a taxi path's end node is read from the element itself, not guessed from
  the next path's start;
- parking elements are variable length (4 bytes per airline code);
- the last taxi point of a record is not dropped.
"""
import struct
import unittest

import airport_layout
from bgl_extractor import decode_lonlat_dword

ARP = (47.4369, 19.2556)


def _lonlat(lat, lon):
    lat_raw = round((90.0 - lat) * (536870912.0 / 180.0))
    lon_raw = round((lon + 180.0) * (805306368.0 / 360.0))
    return struct.pack("<II", lon_raw, lat_raw)


def _rec(rec_type, body):
    return struct.pack("<HI", rec_type, 6 + len(body)) + body


def _ident(s):
    v = 0
    for ch in s:
        v = v * 38 + (ord(ch) - ord("0") + 2 if ch.isdigit() else ord(ch) - ord("A") + 12)
    return v


def _airport(subs, rec_id=0x003C, ident="LHBP"):
    head = struct.pack("<HHH", 1, 0, 0)  # counts
    head += _lonlat(*ARP) + struct.pack("<i", 151000)
    head += _lonlat(ARP[0] + 0.002, ARP[1] + 0.002) + struct.pack("<i", 190000)
    head += struct.pack("<f", -5.0) + struct.pack("<III", _ident(ident) << 5, _ident("LH") << 5, 0)
    head += b"\x00" * 16  # MSFS 2020 tail (to 0x44)
    if rec_id == 0x0113:
        tail = bytearray(24)
        head = head[:0x4C - 6 - 16] + head[-16:]  # keep length; ident moves to 0x4C below
        head = bytearray(head) + tail
        struct.pack_into("<I", head, 40 - 6, 0)
        struct.pack_into("<Q", head, 0x4C - 6, _ident(ident) << 6)
        head = bytes(head)
    return _rec(rec_id, head + b"".join(subs))


def _runway(lat, lon, length, width, heading, extra_subs=b""):
    body = struct.pack("<H", 4)  # asphalt
    body += bytes([13, 1, 31, 2])  # 13L / 31R
    body += struct.pack("<II", 0, 0) + _lonlat(lat, lon) + struct.pack("<i", 150000)
    body += struct.pack("<ffff", length, width, heading, 300.0)
    body += struct.pack("<HBB", 0x63, 0b1011, 0)
    body += b"\x00" * 44  # MSFS: unknowns + material GUID
    return _rec(0x00CE, body + extra_subs)


def _threshold(rec_type, metres):
    return _rec(rec_type, struct.pack("<H", 4) + b"\x00" * 16 + struct.pack("<ff", metres, 45.0))


def _papi(rec_type, pitch):
    return _rec(rec_type, struct.pack("<Hffff", 8, -20.0, 300.0, 9.0, pitch))


def _taxi_points(points):
    body = struct.pack("<H", len(points))
    for lat, lon, kind in points:
        body += bytes([kind, 0, 0, 0]) + _lonlat(lat, lon)
    return _rec(0x001A, body)


def _taxi_names(names):
    body = struct.pack("<H", len(names))
    for n in names:
        body += n.encode().ljust(8, b"\x00")
    return _rec(0x001D, body)


def _taxi_paths(paths):
    body = struct.pack("<H", len(paths))
    for start, end, kind, name_idx, width in paths:
        flags = (2 << 12) if kind == 2 else 0  # runway paths: designator (R) in the top 4 bits
        el = struct.pack("<HHBBBB", start, flags, kind, name_idx, 0b11, 4) + struct.pack("<f", width)
        el += b"\x00" * (46 - len(el)) + struct.pack("<H", end)
        assert len(el) == 48
        body += el
    return _rec(0x00D4, body)


def _parkings(stands):
    body = struct.pack("<H", len(stands))
    for lat, lon, heading, radius, kind, name_code, number, airlines in stands:
        flags = name_code | (kind << 8) | (number << 12) | (len(airlines) << 24)
        body += struct.pack("<Iff", flags, radius, heading) + b"\x00" * 16 + _lonlat(lat, lon)
        for a in airlines:
            body += a.encode().ljust(4, b"\x00")
        body += b"\x00" * 20
    return _rec(0x00E7, body)


def _apron(vertices):
    body = b"\x00" * (0x30 - 6) + struct.pack("<H", len(vertices)) + b"\x00\x00"
    for v in vertices:
        body += _lonlat(*v)
    return _rec(0x00D0, body)


def _light_string(vertices, name):
    body = b"\x00\x00" + struct.pack("<HH", len(vertices), len(name)) + b"\x00" * 12
    for v in vertices:
        body += _lonlat(*v)
    return _rec(0x0031, body + name.encode())


def _painted_line(vertices, style, lit, count_extra=0):
    body = bytes([(style << 1) | int(lit), 0]) + struct.pack("<H", len(vertices) + count_extra)
    body += b"\x00" * 2 + b"\x11" * 16
    for v in vertices:
        body += _lonlat(*v)
    return _rec(0x00CF, body)


def _sign(lat, lon, heading, label):
    body = b"\x00\x00" + _lonlat(lat, lon) + struct.pack("<i", 0) + struct.pack("<f", heading)
    body += bytes([2]) + label.encode() + b"\x00"
    return _rec(0x00D9, body)


def _com(kind, hz, name):
    return _rec(0x0012, struct.pack("<HI", kind, hz) + name.encode().ljust(0x30, b"\x00"))


def _full_airport(rec_id=0x003C):
    subs = [
        _rec(0x0019, b"Budapest Liszt Ferenc\x00"),
        _runway(47.4300, 19.2500, 3010.0, 45.0, 135.0,
                _threshold(0x0005, 150.0) + _papi(0x000B, 3.0) + _rec(0x00DF, bytes([0x03 | 0x40, 0]))),
        _com(6, 118_100_000, "BUDAPEST TOWER"),
        _taxi_points([(47.4370, 19.2550, 1), (47.4372, 19.2552, 2), (47.4374, 19.2554, 1)]),
        _taxi_names(["", "A", "B"]),
        _taxi_paths([(0, 1, 1, 1, 23.0), (1, 2, 2, 31, 45.0), (0, 0, 3, 2, 20.0)]),
        _parkings([(47.4380, 19.2560, 90.0, 30.0, 0x0A, 0x0C, 12, ["MAH", "WZZ"]),
                   (47.4382, 19.2562, 180.0, 10.0, 0x01, 0x01, 3, []),
                   (47.4384, 19.2564, 0.0, 5.0, 0x0C, 0x01, 4, [])]),
        _apron([(47.437, 19.255), (47.438, 19.255), (47.438, 19.256), (47.437, 19.256)]),
        _light_string([(47.437, 19.255), (47.4372, 19.2552)], "TAXIWAY_CENTER_GREEN"),
        _painted_line([(47.437, 19.255), (47.4371, 19.2551)], style=1, lit=True),
        _sign(47.4373, 19.2553, 45.0, "l[A]d[B>]"),
    ]
    return _airport(subs, rec_id=rec_id)


class TestAirportRecord(unittest.TestCase):
    def _check(self, layout):
        self.assertEqual(layout.ident, "LHBP")
        self.assertEqual(layout.name, "Budapest Liszt Ferenc")
        self.assertAlmostEqual(layout.lat, ARP[0], places=5)
        self.assertAlmostEqual(layout.alt_m, 151.0)
        self.assertEqual(len(layout.runways), 1)
        self.assertEqual(len(layout.taxi_points), 3, "the last taxi point is kept")
        self.assertEqual(len(layout.parkings), 3)
        self.assertEqual(len(layout.aprons), 1)
        self.assertEqual(len(layout.light_strings), 1)
        self.assertEqual(len(layout.painted_lines), 1)
        self.assertEqual(len(layout.signs), 1)
        self.assertEqual(len(layout.coms), 1)

    def test_msfs2020_head(self):
        self._check(airport_layout.decode_airport_layout(_full_airport(0x003C)))

    def test_msfs2020_newer_record_id(self):
        self._check(airport_layout.decode_airport_layout(_full_airport(0x0056)))

    def test_msfs2024_head_and_64bit_ident(self):
        self._check(airport_layout.decode_airport_layout(_full_airport(0x0113)))

    def test_runway_fields_and_subrecords(self):
        rw = airport_layout.decode_airport_layout(_full_airport()).runways[0]
        self.assertAlmostEqual(rw.length_m, 3010.0)
        self.assertAlmostEqual(rw.width_m, 45.0)
        self.assertAlmostEqual(rw.heading_true, 135.0)
        self.assertEqual(rw.primary.name, "13L")
        self.assertEqual(rw.secondary.name, "31R")
        self.assertAlmostEqual(rw.primary.displaced_m, 150.0)
        v = rw.primary.vasi[0]
        self.assertEqual((v.kind, v.side, v.pitch, v.bias_x, v.bias_z), (8, "L", 3.0, -20.0, 300.0))
        self.assertEqual(rw.primary.approach_system, 3)
        self.assertTrue(rw.primary.reil)
        self.assertEqual(rw.edge_lights, 3)
        self.assertEqual(rw.centre_lights, 2)

    def test_taxi_paths_use_their_own_end_node(self):
        layout = airport_layout.decode_airport_layout(_full_airport())
        paths = layout.taxi_paths
        self.assertEqual([(p.start, p.end, p.kind) for p in paths], [(0, 1, 1), (1, 2, 2), (0, 0, 3)])
        self.assertEqual(paths[0].name, "A")
        self.assertEqual(paths[1].name, "31R", "a runway path is named after its runway end")
        self.assertAlmostEqual(paths[0].width_m, 23.0)
        self.assertTrue(paths[0].centre_line and paths[0].centre_lit)
        # The parking path's end numbers stand 0, which follows the 3 taxi points.
        self.assertEqual(layout.taxi_edges, [(0, 1), (1, 2), (0, 3)])

    def test_parkings_with_airline_codes(self):
        stands = airport_layout.decode_airport_layout(_full_airport()).parkings
        self.assertEqual(stands[0].airlines, ["MAH", "WZZ"])
        self.assertEqual(stands[0].display_name, "A12")
        self.assertAlmostEqual(stands[0].heading, 90.0)
        self.assertAlmostEqual(stands[0].radius_m, 30.0)
        self.assertEqual(stands[1].display_name, "Parking 3")
        self.assertAlmostEqual(stands[1].lat, 47.4382, places=5, msg="later stands stay aligned")
        self.assertAlmostEqual(stands[2].lat, 47.4384, places=5)
        layout = airport_layout.decode_airport_layout(_full_airport())
        self.assertEqual(len(layout.ramp_starts), 2, "the fuel stand isn't a ramp start")

    def test_light_string_preset_decides_colour(self):
        ls = airport_layout.decode_airport_layout(_full_airport()).light_strings[0]
        self.assertEqual(ls.name, "TAXIWAY_CENTER_GREEN")
        self.assertEqual(ls.light_type, 101)

    def test_painted_line_style_and_lit_bit(self):
        pl = airport_layout.decode_airport_layout(_full_airport()).painted_lines[0]
        self.assertEqual(pl.style, 1)
        self.assertTrue(pl.lit)
        self.assertEqual(len(pl.vertices), 2)

    def test_painted_line_count_one_too_high_still_decodes(self):
        blob = _airport([_painted_line([(47.437, 19.255), (47.4371, 19.2551), (47.4372, 19.2552)],
                                       style=0, lit=False, count_extra=1)])
        pl = airport_layout.decode_airport_layout(blob).painted_lines
        self.assertEqual(len(pl), 1)
        self.assertEqual(len(pl[0].vertices), 3)

    def test_sign(self):
        s = airport_layout.decode_airport_layout(_full_airport()).signs[0]
        self.assertEqual(s.label, "l[A]d[B>]")
        self.assertAlmostEqual(s.heading, 45.0)
        self.assertAlmostEqual(s.lat, 47.4373, places=5)

    def test_com(self):
        c = airport_layout.decode_airport_layout(_full_airport()).coms[0]
        self.assertEqual((c.kind, c.freq_khz, c.name), (6, 118100, "BUDAPEST TOWER"))

    def test_garbage_is_an_empty_layout(self):
        self.assertTrue(airport_layout.decode_airport_layout(b"\x00" * 0x60).is_empty())


class TestHelpers(unittest.TestCase):
    def test_runway_names(self):
        self.assertEqual(airport_layout.runway_name(9, 1), "09L")
        self.assertEqual(airport_layout.runway_name(27, 0), "27")
        self.assertEqual(airport_layout.runway_name(38, 0), "NE")

    def test_ident_decoding(self):
        self.assertEqual(airport_layout.decode_ident(_ident("KORD") << 5), "KORD")
        self.assertEqual(airport_layout.decode_ident64(_ident("EGLC") << 6), "EGLC")

    def test_decode_lonlat_matches_bgl_extractor(self):
        lon_raw, lat_raw = struct.unpack("<II", _lonlat(47.43, 19.26))
        self.assertAlmostEqual(decode_lonlat_dword(lat_raw, is_lat=True), 47.43, places=4)
        self.assertAlmostEqual(decode_lonlat_dword(lon_raw, is_lat=False), 19.26, places=4)


if __name__ == "__main__":
    unittest.main()
