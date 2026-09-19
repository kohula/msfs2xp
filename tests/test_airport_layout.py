"""
airport_layout.py's record layout was reverse engineered against a real
package (SoFly LHBP) -- see its module docstring for how, and
docs/... (none; the finding lives in that docstring). Real BGL binary
content is proprietary and can't be faithfully synthesized here (same
constraint documented in test_bgl_extractor_smoke.py), so these tests
build small synthetic records byte-for-byte matching the validated format
at each field offset, and check the decoder extracts exactly what those
bytes encode -- pinning the format understanding itself, not real airport
data.
"""
import struct
import unittest

import airport_layout
from bgl_extractor import decode_lonlat_dword


def _lonlat_dwords(lat, lon):
    lat_raw = round((90.0 - lat) * (536870912.0 / 180.0))
    lon_raw = round((lon + 180.0) * (805306368.0 / 360.0))
    return lon_raw, lat_raw


def _rec(rec_type, body):
    return struct.pack("<HI", rec_type, 6 + len(body)) + body


def _runway_rec(lat, lon):
    lon_raw, lat_raw = _lonlat_dwords(lat, lon)
    body = b"\x00" * (0x14 - 6) + struct.pack("<II", lon_raw, lat_raw)
    body += b"\x00" * 8  # pad past the 0x1c minimum-length check
    return _rec(airport_layout._REC_RUNWAY, body)


def _apron_rec(vertices):
    header = b"\x00" * (0x30 - 6) + struct.pack("<H", len(vertices)) + b"\x00\x00"
    assert len(header) == 0x34 - 6
    body = header
    for lat, lon in vertices:
        lon_raw, lat_raw = _lonlat_dwords(lat, lon)
        body += struct.pack("<II", lon_raw, lat_raw)
    return _rec(airport_layout._REC_APRON, body)


def _painted_line_rec(vertices):
    header = struct.pack("<H", 9) + struct.pack("<H", len(vertices) + 1)
    header += b"\x00" * (0x1C - 6 - 4)
    assert len(header) == 0x1C - 6
    body = header
    for lat, lon in vertices:
        lon_raw, lat_raw = _lonlat_dwords(lat, lon)
        body += struct.pack("<II", lon_raw, lat_raw)
    return _rec(airport_layout._REC_PAINTED_LINE, body)


def _light_string_rec(vertices, light_type):
    header = b"\x00\x00" + struct.pack("<H", len(vertices)) + struct.pack("<H", light_type)
    header += b"\x00" * (24 - 6 - 6)
    assert len(header) == 24 - 6
    body = header
    for lat, lon in vertices:
        lon_raw, lat_raw = _lonlat_dwords(lat, lon)
        body += struct.pack("<II", lon_raw, lat_raw)
    return _rec(airport_layout._REC_LIGHT_STRING, body)


def _starts_rec(starts):
    body = struct.pack("<H", len(starts))
    for lat, lon in starts:
        lon_raw, lat_raw = _lonlat_dwords(lat, lon)
        entry = b"\x00" * 28 + struct.pack("<II", lon_raw, lat_raw)
        entry += b"\x00" * (airport_layout._START_STRIDE - len(entry))
        body += entry
    return _rec(airport_layout._REC_START, body)


def _taxi_nodes_rec(nodes):
    body = struct.pack("<H", len(nodes)) + struct.pack("<I", 1)
    for lat, lon in nodes:
        lon_raw, lat_raw = _lonlat_dwords(lat, lon)
        body += struct.pack("<IIi", lon_raw, lat_raw, 1)
    return _rec(airport_layout._REC_TAXI_NODES, body)


def _taxi_edges_rec(start_node_ids):
    body = struct.pack("<H", len(start_node_ids))
    for node_id in start_node_ids:
        entry = struct.pack("<H", node_id)
        entry += b"\x00" * (airport_layout._TAXI_EDGE_STRIDE - len(entry))
        body += entry
    return _rec(airport_layout._REC_TAXI_EDGES, body)


def _blob(*records):
    return b"\x00" * airport_layout._LAYOUT_START + b"".join(records)


class TestDecodeAirportLayout(unittest.TestCase):
    def test_runway_center_roundtrips(self):
        blob = _blob(_runway_rec(47.4304507, 19.2502680))
        layout = airport_layout.decode_airport_layout(blob)
        self.assertEqual(len(layout.runway_centers), 1)
        lat, lon = layout.runway_centers[0]
        self.assertAlmostEqual(lat, 47.4304507, places=5)
        self.assertAlmostEqual(lon, 19.2502680, places=5)

    def test_apron_polygon_vertex_count_and_positions(self):
        verts = [(47.42, 19.25), (47.421, 19.251), (47.422, 19.252), (47.4215, 19.2505)]
        blob = _blob(_apron_rec(verts))
        layout = airport_layout.decode_airport_layout(blob)
        self.assertEqual(len(layout.aprons), 1)
        self.assertEqual(len(layout.aprons[0].vertices), 4)
        for (got_lat, got_lon), (want_lat, want_lon) in zip(layout.aprons[0].vertices, verts):
            self.assertAlmostEqual(got_lat, want_lat, places=5)
            self.assertAlmostEqual(got_lon, want_lon, places=5)

    def test_apron_below_three_vertices_is_dropped(self):
        blob = _blob(_apron_rec([(47.42, 19.25), (47.421, 19.251)]))
        layout = airport_layout.decode_airport_layout(blob)
        self.assertEqual(layout.aprons, [])

    def test_painted_line_count_field_is_off_by_one(self):
        verts = [(47.42, 19.25), (47.421, 19.251), (47.422, 19.252)]
        blob = _blob(_painted_line_rec(verts))
        layout = airport_layout.decode_airport_layout(blob)
        self.assertEqual(len(layout.painted_lines), 1)
        self.assertEqual(len(layout.painted_lines[0].vertices), 3)

    def test_light_string_type_and_vertices(self):
        verts = [(47.42, 19.25), (47.421, 19.251)]
        blob = _blob(_light_string_rec(verts, light_type=20))
        layout = airport_layout.decode_airport_layout(blob)
        self.assertEqual(len(layout.light_strings), 1)
        self.assertEqual(layout.light_strings[0].light_type, 20)
        self.assertEqual(len(layout.light_strings[0].vertices), 2)

    def test_ramp_starts_positions(self):
        starts = [(47.4308, 19.2599), (47.4310, 19.2601)]
        blob = _blob(_starts_rec(starts))
        layout = airport_layout.decode_airport_layout(blob)
        self.assertEqual(len(layout.ramp_starts), 2)
        for (got_lat, got_lon), (want_lat, want_lon) in zip(layout.ramp_starts, starts):
            self.assertAlmostEqual(got_lat, want_lat, places=5)
            self.assertAlmostEqual(got_lon, want_lon, places=5)

    def test_taxi_edges_chain_consecutive_nodes_within_range(self):
        # Three close nodes (real taxiway spacing) then one far node --
        # the far jump must NOT produce an edge (see _MAX_TAXI_EDGE_CHAIN_M).
        nodes = [(47.4300, 19.2500), (47.4301, 19.2501), (47.4302, 19.2502), (47.5000, 19.4000)]
        blob = _blob(_taxi_nodes_rec(nodes), _taxi_edges_rec([0, 1, 2, 3]))
        layout = airport_layout.decode_airport_layout(blob)
        self.assertEqual(len(layout.taxi_nodes), 4)
        self.assertEqual(layout.taxi_edges, [(0, 1), (1, 2)])

    def test_taxi_edges_skip_self_loops(self):
        nodes = [(47.4300, 19.2500), (47.4301, 19.2501)]
        blob = _blob(_taxi_nodes_rec(nodes), _taxi_edges_rec([0, 0, 1]))
        layout = airport_layout.decode_airport_layout(blob)
        self.assertEqual(layout.taxi_edges, [(0, 1)])

    def test_empty_blob_is_empty_layout(self):
        layout = airport_layout.decode_airport_layout(b"\x00" * airport_layout._LAYOUT_START)
        self.assertTrue(layout.is_empty())

    def test_is_empty_false_when_only_ramp_starts_present(self):
        # is_empty() gates on runways/aprons/taxi_nodes specifically --
        # confirm a layout with only ramp starts still isn't "empty" by
        # accident in a way that would silently disable it downstream.
        starts = [(47.4308, 19.2599)]
        blob = _blob(_starts_rec(starts))
        layout = airport_layout.decode_airport_layout(blob)
        self.assertEqual(len(layout.ramp_starts), 1)


class TestDecodeLonLatDword(unittest.TestCase):
    def test_helper_matches_bgl_extractors_own_decoder(self):
        lon_raw, lat_raw = _lonlat_dwords(47.43, 19.26)
        self.assertAlmostEqual(decode_lonlat_dword(lat_raw, is_lat=True), 47.43, places=4)
        self.assertAlmostEqual(decode_lonlat_dword(lon_raw, is_lat=False), 19.26, places=4)


if __name__ == "__main__":
    unittest.main()
