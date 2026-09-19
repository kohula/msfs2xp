"""Native MSFS airport-layout extraction.

Decodes runway/apron/taxiway/painted-line/light-string/ramp-start records
directly from the MSFS package's own BGL "Airport" section, as an
alternative data source to apt_dat.py's real-world-stock-block lookup.
apt_dat.py's own docstring explains why it normally reuses X-Plane's stock
Global Airports block instead of decoding this from the BGL: there's no
MSFS equivalent of X-Plane's ATC taxi-route graph, so *some* real graph is
needed. This module supplies that graph from the package itself instead,
so a custom-rebuilt airport's own taxiway/apron layout (which may differ
from the real-world stock block) is what X-Plane's ATC/AI actually route
against, and so runway/taxiway/apron positions agree with this project's
own MSFS-derived 3D pavement rendering rather than a real-world source with
no particular relationship to it.

Record layout below was reverse engineered against a real package
(SoFly LHBP) by locating the offset at which a plain 6-byte (type u16,
size u32) record stream parses cleanly to the exact end of the primary
0x0113 "Airport" record's payload, then validating each record type's
decoded coordinates/counts against two independent ground truths: X-Plane's
own real-world stock apt.dat block for the same airport (runway centers
decoded within 2-5m of the real published thresholds), and a reference
converter's own reported per-category counts for this exact package
(runways/apron pieces/painted lines/light strings/ramp starts/taxi nodes
all match exactly). Every count field below was checked against EVERY
record of its type in that package with zero mismatches, not just a
sample.
"""

import math
import struct
from dataclasses import dataclass, field as _dc_field

from geo_transform import decode_lonlat_dword

_LAYOUT_START = 0x5C
"""Fixed offset where the nested record stream begins inside the primary
0x0113 Airport record, past its own reference-point/count header (the ARP
lon/lat/alt bgl_extractor.py already reads live at offset 12/16/20). Not
derived from any documented field -- found by brute-force offset search;
kept as a constant because every offset in [0x10, 0x100) except this one
either breaks mid-stream or yields nonsense record-type histograms."""

_REC_RUNWAY = 0x00CE
_REC_APRON = 0x00D0
_REC_PAINTED_LINE = 0x00CF
_REC_LIGHT_STRING = 0x0031
_REC_START = 0x00E7
_REC_TAXI_NODES = 0x001A
_REC_TAXI_EDGES = 0x00D4

_START_STRIDE = 56
_START_LONLAT_OFFSET = 28
_TAXI_NODE_HEADER = 12
_TAXI_NODE_STRIDE = 12
_TAXI_EDGE_HEADER = 8
_TAXI_EDGE_STRIDE = 48

_MAX_TAXI_EDGE_CHAIN_M = 500.0
"""Consecutive entries in the taxi-edge array chain together: edge i
connects node[start[i]] to node[start[i+1]]. Validated against LHBP's own
2470-entry array: the median chained-node distance is 5.5m (real taxiway
segment spacing), with the long tail being where the array moves from the
end of one taxiway chain to the start of an unrelated one, not a real
edge -- those non-edges are dropped instead of emitted as a bogus
500m+ "taxiway"."""


@dataclass
class Polygon:
    vertices: list  # [(lat, lon), ...]


@dataclass
class LightString:
    vertices: list  # [(lat, lon), ...]
    light_type: int


@dataclass
class AirportLayout:
    runway_centers: list = _dc_field(default_factory=list)  # [(lat, lon), ...]
    aprons: list = _dc_field(default_factory=list)  # [Polygon, ...]
    painted_lines: list = _dc_field(default_factory=list)  # [Polygon, ...]
    light_strings: list = _dc_field(default_factory=list)  # [LightString, ...]
    ramp_starts: list = _dc_field(default_factory=list)  # [(lat, lon), ...]
    taxi_nodes: list = _dc_field(default_factory=list)  # [(lat, lon), ...]
    taxi_edges: list = _dc_field(default_factory=list)  # [(node_a, node_b), ...]

    def is_empty(self):
        return not (self.runway_centers or self.aprons or self.taxi_nodes)


def _walk_records(blob, start):
    pos, end = start, len(blob)
    out = []
    while pos + 6 <= end:
        rec_type, rec_size = struct.unpack_from("<HI", blob, pos)
        if rec_size < 6 or pos + rec_size > end:
            break
        out.append((rec_type, blob[pos:pos + rec_size]))
        pos += rec_size
    return out


def _read_vertex_array(raw, start, count):
    verts = []
    o = start
    for _ in range(max(0, count)):
        if o + 8 > len(raw):
            break
        lon_raw, lat_raw = struct.unpack_from("<II", raw, o)
        verts.append((decode_lonlat_dword(lat_raw, is_lat=True),
                      decode_lonlat_dword(lon_raw, is_lat=False)))
        o += 8
    return verts


def _dist_m(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    dx = (lon2 - lon1) * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    dy = (lat2 - lat1) * 110540.0
    return math.hypot(dx, dy)


def decode_airport_layout(blob: bytes) -> AirportLayout:
    """Decode one Airport section's raw subsection bytes (the same blob
    bgl_extractor.py already stashes per-section into result["airport_sections"])
    into an AirportLayout. Returns an empty AirportLayout (is_empty() True)
    on anything unrecognized -- callers should treat that the same as "no
    native layout available" and fall back to the stock-block path, not as
    an error."""
    layout = AirportLayout()

    for rec_type, raw in _walk_records(blob, _LAYOUT_START):
        if rec_type == _REC_RUNWAY and len(raw) >= 0x1C:
            lon_raw, lat_raw = struct.unpack_from("<II", raw, 0x14)
            layout.runway_centers.append((decode_lonlat_dword(lat_raw, is_lat=True),
                                           decode_lonlat_dword(lon_raw, is_lat=False)))

        elif rec_type == _REC_APRON and len(raw) >= 0x34:
            nv = struct.unpack_from("<H", raw, 0x30)[0]
            verts = _read_vertex_array(raw, 0x34, nv)
            if len(verts) >= 3:
                layout.aprons.append(Polygon(verts))

        elif rec_type == _REC_PAINTED_LINE and len(raw) >= 0x1C:
            nv = struct.unpack_from("<H", raw, 8)[0] - 1
            verts = _read_vertex_array(raw, 0x1C, nv)
            if len(verts) >= 2:
                layout.painted_lines.append(Polygon(verts))

        elif rec_type == _REC_LIGHT_STRING and len(raw) >= 24:
            nv = struct.unpack_from("<H", raw, 8)[0]
            light_type = struct.unpack_from("<H", raw, 10)[0]
            verts = _read_vertex_array(raw, 24, nv)
            if verts:
                layout.light_strings.append(LightString(verts, light_type))

        elif rec_type == _REC_START and len(raw) >= _TAXI_EDGE_HEADER:
            count = struct.unpack_from("<H", raw, 6)[0]
            for i in range(count):
                eoff = _TAXI_EDGE_HEADER + i * _START_STRIDE
                if eoff + _START_STRIDE > len(raw):
                    break
                lon_raw, lat_raw = struct.unpack_from("<II", raw, eoff + _START_LONLAT_OFFSET)
                layout.ramp_starts.append((decode_lonlat_dword(lat_raw, is_lat=True),
                                            decode_lonlat_dword(lon_raw, is_lat=False)))

        elif rec_type == _REC_TAXI_NODES and len(raw) >= _TAXI_NODE_HEADER:
            header_count = struct.unpack_from("<H", raw, 6)[0]
            navail = min(header_count, (len(raw) - _TAXI_NODE_HEADER) // _TAXI_NODE_STRIDE)
            o = _TAXI_NODE_HEADER
            for _ in range(navail):
                lon_raw, lat_raw, _extra = struct.unpack_from("<IIi", raw, o)
                layout.taxi_nodes.append((decode_lonlat_dword(lat_raw, is_lat=True),
                                           decode_lonlat_dword(lon_raw, is_lat=False)))
                o += _TAXI_NODE_STRIDE

        elif rec_type == _REC_TAXI_EDGES and len(raw) >= _TAXI_EDGE_HEADER:
            count = struct.unpack_from("<H", raw, 6)[0]
            starts = []
            for i in range(count):
                eoff = _TAXI_EDGE_HEADER + i * _TAXI_EDGE_STRIDE
                if eoff + 2 > len(raw):
                    break
                starts.append(struct.unpack_from("<H", raw, eoff)[0])
            n_nodes = len(layout.taxi_nodes)
            for i in range(len(starts) - 1):
                a, b = starts[i], starts[i + 1]
                if a == b or a >= n_nodes or b >= n_nodes:
                    continue
                if _dist_m(layout.taxi_nodes[a], layout.taxi_nodes[b]) <= _MAX_TAXI_EDGE_CHAIN_M:
                    layout.taxi_edges.append((a, b))

    return layout
