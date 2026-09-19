"""
dsf_compiler.py was ported byte-for-byte (confirmed zero divergence from
the original non-GPU version via diff before porting) -- this pins its
output against a golden baseline so any FUTURE edit that changes behavior
is caught, rather than re-deriving correctness from the DSF spec each time.
"""
import hashlib
import math
import struct
import tempfile
import unittest
from pathlib import Path

import dsf_compiler


def _iter_atoms(data):
    """Walks one level of DSF atoms (4-byte magic, 4-byte little-endian
    total length INCLUDING this 8-byte header, then payload) -- the same
    format at every nesting level (top-level file atoms, and the sub-atoms
    packed inside NFED/DOEG's own payloads), so this one helper works for
    both in the polygon-structure test below."""
    i = 0
    while i < len(data):
        magic = data[i:i + 4]
        length = struct.unpack('<I', data[i + 4:i + 8])[0]
        yield magic, data[i + 8:i + length]
        i += length


class TestDsfCompilerGolden(unittest.TestCase):
    def test_simple_tile_matches_golden_hash(self):
        """A small, fixed tile (2 draped objects, 1 AGL object, 1 exclusion
        rectangle) compiled today must always produce these exact bytes.
        If this test ever fails after an intentional change, regenerate the
        golden hash below and note why in the commit -- don't just update it
        to make the test pass without understanding what changed."""
        objects = [
            {"name": "test_obj_a", "lat": 47.1234, "lon": 8.5678, "hdg": 90.0, "agl": 0.0},
            {"name": "test_obj_b", "lat": 47.1250, "lon": 8.5690, "hdg": 180.0, "agl": 0.0},
            {"name": "test_obj_c", "lat": 47.1240, "lon": 8.5680, "hdg": 0.0, "agl": 3.5},
        ]
        exclusions = [{"west": 8.5, "south": 47.1, "east": 8.6, "north": 47.2}]

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "test_tile.dsf"
            dsf_compiler.build_dsf(47, 8, objects, out_path, exclusions=exclusions)
            raw = out_path.read_bytes()

        self.assertEqual(raw[:8], b"XPLNEDSF")
        digest = hashlib.sha256(raw).hexdigest()
        # Golden hash. Regenerated 2026-09-02: exclusion PROP values are now
        # SLASH-delimited ("west/south/east/north"), not space-delimited --
        # X-Plane splits the value on "/" and silently drops a space-delimited
        # rectangle whole, so none of the sim/exclude_* zones took effect
        # before this. See dsf_compiler._exclusion_props / EXCLUSION_PROP_KEYS.
        # (Prior regen 2026-08-30: localised float32-exact SCAL, ~1 mm coord
        # quantum -- test_pool_scal_gives_sub_decimetre_placement_precision.)
        golden = "e5e034611d2ef2969e8bdc12fc3513cf203f8aef98794c6cb8ba46bb45980f11"
        self.assertEqual(digest, golden, "dsf_compiler.py output changed -- verify intentionally before updating golden")

    def _decode_plane(self, buf, off, n):
        """Decode one encType-3 (differenced + RLE, u16) DSF pool plane;
        returns (values, new_offset)."""
        assert buf[off] == 3, buf[off]
        off += 1
        diffs = []
        while len(diffs) < n:
            c = buf[off]
            off += 1
            if c < 128:
                for _ in range(c):
                    diffs.append(struct.unpack('<H', buf[off:off + 2])[0])
                    off += 2
            else:
                v = struct.unpack('<H', buf[off:off + 2])[0]
                off += 2
                diffs.extend([v] * (c - 128))
        vals = [diffs[0]]
        for k in range(1, len(diffs)):
            vals.append((vals[-1] + diffs[k]) % 65536)
        return vals[:n], off

    def test_pool_scal_gives_sub_decimetre_placement_precision(self):
        """The whole-tile SCAL used to quantise every placement to ~1.7 m
        N/S. The localised, float32-exact SCAL must round-trip an
        airport-cluster of objects to well under 0.1 m, keep every decoded
        coordinate inside the tile, and use only exactly-float32
        representable SCAL offset/span values (1/256-deg grid)."""
        objs = [
            {"name": "o", "lat": 47.4308 + i * 0.0009, "lon": 19.2617 + i * 0.0007,
             "hdg": (i * 37) % 360, "agl": 0.0}
            for i in range(25)
        ]
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "prec.dsf"
            dsf_compiler.build_dsf(47, 19, objs, out_path)
            raw = out_path.read_bytes()

        body = raw[12:-16]
        atoms = dict(_iter_atoms(body))
        geod = list(_iter_atoms(atoms[b'DOEG']))
        loop = next(p for m, p in geod if m == b'LOOP')
        lacs = next(p for m, p in geod if m == b'LACS')
        n, planes = struct.unpack('<IB', loop[:5])
        self.assertEqual((n, planes), (25, 3))
        scal = struct.unpack('<6f', lacs[:24])
        (lon_span, lon_off, lat_span, lat_off, hdg_span, hdg_off) = scal

        # SCAL offset/span must be exact on the 1/256-deg grid.
        for v in (lon_span, lon_off, lat_span, lat_off):
            self.assertEqual(v * 256.0, round(v * 256.0), f"{v} is not on the 1/256-deg grid")
        self.assertEqual((hdg_span, hdg_off), (360.0, 0.0))

        off = 5
        lon_raw, off = self._decode_plane(loop, off, n)
        lat_raw, off = self._decode_plane(loop, off, n)
        worst = 0.0
        for i, o in enumerate(objs):
            lon = lon_off + lon_raw[i] / 65535.0 * lon_span
            lat = lat_off + lat_raw[i] / 65535.0 * lat_span
            self.assertTrue(47.0 <= lat < 48.0 and 19.0 <= lon < 20.0, "decoded coord left the tile")
            worst = max(worst, abs(lat - o["lat"]) * 111320.0,
                        abs(lon - o["lon"]) * 111320.0 * 0.68)
        self.assertLess(worst, 0.10, f"placement round-trip error {worst*100:.1f} cm exceeds 10 cm")

    def test_empty_objects_still_produces_valid_dsf(self):
        """An exclusion-only tile (no placed objects) must still compile --
        this is the real case main.py relies on for exclusion rectangles
        touching a tile with nothing else placed in it."""
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "empty_tile.dsf"
            dsf_compiler.build_dsf(46, 8, [], out_path, exclusions=[
                {"west": 8.0, "south": 46.0, "east": 9.0, "north": 47.0,
                 "categories": tuple(dsf_compiler.EXCLUSION_PROP_KEYS.keys())},
            ])
            raw = out_path.read_bytes()
        self.assertEqual(raw[:8], b"XPLNEDSF")
        for key in dsf_compiler.EXCLUSION_PROP_KEYS.values():
            self.assertIn(key.encode("ascii"), raw)
        # The rectangle value must be SLASH-delimited -- X-Plane silently
        # drops a space-delimited exclusion whole (verified vs Aerosoft's
        # shipping sceneries). Guard against a regression to spaces.
        self.assertIn(b"8.000000/46.000000/9.000000/47.000000", raw)
        self.assertNotIn(b"8.000000 46.000000 9.000000 47.000000", raw)

    def test_library_substitution_object_uses_library_path(self):
        objects = [{"name": None, "library_path": "lib/airport/Common_Elements/foo.obj",
                    "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0}]
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "lib_tile.dsf"
            dsf_compiler.build_dsf(47, 8, objects, out_path)
            raw = out_path.read_bytes()
        self.assertIn(b"lib/airport/Common_Elements/foo.obj", raw)

    def test_polygons_none_and_empty_list_match_omitted_param(self):
        """build_dsf grew a `polygons` parameter for the new .pol/DSF-
        polygon conversion path -- every call site that predates it (and
        every call this run makes when the polygons= feature is off) must
        keep producing BYTE-IDENTICAL output, not just "equivalent"
        output. Pins that omitting the parameter, passing polygons=None,
        and passing polygons=[] are all exactly the same no-op."""
        objects = [{"name": "test_obj_a", "lat": 47.1234, "lon": 8.5678, "hdg": 90.0, "agl": 0.0}]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p_omitted = td / "omitted.dsf"
            p_none = td / "none.dsf"
            p_empty = td / "empty.dsf"
            dsf_compiler.build_dsf(47, 8, objects, p_omitted)
            dsf_compiler.build_dsf(47, 8, objects, p_none, polygons=None)
            dsf_compiler.build_dsf(47, 8, objects, p_empty, polygons=[])
            self.assertEqual(p_omitted.read_bytes(), p_none.read_bytes())
            self.assertEqual(p_omitted.read_bytes(), p_empty.read_bytes())

    def test_polygon_tile_structure(self):
        """A tile with both an ordinary OBJECT placement AND draped-polygon
        triangles must produce: a POLY (YLOP) definition-table entry for
        the .pol path, a SECOND point pool (4-plane: lon/lat/s/t) inside
        GEOD alongside the ordinary 3-plane OBJECT pool, and PolygonRange
        (opcode 13) commands in CMDS -- one per triangle -- each carrying
        the explicit-per-vertex-UV sentinel param (65535), per pol_writer.py's
        own always-explicit-UV design and the real DSFTool-verified
        encoding this was built against."""
        objects = [{"name": "pad", "lat": 47.1, "lon": 8.1, "hdg": 0.0, "agl": 0.0}]
        polygons = [
            {"pol_path": "polygons/markA.pol", "points": [
                (8.10001, 47.10001, 0.0, 0.0),
                (8.10002, 47.10001, 1.0, 0.0),
                (8.10002, 47.10002, 1.0, 1.0),
            ]},
            {"pol_path": "polygons/markA.pol", "points": [
                (8.10001, 47.10001, 0.0, 0.0),
                (8.10002, 47.10002, 1.0, 1.0),
                (8.10001, 47.10002, 0.0, 1.0),
            ]},
        ]
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "poly_tile.dsf"
            dsf_compiler.build_dsf(47, 8, objects, out_path, polygons=polygons)
            raw = out_path.read_bytes()

        self.assertEqual(raw[:8], b"XPLNEDSF")
        self.assertIn(b"polygons/markA.pol\x00", raw)

        body = raw[12:-16]  # strip the 12-byte file header and 16-byte trailing MD5
        atoms = dict(_iter_atoms(body))
        self.assertIn(b'NFED', atoms)
        defn_atoms = dict(_iter_atoms(atoms[b'NFED']))
        self.assertEqual(defn_atoms[b'YLOP'], b"polygons/markA.pol\x00")

        self.assertIn(b'DOEG', atoms)
        # list, not dict -- GEOD legitimately carries TWO LOOP/LACS pairs
        # here (the ordinary OBJECT pool plus the new polygon-vertex pool),
        # and a dict would silently keep only the last one.
        geod_atoms = list(_iter_atoms(atoms[b'DOEG']))
        loop_payloads = [p for m, p in geod_atoms if m == b'LOOP']
        self.assertEqual(len(loop_payloads), 2, "expected the object pool plus a separate polygon-vertex pool")
        n, num_planes = struct.unpack('<IB', loop_payloads[1][:5])
        self.assertEqual(n, 6, "2 triangles x 3 vertices each, no AGL pool in between to shift the count")
        self.assertEqual(num_planes, 4, "lon/lat/s/t")

        self.assertIn(b'SDMC', atoms)
        cmds = atoms[b'SDMC']
        poly_range_count = 0
        i = 0
        while i < len(cmds):
            op = cmds[i]
            if op == 13:
                param, start, end = struct.unpack('<HHH', cmds[i + 1:i + 7])
                self.assertEqual(param, 65535, "polygon ranges must use the explicit-per-vertex-UV sentinel")
                self.assertEqual(end - start, 3, "one triangle == 3 contiguous pool points per range")
                poly_range_count += 1
                i += 7
            elif op in (1, 4, 7):
                i += 3
            else:
                self.fail(f"unexpected CMDS opcode {op} at byte {i}")
        self.assertEqual(poly_range_count, 2, "one PolygonRange command per triangle")

    def test_polygon_pool_splits_across_multiple_pools_past_u16_limit(self):
        """Confirmed real crash: DSF PolygonRange's start/end pool-index
        fields are u16 (max 65535, see dsf_compiler._MAX_POLYGON_POOL_POINTS's
        own comment), but a real airport's draped triangle soup easily
        produces far more points than that in one tile (LHBP's own base
        tile alone: 250k+ triangles, 750k+ points) -- struct.pack('<H', ...)
        raised "requires 0 <= number <= 65535" the first time this ran
        against real converted data. build_dsf must split into multiple
        pools instead of ever emitting an out-of-range index, keeping every
        individual polygon's own points (one triangle) whole within a
        single pool."""
        n_triangles = 25000  # 75000 points, > 65535 -- must span 2+ pools
        polygons = []
        for i in range(n_triangles):
            lon = 8.10000 + i * 0.0000001
            polygons.append({
                "pol_path": "polygons/big.pol",
                "points": [
                    (lon, 47.10000, 0.0, 0.0),
                    (lon + 0.00000001, 47.10000, 1.0, 0.0),
                    (lon, 47.10001, 0.0, 1.0),
                ],
            })

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "huge_poly_tile.dsf"
            dsf_compiler.build_dsf(47, 8, [], out_path, polygons=polygons)  # must not raise
            raw = out_path.read_bytes()

        body = raw[12:-16]
        atoms = dict(_iter_atoms(body))
        geod_atoms = list(_iter_atoms(atoms[b'DOEG']))
        loop_payloads = [p for m, p in geod_atoms if m == b'LOOP']
        # 1 for the (empty) object pool always present, plus however many
        # polygon-vertex pools the 75000 points needed.
        self.assertGreater(len(loop_payloads), 2, "75000 points must not fit in a single u16-indexed pool")

        total_pool_points = 0
        for payload in loop_payloads[1:]:
            n, num_planes = struct.unpack('<IB', payload[:5])
            self.assertLessEqual(n, dsf_compiler._MAX_POLYGON_POOL_POINTS)
            total_pool_points += n
        self.assertEqual(total_pool_points, n_triangles * 3)

        cmds = atoms[b'SDMC']
        poly_range_count = 0
        i = 0
        while i < len(cmds):
            op = cmds[i]
            if op == 13:
                param, start, end = struct.unpack('<HHH', cmds[i + 1:i + 7])
                self.assertLessEqual(end, 65535)
                poly_range_count += 1
                i += 7
            elif op in (1, 4, 7):
                i += 3
            else:
                self.fail(f"unexpected CMDS opcode {op} at byte {i}")
        self.assertEqual(poly_range_count, n_triangles)


if __name__ == "__main__":
    unittest.main()
