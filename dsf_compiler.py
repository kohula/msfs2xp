"""
DSF Compiler Module (v2)
Encodes 16-bit vertex pools using differenced + run-length encoding (encType 3),
orders POOL before SCAL inside GEOD, and compiles standard X-Plane 12 DSF binaries.
"""

import sys
import math
import struct
import hashlib
from pathlib import Path

# --- DSF ATOM PACKING HELPERS ---
def pack_atom(magic_bytes, payload):
    atom_length = len(payload) + 8
    return magic_bytes + struct.pack('<I', atom_length) + payload


def create_string_table(items):
    if not items:
        return b""
    return b'\0'.join([item.encode('utf-8') for item in items]) + b'\0'


# --- POOL ENCODING: differenced + run-length ---
def _encode_run_length(values):
    values = list(values)
    count = 1
    prev = values[0]
    individuals = []
    for value in values[1:] + ['_END_']:
        if value != prev or count == 127:
            if len(individuals) == 127:
                yield (len(individuals), individuals)
                individuals = []
            if count > 1:
                if individuals:
                    yield (len(individuals), individuals)
                yield (count + 128, [prev])
                individuals = []
            else:
                individuals.append(prev)
            count = 0
        prev = value
        count += 1
    if individuals:
        yield (len(individuals), individuals)


def _encode_plane_differenced(raw_values):
    payload = bytearray(struct.pack('<B', 3))  # encType 3
    if not raw_values:
        return bytes(payload)

    diffs = [raw_values[0]]
    for i in range(1, len(raw_values)):
        diffs.append((raw_values[i] - raw_values[i - 1]) % 65536)

    for count, vals in _encode_run_length(diffs):
        payload += struct.pack('<B', count)
        for v in vals:
            payload += struct.pack('<H', v)
    return bytes(payload)


# --- THE GEODATA & POOL STRUCTURES ---
# X-Plane's DSF OBJECT command has no vertical field of its own, but the
# point POOL it draws from can carry a 4th plane (elevation) instead of the
# usual 3 (lon/lat/heading) -- X-Plane's own DSF reader treats any object
# drawn from a 4-plane pool as explicit-height rather than terrain-draped,
# defaulting to MSL (mean sea level) unless a CMDS "comment" switches it to
# AGL (above ground level) first. Confirmed directly against X-Plane's own
# SDK source (xptools/src/DSF/DSFLib.h and DSFLib.cpp):
#   enum obj_elev_mode { obj_ModeDraped=0, obj_ModeMSL=1, obj_ModeAGL=2 };
#   planeDepths[currentPool]==4 ? curObjMode : obj_ModeDraped   (reader)
#   dsf_Cmd_Comment8 = 32, dsf_Comment_AGL = 2                  (DSFDefs.h)
# This is the ONLY place X-Plane's scenery-object placement can carry a
# vertical offset -- there is no other field, flag, or atom for it -- so an
# object that genuinely needs to sit above/below its ground contact point
# (e.g. a person standing on an upper floor, matching a source BGL/SPB
# placement's AGL altitude) has to be placed from this second, 4-plane pool
# instead of the ordinary terrain-draped one.
_AGL_HEIGHT_SCALE = 800.0
_AGL_HEIGHT_OFFSET = -400.0  # raw 16-bit range covers -400m..+400m (~1.2cm/step)


def _encode_plane_raw(values, minimum, maximum):
    span = maximum - minimum
    raw = [max(0, min(65535, int(round((v - minimum) / span * 65535)))) for v in values]
    return _encode_plane_differenced(raw)


# A pool whose SCAL multiplier spans the whole 1-degree tile resolves each
# encoded coordinate to only 1/65535 deg after the 16-bit quantisation --
# about 1.7 m north/south and 1.1 m east/west at mid latitudes. That is
# plainly visible as every placed object, and every draped ground marking,
# sitting a metre or two off its real position (and adjacent merged layers
# shifting relative to each other, since each rounds independently).
# Tightening every pool's SCAL to the lon/lat SPAN of the points it actually
# holds spends all 16 bits across just the airport's own footprint
# (~0.02-0.05 deg even for a big hub), bringing the quantum down to a few
# centimetres. Decode is unchanged -- real = offset + raw/65535 * span --
# and inverts the encode.
#
# The SCAL (offset, span) pair is stored as float32, which only has ~7
# significant digits: a raw offset like 47.1234 rounds to ~47.12340164, a
# fixed ~0.18 m error on top of the quantisation. So both are SNAPPED to a
# 1/_SCAL_SNAP-degree grid (offset floored, span ceilinged) -- 1/256 is an
# exact power-of-two fraction that float32 represents with zero error, and
# the grid is fine enough (~1/256 deg ~ 60-90 m) that a snapped span still
# gives a sub-decimetre quantum over any real airport.
_SCAL_SNAP = 256.0
_MIN_SCAL_SPAN_DEG = 1.0 / _SCAL_SNAP  # floor so an all-coincident pool never divides by zero


def _snapped_scal(values, tile_lo, fallback_off, fallback_span):
    """(offset, span) for a localised SCAL plane, both snapped to the
    1/_SCAL_SNAP-degree grid so they are exact in float32: offset floored
    to that grid (<= min value), span ceilinged (>= the value range), and
    clamped so offset+span never leaves this 1-degree tile. `values` empty
    (an empty pool) -> the fallback, which keeps byte output identical for
    the exclusion-only-tile case."""
    if not values:
        return fallback_off, fallback_span
    lo, hi = min(values), max(values)
    off = math.floor(lo * _SCAL_SNAP) / _SCAL_SNAP
    span = math.ceil((hi - off) * _SCAL_SNAP) / _SCAL_SNAP
    span = max(span, _MIN_SCAL_SPAN_DEG)
    tile_hi = math.floor(tile_lo) + 1
    if off + span > tile_hi:
        span = tile_hi - off
    return off, span


def _build_pool_and_scal(tile_lat, tile_lon, objects, include_elevation):
    n = len(objects)
    num_planes = 4 if include_elevation else 3

    lon_off, lon_scale = _snapped_scal([o['lon'] for o in objects], tile_lon, tile_lon, 1.0)
    lat_off, lat_scale = _snapped_scal([o['lat'] for o in objects], tile_lat, tile_lat, 1.0)

    lon_raw = [max(0, min(65535, int(round((o['lon'] - lon_off) / lon_scale * 65535)))) for o in objects]
    lat_raw = [max(0, min(65535, int(round((o['lat'] - lat_off) / lat_scale * 65535)))) for o in objects]
    hdg_raw = [max(0, min(65535, int(round((o['hdg'] % 360.0) * 65535 / 360.0)))) for o in objects]

    pool_payload = bytearray(struct.pack('<IB', n, num_planes))
    pool_payload += _encode_plane_differenced(lon_raw)
    pool_payload += _encode_plane_differenced(lat_raw)
    pool_payload += _encode_plane_differenced(hdg_raw)

    scal_payload = struct.pack('<ff', lon_scale, float(lon_off))
    scal_payload += struct.pack('<ff', lat_scale, float(lat_off))
    scal_payload += struct.pack('<ff', 360.0, 0.0)

    if include_elevation:
        pool_payload += _encode_plane_raw(
            [o['agl'] for o in objects], _AGL_HEIGHT_OFFSET, _AGL_HEIGHT_OFFSET + _AGL_HEIGHT_SCALE
        )
        scal_payload += struct.pack('<ff', _AGL_HEIGHT_SCALE, _AGL_HEIGHT_OFFSET)

    atom_pool = pack_atom(b'LOOP', bytes(pool_payload))
    atom_scal = pack_atom(b'LACS', scal_payload)
    return atom_pool + atom_scal


# DSF pool point indices are u16 everywhere they're referenced from CMDS
# (PolygonRange's own start/end fields included -- see build_cmds_atom),
# so no single pool may hold more than this many points. build_dsf's own
# chunking pass enforces this by splitting the polygon vertex stream into
# multiple pools rather than ever emitting an index that doesn't fit; a
# real airport's draped triangle soup (e.g. LHBP's own base tile: 250k+
# triangles, 750k+ points) routinely needs several.
_MAX_POLYGON_POOL_POINTS = 65535


def _build_polygon_pool_and_scal(tile_lat, tile_lon, points):
    """points: a flat list of (lon, lat, s, t) tuples -- one polygon-POOL
    VERTEX each, not one per placement (a single merged draped-content
    triangle contributes 3 of these). s/t are always in [0,1] (real
    per-vertex UV straight from the source mesh -- see pol_writer.py's own
    docstring for why this project always uses DSF's explicit-UV polygon
    mode), so their own SCAL entry is a trivial identity (scale=1,
    offset=0), same as the lon/lat planes' convention already used by
    _build_pool_and_scal above.

    Byte-for-byte layout (POOL header, plane count, per-plane differenced+
    RLE encoding, SCAL as 4x(scale,offset) float pairs) empirically
    verified against Laminar's own DSFTool (--text2dsf then hex-inspected)
    compiling real per-vertex-UV polygon data -- not just this project's
    own prior working convention for the 3/4-plane OBJECT pools above,
    which _encode_plane_differenced is shared with unchanged."""
    n = len(points)
    # Same localised-SCAL tightening as the OBJECT pool above (see
    # _MIN_POOL_SPAN_DEG's comment): a whole-degree multiplier would quantise
    # every polygon vertex to ~1-1.7 m, visibly shifting draped markings.
    lon_off, lon_scale = _snapped_scal([p[0] for p in points], tile_lon, tile_lon, 1.0)
    lat_off, lat_scale = _snapped_scal([p[1] for p in points], tile_lat, tile_lat, 1.0)

    lon_raw = [max(0, min(65535, int(round((p[0] - lon_off) / lon_scale * 65535)))) for p in points]
    lat_raw = [max(0, min(65535, int(round((p[1] - lat_off) / lat_scale * 65535)))) for p in points]
    s_raw = [max(0, min(65535, int(round(p[2] * 65535)))) for p in points]
    t_raw = [max(0, min(65535, int(round(p[3] * 65535)))) for p in points]

    pool_payload = bytearray(struct.pack('<IB', n, 4))
    pool_payload += _encode_plane_differenced(lon_raw)
    pool_payload += _encode_plane_differenced(lat_raw)
    pool_payload += _encode_plane_differenced(s_raw)
    pool_payload += _encode_plane_differenced(t_raw)

    scal_payload = struct.pack('<ff', lon_scale, float(lon_off))
    scal_payload += struct.pack('<ff', lat_scale, float(lat_off))
    scal_payload += struct.pack('<ff', 1.0, 0.0)
    scal_payload += struct.pack('<ff', 1.0, 0.0)

    atom_pool = pack_atom(b'LOOP', bytes(pool_payload))
    atom_scal = pack_atom(b'LACS', scal_payload)
    return atom_pool + atom_scal


def build_geod_atom(tile_lat, tile_lon, draped_objects, agl_objects, polygon_chunks=None):
    """Builds the single GEOD atom, containing one point pool (index 0, the
    ordinary 3-plane terrain-draped pool -- always present, even if empty,
    so pool indexing stays stable) plus a second 4-plane pool (index 1) for
    AGL-height objects, only when agl_objects is non-empty, plus one
    ADDITIONAL 4-plane pool per entry in polygon_chunks (see build_dsf's own
    chunking pass -- each chunk's points list is already capped at
    _MAX_POLYGON_POOL_POINTS, since DSF pool point indices are u16 and one
    real airport's draped triangle soup routinely needs far more than
    65535 vertices total). Returns (atom_bytes, polygon_pool_indices) --
    the caller (build_cmds_atom) needs to know which pool index each chunk
    ended up at, since it depends on whether the AGL pool was written this
    call (an empty-polygons tile must produce byte-identical GEOD output to
    before this function grew polygon support -- polygon_chunks=None/[]
    skips writing anything new, exactly reproducing the prior two-pool-max
    behavior)."""
    inner = _build_pool_and_scal(tile_lat, tile_lon, draped_objects, include_elevation=False)
    next_index = 1
    if agl_objects:
        inner += _build_pool_and_scal(tile_lat, tile_lon, agl_objects, include_elevation=True)
        next_index += 1
    polygon_pool_indices = []
    for chunk in (polygon_chunks or []):
        inner += _build_polygon_pool_and_scal(tile_lat, tile_lon, chunk["points"])
        polygon_pool_indices.append(next_index)
        next_index += 1
    return pack_atom(b'DOEG', inner), polygon_pool_indices


# --- OBJECT REFERENCES ---
def obj_ref(obj):
    """The string this object is listed under in the DSF's own OBJT table.
    Normally a path into this pack's own "objects/" folder, but an object
    dict can instead carry "library_path" -- a virtual path (e.g.
    "lib/airport/Common_Elements/Lighting/com_Flood_36m.obj") into
    X-Plane's own default library or an installed third-party one. X-Plane
    resolves any OBJT entry that isn't an actual file inside this pack
    against every installed library's own EXPORT declarations, so listing
    a known-real virtual path here, with no matching file of our own,
    is exactly how a custom scenery normally references a stock library
    object instead of shipping (and converting) its own copy."""
    lib_path = obj.get("library_path")
    return lib_path if lib_path else f"objects/{obj['name']}.obj"


# --- THE COMMAND STREAM ---
# dsf_Cmd_PolygonRange (13, 0x0D): u8 cmd, u16 param, u16 first_point_index,
# u16 last_point_index (exclusive) -- draws one polygon winding from a
# CONTIGUOUS run of points in the currently-selected pool. param=65535 is
# the documented (and, per _build_polygon_pool_and_scal's own docstring,
# now empirically DSFTool-verified) sentinel meaning "ignore this .pol's
# own SCALE, read explicit per-vertex S/T from the pool's 3rd/4th planes
# instead". Byte layout confirmed directly against DSFTool --text2dsf
# output (hex-inspected), not just the documented spec text -- including
# confirming pool-select (cmd 1) and set-definition (cmd 4, reusing the
# exact same opcode objects already use) only need to be re-emitted when
# they actually change, not once per polygon.
_CMD_POLYGON_RANGE = 13
_POLYGON_EXPLICIT_UV_PARAM = 65535


def build_cmds_atom(draped_objects, agl_objects, unique_names, polygon_chunks=None, polygon_pool_indices=None):
    """polygon_chunks: optional list of {"points": [...], "ranges": [(def_idx,
    start_idx, end_idx), ...]} dicts, one per polygon POOL (see build_dsf's
    own chunking pass) -- def_idx indexes into the POLY definition table
    (unique .pol paths) the same way object placements index into
    unique_names; start/end index LOCALLY into that chunk's own pool, built
    by build_geod_atom (see its own polygon_pool_indices return value,
    passed straight through here, one pool index per chunk in the same
    order). Chunks are visited strictly in order (never revisited), so each
    only ever needs ONE "select pool" command -- matches how the sorted
    (largest-footprint-first) `polygons` list build_dsf assembles chunks
    from is itself only ever walked forward once."""
    cmds = bytearray()
    current_def = -1
    # unique_names.index(...) per placement is O(defs) -- with a big tile's
    # object count times its distinct-def count that made this the actual
    # "stuck at DSF compilation" cost (no progress output during it either,
    # so a slow tile just looked frozen). One dict, built once, makes each
    # lookup O(1); output order/values are unchanged.
    name_to_idx = {name: i for i, name in enumerate(unique_names)}

    def emit(objects):
        nonlocal current_def
        for i, obj in enumerate(objects):
            def_idx = name_to_idx[obj_ref(obj)]
            if def_idx != current_def:
                cmds.extend(struct.pack('<BH', 4, def_idx))
                current_def = def_idx
            cmds.extend(struct.pack('<BH', 7, i))

    cmds += struct.pack('<BH', 1, 0)  # select pool 0 (draped)
    emit(draped_objects)

    if agl_objects:
        # dsf_Cmd_Comment8(32), commentLen=6, dsf_Comment_AGL(2), want_agl=1
        # -- switches every subsequent object placed from a 4-plane pool
        # into AGL mode (instead of the default MSL) until changed again.
        cmds += struct.pack('<BBHi', 32, 6, 2, 1)
        cmds += struct.pack('<BH', 1, 1)  # select pool 1 (agl)
        emit(agl_objects)

    if polygon_chunks:
        current_poly_def = -1
        for chunk, pool_idx in zip(polygon_chunks, polygon_pool_indices):
            if not chunk["ranges"]:
                continue
            cmds += struct.pack('<BH', 1, pool_idx)  # select this chunk's polygon pool
            for def_idx, start_idx, end_idx in chunk["ranges"]:
                if def_idx != current_poly_def:
                    cmds.extend(struct.pack('<BH', 4, def_idx))
                    current_poly_def = def_idx
                cmds.extend(struct.pack('<BHHH', _CMD_POLYGON_RANGE, _POLYGON_EXPLICIT_UV_PARAM, start_idx, end_idx))

    return pack_atom(b'SDMC', bytes(cmds))


# --- EXCLUSION ZONES ---
# X-Plane suppresses its own default/autogen scenery inside a DSF tile purely
# through PROP key/value pairs in the DSF header -- nothing to do with the
# source .bgl. Each of these keys takes one "west/south/east/north" rectangle
# -- the four numbers are SLASH-delimited, not space-delimited (verified
# against Aerosoft's shipping X-Plane sceneries, e.g. EGLL:
# `sim/exclude_net = -0.4231/51.4804/-0.4222/51.4807`). X-Plane splits the
# value on "/"; a space-delimited value is one unparseable token and the
# WHOLE rectangle is silently dropped -- which is why none of these took
# effect before 2026-09-02. X-Plane accepts repeated occurrences of the same
# key to carve out multiple independent rectangles within one tile.
EXCLUSION_PROP_KEYS = {
    "obj":  "sim/exclude_obj",   # individually placed 3-D objects
    "agb":  "sim/exclude_agb",   # autogen BUILDINGS (generated along roads/
                                  # blocks -- a totally separate system from
                                  # "obj". Deliberately keeping "net" (roads)
                                  # unexcluded around the airport boundary
                                  # means the default autogen system still
                                  # has a road network to generate buildings
                                  # along there; without this key, those
                                  # buildings keep spawning inside the
                                  # airport footprint alongside our own.
    "for":  "sim/exclude_for",   # forests
    "fac":  "sim/exclude_fac",   # facades
    "bea":  "sim/exclude_bea",   # beacons
    "lin":  "sim/exclude_lin",   # draped lines
    "pol":  "sim/exclude_pol",   # draped polygons
    "net":  "sim/exclude_net",   # roads/rail vector networks
    "str":  "sim/exclude_str",   # autogen "strings" (object rows)
}


def _exclusion_props(exclusions, categories):
    """Build the flat PROP key/value list for a set of exclusion rectangles.

    `exclusions` is a list of dicts with west/south/east/north in degrees
    (already reordered to X-Plane's own corner convention -- callers are
    responsible for that, since the source BGL format uses NW/SE corners
    while X-Plane's PROP values are west/south/east/north).
    `categories` is which sim/exclude_* keys to emit per rectangle by
    default; defaults to "obj" only (see module docstring below for why).
    An individual rect dict may carry its own "categories" key to override
    this default just for that one rectangle -- used for the airport-wide
    boundary exclusion, which needs a much broader category set (everything
    except roads) than the precise per-feature rectangles decoded from the
    source BGL's own Exclusion section.
    """
    props = []
    for rect in exclusions:
        value = f"{rect['west']:.6f}/{rect['south']:.6f}/{rect['east']:.6f}/{rect['north']:.6f}"
        for cat in rect.get("categories", categories):
            key = EXCLUSION_PROP_KEYS.get(cat)
            if key:
                props += [key, value]
    return props


# --- MAIN COMPILER FUNCTION ---
def build_dsf(tile_lat, tile_lon, objects, out_path: Path, exclusions=None, exclusion_categories=("obj",), polygons=None):
    """Compile one DSF tile.

    exclusions: optional list of {"west","south","east","north"} dicts (degrees,
        X-Plane corner convention) already filtered to rectangles that fall in
        this tile. Each becomes its own sim/exclude_* PROP entry -- a precise,
        per-rectangle translation of the source BGL's Exclusion (0x2E) section,
        not a single whole-tile blanket exclusion.
    exclusion_categories: which sim/exclude_* keys to emit per rectangle.
        Defaults to ("obj",) -- suppress default/autogen 3-D objects only,
        since the source BGL's per-record `flags` bitmask isn't decodable
        from the compiled binary format with confidence (see
        bgl_extractor.parse_exclusion_rectangles for why). Pass e.g.
        ("obj", "for") if you also want autogen forest suppressed under the
        same rectangles.
    polygons: optional list of {"pol_path": str, "points": [(lon,lat,s,t) x3]}
        dicts, ONE PER TRIANGLE -- "pol_path" is the already-written .pol
        file's own relative path (pol_writer.write_pol_for_texture's return
        value), used as a POLYGON_DEF entry the exact same way obj_ref(obj)
        is used as an OBJT entry above. Left empty/None, this function's
        output is byte-identical to before this parameter existed (no POLY
        definitions, no polygon pool, no polygon commands) -- draped ground
        content converted via the legacy OBJ8 path never populates this.
    """
    props = [
        "sim/west", str(tile_lon),
        "sim/east", str(tile_lon + 1),
        "sim/south", str(tile_lat),
        "sim/north", str(tile_lat + 1),
        "sim/planet", "earth",
        "sim/creation_agent", "Python_Native_Compiler",
        "sim/overlay", "1"
    ]
    if exclusions:
        props += _exclusion_props(exclusions, exclusion_categories)
    atom_prop = pack_atom(b'PORP', create_string_table(props))
    atom_head = pack_atom(b'DAEH', atom_prop)

    # `not in unique_obj_names` on a growing list is O(n) per placement,
    # O(n*defs) total -- with this build's much larger object counts (MSFS
    # shared library + propdefs decoding both now feeding placements) that
    # was the real cost behind "stuck at DSF compilation" with no progress
    # output during it. A side-set makes the membership check O(1); output
    # order/values (first-seen order) are unchanged.
    unique_obj_names = []
    _seen_obj_names = set()
    for obj in objects:
        part_name = obj_ref(obj)
        if part_name not in _seen_obj_names:
            _seen_obj_names.add(part_name)
            unique_obj_names.append(part_name)

    # Objects that need to sit above/below their DSF ground-contact point
    # (a real "agl" value, in meters) go through the second, 4-plane AGL
    # pool built below; everything else keeps using the ordinary
    # terrain-draped pool exactly as before.
    draped_objects = [o for o in objects if abs(o.get("agl", 0.0)) < 0.01]
    agl_objects = [o for o in objects if abs(o.get("agl", 0.0)) >= 0.01]

    polygons = polygons or []
    unique_pol_paths = []
    _seen_pol_paths = set()
    for poly in polygons:
        if poly["pol_path"] not in _seen_pol_paths:
            _seen_pol_paths.add(poly["pol_path"])
            unique_pol_paths.append(poly["pol_path"])
    # Same O(n)-membership / O(n)-index traps as unique_obj_names above --
    # matters here too since a tile's polygon soup can run into the
    # hundreds of thousands (see the docstring below).
    pol_to_idx = {path: i for i, path in enumerate(unique_pol_paths)}

    # DSF pool point indices are u16, so one pool holds at most
    # _MAX_POLYGON_POOL_POINTS points -- a large airport's draped triangle
    # soup can need far more. Split into multiple pools, never letting one
    # polygon's points (a triangle) straddle a chunk boundary.
    polygon_chunks = []
    current_points, current_ranges = [], []
    for poly in polygons:
        if current_points and len(current_points) + len(poly["points"]) > _MAX_POLYGON_POOL_POINTS:
            polygon_chunks.append({"points": current_points, "ranges": current_ranges})
            current_points, current_ranges = [], []
        start = len(current_points)
        current_points.extend(poly["points"])
        end = len(current_points)
        current_ranges.append((pol_to_idx[poly["pol_path"]], start, end))
    if current_points:
        polygon_chunks.append({"points": current_points, "ranges": current_ranges})

    atom_tert = pack_atom(b'TRET', b'')
    atom_objt = pack_atom(b'TJBO', create_string_table(unique_obj_names))
    atom_poly = pack_atom(b'YLOP', create_string_table(unique_pol_paths))
    atom_netw = pack_atom(b'WTEN', b'')
    atom_demn = pack_atom(b'NMED', b'')

    atom_defn = pack_atom(b'NFED', atom_tert + atom_objt + atom_poly + atom_netw + atom_demn)
    atom_geod, polygon_pool_indices = build_geod_atom(
        tile_lat, tile_lon, draped_objects, agl_objects, polygon_chunks)
    atom_cmds = build_cmds_atom(
        draped_objects, agl_objects, unique_obj_names, polygon_chunks, polygon_pool_indices)

    dsf_header = b'XPLNEDSF' + struct.pack('<I', 1)
    full_dsf_data = dsf_header + atom_head + atom_defn + atom_geod + atom_cmds
    md5_hash = hashlib.md5(full_dsf_data).digest()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(full_dsf_data)
        f.write(md5_hash)

    print(f"[SUCCESS] Compiled DSF: {out_path.name}")
