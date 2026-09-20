"""
Fully automated extractor and cleaner for MSFS 2020/2024 scenery .bgl files.
Extracts embedded 3D models (glTF/GLB), converts to OBJ, 
extracts placement coordinates (SceneryObjects) using dynamic offset calibration 
and per-record fallback for variable-length records, and generates a full 
GLB parameter/metadata report.
"""

import csv
import hashlib
import os
import re
import shutil
import struct
import sys
import json
import io
import math
import logging
import threading

import airport_layout
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cache_utils
import geo_transform
from geo_transform import decode_lonlat_dword  # noqa: F401 -- re-exported; moved there to avoid a circular import with airport_layout.py

try:
    import zstandard as zstd
except ImportError:
    zstd = None

logger = logging.getLogger(__name__)

# Process-lifetime cache for extract_spb_placements' loaded property-def
# bank, keyed by propdefs directory path -- see the comment at its use
# site for why (re-parsing ~200 XML files takes ~4s and was happening once
# per .spb file instead of once per run).
_PROPDEFS_CACHE = {}

SECTION_TYPES = {
    0x0: "None", 0x1: "Copyright", 0x2: "Guid", 0x3: "Airport",
    0x13: "IlsVor", 0x17: "Ndb", 0x18: "Marker", 0x20: "Boundary",
    0x22: "Waypoint", 0x23: "Geopol", 0x25: "SceneryObject",
    0x27: "NameList", 0x28: "VorIlsIcaoIndex", 0x29: "NdbIcaoIndex",
    0x2A: "WaypointIcaoIndex", 0x2B: "ModelData", 0x2C: "AirportSummary",
    0x2E: "Exclusion", 0x2F: "TimeZone", 0x65: "TerrainVectorDb",
}

GLB_MAGIC = b"glTF"
HEADER_LEN = {"SceneryObject": 4, "Airport": 6}

KNOWN_LAYOUTS = {
    0x000b: {
        "guid_off": 44, "pos_off": 4, "order": "lonlat",
        "heading_off": 22, "heading_type": "bams16", "confidence": "high",
        "rec_len": 0x40,
    },
    0x001b: {
        "guid_off": 48, "pos_off": 4, "order": "lonlat",
        "heading_off": 22, "heading_type": "bams16",
        "instance_id_off": 28, "confidence": "high",
        "rec_len": 0x40,
    },
    0x000a: {
        "guid_off": None, "no_guid": True, "pos_off": 4, "order": "lonlat",
        "heading_off": 22, "heading_type": "bams16", "confidence": "medium",
    },
    0x000c: {
        "guid_off": 28, "pos_off": 4, "order": "lonlat",
        "heading_off": 22, "heading_type": "bams16", "confidence": "medium",
        "rec_len": 0x2E,
    },
    0x000d: {
        "guid_off": None, "no_guid": True, "pos_off": 4, "order": "lonlat",
        "heading_off": 22, "heading_type": "bams16", "confidence": "medium",
        "name_off": 28, "name_kind": "stringz",
    },
    0x0010: {
        "guid_off": 28, "pos_off": 4, "order": "lonlat",
        "heading_off": 22, "heading_type": "bams16", "confidence": "medium",
    },
    0x0012: {
        "guid_off": 28, "pos_off": 4, "order": "lonlat",
        "heading_off": 22, "heading_type": "bams16", "confidence": "medium",
    },
}

TAXIWAY_SIGN_TYPE = 0x000E
ATTACHED_OBJECT_START_ID = 0x1002
ATTACHED_OBJECT_END_ID = 0x1001
ATTACHED_OBJECT_MARKER_SIZE = 0x0004
ATTACHED_OBJECT_INSTANCE_GUID_OFF = 0x18
ATTACHED_OBJECT_NAME_OFF = 0x36

SCENERY_OBJECT_TYPE_NAMES = {
    0x000a: "GenericBuilding",
    0x000b: "LibraryObject",
    0x000c: "Windsock",
    0x000d: "Effect",
    0x000e: "TaxiwaySign",
    0x0010: "Trigger",
    0x0012: "ExtrusionBridge",
    0x001b: "LibraryObject(FS9)",
    ATTACHED_OBJECT_START_ID: "AttachedObject-Start",
    ATTACHED_OBJECT_END_ID: "AttachedObject-End",
}

class Section:
    __slots__ = ("raw_type_val", "type_name", "subsec_count", "file_offset",
                 "total_size", "subsec_size", "subsections")

    def __init__(self, data: bytes, offset: int):
        (self.raw_type_val, size_val, self.subsec_count,
         self.file_offset, self.total_size) = struct.unpack_from("<IIIII", data, offset)
        self.subsec_size = ((size_val & 0x10000) | 0x40000) >> 0x0E
        self.type_name = SECTION_TYPES.get(self.raw_type_val, f"Unknown_{self.raw_type_val:#x}")
        self.subsections = []

class Subsection:
    __slots__ = ("qmid1", "qmid2", "data_offset", "data_size")

    def __init__(self, data: bytes, offset: int, size16: bool):
        if size16:
            self.qmid1, _n, self.data_offset, self.data_size = struct.unpack_from("<IIII", data, offset)
            self.qmid2 = 0
        else:
            self.qmid1, self.qmid2, _n, self.data_offset, self.data_size = struct.unpack_from("<IIIII", data, offset)

def parse_bgl(data: bytes):
    magic1, header_size, _lo, _hi, _magic2, num_sections = struct.unpack_from("<IIIIII", data, 0)
    if magic1 != 0x19920201:
        logger.warning(f"Unexpected file magic {magic1:#010x} -- may not be a valid BGL.")

    sections = []
    pos = header_size
    for _ in range(num_sections):
        sec = Section(data, pos)
        sections.append(sec)
        pos += 20

    for sec in sections:
        sub_size = sec.subsec_size
        for i in range(sec.subsec_count):
            off = sec.file_offset + i * sub_size
            sec.subsections.append(Subsection(data, off, sub_size == 16))

    return sections

def glob_ci(root: Path, ext: str):
    """Case-insensitive rglob by extension (ext including the leading dot,
    e.g. ".bgl"). Path.rglob("*.bgl") is case-SENSITIVE on Linux/macOS, and
    real packages do ship uppercase .BGL extensions (harmless on Windows'
    case-insensitive filesystem) -- rglob would silently skip those files
    entirely, no error, no warning."""
    ext = ext.lower()
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ext]


def placement_height_offset(alt: float, is_agl: bool, airport_alt: float) -> float:
    """Converts a placement's raw "alt" field into the same "how far to
    bake this object up/down from the DSF terrain contact point" value
    that main.py's bake_obj_height_offset() already applies for
    SPB-attached objects (see extract_spb_placements's "dy").

    X-Plane's DSF OBJECT point pool has no vertical/AGL field -- every
    placed object sits exactly on the compiled terrain mesh -- so any
    placement whose source alt says otherwise (a person standing on an
    upper floor/balcony, a rooftop antenna, a tower deck) needs that
    difference baked into the exported geometry, or it silently ends up
    sitting at ground level instead.

    When is_agl is True, alt already IS that offset (it's defined as
    meters above whatever surface is below the placement point). When
    False, alt is an absolute altitude, and airport_alt -- the airport's
    own reference elevation, the best terrain-height estimate available
    in this pipeline (no terrain raster data is available) -- is
    subtracted to approximate the same thing.
    """
    return alt if is_agl else (alt - airport_alt)

def walk_records(data: bytes, start: int, size: int, is_scenery_obj: bool = False):
    pos, end = start, start + size
    out = []
    fmt = "<HH" if is_scenery_obj else "<HI"
    hdr_len = 4 if is_scenery_obj else 6

    while pos + hdr_len <= end:
        rec_type, rec_size = struct.unpack_from(fmt, data, pos)
        if rec_size < hdr_len or pos + rec_size > end:
            break
        out.append((rec_type, data[pos:pos + rec_size]))
        pos += rec_size
    return out

_ASCII_PRINTABLE = frozenset(range(0x20, 0x7f))

def extract_attach_name(rec: bytes):
    if len(rec) < ATTACHED_OBJECT_NAME_OFF:
        return None, None

    instance_guid = rec[ATTACHED_OBJECT_INSTANCE_GUID_OFF:
                         ATTACHED_OBJECT_INSTANCE_GUID_OFF + 16].hex().lower()
    raw = rec[ATTACHED_OBJECT_NAME_OFF:]
    nul = raw.find(b"\x00")
    name_bytes = raw[:nul] if nul != -1 else raw
    if not name_bytes or not all(b in _ASCII_PRINTABLE for b in name_bytes):
        return None, instance_guid
    return name_bytes.decode("ascii", errors="replace"), instance_guid

def _decode_libobj_anchor(rec: bytes):
    """(lat, lon, hdg_deg) of a 0x000b/0x001b-style placement record, or
    None -- pos dwords at +4 (lon) / +8 (lat), heading BAMS16 at +22."""
    try:
        if len(rec) < 24:
            return None
        lon_val, lat_val = struct.unpack_from("<II", rec, 4)
        h = struct.unpack_from("<H", rec, 22)[0]
        return (decode_lonlat_dword(lat_val, True),
                decode_lonlat_dword(lon_val, False),
                h * (360.0 / 65536.0))
    except struct.error:
        return None


def walk_scenery_object_records(data: bytes, start: int, size: int):
    pos, end = start, start + size
    records = []
    skipped = []
    last_anchor = None   # (lat, lon, hdg) of the most recent placed record -- an
                         # AttachedObject's own position is a bias off THIS parent

    while pos + 4 <= end:
        rec_type, rec_size = struct.unpack_from("<HH", data, pos)
        if rec_size < 4 or pos + rec_size > end:
            break

        if rec_type == ATTACHED_OBJECT_START_ID and rec_size == ATTACHED_OBJECT_MARKER_SIZE:
            pos += rec_size
            if pos + 4 > end: break
            attach_type, attach_size = struct.unpack_from("<HH", data, pos)
            if attach_size < 4 or pos + attach_size > end: break
            attach_payload = data[pos:pos + attach_size]
            pos += attach_size

            name, instance_guid = extract_attach_name(attach_payload)
            skipped.append({
                "reason": "attached-object",
                "attach_record_type": attach_type,
                "instance_guid": instance_guid,
                "name": name,
                # parent position -- the attach's own offset off this is not
                # decoded, so this is "on/at the parent object", good enough
                # for a door/light hung on a building.
                "anchor_lat": last_anchor[0] if last_anchor else None,
                "anchor_lon": last_anchor[1] if last_anchor else None,
                "anchor_hdg": last_anchor[2] if last_anchor else 0.0,
            })

            if pos + 4 <= end:
                end_type, end_size = struct.unpack_from("<HH", data, pos)
                if end_type == ATTACHED_OBJECT_END_ID and end_size == ATTACHED_OBJECT_MARKER_SIZE:
                    pos += end_size
            continue

        if rec_type == TAXIWAY_SIGN_TYPE:
            payload = data[pos:pos + rec_size]
            lon = lat = None
            if len(payload) >= 12:
                lon_val, lat_val = struct.unpack_from("<II", payload, 4)
                lon, lat = decode_lonlat_dword(lon_val, False), decode_lonlat_dword(lat_val, True)
            num_signs = struct.unpack_from("<I", payload, 0x1C)[0] if len(payload) >= 0x20 else None
            skipped.append({
                "reason": "taxiway-sign-array",
                "anchor_lat": lat,
                "anchor_lon": lon,
                "num_signs": num_signs,
                "name": None,
            })
            pos += rec_size
            continue

        records.append((rec_type, data[pos:pos + rec_size]))
        if rec_type in KNOWN_LAYOUTS:
            a = _decode_libobj_anchor(data[pos:pos + rec_size])
            if a is not None:
                last_anchor = a
        pos += rec_size

    return records, skipped

def find_embedded_library_objects(blob: bytes):
    hits = []
    for rec_type, layout in KNOWN_LAYOUTS.items():
        rec_len = layout.get("rec_len")
        if rec_len is None: continue
        pat = struct.pack("<HH", rec_type, rec_len)
        idx = -1
        while True:
            idx = blob.find(pat, idx + 1)
            if idx == -1: break
            if idx + rec_len > len(blob): continue
            hits.append((rec_type, idx, blob[idx:idx + rec_len]))
    hits.sort(key=lambda h: h[1])
    return hits

def extract_airport_embedded_placements(data: bytes, airport_sections, guid_map: dict, _log, arp_lat=None, arp_lon=None, arp_alt=0.0):
    found_placements = []

    for sec in airport_sections:
        for sub in sec.subsections:
            blob = data[sub.data_offset: sub.data_offset + sub.data_size]
            hits = find_embedded_library_objects(blob)
            if not hits: continue
            for rec_type, blob_off, rec in hits:
                layout = KNOWN_LAYOUTS[rec_type]
                guid_off, pos_off = layout["guid_off"], layout["pos_off"]
                heading_off, heading_type = layout["heading_off"], layout["heading_type"]

                # Extract Altitude and AGL flags safely
                is_agl = True
                if len(rec) >= pos_off + 14:
                    a, b, alt_val, flags = struct.unpack_from("<IIiH", rec, pos_off)
                    lon, lat = decode_lonlat_dword(a, False), decode_lonlat_dword(b, True)
                    alt = alt_val / 1000.0
                    is_agl = bool(flags & 0x0001)
                elif len(rec) >= pos_off + 12:
                    a, b, alt_val = struct.unpack_from("<IIi", rec, pos_off)
                    lon, lat = decode_lonlat_dword(a, False), decode_lonlat_dword(b, True)
                    alt = alt_val / 1000.0
                else:
                    a, b = struct.unpack_from("<II", rec, pos_off)
                    lon, lat = decode_lonlat_dword(a, False), decode_lonlat_dword(b, True)
                    alt = 0.0

                guid_hex = rec[guid_off:guid_off + 16].hex().lower()

                pitch, roll, heading = 0.0, 0.0, 0.0
                if heading_off is not None and len(rec) >= heading_off + 4:
                    pitch_off = heading_off - 4
                    roll_off = heading_off - 2
                    try:
                        p_val = struct.unpack_from("<H", rec, pitch_off)[0] if len(rec) >= pitch_off+2 else 0
                        r_val = struct.unpack_from("<H", rec, roll_off)[0] if len(rec) >= roll_off+2 else 0
                        h_val = struct.unpack_from("<H", rec, heading_off)[0]
                        pitch = float(p_val * (360.0 / 65536.0))
                        roll = float(r_val * (360.0 / 65536.0))
                        heading = float(h_val * (360.0 / 65536.0))
                    except Exception:
                        pass

                found_placements.append({
                    "guid": guid_hex,
                    "title": None,
                    "livery": None,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "height_offset": placement_height_offset(alt, is_agl, arp_alt),
                    "pitch": pitch,
                    "roll": roll,
                    "hdg": heading,
                    "is_agl": is_agl,
                    "qmid1": sub.qmid1,
                    "qmid2": sub.qmid2,
                    "source": "Airport-Embedded",
                })
    return found_placements

def calibrate_guid_offset(records: list):
    if not records: return None
    rec_len = len(records[0])
    best = None
    for off in range(4, rec_len - 16 + 1):
        ok = True
        for rec in records:
            if (rec[off + 8] & 0xC0) != 0x80:
                ok = False
                break
        if ok:
            best = off
            break
    return best

def calibrate_heading_offset(records: list, claimed_ranges: list):
    if not records: return None, None
    rec_len = len(records[0])
    for off in range(4, rec_len - 4 + 1):
        if any(off in r or (off + 3) in r for r in claimed_ranges): continue
        ok = True
        for rec in records:
            (val,) = struct.unpack_from("<f", rec, off)
            if not (0.0 <= val < 360.0):
                ok = False
                break
        if ok: return off, "float"

    if 38 + 2 <= rec_len:
        if not any(38 in r or 39 in r for r in claimed_ranges):
            return 38, "bams32"
    return None, None

def find_guid_candidates(rec: bytes, start: int = 4):
    candidates = []
    for off in range(start, len(rec) - 15):
        if (rec[off + 8] & 0xC0) == 0x80:
            candidates.append(off)
    return candidates

def resolve_guid_hex(rec: bytes, guid_off, guid_map: dict) -> str:
    if guid_off is not None and len(rec) >= guid_off + 16:
        col_guid = rec[guid_off:guid_off + 16].hex().lower()
        if not guid_map or col_guid in guid_map:
            return col_guid
    else:
        col_guid = None

    candidates = find_guid_candidates(rec)
    for off in candidates:
        cand_hex = rec[off:off + 16].hex().lower()
        if cand_hex in guid_map: return cand_hex

    if col_guid is not None: return col_guid
    if candidates: return rec[candidates[0]:candidates[0] + 16].hex().lower()
    return ""

_PRINTABLE = frozenset(range(0x20, 0x7f))

def _extract_trailing_names(rec: bytes, start: int = 16):
    n = len(rec)
    best = None
    for pos in range(start, n - 6):
        len1, len2 = struct.unpack_from("<IH", rec, pos)
        if not (1 <= len1 <= 40): continue
        name1_start = pos + 6
        if name1_start + len1 > n: continue
        name1 = rec[name1_start:name1_start + len1]
        if not all(b in _PRINTABLE for b in name1): continue

        names = [name1.decode("ascii")]
        cur = name1_start + len1
        if len2 > 0:
            sep = cur + 1 if (cur < n and rec[cur] == 0) else cur
            if sep + len2 <= n:
                name2 = rec[sep:sep + len2]
                if all(b in _PRINTABLE for b in name2):
                    names.append(name2.decode("ascii"))
                    cur = sep + len2

        leftover = n - cur
        if 0 <= leftover <= 4 and all(b == 0 for b in rec[cur:n]):
            if best is None or leftover < best[0]: best = (leftover, names)
    if best: return best[1]

    for pos in range(start, n - 2):
        (length,) = struct.unpack_from("<H", rec, pos)
        if not (2 <= length <= 128): continue
        str_start = pos + 2
        if str_start + length > n: continue
        chunk = rec[str_start:str_start + length]
        if chunk[-1] != 0 or not all(b in _PRINTABLE for b in chunk[:-1]): continue
        leftover = n - (str_start + length)
        if 0 <= leftover <= 2 and all(b == 0 for b in rec[str_start + length:n]):
            if best is None or leftover < best[0]: best = (leftover, [chunk[:-1].decode("ascii")])

    return best[1] if best else []

def looks_like_title_record(records: list, min_fraction: float = 0.8) -> bool:
    if not records: return False
    hits = sum(1 for r in records if _extract_trailing_names(r))
    return (hits / len(records)) >= min_fraction

def format_guid(guid_bytes: bytes) -> str:
    d1, d2, d3 = struct.unpack_from("<IHH", guid_bytes, 0)
    d4 = guid_bytes[8:16]
    return f"{d1:08x}-{d2:04x}-{d3:04x}-{d4[0]:02x}{d4[1]:02x}-{d4[2:].hex()}"


def guid_str_to_bytes(guid_str: str) -> bytes:
    """Inverse of format_guid -- "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
    (braces optional) to the same 16-byte .NET-style layout format_guid
    reads (first 3 fields little-endian, last 8 bytes as-is)."""
    guid_str = guid_str.strip().strip("{}")
    parts = guid_str.split("-")
    d1 = int(parts[0], 16)
    d2 = int(parts[1], 16)
    d3 = int(parts[2], 16)
    d4 = bytes.fromhex(parts[3] + parts[4])
    return struct.pack("<IHH", d1, d2, d3) + d4


_MATERIAL_LIB_NAME_RE = re.compile(r'Name="([^"]+)"\s+Guid="\{?([0-9A-Fa-f-]+)\}?"')


def find_material_libraries(target_path: Path, log_callback=None):
    """Scans target_path for any "MaterialLibs/<vendor>/Library.xml" catalog
    (e.g. iniBuilds' EGLC ships "MaterialLibs/inibuilds-materials/
    Library.xml") and returns {16-byte material guid: material name}
    across every one found.

    These catalogs are how a package's own custom ground-surface/vector-
    polygon system (see scan_terrain_vector_db) resolves a bare material
    GUID reference into an actual texture set -- a glTF-embedded material
    this pipeline's normal mesh_convert path already handles never uses
    this indirection at all, so this is ADDITIVE, only consulted for
    TerrainVectorDb's own material references."""
    def _log(msg, level="info"):
        if log_callback: log_callback(msg, level)
        else: logger.info(msg)

    materials = {}
    if not target_path.is_dir():
        return materials
    for lib_xml in target_path.rglob("Library.xml"):
        try:
            text = lib_xml.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found_here = 0
        for m in _MATERIAL_LIB_NAME_RE.finditer(text):
            name, guid_str = m.group(1), m.group(2)
            try:
                materials[guid_str_to_bytes(guid_str)] = name
                found_here += 1
            except (ValueError, IndexError):
                continue
        if found_here:
            _log(f"      [MaterialLib] {lib_xml.relative_to(target_path)}: {found_here} material(s) catalogued", "info")
    return materials


# The 16-byte type-tag GUID that, within a TerrainVectorDb polygon
# record's own self-describing type+length+value property list, marks the
# VALUE as a reference into a package's own MaterialLibs/*/Library.xml
# catalog (see scan_terrain_vector_db's own docstring for the full
# reverse-engineering story of how this was confirmed).
_TVDB_MATERIAL_REF_TYPE = guid_str_to_bytes("cd28efd0-d3f0-43b6-9f88-4e4707344a04")

# Sane upper bound on a single TLV chunk's declared length, purely as a
# corruption/misalignment guard (see scan_terrain_vector_db) -- every real
# chunk observed was well under 100 bytes except the confirmed 64-byte
# "width profile" field; this is deliberately generous, not a real format
# constant.
_TVDB_MAX_PLAUSIBLE_CHUNK_LEN = 4096


def _tvdb_try_chunk(data: bytes, pos: int, data_end: int):
    """Returns (type_guid_bytes, length, value_start) if data[pos:] looks
    like a plausible type(16)+length(4) TLV header whose declared value
    fits within [pos, data_end) and isn't absurd, else None. See
    scan_terrain_vector_db's docstring for what this format actually is."""
    if pos + 20 > data_end:
        return None
    length = struct.unpack_from("<I", data, pos + 16)[0]
    if length < 0 or length > _TVDB_MAX_PLAUSIBLE_CHUNK_LEN or pos + 20 + length > data_end:
        return None
    return data[pos:pos + 16], length, pos + 20


def _tvdb_find_resync(data: bytes, start: int, data_end: int, min_consecutive: int = 8):
    """After a TLV chunk's declared length stops being plausible (the
    confirmed signal that non-TLV geometry/vertex data began -- see
    scan_terrain_vector_db), scans forward byte-by-byte for a position
    where at least min_consecutive chunks parse consistently and
    consecutively, and returns that position (or None if the rest of the
    subsection never resyncs). This is how the geometry blob between one
    polygon's material-reference properties and the next polygon's own
    property list gets skipped without needing to understand its
    contents."""
    pos = start
    while pos < data_end - 20:
        p = pos
        ok = True
        for _ in range(min_consecutive):
            parsed = _tvdb_try_chunk(data, p, data_end)
            if parsed is None:
                ok = False
                break
            _, length, value_start = parsed
            p = value_start + length
        if ok:
            return pos
        pos += 1
    return None


def scan_terrain_vector_db(data: bytes, sub_offset: int, sub_size: int, log_callback=None):
    """Best-effort scan of one TerrainVectorDb subsection (BGL section
    type 0x65) for ground-surface material references. Returns
    (feature_count, {material guid bytes, ...}).

    TerrainVectorDb is MSFS World Editor's own vector-polygon/vegetation
    database -- confirmed, through direct reverse-engineering against a
    real package (iniBuilds' EGLC), to hold custom ground-material
    polygons (taxiway/apron/runway pavement painted with a material from
    the package's own MaterialLibs/*/Library.xml, referenced by GUID) and
    individual tree/vegetation placements, both using the SAME underlying
    self-describing property scheme: a repeating [16-byte type-tag GUID]
    [4-byte length][length bytes of value] sequence, where the type GUID
    identifies the property's meaning (several confirmed: a feature-type
    marker, a material reference -- _TVDB_MATERIAL_REF_TYPE -- several
    scalar float/int fields, and a 64-byte "width profile" curve).

    What this does NOT recover: actual vertex/boundary geometry. After
    each polygon's property list, a variable-length block begins (visible
    as a delta/varint-shaped stream of small numeric sub-records, one per
    real airport polygon) that does not use this TLV scheme -- extensive
    testing (every standard absolute-coordinate encoding: float64/float32
    degrees, E6/E7 fixed-point, the classic FS9/BGL scaled-dword encoding,
    and varint/zigzag deltas, checked at every byte offset against the
    package's own real airport reference point) found no valid decode.
    Rather than guess and risk silently wrong geometry, this function
    treats that whole block as opaque: it locates where a polygon's own
    property list ends (the point where chunk lengths stop being
    plausible), resyncs past the geometry to the next recognizable TLV
    chunk (_tvdb_find_resync), and keeps collecting material references
    from however many more polygons follow. So the material-reference
    result here is reliable even though the geometry itself is unrecovered.
    """
    def _log(msg, level="info"):
        if log_callback: log_callback(msg, level)
        else: logger.info(msg)

    data_end = sub_offset + sub_size
    materials_found = set()
    feature_count = 0
    pos = sub_offset
    iterations = 0
    max_iterations = sub_size + 1000  # generous; just a corruption/infinite-loop guard

    # Always go through the validated (N-consecutive-chunk) resync before
    # trusting a position, including the very first one: a subsection's
    # own leading header bytes (qmid/count fields, not TLV data) can
    # coincidentally look like a plausible chunk, sending the walk into
    # garbage before it reaches the real first record, with no later
    # chance to recover the material references it skipped past.
    while pos < data_end:
        resync = _tvdb_find_resync(data, pos, data_end)
        if resync is None:
            break
        pos = resync
        feature_count += 1
        while pos < data_end and iterations < max_iterations:
            iterations += 1
            parsed = _tvdb_try_chunk(data, pos, data_end)
            if parsed is None:
                break
            type_guid, length, value_start = parsed
            if type_guid == _TVDB_MATERIAL_REF_TYPE and length == 16:
                materials_found.add(data[value_start:value_start + 16])
            pos = value_start + length
    return feature_count, materials_found


def parse_modeldata_index(data: bytes, start: int, size: int):
    end = start + size
    pos = start
    records = []
    while pos + 24 <= end:
        guid = data[pos:pos + 16]
        off_val, sz_val = struct.unpack_from("<II", data, pos + 16)
        target = start + off_val
        if target < start or target + sz_val > end or data[target:target + 4] != b"RIFF": break
        records.append((guid, target, sz_val))
        pos += 24
    return records

def parse_exclusion_rectangles(data: bytes, start: int, size: int):
    end = start + size
    pos = start
    rects = []
    while pos + 20 <= end:
        flags, a, b, c, d = struct.unpack_from("<IIIII", data, pos)
        rects.append({
            "flags": flags,
            "west": decode_lonlat_dword(a, False),
            "north": decode_lonlat_dword(b, True),
            "east": decode_lonlat_dword(c, False),
            "south": decode_lonlat_dword(d, True),
        })
        pos += 20
    return rects

def extract_riff_model(data: bytes, container_off: int, container_size: int):
    tag, riff_size, form = struct.unpack_from("<4sI4s", data, container_off)
    if tag != b"RIFF" or form != b"GLTF": return None, None
    subpos = container_off + 12
    container_end = container_off + 8 + riff_size
    name, glb_bytes = None, None

    while subpos + 8 <= container_end:
        cc, csz = struct.unpack_from("<4sI", data, subpos)
        cdata_start = subpos + 8

        if cc == b"GXML":
            xml = data[cdata_start:cdata_start + csz].decode("utf-8", errors="ignore")
            m_name = re.search(r'name="([^"]*)"', xml)
            name = m_name.group(1) if m_name else None
        elif cc == b"GLBD":
            if cdata_start + 4 <= container_end:
                magic_check = data[cdata_start:cdata_start + 4]
                if magic_check == b"GLB\x00":
                    _ncc, ncsz = struct.unpack_from("<4sI", data, cdata_start)
                    glb_bytes = data[cdata_start + 8:cdata_start + 8 + ncsz]
                else:
                    glb_bytes = data[cdata_start:cdata_start + csz]

        subpos += 8 + csz
        if csz % 2 == 1: subpos += 1

    return name, glb_bytes

def _decompress_and_write_riff_model(data: bytes, container_off: int, container_size: int, guid: bytes, out_dir: Path):
    """Shared by extract_models_from_modeldata (a scenery package's own
    ModelData sections) and extract_from_install_index (an MSFS install's
    own base .bgl files, for stock/ASOBO objects a package doesn't ship
    geometry for) -- same RIFF extraction, GLBZ/zstd decompression, and
    atomic write either way, since it's the exact same on-disk model
    format regardless of which .bgl the container came from. Returns
    (guid_hex, file_stem, name, out_path) or None if this container
    isn't a usable GLB/GLBZ model."""
    name, glb_bytes = extract_riff_model(data, container_off, container_size)
    if not glb_bytes:
        return None

    guid_hex = guid.hex().lower()
    guid_display = format_guid(guid)
    magic = glb_bytes[:4]

    if magic == GLB_MAGIC:
        pass
    elif magic == b"GLBZ":
        if zstd is None:
            return None
        try:
            decomp_size = struct.unpack_from("<I", glb_bytes, 8)[0]
            dctx = zstd.ZstdDecompressor()
            glb_bytes = dctx.decompress(glb_bytes[12:], max_output_size=decomp_size)
        except Exception:
            return None
    else:
        return None

    base = re.sub(r'[^A-Za-z0-9_-]+', '_', name) if name else guid_display.replace('-', '')
    # Full 32-hex-char GUID, not just its first dash-delimited segment (8
    # hex chars): a large run combining a package's own ModelData GUIDs
    # with on-demand base-game GUIDs has thousands of distinct models,
    # and an 8-char prefix is a real birthday-paradox collision risk at
    # that scale -- two different models colliding on file_stem would
    # have one silently overwrite the other's .glb, so every placement
    # referencing the losing GUID would load the wrong geometry.
    file_stem = f"{base}_{guid_hex}"

    out_path = out_dir / f"{file_stem}.glb"
    # Atomic write: the same GUID model can legitimately appear in
    # multiple .bgl files (a shared library object), which the GPU
    # fork's process pool may now parse concurrently -- a plain
    # write_bytes() here could let two processes' writes to the same
    # path interleave. Content is deterministic from the GUID either
    # way, so whichever temp file wins the rename is fine; this just
    # guarantees the final file is never a half-written mix of both.
    temp_path = out_path.with_name(f"{out_path.name}.tmp_{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}")
    temp_path.write_bytes(glb_bytes)
    # Windows can raise PermissionError on os.replace() if another
    # process momentarily has the destination open (e.g. another
    # worker mid-read of this same shared model) -- retry briefly
    # rather than failing this model's extraction over a transient lock.
    for attempt in range(8):
        try:
            os.replace(temp_path, out_path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))

    return guid_hex, file_stem, name, out_path


def extract_models_from_modeldata(data: bytes, start: int, size: int, out_dir: Path, log_callback=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = parse_modeldata_index(data, start, size)

    extracted, guid_map, name_map = [], {}, {}
    for guid, container_off, container_size in records:
        result = _decompress_and_write_riff_model(data, container_off, container_size, guid, out_dir)
        if result is None:
            continue
        guid_hex, file_stem, name, out_path = result
        extracted.append(out_path)
        guid_map[guid_hex] = file_stem
        if name:
            name_map[guid_hex] = name

    return extracted, guid_map, name_map


def _index_one_bgl_for_guids(bgl_path_str):
    """Lightweight sibling of _parse_one_bgl_uncached, for indexing an
    MSFS install's own base .bgl files (not a scenery package's) for
    stock/ASOBO GUIDs a package doesn't ship geometry for. Only reads each
    file's ModelData INDEX tables (guid -> RIFF container offset/size) --
    never extracts/decompresses the actual model geometry, since most
    indexed GUIDs will never actually be looked up. Used by
    build_install_guid_index() to build a persistent, cached
    guid -> (bgl_path, container_off, container_size) index once per
    configured MSFS install root."""
    bgl = Path(bgl_path_str)
    index = {}
    try:
        data = bgl.read_bytes()
        sections = parse_bgl(data)
        for sec in sections:
            if sec.type_name != "ModelData":
                continue
            for sub in sec.subsections:
                for guid, container_off, container_size in parse_modeldata_index(data, sub.data_offset, sub.data_size):
                    guid_hex = guid.hex().lower()
                    if guid_hex not in index:
                        index[guid_hex] = (bgl_path_str, container_off, container_size)
    except Exception:
        pass
    return index


# Files that sit directly in the true top-level MSFS install folder --
# used by _resolve_real_install_root to recognize it regardless of which
# subfolder the user actually pointed the install-root field at.
_MSFS_ROOT_MARKERS = ("FlightSimulator.exe", "Content.xml")


def _resolve_real_install_root(user_selected_root):
    """Some MSFS installs (confirmed on a real Steam install: the top
    folder holds a "Packages" directory with only a handful of small
    fs-base-* packages, while the actual bulk content -- including base
    generic library objects like the taxiway/runway/approach light models
    this exists to find -- lives in a SIBLING "HLM_Packages/Official/Steam"
    tree instead) put the real content root somewhere other than the
    "Packages" folder most guidance tells users to pick. Since GUID
    indexing only ever scans *downward* (rglob), pointing install_root at
    "Packages" alone would never reach "HLM_Packages" at all -- it's a
    sibling, not a descendant.

    Walks upward from whatever folder was actually selected looking for
    the true top-level install folder (identified by FlightSimulator.exe
    or Content.xml sitting directly in it, both standard top-level
    install files across MSFS's known layouts), and uses THAT as the scan
    root instead -- so indexing finds the real content regardless of
    exactly which subfolder the user browsed to. Falls back to the
    original selection unchanged if no such marker is found nearby (e.g.
    a non-standard/portable install), rather than guessing further.
    """
    selected = Path(user_selected_root)
    for candidate in [selected] + list(selected.parents):
        if any((candidate / marker).exists() for marker in _MSFS_ROOT_MARKERS):
            return candidate
    return selected


def build_install_guid_index(install_root, log_callback=None):
    """Builds a {guid_hex: (bgl_path, container_off, container_size)} index
    across every .bgl file under an MSFS install root, so extract() can
    resolve placements whose GUID points at an ASOBO stock/library model
    the scenery package itself never ships geometry for (the package's own
    BGLs only ever reference that GUID, never define it).

    This is deliberately the ONE expensive part of stock-object resolution:
    a full MSFS install commonly has many thousands of base-game .bgl
    files. Disk-cached via cache_utils, keyed on the install root's own
    path, so it only runs once per configured install -- not once per
    pipeline run. Delete the cache folder (see cache_utils.cache_root(),
    logged at pipeline start) to force a rescan after an MSFS update.
    """
    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)

    original_root = Path(install_root)
    install_root = _resolve_real_install_root(install_root)
    if install_root != original_root:
        _log(f"MSFS install root: using {install_root} (found FlightSimulator.exe/Content.xml there) "
             f"instead of the selected {original_root}, so content living in a sibling folder "
             f"(e.g. HLM_Packages/Official) isn't missed.", "info")
    key_parts = (str(install_root.resolve()),)
    cached = cache_utils.get("msfs_install_index", *key_parts)
    if cached is not None:
        _log(f"Using cached MSFS install GUID index for {install_root} ({len(cached)} model(s) indexed).", "info")
        return cached

    bgl_files = glob_ci(install_root, ".bgl")
    _log(f"Indexing {len(bgl_files)} .bgl file(s) under {install_root} for stock MSFS objects "
         f"-- this happens once and is cached for future runs...", "info")

    index = {}
    if bgl_files:
        max_workers = max(1, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for partial in pool.map(_index_one_bgl_for_guids, [str(p) for p in bgl_files], chunksize=32):
                for guid_hex, entry in partial.items():
                    index.setdefault(guid_hex, entry)  # first file found wins on a duplicate GUID

    cache_utils.set("msfs_install_index", index, *key_parts)
    _log(f"Indexed {len(index)} stock model GUID(s) from {install_root}.", "success")
    return index


def extract_from_install_index(guids_needed, install_index, models_dir, log_callback=None):
    """For each GUID in guids_needed that install_index actually covers,
    extracts+writes that one model's .glb -- reading only the ONE RIFF
    container needed per GUID from whichever install .bgl file
    install_index says it's in (grouped so each install .bgl is opened at
    most once, even if it supplies several needed GUIDs), rather than
    re-parsing that file's whole ModelData section the way a full package
    extraction would. Returns (guid_map_additions, name_map_additions) in
    the exact shape extract()'s other two GUID sources (a package's own
    ModelData sections, discover_simobjects) already use, so callers merge
    it the same way -- see extract()."""
    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    guid_map, name_map = {}, {}

    by_file = {}
    for guid_hex in guids_needed:
        entry = install_index.get(guid_hex)
        if entry is not None:
            by_file.setdefault(entry[0], []).append((guid_hex, entry[1], entry[2]))

    if not by_file:
        return guid_map, name_map

    for bgl_path_str, entries in by_file.items():
        try:
            data = Path(bgl_path_str).read_bytes()
        except OSError as e:
            _log(f"[MSFS install] Could not read {bgl_path_str}: {e}", "warning")
            continue
        for guid_hex, container_off, container_size in entries:
            try:
                guid_bytes = bytes.fromhex(guid_hex)
                result = _decompress_and_write_riff_model(data, container_off, container_size, guid_bytes, models_dir)
                if result is None:
                    continue
                found_guid_hex, file_stem, name, _out_path = result
                guid_map[found_guid_hex] = file_stem
                if name:
                    name_map[found_guid_hex] = name
            except Exception as e:
                _log(f"[MSFS install] Failed extracting GUID {guid_hex}: {e}", "warning")

    if guid_map:
        _log(f"Recovered {len(guid_map)} stock MSFS object(s) from your MSFS install for "
             f"placements this package doesn't ship geometry for.", "success")
    return guid_map, name_map


def resolve_attach_offset(x, y, z, anchor_lat, anchor_lon, anchor_pitch, anchor_roll, anchor_hdg):
    """A SimPropAttach child's local OffsetXYZ (x, y, z) -> its real-world
    (lat, lon, dy) once the parent container's own Pitch/Roll/Heading is
    applied. dy is the vertical component (metres above/below the
    container's own contact point) -- there's no separate lat/lon
    component to it.

    Pitch/Roll are applied here as a local pre-rotation of the offset (a
    ground SceneryObject's own pitch/roll is essentially always 0 in
    practice, making this step a no-op for the overwhelming majority of
    real placements); the HEADING rotation + lat/lon conversion is handed
    to geo_transform.local_offset_to_latlon -- the one place in this whole
    project that convention is defined, and every other placement/
    recentering/terrain-fit call already uses it.

    BUG FIXED here: this function used to hand-roll its own heading
    rotation (dx = x2*cy + z2*sy, dz = -x2*sy + z2*cy) instead of calling
    geo_transform.rotate_xz. At heading 0 both agree on X, but the Z
    (north/south) term came out with the OPPOSITE sign from geo_transform's
    -- every SimPropAttach child (a building's separate "interior" object,
    most apron lights/lamps, jetways/doors resolved via this path) was
    mirrored north<->south around its own container's anchor, worse the
    further the OffsetXYZ reaches and the more the container's own heading
    rotates that mirroring into a sideways/diagonal miss -- e.g. a
    building's separate interior model rendering out to the side of its
    own exterior shell instead of inside it."""
    p = math.radians(anchor_pitch)
    r = math.radians(anchor_roll)
    cx, sx = math.cos(p), math.sin(p)
    cz, sz = math.cos(r), math.sin(r)

    # 1. Roll (around Z axis)
    x1 = x * cz + y * sz
    y1 = -x * sz + y * cz
    z1 = z

    # 2. Pitch (around X axis)
    x2 = x1
    y2 = y1 * cx + z1 * sx
    z2 = -y1 * sx + z1 * cx

    dy = y2
    lat, lon = geo_transform.local_offset_to_latlon(anchor_lat, anchor_lon, anchor_hdg, x2, z2)
    return lat, lon, dy


def _resolve_propdefs_dir(explicit, spb2xml_dir, _log):
    """Locates the MSFS SDK "Propdefs" XML folder .spb decoding needs to
    make any sense of a SimPropContainer's binary property data.

    These are Microsoft/Asobo's own SDK/game data files (manifest.json
    inside a real copy reads `"creator": "Asobo Studio"`) -- NOT something
    this project can legally bundle inside a redistributed app. So they're
    never shipped in the exe or the source tree; instead this searches, in
    order: (1) `explicit` -- the folder the user configured (Settings ->
    "Propdefs folder", or the MSFS2XP_PROPDEFS_DIR env var), pointed at
    THEIR OWN MSFS install/SDK's copy (or an exported copy kept locally --
    load_propdefs walks every .xml recursively, so the exact internal
    folder layout doesn't matter, only that everything's under one root);
    (2) a `propdefs` folder next to spb2xml, kept only for a dev machine
    that already has one sitting there -- never present in a delivered
    copy of this app. Returns the resolved directory, or None (warning
    logged) if nothing usable was found anywhere.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("MSFS2XP_PROPDEFS_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [spb2xml_dir / "Common", spb2xml_dir / "propdefs" / "Common", spb2xml_dir / "propdefs"]

    for base in candidates:
        for cand in (base, base / "Common"):
            try:
                if cand.is_dir() and next(cand.rglob("*.xml"), None) is not None:
                    return cand
            except OSError:
                continue
    _log("      [SPB] No Propdefs folder configured or found -- SimPropContainer-based "
         "placements (many apron lights/lamps, jetways, animated doors, building "
         "interiors) can't be decoded without it. These are Microsoft/Asobo's own MSFS "
         "SDK data files, not shipped with this tool -- point Settings -> \"Propdefs "
         "folder (optional)\" at your own MSFS install/SDK's Propdefs folder.", "warning")
    return None


def extract_spb_placements(spb_path: Path, airport_lat: float, airport_lon: float, airport_alt: float, existing_placements: list, _log, name_map: dict = None, propdefs_dir: str = None, allow_fallback: bool = True):
    import sys
    import uuid

    cwd = Path.cwd()
    spb2xml_dir = cwd / "spb2xml"
    if not spb2xml_dir.exists():
        spb2xml_dir = Path(__file__).resolve().parent / "spb2xml"

    if spb2xml_dir.exists() and str(spb2xml_dir) not in sys.path:
        sys.path.insert(0, str(spb2xml_dir))

    try:
        from decompiler import Decompiler
        from propdefs import load_propdefs
    except ImportError:
        _log(f"      [SPB] Could not import 'decompiler' from {spb2xml_dir}. Ensure spb2xml is present.", "warning")
        return []

    propdefs_dir = _resolve_propdefs_dir(propdefs_dir, spb2xml_dir, _log)
    if propdefs_dir is None:
        return []

    try:
        # load_propdefs re-parses every property-definition XML under
        # propdefs_dir from scratch each call, and this function runs
        # once per .spb file -- cached by directory path since they're
        # identical for the whole lifetime of this process.
        cache_key = str(propdefs_dir)
        bank = _PROPDEFS_CACHE.get(cache_key)
        if bank is None:
            bank = load_propdefs(cache_key)
            _PROPDEFS_CACHE[cache_key] = bank
        dec = Decompiler(str(spb_path), bank)
        root = dec.decompile()
    except Exception as e:
        _log(f"      [SPB] Failed to parse {spb_path.name}: {e}", "error")
        return []
        
    container_guid_hex = None
    container_guid_node = root.find(".//SimBase.GUID")
    
    if container_guid_node is not None and container_guid_node.text:
        try:
            raw_guid = container_guid_node.text.strip("{}")
            container_guid_hex = uuid.UUID(raw_guid).bytes_le.hex().lower()
        except Exception:
            pass

    # A SimPropContainer's GUID is placed once per real-world instance in
    # the scenery BGL (an apron light/lamp fixture can repeat 50+ times).
    # Expand the container against EVERY matching placement (one
    # attached-model instance per container instance).
    #
    # A container whose GUID is only ever produced by ANOTHER .spb's
    # attach output (nested attach chains -- e.g. a seat/furniture
    # cluster attached to a master container that is itself
    # SPB-attached, not a raw BGL placement) legitimately has no anchor
    # the first time it's processed, purely because of file-processing
    # order. extract()'s own orchestration runs a MULTI-PASS retry:
    # allow_fallback=False here returns None -- a distinct "retry me
    # after other .spb files have had a chance to resolve" signal, never
    # placed at the airport's reference point as a guess. Only once a
    # full pass makes no further progress does extract() drop what's
    # still pending (confirmed real symptom of a too-blunt single-pass
    # skip: 2616 -> 1477 unique placement groups on a real EGLC
    # conversion, breaking interior content -- "the seats of the
    # building... outside of the building"). allow_fallback=True (the
    # default, used by direct callers e.g. tests) keeps the single-pass
    # airport-centre fallback for compatibility.
    anchors = []
    if container_guid_hex:
        anchors = [p for p in existing_placements if p.get("guid") == container_guid_hex]
    used_fallback = not anchors
    if used_fallback:
        if not allow_fallback:
            return None
        if airport_lat is None or airport_lon is None:
            _log(f"      [SPB] {spb_path.name}: Parent container placement NOT FOUND. Skipping.", "error")
            return []
        anchors = [{"lat": airport_lat, "lon": airport_lon, "alt": airport_alt,
                    "pitch": 0.0, "roll": 0.0, "hdg": 0.0, "is_agl": True}]

    found = []
    name_map = name_map or {}
    _attaches = root.findall(".//SimPropAttach")

    for _anchor in anchors:
      anchor_lat = _anchor["lat"]
      anchor_lon = _anchor["lon"]
      anchor_alt = _anchor.get("alt", 0.0)
      anchor_pitch = _anchor.get("pitch", 0.0)
      anchor_roll = _anchor.get("roll", 0.0)
      anchor_hdg = _anchor.get("hdg", 0.0)
      anchor_is_agl = _anchor.get("is_agl", True)
      for attach in _attaches:
        offset_node = attach.find(".//OffsetXYZ")
        guid_node = attach.find(".//WorldBase.MDLGuid")
        hdg_node = attach.find(".//Orientation")

        # offset_node is legitimately allowed to be missing: MSFS's binary
        # .spb format omits a property entirely when it equals its default,
        # and OffsetXYZ defaults to (0,0,0) -- a model glued directly to
        # its parent (common) has WorldBase.MDLGuid but no OffsetXYZ
        # element. Only WorldBase.MDLGuid is actually required.
        attach_guid_hex = None
        if guid_node is not None and guid_node.text:
            try:
                attach_guid_str = guid_node.text.strip("{}")
                attach_guid_hex = uuid.UUID(attach_guid_str).bytes_le.hex().lower()
            except Exception:
                attach_guid_hex = None
        else:
            # A second, fairly common attach shape has no WorldBase.MDLGuid
            # at all, instead a SimContain.Container/WorldBase.ContainerTitle
            # referencing another SimObject by title. Resolved the same way
            # regular title-based SceneryObject placements are resolved
            # elsewhere in this module: scan name_map (keyed by GUID, valued
            # by title) for a case-insensitive title match.
            title_node = attach.find(".//WorldBase.ContainerTitle")
            if title_node is not None and title_node.text:
                title = title_node.text.strip()
                for g, nm in name_map.items():
                    if nm.lower() == title.lower():
                        attach_guid_hex = g
                        break

        if attach_guid_hex:
            try:
                if offset_node is not None and offset_node.text:
                    parts = offset_node.text.split(",")
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                else:
                    x, y, z = 0.0, 0.0, 0.0

                lat, lon, dy = resolve_attach_offset(x, y, z, anchor_lat, anchor_lon,
                                                      anchor_pitch, anchor_roll, anchor_hdg)
                alt = anchor_alt + dy
                # height_offset must be "alt" (anchor_alt + dy), not dy
                # alone: dy is only the offset from the parent
                # SimPropContainer's own contact point, and silently
                # drops anchor_alt (the parent's own height above real
                # terrain contact) when the container is itself elevated
                # -- desyncing this object from anything else placed at
                # the same real-world point via a different path. X-Plane's
                # DSF OBJECT point pool has no vertical/AGL field of its
                # own, so this full accumulated alt is what has to be
                # baked into the exported .obj geometry downstream.

                c_pitch, c_roll, c_hdg = 0.0, 0.0, 0.0
                if hdg_node is not None and hdg_node.text:
                    h_parts = hdg_node.text.split(",")
                    if len(h_parts) >= 3:
                        c_pitch, c_roll, c_hdg = float(h_parts[0]), float(h_parts[1]), float(h_parts[2])
                
                final_pitch = (anchor_pitch + c_pitch) % 360.0
                final_roll = (anchor_roll + c_roll) % 360.0
                final_hdg = (anchor_hdg + c_hdg) % 360.0
                        
                found.append({
                    "guid": attach_guid_hex,
                    "title": attach.get("DisplayName"),
                    "livery": None,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "height_offset": alt,
                    "pitch": final_pitch,
                    "roll": final_roll,
                    "hdg": final_hdg,
                    "is_agl": anchor_is_agl,
                    "qmid1": 0,
                    "qmid2": 0,
                    # the wrapping SimPropContainer's GUID -- lets extract()
                    # drop the now-redundant raw container placements so
                    # they don't get re-reported as unresolved
                    "container_guid": container_guid_hex,
                    "source": f"SPB-SimPropContainer{' (Fallback)' if used_fallback else ''}"
                })
            except Exception as e:
                _log(f"      [SPB] Error parsing offsets in {spb_path.name}: {e}", "warning")

    if not found:
        _log(f"      [SPB] No valid attached models found inside {spb_path.name}.", "warning")
    elif used_fallback:
        _log(f"      [SPB] {spb_path.name}: Parent container placement NOT FOUND. Falling back to airport center.", "warning")
    else:
        _n_attach = max(1, len(found) // max(1, len(anchors)))
        _log(f"      [SPB] {spb_path.name}: ANCHORED {_n_attach} SimPropObject(s) to "
             f"{len(anchors)} container placement(s) -> {len(found)} instance(s).", "info")

    return found


def _parse_sim_cfg_title_by_model(cfg_path: Path):
    """Parses a SimObject's sim.cfg for {model value -> title value} pairs,
    one per [fltsim.N] section (title= and model= must be read from the
    SAME section, not just anywhere in the file, since a sim.cfg can
    register several different models/titles side by side)."""
    title_by_model = {}
    current_title = None
    current_model = None
    in_fltsim_section = False
    try:
        text = cfg_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return title_by_model

    def _flush():
        if current_title is not None and current_model:
            title_by_model[current_model] = current_title

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            _flush()
            in_fltsim_section = line.lower().startswith("[fltsim.")
            current_title, current_model = None, None
            continue
        if not in_fltsim_section or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if key == "title":
            current_title = value
        elif key == "model":
            current_model = value
    _flush()
    return title_by_model


def _iter_simobject_models(target_path: Path, log_callback=None):
    """Scans every sim.cfg under target_path and yields (guid_hex, title,
    gltf_or_glb_src_path, xml_path) for each resolvable SimObject model --
    the scanning/parsing half of what discover_simobjects does, WITHOUT
    copying any files, so it can also be used to just INDEX a large tree
    (an MSFS install root) without eagerly copying every model found in
    it, most of which a given package will never actually reference. See
    build_install_simobject_index for that use.

    xml_path is the model's own ModelInfo/ModelBehaviors XML file (already
    located below to read the GUID out of its <ModelInfo> element) --
    yielded so callers can stage the SAME file forward as a real,
    on-disk companion to the copied model (see _copy_simobject_model),
    which is what convert()'s parse_time_behavior actually reads for
    blink/business-hours/proximity animation triggers. Previously this
    function read the XML only for its own internal GUID parsing and threw
    the path away, so no copied SimObject model ever had its own behavior
    XML alongside it and convert() could never find one -- animations
    (including this project's own FlyWithLua-driven blink/proximity
    triggers) never fired for any SimObject-sourced model, regardless of
    what real animation channels or behavior XML it actually had.

    ModelFile can point at either a loose .gltf (+ sibling .bin buffer(s))
    or a self-contained .glb -- both are handled identically here (and by
    _copy_simobject_model below), since MSFS ships stock SimObjects in
    both forms."""
    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)
        else:
            logger.info(msg)

    sim_cfgs = list(target_path.rglob("sim.cfg"))
    for cfg_path in sim_cfgs:
        sim_folder = cfg_path.parent
        title_by_model = _parse_sim_cfg_title_by_model(cfg_path)

        for model_value, title in title_by_model.items():
            model_dir = sim_folder / f"model.{model_value}"
            if not model_dir.is_dir():
                continue
            xml_path = model_dir / f"{model_value}.xml"
            if not xml_path.exists():
                xml_candidates = list(model_dir.glob("*.xml"))
                if not xml_candidates:
                    continue
                xml_path = xml_candidates[0]

            # Some of these files aren't well-formed XML: MSFS's own tooling
            # writes a <ModelInfo>...</ModelInfo> element (guid + LODs, all
            # we need here) immediately followed by a SIBLING top-level
            # <ModelBehaviors>...</ModelBehaviors> element -- two roots in
            # one file, which a standard parser rejects outright. Since
            # everything needed lives inside that first, well-formed
            # <ModelInfo> fragment, parse just that slice instead of the
            # whole file.
            try:
                xml_text = xml_path.read_text(encoding="utf-8", errors="replace")
                end = xml_text.find("</ModelInfo>")
                if end != -1:
                    xml_text = xml_text[: end + len("</ModelInfo>")]
                root = ET.fromstring(xml_text)
            except Exception as e:
                _log(f"      [SimObjects] Failed parsing {xml_path.name}: {e}", "warning")
                continue

            raw_guid = root.get("guid", "")
            try:
                guid_hex = uuid.UUID(raw_guid.strip("{}")).bytes_le.hex().lower()
            except (ValueError, AttributeError):
                continue

            lod_files = [lod.get("ModelFile") for lod in root.findall(".//LOD") if lod.get("ModelFile")]
            if not lod_files:
                continue
            chosen = next((f for f in lod_files if "_LOD0" in f), lod_files[0])
            model_src = model_dir / chosen
            if not model_src.exists():
                continue

            yield guid_hex, title, model_src, xml_path


def _read_glb_json(path: Path):
    """Reads just the JSON chunk of a binary glTF (.glb) container -- the
    first chunk, always JSON per spec -- without touching the (often much
    larger) BIN chunk that follows it. Returns the parsed dict, or None if
    the file isn't a well-formed GLB."""
    try:
        with path.open("rb") as f:
            header = f.read(12)
            if len(header) < 12 or header[:4] != GLB_MAGIC:
                return None
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                return None
            chunk_length, chunk_type = struct.unpack_from("<II", chunk_header, 0)
            if chunk_type != 0x4E4F534A:  # ASCII "JSON", little-endian uint32
                return None
            return json.loads(f.read(chunk_length).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _copy_simobject_model(model_src: Path, models_dir: Path, log_callback=None, xml_src: Path = None):
    """Copies one SimObject model (a loose .gltf, or a .glb -- either can
    carry sibling buffer/image files its own JSON references by URI) into
    models_dir, exactly like any other extracted model. Returns the
    destination file's stem (for guid_map), or None on failure.

    MSFS's own SimObject convention is for every object to ship its model
    as a generically named "model.glb"/"model.gltf" inside its own
    dedicated subfolder -- staging by bare filename alone collides
    whenever two different SimObjects share that filename, which is the
    norm, not an edge case. Disambiguating the destination name avoids
    one SimObject's GUID silently mapping to a different one's files.

    Disambiguated with a short hash of the model's own FULL SOURCE PATH,
    prefixed (not appended) so any trailing "_LODnn" the source filename
    already carries stays at the true end of the stem -- xml staging below
    strips that suffix with an end-anchored regex, which would silently
    stop matching if something were appended after it instead. Unique
    regardless of what naming convention (or lack of one) the source
    package uses, and STABLE across repeated runs of the same source path,
    so the existing disk-cache-friendly "if not dest.exists(): skip" reuse
    for a genuinely identical model still works.

    xml_src, if given, is that model's own ModelInfo/ModelBehaviors XML --
    staged alongside the copied model under the exact filename convert()'s
    own parse_time_behavior expects: the model's LOD-stripped stem plus
    ".xml", in the SAME directory (glb_path.parent) it looks in. Without
    this, convert() can never find a real, on-disk behavior XML for any
    SimObject-sourced model, so its blink/business-hours/proximity
    animation detection is permanently a no-op regardless of what
    animation channels or behavior data the source model actually has."""
    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)
        else:
            logger.info(msg)

    path_hash = hashlib.md5(str(model_src).encode("utf-8")).hexdigest()[:8]
    dest = models_dir / f"{path_hash}_{model_src.name}"
    try:
        if not dest.exists():
            shutil.copyfile(model_src, dest)
            # glTF's external-URI mechanism for buffers/images isn't
            # exclusive to loose .gltf -- a .glb container's "self-
            # contained" guarantee only covers its own embedded BIN
            # chunk, so its images[] entries can still carry an external
            # "uri" instead of a bufferView, same as .gltf. Real
            # install-sourced GSE/vehicle/character SimObjects are
            # shipped as .glb files that do exactly this, sharing one
            # common texture library across many models. Copying the
            # specific files this model's own glTF JSON references
            # (never a bulk directory copy) into models_dir alongside it
            # is what lets convert()'s extract_image() find them (its
            # first candidate is glb_parent/uri).
            #
            # The image URI isn't always a same-folder sibling either --
            # a character SimObject's own model folder can carry no
            # textures at all, with them living in a shared library two
            # levels up (a sibling "texture" folder of the whole
            # SimObject category), reused across every model variant
            # under it. Every plausible nesting depth is tried, closest
            # first, so the common same-folder case still resolves on
            # the first candidate.
            suffix = model_src.suffix.lower()
            if suffix in (".gltf", ".glb"):
                try:
                    if suffix == ".gltf":
                        gltf_json = json.loads(model_src.read_text(encoding="utf-8"))
                    else:
                        gltf_json = _read_glb_json(model_src)
                    if gltf_json is not None:
                        for buf in gltf_json.get("buffers", []):
                            uri = buf.get("uri")
                            if uri and not uri.startswith("data:"):
                                buf_src = model_src.parent / uri
                                if buf_src.exists():
                                    shutil.copyfile(buf_src, models_dir / Path(uri).name)
                        parents = [model_src.parent, model_src.parent.parent, model_src.parent.parent.parent]
                        for img in gltf_json.get("images", []):
                            uri = img.get("uri")
                            if not uri or uri.startswith("data:"):
                                continue
                            img_dest = models_dir / Path(uri).name
                            if img_dest.exists():
                                continue
                            uri_name = Path(uri).name
                            candidates = [p / uri for p in parents] + [
                                p / tex_folder / uri_name
                                for p in parents
                                for tex_folder in ("texture", "TEXTURE")
                            ]
                            img_src = next((c for c in candidates if c.exists()), None)
                            if img_src is not None:
                                shutil.copyfile(img_src, img_dest)
                except Exception as e:
                    _log(f"      [SimObjects] Failed copying buffers/textures for {model_src.name}: {e}", "warning")
    except Exception as e:
        _log(f"      [SimObjects] Failed copying {model_src.name}: {e}", "warning")
        return None

    if xml_src is not None:
        try:
            if xml_src.exists():
                xml_stem = re.sub(r"_LOD[0-9]+$", "", dest.stem)
                xml_dest = models_dir / f"{xml_stem}.xml"
                if not xml_dest.exists():
                    shutil.copyfile(xml_src, xml_dest)
        except Exception as e:
            _log(f"      [SimObjects] Failed copying behavior XML for {model_src.name}: {e}", "warning")

    return dest.stem


def discover_simobjects(target_path: Path, out_dir: Path, log_callback=None):
    """MSFS SimObjects (jetways, GSE, doors, blinking lights, barriers, ...)
    live as standalone loose .gltf+.bin (or self-contained .glb) files under
    a "SimObjects" folder, referenced from scenery placements by TITLE
    (matching sim.cfg's title=), never embedded inside a .bgl's ModelData
    section the way regular scenery models are -- so extract()'s normal
    RIFF/GLB extraction never finds or converts them, and any placement
    record referencing one by title silently fails to match (falls into
    the "orphaned" bucket).

    This scans every sim.cfg under target_path (see _iter_simobject_models)
    and copies each resolved model into out_dir/models (see
    _copy_simobject_model) so the rest of the pipeline treats it exactly
    like any other extracted model -- including its own animations (a
    SimObject's .gltf can carry the exact same kind of ANIM_ channels a
    regular scenery model does; convert() doesn't care which discovery path
    found it). Returns (guid_map_additions, name_map_additions) in the same
    shape extract() already merges into its own guid_map/name_map --
    guid_map keyed by the model's own GUID for GUID-based placements,
    name_map keyed by that same GUID pointing at the sim.cfg title, so the
    EXISTING title-based placement-matching logic (already used for ~2/3 of
    title placements) picks these up with no changes to the matching logic
    itself.
    """
    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)
        else:
            logger.info(msg)

    guid_additions = {}
    name_additions = {}
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    found_count = 0
    sim_cfg_count = len(list(target_path.rglob("sim.cfg")))
    for guid_hex, title, model_src, xml_path in _iter_simobject_models(target_path, log_callback=log_callback):
        model_stem = _copy_simobject_model(model_src, models_dir, log_callback=log_callback, xml_src=xml_path)
        if model_stem is None:
            continue
        guid_additions[guid_hex] = model_stem
        name_additions[guid_hex] = title
        found_count += 1

    if found_count:
        _log(f"      [SimObjects] Discovered {found_count} SimObject model(s) "
             f"(jetways/GSE/doors/etc.) from {sim_cfg_count} sim.cfg file(s).", "info")
    return guid_additions, name_additions


def build_install_simobject_index(install_root, log_callback=None):
    """Cached {guid_hex: (model_src_path, title)} index of every loose
    SimObject-style model (jetways/GSE/animated vehicles/doors/etc,
    discovered via sim.cfg + ModelInfo XML -- see _iter_simobject_models)
    under an MSFS install root. Companion to build_install_guid_index,
    which only covers .bgl-embedded ModelData -- MSFS ships plenty of its
    own stock objects the OTHER way instead, as loose .gltf/.glb files,
    including several with real animations that convert() already knows
    how to translate once the model file itself is on disk, exactly the
    same as a package's own SimObjects (it's the identical file format
    either way -- discover_simobjects already handles both).

    Indexes only (doesn't copy any files) for the same reason
    build_install_guid_index doesn't eagerly extract every GUID it finds:
    most stock SimObjects in a full install will never be referenced by
    any given package. Cached via cache_utils, keyed on the install root's
    own path, so a full install only gets scanned once."""
    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)

    original_root = Path(install_root)
    install_root = _resolve_real_install_root(install_root)
    if install_root != original_root:
        _log(f"MSFS install root: using {install_root} instead of the selected {original_root} "
             f"(see build_install_guid_index's log for why).", "info")
    key_parts = (str(install_root.resolve()),)
    cached = cache_utils.get("msfs_install_simobject_index", *key_parts)
    if cached is not None:
        _log(f"Using cached MSFS install SimObject index for {install_root} ({len(cached)} model(s) indexed).", "info")
        return cached

    _log(f"Indexing loose SimObject models (jetways/GSE/animated vehicles/doors) under "
         f"{install_root} -- this happens once and is cached for future runs...", "info")

    index = {}
    for guid_hex, title, model_src, xml_path in _iter_simobject_models(install_root, log_callback=log_callback):
        index.setdefault(guid_hex, (str(model_src), title, str(xml_path) if xml_path else None))

    cache_utils.set("msfs_install_simobject_index", index, *key_parts)
    _log(f"Indexed {len(index)} loose SimObject model(s) from {install_root}.", "success")
    return index


def extract_simobjects_from_install_index(guids_needed, install_index, models_dir, log_callback=None):
    """Companion to extract_from_install_index, for the loose-.gltf/.glb
    SimObject index instead of the packed-.bgl ModelData index. Copies
    just the models actually needed (not every stock SimObject in the
    whole install) into models_dir. Returns (guid_map_additions,
    name_map_additions) in the same shape extract()'s other GUID sources
    already use."""
    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    guid_map, name_map = {}, {}

    for guid_hex in guids_needed:
        entry = install_index.get(guid_hex)
        if entry is None:
            continue
        # 2-tuple (no xml path) is a still-valid, older on-disk cache_utils
        # entry from before this index started storing one -- treat as "no
        # behavior XML known" rather than erroring on it.
        if len(entry) == 3:
            model_src_str, title, xml_path_str = entry
        else:
            model_src_str, title = entry
            xml_path_str = None
        xml_path = Path(xml_path_str) if xml_path_str else None
        model_stem = _copy_simobject_model(
            Path(model_src_str), models_dir, log_callback=log_callback, xml_src=xml_path)
        if model_stem is None:
            continue
        guid_map[guid_hex] = model_stem
        name_map[guid_hex] = title

    if guid_map:
        _log(f"Recovered {len(guid_map)} stock MSFS SimObject model(s) (loose .gltf/.glb, e.g. "
             f"animated GSE/jetways/doors) from your MSFS install.", "success")
    return guid_map, name_map


def _parse_one_bgl(bgl_path_str, models_dir_str, scan_terrain_vectors=True):
    """Picklable per-file worker for extract()'s BGL parsing loop -- runs
    inside a ProcessPoolExecutor. Disk-cached: a .bgl file's parse result
    only depends on its own bytes and this module's own parsing logic, not
    on which output directory this particular run happens to be using, so
    a cache hit skips re-parsing the binary AND re-extracting its embedded
    models entirely -- only a handful of small file copies into this run's
    (fresh, per-run) models_dir.

    Cache key includes bgl_extractor.py's own content hash (see
    cache_utils.module_version), so editing this file's parsing logic
    during the upcoming bug-fixing pass automatically invalidates every
    cache entry that depended on the old behavior -- it will never keep
    serving pre-fix results silently. Also includes scan_terrain_vectors,
    so toggling that GUI checkbox between runs can't return a stale
    cached result computed with the OTHER setting."""
    bgl = Path(bgl_path_str)
    models_dir = Path(models_dir_str)

    key_parts = (cache_utils.file_identity(bgl), cache_utils.module_version(__file__), str(scan_terrain_vectors))
    cached = cache_utils.get("bgl_parse", *key_parts)
    if cached is not None:
        result, model_filenames = cached
        src_dir = cache_utils.entry_dir("bgl_parse", *key_parts)
        if all((src_dir / name).exists() for name in model_filenames):
            models_dir.mkdir(parents=True, exist_ok=True)
            extracted = []
            for name in model_filenames:
                dst = models_dir / name
                shutil.copy2(src_dir / name, dst)
                extracted.append(dst)
            result = dict(result)
            result["extracted_models"] = extracted
            result["logs"] = list(result["logs"]) + [(f"      [cache] {bgl.name}: reused previous parse result", "info")]
            return result
        # Cache entry incomplete/pruned -- fall through and reparse.

    result = _parse_one_bgl_uncached(bgl, models_dir, scan_terrain_vectors)

    if result["error"] is None:
        try:
            dest_dir = cache_utils.entry_dir("bgl_parse", *key_parts)
            model_filenames = []
            for p in result["extracted_models"]:
                shutil.copy2(p, dest_dir / p.name)
                model_filenames.append(p.name)
            cacheable = dict(result)
            cacheable["extracted_models"] = []  # paths are run-specific; filenames cached separately
            cache_utils.set("bgl_parse", (cacheable, model_filenames), *key_parts)
        except OSError:
            pass  # Caching is a pure speed optimization -- never fail the parse over it.

    return result


def _parse_one_bgl_uncached(bgl, models_dir, scan_terrain_vectors=True):
    """The actual per-file parse -- see _parse_one_bgl for the caching
    wrapper and why every file independently re-checks for the airport
    reference point (rather than short-circuiting once some other file has
    already found one, like the original serial loop did): it's cheap, and
    it lets extract() reproduce the exact original "first file in listing
    order that has coordinates wins" behavior by simply picking from
    results in order, even though workers/cache hits finish in arbitrary
    order."""
    logs = []

    def _wlog(msg, level="info"):
        logs.append((msg, level))

    result = {
        "bgl_name": bgl.name,
        "extracted_models": [],
        "guid_map": {},
        "name_map": {},
        "airport_sections": [],
        "records": [],
        "skipped_objects": [],
        "exclusions": [],
        "airport_lat": None,
        "airport_lon": None,
        "airport_alt": 0.0,
        "airport_layout": None,
        "terrain_vector_feature_count": 0,
        "terrain_vector_materials": set(),
        "error": None,
    }

    try:
        data = bgl.read_bytes()
        sections = parse_bgl(data)

        airport_secs_this_file = [s for s in sections if s.type_name == "Airport"]

        for sec in airport_secs_this_file:
            for sub in sec.subsections:
                for rec_type, payload in walk_records(data, sub.data_offset, sub.data_size, is_scenery_obj=False):
                    if rec_type in (0x0113, 0x003C) and len(payload) >= 24:
                        lon_val, lat_val, alt_val = struct.unpack_from("<IIi", payload, 12)
                        result["airport_lon"] = decode_lonlat_dword(lon_val, is_lat=False)
                        result["airport_lat"] = decode_lonlat_dword(lat_val, is_lat=True)
                        result["airport_alt"] = alt_val / 1000.0
                        _wlog(f"      Found Airport coordinates for offset base: Lat {result['airport_lat']:.5f}, "
                              f"Lon {result['airport_lon']:.5f}, Alt {result['airport_alt']:.1f}m", "info")
                        layout = airport_layout.decode_airport_layout(payload)
                        if not layout.is_empty():
                            result["airport_layout"] = layout
                            _wlog(f"      Native airport layout: {len(layout.runway_centers)} runways, "
                                  f"{len(layout.aprons)} aprons, {len(layout.painted_lines)} painted lines, "
                                  f"{len(layout.taxi_nodes)} taxi nodes, {len(layout.taxi_edges)} taxi edges, "
                                  f"{len(layout.ramp_starts)} ramp starts", "info")
                        break
                if result["airport_lat"] is not None:
                    break
            if result["airport_lat"] is not None:
                break

        for sec in airport_secs_this_file:
            result["airport_sections"].append((data, sec))

        for sec in [s for s in sections if s.type_name == "ModelData"]:
            for sub in sec.subsections:
                found, sub_guid_map, sub_name_map = extract_models_from_modeldata(
                    data, sub.data_offset, sub.data_size, models_dir, log_callback=_wlog)
                result["extracted_models"].extend(found)
                result["guid_map"].update(sub_guid_map)
                result["name_map"].update(sub_name_map)

        for sec in [s for s in sections if s.type_name == "SceneryObject"]:
            for sub in sec.subsections:
                recs, skipped = walk_scenery_object_records(data, sub.data_offset, sub.data_size)
                for rec_type, payload in recs:
                    result["records"].append((sec.type_name, rec_type, payload, sub.qmid1, sub.qmid2))
                result["skipped_objects"].extend(skipped)

        for sec in [s for s in sections if s.type_name == "Exclusion"]:
            for sub in sec.subsections:
                result["exclusions"].extend(parse_exclusion_rectangles(data, sub.data_offset, sub.data_size))

        if scan_terrain_vectors:
            for sec in [s for s in sections if s.type_name == "TerrainVectorDb"]:
                for sub in sec.subsections:
                    count, materials = scan_terrain_vector_db(data, sub.data_offset, sub.data_size, log_callback=_wlog)
                    result["terrain_vector_feature_count"] += count
                    result["terrain_vector_materials"] |= materials

    except Exception as e:
        result["error"] = str(e)

    result["logs"] = logs
    return result


def _dedupe_placements(placements):
    """CONFIRMED REAL BUG (a real EGLC package): a GUID identifies a
    MODEL/TYPE, shared across every real-world instance of it -- never a
    unique placement. Two entirely different extraction paths (a raw BGL
    SceneryObject record, and the same object ALSO reachable via SPB
    container-attach expansion) can independently produce a placement
    for the exact same real-world instance: same GUID, same position,
    same heading. Left in, this doubles the object in the output --
    visually a duplicate, and z-fighting/glitching for anything draped.
    Keys on (guid, lat, lon, hdg) rounded to a few decimal places (float
    jitter from two different derivations of the same real-world point),
    keeping the FIRST occurrence of each key and dropping the rest.
    Returns (deduped_list, dropped_count)."""
    seen = set()
    out = []
    dropped = 0
    for p in placements:
        guid = p.get("guid")
        lat, lon, hdg = p.get("lat"), p.get("lon"), p.get("hdg")
        if guid is None or lat is None or lon is None or hdg is None:
            out.append(p)
            continue
        key = (guid, round(float(lat), 6), round(float(lon), 6), round(float(hdg), 1))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(p)
    return out, dropped


def extract(target_path: Path, out_dir: Path, log_callback=None, msfs_install_root=None, scan_terrain_vectors=True,
            propdefs_dir=None):
    def _log(msg, level="info"):
        if log_callback: log_callback(msg, level)
        else: logger.info(msg)

    if zstd is None:
        # GLBZ (zstd-compressed) is the normal encoding for a .bgl's own
        # embedded RIFF/GLTF models in virtually every modern MSFS
        # package -- without zstd, everything except already-loose
        # SimObject .glb files silently fails to convert, with no other
        # diagnostic trail pointing at why. requirements.txt lists
        # "zstandard" as a real, non-optional dependency; this is the
        # loud version of that for anyone who skipped it anyway.
        _log(
            "'zstandard' isn't installed (pip install zstandard) -- every embedded model compressed "
            "with GLBZ, which is how virtually all modern MSFS packages store their own .bgl-embedded "
            "geometry (terminal buildings, pavement, markings, ...), will be silently skipped this run. "
            "Only already-loose SimObject .glb files (jetways/GSE/doors, discovered separately) will "
            "still convert. This is not optional for a real package -- install it and re-run.",
            "error")

    if target_path.is_dir():
        bgl_files = glob_ci(target_path, ".bgl")
        spb_files = glob_ci(target_path, ".spb")
    else:
        bgl_files = [target_path] if target_path.suffix.lower() == ".bgl" else []
        spb_files = [target_path] if target_path.suffix.lower() == ".spb" else []

    if not bgl_files and not spb_files:
        return [], {}, [], None, None, None

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    extracted_models = []
    all_records = []
    all_exclusions = []
    all_airport_sections = []
    all_skipped_objects = []
    all_terrain_vector_feature_count = 0
    all_terrain_vector_materials = set()
    guid_map = {}
    name_map = {}

    if target_path.is_dir():
        simobj_guids, simobj_names = discover_simobjects(target_path, out_dir, log_callback=log_callback)
        guid_map.update(simobj_guids)
        name_map.update(simobj_names)

    airport_lat = None
    airport_lon = None
    airport_alt = 0.0
    native_airport_layout = None

    # Each .bgl file's parse is independent, so it's dispatched to a
    # process pool; results are merged back in the original bgl_files
    # order so log ordering, guid_map/name_map merge order, and the
    # "first file in listing order with airport coordinates wins" rule
    # all stay deterministic.
    max_workers = max(1, os.cpu_count() or 1)
    results_by_name = {}
    if bgl_files:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_parse_one_bgl, str(bgl), str(models_dir), scan_terrain_vectors): bgl
                for bgl in bgl_files
            }
            for future in futures:
                bgl = futures[future]
                try:
                    results_by_name[bgl] = future.result()
                except Exception as e:
                    results_by_name[bgl] = {"error": str(e), "logs": []}

    for bgl in bgl_files:
        _log(f"  -> Analyzing: {bgl.name}", "info")
        r = results_by_name.get(bgl)
        if r is None:
            _log(f"Failed parsing {bgl.name}: worker produced no result", "error")
            continue

        for msg, level in r.get("logs", []):
            _log(msg, level)

        if r.get("error"):
            _log(f"Failed parsing {bgl.name}: {r['error']}", "error")
            continue

        extracted_models.extend(r["extracted_models"])
        guid_map.update(r["guid_map"])
        name_map.update(r["name_map"])
        all_airport_sections.extend(r["airport_sections"])
        all_records.extend(r["records"])
        all_skipped_objects.extend(r["skipped_objects"])
        all_exclusions.extend(r["exclusions"])
        all_terrain_vector_feature_count += r.get("terrain_vector_feature_count", 0)
        all_terrain_vector_materials |= r.get("terrain_vector_materials", set())

        if airport_lat is None and r["airport_lat"] is not None:
            airport_lat = r["airport_lat"]
            airport_lon = r["airport_lon"]
            airport_alt = r["airport_alt"]

        if native_airport_layout is None and r.get("airport_layout") is not None:
            native_airport_layout = r["airport_layout"]

    placements = []

    if all_records:
        by_type = defaultdict(list)
        for source, rec_type, payload, qmid1, qmid2 in all_records:
            by_type[(source, rec_type)].append((payload, qmid1, qmid2))

        for (source, rec_type), items in by_type.items():
            len_counts = Counter(len(item[0]) for item in items)

            recs_by_len = defaultdict(list)
            for payload, qmid1, qmid2 in items:
                recs_by_len[len(payload)].append((payload, qmid1, qmid2))

            for rec_len, sub_items in recs_by_len.items():
                recs = [it[0] for it in sub_items]

                if rec_len < 22:
                    continue

                is_title_bucket = False
                no_guid = False
                name_off, name_kind = None, None

                if rec_type in KNOWN_LAYOUTS:
                    layout = KNOWN_LAYOUTS[rec_type]
                    guid_off = layout["guid_off"]
                    pos_off = layout["pos_off"]
                    order = layout["order"]
                    heading_off = layout["heading_off"]
                    heading_type = layout["heading_type"]
                    no_guid = layout.get("no_guid", False)
                    name_off = layout.get("name_off")
                    name_kind = layout.get("name_kind")
                elif looks_like_title_record(recs):
                    is_title_bucket = True
                    guid_off = None
                    pos_off = HEADER_LEN.get(source, 4)
                    order = "lonlat"
                    heading_off, heading_type = calibrate_heading_offset(
                        recs, [range(pos_off, pos_off + 8)])
                else:
                    if len(recs) >= 2:
                        guid_off = calibrate_guid_offset(recs)
                    else:
                        guid_off = None
                    pos_off = HEADER_LEN.get(source, 4)
                    order = "lonlat"
                    claimed = []
                    if guid_off is not None:
                        claimed.append(range(guid_off, guid_off + 16))
                    claimed.append(range(pos_off, pos_off + 8))
                    heading_off, heading_type = calibrate_heading_offset(recs, claimed)

                for rec, qmid1, qmid2 in sub_items:
                    if len(rec) < pos_off + 8: continue

                    # EXTRACT ALTITUDE SAFELY
                    is_agl = True
                    alt = 0.0
                    if len(rec) >= pos_off + 14:
                        a, b, alt_val, flags = struct.unpack_from("<IIiH", rec, pos_off)
                        alt = alt_val / 1000.0
                        is_agl = bool(flags & 0x0001)
                    elif len(rec) >= pos_off + 12:
                        a, b, alt_val = struct.unpack_from("<IIi", rec, pos_off)
                        alt = alt_val / 1000.0
                    elif len(rec) >= pos_off + 8:
                        a, b = struct.unpack_from("<II", rec, pos_off)

                    if order == "lonlat":
                        lon, lat = decode_lonlat_dword(a, False), decode_lonlat_dword(b, True)
                    else:
                        lat, lon = decode_lonlat_dword(a, True), decode_lonlat_dword(b, False)

                    title, livery, guid_hex = None, None, ""

                    if is_title_bucket:
                        names = _extract_trailing_names(rec)
                        if names:
                            title = names[0]
                            if len(names) > 1: livery = names[1]
                            for g, nm in name_map.items():
                                if nm.lower() == title.lower():
                                    guid_hex = g
                                    break
                    elif no_guid:
                        if name_off is not None and name_kind == "stringz" and len(rec) > name_off:
                            raw = rec[name_off:]
                            nul = raw.find(b"\x00")
                            name_bytes = raw[:nul] if nul != -1 else raw
                            if name_bytes and all(b in _ASCII_PRINTABLE for b in name_bytes):
                                title = name_bytes.decode("ascii", errors="replace")
                    else:
                        guid_hex = resolve_guid_hex(rec, guid_off, guid_map)

                    # EXTRACT PITCH AND ROLL
                    pitch, roll, heading = 0.0, 0.0, 0.0
                    if heading_off is not None and len(rec) >= heading_off + 4:
                        try:
                            if heading_type == "bams16":
                                pitch_off = heading_off - 4
                                roll_off = heading_off - 2
                                if pitch_off >= 0 and len(rec) >= pitch_off + 2:
                                    p_val = struct.unpack_from("<H", rec, pitch_off)[0]
                                    pitch = float(p_val * (360.0 / 65536.0))
                                if roll_off >= 0 and len(rec) >= roll_off + 2:
                                    r_val = struct.unpack_from("<H", rec, roll_off)[0]
                                    roll = float(r_val * (360.0 / 65536.0))
                                h_val = struct.unpack_from("<H", rec, heading_off)[0]
                                heading = float(h_val * (360.0 / 65536.0))
                        except Exception:
                            pass

                    placements.append({
                        "guid": guid_hex,
                        "title": title,
                        "livery": livery,
                        "lat": lat,
                        "lon": lon,
                        "alt": alt,
                        "height_offset": placement_height_offset(alt, is_agl, airport_alt),
                        "pitch": pitch,
                        "roll": roll,
                        "hdg": heading,
                        "is_agl": is_agl,
                        "qmid1": qmid1,
                        "qmid2": qmid2,
                        "source": source
                    })

    if all_airport_sections:
        by_file = defaultdict(list)
        for data, sec in all_airport_sections:
            by_file[id(data)].append((data, sec))
        for group in by_file.values():
            data = group[0][0]
            secs = [sec for _, sec in group]
            embedded = extract_airport_embedded_placements(data, secs, guid_map, _log, airport_lat, airport_lon, airport_alt)
            placements.extend(embedded)

    if spb_files:
        _log(f"Found {len(spb_files)} SPB file(s). Attempting in-memory SimPropContainer extraction...", "info")
        # A container GUID is only safe to drop its raw wrapper placement
        # for when its SPB actually produced at least one child placement
        # that resolves to a model (guid_map or a title cross-match) --
        # otherwise the wrapper must survive so it still reaches the
        # unresolved picker.
        _resolvably_expanded = set()
        # Multi-pass: a container whose own real-world anchor is only
        # producible by ANOTHER .spb's attach output (nested attach
        # chains) has no anchor yet on an early pass purely because of
        # file order, not because it's genuinely orphaned -- see
        # extract_spb_placements' own docstring/comment. Retry whatever
        # didn't resolve after every pass that made progress; once a full
        # pass resolves nothing new, anything still pending has no
        # ancestor chain reaching a real placement at all and is dropped
        # (never placed at the airport's reference point as a guess).
        _pending = list(spb_files)
        while _pending:
            _next_pending = []
            _progress = False
            for spb_path in _pending:
                spb_placements = extract_spb_placements(
                    spb_path, airport_lat, airport_lon, airport_alt, placements, _log,
                    name_map=name_map, propdefs_dir=propdefs_dir, allow_fallback=False)
                if spb_placements is None:
                    _next_pending.append(spb_path)
                    continue
                _progress = True
                if spb_placements:
                    cg = spb_placements[0].get("container_guid")
                    if cg and any(c.get("guid") in guid_map for c in spb_placements):
                        _resolvably_expanded.add(cg)
                    placements.extend(spb_placements)
            if not _progress:
                if _next_pending:
                    _log(f"      {len(_next_pending)} SPB file(s) never found their own container's "
                         f"real-world placement (not a raw BGL placement, and no chain through "
                         f"another .spb resolved it either) -- their content has no legitimate "
                         f"position and was dropped rather than guessed at the airport's reference "
                         f"point: {', '.join(p.name for p in _next_pending[:8])}"
                         f"{', ...' if len(_next_pending) > 8 else ''}", "warning")
                break
            _pending = _next_pending
        if _resolvably_expanded:
            _before = len(placements)
            placements = [
                p for p in placements
                if not (
                    p.get("guid") in _resolvably_expanded
                    and p.get("container_guid") is None      # not an SPB child itself
                    and p.get("guid") not in guid_map        # not independently resolvable
                )
            ]
            _dropped = _before - len(placements)
            if _dropped:
                _log(f"      Removed {_dropped} raw SimPropContainer wrapper placement(s) -- their "
                     f"contained model(s) were expanded and resolved above.", "info")

    # AttachedObjects (a door / light / effect hung off a parent SceneryObject
    # with a bias offset) were parsed for name + parent position but never
    # placed -- the bias isn't decoded. Rather than drop them entirely, place
    # each one AT ITS PARENT (approximate but on the right building) whenever
    # its model resolves, either directly by instance GUID or by name via the
    # SimObject title map. This recovers e.g. LHBP's 53 terminal entrance
    # doors, which are all AttachedObjects on the terminal shell.
    _title_to_guid = {}
    for _g, _t in name_map.items():
        if _t:
            _title_to_guid.setdefault(_t.strip().lower(), _g)
    _att_placed = 0
    for s in all_skipped_objects:
        if s.get("reason") != "attached-object" or s.get("anchor_lat") is None:
            continue
        _g = (s.get("instance_guid") or "").strip().strip("{}").lower()
        if _g not in guid_map:
            _g = _title_to_guid.get((s.get("name") or "").strip().lower())
        if not _g or _g not in guid_map:
            continue
        placements.append({
            "guid": _g, "title": s.get("name"), "livery": None,
            "lat": s["anchor_lat"], "lon": s["anchor_lon"], "alt": 0.0,
            "height_offset": 0.0, "pitch": 0.0, "roll": 0.0,
            "hdg": s.get("anchor_hdg", 0.0) or 0.0, "is_agl": True,
            "qmid1": 0, "qmid2": 0, "source": "AttachedObject@parent",
        })
        _att_placed += 1
    if _att_placed:
        _log(f"      Placed {_att_placed} AttachedObject(s) (doors / lights / effects hung off a "
             f"parent) at their parent object's position -- bias offset not decoded, so approximate.",
             "info")

    # RESTORED: Phase 3 Reporting orphaned models
    placed_guids = {p["guid"] for p in placements}
    unmatched_guids = []

    for guid, file_stem in guid_map.items():
        if guid not in placed_guids:
            unmatched_guids.append(guid)

    if unmatched_guids:
        _log(f"Found {len(unmatched_guids)} orphaned model(s) with no placement.", "info")
        _log(f"      Full GUID dump of unplaced models (cross-check these against ModelConverterX):", "info")
        for guid in unmatched_guids:
            file_stem = guid_map.get(guid, guid)
            try:
                display_guid = format_guid(bytes.fromhex(guid))
            except (ValueError, struct.error):
                display_guid = f"<unparsable raw hex: {guid}>"
            _log(f"        - {file_stem}  =>  {{{display_guid}}}", "info")

    # RESTORED: Logging skipped taxiway signs and attached objects
    if all_skipped_objects:
        attached = [s for s in all_skipped_objects if s["reason"] == "attached-object"]
        signs = [s for s in all_skipped_objects if s["reason"] == "taxiway-sign-array"]

        if attached:
            named = [s["name"] for s in attached if s["name"]]
            type_name = SCENERY_OBJECT_TYPE_NAMES
            _log(f"      {len(attached)} AttachedObject(s) (beacons/effects/library "
                 f"objects hung off a parent object) were NOT placed -- their position "
                 f"is a bias relative to the parent, not an absolute coordinate, which "
                 f"this script does not yet resolve.", "warning")
            for name in named[:25]:
                _log(f"        - {name}", "info")
            if len(named) > 25:
                _log(f"        ... and {len(named) - 25} more", "info")
            unnamed = len(attached) - len(named)
            if unnamed:
                _log(f"        ({unnamed} attached object(s) had no readable name)", "info")

        if signs:
            total_signs = sum(s["num_signs"] or 0 for s in signs)
            _log(f"      {len(signs)} TaxiwaySign record(s) ({total_signs} individual "
                 f"sign(s) total) were NOT placed -- they're a per-airport array of "
                 f"label + offset-from-anchor entries with no model/GUID, not a single "
                 f"placement.", "info")

    # ASOBO/stock-library objects: some placements reference a GUID never
    # defined anywhere in this package's own BGLs -- one of MSFS's own
    # base-sim assets (generic GSE, lights, jetways). If a local MSFS
    # install is configured, resolve just those still-unmatched GUIDs
    # from its own base .bgl files -- same container format/parsing code,
    # different (optional) search root.
    if msfs_install_root:
        still_unmatched = {p["guid"] for p in placements if p["guid"] and p["guid"] not in guid_map}
        if still_unmatched:
            _log(f"{len(still_unmatched)} unique GUID(s) have placements but no model in this "
                 f"package -- checking your configured MSFS install for stock objects...", "info")
            # Two independent sources, same reason a package itself has two
            # (extract_models_from_modeldata + discover_simobjects above):
            # MSFS ships some stock objects packed inside a .bgl's ModelData
            # section, others as loose .gltf/.glb SimObject files -- a given
            # GUID is only ever defined in ONE of the two, never both, so
            # both get checked and merged the same way.
            try:
                install_index = build_install_guid_index(msfs_install_root, log_callback=log_callback)
                install_guids, install_names = extract_from_install_index(
                    still_unmatched, install_index, models_dir, log_callback=log_callback)
                # Package-local geometry (already in guid_map) always wins
                # on a conflict -- it's more likely a repaint/livery variant
                # specific to this package, not the generic stock asset.
                for guid_hex, file_stem in install_guids.items():
                    guid_map.setdefault(guid_hex, file_stem)
                for guid_hex, name in install_names.items():
                    name_map.setdefault(guid_hex, name)
            except Exception as e:
                _log(f"Failed searching MSFS install's ModelData sections for stock objects: {e}", "warning")

            still_unmatched = {p["guid"] for p in placements if p["guid"] and p["guid"] not in guid_map}
            if still_unmatched:
                try:
                    simobject_index = build_install_simobject_index(msfs_install_root, log_callback=log_callback)
                    simobject_guids, simobject_names = extract_simobjects_from_install_index(
                        still_unmatched, simobject_index, models_dir, log_callback=log_callback)
                    for guid_hex, file_stem in simobject_guids.items():
                        guid_map.setdefault(guid_hex, file_stem)
                    for guid_hex, name in simobject_names.items():
                        name_map.setdefault(guid_hex, name)
                except Exception as e:
                    _log(f"Failed searching MSFS install's SimObjects for stock objects: {e}", "warning")

    placements, _dupes_dropped = _dedupe_placements(placements)
    if _dupes_dropped:
        _log(f"      Dropped {_dupes_dropped} duplicate placement(s) -- same model GUID at the same "
             f"real-world position/heading, reached through more than one extraction path.", "warning")

    matched_placements = sum(1 for p in placements if p["guid"] in guid_map)
    title_placements = [p for p in placements if p.get("title")]
    title_resolved = sum(1 for p in title_placements if p["guid"] in guid_map)
    airport_embedded = [p for p in placements if p.get("source") == "Airport-Embedded"]

    _log(f"Extraction complete! Found {len(placements)} placements total.", "success")
    _log(f"BGL extraction complete. {len(guid_map)} models indexed by GUID; "
         f"{matched_placements}/{len(placements)} placements have a matching model.", "success")

    if title_placements:
        _log(f"      {len(title_placements)} placements were name/title-based "
             f"(SimObject references, e.g. GSE/vehicles/people); "
             f"{title_resolved}/{len(title_placements)} were cross-matched to a "
             f"locally-extracted model by name.", "info")

    if airport_embedded:
        _log(f"      {len(airport_embedded)} placements came from LibraryObject "
             f"records embedded INSIDE Airport section blobs (Jetway / Included "
             f"Tower Scenery Object) rather than the flat SceneryObject section.", "info")

    # RESTORED: Phase 4 Output formatted GUIDs to JSON
    if guid_map:
        json_path = out_dir / "extracted_guids.json"
        formatted_guid_map = {}
        for raw_hex, file_stem in guid_map.items():
            try:
                display_guid = "{" + format_guid(bytes.fromhex(raw_hex)).upper() + "}"
                formatted_guid_map[display_guid] = file_stem
            except Exception:
                formatted_guid_map[raw_hex] = file_stem

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_guid_map, f, indent=4)
            _log(f"      Saved {len(formatted_guid_map)} formatted model GUIDs to {json_path.name} for MCX cross-checking.", "info")
        except Exception as e:
            _log(f"      Failed to save extracted_guids.json: {e}", "warning")

    # TerrainVectorDb (MSFS World Editor's own vector-polygon/vegetation
    # database, BGL section 0x65) -- some packages store their entire
    # taxiway/runway/apron pavement here instead of as placed glTF
    # models. Deliberately not converted into geometry: the per-vertex
    # boundary coordinates use an encoding that resisted reverse-
    # engineering, and fabricating an approximate shape risks something
    # worse than an honest gap (e.g. pavement texture over open water for
    # a dockland airport). Surfaced loudly here so a missing-pavement
    # report for this class of package isn't a silent mystery.
    if all_terrain_vector_feature_count:
        material_names = find_material_libraries(target_path, log_callback=log_callback) if target_path.is_dir() else {}
        resolved = sorted({material_names.get(g, format_guid(g)) for g in all_terrain_vector_materials})
        materials_note = f" referencing material(s): {', '.join(resolved)}" if resolved else ""
        _log(f"      [TerrainVectorDb] Found ~{all_terrain_vector_feature_count} custom ground-polygon/vegetation "
             f"feature(s){materials_note}. This package's own vector-polygon ground system isn't converted to "
             f"geometry yet (see convert log docs) -- its pavement/ground-marking shapes will NOT appear in the "
             f"output. Everything else (buildings, GSE, signs, lights, ordinary models) is unaffected.", "warning")

    return placements, guid_map, all_exclusions, airport_lat, airport_lon, native_airport_layout
