"""Native MSFS airport-layout extraction.

Decodes an MSFS BGL "Airport" record and its sub-records -- runways (with
thresholds, VASI/PAPI and approach lights), helipads, COM frequencies,
taxi points, taxi paths, taxiway names, parking stands, aprons, light
strings, painted lines and taxiway signs -- into an AirportLayout that
apt_dat.py turns into a complete apt.dat airport.

Where each record's fields live:

- The airport record itself is a fixed head followed by a chain of
  sub-records (u16 id, u32 size including the 6-byte header). The head
  grew with every simulator generation: 0x003C/0x0056 (MSFS 2020) keep
  their sub-records from 0x44, 0x0113 (MSFS 2024) from 0x5C (24 more
  bytes, the ident moved to a 64-bit field at 0x4C). Rather than trusting
  one offset per record id, _find_subrecord_start tries every plausible
  head length and keeps the one whose chain is made of known sub-record
  ids and ends exactly at the end of the record -- a wrong offset breaks
  the chain almost immediately. (The previous decoder hard-coded 0x5C,
  found on one MSFS 2024 package, so every MSFS 2020 airport silently
  decoded as empty.)
- Packed arrays (taxi points, names, paths, parking) are a u16 count at
  6 and elements from 8. The element size is derived from the record size
  where possible, which absorbs the fields MSFS 2024 appended.
- A taxi path element holds BOTH its start and end node: start is the
  u16 at element offset 0, end is the u16 at offset 46 of the 48-byte
  MSFS element (older formats pack it into the low 12 bits of the word at
  offset 2). A parking path's end numbers the stand, not a taxi point.
- A parking element is variable length: 4 + 4 + 4 + 16 bytes of flags,
  radius, heading and tee offsets, the position, then 4 bytes per airline
  code and a fixed trailer, so the trailer length is found by trying the
  known ones and keeping the one that consumes the record exactly.
- Records whose layout differs between versions in ways nobody has
  documented (aprons, painted lines, signs) are anchored on their
  coordinate payload: eight bytes that decode to a point within ~20 km of
  the airport reference point almost never happen by accident.
"""

import math
import struct
from dataclasses import dataclass, field as _dc_field

from geo_transform import decode_lonlat_dword

# --- record ids -------------------------------------------------------------

AIRPORT_RECORD_IDS = (0x003C, 0x0056, 0x0113)
_REC_AIRPORT_2024 = 0x0113

_AP_NAME = 0x0019
_AP_TOWER_OBJ = 0x0066
_AP_RUNWAYS = (0x0004, 0x003E, 0x00CE)
_AP_RUNWAY_MSFS = 0x00CE
_AP_HELIPAD = 0x0026
_AP_START = 0x0011
_AP_COM = 0x0012
_AP_APRONS = (0x0037, 0x00AF, 0x00D3, 0x00D0)
_AP_APRON_MSFS = (0x00D3, 0x00D0)
_AP_LIGHT_STRING = 0x0031
_AP_TAXI_POINTS = (0x001A, 0x00AC)
_AP_TAXI_POINT_P3D5 = 0x00AC
_AP_TAXI_NAME = 0x001D
_AP_TAXI_PATHS = (0x001C, 0x0040, 0x00AE, 0x00D4)
_AP_TAXI_PATH_MSFS = 0x00D4
_AP_PARKINGS = (0x003D, 0x00AD, 0x00E7, 0x001B)
_AP_PARKING_MSFS = 0x00E7
_AP_PARKING_P3D5 = 0x00AD
_AP_PARKING_FS9 = 0x001B
_AP_PAINTED_LINE = 0x00CF
_AP_SIGN = 0x00D9

# Every airport sub-record id seen in FSX/P3D/MSFS files; used only to
# judge which head length makes the sub-record chain parse cleanly.
_KNOWN_AIRPORT_SUBRECORDS = frozenset({
    0x0019, 0x0066, 0x0004, 0x003E, 0x00CE, 0x0022, 0x0026, 0x0011, 0x0012,
    0x0033, 0x00DB, 0x0037, 0x00AF, 0x00D3, 0x00D0, 0x0030, 0x0041, 0x00B0,
    0x0031, 0x001A, 0x00AC, 0x003D, 0x00AD, 0x00E7, 0x001B, 0x001C, 0x0040,
    0x00AE, 0x00D4, 0x001D, 0x003A, 0x0024, 0x00FA, 0x0038, 0x0039, 0x003B,
    0x0042, 0x0048, 0x0057, 0x0058, 0x0059, 0x005A, 0x005B, 0x00CD, 0x00CF,
    0x00D8, 0x00D9, 0x00DD, 0x00DE, 0x00E8, 0x00E9, 0x005C, 0x005D, 0x006A,
    0x00FB, 0x00FF, 0x0102,
})
_KNOWN_RUNWAY_SUBRECORDS = frozenset({
    0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x000A, 0x0065, 0x0066,
    0x000B, 0x000C, 0x000D, 0x000E, 0x000F, 0x0010, 0x00DF, 0x00E0,
    0x003E, 0x00CB,
})

# Head lengths (bytes from the start of the record, 6-byte header included)
# of the airport record across generations: FS9, FSX, P3Dv5, MSFS 2020;
# MSFS 2024 is 24 bytes longer than MSFS 2020 and future builds may add more.
_AIRPORT_HEAD_CANDIDATES = (0x44, 0x5C, 0x38, 0x40, 0x34) + tuple(0x44 + 4 * k for k in range(1, 9))
# Runway: FSX 46-byte body, P3D +16, MSFS +44 (and possibly more later).
_RUNWAY_HEAD_CANDIDATES = (6 + 46, 6 + 62, 6 + 90) + tuple(6 + 90 + 4 * k for k in range(1, 17))

_TAXI_POINT_SIZE = 12
_TAXI_POINT_SIZE_P3D5 = 16
_TAXI_NAME_SIZE = 8
_TAXI_PATH_SIZE = {0x001C: 20, 0x0040: 36, 0x00AE: 40, 0x00D4: 48}
_TAXI_PATH_END_OFFSET_MSFS = 46

# Within this many degrees of the airport reference point a decoded
# coordinate is taken as genuine (~22 km -- larger than any airport).
_NEAR_DEG = 0.2

# --- data ------------------------------------------------------------------


@dataclass
class Polygon:
    vertices: list  # [(lat, lon), ...]


@dataclass
class LightString:
    vertices: list  # [(lat, lon), ...]
    name: str = ""  # the MSFS light preset name, which says its colour/use

    @property
    def light_type(self):
        return light_code_for_preset(self.name)


@dataclass
class PaintedLine:
    vertices: list
    style: int = 0  # MSFS style index (the record stores style << 1 | lit)
    lit: bool = False


@dataclass
class Vasi:
    kind: int  # 1-6 VASI variants, 7 PAPI-2, 8 PAPI-4, 9 tri-colour, 10 pulsating, 11 T-VASI, 12 ball, 13 APAP
    side: str  # "L" or "R" of the runway, seen on approach
    pitch: float  # glide path, degrees
    bias_x: float = 0.0  # metres off the centreline
    bias_z: float = 0.0  # metres along the runway from the threshold


@dataclass
class RunwayEnd:
    number: int
    designator: int
    displaced_m: float = 0.0
    blast_pad_m: float = 0.0
    overrun_m: float = 0.0
    vasi: list = _dc_field(default_factory=list)  # [Vasi, ...]
    approach_system: int = 0
    reil: bool = False
    touchdown_lights: bool = False

    @property
    def name(self):
        return runway_name(self.number, self.designator)


@dataclass
class Runway:
    lat: float
    lon: float
    alt_m: float
    length_m: float
    width_m: float
    heading_true: float
    surface: int
    marking_flags: int
    light_flags: int
    pattern_flags: int
    primary: RunwayEnd
    secondary: RunwayEnd

    @property
    def edge_lights(self):
        return self.light_flags & 0x3

    @property
    def centre_lights(self):
        return (self.light_flags >> 2) & 0x3


@dataclass
class Helipad:
    lat: float
    lon: float
    heading: float
    length_m: float
    width_m: float
    surface: int
    transparent: bool = False
    closed: bool = False


@dataclass
class Com:
    kind: int
    freq_khz: int
    name: str


@dataclass
class TaxiPoint:
    lat: float
    lon: float
    kind: int  # 2 hold short, 4 ILS hold short, 5/6 the same without a painted bar
    orientation: int = 0

    @property
    def is_hold_short(self):
        return self.kind in (2, 4, 5, 6)

    @property
    def is_ils_hold(self):
        return self.kind in (4, 6)


@dataclass
class TaxiPath:
    start: int
    end: int  # a parking path's end numbers the stand (see AirportLayout.node_index_of_path_end)
    kind: int  # 1 taxi, 2 runway, 3 parking, 4 path, 5 closed, 6 vehicle, 7 road
    name: str
    width_m: float
    centre_line: bool = False
    centre_lit: bool = False
    left_edge_lit: bool = False
    right_edge_lit: bool = False

    @property
    def is_aircraft_route(self):
        return self.kind in (1, 2, 3, 4)

    @property
    def is_vehicle_route(self):
        return self.kind in (6, 7)


@dataclass
class Parking:
    lat: float
    lon: float
    heading: float
    radius_m: float
    kind: int
    name_code: int
    number: int
    suffix: int = 0
    airlines: list = _dc_field(default_factory=list)

    @property
    def display_name(self):
        return parking_display_name(self.name_code, self.number, self.suffix)


@dataclass
class Sign:
    lat: float
    lon: float
    heading: float
    size: int
    label: str


@dataclass
class AirportLayout:
    ident: str = ""
    name: str = ""
    region: str = ""
    lat: float = None
    lon: float = None
    alt_m: float = 0.0
    tower: tuple = None  # (lat, lon, alt_m) or None
    runways: list = _dc_field(default_factory=list)
    helipads: list = _dc_field(default_factory=list)
    coms: list = _dc_field(default_factory=list)
    taxi_points: list = _dc_field(default_factory=list)
    taxi_paths: list = _dc_field(default_factory=list)
    parkings: list = _dc_field(default_factory=list)
    aprons: list = _dc_field(default_factory=list)  # [Polygon, ...]
    painted_lines: list = _dc_field(default_factory=list)
    light_strings: list = _dc_field(default_factory=list)
    signs: list = _dc_field(default_factory=list)
    windsocks: list = _dc_field(default_factory=list)  # [(lat, lon), ...] -- filled by bgl_extractor
    warnings: list = _dc_field(default_factory=list)

    def is_empty(self):
        return not (self.runways or self.helipads or self.aprons or self.taxi_points)

    # --- the network as one node list --------------------------------------
    # Taxi points first, then one node per parking stand: a parking path's
    # end numbers its stand from zero, so it lands at len(taxi_points) + n.

    @property
    def taxi_nodes(self):
        """[(lat, lon), ...] -- taxi points, then parking stands."""
        return [(p.lat, p.lon) for p in self.taxi_points] + [(p.lat, p.lon) for p in self.parkings]

    def path_end_node(self, path):
        return len(self.taxi_points) + path.end if path.kind == 3 else path.end

    @property
    def taxi_edges(self):
        """[(node_a, node_b), ...] of every aircraft route, into taxi_nodes."""
        n = len(self.taxi_points) + len(self.parkings)
        out = []
        for p in self.taxi_paths:
            a, b = p.start, self.path_end_node(p)
            if p.is_aircraft_route and a != b and a < n and b < n:
                out.append((a, b))
        return out

    @property
    def runway_centers(self):
        return [(r.lat, r.lon) for r in self.runways]

    @property
    def ramp_starts(self):
        return [(p.lat, p.lon) for p in self.parkings if is_aircraft_stand(p.kind)]


# --- small decoders -----------------------------------------------------------


def _u8(b, at):
    return b[at] if at < len(b) else 0


def _u16(b, at):
    return struct.unpack_from("<H", b, at)[0] if at + 2 <= len(b) else 0


def _u32(b, at):
    return struct.unpack_from("<I", b, at)[0] if at + 4 <= len(b) else 0


def _i32(b, at):
    return struct.unpack_from("<i", b, at)[0] if at + 4 <= len(b) else 0


def _f32(b, at):
    if at + 4 > len(b):
        return 0.0
    v = struct.unpack_from("<f", b, at)[0]
    return v if math.isfinite(v) else 0.0


def _pair(b, at):
    """(lat, lon) of the lon/lat dword pair at `at`."""
    lon_raw, lat_raw = struct.unpack_from("<II", b, at)
    return decode_lonlat_dword(lat_raw, is_lat=True), decode_lonlat_dword(lon_raw, is_lat=False)


def _string(b):
    return b.split(b"\x00", 1)[0].decode("latin-1", errors="replace").strip()


def _base38(value, max_chars):
    out = []
    while value > 0 and len(out) < max_chars:
        c = value % 38
        value //= 38
        if 2 <= c <= 11:
            out.append(chr(ord("0") + c - 2))
        elif c >= 12:
            out.append(chr(ord("A") + c - 12))
        else:
            break
    return "".join(reversed(out))


def decode_ident(raw, shifted=True):
    """A packed base-38 identifier (ICAO code, region) from its u32."""
    return _base38(raw >> 5 if shifted else raw, 5)


def decode_ident64(raw):
    """MSFS 2024's 64-bit identifier (low six bits are flags)."""
    return _base38(raw >> 6, 8)


_DESIGNATORS = {1: "L", 2: "R", 3: "C", 4: "W", 5: "A", 6: "B"}
_COMPASS = {37: "N", 38: "NE", 39: "E", 40: "SE", 41: "S", 42: "SW", 43: "W", 44: "NW"}


def runway_name(number, designator):
    if number in _COMPASS:
        return _COMPASS[number]
    return f"{number:02d}{_DESIGNATORS.get(designator, '')}"


def is_aircraft_stand(kind):
    return kind not in (0x0C, 0x0D)  # fuel, vehicle


def parking_display_name(name_code, number, suffix_code):
    """The stand name MSFS shows: "A12", "Gate 4", "Parking 7B"."""
    def letter(code):
        return chr(ord("A") + code - 0x0C) if 0x0C <= code <= 0x25 else ""
    suffix = letter(suffix_code)
    if letter(name_code):
        return f"{letter(name_code)}{number}{suffix}"
    prefix = {0x01: "Parking", 0x02: "N Parking", 0x03: "NE Parking", 0x04: "E Parking",
              0x05: "SE Parking", 0x06: "S Parking", 0x07: "SW Parking", 0x08: "W Parking",
              0x09: "NW Parking", 0x0A: "Gate", 0x0B: "Dock"}.get(name_code, "")
    return f"{prefix} {number}{suffix}" if prefix else f"{number}{suffix}"


def light_code_for_preset(name):
    """apt.dat light code (101 green centre, 102 blue edge, 103 amber hold,
    105 ILS hold / lead-off, 106 red stop bar) for an MSFS light-string
    preset. Sceneries put every kind of taxiway light in the "apron edge
    lights" record and name the preset after its colour and use; an
    unnamed one is the SDK's blue edge light."""
    words = [w for w in "".join(c if c.isalnum() else " " for c in name.lower()).split() if w]

    def has(*prefixes):
        return any(w.startswith(p) for w in words for p in prefixes)

    if has("blue"):
        return 102
    if has("red", "stop"):
        return 106
    if has("ils"):
        return 105
    if has("hold"):
        return 103
    if has("orange", "amber", "yellow", "exit", "leadoff"):
        return 105
    if has("green", "cent", "taxi"):
        return 101
    return 102


# --- record walking -----------------------------------------------------------


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


def _chain_score(blob, start, known):
    """(all ids known, ends exactly at the record end, record count) for
    the sub-record chain starting at `start`, or None if nothing parses."""
    if start > len(blob):
        return None
    if start == len(blob):
        return (True, True, 0)
    pos, count, clean = start, 0, True
    while pos + 6 <= len(blob):
        rec_type, rec_size = struct.unpack_from("<HI", blob, pos)
        if rec_size < 6 or pos + rec_size > len(blob):
            break
        clean = clean and rec_type in known
        count += 1
        pos += rec_size
    if count == 0:
        return None
    return (clean, pos == len(blob), count)


def _find_subrecord_start(blob, candidates, known):
    best, best_score = None, None
    for c in candidates:
        score = _chain_score(blob, c, known)
        if score is not None and (best_score is None or score > best_score):
            best, best_score = c, score
    return best


def _element_size(rec, count, minimum):
    """Element size of a packed array record (count at 6, elements from 8),
    derived from the record size when it divides evenly."""
    available = len(rec) - 8
    if count > 0 and available % count == 0 and available // count >= minimum:
        return available // count
    return minimum


class _Near:
    def __init__(self, lat, lon):
        self.lat, self.lon = lat, lon
        self.lon_tol = _NEAR_DEG / max(abs(math.cos(math.radians(lat))), 0.05)

    def accepts(self, lat, lon):
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return False
        dlon = abs(lon - self.lon)
        dlon = min(dlon, 360.0 - dlon)
        return abs(lat - self.lat) <= _NEAR_DEG and dlon <= self.lon_tol

    def run_length(self, b, at):
        n = 0
        while at + 8 <= len(b) and self.accepts(*_pair(b, at)):
            n += 1
            at += 8
        return n

    def longest_run(self, b, min_count, start=6):
        best = None
        at = start
        while at + 8 <= len(b):
            n = self.run_length(b, at)
            if n >= min_count and (best is None or n > best[1]):
                best = (at, n)
            at += 8 * n if n else 1
        return best


def _vertices(b, at, count):
    return [_pair(b, at + 8 * i) for i in range(count) if at + 8 * i + 8 <= len(b)]


# --- sub-record parsers --------------------------------------------------------


def _parse_runway(rec, msfs):
    surface = _u16(rec, 6) & 0x7F
    primary = RunwayEnd(_u8(rec, 8), _u8(rec, 9) & 0x0F)
    secondary = RunwayEnd(_u8(rec, 10), _u8(rec, 11) & 0x0F)
    lat, lon = _pair(rec, 20)
    rw = Runway(
        lat=lat, lon=lon, alt_m=_i32(rec, 28) / 1000.0,
        length_m=_f32(rec, 32), width_m=_f32(rec, 36), heading_true=_f32(rec, 40) % 360.0,
        surface=surface, marking_flags=_u16(rec, 48), light_flags=_u8(rec, 50), pattern_flags=_u8(rec, 51),
        primary=primary, secondary=secondary,
    )
    start = _find_subrecord_start(rec, _RUNWAY_HEAD_CANDIDATES, _KNOWN_RUNWAY_SUBRECORDS)
    if start is None:
        return rw

    def ext_length(sub, with_material):
        at = 6 + 2 + (16 if with_material else 0)
        v = _f32(sub, at)
        return v if v > 0 else 0.0

    for sub_type, sub in _walk_records(rec, start):
        if sub_type in (0x0005, 0x0006):
            end = primary if sub_type == 0x0005 else secondary
            end.displaced_m = ext_length(sub, msfs)
        elif sub_type in (0x0007, 0x0008):
            end = primary if sub_type == 0x0007 else secondary
            end.blast_pad_m = ext_length(sub, msfs)
        elif sub_type in (0x0009, 0x000A):
            end = primary if sub_type == 0x0009 else secondary
            end.overrun_m = ext_length(sub, msfs)
        elif sub_type in (0x0065, 0x0066):  # MSFS overruns carry no material GUID
            end = primary if sub_type == 0x0065 else secondary
            end.overrun_m = ext_length(sub, False)
        elif sub_type in (0x000B, 0x000C, 0x000D, 0x000E):
            end = primary if sub_type in (0x000B, 0x000C) else secondary
            side = "L" if sub_type in (0x000B, 0x000D) else "R"
            bias_x, bias_z, pitch = _f32(sub, 8), _f32(sub, 12), _f32(sub, 20)
            end.vasi.append(Vasi(_u16(sub, 6), side, pitch if pitch > 0.1 else 3.0, bias_x, bias_z))
        elif sub_type in (0x000F, 0x0010, 0x00DF, 0x00E0):
            end = primary if sub_type in (0x000F, 0x00DF) else secondary
            flags = _u8(sub, 6)
            end.approach_system = flags & 0x1F
            end.reil = bool(flags & 0x40)
            end.touchdown_lights = bool(flags & 0x80)
    return rw


def _parse_helipad(rec):
    flags = _u8(rec, 7)
    lat, lon = _pair(rec, 12)
    return Helipad(lat=lat, lon=lon, heading=_f32(rec, 32), length_m=_f32(rec, 24), width_m=_f32(rec, 28),
                   surface=_u8(rec, 6) & 0x7F, transparent=bool(flags & 0x10), closed=bool(flags & 0x20))


def _parse_com(rec):
    name = _string(rec[12:12 + 0x30])
    name = "".join(c for c in name if c.isalnum() or c in " -/.").strip()
    return Com(kind=_u16(rec, 6) & 0xFF, freq_khz=_u32(rec, 8) // 1000, name=name)


def _parse_taxi_points(rec):
    count = _u16(rec, 6)
    size = _element_size(rec, count, _TAXI_POINT_SIZE_P3D5 if _u16(rec, 0) == _AP_TAXI_POINT_P3D5
                         else _TAXI_POINT_SIZE)
    out = []
    for i in range(count):
        at = 8 + i * size
        if at + _TAXI_POINT_SIZE > len(rec):
            break
        lat, lon = _pair(rec, at + 4)
        out.append(TaxiPoint(lat=lat, lon=lon, kind=rec[at], orientation=rec[at + 1]))
    return out


def _parse_taxi_names(rec):
    count = _u16(rec, 6)
    size = _element_size(rec, count, _TAXI_NAME_SIZE)
    return [_string(rec[8 + i * size:8 + i * size + _TAXI_NAME_SIZE]) for i in range(count)
            if 8 + i * size + _TAXI_NAME_SIZE <= len(rec)]


def _parse_taxi_paths(rec):
    rec_id = _u16(rec, 0)
    count = _u16(rec, 6)
    size = _element_size(rec, count, _TAXI_PATH_SIZE.get(rec_id, 20))
    msfs = rec_id == _AP_TAXI_PATH_MSFS
    out = []
    for i in range(count):
        at = 8 + i * size
        if at + 20 > len(rec):
            break
        flags = _u16(rec, at + 2)
        end = flags & 0x0FFF
        if msfs and at + _TAXI_PATH_END_OFFSET_MSFS + 2 <= len(rec):
            end = _u16(rec, at + _TAXI_PATH_END_OFFSET_MSFS)
        type_byte = rec[at + 4]
        edge_byte = rec[at + 6]
        out.append(dict(
            start=_u16(rec, at), end=end, kind=type_byte & 0x0F,
            runway_designator=(flags >> 12) & 0x0F, name_index=rec[at + 5],
            centre_line=bool(edge_byte & 1), centre_lit=bool(edge_byte & 2),
            left_edge_lit=bool(edge_byte & 0x10), right_edge_lit=bool(edge_byte & 0x80),
            width_m=_f32(rec, at + 8),
        ))
    return out


def _try_parkings(rec, count, has_tees, trailer):
    out = []
    at = 8
    for _ in range(count):
        if at + 12 > len(rec):
            return None
        flags = _u32(rec, at)
        n_airlines = (flags >> 24) & 0xFF
        radius, heading = _f32(rec, at + 4), _f32(rec, at + 8)
        at += 12 + (16 if has_tees else 0)
        if at + 8 > len(rec):
            return None
        lat, lon = _pair(rec, at)
        at += 8
        if at + 4 * n_airlines + trailer > len(rec):
            return None
        airlines = [_string(rec[at + 4 * k:at + 4 * k + 4]) for k in range(n_airlines)]
        at += 4 * n_airlines
        suffix = rec[at + 1] if trailer >= 2 else 0
        at += trailer
        out.append(Parking(lat=lat, lon=lon, heading=heading % 360.0, radius_m=radius,
                           kind=(flags >> 8) & 0xF, name_code=flags & 0x3F, number=(flags >> 12) & 0xFFF,
                           suffix=suffix, airlines=[a for a in airlines if a]))
    return out, at


def _parse_parkings(rec, warnings):
    rec_id = _u16(rec, 0)
    count = _u16(rec, 6)
    if rec_id == _AP_PARKING_MSFS:
        trailers = (20, 24, 28, 32, 16, 36, 40, 0)
    elif rec_id == _AP_PARKING_P3D5:
        trailers = (4, 0)
    else:
        trailers = (0, 4)
    has_tees = rec_id != _AP_PARKING_FS9
    fallback = None
    for trailer in trailers:
        res = _try_parkings(rec, count, has_tees, trailer)
        if res is None:
            continue
        stands, end = res
        if end == len(rec):
            return stands
        fallback = fallback or stands
    warnings.append(f"parking record 0x{rec_id:04X}: no known element layout consumes it exactly")
    return fallback or []


def _parse_apron(rec, near):
    """Boundary of an apron. The validated MSFS layout keeps the vertex
    count at 0x30 and the vertices from 0x34; anything else is found by its
    coordinate run."""
    rec_id = _u16(rec, 0)
    if rec_id in _AP_APRON_MSFS and len(rec) >= 0x34:
        nv = _u16(rec, 0x30)
        if nv >= 3 and 0x34 + 8 * nv <= len(rec):
            verts = _vertices(rec, 0x34, nv)
            if near is None or all(near.accepts(*v) for v in verts):
                return verts
    if near is None:
        return []
    found = near.longest_run(rec, 3)
    return _vertices(rec, found[0], found[1]) if found else []


def _parse_light_string(rec, near):
    """Light string: vertex count at 8, preset-name length at 10, vertices
    from 24, then the MSFS light preset name."""
    count, name_len = _u16(rec, 8), _u16(rec, 10)
    end = 24 + 8 * count
    if count >= 2 and end + name_len <= len(rec) and (near is None or near.run_length(rec, 24) >= count):
        return LightString(_vertices(rec, 24, count), _string(rec[end:end + name_len]))
    if count >= 1 and end <= len(rec) and near is None:
        return LightString(_vertices(rec, 24, count))
    found = near.longest_run(rec, 2) if near is not None else None
    return LightString(_vertices(rec, found[0], found[1])) if found else None


def _parse_painted_line(rec, near):
    """Painted line: style byte at 6 ((style << 1) | lit), vertex count at 8,
    material GUID at 12..28, vertices from 28. Some files count one more
    vertex than they store, so the count is checked against the size."""
    if len(rec) >= 28:
        count = _u16(rec, 8)
        stored = (len(rec) - 28) // 8
        for n in (count, count - 1):
            if n >= 2 and n == stored:
                verts = _vertices(rec, 28, n)
                if near is None or all(near.accepts(*v) for v in verts):
                    return PaintedLine(verts, style=rec[6] >> 1, lit=bool(rec[6] & 1))
    if near is None:
        return None
    found = near.longest_run(rec, 2)
    return PaintedLine(_vertices(rec, found[0], found[1])) if found else None


_SIGN_LABEL_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]<>^_-|/\\'` ")


def _parse_sign(rec, near):
    """Taxiway sign: a position near the airport, a heading after it, and a
    label string that opens with a panel-type letter (l, d, m, i, r, u)."""
    if near is None:
        return None
    pos_at = None
    for at in range(6, len(rec) - 7):
        if near.accepts(*_pair(rec, at)):
            pos_at = at
            break
    if pos_at is None:
        return None
    lat, lon = _pair(rec, pos_at)
    heading = 0.0
    for at in range(pos_at + 8, min(pos_at + 28, len(rec) - 4), 4):
        v = struct.unpack_from("<f", rec, at)[0]
        # A millimetre altitude read as a float is subnormal -- skip those.
        if math.isfinite(v) and abs(v) >= 1.2e-38 and 0.01 <= v <= 360.0:
            heading = v
            break
    best = None
    i = 6
    while i < len(rec):
        if rec[i] in _SIGN_LABEL_BYTES:
            start = i
            while i < len(rec) and rec[i] in _SIGN_LABEL_BYTES:
                i += 1
            if i - start >= 2 and chr(rec[start]) in "ldmiru" and (best is None or i - start > best[1]):
                best = (start, i - start)
        else:
            i += 1
    if best is None:
        return None
    label = rec[best[0]:best[0] + best[1]].decode("ascii")
    size = 3
    for probe in (best[0] - 1, best[0] - 2, pos_at + 12, pos_at + 13):
        if 0 <= probe < len(rec) and 1 <= rec[probe] <= 5:
            size = rec[probe]
            break
    return Sign(lat=lat, lon=lon, heading=heading, size=size, label=label)


# --- the airport record ---------------------------------------------------------


def decode_airport_layout(blob: bytes) -> AirportLayout:
    """Decode one airport record (6-byte header included). Returns an empty
    AirportLayout (is_empty() True) when nothing recognisable is found --
    callers treat that as "no native layout" and fall back to the stock
    X-Plane airport."""
    layout = AirportLayout()
    if len(blob) < 0x34:
        return layout
    rec_id = _u16(blob, 0)
    layout.lat, layout.lon = _pair(blob, 12)
    layout.alt_m = _i32(blob, 20) / 1000.0
    layout.ident = decode_ident(_u32(blob, 40))
    layout.region = decode_ident(_u32(blob, 44))
    if rec_id == _REC_AIRPORT_2024 and len(blob) >= 0x54:
        ident = decode_ident64(struct.unpack_from("<Q", blob, 0x4C)[0])
        if ident:
            layout.ident = ident
    tlat, tlon = _pair(blob, 24)
    talt = _i32(blob, 32) / 1000.0
    near = _Near(layout.lat, layout.lon)
    if near is not None and near.accepts(tlat, tlon) and abs(talt - layout.alt_m) > 0.5:
        layout.tower = (tlat, tlon, talt)

    start = _find_subrecord_start(blob, _AIRPORT_HEAD_CANDIDATES, _KNOWN_AIRPORT_SUBRECORDS)
    if start is None:
        layout.warnings.append(f"airport record 0x{rec_id:04X}: no readable sub-records")
        return layout
    msfs = start >= 0x44 or rec_id in AIRPORT_RECORD_IDS

    taxi_names = []
    raw_paths = []
    for sub_type, sub in _walk_records(blob, start):
        try:
            if sub_type == _AP_NAME:
                layout.name = _string(sub[6:])
            elif sub_type in _AP_RUNWAYS and len(sub) >= 52:
                layout.runways.append(_parse_runway(sub, msfs or sub_type == _AP_RUNWAY_MSFS))
            elif sub_type == _AP_HELIPAD and len(sub) >= 36:
                layout.helipads.append(_parse_helipad(sub))
            elif sub_type == _AP_COM and len(sub) >= 12:
                com = _parse_com(sub)
                if com.freq_khz > 0:
                    layout.coms.append(com)
            elif sub_type in _AP_TAXI_POINTS:
                layout.taxi_points.extend(_parse_taxi_points(sub))
            elif sub_type == _AP_TAXI_NAME:
                taxi_names.extend(_parse_taxi_names(sub))
            elif sub_type in _AP_TAXI_PATHS:
                raw_paths.extend(_parse_taxi_paths(sub))
            elif sub_type in _AP_PARKINGS:
                layout.parkings.extend(_parse_parkings(sub, layout.warnings))
            elif sub_type in _AP_APRONS:
                verts = _parse_apron(sub, near)
                if len(verts) >= 3:
                    layout.aprons.append(Polygon(verts))
            elif sub_type == _AP_LIGHT_STRING:
                ls = _parse_light_string(sub, near)
                if ls is not None and len(ls.vertices) >= 2:
                    layout.light_strings.append(ls)
            elif sub_type == _AP_PAINTED_LINE:
                pl = _parse_painted_line(sub, near)
                if pl is not None:
                    layout.painted_lines.append(pl)
            elif sub_type == _AP_SIGN:
                sign = _parse_sign(sub, near)
                if sign is not None:
                    layout.signs.append(sign)
        except (struct.error, IndexError) as e:
            layout.warnings.append(f"sub-record 0x{sub_type:04X}: {e}")

    for p in raw_paths:
        if p["kind"] == 2:
            name = runway_name(p["name_index"], p["runway_designator"])
        else:
            idx = p["name_index"]
            name = taxi_names[idx] if idx < len(taxi_names) else ""
        layout.taxi_paths.append(TaxiPath(
            start=p["start"], end=p["end"], kind=p["kind"], name=name, width_m=p["width_m"],
            centre_line=p["centre_line"], centre_lit=p["centre_lit"],
            left_edge_lit=p["left_edge_lit"], right_edge_lit=p["right_edge_lit"]))
    return layout
