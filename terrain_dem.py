"""
Real X-Plane ground-elevation sampling from the user's own installed
default "Global Scenery" DSF tiles.

Used by the large-building terrain-fit step: a rigid MSFS building mesh
assumes flat ground under its footprint, so real per-corner elevation from
the DSF's own raster (see main.py's placement step) is what lets a large
building be tilted/deformed to match actual terrain instead.

FORMAT NOTES -- every atom FourCC below is stored on disk as the *reverse*
of its conventional name (e.g. "HEAD" is the 4 bytes b'DAEH' on disk),
matching dsf_compiler.py's own pack_atom() convention in this codebase.

  - Default global scenery DSFs are valid 7z archives (magic
    "7z\\xBC\\xAF\\x27\\x1C"); py7zr unwraps that layer, guarded so a
    missing py7zr install just means "no terrain data available".
  - Elevation is one of several raster layers stored as top-level "DEMS"
    (b'SMED') atoms. Each contains a "DEMI" (b'IMED') info sub-atom
    (bytes-per-pixel, flags, width, height, scale, offset) and a "DEMD"
    (b'DMED') sub-atom holding the raw width*height grid --
    real_value = raw * scale + offset. Layer NAMES live in a "DEMN"
    (b'NMED') string table nested inside "DEFN" (b'NFED'), mapping to DEMS
    atoms by position (Nth name <-> Nth DEMS atom).
  - This is the same raster X-Plane's own default-global-scenery mesh was
    generated from, so sampling it (~3-arc-second post spacing) is a valid
    proxy for real terrain height without decoding the heavier CMDS mesh.

Every failure mode here (missing file, missing py7zr, malformed atom,
NODATA samples) degrades to returning None rather than raising -- a
missing/unavailable DEM only skips the terrain-fit improvement, never
breaks a conversion run.
"""

import io
import math
import struct
import sys
import tempfile
from pathlib import Path

try:
    import py7zr
    _HAVE_PY7ZR = True
except ImportError:
    _HAVE_PY7ZR = False

_MAGIC = b"XPLNEDSF"
_ATOM_DEFN = b"NFED"  # "DEFN" reversed
_ATOM_DEMN = b"NMED"  # "DEMN" reversed -- raster layer *names*, inside DEFN
_ATOM_DEMS = b"SMED"  # "DEMS" reversed -- one per raster layer
_ATOM_DEMI = b"IMED"  # "DEMI" reversed -- raster info header
_ATOM_DEMD = b"DMED"  # "DEMD" reversed -- raw raster grid

_SEVEN_ZIP_MAGIC = b"7z\xbc\xaf\x27\x1c"

# DEMI's "flags" field is a bitfield whose low 2 bits are a format ENUM
# (dsf_Raster_Format_Mask = 3, per xptools' DSFDefs.h), not independent
# booleans: Float=0, Int=1, Unsigned_Int=2, Unsigned_Int_Normalized=3;
# bit 2 (value 4) is dsf_Raster_Post, irrelevant to decoding values. A
# real elevation layer's flags value is 5 (Int + Post bit) -- treating
# bit 0 alone as "is float" misdecodes Int layers as float32.
_RASTER_FORMAT_MASK = 0x3
_RASTER_FORMAT_FLOAT = 0
_RASTER_FORMAT_INT = 1
_RASTER_FORMAT_UNSIGNED_INT = 2
_RASTER_FORMAT_UNSIGNED_INT_NORMALIZED = 3

# Next to this script, not the system temp/profile drive -- see
# cache_utils.py's module docstring. Frozen-EXE handling matches
# cache_utils.py's (__file__ resolves inside the ephemeral PyInstaller
# extraction dir once bundled, not the real .exe's folder).
if getattr(sys, "frozen", False):
    _SCRIPT_DIR = Path(sys.executable).resolve().parent
else:
    _SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_TEMP_ROOT = _SCRIPT_DIR / "_temp"

_GLOBAL_SCENERY_CANDIDATES = [
    "Global Scenery/X-Plane 12 Global Scenery/Earth nav data",
    "Global Scenery/X-Plane 11 Global Scenery/Earth nav data",
]

# Sentinel raw value meaning "no data here, consult the mesh instead" --
# always the most-negative representable value for a signed layer (the
# elevation layer is signed 16-bit in every known default-scenery DSF).
_NODATA_RAW = {1: -128, 2: -32768, 4: -2147483648}

_dem_cache = {}  # dsf_path (str) -> {layer_name: DemLayer}


class DemLayer:
    """One raster layer's grid, decoded lazily per-sample (never expands
    the whole grid into a Python list -- tiles can be 1000+ posts per
    side)."""

    __slots__ = ("width", "height", "scale", "offset", "bpp", "is_float", "is_signed", "data")

    def __init__(self, width, height, scale, offset, bpp, is_float, is_signed, data):
        self.width = width
        self.height = height
        self.scale = scale
        self.offset = offset
        self.bpp = bpp
        self.is_float = is_float
        self.is_signed = is_signed
        self.data = data

    def _fmt_char(self):
        if self.is_float:
            return "f"
        return {1: "b" if self.is_signed else "B",
                2: "h" if self.is_signed else "H",
                4: "i" if self.is_signed else "I"}[self.bpp]

    def raw_at(self, row, col):
        """Raw (undecoded) sample at (row, col); row 0 = south edge, col 0
        = west edge, per the DSF raster spec's bottom-up/west-first scan
        order. Returns None if out of range."""
        if not (0 <= row < self.height and 0 <= col < self.width):
            return None
        idx = (row * self.width + col) * self.bpp
        if idx + self.bpp > len(self.data):
            return None
        return struct.unpack_from("<" + self._fmt_char(), self.data, idx)[0]

    def value_at(self, row, col):
        """Decoded real-world value at an exact post, or None if that post
        is NODATA."""
        raw = self.raw_at(row, col)
        if raw is None:
            return None
        if not self.is_float and raw == _NODATA_RAW.get(self.bpp):
            return None
        return raw * self.scale + self.offset

    def bilinear(self, frac_row, frac_col):
        """frac_row/frac_col in [0, 1] across the whole tile. Returns None
        if any of the 4 surrounding posts is NODATA -- blending a real
        value against a NODATA sentinel would silently produce garbage, so
        the whole sample is treated as unavailable instead."""
        row_f = frac_row * (self.height - 1)
        col_f = frac_col * (self.width - 1)
        row0 = max(0, min(self.height - 2, int(math.floor(row_f))))
        col0 = max(0, min(self.width - 2, int(math.floor(col_f))))
        ty = row_f - row0
        tx = col_f - col0

        v00 = self.value_at(row0, col0)
        v10 = self.value_at(row0, col0 + 1)
        v01 = self.value_at(row0 + 1, col0)
        v11 = self.value_at(row0 + 1, col0 + 1)
        if v00 is None or v10 is None or v01 is None or v11 is None:
            return None
        top = v00 + (v10 - v00) * tx
        bottom = v01 + (v11 - v01) * tx
        return top + (bottom - top) * ty


def _tile_path_parts(lat, lon):
    tile_lat = int(math.floor(lat))
    tile_lon = int(math.floor(lon))
    band_lat = int(math.floor(tile_lat / 10.0) * 10)
    band_lon = int(math.floor(tile_lon / 10.0) * 10)
    folder = f"{band_lat:+03d}{band_lon:+04d}"
    filename = f"{tile_lat:+03d}{tile_lon:+04d}.dsf"
    return folder, filename, tile_lat, tile_lon


def find_dsf_for_latlon(xplane_root, lat, lon):
    """Locates the default-global-scenery DSF tile covering (lat, lon), or
    None if no candidate exists under xplane_root."""
    xplane_root = Path(xplane_root)
    folder, filename, _, _ = _tile_path_parts(lat, lon)
    for rel in _GLOBAL_SCENERY_CANDIDATES:
        candidate = xplane_root / rel / folder / filename
        if candidate.is_file():
            return candidate
    # Fallback: scan any "Global Scenery/*/Earth nav data" pack -- covers
    # renamed or region-split global scenery installs the two known exact
    # paths above don't match.
    global_scenery_root = xplane_root / "Global Scenery"
    if global_scenery_root.is_dir():
        try:
            pack_dirs = list(global_scenery_root.iterdir())
        except OSError:
            pack_dirs = []
        for pack_dir in pack_dirs:
            candidate = pack_dir / "Earth nav data" / folder / filename
            if candidate.is_file():
                return candidate
    return None


def _iter_atoms(data, start, end):
    pos = start
    while pos + 8 <= end:
        atom_id = data[pos:pos + 4]
        atom_len = struct.unpack_from("<I", data, pos + 4)[0]
        if atom_len < 8 or pos + atom_len > end:
            break  # malformed/truncated -- stop rather than risk garbage reads
        yield atom_id, pos + 8, pos + atom_len
        pos += atom_len


def _parse_string_table(payload):
    if not payload:
        return []
    text = payload.decode("utf-8", errors="replace")
    if text.endswith("\x00"):
        text = text[:-1]
    return text.split("\x00") if text else []


def _decompress_if_7z(raw):
    if raw[:6] != _SEVEN_ZIP_MAGIC:
        return raw
    if not _HAVE_PY7ZR:
        return None
    # py7zr's SevenZipFile has no in-memory read-back API (extract/
    # extractall always write to disk), so the single inner member is
    # unpacked into a scratch temp dir and read back from there.
    _LOCAL_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(_LOCAL_TEMP_ROOT)) as scratch:
        with py7zr.SevenZipFile(io.BytesIO(raw), mode="r") as archive:
            names = archive.getnames()
            if not names:
                return None
            archive.extract(path=scratch, targets=[names[0]])
        extracted_path = Path(scratch) / names[0]
        if not extracted_path.is_file():
            return None
        return extracted_path.read_bytes()


def parse_dem_layers(dsf_path):
    """Returns {layer_name: DemLayer} for every raster layer in one DSF
    file, or {} on any read/decompress/parse failure -- terrain-fit is a
    best-effort visual improvement, never something that should raise and
    interrupt a conversion run."""
    try:
        raw = Path(dsf_path).read_bytes()
    except OSError:
        return {}

    try:
        raw = _decompress_if_7z(raw)
    except Exception:
        return {}
    if raw is None or len(raw) < 12 or raw[:8] != _MAGIC:
        return {}

    layer_names = []
    # (info_tuple, data_bytes) in the order encountered, across every DEMS
    # atom -- a single top-level DEMS atom can pack MULTIPLE raster layers
    # (e.g. elevation + sea_level + bathymetry) as repeated (DEMI, DEMD)
    # sub-atom pairs, not one DEMS atom per layer, so every pair must be
    # kept rather than just the last one found inside each DEMS atom.
    raster_pairs = []
    try:
        for atom_id, body_start, body_end in _iter_atoms(raw, 12, len(raw)):
            if atom_id == _ATOM_DEFN:
                for sub_id, sub_start, sub_end in _iter_atoms(raw, body_start, body_end):
                    if sub_id == _ATOM_DEMN:
                        layer_names = _parse_string_table(raw[sub_start:sub_end])
            elif atom_id == _ATOM_DEMS:
                pending_info = None
                for sub_id, sub_start, sub_end in _iter_atoms(raw, body_start, body_end):
                    if sub_id == _ATOM_DEMI:
                        payload = raw[sub_start:sub_end]
                        if len(payload) < 20:
                            pending_info = None
                            continue
                        _version, bpp, flags, width, height, scale, offset = struct.unpack_from("<BBHIIff", payload, 0)
                        pending_info = (bpp, flags, width, height, scale, offset)
                    elif sub_id == _ATOM_DEMD:
                        if pending_info is not None:
                            raster_pairs.append((pending_info, raw[sub_start:sub_end]))
                            pending_info = None
    except struct.error:
        return {}

    layers = {}
    for idx, (info, data_bytes) in enumerate(raster_pairs):
        name = layer_names[idx] if idx < len(layer_names) else f"layer_{idx}"
        bpp, flags, width, height, scale, offset = info
        if bpp not in (1, 2, 4) or width <= 1 or height <= 1:
            continue
        raster_format = flags & _RASTER_FORMAT_MASK
        is_float = raster_format == _RASTER_FORMAT_FLOAT
        is_signed = raster_format == _RASTER_FORMAT_INT
        # Formats 2/3 (unsigned int / unsigned int normalized) both decode
        # via DemLayer's unsigned struct chars with is_signed=False;
        # "normalized" scaling isn't handled specially (unused in practice).
        layers[name] = DemLayer(width, height, scale, offset, bpp, is_float, is_signed, data_bytes)
    return layers


def _get_cached_layers(dsf_path):
    key = str(dsf_path)
    if key not in _dem_cache:
        _dem_cache[key] = parse_dem_layers(dsf_path)
    return _dem_cache[key]


def get_elevation(xplane_root, lat, lon, layer_name="elevation"):
    """Real X-Plane terrain elevation (meters) at (lat, lon), bilinearly
    sampled from the default global scenery's own elevation raster, or
    None if unavailable (no X-Plane install found, tile missing, py7zr not
    installed, or that post is NODATA in the source data)."""
    if xplane_root is None:
        return None
    dsf_path = find_dsf_for_latlon(xplane_root, lat, lon)
    if dsf_path is None:
        return None
    layers = _get_cached_layers(dsf_path)
    layer = layers.get(layer_name)
    if layer is None:
        return None
    _, _, tile_lat, tile_lon = _tile_path_parts(lat, lon)
    frac_lon = lon - tile_lon
    frac_lat = lat - tile_lat
    return layer.bilinear(frac_lat, frac_lon)
