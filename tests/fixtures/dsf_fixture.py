"""
Synthetic DSF byte construction for terrain_dem.py tests.

Mirrors the real DSF atom structure this project's own dsf_compiler.py
writes and terrain_dem.py reads: every FourCC below is stored on disk as
the REVERSE of its conventional name (e.g. "HEAD" is the 4 bytes
b'DAEH') -- see terrain_dem.py's own module docstring for why this
reversed convention is trusted as ground truth (it matches what
dsf_compiler.py already writes and X-Plane already accepts).
"""

import struct
import array


def pack_atom(magic: bytes, payload: bytes) -> bytes:
    return magic + struct.pack("<I", len(payload) + 8) + payload


def string_table(items) -> bytes:
    if not items:
        return b""
    return b"\x00".join(i.encode("utf-8") for i in items) + b"\x00"


def build_elevation_dsf(grid: list[list[float]], scale: float = 1.0, offset: float = 0.0,
                         layer_name: str = "elevation") -> bytes:
    """grid: row-major, row 0 = south edge, col 0 = west edge (matches the
    real DSF raster scan order). Values are raw int16 samples; real_value =
    raw * scale + offset once decoded."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    raw_values = array.array("h")
    for row in grid:
        for v in row:
            raw_values.append(int(v))
    demd_payload = raw_values.tobytes()
    # flags=1: dsf_Raster_Format_Int (signed), per xptools' DSFDefs.h --
    # the low 2 bits of "flags" are a format ENUM (0=float, 1=int/signed,
    # 2=unsigned int, 3=unsigned int normalized), not independent bit
    # flags. Matches a real installed X-Plane 11 default elevation tile's
    # own flags value (5 = format 1 + the unrelated "Post" bit), verified
    # by hand against real scenery data.
    demi_payload = struct.pack("<BBHIIff", 1, 2, 1, width, height, scale, offset)  # bpp=2, flags=1 (signed int)

    atom_dems = pack_atom(b"SMED", pack_atom(b"IMED", demi_payload) + pack_atom(b"DMED", demd_payload))
    atom_defn = pack_atom(b"NFED", pack_atom(b"NMED", string_table([layer_name])))
    return b"XPLNEDSF" + struct.pack("<I", 1) + atom_defn + atom_dems


def build_multi_layer_elevation_dsf(layers: list[tuple[str, list[list[float]]]],
                                     scale: float = 1.0, offset: float = 0.0) -> bytes:
    """Multiple raster layers (name, grid) packed into a SINGLE top-level
    DEMS/SMED atom as repeated (IMED, DMED) sub-atom pairs, in the same
    order as the DEMN name table -- mirrors the real on-disk structure of
    a real default-global-scenery tile (elevation + sea_level + bathymetry
    all inside one SMED atom), which an earlier version of terrain_dem.py
    got wrong (see terrain_dem.py's own parse_dem_layers comments)."""
    pairs = b""
    for _name, grid in layers:
        height = len(grid)
        width = len(grid[0]) if height else 0
        raw_values = array.array("h")
        for row in grid:
            for v in row:
                raw_values.append(int(v))
        demi_payload = struct.pack("<BBHIIff", 1, 2, 1, width, height, scale, offset)
        pairs += pack_atom(b"IMED", demi_payload) + pack_atom(b"DMED", raw_values.tobytes())

    atom_dems = pack_atom(b"SMED", pairs)
    atom_defn = pack_atom(b"NFED", pack_atom(b"NMED", string_table([name for name, _ in layers])))
    return b"XPLNEDSF" + struct.pack("<I", 1) + atom_defn + atom_dems
