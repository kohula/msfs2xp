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


def _pool_atom(points, scales, offsets, mode=0):
    """POOL + SCAL atoms for points [(plane values...)] given each plane's
    scale/offset; values are quantised to uint16 as X-Plane stores them."""
    import struct as _s
    planes = len(points[0])
    raw = [[round((p[k] - offsets[k]) / scales[k] * 65535.0) for p in points] for k in range(planes)]
    body = _s.pack("<IB", len(points), planes)
    for col in raw:
        if mode == 3:  # run-length + differenced: deltas as literal runs
            deltas, prev = [], 0
            for v in col:
                deltas.append((v - prev) & 0xFFFF)
                prev = v
            body += bytes([3])
            for i in range(0, len(deltas), 127):
                chunk = deltas[i:i + 127]
                body += bytes([len(chunk)]) + _s.pack(f"<{len(chunk)}H", *chunk)
        else:
            body += bytes([0]) + _s.pack(f"<{len(col)}H", *col)
    scal = b"".join(_s.pack("<ff", scales[k], offsets[k]) for k in range(planes))
    return pack_atom(b"LOOP", body) + pack_atom(b"LACS", scal)


def build_mesh_dsf(points, commands, grid=None, tile=(47, 8), pool_mode=0, elev_range=(0.0, 1000.0)):
    """A DSF with one 16-bit point pool of (lon, lat, elev) points and a
    CMDS stream; optionally an elevation raster (see build_elevation_dsf).
    elev_range: (offset, scale) of the elevation plane."""
    lat0, lon0 = tile
    off, sc = elev_range
    geod = pack_atom(b"DOEG", _pool_atom(points, [1.0, 1.0, sc], [lon0, lat0, off], pool_mode))
    cmds = pack_atom(b"SDMC", commands)
    head = build_elevation_dsf(grid) if grid is not None else b"XPLNEDSF" + struct.pack("<I", 1)
    return head + geod + cmds
