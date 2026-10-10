"""The terrain X-Plane actually draws, from a DSF's own triangle mesh.

terrain_dem.py samples the DSF's elevation raster. X-Plane does not draw
that raster: it draws the tile's terrain mesh (a triangulated surface
built from it, with far fewer points), so between mesh vertices the ground
you see can sit decimetres -- on rough ground metres -- off the raster.
Everything placed against the raster then floats or sinks by that much.

This decodes the mesh itself:

  - GEOD atom: point pools. Each POOL atom is a planar array of uint16
    values (array size, plane count, then per plane an encoding byte --
    0 raw, 1 differenced, 2 run-length, 3 run-length + differenced -- and
    the data); the SCAL atom after it gives each plane's scale and offset
    (value = raw / 65535 * scale + offset). Terrain points are
    lon, lat, elevation, (normal, texture...). An elevation of -32768
    means "take it from the elevation raster at this point".
  - CMDS atom: the command stream. Terrain patches (commands 16-18, with
    flags -- bit 0 physical, bit 1 overlay -- and an LOD range) are followed by
    triangles, strips and fans of pool indices (23-31). Only physical
    patches drawn from 0 m are kept (the base surface, not far LODs or
    overlays drawn on top of it).

Sampling is barycentric inside the triangle covering the point. Parsed
meshes are cached on disk (numpy files, keyed by the DSF's identity) so the
terrain-fit worker processes load rather than re-decode them.
"""

import math
import struct
from pathlib import Path

import numpy as np

_GEOD = b"DOEG"
_POOL = b"LOOP"
_SCAL = b"LACS"
_CMDS = b"SDMC"
_RASTER_ELEVATION = -32768.0
_CELL_DEG = 1.0 / 256.0
FORMAT_VERSION = "1"

# argument sizes of commands with fixed-size arguments
_FIXED = {1: 2, 2: 4, 3: 1, 4: 2, 5: 4, 6: 1, 7: 2, 8: 4, 10: 4, 13: 6, 16: 0, 17: 1, 18: 9,
          25: 4, 28: 4, 31: 4}


def _iter_atoms(data, start, end):
    pos = start
    while pos + 8 <= end:
        atom_id = data[pos:pos + 4]
        atom_len = struct.unpack_from("<I", data, pos + 4)[0]
        if atom_len < 8 or pos + atom_len > end:
            break
        yield atom_id, pos + 8, pos + atom_len
        pos += atom_len


def _decode_plane(data, pos, count, mode):
    """One plane of a POOL atom: (uint16 array, new position)."""
    if mode in (0, 1):
        vals = np.frombuffer(data, dtype="<u2", count=count, offset=pos).astype(np.int64)
        pos += 2 * count
    elif mode in (2, 3):
        vals = np.empty(count, dtype=np.int64)
        n = 0
        while n < count:
            code = data[pos]
            pos += 1
            run = code & 0x7F
            if code & 0x80:
                v = struct.unpack_from("<H", data, pos)[0]
                pos += 2
                vals[n:n + run] = v
            else:
                vals[n:n + run] = np.frombuffer(data, dtype="<u2", count=run, offset=pos)
                pos += 2 * run
            n += run
            if run == 0:
                raise ValueError("empty run")
    else:
        raise ValueError(f"pool encoding {mode}")
    if mode in (1, 3):
        vals = np.cumsum(vals) & 0xFFFF
    return vals, pos


def _decode_pool(payload):
    count, planes = struct.unpack_from("<IB", payload, 0)
    pos = 5
    out = np.empty((count, planes), dtype=np.float64)
    for p in range(planes):
        mode = payload[pos]
        pos += 1
        vals, pos = _decode_plane(payload, pos, count, mode)
        out[:, p] = vals
    return out


def _pools(data, start, end):
    """Decoded 16-bit point pools of a GEOD atom, in order."""
    pools, raw = [], None
    for atom_id, s, e in _iter_atoms(data, start, end):
        if atom_id == _POOL:
            raw = _decode_pool(data[s:e])
        elif atom_id == _SCAL and raw is not None:
            planes = raw.shape[1]
            sc = struct.unpack_from(f"<{2 * planes}f", data, s)
            scale = np.array(sc[0::2], dtype=np.float64)
            offset = np.array(sc[1::2], dtype=np.float64)
            pools.append(raw / 65535.0 * scale + offset)
            raw = None
    return pools


def _triangles(cmds, pools):
    """Pool (index, index) pairs of every base-surface triangle in the
    command stream: array (N, 3, 2) of (pool, point)."""
    out = []
    pos, end = 0, len(cmds)
    pool = 0
    keep = True          # current patch: physical and drawn from 0 m
    flags, near = 1, 0.0

    def add_list(idx):
        if keep and len(idx) >= 3:
            idx = np.asarray(idx[: len(idx) // 3 * 3], dtype=np.int64).reshape(-1, 3)
            out.append(np.stack([np.full_like(idx, pool), idx], axis=-1))

    def add_strip(idx):
        if keep and len(idx) >= 3:
            i = np.asarray(idx, dtype=np.int64)
            tri = np.stack([i[:-2], i[1:-1], i[2:]], axis=1)
            out.append(np.stack([np.full_like(tri, pool), tri], axis=-1))

    def add_fan(idx):
        if keep and len(idx) >= 3:
            i = np.asarray(idx, dtype=np.int64)
            tri = np.stack([np.full(len(i) - 2, i[0]), i[1:-1], i[2:]], axis=1)
            out.append(np.stack([np.full_like(tri, pool), tri], axis=-1))

    def add_cross(pairs, kind):
        if not keep or len(pairs) < 3:
            return
        p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if kind == "list":
            sel = [p[k:k + 3] for k in range(0, len(p) - 2, 3)]
        elif kind == "strip":
            sel = [p[k:k + 3] for k in range(len(p) - 2)]
        else:
            sel = [np.stack([p[0], p[k], p[k + 1]]) for k in range(1, len(p) - 1)]
        if sel:
            out.append(np.stack(sel))

    while pos < end:
        cmd = cmds[pos]
        pos += 1
        if cmd in _FIXED:
            n = _FIXED[cmd]
            if cmd == 1:
                pool = struct.unpack_from("<H", cmds, pos)[0]
            elif cmd == 16:
                keep = bool(flags & 1) and not (flags & 2) and near <= 0.0
            elif cmd == 17:
                flags = cmds[pos]
                keep = bool(flags & 1) and not (flags & 2) and near <= 0.0
            elif cmd == 18:
                flags = cmds[pos]
                near, _far = struct.unpack_from("<ff", cmds, pos + 1)
                keep = bool(flags & 1) and not (flags & 2) and near <= 0.0
            elif cmd == 25:
                a, b = struct.unpack_from("<HH", cmds, pos)
                add_list(list(range(a, b)))
            elif cmd == 28:
                a, b = struct.unpack_from("<HH", cmds, pos)
                add_strip(list(range(a, b)))
            elif cmd == 31:
                a, b = struct.unpack_from("<HH", cmds, pos)
                add_fan(list(range(a, b)))
            pos += n
        elif cmd in (9, 23, 26, 29):
            count = cmds[pos]
            idx = np.frombuffer(cmds, dtype="<u2", count=count, offset=pos + 1)
            pos += 1 + 2 * count
            if cmd == 23:
                add_list(idx)
            elif cmd == 26:
                add_strip(idx)
            elif cmd == 29:
                add_fan(idx)
        elif cmd in (24, 27, 30):
            count = cmds[pos]
            pairs = np.frombuffer(cmds, dtype="<u2", count=2 * count, offset=pos + 1)
            pos += 1 + 4 * count
            add_cross(pairs, {24: "list", 27: "strip", 30: "fan"}[cmd])
        elif cmd == 11:
            count = cmds[pos]
            pos += 1 + 4 * count
        elif cmd == 12:
            count = cmds[pos + 2]
            pos += 3 + 2 * count
        elif cmd == 14:
            windings = cmds[pos + 2]
            pos += 3
            for _ in range(windings):
                count = cmds[pos]
                pos += 1 + 2 * count
        elif cmd == 15:
            windings = cmds[pos + 2]
            pos += 3 + 2 * (windings + 1)
        elif cmd == 32:
            pos += 1 + cmds[pos]
        elif cmd == 33:
            pos += 2 + struct.unpack_from("<H", cmds, pos)[0]
        elif cmd == 34:
            pos += 4 + struct.unpack_from("<I", cmds, pos)[0]
        else:
            raise ValueError(f"unknown DSF command {cmd} at {pos - 1}")
    if not out:
        return np.zeros((0, 3, 2), dtype=np.int64)
    return np.concatenate(out)


def parse_mesh(dsf_bytes, raster_elevation=None):
    """(lon, lat, elev) arrays of shape (N, 3) for the base terrain
    triangles of one decompressed DSF, or None. raster_elevation(lats,
    lons) (arrays; nan where unknown) supplies the height of points whose
    elevation is "from the raster"."""
    if dsf_bytes is None or dsf_bytes[:8] != b"XPLNEDSF":
        return None
    pools, cmds = [], None
    for atom_id, s, e in _iter_atoms(dsf_bytes, 12, len(dsf_bytes)):
        if atom_id == _GEOD:
            pools = _pools(dsf_bytes, s, e)
        elif atom_id == _CMDS:
            cmds = dsf_bytes[s:e]
    if not pools or cmds is None:
        return None
    tris = _triangles(cmds, pools)
    if not len(tris):
        return None
    ok = np.ones(len(tris), dtype=bool)
    for k in range(3):
        p = tris[:, k, 0]
        ok &= p < len(pools)
    tris = tris[ok]
    lon = np.empty(tris.shape[:2])
    lat = np.empty(tris.shape[:2])
    elev = np.empty(tris.shape[:2])
    valid = np.ones(len(tris), dtype=bool)
    for pi, pts in enumerate(pools):
        if pts.shape[1] < 3:
            continue
        for k in range(3):
            sel = tris[:, k, 0] == pi
            idx = tris[sel, k, 1]
            inside = idx < len(pts)
            rows = np.where(sel)[0]
            valid[rows[~inside]] = False
            idx = np.where(inside, idx, 0)
            lon[sel, k] = pts[idx, 0]
            lat[sel, k] = pts[idx, 1]
            elev[sel, k] = pts[idx, 2]
    lon, lat, elev = lon[valid], lat[valid], elev[valid]
    from_raster = elev < _RASTER_ELEVATION + 0.5
    if from_raster.any():
        if raster_elevation is None:
            return None
        elev[from_raster] = raster_elevation(lat[from_raster], lon[from_raster])
        good = ~np.isnan(elev).any(axis=1)
        lon, lat, elev = lon[good], lat[good], elev[good]
    return lon, lat, elev


class Mesh:
    """Barycentric sampling of a parsed mesh. Coordinates are kept
    relative to the tile's south-west corner for float precision."""

    def __init__(self, lon, lat, elev, tile_lon, tile_lat):
        self.tile_lon, self.tile_lat = tile_lon, tile_lat
        self.x = np.asarray(lon, dtype=np.float64) - tile_lon
        self.y = np.asarray(lat, dtype=np.float64) - tile_lat
        self.z = np.asarray(elev, dtype=np.float64)
        self.x0, self.x1 = self.x.min(axis=1), self.x.max(axis=1)
        self.y0, self.y1 = self.y.min(axis=1), self.y.max(axis=1)
        self._cells = {}

    def _candidates(self, x, y):
        key = (int(math.floor(x / _CELL_DEG)), int(math.floor(y / _CELL_DEG)))
        idx = self._cells.get(key)
        if idx is None:
            cx0, cy0 = key[0] * _CELL_DEG, key[1] * _CELL_DEG
            hit = ((self.x1 >= cx0) & (self.x0 <= cx0 + _CELL_DEG)
                   & (self.y1 >= cy0) & (self.y0 <= cy0 + _CELL_DEG))
            idx = np.nonzero(hit)[0]
            self._cells[key] = idx
        return idx

    def elevation(self, lat, lon):
        x, y = lon - self.tile_lon, lat - self.tile_lat
        idx = self._candidates(x, y)
        if not len(idx):
            return None
        tx, ty, tz = self.x[idx], self.y[idx], self.z[idx]
        ax, ay = tx[:, 0], ty[:, 0]
        bx, by = tx[:, 1], ty[:, 1]
        cx, cy = tx[:, 2], ty[:, 2]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        ok = np.abs(det) > 1e-18
        det = np.where(ok, det, 1.0)
        w1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det
        w2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det
        w3 = 1.0 - w1 - w2
        eps = -1e-9
        inside = ok & (w1 >= eps) & (w2 >= eps) & (w3 >= eps)
        if not inside.any():
            return None
        k = int(np.argmax(inside))
        return float(w1[k] * tz[k, 0] + w2[k] * tz[k, 1] + w3[k] * tz[k, 2])


def save(path, lon, lat, elev):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{np.random.randint(1 << 30)}.npz")
    np.savez(tmp, lon=lon, lat=lat, elev=elev)
    tmp.replace(path)


def load(path):
    with np.load(path) as f:
        return f["lon"], f["lat"], f["elev"]
