"""
One ground height for every place in the scenery.

MSFS stands every object on one level airport ground: a placement's height
is a height above that ground, and two objects that touch in MSFS (a floor
and the seat on it, a deck and its pillars, a wall and the window in it)
touch at those heights. X-Plane's terrain under the airport isn't level,
so each object needs the X-Plane ground height added -- and if each one
picks its own correction (a shift worked out from its own footprint, a
level for its own group), two touching objects get two different
corrections and come apart.

So every object reads its correction from one shared field instead: a
grid of X-Plane terrain heights over the scenery (posts SPACING_M apart,
bilinear between them). Every vertex is moved by the grid's height at its
own position, whichever object it belongs to, so whatever touched in MSFS
still touches: same place, same height. Vertical things stay vertical
(top and bottom share their x/z), and things standing on the ground stand
on X-Plane's ground. Large faces are subdivided where the grid bends
under them (see terrain_fit.refine), so a floor follows the grid between
its corners as closely as a chair standing on it does.

Steep ground is the one place a building isn't bent to the terrain: the
grid itself is levelled under it (level(), the posts under its footprint
set to the highest ground there), so the building -- and everything
placed inside it -- stays level, and a skirt closes the gap below its
walls (terrain_fit).

Sparse: posts exist only in BLOCK x BLOCK cell blocks around placed
objects (cover()), so a large scenery doesn't sample square kilometres of
empty terrain. Outside them the grid has no height (NaN / None).
"""

import math
from pathlib import Path

import numpy as np

import geo_transform

SPACING_M = 4.0
BLOCK = 16  # cells per block side (64 m at SPACING_M)


def local_to_latlon_many(base_lat, base_lon, heading_deg, xs, zs):
    """geo_transform.local_offset_to_latlon over arrays."""
    xs = np.asarray(xs, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    h = math.radians(heading_deg)
    rx = xs * math.cos(h) - zs * math.sin(h)
    rz = xs * math.sin(h) + zs * math.cos(h)
    m_lat, m_lon = geo_transform.metres_per_degree(base_lat)
    return base_lat - rz / m_lat, base_lon + rx / m_lon


class Grid:
    """lat0/lon0: post (0, 0); dlat/dlon: post spacing in degrees.
    blocks: {(bi, bj): (BLOCK+1, BLOCK+1) heights}, posts (bi*BLOCK + r,
    bj*BLOCK + c); neighbouring blocks share their edge posts."""

    def __init__(self, lat0, lon0, dlat, dlon, blocks=None, levelled=None):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.dlat, self.dlon = float(dlat), float(dlon)
        self.blocks = blocks if blocks is not None else {}
        self.levelled = levelled if levelled is not None else {}

    @classmethod
    def around(cls, lat, lon, spacing=SPACING_M):
        m_lat, m_lon = geo_transform.metres_per_degree(lat)
        dlat, dlon = spacing / m_lat, spacing / m_lon
        # posts on a fixed lattice, so grids built around nearby points agree
        return cls(math.floor(lat / dlat) * dlat, math.floor(lon / dlon) * dlon, dlat, dlon)

    # --- building -----------------------------------------------------
    def _post_latlon(self, key):
        bi, bj = key
        r = np.arange(BLOCK + 1) + bi * BLOCK
        c = np.arange(BLOCK + 1) + bj * BLOCK
        return np.meshgrid(self.lat0 + r * self.dlat, self.lon0 + c * self.dlon, indexing="ij")

    def blocks_touching(self, lat_min, lat_max, lon_min, lon_max):
        """Keys of the blocks a box (plus one post around it) touches."""
        i0 = math.floor((lat_min - self.lat0) / self.dlat) - 1
        i1 = math.floor((lat_max - self.lat0) / self.dlat) + 1
        j0 = math.floor((lon_min - self.lon0) / self.dlon) - 1
        j1 = math.floor((lon_max - self.lon0) / self.dlon) + 1
        return {(bi, bj) for bi in range(i0 // BLOCK, i1 // BLOCK + 1)
                for bj in range(j0 // BLOCK, j1 // BLOCK + 1)}

    def cover_blocks(self, keys, sample):
        """Samples the posts of every listed block not there yet with
        sample(lat, lon) -> metres or None. Returns how many were added."""
        added = 0
        for key in sorted(keys):
            if key in self.blocks:
                continue
            lats, lons = self._post_latlon(key)
            z = np.full(lats.shape, np.nan)
            for idx in np.ndindex(z.shape):
                v = sample(float(lats[idx]), float(lons[idx]))
                if v is not None:
                    z[idx] = v
            self.add_blocks({key: z})
            added += 1
        return added

    def cover(self, lat_min, lat_max, lon_min, lon_max, sample):
        """cover_blocks for the blocks a box touches."""
        return self.cover_blocks(self.blocks_touching(lat_min, lat_max, lon_min, lon_max), sample)

    def add_blocks(self, blocks):
        """Takes sampled blocks {(bi, bj): heights} (from another process)."""
        for key, z in blocks.items():
            self.blocks[key] = np.asarray(z, dtype=np.float64)
            self.levelled[key] = np.zeros(self.blocks[key].shape, dtype=bool)

    def level(self, corners, height):
        """Sets every post inside the polygon `corners` [(lat, lon), ...]
        (convex, either winding) to `height`, except posts an earlier
        level() already set. Returns how many posts changed."""
        lats = [a for a, _ in corners]
        lons = [o for _, o in corners]
        i0 = math.floor((min(lats) - self.lat0) / self.dlat)
        i1 = math.ceil((max(lats) - self.lat0) / self.dlat)
        j0 = math.floor((min(lons) - self.lon0) / self.dlon)
        j1 = math.ceil((max(lons) - self.lon0) / self.dlon)
        # metres-like axes so the edge test isn't skewed by degree aspect
        sy, sx = 1.0 / self.dlat, 1.0 / self.dlon
        poly = [(a * sy, o * sx) for a, o in corners]
        changed = 0
        for bi in range(i0 // BLOCK, i1 // BLOCK + 1):
            for bj in range(j0 // BLOCK, j1 // BLOCK + 1):
                z = self.blocks.get((bi, bj))
                if z is None:
                    continue
                plat, plon = self._post_latlon((bi, bj))
                py, px = plat * sy, plon * sx
                inside_pos = np.ones(z.shape, dtype=bool)
                inside_neg = np.ones(z.shape, dtype=bool)
                for k in range(len(poly)):
                    ay, ax = poly[k]
                    by, bx = poly[(k + 1) % len(poly)]
                    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                    inside_pos &= cross >= -1e-9
                    inside_neg &= cross <= 1e-9
                hit = (inside_pos | inside_neg) & ~self.levelled[(bi, bj)]
                if hit.any():
                    z[hit] = height
                    self.levelled[(bi, bj)] |= hit
                    changed += int(hit.sum())
        return changed

    # --- reading ------------------------------------------------------
    def heights(self, lats, lons):
        """Bilinear heights at arrays of points (over the posts that have
        terrain); NaN where the grid has no posts."""
        lats = np.atleast_1d(np.asarray(lats, dtype=np.float64))
        lons = np.atleast_1d(np.asarray(lons, dtype=np.float64))
        out = np.full(lats.shape, np.nan)
        fi = (lats - self.lat0) / self.dlat
        fj = (lons - self.lon0) / self.dlon
        i = np.floor(fi).astype(np.int64)
        j = np.floor(fj).astype(np.int64)
        ti, tj = fi - i, fj - j
        bi, bj = i // BLOCK, j // BLOCK
        li, lj = i - bi * BLOCK, j - bj * BLOCK
        keys = bi * 1_000_003 + bj
        for key in np.unique(keys):
            sel = np.nonzero(keys == key)[0]
            z = self.blocks.get((int(bi[sel[0]]), int(bj[sel[0]])))
            if z is None:
                continue
            r, c = li[sel], lj[sel]
            a, b = ti[sel], tj[sel]
            posts = np.stack([z[r, c], z[r, c + 1], z[r + 1, c], z[r + 1, c + 1]])
            w = np.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b])
            # a post without terrain (a missing neighbour tile) drops out
            ok = ~np.isnan(posts)
            w = np.where(ok, w, 0.0)
            total = w.sum(axis=0)
            vals = (w * np.where(ok, posts, 0.0)).sum(axis=0)
            out[sel] = np.where(total > 1e-12, vals / np.where(total > 1e-12, total, 1.0), np.nan)
        return out

    def height(self, lat, lon):
        v = float(self.heights([lat], [lon])[0])
        return None if math.isnan(v) else v

    def __len__(self):
        return len(self.blocks)

    # --- files (worker processes load the parent's grid) ---------------
    def save(self, path):
        keys = sorted(self.blocks)
        np.savez(path, meta=np.array([self.lat0, self.lon0, self.dlat, self.dlon]),
                 keys=np.array(keys, dtype=np.int64).reshape(-1, 2),
                 z=np.array([self.blocks[k] for k in keys]).reshape(-1, BLOCK + 1, BLOCK + 1),
                 lev=np.array([self.levelled[k] for k in keys], dtype=bool).reshape(-1, BLOCK + 1, BLOCK + 1))

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            lat0, lon0, dlat, dlon = (float(v) for v in f["meta"])
            keys = [tuple(int(v) for v in k) for k in f["keys"]]
            blocks = {k: f["z"][n].copy() for n, k in enumerate(keys)}
            levelled = {k: f["lev"][n].copy() for n, k in enumerate(keys)}
        return cls(lat0, lon0, dlat, dlon, blocks, levelled)


_loaded = {}


def load_cached(path):
    """Grid.load, once per process per file."""
    key = str(Path(path))
    if key not in _loaded:
        _loaded.clear()
        _loaded[key] = Grid.load(path)
    return _loaded[key]
