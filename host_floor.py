"""Props inside a building stand on the building's floor, not on the ground
under each of them.

MSFS builds an airport on level ground: a terminal and the seats, people
and counters placed inside it all start from the same ground level. X-Plane
places every object on its own terrain at its own anchor point, and a large
building is moved as one rigid body (terrain_fit's vertical shift), so where
the X-Plane terrain under a terminal isn't level, the props inside it end up
sunk into the floor or floating above it by however much the ground under
each one differs from the ground the building was set on.

This finds, for each small placement, the large building that has a floor
right under it -- a near-horizontal triangle of the building covering the
prop's anchor point at the prop's own height (so an apron inside the bounding
box of an L-shaped terminal, or the ground under an open canopy, doesn't
count) -- and gives it a height that puts it on that building's base level:

    prop height += (ground at building anchor + building shift) - ground at prop

The terrain itself is not changed.
"""

import math

import numpy as np

from geo_transform import latlon_offset_to_local, metres_per_degree

HOST_MIN_SIDE_M = 20.0  # a building at least this long can host props
HOST_MIN_HEIGHT_M = 2.5  # ...and this tall: a flat ground sheet is not a building floor
PROP_MAX_AREA_RATIO = 0.25  # a prop's footprint is at most this share of its host's
_MIN_UP_NORMAL = 0.5  # floor/roof/ceiling triangles, not walls
FLOOR_TOLERANCE_M = 0.5  # how close the floor must be to the prop's own height
_CELL_M = 4.0
_INDEX_DEG = 0.002  # host lookup grid, ~200 m


class Cover:
    """A building's near-horizontal triangles (floors, roofs, ceilings) in
    its own local metres, indexed by x/z. `placed` (optional, same shape)
    holds the same triangles as the building is actually placed -- after
    terrain fitting (shift, warp, skirt lift)."""

    def __init__(self, tris, placed=None):
        self.tris = tris  # (N, 3, 3) x/y/z
        self.placed = placed if placed is not None and placed.shape == tris.shape else tris
        self.cells = {}
        xz = tris[:, :, [0, 2]]
        lo = np.floor(xz.min(axis=1) / _CELL_M).astype(int)
        hi = np.floor(xz.max(axis=1) / _CELL_M).astype(int)
        for i in range(len(tris)):
            for cx in range(lo[i, 0], hi[i, 0] + 1):
                for cz in range(lo[i, 1], hi[i, 1] + 1):
                    self.cells.setdefault((cx, cz), []).append(i)

    def heights_at(self, x, z, placed=False):
        """Local heights of the triangles over/under (x, z) -- as modelled,
        or (placed=True) as placed, in the same order."""
        idx = self.cells.get((int(math.floor(x / _CELL_M)), int(math.floor(z / _CELL_M))))
        if not idx:
            return np.zeros(0)
        t = self.tris[idx]
        tp = self.placed[idx]
        ax, az = t[:, 0, 0], t[:, 0, 2]
        bx, bz = t[:, 1, 0], t[:, 1, 2]
        cx, cz = t[:, 2, 0], t[:, 2, 2]
        det = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
        ok = np.abs(det) > 1e-12
        det = np.where(ok, det, 1.0)
        w1 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / det
        w2 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / det
        w3 = 1.0 - w1 - w2
        eps = -1e-9
        inside = ok & (w1 >= eps) & (w2 >= eps) & (w3 >= eps)
        src = tp if placed else t
        y = w1 * src[:, 0, 1] + w2 * src[:, 1, 1] + w3 * src[:, 2, 1]
        return y[inside]

    def has_floor(self, x, z, y, tol=FLOOR_TOLERANCE_M):
        h = self.heights_at(x, z)
        return bool(len(h)) and bool(np.any(np.abs(h - y) <= tol))

    def floor_at(self, x, z, y, tol=FLOOR_TOLERANCE_M):
        """(modelled height, placed height) of the floor nearest level y
        at (x, z), or None."""
        h = self.heights_at(x, z)
        if not len(h):
            return None
        k = int(np.argmin(np.abs(h - y)))
        if abs(h[k] - y) > tol:
            return None
        return float(h[k]), float(self.heights_at(x, z, placed=True)[k])


def horizontal_cover(irs, placed_irs=None):
    """Cover of the near-horizontal triangles of these MeshIRs (one
    placement's rigid sub-objects, sharing one local frame), or None.
    placed_irs: the same sub-objects as placed (terrain-fitted copies,
    same order); a fitted copy keeps the original's triangles first (a
    skirt only adds triangles after them)."""
    parts, placed_parts = [], []
    for k, ir in enumerate(irs):
        if ir is None or not len(ir.indices) or not len(ir.positions):
            continue
        idx = np.asarray(ir.indices, dtype=np.int64).reshape(-1, 3)
        tri = ir.positions[idx]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(n, axis=1)
        keep = (length > 1e-9) & (np.abs(n[:, 1]) >= _MIN_UP_NORMAL * np.maximum(length, 1e-12))
        if not np.any(keep):
            continue
        parts.append(tri[keep])
        fitted = placed_irs[k] if placed_irs is not None and k < len(placed_irs) else None
        ptri = tri
        if fitted is not None and len(fitted.positions) >= len(ir.positions) \
                and len(fitted.indices) >= len(ir.indices):
            fidx = np.asarray(fitted.indices, dtype=np.int64).reshape(-1, 3)[:len(idx)]
            if np.array_equal(fidx, idx):
                ptri = fitted.positions[fidx]
        placed_parts.append(ptri[keep])
    if not parts:
        return None
    return Cover(np.concatenate(parts), np.concatenate(placed_parts))


def footprint(irs):
    """(x_min, x_max, z_min, z_max) over these MeshIRs' positions, or None."""
    pts = [ir.positions for ir in irs if ir is not None and len(ir.positions)]
    if not pts:
        return None
    p = np.concatenate(pts)
    return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 2].min()), float(p[:, 2].max())


class Host:
    """A building props can stand in. agl: the building's own height above
    its anchor's ground (its placement height_offset plus convert()'s
    recenter lift, i.e. what local y = 0 sits at); base: ground at the
    anchor plus the building's own terrain-fit shift."""

    def __init__(self, key, lat, lon, hdg, bbox, cover, agl, base):
        self.key = key
        self.lat, self.lon, self.hdg = lat, lon, hdg
        self.bbox = bbox
        self.cover = cover
        self.agl = agl
        self.base = base
        self.area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
        self.radius = max(math.hypot(x, z) for x in bbox[:2] for z in bbox[2:])

    def floor_under(self, lat, lon, height_offset):
        """(modelled, placed) local height of this building's floor under a
        prop at (lat, lon) standing height_offset m above the ground, or
        None."""
        x, z = latlon_offset_to_local(self.lat, self.lon, self.hdg, lat, lon)
        x0, x1, z0, z1 = self.bbox
        if not (x0 <= x <= x1 and z0 <= z <= z1):
            return None
        return self.cover.floor_at(x, z, height_offset - self.agl)

    def holds(self, lat, lon, height_offset):
        """Does this building have a floor under (lat, lon) at the level a
        prop placed `height_offset` metres above the ground stands on?"""
        x, z = latlon_offset_to_local(self.lat, self.lon, self.hdg, lat, lon)
        x0, x1, z0, z1 = self.bbox
        return x0 <= x <= x1 and z0 <= z <= z1 and self.cover.has_floor(x, z, height_offset - self.agl)


def is_host_size(bbox, height=None):
    """Long enough (bbox: x_min, x_max, z_min, z_max) and, when `height`
    is given, tall enough to be a building."""
    if bbox is None or max(bbox[1] - bbox[0], bbox[3] - bbox[2]) < HOST_MIN_SIDE_M:
        return False
    return height is None or height >= HOST_MIN_HEIGHT_M


def height_of(irs):
    """Vertical extent of these MeshIRs' positions (0 if none)."""
    ys = [ir.positions[:, 1] for ir in irs if ir is not None and len(ir.positions)]
    if not ys:
        return 0.0
    y = np.concatenate(ys)
    return float(y.max() - y.min())


class HostIndex:
    def __init__(self, hosts):
        self.hosts = list(hosts)
        self.cells = {}
        for h in self.hosts:
            m_lat, m_lon = metres_per_degree(h.lat)
            r_lat, r_lon = h.radius / m_lat, h.radius / m_lon
            for i in range(int(math.floor((h.lat - r_lat) / _INDEX_DEG)), int(math.floor((h.lat + r_lat) / _INDEX_DEG)) + 1):
                for j in range(int(math.floor((h.lon - r_lon) / _INDEX_DEG)), int(math.floor((h.lon + r_lon) / _INDEX_DEG)) + 1):
                    self.cells.setdefault((i, j), []).append(h)

    def host_for(self, lat, lon, height_offset, area, exclude_key=None):
        """The largest host with a floor under a prop at (lat, lon),
        `height_offset` m above the ground, of footprint `area` m^2 -- or
        None."""
        best = None
        for h in self.cells.get((int(math.floor(lat / _INDEX_DEG)), int(math.floor(lon / _INDEX_DEG))), ()):
            if h.key == exclude_key or area > PROP_MAX_AREA_RATIO * h.area:
                continue
            if (best is None or h.area > best.area) and h.holds(lat, lon, height_offset):
                best = h
        return best
