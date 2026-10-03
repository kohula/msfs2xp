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
PROP_MAX_AREA_RATIO = 0.25  # a prop's footprint is at most this share of its host's
_MIN_UP_NORMAL = 0.5  # floor/roof/ceiling triangles, not walls
FLOOR_TOLERANCE_M = 0.5  # how close the floor must be to the prop's own height
_CELL_M = 4.0
_INDEX_DEG = 0.002  # host lookup grid, ~200 m


class Cover:
    """A building's near-horizontal triangles (floors, roofs, ceilings) in
    its own local metres, indexed by x/z."""

    def __init__(self, tris):
        self.tris = tris  # (N, 3, 3) x/y/z
        self.cells = {}
        xz = tris[:, :, [0, 2]]
        lo = np.floor(xz.min(axis=1) / _CELL_M).astype(int)
        hi = np.floor(xz.max(axis=1) / _CELL_M).astype(int)
        for i in range(len(tris)):
            for cx in range(lo[i, 0], hi[i, 0] + 1):
                for cz in range(lo[i, 1], hi[i, 1] + 1):
                    self.cells.setdefault((cx, cz), []).append(i)

    def heights_at(self, x, z):
        """Local heights of the triangles over/under (x, z)."""
        idx = self.cells.get((int(math.floor(x / _CELL_M)), int(math.floor(z / _CELL_M))))
        if not idx:
            return np.zeros(0)
        t = self.tris[idx]
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
        y = w1 * t[:, 0, 1] + w2 * t[:, 1, 1] + w3 * t[:, 2, 1]
        return y[inside]

    def has_floor(self, x, z, y, tol=FLOOR_TOLERANCE_M):
        h = self.heights_at(x, z)
        return bool(len(h)) and bool(np.any(np.abs(h - y) <= tol))


def horizontal_cover(irs):
    """Cover of the near-horizontal triangles of these MeshIRs (one
    placement's rigid sub-objects, sharing one local frame), or None."""
    parts = []
    for ir in irs:
        if ir is None or not len(ir.indices) or not len(ir.positions):
            continue
        tri = ir.positions[np.asarray(ir.indices, dtype=np.int64).reshape(-1, 3)]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(n, axis=1)
        keep = (length > 1e-9) & (np.abs(n[:, 1]) >= _MIN_UP_NORMAL * np.maximum(length, 1e-12))
        if np.any(keep):
            parts.append(tri[keep])
    if not parts:
        return None
    return Cover(np.concatenate(parts))


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

    def holds(self, lat, lon, height_offset):
        """Does this building have a floor under (lat, lon) at the level a
        prop placed `height_offset` metres above the ground stands on?"""
        x, z = latlon_offset_to_local(self.lat, self.lon, self.hdg, lat, lon)
        x0, x1, z0, z1 = self.bbox
        return x0 <= x <= x1 and z0 <= z <= z1 and self.cover.has_floor(x, z, height_offset - self.agl)


def is_host_size(bbox):
    return bbox is not None and max(bbox[1] - bbox[0], bbox[3] - bbox[2]) >= HOST_MIN_SIDE_M


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
