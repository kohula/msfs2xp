"""Bare lights placed next to a lamp post are attached to its lamp.

Some packages draw the lamp posts and add the light separately: a model
holding nothing but a light, placed in the air somewhere near the post's
top. Converted where they were placed, such lights hang beside the post
instead of in it. A bare light that is near a glowing lamp head at about
its height is moved into that head.

One light per lamp head: when a second bare light finds the same head it
is removed, and a head whose post brings its own light keeps only that.
"""

import math

import numpy as np

import geo_transform

REACH_M = 12.0          # a bare light this close (horizontally) to a lamp head is attached to it
HEIGHT_TOL_M = 5.0      # ...when it hangs within this of the lamp's height
MIN_HANG_M = 3.0        # lights lower than this are ground lights, never moved
HEAD_MIN_Y_M = 4.0      # a lamp head sits at least this high on its model
HEAD_MAX_M = 2.5        # and is at most this across
OWN_LIGHT_M = 2.0       # a head with a light of its own this close is already lit
_CELL_M = 0.6
_GRID_DEG = 0.0003      # ~30 m buckets


def lamp_heads(ir):
    """Lamp heads of one lit sub-object (MeshIR with a night texture):
    centres (local x, y, z) of small clusters of its vertices high up."""
    if ir is None or not getattr(ir, "texture_lit", None) or not len(ir.positions):
        return []
    v = np.asarray(ir.positions, dtype=np.float64)
    v = v[v[:, 1] >= HEAD_MIN_Y_M]
    if not len(v):
        return []
    cells = {}
    for i, key in enumerate(map(tuple, np.floor(v / _CELL_M).astype(np.int64))):
        cells.setdefault(key, []).append(i)
    heads, seen = [], set()
    for start in cells:
        if start in seen:
            continue
        group, stack = [], [start]
        seen.add(start)
        while stack:
            c = stack.pop()
            group.extend(cells[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        n = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if n in cells and n not in seen:
                            seen.add(n)
                            stack.append(n)
        g = v[group]
        if float((g.max(axis=0) - g.min(axis=0)).max()) <= HEAD_MAX_M:
            c = g.mean(axis=0)
            heads.append((float(c[0]), float(c[1]), float(c[2])))
    return heads


def _world(entry, local):
    x, y, z = local
    lat, lon = geo_transform.local_offset_to_latlon(entry["lat"], entry["lon"], entry.get("hdg", 0.0), x, z)
    return lat, lon, entry.get("agl", 0.0) + y


def _metres(a, b):
    m_lat, m_lon = geo_transform.metres_per_degree(a[0])
    return (b[0] - a[0]) * m_lat, (b[1] - a[1]) * m_lon, b[2] - a[2]


def attach_to_lamps(dsf_tiles, load_ir, bare_names):
    """Moves bare-light placements (entry names in bare_names) in
    dsf_tiles into lamp heads. load_ir(name) gives an entry's MeshIR or
    None. Returns (moved, dropped)."""
    heads, own = [], []
    for objects in dsf_tiles.values():
        for e in objects:
            name = e.get("name")
            if not name or name in bare_names:
                continue
            ir = load_ir(name)
            if ir is None:
                continue
            for h in lamp_heads(ir):
                heads.append(_world(e, h))
            for lt in getattr(ir, "lights", None) or []:
                own.append(_world(e, lt.pos))
    if not heads:
        return 0, 0

    grid = {}
    for k, h in enumerate(heads):
        grid.setdefault((int(math.floor(h[0] / _GRID_DEG)), int(math.floor(h[1] / _GRID_DEG))), []).append(k)

    def near(p):
        gi, gj = int(math.floor(p[0] / _GRID_DEG)), int(math.floor(p[1] / _GRID_DEG))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                yield from grid.get((gi + di, gj + dj), ())

    taken = set()
    for p in own:
        for k in near(p):
            if math.dist((0, 0, 0), _metres(p, heads[k])) <= OWN_LIGHT_M:
                taken.add(k)

    moved = dropped = 0
    for tile, objects in dsf_tiles.items():
        keep = []
        for e in objects:
            if e.get("name") not in bare_names:
                keep.append(e)
                continue
            ir = load_ir(e["name"])
            lights = getattr(ir, "lights", None) or []
            if not lights or len(lights) > 2 or (ir is not None and len(ir.positions)):
                keep.append(e)
                continue
            light = _world(e, lights[0].pos)
            if light[2] < MIN_HANG_M:
                keep.append(e)
                continue
            best, best_d = None, None
            for k in near(light):
                dn, de, dh = _metres(light, heads[k])
                if math.hypot(dn, de) <= REACH_M and abs(dh) <= HEIGHT_TOL_M:
                    d = math.sqrt(dn * dn + de * de + dh * dh)
                    if best is None or d < best_d:
                        best, best_d = k, d
            if best is None:
                keep.append(e)
                continue
            if best in taken:
                dropped += 1
                continue
            taken.add(best)
            dn, de, dh = _metres(light, heads[best])
            m_lat, m_lon = geo_transform.metres_per_degree(e["lat"])
            e["lat"] += dn / m_lat
            e["lon"] += de / m_lon
            e["agl"] = e.get("agl", 0.0) + dh
            moved += 1
            keep.append(e)
        objects[:] = keep
    return moved, dropped
