"""
Fits placed objects onto X-Plane's terrain through the shared ground grid.

Every MSFS height is a height above one level airport ground; X-Plane's
terrain isn't level. ground_grid.py holds one terrain height per place,
and every placed object is moved by it vertex by vertex:

    placed vertex height = grid(vertex position) + MSFS height of the vertex

X-Plane stands an object's origin on its own terrain (plus the
placement's AGL), so a fitted copy carries grid(vertex) - grid(origin) in
its geometry, and the placement's AGL gets grid(origin) - terrain(origin)
(the "lift": a few centimetres of grid interpolation, or the level of a
levelled steep-ground building). Two objects at the same place therefore
always get the same height, whatever they belong to.

Faces larger than the grid's bends are subdivided first (refine), so a big
floor follows the grid between its corners the way a chair standing on it
does -- otherwise the floor would be flat between its far-apart corners
while the chair follows the grid. Vertical edges are never split (top and
bottom share their x/z), so walls and pillars aren't cut up.

A building on steep ground isn't bent to the slope: the pipeline levels the
grid under it (ground_grid.Grid.level) before fitting, so it -- and
whatever is placed inside it -- stays level, and a skirt is added below
its walls down to the terrain (_add_skirt).

Draped geometry is warped too: X-Plane drapes it at render time anyway,
but its shadow is drawn from the authored heights.

Unreadable terrain only leaves an object as converted; a fit never stops a
conversion.
"""

import dataclasses
import hashlib
import math
import pickle
from pathlib import Path

import numpy as np

import ground_grid
import terrain_dem
from mesh_convert import mesh_ir

_NEGLIGIBLE_M = 0.01      # every vertex this close to its origin's grid height: no fitted copy
_REFINE_TOL_M = 0.02      # an edge is split when the grid bends more than this under its middle
_REFINE_MIN_EDGE_M = 0.5  # ...and it is longer than this (horizontally)
_REFINE_MAX_ROUNDS = 12
_REFINE_MAX_GROWTH = 8    # a mesh is never refined past this many times its triangles...
_REFINE_MIN_BUDGET = 50_000  # ...plus this many (a big floor can be two triangles)
_LIFT_MIN_M = 0.005       # smaller placement lifts are left out

# A building on steep ground is levelled instead of bent: steep means the
# ground's gradient (a plane fitted to samples across its footprint) is
# over _SKIRT_MIN_GRADIENT, the ground drops more than _SKIRT_MIN_SPREAD_M
# under it, and it is taller than _SKIRT_MIN_HEIGHT_M (a flat plate just
# follows the ground).
_SKIRT_MIN_HEIGHT_M = 1.0
_SKIRT_MIN_GRADIENT = 0.03  # 3 % (~1.7 degrees): about 1 m across a 30 m building
_SKIRT_MIN_SPREAD_M = 0.3
_SKIRT_MAX_AGL_M = 0.5      # only things standing on the ground get a skirt
_SKIRT_BAND_M = 0.25  # how far above the object's lowest point a wall edge still counts as its bottom
_SKIRT_WALL_NY = 0.5  # a triangle whose normal's |y| is below this is a wall
_SKIRT_MIN_DROP_M = 0.05  # no skirt piece where the wall already meets the ground

_ir_cache = {}      # obj_stem -> MeshIR, or None if no sidecar / no geometry or lights
_group_cache = {}   # group key -> {obj_stem: (result_stem, applied, reason)}
_lift_cache = {}    # group key -> metres to add to the placement's AGL


def _load_ir(obj_dir: Path, obj_stem: str):
    if obj_stem in _ir_cache:
        return _ir_cache[obj_stem]
    sidecar = mesh_ir.sidecar_path_for(obj_dir / f"{obj_stem}.obj")
    ir = None
    if sidecar.exists():
        try:
            loaded = mesh_ir.load(sidecar)
            if len(loaded.positions) or loaded.lights:
                ir = loaded
        except (OSError, EOFError, pickle.UnpicklingError):
            ir = None
    _ir_cache[obj_stem] = ir
    return ir


def group_key(obj_stems, lat, lon, heading_deg, skip_draped_positions=False):
    return (tuple(sorted(obj_stems)), round(lat, 6), round(lon, 6), round(heading_deg, 2),
            bool(skip_draped_positions))


def group_lift(key):
    """Metres the placement's AGL needs on top of its MSFS height (see the
    module docstring), or 0.0."""
    return _lift_cache.get(key, 0.0)


# --- footprints ------------------------------------------------------------

def model_extent(obj_dir, obj_stems):
    """Extent of one placed model's sub-objects in its own frame:
    {"x0", "x1", "z0", "z1"} over all geometry and lights, plus "rigid":
    (y_min, y_max) of its non-draped geometry, or None without any.
    None when nothing has a readable sidecar."""
    obj_dir = Path(obj_dir)
    xs, zs, ys = [], [], []
    for stem in obj_stems:
        ir = _load_ir(obj_dir, stem)
        if ir is None:
            continue
        if len(ir.positions):
            xs += [float(ir.positions[:, 0].min()), float(ir.positions[:, 0].max())]
            zs += [float(ir.positions[:, 2].min()), float(ir.positions[:, 2].max())]
            if not ir.draped:
                ys += [float(ir.positions[:, 1].min()), float(ir.positions[:, 1].max())]
        for light in ir.lights:
            xs.append(float(light.pos[0]))
            zs.append(float(light.pos[2]))
    if not xs:
        return None
    return {"x0": min(xs), "x1": max(xs), "z0": min(zs), "z1": max(zs),
            "rigid": (min(ys), max(ys)) if ys else None}


def footprint_corners(lat, lon, heading_deg, ext, margin=0.0):
    """The extent's rectangle as four (lat, lon) corners, grown by margin."""
    xs = [ext["x0"] - margin, ext["x1"] + margin, ext["x1"] + margin, ext["x0"] - margin]
    zs = [ext["z0"] - margin, ext["z0"] - margin, ext["z1"] + margin, ext["z1"] + margin]
    lats, lons = ground_grid.local_to_latlon_many(lat, lon, heading_deg, xs, zs)
    return list(zip(lats.tolist(), lons.tolist()))


def _ground_gradient(samples):
    """Steepness (rise over run) of the plane best fitting samples
    [(local_x, local_z, height), ...]; 0 with fewer than 3 usable samples."""
    if len(samples) < 3:
        return 0.0
    a = np.array([[x, z, 1.0] for x, z, _ in samples], dtype=np.float64)
    d = np.array([v for _, _, v in samples], dtype=np.float64)
    if np.linalg.matrix_rank(a) < 3:
        return 0.0
    (gx, gz, _), *_ = np.linalg.lstsq(a, d, rcond=None)
    return float(np.hypot(gx, gz))


def _robust_max(values, mad_k=3.0):
    """The highest value once spikes (modified z-score past mad_k against
    the median) are thrown out."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) >= 3:
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad > 1e-9:
            clean = v[np.abs(v - med) / (1.4826 * mad) <= mad_k]
            if len(clean):
                v = clean
    return float(v.max())


def steep_level(grid, lat, lon, heading_deg, ext, agl=0.0):
    """The level a building at this placement is set to when it stands on
    steep ground (see the module docstring), or None when it just follows
    the grid."""
    if ext is None or ext.get("rigid") is None or abs(agl) > _SKIRT_MAX_AGL_M:
        return None
    y0, y1 = ext["rigid"]
    if y1 - y0 <= _SKIRT_MIN_HEIGHT_M:
        return None
    pts = [(x, z) for x in (ext["x0"], (ext["x0"] + ext["x1"]) / 2, ext["x1"])
           for z in (ext["z0"], (ext["z0"] + ext["z1"]) / 2, ext["z1"])]
    lats, lons = ground_grid.local_to_latlon_many(lat, lon, heading_deg, [p[0] for p in pts], [p[1] for p in pts])
    hs = grid.heights(lats, lons)
    samples = [(x, z, float(h)) for (x, z), h in zip(pts, hs) if not math.isnan(h)]
    if len(samples) < 3:
        return None
    vals = [h for _, _, h in samples]
    if max(vals) - min(vals) <= _SKIRT_MIN_SPREAD_M or _ground_gradient(samples) <= _SKIRT_MIN_GRADIENT:
        return None
    return _robust_max(vals)


# --- geometry --------------------------------------------------------------

def refine(positions, normals, uvs, indices, delta, tol=_REFINE_TOL_M, min_edge=_REFINE_MIN_EDGE_M):
    """Splits every edge under whose middle the height field delta(xs, zs)
    bends more than tol away from the straight line between its ends
    (horizontal length over min_edge), until none is left. The decision
    depends only on an edge's two end positions, so the triangles on both
    sides of an edge -- also across a UV seam, where they don't share
    vertex indices -- split it alike and no cracks open. Returns new
    (positions, normals, uvs, indices)."""
    pos = np.asarray(positions, dtype=np.float64)
    tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if not len(tris):
        return pos, normals, uvs, indices
    has_n = len(normals) == len(pos)
    has_uv = len(uvs) == len(pos)
    nrm = np.asarray(normals, dtype=np.float64) if has_n else None
    uv = np.asarray(uvs, dtype=np.float64) if has_uv else None
    limit = len(tris) * _REFINE_MAX_GROWTH + _REFINE_MIN_BUDGET
    dv = delta(pos[:, 0], pos[:, 2])
    for _ in range(_REFINE_MAX_ROUNDS):
        edges = tris[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        lo, hi = edges.min(axis=1), edges.max(axis=1)
        n = len(pos) + 1
        keys, inv = np.unique(lo * n + hi, return_inverse=True)
        ua, ub = keys // n, keys % n
        pa, pb = pos[ua], pos[ub]
        length = np.hypot(pa[:, 0] - pb[:, 0], pa[:, 2] - pb[:, 2])
        mid = (pa + pb) / 2.0
        dev = np.abs(delta(mid[:, 0], mid[:, 2]) - (dv[ua] + dv[ub]) / 2.0)
        split = (length > min_edge) & (dev > tol)
        count = int(split.sum())
        if not count or len(tris) + 3 * count > limit:
            break
        new_id = np.full(len(keys), -1, dtype=np.int64)
        new_id[split] = np.arange(len(pos), len(pos) + count)
        pos = np.vstack([pos, mid[split]])
        dv = np.concatenate([dv, delta(mid[split][:, 0], mid[split][:, 2])])
        if has_n:
            nm = (nrm[ua[split]] + nrm[ub[split]]) / 2.0
            ln = np.linalg.norm(nm, axis=1, keepdims=True)
            nm = np.where(ln > 1e-12, nm / np.where(ln > 1e-12, ln, 1.0), nrm[ua[split]])
            nrm = np.vstack([nrm, nm])
        if has_uv:
            uv = np.vstack([uv, (uv[ua[split]] + uv[ub[split]]) / 2.0])
        mids = new_id[inv].reshape(-1, 3)  # midpoint of edge ab, bc, ca (-1: not split)
        tris = _split_triangles(tris, mids)
    return (pos, nrm if has_n else normals, uv if has_uv else uvs, tris.reshape(-1))


def _split_triangles(tris, mids):
    """tris (T, 3), mids (T, 3) edge-midpoint ids or -1 -> new (T', 3),
    keeping every triangle's winding."""
    s = mids >= 0
    nsplit = s.sum(axis=1)
    out = [tris[nsplit == 0]]

    def rolled(sel, r):
        cols = (np.arange(3)[None, :] + r[:, None]) % 3
        return tris[sel][np.arange(len(r))[:, None], cols], mids[sel][np.arange(len(r))[:, None], cols]

    one = np.nonzero(nsplit == 1)[0]
    if len(one):
        v, m = rolled(one, np.argmax(s[one], axis=1))  # the split edge first
        out += [np.stack([v[:, 0], m[:, 0], v[:, 2]], axis=1), np.stack([m[:, 0], v[:, 1], v[:, 2]], axis=1)]
    two = np.nonzero(nsplit == 2)[0]
    if len(two):
        unsplit = np.argmin(s[two], axis=1)
        v, m = rolled(two, (unsplit + 1) % 3)  # the unsplit edge last (c-a)
        out += [np.stack([m[:, 0], v[:, 1], m[:, 1]], axis=1),
                np.stack([v[:, 0], m[:, 0], m[:, 1]], axis=1),
                np.stack([v[:, 0], m[:, 1], v[:, 2]], axis=1)]
    three = np.nonzero(nsplit == 3)[0]
    if len(three):
        v, m = tris[three], mids[three]
        out += [np.stack([v[:, 0], m[:, 0], m[:, 2]], axis=1), np.stack([m[:, 0], v[:, 1], m[:, 1]], axis=1),
                np.stack([m[:, 2], m[:, 1], v[:, 2]], axis=1), np.stack([m[:, 0], m[:, 1], m[:, 2]], axis=1)]
    return np.vstack(out)


def _add_skirt(ir, top, y_min, delta_at):
    """The object lifted level by `top`, plus a skirt: for every bottom
    edge of a wall (a near-vertical triangle's edge within _SKIRT_BAND_M of
    the object's lowest point), a new quad from that edge down to the
    terrain under it (delta_at(x, z), never above `top`). The original
    geometry is only translated, so roof and floors stay level. Returns
    (positions, normals, uvs, indices)."""
    pos = ir.positions.copy()
    pos[:, 1] += top
    normals = np.asarray(ir.normals, dtype=np.float64) if len(ir.normals) == len(pos) else np.tile([0.0, 1.0, 0.0], (len(pos), 1))
    uvs = np.asarray(ir.uvs, dtype=np.float64) if len(ir.uvs) == len(pos) else np.zeros((len(pos), 2))
    tris = np.asarray(ir.indices, dtype=np.int64).reshape(-1, 3)
    new_pos, new_nrm, new_uv, new_idx = [], [], [], []
    done = set()
    base = ir.positions[:, 1] <= y_min + _SKIRT_BAND_M
    k = len(pos)
    for tri in tris:
        a, b, c = (int(i) for i in tri)
        pa, pb, pc = ir.positions[a], ir.positions[b], ir.positions[c]
        n = np.cross(pb - pa, pc - pa)
        length = float(np.linalg.norm(n))
        if length < 1e-9 or abs(n[1]) / length >= _SKIRT_WALL_NY:
            continue
        n = n / length
        # edges in the triangle's own winding order keep the skirt facing
        # the same way as the wall above it
        for i, j in ((a, b), (b, c), (c, a)):
            if not (base[i] and base[j]):
                continue
            key = (min(i, j), max(i, j))
            if key in done:
                continue
            done.add(key)
            drops = []
            for v in (i, j):
                d = delta_at(float(ir.positions[v, 0]), float(ir.positions[v, 2]))
                drops.append(top if d is None else min(d, top))
            if top - drops[0] < _SKIRT_MIN_DROP_M and top - drops[1] < _SKIRT_MIN_DROP_M:
                continue
            ti, tj = pos[i], pos[j]
            bi = (ti[0], ir.positions[i, 1] + drops[0], ti[2])
            bj = (tj[0], ir.positions[j, 1] + drops[1], tj[2])
            new_pos += [tuple(ti), tuple(tj), bi, bj]
            new_nrm += [tuple(n)] * 4
            new_uv += [tuple(uvs[i]), tuple(uvs[j]), tuple(uvs[i]), tuple(uvs[j])]
            # (j, i, i_below) and (j, i_below, j_below): a point below the
            # edge needs the edge reversed to keep the wall's orientation
            new_idx += [k + 1, k, k + 2, k + 1, k + 2, k + 3]
            k += 4
    if new_pos:
        pos = np.vstack([pos, np.array(new_pos, dtype=np.float64)])
        normals = np.vstack([normals, np.array(new_nrm, dtype=np.float64)])
        uvs = np.vstack([uvs, np.array(new_uv, dtype=np.float64)])
    indices = np.concatenate([np.asarray(ir.indices, dtype=np.int64), np.array(new_idx, dtype=np.int64)])
    return pos, normals, uvs, indices


# --- fitting ---------------------------------------------------------------

def local_grid(xplane_root, lat, lon, heading_deg, ext, agl=0.0):
    """A grid for one placement on its own (no shared pipeline grid):
    posts around its footprint, levelled when it stands on steep ground.
    Returns (grid, level or None)."""
    grid = ground_grid.Grid.around(lat, lon)
    corners = footprint_corners(lat, lon, heading_deg, ext, margin=ground_grid.SPACING_M)
    lats = [a for a, _ in corners]
    lons = [o for _, o in corners]
    grid.cover(min(lats), max(lats), min(lons), max(lons),
               lambda a, o: terrain_dem.get_elevation(xplane_root, a, o))
    level = steep_level(grid, lat, lon, heading_deg, ext, agl)
    if level is not None:
        grid.level(footprint_corners(lat, lon, heading_deg, ext, margin=ground_grid.SPACING_M), level)
    return grid, level


def get_or_create_fitted_group(obj_dir, obj_stems, base_lat, base_lon, heading_deg, xplane_root,
                               skip_draped_positions=False, grid=None, skirt_level=None, agl=0.0):
    """obj_stems: every sub-object .obj stem one model produced, for ONE
    placement. They are fitted together (one origin, one grid). grid: the
    pipeline's shared ground_grid.Grid, already levelled where needed --
    skirt_level is then the level the pipeline set under this placement
    (None when it didn't). Without a grid, one is built for this placement
    alone (local_grid). Returns {obj_stem: (result_stem, applied, reason)};
    group_lift() then gives the AGL to add to the placement's non-draped
    parts.

    skip_draped_positions: leave draped geometry as it is (the .pol path
    writes it as DSF polygons, which have no heights)."""
    obj_dir = Path(obj_dir)
    key = group_key(obj_stems, base_lat, base_lon, heading_deg, skip_draped_positions)
    if key in _group_cache:
        return _group_cache[key]

    def done(result, lift=0.0):
        _group_cache[key] = result
        _lift_cache[key] = lift
        return result

    if xplane_root is None:
        return done({stem: (stem, False, "no_xplane_root") for stem in obj_stems})
    loaded = {stem: _load_ir(obj_dir, stem) for stem in obj_stems}
    ext = model_extent(obj_dir, obj_stems)
    if ext is None:
        return done({stem: (stem, False, "disqualified") for stem in obj_stems})
    if grid is None:
        grid, skirt_level = local_grid(xplane_root, base_lat, base_lon, heading_deg, ext, agl)

    origin_grid = grid.height(base_lat, base_lon)
    origin_terrain = terrain_dem.get_elevation(xplane_root, base_lat, base_lon)
    if origin_grid is None or origin_terrain is None:
        return done({stem: (stem, False, "terrain_unavailable") for stem in obj_stems})
    lift = origin_grid - origin_terrain
    lift = lift if abs(lift) >= _LIFT_MIN_M else 0.0

    def delta(xs, zs):
        lats, lons = ground_grid.local_to_latlon_many(base_lat, base_lon, heading_deg, xs, zs)
        d = grid.heights(lats, lons) - origin_grid
        return np.where(np.isnan(d), 0.0, d)

    def terrain_delta(x, z):
        lat, lon = ground_grid.local_to_latlon_many(base_lat, base_lon, heading_deg, [x], [z])
        t = terrain_dem.get_elevation(xplane_root, float(lat[0]), float(lon[0]))
        return None if t is None else t - origin_grid

    # negligible when no vertex (or light) of the whole model moves
    spread = 0.0
    for ir in loaded.values():
        if ir is None:
            continue
        if len(ir.positions) and not (ir.draped and skip_draped_positions):
            spread = max(spread, float(np.abs(delta(ir.positions[:, 0], ir.positions[:, 2])).max()))
        if ir.lights:
            pts = np.array([l.pos for l in ir.lights], dtype=np.float64)
            spread = max(spread, float(np.abs(delta(pts[:, 0], pts[:, 2])).max()))
    if spread < _NEGLIGIBLE_M and skirt_level is None:
        return done({stem: (stem, False, "negligible") for stem in obj_stems}, lift)

    rigid_y_min = ext["rigid"][0] if ext.get("rigid") else 0.0
    digest = hashlib.md5(f"{key[1]}_{key[2]}_{key[3]}_{skirt_level}".encode("utf-8")).hexdigest()[:10]
    result = {}
    for stem in obj_stems:
        ir = loaded[stem]
        if ir is None:
            result[stem] = (stem, False, "not_applicable")
            continue
        has_geo = bool(len(ir.positions))
        if has_geo and ir.draped and skip_draped_positions and not ir.lights:
            result[stem] = (stem, False, "skipped_for_polygon_mode")
            continue
        fitted = dataclasses.replace(ir, name=f"{stem}_tfit_{digest}", tilted=False if has_geo else ir.tilted)
        reason = "applied"
        if has_geo and not (ir.draped and skip_draped_positions):
            if skirt_level is not None and not ir.draped:
                fitted.positions, fitted.normals, fitted.uvs, fitted.indices = _add_skirt(
                    ir, 0.0, rigid_y_min, terrain_delta)
                reason = "applied_skirt"
            else:
                pos, nrm, uv, idx = (ir.positions, ir.normals, ir.uvs, ir.indices) if ir.draped else refine(
                    ir.positions, ir.normals, ir.uvs, ir.indices, delta)
                pos = np.array(pos, dtype=np.float64)
                pos[:, 1] += delta(pos[:, 0], pos[:, 2])
                fitted.positions, fitted.normals, fitted.uvs = pos, nrm, uv
                fitted.indices = np.asarray(idx, dtype=np.int64)
        if ir.lights:
            pts = np.array([l.pos for l in ir.lights], dtype=np.float64)
            ds = delta(pts[:, 0], pts[:, 2]) if skirt_level is None else np.zeros(len(pts))
            fitted.lights = [dataclasses.replace(l, pos=(l.pos[0], l.pos[1] + float(d), l.pos[2]))
                             for l, d in zip(ir.lights, ds)]
        path = obj_dir / f"{fitted.name}.obj"
        if not path.exists():
            mesh_ir.write_obj8(fitted, path)
            # draped_merge and later passes read the sidecar, not the .obj
            mesh_ir.save(fitted, mesh_ir.sidecar_path_for(path))
        result[stem] = (fitted.name, True, reason)
    return done(result, lift)
