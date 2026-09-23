"""
Cross-object draped-layer merge.

MSFS often represents one continuous painted/paved surface (a base
asphalt fill, a paint stripe, a wear overlay, ...) as separate placed
objects at the same real-world location, not as sub-materials of one
glTF model. mesh_convert's own per-file draw-order ranking
(draped_ranking.rank_draped_layer_offsets) can't prevent two such
objects landing on the same (layer_group, offset) if they came from
different source files, causing z-fighting/floating-veil artifacts.

Runs once per DSF tile, after every placement in it is known (main.py,
after terrain-fit, before the tile is written): merges every draped
object sharing a texture within that tile into one combined mesh (world-
space triangles pooled into a tile-relative-metres vertex/index stream,
with near-duplicate vertices welded so adjacent pieces of the same real
surface share exact boundary vertices instead of leaving a seam). The
merged object replaces its members in the tile's object list; unmerged/
non-draped/AGL-mounted objects are untouched.

Every merged layer in a tile shares ONE WGS84 metres/degree conversion
and, in OBJ8 mode, one recenter centre + placement anchor -- otherwise
separate-material pieces of one real marking (a crosswalk's red bed vs.
its white bars) would frame/recenter independently and slide apart.

Green terrain-through stripes on the apron are draped-render CRACKS, not
coverage gaps -- X-Plane drapes each pavement OBJ8 onto the terrain mesh
independently, so a hairline crack opens along any shared edge. Three
passes, cheapest first: _weld_component_gaps (index-welds a mesh's own
disconnected tiles into one component), _snap_pavement_seams (<=0.15m
position-snap of boundary vertices ACROSS separate pavement objects, run
after ranking is frozen), and _fill_pavement_gaps (an opaque concrete
underlay at the bottom band as a last-resort safety net, toggle
MSFS2XP_BRIDGE_SEAMS=0). None of the three adds/moves an interior vertex
or changes ranking of any real layer.

Painted markings get no vertex weld; same-texture markings pool into one
draw call with duplicate/degenerate faces dropped geometrically
(_dedupe_faces_geom). The markings-band "rest" tier folds colour/wear
variants of one numbered marking to a single _family_key before
area-ranking, since ranking per texture alone overflows the 9 slots.

Consumes each candidate's .meshir.pkl sidecar (mesh_convert.mesh_ir) --
real float64 numpy arrays, not a reparse of the .obj text. mesh_convert
never writes a sidecar for animated/blink objects, which excludes them
here for free (a merged object can only carry one ATTR_light_level/
ANIM_ directive for its whole stream, so folding several independently-
triggered blink objects together would drop all but one's behavior).

Also assigns each remaining draped object a draw slot: one of five
ordered X-Plane draped bands by what the layer physically is (solid
colour fill < textured base pavement < wear/stains < painted lines <
signage, _draped_group_for_texture), and an offset within that band.
Outside `markings` the offset comes from _family_key footprint-area
ranking. Inside `markings`, order is decided by geometry in three tiers:
a solid colour bed pinned to the bottom (-5); everything else split by
connected-component size (_component_size_split) into a STRUCTURE part
(ranked -4..+4 by area) and a GLYPH part (small compact components --
text, arrowheads -- pinned to +5), so both light-text-on-dark and
dark-text-on-light markings come out right. No source-model or texture
names are hard-coded; it's colour-suffix/component-size/material-word
heuristics that hold across MSFS ground-poly packages generally.
"""

import dataclasses
import math
import os
import pickle
import re
from pathlib import Path

import numpy as np

import pol_writer
from geo_transform import local_offset_to_latlon, rotate_xz, metres_per_degree
from mesh_convert import mesh_ir

# Default tolerance for _weld_vertices. The merge pipeline no longer calls
# _weld_vertices at all -- same-texture pieces are POOLED, not snapped
# together, and pavement gaps are closed afterwards with outward boundary
# skirts (see _fill_pavement_gaps) -- but the function is kept as a
# general, unit-tested mesh utility and this is its default eps.
_MERGE_WELD_EPS_M = 0.20

# --- Draped draw-band routing -----------------------------------------------
# X-Plane's ATTR_layer_group_draped gives only 11 draw slots (-5..+5)
# within a single group -- a dense airport tile easily has dozens of
# distinct draped materials, overflowing into a coarse area-magnitude
# bucket with no draw-order guarantee. Routing each layer into one of
# five ordered draped bands by what it physically is -- solid base-
# colour fill under textured pavement under wear/stains under painted
# lines under signage -- multiplies the usable slots and makes the stack
# order systematic. Listed bottom-to-top in X-Plane's own draw order.
_DRAPED_GROUP_GROUND = "shoulders"  # solid black/white "ground" base-colour fills
_DRAPED_GROUP_BASE = "taxiways"     # textured asphalt/concrete/apron fills
_DRAPED_GROUP_WEAR = "runways"      # dirt/cracks/stains/tire marks/decals
_DRAPED_GROUP_LINES = "markings"    # painted lines & markings (the default)
_DRAPED_GROUP_SIGNS = "airports"    # painted signage / text / logos / banners
_DRAPED_GROUP_ORDER = (
    _DRAPED_GROUP_GROUND, _DRAPED_GROUP_BASE, _DRAPED_GROUP_WEAR,
    _DRAPED_GROUP_LINES, _DRAPED_GROUP_SIGNS)

# MSFS ground-poly systems paint a solid-colour "ground" polygon (a 1-px
# black or white image tiled huge) as the very base, then blend textured
# pavement on top. X-Plane has no blend, so if the solid fill lands ABOVE
# the textured pavement it shows as flat black / bright-white slabs over
# the apron -- confirmed as the "brighter tiles" report. It has to be the
# bottom-most band so the real pavement covers it.
_GROUND_FILL_KEYWORDS = ("blackground", "whiteground", "greyground", "grayground",
                         "_gp_black", "_gp_white", "baseground", "groundbase",
                         "_ground_albd", "solidground")
_SIGN_KEYWORDS = ("sign", "logo", "banner", "placard", "board", "letter", "glyph",
                  "digits", "directional", "guidance", "standtype", "stand_type",
                  "callout", "designation", "product_banner", "chkpnt", "checkpoint",
                  "_text_", "text_albd")
_BASE_KEYWORDS = ("asphalt", "concrete", "concret", "tarmac", "pavement", "apron",
                  "baselayer", "base_layer", "basecolor", "base_color", "groundpoly",
                  "ground_poly", "gp_base", "runway_base", "taxiway_base", "_tiles_",
                  "tiles_1", "_asp_", "asp_w", "asp_d", "conc_tile", "_conc_", "gravel",
                  # CONFIRMED REAL BUG these were missing for: a bare "tile"
                  # substring (not just the underscore-wrapped "_tiles_"/
                  # "tiles_1" forms above) catches real MSFS ground-poly
                  # material names like "SmallTiles" (ini_GP_GEN_SmallTiles_
                  # 4m_01) that glue straight onto another word with no
                  # underscore -- previously matched NONE of this module's
                  # keyword lists at all, silently falling through to the
                  # _DRAPED_GROUP_LINES default (the "markings" band, for
                  # painted lines/text/signage) instead of the "taxiways"
                  # base-pavement band it actually belongs in. Confirmed
                  # real symptom this caused: real tile/paver/ballast-detail
                  # ground content competing for the markings band's own
                  # narrow -4..+4 structure-tier ranking against genuine
                  # painted line markings, once mesh_convert.convert()
                  # started draping this content (see is_near_ground_flat's
                  # own comment) -- reported as pavement pieces flickering/
                  # z-fighting against each other in-sim. "paver"/"cobble"/
                  # "sett"/"ballast" cover the same real-world content
                  # class described directly by a user report: "the stones
                  # under the rail of the train".
                  "tile", "paver", "cobble", "sett", "ballast")
_WEAR_KEYWORDS = ("decal", "dirt", "crack", "stain", "leak", "grud", "grunge", "tire",
                  "skid", "wear", "damage", "oil", "rubber", "patch", "weather",
                  "mud", "puddle", "scuff", "roof", "grass", "turf", "moss", "seam",
                  "tileseam", "manhole", "gutter", "grate", "drain")


def _draped_group_for_texture(texture):
    """Which of the five ordered draped draw bands a same-texture layer
    belongs in, by keyword match on its texture file name -- see the
    _DRAPED_GROUP_* comment. Order of the checks matters: a solid
    black/white 'ground' fill is base, not a marking; 'SignDecals' is
    signage, not wear."""
    name = Path(texture).name.lower()
    if any(k in name for k in _GROUND_FILL_KEYWORDS):
        return _DRAPED_GROUP_GROUND
    if any(k in name for k in _SIGN_KEYWORDS):
        return _DRAPED_GROUP_SIGNS
    # "tileseam" must still win as WEAR (see _WEAR_KEYWORDS) despite the
    # bare "tile" keyword added to _BASE_KEYWORDS below -- without this
    # guard the more generic "tile" substring match would shadow it here,
    # since BASE is checked before WEAR.
    if any(k in name for k in _BASE_KEYWORDS) and "tileseam" not in name:
        return _DRAPED_GROUP_BASE
    if any(k in name for k in _WEAR_KEYWORDS):
        return _DRAPED_GROUP_WEAR
    return _DRAPED_GROUP_LINES


# One MSFS ground-poly material is exported as dozens of near-identical
# textures differing only in a per-placement bake -- an alpha or colour
# multiplier (shs_decal_dirt_01_albd_a87, _cf43_43_43_a96), a base-pavement
# finish variant (concretetiles_001_btint / _washed / _dark), or a line-
# marking colour variant (lines_003 vs _003y "yellow" vs _003bl "black").
# Each counted as its own draped layer easily blows past a band's 11
# offset slots, clamping everything onto one shared offset with no draw
# order. Collapsing every bake/finish/colour of one base material onto
# ONE ranking family gives the bands back their slots; within a family
# the sub-layers order by area via stream order (bigger = underneath).
_INSTANCE_BAKE_RE = re.compile(r'_(?:a\d{1,3}|cf\d{1,3}_\d{1,3}_\d{1,3})$')
_BASE_FINISH_TAGS = ("_btint", "_br", "_washed", "_old2", "_old", "_dirt", "_darkest",
                     "_dark", "_light", "_div", "_green", "_new", "_clean")
_BASE_FAMILY_HINT = ("concret", "asphalt", "tarmac", "pavement", "_tile", "asp_", "conc_")
# Marking colour / wear variants: lines_003y, lines_003bl, lines_002_y,
# lines_004_b, lines_004r, lines_005_r, lines_001_worn, lines_001_ex …
# The number stays (lines_003 and lines_004 are genuinely different
# markings); only the trailing colour/finish token is dropped.
_MARKING_FAMILY_HINT = ("line", "marking", "road_mark", "gp_lines")
_MARKING_VARIANT_RE = re.compile(
    r'(?<=\d)(?:y|bl|r|b|ry|by)(?=_albd$)'                 # glued: ..._003y_albd
    r'|_(?:y|bl|r|b|ry|by|worn|ex|old|new|dark|light)(?=_albd$)'  # separated: ..._003_y_albd
)


def _family_key(texture):
    """Ranking family for a draped texture: the base-material name with
    per-instance bakes (…_a87, …_cf43_43_43_a96), base-pavement finishes
    (…_btint, …_dark) and line-marking colour variants (…_003y, …_004_bl)
    all stripped, so every copy of one real surface takes ONE draw-order
    slot instead of overflowing the band. The distinguishing number is
    kept -- lines_003 and lines_004 stay separate families."""
    name = Path(texture).name.lower()
    for ext in (".png", ".dds", ".ktx2", ".ktx"):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    prev = None
    while prev != name:
        prev = name
        name = _INSTANCE_BAKE_RE.sub("", name)
    if any(h in name for h in _BASE_FAMILY_HINT):
        changed = True
        while changed:
            changed = False
            for tag in _BASE_FINISH_TAGS:
                if name.endswith(tag + "_albd"):
                    name, changed = name[:-len(tag) - 5] + "_albd", True
                elif name.endswith(tag):
                    name, changed = name[:-len(tag)], True
    if any(h in name for h in _MARKING_FAMILY_HINT):
        prev = None
        while prev != name:
            prev = name
            name = _MARKING_VARIANT_RE.sub("", name)
    return name


# Draw order inside the markings band is by paint COLOUR ROLE, not area
# or line number, since X-Plane gives no stream-order guarantee between
# draped OBJ8s at the same layer_group+offset. Only one rung is fixed by
# colour: a solid coloured ground bed (a red keep-clear box, a green
# strip) is always the bottom, since ranking it by area alone let it land
# on top of the (fewer-triangle) white crosswalk bars it should sit
# under. Everything else orders by footprint area, biggest-first =
# underneath, so a big panel sinks below its own small glyphs regardless
# of which of the two is the darker colour.
_MARKING_ROLE_BED, _MARKING_ROLE_OTHER = 0, 1
_MARKING_TAG_RE = re.compile(r'(?:_|(?<=\d))(r|ry)(?=_albd$|$)')


def _marking_colour_role(texture):
    """_MARKING_ROLE_BED for a solid coloured ground bed (red / green), else
    _MARKING_ROLE_OTHER. Reads the trailing colour token the same way
    _family_key strips it -- glued (lines_004r) or separated (lines_005_r) --
    plus the explicit words red/green."""
    name = Path(texture).name.lower()
    for ext in (".png", ".dds", ".ktx2", ".ktx"):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    name = _INSTANCE_BAKE_RE.sub("", name)
    if ("green" in name or "_grn" in name or "red" in name
            or name.endswith(("r_albd", "_r_albd"))
            or _MARKING_TAG_RE.search(name)):
        return _MARKING_ROLE_BED
    return _MARKING_ROLE_OTHER


def _rank_band_by_family(band_area_by_stem, family_by_stem):
    """Assign every draped stem in ONE draw band an ATTR_layer_group_draped
    offset. Stems collapse into ranking families (_family_key) FIRST -- so
    the ~50 per-placement bakes of one dirt decal, or the retint copies of
    one apron, count as a single layer, not 50 -- then the families are
    ordered by total footprint area (largest first == drawn first ==
    underneath) and handed one offset each, -5, -4, …, clamped to +5 once a
    band has more than 11 families (the small top overlays that clamp there
    are the least damaging place to run out). Every member of a family
    shares that family's offset. Used for every band EXCEPT `markings`,
    which is ranked per-texture by footprint area (a coloured bed forced to
    the bottom) so colour variants of one marking keep distinct slots --
    see the call site."""
    if not band_area_by_stem:
        return {}
    fam_area = {}
    fam_members = {}
    for stem, area in band_area_by_stem.items():
        fam = family_by_stem.get(stem, stem)
        fam_area[fam] = fam_area.get(fam, 0.0) + area
        fam_members.setdefault(fam, []).append(stem)
    out = {}
    for slot_i, fam in enumerate(sorted(fam_area, key=lambda f: fam_area[f], reverse=True)):
        offset = max(-5, min(5, -5 + slot_i))
        for stem in fam_members[fam]:
            out[stem] = offset
    return out

# Safety cap; huge groups are skipped (left as separately placed, unwelded
# objects) rather than merged, to leave a ceiling against a pathological
# package (e.g. thousands of accidental duplicate placements) while
# staying well above any real single texture group's vertex count --
# _weld_vertices is a spatial-hash + union-find pass, not O(n^2), so cost
# scales close to linearly with vertex count for reasonably-distributed
# data.
_MAX_MERGED_VERTS = 1_000_000


def _load_candidate(obj_dir: Path, obj_stem: str):
    """Returns the loaded MeshIR if this stem is a genuine draped merge
    candidate (has a sidecar, is draped, has real geometry), else None."""
    sidecar = mesh_ir.sidecar_path_for(obj_dir / f"{obj_stem}.obj")
    if not sidecar.exists():
        return None
    try:
        ir = mesh_ir.load(sidecar)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    if not ir.draped or not len(ir.positions) or not ir.texture:
        return None
    return ir


def _member_point_to_tile_frame(local_x, local_z, member_lat, member_lon, heading_deg,
                                tile_lat, tile_lon, m_lat, m_lon):
    """One member vertex's local metres -> metres in a frame anchored at
    (tile_lat, tile_lon), using ONE pair of WGS84-accurate metres/degree
    (m_lat, m_lon, computed once for EVERY merged draped layer in the tile,
    near the mean latitude of all of them) for BOTH the member-anchor
    placement and the tile frame.

    The old path went local -> lat/lon (scaled by cos(member_lat)) ->
    tile-frame metres (scaled by cos(tile_lat)). tile_lat is math.floor()
    of the real latitude, so those two cos values were ~0.5deg apart at
    47N -- a ~0.5% east/west stretch, on top of the ~0.35% the flat
    EARTH_M_PER_DEG constant was already short by. Together that walked a
    merged taxi-line / marking layer visibly off the geodetically-placed
    default markings, worse the further a vertex sat from the layer centre.
    X-Plane linearises each placed object's tangent plane at its own anchor
    with full WGS84, so a single consistent WGS84 scale here matches it."""
    rx, rz = rotate_xz(local_x, local_z, heading_deg)
    real_lat = member_lat - rz / m_lat
    real_lon = member_lon + rx / m_lon
    return (real_lon - tile_lon) * m_lon, -(real_lat - tile_lat) * m_lat


def _tile_frame_to_latlon(tile_lat, tile_lon, px, pz, m_lat, m_lon):
    """Inverse of _member_point_to_tile_frame's framing, with the SAME
    (m_lat, m_lon) so the round trip is exact."""
    return tile_lat - pz / m_lat, tile_lon + px / m_lon


def _triangles_to_polygons(pol_path, base_lat, base_lon, heading_deg, positions, uvs, indices,
                           frame_m=None):
    """Serializes a draped triangle soup into dsf_compiler.build_dsf's own
    "polygons" format: one {"pol_path": ..., "points": [(lon,lat,s,t) x3]}
    dict per triangle.

    frame_m=(m_lat, m_lon): the positions are already in the
    (base_lat, base_lon)-anchored tile frame a merge produced (see
    _member_point_to_tile_frame) -- convert them back with
    _tile_frame_to_latlon and the SAME tile-wide WGS84 metres/degree every
    merged layer in the tile shares.
    frame_m=None: base_lat/base_lon/heading_deg is a single unmerged draped
    object's own placement and `positions` its untransformed local mesh --
    the legacy local_offset_to_latlon path, no tile-frame round trip.

    Reuses the SAME already-welded/deduped/gap-closed triangle soup
    merge_draped_layers_in_tile always produces (for merges) unchanged --
    no separate boundary-ring/hole classification needed, since an absence
    of triangles here is already just an absence of DSF polygon primitives
    there, the direct polygon analog of write_obj8's own IDX-per-triangle
    stream. Y (elevation) is intentionally never read: draped content is
    always Y-zeroed upstream (convert.py's own draped-Y-zeroing pass) and a
    DRAPED_POLYGON has no elevation field of its own at all -- X-Plane
    projects it onto real sampled terrain at render time, the entire point
    of using this primitive over an OBJ8 draped object."""
    n_tris = len(indices) // 3
    polygons = []
    for t in range(n_tris):
        points = []
        for k in range(3):
            vi = int(indices[3 * t + k])
            px, pz = float(positions[vi, 0]), float(positions[vi, 2])
            if frame_m is None:
                lat, lon = local_offset_to_latlon(base_lat, base_lon, heading_deg, px, pz)
            else:
                lat, lon = _tile_frame_to_latlon(base_lat, base_lon, px, pz, frame_m[0], frame_m[1])
            points.append((lon, lat, float(uvs[vi, 0]), float(uvs[vi, 1])))
        polygons.append({"pol_path": pol_path, "points": points})
    return polygons


def _vertex_components(n_verts, indices):
    """Union-find over triangle edges: returns an int array `label` of
    length n_verts where label[i] == label[j] iff vertices i and j are
    joined by a chain of shared triangle edges. Only the partition matters
    to callers (they compare labels for equality), not which member is the
    root."""
    parent = list(range(n_verts))

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for t in range(len(indices) // 3):
        a, b, c = int(indices[3 * t]), int(indices[3 * t + 1]), int(indices[3 * t + 2])
        union(a, b)
        union(b, c)
    return np.array([find(i) for i in range(n_verts)], dtype=np.int64)


def _weld_vertices(positions, normals, uvs, indices, eps=_MERGE_WELD_EPS_M, only_indices=None,
                   uv_eps=None):
    """Snaps near-duplicate positions (within eps, real Euclidean distance)
    to a single shared vertex, remapping every triangle index accordingly.

    uv_eps, if given, additionally requires the two vertices' UVs to be
    within uv_eps of each other before they weld. A weld should close a
    seam between two pieces of one continuous surface (UVs match, or
    nearly, at the shared edge) or drop an exact duplicate, not fuse two
    things that merely sit near each other -- on a MASK/BLEND marking
    atlas, welding position-only can yank a small glyph's UV across the
    atlas and smear it. On an opaque tiling pavement a UV step at a weld
    is invisible, so callers leave uv_eps None there.

    Union-find over a spatial hash, checking each point's own quantized
    grid cell and all 26 neighboring cells (3x3x3), not just its own --
    two points closer than eps can still round to different adjacent
    cells if they straddle a grid line, which checking only the same
    cell would miss.

    only_indices, if given, restricts candidacy to that subset of vertex
    indices (both bucketing and comparison), so an excluded vertex can
    never end up unioned with anything, no matter how loose eps is --
    for callers that want a tolerance-raised weld restricted to a known-
    safe subset (e.g. only naked/boundary vertices).

    Never unions two vertices already mesh-connected (joined by a chain
    of shared triangle edges) before this pass runs, regardless of
    distance -- otherwise a real object smaller than eps in every
    direction (a short paint dash, a small decal) has its own opposite
    corners within eps of each other and a naive distance-only check
    collapses it to a degenerate point. Two corners of the same original
    polygon are always mesh-connected already; two facing corners of
    genuinely separate pieces (the actual weld target) never are, so
    gating on pre-existing connectivity makes a small object's self-
    collapse structurally impossible rather than just unlikely.
    """
    n = len(positions)
    if n == 0:
        return positions, normals, uvs, indices

    starting_component = _vertex_components(n, indices)

    candidate_indices = range(n) if only_indices is None else only_indices
    cell = np.round(positions / eps).astype(np.int64)
    buckets = {}
    for i in candidate_indices:
        buckets.setdefault((cell[i, 0], cell[i, 1], cell[i, 2]), []).append(i)

    parent = list(range(n))

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    eps2 = eps * eps
    uv_eps2 = None if uv_eps is None else uv_eps * uv_eps
    for (kx, ky, kz), members in buckets.items():
        neighbor_indices = []
        for dx, dy, dz in offsets:
            neighbor_indices.extend(buckets.get((kx + dx, ky + dy, kz + dz), []))
        for i in members:
            pi = positions[i]
            for j in neighbor_indices:
                if j <= i:
                    continue
                if starting_component[i] == starting_component[j]:
                    continue
                d = positions[j] - pi
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > eps2:
                    continue
                if uv_eps2 is not None:
                    du = uvs[j] - uvs[i]
                    if du[0] * du[0] + du[1] * du[1] > uv_eps2:
                        continue
                union(i, j)

    # Every pairwise union() call above passes (i, j) with i < j and keeps
    # find(i)'s root as the survivor, so by induction the final root of any
    # connected component is always its lowest original index -- i.e.
    # unique_roots (ascending, since np.unique sorts) already IS "first-
    # seen vertex in original order keeps its own normal/uv", matching the
    # documented contract with no extra bookkeeping needed.
    roots = np.array([find(i) for i in range(n)])
    unique_roots, inverse = np.unique(roots, return_inverse=True)

    new_positions = positions[unique_roots]
    new_normals = normals[unique_roots]
    new_uvs = uvs[unique_roots]
    new_indices = inverse[indices]
    return new_positions, new_normals, new_uvs, new_indices


def _boundary_edges(indices):
    """Returns {(min_vertex_idx, max_vertex_idx): use_count} for every
    undirected edge across all triangles. An edge used by exactly one
    triangle is a naked/boundary edge -- either the true outer edge of the
    whole merged surface, or an unclosed gap between two pieces that should
    have met there but didn't quite."""
    n_tris = len(indices) // 3
    counts = {}
    for t in range(n_tris):
        a, b, c = int(indices[3 * t]), int(indices[3 * t + 1]), int(indices[3 * t + 2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _mesh_surface_area(positions, indices):
    """Real summed triangle area (XZ plane -- draped content is always
    Y-zeroed), not a bounding-box approximation. A merge group's own
    bbox span is a reasonable stand-in for footprint on one contiguous
    surface, but merge groups here form by SHARED TEXTURE across a
    whole tile, not proximity -- a decal atlas reused on dozens of
    small scattered objects welds into one group whose bbox spans
    nearly the airport's full extent despite each patch being tiny,
    which would win the largest-first ranking outright. Real summed
    triangle area doesn't have this failure mode: scattered small
    patches sum to a small real total regardless of how far apart they
    are."""
    if len(indices) == 0:
        return 0.0
    tri = indices.reshape(-1, 3)
    p0, p1, p2 = positions[tri[:, 0]], positions[tri[:, 1]], positions[tri[:, 2]]
    e1 = p1[:, [0, 2]] - p0[:, [0, 2]]
    e2 = p2[:, [0, 2]] - p0[:, [0, 2]]
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return float(np.abs(cross).sum() / 2.0)


def _dedupe_faces(positions, normals, uvs, indices):
    """Removes, after _weld_vertices:
      * DEGENERATE triangles -- fewer than 3 distinct vertex indices. Even a
        tight weld can pull two corners of one triangle onto the same point
        transitively (corner A welds to a vertex X of another piece, corner
        C welds to X too), collapsing the triangle to a sliver/line that
        renders as nothing or a stray edge.
      * EXACT-DUPLICATE triangles -- same 3 welded indices, any winding: two
        independently-authored objects (e.g. a base apron polygon exported
        twice, once from the package BGL and once from a duplicate placement
        record) covering the exact same footprint with the same texture.
        Two fully-coincident same-material draped layers z-fight (flicker
        between the two surfaces) and can read as a "hole" in the pavement.

    Since _weld_vertices already snapped near-duplicate positions onto ONE
    shared vertex, a true duplicate references the identical 3 post-weld
    indices -- a plain index-set dedup catches it, no distance/area test.
    Returns (positions, normals, uvs, indices, removed_triangle_count)."""
    n_tris = len(indices) // 3
    if n_tris == 0:
        return positions, normals, uvs, indices, 0

    tri_idx = indices.reshape(n_tris, 3)
    seen = set()
    keep_tri = []
    for t in range(n_tris):
        a, b, c = int(tri_idx[t, 0]), int(tri_idx[t, 1]), int(tri_idx[t, 2])
        if a == b or b == c or a == c:
            continue  # degenerate -- collapsed by the weld
        key = tuple(sorted((a, b, c)))
        if key in seen:
            continue
        seen.add(key)
        keep_tri.append(t)

    removed = n_tris - len(keep_tri)
    if removed == 0:
        return positions, normals, uvs, indices, 0

    kept_indices = tri_idx[keep_tri].reshape(-1)
    used = np.unique(kept_indices)
    remap = np.full(len(positions), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))

    new_positions = positions[used]
    new_normals = normals[used]
    new_uvs = uvs[used]
    new_indices = remap[kept_indices]
    return new_positions, new_normals, new_uvs, new_indices, removed


# --- markings: glyph / structure split ------------------------------------
# The reliable, colour-blind signal for "text/legend painted ON a panel"
# vs. "the panel/line/bed itself" is GEOMETRY, not footprint area: a
# background panel, coloured bed or line network is one (or a few) large
# connected components; painted characters/numbers/arrowheads are many
# small compact ones. Ranking by total texture area alone flips depending
# on whether that texture is used as a big box or small text elsewhere in
# the same tile. So every markings merge is split by connected-component
# size into a structure sub-mesh (ranked low) and a glyph sub-mesh (pinned
# to the top slot), which holds regardless of which colour is which.
_GLYPH_MAX_AREA_M2 = 4.0
_GLYPH_MAX_DIAG_M = 6.0


def _component_size_split(positions, normals, uvs, indices):
    """Partition a triangle mesh into (structure, glyph) sub-meshes by
    connected-component size. glyph = components small in BOTH area
    (<=_GLYPH_MAX_AREA_M2) and extent (bbox diagonal <=_GLYPH_MAX_DIAG_M);
    structure = everything else. Either side may be None. Each non-None
    value is a (positions, normals, uvs, indices) tuple."""
    n = len(positions)
    if n == 0 or len(indices) == 0:
        return (positions, normals, uvs, indices), None
    comp = _vertex_components(n, indices)
    tri = indices.reshape(-1, 3)
    tri_comp = comp[tri[:, 0]]
    xz = positions[:, [0, 2]]

    glyph_comps = set()
    for c in np.unique(tri_comp):
        ct = tri[tri_comp == c]
        p0, p1, p2 = xz[ct[:, 0]], xz[ct[:, 1]], xz[ct[:, 2]]
        area = float(np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                            - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])).sum() / 2.0)
        cv = xz[comp == c]
        diag = float(np.hypot(*(cv.max(axis=0) - cv.min(axis=0)))) if len(cv) else 0.0
        if area <= _GLYPH_MAX_AREA_M2 and diag <= _GLYPH_MAX_DIAG_M:
            glyph_comps.add(int(c))

    if not glyph_comps:
        return (positions, normals, uvs, indices), None
    glyph_tri = np.array([int(c) in glyph_comps for c in tri_comp])
    if glyph_tri.all():
        return None, (positions, normals, uvs, indices)

    def _take(mask):
        keep = tri[mask].reshape(-1)
        used = np.unique(keep)
        remap = np.full(n, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return positions[used], normals[used], uvs[used], remap[keep]

    return _take(~glyph_tri), _take(glyph_tri)


def _glyph_area_fraction(positions, indices):
    """Fraction of total triangle area sitting in glyph-sized connected
    components (small in BOTH area and extent). Used to classify a markings
    object that is NOT split (a single, no merge partner) as text-or-not."""
    n = len(positions)
    if n == 0 or len(indices) == 0:
        return 0.0
    comp = _vertex_components(n, indices)
    tri = indices.reshape(-1, 3)
    tri_comp = comp[tri[:, 0]]
    xz = positions[:, [0, 2]]
    total = glyph = 0.0
    for c in np.unique(tri_comp):
        ct = tri[tri_comp == c]
        p0, p1, p2 = xz[ct[:, 0]], xz[ct[:, 1]], xz[ct[:, 2]]
        a = float(np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                         - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])).sum() / 2.0)
        cv = xz[comp == c]
        diag = float(np.hypot(*(cv.max(axis=0) - cv.min(axis=0)))) if len(cv) else 0.0
        total += a
        if a <= _GLYPH_MAX_AREA_M2 and diag <= _GLYPH_MAX_DIAG_M:
            glyph += a
    return glyph / total if total > 1e-9 else 0.0


# --- Pavement underlay: seal the draped-render cracks with one solid sheet --
# The apron is a stack of separate draped pavement objects, each draped
# onto the terrain mesh independently, so a hairline crack opens along
# the boundary between any two of them, or along a tile boundary that
# wasn't index-welded -- where terrain (grass green) shows through. The
# seam welds (_weld_component_gaps within an object, _snap_pavement_seams
# across objects) close most of these, but not all.
#
# _fill_pavement_gaps is the safety net: one opaque underlay under the
# whole paved footprint, at the bottom draped band, on a neutral concrete
# texture. It rasterises the union of all pavement coverage, morphologically
# CLOSES it by _PAVEMENT_FILL_CLOSE_M (fills every notch/seam <= 2*close),
# and emits the closed-minus-original region as greedy-rectangle quads. A
# real grass opening wider than ~2*close stays open. Runs once per tile
# after ranking is frozen, as its own hard-pinned bottom-band object that
# never re-enters ranking. Toggle off with MSFS2XP_BRIDGE_SEAMS=0.
_PAVEMENT_FILL_CLOSE_M = 0.80    # morphological-close radius (seals seams/cracks)
_PAVEMENT_FILL_RES_M = 0.40      # raster cell size (processed in 600 m chunks)
_PAVEMENT_FILL_MAX_TRIS = 3_000_000
_PAVEMENT_FILL_BANDS = (_DRAPED_GROUP_GROUND, _DRAPED_GROUP_BASE, _DRAPED_GROUP_WEAR)


def _reach_dir(m, n, axis, forward):
    """A cell is True iff `m` is True within n cells of it in the given
    direction (axis 0 = Z/rows, 1 = X/cols; forward = toward higher index).
    Cumulative -- n passes of a 1-cell shift-OR."""
    o = m.copy()
    for _ in range(int(n)):
        if axis == 1 and forward:
            o[:, 1:] |= o[:, :-1]
        elif axis == 1:
            o[:, :-1] |= o[:, 1:]
        elif forward:
            o[1:, :] |= o[:-1, :]
        else:
            o[:-1, :] |= o[1:, :]
    return o


def _raster_tris_xz(xz_tris, x0, z0, res, W, H, out):
    """OR every triangle in xz_tris ((n,3,2), XZ metres) into the boolean
    grid `out` (H rows = Z, W cols = X) wherever it covers a cell centre."""
    ax = xz_tris[:, 0, 0]; az = xz_tris[:, 0, 1]
    bx = xz_tris[:, 1, 0]; bz = xz_tris[:, 1, 1]
    cx = xz_tris[:, 2, 0]; cz = xz_tris[:, 2, 1]
    lo_x = np.clip(np.floor((np.minimum.reduce([ax, bx, cx]) - x0) / res).astype(np.int64), 0, W)
    hi_x = np.clip(np.ceil((np.maximum.reduce([ax, bx, cx]) - x0) / res).astype(np.int64) + 1, 0, W)
    lo_z = np.clip(np.floor((np.minimum.reduce([az, bz, cz]) - z0) / res).astype(np.int64), 0, H)
    hi_z = np.clip(np.ceil((np.maximum.reduce([az, bz, cz]) - z0) / res).astype(np.int64) + 1, 0, H)
    for k in range(len(xz_tris)):
        j0, j1, i0, i1 = int(lo_x[k]), int(hi_x[k]), int(lo_z[k]), int(hi_z[k])
        if j0 >= j1 or i0 >= i1:
            continue
        gx = ((np.arange(j0, j1) + 0.5) * res + x0)[None, :]
        gz = ((np.arange(i0, i1) + 0.5) * res + z0)[:, None]
        d1 = (gx - bx[k]) * (az[k] - bz[k]) - (ax[k] - bx[k]) * (gz - bz[k])
        d2 = (gx - cx[k]) * (bz[k] - cz[k]) - (bx[k] - cx[k]) * (gz - cz[k])
        d3 = (gx - ax[k]) * (cz[k] - az[k]) - (cx[k] - ax[k]) * (gz - az[k])
        inside = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
        out[i0:i1, j0:j1] |= inside


def _mask_to_quads(mask, x0, z0, res):
    """Greedy rectangle decomposition of a boolean mask -> list of
    (wx0, wz0, wx1, wz1) axis-aligned world rectangles that exactly cover
    the set cells. Keeps the emitted triangle count tiny (a 1 m x 200 m
    stripe becomes a few rects, not thousands of pixels)."""
    H, W = mask.shape
    work = mask.copy()
    rects = []
    ys, xs = np.nonzero(work)
    if len(ys) == 0:
        return rects
    for i in range(ys.min(), ys.max() + 1):
        row = work[i]
        j = 0
        while j < W:
            if not row[j]:
                j += 1
                continue
            k = j
            while k < W and row[k]:
                k += 1
            h = 1
            while i + h < H and work[i + h, j:k].all():
                h += 1
            work[i:i + h, j:k] = False
            rects.append((x0 + j * res, z0 + i * res, x0 + k * res, z0 + (i + h) * res))
            j = k
    return rects


_PAVEMENT_SEAM_WELD_EPS_M = 0.25   # boundary-vertex weld radius for base pavement
# The apron concrete strips are OFFSET, not merely un-indexed: measured on the
# real export the gaps between adjacent concretetiles_001 strips run 0.3-1.2 m
# (median 0.57, p90 1.2). 1.2 m catches ~91 % of them; the wider ones left are
# deliberate expansion channels. A boundary vertex moves at most eps onto its
# neighbour's edge -- on a ~15 m strip that is a <=8 % edge shift on an
# otherwise-uniform grid, invisible on tiling concrete and far less ugly than
# the terrain-through line it removes.


def _naked_vertex_mask(indices, n):
    """Boolean mask of vertices that touch a NAKED edge -- one that appears in
    exactly one triangle (a free/boundary edge, not an interior shared one)."""
    tri = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if not len(tri):
        return np.zeros(n, dtype=bool)
    ek = np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1)
    es = ek[np.lexsort((ek[:, 1], ek[:, 0]))]
    eqprev = np.all(es[1:] == es[:-1], axis=1)
    dup = np.zeros(len(es), dtype=bool)
    dup[1:] |= eqprev
    dup[:-1] |= eqprev
    m = np.zeros(n, dtype=bool)
    m[es[~dup].reshape(-1)] = True
    return m


def _weld_component_gaps(positions, normals, uvs, indices, eps=_PAVEMENT_SEAM_WELD_EPS_M):
    """Close the terrain-through stripes along the tile boundaries of a
    grid-built base-pavement mesh. Measured on the real export: `asp_w_02` is
    342 separate connected components, and `concretetiles_001` has ~2600 pairs
    of naked-edge vertices from DIFFERENT components sitting at the SAME
    position but never index-merged -- X-Plane's draped renderer projects the
    two components onto terrain independently and a sub-pixel crack opens
    along every such shared edge (the regular green grid).

    Fix: an actual INDEX weld (via _weld_vertices) of the BOUNDARY vertices
    only, eps ~0.3 m, no UV gate (a UV step on tiling pavement is invisible).
    Merging the coincident boundary verts joins the tiles into one connected
    mesh, so there is no longer an inter-component edge to crack. Interior
    vertices are excluded (`only_indices`), and _weld_vertices never unions
    two already mesh-connected vertices, so a tile can't collapse onto itself.
    Triangle count and winding are unchanged -> footprint area /
    _component_size_split / ranking are unaffected.
    Returns (positions, normals, uvs, indices, n_welded_out)."""
    P = np.asarray(positions, dtype=np.float64)
    n = len(P)
    if n < 2 or indices is None or len(indices) == 0:
        return positions, normals, uvs, indices, 0
    naked = _naked_vertex_mask(indices, n)
    if naked.sum() < 2:
        return positions, normals, uvs, indices, 0
    nP, nN, nU, nI = _weld_vertices(
        P, np.asarray(normals), np.asarray(uvs), np.asarray(indices),
        eps=eps, only_indices=np.nonzero(naked)[0].tolist(), uv_eps=None)
    # Merging a sliver triangle's two boundary corners (directly, or via two
    # verts that were already coincident) collapses it to zero area -- drop
    # those so a single keeps no cruft (a merge re-runs _dedupe_faces_geom
    # anyway).
    tri = nI.reshape(-1, 3)
    a, b, c = nP[tri[:, 0]], nP[tri[:, 1]], nP[tri[:, 2]]
    ar2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 2] - a[:, 2])
                 - (b[:, 2] - a[:, 2]) * (c[:, 0] - a[:, 0]))
    keep = ar2 > 2e-6
    if not keep.all():
        kept = tri[keep].reshape(-1)
        used = np.unique(kept)
        remap = np.full(len(nP), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        nP, nN, nU, nI = nP[used], nN[used], nU[used], remap[kept]
    return nP, nN, nU, nI, int(n - len(nP))


_PAVEMENT_CROSS_SNAP_EPS_M = 0.20   # cross-OBJECT boundary snap radius (see _PAVEMENT_SEAM_WELD_EPS_M)


def _snap_pavement_seams(pieces, eps=_PAVEMENT_CROSS_SNAP_EPS_M):
    """Make the shared edges between SEPARATE opaque-pavement objects bit-
    identical, so X-Plane's draped renderer projects both sides onto the same
    terrain point and the hairline green crack between them closes. `pieces`
    is a list of dicts {"xz": (n,2) float64 world XZ, "idx": (m,) int}. It
    POSITION-snaps -- never merges an index, never moves a vertex more than
    eps, never touches an interior (shared) vertex -- so each object stays its
    own file and its triangle set / winding / footprint area are unchanged
    (ranking, which is already frozen before this runs, is unaffected).

    For each cross-piece pair of NAKED vertices within eps, the higher
    (piece, vertex) index snaps onto the lower one's ORIGINAL position;
    iterated so a short a->b->c chain settles. Two vertices of the SAME piece
    are never snapped together (would need a within-object weld, done
    already). Mutates each piece's "xz" in place; returns n_snapped."""
    parts = [p for p in pieces if len(p.get("xz", ())) and len(p.get("idx", ()))]
    if len(parts) < 2:
        return 0

    naked_of = []
    comp_of = []
    for p in parts:
        P = np.asarray(p["xz"], dtype=np.float64)
        idx = np.asarray(p["idx"], dtype=np.int64)
        naked_of.append(_naked_vertex_mask(idx, len(P)))
        # connected components (so a chain can't fold one piece into itself)
        tri = idx.reshape(-1, 3)
        par = np.arange(len(P))

        def _f(x, par=par):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for a, b in np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])):
            ra, rb = _f(int(a)), _f(int(b))
            if ra != rb:
                par[ra] = rb
        comp_of.append(np.array([_f(i) for i in range(len(P))], dtype=np.int64))

    # flat pool of (piece, vertex) for every naked vertex, with a global id
    gid = []
    for pi, p in enumerate(parts):
        for vi in np.nonzero(naked_of[pi])[0]:
            gid.append((pi, int(vi)))
    if len(gid) < 2:
        return 0
    G = len(gid)
    gp = np.array([g[0] for g in gid])
    gv = np.array([g[1] for g in gid])
    P0 = np.array([parts[gp[k]]["xz"][gv[k]] for k in range(G)], dtype=np.float64)

    e2 = eps * eps
    cur = P0.copy()
    for _round in range(4):
        cell = eps
        keys = np.floor(cur / cell).astype(np.int64)
        bucket = {}
        for k in range(G):
            bucket.setdefault((int(keys[k, 0]), int(keys[k, 1])), []).append(k)
        target = np.full(G, -1, dtype=np.int64)
        tbest = np.full(G, e2 + 1.0, dtype=np.float64)
        for (bx, bz), members in bucket.items():
            cand = []
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cand.extend(bucket.get((bx + dx, bz + dz), ()))
            for ki in members:
                pi = gp[ki]
                xi, zi = cur[ki]
                for kj in cand:
                    if kj == ki or gp[kj] == pi:
                        continue
                    d2 = (cur[kj, 0] - xi) ** 2 + (cur[kj, 1] - zi) ** 2
                    if d2 <= e2 and d2 < tbest[ki]:
                        tbest[ki] = d2
                        target[ki] = kj
        mv = (target >= 0) & (target < np.arange(G))
        if not mv.any():
            break
        nxt = cur.copy()
        nxt[mv] = cur[target[mv]]
        moved = np.hypot(nxt[:, 0] - P0[:, 0], nxt[:, 1] - P0[:, 1])
        nxt[moved > eps + 1e-6] = cur[moved > eps + 1e-6]
        if np.allclose(nxt, cur):
            break
        cur = nxt

    n_snap = 0
    for k in range(G):
        if not np.array_equal(cur[k], P0[k]):
            parts[gp[k]]["xz"][gv[k]] = cur[k]
            n_snap += 1
    return n_snap


def _dedupe_faces_geom(positions, normals, uvs, indices, min_area=1e-6):
    """Position-keyed face cleanup (works with or without a prior vertex
    weld, unlike _dedupe_faces which relies on shared indices): drop
    * triangles smaller than min_area in the XZ plane (slivers / degenerate),
    * exact-duplicate triangles -- the same three vertex POSITIONS (rounded
      to 1e-4 m) AND the same three UVs (rounded to 1e-4), any winding: a
      genuine double-placed polygon that would z-fight or read as a hole.
      UV is part of the key so two coincident-footprint quads that sample
      DIFFERENT parts of a marking atlas (a panel + a legend on it) are
      kept, not collapsed to one.
    Returns (positions, normals, uvs, indices, removed_triangle_count)."""
    idx = np.asarray(indices, dtype=np.int64)
    n_tris = len(idx) // 3
    if n_tris == 0:
        return positions, normals, uvs, idx, 0
    P = np.asarray(positions, dtype=np.float64)
    U = np.asarray(uvs, dtype=np.float64)
    q = np.round(P[:, [0, 2]] / 1e-4).astype(np.int64)
    qu = np.round(U / 1e-4).astype(np.int64) if len(U) == len(P) else np.zeros((len(P), 2), np.int64)
    tri = idx.reshape(n_tris, 3)
    seen = set()
    keep = []
    for t in range(n_tris):
        a, b, c = int(tri[t, 0]), int(tri[t, 1]), int(tri[t, 2])
        ax, az = P[a, 0], P[a, 2]
        area2 = abs((P[b, 0] - ax) * (P[c, 2] - az) - (P[b, 2] - az) * (P[c, 0] - ax))
        if area2 < 2.0 * min_area:
            continue
        key = tuple(sorted(((int(q[a, 0]), int(q[a, 1]), int(qu[a, 0]), int(qu[a, 1])),
                            (int(q[b, 0]), int(q[b, 1]), int(qu[b, 0]), int(qu[b, 1])),
                            (int(q[c, 0]), int(q[c, 1]), int(qu[c, 0]), int(qu[c, 1])))))
        if key in seen:
            continue
        seen.add(key)
        keep.append(t)
    removed = n_tris - len(keep)
    if removed == 0:
        return positions, normals, uvs, idx, 0
    kept = tri[keep].reshape(-1)
    used = np.unique(kept)
    remap = np.full(len(P), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return (P[used], np.asarray(normals)[used], np.asarray(uvs)[used], remap[kept], removed)


def _dilate_bin(m, r):
    """Binary dilation by r cells (4 axis-aligned shift-ORs, r times)."""
    o = m
    for _ in range(int(r)):
        o = (o | np.pad(o, ((1, 0), (0, 0)))[:-1] | np.pad(o, ((0, 1), (0, 0)))[1:]
             | np.pad(o, ((0, 0), (1, 0)))[:, :-1] | np.pad(o, ((0, 0), (0, 1)))[:, 1:])
    return o


def _pick_underlay_texture(parts):
    """Used by _fill_pavement_gaps to pick a texture for its bottom-band
    pavement underlay from `parts` (dicts with "texture"/"alpha"/"group"/
    "area"/"positions"), avoiding two kinds that would make the underlay
    itself visible where it shows through a gap -- a BLEND layer's albedo
    (often alpha~0 -> alpha-tested away entirely) and a flat GROUND-band
    solid-colour fill (reads as a black/white sliver). Prefer a BASE-band
    (textured asphalt/concrete) OPAQUE layer, then any non-BLEND non-
    GROUND layer, then any non-BLEND, then anything; within the chosen
    tier pick the largest by real footprint area, then vertex count.
    Returns None if `parts` is empty."""
    def _tex_rank(p):
        return (p.get("area") or 0.0, len(p["positions"]))
    for _pred in (lambda p: p.get("alpha") != "BLEND" and p.get("group") == _DRAPED_GROUP_BASE,
                  lambda p: p.get("alpha") != "BLEND" and p.get("group") != _DRAPED_GROUP_GROUND,
                  lambda p: p.get("alpha") != "BLEND",
                  lambda p: True):
        _tex_cand = [p for p in parts if _pred(p)]
        if _tex_cand:
            return max(_tex_cand, key=_tex_rank)["texture"]
    return None


def _fill_pavement_gaps(pieces, log=lambda *_a, **_k: None):
    """pieces: every pavement slab already in the shared tile frame, each
        {"texture", "alpha", "positions" (n,3), "indices" (m,), "area"}.

    Builds ONE solid opaque UNDERLAY under the whole paved footprint: rasterise
    the union of ALL pavement coverage (opaque AND blend -- the apron outline)
    in ~600 m chunks, morphologically CLOSE it by _PAVEMENT_FILL_CLOSE_M to
    seal every hairline seam / draped-render crack between the stacked layers,
    and emit the closed-minus-original region as a handful of flat quads on a
    neutral concrete texture at the very bottom draped band. Nothing above
    moves; a crack now shows grey concrete, not green terrain. A real grass
    opening wider than ~2*close stays open (the close can't bridge it); its
    rim gains at most a `close`-wide grey fringe.
    Returns [{"texture", "positions", "normals", "uvs", "indices"}] (0 or 1)."""
    parts = [p for p in pieces if len(p.get("positions", ())) and len(p.get("indices", ()))]
    if len(parts) < 2:
        return []

    tset = []
    for p in parts:
        P = np.asarray(p["positions"], dtype=np.float64)
        tri = np.asarray(p["indices"], dtype=np.int64).reshape(-1, 3)
        if not len(tri):
            continue
        xzt = P[tri][:, :, [0, 2]]
        tset.append((xzt,
                     float(xzt[:, :, 0].min()), float(xzt[:, :, 0].max()),
                     float(xzt[:, :, 1].min()), float(xzt[:, :, 1].max())))
    if len(tset) < 2:
        return []

    gx0 = min(t[1] for t in tset)
    gx1 = max(t[2] for t in tset)
    gz0 = min(t[3] for t in tset)
    gz1 = max(t[4] for t in tset)
    res = _PAVEMENT_FILL_RES_M
    r = max(1, int(round(_PAVEMENT_FILL_CLOSE_M / res)))
    chunk = 600.0
    pad = (r + 2) * res

    all_rects = []
    total_cells = 0
    cxa = int(math.floor(gx0 / chunk)); cxb = int(math.floor(gx1 / chunk))
    cza = int(math.floor(gz0 / chunk)); czb = int(math.floor(gz1 / chunk))
    for ci in range(cxa, cxb + 1):
        for cj in range(cza, czb + 1):
            ix0, ix1 = ci * chunk, (ci + 1) * chunk
            iz0, iz1 = cj * chunk, (cj + 1) * chunk
            qx0, qx1 = ix0 - pad, ix1 + pad
            qz0, qz1 = iz0 - pad, iz1 + pad
            local = [t for t in tset if t[2] > qx0 and t[1] < qx1 and t[4] > qz0 and t[3] < qz1]
            if not local:
                continue
            W = int((qx1 - qx0) / res) + 1
            H = int((qz1 - qz0) / res) + 1
            covered = np.zeros((H, W), dtype=bool)
            for xzt, *_bb in local:
                _raster_tris_xz(xzt, qx0, qz0, res, W, H, covered)
            if not covered.any():
                continue
            # morphological close: dilate r, erode r -> fills every notch /
            # seam <= 2r wide, barely extends a straight edge.
            closed = ~_dilate_bin(~_dilate_bin(covered, r), r)
            fill = closed & ~covered
            j0 = max(0, int((ix0 - qx0) / res)); j1 = min(W, int((ix1 - qx0) / res))
            i0 = max(0, int((iz0 - qz0) / res)); i1 = min(H, int((iz1 - qz0) / res))
            mask = np.zeros((H, W), dtype=bool)
            mask[i0:i1, j0:j1] = True
            fill &= mask
            if not fill.any():
                continue
            total_cells += int(fill.sum())
            all_rects.extend(_mask_to_quads(fill, qx0, qz0, res))

    if not all_rects:
        return []
    if total_cells * 2 > _PAVEMENT_FILL_MAX_TRIS:
        log(f"Pavement gap fill: {total_cells} cells over the {_PAVEMENT_FILL_MAX_TRIS}-triangle "
            f"cap; skipping.", "warning")
        return []

    # Texture for the fill: drawn opaque and entirely under real layers,
    # so its own albedo is invisible except in a gap -- but a BLEND
    # layer's texture (near-zero alpha, alpha-tested away) or a GROUND-
    # band solid-colour fill (reads as a black/white sliver) must be
    # avoided. See _pick_underlay_texture's own tier preference.
    tex = _pick_underlay_texture(parts)
    pos, uv, idx = [], [], []
    for (rx0, rz0, rx1, rz1) in all_rects:
        b = len(pos)
        pos.extend([[rx0, 0.0, rz0], [rx1, 0.0, rz0], [rx1, 0.0, rz1], [rx0, 0.0, rz1]])
        uv.extend([[rx0 / 10.0, rz0 / 10.0], [rx1 / 10.0, rz0 / 10.0],
                   [rx1 / 10.0, rz1 / 10.0], [rx0 / 10.0, rz1 / 10.0]])   # ~10 m tiling
        idx.extend([b, b + 1, b + 2, b, b + 2, b + 3])
    P = np.asarray(pos, dtype=np.float64)
    log(f"Pavement underlay: sealed {total_cells * res * res:.0f} m2 of draped-render seams/cracks "
        f"under the paved footprint with {len(all_rects)} concrete quad(s) at the bottom band "
        f"(raster {res} m, close {_PAVEMENT_FILL_CLOSE_M} m).", "info")
    return [{"texture": tex, "positions": P,
             "normals": np.tile(np.array([0.0, 1.0, 0.0]), (len(P), 1)),
             "uvs": np.asarray(uv, dtype=np.float64),
             "indices": np.asarray(idx, dtype=np.int64)}]


def merge_draped_layers_in_tile(obj_dir, tile_lat, tile_lon, entries, log_callback=None,
                                 use_polygons=False, polygons_dir=None):
    """entries: the list of placement-entry dicts for ONE DSF tile (each
    with 'name', 'lat', 'lon', 'hdg', 'agl', optionally 'footprint_area'
    and 'library_path'). Returns (new_entries, polygons) for that tile:
    draped, non-animated, ground-level (agl==0) members sharing the same
    texture are replaced by one new merged entry each; everything else
    passes through unchanged.

    use_polygons=False (the default, and the permanent fallback, not a
    temporary one): `polygons` is always empty, and every draped object
    is still written as an .obj + placement entry.

    use_polygons=True: draped content instead serializes straight into
    DSF polygon primitives (_triangles_to_polygons) referencing a real
    .pol file (pol_writer.py, memoized per texture) -- no placement entry,
    since a DSF polygon carries its own absolute lon/lat per vertex.
    `polygons` is returned sorted largest-footprint-first (DSF draws
    same-layer-group/offset polygons in CMDS stream order), keeping big
    base fills underneath small markings/signs."""
    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)

    obj_dir = Path(obj_dir)
    groups = {}
    passthrough = []

    for entry in entries:
        if entry.get("library_path") or not entry.get("name") or abs(entry.get("agl", 0.0)) >= 0.01:
            passthrough.append(entry)
            continue
        ir = _load_candidate(obj_dir, entry["name"])
        if ir is None:
            passthrough.append(entry)
            continue
        groups.setdefault(ir.texture, []).append((entry, ir))

    # Two passes: first weld every qualifying same-texture group into its
    # combined mesh and compute its footprint area without deciding an
    # offset yet, then rank every merge result's footprint area against
    # every OTHER merge result in this same tile (mesh_convert.
    # draped_ranking.rank_draped_layer_offsets) so two merged groups of
    # different textures can't collide on the same (layer_group, offset).
    pending_merges = []
    # Single (no same-texture merge partner) draped objects -- most real
    # taxi/gate signs -- collected separately from `passthrough` so they
    # can still take part in the tile-wide re-rank below instead of
    # keeping whatever offset their own source file assigned them.
    singles = []
    merge_count = 0

    # ONE WGS84 metres/degree pair for EVERY merged draped layer in this
    # tile, at the mean latitude of all their members -- not per-texture-
    # group. Two coincident bits of the same real marking (a crosswalk's
    # red bed and its white bars) are separate MSFS materials -> separate
    # merge groups here; if each recentres around its own footprint and
    # picks its own metres/degree, the layers reconstruct to slightly
    # different positions and visibly slide apart. Sharing one metres/
    # degree and one recenter centre/anchor keeps coincident content in
    # different groups landing on exactly the same coordinates.
    _all_member_lats = [e["lat"] for mem in groups.values() for e, _ in mem]
    tile_ref_lat = (sum(_all_member_lats) / len(_all_member_lats)) if _all_member_lats else (tile_lat + 0.5)
    tile_m_lat, tile_m_lon = metres_per_degree(tile_ref_lat)

    for texture, members in groups.items():
        if len(members) < 2:
            entry, ir = members[0]
            singles.append((entry, ir))
            continue

        all_positions, all_normals, all_uvs, all_indices = [], [], [], []
        for entry, ir in members:
            v_off = sum(len(p) for p in all_positions)
            heading = entry["hdg"]
            mx, mz = [], []
            for x, z in zip(ir.positions[:, 0], ir.positions[:, 2]):
                px, pz = _member_point_to_tile_frame(
                    float(x), float(z), entry["lat"], entry["lon"], heading,
                    tile_lat, tile_lon, tile_m_lat, tile_m_lon)
                mx.append(px)
                mz.append(pz)
            member_positions = np.column_stack([mx, ir.positions[:, 1], mz])

            rot_normals = np.empty_like(ir.normals)
            for i, (nx, nz) in enumerate(zip(ir.normals[:, 0], ir.normals[:, 2])):
                rnx, rnz = rotate_xz(float(nx), float(nz), heading)
                rot_normals[i, 0] = rnx
                rot_normals[i, 2] = rnz
            rot_normals[:, 1] = ir.normals[:, 1]

            all_positions.append(member_positions)
            all_normals.append(rot_normals)
            all_uvs.append(ir.uvs)
            all_indices.append(ir.indices + v_off)

        combined_positions = np.concatenate(all_positions, axis=0)
        combined_normals = np.concatenate(all_normals, axis=0)
        combined_uvs = np.concatenate(all_uvs, axis=0)
        combined_indices = np.concatenate(all_indices, axis=0)

        if len(combined_positions) > _MAX_MERGED_VERTS:
            _log(f"Tile {tile_lat},{tile_lon}: skipped merging {len(members)} objects sharing "
                 f"texture {Path(texture).name} -- combined vertex count ({len(combined_positions)}) "
                 f"is too large to merge safely; left as separately placed objects.", "warning")
            passthrough.extend(entry for entry, _ in members)
            continue

        member_alpha_modes = {ir.alpha_mode for _, ir in members}
        if "BLEND" in member_alpha_modes:
            merged_alpha_mode = "BLEND"
        elif "MASK" in member_alpha_modes:
            merged_alpha_mode = "MASK"
        else:
            merged_alpha_mode = "OPAQUE"
        merged_group = _draped_group_for_texture(texture)

        # No vertex-to-vertex position weld for painted markings (not
        # needed -- _component_size_split reads the source objects' own
        # connectivity regardless). Same-texture markings are pooled into
        # one draw call and deduped.
        #
        # Base-pavement DOES get one: _weld_component_gaps index-welds the
        # boundary vertices of disconnected grid tiles (eps ~0.3m, naked
        # verts only) so they join into one connected mesh and X-Plane's
        # draped renderer stops opening a sub-pixel crack along every
        # coincident-but-separate tile edge. Triangle set/winding
        # unchanged, so footprint area/ranking are untouched. Runs before
        # _dedupe_faces_geom.
        weld_positions = combined_positions
        weld_normals = combined_normals
        weld_uvs = combined_uvs
        weld_indices = combined_indices
        if merged_alpha_mode != "BLEND" and merged_group in (
                _DRAPED_GROUP_GROUND, _DRAPED_GROUP_BASE, _DRAPED_GROUP_WEAR):
            weld_positions, weld_normals, weld_uvs, weld_indices, _n_seam = _weld_component_gaps(
                weld_positions, weld_normals, weld_uvs, weld_indices)
            if _n_seam:
                _log(f"Tile {tile_lat},{tile_lon}: seam-welded {_n_seam} boundary vertex(es) in the "
                     f"{Path(texture).name} base-pavement merge (joins its disconnected tiles into one "
                     f"mesh -- closes the terrain-through grid stripes).", "info")
        weld_positions, weld_normals, weld_uvs, weld_indices, dup_tri_count = _dedupe_faces_geom(
            weld_positions, weld_normals, weld_uvs, weld_indices)

        merge_count += 1
        merged_stem = f"_merged_tile_{tile_lat}_{tile_lon}_{merge_count}"

        # The merged MeshIR must carry each member's real alpha_mode/
        # double_sided/alpha_cutoff, not MeshIR's own OPAQUE/single-sided
        # defaults -- same-texture members overwhelmingly agree on these
        # already (normally a property of the shared material), so this
        # just unions across all of them: BLEND (or MASK) wins over OPAQUE
        # since a merged mesh can't selectively blend part of itself, and
        # double_sided wins the same way, since one double-sided member
        # needs both faces rendered or the others' backfaces cull wrong.
        merged_double_sided = any(ir.double_sided for _, ir in members)
        merged_alpha_cutoff = next((ir.alpha_cutoff for _, ir in members if ir.alpha_mode == "MASK"), 0.5)

        def _push(stem, P, N, U, I, markings_layer, dup=dup_tri_count):
            if I is None or len(I) == 0:
                return
            pending_merges.append({
                "stem": stem, "texture": texture, "positions": P,
                "normals": N, "uvs": U, "indices": I,
                "member_count": len(members), "combined_vertex_count": len(combined_positions),
                "footprint_area": _mesh_surface_area(P, I) or None, "dup_tri_count": dup,
                "alpha_mode": merged_alpha_mode, "double_sided": merged_double_sided,
                "alpha_cutoff": merged_alpha_cutoff, "group": merged_group,
                "markings_layer": markings_layer,
            })

        is_bed = (merged_group == _DRAPED_GROUP_LINES
                  and _marking_colour_role(texture) == _MARKING_ROLE_BED)
        if merged_group == _DRAPED_GROUP_LINES and not is_bed:
            # Split the markings merge into its big structure (panel / line /
            # fill) and its small compact glyphs (characters, numbers,
            # arrowheads); the glyph sub-mesh is pinned to the top draw slot
            # so text always sits on its panel, regardless of which is which
            # colour. See _component_size_split.
            struct, glyph = _component_size_split(
                weld_positions, weld_normals, weld_uvs, weld_indices)
            if struct and glyph:
                _push(merged_stem, *struct, "struct")
                _push(f"{merged_stem}_glyph", *glyph, "glyph", dup=0)
            elif glyph:
                _push(merged_stem, *glyph, "glyph")
            else:
                _push(merged_stem, *struct, "struct")
        else:
            _push(merged_stem, weld_positions, weld_normals, weld_uvs, weld_indices,
                  "bed" if is_bed else None)

    # Give every draped object left in this tile -- every merge result AND
    # every single (unmerged) draped object -- an ATTR_layer_group_draped
    # offset within its band. The `markings` band is ordered by paint colour
    # role (_marking_colour_role); every other band is ranked by footprint
    # family (_rank_band_by_family). Partitioning by band first keeps one
    # band's ~20 layers from all fighting for a single -5..+5 range.
    single_texture = {}
    single_areas = {}
    single_groups = {}
    single_families = {}
    for idx, (entry, ir) in enumerate(singles):
        key = f"_single_{idx}"
        area = ir.footprint_area_m2
        if area is None and len(ir.positions):
            area = _mesh_surface_area(ir.positions, ir.indices)
        single_texture[key] = ir.texture
        single_areas[key] = area or 0.0
        single_groups[key] = _draped_group_for_texture(ir.texture)
        single_families[key] = _family_key(ir.texture)

        # Seam-weld an unmerged base-pavement mesh's tiles too -- LHBP's
        # biggest offender, `asp_w_02` (BaseLayer Asphalt_03), arrives as a
        # SINGLE (342 disconnected tile components) and r13's per-texture
        # merge weld never touched it. Ranking area was captured above from
        # the pre-weld geometry; an index weld of boundary verts leaves the
        # triangle set intact, so it can't perturb the ranking.
        if (single_groups[key] in (_DRAPED_GROUP_GROUND, _DRAPED_GROUP_BASE, _DRAPED_GROUP_WEAR)
                and ir.alpha_mode != "BLEND" and len(ir.positions)):
            _wp, _wn, _wu, _wi, _n_seam = _weld_component_gaps(
                ir.positions, ir.normals, ir.uvs, ir.indices)
            if _n_seam:
                ir = dataclasses.replace(ir, positions=_wp, normals=_wn, uvs=_wu, indices=_wi)
                singles[idx] = (entry, ir)
                _log(f"Tile {tile_lat},{tile_lon}: seam-welded {_n_seam} boundary vertex(es) in the "
                     f"unmerged {Path(ir.texture).name} base-pavement layer (joins its disconnected "
                     f"tiles into one mesh -- closes the terrain-through grid stripes).", "info")

    texture_by_stem = {m["stem"]: m["texture"] for m in pending_merges}
    texture_by_stem.update(single_texture)
    areas_by_stem = {m["stem"]: (m["footprint_area"] or 0.0) for m in pending_merges}
    areas_by_stem.update(single_areas)
    group_by_stem = {m["stem"]: m["group"] for m in pending_merges}
    group_by_stem.update(single_groups)
    family_by_stem = {m["stem"]: _family_key(m["texture"]) for m in pending_merges}
    family_by_stem.update(single_families)

    offsets_by_stem = {}
    for band in _DRAPED_GROUP_ORDER:
        band_stems = [k for k in areas_by_stem
                      if group_by_stem.get(k, _DRAPED_GROUP_LINES) == band]
        if not band_stems:
            continue
        if band == _DRAPED_GROUP_LINES:
            # Three tiers, none decided by colour vs colour:
            #  -5      : a solid coloured BED (red keep-clear box, green
            #            strip) -- white/yellow paint and legends go ON it.
            #  +5      : GLYPH sub-meshes (small compact components) --
            #            always on top of whatever panel they sit on,
            #            whichever colour each is. See _component_size_split.
            #  -4..+4  : everything else, folded to _family_key ranking
            #            families first (every colour/finish variant of one
            #            numbered marking collapses to one slot, same as
            #            _rank_band_by_family does for other bands), then
            #            area-ranked, biggest first = drawn first = under.
            # A single (un-merged) markings object is classified glyph/
            # structure by how much of its area is glyph-sized components,
            # so a standalone stand-number/text model still lands on top.
            layer_of = {m["stem"]: m.get("markings_layer") for m in pending_merges}
            single_ir = {f"_single_{i}": ir for i, (_e, ir) in enumerate(singles)}

            def _is_glyph(k):
                if layer_of.get(k) == "glyph":
                    return True
                ir = single_ir.get(k)
                return ir is not None and _glyph_area_fraction(ir.positions, ir.indices) > 0.5

            bed = [k for k in band_stems
                   if layer_of.get(k) == "bed"
                   or _marking_colour_role(texture_by_stem.get(k, "")) == _MARKING_ROLE_BED]
            glyph = [k for k in band_stems if k not in bed and _is_glyph(k)]
            rest = [k for k in band_stems if k not in bed and k not in glyph]
            for k in bed:
                offsets_by_stem[k] = -5
            for k in glyph:
                offsets_by_stem[k] = 5
            rest_fam_area = {}
            rest_fam_members = {}
            for k in rest:
                fam = family_by_stem.get(k, k)
                rest_fam_area[fam] = rest_fam_area.get(fam, 0.0) + areas_by_stem[k]
                rest_fam_members.setdefault(fam, []).append(k)
            for i, fam in enumerate(sorted(rest_fam_area, key=lambda f: rest_fam_area[f], reverse=True)):
                off = max(-4, min(4, -4 + i))
                for k in rest_fam_members[fam]:
                    offsets_by_stem[k] = off
            if len(rest_fam_area) > 9 and not use_polygons:
                _log(f"Tile {tile_lat},{tile_lon}: {len(rest_fam_area)} distinct markings ranking "
                     f"families in the 'rest' tier -- more than its 9 (-4..+4) slots, so the smallest "
                     f"clamp onto +4 together. Still far better than per-texture (~27) fighting for 9.",
                     "warning")
            continue
        band_areas = {k: areas_by_stem[k] for k in band_stems}
        band_families = {family_by_stem.get(k, k) for k in band_stems}
        if len(band_families) > 11 and not use_polygons:
            _log(f"Tile {tile_lat},{tile_lon}: {len(band_families)} distinct draped ranking families "
                 f"routed to the '{band}' draw band -- more than its 11 (-5..+5) offset slots, so the "
                 f"lowest-priority ones clamp onto the +5 slot together. Still far better than the whole "
                 f"tile's layers fighting for one band.", "warning")
        offsets_by_stem.update(_rank_band_by_family(band_areas, family_by_stem))

    def _to_tile_frame(ir, lat, lon, hdg):
        pos = np.empty((len(ir.positions), 3), dtype=np.float64)
        for k, (x, z) in enumerate(zip(ir.positions[:, 0], ir.positions[:, 2])):
            px, pz = _member_point_to_tile_frame(
                float(x), float(z), lat, lon, hdg, tile_lat, tile_lon, tile_m_lat, tile_m_lon)
            pos[k, 0] = px
            pos[k, 1] = ir.positions[k, 1]
            pos[k, 2] = pz
        return pos

    # ONE recenter centre + ONE placement anchor for EVERY merged draped
    # .obj in this tile (bbox centre of all merged geometry, in the tile
    # frame), so coincident geometry from different texture groups
    # (crosswalk bed vs bars; stand box vs outline vs text) writes
    # identical VT coordinates and renders in exactly the same place --
    # recentering each group around its own footprint instead gives each
    # a different anchor and slides the layers apart.
    if pending_merges:
        _mx = np.concatenate([m["positions"][:, 0] for m in pending_merges if len(m["positions"])] or [np.zeros(1)])
        _mz = np.concatenate([m["positions"][:, 2] for m in pending_merges if len(m["positions"])] or [np.zeros(1)])
        common_cx = float(_mx.min() + _mx.max()) / 2.0
        common_cz = float(_mz.min() + _mz.max()) / 2.0
    else:
        common_cx = common_cz = 0.0
    common_anchor_lat, common_anchor_lon = _tile_frame_to_latlon(
        tile_lat, tile_lon, common_cx, common_cz, tile_m_lat, tile_m_lon)

    # --- Pavement SINGLES join the merges' shared frame + anchor ----------
    # X-Plane linearises each object's tangent plane at its own anchor, so
    # a re-ranked single kept on its own DSF placement while every merge
    # recentred onto common_anchor opened a small but constant terrain-
    # through gap along every merge<->single tile boundary. Fix: transform
    # each pavement-band single into the tile frame, recentre it by the
    # SAME common_cx/cz, and place it at common_anchor with hdg 0 --
    # identical treatment to a merge. Markings/signs singles are left on
    # their own anchor (small, and their placement must not shift).
    _PAVE_SNAP_BANDS = (_DRAPED_GROUP_GROUND, _DRAPED_GROUP_BASE, _DRAPED_GROUP_WEAR)
    _common_frame_singles = set()
    for si_, (s_entry, s_ir) in enumerate(singles):
        if (single_groups.get(f"_single_{si_}") in _PAVE_SNAP_BANDS
                and len(s_ir.positions)):
            tf = _to_tile_frame(s_ir, s_entry["lat"], s_entry["lon"], s_entry["hdg"])
            tf[:, 0] -= common_cx
            tf[:, 2] -= common_cz
            singles[si_] = (s_entry, dataclasses.replace(s_ir, positions=tf))
            _common_frame_singles.add(si_)
    if _common_frame_singles:
        _log(f"Tile {tile_lat},{tile_lon}: reframed {len(_common_frame_singles)} unmerged "
             f"base-pavement layer(s) onto the merges' shared anchor (removes the ~0.75 m "
             f"merge<->single offset that opened the terrain-through apron grid).", "info")

    # --- Cross-object pavement seam snap --------------------------------
    # Ranking is frozen. Even sharing one frame, two separate draped OBJ8s
    # can leave a sub-pixel wedge where coincident-but-un-indexed edges
    # meet. _snap_pavement_seams position-snaps such boundary vertices
    # (<=_PAVEMENT_CROSS_SNAP_EPS_M) onto one shared position -- never
    # merges an index or moves an interior/markings/signs vertex.
    if not use_polygons and os.environ.get("MSFS2XP_PAVE_SEAM_SNAP", "1") == "1":
        _snap_parts, _snap_writeback = [], []
        for m in pending_merges:
            if m["group"] in _PAVE_SNAP_BANDS and m["alpha_mode"] != "BLEND" and len(m["positions"]):
                xz = np.array(m["positions"][:, [0, 2]], dtype=np.float64)
                _snap_parts.append({"xz": xz, "idx": np.asarray(m["indices"])})
                _snap_writeback.append(("merge", m, xz))
        for si_ in _common_frame_singles:
            s_entry, s_ir = singles[si_]
            if s_ir.alpha_mode != "BLEND" and len(s_ir.positions):
                xz = np.array(s_ir.positions[:, [0, 2]], dtype=np.float64)
                _snap_parts.append({"xz": xz, "idx": np.asarray(s_ir.indices)})
                _snap_writeback.append(("single", si_, xz))
        if len(_snap_parts) >= 2:
            _n_snap = _snap_pavement_seams(_snap_parts)
            if _n_snap:
                for kind, ref, xz in _snap_writeback:
                    if kind == "merge":
                        ref["positions"] = ref["positions"].copy()
                        ref["positions"][:, 0] = xz[:, 0]
                        ref["positions"][:, 2] = xz[:, 1]
                    else:
                        s_entry, s_ir = singles[ref]
                        npos = np.array(s_ir.positions, dtype=np.float64)
                        npos[:, 0] = xz[:, 0]
                        npos[:, 2] = xz[:, 1]
                        singles[ref] = (s_entry, dataclasses.replace(s_ir, positions=npos))
                _log(f"Tile {tile_lat},{tile_lon}: cross-object pavement seam snap moved {_n_snap} "
                     f"boundary vertex(es) onto a shared position (<= {_PAVEMENT_CROSS_SNAP_EPS_M} m, "
                     f"no index merged, ranking frozen).", "info")

    # --- Tile-wide pavement gap fill (_fill_pavement_gaps) -- OFF by default.
    # The solid opaque UNDERLAY under the whole paved footprint -- the safety
    # net for whatever draped-render cracks the seam welds above don't close
    # (a crack over the underlay shows grey concrete, not green terrain). See
    # _fill_pavement_gaps. Toggle off with MSFS2XP_BRIDGE_SEAMS=0.
    bridge_meshes = []
    if not use_polygons and os.environ.get("MSFS2XP_BRIDGE_SEAMS", "0") == "1":
        # EVERY pavement-band layer, opaque AND blend -- the union of all of
        # them is the apron/taxiway/runway outline the underlay must cover.
        pave_pieces = []
        for m in pending_merges:
            if m["group"] in _PAVEMENT_FILL_BANDS:
                pave_pieces.append({"texture": m["texture"], "alpha": m["alpha_mode"],
                                    "group": m["group"], "area": m["footprint_area"] or 0.0,
                                    "positions": m["positions"], "indices": m["indices"]})
        for si_, (s_entry, s_ir) in enumerate(singles):
            _sg = single_groups.get(f"_single_{si_}")
            if _sg in _PAVEMENT_FILL_BANDS and len(s_ir.positions):
                if si_ in _common_frame_singles:
                    _pp = np.array(s_ir.positions, dtype=np.float64)   # already common frame
                    _pp[:, 0] += common_cx
                    _pp[:, 2] += common_cz
                else:
                    _pp = _to_tile_frame(s_ir, s_entry["lat"], s_entry["lon"], s_entry["hdg"])
                pave_pieces.append({"texture": s_ir.texture, "alpha": s_ir.alpha_mode,
                                    "group": _sg, "area": single_areas.get(f"_single_{si_}", 0.0),
                                    "positions": _pp, "indices": s_ir.indices})
        for entry in passthrough:
            ir = _load_candidate(obj_dir, entry.get("name", ""))
            _pg = _draped_group_for_texture(ir.texture) if ir is not None else None
            if (ir is None or not len(ir.positions) or _pg not in _PAVEMENT_FILL_BANDS):
                continue
            pave_pieces.append({"texture": ir.texture, "alpha": ir.alpha_mode,
                                "group": _pg,
                                "area": ir.footprint_area_m2 or _mesh_surface_area(ir.positions, ir.indices),
                                "positions": _to_tile_frame(ir, entry["lat"], entry["lon"], entry.get("hdg", 0.0)),
                                "indices": ir.indices})
        if len(pave_pieces) >= 2:
            bridge_meshes = _fill_pavement_gaps(pave_pieces, _log)

        if os.environ.get("MSFS2XP_DUMP"):
            import pickle as _pk
            allp = []
            for m in pending_merges:
                allp.append({"kind": "merge", "stem": m["stem"], "texture": m["texture"],
                             "group": m["group"], "alpha": m["alpha_mode"],
                             "pos": np.asarray(m["positions"]), "idx": np.asarray(m["indices"])})
            for si_, (s_entry, s_ir) in enumerate(singles):
                if si_ in _common_frame_singles:
                    _dp = np.array(s_ir.positions, dtype=np.float64)
                    _dp[:, 0] += common_cx
                    _dp[:, 2] += common_cz
                else:
                    _dp = _to_tile_frame(s_ir, s_entry["lat"], s_entry["lon"], s_entry["hdg"])
                allp.append({"kind": "single", "stem": s_entry["name"], "texture": s_ir.texture,
                             "group": single_groups.get(f"_single_{si_}"), "alpha": s_ir.alpha_mode,
                             "pos": _dp, "idx": np.asarray(s_ir.indices)})
            for entry in passthrough:
                ir = _load_candidate(obj_dir, entry.get("name", ""))
                if ir is None or not len(ir.positions):
                    continue
                allp.append({"kind": "passthrough", "stem": entry["name"], "texture": ir.texture,
                             "group": _draped_group_for_texture(ir.texture), "alpha": ir.alpha_mode,
                             "pos": _to_tile_frame(ir, entry["lat"], entry["lon"], entry.get("hdg", 0.0)),
                             "idx": np.asarray(ir.indices)})
            for bi, bm in enumerate(bridge_meshes):
                allp.append({"kind": "skirt", "stem": f"skirt_{bi}", "texture": bm["texture"],
                             "group": "shoulders", "alpha": "OPAQUE",
                             "pos": np.asarray(bm["positions"]), "idx": np.asarray(bm["indices"])})
            with open(f"/work/_dump_{tile_lat}_{tile_lon}.pkl", "wb") as _f:
                _pk.dump(allp, _f)
            _log(f"MSFS2XP_DUMP: wrote {len(allp)} draped pieces to _dump_{tile_lat}_{tile_lon}.pkl", "info")

    merged_entries = []
    polygon_groups = []  # [(footprint_area, [triangle-dicts]), ...] -- polygon mode only
    for m in pending_merges:
        offset = offsets_by_stem[m["stem"]]
        group = m["group"]
        if use_polygons:
            pol_path = pol_writer.write_pol_for_texture(polygons_dir, Path(m["texture"]).name, layer_group=group)
            tris = _triangles_to_polygons(
                pol_path, tile_lat, tile_lon, 0.0, m["positions"], m["uvs"], m["indices"],
                frame_m=(tile_m_lat, tile_m_lon))
            polygon_groups.append((m["footprint_area"] or 0.0, tris))
        else:
            recentered_positions = m["positions"].copy()
            recentered_positions[:, 0] -= common_cx
            recentered_positions[:, 2] -= common_cz

            merged_ir = mesh_ir.MeshIR(
                name=m["stem"], positions=recentered_positions, normals=m["normals"], uvs=m["uvs"],
                indices=m["indices"], texture=m["texture"], draped=True, draped_layer_offset=offset,
                draped_layer_group=group,
                alpha_mode=m["alpha_mode"], double_sided=m["double_sided"], alpha_cutoff=m["alpha_cutoff"],
            )
            merged_path = obj_dir / f"{m['stem']}.obj"
            mesh_ir.write_obj8(merged_ir, merged_path)

            new_entry = {"name": m["stem"], "lat": common_anchor_lat, "lon": common_anchor_lon,
                         "hdg": 0.0, "agl": 0.0}
            if m["footprint_area"]:
                new_entry["footprint_area"] = m["footprint_area"]
            merged_entries.append(new_entry)

        dup_note = (
            f" -- also removed {m['dup_tri_count']} fully-duplicate triangle(s) left over from "
            f"overlapping same-material layers" if m["dup_tri_count"] else ""
        )
        offset_note = "converted to a real .pol DSF polygon (no draw-order slot needed)" if use_polygons \
            else f"ATTR_layer_group_draped {group} {offset}"
        _log(f"Tile {tile_lat},{tile_lon}: merged {m['member_count']} draped objects sharing texture "
             f"{Path(m['texture']).name} into one combined mesh ({len(m['positions'])} vertices after "
             f"welding, {m['combined_vertex_count']} before), {offset_note} -- removes "
             f"the draw-order collision risk both between them and against any other merged surface in "
             f"this tile, and any seam/gap at their shared boundary.{dup_note}", "info")

    for idx, (entry, ir) in enumerate(singles):
        key = f"_single_{idx}"
        offset = offsets_by_stem[key]
        group = single_groups[key]
        if use_polygons:
            pol_path = pol_writer.write_pol_for_texture(polygons_dir, Path(ir.texture).name, layer_group=group)
            tris = _triangles_to_polygons(
                pol_path, entry["lat"], entry["lon"], entry["hdg"], ir.positions, ir.uvs, ir.indices)
            polygon_groups.append((single_areas[key], tris))
            _log(f"Tile {tile_lat},{tile_lon}: converted {entry['name']} (no same-texture merge partner) "
                 f"to a real .pol DSF polygon in the '{group}' draw band.", "info")
            continue
        cur_group = getattr(ir, "draped_layer_group", None) or _DRAPED_GROUP_LINES
        in_common_frame = idx in _common_frame_singles
        if offset == ir.draped_layer_offset and group == cur_group and not in_common_frame:
            # Unchanged -- no need to write a new file, just pass the
            # original entry through exactly as before.
            passthrough.append(entry)
            continue
        ranked_stem = f"{entry['name']}_dlrank{offset}"
        ranked_ir = dataclasses.replace(
            ir, name=ranked_stem, draped_layer_offset=offset, draped_layer_group=group)
        mesh_ir.write_obj8(ranked_ir, obj_dir / f"{ranked_stem}.obj")
        new_entry = dict(entry)
        new_entry["name"] = ranked_stem
        if in_common_frame:
            # geometry is already in the tile frame, recentred by common_cx/cz
            # -> place it exactly where every merge is placed.
            new_entry["lat"] = common_anchor_lat
            new_entry["lon"] = common_anchor_lon
            new_entry["hdg"] = 0.0
        passthrough.append(new_entry)
        _log(f"Tile {tile_lat},{tile_lon}: re-ranked {entry['name']} (no same-texture merge partner) to "
             f"ATTR_layer_group_draped {group} {offset} (was {cur_group} {ir.draped_layer_offset}), "
             f"against every other draped object in this tile -- fixes small-over-large draw order "
             f"(e.g. a sign panel rendering under the pavement it sits on) that per-file ranking "
             f"alone can't see.", "info")

    # Write the cross-piece seam bridges (if any) as their own bottom-band
    # draped objects. Deliberately kept out of pending_merges / singles and
    # given a hard-coded band + offset so they cannot perturb any draw-order
    # ranking. Sharing the tile's common recenter centre + anchor keeps
    # their vertices bit-for-bit coincident with the slab vertices they were
    # copied from.
    for bi, bm in enumerate(bridge_meshes, start=1):
        b_pos = bm["positions"].copy()
        b_pos[:, 0] -= common_cx
        b_pos[:, 2] -= common_cz
        b_stem = f"_seam_bridge_tile_{tile_lat}_{tile_lon}_{bi}"
        bridge_ir = mesh_ir.MeshIR(
            name=b_stem, positions=b_pos, normals=bm["normals"], uvs=bm["uvs"],
            indices=bm["indices"], texture=bm["texture"], draped=True,
            draped_layer_offset=-5, draped_layer_group=_DRAPED_GROUP_GROUND,
            alpha_mode="OPAQUE", double_sided=True, alpha_cutoff=0.5)
        mesh_ir.write_obj8(bridge_ir, obj_dir / f"{b_stem}.obj")
        merged_entries.append({"name": b_stem, "lat": common_anchor_lat,
                               "lon": common_anchor_lon, "hdg": 0.0, "agl": 0.0})
        _log(f"Tile {tile_lat},{tile_lon}: bridged pavement seams with {len(bm['indices']) // 3} "
             f"new triangle(s) on texture {Path(bm['texture']).name} -- ATTR_layer_group_draped "
             f"{_DRAPED_GROUP_GROUND} -5 (bottom band, under every real layer). Closes terrain-through "
             f"gaps between adjacent base-pavement slabs of different textures; adds no vertex to and "
             f"moves no vertex of any existing slab, and changes no draw-order ranking.", "info")

    # Largest footprint first (== drawn first == underneath), same
    # convention as offsets_by_stem's own ranking and the objects=sorted(...)
    # tie-breaker in main.py -- DSF draws polygons sharing a LAYER_GROUP in
    # CMDS stream order, so this keeps base fills under small markings/signs
    # as a tie-breaker within each band (the .pol's own per-band LAYER_GROUP,
    # set from _draped_group_for_texture, is the primary ordering).
    polygon_groups.sort(key=lambda g: g[0], reverse=True)
    polygons = [tri for _, tris in polygon_groups for tri in tris]

    return passthrough + merged_entries, polygons
