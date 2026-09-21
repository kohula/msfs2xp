import os
import sys
import re
import math
import shutil
import tempfile
import struct
import json
import io
import hashlib
import threading
import multiprocessing
import pickle
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageTk, ImageDraw
import texture2ddecoder

import bgl_extractor
import dsf_compiler
import apt_dat
import mesh_convert
import cache_utils
import gpu_accel
import terrain_fit
import draped_merge
import pick_replacements
import scenery_viewer
import geo_transform
from mesh_convert import mesh_ir

CONFIG_FILE = Path("msfs2xp_config.json")

# Next to this script, not the system temp/profile drive -- a package
# extraction can run into the GB range per run and would otherwise quietly
# fill a small C: system drive. Under a frozen PyInstaller EXE, __file__
# resolves inside the ephemeral extraction temp dir, so anchor on
# sys.executable's own folder instead.
if getattr(sys, "frozen", False):
    _SCRIPT_DIR = Path(sys.executable).resolve().parent
else:
    _SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_TEMP_ROOT = _SCRIPT_DIR / "_temp"

# Last-resort library substitution for placements with no real MSFS model
# available at all (nothing extracted/converted to fall back to) -- never
# applied to anything we could actually convert, since the real geometry
# is always better than a stand-in. Matched by keyword against the
# normalized placement title (titles for the same object class vary by
# package/author, so no single exact string covers them all). Paths are
# real EXPORT declarations from an X-Plane 11 install's own library.txt.
LIBRARY_SUBSTITUTION_KEYWORDS = [
    ("windsock", "lib/airport/landscape/windsock_lit.obj"),
    ("wind_sock", "lib/airport/landscape/windsock_lit.obj"),
    ("wind sock", "lib/airport/landscape/windsock_lit.obj"),
    ("floodlight", "lib/airport/Common_Elements/Lighting/com_Flood_36m.obj"),
    ("flood_light", "lib/airport/Common_Elements/Lighting/com_Flood_36m.obj"),
    ("flood light", "lib/airport/Common_Elements/Lighting/com_Flood_36m.obj"),
    ("apronlight", "lib/airport/Common_Elements/Lighting/com_DownLight.obj"),
    ("apron_light", "lib/airport/Common_Elements/Lighting/com_DownLight.obj"),
    ("apron light", "lib/airport/Common_Elements/Lighting/com_DownLight.obj"),
    ("ramplight", "lib/airport/Common_Elements/Lighting/Dir_Ramp_Lit_Med.obj"),
    ("ramp_light", "lib/airport/Common_Elements/Lighting/Dir_Ramp_Lit_Med.obj"),
    ("ramp light", "lib/airport/Common_Elements/Lighting/Dir_Ramp_Lit_Med.obj"),
]

_COPY_SUFFIX_RE = re.compile(r"\s*\(copy\s*\d*\)\s*$", re.IGNORECASE)


def normalize_placement_title(title):
    if not title:
        return None
    return _COPY_SUFFIX_RE.sub("", title).strip().lower()


# Library-object stem <-> placement-title matching, for the GUID-miss
# fallback in the DSF step. MCX names a converted model file with an
# 8-hex library prefix and an MSFS _LOD<n> / _Static[_<32hex>] suffix
# that a placement's plain title never carries. Stripping both lets a
# titled placement resolve to a converted model even when a container/
# cache/library quirk broke its GUID path.
_LIB_STEM_PREFIX_RE = re.compile(r'^[0-9a-fA-F]{8}_')
_LIB_STEM_SUFFIX_RE = re.compile(r'(?:_Static(?:_[0-9a-fA-F]{32})?|_LOD\d+)+$', re.IGNORECASE)


def _model_stem_basename(name):
    if not name:
        return ""
    s = _LIB_STEM_PREFIX_RE.sub("", str(name).strip())
    s = _LIB_STEM_SUFFIX_RE.sub("", s)
    return s.strip("_").lower()


def build_title_stem_index(converted_stems_map):
    """{normalized basename -> converted-model stem}. On a collision,
    prefer a _Static (closed-pose) variant, then the shortest stem."""
    idx = {}
    for stem in converted_stems_map:
        key = _model_stem_basename(stem)
        if not key:
            continue
        cur = idx.get(key)
        if cur is None:
            idx[key] = stem
            continue
        cur_static, new_static = "static" in cur.lower(), "static" in stem.lower()
        if (new_static and not cur_static) or (new_static == cur_static and len(stem) < len(cur)):
            idx[key] = stem
    return idx


SKIP_SUBSTITUTION = "__SKIP__"  # sentinel: user explicitly chose to place nothing for this object


def load_object_replacements(*dirs):
    """Load the user's saved "this missing object -> that X-Plane library
    path" picks (written by pick_replacements.py, see message from the
    user asking for a browser to map unresolved ASOBO taxiway/runway/PAPI
    lights etc.). Looked for as object_replacements.json in each of `dirs`
    in turn (typically the source package dir, then the app dir), merged
    with later dirs winning. Keys are matched case-insensitively against
    a placement's GUID (braces and case irrelevant) and its normalized
    title. Value is an X-Plane 'lib/...' path, or "SKIP"/"" to
    deliberately place nothing (and stop re-reporting it as unresolved).
    A missing or unparseable file just contributes nothing."""
    out = {}
    for d in dirs:
        if not d:
            continue
        try:
            raw = json.loads((Path(d) / "object_replacements.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries = raw.get("replacements", raw) if isinstance(raw, dict) else {}
        if not isinstance(entries, dict):
            continue
        for k, v in entries.items():
            if isinstance(v, dict):
                v = v.get("library_path") or v.get("lib") or ""
            if not isinstance(v, str):
                continue
            key = str(k).strip().strip("{}").lower()
            if not key:
                continue
            v = v.strip()
            if not v or v.upper() == "SKIP":
                out[key] = SKIP_SUBSTITUTION
            else:
                out[key] = v
    return out


def _msguid_unswap(disp):
    """{AA78743D-9045-4885-BAAE-F910506C950C} -> the 32-hex-no-dash form
    bgl_extractor stores in a placement's 'guid' field. MS GUID text form
    is little-endian for its first 3 groups, so those bytes reverse."""
    x = str(disp).strip().strip("{}").replace("-", "")
    try:
        b = bytes.fromhex(x)
    except ValueError:
        return ""
    if len(b) != 16:
        return ""
    return (b[0:4][::-1] + b[4:6][::-1] + b[6:8][::-1] + b[8:16]).hex()


def load_simprop_names(pkg):
    """{placement-guid (32 hex) -> friendly name} from the package's own
    SimPropContainers/simPropContainers.json. That file lists every
    building-interior / attached-prop container the package defines, each
    with a readable 'descr' (LHBP_Terminal_Int_1_SimPropContainer,
    SHS_Clutter_ApronLight_001, ...) -- used to give the replacement
    picker a real name instead of a bare GUID for placements whose own
    title is blank. Missing/broken file -> {}."""
    out = {}
    try:
        raw = json.loads((Path(pkg) / "SimPropContainers" / "simPropContainers.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for e in (raw.get("content", []) if isinstance(raw, dict) else []):
        if not isinstance(e, dict):
            continue
        g = _msguid_unswap(e.get("guid", ""))
        d = (e.get("descr") or "").strip()
        if g and d:
            out[g] = d
    return out


def resolve_library_substitution(title, guid=None, approximate=False, replacements=None):
    """Returns a library_path to substitute for a missing placement, the
    SKIP_SUBSTITUTION sentinel (user chose to place nothing), or None.

    `replacements` (from load_object_replacements) is checked FIRST and
    wins over everything -- an explicit user pick for this exact object,
    keyed by GUID or normalized title. Then the built-in exact keyword
    (substring) match against LIBRARY_SUBSTITUTION_KEYWORDS, checked
    regardless of `approximate`. Only when both find nothing AND
    `approximate` is explicitly enabled (the GUI's "Approximate name-
    matching" toggle, off by default) does this fall back to a fuzzy
    (difflib, standard-library -- not a hosted/external AI service)
    string-similarity match against the same keyword list."""
    normalized = normalize_placement_title(title)
    if replacements:
        for key in (guid, normalized):
            if not key:
                continue
            hit = replacements.get(str(key).strip().strip("{}").lower())
            if hit:
                return SKIP_SUBSTITUTION if hit.upper() in ("SKIP", SKIP_SUBSTITUTION) else hit
    if not normalized:
        return None
    for keyword, lib_path in LIBRARY_SUBSTITUTION_KEYWORDS:
        if keyword in normalized:
            return lib_path
    if not approximate:
        return None
    import difflib
    keywords = [k for k, _ in LIBRARY_SUBSTITUTION_KEYWORDS]
    matches = difflib.get_close_matches(normalized, keywords, n=1, cutoff=0.6)
    if not matches:
        return None
    matched_keyword = matches[0]
    return next(lib_path for keyword, lib_path in LIBRARY_SUBSTITUTION_KEYWORDS if keyword == matched_keyword)


def _dilate_mask(mask, r):
    """Binary dilation of a 2-D bool array by `r` cells, 8-connected
    (includes diagonals) so a single occupied cell grows to a (2r+1)
    square. Used to give the road-exclusion footprint a skirt past the
    outermost placed object."""
    if r <= 0:
        return mask
    out = mask
    for _ in range(int(r)):
        m = out
        d = m.copy()
        d[:-1, :] |= m[1:, :]
        d[1:, :] |= m[:-1, :]
        d[:, :-1] |= m[:, 1:]
        d[:, 1:] |= m[:, :-1]
        d[:-1, :-1] |= m[1:, 1:]
        d[1:, 1:] |= m[:-1, :-1]
        d[:-1, 1:] |= m[1:, :-1]
        d[1:, :-1] |= m[:-1, 1:]
        out = d
    return out


def _greedy_rects_from_mask(mask):
    """Cover every True cell of a 2-D bool array with a small set of
    axis-aligned rectangles (greedy: grab the widest run on a row, then
    grow it downward as far as the whole span stays True and unclaimed).
    Returns a list of (row0, col0, row1, col1) inclusive tuples. Not a
    minimal cover, but blobby airport footprints decompose into only a
    few dozen rectangles, which is what matters for the emitted
    sim/exclude_net prop count."""
    ny, nx = mask.shape
    used = np.zeros_like(mask)
    rects = []
    for y in range(ny):
        x = 0
        while x < nx:
            if mask[y, x] and not used[y, x]:
                x1 = x
                while x1 + 1 < nx and mask[y, x1 + 1] and not used[y, x1 + 1]:
                    x1 += 1
                y1 = y
                while (y1 + 1 < ny and mask[y1 + 1, x:x1 + 1].all()
                       and not used[y1 + 1, x:x1 + 1].any()):
                    y1 += 1
                used[y:y1 + 1, x:x1 + 1] = True
                rects.append((y, x, y1, x1))
                x = x1 + 1
            else:
                x += 1
    return rects


def _built_up_exclusion_rects(lats, lons, cell_m=200.0, dilate=1, min_points=5, max_rects=80):
    """Axis-aligned lat/lon rectangles that cover ONLY where this
    conversion actually placed objects -- for the roads/rail
    (sim/exclude_net / sim/exclude_str) exclusion, so default X-Plane car
    roads get suppressed over the airport's real built-up footprint but
    NOT out in the surrounding fields (the user's note: the old single
    4.5km radius box was "a bit too big").

    Rasterises every placement onto a ~`cell_m` grid, dilates it
    `dilate` cell(s) so a road hugging the outermost building's edge is
    still covered, then greedy-rectangles the occupied mask. Coarsens the
    grid and retries if the result needs more than `max_rects` rects.
    Returns [] (caller falls back to the fixed-radius box) when there are
    fewer than `min_points` placements, a stray far-flung coordinate
    would blow the grid up, or even the coarsest grid stays too
    fragmented."""
    pts = [(la, lo) for la, lo in zip(lats, lons) if la is not None and lo is not None]
    if len(pts) < min_points:
        return []
    arr = np.asarray(pts, dtype=np.float64)
    lat0, lon0 = float(arr[:, 0].min()), float(arr[:, 1].min())
    ref_lat = float(arr[:, 0].mean())
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * max(math.cos(math.radians(ref_lat)), 1e-6)

    chosen = None
    for _cell in (cell_m, cell_m * 2, cell_m * 4, cell_m * 8, cell_m * 16):
        xs = (arr[:, 1] - lon0) * m_per_deg_lon
        ys = (arr[:, 0] - lat0) * m_per_deg_lat
        gx = np.floor(xs / _cell).astype(np.int64)
        gy = np.floor(ys / _cell).astype(np.int64)
        nx = int(gx.max()) + 1
        ny = int(gy.max()) + 1
        if nx * ny > 4_000_000:      # pathological -- would allocate >0.5 GB; stop entirely
            return []
        if max(nx, ny) > 1200:       # spans >200km at this cell size -- not a real airport footprint; try coarser
            continue
        occ = np.zeros((ny, nx), dtype=bool)
        occ[gy, gx] = True
        occ = _dilate_mask(occ, dilate)
        rects = _greedy_rects_from_mask(occ)
        if len(rects) <= max_rects:
            chosen = (_cell, rects)
            break
    if chosen is None:  # too few points survived, or a stray coordinate kept every grid too big/fragmented
        return []

    _cell, rects = chosen
    pad_lat = 5.0 / m_per_deg_lat
    pad_lon = 5.0 / m_per_deg_lon
    out = []
    for (gy0, gx0, gy1, gx1) in rects:
        out.append({
            "west":  lon0 + (gx0 * _cell) / m_per_deg_lon - pad_lon,
            "east":  lon0 + ((gx1 + 1) * _cell) / m_per_deg_lon + pad_lon,
            "south": lat0 + (gy0 * _cell) / m_per_deg_lat - pad_lat,
            "north": lat0 + ((gy1 + 1) * _cell) / m_per_deg_lat + pad_lat,
        })
    return out


def _points_in_polygon(px, py, poly_x, poly_y):
    """Vectorized ray-casting point-in-polygon test (crossing-number
    algorithm). px/py: 1-D arrays of query points. poly_x/poly_y: the
    polygon's own vertices, in order (implicitly closed -- the last
    vertex connects back to the first). Returns a bool array, one per
    query point. No external geometry library in this project (numpy +
    Pillow only), so this is a from-scratch, dependency-free
    implementation of the standard algorithm."""
    n = len(poly_x)
    inside = np.zeros(len(px), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        # Edge (j -> i) crosses the horizontal ray from (px, py) iff its
        # two endpoints straddle py, and the edge's own X at that Y is
        # to the right of px -- each qualifying crossing flips "inside".
        straddles = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at_y = xj + (py - yj) * (xi - xj) / (yi - yj)
        crosses = straddles & (px < x_at_y)
        inside ^= crosses
        j = i
    return inside


def _polygon_interior_exclusion_rects(boundary_points, cell_m=100.0, max_rects=80):
    """Axis-aligned lat/lon rectangles that tightly cover a REAL boundary
    ring's own INTERIOR shape -- not just its bounding box. A long, thin
    runway-shaped boundary (or an L-shaped/irregular airport perimeter)
    has a bounding box far larger than the ring itself; rasterizing the
    ring's actual interior and greedy-rectangling that mask (same
    pattern as _built_up_exclusion_rects) stays tight to the real shape
    instead. boundary_points: [(lat, lon), ...] ring vertices (as
    apt_dat.extract_boundary_ring returns them). Returns [] for a
    degenerate ring (fewer than 3 points) or if the rasterized grid would
    be pathologically large."""
    if len(boundary_points) < 3:
        return []
    arr = np.asarray(boundary_points, dtype=np.float64)
    lats, lons = arr[:, 0], arr[:, 1]
    lat0, lon0 = float(lats.min()), float(lons.min())
    ref_lat = float(lats.mean())
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * max(math.cos(math.radians(ref_lat)), 1e-6)

    poly_x = (lons - lon0) * m_per_deg_lon
    poly_y = (lats - lat0) * m_per_deg_lat

    chosen = None
    for _cell in (cell_m, cell_m * 2, cell_m * 4, cell_m * 8):
        nx = int(np.ceil(float(poly_x.max()) / _cell)) + 1
        ny = int(np.ceil(float(poly_y.max()) / _cell)) + 1
        if nx * ny > 4_000_000:
            return []
        if max(nx, ny) > 1200:
            continue
        gy, gx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        cell_cx = (gx.ravel() + 0.5) * _cell
        cell_cy = (gy.ravel() + 0.5) * _cell
        inside = _points_in_polygon(cell_cx, cell_cy, poly_x, poly_y)
        mask = inside.reshape(ny, nx)
        if not mask.any():
            continue
        mask = _dilate_mask(mask, 1)
        rects = _greedy_rects_from_mask(mask)
        if len(rects) <= max_rects:
            chosen = (_cell, rects)
            break
    if chosen is None:
        return []

    _cell, rects = chosen
    pad_lat = 5.0 / m_per_deg_lat
    pad_lon = 5.0 / m_per_deg_lon
    out = []
    for (gy0, gx0, gy1, gx1) in rects:
        out.append({
            "west":  lon0 + (gx0 * _cell) / m_per_deg_lon - pad_lon,
            "east":  lon0 + ((gx1 + 1) * _cell) / m_per_deg_lon + pad_lon,
            "south": lat0 + (gy0 * _cell) / m_per_deg_lat - pad_lat,
            "north": lat0 + ((gy1 + 1) * _cell) / m_per_deg_lat + pad_lat,
        })
    return out


def _convex_hull_2d(points):
    """Andrew's monotone chain, from scratch (no external geometry
    library, matching this project's own convention -- see
    _points_in_polygon). points: Nx2 array of (x, z). Returns hull
    vertices in CCW order as an (M, 2) array; M can be 1 or 2 for a
    degenerate (single point / collinear) input."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) < 3:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _rasterize_footprint_mask(xz, tris, cell_m, max_cells=2_000_000):
    """Rasterizes a 2-D footprint onto a cell_m-resolution boolean grid.
    xz: (N, 2) local X/Z vertex positions. tris: (M, 3) int array of
    triangle indices into xz, or None/empty. Fills every real triangle via
    Pillow's polygon fill (measured ~3us/triangle -- even a combined
    multi-part placement with a few hundred thousand triangles rasterizes
    in well under a second, so this is cheap enough to run per placement).
    Falls back to filling just the convex hull as one solid polygon when
    tris is empty (a lights-only/point-cloud sidecar with no faces, or a
    test fixture that only supplies corner positions) -- still a real
    filled area, not per-edge boxes.

    Escalates cell_m (doubling, same retry pattern as
    _built_up_exclusion_rects/_polygon_interior_exclusion_rects) if the
    grid would exceed max_cells. Returns (mask, x_min, z_min, actual_cell)
    or (None, None, None, None) if nothing usable was rasterized."""
    x_min, z_min = float(xz[:, 0].min()), float(xz[:, 1].min())
    x_max, z_max = float(xz[:, 0].max()), float(xz[:, 1].max())
    span_x, span_z = max(x_max - x_min, 1e-6), max(z_max - z_min, 1e-6)

    _cell = cell_m
    for _cell in (cell_m, cell_m * 2, cell_m * 4, cell_m * 8, cell_m * 16):
        nx = max(1, int(math.ceil(span_x / _cell)) + 1)
        nz = max(1, int(math.ceil(span_z / _cell)) + 1)
        if nx * nz <= max_cells:
            break
    else:
        return None, None, None, None

    img = Image.new("L", (nx, nz), 0)
    draw = ImageDraw.Draw(img)
    if tris is not None and len(tris):
        px = (xz[:, 0] - x_min) / _cell
        pz = (xz[:, 1] - z_min) / _cell
        for a, b, c in tris:
            draw.polygon([(px[a], pz[a]), (px[b], pz[b]), (px[c], pz[c])], fill=255)
    else:
        hull = _convex_hull_2d(xz)
        if len(hull) >= 3:
            poly = [((hx - x_min) / _cell, (hz - z_min) / _cell) for hx, hz in hull]
            draw.polygon(poly, fill=255)
        elif len(hull) == 2:
            (hx0, hz0), (hx1, hz1) = hull
            draw.line([((hx0 - x_min) / _cell, (hz0 - z_min) / _cell),
                       ((hx1 - x_min) / _cell, (hz1 - z_min) / _cell)], fill=255, width=1)
        else:
            return None, None, None, None

    mask = np.asarray(img, dtype=bool)
    if not mask.any():
        return None, None, None, None
    return mask, x_min, z_min, _cell


def _per_object_exclusion_rects(obj_dir, footprint_candidates, pad_m=2.0, cell_m=1.0):
    """Exclusion rectangles that cover each converted object's OWN
    footprint using close to the MINIMUM real area needed -- not one
    shared airport-wide set, not one bounding box per object, and not even
    one rectangle per convex-hull edge (that still over-covers a concave
    or multi-armed footprint: the local axis-aligned box around one
    diagonal hull edge, or the hull itself, both cover ground a concave
    notch -- e.g. the gap between an X-shaped building's two arms -- never
    actually occupies. A solid, simply-convex object like a plain square
    building doesn't need 4 separate edge-hugging rects either; one
    minimal rectangle already covers it exactly).

    Rasterizes the object's REAL triangles (not just its hull) onto a
    cell_m-resolution local grid (_rasterize_footprint_mask), then greedy-
    rectangles the occupied mask -- the SAME scanline-run-then-extend-
    while-identical decomposition already used for the road/rail and
    airport-boundary-interior exclusions (_greedy_rects_from_mask): a run
    of occupied cells on one grid line only grows into a taller rectangle
    while the WHOLE run stays identically occupied on the next line, so
    two parts that diverge (an X's arms) split into separate rectangles
    right where they stop lining up, while a uniform strip merges into one
    rectangle instead of fragmenting. Each resulting rectangle is padded by
    pad_m on every side (in local metres, within the requested 1-3m growth
    range) before being rotated into real-world lat/lon by the placement's
    own heading (geo_transform.local_offset_to_latlon, same convention
    used everywhere else in this pipeline).

    footprint_candidates: [(generated_stems, base_lat, base_lon,
    heading_deg), ...] -- one entry per real placement (base_lat/base_lon
    are that placement's own already mid_x/mid_z-recentering-compensated
    anchor, matching the local coordinate frame each generated stem's
    .meshir.pkl sidecar was written in). Every named stem's sidecar (if
    any) contributes its real local triangles to ONE shared raster per
    placement -- objectwise, not per material sub-object, so a multi-part
    building's footprint is the union of all its parts, not several
    independently-hulled pieces.

    Terrain-fit's own corrected copies aren't needed here: rigid shift and
    draped Y-warps don't materially change an object's own XZ footprint,
    and reading the pre-terrain-fit original avoids a dependency on
    terrain-fit having already run for this stem.

    Skips stems with no sidecar or no real geometry (lights-only/animated-
    only objects have no footprint to exclude). Returns [] if nothing
    produced a usable footprint at all, so the caller can fall back to
    the airport-wide shape.
    """
    rects = []
    sidecar_cache = {}
    for generated_stems, base_lat, base_lon, heading_deg in footprint_candidates:
        xz_parts = []
        tri_parts = []
        offset = 0
        for stem in generated_stems:
            if stem in sidecar_cache:
                positions, indices = sidecar_cache[stem]
            else:
                sidecar = mesh_ir.sidecar_path_for(obj_dir / f"{stem}.obj")
                positions, indices = None, None
                if sidecar.exists():
                    try:
                        ir = mesh_ir.load(sidecar)
                        if len(ir.positions):
                            positions = ir.positions[:, [0, 2]]
                            indices = ir.indices
                    except (OSError, EOFError, pickle.UnpicklingError):
                        positions, indices = None, None
                sidecar_cache[stem] = (positions, indices)
            if positions is None:
                continue
            xz_parts.append(positions)
            if indices is not None and len(indices) >= 3:
                tri_parts.append(np.asarray(indices, dtype=np.int64).reshape(-1, 3) + offset)
            offset += len(positions)
        if not xz_parts:
            continue
        xz = np.concatenate(xz_parts, axis=0)
        tris = np.concatenate(tri_parts, axis=0) if tri_parts else None

        mask, x_min, z_min, cell = _rasterize_footprint_mask(xz, tris, cell_m)
        if mask is None:
            continue
        grid_rects = _greedy_rects_from_mask(mask)
        if not grid_rects:
            continue

        for (gy0, gx0, gy1, gx1) in grid_rects:
            lx_min = x_min + gx0 * cell - pad_m
            lx_max = x_min + (gx1 + 1) * cell + pad_m
            lz_min = z_min + gy0 * cell - pad_m
            lz_max = z_min + (gy1 + 1) * cell + pad_m
            corners = [(lx_min, lz_min), (lx_max, lz_min), (lx_max, lz_max), (lx_min, lz_max)]
            lats, lons = [], []
            for cx, cz in corners:
                la, lo = geo_transform.local_offset_to_latlon(base_lat, base_lon, heading_deg, cx, cz)
                lats.append(la)
                lons.append(lo)
            rects.append({
                "west": min(lons),
                "east": max(lons),
                "south": min(lats),
                "north": max(lats),
            })
    return rects


TRANSPARENT_KEY = "#000001"  # Color key used for window corner rounding transparency

# Vulkan Format Mapping for KTX2
VK_FORMAT_BC1_RGB_UNORM_BLOCK  = 131
VK_FORMAT_BC1_RGBA_UNORM_BLOCK = 133
VK_FORMAT_BC3_UNORM_BLOCK      = 137
VK_FORMAT_BC4_UNORM_BLOCK      = 139
VK_FORMAT_BC5_UNORM_BLOCK      = 141
VK_FORMAT_BC5_SNORM_BLOCK      = 142
VK_FORMAT_BC7_UNORM_BLOCK      = 145
VK_FORMAT_R8G8B8A8_UNORM       = 37
VK_FORMAT_ASTC_4x4_UNORM_BLOCK = 157
VK_FORMAT_ASTC_4x4_SRGB_BLOCK  = 158

try:
    import zstandard
except ImportError:
    zstandard = None

try:
    import zlib as _zlib_module
except ImportError:
    _zlib_module = None

# KTX2 supercompressionScheme values (KTX2 spec)
_KTX2_SUPERCOMPRESSION_NONE   = 0
_KTX2_SUPERCOMPRESSION_BASIS  = 1  # true ETC1S+BasisLZ -- needs the real Basis transcoder, not decodable here
_KTX2_SUPERCOMPRESSION_ZSTD   = 2
_KTX2_SUPERCOMPRESSION_ZLIB   = 3


def _build_dds_bytes(width, height, compressed_bytes, fourcc, dxgi_format=None, block_size=16):
    """Builds a minimal, single-mip-level DDS file (magic + DDS_HEADER,
    with a DDS_HEADER_DXT10 extension when dxgi_format is given) wrapping
    compressed_bytes UNCHANGED -- a pure container rewrap, not a
    recompress: the GPU-native block data X-Plane's own DDS loader reads
    is bit-for-bit identical to what the source KTX2 already had. Byte
    offsets (fourCC at 84, compressed data at 128, or 148 with a DX10
    header) match mesh_convert.convert.decode_dds_bytes_to_png's own
    reader exactly, so a file this writes decodes correctly there too."""
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PIXELFORMAT = 0x1000
    DDSD_LINEARSIZE = 0x80000
    DDSCAPS_TEXTURE = 0x1000
    DDPF_FOURCC = 0x4

    blocks_w = max(1, (width + 3) // 4)
    blocks_h = max(1, (height + 3) // 4)
    linear_size = blocks_w * blocks_h * block_size
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE

    header_pre_pf = struct.pack("<7I11I", 124, flags, height, width, linear_size, 0, 0, *([0] * 11))
    pixelformat = struct.pack("<2I4s5I", 32, DDPF_FOURCC, fourcc, 0, 0, 0, 0, 0)
    caps = struct.pack("<5I", DDSCAPS_TEXTURE, 0, 0, 0, 0)
    out = b"DDS " + header_pre_pf + pixelformat + caps
    if dxgi_format is not None:
        out += struct.pack("<5I", dxgi_format, 3, 0, 1, 0)  # resourceDimension=3 (TEXTURE2D)
    return out + compressed_bytes


def _bc5_snorm_to_unorm_bytes(data):
    """See the identical helper in mesh_convert/convert.py for the full
    explanation: texture2ddecoder.decode_bc5 only implements the UNORM
    (unsigned) endpoint convention, so BC5_SNORM normal-map data needs its
    two endpoint bytes per 8-byte half-block bias-flipped (XOR 0x80) before
    decoding, or you get garish rainbow noise instead of a normal map."""
    arr = np.frombuffer(data, dtype=np.uint8)
    n_blocks = len(arr) // 16
    if n_blocks == 0:
        return data
    blocks = arr[: n_blocks * 16].reshape(n_blocks, 16).copy()
    blocks[:, 0] ^= 0x80
    blocks[:, 1] ^= 0x80
    blocks[:, 8] ^= 0x80
    blocks[:, 9] ^= 0x80
    tail = data[n_blocks * 16:]
    return blocks.tobytes() + bytes(tail)


def _decompress_supercompressed(raw_bytes, scheme):
    """Undo container-level supercompression (Zstd/ZLIB) on a KTX2 level's
    payload. This is independent of the block format (BC7/ASTC/etc) that's
    compressed *inside* it -- it's just generic byte compression layered on
    top to shrink the package on disk. Returns None if it can't be handled."""
    if scheme == _KTX2_SUPERCOMPRESSION_ZSTD:
        if zstandard is None:
            return None
        return zstandard.ZstdDecompressor().decompress(raw_bytes)
    if scheme == _KTX2_SUPERCOMPRESSION_ZLIB:
        if _zlib_module is None:
            return None
        return _zlib_module.decompress(raw_bytes)
    return raw_bytes


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def force_taskbar_icon(root):
    """Forces Windows OS DWM to render the frameless window on the Taskbar immediately."""
    if sys.platform == "win32":
        try:
            import ctypes
            root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            if hwnd == 0:
                hwnd = root.winfo_id()

            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW = 0x00040000
            WS_EX_TOOLWINDOW = 0x00000080

            # Strip TOOLWINDOW flag and assign APPWINDOW flag
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

            # Force Windows Shell to refresh frame state immediately
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOZORDER = 0x0004
            SWP_FRAMECHANGED = 0x0020
            ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED
            )
        except Exception:
            pass


def set_app_icon_and_taskbar(root, icon_filename="iconfin.ico"):
    """Applies window icon and sets Windows Process AppUserModelID."""
    icon_path = get_resource_path(icon_filename)

    if os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception:
            try:
                img = Image.open(icon_path)
                photo = ImageTk.PhotoImage(img)
                root.iconphoto(True, photo)
            except Exception:
                pass

    if sys.platform == "win32":
        try:
            import ctypes
            my_app_id = "msfs2xp.converter.1.0"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)
        except Exception:
            pass

    force_taskbar_icon(root)


def _clean_texture_stem(file_path):
    """Shared by clean_texture_name and decode_or_repackage_ktx2: the
    stem (no extension) after stripping MSFS's compound source
    extensions. Must stay in sync with
    mesh_convert.convert.clean_texture_stem (the compound-extension list
    is duplicated, not shared, since this one runs standalone during
    Step 2's bulk pre-decode pass before mesh_convert is even involved).
    Includes .tif/.tif.ktx2/.tif.dds since some packages (e.g. iniBuilds'
    EGLC) ship source textures as .TIF."""
    name = Path(file_path).name.lower()
    for ext in ['.png.ktx2', '.png.dds', '.tif.ktx2', '.tif.dds',
                '.ktx2', '.dds', '.tga', '.tiff', '.tif', '.jpg', '.jpeg', '.png']:
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def clean_texture_name(file_path):
    """Strips MSFS compound extensions to return a clean .png filename for
    X-Plane. See _clean_texture_stem for the actual stripping logic."""
    return _clean_texture_stem(file_path) + ".png"


def _available_memory_gb():
    """Currently-available physical RAM in GB, or None if it can't be
    determined (never raises)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / (1024 ** 3)
    except Exception:
        return None


def pool_worker_count():
    """os.cpu_count(), capped down when available RAM is tight. GPU fork
    note: unlike the original thread pools (which shared one process's
    memory), each process-pool worker carries its own full Python+numpy+
    PIL interpreter plus whatever a large/complex model (e.g. a multi-part
    terminal building) peaks at while converting -- on a memory-modest
    machine, running a full os.cpu_count() of those simultaneously can hit
    MemoryError where the single-process thread version wouldn't have.
    Budgets ~1.5GB of currently-available RAM per worker (generous for the
    overwhelming majority of models) and only pulls the worker count down
    when that's actually tight; well-specced machines are unaffected."""
    cpu_workers = os.cpu_count() or 1
    avail_gb = _available_memory_gb()
    if avail_gb is None:
        return cpu_workers
    mem_workers = max(1, int(avail_gb / 1.5))
    return max(1, min(cpu_workers, mem_workers))


def _terrain_fit_group_worker(obj_stems, base_lat, base_lon, heading_deg,
                              skip_draped, obj_dir_str, xplane_root_str):
    """Worker for the parallel terrain-fit pre-warm in step 4. Runs ONE
    terrain_fit group in this subprocess. The heavy part -- pickle-loading
    each sub-object's MeshIR, sampling the DEM, building the warped copy,
    writing the `*_tfit_*.obj` + `.meshir.pkl` -- is CPU- and GIL-bound, so
    this has to be a process, not a thread, to actually parallelize (a
    thread pool measured ~1 core of throughput). The `*_tfit_*` files land
    on the shared obj_dir; the small (all-str/bool) result dict returned
    here lets the parent's placement loop consume the fit without
    recomputing it. Returns (group_key, {stem: (result_stem, applied,
    reason)}, transform), or (group_key, None, None) if the fit raised.

    CONFIRMED REAL BUG: this used to return only the per-stem result dict.
    terrain_fit._group_transform_cache -- what the parent's own anchor-
    clustering pass (main.py, right after this pre-warm) reads via
    get_cached_transform() to find a vertical shift one placement's
    terrain-fit computed, and retroactively apply it to a DIFFERENT placement sharing
    the same real-world anchor (e.g. a building's own shell vs. its
    separately-placed glass/interior) -- lives in the terrain_fit MODULE
    in THIS subprocess. A ProcessPoolExecutor worker's module state is
    never shared back to the parent, so the parent's own cache stayed
    permanently empty for every group pre-warmed here (i.e. essentially
    all of them), silently no-opping the anchor-clustering pass entirely:
    0 sub-objects ever actually got linked on a real conversion, despite
    the log claiming candidates existed. Confirmed real symptom (user):
    differently-sourced parts of one building (materials/glass vs. shell)
    getting terrain-fitted independently instead of moving together."""
    import terrain_fit
    group_key = (tuple(sorted(obj_stems)), round(base_lat, 6), round(base_lon, 6),
                 round(heading_deg, 2), bool(skip_draped))
    try:
        res = terrain_fit.get_or_create_fitted_group(
            Path(obj_dir_str), list(obj_stems), base_lat, base_lon, heading_deg,
            Path(xplane_root_str), skip_draped_positions=bool(skip_draped))
        transform = terrain_fit.get_cached_transform(group_key)
        return group_key, res, transform
    except Exception:
        return group_key, None, None


# Matches the TEXTURE/TEXTURE_NORMAL/TEXTURE_LIT lines mesh_convert.convert
# writes into each .obj it produces, e.g. "TEXTURE ../textures/foo.png" --
# used by cached_convert() below to know exactly which texture files a
# cached model's .obj output depends on, without guessing.
_OBJ_TEXTURE_LINE_RE = re.compile(r'^TEXTURE(?:_NORMAL|_LIT)?\s+\.\./textures/(.+)$', re.MULTILINE)

# Sidecars a single convert() call can produce alongside each .obj, that a
# cache hit needs to restore too -- .meshir.pkl (mesh_convert.mesh_ir) is
# what terrain_fit.py/draped_merge.py consume instead of regex-parsing the
# .obj text; without restoring it on a cache hit, those two would see the
# .obj but no sidecar and have nothing to correct/merge for that model.
_CONVERT_SIDECAR_SUFFIXES = (".proximity.json", ".footprint.json", ".meshir.pkl")


def _is_valid_texture(path):
    """True only if path opens and fully decodes as an image -- guards
    cached_convert()'s own disk cache against permanently perpetuating a
    corrupted texture (a cache hit skips mesh_convert.convert() entirely,
    so a bad file already in the cache would otherwise keep getting
    replayed forever via a plain shutil.copy2). img.load(), not just the
    lazy Image.open(), forces the real decode."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def _package_version(package_module):
    """Like cache_utils.module_version, but for a PACKAGE (mesh_convert is
    a directory of .py files, not one file) -- combines the content hash
    of every .py file under the package's own directory so an edit to ANY
    of them (materials, flatness, animation, lights, draped_ranking,
    mesh_ir, convert.py itself) invalidates the cache, not just edits to
    __init__.py."""
    pkg_dir = Path(package_module.__file__).resolve().parent
    hashes = sorted(cache_utils.module_version(p) for p in pkg_dir.glob("*.py"))
    return hashlib.blake2b("".join(hashes).encode("utf-8"), digest_size=8).hexdigest()


def cached_convert(glb_path, obj_dir, tex_dir, ext_tex_dir, pitch, yaw, roll, disable_cache=False,
                    static_doors=False):
    """Disk-cached wrapper around mesh_convert.convert -- this is the
    picklable function submitted to the mesh-conversion process pool. On a
    cache hit it skips the actual GLTF parse/vertex processing/OBJ write
    entirely and just copies previously-produced files into place; on a
    miss it converts normally, then captures the .obj(s), any sidecars
    (see _CONVERT_SIDECAR_SUFFIXES), and exactly the texture files those
    .obj files reference (parsed out of their own TEXTURE* lines, so
    there's no reliance on guessing or on snapshotting tex_dir under
    concurrent workers) into the cache for next time.

    Safe under re-runs even after the source package changes: the cache key
    includes the .glb's own path+size+mtime, so a re-exported/updated model
    simply misses and reconverts. Safe across bug-fixing edits to the
    mesh_convert package too: the key also includes a combined content hash
    of every .py file in it (see _package_version), so any edit there
    invalidates every entry that depended on the old conversion logic.

    Also keys on bgl_extractor.py's own content hash, not just
    mesh_convert's: for an install-sourced SimObject, bgl_extractor.py
    (_copy_simobject_model) is what stages the model's referenced texture
    files next to it in models_dir BEFORE this ever runs -- an implicit
    on-disk input to convert() that isn't part of glb_path itself, so a fix
    there (e.g. widening the search for where a model's textures actually
    live) would otherwise leave every already-cached model silently
    pointing at whatever fallback/stub result the OLD staging produced,
    with nothing forcing a reconvert to pick the fix up.
    """
    key_parts = (
        cache_utils.file_identity(glb_path), str(pitch), str(yaw), str(roll),
        _package_version(mesh_convert),
        cache_utils.module_version(bgl_extractor.__file__),
        # external_textures_dir is a real input to convert() (extract_
        # image's package/base-game texture fallback search), not just an
        # output location -- changing it must invalidate a cached "texture
        # not found" result. Accepts a single path or a list (source
        # package root, then MSFS install root); stringified per-item
        # rather than relying on repr().
        "|".join(str(p) for p in ext_tex_dir) if isinstance(ext_tex_dir, (list, tuple)) else (str(ext_tex_dir) if ext_tex_dir else ""),
        str(static_doors),
    )
    obj_dir = Path(obj_dir)
    tex_dir = Path(tex_dir)

    cached = None if disable_cache else cache_utils.get("mesh_convert", *key_parts)
    if cached is not None:
        obj_names, sidecar_names, tex_names = cached
        src_dir = cache_utils.entry_dir("mesh_convert", *key_parts)
        if (all((src_dir / name).exists() for name in obj_names)
                and all(_is_valid_texture(src_dir / name) for name in tex_names if (src_dir / name).exists())):
            result_paths = []
            for name in obj_names:
                dst = obj_dir / name
                shutil.copy2(src_dir / name, dst)
                result_paths.append(dst)
            for name in sidecar_names:
                src = src_dir / name
                if src.exists():
                    shutil.copy2(src, obj_dir / name)
            for name in tex_names:
                src = src_dir / name
                dst = tex_dir / name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
            return result_paths
        # Cache entry incomplete/pruned -- fall through and reconvert.

    result_paths = mesh_convert.convert(glb_path, obj_dir, tex_dir, ext_tex_dir, pitch, yaw, roll,
                                         disable_proximity_animation=static_doors)

    if result_paths and not disable_cache:
        try:
            dest_dir = cache_utils.entry_dir("mesh_convert", *key_parts)
            obj_names, sidecar_names, tex_names = [], [], set()
            for p in result_paths:
                obj_names.append(p.name)
                shutil.copy2(p, dest_dir / p.name)
                try:
                    text = p.read_text(encoding="utf-8")
                except OSError:
                    text = ""
                tex_names.update(m.group(1).strip() for m in _OBJ_TEXTURE_LINE_RE.finditer(text))
                for suffix in _CONVERT_SIDECAR_SUFFIXES:
                    sidecar = p.with_name(p.stem + suffix)
                    if sidecar.exists():
                        sidecar_names.append(sidecar.name)
                        shutil.copy2(sidecar, dest_dir / sidecar.name)
            # Unlike the sidecars above, originoffset.json is keyed per
            # MODEL (glb_path's own stem), not per generated .obj stem --
            # convert() writes exactly one, shared across every sibling
            # .obj this model produced. Without caching it too, a cache
            # HIT would restore every .obj but never this file, silently
            # leaving main.py's placement-offset lookup with nothing to
            # find and the object un-recentered -- the recentering fix
            # would appear to stop working the moment a model is re-run
            # from cache instead of freshly converted.
            origin_sidecar = obj_dir / f"{Path(glb_path).stem}.originoffset.json"
            if origin_sidecar.exists():
                sidecar_names.append(origin_sidecar.name)
                shutil.copy2(origin_sidecar, dest_dir / origin_sidecar.name)
            for name in tex_names:
                src = tex_dir / name
                # _is_valid_texture guard: convert() itself should never
                # produce a broken file, but this is the one spot that
                # decides what gets locked into the PERSISTENT disk cache
                # for every future run -- worth the cheap decode check so
                # nothing here can start a corrupted texture's cache
                # lifetime in the first place (see _is_valid_texture's
                # docstring for why a bad entry, once cached, would
                # otherwise never get fixed by any later code change).
                if src.exists() and _is_valid_texture(src):
                    shutil.copy2(src, dest_dir / name)
            cache_utils.set("mesh_convert", (obj_names, sidecar_names, list(tex_names)), *key_parts)
        except OSError:
            pass  # Caching is a pure speed optimization -- never fail the conversion over it.

    return result_paths


# --- PURE PYTHON KTX2 DECODER ---
def parse_ktx2_header(file_path):
    with open(file_path, "rb") as f:
        magic = f.read(12)
        if magic != b"\xabKTX 20\xbb\r\n\x1a\n":
            raise ValueError(f"Not a valid KTX2 file: {file_path}")

        header_data = f.read(17 * 4)
        unpacked = struct.unpack("<17I", header_data)
        
        vk_format = unpacked[0]
        pixel_width = unpacked[2]
        pixel_height = unpacked[3]
        level_count = unpacked[7]
        supercompression_scheme = unpacked[8]

        level_index_data = f.read(level_count * 24)
        offset, length, _ = struct.unpack("<3Q", level_index_data[:24])

        f.seek(offset)
        compressed_bytes = f.read(length)

    return {
        "width": pixel_width,
        "height": pixel_height,
        "vk_format": vk_format,
        "supercompression": supercompression_scheme,
        "compressed_bytes": compressed_bytes,
    }

def decode_ktx2_to_png(input_path, output_path):
    try:
        info = parse_ktx2_header(input_path)

        scheme = info["supercompression"]
        if scheme == _KTX2_SUPERCOMPRESSION_BASIS:
            return "Skipped: true ETC1S/BasisLZ supercompression requires the Basis Universal transcoder."

        width = info["width"]
        height = info["height"]
        vk_fmt = info["vk_format"]

        raw_data = _decompress_supercompressed(info["compressed_bytes"], scheme)
        if raw_data is None:
            name = {2: "Zstd", 3: "ZLIB"}.get(scheme, str(scheme))
            return f"Skipped: {name}-supercompressed KTX2 but the '{name.lower()}' module isn't installed."

        if vk_fmt in (VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_UNORM_BLOCK):
            decoded = texture2ddecoder.decode_bc1(raw_data, width, height)
        elif vk_fmt == VK_FORMAT_BC3_UNORM_BLOCK:
            decoded = texture2ddecoder.decode_bc3(raw_data, width, height)
        elif vk_fmt == VK_FORMAT_BC4_UNORM_BLOCK:
            decoded = texture2ddecoder.decode_bc4(raw_data, width, height)
        elif vk_fmt in (VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC5_SNORM_BLOCK):
            bc5_data = raw_data
            if vk_fmt == VK_FORMAT_BC5_SNORM_BLOCK:
                bc5_data = _bc5_snorm_to_unorm_bytes(bc5_data)
            decoded = texture2ddecoder.decode_bc5(bc5_data, width, height)
        elif vk_fmt == VK_FORMAT_BC7_UNORM_BLOCK:
            decoded = texture2ddecoder.decode_bc7(raw_data, width, height)
        elif vk_fmt in (VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_FORMAT_ASTC_4x4_SRGB_BLOCK):
            # UASTC textures are plain ASTC 4x4 blocks -- no Basis transcoder needed here.
            decoded = texture2ddecoder.decode_astc(raw_data, width, height, 4, 4)
        elif vk_fmt == VK_FORMAT_R8G8B8A8_UNORM:
            decoded = raw_data
        else:
            return f"Unsupported VkFormat: {vk_fmt}"

        img = Image.frombytes("RGBA", (width, height), decoded, "raw", "BGRA")

        # FIX: Only run normal map math on _norm textures. Do not touch _comp textures.
        if "_norm" in Path(input_path).name.lower():
            image_data = np.array(img)
            if len(image_data.shape) >= 3 and image_data.shape[2] >= 2:
                x = image_data[..., 0].astype(np.float32) / 127.5 - 1.0
                y = image_data[..., 1].astype(np.float32) / 127.5 - 1.0
                z = np.sqrt(np.clip(1.0 - x*x - y*y, 0.0, 1.0))
                
                new_image = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
                new_image[..., 0] = image_data[..., 0]
                new_image[..., 1] = image_data[..., 1]
                new_image[..., 2] = np.clip(z * 255.0, 0, 255).astype(np.uint8)
                img = Image.fromarray(new_image)

        img.save(output_path, "PNG")
        return True
    except Exception as e:
        return str(e)


def repackage_ktx2_to_dds(input_path, output_path):
    """Wraps a KTX2 file's own GPU-native block-compressed payload
    directly in a DDS container instead of fully decoding to raw pixels
    and re-encoding as PNG -- same bytes, different header, so X-Plane's
    DDS loader reads the identical GPU data with zero quality loss and
    none of the decode+encode cost or size/VRAM penalty. Returns True on
    success, or a message string explaining why repackaging wasn't
    possible -- same calling convention as decode_ktx2_to_png, and the
    caller (decode_or_repackage_ktx2) falls back to that full decode in
    that case.

    ONLY BC1 and BC3 are repackaged (legacy "DXT1"/"DXT5" FourCC, no
    DDS_HEADER_DXT10 extension) -- CONFIRMED REAL REGRESSION: an earlier
    version of this also repackaged BC4/BC5/BC7 via a DX10-header DDS,
    which round-tripped fine through this project's OWN reader but made
    "almost everything" grey with "Some scenery textures could not be
    loaded" in real X-Plane 11. X-Plane's own official DDSTool manual
    (developer.x-plane.com/docs/scenery/ddstool-manual) documents ONLY
    DXT1/DXT3/DXT5 support and never mentions BC4/BC5/BC7 or a DX10
    header anywhere -- X-Plane 11's DDS loader appears to predate that
    extension entirely. BC7 in particular is the common format for
    modern MSFS albedo/PBR textures, so this cost most of the real-world
    win the earlier version measured; still correctly skips ahead of the
    "Unsupported VkFormat" fallback to the safe, proven full-decode path
    for every format it can't safely handle, rather than guessing again.

    MUST NOT be used for a normal map: BC5-compressed normal maps only
    carry 2 channels (X/Y) -- decode_ktx2_to_png's own Z-channel
    reconstruction (sqrt(1 - x^2 - y^2), a few lines above) has to
    actually run and bake a real 3rd channel in. A raw block-data
    passthrough would skip that entirely and silently produce broken
    lighting in-sim. The caller is responsible for routing "_norm"
    textures to decode_ktx2_to_png instead, using the same filename
    convention decode_ktx2_to_png itself already keys its own
    reconstruction on."""
    try:
        info = parse_ktx2_header(input_path)
    except Exception as e:
        return str(e)

    scheme = info["supercompression"]
    if scheme == _KTX2_SUPERCOMPRESSION_BASIS:
        return "Skipped: true ETC1S/BasisLZ supercompression requires the Basis Universal transcoder."

    width, height, vk_fmt = info["width"], info["height"], info["vk_format"]
    raw_data = _decompress_supercompressed(info["compressed_bytes"], scheme)
    if raw_data is None:
        name = {2: "Zstd", 3: "ZLIB"}.get(scheme, str(scheme))
        return f"Skipped: {name}-supercompressed KTX2 but the '{name.lower()}' module isn't installed."

    if vk_fmt in (VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_UNORM_BLOCK):
        dds_bytes = _build_dds_bytes(width, height, raw_data, b"DXT1", block_size=8)
    elif vk_fmt == VK_FORMAT_BC3_UNORM_BLOCK:
        dds_bytes = _build_dds_bytes(width, height, raw_data, b"DXT5", block_size=16)
    else:
        return f"Unsupported VkFormat for repackaging (X-Plane 11's DDS loader has no confirmed DX10/BC4-7 support): {vk_fmt}"

    try:
        output_path.write_bytes(dds_bytes)
    except OSError as e:
        return str(e)
    return True


def decode_or_repackage_ktx2(input_path, out_dir):
    """Picklable ProcessPoolExecutor entry point for Step 2's bulk KTX2
    pass. Repackages to .dds (near-zero cost, zero quality loss) for
    every texture EXCEPT a normal map ("_norm" in the name, the same
    convention decode_ktx2_to_png itself already keys its Z-channel
    reconstruction on), which still needs the real decode+reconstruct+
    re-encode -- a raw passthrough would silently break its lighting.
    Falls back to the full decode-to-PNG path for anything
    repackage_ktx2_to_dds can't handle (true Basis/ETC1S
    supercompression, an unsupported VkFormat, or a missing
    decompression module)."""
    stem = _clean_texture_stem(input_path)
    if "_norm" not in stem:
        dds_path = out_dir / f"{stem}.dds"
        if repackage_ktx2_to_dds(input_path, dds_path) is True:
            return True
    png_path = out_dir / f"{stem}.png"
    return decode_ktx2_to_png(input_path, png_path)


# --- CANVAS ROUNDED GRAPHICS HELPERS ---
def draw_rounded_polygon(canvas, x1, y1, x2, y2, radius=10, **kwargs):
    """Draws a smooth rounded rectangle on a Tkinter Canvas."""
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    radius = max(2.0, min(radius, (x2 - x1) / 2.0, (y2 - y1) / 2.0))
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


# --- CUSTOM ROUNDED WIDGETS ---
class RoundedCard(tk.Canvas):
    """A dark card container with smooth rounded corners."""
    def __init__(self, parent, bg_color="#141414", card_bg="#1c1c1e", border_color="#2a2a2e", radius=12, **kwargs):
        super().__init__(parent, bg=bg_color, highlightthickness=0, bd=0, **kwargs)
        self.bg_color = bg_color
        self.card_bg = card_bg
        self.border_color = border_color
        self.radius = radius
        
        self.inner = tk.Frame(self, bg=card_bg)
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        self.delete("card_bg")
        w, h = event.width, event.height
        if w > 2 and h > 2:
            draw_rounded_polygon(self, 1, 1, w - 1, h - 1, self.radius,
                                 fill=self.card_bg, outline=self.border_color, width=1, tags="card_bg")
            self.create_window(w / 2, h / 2, window=self.inner, width=w - 24, height=h - 24)


class RoundedButton(tk.Canvas):
    """A custom interactive button with smooth rounded borders."""
    def __init__(self, parent, text="", command=None, radius=8, bg_color="#1c1c1e",
                 btn_bg="#28282e", hover_bg="#38383e", active_bg="#1f1f24", disabled_bg="#18181a",
                 fg="#ffffff", disabled_fg="#55555e", font=("Segoe UI", 9, "bold"), width=85, height=32, **kwargs):
        super().__init__(parent, width=width, height=height, bg=bg_color, highlightthickness=0, bd=0, cursor="hand2", **kwargs)
        self.command = command
        self.radius = radius
        self.btn_bg = btn_bg
        self.hover_bg = hover_bg
        self.active_bg = active_bg
        self.disabled_bg = disabled_bg
        self.curr_bg = btn_bg
        self.fg = fg
        self.disabled_fg = disabled_fg
        self.text_str = text
        self.font = font
        self.state = "normal"
        
        self.bind("<Configure>", self.redraw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def redraw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w > 2 and h > 2:
            bg = self.curr_bg if self.state == "normal" else self.disabled_bg
            fg = self.fg if self.state == "normal" else self.disabled_fg
            draw_rounded_polygon(self, 1, 1, w - 1, h - 1, self.radius, fill=bg, outline=bg)
            self.create_text(w / 2, h / 2, text=self.text_str, fill=fg, font=self.font)

    def _on_enter(self, e):
        if self.state == "normal":
            self.curr_bg = self.hover_bg
            self.redraw()

    def _on_leave(self, e):
        if self.state == "normal":
            self.curr_bg = self.btn_bg
            self.redraw()

    def _on_press(self, e):
        if self.state == "normal":
            self.curr_bg = self.active_bg
            self.redraw()

    def _on_release(self, e):
        if self.state == "normal":
            self.curr_bg = self.hover_bg
            self.redraw()
            if self.command:
                self.command()

    def set_state(self, state):
        self.state = state
        self.config(cursor="hand2" if state == "normal" else "arrow")
        self.redraw()


class RoundedEntry(tk.Canvas):
    """An input field enclosed within a rounded border box."""
    def __init__(self, parent, textvariable=None, radius=8, bg_color="#1c1c1e",
                 entry_bg="#101012", border_color="#2a2a2e", fg="#ffffff", font=("Segoe UI", 9), **kwargs):
        super().__init__(parent, height=32, bg=bg_color, highlightthickness=0, bd=0, **kwargs)
        self.radius = radius
        self.entry_bg = entry_bg
        self.border_color = border_color
        
        self.entry = tk.Entry(self, textvariable=textvariable, bg=entry_bg, fg=fg,
                              insertbackground=fg, bd=0, font=font, highlightthickness=0)
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        self.delete("all")
        w, h = event.width, event.height
        if w > 2 and h > 2:
            draw_rounded_polygon(self, 1, 1, w - 1, h - 1, self.radius, fill=self.entry_bg, outline=self.border_color, width=1)
            self.create_window(12, h / 2, window=self.entry, anchor="w", width=w - 24)


class RoundedProgressBar(tk.Canvas):
    """A pill-shaped smooth custom progress bar."""
    def __init__(self, parent, height=14, radius=7, bg_color="#1c1c1e",
                 trough_color="#101012", bar_color="#2563eb", **kwargs):
        super().__init__(parent, height=height, bg=bg_color, highlightthickness=0, bd=0, **kwargs)
        self.radius = radius
        self.trough_color = trough_color
        self.bar_color = bar_color
        self.value = 0
        self.maximum = 100
        self.mode = "determinate"
        self._anim_pos = 0
        self._anim_job = None
        self.bind("<Configure>", self.redraw)

    def set_progress(self, val, max_val=100):
        self.stop_animation()
        self.mode = "determinate"
        self.value = val
        self.maximum = max_val
        self.redraw()

    def set_color(self, color):
        self.bar_color = color
        self.redraw()

    def start_indeterminate(self):
        self.mode = "indeterminate"
        if not self._anim_job:
            self._animate()

    def stop_animation(self):
        if self._anim_job:
            self.after_cancel(self._anim_job)
            self._anim_job = None

    def _animate(self):
        if self.mode == "indeterminate":
            self._anim_pos = (self._anim_pos + 4) % 100
            self.redraw()
            self._anim_job = self.after(30, self._animate)

    def redraw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 2 or h <= 2:
            return
            
        r = min(self.radius, h / 2)
        draw_rounded_polygon(self, 0, 0, w, h, r, fill=self.trough_color, outline="")
        
        if self.mode == "determinate":
            if self.maximum > 0 and self.value > 0:
                pct = min(1.0, max(0.0, self.value / float(self.maximum)))
                pw = max(r * 2, int(w * pct))
                draw_rounded_polygon(self, 0, 0, min(w, pw), h, r, fill=self.bar_color, outline="")
        else:
            bw = int(w * 0.3)
            start_x = int((w + bw) * (self._anim_pos / 100.0)) - bw
            end_x = start_x + bw
            
            x1 = max(0, start_x)
            x2 = min(w, end_x)
            if x2 - x1 > r:
                draw_rounded_polygon(self, x1, 0, x2, h, r, fill=self.bar_color, outline="")


# --- MAIN APP CLASS ---
class ModularPythonConverterApp:
    def __init__(self, root):
        self.root = root
        
        # Hide window initially to prevent taskbar glitches
        self.root.withdraw()
        
        self.root.overrideredirect(True)
        self.root.geometry("1240x758")
        self.root.minsize(1100, 638)
        
        self.pkg_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.msfs_var = tk.StringVar()  # optional: MSFS install root, for stock/ASOBO object resolution
        # Optional: your own copy of the MSFS SDK's "Propdefs" folder,
        # needed to decode SimPropContainer-based placements (apron
        # lights/lamps, jetways, animated doors, building interiors). Not
        # bundled with this app -- those are Microsoft/Asobo's own SDK
        # data files, not this project's to redistribute.
        self.propdefs_var = tk.StringVar()
        # xp12: modern X-Plane 11.50+/XP12 apt.dat spec (1200, the default
        # apt_dat.write_apt_dat already writes as the ACTIVE "apt.dat").
        # xp11: legacy pre-11.50 X-Plane 11 -- write_apt_dat already
        # produces this exact file too (spec 1100, jetway rows stripped),
        # today only as a ".xp11" sidecar the user has to manually rename
        # over apt.dat; this toggle makes it the active file instead when
        # the user's actual target is an older XP11 install.
        self.xp_version_var = tk.StringVar(value="xp12")
        self.clean_run_var = tk.BooleanVar(value=False)
        self.disable_cache_var = tk.BooleanVar(value=False)
        # Proximity-triggered animations (doors that open as the aircraft
        # approaches) need the companion msfs2xp_proximity_animator
        # FlyWithLua script to actually move -- without it, or while the
        # animated geometry is still misplaced, render them as static
        # rigid geometry at rest pose instead (see mesh_convert.convert's
        # disable_proximity_animation parameter). Default ON.
        self.static_doors_var = tk.BooleanVar(value=True)
        # See resolve_library_substitution's own docstring for what this
        # does (a local string-similarity heuristic, not a hosted service).
        self.approximate_substitution_var = tk.BooleanVar(value=False)
        # Scans BGL TerrainVectorDb sections (MSFS World Editor's vector-
        # polygon/vegetation database -- some packages, e.g. iniBuilds'
        # EGLC, store their whole taxiway/runway/apron pavement here
        # instead of as placed glTF models) and logs which ground
        # materials each polygon references. Diagnostic-only (material
        # references, not geometry) -- see bgl_extractor.scan_terrain_
        # vector_db's own docstring for what was and wasn't decoded.
        self.scan_terrain_vectors_var = tk.BooleanVar(value=True)
        # Opt-in .pol/DSF-polygon rollout: draped ground content (pavement
        # fills, painted markings/signs) as real X-Plane .pol draped
        # polygons via native DSF primitives instead of OBJ8 .obj files
        # with ATTR_draped -- see draped_merge.merge_draped_layers_in_
        # tile's own docstring. Off keeps the existing OBJ8 path, which
        # stays a permanent fallback, not a temporary one.
        self.pol_polygons_var = tk.BooleanVar(value=False)
        # When on (default), a conversion that hits objects it can't match
        # to any converted geometry or substitution pauses just before DSF
        # compile and pops up the replacement picker (pick_replacements) so
        # the user can map them to X-Plane library objects / skip them, and
        # those picks are applied to THIS run. Headless runners set this
        # False (there's no display / event loop to host the dialog).
        self.prompt_replacements_var = tk.BooleanVar(value=True)
        self.is_maximized = False
        self._is_minimizing = False
        
        if sys.platform == "win32":
            try:
                self.root.wm_attributes("-transparentcolor", TRANSPARENT_KEY)
            except Exception:
                pass
        self.root.configure(bg=TRANSPARENT_KEY)

        set_app_icon_and_taskbar(self.root, "iconfin.ico")

        self.build_custom_window_shell()
        self.build_ui()
        self.load_config()

        # Reveal window and force taskbar refresh after window map initialization
        self.root.deiconify()
        self.root.after(50, lambda: force_taskbar_icon(self.root))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Map>", self._on_window_map)

    def build_custom_window_shell(self):
        self.window_canvas = tk.Canvas(self.root, bg=TRANSPARENT_KEY, highlightthickness=0, bd=0)
        self.window_canvas.pack(fill=tk.BOTH, expand=True)

        self.bg_dark = "#141414"
        self.card_bg = "#1c1c1e"
        self.title_bg = "#181818"
        self.accent_blue = "#2563eb"
        self.border_color = "#2a2a2e"

        self.main_container = tk.Frame(self.window_canvas, bg=self.bg_dark)
        
        self.window_canvas.bind("<Configure>", self._draw_window_shape)

        # --- CUSTOM TITLE BAR ---
        self.title_bar = tk.Frame(self.main_container, bg=self.title_bg, height=38)
        self.title_bar.pack(fill=tk.X, side=tk.TOP)
        self.title_bar.pack_propagate(False)

        dots_frame = tk.Frame(self.title_bar, bg=self.title_bg)
        dots_frame.pack(side=tk.LEFT, padx=14)

        def create_dot(parent, color, hover_color, command):
            canvas = tk.Canvas(parent, width=12, height=12, bg=self.title_bg, highlightthickness=0, cursor="hand2")
            dot = canvas.create_oval(1, 1, 11, 11, fill=color, outline=color)
            canvas.bind("<Enter>", lambda e: canvas.itemconfig(dot, fill=hover_color, outline=hover_color))
            canvas.bind("<Leave>", lambda e: canvas.itemconfig(dot, fill=color, outline=color))
            canvas.bind("<Button-1>", lambda e: command())
            return canvas

        self.close_dot = create_dot(dots_frame, "#ff5f56", "#e0443e", self.on_close)
        self.close_dot.pack(side=tk.LEFT, padx=3)

        self.min_dot = create_dot(dots_frame, "#ffbd2e", "#dea123", self.minimize_window)
        self.min_dot.pack(side=tk.LEFT, padx=3)

        self.max_dot = create_dot(dots_frame, "#27c93f", "#1aab29", self.toggle_maximize)
        self.max_dot.pack(side=tk.LEFT, padx=3)

        self.title_label = tk.Label(self.title_bar, text="MSFS2XP", bg=self.title_bg,
                                    fg="#a0a0a0", font=("Segoe UI", 9, "bold"))
        self.title_label.pack(side=tk.LEFT, padx=12)

        for widget in (self.title_bar, self.title_label):
            widget.bind("<Button-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._do_drag)
            widget.bind("<Double-Button-1>", lambda e: self.toggle_maximize())

        self.grip = tk.Label(self.main_container, text="◢", bg=self.bg_dark, fg="#33333e", cursor="size_nw_se")
        self.grip.place(relx=1.0, rely=1.0, anchor="se")
        self.grip.bind("<Button-1>", self._start_resize)
        self.grip.bind("<B1-Motion>", self._do_resize)

    def _draw_window_shape(self, event):
        self.window_canvas.delete("win_bg")
        w, h = event.width, event.height
        if w > 2 and h > 2:
            r = 14 if not self.is_maximized else 0
            draw_rounded_polygon(self.window_canvas, 0, 0, w, h, radius=r,
                                 fill=self.bg_dark, outline=self.border_color, width=1, tags="win_bg")
            self.window_canvas.create_window(w / 2, h / 2, window=self.main_container, width=w - 2, height=h - 2)

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _do_drag(self, event):
        if self.is_maximized:
            return
        x = self.root.winfo_x() + (event.x - self._drag_x)
        y = self.root.winfo_y() + (event.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")

    def _start_resize(self, event):
        self._resize_x = event.x_root
        self._resize_y = event.y_root
        self._start_w = self.root.winfo_width()
        self._start_h = self.root.winfo_height()

    def _do_resize(self, event):
        delta_w = event.x_root - self._resize_x
        delta_h = event.y_root - self._resize_y
        new_w = max(800, self._start_w + delta_w)
        new_h = max(600, self._start_h + delta_h)
        self.root.geometry(f"{new_w}x{new_h}")

    def minimize_window(self):
        self._is_minimizing = True
        self.root.overrideredirect(False)
        self.root.iconify()

    def _on_window_map(self, event):
        if event.widget == self.root and self.root.state() == 'normal':
            if getattr(self, '_is_minimizing', False):
                self.root.overrideredirect(True)
                force_taskbar_icon(self.root)
                self._is_minimizing = False

    def toggle_maximize(self):
        if not self.is_maximized:
            self._prev_geom = self.root.geometry()
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            self.root.geometry(f"{sw}x{sh}+0+0")
            self.is_maximized = True
        else:
            if hasattr(self, '_prev_geom'):
                self.root.geometry(self._prev_geom)
            self.is_maximized = False
        self._draw_window_shape(type('Event', (), {'width': self.root.winfo_width(), 'height': self.root.winfo_height()})())

    def load_config(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                    self.pkg_var.set(data.get("pkg_dir", ""))
                    self.out_var.set(data.get("out_dir", ""))
                    self.msfs_var.set(data.get("msfs_install_dir", ""))
                    self.propdefs_var.set(data.get("propdefs_dir", ""))
                    self.xp_version_var.set(data.get("xp_version", "xp12"))
                    self.clean_run_var.set(data.get("clean_run", False))
                    self.disable_cache_var.set(data.get("disable_cache", False))
                    # Default True here too, matching the BooleanVar's own
                    # declared default above -- an old saved config from
                    # before this key existed (or from a run where it was
                    # left unchecked) must not silently revert a returning
                    # user to animated, unreliable doors; "static except
                    # lights" is the intended default regardless of what an
                    # old config file does or doesn't have recorded.
                    self.static_doors_var.set(data.get("static_doors", True))
                    self.approximate_substitution_var.set(data.get("approximate_substitution", False))
                    self.scan_terrain_vectors_var.set(data.get("scan_terrain_vectors", True))
                    self.pol_polygons_var.set(data.get("pol_polygons", False))
                    geom = data.get("geometry")
                    if geom:
                        self.root.geometry(geom)
            except Exception:
                pass

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump({
                    "pkg_dir": self.pkg_var.get(),
                    "out_dir": self.out_var.get(),
                    "msfs_install_dir": self.msfs_var.get(),
                    "propdefs_dir": self.propdefs_var.get(),
                    "xp_version": self.xp_version_var.get(),
                    "clean_run": self.clean_run_var.get(),
                    "disable_cache": self.disable_cache_var.get(),
                    "static_doors": self.static_doors_var.get(),
                    "approximate_substitution": self.approximate_substitution_var.get(),
                    "scan_terrain_vectors": self.scan_terrain_vectors_var.get(),
                    "pol_polygons": self.pol_polygons_var.get(),
                    "geometry": self.root.geometry()
                }, f)
        except Exception:
            pass

    def on_close(self):
        self.save_config()
        self.root.destroy()

    def build_ui(self):
        main = tk.Frame(self.main_container, bg=self.bg_dark)
        main.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Two columns: the controls (setup / options / progress / execute)
        # stack down a fixed-width left column, and the run log gets its
        # own full-height panel on the right instead of being crammed
        # under everything. left_col has BOTH width and height pinned with
        # geometry propagation off, so its child cards can't drive it wider
        # and the split stays stable.
        body = tk.Frame(main, bg=self.bg_dark)
        body.pack(fill=tk.BOTH, expand=True)
        left_col = tk.Frame(body, bg=self.bg_dark, width=596, height=678)
        left_col.pack(side=tk.LEFT, fill=tk.Y)
        left_col.pack_propagate(False)
        right_col = tk.Frame(body, bg=self.bg_dark)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14, 0))

        # --- SETUP CARD ---
        setup_card = RoundedCard(left_col, bg_color=self.bg_dark, card_bg=self.card_bg, radius=12, height=201)
        setup_card.pack(fill=tk.X, pady=(0, 12))
        
        setup_inner = setup_card.inner
        setup_inner.columnconfigure(1, weight=1)

        tk.Label(setup_inner, text="SETUP", bg=self.card_bg, fg="#3b82f6", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        tk.Label(setup_inner, text="Source:", bg=self.card_bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(
            row=1, column=0, sticky="e", padx=(0, 10), pady=4)
        
        self.src_entry = RoundedEntry(setup_inner, textvariable=self.pkg_var, bg_color=self.card_bg)
        self.src_entry.grid(row=1, column=1, sticky="we", padx=5, pady=4)
        
        RoundedButton(setup_inner, text="Browse...", command=lambda: self.pkg_var.set(filedialog.askdirectory()),
                      bg_color=self.card_bg, width=85, height=30).grid(row=1, column=2, padx=(5, 0), pady=4)

        tk.Label(setup_inner, text="Target:", bg=self.card_bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(
            row=2, column=0, sticky="e", padx=(0, 10), pady=4)
        
        self.tgt_entry = RoundedEntry(setup_inner, textvariable=self.out_var, bg_color=self.card_bg)
        self.tgt_entry.grid(row=2, column=1, sticky="we", padx=5, pady=4)
        
        RoundedButton(setup_inner, text="Browse...", command=lambda: self.out_var.set(filedialog.askdirectory()),
                      bg_color=self.card_bg, width=85, height=30).grid(row=2, column=2, padx=(5, 0), pady=4)

        tk.Label(setup_inner, text="MSFS 2020 (optional):", bg=self.card_bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(
            row=3, column=0, sticky="e", padx=(0, 10), pady=4)

        self.msfs_entry = RoundedEntry(setup_inner, textvariable=self.msfs_var, bg_color=self.card_bg)
        self.msfs_entry.grid(row=3, column=1, sticky="we", padx=5, pady=4)

        RoundedButton(setup_inner, text="Browse...", command=lambda: self.msfs_var.set(filedialog.askdirectory()),
                      bg_color=self.card_bg, width=85, height=30).grid(row=3, column=2, padx=(5, 0), pady=4)

        tk.Label(setup_inner, text="Propdefs folder (optional):", bg=self.card_bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(
            row=4, column=0, sticky="e", padx=(0, 10), pady=4)

        # Your own copy of the MSFS SDK's "Propdefs" XML folder --
        # Microsoft/Asobo's own SDK data, not something this app can
        # legally bundle. Needed only to decode SimPropContainer-based
        # placements (many apron lights/lamps, jetways, animated doors,
        # building interiors); everything else converts fine without it.
        self.propdefs_entry = RoundedEntry(setup_inner, textvariable=self.propdefs_var, bg_color=self.card_bg)
        self.propdefs_entry.grid(row=4, column=1, sticky="we", padx=5, pady=4)

        RoundedButton(setup_inner, text="Browse...", command=lambda: self.propdefs_var.set(filedialog.askdirectory()),
                      bg_color=self.card_bg, width=85, height=30).grid(row=4, column=2, padx=(5, 0), pady=4)

        # --- OPTIONS CARD ---
        options_card = RoundedCard(left_col, bg_color=self.bg_dark, card_bg=self.card_bg, radius=12, height=222)
        options_card.pack(fill=tk.X, pady=(0, 12))

        options_inner = options_card.inner
        options_inner.columnconfigure(3, weight=1)

        tk.Label(options_inner, text="OPTIONS", bg=self.card_bg, fg="#3b82f6", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        def _styled_checkbutton(parent, text, variable):
            return tk.Checkbutton(
                parent, text=text, variable=variable, bg=self.card_bg, fg="#cccccc",
                activebackground=self.card_bg, activeforeground="#ffffff", selectcolor=self.bg_dark,
                highlightthickness=0, bd=0, font=("Segoe UI", 9), anchor="w")

        def _styled_radiobutton(parent, text, variable, value):
            return tk.Radiobutton(
                parent, text=text, variable=variable, value=value, bg=self.card_bg, fg="#cccccc",
                activebackground=self.card_bg, activeforeground="#ffffff", selectcolor=self.bg_dark,
                highlightthickness=0, bd=0, font=("Segoe UI", 9), anchor="w")

        tk.Label(options_inner, text="Target X-Plane:", bg=self.card_bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(
            row=1, column=0, sticky="e", padx=(0, 10), pady=4)
        _styled_radiobutton(options_inner, "XP11.50+ / XP12 (modern apt.dat)", self.xp_version_var, "xp12").grid(
            row=1, column=1, sticky="w", pady=4)
        _styled_radiobutton(options_inner, "XP11 legacy (pre-11.50)", self.xp_version_var, "xp11").grid(
            row=1, column=2, sticky="w", pady=4)

        _styled_checkbutton(
            options_inner, "Clean run (wipe cache + temp files first)", self.clean_run_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        _styled_checkbutton(
            options_inner, "Disable cache for this run", self.disable_cache_var
        ).grid(row=2, column=2, sticky="w", pady=4)
        RoundedButton(options_inner, text="Clear Cache Now", command=self.clear_cache_now,
                      bg_color=self.card_bg, width=140, height=28).grid(row=2, column=3, sticky="e", pady=4)

        _styled_checkbutton(
            options_inner, "Static doors (disable proximity- and business-hours-triggered animations)", self.static_doors_var
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=4)
        _styled_checkbutton(
            options_inner, "Approximate name-matching for unresolved base-game objects (heuristic)",
            self.approximate_substitution_var
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=4)
        _styled_checkbutton(
            options_inner, "Scan for TerrainVectorDb ground-polygon materials (diagnostic log only, no geometry)",
            self.scan_terrain_vectors_var
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=4)
        _styled_checkbutton(
            options_inner, "Convert draped pavement/markings to real .pol DSF polygons (experimental)",
            self.pol_polygons_var
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=4)

        # --- PROGRESS CARD ---
        prog_card = RoundedCard(left_col, bg_color=self.bg_dark, card_bg=self.card_bg, radius=12, height=210)
        prog_card.pack(fill=tk.X, pady=(0, 12))
        
        prog_inner = prog_card.inner
        prog_inner.columnconfigure(1, weight=1)

        tk.Label(prog_inner, text="PROGRESS", bg=self.card_bg, fg="#3b82f6", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        tk.Label(prog_inner, text="Overall pipeline:", bg=self.card_bg, fg="#cccccc", font=("Segoe UI", 9)).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        
        self.overall_bar = RoundedProgressBar(prog_inner, height=16, radius=8, bg_color=self.card_bg, bar_color=self.accent_blue)
        self.overall_bar.grid(row=2, column=0, columnspan=2, sticky="we", pady=(0, 8))

        tk.Frame(prog_inner, bg="#2a2a2e", height=1).grid(row=3, column=0, columnspan=2, sticky="we", pady=(0, 8))

        step_frame = tk.Frame(prog_inner, bg=self.card_bg)
        step_frame.grid(row=4, column=0, columnspan=2, sticky="we")
        step_frame.columnconfigure(2, weight=1)

        def create_step_row(parent, number, label_text, row):
            tk.Label(parent, text=f"#{number}", bg=self.card_bg, fg="#3b82f6", font=("Segoe UI", 9, "bold"), width=3, anchor="w").grid(
                row=row, column=0, sticky="w", padx=(0, 4), pady=3)
            tk.Label(parent, text=label_text, bg=self.card_bg, fg="#888888", font=("Segoe UI", 9), width=16, anchor="w").grid(
                row=row, column=1, sticky="w", padx=(0, 10), pady=3)
            bar = RoundedProgressBar(parent, height=12, radius=6, bg_color=self.card_bg, bar_color=self.accent_blue)
            bar.grid(row=row, column=2, sticky="we", pady=3)
            return bar

        self.step1_bar = create_step_row(step_frame, 1, "BGL extraction", 0)
        self.step2_bar = create_step_row(step_frame, 2, "Texture extraction", 1)
        self.step3_bar = create_step_row(step_frame, 3, "Mesh conversion", 2)
        self.step4_bar = create_step_row(step_frame, 4, "DSF compilation", 3)

        # --- EXECUTE BUTTON ---
        btn_frame = tk.Frame(left_col, bg=self.bg_dark)
        btn_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.start_btn = RoundedButton(btn_frame, text="EXECUTE", command=self.start, radius=10,
                                       bg_color=self.bg_dark, btn_bg=self.accent_blue, hover_bg="#1d4ed8",
                                       active_bg="#1e40af", fg="#ffffff", font=("Segoe UI", 10, "bold"),
                                       width=160, height=38)
        self.start_btn.pack()

        # --- LOGS CARD (own full-height panel, right column) ---
        log_card = RoundedCard(right_col, bg_color=self.bg_dark, card_bg=self.card_bg, radius=12)
        log_card.pack(fill=tk.BOTH, expand=True)
        
        log_inner = log_card.inner
        
        tk.Label(log_inner, text="Logs", bg=self.card_bg, fg="#3b82f6", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))

        self.log_area = ScrolledText(log_inner, bg="#0d0d0f", fg="#cccccc", font=("Consolas", 9),
                                     bd=0, highlightthickness=0, insertbackground="#ffffff")
        self.log_area.pack(fill=tk.BOTH, expand=True)

        self.log_area.tag_config("info", foreground="#cccccc")
        self.log_area.tag_config("success", foreground="#4ade80")
        self.log_area.tag_config("warning", foreground="#facc15")
        self.log_area.tag_config("error", foreground="#f87171")
        self.log_area.tag_config("header", foreground="#38bdf8", font=("Consolas", 10, "bold"))

    def _append_log(self, text, level):
        self.log_area.insert(tk.END, text + "\n", level)
        self.log_area.see(tk.END)

    def log(self, text, level="info"):
        self.root.after(0, self._append_log, text, level)
        
    def update_progress(self, bar, value, maximum=100):
        self.root.after(0, lambda: bar.set_progress(value, maximum))

    def stop_indeterminate(self, bar):
        self.root.after(0, lambda: bar.set_progress(0, 100))

    def mark_overall_complete(self):
        self.root.after(0, self._mark_overall_complete)

    def _mark_overall_complete(self):
        self.overall_bar.set_color("#22c55e")
        self.overall_bar.set_progress(100, 100)

    def reset_overall_bar(self):
        self.overall_bar.set_color(self.accent_blue)
        self.overall_bar.set_progress(0, 100)

    def set_indeterminate(self, bar):
        self.root.after(0, lambda: bar.start_indeterminate())

    def start(self):
        if not self.pkg_var.get() or not self.out_var.get():
            messagebox.showerror("Error", "Please provide both directories.")
            return
            
        self.save_config()
        self.start_btn.set_state("disabled")
        self.log_area.delete(1.0, tk.END)
        self.reset_overall_bar()
        
        for bar in (self.overall_bar, self.step1_bar, self.step2_bar, self.step3_bar, self.step4_bar):
            bar.set_color(self.accent_blue)
            self.stop_indeterminate(bar)
            self.update_progress(bar, 0)
            
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def _wipe_cache_and_temp(self, log_fn):
        """Deletes cache_utils' persistent disk cache and every leftover
        _temp/ scratch folder (a hard-killed/crashed run skips the
        finally: block that normally self-cleans _temp/py-msfs-*, which
        can leave multi-GB orphans behind). Never raises -- both a "clean
        run" and the standalone Clear Cache button treat a failed delete
        as a warning, not a reason to abort."""
        cache_root = cache_utils.cache_root()
        if cache_root.exists():
            try:
                shutil.rmtree(cache_root)
                log_fn(f"Cleared disk cache: {cache_root}", "info")
            except OSError as e:
                log_fn(f"Could not fully clear disk cache {cache_root}: {e}", "warning")

        if _LOCAL_TEMP_ROOT.exists():
            removed = 0
            for entry in _LOCAL_TEMP_ROOT.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                    removed += 1
                except OSError as e:
                    log_fn(f"Could not remove leftover temp entry {entry}: {e}", "warning")
            if removed:
                log_fn(f"Cleared {removed} leftover _temp/ entr{'y' if removed == 1 else 'ies'}.", "info")

    def clear_cache_now(self):
        """Standalone action (the OPTIONS card's "Clear Cache Now" button)
        -- wipes the disk cache and any leftover _temp/ scratch folders
        immediately, independent of running a conversion. Runs off the
        main thread since the cache can grow into the tens of GB (see
        cache_utils.py's own module docstring) and a big delete shouldn't
        freeze the window."""
        if not messagebox.askyesno(
                "Clear Cache",
                f"Delete the entire disk cache at:\n{cache_utils.cache_root()}\n\n"
                f"and any leftover temp scratch folders? This can't be undone, but "
                f"everything in it is fully reproducible by re-running a conversion."):
            return
        threading.Thread(target=self._wipe_cache_and_temp, args=(self.log,), daemon=True).start()

    def run_pipeline(self):
        pkg = Path(self.pkg_var.get())
        out = Path(self.out_var.get())

        if self.clean_run_var.get():
            self.log("Clean run requested -- wiping disk cache and leftover temp files first...", "info")
            self._wipe_cache_and_temp(self.log)

        _LOCAL_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="py-msfs-", dir=str(_LOCAL_TEMP_ROOT)))

        obj_dir = out / "objects"
        tex_dir = out / "textures"

        obj_dir.mkdir(parents=True, exist_ok=True)
        tex_dir.mkdir(parents=True, exist_ok=True)

        # Read once up front (not re-read per placement/tile below): whether
        # this run converts draped ground content to real .pol DSF polygons
        # instead of OBJ8 .obj files -- gates both terrain_fit (a real
        # DRAPED_POLYGON has no elevation field of its own, so per-vertex
        # terrain warping draped content is skipped entirely in this mode --
        # see terrain_fit.get_or_create_fitted_group's own docstring) and
        # draped_merge (which content actually becomes polygons).
        use_pol_polygons = self.pol_polygons_var.get()

        all_placements = []
        guid_map = {}
        all_exclusions = []
        airport_ref_lat = None
        airport_ref_lon = None
        msfs_install_root = None

        self.log(gpu_accel.describe(), "info")
        self.log(f"Disk cache: {cache_utils.cache_root()}", "info")

        try:
            # --- STEP 1: BGL Extraction ---
            self.log(f"\n{'='*50}\n1. EXTRACTING BGL FILES\n{'='*50}", "header")
            self.set_indeterminate(self.step1_bar)

            try:
                msfs_install_root = self.msfs_var.get().strip() or None
                propdefs_dir = self.propdefs_var.get().strip() or None
                placements, guids, exclusions, airport_ref_lat, airport_ref_lon, native_airport_layout = bgl_extractor.extract(
                    pkg, temp_dir, log_callback=self.log, msfs_install_root=msfs_install_root,
                    scan_terrain_vectors=self.scan_terrain_vectors_var.get(), propdefs_dir=propdefs_dir)
                all_placements.extend(placements)
                guid_map.update(guids)
                all_exclusions.extend(exclusions)
            except Exception as e:
                self.log(f"Fatal error parsing BGL directory: {e}", "error")

            # Look up the matching real-world airport's complete apt.dat
            # block once, right after we have the BGL's airport reference
            # point -- it's reused twice: to carve out an exclusion
            # rectangle over the airport's own boundary (needed before DSF
            # compilation, below) and to actually write apt.dat in Step 5,
            # so the ~300MB default database only gets scanned once.
            # This match also gates the airport-wide exclusion rectangle
            # below (Step 4) -- every precondition here fails silently by
            # design elsewhere in the codebase (returns None, no exception),
            # which previously meant "exclusion isn't covering the airport"
            # had no visible explanation anywhere in the log. Each branch
            # below now logs exactly which precondition didn't hold.
            matched_apt_block = None
            matched_apt_ident = None
            # MSFS2XP_XPLANE_ROOT lets a headless/CI run (or any run whose
            # output folder isn't literally inside the target X-Plane's own
            # "Custom Scenery") point at the real install so Step 5's apt.dat
            # reuse and the airport-boundary exclusion still work. Falls back
            # to walking up from the output path as before.
            xplane_root_override = os.environ.get("MSFS2XP_XPLANE_ROOT", "").strip()
            if xplane_root_override and Path(xplane_root_override).is_dir():
                xplane_root = Path(xplane_root_override)
                self.log(f"Using X-Plane root from MSFS2XP_XPLANE_ROOT: {xplane_root}", "info")
            else:
                if xplane_root_override:
                    self.log(f"MSFS2XP_XPLANE_ROOT is set to '{xplane_root_override}' but that isn't a "
                             f"directory -- falling back to the output-path search.", "warning")
                xplane_root = apt_dat.find_xplane_root(out)
            if xplane_root is None:
                self.log(
                    "No real-world airport match: the output folder isn't nested inside a "
                    "\"Custom Scenery\" directory, so the X-Plane install root (and its default "
                    "Global Airports apt.dat) couldn't be located. Set MSFS2XP_XPLANE_ROOT to the "
                    "X-Plane install path if it's elsewhere. The airport-boundary exclusion "
                    "and the real apt.dat reuse in Step 5 both depend on this and will be skipped.",
                    "warning")
            elif airport_ref_lat is None:
                self.log(
                    "No real-world airport match: no airport reference point was found while "
                    "parsing the BGL files (Step 1), so there's no coordinate to match against "
                    "the default Global Airports apt.dat. The airport-boundary exclusion and the "
                    "real apt.dat reuse in Step 5 both depend on this and will be skipped.",
                    "warning")
            else:
                global_apt_dat = apt_dat.find_global_airports_apt_dat(xplane_root)
                if global_apt_dat is None:
                    self.log(
                        f"No real-world airport match: no Global Airports apt.dat found under "
                        f"{xplane_root} (checked both the X-Plane 11 and 12 default locations). "
                        f"The airport-boundary exclusion and the real apt.dat reuse in Step 5 both "
                        f"depend on this and will be skipped.",
                        "warning")
                else:
                    result = apt_dat.find_nearest_airport_block(
                        global_apt_dat, airport_ref_lat, airport_ref_lon, log_callback=self.log)
                    if result is None:
                        self.log(
                            f"No real-world airport match: {global_apt_dat} had no parseable "
                            f"airport blocks at all. The airport-boundary exclusion and the real "
                            f"apt.dat reuse in Step 5 both depend on this and will be skipped.",
                            "warning")
                    elif result[2] > 15.0:
                        self.log(
                            f"No real-world airport match: the nearest airport in the default "
                            f"database ({result[1]}) is {result[2]:.1f}km from this package's BGL "
                            f"airport reference point -- too far to be the same airport (15km "
                            f"limit). The airport-boundary exclusion and the real apt.dat reuse in "
                            f"Step 5 both depend on this and will be skipped.",
                            "warning")
                    else:
                        matched_apt_block, matched_apt_ident, _ = result

            self.stop_indeterminate(self.step1_bar)
            self.update_progress(self.step1_bar, 100, 100)
            self.update_progress(self.overall_bar, 25, 100)

            # --- STEP 2: Textures ---
            self.log(f"\n{'='*50}\n2. EXTRACTING TEXTURES\n{'='*50}", "header")
            pkg_files = list(pkg.rglob("*"))
            
            valid_exts = {'.png', '.dds', '.jpg', '.jpeg', '.tga'}
            standard_textures = [p for p in pkg_files if p.is_file() and p.suffix.lower() in valid_exts and p.suffix.lower() != '.ktx2']
            
            copied_count = 0
            for p in standard_textures:
                raw_name = p.name.lower()
                actual_ext = p.suffix.lower()
                
                clean_base = raw_name
                for ext in ['.png.dds', '.dds', '.jpg', '.jpeg', '.tga', '.png']:
                    if clean_base.endswith(ext):
                        clean_base = clean_base[:-len(ext)]
                        break
                        
                target_name = clean_base + actual_ext
                target = tex_dir / target_name
                
                if not target.exists():
                    try:
                        shutil.copy2(p, target)
                        copied_count += 1
                    except Exception:
                        pass
            self.log(f"Copied {copied_count} standard textures into 'textures' folder.", "success")

            ktx2_files = [p for p in pkg_files if p.is_file() and p.suffix.lower() == '.ktx2']
            # Case-insensitive (see bgl_extractor.glob_ci's own docstring --
            # same real gap: files copied out of a SimObject's own source
            # folder (_copy_simobject_model) keep their original on-disk
            # extension case, and a plain rglob("*.ktx2") silently misses an
            # uppercase ".KTX2" one on any case-sensitive filesystem).
            ktx2_files += [p for p in bgl_extractor.glob_ci(temp_dir, ".ktx2") if p not in ktx2_files]
            total_ktx = len(ktx2_files)
            
            if total_ktx > 0:
                self.log(f"Decoding {total_ktx} KTX2 texture files into PNG...", "info")
                decoded_success = 0
                # GPU fork: ProcessPoolExecutor instead of threads -- true
                # parallelism across CPU cores instead of GIL-bound threads,
                # since KTX2 decode/decompress is heavy Python+C-extension work.
                with ProcessPoolExecutor(max_workers=pool_worker_count()) as pool:
                    futures = {
                        pool.submit(decode_or_repackage_ktx2, ktx, tex_dir): ktx
                        for ktx in ktx2_files
                    }
                    for i, future in enumerate(as_completed(futures), 1):
                        res = future.result()
                        if res is True:
                            decoded_success += 1
                        else:
                            self.log(f"Failed decoding {futures[future].name}: {res}", "warning")
                            
                        self.update_progress(self.step2_bar, i, total_ktx)
                        self.update_progress(self.overall_bar, 25 + (25 * (i / total_ktx)), 100)
                        
                        if i % max(1, (total_ktx // 5)) == 0 or i == total_ktx: 
                            self.log(f"Processed KTX2 batch {i}/{total_ktx}", "info")
                self.log(f"Successfully decoded {decoded_success}/{total_ktx} KTX2 files.", "success")
            else:
                self.update_progress(self.step2_bar, 100, 100)
                self.update_progress(self.overall_bar, 50, 100)

            # --- STEP 3: Mesh Conversion ---
            # Case-insensitive -- same real gap as bgl_extractor.glob_ci
            # (see its docstring): a SimObject-sourced model copied by
            # _copy_simobject_model keeps whatever extension case the
            # source package shipped it with.
            model_files = bgl_extractor.glob_ci(temp_dir, ".glb") + bgl_extractor.glob_ci(temp_dir, ".gltf")
            total_models = len(model_files)
            self.log(f"\n--- 3. Converting {total_models} 3D Meshes (Multi-process) ---")
            offsets = {}
            converted_stems_map = {}

            if total_models > 0:
                completed_models = 0
                failed_models = 0

                # extract_image's own texture search list (mesh_convert/
                # convert.py): the source package root FIRST (a texture can
                # be genuinely shipped but unreachable via the model's own
                # relative URI or the normal sibling-folder search, e.g. a
                # stale dev-machine path), THEN the real MSFS install root
                # when configured, so a package's own bundled texture always
                # wins over a same-named base-game one, with base-game as
                # fallback when the package doesn't ship it.
                external_textures_root = [pkg] + ([Path(msfs_install_root)] if msfs_install_root else [])

                # GPU fork: ProcessPoolExecutor instead of threads -- escapes
                # the GIL for the non-numpy portions of conversion (GLTF/JSON
                # parsing, struct unpacking, string formatting).
                with ProcessPoolExecutor(max_workers=pool_worker_count()) as executor:
                    futures = {
                        executor.submit(
                            cached_convert,
                            m, obj_dir, tex_dir, external_textures_root, "0.0", "180.0", "0.0",
                            self.disable_cache_var.get(), self.static_doors_var.get()
                        ): m.stem for m in model_files
                    }
                    
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            result_paths = future.result(timeout=60.0)
                            if result_paths:
                                completed_models += 1
                                converted_stems_map[name] = [p.stem for p in result_paths]
                                # convert() re-centers this model around its own XZ
                                # footprint (and re-zeroes its lowest Y point to 0,
                                # whichever direction it started on) and records the
                                # removed offset in a sidecar
                                # next to the .obj files it wrote -- read it back
                                # so the placement loop below (the "offsets"
                                # lookup) can compensate, keeping the object's
                                # final rendered position unchanged despite the
                                # re-centered local geometry.
                                offset_sidecar = obj_dir / f"{name}.originoffset.json"
                                if offset_sidecar.exists():
                                    try:
                                        data = json.loads(offset_sidecar.read_text(encoding="utf-8"))
                                        offsets[name.lower()] = (data["x"], data["y"], data["z"])
                                    except (OSError, ValueError, KeyError):
                                        pass
                                self.log(f"Successfully converted model [{completed_models}/{total_models}]: {name} (split into {len(result_paths)} parts)")
                            else:
                                failed_models += 1
                                self.log(f"Failed converting model {name}: Returned None", "warning")
                        except TimeoutError:
                            failed_models += 1
                            self.log(f"[TIMEOUT WARNING] Conversion for model '{name}' exceeded 60s limit and was skipped.", "warning")
                        except Exception as e:
                            failed_models += 1
                            self.log(f"Error converting model '{name}': {e}", "error")
                            
                        self.update_progress(self.step3_bar, completed_models + failed_models, total_models)
                        self.update_progress(self.overall_bar, 50 + (25 * ((completed_models + failed_models) / total_models)), 100)

            self.log(f"Mesh conversion finished. Successfully converted {completed_models}/{total_models} models.")

            # Proximity-triggered animations (MSFS's Z:VisibleRadiusBox
            # pattern -- doors/barriers/gates that open when the aircraft
            # gets close) have no equivalent stock X-Plane dataref, so
            # mesh_convert keys their ANIM_ blocks to a custom
            # "msfs2xp/proximity/<model>" dataref instead and drops a small
            # "<obj_stem>.proximity.json" sidecar next to each animated
            # .obj it writes. This converter has no idea where any of those
            # objects actually get PLACED (that's decided below, per-tile,
            # from all_placements) -- so cross-referencing happens here:
            # every matched placement whose generated .obj has a sidecar
            # gets recorded into a manifest the companion FlyWithLua plugin
            # reads at runtime to know which datarefs to drive from where.
            proximity_sidecars = {}
            for sidecar in obj_dir.glob("*.proximity.json"):
                try:
                    proximity_sidecars[Path(sidecar.stem).stem] = json.loads(sidecar.read_text(encoding="utf-8"))["dataref"]
                except (OSError, ValueError, KeyError):
                    pass
            proximity_objects = []

            # Same sidecar pattern, for draped ground-marking footprint
            # area (see mesh_convert.convert.compute_file_flatness_and_reference's
            # footprint_area docstring) -- used below to sort each DSF
            # tile's objects so smaller/more specific marking layers (text,
            # painted lines) draw after, and so land visually on top of,
            # broader ones (a background color fill, base asphalt).
            footprint_sidecars = {}
            for sidecar in obj_dir.glob("*.footprint.json"):
                try:
                    footprint_sidecars[Path(sidecar.stem).stem] = json.loads(sidecar.read_text(encoding="utf-8"))["area_m2"]
                except (OSError, ValueError, KeyError):
                    pass

            # Which .obj parts are DRAPED (carry ATTR_draped). Needed below to
            # force agl=0 on them: a draped part conforms to terrain per-
            # vertex, so a vertical placement offset is meaningless AND routes
            # it through the DSF 4-plane AGL pool, which X-Plane reads as
            # "explicit height, NOT terrain-draped" -- silently un-draping the
            # markings. .footprint.json alone is NOT this signal (convert only
            # writes it for a draped builder that also accrued per-block
            # footprint areas), so scan the written .obj text directly.
            draped_stems = set()
            for objf in obj_dir.glob("*.obj"):
                try:
                    if "\nATTR_draped\n" in objf.read_text(encoding="utf-8", errors="replace"):
                        draped_stems.add(objf.stem)
                except OSError:
                    pass

            # --- STEP 4: DSF Compilation ---
            self.log(f"\n{'='*50}\n4. COMPILING DSF TILES\n{'='*50}", "header")

            # Clear last run's terrain-fit copies. terrain_fit names each
            # *_tfit_<digest>.obj (own-group fit) or *_tfitlink_<digest>.obj
            # (retroactive shared-rotation link) purely from the placement's
            # lat/lon/hdg, NOT the source .obj's content, and skips the
            # write when a file of that name already exists -- so after the
            # source .obj changes (a new light format, a geometry fix) but
            # the placement doesn't move, the DSF keeps pulling in the STALE
            # copy from a previous run (confirmed: every one of 1217
            # *_tfit_* files survived a full re-convert, silently undoing
            # the LIGHT_SPILL / wig-wag / down-light fixes for every
            # terrain-fitted placement). Also clears the orphans left when
            # a placement DOES move. terrain_fit regenerates what this run
            # needs; its in-process group cache still de-dupes the work.
            #
            # CONFIRMED REAL BUG: this used to glob "*_tfit_*" only, which
            # does NOT match "*_tfitlink_*" (no "_tfit_" substring inside
            # "_tfitlink_" -- "link" follows "tfit" with no underscore) --
            # found a real *_tfitlink_* file a full day stale sitting next
            # to today's fresh geometry for the same object on a real EGLC
            # run. Widened to "*_tfit*" so both variants are always cleared.
            _tfit_cleared = 0
            for _tf in list(obj_dir.glob("*_tfit*")):
                try:
                    _tf.unlink()
                    if _tf.suffix == ".obj":
                        _tfit_cleared += 1
                except OSError:
                    pass
            if _tfit_cleared:
                self.log(f"Cleared {_tfit_cleared} terrain-fit copy(ies) from a previous run "
                         f"so this run's source-.obj fixes actually reach the DSF.", "info")

            dsf_tiles = {}
            matched_count = 0
            unmatched_guid_samples = set()
            # Every placement that matched NO converted geometry and NO
            # substitution -- collected by (guid, normalized title) so the
            # companion pick_replacements.py browser can list each distinct
            # missing object once with a real count. Written to
            # unresolved_objects.json in the source package dir at the end
            # of this step.
            unresolved_objects = {}
            unmatched_placements = []  # the actual placement dicts, for re-resolution after the mid-run prompt
            # User's saved "missing object -> X-Plane library path" picks
            # for this package (pick_replacements.py writes them). Checked
            # before the built-in keyword table, and a "SKIP" pick stops
            # an object being re-listed as unresolved.
            object_replacements = load_object_replacements(pkg, _SCRIPT_DIR)
            if object_replacements:
                self.log(f"Loaded {len(object_replacements)} saved object-replacement pick(s) from "
                         f"object_replacements.json.", "info")
            # GUID -> friendly name, so an unresolved placement with a blank
            # title still shows as e.g. "SHS_Clutter_ApronLight_001" in the
            # picker instead of a hex string.
            simprop_names = load_simprop_names(pkg)
            agl_placement_count = 0
            terrain_fit_applied_count = 0
            terrain_fit_unavailable_warned = False
            terrain_fit_reason_counts = {}
            # Proxy threshold for "this placement is a complex, likely
            # building-scale model" -- a model this heavily split by
            # material is never a small prop. Capped so one airport with
            # thousands of tiny multi-material signs can't flood the log.
            _COMPLEX_MODEL_STEM_THRESHOLD = 10
            _COMPLEX_MODEL_LOG_LIMIT = 40
            complex_model_log_count = 0

            library_substitution_count = 0

            # GUID-miss fallback: resolve a placement to a converted model
            # by its TITLE when guid_map doesn't. Used by BOTH the
            # terrain-fit pre-warm and the placement loop so a title-
            # resolved object gets the same treatment a guid-resolved one
            # would (pre-warmed fit, offsets, AGL) instead of silently
            # disappearing.
            title_stem_index = build_title_stem_index(converted_stems_map)

            def _resolve_original_stem(pl):
                st = guid_map.get(pl["guid"])
                if st and st in converted_stems_map:
                    return st
                tk = _model_stem_basename(pl.get("title"))
                if tk:
                    cand = title_stem_index.get(tk)
                    if cand:
                        return cand
                return st  # None or an unconverted stem -> falls through to the picker

            # --- terrain-fit pre-warm (multi-process) --------------------
            # terrain_fit.get_or_create_fitted_group() is the expensive part
            # of the placement loop below: pickle-loading each sub-object's
            # MeshIR, sampling the DEM, warping the mesh, writing a
            # *_tfit_*.obj copy. Now that py7zr lets it actually run here it
            # otherwise makes step 4 a single-threaded slog over thousands
            # of placements (~33 min at LHBP). That work is CPU/GIL-bound,
            # so this runs it across a PROCESS pool (a thread pool measured
            # ~1 core) -- same pattern as the mesh + KTX2 phases above.
            # Jobs are de-duplicated by the exact group key terrain_fit
            # itself uses, so no two workers write the same *_tfit_* path.
            # Each worker writes its *_tfit_* files to the shared obj_dir
            # and hands back the small result dict; the placement loop then
            # reads precomputed_fits and never recomputes (the parent
            # process's own terrain_fit._group_cache is not shared with the
            # workers, so consuming the returned dict is what avoids the
            # duplicate work).
            precomputed_fits = {}
            if xplane_root is not None:
                _seen_gk = set()
                _tf_jobs = []
                for _p in all_placements:
                    _os = _resolve_original_stem(_p)
                    if not (_os and _os in converted_stems_map):
                        continue
                    _mx, _my, _mz = offsets.get(_os.lower(), (0.0, 0.0, 0.0))
                    _alat, _alon = geo_transform.local_offset_to_latlon(
                        _p["lat"], _p["lon"], _p["hdg"], _mx, _mz)
                    _stems = converted_stems_map[_os]
                    _gk = (tuple(sorted(_stems)), round(_alat, 6), round(_alon, 6),
                           round(_p["hdg"], 2), bool(use_pol_polygons))
                    if _gk in _seen_gk:
                        continue
                    _seen_gk.add(_gk)
                    _tf_jobs.append((_stems, _alat, _alon, _p["hdg"]))
                if _tf_jobs:
                    _tf_workers = pool_worker_count()
                    self.log(f"Terrain-fit pre-warm: {len(_tf_jobs)} unique placement group(s) "
                             f"across {_tf_workers} process(es)...", "info")
                    _obj_dir_s, _xp_s, _skip_d = str(obj_dir), str(xplane_root), bool(use_pol_polygons)
                    try:
                        with ProcessPoolExecutor(max_workers=_tf_workers) as _ex:
                            _futs = [
                                _ex.submit(_terrain_fit_group_worker, _st, _la, _lo, _hd,
                                           _skip_d, _obj_dir_s, _xp_s)
                                for (_st, _la, _lo, _hd) in _tf_jobs
                            ]
                            for _f in as_completed(_futs):
                                _k, _r, _t = _f.result()
                                if _r is not None:
                                    precomputed_fits[_k] = _r
                                    # Re-seed the PARENT process's own
                                    # transform cache from the worker's
                                    # result -- see _terrain_fit_group_
                                    # worker's docstring for why this is
                                    # required for the anchor-clustering
                                    # pass below to find anything at all.
                                    if _t is not None:
                                        terrain_fit._group_transform_cache[_k] = _t
                        self.log(f"Terrain-fit pre-warm complete: {len(precomputed_fits)}/{len(_tf_jobs)} "
                                 f"group(s) computed -- placement loop will read these.", "info")
                    except Exception as e:
                        self.log(f"Terrain-fit pre-warm pool failed ({e}) -- the placement loop will "
                                 f"compute each fit inline instead.", "warning")
                        precomputed_fits = {}

            # Anchor-clustering for terrain-fit: one real-world building
            # instance can be split across multiple placements from
            # different source paths (e.g. an SPB-attached exterior shell
            # and a plain-BGL-placed interior at the same real-world
            # anchor) that never share a model stem, so they never reach
            # the same terrain_fit group_key. Collected per-placement
            # below; resolved into cross-placement corrections in one pass
            # after this loop, once every placement's own fit is known.
            _anchor_cluster_candidates = []

            # Per-object exclusion-zone footprints: collected here (real
            # geometry + real placement anchor), resolved into rectangles
            # in one pass after this loop -- see _per_object_exclusion_rects.
            _footprint_exclusion_candidates = []

            for p in all_placements:
                original_stem = _resolve_original_stem(p)
                offset = offsets.get(original_stem.lower(), (0.0, 0.0, 0.0)) if original_stem else None

                if original_stem and original_stem in converted_stems_map:
                    matched_count += 1
                    mid_x, mid_y, mid_z = offset

                    # convert() re-centers this model's own geometry around
                    # its XZ footprint center (mid_x/mid_z) and re-zeroes
                    # its lowest Y point to 0 (mid_y, folded into agl
                    # below) -- both need to be undone here at placement
                    # time so the final rendered position is unchanged:
                    # the placement anchor moves by (mid_x, mid_z), rotated
                    # into real-world lat/lon by the object's heading (see
                    # geo_transform.local_offset_to_latlon).
                    abs_lat, abs_lon = geo_transform.local_offset_to_latlon(
                        p["lat"], p["lon"], p["hdg"], mid_x, mid_z
                    )
                    tile_lat, tile_lon = math.floor(abs_lat), math.floor(abs_lon)

                    # HEIGHT/Y-OFFSET FIX: SPB-attached objects (jetways, GSE,
                    # towers, etc.) and BGL placements with a real AGL altitude
                    # carry a genuine vertical offset relative to their DSF
                    # ground-contact point, computed by bgl_extractor as
                    # "height_offset". DSF's OBJECT command has no vertical
                    # field of its own, but X-Plane supports explicit-height
                    # placement via a second, 4-plane (lon/lat/heading/
                    # elevation) point pool in AGL mode -- dsf_compiler.
                    # build_dsf routes any object with a nonzero "agl" key
                    # through that pool instead of the ordinary terrain-
                    # draped one. mid_y is added in here too, since the AGL
                    # pool is always terrain-relative, never MSL.
                    agl = p.get("height_offset", 0.0) + mid_y
                    if abs(agl) >= 0.01:
                        agl_placement_count += 1

                    generated_stems = converted_stems_map[original_stem]
                    _footprint_exclusion_candidates.append((generated_stems, abs_lat, abs_lon, p["hdg"]))

                    # Large-building terrain fit: a rigid mesh assumes flat
                    # ground under its whole footprint, but real X-Plane
                    # terrain often isn't (see terrain_fit.py). Computed ONCE
                    # for the whole group of sibling .obj files one model+
                    # placement produced, sharing one footprint bbox and one
                    # sampled correction grid, so they move together as a
                    # single rigid body instead of each sub-object drifting
                    # independently. Only ever swaps in a distinct, per-
                    # placement-corrected copy of each .obj -- never mutates
                    # the shared originals other placements may still use.
                    #
                    # Also applied to AGL-mounted placements: the correction
                    # is already anchor-relative (each vertex's delta is
                    # real-elevation-at-that-vertex minus real-elevation-at-
                    # the-anchor, see _point_elevation_delta/origin_elev), so
                    # it accounts for terrain variation ACROSS the footprint
                    # without double-counting the AGL offset itself, which
                    # only sets the anchor's own height.
                    # Prefer the result the pre-warm pool already computed
                    # for this group; only compute inline on a miss.
                    _fit_gk = (tuple(sorted(generated_stems)), round(abs_lat, 6), round(abs_lon, 6),
                               round(p["hdg"], 2), bool(use_pol_polygons))
                    fit_results = precomputed_fits.get(_fit_gk)
                    if fit_results is None:
                        fit_results = terrain_fit.get_or_create_fitted_group(
                            obj_dir, generated_stems, abs_lat, abs_lon, p["hdg"], xplane_root,
                            skip_draped_positions=use_pol_polygons)

                    warned_unavailable_this_group = False
                    group_any_applied = False
                    group_reasons = set()
                    _stem_entries = {}
                    for obj_stem in generated_stems:
                        placed_stem, fit_applied, fit_reason = fit_results[obj_stem]
                        terrain_fit_reason_counts[fit_reason] = terrain_fit_reason_counts.get(fit_reason, 0) + 1
                        if fit_applied:
                            terrain_fit_applied_count += 1
                            group_any_applied = True
                        else:
                            group_reasons.add(fit_reason)
                            if fit_reason == "terrain_unavailable" and not terrain_fit_unavailable_warned and not warned_unavailable_this_group:
                                warned_unavailable_this_group = True
                                terrain_fit_unavailable_warned = True
                                self.log(
                                    "Large-building terrain fit: real X-Plane terrain data isn't "
                                    "available for this airport (no X-Plane install found, no default "
                                    "elevation tile for this area, or py7zr isn't installed -- "
                                    "'pip install py7zr' to enable it). Large buildings will keep their "
                                    "originally converted, unwarped geometry.", "warning")

                        # A DRAPED part conforms to real terrain per-vertex, so
                        # a vertical placement offset on it is meaningless -- and
                        # routing it through the DSF's 4-plane AGL pool makes
                        # X-Plane read it as "explicit height, NOT terrain-
                        # draped", silently un-draping it. Any model whose
                        # recenter produced a Y lift (a footing a few cm below
                        # origin is enough) or that carries an SPB height_offset
                        # then had ALL its parts -- ground markings included --
                        # go AGL: confirmed as ~2/3 of every placement landing in
                        # the AGL pool and stand-number text floating/z-fighting
                        # away. draped_stems (built by scanning each .obj for
                        # ATTR_draped) is the reliable draped test -- placed_stem
                        # is what a terrain-fit swap may have renamed it to,
                        # obj_stem is convert()'s original.
                        footprint_area = footprint_sidecars.get(obj_stem)
                        part_is_draped = obj_stem in draped_stems or placed_stem in draped_stems
                        entry = {
                            "name": placed_stem, "lat": abs_lat, "lon": abs_lon, "hdg": p["hdg"],
                            "agl": 0.0 if part_is_draped else agl,
                        }
                        if footprint_area is not None:
                            entry["footprint_area"] = footprint_area
                        dsf_tiles.setdefault((tile_lat, tile_lon), []).append(entry)
                        dataref = proximity_sidecars.get(obj_stem)
                        if dataref:
                            proximity_objects.append({"dataref": dataref, "lat": abs_lat, "lon": abs_lon})
                        _stem_entries[obj_stem] = (entry, fit_applied, part_is_draped)

                    _anchor_cluster_candidates.append({
                        "group_key": _fit_gk, "raw_lat": p["lat"], "raw_lon": p["lon"], "hdg": p["hdg"],
                        "agl": agl, "mid_x": mid_x, "mid_z": mid_z,
                        "any_applied": group_any_applied, "stem_entries": _stem_entries,
                    })

                    # A model split into many sub-object files (walls/roof/
                    # windows/trim/...) is the closest cheap proxy available
                    # here for "this is a large, complex building" without
                    # re-deriving terrain_fit's own footprint math a second
                    # time. Named per-model logging for exactly these,
                    # capped, is what makes "is my terminal actually getting
                    # corrected, and if not why" answerable from the log
                    # directly instead of needing pixel-level forensics --
                    # confirmed necessary: main.py previously only ever
                    # logged ONE aggregate "N placements corrected" number,
                    # with zero way to tell a specific building's own
                    # outcome, or that TILTED was excluding it (see
                    # terrain_fit.py's own docstring for why that mattered).
                    if len(generated_stems) >= _COMPLEX_MODEL_STEM_THRESHOLD and not group_any_applied and complex_model_log_count < _COMPLEX_MODEL_LOG_LIMIT:
                        complex_model_log_count += 1
                        self.log(
                            f"Large-building terrain fit: '{original_stem}' ({len(generated_stems)} "
                            f"sub-objects) was NOT corrected -- {sorted(group_reasons)}", "info")
                else:
                    lib_path = resolve_library_substitution(
                        p.get("title"), guid=p.get("guid"),
                        approximate=self.approximate_substitution_var.get(),
                        replacements=object_replacements)
                    if lib_path == SKIP_SUBSTITUTION:
                        # user deliberately chose to place nothing for this
                        # object -- count it as resolved, don't re-list it
                        matched_count += 1
                    elif lib_path:
                        matched_count += 1
                        library_substitution_count += 1
                        tile_lat, tile_lon = math.floor(p["lat"]), math.floor(p["lon"])
                        dsf_tiles.setdefault((tile_lat, tile_lon), []).append({
                            "name": None, "library_path": lib_path,
                            "lat": p["lat"], "lon": p["lon"], "hdg": p["hdg"], "agl": 0.0
                        })
                    else:
                        if p["guid"]:
                            unmatched_guid_samples.add(p["guid"])
                        _gkey = str(p.get("guid") or "").strip().strip("{}").lower()
                        _key = _gkey or (normalize_placement_title(p.get("title")) or "?")
                        _nm = (p.get("title") or simprop_names.get(_gkey) or "").strip()
                        rec = unresolved_objects.setdefault(_key, {
                            "title": _nm, "guid": p.get("guid") or "", "count": 0, "positions": []})
                        if _nm and not rec["title"]:
                            rec["title"] = _nm
                        rec["count"] += 1
                        if p.get("lat") is not None and p.get("lon") is not None and len(rec["positions"]) < 400:
                            rec["positions"].append([round(float(p["lat"]), 7), round(float(p["lon"]), 7)])
                        unmatched_placements.append(p)

            # Resolve anchor clusters: for every bucket of 2+ placements
            # sharing a real-world anchor (position + heading, rounded to
            # ~1m/0.1 degree -- tight on purpose, this only needs to catch
            # "the same real-world instance decoded via two different
            # paths", not "buildings near each other") but DIFFERENT model
            # stems (so they never shared a terrain_fit group_key),
            # propagate whichever member's vertical shift actually got
            # applied onto every other member that came back disqualified/
            # rigid_skip/negligible on its own. See terrain_fit.
            # apply_shared_shift_to_group's own docstring for why sharing
            # one anchor point makes reusing the identical shift correct
            # regardless of each object's own local-frame convention.
            #
            # Bucket key is the RAW placement anchor (p["lat"]/p["lon"]),
            # not each candidate's own recenter-compensated abs_lat/
            # abs_lon: two placements from the same BGL/SPB record share
            # the raw anchor exactly, but each is an independent source
            # model with its own mesh_convert recenter offset, so their
            # compensated positions differ by design and won't reliably
            # land in the same rounded bucket.
            _anchor_buckets = {}
            for cand in _anchor_cluster_candidates:
                key = (round(cand["raw_lat"], 5), round(cand["raw_lon"], 5), round(cand["hdg"], 1))
                _anchor_buckets.setdefault(key, []).append(cand)

            _linked_count = 0
            _multi_group_buckets = [b for b in _anchor_buckets.values() if len({c["group_key"] for c in b}) >= 2]
            if _multi_group_buckets:
                _with_ref = sum(1 for b in _multi_group_buckets if any(c["any_applied"] for c in b))
                self.log(f"Anchor clustering: {len(_multi_group_buckets)} real-world anchor(s) are shared by "
                         f"2+ differently-named placements ({_with_ref} of them have at least one member with "
                         f"an applied terrain-fit shift to share with the others).", "info")
            for bucket in _anchor_buckets.values():
                if len({c["group_key"] for c in bucket}) < 2:
                    continue
                applied_cand = next((c for c in bucket if c["any_applied"]), None)
                if applied_cand is None:
                    continue
                transform = terrain_fit.get_cached_transform(applied_cand["group_key"])
                if transform is None or transform.get("vertical_shift") is None:
                    continue

                for cand in bucket:
                    if cand["any_applied"] or cand["group_key"] == applied_cand["group_key"]:
                        continue
                    unresolved_stems = [
                        stem for stem, (entry, fit_applied, part_is_draped) in cand["stem_entries"].items()
                        if not fit_applied and not part_is_draped
                    ]
                    if not unresolved_stems:
                        continue
                    # No anchor-delta bookkeeping needed here (unlike the
                    # rotation this replaced): the shift is a property of
                    # the shared real-world anchor point, not of either
                    # candidate's own local-frame convention -- see
                    # apply_shared_shift_to_group's own docstring.
                    linked_results = terrain_fit.apply_shared_shift_to_group(
                        obj_dir, unresolved_stems, transform, xplane_root)
                    for stem in unresolved_stems:
                        new_name, linked_applied, _linked_reason = linked_results[stem]
                        if linked_applied:
                            entry, _, _ = cand["stem_entries"][stem]
                            entry["name"] = new_name
                            _linked_count += 1

            if _linked_count:
                self.log(f"{_linked_count} placement sub-object(s) linked to a sibling placement's terrain-fit "
                         f"shift at the same real-world anchor.", "info")

            if agl_placement_count:
                self.log(f"{agl_placement_count} placement(s) use native DSF AGL height placement to fix floating/sunken SPB-attached or upper-floor objects.", "info")
            if terrain_fit_applied_count:
                self.log(f"{terrain_fit_applied_count} large-building placement(s) warped against real sampled X-Plane terrain to fix floating/sunken corners.", "info")
            if terrain_fit_reason_counts:
                self.log(f"Large-building terrain fit outcomes (by sub-object): {dict(sorted(terrain_fit_reason_counts.items()))}", "info")
            if complex_model_log_count >= _COMPLEX_MODEL_LOG_LIMIT:
                self.log(f"(complex-model terrain-fit logging capped at {_COMPLEX_MODEL_LOG_LIMIT} -- more were skipped)", "info")

            self.log(f"Successfully mapped placement-to-model matches: {matched_count}/{len(all_placements)}", "success")
            if library_substitution_count:
                self.log(f"{library_substitution_count} placement(s) substituted with X-Plane's own default "
                         f"library objects (generic floodlight/apron light fixtures) instead of converted "
                         f"geometry, for native day/night lighting behavior.", "info")
            if unmatched_guid_samples:
                self.log(f"Warning: {len(unmatched_guid_samples)} unique GUIDs had placements but no matching extracted model.", "warning")

            # Write the list of still-unresolved objects to the source
            # package dir so the companion pick_replacements.py browser can
            # let the user map each one to an X-Plane library object (e.g.
            # ASOBO taxiway/runway lights, the PAPI -- assets that simply
            # aren't in the package) and have it applied on the next run.
            try:
                unresolved_path = pkg / "unresolved_objects.json"
                if unresolved_objects:
                    payload = {
                        "_comment": "Objects this conversion could not match to any converted geometry or "
                                    "substitution. Run pick_replacements.py to map them to X-Plane library "
                                    "paths; that writes object_replacements.json next to this file, which "
                                    "the converter reads on the next run. 'positions' are [lat,lon] samples "
                                    "(capped at 400) for the picker's map view; 'output' is the converted "
                                    "pack the map draws its pavement from.",
                        "package": str(pkg),
                        "output": str(out),
                        "objects": sorted(
                            ({"key": k, **v} for k, v in unresolved_objects.items()),
                            key=lambda r: (-r["count"], (r["title"] or r["key"]).lower())),
                    }
                    unresolved_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    self.log(f"{len(unresolved_objects)} distinct object(s) unresolved -- wrote "
                             f"{unresolved_path.name} (run pick_replacements.py to assign X-Plane "
                             f"replacements).", "info")
                elif unresolved_path.exists():
                    unresolved_path.unlink()
            except OSError as e:
                self.log(f"Could not write unresolved_objects.json: {e}", "warning")

            # Base layer for the picker's map view: the just-compiled DSF's
            # pavement/taxiway footprints in world coords + a thinned set
            # of every placed object's position, dumped so the picker can
            # draw a top-down airport map instantly without re-parsing the
            # DSF and 10k+ .obj files itself. Best-effort -- only when there
            # is something to pick, and never fatal.
            if unresolved_objects:
                try:
                    self.log("Building the picker's map base (reads the compiled DSF)...", "info")
                    _scene = scenery_viewer.build_scene(out)
                    # OBJ8 .obj has no triangle index in this reader, so the
                    # draped footprints are dumped as a thinned POINT CLOUD
                    # (a scatter of every Nth pavement vertex) -- enough to
                    # read the runway/apron/taxiway silhouette without the
                    # spanning-triangle garbage a raw vertex-order polygon
                    # would give.
                    _pav = []
                    _budget = 30000
                    for _d in _scene.get("draped", []):
                        _r = _d.get("ring") or []
                        if len(_r) < 3:
                            continue
                        _st = max(1, len(_r) // 400)
                        _pav.extend([round(a, 7), round(o, 7)] for a, o in _r[::_st])
                    if len(_pav) > _budget:
                        _st = len(_pav) // _budget + 1
                        _pav = _pav[::_st]
                    _ctx = [[round(r["lat"], 7), round(r["lon"], 7)]
                            for r in _scene.get("rigid", [])[:6000]]
                    _pool = (_pav + _ctx
                             + [p for v in unresolved_objects.values() for p in v["positions"]])
                    if len(_pool) >= 2:
                        _la = [p[0] for p in _pool]
                        _lo = [p[1] for p in _pool]
                        (pkg / "unresolved_map.json").write_text(json.dumps({
                            "bounds": [min(_la), max(_la), min(_lo), max(_lo)],
                            "pavement_pts": _pav, "context": _ctx,
                        }), encoding="utf-8")
                        self.log(f"Picker map base: {len(_pav)} pavement point(s) + "
                                 f"{len(_ctx)} object marker(s) -> unresolved_map.json", "info")
                except Exception as e:
                    self.log(f"(couldn't build the picker map base: {e} -- the picker will fall "
                             f"back to plotting positions only)", "info")
            else:
                try:
                    (pkg / "unresolved_map.json").unlink()
                except OSError:
                    pass

            # Mid-run replacement prompt: if anything is still unresolved
            # and we're interactive, pop up the picker NOW -- before the
            # DSF is compiled -- so the user's picks (an X-Plane library
            # object, or an explicit Skip) are applied to THIS conversion,
            # not just saved for next time. run_pipeline() runs on a worker
            # thread, so the Toplevel is built on the main thread via
            # root.after and this thread blocks on an Event until it closes.
            if unresolved_objects and self.prompt_replacements_var.get():
                try:
                    _prompt_done = threading.Event()

                    def _open_picker():
                        try:
                            pick_replacements.prompt_modal(
                                self.root, Path(pkg),
                                Path(xplane_root) if xplane_root else None)
                        except Exception as _e:
                            self.log(f"Replacement picker failed ({_e}) -- continuing without it.", "warning")
                        finally:
                            _prompt_done.set()

                    self.log(f"{len(unresolved_objects)} unresolved object(s) -- opening the replacement "
                             f"picker before DSF compile...", "info")
                    self.root.after(0, _open_picker)
                    if not _prompt_done.wait(timeout=3600):
                        self.log("Replacement picker still open after 60 min -- continuing without waiting.", "warning")

                    # Re-resolve the still-unmatched placements against the
                    # freshly-saved picks and place the ones that now hit.
                    object_replacements = load_object_replacements(pkg, _SCRIPT_DIR)
                    _readded = 0
                    _skipped = 0
                    for _p in unmatched_placements:
                        _lp = resolve_library_substitution(
                            _p.get("title"), guid=_p.get("guid"),
                            approximate=self.approximate_substitution_var.get(),
                            replacements=object_replacements)
                        if _lp == SKIP_SUBSTITUTION:
                            _skipped += 1
                        elif _lp:
                            _tl, _tn = math.floor(_p["lat"]), math.floor(_p["lon"])
                            dsf_tiles.setdefault((_tl, _tn), []).append({
                                "name": None, "library_path": _lp,
                                "lat": _p["lat"], "lon": _p["lon"], "hdg": _p["hdg"], "agl": 0.0,
                            })
                            _readded += 1
                    if _readded or _skipped:
                        matched_count += _readded + _skipped
                        library_substitution_count += _readded
                        self.log(f"Replacement picks applied to this run: {_readded} object(s) placed from "
                                 f"the X-Plane library, {_skipped} object(s) skipped on purpose.", "success")
                except Exception as e:
                    self.log(f"Mid-run replacement prompt skipped ({e}).", "warning")

            # Cross-object draped-layer merge: MSFS commonly represents one
            # continuous paved/painted surface as several SEPARATE placed
            # objects (not sub-materials of one model) sharing the same
            # texture -- mesh_convert's own per-file draw-order ranking
            # can't prevent two such objects from landing on the same
            # (layer_group, offset) if they came from different source
            # files, which is exactly the recurring z-fighting/floating-veil
            # symptom. Runs per tile, after terrain-fit (so it sees final,
            # possibly-corrected object stems): combines every draped object
            # sharing a texture within that tile into one welded mesh. See
            # draped_merge.py for what it does and doesn't cover.
            polygons_dir = obj_dir.parent / "polygons"
            tile_polygons = {}
            if dsf_tiles:
                objects_before = sum(len(objs) for objs in dsf_tiles.values())
                for (t_lat, t_lon), objects in dsf_tiles.items():
                    new_objects, polys = draped_merge.merge_draped_layers_in_tile(
                        obj_dir, t_lat, t_lon, objects, log_callback=self.log,
                        use_polygons=use_pol_polygons, polygons_dir=polygons_dir)
                    dsf_tiles[(t_lat, t_lon)] = new_objects
                    if polys:
                        tile_polygons[(t_lat, t_lon)] = polys
                objects_after = sum(len(objs) for objs in dsf_tiles.values())
                if objects_after < objects_before:
                    self.log(f"Cross-object draped-layer merge: {objects_before} placed objects "
                             f"reduced to {objects_after} after merging same-texture draped layers "
                             f"within each tile.", "info")
                if tile_polygons:
                    total_polys = sum(len(p) for p in tile_polygons.values())
                    self.log(f"Converted {total_polys} draped triangle(s) across {len(tile_polygons)} "
                             f"tile(s) to real .pol DSF polygons.", "info")

            # Airport-wide exclusion: suppress EVERY default category
            # (including roads/rail, "net") across the airport's real
            # extent -- this scenery's own converted buildings, ground
            # clutter, taxiway/apron pavement markings etc. are a complete
            # replacement for the default airport, and leaving any category
            # unexcluded left default surfaces visibly showing through
            # underneath the new scenery instead.
            #
            # Built from the UNION of two independent sources, not the
            # matched default apt.dat block's own "130 Airport Boundary"
            # ring alone: that ring is X-Plane's own (sometimes old/coarse,
            # occasionally smaller than the real, more detailed MSFS-
            # modeled airport) boundary polygon -- a real airport whose
            # MSFS scenery extends past that old boundary (a newer cargo
            # apron, an expanded ramp) got NO exclusion coverage at all
            # past that edge, letting default trees/autogen buildings/
            # pavement show through right at the airport despite
            # everything else in this feature working. This also means an
            # airport-wide exclusion now exists even when no default
            # apt.dat match was found at all (no match within 15km, no
            # X-Plane install configured, ...).
            #
            # Padded generously (300m/150m, not the ~50m that used to be
            # plenty for boundary-node rounding alone): this single
            # rectangle is now the ONLY exclusion this run emits (see
            # below), so it has to comfortably clear every placed object
            # on its own with margin to spare, not just hug the tightest
            # box that technically contains everything.
            placements_bbox = None
            if dsf_tiles:
                all_lats = [e["lat"] for objs in dsf_tiles.values() for e in objs]
                all_lons = [e["lon"] for objs in dsf_tiles.values() for e in objs]
                if all_lats:
                    placements_bbox = apt_dat.boundary_bounding_box(
                        list(zip(all_lats, all_lons)), pad_km=0.3)

            boundary_bbox = None
            boundary_points = []
            if matched_apt_block:
                boundary_points = apt_dat.extract_boundary_ring(matched_apt_block)
                boundary_bbox = apt_dat.boundary_bounding_box(boundary_points, pad_km=0.15)

            # Third, always-available candidate: a generous fixed-radius
            # square centered on the BGL's own airport reference point.
            # boundary_bbox and placements_bbox both have real, confirmed
            # gaps: boundary_bbox depends on finding a default apt.dat
            # match at all (and that match's own boundary ring can be
            # coarse/wrong for an unusual real airport), and placements_bbox
            # only ever covers whatever THIS run actually extracted --
            # which, for scenery whose pavement comes from a BGL vector/
            # polygon format this converter doesn't parse yet (confirmed
            # real case: iniBuilds' EGLC, which stores its entire runway/
            # taxiway/apron shape in a TerrainVectorDb section, not as
            # placed objects at all), silently excludes exactly the area
            # that most needs covering. A fixed radius around the
            # reference point doesn't depend on either of those succeeding
            # -- it's always available and always covers a real airport's
            # full extent by construction, at the cost of being less tight
            # than a shape-aware boundary for a small airport.
            reference_bbox = None
            if airport_ref_lat is not None and airport_ref_lon is not None:
                reference_bbox = apt_dat.boundary_bounding_box(
                    [(airport_ref_lat, airport_ref_lon)], pad_km=2.5)

            # Sanity-gate boundary_bbox/reference_bbox against
            # placements_bbox before unioning them (see apt_dat.
            # bbox_is_near) -- a mismatched-airport boundary/reference
            # candidate could otherwise make the exclusion rectangle
            # swallow far more than the actual converted area.
            # placements_bbox, built directly from every converted
            # object's own coordinates, has no equivalent failure mode
            # whenever there's real placement data, so it's the reference
            # every other candidate gets checked against here.
            _EXCLUSION_SANITY_KM = 15.0  # matches the existing default-apt.dat match radius (see the 15km gate above) -- even a very large hub airport's own true footprint tops out a few km across
            if placements_bbox is not None:
                if not apt_dat.bbox_is_near(boundary_bbox, placements_bbox, _EXCLUSION_SANITY_KM):
                    self.log(
                        f"Discarding the default-apt.dat boundary match exclusion candidate: it's "
                        f"centered more than {_EXCLUSION_SANITY_KM:.0f}km from the actual converted "
                        f"placements' own extent -- almost certainly a mismatched airport rather than "
                        f"this one's real location. Including it would balloon the single airport-wide "
                        f"exclusion rectangle far past the real airport instead of tightly covering it.",
                        "warning")
                    boundary_bbox = None
                if not apt_dat.bbox_is_near(reference_bbox, placements_bbox, _EXCLUSION_SANITY_KM):
                    self.log(
                        f"Discarding the BGL airport-reference-point radius exclusion candidate: it's "
                        f"centered more than {_EXCLUSION_SANITY_KM:.0f}km from the actual converted "
                        f"placements' own extent -- almost certainly a bad/stray coordinate rather than "
                        f"this airport's real location. Including it would balloon the single "
                        f"airport-wide exclusion rectangle far past the real airport instead of tightly "
                        f"covering it.",
                        "warning")
                    reference_bbox = None

            # CONFIRMED REAL BUG: unioning boundary_bbox/placements_bbox/
            # reference_bbox into ONE combined rectangle (their bounding
            # box's bounding box) reliably balloons past the real airport
            # -- a real boundary ring or placement extent is rarely a
            # clean rectangle, and reference_bbox alone is a blunt 5km-
            # wide square. Confirmed on a real EGLC conversion: this wiped
            # out X-Plane's default scenery over a huge area of unrelated
            # surrounding city, well past the airport itself.
            #
            # PREFERRED source, per explicit user instruction: one tight
            # rectangle PER converted object (_per_object_exclusion_rects),
            # not one shared shape covering the airport's whole combined
            # extent -- a real airport has genuine gaps between buildings
            # (grass, taxiways, empty apron) that an airport-wide shape
            # still swallowed whole, suppressing default scenery in places
            # nothing was ever placed. Only falls back to the OLD
            # airport-wide shape-aware union (rasterize the real boundary
            # ring's own interior via _polygon_interior_exclusion_rects,
            # plus the real placed-object extent via
            # _built_up_exclusion_rects, same rasterize+dilate+greedy-
            # rectangle-cover pattern used for roads/rail below) when the
            # per-object pass produced nothing at all (no sidecars
            # available for any placement). Roads/rail (net/str) are
            # deliberately NOT part of this -- those stay airport-wide,
            # built separately below.
            shaped_rects = _per_object_exclusion_rects(obj_dir, _footprint_exclusion_candidates)
            _used_per_object_rects = bool(shaped_rects)
            if not shaped_rects:
                if boundary_points:
                    shaped_rects.extend(_polygon_interior_exclusion_rects(boundary_points))
                if placements_bbox is not None and all_lats:
                    shaped_rects.extend(_built_up_exclusion_rects(all_lats, all_lons, cell_m=150.0, dilate=1))

            # Replace (not append to) the package's own small, per-BGL
            # "obj"-only Exclusion rectangles: those are real data (the
            # original MSFS scenery author's own fine-grained "don't draw
            # this one default tree here" cutouts), but rendering the
            # scenery as dozens of scattered small rectangles alongside
            # this run's own shape-aware ones is confusing to inspect and,
            # since the shape-aware set already excludes every category
            # (not just "obj") across the airport's whole extent, makes
            # every one of those small rectangles strictly redundant --
            # the shape-aware set already covers everything they cover
            # and more. Only fall back to the small per-feature ones when
            # nothing shape-aware could be built at all (no apt.dat match
            # AND no placements), so at least something suppresses
            # default objects.
            if shaped_rects or reference_bbox:
                superseded_count = len(all_exclusions)
                if shaped_rects:
                    # net/str deliberately excluded here even when shaped_rects
                    # fell back to the old airport-wide shape -- roads/rail
                    # always get their own, separately-built airport-wide
                    # rectangles below (road_rects), so including them here
                    # too would only be redundant, never wrong, but leaving
                    # it out keeps exactly one source of truth for road
                    # exclusion regardless of which shaped_rects branch ran.
                    _non_road_categories = tuple(k for k in dsf_compiler.EXCLUSION_PROP_KEYS.keys() if k not in ("net", "str"))
                    for r in shaped_rects:
                        r["categories"] = _non_road_categories
                    all_exclusions = shaped_rects
                else:
                    reference_bbox["categories"] = tuple(dsf_compiler.EXCLUSION_PROP_KEYS.keys())
                    all_exclusions = [reference_bbox]
                source_parts = []
                if _used_per_object_rects:
                    source_parts.append("each converted object's own footprint edges (per-object, +2m each)")
                else:
                    if boundary_points and shaped_rects:
                        source_parts.append(f"{matched_apt_ident}'s default boundary shape ({len(boundary_points)} point(s))")
                    if placements_bbox and shaped_rects:
                        source_parts.append("placed-object extent")
                if not shaped_rects and reference_bbox:
                    source_parts.append("a 2.5km fixed radius around the airport reference point")
                source_note = " + ".join(source_parts) if source_parts else "no source available"
                # Roads/rail (net, str) get their OWN set of exclusion
                # rectangles, shaped to where the converted content actually
                # is rather than one big radius: a ~200m grid over every
                # placed object's coordinate, dilated one cell so a road
                # running right along a building edge is still covered, then
                # greedy-rectangled. Default X-Plane car roads only get
                # suppressed where this scenery has something, so roads out
                # in the surrounding fields stay. Falls back to a single
                # 4.5km radius box if there aren't enough placements to
                # build a shape from.
                road_rects = []
                if dsf_tiles:
                    _rlats = [e["lat"] for objs in dsf_tiles.values() for e in objs]
                    _rlons = [e["lon"] for objs in dsf_tiles.values() for e in objs]
                    road_rects = _built_up_exclusion_rects(_rlats, _rlons, cell_m=200.0, dilate=1)
                if road_rects:
                    for r in road_rects:
                        r["categories"] = ("net", "str")
                        all_exclusions.append(r)
                    road_note = f"plus {len(road_rects)} sim/exclude_net rectangle(s) fitted to the converted-object extent for default car roads"
                elif airport_ref_lat is not None and airport_ref_lon is not None:
                    road_bbox = apt_dat.boundary_bounding_box(
                        [(airport_ref_lat, airport_ref_lon)], pad_km=4.5)
                    if road_bbox:
                        road_bbox["categories"] = ("net", "str")
                        all_exclusions.append(road_bbox)
                    road_note = "plus a wider 4.5km sim/exclude_net rectangle for default car roads (not enough placements to fit tighter)"
                else:
                    road_note = "no road exclusion (no placements, no reference point)"
                superseded_note = f" -- superseding {superseded_count} small per-BGL exclusion rectangle(s)" if superseded_count else ""
                self.log(f"Using {len(shaped_rects) or 1} shape-aware exclusion rectangle(s) covering "
                         f"{source_note}{superseded_note}, {road_note}.", "info")
            else:
                self.log(
                    "No airport-boundary exclusion could be built (no default apt.dat match, no "
                    "\"130 Airport Boundary\" row, and no placements) -- only per-BGL Exclusion "
                    "sections, if any, apply.", "warning")

            tile_exclusions = {}
            for ex in all_exclusions:
                tile_rect = {
                    "west": ex["west"], "south": ex["south"],
                    "east": ex["east"], "north": ex["north"],
                }
                if "categories" in ex:
                    tile_rect["categories"] = ex["categories"]
                # Every DSF tile the rectangle's bounding box actually
                # overlaps, not just whichever single tile its CENTER point
                # falls in -- a rectangle straddling a 1-degree tile
                # boundary (routine for anything but a small airport dead
                # center in its own tile; the whole point of THIS
                # particular rectangle is to cover the airport's real,
                # possibly-boundary-straddling footprint) used to get
                # silently dropped from every tile except the one holding
                # its midpoint, leaving default scenery showing through in
                # the rest -- exactly "exclusion doesn't cover the whole
                # airport". A rectangle edge landing exactly ON a tile
                # boundary (north/east == an integer degree) contributes
                # zero-width overlap into the tile beyond that edge, so
                # that edge tile is excluded from the range rather than
                # spuriously included.
                lat0, lat1 = math.floor(ex["south"]), math.floor(ex["north"])
                lon0, lon1 = math.floor(ex["west"]), math.floor(ex["east"])
                if lat1 == ex["north"] and lat1 > lat0:
                    lat1 -= 1
                if lon1 == ex["east"] and lon1 > lon0:
                    lon1 -= 1
                for t_lat in range(lat0, lat1 + 1):
                    for t_lon in range(lon0, lon1 + 1):
                        tile_exclusions.setdefault((t_lat, t_lon), []).append(tile_rect)
            if all_exclusions:
                per_feature_count = sum(1 for ex in all_exclusions if "categories" not in ex)
                note = (
                    f" ({per_feature_count} from the package's own BGL Exclusion sections, "
                    f"'obj' category only -- suppresses default 3-D objects but NOT default "
                    f"pavement/lines/polygons, since the BGL's own per-record category flags "
                    f"aren't decoded)" if per_feature_count else ""
                )
                self.log(f"Routed {len(all_exclusions)} exclusion rectangles into "
                         f"{len(tile_exclusions)} DSF tile(s).{note}", "info")
            else:
                self.log(
                    "No exclusion rectangles at all this run -- neither the package's own BGL "
                    "Exclusion sections nor the airport-boundary match (see above, if any) "
                    "produced any.",
                    "info")

            # Union with tile_exclusions' own keys, not just dsf_tiles': an
            # exclusion rectangle can (and, for anything but a small airport
            # dead-center in one tile, commonly does -- see above) reach a
            # neighboring tile that has no PLACED OBJECTS of its own at all,
            # e.g. a mostly-empty grass corner of the airport near a tile
            # boundary. Iterating dsf_tiles alone meant that tile's .dsf was
            # simply never written, silently dropping its exclusion (not
            # just misrouting it) -- an empty-objects DSF that carries only
            # the exclusion props is a real, valid tile X-Plane loads
            # correctly (dsf_compiler's pool/cmds encoding already handles
            # zero objects).
            all_tile_keys = set(dsf_tiles.keys()) | set(tile_exclusions.keys())
            if not all_tile_keys:
                self.log("[CRITICAL ERROR] No DSF tiles generated! Zero placements successfully matched to valid models.", "error")
                self.update_progress(self.step4_bar, 100, 100)
                self.update_progress(self.overall_bar, 100, 100)
            else:
                total_tiles = len(all_tile_keys)
                completed_tiles = 0
                for t_lat, t_lon in all_tile_keys:
                    objects = dsf_tiles.get((t_lat, t_lon), [])
                    folder_name = f"{math.floor(t_lat / 10) * 10:+03d}{math.floor(t_lon / 10) * 10:+04d}"
                    file_name = f"{t_lat:+03d}{t_lon:+04d}"

                    dsf_dest = out / "Earth nav data" / folder_name / f"{file_name}.dsf"
                    dsf_dest.parent.mkdir(parents=True, exist_ok=True)

                    # NOTE: real draw-order among overlapping draped/flat
                    # ground markings comes from each .obj's own
                    # ATTR_layer_group_draped offset (mesh_convert/
                    # draped_ranking.py's draped_layer_offset) -- X-Plane's OBJ8 spec is
                    # explicit that CMDS stream order gives NO ordering
                    # guarantee at all for draped geometry, so this sort is
                    # not the ordering mechanism, just a harmless,
                    # redundant-with-the-real-fix tie-breaker kept for
                    # cases where two layers happen to land in the exact
                    # same layer group AND offset.
                    objects = sorted(objects, key=lambda o: o.get("footprint_area", float("inf")), reverse=True)

                    tile_excl = tile_exclusions.get((t_lat, t_lon), [])
                    tile_polys = tile_polygons.get((t_lat, t_lon), [])
                    excl_note = f", {len(tile_excl)} exclusion rect(s)" if tile_excl else ""
                    poly_note = f", {len(tile_polys)} draped polygon(s)" if tile_polys else ""
                    self.log(f"Compiling DSF Tile {file_name} with {len(objects)} items{excl_note}{poly_note}...", "info")
                    try:
                        dsf_compiler.build_dsf(t_lat, t_lon, objects, dsf_dest, exclusions=tile_excl, polygons=tile_polys)
                    except Exception as e:
                        self.log(f"Failed to compile DSF tile {file_name}: {e}", "error")
                        
                    completed_tiles += 1
                    self.update_progress(self.step4_bar, completed_tiles, total_tiles)
                    self.update_progress(self.overall_bar, 75 + (25 * (completed_tiles / total_tiles)), 100)

            for pattern in ("*.proximity.json", "*.footprint.json"):
                for sidecar in obj_dir.glob(pattern):
                    try:
                        sidecar.unlink()
                    except OSError:
                        pass

            if proximity_objects:
                # Plain space-separated text, not JSON -- the companion
                # FlyWithLua script has to parse this with bare Lua string
                # ops (no guaranteed JSON library across FlyWithLua
                # versions), so the on-disk format is kept as simple as the
                # data itself: one placement per line, "dataref lat lon".
                # Dataref names never contain whitespace (mesh_convert's
                # sanitize_name replaces anything outside [A-Za-z0-9._-]
                # with "_", and the only literal characters added on top are
                # "/"), so plain whitespace splitting is unambiguous.
                plugin_data_dir = out / "plugin_data"
                plugin_data_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = plugin_data_dir / "msfs2xp_proximity.dat"
                with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("# msfs2xp proximity-animation manifest -- generated, do not hand-edit\n")
                    f.write("# dataref lat lon\n")
                    for obj in proximity_objects:
                        f.write(f"{obj['dataref']} {obj['lat']:.7f} {obj['lon']:.7f}\n")
                self.log(f"Wrote proximity-animation manifest ({len(proximity_objects)} object(s), "
                         f"{len(set(o['dataref'] for o in proximity_objects))} unique dataref(s)) to "
                         f"{manifest_path}. These objects render correctly but stay in their default "
                         f"(closed) pose without the companion 'msfs2xp Proximity Animator' FlyWithLua "
                         f"script installed -- see the plugin zip.", "info")

            # --- STEP 5: apt.dat Generation ---
            # X-Plane picks exactly one scenery pack's apt.dat block per ICAO
            # (highest priority wins outright, it does not merge runway data
            # from one pack with ramp data from another), so this custom
            # pack's apt.dat has to be a complete, real airport definition --
            # runways, taxiways, ATC flow/routing, ramps -- not just what we
            # can derive from the MSFS BGL. Reuses the same real-world
            # airport's complete block from the user's own installed
            # X-Plane "Global Airports" pack instead (see apt_dat.py).
            self.log(f"\n{'='*50}\n5. GENERATING apt.dat\n{'='*50}", "header")
            if matched_apt_block:
                apt_path = out / "Earth nav data" / "apt.dat"
                # keep_lighting=True: X-Plane draws this airport's real
                # runway edge/centreline/approach/TDZ/REIL lights, PAPI and
                # taxiway edge lights from its own Global Airports data.
                # MSFS represents all of that procedurally (not as converted
                # objects), so without this the airport has no
                # runway/taxiway/approach lighting at all. Pavement + markings
                # stay suppressed -- only the lights come back.
                apt_dat.write_apt_dat(apt_path, matched_apt_block,
                                      source_note=f"matched {matched_apt_ident}", keep_lighting=True,
                                      native_layout=native_airport_layout, airport_name=matched_apt_ident or "")
                if native_airport_layout and not native_airport_layout.is_empty():
                    self.log(f"apt.dat: repositioned {len(native_airport_layout.runway_centers)} runway(s) and "
                             f"replaced the ATC taxi network ({len(native_airport_layout.taxi_nodes)} nodes, "
                             f"{len(native_airport_layout.taxi_edges)} edges) + {len(native_airport_layout.ramp_starts)} "
                             f"ramp starts with this package's own MSFS layout.", "info")
                target_xp = self.xp_version_var.get()
                if target_xp == "xp11":
                    # write_apt_dat always writes BOTH the modern (1200
                    # spec) file and a ".xp11" legacy sidecar (1100 spec,
                    # jetway rows stripped) -- normally the user has to
                    # manually rename the sidecar over apt.dat for an older
                    # (pre-11.50) X-Plane 11 install. Do that automatically
                    # here when the user has told us that's their actual
                    # target, instead of leaving a modern-spec apt.dat
                    # active by default and requiring a manual step.
                    legacy_path = apt_path.with_name(apt_path.name + ".xp11")
                    shutil.copyfile(legacy_path, apt_path)
                    self.log(f"Wrote apt.dat ({len(matched_apt_block)} lines, airport {matched_apt_ident}) -- "
                             f"legacy X-Plane 11 (pre-11.50) variant active, per the selected target version.",
                             "success")
                else:
                    self.log(f"Wrote apt.dat ({len(matched_apt_block)} lines, airport {matched_apt_ident}), plus an "
                             f"apt.dat.xp11 fallback (jetways stripped, legacy 1100 header) for older X-Plane 11 "
                             f"installs that don't read the 1200 spec -- rename it over apt.dat manually if needed.",
                             "success")
            else:
                self.log("No matching default airport found -- skipping apt.dat generation.", "warning")

            # objects/ picked up pipeline-internal scratch sidecars along the
            # way (.meshir.pkl for terrain_fit/draped_merge, .footprint.json/
            # .proximity.json/.originoffset.json for main.py's own placement
            # loop) -- X-Plane never reads any of them, so they don't belong
            # in the final shipped scenery pack. Swept up here, once, at the
            # very end, rather than as each pass finishes, since draped_merge
            # (Step 4) still needs .meshir.pkl sidecars for objects that
            # DIDN'T get merged (as does any future re-run against the same
            # obj_dir).
            removed_sidecars = 0
            for suffix in (*_CONVERT_SIDECAR_SUFFIXES, ".originoffset.json"):
                for sidecar in obj_dir.glob(f"*{suffix}"):
                    try:
                        sidecar.unlink()
                        removed_sidecars += 1
                    except OSError:
                        pass
            if removed_sidecars:
                self.log(f"Cleaned up {removed_sidecars} intermediate sidecar file(s) from {obj_dir} "
                         f"(.meshir.pkl/.footprint.json/.proximity.json/.originoffset.json -- pipeline-"
                         f"internal only, not read by X-Plane).", "info")

            if dsf_tiles:
                self.mark_overall_complete()
            self.log(f"\n{'='*50}\nPIPELINE EXECUTION FINISHED\n{'='*50}", "success")

        finally:
            # Previously only reached on the success path (it sat at the end
            # of the try body) -- any exception raised earlier in the
            # pipeline meant this never ran, silently leaking the whole
            # extracted-package temp dir (easily multiple GB) on every
            # failed run. Moved into finally so it always runs, success or
            # not.
            shutil.rmtree(temp_dir, ignore_errors=True)
            # mesh_convert's disk cache has no reuse-based eviction (see
            # cache_utils.enforce_size_cap's own docstring for why it grew
            # unbounded to 392GB in this project's history) -- a periodic
            # sweep here, success or failure, is the safety net. 30GB is
            # generous for a single-airport iteration loop while nowhere
            # near "fills the disk" territory.
            cache_utils.enforce_size_cap("mesh_convert", 30 * 1024**3, log_callback=self.log)
            self.root.after(0, lambda: self.start_btn.set_state("normal"))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = ModularPythonConverterApp(root)
    root.mainloop()
