"""
apt.dat generation.

X-Plane resolves duplicate ICAOs across scenery packs by taking the whole
airport block from the single highest-priority pack, never merging fields
from two packs -- so a custom apt.dat must be a *complete* airport
(runways, taxiways, ATC flow, ramps) or it silently loses whatever it
omitted once it outranks the default.

Base layer: the real-world airport's block from the user's installed
X-Plane "Global Airports" pack, found by nearest-reference-point match
(more robust than decoding an ICAO string out of the BGL).

On top of that, `reposition_runways`, `replace_pavement_with_native` and
`replace_taxi_network_and_starts` overlay a NATIVE layout decoded from
the MSFS package's own BGL Airport record (see airport_layout.py) where
available: runway positions shifted onto this package's coordinates,
pavement/apron boundary rows replaced outright (so X-Plane's runtime
terrain-flattening matches where the native layout actually is, not the
stock real-world survey position -- a custom-rebuilt payware airport can
legitimately differ from that by more than a trivial amount), and the
ATC taxi-route network + ramp starts replaced with ones built from this
airport's own taxiways/aprons.
Painted-line/lighting rows stay on the stock block -- this project's
draped-mesh pipeline already renders the real baked pavement art and
painted markings, and there's no native replacement yet for the stock
lighting fields.
"""

import math
from pathlib import Path

import cache_utils

# Relative locations of the default "Global Airports" apt.dat across known
# X-Plane installation layouts.
_GLOBAL_AIRPORTS_CANDIDATES = [
    "Custom Scenery/Global Airports/Earth nav data/apt.dat",  # X-Plane 11
    "Global Scenery/Global Airports/Earth nav data/apt.dat",  # X-Plane 12
]

_HEADER_ROW_CODES = {"1", "16", "17"}
_BOUNDARY_NODE_ROW_CODES = {"111", "112", "113", "114", "115", "116"}

# apt.dat's own documented surface type 15: "Transparent -- hard surface,
# but no texture/markings, for use in custom scenery". This is the
# official, spec-defined way to keep a runway/pavement polygon's row
# PRESENT (so ATC, AI ground routing and taxi-route logic, and the
# physical hard-surface collision plane all keep working exactly as
# before) while suppressing X-Plane's own default rendering of it -- the
# whole point being that this scenery's own converted MSFS pavement is
# what should actually be visible there instead.
_TRANSPARENT_SURFACE_CODE = "15"


def find_xplane_root(out_dir: Path):
    """Walks up from the scenery output directory looking for the
    "Custom Scenery" ancestor every X-Plane install has, and returns its
    parent (the X-Plane root). Returns None if out_dir isn't actually
    inside a "Custom Scenery" folder."""
    out_dir = Path(out_dir).resolve()
    for parent in [out_dir] + list(out_dir.parents):
        if parent.name == "Custom Scenery":
            return parent.parent
    return None


def find_global_airports_apt_dat(xplane_root: Path):
    for rel in _GLOBAL_AIRPORTS_CANDIDATES:
        candidate = Path(xplane_root) / rel
        if candidate.is_file():
            return candidate
    return None


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _build_airport_index(apt_dat_path):
    """One binary-mode pass over the (potentially huge, ~300MB+) default
    apt.dat recording (ident, lat, lon, byte_offset) for every airport
    header -- never holds a block's body in memory, just where it starts.
    Binary mode + explicit f.tell()/f.readline() (rather than `for line in
    f`, whose read-ahead buffering makes tell() unreliable in text mode) so
    the recorded offsets are exact seek targets later."""
    index = []
    current_ident = None
    current_lat = None
    current_lon = None
    current_offset = None

    def _finalize():
        if current_ident is not None and current_lat is not None and current_lon is not None:
            index.append((current_ident, current_lat, current_lon, current_offset))

    with open(apt_dat_path, "rb") as f:
        offset = f.tell()
        raw = f.readline()
        while raw:
            stripped = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
            parts = stripped.split()
            if parts:
                code = parts[0]
                if code in _HEADER_ROW_CODES and len(parts) >= 5:
                    _finalize()
                    current_ident = parts[4]
                    current_lat = None
                    current_lon = None
                    current_offset = offset
                elif code == "99":
                    break
                elif code == "1302" and len(parts) >= 3:
                    if parts[1] == "datum_lat":
                        try:
                            current_lat = float(parts[2])
                        except ValueError:
                            pass
                    elif parts[1] == "datum_lon":
                        try:
                            current_lon = float(parts[2])
                        except ValueError:
                            pass
                    elif parts[1] == "icao_code" and parts[2]:
                        current_ident = parts[2]
            offset = f.tell()
            raw = f.readline()
        _finalize()

    return index


def _extract_block_at_offset(apt_dat_path, byte_offset):
    """Re-reads just one airport's block (header line up to, but not
    including, the next header row or the terminating "99" row), seeking
    straight to it instead of scanning the whole file."""
    lines = []
    with open(apt_dat_path, "rb") as f:
        f.seek(byte_offset)
        first = True
        for raw in f:
            stripped = raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
            parts = stripped.split()
            code = parts[0] if parts else ""
            if not first and code in _HEADER_ROW_CODES and len(parts) >= 5:
                break
            if code == "99":
                break
            lines.append(stripped)
            first = False
    return lines


def find_nearest_airport_block(apt_dat_path: Path, target_lat: float, target_lon: float, log_callback=None):
    """Returns (block_lines, ident, distance_km) for the nearest airport's
    complete block, or None if the file has no parseable airport blocks at
    all.

    GPU fork: the (ident, lat, lon, byte_offset) index built by
    _build_airport_index() is disk-cached per apt.dat file (path+size+
    mtime), so only the FIRST run against a given X-Plane install's ~300MB+
    default apt.dat pays for a full-file scan -- every later run (this tool
    gets re-run a lot while iterating on a package) loads the small cached
    index, finds the nearest entry in memory, and seeks straight to just
    that one block instead of re-scanning the whole file.
    """
    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)

    apt_dat_path = Path(apt_dat_path)
    key_parts = (cache_utils.file_identity(apt_dat_path),)
    index = cache_utils.get("apt_dat_index", *key_parts)
    if index is None:
        index = _build_airport_index(apt_dat_path)
        cache_utils.set("apt_dat_index", index, *key_parts)

    if not index:
        _log(f"No parseable airport blocks found in {apt_dat_path}", "warning")
        return None

    best_ident, best_offset, best_distance = None, None, None
    for ident, lat, lon, offset in index:
        dist = _haversine_km(target_lat, target_lon, lat, lon)
        if best_distance is None or dist < best_distance:
            best_distance, best_ident, best_offset = dist, ident, offset

    best_lines = _extract_block_at_offset(apt_dat_path, best_offset)

    _log(f"Nearest airport match in {apt_dat_path.name}: {best_ident} ({best_distance*1000:.0f}m from BGL airport reference point)", "info")
    return best_lines, best_ident, best_distance


def extract_boundary_ring(block_lines):
    """Finds the "130 Airport Boundary" row (if any) within an already-
    extracted airport block and returns the (lat, lon) points of the node
    rows that immediately follow it -- these are a totally different
    row-code family (111-116) also used for taxiway/pavement polygons
    elsewhere in the SAME block, so this only collects the run of node rows
    starting right after "130" and stops at the first row that isn't one,
    rather than grabbing every 111-116 row in the whole file.
    """
    points = []
    in_boundary = False
    for line in block_lines:
        parts = line.split()
        if not parts:
            continue
        code = parts[0]
        if code == "130":
            in_boundary = True
            continue
        if in_boundary:
            if code in _BOUNDARY_NODE_ROW_CODES and len(parts) >= 3:
                try:
                    points.append((float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
                continue
            break
    return points


def boundary_bounding_box(points, pad_km=0.05):
    """Returns {"west","south","east","north"} enclosing all (lat, lon)
    points, padded outward by pad_km (default 50m, just enough to cover
    boundary-node/runway-edge rounding) on every side. None if points is
    empty."""
    if not points:
        return None
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    lat_pad = pad_km / 111.0
    mid_lat = sum(lats) / len(lats)
    lon_pad = pad_km / (111.0 * max(0.1, math.cos(math.radians(mid_lat))))
    return {
        "west": min(lons) - lon_pad,
        "east": max(lons) + lon_pad,
        "south": min(lats) - lat_pad,
        "north": max(lats) + lat_pad,
    }


def bbox_is_near(candidate, trusted, max_km):
    """True if `candidate`'s own center point is within max_km of
    `trusted`'s center point, or if either is None.

    Sanity-gates a bounding box from a single, fragile source (a
    default-apt.dat nearest-airport match; a hand-decoded BGL Airport
    reference point) against one built from real converted placement
    coordinates, before unioning them -- a plain min/max union has no
    other way to reject a candidate centered somewhere else entirely."""
    if candidate is None or trusted is None:
        return True
    t_lat = (trusted["south"] + trusted["north"]) / 2.0
    t_lon = (trusted["west"] + trusted["east"]) / 2.0
    c_lat = (candidate["south"] + candidate["north"]) / 2.0
    c_lon = (candidate["west"] + candidate["east"]) / 2.0
    return _haversine_km(t_lat, t_lon, c_lat, c_lon) <= max_km


def strip_jetway_rows(block_lines):
    """Drops row code 1500 (jetways) -- the one specific row family that
    doesn't exist at all in the pre-1200 apt.dat spec and is what X-Plane
    actually complains about ("no jetways in pre 1200 apt.dat files") when
    a file declares an older version but contains one. Deliberately narrow:
    only removes exactly that row family rather than guessing at every row
    code the 1200 spec added, since getting that guess wrong risks quietly
    breaking a legacy-header file in some OTHER way instead."""
    return [line for line in block_lines if line.split(None, 1)[:1] != ["1500"]]


def _linear_feature_node_is_lit(node_line):
    """A row-120 (free-standing linear feature) node row looks like
    ``11x lat lon [bez_lat bez_lon] [line_style [light_style]]``.  It draws
    LIGHTS -- taxiway green centre-line, amber ILS-hold centre-line,
    lead-in / hold-short lights -- if any trailing style token is an
    apt.dat light code: painted line styles are 1-99, light styles are
    101+ (X-Plane apt.dat spec).  SoFly LHBP encodes ~80 lit taxi
    centre-line features this way, which the old blanket row-120 drop was
    throwing away (user: "nowhere the centerline or the taxiedge")."""
    p = node_line.split()
    if len(p) < 3:
        return False
    style_start = 5 if p[0] in ("112", "114", "116") else 3
    for tok in p[style_start:]:
        try:
            if int(tok) >= 100:
                return True
        except ValueError:
            return False
    return False


def _strip_paint_keep_light(node_line):
    """Rewrite a kept row-120 node so it draws ONLY its lights, not its
    painted line: ``11x lat lon [bez] <lightcode>`` if the node carries a
    light style (>=100 in either style slot), else just ``11x lat lon
    [bez]`` (shape only). The converted package already carries every
    painted taxi/apron marking as its own draped geometry -- leaving the
    default's painted stripe on too is what "some of the original markings
    appeared again" was: 144 of the 330 kept nodes had a paint style
    (60/58/61/...) riding alongside the light code."""
    p = node_line.split()
    if len(p) < 3:
        return node_line
    bez = p[0] in ("112", "114", "116")
    head = p[:5] if bez else p[:3]
    styles = p[5:] if bez else p[3:]
    light = None
    for tok in styles:
        try:
            if int(tok) >= 100:
                light = tok
                break
        except ValueError:
            break
    return " ".join(head + ([light] if light else []))


def anonymize_visual_pavement(block_lines, keep_lighting=True):
    """Suppresses the default Global Airports block's own visible
    pavement, painted markings, signage and beacon, while keeping
    everything operational (ATC frequencies, taxi-route network, ramp
    starts, helipads, and the runway/taxiway rows themselves).

    keep_lighting (default True): keep X-Plane's own runway/taxiway/
    approach/PAPI/VASI lighting, since MSFS represents all of that
    procedurally rather than as placed objects the conversion could
    reproduce -- zeroing it leaves the airport dark at night. Set False
    only for a package that genuinely converts its own lighting.

    This block is copied wholesale from the user's own X-Plane install,
    so once written here it's this package's own active apt.dat data --
    nothing at the DSF exclusion-rectangle level (main.py) can suppress a
    beacon/light/sign this block still declares; it has to not be
    declared here instead. Techniques used, since apt.dat only gives some
    rows an "exists but invisible" option:
      - Row 100 (runway), 102 (helipad), 110 (pavement polygon): surface
        type set to 15 ("Transparent") -- present, collidable, no default
        texture/markings. Shoulder surface zeroed too.
      - Row 100 also zeroes centerline/edge lights, auto-distance-signs,
        and per-end markings/approach/TDZ/REIL fields. Row 102 loses its
        markings/edge-light fields the same way.
      - Rows 111-116 (pavement/boundary nodes): truncated to just their
        coordinates (plus bezier control point for 112/114/116), dropping
        the painted-line and edge-light codes X-Plane draws from these
        specifically (not row 120), while keeping the polygon shape for
        the hard surface + ATC ground routing.
      - Rows 18 (beacon), 19 (windsock), 20 (signs), 21 (VASI/PAPI/
        wig-wag) and 120 (painted lines) are dropped outright, no
        invisible variant needed -- 19/21 because this package's own BGL
        extraction already places real MSFS-sourced windsocks/lights as
        converted objects, so keeping the default's would duplicate them.
    """
    # 111/113/115 nodes: "lat lon [line_type [light_type]]" -> keep 3 tokens.
    # 112/114/116 nodes: "lat lon bez_lat bez_lon [line_type [light_type]]" -> keep 5.
    node_keep = {"111": 3, "113": 3, "115": 3, "112": 5, "114": 5, "116": 5}

    out = []
    i = 0
    n = len(block_lines)
    while i < n:
        line = block_lines[i]
        parts = line.split()
        code = parts[0] if parts else ""

        drop_codes = ("18", "19", "20", "1500", "1501") if keep_lighting else ("18", "19", "20", "21", "1500", "1501")
        if code in drop_codes:
            # 18 beacon / 19 windsock / 20 taxi signs / 1500-1501 auto
            # jetways -- things this package already ships its own converted
            # MSFS versions of, so leaving the default's in just doubles
            # them. Row 21 (VASI/PAPI/wig-wag) is KEPT when keep_lighting:
            # MSFS approach-slope + runway-guard lighting is procedural, not
            # a converted object, so dropping it left no PAPI at all.
            i += 1
            continue

        if code == "120":
            j = i + 1
            feat_nodes = []
            while j < n:
                nxt = block_lines[j].split()
                if nxt and nxt[0] in _BOUNDARY_NODE_ROW_CODES:
                    feat_nodes.append(block_lines[j])
                    j += 1
                    continue
                break
            if keep_lighting and any(_linear_feature_node_is_lit(x) for x in feat_nodes):
                # This linear feature carries taxiway centre-line/hold-
                # short/lead-in LIGHTS, which MSFS renders procedurally
                # (not converted) -- keep the feature but strip each
                # node's PAINTED-line style, keeping only the light code,
                # since the converted package already ships every
                # painted marking as its own draped geometry.
                out.append(line)
                out.extend(_strip_paint_keep_light(x) for x in feat_nodes)
            # else: purely-painted decoration -> drop the feature and nodes
            i = j
            continue

        if code == "100" and len(parts) >= 8:
            parts[2] = _TRANSPARENT_SURFACE_CODE
            parts[3] = "0"  # paved shoulder: none
            parts[7] = "0"  # auto-generate distance-remaining signs: off
            if not keep_lighting:
                parts[5] = "0"  # centerline lights: off
                parts[6] = "0"  # edge lighting: off
            # Per-end blocks: 8 header fields, then 9 fields per end
            # (desig lat lon disp overrun markings apch tdz reil). Always
            # zero markings(+5) -- the converted pavement carries them.
            # Only zero apch(+6)/tdz(+7)/reil(+8) when not keeping lighting.
            _end_offs = (5,) if keep_lighting else (5, 6, 7, 8)
            for base in (8, 17):
                for off in _end_offs:
                    if base + off < len(parts):
                        parts[base + off] = "0"
            out.append(" ".join(parts))
            i += 1
            continue

        if code == "102" and len(parts) >= 9:
            # 102 desig lat lon orient len width surface markings shoulder smooth edge_lights
            parts[7] = _TRANSPARENT_SURFACE_CODE  # surface
            parts[8] = "0"                        # markings
            if not keep_lighting and len(parts) >= 12:
                parts[11] = "0"                  # edge lights
            out.append(" ".join(parts))
            i += 1
            continue

        if code == "110" and len(parts) >= 2:
            parts[1] = _TRANSPARENT_SURFACE_CODE
            out.append(" ".join(parts))
            i += 1
            continue

        if code in node_keep:
            base = node_keep[code]
            # parts: [code, lat, lon, (bez_lat, bez_lon,) line_type?, light_type?]
            if keep_lighting and _linear_feature_node_is_lit(line):
                # Keep the taxiway/apron EDGE-LIGHT code, drop the painted
                # edge line that rides with it (line style 53 etc). 312 of
                # LHBP's pavement boundary nodes carry a `53 102` pair --
                # keeping it verbatim repaints X-Plane's default edge
                # stripes right over the converted markings ("the original
                # XP scenery still appears"). `<coords> 102` -- light code
                # in the style-1 slot -- is a form the source itself uses.
                out.append(_strip_paint_keep_light(line))
            else:
                out.append(" ".join(parts[:base]))
            i += 1
            continue

        out.append(line)
        i += 1

    return out


_APT_DAT_VERSION = "1100"


def _dist_m(a, b):
    """Flat local approximation (same as geo_transform.metres_per_degree's
    intent, kept local to avoid a new import for just a nearest-match
    distance check) -- fine at the runway-to-runway matching distances
    this is used for."""
    lat1, lon1 = a
    lat2, lon2 = b
    dlat = (lat2 - lat1) * 111320.0
    dlon = (lon2 - lon1) * 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dlat, dlon)


def _runway_end_positions(parts):
    return (float(parts[9]), float(parts[10])), (float(parts[18]), float(parts[19]))


def reposition_runways(block_lines, native_runway_centers):
    """Shift each stock row-100 runway's two endpoints by the delta between
    its own stock-block center and the nearest natively-decoded MSFS
    runway center (matched by proximity, each native center consumed at
    most once), preserving every other field (width, surface, lighting,
    markings, displaced thresholds) exactly. Deliberately NOT a full
    native rebuild of row 100: the row's other ~20 fields have no
    confirmed BGL decode (see airport_layout.py's docstring on what
    wasn't decoded), and the stock block's real-world values for those are
    a better bet than a guess. Position is the one thing confirmed wrong
    today -- a custom-rebuilt package's runway can legitimately sit a few
    meters from the real-world stock coordinates."""
    if not native_runway_centers:
        return block_lines

    remaining = list(native_runway_centers)
    out = []
    for line in block_lines:
        parts = line.split()
        if parts[:1] == ["100"] and len(parts) >= 20 and remaining:
            end1, end2 = _runway_end_positions(parts)
            stock_center = ((end1[0] + end2[0]) / 2.0, (end1[1] + end2[1]) / 2.0)
            nearest = min(remaining, key=lambda c: _dist_m(stock_center, c))
            dlat, dlon = nearest[0] - stock_center[0], nearest[1] - stock_center[1]
            remaining.remove(nearest)
            parts[9] = f"{end1[0] + dlat:.8f}"
            parts[10] = f"{end1[1] + dlon:.8f}"
            parts[18] = f"{end2[0] + dlat:.8f}"
            parts[19] = f"{end2[1] + dlon:.8f}"
            out.append(" ".join(parts))
            continue
        out.append(line)
    return out


def replace_pavement_with_native(block_lines, layout):
    """Strips the stock block's own pavement/apron boundary rows (110 +
    its 111-116 node rows) and replaces them with ones built from this
    package's own MSFS apron layout (airport_layout.py's native BGL
    decode), so X-Plane's runtime terrain-flattening -- driven by these
    boundary rows, not by the runway centerline alone -- matches where
    this package's own converted draped-mesh pavement actually sits.

    CONFIRMED REAL BUG this fixes: reposition_runways only shifts the
    runway CENTERLINE (row 100) to the native position; the surrounding
    pavement boundary was left at the stock block's own real-world
    position. A custom-rebuilt payware airport's own layout can
    legitimately differ from the real-world survey data by more than a
    trivial amount, so X-Plane was flattening terrain around the OLD
    (stock) boundary while this package's own MSFS-derived visual
    pavement sits at the NEW (native) one -- reported in-sim as "the
    runway is floating" (and per the user, "I don't want the native
    X-Plane [pavement] ... however the MSFS texturized ones [should be
    what's] exported").

    Each native apron polygon becomes one row 110 (surface forced
    transparent, matching anonymize_visual_pavement's own convention --
    this package's own draped mesh is what should actually be visible,
    same as the stock block's own pavement already gets anonymized to)
    followed by one row 111 per vertex except the last, which closes the
    loop as row 113. No curve/bezier data is available from the native
    decode, so every edge is a straight segment (111/113 only, never
    112/114). Runway (100), taxi network (1200s), ramp starts (1300s)
    and painted-line (120) rows are untouched -- see reposition_runways/
    replace_taxi_network_and_starts for those."""
    if not layout.aprons:
        return block_lines

    out = []
    i = 0
    n = len(block_lines)
    while i < n:
        line = block_lines[i]
        parts = line.split()
        code = parts[0] if parts else ""
        if code == "110":
            j = i + 1
            while j < n:
                nxt_parts = block_lines[j].split()
                if nxt_parts and nxt_parts[0] in _BOUNDARY_NODE_ROW_CODES:
                    j += 1
                    continue
                break
            i = j
            continue
        out.append(line)
        i += 1

    for poly in layout.aprons:
        verts = poly.vertices
        if len(verts) < 3:
            continue
        out.append(f"110 {_TRANSPARENT_SURFACE_CODE} 0.25 0.0")
        for lat, lon in verts[:-1]:
            out.append(f"111 {lat:.8f} {lon:.8f}")
        last_lat, last_lon = verts[-1]
        out.append(f"113 {last_lat:.8f} {last_lon:.8f}")

    return out


_TAXI_NETWORK_ROW_CODES = {"1200", "1201", "1202", "1204", "1206"}
"""1206 ("<node1> <node2> <direction>", a truck/ground-vehicle-only taxi
edge -- X-Plane's ground-service-vehicle AI routes over these separately
from the aircraft 1202 network) references 1201 node IDs exactly like
1202 does, so renumbering the 1201 nodes for a native layout leaves 1206
rows dangling unless they're re-anchored too. No native BGL decode exists
for truck-only edges, so they can't be rebuilt from scratch; instead
they're RE-ANCHORED onto the new node numbering by nearest position (see
_remap_truck_edges) rather than dropped, since dropping them silently
disables ground-service-vehicle AI at this airport entirely."""
_RAMP_START_ROW_CODES = {"1300", "1301", "1400", "1401"}

_MAX_STAND_MATCH_M = 50.0
"""Max distance to borrow a stock block's ramp-start heading/name for a
native one -- generous enough for a custom-rebuilt stand to have moved a
bit from the real-world stock position (matches reposition_runways'
own tolerance-by-nearest-match philosophy), but tight enough not to
borrow an unrelated stand's name/orientation from across the apron."""


def _parse_stock_ramp_starts(block_lines):
    """Row 1300: "1300 <lat> <lon> <heading> <type> <airplane_types>
    <name>". Parsed here (before replace_taxi_network_and_starts strips
    these rows) so a native ramp start with no confirmed heading/name of
    its own (see that function's docstring) can borrow the closest real
    stand's -- position is NOT borrowed; the native decode already gives
    that directly, and more accurately, for this specific package."""
    starts = []
    for line in block_lines:
        parts = line.split()
        if parts[:1] != ["1300"] or len(parts) < 6:
            continue
        try:
            lat, lon, hdg = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        name = " ".join(parts[6:])
        starts.append((lat, lon, hdg, name))
    return starts


def _match_ramp_start_metadata(native_starts, stock_starts):
    """Greedy nearest-match, each stock start consumed at most once (same
    one-to-one philosophy as reposition_runways): for each native (lat,
    lon), finds the closest not-yet-used stock 1300 row within
    _MAX_STAND_MATCH_M and returns its (heading, name); (None, None) for
    a native start with no close-enough stock match."""
    available = list(range(len(stock_starts)))
    results = []
    for lat, lon in native_starts:
        best_idx, best_dist = None, None
        for idx in available:
            s_lat, s_lon, _, _ = stock_starts[idx]
            d = _dist_m((lat, lon), (s_lat, s_lon))
            if best_dist is None or d < best_dist:
                best_idx, best_dist = idx, d
        if best_idx is not None and best_dist <= _MAX_STAND_MATCH_M:
            _, _, hdg, name = stock_starts[best_idx]
            results.append((hdg, name))
            available.remove(best_idx)
        else:
            results.append((None, None))
    return results


def _parse_stock_taxi_nodes(block_lines):
    """Row 1201: "1201 <lat> <lon> <usage> <node_id> <name>". Parsed here
    (before replace_taxi_network_and_starts strips these rows) purely so
    a stock 1206 ground-service-vehicle-only edge (see
    _TAXI_NETWORK_ROW_CODES's own docstring -- no native BGL decode
    exists for these) can be re-anchored onto the NEW native node
    numbering by position instead of being dropped outright. Keyed by
    the stock file's own node_id string, not assumed to be a plain
    0..N-1 index."""
    nodes = {}
    for line in block_lines:
        parts = line.split()
        if parts[:1] != ["1201"] or len(parts) < 5:
            continue
        try:
            lat, lon = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        nodes[parts[4]] = (lat, lon)
    return nodes


def _parse_stock_truck_edges(block_lines):
    """Row 1206: "1206 <node1> <node2> <direction>". Parsed here for the
    same re-anchoring reason as _parse_stock_taxi_nodes."""
    edges = []
    for line in block_lines:
        parts = line.split()
        if parts[:1] != ["1206"] or len(parts) < 4:
            continue
        edges.append((parts[1], parts[2], parts[3]))
    return edges


_MAX_TRUCK_NODE_MATCH_M = 50.0


def _remap_truck_edges(stock_nodes, stock_truck_edges, native_nodes):
    """Re-anchors the stock block's 1206 ground-service-vehicle-only taxi
    edges onto the new native taxi-node numbering (native_nodes:
    layout.taxi_nodes, index == its eventual 1201 node_id) by nearest
    position, rather than dropping them: there's no BGL source at all for
    which taxiways trucks specifically use, so this carries over the real
    truck-routing topology onto the new node geometry instead.

    Each stock node id is matched to its nearest native node ONCE and
    cached (not consumed like reposition_runways/_match_ramp_start_
    metadata do), since a stock node is typically shared by several
    edges and they must all resolve to the same native node to stay
    connected. An edge with no close-enough match (nothing within
    _MAX_TRUCK_NODE_MATCH_M) or that collapses to a self-loop is
    dropped."""
    if not stock_truck_edges or not native_nodes:
        return []

    match_cache = {}

    def _nearest_native(pos):
        if pos not in match_cache:
            best_idx, best_dist = None, None
            for idx, n_pos in enumerate(native_nodes):
                d = _dist_m(pos, n_pos)
                if best_dist is None or d < best_dist:
                    best_idx, best_dist = idx, d
            match_cache[pos] = best_idx if best_idx is not None and best_dist <= _MAX_TRUCK_NODE_MATCH_M else None
        return match_cache[pos]

    remapped = []
    for a_id, b_id, direction in stock_truck_edges:
        a_pos, b_pos = stock_nodes.get(a_id), stock_nodes.get(b_id)
        if a_pos is None or b_pos is None:
            continue
        new_a, new_b = _nearest_native(a_pos), _nearest_native(b_pos)
        if new_a is None or new_b is None or new_a == new_b:
            continue
        remapped.append((new_a, new_b, direction))
    return remapped


def replace_taxi_network_and_starts(block_lines, layout, airport_name=""):
    """Strip the stock block's ATC taxi-route network and ramp-start rows
    and replace them with ones built from this package's own MSFS layout,
    so ATC/AI ground routing and startup positions match this specific
    airport rather than the real-world stock layout. Runway (100),
    pavement (110-116, see replace_pavement_with_native) and painted-line
    (120) rows are untouched here.

    Heading/name on native ramp starts have no confirmed BGL decode, so
    each borrows the closest stock stand's (see _match_ramp_start_
    metadata) when one is within reach, falling back to a generic
    placeholder otherwise. type/airplane_types stay "misc"/"all" and taxi
    edges stay "twoway" since the source data doesn't confirm any
    restriction and a wrong restriction would silently block routing.

    Ground-service-vehicle routing (1206) has no native BGL decode at
    all, so it's re-anchored from the stock block's own truck-route
    topology instead -- see _remap_truck_edges."""
    if not layout.taxi_nodes and not layout.ramp_starts:
        return block_lines

    stock_starts = _parse_stock_ramp_starts(block_lines)
    stock_taxi_nodes = _parse_stock_taxi_nodes(block_lines)
    stock_truck_edges = _parse_stock_truck_edges(block_lines)

    drop_codes = _TAXI_NETWORK_ROW_CODES | _RAMP_START_ROW_CODES
    out = []
    for line in block_lines:
        stripped = line.strip()
        code = stripped.split(None, 1)[0] if stripped else ""
        if code in drop_codes:
            continue
        out.append(line)

    if layout.taxi_nodes:
        out.append(f"1200 {airport_name}".rstrip())
        for i, (lat, lon) in enumerate(layout.taxi_nodes):
            out.append(f"1201 {lat:.8f} {lon:.8f} both {i} n{i}")
        for node_a, node_b in layout.taxi_edges:
            out.append(f"1202 {node_a} {node_b} twoway taxiway_F")
        for new_a, new_b, direction in _remap_truck_edges(stock_taxi_nodes, stock_truck_edges, layout.taxi_nodes):
            out.append(f"1206 {new_a} {new_b} {direction}")

    matched_metadata = _match_ramp_start_metadata(layout.ramp_starts, stock_starts)
    for i, ((lat, lon), (hdg, name)) in enumerate(zip(layout.ramp_starts, matched_metadata)):
        hdg = hdg if hdg is not None else 0.0
        name = name if name else f"Start {i + 1}"
        out.append(f"1300 {lat:.8f} {lon:.8f} {hdg:.2f} misc all {name}")

    return out


def write_apt_dat(out_path: Path, block_lines, source_note: str, keep_lighting=True, native_layout=None, airport_name=""):
    if native_layout is not None and not native_layout.is_empty():
        block_lines = reposition_runways(block_lines, native_layout.runway_centers)
        block_lines = replace_pavement_with_native(block_lines, native_layout)
        block_lines = replace_taxi_network_and_starts(block_lines, native_layout, airport_name=airport_name)
    block_lines = anonymize_visual_pavement(block_lines, keep_lighting=keep_lighting)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Header version 1100: every surviving row code (anonymize_visual_
    # pavement already drops anything needing a newer spec, e.g. 1500
    # jetways) is valid at 1100, and a 1200-header file was observed NOT
    # overriding the 1130 Global Airports block for the same ICAO, unlike
    # 1000/1100 (what every working custom-airport pack ships).
    for path, lines in ((out_path, block_lines),
                        (out_path.with_name(out_path.name + ".xp11"), strip_jetway_rows(block_lines))):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("I\n")
            f.write(f"{_APT_DAT_VERSION} Generated by msfs2xp ({source_note})\n\n")
            for line in lines:
                f.write(line + "\n")
            f.write("99\n")


def generate_apt_dat(out_dir: Path, xplane_root, airport_lat, airport_lon, log_callback=None,
                      max_plausible_distance_km=15.0):
    """Top-level entry point: locate the default Global Airports apt.dat,
    find the nearest complete airport block to (airport_lat, airport_lon),
    and write it out to out_dir/"Earth nav data"/apt.dat. Returns True on
    success, False otherwise (never raises -- a missing/failed apt.dat
    shouldn't abort the rest of the conversion).
    """
    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)

    if airport_lat is None or airport_lon is None:
        _log("No BGL airport reference point available -- skipping apt.dat generation.", "warning")
        return False

    if xplane_root is None:
        _log("Could not locate the X-Plane installation root from the output path "
             "(expected a \"Custom Scenery\" ancestor folder) -- skipping apt.dat generation.", "warning")
        return False

    global_apt_dat = find_global_airports_apt_dat(xplane_root)
    if global_apt_dat is None:
        _log(f"Could not find the default Global Airports apt.dat under {xplane_root} -- "
             f"skipping apt.dat generation.", "warning")
        return False

    result = find_nearest_airport_block(global_apt_dat, airport_lat, airport_lon, log_callback=log_callback)
    if result is None:
        return False

    block_lines, ident, distance_km = result
    if distance_km > max_plausible_distance_km:
        _log(f"Nearest airport match ({ident}) is {distance_km:.1f}km from the BGL airport "
             f"reference point -- too far to be a confident match, skipping apt.dat generation.", "warning")
        return False

    out_path = Path(out_dir) / "Earth nav data" / "apt.dat"
    write_apt_dat(out_path, block_lines, source_note=f"matched {ident} from {global_apt_dat}")
    _log(f"Wrote {out_path} ({len(block_lines)} lines, airport {ident}, "
         f"{distance_km*1000:.0f}m from reference point).", "success")
    return True
