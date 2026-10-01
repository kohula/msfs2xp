"""A complete apt.dat airport built from the package's own BGL airport record.

Input is an airport_layout.AirportLayout (decoded from the MSFS package);
output is the list of apt.dat rows for one airport, ready for
apt_dat.write_airport_lines. Everything X-Plane needs to operate the
airport comes from MSFS: runway ends, displaced thresholds, approach
lighting, VASI/PAPI, frequencies, taxiway signs, windsocks, the ATC taxi
network (with runway edges, taxiway names, width classes and hot zones),
and the ramp starts with their heading, type, size class and airlines.

The VISUAL ground stays this project's own: runways and apron polygons are
written with surface 15 (transparent) by default, so the converted MSFS
pavement drawn as draped geometry is what shows, while X-Plane still gets
the hard surface and the lights. `runway_surface="native"` draws real
X-Plane runways with markings instead, for packages that ship no runway
geometry of their own; `painted_lines=True` adds the MSFS painted-line
records as apt.dat lines (off by default -- packages whose draped models
already carry their markings would show them twice).

A matched stock Global Airports block, when there is one, contributes what
MSFS has no equivalent for: ATC runway-use flows, airport metadata (city,
country, IATA code...), the rotating beacon, and ground-vehicle routes when
the package defines none.
"""

import math
from collections import deque

import apt_dat
from airport_layout import is_aircraft_stand
from geo_transform import metres_per_degree

TRANSPARENT = 15
FEET_PER_M = 3.2808399

# MSFS surface code -> X-Plane surface code.
_SURFACE = {
    0x00: 2, 0x03: 2, 0x10: 2, 0x12: 2, 0x14: 2,  # concrete, cement, steel mats, brick, planks
    0x04: 1, 0x0F: 1, 0x11: 1, 0x13: 1, 0x17: 1,  # asphalt, oil-treated, bituminous, macadam, tarmac
    0x01: 3,  # grass
    0x07: 4, 0x0C: 4, 0x0D: 4, 0x15: 4, 0x16: 4,  # clay, dirt, coral, sand, shale
    0x0E: 5,  # gravel
    0x08: 12, 0x09: 12,  # snow, ice -> dry lakebed is the nearest "white" X-Plane has
    0x02: 13,  # water
    0x3F: TRANSPARENT,
}
_WATER = 0x02

# MSFS approach lighting system -> apt.dat code.
_ALS = {1: 11, 2: 9, 3: 8, 4: 6, 5: 5, 6: 1, 7: 2, 8: 12, 9: 3, 10: 4, 11: 10, 12: 7, 13: 7, 14: 7}

# MSFS COM type -> apt.dat 8.33 kHz frequency row.
_FREQ_ROW = {1: 1050, 12: 1050, 13: 1050, 2: 1051, 3: 1051, 4: 1051, 7: 1052, 14: 1052, 15: 1052,
             5: 1053, 6: 1054, 8: 1055, 9: 1056}

# MSFS parking types.
_GATES = (0x08, 0x09, 0x0A, 0x0F, 0x10)
_CARGO = (0x05, 0x06)
_MILITARY = (0x06, 0x07)

# MSFS painted-line style index -> (apt.dat line code, light code when lit).
_PAINTED = {
    0: (1, 101), 12: (1, 101), 10: (1, 101),  # centre lines
    1: (4, 103), 2: (4, 103), 19: (4, 103), 20: (4, 103),  # runway hold short
    7: (5, 103),  # taxiway hold short
    8: (6, 105),  # ILS hold short
    4: (3, 102), 15: (3, 102), 16: (3, 102),  # solid edge
    3: (9, 102),  # dashed edge
    11: (2, 0), 14: (2, 0),  # non-movement boundary
    6: (20, 0), 13: (20, 0),  # white solid
    5: (22, 0), 9: (22, 0),  # white broken
    18: (30, 0), 17: (32, 0),  # red
}
_PAINTED_REVERSED = (2, 14, 20)  # backward hold-short / boundary styles: X-Plane encodes the side by node order

# A taxi edge further than this from a runway's centreline is never in its hot zone.
_HOT_ZONE_BAND_M = 150.0


# --- geometry -----------------------------------------------------------------


class _Plane:
    """Local east/north metres around a reference point (WGS84 scale)."""

    def __init__(self, lat, lon):
        self.lat, self.lon = lat, lon
        self.m_lat, self.m_lon = metres_per_degree(lat)

    def xy(self, lat, lon):
        return (lon - self.lon) * self.m_lon, (lat - self.lat) * self.m_lat

    def latlon(self, x, y):
        return self.lat + y / self.m_lat, self.lon + x / self.m_lon


def _move(lat, lon, bearing_deg, dist_m):
    m_lat, m_lon = metres_per_degree(lat)
    b = math.radians(bearing_deg)
    return lat + dist_m * math.cos(b) / m_lat, lon + dist_m * math.sin(b) / m_lon


def runway_ends(rw):
    """(primary, secondary) threshold positions. MSFS stores the centre,
    true heading and length; the primary end is the one the heading points
    away from."""
    half = rw.length_m / 2.0
    return _move(rw.lat, rw.lon, rw.heading_true + 180.0, half), _move(rw.lat, rw.lon, rw.heading_true, half)


def _dist_to_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 < 1e-9 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _convex_hull(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        return []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull if len(hull) >= 3 else []


def _token(s, fallback):
    t = "".join(c for c in str(s) if not c.isspace())
    return t or fallback


def _text(s):
    return " ".join(str(s).split())


# --- signs -----------------------------------------------------------------------

_SIGN_GLYPHS = {">": "{^r}", "<": "{^l}", "^": "{^u}", "v": "{^d}", "'": "{^ru}", "`": "{^lu}",
                "/": "{^ld}", "\\": "{^rd}", "_": "{_}", " ": "{_}", "|": "|", "-": "-"}
_SIGN_COLOURS = {"l": "{@L}", "d": "{@Y}", "i": "{@Y}", "u": "{@Y}", "m": "{@R}", "r": "{@R}"}


def translate_sign(label):
    """MSFS sign label -> X-Plane sign text, or None if it can't be drawn.
    MSFS: a lower-case letter picks the panel colour for what follows
    (l location, d/i/u direction or information, m/r mandatory), brackets
    draw a border, punctuation draws arrows. X-Plane: {@L} {@Y} {@R} pick
    the colour (a colour change starts a new panel), {^r} etc. are arrows,
    borders are automatic -- so brackets are dropped."""
    out = []
    pending = None
    glyphs = 0
    for c in label:
        if c in _SIGN_COLOURS:
            pending = _SIGN_COLOURS[c]
            continue
        if c in "[]":
            continue
        if c.isupper() or c.isdigit():
            glyph = c
        elif c in _SIGN_GLYPHS:
            glyph = _SIGN_GLYPHS[c]
        else:
            return None
        if pending and (not out or out[-1] != pending):
            out.append(pending)
        pending = None
        out.append(glyph)
        glyphs += 1
    if not glyphs:
        return None
    if not out[0].startswith("{@"):
        out.insert(0, "{@Y}")
    return "".join(out)


# --- the airport ------------------------------------------------------------------


def _stock_rows(stock_block, codes):
    return [line for line in (stock_block or []) if line.split(None, 1)[:1] and line.split(None, 1)[0] in codes]


def _header(layout, stock_block):
    water = [rw for rw in layout.runways if rw.surface == _WATER]
    if not layout.runways and layout.helipads:
        kind = 17
    elif layout.runways and len(water) == len(layout.runways):
        kind = 16
    else:
        kind = 1
    ident = layout.ident
    name = layout.name
    stock_head = next((line.split() for line in (stock_block or []) if line.split()[:1] in (["1"], ["16"], ["17"])), None)
    if stock_head and len(stock_head) >= 5:
        ident = ident or stock_head[4]
        name = name or " ".join(stock_head[5:])
    ident = _token(ident, "XXXX")
    has_tower = 1 if layout.tower else 0
    return [f"{kind} {round(layout.alt_m * FEET_PER_M)} {has_tower} 0 {ident} {_text(name) or ident}"], ident, kind


def _metadata(layout, stock_block, ident):
    meta = {}
    for line in _stock_rows(stock_block, {"1302"}):
        parts = line.split(None, 2)
        if len(parts) >= 2:
            meta[parts[1]] = parts[2] if len(parts) > 2 else ""
    meta.pop("flatten", None)  # terrain_fit fits buildings to the unflattened terrain
    if len(ident) == 4 and ident.isalpha() and ident.isupper():
        meta["icao_code"] = ident
    meta["datum_lat"] = f"{layout.lat:.6f}"
    meta["datum_lon"] = f"{layout.lon:.6f}"
    if len(layout.region) == 2 and "region_code" not in meta:
        meta["region_code"] = layout.region
    return [f"1302 {k} {v}".rstrip() for k, v in meta.items()]


def _marking_code(flags):
    alternate = bool(flags & ((1 << 13) | (1 << 14) | (1 << 15)))
    if flags & (1 << 6):  # precision
        return 7 if alternate else 3
    if flags & ((1 << 3) | (1 << 2)):  # touchdown / fixed distance
        return 6 if alternate else 2
    if flags & ((1 << 1) | (1 << 5) | (1 << 0) | (1 << 4)):  # threshold / ident / edges / dashes
        return 1
    return 0


def _runway_rows(layout, transparent, report):
    rows, lights = [], []
    for rw in layout.runways:
        if not (rw.length_m > 1.0 and rw.width_m > 0.0):
            continue
        p, s = runway_ends(rw)
        if rw.surface == _WATER:
            rows.append(f"101 {rw.width_m:.2f} 1 {rw.primary.name} {p[0]:.8f} {p[1]:.8f} "
                        f"{rw.secondary.name} {s[0]:.8f} {s[1]:.8f}")
            report["water runways"] = report.get("water runways", 0) + 1
            continue
        surface = TRANSPARENT if transparent else _SURFACE.get(rw.surface, 1)
        markings = 0 if transparent else _marking_code(rw.marking_flags)
        shoulder = 0 if transparent or not (rw.marking_flags & (1 << 7)) else (2 if surface == 2 else 1)
        row = (f"100 {rw.width_m:.2f} {surface} {shoulder} 0.25 {1 if rw.centre_lights else 0} "
               f"{min(rw.edge_lights, 3)} 0")
        for end, pos in ((rw.primary, p), (rw.secondary, s)):
            row += (f" {end.name} {pos[0]:.8f} {pos[1]:.8f} {min(end.displaced_m, rw.length_m * 0.9):.2f} "
                    f"{max(end.blast_pad_m, end.overrun_m):.2f} {markings} {_ALS.get(end.approach_system, 0)} "
                    f"{1 if end.touchdown_lights else 0} {1 if end.reil else 0}")
        rows.append(row)
        report["runways"] = report.get("runways", 0) + 1

        # VASI/PAPI: one row per unit, beside the touchdown zone, facing the approach.
        for end, pos, landing in ((rw.primary, p, rw.heading_true), (rw.secondary, s, rw.heading_true + 180.0)):
            threshold = _move(pos[0], pos[1], landing, end.displaced_m)
            for v in end.vasi:
                if v.kind in (7, 8, 12, 13):
                    code = 2 if v.side == "L" else 3
                elif v.kind == 9:
                    code = 5
                elif 1 <= v.kind <= 11:
                    code = 1
                else:
                    continue
                along = v.bias_z if 30.0 <= v.bias_z <= 1500.0 else 300.0
                lateral = abs(v.bias_x) if rw.width_m / 2.0 < abs(v.bias_x) < 200.0 else rw.width_m / 2.0 + 15.0
                at = _move(*threshold, landing, along)
                at = _move(*at, landing + (-90.0 if v.side == "L" else 90.0), lateral)
                lights.append(f"21 {at[0]:.8f} {at[1]:.8f} {code} {landing % 360.0:.2f} {v.pitch:.2f} "
                              f"{end.name} {'PAPI' if code in (2, 3) else 'VASI'}")
                report["glideslope indicators"] = report.get("glideslope indicators", 0) + 1
    for i, h in enumerate(layout.helipads):
        if h.length_m <= 0 or h.width_m <= 0:
            continue
        surface = TRANSPARENT if (transparent or h.transparent) else _SURFACE.get(h.surface, 2)
        rows.append(f"102 H{i + 1} {h.lat:.8f} {h.lon:.8f} {h.heading % 360.0:.2f} {h.length_m:.2f} "
                    f"{h.width_m:.2f} {surface} 0 0 0.25 0")
        report["helipads"] = report.get("helipads", 0) + 1
    return rows, lights


def _linear_feature(name, points, line, light, closed=False):
    """One 120 feature. Node style tokens: the line code then the light
    code; a light-only node carries just the light code (the form this
    project already uses for lights kept from stock blocks)."""
    if len(points) < 2:
        return []
    rows = [f"120 {_text(name) or 'line'}"]
    styles = " ".join(str(c) for c in ((line, light) if line else (light,) if light else ()))
    for i, (lat, lon) in enumerate(points):
        last = i == len(points) - 1
        if last and not closed:
            rows.append(f"115 {lat:.8f} {lon:.8f}")
        else:
            code = 113 if (last and closed) else 111
            rows.append(f"{code} {lat:.8f} {lon:.8f} {styles}".rstrip())
    return rows


def _closed(vertices):
    if len(vertices) >= 4:
        a, b = vertices[0], vertices[-1]
        if abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9:
            return vertices[:-1], True
    return vertices, False


def _line_rows(layout, plane, painted_lines, report):
    rows = []
    for ls in layout.light_strings:
        pts, closed = _closed(ls.vertices)
        rows += _linear_feature(ls.name or "Taxiway lights", pts, 0, ls.light_type, closed)
        report["light strings"] = report.get("light strings", 0) + 1

    if painted_lines:
        for pl in layout.painted_lines:
            code = _PAINTED.get(pl.style)
            if code is None:
                continue
            pts, closed = _closed(list(pl.vertices))
            if pl.style in _PAINTED_REVERSED:
                pts = pts[::-1]
            rows += _linear_feature("Painted line", pts, code[0], code[1] if pl.lit else 0, closed)
            report["painted lines"] = report.get("painted lines", 0) + 1

    # No light strings: light the taxiways from the taxi path flags, the way
    # MSFS does procedurally for airports without explicit light strings.
    if not layout.light_strings:
        nodes = layout.taxi_nodes
        for p in layout.taxi_paths:
            if p.kind not in (1, 3, 4):
                continue
            a, b = p.start, layout.path_end_node(p)
            if a >= len(nodes) or b >= len(nodes) or a == b:
                continue
            (la, lo_a), (lb, lo_b) = nodes[a], nodes[b]
            if p.centre_lit:
                rows += _linear_feature(p.name or "Taxiway", [(la, lo_a), (lb, lo_b)], 0, 101)
                report["taxiway centre lights"] = report.get("taxiway centre lights", 0) + 1
            if p.left_edge_lit or p.right_edge_lit:
                xa, ya = plane.xy(la, lo_a)
                xb, yb = plane.xy(lb, lo_b)
                length = math.hypot(xb - xa, yb - ya)
                if length < 0.5:
                    continue
                half = max(p.width_m, 10.0) / 2.0
                nx, ny = -(yb - ya) / length * half, (xb - xa) / length * half
                for lit, sign in ((p.left_edge_lit, 1.0), (p.right_edge_lit, -1.0)):
                    if lit:
                        rows += _linear_feature(p.name or "Taxiway", [plane.latlon(xa + sign * nx, ya + sign * ny),
                                                                    plane.latlon(xb + sign * nx, yb + sign * ny)],
                                                0, 102)
                        report["taxiway edge lights"] = report.get("taxiway edge lights", 0) + 1
    return rows


def _pavement_rows(layout, report):
    rows = []
    for poly in layout.aprons:
        verts, _ = _closed(list(poly.vertices))
        if len(verts) < 3:
            continue
        rows.append(f"110 {TRANSPARENT} 0.25 0.00 Apron")
        for lat, lon in verts[:-1]:
            rows.append(f"111 {lat:.8f} {lon:.8f}")
        rows.append(f"113 {verts[-1][0]:.8f} {verts[-1][1]:.8f}")
        report["apron polygons"] = report.get("apron polygons", 0) + 1
    return rows


def _boundary_rows(layout, plane, margin_m=60.0):
    pts = []
    for rw in layout.runways:
        for lat, lon in runway_ends(rw):
            pts.append(plane.xy(lat, lon))
    for poly in layout.aprons:
        pts += [plane.xy(*v) for v in poly.vertices]
    pts += [plane.xy(p.lat, p.lon) for p in layout.parkings]
    pts += [plane.xy(h.lat, h.lon) for h in layout.helipads]
    hull = _convex_hull([(round(x, 2), round(y, 2)) for x, y in pts])
    if not hull:
        return []
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    ring = []
    for x, y in hull:
        d = max(math.hypot(x - cx, y - cy), 1e-9)
        ring.append(plane.latlon(x + (x - cx) / d * margin_m, y + (y - cy) / d * margin_m))
    rows = ["130 Airport Boundary"]
    for lat, lon in ring[:-1]:
        rows.append(f"111 {lat:.8f} {lon:.8f}")
    rows.append(f"113 {ring[-1][0]:.8f} {ring[-1][1]:.8f}")
    return rows


def _point_rows(layout, stock_block, kind, report):
    rows = []
    if layout.tower:
        lat, lon, alt = layout.tower
        height_ft = max((alt - layout.alt_m) * FEET_PER_M, 10.0)
        # draw 0: the package's own tower model comes across with the buildings.
        rows.append(f"14 {lat:.8f} {lon:.8f} {height_ft:.1f} 0 Tower")
    beacons = _stock_rows(stock_block, {"18"})
    rows += beacons
    if layout.windsocks:
        for lat, lon in layout.windsocks:
            rows.append(f"19 {lat:.8f} {lon:.8f} 1 Windsock")
        report["windsocks"] = len(layout.windsocks)
    else:
        rows += _stock_rows(stock_block, {"19"})
    for s in layout.signs:
        text = translate_sign(s.label)
        if not text:
            report["signs dropped (untranslatable)"] = report.get("signs dropped (untranslatable)", 0) + 1
            continue
        size = 1 if s.size <= 2 else 2 if s.size == 3 else 3
        rows.append(f"20 {s.lat:.8f} {s.lon:.8f} {s.heading % 360.0:.2f} 0 {size} {text}")
        report["taxiway signs"] = report.get("taxiway signs", 0) + 1
    return rows


def _frequency_rows(layout, stock_block, ident):
    rows = []
    for c in layout.coms:
        code = _FREQ_ROW.get(c.kind)
        if code and 108000 <= c.freq_khz <= 137000:
            rows.append(f"{code} {c.freq_khz} {_text(c.name) or ident}")
    if not rows:
        rows = _stock_rows(stock_block, {"50", "51", "52", "53", "54", "55", "56",
                                         "1050", "1051", "1052", "1053", "1054", "1055", "1056"})
    return rows


def _flow_rows(layout, stock_block):
    """ATC runway-use flows from the stock block, kept only when every
    runway they name exists in this airport (MSFS has no flows at all)."""
    rows = _stock_rows(stock_block, {"1000", "1001", "1002", "1003", "1004", "1100", "1101", "1110"})
    names = {e.name for rw in layout.runways for e in (rw.primary, rw.secondary)}
    for line in rows:
        parts = line.split()
        if parts[0] in ("1100", "1101") and len(parts) > 1 and parts[1] not in names:
            return []
    return rows


def _width_class_from_width(width_m):
    for limit, cls in ((10.5, "A"), (15.0, "B"), (18.0, "C"), (23.0, "D"), (25.0, "E")):
        if width_m < limit:
            return cls
    return "F"


def _width_class_from_span(span_m):
    for limit, cls in ((15.0, "A"), (24.0, "B"), (36.0, "C"), (52.0, "D"), (65.0, "E")):
        if span_m < limit:
            return cls
    return "F"


def _network_rows(layout, plane, stock_block, airport_name, report):
    nodes = layout.taxi_nodes
    n_nodes = len(nodes)
    ids = {}
    node_rows = []

    def node_id(i):
        if i not in ids:
            ids[i] = len(ids)
            lat, lon = nodes[i]
            node_rows.append(f"1201 {lat:.8f} {lon:.8f} both {ids[i]} n{ids[i]}")
        return ids[i]

    runway_of_end = {}
    runway_geo = []
    for k, rw in enumerate(layout.runways):
        if rw.surface == _WATER:
            continue
        p, s = runway_ends(rw)
        runway_geo.append((rw, plane.xy(*p), plane.xy(*s)))
        for e in (rw.primary, rw.secondary):
            runway_of_end[e.name] = len(runway_geo) - 1

    edges = []  # (a, b, kind_token, name, runway_index or None)
    truck = []
    seen = set()
    for p in layout.taxi_paths:
        a, b = p.start, layout.path_end_node(p)
        if a == b or a >= n_nodes or b >= n_nodes or not (p.is_aircraft_route or p.is_vehicle_route):
            continue
        xa, xb = plane.xy(*nodes[a]), plane.xy(*nodes[b])
        if math.hypot(xa[0] - xb[0], xa[1] - xb[1]) < 0.5:
            continue
        key = (min(a, b), max(a, b), p.is_vehicle_route)
        if key in seen:
            continue
        seen.add(key)
        if p.is_vehicle_route:
            truck.append((a, b, p.name))
        elif p.kind == 2:
            ri = runway_of_end.get(p.name)
            rname = f"{runway_geo[ri][0].primary.name}/{runway_geo[ri][0].secondary.name}" if ri is not None else p.name
            edges.append((a, b, "runway", rname, ri))
        else:
            edges.append((a, b, f"taxiway_{_width_class_from_width(p.width_m)}", p.name, None))
    if not edges and not truck:
        return []

    # Hot zones: from each runway's own edges, walk the network until a
    # hold-short node, staying within a band around the runway.
    adjacency = {}
    for e, (a, b, *_rest) in enumerate(edges):
        adjacency.setdefault(a, []).append(e)
        adjacency.setdefault(b, []).append(e)
    zones = [dict() for _ in edges]  # edge -> {"departure": set(), "arrival": set(), "ils": set()}
    n_points = len(layout.taxi_points)
    for ri, (rw, ra, rb) in enumerate(runway_geo):
        pair = f"{rw.primary.name},{rw.secondary.name}"
        queue = deque()
        visited_nodes = set()
        for e, edge in enumerate(edges):
            if edge[4] == ri:
                zones[e].setdefault("departure", set()).add(pair)
                zones[e].setdefault("arrival", set()).add(pair)
                for n in edge[:2]:
                    if n not in visited_nodes:
                        visited_nodes.add(n)
                        queue.append(n)
        visited_edges = set()
        while queue:
            n = queue.popleft()
            for e in adjacency.get(n, ()):
                if edges[e][4] is not None or e in visited_edges:
                    continue
                visited_edges.add(e)
                other = edges[e][1] if edges[e][0] == n else edges[e][0]
                if _dist_to_segment(plane.xy(*nodes[other]), ra, rb) > _HOT_ZONE_BAND_M:
                    continue
                zones[e].setdefault("departure", set()).add(pair)
                zones[e].setdefault("arrival", set()).add(pair)
                point = layout.taxi_points[other] if other < n_points else None
                if point is not None and point.is_hold_short:
                    if point.is_ils_hold:
                        zones[e].setdefault("ils", set()).add(pair)
                    continue
                if other not in visited_nodes:
                    visited_nodes.add(other)
                    queue.append(other)

    rows = []
    edge_rows = []
    hot = 0
    for e, (a, b, kind, name, _ri) in enumerate(edges):
        ia, ib = node_id(a), node_id(b)
        edge_rows.append(f"1202 {ia} {ib} twoway {kind} {_token(name, '')}".rstrip())
        if zones[e] and kind != "runway":
            hot += 1
        for phase in ("departure", "arrival", "ils"):
            if phase in zones[e]:
                edge_rows.append(f"1204 {phase} {','.join(sorted(zones[e][phase]))}")
    truck_rows = []
    for a, b, name in truck:
        truck_rows.append(f"1206 {node_id(a)} {node_id(b)} twoway {_token(name, '')}".rstrip())
    if not truck_rows and stock_block:
        # MSFS defined no vehicle roads: carry the stock ground-vehicle
        # topology over onto this network by nearest node.
        used = sorted(ids, key=ids.get)
        native_positions = [nodes[i] for i in used]
        for new_a, new_b, direction in apt_dat._remap_truck_edges(
                apt_dat._parse_stock_taxi_nodes(stock_block), apt_dat._parse_stock_truck_edges(stock_block),
                native_positions):
            truck_rows.append(f"1206 {new_a} {new_b} {direction}")
    rows.append(f"1200 {_text(airport_name)}".rstrip())
    rows += node_rows + edge_rows + truck_rows
    report["taxi network nodes"] = len(node_rows)
    report["taxi network edges"] = len(edges)
    report["hot-zone taxi edges"] = hot
    report["ground vehicle edges"] = len(truck_rows)
    return rows


def _ramp_rows(layout, report):
    rows = []
    names = {}
    for p in layout.parkings:
        if not is_aircraft_stand(p.kind):
            continue
        base = p.display_name or "Stand"
        names[base] = names.get(base, 0) + 1
        name = base if names[base] == 1 else f"{base}-{names[base]}"
        if p.kind in _GATES:
            location, ops = "gate", "airline"
        elif p.kind in _MILITARY:
            location, ops = "misc", "military"
        elif p.kind in _CARGO:
            location, ops = "misc", "cargo"
        else:
            location, ops = "tie_down", "general_aviation"
        r = p.radius_m
        if p.kind == 0x07:
            types = "fighters"
        elif p.kind == 0x0B:
            types = "props"
        elif r >= 26.0:
            types = "heavy|jets"
        elif r >= 16.0:
            types = "jets|turboprops"
        elif r >= 9.0:
            types = "turboprops|props"
        else:
            types = "props|helos"
        rows.append(f"1300 {p.lat:.8f} {p.lon:.8f} {p.heading % 360.0:.2f} {location} {types} {_text(name)}")
        airlines = " ".join(a.lower() for a in p.airlines if len(a) == 3 and a.isalpha())
        rows.append(f"1301 {_width_class_from_span(r * 2.0)} {ops} {airlines}".rstrip())
        report["ramp starts"] = report.get("ramp starts", 0) + 1
    return rows


def build_native_airport(layout, stock_block=None, runway_surface="transparent", painted_lines=False):
    """apt.dat rows for one airport from its decoded MSFS layout. Returns
    (rows, report) where report counts what was written, by kind."""
    report = {}
    plane = _Plane(layout.lat, layout.lon)
    rows, ident, kind = _header(layout, stock_block)
    rows += _metadata(layout, stock_block, ident)
    runway_rows, light_objects = _runway_rows(layout, runway_surface != "native", report)
    rows += runway_rows
    rows += _pavement_rows(layout, report)
    rows += _line_rows(layout, plane, painted_lines, report)
    rows += _boundary_rows(layout, plane)
    rows += _point_rows(layout, stock_block, kind, report)
    rows += light_objects
    rows += _frequency_rows(layout, stock_block, ident)
    rows += _flow_rows(layout, stock_block)
    rows += _network_rows(layout, plane, stock_block, layout.name or ident, report)
    rows += _ramp_rows(layout, report)
    return rows, report


def is_usable(layout):
    """X-Plane rejects an airport without a runway or helipad."""
    return layout is not None and bool(layout.runways or layout.helipads)
