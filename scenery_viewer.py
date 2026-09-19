#!/usr/bin/env python3
"""
Lightweight top-down debug viewer for a compiled X-Plane scenery pack.

Reads the pack's own compiled .dsf tiles (Earth nav data/**/*.dsf) and the
OBJ8 .obj files they reference directly -- no X-Plane install, no 3-D
renderer, no multi-minute sim load -- and renders a single self-contained
HTML page with a top-down SVG map: every placed object's real-world
position/heading, every draped (ground-marking/pavement) object's actual
footprint outline colored by its ATTR_layer_group_draped offset (so two
objects sharing an offset -- a z-fight/"ghost layer" risk -- visibly share
a color), and every DSF exclusion-zone rectangle. Built specifically to
let a fix (shifted pavement, draw-order collisions, exclusion coverage,
...) be checked in seconds instead of waiting for X-Plane to load the
scenery.

Reads exactly the atom/encoding conventions dsf_compiler.py itself writes
(see that module's own docstrings) -- this is a matching reader for OUR
OWN compiler's output, not a general-purpose third-party DSF parser.

Usage:
    python scenery_viewer.py <path to a Custom Scenery pack> [-o out.html]
"""
import argparse
import struct
import webbrowser
from pathlib import Path

_MAGIC = b"XPLNEDSF"

# Every atom FourCC is stored on disk as the reverse of its conventional
# name -- matches dsf_compiler.py's own pack_atom() calls and terrain_dem.py's
# read-side comment about the same convention.
_ATOM_HEAD = b"DAEH"
_ATOM_PROP = b"PORP"
_ATOM_DEFN = b"NFED"
_ATOM_OBJT = b"TJBO"
_ATOM_GEOD = b"DOEG"
_ATOM_POOL = b"LOOP"
_ATOM_SCAL = b"LACS"
_ATOM_CMDS = b"SDMC"

_AGL_HEIGHT_SCALE = 800.0
_AGL_HEIGHT_OFFSET = -400.0

_CMD_POOL_SELECT = 1
_CMD_SET_DEF16 = 4
_CMD_OBJECT = 7
_CMD_COMMENT8 = 32

_EXCLUDE_KEYS = {
    "sim/exclude_obj": "obj", "sim/exclude_agb": "agb", "sim/exclude_for": "for",
    "sim/exclude_fac": "fac", "sim/exclude_bea": "bea", "sim/exclude_lin": "lin",
    "sim/exclude_pol": "pol", "sim/exclude_net": "net", "sim/exclude_str": "str",
}

_EXCLUDE_COLORS = {
    "obj": "#f87171", "agb": "#fb923c", "for": "#4ade80", "fac": "#facc15",
    "bea": "#e879f9", "lin": "#38bdf8", "pol": "#a78bfa", "net": "#94a3b8", "str": "#f472b6",
}

# Distinct, high-contrast palette cycled by draped_layer_offset (-5..+5) --
# two draped objects sharing a color share an offset, which is exactly the
# X-Plane z-fight/"ghost layer" risk this tool exists to make visible.
_LAYER_PALETTE = [
    "#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e",
    "#14b8a6", "#06b6d4", "#3b82f6", "#8b5cf6", "#d946ef", "#f43f5e",
]


def _iter_atoms(data, start, end):
    pos = start
    while pos + 8 <= end:
        atom_id = data[pos:pos + 4]
        atom_len = struct.unpack_from("<I", data, pos + 4)[0]
        if atom_len < 8 or pos + atom_len > end:
            break
        yield atom_id, pos + 8, pos + atom_len
        pos += atom_len


def _parse_string_table(payload):
    if not payload:
        return []
    text = payload.decode("utf-8", errors="replace")
    if text.endswith("\x00"):
        text = text[:-1]
    return text.split("\x00") if text else []


def _decode_plane(data, pos, count):
    """Inverse of dsf_compiler._encode_plane_differenced: encType byte(3),
    then a run of (count_byte, value(s)) groups -- count_byte<=127 means
    that many individual raw values follow; count_byte>127 means ONE value
    follows, repeated (count_byte-128) times. The decoded stream is a
    cumulative-sum-mod-65536 of DIFFERENCES, first value raw."""
    enc_type = data[pos]
    pos += 1
    diffs = []
    while len(diffs) < count:
        count_byte = data[pos]
        pos += 1
        if count_byte <= 127:
            for _ in range(count_byte):
                diffs.append(struct.unpack_from("<H", data, pos)[0])
                pos += 2
        else:
            repeat = count_byte - 128
            v = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            diffs.extend([v] * repeat)
    raw = []
    acc = 0
    for i, d in enumerate(diffs):
        acc = d if i == 0 else (acc + d) % 65536
        raw.append(acc)
    return raw, pos


class DsfTile:
    def __init__(self):
        self.exclusions = []  # [{"category":.., "west":.., "south":.., "east":.., "north":..}]
        self.objects = []     # [{"name":.., "lon":.., "lat":.., "hdg":.., "agl":.., "is_agl":bool}]


def parse_dsf(path: Path):
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:8] != _MAGIC:
        return None
    tile = DsfTile()

    obj_names = []
    pools = {}  # pool_index -> {"lon":[...], "lat":[...], "hdg":[...], "agl":[...] or None}

    for atom_id, body_start, body_end in _iter_atoms(raw, 12, len(raw)):
        if atom_id == _ATOM_HEAD:
            for sub_id, s, e in _iter_atoms(raw, body_start, body_end):
                if sub_id == _ATOM_PROP:
                    props = _parse_string_table(raw[s:e])
                    for i in range(0, len(props) - 1, 2):
                        key, value = props[i], props[i + 1]
                        cat = _EXCLUDE_KEYS.get(key)
                        if cat:
                            # X-Plane exclusion values are "/"-delimited; also
                            # accept whitespace so pre-2026-09-02 DSFs still read.
                            parts = value.replace("/", " ").split()
                            if len(parts) == 4:
                                w, s2, e2, n = (float(p) for p in parts)
                                tile.exclusions.append(
                                    {"category": cat, "west": w, "south": s2, "east": e2, "north": n})
        elif atom_id == _ATOM_DEFN:
            for sub_id, s, e in _iter_atoms(raw, body_start, body_end):
                if sub_id == _ATOM_OBJT:
                    obj_names = _parse_string_table(raw[s:e])
        elif atom_id == _ATOM_GEOD:
            pool_idx = 0
            pos = body_start
            while pos < body_end:
                # Each pool is one LOOP (POOL) atom immediately followed by
                # one LACS (SCAL) atom -- both top-level children of GEOD.
                atom_id2, s2, e2 = next(_iter_atoms(raw, pos, body_end))
                if atom_id2 != _ATOM_POOL:
                    break
                n, num_planes = struct.unpack_from("<IB", raw, s2)
                ppos = s2 + 5
                planes_raw = []
                for _ in range(num_planes):
                    values, ppos = _decode_plane(raw, ppos, n)
                    planes_raw.append(values)
                pos = e2

                atom_id3, s3, e3 = next(_iter_atoms(raw, pos, body_end))
                scal_pairs = []
                if atom_id3 == _ATOM_SCAL:
                    sp = s3
                    for _ in range(num_planes):
                        scale, offset = struct.unpack_from("<ff", raw, sp)
                        scal_pairs.append((scale, offset))
                        sp += 8
                    pos = e3

                def _decode(plane_idx):
                    scale, offset = scal_pairs[plane_idx]
                    return [v / 65535.0 * scale + offset for v in planes_raw[plane_idx]]

                pool = {"lon": _decode(0), "lat": _decode(1), "hdg": _decode(2), "agl": None}
                if num_planes >= 4:
                    pool["agl"] = _decode(3)
                pools[pool_idx] = pool
                pool_idx += 1
        elif atom_id == _ATOM_CMDS:
            pos = body_start
            current_pool = 0
            current_def = None
            agl_mode = False
            while pos < body_end:
                opcode = raw[pos]
                if opcode == _CMD_POOL_SELECT:
                    current_pool = struct.unpack_from("<H", raw, pos + 1)[0]
                    pos += 3
                elif opcode == _CMD_SET_DEF16:
                    current_def = struct.unpack_from("<H", raw, pos + 1)[0]
                    pos += 3
                elif opcode == _CMD_OBJECT:
                    i = struct.unpack_from("<H", raw, pos + 1)[0]
                    pos += 3
                    pool = pools.get(current_pool)
                    if pool is not None and current_def is not None and current_def < len(obj_names):
                        entry = {
                            "name": obj_names[current_def],
                            "lon": pool["lon"][i], "lat": pool["lat"][i], "hdg": pool["hdg"][i],
                            "agl": pool["agl"][i] if pool["agl"] else 0.0,
                            "is_agl": agl_mode and pool["agl"] is not None,
                        }
                        tile.objects.append(entry)
                elif opcode == _CMD_COMMENT8:
                    length = raw[pos + 1]
                    payload = raw[pos + 2: pos + 2 + length]
                    if length >= 2:
                        comment_type = struct.unpack_from("<H", payload, 0)[0]
                        if comment_type == 2 and length >= 6:  # dsf_Comment_AGL
                            want_agl = struct.unpack_from("<i", payload, 2)[0]
                            agl_mode = bool(want_agl)
                    pos += 2 + length
                else:
                    # Unknown/unhandled opcode for this reader's own limited
                    # command vocabulary (this tool only ever needs to read
                    # what dsf_compiler.py itself writes) -- stop rather
                    # than risk misparsing the rest of the stream.
                    break
    return tile


def parse_obj8(path: Path):
    """Returns {"positions": [(x,y,z), ...], "draped": bool, "tilted": bool,
    "layer_offset": int or None, "texture": str, "lights": [(x,y,z), ...]}
    from a real OBJ8 .obj this pack's own dsf_compiler/mesh_convert wrote."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    positions = []
    lights = []
    draped = False
    tilted = False
    layer_offset = None
    texture = ""
    for line in text.splitlines():
        if line.startswith("VT "):
            parts = line.split()
            positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("LIGHT_SPILL_CUSTOM"):
            parts = line.split()
            lights.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("ATTR_draped"):
            draped = True
        elif line.startswith("ATTR_layer_group_draped"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    layer_offset = int(parts[2])
                except ValueError:
                    pass
        elif line == "TILTED":
            tilted = True
        elif line.startswith("TEXTURE "):
            texture = line[len("TEXTURE "):].strip()
    return {
        "positions": positions, "draped": draped, "tilted": tilted,
        "layer_offset": layer_offset, "texture": texture, "lights": lights,
    }


def _local_xz_to_latlon(lat0, lon0, heading_deg, x, z):
    import math
    hdg = math.radians(heading_deg)
    rx = x * math.cos(hdg) - z * math.sin(hdg)
    rz = x * math.sin(hdg) + z * math.cos(hdg)
    lat = lat0 - (rz / 111139.0)
    lon = lon0 + (rx / (111139.0 * math.cos(math.radians(lat0))))
    return lat, lon


def build_scene(pack_dir: Path):
    nav_dir = pack_dir / "Earth nav data"
    dsf_paths = sorted(nav_dir.rglob("*.dsf")) if nav_dir.is_dir() else []
    objects_dir = pack_dir / "objects"
    obj_cache = {}

    def _get_obj(name):
        if name.endswith(".obj"):
            name_no_ext = name[:-4]
        else:
            name_no_ext = name
        if name not in obj_cache:
            candidate = objects_dir / Path(name).name if name.startswith("objects/") else pack_dir / name
            obj_cache[name] = parse_obj8(candidate) if candidate.is_file() else None
        return obj_cache[name]

    scene = {"exclusions": [], "rigid": [], "draped": [], "library": [], "lights": [], "tile_count": 0}

    for dsf_path in dsf_paths:
        tile = parse_dsf(dsf_path)
        if tile is None:
            continue
        scene["tile_count"] += 1
        scene["exclusions"].extend(tile.exclusions)

        for obj in tile.objects:
            name = obj["name"]
            if not name.startswith("objects/"):
                scene["library"].append({
                    "lat": obj["lat"], "lon": obj["lon"], "hdg": obj["hdg"], "path": name,
                })
                continue
            ir = _get_obj(name)
            if ir is None:
                scene["library"].append({
                    "lat": obj["lat"], "lon": obj["lon"], "hdg": obj["hdg"], "path": name + " (unreadable)",
                })
                continue

            for lx, ly, lz in ir["lights"]:
                lat, lon = _local_xz_to_latlon(obj["lat"], obj["lon"], obj["hdg"], lx, lz)
                scene["lights"].append({"lat": lat, "lon": lon})

            if ir["draped"] and ir["positions"]:
                ring = [
                    _local_xz_to_latlon(obj["lat"], obj["lon"], obj["hdg"], px, pz)
                    for px, py, pz in ir["positions"]
                ]
                scene["draped"].append({
                    "name": name, "ring": ring, "layer_offset": ir["layer_offset"] or 0,
                    "texture": ir["texture"],
                })
            else:
                scene["rigid"].append({
                    "name": name, "lat": obj["lat"], "lon": obj["lon"], "hdg": obj["hdg"],
                    "tilted": ir["tilted"], "agl": obj["agl"], "is_agl": obj["is_agl"],
                })

    return scene


def _project(lat, lon, bounds, px_w, px_h, pad=20):
    import math
    west, south, east, north = bounds
    mid_lat = (south + north) / 2.0
    lon_scale = math.cos(math.radians(mid_lat))
    w = (east - west) * lon_scale or 1e-9
    h = (north - south) or 1e-9
    x = pad + ((lon - west) * lon_scale / w) * (px_w - 2 * pad)
    y = pad + (1.0 - (lat - south) / h) * (px_h - 2 * pad)
    return x, y


def render_html(scene, out_path: Path, title="Scenery Viewer"):
    all_lats, all_lons = [], []
    for coll, latf, lonf in (
        (scene["exclusions"], lambda e: (e["south"], e["north"]), lambda e: (e["west"], e["east"])),
    ):
        for e in coll:
            all_lats += [e["south"], e["north"]]
            all_lons += [e["west"], e["east"]]
    for r in scene["rigid"]:
        all_lats.append(r["lat"]); all_lons.append(r["lon"])
    for d in scene["draped"]:
        for lat, lon in d["ring"]:
            all_lats.append(lat); all_lons.append(lon)
    for lib in scene["library"]:
        all_lats.append(lib["lat"]); all_lons.append(lib["lon"])
    for lt in scene["lights"]:
        all_lats.append(lt["lat"]); all_lons.append(lt["lon"])

    if not all_lats:
        bounds = (-1, -1, 1, 1)
    else:
        bounds = (min(all_lons), min(all_lats), max(all_lons), max(all_lats))

    px_w, px_h = 1600, 1000
    svg_parts = []

    for e in scene["exclusions"]:
        x1, y1 = _project(e["north"], e["west"], bounds, px_w, px_h)
        x2, y2 = _project(e["south"], e["east"], bounds, px_w, px_h)
        color = _EXCLUDE_COLORS.get(e["category"], "#999999")
        svg_parts.append(
            f'<rect x="{min(x1,x2):.1f}" y="{min(y1,y2):.1f}" width="{abs(x2-x1):.1f}" height="{abs(y2-y1):.1f}" '
            f'fill="{color}" fill-opacity="0.08" stroke="{color}" stroke-width="1" stroke-dasharray="4,3">'
            f'<title>exclude_{e["category"]}</title></rect>'
        )

    for d in scene["draped"]:
        pts = " ".join(f"{_project(lat, lon, bounds, px_w, px_h)[0]:.1f},{_project(lat, lon, bounds, px_w, px_h)[1]:.1f}"
                        for lat, lon in d["ring"])
        color = _LAYER_PALETTE[(d["layer_offset"] + 5) % len(_LAYER_PALETTE)]
        tex_name = Path(d["texture"]).name if d["texture"] else "?"
        svg_parts.append(
            f'<polygon points="{pts}" fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="1">'
            f'<title>{d["name"]} | layer_offset={d["layer_offset"]} | texture={tex_name}</title></polygon>'
        )

    for r in scene["rigid"]:
        x, y = _project(r["lat"], r["lon"], bounds, px_w, px_h)
        import math
        hdg = math.radians(r["hdg"])
        ax, ay = x + 8 * math.sin(hdg), y - 8 * math.cos(hdg)
        color = "#fb7185" if r["tilted"] else "#60a5fa"
        agl_note = f' | AGL={r["agl"]:.2f}m' if r["is_agl"] else ""
        svg_parts.append(
            f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="{color}" stroke-width="1.5"/>'
            f'<title>{r["name"]} | hdg={r["hdg"]:.1f} | {"TILTED" if r["tilted"] else "rigid"}{agl_note}</title></g>'
        )

    for lib in scene["library"]:
        x, y = _project(lib["lat"], lib["lon"], bounds, px_w, px_h)
        svg_parts.append(
            f'<rect x="{x-3:.1f}" y="{y-3:.1f}" width="6" height="6" fill="#facc15" stroke="#78350f" stroke-width="0.75">'
            f'<title>{lib["path"]}</title></rect>'
        )

    for lt in scene["lights"]:
        x, y = _project(lt["lat"], lt["lon"], bounds, px_w, px_h)
        svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.5" fill="#fde68a" fill-opacity="0.9"/>')

    legend_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
        f'<span style="width:12px;height:12px;background:{c};display:inline-block;border-radius:2px;"></span>'
        f'<span>exclude_{cat}</span></div>'
        for cat, c in _EXCLUDE_COLORS.items()
    )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
* {{ box-sizing: border-box; }}
html, body {{ height:100%; margin:0; }}
body {{ background:#141414; color:#e5e5e5; font-family:'Segoe UI',sans-serif; padding:16px; display:flex; flex-direction:column; }}
h1 {{ font-size:16px; color:#3b82f6; margin:0 0 4px; flex:none; }}
.stats {{ color:#a0a0a0; font-size:12px; margin-bottom:8px; flex:none; }}
.hint {{ color:#666; font-size:11px; margin-bottom:8px; flex:none; }}
.layout {{ display:flex; gap:16px; align-items:stretch; flex:1; min-height:0; }}
.canvas-wrap {{ flex:1; min-width:0; background:#0d0d0f; border:1px solid #2a2a2e; border-radius:8px; overflow:hidden; position:relative; cursor:grab; }}
.canvas-wrap.dragging {{ cursor:grabbing; }}
.canvas-wrap svg {{ width:100%; height:100%; display:block; }}
.legend {{ background:#1c1c1e; border:1px solid #2a2a2e; border-radius:8px; padding:12px; font-size:12px; min-width:210px; max-width:210px; flex:none; overflow-y:auto; }}
.legend h2 {{ font-size:12px; color:#3b82f6; margin:0 0 8px; }}
.legend-item {{ display:flex; align-items:center; gap:6px; margin:3px 0; }}
.dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
</style></head>
<body>
<h1>{title}</h1>
<div class="stats">
  {scene["tile_count"]} DSF tile(s) &middot;
  {len(scene["rigid"])} rigid object(s) &middot;
  {len(scene["draped"])} draped/marking object(s) &middot;
  {len(scene["library"])} library-substituted object(s) &middot;
  {len(scene["lights"])} point light(s) &middot;
  {len(scene["exclusions"])} exclusion rectangle(s)
</div>
<div class="hint">Scroll/wheel to zoom, drag to pan, double-click to reset. Hover any shape for details.</div>
<div class="layout">
<div class="canvas-wrap" id="canvasWrap">
<svg id="mainSvg" viewBox="0 0 {px_w} {px_h}">
<g id="panZoomGroup">
{''.join(svg_parts)}
</g>
</svg>
</div>
<div class="legend">
  <h2>Exclusion categories</h2>
  {legend_rows}
  <h2 style="margin-top:14px;">Objects</h2>
  <div class="legend-item"><span class="dot" style="background:#60a5fa;"></span>rigid object (heading arrow)</div>
  <div class="legend-item"><span class="dot" style="background:#fb7185;"></span>TILTED object</div>
  <div class="legend-item"><span style="width:10px;height:10px;background:#facc15;display:inline-block;"></span>library-substituted object</div>
  <div class="legend-item"><span class="dot" style="background:#fde68a;"></span>point light</div>
  <h2 style="margin-top:14px;">Draped layers</h2>
  <div style="color:#a0a0a0;">Filled color = draped_layer_offset.<br>Two shapes sharing a color occupy the same
  X-Plane draw-order slot -- a real z-fight/"ghost layer" risk if they also overlap.</div>
</div>
</div>
<script>
(function() {{
  var wrap = document.getElementById('canvasWrap');
  var svg = document.getElementById('mainSvg');
  var group = document.getElementById('panZoomGroup');
  var viewBox = {{x: 0, y: 0, w: {px_w}, h: {px_h}}};
  var initial = {{x: 0, y: 0, w: {px_w}, h: {px_h}}};

  function applyViewBox() {{
    svg.setAttribute('viewBox', viewBox.x + ' ' + viewBox.y + ' ' + viewBox.w + ' ' + viewBox.h);
  }}

  wrap.addEventListener('wheel', function(e) {{
    e.preventDefault();
    var rect = wrap.getBoundingClientRect();
    var mx = viewBox.x + (e.clientX - rect.left) / rect.width * viewBox.w;
    var my = viewBox.y + (e.clientY - rect.top) / rect.height * viewBox.h;
    var factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    viewBox.w *= factor;
    viewBox.h *= factor;
    viewBox.x = mx - (mx - viewBox.x) * factor;
    viewBox.y = my - (my - viewBox.y) * factor;
    applyViewBox();
  }}, {{passive: false}});

  var dragging = false, lastX = 0, lastY = 0;
  wrap.addEventListener('mousedown', function(e) {{
    dragging = true;
    lastX = e.clientX; lastY = e.clientY;
    wrap.classList.add('dragging');
  }});
  window.addEventListener('mousemove', function(e) {{
    if (!dragging) return;
    var rect = wrap.getBoundingClientRect();
    var dx = (e.clientX - lastX) / rect.width * viewBox.w;
    var dy = (e.clientY - lastY) / rect.height * viewBox.h;
    viewBox.x -= dx; viewBox.y -= dy;
    lastX = e.clientX; lastY = e.clientY;
    applyViewBox();
  }});
  window.addEventListener('mouseup', function() {{
    dragging = false;
    wrap.classList.remove('dragging');
  }});
  wrap.addEventListener('dblclick', function() {{
    viewBox = {{x: initial.x, y: initial.y, w: initial.w, h: initial.h}};
    applyViewBox();
  }});
}})();
</script>
</body></html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack_dir", type=Path, help="Custom Scenery pack directory (contains 'Earth nav data/')")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output HTML path (default: <pack>/_viewer.html)")
    ap.add_argument("--open", action="store_true", help="Open the result in the default browser when done")
    args = ap.parse_args()

    pack_dir = args.pack_dir.resolve()
    out_path = args.output or (pack_dir / "_viewer.html")

    scene = build_scene(pack_dir)
    render_html(scene, out_path, title=f"Scenery Viewer -- {pack_dir.name}")

    print(f"Parsed {scene['tile_count']} DSF tile(s) under {pack_dir}")
    print(f"  rigid objects:    {len(scene['rigid'])}")
    print(f"  draped objects:   {len(scene['draped'])}")
    print(f"  library objects:  {len(scene['library'])}")
    print(f"  point lights:     {len(scene['lights'])}")
    print(f"  exclusion rects:  {len(scene['exclusions'])}")
    print(f"Wrote {out_path}")

    if args.open:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
