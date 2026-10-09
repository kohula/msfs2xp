"""Working jetways that keep the airport's own look.

An MSFS jetway is a rigged model: its ModelInfo XML names an IK chain from a
fixed point on the rotunda to the middle of the cab (IK_MainHandle, e.g.
Rotation_Base .. Pivot) and limits how each node of the chain may move
(IKConstraint): the rotunda turns (Heading), the tunnel tips (Bank/Pitch),
the telescoping sections slide along their own X axis between min and max,
the cab turns (Heading). A second chain locks the wheels to the ground
(IK_WheelsGroundLock): a leg that extends (X) and a bogie that steers. The
model's rest pose is the parked jetway; every bone runs along its parent's
local +X.

X-Plane 12 drives a jetway from one apt.dat row (1500: rotunda position,
heading, style, tunnel size, parked tunnel heading/length and cab heading)
and lets the scenery replace its art with an object of its own (a 1501 row
under it). That object stands with its origin on the rotunda, the parked
tunnel along -Z, and moves its parts with X-Plane's jetway datarefs:

    jw_base_rotation      the whole jetway about the rotunda (deg, + clockwise)
    jw_tunnel_pitch       the tunnel about its pitch pivot (deg, + tip down)
    jw_tunnel_extension   tunnel length, rotunda to cab (m)
    jw_cabin_rotation     the cab about its pivot (deg, + clockwise)
    jw_bogie_elevation    the wheel leg up/down (m)
    jw_bogie_rotation     the bogie steering (deg)

This module splits the MSFS model into those moving parts by its node tree
(a skinned mesh goes, per triangle, with the bone most of its vertices
follow), turns it into X-Plane's frame (the same 180-degree turn every
converted model gets), puts every material on one texture sheet (X-Plane
draws one object per jetway, with one texture) and writes the object with
nested animations. What a 1500 row needs is returned alongside.

A model whose rig can't be read (no IK_MainHandle chain, no telescoping
section) is left to the ordinary static conversion.
"""

import importlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import geo_transform

# the module (the package's own `convert` name is the function)
mc = importlib.import_module("mesh_convert.convert")

DATAREF = "sim/graphics/animation/jetways/"
# Tunnel reach of X-Plane's jetway size codes 0-3 (shortest, longest), m.
TUNNEL_SIZES = ((11.0, 23.0), (14.0, 29.0), (17.0, 38.0), (20.0, 47.0))
# Added to a 1500 row's size code: that jetway docks to the aircraft's
# second door.
DOOR_2 = 10
ATLAS_MAX = 4096
_CELL = 512
_PAD_PX = 2
# The flip every converted model gets (convert() is called with yaw 180):
# MSFS forward (+Z) becomes X-Plane north (-Z).
_TURN = np.diag([-1.0, 1.0, -1.0])

_CHAIN_RE = re.compile(r"<IKChain\b([^>]*)>(.*?)</IKChain>", re.IGNORECASE | re.DOTALL)
_CONSTRAINT_RE = re.compile(r"<IKConstraint\b[^>]*>(.*?)</IKConstraint>", re.IGNORECASE | re.DOTALL)
_TAG_TEXT_RE = r"<{0}\b[^>]*>\s*([^<]*?)\s*</{0}>"
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_AXIS_RE = re.compile(r"<(X|Y|Z|Heading|Pitch|Bank)\b([^>]*)/?>", re.IGNORECASE)


@dataclass
class Constraint:
    axis: str            # "x", "heading", "bank", ...
    lo: float = None
    hi: float = None


def read_ik(xml_text):
    """({chain name: (start, end)}, {node name lower-case: Constraint})
    from a ModelInfo XML."""
    chains = {}
    for m in _CHAIN_RE.finditer(xml_text or ""):
        attrs = {k.lower(): v for k, v in _ATTR_RE.findall(m.group(1))}
        body = m.group(2)
        start = re.search(_TAG_TEXT_RE.format("Start"), body, re.IGNORECASE)
        end = re.search(_TAG_TEXT_RE.format("End"), body, re.IGNORECASE)
        name = attrs.get("name", "")
        if start and end and name:
            chains[name] = (start.group(1), end.group(1))
    constraints = {}
    for m in _CONSTRAINT_RE.finditer(xml_text or ""):
        body = m.group(1)
        node = re.search(_TAG_TEXT_RE.format("Node"), body, re.IGNORECASE)
        if not node:
            continue
        axis = _AXIS_RE.search(body)
        if not axis:
            constraints[node.group(1).lower()] = Constraint("none")
            continue
        attrs = {k.lower(): v for k, v in _ATTR_RE.findall(axis.group(2))}

        def num(k):
            try:
                return float(attrs[k])
            except (KeyError, ValueError):
                return None
        constraints[node.group(1).lower()] = Constraint(axis.group(1).lower(), num("min"), num("max"))
    return chains, constraints


def is_jetway_xml(xml_text):
    chains, _ = read_ik(xml_text)
    return any(name.lower() == "ik_mainhandle" for name in chains)


@dataclass
class Part:
    kind: str            # base, pitch, ext, cab, leg, bogie
    node: int
    parent: int = None   # index into Rig.parts
    # filled in by measuring
    pivot: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (0.0, 0.0, -1.0)
    share: float = 0.0


@dataclass
class Jetway:
    """A converted jetway: the object written and what its 1500 row needs,
    in the converted model's own frame (x east, z south at heading 0)."""
    obj_name: str
    rotunda_xz: tuple        # where the rotunda stands in the model's frame
    tunnel_bearing: float    # parked tunnel direction, deg clockwise from the model's -Z
    parked_length: float
    shortest: float
    longest: float
    size: int
    cab_offset: float        # parked cab heading minus tunnel heading, deg
    style: int
    parts: list = field(default_factory=list)


def _bearing(dx, dz):
    return math.degrees(math.atan2(dx, -dz))


def _node_parents(nodes):
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []) or []:
            if isinstance(c, int) and 0 <= c < len(nodes):
                parent[c] = i
    return parent


def _descendants(nodes, i):
    out, stack = [], list(nodes[i].get("children", []) or [])
    while stack and len(out) <= len(nodes):
        n = stack.pop()
        out.append(n)
        stack.extend(nodes[n].get("children", []) or [])
    return out


def _find(names, wanted):
    return names.get((wanted or "").strip().lower())


def rig_parts(gltf, xml_text):
    """The moving parts of a jetway's node tree (base first, each part's
    parent part before it), or None when the rig doesn't read."""
    chains, constraints = read_ik(xml_text)
    nodes = gltf.get("nodes", [])
    names = {}
    for i, n in enumerate(nodes):
        names.setdefault(str(n.get("name", "")).lower(), i)
    main = next((v for k, v in chains.items() if k.lower() == "ik_mainhandle"), None)
    if main is None:
        return None
    start, end = _find(names, main[0]), _find(names, main[1])
    if start is None or end is None:
        return None
    parent = _node_parents(nodes)
    path = [end]
    while path[-1] != start:
        p = parent.get(path[-1])
        if p is None or len(path) > len(nodes):
            return None
        path.append(p)
    path.reverse()

    def axis_of(i):
        c = constraints.get(str(nodes[i].get("name", "")).lower())
        return c.axis if c else None

    exts = [i for i in path[1:] if axis_of(i) == "x"]
    if not exts:
        return None
    pitch = next((i for i in path[1:] if axis_of(i) in ("bank", "pitch", "z")), None)
    cab = next((i for i in [end] + _descendants(nodes, end) if axis_of(i) == "heading"), end)
    chosen = [("base", start)]
    if pitch is not None and pitch not in exts:
        chosen.append(("pitch", pitch))
    chosen += [("ext", i) for i in exts]
    if cab not in exts and cab != start:
        chosen.append(("cab", cab))
    wheels = next((v for k, v in chains.items() if "wheel" in k.lower()), None)
    if wheels is not None:
        w_start, w_end = _find(names, wheels[0]), _find(names, wheels[1])
        if w_start is not None and w_end is not None:
            w_path = [w_end]
            while w_path[-1] != w_start and parent.get(w_path[-1]) is not None and len(w_path) <= len(nodes):
                w_path.append(parent[w_path[-1]])
            leg = next((i for i in reversed(w_path) if axis_of(i) == "x"), None)
            if leg is not None:
                chosen.append(("leg", leg))
            below = [w_end] + _descendants(nodes, w_end)
            bogie = next((i for i in below if "orient" in str(nodes[i].get("name", "")).lower()), None)
            if bogie is not None:
                chosen.append(("bogie", bogie))
    # each part's parent part: its nearest ancestor that is a part
    by_node = {node: k for k, (_, node) in enumerate(chosen)}
    parts = []
    for kind, node in chosen:
        up, owner = parent.get(node), None
        while up is not None:
            if up in by_node and by_node[up] < len(parts):
                owner = by_node[up]
                break
            up = parent.get(up)
        parts.append(Part(kind, node, owner))
    return parts


def _part_of_nodes(gltf, parts):
    """{node: part index} -- every node below a part (and not below a
    deeper part) rides on it."""
    nodes = gltf.get("nodes", [])
    owner = {}
    order = sorted(range(len(parts)), key=lambda k: len(_ancestors(nodes, parts[k].node)))
    for k in order:  # shallow parts first, deeper ones overwrite their subtrees
        owner[parts[k].node] = k
        for d in _descendants(nodes, parts[k].node):
            owner[d] = k
    return owner


def _ancestors(nodes, i):
    parent = _node_parents(nodes)
    out = []
    while i in parent and len(out) <= len(nodes):
        i = parent[i]
        out.append(i)
    return out


def _tunnel_size(shortest, longest):
    best, best_overlap = 0, -1.0
    for code, (lo, hi) in enumerate(TUNNEL_SIZES):
        overlap = max(0.0, min(longest, hi) - max(shortest, lo))
        if overlap > best_overlap:
            best, best_overlap = code, overlap
    return best


def _style(model_name):
    return 1 if "glass" in model_name.lower() else 0


# --- geometry ---------------------------------------------------------------

@dataclass
class _Tri:
    part: int            # -1: static
    material: int
    verts: list          # 3 x (pos3, normal3, uv2)


def _primitive_triangles(gltf, buffers, node_idx, world, joints_world, part_of, prim):
    """Triangles of one primitive in the model's world frame (before the
    180-degree turn): [(part, [(pos, normal, uv) x3])]."""
    attrs = prim.get("attributes", {})
    if "POSITION" not in attrs:
        return []
    pos = mc.read_accessor(gltf, buffers, attrs["POSITION"])[:, :3].astype(np.float64)
    nrm = mc.read_accessor(gltf, buffers, attrs["NORMAL"])[:, :3].astype(np.float64) if "NORMAL" in attrs else None
    mat = (gltf.get("materials") or [{}])[prim["material"]] if prim.get("material") is not None else {}
    tex_coord = ((mat.get("pbrMetallicRoughness") or {}).get("baseColorTexture") or {}).get("texCoord", 0)
    uv_acc = attrs.get(f"TEXCOORD_{tex_coord}", attrs.get("TEXCOORD_0"))
    uv = mc.read_accessor(gltf, buffers, uv_acc, force_normalized=True)[:, :2].astype(np.float64) \
        if uv_acc is not None else np.zeros((len(pos), 2))
    if "indices" in prim:
        idx = mc.read_accessor(gltf, buffers, prim["indices"]).astype(np.int64).reshape(-1)
    else:
        idx = np.arange(len(pos), dtype=np.int64)
    extras = (prim.get("extras") or {}).get("ASOBO_primitive", {})
    start = extras.get("StartIndex", 0)
    count = extras["PrimitiveCount"] * 3 if "PrimitiveCount" in extras else len(idx) - start
    idx = idx[start:start + count] + extras.get("BaseVertexIndex", 0)
    idx = idx[: len(idx) // 3 * 3]
    if not len(idx) or idx.max() >= len(pos) or (uv is not None and idx.max() >= len(uv)):
        return []

    # per vertex: the matrix placing it and the part it rides on
    if joints_world is not None and "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
        j = mc.read_accessor(gltf, buffers, attrs["JOINTS_0"]).astype(np.int64)
        w = mc.read_accessor(gltf, buffers, attrs["WEIGHTS_0"], force_normalized=True).astype(np.float64)
        dominant = j[np.arange(len(j)), np.argmax(w, axis=1)]
        mats = [joints_world[d][0] if 0 <= d < len(joints_world) else world for d in dominant]
        vparts = [joints_world[d][1] if 0 <= d < len(joints_world) else part_of.get(node_idx, -1)
                  for d in dominant]
    else:
        mats = None
        vparts = None
    node_part = part_of.get(node_idx, -1)

    def place(i):
        m = mats[i] if mats is not None else world
        p = (m @ np.append(pos[i], 1.0))[:3]
        n = (m[:3, :3] @ nrm[i]) if nrm is not None else np.array([0.0, 1.0, 0.0])
        ln = np.linalg.norm(n)
        return p, (n / ln if ln > 1e-12 else np.array([0.0, 1.0, 0.0])), uv[i]

    out = []
    for t in range(0, len(idx), 3):
        a, b, c = idx[t], idx[t + 1], idx[t + 2]
        if vparts is not None:
            votes = [vparts[a], vparts[b], vparts[c]]
            part = max(set(votes), key=votes.count)
        else:
            part = node_part
        out.append((part, [place(a), place(b), place(c)]))
    return out


def _joints_world(gltf, buffers, node, world_of, part_of):
    """Per skin joint: (matrix placing a bind-pose vertex at rest, part)."""
    skins = gltf.get("skins", [])
    s = node.get("skin")
    if s is None or not (0 <= s < len(skins)):
        return None
    skin = skins[s]
    joints = skin.get("joints", [])
    ibm = None
    if skin.get("inverseBindMatrices") is not None:
        flat = mc.read_accessor(gltf, buffers, skin["inverseBindMatrices"]).reshape(-1, 16)
        ibm = [m.reshape(4, 4).T for m in flat]
    out = []
    for k, jn in enumerate(joints):
        inv = ibm[k] if ibm is not None and k < len(ibm) else np.eye(4)
        out.append((world_of.get(jn, np.eye(4)) @ inv, part_of.get(jn, -1)))
    return out


# --- the object ---------------------------------------------------------------

def build(model_path, xml_text, objects_dir, textures_dir, external_textures, obj_name, glass_opacity=None):
    """Writes objects_dir/<obj_name>.obj (+ its texture sheets) for the
    jetway model at model_path, or returns None when its rig doesn't
    read."""
    model_path = Path(model_path)
    try:
        gltf, bin_chunk = mc.parse_model_file(model_path)
        buffers = mc.load_buffers(gltf, bin_chunk, model_path)
    except Exception:
        return None
    parts = rig_parts(gltf, xml_text)
    if not parts:
        return None
    nodes = gltf.get("nodes", [])
    animations = mc.read_gltf_animations(gltf, buffers)
    world_of = mc.collect_world_transforms(gltf, animations)
    part_of = _part_of_nodes(gltf, parts)

    def wpos(i):
        return _TURN @ (world_of.get(i, np.eye(4)) @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]

    def wdir(i, local):
        m = world_of.get(i, np.eye(4))[:3, :3]
        v = _TURN @ (m @ np.asarray(local, dtype=np.float64))
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    parent = _node_parents(nodes)
    base = parts[0]
    rotunda = wpos(base.node)
    exts = [p for p in parts if p.kind == "ext"]
    # bones run along their parent's +X: the parked tunnel's direction
    tunnel = wdir(parent.get(exts[0].node, base.node), (1.0, 0.0, 0.0))
    horiz = math.hypot(tunnel[0], tunnel[2])
    if horiz < 1e-6:
        return None
    bearing = _bearing(tunnel[0], tunnel[2])
    elevation = math.degrees(math.atan2(tunnel[1], horiz))

    # into the jetway object's frame: origin on the rotunda (at the
    # model's ground), parked tunnel along -Z
    h = math.radians(-bearing)
    rot = np.array([[math.cos(h), 0.0, -math.sin(h)], [0.0, 1.0, 0.0], [math.sin(h), 0.0, math.cos(h)]])
    origin = np.array([rotunda[0], 0.0, rotunda[2]])

    def to_obj(p):
        return rot @ (_TURN @ p - origin)

    def to_obj_dir(d):
        return rot @ (_TURN @ d)

    cab = next((p for p in parts if p.kind == "cab"), None)
    cab_node = cab.node if cab else exts[-1].node
    cab_pos = rot @ (wpos(cab_node) - origin)
    parked = math.hypot(cab_pos[0], cab_pos[2])
    constraints = read_ik(xml_text)[1]
    ranges = []
    for p in exts:
        c = constraints.get(str(nodes[p.node].get("name", "")).lower())
        rest = float((nodes[p.node].get("translation") or [0.0])[0])
        lo = c.lo if c and c.lo is not None else rest
        hi = c.hi if c and c.hi is not None else rest
        ranges.append((min(lo, rest), max(hi, rest), rest))
    span = sum(hi - lo for lo, hi, _ in ranges)
    shortest = parked - sum(rest - lo for lo, _, rest in ranges)
    longest = parked + sum(hi - rest for _, hi, rest in ranges)
    if span <= 1e-6:
        return None
    size = _tunnel_size(shortest, longest)

    cab_offset = 0.0
    if cab is not None:
        cab_dir = wdir(cab.node, (1.0, 0.0, 0.0))
        if math.hypot(cab_dir[0], cab_dir[2]) > 1e-6:
            cab_offset = ((_bearing(cab_dir[0], cab_dir[2]) - bearing + 180.0) % 360.0) - 180.0

    for p, (lo, hi, _rest) in zip(exts, ranges):
        p.share = (hi - lo) / span
    for p in parts:
        p.pivot = tuple(float(v) for v in rot @ (wpos(p.node) - origin))
        if p.kind == "ext":
            p.direction = tuple(float(v) for v in rot @ wdir(parent.get(p.node, base.node), (1.0, 0.0, 0.0)))

    # geometry
    tris = []
    materials = gltf.get("materials") or []
    for ni, node in enumerate(nodes):
        if node.get("mesh") is None or ni not in world_of:
            continue
        mesh = (gltf.get("meshes") or [])[node["mesh"]]
        joints = _joints_world(gltf, buffers, node, world_of, part_of)
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4 or mc._primitive_material_is_excluded(gltf, prim):
                continue
            mat_idx = prim.get("material", -1)
            for part, verts in _primitive_triangles(gltf, buffers, ni, world_of[ni], joints, part_of, prim):
                tris.append(_Tri(part, mat_idx if mat_idx is not None else -1,
                                 [(to_obj(v[0]), to_obj_dir(v[1]), v[2]) for v in verts]))
    if not tris:
        return None

    used = sorted({t.material for t in tris})
    atlas, lit, cells, blend = _atlas(gltf, buffers, model_path, used, materials, textures_dir,
                                      external_textures, obj_name, glass_opacity)
    tex_names = (f"{obj_name}_atlas.png", f"{obj_name}_atlas_lit.png" if lit is not None else None)
    atlas.save(Path(textures_dir) / tex_names[0])
    if lit is not None:
        lit.save(Path(textures_dir) / tex_names[1])

    _write_obj(Path(objects_dir) / f"{obj_name}.obj", tris, parts, cells, blend, materials, tex_names,
               parked, TUNNEL_SIZES[size], -elevation, cab_offset)
    return Jetway(obj_name=obj_name, rotunda_xz=(float(rotunda[0]), float(rotunda[2])),
                  tunnel_bearing=bearing, parked_length=parked, shortest=shortest, longest=longest,
                  size=size, cab_offset=cab_offset, style=_style(model_path.stem), parts=parts)


def _material_image(gltf, buffers, model_path, mat, textures_dir, external_textures, key):
    tex = mc.find_base_color_texture(mat) if key == "base" else mc.find_emissive_texture(mat)
    if not tex or "index" not in tex:
        return None
    img_idx, _ = mc.texture_image_index(gltf, tex["index"])
    if img_idx is None:
        return None
    try:
        name = mc.extract_image(gltf, buffers, img_idx, model_path, Path(textures_dir), external_textures, {},
                                (255, 255, 255, 255), allow_dds_passthrough=False)
        if not name:
            return None
        return Image.open(Path(textures_dir) / name).convert("RGBA")
    except Exception:
        return None


def _atlas(gltf, buffers, model_path, used, materials, textures_dir, external_textures, obj_name, glass_opacity):
    """One sheet for every material: (base RGBA, lit RGB or None,
    {material: (col, row, n)}, {material: "blend"|"mask"|"opaque"})."""
    n = max(1, math.ceil(math.sqrt(len(used))))
    cell = min(_CELL, ATLAS_MAX // n)
    base = Image.new("RGBA", (n * cell, n * cell), (128, 128, 128, 255))
    lit = Image.new("RGB", (n * cell, n * cell), (0, 0, 0))
    any_lit = False
    cells, blend = {}, {}
    opacity = mc.DEFAULT_GLASS_OPACITY if glass_opacity is None else glass_opacity
    for k, m in enumerate(used):
        col, row = k % n, k // n
        cells[m] = (col, row, n)
        mat = materials[m] if 0 <= m < len(materials) else {}
        pbr = mat.get("pbrMetallicRoughness") or {}
        factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        img = _material_image(gltf, buffers, model_path, mat, textures_dir, external_textures, "base")
        if img is None:
            rgba = mc.untextured_swatch_color(tuple(int(max(0, min(1, c)) * 255) for c in (factor + [1.0])[:4]),
                                              float(pbr.get("metallicFactor", 0.0) or 0.0))
            img = Image.new("RGBA", (cell, cell), rgba)
        else:
            img = img.resize((cell, cell), Image.BILINEAR)
            arr = np.asarray(img).astype(np.float32)
            arr *= np.array([max(0.0, min(1.0, c)) for c in (factor + [1.0])[:4]], dtype=np.float32)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
        mode = mat.get("alphaMode", "OPAQUE")
        name = str(mat.get("name", "")).lower()
        if mode == "BLEND" and any(w in name for w in ("glass", "window", "vitre", "windshield")):
            arr = np.asarray(img).copy()
            arr[..., 3] = (arr[..., 3].astype(np.float32) * mc.glass_alpha_255(opacity) / 255.0).astype(np.uint8)
            img = Image.fromarray(arr, "RGBA")
        if mode == "OPAQUE":
            img.putalpha(255)
        blend[m] = {"BLEND": "blend", "MASK": "mask"}.get(mode, "opaque")
        base.paste(img, (col * cell, row * cell))

        emissive = mat.get("emissiveFactor") or [0.0, 0.0, 0.0]
        e_img = _material_image(gltf, buffers, model_path, mat, textures_dir, external_textures, "emissive")
        e_factor = [max(0.0, min(1.0, c)) for c in (emissive + [0, 0, 0])[:3]]
        if e_img is not None:
            arr = np.asarray(e_img.resize((cell, cell), Image.BILINEAR).convert("RGB")).astype(np.float32)
            if max(e_factor) > 0:
                arr *= np.array(e_factor, dtype=np.float32)
            lit.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB"), (col * cell, row * cell))
            any_lit = True
        elif max(e_factor) > 0.05:
            lit.paste(Image.new("RGB", (cell, cell), tuple(int(c * 255) for c in e_factor)),
                      (col * cell, row * cell))
            any_lit = True
    return base, (lit if any_lit else None), cells, blend


def _atlas_uvs(uvs, cell_info, cell_px):
    """A triangle's three UVs, shifted into one tile and onto its cell
    (OBJ convention, v up). A triangle spanning several repeats of a
    tiling texture is clamped to one."""
    col, row, n = cell_info
    u = np.array([p[0] for p in uvs], dtype=np.float64)
    v = np.array([p[1] for p in uvs], dtype=np.float64)
    u -= math.floor(u.min()) if np.isfinite(u).all() else 0.0
    v -= math.floor(v.min()) if np.isfinite(v).all() else 0.0
    u, v = np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)
    inset = _PAD_PX / cell_px if cell_px > 4 * _PAD_PX else 0.0
    u = (col + inset + u * (1.0 - 2 * inset)) / n
    y = (row + inset + v * (1.0 - 2 * inset)) / n  # from the top of the sheet
    return list(zip(u, 1.0 - y))


def _fmt(x):
    return f"{x:.5f}"


def _write_obj(path, tris, parts, cells, blend, materials, tex_names, parked, size_range, pitch_rest, cab_offset):
    vt, idx = [], []
    groups = {}  # (part, material) -> [vertex indices]
    for t in tris:
        groups.setdefault((t.part, t.material), [])
    for t in tris:
        cell = cells[t.material]
        uvs = _atlas_uvs([v[2] for v in t.verts], cell, min(_CELL, ATLAS_MAX // cell[2]))
        base = len(vt)
        for (p, n, _), (u, v) in zip(t.verts, uvs):
            vt.append(f"VT {_fmt(p[0])} {_fmt(p[1])} {_fmt(p[2])} {_fmt(n[0])} {_fmt(n[1])} {_fmt(n[2])} "
                      f"{u:.6f} {v:.6f}\n")
        groups[(t.part, t.material)].extend([base, base + 1, base + 2])

    # one index list, group after group in the order they are drawn
    ranges = {}
    children = {k: [] for k in range(len(parts))}
    roots = []
    for k, p in enumerate(parts):
        (children[p.parent] if p.parent is not None else roots).append(k)
    order = [(-1, m) for (pt, m) in groups if pt == -1]

    def walk(k):
        order.extend((pt, m) for (pt, m) in groups if pt == k)
        for c in children[k]:
            walk(c)
    for r in roots:
        walk(r)
    for key in order:
        ranges[key] = (len(idx), len(groups[key]))
        idx.extend(groups[key])

    lo, hi = size_range
    lines = ["I\n", "800\n", "OBJ\n", "\n", f"TEXTURE ../textures/{tex_names[0]}\n"]
    if tex_names[1]:
        lines.append(f"TEXTURE_LIT ../textures/{tex_names[1]}\n")
    lines.append(f"POINT_COUNTS {len(vt)} 0 0 {len(idx)}\n\n")
    lines += vt
    full = len(idx) // 10 * 10
    for i in range(0, full, 10):
        lines.append("IDX10 " + " ".join(map(str, idx[i:i + 10])) + "\n")
    lines += [f"IDX {x}\n" for x in idx[full:]]
    lines.append("\n")

    def draw(key):
        start, count = ranges[key]
        mat = materials[key[1]] if 0 <= key[1] < len(materials) else {}
        mode = blend.get(key[1], "opaque")
        lines.append("ATTR_blend\n" if mode == "blend" else "ATTR_no_blend 0.5\n")
        lines.append("ATTR_no_cull\n" if mat.get("doubleSided") else "ATTR_cull\n")
        lines.append(f"TRIS {start} {count}\n")

    def trans(p, sign=1.0):
        x, y, z = (sign * c for c in p)
        lines.append(f"ANIM_trans {_fmt(x)} {_fmt(y)} {_fmt(z)} {_fmt(x)} {_fmt(y)} {_fmt(z)} 0 0 no_ref\n")

    def rotate(axis, dref, keys):
        lines.append(f"ANIM_rotate_begin {axis} {DATAREF}{dref}\n")
        for value, angle in keys:
            lines.append(f"ANIM_rotate_key {value:.4f} {angle:.4f}\n")
        lines.append("ANIM_rotate_end\n")

    def emit(k):
        p = parts[k]
        lines.append("ANIM_begin\n")
        if p.kind == "base":
            rotate("0 1 0", "jw_base_rotation", [(-180.0, 180.0), (180.0, -180.0)])
        elif p.kind == "pitch":
            trans(p.pivot)
            rotate("1 0 0", "jw_tunnel_pitch", [(pitch_rest - 30.0, 30.0), (pitch_rest + 30.0, -30.0)])
            trans(p.pivot, -1.0)
        elif p.kind == "ext":
            d = np.asarray(p.direction)
            a, b = d * p.share * (lo - parked), d * p.share * (hi - parked)
            lines.append(f"ANIM_trans_begin {DATAREF}jw_tunnel_extension\n")
            lines.append(f"ANIM_trans_key {lo:.4f} {_fmt(a[0])} {_fmt(a[1])} {_fmt(a[2])}\n")
            lines.append(f"ANIM_trans_key {hi:.4f} {_fmt(b[0])} {_fmt(b[1])} {_fmt(b[2])}\n")
            lines.append("ANIM_trans_end\n")
        elif p.kind == "cab":
            trans(p.pivot)
            rotate("0 1 0", "jw_cabin_rotation", [(cab_offset - 120.0, 120.0), (cab_offset + 120.0, -120.0)])
            trans(p.pivot, -1.0)
        elif p.kind == "leg":
            lines.append(f"ANIM_trans_begin {DATAREF}jw_bogie_elevation\n")
            lines.append("ANIM_trans_key -5.0000 0.00000 -5.00000 0.00000\n")
            lines.append("ANIM_trans_key 5.0000 0.00000 5.00000 0.00000\n")
            lines.append("ANIM_trans_end\n")
        elif p.kind == "bogie":
            trans(p.pivot)
            rotate("0 1 0", "jw_bogie_rotation", [(-180.0, 180.0), (180.0, -180.0)])
            trans(p.pivot, -1.0)
        for key in [key for key in order if key[0] == k]:
            draw(key)
        for c in children[k]:
            emit(c)
        lines.append("ANIM_end\n")

    for key in [key for key in order if key[0] == -1]:
        draw(key)
    for r in roots:
        emit(r)
    Path(path).write_text("".join(lines), encoding="utf-8")


# --- apt.dat ---------------------------------------------------------------

def placed(jetway, lat, lon, heading):
    """(rotunda lat, lon, parked tunnel heading) of a converted jetway
    placed like any converted model at (lat, lon, heading)."""
    r_lat, r_lon = geo_transform.local_offset_to_latlon(lat, lon, heading, *jetway.rotunda_xz)
    return r_lat, r_lon, (heading + jetway.tunnel_bearing) % 360.0


def rows_for(instances, stands=()):
    """apt.dat rows (1500 + 1501 each) for placed jetways: instances are
    (Jetway, lat, lon, heading). Where a stand has two or more, the cab
    parked furthest forward docks to door 1 and the others to door 2."""
    placed_rows = []
    for jw, lat, lon, hdg in instances:
        r_lat, r_lon, tunnel = placed(jw, lat, lon, hdg)
        m_lat, m_lon = geo_transform.metres_per_degree(r_lat)
        t = math.radians(tunnel)
        cab_e = jw.parked_length * math.sin(t)
        cab_n = jw.parked_length * math.cos(t)
        cab = (r_lat + cab_n / m_lat, r_lon + cab_e / m_lon)
        placed_rows.append([jw, r_lat, r_lon, tunnel, cab, jw.size])

    by_stand = {}
    for k, row in enumerate(placed_rows):
        best, best_d = None, 80.0
        for s_idx, s in enumerate(stands):
            m_lat, m_lon = geo_transform.metres_per_degree(s.lat)
            d = math.hypot((row[4][0] - s.lat) * m_lat, (row[4][1] - s.lon) * m_lon)
            if d < best_d:
                best, best_d = s_idx, d
        if best is not None:
            s = stands[best]
            m_lat, m_lon = geo_transform.metres_per_degree(s.lat)
            h = math.radians(s.heading)
            forward = (row[4][0] - s.lat) * m_lat * math.cos(h) + (row[4][1] - s.lon) * m_lon * math.sin(h)
            by_stand.setdefault(best, []).append((forward, k))
    for members in by_stand.values():
        if len(members) >= 2:
            first = max(members)[1]
            for _, k in members:
                if k != first:
                    placed_rows[k][5] += DOOR_2

    rows = []
    for jw, r_lat, r_lon, tunnel, _cab, size in placed_rows:
        cab_heading = (tunnel + jw.cab_offset) % 360.0
        rows.append(f"1500 {r_lat:.8f} {r_lon:.8f} {tunnel:.1f} {jw.style} {size} {tunnel:.1f} "
                    f"{jw.parked_length:.2f} {cab_heading:.1f}")
        rows.append(f"1501 objects/{jw.obj_name}.obj")
    return rows
