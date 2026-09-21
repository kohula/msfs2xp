#!/usr/bin/env python3
"""
msfs_glb2obj.py — Convert a Microsoft Flight Simulator .glb model to X-Plane OBJ8 format.
"""

import argparse
import base64
import gc
import io
import json
import logging
import math
import os
import re
import shutil
import struct
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import texture2ddecoder
except ImportError:
    texture2ddecoder = None

import gpu_accel
from .draped_ranking import draped_layer_offset, rank_draped_layer_offsets
from . import mesh_ir as mesh_ir_module

logger = logging.getLogger(__name__)

_EXPORT_LOCK = threading.Lock()
_TEXTURE_LOCK = threading.Lock()
_EXPORTED_COUNT = 0


def _unique_suffix():
    """Collision-safe temp-file suffix across both threads AND separate OS
    processes (GPU fork runs mesh conversion in a ProcessPoolExecutor, where
    threading.get_ident() alone can collide between processes' main threads)."""
    return f"{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}"


def _atomic_replace(src, dst, retries=8, base_delay=0.05):
    """os.replace() with retry-with-backoff: on Windows, replacing a file
    another process has open for reading (e.g. a shared texture another
    worker is mid-read on) raises PermissionError instead of just
    working. The other process's read handle is always short-lived."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (attempt + 1))

COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

TYPE_NUM_COMPONENTS = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4, "MAT3": 9, "MAT4": 16,
}

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

_KTX2_SUPERCOMPRESSION_NONE   = 0
_KTX2_SUPERCOMPRESSION_BASIS  = 1  # true ETC1S+BasisLZ -- needs the real Basis transcoder, not decodable here
_KTX2_SUPERCOMPRESSION_ZSTD   = 2
_KTX2_SUPERCOMPRESSION_ZLIB   = 3


def _decompress_supercompressed(raw_bytes, scheme):
    if scheme == _KTX2_SUPERCOMPRESSION_ZSTD:
        if zstandard is None:
            return None
        return zstandard.ZstdDecompressor().decompress(raw_bytes)
    if scheme == _KTX2_SUPERCOMPRESSION_ZLIB:
        if _zlib_module is None:
            return None
        return _zlib_module.decompress(raw_bytes)
    return raw_bytes


class MatBuilder:
    def __init__(self, name):
        self.name = name
        self.vertices = []
        self.uvs = []
        self.normals = []
        self.indices = []
        self.texture_name = None
        self.normal_texture_name = None
        self.vertex_blocks = {}
        self.base_color_factor = (255, 255, 255, 255)
        self.alpha_mode = "OPAQUE"
        self.alpha_cutoff = 0.5
        self.double_sided = False
        self.is_glass = False
        self.is_decal = False
        self.is_near_ground_flat = False  # per-material ground-level detection -- currently informational only, see convert()
        self.all_source_nodes_flat = True  # AND-reduced across every contributing node -- see convert()
        self.block_footprint_areas = []  # per-node-block XZ bbox area, m^2 -- see convert()'s footprint write-up

        self.uv_scale = [1.0, 1.0]
        self.uv_offset = [0.0, 0.0]
        self.tex_coord = 0

        self.emissive_texture_name = None
        self.anim_translate_keys = None  # [(dataref_value, dx, dy, dz), ...] or None
        self.anim_dataref = None
        self.anim_pivot = None  # (px, py, pz) world-space rotation pivot, or None
        self.anim_rotate = None  # (axis_x, axis_y, axis_z) or None
        self.anim_rotate_keys = None  # [(dataref_value, angle_degrees), ...] or None
        self.light_level_dataref = None  # (v1, v2, dataref) or None
        self.proximity_dataref = None  # custom plugin-driven dataref name, or None


def parse_glb(path):
    data = Path(path).read_bytes()
    magic, version, total_len = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"{path} is not a valid .glb file")

    offset = 12
    json_data = None
    bin_data = None
    while offset < total_len:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset: offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            json_data = json.loads(chunk.rstrip(b"\x00 \t\n\r").decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_data = chunk
    if json_data is None:
        raise ValueError("No JSON chunk found in glb")
    return json_data, bin_data


def parse_gltf(path):
    """Standalone (non-binary) .gltf files -- used by MSFS SimObjects models
    (jetways, GSE, doors, ...), unlike the embedded-in-scenery-BGL models
    which are always packed as .glb -- are just plain JSON with no BIN
    chunk at all; their buffer.uri values point at sibling .bin files on
    disk, which load_buffers() already resolves via the external-file-URI
    branch. Returns (json_data, None) to match parse_glb's signature."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), None


def parse_model_file(path):
    """Dispatches to parse_glb or parse_gltf by file extension."""
    if Path(path).suffix.lower() == ".gltf":
        return parse_gltf(path)
    return parse_glb(path)


def load_buffers(gltf, bin_chunk, glb_path):
    buffers = []
    for buf in gltf.get("buffers", []):
        uri = buf.get("uri")
        if uri is None:
            buffers.append(bin_chunk)
        elif uri.startswith("data:"):
            _, b64 = uri.split(",", 1)
            buffers.append(base64.b64decode(b64))
        else:
            buffers.append((Path(glb_path).parent / uri).read_bytes())
    return buffers


def read_accessor(gltf, buffers, accessor_index, force_normalized=False):
    accessor = gltf["accessors"][accessor_index]
    component_type = accessor["componentType"]
    dtype = COMPONENT_DTYPES[component_type]
    ncomp = TYPE_NUM_COMPONENTS[accessor["type"]]
    count = accessor["count"]

    if "bufferView" not in accessor:
        return np.zeros((count, ncomp), dtype=np.float32)

    bv = gltf["bufferViews"][accessor["bufferView"]]
    buf_idx = bv.get("bufferIndex", bv.get("buffer", 0))
    buf = buffers[buf_idx]

    byte_offset = bv.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    itemsize = np.dtype(dtype).itemsize
    default_stride = ncomp * itemsize
    stride = bv.get("byteStride", default_stride)

    max_bytes = (count - 1) * stride + default_stride
    raw1d = np.frombuffer(buf, dtype=np.uint8, count=max_bytes, offset=byte_offset)

    if stride == default_stride:
        arr = raw1d.view(dtype).reshape(count, ncomp)
    else:
        pad_len = count * stride - max_bytes
        if pad_len > 0:
            raw1d = np.pad(raw1d, (0, pad_len), mode='constant')
        raw2d = raw1d.reshape(count, stride)
        arr = raw2d[:, :default_stride].flatten().view(dtype).reshape(count, ncomp)

    # MSFS/ASOBO's glTF export packs TEXCOORD data as IEEE-754 half-
    # precision floats inside an accessor merely labeled componentType
    # SHORT (5122), since glTF has no formal half-float type, without
    # ever setting "normalized". Reading that bit pattern as a signed-
    # normalized integer instead (the spec-compliant behavior) silently
    # produces wrong UVs (float16 1.0 = 0x3C00 = int16 15360, and
    # 15360/32767 = 0.469, not 1.0) -- crops atlas-based textures to the
    # wrong corner. Only triggers when "normalized" is unset, so a
    # genuinely spec-compliant file still gets the normalize path below.
    if force_normalized and component_type == 5122 and not accessor.get("normalized"):
        return arr.view(np.float16).astype(np.float64)

    if accessor.get("normalized") or (force_normalized and dtype != np.float32):
        info = np.iinfo(dtype)
        if info.min == 0:
            arr = arr.astype(np.float64) / info.max
        else:
            arr = np.maximum(arr.astype(np.float64) / info.max, -1.0)

    return arr.astype(np.float64)


def node_instance_matrices(gltf, buffers, node):
    ext = node.get("extensions", {}).get("EXT_mesh_gpu_instancing")
    if not ext:
        return None
    attrs = ext.get("attributes", {})
    translations = read_accessor(gltf, buffers, attrs["TRANSLATION"]) if "TRANSLATION" in attrs else None
    rotations = read_accessor(gltf, buffers, attrs["ROTATION"]) if "ROTATION" in attrs else None
    scales = read_accessor(gltf, buffers, attrs["SCALE"]) if "SCALE" in attrs else None

    count = next((len(arr) for arr in (translations, rotations, scales) if arr is not None), None)
    if count is None:
        return None

    matrices = []
    for i in range(count):
        t = np.eye(4)
        if translations is not None:
            t[0:3, 3] = translations[i]
        r = quat_to_matrix(rotations[i]) if rotations is not None else np.eye(4)
        s = np.eye(4)
        if scales is not None:
            s[0, 0], s[1, 1], s[2, 2] = scales[i]
        matrices.append(t @ r @ s)
    return matrices


def quat_to_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(4)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    m = np.eye(4)
    m[0, 0] = 1 - (yy + zz)
    m[0, 1] = xy - wz
    m[0, 2] = xz + wy
    m[1, 0] = xy + wz
    m[1, 1] = 1 - (xx + zz)
    m[1, 2] = yz - wx
    m[2, 0] = xz - wy
    m[2, 1] = yz + wx
    m[2, 2] = 1 - (xx + yy)
    return m


def node_local_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")
    t = np.eye(4)
    if "translation" in node:
        t[0:3, 3] = node["translation"]
    r = quat_to_matrix(node["rotation"]) if "rotation" in node else np.eye(4)
    s = np.eye(4)
    if "scale" in node:
        sx, sy, sz = node["scale"]
        s[0, 0], s[1, 1], s[2, 2] = sx, sy, sz
    return t @ r @ s


def _node_scale_only(node):
    """Just the scale component of a node's own authored transform -- used
    by node_local_matrix_at_rest, which overrides translation/rotation but
    never scale (MSFS doesn't animate scale on these nodes)."""
    if "matrix" in node:
        m3 = np.array(node["matrix"], dtype=np.float64).reshape(4, 4, order="F")[:3, :3]
        return np.linalg.norm(m3, axis=0)
    if "scale" in node:
        return np.array(node["scale"], dtype=np.float64)
    return np.array([1.0, 1.0, 1.0])


def _rest_open_rotation_values(values):
    """Picks whichever of a rotation channel's FIRST or LAST keyframe sits
    closer to identity as the rest/closed pose, rather than always trusting
    values[0] -- some hand-authored door/barrier animations close at
    values[-1] instead. Returns (rest_quat, open_quat) as float64 (x,y,z,w)."""
    v0 = np.asarray(values[0][:4], dtype=np.float64)
    v1 = np.asarray(values[-1][:4], dtype=np.float64)
    return (v0, v1) if abs(v0[3]) >= abs(v1[3]) else (v1, v0)


def node_local_matrix_at_rest(node, node_idx, gltf_animations):
    """Like node_local_matrix(), but for a node carrying a translation/
    rotation animation channel: builds the static/rest transform from the
    animation's own rest keyframe (values[0] for translation; for
    rotation, see _rest_open_rotation_values) instead of the node's
    separately-authored bind-pose TRS. Some exporters leave an animated
    node's static transform at identity, expecting every pose -- including
    rest -- to come from the sampler; without this, un-animated geometry
    would bake at the wrong pose (e.g. a door rendering open by default)
    while the ANIM_ delta (computed the same way) stays correct."""
    anim = gltf_animations.get(node_idx) if gltf_animations else None
    if not anim or len(anim["values"]) == 0:
        return node_local_matrix(node)

    scale = _node_scale_only(node)
    s = np.eye(4)
    s[0, 0], s[1, 1], s[2, 2] = scale

    if anim["path"] == "translation":
        t = np.eye(4)
        t[0:3, 3] = np.asarray(anim["values"][0][:3], dtype=np.float64)
        r = quat_to_matrix(node["rotation"]) if "rotation" in node else np.eye(4)
    else:  # "rotation"
        t = np.eye(4)
        if "translation" in node:
            t[0:3, 3] = node["translation"]
        rest_quat, _ = _rest_open_rotation_values(anim["values"])
        r = quat_to_matrix(rest_quat)

    return t @ r @ s


def _primitive_material_is_excluded(gltf, prim):
    """True for a primitive whose material marks it as never actually
    rendered -- an invisible collision hull, a lightcone proxy, a shadow
    volume. The main emission loop in convert() has always skipped these
    (see its own identical check); compute_file_flatness_and_reference used
    to scan every primitive with no such filter, which meant a node's
    flat/non-flat verdict could be computed from a triangle population that
    didn't match what was actually written to the .obj at all -- e.g. a
    sign node whose invisible collision box (a handful of large,
    non-horizontal triangles) got counted toward "is this node flat"
    alongside its real, genuinely flat visible face, silently pulling the
    verdict either direction depending on which primitive type happened to
    dominate the node's triangle count. Both call sites now share this one
    check so they can never drift apart like that again."""
    mat_idx = prim.get("material")
    mats = gltf.get("materials", [])
    mat = mats[mat_idx] if mat_idx is not None and mat_idx < len(mats) else {}
    if "ASOBO_material_invisible" in mat.get("extensions", {}):
        return True
    raw_mat_name = mat.get("name", "")
    return any(x in raw_mat_name.lower() for x in ["collision", "invisible", "lightcone", "light_cone", "shadow_volume"])


def compute_file_flatness_and_reference(gltf, buffers, world_transforms, flat_eps=0.1, merge_gap=0.05, normal_up_threshold=0.9):
    """Scans every triangle in the ENTIRE file (every node, every mesh,
    every primitive) once. Returns (flat_fraction, reference_height):

      - flat_fraction: fraction of ALL triangles in the file that are
        individually near-flat: their 3 vertices vary in height by less
        than `flat_eps` AND the triangle's face normal points close to
        straight up/down (within `normal_up_threshold`, a cosine of the
        angle from vertical). Both conditions matter -- height-spread
        alone isn't enough, because a finely tessellated 3D surface (a
        human character's body, a curved fairing) is also made of many
        individually SMALL triangles that each have a tiny Y-spread
        purely from their size, while still facing sideways. Requiring a
        near-horizontal normal is what actually captures "this is a flat
        ground decal/marking", not merely "this triangle happens to be
        small". Used to decide whether a file counts as "almost one
        dimensional" -- i.e. contains nothing but flat 2D decals/markings,
        no genuine 3D structure anywhere in it. That's an all-or-nothing,
        file-wide judgment: a file that's mostly flat but has even one
        real 3D object mixed in (a vent stack, a building, a curb wall)
        does NOT qualify -- the whole file is left untouched in that
        case, rather than trying to guess which parts are safe to
        flatten.

      - reference_height: the file's own dominant/majority elevation,
        taken only from the flat triangles' vertices (so a handful of
        non-flat triangles in a borderline file don't skew it). This is
        NOT assumed to be 0 -- if the file's real, intended elevation is
        (say) 12m up on a roof or a bridge deck, that's what gets
        preserved. Vertices within `merge_gap` of each other are treated
        as the same level (absorbs float/export jitter); the reference
        is the weighted average height of whichever level has the most
        vertex support. None if there's no flat geometry at all to
        derive a reference from.

      - max_radius: the largest 3-D distance, in world space, from the
        object's own local origin (0,0,0 -- the point TILTED would pivot
        the whole rigid object around) to any (non-stray, see
        flag_stray_vertices) vertex in the file.

      - max_horizontal_radius: the same, but measured only in the
        horizontal (X/Z) plane, ignoring height. This is the one that
        actually matters for TILTED: rotating the whole object to match a
        SAMPLED terrain normal only exactly cancels out real vertical
        error if that one sampled point is perfectly representative of the
        true average slope under the whole footprint. Any mismatch between
        the sampled point and the real terrain (a DSF mesh seam, a locally
        atypical spot) turns into vertical positional error proportional
        to how far a vertex sits HORIZONTALLY from the pivot -- a vertex
        directly above the pivot barely moves vertically even under a
        badly wrong tilt, one 40m off to the side swings by meters for the
        same angular error. So a building whose footprint stays close to
        its own origin can safely take TILTED; one with real geometry (an
        upper floor, a wing) offset far from origin horizontally is
        exactly the case that showed up as an interior floor "shifted"
        under TILTED even though the object's overall placement was fine
        (see convert()).

      - node_stats: {node_idx: (flat_fraction, reference_height)}, the
        exact same two tallies as above but broken out PER NODE instead of
        aggregated across the whole file. Computed but currently UNUSED by
        convert() (this_node_is_flat = file_is_flat_only everywhere -- see
        that assignment's own comment for why per-node was tried and
        reverted: it can split one continuous real-world surface into a
        draped half and a rigid half that can never weld against each
        other). Kept for any future diagnostic use.

      - material_stats: {mat_idx: (flat_fraction, reference_height)}, the
        same two tallies broken out PER MATERIAL instead -- matches the
        actual granularity convert() builds output objects at (one
        "builder" per mat_idx, see builder_key in convert()), unlike
        node_stats above. Used to DETECT materials the file-wide verdict
        rejects but that are themselves near-flat and near ground level: a
        real EGLC ground-layer model packs ~25 unrelated materials
        (asphalt, concrete, paint markings, individual paver/tile decals,
        plus one genuinely 3D ramp detail) into ONE file, and that one
        non-flat material vetoes ATTR_draped for every other material too
        under the file-wide, all-or-nothing check -- confirmed real
        symptom: paver/tile materials each individually ~95%+ flat (by
        this same per-triangle test, just at a looser per-material
        threshold -- see _NEAR_GROUND_FLAT_FRACTION_THRESHOLD in
        convert()) end up written as RIGID objects instead, at whatever
        small nonzero local Y they happened to be authored at (a baked
        authoring-tool artifact, confirmed ~1.5m for one real case), which
        repro's exactly as "pavement floating above the ground" once
        compiled. convert() only trusts material_stats for a material when
        its own reference_height is ALSO close to the file's overall
        reference_height (see _NEAR_GROUND_FLAT_TOLERANCE_M in convert())
        -- flatness alone isn't enough, since a building's flat ROOF or a
        bridge deck's flat TOP would also pass a pure per-material
        flatness test despite being meters above the file's own ground
        level, and must stay rigid (this is exactly the failure mode an
        earlier, more aggressive per-material attempt hit this session:
        it had no ground-proximity check at all). A material detected this
        way stays RIGID (is_near_ground_flat is informational only, not a
        drape/drop decision) instead of being draped with a guessed layer
        rank -- MSFS's own intended stacking order for this kind of small
        patch/paver detail can't be recovered from the source data, so
        it's left at its own authored position rather than guessing where
        in the draw order it belongs. Used to be DROPPED from the output
        entirely instead; reverted per a real-world comparison against
        another converter's output for the same content (see convert()'s
        own is_near_ground_flat comment), which showed dropping it was
        worse than leaving it rigid.
    """
    total_tris = 0
    flat_tris = 0
    flat_y_values = []
    max_radius = 0.0
    max_horizontal_radius = 0.0

    # Per-node breakdown of the exact same tallies as the file-wide ones
    # above -- used by convert() to decide flattening/draping PER NODE
    # instead of only ever all-or-nothing for the whole file (see
    # node_stats in the return value and its docstring entry below). The
    # file-wide totals above are kept completely unchanged alongside this
    # -- TILTED's decision (see convert()) still uses only those, exactly
    # as before.
    node_total_tris = {}
    node_flat_tris = {}
    node_flat_y_values = {}

    # Per-material breakdown -- see material_stats in the return value's
    # own docstring entry above for why this is tracked at material
    # granularity (matching convert()'s own builder_key) rather than
    # per-node.
    mat_total_tris = {}
    mat_flat_tris = {}
    mat_flat_y_values = {}

    for node_idx, node in enumerate(gltf.get("nodes", [])):
        if "mesh" not in node:
            continue
        base_world = world_transforms.get(node_idx, np.eye(4))
        mesh = gltf["meshes"][node["mesh"]]
        indices_cache = {}
        pos_cache = {}
        for prim in mesh.get("primitives", []):
            pos_acc = prim.get("attributes", {}).get("POSITION")
            idx_acc = prim.get("indices")
            if pos_acc is None or idx_acc is None:
                continue
            if _primitive_material_is_excluded(gltf, prim):
                continue

            if pos_acc not in pos_cache:
                positions = read_accessor(gltf, buffers, pos_acc)
                positions_h = np.hstack([positions, np.ones((len(positions), 1))])
                pos_cache[pos_acc] = (base_world @ positions_h.T).T[:, :3]
            pos_world = pos_cache[pos_acc]
            y = pos_world[:, 1]
            if len(pos_world):
                # Excludes the same degenerate/leftover stray vertices
                # flag_stray_vertices drops from actual rendering (see its
                # own docstring) -- without this, a single corrupted vertex
                # hundreds of meters from an otherwise building-scale mesh
                # inflates max_radius by 10-20x, which used to make the
                # TILTED decision (see convert()) look at a number with no
                # relationship to the file's real size.
                not_stray = ~flag_stray_vertices(pos_world)
                clean = pos_world[not_stray]
                if len(clean):
                    max_radius = max(max_radius, float(np.linalg.norm(clean, axis=1).max()))
                    max_horizontal_radius = max(max_horizontal_radius, float(np.linalg.norm(clean[:, [0, 2]], axis=1).max()))

            if idx_acc not in indices_cache:
                indices_cache[idx_acc] = read_accessor(gltf, buffers, idx_acc).astype(np.int64).reshape(-1)
            full_idx = indices_cache[idx_acc]
            extras = prim.get("extras", {}).get("ASOBO_primitive", {})
            start = extras.get("StartIndex", 0)
            base_v = extras.get("BaseVertexIndex", 0)
            idx_count = (extras["PrimitiveCount"] * 3) if "PrimitiveCount" in extras else (len(full_idx) - start)
            local = full_idx[start:start + idx_count] + base_v
            if len(local) < 3:
                continue
            tri = local[: (len(local) // 3) * 3].reshape(-1, 3)
            spread = y[tri].max(axis=1) - y[tri].min(axis=1)

            # Orientation gate: a genuinely flat ground decal/marking
            # triangle lies roughly in the horizontal plane, so its face
            # normal points close to straight up or down. Small triangles
            # from fine tessellation on a vertical/curved 3D surface (e.g.
            # a character's torso) have a small Y-spread too, but their
            # normals point sideways -- this is what actually tells the
            # two cases apart. Degenerate/zero-area triangles have no
            # meaningful normal, so they're left to the spread test alone
            # rather than forced to fail here.
            v0 = pos_world[tri[:, 0]]
            v1 = pos_world[tri[:, 1]]
            v2 = pos_world[tri[:, 2]]
            face_normal = np.cross(v1 - v0, v2 - v0)
            face_normal_len = np.linalg.norm(face_normal, axis=1)
            safe_len = np.where(face_normal_len > 1e-12, face_normal_len, 1.0)
            normal_up_ratio = np.abs(face_normal[:, 1]) / safe_len
            is_horizontal = (face_normal_len <= 1e-12) | (normal_up_ratio >= normal_up_threshold)

            # Real runway/taxiway pavement is rarely dead-level -- it's
            # graded a percent or two for drainage. A fixed absolute
            # flat_eps doesn't scale with triangle size: a big ground-
            # marking triangle spanning tens of meters can rack up more
            # than flat_eps of honest grade-driven height difference and
            # get misclassified as real 3D geometry, skipping the
            # ATTR_draped write it should get. So a triangle also counts
            # as flat if its height spread is within a grade allowance of
            # its own horizontal size, on top of the fixed flat_eps floor.
            xz0, xz1, xz2 = v0[:, [0, 2]], v1[:, [0, 2]], v2[:, [0, 2]]
            horiz_extent = np.maximum(
                np.maximum(np.linalg.norm(xz0 - xz1, axis=1), np.linalg.norm(xz1 - xz2, axis=1)),
                np.linalg.norm(xz2 - xz0, axis=1),
            )
            max_grade = 0.15
            allowed_spread = np.maximum(flat_eps, horiz_extent * max_grade)

            total_tris += len(spread)
            is_flat = (spread <= allowed_spread) & is_horizontal
            flat_tris += int(is_flat.sum())
            if is_flat.any():
                flat_y_values.append(y[tri[is_flat]].reshape(-1))

            node_total_tris[node_idx] = node_total_tris.get(node_idx, 0) + len(spread)
            node_flat_tris[node_idx] = node_flat_tris.get(node_idx, 0) + int(is_flat.sum())
            if is_flat.any():
                node_flat_y_values.setdefault(node_idx, []).append(y[tri[is_flat]].reshape(-1))

            mat_idx = prim.get("material")
            mat_total_tris[mat_idx] = mat_total_tris.get(mat_idx, 0) + len(spread)
            mat_flat_tris[mat_idx] = mat_flat_tris.get(mat_idx, 0) + int(is_flat.sum())
            if is_flat.any():
                mat_flat_y_values.setdefault(mat_idx, []).append(y[tri[is_flat]].reshape(-1))

    material_stats = {}
    for mat_idx, m_total in mat_total_tris.items():
        m_flat_fraction = (mat_flat_tris.get(mat_idx, 0) / m_total) if m_total else 0.0
        m_reference_height = None
        m_flat_ys = mat_flat_y_values.get(mat_idx)
        if m_flat_ys:
            m_all_y = np.concatenate(m_flat_ys)
            _, _, m_reference_height, _ = _cluster_height_bands(m_all_y, merge_gap=merge_gap)
        material_stats[mat_idx] = (m_flat_fraction, m_reference_height)

    node_stats = {}
    for node_idx, n_total in node_total_tris.items():
        n_flat_fraction = (node_flat_tris.get(node_idx, 0) / n_total) if n_total else 0.0
        n_reference_height = None
        n_flat_ys = node_flat_y_values.get(node_idx)
        if n_flat_ys:
            n_all_y = np.concatenate(n_flat_ys)
            _, _, n_reference_height, _ = _cluster_height_bands(n_all_y, merge_gap=merge_gap)
        node_stats[node_idx] = (n_flat_fraction, n_reference_height)

    flat_fraction = (flat_tris / total_tris) if total_tris else 0.0

    reference_height = None
    if flat_y_values:
        all_y = np.concatenate(flat_y_values)
        _, _, reference_height, _ = _cluster_height_bands(all_y, merge_gap=merge_gap)

    return flat_fraction, reference_height, max_radius, max_horizontal_radius, node_stats, material_stats


def _cluster_height_bands(y_values, merge_gap=0.05):
    """1D-cluster an array of 'up-axis' (Y) coordinates into bands, merging
    values that are within `merge_gap` of their neighbor. Returns
    (cluster_id_per_vertex, dominant_cluster_id, dominant_ref_height,
    n_clusters), where dominant_ref_height is the weighted-average height
    of whichever band has the most vertex support.
    """
    rounded = np.round(y_values, 3)
    uniq, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(uniq)
    uniq, counts = uniq[order], counts[order]

    gaps = np.diff(uniq)
    boundaries = uniq[1:][gaps > merge_gap]
    cluster_of_uniq = np.searchsorted(boundaries, uniq, side="right")
    n_clusters = int(cluster_of_uniq.max()) + 1 if len(uniq) else 1

    cluster_of_vertex = np.searchsorted(boundaries, rounded, side="right")

    support = np.bincount(cluster_of_uniq, weights=counts, minlength=n_clusters)
    dominant_cluster = int(np.argmax(support))
    dom_mask_uniq = cluster_of_uniq == dominant_cluster
    ref_height = float(
        np.average(uniq[dom_mask_uniq], weights=counts[dom_mask_uniq])
    ) if dom_mask_uniq.any() else 0.0

    return cluster_of_vertex, dominant_cluster, ref_height, n_clusters


def flatten_to_reference(positions_world, reference_height):
    """For files that qualify as 'almost one dimensional' (see
    compute_file_flatness_and_reference): compress every vertex onto the
    exact same level -- the file's own dominant elevation -- removing any
    offset between objects entirely. Unlike an unconditional snap-to-zero,
    this preserves real elevation: if the file's true, majority level is
    genuinely up on a roof or bridge deck, that's what every vertex gets
    set to, not 0. X/Z are untouched.
    """
    positions_world[:, 1] = reference_height
    return positions_world


_IMPLAUSIBLE_ELEVATION_LIMIT = 100.0


def _clamp_flat_reference_height(reference_height, label, glb_name):
    """Same clamping rule convert() already applies to the file-wide
    reference height (see the comments at its one call site): a dominant
    flat-layer elevation that's negative (unrepresentable -- DSF placement
    has no per-object vertical offset) or wildly implausible for ground-
    overlay content (an export losing track of a sane local origin, not a
    real airport being literally that uneven) gets clamped to y=0 instead
    of teleporting the geometry there. Shared here so the same rule applies
    per-node, not just to the file-wide aggregate."""
    if reference_height < 0.0:
        logger.info(f"{glb_name}: {label} dominant level {reference_height:.4f}m is below ground -- clamping reference to y=0")
        return 0.0
    if abs(reference_height) > _IMPLAUSIBLE_ELEVATION_LIMIT:
        logger.info(
            f"{glb_name}: {label} dominant level {reference_height:.4f}m is not a plausible "
            f"elevation for flat ground-overlay content -- clamping reference to y=0"
        )
        return 0.0
    return reference_height


def flag_stray_vertices(positions_world, percentile=1.0, pad_multiple=3.0, min_pad=25.0):
    """Flags vertices that sit wildly outside the REST of this same
    primitive's own vertex cloud on any axis -- e.g. a corrupted/leftover
    proxy or LOD-separator vertex in the source asset, exported as part of
    a real triangle even though it was never meant to be visible geometry
    (a classic symptom: a handful of vertices hundreds of meters from an
    otherwise building-scale mesh, forming a huge degenerate "slab"
    triangle once connected to normal vertices).

    Deliberately geometry-only and name-independent: uses percentiles of
    the primitive's own coordinates (not fixed world thresholds), so it
    scales to whatever that primitive's natural size is -- a large but
    well-populated polygon (many points spread along its real length,
    e.g. a runway centerline) keeps expanding the percentile bounds with
    it and is never flagged, whereas a handful of stray points far outside
    where the bulk of the primitive's own vertices sit gets caught
    regardless of what the mesh or material is named.
    """
    if len(positions_world) == 0:
        return np.zeros(0, dtype=bool)
    lo = np.percentile(positions_world, percentile, axis=0)
    hi = np.percentile(positions_world, 100 - percentile, axis=0)
    pad = np.maximum((hi - lo) * pad_multiple, min_pad)
    return np.any((positions_world < lo - pad) | (positions_world > hi + pad), axis=1)


def collect_world_transforms(gltf, gltf_animations=None):
    nodes = gltf.get("nodes", [])
    world = {}

    def visit(idx, parent_matrix):
        node = nodes[idx]
        local = node_local_matrix_at_rest(node, idx, gltf_animations)
        m = parent_matrix @ local
        world[idx] = m
        for child in node.get("children", []):
            visit(child, m)

    scene_idx = gltf.get("scene", 0)
    roots = gltf["scenes"][scene_idx]["nodes"] if gltf.get("scenes") else range(len(nodes))
    for r in roots:
        visit(r, np.eye(4))
    return world


def collect_node_parents(gltf):
    """{child_node_idx: parent_node_idx} across the whole node graph. MSFS
    routinely animates an empty joint/pivot node (e.g. a barrier's hinge)
    whose actual visible geometry lives on a separate child mesh node --
    this is what lets convert() walk a mesh node's ancestors to find the
    animation that actually drives it, instead of only ever checking the
    mesh node's own index (which read_gltf_animations, keyed by the node
    the glTF channel literally targets, would never match in that shape)."""
    parents = {}
    for idx, node in enumerate(gltf.get("nodes", [])):
        for child in node.get("children", []):
            parents[child] = idx
    return parents


_TEXTURE_COMPOUND_EXTS = [
    '.png.ktx2', '.png.dds', '.tif.ktx2', '.tif.dds',  # e.g. inibuilds EGLC ships *.TIF.KTX2
    '.ktx2', '.dds', '.tga', '.tiff', '.tif', '.jpeg', '.jpg', '.png',
]


_DUPLICATE_SUFFIX_RE = re.compile(r'\.\d+$')


def clean_texture_stem(name_or_uri):
    if not name_or_uri:
        return ""
    # Split on both separators: MSFS glTF image URIs are often a full
    # Windows path with backslashes, which Path(...).name won't split on
    # a POSIX host.
    name = re.split(r"[\\/]", str(name_or_uri))[-1].lower()
    # Some DCC exporters append a ".NNN" duplicate-name disambiguator
    # after the real extension (e.g. "foo_albd.png.001") -- strip it
    # before the compound-extension check below, or it won't match.
    name = _DUPLICATE_SUFFIX_RE.sub('', name)
    for ext in _TEXTURE_COMPOUND_EXTS:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    return name


def sanitize_name(name):
    if not name:
        return "texture"
    keep = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return keep or "texture"


def material_name(gltf, material_index):
    if material_index is None:
        return "default_material"
    materials = gltf.get("materials", [])
    if material_index >= len(materials):
        return f"material_{material_index}"
    mat = materials[material_index]
    name = mat.get("name") or f"material_{material_index}"
    keep = "".join(c if c.isalnum() or c in "-" else "_" for c in name)
    return keep or "material"


def _bc5_snorm_to_unorm_bytes(data):
    """texture2ddecoder's decode_bc5 only implements UNORM (endpoint bytes
    as unsigned 0..255). BC5_SNORM (the normal-map convention) stores them
    as signed two's-complement instead. XOR 0x80 on each endpoint byte is
    the standard order-preserving SNORM->UNORM bias trick, landing on the
    same 128-biased range the normal-map reconstruction already expects.
    Only the 2 endpoint bytes of each 8-byte half-block are touched."""
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


def decode_ktx2_bytes_to_png(raw_bytes, out_png_path):
    try:
        if texture2ddecoder is None or not raw_bytes.startswith(b"\xabKTX 20\xbb\r\n\x1a\n"):
            return False

        header_data = raw_bytes[12:12 + 17 * 4]
        unpacked = struct.unpack("<17I", header_data)
        vk_format, width, height, level_count, supercompression = unpacked[0], unpacked[2], unpacked[3], unpacked[7], unpacked[8]

        if supercompression == _KTX2_SUPERCOMPRESSION_BASIS:
            return False

        level_index_offset = 12 + 17 * 4
        level_index_data = raw_bytes[level_index_offset : level_index_offset + level_count * 24]
        offset, length, _ = struct.unpack("<3Q", level_index_data[:24])
        compressed_bytes = raw_bytes[offset : offset + length]

        compressed_bytes = _decompress_supercompressed(compressed_bytes, supercompression)
        if compressed_bytes is None:
            return False

        if vk_format in (VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_UNORM_BLOCK):
            decoded = texture2ddecoder.decode_bc1(compressed_bytes, width, height)
        elif vk_format == VK_FORMAT_BC3_UNORM_BLOCK:
            decoded = texture2ddecoder.decode_bc3(compressed_bytes, width, height)
        elif vk_format == VK_FORMAT_BC4_UNORM_BLOCK:
            decoded = texture2ddecoder.decode_bc4(compressed_bytes, width, height)
        elif vk_format in (VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC5_SNORM_BLOCK):
            bc5_data = compressed_bytes
            if vk_format == VK_FORMAT_BC5_SNORM_BLOCK:
                bc5_data = _bc5_snorm_to_unorm_bytes(bc5_data)
            decoded = texture2ddecoder.decode_bc5(bc5_data, width, height)
        elif vk_format == VK_FORMAT_BC7_UNORM_BLOCK:
            decoded = texture2ddecoder.decode_bc7(compressed_bytes, width, height)
        elif vk_format in (VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_FORMAT_ASTC_4x4_SRGB_BLOCK):
            decoded = texture2ddecoder.decode_astc(compressed_bytes, width, height, 4, 4)
        elif vk_format == VK_FORMAT_R8G8B8A8_UNORM:
            decoded = compressed_bytes
        else:
            return False

        img = Image.frombytes("RGBA", (width, height), decoded, "raw", "BGRA")
        img.save(str(out_png_path), "PNG")
        return True
    except Exception:
        return False


def decode_dds_bytes_to_png(raw_bytes, out_png_path):
    if texture2ddecoder is None or not raw_bytes.startswith(b"DDS "):
        return False
    try:
        height, width = struct.unpack_from("<II", raw_bytes, 12)
        fourcc = raw_bytes[84:88]
        offset = 128
        fmt = None
        bc5_signed = False

        if fourcc == b"DX10":
            dxgi_format = struct.unpack_from("<I", raw_bytes, 128)[0]
            offset = 148
            if dxgi_format in (70, 71, 72):
                fmt = "bc1"
            elif dxgi_format in (73, 74, 75, 76, 77, 78):
                fmt = "bc3"
            elif dxgi_format in (79, 80, 81):
                fmt = "bc4"
            elif dxgi_format in (82, 83, 84):
                fmt = "bc5"
                bc5_signed = dxgi_format == 84  # BC5_SNORM
            elif dxgi_format in (97, 98, 99):
                fmt = "bc7"
        else:
            if fourcc == b"DXT1":
                fmt = "bc1"
            elif fourcc in (b"DXT3", b"DXT5"):
                fmt = "bc3"
            elif fourcc in (b"ATI1", b"BC4U", b"BC4S"):
                fmt = "bc4"
            elif fourcc in (b"ATI2", b"BC5U", b"BC5S"):
                fmt = "bc5"
                bc5_signed = fourcc == b"BC5S"
            elif fourcc.startswith(b"BC7"):
                fmt = "bc7"

        if not fmt:
            return False

        compressed_bytes = raw_bytes[offset:]
        if fmt == "bc1":
            decoded = texture2ddecoder.decode_bc1(compressed_bytes, width, height)
        elif fmt == "bc3":
            decoded = texture2ddecoder.decode_bc3(compressed_bytes, width, height)
        elif fmt == "bc4":
            decoded = texture2ddecoder.decode_bc4(compressed_bytes, width, height)
        elif fmt == "bc5":
            # See _bc5_snorm_to_unorm_bytes: the decoder only understands the
            # UNORM endpoint convention, so SNORM (signed) BC5 -- the normal
            # -map case -- needs its endpoint bytes bias-flipped first.
            if bc5_signed:
                compressed_bytes = _bc5_snorm_to_unorm_bytes(compressed_bytes)
            decoded = texture2ddecoder.decode_bc5(compressed_bytes, width, height)
        elif fmt == "bc7":
            decoded = texture2ddecoder.decode_bc7(compressed_bytes, width, height)
        else:
            return False

        img = Image.frombytes("RGBA", (width, height), decoded, "raw", "BGRA")
        img.save(str(out_png_path), "PNG")
        return True
    except Exception as e:
        logger.debug(f"DDS decoding exception: {e}")
        return False


def _is_valid_image(path):
    """True only if path opens AND fully decodes without error -- used
    everywhere this pipeline decides whether an existing output file can
    be trusted as "already converted, skip it" rather than regenerated.
    img.load() forces the real decode (Image.open() alone is lazy and can
    miss a broken IDAT stream), so a corrupted file on disk gets caught
    and regenerated here instead of crashing X-Plane at load time."""
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def _is_reusable_texture(png_path):
    """Like _is_valid_image, but additionally excludes our own 2x2
    solid-color fallback stubs (written when a texture failed to decode)
    -- callers deciding whether to skip a REAL texture extraction need
    "is this genuinely the real thing", not just "is this a valid image",
    or a stub written on a previous/failed run would get treated as
    'already extracted' forever and the real texture would never be
    retried."""
    try:
        with Image.open(png_path) as img:
            img.load()
            return img.size != (2, 2)
    except Exception:
        return False


_REAL_TRANSPARENCY_ALPHA_THRESHOLD = 250
# Alpha (0-255) a glass/window material is forced to when its own albedo
# carries no real transparency -- deliberately low, for a genuinely clear
# pane rather than a grey sheet. Paired with forced double-sided
# (ATTR_no_cull) so the far wall/interior still draws behind it. Raise
# toward ~80-120 for a more visibly tinted look.
_GLASS_TRANSLUCENCY_FLOOR = 24


def _texture_has_real_transparency(png_path):
    """True if this (already-decoded, on-disk) texture's own alpha channel
    dips meaningfully below opaque somewhere -- the real punch-through
    pane/frame pattern a genuine glass texture has, even when its
    baseColorFactor is the fully-opaque glTF default. Used to downgrade a
    BLEND material back to OPAQUE when its texture has no real
    transparency either. getextrema() is a fast C-level PIL op, not a
    per-pixel Python loop."""
    try:
        with Image.open(png_path) as img:
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            alpha_min, _alpha_max = img.getchannel("A").getextrema()
        return alpha_min < _REAL_TRANSPARENCY_ALPHA_THRESHOLD
    except Exception:
        return False


def save_as_png(raw_bytes, out_png_path, fallback_color):
    temp_path = out_png_path.with_name(f"{out_png_path.name}.tmp_{_unique_suffix()}")
    success = False

    if decode_ktx2_bytes_to_png(raw_bytes, temp_path):
        success = True
    elif decode_dds_bytes_to_png(raw_bytes, temp_path):
        success = True
    else:
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            img = img.convert("RGBA")
            img.save(str(temp_path), "PNG")
            success = True
        except Exception:
            try:
                # Silent until now: this substitutes a flat, uniform 2x2
                # color stub for the ENTIRE texture whenever its bytes can't
                # be decoded by any path above (not KTX2/DDS, and not a
                # format PIL recognizes either -- e.g. an unsupported
                # compressed variant). For an emissive/TEXTURE_LIT slot
                # specifically, that stub becomes the object's WHOLE night
                # glow with no background/text distinction at all -- e.g. a
                # sign rendering as a single flat color instead of readable
                # text on a colored background -- which looks identical to
                # (and is easy to mistake for) the separate hue-destroying
                # brightness-scaling issue apply_emissive_factor guards
                # against. Logging it means that specific failure mode is
                # visible instead of silently masquerading as "converted
                # fine, just looks wrong".
                logger.warning(
                    f"{out_png_path.name}: texture bytes could not be decoded (not KTX2/DDS, "
                    f"and not a format PIL recognizes) -- substituting a flat {fallback_color} "
                    f"color stub for the whole texture."
                )
                img = Image.new("RGBA", (2, 2), fallback_color)
                img.save(str(temp_path), "PNG")
                success = True
            except Exception:
                success = False

    if success and temp_path.exists():
        try:
            _atomic_replace(temp_path, out_png_path)
        except OSError:
            if temp_path.exists():
                temp_path.unlink()
            success = False
    elif temp_path.exists():
        temp_path.unlink()

    return success


def texture_image_index(gltf, texture_index):
    if texture_index is None:
        return None, False
    textures = gltf.get("textures", [])
    if not isinstance(texture_index, int) or texture_index < 0 or texture_index >= len(textures):
        return None, False
    tex = textures[texture_index]

    if "source" in tex and tex["source"] is not None:
        return tex["source"], False

    exts = tex.get("extensions", {})
    for ext_name, ext_data in exts.items():
        if isinstance(ext_data, dict) and "source" in ext_data and ext_data["source"] is not None:
            return ext_data["source"], True

    return None, False


def find_base_color_texture(mat):
    if not mat or not isinstance(mat, dict):
        return None

    pbr = mat.get("pbrMetallicRoughness", {})
    if "baseColorTexture" in pbr:
        return pbr["baseColorTexture"]

    exts = mat.get("extensions", {})

    if "KHR_materials_pbrSpecularGlossiness" in exts:
        spec = exts["KHR_materials_pbrSpecularGlossiness"]
        if "diffuseTexture" in spec:
            return spec["diffuseTexture"]

    if "ASOBO_material_decal" in exts:
        decal = exts["ASOBO_material_decal"]
        if "decalColorTexture" in decal:
            return decal["decalColorTexture"]

    if "ASOBO_material_day_night" in exts:
        dn = exts["ASOBO_material_day_night"]
        if "dayNightTexture" in dn:
            return dn["dayNightTexture"]

    if "ASOBO_material_detail_map" in exts:
        detail = exts["ASOBO_material_detail_map"]
        if "detailColorTexture" in detail:
            return detail["detailColorTexture"]

    return None


def find_normal_texture(mat):
    if not mat or not isinstance(mat, dict):
        return None
    if "normalTexture" in mat:
        return mat["normalTexture"]
    exts = mat.get("extensions", {})
    if "ASOBO_material_detail_map" in exts:
        detail = exts["ASOBO_material_detail_map"]
        if "detailNormalTexture" in detail:
            return detail["detailNormalTexture"]
    return None


def find_emissive_texture(mat):
    if not mat or not isinstance(mat, dict):
        return None
    return mat.get("emissiveTexture")


# Model-name gate for synthesizing a night light out of an emissive-only
# fixture (apron pole, flood, wig-wag, street lamp) that ships no
# ASOBO_macro_light / KHR_lights_punctual node. Kept deliberately narrow --
# a lit sign, window or facade must NOT be turned into a floodlight.
_LIGHT_FIXTURE_KW = ("light", "lamp", "flood", "wigwag", "wig_wag", "guard",
                     "clearance", "beacon", "streetlamp", "apronlight", "projector",
                     "bollard")
_NOT_LIGHT_FIXTURE_KW = ("sign", "window", "glass", "screen", "billboard", "poster",
                         "banner", "logo", "decal", "atlas", "hangar", "terminal",
                         "_b_", "roof", "facade", "wall", "shelter", "canopy")


def _looks_like_light_fixture(model_name):
    n = (model_name or "").lower()
    if any(k in n for k in _NOT_LIGHT_FIXTURE_KW):
        return False
    return any(k in n for k in _LIGHT_FIXTURE_KW)


# A wig-wag / runway-guard / clearance-bar fixture: X-Plane ships a real
# built-in animated named light for it (wigwag_y1 / wigwag_y2 -- the two
# alternating heads), self-flashing and day/night gated in the sim engine
# with no plugin -- far more robust than a custom blink dataref.
_WIGWAG_KW = ("wigwag", "wig_wag", "runwayguard", "runway_guard", "rwyguard",
              "clearancebar", "clearance_bar")


def _looks_like_wigwag(model_name):
    return any(k in (model_name or "").lower() for k in _WIGWAG_KW)


# A pole/mast/apron/flood fixture throws its light at the GROUND. MSFS
# sometimes hands us the emitter axis pointing up (+Y) -- the node
# transform that would have aimed it never got folded into the direction
# vector. For these named fixture types an up-pointing spill light is
# always wrong, so force it back down. Deliberately excludes uplights /
# beacons / facade washes, which really do point up.
_DOWNLIGHT_KW = ("streetlamp", "street_lamp", "apronlight", "apron_light",
                 "polelight", "pole_light", "flood", "projector",
                 "arealight", "area_light", "baselight", "base_light", "lamp")
_NOT_DOWNLIGHT_KW = ("uplight", "up_light", "beacon", "facade", "wall", "spot_up")


def _is_downlight_fixture(model_name):
    n = (model_name or "").lower()
    if any(k in n for k in _NOT_DOWNLIGHT_KW):
        return False
    return any(k in n for k in _DOWNLIGHT_KW)


def read_gltf_animations(gltf, buffers):
    """Returns {node_idx: {"path": "translation"|"rotation"|..., "times": np.ndarray(N,),
    "values": np.ndarray(N,3 or 4)}} across every animation/channel in the
    file. Only the first channel found per node is kept (MSFS SimObjects
    animate one property per node in practice); "times" are in the glTF
    clip's own seconds, "values" are raw translation vec3 / rotation quat
    (xyzw) samples straight from the accessor, unnormalized/un-transformed.
    """
    result = {}
    for anim in gltf.get("animations", []):
        samplers = anim.get("samplers", [])
        for channel in anim.get("channels", []):
            target = channel.get("target", {})
            node_idx = target.get("node")
            path = target.get("path")
            if node_idx is None or path not in ("translation", "rotation") or node_idx in result:
                continue
            sampler_idx = channel.get("sampler")
            if sampler_idx is None or sampler_idx >= len(samplers):
                continue
            sampler = samplers[sampler_idx]
            try:
                times = read_accessor(gltf, buffers, sampler["input"]).reshape(-1)
                values = read_accessor(gltf, buffers, sampler["output"])
            except (KeyError, IndexError):
                continue
            if len(times) < 2:
                continue
            result[node_idx] = {"path": path, "times": times, "values": values}
    return result


# MSFS's per-model behavior XML expresses its animation trigger as an
# RPN-like formula string (the XML gauge/EMISSIVE_CODE language). Rather
# than a general interpreter for that whole language, this recognizes
# THREE trigger shapes: two replicable with a plain always-on X-Plane
# dataref (no plugin) -- a business-hours-style local-time window, and a
# fixed-period zulu-time modulo blink -- and one, "proximity" (MSFS's own
# "Z:VisibleRadiusBox", set when the aircraft enters the object's
# interaction radius), with no built-in X-Plane equivalent, so it needs a
# real plugin driving a custom per-object dataref from the aircraft's
# live position (see the FlyWithLua script this feeds). Anything else
# (velocity/IK-driven rigs, mission-scripted vehicles, ...) falls back to
# static geometry.
_TIME_WINDOW_RE = re.compile(
    r"LOCAL TIME,\s*Seconds\)\s*(\d+)\s*&gt;\s*\(E:LOCAL TIME,\s*Seconds\)\s*(\d+)\s*&lt;"
)
_BLINK_RE = re.compile(
    r"ZULU TIME,\s*seconds\)\s*([\d.]+)\s*%\s*([\d.]+)\s*&gt;"
)
_PROXIMITY_RE = re.compile(
    r"Z:VisibleRadiusBox,\s*Number\)\s*1\s*==\s*if\{\s*([\d.]+)\s*\}\s*els\{\s*0\s*\}"
)


def parse_time_behavior(xml_path):
    """Returns ("business_hours", open_start_sec, open_end_sec),
    ("blink", period, threshold), ("proximity", anim_length), or None, by
    pattern-matching the model's behavior XML against the three
    known-replicable trigger shapes (the third needs a plugin -- see
    _PROXIMITY_RE above -- the caller decides what to do with it)."""
    if not xml_path or not xml_path.exists():
        return None
    try:
        text = xml_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    m = _TIME_WINDOW_RE.search(text)
    if m:
        return ("business_hours", float(m.group(1)), float(m.group(2)))

    m = _BLINK_RE.search(text)
    if m:
        return ("blink", float(m.group(1)), float(m.group(2)))

    m = _PROXIMITY_RE.search(text)
    if m:
        return ("proximity", float(m.group(1)))

    # A behavior XML existing but matching none of the three known shapes
    # is expected for most models (MSFS behaviors cover far more than
    # these three), so this stays DEBUG rather than a warning -- but
    # without ANY visibility here, "why didn't this blink/business-hours
    # object animate" had no way to be answered short of grepping every
    # output .obj for its dataref by hand (confirmed necessary against a
    # real airport package: zero objects out of 3085 referenced either
    # blink dataref, and there was no log line anywhere explaining why).
    logger.debug(f"{xml_path.name}: behavior XML present but matched none of business_hours/blink/proximity")
    return None


def quat_to_axis_angle_matrix(rot3):
    """Converts a 3x3 rotation matrix (assumed orthonormal) into (axis,
    angle_degrees) via the standard trace/cross-product formula. Returns
    ((0,0,1), 0.0) for an identity/near-identity input rather than dividing
    by a near-zero sin(angle)."""
    trace = np.clip((np.trace(rot3) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(trace)
    if angle < 1e-6:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([
        rot3[2, 1] - rot3[1, 2],
        rot3[0, 2] - rot3[2, 0],
        rot3[1, 0] - rot3[0, 1],
    ])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return axis / norm, math.degrees(angle)


_EXTERNAL_TEXTURE_EXTENSIONS = {".png", ".dds", ".jpg", ".jpeg", ".tif", ".tiff", ".ktx2", ".bmp", ".tga"}
_EXTERNAL_TEXTURE_INDEX_CACHE = {}  # str(one dir) -> {clean_stem: Path}, memoized per worker process


def _external_texture_roots(external_textures_dir):
    """Normalizes external_textures_dir into an ordered list of individual
    directories -- accepted either as one path or as an iterable of paths
    (the source package's own root -- a texture can be shipped somewhere
    the author's own relative URI doesn't point at, e.g. a stale dev-
    machine path -- AND, when configured, the MSFS base-game install
    root, searched in that
    priority order so a package's own bundled texture always wins over a
    same-named base-game one)."""
    if external_textures_dir is None:
        return []
    if isinstance(external_textures_dir, (str, Path)):
        return [external_textures_dir]
    return list(external_textures_dir)


def _external_texture_index(one_dir):
    """{clean_stem: Path} for every texture-looking file under one_dir,
    built once and memoized module-level (persists across every convert()
    call in this worker process) -- one_dir can be an entire MSFS install
    root, tens of GB / 100k+ files, so this avoids a full rglob per
    missing texture."""
    key = str(one_dir)
    index = _EXTERNAL_TEXTURE_INDEX_CACHE.get(key)
    if index is None:
        index = {}
        try:
            for p in Path(one_dir).rglob("*"):
                if p.suffix.lower() in _EXTERNAL_TEXTURE_EXTENSIONS and p.is_file():
                    index.setdefault(clean_texture_stem(p.name), p)
        except OSError:
            pass
        _EXTERNAL_TEXTURE_INDEX_CACHE[key] = index
    return index


def _find_in_external_texture_roots(external_textures_dir, stem):
    """Looks up stem across every root in external_textures_dir, in order
    -- returns the first match (source package root before base-game
    install root, matching _external_texture_roots' own priority)."""
    for root in _external_texture_roots(external_textures_dir):
        match = _external_texture_index(root).get(stem)
        if match is not None:
            return match
    return None


def extract_image(gltf, buffers, image_index, glb_path, textures_dir, external_textures_dir, cache, fallback_color,
                   allow_dds_passthrough=False):
    """allow_dds_passthrough: when True AND the source bytes are already a
    real DDS file, write them straight through as a .dds instead of
    decoding to PNG. X-Plane's OBJ8 TEXTURE line supports .dds natively,
    so decoding is pure waste when nothing about the texture needs to
    change afterward -- CONFIRMED REAL GAP found comparing this project's
    output against a different MSFS->X-Plane converter's: every one of
    OUR textures paid a full decode+re-encode cost even when unmodified,
    a real (~7x observed on one texture) size/VRAM penalty for zero
    quality gain, while the other tool passes DDS through unchanged in
    the common case and only re-encodes when a real modification (a
    baked color/alpha/emissive factor) needs to be applied.

    The CALLER decides this, not this function: it's the caller (the
    material-processing loop in convert()) that knows whether apply_
    color_factor/apply_alpha_factor/apply_emissive_factor will run on
    this specific texture slot afterward -- those functions edit pixel
    data and need a real decoded PNG to work on, so passthrough must stay
    False whenever any of them might still run. The normal-map slot never
    gets any such post-processing, so it can always pass True."""
    if image_index in cache:
        return cache[image_index]

    images = gltf.get("images", [])
    if image_index is None or image_index >= len(images):
        return None

    image = images[image_index]
    name_hint = image.get("name")

    if name_hint:
        base_stem = clean_texture_stem(name_hint)
    elif image.get("uri"):
        base_stem = clean_texture_stem(image["uri"])
    else:
        base_stem = f"texture_{image_index}"

    base_stem = sanitize_name(base_stem)
    out_name = f"{base_stem}.png"
    out_png_path = textures_dir / out_name
    out_dds_path = textures_dir / f"{base_stem}.dds"

    with _TEXTURE_LOCK:
        # Checked BEFORE the .png check, not after: a shared base texture
        # (the common MSFS case -- one neutral texture reused across many
        # differently-tinted/branded objects, or across a BLEND-alpha
        # material next to an OPAQUE one) can have BOTH a compact .dds
        # (either Step 2's own bulk KTX2 pre-decode, main.py's
        # decode_or_repackage_ktx2, or an earlier passthrough-eligible
        # call here) AND a fully-decoded .png (from some OTHER material
        # needing real pixel access) on disk at once. If the .png check
        # ran first, ANY single decode-needing consumer of a shared
        # texture would permanently "poison" every later passthrough-
        # eligible consumer into reusing that .png too, even though it
        # would have been perfectly happy with the already-available
        # .dds -- confirmed on a real EGLC conversion: 1276 textures got
        # both a .dds and a .png, and NONE of the .dds files ended up
        # referenced by any compiled .obj (0/1276), a pure ~2.7GB waste.
        # This check makes the CURRENT caller's own allow_dds_passthrough
        # decide first, independent of what any other caller needed.
        if allow_dds_passthrough and out_dds_path.exists() and out_dds_path.stat().st_size > 100:
            cache[image_index] = out_dds_path.name
            return out_dds_path.name
        if out_png_path.exists() and out_png_path.stat().st_size > 100 and _is_reusable_texture(out_png_path):
            cache[image_index] = out_name
            return out_name

    def _write_raw(raw_bytes, dest_path):
        """Shared by the bufferView and data-uri branches below: DDS
        passthrough when allowed and the bytes really are DDS, else the
        existing decode-to-PNG path. Returns the output filename actually
        written, or None on failure -- same contract save_as_png had."""
        if allow_dds_passthrough and raw_bytes[:4] == b"DDS ":
            temp_path = dest_path.with_name(f"{out_dds_path.name}.tmp_{_unique_suffix()}")
            temp_path.write_bytes(raw_bytes)
            _atomic_replace(temp_path, out_dds_path)
            return out_dds_path.name
        return out_name if save_as_png(raw_bytes, out_png_path, fallback_color) else None

    # Tracks WHY execution fell through to the final fallback-stub write
    # below, for the warning there -- several genuinely different failure
    # shapes (embedded-bytes decode failure, data-URI decode failure, an
    # external URI that couldn't be found anywhere, or a malformed image
    # entry with none of the three) all land at that same final write, and
    # previously logged nothing at all, so there was no way to tell which
    # of them actually happened short of inspecting output pixel data by
    # hand -- confirmed necessary against a real airport package where a
    # character's suit/boots textures came out as flat 2x2 placeholders
    # with zero explanation anywhere.
    failure_reason = "image entry has neither bufferView nor uri"

    if "bufferView" in image:
        bv = gltf["bufferViews"][image["bufferView"]]
        buf_idx = bv.get("bufferIndex", bv.get("buffer", 0))
        offset = bv.get("byteOffset", 0)
        length = bv["byteLength"]
        raw = buffers[buf_idx][offset : offset + length]
        written = _write_raw(raw, out_png_path)
        if written:
            cache[image_index] = written
            return written
        failure_reason = "embedded bufferView image data failed to decode"

    elif image.get("uri", "").startswith("data:"):
        _, b64 = image["uri"].split(",", 1)
        raw = base64.b64decode(b64)
        written = _write_raw(raw, out_png_path)
        if written:
            cache[image_index] = written
            return written
        failure_reason = "embedded data: URI image data failed to decode"

    elif image.get("uri"):
        uri = image["uri"]
        uri_path = Path(uri)
        glb_parent = Path(glb_path).parent

        candidates = [
            glb_parent / uri,
            glb_parent / uri_path.name,
            glb_parent / "texture" / uri_path.name,
            glb_parent / "TEXTURE" / uri_path.name,
            glb_parent.parent / "texture" / uri_path.name,
            glb_parent.parent / "TEXTURE" / uri_path.name,
            # Step 2's own bulk KTX2 pass (main.py) pre-decodes every
            # package texture into textures_dir BEFORE any model gets
            # converted, writing whichever of these two extensions it
            # actually produced (repackaged .dds when possible, decoded
            # .png otherwise -- see decode_or_repackage_ktx2) -- both need
            # to be checked here, not just the .png one, or a pre-decoded
            # .dds sitting right next to this exact texture never gets
            # found at all and a redundant, wasted full re-decode gets
            # substituted (or worse, the flat-color fallback stub if that
            # re-decode also fails).
            textures_dir / out_dds_path.name,
            textures_dir / out_name,
        ]

        match = None
        for c in candidates:
            if not (c.exists() and c.is_file()):
                continue
            # out_png_path already failed the "already converted" check
            # above, so if this candidate IS out_png_path itself, it must
            # not be trusted as a match without re-checking validity --
            # both copy branches below skip self-copies, so a match here
            # without the check would report success on a still-broken file.
            if c.resolve() == out_png_path.resolve() and not _is_valid_image(c):
                continue
            match = c
            break

        if not match and external_textures_dir:
            match = _find_in_external_texture_roots(external_textures_dir, base_stem)

        if match:
            if match.suffix.lower() == ".png" and match.resolve() != out_png_path.resolve():
                temp_path = out_png_path.with_name(f"{out_png_path.name}.tmp_{_unique_suffix()}")
                shutil.copyfile(match, temp_path)
                _atomic_replace(temp_path, out_png_path)
                cache[image_index] = out_name
                return out_name
            elif allow_dds_passthrough and match.suffix.lower() == ".dds":
                # Usually a self-match: Step 2's own bulk KTX2 pass
                # (main.py) already wrote exactly this file at exactly
                # out_dds_path before any model conversion started, so
                # this is normally just a reference, not a copy -- only
                # copy when the match was found somewhere else (e.g. an
                # external texture root).
                if match.resolve() != out_dds_path.resolve():
                    temp_path = out_dds_path.with_name(f"{out_dds_path.name}.tmp_{_unique_suffix()}")
                    shutil.copyfile(match, temp_path)
                    _atomic_replace(temp_path, out_dds_path)
                cache[image_index] = out_dds_path.name
                return out_dds_path.name
            elif match.resolve() != out_png_path.resolve():
                written = _write_raw(match.read_bytes(), out_png_path)
                if written:
                    cache[image_index] = written
                    return written
            else:
                cache[image_index] = out_name
                return out_name
        failure_reason = (
            f"uri '{uri}' not found alongside the model, its texture/TEXTURE "
            f"subfolders, or external_textures_dir"
        )

    if not (out_png_path.exists() and _is_valid_image(out_png_path)):
        with _TEXTURE_LOCK:
            if not (out_png_path.exists() and _is_valid_image(out_png_path)):
                logger.warning(
                    f"{Path(glb_path).name}: texture '{image.get('uri') or name_hint or base_stem}' "
                    f"could not be resolved ({failure_reason}) -- substituting a flat placeholder"
                )
                img = Image.new("RGBA", (2, 2), fallback_color)
                temp_path = out_png_path.with_name(f"{out_png_path.name}.tmp_{_unique_suffix()}")
                img.save(str(temp_path), "PNG")
                _atomic_replace(temp_path, out_png_path)

    cache[image_index] = out_name
    return out_name


def apply_color_factor(png_path, factor_rgb_255):
    """Multiplies a texture's RGB channels by a material's own
    baseColorFactor (glTF spec: final color = baseColorTexture.rgb *
    baseColorFactor.rgb) -- alpha is left untouched (apply_alpha_factor
    handles that separately). MSFS routinely reuses one shared, neutral
    texture across many differently-colored/branded objects, relying on
    this per-material multiply for the actual visible paint color (e.g. a
    generic wall texture tinted per building) -- unlike
    apply_emissive_factor's brightness-preserving stretch, this is a
    plain per-channel multiply, the spec-correct behavior for
    baseColorFactor's normal [0,1] range."""
    if factor_rgb_255 == (255, 255, 255):
        return png_path.name

    # Always .png regardless of png_path's own suffix: this always
    # decodes+modifies+re-saves as PNG below (Image.fromarray(...).save
    # (temp_path, "PNG")), so a caller that passes a .dds path (a
    # texture slot whose OWN allow_dds_passthrough was True for an
    # unrelated reason, e.g. apply_emissive_factor's cross-slot reuse of
    # the base-color texture's own extracted name) must not get a .dds-
    # NAMED file back containing real PNG bytes -- CONFIRMED REAL BUG:
    # X-Plane's Log.txt reported "we are missing the texture" for
    # several files that turned out to be exactly this (a real PNG
    # sitting under a stale/misleading .dds filename).
    tag = "_".join(str(c) for c in factor_rgb_255)
    new_name = f"{png_path.stem}_cf{tag}.png"
    new_path = png_path.with_name(new_name)

    with _TEXTURE_LOCK:
        if new_path.exists() and new_path.stat().st_size > 100 and _is_valid_image(new_path):
            return new_name

    try:
        img = Image.open(png_path).convert("RGBA")
        arr = np.array(img).astype(np.float32)
        for c in range(3):
            arr[..., c] = (arr[..., c] * (factor_rgb_255[c] / 255.0)).clip(0, 255)

        temp_path = new_path.with_name(f"{new_name}.tmp_{_unique_suffix()}")
        Image.fromarray(arr.astype(np.uint8), "RGBA").save(temp_path, "PNG")
        _atomic_replace(temp_path, new_path)
        return new_name
    except Exception:
        return png_path.name


def apply_alpha_factor(png_path, alpha_255):
    # Always .png -- see apply_color_factor's own comment on why.
    new_name = f"{png_path.stem}_a{alpha_255}.png"
    new_path = png_path.with_name(new_name)

    with _TEXTURE_LOCK:
        if new_path.exists() and new_path.stat().st_size > 100 and _is_valid_image(new_path):
            return new_name

    try:
        img = Image.open(png_path).convert("RGBA")
        arr = np.array(img)
        arr[..., 3] = (arr[..., 3].astype(np.float32) * (alpha_255 / 255.0)).clip(0, 255).astype(np.uint8)

        temp_path = new_path.with_name(f"{new_name}.tmp_{_unique_suffix()}")
        Image.fromarray(arr, "RGBA").save(temp_path, "PNG")
        _atomic_replace(temp_path, new_path)
        return new_name
    except Exception:
        return png_path.name


def apply_emissive_factor(png_path, factor_rgb):
    """MSFS pairs an emissive texture with an "emissiveFactor" that's
    routinely far outside glTF's normal [0,1]-per-channel range -- values
    like [100, 100, 100], even [350, 350, 350], show up on real materials
    (light atlases, illuminated taxiway signs), meaning the texture itself
    is authored dim and relies entirely on that multiplier to reach its
    intended on-screen brightness in MSFS's own HDR renderer (which then
    tone-maps the over-bright result back down). X-Plane's TEXTURE_LIT has
    no such tone-mapping and uses raw 0-255 values directly, so a literal
    multiply-and-clip by a factor like 350 saturates almost everything to
    white (e.g. a sign's text and background both clip to the same white).

    So the raw factor is only a TRIGGER; the actual scale is a contrast-
    preserving stretch derived from the image's own brightness
    distribution, applied PER PIXEL proportionally across all three
    channels together (capped so each pixel's brightest channel stays
    <=255) rather than one global multiply with independent per-channel
    clipping. That keeps each pixel's R:G:B ratio -- and therefore hue --
    intact regardless of how much its brightness increases; independent
    per-channel clipping would drift background/text colors toward
    different corners and destroy hue."""
    if all(f < 1.5 for f in factor_rgb):
        return png_path.name

    # Always .png -- see apply_color_factor's own comment on why. Matters
    # even more here: this is sometimes called on builder.texture_name,
    # the BASE COLOR texture's own already-extracted name (see convert()'s
    # "no emissive texture, base colour emits" fallback) -- a texture
    # slot whose OWN allow_dds_passthrough decision has nothing to do
    # with whether THIS unrelated emissive synthesis needs to bake it.
    tag = "_".join(f"{f:.2f}" for f in factor_rgb).replace(".", "p")
    new_name = f"{png_path.stem}_ef{tag}.png"
    new_path = png_path.with_name(new_name)

    with _TEXTURE_LOCK:
        if new_path.exists() and new_path.stat().st_size > 100 and _is_valid_image(new_path):
            return new_name

    try:
        img = Image.open(png_path).convert("RGBA")
        arr = np.array(img).astype(np.float32)
        rgb = arr[..., :3]

        # Reference brightness: a high percentile of the per-pixel max
        # channel value, not the literal max (a couple of stray hot pixels
        # -- compression artifacts, a single bright highlight -- shouldn't
        # single-handedly decide the whole image's scale).
        pixel_max = rgb.max(axis=2)
        reference = float(np.percentile(pixel_max, 99.0))
        if reference < 1.0:
            return png_path.name  # effectively all-black texture; nothing to scale

        scale = min(255.0 / reference, 20.0)
        if scale <= 1.05:
            return png_path.name

        # Per-pixel cap: reduce the multiplier just enough, per pixel, that
        # THIS pixel's own brightest channel lands at 255 instead of
        # clipping past it -- applied identically to all 3 channels of a
        # given pixel, so R:G:B ratio (hue) never shifts. Pixels whose
        # brightest channel stays under 255 at the full global `scale`
        # (the common case -- the reference percentile already sits near
        # 255 by construction) are completely unaffected by this cap.
        pixel_max = rgb.max(axis=2, keepdims=True)
        effective_scale = np.minimum(scale, 255.0 / np.maximum(pixel_max, 1.0))
        arr[..., :3] = (rgb * effective_scale).clip(0, 255)
        arr = arr.astype(np.uint8)

        temp_path = new_path.with_name(f"{new_name}.tmp_{_unique_suffix()}")
        Image.fromarray(arr, "RGBA").save(temp_path, "PNG")
        _atomic_replace(temp_path, new_path)
        return new_name
    except Exception:
        return png_path.name


def _vertex_positions_are_subset(small_verts, big_verts, eps=0.05):
    """True iff every position in `small_verts` has a close (<=eps, metres)
    match somewhere in `big_verts` -- a package-agnostic proxy for "this
    builder's geometry is a duplicate (or a duplicated portion) of that
    one's", used to detect a redundant reverse-wound double-sided-via-
    duplicate-geometry copy (see its caller). A coarse spatial hash of
    `big_verts` keeps this close to O(n) instead of O(n*m) for the
    larger meshes this also needs to handle cheaply."""
    if not small_verts or not big_verts:
        return False
    buckets = {}
    for x, y, z in big_verts:
        buckets.setdefault((round(x / eps), round(y / eps), round(z / eps)), []).append((x, y, z))
    for x, y, z in small_verts:
        cx, cy, cz = round(x / eps), round(y / eps), round(z / eps)
        if not any(
            abs(x - bx) < eps and abs(y - by) < eps and abs(z - bz) < eps
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            for bx, by, bz in buckets.get((cx + dx, cy + dy, cz + dz), ())
        ):
            return False
    return True


def convert(glb_path, objects_dir, textures_dir, external_textures_dir, pitch=0.0, yaw=0.0, roll=0.0,
            disable_proximity_animation=False):
    global _EXPORTED_COUNT
    glb_path = Path(glb_path)
    objects_dir = Path(objects_dir)
    textures_dir = Path(textures_dir)
    objects_dir.mkdir(parents=True, exist_ok=True)
    textures_dir.mkdir(parents=True, exist_ok=True)

    try:
        gltf, bin_chunk = parse_model_file(glb_path)
        buffers = load_buffers(gltf, bin_chunk, glb_path)
        # gltf_animations has to be read before world_transforms so any
        # animated node's REST geometry can be baked from its animation's
        # own values[0] instead of its raw authored TRS -- see
        # node_local_matrix_at_rest for why that distinction matters.
        gltf_animations = read_gltf_animations(gltf, buffers)
        world_transforms = collect_world_transforms(gltf, gltf_animations)
        node_parents = collect_node_parents(gltf)
    except Exception as e:
        logger.error(f"Failed parsing GLB structure for {glb_path.name}: {e}")
        return []

    # Some MSFS SimObjects (jetways, GSE with IK/velocity-driven rigs,
    # proximity-triggered doors) need live gameplay state X-Plane has no
    # dataref equivalent for. A business-hours-style local-time window and
    # a fixed-period zulu-time blink map cleanly onto real X-Plane time
    # datarefs though, detected from the model's own behavior XML (same
    # folder, LOD suffix stripped); anything else falls through untouched.
    xml_path = glb_path.parent / f"{re.sub(r'_LOD[0-9]+$', '', glb_path.stem)}.xml"
    time_behavior = parse_time_behavior(xml_path)
    if disable_proximity_animation and time_behavior is not None and time_behavior[0] in ("proximity", "business_hours"):
        # Both "proximity" and "business_hours" drive mesh rotation/
        # translation (a door/barrier/gate) -- never ATTR_light_level/
        # emissive content, which is exclusively "blink"'s own path and
        # stays untouched by this toggle. "business_hours" would work
        # fine standalone (reads a real stock X-Plane time dataref), but
        # both are folded into the same static toggle: a reliably closed,
        # correctly-anchored door beats a sometimes-moving one. Nulling
        # time_behavior here routes these nodes through the same "no
        # animation detected" path as a model with no behavior XML at all.
        time_behavior = None

    skins = gltf.get("skins", [])

    def find_animated_ancestor(node_idx):
        """Walks node_idx up through node_parents looking for the nearest
        node (itself included) with a glTF animation channel -- the
        animated joint driving a mesh node's geometry is very often that
        mesh node's parent (or higher), not the mesh node itself.

        Falls back to a node's SKIN joints (gltf["skins"][node["skin"]]
        ["joints"]) if the rigid parent-chain walk finds nothing: a
        skinned mesh node (JOINTS_0/WEIGHTS_0 vertex attributes, real
        per-vertex weighted deformation across possibly many joints) isn't
        necessarily a scene-graph descendant of the joint that actually
        drives it at all -- skinned mesh nodes are commonly siblings of
        their armature, deformed via skin.joints rather than node.children
        parentage -- so the rigid walk above returns nothing for them,
        and a hinged door authored this way (plausible for a large,
        complex building/jet-bridge asset, less so for a small simple
        prop) got no ANIM_ block and no rest-pose correction at all.

        This does NOT implement real skinning (no JOINTS_0/WEIGHTS_0
        weight blending across multiple joints) -- it's a scoped
        approximation: treat the mesh as rigidly driven by the FIRST
        animated joint in the skin, which is exactly correct for a door
        whose skin only exists for tooling reasons and is really driven
        by one dominant hinge joint, and only an approximation (better
        than the previous "no correction at all") for anything genuinely
        multi-joint-weighted.

        Returns (ancestor_node_idx, skin_joint_position). skin_joint_position
        is None when the rigid parent-chain walk found the ancestor (the
        common case -- world_transforms[node_idx] already correctly
        incorporates that ancestor's rest pose through ordinary parentage,
        nothing more to do), or the joint's own index within
        skins[skin_idx]["joints"] when the skin fallback below is what
        actually found it -- the caller uses that position to look up the
        matching inverseBindMatrices entry and correct the mesh's own
        static world transform (see the call site), which used to be a
        documented, un-fixed gap: world_transforms[mesh_node_idx] was
        still built purely from the mesh's own node.children parent
        chain, with no knowledge of skin.joints at all, so a skinned
        door's static geometry rendered at whatever pose that unrelated
        parent chain implied -- confirmed as the real cause of a real
        door rendering permanently open regardless of the (correctly
        computed) ANIM_ close delta."""
        cur = node_idx
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            if cur in gltf_animations:
                return cur, None
            cur = node_parents.get(cur)

        skin_idx = gltf["nodes"][node_idx].get("skin") if 0 <= node_idx < len(gltf["nodes"]) else None
        if skin_idx is not None and 0 <= skin_idx < len(skins):
            for joint_pos, joint_idx in enumerate(skins[skin_idx].get("joints", [])):
                if joint_idx in gltf_animations:
                    return joint_idx, joint_pos
        return None, None

    _inv_bind_cache = {}

    def _skin_inverse_bind_matrix(skin_idx, joint_pos):
        """The joint's own inverseBindMatrices entry (mesh-local -> joint
        space), so a skin-fallback-driven mesh's rest-pose world transform
        can be built as joint_world_at_rest @ inverse_bind -- the missing
        half of find_animated_ancestor's skin fallback (see its own
        docstring). None if the skin has no inverseBindMatrices accessor
        at all (some exporters omit it when every joint's bind pose is
        identity) -- callers treat that the same as an identity matrix."""
        if skin_idx not in _inv_bind_cache:
            skin = skins[skin_idx]
            acc = skin.get("inverseBindMatrices")
            if acc is None:
                _inv_bind_cache[skin_idx] = None
            else:
                mats = read_accessor(gltf, buffers, acc)
                _inv_bind_cache[skin_idx] = [
                    mats[i].reshape(4, 4, order="F") for i in range(mats.shape[0])
                ]
        cached = _inv_bind_cache[skin_idx]
        if cached is None or joint_pos >= len(cached):
            return np.eye(4)
        return cached[joint_pos]

    # Move the pitch/yaw/roll rotation matrix construction up front so the
    # file-wide flatness scan below can judge flatness in the SAME final
    # coordinate frame the geometry actually gets written in (matters only
    # if a non-zero rotation is passed; harmless no-op otherwise).
    pitch, yaw, roll = float(pitch), float(yaw), float(roll)
    p, y, r = math.radians(pitch), math.radians(yaw), math.radians(roll)

    rx = np.array([
        [1, 0, 0],
        [0, math.cos(p), -math.sin(p)],
        [0, math.sin(p), math.cos(p)]
    ])
    ry = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]
    ])
    rz = np.array([
        [math.cos(r), -math.sin(r), 0],
        [math.sin(r), math.cos(r), 0],
        [0, 0, 1]
    ])

    global_rot_matrix = rz @ ry @ rx

    def apply_global_rotation(arr):
        if arr is not None and len(arr):
            return arr @ global_rot_matrix.T
        return arr

    # File-wide flatness check: does this .glb contain nothing but flat,
    # "almost one dimensional" content (ground markings/decals), with no
    # genuine 3D structure anywhere in it? This is judged once for the
    # WHOLE file, not per mesh -- a file that qualifies gets every vertex
    # in every object compressed onto the exact same level: the file's own
    # dominant/majority elevation (not forced to 0 -- if that real
    # elevation is genuinely non-zero, e.g. a roof or bridge deck, it's
    # preserved). A file that has even a little real 3D geometry mixed in
    # is left completely untouched instead of guessing which parts are
    # safe to flatten.
    flat_fraction_threshold = 0.98
    # How close a material's own flat elevation must sit to the file's
    # overall ground-level reference to qualify for the per-material
    # near-ground-flat fallback below (see builder.is_near_ground_flat).
    # Comfortably covers the confirmed real case (~1.5m, a baked authoring
    # offset on an otherwise-ground-level tile material) while staying
    # well under typical ceiling/roof/deck heights, so a genuinely
    # elevated flat surface (a roof, a bridge top) still can't qualify
    # just because it's flat.
    _NEAR_GROUND_FLAT_TOLERANCE_M = 2.0
    # Deliberately looser than flat_fraction_threshold (0.98) -- see the
    # long comment where this is consumed (builder.is_near_ground_flat)
    # for the real measured numbers this was picked from.
    _NEAR_GROUND_FLAT_FRACTION_THRESHOLD = 0.90
    file_flat_fraction, file_reference_height, file_max_radius, file_max_horizontal_radius, node_flatness_stats, material_flatness_stats = compute_file_flatness_and_reference(gltf, buffers, world_transforms)
    file_is_flat_only = file_flat_fraction >= flat_fraction_threshold and file_reference_height is not None

    # TILTED rotates the WHOLE rigid object around its local origin to
    # match the terrain normal X-Plane samples at ITS SINGLE placement
    # point -- it can only ever correct a genuine local SLOPE, never an
    # anchor-elevation offset (a rotation can't move its own origin, so
    # if the anchor's own authored elevation just doesn't match X-Plane's
    # real terrain there, every part of the object stays wrong by that
    # same amount no matter how well the single-point-sampled tilt is
    # fit). That's exactly the more common real problem (confirmed: an
    # un-rotated object sitting at a flat-out wrong height, not a tilted
    # one) and terrain_fit.py's own uniform vertical SHIFT already exists
    # to fix it correctly, with real multi-point sampling + outlier
    # rejection -- CONFIRMED bug this replaces: the shift used to be
    # gated to only large objects (>=300m2), leaving TILTED as the ONLY
    # correction for anything smaller, silently failing on exactly the
    # anchor-offset case it can't fix. TILTED is never written anywhere
    # now; terrain_fit.py's shift applies to every qualifying rigid group
    # regardless of size instead (see its own module docstring).
    #
    # file_is_rigid is still needed on its own: it also gates horizontal
    # re-centering (a rigid/non-draped-only step -- draped multi-layer
    # stacks must NOT be independently re-centered or previously-
    # coincident layers visibly separate), unrelated to TILTED itself.
    file_is_rigid = not file_is_flat_only

    if file_is_flat_only:
        implausible_elevation_limit = 100.0
        if file_reference_height < 0.0:
            # X-Plane / DSF placement has no per-object AGL or vertical
            # -offset field -- Y=0 is pinned to the compiled terrain mesh,
            # so a dominant elevation below that can't be represented even
            # though we're otherwise preserving real elevation here.
            logger.info(
                f"{glb_path.name}: dominant level {file_reference_height:.4f}m is below "
                f"ground -- clamping reference to y=0"
            )
            file_reference_height = 0.0
        elif abs(file_reference_height) > implausible_elevation_limit:
            # Ground-overlay layers (markings, stains, tile seams) are
            # placed at AGL alt=0 by their own BGL/SPB record, flush with
            # the ground regardless of local mesh coordinates. Some
            # exports lose track of a sane local origin and come back with
            # a "dominant" reference height in the hundreds/thousands of
            # meters -- not a real elevation, an export artifact that
            # would otherwise silently teleport the layer away from its
            # siblings. A genuinely elevated flat feature (rooftop
            # marking, bridge deck) stays comfortably under this bound.
            logger.info(
                f"{glb_path.name}: dominant level {file_reference_height:.4f}m is not a "
                f"plausible elevation for flat ground-overlay content -- clamping reference "
                f"to y=0 instead of teleporting the whole file there"
            )
            file_reference_height = 0.0
        logger.info(
            f"{glb_path.name}: {file_flat_fraction*100:.2f}% of triangles are flat -- "
            f"treating as an all-flat file, compressing every object to its dominant "
            f"level y={file_reference_height:.4f}m"
        )
    else:
        logger.info(
            f"{glb_path.name}: only {file_flat_fraction*100:.2f}% of triangles are flat -- "
            f"contains real 3D geometry, leaving heights untouched"
        )

    # Flatness/draping is classified per WHOLE FILE (file_is_flat_only/
    # file_reference_height above), not per node: splitting one material
    # by per-node flatness can cut what's really one continuous real-world
    # surface into a draped half (weldable by draped_merge) and a rigid
    # half that can never weld against it, leaving a seam that wouldn't
    # otherwise exist. The whole-file verdict is coarser but keeps every
    # node in one file on the same side of the flat/rigid line.

    model_name = glb_path.stem
    builders = {}
    image_cache = {}
    clipped_vertex_count = 0
    clipped_builder_names = set()

    def safe_normal_matrix(world4):
        m3 = world4[:3, :3]
        try:
            return np.linalg.inv(m3).T
        except np.linalg.LinAlgError:
            return m3

    # MSFS attaches point/spot lights (edge lights, floodlights, beacons)
    # as a plain node-level "ASOBO_macro_light" extension rather than
    # geometry -- these nodes carry no "mesh" key, so the main per-mesh
    # loop below never sees them. Collected here in their own pass,
    # using the same world_transforms + apply_global_rotation machinery
    # the vertex positions use.
    # KHR_lights_punctual is the standard glTF light extension (a top-
    # level array of definitions, referenced by index) -- used as a
    # fallback when ASOBO_macro_light is absent, for third-party/
    # community content that doesn't use MSFS's own extension. Never
    # both at once, so a light never gets double-counted.
    khr_light_defs = gltf.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])

    light_entries = []
    for node_idx, node in enumerate(gltf.get("nodes", [])):
        macro_light = node.get("extensions", {}).get("ASOBO_macro_light")
        khr_light_ref = node.get("extensions", {}).get("KHR_lights_punctual", {}).get("light") if macro_light is None else None
        khr_light = khr_light_defs[khr_light_ref] if khr_light_ref is not None and 0 <= khr_light_ref < len(khr_light_defs) else None
        if macro_light is None and khr_light is None:
            continue
        base_world = world_transforms.get(node_idx, np.eye(4))
        pos_world = (base_world @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        pos_world = apply_global_rotation(pos_world.reshape(1, 3))[0]

        # Aim direction: glTF's convention (which ASOBO's own node-based
        # lights follow too, based on the tested taxi-edge-light fixture)
        # is that a light points down its local -Z axis by default: rotate
        # that by the node's own world rotation (scale stripped out) plus
        # the file's global pitch/yaw/roll.
        rot3 = base_world[:3, :3]
        col_norms = np.linalg.norm(rot3, axis=0)
        col_norms[col_norms == 0] = 1.0
        dir_world = (rot3 / col_norms) @ np.array([0.0, 0.0, -1.0])
        dir_world = apply_global_rotation(dir_world.reshape(1, 3))[0]
        dir_norm = np.linalg.norm(dir_world)
        dir_world = dir_world / dir_norm if dir_norm > 1e-9 else np.array([0.0, 1.0, 0.0])

        if macro_light is not None:
            color = macro_light.get("color", [1.0, 1.0, 1.0])
            cone_angle = macro_light.get("cone_angle", 360.0)
            intensity = macro_light.get("intensity", 1.0)
            day_night_cycle = bool(macro_light.get("day_night_cycle", False))
            flash_frequency = macro_light.get("flash_frequency", 0.0) or 0.0
        else:
            # KHR_lights_punctual: color/intensity sit directly on the
            # light definition (KHR has no day/night or flash concept at
            # all -- always full-on, matching this project's own prior
            # ASOBO_macro_light default before day_night_cycle/
            # flash_frequency were read). "point"/"directional" have no
            # cone at all (cone_angle=360, matching ASOBO's own
            # omnidirectional convention); "spot" carries its half-angle
            # in radians as outerConeAngle -- double and convert to
            # degrees for ASOBO's full-angle-in-degrees convention.
            color = khr_light.get("color", [1.0, 1.0, 1.0])
            intensity = khr_light.get("intensity", 1.0)
            if khr_light.get("type") == "spot":
                outer = khr_light.get("spot", {}).get("outerConeAngle", math.pi / 4.0)
                cone_angle = math.degrees(outer) * 2.0
            else:
                cone_angle = 360.0
            day_night_cycle = False
            flash_frequency = 0.0

        light_entries.append({
            "pos": pos_world,
            "dir": dir_world,
            "color": color,
            "cone_angle": cone_angle,
            "intensity": intensity,
            # Real ASOBO_macro_light fields (per the MSFS SDK schema) --
            # previously read nowhere, so every light was written as an
            # unconditional "NULL"-dataref (always full-on, day and night
            # alike). day_night_cycle=true means the light should follow
            # the sim's own day/night state (edge lights, apron floods,
            # most airport ground lighting); flash_frequency>0 means it
            # blinks (beacons, some obstruction lights) -- see the dataref
            # selection below, at the LIGHT_SPILL_CUSTOM write site, for
            # how these two combine.
            "day_night_cycle": day_night_cycle,
            "flash_frequency": flash_frequency,
            # flash_duration/flash_phase (per-light duty-cycle and phase
            # offset) and has_symmetry (symmetric vs asymmetric emission)
            # have no reachable X-Plane LIGHT_SPILL_CUSTOM equivalent --
            # its dataref slot only carries a single shared 0..1 curve
            # (see msfs2xp_night_blink.lua) and its cone is inherently
            # symmetric around the aim axis. Not wired; every flashing
            # light in a package shares one common phase/shape, same
            # accepted simplification as the companion blink script's
            # TEXTURE_LIT case.
        })

    for node_idx, node in enumerate(gltf.get("nodes", [])):
        if "mesh" not in node:
            continue

        # The node actually carrying the glTF animation channel that drives
        # this mesh is often an ancestor joint/pivot (e.g. a barrier's
        # hinge empty), not this mesh node itself -- see
        # find_animated_ancestor. Resolved once per mesh node, reused by
        # every primitive/material below. Computed BEFORE base_world since
        # a skin-fallback match corrects base_world itself (see below).
        anim_ancestor_idx, skin_joint_pos = find_animated_ancestor(node_idx)
        anim_ancestor_node = gltf["nodes"][anim_ancestor_idx] if anim_ancestor_idx is not None else None
        anim_ancestor_world = world_transforms.get(anim_ancestor_idx, np.eye(4)) if anim_ancestor_idx is not None else np.eye(4)

        if skin_joint_pos is not None:
            # Skin-fallback case: this mesh's own node.children parent
            # chain has no relation to the joint actually driving it, so
            # world_transforms[node_idx] is meaningless for it -- rebuild
            # its rest-pose world transform from the joint's own rest
            # transform instead, exactly as a true rigid child of that
            # joint would resolve (see find_animated_ancestor's docstring).
            skin_idx = node["skin"]
            inv_bind = _skin_inverse_bind_matrix(skin_idx, skin_joint_pos)
            base_world = anim_ancestor_world @ inv_bind
        else:
            base_world = world_transforms.get(node_idx, np.eye(4))

        instance_mats = node_instance_matrices(gltf, buffers, node)
        worlds = [base_world @ im for im in instance_mats] if instance_mats else [base_world]

        mesh = gltf["meshes"][node["mesh"]]

        for world in worlds:
            # Detect negative determinant (mesh mirrored inside MSFS) so
            # winding order can be reversed to match -- otherwise it
            # renders inside-out. Computed PER INSTANCE from `world`
            # (base_world @ this instance's own matrix), not once per node
            # from base_world alone: normal_mat below is already correctly
            # computed per-instance, so a node using
            # EXT_mesh_gpu_instancing where one instance has its own
            # negative-scale mirror transform (a real technique for e.g.
            # "left-facing vs right-facing sign" variants of one base
            # asset) used to get correctly-flipped normals but STALE,
            # un-flipped winding, since reverse_winding was frozen once at
            # the node level before this per-instance loop even started.
            reverse_winding = np.linalg.det(world[:3, :3]) < 0
            normal_mat = safe_normal_matrix(world)
            indices_cache = {}

            for prim in mesh.get("primitives", []):
                attrs = prim["attributes"]
                if "POSITION" not in attrs:
                    continue

                mat_idx = prim.get("material")

                mat = gltf.get("materials", [])[mat_idx] if mat_idx is not None and mat_idx < len(gltf.get("materials", [])) else {}
                raw_mat_name = mat.get("name", "")
                exts = mat.get("extensions", {})

                if _primitive_material_is_excluded(gltf, prim):
                    continue

                # A node carrying a genuine translation animation (the
                # hangar-door case) needs its geometry isolated into its own
                # sub-object even when it shares a material with static
                # neighbours (the door frame, the wall) -- otherwise the
                # whole shared-material builder would have to move together.
                # Keying only on mat_idx (as before) is kept for every other
                # node, including the blink case (an emissive material
                # flickers uniformly wherever it's used, so no per-node
                # split is needed there).
                animated_node_id = None
                if time_behavior is not None and anim_ancestor_idx is not None:
                    node_anim = gltf_animations.get(anim_ancestor_idx)
                    if node_anim is not None and len(node_anim["values"]) >= 2:
                        kind = time_behavior[0]
                        if kind in ("business_hours", "proximity") and node_anim["path"] in ("translation", "rotation"):
                            animated_node_id = node_idx

                # Reverted to the file-wide verdict (see the removed
                # per-node classification's own replacement comment above)
                # -- every node in this file agrees on flat-vs-rigid.
                this_node_is_flat = file_is_flat_only
                builder_key = (mat_idx, animated_node_id)

                if builder_key not in builders:
                    mat_name = material_name(gltf, mat_idx)
                    if animated_node_id is not None:
                        mat_name = f"{mat_name}_anim{animated_node_id}"
                    builder = MatBuilder(mat_name)

                    if mat_idx is not None:
                        pbr = mat.get("pbrMetallicRoughness", {})
                        color = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
                        builder.base_color_factor = (
                            int(max(0, min(1, color[0])) * 255),
                            int(max(0, min(1, color[1])) * 255),
                            int(max(0, min(1, color[2])) * 255),
                            int(max(0, min(1, color[3])) * 255) if len(color) > 3 else 255
                        )

                        builder.alpha_mode = mat.get("alphaMode", "OPAQUE")
                        builder.alpha_cutoff = mat.get("alphaCutoff", 0.5)
                        builder.double_sided = bool(mat.get("doubleSided", False))

                        # Name-based signal only here -- the real verdict
                        # (builder.is_decal) is finalized a bit further
                        # down, once material_flatness_stats' per-material
                        # reference height is available, so it can be
                        # gated on ground-proximity the same way
                        # is_near_ground_flat is (see that assignment's own
                        # comment for why: a "decal"-named/ASOBO_material_
                        # decal-tagged material isn't always a GROUND
                        # decal -- MSFS also uses that same material type
                        # for a rooftop weathering/grime overlay meant to
                        # stay coincident with its own rigid roof, not get
                        # globally reprojected onto real terrain. CONFIRMED
                        # REAL BUG this fixes: a real LHBP building's
                        # "roof_decal" material -- alpha_mode BLEND,
                        # authored at genuine roof height -- was draped
                        # unconditionally on name alone, with no elevation
                        # check at all (unlike is_near_ground_flat's own
                        # check just below), landing it flat on the ground
                        # far below the roof it was meant to sit on.
                        _decal_name_matched = "decal" in raw_mat_name.lower() or "ASOBO_material_decal" in exts

                        # Narrow DETECTION only -- this used to gate a DROP
                        # (see the removed builder_is_dropped_map below),
                        # discarding a material that only qualified here
                        # entirely rather than guessing a drape rank for it.
                        # Reverted: the real-world comparison against another
                        # converter's output for the same EGLC content
                        # (pavement/rail-ballast detail near the train) showed
                        # its tool just leaves this content as ordinary rigid
                        # (non-draped) geometry -- floating a little proud of
                        # the ground in the worst case -- which reads as fine
                        # in-sim, unlike our DROP which removed it outright.
                        # is_near_ground_flat is now informational only (kept
                        # for material_stats/debugging); the material falls
                        # through to the normal rigid path, so it renders and
                        # -- like every other rigid object since TILTED was
                        # removed -- is eligible for terrain_fit's vertical
                        # shift if its group qualifies. For
                        # material_stats in compute_file_flatness_and_
                        # reference's own docstring for the full real-world
                        # case this covers -- ground-layer models with one
                        # non-flat sibling material vetoing ATTR_draped for
                        # every OTHER, individually-flat material in the
                        # same file. Two conditions, both required:
                        #   1. This material's OWN geometry passes a strict
                        #      per-triangle flatness test, at
                        #      _NEAR_GROUND_FLAT_FRACTION_THRESHOLD (0.90,
                        #      not the file-wide 0.98 -- real tile/paver
                        #      geometry has a small amount of genuine edge/
                        #      seam detail that keeps it under 0.98 even
                        #      when it's unambiguously ground-level pavement.
                        #      Confirmed against the real EGLC source: the
                        #      exact material this exists for,
                        #      ini_GP_GEN_SmallTiles_4m_01, measures 0.9577
                        #      -- 0.98 never fired on it at all. Not an
                        #      arbitrary retreat: 0.90 is the same threshold
                        #      this project's own (currently-unused)
                        #      per-node classification already used.
                        #      Checked every other material in the same
                        #      real file at this threshold too: every
                        #      material that should stay rigid (the genuine
                        #      3D ramp Slope_01, plus Concrete_01/
                        #      Grunge_01/Colour_01) sits at 0.89 or well
                        #      below, comfortably clear of 0.90. The outcome
                        #      of qualifying here is just "stay rigid instead
                        #      of draped" -- a false positive costs one
                        #      small patch-scale object an unnecessary rigid
                        #      placement (still rendered, just not draped), a
                        #      false negative just leaves the original
                        #      rigid/floating symptom in place; neither is as
                        #      costly as it would be if the outcome were
                        #      still "drape with a guessed rank", which is
                        #      why this can stay lenient.
                        #   2. Its own flat elevation is close to the
                        #      file's overall ground-level reference. Flatness
                        #      alone isn't enough: a building's flat ROOF or a
                        #      bridge's flat deck TOP would also pass condition
                        #      1 despite sitting meters above the file's real
                        #      ground level, and must stay rigid (not get
                        #      dropped) -- this is exactly the failure mode a
                        #      more aggressive per-material attempt hit
                        #      earlier this session (it had no ground-
                        #      proximity check at all and wrongly flattened
                        #      chairs/glass/rooftops).
                        _mat_flat_fraction, _mat_ref_height = material_flatness_stats.get(mat_idx, (0.0, None))
                        builder.is_near_ground_flat = (
                            _mat_flat_fraction >= _NEAR_GROUND_FLAT_FRACTION_THRESHOLD
                            and _mat_ref_height is not None
                            and file_reference_height is not None
                            and abs(_mat_ref_height - file_reference_height) <= _NEAR_GROUND_FLAT_TOLERANCE_M
                        )

                        # Finalizing is_decal here (see _decal_name_matched
                        # above): only exclude a decal-named material from
                        # draping when there's POSITIVE evidence it sits
                        # away from the file's own ground level -- an
                        # unknown reference height (None, e.g. non-planar
                        # decal geometry) keeps the permissive, historical
                        # behavior (still draped) rather than guessing,
                        # same asymmetric-default reasoning as is_near_
                        # ground_flat's own tolerance check just above,
                        # just inverted: that one defaults to NOT
                        # qualifying unless proven near ground, this one
                        # defaults to qualifying unless proven far from it,
                        # since the name/extension signal is already a
                        # much stronger positive indicator than mere
                        # flatness is.
                        builder.is_decal = _decal_name_matched and (
                            _mat_ref_height is None or file_reference_height is None
                            or abs(_mat_ref_height - file_reference_height) <= _NEAR_GROUND_FLAT_TOLERANCE_M
                        )

                        # MSFS has shipped several glass extension names
                        # ("ASOBO_material_glass", "_glass_v2", "_kitty_glass")
                        # -- substring-match any extension key mentioning
                        # "glass" rather than enumerating exact names.
                        # Name-based detection also has to catch more than
                        # the word "glass" itself (real packages use
                        # "Window", "Glazing", "CurtainWall", etc.). Bare
                        # "pane"/"glaz" are deliberately excluded -- they
                        # false-match "propane", "japanese", "glaze"d
                        # ceramics.
                        _mat_lc = raw_mat_name.lower()
                        builder.is_glass = (any("glass" in ext_key.lower() for ext_key in exts)
                                            or any(kw in _mat_lc for kw in (
                                                "glass", "window", "glazing", "glaze", "vitr",
                                                "obratno", "steklo", "curtainwall",
                                                "curtain_wall", "facade_glass",
                                                "windshield", "windscreen")))

                        # NOT forcing alpha_mode to BLEND just because the
                        # name/extension mentions "glass" -- plenty of real
                        # "glass"-named materials are deliberately OPAQUE
                        # (reflective/painted glass) or MASK, some even
                        # invisible LOD/collision placeholders. The
                        # material's own alphaMode is the real source of
                        # truth for translucency; is_glass below only
                        # affects double-sidedness/shininess.
                        has_parallax_window = "ASOBO_material_parallax_window" in exts
                        if has_parallax_window:
                            # ASOBO's parallax_window fakes an interior
                            # room's depth via shader trickery on a flat
                            # pane -- X-Plane has no equivalent, and its
                            # authored MASK texture alone reads as a flat
                            # picture, not see-through glass. Real BLEND
                            # translucency is the closer approximation.
                            builder.alpha_mode = "BLEND"
                            builder.is_glass = True
                            # CAP the pane alpha at the floor, don't just
                            # fill it in when the author left it opaque -- a
                            # parallax window authored at, say, 50% still has
                            # to read as clear glass, not a milky sheet.
                            builder.base_color_factor = (
                                builder.base_color_factor[0], builder.base_color_factor[1], builder.base_color_factor[2],
                                min(builder.base_color_factor[3], _GLASS_TRANSLUCENCY_FLOOR))
                        elif builder.is_glass and builder.alpha_mode == "BLEND":
                            # A genuine glass material already authored
                            # BLEND, but relying on the TEXTURE's own alpha
                            # for the real punch-through pattern (its own
                            # baseColorFactor may default to opaque). Cap
                            # the factor alpha at the floor unconditionally
                            # -- glazing should read as genuinely
                            # see-through, not partially tinted.
                            builder.base_color_factor = (
                                builder.base_color_factor[0], builder.base_color_factor[1], builder.base_color_factor[2],
                                min(builder.base_color_factor[3], _GLASS_TRANSLUCENCY_FLOOR))

                        # MSFS glass/alpha-blended surfaces are often
                        # authored single-sided (doubleSided: false) and
                        # rely on MSFS's own shader to render both faces
                        # anyway. Exporting that flag as-is backface-culls
                        # the surface in X-Plane, hiding whatever's behind
                        # it from one direction. Covers every BLEND
                        # material, not just ones the is_glass heuristic
                        # catches.
                        if builder.is_glass or builder.alpha_mode == "BLEND":
                            builder.double_sided = True

                        base_tex = find_base_color_texture(mat)
                        if base_tex and "index" in base_tex:
                            builder.tex_coord = base_tex.get("texCoord", 0)
                            if "extensions" in base_tex and "KHR_texture_transform" in base_tex["extensions"]:
                                transform = base_tex["extensions"]["KHR_texture_transform"]
                                builder.uv_scale = transform.get("scale", [1.0, 1.0])
                                builder.uv_offset = transform.get("offset", [0.0, 0.0])
                                if "texCoord" in transform:
                                    builder.tex_coord = transform["texCoord"]

                            img_idx, _ = texture_image_index(gltf, base_tex["index"])
                            if img_idx is not None:
                                # DDS passthrough only when NOTHING below is
                                # going to read or modify the decoded pixel
                                # data: apply_color_factor/apply_alpha_factor
                                # edit it, and the BLEND-downgrade check
                                # further down (_texture_has_real_
                                # transparency) opens it with plain PIL,
                                # which can't read a passed-through .dds --
                                # any BLEND alpha_mode or a non-white tint
                                # means at least one of those will run, so
                                # both must be ruled out first.
                                _allow_dds = (
                                    builder.alpha_mode != "BLEND"
                                    and builder.base_color_factor[:3] == (255, 255, 255)
                                )
                                builder.texture_name = extract_image(
                                    gltf, buffers, img_idx, glb_path, textures_dir, external_textures_dir, image_cache, builder.base_color_factor,
                                    allow_dds_passthrough=_allow_dds,
                                )

                                if builder.base_color_factor[:3] != (255, 255, 255):
                                    builder.texture_name = apply_color_factor(
                                        textures_dir / builder.texture_name, builder.base_color_factor[:3])

                                if builder.alpha_mode == "BLEND" and builder.base_color_factor[3] < 255:
                                    builder.texture_name = apply_alpha_factor(textures_dir / builder.texture_name, builder.base_color_factor[3])

                                # Some exporters stamp alphaMode BLEND
                                # near-universally rather than as a real
                                # translucency signal -- plain walls/frames
                                # end up with BLEND + opaque factor too.
                                # Respecting alphaMode alone would route
                                # those through X-Plane's depth-sort-
                                # dependent blend path for no reason, so a
                                # material only keeps BLEND here if its own
                                # texture actually carries real
                                # transparency (checked via decoded alpha
                                # extrema, not by re-trusting alphaMode).
                                if builder.alpha_mode == "BLEND" and builder.base_color_factor[3] >= 250:
                                    if builder.is_glass:
                                        # Glass with an opaque-ish albedo still has to READ as
                                        # glass: keep BLEND and bake a translucency floor into
                                        # the texture's alpha now (apply_alpha_factor already
                                        # ran above with the then-opaque factor).
                                        builder.base_color_factor = (
                                            builder.base_color_factor[0], builder.base_color_factor[1],
                                            builder.base_color_factor[2], _GLASS_TRANSLUCENCY_FLOOR)
                                        builder.texture_name = apply_alpha_factor(
                                            textures_dir / builder.texture_name, _GLASS_TRANSLUCENCY_FLOOR)
                                    elif not _texture_has_real_transparency(textures_dir / builder.texture_name):
                                        builder.alpha_mode = "OPAQUE"

                        normal_tex = find_normal_texture(mat)
                        if normal_tex and "index" in normal_tex:
                            img_idx, _ = texture_image_index(gltf, normal_tex["index"])
                            if img_idx is not None:
                                # No post-processing ever touches the normal
                                # map slot -- safe to always pass through a
                                # source .dds unmodified (see
                                # allow_dds_passthrough's own docstring).
                                builder.normal_texture_name = extract_image(
                                    gltf, buffers, img_idx, glb_path, textures_dir, external_textures_dir, image_cache, (128, 128, 255, 255),
                                    allow_dds_passthrough=True,
                                )

                        if not builder.texture_name:
                            guess_stem = clean_texture_stem(raw_mat_name)
                            if guess_stem:
                                expected_tex = guess_stem + ".png"
                                match = None

                                if (textures_dir / expected_tex).exists() and _is_valid_image(textures_dir / expected_tex):
                                    match = textures_dir / expected_tex
                                elif external_textures_dir:
                                    match = _find_in_external_texture_roots(external_textures_dir, guess_stem)

                                if match:
                                    with _TEXTURE_LOCK:
                                        if match.resolve() != (textures_dir / expected_tex).resolve():
                                            # save_as_png, not a raw
                                            # shutil.copyfile: match came
                                            # from _find_in_external_texture_
                                            # roots, which indexes EVERY
                                            # recognized extension
                                            # (_EXTERNAL_TEXTURE_EXTENSIONS
                                            # includes .ktx2/.dds/.tga/...)
                                            # by clean stem -- a raw copy
                                            # assumed it was already a real
                                            # PNG just because the
                                            # DESTINATION is named
                                            # expected_tex (a ".png" path).
                                            # CONFIRMED REAL CRASH: a raw,
                                            # undecoded .ktx2 file copied
                                            # verbatim to a ".png"-named
                                            # path is not a valid PNG at
                                            # all -- X-Plane hard-crashed
                                            # ("THREAD FATAL ASSERT", a
                                            # real IDAT CRC error) trying
                                            # to load one.
                                            save_as_png(match.read_bytes(), textures_dir / expected_tex, builder.base_color_factor)
                                    builder.texture_name = expected_tex

                        # Same downgrade as above, for a material that still
                        # has no texture at all after every resolution
                        # attempt -- an opaque-alpha BLEND material with
                        # nothing but its own (opaque) baseColorFactor has
                        # no possible source of real transparency left to
                        # check.
                        if builder.alpha_mode == "BLEND" and builder.base_color_factor[3] >= 250 and not builder.texture_name:
                            builder.alpha_mode = "OPAQUE"

                        # Any material with an emissive texture -- building
                        # interior lights seen through windows, illuminated
                        # signage, a beacon bulb -- needs TEXTURE_LIT to
                        # light up in the dark, extracted unconditionally
                        # for every material that has one (not just ones
                        # with a "blink" animation). ATTR_light_level below
                        # is still only set for the blink case; otherwise
                        # X-Plane falls back to its own default day/night
                        # TEXTURE_LIT blend. Applies to glass too -- a
                        # window pane is the most common carrier of an
                        # emissive texture (interior lights glowing through
                        # glass at night), and X-Plane has no fundamental
                        # conflict combining ATTR_blend with TEXTURE_LIT.
                        emissive_tex = find_emissive_texture(mat)
                        if emissive_tex and "index" in emissive_tex:
                            # OBJ8 has exactly one (u,v) pair per vertex --
                            # TEXTURE/TEXTURE_NORMAL/TEXTURE_LIT all sample
                            # the same UV stream, so there's no format-
                            # level way to give the emissive texture its
                            # own transform when the base color texture
                            # also needs a different one. If the base
                            # color texture has no KHR_texture_transform of
                            # its own but the EMISSIVE texture is the one
                            # actually packed into a shared atlas, fall
                            # back to the emissive texture's own transform
                            # -- only when neither texture already claimed
                            # the shared UV stream, and only uv_scale/
                            # uv_offset, never builder.tex_coord (base_tex
                            # may have picked a non-default TEXCOORD
                            # channel for an unrelated reason).
                            base_tex_has_transform = bool(
                                base_tex and "extensions" in base_tex and "KHR_texture_transform" in base_tex["extensions"]
                            )
                            if not base_tex_has_transform and "extensions" in emissive_tex and "KHR_texture_transform" in emissive_tex["extensions"]:
                                transform = emissive_tex["extensions"]["KHR_texture_transform"]
                                builder.uv_scale = transform.get("scale", [1.0, 1.0])
                                builder.uv_offset = transform.get("offset", [0.0, 0.0])

                            img_idx, _ = texture_image_index(gltf, emissive_tex["index"])
                            if img_idx is not None:
                                _emissive_factor_pending = mat.get("emissiveFactor")
                                builder.emissive_texture_name = extract_image(
                                    gltf, buffers, img_idx, glb_path, textures_dir, external_textures_dir, image_cache, (255, 240, 180, 255),
                                    allow_dds_passthrough=not (_emissive_factor_pending and len(_emissive_factor_pending) >= 3),
                                )
                                emissive_factor = mat.get("emissiveFactor")
                                if emissive_factor and len(emissive_factor) >= 3:
                                    builder.emissive_texture_name = apply_emissive_factor(
                                        textures_dir / builder.emissive_texture_name, emissive_factor[:3]
                                    )
                                if time_behavior is not None and time_behavior[0] == "blink":
                                    # A periodic blink (beacon/lamp bulb)
                                    # gets a flicker driven by a custom
                                    # dataref, "msfs2xp/night_blink", that
                                    # the companion "msfs2xp Night Blink"
                                    # FlyWithLua script computes every
                                    # frame -- an approximation of MSFS's
                                    # zulu-time modulo formula, not exposed
                                    # as any stock X-Plane dataref.
                                    # ATTR_light_level replaces X-Plane's
                                    # own day/night TEXTURE_LIT blend
                                    # entirely, so the companion script's
                                    # dataref also folds in real night
                                    # gating (sim/graphics/scenery/
                                    # percent_lights_on) itself, since
                                    # ATTR_light_level only reads one value.
                                    builder.light_level_dataref = (0.3, 0.7, "msfs2xp/night_blink")
                        else:
                            # NO emissive TEXTURE, but the material can
                            # still glow at night by another MSFS
                            # mechanism -- windows/interiors/signage are
                            # often lit this way instead:
                            #   - ASOBO_material_day_night_switch: the whole
                            #     material is explicitly "on at night".
                            #   - ASOBO_material_parallax_window: a fake lit
                            #     interior room painted on the glass.
                            #   - a large emissiveFactor (MSFS's "authored
                            #     dim, multiply way up in HDR" convention)
                            #     with no texture: the base colour emits.
                            # In every case the lit look is the base colour
                            # driven bright, so synthesise a TEXTURE_LIT
                            # from it, contrast-stretched by
                            # apply_emissive_factor.
                            emissive_factor = mat.get("emissiveFactor")
                            _big_factor = bool(emissive_factor) and max(emissive_factor[:3] or [0]) > 1.5
                            _night_switch = "ASOBO_material_day_night_switch" in exts
                            _parallax = "ASOBO_material_parallax_window" in exts
                            if (_big_factor or _night_switch or _parallax) and builder.texture_name:
                                builder.emissive_texture_name = apply_emissive_factor(
                                    textures_dir / builder.texture_name,
                                    emissive_factor[:3] if emissive_factor else [6.0, 6.0, 6.0],
                                )

                        if animated_node_id is not None:
                            # Every delta below is computed in the node's
                            # LOCAL/parent space, then rotated into world
                            # space by the PARENT's own world rotation (not
                            # this node's, which would double-apply the
                            # node's own rotation to a transform that is
                            # actually expressed in the parent's frame) and
                            # then by the file's global pitch/yaw/roll, same
                            # as every vertex position above.
                            node_anim = gltf_animations[anim_ancestor_idx]
                            values = node_anim["values"]

                            # Must match how world_transforms actually built
                            # anim_ancestor_world (node_local_matrix_at_rest,
                            # not the raw node_local_matrix) -- otherwise
                            # dividing it back out below leaves a spurious
                            # rest-vs-bind-pose rotation baked into parent_rot.
                            local = node_local_matrix_at_rest(anim_ancestor_node, anim_ancestor_idx, gltf_animations)
                            try:
                                parent_rot = (anim_ancestor_world @ np.linalg.inv(local))[:3, :3]
                            except np.linalg.LinAlgError:
                                parent_rot = anim_ancestor_world[:3, :3]
                            col_norms = np.linalg.norm(parent_rot, axis=0)
                            col_norms[col_norms == 0] = 1.0
                            parent_rot_n = parent_rot / col_norms

                            # Both trigger shapes drive off a dataref that
                            # sweeps through a "closed" value at the ends of
                            # a keyframe schedule and an "open" value in the
                            # middle -- they only differ in which dataref
                            # and what the schedule's breakpoints are, so
                            # this is computed once here rather than
                            # duplicated per animation kind (translation vs
                            # rotation).
                            if time_behavior[0] == "business_hours":
                                open_start, open_end = time_behavior[1], time_behavior[2]
                                builder.anim_dataref = "sim/time/local_time_sec"
                                schedule_times = [
                                    max(0.0, open_start - 60.0), open_start,
                                    open_end, min(86400.0, open_end + 60.0),
                                ]
                                schedule_values = [0.0, 1.0, 1.0, 0.0]
                            else:  # "proximity"
                                # No stock X-Plane dataref tracks "is the user
                                # aircraft near this specific placed object" --
                                # this drives from a custom dataref, unique
                                # per source model (each MSFS door/barrier/
                                # gate in this package is already its own
                                # distinct SimObject file, so this name is
                                # naturally unique per physical object), that
                                # only the companion FlyWithLua plugin
                                # supplies at runtime from the aircraft's live
                                # distance to this object's known placement(s).
                                dataref_name = f"msfs2xp/proximity/{sanitize_name(model_name)}"
                                builder.proximity_dataref = dataref_name
                                builder.anim_dataref = dataref_name
                                schedule_times = [0.0, 1.0]
                                schedule_values = [0.0, 1.0]

                            if node_anim["path"] == "translation":
                                delta_local = np.asarray(values[-1][:3], dtype=np.float64) - np.asarray(values[0][:3], dtype=np.float64)
                                delta_dir = parent_rot_n @ delta_local
                                delta_world = apply_global_rotation(delta_dir.reshape(1, 3))[0]
                                builder.anim_translate_keys = [
                                    (t, float(delta_world[0]) * v, float(delta_world[1]) * v, float(delta_world[2]) * v)
                                    for t, v in zip(schedule_times, schedule_values)
                                ]
                            else:  # "rotation"
                                q_start, q_end = _rest_open_rotation_values(values)
                                r_start = quat_to_matrix(q_start)[:3, :3]
                                r_end = quat_to_matrix(q_end)[:3, :3]
                                delta_rot_local = r_end @ r_start.T
                                delta_rot_world = parent_rot_n @ delta_rot_local @ parent_rot_n.T
                                axis_world, angle_deg = quat_to_axis_angle_matrix(delta_rot_world)
                                axis_world = apply_global_rotation(axis_world.reshape(1, 3))[0]
                                axis_norm = np.linalg.norm(axis_world)
                                if axis_norm > 1e-9:
                                    axis_world = axis_world / axis_norm
                                builder.anim_rotate = (float(axis_world[0]), float(axis_world[1]), float(axis_world[2]))
                                builder.anim_rotate_keys = [(t, float(angle_deg) * v) for t, v in zip(schedule_times, schedule_values)]

                                # Rotation happens around the node's own
                                # world-space origin (the hinge/pivot
                                # point), not the object's overall local
                                # origin -- re-baseline this sub-object's
                                # vertices around that pivot below and
                                # shift back at OBJ8-write time via a
                                # (static) ANIM_trans wrapping the rotate.
                                pivot = (anim_ancestor_world @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
                                pivot = apply_global_rotation(pivot.reshape(1, 3))[0]
                                builder.anim_pivot = (float(pivot[0]), float(pivot[1]), float(pivot[2]))

                    builders[builder_key] = builder

                builder = builders[builder_key]
                pos_acc = attrs["POSITION"]
                uv_acc_name = f"TEXCOORD_{builder.tex_coord}"
                uv_acc = attrs.get(uv_acc_name)
                norm_acc = attrs.get("NORMAL")

                # Reverted to a plain file-wide flag: this_node_is_flat is
                # now just file_is_flat_only (see its own assignment
                # above), identical for every node in this file.
                builder.all_source_nodes_flat = builder.all_source_nodes_flat and this_node_is_flat

                # Real MSFS/ASOBO exports routinely pack MANY primitives'
                # vertex data into ONE big shared POSITION/NORMAL/TEXCOORD
                # accessor, with each primitive selecting only its own small
                # window of it via the ASOBO_primitive extension's
                # StartIndex/BaseVertexIndex/PrimitiveCount extras (see
                # local_indices below) -- confirmed against a real airport
                # package: one multi-material ground-marking node split by
                # material into 15 separate output .obj files, every one of
                # which carried ~390,000 VT lines despite most needing only
                # a few hundred to a few thousand of them (one had 6
                # referenced indices against 389,216 declared vertices).
                # local_indices/win_lo/win_hi (this primitive's own
                # referenced vertex window within the shared accessor) has
                # to be known BEFORE deciding what to read/append below --
                # computing it first, instead of always reading the FULL
                # accessor, is what keeps a material-split of a large
                # shared-buffer node from having every single split carry
                # the entire buffer's vertex data.
                if "indices" in prim:
                    idx_acc = prim["indices"]
                    if idx_acc not in indices_cache:
                        indices_cache[idx_acc] = read_accessor(gltf, buffers, idx_acc).astype(np.int64).reshape(-1)
                    full_indices = indices_cache[idx_acc]
                else:
                    full_indices = np.arange(gltf["accessors"][pos_acc]["count"], dtype=np.int64)

                extras = prim.get("extras", {}).get("ASOBO_primitive", {})
                start = extras.get("StartIndex", 0)
                base_vertex = extras.get("BaseVertexIndex", 0)
                idx_count = (extras["PrimitiveCount"] * 3) if "PrimitiveCount" in extras else (len(full_indices) - start)
                local_indices = full_indices[start:start + idx_count] + base_vertex

                if len(local_indices):
                    win_lo = int(local_indices.min())
                    win_hi = int(local_indices.max())
                else:
                    win_lo, win_hi = 0, -1  # nothing this primitive actually draws

                block_key = (node_idx, id(world), pos_acc, norm_acc, uv_acc, win_lo, win_hi)

                if block_key not in builder.vertex_blocks:
                    positions = read_accessor(gltf, buffers, pos_acc)[win_lo:win_hi + 1]

                    # GPU fork: only worth dispatching to the GPU for large
                    # meshes -- below gpu_accel.GPU_VERTEX_THRESHOLD the
                    # numpy matmul is faster than the transfer overhead, and
                    # this stays on the exact float64 numpy path either way
                    # whenever no GPU is available (transform_points_gpu
                    # returns None and is a no-op then).
                    positions_world = None
                    if len(positions) >= gpu_accel.GPU_VERTEX_THRESHOLD:
                        positions_world = gpu_accel.transform_points_gpu(positions, world)
                    if positions_world is None:
                        positions_h = np.hstack([positions, np.ones((len(positions), 1))])
                        positions_world = (world @ positions_h.T).T[:, :3]
                    positions_world = apply_global_rotation(positions_world)

                    if this_node_is_flat:
                        # The whole FILE (not just this node) is confirmed
                        # nothing-but-flat: compress every vertex onto the
                        # same level -- the file's own dominant elevation,
                        # preserved as-is (not forced to 0). No partial/
                        # per-band logic.
                        positions_world = flatten_to_reference(positions_world, file_reference_height)
                    else:
                        # Mixed/real-3D file: leave heights exactly as
                        # authored, including any vertex with Y<0 -- a
                        # multi-meter-tall feature (base trim, footing,
                        # eave) below the model's own local origin is
                        # normal, not broken data, so clamping it would
                        # crush real shape into a degenerate sliver.
                        #
                        # The one exception mirrors _clamp_flat_reference_
                        # height's own safety net for the FLAT path: some
                        # exports lose track of a sane local origin and
                        # come back with the whole node sitting hundreds/
                        # thousands of meters away. Only fires when the
                        # node's own vertex cluster is TIGHT (a real
                        # building has genuine multi-meter spread; a
                        # lost-origin artifact doesn't) and sits at an
                        # implausible level -- then shifts the whole
                        # cluster back by a uniform offset, preserving its
                        # shape rather than collapsing anything. Done
                        # before the below-ground count/log just after.
                        not_stray = ~flag_stray_vertices(positions_world)
                        clean_y = positions_world[not_stray, 1] if not_stray.any() else positions_world[:, 1]
                        if len(clean_y):
                            y_span = float(clean_y.max() - clean_y.min())
                            y_level = float(np.median(clean_y))
                            if y_span < 50.0 and abs(y_level) > _IMPLAUSIBLE_ELEVATION_LIMIT:
                                logger.warning(
                                    f"{glb_path.name}: node {node_idx} ({builder.name}) sits at an "
                                    f"implausible level (median y={y_level:.1f}m, internal spread only "
                                    f"{y_span:.1f}m) -- shifting the whole compact cluster back near "
                                    f"y=0 instead of leaving it effectively invisible."
                                )
                                positions_world[:, 1] -= y_level

                        # Negative-Y vertices are NOT clamped here. A
                        # blanket "floor every Y<0 vertex to 0" was tried
                        # and reverted earlier in this project: it crushed
                        # real, coherent below-origin geometry (a
                        # foundation/footing/basement edge) into a
                        # degenerate zero-height sliver. A per-primitive
                        # STRAY-vertex clamp was tried after that and ALSO
                        # reverted: it ran before the existing stray-mask
                        # triangle-drop below (see block_key's stray_mask),
                        # so a vertex this clamped to y=0 no longer looked
                        # like an outlier by the time that check ran and
                        # its degenerate triangle survived -- collapsed
                        # instead of dropped, which is what showed up as
                        # stray near-transparent/textureless rectangles.
                        # flag_stray_vertices' own triangle-drop below is
                        # the correct mechanism for negligible stray/
                        # negative geometry: it REMOVES the whole triangle
                        # rather than moving a vertex into a degenerate
                        # position.

                        # Below-terrain geometry left untouched is not a
                        # rendering bug -- X-Plane just occludes it behind
                        # the opaque compiled terrain, the same as any real
                        # building's basement or footing.
                        below_ground = positions_world[:, 1] < 0.0
                        if below_ground.any():
                            clipped_vertex_count += int(below_ground.sum())
                            clipped_builder_names.add(builder.name)

                    normals_world = None
                    if "NORMAL" in attrs:
                        normals = read_accessor(gltf, buffers, norm_acc)[win_lo:win_hi + 1]
                        if normals.ndim == 2 and normals.shape[1] >= 3:
                            try:
                                normals_world = normals[:, :3] @ normal_mat.T
                            except (ValueError, np.linalg.LinAlgError):
                                normals_world = normals[:, :3]
                            norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
                            norms[norms == 0] = 1
                            normals_world = normals_world / norms
                            normals_world = apply_global_rotation(normals_world)

                    uvs = read_accessor(gltf, buffers, uv_acc, force_normalized=True)[win_lo:win_hi + 1] if uv_acc is not None else None

                    # positions/normals/uvs are read from three independent
                    # accessors, each sliced by the same [win_lo:win_hi+1]
                    # window -- correct only if every accessor genuinely
                    # has at least win_hi+1 entries. glTF requires shared
                    # vertex count across a primitive's attributes, but
                    # ASOBO exports are known to bend spec conventions
                    # elsewhere (see the half-float-in-SHORT UV quirk
                    # above); a short accessor would otherwise truncate
                    # silently and shift the UV/normal index for every
                    # later vertex in this builder. Treated the same as
                    # the "attribute doesn't exist at all" case (a uniform
                    # per-vertex fallback) rather than trusting a short read.
                    if uvs is not None and len(uvs) < len(positions_world):
                        logger.warning(
                            f"{glb_path.name}: node {node_idx} ({builder.name}) has a TEXCOORD accessor "
                            f"too short for its own primitive window ({len(uvs)} rows, needs "
                            f"{len(positions_world)}) -- falling back to a uniform UV for this block "
                            f"instead of an index-shifted read."
                        )
                        uvs = None
                    if normals_world is not None and len(normals_world) < len(positions_world):
                        logger.warning(
                            f"{glb_path.name}: node {node_idx} ({builder.name}) has a NORMAL accessor "
                            f"too short for its own primitive window ({len(normals_world)} rows, needs "
                            f"{len(positions_world)}) -- falling back to a uniform normal for this block "
                            f"instead of an index-shifted read."
                        )
                        normals_world = None

                    if this_node_is_flat and len(positions_world):
                        # This one node/block's own XZ footprint, tracked
                        # per-block rather than only the builder's pooled
                        # union of every block -- a builder pools every
                        # node sharing this material (keyed by material,
                        # not node), so scattered same-material pieces
                        # (e.g. repeated stand-marking text) would
                        # otherwise inflate to the bbox of their combined
                        # spread across the whole file, nothing like any
                        # single piece's real size.
                        bx = positions_world[:, 0]
                        bz = positions_world[:, 2]
                        block_area = float((bx.max() - bx.min()) * (bz.max() - bz.min()))
                        builder.block_footprint_areas.append(block_area)

                    v_start = len(builder.vertices)
                    if builder.anim_pivot is not None:
                        pvx, pvy, pvz = builder.anim_pivot
                        for p_v in positions_world:
                            builder.vertices.append((p_v[0] - pvx, p_v[1] - pvy, p_v[2] - pvz))
                    else:
                        for p_v in positions_world:
                            builder.vertices.append((p_v[0], p_v[1], p_v[2]))

                    if uvs is not None:
                        su, sv = builder.uv_scale
                        ou, ov = builder.uv_offset
                        for uv in uvs:
                            u = uv[0] * su + ou
                            v = uv[1] * sv + ov
                            builder.uvs.append((u, 1.0 - v))
                    else:
                        for _ in positions_world:
                            builder.uvs.append((0.0, 0.0))

                    if normals_world is not None:
                        for n in normals_world:
                            builder.normals.append((n[0], n[1], n[2]))
                    else:
                        for _ in positions_world:
                            builder.normals.append((0.0, 1.0, 0.0))

                    stray_mask = flag_stray_vertices(positions_world)

                    builder.vertex_blocks[block_key] = {
                        "v_offset": v_start,
                        "count": len(positions_world),
                        "stray_mask": stray_mask,
                    }

                block = builder.vertex_blocks[block_key]

                # local_indices is still in absolute-accessor-index space
                # (win_lo..win_hi) -- rebase by win_lo since the appended
                # block above only holds THIS primitive's own [win_lo,
                # win_hi] slice, not the full accessor.
                v_off = block["v_offset"]
                stray_mask = block["stray_mask"]
                for tri_start in range(0, len(local_indices) - 2, 3):
                    a, b, c = local_indices[tri_start:tri_start + 3] - win_lo
                    if stray_mask[a] or stray_mask[b] or stray_mask[c]:
                        # Degenerate/leftover triangle reaching out to a
                        # stray vertex far outside this primitive's own
                        # bulk extent (see flag_stray_vertices) -- drop it
                        # rather than draw it.
                        continue
                    # If mesh scale was negatively inverted in MSFS, reverse
                    # X-Plane's triangle drawing order.
                    if reverse_winding:
                        builder.indices.append((v_off + int(a), v_off + int(c), v_off + int(b)))
                    else:
                        builder.indices.append((v_off + int(a), v_off + int(b), v_off + int(c)))

    # Drop redundant reverse-wound "double-sided via duplicate geometry"
    # copies: some source packages work around MSFS's single-sided glass
    # by manually duplicating a pane with reversed winding instead of
    # marking it doubleSided. The double_sided-forcing fix above already
    # makes the original mesh visible from both sides on its own, so the
    # duplicate becomes two coincident BLEND surfaces at the same
    # position -- which causes intermittent blend-order flicker, since
    # X-Plane's draw order between two coincident translucent surfaces
    # isn't stable frame to frame. Detected geometrically (every vertex of
    # the smaller builder has a close match in the larger one), not by
    # naming convention, so it catches the same pattern regardless of
    # what either builder is named.
    _blend_keys = [k for k, b in builders.items() if b.alpha_mode == "BLEND" and b.vertices]
    _dup_dropped = set()
    for _dup_i, _dup_key_a in enumerate(_blend_keys):
        if _dup_key_a in _dup_dropped:
            continue
        for _dup_key_b in _blend_keys[_dup_i + 1:]:
            if _dup_key_b in _dup_dropped:
                continue
            _dup_a, _dup_b = builders[_dup_key_a], builders[_dup_key_b]
            if len(_dup_b.vertices) <= len(_dup_a.vertices):
                _dup_small_key, _dup_small, _dup_big = _dup_key_b, _dup_b, _dup_a
            else:
                _dup_small_key, _dup_small, _dup_big = _dup_key_a, _dup_a, _dup_b
            if _vertex_positions_are_subset(_dup_small.vertices, _dup_big.vertices):
                _dup_big.double_sided = True
                _dup_dropped.add(_dup_small_key)
    for _dup_key in _dup_dropped:
        del builders[_dup_key]
    if _dup_dropped:
        logger.info(f"{model_name}: dropped {len(_dup_dropped)} redundant reverse-wound duplicate "
                    f"BLEND surface(s) (every vertex already covered by another, now-double-sided "
                    f"BLEND builder) -- removes the coincident-surface blend-order flicker between "
                    f"them.")

    # Re-center the WHOLE model (every builder + light) around its own XZ
    # footprint center, and lift it so its lowest point sits at Y=0 if any
    # vertex is below that -- a sensible, predictable local origin instead
    # of leaving the pivot wherever the source export's arbitrary local
    # origin was, independent of whatever terrain_fit.py's uniform shift
    # later does with the whole rigid object (a pure translation, so it
    # doesn't care where the local origin sits).
    #
    # Must NOT double-move rotation-animated sub-objects: their vertices
    # are already stored PIVOT-relative (see anim_pivot handling above),
    # so a builder with anim_pivot set is recentered by moving its pivot
    # instead of its (already shift-invariant) vertex list.
    #
    # The removed (center_x, center_z) and any Y lift are written to a
    # sidecar (read back by main.py) and folded into the DSF placement
    # instead of just dropped: center_x/center_z become a per-model
    # placement-offset correction, and the Y lift becomes a negative
    # delta added to the placement's AGL offset (dsf_compiler's AGL pool
    # is always terrain-relative, so this preserves runtime terrain-
    # following at the anchor point, only changing the fixed vertical
    # nudge on top).
    # --- Synthesize a night downlight for an emissive-only light fixture ---
    # Some MSFS airport light models (apron poles, floods, wig-wags) carry
    # no ASOBO_macro_light/KHR_lights_punctual node, just an emissive
    # material -- X-Plane self-glows the TEXTURE_LIT lens but casts no
    # light on the ground and can't flash. When the model name marks it
    # as a light fixture with no real light produced for it, drop a
    # synthetic LIGHT_SPILL_CUSTOM (omni + blink for wig-wag/runway-guard,
    # a downward cone otherwise), through the same night-gating/flash
    # write path below.
    if not light_entries and _looks_like_light_fixture(model_name):
        _nlc = model_name.lower()
        _syn_flash = any(k in _nlc for k in ("wigwag", "wig_wag", "guard", "clearance"))
        _syn_flood = any(k in _nlc for k in ("flood", "projector"))
        _syn_pts = []
        for _b in builders.values():
            if not getattr(_b, "vertices", None):
                continue
            _bn = _b.name.lower()
            _is_emis = bool(getattr(_b, "emissive_texture_name", None)) or \
                _bn.endswith(("emis", "emissive", "_lit", "_light"))
            if not _is_emis:
                continue
            _va = np.asarray(_b.vertices, dtype=np.float64)
            if (_va[:, 0].max() - _va[:, 0].min()) > 25.0 or (_va[:, 2].max() - _va[:, 2].min()) > 25.0:
                continue  # not a fixture -- a lit wall/sign slipped the name gate
            _p = (float(_va[:, 0].mean()), float(_va[:, 1].max()), float(_va[:, 2].mean()))
            if not any(abs(_p[0] - q[0]) < 1.0 and abs(_p[2] - q[2]) < 1.0 for q in _syn_pts):
                _syn_pts.append(_p)
        for (_cx, _cy, _cz) in _syn_pts[:6]:
            _pw = apply_global_rotation(np.array([[_cx, _cy, _cz]], dtype=np.float64))[0]
            light_entries.append({
                "pos": (float(_pw[0]), float(_pw[1]), float(_pw[2])),
                "dir": (0.0, 0.0, 0.0) if _syn_flash else (0.0, -1.0, 0.0),
                "color": (1.0, 0.93, 0.78),
                "cone_angle": 360.0 if _syn_flash else 150.0,
                "intensity": 30.0 if _syn_flood else (8.0 if _syn_flash else 5.0),
                "day_night_cycle": True,
                "flash_frequency": 1.0 if _syn_flash else 0.0,
            })
        if light_entries:
            logger.info(f"{model_name}: synthesized {len(light_entries)} night "
                        f"light(s) from an emissive-only fixture (no macro_light in source).")

    # Bounding extremes across every absolute-frame point in this model:
    # each rigid/draped/translate-animated builder's own vertex bbox, each
    # rotation-animated builder's single pivot point, and every light.
    bbox_mins_x, bbox_maxs_x = [], []
    bbox_mins_z, bbox_maxs_z = [], []
    bbox_mins_y = []
    # Per-vertex X/Z, kept alongside the bbox extremes above -- used for
    # the MEDIAN recenter below instead of the bbox midpoint. A vertex set
    # dominated by one dense structure with a few sparse far-flung outliers
    # (e.g. a bundled, unrelated decal reaching hundreds of meters out)
    # keeps the median inside the dense cluster, where the bbox midpoint
    # would get dragged anywhere by just two extreme corner vertices. Lands
    # close to the bbox midpoint anyway for an ordinary, evenly-detailed
    # model, so this is low-risk for the common case.
    _median_x_parts, _median_z_parts = [], []
    for builder in builders.values():
        if builder.anim_pivot is not None:
            px, py, pz = builder.anim_pivot
            bbox_mins_x.append(px); bbox_maxs_x.append(px)
            bbox_mins_z.append(pz); bbox_maxs_z.append(pz)
            bbox_mins_y.append(py)
            _median_x_parts.append(np.array([px], dtype=np.float64))
            _median_z_parts.append(np.array([pz], dtype=np.float64))
        elif builder.vertices:
            arr = np.asarray(builder.vertices, dtype=np.float64)
            bbox_mins_x.append(float(arr[:, 0].min())); bbox_maxs_x.append(float(arr[:, 0].max()))
            bbox_mins_z.append(float(arr[:, 2].min())); bbox_maxs_z.append(float(arr[:, 2].max()))
            bbox_mins_y.append(float(arr[:, 1].min()))
            _median_x_parts.append(arr[:, 0])
            _median_z_parts.append(arr[:, 2])
    for entry in light_entries:
        lx, ly, lz = entry["pos"]
        bbox_mins_x.append(lx); bbox_maxs_x.append(lx)
        bbox_mins_z.append(lz); bbox_maxs_z.append(lz)
        bbox_mins_y.append(ly)
        _median_x_parts.append(np.array([lx], dtype=np.float64))
        _median_z_parts.append(np.array([lz], dtype=np.float64))

    recenter_x = recenter_z = 0.0
    recenter_agl_delta = 0.0
    if bbox_mins_x:
        min_y = min(bbox_mins_y)
        # Symmetric in both directions -- re-zero min_y whether it's
        # above OR below 0 (a multi-figure prop's pivot authored at the
        # formation's center, not any one figure's feet, sits above 0;
        # a dip below 0 is the more common case). Folded into the
        # placement's AGL offset either way. Epsilon matches
        # dsf_compiler's own AGL-routing threshold (abs(agl) >= 0.01).
        y_lift = -min_y if abs(min_y) >= 0.01 else 0.0
        recenter_agl_delta = -y_lift

        # Horizontal (XZ) re-centering only for genuinely rigid files --
        # draped/flat files are very often several separate MSFS objects
        # (base fill + paint-stripe overlay + stain decal) meant to sit
        # exactly coincident, originally sharing one BGL placement
        # lat/lon. Re-centering each independently around its OWN bbox
        # would shift each one's anchor by a different amount and
        # visibly separate previously-aligned layers.
        if file_is_rigid:
            recenter_x = float(np.median(np.concatenate(_median_x_parts)))
            recenter_z = float(np.median(np.concatenate(_median_z_parts)))

        if recenter_x or recenter_z or y_lift:
            for builder in builders.values():
                if builder.anim_pivot is not None:
                    px, py, pz = builder.anim_pivot
                    builder.anim_pivot = (px - recenter_x, py + y_lift, pz - recenter_z)
                elif builder.vertices:
                    builder.vertices = [
                        (x - recenter_x, y + y_lift, z - recenter_z) for (x, y, z) in builder.vertices
                    ]
            for entry in light_entries:
                lx, ly, lz = entry["pos"]
                entry["pos"] = (lx - recenter_x, ly + y_lift, lz - recenter_z)

    try:
        offset_sidecar = objects_dir / f"{model_name}.originoffset.json"
        offset_sidecar.write_text(
            json.dumps({"x": recenter_x, "y": recenter_agl_delta, "z": recenter_z}), encoding="utf-8"
        )
    except OSError:
        logger.warning(f"{glb_path.name}: could not write origin-offset sidecar -- placement won't be re-centered.")

    # Pre-pass: which builders are draped, and their footprint area, needs
    # to be known for EVERY builder before any of them can be assigned a
    # rank-based offset (see draped_ranking.rank_draped_layer_offsets) --
    # a per-builder decision made inline in the write loop below, as
    # before, couldn't see its siblings yet. Cheap (just reads attributes
    # already computed by the per-primitive loop above; no geometry
    # re-scan) and has no side effects, so doing it as a separate pass
    # ahead of the write loop (which DOES have side effects -- lazy
    # texture assignment) is safe.
    draped_areas = {}
    builder_is_draped_map = {}
    for builder_key, builder in builders.items():
        num_verts = len(builder.vertices)
        file_wide_flat = bool(num_verts) and builder.all_source_nodes_flat
        is_decal = getattr(builder, 'is_decal', False)
        # A material that ONLY qualifies through the near-ground-flat
        # fallback (not the file-wide verdict, not the explicit
        # "decal"-named path) used to be DROPPED from the output entirely
        # here. Reverted per real-world comparison against another
        # converter's output for the same EGLC content (pavement/
        # rail-ballast detail near the train): its tool keeps this content
        # as ordinary rigid (non-draped) geometry instead of guessing a
        # drape rank OR omitting it, and that reads fine in-sim even when
        # it ends up floating a little proud of the ground -- unlike our
        # DROP, which removed real content outright. So it now just stays
        # rigid: not draped, not dropped, and (like every other rigid
        # object since TILTED was removed) eligible for terrain_fit's
        # vertical shift if its group qualifies.
        is_draped = file_wide_flat or is_decal
        builder_is_draped_map[builder_key] = is_draped
        if is_draped and builder.block_footprint_areas:
            draped_areas[builder_key] = float(np.median(builder.block_footprint_areas))
    draped_layer_offsets = rank_draped_layer_offsets(draped_areas)

    obj_paths = []
    for builder_key, builder in builders.items():
        if not builder.texture_name:
            # A BLEND material (glass, tinted panels, etc.) with no texture
            # of its own MUST get a texture whose alpha channel actually
            # reflects its base_color_factor alpha -- borrowing an
            # unrelated, normally fully-opaque texture from some other
            # material (as the two fallbacks below do) silently turns it
            # back into an opaque surface showing that other material's
            # colors, which is exactly what makes untextured glass render
            # as "not transparent". Go straight to the flat-color swatch,
            # which is built from this builder's own (already alpha-
            # adjusted, e.g. by the is_glass override) base_color_factor.
            needs_own_alpha = builder.alpha_mode == "BLEND"
            if image_cache and not needs_own_alpha:
                builder.texture_name = list(image_cache.values())[0]
            else:
                existing_pngs = [
                    p.name for p in textures_dir.glob("*.png")
                    if not p.name.endswith("_default.png")
                ] if not needs_own_alpha else []
                if existing_pngs:
                    builder.texture_name = existing_pngs[0]
                else:
                    fallback_name = f"{model_name}_{builder.name}_default.png"
                    fallback_path = textures_dir / fallback_name

                    with _TEXTURE_LOCK:
                        if not (fallback_path.exists() and _is_valid_image(fallback_path)):
                            img = Image.new("RGBA", (2, 2), builder.base_color_factor)
                            temp_path = fallback_path.with_name(f"{fallback_path.name}.tmp_{_unique_suffix()}")
                            img.save(str(temp_path), "PNG")
                            _atomic_replace(temp_path, fallback_path)

                    builder.texture_name = fallback_name

        obj_path = objects_dir / f"{model_name}_{builder.name}.obj"
        num_verts = len(builder.vertices)

        flat_indices = []
        for tri in builder.indices:
            flat_indices.extend(tri)
        num_indices = len(flat_indices)

        # Per-builder now, not file-wide: True only if every node that fed
        # into this builder was individually flat (see all_source_nodes_flat
        # above) -- a builder with no vertices at all never got its flag
        # touched, so it's excluded explicitly rather than defaulting True.
        # Computed in the pre-pass above (draped_areas/builder_is_draped_map)
        # so every draped builder's footprint is already known file-wide by
        # the time rank_draped_layer_offsets needs to compare them -- just
        # looked up here, not recomputed.
        builder_is_draped = builder_is_draped_map[builder_key]
        builder_footprint_area = draped_areas.get(builder_key)

        if builder_is_draped and builder.vertices:
            # ATTR_draped discards this object's authored Y entirely during
            # X-Plane's main color-render pass (see the ATTR_draped comment
            # below) -- but a nonzero Y left over from flat-reference-height
            # computation upstream (_clamp_flat_reference_height deliberately
            # PRESERVES a "plausible" 0-100m elevation for flat content that
            # might legitimately be a roof or bridge deck, e.g. a rigid
            # builder with one flat sub-node) can still leak into OTHER
            # engine passes that don't get the same terrain-conforming
            # treatment -- observed in X-Plane 12 as a shadow floating at
            # roof height above ground-level draped geometry that renders
            # correctly in the beauty pass. Zeroing it here, for draped
            # builders only, is a no-op for the visible render (Y was
            # already being discarded there) and removes the only other
            # place a stale elevation could still be read from. Mutated in
            # place -- not just in the vt_lines string below -- so the
            # MeshIR sidecar a few lines down (which terrain_fit.py/
            # draped_merge.py may later re-serialize verbatim into a
            # corrected-offset copy) sees the same zeroed value instead of
            # reintroducing the stale one on any object they touch.
            builder.vertices = [(v[0], 0.0, v[2]) for v in builder.vertices]

        with obj_path.open("w", encoding="utf-8") as f:
            f.write("I\n800\nOBJ\n\n")

            f.write(f"TEXTURE ../textures/{builder.texture_name}\n")
            if builder.normal_texture_name:
                f.write(f"TEXTURE_NORMAL ../textures/{builder.normal_texture_name}\n")
            if builder.emissive_texture_name:
                f.write(f"TEXTURE_LIT ../textures/{builder.emissive_texture_name}\n")
            f.write("\n")

            f.write(f"POINT_COUNTS {num_verts} 0 0 {num_indices}\n\n")

            # Batched into one join+write instead of one f.write() per line --
            # identical bytes on disk, far fewer syscalls for meshes with
            # thousands of vertices/indices.
            vt_lines = [None] * num_verts
            for i in range(num_verts):
                v = builder.vertices[i]
                n = builder.normals[i]
                st = builder.uvs[i]
                vt_lines[i] = f"VT {v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {n[0]:.5f} {n[1]:.5f} {n[2]:.5f} {st[0]:.5f} {st[1]:.5f}\n"
            f.write("".join(vt_lines))

            f.write("\n")

            idx_lines = []
            for i in range(0, num_indices - 9, 10):
                chunk = flat_indices[i:i+10]
                idx_lines.append("IDX10 " + " ".join(map(str, chunk)) + "\n")

            remainder_start = (num_indices // 10) * 10
            for i in range(remainder_start, num_indices):
                idx_lines.append(f"IDX {flat_indices[i]}\n")
            f.write("".join(idx_lines))

            f.write("\n")
            if builder.double_sided:
                f.write("ATTR_no_cull\n")

            # alpha_mode alone decides blend vs. no-blend -- a second,
            # independent bug from the same root cause as the alpha_mode
            # assignment fix above: this used to check is_glass FIRST,
            # completely bypassing alpha_mode, so a "glass"-named material
            # deliberately authored OPAQUE (or MASK) at the source (e.g. a
            # real EGLC package's "MT_GlassBlack", "ini_Glass_Opaque") got
            # ATTR_blend written anyway regardless of what alpha_mode's own
            # (now-correct) value said -- the assignment fix alone wasn't
            # enough while this write-time check still short-circuited on
            # is_glass by itself. is_glass now only bumps shininess higher
            # for the BLEND case, matching real glass's extra reflectivity,
            # without affecting whether blending happens at all.
            if builder.alpha_mode == "BLEND":
                f.write("ATTR_blend\n")
                f.write(f"ATTR_shiny_rat {'1.0' if getattr(builder, 'is_glass', False) else '0.5'}\n")
            elif builder.alpha_mode == "MASK":
                f.write(f"ATTR_no_blend {builder.alpha_cutoff:.3f}\n")
                f.write("ATTR_shiny_rat 0.5\n")
            else:
                f.write("ATTR_no_blend 0.5\n")
                f.write("ATTR_shiny_rat 0.5\n")

            if builder_is_draped:
                # ATTR_draped is a geometry-stream state command (it has to
                # sit right before the TRIS it applies to, not up in the
                # header) that tells X-Plane to bend this geometry onto the
                # terrain mesh -- exactly what a flat-only file (ground
                # markings, runway/taxiway decals) needs instead of sitting
                # as a rigid object at a fixed height. Also still honored
                # for individually-named decal materials in an otherwise
                # 3-D file.
                f.write("ATTR_draped\n")

                # Draping re-projects every triangle onto terrain
                # regardless of authored Y, so draw order among
                # overlapping draped layers comes ONLY from an explicit
                # layer group + offset -- left unset, X-Plane gives no
                # ordering guarantee between same-group layers at all.
                if builder_footprint_area is not None:
                    # Always "markings" -- no size-based group split.
                    # X-Plane draws a whole group as one tier, so splitting
                    # by size risks two adjacent, physically touching
                    # pieces of the same real surface landing in different
                    # groups purely because one aggregate bbox crossed a
                    # threshold and the other didn't, producing a visible
                    # seam. Every layer stays in one group, ordered only by
                    # offset -- ranked against this file's OTHER draped
                    # layers (rank_draped_layer_offsets, computed once in
                    # the pre-pass above) rather than by fixed area
                    # thresholds, so every layer gets its own distinct
                    # offset instead of colliding within a size bucket.
                    layer_group = "markings"
                    layer_offset = draped_layer_offsets[builder_key]
                    f.write(f"ATTR_layer_group_draped {layer_group} {layer_offset}\n")

            if builder.light_level_dataref:
                v1, v2, dataref = builder.light_level_dataref
                f.write(f"ATTR_light_level {v1:.3f} {v2:.3f} {dataref}\n")

            has_anim = bool(builder.anim_translate_keys) or bool(builder.anim_rotate)
            # A rotation around an off-center pivot (a hinge, a boom-barrier
            # axle) needs a NESTED ANIM_begin/end, not a sibling one: the
            # outer block's ANIM_trans shifts the local coordinate origin
            # out to the pivot (a static offset -- same xyz at both keys,
            # since ANIM_trans_begin/key/end is the only OBJ8 command that
            # can express a translation at all), and only the INNER block's
            # ANIM_rotate -- now turning around that shifted origin, which
            # is exactly the pivot -- rotates the geometry (itself stored
            # relative to the pivot, see anim_pivot above) correctly. A
            # sibling (non-nested) ANIM_trans + ANIM_rotate pair would
            # instead rotate around the object's own unrelated local origin
            # and only translate the whole already-mis-rotated result.
            nested_rotate = bool(builder.anim_rotate and builder.anim_pivot)

            if has_anim:
                f.write("ANIM_begin\n")

            if builder.anim_translate_keys:
                f.write(f"ANIM_trans_begin {builder.anim_dataref}\n")
                for t, dx, dy, dz in builder.anim_translate_keys:
                    f.write(f"ANIM_trans_key {t:.2f} {dx:.5f} {dy:.5f} {dz:.5f}\n")
                f.write("ANIM_trans_end\n")

            if builder.anim_rotate:
                if nested_rotate:
                    px, py, pz = builder.anim_pivot
                    f.write(f"ANIM_trans_begin {builder.anim_dataref}\n")
                    f.write(f"ANIM_trans_key 0.00 {px:.5f} {py:.5f} {pz:.5f}\n")
                    f.write(f"ANIM_trans_key 1.00 {px:.5f} {py:.5f} {pz:.5f}\n")
                    f.write("ANIM_trans_end\n")
                    f.write("ANIM_begin\n")

                ax, ay, az = builder.anim_rotate
                f.write(f"ANIM_rotate_begin {ax:.5f} {ay:.5f} {az:.5f} {builder.anim_dataref}\n")
                for t, angle_deg in builder.anim_rotate_keys:
                    f.write(f"ANIM_rotate_key {t:.2f} {angle_deg:.5f}\n")
                f.write("ANIM_rotate_end\n")

            f.write(f"TRIS 0 {num_indices}\n")

            if nested_rotate:
                f.write("ANIM_end\n")
            if has_anim:
                f.write("ANIM_end\n")

        obj_paths.append(obj_path)

        # MeshIR sidecar: only for non-animated, non-blink builders --
        # terrain_fit.py/draped_merge.py (the only consumers) never touch
        # animated or ATTR_light_level content at all (see their own
        # module docstrings for why), so there's nothing for them to do
        # with a sidecar for those and no point writing one. Real float64
        # numpy arrays straight from the same in-memory data this loop
        # just wrote out as text -- no round trip.
        if not has_anim and not builder.light_level_dataref:
            ir = mesh_ir_module.MeshIR(
                name=obj_path.stem,
                positions=np.array(builder.vertices, dtype=np.float64) if builder.vertices else np.zeros((0, 3)),
                normals=np.array(builder.normals, dtype=np.float64) if builder.normals else np.zeros((0, 3)),
                uvs=np.array(builder.uvs, dtype=np.float64) if builder.uvs else np.zeros((0, 2)),
                indices=np.array(flat_indices, dtype=np.int64),
                texture=(f"../textures/{builder.texture_name}" if builder.texture_name else None),
                # TILTED is never written (see file_is_rigid's own
                # comment above) -- terrain_fit.py's uniform vertical
                # shift replaces it for every qualifying rigid group
                # regardless of size, draped siblings get the per-vertex
                # warp, and this field just has to agree with what
                # actually got written to the .obj (nothing).
                tilted=False,
                draped=builder_is_draped,
                draped_layer_offset=(draped_layer_offsets.get(builder_key) if builder_is_draped else None),
                double_sided=builder.double_sided,
                alpha_mode=builder.alpha_mode,
                alpha_cutoff=builder.alpha_cutoff,
                footprint_area_m2=builder_footprint_area,
                proximity_dataref=builder.proximity_dataref,
            )
            mesh_ir_module.save(ir, mesh_ir_module.sidecar_path_for(obj_path))

        if builder_footprint_area is not None:
            # Secondary, belt-and-suspenders ordering signal: real z-order
            # among overlapping draped layers now comes from
            # ATTR_layer_group_draped (written above, from this same
            # footprint_area) -- X-Plane's spec is explicit that CMDS
            # stream order gives no ordering guarantee at all without a
            # layer group, so this sidecar's sort in main.py is no longer
            # the primary mechanism, just a harmless tie-breaker within the
            # same group. main.py reads it the same way as the proximity
            # sidecar below.
            footprint_sidecar_path = obj_path.with_suffix(".footprint.json")
            with footprint_sidecar_path.open("w", encoding="utf-8") as f:
                json.dump({"area_m2": builder_footprint_area}, f)

        if builder.proximity_dataref:
            # A small sidecar next to the .obj itself, keyed by the exact
            # same stem -- main.py's placement step (which knows the real
            # placement lat/lon this converter has no visibility into) scans
            # for these after conversion to build the manifest the
            # companion FlyWithLua plugin reads at runtime.
            sidecar_path = obj_path.with_suffix(".proximity.json")
            with sidecar_path.open("w", encoding="utf-8") as f:
                json.dump({"dataref": builder.proximity_dataref}, f)

    if light_entries:
        # Lights are self-contained geometry commands (LIGHT_SPILL_CUSTOM),
        # not tied to any material/texture, so they get their own tiny
        # sub-object rather than being folded into one of the per-material
        # ones above. TEXTURE is still mandatory per the OBJ8 spec even
        # with nothing to texture -- left blank, which the spec explicitly
        # allows for "no texture requirement".
        lights_obj_path = objects_dir / f"{model_name}_lights.obj"
        with lights_obj_path.open("w", encoding="utf-8") as f:
            f.write("I\n800\nOBJ\n\n")
            f.write("TEXTURE \n\n")
            f.write("POINT_COUNTS 0 0 0 0\n\n")
            light_ir_entries = []
            _is_wigwag_fixture = _looks_like_wigwag(model_name)
            _wigwag_head = 0
            for entry in light_entries:
                # Clamped the same way every other color source in this file
                # already is (base_color_factor, etc) -- MSFS routinely
                # ships out-of-[0,1] values on light-adjacent fields (see
                # apply_emissive_factor's docstring for emissiveFactor
                # values like [100,100,100]), and unlike those other
                # sources, macro_light's raw color was previously written
                # straight through with no guard at all.
                r = max(0.0, min(1.0, entry["color"][0]))
                g = max(0.0, min(1.0, entry["color"][1]))
                b = max(0.0, min(1.0, entry["color"][2]))
                cone = max(0.0, min(360.0, entry["cone_angle"]))
                # semi = cos(half-angle); the spec calls out 1.0 as the
                # explicit sentinel for "omnidirectional" rather than
                # something the general formula naturally produces at
                # cone=360, so that case is special-cased.
                semi = 1.0 if cone >= 359.0 else math.cos(math.radians(cone / 2.0))
                # MSFS's macro_light has no explicit physical "radius"
                # field -- intensity is the closest proxy. X-Plane's
                # LIGHT_SPILL_CUSTOM "size" is a unitless multiplier
                # against X-Plane's own reference light size (real
                # airport lights in X-Plane's default scenery sit around
                # 1.0-3.0), while MSFS's own "intensity" ranges from 0
                # (taxi edge lights) to 500+ (interior room lights) with
                # no fixed upper bound -- a linear mapping squashes the
                # realistic 0-10 airport-light range into a barely-visible
                # sliver near the floor. Log-scaling spreads that range
                # across a visible portion of the 1.0-3.0 band while
                # keeping the same practical ceiling.
                intensity_norm = min(math.log1p(max(entry["intensity"], 0.0)) / math.log1p(500.0), 1.0)
                size = 1.0 + 2.0 * intensity_norm
                px, py, pz = entry["pos"]
                dx, dy, dz = entry["dir"]

                # macro_light -> X-Plane light. Priorities, in order:
                #   1. FLASHING fixture -> one of X-Plane's OWN built-in
                #      animated named lights (wigwag_y1/y2, obs_strobe_night,
                #      obs_red_night). These flash + day/night gate inside
                #      the sim engine with ZERO plugin dependency. The old
                #      path used custom msfs2xp/* datarefs that only resolve
                #      when the companion FlyWithLua script is installed AND
                #      loaded -- otherwise the light is simply dark forever
                #      (the "wigwag / clearance light does nothing" report).
                #   2. STEADY light -> a real LIGHT_SPILL_CUSTOM.
                #        night-only -> sim/graphics/scenery/percent_lights_on
                #                      (0 by day .. 1 at night; read as a
                #                      continuous intensity multiplier, the
                #                      same role it plays for X-Plane's own
                #                      default scenery lights)
                #        always-on  -> NULL (unconditional full intensity)
                #   A colour we can't map to a named flasher falls through
                #   to (2) as a STEADY lit spill -- visible for everyone,
                #   which beats a flashing-but-dark custom-dataref light.
                is_flashing = entry["flash_frequency"] > 0.0
                is_night_only = entry["day_night_cycle"]

                is_near_white = max(r, g, b) - min(r, g, b) < 0.12 and min(r, g, b) > 0.6
                is_reddish = r > 0.55 and g < 0.45 and b < 0.45

                # A FLASHING light is served by one of X-Plane's own built-in
                # animated named lights -- NOT by the msfs2xp/* custom
                # datarefs, which only exist when the companion FlyWithLua
                # script is installed AND loaded. When it isn't, that dataref
                # reads 0 every frame and the light is simply dark forever
                # (exactly the "wigwag / clearance light does nothing"
                # report). The named lights below are animated + day/night
                # gated entirely inside the sim, zero dependencies:
                #   near-white flash -> obs_strobe_night (night-gated strobe)
                #   red flash        -> obs_red_night    (night-gated red)
                #   other colour     -> fall through to a STEADY night spill
                #                       (lit but not flashing -- still far
                #                       better than dark, and a coloured
                #                       non-red flasher is rare)
                named = None
                if is_flashing and _is_wigwag_fixture:
                    # Alternate the two heads (y1 / y2) across a fixture's
                    # lights so a two-lamp runway-guard bar flip-flops the
                    # way the real thing does.
                    named = "wigwag_y1" if _wigwag_head % 2 == 0 else "wigwag_y2"
                    _wigwag_head += 1
                elif is_flashing and is_near_white:
                    named = "obs_strobe_night"
                elif is_flashing and is_reddish:
                    named = "obs_red_night"
                if named is not None:
                    f.write(f"LIGHT_NAMED {named} {px:.5f} {py:.5f} {pz:.5f}\n")
                    light_ir_entries.append(mesh_ir_module.LightEntry(
                        pos=(px, py, pz), dir=(dx, dy, dz), color=(r, g, b),
                        cone_angle=cone, size=size, dataref="NULL", named_light=named,
                    ))
                    continue

                # STEADY scenery light -> X-Plane's own registered param
                # light `full_custom_halo_night` (or `full_custom_halo` for
                # an always-on one). This is what stock scenery actually
                # uses for warm downward floods; it is a SPILL_HW_DIR light
                # gated night/day inside the sim engine with NO dataref.
                #
                # The previous LIGHT_SPILL_CUSTOM ... sim/graphics/scenery/
                # percent_lights_on was wrong twice over: (1) that command
                # is essentially unused by stock scenery and did not render
                # here at all, and (2) its trailing dataref must be a >=9-
                # float ARRAY; percent_lights_on is a scalar, so X-Plane
                # could not drive the light. Result: every apron/pole light
                # dark even with HDR on.
                param_light = "full_custom_halo" if not is_night_only else "full_custom_halo_night"

                # Zero-length aim is undefined; an up-pointing pole/flood
                # fixture came through with its emitter axis un-transformed
                # -- it lights the ground, so force it down.
                if abs(dx) + abs(dy) + abs(dz) < 1e-6:
                    dx, dy, dz = 0.0, -1.0, 0.0
                elif dy > 0.20 and _is_downlight_fixture(model_name):
                    dx, dy, dz = 0.0, -1.0, 0.0

                # The `S` (size) param is the light's REACH in metres.
                # Our intensity->size formula tops out near 3, which from a
                # 20-30 m apron mast is a pinprick. Give a floodlight-type
                # fixture (by name) OR any raised downward emitter a
                # metre-scale reach scaled to the mount height (X-Plane's
                # own apron flood throws ~120 m) and open the cone right up
                # (low F = wide spill) so the ground pool is broad, not a
                # spotlight. `_is_downlight_fixture` also covers models
                # whose emissive lens sits low on the mesh (SHS_Floodlight
                # _001) where `py` alone would miss.
                _named_flood = _is_downlight_fixture(model_name)
                if dy < -0.3 and (py > 2.0 or _named_flood):
                    _mount = max(py, 12.0) if _named_flood else py
                    _floor = 40.0 if _named_flood else 25.0
                    size = max(size, min(120.0, max(_floor, _mount * 3.0)))
                    semi = min(semi, 0.12)

                # full_custom_halo's RGB is HDR radiance, not clamped [0,1]
                # -- stock warm floods sit at ~1.0 (dim warehouse lamps).
                # An apron / ramp flood reads much brighter than that in
                # reality, so over-drive it, scaled by the source
                # macro_light intensity (user: "their light is low").
                bright = min(1.5 + max(entry["intensity"], 0.0) / 10.0, 4.5)
                rb, gb, bb = min(r * bright, 6.0), min(g * bright, 6.0), min(b * bright, 6.0)

                f.write(
                    # LIGHT_PARAM full_custom_halo_night  px py pz  R G B A
                    #             S  X Y Z  F   -- 3 pos + 9 params, no dref.
                    f"LIGHT_PARAM {param_light} {px:.5f} {py:.5f} {pz:.5f} "
                    f"{rb:.4f} {gb:.4f} {bb:.4f} 1.0 {size:.3f} "
                    f"{dx:.5f} {dy:.5f} {dz:.5f} {semi:.4f}\n"
                )
                light_ir_entries.append(mesh_ir_module.LightEntry(
                    pos=(px, py, pz), dir=(dx, dy, dz), color=(rb, gb, bb),
                    cone_angle=cone, size=size, dataref=param_light,
                ))
        obj_paths.append(lights_obj_path)

        # Per-file dataref-bucket tally (DEBUG) -- both blink datarefs
        # depend entirely on macro_light's own flash_frequency field ever
        # being >0 in the source data, which real packages may simply
        # never set (a real rotating beacon is often its own animated
        # emissive mesh instead, not a flashing macro_light). Without this,
        # "why did every light come out non-blinking" had to be answered by
        # grepping every output .obj by hand -- confirmed necessary against
        # a real airport package where all 205 lights landed in the single
        # night-only-steady bucket.
        bucket_counts = {}
        for e in light_ir_entries:
            bucket = f"LIGHT_NAMED:{e.named_light}" if e.named_light else e.dataref
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        logger.debug(f"{glb_path.name}: {len(light_ir_entries)} macro_light(s) -- {bucket_counts}")

        # MeshIR sidecar for the lights-only pseudo-object: terrain_fit.py
        # is the one consumer (it Y-shifts a corrected building's own
        # LIGHT_SPILL_CUSTOM positions consistently with the rest of the
        # group) -- draped_merge.py never touches this file at all (it has
        # no ATTR_draped, so _parse_draped_candidate already excludes it).
        lights_ir = mesh_ir_module.MeshIR(name=lights_obj_path.stem, tilted=False, lights=light_ir_entries)
        mesh_ir_module.save(lights_ir, mesh_ir_module.sidecar_path_for(lights_obj_path))

    with _EXPORT_LOCK:
        global _EXPORTED_COUNT
        _EXPORTED_COUNT += 1
        current_total = _EXPORTED_COUNT

    logger.info(f"[{current_total} Exported] Successfully compiled X-Plane OBJ8 for {glb_path.name}")
    if clipped_vertex_count:
        logger.info(
            f"{glb_path.name}: {clipped_vertex_count} vertex/vertices sit below y=0 (left "
            f"as authored -- hidden behind the compiled terrain) across "
            f"{len(clipped_builder_names)} sub-object(s): {sorted(clipped_builder_names)}"
        )

    try:
        del gltf, bin_chunk, buffers, world_transforms, builders, image_cache
    except NameError:
        pass

    gc.collect()

    return obj_paths


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("glb", help="Input MSFS .glb file")
    ap.add_argument("--objects", default="../objects", help="Output folder for .obj files")
    ap.add_argument("--textures", default="../texture", help="Output folder for extracted textures")
    ap.add_argument("--external-textures", default=None, help="Folder containing pre-converted PNGs")
    ap.add_argument("--pitch", type=float, default=0.0, help="Rotate model around X axis (degrees)")
    ap.add_argument("--yaw", type=float, default=0.0, help="Rotate model around Y axis (degrees)")
    ap.add_argument("--roll", type=float, default=0.0, help="Rotate model around Z axis (degrees)")
    args = ap.parse_args()

    convert(args.glb, args.objects, args.textures, args.external_textures, args.pitch, args.yaw, args.roll)


if __name__ == "__main__":
    main()