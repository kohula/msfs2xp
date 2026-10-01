"""Texture sizes held to what each texture is drawn on.

MSFS ships every texture at full resolution (often 4096 px) no matter how
small the object, and X-Plane keeps every loaded texture in video memory.
After conversion this pass looks at every object that references each
texture and caps its size by the largest of them: full size for terminals
(60 m+), half for hangars and towers, a quarter for vehicles, an eighth
for people and small props. Normal maps get one step less. Ground content
(draped objects and .pol polygons) keeps full size -- it's seen up close
over large areas. Textures no object or polygon references any more are
deleted.
"""

import math
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from mesh_convert import mesh_ir

_TEXTURE_LINE = re.compile(r"^(TEXTURE|TEXTURE_LIT|TEXTURE_NORMAL)\s+\.\./textures/(\S+)\s*$", re.MULTILINE)
_IMAGE_SUFFIXES = (".png", ".dds", ".jpg", ".jpeg", ".tga")
_MIN_SIDE = 256


def texture_cap(radius_m, max_side):
    """Largest side for a texture drawn on a model of this radius (half its
    bounding-box diagonal, metres)."""
    if radius_m >= 60.0:
        cap = max_side
    elif radius_m >= 15.0:
        cap = max_side // 2
    elif radius_m >= 3.0:
        cap = max_side // 4
    else:
        cap = max_side // 8
    return max(cap, min(_MIN_SIDE, max_side))


def _obj_radius(obj_path, text):
    """Half the bounding-box diagonal of an .obj's geometry, from its MeshIR
    sidecar when there is one (fast), else its VT lines."""
    ir_path = mesh_ir.sidecar_path_for(obj_path)
    if ir_path.exists():
        try:
            ir = mesh_ir.load(ir_path)
            if len(ir.positions):
                return float(np.linalg.norm(ir.positions.max(axis=0) - ir.positions.min(axis=0))) / 2.0
        except Exception:
            pass
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for line in text.splitlines():
        if line.startswith("VT "):
            parts = line.split()
            for k in range(3):
                v = float(parts[1 + k])
                lo[k] = min(lo[k], v)
                hi[k] = max(hi[k], v)
    if lo[0] == math.inf:
        return 0.0
    return math.dist(lo, hi) / 2.0


def collect_texture_radii(out_dir):
    """{texture file name: (largest radius it's drawn on, is a normal map)}
    across every .obj and .pol in the pack. Draped/polygon use counts as
    infinite radius (always full size)."""
    out_dir = Path(out_dir)
    radii = {}

    def note(name, radius, normal):
        prev = radii.get(name)
        if prev is None:
            radii[name] = (radius, normal)
        else:
            radii[name] = (max(prev[0], radius), prev[1] and normal)

    for obj_path in (out_dir / "objects").glob("*.obj"):
        try:
            text = obj_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        refs = _TEXTURE_LINE.findall(text)
        if not refs:
            continue
        radius = math.inf if "\nATTR_draped" in text else _obj_radius(obj_path, text)
        for kind, name in refs:
            # Night textures need less detail than the day texture.
            note(name, radius / 4.0 if kind == "TEXTURE_LIT" else radius, kind == "TEXTURE_NORMAL")
    for pol_path in (out_dir / "polygons").glob("*.pol"):
        try:
            text = pol_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for _kind, name in _TEXTURE_LINE.findall(text):
            note(name, math.inf, False)
    return radii


def _shrink_png(path, cap):
    with Image.open(path) as img:
        w, h = img.size
        if max(w, h) <= cap:
            return False
        f = cap / max(w, h)
        small = img.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
        fmt = "PNG" if path.suffix.lower() == ".png" else None
        small.save(path, fmt)
    return True


_DXT_BLOCK = {b"DXT1": 8, b"DXT3": 16, b"DXT5": 16}


def _shrink_dds(path, cap):
    """Drop the top mip levels of a DXT DDS until it fits -- no re-encode,
    the smaller levels are already in the file."""
    data = path.read_bytes()
    if len(data) < 128 or data[:4] != b"DDS ":
        return False
    height, width = struct.unpack_from("<II", data, 12)
    mips = struct.unpack_from("<I", data, 28)[0]
    block = _DXT_BLOCK.get(data[84:88])
    if block is None or mips < 2 or max(width, height) <= cap:
        return False

    def level_size(w, h):
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block

    drop, offset, w, h = 0, 128, width, height
    while max(w, h) > cap and drop < mips - 1:
        offset += level_size(w, h)
        w, h = max(1, w // 2), max(1, h // 2)
        drop += 1
    header = bytearray(data[:128])
    struct.pack_into("<II", header, 12, h, w)
    struct.pack_into("<I", header, 20, level_size(w, h))
    struct.pack_into("<I", header, 28, mips - drop)
    path.write_bytes(bytes(header) + data[offset:])
    return True


def apply_texture_budget(out_dir, max_side=2048, delete_unused=True, log=None):
    """Cap every referenced texture in out_dir/textures and delete the
    unreferenced ones. Returns (shrunk, deleted)."""
    textures_dir = Path(out_dir) / "textures"
    if not textures_dir.is_dir():
        return 0, 0
    radii = collect_texture_radii(out_dir)
    shrunk = deleted = 0
    for path in textures_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        use = radii.get(path.name)
        if use is None:
            if delete_unused:
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    pass
            continue
        if not max_side:
            continue
        radius, normal = use
        cap = max_side if radius == math.inf else texture_cap(radius, max_side)
        if normal:
            cap = max(cap // 2, min(_MIN_SIDE, cap))
        try:
            if path.suffix.lower() == ".dds":
                changed = _shrink_dds(path, cap)
            else:
                changed = _shrink_png(path, cap)
        except Exception as e:
            if log:
                log(f"Could not resize {path.name}: {e}", "warning")
            continue
        shrunk += int(changed)
    return shrunk, deleted
