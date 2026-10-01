"""Uniformly scaled copies of converted objects.

An MSFS placement carries a scale (the last float of a library-object
record; a SimProp container's scale applies to everything it holds), but a
DSF placement cannot scale an object. So each (object, scale) pair that is
actually placed gets its own copy with the scale baked into the geometry:
vertex positions, light positions and spill sizes, animation translations
and LOD distances. Rotations, normals and texture coordinates are
unchanged by a uniform scale.
"""

import json
from pathlib import Path

from mesh_convert import mesh_ir

# command -> indices (into the whitespace-split tokens, command at 0) of
# the numbers that are lengths and therefore scale.
_SCALED_FIELDS = {
    "VT": (1, 2, 3),
    "VLINE": (1, 2, 3),
    "VLIGHT": (1, 2, 3),
    "LIGHT_NAMED": (2, 3, 4),
    # LIGHT_PARAM <name> x y z <params...>: only the position is certain to
    # be a length for every param light; the full_custom_halo* lights this
    # project writes also carry their spill size at token 9.
    "LIGHT_PARAM": (2, 3, 4),
    "LIGHT_CUSTOM": (1, 2, 3, 8),
    "LIGHT_SPILL_CUSTOM": (1, 2, 3, 8),
    "ANIM_trans": (1, 2, 3, 4, 5, 6),
    "ANIM_trans_key": (2, 3, 4),
    "ATTR_LOD": (1, 2),
    "smoke_black": (1, 2, 3, 4),
    "smoke_white": (1, 2, 3, 4),
}
_HALO_SIZE_TOKEN = 9


def _fmt(v):
    return f"{v:.5f}"


def scale_obj8_text(text, scale):
    out = []
    for line in text.splitlines(keepends=True):
        tokens = line.split()
        fields = _SCALED_FIELDS.get(tokens[0]) if tokens else None
        if not fields:
            out.append(line)
            continue
        if tokens[0] == "LIGHT_PARAM" and len(tokens) > _HALO_SIZE_TOKEN and tokens[1].startswith("full_custom_halo"):
            fields = fields + (_HALO_SIZE_TOKEN,)
        indent = line[:len(line) - len(line.lstrip())]
        try:
            for i in fields:
                if i < len(tokens):
                    tokens[i] = _fmt(float(tokens[i]) * scale)
        except ValueError:
            out.append(line)
            continue
        out.append(indent + " ".join(tokens) + ("\n" if line.endswith("\n") else ""))
    return "".join(out)


def scale_suffix(scale):
    return f"_s{int(round(scale * 1000))}"


def make_scaled_variant(obj_dir, stem, scale):
    """Write `<stem>_s<scale*1000>.obj` (and its pipeline sidecars) next to
    `<stem>.obj` unless it already exists. Returns the new stem, or `stem`
    itself if the source object is missing."""
    obj_dir = Path(obj_dir)
    src = obj_dir / f"{stem}.obj"
    new_stem = stem + scale_suffix(scale)
    dst = obj_dir / f"{new_stem}.obj"
    if dst.exists():
        return new_stem
    try:
        text = src.read_text(encoding="utf-8")
    except OSError:
        return stem
    dst.write_text(scale_obj8_text(text, scale), encoding="utf-8")

    ir_path = mesh_ir.sidecar_path_for(src)
    if ir_path.exists():
        try:
            ir = mesh_ir.load(ir_path)
            ir.name = new_stem
            ir.positions = ir.positions * scale
            for light in ir.lights:
                light.pos = tuple(c * scale for c in light.pos)
                light.size = light.size * scale
            if ir.footprint_area_m2 is not None:
                ir.footprint_area_m2 = ir.footprint_area_m2 * scale * scale
            mesh_ir.save(ir, mesh_ir.sidecar_path_for(dst))
        except Exception:
            pass
    footprint = obj_dir / f"{stem}.footprint.json"
    if footprint.exists():
        try:
            data = json.loads(footprint.read_text(encoding="utf-8"))
            data["area_m2"] = data["area_m2"] * scale * scale
            (obj_dir / f"{new_stem}.footprint.json").write_text(json.dumps(data), encoding="utf-8")
        except (OSError, ValueError, KeyError):
            pass
    proximity = obj_dir / f"{stem}.proximity.json"
    if proximity.exists():
        try:
            (obj_dir / f"{new_stem}.proximity.json").write_bytes(proximity.read_bytes())
        except OSError:
            pass
    return new_stem
