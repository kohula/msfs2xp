"""
Intermediate mesh representation (MeshIR) + its OBJ8 serializer.

Replaces the previous design where terrain_fit.py and draped_merge.py each
regex-parsed already-written OBJ8 TEXT to recover vertex arrays convert()
already had as numpy moments earlier, then reformatted back to text -- a
real perf cost (text I/O + regex on every post-pass) and a real precision/
mis-parse risk for no benefit. Now: convert() writes a MeshIR pickle
sidecar alongside every .obj it produces; terrain_fit/draped_merge load
that sidecar, do their vector math directly on real float64 numpy arrays,
and call write_obj8() once when they actually produce a corrected/merged
file -- the numbers stay in memory as floats from convert() all the way
through both post-passes.

Scoped to exactly what terrain_fit.py and draped_merge.py need, not the
full generality of every OBJ8 feature convert() can emit (animation,
TILTED rigid rotation, KHR light-level blink, ...): both of those modules
only ever touch flat, non-animated, non-blinking draped ground content
(and MeshIR's own `lights` field for the "_lights" companion sub-object) --
see their own module docstrings for why animated/blink/light-level content
is explicitly excluded from what they operate on. Convert() still owns the
FULL OBJ8 write path for every other case; MeshIR/write_obj8 here is the
shared format for the specific subset the two post-passes work with.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LightEntry:
    pos: tuple  # (x, y, z) world-space
    dir: tuple  # (x, y, z) aim direction, normalized
    color: tuple  # (r, g, b), 0..1
    cone_angle: float
    size: float
    dataref: str
    # When set, this light is one of X-Plane's own built-in LIGHT_NAMED
    # types (e.g. "obs_strobe_night") instead of a LIGHT_SPILL_CUSTOM --
    # animated/day-night-gated by X-Plane itself, no dataref/plugin
    # involved. See write_obj8 for what actually gets written.
    named_light: str | None = None


@dataclass
class MeshIR:
    name: str
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    uvs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64))
    texture: str | None = None
    tilted: bool = False
    draped: bool = False
    draped_layer_offset: int | None = None
    # X-Plane ATTR_layer_group_draped GROUP (not the -5..+5 offset, which
    # is draped_layer_offset above). Only 11 offset slots exist WITHIN one
    # group, so routing base pavement / wear decals / painted lines /
    # signage into separate draw bands ("taxiways" < "runways" <
    # "markings" < "airports") avoids collisions on a dense airport.
    draped_layer_group: str = "markings"
    double_sided: bool = False
    alpha_mode: str = "OPAQUE"  # OPAQUE | BLEND | MASK
    alpha_cutoff: float = 0.5
    lights: list = field(default_factory=list)  # list[LightEntry] -- only for the "_lights" pseudo-sub-object
    footprint_area_m2: float | None = None
    proximity_dataref: str | None = None

    @property
    def num_triangles(self) -> int:
        return len(self.indices) // 3


def save(ir: MeshIR, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(ir, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> MeshIR:
    with open(path, "rb") as f:
        return pickle.load(f)


def sidecar_path_for(obj_path: Path) -> Path:
    return obj_path.with_suffix(".meshir.pkl")


def write_obj8(ir: MeshIR, obj_path: Path) -> None:
    """Writes ir as a real OBJ8 .obj file. Format matches convert()'s own
    established conventions exactly (I/800/OBJ header; VT lines with 8
    floats -- position, normal, uv; IDX lines; ATTR_draped +
    ATTR_layer_group_draped <ir.draped_layer_group> <offset> when draped
    (the group is "markings" unless a caller routed this layer into another
    draw band -- see MeshIR.draped_layer_group); a single contiguous TRIS
    block; LIGHT_SPILL_CUSTOM lines for a lights-only MeshIR, matching the
    day/night/flash dataref selection already baked into each
    LightEntry.dataref by whichever caller built it)."""
    lines = ["I\n", "800\n", "OBJ\n", "\n"]

    if ir.tilted:
        lines.append("TILTED\n")

    if len(ir.positions):
        lines.append(f"TEXTURE {ir.texture or ''}\n\n")
        lines.append(f"POINT_COUNTS {len(ir.positions)} 0 0 {len(ir.indices)}\n\n")
        for (x, y, z), (nx, ny, nz), (u, v) in zip(ir.positions, ir.normals, ir.uvs):
            lines.append(f"VT {x:.5f} {y:.5f} {z:.5f} {nx:.5f} {ny:.5f} {nz:.5f} {u:.5f} {v:.5f}\n")
        lines.append("\n")
        for i in ir.indices:
            lines.append(f"IDX {int(i)}\n")
        lines.append("\n")

        if ir.double_sided:
            lines.append("ATTR_no_cull\n")
        if ir.alpha_mode == "BLEND":
            lines.append("ATTR_blend\n")
            lines.append("ATTR_shiny_rat 0.5\n")
        elif ir.alpha_mode == "MASK":
            lines.append(f"ATTR_no_blend {ir.alpha_cutoff:.3f}\n")
            lines.append("ATTR_shiny_rat 0.5\n")
        else:
            lines.append("ATTR_no_blend 0.5\n")
            lines.append("ATTR_shiny_rat 0.5\n")

        if ir.draped:
            lines.append("ATTR_draped\n")
            if ir.draped_layer_offset is not None:
                lines.append(
                    f"ATTR_layer_group_draped {ir.draped_layer_group or 'markings'} "
                    f"{ir.draped_layer_offset}\n")

        lines.append(f"TRIS 0 {len(ir.indices)}\n")
    else:
        lines.append("TEXTURE \n\n")
        lines.append("POINT_COUNTS 0 0 0 0\n\n")

    import math
    for light in ir.lights:
        px, py, pz = light.pos
        if light.named_light:
            lines.append(f"LIGHT_NAMED {light.named_light} {px:.5f} {py:.5f} {pz:.5f}\n")
            continue
        dx, dy, dz = light.dir
        r, g, b = light.color
        cone = max(0.0, min(360.0, light.cone_angle))
        semi = 1.0 if cone >= 359.0 else math.cos(math.radians(cone / 2.0))
        # Steady scenery lights use X-Plane's registered param light
        # `full_custom_halo_night` (SPILL_HW_DIR, night-gated in-engine, no
        # dataref) -- convert.py stashes the param-light name in `.dataref`.
        # Same 12 numbers as LIGHT_SPILL_CUSTOM: px py pz R G B A S X Y Z F.
        pl = light.dataref if str(light.dataref).startswith("full_custom_halo") else "full_custom_halo_night"
        lines.append(
            f"LIGHT_PARAM {pl} {px:.5f} {py:.5f} {pz:.5f} "
            f"{r:.4f} {g:.4f} {b:.4f} 1.0 {light.size:.3f} "
            f"{dx:.5f} {dy:.5f} {dz:.5f} {semi:.4f}\n"
        )

    obj_path.write_text("".join(lines), encoding="utf-8")
