"""
X-Plane .pol (draped polygon) file writer.

One minimal DRAPED_POLYGON .pol file per (texture, layer_group) pair --
memoized per run, since many polygon INSTANCES across many tiles/placements
share one .pol file (unlike .obj files, which are per-placement/merge-result
and never shared this way).

ALWAYS uses DSF's explicit-per-vertex-UV polygon mode (BEGIN_POLYGON's own
param field set to 65535 at the call site in dsf_compiler.py), never the
SCALE-driven auto-tiling mode -- every draped surface here already carries
real per-vertex UV data from its source glTF mesh (mesh_ir.MeshIR.uvs),
including atlas-cropped signage (same shape as Laminar's own
lib/airport/signs/DrapedDirSigns.pol). Auto-tiling only matters for a
polygon with no per-vertex UVs at all, which never applies here.

Still declares SCALE for consistency with real .pol files, even though
its value is spec-documented to be ignored when param is 65535.
"""

from pathlib import Path

_POL_CACHE = {}  # (texture_name, layer_group) -> "polygons/<name>.pol", memoized per run


def clear_cache():
    """Test-only reset of the per-run memoization."""
    _POL_CACHE.clear()


def write_pol_for_texture(polygons_dir: Path, texture_name: str, layer_group: str = "markings") -> str:
    """Writes (once) a minimal .pol file referencing texture_name, under
    polygons_dir (a sibling of the run's objects/ and textures/ folders).
    Returns the .pol's own path relative to the output package root (e.g.
    "polygons/foo.pol"), for use as a POLYGON_DEF/dsf_compiler.build_dsf
    "pol_path" entry -- the direct polygon analog of bgl_extractor/
    dsf_compiler's existing obj_ref() convention for .obj files.

    Idempotent both within one process (the in-memory cache) and across
    separate runs/process-pool workers (skips writing if the file is
    already on disk) -- multiple placements sharing one texture must never
    race to write conflicting content to the same .pol path."""
    polygons_dir = Path(polygons_dir)
    key = (texture_name, layer_group)
    cached = _POL_CACHE.get(key)
    if cached is not None:
        return cached

    pol_name = f"{Path(texture_name).stem}.pol"
    pol_path = polygons_dir / pol_name
    if not pol_path.exists():
        polygons_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "A",
            "850",
            "DRAPED_POLYGON",
            "",
            f"TEXTURE ../textures/{texture_name}",
            f"LAYER_GROUP {layer_group} 0",
            # SCALE is spec-ignored whenever a polygon instance's own param
            # is 65535 (this module's whole reason for existing), but real
            # .pol files still declare it -- 1 1 is an inert placeholder,
            # never actually read for content this project writes.
            "SCALE 1 1",
            "",
        ]
        pol_path.write_text("\n".join(lines), encoding="utf-8")

    rel = f"polygons/{pol_name}"
    _POL_CACHE[key] = rel
    return rel
