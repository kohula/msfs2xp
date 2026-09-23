"""
Rigid-object terrain-fit workaround.

A rigid MSFS building mesh is authored assuming flat ground under its
whole footprint (X-Plane's OBJECT placement only gives it one drape
point). Real X-Plane terrain often isn't flat under a large footprint,
so one side of a big building can visibly float or sink once placed.
This samples real ground elevation (terrain_dem.py) at each vertex's own
real-world position and moves that vertex's Y by the difference from the
object's own drape point -- at whatever resolution the DSF elevation
raster itself supports (already bilinearly interpolated per point, so
there's no coarser grid step to approximate away).

Draped ground-marking/pavement geometry gets the same treatment: ATTR_
draped re-projects it onto real terrain at render time regardless of
authored Y, so this doesn't change the main render, but it keeps
authored Y close to the real contour -- X-Plane 12's shadow pass reads
authored Y even though the beauty pass doesn't, so a flat Y produced a
shadow tracking a flat plane instead of the real sloped surface.

PER-VERTEX-EXACT SAMPLING, NOT A SHARED-BBOX GRID: draped_merge.py runs
after this module and welds adjacent draped objects by snapping near-
coincident vertices. Two adjacent pieces come from different groups, each
processed by a separate call here -- sampling relative to each group's
own local bbox grid produced slightly different Y at the same real-world
boundary point, just enough to push originally-coincident vertices
outside the weld tolerance. Sampling each vertex's own (lat, lon)
directly is group-independent, so coincident vertices stay coincident.

This is a shear/warp, not a physically exact re-drape: normals are left
unchanged and only Y moves, an accepted simplification since corrections
are on the order of a building's own footprint-scale terrain variation.

CORRECTED PER MODEL, NOT PER SUB-OBJECT: mesh_convert's convert() splits
one glTF model into several .obj files (one per material/animated node,
plus a "_lights" one). get_or_create_fitted_group() takes every sibling
stem from one placement and computes ONE shared footprint bbox to decide
whether the group as a whole qualifies (terrain data available, not
AGL-mounted -- AGL placements already carry an explicit runtime
elevation via DSF's AGL pool, so warping local Y on top would double up
rather than correct anything).

Consumes each sibling's .meshir.pkl sidecar (mesh_convert.mesh_ir) --
real float64 numpy arrays from convert()'s own in-memory data, not a
reparse of the .obj text. A stem with no sidecar (animated/blink
objects) is treated as disqualified.

RIGID (non-draped) siblings are never per-vertex-warped: independently
shearing every closely-packed vertex of architectural detail (window
mullions, cornices) reads as visibly warped/jagged geometry, not a clean
tilt, even for a gentle real slope. Only draped siblings (and the
"_lights" sibling's point positions) get the per-vertex warp.

Rigid siblings instead get ONE uniform vertical shift -- every vertex
moves by the exact same amount, a pure translation that preserves shape
and normals exactly (no rotation at all). This replaced an earlier
rotation-based correction (fit a plane to terrain samples, rotate the
whole rigid object to match it): rotation only corrects a genuine TILT,
never a uniform height-offset error -- a rotation around the object's
own origin can't move that origin itself, so if the anchor's own real
elevation differs from what the object assumes, every part of it stays
wrong by that same amount no matter how well the tilt is fit.

No size gate: mesh_convert.convert() used to leave small/medium objects
to X-Plane's own runtime TILTED directive instead of this module's
shift, on the theory that a small footprint's single sampled point is
"representative enough" for a rotation to work with. CONFIRMED REAL
REGRESSION: TILTED is a rotation, so it inherits the same "can't fix an
anchor-offset" blind spot regardless of object size -- an object whose
own anchor elevation just doesn't match X-Plane's real terrain sat
wrong at ANY size, and TILTED being the only correction for anything
under the old ~300m2 gate meant that case silently never got fixed for
most objects. mesh_convert.convert() no longer emits TILTED at all;
every rigid group that reaches this module gets the shift instead,
regardless of footprint size.

The shift itself is a ROBUST estimate, not a single point: several real
terrain samples are taken across the group's shared footprint (the same
grid used for the qualifying negligible-check), a median-absolute-
deviation outlier rejection throws out any sample that disagrees sharply
with the rest (a lone DSF triangulation seam or DEM spike under one
corner), and the mean of what's left is the one number applied to every
rigid vertex in the group.

Terrain sampling failures degrade gracefully: no X-Plane install/py7zr/
origin elevation skips the whole group (no reference, no correction);
an individual vertex's own sample being unavailable (NODATA, tile edge)
only leaves that vertex unwarped. This is a best-effort visual
improvement, never something that should block a conversion run.
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np

import terrain_dem
from geo_transform import local_offset_to_latlon
from mesh_convert import mesh_ir

_NOISE_FLOOR_M = 0.10  # skip a group whose sampled corners all correct by less than this
_SHIFT_OUTLIER_MAD_K = 3.0  # modified-z-score cutoff (see _robust_vertical_shift) for rejecting a spiky sample

_ir_cache = {}       # obj_stem -> loaded MeshIR, or None if no sidecar / TILTED / no geometry+lights
_group_cache = {}     # (tuple(sorted(obj_stems)), lat_r, lon_r, hdg_r) -> {obj_stem: (result_stem, applied, reason)}
_group_transform_cache = {}  # group_key -> transform dict (see get_or_create_fitted_group), or absent if never reached that far


def _load_ir(obj_dir: Path, obj_stem: str):
    if obj_stem in _ir_cache:
        return _ir_cache[obj_stem]
    sidecar = mesh_ir.sidecar_path_for(obj_dir / f"{obj_stem}.obj")
    ir = None
    if sidecar.exists():
        try:
            loaded = mesh_ir.load(sidecar)
            if len(loaded.positions) or loaded.lights:
                ir = loaded
        except (OSError, EOFError, pickle.UnpicklingError):
            ir = None
    _ir_cache[obj_stem] = ir
    return ir


def _point_elevation_delta(base_lat, base_lon, heading_deg, local_x, local_z, xplane_root, origin_elev, cache):
    """Real elevation at this exact real-world point minus origin_elev, or
    None if unavailable (NODATA, tile missing, ...). See the module
    docstring for why this samples each point directly instead of
    interpolating within a shared-bbox grid. cache: a dict scoped to one
    get_or_create_fitted_group() call, keyed by (local_x, local_z) rounded
    to the centimeter -- multiple vertices (across siblings, or shared
    mesh edges) commonly land on the exact or near-exact same local point,
    and a centimeter is already far below both _NOISE_FLOOR_M and
    draped_merge's own weld tolerance, so reusing the sample is safe."""
    key = (round(local_x, 2), round(local_z, 2))
    if key in cache:
        return cache[key]
    lat, lon = local_offset_to_latlon(base_lat, base_lon, heading_deg, local_x, local_z)
    elev = terrain_dem.get_elevation(xplane_root, lat, lon)
    delta = None if elev is None else (elev - origin_elev)
    cache[key] = delta
    return delta


def _robust_vertical_shift(corner_samples, mad_k=_SHIFT_OUTLIER_MAD_K):
    """corner_samples: [(local_x, local_z, elevation_delta), ...] real
    terrain samples across a group's shared footprint. Returns a single
    float -- the one vertical shift to apply to every rigid vertex in the
    group -- or None if there are no samples at all.

    Robust, not a single point: a lone DSF triangulation seam or DEM spike
    under one sample would otherwise skew a plain average by exactly as
    much as it's wrong. Rejects outliers via a modified z-score against
    the median (median absolute deviation, scaled by the standard 1.4826
    constant so MAD is comparable to a normal distribution's std-dev),
    then returns the mean of whatever samples survive. Falls back to
    using every sample if the rejection would leave nothing (e.g. exactly
    two samples that disagree with each other -- no way to tell which one
    is the outlier, so trust both rather than neither) or if there's no
    spread to reject against at all (every sample agrees)."""
    if not corner_samples:
        return None
    deltas = np.array([d for _, _, d in corner_samples], dtype=np.float64)
    if len(deltas) < 3:
        return float(np.mean(deltas))
    med = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - med)))
    if mad < 1e-9:
        return med
    modified_z = np.abs(deltas - med) / (1.4826 * mad)
    clean = deltas[modified_z <= mad_k]
    if len(clean) == 0:
        clean = deltas
    return float(np.mean(clean))


def _apply_vertical_shift(ir, vertical_shift):
    """Pure translation: every vertex's Y moves by the same amount,
    normals untouched (a translation doesn't rotate anything). Returns
    shifted_positions; does not touch normals/uvs/indices."""
    shifted = ir.positions.copy()
    shifted[:, 1] += vertical_shift
    return shifted


def get_cached_transform(group_key):
    """Returns the {'vertical_shift','base_lat','base_lon','heading_deg',
    'origin_elev'} dict get_or_create_fitted_group computed for group_key
    (vertical_shift is None if that group had no usable samples), or None
    if that group_key was never processed far enough to have one
    (disqualified/terrain_unavailable/no_xplane_root -- nothing usable
    either way). Lets a caller (main.py's anchor-clustering pass) find a
    shift computed for ONE placement and apply it to a DIFFERENT
    placement anchored at the same real-world point -- see
    apply_shared_shift_to_group."""
    return _group_transform_cache.get(group_key)


def apply_shared_shift_to_group(obj_dir, obj_stems, transform, xplane_root):
    """Retroactively applies an already-computed vertical shift (from a
    different group_key's terrain-fit result at the same real-world
    anchor -- see main.py's anchor-clustering pass) to stems that came
    back disqualified/rigid_skip/negligible on their own.

    Fixes one real-world building split across multiple placements from
    different source files, which never reach the same group_key since
    they don't share a model stem (e.g. LHBP's ATC tower: an SPB-attached
    shell and a plain-BGL-placed interior at the same anchor). Unlike the
    rotation this replaced, no anchor-delta bookkeeping is needed here:
    the shift is a property of the shared real-world anchor point (how
    much real terrain differs from assumed, AT THAT LAT/LON), not of
    either object's own local coordinate convention, so the identical
    number applies correctly regardless of the two objects' respective
    local origins, recenter offsets, or (non-AGL) height conventions.

    Skips draped geometry (never shifted this way) and anything with no
    real position data (lights are corrected independently, regardless
    of rigid-shift eligibility)."""
    obj_dir = Path(obj_dir)
    vertical_shift = transform.get("vertical_shift") if transform else None
    if vertical_shift is None:
        return {stem: (stem, False, "not_applicable") for stem in obj_stems}

    result = {}
    for stem in obj_stems:
        ir = _load_ir(obj_dir, stem)
        if ir is None or not len(ir.positions) or ir.draped:
            result[stem] = (stem, False, "not_applicable")
            continue

        shifted = _apply_vertical_shift(ir, vertical_shift)

        digest = hashlib.md5(f"{vertical_shift}_{stem}".encode("utf-8")).hexdigest()[:10]
        corrected = mesh_ir.MeshIR(
            name=f"{stem}_tfitlink_{digest}", texture=ir.texture, tilted=False,
            draped=ir.draped, draped_layer_offset=ir.draped_layer_offset,
            double_sided=ir.double_sided, alpha_mode=ir.alpha_mode, alpha_cutoff=ir.alpha_cutoff,
            footprint_area_m2=ir.footprint_area_m2, proximity_dataref=ir.proximity_dataref,
            positions=shifted, normals=ir.normals, uvs=ir.uvs, indices=ir.indices,
        )
        fitted_path = obj_dir / f"{corrected.name}.obj"
        if not fitted_path.exists():
            mesh_ir.write_obj8(corrected, fitted_path)
            mesh_ir.save(corrected, mesh_ir.sidecar_path_for(fitted_path))
        result[stem] = (corrected.name, True, "applied_shared_shift")

    return result


def get_or_create_fitted_group(obj_dir, obj_stems, base_lat, base_lon, heading_deg, xplane_root,
                                skip_draped_positions=False):
    """obj_stems: every sibling .obj stem generated from ONE original
    model for ONE placement (main.py's own converted_stems_map[original_stem]
    list). Returns {obj_stem: (result_stem, applied, reason)}, one entry
    per input stem -- result_stem is obj_stem itself wherever no
    correction was applied to that particular file.

    skip_draped_positions: when True, draped siblings' per-vertex position
    warp is skipped entirely (rigid siblings and lights are still
    corrected normally) -- for the .pol/DSF-polygon conversion path,
    where a real DRAPED_POLYGON has no elevation field of its own (X-Plane
    projects it onto sampled terrain natively) and draped_merge.
    _triangles_to_polygons never reads Y when serializing, so this warp's
    output would be silently discarded anyway; skipping it avoids paying
    for per-vertex sampling that can't affect that render path."""
    obj_dir = Path(obj_dir)

    if xplane_root is None:
        return {stem: (stem, False, "no_xplane_root") for stem in obj_stems}

    lat_r, lon_r, hdg_r = round(base_lat, 6), round(base_lon, 6), round(heading_deg, 2)
    group_key = (tuple(sorted(obj_stems)), lat_r, lon_r, hdg_r, bool(skip_draped_positions))
    if group_key in _group_cache:
        return _group_cache[group_key]

    loaded = {stem: _load_ir(obj_dir, stem) for stem in obj_stems}
    geo_stems = [s for s, ir in loaded.items() if ir is not None and len(ir.positions)]
    if not geo_stems:
        result = {stem: (stem, False, "disqualified") for stem in obj_stems}
        _group_cache[group_key] = result
        return result

    x_min = min(float(loaded[s].positions[:, 0].min()) for s in geo_stems)
    x_max = max(float(loaded[s].positions[:, 0].max()) for s in geo_stems)
    z_min = min(float(loaded[s].positions[:, 2].min()) for s in geo_stems)
    z_max = max(float(loaded[s].positions[:, 2].max()) for s in geo_stems)

    origin_elev = terrain_dem.get_elevation(xplane_root, base_lat, base_lon)
    if origin_elev is None:
        result = {stem: (stem, False, "terrain_unavailable") for stem in obj_stems}
        _group_cache[group_key] = result
        return result

    point_cache = {}

    # A denser grid than 4 bare corners (corners, edge midpoints, center)
    # feeds _robust_vertical_shift's outlier rejection more samples to
    # work with -- important now that ANY one of them being a spike could
    # otherwise pull the single shift value applied to the whole group.
    x_mid = (x_min + x_max) / 2.0
    z_mid = (z_min + z_max) / 2.0
    grid_points = [(gx, gz) for gx in (x_min, x_mid, x_max) for gz in (z_min, z_mid, z_max)]
    corner_samples = [
        (gx, gz, d) for gx, gz, d in (
            (gx, gz, _point_elevation_delta(base_lat, base_lon, heading_deg, gx, gz, xplane_root, origin_elev, point_cache))
            for gx, gz in grid_points
        ) if d is not None
    ]
    if corner_samples and max(abs(d) for _, _, d in corner_samples) < _NOISE_FLOOR_M:
        result = {stem: (stem, False, "negligible") for stem in obj_stems}
        _group_cache[group_key] = result
        return result

    vertical_shift = _robust_vertical_shift(corner_samples)

    # Cached regardless of whether vertical_shift ended up None (an
    # explicit "this group has no correction to offer" is as useful to a
    # cross-group lookup as a real one) -- see get_cached_transform /
    # apply_shared_shift_to_group.
    _group_transform_cache[group_key] = {
        "vertical_shift": vertical_shift,
        "base_lat": base_lat, "base_lon": base_lon, "heading_deg": heading_deg,
        "origin_elev": origin_elev,
    }

    digest = hashlib.md5(f"{lat_r}_{lon_r}_{hdg_r}".encode("utf-8")).hexdigest()[:10]

    result = {}
    for stem in obj_stems:
        ir = loaded[stem]
        if ir is None:
            result[stem] = (stem, False, "not_applicable")
            continue

        # CONFIRMED REAL REGRESSION: this was hardcoded to False, on the
        # reasoning that a per-vertex warp of draped geometry is invisible
        # in-sim (X-Plane re-projects every ATTR_draped surface onto the
        # compiled terrain mesh at render time regardless of the OBJ's own
        # Y) -- true for the main beauty-pass render, but X-Plane 12's
        # shadow pass reads authored Y even though the beauty pass
        # doesn't, so a flat/un-warped Y produced a shadow tracking a
        # flat plane instead of the real (often gently sloped) surface
        # underneath it. Confirmed against an older working snapshot
        # (predates this regression) whose own comment on this exact line
        # names the real-world symptom directly: making draped siblings
        # reach this warp -- and therefore the corrected-copy + sidecar
        # write below, which is what makes them visible to draped_merge.
        # py's own welding/dedup pass -- is what fixed "duplicated/
        # unwelded pavement". `skip_draped_positions` (.pol mode) still
        # skips it: a real DRAPED_POLYGON has no elevation field of its
        # own at all, and draped_merge._triangles_to_polygons never reads
        # Y when serializing, so the warp's output would be silently
        # discarded downstream anyway in that mode.
        # RIGID geometry gets ONE uniform vertical shift, the same number
        # for every vertex (see module docstring for why a rotation-only
        # correction can't fix a uniform height-offset error, and why
        # per-vertex warping would shear rigid architectural detail).
        # Light positions are corrected regardless of the sibling's
        # draped flag -- each LIGHT_SPILL_CUSTOM is an independent point,
        # not mesh topology, so moving it carries none of the shear/
        # re-drape concerns above.
        warp_positions = bool(len(ir.positions)) and ir.draped and not skip_draped_positions
        rigid_shift = bool(len(ir.positions)) and not ir.draped and vertical_shift is not None
        warp_lights = bool(ir.lights)
        if not warp_positions and not rigid_shift and not warp_lights:
            if ir.draped and skip_draped_positions and len(ir.positions):
                result[stem] = (stem, False, "skipped_for_polygon_mode")
            else:
                result[stem] = (stem, False, "rigid_skip" if len(ir.positions) else "disqualified")
            continue

        corrected = mesh_ir.MeshIR(
            # tilted=False whenever this sibling's geometry is corrected
            # (warp or shift), replacing TILTED's own rotation rather
            # than stacking with it. A sibling reaching here only for its
            # lights keeps its original tilted flag untouched.
            name=f"{stem}_tfit_{digest}", texture=ir.texture,
            tilted=False if (warp_positions or rigid_shift) else ir.tilted, draped=ir.draped,
            draped_layer_offset=ir.draped_layer_offset, double_sided=ir.double_sided,
            alpha_mode=ir.alpha_mode, alpha_cutoff=ir.alpha_cutoff,
            footprint_area_m2=ir.footprint_area_m2, proximity_dataref=ir.proximity_dataref,
        )
        if warp_positions:
            warped = ir.positions.copy()
            for i in range(len(warped)):
                d = _point_elevation_delta(
                    base_lat, base_lon, heading_deg,
                    float(ir.positions[i, 0]), float(ir.positions[i, 2]),
                    xplane_root, origin_elev, point_cache)
                if d is not None:
                    warped[i, 1] += d
            corrected.positions = warped
            corrected.normals = ir.normals
            corrected.uvs = ir.uvs
            corrected.indices = ir.indices
        elif rigid_shift:
            # A pure translation -- every vertex moves by the identical
            # amount, shape and normals both entirely unchanged. Shared
            # with apply_shared_shift_to_group's cross-group case -- see
            # _apply_vertical_shift.
            corrected.positions = _apply_vertical_shift(ir, vertical_shift)
            corrected.normals = ir.normals
            corrected.uvs = ir.uvs
            corrected.indices = ir.indices
        elif len(ir.positions):
            corrected.positions = ir.positions
            corrected.normals = ir.normals
            corrected.uvs = ir.uvs
            corrected.indices = ir.indices

        if warp_lights:
            new_lights = []
            for light in ir.lights:
                px, py, pz = light.pos
                d = _point_elevation_delta(base_lat, base_lon, heading_deg, px, pz, xplane_root, origin_elev, point_cache)
                new_lights.append(mesh_ir.LightEntry(
                    pos=(px, py + (d or 0.0), pz), dir=light.dir, color=light.color,
                    cone_angle=light.cone_angle, size=light.size, dataref=light.dataref,
                    named_light=light.named_light,
                ))
            corrected.lights = new_lights

        fitted_path = obj_dir / f"{corrected.name}.obj"
        if not fitted_path.exists():
            mesh_ir.write_obj8(corrected, fitted_path)
            # Also write a MeshIR sidecar, same as convert() does --
            # draped_merge.py's _load_candidate only reads .meshir.pkl,
            # never the .obj text, so a corrected draped stem without one
            # silently falls back to passthrough (never welded/deduped).
            mesh_ir.save(corrected, mesh_ir.sidecar_path_for(fitted_path))
        if rigid_shift and not warp_positions:
            _reason = "applied_vertical_shift"
        else:
            _reason = "applied"
        result[stem] = (corrected.name, True, _reason)

    _group_cache[group_key] = result
    return result
