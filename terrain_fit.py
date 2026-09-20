"""
Large-object terrain-fit workaround.

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
whether the group as a whole qualifies (large enough, terrain data
available, not AGL-mounted -- AGL placements already carry an explicit
runtime elevation via DSF's AGL pool, so warping local Y on top would
double up rather than correct anything).

Consumes each sibling's .meshir.pkl sidecar (mesh_convert.mesh_ir) --
real float64 numpy arrays from convert()'s own in-memory data, not a
reparse of the .obj text. A stem with no sidecar (animated/blink
objects) is treated as disqualified.

RIGID (non-draped) siblings are never per-vertex-warped: independently
shearing every closely-packed vertex of architectural detail (window
mullions, cornices) reads as visibly warped/jagged geometry, not a clean
tilt, even for a gentle real slope -- worse than the crude-but-rigid
single-point TILTED rotation it would replace. Only draped siblings (and
the "_lights" sibling's point positions) get the per-vertex warp.

Rigid siblings instead get ONE rigid rotation (a single 3x3 matrix,
preserving every internal distance) fit against several real terrain
samples across the group's shared footprint, rather than X-Plane's own
runtime TILTED, which rotates around only the one point it samples at
the placement anchor and extrapolates that slope across the whole
footprint -- for a large building a single sampled point can diverge
from the real average slope enough to leave one end floating or sunk.
Both this and TILTED are the same mechanism (one rigid rotation around
the local origin); this just feeds it a better-sampled slope, carrying
none of the per-vertex shear risk. When the fitted plane is negligible,
a rigid sibling is left untouched, TILTED flag and all.

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

_AREA_THRESHOLD_M2 = 300.0
_MIN_SIDE_M = 10.0
# Upper bound: gates RIGID-TILT eligibility -- a single rotation around
# one shared local origin is only a sound model when the group genuinely
# IS one rigid building. A source glTF can bundle a building's real
# structure together with unrelated nearby ground/decal geometry (grass,
# asphalt, parking-line decals), reporting a combined footprint far
# larger (640x640 m observed) than any single real building. Rotating
# such a group can't tear it apart internally, but a separately-modelled
# "interior" object anchored at the same real-world point (too small to
# qualify on its own) stays unrotated and visibly separates from the
# wrongly-rotated exterior. 200 m comfortably covers any real building
# this project has converted (including large terminals/hangars) while
# excluding a 640 m bundled-content outlier.
_MAX_SIDE_M = 200.0
_NOISE_FLOOR_M = 0.10  # skip a group whose 4 bbox-corner samples all correct by less than this

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


_RIGID_TILT_MIN_SIN_THETA = 1e-6  # below this the fitted plane is flat enough that rotating would be a no-op
_RIGID_TILT_MAX_DISPLACEMENT_M = 5.0  # reject a rotation that would move any sampled corner further than this

# --- Foundation skirt -------------------------------------------------------
# The rigid tilt fits the WHOLE building to a best-fit plane through the
# footprint samples -- exact on a planar slope, but rolling ground still
# averages out, leaving the base floating/digging in where the real
# contour deviates. The skirt fixes just the ground-contact band: every
# vertex within _SKIRT_BAND_M of the base gets Y nudged by the residual
# the plane missed at its (x,z), ramped to zero over _SKIRT_BLEND_M above
# the band so there's no crease with the rigid superstructure above.
# Vertical walls stay vertical (top/bottom share an x,z, so the wall just
# grows/shrinks to meet the ground); a flat floor only picks up the small
# residual near its edges. Per-residual clamp guards a lone DSF spike.
_FOUNDATION_SKIRT = True
_SKIRT_BAND_M = 0.6     # full residual conform within this height of the base
_SKIRT_BLEND_M = 2.0    # residual weight ramps 1 -> 0 from _SKIRT_BAND_M to _SKIRT_BAND_M + this
_SKIRT_MAX_M = 1.0      # clamp on a single vertex's residual nudge


def _fit_rigid_tilt_rotation(corner_samples):
    """corner_samples: [(local_x, local_z, elevation_delta), ...] real
    terrain samples across a group's shared footprint. Returns a 3x3
    rotation matrix (numpy float64) rigidly rotating the local up-vector
    (0,1,0) to match the best-fit real-terrain plane's normal, or None
    if the samples don't support a stable fit (fewer than 3, degenerate/
    collinear footprint) or the fitted tilt is negligible.

    Fitted plane: delta = A*local_x + B*local_z through the origin (the
    object's own placement anchor, where delta(0,0) = 0 by construction,
    so no intercept term). (A, B) is the least-squares fit; the normal
    is normalize((-A, 1, -B)), the standard gradient-to-normal relation.

    The returned rotation is the shortest rotation taking (0,1,0) to that
    normal (Rodrigues' formula) -- the same mechanism X-Plane's own
    runtime TILTED directive uses; this just feeds it a better-sampled
    slope, see the module docstring."""
    if len(corner_samples) < 3:
        return None
    xs = np.array([s[0] for s in corner_samples], dtype=np.float64)
    zs = np.array([s[1] for s in corner_samples], dtype=np.float64)
    ds = np.array([s[2] for s in corner_samples], dtype=np.float64)

    design = np.stack([xs, zs], axis=1)
    try:
        coeffs, _residuals, rank, _sv = np.linalg.lstsq(design, ds, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 2:
        return None  # degenerate footprint (e.g. every sample collinear) -- no stable plane

    a_coef, b_coef = float(coeffs[0]), float(coeffs[1])
    normal = np.array([-a_coef, 1.0, -b_coef], dtype=np.float64)
    normal_len = float(np.linalg.norm(normal))
    if normal_len < 1e-9:
        return None
    normal = normal / normal_len

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis = np.cross(up, normal)
    sin_theta = float(np.linalg.norm(axis))
    cos_theta = float(np.dot(up, normal))
    if sin_theta < _RIGID_TILT_MIN_SIN_THETA:
        return None  # already flat enough that rotating would be a visual no-op

    axis = axis / sin_theta
    k_mat = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    rotation = np.eye(3) + sin_theta * k_mat + (1.0 - cos_theta) * (k_mat @ k_mat)
    return rotation


def _apply_rigid_rotation(ir, rigid_rotation, plane_ab, base_lat, base_lon, heading_deg,
                           xplane_root, origin_elev, point_cache, anchor_delta_y=0.0,
                           anchor_delta_x=0.0, anchor_delta_z=0.0):
    """The rotate-plus-foundation-skirt math shared between a group's own
    rigid-tilt correction (get_or_create_fitted_group, every anchor_delta_*
    always 0 -- its siblings already share one local frame) and a
    DIFFERENT group's retroactive one (apply_shared_rotation_to_group,
    for an object anchored at the same real-world lat/lon but a
    different height, e.g. an AGL-mounted object sharing an anchor with a
    ground-anchored one) -- factored out so both use identical math.
    Returns (rotated_positions, rotated_normals); does not touch
    uvs/indices.

    A rotation is only physically correct around the pivot both objects
    actually share, which is the REFERENCE object's own local origin, not
    either object's own (0,0,0) blindly:
    - anchor_delta_y: this object's own AGL height_offset minus the
      reference's. Needed because a rotation applied naively around each
      object's own origin swings a vertically-offset object (e.g. a
      tower's ground-anchored shell vs. its cab-floor-anchored interior,
      ~42m up) sideways by height*sin(angle) relative to the real shared
      pivot -- a real, meters-scale miss for a small angle at that height.
    - anchor_delta_x/z: the same idea, horizontally. mesh_convert.
      convert() re-centers every model independently around its own
      median footprint before this module sees it, so two different
      source models sharing one raw placement anchor generally do NOT
      share a local-origin horizontally either. Passed in by main.py as
      (this object's own recenter_x/z) minus (the reference's), both
      already available from convert()'s per-model origin-offset sidecar.
      Since a tilt rotation's off-diagonal terms couple X/Z into Y, get-
      ting only the vertical half right still leaves a residual vertical
      error too.

    Algebra for all three: shift into the reference's frame (+anchor_
    delta), rotate, shift back (-anchor_delta) -- R@(v+d) - d."""
    _dx, _dy, _dz = anchor_delta_x, anchor_delta_y, anchor_delta_z
    if _dx or _dy or _dz:
        shifted = ir.positions.copy()
        shifted[:, 0] += _dx
        shifted[:, 1] += _dy
        shifted[:, 2] += _dz
        rotated = shifted @ rigid_rotation.T
        rotated[:, 0] -= _dx
        rotated[:, 1] -= _dy
        rotated[:, 2] -= _dz
    else:
        rotated = ir.positions @ rigid_rotation.T
    normals = ir.normals @ rigid_rotation.T

    # The foundation skirt conforms a GROUND-CONTACT band to real terrain
    # residuals -- meaningless (and actively wrong) for an object anchored
    # away from the reference's own ground-level local frame (vertically
    # via anchor_delta_y, or horizontally via anchor_delta_x/z -- either
    # means "terrain right under this vertex" no longer has the simple
    # relationship to the object's own base this skirt assumes). Only ever
    # applied when this object shares the reference's own frame exactly.
    if _FOUNDATION_SKIRT and plane_ab is not None and len(ir.positions) and not (_dx or _dy or _dz):
        pa, pb = plane_ab
        y_base = float(ir.positions[:, 1].min())
        band_top = _SKIRT_BAND_M + _SKIRT_BLEND_M
        for i in range(len(ir.positions)):
            h = float(ir.positions[i, 1]) - y_base
            if h >= band_top:
                continue
            lx, lz = float(ir.positions[i, 0]), float(ir.positions[i, 2])
            exact = _point_elevation_delta(base_lat, base_lon, heading_deg, lx, lz,
                                            xplane_root, origin_elev, point_cache)
            if exact is None:
                continue
            residual = exact - (pa * lx + pb * lz)
            residual = max(-_SKIRT_MAX_M, min(_SKIRT_MAX_M, residual))
            w = 1.0 if h <= _SKIRT_BAND_M else (band_top - h) / _SKIRT_BLEND_M
            rotated[i, 1] += residual * w

    return rotated, normals


def get_cached_transform(group_key):
    """Returns the {'rigid_rotation','plane_ab','base_lat','base_lon',
    'heading_deg','origin_elev'} dict get_or_create_fitted_group computed
    for group_key (rigid_rotation is None if that group didn't end up
    tilted -- oversized/negligible/etc.), or None if that group_key was
    never processed far enough to have one (disqualified/terrain_
    unavailable/no_xplane_root -- nothing usable either way). Lets a
    caller (main.py's anchor-clustering pass) find a rotation computed for
    ONE placement and apply it to a DIFFERENT placement anchored at the
    same real-world point -- see apply_shared_rotation_to_group."""
    return _group_transform_cache.get(group_key)


def apply_shared_rotation_to_group(obj_dir, obj_stems, transform, xplane_root, anchor_delta_y=0.0,
                                    anchor_delta_x=0.0, anchor_delta_z=0.0):
    """Retroactively applies an already-computed rigid rotation (from a
    different group_key's terrain-fit result at the same real-world
    anchor -- see main.py's anchor-clustering pass) to stems that came
    back disqualified/rigid_skip/negligible on their own.

    Fixes one real-world building split across multiple placements from
    different source files, which never reach the same group_key since
    they don't share a model stem (e.g. LHBP's ATC tower: an SPB-attached
    shell and a plain-BGL-placed interior at the same anchor). Reusing
    the identical rotation matrix is only correct once both objects'
    local origins are re-expressed relative to the reference's shared
    pivot -- anchor_delta_y/x/z, see _apply_rigid_rotation's docstring.

    Skips draped geometry (never rotated) and anything with no real
    position data (lights are corrected independently, regardless of
    rigid_tilt eligibility)."""
    obj_dir = Path(obj_dir)
    rigid_rotation = transform.get("rigid_rotation") if transform else None
    if rigid_rotation is None:
        return {stem: (stem, False, "not_applicable") for stem in obj_stems}

    plane_ab = transform.get("plane_ab")
    base_lat, base_lon = transform["base_lat"], transform["base_lon"]
    heading_deg, origin_elev = transform["heading_deg"], transform["origin_elev"]
    point_cache = {}

    result = {}
    for stem in obj_stems:
        ir = _load_ir(obj_dir, stem)
        if ir is None or not len(ir.positions) or ir.draped:
            result[stem] = (stem, False, "not_applicable")
            continue

        rotated, normals = _apply_rigid_rotation(
            ir, rigid_rotation, plane_ab, base_lat, base_lon, heading_deg,
            xplane_root, origin_elev, point_cache, anchor_delta_y=anchor_delta_y,
            anchor_delta_x=anchor_delta_x, anchor_delta_z=anchor_delta_z)

        digest = hashlib.md5(
            f"{base_lat}_{base_lon}_{heading_deg}_{anchor_delta_x}_{anchor_delta_y}_{anchor_delta_z}_{stem}"
            .encode("utf-8")).hexdigest()[:10]
        corrected = mesh_ir.MeshIR(
            name=f"{stem}_tfitlink_{digest}", texture=ir.texture, tilted=False,
            draped=ir.draped, draped_layer_offset=ir.draped_layer_offset,
            double_sided=ir.double_sided, alpha_mode=ir.alpha_mode, alpha_cutoff=ir.alpha_cutoff,
            footprint_area_m2=ir.footprint_area_m2, proximity_dataref=ir.proximity_dataref,
            positions=rotated, normals=normals, uvs=ir.uvs, indices=ir.indices,
        )
        fitted_path = obj_dir / f"{corrected.name}.obj"
        if not fitted_path.exists():
            mesh_ir.write_obj8(corrected, fitted_path)
            mesh_ir.save(corrected, mesh_ir.sidecar_path_for(fitted_path))
        result[stem] = (corrected.name, True, "applied_shared_rotation")

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
    y_max = max(float(loaded[s].positions[:, 1].max()) for s in geo_stems)
    dx = x_max - x_min
    dz = z_max - z_min
    if dx < _MIN_SIDE_M or dz < _MIN_SIDE_M or dx * dz < _AREA_THRESHOLD_M2:
        result = {stem: (stem, False, "disqualified") for stem in obj_stems}
        _group_cache[group_key] = result
        return result
    # Oversized footprint: only rules out the RIGID-TILT rotation below, not
    # this whole group -- a legitimately huge DRAPED decal/scattered-points
    # fixture must still come back "draped_not_warped" (never rigid-tilted
    # in the first place), not get disqualified for a misleading reason.
    oversized_footprint = dx > _MAX_SIDE_M or dz > _MAX_SIDE_M

    origin_elev = terrain_dem.get_elevation(xplane_root, base_lat, base_lon)
    if origin_elev is None:
        result = {stem: (stem, False, "terrain_unavailable") for stem in obj_stems}
        _group_cache[group_key] = result
        return result

    point_cache = {}

    # Cheap pre-check: a 3x3 grid (corners, edge midpoints, center) catches
    # the common near-flat-terrain case without a full per-vertex pass.
    # These same (x, z, delta) samples also feed _fit_rigid_tilt_rotation
    # below -- denser than 4 bare corners makes that least-squares fit
    # less sensitive to any one noisy sample (e.g. a DSF triangulation
    # seam under a single corner).
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

    rigid_rotation = _fit_rigid_tilt_rotation(corner_samples)

    # Reject the rotation for an oversized footprint (see _MAX_SIDE_M's
    # own docstring -- not about the group shearing internally, but about
    # desyncing from a separately-modelled sibling that shares its real-
    # world anchor but stays too small to qualify itself). Also reject on
    # implied displacement alone, as a second, independent guard for a
    # footprint under _MAX_SIDE_M sitting on terrain steep enough that
    # the fitted plane would move a vertex further than plausible.
    if rigid_rotation is not None and oversized_footprint:
        rigid_rotation = None
    height_rejected = False
    if rigid_rotation is not None:
        # CONFIRMED REAL BUG: this only tested displacement at Y=0 (the
        # footprint's own ground-level corners), never at the group's own
        # height -- a rotation that moves the flat ground corners a
        # plausible couple of metres can swing a TALL structure's roof far
        # more than that for the exact same angle (displacement from a
        # rotation scales with distance from the pivot, and a tower's roof
        # sits much farther from the ground-level pivot than its own base
        # does), so a genuinely excessive tilt could still pass this guard
        # for anything tall and narrow -- confirmed real symptom: large
        # buildings visibly floating/leaning after "correction". Testing
        # each XZ corner at both Y=0 and the group's own tallest point
        # closes that gap without needing a full per-vertex scan.
        max_disp = 0.0
        for sx, sz, _ in corner_samples:
            for sy in (0.0, y_max):
                v = np.array([sx, sy, sz], dtype=np.float64)
                disp = float(np.linalg.norm(rigid_rotation @ v - v))
                max_disp = max(max_disp, disp)
        if max_disp > _RIGID_TILT_MAX_DISPLACEMENT_M:
            rigid_rotation = None
            height_rejected = True

    # Same least-squares plane the rotation is derived from (delta = A*x +
    # B*z through the anchor origin) -- fit unconditionally whenever there
    # are enough samples, independent of whether the rotation itself got
    # vetoed, since the ground-skirt-only fallback right below needs a
    # real terrain estimate even when there is no rotation to pair it
    # with. Kept in coefficient form so the foundation skirt can subtract
    # the plane's prediction from the exact per-vertex terrain delta and
    # apply only the residual near the base.
    fitted_ab = None
    if len(corner_samples) >= 3:
        try:
            M = np.array([[sx, sz] for sx, sz, _ in corner_samples], dtype=np.float64)
            rhs = np.array([sd for _, _, sd in corner_samples], dtype=np.float64)
            (pa, pb), *_ = np.linalg.lstsq(M, rhs, rcond=None)
            fitted_ab = (float(pa), float(pb))
        except Exception:
            fitted_ab = None
    plane_ab = fitted_ab if rigid_rotation is not None else None

    # CONFIRMED REAL BUG (found right after the height-aware displacement
    # guard above shipped): rejecting the rotation throws away the
    # foundation skirt too, since the skirt only ever runs alongside a
    # rotation -- reverting a tall building to its dead-flat ORIGINAL
    # geometry, with zero ground-contact correction at all. For terrain
    # sloped enough to fail the roof-height check, that leaves a visible
    # gap under one whole side of the base (not just the roof) -- WORSE
    # than the excessive tilt this guard was meant to fix, and especially
    # visible through a glass facade.
    #
    # CONFIRMED REAL BUG (part 2, found on a live EGLC package): the same
    # gap exists for oversized_footprint, and it's not a rare edge case
    # -- a real, single, continuous terminal building (not bundled-
    # unrelated-content the 200m cap was meant to catch) measured 380m
    # wide, comfortably over _MAX_SIDE_M despite that constant's own
    # docstring claiming 200m "comfortably covers any real building...
    # including large terminals". _MAX_SIDE_M's whole justification for
    # rejecting the ROTATION is about a SEPARATE, differently-anchored
    # sibling desyncing from it (see _MAX_SIDE_M's own docstring) -- a
    # concern that doesn't exist for the ground skirt at all, since it
    # applies ZERO rotation (identity only): there's nothing for a
    # sibling to desync from. So both height_rejected and
    # oversized_footprint still get the base conformed to real terrain
    # -- identity rotation (no tilt) plus the skirt band using the FULL
    # sampled delta (a zero plane, not the fitted one, since there's no
    # rotation for the skirt to subtract a residual against) -- without
    # ever swinging the roof or risking an internal shear.
    ground_skirt = (height_rejected or oversized_footprint) and fitted_ab is not None

    # Cached regardless of whether rigid_rotation ended up None (an
    # explicit "this group has no rotation to offer" is as useful to a
    # cross-group lookup as a real one) -- see get_cached_transform /
    # apply_shared_rotation_to_group.
    _group_transform_cache[group_key] = {
        "rigid_rotation": rigid_rotation, "plane_ab": plane_ab,
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
        # RIGID geometry gets a single rigid rotation (see module
        # docstring for why per-vertex would shear it). Light positions
        # are corrected regardless of the sibling's draped flag -- each
        # LIGHT_SPILL_CUSTOM is an independent point, not mesh topology,
        # so moving it carries none of the shear/re-drape concerns above.
        warp_positions = bool(len(ir.positions)) and ir.draped and not skip_draped_positions
        rigid_tilt = bool(len(ir.positions)) and not ir.draped and rigid_rotation is not None
        ground_skirt_only = (bool(len(ir.positions)) and not ir.draped
                              and rigid_rotation is None and ground_skirt)
        warp_lights = bool(ir.lights)
        if not warp_positions and not rigid_tilt and not ground_skirt_only and not warp_lights:
            if ir.draped and skip_draped_positions and len(ir.positions):
                result[stem] = (stem, False, "skipped_for_polygon_mode")
            else:
                result[stem] = (stem, False, "rigid_skip" if len(ir.positions) else "disqualified")
            continue

        corrected = mesh_ir.MeshIR(
            # tilted=False whenever this sibling's geometry is corrected
            # (warp or rotation), replacing TILTED's own rotation rather
            # than stacking with it. A sibling reaching here only for its
            # lights keeps its original tilted flag untouched.
            name=f"{stem}_tfit_{digest}", texture=ir.texture,
            tilted=False if (warp_positions or rigid_tilt or ground_skirt_only) else ir.tilted, draped=ir.draped,
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
        elif rigid_tilt:
            # Rotates around the local origin (0,0,0) -- the object's own
            # placement anchor -- by one 3x3 matrix for every vertex/
            # normal, so shape stays self-consistent (rotation matrices
            # are orthogonal). Shared with apply_shared_rotation_to_
            # group's cross-group case -- see _apply_rigid_rotation.
            rotated, normals = _apply_rigid_rotation(
                ir, rigid_rotation, plane_ab, base_lat, base_lon, heading_deg,
                xplane_root, origin_elev, point_cache)
            corrected.normals = normals
            corrected.positions = rotated
            corrected.uvs = ir.uvs
            corrected.indices = ir.indices
        elif ground_skirt_only:
            # No rotation (identity) -- only the foundation skirt band
            # runs, and with a ZERO plane (not the fitted one) so its
            # residual IS the full sampled terrain delta rather than a
            # residual on top of a rotation that isn't happening here.
            rotated, normals = _apply_rigid_rotation(
                ir, np.eye(3), (0.0, 0.0), base_lat, base_lon, heading_deg,
                xplane_root, origin_elev, point_cache)
            corrected.normals = normals
            corrected.positions = rotated
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
        if ground_skirt_only and not warp_positions:
            _reason = "ground_skirt_only"
        elif rigid_tilt and not warp_positions:
            _reason = "applied_rigid_tilt"
        else:
            _reason = "applied"
        result[stem] = (corrected.name, True, _reason)

    _group_cache[group_key] = result
    return result
