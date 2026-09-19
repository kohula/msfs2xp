"""
Draped-layer draw-order ranking -- pure functions, no glTF/OBJ8 I/O, split
out from convert.py specifically so this logic (X-Plane's own documented
lack of a draw-order guarantee between draped geometry sharing the same
layer_group+offset, and the fix for it) can be unit-tested in isolation.
"""

# Descending XZ footprint area (m^2) -> ATTR_layer_group_draped offset.
# X-Plane's OBJ8 spec only allows offsets -5..+5; broader/more-general
# draped layers (a big apron color fill) get the most negative offset so
# they draw first/underneath, small/specific overlays (stand-type text,
# a paint stripe) get more positive offsets so they draw last/on top.
_DRAPED_AREA_OFFSET_THRESHOLDS = [
    (100_000.0, -5), (10_000.0, -4), (1_000.0, -3), (100.0, -2), (10.0, -1),
    (1.0, 0), (0.1, 1), (0.01, 2), (0.001, 3),
]
_DRAPED_AREA_MIN_OFFSET = 4


def draped_layer_offset(area_m2):
    """See _DRAPED_AREA_OFFSET_THRESHOLDS -- necessary because X-Plane gives
    no draw-order guarantee at all between draped geometry sharing the same
    (layer_group, offset) pair, which every draped layer used to share
    (offset was always hardcoded to 0). Only used as a fallback by
    rank_draped_layer_offsets below once a single file has more distinct
    draped layers than the -5..+5 range has slots for -- see there for why
    fixed thresholds alone aren't enough."""
    for threshold, offset in _DRAPED_AREA_OFFSET_THRESHOLDS:
        if area_m2 >= threshold:
            return offset
    return _DRAPED_AREA_MIN_OFFSET


def rank_draped_layer_offsets(builder_areas):
    """builder_areas: {builder_key: footprint_area_m2} for every draped
    builder in ONE converted file. Returns {builder_key: offset}.

    draped_layer_offset's fixed area-thresholds bucket by ORDER OF
    MAGNITUDE, so two layers with genuinely different footprints (a
    5,000 sq m apron fill and a 3,000 sq m taxiway strip belonging to the
    same MSFS asset) can still land in the identical bucket -- and with
    no draw-order guarantee between draped geometry sharing the same
    (layer_group, offset), that's exactly a same-tier flicker/occlusion
    collision, not a near miss. Ranking every draped layer IN THIS FILE
    by footprint area instead (largest = drawn first/underneath = most
    negative offset, same directional convention as the threshold scheme)
    and spacing them one full integer apart guarantees every one of them
    gets its own distinct slot, as long as there are at most 11 (the
    whole -5..+5 range) -- which covers the common case this exists for,
    a handful of stacked layers belonging to one MSFS asset (a base fill,
    a few paint stripes, some text).

    Falls back to the fixed-threshold scheme (which can't guarantee
    distinctness, but never runs out of buckets either) once a file has
    MORE than 11 draped layers -- a rare case this can't fully solve
    without spilling into other layer groups (see the ATTR_layer_group_draped
    write site for why that whole-tier-jump tradeoff was reverted), so it
    degrades gracefully there instead of raising.

    Does NOT solve collisions BETWEEN separate converted files whose
    placements happen to physically overlap (each file's ranking starts
    fresh) -- draped_merge.py exists specifically to close that gap by
    merging same-texture objects across files within one DSF tile.
    """
    if not builder_areas:
        return {}
    if len(builder_areas) > 11:
        return {key: draped_layer_offset(area) for key, area in builder_areas.items()}

    ordered = sorted(builder_areas.items(), key=lambda kv: kv[1], reverse=True)
    return {key: -5 + i for i, (key, _area) in enumerate(ordered)}
