"""Flat objects lying on the airport ground: small ones dropped, large ones
draped.

MSFS airports carry many very low objects that sit flush with MSFS's own
level airport ground -- pit and drain covers, in-pavement fixtures, cable
plates, and large ground-cover sheets. In X-Plane the ground there is
X-Plane's terrain, not MSFS's, so a rigid flat object hangs at its anchor's
ground height: small ones hover a little over the grass and pavement, and
a sheet hundreds of metres wide hovers metres above every dip like a
ceiling. Inside the airport (the apt.dat boundary, or a runway strip):

- a flat object (solid geometry under MAX_HEIGHT_M tall, standing on the
  ground in MSFS -- its height above ground under GROUND_AGL_M -- with no
  lights) up to MAX_SIDE_M across is dropped: only centimetres tall, it
  adds nothing seen from a cockpit;
- a larger one is draped instead (ATTR_draped, under the pavement layers),
  so it lies on X-Plane's ground like the rest of the pavement.

Aircraft, vehicles, signs, light fixtures and anything resting on a
building (a prop set on a building's floor) are kept as they are; draped
geometry is never touched.
"""

import math

from geo_transform import metres_per_degree

MAX_HEIGHT_M = 0.5
MAX_SIDE_M = 15.0
GROUND_AGL_M = 1.0
EDGE_MARGIN_M = 10.0
END_MARGIN_M = 30.0
DRAPED_LAYER_GROUP = "shoulders"  # below "taxiways"/"runways"/"markings"
DRAPED_LAYER_OFFSET = -5


class RunwayStrip:
    def __init__(self, lat, lon, heading_deg, half_length, half_width):
        self.lat, self.lon = lat, lon
        h = math.radians(heading_deg)
        self.sin_h, self.cos_h = math.sin(h), math.cos(h)
        self.half_length = half_length
        self.half_width = half_width
        self.m_lat, self.m_lon = metres_per_degree(lat)

    def contains(self, lat, lon):
        east = (lon - self.lon) * self.m_lon
        north = (lat - self.lat) * self.m_lat
        along = east * self.sin_h + north * self.cos_h
        across = east * self.cos_h - north * self.sin_h
        return abs(along) <= self.half_length and abs(across) <= self.half_width


def runway_strips(layout, edge_margin_m=EDGE_MARGIN_M, end_margin_m=END_MARGIN_M):
    """One RunwayStrip per land runway of an airport_layout.AirportLayout."""
    strips = []
    for rw in getattr(layout, "runways", None) or []:
        if not (rw.length_m > 1.0 and rw.width_m > 0.0):
            continue
        ends = max(rw.primary.overrun_m, rw.primary.blast_pad_m,
                   rw.secondary.overrun_m, rw.secondary.blast_pad_m, 0.0)
        strips.append(RunwayStrip(rw.lat, rw.lon, rw.heading_true,
                                  rw.length_m / 2.0 + ends + end_margin_m,
                                  rw.width_m / 2.0 + edge_margin_m))
    return strips


def on_runway(lat, lon, strips):
    return any(s.contains(lat, lon) for s in strips)


def _inside(ring, lat, lon):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        ai, oi = ring[i]
        aj, oj = ring[j]
        if (ai > lat) != (aj > lat) and lon < oi + (lat - ai) * (oj - oi) / (aj - ai):
            inside = not inside
        j = i
    return inside


class AirportGround:
    """The airport's ground: inside its boundary ring [(lat, lon), ...]
    (apt_native.airport_boundary) or on a runway strip."""

    def __init__(self, ring, strips):
        self.ring = list(ring or [])
        self.strips = list(strips or [])
        if self.ring:
            lats = [a for a, _ in self.ring]
            lons = [o for _, o in self.ring]
            self.bbox = (min(lats), max(lats), min(lons), max(lons))

    def __bool__(self):
        return bool(self.ring or self.strips)

    def contains(self, lat, lon):
        if self.ring:
            a0, a1, o0, o1 = self.bbox
            if a0 <= lat <= a1 and o0 <= lon <= o1 and _inside(self.ring, lat, lon):
                return True
        return on_runway(lat, lon, self.strips)


def classify(irs, agl, max_height=MAX_HEIGHT_M, max_side=MAX_SIDE_M, ground_agl=GROUND_AGL_M):
    """For one placement's solid sub-objects (MeshIRs) standing `agl` m
    above the ground: "drop" (small flat clutter), "drape" (a large flat
    sheet) or None (anything else)."""
    if agl is not None and agl > ground_agl:
        return None
    y_min = y_max = x_min = x_max = z_min = z_max = None
    for ir in irs:
        if ir is None:
            continue
        if ir.lights:
            return None
        if not len(ir.positions):
            continue
        p = ir.positions
        y_min = min(y_min, float(p[:, 1].min())) if y_min is not None else float(p[:, 1].min())
        y_max = max(y_max, float(p[:, 1].max())) if y_max is not None else float(p[:, 1].max())
        x_min = min(x_min, float(p[:, 0].min())) if x_min is not None else float(p[:, 0].min())
        x_max = max(x_max, float(p[:, 0].max())) if x_max is not None else float(p[:, 0].max())
        z_min = min(z_min, float(p[:, 2].min())) if z_min is not None else float(p[:, 2].min())
        z_max = max(z_max, float(p[:, 2].max())) if z_max is not None else float(p[:, 2].max())
    if y_min is None or y_max - y_min >= max_height:
        return None
    return "drop" if max(x_max - x_min, z_max - z_min) <= max_side else "drape"


def is_low_flat(irs, max_height=MAX_HEIGHT_M, max_side=MAX_SIDE_M):
    """True for small flat clutter (see classify), ignoring height above
    ground."""
    return classify(irs, None, max_height, max_side) == "drop"
