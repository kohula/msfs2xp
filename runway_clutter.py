"""Flat objects lying on a runway: dropped.

MSFS runways carry many very low objects -- pit and drain covers, in-
pavement fixtures, cable plates -- that sit flush with MSFS's own runway
surface. In X-Plane the runway ground is X-Plane's terrain, not MSFS's, so
these end up hovering above the pavement and grass (or sunk into it), and
being only centimetres tall they add nothing seen from a cockpit. An
object is dropped when:

- its anchor lies on a runway strip: the runway itself, its overrun/blast
  pad, plus a margin beyond each edge for shoulders and edge plates;
- its solid (non-draped) geometry is less than MAX_HEIGHT_M tall and at
  most MAX_SIDE_M across -- aircraft, vehicles, signs and anything with
  real height are kept, as is a large flat piece that could be pavement;
- it carries no lights of its own (runway and approach light fixtures are
  kept).

Draped geometry (markings, decals) is never dropped.
"""

import math

from geo_transform import metres_per_degree

MAX_HEIGHT_M = 0.5
MAX_SIDE_M = 15.0
EDGE_MARGIN_M = 10.0
END_MARGIN_M = 30.0


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


def is_low_flat(irs, max_height=MAX_HEIGHT_M, max_side=MAX_SIDE_M):
    """True for one placement's solid sub-objects (MeshIRs) that are all
    low and small and carry no lights."""
    y_min = y_max = x_min = x_max = z_min = z_max = None
    for ir in irs:
        if ir is None:
            continue
        if ir.lights:
            return False
        if not len(ir.positions):
            continue
        p = ir.positions
        y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
        x0, x1 = float(p[:, 0].min()), float(p[:, 0].max())
        z0, z1 = float(p[:, 2].min()), float(p[:, 2].max())
        y_min = y0 if y_min is None else min(y_min, y0)
        y_max = y1 if y_max is None else max(y_max, y1)
        x_min = x0 if x_min is None else min(x_min, x0)
        x_max = x1 if x_max is None else max(x_max, x1)
        z_min = z0 if z_min is None else min(z_min, z0)
        z_max = z1 if z_max is None else max(z_max, z1)
    if y_min is None:
        return False
    return (y_max - y_min) < max_height and max(x_max - x_min, z_max - z_min) <= max_side
