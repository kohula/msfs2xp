"""
Single source of truth for local-model-offset <-> real-world lat/lon math.
Every caller imports these functions instead of re-deriving the rotation.

Convention (matches OBJ8/MSFS and every placement record this project
reads): heading is degrees clockwise from north; local +X = east and
local -Z = forward/north at heading 0. Real-world coordinates use the
flat-earth approximation (111139.0 m/degree latitude, longitude scaled by
cos(latitude)) that's accurate enough at airport-footprint scale (a few
km at most) -- the same approximation the whole rest of this project's
placement/DSF-tile math already relies on.
"""

import math

EARTH_M_PER_DEG = 111139.0


def decode_lonlat_dword(dword: int, is_lat: bool) -> float:
    """Classic FS9/FSX BGL scaled-dword lat/lon encoding (32-bit LE). Lives
    here rather than in bgl_extractor.py so airport_layout.py can import it
    without a circular import (bgl_extractor.py imports airport_layout)."""
    if is_lat:
        return 90.0 - dword * (180.0 / 536870912.0)
    return dword * (360.0 / 805306368.0) - 180.0

# WGS84 -- for metres_per_degree(), which the draped-layer merge needs
# instead of the flat EARTH_M_PER_DEG constant above, which is only
# accurate to ~1% E-W at mid latitudes. X-Plane's own world<->local
# conversion is full WGS84, so matching it avoids visible drift on
# merged draped layers.
_WGS84_A = 6378137.0
_WGS84_E2 = 6.694379990141e-3


def metres_per_degree(lat_deg: float) -> tuple[float, float]:
    """(metres per degree of latitude, metres per degree of longitude) on
    the WGS84 ellipsoid at lat_deg -- the local-tangent-plane scale to use
    for accurate metre<->degree conversion near a given latitude."""
    phi = math.radians(lat_deg)
    s = math.sin(phi)
    w = math.sqrt(1.0 - _WGS84_E2 * s * s)
    m_per_deg = math.pi / 180.0 * _WGS84_A
    m_lat = m_per_deg * (1.0 - _WGS84_E2) / (w * w * w)
    m_lon = m_per_deg * math.cos(phi) / w
    return m_lat, m_lon


def rotate_xz(x: float, z: float, heading_deg: float) -> tuple[float, float]:
    """Rotate a local (x, z) offset by heading_deg (clockwise from north).
    Pure rotation, no lat/lon involved -- the shared first step of both
    functions below."""
    hdg_rad = math.radians(heading_deg)
    rot_x = x * math.cos(hdg_rad) - z * math.sin(hdg_rad)
    rot_z = x * math.sin(hdg_rad) + z * math.cos(hdg_rad)
    return rot_x, rot_z


def local_offset_to_latlon(base_lat: float, base_lon: float, heading_deg: float,
                            local_x: float, local_z: float) -> tuple[float, float]:
    """Real-world (lat, lon) of a point local_x/local_z meters from
    (base_lat, base_lon), in heading-rotated local space."""
    rot_x, rot_z = rotate_xz(local_x, local_z, heading_deg)
    lat = base_lat - (rot_z / EARTH_M_PER_DEG)
    lon = base_lon + (rot_x / (EARTH_M_PER_DEG * math.cos(math.radians(base_lat))))
    return lat, lon


def latlon_offset_to_local(base_lat: float, base_lon: float, heading_deg: float,
                            point_lat: float, point_lon: float) -> tuple[float, float]:
    """Inverse of local_offset_to_latlon: the local (x, z) offset of
    (point_lat, point_lon) from (base_lat, base_lon), in a frame rotated
    by heading_deg."""
    dlat = point_lat - base_lat
    dlon = point_lon - base_lon
    rot_z = -dlat * EARTH_M_PER_DEG
    rot_x = dlon * EARTH_M_PER_DEG * math.cos(math.radians(base_lat))
    if heading_deg == 0.0:
        return rot_x, rot_z
    return rotate_xz(rot_x, rot_z, -heading_deg)
