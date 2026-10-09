"""Parts MSFS moves by itself: radar dishes, fans, turning signs.

A model's XML marks such an animation as
    <Animation name="..." type="Standard" typeParam="AutoPlay" .../>
and MSFS plays it in a loop for as long as the object is shown. Converted,
it becomes an X-Plane animation on the always-running sim clock
(sim/time/total_running_time_sec) that repeats with the clip's length
(ANIM_keyframe_loop).

X-Plane turns a part about one fixed axis per ANIM_rotate block, so a
rotation channel is converted only when every key turns the part about the
same axis (a dish on its mast, a fan on its hub); anything else stays
still in its rest pose, as before. Angles are unwrapped so a full turn is
one continuous 0..360 sweep, and keys that a straight line through their
neighbours already gives are dropped.
"""

import math
import re

import numpy as np

AUTOPLAY_DATAREF = "sim/time/total_running_time_sec"
_AXIS_TOLERANCE_DEG = 2.0   # every key's axis within this of the main one
_STILL_DEG = 0.05           # a key turned less than this has no axis to check
_THIN_DEG = 0.25            # dropped keys are at most this far off the line
_THIN_M = 0.005

_ANIMATION_TAG_RE = re.compile(r"<Animation\b([^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def autoplay_animation_names(xml_text):
    """Lower-cased names of the animations a model XML marks AutoPlay."""
    names = set()
    for m in _ANIMATION_TAG_RE.finditer(xml_text or ""):
        attrs = {k.lower(): v for k, v in _ATTR_RE.findall(m.group(1))}
        if attrs.get("typeparam", "").strip().lower() == "autoplay" and attrs.get("name"):
            names.add(attrs["name"].strip().lower())
    return names


def _quat_matrix(q):
    x, y, z, w = (float(c) for c in q[:4])
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _axis_angle(m):
    """(unit axis, angle in degrees 0..180) of a rotation matrix."""
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(m) - 1.0) / 2.0))))
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:
        if angle < 90.0:
            return None, 0.0
        # a half turn: the axis is the column of (m + I) with the most length
        sym = m + np.eye(3)
        col = sym[:, int(np.argmax(np.linalg.norm(sym, axis=0)))]
        return col / np.linalg.norm(col), angle
    return axis / n, angle


def rotation_turn(rest_quat, quats, frame):
    """One fixed axis for a rotation channel, or None.

    rest_quat: the pose the geometry is baked in; quats: each key's
    rotation (x, y, z, w) in the node's parent frame; frame: 3x3 turning
    that parent frame into the model's output frame. Returns (axis,
    [angle_deg per key]) with the angles unwrapped into one continuous
    sweep, or None when the keys don't share one axis."""
    rest_t = _quat_matrix(rest_quat).T
    deltas = []
    for q in quats:
        d = frame @ (_quat_matrix(q) @ rest_t) @ frame.T
        deltas.append(_axis_angle(d))
    main = max(deltas, key=lambda d: d[1])
    if main[0] is None or main[1] < _STILL_DEG:
        return None
    axis = main[0]
    cos_tol = math.cos(math.radians(_AXIS_TOLERANCE_DEG))
    angles = []
    for a, ang in deltas:
        if ang < _STILL_DEG or a is None:
            signed = 0.0
        else:
            dot = float(np.dot(a, axis))
            if abs(dot) < cos_tol and ang < 179.0:
                return None
            signed = ang if dot >= 0 else -ang
        if angles:
            prev = angles[-1]
            signed += 360.0 * round((prev - signed) / 360.0)
        angles.append(signed)
    return (float(axis[0]), float(axis[1]), float(axis[2])), angles


def _thin(times, values, tol):
    """Keeps the keys a straight line between the kept neighbours misses
    by more than tol. values: (N, k) array."""
    keep = [0]
    for i in range(1, len(times) - 1):
        t0, v0 = times[keep[-1]], values[keep[-1]]
        t1, v1 = times[i + 1], values[i + 1]
        span = t1 - t0
        f = (times[i] - t0) / span if span > 1e-9 else 0.0
        if np.max(np.abs(v0 + (v1 - v0) * f - values[i])) > tol:
            keep.append(i)
    keep.append(len(times) - 1)
    return keep


def thin_turn(times, angles, tol_deg=_THIN_DEG):
    idx = _thin(list(times), np.asarray(angles, dtype=float).reshape(-1, 1), tol_deg)
    return [(float(times[i]), float(angles[i])) for i in idx]


def thin_slide(times, offsets, tol_m=_THIN_M):
    arr = np.asarray(offsets, dtype=float)
    idx = _thin(list(times), arr, tol_m)
    return [(float(times[i]), *map(float, arr[i])) for i in idx]
