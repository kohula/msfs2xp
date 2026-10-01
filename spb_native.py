"""Reading MSFS 2024 SimProp container files (.spb) without the SDK Propdefs.

A compiled SimPropBinary file describes its own properties: after a small
header comes a table of 20-byte entries -- a property GUID and the size of
its value (0xFFFFFFFF for variable length) -- and the body is a stream of
tagged values whose tag is a 1-based index into that table (a
variable-length value carries a u32 length and may itself be a nested
stream; tag 0 closes a set). A property's meaning comes from its GUID,
never from the index, since every file numbers its own table. So placing a
container's child models needs only five well-known property GUIDs, not
the whole Propdefs XML set: SimPropAttach (one child), its model GUID,
offset, orientation and scale.

Which container a file is comes from the package's simPropContainers.json
(container GUID -> file path), since the container's own GUID property is
not among those five.

This is the fallback when no Propdefs folder is configured; with one,
bgl_extractor keeps using the full spb2xml decompiler (which also resolves
title-referenced children).
"""

import json
import math
import struct
import uuid
from pathlib import Path


def _guid_bytes(text):
    """A GUID string -> its 16 on-disk (Windows mixed-endian) bytes."""
    return uuid.UUID(text.strip("{}")).bytes_le


SIM_PROP_ATTACH = _guid_bytes("ad124d80-114c-4682-9bd0-783fb99c5023")
OFFSET_XYZ = _guid_bytes("b975cd65-7cef-4cc2-9ab1-24b7cfefa02c")
ORIENTATION = _guid_bytes("fbedc683-8576-4138-b70b-383231f6132a")
MDL_GUID = _guid_bytes("8588e41e-89ca-47f9-8210-f8561d93d17c")
SCALE = _guid_bytes("0119970b-fc6f-4979-b416-8e664d97fc54")

_MAGIC = b"\xAC\xEB"
_COUNT_AT = 0x1A
_TABLE_AT = 0x32
_VARIABLE = 0xFFFFFFFF
_MAX_DEPTH = 32


def _parse_stream(data, table, depth):
    """[(guid, bytes) | (guid, [children])] for one tagged stream, or None
    if `data` isn't one (then the caller keeps it as an opaque value)."""
    out = []
    at = 0
    while at < len(data):
        if at + 4 > len(data):
            return None
        tag = struct.unpack_from("<I", data, at)[0]
        at += 4
        if tag == 0:
            continue
        if tag - 1 >= len(table):
            return None
        guid, size = table[tag - 1]
        if size == _VARIABLE:
            if at + 4 > len(data):
                return None
            length = struct.unpack_from("<I", data, at)[0]
            at += 4
            if at + length > len(data):
                return None
            value = data[at:at + length]
            at += length
            nested = _parse_stream(value, table, depth + 1) if depth < _MAX_DEPTH and value else None
            out.append((guid, nested if nested else value))
        else:
            if at + size > len(data):
                return None
            out.append((guid, data[at:at + size]))
            at += size
    return out


def _angle(raw):
    return raw * 360.0 / 4294967296.0


def _signed(a):
    return a - 360.0 if a > 180.0 else a


def _collect(nodes, child):
    for guid, value in nodes:
        if isinstance(value, list):
            _collect(value, child)
        elif guid == MDL_GUID and len(value) >= 16 and any(value[:16]):
            child["guid"] = value[:16].hex()
        elif guid == OFFSET_XYZ and len(value) >= 12:
            xyz = struct.unpack_from("<fff", value, 0)
            if all(math.isfinite(v) for v in xyz):
                child["offset"] = xyz
        elif guid == ORIENTATION and len(value) >= 12:
            p, b, h = struct.unpack_from("<III", value, 0)
            child["pitch"], child["bank"], child["heading"] = _signed(_angle(p)), _signed(_angle(b)), _angle(h)
        elif guid == SCALE and len(value) >= 4:
            s = struct.unpack_from("<f", value, 0)[0]
            if math.isfinite(s) and s > 0:
                child["scale"] = s


def _find_children(nodes, out):
    for guid, value in nodes:
        if not isinstance(value, list):
            continue
        if guid == SIM_PROP_ATTACH:
            child = {"offset": (0.0, 0.0, 0.0), "pitch": 0.0, "bank": 0.0, "heading": 0.0, "scale": 1.0}
            _collect(value, child)
            if "guid" in child:
                out.append(child)
        else:
            _find_children(value, out)


def parse_container(data):
    """Child models of a SimProp container file: [{"guid": hex of the model
    GUID's on-disk bytes, "offset": (x right, y up, z forward) metres,
    "pitch"/"bank"/"heading": degrees, "scale"}, ...]. Lights and other
    non-model children are skipped. Raises ValueError if it isn't one."""
    if len(data) < _TABLE_AT or data[:2] != _MAGIC:
        raise ValueError("not a SimPropBinary file")
    entries = max(0, struct.unpack_from("<I", data, _COUNT_AT)[0] - 1)
    body = _TABLE_AT + 20 * entries
    if entries > 4096 or body > len(data):
        raise ValueError(f"property table of {entries} entries does not fit")
    table = [(data[_TABLE_AT + 20 * i:_TABLE_AT + 20 * i + 16],
              struct.unpack_from("<I", data, _TABLE_AT + 20 * i + 16)[0]) for i in range(entries)]
    nodes = _parse_stream(data[body:], table, 0)
    if nodes is None:
        raise ValueError("unreadable property stream")
    out = []
    _find_children(nodes, out)
    return out


def container_index(root):
    """{container GUID (hex of on-disk bytes): .spb path} from every
    simPropContainers.json under `root` (paths are relative to the package
    root, or to the json's own folder)."""
    root = Path(root)
    index = {}
    for js in root.rglob("*"):
        if js.name.lower() != "simpropcontainers.json" or not js.is_file():
            continue
        try:
            content = json.loads(js.read_text(encoding="utf-8")).get("content", [])
        except (OSError, ValueError, AttributeError):
            continue
        for item in content if isinstance(content, list) else []:
            if not isinstance(item, dict) or not item.get("guid") or not item.get("path"):
                continue
            try:
                key = _guid_bytes(item["guid"]).hex()
            except ValueError:
                continue
            rel = str(item["path"]).replace("\\", "/")
            path = root / rel
            if not path.is_file():
                path = js.parent / rel
            index[key] = path
    return index
