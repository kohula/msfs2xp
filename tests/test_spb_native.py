"""spb_native: SimProp containers read from their own property table, so
SPB placements work without the MSFS SDK Propdefs."""
import json
import os
import struct
import tempfile
import unittest
import uuid
from pathlib import Path

import bgl_extractor
import spb_native

MODEL = uuid.UUID("11111111-2222-3333-4444-555555555555")
CONTAINER = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


def _spb(children):
    table = [(spb_native.SIM_PROP_ATTACH, 0xFFFFFFFF), (spb_native.MDL_GUID, 16), (spb_native.OFFSET_XYZ, 12),
             (spb_native.ORIENTATION, 12), (spb_native.SCALE, 4)]
    head = bytearray(0x32)
    head[:2] = b"\xAC\xEB"
    struct.pack_into("<I", head, 0x1A, len(table) + 1)
    body = b"".join(g + struct.pack("<I", s) for g, s in table)
    for guid, (x, y, z), heading, scale in children:
        inner = struct.pack("<I", 2) + guid.bytes_le
        inner += struct.pack("<I", 3) + struct.pack("<fff", x, y, z)
        inner += struct.pack("<I", 4) + struct.pack("<III", 0, 0, int(heading / 360.0 * 2 ** 32))
        inner += struct.pack("<I", 5) + struct.pack("<f", scale)
        inner += struct.pack("<I", 0)
        body += struct.pack("<II", 1, len(inner)) + inner
    return bytes(head) + body


class TestSpbNative(unittest.TestCase):
    def test_parse_children(self):
        kids = spb_native.parse_container(_spb([(MODEL, (1.0, 2.0, 3.0), 90.0, 1.5)]))
        self.assertEqual(len(kids), 1)
        k = kids[0]
        self.assertEqual(k["guid"], MODEL.bytes_le.hex())
        self.assertEqual(k["offset"], (1.0, 2.0, 3.0))
        self.assertAlmostEqual(k["heading"], 90.0, places=4)
        self.assertAlmostEqual(k["scale"], 1.5)

    def test_not_an_spb(self):
        with self.assertRaises(ValueError):
            spb_native.parse_container(b"nope" * 20)

    def test_placements_without_propdefs(self):
        old_env = os.environ.pop("MSFS2XP_PROPDEFS_DIR", None)
        try:
            with tempfile.TemporaryDirectory() as td:
                pkg = Path(td) / "pkg"
                (pkg / "SimPropContainers").mkdir(parents=True)
                (pkg / "manifest.json").write_text("{}")
                spb = pkg / "SimPropContainers" / "lamp.spb"
                spb.write_bytes(_spb([(MODEL, (0.0, 0.0, 0.0), 45.0, 1.0)]))
                (pkg / "SimPropContainers" / "simPropContainers.json").write_text(json.dumps(
                    {"content": [{"guid": "{" + str(CONTAINER).upper() + "}",
                                  "path": "SimPropContainers/lamp.spb"}]}))
                anchor = {"guid": CONTAINER.bytes_le.hex(), "lat": 47.43, "lon": 19.25, "alt": 0.0,
                          "hdg": 10.0, "is_agl": True, "scale": 2.0}
                logs = []
                found = bgl_extractor.extract_spb_placements(
                    spb, 47.43, 19.25, 0.0, [anchor], lambda m, l="info": logs.append(m),
                    propdefs_dir="/nonexistent", allow_fallback=False)
        finally:
            if old_env is not None:
                os.environ["MSFS2XP_PROPDEFS_DIR"] = old_env
        self.assertEqual(len(found), 1, logs)
        self.assertEqual(found[0]["guid"], MODEL.bytes_le.hex())
        self.assertEqual(found[0]["container_guid"], CONTAINER.bytes_le.hex())
        self.assertAlmostEqual(found[0]["hdg"], 55.0, places=3)
        self.assertAlmostEqual(found[0]["scale"], 2.0)


if __name__ == "__main__":
    unittest.main()
