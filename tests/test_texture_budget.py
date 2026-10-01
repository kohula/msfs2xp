"""texture_budget: textures capped by the size of the objects drawing them,
unreferenced textures removed, DDS shrunk by dropping mip levels."""
import struct
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import texture_budget


def _obj(size_m, texture, draped=False, extra=""):
    h = size_m / 2.0
    return (f"I\n800\nOBJ\n\nTEXTURE ../textures/{texture}\n{extra}\nPOINT_COUNTS 2 0 0 0\n"
            f"VT {-h} 0 0 0 1 0 0 0\nVT {h} 0 0 0 1 0 1 1\n\n" + ("ATTR_draped\n" if draped else "") + "TRIS 0 0\n")


def _dds(side, levels):
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000 | 0x20000
    head = struct.pack("<7I11I", 124, flags, side, side, (side // 4) ** 2 * 8, 0, levels, *([0] * 11))
    pf = struct.pack("<2I4s5I", 32, 0x4, b"DXT1", 0, 0, 0, 0, 0)
    caps = struct.pack("<5I", 0x1000 | 0x8 | 0x400000, 0, 0, 0, 0)
    data = b""
    s = side
    for i in range(levels):
        data += bytes([i]) * (max(1, s // 4) ** 2 * 8)
        s = max(1, s // 2)
    return b"DDS " + head + pf + caps + data


class TestTextureBudget(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        (self.td / "objects").mkdir()
        (self.td / "textures").mkdir()
        (self.td / "polygons").mkdir()
        for name in ("person.png", "terminal.png", "ground.png", "shared.png", "orphan.png", "decal.png"):
            Image.new("RGBA", (4096, 64), (1, 2, 3, 255)).save(self.td / "textures" / name)
        (self.td / "textures" / "van.dds").write_bytes(_dds(2048, 12))
        objs = {
            "person.obj": _obj(1.8, "person.png"),
            "terminal.obj": _obj(400, "terminal.png"),
            "ground.obj": _obj(2, "ground.png", draped=True),
            "small.obj": _obj(1, "shared.png"),
            "big.obj": _obj(200, "shared.png"),
            "van.obj": _obj(8, "van.dds"),
        }
        for n, t in objs.items():
            (self.td / "objects" / n).write_text(t)
        (self.td / "polygons" / "d.pol").write_text("A\n850\nDRAPED_POLYGON\n\nTEXTURE ../textures/decal.png\n")

    def _side(self, name):
        with Image.open(self.td / "textures" / name) as img:
            return max(img.size)

    def test_caps_and_cleanup(self):
        shrunk, deleted = texture_budget.apply_texture_budget(self.td, max_side=2048)
        self.assertEqual(self._side("person.png"), 256)
        self.assertEqual(self._side("terminal.png"), 2048)
        self.assertEqual(self._side("ground.png"), 2048, "draped ground keeps the full cap")
        self.assertEqual(self._side("decal.png"), 2048, "polygon textures keep the full cap")
        self.assertEqual(self._side("shared.png"), 2048, "the largest user decides")
        self.assertFalse((self.td / "textures" / "orphan.png").exists())
        self.assertEqual(deleted, 1)
        data = (self.td / "textures" / "van.dds").read_bytes()
        height, width = struct.unpack_from("<II", data, 12)
        self.assertEqual((width, height), (512, 512), "8 m vehicle: a quarter of 2048")
        self.assertEqual(struct.unpack_from("<I", data, 28)[0], 10)
        self.assertEqual(data[128], 2, "starts at what was mip level 2")

    def test_zero_keeps_sizes(self):
        texture_budget.apply_texture_budget(self.td, max_side=0)
        self.assertEqual(self._side("person.png"), 4096)


if __name__ == "__main__":
    unittest.main()
