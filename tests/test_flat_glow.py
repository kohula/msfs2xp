"""
Near-constant dim emissive maps (a flat grey wash at night) are dropped;
structured maps and near-constant bright ones (sign faces, lit rooms
behind panes) are kept.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mesh_convert.convert import is_flat_glow


def _png(td, name, arr):
    p = Path(td) / name
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(p)
    return p


class TestFlatGlow(unittest.TestCase):
    def test_glass(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertTrue(is_flat_glow(_png(td, "g.png", np.full((32, 32, 3), 70)), glass=True))
            self.assertFalse(is_flat_glow(_png(td, "room.png", np.full((32, 32, 3), 220)), glass=True))

    def test_even_dim_glow_elsewhere(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertTrue(is_flat_glow(_png(td, "bus.png", np.full((32, 32, 3), 60))))
            self.assertFalse(is_flat_glow(_png(td, "sign.png", np.full((32, 32, 3), 220))))
            self.assertFalse(is_flat_glow(_png(td, "black.png", np.zeros((32, 32, 3)))))

    def test_lit_windows_stay(self):
        arr = np.zeros((64, 64, 3))
        arr[8:16, ::8] = 255
        arr[40:48, 4::8] = 230
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(is_flat_glow(_png(td, "w.png", arr), glass=True))


if __name__ == "__main__":
    unittest.main()
