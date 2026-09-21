"""
apply_color_factor / apply_alpha_factor / apply_emissive_factor all
decode+modify+re-save as PNG internally, regardless of their input
file's own extension -- but used to build their OUTPUT filename from
the INPUT path's own suffix, so a caller that passed a .dds path (a
texture slot whose own allow_dds_passthrough happened to be True for an
unrelated reason -- confirmed real case: apply_emissive_factor's
cross-slot reuse of a material's already-extracted BASE COLOR texture
name, whose own allow_dds_passthrough decision has nothing to do with
whether this unrelated emissive synthesis needs to bake it) got back a
file genuinely containing PNG bytes but named ".dds". CONFIRMED REAL
BUG: X-Plane's own Log.txt reported "we are missing the texture" for
several files exactly matching this pattern.
"""
import importlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

convert_module = importlib.import_module("mesh_convert.convert")


class TestFactorBakeAlwaysPng(unittest.TestCase):
    def _make_dds_named_png(self, td):
        """A real, valid PNG's bytes saved under a MISLEADING .dds
        filename -- stands in for a texture slot that legitimately came
        back as .dds from its own extraction (e.g. base color passthrough)."""
        src = Path(td) / "source.dds"
        Image.new("RGBA", (4, 4), (200, 100, 50, 255)).save(src, "PNG")
        return src

    def test_apply_color_factor_output_is_always_png(self):
        with tempfile.TemporaryDirectory() as td:
            src = self._make_dds_named_png(td)
            out_name = convert_module.apply_color_factor(src, (255, 128, 0))
            self.assertTrue(out_name.endswith(".png"), f"expected a .png output name, got {out_name!r}")
            self.assertTrue((Path(td) / out_name).read_bytes().startswith(b"\x89PNG"))

    def test_apply_alpha_factor_output_is_always_png(self):
        with tempfile.TemporaryDirectory() as td:
            src = self._make_dds_named_png(td)
            out_name = convert_module.apply_alpha_factor(src, 128)
            self.assertTrue(out_name.endswith(".png"), f"expected a .png output name, got {out_name!r}")
            self.assertTrue((Path(td) / out_name).read_bytes().startswith(b"\x89PNG"))

    def test_apply_emissive_factor_output_is_always_png(self):
        with tempfile.TemporaryDirectory() as td:
            src = self._make_dds_named_png(td)
            out_name = convert_module.apply_emissive_factor(src, [30.0, 30.0, 30.0])
            self.assertTrue(out_name.endswith(".png"), f"expected a .png output name, got {out_name!r}")
            self.assertTrue((Path(td) / out_name).read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
