"""obj_scale: placements MSFS scales get a copy with the scale baked in."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import obj_scale
from mesh_convert import mesh_ir

_OBJ = """I
800
OBJ

TEXTURE ../textures/a.png
POINT_COUNTS 3 0 0 3
VT 1.00000 2.00000 -3.00000 0.00000 1.00000 0.00000 0.25000 0.75000
VT 0 0 0 0 1 0 0 0
VT 2 0 0 0 1 0 1 0
IDX 0 1 2
LIGHT_NAMED airplane_beacon 1.0 10.0 0.0
LIGHT_PARAM full_custom_halo_night 0.5 4.0 -1.0 1.0 0.9 0.8 1.0 25.000 0.0 -1.0 0.0 0.5
ANIM_begin
ANIM_trans_begin sim/x
ANIM_trans_key 0.00 0.0 0.0 0.0
ANIM_trans_key 1.00 2.0 0.0 -1.0
ANIM_trans_end
ANIM_rotate_begin 0.0 1.0 0.0 sim/y
ANIM_rotate_key 0.00 90.0
ANIM_rotate_end
ANIM_end
TRIS 0 3
"""


class TestObjScale(unittest.TestCase):
    def test_lengths_scale_and_nothing_else(self):
        out = obj_scale.scale_obj8_text(_OBJ, 2.0).splitlines()
        self.assertIn("VT 2.00000 4.00000 -6.00000 0.00000 1.00000 0.00000 0.25000 0.75000", out)
        self.assertIn("LIGHT_NAMED airplane_beacon 2.00000 20.00000 0.00000", out)
        halo = next(l for l in out if l.startswith("LIGHT_PARAM")).split()
        self.assertEqual(halo[2:5], ["1.00000", "8.00000", "-2.00000"])
        self.assertEqual(halo[5:9], ["1.0", "0.9", "0.8", "1.0"], "colour untouched")
        self.assertEqual(halo[9], "50.00000", "spill size scales")
        self.assertIn("ANIM_trans_key 1.00 4.00000 0.00000 -2.00000", out)
        self.assertIn("ANIM_rotate_key 0.00 90.0", out, "angles don't scale")
        self.assertIn("ANIM_rotate_begin 0.0 1.0 0.0 sim/y", out)
        self.assertIn("TRIS 0 3", out)

    def test_variant_files_and_ir_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "bldg_0.obj").write_text(_OBJ, encoding="utf-8")
            ir = mesh_ir.MeshIR(name="bldg_0", positions=np.array([[1.0, 2.0, 3.0]]), footprint_area_m2=10.0)
            mesh_ir.save(ir, mesh_ir.sidecar_path_for(d / "bldg_0.obj"))
            stem = obj_scale.make_scaled_variant(d, "bldg_0", 1.5)
            self.assertEqual(stem, "bldg_0_s1500")
            self.assertTrue((d / "bldg_0_s1500.obj").is_file())
            scaled = mesh_ir.load(mesh_ir.sidecar_path_for(d / "bldg_0_s1500.obj"))
            np.testing.assert_allclose(scaled.positions, [[1.5, 3.0, 4.5]])
            self.assertAlmostEqual(scaled.footprint_area_m2, 22.5)
            self.assertEqual(obj_scale.make_scaled_variant(d, "missing", 2.0), "missing")


if __name__ == "__main__":
    unittest.main()
