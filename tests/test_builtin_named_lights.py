"""
Always-flashing (not day/night-gated) macro_lights close to white color
get X-Plane's own built-in LIGHT_NAMED obs_strobe_night instead of
LIGHT_SPILL_CUSTOM + the custom msfs2xp/blink_always dataref -- real
obstruction/warning strobes flash day and night alike and are white
almost universally in practice, and X-Plane animates/day-night-gates a
named light entirely in its own engine with zero plugin dependency.
Colored always-flashing lights keep the custom dataref path (color
fidelity over a wrong-colored built-in substitute); night-gated flashing
lights (the classic rotating beacon) are untouched by this change either
way -- verified real library path against a real X-Plane 11 install's own
default control-tower objects.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert
from mesh_convert import mesh_ir


class TestBuiltinNamedLights(unittest.TestCase):
    def _convert_with_light(self, color, day_night_cycle, flash_frequency):
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((150, 150, 150, 255)))
        mat = b.add_material("BuildingMat", base_color_texture_index=tex)
        positions = [(-2, 0, -2), (2, 0, -2), (2, 4, -2), (-2, 4, -2)]
        normals = [(0.0, 0.0, -1.0)] * 4
        uvs = [(0.0, 0.0)] * 4
        indices = [0, 1, 2, 0, 2, 3]
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        node_idx = b.add_node(mesh_index=mesh, name="Tower")
        b.add_macro_light(node_idx, color=color, cone_angle=360.0, intensity=10.0,
                           day_night_cycle=day_night_cycle, flash_frequency=flash_frequency)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            glb_path = td / "tower.glb"
            glb_path.write_bytes(b.build())
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()
            result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            self.assertTrue(result)
            lights_obj = next(p for p in result if p.stem.endswith("_lights"))
            return lights_obj.read_text(encoding="utf-8"), obj_dir, lights_obj

    def test_white_always_flashing_light_uses_builtin_named_strobe(self):
        text, _, _ = self._convert_with_light((1.0, 1.0, 1.0), day_night_cycle=False, flash_frequency=2.0)
        self.assertIn("LIGHT_NAMED obs_strobe_night", text)
        self.assertNotIn("msfs2xp/blink_always", text)
        self.assertNotIn("LIGHT_SPILL_CUSTOM", text)

    def test_red_always_flashing_light_uses_builtin_red_obstruction(self):
        """A red always-flashing macro_light is an obstruction light --
        X-Plane's built-in obs_red_night is animated + night-gated in the
        engine with no plugin, so it's preferred over any custom dataref
        (which is dark for anyone without the FlyWithLua script)."""
        text, _, _ = self._convert_with_light((1.0, 0.1, 0.1), day_night_cycle=False, flash_frequency=2.0)
        self.assertIn("LIGHT_NAMED obs_red_night", text)
        self.assertNotIn("msfs2xp/", text)

    def test_night_only_white_flashing_light_uses_builtin_strobe(self):
        """White night+flash -> obs_strobe_night (itself a night-gated
        strobe). No dependency on the msfs2xp/* blink datarefs, which only
        resolve when the companion FlyWithLua script is loaded."""
        text, _, _ = self._convert_with_light((1.0, 1.0, 1.0), day_night_cycle=True, flash_frequency=2.0)
        self.assertIn("LIGHT_NAMED obs_strobe_night", text)
        self.assertNotIn("msfs2xp/", text)

    def test_named_light_survives_meshir_round_trip(self):
        """The confirmed real gap this required fixing: mesh_ir.write_obj8
        previously had no concept of a LIGHT_NAMED entry at all and would
        have written a nonsensical LIGHT_SPILL_CUSTOM line using the
        placeholder dataref string if a named-light entry were ever
        re-serialized (e.g. by terrain_fit.py's large-building Y-warp
        correction, which loads the .meshir.pkl sidecar and calls
        write_obj8 again)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            text, obj_dir, lights_obj = self._convert_with_light_in(
                td, (1.0, 1.0, 1.0), day_night_cycle=False, flash_frequency=2.0)
            sidecar = mesh_ir.sidecar_path_for(lights_obj)
            self.assertTrue(sidecar.exists())
            loaded = mesh_ir.load(sidecar)
            self.assertEqual(len(loaded.lights), 1)
            self.assertEqual(loaded.lights[0].named_light, "obs_strobe_night")

            # Round-trip through write_obj8 again (simulating what
            # terrain_fit does) and confirm it's still a real LIGHT_NAMED
            # line, not a malformed LIGHT_SPILL_CUSTOM using the
            # placeholder as a dataref.
            re_written = obj_dir / "rewritten_lights.obj"
            mesh_ir.write_obj8(loaded, re_written)
            re_text = re_written.read_text(encoding="utf-8")
            self.assertIn("LIGHT_NAMED obs_strobe_night", re_text)
            self.assertNotIn("LIGHT_SPILL_CUSTOM", re_text)

    def _convert_with_light_in(self, td, color, day_night_cycle, flash_frequency):
        """Same as _convert_with_light, but takes an already-open tempdir
        so the caller can keep using the returned paths after this
        returns (the shared helper's own `with tempfile.TemporaryDirectory()`
        deletes the directory the moment it returns, which is fine for
        callers that only use the returned STRING but not for callers
        that need the paths to still exist afterward)."""
        b = GltfBuilder()
        tex = b.add_texture(b.add_image_data_uri((150, 150, 150, 255)))
        mat = b.add_material("BuildingMat", base_color_texture_index=tex)
        positions = [(-2, 0, -2), (2, 0, -2), (2, 4, -2), (-2, 4, -2)]
        normals = [(0.0, 0.0, -1.0)] * 4
        uvs = [(0.0, 0.0)] * 4
        indices = [0, 1, 2, 0, 2, 3]
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        node_idx = b.add_node(mesh_index=mesh, name="Tower")
        b.add_macro_light(node_idx, color=color, cone_angle=360.0, intensity=10.0,
                           day_night_cycle=day_night_cycle, flash_frequency=flash_frequency)

        glb_path = td / "tower.glb"
        glb_path.write_bytes(b.build())
        obj_dir = td / "objects"
        tex_dir = td / "textures"
        obj_dir.mkdir()
        tex_dir.mkdir()
        result = mesh_convert.convert(glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
        self.assertTrue(result)
        lights_obj = next(p for p in result if p.stem.endswith("_lights"))
        return lights_obj.read_text(encoding="utf-8"), obj_dir, lights_obj


if __name__ == "__main__":
    unittest.main()
