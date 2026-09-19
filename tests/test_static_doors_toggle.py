"""
convert()'s disable_proximity_animation parameter (wired to main.py's
"Static doors" GUI toggle) -- proximity-triggered AND business-hours-
triggered animations (doors/barriers that open as the aircraft approaches,
or on a local-time schedule) both render as static rigid geometry at their
own rest pose instead when this is on, without touching blink triggers
(which drive ATTR_light_level/emissive content, never mesh rotation/
translation, and stay animated either way -- "static except the lights").
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import mesh_convert


class TestStaticDoorsToggle(unittest.TestCase):
    def _build_proximity_door(self, td: Path):
        b = GltfBuilder()
        tex = b.add_image_data_uri((180, 180, 180, 255))
        texi = b.add_texture(tex)
        mat = b.add_material("DoorMat", base_color_texture_index=texi)

        positions = [(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 2.0, -0.5), (-0.5, 2.0, -0.5)]
        normals = [(0.0, 0.0, -1.0)] * 4
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        indices = [0, 1, 2, 0, 2, 3]
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        node_idx = b.add_node(mesh_index=mesh, name="DoorPanel")
        b.add_animation(
            target_node=node_idx, path="rotation",
            times=[0.0, 1.0], values=[(0.0, 0.0, 0.0, 1.0), (0.0, 0.70710678, 0.0, 0.70710678)])

        glb_path = td / "door_LOD00.glb"
        glb_path.write_bytes(b.build())
        xml_path = td / "door.xml"
        xml_path.write_text(
            "<ModelInfo guid='{x}'/><ModelBehaviors>"
            "(Z:VisibleRadiusBox, Number) 1 == if{ 3.0 } els{ 0 }"
            "</ModelBehaviors>",
            encoding="utf-8",
        )
        return glb_path

    def _build_business_hours_door(self, td: Path):
        b = GltfBuilder()
        tex = b.add_image_data_uri((180, 180, 180, 255))
        texi = b.add_texture(tex)
        mat = b.add_material("DoorMat", base_color_texture_index=texi)

        positions = [(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 2.0, -0.5), (-0.5, 2.0, -0.5)]
        normals = [(0.0, 0.0, -1.0)] * 4
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        indices = [0, 1, 2, 0, 2, 3]
        mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
        node_idx = b.add_node(mesh_index=mesh, name="DoorPanel")
        b.add_animation(
            target_node=node_idx, path="rotation",
            times=[0.0, 1.0], values=[(0.0, 0.0, 0.0, 1.0), (0.0, 0.70710678, 0.0, 0.70710678)])

        glb_path = td / "hours_door_LOD00.glb"
        glb_path.write_bytes(b.build())
        xml_path = td / "hours_door.xml"
        xml_path.write_text(
            "<ModelInfo guid='{x}'/><ModelBehaviors>"
            "(L:LOCAL TIME, Seconds) 25200 &gt; (E:LOCAL TIME, Seconds) 79200 &lt;"
            "</ModelBehaviors>",
            encoding="utf-8",
        )
        return glb_path

    def _convert(self, glb_path, disable_proximity_animation):
        obj_dir = glb_path.parent / "objects"
        tex_dir = glb_path.parent / "textures"
        obj_dir.mkdir(exist_ok=True)
        tex_dir.mkdir(exist_ok=True)
        result = mesh_convert.convert(
            glb_path, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0",
            disable_proximity_animation=disable_proximity_animation)
        self.assertTrue(result)
        return "\n".join(p.read_text(encoding="utf-8") for p in result)

    def test_proximity_animation_exports_anim_block_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            glb_path = self._build_proximity_door(Path(td))
            text = self._convert(glb_path, disable_proximity_animation=False)
            self.assertIn("ANIM_begin", text)

    def test_static_doors_toggle_suppresses_proximity_animation(self):
        with tempfile.TemporaryDirectory() as td:
            glb_path = self._build_proximity_door(Path(td))
            text = self._convert(glb_path, disable_proximity_animation=True)
            self.assertNotIn("ANIM_begin", text)
            self.assertNotIn("msfs2xp/proximity/", text)

    def test_business_hours_animation_exports_anim_block_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            glb_path = self._build_business_hours_door(Path(td))
            text = self._convert(glb_path, disable_proximity_animation=False)
            self.assertIn("ANIM_begin", text)
            self.assertIn("sim/time/local_time_sec", text)

    def test_static_doors_toggle_suppresses_business_hours_animation(self):
        """Not just proximity -- a business-hours-scheduled door/barrier
        (e.g. a gate that's only down outside operating hours) reads a
        real, always-available X-Plane dataref and would keep animating
        correctly with no companion plugin at all, but the "static doors"
        toggle forces it static too: explicitly requested, "static except
        the lights" means every non-light animation, not just the ones
        that strictly require a companion script."""
        with tempfile.TemporaryDirectory() as td:
            glb_path = self._build_business_hours_door(Path(td))
            text = self._convert(glb_path, disable_proximity_animation=True)
            self.assertNotIn("ANIM_begin", text)
            self.assertNotIn("sim/time/local_time_sec", text)


if __name__ == "__main__":
    unittest.main()
