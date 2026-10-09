"""
Working jetways with the airport's own look (jetway_rig.py): an MSFS
jetway model's rig is split into X-Plane's moving jetway parts, written as
one object on X-Plane's jetway datarefs, and placed by an apt.dat 1500 row
with a 1501 row naming the object.
"""
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from gltf_builder import GltfBuilder  # noqa: E402

import jetway_rig

RIG_XML = """<ModelInfo>
  <IKChain Name="IK_MainHandle"><Start>Rotation_Base</Start><End>Pivot</End></IKChain>
  <IKChain Name="IK_WheelsGroundLock"><Start>Bone07</Start><End>Wheel_Ctrl_Orient</End></IKChain>
  <IKConstraint><Node>Rotation_Base</Node><Heading/></IKConstraint>
  <IKConstraint><Node>Bone01</Node><Bank min="-6" max="6"/></IKConstraint>
  <IKConstraint><Node>Bone02</Node><X min="0.5" max="9.0"/></IKConstraint>
  <IKConstraint><Node>Bone03</Node><X min="0.5" max="10.0"/></IKConstraint>
  <IKConstraint><Node>RotationEndBone</Node><Heading/></IKConstraint>
  <IKConstraint><Node>Bone07</Node></IKConstraint>
  <IKConstraint><Node>Bone08</Node><X/></IKConstraint>
</ModelInfo>"""


def _box(b, mat, cx, cy, cz, hx=0.5, hy=0.5, hz=0.5):
    pos = [(cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz - hz),
           (cx - hx, cy + hy, cz - hz)]
    return b.add_mesh(pos, [0, 1, 2, 0, 2, 3], normals=[(0, 0, -1)] * 4,
                      uvs=[(0, 0), (1, 0), (1, 1), (0, 1)], material_index=mat)


def _jetway_glb():
    """Bones along +X like MSFS's template: rotunda 5 m up, a tipping bone,
    two telescoping sections, the cab pivot 16.5 m out and a wheel leg
    under the outer section. Each part carries one small quad."""
    b = GltfBuilder()
    tex = b.add_texture(b.add_image_data_uri((200, 200, 200, 255)))
    paint = b.add_material("Paint", base_color_texture_index=tex)
    glass = b.add_material("Cab_Glass", alpha_mode="BLEND", base_color_factor=[0.6, 0.7, 0.8, 0.4])
    m = lambda mat, *c: _box(b, mat, *c)  # noqa: E731

    wheel = b.add_node(mesh_index=m(paint, 0, -0.5, 0), name="WheelMesh", top_level=False)
    orient = b.add_node(name="Wheel_Ctrl_Orient", translation=(0, -1.0, 0), children=[wheel], top_level=False)
    leg_mesh = b.add_node(mesh_index=m(paint, 0, -0.5, 0), name="LegMesh", top_level=False)
    bone08 = b.add_node(name="Bone08", translation=(0, -2.0, 0), children=[orient, leg_mesh], top_level=False)
    bone07 = b.add_node(name="Bone07", translation=(4.0, 0, 0), children=[bone08], top_level=False)
    cab_mesh = b.add_node(mesh_index=m(glass, 1.0, 0, 0), name="CabMesh", top_level=False)
    end_bone = b.add_node(name="RotationEndBone", translation=(0, 1.0, 0), children=[cab_mesh], top_level=False)
    pivot = b.add_node(name="Pivot", translation=(12.0, -1.0, 0), children=[end_bone], top_level=False)
    ext3 = b.add_node(mesh_index=m(paint, 1.0, 0, 0), name="Ext3", top_level=False)
    bone03 = b.add_node(name="Bone03", translation=(2.0, 0, 0), children=[pivot, ext3, bone07], top_level=False)
    ext2 = b.add_node(mesh_index=m(paint, 0.5, 0, 0), name="Ext2", top_level=False)
    bone02 = b.add_node(name="Bone02", translation=(1.0, 0, 0), children=[bone03, ext2], top_level=False)
    ext1 = b.add_node(mesh_index=m(paint, 0.5, 0, 0), name="Ext1", top_level=False)
    bone01 = b.add_node(name="Bone01", translation=(1.5, 0, 0), children=[bone02, ext1], top_level=False)
    drum = b.add_node(mesh_index=m(paint, 0, 0, 0), name="Drum", top_level=False)
    base = b.add_node(name="Rotation_Base", translation=(0, 5.0, 0), children=[bone01, drum], top_level=False)
    pedestal = b.add_node(mesh_index=m(paint, 0, 2.5, 0), name="Pedestal", top_level=False)
    b.add_node(name="ROOT", children=[base, pedestal])
    return b.build()


def _build(td, xml=RIG_XML):
    td = Path(td)
    (td / "objects").mkdir(exist_ok=True)
    (td / "textures").mkdir(exist_ok=True)
    glb = td / "Jetway_Glass_LOD00.glb"
    glb.write_bytes(_jetway_glb())
    jw = jetway_rig.build(glb, xml, td / "objects", td / "textures", [td], "Jetway_Glass_jetway")
    text = (td / "objects" / "Jetway_Glass_jetway.obj").read_text() if jw else None
    return jw, text


def _vertices(text):
    return [tuple(float(v) for v in l.split()[1:4]) for l in text.splitlines() if l.startswith("VT ")]


class TestReadRig(unittest.TestCase):
    def test_constraints_and_chains(self):
        chains, constraints = jetway_rig.read_ik(RIG_XML)
        self.assertEqual(chains["IK_MainHandle"], ("Rotation_Base", "Pivot"))
        self.assertEqual(constraints["bone02"].axis, "x")
        self.assertEqual((constraints["bone03"].lo, constraints["bone03"].hi), (0.5, 10.0))
        self.assertTrue(jetway_rig.is_jetway_xml(RIG_XML))
        self.assertFalse(jetway_rig.is_jetway_xml("<ModelInfo><LODS/></ModelInfo>"))


class TestJetwayObject(unittest.TestCase):
    def test_shape_and_reach(self):
        with tempfile.TemporaryDirectory() as td:
            jw, _ = _build(td)
        self.assertIsNotNone(jw)
        self.assertAlmostEqual(jw.parked_length, 16.5, places=3)
        self.assertAlmostEqual(jw.shortest, 14.5, places=3)
        self.assertAlmostEqual(jw.longest, 32.5, places=3)
        self.assertEqual(jw.size, 2)  # 17-38 m covers 14.5-32.5 best
        # bones run along MSFS +X; turned like every converted model, that
        # is west of the model's north
        self.assertAlmostEqual(jw.tunnel_bearing, -90.0, places=3)
        self.assertEqual(jw.style, 1)
        self.assertEqual([p.kind for p in jw.parts], ["base", "pitch", "ext", "ext", "cab", "leg", "bogie"])

    def test_parked_tunnel_lies_along_minus_z(self):
        with tempfile.TemporaryDirectory() as td:
            _, text = _build(td)
        vs = _vertices(text)
        # the cab quad (1 m past the cab pivot) is 17-18 m out along -Z
        cab = [v for v in vs if v[2] < -16.9]
        self.assertTrue(cab)
        for x, _y, z in cab:
            self.assertAlmostEqual(x, 0.0, delta=0.51)
            self.assertLess(z, -16.9)
        # the pedestal stays at the rotunda
        self.assertTrue(any(abs(v[0]) <= 0.5 and abs(v[2]) <= 0.5 and v[1] < 3.5 for v in vs))

    def test_parts_nest_like_a_jetway(self):
        with tempfile.TemporaryDirectory() as td:
            _, text = _build(td)
        drefs = re.findall(r"jetways/(jw_\w+)", text)
        self.assertEqual(drefs[:4], ["jw_base_rotation", "jw_tunnel_pitch", "jw_tunnel_extension",
                                     "jw_tunnel_extension"])
        for d in ("jw_cabin_rotation", "jw_bogie_elevation", "jw_bogie_rotation"):
            self.assertIn(d, drefs)
        self.assertEqual(text.count("ANIM_begin"), text.count("ANIM_end"))
        # the sections share the extension by their travel: 8.5/18 and 9.5/18
        # of (38 - 16.5) m at full reach, along -Z
        keys = [l.split() for l in text.splitlines() if l.startswith("ANIM_trans_key 38.0000")]
        self.assertEqual(len(keys), 2)
        self.assertAlmostEqual(float(keys[0][4]), -21.5 * 8.5 / 18.0, places=3)
        self.assertAlmostEqual(float(keys[1][4]), -21.5 * 9.5 / 18.0, places=3)
        self.assertIn("TEXTURE ../textures/Jetway_Glass_jetway_atlas.png", text)
        self.assertIn("ATTR_blend", text)

    def test_index_count_matches(self):
        with tempfile.TemporaryDirectory() as td:
            _, text = _build(td)
        counts = re.search(r"POINT_COUNTS (\d+) 0 0 (\d+)", text)
        idx = sum(len(l.split()) - 1 for l in text.splitlines() if l.startswith(("IDX ", "IDX10 ")))
        self.assertEqual(int(counts.group(2)), idx)
        self.assertEqual(int(counts.group(1)), len(_vertices(text)))
        tris = sum(int(l.split()[2]) for l in text.splitlines() if l.startswith("TRIS "))
        self.assertEqual(tris, idx)

    def test_no_rig_no_jetway(self):
        with tempfile.TemporaryDirectory() as td:
            jw, _ = _build(td, xml="<ModelInfo/>")
        self.assertIsNone(jw)


class TestRows(unittest.TestCase):
    def _jw(self):
        return jetway_rig.Jetway(obj_name="jw", rotunda_xz=(0.0, 0.0), tunnel_bearing=-90.0, parked_length=16.5,
                                 shortest=14.5, longest=32.5, size=2, cab_offset=0.0, style=1)

    def test_row_pair(self):
        rows = jetway_rig.rows_for([(self._jw(), 47.0, 19.0, 30.0)])
        self.assertEqual(rows[0], "1500 47.00000000 19.00000000 300.0 1 2 300.0 16.50 300.0")
        self.assertEqual(rows[1], "1501 objects/jw.obj")

    def test_second_jetway_at_a_stand_takes_door_two(self):
        class Stand:
            lat, lon, heading = 47.0, 19.0, 0.0
        m_lat = 111_200.0
        jw = self._jw()
        # two rotundas west of a stand facing north, cabs parked 5 m and 15 m
        # ahead of the stand point
        a = (jw, 47.0 + 5 / m_lat, 19.0 + 16.5 / 75_900, 0.0)
        b = (jw, 47.0 + 15 / m_lat, 19.0 + 16.5 / 75_900, 0.0)
        rows = jetway_rig.rows_for([a, b], stands=[Stand()])
        sizes = [int(r.split()[5]) for r in rows if r.startswith("1500 ")]
        self.assertEqual(sizes, [2 + jetway_rig.DOOR_2, 2])


if __name__ == "__main__":
    unittest.main()
