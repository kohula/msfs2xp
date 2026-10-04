"""
Flat objects lying on a runway (covers, plates, flush fixtures) hover over
X-Plane's runway ground; runway_clutter.py picks them out, and the pipeline
drops their solid parts while keeping anything tall, lit or off the runway.
"""
import unittest

import numpy as np

import airport_layout as al
import pipeline
import runway_clutter
import terrain_fit
from geo_transform import local_offset_to_latlon
from mesh_convert import mesh_ir


def _runway():
    end = lambda n: al.RunwayEnd(number=n, designator=0)
    return al.Runway(lat=51.5, lon=0.05, alt_m=5.0, length_m=1500.0, width_m=30.0, heading_true=90.0,
                     surface=4, marking_flags=0, light_flags=0, pattern_flags=0,
                     primary=end(9), secondary=end(27))


def _box(name, sx, sy, sz, lights=()):
    pos = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (0.0, sy) for z in (-sz / 2, sz / 2)])
    return mesh_ir.MeshIR(name=name, positions=pos, lights=list(lights))


class TestRunwayClutter(unittest.TestCase):
    def setUp(self):
        layout = al.AirportLayout(ident="EGLC", name="Test", lat=51.5, lon=0.05, alt_m=5.0)
        layout.runways.append(_runway())
        self.strips = runway_clutter.runway_strips(layout)

    def _at(self, along, across):
        # runway heads east: along = east, across = south
        return local_offset_to_latlon(51.5, 0.05, 0.0, along, across)

    def test_strip_covers_runway_and_shoulders_only(self):
        self.assertTrue(runway_clutter.on_runway(*self._at(0, 0), self.strips))
        self.assertTrue(runway_clutter.on_runway(*self._at(700, 20), self.strips))  # shoulder
        self.assertFalse(runway_clutter.on_runway(*self._at(0, 40), self.strips))  # off to the side
        self.assertFalse(runway_clutter.on_runway(*self._at(900, 0), self.strips))  # past the end

    def test_only_low_small_unlit_objects_count(self):
        self.assertTrue(runway_clutter.is_low_flat([_box("cover", 2, 0.15, 1)]))
        self.assertFalse(runway_clutter.is_low_flat([_box("sign", 3, 1.2, 0.3)]))
        self.assertFalse(runway_clutter.is_low_flat([_box("plate", 40, 0.1, 40)]))
        light = mesh_ir.LightEntry(pos=(0, 0.2, 0), dir=(0, 0, 0), color=(1, 1, 1), cone_angle=360,
                                   size=1, dataref=None, named_light=None)
        self.assertFalse(runway_clutter.is_low_flat([_box("edge_light", 0.3, 0.3, 0.3), _box("l", 0, 0, 0, [light])]))

    def test_pipeline_drops_only_clutter_entries(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            for ir in (_box("cover", 2, 0.15, 1), _box("sign", 3, 1.2, 0.3), _box("decal", 2, 0.0, 2)):
                mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{ir.name}.obj"))
            terrain_fit._ir_cache.clear()
            on = self._at(100, 5)
            off = self._at(100, 80)
            cover_on = {"name": "cover"}
            cover_off = {"name": "cover"}
            sign_on = {"name": "sign"}
            decal_on = {"name": "decal"}
            cands = [
                {"abs_lat": on[0], "abs_lon": on[1], "stem_entries": {
                    "cover": (cover_on, False, False, "negligible"), "decal": (decal_on, False, True, "applied")}},
                {"abs_lat": off[0], "abs_lon": off[1], "stem_entries": {"cover": (cover_off, False, False, "x")}},
                {"abs_lat": on[0], "abs_lon": on[1], "stem_entries": {"sign": (sign_on, False, False, "x")}},
            ]
            tiles = {(51, 0): [cover_on, decal_on, cover_off, sign_on]}
            self.assertEqual(pipeline._drop_runway_clutter(cands, obj_dir, tiles, self.strips), 1)
            kept = tiles[(51, 0)]
            self.assertNotIn(cover_on, [o for o in kept if o is cover_on])
            self.assertTrue(any(o is decal_on for o in kept))
            self.assertTrue(any(o is cover_off for o in kept))
            self.assertTrue(any(o is sign_on for o in kept))
            self.assertEqual(len(kept), 3)
            terrain_fit._ir_cache.clear()


if __name__ == "__main__":
    unittest.main()
