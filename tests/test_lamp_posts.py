"""
Bare lights (a model that is only a light) placed near a lamp post are
attached to the post's lamp head; one light per head; low lights and
lights far from any lamp are left alone.
"""
import unittest

import numpy as np

import geo_transform
import lamp_posts
from mesh_convert import mesh_ir

LAT, LON = 47.0, 19.0


def _post():
    """A 20 m mast's lit sub-object: a lamp head 1 m across at the top."""
    head = [(-0.5, 20.0, -0.5), (0.5, 20.0, -0.5), (0.5, 20.3, 0.5), (-0.5, 20.3, 0.5)]
    return mesh_ir.MeshIR(name="mast_lamp", positions=np.array(head, dtype=float),
                          indices=np.array([0, 1, 2, 0, 2, 3]), texture_lit="../textures/l.png")


def _bare():
    return mesh_ir.MeshIR(name="bare_light", positions=np.zeros((0, 3)),
                          lights=[mesh_ir.LightEntry(pos=(0.0, 0.0, 0.0), dir=(0, -1, 0), color=(1, 1, 1),
                                                     cone_angle=180.0, size=30.0, dataref="x")])


def _at(east, north):
    m_lat, m_lon = geo_transform.metres_per_degree(LAT)
    return LAT + north / m_lat, LON + east / m_lon


class TestLampPosts(unittest.TestCase):
    def _tiles(self, *bare):
        tiles = {(47, 19): [{"name": "mast_lamp", "lat": LAT, "lon": LON, "hdg": 0.0, "agl": 0.0}]}
        for east, north, agl in bare:
            lat, lon = _at(east, north)
            tiles[(47, 19)].append({"name": "bare_light", "lat": lat, "lon": lon, "hdg": 0.0, "agl": agl})
        return tiles

    def _load(self, name):
        return {"mast_lamp": _post(), "bare_light": _bare()}.get(name)

    def test_light_moves_into_the_lamp(self):
        tiles = self._tiles((6.0, 3.0, 18.0))
        self.assertEqual(lamp_posts.attach_to_lamps(tiles, self._load, {"bare_light"}), (1, 0))
        h = tiles[(47, 19)][1]
        self.assertAlmostEqual(h["lat"], LAT, places=7)
        self.assertAlmostEqual(h["lon"], LON, places=7)
        self.assertAlmostEqual(h["agl"], 20.15, places=3)

    def test_one_light_per_lamp(self):
        tiles = self._tiles((6.0, 3.0, 18.0), (-4.0, 1.0, 19.0))
        self.assertEqual(lamp_posts.attach_to_lamps(tiles, self._load, {"bare_light"}), (1, 1))
        self.assertEqual(len(tiles[(47, 19)]), 2)

    def test_ground_and_far_lights_stay(self):
        tiles = self._tiles((6.0, 3.0, 1.0), (40.0, 0.0, 18.0))
        before = [dict(e) for e in tiles[(47, 19)]]
        self.assertEqual(lamp_posts.attach_to_lamps(tiles, self._load, {"bare_light"}), (0, 0))
        self.assertEqual(tiles[(47, 19)], before)

    def test_a_lamp_with_its_own_light_takes_none(self):
        tiles = self._tiles((6.0, 3.0, 18.0))
        tiles[(47, 19)].append({"name": "mast_own_lights", "lat": LAT, "lon": LON, "hdg": 0.0, "agl": 0.0})
        own = mesh_ir.MeshIR(name="mast_own_lights", positions=np.zeros((0, 3)),
                             lights=[mesh_ir.LightEntry(pos=(0.0, 20.0, 0.0), dir=(0, -1, 0), color=(1, 1, 1),
                                                        cone_angle=180.0, size=30.0, dataref="x")])
        load = lambda n: own if n == "mast_own_lights" else self._load(n)  # noqa: E731
        self.assertEqual(lamp_posts.attach_to_lamps(tiles, load, {"bare_light"}), (0, 1))


if __name__ == "__main__":
    unittest.main()
