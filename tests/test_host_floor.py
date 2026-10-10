"""
Props inside a building take the building's base level instead of the
X-Plane terrain under each of them (host_floor.py, and the pipeline pass
using it): a terminal shifted as one rigid body over uneven ground must not
leave its seats and people sunk into the floor or floating above it.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import host_floor
import pipeline
import terrain_dem
import terrain_fit
from geo_transform import local_offset_to_latlon
from mesh_convert import mesh_ir

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from dsf_fixture import build_elevation_dsf  # noqa: E402


def _quad(x0, x1, z0, z1, y):
    return [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], [0, 1, 2, 0, 2, 3]


def _ir(name, quads):
    pos, idx = [], []
    for q, i in quads:
        idx += [len(pos) + k for k in i]
        pos += q
    return mesh_ir.MeshIR(name=name, positions=np.array(pos, dtype=float),
                          normals=np.tile([0.0, 1.0, 0.0], (len(pos), 1)),
                          uvs=np.zeros((len(pos), 2)), indices=np.array(idx, dtype=np.int64))


def _l_shaped_terminal():
    """An L: a 60x20 m wing along x and a 20x60 m wing along z, each with a
    floor at y=0 and a roof at y=8. The 40x40 m corner the L wraps is
    open apron."""
    quads = []
    for (x0, x1, z0, z1) in ((-30, 30, -30, -10), (-30, -10, -10, 30)):
        quads += [_quad(x0, x1, z0, z1, 0.0), _quad(x0, x1, z0, z1, 8.0)]
    return _ir("terminal", quads)


class TestHostFloor(unittest.TestCase):
    LAT, LON, HDG = 47.5, 8.5, 30.0

    def _host(self, agl=0.0, base=100.0):
        ir = _l_shaped_terminal()
        return host_floor.Host("t", self.LAT, self.LON, self.HDG, host_floor.footprint([ir]),
                               host_floor.horizontal_cover([ir]), agl, base)

    def _at(self, x, z):
        return local_offset_to_latlon(self.LAT, self.LON, self.HDG, x, z)

    def test_floor_under_a_prop_in_the_building(self):
        host = self._host()
        self.assertTrue(host.holds(*self._at(0.0, -20.0), 0.0))
        self.assertTrue(host.holds(*self._at(-20.0, 10.0), 0.3))

    def test_open_apron_inside_the_bounding_box_is_not_the_building(self):
        host = self._host()
        self.assertFalse(host.holds(*self._at(10.0, 10.0), 0.0))

    def test_a_prop_not_at_floor_level_is_not_on_this_floor(self):
        host = self._host()
        self.assertFalse(host.holds(*self._at(0.0, -20.0), 4.0))
        # ...but one placed on the roof is
        self.assertTrue(host.holds(*self._at(0.0, -20.0), 8.0))

    def test_floor_level_counts_the_buildings_own_height(self):
        host = self._host(agl=2.0)  # the building stands 2 m up: its floor is at 2 m
        self.assertTrue(host.holds(*self._at(0.0, -20.0), 2.0))
        self.assertFalse(host.holds(*self._at(0.0, -20.0), 0.0))

    def test_index_picks_the_building_and_skips_big_neighbours(self):
        index = host_floor.HostIndex([self._host()])
        lat, lon = self._at(0.0, -20.0)
        self.assertIsNotNone(index.host_for(lat, lon, 0.0, area=2.0))
        self.assertIsNone(index.host_for(lat, lon, 0.0, area=1500.0))
        self.assertIsNone(index.host_for(*self._at(500.0, 500.0), 0.0, area=2.0))

    def test_a_flat_ground_sheet_is_not_a_building(self):
        """A 400 m ground-cover sheet is long enough but not tall enough:
        treated as a building floor, it lifted the signs and barriers on
        it to its own (hovering) height."""
        sheet = _ir("sheet", [_quad(-200, 200, -150, 150, 0.0), _quad(-200, 200, -150, 150, 0.2)])
        self.assertFalse(host_floor.is_host_size(host_floor.footprint([sheet]), host_floor.height_of([sheet])))
        self.assertTrue(host_floor.is_host_size(host_floor.footprint([_l_shaped_terminal()]),
                                                host_floor.height_of([_l_shaped_terminal()])))

    def test_small_things_are_not_hosts(self):
        self.assertFalse(host_floor.is_host_size((-2, 2, -2, 2)))
        self.assertTrue(host_floor.is_host_size((-15, 15, -2, 2)))


class TestPropsOnHostFloors(unittest.TestCase):
    """The pipeline pass over real terrain: X-Plane ground rising 1 m per
    ~15 m eastward under a terminal; a chair 20 m east of the terminal's
    anchor sits on ground ~1.3 m higher, so on its own it would float
    that much above the floor."""

    def tearDown(self):
        terrain_fit._ir_cache.clear()
        terrain_dem._dem_cache.clear()

    def _sloped(self, root):
        grid = [[col * 1000 for col in range(5)] for _ in range(5)]  # ~1000 m per post across ~28 km
        dsf_dir = root / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / "+40+000"
        dsf_dir.mkdir(parents=True, exist_ok=True)
        (dsf_dir / "+47+008.dsf").write_bytes(build_elevation_dsf(grid))

    def test_chair_follows_the_terminal_floor(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xp = td / "XPlane"
            self._sloped(xp)
            obj_dir = td / "objects"
            obj_dir.mkdir()
            mesh_ir.save(_l_shaped_terminal(), mesh_ir.sidecar_path_for(obj_dir / "terminal.obj"))
            mesh_ir.save(_ir("chair", [_quad(-0.3, 0.3, -0.3, 0.3, 0.0), _quad(-0.3, 0.3, -0.3, -0.2, 0.9)]),
                         mesh_ir.sidecar_path_for(obj_dir / "chair.obj"))

            t_lat, t_lon = 47.5, 8.5
            c_lat, c_lon = local_offset_to_latlon(t_lat, t_lon, 0.0, 20.0, -20.0)
            ground_t = terrain_dem.get_elevation(xp, t_lat, t_lon)
            ground_c = terrain_dem.get_elevation(xp, c_lat, c_lon)
            self.assertIsNotNone(ground_t)

            terminal_entry = {"name": "terminal", "agl": 0.0}
            chair_entry = {"name": "chair_tfit_abc", "agl": 0.0}
            cands = [
                {"group_key": (("terminal",), t_lat, t_lon, 0.0, False), "abs_lat": t_lat, "abs_lon": t_lon,
                 "hdg": 0.0, "agl": 0.0, "height_offset": 0.0,
                 "stem_entries": {"terminal": (terminal_entry, False, False, "negligible")}},
                {"group_key": (("chair",), c_lat, c_lon, 0.0, False), "abs_lat": c_lat, "abs_lon": c_lon,
                 "hdg": 0.0, "agl": 0.0, "height_offset": 0.0,
                 "stem_entries": {"chair": (chair_entry, True, False, "applied_rigid_warp")}},
            ]
            hosts, moved = pipeline._place_props_on_host_floors(cands, obj_dir, xp)
            self.assertEqual((hosts, moved), (1, 1))
            # its own terrain fit is dropped, and its height puts it on the
            # terminal's base level: ground at the chair + agl = ground at the terminal
            self.assertEqual(chair_entry["name"], "chair")
            self.assertAlmostEqual(ground_c + chair_entry["agl"], ground_t, places=6)
            self.assertNotAlmostEqual(chair_entry["agl"], 0.0, places=2)
            self.assertEqual(terminal_entry, {"name": "terminal", "agl": 0.0})

    def test_terminal_shift_is_carried_to_its_props(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xp = td / "XPlane"
            self._sloped(xp)
            obj_dir = td / "objects"
            obj_dir.mkdir()
            mesh_ir.save(_l_shaped_terminal(), mesh_ir.sidecar_path_for(obj_dir / "terminal.obj"))
            mesh_ir.save(_ir("person", [_quad(-0.2, 0.2, -0.2, 0.2, 0.0)]),
                         mesh_ir.sidecar_path_for(obj_dir / "person.obj"))
            t_lat, t_lon = 47.5, 8.5
            p_lat, p_lon = local_offset_to_latlon(t_lat, t_lon, 0.0, -20.0, 10.0)
            gk = (("terminal",), t_lat, t_lon, 0.0, False)
            ground_t = terrain_dem.get_elevation(xp, t_lat, t_lon)
            # the terminal's terrain-fitted copy: shifted down 0.4 m
            shifted = _l_shaped_terminal()
            shifted.positions = shifted.positions + np.array([0.0, -0.4, 0.0])
            mesh_ir.save(shifted, mesh_ir.sidecar_path_for(obj_dir / "terminal_tfit_x.obj"))
            terrain_fit._group_transform_cache[gk] = {"vertical_shift": -0.4, "origin_elev": ground_t}
            try:
                person = {"name": "person", "agl": 0.0}
                cands = [
                    {"group_key": gk, "abs_lat": t_lat, "abs_lon": t_lon, "hdg": 0.0, "agl": 0.0,
                     "height_offset": 0.0,
                     "stem_entries": {"terminal": ({"name": "terminal_tfit_x", "agl": 0.0}, True, False,
                                                   "applied_vertical_shift")}},
                    {"group_key": (("person",), p_lat, p_lon, 0.0, False), "abs_lat": p_lat, "abs_lon": p_lon,
                     "hdg": 0.0, "agl": 0.0, "height_offset": 0.0,
                     "stem_entries": {"person": (person, False, False, "negligible")}},
                ]
                self.assertEqual(pipeline._place_props_on_host_floors(cands, obj_dir, xp), (1, 1))
                ground_p = terrain_dem.get_elevation(xp, p_lat, p_lon)
                self.assertAlmostEqual(ground_p + person["agl"], ground_t - 0.4, places=6)
            finally:
                terrain_fit._group_transform_cache.pop(gk, None)

    def test_props_follow_a_warped_building_floor(self):
        """A building warped onto the terrain: its floor under the prop is
        where the fitted copy has it, not the level at its anchor."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xp = td / "XPlane"
            self._sloped(xp)
            obj_dir = td / "objects"
            obj_dir.mkdir()
            mesh_ir.save(_l_shaped_terminal(), mesh_ir.sidecar_path_for(obj_dir / "terminal.obj"))
            warped = _l_shaped_terminal()
            warped.positions = warped.positions.copy()
            warped.positions[:, 1] += 0.05 * warped.positions[:, 0]  # floor rises 5 cm per metre east
            mesh_ir.save(warped, mesh_ir.sidecar_path_for(obj_dir / "terminal_tfit_w.obj"))
            mesh_ir.save(_ir("seat", [_quad(-0.3, 0.3, -0.3, 0.3, 0.0)]), mesh_ir.sidecar_path_for(obj_dir / "seat.obj"))
            t_lat, t_lon = 47.5, 8.5
            p_lat, p_lon = local_offset_to_latlon(t_lat, t_lon, 0.0, 20.0, -20.0)
            seat = {"name": "seat_tfit_q", "agl": 0.0}
            cands = [
                {"group_key": (("terminal",), t_lat, t_lon, 0.0, False), "abs_lat": t_lat, "abs_lon": t_lon,
                 "hdg": 0.0, "agl": 0.0, "height_offset": 0.0,
                 "stem_entries": {"terminal": ({"name": "terminal_tfit_w", "agl": 0.0}, True, False,
                                               "applied_rigid_warp")}},
                {"group_key": (("seat",), p_lat, p_lon, 0.0, False), "abs_lat": p_lat, "abs_lon": p_lon,
                 "hdg": 0.0, "agl": 0.0, "height_offset": 0.0,
                 "stem_entries": {"seat": (seat, True, False, "applied_rigid_warp")}},
            ]
            self.assertEqual(pipeline._place_props_on_host_floors(cands, obj_dir, xp), (1, 1))
            ground_t = terrain_dem.get_elevation(xp, t_lat, t_lon)
            ground_p = terrain_dem.get_elevation(xp, p_lat, p_lon)
            # the warped floor 20 m east of the anchor is 1 m up
            self.assertAlmostEqual(ground_p + seat["agl"], ground_t + 1.0, places=4)
            self.assertEqual(seat["name"], "seat")

    def test_props_on_the_open_apron_keep_their_own_ground(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xp = td / "XPlane"
            self._sloped(xp)
            obj_dir = td / "objects"
            obj_dir.mkdir()
            mesh_ir.save(_l_shaped_terminal(), mesh_ir.sidecar_path_for(obj_dir / "terminal.obj"))
            mesh_ir.save(_ir("tug", [_quad(-1, 1, -2, 2, 0.0)]), mesh_ir.sidecar_path_for(obj_dir / "tug.obj"))
            t_lat, t_lon = 47.5, 8.5
            g_lat, g_lon = local_offset_to_latlon(t_lat, t_lon, 0.0, 10.0, 10.0)  # the open corner of the L
            tug = {"name": "tug_tfit_y", "agl": 0.0}
            cands = [
                {"group_key": (("terminal",), t_lat, t_lon, 0.0, False), "abs_lat": t_lat, "abs_lon": t_lon,
                 "hdg": 0.0, "agl": 0.0, "height_offset": 0.0,
                 "stem_entries": {"terminal": ({"name": "terminal", "agl": 0.0}, False, False, "negligible")}},
                {"group_key": (("tug",), g_lat, g_lon, 0.0, False), "abs_lat": g_lat, "abs_lon": g_lon,
                 "hdg": 0.0, "agl": 0.0, "height_offset": 0.0,
                 "stem_entries": {"tug": (tug, True, False, "applied_rigid_warp")}},
            ]
            self.assertEqual(pipeline._place_props_on_host_floors(cands, obj_dir, xp), (0, 0))
            self.assertEqual(tug, {"name": "tug_tfit_y", "agl": 0.0})


if __name__ == "__main__":
    unittest.main()
