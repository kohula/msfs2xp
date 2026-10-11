"""
Objects are fitted onto X-Plane's terrain through one shared ground grid
(ground_grid.py, terrain_fit.py): two objects at the same place get the
same height whatever they belong to, so a seat stays on its floor and a
deck on its pillars on uneven ground.
"""
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import ground_grid
import terrain_dem
import terrain_fit
from geo_transform import local_offset_to_latlon
from mesh_convert import mesh_ir

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from dsf_fixture import build_elevation_dsf, build_mesh_dsf  # noqa: E402

LAT, LON = 47.5, 8.5


def _ridge_dsf(height=1.0, half_width_deg=0.0005):
    """X-Plane terrain at 100 m with a ridge `height` m high along latitude
    LAT, falling to 100 m within half_width_deg north and south of it."""
    rows = [(47.0, 100.0), (LAT - half_width_deg, 100.0), (LAT, 100.0 + height),
            (LAT + half_width_deg, 100.0), (48.0, 100.0)]
    pts = [(lon, lat, z) for lat, z in rows for lon in (8.0, 9.0)]
    tris = []
    for r in range(len(rows) - 1):
        a, b, c, d = 2 * r, 2 * r + 1, 2 * r + 3, 2 * r + 2
        tris += [a, b, c, a, c, d]
    cmds = bytes([18, 1]) + struct.pack("<ff", 0.0, -1.0)
    cmds += bytes([23, len(tris)]) + struct.pack(f"<{len(tris)}H", *tris)
    return build_mesh_dsf(pts, cmds, elev_range=(90.0, 40.0))


def _install(root, dsf_bytes):
    d = root / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / "+40+000"
    d.mkdir(parents=True, exist_ok=True)
    (d / "+47+008.dsf").write_bytes(dsf_bytes)
    return root


def _quad_ir(name, half_x, half_z, y=0.0, draped=False, walls=0.0):
    """A floor (two triangles) half_x by half_z around the origin, with
    optional walls `walls` m high along its south and north edges."""
    pos = [(-half_x, y, -half_z), (half_x, y, -half_z), (half_x, y, half_z), (-half_x, y, half_z)]
    idx = [0, 2, 1, 0, 3, 2]
    if walls:
        for z in (-half_z, half_z):
            k = len(pos)
            pos += [(-half_x, y, z), (half_x, y, z), (half_x, y + walls, z), (-half_x, y + walls, z)]
            idx += [k, k + 1, k + 2, k, k + 2, k + 3]
    pos = np.array(pos, dtype=np.float64)
    return mesh_ir.MeshIR(name=name, positions=pos, normals=np.tile([0.0, 1.0, 0.0], (len(pos), 1)),
                          uvs=np.zeros((len(pos), 2)), indices=np.array(idx, dtype=np.int64), draped=draped)


def _save(obj_dir, ir):
    mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{ir.name}.obj"))
    (obj_dir / f"{ir.name}.obj").write_text("I\n800\nOBJ\n")


def _surface_y(ir, x, z):
    """Height of the upward-facing surface of ir under (x, z)."""
    best = None
    for a, b, c in ir.indices.reshape(-1, 3):
        pa, pb, pc = ir.positions[a], ir.positions[b], ir.positions[c]
        m = np.array([[pb[0] - pa[0], pc[0] - pa[0]], [pb[2] - pa[2], pc[2] - pa[2]]])
        if abs(np.linalg.det(m)) < 1e-9:
            continue
        u, v = np.linalg.solve(m, [x - pa[0], z - pa[2]])
        if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
            y = pa[1] + u * (pb[1] - pa[1]) + v * (pc[1] - pa[1])
            best = y if best is None else min(best, y)
    return best


class _Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.td = Path(self._td.name)
        self.obj_dir = self.td / "objects"
        self.obj_dir.mkdir()

    def tearDown(self):
        terrain_fit._ir_cache.clear()
        terrain_fit._group_cache.clear()
        terrain_fit._lift_cache.clear()
        terrain_dem._dem_cache.clear()
        terrain_dem._mesh_cache.clear()
        self._td.cleanup()

    def _grid(self, xp, half=120.0):
        a0, o0 = local_offset_to_latlon(LAT, LON, 0.0, -half, half)
        a1, o1 = local_offset_to_latlon(LAT, LON, 0.0, half, -half)
        grid = ground_grid.Grid.around(LAT, LON)
        grid.cover(a0, a1, o0, o1, lambda a, o: terrain_dem.get_elevation(xp, a, o))
        return grid

    def _placed(self, xp, stem, lat, lon, grid=None, hdg=0.0):
        """(fitted MeshIR, height of its origin in X-Plane, result tuple)."""
        res = terrain_fit.get_or_create_fitted_group(self.obj_dir, [stem], lat, lon, hdg, xp, grid=grid)[stem]
        lift = terrain_fit.group_lift(terrain_fit.group_key([stem], lat, lon, hdg))
        ir = mesh_ir.load(mesh_ir.sidecar_path_for(self.obj_dir / f"{res[0]}.obj"))
        return ir, terrain_dem.get_elevation(xp, lat, lon) + lift, res


class TestGrid(unittest.TestCase):
    def _plane_grid(self):
        grid = ground_grid.Grid.around(LAT, LON)
        grid.cover(LAT - 0.001, LAT + 0.001, LON - 0.001, LON + 0.001, lambda a, o: 1000 * (a - LAT) + 50.0)
        return grid

    def test_bilinear_reproduces_a_plane(self):
        grid = self._plane_grid()
        for da in (-0.0007, 0.0, 0.00031):
            self.assertAlmostEqual(grid.height(LAT + da, LON + 0.0002), 1000 * da + 50.0, places=6)
        self.assertIsNone(grid.height(LAT + 1.0, LON))

    def test_a_post_without_terrain_drops_out(self):
        grid = ground_grid.Grid.around(LAT, LON)
        grid.cover(LAT - 0.001, LAT + 0.001, LON - 0.001, LON + 0.001, lambda a, o: None if a > LAT else 20.0)
        self.assertAlmostEqual(grid.height(LAT, LON), 20.0, places=6)

    def test_level_sets_the_posts_inside_once(self):
        grid = self._plane_grid()
        big = [local_offset_to_latlon(LAT, LON, 0.0, x, z) for x, z in ((-30, -30), (30, -30), (30, 30), (-30, 30))]
        small = [local_offset_to_latlon(LAT, LON, 0.0, x, z) for x, z in ((-5, -5), (5, -5), (5, 5), (-5, 5))]
        self.assertGreater(grid.level(big, 77.0), 0)
        self.assertEqual(grid.level(small, 12.0), 0)  # the larger building came first
        self.assertAlmostEqual(grid.height(LAT, LON), 77.0, places=6)
        far_lat, far_lon = local_offset_to_latlon(LAT, LON, 0.0, 0.0, 60.0)
        self.assertAlmostEqual(grid.height(far_lat, far_lon), 1000 * (far_lat - LAT) + 50.0, places=6)

    def test_save_and_load(self):
        grid = self._plane_grid()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "g.npz"
            grid.save(path)
            back = ground_grid.Grid.load(path)
        self.assertEqual(sorted(back.blocks), sorted(grid.blocks))
        self.assertAlmostEqual(back.height(LAT + 0.0004, LON), grid.height(LAT + 0.0004, LON), places=9)


class TestRefine(unittest.TestCase):
    @staticmethod
    def bowl(x, z):
        return 0.0004 * (np.asarray(x) ** 2 + np.asarray(z) ** 2)

    def test_floor_follows_a_bending_field(self):
        ir = _quad_ir("floor", 50, 50, walls=5.0)
        pos, nrm, uv, idx = terrain_fit.refine(ir.positions, ir.normals, ir.uvs, ir.indices, self.bowl)
        self.assertGreater(len(idx), len(ir.indices))
        fitted = mesh_ir.MeshIR(name="f", positions=pos.copy(), indices=np.asarray(idx))
        fitted.positions[:, 1] += self.bowl(pos[:, 0], pos[:, 2])
        rng = np.random.default_rng(1)
        for x, z in rng.uniform(-48, 48, (40, 2)):
            self.assertAlmostEqual(_surface_y(fitted, x, z), float(self.bowl(x, z)), delta=0.025)
        # floor triangles keep facing up
        for a, b, c in np.asarray(idx).reshape(-1, 3):
            pa, pb, pc = pos[a], pos[b], pos[c]
            if max(abs(pa[1]), abs(pb[1]), abs(pc[1])) < 1e-9:
                self.assertGreater(np.cross(pb - pa, pc - pa)[1], 0.0)
        # wall tops sit over the same split points as their bottoms
        tops = {(round(p[0], 6), round(p[2], 6)) for p in pos if abs(p[1] - 5.0) < 1e-9}
        bottoms = {(round(p[0], 6), round(p[2], 6)) for p in pos if abs(p[1]) < 1e-9 and abs(abs(p[2]) - 50) < 1e-9}
        self.assertEqual(tops, bottoms)

    def test_both_sides_of_a_uv_seam_split_alike(self):
        # two triangles sharing an edge geometrically but not by index
        pos = np.array([[0, 0, 0], [60, 0, 0], [0, 0, 60], [60, 0, 0], [0, 0, 60], [60, 0, 60]], dtype=np.float64)
        idx = np.array([0, 2, 1, 3, 4, 5])
        out, _, _, _ = terrain_fit.refine(pos, np.zeros((0, 3)), np.zeros((0, 2)), idx, self.bowl)
        on_seam = {(round(p[0], 6), round(p[2], 6)) for p in out if abs(p[0] + p[2] - 60) < 1e-6}
        # every seam point appears twice (once per side) or is an original corner
        counts = {}
        for p in out:
            if abs(p[0] + p[2] - 60) < 1e-6:
                counts[(round(p[0], 6), round(p[2], 6))] = counts.get((round(p[0], 6), round(p[2], 6)), 0) + 1
        self.assertGreater(len(on_seam), 2)
        self.assertTrue(all(n == 2 for n in counts.values()))

    def test_a_flat_field_changes_nothing(self):
        ir = _quad_ir("floor", 50, 50)
        pos, _, _, idx = terrain_fit.refine(ir.positions, ir.normals, ir.uvs, ir.indices,
                                            lambda x, z: 0.01 * np.asarray(x))
        self.assertEqual(len(pos), len(ir.positions))
        self.assertEqual(len(idx), len(ir.indices))


class TestFit(_Base):
    def test_seat_stays_on_the_floor_over_a_ridge(self):
        xp = _install(self.td / "XP", _ridge_dsf(height=1.0))
        _save(self.obj_dir, _quad_ir("floor", 50, 50, walls=4.0))
        _save(self.obj_dir, _quad_ir("seat", 0.3, 0.3, y=0.45))
        seat_lat, seat_lon = local_offset_to_latlon(LAT, LON, 0.0, 10.0, -20.0)
        for grid in (self._grid(xp), None):
            terrain_fit._group_cache.clear()
            floor, floor_base, res = self._placed(xp, "floor", LAT, LON, grid)
            self.assertEqual(res[2], "applied")
            seat, seat_base, _ = self._placed(xp, "seat", seat_lat, seat_lon, grid)
            floor_under = floor_base + _surface_y(floor, 10.0, -20.0)
            seat_bottom = seat_base + float(seat.positions[:, 1].min())
            self.assertAlmostEqual(seat_bottom - floor_under, 0.45, delta=0.03)

    def test_pillar_meets_the_deck_and_the_ground(self):
        xp = _install(self.td / "XP", _ridge_dsf(height=1.0))
        _save(self.obj_dir, _quad_ir("deck", 40, 40, y=6.0))
        pillar = mesh_ir.MeshIR(name="pillar", positions=np.array(
            [[-0.3, 0, 0], [0.3, 0, 0], [0.3, 6, 0], [-0.3, 6, 0]], dtype=np.float64),
            indices=np.array([0, 1, 2, 0, 2, 3]))
        _save(self.obj_dir, pillar)
        p_lat, p_lon = local_offset_to_latlon(LAT, LON, 0.0, -15.0, 25.0)
        grid = self._grid(xp)
        deck, deck_base, _ = self._placed(xp, "deck", LAT, LON, grid)
        col, col_base, _ = self._placed(xp, "pillar", p_lat, p_lon, grid)
        self.assertAlmostEqual(col_base + 6.0, deck_base + _surface_y(deck, -15.0, 25.0), delta=0.03)
        self.assertAlmostEqual(col_base, terrain_dem.get_elevation(xp, p_lat, p_lon), delta=0.03)

    def test_level_ground_needs_no_copy(self):
        xp = _install(self.td / "XP", build_elevation_dsf([[40, 40], [40, 40]]))
        _save(self.obj_dir, _quad_ir("floor", 50, 50, walls=4.0))
        res = terrain_fit.get_or_create_fitted_group(self.obj_dir, ["floor"], LAT, LON, 0.0, xp)
        self.assertEqual(res["floor"], ("floor", False, "negligible"))
        self.assertEqual(terrain_fit.group_lift(terrain_fit.group_key(["floor"], LAT, LON, 0.0)), 0.0)

    def test_building_on_steep_ground_is_level_with_a_skirt(self):
        xp = _install(self.td / "XP", _ridge_dsf(height=20.0))
        lat, lon = local_offset_to_latlon(LAT, LON, 0.0, 0.0, -20.0)  # on the ridge's north flank
        _save(self.obj_dir, _quad_ir("hall", 15, 15, walls=6.0))
        hall, base, res = self._placed(xp, "hall", lat, lon)
        self.assertEqual(res[2], "applied_skirt")
        original = mesh_ir.load(mesh_ir.sidecar_path_for(self.obj_dir / "hall.obj"))
        # the building itself is only translated: its roof line is level
        np.testing.assert_allclose(hall.positions[:len(original.positions)], original.positions, atol=1e-9)
        # ...standing on the highest ground under it, with a skirt down the slope
        corners = [local_offset_to_latlon(lat, lon, 0.0, x, z) for x in (-15, 15) for z in (-15, 15)]
        ground = [terrain_dem.get_elevation(xp, a, o) for a, o in corners]
        self.assertAlmostEqual(base, max(ground), delta=0.3)
        self.assertGreater(len(hall.positions), len(original.positions))
        self.assertLess(base + float(hall.positions[:, 1].min()), min(ground) + 0.3)

    def test_gentle_slope_follows_the_ground(self):
        xp = _install(self.td / "XP", _ridge_dsf(height=1.0))
        _save(self.obj_dir, _quad_ir("hall", 40, 40, walls=6.0))
        hall, base, res = self._placed(xp, "hall", LAT, LON)
        self.assertEqual(res[2], "applied")
        for x, z in ((-30, -30), (20, 35), (0, 0)):
            a, o = local_offset_to_latlon(LAT, LON, 0.0, x, z)
            self.assertAlmostEqual(base + _surface_y(hall, x, z), terrain_dem.get_elevation(xp, a, o), delta=0.03)

    def test_draped_lights_and_polygon_mode(self):
        xp = _install(self.td / "XP", _ridge_dsf(height=1.0))
        _save(self.obj_dir, _quad_ir("decal", 30, 30, draped=True))
        lights = mesh_ir.MeshIR(name="lamps", lights=[
            mesh_ir.LightEntry(pos=(0.0, 5.0, z), dir=(0, -1, 0), color=(1, 1, 1), cone_angle=0.0, size=1.0,
                               dataref="none") for z in (-30.0, 0.0)])
        _save(self.obj_dir, lights)
        res = terrain_fit.get_or_create_fitted_group(self.obj_dir, ["decal", "lamps"], LAT, LON, 0.0, xp)
        self.assertTrue(res["decal"][1] and res["lamps"][1])
        fitted = mesh_ir.load(mesh_ir.sidecar_path_for(self.obj_dir / f"{res['lamps'][0]}.obj"))
        drop = fitted.lights[0].pos[1] - fitted.lights[1].pos[1]
        a, o = local_offset_to_latlon(LAT, LON, 0.0, 0.0, -30.0)
        # (the grid's 4 m posts round the ridge's sharp crest by a few cm)
        self.assertAlmostEqual(drop, terrain_dem.get_elevation(xp, a, o) - terrain_dem.get_elevation(xp, LAT, LON),
                               delta=0.05)
        terrain_fit._group_cache.clear()
        res = terrain_fit.get_or_create_fitted_group(self.obj_dir, ["decal"], LAT, LON, 0.0, xp,
                                                     skip_draped_positions=True)
        self.assertEqual(res["decal"][:2], ("decal", False))

    def test_no_install(self):
        _save(self.obj_dir, _quad_ir("floor", 5, 5))
        res = terrain_fit.get_or_create_fitted_group(self.obj_dir, ["floor"], LAT, LON, 0.0, None)
        self.assertEqual(res["floor"], ("floor", False, "no_xplane_root"))

    def test_turned_placement(self):
        xp = _install(self.td / "XP", _ridge_dsf(height=1.0))
        _save(self.obj_dir, _quad_ir("floor", 50, 10, walls=3.0))
        hall, base, _ = self._placed(xp, "floor", LAT, LON, hdg=90.0)
        # local +x points south after a 90 degree turn: 40 m south of the ridge
        a, o = local_offset_to_latlon(LAT, LON, 90.0, 40.0, 0.0)
        self.assertLess(a, LAT)
        self.assertAlmostEqual(base + _surface_y(hall, 40.0, 0.0), terrain_dem.get_elevation(xp, a, o), delta=0.03)
        self.assertTrue(math.isfinite(base))


if __name__ == "__main__":
    unittest.main()
