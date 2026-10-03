"""apt_native.build_native_airport: a complete apt.dat airport from a
decoded MSFS layout."""
import math
import unittest

import airport_layout as al
import apt_native
from geo_transform import metres_per_degree


def _layout():
    lay = al.AirportLayout(ident="LHBP", name="Budapest", region="LH", lat=47.4369, lon=19.2556, alt_m=151.0)
    rw = al.Runway(lat=47.4369, lon=19.2556, alt_m=151.0, length_m=3000.0, width_m=45.0, heading_true=90.0,
                   surface=0x04, marking_flags=(1 << 6) | 1, light_flags=0b1011, pattern_flags=0,
                   primary=al.RunwayEnd(9, 1, displaced_m=100.0, approach_system=7, reil=True,
                                        vasi=[al.Vasi(8, "L", 3.0, -30.0, 350.0)]),
                   secondary=al.RunwayEnd(27, 2))
    lay.runways.append(rw)
    # Runway centreline nodes 0-1, a hold-short node 2 north of the runway,
    # a taxiway node 3 further north, and a stand (node 4).
    m_lat, _ = metres_per_degree(47.4369)
    north = lambda m: 47.4369 + m / m_lat
    lay.taxi_points = [al.TaxiPoint(47.4369, 19.2500, 1), al.TaxiPoint(47.4369, 19.2600, 1),
                       al.TaxiPoint(north(80), 19.2600, 2), al.TaxiPoint(north(400), 19.2600, 1)]
    lay.parkings = [al.Parking(north(450), 19.2620, 180.0, 30.0, 0x0A, 0x0C, 12, airlines=["MAH", "WZZ"]),
                    al.Parking(north(450), 19.2640, 0.0, 5.0, 0x0C, 0x01, 1)]
    lay.taxi_paths = [
        al.TaxiPath(0, 1, 2, "09L", 45.0),
        al.TaxiPath(1, 2, 1, "A", 20.0, centre_lit=True),
        al.TaxiPath(2, 3, 1, "A", 23.0),
        al.TaxiPath(3, 0, 3, "", 20.0),  # parking path: end 0 = the first stand
        al.TaxiPath(3, 1, 6, "SVC", 8.0),  # vehicle road
    ]
    lay.aprons = [al.Polygon([(north(420), 19.259), (north(480), 19.259), (north(480), 19.265), (north(420), 19.265)])]
    lay.signs = [al.Sign(north(70), 19.2605, 0.0, 3, "l[A]d[09L>]")]
    lay.coms = [al.Com(6, 118100, "BUDAPEST TOWER")]
    lay.windsocks = [(north(200), 19.250)]
    return lay


def _rows(prefix, rows):
    return [r for r in rows if r.split()[0] == prefix]


class TestNativeAirport(unittest.TestCase):
    def setUp(self):
        self.rows, self.report = apt_native.build_native_airport(_layout())

    def test_header_and_metadata(self):
        self.assertEqual(self.rows[0], "1 495 0 0 LHBP Budapest")
        self.assertIn("1302 icao_code LHBP", self.rows)
        self.assertIn("1302 region_code LH", self.rows)

    def test_runway_ends_from_centre_heading_length(self):
        row = _rows("100", self.rows)[0].split()
        self.assertEqual(row[2], "15", "transparent by default -- the draped MSFS pavement shows")
        self.assertEqual(row[8], "09L")
        self.assertEqual(row[17], "27R")
        _, m_lon = metres_per_degree(47.4369)
        self.assertAlmostEqual(float(row[10]), 19.2556 - 1500.0 / m_lon, places=6)
        self.assertAlmostEqual(float(row[19]), 19.2556 + 1500.0 / m_lon, places=6)
        self.assertEqual(row[11], "100.00", "displaced threshold")
        self.assertEqual(row[13], "0", "no markings on a transparent runway")
        self.assertEqual(row[14], "2", "ALSF-II")
        self.assertEqual(row[16], "1", "REIL")
        self.assertEqual((row[5], row[6]), ("1", "3"), "centre lights on, HIRL")

    def test_native_runway_surface_option(self):
        rows, _ = apt_native.build_native_airport(_layout(), runway_surface="native")
        row = _rows("100", rows)[0].split()
        self.assertEqual(row[2], "1")
        self.assertEqual(row[13], "3", "precision markings")

    def test_papi_beside_the_touchdown_zone(self):
        papi = _rows("21", self.rows)
        self.assertEqual(len(papi), 1)
        parts = papi[0].split()
        self.assertEqual((parts[3], parts[5], parts[6]), ("2", "3.00", "09L"))
        self.assertGreater(float(parts[1]), 47.4369, "left of a runway landing east is north")

    def test_network_uses_real_edges_and_marks_hot_zones(self):
        edges = _rows("1202", self.rows)
        self.assertEqual(len(edges), 4)
        self.assertTrue(any(" runway 09L/27R" in e for e in edges))
        self.assertTrue(any(e.endswith("taxiway_D A") for e in edges))
        rows = self.rows
        i = rows.index(next(e for e in edges if e.split()[1:3] == ["1", "2"]))
        self.assertEqual(rows[i + 1], "1204 departure 09L,27R")
        self.assertEqual(rows[i + 2], "1204 arrival 09L,27R")
        j = rows.index(next(e for e in edges if e.split()[1:3] == ["2", "3"]))
        self.assertFalse(rows[j + 1].startswith("1204"), "beyond the hold-short node is not hot")
        self.assertEqual(len(_rows("1206", self.rows)), 1, "the MSFS vehicle road")

    def test_ramp_starts(self):
        starts = _rows("1300", self.rows)
        self.assertEqual(len(starts), 1, "fuel stands aren't ramp starts")
        self.assertTrue(starts[0].endswith("180.00 gate heavy|jets A12"))
        self.assertEqual(_rows("1301", self.rows), ["1301 E airline mah wzz"])

    def test_signs_frequencies_windsock_boundary(self):
        self.assertEqual(_rows("20", self.rows)[0].split()[-1], "{@L}A{@Y}09L{^r}")
        self.assertEqual(_rows("1054", self.rows), ["1054 118100 BUDAPEST TOWER"])
        self.assertEqual(len(_rows("19", self.rows)), 1)
        self.assertEqual(len(_rows("130", self.rows)), 1)
        self.assertEqual(_rows("110", self.rows)[0].split()[1], "15")

    def test_taxi_lights_from_path_flags_when_there_are_no_light_strings(self):
        lit = [r for r in self.rows if r.startswith("111 ") and r.endswith(" 101")]
        self.assertEqual(len(lit), 1)

    def test_stock_flows_and_beacon_are_borrowed(self):
        stock = ["1 500 0 0 LHBP Budapest Liszt Ferenc", "1302 city Budapest", "1302 flatten 1",
                 "18 47.44 19.26 1 BCN", "1000 West", "1100 09L 11810 arrivals jets 000000 360359 Arr",
                 "1101 09L right"]
        rows, _ = apt_native.build_native_airport(_layout(), stock_block=stock)
        self.assertIn("1302 city Budapest", rows)
        self.assertNotIn("1302 flatten 1", rows)
        self.assertIn("18 47.44 19.26 1 BCN", rows)
        self.assertIn("1100 09L 11810 arrivals jets 000000 360359 Arr", rows)

    def test_flows_naming_unknown_runways_are_dropped(self):
        stock = ["1000 West", "1100 13R 11810 arrivals jets 000000 360359 Arr"]
        rows, _ = apt_native.build_native_airport(_layout(), stock_block=stock)
        self.assertFalse(_rows("1100", rows))

    def test_stock_truck_routes_reanchored_when_msfs_has_none(self):
        lay = _layout()
        lay.taxi_paths = [p for p in lay.taxi_paths if not p.is_vehicle_route]
        stock = ["1201 47.43690010 19.25000010 both s0 a", "1201 47.43690010 19.26000010 both s1 b", "1206 s0 s1 twoway"]
        rows, _ = apt_native.build_native_airport(lay, stock_block=stock)
        self.assertEqual(_rows("1206", rows), ["1206 0 1 twoway"])


class TestSigns(unittest.TestCase):
    def test_translation(self):
        self.assertEqual(apt_native.translate_sign("m[22-04]"), "{@R}22-04")
        self.assertEqual(apt_native.translate_sign("dB<"), "{@Y}B{^l}")
        self.assertIsNone(apt_native.translate_sign("l[A]#"))
        self.assertEqual(apt_native.translate_sign("A"), "{@Y}A")


class TestRunwayEnds(unittest.TestCase):
    def test_primary_is_behind_the_heading(self):
        rw = _layout().runways[0]
        (plat, plon), (slat, slon) = apt_native.runway_ends(rw)
        self.assertLess(plon, slon)
        _, m_lon = metres_per_degree(rw.lat)
        self.assertAlmostEqual((slon - plon) * m_lon, 3000.0, delta=0.5)
        self.assertTrue(math.isclose(plat, slat, abs_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
