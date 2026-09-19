"""
draped_ranking.py: pure functions, split out of mesh_convert/convert.py.
Confirms the exact collision this exists to fix (two layers with genuinely
different footprints landing in the same fixed-threshold bucket) and the
rank-based fix's properties (distinct offsets, largest-drawn-first
ordering, graceful >11-layer fallback).
"""
import unittest

from mesh_convert import draped_ranking


class TestDrapedRanking(unittest.TestCase):
    def test_fixed_threshold_scheme_can_collide(self):
        """This IS the bug rank_draped_layer_offsets exists to fix -- both
        of these genuinely different footprints land in the same
        '>=1000 -> -3' bucket under the old fixed-threshold scheme."""
        a = draped_ranking.draped_layer_offset(5000.0)
        b = draped_ranking.draped_layer_offset(3000.0)
        self.assertEqual(a, b)

    def test_rank_based_offsets_are_all_distinct(self):
        areas = {"apron_fill": 5000.0, "taxiway_strip": 3000.0, "stripe": 40.0, "text": 2.0}
        ranked = draped_ranking.rank_draped_layer_offsets(areas)
        self.assertEqual(len(set(ranked.values())), len(ranked))

    def test_rank_based_offsets_preserve_largest_first_ordering(self):
        areas = {"apron_fill": 5000.0, "taxiway_strip": 3000.0, "stripe": 40.0, "text": 2.0}
        ranked = draped_ranking.rank_draped_layer_offsets(areas)
        self.assertLess(ranked["apron_fill"], ranked["taxiway_strip"])
        self.assertLess(ranked["taxiway_strip"], ranked["stripe"])
        self.assertLess(ranked["stripe"], ranked["text"])

    def test_rank_based_offsets_stay_within_obj8_range(self):
        areas = {f"layer{i}": float(i + 1) for i in range(11)}
        ranked = draped_ranking.rank_draped_layer_offsets(areas)
        self.assertTrue(all(-5 <= v <= 5 for v in ranked.values()))

    def test_more_than_11_layers_falls_back_gracefully(self):
        areas = {f"layer{i}": float(i + 1) * 10 for i in range(15)}
        ranked = draped_ranking.rank_draped_layer_offsets(areas)
        self.assertEqual(len(ranked), 15)  # no crash, every layer still gets SOME offset

    def test_empty_input(self):
        self.assertEqual(draped_ranking.rank_draped_layer_offsets({}), {})


if __name__ == "__main__":
    unittest.main()
